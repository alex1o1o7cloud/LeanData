import Mathlib

namespace total_distance_is_176_l731_73189

/-- Represents a series of linked rings with specific properties -/
structure LinkedRings where
  thickness : ℝ
  topOutsideDiameter : ℝ
  smallestOutsideDiameter : ℝ
  diameterDecrease : ℝ

/-- Calculates the total vertical distance covered by the linked rings -/
def totalVerticalDistance (rings : LinkedRings) : ℝ :=
  sorry

/-- Theorem stating that the total vertical distance is 176 cm for the given conditions -/
theorem total_distance_is_176 (rings : LinkedRings) 
  (h1 : rings.thickness = 2)
  (h2 : rings.topOutsideDiameter = 30)
  (h3 : rings.smallestOutsideDiameter = 10)
  (h4 : rings.diameterDecrease = 2) :
  totalVerticalDistance rings = 176 :=
sorry

end total_distance_is_176_l731_73189


namespace C_nec_not_suff_A_l731_73182

-- Define the propositions
variable (A B C : Prop)

-- Define the relationships between A, B, and C
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_nec_and_suff_C : (B ↔ C)

-- State the theorem to be proved
theorem C_nec_not_suff_A : (C → A) ∧ ¬(A → C) := by sorry

end C_nec_not_suff_A_l731_73182


namespace grass_seed_cost_l731_73120

/-- The cost of a 5-pound bag of grass seed -/
def cost_5lb : ℝ := 13.80

/-- The cost of a 10-pound bag of grass seed -/
def cost_10lb : ℝ := 20.43

/-- The cost of a 25-pound bag of grass seed -/
def cost_25lb : ℝ := 32.25

/-- The minimum amount of grass seed the customer must buy (in pounds) -/
def min_amount : ℝ := 65

/-- The maximum amount of grass seed the customer can buy (in pounds) -/
def max_amount : ℝ := 80

/-- The least possible cost for the customer -/
def least_cost : ℝ := 98.73

theorem grass_seed_cost : 
  2 * cost_25lb + cost_10lb + cost_5lb = least_cost ∧ 
  2 * 25 + 10 + 5 ≥ min_amount ∧
  2 * 25 + 10 + 5 ≤ max_amount := by
  sorry

end grass_seed_cost_l731_73120


namespace binary_1010101_conversion_l731_73103

/-- Converts a binary number represented as a list of bits to its decimal equivalent. -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to its octal representation as a list of digits. -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

/-- The binary representation of 1010101₂ -/
def binary_1010101 : List Bool := [true, false, true, false, true, false, true]

theorem binary_1010101_conversion :
  (binary_to_decimal binary_1010101 = 85) ∧
  (decimal_to_octal 85 = [5, 2, 1]) := by
sorry

end binary_1010101_conversion_l731_73103


namespace expression_evaluation_l731_73129

theorem expression_evaluation : 
  let x : ℚ := -2
  let y : ℚ := 1/2
  ((x + 2*y)^2 - (x + y)*(3*x - y) - 5*y^2) / (2*x) = 5/2 := by
  sorry

end expression_evaluation_l731_73129


namespace inequality_iff_solution_set_l731_73169

-- Define the inequality function
def inequality (x : ℝ) : Prop :=
  Real.log (1 + 27 * x^5) / Real.log (1 + x^2) +
  Real.log (1 + x^2) / Real.log (1 - 2*x^2 + 27*x^4) ≤
  1 + Real.log (1 + 27*x^5) / Real.log (1 - 2*x^2 + 27*x^4)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  (x > -Real.rpow 27 (-1/5) ∧ x ≤ -1/3) ∨
  (x > -Real.sqrt (2/27) ∧ x < 0) ∨
  (x > 0 ∧ x < Real.sqrt (2/27)) ∨
  x = 1/3

-- State the theorem
theorem inequality_iff_solution_set :
  ∀ x : ℝ, inequality x ↔ solution_set x :=
sorry

end inequality_iff_solution_set_l731_73169


namespace reflect_point_5_neg3_l731_73188

/-- Given a point P in the Cartesian coordinate system, 
    this function returns its coordinates with respect to the y-axis. -/
def reflect_y_axis (x y : ℝ) : ℝ × ℝ := (-x, y)

/-- The coordinates of P(5,-3) with respect to the y-axis are (-5,-3). -/
theorem reflect_point_5_neg3 : 
  reflect_y_axis 5 (-3) = (-5, -3) := by sorry

end reflect_point_5_neg3_l731_73188


namespace fourth_intersection_point_l731_73158

/-- The curve xy = 2 intersects a circle at four points. Three of these points are given. -/
def intersection_points : Finset (ℚ × ℚ) :=
  {(4, 1/2), (-2, -1), (2/5, 5)}

/-- The fourth intersection point -/
def fourth_point : ℚ × ℚ := (-16/5, -5/8)

/-- All points satisfy the equation xy = 2 -/
def on_curve (p : ℚ × ℚ) : Prop :=
  p.1 * p.2 = 2

theorem fourth_intersection_point :
  (∀ p ∈ intersection_points, on_curve p) →
  on_curve fourth_point →
  ∃ (a b r : ℚ),
    (∀ p ∈ intersection_points, (p.1 - a)^2 + (p.2 - b)^2 = r^2) ∧
    (fourth_point.1 - a)^2 + (fourth_point.2 - b)^2 = r^2 :=
by sorry

end fourth_intersection_point_l731_73158


namespace candy_bars_total_l731_73125

theorem candy_bars_total (people : Float) (bars_per_person : Float) : 
  people = 3.0 → 
  bars_per_person = 1.66666666699999 → 
  people * bars_per_person = 5.0 := by
  sorry

end candy_bars_total_l731_73125


namespace shaded_area_formula_l731_73178

/-- An equilateral triangle inscribed in a circle -/
structure InscribedTriangle where
  /-- Side length of the equilateral triangle -/
  side_length : ℝ
  /-- The triangle is inscribed in a circle -/
  inscribed : Bool
  /-- Two vertices of the triangle are endpoints of a circle diameter -/
  diameter_endpoints : Bool

/-- The shaded area outside the triangle but inside the circle -/
def shaded_area (t : InscribedTriangle) : ℝ := sorry

/-- Theorem stating the shaded area for a specific inscribed triangle -/
theorem shaded_area_formula (t : InscribedTriangle) 
  (h1 : t.side_length = 10)
  (h2 : t.inscribed = true)
  (h3 : t.diameter_endpoints = true) :
  shaded_area t = (50 * Real.pi / 3) - 25 * Real.sqrt 3 := by sorry

end shaded_area_formula_l731_73178


namespace zachary_pushups_l731_73127

/-- Given the information about David and Zachary's push-ups and crunches, 
    prove that Zachary did 28 push-ups. -/
theorem zachary_pushups (david_pushups zachary_pushups david_crunches zachary_crunches : ℕ) :
  david_pushups = zachary_pushups + 40 →
  david_crunches + 17 = zachary_crunches →
  david_crunches = 45 →
  zachary_crunches = 62 →
  zachary_pushups = 28 := by
  sorry

end zachary_pushups_l731_73127


namespace combination_equation_solution_l731_73101

theorem combination_equation_solution (n : ℕ) : 
  Nat.choose (n + 1) 7 - Nat.choose n 7 = Nat.choose n 8 → n = 14 := by
  sorry

end combination_equation_solution_l731_73101


namespace carl_removed_heads_probability_l731_73199

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents the configuration of three coins on the table -/
def CoinConfiguration := (CoinState × CoinState × CoinState)

/-- The initial configuration with Alice's coin -/
def initialConfig : CoinConfiguration := (CoinState.Heads, CoinState.Heads, CoinState.Heads)

/-- The set of all possible configurations after Bill flips two coins -/
def allConfigurations : Set CoinConfiguration := {
  (CoinState.Heads, CoinState.Heads, CoinState.Heads),
  (CoinState.Heads, CoinState.Heads, CoinState.Tails),
  (CoinState.Heads, CoinState.Tails, CoinState.Heads),
  (CoinState.Heads, CoinState.Tails, CoinState.Tails)
}

/-- The set of configurations that result in two heads showing after Carl removes a coin -/
def twoHeadsConfigurations : Set CoinConfiguration := {
  (CoinState.Heads, CoinState.Heads, CoinState.Heads),
  (CoinState.Heads, CoinState.Heads, CoinState.Tails),
  (CoinState.Heads, CoinState.Tails, CoinState.Heads)
}

/-- The probability of Carl removing a heads coin given that two heads are showing -/
def probHeadsRemoved : ℚ := 3 / 5

theorem carl_removed_heads_probability :
  probHeadsRemoved = 3 / 5 := by sorry

end carl_removed_heads_probability_l731_73199


namespace largest_two_digit_number_l731_73165

def digits : Finset Nat := {1, 2, 4, 6}

def valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ ∃ (a b : Nat), a ∈ digits ∧ b ∈ digits ∧ a ≠ b ∧ n = 10 * a + b

theorem largest_two_digit_number :
  ∀ n, valid_number n → n ≤ 64 :=
by sorry

end largest_two_digit_number_l731_73165


namespace intersection_equals_interval_l731_73133

open Set

-- Define the sets M and N
def M : Set ℝ := {x : ℝ | (x - 2) / (x + 1) ≤ 0}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- Define the interval (-1, 2]
def interval : Set ℝ := Ioc (-1) 2

-- Theorem statement
theorem intersection_equals_interval : M ∩ N = interval := by
  sorry

end intersection_equals_interval_l731_73133


namespace prob_three_white_correct_prob_two_yellow_one_white_correct_total_earnings_correct_l731_73191

-- Define the number of yellow and white balls
def yellow_balls : ℕ := 3
def white_balls : ℕ := 3

-- Define the total number of balls
def total_balls : ℕ := yellow_balls + white_balls

-- Define the number of balls drawn
def balls_drawn : ℕ := 3

-- Define the probability of drawing 3 white balls
def prob_three_white : ℚ := 1 / 20

-- Define the probability of drawing 2 yellow and 1 white ball
def prob_two_yellow_one_white : ℚ := 9 / 20

-- Define the number of draws per day
def draws_per_day : ℕ := 100

-- Define the number of days in a month
def days_in_month : ℕ := 30

-- Define the earnings for non-matching draws
def earn_non_matching : ℤ := 1

-- Define the loss for matching draws
def loss_matching : ℤ := 5

-- Theorem for the probability of drawing 3 white balls
theorem prob_three_white_correct :
  prob_three_white = 1 / 20 := by sorry

-- Theorem for the probability of drawing 2 yellow and 1 white ball
theorem prob_two_yellow_one_white_correct :
  prob_two_yellow_one_white = 9 / 20 := by sorry

-- Theorem for the total earnings in a month
theorem total_earnings_correct :
  (draws_per_day * days_in_month * 
   (earn_non_matching * (1 - (prob_three_white + prob_two_yellow_one_white)) - 
    loss_matching * (prob_three_white + prob_two_yellow_one_white))) = 1200 := by sorry

end prob_three_white_correct_prob_two_yellow_one_white_correct_total_earnings_correct_l731_73191


namespace perpendicular_parallel_implies_perpendicular_planes_parallel_perpendicular_implies_perpendicular_l731_73130

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)
variable (parallelLines : Line → Line → Prop)

-- Theorem 1: If m ⟂ α and m ∥ β, then α ⟂ β
theorem perpendicular_parallel_implies_perpendicular_planes
  (m : Line) (α β : Plane) :
  perpendicular m α → parallel m β → perpendicularPlanes α β :=
sorry

-- Theorem 2: If m ∥ n and m ⟂ α, then α ⟂ n
theorem parallel_perpendicular_implies_perpendicular
  (m n : Line) (α : Plane) :
  parallelLines m n → perpendicular m α → perpendicular n α :=
sorry

end perpendicular_parallel_implies_perpendicular_planes_parallel_perpendicular_implies_perpendicular_l731_73130


namespace larry_dog_time_l731_73141

/-- The number of minutes in half an hour -/
def half_hour : ℕ := 30

/-- The number of minutes spent feeding the dog daily -/
def feeding_time : ℕ := 12

/-- The total number of minutes Larry spends on his dog daily -/
def total_time : ℕ := 72

/-- The number of sessions Larry spends walking and playing with his dog daily -/
def walking_playing_sessions : ℕ := 2

theorem larry_dog_time :
  half_hour * walking_playing_sessions + feeding_time = total_time :=
sorry

end larry_dog_time_l731_73141


namespace student_base_choices_l731_73118

/-- The number of bases available for students to choose from -/
def num_bases : ℕ := 4

/-- The number of students choosing bases -/
def num_students : ℕ := 4

/-- The total number of ways for students to choose bases -/
def total_ways : ℕ := num_bases ^ num_students

theorem student_base_choices : total_ways = 256 := by
  sorry

end student_base_choices_l731_73118


namespace cubic_function_extrema_l731_73153

/-- A function f with two extremum points on ℝ -/
def has_two_extrema (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ y : ℝ, (f y ≤ f x₁ ∨ f y ≥ f x₁) ∧ (f y ≤ f x₂ ∨ f y ≥ f x₂))

/-- The main theorem -/
theorem cubic_function_extrema (a : ℝ) :
  has_two_extrema (λ x : ℝ => x^3 + a*x) → a < 0 :=
by sorry

end cubic_function_extrema_l731_73153


namespace matrix_determinant_l731_73140

theorem matrix_determinant (x y : ℝ) : 
  Matrix.det ![![x, x, y], ![x, y, x], ![y, x, x]] = 3 * x^2 * y - 2 * x^3 - y^3 := by
  sorry

end matrix_determinant_l731_73140


namespace area_ratio_theorem_l731_73100

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents a trapezoid -/
structure Trapezoid :=
  (H : Point) (I : Point) (J : Point) (K : Point)

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if two lines are parallel -/
def areParallel (p1 p2 q1 q2 : Point) : Prop := sorry

/-- Checks if four points are equally spaced on a line -/
def equallySpaced (p1 p2 p3 p4 : Point) : Prop := sorry

/-- Calculates the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the area of a trapezoid -/
def trapezoidArea (t : Trapezoid) : ℝ := sorry

theorem area_ratio_theorem (ABC : Triangle) (HIJK : Trapezoid) 
  (D E F G : Point) :
  isEquilateral ABC →
  areParallel D E B C →
  areParallel F G B C →
  areParallel HIJK.H HIJK.I B C →
  areParallel HIJK.J HIJK.K B C →
  equallySpaced ABC.A D F HIJK.H →
  equallySpaced ABC.A D F HIJK.J →
  trapezoidArea HIJK / triangleArea ABC = 9 / 25 := by
  sorry

end area_ratio_theorem_l731_73100


namespace car_speeds_satisfy_conditions_l731_73144

/-- Represents the scenario of two cars meeting on a road --/
structure CarMeetingScenario where
  distance : ℝ
  speed1 : ℝ
  speed2 : ℝ
  speed_increase1 : ℝ
  speed_increase2 : ℝ
  time_difference : ℝ

/-- Checks if the given car speeds satisfy the meeting conditions --/
def satisfies_conditions (s : CarMeetingScenario) : Prop :=
  s.distance / (s.speed1 - s.speed2) - s.distance / ((s.speed1 + s.speed_increase1) - (s.speed2 + s.speed_increase2)) = s.time_difference

/-- The theorem stating that the given speeds satisfy the conditions --/
theorem car_speeds_satisfy_conditions : ∃ (s : CarMeetingScenario),
  s.distance = 60 ∧
  s.speed_increase1 = 10 ∧
  s.speed_increase2 = 8 ∧
  s.time_difference = 1 ∧
  s.speed1 = 50 ∧
  s.speed2 = 40 ∧
  satisfies_conditions s := by
  sorry


end car_speeds_satisfy_conditions_l731_73144


namespace pencils_in_drawer_l731_73190

theorem pencils_in_drawer (initial_pencils final_pencils : ℕ) 
  (h1 : initial_pencils = 2)
  (h2 : final_pencils = 5) :
  final_pencils - initial_pencils = 3 := by
  sorry

end pencils_in_drawer_l731_73190


namespace new_ratio_is_two_to_three_l731_73119

/-- Represents the ratio of two quantities -/
structure Ratio :=
  (numer : ℚ)
  (denom : ℚ)

/-- The initial ratio of acid to base -/
def initialRatio : Ratio := ⟨4, 1⟩

/-- The initial volume of acid in litres -/
def initialAcidVolume : ℚ := 16

/-- The volume of mixture taken out in litres -/
def volumeTakenOut : ℚ := 10

/-- The volume of base added in litres -/
def volumeBaseAdded : ℚ := 10

/-- Calculate the new ratio of acid to base after the replacement -/
def newRatio : Ratio :=
  let initialBaseVolume := initialAcidVolume / initialRatio.numer * initialRatio.denom
  let totalInitialVolume := initialAcidVolume + initialBaseVolume
  let acidRemoved := volumeTakenOut * (initialRatio.numer / (initialRatio.numer + initialRatio.denom))
  let baseRemoved := volumeTakenOut * (initialRatio.denom / (initialRatio.numer + initialRatio.denom))
  let remainingAcid := initialAcidVolume - acidRemoved
  let remainingBase := initialBaseVolume - baseRemoved + volumeBaseAdded
  ⟨remainingAcid, remainingBase⟩

theorem new_ratio_is_two_to_three :
  newRatio = ⟨2, 3⟩ := by sorry


end new_ratio_is_two_to_three_l731_73119


namespace system_solution_l731_73167

theorem system_solution :
  ∀ (x y a : ℝ),
  (2 * x + y = a) →
  (x + y = 3) →
  (x = 2) →
  (a = 5 ∧ y = 1) := by
sorry

end system_solution_l731_73167


namespace pinky_pies_l731_73192

theorem pinky_pies (helen_pies total_pies : ℕ) 
  (helen_made : helen_pies = 56)
  (total : total_pies = 203) :
  total_pies - helen_pies = 147 := by
  sorry

end pinky_pies_l731_73192


namespace trigonometric_identities_l731_73102

theorem trigonometric_identities (α : Real) 
  (h : (1 + Real.tan α) / (1 - Real.tan α) = 2) : 
  (Real.sin α - 2 * Real.cos α) / (2 * Real.sin α - Real.cos α) = 5 ∧ 
  Real.sin α * Real.cos α + 2 = 23 / 10 := by
  sorry

end trigonometric_identities_l731_73102


namespace M_perfect_square_divisors_l731_73147

/-- Factorial function -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- The product M as defined in the problem -/
def M : ℕ := (factorial 1) * (factorial 2) * (factorial 3) * (factorial 4) * 
              (factorial 5) * (factorial 6) * (factorial 7) * (factorial 8) * (factorial 9)

/-- Count of perfect square divisors of a natural number -/
def count_perfect_square_divisors (n : ℕ) : ℕ := sorry

/-- Theorem stating that M has 672 perfect square divisors -/
theorem M_perfect_square_divisors : count_perfect_square_divisors M = 672 := by sorry

end M_perfect_square_divisors_l731_73147


namespace square_floor_tiles_l731_73115

theorem square_floor_tiles (diagonal_tiles : ℕ) (total_tiles : ℕ) : 
  diagonal_tiles = 37 → total_tiles = 361 → 
  (∃ (side_length : ℕ), 
    2 * side_length - 1 = diagonal_tiles ∧ 
    side_length * side_length = total_tiles) := by
  sorry

end square_floor_tiles_l731_73115


namespace sin_max_implies_even_l731_73138

theorem sin_max_implies_even (f : ℝ → ℝ) (φ a : ℝ) 
  (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ f a) :
  ∀ x, f (x + a) = f (-x + a) := by
sorry

end sin_max_implies_even_l731_73138


namespace projection_matrix_condition_l731_73149

/-- A 2x2 matrix is a projection matrix if and only if its square equals itself. -/
def is_projection_matrix (P : Matrix (Fin 2) (Fin 2) ℚ) : Prop :=
  P * P = P

/-- The specific matrix we're working with -/
def P (a c : ℚ) : Matrix (Fin 2) (Fin 2) ℚ :=
  !![a, 7/17; c, 10/17]

/-- The main theorem: P is a projection matrix if and only if a = 9/17 and c = 10/17 -/
theorem projection_matrix_condition (a c : ℚ) :
  is_projection_matrix (P a c) ↔ a = 9/17 ∧ c = 10/17 := by
  sorry

end projection_matrix_condition_l731_73149


namespace three_fractions_product_one_l731_73170

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem three_fractions_product_one :
  ∃ (a b c d e f : ℕ),
    a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    Nat.gcd a b = 1 ∧ Nat.gcd c d = 1 ∧ Nat.gcd e f = 1 ∧
    (a * c * e : ℚ) / (b * d * f) = 1 := by
  sorry

end three_fractions_product_one_l731_73170


namespace inhabitable_earth_surface_inhabitable_earth_surface_proof_l731_73151

theorem inhabitable_earth_surface : Real → Prop :=
  λ x =>
    let total_surface := 1
    let land_fraction := 1 / 4
    let inhabitable_land_fraction := 1 / 2
    x = land_fraction * inhabitable_land_fraction ∧ 
    x = 1 / 8

-- Proof
theorem inhabitable_earth_surface_proof : inhabitable_earth_surface (1 / 8) := by
  sorry

end inhabitable_earth_surface_inhabitable_earth_surface_proof_l731_73151


namespace hemisphere_surface_area_l731_73134

theorem hemisphere_surface_area (r : ℝ) (h : r = 5) :
  let sphere_area (r : ℝ) := 4 * π * r^2
  let hemisphere_curved_area (r : ℝ) := (sphere_area r) / 2
  let base_area (r : ℝ) := π * r^2
  hemisphere_curved_area r + base_area r = 75 * π := by
sorry

end hemisphere_surface_area_l731_73134


namespace reimu_win_probability_l731_73109

/-- Represents the result of a single coin toss -/
inductive CoinSide
| Red
| Green

/-- Represents the state of a single coin -/
structure Coin :=
  (side1 : CoinSide)
  (side2 : CoinSide)

/-- Represents the game state -/
structure GameState :=
  (coins : Finset Coin)

/-- The number of coins in the game -/
def numCoins : Nat := 4

/-- A game is valid if it has the correct number of coins -/
def validGame (g : GameState) : Prop :=
  g.coins.card = numCoins

/-- The probability of Reimu winning the game -/
def reimuWinProbability (g : GameState) : ℚ :=
  sorry

/-- The main theorem: probability of Reimu winning is 5/16 -/
theorem reimu_win_probability (g : GameState) (h : validGame g) : 
  reimuWinProbability g = 5 / 16 := by
  sorry

end reimu_win_probability_l731_73109


namespace sum_of_digits_l731_73184

/-- Given two single-digit numbers a and b, if ab + ba = 202, then a + b = 12 -/
theorem sum_of_digits (a b : ℕ) : 
  a < 10 → b < 10 → (10 * a + b) + (10 * b + a) = 202 → a + b = 12 :=
by sorry

end sum_of_digits_l731_73184


namespace fraction_sum_equals_ten_thirds_l731_73114

theorem fraction_sum_equals_ten_thirds (a b : ℤ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (h1 : a = 2) (h2 : b = 1) : 
  (a + b) / (a - b) + (a - b) / (a + b) = 10 / 3 := by
  sorry

end fraction_sum_equals_ten_thirds_l731_73114


namespace min_apples_in_basket_l731_73107

theorem min_apples_in_basket : ∃ n : ℕ, n ≥ 23 ∧ 
  (∃ a b c : ℕ, 
    n + 4 = 3 * a ∧
    2 * a + 4 = 3 * b ∧
    2 * b + 4 = 3 * c) ∧
  (∀ m : ℕ, m < n → 
    ¬(∃ a b c : ℕ, 
      m + 4 = 3 * a ∧
      2 * a + 4 = 3 * b ∧
      2 * b + 4 = 3 * c)) :=
by sorry

end min_apples_in_basket_l731_73107


namespace expression_evaluation_l731_73128

theorem expression_evaluation : 
  Real.sqrt ((-3)^2) + (π - 3)^0 - 8^(2/3) + ((-4)^(1/3))^3 = -4 := by sorry

end expression_evaluation_l731_73128


namespace tangent_line_sum_l731_73173

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem tangent_line_sum (h : ∀ x, f 1 + 3 * (x - 1) = 3 * x - 2) : 
  f 1 + (deriv f) 1 = 4 := by
  sorry

end tangent_line_sum_l731_73173


namespace christian_future_age_l731_73152

/-- The age of Brian in years -/
def brian_age : ℕ := sorry

/-- The age of Christian in years -/
def christian_age : ℕ := sorry

/-- The number of years in the future we're considering -/
def years_future : ℕ := 8

/-- Brian's age in the future -/
def brian_future_age : ℕ := 40

theorem christian_future_age :
  christian_age + years_future = 72 :=
by
  have h1 : christian_age = 2 * brian_age := sorry
  have h2 : brian_age + years_future = brian_future_age := sorry
  sorry

end christian_future_age_l731_73152


namespace other_communities_count_l731_73160

theorem other_communities_count (total_boys : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 700 →
  muslim_percent = 44 / 100 →
  hindu_percent = 28 / 100 →
  sikh_percent = 10 / 100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 126 :=
by sorry

end other_communities_count_l731_73160


namespace quadratic_real_roots_l731_73113

theorem quadratic_real_roots (k : ℕ) : 
  (∃ x : ℝ, k * x^2 - 3 * x + 2 = 0) ↔ k = 1 := by
  sorry

end quadratic_real_roots_l731_73113


namespace smallest_area_special_square_l731_73161

/-- A square with vertices on a line and parabola -/
structure SpecialSquare where
  -- One pair of opposite vertices lie on this line
  line : Real → Real
  -- The other pair of opposite vertices lie on this parabola
  parabola : Real → Real
  -- The line is y = -2x + 17
  line_eq : line = fun x => -2 * x + 17
  -- The parabola is y = x^2 - 2
  parabola_eq : parabola = fun x => x^2 - 2

/-- The smallest possible area of a SpecialSquare is 160 -/
theorem smallest_area_special_square (s : SpecialSquare) :
  ∃ (area : Real), area = 160 ∧ 
  (∀ (other_area : Real), other_area ≥ area) :=
by sorry

end smallest_area_special_square_l731_73161


namespace max_residents_in_block_l731_73168

/-- Represents a block of flats -/
structure BlockOfFlats where
  totalFloors : ℕ
  apartmentsPerFloorType1 : ℕ
  apartmentsPerFloorType2 : ℕ
  maxResidentsPerApartment : ℕ

/-- Calculates the maximum number of residents in a block of flats -/
def maxResidents (block : BlockOfFlats) : ℕ :=
  let floorsType1 := block.totalFloors / 2
  let floorsType2 := block.totalFloors - floorsType1
  let totalApartments := floorsType1 * block.apartmentsPerFloorType1 + floorsType2 * block.apartmentsPerFloorType2
  totalApartments * block.maxResidentsPerApartment

/-- Theorem stating the maximum number of residents in the given block of flats -/
theorem max_residents_in_block :
  let block : BlockOfFlats := {
    totalFloors := 12,
    apartmentsPerFloorType1 := 6,
    apartmentsPerFloorType2 := 5,
    maxResidentsPerApartment := 4
  }
  maxResidents block = 264 := by
  sorry

end max_residents_in_block_l731_73168


namespace midpoint_x_sum_eq_vertex_x_sum_l731_73155

/-- Given a triangle in the Cartesian plane, the sum of the x-coordinates of the midpoints
    of its sides is equal to the sum of the x-coordinates of its vertices. -/
theorem midpoint_x_sum_eq_vertex_x_sum (a b c : ℝ) : 
  let vertex_sum := a + b + c
  let midpoint_sum := (a + b) / 2 + (a + c) / 2 + (b + c) / 2
  midpoint_sum = vertex_sum :=
by sorry

end midpoint_x_sum_eq_vertex_x_sum_l731_73155


namespace chess_tournament_games_l731_73117

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 10 players where each player plays every other player once,
    the total number of games played is 45. --/
theorem chess_tournament_games :
  num_games 10 = 45 := by sorry

end chess_tournament_games_l731_73117


namespace bananas_needed_l731_73186

def yogurt_count : ℕ := 5
def slices_per_yogurt : ℕ := 8
def slices_per_banana : ℕ := 10

theorem bananas_needed : 
  (yogurt_count * slices_per_yogurt + slices_per_banana - 1) / slices_per_banana = 4 := by
  sorry

end bananas_needed_l731_73186


namespace no_right_triangle_with_specific_medians_l731_73197

theorem no_right_triangle_with_specific_medians : ∀ (m b a_leg b_leg : ℝ),
  a_leg > 0 → b_leg > 0 →
  (∃ (x y : ℝ), y = m * x + b) →  -- hypotenuse parallel to y = mx + b
  (∃ (x y : ℝ), y = 2 * x + 1) →  -- one median on y = 2x + 1
  (∃ (x y : ℝ), y = 5 * x + 2) →  -- another median on y = 5x + 2
  ¬ (
    -- Right triangle condition
    a_leg^2 + b_leg^2 = (a_leg^2 + b_leg^2) ∧
    -- Hypotenuse parallel to y = mx + b
    m = -b_leg / a_leg ∧
    -- One median on y = 2x + 1
    (2 * b_leg / a_leg = 2 ∨ b_leg / (2 * a_leg) = 2) ∧
    -- Another median on y = 5x + 2
    (2 * b_leg / a_leg = 5 ∨ b_leg / (2 * a_leg) = 5)
  ) := by
  sorry

end no_right_triangle_with_specific_medians_l731_73197


namespace bucket_problem_l731_73179

/-- Represents the state of the two buckets -/
structure BucketState :=
  (large : ℕ)  -- Amount in 7-liter bucket
  (small : ℕ)  -- Amount in 3-liter bucket

/-- Represents a single operation on the buckets -/
inductive BucketOperation
  | FillLarge
  | FillSmall
  | EmptyLarge
  | EmptySmall
  | PourLargeToSmall
  | PourSmallToLarge

/-- Applies a single operation to a bucket state -/
def applyOperation (state : BucketState) (op : BucketOperation) : BucketState :=
  match op with
  | BucketOperation.FillLarge => { large := 7, small := state.small }
  | BucketOperation.FillSmall => { large := state.large, small := 3 }
  | BucketOperation.EmptyLarge => { large := 0, small := state.small }
  | BucketOperation.EmptySmall => { large := state.large, small := 0 }
  | BucketOperation.PourLargeToSmall =>
      let amount := min state.large (3 - state.small)
      { large := state.large - amount, small := state.small + amount }
  | BucketOperation.PourSmallToLarge =>
      let amount := min state.small (7 - state.large)
      { large := state.large + amount, small := state.small - amount }

/-- Applies a sequence of operations to an initial state -/
def applyOperations (initial : BucketState) (ops : List BucketOperation) : BucketState :=
  ops.foldl applyOperation initial

/-- Checks if a specific amount can be measured using a sequence of operations -/
def canMeasure (amount : ℕ) : Prop :=
  ∃ (ops : List BucketOperation),
    (applyOperations { large := 0, small := 0 } ops).large = amount ∨
    (applyOperations { large := 0, small := 0 } ops).small = amount

theorem bucket_problem :
  canMeasure 1 ∧ canMeasure 2 ∧ canMeasure 4 ∧ canMeasure 5 ∧ canMeasure 6 :=
sorry

end bucket_problem_l731_73179


namespace min_value_theorem_l731_73156

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.exp x = y * Real.log x + y * Real.log y) : 
  ∃ (m : ℝ), ∀ (x' y' : ℝ) (hx' : x' > 0) (hy' : y' > 0) 
  (h' : Real.exp x' = y' * Real.log x' + y' * Real.log y'), 
  (Real.exp x' / x' - Real.log y') ≥ m ∧ 
  (Real.exp x / x - Real.log y = m) ∧ 
  m = Real.exp 1 - 1 :=
sorry

end min_value_theorem_l731_73156


namespace prob_product_multiple_of_four_is_two_fifths_l731_73131

/-- A fair 12-sided die -/
def dodecahedral_die := Finset.range 12

/-- A fair 10-sided die -/
def ten_sided_die := Finset.range 10

/-- The probability of an event occurring when rolling a fair n-sided die -/
def prob_fair_die (event : Finset ℕ) (die : Finset ℕ) : ℚ :=
  (event ∩ die).card / die.card

/-- The event of rolling a multiple of 4 -/
def multiple_of_four (die : Finset ℕ) : Finset ℕ :=
  die.filter (λ x => x % 4 = 0)

/-- The probability that the product of two rolls is a multiple of 4 -/
def prob_product_multiple_of_four : ℚ :=
  1 - (1 - prob_fair_die (multiple_of_four dodecahedral_die) dodecahedral_die) *
      (1 - prob_fair_die (multiple_of_four ten_sided_die) ten_sided_die)

theorem prob_product_multiple_of_four_is_two_fifths :
  prob_product_multiple_of_four = 2 / 5 := by
  sorry

end prob_product_multiple_of_four_is_two_fifths_l731_73131


namespace arithmetic_mean_problem_l731_73108

theorem arithmetic_mean_problem (y : ℝ) : 
  (8 + 15 + 20 + 7 + y + 9) / 6 = 12 → y = 13 := by
  sorry

end arithmetic_mean_problem_l731_73108


namespace cindy_pen_addition_l731_73193

theorem cindy_pen_addition (initial_pens : ℕ) (mike_pens : ℕ) (sharon_pens : ℕ) (final_pens : ℕ)
  (h1 : initial_pens = 20)
  (h2 : mike_pens = 22)
  (h3 : sharon_pens = 19)
  (h4 : final_pens = 65) :
  final_pens - (initial_pens + mike_pens - sharon_pens) = 42 :=
by sorry

end cindy_pen_addition_l731_73193


namespace average_first_ten_even_numbers_l731_73146

theorem average_first_ten_even_numbers :
  let first_ten_even : List ℕ := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
  (first_ten_even.sum / first_ten_even.length : ℚ) = 11 := by
sorry

end average_first_ten_even_numbers_l731_73146


namespace box_2_neg2_3_l731_73185

def box (a b c : ℤ) : ℚ := (a ^ b) - (b ^ c) + (c ^ a)

theorem box_2_neg2_3 : box 2 (-2) 3 = 5 / 4 := by sorry

end box_2_neg2_3_l731_73185


namespace product_of_sum_and_cube_sum_l731_73142

theorem product_of_sum_and_cube_sum (a b : ℝ) 
  (h1 : a + b = 4) 
  (h2 : a^3 + b^3 = 136) : 
  a * b = -6 := by
sorry

end product_of_sum_and_cube_sum_l731_73142


namespace fence_painted_fraction_l731_73132

/-- The fraction of a fence Tom paints while Jerry digs a hole -/
def fence_fraction (tom_rate jerry_rate : ℚ) : ℚ :=
  (jerry_rate / tom_rate)

theorem fence_painted_fraction :
  fence_fraction (1 / 60) (1 / 40) = 2 / 3 := by
  sorry

end fence_painted_fraction_l731_73132


namespace h_function_iff_strictly_increasing_l731_73198

-- Define an H function
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

-- Define a strictly increasing function
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- Theorem: A function is an H function if and only if it is strictly increasing
theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end h_function_iff_strictly_increasing_l731_73198


namespace face_mask_profit_l731_73164

/-- Calculates the total profit from selling face masks --/
def calculate_profit (num_boxes : ℕ) (price_per_mask : ℚ) (masks_per_box : ℕ) (total_cost : ℚ) : ℚ :=
  num_boxes * price_per_mask * masks_per_box - total_cost

/-- Proves that the total profit is $15 given the specified conditions --/
theorem face_mask_profit :
  let num_boxes : ℕ := 3
  let price_per_mask : ℚ := 1/2
  let masks_per_box : ℕ := 20
  let total_cost : ℚ := 15
  calculate_profit num_boxes price_per_mask masks_per_box total_cost = 15 := by
  sorry

#eval calculate_profit 3 (1/2) 20 15

end face_mask_profit_l731_73164


namespace decimal_3_is_binary_11_binary_11_is_decimal_3_l731_73112

/-- Converts a natural number to its binary representation as a list of bits (0s and 1s) --/
def toBinary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 2) ((m % 2) :: acc)
    go n []

/-- Converts a list of bits (0s and 1s) to its decimal representation --/
def fromBinary (bits : List ℕ) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + b) 0

theorem decimal_3_is_binary_11 : toBinary 3 = [1, 1] := by
  sorry

theorem binary_11_is_decimal_3 : fromBinary [1, 1] = 3 := by
  sorry

end decimal_3_is_binary_11_binary_11_is_decimal_3_l731_73112


namespace computer_cost_is_400_l731_73121

/-- The cost of the computer Delores bought -/
def computer_cost : ℕ := sorry

/-- The initial amount of money Delores had -/
def initial_money : ℕ := 450

/-- The combined cost of the computer and printer -/
def total_purchase : ℕ := 40

/-- The amount of money Delores had left after the purchase -/
def money_left : ℕ := 10

/-- Theorem stating that the computer cost $400 -/
theorem computer_cost_is_400 :
  computer_cost = 400 :=
by sorry

end computer_cost_is_400_l731_73121


namespace car_distance_calculation_l731_73126

/-- The distance covered by the car in kilometers -/
def car_distance_km : ℝ := 2.2

/-- The distance covered by Amar in meters -/
def amar_distance_m : ℝ := 880

/-- The ratio of Amar's speed to the car's speed -/
def speed_ratio : ℚ := 2 / 5

theorem car_distance_calculation :
  car_distance_km = (amar_distance_m / speed_ratio) / 1000 := by
  sorry

#check car_distance_calculation

end car_distance_calculation_l731_73126


namespace smallest_n_g_greater_than_20_l731_73123

/-- The sum of digits to the right of the decimal point in 1/(3^n) -/
def g (n : ℕ+) : ℕ := sorry

/-- Theorem stating that 4 is the smallest positive integer n such that g(n) > 20 -/
theorem smallest_n_g_greater_than_20 :
  (∀ k : ℕ+, k < 4 → g k ≤ 20) ∧ g 4 > 20 := by sorry

end smallest_n_g_greater_than_20_l731_73123


namespace discount_theorem_l731_73159

/-- Calculates the final price and equivalent discount for a given original price and discounts --/
def discount_calculation (original_price : ℝ) (store_discount : ℝ) (vip_discount : ℝ) : ℝ × ℝ :=
  let final_price := original_price * (1 - store_discount) * (1 - vip_discount)
  let equivalent_discount := 1 - (1 - store_discount) * (1 - vip_discount)
  (final_price, equivalent_discount * 100)

theorem discount_theorem :
  discount_calculation 1500 0.8 0.05 = (1140, 76) := by
  sorry

end discount_theorem_l731_73159


namespace a_investment_l731_73105

/-- A partnership business with three partners A, B, and C. -/
structure Partnership where
  investA : ℕ
  investB : ℕ
  investC : ℕ
  totalProfit : ℕ
  cShareProfit : ℕ

/-- The partnership satisfies the given conditions -/
def validPartnership (p : Partnership) : Prop :=
  p.investB = 8000 ∧
  p.investC = 9000 ∧
  p.totalProfit = 88000 ∧
  p.cShareProfit = 36000

/-- The theorem stating A's investment amount -/
theorem a_investment (p : Partnership) (h : validPartnership p) : 
  p.investA = 5000 := by
  sorry

end a_investment_l731_73105


namespace sandys_remaining_nickels_l731_73111

/-- Given an initial number of nickels and a number of borrowed nickels,
    calculate the remaining nickels. -/
def remaining_nickels (initial : ℕ) (borrowed : ℕ) : ℕ :=
  initial - borrowed

/-- Theorem stating that Sandy's remaining nickels is 11 -/
theorem sandys_remaining_nickels :
  remaining_nickels 31 20 = 11 := by
  sorry

end sandys_remaining_nickels_l731_73111


namespace distribute_objects_l731_73180

theorem distribute_objects (n : ℕ) (k : ℕ) (m : ℕ) (h1 : n = 12) (h2 : k = 3) (h3 : m = 4) (h4 : n = k * m) :
  (Nat.factorial n) / ((Nat.factorial m)^k) = 34650 :=
sorry

end distribute_objects_l731_73180


namespace range_of_a_l731_73110

-- Define sets A and B
def A : Set ℝ := {x | x > 5}
def B (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (h : ∀ x, x ∈ A → x ∈ B a) 
                   (h_not_nec : ∃ x, x ∈ B a ∧ x ∉ A) : 
  a > 5 := by
  sorry

end range_of_a_l731_73110


namespace units_digit_of_square_l731_73154

theorem units_digit_of_square (n : ℕ) : (n ^ 2) % 10 ≠ 8 := by
  sorry

end units_digit_of_square_l731_73154


namespace smallest_multiple_l731_73175

theorem smallest_multiple (n : ℕ) : 
  (∃ k : ℕ, n = 17 * k) ∧ 
  n % 101 = 3 ∧ 
  (∀ m : ℕ, m < n → ¬((∃ k : ℕ, m = 17 * k) ∧ m % 101 = 3)) → 
  n = 306 := by
sorry

end smallest_multiple_l731_73175


namespace third_number_in_proportion_l731_73172

theorem third_number_in_proportion (x : ℝ) (h : x = 3) : 
  ∃ y : ℝ, (x + 1) / (x + 5) = (x + 5) / (x + y) → y = 13 := by
  sorry

end third_number_in_proportion_l731_73172


namespace least_common_duration_l731_73195

/-- Represents a business partner -/
structure Partner where
  investment : ℚ
  duration : ℕ

/-- Represents the business venture -/
structure BusinessVenture where
  p : Partner
  q : Partner
  r : Partner
  investmentRatio : Fin 3 → ℚ
  profitRatio : Fin 3 → ℚ

/-- The profit is proportional to the product of investment and duration -/
def profitProportional (bv : BusinessVenture) : Prop :=
  ∃ (k : ℚ), k > 0 ∧
    bv.profitRatio 0 = k * bv.p.investment * bv.p.duration ∧
    bv.profitRatio 1 = k * bv.q.investment * bv.q.duration ∧
    bv.profitRatio 2 = k * bv.r.investment * bv.r.duration

/-- The main theorem -/
theorem least_common_duration (bv : BusinessVenture) 
    (h1 : bv.investmentRatio = ![7, 5, 3])
    (h2 : bv.profitRatio = ![7, 10, 6])
    (h3 : bv.p.duration = 8)
    (h4 : bv.q.duration = 6)
    (h5 : profitProportional bv) :
    bv.r.duration = 6 := by
  sorry

end least_common_duration_l731_73195


namespace cone_volume_l731_73122

/-- A cone with base area π and lateral surface in the shape of a semicircle has volume (√3 / 3)π -/
theorem cone_volume (r h l : ℝ) : 
  r > 0 → h > 0 → l > 0 →
  π * r^2 = π →
  π * l = 2 * π * r →
  h^2 + r^2 = l^2 →
  (1/3) * π * r^2 * h = (Real.sqrt 3 / 3) * π := by
sorry

end cone_volume_l731_73122


namespace towns_distance_l731_73162

/-- Given a map distance and a scale, calculate the actual distance between two towns. -/
def actual_distance (map_distance : ℝ) (scale : ℝ) : ℝ :=
  map_distance * scale

/-- Theorem stating that for a map distance of 20 inches and a scale of 1 inch = 10 miles,
    the actual distance between the towns is 200 miles. -/
theorem towns_distance :
  let map_distance : ℝ := 20
  let scale : ℝ := 10
  actual_distance map_distance scale = 200 := by
sorry

end towns_distance_l731_73162


namespace quadratic_inequality_solution_l731_73177

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 4) ↔ (a * x^2 + b * x - 2 > 0)) → 
  a + b = 2 := by
sorry

end quadratic_inequality_solution_l731_73177


namespace cubic_polynomial_coefficient_l731_73116

/-- 
Given a cubic polynomial y = ax³ + bx² + cx + d, 
if (1, y₁) and (-1, y₂) lie on its graph and y₁ - y₂ = -8, 
then a = -4.
-/
theorem cubic_polynomial_coefficient (a b c d y₁ y₂ : ℝ) : 
  y₁ = a + b + c + d → 
  y₂ = -a + b - c + d → 
  y₁ - y₂ = -8 → 
  a = -4 := by
sorry

end cubic_polynomial_coefficient_l731_73116


namespace triangle_property_l731_73124

/-- Given a triangle ABC where a^2 + c^2 = b^2 + √2*a*c, prove that:
    1. The size of angle B is π/4
    2. The maximum value of √2*cos(A) + cos(C) is 1 -/
theorem triangle_property (a b c : ℝ) (h : a^2 + c^2 = b^2 + Real.sqrt 2 * a * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  let B := Real.arccos ((a^2 + c^2 - b^2) / (2 * a * c))
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  (B = π / 4) ∧
  (∃ (x : ℝ), Real.sqrt 2 * Real.cos A + Real.cos C ≤ x ∧ x = 1) := by
  sorry

end triangle_property_l731_73124


namespace diamond_two_four_l731_73106

-- Define the Diamond operation
def Diamond (a b : ℤ) : ℤ := a * b^3 - b + 2

-- Theorem statement
theorem diamond_two_four : Diamond 2 4 = 126 := by
  sorry

end diamond_two_four_l731_73106


namespace distance_between_cities_l731_73181

/-- The distance between two cities given the speeds and times of two cars traveling between them -/
theorem distance_between_cities (meeting_time : ℝ) (car_b_speed : ℝ) (car_a_remaining_time : ℝ) :
  let car_a_speed := car_b_speed * meeting_time / car_a_remaining_time
  let total_distance := (car_a_speed + car_b_speed) * meeting_time
  meeting_time = 6 ∧ car_b_speed = 69 ∧ car_a_remaining_time = 4 →
  total_distance = (69 * 6 / 4 + 69) * 6 := by
sorry

end distance_between_cities_l731_73181


namespace kylies_coins_l731_73174

theorem kylies_coins (piggy_bank : ℕ) (brother : ℕ) (father : ℕ) (gave_away : ℕ) (left : ℕ) : 
  piggy_bank = 15 → 
  brother = 13 → 
  gave_away = 21 → 
  left = 15 → 
  piggy_bank + brother + father - gave_away = left → 
  father = 8 := by
sorry

end kylies_coins_l731_73174


namespace white_washing_cost_calculation_l731_73194

/-- Calculate the cost of white washing a room's walls --/
def white_washing_cost (room_length room_width room_height : ℝ)
                       (door_height door_width : ℝ)
                       (window_height window_width : ℝ)
                       (num_windows : ℕ)
                       (cost_per_sqft : ℝ) : ℝ :=
  let wall_area := 2 * (room_length + room_width) * room_height
  let door_area := door_height * door_width
  let window_area := num_windows * (window_height * window_width)
  let area_to_wash := wall_area - (door_area + window_area)
  area_to_wash * cost_per_sqft

/-- Theorem stating the cost of white washing the room --/
theorem white_washing_cost_calculation :
  white_washing_cost 25 15 12 6 3 4 3 3 4 = 3624 := by
  sorry

end white_washing_cost_calculation_l731_73194


namespace wombat_claws_l731_73104

theorem wombat_claws (num_wombats num_rheas total_claws : ℕ) 
  (h1 : num_wombats = 9)
  (h2 : num_rheas = 3)
  (h3 : total_claws = 39) :
  ∃ (wombat_claws : ℕ), 
    wombat_claws * num_wombats + num_rheas = total_claws ∧ 
    wombat_claws = 4 := by
  sorry

end wombat_claws_l731_73104


namespace factor_implies_d_value_l731_73183

theorem factor_implies_d_value (d : ℚ) : 
  (∀ x : ℚ, (x - 5) ∣ (d*x^4 + 19*x^3 - 10*d*x^2 + 45*x - 90)) → 
  d = -502/75 := by
  sorry

end factor_implies_d_value_l731_73183


namespace greatest_k_value_l731_73196

theorem greatest_k_value (k : ℝ) : 
  (∃ x y : ℝ, x^2 + k*x + 8 = 0 ∧ y^2 + k*y + 8 = 0 ∧ |x - y| = Real.sqrt 72) →
  k ≤ 2 * Real.sqrt 26 :=
sorry

end greatest_k_value_l731_73196


namespace bakers_cakes_l731_73157

/-- Baker's cake problem -/
theorem bakers_cakes (initial_cakes sold_cakes final_cakes : ℕ) 
  (h1 : initial_cakes = 110)
  (h2 : sold_cakes = 75)
  (h3 : final_cakes = 111) :
  final_cakes - (initial_cakes - sold_cakes) = 76 := by
  sorry

end bakers_cakes_l731_73157


namespace chinese_remainder_theorem_example_l731_73176

theorem chinese_remainder_theorem_example :
  ∀ x : ℤ, x ≡ 9 [ZMOD 17] → x ≡ 5 [ZMOD 11] → x ≡ 60 [ZMOD 187] := by
  sorry

end chinese_remainder_theorem_example_l731_73176


namespace min_distinct_sums_products_l731_73171

theorem min_distinct_sums_products (a b c d : ℤ) (h1 : a < b) (h2 : b < c) (h3 : c < d) :
  let sums := {a + b, a + c, a + d, b + c, b + d, c + d}
  let products := {a * b, a * c, a * d, b * c, b * d, c * d}
  Finset.card (sums ∪ products) ≥ 6 :=
sorry

end min_distinct_sums_products_l731_73171


namespace complex_magnitude_l731_73143

theorem complex_magnitude (z : ℂ) (h : Complex.I - z = 1 + 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l731_73143


namespace first_bell_weight_l731_73139

theorem first_bell_weight (w : ℝ) 
  (h1 : w > 0)  -- Ensuring positive weight
  (h2 : 2 * w > 0)  -- Weight of second bell
  (h3 : 4 * (2 * w) > 0)  -- Weight of third bell
  (h4 : w + 2 * w + 4 * (2 * w) = 550)  -- Total weight condition
  : w = 50 := by
  sorry

end first_bell_weight_l731_73139


namespace line_through_focus_line_intersects_ellipse_l731_73166

/-- The equation of the line l -/
def line (k : ℝ) (x : ℝ) : ℝ := k * x + 4

/-- The equation of the ellipse C -/
def ellipse (x y : ℝ) : Prop := x^2 / 5 + y^2 = 1

/-- The x-coordinate of the left focus of the ellipse -/
def left_focus : ℝ := -2

/-- Theorem: When the line passes through the left focus of the ellipse, k = 2 -/
theorem line_through_focus (k : ℝ) : 
  line k left_focus = 0 → k = 2 :=
sorry

/-- Theorem: The line intersects the ellipse if and only if k is in the specified range -/
theorem line_intersects_ellipse (k : ℝ) : 
  (∃ x y, ellipse x y ∧ y = line k x) ↔ k ≤ -Real.sqrt 3 ∨ k ≥ Real.sqrt 3 :=
sorry

end line_through_focus_line_intersects_ellipse_l731_73166


namespace points_collinear_l731_73148

-- Define the function for log base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- Define the function for log base 2
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

-- Define the line passing through the origin
def line_through_origin (k : ℝ) (x : ℝ) : ℝ := k * x

-- Define the theorem
theorem points_collinear (k a b : ℝ) 
  (ha : line_through_origin k a = log8 a)
  (hb : line_through_origin k b = log8 b)
  (hc : ∃ c, c = (a, log2 a))
  (hd : ∃ d, d = (b, log2 b)) :
  ∃ m, line_through_origin m a = log2 a ∧ 
       line_through_origin m b = log2 b ∧
       line_through_origin m 0 = 0 := by
  sorry


end points_collinear_l731_73148


namespace employee_salary_l731_73135

theorem employee_salary (total_salary : ℝ) (m_salary_percentage : ℝ) 
  (h1 : total_salary = 616)
  (h2 : m_salary_percentage = 120 / 100) : 
  ∃ (n_salary : ℝ), 
    n_salary + m_salary_percentage * n_salary = total_salary ∧ 
    n_salary = 280 := by
  sorry

end employee_salary_l731_73135


namespace ellipse_circle_relation_l731_73137

/-- An ellipse in the Cartesian coordinate system -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- A circle in the Cartesian coordinate system -/
structure Circle where
  r : ℝ
  h_pos : r > 0

/-- A line in the Cartesian coordinate system -/
structure Line where
  k : ℝ
  m : ℝ

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if a point is on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  p.x^2 / e.a^2 + p.y^2 / e.b^2 = 1

/-- Checks if a point is on a circle -/
def on_circle (p : Point) (c : Circle) : Prop :=
  p.x^2 + p.y^2 = c.r^2

/-- Checks if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  l.m^2 = c.r^2 * (1 + l.k^2)

/-- Checks if three points form a right angle -/
def is_right_angle (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

theorem ellipse_circle_relation 
  (e : Ellipse) (c : Circle) (l : Line) (A B : Point) :
  c.r < e.b →
  is_tangent l c →
  on_ellipse A e ∧ on_ellipse B e →
  on_circle A c ∧ on_circle B c →
  is_right_angle A B (Point.mk 0 0) →
  1 / e.a^2 + 1 / e.b^2 = 1 / c.r^2 :=
sorry

end ellipse_circle_relation_l731_73137


namespace power_of_three_mod_eleven_l731_73136

theorem power_of_three_mod_eleven : 3^2023 % 11 = 5 := by
  sorry

end power_of_three_mod_eleven_l731_73136


namespace distinct_roots_range_reciprocal_roots_sum_squares_l731_73187

-- Define the quadratic equation
def quadratic (x m : ℝ) : ℝ := x^2 - 3*x + m - 3

-- Theorem for the range of m
theorem distinct_roots_range (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic x₁ m = 0 ∧ quadratic x₂ m = 0) ↔ m < 21/4 :=
sorry

-- Theorem for the sum of squares of reciprocal roots
theorem reciprocal_roots_sum_squares (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic x₁ m = 0 ∧ quadratic x₂ m = 0 ∧ x₁ * x₂ = 1 →
  x₁^2 + x₂^2 = 7 :=
sorry

end distinct_roots_range_reciprocal_roots_sum_squares_l731_73187


namespace pie_sugar_percentage_l731_73145

/-- Given a pie weighing 200 grams with 50 grams of sugar, 
    prove that 75% of the pie is not sugar. -/
theorem pie_sugar_percentage (total_weight : ℝ) (sugar_weight : ℝ) 
    (h1 : total_weight = 200) 
    (h2 : sugar_weight = 50) : 
    (total_weight - sugar_weight) / total_weight * 100 = 75 := by
  sorry

end pie_sugar_percentage_l731_73145


namespace power_multiplication_l731_73163

theorem power_multiplication (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 := by
  sorry

end power_multiplication_l731_73163


namespace events_mutually_exclusive_but_not_opposite_l731_73150

-- Define the set of cards
inductive Card : Type
| Black : Card
| Red : Card
| White : Card

-- Define the set of individuals
inductive Person : Type
| A : Person
| B : Person
| C : Person

-- Define a distribution of cards
def Distribution := Person → Card

-- Define the event "Individual A gets the red card"
def EventA (d : Distribution) : Prop := d Person.A = Card.Red

-- Define the event "Individual B gets the red card"
def EventB (d : Distribution) : Prop := d Person.B = Card.Red

-- Theorem statement
theorem events_mutually_exclusive_but_not_opposite :
  -- The events are mutually exclusive
  (∀ d : Distribution, ¬(EventA d ∧ EventB d)) ∧
  -- The events are not opposite
  (∃ d : Distribution, ¬EventA d ∧ ¬EventB d) :=
sorry

end events_mutually_exclusive_but_not_opposite_l731_73150
