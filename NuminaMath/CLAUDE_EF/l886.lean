import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_BaO_l886_88697

/-- The molar mass of barium in g/mol -/
noncomputable def molar_mass_Ba : ℝ := 137.33

/-- The molar mass of oxygen in g/mol -/
noncomputable def molar_mass_O : ℝ := 16.00

/-- The molar mass of barium oxide (BaO) in g/mol -/
noncomputable def molar_mass_BaO : ℝ := molar_mass_Ba + molar_mass_O

/-- The mass percentage of oxygen in BaO -/
noncomputable def mass_percentage_O : ℝ := (molar_mass_O / molar_mass_BaO) * 100

/-- Theorem: The mass percentage of oxygen in BaO is approximately 10.43% -/
theorem mass_percentage_O_in_BaO :
  abs (mass_percentage_O - 10.43) < 0.01 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_O_in_BaO_l886_88697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pumps_combined_time_l886_88667

/-- Represents the time (in hours) it takes for a pump to empty a pool -/
structure PumpRate where
  hours : ℝ
  positive : hours > 0

/-- Calculates the combined rate of two pumps -/
noncomputable def combinedRate (a b : PumpRate) : ℝ :=
  1 / a.hours + 1 / b.hours

/-- Calculates the time (in minutes) it takes for two pumps to empty a pool together -/
noncomputable def combinedTime (a b : PumpRate) : ℝ :=
  (1 / combinedRate a b) * 60

/-- The main theorem stating that two pumps with rates of 6 and 9 hours 
    will take 216 minutes to empty a pool together -/
theorem two_pumps_combined_time :
  let pump_a : PumpRate := ⟨6, by norm_num⟩
  let pump_b : PumpRate := ⟨9, by norm_num⟩
  combinedTime pump_a pump_b = 216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_pumps_combined_time_l886_88667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_speed_increase_l886_88611

/-- Calculates the percentage increase between two speeds -/
noncomputable def percentageIncrease (oldSpeed newSpeed : ℝ) : ℝ :=
  (newSpeed - oldSpeed) / oldSpeed * 100

theorem marathon_speed_increase :
  let mondaySpeed : ℝ := 10
  let thursdaySpeed : ℝ := mondaySpeed * 1.5
  let fridaySpeed : ℝ := 24
  percentageIncrease thursdaySpeed fridaySpeed = 60 := by
    -- Unfold the definitions
    unfold percentageIncrease
    -- Simplify the expression
    simp
    -- The rest of the proof would go here
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_marathon_speed_increase_l886_88611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_walked_l886_88681

/-- The side length of a square with area 400 m² -/
noncomputable def square_side : ℝ := Real.sqrt 400

/-- The number of squares in each row and column of the grid -/
def grid_size : ℕ := 4

/-- The length of the outer perimeter of the grid -/
noncomputable def outer_perimeter : ℝ := 4 * grid_size * square_side

/-- The number of inner edges in the grid -/
def inner_edges : ℕ := 2 * grid_size * (grid_size - 1)

/-- The total length of all inner edges -/
noncomputable def inner_length : ℝ := ↑inner_edges * square_side

/-- The theorem stating the total distance walked -/
theorem total_distance_walked : outer_perimeter + inner_length = 800 := by
  -- The proof is omitted for now
  sorry

#eval grid_size
#eval inner_edges

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_distance_walked_l886_88681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l886_88646

theorem function_inequality (f g : ℝ → ℝ) (a b x : ℝ) :
  (∀ y ∈ Set.Icc a b, DifferentiableAt ℝ f y ∧ DifferentiableAt ℝ g y) →
  (∀ y ∈ Set.Icc a b, deriv f y > deriv g y) →
  a < x →
  x < b →
  f x + g a > g x + f a :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_inequality_l886_88646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_S_ratio_is_negative_one_fifth_l886_88615

open BigOperators

def S₁ (n : ℕ) : ℚ := ∑ k in Finset.range n, (-1)^((k + 1) % 3) / (2 : ℚ)^(2019 - k)

def S₂ (n : ℕ) : ℚ := ∑ k in Finset.range n, (-1)^((k + 1) % 3) / (2 : ℚ)^(k + 1)

theorem S_ratio_is_negative_one_fifth : 
  (S₁ 2019) / (S₂ 2019) = -1/5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_S_ratio_is_negative_one_fifth_l886_88615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_is_2231_25_l886_88659

/-- Represents the running contest between Mickey, Johnny, Alex, and Lea -/
structure RunningContest where
  mickey_block : ℚ
  johnny_block : ℚ
  alex_block : ℚ
  lea_block : ℚ
  johnny_laps : ℕ
  lea_laps : ℕ

/-- Calculates the average distance run by each participant -/
def average_distance (contest : RunningContest) : ℚ :=
  let mickey_laps := contest.johnny_laps / 2
  let alex_laps := mickey_laps + 1 + 2 * contest.lea_laps
  let total_distance := contest.johnny_block * contest.johnny_laps +
                        contest.mickey_block * mickey_laps +
                        contest.alex_block * alex_laps +
                        contest.lea_block * contest.lea_laps
  total_distance / 4

/-- Theorem stating that the average distance run is 2231.25 meters -/
theorem average_distance_is_2231_25 (contest : RunningContest)
  (h1 : contest.mickey_block = 250)
  (h2 : contest.johnny_block = 300)
  (h3 : contest.alex_block = 275)
  (h4 : contest.lea_block = 280)
  (h5 : contest.johnny_laps = 8)
  (h6 : contest.lea_laps = 5) :
  average_distance contest = 2231.25 := by
  sorry

#eval average_distance {
  mickey_block := 250,
  johnny_block := 300,
  alex_block := 275,
  lea_block := 280,
  johnny_laps := 8,
  lea_laps := 5
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_distance_is_2231_25_l886_88659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tonys_gas_expenses_eq_91_84_l886_88685

/-- Calculates Tony's total gas expenses for four weeks -/
noncomputable def tonys_gas_expenses : ℝ :=
  let car_efficiency : ℝ := 25 -- miles per gallon
  let work_trip : ℝ := 50 -- miles round trip
  let work_days : ℕ := 5 -- days per week
  let family_visit : ℝ := 15 -- miles extra
  let family_visits : ℕ := 2 -- times per week
  let tank_capacity : ℝ := 10 -- gallons
  let gas_prices : List ℝ := [2, 2.2, 1.9, 2.1] -- dollars per gallon for weeks 1-4

  let weekly_miles : ℝ := work_trip * work_days + family_visit * family_visits
  let weekly_gallons : ℝ := weekly_miles / car_efficiency

  let weekly_expenses : List ℝ := gas_prices.map (· * weekly_gallons)
  weekly_expenses.sum

/-- Theorem stating that Tony's gas expenses for four weeks equal $91.84 -/
theorem tonys_gas_expenses_eq_91_84 : 
  tonys_gas_expenses = 91.84 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tonys_gas_expenses_eq_91_84_l886_88685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_gain_percent_l886_88610

/-- Calculates the gain percentage during a clearance sale given the original selling price,
    original gain percentage, and discount percentage. -/
theorem clearance_sale_gain_percent
  (original_price : ℝ)
  (original_gain_percent : ℝ)
  (discount_percent : ℝ)
  (h1 : original_price = 30)
  (h2 : original_gain_percent = 25)
  (h3 : discount_percent = 10) :
  let cost_price := original_price / (1 + original_gain_percent / 100)
  let sale_price := original_price * (1 - discount_percent / 100)
  let new_gain_percent := (sale_price - cost_price) / cost_price * 100
  new_gain_percent = 12.5 := by
  sorry

#check clearance_sale_gain_percent

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clearance_sale_gain_percent_l886_88610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_is_centroid_l886_88628

/-- Triangle ABC with point G satisfying the given condition -/
structure TriangleWithPoint (V : Type*) [AddCommGroup V] [Module ℝ V] :=
  (A B C G : V)
  (sum_vectors : A - G + (B - G) + (C - G) = 0)

/-- The centroid of a triangle -/
noncomputable def centroid {V : Type*} [AddCommGroup V] [Module ℝ V] (A B C : V) : V :=
  (1/3 : ℝ) • (A + B + C)

/-- Theorem: If G satisfies the given condition, then it is the centroid of the triangle -/
theorem is_centroid {V : Type*} [AddCommGroup V] [Module ℝ V] (t : TriangleWithPoint V) :
  t.G = centroid t.A t.B t.C := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_is_centroid_l886_88628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_is_5_l886_88602

/-- An arithmetic sequence with common difference d -/
noncomputable def arithmetic_sequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  fun n ↦ a₁ + (n - 1 : ℝ) * d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S_n (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a₁ + (n - 1 : ℝ) * d) / 2

/-- The solution set of dx^2 + 2a₁x ≥ 0 is [0,9] -/
def solution_set (d : ℝ) (a₁ : ℝ) : Prop :=
  ∀ x, d * x^2 + 2 * a₁ * x ≥ 0 ↔ 0 ≤ x ∧ x ≤ 9

theorem max_S_n_is_5 (d : ℝ) (a₁ : ℝ) :
  d < 0 →
  solution_set d a₁ →
  ∃ (n : ℕ), n > 0 ∧ ∀ (m : ℕ), m > 0 → S_n a₁ d n ≥ S_n a₁ d m ∧ n = 5 :=
by
  sorry

#check max_S_n_is_5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_S_n_is_5_l886_88602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_A_to_C_l886_88676

/-- Represents a rectangular region with a perimeter and a length-to-width ratio. -/
structure Rectangle where
  perimeter : ℝ
  lengthWidthRatio : ℝ

/-- Calculates the width of a rectangle given its perimeter and length-to-width ratio. -/
noncomputable def calculateWidth (rect : Rectangle) : ℝ :=
  rect.perimeter / (2 * (rect.lengthWidthRatio + 1))

/-- Calculates the area of a rectangle given its perimeter and length-to-width ratio. -/
noncomputable def calculateArea (rect : Rectangle) : ℝ :=
  let width := calculateWidth rect
  width * (rect.lengthWidthRatio * width)

/-- The main theorem stating the ratio of areas between regions A and C. -/
theorem area_ratio_A_to_C : 
  let rectA : Rectangle := { perimeter := 16, lengthWidthRatio := 2 }
  let rectB : Rectangle := { perimeter := 20, lengthWidthRatio := 2 }
  (calculateArea rectA) / (calculateArea rectB) = 16 / 25 := by
  sorry

#check area_ratio_A_to_C

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_ratio_A_to_C_l886_88676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_is_27pi_l886_88622

-- Define the equation of the region
def region_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 8*x - 6*y = 2

-- Define the area of the region
noncomputable def region_area : ℝ := 27 * Real.pi

-- Theorem statement
theorem region_area_is_27pi :
  ∃ (center_x center_y radius : ℝ),
    (∀ (x y : ℝ), region_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) ∧
    region_area = Real.pi * radius^2 := by
  -- Provide the center and radius
  let center_x := -4
  let center_y := 3
  let radius := 3 * Real.sqrt 3
  
  -- Assert the existence of these values
  use center_x, center_y, radius
  
  constructor
  
  -- Prove the equivalence of the equations
  · sorry
  
  -- Prove the area equality
  · sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_region_area_is_27pi_l886_88622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l886_88675

/-- Mary's walking scenario -/
structure WalkingScenario where
  uphill_distance : ℝ
  downhill_distance : ℝ
  uphill_time : ℝ
  downhill_time : ℝ

/-- Calculate the average speed for a round trip -/
noncomputable def average_speed (w : WalkingScenario) : ℝ :=
  (w.uphill_distance + w.downhill_distance) / (w.uphill_time + w.downhill_time)

/-- Mary's specific walking scenario -/
noncomputable def mary_walk : WalkingScenario :=
  { uphill_distance := 1.5,
    downhill_distance := 1.5,
    uphill_time := 45 / 60,  -- Convert 45 minutes to hours
    downhill_time := 15 / 60 }  -- Convert 15 minutes to hours

/-- Theorem: Mary's average speed for the round trip is 3.0 km/hr -/
theorem mary_average_speed :
  average_speed mary_walk = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_average_speed_l886_88675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_on_reciprocal_curve_l886_88688

/-- Area bounded by OA, OB, and arc AB -/
noncomputable def area_OAB_arc (A B O : ℝ × ℝ) : ℝ := sorry

/-- Area bounded by AH_A, BH_B, x-axis, and arc AB -/
noncomputable def area_AH_ABH_B_xaxis_arc (A B H_A H_B : ℝ × ℝ) : ℝ := sorry

/-- The area bounded by OA, OB, and arc AB is equal to the area bounded by AH_A, BH_B, x-axis, and arc AB for any two points on y = 1/x -/
theorem equal_areas_on_reciprocal_curve (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : x₁ < x₂) :
  let f : ℝ → ℝ := fun x ↦ 1 / x
  let A : ℝ × ℝ := (x₁, f x₁)
  let B : ℝ × ℝ := (x₂, f x₂)
  let O : ℝ × ℝ := (0, 0)
  let H_A : ℝ × ℝ := (x₁, 0)
  let H_B : ℝ × ℝ := (x₂, 0)
  area_OAB_arc A B O = area_AH_ABH_B_xaxis_arc A B H_A H_B :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_areas_on_reciprocal_curve_l886_88688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_symmetry_l886_88671

theorem sin_graph_symmetry (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π / 2) :
  (∃! c : ℝ, π / 6 < c ∧ c < π / 3 ∧
    (∀ x : ℝ, Real.sin (2 * (c - x) + φ) = Real.sin (2 * (c + x) + φ))) →
  φ = 5 * π / 12 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_symmetry_l886_88671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_of_sine_function_l886_88698

noncomputable section

-- Define the original function f
noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + 2 * Real.pi / 3)

-- Define the translated function g
noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 6)

-- Theorem statement
theorem translation_of_sine_function :
  ∀ x : ℝ, g x = 2 * Real.sin (2 * x + Real.pi / 3) :=
by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_of_sine_function_l886_88698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l886_88692

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 2 - x^2 / 4 = 1

/-- The equations of the asymptotes -/
def asymptote_equations (x y : ℝ) : Prop :=
  y = Real.sqrt 2 / 2 * x ∨ y = -(Real.sqrt 2 / 2 * x)

/-- Theorem: The asymptotes of the given hyperbola -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, hyperbola_equation x y → asymptote_equations x y :=
by
  sorry

#check hyperbola_asymptotes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_asymptotes_l886_88692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_person_is_knight_l886_88684

/-- Represents a person in the chain, either a knight or a liar -/
inductive Person
| Knight : Person
| Liar : Person

/-- Represents the chain of people -/
def Chain := List Person

/-- Function to determine if a number is even -/
def isEven (n : ℕ) : Bool := n % 2 = 0

/-- Function to simulate the number transformation through the chain -/
def transformNumber (chain : Chain) (start : ℕ) : ℕ :=
  chain.foldl (fun acc person =>
    match person with
    | Person.Knight => acc
    | Person.Liar => if isEven acc then acc + 1 else acc - 1
  ) start

theorem last_person_is_knight 
  (chain : Chain) 
  (first_game_start first_game_end : ℕ)
  (second_game_start second_game_end : ℕ)
  (h1 : chain.head? = some Person.Knight)
  (h2 : transformNumber chain first_game_start = first_game_end)
  (h3 : transformNumber chain.reverse second_game_start = second_game_end)
  (h4 : isEven first_game_start ≠ isEven first_game_end)
  (h5 : isEven second_game_start ≠ isEven second_game_end) :
  chain.getLast? = some Person.Knight := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_person_is_knight_l886_88684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l886_88670

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_of_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

def geometric_sequence (x y z : ℝ) : Prop :=
  y * y = x * z

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (m : ℕ) :
  arithmetic_sequence a →
  a 3 + sum_of_arithmetic_sequence a 5 = 18 →
  a 5 = 7 →
  geometric_sequence (a 3) (a 6) (a m) →
  m = 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_problem_l886_88670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2013_less_than_sin_2013_l886_88601

theorem cos_2013_less_than_sin_2013 : Real.cos (2013 * π / 180) < Real.sin (2013 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2013_less_than_sin_2013_l886_88601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_one_implies_a_eq_neg_two_l886_88607

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/2) * a * x^3 - (3/2) * x^2 + (3/2) * a^2 * x

noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (3/2) * a * x^2 - 3 * x + (3/2) * a^2

theorem max_at_one_implies_a_eq_neg_two (a : ℝ) :
  (∀ x : ℝ, f a x ≤ f a 1) →
  f_derivative a 1 = 0 →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_at_one_implies_a_eq_neg_two_l886_88607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_is_half_hypotenuse_l886_88677

/-- A right triangle with sides 9, 12, and 15 -/
structure RightTriangle where
  de : ℝ
  df : ℝ
  ef : ℝ
  is_right : de^2 = df^2 + ef^2
  de_eq : de = 15
  df_eq : df = 9
  ef_eq : ef = 12

/-- The distance from a vertex to the midpoint of the opposite side in a right triangle -/
noncomputable def distanceToMidpoint (t : RightTriangle) : ℝ := t.de / 2

theorem distance_to_midpoint_is_half_hypotenuse (t : RightTriangle) :
  distanceToMidpoint t = 7.5 := by
  -- Unfold the definition of distanceToMidpoint
  unfold distanceToMidpoint
  -- Use the fact that t.de = 15
  rw [t.de_eq]
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_midpoint_is_half_hypotenuse_l886_88677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_age_problem_solution_l886_88679

/-- Given a person's age A and the number of years hence x, this function
    represents the equation described in the problem. -/
noncomputable def age_equation (A : ℝ) (x : ℝ) : ℝ :=
  (1/2) * (8 * (A + x) - 8 * (A - 8))

/-- Theorem stating that for A = 64, the solution to the age equation is x = 8. -/
theorem age_problem_solution :
  ∃ (x : ℝ), age_equation 64 x = 64 ∧ x = 8 := by
  use 8
  constructor
  · -- Prove age_equation 64 8 = 64
    simp [age_equation]
    ring
  · -- Prove x = 8
    rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_age_problem_solution_l886_88679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_almost_square_divisible_2010_l886_88695

/-- An almost-square is a rectangle whose sides differ by 1 -/
def AlmostSquare (a b : ℕ) : Prop := (a = b + 1) ∨ (b = a + 1)

/-- A rectangle can be divided into n almost-squares if its area is equal to the sum of areas of n almost-squares -/
def DivisibleIntoAlmostSquares (w h n : ℕ) : Prop :=
  ∃ (sizes : Finset (ℕ × ℕ)), sizes.card = n ∧
    (∀ p ∈ sizes, AlmostSquare p.1 p.2) ∧
    (sizes.sum (fun p => p.1 * p.2) = w * h)

/-- There exists an almost-square that can be divided into 2010 almost-squares -/
theorem exists_almost_square_divisible_2010 :
  ∃ (w h : ℕ), AlmostSquare w h ∧ DivisibleIntoAlmostSquares w h 2010 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_almost_square_divisible_2010_l886_88695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integrality_l886_88683

/-- Given a non-zero real number a and a sequence S defined as S n = aⁿ + a⁻ⁿ,
    if there exists a natural number p such that S p and S (p+1) are integers,
    then S n is an integer for all n. -/
theorem sequence_integrality (a : ℝ) (ha : a ≠ 0) :
  let S : ℤ → ℝ := λ n => a^n + a^(-n)
  ∃ p : ℕ, (∃ m : ℤ, S p = m) ∧ (∃ k : ℤ, S (p + 1) = k) →
  ∀ n : ℤ, ∃ l : ℤ, S n = l :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_integrality_l886_88683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_and_briefcase_cost_l886_88644

/-- The total cost of a pen and a briefcase, where the pen costs $4 and the briefcase costs five times the price of the pen, is $24. -/
theorem pen_and_briefcase_cost (pen_cost briefcase_multiplier : ℕ) : 
  pen_cost = 4 → 
  briefcase_multiplier = 5 → 
  pen_cost + briefcase_multiplier * pen_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pen_and_briefcase_cost_l886_88644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l886_88618

-- Define the function
noncomputable def f (x : ℝ) : ℝ := 1 / (x - 2)

-- State the theorem
theorem f_domain : 
  ∀ x : ℝ, x ≠ 2 ↔ ∃ y : ℝ, f x = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_l886_88618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_R_in_triangle_l886_88687

theorem cos_R_in_triangle (P Q R : ℝ) : 
  0 < P ∧ 0 < Q ∧ 0 < R ∧ P + Q + R = Real.pi →  -- Triangle condition
  Real.sin P = 4/5 →
  Real.cos Q = 3/5 →
  Real.cos R = 7/25 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_R_in_triangle_l886_88687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l886_88696

def sequence_a : ℕ → ℤ
  | 0 => 3
  | n + 1 => 4 * sequence_a n + 3

theorem sequence_a_formula (n : ℕ) : sequence_a n = 4^n - 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_a_formula_l886_88696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_power_theorem_l886_88629

theorem derivative_power_theorem (k n : ℕ) (h_k : k > 0) (h_n : n > 0) :
  let f (x : ℝ) := (x^k - 1)⁻¹
  let p (x : ℝ) := (x^k - 1)^(n + 1) * (deriv^[n] f) x
  p 1 = (-1)^(n + 1) * (n! : ℝ) * k^n :=
sorry

#check derivative_power_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_power_theorem_l886_88629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l886_88689

noncomputable section

/-- The distance between two points in 2D space -/
def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The x-coordinate of point A -/
def a_x : ℝ := -4

/-- The y-coordinate of point A -/
def a_y : ℝ := 0

/-- The x-coordinate of point B -/
def b_x : ℝ := 2

/-- The y-coordinate of point B -/
def b_y : ℝ := 6

/-- The x-coordinate of the point we want to prove is equidistant -/
def equidistant_x : ℝ := 2

/-- The y-coordinate of the point we want to prove is equidistant (on x-axis) -/
def equidistant_y : ℝ := 0

theorem equidistant_point : 
  distance a_x a_y equidistant_x equidistant_y = 
  distance b_x b_y equidistant_x equidistant_y := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equidistant_point_l886_88689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_theorem_l886_88636

-- Define the solution sets A and B as functions of b and c
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (b c : ℝ) : Set ℝ := {x | x^2 + b*x + c ≤ 0}

-- State the theorem
theorem solution_sets_theorem : 
  ∃ b c : ℝ, A ∪ B b c = Set.univ ∧ A ∩ B b c = Set.Ioc 3 4 → b + c = -7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_sets_theorem_l886_88636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_samovar_cools_faster_l886_88612

structure Samovar where
  volume : ℝ
  surface_area : ℝ
  temperature : ℝ

/-- The cooling rate of a samovar is proportional to its surface area to volume ratio -/
noncomputable def cooling_rate (s : Samovar) : ℝ := s.surface_area / s.volume

/-- Given two samovars with the same shape, material, and initial temperature,
    the smaller one cools down faster -/
theorem smaller_samovar_cools_faster (small large : Samovar) 
    (h_shape : small.surface_area / small.volume > large.surface_area / large.volume)
    (h_temp : small.temperature = large.temperature)
    (h_material : True) -- Assumption that both samovars are made of the same material
    : cooling_rate small > cooling_rate large := by
  -- Unfold the definition of cooling_rate
  unfold cooling_rate
  -- The proof follows directly from h_shape
  exact h_shape

#check smaller_samovar_cools_faster

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_samovar_cools_faster_l886_88612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_plus_alpha_implies_cos2alpha_plus_sin2alpha_l886_88634

theorem tan_pi_plus_alpha_implies_cos2alpha_plus_sin2alpha
  (α : ℝ)
  (h : Real.tan (π + α) = 2) :
  Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_pi_plus_alpha_implies_cos2alpha_plus_sin2alpha_l886_88634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_difference_range_arithmetic_geometric_difference_existence_l886_88651

-- Part 1
theorem arithmetic_geometric_difference_range (a b : ℕ → ℝ) (d : ℝ) :
  (∀ n, a n = 0 + (n - 1) * d) →
  (∀ n, b n = 2^(n - 1)) →
  (∀ n : Fin 4, |a n.val.succ - b n.val.succ| ≤ 1) →
  7/3 ≤ d ∧ d ≤ 5/2 := by sorry

-- Part 2
theorem arithmetic_geometric_difference_existence (a b : ℕ → ℝ) (m : ℕ) (q : ℝ) (b₁ : ℝ) :
  m > 0 →
  1 < q →
  q < 2^(1/m) →
  b₁ > 0 →
  (∀ n, a n = b₁ + (n - 1) * (a 2 - b₁)) →
  (∀ n, b n = b₁ * q^(n - 1)) →
  ∃ d : ℝ, (∀ n ∈ Finset.range m, |a (n + 2) - b (n + 2)| ≤ b₁) ∧
           b₁ * q - 2 * b₁ ≤ d ∧ d ≤ (b₁ * q^m) / m := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_difference_range_arithmetic_geometric_difference_existence_l886_88651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l886_88652

theorem trigonometric_equation_solution (α β : ℝ) 
  (h1 : 0 < α ∧ α < π / 2) 
  (h2 : 0 < β ∧ β < π / 2) 
  (h3 : Real.tan (α + β) ^ 2 + (Real.tan (α + β))⁻¹ ^ 2 = 3 * α - α ^ 2 - 1 / 4) : 
  β = (3 * π - 6) / 4 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_equation_solution_l886_88652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_two_and_four_l886_88672

/-- A random variable following a normal distribution with mean 2 and some variance σ² -/
noncomputable def ξ : Prob → ℝ := sorry

/-- The probability density function of ξ -/
noncomputable def pdf_ξ : ℝ → ℝ := sorry

/-- The cumulative distribution function of ξ -/
noncomputable def cdf_ξ : ℝ → ℝ := sorry

/-- The mean of the distribution is 2 -/
axiom mean_ξ : ∫ x in Set.univ, x * pdf_ξ x = 2

/-- The probability that ξ is less than or equal to 0 is 0.2 -/
axiom prob_le_zero : cdf_ξ 0 = 0.2

/-- Theorem: The probability that ξ is between 2 and 4 is 0.3 -/
theorem prob_between_two_and_four : cdf_ξ 4 - cdf_ξ 2 = 0.3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_between_two_and_four_l886_88672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l886_88682

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.cos x

-- Define the domain
def domain : Set ℝ := { x | -Real.pi/3 ≤ x ∧ x ≤ 2*Real.pi/3 }

-- State the theorem
theorem f_range : 
  ∀ y ∈ Set.range f, y ∈ Set.Icc (1/4) (5/4) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l886_88682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_to_multiple_of_five_l886_88653

theorem least_positive_integer_to_multiple_of_five : ∃! n : ℕ, n > 0 ∧
  (∀ m : ℕ, m > 0 → (365 + m) % 5 = 0 → n ≤ m) ∧ (365 + n) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_positive_integer_to_multiple_of_five_l886_88653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_batches_for_math_club_l886_88668

/-- Calculates the minimum number of complete batches of cookies needed for a math club event --/
noncomputable def min_cookie_batches (total_students : ℕ) (attendance_drop : ℚ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) : ℕ :=
  let expected_attendance := (total_students : ℚ) * (1 - attendance_drop)
  let total_cookies_needed := expected_attendance * (cookies_per_student : ℚ)
  (Int.ceil (total_cookies_needed / (cookies_per_batch : ℚ))).toNat

/-- Proves that 14 batches are needed for the given conditions --/
theorem cookie_batches_for_math_club : 
  min_cookie_batches 150 (40 / 100) 3 20 = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookie_batches_for_math_club_l886_88668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_l886_88625

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := sin x + a * cos x

def is_strictly_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y ∨ f y < f x

theorem monotonicity_condition (a : ℝ) :
  is_strictly_monotonic (f a) (2*π/3) (7*π/6) ↔ 
  -Real.sqrt 3 / 3 ≤ a ∧ a ≤ Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonicity_condition_l886_88625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l886_88680

-- Define a type for colors
inductive Color
| Black
| White

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define the distance between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- The main theorem
theorem same_color_points_exist (coloring : Coloring) (d : ℝ) :
  ∃ (p1 p2 : Point), coloring p1 = coloring p2 ∧ distance p1 p2 = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_color_points_exist_l886_88680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_seven_l886_88620

-- Define the functions h and j
noncomputable def h (x : ℝ) : ℝ := x + 3
noncomputable def j (x : ℝ) : ℝ := x / 4

-- Define the inverse functions
noncomputable def h_inv (x : ℝ) : ℝ := x - 3
noncomputable def j_inv (x : ℝ) : ℝ := 4 * x

-- State the theorem
theorem composition_equals_seven :
  h (j_inv (h_inv (h_inv (j (h 25))))) = 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_composition_equals_seven_l886_88620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_scalar_multiple_l886_88633

variable {α : Type*} [NormedAddCommGroup α] [Module ℝ α]

theorem norm_scalar_multiple (v : α) :
  ‖v‖ = 5 → ‖(-6 : ℝ) • v‖ = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_norm_scalar_multiple_l886_88633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_combinations_count_l886_88617

def num_students : ℕ := 5
def num_food_options : ℕ := 4

structure Food where
  name : String
deriving Repr, BEq

def rice : Food := ⟨"rice"⟩
def steamed_buns : Food := ⟨"steamed buns"⟩
def stuffed_buns : Food := ⟨"stuffed buns"⟩
def noodles : Food := ⟨"noodles"⟩

def food_options : List Food := [rice, steamed_buns, stuffed_buns, noodles]

structure Student where
  name : String
  food_choice : Food
deriving Repr, BEq

def student_A : Student := ⟨"A", stuffed_buns⟩ -- Initial assignment, will be constrained later

def is_valid_combination (students : List Student) : Prop :=
  (students.length = num_students) ∧
  (∀ f ∈ food_options, ∃ s ∈ students, s.food_choice = f) ∧
  ((students.filter (λ s => s.food_choice == steamed_buns)).length = 1) ∧
  (∀ s ∈ students, s.name = "A" → s.food_choice ≠ rice)

noncomputable def count_valid_combinations : ℕ := sorry

theorem valid_combinations_count :
  count_valid_combinations = 132 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_combinations_count_l886_88617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_theorem_l886_88624

/-- The radius of an inscribed cylinder in a cone -/
noncomputable def inscribed_cylinder_radius (cone_diameter : ℝ) (cone_altitude : ℝ) : ℝ :=
  20 / 9

/-- Theorem: The radius of a right circular cylinder inscribed in a right circular cone -/
theorem inscribed_cylinder_radius_theorem (cone_diameter cone_altitude : ℝ) 
  (h1 : cone_diameter = 8)
  (h2 : cone_altitude = 10) :
  inscribed_cylinder_radius cone_diameter cone_altitude = 20 / 9 := by
  sorry

#check inscribed_cylinder_radius_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_cylinder_radius_theorem_l886_88624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_equation_l886_88660

noncomputable section

-- Define the given curve
def given_curve (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) = (Real.sqrt 2 / 2) * (2 - x)

-- Define the focus of the given curve
def given_curve_focus : ℝ × ℝ := (1, 0)

-- Define the point that lies on hyperbola C
def point_on_C : ℝ × ℝ := (3, -(2 * Real.sqrt 39 / 3))

-- Define the equation of hyperbola C
def hyperbola_C (x y : ℝ) : Prop :=
  3 * x^2 - (3/2) * y^2 = 1

-- Theorem statement
theorem hyperbola_C_equation :
  (∃ (f : ℝ × ℝ), f = given_curve_focus ∧ 
    (∃ (a b : ℝ), hyperbola_C a b ∧ 
      ((a - f.1)^2 / (3 * (1/3)) + (b - f.2)^2 / ((3/2) * (1/3)) = 1))) ∧
  (hyperbola_C point_on_C.1 point_on_C.2) →
  ∀ (x y : ℝ), hyperbola_C x y ↔ 3 * x^2 - (3/2) * y^2 = 1 :=
by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_C_equation_l886_88660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_three_l886_88673

def a (n : ℕ) : ℕ := 
  if n > 0 then Nat.factorial (n + 8) / Nat.factorial (n - 1) else 0

def digit_sum (n : ℕ) : ℕ := 
  let s := toString n
  let t := s.dropRightWhile (· == '0')
  t.foldl (fun acc c => acc + c.toNat - '0'.toNat) 0

def is_smallest_k (k : ℕ) : Prop :=
  k > 0 ∧
  (digit_sum (a k) % 7 = 0) ∧ 
  ∀ m : ℕ, 0 < m ∧ m < k → digit_sum (a m) % 7 ≠ 0

theorem smallest_k_is_three : is_smallest_k 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_is_three_l886_88673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l886_88665

-- Define the ellipse C
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 / 5 + p.2^2 = 1}

-- Define the upper vertex B
def B : ℝ × ℝ := (0, 1)

-- Define a point P on the ellipse
noncomputable def P : ℝ → ℝ × ℝ := λ θ => (Real.sqrt 5 * Real.cos θ, Real.sin θ)

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_PB :
  ∃ (max_dist : ℝ), max_dist = 5/2 ∧
  ∀ θ : ℝ, distance (P θ) B ≤ max_dist := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PB_l886_88665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l886_88662

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- Given conditions for the triangle -/
def SpecialTriangle (t : Triangle) : Prop :=
  t.c = 2 ∧ t.C = Real.pi/3 ∧ 2 * Real.sin (2 * t.A) + Real.sin (2 * t.B + t.C) = Real.sin t.C

theorem triangle_properties (t : Triangle) (h : SpecialTriangle t) :
  (1/2 * t.a * t.b * Real.sin t.C = 2 * Real.sqrt 3 / 3) ∧
  (∀ t' : Triangle, SpecialTriangle t' → t'.a + t'.b + t'.c ≤ 6) := by
  sorry

#check triangle_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l886_88662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_wheels_count_l886_88619

theorem race_wheels_count (total_racers : ℕ) (bicycle_fraction : ℚ) : 
  total_racers = 40 →
  bicycle_fraction = 3 / 5 →
  (bicycle_fraction * total_racers).floor * 2 + (total_racers - (bicycle_fraction * total_racers).floor) * 3 = 96 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_wheels_count_l886_88619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_theorem_l886_88648

/-- Represents a solution with a given volume and alcohol percentage -/
structure Solution where
  volume : ℝ
  alcohol_percentage : ℝ

/-- Calculates the alcohol volume in a solution -/
noncomputable def alcohol_volume (s : Solution) : ℝ :=
  s.volume * (s.alcohol_percentage / 100)

/-- Calculates the percentage of alcohol in a mixture of two solutions -/
noncomputable def mixture_alcohol_percentage (s1 s2 : Solution) : ℝ :=
  let total_volume := s1.volume + s2.volume
  let total_alcohol := alcohol_volume s1 + alcohol_volume s2
  (total_alcohol / total_volume) * 100

/-- The main theorem stating that mixing 200 mL of 10% solution with 50 mL of 30% solution results in 14% solution -/
theorem mixture_theorem :
  let x := Solution.mk 200 10
  let y := Solution.mk 50 30
  mixture_alcohol_percentage x y = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mixture_theorem_l886_88648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_l886_88674

/-- Defines the concept of an axis of symmetry for a function --/
def IsAxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

/-- Theorem stating that x = -π/2 is an axis of symmetry for sin(2x + 5π/2) --/
theorem axis_of_symmetry_sin :
  IsAxisOfSymmetry (λ x : ℝ => Real.sin (2*x + 5*Real.pi/2)) (-Real.pi/2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_sin_l886_88674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_equal_l886_88600

noncomputable def f (a b : ℕ) (x : ℝ) : ℝ := x^2 + 2*a*x + b*(2:ℝ)^x

theorem f_zeros_equal (a b : ℕ) :
  (∃ x : ℝ, f a b x = 0) ∧
  (∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) →
  ((a = 0 ∧ b = 0) ∨ (a = 1 ∧ b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_zeros_equal_l886_88600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l886_88645

-- Define the triangle
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  b = 4 ∧ c = 5 ∧ A = Real.pi/3

-- State the theorem
theorem triangle_properties (a b c A B C : ℝ) 
  (h : Triangle a b c A B C) : 
  a = Real.sqrt 21 ∧ 
  (1/2 * b * c * Real.sin A = 5 * Real.sqrt 3) ∧
  (Real.sin (2 * B) = (4 * Real.sqrt 3) / 7) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l886_88645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completed_in_five_days_l886_88637

/-- Represents the work rate of a person as a fraction of the total work per day -/
def WorkRate := Rat

/-- Calculates the amount of work done by a group in a given number of days -/
def workDone (rates : List Rat) (days : Nat) : Rat :=
  (rates.sum * days)

/-- Represents the problem setup -/
structure WorkProblem where
  rateA : Rat
  rateB : Rat
  rateC : Rat
  initialDays : Nat
  middleDays : Nat

/-- The main theorem to be proved -/
theorem work_completed_in_five_days (p : WorkProblem) : 
  p.rateA = 1/4 → 
  p.rateB = 1/10 → 
  p.rateC = 1/12 → 
  p.initialDays = 2 → 
  p.middleDays = 3 → 
  workDone [p.rateA, p.rateB, p.rateC] p.initialDays + 
  workDone [p.rateB, p.rateC] p.middleDays ≥ 1 := by
  sorry

#eval (1 : Rat) / 4
#eval (1 : Rat) / 10
#eval (1 : Rat) / 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_work_completed_in_five_days_l886_88637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_eq_perfect_square_sequences_exist_l886_88627

def a : ℕ → ℕ
  | 0 => 1  -- Added base case for 0
  | 1 => 9
  | 2 => 89
  | (n+3) => 10 * a (n+2) - a (n+1)

noncomputable def a_closed_form (n : ℕ) : ℝ :=
  ((2 + Real.sqrt 6) / (4 * Real.sqrt 6)) * (5 + 2 * Real.sqrt 6)^n -
  ((2 - Real.sqrt 6) / (4 * Real.sqrt 6)) * (5 - 2 * Real.sqrt 6)^n

theorem a_closed_form_eq (n : ℕ) :
  (a n : ℝ) = a_closed_form n := by
  sorry

theorem perfect_square (n : ℕ) :
  ∃ k : ℕ, (a n * a (n+1) - 1) / 2 = k^2 := by
  sorry

theorem sequences_exist :
  ∃ x y : ℕ → ℕ, ∀ n : ℕ, a n = (x n^2 + 2) / (2 * (x n + y n)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_closed_form_eq_perfect_square_sequences_exist_l886_88627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l886_88621

theorem negation_of_sin_leq_one :
  (¬ (∀ x : ℝ, Real.sin x ≤ 1)) ↔ (∃ x₀ : ℝ, Real.sin x₀ > 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_sin_leq_one_l886_88621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_platform_length_is_423_and_third_l886_88650

/-- The length of the first platform given the train's specifications and crossing times -/
noncomputable def first_platform_length (train_length : ℝ) (first_crossing_time : ℝ) (second_platform_length : ℝ) (second_crossing_time : ℝ) : ℝ :=
  (second_crossing_time * (train_length + second_platform_length) - first_crossing_time * train_length) / first_crossing_time

/-- Theorem stating that the length of the first platform is 423 1/3 meters -/
theorem first_platform_length_is_423_and_third (train_length : ℝ) (first_crossing_time : ℝ) (second_platform_length : ℝ) (second_crossing_time : ℝ) 
    (h1 : train_length = 270)
    (h2 : first_crossing_time = 15)
    (h3 : second_platform_length = 250)
    (h4 : second_crossing_time = 20) :
  first_platform_length train_length first_crossing_time second_platform_length second_crossing_time = 423 + 1/3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_platform_length_is_423_and_third_l886_88650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_negative_l886_88626

noncomputable def f (x : ℝ) : ℝ := |Real.log (x + 1) / Real.log (1/2)|

theorem ac_negative (a b c : ℝ) 
  (h1 : -1 < a) (h2 : a < b) (h3 : b < c)
  (h4 : f a > f c) (h5 : f c > f b) : a * c < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ac_negative_l886_88626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_pair_l886_88699

/-- Returns the digits of a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

/-- Checks if two numbers have no common digits -/
def no_common_digits (a b : ℕ) : Prop :=
  ∀ d : ℕ, d < 10 → (d ∈ digits a → d ∉ digits b) ∧ (d ∈ digits b → d ∉ digits a)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem unique_number_pair : 
  ∃! (a b : ℕ), 
    is_two_digit a ∧ 
    is_two_digit b ∧ 
    b = 2 * a ∧ 
    no_common_digits a b ∧
    (∃ (x y : ℕ), x ∈ digits b ∧ y ∈ digits b ∧ x + y ∈ digits a ∧ x - y ∈ digits a) ∧
    a = 17 ∧ 
    b = 34 :=
by
  sorry

#check unique_number_pair

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_number_pair_l886_88699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_theorem_l886_88666

noncomputable def arithmetic_progression (a : ℝ) (d : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => arithmetic_progression a d n + d

noncomputable def geometric_progression (a : ℝ) (r : ℝ) : ℕ → ℝ
  | 0 => a
  | n + 1 => geometric_progression a r n * r

noncomputable def arithmetic_sum (a : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (2 * a + (n - 1 : ℝ) * d) / 2

noncomputable def geometric_product (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a^n * r^(n * (n - 1) / 2)

theorem arithmetic_geometric_theorem :
  ∀ a d : ℝ, arithmetic_progression a d 2 = 0 → arithmetic_sum a d 5 = 0 ∧
  ∀ a r : ℝ, geometric_progression a r 2 = 4 → geometric_product a r 5 = 1024 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_geometric_theorem_l886_88666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_eq_5001_l886_88656

/-- The sum of the alternating series from -1 to 10002 -/
def alternating_sum : ℕ → ℤ
  | 0 => 0
  | n + 1 => alternating_sum n + (if n % 2 = 0 then -(n + 1 : ℤ) else (n + 1 : ℤ))

/-- The number of terms in the series -/
def num_terms : ℕ := 10002

theorem alternating_sum_eq_5001 : alternating_sum num_terms = 5001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alternating_sum_eq_5001_l886_88656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l886_88640

open Set Real

theorem range_of_a (S T : Set ℝ) (a : ℝ) : 
  S = {x : ℝ | 1/2 < (2 : ℝ)^x ∧ (2 : ℝ)^x < 8} →
  T = {x : ℝ | x < a ∨ x > a + 2} →
  S ∪ T = univ →
  a ∈ Ioo (-1) 1 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l886_88640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_monge_circle_l886_88641

/-- The eccentricity of an ellipse given its Monge circle -/
theorem ellipse_eccentricity_from_monge_circle (a b : ℝ) (h1 : a^2 + b^2 = 3*b^2) 
  (h2 : a > 0) (h3 : b > 0) : 
  Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_from_monge_circle_l886_88641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l886_88658

-- Define the points in the Cartesian coordinate system
def O : ℝ × ℝ := (0, 0)

-- Define variables for coordinates
variable (x₁ y₁ x₂ y₂ : ℝ)

def P₁ : ℝ × ℝ := (x₁, y₁)
def P₂ : ℝ × ℝ := (x₂, y₂)

-- Define the conditions
def first_quadrant (P : ℝ × ℝ) : Prop := P.1 > 0 ∧ P.2 > 0

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- State the theorem
theorem triangle_area_is_one (x₁ y₁ x₂ y₂ : ℝ) :
  first_quadrant (P₁ x₁ y₁) →
  first_quadrant (P₂ x₂ y₂) →
  arithmetic_sequence 1 x₁ x₂ 4 →
  geometric_sequence 1 y₁ y₂ 8 →
  triangle_area O (P₁ x₁ y₁) (P₂ x₂ y₂) = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_one_l886_88658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l886_88664

theorem max_true_statements : 
  ∃ (x : ℝ) (true_statements : List Bool),
  let statements := [
    (0 < x^3 ∧ x^3 < 1),
    (x^2 > 1),
    (-1 < x ∧ x < 0),
    (0 < x ∧ x < 1),
    (0 < x - x^3 ∧ x - x^3 < 1)
  ]
  (∀ i, i < statements.length → (true_statements.get! i = true ↔ statements.get! i)) ∧
  true_statements.count true ≤ 3 ∧
  true_statements.count true = 3 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_true_statements_l886_88664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_exp_inequality_l886_88632

theorem log_exp_inequality (a : ℝ) (h : a > 1) :
  Real.log a / Real.log (1/5) < (1/5 : ℝ) ^ a ∧ (1/5 : ℝ) ^ a < a ^ (1/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_exp_inequality_l886_88632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_cardinality_l886_88694

def is_valid_set (A : Finset ℕ) : Prop :=
  ∀ a b k, a ∈ A → b ∈ A → 
    ¬∃ n : ℤ, (a : ℤ) + (b : ℤ) + 30 * k = n * (n + 1)

theorem max_valid_set_cardinality :
  ∃ (A : Finset ℕ), 
    (∀ a ∈ A, a ≤ 29) ∧ 
    is_valid_set A ∧ 
    A.card = 8 ∧
    ∀ (B : Finset ℕ), (∀ b ∈ B, b ≤ 29) → is_valid_set B → B.card ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_valid_set_cardinality_l886_88694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_max_value_of_f_max_value_of_sum_l886_88605

-- Define the function f
def f (x : ℝ) : ℝ := |x - 4| - |x + 2|

-- Theorem for part 1
theorem range_of_a (a : ℝ) :
  (∀ x, f x - a^2 + 5*a ≥ 0) ↔ (2 ≤ a ∧ a ≤ 3) := by
  sorry

-- Theorem for part 2
theorem max_value_of_f :
  ∃ M, M = 6 ∧ ∀ x, f x ≤ M := by
  sorry

-- Theorem for part 3
theorem max_value_of_sum (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 → a + b + c = 6 →
  Real.sqrt (a + 1) + Real.sqrt (b + 2) + Real.sqrt (c + 3) ≤ 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_max_value_of_f_max_value_of_sum_l886_88605


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l886_88614

noncomputable def S (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * (1 - q^n) / (1 - q)

theorem geometric_sequence_ratio (a : ℝ) (q : ℝ) (h1 : a ≠ 0) (h2 : q ≠ 1) :
  S a q 3 + 3 * S a q 2 = 0 → q = -1 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_ratio_l886_88614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l886_88630

def vector_a : Fin 2 → ℝ := ![(-3), 2]
def vector_b (l : ℝ) : Fin 2 → ℝ := ![(-1), l]

def perpendicular (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem perpendicular_vectors (l : ℝ) :
  perpendicular vector_a (vector_b l) → l = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_vectors_l886_88630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_income_ratio_is_four_l886_88609

/-- Represents the ratio of Mindy's income to Mork's income -/
noncomputable def income_ratio (mork_tax_rate mindy_tax_rate combined_tax_rate : ℝ) : ℝ :=
  (combined_tax_rate - mork_tax_rate) / (mindy_tax_rate - combined_tax_rate)

/-- 
Theorem stating that given the tax rates for Mork, Mindy, and their combined rate,
the ratio of Mindy's income to Mork's income is 4.
-/
theorem income_ratio_is_four :
  income_ratio 0.45 0.20 0.25 = 4 := by
  -- Unfold the definition of income_ratio
  unfold income_ratio
  -- Simplify the expression
  simp
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_stating_income_ratio_is_four_l886_88609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_color_exists_l886_88649

-- Define a type for colors
inductive Color : Type
  | Red : Color
  | Blue : Color

-- Define a coloring function
def coloring : ℤ → Color := sorry

-- Define a set of integers of a given color
def colorSet (c : Color) : Set ℤ :=
  {n : ℤ | coloring n = c}

-- The main theorem
theorem infinite_divisible_color_exists :
  ∃ c : Color, ∀ k : ℕ, Set.Infinite {n : ℤ | n ∈ colorSet c ∧ k ∣ (n.natAbs)} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinite_divisible_color_exists_l886_88649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l886_88690

def is_geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem geometric_sequence_middle_term (a : ℝ) :
  (is_geometric_sequence 1 a 16) → (a = -4 ∨ a = 4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_middle_term_l886_88690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l886_88643

/-- Line l in parametric form -/
noncomputable def line_l (φ : ℝ) (t : ℝ) : ℝ × ℝ :=
  (t * Real.sin φ, 1 + t * Real.cos φ)

/-- Curve C in polar form -/
def curve_C (θ : ℝ) (ρ : ℝ) : Prop :=
  ρ * (Real.cos θ)^2 = 4 * Real.sin θ

/-- Distance between intersection points of line l and curve C -/
noncomputable def intersection_distance (φ : ℝ) : ℝ :=
  4 / (Real.sin φ)^2

theorem min_intersection_distance :
  ∀ φ, 0 < φ ∧ φ < π →
  ∀ θ ρ, curve_C θ ρ →
  ∃ t₁ t₂, 
    (line_l φ t₁).1^2 = 4 * (line_l φ t₁).2 ∧
    (line_l φ t₂).1^2 = 4 * (line_l φ t₂).2 ∧
    intersection_distance φ ≥ 4 ∧
    (φ = π/2 → intersection_distance φ = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_intersection_distance_l886_88643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_even_odd_count_l886_88616

def pascal_triangle : ℕ → ℕ → ℕ
| 0, _ => 1
| n+1, 0 => 1
| n+1, k+1 => pascal_triangle n k + pascal_triangle n (k+1)

def is_even (n : ℕ) : Bool := n % 2 = 0

def count_even_odd (rows : ℕ) : ℕ × ℕ :=
  let count := λ r k => if is_even (pascal_triangle r k) then (1, 0) else (0, 1)
  (List.range (rows + 1)).foldl
    (λ (even_count, odd_count) r =>
      (List.range (r + 1)).foldl
        (λ (ec, oc) k =>
          let (e, o) := count r k
          (ec + e, oc + o))
        (even_count, odd_count))
    (0, 0)

theorem pascal_triangle_even_odd_count :
  count_even_odd 14 = (56, 64) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_even_odd_count_l886_88616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sequence_l886_88661

/-- A polynomial with integer coefficients that is positive for non-negative inputs -/
structure PositiveIntPolynomial where
  toFun : ℕ → ℕ
  pos : ∀ x, toFun x > 0

instance : CoeFun PositiveIntPolynomial (λ _ ↦ ℕ → ℕ) where
  coe := PositiveIntPolynomial.toFun

/-- The sequence defined by the recurrence relation -/
def sequenceP (P : PositiveIntPolynomial) : ℕ → ℕ
  | 0 => 0
  | n + 1 => P (sequenceP P n)

/-- The main theorem -/
theorem gcd_sequence (P : PositiveIntPolynomial) (m k : ℕ) :
  Nat.gcd (sequenceP P m) (sequenceP P k) = sequenceP P (Nat.gcd m k) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_gcd_sequence_l886_88661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_tangent_triangle_angles_l886_88613

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Represents the excircle of a triangle opposite to vertex A -/
structure Excircle where
  center : Point
  radius : ℝ

/-- Represents a point where the excircle is tangent to an extended side of the triangle -/
structure TangentPoint where
  point : Point
  side : Line

/-- Theorem about the angles in the triangle formed by tangent points of an excircle -/
theorem excircle_tangent_triangle_angles (abc : Triangle) (ex : Excircle) 
  (d e f : TangentPoint) : 
  ∃ (θ₁ θ₂ θ₃ : ℝ), 
    (θ₁ > 90 ∧ θ₂ < 90 ∧ θ₃ < 90) ∨ 
    (θ₂ > 90 ∧ θ₁ < 90 ∧ θ₃ < 90) ∨ 
    (θ₃ > 90 ∧ θ₁ < 90 ∧ θ₂ < 90) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_excircle_tangent_triangle_angles_l886_88613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_345_l886_88655

noncomputable def f (x : ℝ) : ℝ :=
  if x > 5 then x^3
  else if -5 ≤ x ∧ x ≤ 5 then 2*x - 3
  else 5

theorem f_sum_equals_345 : f (-7) + f 0 + f 7 = 345 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_equals_345_l886_88655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_alpha_plus_pi_fourth_l886_88639

theorem cos_squared_alpha_plus_pi_fourth (α : Real) 
  (h : Real.sin (2 * α) = 2 / 3) : 
  Real.cos (α + Real.pi / 4) ^ 2 = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_squared_alpha_plus_pi_fourth_l886_88639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_perfect_number_has_at_least_three_prime_divisors_l886_88669

/-- A natural number is perfect if the sum of its positive divisors equals twice the number. -/
def IsPerfect (n : ℕ) : Prop :=
  (Finset.sum (Finset.filter (· ∣ n) (Finset.range n)) id) = 2 * n

/-- The number of distinct prime factors of a natural number. -/
noncomputable def numDistinctPrimeFactors (n : ℕ) : ℕ :=
  Finset.card (Nat.factors n).toFinset

/-- Theorem: An odd perfect number has at least 3 distinct prime divisors. -/
theorem odd_perfect_number_has_at_least_three_prime_divisors (n : ℕ) 
  (h_odd : Odd n) (h_perfect : IsPerfect n) : 
  numDistinctPrimeFactors n ≥ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_perfect_number_has_at_least_three_prime_divisors_l886_88669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eulerian_path_iff_two_odd_vertices_double_traversal_exists_triple_traversal_not_exists_l886_88606

-- Define a simple graph
structure Graph (V : Type) where
  adj : V → V → Prop

-- Define a degree of a vertex in a graph
def degree (G : Graph V) (v : V) : ℕ := sorry

-- Define an Eulerian path
def has_eulerian_path (G : Graph V) : Prop := sorry

-- Define a path that traverses each edge exactly n times
def has_n_traversal_path (G : Graph V) (n : ℕ) : Prop := sorry

-- Define connectivity for a graph
def Connected (G : Graph V) : Prop := sorry

-- Theorem 1: Eulerian Path Characterization
theorem eulerian_path_iff_two_odd_vertices {V : Type} (G : Graph V) :
  has_eulerian_path G ↔ (∃ (v w : V), ∀ x : V, degree G x % 2 = 1 → (x = v ∨ x = w)) :=
sorry

-- Theorem 2a: Double Traversal Existence
theorem double_traversal_exists {V : Type} (G : Graph V) (h : Connected G) :
  has_n_traversal_path G 2 :=
sorry

-- Theorem 2b: Triple Traversal Non-existence
theorem triple_traversal_not_exists {V : Type} (G : Graph V) (h : Connected G) :
  ¬ has_n_traversal_path G 3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eulerian_path_iff_two_odd_vertices_double_traversal_exists_triple_traversal_not_exists_l886_88606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l886_88663

theorem complex_magnitude_proof (m : ℝ) (i : ℂ) (h1 : i * i = -1) 
  (h2 : (m + 1 : ℂ) = 0) (h3 : (2 - m : ℂ) ≠ 0) :
  let z : ℂ := (m + 1) + (2 - m) * i
  Complex.abs ((6 : ℂ) + 3 * i) / z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_magnitude_proof_l886_88663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_same_gender_probability_l886_88603

/-- A person type -/
def Person : Type := Unit

/-- The gender of a person -/
def gender : Person → Gender := sorry

/-- The set of possible genders -/
inductive Gender : Type
| Male
| Female

/-- The event that at least two people in a set have the same gender -/
def at_least_two_same_gender (s : Finset Person) : Prop :=
  ∃ (g : Gender), ∃ (p1 p2 : Person), p1 ∈ s ∧ p2 ∈ s ∧ p1 ≠ p2 ∧ gender p1 = g ∧ gender p2 = g

/-- The probability measure -/
noncomputable def ℙ : Prop → ℝ := sorry

theorem same_gender_probability (people : Finset Person) : 
  Finset.card people = 3 → ℙ (at_least_two_same_gender people) = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_same_gender_probability_l886_88603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distances_l886_88678

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := 4 * x^2 - y^2 + 64 = 0

/-- The distance between two points in 2D space -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-- The theorem stating the relationship between focal distances for a point on the given hyperbola -/
theorem hyperbola_focal_distances (x y xf1 yf1 xf2 yf2 : ℝ) :
  hyperbola x y →
  distance x y xf1 yf1 = 1 →
  distance x y xf2 yf2 = 17 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_focal_distances_l886_88678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_metallic_sheet_length_l886_88686

/-- The length of a rectangular metallic sheet, given specific conditions -/
noncomputable def sheet_length (width cut_size volume : ℝ) : ℝ :=
  let box_length := width - 2 * cut_size
  let box_width := box_length - 2 * cut_size
  let box_height := cut_size
  2 * cut_size + (volume / (box_width * box_height))

/-- Theorem stating the length of the metallic sheet under given conditions -/
theorem metallic_sheet_length :
  sheet_length 36 5 4940 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_metallic_sheet_length_l886_88686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_midpoint_locus_intersection_in_second_quadrant_l886_88693

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop :=
  (3 + m) * x + 4 * y - 3 + 3 * m = 0

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the midpoint trajectory equation
def midpoint_trajectory (x y : ℝ) : Prop :=
  (x - 3/2)^2 + (y - 2)^2 = 1

-- Define the circle C
def circle_C (a b c x y : ℝ) : Prop :=
  (x - b)^2 + (y - c)^2 = a^2

-- Define the two lines
def line1 (a b c x y : ℝ) : Prop :=
  a * x + b * y + c = 0

def line2 (x y : ℝ) : Prop :=
  x + y + 1 = 0

-- Theorem 1
theorem line_passes_through_point :
  ∀ m : ℝ, line_equation m (-3) 3 :=
by
  sorry

-- Theorem 2
theorem midpoint_locus :
  ∀ x₁ y₁ : ℝ, circle_equation x₁ y₁ →
  midpoint_trajectory ((x₁ + 3) / 2) ((y₁ + 4) / 2) :=
by
  sorry

-- Theorem 3
theorem intersection_in_second_quadrant :
  ∀ a b c : ℝ, a > 0 → b > 0 → c > 0 →
  (∃ x : ℝ, circle_C a b c x 0) →
  (∀ x : ℝ, ¬circle_C a b c 0 x) →
  ∃ x y : ℝ, line1 a b c x y ∧ line2 x y ∧ x < 0 ∧ y > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_point_midpoint_locus_intersection_in_second_quadrant_l886_88693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_trip_average_speed_l886_88647

/-- Represents the boat trip with given conditions -/
structure BoatTrip where
  lake_speed : ℝ
  upstream_speed : ℝ
  downstream_speed : ℝ
  lake_distance : ℝ

/-- Calculates the average speed of the boat trip -/
noncomputable def average_speed (trip : BoatTrip) : ℝ :=
  let total_distance := 5 * trip.lake_distance
  let total_time := trip.lake_distance / trip.lake_speed +
                    (2 * trip.lake_distance) / trip.upstream_speed +
                    (2 * trip.lake_distance) / trip.downstream_speed
  total_distance / total_time

/-- Theorem stating that the average speed of the boat trip is 150/31 km/h -/
theorem boat_trip_average_speed (trip : BoatTrip)
  (h1 : trip.lake_speed = 5)
  (h2 : trip.upstream_speed = 4)
  (h3 : trip.downstream_speed = 6)
  : average_speed trip = 150 / 31 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_boat_trip_average_speed_l886_88647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l886_88654

/-- Given a hyperbola C and a line l, prove that the eccentricity of C is 3 -/
theorem hyperbola_eccentricity (k a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let l : ℝ → ℝ → Prop := λ x y ↦ k * x + y - Real.sqrt 2 * k = 0
  let C : ℝ → ℝ → Prop := λ x y ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptote : ℝ → ℝ → Prop := λ x y ↦ k * x + y = 0
  let distance := 4 / 3
  (∃ (x y : ℝ), asymptote x y ∧ 
    (∀ (x' y' : ℝ), asymptote x' y' → 
      (x - x')^2 + (y - y')^2 = distance^2)) →
  Real.sqrt (1 + b^2 / a^2) = 3 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l886_88654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_12_l886_88631

noncomputable def geometric_sequence (a₁ q : ℝ) : ℕ → ℝ := fun n => a₁ * q^(n-1)

noncomputable def geometric_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

theorem geometric_sum_12 (a₁ q : ℝ) :
  geometric_sum a₁ q 4 = 2 →
  geometric_sum a₁ q 8 = 8 →
  geometric_sum a₁ q 12 = 26 :=
by
  sorry

#check geometric_sum_12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sum_12_l886_88631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l886_88691

-- Define the property of being a linear equation in one variable
def is_linear_equation (n : ℤ) : Prop :=
  ∃ (a b : ℝ) (h : a ≠ 0), ∀ x, (n - 2 : ℝ) * x^(|n - 1|) + 5 = a * x + b

-- State the theorem
theorem find_n : 
  (∃ n : ℤ, is_linear_equation n ∧ n - 2 ≠ 0) → 
  (∃ n : ℤ, is_linear_equation n ∧ n - 2 ≠ 0 ∧ n = 0) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_n_l886_88691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_doughnut_machine_completion_time_l886_88623

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Calculates the difference between two times in hours -/
def timeDifference (t1 t2 : Time) : Rat :=
  let totalMinutes := (t2.hours - t1.hours) * 60 + (t2.minutes - t1.minutes)
  totalMinutes / 60

/-- Adds hours to a given time -/
def addHours (t : Time) (h : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + h * 60
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem doughnut_machine_completion_time 
  (start : Time)
  (oneThirdComplete : Time)
  (h : timeDifference start oneThirdComplete = 8/3) :
  addHours start 8 = { hours := 16, minutes := 30 } := by
  sorry

#eval timeDifference 
  { hours := 8, minutes := 30 }
  { hours := 11, minutes := 10 }

#eval addHours 
  { hours := 8, minutes := 30 } 
  8

end NUMINAMATH_CALUDE_ERRORFEEDBACK_doughnut_machine_completion_time_l886_88623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l886_88657

open InnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_projection (a b : V) 
  (h1 : ‖a‖ = 5)
  (h2 : ‖a - b‖ = 6)
  (h3 : ‖a + b‖ = 4) :
  (inner b a / ‖a‖) = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l886_88657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_point_to_line_l886_88608

/-- The distance from a point to a line in a 2D plane -/
noncomputable def distancePointToLine (x₀ y₀ A B C : ℝ) : ℝ :=
  |A * x₀ + B * y₀ + C| / Real.sqrt (A^2 + B^2)

/-- The point coordinates -/
def point : ℝ × ℝ := (-3, 8)

/-- The line equation coefficients for the line y = 5x + 10 in the form Ax + By + C = 0 -/
def line : ℝ × ℝ × ℝ := (5, -1, 10)

theorem distance_from_point_to_line :
  distancePointToLine point.fst point.snd line.1 line.2.1 line.2.2 = 13 * Real.sqrt 26 / 26 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_point_to_line_l886_88608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_l886_88604

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + (1/2) * x^2

-- Define the derivative of f
noncomputable def f_deriv (x : ℝ) : ℝ := x^2 + x

-- Theorem statement
theorem monotonic_increasing_intervals :
  (StrictMonoOn f (Set.Iio (-1)) ∧ StrictMonoOn f (Set.Ioi 0)) ∧
  ¬(StrictMonoOn f (Set.Icc (-1) 0)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_intervals_l886_88604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l886_88635

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2^(x + 2) else x^3

theorem f_composition_negative_one :
  f (f (-1)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_negative_one_l886_88635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_115_l886_88642

def next_term (n : ℕ) : ℕ :=
  if n < 10 then n * 10
  else if n % 2 = 0 then n * 3
  else n + 10

def sequence_term (start : ℕ) : ℕ → ℕ
  | 0 => start
  | n + 1 => next_term (sequence_term start n)

theorem tenth_term_is_115 : sequence_term 15 9 = 115 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tenth_term_is_115_l886_88642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l886_88638

def A : Set ℕ := {0, 1, 2, 3, 4, 5}
def B : Set ℕ := {x : ℕ | x ≤ 4}

theorem intersection_of_A_and_B : A ∩ B = {0, 1, 2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_A_and_B_l886_88638
