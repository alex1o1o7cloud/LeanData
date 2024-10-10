import Mathlib

namespace cylinder_radius_l469_46981

/-- Given a cylinder with height 8 cm and surface area 130π cm², prove its base circle radius is 5 cm. -/
theorem cylinder_radius (h : ℝ) (S : ℝ) (r : ℝ) 
  (height_eq : h = 8)
  (surface_area_eq : S = 130 * Real.pi)
  (surface_area_formula : S = 2 * Real.pi * r^2 + 2 * Real.pi * r * h) :
  r = 5 := by sorry

end cylinder_radius_l469_46981


namespace fundraising_amount_scientific_notation_l469_46909

/-- Represents the amount in yuan --/
def amount : ℝ := 2.175e9

/-- Represents the number of significant figures to preserve --/
def significant_figures : ℕ := 3

/-- Converts a number to scientific notation with a specified number of significant figures --/
noncomputable def to_scientific_notation (x : ℝ) (sig_figs : ℕ) : ℝ × ℤ :=
  sorry

theorem fundraising_amount_scientific_notation :
  to_scientific_notation amount significant_figures = (2.18, 9) := by sorry

end fundraising_amount_scientific_notation_l469_46909


namespace joel_donation_l469_46956

/-- The number of toys Joel donated -/
def joels_toys : ℕ := 22

/-- The number of toys Joel's sister donated -/
def sisters_toys : ℕ := 11

/-- The number of toys Joel's friends donated -/
def friends_toys : ℕ := 75

/-- The total number of donated toys -/
def total_toys : ℕ := 108

theorem joel_donation :
  (friends_toys + sisters_toys + joels_toys = total_toys) ∧
  (joels_toys = 2 * sisters_toys) ∧
  (friends_toys = 18 + 42 + 2 + 13) :=
by sorry

end joel_donation_l469_46956


namespace ball_purchase_equation_l469_46977

/-- Represents the price difference between a basketball and a soccer ball -/
def price_difference : ℝ := 20

/-- Represents the budget for basketballs -/
def basketball_budget : ℝ := 1500

/-- Represents the budget for soccer balls -/
def soccer_ball_budget : ℝ := 800

/-- Represents the quantity difference between basketballs and soccer balls purchased -/
def quantity_difference : ℝ := 5

/-- Theorem stating the equation that represents the relationship between
    the price of soccer balls and the quantities of basketballs and soccer balls purchased -/
theorem ball_purchase_equation (x : ℝ) :
  x > 0 →
  (basketball_budget / (x + price_difference) - soccer_ball_budget / x = quantity_difference) ↔
  (1500 / (x + 20) - 800 / x = 5) :=
by sorry

end ball_purchase_equation_l469_46977


namespace interest_calculation_l469_46967

/-- Given a principal amount P, calculate the compound interest for 2 years at 5% per year -/
def compound_interest (P : ℝ) : ℝ :=
  P * (1 + 0.05)^2 - P

/-- Given a principal amount P, calculate the simple interest for 2 years at 5% per year -/
def simple_interest (P : ℝ) : ℝ :=
  P * 0.05 * 2

/-- Theorem stating that if the compound interest is $615, then the simple interest is $600 -/
theorem interest_calculation (P : ℝ) :
  compound_interest P = 615 → simple_interest P = 600 := by
  sorry

end interest_calculation_l469_46967


namespace max_value_is_nine_l469_46904

def max_value (a b c : ℕ) : ℕ := c * b^a

theorem max_value_is_nine :
  ∃ (a b c : ℕ), a ∈ ({1, 2, 3} : Set ℕ) ∧ b ∈ ({1, 2, 3} : Set ℕ) ∧ c ∈ ({1, 2, 3} : Set ℕ) ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  max_value a b c = 9 ∧
  ∀ (x y z : ℕ), x ∈ ({1, 2, 3} : Set ℕ) → y ∈ ({1, 2, 3} : Set ℕ) → z ∈ ({1, 2, 3} : Set ℕ) →
  x ≠ y → y ≠ z → x ≠ z →
  max_value x y z ≤ 9 :=
by
  sorry

end max_value_is_nine_l469_46904


namespace log_xy_l469_46969

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_xy (x y : ℝ) (h1 : log (x * y^5) = 2) (h2 : log (x^3 * y) = 2) :
  log (x * y) = 6/7 := by sorry

end log_xy_l469_46969


namespace rounding_accuracy_l469_46971

-- Define the rounded number
def rounded_number : ℝ := 5.8 * 10^5

-- Define the accuracy levels
inductive AccuracyLevel
  | Tenth
  | Hundredth
  | Thousandth
  | TenThousandth
  | HundredThousandth

-- Define a function to determine the accuracy level
def determine_accuracy (x : ℝ) : AccuracyLevel :=
  match x with
  | _ => AccuracyLevel.TenThousandth -- We know this is the correct answer from the problem

-- State the theorem
theorem rounding_accuracy :
  determine_accuracy rounded_number = AccuracyLevel.TenThousandth :=
by sorry

end rounding_accuracy_l469_46971


namespace function_root_property_l469_46902

/-- Given a function f(x) = m · 2^x + x^2 + nx, if the set of roots of f(x) is equal to 
    the set of roots of f(f(x)) and is non-empty, then m+n is in the interval [0, 4). -/
theorem function_root_property (m n : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ m * (2^x) + x^2 + n*x
  (∃ x, f x = 0) ∧ 
  (∀ x, f x = 0 ↔ f (f x) = 0) →
  0 ≤ m + n ∧ m + n < 4 := by
sorry

end function_root_property_l469_46902


namespace polynomial_composition_l469_46931

/-- Given f(x) = x² and f(h(x)) = 9x² + 6x + 1, prove that h(x) = 3x + 1 or h(x) = -3x - 1 -/
theorem polynomial_composition (f h : ℝ → ℝ) : 
  (∀ x, f x = x^2) → 
  (∀ x, f (h x) = 9*x^2 + 6*x + 1) → 
  (∀ x, h x = 3*x + 1 ∨ h x = -3*x - 1) := by
sorry

end polynomial_composition_l469_46931


namespace point_coordinates_l469_46936

/-- A point in the two-dimensional plane. -/
structure Point where
  x : ℝ
  y : ℝ

/-- The fourth quadrant of the two-dimensional plane. -/
def fourthQuadrant (p : Point) : Prop := p.x > 0 ∧ p.y < 0

/-- The distance from a point to the x-axis. -/
def distanceToXAxis (p : Point) : ℝ := |p.y|

/-- The distance from a point to the y-axis. -/
def distanceToYAxis (p : Point) : ℝ := |p.x|

/-- Theorem: If a point P is in the fourth quadrant, its distance to the x-axis is 4,
    and its distance to the y-axis is 2, then its coordinates are (2, -4). -/
theorem point_coordinates (P : Point) 
  (h1 : fourthQuadrant P) 
  (h2 : distanceToXAxis P = 4) 
  (h3 : distanceToYAxis P = 2) : 
  P.x = 2 ∧ P.y = -4 := by
  sorry

end point_coordinates_l469_46936


namespace students_playing_neither_sport_l469_46992

theorem students_playing_neither_sport
  (total : ℕ)
  (football : ℕ)
  (tennis : ℕ)
  (both : ℕ)
  (h1 : total = 50)
  (h2 : football = 32)
  (h3 : tennis = 28)
  (h4 : both = 22) :
  total - (football + tennis - both) = 12 :=
by sorry

end students_playing_neither_sport_l469_46992


namespace shopkeeper_gain_percentage_l469_46987

/-- Calculates the gain percentage for a dishonest shopkeeper using a false weight -/
theorem shopkeeper_gain_percentage (false_weight : ℝ) (true_weight : ℝ) : 
  false_weight = 960 →
  true_weight = 1000 →
  (true_weight - false_weight) / false_weight * 100 = (1000 - 960) / 960 * 100 := by
sorry

end shopkeeper_gain_percentage_l469_46987


namespace four_balls_two_boxes_l469_46973

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := 
  (k ^ n) / (Nat.factorial k)

/-- Theorem: There are 8 ways to put 4 distinguishable balls into 2 indistinguishable boxes -/
theorem four_balls_two_boxes : ways_to_put_balls_in_boxes 4 2 = 8 := by
  sorry

#eval ways_to_put_balls_in_boxes 4 2

end four_balls_two_boxes_l469_46973


namespace removed_triangles_area_l469_46947

/-- Given a square with side length x, from which isosceles right triangles
    are removed from each corner to form a rectangle with diagonal 15,
    prove that the total area of the four removed triangles is 112.5. -/
theorem removed_triangles_area (x : ℝ) (r s : ℝ) : 
  (x - r)^2 + (x - s)^2 = 15^2 →
  r + s = x →
  (4 : ℝ) * (1/2 * r * s) = 112.5 := by
  sorry

end removed_triangles_area_l469_46947


namespace distance_to_school_proof_l469_46949

/-- The distance from Xiaohong's home to school in meters -/
def distance_to_school : ℝ := 2720

/-- The distance dad drove Xiaohong towards school in meters -/
def distance_driven : ℝ := 1000

/-- Total travel time (drive + walk) in minutes -/
def total_travel_time : ℝ := 22.5

/-- Time taken to bike from home to school in minutes -/
def biking_time : ℝ := 40

/-- Xiaohong's walking speed in meters per minute -/
def walking_speed : ℝ := 80

/-- The difference between dad's driving speed and Xiaohong's biking speed in meters per minute -/
def speed_difference : ℝ := 800

theorem distance_to_school_proof :
  ∃ (driving_speed : ℝ),
    driving_speed > 0 ∧
    distance_to_school = distance_driven + walking_speed * (total_travel_time - distance_driven / driving_speed) ∧
    distance_to_school = biking_time * (driving_speed - speed_difference) :=
by sorry

end distance_to_school_proof_l469_46949


namespace triangle_condition_line_through_intersection_l469_46908

-- Define the lines
def l1 (x y : ℝ) : Prop := x + y - 4 = 0
def l2 (x y : ℝ) : Prop := x - y + 2 = 0
def l3 (a x y : ℝ) : Prop := a * x - y + 1 - 4 * a = 0

-- Define point M
def M : ℝ × ℝ := (-1, 2)

-- Theorem for the range of a
theorem triangle_condition (a : ℝ) :
  (∃ x y z : ℝ, l1 x y ∧ l2 y z ∧ l3 a z x) ↔ 
  (a ≠ -2/3 ∧ a ≠ 1 ∧ a ≠ -1) :=
sorry

-- Theorem for the equation of line l
theorem line_through_intersection (x y : ℝ) :
  (∃ p q : ℝ, l1 p q ∧ l2 p q) ∧ 
  (abs (3*x + 4*y - 15) / Real.sqrt (3^2 + 4^2) = 2) ↔
  3*x + 4*y - 15 = 0 :=
sorry

end triangle_condition_line_through_intersection_l469_46908


namespace x_power_twenty_is_negative_one_l469_46905

theorem x_power_twenty_is_negative_one (x : ℂ) (h : x + 1/x = Real.sqrt 2) : x^20 = -1 := by
  sorry

end x_power_twenty_is_negative_one_l469_46905


namespace expression_simplification_l469_46900

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x - 1) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1) + 1) = (Real.sqrt 2 - 1) / 2 := by
  sorry

end expression_simplification_l469_46900


namespace two_identical_solutions_l469_46952

/-- The value of k for which the equations y = x^2 and y = 4x + k have two identical solutions -/
def k_value : ℝ := -4

/-- First equation: y = x^2 -/
def eq1 (x y : ℝ) : Prop := y = x^2

/-- Second equation: y = 4x + k -/
def eq2 (x y k : ℝ) : Prop := y = 4*x + k

/-- Two identical solutions exist when k = k_value -/
theorem two_identical_solutions (k : ℝ) :
  (∃ x₁ x₂ y₁ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    eq1 x₁ y₁ ∧ eq2 x₁ y₁ k ∧ 
    eq1 x₂ y₂ ∧ eq2 x₂ y₂ k) ↔ 
  k = k_value :=
sorry

end two_identical_solutions_l469_46952


namespace carries_revenue_l469_46944

/-- Represents the harvest quantities of vegetables -/
structure Harvest where
  tomatoes : ℕ
  carrots : ℕ
  eggplants : ℕ
  cucumbers : ℕ

/-- Represents the selling prices of vegetables -/
structure Prices where
  tomato : ℚ
  carrot : ℚ
  eggplant : ℚ
  cucumber : ℚ

/-- Calculates the total revenue from selling all vegetables -/
def totalRevenue (h : Harvest) (p : Prices) : ℚ :=
  h.tomatoes * p.tomato +
  h.carrots * p.carrot +
  h.eggplants * p.eggplant +
  h.cucumbers * p.cucumber

/-- Theorem stating that Carrie's total revenue is $1156.25 -/
theorem carries_revenue :
  let h : Harvest := { tomatoes := 200, carrots := 350, eggplants := 120, cucumbers := 75 }
  let p : Prices := { tomato := 1, carrot := 3/2, eggplant := 5/2, cucumber := 7/4 }
  totalRevenue h p = 4625/4 := by
  sorry

#eval (4625/4 : ℚ)  -- This should evaluate to 1156.25

end carries_revenue_l469_46944


namespace problem_solution_l469_46948

theorem problem_solution (a b : ℕ+) 
  (h1 : Nat.lcm a b = 5040)
  (h2 : Nat.gcd a b = 24)
  (h3 : a = 240) :
  b = 504 := by
  sorry

end problem_solution_l469_46948


namespace min_value_expression_l469_46919

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a^2 + 4*b^2 + 1/(a*b) ≥ 4 := by
  sorry

end min_value_expression_l469_46919


namespace eighth_term_value_l469_46974

/-- An arithmetic sequence with 30 terms, first term 3, and last term 87 -/
def arithmetic_sequence (n : ℕ) : ℚ :=
  let d := (87 - 3) / (30 - 1)
  3 + (n - 1) * d

/-- The 8th term of the arithmetic sequence -/
def eighth_term : ℚ := arithmetic_sequence 8

theorem eighth_term_value : eighth_term = 675 / 29 := by
  sorry

end eighth_term_value_l469_46974


namespace function_value_2012_l469_46962

theorem function_value_2012 (m n α₁ α₂ : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hα₁ : α₁ ≠ 0) (hα₂ : α₂ ≠ 0) :
  let f : ℝ → ℝ := λ x => m * Real.sin (π * x + α₁) + n * Real.cos (π * x + α₂)
  f 2011 = 1 → f 2012 = -1 := by
sorry

end function_value_2012_l469_46962


namespace count_quadratic_integer_solutions_l469_46918

theorem count_quadratic_integer_solutions :
  ∃ (S : Finset ℕ), 
    (∀ a ∈ S, a > 0 ∧ a ≤ 40) ∧
    (∀ a ∈ S, ∃ x y : ℤ, x ≠ y ∧ x^2 + (3*a + 2)*x + a^2 = 0 ∧ y^2 + (3*a + 2)*y + a^2 = 0) ∧
    (∀ a : ℕ, a > 0 → a ≤ 40 →
      (∃ x y : ℤ, x ≠ y ∧ x^2 + (3*a + 2)*x + a^2 = 0 ∧ y^2 + (3*a + 2)*y + a^2 = 0) →
      a ∈ S) ∧
    Finset.card S = 5 :=
sorry

end count_quadratic_integer_solutions_l469_46918


namespace shaded_area_of_rectangle_l469_46920

/-- The area of the shaded part of a rectangle with specific properties -/
theorem shaded_area_of_rectangle (base height total_area : ℝ) : 
  base = 7 →
  height = 4 →
  total_area = 56 →
  total_area - 2 * (base * height / 2) = 28 :=
by sorry

end shaded_area_of_rectangle_l469_46920


namespace solution_sets_union_l469_46943

-- Define the solution sets M and N
def M (p : ℝ) : Set ℝ := {x | x^2 - p*x + 6 = 0}
def N (q : ℝ) : Set ℝ := {x | x^2 + 6*x - q = 0}

-- State the theorem
theorem solution_sets_union (p q : ℝ) :
  (∃ (x : ℝ), x ∈ M p ∧ x ∈ N q) ∧ (M p ∩ N q = {2}) →
  M p ∪ N q = {2, 3, -8} :=
by sorry

end solution_sets_union_l469_46943


namespace train_crossing_time_l469_46930

/-- The time taken for two trains to cross each other -/
theorem train_crossing_time (length1 length2 speed1 speed2 : ℝ) : 
  length1 = 180 ∧ 
  length2 = 360 ∧ 
  speed1 = 60 * (1000 / 3600) ∧ 
  speed2 = 30 * (1000 / 3600) →
  (length1 + length2) / (speed1 + speed2) = 21.6 := by
  sorry

end train_crossing_time_l469_46930


namespace boar_sausages_problem_l469_46954

theorem boar_sausages_problem (S : ℕ) : 
  (S > 0) →  -- Ensure S is positive
  (3 / 40 : ℚ) * S = 45 → 
  S = 600 := by 
sorry

end boar_sausages_problem_l469_46954


namespace total_weight_moved_l469_46907

/-- Represents an exercise with weight, reps, and sets -/
structure Exercise where
  weight : Nat
  reps : Nat
  sets : Nat

/-- Calculates the total weight moved for a single exercise -/
def totalWeightForExercise (e : Exercise) : Nat :=
  e.weight * e.reps * e.sets

/-- John's workout routine -/
def workoutRoutine : List Exercise := [
  { weight := 15, reps := 10, sets := 3 },  -- Bench press
  { weight := 12, reps := 8,  sets := 4 },  -- Bicep curls
  { weight := 50, reps := 12, sets := 3 },  -- Squats
  { weight := 80, reps := 6,  sets := 2 }   -- Deadlift
]

/-- Theorem stating the total weight John moves during his workout -/
theorem total_weight_moved : 
  (workoutRoutine.map totalWeightForExercise).sum = 3594 := by
  sorry


end total_weight_moved_l469_46907


namespace mingyoungs_math_score_l469_46928

theorem mingyoungs_math_score 
  (korean : ℝ) 
  (english : ℝ) 
  (math : ℝ) 
  (h1 : (korean + english) / 2 = 89) 
  (h2 : (korean + english + math) / 3 = 91) : 
  math = 95 :=
sorry

end mingyoungs_math_score_l469_46928


namespace borrowed_amount_is_6800_l469_46959

/-- Calculates the amount borrowed given interest rates and total interest paid -/
def calculate_borrowed_amount (total_interest : ℚ) (rate1 rate2 rate3 : ℚ) 
  (period1 period2 period3 : ℚ) : ℚ :=
  total_interest / (rate1 * period1 + rate2 * period2 + rate3 * period3)

/-- Proves that the amount borrowed is 6800, given the specified conditions -/
theorem borrowed_amount_is_6800 : 
  let total_interest : ℚ := 8160
  let rate1 : ℚ := 12 / 100
  let rate2 : ℚ := 9 / 100
  let rate3 : ℚ := 13 / 100
  let period1 : ℚ := 3
  let period2 : ℚ := 5
  let period3 : ℚ := 3
  calculate_borrowed_amount total_interest rate1 rate2 rate3 period1 period2 period3 = 6800 := by
  sorry

#eval calculate_borrowed_amount 8160 (12/100) (9/100) (13/100) 3 5 3

end borrowed_amount_is_6800_l469_46959


namespace mixed_doubles_probability_l469_46961

/-- The number of athletes -/
def total_athletes : ℕ := 6

/-- The number of male athletes -/
def male_athletes : ℕ := 3

/-- The number of female athletes -/
def female_athletes : ℕ := 3

/-- The number of coaches -/
def coaches : ℕ := 3

/-- The number of players each coach selects -/
def players_per_coach : ℕ := 2

/-- The probability of all coaches forming mixed doubles teams -/
def probability_mixed_doubles : ℚ := 2/5

theorem mixed_doubles_probability :
  let total_outcomes := (total_athletes.choose players_per_coach * 
                         (total_athletes - players_per_coach).choose players_per_coach * 
                         (total_athletes - 2*players_per_coach).choose players_per_coach) / coaches.factorial
  let favorable_outcomes := male_athletes.choose 1 * female_athletes.choose 1 * 
                            (male_athletes - 1).choose 1 * (female_athletes - 1).choose 1 * 
                            (male_athletes - 2).choose 1 * (female_athletes - 2).choose 1 * 
                            coaches.factorial
  (favorable_outcomes : ℚ) / total_outcomes = probability_mixed_doubles :=
sorry

end mixed_doubles_probability_l469_46961


namespace cos_neg_pi_third_l469_46998

theorem cos_neg_pi_third : Real.cos (-π/3) = 1/2 := by
  sorry

end cos_neg_pi_third_l469_46998


namespace expression_evaluation_l469_46955

theorem expression_evaluation :
  let expr1 := (27 / 8) ^ (-2/3) - (49 / 9) ^ (1/2) + (0.2)^(-2) * (3 / 25)
  let expr2 := -5 * (Real.log 4 / Real.log 9) + (Real.log (32 / 9) / Real.log 3) - 5^(Real.log 3 / Real.log 5)
  (expr1 = 10/9) ∧ (expr2 = -5 * (Real.log 2 / Real.log 3) - 5) := by
  sorry

end expression_evaluation_l469_46955


namespace pyramid_block_count_l469_46950

/-- 
Represents a four-layer pyramid where each layer has three times as many blocks 
as the layer above it, with the top layer being a single block.
-/
def PyramidBlocks : ℕ → ℕ
| 0 => 1  -- Top layer
| n + 1 => 3 * PyramidBlocks n  -- Each subsequent layer

/-- The total number of blocks in the four-layer pyramid -/
def TotalBlocks : ℕ := 
  PyramidBlocks 0 + PyramidBlocks 1 + PyramidBlocks 2 + PyramidBlocks 3

theorem pyramid_block_count : TotalBlocks = 40 := by
  sorry

end pyramid_block_count_l469_46950


namespace rachel_apple_trees_l469_46901

/-- The number of apple trees Rachel has -/
def num_trees : ℕ := 3

/-- The number of apples picked from each tree -/
def apples_per_tree : ℕ := 8

/-- The total number of apples remaining after picking -/
def apples_remaining : ℕ := 9

/-- The initial total number of apples on all trees -/
def initial_apples : ℕ := 33

theorem rachel_apple_trees :
  num_trees * apples_per_tree + apples_remaining = initial_apples :=
sorry

end rachel_apple_trees_l469_46901


namespace union_when_a_is_3_union_equals_real_iff_l469_46958

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 3 < x ∧ x < a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 3}

-- First part of the theorem
theorem union_when_a_is_3 :
  A 3 ∪ B = {x | x < -1 ∨ x > 0} :=
sorry

-- Second part of the theorem
theorem union_equals_real_iff :
  ∀ a : ℝ, A a ∪ B = Set.univ ↔ 0 < a ∧ a < 4 :=
sorry

end union_when_a_is_3_union_equals_real_iff_l469_46958


namespace range_of_k_l469_46941

theorem range_of_k (k : ℝ) : 
  (∀ a b : ℝ, a^2 + b^2 ≥ 2*k*a*b) → k ∈ Set.Icc (-1) 1 := by
  sorry

end range_of_k_l469_46941


namespace g_geq_h_implies_a_leq_one_l469_46951

noncomputable def g (x : ℝ) : ℝ := Real.exp x - Real.exp 1 * x - 1

noncomputable def h (a x : ℝ) : ℝ := a * Real.sin x - Real.exp 1 * x

theorem g_geq_h_implies_a_leq_one (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, g x ≥ h a x) → a ≤ 1 := by
  sorry

end g_geq_h_implies_a_leq_one_l469_46951


namespace trapezoid_area_theorem_l469_46946

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents a trapezoid formed by the arrangement of squares -/
structure Trapezoid where
  squares : List Square
  connector : ℝ × ℝ  -- Represents the connecting line segment

/-- Calculates the area of the trapezoid formed by the arrangement of squares -/
noncomputable def calculateTrapezoidArea (t : Trapezoid) : ℝ :=
  sorry

/-- The main theorem stating the area of the trapezoid -/
theorem trapezoid_area_theorem (s1 s2 s3 s4 : Square) 
  (h1 : s1.sideLength = 3)
  (h2 : s2.sideLength = 5)
  (h3 : s3.sideLength = 7)
  (h4 : s4.sideLength = 7)
  (t : Trapezoid)
  (ht : t.squares = [s1, s2, s3, s4]) :
  abs (calculateTrapezoidArea t - 12.83325) < 0.00001 := by
  sorry

end trapezoid_area_theorem_l469_46946


namespace parents_present_l469_46968

theorem parents_present (total_people : ℕ) (pupils : ℕ) (h1 : total_people = 676) (h2 : pupils = 654) :
  total_people - pupils = 22 := by
  sorry

end parents_present_l469_46968


namespace candle_burn_theorem_l469_46917

theorem candle_burn_theorem (t : ℝ) (h : t > 0) :
  let rate_second : ℝ := (3 / 5) / t
  let rate_third : ℝ := (4 / 7) / t
  let time_second_remaining : ℝ := (2 / 5) / rate_second
  let third_burned_while_second_finishes : ℝ := time_second_remaining * rate_third
  (3 / 7) - third_burned_while_second_finishes = 1 / 21 := by
sorry

end candle_burn_theorem_l469_46917


namespace true_false_questions_count_l469_46957

def number_of_multiple_choice_questions : ℕ := 2
def choices_per_multiple_choice_question : ℕ := 4
def total_answer_key_combinations : ℕ := 480

def valid_true_false_combinations (n : ℕ) : ℕ := 2^n - 2

theorem true_false_questions_count :
  ∃ n : ℕ, 
    n > 0 ∧
    valid_true_false_combinations n * 
    choices_per_multiple_choice_question ^ number_of_multiple_choice_questions = 
    total_answer_key_combinations ∧
    n = 5 := by
  sorry

end true_false_questions_count_l469_46957


namespace trigonometric_equation_solution_l469_46997

theorem trigonometric_equation_solution (x : ℝ) : 
  (abs (Real.sin x) + Real.sin (3 * x)) / (Real.cos x * Real.cos (2 * x)) = 2 / Real.sqrt 3 ↔ 
  (∃ k : ℤ, x = π / 12 + 2 * k * π ∨ x = 7 * π / 12 + 2 * k * π ∨ x = -5 * π / 6 + 2 * k * π) := by
  sorry

end trigonometric_equation_solution_l469_46997


namespace computer_sales_ratio_l469_46924

theorem computer_sales_ratio (total : ℕ) (netbook_fraction : ℚ) (desktops : ℕ) : 
  total = 72 → netbook_fraction = 1/3 → desktops = 12 → 
  (total - (netbook_fraction * total).num - desktops : ℚ) / total = 1/2 := by
  sorry

end computer_sales_ratio_l469_46924


namespace pyramid_volume_l469_46953

/-- The volume of a pyramid with a triangular base and lateral faces forming 45° dihedral angles with the base -/
theorem pyramid_volume (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5) : 
  let p := (a + b + c) / 2
  let S := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let r := S / p
  let H := r
  let V := (1/3) * S * H
  V = 6 := by sorry

end pyramid_volume_l469_46953


namespace vertex_not_zero_l469_46914

/-- The vertex of a quadratic function y = x^2 - (m-2)x + 4 lies on a coordinate axis if and only if
    m = 2 or m = -2 or m = 6 -/
def vertex_on_axis (m : ℝ) : Prop :=
  m = 2 ∨ m = -2 ∨ m = 6

/-- If the vertex of the quadratic function y = x^2 - (m-2)x + 4 lies on a coordinate axis,
    then m ≠ 0 -/
theorem vertex_not_zero (m : ℝ) (h : vertex_on_axis m) : m ≠ 0 := by
  sorry

end vertex_not_zero_l469_46914


namespace count_random_events_l469_46926

-- Define the type for events
inductive Event
  | throwDice : Event
  | pearFall : Event
  | winLottery : Event
  | haveBoy : Event
  | waterBoil : Event

-- Define a function to determine if an event is random
def isRandom (e : Event) : Bool :=
  match e with
  | Event.throwDice => true
  | Event.pearFall => false
  | Event.winLottery => true
  | Event.haveBoy => true
  | Event.waterBoil => false

-- Define the list of all events
def allEvents : List Event :=
  [Event.throwDice, Event.pearFall, Event.winLottery, Event.haveBoy, Event.waterBoil]

-- State the theorem
theorem count_random_events :
  (allEvents.filter isRandom).length = 3 := by
  sorry

end count_random_events_l469_46926


namespace monomial_sum_l469_46942

/-- Given two monomials that form a monomial when added together, prove that m + n = 4 -/
theorem monomial_sum (m n : ℕ) : 
  (∃ (a : ℝ), ∀ (x y : ℝ), 2 * x^(m-1) * y^2 + (1/3) * x^2 * y^(n+1) = a * x^2 * y^2) → 
  m + n = 4 := by
  sorry

end monomial_sum_l469_46942


namespace line_equation_l469_46913

/-- Given two points A(x₁,y₁) and B(x₂,y₂) satisfying the equations 3x₁ - 4y₁ - 2 = 0 and 3x₂ - 4y₂ - 2 = 0,
    the line passing through these points has the equation 3x - 4y - 2 = 0. -/
theorem line_equation (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : 3 * x₁ - 4 * y₁ - 2 = 0) 
  (h₂ : 3 * x₂ - 4 * y₂ - 2 = 0) : 
  ∀ (x y : ℝ), (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) → 3 * x - 4 * y - 2 = 0 := by
  sorry

end line_equation_l469_46913


namespace purchase_decision_l469_46964

/-- Represents the prices and conditions for the company's purchase decision --/
structure PurchaseScenario where
  tablet_price : ℝ
  speaker_price : ℝ
  total_items : ℕ
  discount_rate1 : ℝ
  discount_threshold2 : ℝ
  discount_rate2 : ℝ

/-- Theorem stating the correct prices and cost-effective decision based on the number of tablets --/
theorem purchase_decision (p : PurchaseScenario) 
  (h1 : 2 * p.tablet_price + 3 * p.speaker_price = 7600)
  (h2 : 3 * p.tablet_price = 5 * p.speaker_price)
  (h3 : p.total_items = 30)
  (h4 : p.discount_rate1 = 0.1)
  (h5 : p.discount_threshold2 = 24000)
  (h6 : p.discount_rate2 = 0.2) :
  p.tablet_price = 2000 ∧ 
  p.speaker_price = 1200 ∧ 
  (∀ a : ℕ, a < 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) < 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) ∧
  (∀ a : ℕ, a = 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) = 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) ∧
  (∀ a : ℕ, a > 15 → 
    (1 - p.discount_rate1) * (p.tablet_price * a + p.speaker_price * (p.total_items - a)) > 
    p.discount_threshold2 + (1 - p.discount_rate2) * (p.tablet_price * a + p.speaker_price * (p.total_items - a) - p.discount_threshold2)) :=
by sorry

end purchase_decision_l469_46964


namespace pants_original_price_l469_46963

theorem pants_original_price 
  (total_spent : ℝ)
  (jacket_discount : ℝ)
  (pants_discount : ℝ)
  (jacket_original : ℝ)
  (h1 : total_spent = 306)
  (h2 : jacket_discount = 0.7)
  (h3 : pants_discount = 0.8)
  (h4 : jacket_original = 300) :
  ∃ (pants_original : ℝ), 
    jacket_original * jacket_discount + pants_original * pants_discount = total_spent ∧ 
    pants_original = 120 :=
by sorry

end pants_original_price_l469_46963


namespace triangle_area_l469_46938

/-- The area of a triangle with vertices (0,4,13), (-2,3,9), and (-5,6,9) is (3√30)/4 -/
theorem triangle_area : 
  let A : ℝ × ℝ × ℝ := (0, 4, 13)
  let B : ℝ × ℝ × ℝ := (-2, 3, 9)
  let C : ℝ × ℝ × ℝ := (-5, 6, 9)
  let area := Real.sqrt (
    let s := (Real.sqrt 21 + 3 * Real.sqrt 2 + 3 * Real.sqrt 5) / 2
    s * (s - Real.sqrt 21) * (s - 3 * Real.sqrt 2) * (s - 3 * Real.sqrt 5)
  )
  area = 3 * Real.sqrt 30 / 4 := by
  sorry

end triangle_area_l469_46938


namespace sufficient_condition_absolute_value_l469_46993

theorem sufficient_condition_absolute_value (a : ℝ) : 
  (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) → a ≥ 3 := by
  sorry

end sufficient_condition_absolute_value_l469_46993


namespace max_stamps_proof_l469_46911

/-- The price of a single stamp in cents -/
def stamp_price : ℕ := 50

/-- The discount rate applied when buying more than 100 stamps -/
def discount_rate : ℚ := 1/10

/-- The threshold number of stamps for applying the discount -/
def discount_threshold : ℕ := 100

/-- The total amount available in cents -/
def total_amount : ℕ := 10000

/-- The maximum number of stamps that can be purchased -/
def max_stamps : ℕ := 200

theorem max_stamps_proof :
  (∀ n : ℕ, n ≤ max_stamps → n * stamp_price ≤ total_amount) ∧
  (∀ n : ℕ, n > max_stamps → 
    (if n > discount_threshold 
     then n * stamp_price * (1 - discount_rate)
     else n * stamp_price) > total_amount) :=
by sorry

end max_stamps_proof_l469_46911


namespace characterize_valid_functions_l469_46995

def is_valid_function (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, f (n + 1) > (f n + f (f n)) / 2

theorem characterize_valid_functions :
  ∀ f : ℕ → ℕ, is_valid_function f →
  ∃ b : ℕ, (∀ n < b, f n = n) ∧ (∀ n ≥ b, f n = n + 1) :=
sorry

end characterize_valid_functions_l469_46995


namespace chocolates_gain_percent_l469_46975

/-- Calculates the gain percent given the number of chocolates at cost price and selling price that are equal in value -/
def gain_percent (cost_chocolates : ℕ) (sell_chocolates : ℕ) : ℚ :=
  ((cost_chocolates : ℚ) / sell_chocolates - 1) * 100

/-- Theorem stating that if the cost price of 81 chocolates equals the selling price of 45 chocolates, the gain percent is 80% -/
theorem chocolates_gain_percent :
  gain_percent 81 45 = 80 := by
  sorry

end chocolates_gain_percent_l469_46975


namespace eighteenth_term_of_sequence_l469_46986

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem eighteenth_term_of_sequence : arithmetic_sequence 3 4 18 = 71 := by
  sorry

end eighteenth_term_of_sequence_l469_46986


namespace radical_equality_implies_c_equals_six_l469_46935

theorem radical_equality_implies_c_equals_six 
  (a b c : ℕ) 
  (ha : a > 1) (hb : b > 1) (hc : c > 1) 
  (h : ∀ M : ℝ, M ≠ 1 → M^(1/a + 1/(a*b) + 3/(a*b*c)) = M^(14/24)) : 
  c = 6 := by
sorry

end radical_equality_implies_c_equals_six_l469_46935


namespace sqrt_27_minus_sqrt_3_equals_2_sqrt_3_l469_46983

theorem sqrt_27_minus_sqrt_3_equals_2_sqrt_3 : 
  Real.sqrt 27 - Real.sqrt 3 = 2 * Real.sqrt 3 := by
  sorry

end sqrt_27_minus_sqrt_3_equals_2_sqrt_3_l469_46983


namespace function_properties_l469_46939

/-- Given a function f(x) = x - a*exp(x) + b, where a > 0 and b is real,
    this theorem states two properties:
    1. The maximum value of f(x) is ln(1/a) - 1 + b
    2. If f has two distinct zeros x₁ and x₂, then x₁ + x₂ < -2*ln(a) -/
theorem function_properties (a b : ℝ) (ha : a > 0) :
  let f := fun x => x - a * Real.exp x + b
  (∃ (x : ℝ), ∀ (y : ℝ), f y ≤ f x ∧ f x = Real.log (1 / a) - 1 + b) ∧
  (∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → f x₁ = 0 → f x₂ = 0 → x₁ + x₂ < -2 * Real.log a) := by
  sorry


end function_properties_l469_46939


namespace four_propositions_l469_46932

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + x - m

-- Define what it means for a function to have zero points
def has_zero_points (f : ℝ → ℝ) : Prop := ∃ x, f x = 0

-- Define what it means for four points to be coplanar
def coplanar (E F G H : ℝ × ℝ × ℝ) : Prop := sorry

-- Define what it means for two lines to intersect
def lines_intersect (E F G H : ℝ × ℝ × ℝ) : Prop := sorry

-- Define what it means for an equation to represent a hyperbola
def is_hyperbola (m : ℝ) : Prop := sorry

theorem four_propositions :
  (∀ m > 0, has_zero_points (f m)) ∧ 
  (∀ E F G H, ¬coplanar E F G H → ¬lines_intersect E F G H) ∧
  (∃ E F G H, ¬lines_intersect E F G H ∧ coplanar E F G H) ∧
  (∀ a : ℝ, (∀ x : ℝ, |x+1| + |x-1| ≥ a) ↔ a < 2) ∧
  (∀ m : ℝ, (0 < m ∧ m < 1) ↔ is_hyperbola m) :=
by
  sorry

end four_propositions_l469_46932


namespace quadratic_roots_equal_irrational_l469_46984

theorem quadratic_roots_equal_irrational (d : ℝ) :
  let a : ℝ := 3
  let b : ℝ := -4 * Real.pi
  let c : ℝ := d
  let discriminant := b^2 - 4*a*c
  discriminant = 16 →
  ∃ (x : ℝ), (a*x^2 + b*x + c = 0 ∧ 
              ∀ (y : ℝ), a*y^2 + b*y + c = 0 → y = x) ∧
             (¬ ∃ (p q : ℤ), x = p / q) :=
by sorry

end quadratic_roots_equal_irrational_l469_46984


namespace correct_assignment_count_l469_46937

/-- The number of ways to assign 5 friends to 5 rooms with at most 2 friends per room -/
def assignmentWays : ℕ := 1620

/-- A function that calculates the number of ways to assign n friends to m rooms with at most k friends per room -/
def calculateAssignmentWays (n m k : ℕ) : ℕ :=
  sorry  -- The actual implementation is not provided

theorem correct_assignment_count :
  calculateAssignmentWays 5 5 2 = assignmentWays :=
by sorry

end correct_assignment_count_l469_46937


namespace tenth_student_score_l469_46923

/-- Represents a valid arithmetic sequence of exam scores -/
structure ExamScores where
  scores : Fin 10 → ℕ
  is_arithmetic : ∀ i j k : Fin 10, i.val + k.val = j.val + j.val → scores i + scores k = scores j + scores j
  max_score : ∀ i : Fin 10, scores i ≤ 100
  sum_middle : scores 2 + scores 3 + scores 4 + scores 5 = 354
  contains_96 : ∃ i : Fin 10, scores i = 96

/-- The theorem stating the possible scores for the 10th student -/
theorem tenth_student_score (e : ExamScores) : e.scores 0 = 61 ∨ e.scores 0 = 72 := by
  sorry

end tenth_student_score_l469_46923


namespace isabel_afternoon_runs_l469_46976

/-- Calculates the number of afternoon runs given circuit length, morning runs, and total weekly distance -/
def afternoon_runs (circuit_length : ℕ) (morning_runs : ℕ) (total_weekly_distance : ℕ) : ℕ :=
  (total_weekly_distance - 7 * morning_runs * circuit_length) / circuit_length

/-- Proves that Isabel runs the circuit 21 times in the afternoon during a week -/
theorem isabel_afternoon_runs : 
  afternoon_runs 365 7 25550 = 21 := by
  sorry

end isabel_afternoon_runs_l469_46976


namespace largest_constant_inequality_l469_46916

theorem largest_constant_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x^2 + y^2 = 1) :
  ∃ c : ℝ, c = 1/2 ∧ x^6 + y^6 ≥ c * x * y ∧ ∀ d : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + b^2 = 1 → a^6 + b^6 ≥ d * a * b) → d ≤ c :=
by sorry

end largest_constant_inequality_l469_46916


namespace popcorn_profit_30_bags_l469_46965

/-- Calculates the profit from selling popcorn bags -/
def popcorn_profit (buy_price sell_price : ℕ) (num_bags : ℕ) : ℕ :=
  (sell_price - buy_price) * num_bags

theorem popcorn_profit_30_bags :
  popcorn_profit 4 8 30 = 120 := by
  sorry

end popcorn_profit_30_bags_l469_46965


namespace correct_average_after_error_l469_46925

theorem correct_average_after_error (n : ℕ) (initial_avg : ℚ) (wrong_mark correct_mark : ℚ) :
  n = 10 →
  initial_avg = 100 →
  wrong_mark = 50 →
  correct_mark = 10 →
  (n : ℚ) * initial_avg - wrong_mark + correct_mark = (n : ℚ) * 96 :=
by
  sorry

end correct_average_after_error_l469_46925


namespace fifth_coaster_speed_l469_46922

def rollercoaster_problem (S₁ S₂ S₃ S₄ S₅ : ℝ) : Prop :=
  S₁ = 50 ∧ S₂ = 62 ∧ S₃ = 73 ∧ S₄ = 70 ∧ (S₁ + S₂ + S₃ + S₄ + S₅) / 5 = 59

theorem fifth_coaster_speed :
  ∀ S₁ S₂ S₃ S₄ S₅ : ℝ,
  rollercoaster_problem S₁ S₂ S₃ S₄ S₅ →
  S₅ = 40 := by
  sorry


end fifth_coaster_speed_l469_46922


namespace ratio_equality_l469_46960

theorem ratio_equality {a b c d : ℝ} (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0) 
  (h5 : a / b = c / d) : a / c = b / d := by
  sorry

end ratio_equality_l469_46960


namespace farmer_wheat_harvest_l469_46945

/-- The farmer's wheat harvest problem -/
theorem farmer_wheat_harvest (estimated : ℕ) (actual : ℕ) 
  (h1 : estimated = 48097) 
  (h2 : actual = 48781) : 
  actual - estimated = 684 := by
  sorry

end farmer_wheat_harvest_l469_46945


namespace marsh_birds_total_l469_46990

theorem marsh_birds_total (initial_geese ducks swans herons : ℕ) 
  (h1 : initial_geese = 58)
  (h2 : ducks = 37)
  (h3 : swans = 15)
  (h4 : herons = 22) :
  initial_geese * 2 + ducks + swans + herons = 190 := by
  sorry

end marsh_birds_total_l469_46990


namespace power_equation_solution_l469_46985

theorem power_equation_solution : ∃ y : ℝ, (12 : ℝ) ^ y * 6 ^ 3 / 432 = 72 ∧ y = 2 := by
  sorry

end power_equation_solution_l469_46985


namespace work_completion_time_l469_46929

/-- The number of days it takes for person A to complete the work -/
def days_A : ℝ := 18

/-- The fraction of work completed by A and B together in 2 days -/
def work_completed_2_days : ℝ := 0.19444444444444442

/-- The number of days it takes for person B to complete the work -/
def days_B : ℝ := 24

theorem work_completion_time :
  (1 / days_A + 1 / days_B) * 2 = work_completed_2_days :=
sorry

end work_completion_time_l469_46929


namespace remaining_jet_bars_to_sell_l469_46912

def weekly_goal : ℕ := 90
def monday_sales : ℕ := 45
def tuesday_sales_difference : ℕ := 16

theorem remaining_jet_bars_to_sell :
  weekly_goal - (monday_sales + (monday_sales - tuesday_sales_difference)) = 16 := by
  sorry

end remaining_jet_bars_to_sell_l469_46912


namespace add_1873_minutes_to_noon_l469_46910

def minutes_in_hour : ℕ := 60
def hours_in_day : ℕ := 24

def add_minutes (start_hour : ℕ) (start_minute : ℕ) (minutes_to_add : ℕ) : (ℕ × ℕ) :=
  let total_minutes := start_hour * minutes_in_hour + start_minute + minutes_to_add
  let final_hour := (total_minutes / minutes_in_hour) % hours_in_day
  let final_minute := total_minutes % minutes_in_hour
  (final_hour, final_minute)

theorem add_1873_minutes_to_noon :
  add_minutes 12 0 1873 = (19, 13) :=
sorry

end add_1873_minutes_to_noon_l469_46910


namespace angle_measure_l469_46915

theorem angle_measure (m1 m2 m3 : ℝ) (h1 : m1 = 80) (h2 : m2 = 35) (h3 : m3 = 25) :
  ∃ m4 : ℝ, m4 = 140 ∧ m1 + m2 + m3 + (180 - m4) = 180 := by
  sorry

end angle_measure_l469_46915


namespace six_digit_divisibility_l469_46994

theorem six_digit_divisibility (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) :
  ∃ (k : ℕ), 100100 * a + 10010 * b + 1001 * c = 7 * 11 * 13 * k := by
  sorry

end six_digit_divisibility_l469_46994


namespace darcies_age_l469_46980

/-- Darcie's age problem -/
theorem darcies_age :
  ∀ (darcie_age mother_age father_age : ℚ),
    darcie_age = (1 / 6 : ℚ) * mother_age →
    mother_age = (4 / 5 : ℚ) * father_age →
    father_age = 30 →
    darcie_age = 4 := by
  sorry

end darcies_age_l469_46980


namespace unique_f_2_l469_46906

/-- A function satisfying the given properties -/
def special_function (f : ℝ → ℝ) : Prop :=
  f 1 = 2 ∧ ∀ x y : ℝ, f (x + y) = f x + f y - x * y

theorem unique_f_2 (f : ℝ → ℝ) (hf : special_function f) : 
  f 2 = 3 ∧ ∀ y : ℝ, f 2 = y → y = 3 :=
sorry

end unique_f_2_l469_46906


namespace unique_root_and_sequence_l469_46982

theorem unique_root_and_sequence : ∃! r : ℝ, 
  (2 * r^3 + 5 * r - 2 = 0) ∧ 
  ∃! (a : ℕ → ℕ), (∀ n, a n < a (n+1)) ∧ 
    (2/5 : ℝ) = ∑' n, r^(a n) ∧
    ∀ n, a n = 3*n - 2 := by
  sorry

end unique_root_and_sequence_l469_46982


namespace max_container_volume_l469_46921

/-- The volume of an open-top container made from a rectangular sheet metal --/
def container_volume (l w h : ℝ) : ℝ := h * (l - 2*h) * (w - 2*h)

/-- The theorem stating the maximum volume of the container --/
theorem max_container_volume :
  let l : ℝ := 90
  let w : ℝ := 48
  ∃ (h : ℝ), 
    (h > 0) ∧ 
    (h < w/2) ∧ 
    (h < l/2) ∧
    (∀ (x : ℝ), x > 0 → x < w/2 → x < l/2 → container_volume l w h ≥ container_volume l w x) ∧
    (container_volume l w h = 16848) ∧
    (h = 6) :=
sorry

end max_container_volume_l469_46921


namespace right_triangle_sides_l469_46972

theorem right_triangle_sides : ∃! (a b c : ℝ), 
  ((a = 1 ∧ b = 2 ∧ c = 2) ∨
   (a = 1 ∧ b = 1 ∧ c = Real.sqrt 3) ∨
   (a = 3 ∧ b = 4 ∧ c = 5) ∨
   (a = 4 ∧ b = 5 ∧ c = 6)) ∧
  (a^2 + b^2 = c^2) :=
by sorry

end right_triangle_sides_l469_46972


namespace valid_paths_count_l469_46978

/-- Represents a point in the grid -/
structure GridPoint where
  x : Nat
  y : Nat

/-- Represents the grid with its dimensions and blocked points -/
structure Grid where
  width : Nat
  height : Nat
  blockedPoints : List GridPoint

/-- Calculates the number of valid paths in the grid -/
def countValidPaths (g : Grid) : Nat :=
  sorry

/-- The specific grid from the problem -/
def problemGrid : Grid :=
  { width := 5
  , height := 3
  , blockedPoints := [⟨2, 1⟩, ⟨3, 1⟩] }

theorem valid_paths_count :
  countValidPaths problemGrid = 39 :=
by sorry

end valid_paths_count_l469_46978


namespace sons_age_l469_46979

/-- Proves that given the conditions, the son's present age is 22 years -/
theorem sons_age (father_age son_age : ℕ) : 
  father_age = son_age + 24 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 22 := by
  sorry

end sons_age_l469_46979


namespace sports_club_probability_l469_46940

/-- The probability of selecting two girls when randomly choosing two members from a group. -/
def probability_two_girls (total : ℕ) (girls : ℕ) : ℚ :=
  (girls.choose 2 : ℚ) / (total.choose 2 : ℚ)

/-- The theorem stating the probability of selecting two girls from the sports club. -/
theorem sports_club_probability :
  let total := 15
  let girls := 8
  probability_two_girls total girls = 4 / 15 := by
  sorry

end sports_club_probability_l469_46940


namespace intersection_of_A_and_B_l469_46996

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | |x| > 1}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by sorry

end intersection_of_A_and_B_l469_46996


namespace largest_integer_solution_inequality_l469_46970

theorem largest_integer_solution_inequality (x : ℤ) :
  (∀ y : ℤ, -y ≥ 2*y + 3 → y ≤ -1) ∧ (-(-1) ≥ 2*(-1) + 3) :=
by sorry

end largest_integer_solution_inequality_l469_46970


namespace parallel_lines_exist_points_not_on_line_l469_46927

-- Define the line equation
def line_equation (α x y : ℝ) : Prop :=
  Real.cos α * (x - 2) + Real.sin α * (y + 1) = 1

-- Statement ②: There exist different real numbers α₁, α₂, such that the corresponding lines l₁, l₂ are parallel
theorem parallel_lines_exist : ∃ α₁ α₂ : ℝ, α₁ ≠ α₂ ∧
  ∀ x y : ℝ, line_equation α₁ x y ↔ line_equation α₂ x y :=
sorry

-- Statement ③: There are at least two points in the coordinate plane that are not on the line l
theorem points_not_on_line : ∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧
  (∀ α : ℝ, ¬line_equation α x₁ y₁ ∧ ¬line_equation α x₂ y₂) :=
sorry

end parallel_lines_exist_points_not_on_line_l469_46927


namespace triangle_count_after_12_iterations_l469_46989

/-- The number of triangles after n iterations of the division process -/
def num_triangles (n : ℕ) : ℕ := 3^n

/-- The side length of triangles after n iterations -/
def side_length (n : ℕ) : ℚ := 1 / 2^n

theorem triangle_count_after_12_iterations :
  num_triangles 12 = 531441 ∧ side_length 12 = 1 / 2^12 := by
  sorry

end triangle_count_after_12_iterations_l469_46989


namespace bird_families_left_l469_46966

theorem bird_families_left (initial_families : ℕ) (families_flown_away : ℕ) : 
  initial_families = 67 → families_flown_away = 32 → initial_families - families_flown_away = 35 :=
by sorry

end bird_families_left_l469_46966


namespace equation_one_solution_l469_46988

theorem equation_one_solution : 
  ∃ x₁ x₂ : ℝ, (x₁ - 2)^2 - 5 = 0 ∧ (x₂ - 2)^2 - 5 = 0 ∧ x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 := by
  sorry


end equation_one_solution_l469_46988


namespace bons_win_probability_l469_46999

theorem bons_win_probability : 
  let p : ℝ := (1 : ℝ) / 6  -- Probability of rolling a six
  let q : ℝ := 1 - p        -- Probability of not rolling a six
  ∃ (win_prob : ℝ), 
    win_prob = q * p + q * q * win_prob ∧ 
    win_prob = 5 / 11 := by
  sorry

end bons_win_probability_l469_46999


namespace length_of_AB_l469_46903

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the line with slope tan(30°) passing through (3, 0)
def line (x y : ℝ) : Prop := y = (Real.sqrt 3 / 3) * (x - 3)

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  hyperbola A.1 A.2 ∧ hyperbola B.1 B.2 ∧
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem length_of_AB (A B : ℝ × ℝ) :
  intersection_points A B →
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = (16 / 5) * Real.sqrt 3 :=
sorry

end length_of_AB_l469_46903


namespace zigzag_angle_theorem_l469_46934

/-- In a rectangle with a zigzag line, given specific angles, prove that ∠CDE is 11° --/
theorem zigzag_angle_theorem (ABC BCD DEF EFG : ℝ) (h1 : ABC = 10) (h2 : BCD = 14) 
  (h3 : DEF = 26) (h4 : EFG = 33) : ∃ (CDE : ℝ), CDE = 11 := by
  sorry

end zigzag_angle_theorem_l469_46934


namespace gravel_path_rate_l469_46933

/-- Given a rectangular plot with an inner gravel path, calculate the rate per square meter for gravelling. -/
theorem gravel_path_rate (length width path_width total_cost : ℝ) 
  (h1 : length = 100)
  (h2 : width = 70)
  (h3 : path_width = 2.5)
  (h4 : total_cost = 742.5) : 
  total_cost / ((length * width) - ((length - 2 * path_width) * (width - 2 * path_width))) = 0.9 := by
  sorry

end gravel_path_rate_l469_46933


namespace union_of_M_and_N_N_is_possible_set_l469_46991

def M : Set ℕ := {1, 2}
def N : Set ℕ := {1, 3}

theorem union_of_M_and_N :
  M ∪ N = {1, 2, 3} := by sorry

theorem N_is_possible_set :
  M = {1, 2} → M ∪ N = {1, 2, 3} → N = {1, 3} := by sorry

end union_of_M_and_N_N_is_possible_set_l469_46991
