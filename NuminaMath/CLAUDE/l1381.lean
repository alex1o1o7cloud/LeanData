import Mathlib

namespace factor_tree_value_l1381_138191

-- Define the structure of the factor tree
structure FactorTree :=
  (A B C D E : ℝ)

-- Define the conditions of the factor tree
def valid_factor_tree (t : FactorTree) : Prop :=
  t.A^2 = t.B * t.C ∧
  t.B = 2 * t.D ∧
  t.D = 2 * 4 ∧
  t.C = 7 * t.E ∧
  t.E = 7 * 2

-- Theorem statement
theorem factor_tree_value (t : FactorTree) (h : valid_factor_tree t) : 
  t.A = 28 * Real.sqrt 2 := by
  sorry

end factor_tree_value_l1381_138191


namespace x_equals_one_necessary_and_sufficient_l1381_138178

theorem x_equals_one_necessary_and_sufficient :
  ∀ x : ℝ, (x^2 - 2*x + 1 = 0) ↔ (x = 1) := by
  sorry

end x_equals_one_necessary_and_sufficient_l1381_138178


namespace sara_hotdog_cost_l1381_138194

/-- The cost of Sara's lunch items -/
structure LunchCost where
  total : ℝ
  salad : ℝ
  hotdog : ℝ

/-- Sara's lunch satisfies the given conditions -/
def sara_lunch : LunchCost where
  total := 10.46
  salad := 5.1
  hotdog := 5.36

/-- Theorem: Sara's hotdog cost $5.36 -/
theorem sara_hotdog_cost : sara_lunch.hotdog = 5.36 := by
  sorry

#check sara_hotdog_cost

end sara_hotdog_cost_l1381_138194


namespace midpoint_sum_midpoint_sum_specific_l1381_138164

/-- Given a line segment with endpoints (3, 4) and (9, 18), 
    the sum of the coordinates of its midpoint is 17. -/
theorem midpoint_sum : ℝ → ℝ → ℝ → ℝ → ℝ := fun x₁ y₁ x₂ y₂ =>
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  midpoint_x + midpoint_y

#check midpoint_sum 3 4 9 18 = 17

theorem midpoint_sum_specific : midpoint_sum 3 4 9 18 = 17 := by
  sorry

end midpoint_sum_midpoint_sum_specific_l1381_138164


namespace seven_books_arrangement_l1381_138136

/-- The number of distinct arrangements of books on a shelf -/
def book_arrangements (total : ℕ) (group1 : ℕ) (group2 : ℕ) : ℕ :=
  Nat.factorial total / (Nat.factorial group1 * Nat.factorial group2)

/-- Theorem stating the number of distinct arrangements for the given book configuration -/
theorem seven_books_arrangement :
  book_arrangements 7 3 2 = 420 := by
  sorry

end seven_books_arrangement_l1381_138136


namespace sqrt_3_irrational_l1381_138190

theorem sqrt_3_irrational : Irrational (Real.sqrt 3) := by
  sorry

end sqrt_3_irrational_l1381_138190


namespace geometric_sequence_min_S3_l1381_138175

/-- Given a geometric sequence with positive terms, prove that the minimum value of S_3 is 6 -/
theorem geometric_sequence_min_S3 (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →  -- positive terms
  (∀ n, a (n + 1) = q * a n) →  -- geometric sequence
  a 4 * a 8 = 2 * a 10 →  -- given condition
  (∃ S : ℕ → ℝ, ∀ n, S n = (a 1) * (1 - q^n) / (1 - q)) →  -- sum of first n terms
  (∃ min_S3 : ℝ, ∀ S3, S3 = a 1 + a 2 + a 3 → S3 ≥ min_S3 ∧ min_S3 = 6) :=
by sorry

end geometric_sequence_min_S3_l1381_138175


namespace quadratic_equation_solution_l1381_138140

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ 2*x^2 - 3*x - 1
  ∃ x₁ x₂ : ℝ, x₁ = (3 + Real.sqrt 17) / 4 ∧ 
             x₂ = (3 - Real.sqrt 17) / 4 ∧ 
             f x₁ = 0 ∧ f x₂ = 0 ∧
             ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ := by
  sorry

end quadratic_equation_solution_l1381_138140


namespace f_3_range_l1381_138146

/-- Given a quadratic function f(x) = ax^2 - c satisfying certain conditions,
    prove that f(3) is within a specific range. -/
theorem f_3_range (a c : ℝ) (f : ℝ → ℝ) 
    (h_def : ∀ x, f x = a * x^2 - c)
    (h_1 : -4 ≤ f 1 ∧ f 1 ≤ -1)
    (h_2 : -1 ≤ f 2 ∧ f 2 ≤ 5) :
  -1 ≤ f 3 ∧ f 3 ≤ 20 := by
  sorry

end f_3_range_l1381_138146


namespace min_value_sum_reciprocals_l1381_138145

-- Define the vectors
def OA : ℝ × ℝ := (-2, 4)
def OB (a : ℝ) : ℝ × ℝ := (-a, 2)
def OC (b : ℝ) : ℝ × ℝ := (b, 0)

-- Define the collinearity condition
def collinear (A B C : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), B.1 - A.1 = t * (C.1 - A.1) ∧ B.2 - A.2 = t * (C.2 - A.2)

-- State the theorem
theorem min_value_sum_reciprocals (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_collinear : collinear OA (OB a) (OC b)) :
  (1 / a + 1 / b) ≥ (3 + 2 * Real.sqrt 2) / 2 :=
by sorry

end min_value_sum_reciprocals_l1381_138145


namespace tangent_point_bounds_l1381_138150

/-- A point (a,b) through which two distinct tangent lines can be drawn to the curve y = e^x -/
structure TangentPoint where
  a : ℝ
  b : ℝ
  two_tangents : ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    b = Real.exp t₁ * (a - t₁ + 1) ∧
    b = Real.exp t₂ * (a - t₂ + 1)

/-- If two distinct tangent lines to y = e^x can be drawn through (a,b), then 0 < b < e^a -/
theorem tangent_point_bounds (p : TangentPoint) : 0 < p.b ∧ p.b < Real.exp p.a := by
  sorry

end tangent_point_bounds_l1381_138150


namespace mitzi_remaining_money_l1381_138181

/-- Proves that Mitzi has $9 left after her amusement park expenses -/
theorem mitzi_remaining_money (initial_amount ticket_cost food_cost tshirt_cost : ℕ) 
  (h1 : initial_amount = 75)
  (h2 : ticket_cost = 30)
  (h3 : food_cost = 13)
  (h4 : tshirt_cost = 23) :
  initial_amount - (ticket_cost + food_cost + tshirt_cost) = 9 := by
  sorry

end mitzi_remaining_money_l1381_138181


namespace tree_growth_rate_l1381_138135

/-- Proves that a tree growing from 52 feet to 92 feet in 8 years has an annual growth rate of 5 feet --/
theorem tree_growth_rate (initial_height : ℝ) (final_height : ℝ) (years : ℕ) 
  (h1 : initial_height = 52)
  (h2 : final_height = 92)
  (h3 : years = 8) :
  (final_height - initial_height) / years = 5 := by
  sorry

end tree_growth_rate_l1381_138135


namespace center_transformation_l1381_138129

/-- Reflects a point across the y-axis -/
def reflectY (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Rotates a point 90 degrees clockwise around the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

/-- Translates a point up by a given amount -/
def translateUp (p : ℝ × ℝ) (amount : ℝ) : ℝ × ℝ :=
  (p.1, p.2 + amount)

/-- Applies all transformations to a point -/
def applyTransformations (p : ℝ × ℝ) : ℝ × ℝ :=
  translateUp (rotate90Clockwise (reflectY p)) 4

theorem center_transformation :
  applyTransformations (3, -5) = (-5, 7) := by
  sorry

end center_transformation_l1381_138129


namespace solution_set_when_a_is_3_range_of_a_for_minimum_2_l1381_138149

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Theorem for the first part of the problem
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Theorem for the second part of the problem
theorem range_of_a_for_minimum_2 :
  {a : ℝ | ∀ x₁ : ℝ, f a x₁ ≥ 2} = {a : ℝ | a ≥ 3 ∨ a ≤ -1} := by sorry

end solution_set_when_a_is_3_range_of_a_for_minimum_2_l1381_138149


namespace least_integer_with_ten_factors_l1381_138119

-- Define a function to count the number of distinct positive factors
def countFactors (n : ℕ) : ℕ := sorry

-- Define a function to check if a number has exactly ten distinct positive factors
def hasTenFactors (n : ℕ) : Prop :=
  countFactors n = 10

-- Theorem statement
theorem least_integer_with_ten_factors :
  ∀ n : ℕ, n > 0 → hasTenFactors n → n ≥ 48 :=
sorry

end least_integer_with_ten_factors_l1381_138119


namespace sum_of_three_roots_is_zero_l1381_138148

/-- Given two quadratic polynomials with coefficients a and b, 
    where each has two distinct roots and their product has exactly three distinct roots,
    prove that the sum of these three roots is 0. -/
theorem sum_of_three_roots_is_zero (a b : ℝ) : 
  (∃ x₁ x₂ x₃ x₄ : ℝ, x₁ ≠ x₂ ∧ x₃ ≠ x₄ ∧ 
    (∀ x : ℝ, x^2 + a*x + b = 0 ↔ (x = x₁ ∨ x = x₂)) ∧
    (∀ x : ℝ, x^2 + b*x + a = 0 ↔ (x = x₃ ∨ x = x₄))) →
  (∃! y₁ y₂ y₃ : ℝ, y₁ ≠ y₂ ∧ y₁ ≠ y₃ ∧ y₂ ≠ y₃ ∧
    (∀ x : ℝ, (x^2 + a*x + b) * (x^2 + b*x + a) = 0 ↔ (x = y₁ ∨ x = y₂ ∨ x = y₃))) →
  ∃ y₁ y₂ y₃ : ℝ, y₁ + y₂ + y₃ = 0 :=
by sorry


end sum_of_three_roots_is_zero_l1381_138148


namespace unique_extremum_implies_a_range_l1381_138105

noncomputable def f (a x : ℝ) : ℝ := a * (x - 2) * Real.exp x + Real.log x + 1 / x

theorem unique_extremum_implies_a_range (a : ℝ) :
  (∃! x, ∀ y, f a y ≤ f a x) →
  (∃ x, ∀ y, f a y ≤ f a x ∧ f a x > 0) →
  0 ≤ a ∧ a < 1 / Real.exp 1 :=
sorry

end unique_extremum_implies_a_range_l1381_138105


namespace sarahs_age_l1381_138169

theorem sarahs_age (ana billy mark sarah : ℕ) : 
  ana + 3 = 15 →
  billy = ana / 2 →
  mark = billy + 4 →
  sarah = 3 * mark - 4 →
  sarah = 26 := by
sorry

end sarahs_age_l1381_138169


namespace unknown_number_proof_l1381_138117

theorem unknown_number_proof (x : ℝ) : x^2 + 94^2 = 19872 → x = 105 := by
  sorry

end unknown_number_proof_l1381_138117


namespace parabola_equidistant_point_l1381_138144

/-- 
For a parabola y^2 = 2px where p > 0, with point P(2, 2p) on the parabola, 
origin O(0, 0), and focus F, the point M satisfying |MP| = |MO| = |MF| 
has coordinates (1/4, 7/4).
-/
theorem parabola_equidistant_point (p : ℝ) (h : p > 0) : 
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 2*p*x}
  let P := (2, 2*p)
  let O := (0, 0)
  let F := (p/2, 0)
  ∃ M : ℝ × ℝ, M ∈ parabola ∧ 
    dist M P = dist M O ∧ 
    dist M O = dist M F ∧ 
    M = (1/4, 7/4) :=
by sorry

end parabola_equidistant_point_l1381_138144


namespace quiz_probability_l1381_138126

theorem quiz_probability (n : ℕ) : 
  (1 : ℚ) / 3 * (1 / 2) ^ n = 1 / 12 → n = 2 :=
by sorry

end quiz_probability_l1381_138126


namespace cosine_sum_l1381_138127

theorem cosine_sum (α β : Real) : 
  0 < α ∧ α < Real.pi/2 ∧
  -Real.pi/2 < β ∧ β < 0 ∧
  Real.cos (Real.pi/4 + α) = 1/3 ∧
  Real.cos (Real.pi/4 - β/2) = Real.sqrt 3/3 →
  Real.cos (α + β/2) = 5 * Real.sqrt 3/9 := by
sorry

end cosine_sum_l1381_138127


namespace complex_expression_value_l1381_138185

theorem complex_expression_value : 
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2000 := by
  sorry

end complex_expression_value_l1381_138185


namespace smallest_number_of_cubes_l1381_138177

/-- The number of faces on each cube -/
def faces_per_cube : ℕ := 6

/-- The number of digits (0 to 9) -/
def num_digits : ℕ := 10

/-- The length of the number we need to be able to form -/
def number_length : ℕ := 30

/-- The minimum number of each non-zero digit needed -/
def min_nonzero_digits : ℕ := number_length

/-- The minimum number of zero digits needed -/
def min_zero_digits : ℕ := number_length - 1

/-- The total minimum number of digit instances needed -/
def total_min_digits : ℕ := min_nonzero_digits * (num_digits - 1) + min_zero_digits

/-- The smallest number of cubes needed to form any 30-digit number -/
def min_cubes : ℕ := 50

theorem smallest_number_of_cubes : 
  faces_per_cube * min_cubes ≥ total_min_digits ∧ 
  ∀ n : ℕ, n < min_cubes → faces_per_cube * n < total_min_digits :=
by sorry

end smallest_number_of_cubes_l1381_138177


namespace toothpick_pattern_200th_stage_l1381_138167

/-- 
Given an arithmetic sequence where:
- a is the first term
- d is the common difference
- n is the term number
This function calculates the nth term of the sequence.
-/
def arithmeticSequenceTerm (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a + (n - 1) * d

/--
Theorem: In an arithmetic sequence where the first term is 6 and the common difference is 5,
the 200th term is equal to 1001.
-/
theorem toothpick_pattern_200th_stage :
  arithmeticSequenceTerm 6 5 200 = 1001 := by
  sorry

#eval arithmeticSequenceTerm 6 5 200

end toothpick_pattern_200th_stage_l1381_138167


namespace change_is_five_l1381_138128

/-- Given a meal cost, drink cost, tip percentage, and payment amount, 
    calculate the change received. -/
def calculate_change (meal_cost drink_cost tip_percentage payment : ℚ) : ℚ :=
  let total_before_tip := meal_cost + drink_cost
  let tip_amount := total_before_tip * (tip_percentage / 100)
  let total_with_tip := total_before_tip + tip_amount
  payment - total_with_tip

/-- Theorem stating that given the specified costs and payment, 
    the change received is $5. -/
theorem change_is_five :
  calculate_change 10 2.5 20 20 = 5 := by
  sorry

end change_is_five_l1381_138128


namespace parallel_lines_b_value_l1381_138125

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ : ℝ} : 
  (∃ (b₁ b₂ : ℝ), ∀ x y, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) → m₁ = m₂

/-- Given two lines 3y - 3b = 9x and y - 2 = (b + 9)x that are parallel, prove b = -6 -/
theorem parallel_lines_b_value (b : ℝ) :
  (∃ (y₁ y₂ : ℝ → ℝ), (∀ x, 3 * y₁ x - 3 * b = 9 * x) ∧ 
                       (∀ x, y₂ x - 2 = (b + 9) * x) ∧
                       (∀ x y, y = y₁ x ↔ y = y₂ x)) →
  b = -6 := by
sorry

end parallel_lines_b_value_l1381_138125


namespace joan_balloons_l1381_138120

/-- The number of blue balloons Joan has after gaining more -/
def total_balloons (initial : ℕ) (gained : ℕ) : ℕ :=
  initial + gained

/-- Theorem stating that Joan has 95 blue balloons after gaining more -/
theorem joan_balloons : total_balloons 72 23 = 95 := by
  sorry

end joan_balloons_l1381_138120


namespace minimum_concerts_required_l1381_138184

/-- Represents a concert configuration --/
structure Concert where
  performers : Finset Nat
  listeners : Finset Nat

/-- Represents the festival configuration --/
structure Festival where
  musicians : Finset Nat
  concerts : List Concert

/-- Checks if a festival configuration is valid --/
def isValidFestival (f : Festival) : Prop :=
  f.musicians.card = 6 ∧
  ∀ c ∈ f.concerts, c.performers ⊆ f.musicians ∧
                    c.listeners ⊆ f.musicians ∧
                    c.performers ∩ c.listeners = ∅ ∧
                    c.performers ∪ c.listeners = f.musicians

/-- Checks if each musician has listened to all others --/
def allMusiciansListened (f : Festival) : Prop :=
  ∀ m ∈ f.musicians, ∀ n ∈ f.musicians, m ≠ n →
    ∃ c ∈ f.concerts, m ∈ c.listeners ∧ n ∈ c.performers

/-- The main theorem --/
theorem minimum_concerts_required :
  ∀ f : Festival,
    isValidFestival f →
    allMusiciansListened f →
    f.concerts.length ≥ 4 :=
by sorry

end minimum_concerts_required_l1381_138184


namespace volume_ratio_l1381_138161

variable (V_A V_B V_C : ℝ)

theorem volume_ratio 
  (h1 : V_A = (V_B + V_C) / 2)
  (h2 : V_B = (V_A + V_C) / 5)
  (h3 : V_C ≠ 0) :
  V_C / (V_A + V_B) = 1 := by
sorry

end volume_ratio_l1381_138161


namespace arithmetic_sequence_sum_l1381_138179

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end arithmetic_sequence_sum_l1381_138179


namespace complement_intersection_A_B_l1381_138158

-- Define the sets A and B
def A : Set ℝ := {x | |x - 1| < 2}
def B : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem complement_intersection_A_B :
  (Set.univ \ (A ∩ B)) = {x : ℝ | x < 1 ∨ x ≥ 3} := by sorry

end complement_intersection_A_B_l1381_138158


namespace tan_alpha_plus_pi_fourth_l1381_138156

theorem tan_alpha_plus_pi_fourth (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  Real.tan (α + π / 4) = 3 / 22 := by
  sorry

end tan_alpha_plus_pi_fourth_l1381_138156


namespace coffee_container_weight_l1381_138183

def suki_bags : ℝ := 6.5
def suki_weight_per_bag : ℝ := 22
def jimmy_bags : ℝ := 4.5
def jimmy_weight_per_bag : ℝ := 18
def num_containers : ℕ := 28

theorem coffee_container_weight :
  (suki_bags * suki_weight_per_bag + jimmy_bags * jimmy_weight_per_bag) / num_containers = 8 := by
  sorry

end coffee_container_weight_l1381_138183


namespace danielle_travel_time_l1381_138113

-- Define the speeds and times
def chase_speed : ℝ := 1 -- Normalized speed
def chase_time : ℝ := 180 -- Minutes
def cameron_speed : ℝ := 2 * chase_speed
def danielle_speed : ℝ := 3 * cameron_speed

-- Define the distance (constant for all travelers)
def distance : ℝ := chase_speed * chase_time

-- Theorem to prove
theorem danielle_travel_time : 
  (distance / danielle_speed) = 30 := by
sorry

end danielle_travel_time_l1381_138113


namespace exponent_multiplication_l1381_138123

theorem exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 := by
  sorry

end exponent_multiplication_l1381_138123


namespace stock_price_change_l1381_138131

theorem stock_price_change (initial_price : ℝ) (initial_price_pos : initial_price > 0) : 
  let week1 := initial_price * 1.3
  let week2 := week1 * 0.75
  let week3 := week2 * 1.2
  let week4 := week3 * 0.85
  week4 = initial_price := by sorry

end stock_price_change_l1381_138131


namespace function_g_theorem_l1381_138172

theorem function_g_theorem (g : ℝ → ℝ) 
  (h1 : g 0 = 2)
  (h2 : ∀ x y : ℝ, g (x * y) = g (x^2 + y^2) + 2 * (x - y)^2) :
  ∀ x : ℝ, g x = 2 - 2 * x :=
by sorry

end function_g_theorem_l1381_138172


namespace odd_function_inequality_l1381_138147

-- Define f as a function from real numbers to real numbers
variable (f : ℝ → ℝ)

-- Define the property of f being an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the condition for x > 0
def positive_condition (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, x * (deriv f x) + 2 * f x > 0

-- State the theorem
theorem odd_function_inequality (h1 : is_odd f) (h2 : positive_condition f) :
  4 * f 2 < 9 * f 3 :=
sorry

end odd_function_inequality_l1381_138147


namespace subtraction_problem_l1381_138162

theorem subtraction_problem (minuend : ℝ) (difference : ℝ) (subtrahend : ℝ)
  (h1 : minuend = 98.2)
  (h2 : difference = 17.03)
  (h3 : subtrahend = minuend - difference) :
  subtrahend = 81.17 := by
  sorry

end subtraction_problem_l1381_138162


namespace natashas_average_speed_l1381_138100

/-- Natasha's hill climbing problem -/
theorem natashas_average_speed 
  (time_up : ℝ) 
  (time_down : ℝ) 
  (speed_up : ℝ) 
  (h1 : time_up = 4)
  (h2 : time_down = 2)
  (h3 : speed_up = 1.5)
  : (2 * speed_up * time_up) / (time_up + time_down) = 2 := by
  sorry

#check natashas_average_speed

end natashas_average_speed_l1381_138100


namespace find_number_l1381_138192

theorem find_number : ∃! x : ℝ, (x + 82 + 90 + 88 + 84) / 5 = 88 := by
  sorry

end find_number_l1381_138192


namespace log_identity_l1381_138187

theorem log_identity : (Real.log 2 / Real.log 10) ^ 2 + (Real.log 5 / Real.log 10) ^ 2 + 2 * (Real.log 2 / Real.log 10) * (Real.log 5 / Real.log 10) = 1 := by
  sorry

end log_identity_l1381_138187


namespace sum_of_even_numbers_1_to_200_l1381_138108

theorem sum_of_even_numbers_1_to_200 : 
  (Finset.filter (fun n => Even n) (Finset.range 201)).sum id = 10100 := by
  sorry

end sum_of_even_numbers_1_to_200_l1381_138108


namespace complement_intersection_theorem_l1381_138157

def U : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {3, 4, 5}
def N : Set Nat := {2, 3}

theorem complement_intersection_theorem :
  (U \ N) ∩ M = {4, 5} := by sorry

end complement_intersection_theorem_l1381_138157


namespace multiple_in_difference_l1381_138176

theorem multiple_in_difference (n m : ℤ) (h1 : n = -7) (h2 : 3 * n = m * n - 7) : m = 2 := by
  sorry

end multiple_in_difference_l1381_138176


namespace n_has_9_digits_l1381_138160

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2))

/-- Function to count the number of digits in a natural number -/
def count_digits (x : ℕ) : ℕ := sorry

theorem n_has_9_digits : count_digits n = 9 := by sorry

end n_has_9_digits_l1381_138160


namespace different_orders_count_l1381_138174

def memo_count : ℕ := 11
def processed_memos : Finset ℕ := {9, 10}

def possible_remaining_memos : Finset ℕ := Finset.range 9 ∪ {11}

def insert_positions (n : ℕ) : ℕ := n + 2

/-- The number of different orders for processing the remaining memos -/
def different_orders : ℕ :=
  (Finset.range 9).sum fun j =>
    (Nat.choose 8 j) * (insert_positions j)

theorem different_orders_count :
  different_orders = 1536 := by
  sorry

end different_orders_count_l1381_138174


namespace triangle_equality_l1381_138195

-- Define the triangle ADC
structure TriangleADC where
  AD : ℝ
  DC : ℝ
  D : ℝ
  h1 : AD = DC
  h2 : D = 100

-- Define the triangle CAB
structure TriangleCAB where
  CA : ℝ
  AB : ℝ
  A : ℝ
  h3 : CA = AB
  h4 : A = 20

-- Define the theorem
theorem triangle_equality (ADC : TriangleADC) (CAB : TriangleCAB) :
  CAB.AB = ADC.DC + CAB.AB - CAB.CA :=
sorry

end triangle_equality_l1381_138195


namespace initial_candies_count_l1381_138152

-- Define the given conditions
def candies_given_to_chloe : ℝ := 28.0
def candies_left : ℕ := 6

-- Define the theorem to prove
theorem initial_candies_count : 
  candies_given_to_chloe + candies_left = 34.0 := by
  sorry

end initial_candies_count_l1381_138152


namespace exp_log_properties_l1381_138155

-- Define the exponential and logarithmic functions
noncomputable def exp (a : ℝ) (x : ℝ) : ℝ := a^x
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- Theorem for the properties of exponential and logarithmic functions
theorem exp_log_properties :
  -- Domain and range of exponential function
  (∀ x : ℝ, ∃ y : ℝ, exp 2 x = y) ∧
  (∀ y : ℝ, y > 0 → ∃ x : ℝ, exp 2 x = y) ∧
  (exp 2 0 = 1) ∧
  -- Domain and range of logarithmic function
  (∀ x : ℝ, x > 0 → ∃ y : ℝ, log 2 x = y) ∧
  (∀ y : ℝ, ∃ x : ℝ, x > 0 ∧ log 2 x = y) ∧
  (log 2 1 = 0) ∧
  -- Logarithm properties
  (∀ a M N : ℝ, a > 0 ∧ a ≠ 1 ∧ M > 0 ∧ N > 0 →
    log a (M * N) = log a M + log a N) ∧
  (∀ a N : ℝ, a > 0 ∧ a ≠ 1 ∧ N > 0 →
    exp a (log a N) = N) ∧
  (∀ a b m n : ℝ, a > 0 ∧ a ≠ 1 ∧ b > 0 ∧ m ≠ 0 →
    log (exp a m) (exp b n) = (n / m) * log a b) :=
by sorry

#check exp_log_properties

end exp_log_properties_l1381_138155


namespace t_shape_area_is_20_l1381_138104

/-- Represents the structure inside the square WXYZ -/
structure InternalStructure where
  top_left_side : ℕ
  top_right_side : ℕ
  bottom_right_side : ℕ
  bottom_left_side : ℕ
  rectangle_width : ℕ
  rectangle_height : ℕ

/-- Calculates the area of the T-shaped region -/
def t_shape_area (s : InternalStructure) : ℕ :=
  s.top_left_side * s.top_left_side +
  s.bottom_right_side * s.bottom_right_side +
  s.bottom_left_side * s.bottom_left_side +
  s.rectangle_width * s.rectangle_height

/-- The theorem stating that the area of the T-shaped region is 20 -/
theorem t_shape_area_is_20 (s : InternalStructure)
  (h1 : s.top_left_side = 2)
  (h2 : s.top_right_side = 2)
  (h3 : s.bottom_right_side = 2)
  (h4 : s.bottom_left_side = 2)
  (h5 : s.rectangle_width = 4)
  (h6 : s.rectangle_height = 2) :
  t_shape_area s = 20 := by
  sorry

end t_shape_area_is_20_l1381_138104


namespace profit_A_range_max_a_value_l1381_138138

-- Define the profit functions
def profit_A_before (x : ℝ) : ℝ := 120000 * 500

def profit_A_after (x : ℝ) : ℝ := 120000 * (500 - x) * (1 + 0.005 * x)

def profit_B (x a : ℝ) : ℝ := 120000 * x * (a - 0.013 * x)

-- Theorem for part (I)
theorem profit_A_range (x : ℝ) :
  (0 < x ∧ x ≤ 300) ↔ profit_A_after x ≥ profit_A_before x :=
sorry

-- Theorem for part (II)
theorem max_a_value :
  ∃ (a : ℝ), a = 5.5 ∧
  ∀ (x : ℝ), 0 < x → x ≤ 300 →
  (∀ (a' : ℝ), a' > 0 → profit_B x a' ≤ profit_A_after x → a' ≤ a) :=
sorry

end profit_A_range_max_a_value_l1381_138138


namespace absolute_value_ratio_l1381_138112

theorem absolute_value_ratio (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x^2 + y^2 = 5*x*y) :
  |((x + y) / (x - y))| = Real.sqrt (7/3) := by
sorry

end absolute_value_ratio_l1381_138112


namespace otimes_composition_l1381_138114

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^3 + y^3 - x - y

-- Theorem statement
theorem otimes_composition (a b : ℝ) : 
  otimes a (otimes b a) = a^3 + (b^3 + a^3 - b - a)^3 - a - (b^3 + a^3 - b - a) := by
  sorry

end otimes_composition_l1381_138114


namespace bobbys_shoes_cost_l1381_138142

/-- The total cost for Bobby's handmade shoes -/
def total_cost (mold_cost labor_rate hours discount : ℝ) : ℝ :=
  mold_cost + discount * labor_rate * hours

/-- Theorem stating the total cost for Bobby's handmade shoes is $730 -/
theorem bobbys_shoes_cost :
  total_cost 250 75 8 0.8 = 730 := by
  sorry

end bobbys_shoes_cost_l1381_138142


namespace cleaner_used_is_80_l1381_138170

/-- Represents the flow rate of cleaner through a pipe at different time intervals -/
structure FlowRate :=
  (initial : ℝ)
  (after15min : ℝ)
  (after25min : ℝ)

/-- Calculates the total amount of cleaner used over a 30-minute period -/
def totalCleanerUsed (flow : FlowRate) : ℝ :=
  flow.initial * 15 + flow.after15min * 10 + flow.after25min * 5

/-- The flow rates given in the problem -/
def problemFlow : FlowRate :=
  { initial := 2
  , after15min := 3
  , after25min := 4 }

/-- Theorem stating that the total cleaner used is 80 ounces -/
theorem cleaner_used_is_80 : totalCleanerUsed problemFlow = 80 := by
  sorry

end cleaner_used_is_80_l1381_138170


namespace spiders_can_catch_fly_l1381_138180

-- Define the cube
structure Cube where
  vertices : Finset (Fin 8)
  edges : Finset (Fin 12)

-- Define the creatures
inductive Creature
| Spider
| Fly

-- Define the position of a creature on the cube
structure Position where
  creature : Creature
  vertex : Fin 8

-- Define the speed of creatures
def speed (c : Creature) : ℕ :=
  match c with
  | Creature.Spider => 1
  | Creature.Fly => 3

-- Define the initial state
def initial_state (cube : Cube) : Finset Position :=
  sorry

-- Define the catching condition
def can_catch (cube : Cube) (positions : Finset Position) : Prop :=
  sorry

-- The main theorem
theorem spiders_can_catch_fly (cube : Cube) :
  ∃ (final_positions : Finset Position),
    can_catch cube final_positions :=
  sorry

end spiders_can_catch_fly_l1381_138180


namespace sum_squares_distances_to_chord_ends_l1381_138171

/-- Given a circle with radius R and a point M on its diameter at distance a from the center,
    the sum of squares of distances from M to the ends of any chord parallel to the diameter
    is equal to 2(a² + R²). -/
theorem sum_squares_distances_to_chord_ends
  (R a : ℝ) -- R is the radius, a is the distance from M to the center
  (h₁ : 0 < R) -- R is positive (circle has positive radius)
  (h₂ : 0 ≤ a ∧ a ≤ 2*R) -- M is on the diameter, so 0 ≤ a ≤ 2R
  : ∀ A B : ℝ × ℝ, -- For any points A and B
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * R^2 → -- If AB is a chord (distance AB = diameter)
    (∃ k : ℝ, A.2 = k ∧ B.2 = k) → -- If AB is parallel to x-axis (assuming diameter along x-axis)
    (A.1 - a)^2 + A.2^2 + (B.1 - a)^2 + B.2^2 = 2 * (a^2 + R^2) :=
by sorry

end sum_squares_distances_to_chord_ends_l1381_138171


namespace dragons_games_count_l1381_138186

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (0.4 : ℝ) * initial_games →
    (initial_wins + 5 : ℝ) / (initial_games + 8) = 0.55 →
    initial_games + 8 = 12 :=
by
  sorry

end dragons_games_count_l1381_138186


namespace quadratic_inequality_solution_l1381_138115

/-- 
Given that x·(4x + 3) < d if and only when x ∈ (-5/2, 1), prove that d = 10
-/
theorem quadratic_inequality_solution (d : ℝ) : 
  (∀ x : ℝ, x * (4 * x + 3) < d ↔ -5/2 < x ∧ x < 1) → d = 10 := by
  sorry

end quadratic_inequality_solution_l1381_138115


namespace division_problem_l1381_138143

theorem division_problem (dividend : ℕ) (divisor : ℝ) (remainder : ℕ) (quotient : ℕ) :
  dividend = 17698 →
  divisor = 198.69662921348313 →
  remainder = 14 →
  quotient = 89 →
  (dividend : ℝ) = divisor * (quotient : ℝ) + (remainder : ℝ) :=
by
  sorry

#eval (17698 : ℝ) - 198.69662921348313 * 89 - 14

end division_problem_l1381_138143


namespace intersection_M_P_l1381_138107

-- Define the sets M and P
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x}
def P : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt (x - 1)}

-- State the theorem
theorem intersection_M_P : M ∩ P = {x : ℝ | x ≥ 1} := by sorry

end intersection_M_P_l1381_138107


namespace unpainted_cubes_in_6x6x6_l1381_138141

/-- Represents a cube with painted faces -/
structure PaintedCube where
  size : Nat
  total_cubes : Nat
  painted_squares_per_face : Nat
  num_faces : Nat

/-- The number of unpainted cubes in a painted cube -/
def num_unpainted_cubes (c : PaintedCube) : Nat :=
  c.total_cubes - (c.painted_squares_per_face * c.num_faces / 2)

/-- Theorem: In a 6x6x6 cube with 216 unit cubes and 4 painted squares on each of 6 faces,
    the number of unpainted cubes is 208 -/
theorem unpainted_cubes_in_6x6x6 :
  let c : PaintedCube := {
    size := 6,
    total_cubes := 216,
    painted_squares_per_face := 4,
    num_faces := 6
  }
  num_unpainted_cubes c = 208 := by sorry

end unpainted_cubes_in_6x6x6_l1381_138141


namespace negation_of_forall_positive_negation_of_greater_than_zero_l1381_138103

theorem negation_of_forall_positive (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem negation_of_greater_than_zero :
  (¬ ∀ x : ℝ, x^2 + x + 2 > 0) ↔ (∃ x : ℝ, x^2 + x + 2 ≤ 0) :=
by sorry

end negation_of_forall_positive_negation_of_greater_than_zero_l1381_138103


namespace allocation_schemes_count_l1381_138166

/-- The number of ways to distribute n volunteers into k groups with size constraints -/
def distribute (n : ℕ) (k : ℕ) : ℕ := sorry

/-- The number of ways to assign k groups to k different areas -/
def assign (k : ℕ) : ℕ := sorry

/-- The total number of allocation schemes -/
def total_schemes (n : ℕ) (k : ℕ) : ℕ :=
  distribute n k * assign k

theorem allocation_schemes_count :
  total_schemes 6 4 = 1080 := by sorry

end allocation_schemes_count_l1381_138166


namespace two_arrows_balance_l1381_138133

/-- A polygon with arrows on its sides -/
structure ArrowPolygon where
  n : ℕ  -- number of sides/vertices
  incoming : Fin n → Fin 2  -- number of incoming arrows for each vertex (0, 1, or 2)
  outgoing : Fin n → Fin 2  -- number of outgoing arrows for each vertex (0, 1, or 2)

/-- The sum of incoming arrows equals the number of sides -/
axiom total_arrows_incoming (p : ArrowPolygon) : 
  (Finset.univ.sum p.incoming) = p.n

/-- The sum of outgoing arrows equals the number of sides -/
axiom total_arrows_outgoing (p : ArrowPolygon) : 
  (Finset.univ.sum p.outgoing) = p.n

/-- Theorem: The number of vertices with two incoming arrows equals the number of vertices with two outgoing arrows -/
theorem two_arrows_balance (p : ArrowPolygon) :
  (Finset.univ.filter (fun i => p.incoming i = 2)).card = 
  (Finset.univ.filter (fun i => p.outgoing i = 2)).card := by
  sorry

end two_arrows_balance_l1381_138133


namespace only_prop2_and_prop4_true_l1381_138151

-- Define the types for lines and planes
def Line : Type := sorry
def Plane : Type := sorry

-- Define the relations between lines and planes
def parallel (a b : Plane) : Prop := sorry
def perpendicular (a b : Plane) : Prop := sorry
def contained_in (l : Line) (p : Plane) : Prop := sorry
def line_parallel (a b : Line) : Prop := sorry
def line_perpendicular (a b : Line) : Prop := sorry
def line_parallel_plane (l : Line) (p : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the propositions
def proposition1 (m n : Line) (α β : Plane) : Prop :=
  parallel α β ∧ contained_in m β ∧ contained_in n α → line_parallel m n

def proposition2 (m n : Line) (α β : Plane) : Prop :=
  parallel α β ∧ line_perpendicular_plane m β ∧ line_parallel_plane n α → line_perpendicular m n

def proposition3 (m n : Line) (α β : Plane) : Prop :=
  perpendicular α β ∧ line_perpendicular_plane m α ∧ line_parallel_plane n β → line_parallel m n

def proposition4 (m n : Line) (α β : Plane) : Prop :=
  perpendicular α β ∧ line_perpendicular_plane m α ∧ line_perpendicular_plane n β → line_perpendicular m n

-- Theorem stating that only propositions 2 and 4 are true
theorem only_prop2_and_prop4_true (m n : Line) (α β : Plane) 
  (h_diff_lines : m ≠ n) (h_diff_planes : α ≠ β) : 
  (¬ proposition1 m n α β) ∧ 
  proposition2 m n α β ∧ 
  (¬ proposition3 m n α β) ∧ 
  proposition4 m n α β := by
  sorry

end only_prop2_and_prop4_true_l1381_138151


namespace sum_of_repeating_decimals_l1381_138111

/-- The sum of three specific repeating decimals is 2 -/
theorem sum_of_repeating_decimals : ∃ (x y z : ℚ),
  (∀ n : ℕ, (10 * x - x) * 10^n = 3 * 10^n) ∧
  (∀ n : ℕ, (10 * y - y) * 10^n = 6 * 10^n) ∧
  (∀ n : ℕ, (10 * z - z) * 10^n = 9 * 10^n) ∧
  x + y + z = 2 := by
  sorry

end sum_of_repeating_decimals_l1381_138111


namespace abc_inequalities_l1381_138121

theorem abc_inequalities (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a > b) (h5 : b > c) (h6 : a + b + c = 0) :
  (c / a + a / c ≤ -2) ∧ (-2 < c / a ∧ c / a < -1/2) := by
  sorry

end abc_inequalities_l1381_138121


namespace derivative_value_at_two_l1381_138163

theorem derivative_value_at_two (f : ℝ → ℝ) (f' : ℝ → ℝ) :
  (∀ x, HasDerivAt f (f' x) x) →
  (∀ x, f x = x^2 + 3*x*(f' 2)) →
  f' 2 = -2 := by
sorry

end derivative_value_at_two_l1381_138163


namespace seashells_to_find_l1381_138130

def current_seashells : ℕ := 307
def target_seashells : ℕ := 500

theorem seashells_to_find : target_seashells - current_seashells = 193 := by
  sorry

end seashells_to_find_l1381_138130


namespace promotion_savings_difference_l1381_138182

/-- Calculates the total cost of two pairs of shoes under a given promotion --/
def promotionCost (regularPrice : ℝ) (discountPercent : ℝ) : ℝ :=
  regularPrice + (regularPrice * (1 - discountPercent))

/-- Represents the difference in savings between two promotions --/
def savingsDifference (regularPrice : ℝ) (discountA : ℝ) (discountB : ℝ) : ℝ :=
  promotionCost regularPrice discountB - promotionCost regularPrice discountA

theorem promotion_savings_difference :
  savingsDifference 50 0.3 0.2 = 5 := by sorry

end promotion_savings_difference_l1381_138182


namespace factory_production_l1381_138197

/-- Represents the number of toys produced in a week at a factory -/
def toysPerWeek (daysWorked : ℕ) (toysPerDay : ℕ) : ℕ :=
  daysWorked * toysPerDay

/-- Theorem stating that the factory produces 4560 toys per week -/
theorem factory_production :
  toysPerWeek 4 1140 = 4560 := by
  sorry

end factory_production_l1381_138197


namespace units_digit_of_33_power_l1381_138118

theorem units_digit_of_33_power : ∃ n : ℕ, 33^(33*(7^7)) ≡ 7 [ZMOD 10] :=
by
  sorry

end units_digit_of_33_power_l1381_138118


namespace third_subtraction_difference_1230_411_l1381_138173

/-- The difference obtained from the third subtraction when using the method of successive subtraction to find the GCD of 1230 and 411 -/
def third_subtraction_difference (a b : ℕ) : ℕ :=
  let d₁ := a - b
  let d₂ := d₁ - b
  d₂ - b

theorem third_subtraction_difference_1230_411 :
  third_subtraction_difference 1230 411 = 3 := by
  sorry

end third_subtraction_difference_1230_411_l1381_138173


namespace inequality_and_equality_condition_l1381_138102

theorem inequality_and_equality_condition (a b c : ℝ) 
  (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) 
  (h_condition : a * b + b * c + c * a + 2 * a * b * c = 1) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≥ 2 ∧ 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c = 2 ↔ 
    a = (-3 + Real.sqrt 17) / 4 ∧ 
    b = (-3 + Real.sqrt 17) / 4 ∧ 
    c = (-3 + Real.sqrt 17) / 4) :=
by sorry

end inequality_and_equality_condition_l1381_138102


namespace shooting_stars_count_difference_l1381_138165

theorem shooting_stars_count_difference (bridget_count reginald_count sam_count : ℕ) : 
  bridget_count = 14 →
  reginald_count = bridget_count - 2 →
  sam_count > reginald_count →
  sam_count = (bridget_count + reginald_count + sam_count) / 3 + 2 →
  sam_count - reginald_count = 4 := by
  sorry

end shooting_stars_count_difference_l1381_138165


namespace simplify_expression_l1381_138116

theorem simplify_expression (x : ℝ) : 3*x + 6*x + 9*x + 12*x + 15*x + 18 = 45*x + 18 := by
  sorry

end simplify_expression_l1381_138116


namespace remainder_problem_l1381_138106

theorem remainder_problem (n : ℕ) (h1 : n = 349) (h2 : n % 17 = 9) : n % 13 = 11 := by
  sorry

end remainder_problem_l1381_138106


namespace tan_675_degrees_l1381_138124

theorem tan_675_degrees (n : ℤ) : 
  -180 < n ∧ n < 180 ∧ Real.tan (n * π / 180) = Real.tan (675 * π / 180) →
  n = 135 ∨ n = -45 := by
sorry

end tan_675_degrees_l1381_138124


namespace fold_sequence_counts_l1381_138154

/-- Represents the possible shapes after folding -/
inductive Shape
  | Square
  | IsoscelesTriangle
  | Rectangle (k : ℕ)

/-- Represents a sequence of folds -/
def FoldSequence := List Shape

/-- Counts the number of possible folding sequences -/
def countFoldSequences (n : ℕ) : ℕ :=
  sorry

theorem fold_sequence_counts :
  (countFoldSequences 3 = 5) ∧
  (countFoldSequences 6 = 24) ∧
  (countFoldSequences 9 = 149) := by
  sorry

end fold_sequence_counts_l1381_138154


namespace division_simplification_l1381_138109

theorem division_simplification (a : ℝ) (h : a ≠ 0) : 6 * a^3 / (2 * a^2) = 3 * a := by
  sorry

end division_simplification_l1381_138109


namespace equation_and_inequality_solution_l1381_138139

theorem equation_and_inequality_solution :
  (∃ x : ℝ, (x - 3) * (x - 2) + 18 = (x + 9) * (x + 1) ∧ x = 1) ∧
  (∀ x : ℝ, (3 * x + 4) * (3 * x - 4) < 9 * (x - 2) * (x + 3) ↔ x > 38 / 9) :=
by sorry

end equation_and_inequality_solution_l1381_138139


namespace robin_gum_pieces_l1381_138137

/-- Calculates the total number of gum pieces Robin has. -/
def total_gum_pieces (packages : ℕ) (pieces_per_package : ℕ) (extra_pieces : ℕ) : ℕ :=
  packages * pieces_per_package + extra_pieces

/-- Proves that Robin has 997 pieces of gum in total. -/
theorem robin_gum_pieces :
  total_gum_pieces 43 23 8 = 997 := by
  sorry

end robin_gum_pieces_l1381_138137


namespace range_of_a_l1381_138153

-- Define proposition p
def p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

-- Define proposition q
def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem statement
theorem range_of_a (a : ℝ) :
  p a ∧ q a → a = 1 ∨ a ≤ -2 := by
  sorry

end range_of_a_l1381_138153


namespace lcm_12_18_l1381_138134

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end lcm_12_18_l1381_138134


namespace sandy_change_theorem_l1381_138110

def cappuccino_price : ℝ := 2
def iced_tea_price : ℝ := 3
def cafe_latte_price : ℝ := 1.5
def espresso_price : ℝ := 1

def sandy_order_cappuccinos : ℕ := 3
def sandy_order_iced_teas : ℕ := 2
def sandy_order_cafe_lattes : ℕ := 2
def sandy_order_espressos : ℕ := 2

def paid_amount : ℝ := 20

theorem sandy_change_theorem :
  paid_amount - (cappuccino_price * sandy_order_cappuccinos +
                 iced_tea_price * sandy_order_iced_teas +
                 cafe_latte_price * sandy_order_cafe_lattes +
                 espresso_price * sandy_order_espressos) = 3 := by
  sorry

end sandy_change_theorem_l1381_138110


namespace complement_of_union_l1381_138198

-- Define the universal set U
def U : Set Int := {-2, -1, 0, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

-- State the theorem
theorem complement_of_union :
  (U \ (A ∪ B)) = {-2, 0} := by sorry

end complement_of_union_l1381_138198


namespace remainder_2022_power_2023_power_2024_mod_19_l1381_138132

theorem remainder_2022_power_2023_power_2024_mod_19 :
  (2022 ^ (2023 ^ 2024)) % 19 = 8 := by
  sorry

end remainder_2022_power_2023_power_2024_mod_19_l1381_138132


namespace sports_equipment_purchase_l1381_138122

/-- Represents the cost function for Scheme A -/
def cost_scheme_a (x : ℕ) : ℝ := 25 * x + 550

/-- Represents the cost function for Scheme B -/
def cost_scheme_b (x : ℕ) : ℝ := 22.5 * x + 720

theorem sports_equipment_purchase :
  /- Cost functions are correct -/
  (∀ x : ℕ, x ≥ 10 → cost_scheme_a x = 25 * x + 550 ∧ cost_scheme_b x = 22.5 * x + 720) ∧
  /- Scheme A is more cost-effective for 15 boxes -/
  cost_scheme_a 15 < cost_scheme_b 15 ∧
  /- Scheme A allows purchasing more balls with 1800 yuan budget -/
  (∃ x_a x_b : ℕ, cost_scheme_a x_a ≤ 1800 ∧ cost_scheme_b x_b ≤ 1800 ∧ x_a > x_b) :=
by sorry

end sports_equipment_purchase_l1381_138122


namespace line_equation_sum_l1381_138189

/-- Given a line passing through points (1, -2) and (4, 7), prove that m + b = -2 where y = mx + b is the equation of the line. -/
theorem line_equation_sum (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b → 
    ((x = 1 ∧ y = -2) ∨ (x = 4 ∧ y = 7))) → 
  m + b = -2 := by
  sorry

end line_equation_sum_l1381_138189


namespace ann_speed_l1381_138101

/-- Given cyclists' speeds, prove Ann's speed -/
theorem ann_speed (tom_speed : ℚ) (jerry_speed : ℚ) (ann_speed : ℚ) : 
  tom_speed = 6 →
  jerry_speed = 3/4 * tom_speed →
  ann_speed = 4/3 * jerry_speed →
  ann_speed = 6 := by
sorry

end ann_speed_l1381_138101


namespace complete_square_transform_l1381_138168

theorem complete_square_transform (a : ℝ) : a^2 + 4*a - 5 = (a + 2)^2 - 9 := by
  sorry

end complete_square_transform_l1381_138168


namespace complex_number_condition_l1381_138196

/-- A complex number z satisfying the given conditions -/
def Z : ℂ := sorry

/-- The real part of Z -/
def m : ℝ := Z.re

/-- The imaginary part of Z -/
def n : ℝ := Z.im

/-- The condition that z+2i is a real number -/
axiom h1 : (Z + 2*Complex.I).im = 0

/-- The condition that z/(2-i) is a real number -/
axiom h2 : ((Z / (2 - Complex.I))).im = 0

/-- Definition of the function representing (z+ai)^2 -/
def f (a : ℝ) : ℂ := (Z + a*Complex.I)^2

/-- The theorem to be proved -/
theorem complex_number_condition (a : ℝ) :
  (f a).re < 0 ∧ (f a).im > 0 → a > 6 := by sorry

end complex_number_condition_l1381_138196


namespace abs_diff_eq_sum_abs_iff_product_nonpositive_l1381_138193

theorem abs_diff_eq_sum_abs_iff_product_nonpositive (a b : ℝ) :
  |a - b| = |a| + |b| ↔ a * b ≤ 0 := by sorry

end abs_diff_eq_sum_abs_iff_product_nonpositive_l1381_138193


namespace circle_area_from_circumference_l1381_138159

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), (2 * π * r = 36 * π) → (π * r^2 = 324 * π) := by
  sorry

end circle_area_from_circumference_l1381_138159


namespace parallel_lines_a_value_l1381_138188

-- Define the lines l₁ and l₂
def l₁ (x y a : ℝ) : Prop := 3 * x + 2 * a * y - 5 = 0
def l₂ (x y a : ℝ) : Prop := (3 * a - 1) * x - a * y - 2 = 0

-- Define the parallel condition
def parallel (a : ℝ) : Prop := ∀ x y, l₁ x y a ↔ l₂ x y a

-- Theorem statement
theorem parallel_lines_a_value (a : ℝ) :
  parallel a → (a = 0 ∨ a = -1/6) :=
by sorry

end parallel_lines_a_value_l1381_138188


namespace max_xy_on_line_segment_l1381_138199

/-- Given points A(2,0) and B(0,1), prove that the maximum value of xy for any point P(x,y) on the line segment AB is 1/2 -/
theorem max_xy_on_line_segment : 
  ∀ x y : ℝ, 
  0 ≤ x ∧ x ≤ 2 → -- Condition for x being on the line segment
  x / 2 + y = 1 → -- Equation of the line AB
  x * y ≤ (1 : ℝ) / 2 ∧ 
  ∃ x₀ y₀ : ℝ, 0 ≤ x₀ ∧ x₀ ≤ 2 ∧ x₀ / 2 + y₀ = 1 ∧ x₀ * y₀ = (1 : ℝ) / 2 :=
by sorry

end max_xy_on_line_segment_l1381_138199
