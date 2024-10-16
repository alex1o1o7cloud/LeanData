import Mathlib

namespace NUMINAMATH_CALUDE_function_expression_on_interval_l3071_307107

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem function_expression_on_interval
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 2)
  (h_even : is_even f)
  (h_interval : ∀ x ∈ Set.Icc 2 3, f x = x) :
  ∀ x ∈ Set.Icc (-2) 0, f x = 3 - |x + 1| :=
sorry

end NUMINAMATH_CALUDE_function_expression_on_interval_l3071_307107


namespace NUMINAMATH_CALUDE_elevator_problem_l3071_307163

def elevator_ways (n : ℕ) (k : ℕ) (max_per_floor : ℕ) : ℕ :=
  sorry

theorem elevator_problem : elevator_ways 3 5 2 = 120 := by
  sorry

end NUMINAMATH_CALUDE_elevator_problem_l3071_307163


namespace NUMINAMATH_CALUDE_circle_equation_example_l3071_307199

/-- The standard equation of a circle with center (h,k) and radius r is (x-h)^2 + (y-k)^2 = r^2 -/
def standard_circle_equation (h k r : ℝ) (x y : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- Given a circle with center (3,1) and radius 5, its standard equation is (x-3)^2+(y-1)^2=25 -/
theorem circle_equation_example :
  ∀ x y : ℝ, standard_circle_equation 3 1 5 x y ↔ (x - 3)^2 + (y - 1)^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_example_l3071_307199


namespace NUMINAMATH_CALUDE_total_bulbs_needed_l3071_307135

theorem total_bulbs_needed (medium_lights : ℕ) (small_bulbs : ℕ) (medium_bulbs : ℕ) (large_bulbs : ℕ) :
  medium_lights = 12 →
  small_bulbs = 1 →
  medium_bulbs = 2 →
  large_bulbs = 3 →
  (medium_lights * small_bulbs + 10) * small_bulbs +
  medium_lights * medium_bulbs +
  (2 * medium_lights) * large_bulbs = 118 := by
  sorry

end NUMINAMATH_CALUDE_total_bulbs_needed_l3071_307135


namespace NUMINAMATH_CALUDE_largest_valid_number_l3071_307166

def is_valid (n : ℕ) : Prop :=
  n < 10000 ∧
  (∃ a : ℕ, 4^a ≤ n ∧ n < 4^(a+1) ∧ 4^a ≤ 3*n ∧ 3*n < 4^(a+1)) ∧
  (∃ b : ℕ, 8^b ≤ n ∧ n < 8^(b+1) ∧ 8^b ≤ 7*n ∧ 7*n < 8^(b+1)) ∧
  (∃ c : ℕ, 16^c ≤ n ∧ n < 16^(c+1) ∧ 16^c ≤ 15*n ∧ 15*n < 16^(c+1))

theorem largest_valid_number : 
  is_valid 4369 ∧ ∀ m : ℕ, m > 4369 → ¬(is_valid m) :=
by sorry

end NUMINAMATH_CALUDE_largest_valid_number_l3071_307166


namespace NUMINAMATH_CALUDE_f_composition_result_l3071_307170

noncomputable def f (z : ℂ) : ℂ :=
  if z.im = 0 then -z^3 else z^3

theorem f_composition_result :
  f (f (f (f (-1 + I)))) = -1.79841759e14 - 2.75930025e10 * I :=
by sorry

end NUMINAMATH_CALUDE_f_composition_result_l3071_307170


namespace NUMINAMATH_CALUDE_base_b_problem_l3071_307187

theorem base_b_problem (b : ℕ) : 
  b > 1 ∧ 
  (2 * b + 5 < b^2) ∧ 
  (5 * b + 2 < b^2) ∧ 
  (5 * b + 2 = 2 * (2 * b + 5)) → 
  b = 8 := by sorry

end NUMINAMATH_CALUDE_base_b_problem_l3071_307187


namespace NUMINAMATH_CALUDE_correct_calculation_l3071_307160

theorem correct_calculation : -7 + 3 = -4 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3071_307160


namespace NUMINAMATH_CALUDE_total_fruits_l3071_307176

def papaya_trees : ℕ := 2
def mango_trees : ℕ := 3
def papayas_per_tree : ℕ := 10
def mangos_per_tree : ℕ := 20

theorem total_fruits : 
  papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree = 80 := by
  sorry

end NUMINAMATH_CALUDE_total_fruits_l3071_307176


namespace NUMINAMATH_CALUDE_orange_count_l3071_307182

/-- Represents the number of oranges in a basket -/
structure Basket where
  good : ℕ
  bad : ℕ

/-- Defines the ratio between good and bad oranges -/
def hasRatio (b : Basket) (g : ℕ) (d : ℕ) : Prop :=
  g * b.bad = d * b.good

theorem orange_count (b : Basket) (h1 : hasRatio b 3 1) (h2 : b.bad = 8) : b.good = 24 := by
  sorry

end NUMINAMATH_CALUDE_orange_count_l3071_307182


namespace NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l3071_307129

theorem profit_percent_for_cost_selling_ratio (cost_price selling_price : ℝ) :
  cost_price > 0 →
  selling_price > cost_price →
  cost_price / selling_price = 2 / 3 →
  (selling_price - cost_price) / cost_price * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_for_cost_selling_ratio_l3071_307129


namespace NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l3071_307171

/-- The shortest distance from a point on the curve y = ln x to the line 2x - y + 3 = 0 -/
theorem shortest_distance_ln_to_line : ∃ (d : ℝ), d = (4 + Real.log 2) / Real.sqrt 5 ∧
  ∀ (x y : ℝ), y = Real.log x →
    d ≤ (|2 * x - y + 3|) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_ln_to_line_l3071_307171


namespace NUMINAMATH_CALUDE_absolute_value_sum_l3071_307174

theorem absolute_value_sum (x q : ℝ) : 
  |x - 5| = q ∧ x > 5 → x + q = 2*q + 5 := by
sorry

end NUMINAMATH_CALUDE_absolute_value_sum_l3071_307174


namespace NUMINAMATH_CALUDE_equation_solution_l3071_307115

theorem equation_solution : ∃! x : ℝ, 45 - (28 - (37 - (x - 17))) = 56 ∧ x = 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3071_307115


namespace NUMINAMATH_CALUDE_restaurant_group_children_l3071_307110

/-- The number of children in a restaurant group -/
def num_children (num_adults : ℕ) (meal_cost : ℕ) (total_bill : ℕ) : ℕ :=
  (total_bill - num_adults * meal_cost) / meal_cost

theorem restaurant_group_children :
  num_children 2 3 21 = 5 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_group_children_l3071_307110


namespace NUMINAMATH_CALUDE_carlas_marbles_l3071_307144

theorem carlas_marbles (x : ℕ) : 
  x + 134 - 68 + 56 = 244 → x = 122 := by
  sorry

end NUMINAMATH_CALUDE_carlas_marbles_l3071_307144


namespace NUMINAMATH_CALUDE_salary_difference_l3071_307105

theorem salary_difference (a b : ℝ) (h : b = 1.25 * a) :
  (b - a) / b * 100 = 20 := by sorry

end NUMINAMATH_CALUDE_salary_difference_l3071_307105


namespace NUMINAMATH_CALUDE_simplify_expression_1_l3071_307161

theorem simplify_expression_1 (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (x^2 / (-2*y)) * (6*x*y^2 / x^4) = -3*y/x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l3071_307161


namespace NUMINAMATH_CALUDE_hyperbola_center_correct_l3071_307164

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (4 * x - 8)^2 / 9^2 - (5 * y + 5)^2 / 7^2 = 1

/-- The center of the hyperbola -/
def hyperbola_center : ℝ × ℝ := (2, -1)

/-- Theorem stating that hyperbola_center is the center of the hyperbola defined by hyperbola_equation -/
theorem hyperbola_center_correct :
  ∀ (x y : ℝ), hyperbola_equation x y ↔ 
    hyperbola_equation (x - hyperbola_center.1) (y - hyperbola_center.2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_center_correct_l3071_307164


namespace NUMINAMATH_CALUDE_female_advanced_under_40_l3071_307154

theorem female_advanced_under_40 (total_employees : ℕ) (female_employees : ℕ) (male_employees : ℕ)
  (advanced_degrees : ℕ) (college_degrees : ℕ) (high_school_diplomas : ℕ)
  (male_advanced : ℕ) (male_college : ℕ) (male_high_school : ℕ)
  (female_under_40_ratio : ℚ) :
  total_employees = 280 →
  female_employees = 160 →
  male_employees = 120 →
  advanced_degrees = 120 →
  college_degrees = 100 →
  high_school_diplomas = 60 →
  male_advanced = 50 →
  male_college = 35 →
  male_high_school = 35 →
  female_under_40_ratio = 3/4 →
  ⌊(advanced_degrees - male_advanced : ℚ) * female_under_40_ratio⌋ = 52 :=
by sorry

end NUMINAMATH_CALUDE_female_advanced_under_40_l3071_307154


namespace NUMINAMATH_CALUDE_age_difference_l3071_307125

/-- The problem of finding the age difference between A and B -/
theorem age_difference (a b : ℕ) : b = 36 → a + 10 = 2 * (b - 10) → a - b = 6 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l3071_307125


namespace NUMINAMATH_CALUDE_park_visitors_difference_l3071_307124

theorem park_visitors_difference (total : ℕ) (bikers : ℕ) (hikers : ℕ) :
  total = 676 →
  bikers = 249 →
  total = bikers + hikers →
  hikers > bikers →
  hikers - bikers = 178 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_difference_l3071_307124


namespace NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l3071_307177

theorem lcm_of_numbers_with_given_hcf_and_product (a b : ℕ+) : 
  Nat.gcd a b = 16 → 
  a * b = 2560 → 
  Nat.lcm a b = 160 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_numbers_with_given_hcf_and_product_l3071_307177


namespace NUMINAMATH_CALUDE_find_x_l3071_307138

theorem find_x (a b : ℝ) (x y r : ℝ) (h1 : b ≠ 0) (h2 : r = (3*a)^(3*b)) (h3 : r = a^b * (x + y)^b) (h4 : y = 3*a) :
  x = 27*a^2 - 3*a := by
sorry

end NUMINAMATH_CALUDE_find_x_l3071_307138


namespace NUMINAMATH_CALUDE_function_f_property_l3071_307126

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem function_f_property : 
  (∀ x, f x + 2 * f (27 - x) = x) → f 11 = 7 := by sorry

end NUMINAMATH_CALUDE_function_f_property_l3071_307126


namespace NUMINAMATH_CALUDE_choir_members_proof_l3071_307179

theorem choir_members_proof :
  ∃! n : ℕ, 200 ≤ n ∧ n ≤ 300 ∧ n % 7 = 3 ∧ n % 11 = 6 ∧ n = 220 := by
  sorry

end NUMINAMATH_CALUDE_choir_members_proof_l3071_307179


namespace NUMINAMATH_CALUDE_range_of_a_l3071_307114

/-- Curve C1 in polar coordinates -/
def C1 (ρ θ a : ℝ) : Prop := ρ * (Real.sqrt 2 * Real.cos θ - Real.sin θ) = a

/-- Curve C2 in parametric form -/
def C2 (x y θ : ℝ) : Prop := x = Real.sin θ + Real.cos θ ∧ y = 1 + Real.sin (2 * θ)

/-- C1 in rectangular coordinates -/
def C1_rect (x y a : ℝ) : Prop := Real.sqrt 2 * x - y - a = 0

/-- C2 in rectangular coordinates -/
def C2_rect (x y : ℝ) : Prop := y = x^2 ∧ x ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2)

/-- The main theorem -/
theorem range_of_a (a : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ 
    C1_rect x₁ y₁ a ∧ C2_rect x₁ y₁ ∧
    C1_rect x₂ y₂ a ∧ C2_rect x₂ y₂) ↔
  a ∈ Set.Icc (-1/2) 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l3071_307114


namespace NUMINAMATH_CALUDE_total_undeveloped_area_l3071_307156

def undeveloped_sections : ℕ := 3
def section_area : ℕ := 2435

theorem total_undeveloped_area : undeveloped_sections * section_area = 7305 := by
  sorry

end NUMINAMATH_CALUDE_total_undeveloped_area_l3071_307156


namespace NUMINAMATH_CALUDE_sum_of_solutions_l3071_307137

theorem sum_of_solutions (N : ℝ) : (N * (N + 4) = 8) → (∃ x y : ℝ, x + y = -4 ∧ x * (x + 4) = 8 ∧ y * (y + 4) = 8) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l3071_307137


namespace NUMINAMATH_CALUDE_value_of_y_l3071_307139

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 := by
  sorry

end NUMINAMATH_CALUDE_value_of_y_l3071_307139


namespace NUMINAMATH_CALUDE_no_point_satisfies_both_systems_l3071_307131

/-- A point in the 2D plane satisfies System I if it meets all these conditions -/
def satisfies_system_I (x y : ℝ) : Prop :=
  y < 3 ∧ x - y < 3 ∧ x + y < 4

/-- A point in the 2D plane satisfies System II if it meets all these conditions -/
def satisfies_system_II (x y : ℝ) : Prop :=
  (y - 3) * (x - y - 3) ≥ 0 ∧
  (y - 3) * (x + y - 4) ≤ 0 ∧
  (x - y - 3) * (x + y - 4) ≤ 0

/-- There is no point that satisfies both System I and System II -/
theorem no_point_satisfies_both_systems :
  ¬ ∃ (x y : ℝ), satisfies_system_I x y ∧ satisfies_system_II x y :=
by sorry

end NUMINAMATH_CALUDE_no_point_satisfies_both_systems_l3071_307131


namespace NUMINAMATH_CALUDE_josh_remaining_money_l3071_307143

theorem josh_remaining_money (initial_amount spent_on_drink additional_spent : ℚ) 
  (h1 : initial_amount = 9)
  (h2 : spent_on_drink = 1.75)
  (h3 : additional_spent = 1.25) :
  initial_amount - spent_on_drink - additional_spent = 6 := by
  sorry

end NUMINAMATH_CALUDE_josh_remaining_money_l3071_307143


namespace NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3071_307121

def is_divisible_by_all (n : ℕ) (divisors : List ℕ) : Prop :=
  ∀ d ∈ divisors, (n + 3) % d = 0

theorem smallest_number_divisible_by_all : 
  ∀ n : ℕ, n < 6303 → ¬(is_divisible_by_all n [70, 100, 84]) ∧ 
  is_divisible_by_all 6303 [70, 100, 84] :=
sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_by_all_l3071_307121


namespace NUMINAMATH_CALUDE_negation_of_P_is_true_l3071_307117

theorem negation_of_P_is_true : ∀ (x : ℝ), (x - 1)^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_P_is_true_l3071_307117


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3071_307152

/-- Given a geometric sequence {a_n} with specific properties, prove a_7 = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 * a 4 * a 5 = a 3 * a 6 →                   -- given condition
  a 9 * a 10 = -8 →                               -- given condition
  a 7 = -2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3071_307152


namespace NUMINAMATH_CALUDE_factorial_not_ending_19760_l3071_307104

theorem factorial_not_ending_19760 (n : ℕ+) : ¬ ∃ k : ℕ, (n!:ℕ) % (10^(k+5)) = 19760 * 10^k :=
sorry

end NUMINAMATH_CALUDE_factorial_not_ending_19760_l3071_307104


namespace NUMINAMATH_CALUDE_session_comparison_l3071_307155

theorem session_comparison (a b : ℝ) : 
  a > 0 → -- Assuming a is positive (number of people)
  b = 0.9 * (1.1 * a) → 
  a > b := by
sorry

end NUMINAMATH_CALUDE_session_comparison_l3071_307155


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l3071_307192

/-- The equation of the trajectory of point M -/
def trajectory_equation (x y : ℝ) : Prop :=
  10 * Real.sqrt (x^2 + y^2) = |3*x + 4*y - 12|

/-- The trajectory of point M is an ellipse -/
theorem trajectory_is_ellipse :
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), trajectory_equation x y ↔ 
    ((x - h)^2 / a^2 + (y - k)^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l3071_307192


namespace NUMINAMATH_CALUDE_positive_A_value_l3071_307127

-- Define the # relation
def hash (A B : ℝ) : ℝ := A^2 + B^2

-- Theorem statement
theorem positive_A_value (A : ℝ) (h : hash A 6 = 200) : A = 2 * Real.sqrt 41 :=
sorry

end NUMINAMATH_CALUDE_positive_A_value_l3071_307127


namespace NUMINAMATH_CALUDE_penny_stack_more_valuable_l3071_307168

/-- Represents a stack of coins -/
structure CoinStack :=
  (onePence : ℕ)
  (twoPence : ℕ)
  (fivePence : ℕ)

/-- Calculates the height of a coin stack in millimeters -/
def stackHeight (stack : CoinStack) : ℚ :=
  1.6 * stack.onePence + 2.05 * stack.twoPence + 1.75 * stack.fivePence

/-- Calculates the value of a coin stack in pence -/
def stackValue (stack : CoinStack) : ℕ :=
  stack.onePence + 2 * stack.twoPence + 5 * stack.fivePence

/-- Checks if a stack is valid according to the problem constraints -/
def isValidStack (stack : CoinStack) : Prop :=
  stackHeight stack = stackValue stack ∧ 
  (stack.onePence > 0 ∨ stack.twoPence > 0 ∨ stack.fivePence > 0)

/-- Joe's optimal stack using only 1p and 5p coins -/
def joesStack : CoinStack :=
  ⟨65, 0, 12⟩

/-- Penny's optimal stack using only 2p and 5p coins -/
def pennysStack : CoinStack :=
  ⟨0, 65, 1⟩

theorem penny_stack_more_valuable :
  isValidStack joesStack ∧
  isValidStack pennysStack ∧
  stackValue pennysStack > stackValue joesStack :=
sorry

end NUMINAMATH_CALUDE_penny_stack_more_valuable_l3071_307168


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_equals_3850_l3071_307173

def prime_factorization := (2, 12) :: (3, 18) :: (5, 20) :: (7, 8) :: []

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

def count_perfect_square_factors (factorization : List (ℕ × ℕ)) : ℕ :=
  factorization.foldl (fun acc (p, e) => acc * ((e / 2) + 1)) 1

theorem count_perfect_square_factors_equals_3850 :
  count_perfect_square_factors prime_factorization = 3850 := by
  sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_equals_3850_l3071_307173


namespace NUMINAMATH_CALUDE_total_cost_of_books_l3071_307158

-- Define the number of books for each category
def animal_books : ℕ := 10
def space_books : ℕ := 1
def train_books : ℕ := 3

-- Define the cost per book
def cost_per_book : ℕ := 16

-- Define the total number of books
def total_books : ℕ := animal_books + space_books + train_books

-- Theorem to prove
theorem total_cost_of_books : total_books * cost_per_book = 224 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_of_books_l3071_307158


namespace NUMINAMATH_CALUDE_solution_existence_l3071_307134

theorem solution_existence (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) 
  (h : x + y + 2 * Real.sqrt (x * y) = 2017) :
  (x = 0 ∧ y = 2017) ∨ (x = 2017 ∧ y = 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_existence_l3071_307134


namespace NUMINAMATH_CALUDE_maria_furniture_assembly_l3071_307167

/-- Given the number of chairs, tables, and total assembly time, 
    calculate the time spent on each piece of furniture. -/
def time_per_piece (chairs : ℕ) (tables : ℕ) (total_time : ℕ) : ℚ :=
  (total_time : ℚ) / (chairs + tables : ℚ)

/-- Theorem stating that for 2 chairs, 2 tables, and 32 minutes total time,
    the time per piece is 8 minutes. -/
theorem maria_furniture_assembly : 
  time_per_piece 2 2 32 = 8 := by
  sorry

end NUMINAMATH_CALUDE_maria_furniture_assembly_l3071_307167


namespace NUMINAMATH_CALUDE_M_intersect_N_empty_l3071_307190

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 - x > 0}

-- Define set N
def N : Set ℝ := {x : ℝ | (x - 1) / x < 0}

-- Theorem statement
theorem M_intersect_N_empty : M ∩ N = ∅ := by
  sorry

end NUMINAMATH_CALUDE_M_intersect_N_empty_l3071_307190


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3071_307181

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : geometric_sequence a q)
  (h_pos : q > 0)
  (h_equality : a 3 * a 9 = (a 5)^2) :
  q = 1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l3071_307181


namespace NUMINAMATH_CALUDE_randys_pig_feed_per_week_l3071_307141

/-- Calculates the amount of pig feed needed per week given the daily feed per pig, number of pigs, and days in a week. -/
def pig_feed_per_week (feed_per_pig_per_day : ℕ) (num_pigs : ℕ) (days_in_week : ℕ) : ℕ :=
  feed_per_pig_per_day * num_pigs * days_in_week

/-- Proves that Randy's pigs will be fed 140 pounds of pig feed per week. -/
theorem randys_pig_feed_per_week :
  let feed_per_pig_per_day : ℕ := 10
  let num_pigs : ℕ := 2
  let days_in_week : ℕ := 7
  pig_feed_per_week feed_per_pig_per_day num_pigs days_in_week = 140 := by
  sorry

end NUMINAMATH_CALUDE_randys_pig_feed_per_week_l3071_307141


namespace NUMINAMATH_CALUDE_robin_candy_packages_l3071_307186

/-- Given the total number of candy pieces and the number of pieces per package,
    calculate the number of candy packages. -/
def candy_packages (total_pieces : ℕ) (pieces_per_package : ℕ) : ℕ :=
  total_pieces / pieces_per_package

/-- Theorem stating that Robin has 45 packages of candy. -/
theorem robin_candy_packages :
  candy_packages 405 9 = 45 := by
  sorry

#eval candy_packages 405 9

end NUMINAMATH_CALUDE_robin_candy_packages_l3071_307186


namespace NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3071_307130

theorem choir_arrangement (n : ℕ) : 
  (n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0) → n ≥ 990 :=
by sorry

theorem min_choir_members : 
  ∃ (n : ℕ), n % 9 = 0 ∧ n % 10 = 0 ∧ n % 11 = 0 ∧ n = 990 :=
by sorry

end NUMINAMATH_CALUDE_choir_arrangement_min_choir_members_l3071_307130


namespace NUMINAMATH_CALUDE_simplify_fraction_l3071_307145

theorem simplify_fraction (x : ℝ) (hx : x = Real.sqrt 3) :
  (1 / (1 + x)) * (1 / (1 - x)) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l3071_307145


namespace NUMINAMATH_CALUDE_tan_product_eighths_pi_l3071_307195

theorem tan_product_eighths_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_product_eighths_pi_l3071_307195


namespace NUMINAMATH_CALUDE_five_number_sum_problem_l3071_307197

theorem five_number_sum_problem :
  ∃! (a b c d e : ℕ),
    (∀ (s : Finset ℕ), s.card = 4 → s ⊆ {a, b, c, d, e} →
      (s.sum id = 44 ∨ s.sum id = 45 ∨ s.sum id = 46 ∨ s.sum id = 47)) ∧
    ({a, b, c, d, e} : Finset ℕ).card = 5 ∧
    a = 13 ∧ b = 12 ∧ c = 11 ∧ d = 11 ∧ e = 10 :=
by sorry

end NUMINAMATH_CALUDE_five_number_sum_problem_l3071_307197


namespace NUMINAMATH_CALUDE_min_value_of_f_l3071_307183

def f (x : ℝ) : ℝ := x^2 + 14*x + 24

theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3071_307183


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3071_307133

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Sum of first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  (Finset.range n).sum (seq.a ∘ Nat.succ)

/-- Main theorem -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  S seq 8 - S seq 3 = 10 → S seq 11 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3071_307133


namespace NUMINAMATH_CALUDE_compare_logarithms_and_sqrt_l3071_307153

theorem compare_logarithms_and_sqrt : 
  let a := 2 * Real.log (21/20)
  let b := Real.log (11/10)
  let c := Real.sqrt 1.2 - 1
  c < a ∧ a < b :=
by sorry

end NUMINAMATH_CALUDE_compare_logarithms_and_sqrt_l3071_307153


namespace NUMINAMATH_CALUDE_autumn_pencils_bought_l3071_307198

/-- Calculates the number of pencils Autumn bought given the initial number,
    misplaced pencils, broken pencils, found pencils, and final number of pencils. -/
def pencils_bought (initial : ℕ) (misplaced : ℕ) (broken : ℕ) (found : ℕ) (final : ℕ) : ℕ :=
  final - (initial - misplaced - broken + found)

/-- Proves that Autumn bought 2 pencils given the specific scenario. -/
theorem autumn_pencils_bought :
  pencils_bought 20 7 3 4 16 = 2 := by
  sorry

#eval pencils_bought 20 7 3 4 16

end NUMINAMATH_CALUDE_autumn_pencils_bought_l3071_307198


namespace NUMINAMATH_CALUDE_fourth_section_size_l3071_307100

/-- The number of students in the fourth section of a chemistry class -/
def fourth_section_students : ℕ :=
  -- We'll define this later in the theorem
  42

/-- Represents the data for a chemistry class section -/
structure Section where
  students : ℕ
  mean_marks : ℚ

/-- Calculates the total marks for a section -/
def total_marks (s : Section) : ℚ :=
  s.students * s.mean_marks

/-- Represents the data for all sections of the chemistry class -/
structure ChemistryClass where
  section1 : Section
  section2 : Section
  section3 : Section
  section4 : Section
  overall_average : ℚ

theorem fourth_section_size (c : ChemistryClass) :
  c.section1.students = 65 →
  c.section2.students = 35 →
  c.section3.students = 45 →
  c.section1.mean_marks = 50 →
  c.section2.mean_marks = 60 →
  c.section3.mean_marks = 55 →
  c.section4.mean_marks = 45 →
  c.overall_average = 51.95 →
  c.section4.students = fourth_section_students :=
by
  sorry

#eval fourth_section_students

end NUMINAMATH_CALUDE_fourth_section_size_l3071_307100


namespace NUMINAMATH_CALUDE_defective_products_m1_l3071_307178

theorem defective_products_m1 (m1_production m2_production m3_production : ℝ)
  (m2_defective_rate m3_defective_rate total_defective_rate : ℝ) :
  m1_production = 0.4 →
  m2_production = 0.3 →
  m3_production = 0.3 →
  m2_defective_rate = 0.01 →
  m3_defective_rate = 0.07 →
  total_defective_rate = 0.036 →
  ∃ (m1_defective_rate : ℝ),
    m1_defective_rate * m1_production +
    m2_defective_rate * m2_production +
    m3_defective_rate * m3_production = total_defective_rate ∧
    m1_defective_rate = 0.03 := by
  sorry

end NUMINAMATH_CALUDE_defective_products_m1_l3071_307178


namespace NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3071_307140

/-- Calculates the interest rate for the first part of an investment given the total amount,
    the amount in the first part, the interest rate for the second part, and the total profit. -/
def calculate_first_interest_rate (total_amount : ℕ) (first_part : ℕ) (second_interest_rate : ℕ) (total_profit : ℕ) : ℚ :=
  let second_part := total_amount - first_part
  let second_part_profit := (second_part * second_interest_rate) / 100
  let first_part_profit := total_profit - second_part_profit
  (first_part_profit * 100) / first_part

theorem first_interest_rate_is_ten_percent :
  calculate_first_interest_rate 80000 70000 20 9000 = 10 := by
  sorry

end NUMINAMATH_CALUDE_first_interest_rate_is_ten_percent_l3071_307140


namespace NUMINAMATH_CALUDE_round_trip_speed_calculation_l3071_307101

/-- Proves that given specific conditions for a round trip, the return speed is 37.5 mph -/
theorem round_trip_speed_calculation (distance : ℝ) (speed_ab : ℝ) (avg_speed : ℝ) :
  distance = 150 →
  speed_ab = 75 →
  avg_speed = 50 →
  (2 * distance) / (distance / speed_ab + distance / ((2 * distance) / (2 * distance / avg_speed - distance / speed_ab))) = avg_speed →
  (2 * distance) / (2 * distance / avg_speed - distance / speed_ab) = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_round_trip_speed_calculation_l3071_307101


namespace NUMINAMATH_CALUDE_card_ratio_proof_l3071_307123

theorem card_ratio_proof :
  let full_deck : ℕ := 52
  let num_partial_decks : ℕ := 3
  let num_full_decks : ℕ := 3
  let discarded_cards : ℕ := 34
  let remaining_cards : ℕ := 200
  let total_cards : ℕ := remaining_cards + discarded_cards
  let partial_deck_cards : ℕ := (total_cards - num_full_decks * full_deck) / num_partial_decks
  ∃ (a b : ℕ), a ≠ 0 ∧ b ≠ 0 ∧ partial_deck_cards * b = full_deck * a ∧ a = 1 ∧ b = 2 :=
by sorry

end NUMINAMATH_CALUDE_card_ratio_proof_l3071_307123


namespace NUMINAMATH_CALUDE_cat_mouse_problem_l3071_307136

theorem cat_mouse_problem (n : ℕ+) (h1 : n * (n + 18) = 999919) : n = 991 := by
  sorry

end NUMINAMATH_CALUDE_cat_mouse_problem_l3071_307136


namespace NUMINAMATH_CALUDE_equation_solution_l3071_307128

theorem equation_solution : 
  ∃ (x₁ x₂ : ℚ), (x₁ = 1/6 ∧ x₂ = -1/4) ∧ 
  (∀ x : ℚ, 4*x*(6*x - 1) = 1 - 6*x ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3071_307128


namespace NUMINAMATH_CALUDE_intersection_dot_product_converse_not_always_true_l3071_307122

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define a line passing through (3,0)
def line_through_3_0 (l : ℝ → ℝ) : Prop := l 3 = 0

-- Define intersection points of a line and the parabola
def intersection_points (l : ℝ → ℝ) (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ l A.1 = A.2 ∧ l B.1 = B.2

-- Define dot product of OA and OB
def dot_product (A B : ℝ × ℝ) : ℝ := A.1 * B.1 + A.2 * B.2

-- Theorem 1
theorem intersection_dot_product (l : ℝ → ℝ) (A B : ℝ × ℝ) :
  line_through_3_0 l → intersection_points l A B → dot_product A B = 3 :=
sorry

-- Theorem 2
theorem converse_not_always_true : 
  ∃ (A B : ℝ × ℝ) (l : ℝ → ℝ), parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  dot_product A B = 3 ∧ ¬(line_through_3_0 l) ∧ l A.1 = A.2 ∧ l B.1 = B.2 :=
sorry

end NUMINAMATH_CALUDE_intersection_dot_product_converse_not_always_true_l3071_307122


namespace NUMINAMATH_CALUDE_two_digit_sum_units_digit_l3071_307162

/-- Represents a digit from 0 to 6 -/
inductive Digit
  | zero | one | two | three | four | five | six

/-- Converts a Digit to its corresponding natural number -/
def digitToNat (d : Digit) : Nat :=
  match d with
  | Digit.zero => 0
  | Digit.one => 1
  | Digit.two => 2
  | Digit.three => 3
  | Digit.four => 4
  | Digit.five => 5
  | Digit.six => 6

/-- Represents a two-digit number using Digits -/
structure TwoDigitNumber where
  tens : Digit
  units : Digit

/-- Converts a TwoDigitNumber to its corresponding natural number -/
def twoDigitNumberToNat (n : TwoDigitNumber) : Nat :=
  10 * (digitToNat n.tens) + (digitToNat n.units)

/-- Checks if all digits in two TwoDigitNumbers and their sum are unique and use 0-6 -/
def uniqueDigits (a b : TwoDigitNumber) (sum : Nat) : Prop :=
  let allDigits := [a.tens, a.units, b.tens, b.units, 
                    Digit.zero, Digit.one, Digit.two, Digit.three, Digit.four, Digit.five, Digit.six]
  allDigits.Nodup ∧ (sum / 100 = 1) ∧ (sum % 10 ≠ 0)

theorem two_digit_sum_units_digit 
  (a b : TwoDigitNumber) 
  (h : uniqueDigits a b ((twoDigitNumberToNat a) + (twoDigitNumberToNat b))) :
  ((twoDigitNumberToNat a) + (twoDigitNumberToNat b)) % 10 = 5 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_sum_units_digit_l3071_307162


namespace NUMINAMATH_CALUDE_boat_license_combinations_l3071_307151

/-- The number of possible letters for a boat license -/
def num_letters : ℕ := 3

/-- The number of possible digits for each position in a boat license -/
def num_digits : ℕ := 10

/-- The number of digit positions in a boat license -/
def num_digit_positions : ℕ := 5

/-- The total number of different boat license combinations -/
def total_license_combinations : ℕ := num_letters * (num_digits ^ num_digit_positions)

theorem boat_license_combinations :
  total_license_combinations = 300000 := by
  sorry

end NUMINAMATH_CALUDE_boat_license_combinations_l3071_307151


namespace NUMINAMATH_CALUDE_no_snow_probability_l3071_307189

theorem no_snow_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_snow_probability_l3071_307189


namespace NUMINAMATH_CALUDE_work_time_difference_l3071_307102

def monday_minutes : ℕ := 450
def wednesday_minutes : ℕ := 300

def tuesday_minutes : ℕ := monday_minutes / 2

theorem work_time_difference : wednesday_minutes - tuesday_minutes = 75 := by
  sorry

end NUMINAMATH_CALUDE_work_time_difference_l3071_307102


namespace NUMINAMATH_CALUDE_max_shapes_8x14_l3071_307108

/-- The number of grid points in an m × n rectangle --/
def gridPoints (m n : ℕ) : ℕ := (m + 1) * (n + 1)

/-- The number of grid points covered by each shape --/
def pointsPerShape : ℕ := 8

/-- The maximum number of shapes that can be placed in the grid --/
def maxShapes (m n : ℕ) : ℕ := (gridPoints m n) / pointsPerShape

theorem max_shapes_8x14 :
  maxShapes 8 14 = 16 := by sorry

end NUMINAMATH_CALUDE_max_shapes_8x14_l3071_307108


namespace NUMINAMATH_CALUDE_prob_specific_individual_in_sample_l3071_307147

/-- The probability of selecting a specific individual in a simple random sample -/
theorem prob_specific_individual_in_sample 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (h1 : population_size = 10)
  (h2 : sample_size = 3)
  (h3 : sample_size ≤ population_size) :
  (sample_size : ℚ) / population_size = 3 / 10 := by
  sorry

#check prob_specific_individual_in_sample

end NUMINAMATH_CALUDE_prob_specific_individual_in_sample_l3071_307147


namespace NUMINAMATH_CALUDE_age_difference_proof_l3071_307149

theorem age_difference_proof (A B n : ℚ) : 
  A = B + n →
  A - 2 = 6 * (B - 2) →
  A = 2 * B + 3 →
  n = 25 / 4 := by
sorry

#eval (25 : ℚ) / 4  -- To verify that 25/4 equals 6.25

end NUMINAMATH_CALUDE_age_difference_proof_l3071_307149


namespace NUMINAMATH_CALUDE_inequality_proof_l3071_307148

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 ≥ x*y*z*Real.sqrt 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3071_307148


namespace NUMINAMATH_CALUDE_max_value_fourth_root_plus_sqrt_l3071_307106

theorem max_value_fourth_root_plus_sqrt (a b c d : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) 
  (hb : 0 ≤ b ∧ b ≤ 2) 
  (hc : 0 ≤ c ∧ c ≤ 1) 
  (hd : 0 ≤ d ∧ d ≤ 1) : 
  ∃ (max : ℝ), max = 1 ∧ 
  ∀ x, x = (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) → 
  x ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_fourth_root_plus_sqrt_l3071_307106


namespace NUMINAMATH_CALUDE_alternating_color_probability_l3071_307180

/-- The number of white balls in the box -/
def num_white_balls : ℕ := 6

/-- The number of black balls in the box -/
def num_black_balls : ℕ := 6

/-- The total number of balls in the box -/
def total_balls : ℕ := num_white_balls + num_black_balls

/-- The number of ways to arrange white and black balls -/
def total_arrangements : ℕ := Nat.choose total_balls num_white_balls

/-- The number of arrangements where colors alternate -/
def alternating_arrangements : ℕ := 2

/-- The probability of drawing balls with alternating colors -/
def prob_alternating_colors : ℚ := alternating_arrangements / total_arrangements

theorem alternating_color_probability :
  prob_alternating_colors = 1 / 462 :=
by sorry

end NUMINAMATH_CALUDE_alternating_color_probability_l3071_307180


namespace NUMINAMATH_CALUDE_positive_square_harmonic_properties_l3071_307116

/-- Definition of a positive square harmonic function -/
def PositiveSquareHarmonic (f : ℝ → ℝ) : Prop :=
  (∀ x ∈ Set.Icc 0 1, f x ≥ 0) ∧
  (f 1 = 1) ∧
  (∀ x₁ x₂, x₁ + x₂ ∈ Set.Icc 0 1 → f x₁ + f x₂ ≤ f (x₁ + x₂))

theorem positive_square_harmonic_properties :
  ∀ f : ℝ → ℝ, PositiveSquareHarmonic f →
    (∀ x ∈ Set.Icc 0 1, f x = x^2) ∧
    (f 0 = 0) ∧
    (∀ x ∈ Set.Icc 0 1, f x ≤ 2*x) :=
by sorry

end NUMINAMATH_CALUDE_positive_square_harmonic_properties_l3071_307116


namespace NUMINAMATH_CALUDE_sum_of_dimensions_l3071_307109

/-- A rectangular box with dimensions A, B, and C, where AB = 40, AC = 90, and BC = 360 -/
structure RectangularBox where
  A : ℝ
  B : ℝ
  C : ℝ
  ab_area : A * B = 40
  ac_area : A * C = 90
  bc_area : B * C = 360

/-- The sum of dimensions A, B, and C of the rectangular box is 45 -/
theorem sum_of_dimensions (box : RectangularBox) : box.A + box.B + box.C = 45 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_dimensions_l3071_307109


namespace NUMINAMATH_CALUDE_mobile_purchase_price_l3071_307118

/-- Represents the purchase and sale of items with profit or loss -/
def ItemTransaction (purchase_price : ℚ) (profit_percent : ℚ) : ℚ :=
  purchase_price * (1 + profit_percent / 100)

theorem mobile_purchase_price :
  let grinder_price : ℚ := 15000
  let grinder_loss_percent : ℚ := 5
  let mobile_profit_percent : ℚ := 10
  let total_profit : ℚ := 50

  ∃ mobile_price : ℚ,
    (ItemTransaction grinder_price (-grinder_loss_percent) +
     ItemTransaction mobile_price mobile_profit_percent) -
    (grinder_price + mobile_price) = total_profit ∧
    mobile_price = 8000 :=
by sorry

end NUMINAMATH_CALUDE_mobile_purchase_price_l3071_307118


namespace NUMINAMATH_CALUDE_smallest_integer_solution_l3071_307157

theorem smallest_integer_solution : 
  (∀ x : ℤ, x < 1 → (x : ℚ) / 4 + 3 / 7 ≤ 2 / 3) ∧ 
  (1 : ℚ) / 4 + 3 / 7 > 2 / 3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_solution_l3071_307157


namespace NUMINAMATH_CALUDE_daryl_crate_loading_problem_l3071_307113

theorem daryl_crate_loading_problem :
  let crate_capacity : ℕ := 20
  let num_crates : ℕ := 15
  let total_capacity : ℕ := crate_capacity * num_crates
  let num_nail_bags : ℕ := 4
  let nail_bag_weight : ℕ := 5
  let num_hammer_bags : ℕ := 12
  let hammer_bag_weight : ℕ := 5
  let num_plank_bags : ℕ := 10
  let plank_bag_weight : ℕ := 30
  let total_nail_weight : ℕ := num_nail_bags * nail_bag_weight
  let total_hammer_weight : ℕ := num_hammer_bags * hammer_bag_weight
  let total_plank_weight : ℕ := num_plank_bags * plank_bag_weight
  let total_item_weight : ℕ := total_nail_weight + total_hammer_weight + total_plank_weight
  total_item_weight - total_capacity = 80 :=
by sorry

end NUMINAMATH_CALUDE_daryl_crate_loading_problem_l3071_307113


namespace NUMINAMATH_CALUDE_group_collection_problem_l3071_307184

theorem group_collection_problem (n : ℕ) (total_rupees : ℚ) : 
  (n : ℚ) * n = total_rupees * 100 →
  total_rupees = 19.36 →
  n = 44 := by
  sorry

end NUMINAMATH_CALUDE_group_collection_problem_l3071_307184


namespace NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3071_307111

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end NUMINAMATH_CALUDE_one_third_of_seven_times_nine_l3071_307111


namespace NUMINAMATH_CALUDE_largest_fraction_l3071_307103

theorem largest_fraction : 
  let fractions := [2/5, 3/7, 5/9, 4/11, 3/8]
  ∀ x ∈ fractions, (5:ℚ)/9 ≥ x := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l3071_307103


namespace NUMINAMATH_CALUDE_competition_score_l3071_307142

theorem competition_score (total_judges : Nat) (highest_score lowest_score avg_score : ℝ) :
  total_judges = 9 →
  highest_score = 86 →
  lowest_score = 45 →
  avg_score = 76 →
  (total_judges * avg_score - highest_score - lowest_score) / (total_judges - 2) = 79 := by
  sorry

end NUMINAMATH_CALUDE_competition_score_l3071_307142


namespace NUMINAMATH_CALUDE_orange_juice_remaining_l3071_307194

theorem orange_juice_remaining (initial_amount : ℚ) (given_away : ℚ) : 
  initial_amount = 5 → given_away = 18/7 → initial_amount - given_away = 17/7 := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_remaining_l3071_307194


namespace NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3071_307191

/-- A quadratic function passing through (-1,0) and (3,0) with a minimum value of 28 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  passes_through_minus_one : a * (-1)^2 + b * (-1) + c = 0
  passes_through_three : a * 3^2 + b * 3 + c = 0
  min_value : ∃ (x : ℝ), ∀ (y : ℝ), a * x^2 + b * x + c ≤ a * y^2 + b * y + c ∧ a * x^2 + b * x + c = 28

/-- The sum of coefficients of the quadratic function is 28 -/
theorem quadratic_sum_of_coefficients (f : QuadraticFunction) : f.a + f.b + f.c = 28 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_of_coefficients_l3071_307191


namespace NUMINAMATH_CALUDE_min_value_expression_l3071_307159

theorem min_value_expression (x y : ℝ) (hx : x > 2) (hy : y > 2) :
  (x^3 / (y - 2)) + (y^3 / (x - 2)) ≥ 64 ∧
  ∃ x y, x > 2 ∧ y > 2 ∧ (x^3 / (y - 2)) + (y^3 / (x - 2)) = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3071_307159


namespace NUMINAMATH_CALUDE_root_in_interval_l3071_307185

def f (x : ℝ) : ℝ := x^3 - x - 3

theorem root_in_interval :
  ∃ c ∈ Set.Icc 1 2, f c = 0 :=
sorry

end NUMINAMATH_CALUDE_root_in_interval_l3071_307185


namespace NUMINAMATH_CALUDE_yellow_balls_count_l3071_307112

theorem yellow_balls_count (total : Nat) (white green red purple : Nat) (prob_not_red_purple : Real) :
  total = 60 →
  white = 22 →
  green = 18 →
  red = 15 →
  purple = 3 →
  prob_not_red_purple = 0.7 →
  ∃ yellow : Nat, yellow = 2 ∧ 
    total = white + green + yellow + red + purple ∧
    (white + green + yellow : Real) / total = prob_not_red_purple :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l3071_307112


namespace NUMINAMATH_CALUDE_count_four_digit_snappy_divisible_by_25_l3071_307132

def is_snappy (n : ℕ) : Prop :=
  ∃ a b : ℕ, n = a * 1000 + b * 100 + b * 10 + a ∧ a < 10 ∧ b < 10

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem count_four_digit_snappy_divisible_by_25 :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_four_digit n ∧ is_snappy n ∧ n % 25 = 0) ∧
    s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_count_four_digit_snappy_divisible_by_25_l3071_307132


namespace NUMINAMATH_CALUDE_square_binomial_minus_square_l3071_307175

theorem square_binomial_minus_square : 15^2 + 2*(15*5) + 5^2 - 3^2 = 391 := by
  sorry

end NUMINAMATH_CALUDE_square_binomial_minus_square_l3071_307175


namespace NUMINAMATH_CALUDE_martha_centerpiece_cost_l3071_307150

/-- Calculates the total cost of flowers for centerpieces -/
def total_flower_cost (num_centerpieces : ℕ) (roses_per_centerpiece : ℕ) 
  (orchids_per_centerpiece : ℕ) (lilies_per_centerpiece : ℕ) (cost_per_flower : ℕ) : ℕ :=
  num_centerpieces * (roses_per_centerpiece + orchids_per_centerpiece + lilies_per_centerpiece) * cost_per_flower

/-- Theorem: The total cost for flowers for 6 centerpieces is $2700 -/
theorem martha_centerpiece_cost : 
  total_flower_cost 6 8 16 6 15 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_martha_centerpiece_cost_l3071_307150


namespace NUMINAMATH_CALUDE_avery_wall_time_l3071_307119

/-- The time it takes Avery to build the wall alone -/
def avery_time : ℝ := 4

/-- The time it takes Tom to build the wall alone -/
def tom_time : ℝ := 2

/-- The additional time Tom needs to finish the wall after working with Avery for 1 hour -/
def tom_additional_time : ℝ := 0.5

theorem avery_wall_time : 
  (1 / avery_time + 1 / tom_time) + tom_additional_time / tom_time = 1 := by sorry

end NUMINAMATH_CALUDE_avery_wall_time_l3071_307119


namespace NUMINAMATH_CALUDE_annie_completion_time_correct_l3071_307188

/-- Dan's time to complete the job alone -/
def dan_time : ℝ := 15

/-- Annie's time to complete the job alone -/
def annie_time : ℝ := 3.6

/-- Time Dan works before stopping -/
def dan_work_time : ℝ := 6

/-- Time Annie takes to finish the job after Dan stops -/
def annie_finish_time : ℝ := 6

/-- The theorem stating that Annie's time to complete the job alone is correct -/
theorem annie_completion_time_correct :
  (dan_work_time / dan_time) + (annie_finish_time / annie_time) = 1 := by
  sorry

end NUMINAMATH_CALUDE_annie_completion_time_correct_l3071_307188


namespace NUMINAMATH_CALUDE_puppy_sleep_duration_l3071_307196

theorem puppy_sleep_duration (connor_sleep : ℕ) (luke_sleep : ℕ) (puppy_sleep : ℕ) : 
  connor_sleep = 6 →
  luke_sleep = connor_sleep + 2 →
  puppy_sleep = 2 * luke_sleep →
  puppy_sleep = 16 := by
sorry

end NUMINAMATH_CALUDE_puppy_sleep_duration_l3071_307196


namespace NUMINAMATH_CALUDE_other_endpoint_coordinates_l3071_307120

/-- Given a line segment with midpoint (3, 0) and one endpoint at (7, -4), 
    prove that the other endpoint is at (-1, 4) -/
theorem other_endpoint_coordinates :
  ∀ (A B : ℝ × ℝ),
    (A.1 + B.1) / 2 = 3 ∧
    (A.2 + B.2) / 2 = 0 ∧
    A = (7, -4) →
    B = (-1, 4) := by
  sorry

end NUMINAMATH_CALUDE_other_endpoint_coordinates_l3071_307120


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3071_307146

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_eq_4 : x + y + z = 4) : 
  (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 4 → 
    9/x + 1/y + 25/z ≤ 9/a + 1/b + 25/c) ∧ 
  9/x + 1/y + 25/z = 20.25 := by
sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l3071_307146


namespace NUMINAMATH_CALUDE_max_value_when_min_ratio_l3071_307165

theorem max_value_when_min_ratio (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x^2 - 3*x*y + 4*y^2 - z = 0) :
  ∃ (max_value : ℝ), max_value = 2 ∧
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 →
  x'^2 - 3*x'*y' + 4*y'^2 - z' = 0 →
  (z' / (x' * y') ≥ z / (x * y)) →
  x' + 2*y' - z' ≤ max_value :=
sorry

end NUMINAMATH_CALUDE_max_value_when_min_ratio_l3071_307165


namespace NUMINAMATH_CALUDE_product_of_four_consecutive_integers_plus_one_is_perfect_square_l3071_307193

theorem product_of_four_consecutive_integers_plus_one_is_perfect_square (n : ℤ) :
  ∃ m : ℤ, (n - 1) * n * (n + 1) * (n + 2) + 1 = m ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_four_consecutive_integers_plus_one_is_perfect_square_l3071_307193


namespace NUMINAMATH_CALUDE_smallest_product_l3071_307169

def S : Finset ℤ := {-9, -5, -1, 1, 4}

theorem smallest_product (a b : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hab : a ≠ b) :
  ∃ (x y : ℤ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y), 
    x * y ≤ a * b ∧ x * y = -36 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_l3071_307169


namespace NUMINAMATH_CALUDE_sum_equals_17b_l3071_307172

theorem sum_equals_17b (b : ℝ) (a c d : ℝ) 
  (ha : a = 3 * b) 
  (hc : c = 2 * a) 
  (hd : d = c + b) : 
  a + b + c + d = 17 * b := by
sorry

end NUMINAMATH_CALUDE_sum_equals_17b_l3071_307172
