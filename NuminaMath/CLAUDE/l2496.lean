import Mathlib

namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l2496_249633

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - x + k ≠ 0) → k > 1/4 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l2496_249633


namespace NUMINAMATH_CALUDE_twelfth_term_value_l2496_249604

/-- The nth term of a geometric sequence -/
def geometric_term (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

/-- The 12th term of the specific geometric sequence -/
def twelfth_term : ℚ :=
  geometric_term 5 (2/5) 12

theorem twelfth_term_value : twelfth_term = 10240/48828125 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_value_l2496_249604


namespace NUMINAMATH_CALUDE_paul_crayons_left_l2496_249607

/-- The number of crayons Paul had at the end of the school year -/
def crayons_left (initial_crayons lost_crayons : ℕ) : ℕ :=
  initial_crayons - lost_crayons

/-- Theorem: Paul had 291 crayons left at the end of the school year -/
theorem paul_crayons_left : crayons_left 606 315 = 291 := by
  sorry

end NUMINAMATH_CALUDE_paul_crayons_left_l2496_249607


namespace NUMINAMATH_CALUDE_min_side_length_l2496_249638

/-- Given two triangles PQR and SQR sharing side QR, prove that the minimum possible
    integral length of QR is 15 cm, given the lengths of other sides. -/
theorem min_side_length (PQ PR SR QS : ℝ) (hPQ : PQ = 7) (hPR : PR = 15) (hSR : SR = 10) (hQS : QS = 25) :
  ∃ (QR : ℕ), QR ≥ 15 ∧ ∀ (n : ℕ), n ≥ 15 → (n : ℝ) > PR - PQ ∧ (n : ℝ) > QS - SR :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l2496_249638


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l2496_249649

def initial_thickness : ℝ := 0.1
def num_folds : ℕ := 4

theorem paper_folding_thickness :
  initial_thickness * (2 ^ num_folds) = 1.6 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l2496_249649


namespace NUMINAMATH_CALUDE_subset_properties_l2496_249698

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the property that B is a subset of A
def is_subset_of_A (B : Set ℝ) : Prop := B ⊆ A

-- Theorem statement
theorem subset_properties (B : Set ℝ) (h : A ∩ B = B) :
  is_subset_of_A ∅ ∧
  is_subset_of_A {1} ∧
  is_subset_of_A A ∧
  ¬is_subset_of_A {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_subset_properties_l2496_249698


namespace NUMINAMATH_CALUDE_min_value_expression_l2496_249684

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (9 * b) / (4 * a) + (a + b) / b ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2496_249684


namespace NUMINAMATH_CALUDE_exam_max_marks_l2496_249668

theorem exam_max_marks (victor_score : ℝ) (victor_percentage : ℝ) (max_marks : ℝ) : 
  victor_score = 184 → 
  victor_percentage = 0.92 → 
  victor_score = victor_percentage * max_marks → 
  max_marks = 200 := by
sorry

end NUMINAMATH_CALUDE_exam_max_marks_l2496_249668


namespace NUMINAMATH_CALUDE_equal_costs_at_twenty_l2496_249648

/-- Represents the cost function for company A -/
def cost_A (x : ℝ) : ℝ := 450 * x + 1000

/-- Represents the cost function for company B -/
def cost_B (x : ℝ) : ℝ := 500 * x

/-- Theorem stating that the costs are equal when 20 desks are purchased -/
theorem equal_costs_at_twenty :
  ∃ (x : ℝ), x = 20 ∧ cost_A x = cost_B x :=
sorry

end NUMINAMATH_CALUDE_equal_costs_at_twenty_l2496_249648


namespace NUMINAMATH_CALUDE_perpendicular_vector_implies_y_coord_l2496_249621

/-- Given two points A and B, and a vector a, if AB is perpendicular to a, 
    then the y-coordinate of B is -4. -/
theorem perpendicular_vector_implies_y_coord (A B : ℝ × ℝ) (a : ℝ × ℝ) : 
  A = (-1, 2) → 
  B.1 = 2 → 
  a = (2, 1) → 
  (B.1 - A.1, B.2 - A.2) • a = 0 → 
  B.2 = -4 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vector_implies_y_coord_l2496_249621


namespace NUMINAMATH_CALUDE_reeya_fourth_subject_score_l2496_249660

theorem reeya_fourth_subject_score 
  (score1 score2 score3 : ℕ) 
  (average : ℚ) 
  (h1 : score1 = 55)
  (h2 : score2 = 67)
  (h3 : score3 = 76)
  (h4 : average = 67)
  (h5 : ∀ s : ℕ, s ≤ 100) -- Assuming all scores are out of 100
  : ∃ score4 : ℕ, 
    (score1 + score2 + score3 + score4 : ℚ) / 4 = average ∧ 
    score4 = 70 := by
  sorry

end NUMINAMATH_CALUDE_reeya_fourth_subject_score_l2496_249660


namespace NUMINAMATH_CALUDE_lunch_breakfast_difference_l2496_249624

def breakfast_cost : ℚ := 2 + 3 + 4 + 3.5 + 1.5

def lunch_base_cost : ℚ := 3.5 + 4 + 5.25 + 6 + 1 + 3

def service_charge (cost : ℚ) : ℚ := cost * (1 + 0.1)

def food_tax (cost : ℚ) : ℚ := cost * (1 + 0.05)

def lunch_total_cost : ℚ := food_tax (service_charge lunch_base_cost)

theorem lunch_breakfast_difference :
  lunch_total_cost - breakfast_cost = 12.28 := by sorry

end NUMINAMATH_CALUDE_lunch_breakfast_difference_l2496_249624


namespace NUMINAMATH_CALUDE_no_nontrivial_integer_solutions_l2496_249630

theorem no_nontrivial_integer_solutions :
  ∀ (x y z : ℤ), x^3 + 2*y^3 + 4*z^3 - 6*x*y*z = 0 → x = 0 ∧ y = 0 ∧ z = 0 :=
by sorry

end NUMINAMATH_CALUDE_no_nontrivial_integer_solutions_l2496_249630


namespace NUMINAMATH_CALUDE_problem_1_solution_problem_2_no_solution_l2496_249658

-- Problem 1
theorem problem_1_solution (x : ℝ) :
  (x / (2*x - 5) + 5 / (5 - 2*x) = 1) ↔ (x = 0) :=
sorry

-- Problem 2
theorem problem_2_no_solution :
  ¬∃ (x : ℝ), ((2*x + 9) / (3*x - 9) = (4*x - 7) / (x - 3) + 2) :=
sorry

end NUMINAMATH_CALUDE_problem_1_solution_problem_2_no_solution_l2496_249658


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2496_249676

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*x - 2*y + 1 = 0

/-- A point on the line -/
def point : ℝ × ℝ := (-2, 5)

/-- Possible equations of the tangent line -/
def tangent_line_eq1 (x : ℝ) : Prop := x = -2
def tangent_line_eq2 (x y : ℝ) : Prop := 15*x + 8*y - 10 = 0

/-- The main theorem -/
theorem tangent_line_equation :
  ∃ (x y : ℝ), (x = point.1 ∧ y = point.2) ∧
  (∀ (x' y' : ℝ), circle_equation x' y' →
    (tangent_line_eq1 x ∨ tangent_line_eq2 x y) ∧
    (x = x' ∧ y = y' → ¬circle_equation x y)) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2496_249676


namespace NUMINAMATH_CALUDE_problem_solution_l2496_249682

theorem problem_solution (p q : ℚ) 
  (h1 : 5 * p + 7 * q = 19)
  (h2 : 7 * p + 5 * q = 26) : 
  p = 29 / 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2496_249682


namespace NUMINAMATH_CALUDE_function_passes_through_point_l2496_249643

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ y : ℝ, y = a * 2 - 2 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_function_passes_through_point_l2496_249643


namespace NUMINAMATH_CALUDE_toy_bear_production_efficiency_l2496_249663

theorem toy_bear_production_efficiency (B H : ℝ) (H' : ℝ) : 
  B > 0 → H > 0 →
  (1.8 * B = 2 * (B / H) * H') →
  (H - H') / H * 100 = 10 :=
by sorry

end NUMINAMATH_CALUDE_toy_bear_production_efficiency_l2496_249663


namespace NUMINAMATH_CALUDE_sqrt_5_simplest_l2496_249642

def is_simplest_sqrt (x : ℝ) : Prop :=
  ∀ y : ℝ, y > 0 → x = Real.sqrt y → ¬∃ (a b : ℝ), b ≠ 1 ∧ y = a * b^2

theorem sqrt_5_simplest :
  is_simplest_sqrt (Real.sqrt 5) ∧
  ¬is_simplest_sqrt (Real.sqrt 9) ∧
  ¬is_simplest_sqrt (Real.sqrt 18) ∧
  ¬is_simplest_sqrt (Real.sqrt (1/2)) :=
sorry

end NUMINAMATH_CALUDE_sqrt_5_simplest_l2496_249642


namespace NUMINAMATH_CALUDE_inequality_system_solution_l2496_249699

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 2 < x ∧ (1/3) * x < -2) ↔ x < -6 := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l2496_249699


namespace NUMINAMATH_CALUDE_no_given_factors_of_polynomial_l2496_249680

theorem no_given_factors_of_polynomial :
  let p (x : ℝ) := x^4 - 2*x^2 + 9
  let factors := [
    (fun x => x^2 + 3),
    (fun x => x + 1),
    (fun x => x^2 - 3),
    (fun x => x^2 + 2*x - 3)
  ]
  ∀ f ∈ factors, ¬ (∃ q : ℝ → ℝ, ∀ x, p x = f x * q x) :=
by sorry

end NUMINAMATH_CALUDE_no_given_factors_of_polynomial_l2496_249680


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l2496_249656

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l2496_249656


namespace NUMINAMATH_CALUDE_orange_count_l2496_249635

theorem orange_count (total : ℕ) (apple_ratio : ℕ) (orange_count : ℕ) : 
  total = 40 →
  apple_ratio = 3 →
  orange_count + apple_ratio * orange_count = total →
  orange_count = 10 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l2496_249635


namespace NUMINAMATH_CALUDE_same_color_combination_probability_l2496_249694

def total_candies : ℕ := 12 + 8 + 5

theorem same_color_combination_probability :
  let red : ℕ := 12
  let blue : ℕ := 8
  let green : ℕ := 5
  let total : ℕ := total_candies
  
  -- Probability of picking two red candies
  let p_red : ℚ := (red * (red - 1)) / (total * (total - 1)) *
                   ((red - 2) * (red - 3)) / ((total - 2) * (total - 3))
  
  -- Probability of picking two blue candies
  let p_blue : ℚ := (blue * (blue - 1)) / (total * (total - 1)) *
                    ((blue - 2) * (blue - 3)) / ((total - 2) * (total - 3))
  
  -- Probability of picking two green candies
  let p_green : ℚ := (green * (green - 1)) / (total * (total - 1)) *
                     ((green - 2) * (green - 3)) / ((total - 2) * (total - 3))
  
  -- Total probability of picking the same color combination
  p_red + p_blue + p_green = 11 / 77 :=
by sorry

end NUMINAMATH_CALUDE_same_color_combination_probability_l2496_249694


namespace NUMINAMATH_CALUDE_smallest_bob_number_l2496_249688

def alice_number : ℕ := 36

def has_all_prime_factors (n m : ℕ) : Prop :=
  ∀ p : ℕ, p.Prime → (p ∣ n → p ∣ m)

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), 
    bob_number > 0 ∧
    has_all_prime_factors alice_number bob_number ∧
    (∀ k : ℕ, k > 0 ∧ has_all_prime_factors alice_number k → bob_number ≤ k) ∧
    bob_number = 6 := by
  sorry

end NUMINAMATH_CALUDE_smallest_bob_number_l2496_249688


namespace NUMINAMATH_CALUDE_odd_integer_divides_power_factorial_minus_one_l2496_249685

theorem odd_integer_divides_power_factorial_minus_one (n : ℕ) (h_odd : Odd n) (h_ge_one : n ≥ 1) :
  n ∣ 2^(n!) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_integer_divides_power_factorial_minus_one_l2496_249685


namespace NUMINAMATH_CALUDE_investment_balance_l2496_249657

/-- Proves that given an initial investment of 1800 at 7% interest, an additional investment of 1800 at 10% interest will result in a total annual income equal to 8.5% of the entire investment. -/
theorem investment_balance (initial_investment : ℝ) (additional_investment : ℝ) 
  (initial_rate : ℝ) (additional_rate : ℝ) (total_rate : ℝ) : 
  initial_investment = 1800 →
  additional_investment = 1800 →
  initial_rate = 0.07 →
  additional_rate = 0.10 →
  total_rate = 0.085 →
  initial_rate * initial_investment + additional_rate * additional_investment = 
    total_rate * (initial_investment + additional_investment) :=
by sorry

end NUMINAMATH_CALUDE_investment_balance_l2496_249657


namespace NUMINAMATH_CALUDE_no_multiple_of_five_2C4_l2496_249628

theorem no_multiple_of_five_2C4 : 
  ¬ ∃ (C : ℕ), 
    (100 ≤ 200 + 10 * C + 4) ∧ 
    (200 + 10 * C + 4 < 1000) ∧ 
    (C < 10) ∧ 
    ((200 + 10 * C + 4) % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_multiple_of_five_2C4_l2496_249628


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l2496_249683

theorem green_shirt_pairs (red_students : ℕ) (green_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (red_red_pairs : ℕ) :
  red_students = 70 →
  green_students = 58 →
  total_students = 128 →
  total_pairs = 64 →
  red_red_pairs = 34 →
  ∃ (green_green_pairs : ℕ), green_green_pairs = 28 :=
by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l2496_249683


namespace NUMINAMATH_CALUDE_log_inequality_l2496_249661

theorem log_inequality (n : ℕ+) (k : ℕ) (h : k = (Nat.factors n).card) :
  Real.log n ≥ k * Real.log 2 := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l2496_249661


namespace NUMINAMATH_CALUDE_triangle_property_l2496_249616

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  b = 5 →
  c = 7 →
  (a + c) / b = (Real.sin B + Real.sin A) / (Real.sin C - Real.sin A) →
  C = 2 * Real.pi / 3 ∧
  (1 / 2) * a * b * Real.sin C = 15 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l2496_249616


namespace NUMINAMATH_CALUDE_quadratic_vertex_l2496_249696

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + (8 - m)*x + 12

-- Define the derivative of the function
def f' (m : ℝ) (x : ℝ) : ℝ := -2*x + (8 - m)

-- Theorem statement
theorem quadratic_vertex (m : ℝ) :
  (∀ x > 2, (f' m x < 0)) ∧ 
  (∀ x < 2, (f' m x > 0)) →
  m = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l2496_249696


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2496_249632

/-- Given an arithmetic sequence {aₙ} with sum Sₙ of the first n terms,
    if S₆ > S₇ > S₅, then the common difference d < 0 and |a₆| > |a₇| -/
theorem arithmetic_sequence_properties
  (a : ℕ → ℝ)  -- The arithmetic sequence
  (S : ℕ → ℝ)  -- The sum function
  (d : ℝ)      -- The common difference
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)  -- Definition of Sₙ
  (h2 : ∀ n, a (n + 1) = a n + d)        -- Definition of arithmetic sequence
  (h3 : S 6 > S 7)                       -- Given condition
  (h4 : S 7 > S 5)                       -- Given condition
  : d < 0 ∧ |a 6| > |a 7| := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2496_249632


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2496_249634

theorem arithmetic_expression_equality : 8 + 18 / 3 - 4 * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2496_249634


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_4_range_of_a_l2496_249646

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 3| + |x - 1|

-- Theorem for part (I)
theorem solution_set_f_greater_than_4 :
  {x : ℝ | f x > 4} = {x : ℝ | x < -2 ∨ x > 0} := by sorry

-- Theorem for part (II)
theorem range_of_a (a : ℝ) :
  (∃ x ∈ Set.Icc (-3/2) 1, a + 1 > f x) → a > 3/2 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_4_range_of_a_l2496_249646


namespace NUMINAMATH_CALUDE_orange_groups_count_l2496_249691

/-- The number of groups of oranges in Philip's collection -/
def orange_groups (total_oranges : ℕ) (oranges_per_group : ℕ) : ℕ :=
  total_oranges / oranges_per_group

/-- Theorem stating that the number of orange groups is 16 -/
theorem orange_groups_count :
  orange_groups 384 24 = 16 := by
  sorry

end NUMINAMATH_CALUDE_orange_groups_count_l2496_249691


namespace NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2496_249637

theorem rectangle_circle_area_ratio :
  ∀ (w r : ℝ),
  w > 0 → r > 0 →
  6 * w = 2 * π * r →
  (2 * w * w) / (π * r^2) = 2 * π / 9 := by
sorry

end NUMINAMATH_CALUDE_rectangle_circle_area_ratio_l2496_249637


namespace NUMINAMATH_CALUDE_min_value_of_function_l2496_249640

theorem min_value_of_function :
  let f : ℝ → ℝ := λ x => 5/4 - Real.sin x^2 - 3 * Real.cos x
  ∃ x₀ : ℝ, ∀ x : ℝ, f x ≥ f x₀ ∧ f x₀ = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_function_l2496_249640


namespace NUMINAMATH_CALUDE_jason_has_36_seashells_l2496_249697

/-- The number of seashells Jason has now, given his initial count and the number he gave away. -/
def jasonsSeashells (initialCount gaveAway : ℕ) : ℕ :=
  initialCount - gaveAway

/-- Theorem stating that Jason has 36 seashells after giving some away. -/
theorem jason_has_36_seashells : jasonsSeashells 49 13 = 36 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_36_seashells_l2496_249697


namespace NUMINAMATH_CALUDE_tesseract_triangles_l2496_249611

/-- The number of vertices in a tesseract -/
def tesseract_vertices : ℕ := 16

/-- The number of vertices required to form a triangle -/
def triangle_vertices : ℕ := 3

/-- The number of distinct triangles in a tesseract -/
def distinct_triangles : ℕ := Nat.choose tesseract_vertices triangle_vertices

theorem tesseract_triangles : distinct_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_tesseract_triangles_l2496_249611


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2496_249617

theorem arithmetic_sequence_middle_term (z : ℤ) :
  (∃ (a d : ℤ), 3^2 = a ∧ z = a + d ∧ 3^3 = a + 2*d) → z = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l2496_249617


namespace NUMINAMATH_CALUDE_project_cost_sharing_l2496_249653

/-- Given initial payments P and Q, and an additional cost R, 
    calculate the amount Javier must pay to Cora for equal cost sharing. -/
theorem project_cost_sharing 
  (P Q R : ℝ) 
  (h1 : R = 3 * Q - 2 * P) 
  (h2 : P < Q) : 
  (2 * Q - P) / 2 = (P + Q + R) / 2 - Q := by sorry

end NUMINAMATH_CALUDE_project_cost_sharing_l2496_249653


namespace NUMINAMATH_CALUDE_largest_product_l2496_249695

def digits : List Nat := [5, 6, 7, 8]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ (n / 10) ∈ digits ∧ (n % 10) ∈ digits

def valid_pair (a b : Nat) : Prop :=
  is_valid_number a ∧ is_valid_number b ∧
  (a / 10 ≠ b / 10) ∧ (a / 10 ≠ b % 10) ∧ (a % 10 ≠ b / 10) ∧ (a % 10 ≠ b % 10)

theorem largest_product :
  ∀ a b : Nat, valid_pair a b → a * b ≤ 3886 :=
by sorry

end NUMINAMATH_CALUDE_largest_product_l2496_249695


namespace NUMINAMATH_CALUDE_hash_three_two_l2496_249627

-- Define the # operation
def hash (a b : ℕ) : ℕ := a * (b + 1) + a * b + b^2

-- Theorem statement
theorem hash_three_two : hash 3 2 = 19 := by
  sorry

end NUMINAMATH_CALUDE_hash_three_two_l2496_249627


namespace NUMINAMATH_CALUDE_range_of_m_l2496_249662

-- Define propositions P and Q
def P (m : ℝ) : Prop := ∀ x : ℝ, x^2 - 2*m*x + m ≠ 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m*x + 1 ≥ 0

-- Define the condition that either P or Q is true, and both P and Q are false
def condition (m : ℝ) : Prop := 
  (P m ∨ Q m) ∧ ¬(P m ∧ Q m)

-- Theorem statement
theorem range_of_m :
  ∀ m : ℝ, condition m ↔ ((-2 ≤ m ∧ m ≤ 0) ∨ (1 ≤ m ∧ m ≤ 2)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l2496_249662


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l2496_249650

theorem sqrt_product_equality : Real.sqrt 3 * Real.sqrt 2 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l2496_249650


namespace NUMINAMATH_CALUDE_visitors_not_enjoy_not_understand_l2496_249609

-- Define the total number of visitors
def V : ℕ := 560

-- Define the number of visitors who enjoyed the painting
def E : ℕ := (3 * V) / 4

-- Define the number of visitors who understood the painting
def U : ℕ := E

-- Theorem to prove
theorem visitors_not_enjoy_not_understand : V - E = 140 := by
  sorry

end NUMINAMATH_CALUDE_visitors_not_enjoy_not_understand_l2496_249609


namespace NUMINAMATH_CALUDE_no_odd_4digit_div5_no05_l2496_249669

theorem no_odd_4digit_div5_no05 : 
  ¬ ∃ n : ℕ, 
    1000 ≤ n ∧ n < 10000 ∧  -- 4-digit
    n % 2 = 1 ∧             -- odd
    n % 5 = 0 ∧             -- divisible by 5
    (∀ d : ℕ, d < 4 → (n / 10^d) % 10 ≠ 0 ∧ (n / 10^d) % 10 ≠ 5) -- no 0 or 5 digits
    := by sorry

end NUMINAMATH_CALUDE_no_odd_4digit_div5_no05_l2496_249669


namespace NUMINAMATH_CALUDE_original_number_is_45_l2496_249610

theorem original_number_is_45 (x : ℝ) : x - 30 = x / 3 → x = 45 := by sorry

end NUMINAMATH_CALUDE_original_number_is_45_l2496_249610


namespace NUMINAMATH_CALUDE_co_captains_probability_l2496_249625

def team_sizes : List Nat := [6, 8, 9, 10]
def num_teams : Nat := 4
def co_captains_per_team : Nat := 3

def probability_co_captains (n : Nat) : Rat :=
  6 / (n * (n - 1) * (n - 2))

theorem co_captains_probability : 
  (1 / num_teams) * (team_sizes.map probability_co_captains).sum = 37 / 1680 := by
  sorry

end NUMINAMATH_CALUDE_co_captains_probability_l2496_249625


namespace NUMINAMATH_CALUDE_nathan_family_storage_cost_l2496_249636

/-- The cost to store items for a group at the temple shop -/
def storage_cost (num_people : ℕ) (objects_per_person : ℕ) (cost_per_object : ℕ) : ℕ :=
  num_people * objects_per_person * cost_per_object

/-- Proof that the storage cost for Nathan and his parents is 165 dollars -/
theorem nathan_family_storage_cost :
  storage_cost 3 5 11 = 165 := by
  sorry

end NUMINAMATH_CALUDE_nathan_family_storage_cost_l2496_249636


namespace NUMINAMATH_CALUDE_least_number_of_cubes_l2496_249612

def block_length : ℕ := 15
def block_width : ℕ := 30
def block_height : ℕ := 75

theorem least_number_of_cubes :
  let gcd := Nat.gcd (Nat.gcd block_length block_width) block_height
  let cube_side := gcd
  let num_cubes := (block_length * block_width * block_height) / (cube_side * cube_side * cube_side)
  num_cubes = 10 := by sorry

end NUMINAMATH_CALUDE_least_number_of_cubes_l2496_249612


namespace NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l2496_249602

theorem tens_digit_of_9_pow_2023 : ∃ n : ℕ, n ≥ 10 ∧ n < 100 ∧ 9^2023 % 100 = n ∧ (n / 10) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_9_pow_2023_l2496_249602


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l2496_249605

theorem unique_solution_for_equation :
  ∃! (x y : ℝ), (x - 10)^2 + (y - 11)^2 + (x - y)^2 = 1/3 ∧ 
  x = 10 + 1/3 ∧ y = 10 + 2/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l2496_249605


namespace NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2496_249614

/-- In an isosceles triangle DEF where angle D is congruent to angle E, 
    and the measure of angle E is three times the measure of angle F, 
    the measure of angle D is 540/7 degrees. -/
theorem isosceles_triangle_angle_measure (D E F : ℝ) : 
  D = E →                         -- Angle D is congruent to angle E
  E = 3 * F →                     -- Measure of angle E is three times the measure of angle F
  D + E + F = 180 →               -- Sum of angles in a triangle is 180 degrees
  D = 540 / 7 := by sorry         -- Measure of angle D is 540/7 degrees

end NUMINAMATH_CALUDE_isosceles_triangle_angle_measure_l2496_249614


namespace NUMINAMATH_CALUDE_alex_friends_cookout_l2496_249608

theorem alex_friends_cookout (burgers_per_guest : ℕ) (buns_per_pack : ℕ) (packs_of_buns : ℕ) 
  (h1 : burgers_per_guest = 3)
  (h2 : buns_per_pack = 8)
  (h3 : packs_of_buns = 3) :
  ∃ (friends : ℕ), friends = 9 ∧ 
    (packs_of_buns * buns_per_pack) / burgers_per_guest + 1 = friends :=
by
  sorry

end NUMINAMATH_CALUDE_alex_friends_cookout_l2496_249608


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_properties_l2496_249647

/-- Calculate the nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Calculate the number of dots in the perimeter of the nth triangular figure -/
def perimeter_dots (n : ℕ) : ℕ := n + 2 * (n - 1)

theorem thirtieth_triangular_number_properties :
  (triangular_number 30 = 465) ∧ (perimeter_dots 30 = 88) := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_properties_l2496_249647


namespace NUMINAMATH_CALUDE_parabola_transformation_l2496_249644

/-- Represents a parabola in the Cartesian plane -/
structure Parabola where
  a : ℝ
  h : ℝ
  k : ℝ

/-- Shifts a parabola horizontally -/
def shift_horizontal (p : Parabola) (dx : ℝ) : Parabola :=
  { a := p.a, h := p.h - dx, k := p.k }

/-- Shifts a parabola vertically -/
def shift_vertical (p : Parabola) (dy : ℝ) : Parabola :=
  { a := p.a, h := p.h, k := p.k + dy }

/-- The initial parabola y = -(x+1)^2 + 2 -/
def initial_parabola : Parabola :=
  { a := -1, h := 1, k := 2 }

/-- The final parabola y = -(x+2)^2 -/
def final_parabola : Parabola :=
  { a := -1, h := 2, k := 0 }

theorem parabola_transformation :
  (shift_vertical (shift_horizontal initial_parabola 1) (-2)) = final_parabola := by
  sorry

end NUMINAMATH_CALUDE_parabola_transformation_l2496_249644


namespace NUMINAMATH_CALUDE_unique_correct_expression_l2496_249613

theorem unique_correct_expression :
  ((-3 - 1 = -2) = False) ∧
  ((-2 * (-1/2) = 1) = True) ∧
  ((16 / (-4/3) = 12) = False) ∧
  ((-3^2 / 4 = 9/4) = False) := by
  sorry

end NUMINAMATH_CALUDE_unique_correct_expression_l2496_249613


namespace NUMINAMATH_CALUDE_coin_flip_theorem_l2496_249679

/-- Represents the state of coins on a table -/
structure CoinState where
  total_coins : ℕ
  two_ruble_coins : ℕ
  five_ruble_coins : ℕ
  visible_sum : ℕ

/-- Checks if a CoinState is valid according to the problem conditions -/
def is_valid_state (state : CoinState) : Prop :=
  state.total_coins = 14 ∧
  state.two_ruble_coins + state.five_ruble_coins = state.total_coins ∧
  state.two_ruble_coins > 0 ∧
  state.five_ruble_coins > 0 ∧
  state.visible_sum ≤ 2 * state.two_ruble_coins + 5 * state.five_ruble_coins

/-- Calculates the new visible sum after flipping all coins -/
def flipped_sum (state : CoinState) : ℕ :=
  2 * state.two_ruble_coins + 5 * state.five_ruble_coins - state.visible_sum

/-- The main theorem to prove -/
theorem coin_flip_theorem (state : CoinState) :
  is_valid_state state →
  flipped_sum state = 3 * state.visible_sum →
  state.five_ruble_coins = 4 ∨ state.five_ruble_coins = 8 ∨ state.five_ruble_coins = 12 := by
  sorry


end NUMINAMATH_CALUDE_coin_flip_theorem_l2496_249679


namespace NUMINAMATH_CALUDE_prob_monochromatic_triangle_l2496_249671

/-- A regular hexagon with colored edges -/
structure ColoredHexagon where
  /-- The probability of an edge being colored red -/
  p : ℝ
  /-- Assumption that p is between 0 and 1 -/
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The number of edges (sides and diagonals) in a regular hexagon -/
def num_edges : ℕ := 15

/-- The number of triangles in a regular hexagon -/
def num_triangles : ℕ := 20

/-- The probability of a specific triangle not being monochromatic -/
def prob_not_monochromatic (h : ColoredHexagon) : ℝ :=
  3 * h.p^2 * (1 - h.p) + 3 * (1 - h.p)^2 * h.p

/-- The main theorem: probability of at least one monochromatic triangle -/
theorem prob_monochromatic_triangle (h : ColoredHexagon) :
  h.p = 1/2 → 1 - (prob_not_monochromatic h)^num_triangles = 1 - (3/4)^20 := by
  sorry

end NUMINAMATH_CALUDE_prob_monochromatic_triangle_l2496_249671


namespace NUMINAMATH_CALUDE_closest_integer_to_sqrt17_minus_1_l2496_249601

theorem closest_integer_to_sqrt17_minus_1 : 
  ∃ (n : ℤ), ∀ (m : ℤ), |n - (Real.sqrt 17 - 1)| ≤ |m - (Real.sqrt 17 - 1)| ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_sqrt17_minus_1_l2496_249601


namespace NUMINAMATH_CALUDE_braking_distance_properties_l2496_249677

/-- Represents the braking distance in meters -/
def braking_distance (v : ℝ) : ℝ := 0.25 * v

/-- The maximum legal speed on highways in km/h -/
def max_legal_speed : ℝ := 120

theorem braking_distance_properties :
  (braking_distance 60 = 15) ∧
  (braking_distance 128 = 32) ∧
  (128 > max_legal_speed) := by
  sorry

end NUMINAMATH_CALUDE_braking_distance_properties_l2496_249677


namespace NUMINAMATH_CALUDE_different_color_sock_pairs_l2496_249600

theorem different_color_sock_pairs (white : ℕ) (brown : ℕ) (blue : ℕ) : 
  white = 5 → brown = 4 → blue = 3 → 
  (white * brown + brown * blue + white * blue = 47) :=
by
  sorry

end NUMINAMATH_CALUDE_different_color_sock_pairs_l2496_249600


namespace NUMINAMATH_CALUDE_max_even_distribution_l2496_249666

/-- Represents the pencil distribution problem --/
def PencilDistribution (initial_pencils : ℕ) (initial_containers : ℕ) (first_addition : ℕ) (second_addition : ℕ) (final_containers : ℕ) : Prop :=
  let total_pencils : ℕ := initial_pencils + first_addition + second_addition
  let even_distribution : ℕ := total_pencils / final_containers
  even_distribution * final_containers ≤ total_pencils ∧
  (even_distribution + 1) * final_containers > total_pencils

/-- Theorem stating the maximum even distribution of pencils --/
theorem max_even_distribution :
  PencilDistribution 150 5 30 47 6 →
  ∃ (n : ℕ), n = 37 ∧ PencilDistribution 150 5 30 47 6 := by
  sorry

#check max_even_distribution

end NUMINAMATH_CALUDE_max_even_distribution_l2496_249666


namespace NUMINAMATH_CALUDE_lantern_tower_top_count_l2496_249693

/-- Represents a tower with geometric progression of lanterns -/
structure LanternTower where
  levels : ℕ
  ratio : ℕ
  total : ℕ
  top : ℕ

/-- Calculates the sum of a geometric sequence -/
def geometricSum (a : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

/-- Theorem: In a 7-level tower where the number of lanterns doubles at each level
    from top to bottom, and the total number of lanterns is 381,
    the number of lanterns at the top level is 3. -/
theorem lantern_tower_top_count (tower : LanternTower)
    (h1 : tower.levels = 7)
    (h2 : tower.ratio = 2)
    (h3 : tower.total = 381)
    : tower.top = 3 := by
  sorry

#check lantern_tower_top_count

end NUMINAMATH_CALUDE_lantern_tower_top_count_l2496_249693


namespace NUMINAMATH_CALUDE_triangle_property_l2496_249618

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_property (t : Triangle) 
  (h1 : t.b * (1 + Real.cos t.C) = t.c * (2 - Real.cos t.B))
  (h2 : t.C = π / 3)
  (h3 : (1 / 2) * t.a * t.b * Real.sin t.C = 4 * Real.sqrt 3) :
  (t.a + t.b = 2 * t.c) ∧ (t.c = 4) := by
  sorry


end NUMINAMATH_CALUDE_triangle_property_l2496_249618


namespace NUMINAMATH_CALUDE_absent_laborers_count_l2496_249675

/-- Represents the number of laborers originally employed -/
def total_laborers : ℕ := 20

/-- Represents the original number of days planned to complete the work -/
def original_days : ℕ := 15

/-- Represents the actual number of days taken to complete the work -/
def actual_days : ℕ := 20

/-- Represents the total amount of work in laborer-days -/
def total_work : ℕ := total_laborers * original_days

/-- Calculates the number of absent laborers -/
def absent_laborers : ℕ := total_laborers - (total_work / actual_days)

theorem absent_laborers_count : absent_laborers = 5 := by
  sorry

end NUMINAMATH_CALUDE_absent_laborers_count_l2496_249675


namespace NUMINAMATH_CALUDE_discounted_price_calculation_l2496_249652

/-- The actual price of the good in Rupees -/
def actual_price : ℝ := 9502.923976608186

/-- The first discount rate -/
def discount1 : ℝ := 0.20

/-- The second discount rate -/
def discount2 : ℝ := 0.10

/-- The third discount rate -/
def discount3 : ℝ := 0.05

/-- The discounted price after applying three successive discounts -/
def discounted_price (p : ℝ) (d1 d2 d3 : ℝ) : ℝ :=
  p * (1 - d1) * (1 - d2) * (1 - d3)

/-- Theorem stating that the discounted price is approximately 6498.40 -/
theorem discounted_price_calculation :
  ∃ ε > 0, abs (discounted_price actual_price discount1 discount2 discount3 - 6498.40) < ε :=
sorry

end NUMINAMATH_CALUDE_discounted_price_calculation_l2496_249652


namespace NUMINAMATH_CALUDE_valve_emission_difference_l2496_249645

/-- The difference in water emission rates between two valves filling a pool -/
theorem valve_emission_difference (pool_capacity : ℝ) (both_valves_time : ℝ) (first_valve_time : ℝ) : 
  pool_capacity > 0 → 
  both_valves_time > 0 → 
  first_valve_time > 0 → 
  pool_capacity / both_valves_time - pool_capacity / first_valve_time = 50 := by
  sorry

#check valve_emission_difference 12000 48 120

end NUMINAMATH_CALUDE_valve_emission_difference_l2496_249645


namespace NUMINAMATH_CALUDE_circle_pencil_theorem_l2496_249674

/-- Definition of a circle in 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  R : ℝ

/-- Left-hand side of circle equation -/
def K (C : Circle) (x y : ℝ) : ℝ :=
  (x - C.a)^2 + (y - C.b)^2 - C.R^2

/-- Type of circle pencil -/
inductive PencilType
  | Elliptic
  | Parabolic
  | Hyperbolic

/-- Theorem about circle pencils -/
theorem circle_pencil_theorem (C₁ C₂ : Circle) :
  ∃ (radical_axis : Set (ℝ × ℝ)) (pencil_type : PencilType),
    (∀ (t : ℝ), ∃ (C : Circle), ∀ (x y : ℝ),
      K C₁ x y + t * K C₂ x y = 0 ↔ K C x y = 0) ∧
    (∀ (C : Circle), (∀ (x y : ℝ), K C x y = 0 → (x, y) ∈ radical_axis) →
      ∃ (t : ℝ), ∀ (x y : ℝ), K C₁ x y + t * K C₂ x y = 0 ↔ K C x y = 0) ∧
    (pencil_type = PencilType.Elliptic →
      ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ K C₁ p₁.1 p₁.2 = 0 ∧ K C₂ p₁.1 p₁.2 = 0 ∧
                          K C₁ p₂.1 p₂.2 = 0 ∧ K C₂ p₂.1 p₂.2 = 0) ∧
    (pencil_type = PencilType.Parabolic →
      ∃ (p : ℝ × ℝ), K C₁ p.1 p.2 = 0 ∧ K C₂ p.1 p.2 = 0 ∧
        ∀ (ε : ℝ), ε > 0 → ∃ (q : ℝ × ℝ), q ≠ p ∧ 
          abs (K C₁ q.1 q.2) < ε ∧ abs (K C₂ q.1 q.2) < ε) ∧
    (pencil_type = PencilType.Hyperbolic →
      ∀ (x y : ℝ), K C₁ x y = 0 → K C₂ x y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_pencil_theorem_l2496_249674


namespace NUMINAMATH_CALUDE_homework_reduction_equation_l2496_249689

theorem homework_reduction_equation 
  (initial_duration : ℝ) 
  (final_duration : ℝ) 
  (x : ℝ) 
  (h1 : initial_duration = 90) 
  (h2 : final_duration = 60) 
  (h3 : 0 ≤ x ∧ x < 1) : 
  initial_duration * (1 - x)^2 = final_duration := by
sorry

end NUMINAMATH_CALUDE_homework_reduction_equation_l2496_249689


namespace NUMINAMATH_CALUDE_number_operation_result_l2496_249667

theorem number_operation_result (x : ℝ) : (x - 5) / 7 = 7 → (x - 24) / 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_operation_result_l2496_249667


namespace NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_9_l2496_249629

/-- Reverses the digits of a positive integer -/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Checks if a number is a four-digit integer -/
def isFourDigit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

theorem greatest_four_digit_divisible_by_63_and_9 :
  ∃ (p : ℕ), 
    isFourDigit p ∧ 
    isFourDigit (reverseDigits p) ∧ 
    p % 63 = 0 ∧ 
    (reverseDigits p) % 63 = 0 ∧ 
    p % 9 = 0 ∧
    ∀ (x : ℕ), 
      isFourDigit x ∧ 
      isFourDigit (reverseDigits x) ∧ 
      x % 63 = 0 ∧ 
      (reverseDigits x) % 63 = 0 ∧ 
      x % 9 = 0 → 
      x ≤ p ∧
    p = 9507 := by
  sorry

end NUMINAMATH_CALUDE_greatest_four_digit_divisible_by_63_and_9_l2496_249629


namespace NUMINAMATH_CALUDE_abs_sum_minimum_l2496_249686

theorem abs_sum_minimum (x : ℝ) : 
  |x + 1| + |x + 3| + |x + 6| ≥ 5 ∧ ∃ y : ℝ, |y + 1| + |y + 3| + |y + 6| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_minimum_l2496_249686


namespace NUMINAMATH_CALUDE_expression_equals_zero_l2496_249655

theorem expression_equals_zero (x y z : ℝ) (h : x*y + y*z + z*x = 0) :
  3*x*y*z + x^2*(y + z) + y^2*(z + x) + z^2*(x + y) = 0 := by sorry

end NUMINAMATH_CALUDE_expression_equals_zero_l2496_249655


namespace NUMINAMATH_CALUDE_garden_length_l2496_249672

/-- Proves that a rectangular garden with length twice its width and perimeter 180 yards has a length of 60 yards -/
theorem garden_length (width : ℝ) (length : ℝ) : 
  length = 2 * width →  -- Length is twice the width
  2 * width + 2 * length = 180 →  -- Perimeter is 180 yards
  length = 60 := by
sorry


end NUMINAMATH_CALUDE_garden_length_l2496_249672


namespace NUMINAMATH_CALUDE_circle_equation_m_range_l2496_249681

/-- Given an equation x^2 + y^2 - 2x - 4y + m = 0 that represents a circle, prove that m < 5 -/
theorem circle_equation_m_range (m : ℝ) :
  (∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0 ↔ (x - 1)^2 + (y - 2)^2 = r^2) →
  m < 5 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_m_range_l2496_249681


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_7200_l2496_249603

theorem largest_divisor_of_n_squared_div_7200 (n : ℕ) (h1 : n > 0) (h2 : 7200 ∣ n^2) :
  (60 ∣ n) ∧ ∀ k : ℕ, k ∣ n → k ≤ 60 :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_squared_div_7200_l2496_249603


namespace NUMINAMATH_CALUDE_exists_special_function_l2496_249654

theorem exists_special_function :
  ∃ f : ℕ → ℕ, ∀ n : ℕ, f (f (n + 1)) = f (f n) + 2^(n - 1) := by
  sorry

end NUMINAMATH_CALUDE_exists_special_function_l2496_249654


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_one_two_l2496_249615

/-- The equation of a direct proportion function passing through (1, 2) -/
theorem direct_proportion_through_point_one_two :
  ∀ (k : ℝ), (∃ f : ℝ → ℝ, (∀ x, f x = k * x) ∧ f 1 = 2) → 
  (∀ x, k * x = 2 * x) :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_one_two_l2496_249615


namespace NUMINAMATH_CALUDE_third_train_speed_l2496_249641

/-- Calculates the speed of the third train given the conditions of the problem -/
theorem third_train_speed
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (third_train_length : ℝ)
  (goods_train_pass_time : ℝ)
  (third_train_pass_time : ℝ)
  (h_man_train_speed : man_train_speed = 45)
  (h_goods_train_length : goods_train_length = 340)
  (h_third_train_length : third_train_length = 480)
  (h_goods_train_pass_time : goods_train_pass_time = 8)
  (h_third_train_pass_time : third_train_pass_time = 12) :
  ∃ (third_train_speed : ℝ), third_train_speed = 99 := by
  sorry


end NUMINAMATH_CALUDE_third_train_speed_l2496_249641


namespace NUMINAMATH_CALUDE_pole_shortening_l2496_249692

/-- Given a pole of length 20 meters that is shortened by 30%, prove that its new length is 14 meters. -/
theorem pole_shortening (original_length : ℝ) (shortening_percentage : ℝ) (new_length : ℝ) :
  original_length = 20 →
  shortening_percentage = 30 →
  new_length = original_length * (1 - shortening_percentage / 100) →
  new_length = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_pole_shortening_l2496_249692


namespace NUMINAMATH_CALUDE_john_driving_equation_l2496_249678

def speed_before_lunch : ℝ := 60
def speed_after_lunch : ℝ := 90
def total_distance : ℝ := 300
def total_time : ℝ := 4
def lunch_break : ℝ := 0.5

theorem john_driving_equation (t : ℝ) : 
  speed_before_lunch * t + speed_after_lunch * (total_time - lunch_break - t) = total_distance :=
sorry

end NUMINAMATH_CALUDE_john_driving_equation_l2496_249678


namespace NUMINAMATH_CALUDE_math_competition_average_score_l2496_249673

theorem math_competition_average_score 
  (total_people : ℕ) 
  (group_average : ℚ) 
  (xiaoming_score : ℚ) 
  (h1 : total_people = 10)
  (h2 : group_average = 84)
  (h3 : xiaoming_score = 93) :
  let remaining_people := total_people - 1
  let total_score := group_average * total_people
  let remaining_score := total_score - xiaoming_score
  remaining_score / remaining_people = 83 := by
sorry

end NUMINAMATH_CALUDE_math_competition_average_score_l2496_249673


namespace NUMINAMATH_CALUDE_fraction_equality_l2496_249665

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) 
  (h3 : a/b + (a+5*b)/(b+5*a) = 2) : a/b = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2496_249665


namespace NUMINAMATH_CALUDE_fifth_grade_class_size_is_correct_l2496_249631

/-- Represents the number of students in each fifth grade class -/
def fifth_grade_class_size : ℕ := 27

/-- Represents the total number of third grade classes -/
def third_grade_classes : ℕ := 5

/-- Represents the number of students in each third grade class -/
def third_grade_class_size : ℕ := 30

/-- Represents the total number of fourth grade classes -/
def fourth_grade_classes : ℕ := 4

/-- Represents the number of students in each fourth grade class -/
def fourth_grade_class_size : ℕ := 28

/-- Represents the total number of fifth grade classes -/
def fifth_grade_classes : ℕ := 4

/-- Represents the cost of a hamburger in cents -/
def hamburger_cost : ℕ := 210

/-- Represents the cost of carrots in cents -/
def carrots_cost : ℕ := 50

/-- Represents the cost of a cookie in cents -/
def cookie_cost : ℕ := 20

/-- Represents the total cost of all students' lunches in cents -/
def total_lunch_cost : ℕ := 103600

theorem fifth_grade_class_size_is_correct : 
  fifth_grade_class_size * fifth_grade_classes * (hamburger_cost + carrots_cost + cookie_cost) + 
  third_grade_classes * third_grade_class_size * (hamburger_cost + carrots_cost + cookie_cost) + 
  fourth_grade_classes * fourth_grade_class_size * (hamburger_cost + carrots_cost + cookie_cost) = 
  total_lunch_cost :=
by sorry

end NUMINAMATH_CALUDE_fifth_grade_class_size_is_correct_l2496_249631


namespace NUMINAMATH_CALUDE_quadratic_factorization_l2496_249670

theorem quadratic_factorization (a : ℝ) : 
  (∃ m n : ℝ, ∀ x y : ℝ, 
    x^2 + 7*x*y + a*y^2 - 5*x - 45*y - 24 = (x - 8 + m*y) * (x + 3 + n*y)) → 
  a = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l2496_249670


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_average_increase_by_2_min_score_before_12th_consecutive_scores_before_12th_l2496_249623

/-- Represents a cricket batsman's performance -/
structure Batsman where
  innings : Nat
  totalRuns : Nat
  minScore : Nat
  consecutiveScores : Nat

/-- Calculates the average score of a batsman -/
def average (b : Batsman) : Rat :=
  b.totalRuns / b.innings

/-- Represents the batsman's performance after 11 innings -/
def initialBatsman : Batsman :=
  { innings := 11
  , totalRuns := 11 * 24  -- 11 * average before 12th innings
  , minScore := 20
  , consecutiveScores := 25 }

/-- Represents the batsman's performance after 12 innings -/
def finalBatsman : Batsman :=
  { innings := 12
  , totalRuns := initialBatsman.totalRuns + 48
  , minScore := 20
  , consecutiveScores := 25 }

theorem batsman_average_after_12th_innings :
  average finalBatsman = 26 := by
  sorry

theorem average_increase_by_2 :
  average finalBatsman - average initialBatsman = 2 := by
  sorry

theorem min_score_before_12th :
  initialBatsman.minScore ≥ 20 := by
  sorry

theorem consecutive_scores_before_12th :
  initialBatsman.consecutiveScores = 25 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_average_increase_by_2_min_score_before_12th_consecutive_scores_before_12th_l2496_249623


namespace NUMINAMATH_CALUDE_marked_price_calculation_jobber_pricing_l2496_249606

theorem marked_price_calculation (original_price : ℝ) (discount_percent : ℝ) 
  (gain_percent : ℝ) (final_discount_percent : ℝ) : ℝ :=
  let purchase_price := original_price * (1 - discount_percent / 100)
  let selling_price := purchase_price * (1 + gain_percent / 100)
  let marked_price := selling_price / (1 - final_discount_percent / 100)
  marked_price

theorem jobber_pricing : 
  marked_price_calculation 30 15 50 25 = 51 := by
  sorry

end NUMINAMATH_CALUDE_marked_price_calculation_jobber_pricing_l2496_249606


namespace NUMINAMATH_CALUDE_homothety_transforms_circles_l2496_249664

/-- Two circles in a plane -/
structure TangentCircles where
  S₁ : Set (ℝ × ℝ)
  S₂ : Set (ℝ × ℝ)
  O₁ : ℝ × ℝ
  O₂ : ℝ × ℝ
  r : ℝ
  R : ℝ
  K : ℝ × ℝ
  h_circle₁ : ∀ p ∈ S₁, dist p O₁ = r
  h_circle₂ : ∀ p ∈ S₂, dist p O₂ = R
  h_tangent : K ∈ S₁ ∧ K ∈ S₂
  h_external : dist O₁ O₂ = r + R

/-- Homothety transformation -/
def homothety (center : ℝ × ℝ) (k : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (center.1 + k * (p.1 - center.1), center.2 + k * (p.2 - center.2))

/-- Main theorem: Homothety transforms one circle into another -/
theorem homothety_transforms_circles (tc : TangentCircles) :
  ∃ h : Set (ℝ × ℝ) → Set (ℝ × ℝ),
    h tc.S₁ = tc.S₂ ∧
    ∀ p ∈ tc.S₁, h {p} = {homothety tc.K (tc.R / tc.r) p} :=
  sorry

end NUMINAMATH_CALUDE_homothety_transforms_circles_l2496_249664


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2496_249619

/-- Given that x varies inversely as the square of y, prove that x = 2.25 when y = 2,
    given that y = 3 when x = 1. -/
theorem inverse_variation_problem (x y : ℝ) (k : ℝ) : 
  (∀ (x y : ℝ), x = k / (y^2)) →  -- x varies inversely as the square of y
  (1 = k / (3^2)) →               -- y = 3 when x = 1
  (2.25 = k / (2^2))              -- x = 2.25 when y = 2
  := by sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2496_249619


namespace NUMINAMATH_CALUDE_volleyball_club_girls_l2496_249687

theorem volleyball_club_girls (total : ℕ) (present : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 36 →
  present = 24 →
  boys + girls = total →
  boys + (1/3 : ℚ) * girls = present →
  girls = 18 :=
by sorry

end NUMINAMATH_CALUDE_volleyball_club_girls_l2496_249687


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2496_249659

theorem not_p_sufficient_not_necessary_for_not_q :
  ∃ (x : ℝ), (¬(|x + 1| > 2) → ¬(5*x - 6 > x^2)) ∧
             ∃ (y : ℝ), ¬(5*y - 6 > y^2) ∧ (|y + 1| > 2) := by
  sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l2496_249659


namespace NUMINAMATH_CALUDE_minimize_plates_l2496_249622

/-- Represents the number of units of each product produced by a single plate of each type -/
def PlateProduction := Fin 2 → Fin 2 → ℕ

/-- The required production amounts for products A and B -/
def RequiredProduction := Fin 2 → ℕ

/-- The solution represented as the number of plates of each type used -/
def Solution := Fin 2 → ℕ

/-- Checks if a solution satisfies the production requirements -/
def satisfiesRequirements (plate_prod : PlateProduction) (req_prod : RequiredProduction) (sol : Solution) : Prop :=
  ∀ i, (sol 0 * plate_prod 0 i + sol 1 * plate_prod 1 i) = req_prod i

/-- Calculates the total number of plates used in a solution -/
def totalPlates (sol : Solution) : ℕ :=
  sol 0 + sol 1

theorem minimize_plates (plate_prod : PlateProduction) (req_prod : RequiredProduction) :
  let solution : Solution := ![6, 2]
  satisfiesRequirements plate_prod req_prod solution ∧
  (∀ other : Solution, satisfiesRequirements plate_prod req_prod other →
    totalPlates solution ≤ totalPlates other) :=
by
  sorry

end NUMINAMATH_CALUDE_minimize_plates_l2496_249622


namespace NUMINAMATH_CALUDE_min_value_complex_expression_l2496_249651

theorem min_value_complex_expression (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (min : ℝ), min = 0 ∧ ∀ w : ℂ, Complex.abs w = 2 → Complex.abs ((w - 2)^2 * (w + 2)) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_complex_expression_l2496_249651


namespace NUMINAMATH_CALUDE_volunteer_schedule_lcm_l2496_249626

theorem volunteer_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 (Nat.lcm 9 10))) = 360 := by
  sorry

end NUMINAMATH_CALUDE_volunteer_schedule_lcm_l2496_249626


namespace NUMINAMATH_CALUDE_tan_beta_calculation_l2496_249620

open Real

theorem tan_beta_calculation (α β : ℝ) (h1 : 0 < α ∧ α < π/2) (h2 : 0 < β ∧ β < π/2)
  (h3 : sin α = 4/5) (h4 : tan (α - β) = 2/3) : tan β = 6/17 := by
  sorry

end NUMINAMATH_CALUDE_tan_beta_calculation_l2496_249620


namespace NUMINAMATH_CALUDE_one_third_square_coloring_l2496_249690

theorem one_third_square_coloring (n : ℕ) (k : ℕ) : n = 18 ∧ k = 6 → Nat.choose n k = 18564 := by
  sorry

end NUMINAMATH_CALUDE_one_third_square_coloring_l2496_249690


namespace NUMINAMATH_CALUDE_sequence_equality_l2496_249639

theorem sequence_equality (a : Fin 100 → ℝ)
  (h1 : ∀ n : Fin 98, a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0)
  (h2 : a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0)
  (h3 : a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)
  (h4 : a 10 = 10) :
  a 22 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sequence_equality_l2496_249639
