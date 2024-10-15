import Mathlib

namespace NUMINAMATH_CALUDE_homework_problems_l3617_361712

theorem homework_problems (t p : ℕ) (ht : t > 0) (hp : p > 10) : 
  (∀ (t' : ℕ), t' > 0 → p * t = (2 * p - 2) * (t' - 1) → t' = t) →
  p * t = 48 := by
sorry

end NUMINAMATH_CALUDE_homework_problems_l3617_361712


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3617_361734

/-- The intersection points of a line and a circle -/
theorem line_circle_intersection :
  let line := { p : ℝ × ℝ | p.1 + p.2 = 1 }
  let circle := { p : ℝ × ℝ | p.1^2 + p.2^2 = 9 }
  let point1 := ((1 + Real.sqrt 17) / 2, (1 - Real.sqrt 17) / 2)
  let point2 := ((1 - Real.sqrt 17) / 2, (1 + Real.sqrt 17) / 2)
  (point1 ∈ line ∧ point1 ∈ circle) ∧ 
  (point2 ∈ line ∧ point2 ∈ circle) ∧
  (∀ p ∈ line ∩ circle, p = point1 ∨ p = point2) :=
by
  sorry


end NUMINAMATH_CALUDE_line_circle_intersection_l3617_361734


namespace NUMINAMATH_CALUDE_rectangle_area_proof_l3617_361727

/-- Given a rectangle EFGH with vertices E(0, 0), F(0, 5), G(y, 5), and H(y, 0),
    where y > 0 and the area of the rectangle is 45 square units,
    prove that y = 9. -/
theorem rectangle_area_proof (y : ℝ) : y > 0 → y * 5 = 45 → y = 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_proof_l3617_361727


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l3617_361775

theorem distance_from_origin_to_point (z : ℂ) : 
  z = 1260 + 1680 * Complex.I → Complex.abs z = 2100 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l3617_361775


namespace NUMINAMATH_CALUDE_opposite_of_2023_l3617_361778

theorem opposite_of_2023 : 
  ∃ y : ℤ, y + 2023 = 0 ∧ y = -2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_2023_l3617_361778


namespace NUMINAMATH_CALUDE_three_digit_number_property_l3617_361716

theorem three_digit_number_property (a b c : ℕ) : 
  a ≠ 0 → 
  a < 10 → b < 10 → c < 10 →
  10 * b + c = 8 * a →
  10 * a + b = 8 * c →
  (10 * a + c) / b = 17 :=
sorry

end NUMINAMATH_CALUDE_three_digit_number_property_l3617_361716


namespace NUMINAMATH_CALUDE_power_equality_l3617_361721

theorem power_equality (n : ℕ) : 9^4 = 3^n → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l3617_361721


namespace NUMINAMATH_CALUDE_average_weight_of_class_l3617_361794

/-- The average weight of a class with two sections -/
theorem average_weight_of_class 
  (studentsA : ℕ) (studentsB : ℕ) 
  (avgWeightA : ℚ) (avgWeightB : ℚ) :
  studentsA = 40 →
  studentsB = 30 →
  avgWeightA = 50 →
  avgWeightB = 60 →
  (studentsA * avgWeightA + studentsB * avgWeightB) / (studentsA + studentsB : ℚ) = 3800 / 70 := by
  sorry

#eval (3800 : ℚ) / 70

end NUMINAMATH_CALUDE_average_weight_of_class_l3617_361794


namespace NUMINAMATH_CALUDE_geometric_sequence_tan_value_l3617_361733

/-- Given a geometric sequence {a_n} where a₁a₁₃ + 2a₇² = 4π, prove that tan(a₂a₁₂) = √3 -/
theorem geometric_sequence_tan_value (a : ℕ → ℝ) 
  (h_geometric : ∀ n, a (n + 1) / a n = a 2 / a 1)  -- Geometric sequence condition
  (h_sum : a 1 * a 13 + 2 * (a 7)^2 = 4 * Real.pi)  -- Given equation
  : Real.tan (a 2 * a 12) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_tan_value_l3617_361733


namespace NUMINAMATH_CALUDE_min_value_is_three_l3617_361738

/-- A quadratic function f(x) = ax² + bx + c where b > a and f(x) ≥ 0 for all x ∈ ℝ -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : b > a
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0

/-- The minimum value of (a+b+c)/(b-a) for a QuadraticFunction is 3 -/
theorem min_value_is_three (f : QuadraticFunction) : 
  (∀ x : ℝ, (f.a + f.b + f.c) / (f.b - f.a) ≥ 3) ∧ 
  (∃ x : ℝ, (f.a + f.b + f.c) / (f.b - f.a) = 3) := by
  sorry

end NUMINAMATH_CALUDE_min_value_is_three_l3617_361738


namespace NUMINAMATH_CALUDE_symmetry_coordinates_l3617_361781

/-- Two points are symmetrical about the y-axis if their x-coordinates are negatives of each other
    and their y-coordinates are the same. -/
def symmetrical_about_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_coordinates :
  let p : ℝ × ℝ := (4, -5)
  let q : ℝ × ℝ := (a, b)
  symmetrical_about_y_axis p q → a = -4 ∧ b = -5 := by
sorry

end NUMINAMATH_CALUDE_symmetry_coordinates_l3617_361781


namespace NUMINAMATH_CALUDE_correct_calculation_l3617_361797

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * y * x^2 = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3617_361797


namespace NUMINAMATH_CALUDE_gift_cost_problem_l3617_361789

theorem gift_cost_problem (initial_friends : ℕ) (dropped_out : ℕ) (extra_cost : ℝ) :
  initial_friends = 10 →
  dropped_out = 4 →
  extra_cost = 8 →
  ∃ (total_cost : ℝ),
    total_cost / (initial_friends - dropped_out : ℝ) = total_cost / initial_friends + extra_cost ∧
    total_cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_problem_l3617_361789


namespace NUMINAMATH_CALUDE_greatest_power_of_two_l3617_361750

theorem greatest_power_of_two (n : ℕ) : 
  (∃ k : ℕ, (10^1004 - 4^502) = k * 2^1007) ∧ 
  (∀ m : ℕ, m > 1007 → ¬(∃ k : ℕ, (10^1004 - 4^502) = k * 2^m)) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_l3617_361750


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l3617_361768

def geometric_series (n : ℕ) : ℚ :=
  match n with
  | 0 => 5 / 3
  | 1 => 30 / 7
  | 2 => 180 / 49
  | _ => 0  -- We only define the first three terms explicitly

theorem common_ratio_of_geometric_series :
  ∃ r : ℚ, ∀ n : ℕ, n > 0 → geometric_series (n + 1) = r * geometric_series n :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_series_l3617_361768


namespace NUMINAMATH_CALUDE_megan_markers_count_l3617_361726

/-- The number of markers Megan has after receiving and giving away some -/
def final_markers (initial : ℕ) (received : ℕ) (given_away : ℕ) : ℕ :=
  initial + received - given_away

/-- Theorem stating that Megan's final number of markers is correct -/
theorem megan_markers_count :
  final_markers 217 109 35 = 291 :=
by sorry

end NUMINAMATH_CALUDE_megan_markers_count_l3617_361726


namespace NUMINAMATH_CALUDE_subset_condition_empty_intersection_condition_l3617_361703

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 3}

-- Theorem for the subset condition
theorem subset_condition (a : ℝ) : 
  A ⊆ B a ↔ a ∈ Set.Icc (-1/2) 0 :=
sorry

-- Theorem for the empty intersection condition
theorem empty_intersection_condition (a : ℝ) :
  A ∩ B a = ∅ ↔ a ∈ Set.Iic (-2) ∪ Set.Ici (3/2) :=
sorry

end NUMINAMATH_CALUDE_subset_condition_empty_intersection_condition_l3617_361703


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l3617_361739

-- Define the sets M and N
def M : Set ℝ := {x | |x| ≤ 3}
def N : Set ℝ := {x | x < 2}

-- State the theorem
theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x | x < -3} := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l3617_361739


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3617_361782

theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ q : ℝ, q > 0 ∧ ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : 6 * a 1 + 4 * a 2 = 2 * a 3) :
  (a 11 + a 13 + a 16 + a 20 + a 21) / (a 8 + a 10 + a 13 + a 17 + a 18) = 27 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3617_361782


namespace NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_k_l3617_361728

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| - |x + 3|

-- Theorem for the first part of the problem
theorem solution_set_f_greater_than_two :
  {x : ℝ | f x > 2} = {x : ℝ | x < -2} := by sorry

-- Theorem for the second part of the problem
theorem range_of_k (k : ℝ) :
  (∀ x ∈ Set.Icc (-3) (-1), f x ≤ k * x + 1) ↔ k ≤ -1 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_greater_than_two_range_of_k_l3617_361728


namespace NUMINAMATH_CALUDE_probability_bounds_l3617_361731

theorem probability_bounds (n : ℕ) (m₀ : ℕ) (p : ℝ) 
  (h_n : n = 120) 
  (h_m₀ : m₀ = 32) 
  (h_most_probable : m₀ = ⌊n * p + 0.5⌋) : 
  32 / 121 ≤ p ∧ p ≤ 33 / 121 := by
  sorry

end NUMINAMATH_CALUDE_probability_bounds_l3617_361731


namespace NUMINAMATH_CALUDE_quadratic_equation_set_equivalence_l3617_361787

theorem quadratic_equation_set_equivalence :
  {x : ℝ | x^2 - 3*x + 2 = 0} = {1, 2} := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_set_equivalence_l3617_361787


namespace NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solution_l3617_361702

-- Equation 1
theorem equation_one_solutions (x : ℝ) :
  9 * x^2 - (x - 1)^2 = 0 ↔ x = -0.5 ∨ x = 0.25 := by sorry

-- Equation 2
theorem equation_two_solutions (x : ℝ) :
  x * (x - 3) = 10 ↔ x = 5 ∨ x = -2 := by sorry

-- Equation 3
theorem equation_three_solution (x : ℝ) :
  (x + 3)^2 = 2 * x + 5 ↔ x = -2 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_equation_two_solutions_equation_three_solution_l3617_361702


namespace NUMINAMATH_CALUDE_nine_in_M_ten_not_in_M_l3617_361753

/-- The set M of integers that can be expressed as the difference of two squares of integers -/
def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

/-- 9 belongs to the set M -/
theorem nine_in_M : (9 : ℤ) ∈ M := by sorry

/-- 10 does not belong to the set M -/
theorem ten_not_in_M : (10 : ℤ) ∉ M := by sorry

end NUMINAMATH_CALUDE_nine_in_M_ten_not_in_M_l3617_361753


namespace NUMINAMATH_CALUDE_divisibility_condition_l3617_361730

theorem divisibility_condition (x y : ℕ+) :
  (∃ (k : ℕ+), k * (2 * x + 7 * y) = 7 * x + 2 * y) ↔
  (∃ (a : ℕ+), (x = a ∧ y = a) ∨ (x = 4 * a ∧ y = a) ∨ (x = 19 * a ∧ y = a)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_condition_l3617_361730


namespace NUMINAMATH_CALUDE_dress_discount_percentage_l3617_361767

/-- Calculates the final discount percentage for a dress purchase with multiple discounts -/
theorem dress_discount_percentage (original_price : ℝ) (store_discount : ℝ) (member_discount : ℝ) :
  original_price = 350 →
  store_discount = 0.20 →
  member_discount = 0.10 →
  let price_after_store_discount := original_price * (1 - store_discount)
  let final_price := price_after_store_discount * (1 - member_discount)
  let total_discount := original_price - final_price
  let final_discount_percentage := (total_discount / original_price) * 100
  ∃ ε > 0, |final_discount_percentage - 28| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_dress_discount_percentage_l3617_361767


namespace NUMINAMATH_CALUDE_crackers_distribution_l3617_361774

/-- The number of crackers Matthew had initially -/
def total_crackers : ℕ := 32

/-- The number of friends Matthew gave crackers to -/
def num_friends : ℕ := 4

/-- The number of crackers each friend received -/
def crackers_per_friend : ℕ := total_crackers / num_friends

theorem crackers_distribution :
  crackers_per_friend = 8 := by sorry

end NUMINAMATH_CALUDE_crackers_distribution_l3617_361774


namespace NUMINAMATH_CALUDE_no_consecutive_integers_without_real_solutions_l3617_361725

theorem no_consecutive_integers_without_real_solutions :
  ¬ ∃ (b c : ℕ), 
    c = b + 1 ∧ 
    b > 0 ∧
    b^2 < 4*c ∧
    c^2 < 4*b :=
by sorry

end NUMINAMATH_CALUDE_no_consecutive_integers_without_real_solutions_l3617_361725


namespace NUMINAMATH_CALUDE_place_face_value_difference_l3617_361748

def number : ℕ := 856973

def digit_of_interest : ℕ := 7

def place_value (n : ℕ) (d : ℕ) : ℕ :=
  if n / 100 % 10 = d then d * 10 else 0

def face_value (d : ℕ) : ℕ := d

theorem place_face_value_difference :
  place_value number digit_of_interest - face_value digit_of_interest = 63 := by
  sorry

end NUMINAMATH_CALUDE_place_face_value_difference_l3617_361748


namespace NUMINAMATH_CALUDE_cantaloupe_price_l3617_361791

/-- Represents the problem of finding the price of cantaloupes --/
def CantalouperPriceProblem (C : ℚ) : Prop :=
  let initial_cantaloupes : ℕ := 30
  let initial_honeydews : ℕ := 27
  let dropped_cantaloupes : ℕ := 2
  let rotten_honeydews : ℕ := 3
  let final_cantaloupes : ℕ := 8
  let final_honeydews : ℕ := 9
  let honeydew_price : ℚ := 3
  let total_revenue : ℚ := 85
  let sold_cantaloupes : ℕ := initial_cantaloupes - final_cantaloupes - dropped_cantaloupes
  let sold_honeydews : ℕ := initial_honeydews - final_honeydews - rotten_honeydews
  C * sold_cantaloupes + honeydew_price * sold_honeydews = total_revenue

/-- Theorem stating that the price of each cantaloupe is $2 --/
theorem cantaloupe_price : ∃ C : ℚ, CantalouperPriceProblem C ∧ C = 2 := by
  sorry

end NUMINAMATH_CALUDE_cantaloupe_price_l3617_361791


namespace NUMINAMATH_CALUDE_bus_seating_capacity_l3617_361784

theorem bus_seating_capacity : 
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let people_per_seat : ℕ := 3
  let back_seat_capacity : ℕ := 11
  
  left_seats * people_per_seat + right_seats * people_per_seat + back_seat_capacity = 92 :=
by sorry

end NUMINAMATH_CALUDE_bus_seating_capacity_l3617_361784


namespace NUMINAMATH_CALUDE_range_of_a_l3617_361760

theorem range_of_a (a : ℝ) :
  (∃ x₀ : ℝ, -1 < x₀ ∧ x₀ < 1 ∧ 2 * a * x₀ - a + 3 = 0) →
  (a < -3 ∨ a > 1) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3617_361760


namespace NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l3617_361795

theorem max_of_min_is_sqrt_two (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) ≤ Real.sqrt 2 ∧
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (⨅ z ∈ ({x, 1/y, y + 1/x} : Set ℝ), z) = Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_of_min_is_sqrt_two_l3617_361795


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3617_361799

def B : Matrix (Fin 2) (Fin 2) ℝ := !![4, 1; 0, 3]

theorem B_power_15_minus_3_power_14 : 
  B^15 - 3 • B^14 = !![4^14, 4^14; 0, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l3617_361799


namespace NUMINAMATH_CALUDE_functional_equation_2013_l3617_361766

/-- Given a function f: ℝ → ℝ satisfying f(x-y) = f(x) + f(y) - 2xy for all real x and y,
    prove that f(2013) = 4052169 -/
theorem functional_equation_2013 (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x - y) = f x + f y - 2 * x * y) : 
    f 2013 = 4052169 := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_2013_l3617_361766


namespace NUMINAMATH_CALUDE_max_min_S_l3617_361704

theorem max_min_S (x y z : ℚ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (eq1 : 3 * x + 2 * y + z = 5)
  (eq2 : x + y - z = 2)
  (S : ℚ := 2 * x + y - z) :
  (∀ s : ℚ, S ≤ s → s ≤ 3) ∧ (∀ s : ℚ, 2 ≤ s → s ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_max_min_S_l3617_361704


namespace NUMINAMATH_CALUDE_cost_of_potatoes_l3617_361790

/-- Proves that the cost of each bag of potatoes is $6 -/
theorem cost_of_potatoes (chicken_price : ℝ) (celery_price : ℝ) (total_cost : ℝ) :
  chicken_price = 3 →
  celery_price = 2 →
  total_cost = 35 →
  (5 * chicken_price + 4 * celery_price + 2 * ((total_cost - 5 * chicken_price - 4 * celery_price) / 2)) = total_cost →
  (total_cost - 5 * chicken_price - 4 * celery_price) / 2 = 6 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_potatoes_l3617_361790


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3617_361759

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) :
  a 4 + a 8 = 16 → a 2 + a 10 = 16 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3617_361759


namespace NUMINAMATH_CALUDE_sphere_volume_l3617_361711

theorem sphere_volume (r : ℝ) (h : 4 * π * r^2 = 2 * Real.sqrt 3 * π * (2 * r)) :
  (4 / 3) * π * r^3 = 4 * Real.sqrt 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_l3617_361711


namespace NUMINAMATH_CALUDE_base_prime_rep_132_l3617_361751

def base_prime_representation (n : ℕ) : List ℕ :=
  sorry

theorem base_prime_rep_132 :
  base_prime_representation 132 = [2, 1, 0, 1] :=
by
  sorry

end NUMINAMATH_CALUDE_base_prime_rep_132_l3617_361751


namespace NUMINAMATH_CALUDE_equation_represents_statement_l3617_361707

/-- Represents an unknown number -/
def n : ℤ := sorry

/-- The statement "a number increased by five equals 15" -/
def statement : Prop := n + 5 = 15

/-- Theorem stating that the equation correctly represents the given statement -/
theorem equation_represents_statement : statement ↔ n + 5 = 15 := by sorry

end NUMINAMATH_CALUDE_equation_represents_statement_l3617_361707


namespace NUMINAMATH_CALUDE_association_and_likelihood_ratio_l3617_361736

-- Define the contingency table
def excellent_math_excellent_chinese : ℕ := 45
def excellent_math_not_excellent_chinese : ℕ := 35
def not_excellent_math_excellent_chinese : ℕ := 45
def not_excellent_math_not_excellent_chinese : ℕ := 75

def total_sample_size : ℕ := 200

-- Define the chi-square test statistic
def chi_square_statistic : ℚ :=
  (total_sample_size * (excellent_math_excellent_chinese * not_excellent_math_not_excellent_chinese - 
  excellent_math_not_excellent_chinese * not_excellent_math_excellent_chinese)^2) / 
  ((excellent_math_excellent_chinese + excellent_math_not_excellent_chinese) * 
  (not_excellent_math_excellent_chinese + not_excellent_math_not_excellent_chinese) * 
  (excellent_math_excellent_chinese + not_excellent_math_excellent_chinese) * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese))

-- Define the critical value at α = 0.01
def critical_value : ℚ := 6635 / 1000

-- Define the likelihood ratio L(B|A)
def likelihood_ratio : ℚ := 
  (not_excellent_math_not_excellent_chinese * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese)) / 
  (excellent_math_not_excellent_chinese * 
  (excellent_math_not_excellent_chinese + not_excellent_math_not_excellent_chinese))

theorem association_and_likelihood_ratio : 
  chi_square_statistic > critical_value ∧ likelihood_ratio = 15 / 7 := by sorry

end NUMINAMATH_CALUDE_association_and_likelihood_ratio_l3617_361736


namespace NUMINAMATH_CALUDE_zucchini_amount_l3617_361785

def eggplant_pounds : ℝ := 5
def eggplant_price : ℝ := 2
def tomato_pounds : ℝ := 4
def tomato_price : ℝ := 3.5
def onion_pounds : ℝ := 3
def onion_price : ℝ := 1
def basil_pounds : ℝ := 1
def basil_price : ℝ := 2.5
def zucchini_price : ℝ := 2
def quarts_yield : ℝ := 4
def quart_price : ℝ := 10

theorem zucchini_amount (zucchini_pounds : ℝ) :
  eggplant_pounds * eggplant_price +
  zucchini_pounds * zucchini_price +
  tomato_pounds * tomato_price +
  onion_pounds * onion_price +
  basil_pounds * basil_price * 2 =
  quarts_yield * quart_price →
  zucchini_pounds = 4 := by sorry

end NUMINAMATH_CALUDE_zucchini_amount_l3617_361785


namespace NUMINAMATH_CALUDE_sector_area_sixty_degrees_radius_six_l3617_361763

/-- The area of a circular sector with central angle π/3 and radius 6 is 6π -/
theorem sector_area_sixty_degrees_radius_six : 
  let r : ℝ := 6
  let α : ℝ := π / 3
  let sector_area := (1 / 2) * r^2 * α
  sector_area = 6 * π := by sorry

end NUMINAMATH_CALUDE_sector_area_sixty_degrees_radius_six_l3617_361763


namespace NUMINAMATH_CALUDE_ted_age_l3617_361780

/-- Given that Ted's age is 20 years less than three times Sally's age,
    and the sum of their ages is 70, prove that Ted is 47.5 years old. -/
theorem ted_age (sally_age : ℝ) (ted_age : ℝ) 
  (h1 : ted_age = 3 * sally_age - 20)
  (h2 : ted_age + sally_age = 70) : 
  ted_age = 47.5 := by
  sorry

end NUMINAMATH_CALUDE_ted_age_l3617_361780


namespace NUMINAMATH_CALUDE_ac_length_l3617_361776

/-- Triangle ABC with specific properties -/
structure SpecialTriangle where
  -- Points A, B, C
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  -- Length conditions
  ab_length : dist A B = 7
  bc_length : dist B C = 24
  -- Area condition
  area : abs ((A.1 - C.1) * (B.2 - A.2) - (A.2 - C.2) * (B.1 - A.1)) / 2 = 84
  -- Median condition
  median_length : dist A ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = 12.5

/-- Theorem about the length of AC in the special triangle -/
theorem ac_length (t : SpecialTriangle) : dist t.A t.C = 25 := by
  sorry

end NUMINAMATH_CALUDE_ac_length_l3617_361776


namespace NUMINAMATH_CALUDE_area_of_rectangle_S_l3617_361741

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents a square with side length -/
structure Square where
  side : ℝ

/-- The configuration of shapes within the larger square -/
structure Configuration where
  largerSquare : Square
  rectangle : Rectangle
  smallerSquare : Square
  rectangleS : Rectangle

/-- The conditions of the problem -/
def validConfiguration (c : Configuration) : Prop :=
  c.rectangle.width = 2 ∧
  c.rectangle.height = 4 ∧
  c.smallerSquare.side = 2 ∧
  c.largerSquare.side ≥ 4 ∧
  c.largerSquare.side ^ 2 = 
    c.rectangle.width * c.rectangle.height +
    c.smallerSquare.side ^ 2 +
    c.rectangleS.width * c.rectangleS.height

theorem area_of_rectangle_S (c : Configuration) 
  (h : validConfiguration c) : 
  c.rectangleS.width * c.rectangleS.height = 4 :=
sorry

end NUMINAMATH_CALUDE_area_of_rectangle_S_l3617_361741


namespace NUMINAMATH_CALUDE_complex_division_equality_l3617_361710

theorem complex_division_equality : Complex.I = (3 + 2 * Complex.I) / (2 - 3 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_complex_division_equality_l3617_361710


namespace NUMINAMATH_CALUDE_min_obtuse_triangles_2003gon_l3617_361793

/-- A polygon inscribed in a circle -/
structure InscribedPolygon where
  n : ℕ
  n_ge_3 : n ≥ 3

/-- A triangulation of a polygon -/
structure Triangulation (P : InscribedPolygon) where
  triangle_count : ℕ
  triangle_count_eq : triangle_count = P.n - 2
  obtuse_count : ℕ
  acute_count : ℕ
  right_count : ℕ
  total_count : obtuse_count + acute_count + right_count = triangle_count
  max_non_obtuse : acute_count + right_count ≤ 2

/-- The theorem statement -/
theorem min_obtuse_triangles_2003gon :
  let P : InscribedPolygon := ⟨2003, by norm_num⟩
  ∀ T : Triangulation P, T.obtuse_count ≥ 1999 :=
by sorry

end NUMINAMATH_CALUDE_min_obtuse_triangles_2003gon_l3617_361793


namespace NUMINAMATH_CALUDE_intersection_nonempty_condition_l3617_361737

theorem intersection_nonempty_condition (m n : ℝ) :
  let A : Set ℝ := {x | m - 1 < x ∧ x < m + 1}
  let B : Set ℝ := {x | 3 - n < x ∧ x < 4 - n}
  (∃ x, x ∈ A ∩ B) ↔ (2 < m + n ∧ m + n < 5) := by sorry

end NUMINAMATH_CALUDE_intersection_nonempty_condition_l3617_361737


namespace NUMINAMATH_CALUDE_circle_area_difference_l3617_361719

theorem circle_area_difference (r₁ r₂ r : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) :
  π * r₂^2 - π * r₁^2 = π * r^2 → r = 20 :=
by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3617_361719


namespace NUMINAMATH_CALUDE_town_neighborhoods_count_l3617_361706

/-- Represents a town with neighborhoods and street lights -/
structure Town where
  total_lights : ℕ
  lights_per_road : ℕ
  roads_per_neighborhood : ℕ

/-- Calculates the number of neighborhoods in the town -/
def number_of_neighborhoods (t : Town) : ℕ :=
  (t.total_lights / t.lights_per_road) / t.roads_per_neighborhood

/-- Theorem: The number of neighborhoods in the given town is 10 -/
theorem town_neighborhoods_count :
  let t : Town := {
    total_lights := 20000,
    lights_per_road := 500,
    roads_per_neighborhood := 4
  }
  number_of_neighborhoods t = 10 := by
  sorry

end NUMINAMATH_CALUDE_town_neighborhoods_count_l3617_361706


namespace NUMINAMATH_CALUDE_max_pieces_in_5x5_grid_l3617_361746

theorem max_pieces_in_5x5_grid : ∀ (n : ℕ),
  (∃ (areas : List ℕ), 
    areas.length = n ∧ 
    areas.sum = 25 ∧ 
    areas.Nodup ∧ 
    (∀ a ∈ areas, a > 0)) →
  n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_pieces_in_5x5_grid_l3617_361746


namespace NUMINAMATH_CALUDE_two_word_sentences_count_correct_count_l3617_361729

def word : String := "YARIŞMA"

theorem two_word_sentences_count : ℕ :=
  let n : ℕ := word.length
  let repeated_letter_count : ℕ := 2  -- 'A' appears twice
  let permutations : ℕ := n.factorial / repeated_letter_count.factorial
  let space_positions : ℕ := n + 1
  permutations * space_positions

theorem correct_count : two_word_sentences_count = 20160 := by
  sorry

end NUMINAMATH_CALUDE_two_word_sentences_count_correct_count_l3617_361729


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achievable_l3617_361761

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 :=
by sorry

theorem min_value_reciprocal_sum_achievable :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 3 * b = 1 ∧ 1 / a + 3 / b = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_min_value_reciprocal_sum_achievable_l3617_361761


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l3617_361796

theorem absolute_value_simplification : |(-5^2 + 6 * 2)| = 13 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l3617_361796


namespace NUMINAMATH_CALUDE_eugene_payment_l3617_361745

/-- The cost of a single T-shirt in dollars -/
def tshirt_cost : ℚ := 20

/-- The cost of a single pair of pants in dollars -/
def pants_cost : ℚ := 80

/-- The cost of a single pair of shoes in dollars -/
def shoes_cost : ℚ := 150

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 0.1

/-- The number of T-shirts Eugene buys -/
def num_tshirts : ℕ := 4

/-- The number of pairs of pants Eugene buys -/
def num_pants : ℕ := 3

/-- The number of pairs of shoes Eugene buys -/
def num_shoes : ℕ := 2

/-- The total cost before discount -/
def total_cost_before_discount : ℚ :=
  tshirt_cost * num_tshirts + pants_cost * num_pants + shoes_cost * num_shoes

/-- The amount Eugene has to pay after the discount -/
def amount_to_pay : ℚ := total_cost_before_discount * (1 - discount_rate)

theorem eugene_payment :
  amount_to_pay = 558 := by sorry

end NUMINAMATH_CALUDE_eugene_payment_l3617_361745


namespace NUMINAMATH_CALUDE_cubic_expression_value_l3617_361783

theorem cubic_expression_value (x : ℝ) (h : x^2 + 3*x - 1 = 0) : 
  x^3 + 5*x^2 + 5*x + 18 = 20 := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l3617_361783


namespace NUMINAMATH_CALUDE_equal_parts_in_one_to_one_mix_l3617_361742

/-- Represents a substrate composition with parts of bark, peat, and sand -/
structure Substrate :=
  (bark : ℚ)
  (peat : ℚ)
  (sand : ℚ)

/-- Orchid-1 substrate composition -/
def orchid1 : Substrate :=
  { bark := 3
    peat := 2
    sand := 1 }

/-- Orchid-2 substrate composition -/
def orchid2 : Substrate :=
  { bark := 1
    peat := 2
    sand := 3 }

/-- Mixes two substrates in given proportions -/
def mixSubstrates (s1 s2 : Substrate) (r1 r2 : ℚ) : Substrate :=
  { bark := r1 * s1.bark + r2 * s2.bark
    peat := r1 * s1.peat + r2 * s2.peat
    sand := r1 * s1.sand + r2 * s2.sand }

/-- Checks if all components of a substrate are equal -/
def hasEqualParts (s : Substrate) : Prop :=
  s.bark = s.peat ∧ s.peat = s.sand

theorem equal_parts_in_one_to_one_mix :
  hasEqualParts (mixSubstrates orchid1 orchid2 1 1) :=
by sorry


end NUMINAMATH_CALUDE_equal_parts_in_one_to_one_mix_l3617_361742


namespace NUMINAMATH_CALUDE_abs_neg_eight_eq_eight_l3617_361788

theorem abs_neg_eight_eq_eight :
  abs (-8 : ℤ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_eight_eq_eight_l3617_361788


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l3617_361717

/-- Quadratic equation parameters -/
structure QuadraticParams where
  m : ℝ

/-- Roots of the quadratic equation -/
structure QuadraticRoots where
  x₁ : ℝ
  x₂ : ℝ

/-- Main theorem about the quadratic equation x^2 + mx + m - 2 = 0 -/
theorem quadratic_equation_properties (p : QuadraticParams) :
  -- If -2 is one root, the other root is 0
  (∃ (r : QuadraticRoots), r.x₁ = -2 ∧ r.x₂ = 0 ∧ 
    r.x₁^2 + p.m * r.x₁ + p.m - 2 = 0 ∧ 
    r.x₂^2 + p.m * r.x₂ + p.m - 2 = 0) ∧
  -- The equation always has two distinct real roots
  (∀ (x : ℝ), x^2 + p.m * x + p.m - 2 = 0 → 
    ∃ (r : QuadraticRoots), r.x₁ ≠ r.x₂ ∧ 
    r.x₁^2 + p.m * r.x₁ + p.m - 2 = 0 ∧ 
    r.x₂^2 + p.m * r.x₂ + p.m - 2 = 0) ∧
  -- If x₁^2 + x₂^2 + m(x₁ + x₂) = m^2 + 1, then m = -3 or m = 1
  (∀ (r : QuadraticRoots), 
    r.x₁^2 + r.x₂^2 + p.m * (r.x₁ + r.x₂) = p.m^2 + 1 →
    p.m = -3 ∨ p.m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l3617_361717


namespace NUMINAMATH_CALUDE_sqrt_sum_equality_l3617_361786

theorem sqrt_sum_equality : Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equality_l3617_361786


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3617_361764

theorem sqrt_equation_solution : 
  let x : ℝ := 12/5
  (Real.sqrt (6*x)) / (Real.sqrt (4*(x-2))) = 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3617_361764


namespace NUMINAMATH_CALUDE_exponent_multiplication_l3617_361714

theorem exponent_multiplication (a : ℝ) : a * a^3 = a^4 := by sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l3617_361714


namespace NUMINAMATH_CALUDE_three_possible_medians_l3617_361732

-- Define the set of game scores
def gameScores (x y : ℤ) : Finset ℤ := {x, 11, 13, y, 12}

-- Define the median of a set of integers
def median (s : Finset ℤ) : ℤ := sorry

-- Theorem statement
theorem three_possible_medians :
  ∃ (m₁ m₂ m₃ : ℤ), ∀ (x y : ℤ),
    (∃ (m : ℤ), median (gameScores x y) = m) →
    (m = m₁ ∨ m = m₂ ∨ m = m₃) ∧
    (m₁ ≠ m₂ ∧ m₁ ≠ m₃ ∧ m₂ ≠ m₃) :=
  sorry

#check three_possible_medians

end NUMINAMATH_CALUDE_three_possible_medians_l3617_361732


namespace NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l3617_361777

-- Define the given line
def given_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Define the point that the new line passes through
def point : ℝ × ℝ := (-1, 3)

-- Define the equation of the new line
def new_line (x y : ℝ) : Prop := x - 2 * y + 7 = 0

-- Theorem statement
theorem line_through_point_parallel_to_given : 
  (∀ (x y : ℝ), new_line x y ↔ ∃ (k : ℝ), x - point.1 = k * 1 ∧ y - point.2 = k * (-1/2)) ∧
  (∀ (x₁ y₁ x₂ y₂ : ℝ), given_line x₁ y₁ ∧ given_line x₂ y₂ → 
    (x₂ - x₁) * (-1/2) = (y₂ - y₁) * 1) ∧
  new_line point.1 point.2 :=
sorry

end NUMINAMATH_CALUDE_line_through_point_parallel_to_given_l3617_361777


namespace NUMINAMATH_CALUDE_radio_loss_percentage_l3617_361762

/-- Calculates the loss percentage given the cost price and selling price. -/
def loss_percentage (cost_price selling_price : ℚ) : ℚ :=
  (cost_price - selling_price) / cost_price * 100

/-- Proves that the loss percentage for a radio with cost price 2400 and selling price 2100 is 12.5%. -/
theorem radio_loss_percentage :
  let cost_price : ℚ := 2400
  let selling_price : ℚ := 2100
  loss_percentage cost_price selling_price = 25/2 := by
  sorry

end NUMINAMATH_CALUDE_radio_loss_percentage_l3617_361762


namespace NUMINAMATH_CALUDE_kim_no_tests_probability_l3617_361771

theorem kim_no_tests_probability 
  (p_math : ℝ) 
  (p_history : ℝ) 
  (h_math : p_math = 5/8) 
  (h_history : p_history = 1/3) 
  (h_independent : True)  -- Represents the independence of events
  : 1 - p_math - p_history + p_math * p_history = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_kim_no_tests_probability_l3617_361771


namespace NUMINAMATH_CALUDE_wizard_concoction_combinations_l3617_361722

/-- Represents the number of herbs available --/
def num_herbs : ℕ := 4

/-- Represents the number of crystals available --/
def num_crystals : ℕ := 6

/-- Represents the number of incompatible combinations --/
def num_incompatible : ℕ := 3

/-- Theorem stating the number of valid combinations for the wizard's concoction --/
theorem wizard_concoction_combinations : 
  num_herbs * num_crystals - num_incompatible = 21 := by
  sorry

end NUMINAMATH_CALUDE_wizard_concoction_combinations_l3617_361722


namespace NUMINAMATH_CALUDE_complementary_angles_can_be_equal_l3617_361772

-- Define what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- State the theorem
theorem complementary_angles_can_be_equal :
  ∃ (α : ℝ), complementary α α :=
sorry

-- The existence of such an angle pair disproves the statement
-- "Two complementary angles are not equal"

end NUMINAMATH_CALUDE_complementary_angles_can_be_equal_l3617_361772


namespace NUMINAMATH_CALUDE_part_one_part_two_l3617_361701

-- Define the propositions p and q
def p (a x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0

def q (x : ℝ) : Prop := abs (x - 1) ≤ 2 ∧ (x + 3) / (x - 2) ≥ 0

-- Part 1
theorem part_one : 
  ∀ x : ℝ, p 1 x ∧ q x → 2 < x ∧ x < 3 :=
sorry

-- Part 2
theorem part_two :
  (∀ x : ℝ, q x → (∀ a : ℝ, a > 0 → p a x)) ∧ 
  (∃ x a : ℝ, a > 0 ∧ p a x ∧ ¬q x) → 
  ∀ a : ℝ, 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3617_361701


namespace NUMINAMATH_CALUDE_average_shift_l3617_361749

theorem average_shift (a b c : ℝ) : 
  (a + b + c) / 3 = 5 → ((a - 2) + (b - 2) + (c - 2)) / 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_average_shift_l3617_361749


namespace NUMINAMATH_CALUDE_f_monotonicity_f_monotonic_increasing_iff_f_monotonic_decreasing_increasing_iff_l3617_361718

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - a * x - 1

theorem f_monotonicity (a : ℝ) :
  (a > 0 → ∀ x y, x > Real.log a → y > Real.log a → x < y → f a x < f a y) ∧
  (a ≤ 0 → ∀ x y, x < y → f a x < f a y) :=
sorry

theorem f_monotonic_increasing_iff (a : ℝ) :
  (∀ x y, x < y → f a x < f a y) ↔ a ≤ 0 :=
sorry

theorem f_monotonic_decreasing_increasing_iff (a : ℝ) :
  (∀ x y, x < y → x ≤ 0 → f a x > f a y) ∧
  (∀ x y, x < y → x ≥ 0 → f a x < f a y) ↔
  a = 1 :=
sorry

end NUMINAMATH_CALUDE_f_monotonicity_f_monotonic_increasing_iff_f_monotonic_decreasing_increasing_iff_l3617_361718


namespace NUMINAMATH_CALUDE_normal_dist_probability_l3617_361752

variable (ξ : Real)
variable (μ δ : Real)

-- ξ follows a normal distribution with mean μ and variance δ²
def normal_dist (ξ μ δ : Real) : Prop := sorry

-- Probability function
noncomputable def P (event : Real → Prop) : Real := sorry

theorem normal_dist_probability 
  (h1 : normal_dist ξ μ δ)
  (h2 : P (λ x => x > 4) = P (λ x => x < 2))
  (h3 : P (λ x => x ≤ 0) = 0.2) :
  P (λ x => 0 < x ∧ x < 6) = 0.6 := by sorry

end NUMINAMATH_CALUDE_normal_dist_probability_l3617_361752


namespace NUMINAMATH_CALUDE_checkerboard_tiling_l3617_361709

/-- The size of the checkerboard -/
def boardSize : Nat := 8

/-- The length of a trimino -/
def triminoLength : Nat := 3

/-- The width of a trimino -/
def triminoWidth : Nat := 1

/-- The area of the checkerboard -/
def boardArea : Nat := boardSize * boardSize

/-- The area of a trimino -/
def triminoArea : Nat := triminoLength * triminoWidth

theorem checkerboard_tiling (boardSize triminoLength triminoWidth : Nat) :
  ¬(boardArea % triminoArea = 0) ∧
  ((boardArea - 1) % triminoArea = 0) := by
  sorry

#check checkerboard_tiling

end NUMINAMATH_CALUDE_checkerboard_tiling_l3617_361709


namespace NUMINAMATH_CALUDE_multiply_mistake_l3617_361724

theorem multiply_mistake (x : ℝ) : 43 * x - 34 * x = 1242 → x = 138 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mistake_l3617_361724


namespace NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3617_361773

theorem right_triangle_segment_ratio (a b c r s : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 → s > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  r * s = c^2 →      -- Geometric mean theorem
  r * c = a^2 →      -- Geometric mean theorem
  s * c = b^2 →      -- Geometric mean theorem
  a / b = 2 / 5 →    -- Given ratio of legs
  r / s = 4 / 25 :=  -- Conclusion to prove
by sorry

end NUMINAMATH_CALUDE_right_triangle_segment_ratio_l3617_361773


namespace NUMINAMATH_CALUDE_max_product_sum_l3617_361792

theorem max_product_sum (a b c d : ℕ) : 
  a ∈ ({1, 3, 4, 5} : Finset ℕ) →
  b ∈ ({1, 3, 4, 5} : Finset ℕ) →
  c ∈ ({1, 3, 4, 5} : Finset ℕ) →
  d ∈ ({1, 3, 4, 5} : Finset ℕ) →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  (a * b + b * c + c * d + d * a) ≤ 42 :=
by sorry

end NUMINAMATH_CALUDE_max_product_sum_l3617_361792


namespace NUMINAMATH_CALUDE_function_composition_sum_l3617_361798

theorem function_composition_sum (a b : ℝ) :
  (∀ x, (5 * (a * x + b) - 7) = 4 * x + 6) →
  a + b = 17 / 5 := by
sorry

end NUMINAMATH_CALUDE_function_composition_sum_l3617_361798


namespace NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l3617_361765

/-- Given four points A, B, X, and Y in a 2D plane, where segment AB is parallel to segment XY,
    prove that the y-coordinate of Y is -1. -/
theorem parallel_segments_y_coordinate
  (A B X Y : ℝ × ℝ)
  (hA : A = (-2, -2))
  (hB : B = (2, -6))
  (hX : X = (1, 5))
  (hY : Y = (7, Y.2))
  (h_parallel : (B.1 - A.1) * (Y.2 - X.2) = (Y.1 - X.1) * (B.2 - A.2)) :
  Y.2 = -1 :=
sorry

end NUMINAMATH_CALUDE_parallel_segments_y_coordinate_l3617_361765


namespace NUMINAMATH_CALUDE_container_filling_l3617_361700

theorem container_filling (initial_percentage : Real) (added_amount : Real) (capacity : Real) :
  initial_percentage = 0.4 →
  added_amount = 14 →
  capacity = 40 →
  (initial_percentage * capacity + added_amount) / capacity = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_container_filling_l3617_361700


namespace NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3617_361705

-- Define a positive geometric sequence
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n > 0 ∧ ∃ r : ℝ, r > 0 ∧ ∀ k : ℕ, a (k + 1) = r * a k

-- State the theorem
theorem fifth_term_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_geom : is_positive_geometric_sequence a)
  (h_roots : a 4 * a 6 = 6 ∧ a 4 + a 6 = 5) :
  a 5 = Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_fifth_term_of_geometric_sequence_l3617_361705


namespace NUMINAMATH_CALUDE_sons_age_l3617_361755

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 5 = 3 * (son_age + 5) →
  son_age = 10 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l3617_361755


namespace NUMINAMATH_CALUDE_mean_proportional_existence_l3617_361735

theorem mean_proportional_existence (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  ∃ x : ℝ, x ^ 2 = a * b :=
by sorry

end NUMINAMATH_CALUDE_mean_proportional_existence_l3617_361735


namespace NUMINAMATH_CALUDE_sunlovers_happy_days_l3617_361740

theorem sunlovers_happy_days (D R : ℕ) : 
  (D^2 + 4) * (R^2 + 4) - 2 * D * (R^2 + 4) - 2 * R * (D^2 + 4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_sunlovers_happy_days_l3617_361740


namespace NUMINAMATH_CALUDE_basketball_substitutions_l3617_361756

def substitution_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | k + 1 => (5 - k) * (11 + k) * substitution_ways k

def total_substitution_ways : ℕ :=
  (List.range 6).map substitution_ways |> List.sum

theorem basketball_substitutions :
  total_substitution_ways % 1000 = 736 := by
  sorry

end NUMINAMATH_CALUDE_basketball_substitutions_l3617_361756


namespace NUMINAMATH_CALUDE_decimal_digit_of_fraction_thirteenth_over_seventeen_150th_digit_l3617_361770

theorem decimal_digit_of_fraction (n : ℕ) (a b : ℕ) (h : b ≠ 0) :
  ∃ (d : ℕ), d < 10 ∧ d = (a * 10^n) % b :=
sorry

theorem thirteenth_over_seventeen_150th_digit :
  ∃ (d : ℕ), d < 10 ∧ d = (13 * 10^150) % 17 ∧ d = 1 :=
sorry

end NUMINAMATH_CALUDE_decimal_digit_of_fraction_thirteenth_over_seventeen_150th_digit_l3617_361770


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3617_361779

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 3 + a 8 = 22)
  (h_a6 : a 6 = 7) :
  a 5 = 15 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3617_361779


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3617_361769

theorem sqrt_product_equality : Real.sqrt 54 * Real.sqrt 48 * Real.sqrt 6 = 72 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3617_361769


namespace NUMINAMATH_CALUDE_smallest_n_with_abc_property_l3617_361743

def has_abc_property (n : ℕ) : Prop :=
  ∀ (A B : Set ℕ), A ∪ B = Finset.range (n + 1) → A ∩ B = ∅ →
    (∃ (a b c : ℕ), a ∈ A ∧ b ∈ A ∧ c ∈ A ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c) ∨
    (∃ (a b c : ℕ), a ∈ B ∧ b ∈ B ∧ c ∈ B ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b = c)

theorem smallest_n_with_abc_property :
  (∀ m < 96, ¬(has_abc_property m)) ∧ has_abc_property 96 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_with_abc_property_l3617_361743


namespace NUMINAMATH_CALUDE_abs_neg_five_eq_five_l3617_361713

theorem abs_neg_five_eq_five : abs (-5 : ℤ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_five_eq_five_l3617_361713


namespace NUMINAMATH_CALUDE_matilda_has_420_jellybeans_l3617_361720

/-- The number of jellybeans Steve has -/
def steve_jellybeans : ℕ := 84

/-- The number of jellybeans Matt has -/
def matt_jellybeans : ℕ := 10 * steve_jellybeans

/-- The number of jellybeans Matilda has -/
def matilda_jellybeans : ℕ := matt_jellybeans / 2

/-- Theorem stating that Matilda has 420 jellybeans -/
theorem matilda_has_420_jellybeans : matilda_jellybeans = 420 := by
  sorry

end NUMINAMATH_CALUDE_matilda_has_420_jellybeans_l3617_361720


namespace NUMINAMATH_CALUDE_mark_sold_nine_boxes_l3617_361754

/-- Given that Mark and Ann were allocated n boxes of cookies to sell, prove that Mark sold 9 boxes. -/
theorem mark_sold_nine_boxes (n : ℕ) (mark_boxes ann_boxes : ℕ) : 
  n = 10 →
  mark_boxes < n →
  ann_boxes = n - 2 →
  mark_boxes ≥ 1 →
  ann_boxes ≥ 1 →
  mark_boxes + ann_boxes < n →
  mark_boxes = 9 := by
sorry

end NUMINAMATH_CALUDE_mark_sold_nine_boxes_l3617_361754


namespace NUMINAMATH_CALUDE_arcsin_sum_equals_pi_over_four_l3617_361708

theorem arcsin_sum_equals_pi_over_four :
  Real.arcsin (1 / Real.sqrt 10) + Real.arcsin (1 / Real.sqrt 26) + 
  Real.arcsin (1 / Real.sqrt 50) + Real.arcsin (1 / Real.sqrt 65) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_sum_equals_pi_over_four_l3617_361708


namespace NUMINAMATH_CALUDE_ryan_marbles_ryan_has_28_marbles_l3617_361715

theorem ryan_marbles (chris_marbles : ℕ) (remaining_marbles : ℕ) : ℕ :=
  let total_marbles := chris_marbles + remaining_marbles * 2
  total_marbles - chris_marbles

theorem ryan_has_28_marbles :
  ryan_marbles 12 20 = 28 :=
by sorry

end NUMINAMATH_CALUDE_ryan_marbles_ryan_has_28_marbles_l3617_361715


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l3617_361757

/-- The probability of picking 2 red balls from a bag containing 3 red balls, 4 blue balls, and 4 green balls. -/
theorem probability_two_red_balls (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (green_balls : ℕ) : 
  total_balls = red_balls + blue_balls + green_balls →
  red_balls = 3 →
  blue_balls = 4 →
  green_balls = 4 →
  (red_balls.choose 2 : ℚ) / (total_balls.choose 2) = 3 / 55 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l3617_361757


namespace NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l3617_361747

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + 2*x^3 + 7*x

-- State the theorem
theorem f_difference_at_3_and_neg_3 : f 3 - f (-3) = 636 := by
  sorry

end NUMINAMATH_CALUDE_f_difference_at_3_and_neg_3_l3617_361747


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3617_361744

theorem arithmetic_square_root_of_16 : ∃ (x : ℝ), x ≥ 0 ∧ x^2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3617_361744


namespace NUMINAMATH_CALUDE_train_length_l3617_361758

theorem train_length (time : Real) (speed_kmh : Real) (length : Real) : 
  time = 2.222044458665529 →
  speed_kmh = 162 →
  length = speed_kmh * (1000 / 3600) * time →
  length = 100 := by
sorry


end NUMINAMATH_CALUDE_train_length_l3617_361758


namespace NUMINAMATH_CALUDE_prob_select_seventh_grade_prob_select_one_from_each_grade_l3617_361723

structure School :=
  (seventh_grade : Finset Nat)
  (eighth_grade : Finset Nat)
  (h1 : seventh_grade.card = 2)
  (h2 : eighth_grade.card = 2)
  (h3 : seventh_grade ∩ eighth_grade = ∅)

def total_students (s : School) : Finset Nat :=
  s.seventh_grade ∪ s.eighth_grade

theorem prob_select_seventh_grade (s : School) :
  (s.seventh_grade.card : ℚ) / (total_students s).card = 1 / 2 := by sorry

theorem prob_select_one_from_each_grade (s : School) :
  let total_pairs := (total_students s).card.choose 2
  let mixed_pairs := s.seventh_grade.card * s.eighth_grade.card * 2
  (mixed_pairs : ℚ) / total_pairs = 2 / 3 := by sorry

end NUMINAMATH_CALUDE_prob_select_seventh_grade_prob_select_one_from_each_grade_l3617_361723
