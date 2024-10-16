import Mathlib

namespace NUMINAMATH_CALUDE_wednesday_discount_percentage_wednesday_jeans_discount_approx_40_82_percent_l3867_386720

/-- Calculates the additional Wednesday discount for jeans given the original price,
    summer discount percentage, and final price after all discounts. -/
theorem wednesday_discount_percentage 
  (original_price : ℝ) 
  (summer_discount_percent : ℝ) 
  (final_price : ℝ) : ℝ :=
  let price_after_summer_discount := original_price * (1 - summer_discount_percent / 100)
  let additional_discount := price_after_summer_discount - final_price
  let wednesday_discount_percent := (additional_discount / price_after_summer_discount) * 100
  wednesday_discount_percent

/-- The additional Wednesday discount for jeans is approximately 40.82% -/
theorem wednesday_jeans_discount_approx_40_82_percent : 
  ∃ ε > 0, abs (wednesday_discount_percentage 49 50 14.5 - 40.82) < ε :=
sorry

end NUMINAMATH_CALUDE_wednesday_discount_percentage_wednesday_jeans_discount_approx_40_82_percent_l3867_386720


namespace NUMINAMATH_CALUDE_odd_function_half_value_l3867_386778

def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x

theorem odd_function_half_value (a : ℝ) :
  (∀ x, f a (-x) = -(f a x)) → f a (1/2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_half_value_l3867_386778


namespace NUMINAMATH_CALUDE_number_problem_l3867_386784

theorem number_problem (x : ℝ) : (x / 6) * 12 = 9 → x = 4.5 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3867_386784


namespace NUMINAMATH_CALUDE_pentadecagon_triangles_l3867_386706

/-- A regular pentadecagon is a 15-sided regular polygon -/
def regular_pentadecagon : ℕ := 15

/-- The number of vertices to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Proposition: The number of triangles formed by the vertices of a regular pentadecagon is 455 -/
theorem pentadecagon_triangles :
  (regular_pentadecagon.choose triangle_vertices) = 455 :=
sorry

end NUMINAMATH_CALUDE_pentadecagon_triangles_l3867_386706


namespace NUMINAMATH_CALUDE_ball_max_height_l3867_386753

/-- The height function of the ball -/
def h (t : ℝ) : ℝ := -5 * t^2 + 20 * t + 10

/-- Theorem stating that the maximum height reached by the ball is 30 meters -/
theorem ball_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 30 :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3867_386753


namespace NUMINAMATH_CALUDE_ingrids_tax_rate_l3867_386749

theorem ingrids_tax_rate 
  (john_tax_rate : ℝ)
  (john_income : ℝ)
  (ingrid_income : ℝ)
  (combined_tax_rate : ℝ)
  (h1 : john_tax_rate = 0.30)
  (h2 : john_income = 58000)
  (h3 : ingrid_income = 72000)
  (h4 : combined_tax_rate = 0.3554)
  : (combined_tax_rate * (john_income + ingrid_income) - john_tax_rate * john_income) / ingrid_income = 0.40 := by
  sorry

end NUMINAMATH_CALUDE_ingrids_tax_rate_l3867_386749


namespace NUMINAMATH_CALUDE_middle_circle_number_l3867_386758

def numbers : List ℕ := [1, 5, 6, 7, 13, 14, 17, 22, 26]

def middle_fixed : List ℕ := [13, 17]

def total_sum : ℕ := numbers.sum

def group_sum : ℕ := total_sum / 3

theorem middle_circle_number (x : ℕ) 
  (h1 : x ∈ numbers)
  (h2 : ∀ (a b c : ℕ), a ∈ numbers → b ∈ numbers → c ∈ numbers → 
       a ≠ b → b ≠ c → a ≠ c → 
       (a + b + c = group_sum) → 
       (a = 13 ∧ b = 17) ∨ (a = 13 ∧ c = 17) ∨ (b = 13 ∧ c = 17) → 
       x = c ∨ x = b) :
  x = 7 := by sorry

end NUMINAMATH_CALUDE_middle_circle_number_l3867_386758


namespace NUMINAMATH_CALUDE_sandy_book_purchase_l3867_386721

/-- The number of books Sandy bought from the first shop -/
def books_first_shop : ℕ := 65

/-- The amount Sandy spent at the first shop -/
def amount_first_shop : ℚ := 1380

/-- The amount Sandy spent at the second shop -/
def amount_second_shop : ℚ := 900

/-- The average price Sandy paid per book -/
def average_price : ℚ := 19

/-- The number of books Sandy bought from the second shop -/
def books_second_shop : ℕ := 55

theorem sandy_book_purchase :
  (amount_first_shop + amount_second_shop) / (books_first_shop + books_second_shop : ℚ) = average_price :=
by sorry

end NUMINAMATH_CALUDE_sandy_book_purchase_l3867_386721


namespace NUMINAMATH_CALUDE_sum_of_digits_9ab_l3867_386752

/-- The number of digits in the sequence -/
def n : ℕ := 2023

/-- Integer a consisting of n nines in base 10 -/
def a : ℕ := 10^n - 1

/-- Integer b consisting of n sixes in base 10 -/
def b : ℕ := 2 * (10^n - 1) / 3

/-- The product 9ab -/
def prod : ℕ := 9 * a * b

/-- Sum of digits function -/
def sum_of_digits (m : ℕ) : ℕ := sorry

theorem sum_of_digits_9ab : sum_of_digits prod = 20235 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_9ab_l3867_386752


namespace NUMINAMATH_CALUDE_student_probability_problem_l3867_386732

theorem student_probability_problem (p q : ℝ) 
  (h_p_pos : 0 < p) (h_q_pos : 0 < q) (h_p_le_one : p ≤ 1) (h_q_le_one : q ≤ 1)
  (h_p_gt_q : p > q)
  (h_at_least_one : 1 - (1 - p) * (1 - q) = 5/6)
  (h_both_correct : p * q = 1/3)
  : p = 2/3 ∧ q = 1/2 ∧ 
    (1 - p)^2 * 2 * (1 - q) * q + (1 - p)^2 * q^2 + 2 * (1 - p) * p * q^2 = 7/36 := by
  sorry

end NUMINAMATH_CALUDE_student_probability_problem_l3867_386732


namespace NUMINAMATH_CALUDE_power_of_negative_cube_l3867_386728

theorem power_of_negative_cube (x : ℝ) : (-x^3)^4 = x^12 := by
  sorry

end NUMINAMATH_CALUDE_power_of_negative_cube_l3867_386728


namespace NUMINAMATH_CALUDE_abs_difference_range_l3867_386768

theorem abs_difference_range (t : ℝ) : let f := λ x : ℝ => Real.sin x + Real.cos x
                                        let g := λ x : ℝ => 2 * Real.cos x
                                        0 ≤ |f t - g t| ∧ |f t - g t| ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_range_l3867_386768


namespace NUMINAMATH_CALUDE_percentage_of_boys_l3867_386701

theorem percentage_of_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  (boy_ratio : ℚ) / (boy_ratio + girl_ratio) * 100 = 42.86 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_boys_l3867_386701


namespace NUMINAMATH_CALUDE_shipment_weight_problem_l3867_386723

theorem shipment_weight_problem (x y : ℕ) : 
  x + (30 - x) = 30 →  -- Total number of boxes is 30
  10 * x + y * (30 - x) = 18 * 30 →  -- Initial average weight is 18 pounds
  10 * x + y * (15 - x) = 16 * 15 →  -- New average weight after removing 15 heavier boxes
  y = 20 := by sorry

end NUMINAMATH_CALUDE_shipment_weight_problem_l3867_386723


namespace NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3867_386709

theorem arithmetic_expression_evaluation :
  1 / 2 + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_evaluation_l3867_386709


namespace NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l3867_386729

theorem quadratic_roots_abs_less_than_one (a b : ℝ) 
  (h1 : |a| + |b| < 1) 
  (h2 : a^2 - 4*b ≥ 0) : 
  ∀ x, x^2 + a*x + b = 0 → |x| < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_abs_less_than_one_l3867_386729


namespace NUMINAMATH_CALUDE_congruence_problem_l3867_386731

def binomial_sum (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ k => Nat.choose n k * 2^k)

theorem congruence_problem (a b : ℤ) :
  a = binomial_sum 20 ∧ a ≡ b [ZMOD 10] → b = 2011 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3867_386731


namespace NUMINAMATH_CALUDE_count_multiples_theorem_l3867_386782

/-- The count of positive integers not exceeding 500 that are multiples of 2 or 5 but not 6 -/
def count_multiples : ℕ := sorry

/-- The upper bound of the range -/
def upper_bound : ℕ := 500

/-- Predicate for a number being a multiple of 2 or 5 but not 6 -/
def is_valid_multiple (n : ℕ) : Prop :=
  n ≤ upper_bound ∧ (n % 2 = 0 ∨ n % 5 = 0) ∧ n % 6 ≠ 0

theorem count_multiples_theorem : count_multiples = 217 := by sorry

end NUMINAMATH_CALUDE_count_multiples_theorem_l3867_386782


namespace NUMINAMATH_CALUDE_functional_equation_solution_l3867_386712

/-- The functional equation that f must satisfy -/
def functional_equation (f : ℝ → ℝ) (α β : ℝ) : Prop :=
  ∀ x y, x > 0 → y > 0 → f x * f y = y^α * f (x/2) + x^β * f (y/2)

/-- The theorem stating the possible forms of f -/
theorem functional_equation_solution (f : ℝ → ℝ) (α β : ℝ) :
  functional_equation f α β →
  (∃ c : ℝ, c = 2^(1-α) ∧ ∀ x, x > 0 → f x = c * x^α) ∨
  (∀ x, x > 0 → f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l3867_386712


namespace NUMINAMATH_CALUDE_lcm_gcf_ratio_l3867_386767

theorem lcm_gcf_ratio (a b : ℕ) (ha : a = 210) (hb : b = 462) : 
  Nat.lcm a b / Nat.gcd a b = 55 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcf_ratio_l3867_386767


namespace NUMINAMATH_CALUDE_nine_digit_integers_count_l3867_386727

/-- The count of 9-digit positive integers, given that a 9-digit number cannot start with 0. -/
def count_nine_digit_integers : ℕ :=
  9 * 10^8

/-- Theorem stating that the count of 9-digit positive integers is 900,000,000. -/
theorem nine_digit_integers_count :
  count_nine_digit_integers = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_nine_digit_integers_count_l3867_386727


namespace NUMINAMATH_CALUDE_absolute_difference_in_terms_of_sum_and_product_l3867_386798

theorem absolute_difference_in_terms_of_sum_and_product (x₁ x₂ a b : ℝ) 
  (h_sum : x₁ + x₂ = a) (h_product : x₁ * x₂ = b) : 
  |x₁ - x₂| = Real.sqrt (a^2 - 4*b) := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_in_terms_of_sum_and_product_l3867_386798


namespace NUMINAMATH_CALUDE_converse_propositions_l3867_386765

-- Define the basic concepts
def Point : Type := ℝ × ℝ × ℝ
def Line : Type := Point → Prop

-- Define the relationships
def coplanar (a b c d : Point) : Prop := sorry
def collinear (a b c : Point) : Prop := sorry
def have_common_point (l₁ l₂ : Line) : Prop := sorry
def skew_lines (l₁ l₂ : Line) : Prop := sorry

-- State the theorem
theorem converse_propositions :
  (∀ a b c d : Point, (¬collinear a b c ∧ ¬collinear a b d ∧ ¬collinear a c d ∧ ¬collinear b c d) → ¬coplanar a b c d) = false ∧
  (∀ l₁ l₂ : Line, skew_lines l₁ l₂ → ¬have_common_point l₁ l₂) = true :=
sorry

end NUMINAMATH_CALUDE_converse_propositions_l3867_386765


namespace NUMINAMATH_CALUDE_square_of_binomial_l3867_386775

theorem square_of_binomial (x : ℝ) : ∃ (a : ℝ), x^2 - 20*x + 100 = (x - a)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_binomial_l3867_386775


namespace NUMINAMATH_CALUDE_unique_solution_for_equation_l3867_386743

theorem unique_solution_for_equation (n : ℕ+) (p : ℕ) : 
  Nat.Prime p → (n : ℕ)^8 - (n : ℕ)^2 = p^5 + p^2 → (n = 2 ∧ p = 3) :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_for_equation_l3867_386743


namespace NUMINAMATH_CALUDE_expansion_coefficient_l3867_386771

/-- The coefficient of the x^(3/2) term in the expansion of (√x - a/√x)^5 -/
def coefficient (a : ℝ) : ℝ := -5 * a

theorem expansion_coefficient (a : ℝ) :
  coefficient a = 30 → a = -6 := by sorry

end NUMINAMATH_CALUDE_expansion_coefficient_l3867_386771


namespace NUMINAMATH_CALUDE_triangle_properties_l3867_386711

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  a > 0 ∧ b > 0 ∧ c > 0 →
  c = a * (Real.cos B + Real.sqrt 3 * Real.sin B) →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 4 →
  a = 1 →
  A = π / 6 ∧ a + b + c = Real.sqrt 3 + 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_properties_l3867_386711


namespace NUMINAMATH_CALUDE_solve_equation_l3867_386780

theorem solve_equation (y : ℚ) (h : (1 : ℚ) / 3 + 1 / y = 7 / 12) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3867_386780


namespace NUMINAMATH_CALUDE_root_equation_l3867_386754

theorem root_equation (p : ℝ) :
  (0 ≤ p ∧ p ≤ 4/3) →
  (∃! x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) = x ∧
             x = (4 - p) / Real.sqrt (8 * (2 - p))) ∧
  (p < 0 ∨ p > 4/3) →
  (∀ x : ℝ, Real.sqrt (x^2 - p) + 2 * Real.sqrt (x^2 - 1) ≠ x) := by
sorry

end NUMINAMATH_CALUDE_root_equation_l3867_386754


namespace NUMINAMATH_CALUDE_polynomial_existence_l3867_386746

theorem polynomial_existence : 
  ∃ (p : ℝ → ℝ), 
    (∃ (a b c : ℝ), ∀ x, p x = a * x^2 + b * x + c) ∧ 
    p 0 = 100 ∧ 
    p 1 = 90 ∧ 
    p 2 = 70 ∧ 
    p 3 = 40 ∧ 
    p 4 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_existence_l3867_386746


namespace NUMINAMATH_CALUDE_base3_to_base10_conversion_l3867_386722

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- The base-3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 1, 2]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 196 := by
  sorry

end NUMINAMATH_CALUDE_base3_to_base10_conversion_l3867_386722


namespace NUMINAMATH_CALUDE_cube_face_planes_divide_space_l3867_386781

-- Define a cube in 3D space
def Cube := Set (ℝ × ℝ × ℝ)

-- Define the planes that each face of the cube lies on
def FacePlanes (c : Cube) := Set (Set (ℝ × ℝ × ℝ))

-- Define a function that counts the number of regions created by the face planes
def countRegions (c : Cube) : ℕ := sorry

-- Theorem stating that the face planes of a cube divide space into 27 regions
theorem cube_face_planes_divide_space (c : Cube) : 
  countRegions c = 27 := by sorry

end NUMINAMATH_CALUDE_cube_face_planes_divide_space_l3867_386781


namespace NUMINAMATH_CALUDE_teacher_assignment_problem_l3867_386739

theorem teacher_assignment_problem :
  let n : ℕ := 4  -- number of teachers
  let k : ℕ := 3  -- number of classes
  let ways : ℕ := (n.choose 2) * (k.factorial)  -- C(4,2) * A(3,3)
  ways = 36 := by sorry

end NUMINAMATH_CALUDE_teacher_assignment_problem_l3867_386739


namespace NUMINAMATH_CALUDE_binomial_10_5_l3867_386799

theorem binomial_10_5 : Nat.choose 10 5 = 252 := by
  sorry

end NUMINAMATH_CALUDE_binomial_10_5_l3867_386799


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3867_386794

/-- 
Given an arithmetic sequence {a_n} with common difference d ≥ 0,
if a_2^2 is the arithmetic mean of a_1^2 and a_3^2 - 2, then d = 1.
-/
theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ) :
  d ≥ 0 →
  (∀ n, a (n + 1) = a n + d) →
  a 2^2 = (a 1^2 + (a 3^2 - 2)) / 2 →
  d = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3867_386794


namespace NUMINAMATH_CALUDE_teacher_assignment_count_l3867_386796

/-- The number of ways to assign four teachers to three classes --/
def total_assignments : ℕ := 36

/-- The number of ways to assign teachers A and B to the same class --/
def ab_same_class : ℕ := 6

/-- The number of ways to assign four teachers to three classes with A and B in different classes --/
def valid_assignments : ℕ := total_assignments - ab_same_class

theorem teacher_assignment_count :
  valid_assignments = 30 :=
sorry

end NUMINAMATH_CALUDE_teacher_assignment_count_l3867_386796


namespace NUMINAMATH_CALUDE_second_term_base_l3867_386707

theorem second_term_base (x y : ℕ) (base : ℝ) : 
  3^x * base^y = 19683 → x - y = 9 → x = 9 → base = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_second_term_base_l3867_386707


namespace NUMINAMATH_CALUDE_a_range_theorem_l3867_386787

-- Define the sequence a_n
def a_n (a n : ℝ) : ℝ := a * n^2 + n + 5

-- State the theorem
theorem a_range_theorem (a : ℝ) :
  (∀ n : ℕ, a_n a n < a_n a (n + 1) ∧ n ≤ 3) ∧
  (∀ n : ℕ, a_n a n > a_n a (n + 1) ∧ n ≥ 8) →
  -1/7 < a ∧ a < -1/17 :=
sorry

end NUMINAMATH_CALUDE_a_range_theorem_l3867_386787


namespace NUMINAMATH_CALUDE_family_pizza_order_correct_l3867_386776

def family_pizza_order (adults : Nat) (children : Nat) (adult_slices : Nat) (child_slices : Nat) (slices_per_pizza : Nat) : Nat :=
  let total_slices := adults * adult_slices + children * child_slices
  (total_slices + slices_per_pizza - 1) / slices_per_pizza

theorem family_pizza_order_correct :
  family_pizza_order 2 12 5 2 6 = 6 := by
  sorry

end NUMINAMATH_CALUDE_family_pizza_order_correct_l3867_386776


namespace NUMINAMATH_CALUDE_negation_quadratic_inequality_l3867_386764

theorem negation_quadratic_inequality (x : ℝ) : 
  ¬(x^2 - x + 3 > 0) ↔ x^2 - x + 3 ≤ 0 := by sorry

end NUMINAMATH_CALUDE_negation_quadratic_inequality_l3867_386764


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3867_386734

theorem simplify_sqrt_expression (x : ℝ) (hx : x ≠ 0) :
  Real.sqrt (1 + ((x^6 - 2) / (3 * x^3))^2) = (Real.sqrt (x^12 + 5*x^6 + 4)) / (3 * x^3) :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3867_386734


namespace NUMINAMATH_CALUDE_tea_cost_theorem_l3867_386786

/-- Represents the cost calculation for tea sets and cups under different options -/
def tea_cost (x : ℕ) : Prop :=
  let tea_set_price : ℕ := 200
  let tea_cup_price : ℕ := 20
  let option1_cost : ℕ := 20 * x + 5400
  let option2_cost : ℕ := 19 * x + 5700
  (x > 30) →
  (option1_cost = 30 * tea_set_price + tea_cup_price * (x - 30)) ∧
  (option2_cost = (30 * tea_set_price + x * tea_cup_price) * 95 / 100) ∧
  (x = 50 → option1_cost < option2_cost)

theorem tea_cost_theorem :
  ∀ x : ℕ, tea_cost x :=
sorry

end NUMINAMATH_CALUDE_tea_cost_theorem_l3867_386786


namespace NUMINAMATH_CALUDE_morning_ribbons_l3867_386755

theorem morning_ribbons (initial : ℕ) (afternoon : ℕ) (remaining : ℕ) : 
  initial = 38 → afternoon = 16 → remaining = 8 → initial - afternoon - remaining = 14 := by
  sorry

end NUMINAMATH_CALUDE_morning_ribbons_l3867_386755


namespace NUMINAMATH_CALUDE_odd_symmetric_function_property_l3867_386762

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- A function f: ℝ → ℝ is symmetric about x = 3 if f(3+x) = f(3-x) for all x -/
def SymmetricAboutThree (f : ℝ → ℝ) : Prop :=
  ∀ x, f (3 + x) = f (3 - x)

theorem odd_symmetric_function_property (f : ℝ → ℝ) 
  (h_odd : OddFunction f)
  (h_sym : SymmetricAboutThree f)
  (h_def : ∀ x ∈ Set.Ioo 0 3, f x = 2^x) :
  ∀ x ∈ Set.Ioo (-6) (-3), f x = -(2^(x + 6)) := by
  sorry

end NUMINAMATH_CALUDE_odd_symmetric_function_property_l3867_386762


namespace NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3867_386745

-- System (1)
theorem system_one_solution (x y : ℝ) : 
  x = 5 - y ∧ x - 3*y = 1 → x = 4 ∧ y = 1 := by sorry

-- System (2)
theorem system_two_solution (x y : ℝ) :
  x - 2*y = 6 ∧ 2*x + 3*y = -2 → x = 2 ∧ y = -2 := by sorry

end NUMINAMATH_CALUDE_system_one_solution_system_two_solution_l3867_386745


namespace NUMINAMATH_CALUDE_project_distribution_count_l3867_386750

/-- The number of ways to distribute 8 distinct projects among 4 companies -/
def distribute_projects : ℕ :=
  Nat.choose 8 3 * Nat.choose 5 1 * Nat.choose 4 2 * Nat.choose 2 2

/-- Theorem stating that the number of ways to distribute the projects is 1680 -/
theorem project_distribution_count : distribute_projects = 1680 := by
  sorry

end NUMINAMATH_CALUDE_project_distribution_count_l3867_386750


namespace NUMINAMATH_CALUDE_log_equation_solution_l3867_386789

theorem log_equation_solution :
  ∃! x : ℝ, x > 0 ∧ Real.log x - Real.log 6 = 2 :=
by
  use 3/2
  sorry

end NUMINAMATH_CALUDE_log_equation_solution_l3867_386789


namespace NUMINAMATH_CALUDE_sum_of_integers_between_2_and_15_l3867_386773

theorem sum_of_integers_between_2_and_15 : 
  (Finset.range 12).sum (fun i => i + 3) = 102 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_between_2_and_15_l3867_386773


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_i_cubed_plus_one_l3867_386716

theorem imaginary_part_of_i_times_i_cubed_plus_one (i : ℂ) : 
  Complex.im (i * (i^3 + 1)) = 1 :=
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_i_cubed_plus_one_l3867_386716


namespace NUMINAMATH_CALUDE_specific_tetrahedron_properties_l3867_386717

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem specific_tetrahedron_properties :
  let A₁ : Point3D := ⟨-2, 0, -4⟩
  let A₂ : Point3D := ⟨-1, 7, 1⟩
  let A₃ : Point3D := ⟨4, -8, -4⟩
  let A₄ : Point3D := ⟨1, -4, 6⟩
  (tetrahedronVolume A₁ A₂ A₃ A₄ = 250 / 3) ∧
  (tetrahedronHeight A₁ A₂ A₃ A₄ = 5 * Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_CALUDE_specific_tetrahedron_properties_l3867_386717


namespace NUMINAMATH_CALUDE_max_sum_reciprocal_zeros_l3867_386733

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0

theorem max_sum_reciprocal_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ 1/x₁ + 1/x₂ ≤ 9/4) ∧
  (∃ k₀ : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k₀ x₁ = 0 ∧ f k₀ x₂ = 0 ∧ 1/x₁ + 1/x₂ = 9/4) :=
by sorry

end NUMINAMATH_CALUDE_max_sum_reciprocal_zeros_l3867_386733


namespace NUMINAMATH_CALUDE_hawk_percentage_l3867_386702

theorem hawk_percentage (total : ℝ) (hawk paddyfield kingfisher other : ℝ) : 
  total > 0 ∧
  hawk ≥ 0 ∧ paddyfield ≥ 0 ∧ kingfisher ≥ 0 ∧ other ≥ 0 ∧
  hawk + paddyfield + kingfisher + other = total ∧
  paddyfield = 0.4 * (total - hawk) ∧
  kingfisher = 0.25 * paddyfield ∧
  other = 0.35 * total →
  hawk = 0.3 * total :=
by sorry

end NUMINAMATH_CALUDE_hawk_percentage_l3867_386702


namespace NUMINAMATH_CALUDE_absolute_value_sum_inequality_l3867_386772

theorem absolute_value_sum_inequality (x : ℝ) :
  |x - 1| + |x - 2| > 5 ↔ x < -1 ∨ x > 4 :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_sum_inequality_l3867_386772


namespace NUMINAMATH_CALUDE_athlete_arrangements_correct_l3867_386777

/-- The number of ways to arrange 5 athletes on 5 tracks with exactly two matches -/
def athlete_arrangements : ℕ := 20

/-- Proof that the number of arrangements is correct -/
theorem athlete_arrangements_correct : athlete_arrangements = 20 := by
  sorry

end NUMINAMATH_CALUDE_athlete_arrangements_correct_l3867_386777


namespace NUMINAMATH_CALUDE_sector_central_angle_l3867_386759

/-- Given a sector with area 1 cm² and perimeter 4 cm, its central angle is 2 radians. -/
theorem sector_central_angle (r : ℝ) (θ : ℝ) 
  (h_area : (1/2) * θ * r^2 = 1)
  (h_perimeter : 2*r + θ*r = 4) :
  θ = 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_central_angle_l3867_386759


namespace NUMINAMATH_CALUDE_book_combinations_l3867_386724

theorem book_combinations (n m : ℕ) (h1 : n = 15) (h2 : m = 3) : Nat.choose n m = 455 := by
  sorry

end NUMINAMATH_CALUDE_book_combinations_l3867_386724


namespace NUMINAMATH_CALUDE_class_size_proof_l3867_386725

/-- Proves that the number of students in a class is 27 given specific score distributions and averages -/
theorem class_size_proof (n : ℕ) : 
  (5 : ℝ) * 95 + (3 : ℝ) * 0 + ((n : ℝ) - 8) * 45 = (n : ℝ) * 49.25925925925926 → 
  n = 27 := by
  sorry

end NUMINAMATH_CALUDE_class_size_proof_l3867_386725


namespace NUMINAMATH_CALUDE_total_silverware_l3867_386785

/-- The number of types of silverware --/
def num_types : ℕ := 4

/-- The initial number of each type for personal use --/
def initial_personal : ℕ := 5

/-- The number of extra pieces of each type for guests --/
def extra_for_guests : ℕ := 10

/-- The reduction in the number of spoons --/
def spoon_reduction : ℕ := 4

/-- The reduction in the number of butter knives --/
def butter_knife_reduction : ℕ := 4

/-- The reduction in the number of steak knives --/
def steak_knife_reduction : ℕ := 5

/-- The reduction in the number of forks --/
def fork_reduction : ℕ := 3

/-- The theorem stating the total number of silverware pieces Stephanie will buy --/
theorem total_silverware : 
  (initial_personal + extra_for_guests - spoon_reduction) +
  (initial_personal + extra_for_guests - butter_knife_reduction) +
  (initial_personal + extra_for_guests - steak_knife_reduction) +
  (initial_personal + extra_for_guests - fork_reduction) = 44 := by
  sorry

end NUMINAMATH_CALUDE_total_silverware_l3867_386785


namespace NUMINAMATH_CALUDE_halfway_fraction_reduced_l3867_386741

theorem halfway_fraction_reduced (a b c d e f : ℚ) : 
  a = 3/4 → 
  b = 5/6 → 
  c = (a + b) / 2 → 
  d = 1/12 → 
  e = c - d → 
  f = 17/24 → 
  e = f := by sorry

end NUMINAMATH_CALUDE_halfway_fraction_reduced_l3867_386741


namespace NUMINAMATH_CALUDE_bargain_bin_book_count_l3867_386700

/-- Calculates the final number of books in a bargain bin after selling and adding books. -/
def final_book_count (initial : ℕ) (sold : ℕ) (added : ℕ) : ℕ :=
  initial - sold + added

/-- Proves that the final number of books in the bin is correct for the given scenario. -/
theorem bargain_bin_book_count :
  final_book_count 4 3 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_bargain_bin_book_count_l3867_386700


namespace NUMINAMATH_CALUDE_bus_distance_l3867_386797

/-- Represents the distance traveled by each mode of transportation -/
structure TravelDistances where
  total : ℝ
  plane : ℝ
  train : ℝ
  bus : ℝ

/-- The conditions of the travel problem -/
def travel_conditions (d : TravelDistances) : Prop :=
  d.total = 900 ∧
  d.plane = d.total / 3 ∧
  d.train = 2 / 3 * d.bus ∧
  d.total = d.plane + d.train + d.bus

/-- The theorem stating that under the given conditions, the bus travel distance is 360 km -/
theorem bus_distance (d : TravelDistances) (h : travel_conditions d) : d.bus = 360 := by
  sorry

end NUMINAMATH_CALUDE_bus_distance_l3867_386797


namespace NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l3867_386792

theorem abs_sum_reciprocals_ge_two (a b : ℝ) (h : a * b ≠ 0) :
  |a / b + b / a| ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_reciprocals_ge_two_l3867_386792


namespace NUMINAMATH_CALUDE_divisible_by_24_l3867_386736

theorem divisible_by_24 (a : ℤ) : ∃ k : ℤ, (a^2 + 3*a + 1)^2 - 1 = 24*k := by
  sorry

end NUMINAMATH_CALUDE_divisible_by_24_l3867_386736


namespace NUMINAMATH_CALUDE_solve_equation_for_x_l3867_386708

theorem solve_equation_for_x (x y : ℤ) 
  (h1 : x > y) 
  (h2 : y > 0) 
  (h3 : x + y + x * y = 101) : 
  x = 50 := by
sorry

end NUMINAMATH_CALUDE_solve_equation_for_x_l3867_386708


namespace NUMINAMATH_CALUDE_association_member_condition_l3867_386748

/-- Represents a member of the association -/
structure Member where
  number : Nat
  country : Fin 6

/-- The set of all members in the association -/
def Association := Fin 1978 → Member

/-- Predicate to check if a member's number satisfies the condition -/
def SatisfiesCondition (assoc : Association) (m : Member) : Prop :=
  ∃ (a b : Member),
    a.country = m.country ∧ b.country = m.country ∧
    ((a.number + b.number = m.number) ∨ (2 * a.number = m.number))

/-- Main theorem -/
theorem association_member_condition (assoc : Association) :
  ∃ (m : Member), m ∈ Set.range assoc ∧ SatisfiesCondition assoc m := by
  sorry


end NUMINAMATH_CALUDE_association_member_condition_l3867_386748


namespace NUMINAMATH_CALUDE_eight_to_power_divided_by_four_l3867_386790

theorem eight_to_power_divided_by_four (n : ℕ) : 
  n = 8^2022 → n / 4 = 4^3032 := by sorry

end NUMINAMATH_CALUDE_eight_to_power_divided_by_four_l3867_386790


namespace NUMINAMATH_CALUDE_family_member_bites_eq_two_l3867_386719

/-- The number of mosquito bites each family member (excluding Cyrus) has, given the conditions in the problem. -/
def family_member_bites : ℕ :=
  let cyrus_arm_leg_bites : ℕ := 14
  let cyrus_body_bites : ℕ := 10
  let cyrus_total_bites : ℕ := cyrus_arm_leg_bites + cyrus_body_bites
  let family_size : ℕ := 6
  let family_total_bites : ℕ := cyrus_total_bites / 2
  family_total_bites / family_size

theorem family_member_bites_eq_two : family_member_bites = 2 := by
  sorry

end NUMINAMATH_CALUDE_family_member_bites_eq_two_l3867_386719


namespace NUMINAMATH_CALUDE_linear_equation_sum_l3867_386788

theorem linear_equation_sum (m n : ℤ) : 
  (∃ a b c : ℝ, ∀ x y : ℝ, (n - 1) * x^(n^2) - 3 * y^(m - 2023) = a * x + b * y + c) → 
  m + n = 2023 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_sum_l3867_386788


namespace NUMINAMATH_CALUDE_square_a_times_a_plus_four_l3867_386774

theorem square_a_times_a_plus_four (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_a_times_a_plus_four_l3867_386774


namespace NUMINAMATH_CALUDE_pyramid_z_value_l3867_386769

/-- Represents a three-level pyramid structure -/
structure Pyramid where
  z : ℕ
  x : ℕ
  y : ℕ
  bottom_left : ℕ
  bottom_middle : ℕ
  bottom_right : ℕ

/-- Checks if the pyramid satisfies the given conditions -/
def is_valid_pyramid (p : Pyramid) : Prop :=
  p.bottom_left = p.z * p.x ∧
  p.bottom_middle = p.x * p.y ∧
  p.bottom_right = p.y * p.z

theorem pyramid_z_value :
  ∀ p : Pyramid,
    is_valid_pyramid p →
    p.bottom_left = 8 →
    p.bottom_middle = 40 →
    p.bottom_right = 10 →
    p.z = 4 :=
by
  sorry


end NUMINAMATH_CALUDE_pyramid_z_value_l3867_386769


namespace NUMINAMATH_CALUDE_prairie_area_l3867_386756

/-- The total area of a prairie given the area covered by a dust storm and the area left untouched -/
theorem prairie_area (dust_covered : ℕ) (untouched : ℕ) : dust_covered = 64535 → untouched = 522 → dust_covered + untouched = 65057 := by
  sorry

#check prairie_area

end NUMINAMATH_CALUDE_prairie_area_l3867_386756


namespace NUMINAMATH_CALUDE_rectangle_two_axes_l3867_386757

-- Define the types of shapes
inductive Shape
  | EquilateralTriangle
  | Parallelogram
  | Rectangle
  | Square

-- Define a function to count axes of symmetry
def axesOfSymmetry (s : Shape) : ℕ :=
  match s with
  | Shape.EquilateralTriangle => 3
  | Shape.Parallelogram => 0
  | Shape.Rectangle => 2
  | Shape.Square => 4

-- Theorem statement
theorem rectangle_two_axes :
  ∀ s : Shape, axesOfSymmetry s = 2 ↔ s = Shape.Rectangle :=
by sorry

end NUMINAMATH_CALUDE_rectangle_two_axes_l3867_386757


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l3867_386710

/-- 
If a quadratic equation x² - 2x + m = 0 has two equal real roots,
then m = 1.
-/
theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   (∀ y : ℝ, y^2 - 2*y + m = 0 → y = x)) → 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l3867_386710


namespace NUMINAMATH_CALUDE_cubic_function_theorem_l3867_386730

/-- Given a function f and its derivative f', g is defined as their sum -/
def g (f : ℝ → ℝ) (f' : ℝ → ℝ) : ℝ → ℝ := λ x => f x + f' x

/-- f is a cubic function with parameters a and b -/
def f (a b : ℝ) : ℝ → ℝ := λ x => a * x^3 + x^2 + b * x

/-- f' is the derivative of f -/
def f' (a b : ℝ) : ℝ → ℝ := λ x => 3 * a * x^2 + 2 * x + b

theorem cubic_function_theorem (a b : ℝ) :
  (∀ x, g (f a b) (f' a b) (-x) = -(g (f a b) (f' a b) x)) →
  (f a b = λ x => -1/3 * x^3 + x^2) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, g (f a b) (f' a b) y ≤ g (f a b) (f' a b) x) ∧
  (g (f a b) (f' a b) x = 5/3) ∧
  (∃ x ∈ Set.Icc 1 2, ∀ y ∈ Set.Icc 1 2, g (f a b) (f' a b) x ≤ g (f a b) (f' a b) y) ∧
  (g (f a b) (f' a b) x = 4/3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_function_theorem_l3867_386730


namespace NUMINAMATH_CALUDE_complex_expression_equals_23_over_150_l3867_386742

theorem complex_expression_equals_23_over_150 : 
  let x := (27/8)^(2/3) - (49/9)^(1/2) + 0.008^(2/3) / 0.02^(1/2) * 0.32^(1/2)
  (x / 0.0625^0.25) = 23/150 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equals_23_over_150_l3867_386742


namespace NUMINAMATH_CALUDE_hotel_charge_comparison_l3867_386715

theorem hotel_charge_comparison (G : ℝ) (P R : ℝ) 
  (hP : P = G * (1 - 0.1))
  (hR : R = G * (1 + 0.8)) :
  P = R * (1 - 0.5) :=
by sorry

end NUMINAMATH_CALUDE_hotel_charge_comparison_l3867_386715


namespace NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3867_386713

/-- The volume of a tetrahedron formed by alternately colored vertices of a cube -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) 
  (h_side_length : cube_side_length = 10) : ℝ :=
by
  -- The volume of the tetrahedron formed by alternately colored vertices
  -- of a cube with side length 10 units is 1000/3 cubic units
  sorry

#check tetrahedron_volume_in_cube

end NUMINAMATH_CALUDE_tetrahedron_volume_in_cube_l3867_386713


namespace NUMINAMATH_CALUDE_f_2_eq_0_l3867_386703

def f (x : ℝ) : ℝ := x^6 - 12*x^5 + 60*x^4 - 160*x^3 + 240*x^2 - 192*x + 64

def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

theorem f_2_eq_0 :
  f 2 = horner_eval [1, -12, 60, -160, 240, -192, 64] 2 ∧
  horner_eval [1, -12, 60, -160, 240, -192, 64] 2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_f_2_eq_0_l3867_386703


namespace NUMINAMATH_CALUDE_circle_ratio_after_diameter_increase_l3867_386751

/-- Theorem: For any circle with an initial diameter of 2r units, 
if the diameter is increased by 4 units, 
the ratio of the new circumference to the new diameter is equal to π. -/
theorem circle_ratio_after_diameter_increase (r : ℝ) (r_pos : r > 0) : 
  let initial_diameter : ℝ := 2 * r
  let new_diameter : ℝ := initial_diameter + 4
  let new_circumference : ℝ := 2 * π * (r + 2)
  new_circumference / new_diameter = π :=
by sorry

end NUMINAMATH_CALUDE_circle_ratio_after_diameter_increase_l3867_386751


namespace NUMINAMATH_CALUDE_opposite_colors_in_prism_l3867_386747

-- Define the set of colors
inductive Color
  | Red
  | Yellow
  | Blue
  | Black
  | White
  | Green

-- Define a cube as a function from faces to colors
def Cube := Fin 6 → Color

-- Define the property of having all different colors
def allDifferentColors (c : Cube) : Prop :=
  ∀ i j : Fin 6, i ≠ j → c i ≠ c j

-- Define the property of opposite faces having the same color in a rectangular prism
def oppositeColorsSame (c : Cube) : Prop :=
  (c 0 = Color.Red ∧ c 5 = Color.Green) ∨
  (c 0 = Color.Green ∧ c 5 = Color.Red) ∧
  (c 1 = Color.Yellow ∧ c 4 = Color.Blue) ∨
  (c 1 = Color.Blue ∧ c 4 = Color.Yellow) ∧
  (c 2 = Color.Black ∧ c 3 = Color.White) ∨
  (c 2 = Color.White ∧ c 3 = Color.Black)

-- Theorem stating the opposite colors in the rectangular prism
theorem opposite_colors_in_prism (c : Cube) 
  (h1 : allDifferentColors c) 
  (h2 : oppositeColorsSame c) :
  (c 0 = Color.Red → c 5 = Color.Green) ∧
  (c 1 = Color.Yellow → c 4 = Color.Blue) ∧
  (c 2 = Color.Black → c 3 = Color.White) :=
by sorry

end NUMINAMATH_CALUDE_opposite_colors_in_prism_l3867_386747


namespace NUMINAMATH_CALUDE_two_std_dev_less_than_mean_example_l3867_386763

/-- For a normal distribution with given mean and standard deviation,
    calculate the value that is exactly 2 standard deviations less than the mean -/
def twoStdDevLessThanMean (mean : ℝ) (stdDev : ℝ) : ℝ :=
  mean - 2 * stdDev

/-- Theorem stating that for a normal distribution with mean 12 and standard deviation 1.2,
    the value exactly 2 standard deviations less than the mean is 9.6 -/
theorem two_std_dev_less_than_mean_example :
  twoStdDevLessThanMean 12 1.2 = 9.6 := by
  sorry

end NUMINAMATH_CALUDE_two_std_dev_less_than_mean_example_l3867_386763


namespace NUMINAMATH_CALUDE_max_value_of_function_l3867_386779

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 3/2) : 
  x * (3 - 2*x) ≤ 9/8 ∧ ∃ x₀, 0 < x₀ ∧ x₀ < 3/2 ∧ x₀ * (3 - 2*x₀) = 9/8 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_function_l3867_386779


namespace NUMINAMATH_CALUDE_fans_attended_l3867_386735

def stadium_capacity : ℕ := 60000
def seats_sold_percentage : ℚ := 75 / 100
def fans_stayed_home : ℕ := 5000

theorem fans_attended (capacity : ℕ) (sold_percentage : ℚ) (stayed_home : ℕ) 
  (h1 : capacity = stadium_capacity)
  (h2 : sold_percentage = seats_sold_percentage)
  (h3 : stayed_home = fans_stayed_home) :
  (capacity : ℚ) * sold_percentage - stayed_home = 40000 := by
  sorry

end NUMINAMATH_CALUDE_fans_attended_l3867_386735


namespace NUMINAMATH_CALUDE_find_n_l3867_386760

theorem find_n (a n : ℕ) (h1 : a^2 % n = 8) (h2 : a^3 % n = 25) : n = 113 := by
  sorry

end NUMINAMATH_CALUDE_find_n_l3867_386760


namespace NUMINAMATH_CALUDE_prism_volume_problem_l3867_386795

/-- 
Given a rectangular prism with dimensions 15 cm × 5 cm × 4 cm and a smaller prism
with dimensions y cm × 5 cm × x cm removed, if the remaining volume is 120 cm³,
then x + y = 15, where x and y are integers.
-/
theorem prism_volume_problem (x y : ℤ) : 
  (15 * 5 * 4 - y * 5 * x = 120) → (x + y = 15) := by sorry

end NUMINAMATH_CALUDE_prism_volume_problem_l3867_386795


namespace NUMINAMATH_CALUDE_point_on_line_l3867_386737

/-- A point on a 2D plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear --/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p1.x) = (p3.y - p1.y) * (p2.x - p1.x)

theorem point_on_line (t : ℝ) :
  let p1 : Point := ⟨2, 4⟩
  let p2 : Point := ⟨10, 1⟩
  let p3 : Point := ⟨t, 7⟩
  collinear p1 p2 p3 → t = -6 := by
  sorry


end NUMINAMATH_CALUDE_point_on_line_l3867_386737


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l3867_386791

theorem sum_of_x_and_y (x y : ℝ) (some_number : ℝ) 
  (h1 : x + y = some_number) 
  (h2 : x - y = 5) 
  (h3 : x = 10) 
  (h4 : y = 5) : 
  x + y = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l3867_386791


namespace NUMINAMATH_CALUDE_milk_fraction_in_cup1_l3867_386783

theorem milk_fraction_in_cup1 (initial_tea : ℝ) (initial_milk : ℝ) (cup_size : ℝ) : 
  initial_tea = 6 →
  initial_milk = 8 →
  cup_size = 12 →
  let tea_transferred_to_cup2 := initial_tea / 3
  let tea_in_cup1_after_first_transfer := initial_tea - tea_transferred_to_cup2
  let total_in_cup2_after_first_transfer := initial_milk + tea_transferred_to_cup2
  let amount_transferred_back := total_in_cup2_after_first_transfer / 4
  let milk_ratio_in_cup2 := initial_milk / total_in_cup2_after_first_transfer
  let milk_transferred_back := amount_transferred_back * milk_ratio_in_cup2
  let final_tea_in_cup1 := tea_in_cup1_after_first_transfer + (amount_transferred_back - milk_transferred_back)
  let final_milk_in_cup1 := milk_transferred_back
  let total_liquid_in_cup1 := final_tea_in_cup1 + final_milk_in_cup1
  final_milk_in_cup1 / total_liquid_in_cup1 = 2 / 6.5 :=
by sorry

end NUMINAMATH_CALUDE_milk_fraction_in_cup1_l3867_386783


namespace NUMINAMATH_CALUDE_interest_time_period_l3867_386726

def principal : ℝ := 8999.999999999993
def rate : ℝ := 0.20

theorem interest_time_period : 
  ∃ t : ℝ, t = 2 ∧ 
  principal * ((1 + rate) ^ t - 1) - principal * rate * t = 360 := by
  sorry

end NUMINAMATH_CALUDE_interest_time_period_l3867_386726


namespace NUMINAMATH_CALUDE_count_linear_inequalities_one_variable_l3867_386770

-- Define a structure for an expression
structure Expression where
  is_linear_inequality : Bool
  has_one_variable : Bool

-- Define the six expressions
def expressions : List Expression := [
  { is_linear_inequality := true,  has_one_variable := true  }, -- ①
  { is_linear_inequality := false, has_one_variable := true  }, -- ②
  { is_linear_inequality := false, has_one_variable := true  }, -- ③
  { is_linear_inequality := true,  has_one_variable := true  }, -- ④
  { is_linear_inequality := true,  has_one_variable := true  }, -- ⑤
  { is_linear_inequality := true,  has_one_variable := false }  -- ⑥
]

-- Theorem statement
theorem count_linear_inequalities_one_variable :
  (expressions.filter (fun e => e.is_linear_inequality && e.has_one_variable)).length = 3 := by
  sorry

end NUMINAMATH_CALUDE_count_linear_inequalities_one_variable_l3867_386770


namespace NUMINAMATH_CALUDE_hiring_probability_l3867_386793

/-- The number of candidates -/
def numCandidates : ℕ := 4

/-- The number of people to be hired -/
def numHired : ℕ := 2

/-- The probability of hiring at least one of two specific candidates -/
def probAtLeastOne : ℚ := 5/6

theorem hiring_probability :
  (numCandidates : ℚ) > 0 ∧ numHired ≤ numCandidates →
  (1 : ℚ) - (Nat.choose (numCandidates - 2) numHired : ℚ) / (Nat.choose numCandidates numHired : ℚ) = probAtLeastOne :=
sorry

end NUMINAMATH_CALUDE_hiring_probability_l3867_386793


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l3867_386761

theorem sufficient_not_necessary :
  (∀ a b c d : ℝ, a > b ∧ c > d → a + c > b + d) ∧
  (∃ a b c d : ℝ, a + c > b + d ∧ ¬(a > b ∧ c > d)) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l3867_386761


namespace NUMINAMATH_CALUDE_weight_of_ten_moles_example_l3867_386744

/-- Calculates the weight of a given number of moles of a compound with a known molecular weight. -/
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Proves that the weight of 10 moles of a compound with a molecular weight of 1080 grams/mole is 10800 grams. -/
theorem weight_of_ten_moles_example : weight_of_compound 10 1080 = 10800 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_ten_moles_example_l3867_386744


namespace NUMINAMATH_CALUDE_sufficient_condition_transitivity_l3867_386738

theorem sufficient_condition_transitivity 
  (C B A : Prop) 
  (h1 : C → B) 
  (h2 : B → A) : 
  C → A := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_transitivity_l3867_386738


namespace NUMINAMATH_CALUDE_evaluate_expression_l3867_386718

theorem evaluate_expression : -(16 / 4 * 11 - 50 + 2^3 * 5) = -34 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3867_386718


namespace NUMINAMATH_CALUDE_polynomial_factor_coefficients_l3867_386714

theorem polynomial_factor_coefficients : 
  ∃ (a b : ℤ), 
    (∃ (d : ℤ), 3 * X ^ 4 + b * X ^ 3 + 45 * X ^ 2 - 21 * X + 8 = 
      (2 * X ^ 2 - 3 * X + 2) * (a * X ^ 2 + d * X + 4)) ∧ 
    a = 3 ∧ 
    b = -27 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_coefficients_l3867_386714


namespace NUMINAMATH_CALUDE_infinite_inscribed_rectangles_l3867_386766

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A rectangle defined by its four vertices -/
structure Rectangle :=
  (A : Point)
  (B : Point)
  (C : Point)
  (D : Point)

/-- Predicate to check if a point lies on a side of a rectangle -/
def PointOnSide (P : Point) (R : Rectangle) : Prop :=
  (P.x = R.A.x ∧ R.A.y ≤ P.y ∧ P.y ≤ R.B.y) ∨
  (P.y = R.B.y ∧ R.B.x ≤ P.x ∧ P.x ≤ R.C.x) ∨
  (P.x = R.C.x ∧ R.C.y ≥ P.y ∧ P.y ≥ R.D.y) ∨
  (P.y = R.D.y ∧ R.D.x ≥ P.x ∧ P.x ≥ R.A.x)

/-- Predicate to check if four points form a rectangle -/
def IsRectangle (E F G H : Point) : Prop :=
  (E.x - F.x) * (G.x - H.x) + (E.y - F.y) * (G.y - H.y) = 0 ∧
  (E.x - H.x) * (F.x - G.x) + (E.y - H.y) * (F.y - G.y) = 0

theorem infinite_inscribed_rectangles (ABCD : Rectangle) :
  ∃ (S : Set (Point × Point × Point × Point)),
    (∀ (E F G H : Point), (E, F, G, H) ∈ S →
      PointOnSide E ABCD ∧ PointOnSide F ABCD ∧
      PointOnSide G ABCD ∧ PointOnSide H ABCD ∧
      IsRectangle E F G H) ∧
    Set.Infinite S :=
  sorry

end NUMINAMATH_CALUDE_infinite_inscribed_rectangles_l3867_386766


namespace NUMINAMATH_CALUDE_micah_typing_speed_l3867_386704

/-- The number of words Isaiah can type per minute. -/
def isaiah_words_per_minute : ℕ := 40

/-- The number of minutes in an hour. -/
def minutes_per_hour : ℕ := 60

/-- The difference in words typed per hour between Isaiah and Micah. -/
def word_difference_per_hour : ℕ := 1200

/-- The number of words Micah can type per minute. -/
def micah_words_per_minute : ℕ := 20

/-- Theorem stating that Micah can type 20 words per minute given the conditions. -/
theorem micah_typing_speed : micah_words_per_minute = 20 := by sorry

end NUMINAMATH_CALUDE_micah_typing_speed_l3867_386704


namespace NUMINAMATH_CALUDE_honzik_payment_l3867_386705

theorem honzik_payment (lollipop_price ice_cream_price : ℕ) : 
  (3 * lollipop_price = 24) →
  (∃ n : ℕ, 2 ≤ n ∧ n ≤ 9 ∧ 4 * lollipop_price + n * ice_cream_price = 109) →
  lollipop_price + ice_cream_price = 19 :=
by sorry

end NUMINAMATH_CALUDE_honzik_payment_l3867_386705


namespace NUMINAMATH_CALUDE_prob_sum_less_than_12_l3867_386740

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where the sum is less than 12 -/
def favorableOutcomes : ℕ := totalOutcomes - 1

/-- The probability of rolling a sum less than 12 with two fair six-sided dice -/
theorem prob_sum_less_than_12 : 
  (favorableOutcomes : ℚ) / totalOutcomes = 35 / 36 := by sorry

end NUMINAMATH_CALUDE_prob_sum_less_than_12_l3867_386740
