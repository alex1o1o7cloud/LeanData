import Mathlib

namespace min_abs_sum_of_quadratic_roots_l1905_190528

theorem min_abs_sum_of_quadratic_roots : ∃ (α β : ℝ), 
  (∀ y : ℝ, y^2 - 6*y + 5 = 0 ↔ y = α ∨ y = β) ∧
  (∀ x : ℝ, |x - α| + |x - β| ≥ 4) ∧
  (∃ x : ℝ, |x - α| + |x - β| = 4) := by
sorry

end min_abs_sum_of_quadratic_roots_l1905_190528


namespace no_integer_solutions_l1905_190518

theorem no_integer_solutions : ¬∃ (x y z : ℤ),
  (x^2 - 3*x*y + 2*y^2 - z^2 = 27) ∧
  (-x^2 + 6*y*z + 2*z^2 = 52) ∧
  (x^2 + x*y + 8*z^2 = 110) := by
  sorry

end no_integer_solutions_l1905_190518


namespace middle_part_of_proportional_division_l1905_190510

theorem middle_part_of_proportional_division (total : ℝ) (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 104 ∧ a = 2 ∧ b = (1 : ℝ) / 2 ∧ c = (1 : ℝ) / 4 →
  ∃ x : ℝ, a * x + b * x + c * x = total ∧ b * x = 20.8 :=
by sorry

end middle_part_of_proportional_division_l1905_190510


namespace sufficient_not_necessary_condition_l1905_190524

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (0 < a ∧ a < b → 1/a > 1/b) ∧
  ∃ a b : ℝ, 1/a > 1/b ∧ ¬(0 < a ∧ a < b) := by
  sorry

end sufficient_not_necessary_condition_l1905_190524


namespace limit_example_l1905_190546

open Real

/-- The limit of (9x^2 - 1) / (x + 1/3) as x approaches -1/3 is -6 -/
theorem limit_example : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 
    0 < |x + 1/3| → |x + 1/3| < δ → 
    |(9*x^2 - 1) / (x + 1/3) + 6| < ε := by
sorry

end limit_example_l1905_190546


namespace expression_evaluation_l1905_190542

theorem expression_evaluation :
  let a : ℚ := 1/3
  let b : ℤ := -1
  4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b) = -3 := by
  sorry

end expression_evaluation_l1905_190542


namespace least_common_solution_l1905_190580

theorem least_common_solution : ∃ x : ℕ, 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 7 ∧ 
  x % 7 = 6 ∧
  (∀ y : ℕ, y > 0 ∧ y % 6 = 5 ∧ y % 8 = 7 ∧ y % 7 = 6 → x ≤ y) ∧
  x = 167 := by
sorry

end least_common_solution_l1905_190580


namespace conference_schedule_ways_l1905_190516

/-- Represents the number of lecturers --/
def n : ℕ := 7

/-- Represents the number of lecturers with specific ordering constraints --/
def k : ℕ := 3

/-- Calculates the number of ways to schedule n lecturers with k lecturers having specific ordering constraints --/
def schedule_ways (n : ℕ) (k : ℕ) : ℕ :=
  (n - k + 1) * Nat.factorial (n - k)

/-- Theorem stating that the number of ways to schedule 7 lecturers with 3 having specific ordering constraints is 600 --/
theorem conference_schedule_ways : schedule_ways n k = 600 := by
  sorry

end conference_schedule_ways_l1905_190516


namespace double_plus_five_positive_l1905_190553

theorem double_plus_five_positive (m : ℝ) :
  (2 * m + 5 > 0) ↔ (∃ x > 0, x = 2 * m + 5) :=
by sorry

end double_plus_five_positive_l1905_190553


namespace baseball_cards_per_page_l1905_190531

theorem baseball_cards_per_page : 
  ∀ (cards_per_page : ℕ+) (full_pages : ℕ+),
  cards_per_page.val * full_pages.val + 1 = 7 →
  cards_per_page = 2 := by
sorry

end baseball_cards_per_page_l1905_190531


namespace road_trip_cost_equalization_l1905_190576

/-- The amount Jamie must give Dana to equalize costs on a road trip -/
theorem road_trip_cost_equalization
  (X Y Z : ℝ)  -- Amounts paid by Alexi, Jamie, and Dana respectively
  (hXY : Y > X)  -- Jamie paid more than Alexi
  (hYZ : Z > Y)  -- Dana paid more than Jamie
  : (X + Z - 2*Y) / 3 = 
    ((X + Y + Z) / 3 - Y)  -- Amount Jamie should give Dana
    := by sorry

end road_trip_cost_equalization_l1905_190576


namespace fifth_month_sales_l1905_190583

def sales_1 : ℕ := 5420
def sales_2 : ℕ := 5660
def sales_3 : ℕ := 6200
def sales_4 : ℕ := 6350
def sales_6 : ℕ := 6470
def average_sale : ℕ := 6100
def num_months : ℕ := 6

theorem fifth_month_sales :
  ∃ (sales_5 : ℕ),
    sales_5 = num_months * average_sale - (sales_1 + sales_2 + sales_3 + sales_4 + sales_6) ∧
    sales_5 = 6500 := by
  sorry

end fifth_month_sales_l1905_190583


namespace stating_calculate_total_applicants_l1905_190592

/-- Represents the proportion of students who applied to first-tier colleges in a sample -/
def sample_proportion (sample_size : ℕ) (applicants_in_sample : ℕ) : ℚ :=
  applicants_in_sample / sample_size

/-- Represents the proportion of students who applied to first-tier colleges in the population -/
def population_proportion (population_size : ℕ) (total_applicants : ℕ) : ℚ :=
  total_applicants / population_size

/-- 
Theorem stating that if the sample proportion equals the population proportion,
then the total number of applicants in the population can be calculated.
-/
theorem calculate_total_applicants 
  (population_size : ℕ) 
  (sample_size : ℕ) 
  (applicants_in_sample : ℕ) 
  (h1 : population_size = 1000)
  (h2 : sample_size = 150)
  (h3 : applicants_in_sample = 60) :
  ∃ (total_applicants : ℕ),
    sample_proportion sample_size applicants_in_sample = 
    population_proportion population_size total_applicants ∧ 
    total_applicants = 400 := by
  sorry

end stating_calculate_total_applicants_l1905_190592


namespace inequality_proof_l1905_190566

open Real

theorem inequality_proof (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x > 0, f x = Real.log x - 3 * x) →
  (∀ x > 0, f x ≤ x * (a * Real.exp x - 4) + b) →
  a + b ≥ 0 := by
    sorry

end inequality_proof_l1905_190566


namespace system_inconsistent_l1905_190506

-- Define the coefficient matrix A
def A : Matrix (Fin 4) (Fin 5) ℚ :=
  !![1, 2, -1, 3, -1;
     2, -1, 3, 1, -1;
     1, -1, 1, 2, 0;
     4, 0, 3, 6, -2]

-- Define the augmented matrix Â
def A_hat : Matrix (Fin 4) (Fin 6) ℚ :=
  !![1, 2, -1, 3, -1, 0;
     2, -1, 3, 1, -1, -1;
     1, -1, 1, 2, 0, 2;
     4, 0, 3, 6, -2, 5]

-- Theorem statement
theorem system_inconsistent :
  Matrix.rank A < Matrix.rank A_hat :=
sorry

end system_inconsistent_l1905_190506


namespace polygon_sides_l1905_190586

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end polygon_sides_l1905_190586


namespace square_fence_poles_l1905_190594

theorem square_fence_poles (poles_per_side : ℕ) (h : poles_per_side = 27) :
  poles_per_side * 4 - 4 = 104 :=
by sorry

end square_fence_poles_l1905_190594


namespace student_distribution_l1905_190588

theorem student_distribution (total : ℕ) (a b : ℕ) : 
  total = 81 →
  a + b = total →
  a = b - 9 →
  a = 36 ∧ b = 45 := by
sorry

end student_distribution_l1905_190588


namespace orange_cost_l1905_190523

theorem orange_cost (calorie_per_orange : ℝ) (total_money : ℝ) (required_calories : ℝ) (money_left : ℝ) :
  calorie_per_orange = 80 →
  total_money = 10 →
  required_calories = 400 →
  money_left = 4 →
  (total_money - money_left) / (required_calories / calorie_per_orange) = 1.2 := by
  sorry

end orange_cost_l1905_190523


namespace fabric_width_l1905_190539

/-- Given a rectangular piece of fabric with area 24 square centimeters and length 8 centimeters,
    prove that its width is 3 centimeters. -/
theorem fabric_width (area : ℝ) (length : ℝ) (width : ℝ) 
    (h1 : area = 24) 
    (h2 : length = 8) 
    (h3 : area = length * width) : width = 3 := by
  sorry

end fabric_width_l1905_190539


namespace christian_age_when_brian_is_40_l1905_190502

/-- Represents a person's age --/
structure Age where
  current : ℕ
  future : ℕ

/-- Represents the ages of Christian and Brian --/
structure AgeRelation where
  christian : Age
  brian : Age
  yearsUntilFuture : ℕ

/-- The conditions of the problem --/
def problemConditions (ages : AgeRelation) : Prop :=
  ages.christian.current = 2 * ages.brian.current ∧
  ages.brian.future = 40 ∧
  ages.christian.future = 72 ∧
  ages.christian.future = ages.christian.current + ages.yearsUntilFuture ∧
  ages.brian.future = ages.brian.current + ages.yearsUntilFuture

/-- The theorem to prove --/
theorem christian_age_when_brian_is_40 (ages : AgeRelation) :
  problemConditions ages → ages.christian.future = 72 := by
  sorry


end christian_age_when_brian_is_40_l1905_190502


namespace main_theorem_l1905_190503

/-- The set of natural numbers with an odd number of 1s in their binary representation up to 2^n - 1 -/
def A (n : ℕ) : Finset ℕ :=
  sorry

/-- The set of natural numbers with an even number of 1s in their binary representation up to 2^n - 1 -/
def B (n : ℕ) : Finset ℕ :=
  sorry

/-- The difference between the sum of nth powers of numbers in A and B -/
def S (n : ℕ) : ℤ :=
  (A n).sum (fun x => x^n) - (B n).sum (fun x => x^n)

/-- The main theorem stating the closed form of S(n) -/
theorem main_theorem (n : ℕ) : S n = (-1)^(n-1) * (n.factorial : ℤ) * 2^(n*(n-1)/2) :=
  sorry

end main_theorem_l1905_190503


namespace triangle_equilateral_l1905_190578

/-- 
Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
if (a+b+c)(b+c-a) = 3bc and sin A = 2sin B cos C, 
then the triangle is equilateral with A = B = C = 60°
-/
theorem triangle_equilateral (a b c A B C : ℝ) : 
  (a + b + c) * (b + c - a) = 3 * b * c →
  Real.sin A = 2 * Real.sin B * Real.cos C →
  A = 60 * π / 180 ∧ B = 60 * π / 180 ∧ C = 60 * π / 180 := by
  sorry


end triangle_equilateral_l1905_190578


namespace square_sum_given_squared_sum_and_product_l1905_190529

theorem square_sum_given_squared_sum_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 49) 
  (h2 : x * y = 10) : 
  x^2 + y^2 = 29 := by
sorry

end square_sum_given_squared_sum_and_product_l1905_190529


namespace max_sum_is_21_l1905_190519

/-- Represents a nonzero digit (1-9) -/
def NonzeroDigit := { d : ℕ // 1 ≤ d ∧ d ≤ 9 }

/-- Calculates An for a given nonzero digit a and positive integer n -/
def An (a : NonzeroDigit) (n : ℕ+) : ℕ :=
  a.val * (10^n.val - 1) / 9

/-- Calculates Bn for a given nonzero digit b and positive integer n -/
def Bn (b : NonzeroDigit) (n : ℕ+) : ℕ :=
  b.val * (10^n.val - 1) / 9

/-- Calculates Cn for a given nonzero digit c and positive integer n -/
def Cn (c : NonzeroDigit) (n : ℕ+) : ℕ :=
  c.val * (10^(n.val + 1) - 1) / 9

/-- Checks if the equation Cn - Bn = An^2 holds for given a, b, c, and n -/
def EquationHolds (a b c : NonzeroDigit) (n : ℕ+) : Prop :=
  Cn c n - Bn b n = (An a n)^2

/-- Checks if there exist at least two distinct positive integers n for which the equation holds -/
def ExistTwoDistinctN (a b c : NonzeroDigit) : Prop :=
  ∃ n₁ n₂ : ℕ+, n₁ ≠ n₂ ∧ EquationHolds a b c n₁ ∧ EquationHolds a b c n₂

/-- The main theorem stating that the maximum value of a + b + c is 21 -/
theorem max_sum_is_21 :
  ∀ a b c : NonzeroDigit,
  ExistTwoDistinctN a b c →
  a.val + b.val + c.val ≤ 21 :=
sorry

end max_sum_is_21_l1905_190519


namespace history_book_pages_l1905_190525

theorem history_book_pages (novel_pages science_pages history_pages : ℕ) : 
  novel_pages = history_pages / 2 →
  science_pages = 4 * novel_pages →
  science_pages = 600 →
  history_pages = 300 := by
sorry

end history_book_pages_l1905_190525


namespace sum_of_divisors_156_l1905_190544

/-- The sum of positive whole number divisors of a natural number n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of positive whole number divisors of 156 is 392 -/
theorem sum_of_divisors_156 : sum_of_divisors 156 = 392 := by sorry

end sum_of_divisors_156_l1905_190544


namespace tan_cot_45_simplification_l1905_190535

theorem tan_cot_45_simplification :
  let tan_45 : ℝ := 1
  let cot_45 : ℝ := 1
  (tan_45^3 + cot_45^3) / (tan_45 + cot_45) = 1 := by
  sorry

end tan_cot_45_simplification_l1905_190535


namespace quadratic_equation_solution_l1905_190537

theorem quadratic_equation_solution (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hmn : m ≠ n) :
  ∃ (a b : ℝ), ∀ (x : ℝ),
    x = a * m + b * n →
    (x + m)^2 - (x + n)^2 = (m - n)^2 →
    a = 0 ∧ b = -1 := by
  sorry

end quadratic_equation_solution_l1905_190537


namespace no_valid_triangle_difference_l1905_190501

theorem no_valid_triangle_difference (n : ℕ) : 
  ((n + 3) * (n + 4)) / 2 - (n * (n + 1)) / 2 ≠ 111 := by
  sorry

end no_valid_triangle_difference_l1905_190501


namespace problem_statement_l1905_190584

-- Define proposition p
def p : Prop := ∀ a : ℝ, a^2 ≥ 0

-- Define function f
def f (x : ℝ) : ℝ := x^2 - x

-- Define proposition q
def q : Prop := ∀ x y : ℝ, 0 < x ∧ x < y → f x < f y

-- Theorem statement
theorem problem_statement : p ∨ q := by sorry

end problem_statement_l1905_190584


namespace problem_1_problem_2_l1905_190568

-- Problem 1
theorem problem_1 (x y : ℝ) : (x - y)^2 + x * (x + 2*y) = 2*x^2 + y^2 := by
  sorry

-- Problem 2
theorem problem_2 (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  ((-3*x + 4) / (x - 1) + x) / ((x - 2) / (x^2 - x)) = x^2 - 2*x := by
  sorry

end problem_1_problem_2_l1905_190568


namespace x_plus_y_value_l1905_190565

theorem x_plus_y_value (x y : ℝ) 
  (h1 : |x| + x + y = 10) 
  (h2 : x + |y| - y = 12) : 
  x + y = 18/5 := by
sorry

end x_plus_y_value_l1905_190565


namespace cubic_equation_solution_l1905_190536

theorem cubic_equation_solution : 
  ∀ x y : ℕ, x^3 - y^3 = x * y + 61 → x = 6 ∧ y = 5 := by
sorry

end cubic_equation_solution_l1905_190536


namespace line_configuration_theorem_l1905_190509

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  n : ℕ  -- number of lines
  total_intersections : ℕ  -- total number of intersection points
  triple_intersections : ℕ  -- number of points where three lines intersect

/-- The theorem statement -/
theorem line_configuration_theorem (config : LineConfiguration) :
  config.n > 0 ∧
  config.total_intersections = 16 ∧
  config.triple_intersections = 6 ∧
  (∀ (i j : ℕ), i < config.n → j < config.n → i ≠ j → ∃ (p : ℕ), p < config.total_intersections) ∧
  (∀ (i j k l : ℕ), i < config.n → j < config.n → k < config.n → l < config.n →
    i ≠ j → i ≠ k → i ≠ l → j ≠ k → j ≠ l → k ≠ l →
    ¬∃ (p : ℕ), p < config.total_intersections) →
  config.n = 8 :=
sorry

end line_configuration_theorem_l1905_190509


namespace special_triangle_sides_special_triangle_right_l1905_190517

/-- A triangle with sides in arithmetic progression and area 6 -/
structure SpecialTriangle where
  a : ℝ
  area : ℝ
  sides_arithmetic : a > 0 ∧ area = 6 ∧ a * (a + 1) * (a + 2) / 4 = area

/-- The sides of the special triangle are 3, 4, and 5 -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 3 ∧ t.a + 1 = 4 ∧ t.a + 2 = 5 :=
sorry

/-- The special triangle is a right triangle -/
theorem special_triangle_right (t : SpecialTriangle) : 
  t.a ^ 2 + (t.a + 1) ^ 2 = (t.a + 2) ^ 2 :=
sorry

end special_triangle_sides_special_triangle_right_l1905_190517


namespace square_area_relation_l1905_190504

theorem square_area_relation (a b : ℝ) : 
  let diagonal_I := 2*a + 3*b
  let area_I := (diagonal_I^2) / 2
  let area_II := 3 * area_I
  area_II = (3 * (2*a + 3*b)^2) / 2 := by sorry

end square_area_relation_l1905_190504


namespace correct_field_equation_l1905_190549

/-- Represents a rectangular field with given area and width-length relationship -/
structure RectangularField where
  area : ℕ
  lengthWidthDiff : ℕ

/-- The equation representing the relationship between length and area for the given field -/
def fieldEquation (field : RectangularField) (x : ℕ) : Prop :=
  x * (x - field.lengthWidthDiff) = field.area

/-- Theorem stating that the equation correctly represents the given field properties -/
theorem correct_field_equation (field : RectangularField) 
    (h1 : field.area = 864) (h2 : field.lengthWidthDiff = 12) :
    ∃ x : ℕ, fieldEquation field x :=
  sorry

end correct_field_equation_l1905_190549


namespace imaginary_part_of_complex_fraction_l1905_190514

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (5 * Complex.I) / (1 + 2 * Complex.I) → Complex.im z = 1 := by
  sorry

end imaginary_part_of_complex_fraction_l1905_190514


namespace ellipse_properties_l1905_190541

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 2 = 1

-- Define the line
def line (m x y : ℝ) : Prop := y = m * (x - 1)

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ellipse p.1 p.2 ∧ line m p.1 p.2}

-- Theorem statement
theorem ellipse_properties :
  -- Part 1: Standard equation of the ellipse
  (∀ x y : ℝ, ellipse x y ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  -- Part 2: Line intersects ellipse at two distinct points
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧ A ≠ B) ∧
  -- Part 3: No real m exists such that the circle with diameter AB passes through origin
  ¬(∃ m : ℝ, ∃ A B : ℝ × ℝ, A ∈ intersection_points m ∧ B ∈ intersection_points m ∧
    A.1 * B.1 + A.2 * B.2 = 0) :=
by sorry

end ellipse_properties_l1905_190541


namespace midpoint_area_in_square_l1905_190590

/-- The area enclosed by midpoints of line segments in a square --/
theorem midpoint_area_in_square (s : ℝ) (h : s = 3) : 
  let midpoint_area := s^2 - (s^2 * Real.pi) / 4
  midpoint_area = 9 - (9 * Real.pi) / 4 := by
  sorry

end midpoint_area_in_square_l1905_190590


namespace geometry_statements_l1905_190589

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (line_in_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (plane_intersection : Plane → Plane → Line)

variable (m n : Line)
variable (α β : Plane)

-- Assume m and n are distinct, α and β are different
variable (h_distinct_lines : m ≠ n)
variable (h_different_planes : α ≠ β)

theorem geometry_statements :
  (parallel_line_plane m α ∧ perpendicular_line_plane n β ∧ parallel_lines m n → perpendicular_planes α β) ∧
  (perpendicular_line_plane m α ∧ parallel_lines m n → perpendicular_line_plane n α) ∧
  ¬(perpendicular_lines m n ∧ line_in_plane n α ∧ line_in_plane m β → perpendicular_planes α β) ∧
  (parallel_line_plane m β ∧ line_in_plane m α ∧ plane_intersection α β = n → parallel_lines m n) :=
by sorry

end geometry_statements_l1905_190589


namespace fraction_equality_implies_cross_product_l1905_190575

theorem fraction_equality_implies_cross_product (x y : ℚ) :
  x / 2 = y / 3 → 3 * x = 2 * y ∧ ¬(2 * x = 3 * y) := by
  sorry

end fraction_equality_implies_cross_product_l1905_190575


namespace power_function_increasing_exponent_l1905_190572

theorem power_function_increasing_exponent (a : ℝ) :
  (∀ x y : ℝ, 0 < x ∧ x < y → x^a < y^a) → a > 0 := by sorry

end power_function_increasing_exponent_l1905_190572


namespace a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b_l1905_190599

theorem a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b (a b : ℝ) :
  a^2 * b > a * b^2 ↔ 1/a < 1/b :=
by sorry

end a_squared_b_gt_ab_squared_iff_one_over_a_lt_one_over_b_l1905_190599


namespace parabola_directrix_l1905_190571

/-- The directrix of the parabola y = -1/4 * x^2 is y = 1 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = -1/4 * x^2 → (∃ (d : ℝ), d = 1 ∧ 
    ∀ (p : ℝ × ℝ), p.2 = -1/4 * p.1^2 → 
      ∃ (f : ℝ), (p.1 - 0)^2 + (p.2 - f)^2 = (p.2 - d)^2) :=
by sorry

end parabola_directrix_l1905_190571


namespace conical_tube_surface_area_l1905_190522

/-- The surface area of a conical tube formed by rolling a semicircular paper. -/
theorem conical_tube_surface_area (r : ℝ) (h : r = 2) : 
  (π * r) = Real.pi * 2 := by
  sorry

end conical_tube_surface_area_l1905_190522


namespace truck_catches_bus_l1905_190550

-- Define the vehicles
structure Vehicle :=
  (speed : ℝ)

-- Define the initial positions
def initial_position (bus truck car : Vehicle) : Prop :=
  truck.speed > car.speed ∧ bus.speed > truck.speed

-- Define the time when car catches up with truck
def car_catches_truck (t : ℝ) : Prop := t = 10

-- Define the time when car catches up with bus
def car_catches_bus (t : ℝ) : Prop := t = 15

-- Theorem to prove
theorem truck_catches_bus 
  (bus truck car : Vehicle) 
  (h1 : initial_position bus truck car)
  (h2 : car_catches_truck 10)
  (h3 : car_catches_bus 15) :
  ∃ (t : ℝ), t = 15 ∧ 
    (truck.speed * (15 + t) = bus.speed * 15) :=
sorry

end truck_catches_bus_l1905_190550


namespace cosine_difference_equals_negative_seven_thousandths_l1905_190532

theorem cosine_difference_equals_negative_seven_thousandths :
  let α := Real.arcsin (3/5)
  let β := Real.arcsin (4/5)
  (Real.cos (3*Real.pi/2 - α/2))^6 - (Real.cos (5*Real.pi/2 + β/2))^6 = -7/1000 := by
sorry

end cosine_difference_equals_negative_seven_thousandths_l1905_190532


namespace slope_angle_of_line_l1905_190596

/-- The slope angle of the line x - y + 1 = 0 is 45 degrees -/
theorem slope_angle_of_line (x y : ℝ) : 
  x - y + 1 = 0 → Real.arctan 1 = π / 4 := by
  sorry

end slope_angle_of_line_l1905_190596


namespace no_hall_with_101_people_l1905_190508

/-- Represents a person in the hall -/
inductive Person
| knight : Person
| liar : Person

/-- Represents the hall with people and their pointing relationships -/
structure Hall :=
  (people : Finset Nat)
  (type : Nat → Person)
  (points_to : Nat → Nat)
  (in_hall : ∀ n, n ∈ people → points_to n ∈ people)
  (all_pointed_at : ∀ n ∈ people, ∃ m ∈ people, points_to m = n)
  (knight_points_to_liar : ∀ n ∈ people, type n = Person.knight → type (points_to n) = Person.liar)
  (liar_points_to_knight : ∀ n ∈ people, type n = Person.liar → type (points_to n) = Person.knight)

/-- Theorem stating that it's impossible to have exactly 101 people in the hall -/
theorem no_hall_with_101_people : ¬ ∃ (h : Hall), Finset.card h.people = 101 := by
  sorry

end no_hall_with_101_people_l1905_190508


namespace train_length_l1905_190540

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 15 → ∃ (length : ℝ), abs (length - 250.05) < 0.01 := by
  sorry


end train_length_l1905_190540


namespace total_eggs_l1905_190577

theorem total_eggs (num_students : ℕ) (eggs_per_student : ℕ) (h1 : num_students = 7) (h2 : eggs_per_student = 8) :
  num_students * eggs_per_student = 56 := by
  sorry

end total_eggs_l1905_190577


namespace apollonius_circle_l1905_190527

/-- The Apollonius Circle Theorem -/
theorem apollonius_circle (x y : ℝ) : 
  let A : ℝ × ℝ := (2, 0)
  let B : ℝ × ℝ := (8, 0)
  let P : ℝ × ℝ := (x, y)
  let dist (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  dist P A / dist P B = 1/2 → x^2 + y^2 = 16 :=
by
  sorry

end apollonius_circle_l1905_190527


namespace s_99_digits_l1905_190595

/-- s(n) is an n-digit number formed by attaching the first n perfect squares, in order, into one integer. -/
def s (n : ℕ) : ℕ := sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The theorem states that s(99) has 189 digits -/
theorem s_99_digits : num_digits (s 99) = 189 := by sorry

end s_99_digits_l1905_190595


namespace probability_of_listening_second_class_l1905_190564

/-- Represents the duration of a class in minutes -/
def class_duration : ℕ := 40

/-- Represents the duration of a break between classes in minutes -/
def break_duration : ℕ := 10

/-- Represents the start time of the first class in minutes after midnight -/
def first_class_start : ℕ := 8 * 60

/-- Represents the earliest arrival time of the student in minutes after midnight -/
def earliest_arrival : ℕ := 9 * 60 + 10

/-- Represents the latest arrival time of the student in minutes after midnight -/
def latest_arrival : ℕ := 10 * 60

/-- Represents the duration of the arrival window in minutes -/
def arrival_window : ℕ := latest_arrival - earliest_arrival

/-- Represents the duration of the favorable arrival window in minutes -/
def favorable_window : ℕ := 10

/-- The probability of the student listening to the second class for no less than 10 minutes -/
theorem probability_of_listening_second_class :
  (favorable_window : ℚ) / arrival_window = 1 / 5 := by sorry

end probability_of_listening_second_class_l1905_190564


namespace broadcast_orders_count_l1905_190520

/-- The number of ways to arrange 6 commercial ads and 2 public service ads 
    with specific constraints -/
def broadcast_orders : ℕ :=
  let n_commercials : ℕ := 6
  let n_public_service : ℕ := 2
  let n_spaces : ℕ := n_commercials - 1
  let ways_to_place_public_service : ℕ := n_spaces * (n_spaces - 2)
  Nat.factorial n_commercials * ways_to_place_public_service

/-- Theorem stating the number of different broadcast orders -/
theorem broadcast_orders_count :
  broadcast_orders = 10800 := by
  sorry

end broadcast_orders_count_l1905_190520


namespace function_symmetry_about_origin_l1905_190574

/-- The function f(x) = x^5 + x^3 is odd, implying symmetry about the origin -/
theorem function_symmetry_about_origin (x : ℝ) : 
  ((-x)^5 + (-x)^3) = -(x^5 + x^3) := by sorry

end function_symmetry_about_origin_l1905_190574


namespace both_hit_probability_l1905_190505

/-- The probability of person A hitting the target -/
def prob_A : ℚ := 8 / 10

/-- The probability of person B hitting the target -/
def prob_B : ℚ := 7 / 10

/-- The theorem stating that the probability of both A and B hitting the target
    is equal to the product of their individual probabilities -/
theorem both_hit_probability :
  (prob_A * prob_B : ℚ) = 14 / 25 := by sorry

end both_hit_probability_l1905_190505


namespace correct_purchase_ways_l1905_190533

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of products they purchase collectively -/
def total_products : ℕ := 3

/-- Function to calculate the number of ways Alpha and Beta can purchase products -/
def purchase_ways : ℕ := sorry

/-- Theorem stating the correct number of ways to purchase products -/
theorem correct_purchase_ways : purchase_ways = 656 := by sorry

end correct_purchase_ways_l1905_190533


namespace root_sum_reciprocal_squares_l1905_190556

theorem root_sum_reciprocal_squares (a b c : ℝ) : 
  a^3 - 6*a^2 + 11*a - 6 = 0 → 
  b^3 - 6*b^2 + 11*b - 6 = 0 → 
  c^3 - 6*c^2 + 11*c - 6 = 0 → 
  a ≠ b → b ≠ c → a ≠ c →
  1/a^2 + 1/b^2 + 1/c^2 = 49/36 := by
sorry

end root_sum_reciprocal_squares_l1905_190556


namespace basket_capacity_l1905_190570

/-- The number of placards taken by each person -/
def placards_per_person : ℕ := 2

/-- The number of people who entered the stadium -/
def people_entered : ℕ := 2317

/-- The total number of placards taken -/
def total_placards : ℕ := people_entered * placards_per_person

theorem basket_capacity : total_placards = 4634 := by
  sorry

end basket_capacity_l1905_190570


namespace sin_seven_halves_pi_plus_theta_l1905_190567

theorem sin_seven_halves_pi_plus_theta (θ : Real) 
  (h : Real.cos (3 * Real.pi + θ) = -(2 * Real.sqrt 2) / 3) : 
  Real.sin ((7 / 2) * Real.pi + θ) = -(2 * Real.sqrt 2) / 3 := by
  sorry

end sin_seven_halves_pi_plus_theta_l1905_190567


namespace courier_package_ratio_l1905_190513

theorem courier_package_ratio : 
  ∀ (total_packages yesterday_packages today_packages : ℕ),
    total_packages = 240 →
    yesterday_packages = 80 →
    total_packages = yesterday_packages + today_packages →
    (today_packages : ℚ) / (yesterday_packages : ℚ) = 2 := by
  sorry

end courier_package_ratio_l1905_190513


namespace sum_of_squares_problem_l1905_190512

theorem sum_of_squares_problem (x y z : ℝ) 
  (nonneg_x : x ≥ 0) (nonneg_y : y ≥ 0) (nonneg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 52)
  (sum_of_products : x*y + y*z + z*x = 27) :
  x + y + z = Real.sqrt 106 := by
  sorry

end sum_of_squares_problem_l1905_190512


namespace three_heads_probability_l1905_190581

/-- The probability of getting heads on a single flip of a fair coin -/
def prob_heads : ℚ := 1/2

/-- The probability of getting three heads in a row when flipping a fair coin -/
def prob_three_heads : ℚ := prob_heads * prob_heads * prob_heads

theorem three_heads_probability : prob_three_heads = 1/8 := by
  sorry

end three_heads_probability_l1905_190581


namespace expansion_coefficient_l1905_190597

/-- The coefficient of x^(3/2) in the expansion of (√x - a/√x)^5 -/
def coefficient_x_3_2 (a : ℝ) : ℝ := 
  (5 : ℝ) * (-a)

theorem expansion_coefficient (a : ℝ) : 
  coefficient_x_3_2 a = 30 → a = -6 := by
sorry

end expansion_coefficient_l1905_190597


namespace min_roots_sum_squared_l1905_190598

/-- Given a quadratic equation x^2 + 2(k+3)x + k^2 + 3 = 0 with real parameter k,
    this function returns the value of (α - 1)^2 + (β - 1)^2,
    where α and β are the roots of the equation. -/
def rootsSumSquared (k : ℝ) : ℝ :=
  2 * (k + 4)^2 - 12

/-- The minimum value of (α - 1)^2 + (β - 1)^2 where α and β are real roots of
    x^2 + 2(k+3)x + k^2 + 3 = 0, and k is a real parameter. -/
theorem min_roots_sum_squared :
  ∃ (m : ℝ), m = 6 ∧ ∀ (k : ℝ), (∀ (x : ℝ), x^2 + 2*(k+3)*x + k^2 + 3 ≥ 0) →
    rootsSumSquared k ≥ m :=
  sorry

end min_roots_sum_squared_l1905_190598


namespace fraction_calculation_l1905_190554

theorem fraction_calculation : (2 / 3 * 4 / 7 * 5 / 8) + 1 / 6 = 17 / 42 := by
  sorry

end fraction_calculation_l1905_190554


namespace arithmetic_sequence_common_difference_l1905_190534

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℚ)  -- arithmetic sequence
  (S : ℕ → ℚ)  -- sum function
  (h1 : a 1 = 2022)  -- first term
  (h2 : S 20 = 22)  -- sum of first 20 terms
  (h3 : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * (a 2 - a 1)))  -- sum formula
  (h4 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1)  -- common difference property
  : a 2 - a 1 = -20209 / 95 := by
  sorry

end arithmetic_sequence_common_difference_l1905_190534


namespace benjamins_house_paintable_area_l1905_190559

/-- Represents the dimensions of a room --/
structure RoomDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the total paintable area in Benjamin's house --/
def total_paintable_area (
  num_bedrooms : ℕ
  ) (room_dims : RoomDimensions)
  (unpaintable_area : ℝ) : ℝ :=
  let wall_area := 2 * (room_dims.length * room_dims.height + room_dims.width * room_dims.height)
  let paintable_area_per_room := wall_area - unpaintable_area
  num_bedrooms * paintable_area_per_room

/-- Theorem stating the total paintable area in Benjamin's house --/
theorem benjamins_house_paintable_area :
  total_paintable_area 4 ⟨14, 12, 9⟩ 70 = 1592 := by
  sorry

end benjamins_house_paintable_area_l1905_190559


namespace existence_of_special_subset_l1905_190585

theorem existence_of_special_subset : 
  ∃ X : Set ℕ+, 
    ∀ n : ℕ+, ∃! (pair : ℕ+ × ℕ+), 
      pair.1 ∈ X ∧ pair.2 ∈ X ∧ n = pair.1 - pair.2 := by
  sorry

end existence_of_special_subset_l1905_190585


namespace ravi_work_time_l1905_190558

-- Define the work completion times
def prakash_time : ℝ := 75
def combined_time : ℝ := 30

-- Define Ravi's time as a variable
def ravi_time : ℝ := 50

-- Theorem statement
theorem ravi_work_time :
  (1 / ravi_time + 1 / prakash_time = 1 / combined_time) →
  ravi_time = 50 := by
  sorry

end ravi_work_time_l1905_190558


namespace at_least_two_primes_in_sequence_l1905_190593

theorem at_least_two_primes_in_sequence : ∃ (m n : ℕ), 
  2 ≤ m ∧ 2 ≤ n ∧ m ≠ n ∧ 
  Nat.Prime (m^3 + m + 1) ∧ 
  Nat.Prime (n^3 + n + 1) :=
sorry

end at_least_two_primes_in_sequence_l1905_190593


namespace factorization_of_4x2_minus_16y2_l1905_190515

theorem factorization_of_4x2_minus_16y2 (x y : ℝ) : 4 * x^2 - 16 * y^2 = 4 * (x + 2*y) * (x - 2*y) := by
  sorry

end factorization_of_4x2_minus_16y2_l1905_190515


namespace sqrt_equation_solution_l1905_190557

theorem sqrt_equation_solution :
  ∃ x : ℝ, 3 * Real.sqrt (x + 15) = 36 ∧ x = 129 :=
by sorry

end sqrt_equation_solution_l1905_190557


namespace registration_scientific_notation_equality_l1905_190560

/-- The number of people registered for the national college entrance examination in 2023 -/
def registration_number : ℕ := 12910000

/-- The scientific notation representation of the registration number -/
def scientific_notation : ℝ := 1.291 * (10 ^ 7)

/-- Theorem stating that the registration number is equal to its scientific notation representation -/
theorem registration_scientific_notation_equality :
  (registration_number : ℝ) = scientific_notation :=
sorry

end registration_scientific_notation_equality_l1905_190560


namespace abs_function_symmetric_about_y_axis_l1905_190579

def f (x : ℝ) : ℝ := |x|

theorem abs_function_symmetric_about_y_axis :
  ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end abs_function_symmetric_about_y_axis_l1905_190579


namespace picture_area_l1905_190511

theorem picture_area (x y : ℕ) (hx : x > 1) (hy : y > 1)
  (h_frame_area : (2 * x + 5) * (y + 4) = 60) : x * y = 6 := by
  sorry

end picture_area_l1905_190511


namespace opposite_angles_equal_l1905_190591

/-- Two angles are opposite if they are formed by two intersecting lines and are not adjacent. -/
def are_opposite_angles (α β : Real) : Prop := sorry

/-- The measure of an angle in radians. -/
def angle_measure (α : Real) : ℝ := sorry

theorem opposite_angles_equal (α β : Real) :
  are_opposite_angles α β → angle_measure α = angle_measure β := by sorry

end opposite_angles_equal_l1905_190591


namespace function_transformation_l1905_190582

theorem function_transformation (f : ℝ → ℝ) :
  (∀ x, f (x - 1) = x^2 + 6*x) →
  (∀ x, f x = x^2 + 8*x + 7) :=
by
  sorry

end function_transformation_l1905_190582


namespace smallest_factor_sum_factorization_exists_l1905_190563

theorem smallest_factor_sum (b : ℤ) : 
  (∃ (p q : ℤ), x^2 + b*x + 2007 = (x + p) * (x + q)) →
  b ≥ 232 :=
by sorry

theorem factorization_exists : 
  ∃ (b p q : ℤ), (b = p + q) ∧ (p * q = 2007) ∧ 
  (x^2 + b*x + 2007 = (x + p) * (x + q)) ∧
  (b = 232) :=
by sorry

end smallest_factor_sum_factorization_exists_l1905_190563


namespace andy_remaining_demerits_l1905_190561

/-- The maximum number of demerits Andy can get in a month before getting fired -/
def max_demerits : ℕ := 50

/-- The number of demerits Andy gets per instance of being late -/
def demerits_per_late : ℕ := 2

/-- The number of times Andy was late -/
def times_late : ℕ := 6

/-- The number of demerits Andy got for making an inappropriate joke -/
def demerits_for_joke : ℕ := 15

/-- The number of additional demerits Andy can get before being fired -/
def remaining_demerits : ℕ := max_demerits - (demerits_per_late * times_late + demerits_for_joke)

theorem andy_remaining_demerits : remaining_demerits = 23 := by
  sorry

end andy_remaining_demerits_l1905_190561


namespace total_cost_proof_l1905_190526

/-- The cost of a single ticket in dollars -/
def ticket_cost : ℝ := 44

/-- The number of tickets purchased -/
def num_tickets : ℕ := 7

/-- The total cost of tickets in dollars -/
def total_cost : ℝ := ticket_cost * num_tickets

theorem total_cost_proof : total_cost = 308 := by
  sorry

end total_cost_proof_l1905_190526


namespace inequality_proof_l1905_190530

theorem inequality_proof (x y z : ℝ) 
  (sum_zero : x + y + z = 0)
  (abs_sum_le_one : |x| + |y| + |z| ≤ 1) :
  x + y/3 + z/5 ≤ 2/5 := by
sorry

end inequality_proof_l1905_190530


namespace quadratic_equation_coefficient_l1905_190545

theorem quadratic_equation_coefficient (a : ℝ) : 
  (∀ x, ∃ y, y = (a - 3) * x^2 - 3 * x - 4) → a ≠ 3 :=
by sorry

end quadratic_equation_coefficient_l1905_190545


namespace geometric_sequence_ratio_l1905_190587

/-- Given a geometric sequence {a_n} with positive terms, where a_1, (1/2)a_3, 2a_2 form an arithmetic sequence,
    prove that (a_8 + a_9) / (a_6 + a_7) = 3 + 2√2 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0)
  (h_geometric : ∃ q : ℝ, ∀ n, a (n + 1) = q * a n)
  (h_arithmetic : ∃ d : ℝ, a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end geometric_sequence_ratio_l1905_190587


namespace gunny_bag_capacity_is_13_tons_l1905_190562

/-- Represents the weight of a packet in pounds -/
def packet_weight : ℚ := 16 + 4 / 16

/-- Represents the number of packets -/
def num_packets : ℕ := 1680

/-- Represents the number of pounds in a ton -/
def pounds_per_ton : ℕ := 2100

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℚ := (num_packets * packet_weight) / pounds_per_ton

theorem gunny_bag_capacity_is_13_tons : gunny_bag_capacity = 13 := by
  sorry

end gunny_bag_capacity_is_13_tons_l1905_190562


namespace additional_planes_needed_l1905_190507

def current_planes : ℕ := 29
def row_size : ℕ := 8

theorem additional_planes_needed :
  (row_size - (current_planes % row_size)) % row_size = 3 := by sorry

end additional_planes_needed_l1905_190507


namespace ultra_high_yield_interest_l1905_190548

/-- The compound interest formula -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- The interest earned from an investment -/
def interest_earned (principal : ℝ) (final_amount : ℝ) : ℝ :=
  final_amount - principal

/-- Theorem: The interest earned on a 500-dollar investment compounded annually at 3% for 10 years is approximately 172 dollars -/
theorem ultra_high_yield_interest :
  let principal : ℝ := 500
  let rate : ℝ := 0.03
  let years : ℕ := 10
  let final_amount := compound_interest principal rate years
  let earned := interest_earned principal final_amount
  ∃ ε > 0, |earned - 172| < ε :=
by sorry

end ultra_high_yield_interest_l1905_190548


namespace sqrt_450_simplification_l1905_190538

theorem sqrt_450_simplification : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end sqrt_450_simplification_l1905_190538


namespace intensity_after_three_plates_l1905_190543

/-- The intensity of light after passing through a number of glass plates -/
def intensity (a : ℝ) (n : ℕ) : ℝ :=
  a * (0.9 ^ n)

/-- Theorem: The intensity of light with original intensity a after passing through 3 glass plates is 0.729a -/
theorem intensity_after_three_plates (a : ℝ) :
  intensity a 3 = 0.729 * a := by
  sorry

end intensity_after_three_plates_l1905_190543


namespace fruit_boxes_distribution_l1905_190551

/-- Given 22 boxes distributed among 3 types of fruits, 
    prove that there must be at least 8 boxes of one type of fruit. -/
theorem fruit_boxes_distribution (boxes : ℕ) (fruit_types : ℕ) 
  (h1 : boxes = 22) (h2 : fruit_types = 3) : 
  ∃ (type : ℕ), type ≤ fruit_types ∧ 
  ∃ (boxes_of_type : ℕ), boxes_of_type ≥ 8 ∧ 
  boxes_of_type ≤ boxes := by
  sorry

end fruit_boxes_distribution_l1905_190551


namespace junior_count_l1905_190547

theorem junior_count (total : ℕ) (junior_percent : ℚ) (senior_percent : ℚ) :
  total = 28 →
  junior_percent = 1/4 →
  senior_percent = 1/10 →
  ∃ (juniors seniors : ℕ),
    juniors + seniors = total ∧
    junior_percent * juniors = senior_percent * seniors ∧
    juniors = 8 := by
  sorry

end junior_count_l1905_190547


namespace N_is_composite_l1905_190573

/-- The number formed by 2n ones -/
def N (n : ℕ) : ℕ := (10^(2*n) - 1) / 9

/-- Theorem: For all natural numbers n ≥ 1, N(n) is composite -/
theorem N_is_composite (n : ℕ) (h : n ≥ 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ N n = a * b := by
  sorry

end N_is_composite_l1905_190573


namespace ellipse_properties_l1905_190500

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- A point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Properties of the ellipse and related points -/
structure EllipseProperties (E : Ellipse) where
  O : Point
  A : Point
  B : Point
  M : Point
  C : Point
  N : Point
  h_O : O.x = 0 ∧ O.y = 0
  h_A : A.x = E.a ∧ A.y = 0
  h_B : B.x = 0 ∧ B.y = E.b
  h_M : M.x = 2 * E.a / 3 ∧ M.y = E.b / 3
  h_OM_slope : (M.y - O.y) / (M.x - O.x) = Real.sqrt 5 / 10
  h_C : C.x = -E.a ∧ C.y = 0
  h_N : N.x = (B.x + C.x) / 2 ∧ N.y = (B.y + C.y) / 2
  h_symmetric : ∃ (S : Point), S.y = 13 / 2 ∧
    (S.x - N.x) * (E.a / E.b + E.b / E.a) = S.y + N.y

/-- The main theorem to prove -/
theorem ellipse_properties (E : Ellipse) (props : EllipseProperties E) :
  (Real.sqrt (E.a^2 - E.b^2) / E.a = 2 * Real.sqrt 5 / 5) ∧
  (E.a^2 = 45 ∧ E.b^2 = 9) :=
sorry

end ellipse_properties_l1905_190500


namespace inequality_reversal_l1905_190569

theorem inequality_reversal (a b c : ℝ) (h1 : a < b) (h2 : c < 0) : ¬(a * c < b * c) := by
  sorry

end inequality_reversal_l1905_190569


namespace inequality_proof_l1905_190555

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^2 + b^2 + c^2 ≥ a*b + b*c + c*a) ∧ ((a + b + c)^2 ≥ 3*(a*b + b*c + c*a)) := by
  sorry

end inequality_proof_l1905_190555


namespace angle_bisector_relation_l1905_190552

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is on the angle bisector of the second and fourth quadrants -/
def isOnAngleBisector (p : Point) : Prop :=
  (p.x < 0 ∧ p.y > 0) ∨ (p.x > 0 ∧ p.y < 0) ∨ p = ⟨0, 0⟩

/-- Theorem stating that for any point on the angle bisector of the second and fourth quadrants, 
    its x-coordinate is the negative of its y-coordinate -/
theorem angle_bisector_relation (p : Point) (h : isOnAngleBisector p) : p.x = -p.y := by
  sorry

end angle_bisector_relation_l1905_190552


namespace oil_needed_calculation_l1905_190521

structure Vehicle where
  cylinders : ℕ
  oil_per_cylinder : ℕ
  oil_in_engine : ℕ

def additional_oil_needed (v : Vehicle) : ℕ :=
  v.cylinders * v.oil_per_cylinder - v.oil_in_engine

def car : Vehicle := {
  cylinders := 6,
  oil_per_cylinder := 8,
  oil_in_engine := 16
}

def truck : Vehicle := {
  cylinders := 8,
  oil_per_cylinder := 10,
  oil_in_engine := 20
}

def motorcycle : Vehicle := {
  cylinders := 4,
  oil_per_cylinder := 6,
  oil_in_engine := 8
}

theorem oil_needed_calculation :
  additional_oil_needed car = 32 ∧
  additional_oil_needed truck = 60 ∧
  additional_oil_needed motorcycle = 16 := by
  sorry

end oil_needed_calculation_l1905_190521
