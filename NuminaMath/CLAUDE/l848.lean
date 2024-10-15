import Mathlib

namespace NUMINAMATH_CALUDE_complex_power_six_l848_84882

theorem complex_power_six : (2 + 3*I : ℂ)^6 = -845 + 2028*I := by sorry

end NUMINAMATH_CALUDE_complex_power_six_l848_84882


namespace NUMINAMATH_CALUDE_estate_area_calculation_l848_84863

/-- Represents the scale of the map in miles per inch -/
def scale : ℚ := 300 / 2

/-- The length of the first side of the rectangle on the map in inches -/
def side1_map : ℚ := 10

/-- The length of the second side of the rectangle on the map in inches -/
def side2_map : ℚ := 6

/-- Converts a length on the map to the actual length in miles -/
def map_to_miles (map_length : ℚ) : ℚ := map_length * scale

/-- Calculates the area of a rectangle given its side lengths -/
def rectangle_area (length width : ℚ) : ℚ := length * width

theorem estate_area_calculation :
  rectangle_area (map_to_miles side1_map) (map_to_miles side2_map) = 1350000 := by
  sorry

end NUMINAMATH_CALUDE_estate_area_calculation_l848_84863


namespace NUMINAMATH_CALUDE_wang_liang_set_exists_l848_84873

theorem wang_liang_set_exists : ∃ (a b : ℕ), 
  1 ≤ a ∧ a ≤ 13 ∧ 
  1 ≤ b ∧ b ≤ 13 ∧ 
  (a - a / b) * b = 24 ∧ 
  (a ≠ 4 ∨ b ≠ 7) := by
  sorry

end NUMINAMATH_CALUDE_wang_liang_set_exists_l848_84873


namespace NUMINAMATH_CALUDE_area_enclosed_by_cosine_and_lines_l848_84879

theorem area_enclosed_by_cosine_and_lines :
  let f (x : ℝ) := Real.cos x
  let a : ℝ := -π/3
  let b : ℝ := π/3
  ∫ x in a..b, f x = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_cosine_and_lines_l848_84879


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l848_84860

theorem fraction_equality_implies_values (a b : ℚ) : 
  (∀ n : ℕ, (1 : ℚ) / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) →
  a = 1 / 2 ∧ b = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l848_84860


namespace NUMINAMATH_CALUDE_battery_current_l848_84806

/-- Given a battery with voltage 48V, prove that when connected to a 12Ω resistance, 
    the resulting current is 4A. -/
theorem battery_current (V R I : ℝ) : 
  V = 48 → R = 12 → I = V / R → I = 4 := by sorry

end NUMINAMATH_CALUDE_battery_current_l848_84806


namespace NUMINAMATH_CALUDE_min_components_for_reliability_l848_84887

/-- The probability of a single component working properly -/
def p : ℝ := 0.5

/-- The minimum required probability for the entire circuit to work properly -/
def min_prob : ℝ := 0.95

/-- The function that calculates the probability of the entire circuit working properly -/
def circuit_prob (n : ℕ) : ℝ := 1 - p^n

/-- Theorem stating the minimum number of components required -/
theorem min_components_for_reliability :
  ∃ n : ℕ, (∀ m : ℕ, m < n → circuit_prob m < min_prob) ∧ circuit_prob n ≥ min_prob :=
sorry

end NUMINAMATH_CALUDE_min_components_for_reliability_l848_84887


namespace NUMINAMATH_CALUDE_min_sum_squares_l848_84816

theorem min_sum_squares (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) :
  ∃ (m : ℝ), m = 1 ∧ ∀ (a b : ℝ), (a + 5)^2 + (b - 12)^2 = 14^2 → x^2 + y^2 ≥ m := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_l848_84816


namespace NUMINAMATH_CALUDE_middle_manager_sample_size_l848_84861

theorem middle_manager_sample_size
  (total_employees : ℕ) (total_middle_managers : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : total_middle_managers = 150)
  (h3 : sample_size = 200) :
  (total_middle_managers : ℚ) / total_employees * sample_size = 30 :=
by sorry

end NUMINAMATH_CALUDE_middle_manager_sample_size_l848_84861


namespace NUMINAMATH_CALUDE_cone_volume_l848_84841

/-- The volume of a cone with slant height 5 cm and height 4 cm is 12π cm³ -/
theorem cone_volume (l h : ℝ) (hl : l = 5) (hh : h = 4) :
  let r := Real.sqrt (l^2 - h^2)
  (1/3 : ℝ) * Real.pi * r^2 * h = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l848_84841


namespace NUMINAMATH_CALUDE_lanas_tulips_l848_84846

/-- The number of tulips Lana picked -/
def tulips : ℕ := sorry

/-- The total number of flowers Lana picked -/
def total_flowers : ℕ := sorry

/-- The number of flowers used for bouquets -/
def used_flowers : ℕ := 70

/-- The number of leftover flowers -/
def leftover_flowers : ℕ := 3

/-- The number of roses Lana picked -/
def roses : ℕ := 37

theorem lanas_tulips :
  (total_flowers = tulips + roses) ∧
  (total_flowers = used_flowers + leftover_flowers) →
  tulips = 36 := by sorry

end NUMINAMATH_CALUDE_lanas_tulips_l848_84846


namespace NUMINAMATH_CALUDE_complex_sum_theorem_l848_84848

theorem complex_sum_theorem :
  let A : ℂ := 3 + 2*I
  let O : ℂ := -1 - 2*I
  let P : ℂ := 2*I
  let S : ℂ := 1 + 3*I
  A - O + P + S = 5 + 9*I :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_theorem_l848_84848


namespace NUMINAMATH_CALUDE_valid_palindrome_count_l848_84812

def valid_digits : Finset Nat := {0, 7, 8, 9}

def is_palindrome (n : Nat) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def count_valid_palindromes : Nat :=
  (valid_digits.filter (· ≠ 0)).card *
  valid_digits.card ^ 2 *
  valid_digits.card ^ 2 *
  valid_digits.card

theorem valid_palindrome_count :
  count_valid_palindromes = 3072 := by sorry

end NUMINAMATH_CALUDE_valid_palindrome_count_l848_84812


namespace NUMINAMATH_CALUDE_min_side_length_l848_84825

theorem min_side_length (AB AC DC BD BC : ℕ) : 
  AB = 7 → AC = 15 → DC = 10 → BD = 25 → BC > 0 →
  (AB + BC > AC) → (AC + BC > AB) →
  (BD + DC > BC) → (BD + BC > DC) →
  BC ≥ 15 ∧ ∃ (BC : ℕ), BC = 15 ∧ 
    AB + BC > AC ∧ AC + BC > AB ∧
    BD + DC > BC ∧ BD + BC > DC :=
by sorry

end NUMINAMATH_CALUDE_min_side_length_l848_84825


namespace NUMINAMATH_CALUDE_expand_and_simplify_l848_84849

theorem expand_and_simplify (a : ℝ) : a * (a + 2) - 2 * a = a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l848_84849


namespace NUMINAMATH_CALUDE_geometric_sequences_l848_84856

def is_geometric (a : ℕ → ℝ) : Prop := ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequences (a : ℕ → ℝ) (h : is_geometric a) :
  (is_geometric (λ n => 1 / a n)) ∧ (is_geometric (λ n => a n * a (n + 1))) := by sorry

end NUMINAMATH_CALUDE_geometric_sequences_l848_84856


namespace NUMINAMATH_CALUDE_elderly_in_sample_l848_84898

/-- Represents the number of employees in each age group -/
structure EmployeeCount where
  young : ℕ
  middle : ℕ
  elderly : ℕ

/-- Represents the sampled employees -/
structure SampledEmployees where
  young : ℕ
  elderly : ℕ

/-- Theorem stating the number of elderly employees in the sample -/
theorem elderly_in_sample
  (total : ℕ)
  (employees : EmployeeCount)
  (sample : SampledEmployees)
  (h1 : total = employees.young + employees.middle + employees.elderly)
  (h2 : employees.young = 160)
  (h3 : employees.middle = 2 * employees.elderly)
  (h4 : total = 430)
  (h5 : sample.young = 32)
  : sample.elderly = 18 := by
  sorry

end NUMINAMATH_CALUDE_elderly_in_sample_l848_84898


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l848_84899

-- Define the function f(x) = ax + b
def f (a b x : ℝ) : ℝ := a * x + b

-- State the theorem
theorem solution_set_of_inequality 
  (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f a b x < f a b y) 
  (h_intersect : f a b 2 = 0) :
  {x : ℝ | b * x^2 - a * x > 0} = {x : ℝ | -1/2 < x ∧ x < 0} := by
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l848_84899


namespace NUMINAMATH_CALUDE_triangle_dance_nine_people_l848_84876

def triangle_dance_rounds (n : ℕ) : ℕ :=
  if n % 3 = 0 then
    (Nat.factorial n) / ((Nat.factorial 3)^3 * Nat.factorial (n / 3))
  else
    0

theorem triangle_dance_nine_people :
  triangle_dance_rounds 9 = 280 :=
by sorry

end NUMINAMATH_CALUDE_triangle_dance_nine_people_l848_84876


namespace NUMINAMATH_CALUDE_total_cantelopes_l848_84818

theorem total_cantelopes (fred_cantelopes tim_cantelopes : ℕ) 
  (h1 : fred_cantelopes = 38) 
  (h2 : tim_cantelopes = 44) : 
  fred_cantelopes + tim_cantelopes = 82 := by
sorry

end NUMINAMATH_CALUDE_total_cantelopes_l848_84818


namespace NUMINAMATH_CALUDE_hou_debang_developed_alkali_process_l848_84815

-- Define a type for scientists
inductive Scientist
| HouDebang
| HouGuangtian
| HouXianglin
| HouXueyu

-- Define a type for chemical processes
structure ChemicalProcess where
  name : String
  developer : Scientist
  developmentDate : String

-- Define the Hou's Alkali Process
def housAlkaliProcess : ChemicalProcess := {
  name := "Hou's Alkali Process",
  developer := Scientist.HouDebang,
  developmentDate := "March 1941"
}

-- Theorem statement
theorem hou_debang_developed_alkali_process :
  housAlkaliProcess.developer = Scientist.HouDebang ∧
  housAlkaliProcess.name = "Hou's Alkali Process" ∧
  housAlkaliProcess.developmentDate = "March 1941" :=
by sorry

end NUMINAMATH_CALUDE_hou_debang_developed_alkali_process_l848_84815


namespace NUMINAMATH_CALUDE_square_side_length_l848_84832

theorem square_side_length (circle_area : ℝ) (h1 : circle_area = 100) :
  let square_perimeter := circle_area
  let square_side := square_perimeter / 4
  square_side = 25 := by sorry

end NUMINAMATH_CALUDE_square_side_length_l848_84832


namespace NUMINAMATH_CALUDE_quadratic_local_symmetry_exponential_local_symmetry_range_l848_84867

/-- A function f has a local symmetry point at x₀ if f(-x₀) = -f(x₀) -/
def has_local_symmetry_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f (-x₀) = -f x₀

/-- Theorem 1: The quadratic function ax² + bx - a has a local symmetry point -/
theorem quadratic_local_symmetry
  (a b : ℝ) (ha : a ≠ 0) :
  ∃ x₀ : ℝ, has_local_symmetry_point (fun x ↦ a * x^2 + b * x - a) x₀ :=
sorry

/-- Theorem 2: Range of m for which 4^x - m * 2^(n+1) + m - 3 has a local symmetry point -/
theorem exponential_local_symmetry_range (n : ℝ) :
  ∃ m_min m_max : ℝ,
    (∀ m : ℝ, (∃ x₀ : ℝ, has_local_symmetry_point (fun x ↦ 4^x - m * 2^(n+1) + m - 3) x₀)
              ↔ m_min ≤ m ∧ m ≤ m_max) ∧
    m_min = 1 - Real.sqrt 3 ∧
    m_max = 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_local_symmetry_exponential_local_symmetry_range_l848_84867


namespace NUMINAMATH_CALUDE_expand_product_l848_84829

theorem expand_product (x : ℝ) : (2*x + 3) * (x + 10) = 2*x^2 + 23*x + 30 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l848_84829


namespace NUMINAMATH_CALUDE_euclidean_division_37_by_5_l848_84813

theorem euclidean_division_37_by_5 :
  ∃ (q r : ℤ), 37 = 5 * q + r ∧ 0 ≤ r ∧ r < 5 ∧ q = 7 ∧ r = 2 :=
by sorry

end NUMINAMATH_CALUDE_euclidean_division_37_by_5_l848_84813


namespace NUMINAMATH_CALUDE_cake_icing_theorem_l848_84803

/-- Represents a rectangular prism cake with icing on specific sides -/
structure CakeWithIcing where
  length : ℕ
  width : ℕ
  height : ℕ
  hasTopIcing : Bool
  hasFrontIcing : Bool
  hasBackIcing : Bool

/-- Counts the number of 1x1x1 cubes with icing on exactly two sides -/
def countCubesWithTwoSidesIced (cake : CakeWithIcing) : ℕ :=
  sorry

/-- The main theorem stating that a 5x5x3 cake with top, front, and back icing
    will have exactly 30 small cubes with icing on two sides when divided into 1x1x1 cubes -/
theorem cake_icing_theorem :
  let cake : CakeWithIcing := {
    length := 5,
    width := 5,
    height := 3,
    hasTopIcing := true,
    hasFrontIcing := true,
    hasBackIcing := true
  }
  countCubesWithTwoSidesIced cake = 30 := by
  sorry

end NUMINAMATH_CALUDE_cake_icing_theorem_l848_84803


namespace NUMINAMATH_CALUDE_least_integer_absolute_value_l848_84884

theorem least_integer_absolute_value (x : ℤ) : (∀ y : ℤ, |3*y + 4| ≤ 18 → y ≥ -7) ∧ |3*(-7) + 4| ≤ 18 := by
  sorry

end NUMINAMATH_CALUDE_least_integer_absolute_value_l848_84884


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l848_84839

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.35 * L) (h2 : (L' * B') / (L * B) = 1.0665) : B' = 0.79 * B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l848_84839


namespace NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l848_84895

theorem divide_by_fraction (a b : ℚ) (hb : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_sixth : 12 / (1 / 6) = 72 := by sorry

end NUMINAMATH_CALUDE_divide_by_fraction_twelve_divided_by_one_sixth_l848_84895


namespace NUMINAMATH_CALUDE_problem_statement_l848_84858

theorem problem_statement (a b : ℝ) (h1 : a - b = 5) (h2 : a * b = 3) :
  (a + 1) * (b - 1) = -3 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l848_84858


namespace NUMINAMATH_CALUDE_expression_evaluation_l848_84880

theorem expression_evaluation : -2^3 + 36 / 3^2 * (-1/2) + |(-5)| = -5 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l848_84880


namespace NUMINAMATH_CALUDE_sum_of_roots_is_eight_l848_84834

/-- A function f is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

/-- The number of real roots of an equation -/
noncomputable def numRealRoots (f : ℝ → ℝ) : ℕ :=
  sorry

/-- Theorem: For an even function f and an odd function g, 
    the sum of the number of real roots of f(f(x)) = 0, f(g(x)) = 0, 
    g(g(x)) = 0, and g(f(x)) = 0 is equal to 8 -/
theorem sum_of_roots_is_eight 
    (f : ℝ → ℝ) (g : ℝ → ℝ) 
    (hf : IsEven f) (hg : IsOdd g) : 
  numRealRoots (λ x => f (f x)) + 
  numRealRoots (λ x => f (g x)) + 
  numRealRoots (λ x => g (g x)) + 
  numRealRoots (λ x => g (f x)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_eight_l848_84834


namespace NUMINAMATH_CALUDE_fifth_stack_cups_l848_84809

def cup_sequence : ℕ → ℕ
  | 0 => 17
  | 1 => 21
  | 2 => 25
  | 3 => 29
  | n + 4 => cup_sequence n + 4

theorem fifth_stack_cups : cup_sequence 4 = 33 := by
  sorry

end NUMINAMATH_CALUDE_fifth_stack_cups_l848_84809


namespace NUMINAMATH_CALUDE_lunks_needed_for_20_apples_l848_84842

/-- Exchange rate between lunks and kunks -/
def lunks_to_kunks (l : ℚ) : ℚ := (1/2) * l

/-- Exchange rate between kunks and apples -/
def kunks_to_apples (k : ℚ) : ℚ := (5/3) * k

/-- The number of lunks required to purchase a given number of apples -/
def lunks_for_apples (a : ℚ) : ℚ := 
  let kunks := (3/5) * a
  (2/4) * kunks

theorem lunks_needed_for_20_apples : 
  lunks_for_apples 20 = 24 := by sorry

end NUMINAMATH_CALUDE_lunks_needed_for_20_apples_l848_84842


namespace NUMINAMATH_CALUDE_window_area_ratio_l848_84836

theorem window_area_ratio (AB : ℝ) (h1 : AB = 40) : 
  let AD : ℝ := 3 / 2 * AB
  let rectangle_area : ℝ := AD * AB
  let semicircle_area : ℝ := π * (AB / 2) ^ 2
  rectangle_area / semicircle_area = 6 / π := by sorry

end NUMINAMATH_CALUDE_window_area_ratio_l848_84836


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_b_l848_84894

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_b_l848_84894


namespace NUMINAMATH_CALUDE_complex_equation_solution_l848_84854

theorem complex_equation_solution (z : ℂ) :
  (1 - Complex.I)^2 * z = 3 + 2 * Complex.I →
  z = -1 + (3/2) * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l848_84854


namespace NUMINAMATH_CALUDE_difference_of_squares_problem_solution_l848_84857

theorem difference_of_squares (k : ℝ) : 
  (5 + k) * (5 - k) = 5^2 - k^2 := by sorry

theorem problem_solution : 
  ∃ n : ℝ, (5 + 2) * (5 - 2) = 5^2 - n ∧ n = 2^2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_problem_solution_l848_84857


namespace NUMINAMATH_CALUDE_calculate_expression_l848_84872

theorem calculate_expression : (1 + Real.pi) ^ 0 + 2 - |(-3)| + 2 * Real.sin (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l848_84872


namespace NUMINAMATH_CALUDE_percentage_increase_income_l848_84874

/-- Calculate the percentage increase in combined weekly income --/
theorem percentage_increase_income (initial_job_income initial_side_income final_job_income final_side_income : ℝ) :
  initial_job_income = 50 →
  initial_side_income = 20 →
  final_job_income = 90 →
  final_side_income = 30 →
  let initial_total := initial_job_income + initial_side_income
  let final_total := final_job_income + final_side_income
  let increase := final_total - initial_total
  let percentage_increase := (increase / initial_total) * 100
  ∀ ε > 0, |percentage_increase - 71.43| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_income_l848_84874


namespace NUMINAMATH_CALUDE_polynomial_multiple_condition_l848_84889

/-- A polynomial f(x) of the form x^4 + p x^2 + q x + a^2 is a multiple of (x^2 - 1) 
    if and only if p = -(a^2 + 1), q = 0, and the other factor is (x^2 - a^2) -/
theorem polynomial_multiple_condition (a : ℝ) :
  ∃ (p q : ℝ), ∀ (x : ℝ), 
    (x^4 + p*x^2 + q*x + a^2 = (x^2 - 1) * (x^2 - a^2)) ↔ 
    (p = -(a^2 + 1) ∧ q = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiple_condition_l848_84889


namespace NUMINAMATH_CALUDE_division_count_correct_l848_84808

def num_couples : ℕ := 5
def first_group_size : ℕ := 6
def min_couples_in_first_group : ℕ := 2

/-- The number of ways to divide 5 couples into two groups, 
    where the first group contains 6 people including at least two couples. -/
def num_divisions : ℕ := 130

theorem division_count_correct : 
  ∀ (n : ℕ) (k : ℕ) (m : ℕ),
  n = num_couples → 
  k = first_group_size → 
  m = min_couples_in_first_group →
  num_divisions = (Nat.choose n 2 * (Nat.choose ((n - 2) * 2) 2 - Nat.choose (n - 2) 1)) + 
                   Nat.choose n 3 :=
by sorry

end NUMINAMATH_CALUDE_division_count_correct_l848_84808


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l848_84865

theorem basketball_lineup_combinations (n : Nat) (k : Nat) (m : Nat) : 
  n = 20 → k = 13 → m = 1 →
  n * Nat.choose (n - 1) (k - m) = 1007760 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l848_84865


namespace NUMINAMATH_CALUDE_updated_mean_after_decrement_l848_84835

def original_mean : ℝ := 250
def num_observations : ℕ := 100
def decrement : ℝ := 20

theorem updated_mean_after_decrement :
  (original_mean * num_observations - decrement * num_observations) / num_observations = 230 := by
  sorry

end NUMINAMATH_CALUDE_updated_mean_after_decrement_l848_84835


namespace NUMINAMATH_CALUDE_range_of_a_for_no_real_roots_l848_84824

theorem range_of_a_for_no_real_roots (a : ℝ) : 
  (∀ x : ℝ, x^2 + (a + 1) * x + 1 > 0) ↔ a ∈ Set.Ioo (-3 : ℝ) 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_for_no_real_roots_l848_84824


namespace NUMINAMATH_CALUDE_vertex_of_quadratic_l848_84852

def f (x : ℝ) : ℝ := -3 * (x - 2)^2

theorem vertex_of_quadratic :
  ∃ (a : ℝ), a < 0 ∧ ∀ (x : ℝ), f x = a * (x - 2)^2 ∧ f 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_vertex_of_quadratic_l848_84852


namespace NUMINAMATH_CALUDE_minimum_computer_units_l848_84855

theorem minimum_computer_units (x : ℕ) : x ≥ 105 ↔ 
  (5500 * 60 + 5000 * (x - 60) > 550000) := by sorry

end NUMINAMATH_CALUDE_minimum_computer_units_l848_84855


namespace NUMINAMATH_CALUDE_original_price_after_percentage_changes_l848_84827

theorem original_price_after_percentage_changes
  (d r s : ℝ) 
  (h1 : 0 < r ∧ r < 100) 
  (h2 : 0 < s ∧ s < 100) 
  (h3 : s < r) :
  let x := (d * 10000) / (10000 + 100 * (r - s) - r * s)
  x * (1 + r / 100) * (1 - s / 100) = d :=
by sorry

end NUMINAMATH_CALUDE_original_price_after_percentage_changes_l848_84827


namespace NUMINAMATH_CALUDE_angle_sum_sine_l848_84851

/-- Given an angle θ with vertex at the origin, initial side on the positive x-axis,
    and terminal side passing through (t, 2t) where t < 0,
    prove that sin(θ + π/3) = -(2√5 + √15)/10 -/
theorem angle_sum_sine (t : ℝ) (θ : ℝ) (h1 : t < 0) 
    (h2 : Real.cos θ = -1 / Real.sqrt 5) 
    (h3 : Real.sin θ = -2 / Real.sqrt 5) : 
  Real.sin (θ + π/3) = -(2 * Real.sqrt 5 + Real.sqrt 15) / 10 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_sine_l848_84851


namespace NUMINAMATH_CALUDE_ali_and_leila_trip_cost_l848_84877

/-- The total cost of a trip for two people with a given original price and discount. -/
def trip_cost (original_price discount : ℕ) : ℕ :=
  2 * (original_price - discount)

/-- Theorem stating that the trip cost for Ali and Leila is $266. -/
theorem ali_and_leila_trip_cost :
  trip_cost 147 14 = 266 := by
  sorry

#eval trip_cost 147 14  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_ali_and_leila_trip_cost_l848_84877


namespace NUMINAMATH_CALUDE_parabola_intersection_distance_l848_84805

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus with inclination angle π/4
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ 
  line A.1 A.2 ∧ line B.1 B.2 ∧
  A ≠ B

-- Theorem statement
theorem parabola_intersection_distance 
  (A B : ℝ × ℝ) 
  (h : intersection_points A B) : 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_distance_l848_84805


namespace NUMINAMATH_CALUDE_hyperbola_m_value_l848_84868

/-- The equation of a hyperbola with one focus at (-3,0) -/
def hyperbola_equation (x y m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

/-- The focus of the hyperbola is at (-3,0) -/
def focus_at_minus_three : ℝ × ℝ := (-3, 0)

/-- Theorem stating that m = 8 for the given hyperbola -/
theorem hyperbola_m_value :
  ∃ (m : ℝ), (∀ (x y : ℝ), hyperbola_equation x y m) ∧ 
  (focus_at_minus_three.1 = -3) ∧ (focus_at_minus_three.2 = 0) →
  m = 8 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_m_value_l848_84868


namespace NUMINAMATH_CALUDE_inequality_proof_l848_84843

theorem inequality_proof (x : ℝ) (h : x ≥ (1/2 : ℝ)) :
  Real.sqrt (9*x + 7) < Real.sqrt x + Real.sqrt (x + 1) + Real.sqrt (x + 2) ∧
  Real.sqrt x + Real.sqrt (x + 1) + Real.sqrt (x + 2) < Real.sqrt (9*x + 9) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l848_84843


namespace NUMINAMATH_CALUDE_cylinder_volume_l848_84866

/-- The volume of a cylinder with height 4 and circular faces with circumference 10π is 100π. -/
theorem cylinder_volume (h : ℝ) (c : ℝ) (v : ℝ) : 
  h = 4 → c = 10 * Real.pi → v = Real.pi * (c / (2 * Real.pi))^2 * h → v = 100 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_l848_84866


namespace NUMINAMATH_CALUDE_trig_identity_l848_84837

theorem trig_identity : 
  2 * Real.sin (50 * π / 180) + 
  Real.sin (10 * π / 180) * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) * 
  Real.sqrt (2 * Real.sin (80 * π / 180) ^ 2) = Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_trig_identity_l848_84837


namespace NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l848_84890

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

/-- The number of prime divisors of 50! is equal to the number of prime numbers less than or equal to 50. -/
theorem prime_divisors_of_50_factorial (p : ℕ → Prop) :
  (∃ (n : ℕ), p n ∧ n ∣ factorial 50) ↔ (∃ (n : ℕ), p n ∧ n ≤ 50) :=
sorry

end NUMINAMATH_CALUDE_prime_divisors_of_50_factorial_l848_84890


namespace NUMINAMATH_CALUDE_binary_digit_difference_l848_84845

/-- Returns the number of digits in the base-2 representation of a natural number -/
def binaryDigits (n : ℕ) : ℕ :=
  if n = 0 then 1 else Nat.log2 n + 1

/-- The difference in the number of digits between the base-2 representations of 1200 and 200 is 3 -/
theorem binary_digit_difference : binaryDigits 1200 - binaryDigits 200 = 3 := by
  sorry

end NUMINAMATH_CALUDE_binary_digit_difference_l848_84845


namespace NUMINAMATH_CALUDE_tan_equality_solution_l848_84819

theorem tan_equality_solution (n : ℤ) :
  -150 < n ∧ n < 150 ∧ Real.tan (n * π / 180) = Real.tan (1340 * π / 180) →
  n = 80 ∨ n = -100 := by
sorry

end NUMINAMATH_CALUDE_tan_equality_solution_l848_84819


namespace NUMINAMATH_CALUDE_geometric_sequence_a5_l848_84822

/-- A geometric sequence with real terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Theorem: In a geometric sequence with a₁ = 1 and a₃ = 2, a₅ = 4 -/
theorem geometric_sequence_a5 (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_a1 : a 1 = 1) 
  (h_a3 : a 3 = 2) : 
  a 5 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a5_l848_84822


namespace NUMINAMATH_CALUDE_triangle_shape_l848_84864

theorem triangle_shape (A B C : Real) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (eq : (Real.cos A + 2 * Real.cos C) / (Real.cos A + 2 * Real.cos B) = Real.sin B / Real.sin C) :
  A = π / 2 ∨ B = C :=
by sorry

end NUMINAMATH_CALUDE_triangle_shape_l848_84864


namespace NUMINAMATH_CALUDE_problem_solution_l848_84897

theorem problem_solution (x y z : ℚ) 
  (h1 : 2*x + y + z = 14)
  (h2 : 2*x + y = 7)
  (h3 : x + 2*y = 10) :
  (x + y - z) / 3 = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l848_84897


namespace NUMINAMATH_CALUDE_quartic_roots_polynomial_problem_l848_84886

theorem quartic_roots_polynomial_problem (a b c d : ℂ) (P : ℂ → ℂ) :
  (a^4 + 4*a^3 + 6*a^2 + 8*a + 10 = 0) →
  (b^4 + 4*b^3 + 6*b^2 + 8*b + 10 = 0) →
  (c^4 + 4*c^3 + 6*c^2 + 8*c + 10 = 0) →
  (d^4 + 4*d^3 + 6*d^2 + 8*d + 10 = 0) →
  (P a = b + c + d) →
  (P b = a + c + d) →
  (P c = a + b + d) →
  (P d = a + b + c) →
  (P (a + b + c + d) = -20) →
  (∀ x, P x = -10/37*x^4 - 30/37*x^3 - 56/37*x^2 - 118/37*x - 148/37) :=
by sorry

end NUMINAMATH_CALUDE_quartic_roots_polynomial_problem_l848_84886


namespace NUMINAMATH_CALUDE_digit_difference_in_base_d_l848_84844

/-- Given two digits A and B in base d > 7 such that AB̅_d + AA̅_d = 172_d, prove that A_d - B_d = 5 --/
theorem digit_difference_in_base_d (d : ℕ) (A B : ℕ) (h1 : d > 7) 
  (h2 : A < d ∧ B < d) 
  (h3 : A * d + B + A * d + A = 1 * d^2 + 7 * d + 2) : 
  A - B = 5 := by
  sorry

end NUMINAMATH_CALUDE_digit_difference_in_base_d_l848_84844


namespace NUMINAMATH_CALUDE_inequality_solution_set_l848_84811

theorem inequality_solution_set :
  {x : ℝ | (|x| + x) * (Real.sin x - 2) < 0} = Set.Ioo 0 Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l848_84811


namespace NUMINAMATH_CALUDE_parabola_focal_line_theorem_l848_84875

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a parabola y^2 = 4x -/
def Parabola := {p : Point2D | p.y^2 = 4 * p.x}

/-- The focal length of the parabola y^2 = 4x -/
def focal_length : ℝ := 2

/-- A line passing through the focus of the parabola -/
structure FocalLine where
  intersects_parabola : Point2D → Point2D → Prop

/-- The length of a line segment between two points -/
def line_segment_length (A B : Point2D) : ℝ := sorry

theorem parabola_focal_line_theorem (l : FocalLine) (A B : Point2D) 
  (h1 : A ∈ Parabola) (h2 : B ∈ Parabola) 
  (h3 : l.intersects_parabola A B) 
  (h4 : (A.x + B.x) / 2 = 3) :
  line_segment_length A B = 8 := by sorry

end NUMINAMATH_CALUDE_parabola_focal_line_theorem_l848_84875


namespace NUMINAMATH_CALUDE_rs_value_l848_84853

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs : 0 < s) 
  (h1 : r^2 + s^2 = 1) (h2 : r^4 + s^4 = 5/8) : r * s = Real.sqrt 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_rs_value_l848_84853


namespace NUMINAMATH_CALUDE_hundredth_term_equals_30503_l848_84804

/-- A sequence of geometric designs -/
def f (n : ℕ) : ℕ := 3 * n^2 + 5 * n + 3

/-- The theorem stating that the 100th term of the sequence equals 30503 -/
theorem hundredth_term_equals_30503 :
  f 0 = 3 ∧ f 1 = 11 ∧ f 2 = 25 ∧ f 3 = 45 → f 100 = 30503 := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_equals_30503_l848_84804


namespace NUMINAMATH_CALUDE_population_after_10_years_l848_84801

/-- Given an initial population and growth rate, calculates the population after n years -/
def population (M : ℝ) (p : ℝ) (n : ℕ) : ℝ := M * (1 + p) ^ n

/-- Theorem: The population after 10 years with initial population M and growth rate p is M(1+p)^10 -/
theorem population_after_10_years (M p : ℝ) : 
  population M p 10 = M * (1 + p)^10 := by
  sorry

end NUMINAMATH_CALUDE_population_after_10_years_l848_84801


namespace NUMINAMATH_CALUDE_original_paint_intensity_l848_84885

/-- Proves that the original paint intensity was 50% given the mixing conditions --/
theorem original_paint_intensity 
  (replaced_fraction : ℚ) 
  (replacement_intensity : ℚ) 
  (final_intensity : ℚ) : 
  replaced_fraction = 2/3 → 
  replacement_intensity = 1/5 → 
  final_intensity = 3/10 → 
  (1 - replaced_fraction) * (1/2) + replaced_fraction * replacement_intensity = final_intensity := by
  sorry

#eval (1 - 2/3) * (1/2) + 2/3 * (1/5) == 3/10

end NUMINAMATH_CALUDE_original_paint_intensity_l848_84885


namespace NUMINAMATH_CALUDE_parallelogram_perimeter_plus_area_l848_84881

/-- A parallelogram with integer coordinates -/
structure Parallelogram where
  v1 : ℤ × ℤ
  v2 : ℤ × ℤ
  v3 : ℤ × ℤ
  v4 : ℤ × ℤ

/-- The perimeter of a parallelogram -/
def perimeter (p : Parallelogram) : ℝ :=
  sorry

/-- The area of a parallelogram -/
def area (p : Parallelogram) : ℝ :=
  sorry

theorem parallelogram_perimeter_plus_area :
  let p : Parallelogram := ⟨(1,1), (6,3), (9,3), (4,1)⟩
  perimeter p + area p = 2 * Real.sqrt 29 + 12 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_perimeter_plus_area_l848_84881


namespace NUMINAMATH_CALUDE_symmetric_points_on_parabola_l848_84814

/-- Given two points on a parabola that are symmetric about a line, prove the value of m -/
theorem symmetric_points_on_parabola (x₁ x₂ y₁ y₂ m : ℝ) : 
  y₁ = 2 * x₁^2 →                   -- A is on the parabola
  y₂ = 2 * x₂^2 →                   -- B is on the parabola
  (y₁ + y₂) / 2 = (x₁ + x₂) / 2 + m →  -- Midpoint of A and B is on y = x + m
  (y₂ - y₁) / (x₂ - x₁) = -1 →      -- Slope of AB is perpendicular to y = x + m
  x₁ * x₂ = -1/2 →                  -- Given condition
  m = 3/2 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_on_parabola_l848_84814


namespace NUMINAMATH_CALUDE_intersection_complement_equal_set_l848_84878

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_equal_set : M ∩ (Set.univ \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_set_l848_84878


namespace NUMINAMATH_CALUDE_no_negative_exponents_l848_84823

theorem no_negative_exponents (a b c d : Int) 
  (h1 : (5 : ℝ)^a + (5 : ℝ)^b = (3 : ℝ)^c + (3 : ℝ)^d)
  (h2 : Even a) (h3 : Even b) (h4 : Even c) (h5 : Even d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_no_negative_exponents_l848_84823


namespace NUMINAMATH_CALUDE_mary_trip_time_and_cost_l848_84833

-- Define the problem parameters
def uber_to_house : ℕ := 10
def uber_cost : ℚ := 15
def airport_time_factor : ℕ := 5
def bag_check_time : ℕ := 15
def luggage_fee_eur : ℚ := 20
def security_time_factor : ℕ := 3
def boarding_wait : ℕ := 20
def takeoff_wait_factor : ℕ := 2
def first_layover : ℕ := 205  -- 3 hours 25 minutes in minutes
def flight_delay : ℕ := 45
def second_layover : ℕ := 110  -- 1 hour 50 minutes in minutes
def time_zone_change : ℕ := 3
def usd_to_eur : ℚ := 0.85
def usd_to_gbp : ℚ := 0.75
def meal_cost_gbp : ℚ := 10

-- Define the theorem
theorem mary_trip_time_and_cost :
  let total_time : ℕ := uber_to_house + (uber_to_house * airport_time_factor) + 
                        bag_check_time + (bag_check_time * security_time_factor) + 
                        boarding_wait + (boarding_wait * takeoff_wait_factor) + 
                        first_layover + flight_delay + second_layover
  let total_time_hours : ℕ := total_time / 60 + time_zone_change
  let total_cost : ℚ := uber_cost + (luggage_fee_eur / usd_to_eur) + (meal_cost_gbp / usd_to_gbp)
  total_time_hours = 12 ∧ total_cost = 51.86 := by sorry

end NUMINAMATH_CALUDE_mary_trip_time_and_cost_l848_84833


namespace NUMINAMATH_CALUDE_complex_square_sum_positive_l848_84840

theorem complex_square_sum_positive (z₁ z₂ z₃ : ℂ) :
  (z₁^2 + z₂^2 : ℂ).re > (-z₃^2 : ℂ).re → (z₁^2 + z₂^2 + z₃^2 : ℂ).re > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_positive_l848_84840


namespace NUMINAMATH_CALUDE_right_angle_complementary_angle_l848_84817

theorem right_angle_complementary_angle (x : ℝ) : 
  x + 23 = 90 → x = 67 := by
  sorry

end NUMINAMATH_CALUDE_right_angle_complementary_angle_l848_84817


namespace NUMINAMATH_CALUDE_simplify_polynomial_expression_l848_84883

theorem simplify_polynomial_expression (x : ℝ) : 
  3 * ((5 * x^2 - 4 * x + 8) - (3 * x^2 - 2 * x + 6)) = 6 * x^2 - 6 * x + 6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_polynomial_expression_l848_84883


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l848_84871

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 9 → y = 53 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l848_84871


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l848_84826

def repeating_decimal_12 : ℚ := 4 / 33
def repeating_decimal_34 : ℚ := 34 / 99

theorem sum_of_repeating_decimals :
  repeating_decimal_12 + repeating_decimal_34 = 46 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l848_84826


namespace NUMINAMATH_CALUDE_min_value_and_range_l848_84820

variable (x y z : ℝ)

def t (x y z : ℝ) : ℝ := x^2 + y^2 + 2*z^2

theorem min_value_and_range :
  (x + y + 2*z = 1) →
  (∃ (min : ℝ), ∀ x y z, t x y z ≥ min ∧ ∃ x y z, t x y z = min) ∧
  (t x y z = 1/2 → 0 ≤ z ∧ z ≤ 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_range_l848_84820


namespace NUMINAMATH_CALUDE_store_display_cans_l848_84830

/-- Represents the number of cans in each layer of the display -/
def canSequence : ℕ → ℚ
  | 0 => 30
  | n + 1 => canSequence n - 3

/-- The number of layers in the display -/
def numLayers : ℕ := 11

/-- The total number of cans in the display -/
def totalCans : ℚ := (numLayers : ℚ) * (canSequence 0 + canSequence (numLayers - 1)) / 2

theorem store_display_cans : totalCans = 170.5 := by
  sorry

end NUMINAMATH_CALUDE_store_display_cans_l848_84830


namespace NUMINAMATH_CALUDE_eight_items_four_categories_l848_84831

/-- The number of ways to assign n distinguishable items to k distinct categories -/
def assignments (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 65536 ways to assign 8 distinguishable items to 4 distinct categories -/
theorem eight_items_four_categories : assignments 8 4 = 65536 := by
  sorry

end NUMINAMATH_CALUDE_eight_items_four_categories_l848_84831


namespace NUMINAMATH_CALUDE_train_length_l848_84870

theorem train_length (train_speed : Real) (bridge_length : Real) (crossing_time : Real) :
  train_speed = 45 * (1000 / 3600) ∧
  bridge_length = 220 ∧
  crossing_time = 30 →
  (train_speed * crossing_time) - bridge_length = 155 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l848_84870


namespace NUMINAMATH_CALUDE_total_views_and_likes_theorem_l848_84859

def total_views_and_likes (
  initial_yt_views : ℕ) (initial_yt_likes : ℕ)
  (initial_other_views : ℕ) (initial_other_likes : ℕ)
  (yt_view_increase_factor : ℕ) (yt_like_increase_factor : ℕ)
  (other_view_increase_factor : ℕ) (other_like_increase_percent : ℕ)
  (additional_yt_views : ℕ) (additional_yt_likes : ℕ)
  (additional_other_views : ℕ) (additional_other_likes : ℕ) : ℕ × ℕ :=
  let yt_views_after_4_days := initial_yt_views + initial_yt_views * yt_view_increase_factor
  let yt_likes_after_4_days := initial_yt_likes + initial_yt_likes * (yt_like_increase_factor - 1)
  let other_views_after_4_days := initial_other_views + initial_other_views * (other_view_increase_factor - 1)
  let other_likes_after_4_days := initial_other_likes + initial_other_likes * other_like_increase_percent / 100
  let final_yt_views := yt_views_after_4_days + additional_yt_views
  let final_yt_likes := yt_likes_after_4_days + additional_yt_likes
  let final_other_views := other_views_after_4_days + additional_other_views
  let final_other_likes := other_likes_after_4_days + additional_other_likes
  (final_yt_views + final_other_views, final_yt_likes + final_other_likes)

theorem total_views_and_likes_theorem :
  total_views_and_likes 4000 500 2000 300 10 3 2 50 50000 2000 30000 500 = (130000, 5250) := by
  sorry

end NUMINAMATH_CALUDE_total_views_and_likes_theorem_l848_84859


namespace NUMINAMATH_CALUDE_problem_parallelogram_area_l848_84800

/-- A parallelogram in 2D space defined by four vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Calculate the area of a parallelogram -/
def area (p : Parallelogram) : ℝ := sorry

/-- The specific parallelogram from the problem -/
def problem_parallelogram : Parallelogram :=
  { v1 := (0, 0)
    v2 := (4, 0)
    v3 := (1, 5)
    v4 := (5, 5) }

/-- Theorem stating that the area of the problem parallelogram is 20 -/
theorem problem_parallelogram_area :
  area problem_parallelogram = 20 := by sorry

end NUMINAMATH_CALUDE_problem_parallelogram_area_l848_84800


namespace NUMINAMATH_CALUDE_enthalpy_relationship_l848_84838

/-- Represents the enthalpy change of a chemical reaction -/
structure EnthalpyChange where
  value : ℝ
  units : String

/-- Represents a chemical reaction with its enthalpy change -/
structure ChemicalReaction where
  equation : String
  enthalpyChange : EnthalpyChange

/-- Given chemical reactions and their enthalpy changes, prove that 2a = b < 0 -/
theorem enthalpy_relationship (
  reaction1 reaction2 reaction3 reaction4 : ChemicalReaction
) (h1 : reaction1.equation = "H₂(g) + ½O₂(g) → H₂O(g)")
  (h2 : reaction2.equation = "2H₂(g) + O₂(g) → 2H₂O(g)")
  (h3 : reaction3.equation = "H₂(g) + ½O₂(g) → H₂O(l)")
  (h4 : reaction4.equation = "2H₂(g) + O₂(g) → 2H₂O(l)")
  (h5 : reaction1.enthalpyChange.units = "KJ·mol⁻¹")
  (h6 : reaction2.enthalpyChange.units = "KJ·mol⁻¹")
  (h7 : reaction3.enthalpyChange.units = "KJ·mol⁻¹")
  (h8 : reaction4.enthalpyChange.units = "KJ·mol⁻¹")
  (h9 : reaction1.enthalpyChange.value = reaction3.enthalpyChange.value)
  (h10 : reaction2.enthalpyChange.value = reaction4.enthalpyChange.value) :
  2 * reaction1.enthalpyChange.value = reaction2.enthalpyChange.value ∧ 
  reaction2.enthalpyChange.value < 0 := by
  sorry


end NUMINAMATH_CALUDE_enthalpy_relationship_l848_84838


namespace NUMINAMATH_CALUDE_parabola_vertex_l848_84896

/-- A parabola is defined by the equation y = -3(x-1)^2 - 2 -/
def parabola (x y : ℝ) : Prop := y = -3 * (x - 1)^2 - 2

/-- The vertex of a parabola is the point where it reaches its maximum or minimum -/
def is_vertex (x y : ℝ) : Prop := parabola x y ∧ ∀ x' y', parabola x' y' → y ≤ y'

/-- The vertex of the parabola y = -3(x-1)^2 - 2 is at (1, -2) -/
theorem parabola_vertex : is_vertex 1 (-2) := by sorry

end NUMINAMATH_CALUDE_parabola_vertex_l848_84896


namespace NUMINAMATH_CALUDE_function_upper_bound_l848_84888

/-- A function satisfying the given inequality condition -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≥ 0 → y ≥ 0 → f x * f y ≤ y^2 * f (x/2) + x^2 * f (y/2)

/-- A function that is bounded on [0,1] -/
def IsBoundedOnUnitInterval (f : ℝ → ℝ) : Prop :=
  ∃ M > 0, ∀ x, 0 ≤ x → x ≤ 1 → |f x| ≤ M

theorem function_upper_bound
    (f : ℝ → ℝ)
    (h1 : SatisfiesInequality f)
    (h2 : IsBoundedOnUnitInterval f) :
    ∀ x, x ≥ 0 → f x ≤ (1/2) * x^2 := by
  sorry

end NUMINAMATH_CALUDE_function_upper_bound_l848_84888


namespace NUMINAMATH_CALUDE_f_range_l848_84807

-- Define the function f
def f (x : ℝ) : ℝ := -(x - 5)^2 + 1

-- Define the domain
def domain : Set ℝ := {x | 2 < x ∧ x < 6}

-- Define the range
def range : Set ℝ := {y | -8 < y ∧ y ≤ 1}

-- Theorem statement
theorem f_range : 
  ∀ y ∈ range, ∃ x ∈ domain, f x = y ∧
  ∀ x ∈ domain, f x ∈ range :=
sorry

end NUMINAMATH_CALUDE_f_range_l848_84807


namespace NUMINAMATH_CALUDE_chess_team_size_l848_84893

theorem chess_team_size (total_students : ℕ) (percentage : ℚ) (team_size : ℕ) : 
  total_students = 160 → percentage = 1/10 → team_size = (total_students : ℚ) * percentage → team_size = 16 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_size_l848_84893


namespace NUMINAMATH_CALUDE_unique_right_triangle_l848_84810

/-- Represents a triangle with sides a, b, and c -/
structure Triangle where
  a : ℕ+
  b : ℕ+
  c : ℕ+

/-- Checks if a triangle is right-angled -/
def Triangle.isRight (t : Triangle) : Prop :=
  t.a ^ 2 + t.b ^ 2 = t.c ^ 2 ∨ t.b ^ 2 + t.c ^ 2 = t.a ^ 2 ∨ t.c ^ 2 + t.a ^ 2 = t.b ^ 2

/-- The main theorem -/
theorem unique_right_triangle :
  ∃! k : ℕ+, 
    (∃ t : Triangle, t.a = 8 ∧ t.b = 12 ∧ t.c = k) ∧ 
    (∃ t : Triangle, t.a + t.b + t.c = 30 ∧ t.a = 8 ∧ t.b = 12 ∧ t.c = k) ∧
    (∃ t : Triangle, t.a = 8 ∧ t.b = 12 ∧ t.c = k ∧ t.isRight) :=
by sorry

end NUMINAMATH_CALUDE_unique_right_triangle_l848_84810


namespace NUMINAMATH_CALUDE_sum_of_digits_of_product_94_nines_94_fours_l848_84862

/-- A number consisting of n repeated digits d -/
def repeatedDigits (n : ℕ) (d : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem sum_of_digits_of_product_94_nines_94_fours :
  sumOfDigits (repeatedDigits 94 9 * repeatedDigits 94 4) = 846 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_product_94_nines_94_fours_l848_84862


namespace NUMINAMATH_CALUDE_banana_arrangements_l848_84869

/-- The number of distinct arrangements of letters in a word -/
def distinctArrangements (totalLetters : ℕ) (letterFrequencies : List ℕ) : ℕ :=
  Nat.factorial totalLetters / (letterFrequencies.map Nat.factorial).prod

/-- The number of distinct arrangements of the letters in "banana" -/
def bananaArrangements : ℕ :=
  distinctArrangements 6 [1, 2, 3]

theorem banana_arrangements :
  bananaArrangements = 60 := by
  sorry

#eval bananaArrangements

end NUMINAMATH_CALUDE_banana_arrangements_l848_84869


namespace NUMINAMATH_CALUDE_point_A_coordinates_l848_84891

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the left -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- Translate a point upwards -/
def translateUp (p : Point) (d : ℝ) : Point :=
  { x := p.x, y := p.y + d }

/-- The theorem stating the coordinates of point A -/
theorem point_A_coordinates (A : Point) 
  (hB : ∃ d : ℝ, translateLeft A d = Point.mk 1 2)
  (hC : ∃ d : ℝ, translateUp A d = Point.mk 3 4) : 
  A = Point.mk 3 2 := by sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l848_84891


namespace NUMINAMATH_CALUDE_asterisk_equation_solution_l848_84828

theorem asterisk_equation_solution :
  ∃! x : ℝ, x > 0 ∧ (x / 20) * (x / 180) = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_asterisk_equation_solution_l848_84828


namespace NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l848_84821

def average_expenditure_jan_to_jun : ℚ := 4200
def january_expenditure : ℚ := 1200
def july_expenditure : ℚ := 1500

theorem average_expenditure_feb_to_jul :
  let total_jan_to_jun := 6 * average_expenditure_jan_to_jun
  let total_feb_to_jun := total_jan_to_jun - january_expenditure
  let total_feb_to_jul := total_feb_to_jun + july_expenditure
  total_feb_to_jul / 6 = 4250 := by sorry

end NUMINAMATH_CALUDE_average_expenditure_feb_to_jul_l848_84821


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l848_84847

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 10 ∧ x * y = 26) → 
  m + n = 108 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l848_84847


namespace NUMINAMATH_CALUDE_tan_negative_23pi_over_3_l848_84892

theorem tan_negative_23pi_over_3 : Real.tan (-23 * Real.pi / 3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_negative_23pi_over_3_l848_84892


namespace NUMINAMATH_CALUDE_minimize_sum_squared_distances_l848_84802

-- Define the points A, B, C
def A : ℝ × ℝ := (3, -1)
def B : ℝ × ℝ := (-1, 4)
def C : ℝ × ℝ := (1, -6)

-- Define the function to calculate the sum of squared distances
def sumSquaredDistances (P : ℝ × ℝ) : ℝ :=
  let (x, y) := P
  (x - A.1)^2 + (y - A.2)^2 +
  (x - B.1)^2 + (y - B.2)^2 +
  (x - C.1)^2 + (y - C.2)^2

-- Define the point P
def P : ℝ × ℝ := (1, -1)

-- Theorem statement
theorem minimize_sum_squared_distances :
  ∀ Q : ℝ × ℝ, sumSquaredDistances P ≤ sumSquaredDistances Q :=
sorry

end NUMINAMATH_CALUDE_minimize_sum_squared_distances_l848_84802


namespace NUMINAMATH_CALUDE_sine_sum_equality_l848_84850

theorem sine_sum_equality : 
  Real.sin (7 * π / 30) + Real.sin (11 * π / 30) - Real.sin (π / 30) - Real.sin (13 * π / 30) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sine_sum_equality_l848_84850
