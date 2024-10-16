import Mathlib

namespace NUMINAMATH_CALUDE_function_value_equality_l1986_198671

theorem function_value_equality (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = 2^x - 5) → f m = 3 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_function_value_equality_l1986_198671


namespace NUMINAMATH_CALUDE_area_of_bounded_region_l1986_198627

/-- The equation of the graph that partitions the plane -/
def graph_equation (x y : ℝ) : Prop :=
  y^2 + 2*x*y + 50*abs x = 500

/-- The bounded region formed by the graph -/
def bounded_region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ (x y : ℝ), p = (x, y) ∧ 
    ((y = 25 - 2*x ∧ x ≥ 0) ∨ (y = -25 - 2*x ∧ x < 0)) ∧
    -25 ≤ y ∧ y ≤ 25}

/-- The area of the bounded region is 1250 -/
theorem area_of_bounded_region :
  MeasureTheory.volume bounded_region = 1250 := by sorry

end NUMINAMATH_CALUDE_area_of_bounded_region_l1986_198627


namespace NUMINAMATH_CALUDE_tan_150_degrees_l1986_198660

theorem tan_150_degrees : Real.tan (150 * π / 180) = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_150_degrees_l1986_198660


namespace NUMINAMATH_CALUDE_general_quadratic_is_quadratic_specific_quadratic_is_quadratic_l1986_198608

/-- Definition of a quadratic equation -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation ax² + bx + c = 0 is quadratic -/
theorem general_quadratic_is_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  is_quadratic_equation (λ x => a * x^2 + b * x + c) :=
sorry

/-- The equation x² - 4 = 0 is quadratic -/
theorem specific_quadratic_is_quadratic :
  is_quadratic_equation (λ x => x^2 - 4) :=
sorry

end NUMINAMATH_CALUDE_general_quadratic_is_quadratic_specific_quadratic_is_quadratic_l1986_198608


namespace NUMINAMATH_CALUDE_unmarked_trees_l1986_198607

def mark_trees (n : ℕ) : Finset ℕ :=
  Finset.filter (fun i => i % 2 = 1 ∨ i % 3 = 1) (Finset.range n)

theorem unmarked_trees :
  (Finset.range 13 \ mark_trees 13).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_unmarked_trees_l1986_198607


namespace NUMINAMATH_CALUDE_triangle_max_third_side_l1986_198691

theorem triangle_max_third_side (a b : ℝ) (ha : a = 4) (hb : b = 9) :
  ∃ (c : ℕ), c ≤ 12 ∧ 
  (∀ (d : ℕ), d > c → ¬(a + b > d ∧ a + d > b ∧ b + d > a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_max_third_side_l1986_198691


namespace NUMINAMATH_CALUDE_probability_mame_on_top_l1986_198653

/-- Represents a folded paper with a total number of sections -/
structure FoldedPaper where
  total_sections : ℕ
  marked_sections : ℕ

/-- The probability of a marked section being on top after random folding -/
def probability_on_top (paper : FoldedPaper) : ℚ :=
  paper.marked_sections / paper.total_sections

theorem probability_mame_on_top :
  let paper : FoldedPaper := { total_sections := 8, marked_sections := 1 }
  probability_on_top paper = 1 / 8 := by
    sorry

end NUMINAMATH_CALUDE_probability_mame_on_top_l1986_198653


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_of_exponents_l1986_198646

/-- Two terms are considered like terms if their variables and corresponding exponents match -/
def are_like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ (a b : ℕ), ∃ (c : ℚ), term1 a b = c * term2 a b ∨ term2 a b = c * term1 a b

/-- The first term in our problem -/
def term1 (m : ℕ) (a b : ℕ) : ℚ := 2 * (a ^ m) * (b ^ 3)

/-- The second term in our problem -/
def term2 (n : ℕ) (a b : ℕ) : ℚ := -3 * a * (b ^ n)

theorem like_terms_imply_sum_of_exponents (m n : ℕ) :
  are_like_terms (term1 m) (term2 n) → m + n = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_of_exponents_l1986_198646


namespace NUMINAMATH_CALUDE_number_reading_and_approximation_l1986_198650

def number : ℕ := 60008205

def read_number (n : ℕ) : String := sorry

def approximate_to_ten_thousands (n : ℕ) : ℕ := sorry

theorem number_reading_and_approximation :
  (read_number number = "sixty million eight thousand two hundred and five") ∧
  (approximate_to_ten_thousands number = 6001) := by sorry

end NUMINAMATH_CALUDE_number_reading_and_approximation_l1986_198650


namespace NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l1986_198644

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between three circles and a line -/
def max_circle_line_intersections : ℕ := 6

/-- The maximum number of intersection points between 3 different circles and 1 straight line -/
theorem max_intersections_three_circles_one_line :
  max_circle_intersections + max_circle_line_intersections = 12 := by sorry

end NUMINAMATH_CALUDE_max_intersections_three_circles_one_line_l1986_198644


namespace NUMINAMATH_CALUDE_area_PQR_approx_5_96_l1986_198675

-- Define the square pyramid
def square_pyramid (side_length : ℝ) (height : ℝ) :=
  {base_side : ℝ // base_side = side_length ∧ height > 0}

-- Define points P, Q, and R
def point_P (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry
def point_Q (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry
def point_R (pyramid : square_pyramid 4 8) : ℝ × ℝ × ℝ := sorry

-- Define the area of triangle PQR
def area_PQR (pyramid : square_pyramid 4 8) : ℝ := sorry

-- Theorem statement
theorem area_PQR_approx_5_96 (pyramid : square_pyramid 4 8) :
  ∃ ε > 0, |area_PQR pyramid - 5.96| < ε :=
sorry

end NUMINAMATH_CALUDE_area_PQR_approx_5_96_l1986_198675


namespace NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l1986_198605

/-- The amount of trout eaten daily by the polar bear in buckets -/
def trout_amount : ℝ := 0.2

/-- The amount of salmon eaten daily by the polar bear in buckets -/
def salmon_amount : ℝ := 0.4

/-- The total amount of fish eaten daily by the polar bear in buckets -/
def total_fish_amount : ℝ := trout_amount + salmon_amount

theorem polar_bear_daily_fish_consumption :
  total_fish_amount = 0.6 := by sorry

end NUMINAMATH_CALUDE_polar_bear_daily_fish_consumption_l1986_198605


namespace NUMINAMATH_CALUDE_line_tangent_to_circle_l1986_198648

theorem line_tangent_to_circle (a : ℝ) : 
  (∀ x y : ℝ, (x - a)^2 + (y - 3)^2 = 8 → y = x + 4 → 
    ∀ x' y' : ℝ, (x' - a)^2 + (y' - 3)^2 < 8 → y' ≠ x' + 4) →
  a = 3 ∨ a = -5 :=
by sorry

end NUMINAMATH_CALUDE_line_tangent_to_circle_l1986_198648


namespace NUMINAMATH_CALUDE_clock_synchronization_l1986_198643

theorem clock_synchronization (arthur_gain oleg_gain cycle : ℕ) 
  (h1 : arthur_gain = 15)
  (h2 : oleg_gain = 12)
  (h3 : cycle = 720) :
  let sync_days := Nat.lcm (cycle / arthur_gain) (cycle / oleg_gain)
  sync_days = 240 ∧ 
  ∀ k : ℕ, k < sync_days → ¬(arthur_gain * k % cycle = 0 ∧ oleg_gain * k % cycle = 0) := by
  sorry

end NUMINAMATH_CALUDE_clock_synchronization_l1986_198643


namespace NUMINAMATH_CALUDE_evaluate_expression_l1986_198677

theorem evaluate_expression (a b : ℕ+) (h : 2^(a:ℕ) * 3^(b:ℕ) = 324) : 
  2^(b:ℕ) * 3^(a:ℕ) = 144 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1986_198677


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l1986_198632

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a/c = b/d = 4/5, then the ratio of their areas is 16/25. -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 4 / 5) (h2 : b / d = 4 / 5) :
  (a * b) / (c * d) = 16 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l1986_198632


namespace NUMINAMATH_CALUDE_stamp_coverage_possible_l1986_198629

/-- Represents a square grid -/
structure Grid (n : ℕ) :=
  (cells : Fin n → Fin n → Bool)

/-- Represents a stamp with black cells -/
structure Stamp (n : ℕ) :=
  (cells : Fin n → Fin n → Bool)
  (black_count : ℕ)
  (black_count_eq : black_count = 102)

/-- Applies a stamp to a grid at a specific position -/
def apply_stamp (g : Grid n) (s : Stamp m) (pos_x pos_y : ℕ) : Grid n :=
  sorry

/-- Checks if a grid is fully covered except for one corner -/
def is_covered_except_corner (g : Grid n) : Prop :=
  sorry

/-- Main theorem: It's possible to cover a 101x101 grid except for one corner
    using a 102-cell stamp 100 times -/
theorem stamp_coverage_possible :
  ∃ (g : Grid 101) (s : Stamp 102) (stamps : List (ℕ × ℕ)),
    stamps.length = 100 ∧
    is_covered_except_corner (stamps.foldl (λ acc (x, y) => apply_stamp acc s x y) g) :=
  sorry

end NUMINAMATH_CALUDE_stamp_coverage_possible_l1986_198629


namespace NUMINAMATH_CALUDE_buffet_dressing_cases_l1986_198658

/-- Represents the number of cases for each type of dressing -/
structure DressingCases where
  ranch : ℕ
  caesar : ℕ
  italian : ℕ
  thousandIsland : ℕ

/-- Checks if the ratios between dressing cases are correct -/
def correctRatios (cases : DressingCases) : Prop :=
  7 * cases.caesar = 2 * cases.ranch ∧
  cases.caesar * 3 = cases.italian ∧
  3 * cases.thousandIsland = 2 * cases.italian

/-- The theorem to be proved -/
theorem buffet_dressing_cases : 
  ∃ (cases : DressingCases), 
    cases.ranch = 28 ∧
    cases.caesar = 8 ∧
    cases.italian = 24 ∧
    cases.thousandIsland = 16 ∧
    correctRatios cases :=
by sorry

end NUMINAMATH_CALUDE_buffet_dressing_cases_l1986_198658


namespace NUMINAMATH_CALUDE_division_of_decimals_l1986_198603

theorem division_of_decimals : (0.08 / 0.002) / 0.04 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_division_of_decimals_l1986_198603


namespace NUMINAMATH_CALUDE_expression_evaluation_l1986_198665

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := -3
  let z : ℝ := 1
  2 * x^2 + 3 * y^2 - 4 * z^2 + 5 * x * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1986_198665


namespace NUMINAMATH_CALUDE_inequality_proof_l1986_198616

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a * b + b * c + c * a = 1) : 
  (a + b) / Real.sqrt (a * b * (1 - a * b)) + 
  (b + c) / Real.sqrt (b * c * (1 - b * c)) + 
  (c + a) / Real.sqrt (c * a * (1 - c * a)) ≤ 
  Real.sqrt 2 / (a * b * c) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1986_198616


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l1986_198633

theorem root_difference_implies_k_value (k : ℝ) : 
  (∃ r s : ℝ, 
    (r^2 + k*r + 12 = 0 ∧ s^2 + k*s + 12 = 0) ∧ 
    ((r+3)^2 - k*(r+3) + 12 = 0 ∧ (s+3)^2 - k*(s+3) + 12 = 0)) 
  → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l1986_198633


namespace NUMINAMATH_CALUDE_no_valid_n_l1986_198699

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l1986_198699


namespace NUMINAMATH_CALUDE_count_five_digit_numbers_with_eight_l1986_198687

/-- The number of five-digit numbers in decimal notation containing at least one digit 8 -/
def fiveDigitNumbersWithEight : ℕ := 37512

/-- The total number of five-digit numbers in decimal notation -/
def totalFiveDigitNumbers : ℕ := 90000

/-- The number of five-digit numbers in decimal notation not containing the digit 8 -/
def fiveDigitNumbersWithoutEight : ℕ := 52488

/-- Theorem stating that the number of five-digit numbers containing at least one digit 8
    is equal to the total number of five-digit numbers minus the number of five-digit numbers
    not containing the digit 8 -/
theorem count_five_digit_numbers_with_eight :
  fiveDigitNumbersWithEight = totalFiveDigitNumbers - fiveDigitNumbersWithoutEight :=
by sorry

end NUMINAMATH_CALUDE_count_five_digit_numbers_with_eight_l1986_198687


namespace NUMINAMATH_CALUDE_sector_area_l1986_198602

theorem sector_area (angle : Real) (radius : Real) : 
  angle = 150 * π / 180 → 
  radius = 2 → 
  (angle * radius^2) / 2 = (5/3) * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l1986_198602


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l1986_198631

noncomputable def f (x : ℝ) : ℝ := -2 * x + x^3

theorem monotonic_decreasing_interval_of_f :
  ∀ x : ℝ, (x > -Real.sqrt 6 / 3 ∧ x < Real.sqrt 6 / 3) ↔ 
    StrictMonoOn f (Set.Ioo (-Real.sqrt 6 / 3) (Real.sqrt 6 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_of_f_l1986_198631


namespace NUMINAMATH_CALUDE_min_sum_of_product_l1986_198639

theorem min_sum_of_product (a b c : ℕ+) (h : a * b * c = 1806) :
  ∃ (x y z : ℕ+), x * y * z = 1806 ∧ x + y + z ≤ a + b + c ∧ x + y + z = 153 :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_l1986_198639


namespace NUMINAMATH_CALUDE_fraction_comparison_l1986_198664

theorem fraction_comparison (a b c d : ℚ) 
  (h1 : a / b < c / d) 
  (h2 : b > d) 
  (h3 : d > 0) : 
  (a + c) / (b + d) < (1 / 2) * (a / b + c / d) := by
  sorry

end NUMINAMATH_CALUDE_fraction_comparison_l1986_198664


namespace NUMINAMATH_CALUDE_parallel_planes_theorem_l1986_198676

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations and operations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (intersect : Line → Line → Set Point)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem parallel_planes_theorem 
  (l m : Line) (α β : Plane) (P : Point) :
  l ≠ m →
  α ≠ β →
  subset l α →
  subset m α →
  intersect l m = {P} →
  parallel l β →
  parallel m β →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_theorem_l1986_198676


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_l1986_198695

def is_solution (x y : ℕ) : Prop := 2 * x + y = 5

theorem non_negative_integer_solutions :
  {p : ℕ × ℕ | is_solution p.1 p.2} = {(0, 5), (1, 3), (2, 1)} := by
  sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_l1986_198695


namespace NUMINAMATH_CALUDE_gary_remaining_money_l1986_198641

/-- Calculates the remaining money after a purchase. -/
def remaining_money (initial : ℕ) (spent : ℕ) : ℕ :=
  initial - spent

/-- Proves that Gary has 18 dollars left after his purchase. -/
theorem gary_remaining_money :
  remaining_money 73 55 = 18 := by
  sorry

end NUMINAMATH_CALUDE_gary_remaining_money_l1986_198641


namespace NUMINAMATH_CALUDE_history_not_statistics_l1986_198645

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 89 →
  history = 36 →
  statistics = 32 →
  history_or_statistics = 59 →
  history - (history + statistics - history_or_statistics) = 27 := by
  sorry

end NUMINAMATH_CALUDE_history_not_statistics_l1986_198645


namespace NUMINAMATH_CALUDE_cat_dressing_probability_l1986_198698

def num_legs : ℕ := 4
def num_clothing_types : ℕ := 3

def probability_correct_order_one_leg : ℚ := 1 / 6

theorem cat_dressing_probability :
  (probability_correct_order_one_leg ^ num_legs : ℚ) = 1 / 1296 := by
  sorry

end NUMINAMATH_CALUDE_cat_dressing_probability_l1986_198698


namespace NUMINAMATH_CALUDE_local_max_implies_a_gt_half_l1986_198694

/-- The function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - a * x^2 + (2*a - 1) * x

/-- The derivative of f(x) -/
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - 2*a*x + 2*a

theorem local_max_implies_a_gt_half (a : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a x ≤ f a 1) →
  a > 1/2 :=
sorry

end NUMINAMATH_CALUDE_local_max_implies_a_gt_half_l1986_198694


namespace NUMINAMATH_CALUDE_product_of_powers_and_primes_l1986_198623

theorem product_of_powers_and_primes :
  2^4 * 3 * 5^3 * 7 * 11 = 2310000 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_and_primes_l1986_198623


namespace NUMINAMATH_CALUDE_milk_for_cookies_l1986_198692

/-- Given the ratio of milk to cookies and the conversion between quarts and pints,
    calculate the amount of milk needed for a different number of cookies. -/
theorem milk_for_cookies (cookies_base : ℕ) (quarts_base : ℕ) (cookies_target : ℕ) :
  cookies_base > 0 →
  quarts_base > 0 →
  cookies_target > 0 →
  (cookies_base = 18 ∧ quarts_base = 3 ∧ cookies_target = 15) →
  (∃ (pints_target : ℚ), pints_target = 5 ∧
    pints_target = (quarts_base * 2 : ℚ) * cookies_target / cookies_base) :=
by
  sorry

#check milk_for_cookies

end NUMINAMATH_CALUDE_milk_for_cookies_l1986_198692


namespace NUMINAMATH_CALUDE_special_triangle_angles_l1986_198681

-- Define a triangle with sides a, b, c and angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the specific triangle
def SpecialTriangle (t : Triangle) : Prop :=
  t.C = 2 * t.A ∧ t.b = 2 * t.a

-- Theorem statement
theorem special_triangle_angles (t : Triangle) 
  (h : SpecialTriangle t) : 
  t.A = 30 ∧ t.B = 90 ∧ t.C = 60 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_angles_l1986_198681


namespace NUMINAMATH_CALUDE_combined_large_cheese_volume_l1986_198622

/-- The volume of a normal rectangular block of cheese in cubic feet -/
def normal_rectangular_volume : ℝ := 4

/-- The volume of a normal cylindrical block of cheese in cubic feet -/
def normal_cylindrical_volume : ℝ := 6

/-- The width multiplier for a large rectangular block -/
def large_rect_width_mult : ℝ := 1.5

/-- The depth multiplier for a large rectangular block -/
def large_rect_depth_mult : ℝ := 3

/-- The length multiplier for a large rectangular block -/
def large_rect_length_mult : ℝ := 2

/-- The radius multiplier for a large cylindrical block -/
def large_cyl_radius_mult : ℝ := 2

/-- The height multiplier for a large cylindrical block -/
def large_cyl_height_mult : ℝ := 3

/-- Theorem stating the combined volume of a large rectangular block and a large cylindrical block -/
theorem combined_large_cheese_volume :
  (normal_rectangular_volume * large_rect_width_mult * large_rect_depth_mult * large_rect_length_mult) +
  (normal_cylindrical_volume * large_cyl_radius_mult^2 * large_cyl_height_mult) = 108 := by
  sorry

end NUMINAMATH_CALUDE_combined_large_cheese_volume_l1986_198622


namespace NUMINAMATH_CALUDE_percentage_five_half_years_or_more_l1986_198686

/-- Represents the number of employees in each time period -/
structure EmployeeDistribution :=
  (less_than_half_year : ℕ)
  (half_to_one_year : ℕ)
  (one_to_one_half_years : ℕ)
  (one_half_to_two_years : ℕ)
  (two_to_two_half_years : ℕ)
  (two_half_to_three_years : ℕ)
  (three_to_three_half_years : ℕ)
  (three_half_to_four_years : ℕ)
  (four_to_four_half_years : ℕ)
  (four_half_to_five_years : ℕ)
  (five_to_five_half_years : ℕ)
  (five_half_to_six_years : ℕ)
  (six_to_six_half_years : ℕ)

/-- Calculates the total number of employees -/
def total_employees (d : EmployeeDistribution) : ℕ :=
  d.less_than_half_year +
  d.half_to_one_year +
  d.one_to_one_half_years +
  d.one_half_to_two_years +
  d.two_to_two_half_years +
  d.two_half_to_three_years +
  d.three_to_three_half_years +
  d.three_half_to_four_years +
  d.four_to_four_half_years +
  d.four_half_to_five_years +
  d.five_to_five_half_years +
  d.five_half_to_six_years +
  d.six_to_six_half_years

/-- Calculates the number of employees working for 5.5 years or more -/
def employees_five_half_years_or_more (d : EmployeeDistribution) : ℕ :=
  d.five_half_to_six_years + d.six_to_six_half_years

/-- Theorem stating that the percentage of employees working for 5.5 years or more is (2/38) * 100 -/
theorem percentage_five_half_years_or_more (d : EmployeeDistribution) 
  (h1 : d.less_than_half_year = 4)
  (h2 : d.half_to_one_year = 6)
  (h3 : d.one_to_one_half_years = 7)
  (h4 : d.one_half_to_two_years = 4)
  (h5 : d.two_to_two_half_years = 3)
  (h6 : d.two_half_to_three_years = 3)
  (h7 : d.three_to_three_half_years = 3)
  (h8 : d.three_half_to_four_years = 2)
  (h9 : d.four_to_four_half_years = 2)
  (h10 : d.four_half_to_five_years = 1)
  (h11 : d.five_to_five_half_years = 1)
  (h12 : d.five_half_to_six_years = 1)
  (h13 : d.six_to_six_half_years = 1) :
  (employees_five_half_years_or_more d : ℚ) / (total_employees d : ℚ) * 100 = 526 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_five_half_years_or_more_l1986_198686


namespace NUMINAMATH_CALUDE_retail_price_calculation_l1986_198626

theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) : 
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ), 
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
  sorry

end NUMINAMATH_CALUDE_retail_price_calculation_l1986_198626


namespace NUMINAMATH_CALUDE_pencils_sold_is_24_l1986_198683

/-- The total number of pencils sold in a school store sale -/
def total_pencils_sold : ℕ :=
  let first_group := 2  -- number of students in the first group
  let second_group := 6 -- number of students in the second group
  let third_group := 2  -- number of students in the third group
  let pencils_first := 2  -- pencils bought by each student in the first group
  let pencils_second := 3 -- pencils bought by each student in the second group
  let pencils_third := 1  -- pencils bought by each student in the third group
  first_group * pencils_first + second_group * pencils_second + third_group * pencils_third

/-- Theorem stating that the total number of pencils sold is 24 -/
theorem pencils_sold_is_24 : total_pencils_sold = 24 := by
  sorry

end NUMINAMATH_CALUDE_pencils_sold_is_24_l1986_198683


namespace NUMINAMATH_CALUDE_wang_trip_distance_l1986_198614

/-- The distance between Mr. Wang's home and location A -/
def distance : ℝ := 330

theorem wang_trip_distance : 
  ∀ x : ℝ, 
  x > 0 → 
  (x / 100 + x / 120) - (x / 150 + 2 * x / 198) = 31 / 60 → 
  x = distance := by
sorry

end NUMINAMATH_CALUDE_wang_trip_distance_l1986_198614


namespace NUMINAMATH_CALUDE_range_of_m_l1986_198666

-- Define the sets P and M
def P : Set ℝ := {x | x^2 ≤ 4}
def M (m : ℝ) : Set ℝ := {m}

-- State the theorem
theorem range_of_m (m : ℝ) : (P ∩ M m = M m) → m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l1986_198666


namespace NUMINAMATH_CALUDE_class_payment_problem_l1986_198649

theorem class_payment_problem (total_students : ℕ) (full_payers : ℕ) (half_payers : ℕ) (total_amount : ℕ) :
  total_students = 25 →
  full_payers = 21 →
  half_payers = 4 →
  total_amount = 1150 →
  ∃ (full_payment : ℕ), 
    full_payment * full_payers + (full_payment / 2) * half_payers = total_amount ∧
    full_payment = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_class_payment_problem_l1986_198649


namespace NUMINAMATH_CALUDE_number_problem_l1986_198674

theorem number_problem (x : ℝ) : 0.4 * x - 11 = 23 → x = 85 := by sorry

end NUMINAMATH_CALUDE_number_problem_l1986_198674


namespace NUMINAMATH_CALUDE_ellipse_dimensions_l1986_198624

/-- An ellipse with foci F₁ and F₂, and a point P on the ellipse. -/
structure Ellipse (a b : ℝ) where
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  P : ℝ × ℝ
  on_ellipse : (P.1 / a) ^ 2 + (P.2 / b) ^ 2 = 1
  perpendicular : (P.1 - F₁.1) * (P.2 - F₂.2) + (P.2 - F₁.2) * (P.1 - F₂.1) = 0
  triangle_area : abs ((F₁.1 - P.1) * (F₂.2 - P.2) - (F₁.2 - P.2) * (F₂.1 - P.1)) / 2 = 9
  triangle_perimeter : dist P F₁ + dist P F₂ + dist F₁ F₂ = 18

/-- The theorem stating that under the given conditions, a = 5 and b = 3 -/
theorem ellipse_dimensions (E : Ellipse a b) : a = 5 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_dimensions_l1986_198624


namespace NUMINAMATH_CALUDE_no_quadratic_trinomial_sequence_with_all_integral_roots_l1986_198662

/-- A sequence of quadratic trinomials -/
def QuadraticTrinomialSequence := ℕ → (ℝ → ℝ)

/-- Condition: P_n is the sum of the two preceding trinomials for n ≥ 3 -/
def IsSumOfPrecedingTrinomials (P : QuadraticTrinomialSequence) : Prop :=
  ∀ n : ℕ, n ≥ 3 → P n = P (n - 1) + P (n - 2)

/-- Condition: P_1 and P_2 do not have common roots -/
def NoCommonRoots (P : QuadraticTrinomialSequence) : Prop :=
  ∀ x : ℝ, P 1 x = 0 → P 2 x ≠ 0

/-- Condition: P_n has at least one integral root for all n -/
def HasIntegralRoot (P : QuadraticTrinomialSequence) : Prop :=
  ∀ n : ℕ, ∃ k : ℤ, P n k = 0

/-- Theorem: There does not exist a sequence of quadratic trinomials satisfying all conditions -/
theorem no_quadratic_trinomial_sequence_with_all_integral_roots :
  ¬ ∃ P : QuadraticTrinomialSequence,
    IsSumOfPrecedingTrinomials P ∧ NoCommonRoots P ∧ HasIntegralRoot P :=
by
  sorry

end NUMINAMATH_CALUDE_no_quadratic_trinomial_sequence_with_all_integral_roots_l1986_198662


namespace NUMINAMATH_CALUDE_different_log_differences_l1986_198636

theorem different_log_differences (primes : Finset ℕ) : 
  primes = {3, 5, 7, 11} → 
  Finset.card (Finset.image (λ (p : ℕ × ℕ) => Real.log p.1 - Real.log p.2) 
    (Finset.filter (λ (p : ℕ × ℕ) => p.1 ≠ p.2) (primes.product primes))) = 12 := by
  sorry

end NUMINAMATH_CALUDE_different_log_differences_l1986_198636


namespace NUMINAMATH_CALUDE_f_2_equals_13_l1986_198682

def a (k n : ℕ) : ℕ := 10^(k+1) + n^3

def b (k n : ℕ) : ℕ := (a k n) / (10^k)

def f (k : ℕ) : ℕ := sorry

theorem f_2_equals_13 : f 2 = 13 := by sorry

end NUMINAMATH_CALUDE_f_2_equals_13_l1986_198682


namespace NUMINAMATH_CALUDE_sum_of_ages_l1986_198642

theorem sum_of_ages (tom_age antonette_age : ℝ) : 
  tom_age = 40.5 → 
  antonette_age = 13.5 → 
  tom_age = 3 * antonette_age → 
  tom_age + antonette_age = 54 := by
sorry

end NUMINAMATH_CALUDE_sum_of_ages_l1986_198642


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l1986_198621

/-- The sum of the first n 9's (e.g., 9, 99, 999, ...) -/
def sumOfNines (n : ℕ) : ℕ := (10^n - 1)

/-- The sum of the digits of a natural number -/
def digitSum (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + digitSum (n / 10)

/-- M is defined as the sum of the first five 9's -/
def M : ℕ := (sumOfNines 1) + (sumOfNines 2) + (sumOfNines 3) + (sumOfNines 4) + (sumOfNines 5)

theorem sum_of_digits_M : digitSum M = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l1986_198621


namespace NUMINAMATH_CALUDE_power_difference_l1986_198679

theorem power_difference (a m n : ℝ) (hm : a ^ m = 3) (hn : a ^ n = 5) : 
  a ^ (m - n) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_power_difference_l1986_198679


namespace NUMINAMATH_CALUDE_problem_solution_l1986_198606

theorem problem_solution (x : ℝ) (h : x^2 - 5*x = 14) :
  (x - 1) * (2*x - 1) - (x + 1)^2 + 1 = 15 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1986_198606


namespace NUMINAMATH_CALUDE_select_four_from_fifteen_l1986_198685

theorem select_four_from_fifteen (n : ℕ) (h : n = 15) :
  (n * (n - 1) * (n - 2) * (n - 3)) = 32760 := by
  sorry

end NUMINAMATH_CALUDE_select_four_from_fifteen_l1986_198685


namespace NUMINAMATH_CALUDE_part_1_part_2_l1986_198657

/-- Definition of set A -/
def A (a : ℝ) : Set ℝ := {a - 3, 2 * a - 1, a^2 + 1}

/-- Definition of set B -/
def B (x : ℝ) : Set ℝ := {0, 1, x}

/-- Theorem for part 1 -/
theorem part_1 (a : ℝ) : -3 ∈ A a → a = 0 ∨ a = -1 := by sorry

/-- Theorem for part 2 -/
theorem part_2 (x : ℝ) : x^2 ∈ B x → x = -1 := by sorry

end NUMINAMATH_CALUDE_part_1_part_2_l1986_198657


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1986_198651

theorem inequality_solution_set (x : ℝ) : 
  (-2 * x^2 + x < -3) ↔ (x < -1 ∨ x > 3/2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1986_198651


namespace NUMINAMATH_CALUDE_no_three_integers_divisibility_l1986_198693

theorem no_three_integers_divisibility : ¬∃ (x y z : ℤ), 
  x > 1 ∧ y > 1 ∧ z > 1 ∧
  (y ∣ (x^2 - 1)) ∧ (z ∣ (x^2 - 1)) ∧
  (x ∣ (y^2 - 1)) ∧ (z ∣ (y^2 - 1)) ∧
  (x ∣ (z^2 - 1)) ∧ (y ∣ (z^2 - 1)) :=
by sorry

end NUMINAMATH_CALUDE_no_three_integers_divisibility_l1986_198693


namespace NUMINAMATH_CALUDE_amount_to_return_l1986_198613

/-- Represents the exchange rate in rubles per dollar -/
def exchange_rate : ℝ := 58.15

/-- Represents the initial deposit in USD -/
def initial_deposit : ℝ := 10000

/-- Calculates the amount to be returned in rubles -/
def amount_in_rubles : ℝ := initial_deposit * exchange_rate

/-- Theorem stating that the amount to be returned is 581,500 rubles -/
theorem amount_to_return : amount_in_rubles = 581500 := by
  sorry

end NUMINAMATH_CALUDE_amount_to_return_l1986_198613


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l1986_198654

/-- Given a class with 100 students where there are 20 more boys than girls,
    prove that the ratio of boys to girls is 3:2. -/
theorem boys_to_girls_ratio (total : ℕ) (difference : ℕ) : 
  total = 100 → difference = 20 → 
  ∃ (boys girls : ℕ), 
    boys + girls = total ∧ 
    boys = girls + difference ∧
    boys / girls = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l1986_198654


namespace NUMINAMATH_CALUDE_valentines_count_l1986_198689

theorem valentines_count (boys girls : ℕ) : 
  boys * girls = boys + girls + 16 → boys * girls = 36 := by
  sorry

end NUMINAMATH_CALUDE_valentines_count_l1986_198689


namespace NUMINAMATH_CALUDE_work_completion_time_l1986_198615

/-- The number of days it takes for A to complete the work alone -/
def days_A : ℝ := 30

/-- The number of days it takes for B to complete the work alone -/
def days_B : ℝ := 55

/-- The number of days it takes for A and B to complete the work together -/
def days_AB : ℝ := 19.411764705882355

theorem work_completion_time :
  (1 / days_A) + (1 / days_B) = (1 / days_AB) := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l1986_198615


namespace NUMINAMATH_CALUDE_symmetry_condition_l1986_198659

/-- The necessary condition for y = 2x to be an axis of symmetry of y = (px + q)/(rx + s) --/
theorem symmetry_condition (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) :
  (∀ x y : ℝ, y = (p * x + q) / (r * x + s) → 
    ∃ x' y', y' = (p * x' + q) / (r * x' + s) ∧ x = y' / 2 ∧ y = x') →
  p + s = 0 := by
sorry

end NUMINAMATH_CALUDE_symmetry_condition_l1986_198659


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l1986_198625

/-- Two real numbers are inversely proportional -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : InverselyProportional x₁ y₁)
  (h2 : InverselyProportional x₂ y₂)
  (h3 : x₁ = 30)
  (h4 : y₁ = 8)
  (h5 : y₂ = 24) :
  x₂ = 10 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l1986_198625


namespace NUMINAMATH_CALUDE_cost_to_fly_D_to_E_l1986_198647

/-- Represents a city in the triangle --/
inductive City
| D
| E
| F

/-- Calculates the cost of flying between two cities --/
def flyCost (distance : ℝ) : ℝ :=
  120 + 0.12 * distance

/-- The triangle formed by the cities --/
structure Triangle where
  DE : ℝ
  DF : ℝ
  isRightAngled : True

/-- The problem setup --/
structure TripProblem where
  cities : Triangle
  flyFromDToE : True

theorem cost_to_fly_D_to_E (problem : TripProblem) : 
  flyCost problem.cities.DE = 660 :=
sorry

end NUMINAMATH_CALUDE_cost_to_fly_D_to_E_l1986_198647


namespace NUMINAMATH_CALUDE_probability_of_event_a_l1986_198673

theorem probability_of_event_a 
  (prob_b : ℝ) 
  (prob_a_and_b : ℝ) 
  (prob_neither_a_nor_b : ℝ) 
  (h1 : prob_b = 0.40)
  (h2 : prob_a_and_b = 0.15)
  (h3 : prob_neither_a_nor_b = 0.5499999999999999) : 
  ∃ (prob_a : ℝ), prob_a = 0.20 := by
  sorry

#check probability_of_event_a

end NUMINAMATH_CALUDE_probability_of_event_a_l1986_198673


namespace NUMINAMATH_CALUDE_smallest_t_value_l1986_198617

theorem smallest_t_value : 
  let f (t : ℝ) := (16*t^2 - 36*t + 15)/(4*t - 3) + 4*t
  ∃ t_min : ℝ, t_min = (51 - Real.sqrt 2073) / 8 ∧
  (∀ t : ℝ, f t = 7*t + 6 → t ≥ t_min) ∧
  (f t_min = 7*t_min + 6) := by
sorry

end NUMINAMATH_CALUDE_smallest_t_value_l1986_198617


namespace NUMINAMATH_CALUDE_total_water_filled_jars_l1986_198618

/-- Represents the number of jars of each size -/
def num_jars_per_size : ℚ := 20

/-- Represents the total number of jar sizes -/
def num_jar_sizes : ℕ := 3

/-- Represents the total volume of water in gallons -/
def total_water : ℚ := 35

/-- Theorem stating the total number of water-filled jars -/
theorem total_water_filled_jars : 
  (1/4 + 1/2 + 1) * num_jars_per_size = total_water ∧ 
  num_jars_per_size * num_jar_sizes = 60 := by
  sorry

#check total_water_filled_jars

end NUMINAMATH_CALUDE_total_water_filled_jars_l1986_198618


namespace NUMINAMATH_CALUDE_parabola_c_is_one_l1986_198670

/-- A parabola with equation y = 2x^2 + c and vertex at (0, 1) -/
structure Parabola where
  c : ℝ
  vertex_x : ℝ
  vertex_y : ℝ
  eq_vertex : vertex_y = 2 * vertex_x^2 + c
  is_vertex_zero_one : vertex_x = 0 ∧ vertex_y = 1

/-- The value of c for a parabola with equation y = 2x^2 + c and vertex at (0, 1) is 1 -/
theorem parabola_c_is_one (p : Parabola) : p.c = 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_c_is_one_l1986_198670


namespace NUMINAMATH_CALUDE_at_least_one_genuine_certain_l1986_198619

theorem at_least_one_genuine_certain (total : ℕ) (genuine : ℕ) (defective : ℕ) (selected : ℕ)
  (h1 : total = 8)
  (h2 : genuine = 5)
  (h3 : defective = 3)
  (h4 : total = genuine + defective)
  (h5 : selected = 4) :
  ∀ (selection : Finset (Fin total)),
    selection.card = selected →
    ∃ (i : Fin total), i ∈ selection ∧ i.val < genuine :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_genuine_certain_l1986_198619


namespace NUMINAMATH_CALUDE_otimes_sqrt_two_otimes_sum_zero_l1986_198628

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (1 - b)

-- Theorem 1
theorem otimes_sqrt_two : otimes (1 + Real.sqrt 2) (Real.sqrt 2) = -1 := by sorry

-- Theorem 2
theorem otimes_sum_zero (a b : ℝ) : a + b = 0 → otimes a a + otimes b b = 2 * a * b := by sorry

end NUMINAMATH_CALUDE_otimes_sqrt_two_otimes_sum_zero_l1986_198628


namespace NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1986_198680

theorem smallest_solution_abs_equation :
  ∃ (x : ℝ), x * |x| = 3 * x + 2 ∧
  (∀ (y : ℝ), y * |y| = 3 * y + 2 → x ≤ y) ∧
  x = -2 := by
  sorry

end NUMINAMATH_CALUDE_smallest_solution_abs_equation_l1986_198680


namespace NUMINAMATH_CALUDE_smallest_divisible_addition_l1986_198684

theorem smallest_divisible_addition (n : ℕ) : 
  (∀ m : ℕ, m < n → ¬((2013 + m) % 11 = 0 ∧ (2013 + m) % 13 = 0)) ∧
  (2013 + n) % 11 = 0 ∧ (2013 + n) % 13 = 0 →
  n = 132 := by
sorry

end NUMINAMATH_CALUDE_smallest_divisible_addition_l1986_198684


namespace NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l1986_198638

theorem abs_sum_zero_implies_sum (a b : ℝ) : 
  |a - 2| + |b + 3| = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_abs_sum_zero_implies_sum_l1986_198638


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1986_198696

theorem gcd_of_three_numbers :
  Nat.gcd 45321 (Nat.gcd 76543 123456) = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1986_198696


namespace NUMINAMATH_CALUDE_compare_sqrt_expressions_l1986_198655

theorem compare_sqrt_expressions : 2 * Real.sqrt 3 < 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_compare_sqrt_expressions_l1986_198655


namespace NUMINAMATH_CALUDE_subset_count_divisible_by_prime_l1986_198630

theorem subset_count_divisible_by_prime (p : Nat) (hp : Nat.Prime p) (hp_odd : Odd p) :
  let S := Finset.range (2 * p)
  (Finset.filter (fun A : Finset Nat =>
    A.card = p ∧ (A.sum id) % p = 0) (Finset.powerset S)).card =
  (1 / p) * (Nat.choose (2 * p) p - 2) + 2 := by
  sorry

end NUMINAMATH_CALUDE_subset_count_divisible_by_prime_l1986_198630


namespace NUMINAMATH_CALUDE_magnitude_of_z_plus_two_l1986_198669

/-- Given a complex number z = (1+i)/i, prove that the magnitude of z+2 is √10 -/
theorem magnitude_of_z_plus_two (z : ℂ) : z = (1 + Complex.I) / Complex.I → Complex.abs (z + 2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_plus_two_l1986_198669


namespace NUMINAMATH_CALUDE_box_width_calculation_l1986_198672

/-- Given a rectangular box with specified dimensions and features, calculate its width -/
theorem box_width_calculation (length : ℝ) (road_width : ℝ) (lawn_area : ℝ) : 
  length = 60 →
  road_width = 3 →
  lawn_area = 2109 →
  ∃ (width : ℝ), width = 37.15 ∧ length * width - 2 * (length / 3) * road_width = lawn_area :=
by sorry

end NUMINAMATH_CALUDE_box_width_calculation_l1986_198672


namespace NUMINAMATH_CALUDE_polygon_sides_count_l1986_198634

/-- A convex polygon with the sum of all angles except one equal to 2970° has 19 sides. -/
theorem polygon_sides_count (n : ℕ) (sum_angles : ℝ) (h1 : sum_angles = 2970) : n = 19 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l1986_198634


namespace NUMINAMATH_CALUDE_gift_wrapping_problem_l1986_198635

/-- Given three rolls of wrapping paper where the first roll wraps 3 gifts,
    the second roll wraps 5 gifts, and the third roll wraps 4 gifts with no paper leftover,
    prove that the total number of gifts wrapped is 12. -/
theorem gift_wrapping_problem (rolls : Nat) (first_roll : Nat) (second_roll : Nat) (third_roll : Nat)
    (h1 : rolls = 3)
    (h2 : first_roll = 3)
    (h3 : second_roll = 5)
    (h4 : third_roll = 4)
    (h5 : rolls * first_roll ≥ first_roll + second_roll + third_roll) :
    first_roll + second_roll + third_roll = 12 := by
  sorry

end NUMINAMATH_CALUDE_gift_wrapping_problem_l1986_198635


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l1986_198637

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  units : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_units : units ≥ 0 ∧ units ≤ 9

/-- The conditions for the three-digit number -/
def satisfiesConditions (n : ThreeDigitNumber) : Prop :=
  n.units + n.hundreds = n.tens ∧
  7 * n.hundreds = n.units + n.tens + 2 ∧
  n.units + n.tens + n.hundreds = 14

/-- The theorem stating that 275 is the only three-digit number satisfying the conditions -/
theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber, satisfiesConditions n ∧ 
    n.hundreds = 2 ∧ n.tens = 7 ∧ n.units = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l1986_198637


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1986_198611

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1986_198611


namespace NUMINAMATH_CALUDE_sum_of_coordinates_A_l1986_198620

/-- Given three points A, B, and C in a 2D plane satisfying specific conditions,
    prove that the sum of the coordinates of A is 3. -/
theorem sum_of_coordinates_A (A B C : ℝ × ℝ) : 
  (C.1 - A.1) / (B.1 - A.1) = 1/2 →
  (C.2 - A.2) / (B.2 - A.2) = 1/2 →
  B = (2, 5) →
  C = (6, -3) →
  A.1 + A.2 = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_A_l1986_198620


namespace NUMINAMATH_CALUDE_second_employee_hourly_rate_l1986_198668

/-- Proves that the hourly rate of the second employee before subsidy is $22 -/
theorem second_employee_hourly_rate 
  (first_employee_rate : ℝ)
  (subsidy : ℝ)
  (weekly_savings : ℝ)
  (hours_per_week : ℝ)
  (h1 : first_employee_rate = 20)
  (h2 : subsidy = 6)
  (h3 : weekly_savings = 160)
  (h4 : hours_per_week = 40)
  : ∃ (second_employee_rate : ℝ), 
    hours_per_week * first_employee_rate - hours_per_week * (second_employee_rate - subsidy) = weekly_savings ∧ 
    second_employee_rate = 22 :=
by sorry

end NUMINAMATH_CALUDE_second_employee_hourly_rate_l1986_198668


namespace NUMINAMATH_CALUDE_logarithm_equality_l1986_198688

theorem logarithm_equality (c d : ℝ) : 
  c = Real.log 400 / Real.log 4 → d = Real.log 20 / Real.log 2 → c = d := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l1986_198688


namespace NUMINAMATH_CALUDE_probability_all_different_at_most_one_odd_l1986_198610

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 3

/-- The total number of possible outcomes when rolling three dice -/
def totalOutcomes : ℕ := numFaces ^ numDice

/-- The number of favorable outcomes (all different numbers with at most one odd) -/
def favorableOutcomes : ℕ := 60

/-- The probability of rolling three dice and getting all different numbers with at most one odd number -/
def probabilityAllDifferentAtMostOneOdd : ℚ := favorableOutcomes / totalOutcomes

theorem probability_all_different_at_most_one_odd :
  probabilityAllDifferentAtMostOneOdd = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_probability_all_different_at_most_one_odd_l1986_198610


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1986_198652

theorem quadratic_equation_root (b : ℝ) : 
  (2 : ℝ) ^ 2 * 2 + b * 2 - 4 = 0 → b = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1986_198652


namespace NUMINAMATH_CALUDE_ten_teams_in_tournament_l1986_198601

/-- The number of games played in a round-robin tournament with n teams -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 45 games, there were 10 teams -/
theorem ten_teams_in_tournament (h : games_played 10 = 45) : 
  ∃ (n : ℕ), n = 10 ∧ games_played n = 45 :=
by sorry

end NUMINAMATH_CALUDE_ten_teams_in_tournament_l1986_198601


namespace NUMINAMATH_CALUDE_fraction_simplification_l1986_198612

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1986_198612


namespace NUMINAMATH_CALUDE_c_investment_value_l1986_198678

/-- Represents the investment and profit distribution in a partnership business --/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that given the conditions of the problem, c's investment is $72,000 --/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 45000)
  (h2 : p.b_investment = 63000)
  (h3 : p.total_profit = 60000)
  (h4 : p.c_profit = 24000)
  (h5 : p.c_profit * (p.a_investment + p.b_investment + p.c_investment) = p.total_profit * p.c_investment) :
  p.c_investment = 72000 := by
  sorry


end NUMINAMATH_CALUDE_c_investment_value_l1986_198678


namespace NUMINAMATH_CALUDE_prince_cd_spend_l1986_198667

/-- Calculates the amount spent on CDs given the total number of CDs,
    percentage of expensive CDs, prices, and buying pattern. -/
def calculate_cd_spend (total_cds : ℕ) (expensive_percentage : ℚ) 
                       (expensive_price cheap_price : ℚ) 
                       (expensive_bought_ratio : ℚ) : ℚ :=
  let expensive_cds : ℚ := expensive_percentage * total_cds
  let cheap_cds : ℚ := (1 - expensive_percentage) * total_cds
  let expensive_bought : ℚ := expensive_bought_ratio * expensive_cds
  expensive_bought * expensive_price + cheap_cds * cheap_price

/-- Proves that Prince spent $1000 on CDs given the problem conditions. -/
theorem prince_cd_spend : 
  calculate_cd_spend 200 (40/100) 10 5 (1/2) = 1000 := by
  sorry

end NUMINAMATH_CALUDE_prince_cd_spend_l1986_198667


namespace NUMINAMATH_CALUDE_angle_addition_theorem_l1986_198690

/-- Represents an angle in degrees and minutes -/
structure Angle where
  degrees : ℕ
  minutes : ℕ
  valid : minutes < 60

/-- Addition of two angles -/
def add_angles (a b : Angle) : Angle :=
  let total_minutes := a.degrees * 60 + a.minutes + b.degrees * 60 + b.minutes
  { degrees := total_minutes / 60,
    minutes := total_minutes % 60,
    valid := by sorry }

/-- The theorem to prove -/
theorem angle_addition_theorem :
  let a := { degrees := 48, minutes := 39, valid := by sorry }
  let b := { degrees := 67, minutes := 31, valid := by sorry }
  let result := add_angles a b
  result.degrees = 116 ∧ result.minutes = 10 := by sorry

end NUMINAMATH_CALUDE_angle_addition_theorem_l1986_198690


namespace NUMINAMATH_CALUDE_cat_dog_food_difference_l1986_198656

theorem cat_dog_food_difference :
  let cat_packages : ℕ := 6
  let dog_packages : ℕ := 2
  let cans_per_cat_package : ℕ := 9
  let cans_per_dog_package : ℕ := 3
  let total_cat_cans := cat_packages * cans_per_cat_package
  let total_dog_cans := dog_packages * cans_per_dog_package
  total_cat_cans - total_dog_cans = 48 :=
by sorry

end NUMINAMATH_CALUDE_cat_dog_food_difference_l1986_198656


namespace NUMINAMATH_CALUDE_shipping_cost_calculation_l1986_198609

def fish_weight : ℕ := 540
def crate_capacity : ℕ := 30
def crate_cost : ℚ := 3/2

theorem shipping_cost_calculation :
  (fish_weight / crate_capacity) * crate_cost = 27 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_calculation_l1986_198609


namespace NUMINAMATH_CALUDE_john_total_distance_l1986_198663

-- Define the driving speed
def speed : ℝ := 45

-- Define the first driving duration
def duration1 : ℝ := 2

-- Define the second driving duration
def duration2 : ℝ := 3

-- Theorem to prove
theorem john_total_distance :
  speed * (duration1 + duration2) = 225 := by
  sorry

end NUMINAMATH_CALUDE_john_total_distance_l1986_198663


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1520_l1986_198604

-- Define the parameters of the problem
def num_bedrooms : ℕ := 4
def room_length : ℝ := 14
def room_width : ℝ := 11
def room_height : ℝ := 9
def unpaintable_area : ℝ := 70

-- Calculate the total wall area of one bedroom
def total_wall_area : ℝ := 2 * (room_length * room_height + room_width * room_height)

-- Calculate the paintable area of one bedroom
def paintable_area_per_room : ℝ := total_wall_area - unpaintable_area

-- Theorem statement
theorem total_paintable_area_is_1520 :
  num_bedrooms * paintable_area_per_room = 1520 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1520_l1986_198604


namespace NUMINAMATH_CALUDE_donna_truck_weight_l1986_198600

/-- The weight of Donna's fully loaded truck -/
def truck_weight : ℕ :=
  let empty_truck_weight : ℕ := 12000
  let soda_crate_weight : ℕ := 50
  let soda_crate_count : ℕ := 20
  let dryer_weight : ℕ := 3000
  let dryer_count : ℕ := 3
  let soda_weight : ℕ := soda_crate_weight * soda_crate_count
  let produce_weight : ℕ := 2 * soda_weight
  let dryers_weight : ℕ := dryer_weight * dryer_count
  empty_truck_weight + soda_weight + produce_weight + dryers_weight

/-- Theorem stating that Donna's fully loaded truck weighs 24,000 pounds -/
theorem donna_truck_weight : truck_weight = 24000 := by
  sorry

end NUMINAMATH_CALUDE_donna_truck_weight_l1986_198600


namespace NUMINAMATH_CALUDE_complex_math_expression_equals_35_l1986_198697

theorem complex_math_expression_equals_35 :
  ((9^2 + (3^3 - 1) * 4^2) % 6 : ℕ) * Real.sqrt 49 + (15 - 3 * 5) = 35 := by
  sorry

end NUMINAMATH_CALUDE_complex_math_expression_equals_35_l1986_198697


namespace NUMINAMATH_CALUDE_fraction_equality_l1986_198640

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = -3) : 
  (x + 4 * y) / (4 * x - y) = 39 / 37 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1986_198640


namespace NUMINAMATH_CALUDE_area_fraction_above_line_l1986_198661

/-- A square with side length 3 -/
def square_side : ℝ := 3

/-- The first point of the line -/
def point1 : ℝ × ℝ := (3, 2)

/-- The second point of the line -/
def point2 : ℝ × ℝ := (6, 0)

/-- The theorem stating that the fraction of the square's area above the line is 2/3 -/
theorem area_fraction_above_line : 
  let square_area := square_side ^ 2
  let triangle_base := point2.1 - point1.1
  let triangle_height := point1.2
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let area_above_line := square_area - triangle_area
  (area_above_line / square_area) = (2 : ℝ) / 3 := by sorry

end NUMINAMATH_CALUDE_area_fraction_above_line_l1986_198661
