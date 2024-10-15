import Mathlib

namespace NUMINAMATH_CALUDE_least_number_divisible_l2214_221488

theorem least_number_divisible (n : ℕ) : n = 858 ↔ 
  (∀ m : ℕ, m < n → 
    ¬((m + 6) % 24 = 0 ∧ 
      (m + 6) % 32 = 0 ∧ 
      (m + 6) % 36 = 0 ∧ 
      (m + 6) % 54 = 0)) ∧
  ((n + 6) % 24 = 0 ∧ 
   (n + 6) % 32 = 0 ∧ 
   (n + 6) % 36 = 0 ∧ 
   (n + 6) % 54 = 0) :=
by sorry

end NUMINAMATH_CALUDE_least_number_divisible_l2214_221488


namespace NUMINAMATH_CALUDE_max_abc_value_l2214_221462

theorem max_abc_value (a b c : ℝ) (sum_eq : a + b + c = 5) (prod_sum_eq : a * b + b * c + c * a = 7) :
  ∀ x y z : ℝ, x + y + z = 5 → x * y + y * z + z * x = 7 → a * b * c ≥ x * y * z ∧ ∃ p q r : ℝ, p + q + r = 5 ∧ p * q + q * r + r * p = 7 ∧ p * q * r = 3 :=
sorry

end NUMINAMATH_CALUDE_max_abc_value_l2214_221462


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2214_221464

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (a 1 + a n) / 2
  sum_13 : sum 13 = 104
  a_6 : a 6 = 5

/-- The common difference of the arithmetic sequence is 3 -/
theorem arithmetic_sequence_common_difference (seq : ArithmeticSequence) : seq.d = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2214_221464


namespace NUMINAMATH_CALUDE_mk97_equality_check_l2214_221426

/-- The MK-97 microcalculator operations -/
class Calculator where
  /-- Check if two numbers are equal -/
  equal : ℝ → ℝ → Prop
  /-- Add two numbers -/
  add : ℝ → ℝ → ℝ
  /-- Find roots of a quadratic equation -/
  quadratic_roots : ℝ → ℝ → Option (ℝ × ℝ)

/-- The theorem to be proved -/
theorem mk97_equality_check (x : ℝ) :
  x = 1 ↔ x ≠ 0 ∧ (4 * (x^2 - x) = 0) :=
sorry

end NUMINAMATH_CALUDE_mk97_equality_check_l2214_221426


namespace NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2214_221475

theorem police_emergency_number_prime_divisor (n : ℕ) (k : ℕ) (h : n = 100 * k + 133) :
  ∃ p : ℕ, p.Prime ∧ p > 7 ∧ p ∣ n := by
  sorry

end NUMINAMATH_CALUDE_police_emergency_number_prime_divisor_l2214_221475


namespace NUMINAMATH_CALUDE_square_garden_area_perimeter_difference_l2214_221444

theorem square_garden_area_perimeter_difference :
  ∀ (s : ℝ), 
    s > 0 →
    4 * s = 28 →
    s^2 - 4 * s = 21 :=
by
  sorry

end NUMINAMATH_CALUDE_square_garden_area_perimeter_difference_l2214_221444


namespace NUMINAMATH_CALUDE_ratio_lcm_problem_l2214_221472

theorem ratio_lcm_problem (a b x : ℕ+) (h_ratio : a.val * x.val = 8 * b.val) 
  (h_lcm : Nat.lcm a.val b.val = 432) (h_a : a = 48) : b = 72 := by
  sorry

end NUMINAMATH_CALUDE_ratio_lcm_problem_l2214_221472


namespace NUMINAMATH_CALUDE_janet_gained_lives_l2214_221458

/-- Calculates the number of lives Janet gained in a video game level -/
def lives_gained (initial : ℕ) (lost : ℕ) (final : ℕ) : ℕ :=
  final - (initial - lost)

/-- Proves that Janet gained 32 lives in the next level -/
theorem janet_gained_lives : lives_gained 38 16 54 = 32 := by
  sorry

end NUMINAMATH_CALUDE_janet_gained_lives_l2214_221458


namespace NUMINAMATH_CALUDE_parabola_translation_l2214_221439

/-- Represents a parabola in the form y = ax^2 + bx + c --/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically --/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation :
  let original := Parabola.mk 1 0 0  -- y = x^2
  let translated := translate original 2 1
  translated = Parabola.mk 1 (-4) 5  -- y = (x-2)^2 + 1
  := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l2214_221439


namespace NUMINAMATH_CALUDE_book_pages_proof_l2214_221456

/-- Calculates the total number of pages in a book given the reading schedule --/
def total_pages (pages_per_day_first_four : ℕ) (pages_per_day_next_two : ℕ) (pages_last_day : ℕ) : ℕ :=
  4 * pages_per_day_first_four + 2 * pages_per_day_next_two + pages_last_day

/-- Proves that the total number of pages in the book is 264 --/
theorem book_pages_proof : total_pages 42 38 20 = 264 := by
  sorry

#eval total_pages 42 38 20

end NUMINAMATH_CALUDE_book_pages_proof_l2214_221456


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l2214_221460

def vec_parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-4, 2)  -- Derived from a - (1/2)b = (3,1)
  let c : ℝ × ℝ := (x, 3)
  vec_parallel (2 * a + b) c → x = -1 :=
by sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l2214_221460


namespace NUMINAMATH_CALUDE_inequality_for_positive_numbers_l2214_221423

theorem inequality_for_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  (a + 1)⁻¹ + (b + 1)⁻¹ ≥ 4/3 := by sorry

end NUMINAMATH_CALUDE_inequality_for_positive_numbers_l2214_221423


namespace NUMINAMATH_CALUDE_overlapping_circles_common_chord_l2214_221465

theorem overlapping_circles_common_chord 
  (r : ℝ) 
  (h : r = 12) : 
  let chord_length := 2 * r * Real.sqrt 3
  chord_length = 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_circles_common_chord_l2214_221465


namespace NUMINAMATH_CALUDE_line_ellipse_intersections_l2214_221417

/-- The line equation 3x + 4y = 12 -/
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The ellipse equation (x-1)^2 + 4y^2 = 4 -/
def ellipse_eq (x y : ℝ) : Prop := (x - 1)^2 + 4 * y^2 = 4

/-- The number of intersections between the line and the ellipse -/
def num_intersections : ℕ := 0

/-- Theorem stating that the number of intersections between the line and the ellipse is 0 -/
theorem line_ellipse_intersections :
  ∀ x y : ℝ, line_eq x y ∧ ellipse_eq x y → num_intersections = 0 :=
by sorry

end NUMINAMATH_CALUDE_line_ellipse_intersections_l2214_221417


namespace NUMINAMATH_CALUDE_f_times_g_equals_one_l2214_221454

/-- The formal power series f(x) defined as an infinite geometric series -/
noncomputable def f (x : ℝ) : ℝ := ∑' n, x^n

/-- The function g(x) defined as 1 - x -/
def g (x : ℝ) : ℝ := 1 - x

/-- Theorem stating that f(x)g(x) = 1 -/
theorem f_times_g_equals_one (x : ℝ) (hx : |x| < 1) : f x * g x = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_times_g_equals_one_l2214_221454


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_implies_a_in_open_interval_l2214_221457

-- Define the complex number z
def z (a : ℝ) : ℂ := (1 + a * Complex.I) * (1 - Complex.I)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (w : ℂ) : Prop := 0 < w.re ∧ w.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant_implies_a_in_open_interval :
  ∀ a : ℝ, in_fourth_quadrant (z a) → -1 < a ∧ a < 1 := by
  sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_implies_a_in_open_interval_l2214_221457


namespace NUMINAMATH_CALUDE_correct_number_of_men_l2214_221468

/-- The number of men in the first group that completes a job in 15 days,
    given that 25 men can finish the same job in 18 days. -/
def number_of_men : ℕ := 30

/-- The number of days taken by the first group to complete the job. -/
def days_first_group : ℕ := 15

/-- The number of men in the second group. -/
def men_second_group : ℕ := 25

/-- The number of days taken by the second group to complete the job. -/
def days_second_group : ℕ := 18

/-- Theorem stating that the number of men in the first group is correct. -/
theorem correct_number_of_men :
  number_of_men * days_first_group = men_second_group * days_second_group :=
sorry

end NUMINAMATH_CALUDE_correct_number_of_men_l2214_221468


namespace NUMINAMATH_CALUDE_function_analysis_l2214_221445

def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x + a - 1| - a^2 - 2*a

theorem function_analysis (a : ℝ) :
  (a = 3 → {x : ℝ | f 3 x ≥ -10} = {x : ℝ | x ≥ 3 ∨ x ≤ -1}) ∧
  ({a : ℝ | ∀ x, f a x ≥ 0} = {a : ℝ | -2 ≤ a ∧ a ≤ 0}) :=
by sorry

end NUMINAMATH_CALUDE_function_analysis_l2214_221445


namespace NUMINAMATH_CALUDE_sequence_formula_l2214_221424

def S₁ (n : ℕ) : ℕ := n^2

def S₂ (n : ℕ) : ℕ := n^2 + n + 1

def a₁ (n : ℕ) : ℕ := 2*n - 1

def a₂ (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2*n

theorem sequence_formula (n : ℕ) (h : n ≥ 1) :
  (∀ k, S₁ k - S₁ (k-1) = a₁ k) ∧
  (∀ k, S₂ k - S₂ (k-1) = a₂ k) :=
sorry

end NUMINAMATH_CALUDE_sequence_formula_l2214_221424


namespace NUMINAMATH_CALUDE_total_annual_interest_l2214_221434

def total_investment : ℕ := 3200
def first_part : ℕ := 800
def first_rate : ℚ := 3 / 100
def second_rate : ℚ := 5 / 100

def second_part : ℕ := total_investment - first_part

def interest_first : ℚ := (first_part : ℚ) * first_rate
def interest_second : ℚ := (second_part : ℚ) * second_rate

theorem total_annual_interest :
  interest_first + interest_second = 144 := by sorry

end NUMINAMATH_CALUDE_total_annual_interest_l2214_221434


namespace NUMINAMATH_CALUDE_sin_cos_difference_identity_l2214_221436

theorem sin_cos_difference_identity :
  Real.sin (47 * π / 180) * Real.cos (17 * π / 180) - 
  Real.cos (47 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_identity_l2214_221436


namespace NUMINAMATH_CALUDE_problem_solution_l2214_221486

/-- Calculates the total number of new cans that can be made from a given number of cans,
    considering that newly made cans can also be recycled. -/
def totalNewCans (initialCans : ℕ) (damagedCans : ℕ) (requiredForNewCan : ℕ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions of the problem,
    the total number of new cans that can be made is 95. -/
theorem problem_solution :
  totalNewCans 500 20 6 = 95 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2214_221486


namespace NUMINAMATH_CALUDE_quadratic_decreasing_l2214_221430

/-- Theorem: For a quadratic function y = ax² + 2ax + c where a < 0,
    and points A(1, y₁) and B(2, y₂) on this function, y₁ - y₂ > 0. -/
theorem quadratic_decreasing (a c y₁ y₂ : ℝ) (ha : a < 0) 
  (h1 : y₁ = a + 2*a + c) 
  (h2 : y₂ = 4*a + 4*a + c) : 
  y₁ - y₂ > 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_l2214_221430


namespace NUMINAMATH_CALUDE_trajectory_is_parabola_l2214_221409

/-- The trajectory of a point equidistant from a fixed point and a line is a parabola -/
theorem trajectory_is_parabola (x y : ℝ) : 
  (x^2 + (y + 3)^2)^(1/2) = |y - 3| → x^2 = -12*y :=
by sorry

end NUMINAMATH_CALUDE_trajectory_is_parabola_l2214_221409


namespace NUMINAMATH_CALUDE_square_ends_in_six_tens_digit_odd_l2214_221498

theorem square_ends_in_six_tens_digit_odd (n : ℤ) : 
  n^2 % 100 = 6 → (n^2 / 10) % 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_ends_in_six_tens_digit_odd_l2214_221498


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_length_l2214_221491

def repeating_decimal_length (n m : ℕ) : ℕ :=
  sorry

theorem seven_thirteenths_repeating_length :
  repeating_decimal_length 7 13 = 6 := by sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_length_l2214_221491


namespace NUMINAMATH_CALUDE_river_depth_problem_l2214_221420

theorem river_depth_problem (d k : ℝ) : 
  (d + 0.5 * d + k = 1.5 * (d + 0.5 * d)) →  -- Depth in mid-July is 1.5 times the depth at the end of May
  (1.5 * (d + 0.5 * d) = 45) →               -- Final depth in mid-July is 45 feet
  (d = 15 ∧ k = 11.25) :=                    -- Initial depth is 15 feet and depth increase in June is 11.25 feet
by sorry

end NUMINAMATH_CALUDE_river_depth_problem_l2214_221420


namespace NUMINAMATH_CALUDE_expenditure_ratio_proof_l2214_221405

/-- Represents the financial data of a person -/
structure PersonFinance where
  income : ℕ
  savings : ℕ
  expenditure : ℕ

/-- The problem statement -/
theorem expenditure_ratio_proof 
  (p1 p2 : PersonFinance)
  (h1 : p1.income = 3000)
  (h2 : p1.income * 4 = p2.income * 5)
  (h3 : p1.savings = 1200)
  (h4 : p2.savings = 1200)
  (h5 : p1.expenditure = p1.income - p1.savings)
  (h6 : p2.expenditure = p2.income - p2.savings)
  : p1.expenditure * 2 = p2.expenditure * 3 := by
  sorry

end NUMINAMATH_CALUDE_expenditure_ratio_proof_l2214_221405


namespace NUMINAMATH_CALUDE_complex_simplification_l2214_221474

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Proof that 3(4-2i) + 2i(3+2i) = 8 -/
theorem complex_simplification : 3 * (4 - 2 * i) + 2 * i * (3 + 2 * i) = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l2214_221474


namespace NUMINAMATH_CALUDE_books_sold_in_three_days_l2214_221470

/-- The total number of books sold over three days -/
def total_books_sold (tuesday_sales wednesday_sales thursday_sales : ℕ) : ℕ :=
  tuesday_sales + wednesday_sales + thursday_sales

/-- Theorem stating the total number of books sold over three days -/
theorem books_sold_in_three_days :
  ∃ (tuesday_sales wednesday_sales thursday_sales : ℕ),
    tuesday_sales = 7 ∧
    wednesday_sales = 3 * tuesday_sales ∧
    thursday_sales = 3 * wednesday_sales ∧
    total_books_sold tuesday_sales wednesday_sales thursday_sales = 91 := by
  sorry

end NUMINAMATH_CALUDE_books_sold_in_three_days_l2214_221470


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_rotation_volume_l2214_221435

theorem isosceles_right_triangle_rotation_volume :
  ∀ (r h : ℝ), r = 1 → h = 1 →
  (1 / 3 : ℝ) * Real.pi * r^2 * h = Real.pi / 3 := by
sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_rotation_volume_l2214_221435


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2214_221490

theorem greatest_divisor_with_remainders : Nat.gcd (1657 - 6) (2037 - 5) = 127 := by
  sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l2214_221490


namespace NUMINAMATH_CALUDE_a_less_than_neg_one_l2214_221416

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem a_less_than_neg_one (f : ℝ → ℝ) (a : ℝ) 
  (h_odd : is_odd f)
  (h_period : has_period f 3)
  (h_f1 : f 1 > 1)
  (h_f2 : f 2 = a) :
  a < -1 := by sorry

end NUMINAMATH_CALUDE_a_less_than_neg_one_l2214_221416


namespace NUMINAMATH_CALUDE_jason_has_21_toys_l2214_221425

/-- The number of toys Rachel has -/
def rachel_toys : ℕ := 1

/-- The number of toys John has -/
def john_toys : ℕ := rachel_toys + 6

/-- The number of toys Jason has -/
def jason_toys : ℕ := 3 * john_toys

/-- Theorem: Jason has 21 toys -/
theorem jason_has_21_toys : jason_toys = 21 := by
  sorry

end NUMINAMATH_CALUDE_jason_has_21_toys_l2214_221425


namespace NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_l2214_221433

theorem number_divided_by_6_multiplied_by_12 :
  ∃ x : ℝ, (x / 6) * 12 = 15 ∧ x = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_6_multiplied_by_12_l2214_221433


namespace NUMINAMATH_CALUDE_knights_on_red_chairs_l2214_221497

structure Room where
  total_chairs : Nat
  knights : Nat
  liars : Nat
  knights_on_red : Nat
  liars_on_blue : Nat

/-- The room satisfies the initial conditions -/
def initial_condition (r : Room) : Prop :=
  r.total_chairs = 20 ∧ 
  r.knights + r.liars = r.total_chairs

/-- The room satisfies the conditions after switching seats -/
def after_switch_condition (r : Room) : Prop :=
  r.knights_on_red + (r.knights - r.knights_on_red) = r.total_chairs / 2 ∧
  (r.liars - r.liars_on_blue) + r.liars_on_blue = r.total_chairs / 2 ∧
  r.knights_on_red = r.liars_on_blue

theorem knights_on_red_chairs (r : Room) 
  (h1 : initial_condition r) 
  (h2 : after_switch_condition r) : 
  r.knights_on_red = 5 := by
  sorry

end NUMINAMATH_CALUDE_knights_on_red_chairs_l2214_221497


namespace NUMINAMATH_CALUDE_base_n_1001_not_prime_l2214_221493

/-- For a positive integer n ≥ 2, 1001_n represents n^3 + 1 in base 10 -/
def base_n_1001 (n : ℕ) : ℕ := n^3 + 1

/-- A number is composite if it has a factor between 1 and itself -/
def is_composite (m : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < m ∧ m % k = 0

theorem base_n_1001_not_prime : 
  ∀ n : ℕ, n ≥ 2 → is_composite (base_n_1001 n) := by
  sorry

end NUMINAMATH_CALUDE_base_n_1001_not_prime_l2214_221493


namespace NUMINAMATH_CALUDE_expression_evaluation_l2214_221412

theorem expression_evaluation (x y z k : ℤ) 
  (hx : x = 25) (hy : y = 12) (hz : z = 3) (hk : k = 4) :
  (x - (y - z)) - ((x - y) - (z + k)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2214_221412


namespace NUMINAMATH_CALUDE_complex_number_simplification_l2214_221440

theorem complex_number_simplification :
  (-5 - 3 * Complex.I) * 2 - (2 + 5 * Complex.I) = -12 - 11 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_simplification_l2214_221440


namespace NUMINAMATH_CALUDE_distance_BD_l2214_221499

/-- Given three points B, C, and D in a 2D plane, prove that the distance between B and D is 13. -/
theorem distance_BD (B C D : ℝ × ℝ) : 
  B = (3, 9) → C = (3, -3) → D = (-2, -3) → 
  Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_BD_l2214_221499


namespace NUMINAMATH_CALUDE_work_completion_time_l2214_221404

/-- Given workers a, b, and c who can complete a work in 16, x, and 12 days respectively,
    and together they complete the work in 3.2 days, prove that x = 6. -/
theorem work_completion_time (x : ℝ) 
  (h1 : 1/16 + 1/x + 1/12 = 1/3.2) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2214_221404


namespace NUMINAMATH_CALUDE_cube_sum_expression_l2214_221427

theorem cube_sum_expression (x y z w a b c d : ℝ) 
  (hxy : x * y = a)
  (hxz : x * z = b)
  (hyz : y * z = c)
  (hxw : x * w = d)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hz : z ≠ 0)
  (hw : w ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (hd : d ≠ 0) :
  x^3 + y^3 + z^3 + w^3 = (a^3 * d^3 + a^3 * c^3 + b^3 * d^3 + d^3 * b^3) / (a * b * c * d) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_expression_l2214_221427


namespace NUMINAMATH_CALUDE_correct_point_satisfies_conditions_l2214_221455

def point_satisfies_conditions (x y : ℝ) : Prop :=
  -2 < x ∧ x < 0 ∧ 2 < y ∧ y < 4

theorem correct_point_satisfies_conditions :
  point_satisfies_conditions (-1) 3 ∧
  ¬ point_satisfies_conditions 1 3 ∧
  ¬ point_satisfies_conditions 1 (-3) ∧
  ¬ point_satisfies_conditions (-3) 1 ∧
  ¬ point_satisfies_conditions 3 (-1) :=
by sorry

end NUMINAMATH_CALUDE_correct_point_satisfies_conditions_l2214_221455


namespace NUMINAMATH_CALUDE_function_characterization_l2214_221403

theorem function_characterization (a : ℝ) (ha : a > 0) :
  ∀ (f : ℕ → ℝ),
    (∀ (k m : ℕ), k > 0 ∧ m > 0 ∧ a * m ≤ k ∧ k < (a + 1) * m → f (k + m) = f k + f m) ↔
    ∃ (b : ℝ), ∀ (n : ℕ), f n = b * n :=
by sorry

end NUMINAMATH_CALUDE_function_characterization_l2214_221403


namespace NUMINAMATH_CALUDE_production_and_salary_optimization_l2214_221415

-- Define the variables and constants
def workday_minutes : ℕ := 8 * 60
def base_salary : ℕ := 100
def type_b_wage : ℚ := 2.5
def total_products : ℕ := 28

-- Define the production time equations
def production_equation_1 (x y : ℚ) : Prop := 6 * x + 4 * y = 170
def production_equation_2 (x y : ℚ) : Prop := 10 * x + 10 * y = 350

-- Define the time constraint
def time_constraint (x y : ℚ) (m : ℕ) : Prop :=
  x * m + y * (total_products - m) ≤ workday_minutes

-- Define the salary function
def salary (a : ℚ) (m : ℕ) : ℚ :=
  a * m + type_b_wage * (total_products - m) + base_salary

-- Theorem statement
theorem production_and_salary_optimization
  (x y : ℚ) (a : ℚ) (h_a : 2 < a ∧ a < 3) :
  (production_equation_1 x y ∧ production_equation_2 x y) →
  (x = 15 ∧ y = 20) ∧
  (∀ m : ℕ, m ≤ total_products →
    time_constraint x y m →
    (2 < a ∧ a < 2.5 → salary a 16 ≥ salary a m) ∧
    (a = 2.5 → salary a m = salary a 16) ∧
    (2.5 < a ∧ a < 3 → salary a 28 ≥ salary a m)) :=
sorry

end NUMINAMATH_CALUDE_production_and_salary_optimization_l2214_221415


namespace NUMINAMATH_CALUDE_factorization_x2_minus_4x_minus_12_minimum_value_4x2_plus_4x_minus_1_l2214_221438

-- Problem 1
theorem factorization_x2_minus_4x_minus_12 :
  ∀ x : ℝ, x^2 - 4*x - 12 = (x - 6) * (x + 2) := by sorry

-- Problem 2
theorem minimum_value_4x2_plus_4x_minus_1 :
  ∀ x : ℝ, 4*x^2 + 4*x - 1 ≥ -2 ∧
  ∃ x : ℝ, 4*x^2 + 4*x - 1 = -2 ∧ x = -1/2 := by sorry

end NUMINAMATH_CALUDE_factorization_x2_minus_4x_minus_12_minimum_value_4x2_plus_4x_minus_1_l2214_221438


namespace NUMINAMATH_CALUDE_speeding_ticket_percentage_l2214_221437

/-- The percentage of motorists who exceed the speed limit -/
def exceed_speed_limit : ℝ := 12.5

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 20

/-- The percentage of motorists who receive speeding tickets -/
def receive_ticket_percentage : ℝ := 10

/-- Theorem stating that the percentage of motorists receiving speeding tickets is 10% -/
theorem speeding_ticket_percentage :
  receive_ticket_percentage = exceed_speed_limit * (100 - no_ticket_percentage) / 100 := by
  sorry

end NUMINAMATH_CALUDE_speeding_ticket_percentage_l2214_221437


namespace NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_bound_l2214_221419

theorem product_of_reciprocal_minus_one_bound 
  (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_one : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_product_of_reciprocal_minus_one_bound_l2214_221419


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2214_221446

-- Define the line equation
def line_equation (x y : ℝ) : Prop := y = 2 * x - 1

-- Define the theorem
theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : line_equation (-3) y1) 
  (h2 : line_equation (-5) y2) : 
  y1 > y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2214_221446


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l2214_221485

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 3 * x + 2

-- State the theorem
theorem solution_set_implies_sum (a b : ℝ) :
  (∀ x, 1 < x ∧ x < b ↔ f a x < 0) →
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l2214_221485


namespace NUMINAMATH_CALUDE_gcf_72_108_l2214_221463

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end NUMINAMATH_CALUDE_gcf_72_108_l2214_221463


namespace NUMINAMATH_CALUDE_borel_sets_closed_under_countable_operations_l2214_221407

-- Define the σ-algebra of Borel sets
def BorelSets : Set (Set ℝ) := sorry

-- Define the property of being generated by open sets
def GeneratedByOpenSets (S : Set (Set ℝ)) : Prop := sorry

-- Define closure under countable union
def ClosedUnderCountableUnion (S : Set (Set ℝ)) : Prop := sorry

-- Define closure under countable intersection
def ClosedUnderCountableIntersection (S : Set (Set ℝ)) : Prop := sorry

-- Theorem statement
theorem borel_sets_closed_under_countable_operations :
  GeneratedByOpenSets BorelSets →
  ClosedUnderCountableUnion BorelSets ∧ ClosedUnderCountableIntersection BorelSets := by
  sorry

end NUMINAMATH_CALUDE_borel_sets_closed_under_countable_operations_l2214_221407


namespace NUMINAMATH_CALUDE_jacks_paycheck_l2214_221481

theorem jacks_paycheck (paycheck : ℝ) : 
  (0.2 * (0.8 * paycheck) = 20) → paycheck = 125 := by
  sorry

end NUMINAMATH_CALUDE_jacks_paycheck_l2214_221481


namespace NUMINAMATH_CALUDE_trigonometric_equation_system_solution_l2214_221453

theorem trigonometric_equation_system_solution :
  ∃ (x y : ℝ),
    3 * Real.cos x + 4 * Real.sin x = -1.4 ∧
    13 * Real.cos x - 41 * Real.cos y = -45 ∧
    13 * Real.sin x + 41 * Real.sin y = 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_system_solution_l2214_221453


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l2214_221492

/-- The number of ways to distribute n students to k towns, ensuring each town receives at least one student. -/
def distribute_students (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- The number of ways to choose r items from n items. -/
def binomial_coefficient (n : ℕ) (r : ℕ) : ℕ :=
  sorry

/-- The number of permutations of n items. -/
def permutations (n : ℕ) : ℕ :=
  sorry

theorem student_distribution_theorem :
  distribute_students 4 3 = 36 :=
by sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l2214_221492


namespace NUMINAMATH_CALUDE_work_speed_l2214_221461

/-- Proves that given a round trip of 2 hours, 72 minutes to work, and 90 km/h return speed, the speed to work is 60 km/h -/
theorem work_speed (total_time : Real) (time_to_work : Real) (return_speed : Real) :
  total_time = 2 ∧ 
  time_to_work = 72 / 60 ∧ 
  return_speed = 90 →
  (2 * return_speed * time_to_work) / (total_time + time_to_work) = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_speed_l2214_221461


namespace NUMINAMATH_CALUDE_point_q_midpoint_l2214_221467

/-- Given five points on a line, prove that Q is the midpoint of A and B -/
theorem point_q_midpoint (O A B C D Q : ℝ) (l m n p : ℝ) : 
  O < A ∧ A < B ∧ B < C ∧ C < D →  -- Points are in order
  A - O = l →  -- OA = l
  B - O = m →  -- OB = m
  C - O = n →  -- OC = n
  D - O = p →  -- OD = p
  A ≤ Q ∧ Q ≤ B →  -- Q is between A and B
  (C - Q) / (Q - D) = (B - Q) / (Q - A) →  -- CQ : QD = BQ : QA
  Q - O = (l + m) / 2 :=  -- OQ = (l + m) / 2
by sorry

end NUMINAMATH_CALUDE_point_q_midpoint_l2214_221467


namespace NUMINAMATH_CALUDE_point_movement_on_number_line_l2214_221400

theorem point_movement_on_number_line (A B : ℝ) : 
  A = 7 → B - A = 3 → B = 10 := by sorry

end NUMINAMATH_CALUDE_point_movement_on_number_line_l2214_221400


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2214_221476

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 3.5) : x^2 + 1/x^2 = 10.25 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2214_221476


namespace NUMINAMATH_CALUDE_brush_chess_prices_l2214_221480

theorem brush_chess_prices (brush_price chess_price : ℚ) : 
  (5 * brush_price + 12 * chess_price = 315) →
  (8 * brush_price + 6 * chess_price = 240) →
  (brush_price = 15 ∧ chess_price = 20) := by
sorry

end NUMINAMATH_CALUDE_brush_chess_prices_l2214_221480


namespace NUMINAMATH_CALUDE_upstream_speed_l2214_221408

/-- 
Given a man's rowing speed in still water and his speed downstream, 
this theorem proves that his speed upstream can be calculated as the 
difference between his speed in still water and half the difference 
between his downstream speed and his speed in still water.
-/
theorem upstream_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still > 0) 
  (h2 : speed_downstream > speed_still) : 
  speed_still - (speed_downstream - speed_still) / 2 = 
  speed_still - (speed_downstream - speed_still) / 2 :=
by sorry

end NUMINAMATH_CALUDE_upstream_speed_l2214_221408


namespace NUMINAMATH_CALUDE_train_passing_time_train_passes_jogger_in_40_seconds_l2214_221422

/-- Calculates the time for a train to pass a jogger given their initial speeds,
    distances, and speed reduction due to incline. -/
theorem train_passing_time (jogger_speed train_speed : ℝ)
                           (initial_distance train_length : ℝ)
                           (incline_reduction : ℝ) : ℝ :=
  let jogger_effective_speed := jogger_speed * (1 - incline_reduction)
  let train_effective_speed := train_speed * (1 - incline_reduction)
  let relative_speed := train_effective_speed - jogger_effective_speed
  let total_distance := initial_distance + train_length
  total_distance / relative_speed * (3600 / 1000)

/-- The time for the train to pass the jogger is 40 seconds. -/
theorem train_passes_jogger_in_40_seconds :
  train_passing_time 9 45 240 120 0.1 = 40 := by
  sorry


end NUMINAMATH_CALUDE_train_passing_time_train_passes_jogger_in_40_seconds_l2214_221422


namespace NUMINAMATH_CALUDE_apple_picking_problem_l2214_221432

theorem apple_picking_problem (x : ℝ) : 
  x + (3/4) * x + 600 = 2600 → x = 1142 := by
  sorry

end NUMINAMATH_CALUDE_apple_picking_problem_l2214_221432


namespace NUMINAMATH_CALUDE_min_even_integers_l2214_221451

theorem min_even_integers (a b c d e f : ℤ) : 
  a + b = 30 →
  a + b + c + d = 50 →
  a + b + c + d + e + f = 70 →
  Even e →
  Even f →
  (∃ (count : ℕ), count ≥ 2 ∧ 
    count = (if Even a then 1 else 0) +
            (if Even b then 1 else 0) +
            (if Even c then 1 else 0) +
            (if Even d then 1 else 0) + 2) :=
by sorry

end NUMINAMATH_CALUDE_min_even_integers_l2214_221451


namespace NUMINAMATH_CALUDE_target_practice_probabilities_l2214_221477

/-- Represents a shooter in the target practice scenario -/
structure Shooter where
  hit_probability : ℝ
  num_shots : ℕ

/-- Calculates the probability of the given event -/
def calculate_probability (s1 s2 : Shooter) (event : Shooter → Shooter → ℝ) : ℝ :=
  event s1 s2

/-- The scenario with two shooters -/
def target_practice_scenario : Prop :=
  ∃ (s1 s2 : Shooter),
    s1.hit_probability = 0.8 ∧
    s2.hit_probability = 0.6 ∧
    s1.num_shots = 2 ∧
    s2.num_shots = 3 ∧
    (calculate_probability s1 s2 (λ _ _ => 0.99744) = 
     calculate_probability s1 s2 (λ s1 s2 => 1 - (1 - s1.hit_probability)^s1.num_shots * (1 - s2.hit_probability)^s2.num_shots)) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.13824) = 
     calculate_probability s1 s2 (λ s1 s2 => (s1.num_shots * s1.hit_probability * (1 - s1.hit_probability)) * 
                                             (Nat.choose s2.num_shots 2 * s2.hit_probability^2 * (1 - s2.hit_probability)))) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.87328) = 
     calculate_probability s1 s2 (λ s1 s2 => 1 - (1 - s1.hit_probability^2) * 
                                             (1 - s2.hit_probability^2 - s2.hit_probability^3))) ∧
    (calculate_probability s1 s2 (λ _ _ => 0.032) = 
     calculate_probability s1 s2 (λ s1 s2 => (s1.num_shots * s1.hit_probability * (1 - s1.hit_probability)^(s1.num_shots - 1) * (1 - s2.hit_probability)^s2.num_shots) + 
                                             ((1 - s1.hit_probability)^s1.num_shots * s2.num_shots * s2.hit_probability * (1 - s2.hit_probability)^(s2.num_shots - 1))))

theorem target_practice_probabilities : target_practice_scenario := sorry

end NUMINAMATH_CALUDE_target_practice_probabilities_l2214_221477


namespace NUMINAMATH_CALUDE_solve_equation_for_m_l2214_221402

theorem solve_equation_for_m : ∃ m : ℤ, 
  62519 * 9999^2 / 314 * (314 - m) = 547864 ∧ m = -547550 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_for_m_l2214_221402


namespace NUMINAMATH_CALUDE_decimal_addition_l2214_221442

theorem decimal_addition : (0.0935 : ℚ) + (0.007 : ℚ) + (0.2 : ℚ) = (0.3005 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l2214_221442


namespace NUMINAMATH_CALUDE_negation_of_absolute_value_plus_square_nonnegative_l2214_221401

theorem negation_of_absolute_value_plus_square_nonnegative :
  (¬ ∀ x : ℝ, |x| + x^2 ≥ 0) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_absolute_value_plus_square_nonnegative_l2214_221401


namespace NUMINAMATH_CALUDE_fixed_point_on_line_l2214_221494

/-- The line (m-1)x+(2m-1)y=m-5 always passes through the point (9, -4) for all real m -/
theorem fixed_point_on_line (m : ℝ) : (m - 1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_on_line_l2214_221494


namespace NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l2214_221443

theorem systematic_sampling_smallest_number
  (n : ℕ) -- Total number of products
  (k : ℕ) -- Sample size
  (x : ℕ) -- A number in the sample
  (h1 : n = 80) -- Total number of products is 80
  (h2 : k = 5) -- Sample size is 5
  (h3 : x = 42) -- The number 42 is in the sample
  (h4 : x < n) -- The number in the sample is less than the total number of products
  : ∃ (interval : ℕ) (smallest : ℕ),
    interval = n / k ∧
    x = interval * 2 + smallest ∧
    smallest = 10 :=
by sorry

end NUMINAMATH_CALUDE_systematic_sampling_smallest_number_l2214_221443


namespace NUMINAMATH_CALUDE_average_towel_price_l2214_221473

def towel_price_1 : ℕ := 100
def towel_price_2 : ℕ := 150
def towel_price_3 : ℕ := 650

def towel_count_1 : ℕ := 3
def towel_count_2 : ℕ := 5
def towel_count_3 : ℕ := 2

def total_cost : ℕ := towel_price_1 * towel_count_1 + towel_price_2 * towel_count_2 + towel_price_3 * towel_count_3
def total_towels : ℕ := towel_count_1 + towel_count_2 + towel_count_3

theorem average_towel_price :
  total_cost / total_towels = 235 := by sorry

end NUMINAMATH_CALUDE_average_towel_price_l2214_221473


namespace NUMINAMATH_CALUDE_integer_solution_abc_l2214_221466

theorem integer_solution_abc : 
  ∀ a b c : ℤ, 1 < a ∧ a < b ∧ b < c ∧ ((a - 1) * (b - 1) * (c - 1) ∣ (a * b * c - 1)) → 
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_abc_l2214_221466


namespace NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2214_221411

theorem largest_x_satisfying_equation : 
  ∃ x : ℚ, x = 3/25 ∧ 
  (∀ y : ℚ, y ≥ 0 → Real.sqrt (3 * y) = 5 * y → y ≤ x) ∧
  Real.sqrt (3 * x) = 5 * x :=
by sorry

end NUMINAMATH_CALUDE_largest_x_satisfying_equation_l2214_221411


namespace NUMINAMATH_CALUDE_two_distinct_arrangements_l2214_221479

/-- Represents a face of a cube -/
inductive Face : Type
| F | E | A | B | H | J

/-- Represents an arrangement of numbers on a cube -/
def Arrangement := Face → Fin 6

/-- Two faces are adjacent if they share an edge -/
def adjacent : Face → Face → Prop :=
  sorry

/-- Two numbers are consecutive if they differ by 1 or are 1 and 6 -/
def consecutive (a b : Fin 6) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 5 ∧ b = 0) ∨ (a = 0 ∧ b = 5)

/-- An arrangement is valid if consecutive numbers are on adjacent faces -/
def valid_arrangement (arr : Arrangement) : Prop :=
  ∀ f1 f2 : Face, adjacent f1 f2 → consecutive (arr f1) (arr f2)

/-- Two arrangements are equivalent if they can be transformed into each other
    by cube symmetry or cyclic permutation of numbers -/
def equivalent_arrangements (arr1 arr2 : Arrangement) : Prop :=
  sorry

/-- The number of distinct valid arrangements -/
def num_distinct_arrangements : ℕ :=
  sorry

theorem two_distinct_arrangements :
  num_distinct_arrangements = 2 :=
sorry

end NUMINAMATH_CALUDE_two_distinct_arrangements_l2214_221479


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2214_221469

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h : a^3 + a*b + b^3 = 0) : 
  (a^10 + b^10) / (a + b)^10 = 1/18 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2214_221469


namespace NUMINAMATH_CALUDE_same_prime_factors_implies_power_of_two_l2214_221483

theorem same_prime_factors_implies_power_of_two (b m n : ℕ) 
  (hb : b ≠ 1) (hmn : m ≠ n) 
  (h_same_factors : ∀ p : ℕ, Nat.Prime p → (p ∣ (b^m - 1) ↔ p ∣ (b^n - 1))) :
  ∃ k : ℕ, b + 1 = 2^k := by
sorry

end NUMINAMATH_CALUDE_same_prime_factors_implies_power_of_two_l2214_221483


namespace NUMINAMATH_CALUDE_ducks_killed_per_year_is_correct_l2214_221459

/-- The number of ducks killed every year -/
def ducks_killed_per_year : ℕ := 20

/-- The original flock size -/
def original_flock_size : ℕ := 100

/-- The number of ducks born every year -/
def ducks_born_per_year : ℕ := 30

/-- The number of years before joining with another flock -/
def years_before_joining : ℕ := 5

/-- The size of the other flock -/
def other_flock_size : ℕ := 150

/-- The combined flock size after joining -/
def combined_flock_size : ℕ := 300

theorem ducks_killed_per_year_is_correct :
  original_flock_size + years_before_joining * (ducks_born_per_year - ducks_killed_per_year) + other_flock_size = combined_flock_size :=
by sorry

end NUMINAMATH_CALUDE_ducks_killed_per_year_is_correct_l2214_221459


namespace NUMINAMATH_CALUDE_hexagon_puzzle_solution_l2214_221449

/-- Represents the positions in the hexagon puzzle --/
inductive Position
| A | B | C | D | E | F

/-- Represents a valid assignment of digits to positions --/
def Assignment := Position → Fin 6

/-- Checks if an assignment is valid (uses each digit exactly once) --/
def isValidAssignment (a : Assignment) : Prop :=
  ∀ (i : Fin 6), ∃! (p : Position), a p = i

/-- Checks if an assignment satisfies the sum condition for all lines --/
def satisfiesSumCondition (a : Assignment) : Prop :=
  (a Position.A + a Position.C + 9 = 15) ∧
  (a Position.A + 8 + a Position.F = 15) ∧
  (7 + a Position.C + a Position.E = 15) ∧
  (7 + a Position.D + a Position.F = 15) ∧
  (9 + a Position.B + a Position.D = 15) ∧
  (a Position.A + a Position.D + a Position.E = 15)

/-- The main theorem stating the existence and uniqueness of a valid solution --/
theorem hexagon_puzzle_solution :
  ∃! (a : Assignment), isValidAssignment a ∧ satisfiesSumCondition a :=
sorry

end NUMINAMATH_CALUDE_hexagon_puzzle_solution_l2214_221449


namespace NUMINAMATH_CALUDE_range_of_a_l2214_221413

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a ≤ |x - 5| + |x - 3|) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2214_221413


namespace NUMINAMATH_CALUDE_unique_solution_of_equation_l2214_221447

theorem unique_solution_of_equation :
  ∃! x : ℝ, x ≠ 3 ∧ x + 36 / (x - 3) = -9 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_of_equation_l2214_221447


namespace NUMINAMATH_CALUDE_smallest_undefined_value_l2214_221495

theorem smallest_undefined_value (y : ℝ) :
  let f := fun y : ℝ => (y - 3) / (9 * y^2 - 56 * y + 7)
  let roots := {y : ℝ | 9 * y^2 - 56 * y + 7 = 0}
  ∃ (smallest : ℝ), smallest ∈ roots ∧ 
    (∀ y ∈ roots, y ≥ smallest) ∧
    (∀ z < smallest, f z ≠ 0⁻¹) :=
by sorry

end NUMINAMATH_CALUDE_smallest_undefined_value_l2214_221495


namespace NUMINAMATH_CALUDE_angle_trisection_l2214_221496

theorem angle_trisection (n : ℕ) (h : ¬ 3 ∣ n) :
  ∃ (a b : ℤ), 3 * a + n * b = 1 :=
by sorry

end NUMINAMATH_CALUDE_angle_trisection_l2214_221496


namespace NUMINAMATH_CALUDE_positive_sum_from_absolute_difference_l2214_221414

theorem positive_sum_from_absolute_difference (a b : ℝ) : 
  b - |a| > 0 → a + b > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_sum_from_absolute_difference_l2214_221414


namespace NUMINAMATH_CALUDE_circle_plus_k_circle_plus_k_k_l2214_221452

-- Define the ⊕ operation
def circle_plus (x y : ℝ) : ℝ := x^3 + x - y

-- Theorem statement
theorem circle_plus_k_circle_plus_k_k (k : ℝ) : circle_plus k (circle_plus k k) = k := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_k_circle_plus_k_k_l2214_221452


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2214_221478

/-- The area of the shaded region formed by a square with side length 10 cm
    and four quarter circles drawn at its corners is equal to 100 - 25π cm². -/
theorem shaded_area_square_with_quarter_circles :
  let square_side : ℝ := 10
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_radius : ℝ := square_side / 2
  let full_circle_area : ℝ := π * quarter_circle_radius ^ 2
  let shaded_area : ℝ := square_area - full_circle_area
  shaded_area = 100 - 25 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l2214_221478


namespace NUMINAMATH_CALUDE_equation_solutions_l2214_221428

theorem equation_solutions :
  (∃ x : ℝ, 0.5 * x + 1.1 = 6.5 - 1.3 * x ∧ x = 3) ∧
  (∃ x : ℝ, (1/6) * (3 * x - 9) = (2/5) * x - 3 ∧ x = -15) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2214_221428


namespace NUMINAMATH_CALUDE_levi_has_five_lemons_l2214_221406

/-- The number of lemons each person has -/
structure LemonCounts where
  levi : ℕ
  jayden : ℕ
  eli : ℕ
  ian : ℕ

/-- The conditions of the lemon problem -/
def LemonProblem (counts : LemonCounts) : Prop :=
  counts.jayden = counts.levi + 6 ∧
  counts.jayden * 3 = counts.eli ∧
  counts.eli * 2 = counts.ian ∧
  counts.levi + counts.jayden + counts.eli + counts.ian = 115

/-- Theorem stating that under the given conditions, Levi has 5 lemons -/
theorem levi_has_five_lemons :
  ∃ (counts : LemonCounts), LemonProblem counts ∧ counts.levi = 5 := by
  sorry

end NUMINAMATH_CALUDE_levi_has_five_lemons_l2214_221406


namespace NUMINAMATH_CALUDE_admission_ratio_theorem_l2214_221429

def admission_problem (a c : ℕ+) : Prop :=
  30 * a.val + 15 * c.val = 2550

def ratio_closest_to_one (a c : ℕ+) : Prop :=
  ∀ (x y : ℕ+), admission_problem x y →
    |((a:ℚ) / c) - 1| ≤ |((x:ℚ) / y) - 1|

theorem admission_ratio_theorem :
  ∃ (a c : ℕ+), admission_problem a c ∧ ratio_closest_to_one a c ∧ a.val = 57 ∧ c.val = 56 :=
sorry

end NUMINAMATH_CALUDE_admission_ratio_theorem_l2214_221429


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2214_221471

theorem divisibility_theorem (n : ℕ) (h : ∃ m : ℕ, 2^n - 2 = n * m) :
  ∃ k : ℕ, 2^(2^n - 1) - 2 = (2^n - 1) * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2214_221471


namespace NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2214_221410

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x ∈ Set.Ioo 0 2, P x) ↔ (∀ x ∈ Set.Ioo 0 2, ¬P x) := by sorry

theorem quadratic_inequality_negation :
  (¬ ∃ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x ∈ Set.Ioo 0 2, x^2 + 2*x + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_quadratic_inequality_negation_l2214_221410


namespace NUMINAMATH_CALUDE_probability_two_acceptable_cans_l2214_221418

theorem probability_two_acceptable_cans (total_cans : Nat) (acceptable_cans : Nat) 
  (h1 : total_cans = 6)
  (h2 : acceptable_cans = 4) : 
  (Nat.choose acceptable_cans 2 : ℚ) / (Nat.choose total_cans 2 : ℚ) = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_acceptable_cans_l2214_221418


namespace NUMINAMATH_CALUDE_new_stereo_price_l2214_221450

theorem new_stereo_price 
  (old_cost : ℝ) 
  (trade_in_percentage : ℝ) 
  (new_discount_percentage : ℝ) 
  (out_of_pocket : ℝ) 
  (h1 : old_cost = 250)
  (h2 : trade_in_percentage = 0.8)
  (h3 : new_discount_percentage = 0.25)
  (h4 : out_of_pocket = 250) :
  let trade_in_value := old_cost * trade_in_percentage
  let total_spent := trade_in_value + out_of_pocket
  let original_price := total_spent / (1 - new_discount_percentage)
  original_price = 600 := by sorry

end NUMINAMATH_CALUDE_new_stereo_price_l2214_221450


namespace NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2214_221448

theorem least_positive_integer_with_remainders : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (n % 2 = 1) ∧ 
  (n % 3 = 2) ∧ 
  (n % 4 = 3) ∧ 
  (n % 5 = 4) ∧ 
  (∀ m : ℕ, m > 0 ∧ m % 2 = 1 ∧ m % 3 = 2 ∧ m % 4 = 3 ∧ m % 5 = 4 → m ≥ n) ∧
  n = 59 := by
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_with_remainders_l2214_221448


namespace NUMINAMATH_CALUDE_polynomial_equality_l2214_221441

theorem polynomial_equality (a b : ℝ) :
  (∀ x : ℝ, (x - 2) * (x + 3) = x^2 + a*x + b) →
  (a = 1 ∧ b = -6) := by
sorry

end NUMINAMATH_CALUDE_polynomial_equality_l2214_221441


namespace NUMINAMATH_CALUDE_f_increasing_and_range_l2214_221484

noncomputable def f (x : ℝ) : ℝ := 1 - 2 / (2^x + 1)

theorem f_increasing_and_range :
  (∀ x y : ℝ, x < y → f x < f y) ∧
  (Set.range (fun x => f x) ∩ Set.Icc 0 1 = Set.Icc 0 (1/3)) := by sorry

end NUMINAMATH_CALUDE_f_increasing_and_range_l2214_221484


namespace NUMINAMATH_CALUDE_divisible_by_72_digits_l2214_221431

theorem divisible_by_72_digits (a b : Nat) : 
  a < 10 → b < 10 → 
  (42000 + 1000 * a + 40 + b) % 72 = 0 → 
  ((a = 8 ∧ b = 0) ∨ (a = 0 ∧ b = 8)) :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_72_digits_l2214_221431


namespace NUMINAMATH_CALUDE_student_grades_l2214_221487

theorem student_grades (grade1 grade2 grade3 : ℚ) : 
  grade1 = 60 → 
  grade3 = 85 → 
  (grade1 + grade2 + grade3) / 3 = 75 → 
  grade2 = 80 := by
sorry

end NUMINAMATH_CALUDE_student_grades_l2214_221487


namespace NUMINAMATH_CALUDE_arcsin_neg_one_l2214_221421

theorem arcsin_neg_one : Real.arcsin (-1) = -π / 2 := by sorry

end NUMINAMATH_CALUDE_arcsin_neg_one_l2214_221421


namespace NUMINAMATH_CALUDE_complex_power_eight_l2214_221482

theorem complex_power_eight : (2 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^8 = -128 - 128 * Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_eight_l2214_221482


namespace NUMINAMATH_CALUDE_triangle_problem_l2214_221489

-- Define the triangle
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides

-- State the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : Real.sin (t.A + t.C) = 8 * (Real.sin (t.B / 2))^2)
  (h2 : t.a + t.c = 6)
  (h3 : (1/2) * t.a * t.c * Real.sin t.B = 2) :
  Real.cos t.B = 15/17 ∧ t.b = 2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l2214_221489
