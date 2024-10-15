import Mathlib

namespace NUMINAMATH_GPT_rectangle_area_formula_l1020_102068

-- Define the given conditions: perimeter is 20, one side length is x
def rectangle_perimeter (P x : ℝ) (w : ℝ) : Prop := P = 2 * (x + w)
def rectangle_area (x w : ℝ) : ℝ := x * w

-- The theorem to prove
theorem rectangle_area_formula (x : ℝ) (h_perimeter : rectangle_perimeter 20 x (10 - x)) : 
  rectangle_area x (10 - x) = x * (10 - x) := 
by 
  sorry

end NUMINAMATH_GPT_rectangle_area_formula_l1020_102068


namespace NUMINAMATH_GPT_smallest_hiding_number_l1020_102064

/-- Define the concept of "hides" -/
def hides (A B : ℕ) : Prop :=
  ∃ (remove : ℕ → ℕ), remove A = B

/-- The smallest natural number that hides all numbers from 2000 to 2021 is 20012013456789 -/
theorem smallest_hiding_number : hides 20012013456789 2000 ∧ hides 20012013456789 2001 ∧ hides 20012013456789 2002 ∧
    hides 20012013456789 2003 ∧ hides 20012013456789 2004 ∧ hides 20012013456789 2005 ∧ hides 20012013456789 2006 ∧
    hides 20012013456789 2007 ∧ hides 20012013456789 2008 ∧ hides 20012013456789 2009 ∧ hides 20012013456789 2010 ∧
    hides 20012013456789 2011 ∧ hides 20012013456789 2012 ∧ hides 20012013456789 2013 ∧ hides 20012013456789 2014 ∧
    hides 20012013456789 2015 ∧ hides 20012013456789 2016 ∧ hides 20012013456789 2017 ∧ hides 20012013456789 2018 ∧
    hides 20012013456789 2019 ∧ hides 20012013456789 2020 ∧ hides 20012013456789 2021 :=
by
  sorry

end NUMINAMATH_GPT_smallest_hiding_number_l1020_102064


namespace NUMINAMATH_GPT_bake_sale_donation_l1020_102052

theorem bake_sale_donation :
  let total_earning := 400
  let cost_of_ingredients := 100
  let donation_homeless_piggy := 10
  let total_donation_homeless := 160
  let donation_homeless := total_donation_homeless - donation_homeless_piggy
  let available_for_donation := total_earning - cost_of_ingredients
  let donation_food_bank := available_for_donation - donation_homeless
  (donation_homeless / donation_food_bank) = 1 := 
by
  sorry

end NUMINAMATH_GPT_bake_sale_donation_l1020_102052


namespace NUMINAMATH_GPT_composite_number_N_l1020_102037

theorem composite_number_N (y : ℕ) (hy : y > 1) : ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = (y ^ 125 - 1) / (3 ^ 22 - 1) :=
by
  -- use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_composite_number_N_l1020_102037


namespace NUMINAMATH_GPT_magnitude_of_complex_l1020_102087

variable (z : ℂ)
variable (h : Complex.I * z = 3 - 4 * Complex.I)

theorem magnitude_of_complex :
  Complex.abs z = 5 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_complex_l1020_102087


namespace NUMINAMATH_GPT_max_composite_numbers_l1020_102039

-- Definitions and conditions
def is_composite (n : ℕ) : Prop := 2 < n ∧ ∃ d, d ∣ n ∧ 1 < d ∧ d < n

def less_than_1500 (n : ℕ) : Prop := n < 1500

def gcd_is_one (a b : ℕ) : Prop := Int.gcd a b = 1

-- The problem statement
theorem max_composite_numbers (numbers : List ℕ) (h_composite : ∀ n ∈ numbers, is_composite n) 
  (h_less_than_1500 : ∀ n ∈ numbers, less_than_1500 n)
  (h_pairwise_gcd_is_one : (numbers.Pairwise gcd_is_one)) : numbers.length ≤ 12 := 
  sorry

end NUMINAMATH_GPT_max_composite_numbers_l1020_102039


namespace NUMINAMATH_GPT_square_paper_side_length_l1020_102005

theorem square_paper_side_length :
  ∀ (edge_length : ℝ) (num_pieces : ℕ) (side_length : ℝ),
  edge_length = 12 ∧ num_pieces = 54 ∧ 6 * (edge_length ^ 2) = num_pieces * (side_length ^ 2)
  → side_length = 4 :=
by
  intros edge_length num_pieces side_length h
  sorry

end NUMINAMATH_GPT_square_paper_side_length_l1020_102005


namespace NUMINAMATH_GPT_sequence_uniquely_determined_l1020_102012

theorem sequence_uniquely_determined (a : ℕ → ℝ) (p q : ℝ) (a0 a1 : ℝ)
  (h : ∀ n, a (n + 2) = p * a (n + 1) + q * a n)
  (h0 : a 0 = a0)
  (h1 : a 1 = a1) :
  ∀ n, ∃! a_n, a n = a_n :=
sorry

end NUMINAMATH_GPT_sequence_uniquely_determined_l1020_102012


namespace NUMINAMATH_GPT_find_water_bottles_l1020_102016

def water_bottles (W A : ℕ) :=
  A = W + 6 ∧ W + A = 54 → W = 24

theorem find_water_bottles (W A : ℕ) (h1 : A = W + 6) (h2 : W + A = 54) : W = 24 :=
by sorry

end NUMINAMATH_GPT_find_water_bottles_l1020_102016


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1020_102058

-- Define \(\frac{1}{x} < 2\) and \(x > \frac{1}{2}\)
def condition1 (x : ℝ) : Prop := 1 / x < 2
def condition2 (x : ℝ) : Prop := x > 1 / 2

-- Theorem stating that condition1 is necessary but not sufficient for condition2
theorem necessary_but_not_sufficient (x : ℝ) : condition1 x → condition2 x ↔ true :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1020_102058


namespace NUMINAMATH_GPT_smallest_value_36k_minus_5l_l1020_102072

theorem smallest_value_36k_minus_5l (k l : ℕ) :
  ∃ k l, 0 < 36^k - 5^l ∧ (∀ k' l', (0 < 36^k' - 5^l' → 36^k - 5^l ≤ 36^k' - 5^l')) ∧ 36^k - 5^l = 11 :=
by sorry

end NUMINAMATH_GPT_smallest_value_36k_minus_5l_l1020_102072


namespace NUMINAMATH_GPT_arthur_bought_2_hamburgers_on_second_day_l1020_102098

theorem arthur_bought_2_hamburgers_on_second_day
  (H D X: ℕ)
  (h1: 3 * H + 4 * D = 10)
  (h2: D = 1)
  (h3: 2 * X + 3 * D = 7):
  X = 2 :=
by
  sorry

end NUMINAMATH_GPT_arthur_bought_2_hamburgers_on_second_day_l1020_102098


namespace NUMINAMATH_GPT_solve_3x_plus_5_squared_l1020_102071

theorem solve_3x_plus_5_squared (x : ℝ) (h : 5 * x - 6 = 15 * x + 21) : 
  3 * (x + 5) ^ 2 = 2523 / 100 :=
by
  sorry

end NUMINAMATH_GPT_solve_3x_plus_5_squared_l1020_102071


namespace NUMINAMATH_GPT_distance_between_trees_l1020_102022

-- Lean 4 statement for the proof problem
theorem distance_between_trees (n : ℕ) (yard_length : ℝ) (h_n : n = 26) (h_length : yard_length = 600) :
  yard_length / (n - 1) = 24 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_trees_l1020_102022


namespace NUMINAMATH_GPT_three_point_sixty_eight_as_fraction_l1020_102078

theorem three_point_sixty_eight_as_fraction : 3.68 = 92 / 25 := 
by 
  sorry

end NUMINAMATH_GPT_three_point_sixty_eight_as_fraction_l1020_102078


namespace NUMINAMATH_GPT_sum_reciprocal_of_roots_l1020_102086

variables {m n : ℝ}

-- Conditions: m and n are real roots of the quadratic equation x^2 + 4x - 1 = 0
def is_root (a : ℝ) : Prop := a^2 + 4 * a - 1 = 0

theorem sum_reciprocal_of_roots (hm : is_root m) (hn : is_root n) : 
  (1 / m) + (1 / n) = 4 :=
by sorry

end NUMINAMATH_GPT_sum_reciprocal_of_roots_l1020_102086


namespace NUMINAMATH_GPT_number_of_valid_sequences_l1020_102093

-- Definitions for conditions
def digit := Fin 10 -- Digit can be any number from 0 to 9
def is_odd (n : digit) : Prop := n.val % 2 = 1
def is_even (n : digit) : Prop := n.val % 2 = 0

def valid_sequence (s : Fin 8 → digit) : Prop :=
  ∀ i : Fin 7, (is_odd (s i) ↔ is_even (s (i+1)))

-- Theorem statement
theorem number_of_valid_sequences : 
  ∃ n, n = 781250 ∧ 
    ∃ s : (Fin 8 → digit), valid_sequence s :=
sorry -- Proof is not required

end NUMINAMATH_GPT_number_of_valid_sequences_l1020_102093


namespace NUMINAMATH_GPT_sum_of_integers_between_60_and_460_ending_in_2_is_10280_l1020_102088

-- We define the sequence.
def endsIn2Seq : List Int := List.range' 62 (452 + 1 - 62) 10  -- Generates [62, 72, ..., 452]

-- The sum of the sequence.
def sumEndsIn2Seq : Int := endsIn2Seq.sum

-- The theorem to prove the desired sum.
theorem sum_of_integers_between_60_and_460_ending_in_2_is_10280 :
  sumEndsIn2Seq = 10280 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_sum_of_integers_between_60_and_460_ending_in_2_is_10280_l1020_102088


namespace NUMINAMATH_GPT_petya_vacation_days_l1020_102094

-- Defining the conditions
def total_days : ℕ := 90

def swims (d : ℕ) : Prop := d % 2 = 0
def shops (d : ℕ) : Prop := d % 3 = 0
def solves_math (d : ℕ) : Prop := d % 5 = 0

def does_all (d : ℕ) : Prop := swims d ∧ shops d ∧ solves_math d

def does_any_task (d : ℕ) : Prop := swims d ∨ shops d ∨ solves_math d

-- "Pleasant" days definition: swims, not shops, not solves math
def is_pleasant_day (d : ℕ) : Prop := swims d ∧ ¬shops d ∧ ¬solves_math d
-- "Boring" days definition: does nothing
def is_boring_day (d : ℕ) : Prop := ¬does_any_task d

-- Theorem stating the number of pleasant and boring days
theorem petya_vacation_days :
  (∃ pleasant_days : Finset ℕ, pleasant_days.card = 24 ∧ ∀ d ∈ pleasant_days, is_pleasant_day d)
  ∧ (∃ boring_days : Finset ℕ, boring_days.card = 24 ∧ ∀ d ∈ boring_days, is_boring_day d) :=
by
  sorry

end NUMINAMATH_GPT_petya_vacation_days_l1020_102094


namespace NUMINAMATH_GPT_integer_squares_l1020_102035

theorem integer_squares (x y : ℤ) 
  (hx : ∃ a : ℤ, x + y = a^2)
  (h2x3y : ∃ b : ℤ, 2 * x + 3 * y = b^2)
  (h3xy : ∃ c : ℤ, 3 * x + y = c^2) : 
  x = 0 ∧ y = 0 := 
by { sorry }

end NUMINAMATH_GPT_integer_squares_l1020_102035


namespace NUMINAMATH_GPT_abs_gt_implies_nec_not_suff_l1020_102047

theorem abs_gt_implies_nec_not_suff {a b : ℝ} : 
  (|a| > b) → (∀ (a b : ℝ), a > b → |a| > b) ∧ ¬(∀ (a b : ℝ), |a| > b → a > b) :=
by
  sorry

end NUMINAMATH_GPT_abs_gt_implies_nec_not_suff_l1020_102047


namespace NUMINAMATH_GPT_smallest_n_divides_l1020_102017

theorem smallest_n_divides (m : ℕ) (h1 : m % 2 = 1) (h2 : m > 2) :
  ∃ n : ℕ, 2^(1988) = n ∧ 2^1989 ∣ m^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divides_l1020_102017


namespace NUMINAMATH_GPT_find_A_from_equation_and_conditions_l1020_102038

theorem find_A_from_equation_and_conditions 
  (A B C D : ℕ)
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D)
  (h4 : B ≠ C) (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : 10 * A + B ≠ 0)
  (h8 : 10 * 10 * 10 * A + 10 * 10 * B + 8 * 10 + 2 - (900 + C * 10 + 9) = 490 + 3 * 10 + D) :
  A = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_A_from_equation_and_conditions_l1020_102038


namespace NUMINAMATH_GPT_platform_length_l1020_102095

noncomputable def length_of_platform (L : ℝ) : Prop :=
  ∃ (a : ℝ), 
    -- Train starts from rest
    (0 : ℝ) * 24 + (1/2) * a * 24^2 = 300 ∧
    -- Train crosses a platform in 39 seconds
    (0 : ℝ) * 39 + (1/2) * a * 39^2 = 300 + L ∧
    -- Constant acceleration found
    a = (25 : ℝ) / 24

-- Claim that length of platform should be 492.19 meters
theorem platform_length : length_of_platform 492.19 :=
sorry

end NUMINAMATH_GPT_platform_length_l1020_102095


namespace NUMINAMATH_GPT_linear_equation_l1020_102053

noncomputable def is_linear (k : ℝ) : Prop :=
  2 * (|k|) = 1 ∧ k ≠ 1

theorem linear_equation (k : ℝ) : is_linear k ↔ k = -1 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_l1020_102053


namespace NUMINAMATH_GPT_rhombus_diagonal_l1020_102027

theorem rhombus_diagonal (side : ℝ) (short_diag : ℝ) (long_diag : ℝ) 
  (h1 : side = 37) (h2 : short_diag = 40) :
  long_diag = 62 :=
sorry

end NUMINAMATH_GPT_rhombus_diagonal_l1020_102027


namespace NUMINAMATH_GPT_g_g_g_3_equals_107_l1020_102021

def g (x : ℝ) : ℝ := 3 * x + 2

theorem g_g_g_3_equals_107 : g (g (g 3)) = 107 := 
by 
  sorry

end NUMINAMATH_GPT_g_g_g_3_equals_107_l1020_102021


namespace NUMINAMATH_GPT_S6_value_l1020_102004

noncomputable def S_m (x : ℝ) (m : ℕ) : ℝ := x^m + (1/x)^m

theorem S6_value (x : ℝ) (h : x + 1/x = 4) : S_m x 6 = 2700 :=
by
  -- Skipping proof
  sorry

end NUMINAMATH_GPT_S6_value_l1020_102004


namespace NUMINAMATH_GPT_rectangle_perimeter_l1020_102061

-- Conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def area_of_triangle (base height : ℕ) : ℕ :=
  (base * height) / 2

def area_of_rectangle (length width : ℕ) : ℕ :=
  length * width

def perimeter_of_rectangle (length width : ℕ) : ℕ :=
  2 * (length + width)

-- Given conditions from the problem
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15
def width_of_rectangle : ℕ := 6

-- Main theorem
theorem rectangle_perimeter :
  is_right_triangle a b c →
  area_of_triangle a b = area_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle →
  perimeter_of_rectangle (area_of_triangle a b / width_of_rectangle) width_of_rectangle = 30 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_l1020_102061


namespace NUMINAMATH_GPT_exterior_angle_polygon_l1020_102024

theorem exterior_angle_polygon (n : ℕ) (h : 360 / n = 72) : n = 5 :=
by
  sorry

end NUMINAMATH_GPT_exterior_angle_polygon_l1020_102024


namespace NUMINAMATH_GPT_no_pre_period_decimal_representation_l1020_102063

theorem no_pre_period_decimal_representation (m : ℕ) (h : Nat.gcd m 10 = 1) : ¬∃ k : ℕ, ∃ a : ℕ, 0 < a ∧ 10^a < m ∧ (10^a - 1) % m = k ∧ k ≠ 0 :=
sorry

end NUMINAMATH_GPT_no_pre_period_decimal_representation_l1020_102063


namespace NUMINAMATH_GPT_ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l1020_102044

theorem ellipse_equation_x_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 4 ∧ b = 3 ∧ a = 5 ∧ (x^2 / a^2) + (y^2 / b^2) = 1 := by
  sorry

theorem ellipse_equation_y_axis_foci (x y : ℝ) (h : 3 * x + 4 * y = 12) :
  ∃ a b c : ℝ, c = 3 ∧ b = 4 ∧ a = 5 ∧ (x^2 / b^2) + (y^2 / a^2) = 1 := by
  sorry

end NUMINAMATH_GPT_ellipse_equation_x_axis_foci_ellipse_equation_y_axis_foci_l1020_102044


namespace NUMINAMATH_GPT_solution_set_inequality_l1020_102099

theorem solution_set_inequality (a x : ℝ) (h : 0 < a ∧ a < 1) : 
  ((a - x) * (x - (1 / a)) > 0) ↔ (a < x ∧ x < 1 / a) :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l1020_102099


namespace NUMINAMATH_GPT_gcf_120_180_240_l1020_102090

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end NUMINAMATH_GPT_gcf_120_180_240_l1020_102090


namespace NUMINAMATH_GPT_triangle_largest_angle_l1020_102040

theorem triangle_largest_angle (x : ℝ) (hx : x + 2 * x + 3 * x = 180) :
  3 * x = 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_largest_angle_l1020_102040


namespace NUMINAMATH_GPT_number_of_plants_l1020_102001

--- The given problem conditions and respective proof setup
axiom green_leaves_per_plant : ℕ
axiom yellow_turn_fall_off : ℕ
axiom green_leaves_total : ℕ

def one_third (n : ℕ) : ℕ := n / 3

-- Specify the given conditions
axiom leaves_per_plant_cond : green_leaves_per_plant = 18
axiom fall_off_cond : yellow_turn_fall_off = one_third green_leaves_per_plant
axiom total_leaves_cond : green_leaves_total = 36

-- Proof statement for the number of tea leaf plants
theorem number_of_plants : 
  (green_leaves_per_plant - yellow_turn_fall_off) * 3 = green_leaves_total :=
by
  sorry

end NUMINAMATH_GPT_number_of_plants_l1020_102001


namespace NUMINAMATH_GPT_boxcar_capacity_ratio_l1020_102070

theorem boxcar_capacity_ratio :
  ∀ (total_capacity : ℕ)
    (num_red num_blue num_black : ℕ)
    (black_capacity blue_capacity : ℕ)
    (red_capacity : ℕ),
    num_red = 3 →
    num_blue = 4 →
    num_black = 7 →
    black_capacity = 4000 →
    blue_capacity = 2 * black_capacity →
    total_capacity = 132000 →
    total_capacity = num_red * red_capacity + num_blue * blue_capacity + num_black * black_capacity →
    (red_capacity / blue_capacity = 3) :=
by
  intros total_capacity num_red num_blue num_black black_capacity blue_capacity red_capacity
         h_num_red h_num_blue h_num_black h_black_capacity h_blue_capacity h_total_capacity h_combined_capacity
  sorry

end NUMINAMATH_GPT_boxcar_capacity_ratio_l1020_102070


namespace NUMINAMATH_GPT_product_has_no_linear_term_l1020_102062

theorem product_has_no_linear_term (m : ℝ) (h : ((x : ℝ) → (x - m) * (x - 3) = x^2 + 3 * m)) : m = -3 := 
by
  sorry

end NUMINAMATH_GPT_product_has_no_linear_term_l1020_102062


namespace NUMINAMATH_GPT_factorization_identity_l1020_102097

-- Define the initial expression
def initial_expr (x : ℝ) : ℝ := 2 * x^2 - 2

-- Define the factorized form
def factorized_expr (x : ℝ) : ℝ := 2 * (x + 1) * (x - 1)

-- The theorem stating the equality
theorem factorization_identity (x : ℝ) : initial_expr x = factorized_expr x := 
by sorry

end NUMINAMATH_GPT_factorization_identity_l1020_102097


namespace NUMINAMATH_GPT_cubic_sum_identity_l1020_102084

theorem cubic_sum_identity (a b c : ℝ) (h1 : a + b + c = 13) (h2 : ab + ac + bc = 40) : 
  a^3 + b^3 + c^3 - 3 * a * b * c = 637 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_identity_l1020_102084


namespace NUMINAMATH_GPT_class_B_more_uniform_than_class_A_l1020_102085

-- Definitions based on the given problem
def class_height_variance (class_name : String) : ℝ :=
  if class_name = "A" then 3.24 else if class_name = "B" then 1.63 else 0

-- The theorem statement proving that Class B has more uniform heights (smaller variance)
theorem class_B_more_uniform_than_class_A :
  class_height_variance "B" < class_height_variance "A" :=
by
  sorry

end NUMINAMATH_GPT_class_B_more_uniform_than_class_A_l1020_102085


namespace NUMINAMATH_GPT_MitchWorks25Hours_l1020_102096

noncomputable def MitchWorksHours : Prop :=
  let weekday_earnings_rate := 3
  let weekend_earnings_rate := 6
  let weekly_earnings := 111
  let weekend_hours := 6
  let weekday_hours (x : ℕ) := 5 * x
  let weekend_earnings := weekend_hours * weekend_earnings_rate
  let weekday_earnings (x : ℕ) := x * weekday_earnings_rate
  let total_weekday_earnings (x : ℕ) := weekly_earnings - weekend_earnings
  ∀ (x : ℕ), weekday_earnings x = total_weekday_earnings x → x = 25

theorem MitchWorks25Hours : MitchWorksHours := by
  sorry

end NUMINAMATH_GPT_MitchWorks25Hours_l1020_102096


namespace NUMINAMATH_GPT_investment_of_D_l1020_102033

/--
Given C and D started a business where C invested Rs. 1000 and D invested some amount.
They made a total profit of Rs. 500, and D's share of the profit is Rs. 100.
So, how much did D invest in the business?
-/
theorem investment_of_D 
  (C_invested : ℕ) (D_share : ℕ) (total_profit : ℕ) 
  (H1 : C_invested = 1000) 
  (H2 : D_share = 100) 
  (H3 : total_profit = 500) 
  : ∃ D : ℕ, D = 250 :=
by
  sorry

end NUMINAMATH_GPT_investment_of_D_l1020_102033


namespace NUMINAMATH_GPT_kaleb_first_load_pieces_l1020_102077

-- Definitions of given conditions
def total_pieces : ℕ := 39
def num_equal_loads : ℕ := 5
def pieces_per_load : ℕ := 4

-- Definition for calculation of pieces in equal loads
def pieces_in_equal_loads : ℕ := num_equal_loads * pieces_per_load

-- Definition for pieces in the first load
def pieces_in_first_load : ℕ := total_pieces - pieces_in_equal_loads

-- Statement to prove that the pieces in the first load is 19
theorem kaleb_first_load_pieces : pieces_in_first_load = 19 := 
by
  -- The proof is skipped
  sorry

end NUMINAMATH_GPT_kaleb_first_load_pieces_l1020_102077


namespace NUMINAMATH_GPT_new_ticket_price_l1020_102042

theorem new_ticket_price (a : ℕ) (x : ℝ) (initial_price : ℝ) (revenue_increase : ℝ) (spectator_increase : ℝ)
  (h₀ : initial_price = 25)
  (h₁ : spectator_increase = 1.5)
  (h₂ : revenue_increase = 1.14)
  (h₃ : x = 0.76):
  initial_price * x = 19 :=
by
  sorry

end NUMINAMATH_GPT_new_ticket_price_l1020_102042


namespace NUMINAMATH_GPT_solve_trigonometric_equation_l1020_102030

theorem solve_trigonometric_equation :
  ∃ (S : Finset ℝ), (∀ X ∈ S, 0 < X ∧ X < 360 ∧ 1 + 2 * Real.sin (X * Real.pi / 180) - 4 * (Real.sin (X * Real.pi / 180))^2 - 8 * (Real.sin (X * Real.pi / 180))^3 = 0) ∧ S.card = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_trigonometric_equation_l1020_102030


namespace NUMINAMATH_GPT_probability_of_log2_condition_l1020_102000

noncomputable def probability_log_condition : ℝ :=
  let a := 0
  let b := 9
  let log_lower_bound := 1
  let log_upper_bound := 2
  let exp_lower_bound := 2^log_lower_bound
  let exp_upper_bound := 2^log_upper_bound
  (exp_upper_bound - exp_lower_bound) / (b - a)

theorem probability_of_log2_condition :
  probability_log_condition = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_log2_condition_l1020_102000


namespace NUMINAMATH_GPT_count_neither_multiples_of_2_nor_3_l1020_102091

theorem count_neither_multiples_of_2_nor_3 : 
  let count_multiples (k n : ℕ) : ℕ := n / k
  let total_numbers := 100
  let multiples_of_2 := count_multiples 2 total_numbers
  let multiples_of_3 := count_multiples 3 total_numbers
  let multiples_of_6 := count_multiples 6 total_numbers
  let multiples_of_2_or_3 := multiples_of_2 + multiples_of_3 - multiples_of_6
  total_numbers - multiples_of_2_or_3 = 33 :=
by 
  sorry

end NUMINAMATH_GPT_count_neither_multiples_of_2_nor_3_l1020_102091


namespace NUMINAMATH_GPT_largest_neg_int_solution_l1020_102041

theorem largest_neg_int_solution :
  ∃ x : ℤ, 26 * x + 8 ≡ 4 [ZMOD 18] ∧ ∀ y : ℤ, 26 * y + 8 ≡ 4 [ZMOD 18] → y < -14 → false :=
by
  sorry

end NUMINAMATH_GPT_largest_neg_int_solution_l1020_102041


namespace NUMINAMATH_GPT_sum_of_money_l1020_102050

theorem sum_of_money (P R : ℝ) (h : (P * 2 * (R + 3) / 100) = (P * 2 * R / 100) + 300) : P = 5000 :=
by
    -- We are given that the sum of money put at 2 years SI rate is Rs. 300 more when rate is increased by 3%.
    sorry

end NUMINAMATH_GPT_sum_of_money_l1020_102050


namespace NUMINAMATH_GPT_find_a_l1020_102051

theorem find_a (a : ℝ) (h1 : ∀ x : ℝ, a^(2*x - 4) ≤ 2^(x^2 - 2*x)) (ha_pos : a > 0) (ha_neq1 : a ≠ 1) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1020_102051


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_term_ratio_l1020_102032

theorem arithmetic_sequence_geometric_term_ratio (a : ℕ → ℤ) (d : ℤ) (h₀ : d ≠ 0)
  (h₁ : a 1 = a 1)
  (h₂ : a 3 = a 1 + 2 * d)
  (h₃ : a 4 = a 1 + 3 * d)
  (h_geom : (a 1 + 2 * d)^2 = a 1 * (a 1 + 3 * d)) :
  (a 1 + a 5 + a 17) / (a 2 + a 6 + a 18) = 8 / 11 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_term_ratio_l1020_102032


namespace NUMINAMATH_GPT_find_x_l1020_102018

-- Definitions based on the problem conditions
def angle_CDE : ℝ := 90 -- angle CDE in degrees
def angle_ECB : ℝ := 68 -- angle ECB in degrees

-- Theorem statement
theorem find_x (x : ℝ) 
  (h1 : angle_CDE = 90) 
  (h2 : angle_ECB = 68) 
  (h3 : angle_CDE + x + angle_ECB = 180) : 
  x = 22 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l1020_102018


namespace NUMINAMATH_GPT_evaluate_difference_of_squares_l1020_102066

theorem evaluate_difference_of_squares :
  (50^2 - 30^2 = 1600) :=
by sorry

end NUMINAMATH_GPT_evaluate_difference_of_squares_l1020_102066


namespace NUMINAMATH_GPT_parallelogram_base_l1020_102080

theorem parallelogram_base (A h b : ℝ) (hA : A = 375) (hh : h = 15) : b = 25 :=
by
  sorry

end NUMINAMATH_GPT_parallelogram_base_l1020_102080


namespace NUMINAMATH_GPT_sum_of_leading_digits_l1020_102013

def leading_digit (n : ℕ) (x : ℝ) : ℕ := sorry

def M := 10^500 - 1

def g (r : ℕ) : ℕ := leading_digit r (M^(1 / r))

theorem sum_of_leading_digits :
  g 3 + g 4 + g 5 + g 7 + g 8 = 10 := sorry

end NUMINAMATH_GPT_sum_of_leading_digits_l1020_102013


namespace NUMINAMATH_GPT_solve_k_values_l1020_102046

def has_positive_integer_solution (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = k * a * b * c

def infinitely_many_solutions (k : ℕ) : Prop :=
  ∃ (a b c : ℕ → ℕ), (∀ n, a n > 0 ∧ b n > 0 ∧ c n > 0 ∧ a n^2 + b n^2 + c n^2 = k * a n * b n * c n) ∧
  (∀ n, ∃ x y: ℤ, x^2 + y^2 = (a n * b n))

theorem solve_k_values :
  ∃ k : ℕ, (k = 1 ∨ k = 3) ∧ has_positive_integer_solution k ∧ infinitely_many_solutions k :=
sorry

end NUMINAMATH_GPT_solve_k_values_l1020_102046


namespace NUMINAMATH_GPT_distance_focus_asymptote_l1020_102002

noncomputable def focus := (Real.sqrt 6 / 2, 0)
def asymptote (x y : ℝ) := x - Real.sqrt 2 * y = 0
def hyperbola (x y : ℝ) := x^2 - 2 * y^2 = 1

theorem distance_focus_asymptote :
  let d := (Real.sqrt 6 / 2, 0)
  let A := 1
  let B := -Real.sqrt 2
  let C := 0
  let numerator := abs (A * d.1 + B * d.2 + C)
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator = Real.sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_distance_focus_asymptote_l1020_102002


namespace NUMINAMATH_GPT_P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l1020_102057

-- Define the problem conditions and questions
def P_1 (n : ℕ) : ℚ := sorry
def P_2 (n : ℕ) : ℚ := sorry

-- Part (a)
theorem P2_3_eq_2_3 : P_2 3 = 2 / 3 := sorry

-- Part (b)
theorem P1_n_eq_1_n (n : ℕ) (h : n ≥ 1): P_1 n = 1 / n := sorry

-- Part (c)
theorem P2_recurrence (n : ℕ) (h : n ≥ 2) : 
  P_2 n = (2 / n) * P_1 (n-1) + ((n-2) / n) * P_2 (n-1) := sorry

-- Part (d)
theorem P2_n_eq_2_n (n : ℕ) (h : n ≥ 1): P_2 n = 2 / n := sorry

end NUMINAMATH_GPT_P2_3_eq_2_3_P1_n_eq_1_n_P2_recurrence_P2_n_eq_2_n_l1020_102057


namespace NUMINAMATH_GPT_largest_integer_x_l1020_102073

theorem largest_integer_x (x : ℤ) : (x / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) → x ≤ 7 ∧ (7 / 4 : ℚ) + (3 / 7 : ℚ) < (9 / 4 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_x_l1020_102073


namespace NUMINAMATH_GPT_no_solution_for_equation_l1020_102036

theorem no_solution_for_equation (x y z : ℤ) : x^3 + y^3 ≠ 9 * z + 5 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_for_equation_l1020_102036


namespace NUMINAMATH_GPT_sum_of_polynomial_roots_l1020_102010

theorem sum_of_polynomial_roots:
  ∀ (a b : ℝ),
  (a^2 - 5 * a + 6 = 0) ∧ (b^2 - 5 * b + 6 = 0) →
  a^3 + a^4 * b^2 + a^2 * b^4 + b^3 + a * b^3 + b * a^3 = 683 := by
  intros a b h
  sorry

end NUMINAMATH_GPT_sum_of_polynomial_roots_l1020_102010


namespace NUMINAMATH_GPT_solve_eqn_l1020_102026

noncomputable def root_expr (a b k x : ℝ) : ℝ := Real.sqrt ((a + b * Real.sqrt k)^x)

theorem solve_eqn: {x : ℝ | root_expr 3 2 2 x + root_expr 3 (-2) 2 x = 6} = {2, -2} :=
by
  sorry

end NUMINAMATH_GPT_solve_eqn_l1020_102026


namespace NUMINAMATH_GPT_john_finish_work_alone_in_48_days_l1020_102019

variable {J R : ℝ}

theorem john_finish_work_alone_in_48_days
  (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 2 / 3)
  (h3 : 16 * J = 1 / 3) :
  1 / J = 48 := 
by
  sorry

end NUMINAMATH_GPT_john_finish_work_alone_in_48_days_l1020_102019


namespace NUMINAMATH_GPT_integer_solution_l1020_102029

theorem integer_solution (n m : ℤ) (h : (n + 2)^4 - n^4 = m^3) : (n = -1 ∧ m = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_l1020_102029


namespace NUMINAMATH_GPT_total_dollars_l1020_102054

theorem total_dollars (mark_dollars : ℚ) (carolyn_dollars : ℚ) (mark_money : mark_dollars = 7 / 8) (carolyn_money : carolyn_dollars = 2 / 5) :
  mark_dollars + carolyn_dollars = 1.275 := sorry

end NUMINAMATH_GPT_total_dollars_l1020_102054


namespace NUMINAMATH_GPT_incorrect_arrangements_hello_l1020_102055

-- Given conditions: the word "hello" with letters 'h', 'e', 'l', 'l', 'o'
def letters : List Char := ['h', 'e', 'l', 'l', 'o']

-- The number of permutations of the letters in "hello" excluding the correct order
-- We need to prove that the number of incorrect arrangements is 59.
theorem incorrect_arrangements_hello : 
  (List.permutations letters).length - 1 = 59 := 
by sorry

end NUMINAMATH_GPT_incorrect_arrangements_hello_l1020_102055


namespace NUMINAMATH_GPT_treaty_signed_on_friday_l1020_102031

def days_between (start_date : Nat) (end_date : Nat) : Nat := sorry

def day_of_week (start_day : Nat) (days_elapsed : Nat) : Nat :=
  (start_day + days_elapsed) % 7

def is_leap_year (year : Nat) : Bool :=
  if year % 4 = 0 then
    if year % 100 = 0 then
      if year % 400 = 0 then true else false
    else true
  else false

noncomputable def days_from_1802_to_1814 : Nat :=
  let leap_years := [1804, 1808, 1812]
  let normal_year_days := 365 * 9
  let leap_year_days := 366 * 3
  normal_year_days + leap_year_days

noncomputable def days_from_feb_5_to_apr_11_1814 : Nat :=
  24 + 31 + 11 -- days in February, March, and April 11

noncomputable def total_days_elapsed : Nat :=
  days_from_1802_to_1814 + days_from_feb_5_to_apr_11_1814

noncomputable def start_day : Nat := 5 -- Friday (0 = Sunday, ..., 5 = Friday, 6 = Saturday)

theorem treaty_signed_on_friday : day_of_week start_day total_days_elapsed = 5 := sorry

end NUMINAMATH_GPT_treaty_signed_on_friday_l1020_102031


namespace NUMINAMATH_GPT_income_percentage_increase_l1020_102006

theorem income_percentage_increase (b : ℝ) (a : ℝ) (h : a = b * 0.75) :
  (b - a) / a * 100 = 33.33 :=
by
  sorry

end NUMINAMATH_GPT_income_percentage_increase_l1020_102006


namespace NUMINAMATH_GPT_paint_total_gallons_l1020_102069

theorem paint_total_gallons
  (white_paint_gallons : ℕ)
  (blue_paint_gallons : ℕ)
  (h_wp : white_paint_gallons = 660)
  (h_bp : blue_paint_gallons = 6029) :
  white_paint_gallons + blue_paint_gallons = 6689 := 
by
  sorry

end NUMINAMATH_GPT_paint_total_gallons_l1020_102069


namespace NUMINAMATH_GPT_sum_of_transformed_parabolas_is_non_horizontal_line_l1020_102007

theorem sum_of_transformed_parabolas_is_non_horizontal_line
    (a b c : ℝ)
    (f g : ℝ → ℝ)
    (hf : ∀ x, f x = a * (x - 8)^2 + b * (x - 8) + c)
    (hg : ∀ x, g x = -a * (x + 8)^2 - b * (x + 8) - (c - 3)) :
    ∃ m q : ℝ, ∀ x : ℝ, (f x + g x) = m * x + q ∧ m ≠ 0 :=
by sorry

end NUMINAMATH_GPT_sum_of_transformed_parabolas_is_non_horizontal_line_l1020_102007


namespace NUMINAMATH_GPT_correct_operations_l1020_102075

theorem correct_operations : 6 * 3 + 4 + 2 = 24 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_correct_operations_l1020_102075


namespace NUMINAMATH_GPT_taxi_speed_l1020_102023

theorem taxi_speed (v : ℝ) (hA : ∀ v : ℝ, 3 * v = 6 * (v - 30)) : v = 60 :=
by
  sorry

end NUMINAMATH_GPT_taxi_speed_l1020_102023


namespace NUMINAMATH_GPT_solve_inequality_l1020_102015

theorem solve_inequality (x : ℝ) : 
  (x / (x^2 + x - 6) ≥ 0) ↔ (x < -3) ∨ (x = 0) ∨ (0 < x ∧ x < 2) :=
by 
  sorry 

end NUMINAMATH_GPT_solve_inequality_l1020_102015


namespace NUMINAMATH_GPT_cycling_speed_l1020_102089

-- Definitions based on given conditions.
def ratio_L_B : ℕ := 1
def ratio_B_L : ℕ := 2
def area_of_park : ℕ := 20000
def time_in_minutes : ℕ := 6

-- The question translated to Lean 4 statement.
theorem cycling_speed (L B : ℕ) (h1 : ratio_L_B * B = ratio_B_L * L)
  (h2 : L * B = area_of_park)
  (h3 : B = 2 * L) :
  (2 * L + 2 * B) / (time_in_minutes / 60) = 6000 := by
  sorry

end NUMINAMATH_GPT_cycling_speed_l1020_102089


namespace NUMINAMATH_GPT_target_expression_l1020_102065

variable (a b : ℤ)

-- Definitions based on problem conditions
def op1 (x y : ℤ) : ℤ := x + y  -- "!" could be addition
def op2 (x y : ℤ) : ℤ := x - y  -- "?" could be subtraction in one order

-- Using these operations to create expressions
def exp1 (a b : ℤ) := op1 (op2 a b) (op2 b a)

def exp2 (x y : ℤ) := op2 (op2 x 0) (op2 0 y)

-- The final expression we need to check
def final_exp (a b : ℤ) := exp1 (20 * a) (18 * b)

-- Theorem proving the final expression equals target
theorem target_expression : final_exp a b = 20 * a - 18 * b :=
sorry

end NUMINAMATH_GPT_target_expression_l1020_102065


namespace NUMINAMATH_GPT_intersection_complement_l1020_102082

def A : Set ℝ := {x | 1 < x ∧ x < 4}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem intersection_complement :
  A ∩ ({x | x < -1 ∨ x > 3} : Set ℝ) = {x | 3 < x ∧ x < 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_complement_l1020_102082


namespace NUMINAMATH_GPT_carla_marble_purchase_l1020_102081

variable (started_with : ℕ) (now_has : ℕ) (bought : ℕ)

theorem carla_marble_purchase (h1 : started_with = 53) (h2 : now_has = 187) : bought = 134 := by
  sorry

end NUMINAMATH_GPT_carla_marble_purchase_l1020_102081


namespace NUMINAMATH_GPT_visits_per_hour_l1020_102008

open Real

theorem visits_per_hour (price_per_visit : ℝ) (hours_per_day : ℕ) (days_per_month : ℕ) (total_earnings : ℝ) 
  (h_price : price_per_visit = 0.10)
  (h_hours : hours_per_day = 24)
  (h_days : days_per_month = 30)
  (h_earnings : total_earnings = 3600) :
  (total_earnings / (price_per_visit * hours_per_day * days_per_month) : ℝ) = 50 :=
by
  sorry

end NUMINAMATH_GPT_visits_per_hour_l1020_102008


namespace NUMINAMATH_GPT_marla_errand_time_l1020_102025

theorem marla_errand_time :
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  total_time = 110 := by
  let drive_time_one_way := 20
  let parent_teacher_night := 70
  let drive_time_total := drive_time_one_way * 2
  let total_time := drive_time_total + parent_teacher_night
  sorry

end NUMINAMATH_GPT_marla_errand_time_l1020_102025


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_properties_l1020_102074

theorem arithmetic_sequence_sum_properties {S : ℕ → ℝ} {a : ℕ → ℝ} (d : ℝ)
  (h1 : S 6 > S 7) (h2 : S 7 > S 5) :
  let a6 := (S 6 - S 5)
  let a7 := (S 7 - S 6)
  (d = a7 - a6) →
  d < 0 ∧ S 12 > 0 ∧ ¬(∀ n, S n = S 11) ∧ abs a6 > abs a7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_properties_l1020_102074


namespace NUMINAMATH_GPT_trajectory_C_find_m_l1020_102067

noncomputable def trajectory_C_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 7

theorem trajectory_C (x y : ℝ) (hx : trajectory_C_eq x y) :
  (x - 3)^2 + y^2 = 7 := by
  sorry

theorem find_m (m : ℝ) : (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = 3 + m ∧ x1 * x2 + (1/(2:ℝ)) * ((m^2 + 2)/(2:ℝ)) = 0 ∧ x1 * x2 + (x1 - m) * (x2 - m) = 0) → m = 1 ∨ m = 2 := by
  sorry

end NUMINAMATH_GPT_trajectory_C_find_m_l1020_102067


namespace NUMINAMATH_GPT_find_a_in_triangle_l1020_102076

theorem find_a_in_triangle (C : ℝ) (b c : ℝ) (hC : C = 60) (hb : b = 1) (hc : c = Real.sqrt 3) :
  ∃ (a : ℝ), a = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_in_triangle_l1020_102076


namespace NUMINAMATH_GPT_find_m_no_solution_l1020_102060

-- Define the condition that the equation has no solution
def no_solution (m : ℤ) : Prop :=
  ∀ x : ℤ, (x + m)/(4 - x^2) + x / (x - 2) ≠ 1

-- State the proof problem in Lean 4
theorem find_m_no_solution : ∀ m : ℤ, no_solution m → (m = 2 ∨ m = 6) :=
by
  sorry

end NUMINAMATH_GPT_find_m_no_solution_l1020_102060


namespace NUMINAMATH_GPT_find_total_results_l1020_102045

noncomputable def total_results (S : ℕ) (n : ℕ) (sum_first6 sum_last6 sixth_result : ℕ) :=
  (S = 52 * n) ∧ (sum_first6 = 6 * 49) ∧ (sum_last6 = 6 * 52) ∧ (sixth_result = 34)

theorem find_total_results {S n sum_first6 sum_last6 sixth_result : ℕ} :
  total_results S n sum_first6 sum_last6 sixth_result → n = 11 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_find_total_results_l1020_102045


namespace NUMINAMATH_GPT_time_after_3577_minutes_l1020_102009

-- Definitions
def startingTime : Nat := 6 * 60 -- 6:00 PM in minutes
def startDate : String := "2020-12-31"
def durationMinutes : Nat := 3577
def minutesInHour : Nat := 60
def hoursInDay : Nat := 24

-- Theorem to prove that 3577 minutes after 6:00 PM on December 31, 2020 is January 3 at 5:37 AM
theorem time_after_3577_minutes : 
  (durationMinutes + startingTime) % (hoursInDay * minutesInHour) = 5 * minutesInHour + 37 :=
  by
  sorry -- proof goes here

end NUMINAMATH_GPT_time_after_3577_minutes_l1020_102009


namespace NUMINAMATH_GPT_remainder_of_prime_when_divided_by_240_l1020_102092

theorem remainder_of_prime_when_divided_by_240 (n : ℕ) (hn : n > 0) (hp : Nat.Prime (2^n + 1)) : (2^n + 1) % 240 = 17 := 
sorry

end NUMINAMATH_GPT_remainder_of_prime_when_divided_by_240_l1020_102092


namespace NUMINAMATH_GPT_least_8_heavy_three_digit_l1020_102003

def is_8_heavy (n : ℕ) : Prop :=
  n % 8 > 6

def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

theorem least_8_heavy_three_digit : ∃ n : ℕ, is_three_digit n ∧ is_8_heavy n ∧ ∀ m : ℕ, is_three_digit m ∧ is_8_heavy m → n ≤ m := 
sorry

end NUMINAMATH_GPT_least_8_heavy_three_digit_l1020_102003


namespace NUMINAMATH_GPT_find_a_if_odd_l1020_102011

theorem find_a_if_odd :
  ∀ (a : ℝ), (∀ x : ℝ, (a * (-x)^3 + (a - 1) * (-x)^2 + (-x) = -(a * x^3 + (a - 1) * x^2 + x))) → 
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_odd_l1020_102011


namespace NUMINAMATH_GPT_sides_of_regular_polygon_l1020_102059

theorem sides_of_regular_polygon 
    (sum_interior_angles : ∀ n : ℕ, (n - 2) * 180 = 1440) :
  ∃ n : ℕ, n = 10 :=
by
  sorry

end NUMINAMATH_GPT_sides_of_regular_polygon_l1020_102059


namespace NUMINAMATH_GPT_divides_x_by_5_l1020_102014

theorem divides_x_by_5 (x y : ℤ) (hx1 : 1 < x) (hx_pos : 0 < x) (hy_pos : 0 < y) 
(h_eq : 2 * x^2 - 1 = y^15) : 5 ∣ x := by
  sorry

end NUMINAMATH_GPT_divides_x_by_5_l1020_102014


namespace NUMINAMATH_GPT_percentage_of_teachers_without_issues_l1020_102020

theorem percentage_of_teachers_without_issues (total_teachers : ℕ) 
    (high_bp_teachers : ℕ) (heart_issue_teachers : ℕ) 
    (both_issues_teachers : ℕ) (h1 : total_teachers = 150) 
    (h2 : high_bp_teachers = 90) 
    (h3 : heart_issue_teachers = 60) 
    (h4 : both_issues_teachers = 30) : 
    (total_teachers - (high_bp_teachers + heart_issue_teachers - both_issues_teachers)) / total_teachers * 100 = 20 :=
by sorry

end NUMINAMATH_GPT_percentage_of_teachers_without_issues_l1020_102020


namespace NUMINAMATH_GPT_smaller_angle_at_8_15_pm_l1020_102056

noncomputable def smaller_angle_between_clock_hands (minute_hand_degrees_per_min: ℝ) (hour_hand_degrees_per_min: ℝ) (time_in_minutes: ℝ) : ℝ := sorry

theorem smaller_angle_at_8_15_pm :
  smaller_angle_between_clock_hands 6 0.5 495 = 157.5 :=
sorry

end NUMINAMATH_GPT_smaller_angle_at_8_15_pm_l1020_102056


namespace NUMINAMATH_GPT_total_birds_times_types_l1020_102049

-- Defining the number of adults and offspring for each type of bird.
def num_ducks1 : ℕ := 2
def num_ducklings1 : ℕ := 5
def num_ducks2 : ℕ := 6
def num_ducklings2 : ℕ := 3
def num_ducks3 : ℕ := 9
def num_ducklings3 : ℕ := 6

def num_geese : ℕ := 4
def num_goslings : ℕ := 7

def num_swans : ℕ := 3
def num_cygnets : ℕ := 4

-- Calculate total number of birds
def total_ducks := (num_ducks1 * num_ducklings1 + num_ducks1) + (num_ducks2 * num_ducklings2 + num_ducks2) +
                      (num_ducks3 * num_ducklings3 + num_ducks3)

def total_geese := num_geese * num_goslings + num_geese
def total_swans := num_swans * num_cygnets + num_swans

def total_birds := total_ducks + total_geese + total_swans

-- Calculate the number of different types of birds
def num_types_of_birds : ℕ := 3 -- ducks, geese, swans

-- The final Lean statement to be proven
theorem total_birds_times_types :
  total_birds * num_types_of_birds = 438 :=
  by sorry

end NUMINAMATH_GPT_total_birds_times_types_l1020_102049


namespace NUMINAMATH_GPT_cat_weight_problem_l1020_102043

variable (female_cat_weight male_cat_weight : ℕ)

theorem cat_weight_problem
  (h1 : male_cat_weight = 2 * female_cat_weight)
  (h2 : female_cat_weight + male_cat_weight = 6) :
  female_cat_weight = 2 :=
by
  sorry

end NUMINAMATH_GPT_cat_weight_problem_l1020_102043


namespace NUMINAMATH_GPT_remainder_when_divided_l1020_102048

theorem remainder_when_divided (P K Q R K' Q' S' T : ℕ)
  (h1 : P = K * Q + R)
  (h2 : Q = K' * Q' + S')
  (h3 : R * Q' = T) :
  P % (K * K') = K * S' + (T / Q') :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_l1020_102048


namespace NUMINAMATH_GPT_office_light_ratio_l1020_102028

theorem office_light_ratio (bedroom_light: ℕ) (living_room_factor: ℕ) (total_energy: ℕ) 
  (time: ℕ) (ratio: ℕ) (office_light: ℕ) :
  bedroom_light = 6 →
  living_room_factor = 4 →
  total_energy = 96 →
  time = 2 →
  ratio = 3 →
  total_energy = (bedroom_light * time) + (office_light * time) + ((bedroom_light * living_room_factor) * time) →
  (office_light / bedroom_light) = ratio :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4] at h6
  -- The actual solution steps would go here
  sorry

end NUMINAMATH_GPT_office_light_ratio_l1020_102028


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1020_102034

theorem arithmetic_sequence_properties (a : ℕ → ℤ) (T : ℕ → ℤ) (h1 : ∀ n, a (n + 1) - a n = a 1 - a 0) (h2 : a 4 = a 2 + 4) (h3 : a 3 = 6) :
  (∀ n, a n = 2 * n) ∧ (∀ n, T n = (4 / 3 * (4^n - 1))) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1020_102034


namespace NUMINAMATH_GPT_alpha_epsilon_time_difference_l1020_102083

def B := 100
def M := 120
def A := B - 10

theorem alpha_epsilon_time_difference : M - A = 30 := by
  sorry

end NUMINAMATH_GPT_alpha_epsilon_time_difference_l1020_102083


namespace NUMINAMATH_GPT_yellow_marbles_in_C_l1020_102079

theorem yellow_marbles_in_C 
  (Y : ℕ)
  (conditionA : 4 - 2 ≠ 6)
  (conditionB : 6 - 1 ≠ 6)
  (conditionC1 : 3 > Y → 3 - Y = 6)
  (conditionC2 : Y > 3 → Y - 3 = 6) :
  Y = 9 :=
by
  sorry

end NUMINAMATH_GPT_yellow_marbles_in_C_l1020_102079
