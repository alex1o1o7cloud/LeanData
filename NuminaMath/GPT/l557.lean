import Mathlib

namespace NUMINAMATH_GPT_pencil_case_cost_l557_55725

-- Defining given conditions
def initial_amount : ℕ := 10
def toy_truck_cost : ℕ := 3
def remaining_amount : ℕ := 5
def total_spent : ℕ := initial_amount - remaining_amount

-- Proof statement
theorem pencil_case_cost : total_spent - toy_truck_cost = 2 :=
by
  sorry

end NUMINAMATH_GPT_pencil_case_cost_l557_55725


namespace NUMINAMATH_GPT_lychees_remaining_l557_55702
-- Definitions of the given conditions
def initial_lychees : ℕ := 500
def sold_lychees : ℕ := initial_lychees / 2
def home_lychees : ℕ := initial_lychees - sold_lychees
def eaten_lychees : ℕ := (3 * home_lychees) / 5

-- Statement to prove
theorem lychees_remaining : home_lychees - eaten_lychees = 100 := by
  sorry

end NUMINAMATH_GPT_lychees_remaining_l557_55702


namespace NUMINAMATH_GPT_ratio_of_boys_to_total_students_l557_55762

theorem ratio_of_boys_to_total_students
  (p : ℝ)
  (h : p = (3/4) * (1 - p)) :
  p = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_total_students_l557_55762


namespace NUMINAMATH_GPT_min_y_value_l557_55770

theorem min_y_value (x : ℝ) : 
  (∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 ∧ (y = 12 ↔ x = -1)) :=
sorry

end NUMINAMATH_GPT_min_y_value_l557_55770


namespace NUMINAMATH_GPT_mean_of_two_fractions_l557_55753

theorem mean_of_two_fractions :
  ( (2 : ℚ) / 3 + (4 : ℚ) / 9 ) / 2 = 5 / 9 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_two_fractions_l557_55753


namespace NUMINAMATH_GPT_find_f_minus_1_l557_55727

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f_at_2 : f 2 = 4

theorem find_f_minus_1 : f (-1) = -2 := 
by 
  sorry

end NUMINAMATH_GPT_find_f_minus_1_l557_55727


namespace NUMINAMATH_GPT_smallest_digit_divisibility_l557_55721

theorem smallest_digit_divisibility : 
  ∃ d : ℕ, (d < 10) ∧ (∃ k1 k2 : ℤ, 5 + 2 + 8 + d + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d + 7 + 4 = 3 * k2) ∧ (∀ d' : ℕ, (d' < 10) ∧ 
  (∃ k1 k2 : ℤ, 5 + 2 + 8 + d' + 7 + 4 = 9 * k1 ∧ 5 + 2 + 8 + d' + 7 + 4 = 3 * k2) → d ≤ d') :=
by
  sorry

end NUMINAMATH_GPT_smallest_digit_divisibility_l557_55721


namespace NUMINAMATH_GPT_expand_and_simplify_l557_55742

theorem expand_and_simplify (x y : ℝ) : 
  (x + 6) * (x + 8 + y) = x^2 + 14 * x + x * y + 48 + 6 * y :=
by sorry

end NUMINAMATH_GPT_expand_and_simplify_l557_55742


namespace NUMINAMATH_GPT_first_discount_percentage_l557_55728

theorem first_discount_percentage (d : ℝ) (h : d > 0) :
  (∃ x : ℝ, (0 < x) ∧ (x < 100) ∧ 0.6 * d = (d * (1 - x / 100)) * 0.8) → x = 25 :=
by
  sorry

end NUMINAMATH_GPT_first_discount_percentage_l557_55728


namespace NUMINAMATH_GPT_Xiaobing_jumps_189_ropes_per_minute_l557_55717

-- Define conditions and variables
variable (x : ℕ) -- The number of ropes Xiaohan jumps per minute

-- Conditions:
-- 1. Xiaobing jumps x + 21 ropes per minute
-- 2. Time taken for Xiaobing to jump 135 ropes is the same as the time taken for Xiaohan to jump 120 ropes

theorem Xiaobing_jumps_189_ropes_per_minute (h : 135 * x = 120 * (x + 21)) :
    x + 21 = 189 :=
by
  sorry -- Proof is not required as per instructions

end NUMINAMATH_GPT_Xiaobing_jumps_189_ropes_per_minute_l557_55717


namespace NUMINAMATH_GPT_inequality_inequality_l557_55719

theorem inequality_inequality
  (x y z : ℝ) :
  (x^2 / (x^2 + 2 * y * z) + y^2 / (y^2 + 2 * z * x) + z^2 / (z^2 + 2 * x * y) ≥ 1) :=
by sorry

end NUMINAMATH_GPT_inequality_inequality_l557_55719


namespace NUMINAMATH_GPT_conic_sections_are_parabolas_l557_55743

theorem conic_sections_are_parabolas (x y : ℝ) :
  y^6 - 9*x^6 = 3*y^3 - 1 → ∃ k : ℝ, (y^3 - 1 = k * 3 * x^3 ∨ y^3 = -k * 3 * x^3 + 1) := by
  sorry

end NUMINAMATH_GPT_conic_sections_are_parabolas_l557_55743


namespace NUMINAMATH_GPT_value_of_a_l557_55773

def f (a x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem value_of_a (a : ℝ) (h : f_prime a (-1) = 4) : a = 10 / 3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_value_of_a_l557_55773


namespace NUMINAMATH_GPT_find_m_of_line_with_slope_l557_55790

theorem find_m_of_line_with_slope (m : ℝ) (h_pos : m > 0)
(h_slope : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end NUMINAMATH_GPT_find_m_of_line_with_slope_l557_55790


namespace NUMINAMATH_GPT_divide_rope_into_parts_l557_55784

theorem divide_rope_into_parts:
  (∀ rope_length : ℝ, rope_length = 5 -> ∀ parts : ℕ, parts = 4 -> (∀ i : ℕ, i < parts -> ((rope_length / parts) = (5 / 4)))) :=
by sorry

end NUMINAMATH_GPT_divide_rope_into_parts_l557_55784


namespace NUMINAMATH_GPT_relationship_among_log_exp_powers_l557_55709

theorem relationship_among_log_exp_powers :
  let a := Real.log 0.3 / Real.log 2
  let b := Real.exp (0.3 * Real.log 2)
  let c := Real.exp (0.2 * Real.log 0.3)
  a < c ∧ c < b :=
by
  sorry

end NUMINAMATH_GPT_relationship_among_log_exp_powers_l557_55709


namespace NUMINAMATH_GPT_quotient_is_8_l557_55751

def dividend : ℕ := 64
def divisor : ℕ := 8
def quotient := dividend / divisor

theorem quotient_is_8 : quotient = 8 := 
by 
  show quotient = 8 
  sorry

end NUMINAMATH_GPT_quotient_is_8_l557_55751


namespace NUMINAMATH_GPT_razorback_tshirt_profit_l557_55740

theorem razorback_tshirt_profit :
  let profit_per_tshirt := 9
  let cost_per_tshirt := 4
  let num_tshirts_sold := 245
  let discount := 0.2
  let selling_price := profit_per_tshirt + cost_per_tshirt
  let discount_amount := discount * selling_price
  let discounted_price := selling_price - discount_amount
  let total_revenue := discounted_price * num_tshirts_sold
  let total_production_cost := cost_per_tshirt * num_tshirts_sold
  let total_profit := total_revenue - total_production_cost
  total_profit = 1568 :=
by
  sorry

end NUMINAMATH_GPT_razorback_tshirt_profit_l557_55740


namespace NUMINAMATH_GPT_largest_q_value_l557_55711

theorem largest_q_value : ∃ q, q >= 1 ∧ q^4 - q^3 - q - 1 ≤ 0 ∧ (∀ r, r >= 1 ∧ r^4 - r^3 - r - 1 ≤ 0 → r ≤ q) ∧ q = (Real.sqrt 5 + 1) / 2 := 
sorry

end NUMINAMATH_GPT_largest_q_value_l557_55711


namespace NUMINAMATH_GPT_books_sold_on_monday_l557_55703

def InitialStock : ℕ := 800
def BooksNotSold : ℕ := 600
def BooksSoldTuesday : ℕ := 10
def BooksSoldWednesday : ℕ := 20
def BooksSoldThursday : ℕ := 44
def BooksSoldFriday : ℕ := 66

def TotalBooksSold : ℕ := InitialStock - BooksNotSold
def BooksSoldAfterMonday : ℕ := BooksSoldTuesday + BooksSoldWednesday + BooksSoldThursday + BooksSoldFriday

theorem books_sold_on_monday : 
  TotalBooksSold - BooksSoldAfterMonday = 60 := by
  sorry

end NUMINAMATH_GPT_books_sold_on_monday_l557_55703


namespace NUMINAMATH_GPT_minimum_value_of_expression_l557_55757

theorem minimum_value_of_expression (x : ℝ) (hx : x > 0) :
  2 * x + 1 / x^6 ≥ 3 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l557_55757


namespace NUMINAMATH_GPT_micah_total_strawberries_l557_55767

theorem micah_total_strawberries (eaten saved total : ℕ) 
  (h1 : eaten = 6) 
  (h2 : saved = 18) 
  (h3 : total = eaten + saved) : 
  total = 24 := 
by
  sorry

end NUMINAMATH_GPT_micah_total_strawberries_l557_55767


namespace NUMINAMATH_GPT_hua_luogeng_optimal_selection_l557_55792

def concept_in_optimal_selection_method (concept : String) : Prop :=
  concept = "Golden ratio"

theorem hua_luogeng_optimal_selection (concept options : List String) 
  (h_options : options = ["Golden ratio", "Mean", "Mode", "Median"])
  (h_concept : "Golden ratio" ∈ options) :
  concept_in_optimal_selection_method "Golden ratio" :=
by
  -- Proof by assumption
  sorry

end NUMINAMATH_GPT_hua_luogeng_optimal_selection_l557_55792


namespace NUMINAMATH_GPT_nina_homework_total_l557_55755

def ruby_math_homework : ℕ := 6

def ruby_reading_homework : ℕ := 2

def nina_math_homework : ℕ := ruby_math_homework * 4 + ruby_math_homework

def nina_reading_homework : ℕ := ruby_reading_homework * 8 + ruby_reading_homework

def nina_total_homework : ℕ := nina_math_homework + nina_reading_homework

theorem nina_homework_total :
  nina_total_homework = 48 :=
by
  unfold nina_total_homework
  unfold nina_math_homework
  unfold nina_reading_homework
  unfold ruby_math_homework
  unfold ruby_reading_homework
  sorry

end NUMINAMATH_GPT_nina_homework_total_l557_55755


namespace NUMINAMATH_GPT_find_f2_l557_55736

variable (a b : ℝ)

def f (x : ℝ) : ℝ := a * x^3 + b * x + 1

theorem find_f2 (h : f a b (-2) = 3) : f a b 2 = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_f2_l557_55736


namespace NUMINAMATH_GPT_mole_can_sustain_l557_55716

noncomputable def mole_winter_sustainability : Prop :=
  ∃ (grain millet : ℕ), 
    grain = 8 ∧ 
    millet = 0 ∧ 
    ∀ (month : ℕ), 1 ≤ month ∧ month ≤ 3 → 
      ((grain ≥ 3 ∧ (grain - 3) + millet <= 12) ∨ 
      (grain ≥ 1 ∧ millet ≥ 3 ∧ (grain - 1) + (millet - 3) <= 12)) ∧
      ((∃ grain_exchanged millet_gained : ℕ, 
         grain_exchanged ≤ grain ∧
         millet_gained = 2 * grain_exchanged ∧
         grain - grain_exchanged + millet_gained <= 12 ∧
         grain = grain - grain_exchanged) → 
      (grain = 0 ∧ millet = 0))

theorem mole_can_sustain : mole_winter_sustainability := 
sorry 

end NUMINAMATH_GPT_mole_can_sustain_l557_55716


namespace NUMINAMATH_GPT_distance_between_points_l557_55760

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end NUMINAMATH_GPT_distance_between_points_l557_55760


namespace NUMINAMATH_GPT_lines_perpendicular_l557_55731

structure Vec3 :=
(x : ℝ) 
(y : ℝ) 
(z : ℝ)

def line1_dir (x : ℝ) : Vec3 := ⟨x, -1, 2⟩
def line2_dir : Vec3 := ⟨2, 1, 4⟩

def dot_product (v1 v2 : Vec3) : ℝ :=
v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

theorem lines_perpendicular (x : ℝ) :
  dot_product (line1_dir x) line2_dir = 0 ↔ x = -7 / 2 :=
by sorry

end NUMINAMATH_GPT_lines_perpendicular_l557_55731


namespace NUMINAMATH_GPT_top_card_is_king_l557_55713

noncomputable def num_cards := 52
noncomputable def num_kings := 4
noncomputable def probability_king := num_kings / num_cards

theorem top_card_is_king :
  probability_king = 1 / 13 := by
  sorry

end NUMINAMATH_GPT_top_card_is_king_l557_55713


namespace NUMINAMATH_GPT_divisor_problem_l557_55735

theorem divisor_problem (n : ℕ) (hn_pos : 0 < n) (h72 : Nat.totient n = 72) (h5n : Nat.totient (5 * n) = 96) : ∃ k : ℕ, (n = 5^k * m ∧ Nat.gcd m 5 = 1) ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_divisor_problem_l557_55735


namespace NUMINAMATH_GPT_expression_of_24ab_in_P_and_Q_l557_55776

theorem expression_of_24ab_in_P_and_Q (a b : ℕ) (P Q : ℝ)
  (hP : P = 2^a) (hQ : Q = 5^b) : 24^(a*b) = P^(3*b) * 3^(a*b) := 
  by
  sorry

end NUMINAMATH_GPT_expression_of_24ab_in_P_and_Q_l557_55776


namespace NUMINAMATH_GPT_option_D_forms_triangle_l557_55715

theorem option_D_forms_triangle (a b c : ℝ) (ha : a = 6) (hb : b = 8) (hc : c = 9) : 
  a + b > c ∧ a + c > b ∧ b + c > a := by
  sorry

end NUMINAMATH_GPT_option_D_forms_triangle_l557_55715


namespace NUMINAMATH_GPT_dino_remaining_balance_is_4650_l557_55726

def gigA_hours : Nat := 20
def gigA_rate : Nat := 10

def gigB_hours : Nat := 30
def gigB_rate : Nat := 20

def gigC_hours : Nat := 5
def gigC_rate : Nat := 40

def gigD_hours : Nat := 15
def gigD_rate : Nat := 25

def gigE_hours : Nat := 10
def gigE_rate : Nat := 30

def january_expense : Nat := 500
def february_expense : Nat := 550
def march_expense : Nat := 520
def april_expense : Nat := 480

theorem dino_remaining_balance_is_4650 :
  let gigA_earnings := gigA_hours * gigA_rate
  let gigB_earnings := gigB_hours * gigB_rate
  let gigC_earnings := gigC_hours * gigC_rate
  let gigD_earnings := gigD_hours * gigD_rate
  let gigE_earnings := gigE_hours * gigE_rate

  let total_monthly_earnings := gigA_earnings + gigB_earnings + gigC_earnings + gigD_earnings + gigE_earnings

  let total_expenses := january_expense + february_expense + march_expense + april_expense

  let total_earnings_four_months := total_monthly_earnings * 4

  total_earnings_four_months - total_expenses = 4650 :=
by {
  sorry
}

end NUMINAMATH_GPT_dino_remaining_balance_is_4650_l557_55726


namespace NUMINAMATH_GPT_find_number_of_elements_l557_55707

theorem find_number_of_elements (n S : ℕ) (h1: (S + 26) / n = 15) (h2: (S + 36) / n = 16) : n = 10 := by
  sorry

end NUMINAMATH_GPT_find_number_of_elements_l557_55707


namespace NUMINAMATH_GPT_smallest_sum_is_4_9_l557_55718

theorem smallest_sum_is_4_9 :
  min
    (min
      (min
        (min (1/3 + 1/4) (1/3 + 1/5))
        (min (1/3 + 1/6) (1/3 + 1/7)))
      (1/3 + 1/9)) = 4/9 :=
  by sorry

end NUMINAMATH_GPT_smallest_sum_is_4_9_l557_55718


namespace NUMINAMATH_GPT_find_a_l557_55763

-- Define the function f
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + a^2

-- Define the derivative of f
def f_prime (x a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_a (a b : ℝ) (h1 : f_prime 1 a b = 0) (h2 : f 1 a b = 10) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l557_55763


namespace NUMINAMATH_GPT_find_x_value_l557_55780

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end NUMINAMATH_GPT_find_x_value_l557_55780


namespace NUMINAMATH_GPT_factor_expression_l557_55793

theorem factor_expression (x : ℝ) : 
  ((4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8)) = 2 * x^2 * (5 * x + 31) := 
by sorry

end NUMINAMATH_GPT_factor_expression_l557_55793


namespace NUMINAMATH_GPT_picking_ball_is_random_event_l557_55791

-- Definitions based on problem conditions
def total_balls := 201
def black_balls := 200
def white_balls := 1

-- The goal to prove
theorem picking_ball_is_random_event : 
  (total_balls = black_balls + white_balls) ∧ 
  (black_balls > 0) ∧ 
  (white_balls > 0) → 
  random_event :=
by sorry

end NUMINAMATH_GPT_picking_ball_is_random_event_l557_55791


namespace NUMINAMATH_GPT_simplify_eval_expression_l557_55786

theorem simplify_eval_expression (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  3 * (2 * x^2 + x * y + 1 / 3) - (3 * x^2 + 4 * x * y - y^2) = 11 :=
by
  rw [hx, hy]
  sorry

end NUMINAMATH_GPT_simplify_eval_expression_l557_55786


namespace NUMINAMATH_GPT_tan_a6_of_arithmetic_sequence_l557_55783

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem tan_a6_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (H1 : arithmetic_sequence a)
  (H2 : sum_of_first_n_terms a S)
  (H3 : S 11 = 22 * Real.pi / 3) : 
  Real.tan (a 6) = -Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_tan_a6_of_arithmetic_sequence_l557_55783


namespace NUMINAMATH_GPT_translated_circle_eq_l557_55788

theorem translated_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 16) →
  (x + 5) ^ 2 + (y + 3) ^ 2 = 16 :=
by
  sorry

end NUMINAMATH_GPT_translated_circle_eq_l557_55788


namespace NUMINAMATH_GPT_kirill_height_l557_55754

theorem kirill_height (K B : ℕ) (h1 : K = B - 14) (h2 : K + B = 112) : K = 49 :=
by
  sorry

end NUMINAMATH_GPT_kirill_height_l557_55754


namespace NUMINAMATH_GPT_difference_of_squares_example_product_calculation_factorization_by_completing_square_l557_55798

/-
  Theorem: The transformation in the step \(195 \times 205 = 200^2 - 5^2\) uses the difference of squares formula.
-/

theorem difference_of_squares_example : 
  (195 * 205 = (200 - 5) * (200 + 5)) ∧ ((200 - 5) * (200 + 5) = 200^2 - 5^2) :=
  sorry

/-
  Theorem: Calculate \(9 \times 11 \times 101 \times 10001\) using a simple method.
-/

theorem product_calculation : 
  9 * 11 * 101 * 10001 = 99999999 :=
  sorry

/-
  Theorem: Factorize \(a^2 - 6a + 8\) using the completing the square method.
-/

theorem factorization_by_completing_square (a : ℝ) :
  a^2 - 6 * a + 8 = (a - 2) * (a - 4) :=
  sorry

end NUMINAMATH_GPT_difference_of_squares_example_product_calculation_factorization_by_completing_square_l557_55798


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l557_55758

def is_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

theorem point_in_fourth_quadrant :
  is_fourth_quadrant 2 (-3) :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l557_55758


namespace NUMINAMATH_GPT_quadratic_root_l557_55700

theorem quadratic_root (a b c : ℝ) (h : 9 * a - 3 * b + c = 0) : 
  a * (-3)^2 + b * (-3) + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_l557_55700


namespace NUMINAMATH_GPT_brownies_each_l557_55781

theorem brownies_each (num_columns : ℕ) (num_rows : ℕ) (total_people : ℕ) (total_brownies : ℕ) 
(h1 : num_columns = 6) (h2 : num_rows = 3) (h3 : total_people = 6) 
(h4 : total_brownies = num_columns * num_rows) : 
total_brownies / total_people = 3 := 
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_brownies_each_l557_55781


namespace NUMINAMATH_GPT_larger_cuboid_length_is_16_l557_55747

def volume (l w h : ℝ) : ℝ := l * w * h

def cuboid_length_proof : Prop :=
  ∀ (length_large : ℝ), 
  (volume 5 4 3 * 32 = volume length_large 10 12) → 
  length_large = 16

theorem larger_cuboid_length_is_16 : cuboid_length_proof :=
by
  intros length_large eq_volume
  sorry

end NUMINAMATH_GPT_larger_cuboid_length_is_16_l557_55747


namespace NUMINAMATH_GPT_max_intersections_l557_55774

theorem max_intersections (X Y : Type) [Fintype X] [Fintype Y]
  (hX : Fintype.card X = 20) (hY : Fintype.card Y = 10) : 
  ∃ (m : ℕ), m = 8550 := by
  sorry

end NUMINAMATH_GPT_max_intersections_l557_55774


namespace NUMINAMATH_GPT_contrapositive_inequality_l557_55759

theorem contrapositive_inequality {x y : ℝ} (h : x^2 ≤ y^2) : x ≤ y :=
  sorry

end NUMINAMATH_GPT_contrapositive_inequality_l557_55759


namespace NUMINAMATH_GPT_find_k_l557_55787

noncomputable section

variables {a b k : ℝ}

theorem find_k 
  (h1 : 4^a = k) 
  (h2 : 9^b = k)
  (h3 : 1 / a + 1 / b = 2) : 
  k = 6 :=
sorry

end NUMINAMATH_GPT_find_k_l557_55787


namespace NUMINAMATH_GPT_find_percentage_l557_55738

theorem find_percentage (P : ℝ) : 
  (∀ x : ℝ, x = 0.40 * 800 → x = P / 100 * 650 + 190) → P = 20 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_find_percentage_l557_55738


namespace NUMINAMATH_GPT_find_n_l557_55704

theorem find_n : ∀ (n x : ℝ), (3639 + n - x = 3054) → (x = 596.95) → (n = 11.95) :=
by
  intros n x h1 h2
  sorry

end NUMINAMATH_GPT_find_n_l557_55704


namespace NUMINAMATH_GPT_find_min_value_l557_55730

noncomputable def problem (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧
  (27^x + y^4 - 3^x - 1 = 0)

theorem find_min_value :
  ∃ x y : ℝ, problem x y ∧ 
  (∀ (x' y' : ℝ), problem x' y' → (x^3 + y^3) ≤ (x'^3 + y'^3)) ∧ (x^3 + y^3 = -1) := 
sorry

end NUMINAMATH_GPT_find_min_value_l557_55730


namespace NUMINAMATH_GPT_scalar_mult_l557_55785

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem scalar_mult (a : α) (h : a ≠ 0) : (-4) • (3 • a) = -12 • a :=
  sorry

end NUMINAMATH_GPT_scalar_mult_l557_55785


namespace NUMINAMATH_GPT_maria_remaining_towels_l557_55775

def total_towels_initial := 40 + 44
def towels_given_away := 65

theorem maria_remaining_towels : (total_towels_initial - towels_given_away) = 19 := by
  sorry

end NUMINAMATH_GPT_maria_remaining_towels_l557_55775


namespace NUMINAMATH_GPT_rectangle_area_l557_55779

noncomputable def width := 14
noncomputable def length := width + 6
noncomputable def perimeter := 2 * width + 2 * length
noncomputable def area := width * length

theorem rectangle_area (h1 : length = width + 6) (h2 : perimeter = 68) : area = 280 := 
by 
  have hw : width = 14 := by sorry 
  have hl : length = 20 := by sorry 
  have harea : area = 280 := by sorry
  exact harea

end NUMINAMATH_GPT_rectangle_area_l557_55779


namespace NUMINAMATH_GPT_range_of_m_l557_55768

def proposition_p (m : ℝ) : Prop :=
  ∀ x > 0, m^2 + 2 * m - 1 ≤ x + 1 / x

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (5 - m^2) ^ x > (5 - m^2) ^ (x - 1)

theorem range_of_m (m : ℝ) : (proposition_p m ∨ proposition_q m) ∧ ¬ (proposition_p m ∧ proposition_q m) ↔ (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l557_55768


namespace NUMINAMATH_GPT_quadratic_vertex_coordinates_l557_55771

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ :=
  -2 * (x + 1)^2 - 4

-- State the main theorem to be proved: The vertex of the quadratic function is at (-1, -4)
theorem quadratic_vertex_coordinates : 
  ∃ h k : ℝ, ∀ x : ℝ, quadratic x = -2 * (x + h)^2 + k ∧ h = -1 ∧ k = -4 := 
by
  -- proof required here
  sorry

end NUMINAMATH_GPT_quadratic_vertex_coordinates_l557_55771


namespace NUMINAMATH_GPT_sum_of_money_l557_55766

-- Conditions
def mass_record_coin_kg : ℝ := 100  -- 100 kg
def mass_one_pound_coin_g : ℝ := 10  -- 10 g

-- Conversion factor
def kg_to_g : ℝ := 1000

-- Question: Prove the sum of money in £1 coins that weighs the same as the record-breaking coin is £10,000.
theorem sum_of_money 
  (mass_record_coin_g := mass_record_coin_kg * kg_to_g)
  (number_of_coins := mass_record_coin_g / mass_one_pound_coin_g) 
  (sum_of_money := number_of_coins) : 
  sum_of_money = 10000 :=
  sorry

end NUMINAMATH_GPT_sum_of_money_l557_55766


namespace NUMINAMATH_GPT_yogurt_combinations_l557_55795

-- Define the conditions from a)
def num_flavors : ℕ := 5
def num_toppings : ℕ := 8
def num_sizes : ℕ := 3

-- Define the problem in a theorem statement
theorem yogurt_combinations : num_flavors * ((num_toppings * (num_toppings - 1)) / 2) * num_sizes = 420 :=
by
  -- sorry is used here to skip the proof
  sorry

end NUMINAMATH_GPT_yogurt_combinations_l557_55795


namespace NUMINAMATH_GPT_max_terms_in_arithmetic_seq_l557_55769

variable (a n : ℝ)

def arithmetic_seq_max_terms (a n : ℝ) : Prop :=
  let d := 4
  a^2 + (n - 1) * (a + d) + (n - 1) * n / 2 * d ≤ 100

theorem max_terms_in_arithmetic_seq (a n : ℝ) (h : arithmetic_seq_max_terms a n) : n ≤ 8 :=
sorry

end NUMINAMATH_GPT_max_terms_in_arithmetic_seq_l557_55769


namespace NUMINAMATH_GPT_son_age_l557_55741

theorem son_age {x : ℕ} {father son : ℕ} 
  (h1 : father = 4 * son)
  (h2 : (son - 10) + (father - 10) = 60)
  (h3 : son = x)
  : x = 16 := 
sorry

end NUMINAMATH_GPT_son_age_l557_55741


namespace NUMINAMATH_GPT_ratio_of_larger_to_smaller_l557_55789

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) (h3 : 0 < x) (h4 : 0 < y) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_larger_to_smaller_l557_55789


namespace NUMINAMATH_GPT_fraction_evaluation_l557_55765

theorem fraction_evaluation :
  (2 + 4 - 8 + 16 + 32 - 64 + 128 : ℚ) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_evaluation_l557_55765


namespace NUMINAMATH_GPT_saeyoung_yen_value_l557_55782

-- Define the exchange rate
def exchange_rate : ℝ := 17.25

-- Define Saeyoung's total yuan
def total_yuan : ℝ := 1000 + 10

-- Define the total yen based on the exchange rate
def total_yen : ℝ := total_yuan * exchange_rate

-- State the theorem
theorem saeyoung_yen_value : total_yen = 17422.5 :=
by
  sorry

end NUMINAMATH_GPT_saeyoung_yen_value_l557_55782


namespace NUMINAMATH_GPT_land_area_decreases_l557_55737

theorem land_area_decreases (a : ℕ) (h : a > 4) : (a * a) > ((a + 4) * (a - 4)) :=
by
  sorry

end NUMINAMATH_GPT_land_area_decreases_l557_55737


namespace NUMINAMATH_GPT_find_ratio_l557_55772

noncomputable def p (x : ℝ) : ℝ := 3 * x * (x - 5)
noncomputable def q (x : ℝ) : ℝ := (x + 2) * (x - 5)

theorem find_ratio : (p 3) / (q 3) = 9 / 5 := by
  sorry

end NUMINAMATH_GPT_find_ratio_l557_55772


namespace NUMINAMATH_GPT_necklaces_caught_l557_55761

theorem necklaces_caught
  (LatchNecklaces RhondaNecklaces BoudreauxNecklaces: ℕ)
  (h1 : LatchNecklaces = 3 * RhondaNecklaces - 4)
  (h2 : RhondaNecklaces = BoudreauxNecklaces / 2)
  (h3 : BoudreauxNecklaces = 12) :
  LatchNecklaces = 14 := by
  sorry

end NUMINAMATH_GPT_necklaces_caught_l557_55761


namespace NUMINAMATH_GPT_initial_quantity_of_milk_l557_55796

-- Define initial condition for the quantity of milk in container A
noncomputable def container_A : ℝ := 1184

-- Define the quantities of milk in containers B and C
def container_B (A : ℝ) : ℝ := 0.375 * A
def container_C (A : ℝ) : ℝ := 0.625 * A

-- Define the final equal quantities of milk after transfer
def equal_quantity (A : ℝ) : ℝ := container_B A + 148

-- The proof statement that must be true
theorem initial_quantity_of_milk :
  ∀ (A : ℝ), container_B A + 148 = equal_quantity A → A = container_A :=
by
  intros A h
  rw [equal_quantity] at h
  sorry

end NUMINAMATH_GPT_initial_quantity_of_milk_l557_55796


namespace NUMINAMATH_GPT_quadratic_inequality_l557_55701

theorem quadratic_inequality (t x₁ x₂ : ℝ) (α β : ℝ)
  (ht : (2 * x₁^2 - t * x₁ - 2 = 0) ∧ (2 * x₂^2 - t * x₂ - 2 = 0))
  (hx : α ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ β)
  (hαβ : α < β)
  (roots : α + β = t / 2 ∧ α * β = -1) :
  4*x₁*x₂ - t*(x₁ + x₂) - 4 < 0 := 
sorry

end NUMINAMATH_GPT_quadratic_inequality_l557_55701


namespace NUMINAMATH_GPT_sum_of_roots_l557_55797

theorem sum_of_roots {x1 x2 x3 k m : ℝ} (h1 : x1 ≠ x2) (h2 : x2 ≠ x3) (h3 : x1 ≠ x3)
  (h4 : 2 * x1^3 - k * x1 = m) (h5 : 2 * x2^3 - k * x2 = m) (h6 : 2 * x3^3 - k * x3 = m) :
  x1 + x2 + x3 = 0 :=
sorry

end NUMINAMATH_GPT_sum_of_roots_l557_55797


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l557_55748

def A : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def B : Set ℝ := { y | 2 ≤ y ∧ y ≤ 5 }

theorem intersection_of_A_and_B :
  A ∩ B = { x | 2 ≤ x ∧ x < 3 } :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l557_55748


namespace NUMINAMATH_GPT_marbles_solid_color_non_yellow_l557_55722

theorem marbles_solid_color_non_yellow (total_marble solid_colored solid_yellow : ℝ)
    (h1: solid_colored = 0.90 * total_marble)
    (h2: solid_yellow = 0.05 * total_marble) :
    (solid_colored - solid_yellow) / total_marble = 0.85 := by
  -- sorry is used to skip the proof
  sorry

end NUMINAMATH_GPT_marbles_solid_color_non_yellow_l557_55722


namespace NUMINAMATH_GPT_triangle_angle_properties_l557_55723

theorem triangle_angle_properties
  (a b : ℕ)
  (h₁ : a = 45)
  (h₂ : b = 70) :
  ∃ (c : ℕ), a + b + c = 180 ∧ c = 65 ∧ max (max a b) c = 70 := by
  sorry

end NUMINAMATH_GPT_triangle_angle_properties_l557_55723


namespace NUMINAMATH_GPT_value_of_f_g_5_l557_55745

def g (x : ℕ) : ℕ := 4 * x - 5
def f (x : ℕ) : ℕ := 6 * x + 11

theorem value_of_f_g_5 : f (g 5) = 101 := by
  sorry

end NUMINAMATH_GPT_value_of_f_g_5_l557_55745


namespace NUMINAMATH_GPT_find_a_if_even_function_l557_55799

-- Problem statement in Lean 4
theorem find_a_if_even_function (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 - 2 * (a + 1) * x + 1) 
  (hf_even : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end NUMINAMATH_GPT_find_a_if_even_function_l557_55799


namespace NUMINAMATH_GPT_pirates_coins_l557_55739

noncomputable def coins (x : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0     => x
  | k + 1 => (coins x k) - (coins x k * (k + 2) / 15)

theorem pirates_coins (x : ℕ) (H : x = 2^15 * 3^8 * 5^14) :
  ∃ n : ℕ, n = coins x 14 :=
sorry

end NUMINAMATH_GPT_pirates_coins_l557_55739


namespace NUMINAMATH_GPT_bob_favorite_number_is_correct_l557_55734

def bob_favorite_number : ℕ :=
  99

theorem bob_favorite_number_is_correct :
  50 < bob_favorite_number ∧
  bob_favorite_number < 100 ∧
  bob_favorite_number % 11 = 0 ∧
  bob_favorite_number % 2 ≠ 0 ∧
  (bob_favorite_number / 10 + bob_favorite_number % 10) % 3 = 0 :=
by
  sorry

end NUMINAMATH_GPT_bob_favorite_number_is_correct_l557_55734


namespace NUMINAMATH_GPT_correct_solution_to_equation_l557_55706

theorem correct_solution_to_equation :
  ∃ x m : ℚ, (m = 3 ∧ x = 14 / 23 → 7 * (2 - 2 * x) = 3 * (3 * x - m) + 63) ∧ (∃ x : ℚ, (∃ m : ℚ, m = 3) ∧ (7 * (2 - 2 * x) - (3 * (3 * x - 3) + 63) = 0)) →
  x = 2 := 
sorry

end NUMINAMATH_GPT_correct_solution_to_equation_l557_55706


namespace NUMINAMATH_GPT_problem_solution_l557_55744

noncomputable def corrected_angles 
  (x1_star x2_star x3_star : ℝ) 
  (σ : ℝ) 
  (h_sum : x1_star + x2_star + x3_star - 180.0 = 0)  
  (h_var : σ^2 = (0.1)^2) : ℝ × ℝ × ℝ :=
  let Δ := 2.0 / 3.0 * 0.667
  let Δx1 := Δ * (σ^2 / 2)
  let Δx2 := Δ * (σ^2 / 2)
  let Δx3 := Δ * (σ^2 / 2)
  let corrected_x1 := x1_star - Δx1
  let corrected_x2 := x2_star - Δx2
  let corrected_x3 := x3_star - Δx3
  (corrected_x1, corrected_x2, corrected_x3)

theorem problem_solution :
  corrected_angles 31 62 89 (0.1) sorry sorry = (30.0 + 40 / 60, 61.0 + 40 / 60, 88 + 20 / 60) := 
  sorry

end NUMINAMATH_GPT_problem_solution_l557_55744


namespace NUMINAMATH_GPT_mary_peter_lucy_chestnuts_l557_55705

noncomputable def mary_picked : ℕ := 12
noncomputable def peter_picked : ℕ := mary_picked / 2
noncomputable def lucy_picked : ℕ := peter_picked + 2
noncomputable def total_picked : ℕ := mary_picked + peter_picked + lucy_picked

theorem mary_peter_lucy_chestnuts : total_picked = 26 := by
  sorry

end NUMINAMATH_GPT_mary_peter_lucy_chestnuts_l557_55705


namespace NUMINAMATH_GPT_problem_correctness_l557_55729

theorem problem_correctness (a b x y m : ℝ) 
  (h1 : a + b = 0) 
  (h2 : x * y = 1) 
  (h3 : |m| = 2) : 
  (m = 2 ∨ m = -2) ∧ (m^2 + (a + b) / 2 + (- (x * y)) ^ 2023 = 3) := 
by
  sorry

end NUMINAMATH_GPT_problem_correctness_l557_55729


namespace NUMINAMATH_GPT_triangle_ABC_BC_length_l557_55732

theorem triangle_ABC_BC_length 
  (A B C D : ℝ)
  (AB AD DC AC BD BC : ℝ)
  (h1 : BD = 20)
  (h2 : AC = 69)
  (h3 : AB = 29)
  (h4 : BD^2 + DC^2 = BC^2)
  (h5 : AD^2 + BD^2 = AB^2)
  (h6 : AC = AD + DC) : 
  BC = 52 := 
by
  sorry

end NUMINAMATH_GPT_triangle_ABC_BC_length_l557_55732


namespace NUMINAMATH_GPT_fraction_sent_for_production_twice_l557_55714

variable {x : ℝ} (hx : x > 0)

theorem fraction_sent_for_production_twice :
  let initial_sulfur := (1.5 / 100 : ℝ)
  let first_sulfur_addition := (0.5 / 100 : ℝ)
  let second_sulfur_addition := (2 / 100 : ℝ) 
  (initial_sulfur - initial_sulfur * x + first_sulfur_addition * x -
    ((initial_sulfur - initial_sulfur * x + first_sulfur_addition * x) * x) + 
    second_sulfur_addition * x = initial_sulfur) → x = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_sent_for_production_twice_l557_55714


namespace NUMINAMATH_GPT_horses_put_by_c_l557_55720

theorem horses_put_by_c (a_horses a_months b_horses b_months c_months total_cost b_cost : ℕ) (x : ℕ) 
  (h1 : a_horses = 12) 
  (h2 : a_months = 8) 
  (h3 : b_horses = 16) 
  (h4 : b_months = 9) 
  (h5 : c_months = 6) 
  (h6 : total_cost = 870) 
  (h7 : b_cost = 360) 
  (h8 : 144 / (96 + 144 + 6 * x) = 360 / 870) : 
  x = 18 := 
by 
  sorry

end NUMINAMATH_GPT_horses_put_by_c_l557_55720


namespace NUMINAMATH_GPT_families_received_boxes_l557_55764

theorem families_received_boxes (F : ℕ) (box_decorations total_decorations : ℕ)
  (h_box_decorations : box_decorations = 10)
  (h_total_decorations : total_decorations = 120)
  (h_eq : box_decorations * (F + 1) = total_decorations) :
  F = 11 :=
by
  sorry

end NUMINAMATH_GPT_families_received_boxes_l557_55764


namespace NUMINAMATH_GPT_thirty_sixty_ninety_triangle_area_l557_55778

theorem thirty_sixty_ninety_triangle_area (hypotenuse : ℝ) (angle : ℝ) (area : ℝ)
  (h_hypotenuse : hypotenuse = 12)
  (h_angle : angle = 30)
  (h_area : area = 18 * Real.sqrt 3) :
  ∃ (base height : ℝ), 
    base = hypotenuse / 2 ∧ 
    height = (hypotenuse / 2) * Real.sqrt 3 ∧ 
    area = (1 / 2) * base * height :=
by {
  sorry
}

end NUMINAMATH_GPT_thirty_sixty_ninety_triangle_area_l557_55778


namespace NUMINAMATH_GPT_garden_width_l557_55752

theorem garden_width (w : ℝ) (h : ℝ) 
  (h1 : w * h ≥ 150)
  (h2 : h = w + 20)
  (h3 : 2 * (w + h) ≤ 70) :
  w = -10 + 5 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_GPT_garden_width_l557_55752


namespace NUMINAMATH_GPT_solve_for_difference_l557_55777

variable (a b : ℝ)

theorem solve_for_difference (h1 : a^3 - b^3 = 4) (h2 : a^2 + ab + b^2 + a - b = 4) : a - b = 2 :=
sorry

end NUMINAMATH_GPT_solve_for_difference_l557_55777


namespace NUMINAMATH_GPT_jerry_age_l557_55794

theorem jerry_age (M J : ℕ) (h1 : M = 20) (h2 : M = 2 * J - 8) : J = 14 := 
by
  sorry

end NUMINAMATH_GPT_jerry_age_l557_55794


namespace NUMINAMATH_GPT_abs_diff_inequality_l557_55708

theorem abs_diff_inequality (m : ℝ) : (∃ x : ℝ, |x + 2| - |x + 3| > m) ↔ m < -1 :=
sorry

end NUMINAMATH_GPT_abs_diff_inequality_l557_55708


namespace NUMINAMATH_GPT_students_per_configuration_l557_55712

theorem students_per_configuration (students_per_column : ℕ → ℕ) :
  students_per_column 1 = 15 ∧
  students_per_column 2 = 1 ∧
  students_per_column 3 = 1 ∧
  students_per_column 4 = 6 ∧
  ∀ i j, (i ≠ j ∧ i ≤ 12 ∧ j ≤ 12) → students_per_column i ≠ students_per_column j →
  (∃ n, 13 ≤ n ∧ ∀ k, k < 13 → students_per_column k * n = 60) :=
by
  sorry

end NUMINAMATH_GPT_students_per_configuration_l557_55712


namespace NUMINAMATH_GPT_dhoni_toys_average_cost_l557_55749

theorem dhoni_toys_average_cost (A : ℝ) (h1 : ∃ x1 x2 x3 x4 x5, (x1 + x2 + x3 + x4 + x5) / 5 = A)
  (h2 : 5 * A = 5 * A)
  (h3 : ∃ x6, x6 = 16)
  (h4 : (5 * A + 16) / 6 = 11) : A = 10 :=
by
  sorry

end NUMINAMATH_GPT_dhoni_toys_average_cost_l557_55749


namespace NUMINAMATH_GPT_speed_of_man_in_still_water_correct_l557_55733

def upstream_speed : ℝ := 25 -- Upstream speed in kmph
def downstream_speed : ℝ := 39 -- Downstream speed in kmph
def speed_in_still_water : ℝ := 32 -- The speed of the man in still water

theorem speed_of_man_in_still_water_correct :
  (upstream_speed + downstream_speed) / 2 = speed_in_still_water :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_in_still_water_correct_l557_55733


namespace NUMINAMATH_GPT_proportional_function_l557_55724

theorem proportional_function (k m : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = k * x) →
  f 2 = -4 →
  (∀ x, f x + m = -2 * x + m) →
  f 2 = -4 ∧ (f 1 + m = 1) →
  k = -2 ∧ m = 3 := 
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_proportional_function_l557_55724


namespace NUMINAMATH_GPT_students_like_basketball_or_cricket_or_both_l557_55746

theorem students_like_basketball_or_cricket_or_both :
  let basketball_lovers := 9
  let cricket_lovers := 8
  let both_lovers := 6
  basketball_lovers + cricket_lovers - both_lovers = 11 :=
by
  sorry

end NUMINAMATH_GPT_students_like_basketball_or_cricket_or_both_l557_55746


namespace NUMINAMATH_GPT_line_common_chord_eq_l557_55750

theorem line_common_chord_eq (a b : ℝ) :
  (∀ (x1 x2 y1 y2 : ℝ), x1^2 + y1^2 = 1 → (x2 - a)^2 + (y2 - b)^2 = 1 → 
    2 * a * x2 + 2 * b * y2 - 3 = 0) :=
sorry

end NUMINAMATH_GPT_line_common_chord_eq_l557_55750


namespace NUMINAMATH_GPT_prove_length_square_qp_l557_55756

noncomputable def length_square_qp (r1 r2 d : ℝ) (x : ℝ) : Prop :=
  r1 = 10 ∧ r2 = 8 ∧ d = 15 ∧ (2*r1*x - (x^2 + r2^2 - d^2) = 0) → x^2 = 164

theorem prove_length_square_qp : length_square_qp 10 8 15 x :=
sorry

end NUMINAMATH_GPT_prove_length_square_qp_l557_55756


namespace NUMINAMATH_GPT_find_x_when_y_is_minus_21_l557_55710

variable (x y k : ℝ)

theorem find_x_when_y_is_minus_21
  (h1 : x * y = k)
  (h2 : x + y = 35)
  (h3 : y = 3 * x)
  (h4 : y = -21) :
  x = -10.9375 := by
  sorry

end NUMINAMATH_GPT_find_x_when_y_is_minus_21_l557_55710
