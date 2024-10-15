import Mathlib

namespace NUMINAMATH_GPT_race_order_l240_24052

theorem race_order (overtakes_G_S_L : (ℕ × ℕ × ℕ))
  (h1 : overtakes_G_S_L.1 = 10)
  (h2 : overtakes_G_S_L.2.1 = 4)
  (h3 : overtakes_G_S_L.2.2 = 6)
  (h4 : ¬(overtakes_G_S_L.2.1 > 0 ∧ overtakes_G_S_L.2.2 > 0))
  (h5 : ∀ i j k : ℕ, i ≠ j → j ≠ k → k ≠ i)
  : overtakes_G_S_L = (10, 4, 6) :=
sorry

end NUMINAMATH_GPT_race_order_l240_24052


namespace NUMINAMATH_GPT_find_integer_n_l240_24021

theorem find_integer_n (n : ℤ) : (⌊(n^2 / 9 : ℝ)⌋ - ⌊(n / 3 : ℝ)⌋ ^ 2 = 5) → n = 14 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_integer_n_l240_24021


namespace NUMINAMATH_GPT_find_a_and_b_find_set_A_l240_24043

noncomputable def f (x a b : ℝ) := 4 ^ x - a * 2 ^ x + b

theorem find_a_and_b (a b : ℝ)
  (h₁ : f 1 a b = -1)
  (h₂ : ∀ x, ∃ t > 0, f x a b = t ^ 2 - a * t + b) :
  a = 4 ∧ b = 3 :=
sorry

theorem find_set_A (a b : ℝ)
  (ha : a = 4) (hb : b = 3) :
  {x : ℝ | f x a b ≤ 35} = {x : ℝ | x ≤ 3} :=
sorry

end NUMINAMATH_GPT_find_a_and_b_find_set_A_l240_24043


namespace NUMINAMATH_GPT_remainder_of_division_l240_24034

theorem remainder_of_division (L S R : ℕ) (h1 : L - S = 1365) (h2 : L = 1637) (h3 : L = 6 * S + R) : R = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_division_l240_24034


namespace NUMINAMATH_GPT_sqrt_sum_equality_l240_24098

open Real

theorem sqrt_sum_equality :
  (sqrt (18 - 8 * sqrt 2) + sqrt (18 + 8 * sqrt 2) = 8) :=
sorry

end NUMINAMATH_GPT_sqrt_sum_equality_l240_24098


namespace NUMINAMATH_GPT_total_bricks_in_wall_l240_24063

theorem total_bricks_in_wall :
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  (rows.sum = 80) := 
by
  let bottom_row_bricks := 18
  let rows := [bottom_row_bricks, bottom_row_bricks - 1, bottom_row_bricks - 2, bottom_row_bricks - 3, bottom_row_bricks - 4]
  sorry

end NUMINAMATH_GPT_total_bricks_in_wall_l240_24063


namespace NUMINAMATH_GPT_describe_graph_of_equation_l240_24025

theorem describe_graph_of_equation :
  (∀ x y : ℝ, (x + y)^3 = x^3 + y^3 → (x = 0 ∨ y = 0 ∨ y = -x)) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_describe_graph_of_equation_l240_24025


namespace NUMINAMATH_GPT_prism_diagonals_not_valid_l240_24018

theorem prism_diagonals_not_valid
  (a b c : ℕ)
  (h3 : a^2 + b^2 = 3^2 ∨ b^2 + c^2 = 3^2 ∨ a^2 + c^2 = 3^2)
  (h4 : a^2 + b^2 = 4^2 ∨ b^2 + c^2 = 4^2 ∨ a^2 + c^2 = 4^2)
  (h6 : a^2 + b^2 = 6^2 ∨ b^2 + c^2 = 6^2 ∨ a^2 + c^2 = 6^2) :
  False := 
sorry

end NUMINAMATH_GPT_prism_diagonals_not_valid_l240_24018


namespace NUMINAMATH_GPT_trapezoid_height_proof_l240_24057

-- Given lengths of the diagonals and the midline of the trapezoid
def diagonal1Length : ℝ := 6
def diagonal2Length : ℝ := 8
def midlineLength : ℝ := 5

-- Target to prove: Height of the trapezoid
def trapezoidHeight : ℝ := 4.8

theorem trapezoid_height_proof :
  ∀ (d1 d2 m : ℝ), d1 = diagonal1Length → d2 = diagonal2Length → m = midlineLength → trapezoidHeight = 4.8 :=
by intros d1 d2 m hd1 hd2 hm; sorry

end NUMINAMATH_GPT_trapezoid_height_proof_l240_24057


namespace NUMINAMATH_GPT_fencing_required_l240_24024

theorem fencing_required (L W : ℕ) (A : ℕ) (hL : L = 20) (hA : A = 680) (hArea : A = L * W) : 2 * W + L = 88 :=
by
  sorry

end NUMINAMATH_GPT_fencing_required_l240_24024


namespace NUMINAMATH_GPT_cone_lateral_area_l240_24031

theorem cone_lateral_area (r l : ℝ) (h_r : r = 3) (h_l : l = 5) : 
  (1 / 2) * (2 * Real.pi * r) * l = 15 * Real.pi :=
by
  rw [h_r, h_l]
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l240_24031


namespace NUMINAMATH_GPT_record_expenditure_l240_24066

theorem record_expenditure (income_recording : ℤ) (expenditure_amount : ℤ) (h : income_recording = 20) : -expenditure_amount = -50 :=
by sorry

end NUMINAMATH_GPT_record_expenditure_l240_24066


namespace NUMINAMATH_GPT_cost_of_each_orange_l240_24091

theorem cost_of_each_orange (calories_per_orange : ℝ) (total_money : ℝ) (calories_needed : ℝ) (money_left : ℝ) :
  calories_per_orange = 80 → 
  total_money = 10 → 
  calories_needed = 400 → 
  money_left = 4 → 
  (total_money - money_left) / (calories_needed / calories_per_orange) = 1.2 := 
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_cost_of_each_orange_l240_24091


namespace NUMINAMATH_GPT_mandy_med_school_ratio_l240_24035

theorem mandy_med_school_ratio 
    (researched_schools : ℕ)
    (applied_ratio : ℚ)
    (accepted_schools : ℕ)
    (h1 : researched_schools = 42)
    (h2 : applied_ratio = 1 / 3)
    (h3 : accepted_schools = 7)
    : (accepted_schools : ℚ) / ((researched_schools : ℚ) * applied_ratio) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_mandy_med_school_ratio_l240_24035


namespace NUMINAMATH_GPT_time_to_fill_bucket_completely_l240_24090

-- Define the conditions given in the problem
def time_to_fill_two_thirds (time_filled: ℕ) : ℕ := 90

-- Define what we need to prove
theorem time_to_fill_bucket_completely (time_filled: ℕ) : 
  time_to_fill_two_thirds time_filled = 90 → time_filled = 135 :=
by
  sorry

end NUMINAMATH_GPT_time_to_fill_bucket_completely_l240_24090


namespace NUMINAMATH_GPT_m_le_three_l240_24081

-- Definitions
def setA (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def setB (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

-- Theorem statement
theorem m_le_three (m : ℝ) : (∀ x : ℝ, setB m x → setA x) → m ≤ 3 := by
  sorry

end NUMINAMATH_GPT_m_le_three_l240_24081


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l240_24077

theorem arithmetic_sequence_problem :
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sum_first_sequence - sum_second_sequence - sum_third_sequence = 188725 :=
by
  let sum_first_sequence := (100 / 2) * (2501 + 2600)
  let sum_second_sequence := (100 / 2) * (401 + 500)
  let sum_third_sequence := (50 / 2) * (401 + 450)
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l240_24077


namespace NUMINAMATH_GPT_probability_of_three_given_sum_seven_l240_24089

theorem probability_of_three_given_sum_seven : 
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7) 
    ∧ (dice1 = 3 ∨ dice2 = 3)) →
  (∃ (dice1 dice2 : ℕ), (1 ≤ dice1 ∧ dice1 ≤ 6 ∧ 1 ≤ dice2 ∧ dice2 ≤ 6) ∧ (dice1 + dice2 = 7)) →
  ∃ (p : ℚ), p = 1/3 :=
by 
  sorry

end NUMINAMATH_GPT_probability_of_three_given_sum_seven_l240_24089


namespace NUMINAMATH_GPT_travel_time_l240_24051

namespace NatashaSpeedProblem

def distance : ℝ := 60
def speed_limit : ℝ := 50
def speed_over_limit : ℝ := 10
def actual_speed : ℝ := speed_limit + speed_over_limit

theorem travel_time : (distance / actual_speed) = 1 := by
  sorry

end NatashaSpeedProblem

end NUMINAMATH_GPT_travel_time_l240_24051


namespace NUMINAMATH_GPT_correct_statement_l240_24073

theorem correct_statement :
  (Real.sqrt (9 / 16) = 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l240_24073


namespace NUMINAMATH_GPT_max_circles_in_annulus_l240_24053

theorem max_circles_in_annulus (r_inner r_outer : ℝ) (h1 : r_inner = 1) (h2 : r_outer = 9) :
  ∃ n : ℕ, n = 3 ∧ ∀ r : ℝ, r = (r_outer - r_inner) / 2 → r * 3 ≤ 360 :=
sorry

end NUMINAMATH_GPT_max_circles_in_annulus_l240_24053


namespace NUMINAMATH_GPT_positive_two_digit_integers_remainder_4_div_9_l240_24065

theorem positive_two_digit_integers_remainder_4_div_9 : ∃ (n : ℕ), 
  (10 ≤ 9 * n + 4) ∧ (9 * n + 4 < 100) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 10 ∧ ∀ m, 1 ≤ m ∧ m ≤ 10 → n = k) :=
by
  sorry

end NUMINAMATH_GPT_positive_two_digit_integers_remainder_4_div_9_l240_24065


namespace NUMINAMATH_GPT_wire_lengths_l240_24033

variables (total_length first second third fourth : ℝ)

def wire_conditions : Prop :=
  total_length = 72 ∧
  first = second + 3 ∧
  third = 2 * second - 2 ∧
  fourth = 0.5 * (first + second + third) ∧
  second + first + third + fourth = total_length

theorem wire_lengths 
  (h : wire_conditions total_length first second third fourth) :
  second = 11.75 ∧ first = 14.75 ∧ third = 21.5 ∧ fourth = 24 :=
sorry

end NUMINAMATH_GPT_wire_lengths_l240_24033


namespace NUMINAMATH_GPT_quadratic_equation_roots_l240_24099

theorem quadratic_equation_roots (m n : ℝ) 
  (h_sum : m + n = -3) 
  (h_prod : m * n = 1) 
  (h_equation : m^2 + 3 * m + 1 = 0) :
  (3 * m + 1) / (m^3 * n) = -1 := 
by sorry

end NUMINAMATH_GPT_quadratic_equation_roots_l240_24099


namespace NUMINAMATH_GPT_max_value_of_expr_l240_24045

theorem max_value_of_expr (A M C : ℕ) (h : A + M + C = 12) : 
  A * M * C + A * M + M * C + C * A ≤ 112 :=
sorry

end NUMINAMATH_GPT_max_value_of_expr_l240_24045


namespace NUMINAMATH_GPT_first_digit_base_4_of_853_l240_24001

theorem first_digit_base_4_of_853 : 
  ∃ (d : ℕ), d = 3 ∧ (d * 256 ≤ 853 ∧ 853 < (d + 1) * 256) :=
by
  sorry

end NUMINAMATH_GPT_first_digit_base_4_of_853_l240_24001


namespace NUMINAMATH_GPT_curve_crosses_itself_at_point_l240_24006

theorem curve_crosses_itself_at_point :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ t₁^2 - 4 = t₂^2 - 4 ∧ t₁^3 - 6 * t₁ + 4 = t₂^3 - 6 * t₂ + 4 ∧ t₁^2 - 4 = 2 ∧ t₁^3 - 6 * t₁ + 4 = 4 :=
by 
  sorry

end NUMINAMATH_GPT_curve_crosses_itself_at_point_l240_24006


namespace NUMINAMATH_GPT_john_new_earnings_l240_24055

theorem john_new_earnings (original_earnings raise_percentage: ℝ)
  (h1 : original_earnings = 60)
  (h2 : raise_percentage = 40) :
  original_earnings * (1 + raise_percentage / 100) = 84 := 
by
  sorry

end NUMINAMATH_GPT_john_new_earnings_l240_24055


namespace NUMINAMATH_GPT_angle_between_east_and_south_is_90_degrees_l240_24016

-- Define the main theorem statement
theorem angle_between_east_and_south_is_90_degrees :
  ∀ (circle : Type) (num_rays : ℕ) (direction : ℕ → ℕ) (north east south : ℕ),
  num_rays = 12 →
  (∀ i, i < num_rays → direction i = (i * 360 / num_rays) % 360) →
  direction north = 0 →
  direction east = 90 →
  direction south = 180 →
  (min ((direction south - direction east) % 360) (360 - (direction south - direction east) % 360)) = 90 :=
by
  intros
  -- Skipped the proof
  sorry

end NUMINAMATH_GPT_angle_between_east_and_south_is_90_degrees_l240_24016


namespace NUMINAMATH_GPT_min_value_expression_l240_24000

theorem min_value_expression (x y : ℝ) (h : y^2 - 2*x + 4 = 0) : 
  ∃ z : ℝ, z = x^2 + y^2 + 2*x ∧ z = -8 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l240_24000


namespace NUMINAMATH_GPT_morse_code_count_l240_24029

noncomputable def morse_code_sequences : Nat :=
  let case_1 := 2            -- 1 dot or dash
  let case_2 := 2 * 2        -- 2 dots or dashes
  let case_3 := 2 * 2 * 2    -- 3 dots or dashes
  let case_4 := 2 * 2 * 2 * 2-- 4 dots or dashes
  let case_5 := 2 * 2 * 2 * 2 * 2 -- 5 dots or dashes
  case_1 + case_2 + case_3 + case_4 + case_5

theorem morse_code_count : morse_code_sequences = 62 := by
  sorry

end NUMINAMATH_GPT_morse_code_count_l240_24029


namespace NUMINAMATH_GPT_range_of_g_l240_24084

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then
  ⌈1 / ((x + 3)^2)⌉
else
  ⌊1 / ((x + 3)^2)⌋

theorem range_of_g :
  ∀ y : ℤ, (∃ x : ℝ, g x = y) ↔ (∃ n : ℕ, y = n + 1) :=
by sorry

end NUMINAMATH_GPT_range_of_g_l240_24084


namespace NUMINAMATH_GPT_division_value_l240_24059

theorem division_value (x : ℚ) (h : (5 / 2) / x = 5 / 14) : x = 7 :=
sorry

end NUMINAMATH_GPT_division_value_l240_24059


namespace NUMINAMATH_GPT_polynomial_not_separable_l240_24023

theorem polynomial_not_separable (f g : Polynomial ℂ) :
  (∀ x y : ℂ, f.eval x * g.eval y = x^200 * y^200 + 1) → False :=
sorry

end NUMINAMATH_GPT_polynomial_not_separable_l240_24023


namespace NUMINAMATH_GPT_no_divide_five_to_n_minus_three_to_n_l240_24013

theorem no_divide_five_to_n_minus_three_to_n (n : ℕ) (h : n ≥ 1) : ¬ (2 ^ n + 65 ∣ 5 ^ n - 3 ^ n) :=
by
  sorry

end NUMINAMATH_GPT_no_divide_five_to_n_minus_three_to_n_l240_24013


namespace NUMINAMATH_GPT_find_other_endpoint_l240_24015

theorem find_other_endpoint (x₁ y₁ x y x_mid y_mid : ℝ) 
  (h1 : x₁ = 5) (h2 : y₁ = 2) (h3 : x_mid = 3) (h4 : y_mid = 10) 
  (hx : (x₁ + x) / 2 = x_mid) (hy : (y₁ + y) / 2 = y_mid) : 
  x = 1 ∧ y = 18 := by
  sorry

end NUMINAMATH_GPT_find_other_endpoint_l240_24015


namespace NUMINAMATH_GPT_sum_reciprocal_eq_l240_24037

theorem sum_reciprocal_eq :
  ∃ (a b : ℕ), a + b = 45 ∧ Nat.lcm a b = 120 ∧ Nat.gcd a b = 5 ∧ 
  (1/a + 1/b = (3 : ℚ) / 40) := by
  sorry

end NUMINAMATH_GPT_sum_reciprocal_eq_l240_24037


namespace NUMINAMATH_GPT_calculate_product_l240_24038

theorem calculate_product :
  6^5 * 3^5 = 1889568 := by
  sorry

end NUMINAMATH_GPT_calculate_product_l240_24038


namespace NUMINAMATH_GPT_notecard_calculation_l240_24068

theorem notecard_calculation (N E : ℕ) (h₁ : N - E = 80) (h₂ : N = 3 * E) : N = 120 :=
sorry

end NUMINAMATH_GPT_notecard_calculation_l240_24068


namespace NUMINAMATH_GPT_unique_positive_solution_eq_15_l240_24064

theorem unique_positive_solution_eq_15 
  (x : ℝ) 
  (h1 : x > 0) 
  (h2 : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_solution_eq_15_l240_24064


namespace NUMINAMATH_GPT_hexagon_midpoints_equilateral_l240_24074

noncomputable def inscribed_hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : Prop :=
  ∀ (M N P : ℝ), 
    true

theorem hexagon_midpoints_equilateral (r : ℝ) (h : ℝ) 
  (hex : ∀ (A B C D E F : ℝ) (O : ℝ), 
    true) : 
  inscribed_hexagon_midpoints_equilateral r h hex :=
sorry

end NUMINAMATH_GPT_hexagon_midpoints_equilateral_l240_24074


namespace NUMINAMATH_GPT_record_withdrawal_example_l240_24041

-- Definitions based on conditions
def ten_thousand_dollars := 10000
def record_deposit (amount : ℕ) : ℤ := amount / ten_thousand_dollars
def record_withdrawal (amount : ℕ) : ℤ := -(amount / ten_thousand_dollars)

-- Lean 4 statement to prove the problem
theorem record_withdrawal_example :
  (record_deposit 30000 = 3) → (record_withdrawal 20000 = -2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_record_withdrawal_example_l240_24041


namespace NUMINAMATH_GPT_ruby_height_l240_24005

/-- Height calculations based on given conditions -/
theorem ruby_height (Janet_height : ℕ) (Charlene_height : ℕ) (Pablo_height : ℕ) (Ruby_height : ℕ) 
  (h₁ : Janet_height = 62) 
  (h₂ : Charlene_height = 2 * Janet_height)
  (h₃ : Pablo_height = Charlene_height + 70)
  (h₄ : Ruby_height = Pablo_height - 2) : Ruby_height = 192 := 
by
  sorry

end NUMINAMATH_GPT_ruby_height_l240_24005


namespace NUMINAMATH_GPT_cubical_tank_fraction_filled_l240_24047

theorem cubical_tank_fraction_filled (a : ℝ) (h1 : ∀ a:ℝ, (a * a * 1 = 16) )
  : (1 / 4) = (16 / (a^3)) :=
by
  sorry

end NUMINAMATH_GPT_cubical_tank_fraction_filled_l240_24047


namespace NUMINAMATH_GPT_other_cube_side_length_l240_24085

theorem other_cube_side_length (s_1 s_2 : ℝ) (h1 : s_1 = 1) (h2 : 6 * s_2^2 / 6 = 36) : s_2 = 6 :=
by
  sorry

end NUMINAMATH_GPT_other_cube_side_length_l240_24085


namespace NUMINAMATH_GPT_average_first_19_natural_numbers_l240_24058

theorem average_first_19_natural_numbers : 
  (1 + 19) / 2 = 10 := 
by 
  sorry

end NUMINAMATH_GPT_average_first_19_natural_numbers_l240_24058


namespace NUMINAMATH_GPT_f_0_plus_f_1_l240_24075

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom f_neg1 : f (-1) = 2

theorem f_0_plus_f_1 : f 0 + f 1 = -2 :=
by
  sorry

end NUMINAMATH_GPT_f_0_plus_f_1_l240_24075


namespace NUMINAMATH_GPT_complex_div_eq_half_add_half_i_l240_24017

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem to be proven
theorem complex_div_eq_half_add_half_i :
  (i / (1 + i)) = (1 / 2 + (1 / 2) * i) :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_complex_div_eq_half_add_half_i_l240_24017


namespace NUMINAMATH_GPT_total_days_of_work_l240_24086

theorem total_days_of_work (r1 r2 r3 r4 : ℝ) (h1 : r1 = 1 / 12) (h2 : r2 = 1 / 8) (h3 : r3 = 1 / 24) (h4 : r4 = 1 / 16) : 
  (1 / (r1 + r2 + r3 + r4) = 3.2) :=
by 
  sorry

end NUMINAMATH_GPT_total_days_of_work_l240_24086


namespace NUMINAMATH_GPT_boys_to_girls_ratio_l240_24027

theorem boys_to_girls_ratio (S G B : ℕ) (h1 : 1 / 2 * G = 1 / 3 * S) (h2 : S = B + G) : B / G = 1 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_boys_to_girls_ratio_l240_24027


namespace NUMINAMATH_GPT_quadratic_real_roots_l240_24044

theorem quadratic_real_roots (m : ℝ) : 
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  (discriminant ≥ 0) ↔ (m ≤ 4 ∧ m ≠ 3) :=
by
  let a := m - 3
  let b := -2
  let c := 1
  let discriminant := b^2 - 4 * a * c
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_l240_24044


namespace NUMINAMATH_GPT_each_child_consumes_3_bottles_per_day_l240_24012

noncomputable def bottles_per_child_per_day : ℕ :=
  let first_group := 14
  let second_group := 16
  let third_group := 12
  let fourth_group := (first_group + second_group + third_group) / 2
  let total_children := first_group + second_group + third_group + fourth_group
  let cases_of_water := 13
  let bottles_per_case := 24
  let initial_bottles := cases_of_water * bottles_per_case
  let additional_bottles := 255
  let total_bottles := initial_bottles + additional_bottles
  let bottles_per_child := total_bottles / total_children
  let days := 3
  bottles_per_child / days

theorem each_child_consumes_3_bottles_per_day :
  bottles_per_child_per_day = 3 :=
by
  sorry

end NUMINAMATH_GPT_each_child_consumes_3_bottles_per_day_l240_24012


namespace NUMINAMATH_GPT_ratio_rect_prism_l240_24097

namespace ProofProblem

variables (w l h : ℕ)
def rect_prism (w l h : ℕ) : Prop := w * l * h = 128

theorem ratio_rect_prism (h1 : rect_prism w l h) :
  (w : ℕ) ≠ 0 ∧ (l : ℕ) ≠ 0 ∧ (h : ℕ) ≠ 0 ∧ 
  (∃ k, w = k ∧ l = k ∧ h = 2 * k) :=
sorry

end ProofProblem

end NUMINAMATH_GPT_ratio_rect_prism_l240_24097


namespace NUMINAMATH_GPT_tan_sum_eq_l240_24095

theorem tan_sum_eq (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1/2 :=
by sorry

end NUMINAMATH_GPT_tan_sum_eq_l240_24095


namespace NUMINAMATH_GPT_missy_tv_watching_time_l240_24010

def reality_show_count : Nat := 5
def reality_show_duration : Nat := 28
def cartoon_duration : Nat := 10

theorem missy_tv_watching_time :
  reality_show_count * reality_show_duration + cartoon_duration = 150 := by
  sorry

end NUMINAMATH_GPT_missy_tv_watching_time_l240_24010


namespace NUMINAMATH_GPT_range_of_a_l240_24002

def proposition_p (a : ℝ) : Prop := a > 1
def proposition_q (a : ℝ) : Prop := 0 < a ∧ a < 4

theorem range_of_a
(a : ℝ)
(h1 : a > 0)
(h2 : ¬ proposition_p a)
(h3 : ¬ proposition_q a)
(h4 : proposition_p a ∨ proposition_q a) :
  (0 < a ∧ a ≤ 1) ∨ (4 ≤ a) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l240_24002


namespace NUMINAMATH_GPT_perimeter_of_region_l240_24060

-- Define the condition
def area_of_region := 512 -- square centimeters
def number_of_squares := 8

-- Define the presumed perimeter
def presumed_perimeter := 144 -- the correct answer

-- Mathematical statement that needs proof
theorem perimeter_of_region (area_of_region: ℕ) (number_of_squares: ℕ) (presumed_perimeter: ℕ) : 
   area_of_region = 512 ∧ number_of_squares = 8 → presumed_perimeter = 144 :=
by 
  sorry

end NUMINAMATH_GPT_perimeter_of_region_l240_24060


namespace NUMINAMATH_GPT_total_pencils_correct_l240_24046

def reeta_pencils : Nat := 20
def anika_pencils : Nat := 2 * reeta_pencils + 4
def total_pencils : Nat := reeta_pencils + anika_pencils

theorem total_pencils_correct : total_pencils = 64 :=
by
  sorry

end NUMINAMATH_GPT_total_pencils_correct_l240_24046


namespace NUMINAMATH_GPT_pencil_count_l240_24028

/-- 
If there are initially 115 pencils in the drawer, and Sara adds 100 more pencils, 
then the total number of pencils in the drawer is 215.
-/
theorem pencil_count (initial_pencils added_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : added_pencils = 100) : 
  initial_pencils + added_pencils = 215 := by
  sorry

end NUMINAMATH_GPT_pencil_count_l240_24028


namespace NUMINAMATH_GPT_mica_should_have_28_26_euros_l240_24088

namespace GroceryShopping

def pasta_cost : ℝ := 3 * 1.70
def ground_beef_cost : ℝ := 0.5 * 8.20
def pasta_sauce_base_cost : ℝ := 3 * 2.30
def pasta_sauce_discount : ℝ := pasta_sauce_base_cost * 0.10
def pasta_sauce_discounted_cost : ℝ := pasta_sauce_base_cost - pasta_sauce_discount
def quesadillas_cost : ℝ := 11.50

def total_cost_before_vat : ℝ :=
  pasta_cost + ground_beef_cost + pasta_sauce_discounted_cost + quesadillas_cost

def vat : ℝ := total_cost_before_vat * 0.05

def total_cost_including_vat : ℝ := total_cost_before_vat + vat

theorem mica_should_have_28_26_euros :
  total_cost_including_vat = 28.26 := by
  -- This is the statement without the proof. 
  sorry

end GroceryShopping

end NUMINAMATH_GPT_mica_should_have_28_26_euros_l240_24088


namespace NUMINAMATH_GPT_distinct_permutations_of_12233_l240_24003

def numFiveDigitIntegers : ℕ :=
  Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2)

theorem distinct_permutations_of_12233 : numFiveDigitIntegers = 30 := by
  sorry

end NUMINAMATH_GPT_distinct_permutations_of_12233_l240_24003


namespace NUMINAMATH_GPT_age_proof_l240_24071

noncomputable def father_age_current := 33
noncomputable def xiaolin_age_current := 3

def father_age (X : ℕ) := 11 * X
def future_father_age (F : ℕ) := F + 7
def future_xiaolin_age (X : ℕ) := X + 7

theorem age_proof (F X : ℕ) (h1 : F = father_age X) 
  (h2 : future_father_age F = 4 * future_xiaolin_age X) : 
  F = father_age_current ∧ X = xiaolin_age_current :=
by 
  sorry

end NUMINAMATH_GPT_age_proof_l240_24071


namespace NUMINAMATH_GPT_bookmark_position_second_book_l240_24067

-- Definitions for the conditions
def pages_per_book := 250
def cover_thickness_ratio := 10
def total_books := 2
def distance_bookmarks_factor := 1 / 3

-- Derived constants
def cover_thickness := cover_thickness_ratio * pages_per_book
def total_pages := (pages_per_book * total_books) + (cover_thickness * total_books * 2)
def distance_between_bookmarks := total_pages * distance_bookmarks_factor
def midpoint_pages_within_book := (pages_per_book / 2) + cover_thickness

-- Definitions for bookmarks positions
def first_bookmark_position := midpoint_pages_within_book
def remaining_pages_after_first_bookmark := distance_between_bookmarks - midpoint_pages_within_book
def second_bookmark_position := remaining_pages_after_first_bookmark - cover_thickness

-- Theorem stating the goal
theorem bookmark_position_second_book :
  35 ≤ second_bookmark_position ∧ second_bookmark_position < 36 :=
sorry

end NUMINAMATH_GPT_bookmark_position_second_book_l240_24067


namespace NUMINAMATH_GPT_find_x_l240_24026

theorem find_x (x : ℝ) (h : (40 / 100) * x = (25 / 100) * 80) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l240_24026


namespace NUMINAMATH_GPT_days_in_month_find_days_in_month_l240_24072

noncomputable def computers_per_thirty_minutes : ℕ := 225 / 100 -- representing 2.25
def monthly_computers : ℕ := 3024
def hours_per_day : ℕ := 24

theorem days_in_month (computers_per_hour : ℕ) (daily_production : ℕ) : ℕ :=
  let computers_per_hour := (2 * computers_per_thirty_minutes)
  let daily_production := (computers_per_hour * hours_per_day)
  (monthly_computers / daily_production)

theorem find_days_in_month :
  days_in_month (2 * computers_per_thirty_minutes) ((2 * computers_per_thirty_minutes) * hours_per_day) = 28 :=
by
  sorry

end NUMINAMATH_GPT_days_in_month_find_days_in_month_l240_24072


namespace NUMINAMATH_GPT_base9_digit_divisible_by_13_l240_24087

theorem base9_digit_divisible_by_13 :
    ∃ (d : ℕ), (0 ≤ d ∧ d ≤ 8) ∧ (13 ∣ (2 * 9^4 + d * 9^3 + 6 * 9^2 + d * 9 + 4)) :=
by
  sorry

end NUMINAMATH_GPT_base9_digit_divisible_by_13_l240_24087


namespace NUMINAMATH_GPT_find_natural_numbers_l240_24076

theorem find_natural_numbers (n k : ℕ) (h : 2^n - 5^k = 7) : n = 5 ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_natural_numbers_l240_24076


namespace NUMINAMATH_GPT_S8_value_l240_24019

theorem S8_value (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 5 / 5 + S 11 / 11 = 12) (h2 : S 11 = S 8 + 1 / a 9 + 1 / a 10 + 1 / a 11) : S 8 = 48 :=
sorry

end NUMINAMATH_GPT_S8_value_l240_24019


namespace NUMINAMATH_GPT_area_units_ordered_correctly_l240_24004

def area_units :=
  ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"]

theorem area_units_ordered_correctly :
  area_units = ["square kilometers", "hectares", "square meters", "square decimeters", "square centimeters"] :=
by
  sorry

end NUMINAMATH_GPT_area_units_ordered_correctly_l240_24004


namespace NUMINAMATH_GPT_solve_first_system_solve_second_system_l240_24042

theorem solve_first_system :
  (exists x y : ℝ, 3 * x + 2 * y = 6 ∧ y = x - 2) ->
  (∃ (x y : ℝ), x = 2 ∧ y = 0) := by
  sorry

theorem solve_second_system :
  (exists m n : ℝ, m + 2 * n = 7 ∧ -3 * m + 5 * n = 1) ->
  (∃ (m n : ℝ), m = 3 ∧ n = 2) := by
  sorry

end NUMINAMATH_GPT_solve_first_system_solve_second_system_l240_24042


namespace NUMINAMATH_GPT_side_length_square_field_l240_24009

-- Definitions based on the conditions.
def time_taken := 56 -- in seconds
def speed := 9 * 1000 / 3600 -- in meters per second, converting 9 km/hr to m/s
def distance_covered := speed * time_taken -- calculating the distance covered in meters
def perimeter := 4 * 35 -- defining the perimeter given the side length is 35

-- Problem statement for proof: We need to prove that the calculated distance covered matches the perimeter.
theorem side_length_square_field : distance_covered = perimeter :=
by
  sorry

end NUMINAMATH_GPT_side_length_square_field_l240_24009


namespace NUMINAMATH_GPT_monotonically_increasing_intervals_inequality_solution_set_l240_24056

-- Given conditions for f
def f (a b c d : ℝ) (x : ℝ) : ℝ := a*x^3 + b*x^2 + c*x + d

-- Ⅰ) Prove the intervals of monotonic increase
theorem monotonically_increasing_intervals (a c : ℝ) (x : ℝ) (h_f : ∀ x, f a 0 c 0 x = a*x^3 + c*x)
  (h_a : a = 1) (h_c : c = -3) :
  (∀ x < -1, f a 0 c 0 x < 0) ∧ (∀ x > 1, f a 0 c 0 x > 0) := 
sorry

-- Ⅱ) Prove the solution sets for the inequality given m
theorem inequality_solution_set (m x : ℝ) :
  (m = 0 → x > 0) ∧
  (m > 0 → (x > 4*m ∨ 0 < x ∧ x < m)) ∧
  (m < 0 → (x > 0 ∨ 4*m < x ∧ x < m)) :=
sorry

end NUMINAMATH_GPT_monotonically_increasing_intervals_inequality_solution_set_l240_24056


namespace NUMINAMATH_GPT_complement_of_M_in_U_l240_24030

-- Definition of the universal set U
def U : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }

-- Definition of the set M
def M : Set ℝ := { 1 }

-- The statement to prove
theorem complement_of_M_in_U : (U \ M) = {x | 1 < x ∧ x ≤ 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_in_U_l240_24030


namespace NUMINAMATH_GPT_no_four_points_with_equal_tangents_l240_24079

theorem no_four_points_with_equal_tangents :
  ∀ (A B C D : ℝ × ℝ),
    A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧
    A ≠ C ∧ B ≠ D →
    ¬ (∀ (P Q : ℝ × ℝ), (P = A ∧ Q = B) ∨ (P = C ∧ Q = D) →
      ∃ (M : ℝ × ℝ) (r : ℝ), M ≠ P ∧ M ≠ Q ∧
      (dist A M = dist C M ∧ dist B M = dist D M ∧
       dist P M > r ∧ dist Q M > r)) :=
by sorry

end NUMINAMATH_GPT_no_four_points_with_equal_tangents_l240_24079


namespace NUMINAMATH_GPT_jungkook_seokjin_books_l240_24096

/-- Given the number of books Jungkook and Seokjin originally had and the number of books they 
   bought, prove that Jungkook has 7 more books than Seokjin. -/
theorem jungkook_seokjin_books
  (jungkook_initial : ℕ)
  (seokjin_initial : ℕ)
  (jungkook_bought : ℕ)
  (seokjin_bought : ℕ)
  (h1 : jungkook_initial = 28)
  (h2 : seokjin_initial = 28)
  (h3 : jungkook_bought = 18)
  (h4 : seokjin_bought = 11) :
  (jungkook_initial + jungkook_bought) - (seokjin_initial + seokjin_bought) = 7 :=
by
  sorry

end NUMINAMATH_GPT_jungkook_seokjin_books_l240_24096


namespace NUMINAMATH_GPT_probability_of_drawing_white_ball_probability_with_additional_white_balls_l240_24022

noncomputable def total_balls := 6 + 9 + 3
noncomputable def initial_white_balls := 3

theorem probability_of_drawing_white_ball :
  (initial_white_balls : ℚ) / (total_balls : ℚ) = 1 / 6 :=
sorry

noncomputable def additional_white_balls_needed := 2

theorem probability_with_additional_white_balls :
  (initial_white_balls + additional_white_balls_needed : ℚ) / (total_balls + additional_white_balls_needed : ℚ) = 1 / 4 :=
sorry

end NUMINAMATH_GPT_probability_of_drawing_white_ball_probability_with_additional_white_balls_l240_24022


namespace NUMINAMATH_GPT_find_common_ratio_l240_24082

def first_term : ℚ := 4 / 7
def second_term : ℚ := 12 / 7

theorem find_common_ratio (r : ℚ) : second_term = first_term * r → r = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_common_ratio_l240_24082


namespace NUMINAMATH_GPT_max_m_value_l240_24050

theorem max_m_value (a b : ℝ) (m : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : ∀ a b, 0 < a → 0 < b → (m / (3 * a + b) - 3 / a - 1 / b ≤ 0)) :
  m ≤ 16 :=
sorry

end NUMINAMATH_GPT_max_m_value_l240_24050


namespace NUMINAMATH_GPT_find_P_l240_24020

-- Define the variables A, B, C and their type
variables (A B C P : ℤ)

-- The main theorem statement according to the given conditions and question
theorem find_P (h1 : A = C + 1) (h2 : A + B = C + P) : P = 1 + B :=
by
  sorry

end NUMINAMATH_GPT_find_P_l240_24020


namespace NUMINAMATH_GPT_total_number_of_components_l240_24080

-- Definitions based on the conditions in the problem
def number_of_B_components := 300
def number_of_C_components := 200
def sample_size := 45
def number_of_A_components_drawn := 20
def number_of_C_components_drawn := 10

-- The statement to be proved
theorem total_number_of_components :
  (number_of_A_components_drawn * (number_of_B_components + number_of_C_components) / sample_size) 
  + number_of_B_components 
  + number_of_C_components 
  = 900 := 
by 
  sorry

end NUMINAMATH_GPT_total_number_of_components_l240_24080


namespace NUMINAMATH_GPT_symmetric_points_on_parabola_l240_24040

theorem symmetric_points_on_parabola
  (x1 x2 : ℝ)
  (m : ℝ)
  (h1 : 2 * x1 * x1 = 2 * x2 * x2)
  (h2 : 2 * x1 * x1 = 2 * x2 * x2 + m)
  (h3 : x1 * x2 = -1 / 2)
  (h4 : x1 + x2 = -1 / 2)
  : m = 3 / 2 :=
sorry

end NUMINAMATH_GPT_symmetric_points_on_parabola_l240_24040


namespace NUMINAMATH_GPT_product_equals_32_l240_24008

theorem product_equals_32 :
  (1/4) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 = 32 :=
by
  sorry

end NUMINAMATH_GPT_product_equals_32_l240_24008


namespace NUMINAMATH_GPT_set_intersection_l240_24048

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem set_intersection :
  A ∩ B = { x | -1 < x ∧ x < 1 } := 
sorry

end NUMINAMATH_GPT_set_intersection_l240_24048


namespace NUMINAMATH_GPT_range_of_a_l240_24014

noncomputable def has_real_roots (a : ℝ) : Prop :=
  ∃ x y : ℝ, x^2 - a*x + 1 = 0 ∧ y^2 - a*y + 1 = 0

def holds_for_all_x (a : ℝ) : Prop :=
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a^2 - 3*a - x + 1 ≤ 0

theorem range_of_a (a : ℝ) :
  (¬ ((has_real_roots a) ∧ (holds_for_all_x a))) ∧ (¬ (¬ (holds_for_all_x a))) → (1 ≤ a ∧ a < 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l240_24014


namespace NUMINAMATH_GPT_ratio_avg_speed_round_trip_l240_24083

def speed_boat := 20
def speed_current := 4
def distance := 2

theorem ratio_avg_speed_round_trip :
  let downstream_speed := speed_boat + speed_current
  let upstream_speed := speed_boat - speed_current
  let time_down := distance / downstream_speed
  let time_up := distance / upstream_speed
  let total_time := time_down + time_up
  let total_distance := distance + distance
  let avg_speed := total_distance / total_time
  avg_speed / speed_boat = 24 / 25 :=
by sorry

end NUMINAMATH_GPT_ratio_avg_speed_round_trip_l240_24083


namespace NUMINAMATH_GPT_minimum_study_tools_l240_24061

theorem minimum_study_tools (n : Nat) : n^3 ≥ 366 → n ≥ 8 := by
  intros h
  sorry

end NUMINAMATH_GPT_minimum_study_tools_l240_24061


namespace NUMINAMATH_GPT_min_dist_l240_24062

open Complex

theorem min_dist (z w : ℂ) (hz : abs (z - (2 - 5 * I)) = 2) (hw : abs (w - (-3 + 4 * I)) = 4) :
  ∃ d, d = abs (z - w) ∧ d ≥ (Real.sqrt 106 - 6) := sorry

end NUMINAMATH_GPT_min_dist_l240_24062


namespace NUMINAMATH_GPT_triangle_cosine_condition_l240_24069

variable {A B C : ℝ} -- Angles of the triangle
variable {a b c : ℝ} -- Sides opposite to angles A, B, and C

-- Definitions according to the problem conditions
def law_of_sines (a b : ℝ) (A B : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B

theorem triangle_cosine_condition (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : law_of_sines a b A B)
  (h1 : a > b) : Real.cos (2 * A) < Real.cos (2 * B) ↔ a > b :=
by
  sorry

end NUMINAMATH_GPT_triangle_cosine_condition_l240_24069


namespace NUMINAMATH_GPT_total_revenue_correct_l240_24054

def sections := 5
def seats_per_section_1_4 := 246
def seats_section_5 := 314
def ticket_price_1_4 := 15
def ticket_price_5 := 20

theorem total_revenue_correct :
  4 * seats_per_section_1_4 * ticket_price_1_4 + seats_section_5 * ticket_price_5 = 21040 :=
by
  sorry

end NUMINAMATH_GPT_total_revenue_correct_l240_24054


namespace NUMINAMATH_GPT_William_won_10_rounds_l240_24011

theorem William_won_10_rounds (H : ℕ) (total_rounds : H + (H + 5) = 15) : H + 5 = 10 := by
  sorry

end NUMINAMATH_GPT_William_won_10_rounds_l240_24011


namespace NUMINAMATH_GPT_find_x_l240_24039

variables {x y z : ℝ}

theorem find_x (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2) (h2 : y^2 / z = 3) (h3 : z^2 / x = 4) :
  x = 144^(1 / 5) :=
by
  sorry

end NUMINAMATH_GPT_find_x_l240_24039


namespace NUMINAMATH_GPT_sum_of_arithmetic_sequence_l240_24032

theorem sum_of_arithmetic_sequence :
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  S_n = 240 := by {
  let a := -3
  let d := 6
  let n := 10
  let a_n := a + (n - 1) * d
  let S_n := (n / 2) * (a + a_n)
  sorry
}

end NUMINAMATH_GPT_sum_of_arithmetic_sequence_l240_24032


namespace NUMINAMATH_GPT_identify_base_7_l240_24049

theorem identify_base_7 :
  ∃ b : ℕ, (b > 1) ∧ 
  (2 * b^4 + 3 * b^3 + 4 * b^2 + 5 * b^1 + 1 * b^0) +
  (1 * b^4 + 5 * b^3 + 6 * b^2 + 4 * b^1 + 2 * b^0) =
  (4 * b^4 + 2 * b^3 + 4 * b^2 + 2 * b^1 + 3 * b^0) ∧
  b = 7 :=
by
  sorry

end NUMINAMATH_GPT_identify_base_7_l240_24049


namespace NUMINAMATH_GPT_decaf_percentage_correct_l240_24092

def initial_stock : ℝ := 400
def initial_decaf_percent : ℝ := 0.20
def additional_stock : ℝ := 100
def additional_decaf_percent : ℝ := 0.70

theorem decaf_percentage_correct :
  ((initial_decaf_percent * initial_stock + additional_decaf_percent * additional_stock) / (initial_stock + additional_stock)) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_decaf_percentage_correct_l240_24092


namespace NUMINAMATH_GPT_ratio_of_boys_to_girls_l240_24070

-- Define the given conditions and provable statement
theorem ratio_of_boys_to_girls (S G : ℕ) (h : (2/3 : ℚ) * G = (1/5 : ℚ) * S) : (S - G) * 3 = 7 * G :=
by
  -- This is a placeholder for solving the proof
  sorry

end NUMINAMATH_GPT_ratio_of_boys_to_girls_l240_24070


namespace NUMINAMATH_GPT_find_a_l240_24078

variables (a b c : ℝ) (A B C : ℝ) (sin : ℝ → ℝ)
variables (sqrt_three_two sqrt_two_two : ℝ)

-- Assume that A = 60 degrees, B = 45 degrees, and b = sqrt(6)
def angle_A : A = π / 3 := by
  sorry

def angle_B : B = π / 4 := by
  sorry

def side_b : b = Real.sqrt 6 := by
  sorry

def sin_60 : sin (π / 3) = sqrt_three_two := by
  sorry

def sin_45 : sin (π / 4) = sqrt_two_two := by
  sorry

-- Prove that a = 3 based on the given conditions
theorem find_a (sin_rule : a / sin A = b / sin B)
  (sin_60_def : sqrt_three_two = Real.sqrt 3 / 2)
  (sin_45_def : sqrt_two_two = Real.sqrt 2 / 2) : a = 3 := by
  sorry

end NUMINAMATH_GPT_find_a_l240_24078


namespace NUMINAMATH_GPT_x_cube_plus_y_cube_l240_24093

theorem x_cube_plus_y_cube (x y : ℝ) (h₁ : x + y = 1) (h₂ : x^2 + y^2 = 3) : x^3 + y^3 = 4 :=
sorry

end NUMINAMATH_GPT_x_cube_plus_y_cube_l240_24093


namespace NUMINAMATH_GPT_point_on_circle_l240_24094

theorem point_on_circle 
    (P : ℝ × ℝ) 
    (h_l1 : 2 * P.1 - 3 * P.2 + 4 = 0)
    (h_l2 : 3 * P.1 - 2 * P.2 + 1 = 0) 
    (h_circle : (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5) : 
    (P.1 - 2) ^ 2 + (P.2 - 4) ^ 2 = 5 :=
by
  sorry

end NUMINAMATH_GPT_point_on_circle_l240_24094


namespace NUMINAMATH_GPT_meeting_distance_and_time_l240_24007

theorem meeting_distance_and_time 
  (total_distance : ℝ)
  (delta_time : ℝ)
  (x : ℝ)
  (V : ℝ)
  (v : ℝ)
  (t : ℝ) :

  -- Conditions 
  total_distance = 150 ∧
  delta_time = 25 ∧
  (150 - 2 * x) = 25 ∧
  (62.5 / v) = (87.5 / V) ∧
  (150 / v) - (150 / V) = 25 ∧
  t = (62.5 / v)

  -- Show that 
  → x = 62.5 ∧ t = 36 + 28 / 60 := 
by 
  sorry

end NUMINAMATH_GPT_meeting_distance_and_time_l240_24007


namespace NUMINAMATH_GPT_cemc_basketball_team_l240_24036

theorem cemc_basketball_team (t g : ℕ) (h_t : t = 6)
  (h1 : 40 * t + 20 * g = 28 * (g + 4)) :
  g = 16 := by
  -- Start your proof here

  sorry

end NUMINAMATH_GPT_cemc_basketball_team_l240_24036
