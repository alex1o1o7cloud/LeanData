import Mathlib

namespace NUMINAMATH_GPT_cube_side_length_is_30_l2020_202032

theorem cube_side_length_is_30
  (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (s : ℝ)
  (h1 : cost_per_kg = 40)
  (h2 : coverage_per_kg = 20)
  (h3 : total_cost = 10800)
  (total_surface_area : ℝ) (W : ℝ) (C : ℝ)
  (h4 : total_surface_area = 6 * s^2)
  (h5 : W = total_surface_area / coverage_per_kg)
  (h6 : C = W * cost_per_kg)
  (h7 : C = total_cost) :
  s = 30 :=
by
  sorry

end NUMINAMATH_GPT_cube_side_length_is_30_l2020_202032


namespace NUMINAMATH_GPT_distinct_remainders_count_l2020_202018

theorem distinct_remainders_count {N : ℕ} (hN : N = 420) :
  ∃ (count : ℕ), (∀ n : ℕ, n ≥ 1 ∧ n ≤ N → ((n % 5 ≠ n % 6) ∧ (n % 5 ≠ n % 7) ∧ (n % 6 ≠ n % 7))) →
  count = 386 :=
by {
  sorry
}

end NUMINAMATH_GPT_distinct_remainders_count_l2020_202018


namespace NUMINAMATH_GPT_calculate_expression_l2020_202068

theorem calculate_expression : (3.242 * 16) / 100 = 0.51872 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2020_202068


namespace NUMINAMATH_GPT_sin_minus_cos_l2020_202070

theorem sin_minus_cos (θ : ℝ) (hθ1 : θ > 0) (hθ2 : θ < π / 2) (h_tan : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by 
  sorry

end NUMINAMATH_GPT_sin_minus_cos_l2020_202070


namespace NUMINAMATH_GPT_sqrt_of_16_l2020_202008

theorem sqrt_of_16 : Real.sqrt 16 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_of_16_l2020_202008


namespace NUMINAMATH_GPT_completing_the_square_result_l2020_202020

theorem completing_the_square_result (x : ℝ) : (x - 2) ^ 2 = 5 ↔ x ^ 2 - 4 * x - 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_completing_the_square_result_l2020_202020


namespace NUMINAMATH_GPT_value_of_2_pow_b_l2020_202066

theorem value_of_2_pow_b (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h1 : (2 ^ a) ^ b = 2 ^ 2) (h2 : 2 ^ a * 2 ^ b = 8) : 2 ^ b = 4 :=
by
  sorry

end NUMINAMATH_GPT_value_of_2_pow_b_l2020_202066


namespace NUMINAMATH_GPT_watch_correct_time_l2020_202097

-- Conditions
def initial_time_slow : ℕ := 4 -- minutes slow at 8:00 AM
def final_time_fast : ℕ := 6 -- minutes fast at 4:00 PM
def total_time_interval : ℕ := 480 -- total time interval in minutes from 8:00 AM to 4:00 PM
def rate_of_time_gain : ℚ := (initial_time_slow + final_time_fast) / total_time_interval

-- Statement to prove
theorem watch_correct_time : 
  ∃ t : ℕ, t = 11 * 60 + 12 ∧ 
  ((8 * 60 + t) * rate_of_time_gain = 4) := 
sorry

end NUMINAMATH_GPT_watch_correct_time_l2020_202097


namespace NUMINAMATH_GPT_angle_C_correct_l2020_202037

theorem angle_C_correct (A B C : ℝ) (h1 : A = 65) (h2 : B = 40) (h3 : A + B + C = 180) : C = 75 :=
sorry

end NUMINAMATH_GPT_angle_C_correct_l2020_202037


namespace NUMINAMATH_GPT_simplify_expression_l2020_202046

theorem simplify_expression (x y : ℝ) (h1 : x = 10) (h2 : y = -1/25) :
  ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2020_202046


namespace NUMINAMATH_GPT_add_two_integers_l2020_202003

/-- If the difference of two positive integers is 5 and their product is 180,
then their sum is 25. -/
theorem add_two_integers {x y : ℕ} (h1: x > y) (h2: x - y = 5) (h3: x * y = 180) : x + y = 25 :=
sorry

end NUMINAMATH_GPT_add_two_integers_l2020_202003


namespace NUMINAMATH_GPT_geometric_series_common_ratio_l2020_202012

theorem geometric_series_common_ratio (a S r : ℝ) (ha : a = 512) (hS : S = 2048) (h_sum : S = a / (1 - r)) : r = 3 / 4 :=
by
  rw [ha, hS] at h_sum 
  sorry

end NUMINAMATH_GPT_geometric_series_common_ratio_l2020_202012


namespace NUMINAMATH_GPT_vec_mag_diff_eq_neg_one_l2020_202085

variables (a b : ℝ × ℝ)

def vec_add_eq := a + b = (2, 3)

def vec_sub_eq := a - b = (-2, 1)

theorem vec_mag_diff_eq_neg_one (h₁ : vec_add_eq a b) (h₂ : vec_sub_eq a b) :
  (a.1 ^ 2 + a.2 ^ 2) - (b.1 ^ 2 + b.2 ^ 2) = -1 :=
  sorry

end NUMINAMATH_GPT_vec_mag_diff_eq_neg_one_l2020_202085


namespace NUMINAMATH_GPT_student_sums_attempted_l2020_202021

theorem student_sums_attempted (sums_right sums_wrong : ℕ) (h1 : sums_wrong = 2 * sums_right) (h2 : sums_right = 16) :
  sums_right + sums_wrong = 48 :=
by
  sorry

end NUMINAMATH_GPT_student_sums_attempted_l2020_202021


namespace NUMINAMATH_GPT_find_v_l2020_202026

variables (a b c : ℝ)

def condition1 := (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -6
def condition2 := (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 8

theorem find_v (h1 : condition1 a b c) (h2 : condition2 a b c) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 17 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_v_l2020_202026


namespace NUMINAMATH_GPT_min_value_fraction_l2020_202077

-- We start by defining the geometric sequence and the given conditions
variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {a1 : ℝ} (h_pos : ∀ n, 0 < a n)
variable (h_geo : ∀ n, a (n + 1) = a n * r)
variable (h_a7 : a 7 = a 6 + 2 * a 5)
variable (h_am_an : ∃ m n, a m * a n = 16 * (a 1)^2)

theorem min_value_fraction : 
  ∃ (m n : ℕ), (a m * a n = 16 * (a 1)^2 ∧ (1/m) + (4/n) = 1) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l2020_202077


namespace NUMINAMATH_GPT_total_animal_crackers_eaten_l2020_202034

-- Define the context and conditions
def number_of_students : ℕ := 20
def uneaten_students : ℕ := 2
def crackers_per_pack : ℕ := 10

-- Define the statement and prove the question equals the answer given the conditions
theorem total_animal_crackers_eaten : 
  (number_of_students - uneaten_students) * crackers_per_pack = 180 := by
  sorry

end NUMINAMATH_GPT_total_animal_crackers_eaten_l2020_202034


namespace NUMINAMATH_GPT_sheets_in_stack_l2020_202042

theorem sheets_in_stack 
  (num_sheets : ℕ) 
  (initial_thickness final_thickness : ℝ) 
  (t_per_sheet : ℝ) 
  (h_initial : num_sheets = 800) 
  (h_thickness : initial_thickness = 4) 
  (h_thickness_per_sheet : initial_thickness / num_sheets = t_per_sheet) 
  (h_final_thickness : final_thickness = 6) 
  : num_sheets * (final_thickness / t_per_sheet) = 1200 := 
by 
  sorry

end NUMINAMATH_GPT_sheets_in_stack_l2020_202042


namespace NUMINAMATH_GPT_speed_of_boat_in_still_water_l2020_202099

variables (Vb Vs : ℝ)

-- Conditions
def condition_1 : Prop := Vb + Vs = 11
def condition_2 : Prop := Vb - Vs = 5

theorem speed_of_boat_in_still_water (h1 : condition_1 Vb Vs) (h2 : condition_2 Vb Vs) : Vb = 8 := 
by sorry

end NUMINAMATH_GPT_speed_of_boat_in_still_water_l2020_202099


namespace NUMINAMATH_GPT_chadsRopeLength_l2020_202013

-- Define the constants and conditions
def joeysRopeLength : ℕ := 56
def joeyChadRatioNumerator : ℕ := 8
def joeyChadRatioDenominator : ℕ := 3

-- Prove that Chad's rope length is 21 cm
theorem chadsRopeLength (C : ℕ) 
  (h_ratio : joeysRopeLength * joeyChadRatioDenominator = joeyChadRatioNumerator * C) : 
  C = 21 :=
sorry

end NUMINAMATH_GPT_chadsRopeLength_l2020_202013


namespace NUMINAMATH_GPT_units_digit_proof_l2020_202082

def units_digit (n : ℤ) : ℤ := n % 10

theorem units_digit_proof :
  ∀ (a b c : ℤ),
  a = 8 →
  b = 18 →
  c = 1988 →
  units_digit (a * b * c - a^3) = 0 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_units_digit_proof_l2020_202082


namespace NUMINAMATH_GPT_agnes_twice_jane_in_years_l2020_202086

def agnes_age := 25
def jane_age := 6

theorem agnes_twice_jane_in_years (x : ℕ) : 
  25 + x = 2 * (6 + x) → x = 13 :=
by
  intro h
  -- The proof steps would go here, but we'll use sorry to skip them
  sorry

end NUMINAMATH_GPT_agnes_twice_jane_in_years_l2020_202086


namespace NUMINAMATH_GPT_smallest_x_solution_l2020_202074

theorem smallest_x_solution (x : ℚ) :
  (7 * (8 * x^2 + 8 * x + 11) = x * (8 * x - 45)) →
  (x = -7/3 ∨ x = -11/16) →
  x = -7/3 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_solution_l2020_202074


namespace NUMINAMATH_GPT_minimum_focal_length_of_hyperbola_l2020_202027

noncomputable def minimum_focal_length (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) 
    (h₃ : (1/2) * a * (2 * b) = 8) : ℝ :=
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let focal_length := 2 * c
  focal_length

theorem minimum_focal_length_of_hyperbola 
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : (1/2) * a * (2 * b) = 8) :
  minimum_focal_length a b h₁ h₂ h₃ = 8 :=
by
  sorry

end NUMINAMATH_GPT_minimum_focal_length_of_hyperbola_l2020_202027


namespace NUMINAMATH_GPT_rationalize_denominator_ABC_l2020_202017

theorem rationalize_denominator_ABC :
  let expr := (2 + Real.sqrt 5) / (3 - 2 * Real.sqrt 5)
  ∃ A B C : ℤ, expr = A + B * Real.sqrt C ∧ A * B * (C:ℤ) = -560 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_ABC_l2020_202017


namespace NUMINAMATH_GPT_wheel_speed_l2020_202094

theorem wheel_speed (r : ℝ) (c : ℝ) (ts tf : ℝ) 
  (h₁ : c = 13) 
  (h₂ : r * ts = c / 5280) 
  (h₃ : (r + 6) * (tf - 1/3 / 3600) = c / 5280) 
  (h₄ : tf = ts - 1 / 10800) :
  r = 12 :=
  sorry

end NUMINAMATH_GPT_wheel_speed_l2020_202094


namespace NUMINAMATH_GPT_cheesecake_factory_hours_per_day_l2020_202004

theorem cheesecake_factory_hours_per_day
  (wage_per_hour : ℝ)
  (days_per_week : ℝ)
  (weeks : ℝ)
  (combined_savings : ℝ)
  (robbie_saves : ℝ)
  (jaylen_saves : ℝ)
  (miranda_saves : ℝ)
  (h : ℝ) :
  wage_per_hour = 10 → days_per_week = 5 → weeks = 4 → combined_savings = 3000 →
  robbie_saves = 2/5 → jaylen_saves = 3/5 → miranda_saves = 1/2 →
  (robbie_saves * (wage_per_hour * h * days_per_week) +
  jaylen_saves * (wage_per_hour * h * days_per_week) +
  miranda_saves * (wage_per_hour * h * days_per_week)) * weeks = combined_savings →
  h = 10 :=
by
  intros hwage hweek hweeks hsavings hrobbie hjaylen hmiranda heq
  sorry

end NUMINAMATH_GPT_cheesecake_factory_hours_per_day_l2020_202004


namespace NUMINAMATH_GPT_geometric_and_arithmetic_sequence_solution_l2020_202050

theorem geometric_and_arithmetic_sequence_solution:
  ∃ a b : ℝ, 
    (a > 0) ∧                  -- a is positive
    (∃ r : ℝ, 10 * r = a ∧ a * r = 1 / 2) ∧   -- geometric sequence condition
    (∃ d : ℝ, a + d = 5 ∧ 5 + d = b) ∧        -- arithmetic sequence condition
    a = Real.sqrt 5 ∧
    b = 10 - Real.sqrt 5 := 
by 
  sorry

end NUMINAMATH_GPT_geometric_and_arithmetic_sequence_solution_l2020_202050


namespace NUMINAMATH_GPT_units_digit_base9_addition_l2020_202047

theorem units_digit_base9_addition : 
  (∃ (d₁ d₂ : ℕ), d₁ < 9 ∧ d₂ < 9 ∧ (85 % 9 = d₁) ∧ (37 % 9 = d₂)) → ((d₁ + d₂) % 9 = 3) :=
by
  sorry

end NUMINAMATH_GPT_units_digit_base9_addition_l2020_202047


namespace NUMINAMATH_GPT_find_total_games_l2020_202015

-- Define the initial conditions
def avg_points_per_game : ℕ := 26
def games_played : ℕ := 15
def goal_avg_points : ℕ := 30
def required_avg_remaining : ℕ := 42

-- Statement of the proof problem
theorem find_total_games (G : ℕ) :
  avg_points_per_game * games_played + required_avg_remaining * (G - games_played) = goal_avg_points * G →
  G = 20 :=
by sorry

end NUMINAMATH_GPT_find_total_games_l2020_202015


namespace NUMINAMATH_GPT_terrell_lifting_problem_l2020_202054

theorem terrell_lifting_problem (w1 w2 w3 n1 n2 : ℕ) (h1 : w1 = 12) (h2 : w2 = 18) (h3 : w3 = 24) (h4 : n1 = 20) :
  60 * n2 = 3 * w1 * n1 → n2 = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_terrell_lifting_problem_l2020_202054


namespace NUMINAMATH_GPT_g_at_2_l2020_202043

def g (x : ℝ) : ℝ := x^2 - 4

theorem g_at_2 : g 2 = 0 := by
  sorry

end NUMINAMATH_GPT_g_at_2_l2020_202043


namespace NUMINAMATH_GPT_sqrt_expression_meaningful_l2020_202041

/--
When is the algebraic expression √(x + 2) meaningful?
To ensure the algebraic expression √(x + 2) is meaningful, 
the expression under the square root, x + 2, must be greater than or equal to 0.
Thus, we need to prove that this condition is equivalent to x ≥ -2.
-/
theorem sqrt_expression_meaningful (x : ℝ) : (x + 2 ≥ 0) ↔ (x ≥ -2) :=
by
  sorry

end NUMINAMATH_GPT_sqrt_expression_meaningful_l2020_202041


namespace NUMINAMATH_GPT_algebraic_expression_value_l2020_202048

theorem algebraic_expression_value (x : ℝ) (h : x = 5) : (3 / (x - 4) - 24 / (x^2 - 16)) = (1 / 3) :=
by
  have hx : x = 5 := h
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l2020_202048


namespace NUMINAMATH_GPT_initial_salty_cookies_count_l2020_202089

-- Define initial conditions
def initial_sweet_cookies : ℕ := 9
def sweet_cookies_ate : ℕ := 36
def salty_cookies_left : ℕ := 3
def salty_cookies_ate : ℕ := 3

-- Theorem to prove the initial salty cookies count
theorem initial_salty_cookies_count (initial_salty_cookies : ℕ) 
    (initial_sweet_cookies : initial_sweet_cookies = 9) 
    (sweet_cookies_ate : sweet_cookies_ate = 36)
    (salty_cookies_ate : salty_cookies_ate = 3) 
    (salty_cookies_left : salty_cookies_left = 3) : 
    initial_salty_cookies = 6 := 
sorry

end NUMINAMATH_GPT_initial_salty_cookies_count_l2020_202089


namespace NUMINAMATH_GPT_triangles_xyz_l2020_202039

theorem triangles_xyz (A B C D P Q R : Type) 
    (u v w x : ℝ)
    (angle_ADB angle_BDC angle_CDA : ℝ)
    (h1 : angle_ADB = 120) 
    (h2 : angle_BDC = 120) 
    (h3 : angle_CDA = 120) :
    x = u + v + w :=
sorry

end NUMINAMATH_GPT_triangles_xyz_l2020_202039


namespace NUMINAMATH_GPT_alyosha_cube_cut_l2020_202091

theorem alyosha_cube_cut (n s : ℕ) (h1 : n > 5) (h2 : n^3 - s^3 = 152)
  : n = 6 := by
  sorry

end NUMINAMATH_GPT_alyosha_cube_cut_l2020_202091


namespace NUMINAMATH_GPT_percentage_of_money_spent_l2020_202045

theorem percentage_of_money_spent (initial_amount remaining_amount : ℝ) (h_initial : initial_amount = 500) (h_remaining : remaining_amount = 350) :
  (((initial_amount - remaining_amount) / initial_amount) * 100) = 30 :=
by
  -- Start the proof
  sorry

end NUMINAMATH_GPT_percentage_of_money_spent_l2020_202045


namespace NUMINAMATH_GPT_form_five_squares_l2020_202058

-- The conditions of the problem as premises
variables (initial_configuration : Set (ℕ × ℕ))               -- Initial positions of 12 matchsticks
          (final_configuration : Set (ℕ × ℕ))                 -- Final positions of matchsticks to form 5 squares
          (fixed_matchsticks : Set (ℕ × ℕ))                    -- Positions of 6 fixed matchsticks
          (movable_matchsticks : Set (ℕ × ℕ))                 -- Positions of 6 movable matchsticks

-- Condition to avoid duplication or free ends
variables (no_duplication : Prop)
          (no_free_ends : Prop)

-- Proof statement
theorem form_five_squares : ∃ rearranged_configuration, 
  rearranged_configuration = final_configuration ∧
  initial_configuration = fixed_matchsticks ∪ movable_matchsticks ∧
  no_duplication ∧
  no_free_ends :=
sorry -- Proof omitted.

end NUMINAMATH_GPT_form_five_squares_l2020_202058


namespace NUMINAMATH_GPT_library_growth_rate_l2020_202069

theorem library_growth_rate (C_2022 C_2024: ℝ) (h₁ : C_2022 = 100000) (h₂ : C_2024 = 144000) :
  ∃ x : ℝ, (1 + x) ^ 2 = C_2024 / C_2022 ∧ x = 0.2 := 
by {
  sorry
}

end NUMINAMATH_GPT_library_growth_rate_l2020_202069


namespace NUMINAMATH_GPT_max_value_of_expression_l2020_202052

theorem max_value_of_expression :
  ∀ (a b c : ℝ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 2 ∧ 0 ≤ c ∧ c ≤ 2 → 
  a^2 * b^2 * c^2 + (2 - a)^2 * (2 - b)^2 * (2 - c)^2 ≤ 64 :=
by sorry

end NUMINAMATH_GPT_max_value_of_expression_l2020_202052


namespace NUMINAMATH_GPT_log_expression_equality_l2020_202060

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_equality :
  Real.log 4 / Real.log 10 + 2 * (Real.log 5 / Real.log 10) + (log_base 2 5) * (log_base 5 8) = 5 := by
  sorry

end NUMINAMATH_GPT_log_expression_equality_l2020_202060


namespace NUMINAMATH_GPT_price_of_second_tea_l2020_202053

theorem price_of_second_tea (price_first_tea : ℝ) (mixture_price : ℝ) (required_ratio : ℝ) (price_second_tea : ℝ) :
  price_first_tea = 62 → mixture_price = 64.5 → required_ratio = 3 → price_second_tea = 65.33 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_price_of_second_tea_l2020_202053


namespace NUMINAMATH_GPT_find_door_height_l2020_202075

theorem find_door_height :
  ∃ (h : ℝ), 
  let l := 25
  let w := 15
  let H := 12
  let A := 80 * H
  let W := 960 - (6 * h + 36)
  let cost := 4 * W
  cost = 3624 ∧ h = 3 := sorry

end NUMINAMATH_GPT_find_door_height_l2020_202075


namespace NUMINAMATH_GPT_complement_intersection_l2020_202090

-- Definitions of the sets as given in the problem
namespace ProofProblem

def U : Set ℤ := {-2, -1, 0, 1, 2}
def M : Set ℤ := {y | y > 0}
def N : Set ℤ := {x | x = -1 ∨ x = 2}

theorem complement_intersection :
  (U \ M) ∩ N = {-1} :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_complement_intersection_l2020_202090


namespace NUMINAMATH_GPT_unit_square_BE_value_l2020_202098

theorem unit_square_BE_value
  (ABCD : ℝ × ℝ → Prop)
  (unit_square : ∀ (a b c d : ℝ × ℝ), ABCD a ∧ ABCD b ∧ ABCD c ∧ ABCD d → 
                  a.1 = 0 ∧ a.2 = 0 ∧ b.1 = 1 ∧ b.2 = 0 ∧ 
                  c.1 = 1 ∧ c.2 = 1 ∧ d.1 = 0 ∧ d.2 = 1)
  (E F G : ℝ × ℝ)
  (on_sides : E.1 = 1 ∧ F.2 = 1 ∧ G.1 = 0)
  (AE_perp_EF : ((E.1 - 0) * (F.2 - E.2)) + ((E.2 - 0) * (F.1 - E.1)) = 0)
  (EF_perp_FG : ((F.1 - E.1) * (G.2 - F.2)) + ((F.2 - E.2) * (G.1 - F.1)) = 0)
  (GA_val : (1 - G.1) = 404 / 1331) :
  ∃ BE, BE = 9 / 11 := 
sorry

end NUMINAMATH_GPT_unit_square_BE_value_l2020_202098


namespace NUMINAMATH_GPT_pigeons_problem_l2020_202059

theorem pigeons_problem
  (x y : ℕ)
  (h1 : 6 * y + 3 = x)
  (h2 : 8 * y = x + 5) : x = 27 := 
sorry

end NUMINAMATH_GPT_pigeons_problem_l2020_202059


namespace NUMINAMATH_GPT_janets_shampoo_days_l2020_202055

-- Definitions from the problem conditions
def rose_shampoo := 1 / 3
def jasmine_shampoo := 1 / 4
def daily_usage := 1 / 12

-- Define the total shampoo and the days lasts
def total_shampoo := rose_shampoo + jasmine_shampoo
def days_lasts := total_shampoo / daily_usage

-- The theorem to be proved
theorem janets_shampoo_days : days_lasts = 7 :=
by sorry

end NUMINAMATH_GPT_janets_shampoo_days_l2020_202055


namespace NUMINAMATH_GPT_geom_sequence_second_term_l2020_202061

noncomputable def geom_sequence_term (a r : ℕ) (n : ℕ) : ℕ := a * r^(n-1)

theorem geom_sequence_second_term 
  (a1 a5: ℕ) (r: ℕ) 
  (h1: a1 = 5)
  (h2: a5 = geom_sequence_term a1 r 5)
  (h3: a5 = 320)
  (h_r: r^4 = 64): 
  geom_sequence_term a1 r 2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_geom_sequence_second_term_l2020_202061


namespace NUMINAMATH_GPT_value_of_expression_l2020_202083

theorem value_of_expression (x : ℤ) (h : x = 5) : x^5 - 10 * x = 3075 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2020_202083


namespace NUMINAMATH_GPT_solve_work_problem_l2020_202063

variables (A B C : ℚ)

-- Conditions
def condition1 := B + C = 1/3
def condition2 := C + A = 1/4
def condition3 := C = 1/24

-- Conclusion (Question translated to proof statement)
theorem solve_work_problem (h1 : condition1 B C) (h2 : condition2 C A) (h3 : condition3 C) : A + B = 1/2 :=
by sorry

end NUMINAMATH_GPT_solve_work_problem_l2020_202063


namespace NUMINAMATH_GPT_perfect_square_count_between_20_and_150_l2020_202073

theorem perfect_square_count_between_20_and_150 :
  let lower_bound := 20
  let upper_bound := 150
  let smallest_ps := 25
  let largest_ps := 144
  let count_squares (a b : Nat) := b - a
  count_squares 4 12 = 8 := sorry

end NUMINAMATH_GPT_perfect_square_count_between_20_and_150_l2020_202073


namespace NUMINAMATH_GPT_decimal_to_binary_25_l2020_202028

theorem decimal_to_binary_25: (1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0) = 25 :=
by 
  sorry

end NUMINAMATH_GPT_decimal_to_binary_25_l2020_202028


namespace NUMINAMATH_GPT_compare_exponents_and_logs_l2020_202023

theorem compare_exponents_and_logs :
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  a > b ∧ b > c :=
by
  -- Definitions from the conditions
  let a := Real.sqrt 2
  let b := Real.log 3 / Real.log π
  let c := Real.log 0.5 / Real.log 2
  -- Proof here (omitted)
  sorry

end NUMINAMATH_GPT_compare_exponents_and_logs_l2020_202023


namespace NUMINAMATH_GPT_verify_number_of_true_props_l2020_202096

def original_prop (a : ℝ) : Prop := a > -3 → a > 0
def converse_prop (a : ℝ) : Prop := a > 0 → a > -3
def inverse_prop (a : ℝ) : Prop := a ≤ -3 → a ≤ 0
def contrapositive_prop (a : ℝ) : Prop := a ≤ 0 → a ≤ -3

theorem verify_number_of_true_props :
  (¬ original_prop a ∧ converse_prop a ∧ inverse_prop a ∧ ¬ contrapositive_prop a) → (2 = 2) := sorry

end NUMINAMATH_GPT_verify_number_of_true_props_l2020_202096


namespace NUMINAMATH_GPT_region_area_l2020_202038

/-- 
  Trapezoid has side lengths 10, 10, 10, and 22. 
  Each side of the trapezoid is the diameter of a semicircle 
  with the two semicircles on the two parallel sides of the trapezoid facing outside 
  and the other two semicircles facing inside the trapezoid.
  The region bounded by these four semicircles has area m + nπ, where m and n are positive integers.
  Prove that m + n = 188.5.
-/
theorem region_area (m n : ℝ) (h1: m = 128) (h2: n = 60.5) : m + n = 188.5 :=
by
  rw [h1, h2]
  norm_num -- simplifies the expression and checks it is equal to 188.5

end NUMINAMATH_GPT_region_area_l2020_202038


namespace NUMINAMATH_GPT_total_flowers_sold_l2020_202019

theorem total_flowers_sold :
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  flowers_mon + flowers_tue + flowers_wed + flowers_thu + flowers_fri + flowers_sat = 78 :=
by
  let flowers_mon := 4
  let flowers_tue := 8
  let flowers_wed := flowers_mon + 3
  let flowers_thu := 6
  let flowers_fri := flowers_mon * 2
  let flowers_sat := 5 * 9
  sorry

end NUMINAMATH_GPT_total_flowers_sold_l2020_202019


namespace NUMINAMATH_GPT_base3_addition_proof_l2020_202062

-- Define the base 3 numbers
def one_3 : ℕ := 1
def twelve_3 : ℕ := 1 * 3 + 2
def two_hundred_twelve_3 : ℕ := 2 * 3^2 + 1 * 3 + 2
def two_thousand_one_hundred_twenty_one_3 : ℕ := 2 * 3^3 + 1 * 3^2 + 2 * 3 + 1

-- Define the correct answer in base 3
def expected_sum_3 : ℕ := 1 * 3^4 + 0 * 3^3 + 2 * 3^2 + 0 * 3 + 0

-- The proof problem
theorem base3_addition_proof :
  one_3 + twelve_3 + two_hundred_twelve_3 + two_thousand_one_hundred_twenty_one_3 = expected_sum_3 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_base3_addition_proof_l2020_202062


namespace NUMINAMATH_GPT_problem_l2020_202000

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1 / 2) * m * x^2 + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := 2 * Real.log x - (2 * m + 1) * x - 1
noncomputable def h (m : ℝ) (x : ℝ) := f m x + g m x

noncomputable def h_deriv (m : ℝ) (x : ℝ) : ℝ := m * x - (2 * m + 1) + (2 / x)

theorem problem (m : ℝ) : h_deriv m 1 = h_deriv m 3 → m = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_l2020_202000


namespace NUMINAMATH_GPT_Bella_average_speed_l2020_202035

theorem Bella_average_speed :
  ∀ (distance time : ℝ), 
  distance = 790 → 
  time = 15.8 → 
  (distance / time) = 50 :=
by intros distance time h_dist h_time
   -- According to the provided distances and time,
   -- we need to prove that the calculated speed is 50.
   sorry

end NUMINAMATH_GPT_Bella_average_speed_l2020_202035


namespace NUMINAMATH_GPT_sum_of_distances_eq_l2020_202029

noncomputable def sum_of_distances_from_vertex_to_midpoints (A B C M N O : ℝ × ℝ) : ℝ :=
  let AM := Real.sqrt ((M.1 - A.1)^2 + (M.2 - A.2)^2)
  let AN := Real.sqrt ((N.1 - A.1)^2 + (N.2 - A.2)^2)
  let AO := Real.sqrt ((O.1 - A.1)^2 + (O.2 - A.2)^2)
  AM + AN + AO

theorem sum_of_distances_eq (A B C M N O : ℝ × ℝ) (h1 : B = (3, 0)) (h2 : C = (3/2, (3 * Real.sqrt 3/2))) (h3 : M = (3/2, 0)) (h4 : N = (9/4, (3 * Real.sqrt 3/4))) (h5 : O = (3/4, (3 * Real.sqrt 3/4))) :
  sum_of_distances_from_vertex_to_midpoints A B C M N O = 3 + (9 / 2) * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_distances_eq_l2020_202029


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l2020_202011

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) 
(h_given : a 2 + a 8 = 10) : 
a 3 + a 7 = 10 :=
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l2020_202011


namespace NUMINAMATH_GPT_right_triangle_ABC_l2020_202071

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Points definitions
def point_A : ℝ × ℝ := (1, 2)
def point_on_line : ℝ × ℝ := (5, -2)

-- Points B and C on the parabola with parameters t and s respectively
def point_B (t : ℝ) : ℝ × ℝ := (t^2, 2 * t)
def point_C (s : ℝ) : ℝ × ℝ := (s^2, 2 * s)

-- Line equation passing through points B and C
def line_eq (s t : ℝ) (x y : ℝ) : Prop :=
  2 * x - (s + t) * y + 2 * s * t = 0

-- Proof goal: Show that triangle ABC is a right triangle
theorem right_triangle_ABC
  (t s : ℝ)
  (hB : parabola (point_B t).1 (point_B t).2)
  (hC : parabola (point_C s).1 (point_C s).2)
  (hlt : point_on_line.1 = (5 : ℝ))
  (hlx : line_eq s t point_on_line.1 point_on_line.2)
  : let A := point_A
    let B := point_B t
    let C := point_C s
    -- Conclusion: triangle ABC is a right triangle
    k_AB * k_AC = -1 :=
  sorry
  where k_AB := (2 * t - 2) / (t^2 - 1)
        k_AC := (2 * s - 2) / (s^2 - 1)
        rel_t_s := (s + 1) * (t + 1) = -4

end NUMINAMATH_GPT_right_triangle_ABC_l2020_202071


namespace NUMINAMATH_GPT_walnut_trees_total_l2020_202025

variable (current_trees : ℕ) (new_trees : ℕ)

theorem walnut_trees_total (h1 : current_trees = 22) (h2 : new_trees = 55) : current_trees + new_trees = 77 :=
by
  sorry

end NUMINAMATH_GPT_walnut_trees_total_l2020_202025


namespace NUMINAMATH_GPT_find_some_multiplier_l2020_202001

theorem find_some_multiplier (m : ℕ) :
  (422 + 404)^2 - (m * 422 * 404) = 324 ↔ m = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_some_multiplier_l2020_202001


namespace NUMINAMATH_GPT_value_of_f_neg_2009_l2020_202030

def f (a b x : ℝ) : ℝ := a * x^7 + b * x - 2

theorem value_of_f_neg_2009 (a b : ℝ) (h : f a b 2009 = 10) :
  f a b (-2009) = -14 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_f_neg_2009_l2020_202030


namespace NUMINAMATH_GPT_at_least_one_success_l2020_202076

-- Define probabilities for A, B, and C
def pA : ℚ := 1 / 2
def pB : ℚ := 2 / 3
def pC : ℚ := 4 / 5

-- Define the probability that none succeed
def pNone : ℚ := (1 - pA) * (1 - pB) * (1 - pC)

-- Define the probability that at least one of them succeeds
def pAtLeastOne : ℚ := 1 - pNone

theorem at_least_one_success : pAtLeastOne = 29 / 30 := 
by sorry

end NUMINAMATH_GPT_at_least_one_success_l2020_202076


namespace NUMINAMATH_GPT_emily_101st_card_is_10_of_Hearts_l2020_202084

def number_sequence : List String := ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
def suit_sequence : List String := ["Hearts", "Diamonds", "Clubs", "Spades"]

-- Function to get the number of a specific card
def card_number (n : ℕ) : String :=
  number_sequence.get! (n % number_sequence.length)

-- Function to get the suit of a specific card
def card_suit (n : ℕ) : String :=
  suit_sequence.get! ((n / suit_sequence.length) % suit_sequence.length)

-- Definition to state the question and the answer
def emily_card (n : ℕ) : String := card_number n ++ " of " ++ card_suit n

-- Proving that the 101st card is "10 of Hearts"
theorem emily_101st_card_is_10_of_Hearts : emily_card 100 = "10 of Hearts" :=
by {
  sorry
}

end NUMINAMATH_GPT_emily_101st_card_is_10_of_Hearts_l2020_202084


namespace NUMINAMATH_GPT_dawn_wash_dishes_time_l2020_202081

theorem dawn_wash_dishes_time (D : ℕ) : 2 * D + 6 = 46 → D = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_dawn_wash_dishes_time_l2020_202081


namespace NUMINAMATH_GPT_translate_A_coordinates_l2020_202080

-- Definitions
def A_initial : ℝ × ℝ := (-3, 2)

def translate_right (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1 + d, p.2)
def translate_down (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ := (p.1, p.2 - d)

-- Final coordinates after transformation
def A' : ℝ × ℝ :=
  let A_translated := translate_right A_initial 4
  translate_down A_translated 3

-- Proof statement
theorem translate_A_coordinates :
  A' = (1, -1) :=
by
  simp [A', translate_right, translate_down, A_initial]
  sorry

end NUMINAMATH_GPT_translate_A_coordinates_l2020_202080


namespace NUMINAMATH_GPT_green_dots_fifth_row_l2020_202049

variable (R : ℕ → ℕ)

-- Define the number of green dots according to the pattern
def pattern (n : ℕ) : ℕ := 3 * n

-- Define conditions for rows
axiom row_1 : R 1 = 3
axiom row_2 : R 2 = 6
axiom row_3 : R 3 = 9
axiom row_4 : R 4 = 12

-- The theorem
theorem green_dots_fifth_row : R 5 = 15 :=
by
  -- Row 5 follows the pattern and should satisfy the condition R 5 = R 4 + 3
  sorry

end NUMINAMATH_GPT_green_dots_fifth_row_l2020_202049


namespace NUMINAMATH_GPT_range_of_x_l2020_202024

theorem range_of_x (x : ℝ) :
  (x + 1 ≥ 0) ∧ (x - 3 ≠ 0) ↔ (x ≥ -1) ∧ (x ≠ 3) :=
by sorry

end NUMINAMATH_GPT_range_of_x_l2020_202024


namespace NUMINAMATH_GPT_total_canoes_by_end_of_april_l2020_202067

def N_F : ℕ := 4
def N_M : ℕ := 3 * N_F
def N_A : ℕ := 3 * N_M
def total_canoes : ℕ := N_F + N_M + N_A

theorem total_canoes_by_end_of_april : total_canoes = 52 := by
  sorry

end NUMINAMATH_GPT_total_canoes_by_end_of_april_l2020_202067


namespace NUMINAMATH_GPT_find_f_13_l2020_202016

variable (f : ℤ → ℤ)

def is_odd_function (f : ℤ → ℤ) := ∀ x : ℤ, f (-x) = -f (x)
def has_period_4 (f : ℤ → ℤ) := ∀ x : ℤ, f (x + 4) = f (x)

theorem find_f_13 (h1 : is_odd_function f) (h2 : has_period_4 f) (h3 : f (-1) = 2) : f 13 = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_13_l2020_202016


namespace NUMINAMATH_GPT_sequence_general_term_l2020_202010

-- Define the sequence and the sum of the sequence
def Sn (n : ℕ) : ℕ := 3 + 2^n

def an (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n - 1)

-- Proposition stating the equivalence
theorem sequence_general_term (n : ℕ) : 
  (n = 1 → an n = 5) ∧ (n ≠ 1 → an n = 2^(n - 1)) :=
by 
  sorry

end NUMINAMATH_GPT_sequence_general_term_l2020_202010


namespace NUMINAMATH_GPT_student_tickets_sold_l2020_202022

theorem student_tickets_sold (A S : ℝ) (h1 : A + S = 59) (h2 : 4 * A + 2.5 * S = 222.50) : S = 9 :=
by
  sorry

end NUMINAMATH_GPT_student_tickets_sold_l2020_202022


namespace NUMINAMATH_GPT_cistern_width_l2020_202040

theorem cistern_width (w : ℝ) (h : 8 * w + 2 * (1.25 * 8) + 2 * (1.25 * w) = 83) : w = 6 :=
by
  sorry

end NUMINAMATH_GPT_cistern_width_l2020_202040


namespace NUMINAMATH_GPT_two_digit_number_sum_l2020_202079

theorem two_digit_number_sum (a b : ℕ)
  (h1 : 1 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by {
  sorry
}

end NUMINAMATH_GPT_two_digit_number_sum_l2020_202079


namespace NUMINAMATH_GPT_smallest_number_with_ten_divisors_l2020_202033

/-- 
  Theorem: The smallest natural number n that has exactly 10 positive divisors is 48.
--/
theorem smallest_number_with_ten_divisors : 
  ∃ (n : ℕ), (∀ (p1 p2 p3 p4 p5 : ℕ) (a1 a2 a3 a4 a5 : ℕ), 
    n = p1^a1 * p2^a2 * p3^a3 * p4^a4 * p5^a5 → 
    n.factors.count = 10) 
    ∧ n = 48 := sorry

end NUMINAMATH_GPT_smallest_number_with_ten_divisors_l2020_202033


namespace NUMINAMATH_GPT_f_half_and_minus_half_l2020_202065

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem f_half_and_minus_half :
  f (1 / 2) + f (-1 / 2) = 2 := by
  sorry

end NUMINAMATH_GPT_f_half_and_minus_half_l2020_202065


namespace NUMINAMATH_GPT_vertical_asymptote_sum_l2020_202057

theorem vertical_asymptote_sum : 
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  ∃ p q : ℝ, (6 * p ^ 2 + 7 * p + 3 = 0) ∧ (6 * q ^ 2 + 7 * q + 3 = 0) ∧ p + q = -11 / 6 :=
by
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  exact sorry

end NUMINAMATH_GPT_vertical_asymptote_sum_l2020_202057


namespace NUMINAMATH_GPT_double_apply_l2020_202007

def op1 (x : ℤ) : ℤ := 9 - x 
def op2 (x : ℤ) : ℤ := x - 9

theorem double_apply (x : ℤ) : op1 (op2 x) = 3 := by
  sorry

end NUMINAMATH_GPT_double_apply_l2020_202007


namespace NUMINAMATH_GPT_geometric_sequence_sum_l2020_202005

theorem geometric_sequence_sum (S : ℕ → ℝ) 
  (S5 : S 5 = 10)
  (S10 : S 10 = 50) :
  S 15 = 210 := 
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l2020_202005


namespace NUMINAMATH_GPT_probability_of_high_value_hand_l2020_202093

noncomputable def bridge_hand_probability : ℚ :=
  let total_combinations : ℕ := Nat.choose 16 4
  let favorable_combinations : ℕ := 1 + 16 + 16 + 16 + 36 + 96 + 16
  favorable_combinations / total_combinations

theorem probability_of_high_value_hand : bridge_hand_probability = 197 / 1820 := by
  sorry

end NUMINAMATH_GPT_probability_of_high_value_hand_l2020_202093


namespace NUMINAMATH_GPT_no_grammatical_errors_in_B_l2020_202044

-- Definitions for each option’s description (conditions)
def sentence_A := "The \"Criminal Law Amendment (IX)\", which was officially implemented on November 1, 2015, criminalizes exam cheating for the first time, showing the government's strong determination to combat exam cheating, and may become the \"magic weapon\" to govern the chaos of exams."
def sentence_B := "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region."
def sentence_C := "Since the implementation of the comprehensive two-child policy, many Chinese families have chosen not to have a second child. It is said that it's not because they don't want to, but because they can't afford it, as the cost of raising a child in China is too high."
def sentence_D := "Although it ended up being a futile effort, having fought for a dream, cried, and laughed, we are without regrets. For us, such experiences are treasures in themselves."

-- The statement that option B has no grammatical errors
theorem no_grammatical_errors_in_B : sentence_B = "The APEC Business Leaders Summit is held during the annual APEC Leaders' Informal Meeting. It is an important platform for dialogue and exchange between leaders of economies and the business community, and it is also the most influential business event in the Asia-Pacific region." :=
by
  sorry

end NUMINAMATH_GPT_no_grammatical_errors_in_B_l2020_202044


namespace NUMINAMATH_GPT_width_of_Carols_rectangle_l2020_202009

theorem width_of_Carols_rectangle 
  (w : ℝ) 
  (h1 : 15 * w = 6 * 50) : w = 20 := 
by 
  sorry

end NUMINAMATH_GPT_width_of_Carols_rectangle_l2020_202009


namespace NUMINAMATH_GPT_more_pie_eaten_l2020_202002

theorem more_pie_eaten (erik_pie : ℝ) (frank_pie : ℝ)
  (h_erik : erik_pie = 0.6666666666666666)
  (h_frank : frank_pie = 0.3333333333333333) :
  erik_pie - frank_pie = 0.3333333333333333 :=
by
  sorry

end NUMINAMATH_GPT_more_pie_eaten_l2020_202002


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l2020_202095

noncomputable def a : ℝ := Real.exp (-2)

noncomputable def b : ℝ := a ^ a

noncomputable def c : ℝ := a ^ b

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by {
  sorry
}

end NUMINAMATH_GPT_relationship_between_a_b_c_l2020_202095


namespace NUMINAMATH_GPT_probability_palindrome_divisible_by_11_is_zero_l2020_202078

def is_palindrome (n : ℕ) :=
  3000 ≤ n ∧ n < 8000 ∧ ∃ (a b : ℕ), 3 ≤ a ∧ a ≤ 7 ∧ 0 ≤ b ∧ b ≤ 9 ∧ n = 1000 * a + 100 * b + 10 * b + a

theorem probability_palindrome_divisible_by_11_is_zero :
  (∃ (n : ℕ), is_palindrome n ∧ n % 11 = 0) → false := by sorry

end NUMINAMATH_GPT_probability_palindrome_divisible_by_11_is_zero_l2020_202078


namespace NUMINAMATH_GPT_two_le_three_l2020_202064

/-- Proof that the proposition "2 ≤ 3" is true given the logical connective. -/
theorem two_le_three : 2 ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_two_le_three_l2020_202064


namespace NUMINAMATH_GPT_total_salary_after_strict_manager_l2020_202031

-- Definitions based on conditions
def total_initial_salary (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  500 * x + (Finset.sum (Finset.range y) s) = 10000

def kind_manager_total (x y : ℕ) (s : ℕ → ℕ) : Prop :=
  1500 * x + (Finset.sum (Finset.range y) s) + 1000 * y = 24000

def strict_manager_total (x y : ℕ) : ℕ :=
  500 * (x + y)

-- Lean statement to prove the required
theorem total_salary_after_strict_manager (x y : ℕ) (s : ℕ → ℕ) 
  (h_total_initial : total_initial_salary x y s) (h_kind_manager : kind_manager_total x y s) :
  strict_manager_total x y = 7000 := by
  sorry

end NUMINAMATH_GPT_total_salary_after_strict_manager_l2020_202031


namespace NUMINAMATH_GPT_sin_alpha_minus_pi_over_6_l2020_202006

variable (α : ℝ)

theorem sin_alpha_minus_pi_over_6 (h1 : 0 < α ∧ α < π / 2) (h2 : Real.cos (α + π / 6) = 3 / 5) : 
  Real.sin (α - π / 6) = (4 - 3 * Real.sqrt 3) / 10 :=
by
  sorry

end NUMINAMATH_GPT_sin_alpha_minus_pi_over_6_l2020_202006


namespace NUMINAMATH_GPT_x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l2020_202072

-- Define the context and main statement
theorem x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta
  (θ : ℝ)
  (hθ₁ : 0 < θ)
  (hθ₂ : θ < (π / 2))
  {x : ℝ}
  (hx : x + 1 / x = 2 * Real.sin θ)
  (n : ℕ) (hn : 0 < n) :
  x^n + 1 / x^n = 2 * Real.sin (n * θ) :=
sorry

end NUMINAMATH_GPT_x_pow_n_plus_inv_pow_n_eq_two_sin_n_theta_l2020_202072


namespace NUMINAMATH_GPT_age_problem_l2020_202036

variable (A B x : ℕ)

theorem age_problem (h1 : A = B + 5) (h2 : B = 35) (h3 : A + x = 2 * (B - x)) : x = 10 :=
sorry

end NUMINAMATH_GPT_age_problem_l2020_202036


namespace NUMINAMATH_GPT_correct_calculation_l2020_202056

theorem correct_calculation (a : ℝ) : -2 * a + (2 * a - 1) = -1 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l2020_202056


namespace NUMINAMATH_GPT_ellie_shoes_count_l2020_202092

variable (E R : ℕ)

def ellie_shoes (E R : ℕ) : Prop :=
  E + R = 13 ∧ E = R + 3

theorem ellie_shoes_count (E R : ℕ) (h : ellie_shoes E R) : E = 8 :=
  by sorry

end NUMINAMATH_GPT_ellie_shoes_count_l2020_202092


namespace NUMINAMATH_GPT_savings_percentage_correct_l2020_202014

theorem savings_percentage_correct :
  let original_price_jacket := 120
  let original_price_shirt := 60
  let original_price_shoes := 90
  let discount_jacket := 0.30
  let discount_shirt := 0.50
  let discount_shoes := 0.25
  let total_original_price := original_price_jacket + original_price_shirt + original_price_shoes
  let savings_jacket := original_price_jacket * discount_jacket
  let savings_shirt := original_price_shirt * discount_shirt
  let savings_shoes := original_price_shoes * discount_shoes
  let total_savings := savings_jacket + savings_shirt + savings_shoes
  let percentage_savings := (total_savings / total_original_price) * 100
  percentage_savings = 32.8 := 
by 
  sorry

end NUMINAMATH_GPT_savings_percentage_correct_l2020_202014


namespace NUMINAMATH_GPT_product_of_repeating_decimal_and_five_l2020_202088

noncomputable def repeating_decimal : ℚ :=
  456 / 999

theorem product_of_repeating_decimal_and_five : 
  (repeating_decimal * 5) = 760 / 333 :=
by
  -- The proof is omitted.
  sorry

end NUMINAMATH_GPT_product_of_repeating_decimal_and_five_l2020_202088


namespace NUMINAMATH_GPT_find_A_l2020_202051

theorem find_A (A : ℕ) (B : ℕ) (h₁ : 0 ≤ B ∧ B ≤ 999) (h₂ : 1000 * A + B = A * (A + 1) / 2) : A = 1999 :=
  sorry

end NUMINAMATH_GPT_find_A_l2020_202051


namespace NUMINAMATH_GPT_slope_of_parallel_line_l2020_202087

-- Definition of the problem conditions
def line_eq (x y : ℝ) : Prop := 3 * x - 6 * y = 12

-- Theorem: Given the line equation, the slope of any line parallel to it is 1/2
theorem slope_of_parallel_line (m : ℝ) (x y : ℝ) (h : line_eq x y) : m = 1 / 2 := sorry

end NUMINAMATH_GPT_slope_of_parallel_line_l2020_202087
