import Mathlib

namespace CEMC_additional_employees_l799_79927

variable (t : ℝ)

def initialEmployees (t : ℝ) := t + 40

def finalEmployeesMooseJaw (t : ℝ) := 1.25 * t

def finalEmployeesOkotoks : ℝ := 26

def finalEmployeesTotal (t : ℝ) := finalEmployeesMooseJaw t + finalEmployeesOkotoks

def netChangeInEmployees (t : ℝ) := finalEmployeesTotal t - initialEmployees t

theorem CEMC_additional_employees (t : ℝ) (h : t = 120) : 
    netChangeInEmployees t = 16 := 
by
    sorry

end CEMC_additional_employees_l799_79927


namespace range_of_m_l799_79947

theorem range_of_m (m : ℝ) :
  (∀ x: ℝ, |x| + |x - 1| > m) ∨ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y) 
  → ¬ ((∀ x: ℝ, |x| + |x - 1| > m) ∧ (∀ x y, x < y → (5 - 2 * m)^x ≤ (5 - 2 * m)^y)) 
  ↔ (1 ≤ m ∧ m < 2) :=
by
  sorry

end range_of_m_l799_79947


namespace find_a9_l799_79999

variable {a : ℕ → ℤ}  -- Define a as a sequence of integers
variable (d : ℤ) (a3 : ℤ) (a4 : ℤ)

-- Define the specific conditions given in the problem
def arithmetic_sequence_condition (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ) : Prop :=
  a 3 + a 4 = 12 ∧ d = 2

-- Define the arithmetic sequence relation
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n + d

-- Statement to prove
theorem find_a9 
  (a : ℕ → ℤ) (d : ℤ) (a3 a4 : ℤ)
  (h1 : arithmetic_sequence_condition a d a3 a4)
  (h2 : arithmetic_sequence a d) :
  a 9 = 17 :=
sorry

end find_a9_l799_79999


namespace sequence_conditions_general_formulas_sum_of_first_n_terms_l799_79936

noncomputable def arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a_n n = a_n 1 + d * (n - 1)

noncomputable def geometric_sequence (b_n : ℕ → ℤ) : Prop :=
  ∃ q : ℤ, q > 0 ∧ ∀ n : ℕ, b_n (n + 1) = b_n n * q

variables {a_n b_n c_n : ℕ → ℤ}
variables (d q : ℤ) (d_pos : 0 < d) (hq : q > 0)
variables (S_n : ℕ → ℤ)

axiom initial_conditions : a_n 1 = 2 ∧ b_n 1 = 2 ∧ a_n 3 = 8 ∧ b_n 3 = 8

theorem sequence_conditions : arithmetic_sequence a_n ∧ geometric_sequence b_n := sorry

theorem general_formulas :
  (∀ n : ℕ, a_n n = 3 * n - 1) ∧
  (∀ n : ℕ, b_n n = 2^n) := sorry

theorem sum_of_first_n_terms :
  (∀ n : ℕ, S_n n = 3 * 2^(n+1) - n - 6) := sorry

end sequence_conditions_general_formulas_sum_of_first_n_terms_l799_79936


namespace n_energetic_all_n_specific_energetic_constraints_l799_79928

-- Proof Problem 1
theorem n_energetic_all_n (a b c : ℕ) (n : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : Nat.gcd a (Nat.gcd b c) = 1) 
(h4 : ∀ n ≥ 1, (a^n + b^n + c^n) % (a + b + c) = 0) :
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 1 ∧ b = 1 ∧ c = 4) := sorry

-- Proof Problem 2
theorem specific_energetic_constraints (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) 
(h3 : Nat.gcd a (Nat.gcd b c) = 1) 
(h4 : (a^2004 + b^2004 + c^2004) % (a + b + c) = 0)
(h5 : (a^2005 + b^2005 + c^2005) % (a + b + c) = 0) 
(h6 : (a^2007 + b^2007 + c^2007) % (a + b + c) ≠ 0) :
  false := sorry

end n_energetic_all_n_specific_energetic_constraints_l799_79928


namespace find_a_values_l799_79952

def setA : Set ℝ := {-1, 1/2, 1}
def setB (a : ℝ) : Set ℝ := {x | a * x^2 = 1 ∧ a ≥ 0}

def full_food (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_food (A B : Set ℝ) : Prop := (∃ x, x ∈ A ∧ x ∈ B) ∧ ¬(A ⊆ B ∨ B ⊆ A)

theorem find_a_values :
  ∀ a : ℝ, full_food setA (setB a) ∨ partial_food setA (setB a) ↔ a = 0 ∨ a = 1 ∨ a = 4 := 
by
  sorry

end find_a_values_l799_79952


namespace slope_of_line_l799_79987

theorem slope_of_line : ∀ (x y : ℝ), 4 * y = -6 * x + 12 → ∃ m b : ℝ, y = m * x + b ∧ m = -3 / 2 :=
by 
sorry

end slope_of_line_l799_79987


namespace Ben_win_probability_l799_79944

theorem Ben_win_probability (lose_prob : ℚ) (no_tie : ¬ ∃ (p : ℚ), p ≠ lose_prob ∧ p + lose_prob = 1) 
  (h : lose_prob = 5/8) : (1 - lose_prob) = 3/8 := by
  sorry

end Ben_win_probability_l799_79944


namespace solve_for_x_l799_79993

theorem solve_for_x (x : ℝ) (h1 : 1 - x^2 = 0) (h2 : x ≠ 1) : x = -1 := 
by 
  sorry

end solve_for_x_l799_79993


namespace seven_digit_numbers_count_l799_79916

/-- Given a six-digit phone number represented by six digits A, B, C, D, E, F:
- There are 7 positions where a new digit can be inserted: before A, between each pair of consecutive digits, and after F.
- Each of these positions can be occupied by any of the 10 digits (0 through 9).
The number of seven-digit numbers that can be formed by adding one digit to the six-digit phone number is 70. -/
theorem seven_digit_numbers_count (A B C D E F : ℕ) (hA : 0 ≤ A ∧ A < 10) (hB : 0 ≤ B ∧ B < 10) 
  (hC : 0 ≤ C ∧ C < 10) (hD : 0 ≤ D ∧ D < 10) (hE : 0 ≤ E ∧ E < 10) (hF : 0 ≤ F ∧ F < 10) : 
  ∃ n : ℕ, n = 70 :=
sorry

end seven_digit_numbers_count_l799_79916


namespace combined_mixture_nuts_l799_79922

def sue_percentage_nuts : ℝ := 0.30
def sue_percentage_dried_fruit : ℝ := 0.70

def jane_percentage_nuts : ℝ := 0.60
def combined_percentage_dried_fruit : ℝ := 0.35

theorem combined_mixture_nuts :
  let sue_contribution := 100.0
  let jane_contribution := 100.0
  let sue_nuts := sue_contribution * sue_percentage_nuts
  let jane_nuts := jane_contribution * jane_percentage_nuts
  let combined_nuts := sue_nuts + jane_nuts
  let total_weight := sue_contribution + jane_contribution
  (combined_nuts / total_weight) * 100 = 45 :=
by
  sorry

end combined_mixture_nuts_l799_79922


namespace eccentricity_of_ellipse_l799_79921

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem eccentricity_of_ellipse :
  let P := (2, 3)
  let F1 := (-2, 0)
  let F2 := (2, 0)
  let d1 := distance P F1
  let d2 := distance P F2
  let a := (d1 + d2) / 2
  let c := distance F1 F2 / 2
  let e := c / a
  e = 1 / 2 := 
by 
  sorry

end eccentricity_of_ellipse_l799_79921


namespace smallest_N_exists_l799_79925

theorem smallest_N_exists (c1 c2 c3 c4 c5 c6 : ℕ) (N : ℕ) :
  (c1 = 6 * c3 - 2) →
  (N + c2 = 6 * c1 - 5) →
  (2 * N + c3 = 6 * c5 - 2) →
  (3 * N + c4 = 6 * c6 - 2) →
  (4 * N + c5 = 6 * c4 - 1) →
  (5 * N + c6 = 6 * c2 - 5) →
  N = 75 :=
by sorry

end smallest_N_exists_l799_79925


namespace value_of_x_plus_y_l799_79957

variable {x y : ℝ}

theorem value_of_x_plus_y (h1 : 1 / x + 1 / y = 1) (h2 : 1 / x - 1 / y = 9) : x + y = -1 / 20 := 
sorry

end value_of_x_plus_y_l799_79957


namespace triangle_range_condition_l799_79946

def triangle_side_range (x : ℝ) : Prop :=
  (1 < x) ∧ (x < 17)

theorem triangle_range_condition (x : ℝ) (a : ℝ) (b : ℝ) :
  (a = 8) → (b = 9) → triangle_side_range x :=
by
  intros h1 h2
  dsimp [triangle_side_range]
  sorry

end triangle_range_condition_l799_79946


namespace bound_c_n_l799_79971

theorem bound_c_n (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) :
  (a 1 = 4) →
  (∀ n, a (n + 1) = a n * (a n - 1)) →
  (∀ n, 2^b n = a n) →
  (∀ n, 2^(n - c n) = b n) →
  ∃ (m M : ℝ), (m = 0) ∧ (M = 1) ∧ ∀ n > 0, m ≤ c n ∧ c n ≤ M :=
by
  intro h1 h2 h3 h4
  use 0
  use 1
  sorry

end bound_c_n_l799_79971


namespace area_of_MNFK_l799_79912

theorem area_of_MNFK (ABNF CMKD MNFK : ℝ) (BN : ℝ) (KD : ℝ) (ABMK : ℝ) (CDFN : ℝ)
  (h1 : BN = 8) (h2 : KD = 9) (h3 : ABMK = 25) (h4 : CDFN = 32) :
  MNFK = 31 :=
by
  have hx : 8 * (MNFK + 25) - 25 = 9 * (MNFK + 32) - 32 := sorry
  exact sorry

end area_of_MNFK_l799_79912


namespace total_heartbeats_during_race_l799_79942

namespace Heartbeats

def avg_heart_beats_per_minute : ℕ := 160
def pace_minutes_per_mile : ℕ := 6
def race_distance_miles : ℕ := 20

theorem total_heartbeats_during_race :
  (race_distance_miles * pace_minutes_per_mile * avg_heart_beats_per_minute = 19200) :=
by
  sorry

end Heartbeats

end total_heartbeats_during_race_l799_79942


namespace sequence_period_16_l799_79978

theorem sequence_period_16 (a : ℝ) (h : a > 0) 
  (u : ℕ → ℝ) (h1 : u 1 = a) (h2 : ∀ n, u (n + 1) = -1 / (u n + 1)) : 
  u 16 = a :=
sorry

end sequence_period_16_l799_79978


namespace amy_total_score_correct_l799_79990

def amyTotalScore (points_per_treasure : ℕ) (treasures_first_level : ℕ) (treasures_second_level : ℕ) : ℕ :=
  (points_per_treasure * treasures_first_level) + (points_per_treasure * treasures_second_level)

theorem amy_total_score_correct:
  amyTotalScore 4 6 2 = 32 :=
by
  -- Proof goes here
  sorry

end amy_total_score_correct_l799_79990


namespace f_of_f_eq_f_l799_79923

def f (x : ℝ) : ℝ := x^2 - 5 * x

theorem f_of_f_eq_f (x : ℝ) : f (f x) = f x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 :=
by
  sorry

end f_of_f_eq_f_l799_79923


namespace market_survey_l799_79929

theorem market_survey (X Y : ℕ) (h1 : X / Y = 9) (h2 : X + Y = 400) : X = 360 :=
by
  sorry

end market_survey_l799_79929


namespace prove_expression_l799_79904

def given_expression : ℤ := -4 + 6 / (-2)

theorem prove_expression : given_expression = -7 := 
by 
  -- insert proof here
  sorry

end prove_expression_l799_79904


namespace cricketer_stats_l799_79902

theorem cricketer_stats :
  let total_runs := 225
  let total_balls := 120
  let boundaries := 4 * 15
  let sixes := 6 * 8
  let twos := 2 * 3
  let singles := 1 * 10
  let perc_boundaries := (boundaries / total_runs.toFloat) * 100
  let perc_sixes := (sixes / total_runs.toFloat) * 100
  let perc_twos := (twos / total_runs.toFloat) * 100
  let perc_singles := (singles / total_runs.toFloat) * 100
  let strike_rate := (total_runs.toFloat / total_balls.toFloat) * 100
  perc_boundaries = 26.67 ∧
  perc_sixes = 21.33 ∧
  perc_twos = 2.67 ∧
  perc_singles = 4.44 ∧
  strike_rate = 187.5 :=
by
  sorry

end cricketer_stats_l799_79902


namespace find_n_value_l799_79965

theorem find_n_value : 
  ∃ (n : ℕ), ∀ (a b c : ℕ), 
    a + b + c = 200 ∧ 
    (∃ bc ca ab : ℕ, bc = b * c ∧ ca = c * a ∧ ab = a * b ∧ n = bc ∧ n = ca ∧ n = ab) → 
    n = 199 := sorry

end find_n_value_l799_79965


namespace book_selling_price_l799_79974

def cost_price : ℕ := 225
def profit_percentage : ℚ := 0.20
def selling_price := cost_price + (profit_percentage * cost_price)

theorem book_selling_price :
  selling_price = 270 :=
by
  sorry

end book_selling_price_l799_79974


namespace remainder_of_p_l799_79975

theorem remainder_of_p (p : ℤ) (h1 : p = 35 * 17 + 10) : p % 35 = 10 := 
  sorry

end remainder_of_p_l799_79975


namespace john_games_l799_79963

variables (G_f G_g B G G_t : ℕ)

theorem john_games (h1: G_f = 21) (h2: B = 23) (h3: G = 6) 
(h4: G_t = G_f + G_g) (h5: G + B = G_t) : G_g = 8 :=
by sorry

end john_games_l799_79963


namespace arrangement_plans_l799_79970

-- Definition of the problem conditions
def numChineseTeachers : ℕ := 2
def numMathTeachers : ℕ := 4
def numTeachersPerSchool : ℕ := 3

-- Definition of the problem statement
theorem arrangement_plans
  (c : ℕ) (m : ℕ) (s : ℕ)
  (h1 : numChineseTeachers = c)
  (h2 : numMathTeachers = m)
  (h3 : numTeachersPerSchool = s)
  (h4 : ∀ a b : ℕ, a + b = numChineseTeachers → a = 1 ∧ b = 1)
  (h5 : ∀ a b : ℕ, a + b = numMathTeachers → a = 2 ∧ b = 2) :
  (c * (1 / 2 * m * (m - 1) / 2)) = 12 :=
sorry

end arrangement_plans_l799_79970


namespace large_square_area_l799_79903

theorem large_square_area (l w : ℕ) (h1 : 2 * (l + w) = 28) : (l + w) * (l + w) = 196 :=
by {
  sorry
}

end large_square_area_l799_79903


namespace g_neither_even_nor_odd_l799_79917

noncomputable def g (x : ℝ) : ℝ := 1 / (3^x - 2) + 1/3

theorem g_neither_even_nor_odd : ¬(∀ x, g x = g (-x)) ∧ ¬(∀ x, g x = -g (-x)) :=
by
  -- insert proof here
  sorry

end g_neither_even_nor_odd_l799_79917


namespace kids_tubing_and_rafting_l799_79951

theorem kids_tubing_and_rafting 
  (total_kids : ℕ) 
  (one_fourth_tubing : ℕ)
  (half_rafting : ℕ)
  (h1 : total_kids = 40)
  (h2 : one_fourth_tubing = total_kids / 4)
  (h3 : half_rafting = one_fourth_tubing / 2) :
  half_rafting = 5 :=
by
  sorry

end kids_tubing_and_rafting_l799_79951


namespace squirrel_acorns_l799_79984

theorem squirrel_acorns :
  ∀ (total_acorns : ℕ)
    (first_month_percent second_month_percent third_month_percent : ℝ)
    (first_month_consumed second_month_consumed third_month_consumed : ℝ),
    total_acorns = 500 →
    first_month_percent = 0.40 →
    second_month_percent = 0.30 →
    third_month_percent = 0.30 →
    first_month_consumed = 0.20 →
    second_month_consumed = 0.25 →
    third_month_consumed = 0.15 →
    let first_month_acorns := total_acorns * first_month_percent
    let second_month_acorns := total_acorns * second_month_percent
    let third_month_acorns := total_acorns * third_month_percent
    let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
    let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
    let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
    remaining_first_month + remaining_second_month + remaining_third_month = 400 := 
by
  intros 
    total_acorns
    first_month_percent second_month_percent third_month_percent
    first_month_consumed second_month_consumed third_month_consumed
    h_total
    h_first_percent
    h_second_percent
    h_third_percent
    h_first_consumed
    h_second_consumed
    h_third_consumed
  let first_month_acorns := total_acorns * first_month_percent
  let second_month_acorns := total_acorns * second_month_percent
  let third_month_acorns := total_acorns * third_month_percent
  let remaining_first_month := first_month_acorns - (first_month_consumed * first_month_acorns)
  let remaining_second_month := second_month_acorns - (second_month_consumed * second_month_acorns)
  let remaining_third_month := third_month_acorns - (third_month_consumed * third_month_acorns)
  sorry

end squirrel_acorns_l799_79984


namespace cos_pi_over_6_minus_2alpha_l799_79926

open Real

noncomputable def tan_plus_pi_over_6 (α : ℝ) := tan (α + π / 6) = 2

theorem cos_pi_over_6_minus_2alpha (α : ℝ) 
  (h1 : π < α ∧ α < 2 * π) 
  (h2 : tan_plus_pi_over_6 α) : 
  cos (π / 6 - 2 * α) = 4 / 5 :=
sorry

end cos_pi_over_6_minus_2alpha_l799_79926


namespace inequality_proof_l799_79958

-- Define the main theorem to be proven.
theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 * (b + c - a) + b^2 * (a + c - b) + c^2 * (a + b - c) ≤ 3 * a * b * c :=
sorry

end inequality_proof_l799_79958


namespace cans_per_bag_l799_79976

theorem cans_per_bag (bags_on_Saturday bags_on_Sunday total_cans : ℕ) (h_saturday : bags_on_Saturday = 3) (h_sunday : bags_on_Sunday = 4) (h_total : total_cans = 63) :
  (total_cans / (bags_on_Saturday + bags_on_Sunday) = 9) :=
by {
  sorry
}

end cans_per_bag_l799_79976


namespace circle_radius_zero_l799_79991

-- Define the given circle equation
def circle_eq (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 6 * y + 13 = 0

-- The proof problem statement
theorem circle_radius_zero : ∀ (x y : ℝ), circle_eq x y → 0 = 0 :=
by
  sorry

end circle_radius_zero_l799_79991


namespace one_thirds_in_nine_halves_l799_79939

theorem one_thirds_in_nine_halves : (9/2) / (1/3) = 27/2 := by
  sorry

end one_thirds_in_nine_halves_l799_79939


namespace poly_at_2_eq_0_l799_79949

def poly (x : ℝ) : ℝ := x^6 - 12 * x^5 + 60 * x^4 - 160 * x^3 + 240 * x^2 - 192 * x + 64

theorem poly_at_2_eq_0 : poly 2 = 0 := by
  sorry

end poly_at_2_eq_0_l799_79949


namespace sum_of_acute_angles_l799_79968

theorem sum_of_acute_angles (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
    (h2 : Real.sin β = 3 * Real.sqrt 10 / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_acute_angles_l799_79968


namespace f_value_at_3_l799_79964

noncomputable def f : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (-x) = -f x

def periodic_shift (f : ℝ → ℝ) :=
  ∀ x : ℝ, f (x + 2) = f x + 2

theorem f_value_at_3 (h_odd : odd_function f) (h_value : f (-1) = 1/2) (h_periodic : periodic_shift f) : 
  f 3 = 3 / 2 := 
sorry

end f_value_at_3_l799_79964


namespace problem_solution_l799_79919

def p1 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def p2 : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_solution : (¬ p1) ∨ (¬ p2) :=
by
  sorry

end problem_solution_l799_79919


namespace total_accepted_cartons_l799_79934

-- Definitions for the number of cartons delivered and damaged for each customer
def cartons_delivered_first_two : Nat := 300
def cartons_delivered_last_three : Nat := 200

def cartons_damaged_first : Nat := 70
def cartons_damaged_second : Nat := 50
def cartons_damaged_third : Nat := 40
def cartons_damaged_fourth : Nat := 30
def cartons_damaged_fifth : Nat := 20

-- Statement to prove
theorem total_accepted_cartons :
  let accepted_first := cartons_delivered_first_two - cartons_damaged_first
  let accepted_second := cartons_delivered_first_two - cartons_damaged_second
  let accepted_third := cartons_delivered_last_three - cartons_damaged_third
  let accepted_fourth := cartons_delivered_last_three - cartons_damaged_fourth
  let accepted_fifth := cartons_delivered_last_three - cartons_damaged_fifth
  accepted_first + accepted_second + accepted_third + accepted_fourth + accepted_fifth = 990 :=
by
  sorry

end total_accepted_cartons_l799_79934


namespace solve_for_p_l799_79994

def cubic_eq_has_natural_roots (p : ℝ) : Prop :=
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  5*(a:ℝ)^3 - 5*(p + 1)*(a:ℝ)^2 + (71*p - 1)*(a:ℝ) + 1 = 66*p ∧
  5*(b:ℝ)^3 - 5*(p + 1)*(b:ℝ)^2 + (71*p - 1)*(b:ℝ) + 1 = 66*p ∧
  5*(c:ℝ)^3 - 5*(p + 1)*(c:ℝ)^2 + (71*p - 1)*(c:ℝ) + 1 = 66*p

theorem solve_for_p : ∀ (p : ℝ), cubic_eq_has_natural_roots p → p = 76 :=
by
  sorry

end solve_for_p_l799_79994


namespace xy_range_l799_79913

open Real

theorem xy_range (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_eq : x + 2 / x + 3 * y + 4 / y = 10) : 
  1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end xy_range_l799_79913


namespace inequality_proof_l799_79907

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
  (h_condition : (a + b + c + d) * (1/a + 1/b + 1/c + 1/d) = 20) :
  (a^2 + b^2 + c^2 + d^2) * (1/(a^2) + 1/(b^2) + 1/(c^2) + 1/(d^2)) ≥ 36 :=
by
  sorry

end inequality_proof_l799_79907


namespace average_jump_difference_l799_79937

-- Define the total jumps and time
def total_jumps_liu_li : ℕ := 480
def total_jumps_zhang_hua : ℕ := 420
def time_minutes : ℕ := 5

-- Define the average jumps per minute
def average_jumps_per_minute (total_jumps : ℕ) (time : ℕ) : ℕ :=
  total_jumps / time

-- State the theorem
theorem average_jump_difference :
  average_jumps_per_minute total_jumps_liu_li time_minutes - 
  average_jumps_per_minute total_jumps_zhang_hua time_minutes = 12 := 
sorry


end average_jump_difference_l799_79937


namespace distinct_solutions_eq_108_l799_79901

theorem distinct_solutions_eq_108 {p q : ℝ} (h1 : (p - 6) * (3 * p + 10) = p^2 - 19 * p + 50)
  (h2 : (q - 6) * (3 * q + 10) = q^2 - 19 * q + 50)
  (h3 : p ≠ q) : (p + 2) * (q + 2) = 108 := 
by
  sorry

end distinct_solutions_eq_108_l799_79901


namespace sum_ratio_arithmetic_sequence_l799_79967

noncomputable def sum_of_arithmetic_sequence (n : ℕ) : ℝ := sorry

theorem sum_ratio_arithmetic_sequence (S : ℕ → ℝ) (hS : ∀ n, S n = sum_of_arithmetic_sequence n)
  (h_cond : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
sorry

end sum_ratio_arithmetic_sequence_l799_79967


namespace mac_runs_faster_than_apple_l799_79966

theorem mac_runs_faster_than_apple :
  let Apple_speed := 3 -- miles per hour
  let Mac_speed := 4 -- miles per hour
  let Distance := 24 -- miles
  let Apple_time := Distance / Apple_speed -- hours
  let Mac_time := Distance / Mac_speed -- hours
  let Time_difference := (Apple_time - Mac_time) * 60 -- converting hours to minutes
  Time_difference = 120 := by
  sorry

end mac_runs_faster_than_apple_l799_79966


namespace product_value_l799_79969

theorem product_value (x : ℝ) (h : (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8)) : (6 + x) * (21 - x) = 1369 / 4 :=
by
  sorry

end product_value_l799_79969


namespace tiles_difference_l799_79995

-- Definitions based on given conditions
def initial_blue_tiles : Nat := 20
def initial_green_tiles : Nat := 10
def first_border_green_tiles : Nat := 24
def second_border_green_tiles : Nat := 36

-- Problem statement
theorem tiles_difference :
  initial_green_tiles + first_border_green_tiles + second_border_green_tiles - initial_blue_tiles = 50 :=
by
  sorry

end tiles_difference_l799_79995


namespace sum_of_first_7_terms_l799_79900

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

theorem sum_of_first_7_terms (h1 : a 2 = 3) (h2 : a 6 = 11)
  (h3 : ∀ n, S n = n * (a 1 + a n) / 2) : S 7 = 49 :=
by 
  sorry

end sum_of_first_7_terms_l799_79900


namespace protective_additive_increase_l799_79979

def percentIncrease (old_val new_val : ℕ) : ℚ :=
  (new_val - old_val) / old_val * 100

theorem protective_additive_increase :
  percentIncrease 45 60 = 33.33 := 
sorry

end protective_additive_increase_l799_79979


namespace f_2019_is_zero_l799_79960

noncomputable def f : ℝ → ℝ := sorry

axiom f_is_non_negative
  (x : ℝ) : 0 ≤ f x

axiom f_satisfies_condition
  (a b c : ℝ) : f (a^3) + f (b^3) + f (c^3) = 3 * f a * f b * f c

axiom f_one_not_one : f 1 ≠ 1

theorem f_2019_is_zero : f 2019 = 0 := 
  sorry

end f_2019_is_zero_l799_79960


namespace max_value_theorem_l799_79935

open Real

noncomputable def max_value (x y : ℝ) : ℝ :=
  x * y * (75 - 5 * x - 3 * y)

theorem max_value_theorem :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y < 75 ∧ max_value x y = 3125 / 3 := by
  sorry

end max_value_theorem_l799_79935


namespace range_b_intersects_ellipse_l799_79940

open Real

noncomputable def line_intersects_ellipse (b : ℝ) : Prop :=
  ∀ θ : ℝ, 0 ≤ θ ∧ θ < π → ∃ x y : ℝ, x = 2 * cos θ ∧ y = 4 * sin θ ∧ y = x + b

theorem range_b_intersects_ellipse :
  ∀ b : ℝ, line_intersects_ellipse b ↔ b ∈ Set.Icc (-2 : ℝ) (2 * sqrt 5) :=
by
  sorry

end range_b_intersects_ellipse_l799_79940


namespace arithmetic_sequence_evaluation_l799_79998

theorem arithmetic_sequence_evaluation :
  25^2 - 23^2 + 21^2 - 19^2 + 17^2 - 15^2 + 13^2 - 11^2 + 9^2 - 7^2 + 5^2 - 3^2 + 1^2 = 337 :=
by 
-- Proof omitted
sorry

end arithmetic_sequence_evaluation_l799_79998


namespace circles_internally_tangent_l799_79996

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x - 6)^2 + y^2 = 1 → 
  (x - 3)^2 + (y - 4)^2 = 36 → 
  true := 
by 
  intros x y h1 h2
  sorry

end circles_internally_tangent_l799_79996


namespace sum_of_three_primes_eq_86_l799_79914

def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem sum_of_three_primes_eq_86 (a b c : ℕ) (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (h_sum : a + b + c = 86) :
  (a, b, c) = (2, 5, 79) ∨ (a, b, c) = (2, 11, 73) ∨ (a, b, c) = (2, 13, 71) ∨ (a, b, c) = (2, 17, 67) ∨
  (a, b, c) = (2, 23, 61) ∨ (a, b, c) = (2, 31, 53) ∨ (a, b, c) = (2, 37, 47) ∨ (a, b, c) = (2, 41, 43) :=
by
  sorry

end sum_of_three_primes_eq_86_l799_79914


namespace survey_blue_percentage_l799_79983

-- Conditions
def red (r : ℕ) := r = 70
def blue (b : ℕ) := b = 80
def green (g : ℕ) := g = 50
def yellow (y : ℕ) := y = 70
def orange (o : ℕ) := o = 30

-- Total responses sum
def total_responses (r b g y o : ℕ) := r + b + g + y + o = 300

-- Percentage of blue respondents
def blue_percentage (b total : ℕ) := (b : ℚ) / total * 100 = 26 + 2/3

-- Theorem statement
theorem survey_blue_percentage (r b g y o : ℕ) (H_red : red r) (H_blue : blue b) (H_green : green g) (H_yellow : yellow y) (H_orange : orange o) (H_total : total_responses r b g y o) : blue_percentage b 300 :=
by {
  sorry
}

end survey_blue_percentage_l799_79983


namespace perfect_square_m_value_l799_79905

theorem perfect_square_m_value (M X : ℤ) (hM : M > 1) (hX_lt_max : X < 8000) (hX_gt_min : 1000 < X) (hX_eq : X = M^3) : 
  (∃ M : ℤ, M > 1 ∧ 1000 < M^3 ∧ M^3 < 8000 ∧ (∃ k : ℤ, X = k * k) ∧ M = 16) :=
by
  use 16
  -- Here, we would normally provide the proof steps to show that 1000 < 16^3 < 8000 and 16^3 is a perfect square
  sorry

end perfect_square_m_value_l799_79905


namespace twelve_percent_greater_l799_79961

theorem twelve_percent_greater :
  ∃ x : ℝ, x = 80 + (12 / 100) * 80 := sorry

end twelve_percent_greater_l799_79961


namespace find_rate_per_kg_of_mangoes_l799_79931

theorem find_rate_per_kg_of_mangoes (r : ℝ) 
  (total_units_paid : ℝ) (grapes_kg : ℝ) (grapes_rate : ℝ)
  (mangoes_kg : ℝ) (total_grapes_cost : ℝ)
  (total_mangoes_cost : ℝ) (total_cost : ℝ) :
  grapes_kg = 8 →
  grapes_rate = 70 →
  mangoes_kg = 10 →
  total_units_paid = 1110 →
  total_grapes_cost = grapes_kg * grapes_rate →
  total_mangoes_cost = total_units_paid - total_grapes_cost →
  r = total_mangoes_cost / mangoes_kg →
  r = 55 := by
  intros
  sorry

end find_rate_per_kg_of_mangoes_l799_79931


namespace find_r_l799_79938

noncomputable def r_value (a b : ℝ) (h : a * b = 3) : ℝ :=
  let r := (a^2 + 1 / b^2) * (b^2 + 1 / a^2)
  r

theorem find_r (a b : ℝ) (h : a * b = 3) : r_value a b h = 100 / 9 := by
  sorry

end find_r_l799_79938


namespace percentage_increase_of_x_l799_79954

theorem percentage_increase_of_x (C x y : ℝ) (P : ℝ) (h1 : x * y = C) (h2 : (x * (1 + P / 100)) * (y * (5 / 6)) = C) :
  P = 20 :=
by
  sorry

end percentage_increase_of_x_l799_79954


namespace arithmetic_sequence_sum_l799_79973

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (S_10_eq : S 10 = 20) (S_20_eq : S 20 = 15) :
  S 30 = -15 :=
by
  sorry

end arithmetic_sequence_sum_l799_79973


namespace difference_of_squares_l799_79911

theorem difference_of_squares (n : ℤ) : 4 - n^2 = (2 + n) * (2 - n) := 
by
  -- Proof goes here
  sorry

end difference_of_squares_l799_79911


namespace largest_volume_sold_in_august_is_21_l799_79943

def volumes : List ℕ := [13, 15, 16, 17, 19, 21]

theorem largest_volume_sold_in_august_is_21
  (sold_volumes_august : List ℕ)
  (sold_volumes_september : List ℕ) :
  sold_volumes_august.length = 3 ∧
  sold_volumes_september.length = 2 ∧
  2 * (sold_volumes_september.sum) = sold_volumes_august.sum ∧
  (sold_volumes_august ++ sold_volumes_september).sum = volumes.sum →
  21 ∈ sold_volumes_august :=
sorry

end largest_volume_sold_in_august_is_21_l799_79943


namespace shekar_biology_marks_l799_79962

theorem shekar_biology_marks (M S SS E A n B : ℕ) 
  (hM : M = 76)
  (hS : S = 65)
  (hSS : SS = 82)
  (hE : E = 67)
  (hA : A = 73)
  (hn : n = 5)
  (hA_eq : A = (M + S + SS + E + B) / n) : 
  B = 75 := 
by
  rw [hM, hS, hSS, hE, hn, hA] at hA_eq
  sorry

end shekar_biology_marks_l799_79962


namespace factorize_expression_l799_79972

theorem factorize_expression (a : ℝ) : a^2 + 5 * a = a * (a + 5) :=
sorry

end factorize_expression_l799_79972


namespace students_in_second_class_l799_79948

theorem students_in_second_class 
    (avg1 : ℝ)
    (n1 : ℕ)
    (avg2 : ℝ)
    (total_avg : ℝ)
    (x : ℕ)
    (h1 : avg1 = 40)
    (h2 : n1 = 26)
    (h3 : avg2 = 60)
    (h4 : total_avg = 53.1578947368421)
    (h5 : (n1 * avg1 + x * avg2) / (n1 + x) = total_avg) :
  x = 50 :=
by
  sorry

end students_in_second_class_l799_79948


namespace marbles_lost_l799_79953

def initial_marbles := 8
def current_marbles := 6

theorem marbles_lost : initial_marbles - current_marbles = 2 :=
by
  sorry

end marbles_lost_l799_79953


namespace school_count_l799_79977

theorem school_count (n : ℕ) (h1 : 2 * n - 1 = 69) (h2 : n < 76) (h3 : n > 29) : (2 * n - 1) / 3 = 23 :=
by
  sorry

end school_count_l799_79977


namespace rate_percent_l799_79945

noncomputable def calculate_rate (P: ℝ) : ℝ :=
  let I : ℝ := 320
  let t : ℝ := 2
  I * 100 / (P * t)

theorem rate_percent (P: ℝ) (hP: P > 0) : calculate_rate P = 4 := 
by
  sorry

end rate_percent_l799_79945


namespace length_AM_is_correct_l799_79909

-- Definitions of the problem conditions
def length_of_square : ℝ := 9

def ratio_AP_PB : ℝ × ℝ := (7, 2)

def radius_of_quarter_circle : ℝ := 9

-- The theorem to prove
theorem length_AM_is_correct
  (AP PB PE : ℝ)
  (x : ℝ)
  (AM : ℝ) 
  (H_AP_PB  : AP = 7 ∧ PB = 2 ∧ PE = 2)
  (H_QD_QE : x = 63 / 11)
  (H_PQ : PQ = 2 + x) :
  AM = 85 / 22 :=
by
  sorry

end length_AM_is_correct_l799_79909


namespace largest_n_l799_79997

noncomputable def is_multiple_of_seven (n : ℕ) : Prop :=
  (6 * (n-3)^3 - n^2 + 10 * n - 15) % 7 = 0

theorem largest_n (n : ℕ) : n < 50000 ∧ is_multiple_of_seven n → n = 49999 :=
by sorry

end largest_n_l799_79997


namespace inequality_abc_l799_79910

theorem inequality_abc (a b c d : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  (a + b + c + d + 1)^2 ≥ 4 * (a^2 + b^2 + c^2 + d^2) :=
by
  -- Proof goes here
  sorry

end inequality_abc_l799_79910


namespace fans_received_all_items_l799_79941

def multiple_of (n m : ℕ) : Prop := ∃ k : ℕ, m = n * k

theorem fans_received_all_items :
  (∀ n, multiple_of 100 n → multiple_of 40 n ∧ multiple_of 60 n ∧ multiple_of 24 n ∧ n ≤ 7200 → ∃ k, n = 600 * k) →
  (∃ k : ℕ, 7200 / 600 = k ∧ k = 12) :=
by
  sorry

end fans_received_all_items_l799_79941


namespace quadratic_inequality_solution_l799_79924

theorem quadratic_inequality_solution :
  ∀ x : ℝ, -1/3 < x ∧ x < 1 → -3 * x^2 + 8 * x + 1 < 0 :=
by
  intro x
  intro h
  sorry

end quadratic_inequality_solution_l799_79924


namespace find_g3_l799_79950

noncomputable def g : ℝ → ℝ := sorry

theorem find_g3 (h : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = x) : g 3 = 1 :=
sorry

end find_g3_l799_79950


namespace problem_statement_l799_79920

theorem problem_statement :
  let pct := 208 / 100
  let initial_value := 1265
  let step1 := pct * initial_value
  let step2 := step1 ^ 2
  let answer := step2 / 12
  answer = 576857.87 := 
by 
  sorry

end problem_statement_l799_79920


namespace sum_integers_neg40_to_60_l799_79982

theorem sum_integers_neg40_to_60 : (Finset.range (60 + 41)).sum (fun i => i - 40) = 1010 := by
  sorry

end sum_integers_neg40_to_60_l799_79982


namespace consecutive_odd_numbers_first_l799_79980

theorem consecutive_odd_numbers_first :
  ∃ x : ℤ, 11 * x = 3 * (x + 4) + 4 * (x + 2) + 16 ∧ x = 9 :=
by 
  sorry

end consecutive_odd_numbers_first_l799_79980


namespace combined_list_correct_l799_79932

def james_friends : ℕ := 75
def john_friends : ℕ := 3 * james_friends
def shared_friends : ℕ := 25
def combined_list : ℕ := james_friends + john_friends - shared_friends

theorem combined_list_correct :
  combined_list = 275 :=
by
  sorry

end combined_list_correct_l799_79932


namespace min_value_of_3x_plus_4y_is_5_l799_79992

theorem min_value_of_3x_plus_4y_is_5 :
  ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → (∃ (b : ℝ), b = 3 * x + 4 * y ∧ ∀ (x y : ℝ), 0 < x → 0 < y → (3 / x + 1 / y = 5) → 3 * x + 4 * y ≥ b) :=
by
  intro x y x_pos y_pos h_eq
  let b := 5
  use b
  simp [b]
  sorry

end min_value_of_3x_plus_4y_is_5_l799_79992


namespace cos_neg_two_pi_over_three_eq_l799_79988

noncomputable def cos_neg_two_pi_over_three : ℝ := -2 * Real.pi / 3

theorem cos_neg_two_pi_over_three_eq :
  Real.cos cos_neg_two_pi_over_three = -1 / 2 :=
sorry

end cos_neg_two_pi_over_three_eq_l799_79988


namespace arithmetic_sequence_a9_l799_79981

theorem arithmetic_sequence_a9 (S : ℕ → ℤ) (a : ℕ → ℤ) :
  S 8 = 4 * a 3 → a 7 = -2 → a 9 = -6 := by
  sorry

end arithmetic_sequence_a9_l799_79981


namespace perpendicular_line_equation_l799_79985

theorem perpendicular_line_equation 
  (p : ℝ × ℝ)
  (L1 : ℝ → ℝ → Prop)
  (L2 : ℝ → ℝ → ℝ → Prop) 
  (hx : p = (1, -1)) 
  (hL1 : ∀ x y, L1 x y ↔ 3 * x - 2 * y = 0) 
  (hL2 : ∀ x y m, L2 x y m ↔ 2 * x + 3 * y + m = 0) :
  ∃ m : ℝ, L2 (p.1) (p.2) m ∧ 2 * p.1 + 3 * p.2 + m = 0 :=
by
  sorry

end perpendicular_line_equation_l799_79985


namespace symmetric_point_xoz_plane_l799_79930

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_xoz (M : Point3D) : Point3D :=
  ⟨M.x, -M.y, M.z⟩

theorem symmetric_point_xoz_plane :
  let M := Point3D.mk 5 1 (-2)
  symmetric_xoz M = Point3D.mk 5 (-1) (-2) :=
by
  sorry

end symmetric_point_xoz_plane_l799_79930


namespace seq_expression_l799_79906

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := n^2 * a n

theorem seq_expression (a : ℕ → ℝ) (h₁ : a 1 = 2) (h₂ : ∀ n ≥ 1, S n a = n^2 * a n) :
  ∀ n ≥ 1, a n = 4 / (n * (n + 1)) :=
by
  sorry

end seq_expression_l799_79906


namespace liquid_left_after_evaporation_l799_79956

-- Definitions
def solution_y (total_mass : ℝ) : ℝ × ℝ :=
  (0.30 * total_mass, 0.70 * total_mass) -- liquid_x, water

def evaporate_water (initial_water : ℝ) (evaporated_mass : ℝ) : ℝ :=
  initial_water - evaporated_mass

-- Condition that new solution is 45% liquid x
theorem liquid_left_after_evaporation 
  (initial_mass : ℝ) 
  (evaporated_mass : ℝ)
  (added_mass : ℝ)
  (new_percentage_liquid_x : ℝ) :
  initial_mass = 8 → 
  evaporated_mass = 4 → 
  added_mass = 4 →
  new_percentage_liquid_x = 0.45 →
  solution_y initial_mass = (2.4, 5.6) →
  evaporate_water 5.6 evaporated_mass = 1.6 →
  solution_y added_mass = (1.2, 2.8) →
  2.4 + 1.2 = 3.6 →
  1.6 + 2.8 = 4.4 →
  0.45 * (3.6 + 4.4) = 3.6 →
  4 = 2.4 + 1.6 := sorry

end liquid_left_after_evaporation_l799_79956


namespace cylindrical_to_rectangular_l799_79989

theorem cylindrical_to_rectangular (r θ z : ℝ) (h1 : r = 6) (h2 : θ = π / 3) (h3 : z = 2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (3, 3 * Real.sqrt 3, 2) := 
by 
  rw [h1, h2, h3]
  sorry

end cylindrical_to_rectangular_l799_79989


namespace multiply_exponents_l799_79933

theorem multiply_exponents (a : ℝ) : (6 * a^2) * (1/2 * a^3) = 3 * a^5 := by
  sorry

end multiply_exponents_l799_79933


namespace derivative_at_one_l799_79918

section

variable {f : ℝ → ℝ}

-- Define the condition
def limit_condition (f : ℝ → ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ Δx, abs Δx < δ → abs ((f (1 + Δx) - f (1 - Δx)) / Δx + 6) < ε

-- State the main theorem
theorem derivative_at_one (h : limit_condition f) : deriv f 1 = -3 :=
by
  sorry

end

end derivative_at_one_l799_79918


namespace painter_red_cells_count_l799_79955

open Nat

/-- Prove the number of red cells painted by the painter in the given 2000 x 70 grid. -/
theorem painter_red_cells_count :
  let rows := 2000
  let columns := 70
  let lcm_rc := Nat.lcm rows columns -- Calculate the LCM of row and column counts
  lcm_rc = 14000 := by
sorry

end painter_red_cells_count_l799_79955


namespace perfect_square_trinomial_l799_79908

noncomputable def p (k : ℝ) (x : ℝ) : ℝ :=
  4 * x^2 + 2 * k * x + 9

theorem perfect_square_trinomial (k : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, p k x = (2 * x + b)^2) → (k = 6 ∨ k = -6) :=
by 
  intro h
  sorry

end perfect_square_trinomial_l799_79908


namespace frank_has_3_cookies_l799_79959

-- The definitions and conditions based on the problem statement
def num_cookies_millie : ℕ := 4
def num_cookies_mike : ℕ := 3 * num_cookies_millie
def num_cookies_frank : ℕ := (num_cookies_mike / 2) - 3

-- The theorem stating the question and the correct answer
theorem frank_has_3_cookies : num_cookies_frank = 3 :=
by 
  -- This is where the proof steps would go, but for now we use sorry
  sorry

end frank_has_3_cookies_l799_79959


namespace number_of_diagonals_in_decagon_l799_79986

-- Definition of the problem condition: a polygon with n = 10 sides
def n : ℕ := 10

-- Theorem stating the number of diagonals in a regular decagon
theorem number_of_diagonals_in_decagon : (n * (n - 3)) / 2 = 35 :=
by
  -- Proof steps will go here
  sorry

end number_of_diagonals_in_decagon_l799_79986


namespace minimum_balls_to_draw_l799_79915

theorem minimum_balls_to_draw
  (red green yellow blue white : ℕ)
  (h_red : red = 30)
  (h_green : green = 25)
  (h_yellow : yellow = 20)
  (h_blue : blue = 15)
  (h_white : white = 10) :
  ∃ (n : ℕ), n = 81 ∧
    (∀ (r g y b w : ℕ), 
       (r + g + y + b + w >= n) →
       ((r ≥ 20 ∨ g ≥ 20 ∨ y ≥ 20 ∨ b ≥ 20 ∨ w ≥ 20) ∧ 
        (r ≥ 10 ∨ g ≥ 10 ∨ y ≥ 10 ∨ b ≥ 10 ∨ w ≥ 10))
    ) := sorry

end minimum_balls_to_draw_l799_79915
