import Mathlib

namespace NUMINAMATH_GPT_circles_intersect_at_2_points_l729_72917

theorem circles_intersect_at_2_points :
  let circle1 := { p : ℝ × ℝ | (p.1 - 5 / 2) ^ 2 + p.2 ^ 2 = 25 / 4 }
  let circle2 := { p : ℝ × ℝ | p.1 ^ 2 + (p.2 - 7 / 2) ^ 2 = 49 / 4 }
  ∃ (P1 P2 : ℝ × ℝ), P1 ∈ circle1 ∧ P1 ∈ circle2 ∧
                     P2 ∈ circle1 ∧ P2 ∈ circle2 ∧
                     P1 ≠ P2 ∧ ∀ (P : ℝ × ℝ), P ∈ circle1 ∧ P ∈ circle2 → P = P1 ∨ P = P2 := 
by 
  sorry

end NUMINAMATH_GPT_circles_intersect_at_2_points_l729_72917


namespace NUMINAMATH_GPT_largest_prime_divisor_in_range_l729_72961

theorem largest_prime_divisor_in_range (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.floor (Real.sqrt n) ∧ 
  (∀ q, Prime q ∧ q ≤ Int.floor (Real.sqrt n) → q ≤ p) :=
sorry

end NUMINAMATH_GPT_largest_prime_divisor_in_range_l729_72961


namespace NUMINAMATH_GPT_solve_for_x_l729_72900

theorem solve_for_x (x : ℝ) : 2 * x + 3 * x + 4 * x = 12 + 9 + 6 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l729_72900


namespace NUMINAMATH_GPT_mean_median_difference_is_minus_4_l729_72923

-- Defining the percentages of students scoring specific points
def perc_60 : ℝ := 0.20
def perc_75 : ℝ := 0.55
def perc_95 : ℝ := 0.10
def perc_110 : ℝ := 1 - (perc_60 + perc_75 + perc_95) -- 0.15

-- Defining the scores
def score_60 : ℝ := 60
def score_75 : ℝ := 75
def score_95 : ℝ := 95
def score_110 : ℝ := 110

-- Calculating the mean score
def mean_score : ℝ := (perc_60 * score_60) + (perc_75 * score_75) + (perc_95 * score_95) + (perc_110 * score_110)

-- Given the median score
def median_score : ℝ := score_75

-- Defining the expected difference
def expected_difference : ℝ := mean_score - median_score

theorem mean_median_difference_is_minus_4 :
  expected_difference = -4 := by sorry

end NUMINAMATH_GPT_mean_median_difference_is_minus_4_l729_72923


namespace NUMINAMATH_GPT_cookie_count_per_box_l729_72904

theorem cookie_count_per_box (A B C T: ℝ) (H1: A = 2) (H2: B = 0.75) (H3: C = 3) (H4: T = 276) :
  T / (A + B + C) = 48 :=
by
  sorry

end NUMINAMATH_GPT_cookie_count_per_box_l729_72904


namespace NUMINAMATH_GPT_no_combination_of_three_coins_sums_to_52_cents_l729_72990

def is_valid_coin (c : ℕ) : Prop :=
  c = 5 ∨ c = 10 ∨ c = 25 ∨ c = 50 ∨ c = 100

theorem no_combination_of_three_coins_sums_to_52_cents :
  ¬ ∃ a b c : ℕ, is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ a + b + c = 52 :=
by 
  sorry

end NUMINAMATH_GPT_no_combination_of_three_coins_sums_to_52_cents_l729_72990


namespace NUMINAMATH_GPT_cyclists_speeds_product_l729_72911

theorem cyclists_speeds_product (u v : ℝ) (hu : u > 0) (hv : v > 0)
  (h₁ : 6 / u = 6 / v + 1 / 12) 
  (h₂ : v / 3 = u / 3 + 4) : 
  u * v = 864 := 
by
  sorry

end NUMINAMATH_GPT_cyclists_speeds_product_l729_72911


namespace NUMINAMATH_GPT_empty_solution_set_l729_72930

theorem empty_solution_set 
  (x : ℝ) 
  (h : -2 + 3 * x - 2 * x^2 > 0) : 
  false :=
by
  -- Discriminant calculation to prove empty solution set
  let delta : ℝ := 9 - 4 * 2 * 2
  have h_delta : delta < 0 := by norm_num
  sorry

end NUMINAMATH_GPT_empty_solution_set_l729_72930


namespace NUMINAMATH_GPT_miles_traveled_total_l729_72970

-- Define the initial distance and the additional distance
def initial_distance : ℝ := 212.3
def additional_distance : ℝ := 372.0

-- Define the total distance as the sum of the initial and additional distances
def total_distance : ℝ := initial_distance + additional_distance

-- Prove that the total distance is 584.3 miles
theorem miles_traveled_total : total_distance = 584.3 := by
  sorry

end NUMINAMATH_GPT_miles_traveled_total_l729_72970


namespace NUMINAMATH_GPT_correct_email_sequence_l729_72941

theorem correct_email_sequence :
  let a := "Open the mailbox"
  let b := "Enter the recipient's address"
  let c := "Enter the subject"
  let d := "Enter the content of the email"
  let e := "Click 'Compose'"
  let f := "Click 'Send'"
  (a, e, b, c, d, f) = ("Open the mailbox", "Click 'Compose'", "Enter the recipient's address", "Enter the subject", "Enter the content of the email", "Click 'Send'") := 
sorry

end NUMINAMATH_GPT_correct_email_sequence_l729_72941


namespace NUMINAMATH_GPT_incorrect_equation_l729_72976

theorem incorrect_equation (x : ℕ) (h : x + 2 * (12 - x) = 20) : 2 * (12 - x) - 20 ≠ x :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_equation_l729_72976


namespace NUMINAMATH_GPT_quadratic_root_condition_l729_72987

theorem quadratic_root_condition (d : ℝ) :
  (∀ x, x^2 + 7 * x + d = 0 → x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → d = 9.8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_root_condition_l729_72987


namespace NUMINAMATH_GPT_parameterized_line_segment_problem_l729_72901

theorem parameterized_line_segment_problem
  (p q r s : ℝ)
  (hq : q = 1)
  (hs : s = 2)
  (hpq : p + q = 6)
  (hrs : r + s = 9) :
  p^2 + q^2 + r^2 + s^2 = 79 := 
sorry

end NUMINAMATH_GPT_parameterized_line_segment_problem_l729_72901


namespace NUMINAMATH_GPT_determine_m_l729_72951

theorem determine_m (m : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 2 ↔ m * (x - 1) > x^2 - x) → m = 2 :=
sorry

end NUMINAMATH_GPT_determine_m_l729_72951


namespace NUMINAMATH_GPT_remainder_19_pow_19_plus_19_mod_20_l729_72948

theorem remainder_19_pow_19_plus_19_mod_20 : (19 ^ 19 + 19) % 20 = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_remainder_19_pow_19_plus_19_mod_20_l729_72948


namespace NUMINAMATH_GPT_goose_eggs_calculation_l729_72926

noncomputable def goose_eggs_total (E : ℕ) : Prop :=
  let hatched := (2/3) * E
  let survived_first_month := (3/4) * hatched
  let survived_first_year := (2/5) * survived_first_month
  survived_first_year = 110

theorem goose_eggs_calculation :
  goose_eggs_total 3300 :=
by
  have h1 : (2 : ℝ) / (3 : ℝ) ≠ 0 := by norm_num
  have h2 : (3 : ℝ) / (4 : ℝ) ≠ 0 := by norm_num
  have h3 : (2 : ℝ) / (5 : ℝ) ≠ 0 := by norm_num
  sorry

end NUMINAMATH_GPT_goose_eggs_calculation_l729_72926


namespace NUMINAMATH_GPT_triangle_right_angled_l729_72992

theorem triangle_right_angled
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * Real.cos C + c * Real.cos B = a * Real.sin A) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 :=
sorry

end NUMINAMATH_GPT_triangle_right_angled_l729_72992


namespace NUMINAMATH_GPT_unique_solution_real_l729_72979

theorem unique_solution_real {x y : ℝ} (h1 : x * (x + y)^2 = 9) (h2 : x * (y^3 - x^3) = 7) :
  x = 1 ∧ y = 2 :=
sorry

end NUMINAMATH_GPT_unique_solution_real_l729_72979


namespace NUMINAMATH_GPT_sum_of_cubes_div_xyz_l729_72997

-- Given: x, y, z are non-zero real numbers, and x + y + z = 0.
-- Prove: (x^3 + y^3 + z^3) / (xyz) = 3.
theorem sum_of_cubes_div_xyz (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x + y + z = 0) :
  (x^3 + y^3 + z^3) / (x * y * z) = 3 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_div_xyz_l729_72997


namespace NUMINAMATH_GPT_octal_to_base5_conversion_l729_72947

-- Define the octal to decimal conversion
def octalToDecimal (n : ℕ) : ℕ :=
  2 * 8^3 + 0 * 8^2 + 1 * 8^1 + 1 * 8^0

-- Define the base-5 number
def base5Representation : ℕ := 13113

-- Theorem statement
theorem octal_to_base5_conversion :
  octalToDecimal 2011 = base5Representation := 
sorry

end NUMINAMATH_GPT_octal_to_base5_conversion_l729_72947


namespace NUMINAMATH_GPT_horse_revolutions_l729_72962

theorem horse_revolutions (r1 r2  : ℝ) (rev1 rev2 : ℕ)
  (h1 : r1 = 30) (h2 : rev1 = 20) (h3 : r2 = 10) : rev2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_horse_revolutions_l729_72962


namespace NUMINAMATH_GPT_min_value_of_f_l729_72937

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  2 * x^3 - 6 * x^2 + m

theorem min_value_of_f :
  ∀ (m : ℝ),
    f 0 m = 3 →
    ∃ x min, x ∈ Set.Icc (-2:ℝ) (2:ℝ) ∧ min = f x m ∧ min = -37 :=
by
  intros m h
  have h' : f 0 m = 3 := h
  -- Proof omitted.
  sorry

end NUMINAMATH_GPT_min_value_of_f_l729_72937


namespace NUMINAMATH_GPT_compute_expression_l729_72965

theorem compute_expression : 9 + 7 * (5 - Real.sqrt 16)^2 = 16 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l729_72965


namespace NUMINAMATH_GPT_alice_bracelets_given_away_l729_72972

theorem alice_bracelets_given_away
    (total_bracelets : ℕ)
    (cost_of_materials : ℝ)
    (price_per_bracelet : ℝ)
    (profit : ℝ)
    (bracelets_given_away : ℕ)
    (bracelets_sold : ℕ)
    (total_revenue : ℝ)
    (h1 : total_bracelets = 52)
    (h2 : cost_of_materials = 3)
    (h3 : price_per_bracelet = 0.25)
    (h4 : profit = 8)
    (h5 : total_revenue = profit + cost_of_materials)
    (h6 : total_revenue = price_per_bracelet * bracelets_sold)
    (h7 : total_bracelets = bracelets_sold + bracelets_given_away) :
    bracelets_given_away = 8 :=
by
  sorry

end NUMINAMATH_GPT_alice_bracelets_given_away_l729_72972


namespace NUMINAMATH_GPT_number_of_sets_l729_72934

theorem number_of_sets (a n : ℕ) (M : Finset ℕ) (h_consecutive : ∀ x ∈ M, ∃ k, x = a + k ∧ k < n) (h_card : M.card ≥ 2) (h_sum : M.sum id = 2002) : n = 7 :=
sorry

end NUMINAMATH_GPT_number_of_sets_l729_72934


namespace NUMINAMATH_GPT_worker_overtime_hours_l729_72950

theorem worker_overtime_hours :
  ∃ (x y : ℕ), 60 * x + 90 * y = 3240 ∧ x + y = 50 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_worker_overtime_hours_l729_72950


namespace NUMINAMATH_GPT_find_third_number_l729_72919

theorem find_third_number (N : ℤ) :
  (1274 % 12 = 2) ∧ (1275 % 12 = 3) ∧ (1285 % 12 = 1) ∧ ((1274 * 1275 * N * 1285) % 12 = 6) →
  N % 12 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_third_number_l729_72919


namespace NUMINAMATH_GPT_gcd_a_b_eq_1023_l729_72996

def a : ℕ := 2^1010 - 1
def b : ℕ := 2^1000 - 1

theorem gcd_a_b_eq_1023 : Nat.gcd a b = 1023 := 
by
  sorry

end NUMINAMATH_GPT_gcd_a_b_eq_1023_l729_72996


namespace NUMINAMATH_GPT_log4_21_correct_l729_72988

noncomputable def log4_21 (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2)
                                     (h2 : Real.log 2 = b * Real.log 7) : ℝ :=
  (a * b + 1) / (2 * b)

theorem log4_21_correct (a b : ℝ) (h1 : Real.log 3 = a * Real.log 2) 
                        (h2 : Real.log 2 = b * Real.log 7) : 
  log4_21 a b h1 h2 = (a * b + 1) / (2 * b) := 
sorry

end NUMINAMATH_GPT_log4_21_correct_l729_72988


namespace NUMINAMATH_GPT_C_finishes_work_in_days_l729_72977

theorem C_finishes_work_in_days :
  (∀ (unit : ℝ) (A B C combined: ℝ),
    combined = 1 / 4 ∧
    A = 1 / 12 ∧
    B = 1 / 24 ∧
    combined = A + B + 1 / C) → 
    C = 8 :=
  sorry

end NUMINAMATH_GPT_C_finishes_work_in_days_l729_72977


namespace NUMINAMATH_GPT_new_marketing_percentage_l729_72964

theorem new_marketing_percentage 
  (total_students : ℕ)
  (initial_finance_percentage : ℕ)
  (initial_marketing_percentage : ℕ)
  (initial_operations_management_percentage : ℕ)
  (new_finance_percentage : ℕ)
  (operations_management_percentage : ℕ)
  (total_percentage : ℕ) :
  total_students = 5000 →
  initial_finance_percentage = 85 →
  initial_marketing_percentage = 80 →
  initial_operations_management_percentage = 10 →
  new_finance_percentage = 92 →
  operations_management_percentage = 10 →
  total_percentage = 175 →
  initial_marketing_percentage - (new_finance_percentage - initial_finance_percentage) = 73 :=
by
  sorry

end NUMINAMATH_GPT_new_marketing_percentage_l729_72964


namespace NUMINAMATH_GPT_odd_func_value_l729_72928

noncomputable def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ
| x => if x > 0 then 2 * x - 3 else 0 -- f(x) is initially set to 0 when x ≤ 0, since we will not use this part directly.

theorem odd_func_value (x : ℝ) (h : x < 0) (hf : isOddFunction f) (hfx : ∀ x > 0, f x = 2 * x - 3) :
  f x = 2 * x + 3 :=
by
  sorry

end NUMINAMATH_GPT_odd_func_value_l729_72928


namespace NUMINAMATH_GPT_cuboid_volume_l729_72983

theorem cuboid_volume (P h : ℝ) (P_eq : P = 32) (h_eq : h = 9) :
  ∃ (s : ℝ), 4 * s = P ∧ s * s * h = 576 :=
by
  sorry

end NUMINAMATH_GPT_cuboid_volume_l729_72983


namespace NUMINAMATH_GPT_hall_100_guests_67_friends_find_clique_l729_72922

theorem hall_100_guests_67_friends_find_clique :
  ∀ (P : Fin 100 → Fin 100 → Prop) (n : Fin 100),
    (∀ i : Fin 100, ∃ S : Finset (Fin 100), (S.card ≥ 67) ∧ (∀ j ∈ S, P i j)) →
    (∃ (A B C D : Fin 100), P A B ∧ P A C ∧ P A D ∧ P B C ∧ P B D ∧ P C D) :=
by
  sorry

end NUMINAMATH_GPT_hall_100_guests_67_friends_find_clique_l729_72922


namespace NUMINAMATH_GPT_intercepts_of_line_l729_72942

-- Define the given line equation
def line_eq (x y : ℝ) : Prop := x / 4 - y / 3 = 1

-- Define the intercepts
def intercepts (x_intercept y_intercept : ℝ) : Prop :=
  (line_eq x_intercept 0) ∧ (line_eq 0 y_intercept)

-- The problem statement: proving the values of intercepts
theorem intercepts_of_line :
  intercepts 4 (-3) :=
by
  sorry

end NUMINAMATH_GPT_intercepts_of_line_l729_72942


namespace NUMINAMATH_GPT_count_balanced_integers_l729_72931

def is_balanced (n : ℕ) : Prop :=
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  (d1 + d2 + d3) = d1 + (d2 + d3) ∧ (100 ≤ n) ∧ (n ≤ 999)

theorem count_balanced_integers : ∃ c, c = 330 ∧ ∀ n, 100 ≤ n ∧ n ≤ 999 → is_balanced n ↔ c = 330 :=
sorry

end NUMINAMATH_GPT_count_balanced_integers_l729_72931


namespace NUMINAMATH_GPT_mass_percentage_Cl_correct_l729_72918

-- Define the given condition
def mass_percentage_of_Cl := 66.04

-- Statement to prove
theorem mass_percentage_Cl_correct : mass_percentage_of_Cl = 66.04 :=
by
  -- This is where the proof would go, but we use sorry as placeholder.
  sorry

end NUMINAMATH_GPT_mass_percentage_Cl_correct_l729_72918


namespace NUMINAMATH_GPT_fishing_problem_l729_72960

theorem fishing_problem :
  ∃ F : ℕ, (F % 3 = 1 ∧
            ((F - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3) % 3 = 1) ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3) % 3 = 1 ∧
            ((((F - 1) / 3 - 1) / 3 - 1) / 3 - 1) = 0) :=
sorry

end NUMINAMATH_GPT_fishing_problem_l729_72960


namespace NUMINAMATH_GPT_ratio_increase_productivity_l729_72915

theorem ratio_increase_productivity (initial current: ℕ) 
  (h_initial: initial = 10) 
  (h_current: current = 25) : 
  (current - initial) / initial = 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_increase_productivity_l729_72915


namespace NUMINAMATH_GPT_min_value_of_even_function_l729_72967

-- Define f(x) = (x + a)(x + b)
def f (x a b : ℝ) : ℝ := (x + a) * (x + b)

-- Given conditions
variables (a b : ℝ)
#check f  -- Ensuring the definition works

-- Prove that the minimum value of f(x) is -4 given that f(x) is an even function
theorem min_value_of_even_function (h_even : ∀ x : ℝ, f x a b = f (-x) a b)
  (h_domain : a + 4 > a) : ∃ c : ℝ, (f c a b = -4) :=
by
  -- We state that this function is even and consider the provided domain.
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_min_value_of_even_function_l729_72967


namespace NUMINAMATH_GPT_molecular_weight_ammonia_l729_72974

def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.008
def count_N : ℕ := 1
def count_H : ℕ := 3

theorem molecular_weight_ammonia :
  (count_N * atomic_weight_N) + (count_H * atomic_weight_H) = 17.034 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_ammonia_l729_72974


namespace NUMINAMATH_GPT_find_x_eq_2_l729_72978

theorem find_x_eq_2 (x : ℕ) (h : 7899665 - 36 * x = 7899593) : x = 2 := 
by
  sorry

end NUMINAMATH_GPT_find_x_eq_2_l729_72978


namespace NUMINAMATH_GPT_general_pattern_specific_computation_l729_72913

theorem general_pattern (n : ℕ) : n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 :=
by
  sorry

theorem specific_computation : 2000 * 2001 * 2002 * 2003 + 1 = 4006001^2 :=
by
  have h := general_pattern 2000
  exact h

end NUMINAMATH_GPT_general_pattern_specific_computation_l729_72913


namespace NUMINAMATH_GPT_senior_tickets_count_l729_72999

theorem senior_tickets_count (A S : ℕ) 
  (h1 : A + S = 510)
  (h2 : 21 * A + 15 * S = 8748) :
  S = 327 :=
sorry

end NUMINAMATH_GPT_senior_tickets_count_l729_72999


namespace NUMINAMATH_GPT_evaluate_expression_l729_72936

theorem evaluate_expression : 60 + (105 / 15) + (25 * 16) - 250 + (324 / 9) ^ 2 = 1513 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l729_72936


namespace NUMINAMATH_GPT_goldfish_problem_l729_72952

theorem goldfish_problem (x : ℕ) : 
  (18 + (x - 5) * 7 = 4) → (x = 3) :=
by
  intros
  sorry

end NUMINAMATH_GPT_goldfish_problem_l729_72952


namespace NUMINAMATH_GPT_no_real_solution_for_quadratic_eq_l729_72986

theorem no_real_solution_for_quadratic_eq (y : ℝ) :
  (8 * y^2 + 155 * y + 3) / (4 * y + 45) = 4 * y + 3 →  (¬ ∃ y : ℝ, (8 * y^2 + 37 * y + 33/2 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solution_for_quadratic_eq_l729_72986


namespace NUMINAMATH_GPT_count_valid_subsets_l729_72929

theorem count_valid_subsets : 
  ∃ (S : Finset (Finset ℕ)), 
    (∀ A ∈ S, A ⊆ {1, 2, 3, 4, 5} ∧ 
    (∀ a ∈ A, 6 - a ∈ A)) ∧ 
    S.card = 7 := 
sorry

end NUMINAMATH_GPT_count_valid_subsets_l729_72929


namespace NUMINAMATH_GPT_power_comparison_l729_72968

noncomputable
def compare_powers : Prop := 
  1.5^(1 / 3.1) < 2^(1 / 3.1) ∧ 2^(1 / 3.1) < 2^(3.1)

theorem power_comparison : compare_powers :=
by
  sorry

end NUMINAMATH_GPT_power_comparison_l729_72968


namespace NUMINAMATH_GPT_fourth_sphere_radius_l729_72946

theorem fourth_sphere_radius (R r : ℝ) (h1 : R > 0)
  (h2 : ∀ (a b c d : ℝ × ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a →
    dist a b = 2*R ∧ dist b c = 2*R ∧ dist c d = 2*R ∧ dist d a = R + r ∧
    dist a c = R + r ∧ dist b d = R + r) :
  r = 4*R/3 :=
  sorry

end NUMINAMATH_GPT_fourth_sphere_radius_l729_72946


namespace NUMINAMATH_GPT_triangle_side_relation_l729_72954

theorem triangle_side_relation (a b c : ℝ) 
    (h_angles : 55 = 55 ∧ 15 = 15 ∧ 110 = 110) :
    c^2 - a^2 = a * b :=
  sorry

end NUMINAMATH_GPT_triangle_side_relation_l729_72954


namespace NUMINAMATH_GPT_percentage_k_equal_125_percent_j_l729_72945

theorem percentage_k_equal_125_percent_j
  (j k l m : ℝ)
  (h1 : 1.25 * j = (x / 100) * k)
  (h2 : 1.5 * k = 0.5 * l)
  (h3 : 1.75 * l = 0.75 * m)
  (h4 : 0.2 * m = 7 * j) :
  x = 25 := 
sorry

end NUMINAMATH_GPT_percentage_k_equal_125_percent_j_l729_72945


namespace NUMINAMATH_GPT_find_x_l729_72969

theorem find_x (x y : ℕ) (h1 : y = 144) (h2 : x^3 * 6^2 / 432 = y) : x = 12 := 
by
  sorry

end NUMINAMATH_GPT_find_x_l729_72969


namespace NUMINAMATH_GPT_range_of_a_l729_72902

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (|x + a| < 3 ↔ 2 < x ∧ x < 3)) ↔ -5 ≤ a ∧ a ≤ 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l729_72902


namespace NUMINAMATH_GPT_number_made_l729_72971

theorem number_made (x y : ℕ) (h1 : x + y = 24) (h2 : x = 11) : 7 * x + 5 * y = 142 := by
  sorry

end NUMINAMATH_GPT_number_made_l729_72971


namespace NUMINAMATH_GPT_geom_series_common_ratio_l729_72973

theorem geom_series_common_ratio (a r S : ℝ) (h1 : S = a / (1 - r)) 
  (h2 : (ar^4) / (1 - r) = S / 64) : r = 1 / 2 :=
sorry

end NUMINAMATH_GPT_geom_series_common_ratio_l729_72973


namespace NUMINAMATH_GPT_seats_taken_l729_72925

variable (num_rows : ℕ) (chairs_per_row : ℕ) (unoccupied_chairs : ℕ)

theorem seats_taken (h1 : num_rows = 40) (h2 : chairs_per_row = 20) (h3 : unoccupied_chairs = 10) :
  num_rows * chairs_per_row - unoccupied_chairs = 790 :=
sorry

end NUMINAMATH_GPT_seats_taken_l729_72925


namespace NUMINAMATH_GPT_expressway_lengths_l729_72940

theorem expressway_lengths (x y : ℕ) (h1 : x + y = 519) (h2 : x = 2 * y - 45) : x = 331 ∧ y = 188 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_expressway_lengths_l729_72940


namespace NUMINAMATH_GPT_remainder_29_times_171997_pow_2000_mod_7_l729_72933

theorem remainder_29_times_171997_pow_2000_mod_7 :
  (29 * 171997^2000) % 7 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_29_times_171997_pow_2000_mod_7_l729_72933


namespace NUMINAMATH_GPT_spring_excursion_participants_l729_72903

theorem spring_excursion_participants (water fruit neither both total : ℕ) 
  (h_water : water = 80) 
  (h_fruit : fruit = 70) 
  (h_neither : neither = 6) 
  (h_both : both = total / 2) 
  (h_total_eq : total = water + fruit - both + neither) : 
  total = 104 := 
  sorry

end NUMINAMATH_GPT_spring_excursion_participants_l729_72903


namespace NUMINAMATH_GPT_no_two_items_share_color_l729_72958

theorem no_two_items_share_color (shirts pants hats : Fin 5) :
  ∃ num_outfits : ℕ, num_outfits = 60 :=
by
  sorry

end NUMINAMATH_GPT_no_two_items_share_color_l729_72958


namespace NUMINAMATH_GPT_power_zero_equals_one_specific_case_l729_72906

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end NUMINAMATH_GPT_power_zero_equals_one_specific_case_l729_72906


namespace NUMINAMATH_GPT_students_neither_music_nor_art_l729_72949

theorem students_neither_music_nor_art
  (total_students : ℕ) (students_music : ℕ) (students_art : ℕ) (students_both : ℕ)
  (h_total : total_students = 500)
  (h_music : students_music = 30)
  (h_art : students_art = 10)
  (h_both : students_both = 10)
  : total_students - (students_music + students_art - students_both) = 460 :=
by
  rw [h_total, h_music, h_art, h_both]
  norm_num
  sorry

end NUMINAMATH_GPT_students_neither_music_nor_art_l729_72949


namespace NUMINAMATH_GPT_determine_values_l729_72991

-- Define the main problem conditions
def A := 1.2
def B := 12

-- The theorem statement capturing the problem conditions and the solution
theorem determine_values (A B : ℝ) (h1 : A + B = 13.2) (h2 : B = 10 * A) : A = 1.2 ∧ B = 12 :=
  sorry

end NUMINAMATH_GPT_determine_values_l729_72991


namespace NUMINAMATH_GPT_sqrt_expression_meaningful_l729_72920

theorem sqrt_expression_meaningful {x : ℝ} : (2 * x - 4) ≥ 0 → x ≥ 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sqrt_expression_meaningful_l729_72920


namespace NUMINAMATH_GPT_smallest_perimeter_of_acute_triangle_with_consecutive_sides_l729_72916

theorem smallest_perimeter_of_acute_triangle_with_consecutive_sides :
  ∃ (a : ℕ), (a > 1) ∧ (∃ (b c : ℕ), b = a + 1 ∧ c = a + 2 ∧ (∃ (C : ℝ), a^2 + b^2 - c^2 < 0 ∧ c = 4)) ∧ (a + (a + 1) + (a + 2) = 9) :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_perimeter_of_acute_triangle_with_consecutive_sides_l729_72916


namespace NUMINAMATH_GPT_tessa_initial_apples_l729_72984

-- Define conditions as variables
variable (initial_apples anita_gave : ℕ)
variable (apples_needed_for_pie : ℕ := 10)
variable (apples_additional_now_needed : ℕ := 1)

-- Define the current amount of apples Tessa has
noncomputable def current_apples :=
  apples_needed_for_pie - apples_additional_now_needed

-- Define the initial apples Tessa had before Anita gave her 5 apples
noncomputable def initial_apples_calculated :=
  current_apples - anita_gave

-- Lean statement to prove the initial number of apples Tessa had
theorem tessa_initial_apples (h_initial_apples : anita_gave = 5) : initial_apples_calculated = 4 :=
by
  -- Here is where the proof would go; we use sorry to indicate it's not provided
  sorry

end NUMINAMATH_GPT_tessa_initial_apples_l729_72984


namespace NUMINAMATH_GPT_no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l729_72989

theorem no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime :
  ¬∃ n : ℕ, 2 ≤ n ∧ Nat.Prime (n^4 + n^2 + 1) :=
sorry

end NUMINAMATH_GPT_no_n_geq_2_for_nquad_plus_nsquare_plus_one_prime_l729_72989


namespace NUMINAMATH_GPT_danny_distance_to_work_l729_72963

-- Define the conditions and the problem in terms of Lean definitions
def distance_to_first_friend : ℕ := 8
def distance_to_second_friend : ℕ := distance_to_first_friend / 2
def total_distance_driven_so_far : ℕ := distance_to_first_friend + distance_to_second_friend
def distance_to_work : ℕ := 3 * total_distance_driven_so_far

-- Lean statement to be proven
theorem danny_distance_to_work :
  distance_to_work = 36 :=
by
  -- This is the proof placeholder
  sorry

end NUMINAMATH_GPT_danny_distance_to_work_l729_72963


namespace NUMINAMATH_GPT_range_of_x_l729_72909

theorem range_of_x (a : ℝ) (x : ℝ) (h0 : 0 ≤ a) (h1 : a ≤ 2) :
  a * x^2 + (a + 1) * x + 1 - (3 / 2) * a < 0 → -2 < x ∧ x < -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l729_72909


namespace NUMINAMATH_GPT_parent_payment_per_year_l729_72966

noncomputable def former_salary : ℕ := 45000
noncomputable def raise_percentage : ℕ := 20
noncomputable def number_of_kids : ℕ := 9

theorem parent_payment_per_year : 
  (former_salary + (raise_percentage * former_salary / 100)) / number_of_kids = 6000 := by
  sorry

end NUMINAMATH_GPT_parent_payment_per_year_l729_72966


namespace NUMINAMATH_GPT_part1_minimum_value_part2_max_k_l729_72932

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x
noncomputable def g (x : ℝ) : ℝ := (x + x * Real.log x) / (x - 1)

theorem part1_minimum_value : ∃ x₀ : ℝ, x₀ = Real.exp (-2) ∧ f x₀ = -Real.exp (-2) := 
by
  use Real.exp (-2)
  sorry

theorem part2_max_k (k : ℤ) : (∀ x > 1, f x > k * (x - 1)) → k ≤ 3 := 
by
  sorry

end NUMINAMATH_GPT_part1_minimum_value_part2_max_k_l729_72932


namespace NUMINAMATH_GPT_circle_radius_l729_72981

theorem circle_radius :
  ∃ radius : ℝ, (∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 16 → (x - 2)^2 + (y - 1)^2 = radius^2)
  ∧ radius = 4 :=
sorry

end NUMINAMATH_GPT_circle_radius_l729_72981


namespace NUMINAMATH_GPT_time_to_fill_is_correct_l729_72935

-- Definitions of rates
variable (R_1 : ℚ) (R_2 : ℚ)

-- Conditions given in the problem
def rate1 := (1 : ℚ) / 8
def rate2 := (1 : ℚ) / 12

-- The resultant rate when both pipes work together
def combined_rate := rate1 + rate2

-- Calculate the time taken to fill the tank
def time_to_fill_tank := 1 / combined_rate

theorem time_to_fill_is_correct (h1 : R_1 = rate1) (h2 : R_2 = rate2) :
  time_to_fill_tank = 24 / 5 := by
  sorry

end NUMINAMATH_GPT_time_to_fill_is_correct_l729_72935


namespace NUMINAMATH_GPT_triangle_area_six_parts_l729_72921

theorem triangle_area_six_parts (S S₁ S₂ S₃ : ℝ) (h₁ : S₁ ≥ 0) (h₂ : S₂ ≥ 0) (h₃ : S₃ ≥ 0) :
  S = (Real.sqrt S₁ + Real.sqrt S₂ + Real.sqrt S₃) ^ 2 := 
sorry

end NUMINAMATH_GPT_triangle_area_six_parts_l729_72921


namespace NUMINAMATH_GPT_area_of_R2_l729_72982

theorem area_of_R2
  (a b : ℝ)
  (h1 : b = 3 * a)
  (h2 : a^2 + b^2 = 225) :
  a * b = 135 / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_R2_l729_72982


namespace NUMINAMATH_GPT_complex_fraction_sum_real_parts_l729_72985

theorem complex_fraction_sum_real_parts (a b : ℝ) (h : (⟨0, 1⟩ / ⟨1, 1⟩ : ℂ) = a + b * ⟨0, 1⟩) : a + b = 1 := by
  sorry

end NUMINAMATH_GPT_complex_fraction_sum_real_parts_l729_72985


namespace NUMINAMATH_GPT_initial_courses_of_bricks_l729_72908

theorem initial_courses_of_bricks (x : ℕ) : 
    400 * x + 2 * 400 - 400 / 2 = 1800 → x = 3 :=
by
  sorry

end NUMINAMATH_GPT_initial_courses_of_bricks_l729_72908


namespace NUMINAMATH_GPT_multiplier_of_product_l729_72939

variable {a b : ℝ}

theorem multiplier_of_product (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : a + b = k * (a * b))
  (h4 : (1 / a) + (1 / b) = 6) : k = 6 := by
  sorry

end NUMINAMATH_GPT_multiplier_of_product_l729_72939


namespace NUMINAMATH_GPT_shaded_area_of_rotated_semicircle_l729_72927

-- Definitions and conditions from the problem
def radius (R : ℝ) : Prop := R > 0
def central_angle (α : ℝ) : Prop := α = 30 * (Real.pi / 180)

-- Lean theorem statement for the proof problem
theorem shaded_area_of_rotated_semicircle (R : ℝ) (hR : radius R) (hα : central_angle 30) : 
  ∃ (area : ℝ), area = (Real.pi * R^2) / 3 :=
by
  -- using proofs of radius and angle conditions
  sorry

end NUMINAMATH_GPT_shaded_area_of_rotated_semicircle_l729_72927


namespace NUMINAMATH_GPT_remaining_black_area_after_five_changes_l729_72910

-- Define a function that represents the change process
noncomputable def remaining_black_area (iterations : ℕ) : ℚ :=
  (3 / 4) ^ iterations

-- Define the original problem statement as a theorem in Lean
theorem remaining_black_area_after_five_changes :
  remaining_black_area 5 = 243 / 1024 :=
by
  sorry

end NUMINAMATH_GPT_remaining_black_area_after_five_changes_l729_72910


namespace NUMINAMATH_GPT_find_y_l729_72959

-- Define the conditions (inversely proportional and sum condition)
def inversely_proportional (x y : ℝ) (k : ℝ) : Prop := x * y = k
def sum_condition (x y : ℝ) : Prop := x + y = 50 ∧ x = 3 * y

-- Given these conditions, prove the value of y when x = -12
theorem find_y (k x y : ℝ)
  (h1 : inversely_proportional x y k)
  (h2 : sum_condition 37.5 12.5)
  (hx : x = -12) :
  y = -39.0625 :=
sorry

end NUMINAMATH_GPT_find_y_l729_72959


namespace NUMINAMATH_GPT_perpendicular_vectors_x_eq_5_l729_72994

def vector_a (x : ℝ) : ℝ × ℝ := (2, x + 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x - 2, -1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors_x_eq_5 (x : ℝ)
  (h : dot_product (vector_a x) (vector_b x) = 0) :
  x = 5 :=
sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_eq_5_l729_72994


namespace NUMINAMATH_GPT_red_light_max_probability_l729_72980

theorem red_light_max_probability {m : ℕ} (h1 : m > 0) (h2 : m < 35) :
  m = 3 ∨ m = 15 ∨ m = 30 ∨ m = 40 → m = 30 :=
by
  sorry

end NUMINAMATH_GPT_red_light_max_probability_l729_72980


namespace NUMINAMATH_GPT_sum_q_p_evaluations_l729_72993

def p (x : ℝ) : ℝ := |x^2 - 4|
def q (x : ℝ) : ℝ := -|x|

theorem sum_q_p_evaluations : 
  q (p (-3)) + q (p (-2)) + q (p (-1)) + q (p (0)) + q (p (1)) + q (p (2)) + q (p (3)) = -20 := 
by 
  sorry

end NUMINAMATH_GPT_sum_q_p_evaluations_l729_72993


namespace NUMINAMATH_GPT_general_eq_line_BC_std_eq_circumscribed_circle_ABC_l729_72957

-- Define the points A, B, and C
def A : ℝ × ℝ := (-1, 1)
def B : ℝ × ℝ := (-1, 2)
def C : ℝ × ℝ := (-4, 1)

-- Prove the general equation of line BC is x + 1 = 0
theorem general_eq_line_BC : ∀ x y : ℝ, (x = -1) → y = 2 ∧ (x = -4) → y = 1 → x + 1 = 0 :=
by
  sorry

-- Prove the standard equation of the circumscribed circle of triangle ABC is (x + 5/2)^2 + (y - 3/2)^2 = 5/2
theorem std_eq_circumscribed_circle_ABC :
  ∀ x y : ℝ,
  (x, y) = (A : ℝ × ℝ) ∨ (x, y) = (B : ℝ × ℝ) ∨ (x, y) = (C : ℝ × ℝ) →
  (x + 5/2)^2 + (y - 3/2)^2 = 5/2 :=
by
  sorry

end NUMINAMATH_GPT_general_eq_line_BC_std_eq_circumscribed_circle_ABC_l729_72957


namespace NUMINAMATH_GPT_melanie_trout_l729_72907

theorem melanie_trout (M : ℕ) (h1 : 2 * M = 16) : M = 8 :=
by
  sorry

end NUMINAMATH_GPT_melanie_trout_l729_72907


namespace NUMINAMATH_GPT_loaves_at_start_l729_72905

variable (X : ℕ) -- X represents the number of loaves at the start of the day.

-- Conditions given in the problem:
def final_loaves (X : ℕ) : Prop := X - 629 + 489 = 2215

-- The theorem to be proved:
theorem loaves_at_start (h : final_loaves X) : X = 2355 :=
by sorry

end NUMINAMATH_GPT_loaves_at_start_l729_72905


namespace NUMINAMATH_GPT_vector_satisfy_condition_l729_72912

structure Point where
  x : ℝ
  y : ℝ

structure Line where
  parametrize : ℝ → Point

def l : Line :=
  { parametrize := λ t => {x := 1 + 4 * t, y := 4 + 3 * t} }

def m : Line :=
  { parametrize := λ s => {x := -5 + 4 * s, y := 6 + 3 * s} }

def A (t : ℝ) : Point := l.parametrize t
def B (s : ℝ) : Point := m.parametrize s

-- The specific point for A and B are not used directly in the further proof statement.

def v : Point := { x := -6, y := 8 }

theorem vector_satisfy_condition :
  ∃ v1 v2 : ℝ, (v1 * -6) + (v2 * 8) = 2 ∧ (v1 = -6 ∧ v2 = 8) :=
sorry

end NUMINAMATH_GPT_vector_satisfy_condition_l729_72912


namespace NUMINAMATH_GPT_DanteSoldCoconuts_l729_72943

variable (Paolo_coconuts : ℕ) (Dante_coconuts : ℕ) (coconuts_left : ℕ)

def PaoloHasCoconuts := Paolo_coconuts = 14

def DanteHasThriceCoconuts := Dante_coconuts = 3 * Paolo_coconuts

def DanteLeftCoconuts := coconuts_left = 32

theorem DanteSoldCoconuts 
  (h1 : PaoloHasCoconuts Paolo_coconuts) 
  (h2 : DanteHasThriceCoconuts Paolo_coconuts Dante_coconuts) 
  (h3 : DanteLeftCoconuts coconuts_left) : 
  Dante_coconuts - coconuts_left = 10 := 
by
  rw [PaoloHasCoconuts, DanteHasThriceCoconuts, DanteLeftCoconuts] at *
  sorry

end NUMINAMATH_GPT_DanteSoldCoconuts_l729_72943


namespace NUMINAMATH_GPT_sniper_B_has_greater_chance_of_winning_l729_72998

-- Define the probabilities for sniper A
def p_A_1 := 0.4
def p_A_2 := 0.1
def p_A_3 := 0.5

-- Define the probabilities for sniper B
def p_B_1 := 0.1
def p_B_2 := 0.6
def p_B_3 := 0.3

-- Define the expected scores for sniper A and B
def E_A := 1 * p_A_1 + 2 * p_A_2 + 3 * p_A_3
def E_B := 1 * p_B_1 + 2 * p_B_2 + 3 * p_B_3

-- The statement we want to prove
theorem sniper_B_has_greater_chance_of_winning : E_B > E_A := by
  simp [E_A, E_B, p_A_1, p_A_2, p_A_3, p_B_1, p_B_2, p_B_3]
  sorry

end NUMINAMATH_GPT_sniper_B_has_greater_chance_of_winning_l729_72998


namespace NUMINAMATH_GPT_randy_biscuits_l729_72955

theorem randy_biscuits (initial_biscuits father_gift mother_gift brother_ate : ℕ) : 
  (initial_biscuits = 32) →
  (father_gift = 13) →
  (mother_gift = 15) →
  (brother_ate = 20) →
  initial_biscuits + father_gift + mother_gift - brother_ate = 40 := by
  sorry

end NUMINAMATH_GPT_randy_biscuits_l729_72955


namespace NUMINAMATH_GPT_imaginary_part_div_z1_z2_l729_72938

noncomputable def z1 := 1 - 3 * Complex.I
noncomputable def z2 := 3 + Complex.I

theorem imaginary_part_div_z1_z2 : 
  Complex.im ((1 + 3 * Complex.I) / (3 + Complex.I)) = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_imaginary_part_div_z1_z2_l729_72938


namespace NUMINAMATH_GPT_remainder_x14_minus_1_div_x_plus_1_l729_72995

-- Define the polynomial f(x) = x^14 - 1
def f (x : ℝ) := x^14 - 1

-- Statement to prove that the remainder when f(x) is divided by x + 1 is 0
theorem remainder_x14_minus_1_div_x_plus_1 : f (-1) = 0 :=
by
  -- This is where the proof would go, but for now, we will just use sorry
  sorry

end NUMINAMATH_GPT_remainder_x14_minus_1_div_x_plus_1_l729_72995


namespace NUMINAMATH_GPT_not_divisible_by_8_l729_72914

theorem not_divisible_by_8 : ¬ (456294604884 % 8 = 0) := 
by
  have h : 456294604884 % 1000 = 884 := sorry -- This step reflects the conclusion that the last three digits are 884.
  have h_div : ¬ (884 % 8 = 0) := sorry -- This reflects that 884 is not divisible by 8.
  sorry

end NUMINAMATH_GPT_not_divisible_by_8_l729_72914


namespace NUMINAMATH_GPT_cost_per_gallon_is_45_l729_72956

variable (totalArea coverage cost_jason cost_jeremy dollars_per_gallon : ℕ)

-- Conditions
def total_area := 1600
def coverage_per_gallon := 400
def num_coats := 2
def contribution_jason := 180
def contribution_jeremy := 180

-- Gallons needed calculation
def gallons_per_coat := total_area / coverage_per_gallon
def total_gallons := gallons_per_coat * num_coats

-- Total cost calculation
def total_cost := contribution_jason + contribution_jeremy

-- Cost per gallon calculation
def cost_per_gallon := total_cost / total_gallons

-- Proof statement
theorem cost_per_gallon_is_45 : cost_per_gallon = 45 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_gallon_is_45_l729_72956


namespace NUMINAMATH_GPT_intersection_sums_l729_72944

def parabola1 (x : ℝ) : ℝ := (x - 2)^2
def parabola2 (y : ℝ) : ℝ := (y - 2)^2 - 6

theorem intersection_sums (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : y1 = parabola1 x1) (h2 : y2 = parabola1 x2)
  (h3 : y3 = parabola1 x3) (h4 : y4 = parabola1 x4)
  (k1 : x1 + 6 = y1^2 - 4*y1 + 4) (k2 : x2 + 6 = y2^2 - 4*y2 + 4)
  (k3 : x3 + 6 = y3^2 - 4*y3 + 4) (k4 : x4 + 6 = y4^2 - 4*y4 + 4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 16 := 
sorry

end NUMINAMATH_GPT_intersection_sums_l729_72944


namespace NUMINAMATH_GPT_range_of_x_l729_72924

variable (x : ℝ)

theorem range_of_x (h1 : 2 - x > 0) (h2 : x - 1 ≥ 0) : 1 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l729_72924


namespace NUMINAMATH_GPT_tan_proof_l729_72953

noncomputable def prove_tan_relation (α β : ℝ) : Prop :=
  2 * (Real.tan α) = 3 * (Real.tan β)

theorem tan_proof (α β : ℝ) (h : Real.tan (α - β) = (Real.sin (2*β)) / (5 - Real.cos (2*β))) : 
  prove_tan_relation α β :=
sorry

end NUMINAMATH_GPT_tan_proof_l729_72953


namespace NUMINAMATH_GPT_total_kids_in_lawrence_county_l729_72975

def kids_stayed_home : ℕ := 644997
def kids_went_to_camp : ℕ := 893835
def kids_from_outside : ℕ := 78

theorem total_kids_in_lawrence_county : kids_stayed_home + kids_went_to_camp = 1538832 := by
  sorry

end NUMINAMATH_GPT_total_kids_in_lawrence_county_l729_72975
