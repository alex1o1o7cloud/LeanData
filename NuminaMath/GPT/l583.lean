import Mathlib

namespace sample_size_calculation_l583_58362

theorem sample_size_calculation :
  let workshop_A := 120
  let workshop_B := 80
  let workshop_C := 60
  let sample_from_C := 3
  let sampling_fraction := sample_from_C / workshop_C
  let sample_A := workshop_A * sampling_fraction
  let sample_B := workshop_B * sampling_fraction
  let sample_C := workshop_C * sampling_fraction
  let n := sample_A + sample_B + sample_C
  n = 13 :=
by
  sorry

end sample_size_calculation_l583_58362


namespace part2_l583_58382

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part2 (x y : ℝ) (h₁ : |x - y - 1| ≤ 1 / 3) (h₂ : |2 * y + 1| ≤ 1 / 6) :
  f x < 1 := 
by
  sorry

end part2_l583_58382


namespace solve_a_l583_58314

theorem solve_a (a x : ℤ) (h₀ : x = 5) (h₁ : a * x - 8 = 20 + a) : a = 7 :=
by
  sorry

end solve_a_l583_58314


namespace mike_notebooks_total_l583_58319

theorem mike_notebooks_total
  (red_notebooks : ℕ)
  (green_notebooks : ℕ)
  (blue_notebooks_cost : ℕ)
  (total_cost : ℕ)
  (red_cost : ℕ)
  (green_cost : ℕ)
  (blue_cost : ℕ)
  (h1 : red_notebooks = 3)
  (h2 : red_cost = 4)
  (h3 : green_notebooks = 2)
  (h4 : green_cost = 2)
  (h5 : total_cost = 37)
  (h6 : blue_cost = 3)
  (h7 : total_cost = red_notebooks * red_cost + green_notebooks * green_cost + blue_notebooks_cost) :
  (red_notebooks + green_notebooks + blue_notebooks_cost / blue_cost = 12) :=
by {
  sorry
}

end mike_notebooks_total_l583_58319


namespace vector_sum_is_zero_l583_58307

variables {V : Type*} [AddCommGroup V]

variables (AB CF BC FA : V)

-- Condition: Vectors form a closed polygon
def vectors_form_closed_polygon (AB CF BC FA : V) : Prop :=
  AB + BC + CF + FA = 0

theorem vector_sum_is_zero
  (h : vectors_form_closed_polygon AB CF BC FA) :
  AB + BC + CF + FA = 0 :=
  h

end vector_sum_is_zero_l583_58307


namespace system_solution_l583_58390

theorem system_solution (x : Fin 1995 → ℤ) :
  (∀ i : (Fin 1995),
    x (i + 1) ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) →
  (∀ n : (Fin 1995),
    (x n = 0 ∧ x (n + 1) = 1 ∧ x (n + 2) = -1) ∨
    (x n = 0 ∧ x (n + 1) = -1 ∧ x (n + 2) = 1)) :=
by sorry

end system_solution_l583_58390


namespace find_m_l583_58329

theorem find_m (m : ℝ) 
  (f g : ℝ → ℝ) 
  (x : ℝ) 
  (hf : f x = x^2 - 3 * x + m) 
  (hg : g x = x^2 - 3 * x + 5 * m) 
  (hx : x = 5) 
  (h_eq : 3 * f x = 2 * g x) :
  m = 10 / 7 := 
sorry

end find_m_l583_58329


namespace quartic_two_real_roots_l583_58334

theorem quartic_two_real_roots
  (a b c d e : ℝ)
  (h : ∃ β : ℝ, β > 1 ∧ a * β^2 + (c - b) * β + e - d = 0)
  (ha : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^4 + b * x1^3 + c * x1^2 + d * x1 + e = 0) ∧ (a * x2^4 + b * x2^3 + c * x2^2 + d * x2 + e = 0) := 
  sorry

end quartic_two_real_roots_l583_58334


namespace trip_time_difference_l583_58394

-- Definitions of the given conditions
def speed_AB := 160 -- speed from A to B in km/h
def speed_BA := 120 -- speed from B to A in km/h
def distance_AB := 480 -- distance between A and B in km

-- Calculation of the time for each trip
def time_AB := distance_AB / speed_AB
def time_BA := distance_AB / speed_BA

-- The statement we need to prove
theorem trip_time_difference :
  (time_BA - time_AB) = 1 :=
by
  sorry

end trip_time_difference_l583_58394


namespace initial_pennies_l583_58385

theorem initial_pennies (P : ℕ)
  (h1 : P - (P / 2 + 1) = P / 2 - 1)
  (h2 : (P / 2 - 1) - (P / 4 + 1 / 2) = P / 4 - 3 / 2)
  (h3 : (P / 4 - 3 / 2) - (P / 8 + 3 / 4) = P / 8 - 9 / 4)
  (h4 : P / 8 - 9 / 4 = 1)
  : P = 26 := 
by
  sorry

end initial_pennies_l583_58385


namespace number_with_29_proper_divisors_is_720_l583_58336

theorem number_with_29_proper_divisors_is_720
  (n : ℕ) (h1 : n < 1000)
  (h2 : ∀ d, 1 < d ∧ d < n -> ∃ m, n = d * m):
  n = 720 := by
  sorry

end number_with_29_proper_divisors_is_720_l583_58336


namespace total_meals_per_week_l583_58389

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end total_meals_per_week_l583_58389


namespace sqrt_mult_minus_two_l583_58349

theorem sqrt_mult_minus_two (x y : ℝ) (hx : x = Real.sqrt 3) (hy : y = Real.sqrt 6) : 
  2 < x * y - 2 ∧ x * y - 2 < 3 := by
  sorry

end sqrt_mult_minus_two_l583_58349


namespace largest_n_l583_58381

def canBeFactored (A B : ℤ) : Bool :=
  A * B = 54

theorem largest_n (n : ℤ) (h : ∃ (A B : ℤ), canBeFactored A B ∧ 3 * B + A = n) :
  n = 163 :=
by
  sorry

end largest_n_l583_58381


namespace orchard_problem_l583_58345

theorem orchard_problem (number_of_peach_trees number_of_apple_trees : ℕ) 
  (h1 : number_of_apple_trees = number_of_peach_trees + 1700)
  (h2 : number_of_apple_trees = 3 * number_of_peach_trees + 200) :
  number_of_peach_trees = 750 ∧ number_of_apple_trees = 2450 :=
by
  sorry

end orchard_problem_l583_58345


namespace AM_GM_HY_order_l583_58395

noncomputable def AM (a b c : ℝ) : ℝ := (a + b + c) / 3
noncomputable def GM (a b c : ℝ) : ℝ := (a * b * c)^(1/3)
noncomputable def HY (a b c : ℝ) : ℝ := 2 * a * b * c / (a * b + b * c + c * a)

theorem AM_GM_HY_order (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  AM a b c > GM a b c ∧ GM a b c > HY a b c := by
  sorry

end AM_GM_HY_order_l583_58395


namespace proof_problem_l583_58351

open Real

-- Definitions
noncomputable def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Conditions
def eccentricity (c a : ℝ) : Prop :=
  c / a = (sqrt 2) / 2

def min_distance_to_focus (a c : ℝ) : Prop :=
  a - c = sqrt 2 - 1

-- Proof problem statement
theorem proof_problem (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (b_lt_a : b < a)
  (ecc : eccentricity c a) (min_dist : min_distance_to_focus a c)
  (x y k m : ℝ) (line_condition : y = k * x + m) :
  ellipse_equation x y a b → ellipse_equation x y (sqrt 2) 1 ∧
  (parabola_equation x y → (y = sqrt 2 / 2 * x + sqrt 2 ∨ y = -sqrt 2 / 2 * x - sqrt 2)) :=
sorry

end proof_problem_l583_58351


namespace total_jury_duty_days_l583_58318

-- Conditions
def jury_selection_days : ℕ := 2
def trial_multiplier : ℕ := 4
def evidence_review_hours : ℕ := 2
def lunch_hours : ℕ := 1
def trial_session_hours : ℕ := 6
def hours_per_day : ℕ := evidence_review_hours + lunch_hours + trial_session_hours
def deliberation_hours_per_day : ℕ := 14 - 2

def deliberation_first_defendant_days : ℕ := 6
def deliberation_second_defendant_days : ℕ := 4
def deliberation_third_defendant_days : ℕ := 5

def deliberation_first_defendant_total_hours : ℕ := deliberation_first_defendant_days * deliberation_hours_per_day
def deliberation_second_defendant_total_hours : ℕ := deliberation_second_defendant_days * deliberation_hours_per_day
def deliberation_third_defendant_total_hours : ℕ := deliberation_third_defendant_days * deliberation_hours_per_day

def deliberation_days_conversion (total_hours: ℕ) : ℕ := (total_hours + deliberation_hours_per_day - 1) / deliberation_hours_per_day

-- Total days spent
def total_days_spent : ℕ :=
  let trial_days := jury_selection_days * trial_multiplier
  let deliberation_days := deliberation_days_conversion deliberation_first_defendant_total_hours + deliberation_days_conversion deliberation_second_defendant_total_hours + deliberation_days_conversion deliberation_third_defendant_total_hours
  jury_selection_days + trial_days + deliberation_days

#eval total_days_spent -- Expected: 25

theorem total_jury_duty_days : total_days_spent = 25 := by
  sorry

end total_jury_duty_days_l583_58318


namespace same_side_of_line_l583_58328

theorem same_side_of_line (a : ℝ) :
  (3 * 3 - 2 * 1 + a) * (3 * (-4) - 2 * 6 + a) > 0 ↔ a < -7 ∨ a > 24 :=
by
  sorry

end same_side_of_line_l583_58328


namespace find_tricias_age_l583_58379

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l583_58379


namespace subset_A_B_l583_58338

def A (a : ℝ) : Set ℝ := {0, -a}
def B (a : ℝ) : Set ℝ := {1, a - 2, 2 * a - 2}

theorem subset_A_B (a : ℝ) : A a ⊆ B a ↔ a = 1 := by
  sorry

end subset_A_B_l583_58338


namespace top_card_yellow_second_card_not_yellow_l583_58330

-- Definitions based on conditions
def total_cards : Nat := 65

def yellow_cards : Nat := 13

def non_yellow_cards : Nat := total_cards - yellow_cards

-- Total combinations of choosing two cards
def total_combinations : Nat := total_cards * (total_cards - 1)

-- Numerator for desired probability 
def desired_combinations : Nat := yellow_cards * non_yellow_cards

-- Target probability
def desired_probability : Rat := Rat.ofInt (desired_combinations) / Rat.ofInt (total_combinations)

-- Mathematical proof statement
theorem top_card_yellow_second_card_not_yellow :
  desired_probability = Rat.ofInt 169 / Rat.ofInt 1040 :=
by
  sorry

end top_card_yellow_second_card_not_yellow_l583_58330


namespace total_packs_equiv_117_l583_58312

theorem total_packs_equiv_117 
  (nancy_cards : ℕ)
  (melanie_cards : ℕ)
  (mary_cards : ℕ)
  (alyssa_cards : ℕ)
  (nancy_pack : ℝ)
  (melanie_pack : ℝ)
  (mary_pack : ℝ)
  (alyssa_pack : ℝ)
  (H_nancy : nancy_cards = 540)
  (H_melanie : melanie_cards = 620)
  (H_mary : mary_cards = 480)
  (H_alyssa : alyssa_cards = 720)
  (H_nancy_pack : nancy_pack = 18.5)
  (H_melanie_pack : melanie_pack = 22.5)
  (H_mary_pack : mary_pack = 15.3)
  (H_alyssa_pack : alyssa_pack = 24) :
  (⌊nancy_cards / nancy_pack⌋₊ + ⌊melanie_cards / melanie_pack⌋₊ + ⌊mary_cards / mary_pack⌋₊ + ⌊alyssa_cards / alyssa_pack⌋₊) = 117 :=
by
  sorry

end total_packs_equiv_117_l583_58312


namespace paper_folding_ratio_l583_58346

theorem paper_folding_ratio :
  ∃ (side length small_perim large_perim : ℕ), 
    side_length = 6 ∧ 
    small_perim = 2 * (3 + 3) ∧ 
    large_perim = 2 * (6 + 3) ∧ 
    small_perim / large_perim = 2 / 3 :=
by sorry

end paper_folding_ratio_l583_58346


namespace original_number_l583_58392

theorem original_number (x : ℝ) (h1 : 74 * x = 19732) : x = 267 := by
  sorry

end original_number_l583_58392


namespace polynomial_division_quotient_l583_58399

theorem polynomial_division_quotient :
  ∀ (x : ℝ), (x^5 - 21*x^3 + 8*x^2 - 17*x + 12) / (x - 3) = (x^4 + 3*x^3 - 12*x^2 - 28*x - 101) :=
by
  sorry

end polynomial_division_quotient_l583_58399


namespace range_of_a_l583_58383

variable (a : ℝ)

def p := ∀ x : ℝ, x^2 + a ≥ 0
def q := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (h : p a ∧ q a) : 0 ≤ a :=
by
  sorry

end range_of_a_l583_58383


namespace kerosene_cost_l583_58313

/-- A dozen eggs cost as much as a pound of rice, a half-liter of kerosene costs as much as 8 eggs,
and each pound of rice costs $0.33. Prove that a liter of kerosene costs 44 cents. -/
theorem kerosene_cost :
  let egg_cost := 0.33 / 12
  let rice_cost := 0.33
  let half_liter_kerosene_cost := 8 * egg_cost
  let liter_kerosene_cost := half_liter_kerosene_cost * 2
  liter_kerosene_cost * 100 = 44 := 
by
  sorry

end kerosene_cost_l583_58313


namespace max_airlines_in_country_l583_58378

-- Definition of the problem parameters
variable (N k : ℕ) 

-- Definition of the problem conditions
variable (hN_pos : 0 < N)
variable (hk_pos : 0 < k)
variable (hN_ge_k : k ≤ N)

-- Definition of the function calculating the maximum number of air routes
def max_air_routes (N k : ℕ) : ℕ :=
  Nat.choose N 2 - Nat.choose k 2

-- Theorem stating the maximum number of airlines given the conditions
theorem max_airlines_in_country (N k : ℕ) (hN_pos : 0 < N) (hk_pos : 0 < k) (hN_ge_k : k ≤ N) :
  max_air_routes N k = Nat.choose N 2 - Nat.choose k 2 :=
by sorry

end max_airlines_in_country_l583_58378


namespace min_y_squared_l583_58341

noncomputable def isosceles_trapezoid_bases (EF GH : ℝ) := EF = 102 ∧ GH = 26

noncomputable def trapezoid_sides (EG FH y : ℝ) := EG = y ∧ FH = y

noncomputable def tangent_circle (center_on_EF tangent_to_EG_FH : Prop) := 
  ∃ P : ℝ × ℝ, true -- center P exists somewhere and lies on EF

theorem min_y_squared (EF GH EG FH y : ℝ) (center_on_EF tangent_to_EG_FH : Prop) 
  (h1 : isosceles_trapezoid_bases EF GH)
  (h2 : trapezoid_sides EG FH y)
  (h3 : tangent_circle center_on_EF tangent_to_EG_FH) : 
  ∃ n : ℝ, n^2 = 1938 :=
sorry

end min_y_squared_l583_58341


namespace right_triangle_acute_angle_l583_58317

theorem right_triangle_acute_angle (A B : ℝ) (h₁ : A + B = 90) (h₂ : A = 40) : B = 50 :=
by
  sorry

end right_triangle_acute_angle_l583_58317


namespace marbles_problem_a_marbles_problem_b_l583_58371

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end marbles_problem_a_marbles_problem_b_l583_58371


namespace hyperbola_asymptotes_l583_58300

theorem hyperbola_asymptotes (a : ℝ) (x y : ℝ) (M : ℝ × ℝ) (F : ℝ × ℝ) :
  (∃ M : ℝ × ℝ, M.1 ^ 2 / a ^ 2 - M.2 ^ 2 = 1 ∧ M.2 ^ 2 = 8 * M.1 ∧ abs (dist M F) = 5) →
  (F.1 = 2 ∧ F.2 = 0) →
  (a = 3 / 5) → 
  (∀ x y : ℝ, (5 * x + 3 * y = 0) ∨ (5 * x - 3 * y = 0)) :=
by
  sorry

end hyperbola_asymptotes_l583_58300


namespace cost_per_person_l583_58333

def total_cost : ℕ := 30000  -- Cost in million dollars
def num_people : ℕ := 300    -- Number of people in million

theorem cost_per_person : total_cost / num_people = 100 :=
by
  sorry

end cost_per_person_l583_58333


namespace matrix_eigenvalue_problem_l583_58343

theorem matrix_eigenvalue_problem (k : ℝ) (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  ((3*x + 4*y = k*x) ∧ (6*x + 3*y = k*y)) → k = 3 :=
by
  sorry

end matrix_eigenvalue_problem_l583_58343


namespace no_such_quadratic_exists_l583_58303

theorem no_such_quadratic_exists : ¬ ∃ (b c : ℝ), 
  (∀ x : ℝ, 6 * x ≤ 3 * x^2 + 3 ∧ 3 * x^2 + 3 ≤ x^2 + b * x + c) ∧
  (x^2 + b * x + c = 1) :=
by
  sorry

end no_such_quadratic_exists_l583_58303


namespace max_xy_value_l583_58344

theorem max_xy_value (x y : ℕ) (h : 7 * x + 4 * y = 140) : xy ≤ 168 :=
by sorry

end max_xy_value_l583_58344


namespace solve_system_of_equations_l583_58367

theorem solve_system_of_equations :
    ∃ x y : ℚ, 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13 / 38 ∧ y = -4 / 19 :=
by
  sorry

end solve_system_of_equations_l583_58367


namespace area_of_quadrilateral_l583_58384

theorem area_of_quadrilateral (A B C : ℝ) (h1 : A + B = C) (h2 : A = 16) (h3 : B = 16) :
  (C - A - B) / 2 = 8 :=
by
  sorry

end area_of_quadrilateral_l583_58384


namespace sonya_fell_times_l583_58350

theorem sonya_fell_times (steven_falls : ℕ) (stephanie_falls : ℕ) (sonya_falls : ℕ) :
  steven_falls = 3 →
  stephanie_falls = steven_falls + 13 →
  sonya_falls = 6 →
  sonya_falls = (stephanie_falls / 2) - 2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at *
  sorry

end sonya_fell_times_l583_58350


namespace mean_score_juniors_is_103_l583_58323

noncomputable def mean_score_juniors : Prop :=
  ∃ (students juniors non_juniors m_j m_nj : ℝ),
  students = 160 ∧
  (students * 82) = (juniors * m_j + non_juniors * m_nj) ∧
  juniors = 0.4 * non_juniors ∧
  m_j = 1.4 * m_nj ∧
  m_j = 103

theorem mean_score_juniors_is_103 : mean_score_juniors :=
by
  sorry

end mean_score_juniors_is_103_l583_58323


namespace min_coach_handshakes_l583_58331

-- Definitions based on the problem conditions
def total_gymnasts : ℕ := 26
def total_handshakes : ℕ := 325

/- 
  The main theorem stating that the fewest number of handshakes 
  the coaches could have participated in is 0.
-/
theorem min_coach_handshakes (n : ℕ) (h : 0 ≤ n ∧ n * (n - 1) / 2 = total_handshakes) : 
  n = total_gymnasts → (total_handshakes - n * (n - 1) / 2) = 0 :=
by 
  intros h_n_eq_26
  sorry

end min_coach_handshakes_l583_58331


namespace largest_possible_b_b_eq_4_of_largest_l583_58391

theorem largest_possible_b (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) : b ≤ 4 := by
  sorry

theorem b_eq_4_of_largest (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) (hb : b = 4) : True := by
  sorry

end largest_possible_b_b_eq_4_of_largest_l583_58391


namespace exists_root_f_between_0_and_1_l583_58364

noncomputable def f (x : ℝ) : ℝ := 4 - 4 * x - Real.exp x

theorem exists_root_f_between_0_and_1 :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
sorry

end exists_root_f_between_0_and_1_l583_58364


namespace intersection_S_T_l583_58308

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_S_T : S ∩ T = T :=
by
  sorry

end intersection_S_T_l583_58308


namespace final_stamp_collection_l583_58375

section StampCollection

structure Collection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)
  (vehicles : ℕ)
  (famous_people : ℕ)

def initial_collections : Collection := {
  nature := 10, architecture := 15, animals := 12, vehicles := 6, famous_people := 4
}

-- define transactions as functions that take a collection and return a modified collection
def transaction1 (c : Collection) : Collection :=
  { c with nature := c.nature + 4, architecture := c.architecture + 5, animals := c.animals + 5, vehicles := c.vehicles + 2, famous_people := c.famous_people + 1 }

def transaction2 (c : Collection) : Collection := 
  { c with nature := c.nature + 2, animals := c.animals - 1 }

def transaction3 (c : Collection) : Collection := 
  { c with animals := c.animals - 5, architecture := c.architecture + 3 }

def transaction4 (c : Collection) : Collection :=
  { c with animals := c.animals - 4, nature := c.nature + 7 }

def transaction7 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles - 2, nature := c.nature + 5 }

def transaction8 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles + 3, famous_people := c.famous_people - 3 }

def final_collection (c : Collection) : Collection :=
  transaction8 (transaction7 (transaction4 (transaction3 (transaction2 (transaction1 c)))))

theorem final_stamp_collection :
  final_collection initial_collections = { nature := 28, architecture := 23, animals := 7, vehicles := 9, famous_people := 2 } :=
by
  -- skip the proof
  sorry

end StampCollection

end final_stamp_collection_l583_58375


namespace females_in_group_l583_58310

theorem females_in_group (n F M : ℕ) (Index_F Index_M : ℝ) 
  (h1 : n = 25) 
  (h2 : Index_F = (n - F) / n)
  (h3 : Index_M = (n - M) / n) 
  (h4 : Index_F - Index_M = 0.36) :
  F = 8 := 
by
  sorry

end females_in_group_l583_58310


namespace area_of_win_sector_l583_58386

theorem area_of_win_sector (r : ℝ) (p : ℝ) (A : ℝ) (h_1 : r = 10) (h_2 : p = 1 / 4) (h_3 : A = π * r^2) : 
  (p * A) = 25 * π := 
by
  sorry

end area_of_win_sector_l583_58386


namespace number_of_five_digit_numbers_with_one_odd_digit_l583_58335

def odd_digits : List ℕ := [1, 3, 5, 7, 9]
def even_digits : List ℕ := [0, 2, 4, 6, 8]

def five_digit_numbers_with_one_odd_digit : ℕ :=
  let num_1st_position := odd_digits.length * even_digits.length ^ 4
  let num_other_positions := 4 * odd_digits.length * (even_digits.length - 1) * (even_digits.length ^ 3)
  num_1st_position + num_other_positions

theorem number_of_five_digit_numbers_with_one_odd_digit :
  five_digit_numbers_with_one_odd_digit = 10625 :=
by
  sorry

end number_of_five_digit_numbers_with_one_odd_digit_l583_58335


namespace unique_point_intersection_l583_58316

theorem unique_point_intersection (k : ℝ) :
  (∃ x y, y = k * x + 2 ∧ y ^ 2 = 8 * x) → 
  ((k = 0) ∨ (k = 1)) :=
by {
  sorry
}

end unique_point_intersection_l583_58316


namespace athenas_min_wins_l583_58361

theorem athenas_min_wins (total_games : ℕ) (games_played : ℕ) (wins_so_far : ℕ) (losses_so_far : ℕ) 
                          (win_percentage_threshold : ℝ) (remaining_games : ℕ) (additional_wins_needed : ℕ) :
  total_games = 44 ∧ games_played = wins_so_far + losses_so_far ∧ wins_so_far = 20 ∧ losses_so_far = 15 ∧ 
  win_percentage_threshold = 0.6 ∧ remaining_games = total_games - games_played ∧ additional_wins_needed = 27 - wins_so_far → 
  additional_wins_needed = 7 :=
by
  sorry

end athenas_min_wins_l583_58361


namespace geometric_sequence_common_ratio_l583_58326

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q)
  (h0 : a 1 = 2) (h1 : a 4 = 1 / 4) : q = 1 / 2 :=
by
  sorry

end geometric_sequence_common_ratio_l583_58326


namespace rabbit_calories_l583_58360

theorem rabbit_calories (C : ℕ) :
  (6 * 300 = 2 * C + 200) → C = 800 :=
by
  intro h
  sorry

end rabbit_calories_l583_58360


namespace minimum_value_of_expression_l583_58354

theorem minimum_value_of_expression (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 := 
sorry

end minimum_value_of_expression_l583_58354


namespace greatest_possible_remainder_l583_58372

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 9 ∧ x % 9 = r ∧ r = 8 :=
by
  use 8
  sorry -- Proof to be filled in

end greatest_possible_remainder_l583_58372


namespace virginia_taught_fewer_years_l583_58309

variable (V A : ℕ)

theorem virginia_taught_fewer_years (h1 : V + A + 40 = 93) (h2 : V = A + 9) : 40 - V = 9 := by
  sorry

end virginia_taught_fewer_years_l583_58309


namespace rectangle_perimeter_is_28_l583_58387

-- Define the variables and conditions
variables (h w : ℝ)

-- Problem conditions
def rectangle_area (h w : ℝ) : Prop := h * w = 40
def width_greater_than_twice_height (h w : ℝ) : Prop := w > 2 * h
def parallelogram_area (h w : ℝ) : Prop := h * (w - h) = 24

-- The theorem stating the perimeter of the rectangle given the conditions
theorem rectangle_perimeter_is_28 (h w : ℝ) 
  (H1 : rectangle_area h w) 
  (H2 : width_greater_than_twice_height h w) 
  (H3 : parallelogram_area h w) :
  2 * h + 2 * w = 28 :=
sorry

end rectangle_perimeter_is_28_l583_58387


namespace visiting_plans_count_l583_58315

-- Let's define the exhibitions
inductive Exhibition
| OperaCultureExhibition
| MingDynastyImperialCellarPorcelainExhibition
| AncientGreenLandscapePaintingExhibition
| ZhaoMengfuCalligraphyAndPaintingExhibition

open Exhibition

-- The condition is that the student must visit at least one painting exhibition in the morning and another in the afternoon
-- Proof that the number of different visiting plans is 10.
theorem visiting_plans_count :
  let exhibitions := [OperaCultureExhibition, MingDynastyImperialCellarPorcelainExhibition, AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  let painting_exhibitions := [AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  ∃ visits : List (Exhibition × Exhibition), (∀ (m a : Exhibition), (m ∈ painting_exhibitions ∨ a ∈ painting_exhibitions)) → visits.length = 10 :=
sorry

end visiting_plans_count_l583_58315


namespace remaining_storage_space_l583_58373

/-- Given that 1 GB = 1024 MB, a hard drive with 300 GB of total storage,
and 300000 MB of used storage, prove that the remaining storage space is 7200 MB. -/
theorem remaining_storage_space (total_gb : ℕ) (mb_per_gb : ℕ) (used_mb : ℕ) :
  total_gb = 300 → mb_per_gb = 1024 → used_mb = 300000 →
  (total_gb * mb_per_gb - used_mb) = 7200 :=
by
  intros h1 h2 h3
  sorry

end remaining_storage_space_l583_58373


namespace area_triangle_BQW_l583_58397

theorem area_triangle_BQW (ABCD : Rectangle) (AZ WC : ℝ) (AB : ℝ)
    (area_trapezoid_ZWCD : ℝ) :
    AZ = WC ∧ AZ = 6 ∧ AB = 12 ∧ area_trapezoid_ZWCD = 120 →
    (1/2) * ((120) - (1/2) * 6 * 12) = 42 :=
by
  intros
  sorry

end area_triangle_BQW_l583_58397


namespace scientific_notation_10200000_l583_58325

theorem scientific_notation_10200000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ |a| ∧ |a| < 10 ∧ 10.2 * 10^7 = a * 10^n := 
sorry

end scientific_notation_10200000_l583_58325


namespace ratio_of_radii_of_touching_circles_l583_58359

theorem ratio_of_radii_of_touching_circles
  (r R : ℝ) (A B C D : ℝ) (h1 : A + B + C = D)
  (h2 : 3 * A = 7 * B)
  (h3 : 7 * B = 2 * C)
  (h4 : R = D / 2)
  (h5 : B = R - 3 * A)
  (h6 : C = R - 2 * A)
  (h7 : r = 4 * A)
  (h8 : R = 6 * A) :
  R / r = 3 / 2 := by
  sorry

end ratio_of_radii_of_touching_circles_l583_58359


namespace men_in_group_initial_l583_58348

variable (M : ℕ)  -- Initial number of men in the group
variable (A : ℕ)  -- Initial average age of the group

theorem men_in_group_initial : (2 * 50 - (18 + 22) = 60) → ((M + 6) = 60 / 6) → (M = 10) :=
by
  sorry

end men_in_group_initial_l583_58348


namespace platform_length_l583_58305

theorem platform_length
  (train_length : ℕ)
  (time_pole : ℕ)
  (time_platform : ℕ)
  (h_train_length : train_length = 300)
  (h_time_pole : time_pole = 18)
  (h_time_platform : time_platform = 39) :
  ∃ (platform_length : ℕ), platform_length = 350 :=
by
  sorry

end platform_length_l583_58305


namespace sport_formulation_water_l583_58324

theorem sport_formulation_water
  (f c w : ℕ)  -- flavoring, corn syrup, and water respectively in standard formulation
  (f_s c_s w_s : ℕ)  -- flavoring, corn syrup, and water respectively in sport formulation
  (corn_syrup_sport : ℤ) -- amount of corn syrup in sport formulation in ounces
  (h_std_ratio : f = 1 ∧ c = 12 ∧ w = 30) -- given standard formulation ratios
  (h_sport_fc_ratio : f_s * 4 = c_s) -- sport formulation flavoring to corn syrup ratio
  (h_sport_fw_ratio : f_s * 60 = w_s) -- sport formulation flavoring to water ratio
  (h_corn_syrup_sport : c_s = corn_syrup_sport) -- amount of corn syrup in sport formulation
  : w_s = 30 := 
by 
  sorry

end sport_formulation_water_l583_58324


namespace B_oxen_count_l583_58327

/- 
  A puts 10 oxen for 7 months.
  B puts some oxen for 5 months.
  C puts 15 oxen for 3 months.
  The rent of the pasture is Rs. 175.
  C should pay Rs. 45 as his share of rent.
  We need to prove that B put 12 oxen for grazing.
-/

def oxen_months (oxen : ℕ) (months : ℕ) : ℕ := oxen * months

def A_ox_months := oxen_months 10 7
def C_ox_months := oxen_months 15 3

def total_rent : ℕ := 175
def C_rent_share : ℕ := 45

theorem B_oxen_count (x : ℕ) : 
  (C_rent_share : ℝ) / total_rent = (C_ox_months : ℝ) / (A_ox_months + 5 * x + C_ox_months) →
  x = 12 := 
by
  sorry

end B_oxen_count_l583_58327


namespace range_m_distinct_roots_l583_58332

theorem range_m_distinct_roots (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (4^x₁ - m * 2^(x₁+1) + 2 - m = 0) ∧ (4^x₂ - m * 2^(x₂+1) + 2 - m = 0)) ↔ 1 < m ∧ m < 2 :=
by
  sorry

end range_m_distinct_roots_l583_58332


namespace wrapping_paper_area_l583_58358

theorem wrapping_paper_area (a : ℝ) (h : ℝ) : h = a ∧ 1 ≥ 0 → 4 * a^2 = 4 * a^2 :=
by sorry

end wrapping_paper_area_l583_58358


namespace catch_two_salmon_l583_58322

def totalTroutWeight : ℕ := 8
def numBass : ℕ := 6
def weightPerBass : ℕ := 2
def totalBassWeight : ℕ := numBass * weightPerBass
def campers : ℕ := 22
def weightPerCamper : ℕ := 2
def totalFishWeightRequired : ℕ := campers * weightPerCamper
def totalTroutAndBassWeight : ℕ := totalTroutWeight + totalBassWeight
def additionalFishWeightRequired : ℕ := totalFishWeightRequired - totalTroutAndBassWeight
def weightPerSalmon : ℕ := 12
def numSalmon : ℕ := additionalFishWeightRequired / weightPerSalmon

theorem catch_two_salmon : numSalmon = 2 := by
  sorry

end catch_two_salmon_l583_58322


namespace three_digit_number_l583_58311

-- Define the variables involved.
variables (a b c n : ℕ)

-- Condition 1: c = 3a
def condition1 (a c : ℕ) : Prop := c = 3 * a

-- Condition 2: n is three-digit number constructed from a, b, and c.
def is_three_digit (a b c n : ℕ) : Prop := n = 100 * a + 10 * b + c

-- Condition 3: n leaves a remainder of 4 when divided by 5.
def condition2 (n : ℕ) : Prop := n % 5 = 4

-- Condition 4: n leaves a remainder of 3 when divided by 11.
def condition3 (n : ℕ) : Prop := n % 11 = 3

-- Define the main theorem
theorem three_digit_number (a b c n : ℕ) 
(h1: condition1 a c) 
(h2: is_three_digit a b c n) 
(h3: condition2 n) 
(h4: condition3 n) : 
n = 359 := 
sorry

end three_digit_number_l583_58311


namespace training_weeks_l583_58365

variable (adoption_fee training_per_week cert_cost insurance_coverage out_of_pocket : ℕ)
variable (x : ℕ)

def adoption_fee_value : ℕ := 150
def training_per_week_cost : ℕ := 250
def certification_cost_value : ℕ := 3000
def insurance_coverage_percentage : ℕ := 90
def total_out_of_pocket : ℕ := 3450

theorem training_weeks :
  adoption_fee = adoption_fee_value →
  training_per_week = training_per_week_cost →
  cert_cost = certification_cost_value →
  insurance_coverage = insurance_coverage_percentage →
  out_of_pocket = total_out_of_pocket →
  (out_of_pocket = adoption_fee + training_per_week * x + (cert_cost * (100 - insurance_coverage)) / 100) →
  x = 12 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5] at h6
  sorry

end training_weeks_l583_58365


namespace codger_feet_l583_58377

theorem codger_feet (F : ℕ) (h1 : 6 = 2 * (5 - 1) * F) : F = 3 := by
  sorry

end codger_feet_l583_58377


namespace Jack_gave_Mike_six_notebooks_l583_58357

theorem Jack_gave_Mike_six_notebooks :
  ∀ (Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike : ℕ),
  Gerald_notebooks = 8 →
  Jack_notebooks_left = 10 →
  notebooks_given_to_Paula = 5 →
  total_notebooks_initial = Gerald_notebooks + 13 →
  jack_notebooks_after_Paula = total_notebooks_initial - notebooks_given_to_Paula →
  notebooks_given_to_Mike = jack_notebooks_after_Paula - Jack_notebooks_left →
  notebooks_given_to_Mike = 6 :=
by
  intros Gerald_notebooks Jack_notebooks_left notebooks_given_to_Paula total_notebooks_initial jack_notebooks_after_Paula notebooks_given_to_Mike
  intros Gerald_notebooks_eq Jack_notebooks_left_eq notebooks_given_to_Paula_eq total_notebooks_initial_eq jack_notebooks_after_Paula_eq notebooks_given_to_Mike_eq
  sorry

end Jack_gave_Mike_six_notebooks_l583_58357


namespace expenditure_on_house_rent_l583_58340

variable (X : ℝ) -- Let X be Bhanu's total income in rupees

-- Condition 1: Bhanu spends 300 rupees on petrol, which is 30% of his income
def condition_on_petrol : Prop := 0.30 * X = 300

-- Definition of remaining income
def remaining_income : ℝ := X - 300

-- Definition of house rent expenditure: 10% of remaining income
def house_rent : ℝ := 0.10 * remaining_income X

-- Theorem: If the condition on petrol holds, then the house rent expenditure is 70 rupees
theorem expenditure_on_house_rent (h : condition_on_petrol X) : house_rent X = 70 :=
  sorry

end expenditure_on_house_rent_l583_58340


namespace crackers_eaten_by_Daniel_and_Elsie_l583_58320

theorem crackers_eaten_by_Daniel_and_Elsie :
  ∀ (initial_crackers remaining_crackers eaten_by_Ally eaten_by_Bob eaten_by_Clair: ℝ),
    initial_crackers = 27.5 →
    remaining_crackers = 10.5 →
    eaten_by_Ally = 3.5 →
    eaten_by_Bob = 4.0 →
    eaten_by_Clair = 5.5 →
    initial_crackers - remaining_crackers = (eaten_by_Ally + eaten_by_Bob + eaten_by_Clair) + (4 : ℝ) :=
by sorry

end crackers_eaten_by_Daniel_and_Elsie_l583_58320


namespace Denise_age_l583_58376

-- Define the ages of Amanda, Carlos, Beth, and Denise
variables (A C B D : ℕ)

-- State the given conditions
def condition1 := A = C - 4
def condition2 := C = B + 5
def condition3 := D = B + 2
def condition4 := A = 16

-- The theorem to prove
theorem Denise_age (A C B D : ℕ) (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 D B) (h4 : condition4 A) : D = 17 :=
by
  sorry

end Denise_age_l583_58376


namespace solve_for_m_l583_58355

theorem solve_for_m (n : ℝ) (m : ℝ) (h : 21 * (m + n) + 21 = 21 * (-m + n) + 21) : m = 1 / 2 := 
sorry

end solve_for_m_l583_58355


namespace cyclic_sum_inequality_l583_58321

noncomputable def cyclic_sum (f : ℝ → ℝ → ℝ) (x y z : ℝ) : ℝ :=
  f x y + f y z + f z x

theorem cyclic_sum_inequality
  (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x = a + (1 / b) - 1) 
  (hy : y = b + (1 / c) - 1) 
  (hz : z = c + (1 / a) - 1)
  (hpx : x > 0) (hpy : y > 0) (hpz : z > 0) :
  cyclic_sum (fun x y => (x * y) / (Real.sqrt (x * y) + 2)) x y z ≥ 1 :=
sorry

end cyclic_sum_inequality_l583_58321


namespace find_S_l583_58393

theorem find_S (a b : ℝ) (R : ℝ) (S : ℝ)
  (h1 : a + b = R) 
  (h2 : a^2 + b^2 = 12)
  (h3 : R = 2)
  (h4 : S = a^3 + b^3) : S = 32 :=
by
  sorry

end find_S_l583_58393


namespace wholesome_bakery_loaves_on_wednesday_l583_58337

theorem wholesome_bakery_loaves_on_wednesday :
  ∀ (L_wed L_thu L_fri L_sat L_sun L_mon : ℕ),
    L_thu = 7 →
    L_fri = 10 →
    L_sat = 14 →
    L_sun = 19 →
    L_mon = 25 →
    L_thu - L_wed = 2 →
    L_wed = 5 :=
by intros L_wed L_thu L_fri L_sat L_sun L_mon;
   intros H_thu H_fri H_sat H_sun H_mon H_diff;
   sorry

end wholesome_bakery_loaves_on_wednesday_l583_58337


namespace valentines_proof_l583_58368

-- Definitions of the conditions in the problem
def original_valentines : ℝ := 58.5
def remaining_valentines : ℝ := 16.25
def valentines_given : ℝ := 42.25

-- The statement that we need to prove
theorem valentines_proof : original_valentines - remaining_valentines = valentines_given := by
  sorry

end valentines_proof_l583_58368


namespace set_problems_l583_58347

def U : Set ℤ := {x | 0 < x ∧ x ≤ 10}
def A : Set ℤ := {1, 2, 4, 5, 9}
def B : Set ℤ := {4, 6, 7, 8, 10}

theorem set_problems :
  (A ∩ B = ({4} : Set ℤ)) ∧
  (A ∪ B = ({1, 2, 4, 5, 6, 7, 8, 9, 10} : Set ℤ)) ∧
  (U \ (A ∪ B) = ({3} : Set ℤ)) ∧
  ((U \ A) ∩ (U \ B) = ({3} : Set ℤ)) :=
by
  sorry

end set_problems_l583_58347


namespace smallest_non_representable_number_l583_58304

theorem smallest_non_representable_number :
  ∀ n : ℕ, (∀ a b c d : ℕ, n = (2^a - 2^b) / (2^c - 2^d) → n < 11) ∧
           (∀ a b c d : ℕ, 11 ≠ (2^a - 2^b) / (2^c - 2^d)) :=
sorry

end smallest_non_representable_number_l583_58304


namespace find_borrowed_interest_rate_l583_58380

theorem find_borrowed_interest_rate :
  ∀ (principal : ℝ) (time : ℝ) (lend_rate : ℝ) (gain_per_year : ℝ) (borrow_rate : ℝ),
  principal = 5000 →
  time = 1 → -- Considering per year
  lend_rate = 0.06 →
  gain_per_year = 100 →
  (principal * lend_rate - gain_per_year = principal * borrow_rate * time) →
  borrow_rate * 100 = 4 :=
by
  intros principal time lend_rate gain_per_year borrow_rate h_principal h_time h_lend_rate h_gain h_equation
  rw [h_principal, h_time, h_lend_rate] at h_equation
  have h_borrow_rate := h_equation
  sorry

end find_borrowed_interest_rate_l583_58380


namespace cos_seventh_eq_sum_of_cos_l583_58370

theorem cos_seventh_eq_sum_of_cos:
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
  (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) ∧
  (b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 + b₇ ^ 2 = 1555 / 4096) :=
sorry

end cos_seventh_eq_sum_of_cos_l583_58370


namespace susan_average_speed_l583_58363

noncomputable def average_speed_trip (d1 d2 : ℝ) (v1 v2 : ℝ) : ℝ :=
  let total_distance := d1 + d2
  let time1 := d1 / v1
  let time2 := d2 / v2
  let total_time := time1 + time2
  total_distance / total_time

theorem susan_average_speed :
  average_speed_trip 60 30 30 60 = 36 := 
by
  -- The proof can be filled in here
  sorry

end susan_average_speed_l583_58363


namespace minimum_15_equal_differences_l583_58396

-- Definition of distinct integers a_i
def distinct_sequence (a : Fin 100 → ℕ) : Prop :=
  ∀ i j : Fin 100, i < j → a i < a j

-- Definition of the differences d_i
def differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, d i = a ⟨i + 1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ i.2) (by norm_num)⟩ - a i

-- Main theorem statement
theorem minimum_15_equal_differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) :
  (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 400) →
  distinct_sequence a →
  differences a d →
  ∃ t : Finset ℕ, t.card ≥ 15 ∧ ∀ x : ℕ, x ∈ t → (∃ i j : Fin 99, i ≠ j ∧ d i = x ∧ d j = x) :=
sorry

end minimum_15_equal_differences_l583_58396


namespace area_of_ring_between_concentric_circles_l583_58339

theorem area_of_ring_between_concentric_circles :
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  area_ring = 95 * Real.pi :=
by
  let radius_large := 12
  let radius_small := 7
  let area_large := Real.pi * radius_large^2
  let area_small := Real.pi * radius_small^2
  let area_ring := area_large - area_small
  show area_ring = 95 * Real.pi
  sorry

end area_of_ring_between_concentric_circles_l583_58339


namespace total_points_other_7_members_is_15_l583_58342

variable (x y : ℕ)
variable (h1 : y ≤ 21)
variable (h2 : y = x * 7 / 15 - 18)
variable (h3 : (1 / 3) * x + (1 / 5) * x + 18 + y = x)

theorem total_points_other_7_members_is_15 (h : x * 7 % 15 = 0) : y = 15 :=
by
  sorry

end total_points_other_7_members_is_15_l583_58342


namespace trajectory_equation_find_m_l583_58398

-- Define points A and B.
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for P:
def P_condition (P : ℝ × ℝ) : Prop :=
  let PA_len := Real.sqrt ((P.1 - 1)^2 + P.2^2)
  let AB_len := Real.sqrt ((1 - (-1))^2 + (0 - 0)^2)
  let PB_dot_AB := (P.1 + 1) * (-2)
  PA_len * AB_len = PB_dot_AB

-- Problem (1): The trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) (hP : P_condition P) : P.2^2 = 4 * P.1 :=
sorry

-- Define orthogonality condition
def orthogonal (M N : ℝ × ℝ) : Prop := 
  let OM := M
  let ON := N
  OM.1 * ON.1 + OM.2 * ON.2 = 0

-- Problem (2): Finding the value of m
theorem find_m (m : ℝ) (hm1 : m ≠ 0) (hm2 : m < 1) 
  (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (M N : ℝ × ℝ) (hM : M.2 = M.1 + m) (hN : N.2 = N.1 + m)
  (hMN : orthogonal M N) : m = -4 :=
sorry

end trajectory_equation_find_m_l583_58398


namespace min_f_value_inequality_solution_l583_58356

theorem min_f_value (x : ℝ) : |x+7| + |x-1| ≥ 8 := by
  sorry

theorem inequality_solution (x : ℝ) (m : ℝ) (h : m = 8) : |x-3| - 2*x ≤ 2*m - 12 ↔ x ≥ -1/3 := by
  sorry

end min_f_value_inequality_solution_l583_58356


namespace coeff_abs_sum_eq_729_l583_58353

-- Given polynomial (2x - 1)^6 expansion
theorem coeff_abs_sum_eq_729 (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (2 * x - 1) ^ 6 = a_6 * x ^ 6 + a_5 * x ^ 5 + a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + a_0 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end coeff_abs_sum_eq_729_l583_58353


namespace larger_of_two_numbers_l583_58369

theorem larger_of_two_numbers (hcf : ℕ) (f1 : ℕ) (f2 : ℕ) 
(h_hcf : hcf = 10) 
(h_f1 : f1 = 11) 
(h_f2 : f2 = 15) 
: max (hcf * f1) (hcf * f2) = 150 :=
by
  have lcm := hcf * f1 * f2
  have num1 := hcf * f1
  have num2 := hcf * f2
  sorry

end larger_of_two_numbers_l583_58369


namespace cubic_polynomial_evaluation_l583_58388

theorem cubic_polynomial_evaluation (Q : ℚ → ℚ) (m : ℚ)
  (hQ0 : Q 0 = 2 * m) 
  (hQ1 : Q 1 = 5 * m) 
  (hQm1 : Q (-1) = 0) : 
  Q 2 + Q (-2) = 8 * m := 
by
  sorry

end cubic_polynomial_evaluation_l583_58388


namespace reciprocal_of_neg_2023_l583_58302

theorem reciprocal_of_neg_2023 : 1 / (-2023) = - (1 / 2023) :=
by 
  -- The proof is omitted.
  sorry

end reciprocal_of_neg_2023_l583_58302


namespace quadratic_b_value_l583_58306

theorem quadratic_b_value {b m : ℝ} (h : ∀ x, x^2 + b * x + 44 = (x + m)^2 + 8) : b = 12 :=
by
  -- hint for proving: expand (x+m)^2 + 8 and equate it with x^2 + bx + 44 to solve for b 
  sorry

end quadratic_b_value_l583_58306


namespace length_BC_l583_58366

noncomputable def center (O : Type) : Prop := sorry   -- Center of the circle.

noncomputable def diameter (AD : Type) : Prop := sorry   -- AD is a diameter.

noncomputable def chord (ABC : Type) : Prop := sorry   -- ABC is a chord.

noncomputable def radius_equal (BO : ℝ) : Prop := BO = 8   -- BO = 8.

noncomputable def angle_ABO (α : ℝ) : Prop := α = 45   -- ∠ABO = 45°.

noncomputable def arc_CD (β : ℝ) : Prop := β = 90   -- Arc CD subtended by ∠AOD = 90°.

theorem length_BC (O AD ABC : Type) (BO : ℝ) (α β γ : ℝ)
  (h1 : center O)
  (h2 : diameter AD)
  (h3 : chord ABC)
  (h4 : radius_equal BO)
  (h5 : angle_ABO α)
  (h6 : arc_CD β)
  : γ = 8 := 
sorry

end length_BC_l583_58366


namespace sum_of_coefficients_l583_58352

theorem sum_of_coefficients :
  ∃ a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ,
  (∀ x : ℤ, (2 - x)^7 = a₀ + a₁ * (1 + x)^2 + a₂ * (1 + x)^3 + a₃ * (1 + x)^4 + a₄ * (1 + x)^5 + a₅ * (1 + x)^6 + a₆ * (1 + x)^7 + a₇ * (1 + x)^8) →
  a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = 129 := by sorry

end sum_of_coefficients_l583_58352


namespace find_angle_A_l583_58301
open Real

theorem find_angle_A
  (a b : ℝ)
  (A B : ℝ)
  (h1 : b = 2 * a)
  (h2 : B = A + 60) :
  A = 30 :=
by 
  sorry

end find_angle_A_l583_58301


namespace negation_of_universal_statement_l583_58374

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by 
  -- Proof steps would be added here
  sorry

end negation_of_universal_statement_l583_58374
