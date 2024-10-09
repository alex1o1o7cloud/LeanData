import Mathlib

namespace max_a_correct_answers_l2424_242406

theorem max_a_correct_answers : 
  ∃ (a b c x y z w : ℕ), 
  a + b + c + x + y + z + w = 39 ∧
  a = b + c ∧
  (a + x + y + w) = a + 5 + (x + y + w) ∧
  b + z = 2 * (c + z) ∧
  23 ≤ a :=
sorry

end max_a_correct_answers_l2424_242406


namespace parallelogram_area_increase_l2424_242453
open Real

/-- The area of the parallelogram increases by 600 square meters when the base is increased by 20 meters. -/
theorem parallelogram_area_increase :
  ∀ (base height new_base : ℝ), 
    base = 65 → height = 30 → new_base = base + 20 → 
    (new_base * height - base * height) = 600 := 
by
  sorry

end parallelogram_area_increase_l2424_242453


namespace solve_inequality_l2424_242495

theorem solve_inequality (a x : ℝ) (h : a < 0) :
  (56 * x^2 + a * x - a^2 < 0) ↔ (a / 8 < x ∧ x < -a / 7) :=
by
  sorry

end solve_inequality_l2424_242495


namespace alex_ride_time_l2424_242454

theorem alex_ride_time
  (T : ℝ) -- time on flat ground
  (flat_speed : ℝ := 20) -- flat ground speed
  (uphill_speed : ℝ := 12) -- uphill speed
  (uphill_time : ℝ := 2.5) -- uphill time
  (downhill_speed : ℝ := 24) -- downhill speed
  (downhill_time : ℝ := 1.5) -- downhill time
  (walk_distance : ℝ := 8) -- distance walked
  (total_distance : ℝ := 164) -- total distance to the town
  (hup : uphill_speed * uphill_time = 30)
  (hdown : downhill_speed * downhill_time = 36)
  (hwalk : walk_distance = 8) :
  flat_speed * T + 30 + 36 + 8 = total_distance → T = 4.5 :=
by
  intros h
  sorry

end alex_ride_time_l2424_242454


namespace marcus_calzones_total_time_l2424_242405

theorem marcus_calzones_total_time :
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  total_time = 124 :=
by
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  sorry

end marcus_calzones_total_time_l2424_242405


namespace bethany_saw_16_portraits_l2424_242428

variable (P S : ℕ)

def bethany_conditions : Prop :=
  S = 4 * P ∧ P + S = 80

theorem bethany_saw_16_portraits (P S : ℕ) (h : bethany_conditions P S) : P = 16 := by
  sorry

end bethany_saw_16_portraits_l2424_242428


namespace pencil_cost_l2424_242427

theorem pencil_cost (P : ℝ) (h1 : 24 * P + 18 = 30) : P = 0.5 :=
by
  sorry

end pencil_cost_l2424_242427


namespace circle_equation_l2424_242401

-- Define the conditions
def chord_length_condition (a b r : ℝ) : Prop := r^2 = a^2 + 1
def arc_length_condition (b r : ℝ) : Prop := r^2 = 2 * b^2
def min_distance_condition (a b : ℝ) : Prop := a = b

-- The main theorem stating the final answer
theorem circle_equation (a b r : ℝ) (h1 : chord_length_condition a b r)
    (h2 : arc_length_condition b r) (h3 : min_distance_condition a b) :
    ((x - a)^2 + (y - a)^2 = 2) ∨ ((x + a)^2 + (y + a)^2 = 2) :=
sorry

end circle_equation_l2424_242401


namespace arithmetic_sequence_sum_l2424_242457

noncomputable def S (n : ℕ) : ℤ :=
  n * (-2012) + n * (n - 1) / 2 * (1 : ℤ)

theorem arithmetic_sequence_sum :
  (S 2012) / 2012 - (S 10) / 10 = 2002 → S 2017 = 2017 :=
by
  sorry

end arithmetic_sequence_sum_l2424_242457


namespace spinner_final_direction_l2424_242491

theorem spinner_final_direction 
  (initial_direction : ℕ) -- 0 for north, 1 for east, 2 for south, 3 for west
  (clockwise_revolutions : ℚ)
  (counterclockwise_revolutions : ℚ)
  (net_revolutions : ℚ) -- derived via net movement calculation
  (final_position : ℕ) -- correct position after net movement
  : initial_direction = 3 → clockwise_revolutions = 9/4 → counterclockwise_revolutions = 15/4 → final_position = 1 :=
by
  sorry

end spinner_final_direction_l2424_242491


namespace part1_part2_l2424_242404

-- Problem part (1)
theorem part1 : (Real.sqrt 12 + Real.sqrt (4 / 3)) * Real.sqrt 3 = 8 := 
  sorry

-- Problem part (2)
theorem part2 : Real.sqrt 48 - Real.sqrt 54 / Real.sqrt 2 + (3 - Real.sqrt 3) * (3 + Real.sqrt 3) = Real.sqrt 3 + 6 := 
  sorry

end part1_part2_l2424_242404


namespace job_completion_l2424_242479

theorem job_completion (A_rate D_rate : ℝ) (h₁ : A_rate = 1 / 12) (h₂ : A_rate + D_rate = 1 / 4) : D_rate = 1 / 6 := 
by 
  sorry

end job_completion_l2424_242479


namespace chords_from_nine_points_l2424_242456

theorem chords_from_nine_points : 
  ∀ (n r : ℕ), n = 9 → r = 2 → (Nat.choose n r) = 36 :=
by
  intros n r hn hr
  rw [hn, hr]
  -- Goal: Nat.choose 9 2 = 36
  sorry

end chords_from_nine_points_l2424_242456


namespace toilet_paper_production_per_day_l2424_242459

theorem toilet_paper_production_per_day 
    (total_production_march : ℕ)
    (days_in_march : ℕ)
    (increase_factor : ℕ)
    (total_production : ℕ)
    (days : ℕ)
    (increase : ℕ)
    (production : ℕ) :
    total_production_march = total_production →
    days_in_march = days →
    increase_factor = increase →
    total_production = 868000 →
    days = 31 →
    increase = 3 →
    production = total_production / days →
    production / increase = 9333
:= by
  intros h1 h2 h3 h4 h5 h6 h7

  sorry

end toilet_paper_production_per_day_l2424_242459


namespace solve_for_y_l2424_242409

theorem solve_for_y (y : ℚ) (h : (4 / 7) * (1 / 5) * y - 2 = 10) : y = 105 := by
  sorry

end solve_for_y_l2424_242409


namespace find_s_l2424_242444

def f (x s : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + 4 * x + s

theorem find_s (s : ℝ) : f (-1) s = 0 → s = 9 :=
by
  sorry

end find_s_l2424_242444


namespace polynomial_power_degree_l2424_242460

noncomputable def polynomial_degree (p : Polynomial ℝ) : ℕ := p.natDegree

theorem polynomial_power_degree : 
  polynomial_degree ((5 * X^3 - 4 * X + 7)^10) = 30 := by
  sorry

end polynomial_power_degree_l2424_242460


namespace arrange_books_l2424_242442

-- We define the conditions about the number of books
def num_algebra_books : ℕ := 4
def num_calculus_books : ℕ := 5
def total_books : ℕ := num_algebra_books + num_calculus_books

-- The combination function which calculates binomial coefficients
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem stating that there are 126 ways to arrange the books
theorem arrange_books : combination total_books num_algebra_books = 126 :=
  by
    sorry

end arrange_books_l2424_242442


namespace geometric_sequence_formula_l2424_242429

noncomputable def a_n (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  a_1 * q^n

theorem geometric_sequence_formula
  (a_1 q : ℝ)
  (h_pos : ∀ n : ℕ, a_n a_1 q n > 0)
  (h_4_eq : a_n a_1 q 4 = (a_n a_1 q 2)^2)
  (h_2_4_sum : a_n a_1 q 2 + a_n a_1 q 4 = 5 / 16) :
  ∀ n : ℕ, a_n a_1 q n = ((1 : ℝ) / 2) ^ n :=
sorry

end geometric_sequence_formula_l2424_242429


namespace work_rate_problem_l2424_242471

theorem work_rate_problem
  (W : ℕ) -- total work
  (A_rate : ℕ) -- A's work rate in days
  (B_rate : ℕ) -- B's work rate in days
  (x : ℕ) -- days A worked alone
  (total_days : ℕ) -- days A and B worked together
  (hA : A_rate = 12) -- A can do the work in 12 days
  (hB : B_rate = 6) -- B can do the work in 6 days
  (hx : total_days = 3) -- remaining days they together work
  : x = 3 := 
by
  sorry

end work_rate_problem_l2424_242471


namespace stratified_sampling_junior_teachers_l2424_242450

theorem stratified_sampling_junior_teachers 
    (total_teachers : ℕ) (senior_teachers : ℕ) 
    (intermediate_teachers : ℕ) (junior_teachers : ℕ) 
    (sample_size : ℕ) 
    (H1 : total_teachers = 200)
    (H2 : senior_teachers = 20)
    (H3 : intermediate_teachers = 100)
    (H4 : junior_teachers = 80) 
    (H5 : sample_size = 50)
    : (junior_teachers * sample_size / total_teachers = 20) := 
  by 
    sorry

end stratified_sampling_junior_teachers_l2424_242450


namespace three_two_three_zero_zero_zero_zero_in_scientific_notation_l2424_242412

theorem three_two_three_zero_zero_zero_zero_in_scientific_notation :
  3230000 = 3.23 * 10^6 :=
sorry

end three_two_three_zero_zero_zero_zero_in_scientific_notation_l2424_242412


namespace cost_of_rope_l2424_242489

theorem cost_of_rope : 
  ∀ (total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost : ℝ),
  total_money = 200 ∧
  sheet_cost = 42 ∧
  propane_burner_cost = 14 ∧
  helium_cost_per_ounce = 1.50 ∧
  helium_per_foot = 113 ∧
  max_height = 9492 ∧
  rope_cost = total_money - (sheet_cost + propane_burner_cost + (max_height / helium_per_foot) * helium_cost_per_ounce) →
  rope_cost = 18 :=
by
  intros total_money sheet_cost propane_burner_cost helium_cost_per_ounce helium_per_foot max_height rope_cost
  rintro ⟨h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max, h_rope⟩
  rw [h_total, h_sheet, h_propane, h_helium, h_perfoot, h_max] at h_rope
  simp only [inv_mul_eq_iff_eq_mul, div_eq_mul_inv] at h_rope
  norm_num at h_rope
  sorry

end cost_of_rope_l2424_242489


namespace remainder_is_cx_plus_d_l2424_242499

-- Given a polynomial Q, assume the following conditions
variables {Q : ℕ → ℚ}

-- Conditions
axiom condition1 : Q 15 = 12
axiom condition2 : Q 10 = 4

theorem remainder_is_cx_plus_d : 
  ∃ c d, (c = 8 / 5) ∧ (d = -12) ∧ 
          ∀ x, Q x % ((x - 10) * (x - 15)) = c * x + d :=
by
  sorry

end remainder_is_cx_plus_d_l2424_242499


namespace remainder_76_pow_77_mod_7_l2424_242496

theorem remainder_76_pow_77_mod_7 : (76 ^ 77) % 7 = 6 := 
by 
  sorry 

end remainder_76_pow_77_mod_7_l2424_242496


namespace total_number_of_coins_is_324_l2424_242464

noncomputable def total_coins (total_sum : ℕ) (coins_20p : ℕ) (coins_25p_value : ℕ) : ℕ :=
    coins_20p + (coins_25p_value / 25)

theorem total_number_of_coins_is_324 (h_sum: 7100 = 71 * 100) (h_coins_20p: 200 * 20 = 4000) :
  total_coins 7100 200 3100 = 324 := by
  sorry

end total_number_of_coins_is_324_l2424_242464


namespace problem_1_problem_2_l2424_242419

noncomputable def a : ℝ := Real.sqrt 7 + 2
noncomputable def b : ℝ := Real.sqrt 7 - 2

theorem problem_1 : a^2 * b + b^2 * a = 6 * Real.sqrt 7 := by
  sorry

theorem problem_2 : a^2 + a * b + b^2 = 25 := by
  sorry

end problem_1_problem_2_l2424_242419


namespace isosceles_triangle_perimeter_l2424_242461

theorem isosceles_triangle_perimeter (a b : ℝ)
  (h1 : b = 7)
  (h2 : a^2 - 8 * a + 15 = 0)
  (h3 : a * 2 > b)
  : 2 * a + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l2424_242461


namespace commodity_price_difference_l2424_242437

theorem commodity_price_difference (r : ℝ) (t : ℕ) :
  let P_X (t : ℕ) := 4.20 * (1 + (2*r + 10)/100)^(t - 2001)
  let P_Y (t : ℕ) := 4.40 * (1 + (r + 15)/100)^(t - 2001)
  P_X t = P_Y t + 0.90  ->
  ∃ t : ℕ, true :=
by
  sorry

end commodity_price_difference_l2424_242437


namespace find_a_l2424_242432

def A (a : ℤ) : Set ℤ := {-4, 2 * a - 1, a * a}
def B (a : ℤ) : Set ℤ := {a - 5, 1 - a, 9}

theorem find_a (a : ℤ) : (9 ∈ (A a ∩ B a)) ∧ (A a ∩ B a = {9}) ↔ a = -3 :=
by
  sorry

end find_a_l2424_242432


namespace percent_alcohol_new_solution_l2424_242492

theorem percent_alcohol_new_solution :
  let original_volume := 40
  let original_percent_alcohol := 5
  let added_alcohol := 2.5
  let added_water := 7.5
  let original_alcohol := original_volume * (original_percent_alcohol / 100)
  let total_alcohol := original_alcohol + added_alcohol
  let new_total_volume := original_volume + added_alcohol + added_water
  (total_alcohol / new_total_volume) * 100 = 9 :=
by
  sorry

end percent_alcohol_new_solution_l2424_242492


namespace total_pencils_l2424_242435

theorem total_pencils (pencils_per_child : ℕ) (children : ℕ) (h1 : pencils_per_child = 2) (h2 : children = 11) : (pencils_per_child * children = 22) := 
by
  sorry

end total_pencils_l2424_242435


namespace range_of_a_l2424_242445

theorem range_of_a (a : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + z^2 = 1 → |a - 1| ≥ x + 2 * y + 2 * z) →
  a ∈ Set.Iic (-2) ∪ Set.Ici 4 :=
by
  sorry

end range_of_a_l2424_242445


namespace series_satisfies_l2424_242472

noncomputable def series (x : ℝ) : ℝ :=
  let S₁ := 1 / (1 + x^2)
  let S₂ := x / (1 + x^2)
  (S₁ - S₂)

theorem series_satisfies (x : ℝ) (hx : 0 < x ∧ x < 1) : 
  x = series x ↔ x^3 + 2 * x - 1 = 0 :=
by 
  -- Proof outline:
  -- 1. Calculate the series S as a function of x
  -- 2. Equate series x to x and simplify to derive the polynomial equation
  sorry

end series_satisfies_l2424_242472


namespace range_of_a_l2424_242478

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, (a - 2) * x ^ 2 + 2 * (a - 2) * x - 4 < 0) ↔ -2 < a ∧ a ≤ 2 := 
sorry

end range_of_a_l2424_242478


namespace cosA_sinB_value_l2424_242469

theorem cosA_sinB_value (A B : ℝ) (hA1 : 0 < A ∧ A < π / 2) (hB1 : 0 < B ∧ B < π / 2)
  (h_tan_eq : (4 + (Real.tan A)^2) * (5 + (Real.tan B)^2) = Real.sqrt 320 * Real.tan A * Real.tan B) :
  Real.cos A * Real.sin B = Real.sqrt 6 / 6 := sorry

end cosA_sinB_value_l2424_242469


namespace savings_if_together_l2424_242498

def price_per_window : ℕ := 150

def discount_offer (n : ℕ) : ℕ := n - n / 7

def cost (n : ℕ) : ℕ := price_per_window * discount_offer n

def alice_windows : ℕ := 9
def bob_windows : ℕ := 10

def separate_cost : ℕ := cost alice_windows + cost bob_windows

def total_windows : ℕ := alice_windows + bob_windows

def together_cost : ℕ := cost total_windows

def savings : ℕ := separate_cost - together_cost

theorem savings_if_together : savings = 150 := by
  sorry

end savings_if_together_l2424_242498


namespace missing_files_correct_l2424_242462

def total_files : ℕ := 60
def files_in_morning : ℕ := total_files / 2
def files_in_afternoon : ℕ := 15
def missing_files : ℕ := total_files - (files_in_morning + files_in_afternoon)

theorem missing_files_correct : missing_files = 15 := by
  sorry

end missing_files_correct_l2424_242462


namespace hours_per_day_is_8_l2424_242448

-- Define the conditions
def hire_two_bodyguards (day_count : ℕ) (total_payment : ℕ) (hourly_rate : ℕ) (daily_hours : ℕ) : Prop :=
  2 * hourly_rate * day_count * daily_hours = total_payment

-- Define the correct answer
theorem hours_per_day_is_8 :
  hire_two_bodyguards 7 2240 20 8 :=
by
  -- Here, you would provide the step-by-step justification, but we use sorry since no proof is required.
  sorry

end hours_per_day_is_8_l2424_242448


namespace second_customer_headphones_l2424_242403

theorem second_customer_headphones
  (H : ℕ)
  (M : ℕ)
  (x : ℕ)
  (H_eq : H = 30)
  (eq1 : 5 * M + 8 * H = 840)
  (eq2 : 3 * M + x * H = 480) :
  x = 4 :=
by
  sorry

end second_customer_headphones_l2424_242403


namespace pure_imaginary_number_l2424_242446

open Complex -- Use the Complex module for complex numbers

theorem pure_imaginary_number (a : ℝ) (h : (a - 1 : ℂ).re = 0) : a = 1 :=
by
  -- This part of the proof is omitted hence we put sorry
  sorry

end pure_imaginary_number_l2424_242446


namespace find_m_parallel_l2424_242422

noncomputable def is_parallel (A1 B1 C1 A2 B2 C2 : ℝ) : Prop :=
  -(A1 / B1) = -(A2 / B2)

theorem find_m_parallel : ∃ m : ℝ, is_parallel (m-1) 3 m 1 (m+1) 2 ∧ m = -2 :=
by
  unfold is_parallel
  exists (-2 : ℝ)
  sorry

end find_m_parallel_l2424_242422


namespace Kristy_baked_cookies_l2424_242410

theorem Kristy_baked_cookies 
  (ate_by_Kristy : ℕ) (given_to_brother : ℕ) 
  (taken_by_first_friend : ℕ) (taken_by_second_friend : ℕ)
  (taken_by_third_friend : ℕ) (cookies_left : ℕ) 
  (h_K : ate_by_Kristy = 2) (h_B : given_to_brother = 1) 
  (h_F1 : taken_by_first_friend = 3) (h_F2 : taken_by_second_friend = 5)
  (h_F3 : taken_by_third_friend = 5) (h_L : cookies_left = 6) :
  ate_by_Kristy + given_to_brother 
  + taken_by_first_friend + taken_by_second_friend 
  + taken_by_third_friend + cookies_left = 22 := 
by
  sorry

end Kristy_baked_cookies_l2424_242410


namespace sum_of_n_l2424_242416

theorem sum_of_n (n : ℤ) (h : (36 : ℤ) % (2 * n - 1) = 0) :
  (n = 1 ∨ n = 2 ∨ n = 5) → 1 + 2 + 5 = 8 :=
by
  intros hn
  have h1 : n = 1 ∨ n = 2 ∨ n = 5 := hn
  sorry

end sum_of_n_l2424_242416


namespace green_square_area_percentage_l2424_242447

noncomputable def flag_side_length (k: ℝ) : ℝ := k
noncomputable def cross_area_fraction : ℝ := 0.49
noncomputable def cross_area (k: ℝ) : ℝ := cross_area_fraction * k^2
noncomputable def cross_width (t: ℝ) : ℝ := t
noncomputable def green_square_side (x: ℝ) : ℝ := x
noncomputable def green_square_area (x: ℝ) : ℝ := x^2

theorem green_square_area_percentage (k: ℝ) (t: ℝ) (x: ℝ)
  (h1: x = 2 * t)
  (h2: 4 * t * (k - t) + x^2 = cross_area k)
  : green_square_area x / (k^2) * 100 = 6.01 :=
by
  sorry

end green_square_area_percentage_l2424_242447


namespace journey_distance_l2424_242451

theorem journey_distance (D : ℝ) (h1 : (D / 40) + (D / 60) = 40) : D = 960 :=
by
  sorry

end journey_distance_l2424_242451


namespace bill_property_taxes_l2424_242452

theorem bill_property_taxes 
  (take_home_salary sales_taxes gross_salary : ℕ)
  (income_tax_rate : ℚ)
  (take_home_salary_eq : take_home_salary = 40000)
  (sales_taxes_eq : sales_taxes = 3000)
  (gross_salary_eq : gross_salary = 50000)
  (income_tax_rate_eq : income_tax_rate = 0.1) :
  let income_taxes := (income_tax_rate * gross_salary) 
  let property_taxes := gross_salary - (income_taxes + sales_taxes + take_home_salary)
  property_taxes = 2000 := by
  sorry

end bill_property_taxes_l2424_242452


namespace find_f2_l2424_242436

-- Define the conditions
variable {f g : ℝ → ℝ} {a : ℝ}

-- Assume f is an odd function
axiom odd_f : ∀ x : ℝ, f (-x) = -f x

-- Assume g is an even function
axiom even_g : ∀ x : ℝ, g (-x) = g x

-- Condition given in the problem
axiom f_g_relation : ∀ x : ℝ, f x + g x = a^x - a^(-x) + 2

-- Condition that g(2) = a
axiom g_at_2 : g 2 = a

-- Condition for a
axiom a_cond : a > 0 ∧ a ≠ 1

-- Proof problem
theorem find_f2 : f 2 = 15 / 4 := by
  sorry

end find_f2_l2424_242436


namespace P_subsetneq_Q_l2424_242485

def P : Set ℝ := { x : ℝ | x > 1 }
def Q : Set ℝ := { x : ℝ | x^2 - x > 0 }

theorem P_subsetneq_Q : P ⊂ Q :=
by
  sorry

end P_subsetneq_Q_l2424_242485


namespace line_passes_through_fixed_point_l2424_242481

variable {a b : ℝ}

theorem line_passes_through_fixed_point : 
  (∀ (x y : ℝ), a + 2 * b = 1 ∧ ax + 3 * y + b = 0 → (x, y) = (1/2, -1/6)) :=
by
  sorry

end line_passes_through_fixed_point_l2424_242481


namespace cori_age_l2424_242443

theorem cori_age (C A : ℕ) (hA : A = 19) (hEq : C + 5 = (A + 5) / 3) : C = 3 := by
  rw [hA] at hEq
  norm_num at hEq
  linarith

end cori_age_l2424_242443


namespace tan_alpha_neg_seven_l2424_242480

noncomputable def tan_alpha (α : ℝ) := Real.tan α

theorem tan_alpha_neg_seven {α : ℝ} 
  (h1 : α ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h2 : Real.cos α ^ 2 + Real.sin (Real.pi + 2 * α) = 3 / 10) : 
  tan_alpha α = -7 := 
sorry

end tan_alpha_neg_seven_l2424_242480


namespace subset_condition_for_a_l2424_242420

theorem subset_condition_for_a (a : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 ≤ 5 / 4 → (|x - 1| + 2 * |y - 2| ≤ a)) → a ≥ 5 / 2 :=
by
  intro H
  sorry

end subset_condition_for_a_l2424_242420


namespace simple_interest_rate_l2424_242426

theorem simple_interest_rate :
  ∀ (P T F : ℝ), P = 1000 → T = 3 → F = 1300 → (F - P) = P * 0.1 * T :=
by
  intros P T F hP hT hF
  sorry

end simple_interest_rate_l2424_242426


namespace quadratic_equation_roots_sum_and_difference_l2424_242434

theorem quadratic_equation_roots_sum_and_difference :
  ∃ (p q : ℝ), 
    p + q = 7 ∧ 
    |p - q| = 9 ∧ 
    (∀ x, (x - p) * (x - q) = x^2 - 7 * x - 8) :=
sorry

end quadratic_equation_roots_sum_and_difference_l2424_242434


namespace max_number_ahn_can_get_l2424_242411

theorem max_number_ahn_can_get :
  ∃ n : ℤ, (10 ≤ n ∧ n ≤ 99) ∧ ∀ m : ℤ, (10 ≤ m ∧ m ≤ 99) → (3 * (300 - n) ≥ 3 * (300 - m)) ∧ 3 * (300 - n) = 870 :=
by sorry

end max_number_ahn_can_get_l2424_242411


namespace initial_points_count_l2424_242408

theorem initial_points_count (k : ℕ) (h : (4 * k - 3) = 101): k = 26 :=
by 
  sorry

end initial_points_count_l2424_242408


namespace smallest_n_with_290_trailing_zeros_in_factorial_l2424_242483

def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 5^2) + (n / 5^3) + (n / 5^4) + (n / 5^5) + (n / 5^6) -- sum until the division becomes zero

theorem smallest_n_with_290_trailing_zeros_in_factorial : 
  ∀ (n : ℕ), n >= 1170 ↔ trailing_zeros n >= 290 ∧ trailing_zeros (n-1) < 290 := 
by { sorry }

end smallest_n_with_290_trailing_zeros_in_factorial_l2424_242483


namespace tangent_line_equation_l2424_242476

theorem tangent_line_equation (a : ℝ) (h : a ≠ 0) :
  (∃ b : ℝ, b = 2 ∧ (∀ x : ℝ, y = a * x^2) ∧ y - a = b * (x - 1)) → 
  ∃ (x y : ℝ), 2 * x - y - 1 = 0 :=
by
  sorry

end tangent_line_equation_l2424_242476


namespace estimate_ratio_l2424_242430

theorem estimate_ratio (A B : ℕ) (A_def : A = 1 * 2 * 7 + 2 * 4 * 14 + 3 * 6 * 21 + 4 * 8 * 28)
  (B_def : B = 1 * 3 * 5 + 2 * 6 * 10 + 3 * 9 * 15 + 4 * 12 * 20) : 0 < A / B ∧ A / B < 1 := by
  sorry

end estimate_ratio_l2424_242430


namespace polynomial_divisibility_l2424_242463

-- Definitions
def f (k l m n : ℕ) (x : ℂ) : ℂ :=
  x^(4 * k) + x^(4 * l + 1) + x^(4 * m + 2) + x^(4 * n + 3)

def g (x : ℂ) : ℂ :=
  x^3 + x^2 + x + 1

-- Theorem statement
theorem polynomial_divisibility (k l m n : ℕ) : ∀ x : ℂ, g x ∣ f k l m n x :=
  sorry

end polynomial_divisibility_l2424_242463


namespace find_n_range_l2424_242421

theorem find_n_range (m n : ℝ) 
  (h_m : -Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :
  (∀ x y z : ℝ, 0 ≤ x^2 + 2 * y^2 + 3 * z^2 + 2 * x * y + 2 * m * z * x + 2 * n * y * z) ↔ 
  (m - Real.sqrt (3 - m^2) ≤ n ∧ n ≤ m + Real.sqrt (3 - m^2)) :=
by
  sorry

end find_n_range_l2424_242421


namespace farmer_profit_l2424_242402

noncomputable def profit_earned : ℕ :=
  let pigs := 6
  let sale_price := 300
  let food_cost_per_month := 10
  let months_group1 := 12
  let months_group2 := 16
  let pigs_group1 := 3
  let pigs_group2 := 3
  let total_food_cost := (pigs_group1 * months_group1 * food_cost_per_month) + 
                         (pigs_group2 * months_group2 * food_cost_per_month)
  let total_revenue := pigs * sale_price
  total_revenue - total_food_cost

theorem farmer_profit : profit_earned = 960 := by
  unfold profit_earned
  sorry

end farmer_profit_l2424_242402


namespace largest_subset_size_with_property_l2424_242486

def no_four_times_property (S : Finset ℕ) : Prop := 
  ∀ {x y}, x ∈ S → y ∈ S → x = 4 * y → False

noncomputable def max_subset_size : ℕ := 145

theorem largest_subset_size_with_property :
  ∃ (S : Finset ℕ), (∀ x ∈ S, x ≤ 150) ∧ no_four_times_property S ∧ S.card = max_subset_size :=
sorry

end largest_subset_size_with_property_l2424_242486


namespace minimum_value_l2424_242431

noncomputable def condition (x : ℝ) : Prop := (2 * x - 1) / 3 - 1 ≥ x - (5 - 3 * x) / 2

noncomputable def target_function (x : ℝ) : ℝ := abs (x - 1) - abs (x + 3)

theorem minimum_value :
  ∃ x : ℝ, condition x ∧ ∀ y : ℝ, condition y → target_function y ≥ target_function x :=
sorry

end minimum_value_l2424_242431


namespace elvins_first_month_bill_l2424_242497

theorem elvins_first_month_bill (F C : ℝ) 
  (h1 : F + C = 52)
  (h2 : F + 2 * C = 76) : 
  F + C = 52 :=
by
  sorry

end elvins_first_month_bill_l2424_242497


namespace percentage_music_students_l2424_242473

variables (total_students : ℕ) (dance_students : ℕ) (art_students : ℕ)
  (music_students : ℕ) (music_percentage : ℚ)

def students_music : ℕ := total_students - (dance_students + art_students)
def percentage_students_music : ℚ := (students_music total_students dance_students art_students : ℚ) / (total_students : ℚ) * 100

theorem percentage_music_students (h1 : total_students = 400)
                                  (h2 : dance_students = 120)
                                  (h3 : art_students = 200) :
  percentage_students_music total_students dance_students art_students = 20 := by {
  sorry
}

end percentage_music_students_l2424_242473


namespace solve_for_x_l2424_242488

theorem solve_for_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 :=
by
  sorry

end solve_for_x_l2424_242488


namespace constant_term_value_l2424_242440

theorem constant_term_value :
  ∀ (x y z k : ℤ), (4 * x + y + z = 80) → (2 * x - y - z = 40) → (x = 20) → (3 * x + y - z = k) → (k = 60) :=
by 
  intros x y z k h₁ h₂ hx h₃
  sorry

end constant_term_value_l2424_242440


namespace factorization_correct_l2424_242494

theorem factorization_correct :
    (∀ (x y : ℝ), x * (2 * x - y) + 2 * y * (2 * x - y) = (x + 2 * y) * (2 * x - y)) :=
by
  intro x y
  sorry

end factorization_correct_l2424_242494


namespace radius_of_cookie_l2424_242424

theorem radius_of_cookie : 
  ∀ x y : ℝ, (x^2 + y^2 - 6.5 = x + 3 * y) → 
  ∃ (c : ℝ × ℝ) (r : ℝ), r = 3 ∧ (x - c.1)^2 + (y - c.2)^2 = r^2 :=
by {
  sorry
}

end radius_of_cookie_l2424_242424


namespace angle_EHG_65_l2424_242467

/-- Quadrilateral $EFGH$ has $EF = FG = GH$, $\angle EFG = 80^\circ$, and $\angle FGH = 150^\circ$; and hence the degree measure of $\angle EHG$ is $65^\circ$. -/
theorem angle_EHG_65 {EF FG GH : ℝ} (h1 : EF = FG) (h2 : FG = GH) 
  (EFG : ℝ) (FGH : ℝ) (h3 : EFG = 80) (h4 : FGH = 150) : 
  ∃ EHG : ℝ, EHG = 65 :=
by
  sorry

end angle_EHG_65_l2424_242467


namespace oxygen_atoms_in_compound_l2424_242418

theorem oxygen_atoms_in_compound (K_weight Br_weight O_weight molecular_weight : ℕ) 
    (hK : K_weight = 39) (hBr : Br_weight = 80) (hO : O_weight = 16) (hMW : molecular_weight = 168) 
    (n : ℕ) :
    168 = 39 + 80 + n * 16 → n = 3 :=
by
  intros h
  sorry

end oxygen_atoms_in_compound_l2424_242418


namespace find_n_solution_l2424_242487

theorem find_n_solution (n : ℚ) (h : (2 / (n+2)) + (4 / (n+2)) + (n / (n+2)) = 4) : n = -2 / 3 := 
by 
  sorry

end find_n_solution_l2424_242487


namespace problem_statement_l2424_242455

-- Given that f(x) is an even function.
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Definition of the main condition f(x) + f(2 - x) = 0.
def special_condition (f : ℝ → ℝ) : Prop := ∀ x, f x + f (2 - x) = 0

-- Theorem: Given the conditions, show that f(x) has a period of 4 and f(x-1) is odd.
theorem problem_statement {f : ℝ → ℝ} (h_even : is_even f) (h_cond : special_condition f) :
  (∀ x, f (4 + x) = f x) ∧ (∀ x, f (-x - 1) = -f (x - 1)) :=
by
  sorry

end problem_statement_l2424_242455


namespace negation_of_p_l2424_242465

open Classical

-- Define proposition p
def p : Prop := ∀ x : ℝ, x^2 + x > 2

-- Define the negation of proposition p
def not_p : Prop := ∃ x : ℝ, x^2 + x ≤ 2

theorem negation_of_p : ¬p ↔ not_p :=
by sorry

end negation_of_p_l2424_242465


namespace find_prime_triplet_l2424_242482

theorem find_prime_triplet (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) :
  3 * p^4 - 5 * q^4 - 4 * r^2 = 26 ↔ (p, q, r) = (5, 3, 19) :=
by
  sorry

end find_prime_triplet_l2424_242482


namespace radius_of_circle_zero_l2424_242400

theorem radius_of_circle_zero :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 10 * y + 41 = 0) →
  (0 : ℝ) = 0 :=
by
  intro h
  sorry

end radius_of_circle_zero_l2424_242400


namespace greatest_sum_l2424_242441

-- stating the conditions
def condition1 (x y : ℝ) := x^2 + y^2 = 130
def condition2 (x y : ℝ) := x * y = 45

-- proving the result
theorem greatest_sum (x y : ℝ) 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x + y = 10 * Real.sqrt 2.2 :=
sorry

end greatest_sum_l2424_242441


namespace train_length_calculation_l2424_242439

theorem train_length_calculation (len1 : ℝ) (speed1_kmph : ℝ) (speed2_kmph : ℝ) (crossing_time : ℝ) (len2 : ℝ) :
  len1 = 120.00001 → 
  speed1_kmph = 120 → 
  speed2_kmph = 80 → 
  crossing_time = 9 → 
  (len1 + len2) = ((speed1_kmph * 1000 / 3600 + speed2_kmph * 1000 / 3600) * crossing_time) → 
  len2 = 379.99949 :=
by
  intros hlen1 hspeed1 hspeed2 htime hdistance
  sorry

end train_length_calculation_l2424_242439


namespace tomato_red_flesh_probability_l2424_242425

theorem tomato_red_flesh_probability :
  (P_yellow_skin : ℝ) = 3 / 8 →
  (P_red_flesh_given_yellow_skin : ℝ) = 8 / 15 →
  (P_yellow_skin_given_not_red_flesh : ℝ) = 7 / 30 →
  (P_red_flesh : ℝ) = 1 / 4 := 
by
  intros h1 h2 h3
  sorry

end tomato_red_flesh_probability_l2424_242425


namespace new_game_cost_l2424_242490

theorem new_game_cost (G : ℕ) (h_initial_money : 83 = G + 9 * 4) : G = 47 := by
  sorry

end new_game_cost_l2424_242490


namespace bears_per_shelf_l2424_242466

def bears_initial : ℕ := 6

def shipment : ℕ := 18

def shelves : ℕ := 4

theorem bears_per_shelf : (bears_initial + shipment) / shelves = 6 := by
  sorry

end bears_per_shelf_l2424_242466


namespace tedra_harvested_2000kg_l2424_242433

noncomputable def totalTomatoesHarvested : ℕ :=
  let wednesday : ℕ := 400
  let thursday : ℕ := wednesday / 2
  let total_wednesday_thursday := wednesday + thursday
  let remaining_friday : ℕ := 700
  let given_away_friday : ℕ := 700
  let friday := remaining_friday + given_away_friday
  total_wednesday_thursday + friday

theorem tedra_harvested_2000kg :
  totalTomatoesHarvested = 2000 := by
  sorry

end tedra_harvested_2000kg_l2424_242433


namespace rectangle_perimeter_is_70_l2424_242477

-- Define the length and width of the rectangle
def length : ℕ := 19
def width : ℕ := 16

-- Define the perimeter function for a rectangle
def perimeter (length width : ℕ) : ℕ := 2 * (length + width)

-- The theorem statement asserting that the perimeter of the given rectangle is 70 cm
theorem rectangle_perimeter_is_70 :
  perimeter length width = 70 := 
sorry

end rectangle_perimeter_is_70_l2424_242477


namespace algebra_inequality_l2424_242475

theorem algebra_inequality
  (x y z : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z)
  (h_cond : x * y + y * z + z * x ≤ 1) :
  (x + 1 / x) * (y + 1 / y) * (z + 1 / z) ≥ 8 * (x + y) * (y + z) * (z + x) :=
by
  sorry

end algebra_inequality_l2424_242475


namespace solve_equation_l2424_242415

theorem solve_equation (x : ℝ) : (x + 2)^2 - 5 * (x + 2) = 0 ↔ (x = -2 ∨ x = 3) :=
by sorry

end solve_equation_l2424_242415


namespace Ellipse_area_constant_l2424_242493

-- Definitions of given conditions and problem setup
def ellipse_equation (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1 ∧ a > b ∧ b > 0

def point_on_ellipse (a b : ℝ) : Prop :=
  ellipse_equation 1 (Real.sqrt 3 / 2) a b

def eccentricity (c a : ℝ) : Prop :=
  c / a = Real.sqrt 3 / 2

def moving_points_on_ellipse (a b x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  ellipse_equation x₁ y₁ a b ∧ ellipse_equation x₂ y₂ a b

def slopes_condition (k₁ k₂ : ℝ) : Prop :=
  k₁ * k₂ = -1/4

def area_OMN := 1

-- Main theorem statement
theorem Ellipse_area_constant
(a b : ℝ) 
(h_ellipse : point_on_ellipse a b)
(h_eccentricity : eccentricity (Real.sqrt 3 / 2 * a) a)
(M N : ℝ × ℝ) 
(h_points : moving_points_on_ellipse a b M.1 M.2 N.1 N.2)
(k₁ k₂ : ℝ) 
(h_slopes : slopes_condition k₁ k₂) : 
a^2 = 4 ∧ b^2 = 1 ∧ area_OMN = 1 := 
sorry

end Ellipse_area_constant_l2424_242493


namespace value_of_m_l2424_242414

theorem value_of_m : 
  (2 ^ 1999 - 2 ^ 1998 - 2 ^ 1997 + 2 ^ 1996 - 2 ^ 1995 = m * 2 ^ 1995) -> m = 5 :=
by 
  sorry

end value_of_m_l2424_242414


namespace find_breadth_of_landscape_l2424_242468

theorem find_breadth_of_landscape (L B A : ℕ) 
  (h1 : B = 8 * L)
  (h2 : 3200 = A / 9)
  (h3 : 3200 * 9 = A) :
  B = 480 :=
by 
  sorry

end find_breadth_of_landscape_l2424_242468


namespace number_of_factors_of_60_l2424_242438

theorem number_of_factors_of_60 : 
  ∃ n, n = 12 ∧ 
  (∀ p k : ℕ, p ∈ [2, 3, 5] → 60 = 2^2 * 3^1 * 5^1 → (∃ d : ℕ, d = (2 + 1) * (1 + 1) * (1 + 1) ∧ n = d)) :=
by sorry

end number_of_factors_of_60_l2424_242438


namespace charlie_age_l2424_242470

variable (J C B : ℝ)

def problem_statement :=
  J = C + 12 ∧ C = B + 7 ∧ J = 3 * B → C = 18

theorem charlie_age : problem_statement J C B :=
by
  sorry

end charlie_age_l2424_242470


namespace cricket_bat_weight_l2424_242423

-- Define the conditions as Lean definitions
def weight_of_basketball : ℕ := 36
def weight_of_basketballs (n : ℕ) := n * weight_of_basketball
def weight_of_cricket_bats (m : ℕ) := m * (weight_of_basketballs 4 / 8)

-- State the theorem and skip the proof
theorem cricket_bat_weight :
  weight_of_cricket_bats 1 = 18 :=
by
  sorry

end cricket_bat_weight_l2424_242423


namespace pasture_rent_share_l2424_242407

theorem pasture_rent_share (x : ℕ) (H1 : (45 / (10 * x + 60 + 45)) * 245 = 63) : 
  x = 7 :=
by {
  sorry
}

end pasture_rent_share_l2424_242407


namespace red_knights_fraction_magic_l2424_242449

theorem red_knights_fraction_magic (total_knights red_knights blue_knights magical_knights : ℕ)
  (h1 : red_knights = (3 / 8 : ℚ) * total_knights)
  (h2 : blue_knights = total_knights - red_knights)
  (h3 : magical_knights = (1 / 4 : ℚ) * total_knights)
  (fraction_red_magic fraction_blue_magic : ℚ) 
  (h4 : fraction_red_magic = 3 * fraction_blue_magic)
  (h5 : magical_knights = red_knights * fraction_red_magic + blue_knights * fraction_blue_magic) :
  fraction_red_magic = 3 / 7 := 
by
  sorry

end red_knights_fraction_magic_l2424_242449


namespace opposite_of_8_is_neg_8_l2424_242458

theorem opposite_of_8_is_neg_8 : - (8 : ℤ) = -8 :=
by
  sorry

end opposite_of_8_is_neg_8_l2424_242458


namespace locus_of_vertices_l2424_242417

theorem locus_of_vertices (t : ℝ) (x y : ℝ) (h : y = x^2 + t * x + 1) : y = 1 - x^2 :=
by
  sorry

end locus_of_vertices_l2424_242417


namespace pascal_elements_sum_l2424_242413

theorem pascal_elements_sum :
  (Nat.choose 20 4 + Nat.choose 20 5) = 20349 :=
by
  sorry

end pascal_elements_sum_l2424_242413


namespace second_group_students_l2424_242484

theorem second_group_students (S : ℕ) : 
    (1200 / 40) = 9 + S + 11 → S = 10 :=
by sorry

end second_group_students_l2424_242484


namespace slope_of_line_n_l2424_242474

noncomputable def tan_double_angle (t : ℝ) : ℝ := (2 * t) / (1 - t^2)

theorem slope_of_line_n :
  let slope_m := 6
  let alpha := Real.arctan slope_m
  let slope_n := tan_double_angle slope_m
  slope_n = -12 / 35 :=
by
  sorry

end slope_of_line_n_l2424_242474
