import Mathlib

namespace quality_of_algorithm_reflects_number_of_operations_l566_56685

-- Definitions
def speed_of_operation_is_important (c : Type) : Prop :=
  ∀ (c1 : c), true

-- Theorem stating that the number of operations within a unit of time is an important sign of the quality of an algorithm
theorem quality_of_algorithm_reflects_number_of_operations {c : Type} 
    (h_speed_important : speed_of_operation_is_important c) : 
  ∀ (a : Type) (q : a), true := 
sorry

end quality_of_algorithm_reflects_number_of_operations_l566_56685


namespace rowing_distance_l566_56605

theorem rowing_distance (v_b : ℝ) (v_s : ℝ) (t_total : ℝ) (D : ℝ) :
  v_b = 9 → v_s = 1.5 → t_total = 48 → D / (v_b + v_s) + D / (v_b - v_s) = t_total → D = 210 :=
by
  intros
  sorry

end rowing_distance_l566_56605


namespace max_value_x_l566_56653

theorem max_value_x : ∃ x, x ^ 2 = 38 ∧ x = Real.sqrt 38 := by
  sorry

end max_value_x_l566_56653


namespace line_through_point_parallel_l566_56643

theorem line_through_point_parallel 
    (x y : ℝ)
    (h0 : (x = -1) ∧ (y = 3))
    (h1 : ∃ c : ℝ, (∀ x y : ℝ, x - 2 * y + c = 0 ↔ x - 2 * y + 3 = 0)) :
     ∃ c : ℝ, ∀ x y : ℝ, (x = -1) ∧ (y = 3) → (∃ (a b : ℝ), a - 2 * b + c = 0) :=
by
  sorry

end line_through_point_parallel_l566_56643


namespace range_of_slope_ellipse_chord_l566_56635

theorem range_of_slope_ellipse_chord :
  ∀ (x₁ y₁ x₂ y₂ x₀ y₀ : ℝ),
    (x₀ = (x₁ + x₂) / 2 ∧ y₀ = (y₁ + y₂) / 2) →
    (x₁^2 + y₁^2 / 4 = 1 ∧ x₂^2 + y₂^2 / 4 = 1) →
    ((1 / 2) ≤ y₀ ∧ y₀ ≤ 1) →
    (-4 ≤ -2 / y₀ ∧ -2 / y₀ ≤ -2) :=
by
  sorry

end range_of_slope_ellipse_chord_l566_56635


namespace alex_buys_17p3_pounds_of_corn_l566_56616

noncomputable def pounds_of_corn (c b : ℝ) : Prop :=
    c + b = 30 ∧ 1.05 * c + 0.39 * b = 23.10

theorem alex_buys_17p3_pounds_of_corn :
    ∃ c b, pounds_of_corn c b ∧ c = 17.3 :=
by
    sorry

end alex_buys_17p3_pounds_of_corn_l566_56616


namespace cost_of_new_shoes_l566_56661

theorem cost_of_new_shoes 
    (R : ℝ) 
    (L_r : ℝ) 
    (L_n : ℝ) 
    (increase_percent : ℝ) 
    (H_R : R = 13.50) 
    (H_L_r : L_r = 1) 
    (H_L_n : L_n = 2) 
    (H_inc_percent : increase_percent = 0.1852) : 
    2 * (R * (1 + increase_percent) / L_n) = 32.0004 := 
by
    sorry

end cost_of_new_shoes_l566_56661


namespace surface_area_comparison_l566_56600

theorem surface_area_comparison (a R : ℝ) (h_eq_volumes : (4 / 3) * Real.pi * R^3 = a^3) :
  6 * a^2 > 4 * Real.pi * R^2 :=
by
  sorry

end surface_area_comparison_l566_56600


namespace smallest_b_for_factoring_l566_56673

theorem smallest_b_for_factoring :
  ∃ b : ℕ, b > 0 ∧
    (∀ r s : ℤ, r * s = 2016 → r + s ≠ b) ∧
    (∀ r s : ℤ, r * s = 2016 → r + s = b → b = 92) :=
sorry

end smallest_b_for_factoring_l566_56673


namespace binary_predecessor_l566_56666

theorem binary_predecessor (N : ℕ) (hN : N = 0b11000) : 0b10111 + 1 = N := 
by
  sorry

end binary_predecessor_l566_56666


namespace convert_to_scientific_notation_l566_56683

theorem convert_to_scientific_notation (H : 1 = 10^9) : 
  3600 * (10 : ℝ)^9 = 3.6 * (10 : ℝ)^12 :=
by
  sorry

end convert_to_scientific_notation_l566_56683


namespace cars_to_hours_l566_56608

def car_interval := 20 -- minutes
def num_cars := 30
def minutes_per_hour := 60

theorem cars_to_hours :
  (car_interval * num_cars) / minutes_per_hour = 10 := by
  sorry

end cars_to_hours_l566_56608


namespace wrapping_paper_l566_56612

theorem wrapping_paper (total_used : ℚ) (decoration_used : ℚ) (presents : ℕ) (other_presents : ℕ) (individual_used : ℚ) 
  (h1 : total_used = 5 / 8) 
  (h2 : decoration_used = 1 / 24) 
  (h3 : presents = 4) 
  (h4 : other_presents = 3) 
  (h5 : individual_used = (5 / 8 - 1 / 24) / 3) : 
  individual_used = 7 / 36 := 
by
  -- The theorem will be proven here.
  sorry

end wrapping_paper_l566_56612


namespace base7_number_divisibility_l566_56678

theorem base7_number_divisibility (x : ℕ) (h : 0 ≤ x ∧ x ≤ 6) :
  (5 * 343 + 2 * 49 + x * 7 + 4) % 29 = 0 ↔ x = 6 := 
by
  sorry

end base7_number_divisibility_l566_56678


namespace known_number_l566_56695

theorem known_number (A B : ℕ) (h_hcf : 1 / (Nat.gcd A B) = 1 / 15) (h_lcm : 1 / Nat.lcm A B = 1 / 312) (h_B : B = 195) : A = 24 :=
by
  -- Skipping proof
  sorry

end known_number_l566_56695


namespace symmetric_points_parabola_l566_56634

theorem symmetric_points_parabola (x1 x2 y1 y2 m : ℝ) (h1 : y1 = 2 * x1^2) (h2 : y2 = 2 * x2^2)
    (h3 : x1 * x2 = -3 / 4) (h_sym: (y2 - y1) / (x2 - x1) = -1)
    (h_mid: (y2 + y1) / 2 = (x2 + x1) / 2 + m) :
    m = 2 := sorry

end symmetric_points_parabola_l566_56634


namespace perimeter_triangle_ABC_eq_18_l566_56621

theorem perimeter_triangle_ABC_eq_18 (h1 : ∀ (Δ : ℕ), Δ = 9) 
(h2 : ∀ (p : ℕ), p = 6) : 
∀ (perimeter_ABC : ℕ), perimeter_ABC = 18 := by
sorry

end perimeter_triangle_ABC_eq_18_l566_56621


namespace volume_inequality_holds_l566_56633

def volume (x : ℕ) : ℤ :=
  (x^2 - 16) * (x^3 + 25)

theorem volume_inequality_holds :
  ∃ (n : ℕ), n = 1 ∧ ∃ x : ℕ, volume x < 1000 ∧ (x - 4) > 0 :=
by
  sorry

end volume_inequality_holds_l566_56633


namespace crayons_left_l566_56604

-- Define the initial number of crayons
def initial_crayons : ℕ := 440

-- Define the crayons given away
def crayons_given : ℕ := 111

-- Define the crayons lost
def crayons_lost : ℕ := 106

-- Prove the final number of crayons left
theorem crayons_left : (initial_crayons - crayons_given - crayons_lost) = 223 :=
by
  sorry

end crayons_left_l566_56604


namespace root_of_polynomial_l566_56640

theorem root_of_polynomial :
  ∀ x : ℝ, (x^2 - 3 * x + 2) * x * (x - 4) = 0 ↔ (x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 4) :=
by 
  sorry

end root_of_polynomial_l566_56640


namespace find_a_plus_b_l566_56670

theorem find_a_plus_b (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + a * x + b) 
  (h2 : { x : ℝ | 0 ≤ f x ∧ f x ≤ 6 - x } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } ∪ {6}) 
  : a + b = 9 := 
sorry

end find_a_plus_b_l566_56670


namespace worth_of_used_car_l566_56625

theorem worth_of_used_car (earnings remaining : ℝ) (earnings_eq : earnings = 5000) (remaining_eq : remaining = 1000) : 
  ∃ worth : ℝ, worth = earnings - remaining ∧ worth = 4000 :=
by
  sorry

end worth_of_used_car_l566_56625


namespace parabola_equation_l566_56623

theorem parabola_equation (p : ℝ) (h_pos : p > 0) (M : ℝ) (h_Mx : M = 3) (h_MF : abs (M + p/2) = 2 * p) :
  (forall x y, y^2 = 2 * p * x) -> (forall x y, y^2 = 4 * x) :=
by
  sorry

end parabola_equation_l566_56623


namespace part_1_part_2_l566_56669

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - m * x
noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := Real.log x + 1 - m
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := x * Real.log x - x - a * x^3
noncomputable def r (x : ℝ) : ℝ := (Real.log x - 1) / x^2
noncomputable def r' (x : ℝ) : ℝ := (3 - 2 * Real.log x) / x^3

theorem part_1 (x : ℝ) (m : ℝ) (h1 : f x m = -1) (h2 : f' x m = 0) :
  m = 1 ∧ (∀ y, y > 0 → y < x → f' y 1 < 0) ∧ (∀ y, y > x → f' y 1 > 0) :=
sorry

theorem part_2 (a : ℝ) :
  (a > 1 / (2 * Real.exp 3) → ∀ x, h x a ≠ 0) ∧
  (a ≤ 0 ∨ a = 1 / (2 * Real.exp 3) → ∃ x, h x a = 0 ∧ ∀ y, h y a = 0 → y = x) ∧
  (0 < a ∧ a < 1 / (2 * Real.exp 3) → ∃ x1 x2, x1 ≠ x2 ∧ h x1 a = 0 ∧ h x2 a = 0) :=
sorry

end part_1_part_2_l566_56669


namespace john_reams_needed_l566_56681

theorem john_reams_needed 
  (pages_flash_fiction_weekly : ℕ := 20) 
  (pages_short_story_weekly : ℕ := 50) 
  (pages_novel_annual : ℕ := 1500) 
  (weeks_in_year : ℕ := 52) 
  (sheets_per_ream : ℕ := 500) 
  (sheets_flash_fiction_weekly : ℕ := 10)
  (sheets_short_story_weekly : ℕ := 25) :
  let sheets_flash_fiction_annual := sheets_flash_fiction_weekly * weeks_in_year
  let sheets_short_story_annual := sheets_short_story_weekly * weeks_in_year
  let total_sheets_annual := sheets_flash_fiction_annual + sheets_short_story_annual + pages_novel_annual
  let reams_needed := (total_sheets_annual + sheets_per_ream - 1) / sheets_per_ream
  reams_needed = 7 := 
by sorry

end john_reams_needed_l566_56681


namespace snowfall_on_friday_l566_56688

def snowstorm (snow_wednesday snow_thursday total_snow : ℝ) : ℝ :=
  total_snow - (snow_wednesday + snow_thursday)

theorem snowfall_on_friday :
  snowstorm 0.33 0.33 0.89 = 0.23 := 
by
  -- (Conditions)
  -- snow_wednesday = 0.33
  -- snow_thursday = 0.33
  -- total_snow = 0.89
  -- (Conclusion) snowstorm 0.33 0.33 0.89 = 0.23
  sorry

end snowfall_on_friday_l566_56688


namespace lucy_snowballs_l566_56601

theorem lucy_snowballs : ∀ (c l : ℕ), c = l + 31 → c = 50 → l = 19 :=
by
  intros c l h1 h2
  sorry

end lucy_snowballs_l566_56601


namespace sports_lottery_systematic_sampling_l566_56662

-- Definition of the sports lottery condition
def is_first_prize_ticket (n : ℕ) : Prop := n % 1000 = 345

-- Statement of the proof problem
theorem sports_lottery_systematic_sampling :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100000 → is_first_prize_ticket n) →
  ∃ interval, (∀ segment_start : ℕ,  segment_start < 1000 → is_first_prize_ticket (segment_start + interval * 999))
  := by sorry

end sports_lottery_systematic_sampling_l566_56662


namespace correct_calculation_l566_56697

theorem correct_calculation (x : ℝ) : (2 * x^5) / (-x)^3 = -2 * x^2 :=
by sorry

end correct_calculation_l566_56697


namespace fisher_eligibility_l566_56690

theorem fisher_eligibility (A1 A2 S : ℕ) (hA1 : A1 = 84) (hS : S = 82) :
  (S ≥ 80) → (A1 + A2 ≥ 170) → (A2 = 86) :=
by
  sorry

end fisher_eligibility_l566_56690


namespace rancher_cows_l566_56676

theorem rancher_cows (H C : ℕ) (h1 : C = 5 * H) (h2 : C + H = 168) : C = 140 := by
  sorry

end rancher_cows_l566_56676


namespace smallest_positive_period_and_symmetry_l566_56614

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (x + (7 * Real.pi / 4)) + 
  Real.cos (x - (3 * Real.pi / 4))

theorem smallest_positive_period_and_symmetry :
  (∃ T > 0, T = 2 * Real.pi ∧ ∀ x, f (x + T) = f x) ∧ 
  (∃ a, a = - (Real.pi / 4) ∧ ∀ x, f (2 * a - x) = f x) :=
by
  sorry

end smallest_positive_period_and_symmetry_l566_56614


namespace jovana_added_pounds_l566_56679

noncomputable def initial_amount : ℕ := 5
noncomputable def final_amount : ℕ := 28

theorem jovana_added_pounds : final_amount - initial_amount = 23 := by
  sorry

end jovana_added_pounds_l566_56679


namespace arithmetic_sequence_problem_l566_56630

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (d : ℝ)
  (a1 : ℝ)
  (h_arithmetic : ∀ n, a n = a1 + (n - 1) * d)
  (h_a4 : a 4 = 5) :
  2 * a 1 - a 5 + a 11 = 10 := 
by
  sorry

end arithmetic_sequence_problem_l566_56630


namespace inequality_l566_56626

theorem inequality (A B : ℝ) (n : ℕ) (hA : 0 ≤ A) (hB : 0 ≤ B) (hn : 1 ≤ n) : (A + B)^n ≤ 2^(n - 1) * (A^n + B^n) := 
  sorry

end inequality_l566_56626


namespace negation_is_all_odd_or_at_least_two_even_l566_56694

-- Define natural numbers a, b, and c.
variables {a b c : ℕ}

-- Define a predicate is_even which checks if a number is even.
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define the statement that exactly one of the natural numbers a, b, and c is even.
def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∨ is_even b ∨ is_even c) ∧
  ¬ (is_even a ∧ is_even b) ∧
  ¬ (is_even a ∧ is_even c) ∧
  ¬ (is_even b ∧ is_even c)

-- Define the negation of the statement that exactly one of the natural numbers a, b, and c is even.
def negation_of_exactly_one_even (a b c : ℕ) : Prop :=
  ¬ exactly_one_even a b c

-- State that the negation of exactly one even number among a, b, c is equivalent to all being odd or at least two being even.
theorem negation_is_all_odd_or_at_least_two_even :
  negation_of_exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c) :=
sorry

end negation_is_all_odd_or_at_least_two_even_l566_56694


namespace television_price_reduction_l566_56607

theorem television_price_reduction (P : ℝ) (h₁ : 0 ≤ P):
  ((P - (P * 0.7 * 0.8)) / P) * 100 = 44 :=
by
  sorry

end television_price_reduction_l566_56607


namespace percentage_relations_with_respect_to_z_l566_56689

variable (x y z w : ℝ)
variable (h1 : x = 1.30 * y)
variable (h2 : y = 0.50 * z)
variable (h3 : w = 2 * x)

theorem percentage_relations_with_respect_to_z : 
  x = 0.65 * z ∧ y = 0.50 * z ∧ w = 1.30 * z := by
  sorry

end percentage_relations_with_respect_to_z_l566_56689


namespace jessica_repay_l566_56631

theorem jessica_repay (P : ℝ) (r : ℝ) (n : ℝ) (x : ℕ)
  (hx : P = 20)
  (hr : r = 0.12)
  (hn : n = 3 * P) :
  x = 17 :=
sorry

end jessica_repay_l566_56631


namespace custom_operation_correct_l566_56686

def custom_operation (a b : ℤ) : ℤ := (a + b) * (a - b)

theorem custom_operation_correct : custom_operation 6 3 = 27 :=
by {
  sorry
}

end custom_operation_correct_l566_56686


namespace fraction_is_one_third_l566_56610

noncomputable def fraction_studying_japanese (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : ℚ :=
  J / (J + S)

theorem fraction_is_one_third (J S : ℕ) (h1 : S = 2 * J) (h2 : 3 / 8 * S + 1 / 4 * J = J) : 
  fraction_studying_japanese J S h1 h2 = 1 / 3 :=
  sorry

end fraction_is_one_third_l566_56610


namespace log_base_change_l566_56680

theorem log_base_change (log_16_32 log_16_inv2: ℝ) : 
  (log_16_32 * log_16_inv2 = -5 / 16) :=
by
  sorry

end log_base_change_l566_56680


namespace decreasing_power_function_has_specific_m_l566_56651

theorem decreasing_power_function_has_specific_m (m : ℝ) (x : ℝ) : 
  (∀ x > 0, (m^2 - m - 1) * x^(m^2 - 2 * m - 3) < 0) → 
  m = 2 :=
by
  sorry

end decreasing_power_function_has_specific_m_l566_56651


namespace freezer_temperature_is_minus_12_l566_56652

theorem freezer_temperature_is_minus_12 (refrigeration_temp freezer_temp : ℤ) (h1 : refrigeration_temp = 5) (h2 : freezer_temp = -12) : freezer_temp = -12 :=
by sorry

end freezer_temperature_is_minus_12_l566_56652


namespace find_b_plus_m_l566_56682

noncomputable def f (a b x : ℝ) : ℝ := Real.log (x + 1) / Real.log a + b 

variable (a b m : ℝ)
-- Conditions
axiom h1 : a > 0
axiom h2 : a ≠ 1
axiom h3 : f a b m = 3

theorem find_b_plus_m : b + m = 3 :=
sorry

end find_b_plus_m_l566_56682


namespace euler_disproof_l566_56645

theorem euler_disproof :
  ∃ (n : ℕ), 0 < n ∧ (133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144) :=
by
  sorry

end euler_disproof_l566_56645


namespace least_k_divisible_by_240_l566_56639

theorem least_k_divisible_by_240 : ∃ (k : ℕ), k^2 % 240 = 0 ∧ k = 60 :=
by
  sorry

end least_k_divisible_by_240_l566_56639


namespace third_median_length_l566_56627

theorem third_median_length (m1 m2 area : ℝ) (h1 : m1 = 5) (h2 : m2 = 10) (h3 : area = 10 * Real.sqrt 10) : 
  ∃ m3 : ℝ, m3 = 3 * Real.sqrt 10 :=
by
  sorry

end third_median_length_l566_56627


namespace avg_of_five_consecutive_from_b_l566_56609

-- Conditions
def avg_of_five_even_consecutive (a : ℕ) : ℕ := (2 * a + (2 * a + 2) + (2 * a + 4) + (2 * a + 6) + (2 * a + 8)) / 5

-- The main theorem
theorem avg_of_five_consecutive_from_b (a : ℕ) : 
  avg_of_five_even_consecutive a = 2 * a + 4 → 
  ((2 * a + 4 + (2 * a + 4 + 1) + (2 * a + 4 + 2) + (2 * a + 4 + 3) + (2 * a + 4 + 4)) / 5) = 2 * a + 6 :=
by
  sorry

end avg_of_five_consecutive_from_b_l566_56609


namespace sum_consecutive_equals_prime_l566_56674

theorem sum_consecutive_equals_prime (m k p : ℕ) (h_prime : Nat.Prime p) :
  (∃ S, S = (m * (2 * k + m - 1)) / 2 ∧ S = p) →
  m = 1 ∨ m = 2 :=
sorry

end sum_consecutive_equals_prime_l566_56674


namespace store_owner_loss_percentage_l566_56665

theorem store_owner_loss_percentage :
  ∀ (initial_value : ℝ) (profit_margin : ℝ) (loss1 : ℝ) (loss2 : ℝ) (loss3 : ℝ) (tax_rate : ℝ),
    initial_value = 100 → profit_margin = 0.10 → loss1 = 0.20 → loss2 = 0.30 → loss3 = 0.25 → tax_rate = 0.12 →
      ((initial_value - initial_value * (1 - loss1) * (1 - loss2) * (1 - loss3)) / initial_value * 100) = 58 :=
by
  intros initial_value profit_margin loss1 loss2 loss3 tax_rate h_initial_value h_profit_margin h_loss1 h_loss2 h_loss3 h_tax_rate
  -- Variable assignments as per given conditions
  have h1 : initial_value = 100 := h_initial_value
  have h2 : profit_margin = 0.10 := h_profit_margin
  have h3 : loss1 = 0.20 := h_loss1
  have h4 : loss2 = 0.30 := h_loss2
  have h5 : loss3 = 0.25 := h_loss3
  have h6 : tax_rate = 0.12 := h_tax_rate
  
  sorry

end store_owner_loss_percentage_l566_56665


namespace proof_problem_l566_56613

noncomputable def find_values (a b c x y z : ℝ) := 
  14 * x + b * y + c * z = 0 ∧ 
  a * x + 24 * y + c * z = 0 ∧ 
  a * x + b * y + 43 * z = 0 ∧ 
  a ≠ 14 ∧ b ≠ 24 ∧ c ≠ 43 ∧ x ≠ 0

theorem proof_problem (a b c x y z : ℝ) 
  (h : find_values a b c x y z):
  (a / (a - 14)) + (b / (b - 24)) + (c / (c - 43)) = 1 :=
by
  sorry

end proof_problem_l566_56613


namespace polygon_number_of_sides_l566_56603

theorem polygon_number_of_sides (h : ∀ (n : ℕ), (360 : ℝ) / (n : ℝ) = 1) : 
  360 = (1:ℝ) :=
  sorry

end polygon_number_of_sides_l566_56603


namespace stickers_given_l566_56618

def total_stickers : ℕ := 100
def andrew_ratio : ℚ := 1 / 5
def bill_ratio : ℚ := 3 / 10

theorem stickers_given (zander_collection : ℕ)
                       (andrew_received : ℚ)
                       (bill_received : ℚ)
                       (total_given : ℚ):
  zander_collection = total_stickers →
  andrew_received = andrew_ratio →
  bill_received = bill_ratio →
  total_given = (andrew_received * zander_collection) + (bill_received * (zander_collection - (andrew_received * zander_collection))) →
  total_given = 44 :=
by
  intros hz har hbr htg
  sorry

end stickers_given_l566_56618


namespace chandra_monsters_l566_56646

def monsters_day_1 : Nat := 2
def monsters_day_2 : Nat := monsters_day_1 * 3
def monsters_day_3 : Nat := monsters_day_2 * 4
def monsters_day_4 : Nat := monsters_day_3 * 5
def monsters_day_5 : Nat := monsters_day_4 * 6

def total_monsters : Nat := monsters_day_1 + monsters_day_2 + monsters_day_3 + monsters_day_4 + monsters_day_5

theorem chandra_monsters : total_monsters = 872 :=
by
  unfold total_monsters
  unfold monsters_day_1
  unfold monsters_day_2
  unfold monsters_day_3
  unfold monsters_day_4
  unfold monsters_day_5
  sorry

end chandra_monsters_l566_56646


namespace sum_ratio_arithmetic_sequence_l566_56611

theorem sum_ratio_arithmetic_sequence (a₁ d : ℚ) (h : d ≠ 0) 
  (S : ℕ → ℚ)
  (h_sum : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2)
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 6 / S 12 = 3 / 10 :=
by
  sorry

end sum_ratio_arithmetic_sequence_l566_56611


namespace integer_condition_l566_56691

theorem integer_condition (p : ℕ) (h : p > 0) : 
  (∃ n : ℤ, (3 * (p: ℤ) + 25) = n * (2 * (p: ℤ) - 5)) ↔ (3 ≤ p ∧ p ≤ 35) :=
sorry

end integer_condition_l566_56691


namespace example_problem_l566_56606

def Z (x y : ℝ) : ℝ := x^2 - 3 * x * y + y^2

theorem example_problem :
  Z 4 3 = -11 := 
by
  -- proof goes here
  sorry

end example_problem_l566_56606


namespace divisibility_of_a81_l566_56684

theorem divisibility_of_a81 
  (p : ℕ) (hp : Nat.Prime p) (hp_gt2 : 2 < p)
  (a : ℕ → ℕ) (h_rec : ∀ n, n * a (n + 1) = (n + 1) * a n - (p / 2)^4) 
  (h_a1 : a 1 = 5) :
  16 ∣ a 81 := 
sorry

end divisibility_of_a81_l566_56684


namespace geometric_sequence_a12_l566_56619

noncomputable def a_n (a1 r : ℝ) (n : ℕ) : ℝ :=
  a1 * r ^ (n - 1)

theorem geometric_sequence_a12 (a1 r : ℝ) 
  (h1 : a_n a1 r 7 * a_n a1 r 9 = 4)
  (h2 : a_n a1 r 4 = 1) :
  a_n a1 r 12 = 16 := sorry

end geometric_sequence_a12_l566_56619


namespace original_plan_trees_average_l566_56699

-- Definitions based on conditions
def original_trees_per_day (x : ℕ) := x
def increased_trees_per_day (x : ℕ) := x + 5
def time_to_plant_60_trees (x : ℕ) := 60 / (x + 5)
def time_to_plant_45_trees (x : ℕ) := 45 / x

-- The main theorem we need to prove
theorem original_plan_trees_average : ∃ x : ℕ, time_to_plant_60_trees x = time_to_plant_45_trees x ∧ x = 15 :=
by
  -- Placeholder for the proof
  sorry

end original_plan_trees_average_l566_56699


namespace gcd_256_162_450_l566_56615

theorem gcd_256_162_450 : Nat.gcd (Nat.gcd 256 162) 450 = 2 := sorry

end gcd_256_162_450_l566_56615


namespace polynomial_coefficients_l566_56654

theorem polynomial_coefficients
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
  (h : (x-3)^8 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3 + 
                a_4 * (x-2)^4 + a_5 * (x-2)^5 + a_6 * (x-2)^6 + 
                a_7 * (x-2)^7 + a_8 * (x-2)^8) :
  (a_0 = 1) ∧ 
  (a_1 / 2 + a_2 / 2^2 + a_3 / 2^3 + a_4 / 2^4 + a_5 / 2^5 + 
   a_6 / 2^6 + a_7 / 2^7 + a_8 / 2^8 = -255 / 256) ∧ 
  (a_0 + a_2 + a_4 + a_6 + a_8 = 128) :=
by sorry

end polynomial_coefficients_l566_56654


namespace find_constant_t_l566_56667

theorem find_constant_t : ∃ t : ℝ, 
  (∀ x : ℝ, (3 * x^2 - 4 * x + 5) * (2 * x^2 + t * x + 8) = 6 * x^4 + (-26) * x^3 + 58 * x^2 + (-76) * x + 40) ↔ t = -6 :=
by {
  sorry
}

end find_constant_t_l566_56667


namespace packs_of_chocolate_l566_56671

theorem packs_of_chocolate (t c k x : ℕ) (ht : t = 42) (hc : c = 4) (hk : k = 22) (hx : x = t - (c + k)) : x = 16 :=
by
  rw [ht, hc, hk] at hx
  simp at hx
  exact hx

end packs_of_chocolate_l566_56671


namespace first_statement_second_statement_difference_between_statements_l566_56692

variable (A B C : Prop)

-- First statement: (A ∨ B) → C
theorem first_statement : (A ∨ B) → C :=
sorry

-- Second statement: (A ∧ B) → C
theorem second_statement : (A ∧ B) → C :=
sorry

-- Proof that shows the difference between the two statements
theorem difference_between_statements :
  ((A ∨ B) → C) ↔ ¬((A ∧ B) → C) :=
sorry

end first_statement_second_statement_difference_between_statements_l566_56692


namespace puddle_base_area_l566_56657

theorem puddle_base_area (rate depth hours : ℝ) (A : ℝ) 
  (h1 : rate = 10) 
  (h2 : depth = 30) 
  (h3 : hours = 3) 
  (h4 : depth * A = rate * hours) : 
  A = 1 := 
by 
  sorry

end puddle_base_area_l566_56657


namespace poly_has_int_solution_iff_l566_56664

theorem poly_has_int_solution_iff (a : ℤ) : 
  (a > 0 ∧ (∃ x : ℤ, a * x^2 + 2 * (2 * a - 1) * x + 4 * a - 7 = 0)) ↔ (a = 1 ∨ a = 5) :=
by {
  sorry
}

end poly_has_int_solution_iff_l566_56664


namespace sum_of_two_rel_prime_numbers_l566_56658

theorem sum_of_two_rel_prime_numbers (k : ℕ) : 
  (∃ a b : ℕ, a > 1 ∧ b > 1 ∧ Nat.gcd a b = 1 ∧ k = a + b) ↔ (k = 5 ∨ k ≥ 7) := sorry

end sum_of_two_rel_prime_numbers_l566_56658


namespace basic_astrophysics_degrees_l566_56622

theorem basic_astrophysics_degrees :
  let microphotonics_pct := 12
  let home_electronics_pct := 24
  let food_additives_pct := 15
  let gmo_pct := 29
  let industrial_lubricants_pct := 8
  let total_budget_percentage := 100
  let full_circle_degrees := 360
  let given_pct_sum := microphotonics_pct + home_electronics_pct + food_additives_pct + gmo_pct + industrial_lubricants_pct
  let astrophysics_pct := total_budget_percentage - given_pct_sum
  let astrophysics_degrees := (astrophysics_pct * full_circle_degrees) / total_budget_percentage
  astrophysics_degrees = 43.2 := by
  sorry

end basic_astrophysics_degrees_l566_56622


namespace line_passes_through_point_l566_56649

-- We declare the variables for the real numbers a, b, and c
variables (a b c : ℝ)

-- We state the condition that a + b - c = 0
def condition1 : Prop := a + b - c = 0

-- We state the condition that not all of a, b, c are zero
def condition2 : Prop := ¬ (a = 0 ∧ b = 0 ∧ c = 0)

-- We state the theorem: the line ax + by + c = 0 passes through the point (-1, -1)
theorem line_passes_through_point (h1 : condition1 a b c) (h2 : condition2 a b c) :
  a * (-1) + b * (-1) + c = 0 := sorry

end line_passes_through_point_l566_56649


namespace triangle_area_l566_56659

-- Defining the rectangle dimensions
def length : ℝ := 35
def width : ℝ := 48

-- Defining the area of the right triangle formed by the diagonal of the rectangle
theorem triangle_area : (1 / 2) * length * width = 840 := by
  sorry

end triangle_area_l566_56659


namespace machines_work_together_l566_56642

theorem machines_work_together (x : ℝ) (h₁ : 1/(x+4) + 1/(x+2) + 1/(x+3) = 1/x) : x = 1 :=
sorry

end machines_work_together_l566_56642


namespace minimum_waste_l566_56624

/-- Zenobia's cookout problem setup -/
def LCM_hot_dogs_buns : Nat := Nat.lcm 10 12

def hot_dog_packages : Nat := LCM_hot_dogs_buns / 10
def bun_packages : Nat := LCM_hot_dogs_buns / 12

def waste_hot_dog_packages : ℝ := hot_dog_packages * 0.4
def waste_bun_packages : ℝ := bun_packages * 0.3
def total_waste : ℝ := waste_hot_dog_packages + waste_bun_packages

theorem minimum_waste :
  hot_dog_packages = 6 ∧ bun_packages = 5 ∧ total_waste = 3.9 :=
by
  sorry

end minimum_waste_l566_56624


namespace beads_problem_l566_56668

theorem beads_problem :
  ∃ b : ℕ, (b % 6 = 5) ∧ (b % 8 = 3) ∧ (b % 9 = 7) ∧ (b = 179) :=
by
  sorry

end beads_problem_l566_56668


namespace min_tablets_to_get_two_each_l566_56637

def least_tablets_to_ensure_two_each (A B : ℕ) (A_eq : A = 10) (B_eq : B = 10) : ℕ :=
  if A ≥ 2 ∧ B ≥ 2 then 4 else 12

theorem min_tablets_to_get_two_each :
  least_tablets_to_ensure_two_each 10 10 rfl rfl = 12 :=
by
  sorry

end min_tablets_to_get_two_each_l566_56637


namespace haley_seeds_in_big_garden_l566_56644

def seeds_in_big_garden (total_seeds : ℕ) (small_gardens : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  total_seeds - (small_gardens * seeds_per_small_garden)

theorem haley_seeds_in_big_garden :
  let total_seeds := 56
  let small_gardens := 7
  let seeds_per_small_garden := 3
  seeds_in_big_garden total_seeds small_gardens seeds_per_small_garden = 35 :=
by
  sorry

end haley_seeds_in_big_garden_l566_56644


namespace linear_system_solution_l566_56672

theorem linear_system_solution (x y : ℚ) (h1 : 3 * x + 2 * y = 7) (h2 : 6 * x - 5 * y = 4) :
  x = 43 / 27 ∧ y = 10 / 9 :=
sorry

end linear_system_solution_l566_56672


namespace intersection_of_M_and_N_l566_56647

-- Define sets M and N
def M : Set ℤ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℤ := {0, 1, 2}

-- The theorem to be proven: M ∩ N = {0, 1, 2}
theorem intersection_of_M_and_N : M ∩ N = {0, 1, 2} :=
by
  sorry

end intersection_of_M_and_N_l566_56647


namespace fraction_problem_l566_56675

theorem fraction_problem (b : ℕ) (h₀ : 0 < b) (h₁ : (b : ℝ) / (b + 35) = 0.869) : b = 232 := 
by
  sorry

end fraction_problem_l566_56675


namespace trip_attendees_trip_cost_savings_l566_56663

theorem trip_attendees (total_people : ℕ) (total_cost : ℕ) (adult_ticket : ℕ) 
(student_discount : ℕ) (group_discount : ℕ) (adults : ℕ) (students : ℕ) :
total_people = 130 → total_cost = 9600 → adult_ticket = 120 →
student_discount = 50 → group_discount = 40 → 
total_people = adults + students → 
total_cost = adults * adult_ticket + students * (adult_ticket * student_discount / 100) →
adults = 30 ∧ students = 100 :=
by sorry

theorem trip_cost_savings (total_people : ℕ) (individual_total_cost : ℕ) 
(group_total_cost : ℕ) (student_tickets : ℕ) (group_tickets : ℕ) 
(adult_ticket : ℕ) (student_discount : ℕ) (group_discount : ℕ) :
(total_people = 130) → (individual_total_cost = 7200 + 1800) → 
(group_total_cost = total_people * (adult_ticket * group_discount / 100)) →
(adult_ticket = 120) → (student_discount = 50) → (group_discount = 40) → 
(total_people = student_tickets + group_tickets) → (student_tickets = 30) → 
(group_tickets = 100) → (7200 + 1800 < 9360) → 
student_tickets = 30 ∧ group_tickets = 100 :=
by sorry

end trip_attendees_trip_cost_savings_l566_56663


namespace hyperbola_condition_l566_56660

theorem hyperbola_condition (k : ℝ) : 
  (0 < k ∧ k < 1) ↔ ∀ x y : ℝ, (x^2 / (k - 1)) + (y^2 / (k + 2)) = 1 → 
  (k - 1 < 0 ∧ k + 2 > 0 ∨ k - 1 > 0 ∧ k + 2 < 0) := 
sorry

end hyperbola_condition_l566_56660


namespace fourth_hexagon_dots_l566_56693

def dots_in_hexagon (n : ℕ) : ℕ :=
  if n = 0 then 1
  else 1 + (12 * (n * (n + 1) / 2))

theorem fourth_hexagon_dots : dots_in_hexagon 4 = 85 :=
by
  unfold dots_in_hexagon
  norm_num
  sorry

end fourth_hexagon_dots_l566_56693


namespace cube_inequality_l566_56677

theorem cube_inequality {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_inequality_l566_56677


namespace product_of_possible_values_l566_56638

theorem product_of_possible_values (x : ℚ) (h : abs ((18 : ℚ) / (2 * x) - 4) = 3) : (x = 9 ∨ x = 9/7) → (9 * (9/7) = 81/7) :=
by
  intros
  sorry

end product_of_possible_values_l566_56638


namespace equation_of_line_l_l566_56620

noncomputable def line_eq (a b c : ℚ) : ℚ → ℚ → Prop := λ x y => a * x + b * y + c = 0

theorem equation_of_line_l : 
  ∃ m : ℚ, 
  (∀ x y : ℚ, 
    (2 * x - 3 * y - 3 = 0 ∧ x + y + 2 = 0 → line_eq 3 1 m x y) ∧ 
    (3 * x + y - 1 = 0 → line_eq 3 1 0 x y)
  ) →
  line_eq 15 5 16 (-3/5) (-7/5) :=
by 
  sorry

end equation_of_line_l_l566_56620


namespace Ryanne_is_7_years_older_than_Hezekiah_l566_56687

theorem Ryanne_is_7_years_older_than_Hezekiah
  (H : ℕ) (R : ℕ)
  (h1 : H = 4)
  (h2 : R + H = 15) :
  R - H = 7 := by
  sorry

end Ryanne_is_7_years_older_than_Hezekiah_l566_56687


namespace tiling_scenarios_unique_l566_56617

theorem tiling_scenarios_unique (m n : ℕ) 
  (h1 : 60 * m + 150 * n = 360) : m = 1 ∧ n = 2 :=
by {
  -- The proof will be provided here
  sorry
}

end tiling_scenarios_unique_l566_56617


namespace simplify_expression_l566_56698

noncomputable def expr := (-1 : ℝ)^2023 + Real.sqrt 9 - Real.pi^0 + Real.sqrt (1 / 8) * Real.sqrt 32

theorem simplify_expression : expr = 3 := 
by sorry

end simplify_expression_l566_56698


namespace length_reduction_by_50_percent_l566_56696

variable (L B L' : ℝ)

def rectangle_dimension_change (L B : ℝ) (perc_area_change : ℝ) (new_breadth_factor : ℝ) : Prop :=
  let original_area := L * B
  let new_breadth := new_breadth_factor * B
  let new_area := L' * new_breadth
  let expected_new_area := (1 + perc_area_change) * original_area
  new_area = expected_new_area

theorem length_reduction_by_50_percent (L B : ℝ) (h1: rectangle_dimension_change L B L' 0.5 3) : 
  L' = 0.5 * L :=
by
  unfold rectangle_dimension_change at h1
  simp at h1
  sorry

end length_reduction_by_50_percent_l566_56696


namespace min_bottles_l566_56602

theorem min_bottles (a b : ℕ) (h1 : a > b) (h2 : b > 1) : 
  ∃ x : ℕ, x = Nat.ceil (a - a / b) := sorry

end min_bottles_l566_56602


namespace particular_number_l566_56636

theorem particular_number {x : ℕ} (h : x - 29 + 64 = 76) : x = 41 := by
  sorry

end particular_number_l566_56636


namespace tory_sold_to_neighbor_l566_56628

def total_cookies : ℕ := 50
def sold_to_grandmother : ℕ := 12
def sold_to_uncle : ℕ := 7
def to_be_sold : ℕ := 26

def sold_to_neighbor : ℕ :=
  total_cookies - to_be_sold - (sold_to_grandmother + sold_to_uncle)

theorem tory_sold_to_neighbor :
  sold_to_neighbor = 5 :=
by
  intros
  sorry

end tory_sold_to_neighbor_l566_56628


namespace total_cups_l566_56656

variable (eggs : ℕ) (flour : ℕ)
variable (h : eggs = 60) (h1 : flour = eggs / 2)

theorem total_cups (eggs : ℕ) (flour : ℕ) (h : eggs = 60) (h1 : flour = eggs / 2) : 
  eggs + flour = 90 := 
by
  sorry

end total_cups_l566_56656


namespace solve_system_of_equations_l566_56641

theorem solve_system_of_equations (x y z : ℝ) :
  (x * y = z) ∧ (x * z = y) ∧ (y * z = x) ↔ 
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = -1 ∧ y = 1 ∧ z = -1) ∨
  (x = 1 ∧ y = -1 ∧ z = -1) ∨
  (x = -1 ∧ y = -1 ∧ z = 1) ∨
  (x = 0 ∧ y = 0 ∧ z = 0) := by
  sorry

end solve_system_of_equations_l566_56641


namespace complement_union_is_correct_l566_56648

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem complement_union_is_correct : (U \ A) ∪ B = {0, 2, 4} := by
  sorry

end complement_union_is_correct_l566_56648


namespace number_of_possible_lists_l566_56655

theorem number_of_possible_lists : 
  let num_balls := 15
  let num_draws := 4
  (num_balls ^ num_draws) = 50625 := by
  sorry

end number_of_possible_lists_l566_56655


namespace Vince_ride_longer_l566_56629

def Vince_ride_length : ℝ := 0.625
def Zachary_ride_length : ℝ := 0.5

theorem Vince_ride_longer : Vince_ride_length - Zachary_ride_length = 0.125 := by
  sorry

end Vince_ride_longer_l566_56629


namespace find_a_l566_56632

variable (m : ℝ)

def root1 := 2 * m - 1
def root2 := m + 4

theorem find_a (h : root1 ^ 2 = root2 ^ 2) : ∃ a : ℝ, a = 9 :=
by
  sorry

end find_a_l566_56632


namespace find_m_l566_56650

theorem find_m (m : ℝ) (a a1 a2 a3 a4 a5 a6 : ℝ) 
  (h1 : (1 + m)^6 = a + a1 + a2 + a3 + a4 + a5 + a6) 
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 = 63)
  (h3 : a = 1) : m = 1 ∨ m = -3 := 
by
  sorry

end find_m_l566_56650
