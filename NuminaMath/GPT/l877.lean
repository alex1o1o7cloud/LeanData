import Mathlib

namespace NUMINAMATH_GPT_no_real_roots_range_a_l877_87735

theorem no_real_roots_range_a (a : ℝ) : (¬∃ x : ℝ, 2 * x^2 + (a - 5) * x + 2 = 0) → 1 < a ∧ a < 9 :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_range_a_l877_87735


namespace NUMINAMATH_GPT_ratio_of_cost_to_selling_price_l877_87750

-- Define the conditions in Lean
variable (C S : ℝ) -- C is the cost price per pencil, S is the selling price per pencil
variable (h : 90 * C - 40 * S = 90 * S)

-- Define the statement to be proved
theorem ratio_of_cost_to_selling_price (C S : ℝ) (h : 90 * C - 40 * S = 90 * S) : (90 * C) / (90 * S) = 13 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_cost_to_selling_price_l877_87750


namespace NUMINAMATH_GPT_weight_lifting_requirement_l877_87743

-- Definitions based on conditions
def weight_25 : Int := 25
def weight_10 : Int := 10
def lifts_25 := 16
def total_weight_25 := 2 * weight_25 * lifts_25

def n_lifts_10 (n : Int) := 2 * weight_10 * n

-- Problem statement and theorem to prove
theorem weight_lifting_requirement (n : Int) : n_lifts_10 n = total_weight_25 ↔ n = 40 := by
  sorry

end NUMINAMATH_GPT_weight_lifting_requirement_l877_87743


namespace NUMINAMATH_GPT_jebb_total_spent_l877_87714

theorem jebb_total_spent
  (cost_of_food : ℝ) (service_fee_rate : ℝ) (tip : ℝ)
  (h1 : cost_of_food = 50)
  (h2 : service_fee_rate = 0.12)
  (h3 : tip = 5) :
  cost_of_food + (cost_of_food * service_fee_rate) + tip = 61 := 
sorry

end NUMINAMATH_GPT_jebb_total_spent_l877_87714


namespace NUMINAMATH_GPT_total_rainfall_cm_l877_87711

theorem total_rainfall_cm :
  let monday := 0.12962962962962962
  let tuesday := 3.5185185185185186 * 0.1
  let wednesday := 0.09259259259259259
  let thursday := 0.10222222222222223 * 2.54
  let friday := 12.222222222222221 * 0.1
  let saturday := 0.2222222222222222
  let sunday := 0.17444444444444446 * 2.54
  monday + tuesday + wednesday + thursday + friday + saturday + sunday = 2.721212629851652 :=
by
  sorry

end NUMINAMATH_GPT_total_rainfall_cm_l877_87711


namespace NUMINAMATH_GPT_remaining_strawberries_l877_87788

-- Define the constants based on conditions
def initial_kg1 : ℕ := 3
def initial_g1 : ℕ := 300
def given_kg1 : ℕ := 1
def given_g1 : ℕ := 900

-- Define the conversion from kilograms to grams
def kg_to_g (kg : ℕ) : ℕ := kg * 1000

-- Calculate initial total grams
def initial_total_g : ℕ := kg_to_g initial_kg1 + initial_g1

-- Calculate given total grams
def given_total_g : ℕ := kg_to_g given_kg1 + given_g1

-- Define the remaining grams
def remaining_g (initial_g : ℕ) (given_g : ℕ) : ℕ := initial_g - given_g

-- Statement to prove
theorem remaining_strawberries : remaining_g initial_total_g given_total_g = 1400 := by
sorry

end NUMINAMATH_GPT_remaining_strawberries_l877_87788


namespace NUMINAMATH_GPT_ratio_of_radii_of_cylinders_l877_87724

theorem ratio_of_radii_of_cylinders
  (r_V r_B h_V h_B : ℝ)
  (h1 : h_V = 1/2 * h_B)
  (h2 : π * r_B^2 * h_B / 2  = 4)
  (h3 : π * r_V^2 * h_V = 16) :
  r_V / r_B = 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_radii_of_cylinders_l877_87724


namespace NUMINAMATH_GPT_quadratic_geometric_sequence_root_l877_87765

theorem quadratic_geometric_sequence_root {a b c : ℝ} (r : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b = a * r) 
  (h3 : c = a * r^2)
  (h4 : a ≥ b) 
  (h5 : b ≥ c) 
  (h6 : c ≥ 0) 
  (h7 : (a * r)^2 - 4 * a * (a * r^2) = 0) : 
  -b / (2 * a) = -1 / 8 := 
sorry

end NUMINAMATH_GPT_quadratic_geometric_sequence_root_l877_87765


namespace NUMINAMATH_GPT_value_ne_one_l877_87756

theorem value_ne_one (a b: ℝ) (h : a * b ≠ 0) : (|a| / a) + (|b| / b) ≠ 1 := 
by 
  sorry

end NUMINAMATH_GPT_value_ne_one_l877_87756


namespace NUMINAMATH_GPT_range_of_a_l877_87720

theorem range_of_a (a : ℝ) : (∃ x : ℝ, |x + 2| + |x - 3| ≤ a) → a ≥ 5 :=
sorry

end NUMINAMATH_GPT_range_of_a_l877_87720


namespace NUMINAMATH_GPT_discount_on_pony_jeans_l877_87744

theorem discount_on_pony_jeans 
  (F P : ℕ)
  (h1 : F + P = 25)
  (h2 : 5 * F + 4 * P = 100) : P = 25 :=
by
  sorry

end NUMINAMATH_GPT_discount_on_pony_jeans_l877_87744


namespace NUMINAMATH_GPT_solution_set_f_x_minus_2_ge_zero_l877_87772

-- Define the necessary conditions and prove the statement
noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_x_minus_2_ge_zero (f_even : ∀ x, f x = f (-x))
  (f_mono : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y)
  (f_one_zero : f 1 = 0) :
  {x : ℝ | f (x - 2) ≥ 0} = {x | x ≥ 3 ∨ x ≤ 1} :=
by {
  sorry
}

end NUMINAMATH_GPT_solution_set_f_x_minus_2_ge_zero_l877_87772


namespace NUMINAMATH_GPT_gift_card_remaining_l877_87791

theorem gift_card_remaining (initial_amount : ℕ) (half_monday : ℕ) (quarter_tuesday : ℕ) : 
  initial_amount = 200 → 
  half_monday = initial_amount / 2 →
  quarter_tuesday = (initial_amount - half_monday) / 4 →
  initial_amount - half_monday - quarter_tuesday = 75 :=
by
  intros h_init h_half h_quarter
  rw [h_init, h_half, h_quarter]
  sorry

end NUMINAMATH_GPT_gift_card_remaining_l877_87791


namespace NUMINAMATH_GPT_prove_system_of_equations_l877_87734

variables (x y : ℕ)

def system_of_equations (x y : ℕ) : Prop :=
  x = 2*y + 4 ∧ x = 3*y - 9

theorem prove_system_of_equations :
  ∀ (x y : ℕ), system_of_equations x y :=
by sorry

end NUMINAMATH_GPT_prove_system_of_equations_l877_87734


namespace NUMINAMATH_GPT_jean_initial_stuffies_l877_87792

variable (S : ℕ) (h1 : S * 2 / 3 / 4 = 10)

theorem jean_initial_stuffies : S = 60 :=
by
  sorry

end NUMINAMATH_GPT_jean_initial_stuffies_l877_87792


namespace NUMINAMATH_GPT_find_x_y_l877_87704

theorem find_x_y (x y : ℕ) (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : y ≥ x) (h4 : x + y ≤ 20) 
  (h5 : ¬(∃ s, (x * y = s) → x + y = s ∧ ∃ a b : ℕ, a * b = s ∧ a ≠ x ∧ b ≠ y))
  (h6 : ∃ s_t, (x + y = s_t) → x * y = s_t):
  x = 2 ∧ y = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_x_y_l877_87704


namespace NUMINAMATH_GPT_find_number_satisfying_condition_l877_87748

-- Define the condition where fifteen percent of x equals 150
def fifteen_percent_eq (x : ℝ) : Prop :=
  (15 / 100) * x = 150

-- Statement to prove the existence of a number x that satisfies the condition, and this x equals 1000
theorem find_number_satisfying_condition : ∃ x : ℝ, fifteen_percent_eq x ∧ x = 1000 :=
by
  -- Proof will be added here
  sorry

end NUMINAMATH_GPT_find_number_satisfying_condition_l877_87748


namespace NUMINAMATH_GPT_complex_expression_value_l877_87798

theorem complex_expression_value :
  (i^3 * (1 + i)^2 = 2) :=
by
  sorry

end NUMINAMATH_GPT_complex_expression_value_l877_87798


namespace NUMINAMATH_GPT_vehicles_with_cd_player_but_no_pw_or_ab_l877_87761

-- Definitions based on conditions from step a)
def P : ℝ := 0.60 -- percentage of vehicles with power windows
def A : ℝ := 0.25 -- percentage of vehicles with anti-lock brakes
def C : ℝ := 0.75 -- percentage of vehicles with a CD player
def PA : ℝ := 0.10 -- percentage of vehicles with both power windows and anti-lock brakes
def AC : ℝ := 0.15 -- percentage of vehicles with both anti-lock brakes and a CD player
def PC : ℝ := 0.22 -- percentage of vehicles with both power windows and a CD player
def PAC : ℝ := 0.00 -- no vehicle has all 3 features

-- The statement we want to prove
theorem vehicles_with_cd_player_but_no_pw_or_ab : C - (PC + AC) = 0.38 := by
  sorry

end NUMINAMATH_GPT_vehicles_with_cd_player_but_no_pw_or_ab_l877_87761


namespace NUMINAMATH_GPT_solve_equation_l877_87769

theorem solve_equation (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
    x^2 * (Real.log 27 / Real.log x) * (Real.log x / Real.log 9) = x + 4 → x = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l877_87769


namespace NUMINAMATH_GPT_only_solution_for_triplet_l877_87745

theorem only_solution_for_triplet (x y z : ℤ) (h : x^2 + y^2 + z^2 - 2 * x * y * z = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
sorry

end NUMINAMATH_GPT_only_solution_for_triplet_l877_87745


namespace NUMINAMATH_GPT_parallel_lines_slope_equality_l877_87707

theorem parallel_lines_slope_equality (m : ℝ) : (∀ x y : ℝ, 3 * x + y - 3 = 0) ∧ (∀ x y : ℝ, 6 * x + m * y + 1 = 0) → m = 2 :=
by 
  sorry

end NUMINAMATH_GPT_parallel_lines_slope_equality_l877_87707


namespace NUMINAMATH_GPT_molecular_weight_of_9_moles_CCl4_l877_87794

-- Define the atomic weight of Carbon (C) and Chlorine (Cl)
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the molecular formula for carbon tetrachloride (CCl4)
def molecular_formula_CCl4 : ℝ := atomic_weight_C + (4 * atomic_weight_Cl)

-- Define the molecular weight of one mole of carbon tetrachloride (CCl4)
def molecular_weight_one_mole_CCl4 : ℝ := molecular_formula_CCl4

-- Define the number of moles
def moles_CCl4 : ℝ := 9

-- Define the result to check
def molecular_weight_nine_moles_CCl4 : ℝ := molecular_weight_one_mole_CCl4 * moles_CCl4

-- State the theorem to prove the molecular weight of 9 moles of carbon tetrachloride is 1384.29 grams
theorem molecular_weight_of_9_moles_CCl4 :
  molecular_weight_nine_moles_CCl4 = 1384.29 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_of_9_moles_CCl4_l877_87794


namespace NUMINAMATH_GPT_variation_of_x_l877_87722

theorem variation_of_x (k j z : ℝ) : ∃ m : ℝ, ∀ x y : ℝ, (x = k * y^2) ∧ (y = j * z^(1 / 3)) → (x = m * z^(2 / 3)) :=
sorry

end NUMINAMATH_GPT_variation_of_x_l877_87722


namespace NUMINAMATH_GPT_gas_cycle_work_done_l877_87701

noncomputable def p0 : ℝ := 10^5
noncomputable def V0 : ℝ := 1

theorem gas_cycle_work_done :
  (3 * Real.pi * p0 * V0 = 942) :=
by
  have h1 : p0 = 10^5 := by rfl
  have h2 : V0 = 1 := by rfl
  sorry

end NUMINAMATH_GPT_gas_cycle_work_done_l877_87701


namespace NUMINAMATH_GPT_sphere_volume_l877_87721

theorem sphere_volume (π : ℝ) (r : ℝ):
  4 * π * r^2 = 144 * π →
  (4 / 3) * π * r^3 = 288 * π :=
by
  sorry

end NUMINAMATH_GPT_sphere_volume_l877_87721


namespace NUMINAMATH_GPT_simplify_expression_l877_87716

variable (a : ℝ)

theorem simplify_expression :
    5 * a^2 - (a^2 - 2 * (a^2 - 3 * a)) = 6 * a^2 - 6 * a := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l877_87716


namespace NUMINAMATH_GPT_pigs_total_l877_87759

theorem pigs_total (initial_pigs : ℕ) (joined_pigs : ℕ) (total_pigs : ℕ) 
  (h1 : initial_pigs = 64) 
  (h2 : joined_pigs = 22) 
  : total_pigs = 86 :=
by
  sorry

end NUMINAMATH_GPT_pigs_total_l877_87759


namespace NUMINAMATH_GPT_count_multiples_of_13_three_digit_l877_87753

-- Definitions based on the conditions in the problem
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def is_multiple_of_13 (n : ℕ) : Prop := ∃ k : ℕ, n = 13 * k

-- Statement of the proof problem
theorem count_multiples_of_13_three_digit :
  ∃ (count : ℕ), count = (76 - 8 + 1) :=
sorry

end NUMINAMATH_GPT_count_multiples_of_13_three_digit_l877_87753


namespace NUMINAMATH_GPT_pizzas_served_dinner_eq_6_l877_87712

-- Definitions based on the conditions
def pizzas_served_lunch : Nat := 9
def pizzas_served_today : Nat := 15

-- The theorem to prove the number of pizzas served during dinner
theorem pizzas_served_dinner_eq_6 : pizzas_served_today - pizzas_served_lunch = 6 := by
  sorry

end NUMINAMATH_GPT_pizzas_served_dinner_eq_6_l877_87712


namespace NUMINAMATH_GPT_mass_percentage_correct_l877_87767

noncomputable def mass_percentage_C_H_N_O_in_C20H25N3O 
  (m_C : ℚ) (m_H : ℚ) (m_N : ℚ) (m_O : ℚ) 
  (atoms_C : ℚ) (atoms_H : ℚ) (atoms_N : ℚ) (atoms_O : ℚ)
  (total_mass : ℚ)
  (percentage_C : ℚ) (percentage_H : ℚ) (percentage_N : ℚ) (percentage_O : ℚ) :=
  atoms_C = 20 ∧ atoms_H = 25 ∧ atoms_N = 3 ∧ atoms_O = 1 ∧ 
  m_C = 12.01 ∧ m_H = 1.008 ∧ m_N = 14.01 ∧ m_O = 16 ∧ 
  total_mass = (atoms_C * m_C) + (atoms_H * m_H) + (atoms_N * m_N) + (atoms_O * m_O) ∧ 
  percentage_C = (atoms_C * m_C / total_mass) * 100 ∧ 
  percentage_H = (atoms_H * m_H / total_mass) * 100 ∧ 
  percentage_N = (atoms_N * m_N / total_mass) * 100 ∧ 
  percentage_O = (atoms_O * m_O / total_mass) * 100 

theorem mass_percentage_correct : 
  mass_percentage_C_H_N_O_in_C20H25N3O 12.01 1.008 14.01 16 20 25 3 1 323.43 74.27 7.79 12.99 4.95 :=
by {
  sorry
}

end NUMINAMATH_GPT_mass_percentage_correct_l877_87767


namespace NUMINAMATH_GPT_contractor_net_earnings_l877_87787

-- Definitions based on given conditions
def total_days : ℕ := 30
def pay_per_day : ℕ := 25
def fine_per_day : ℝ := 7.50
def absent_days : ℕ := 10

-- Calculation of the total amount received (involving both working days' pay and fines for absent days)
def worked_days : ℕ := total_days - absent_days
def total_earnings : ℕ := worked_days * pay_per_day
def total_fine : ℝ := absent_days * fine_per_day
def net_earnings : ℝ := total_earnings - total_fine

-- The Theorem to be proved
theorem contractor_net_earnings : net_earnings = 425 := 
by 
  sorry

end NUMINAMATH_GPT_contractor_net_earnings_l877_87787


namespace NUMINAMATH_GPT_variance_daily_reading_time_l877_87706

theorem variance_daily_reading_time :
  let mean10 := 2.7
  let var10 := 1
  let num10 := 800

  let mean11 := 3.1
  let var11 := 2
  let num11 := 600

  let mean12 := 3.3
  let var12 := 3
  let num12 := 600

  let num_total := num10 + num11 + num12

  let total_mean := (2.7 * 800 + 3.1 * 600 + 3.3 * 600) / 2000

  let var_total := (800 / 2000) * (1 + (2.7 - total_mean)^2) +
                   (600 / 2000) * (2 + (3.1 - total_mean)^2) +
                   (600 / 2000) * (3 + (3.3 - total_mean)^2)

  var_total = 1.966 :=
by
  sorry

end NUMINAMATH_GPT_variance_daily_reading_time_l877_87706


namespace NUMINAMATH_GPT_lincoln_high_school_students_l877_87795

theorem lincoln_high_school_students (total students_in_either_or_both_clubs students_in_photography students_in_science : ℕ)
  (h1 : total = 300)
  (h2 : students_in_photography = 120)
  (h3 : students_in_science = 140)
  (h4 : students_in_either_or_both_clubs = 220):
  ∃ x, x = 40 ∧ (students_in_photography + students_in_science - students_in_either_or_both_clubs = x) := 
by
  use 40
  sorry

end NUMINAMATH_GPT_lincoln_high_school_students_l877_87795


namespace NUMINAMATH_GPT_difference_two_smallest_integers_l877_87742

/--
There is more than one integer greater than 1 which, when divided by any integer k such that 2 ≤ k ≤ 11, has a remainder of 1.
Prove that the difference between the two smallest such integers is 27720.
-/
theorem difference_two_smallest_integers :
  ∃ n₁ n₂ : ℤ, 
  (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (n₁ % k = 1 ∧ n₂ % k = 1)) ∧ 
  n₁ > 1 ∧ n₂ > 1 ∧ 
  ∀ m : ℤ, (∀ k : ℤ, 2 ≤ k ∧ k ≤ 11 → (m % k =  1)) ∧ m > 1 → m = n₁ ∨ m = n₂ → 
  (n₂ - n₁ = 27720) := 
sorry

end NUMINAMATH_GPT_difference_two_smallest_integers_l877_87742


namespace NUMINAMATH_GPT_factor_expression_l877_87737

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) :=
by
sorry

end NUMINAMATH_GPT_factor_expression_l877_87737


namespace NUMINAMATH_GPT_egg_count_l877_87790

theorem egg_count (E : ℕ) (son_daughter_eaten : ℕ) (rhea_husband_eaten : ℕ) (total_eaten : ℕ) (total_eggs : ℕ) (uneaten : ℕ) (trays : ℕ) 
  (H1 : son_daughter_eaten = 2 * 2 * 7)
  (H2 : rhea_husband_eaten = 4 * 2 * 7)
  (H3 : total_eaten = son_daughter_eaten + rhea_husband_eaten)
  (H4 : uneaten = 6)
  (H5 : total_eggs = total_eaten + uneaten)
  (H6 : trays = 2)
  (H7 : total_eggs = E * trays) : 
  E = 45 :=
by
  sorry

end NUMINAMATH_GPT_egg_count_l877_87790


namespace NUMINAMATH_GPT_greatest_fraction_l877_87731

theorem greatest_fraction 
  (w x y z : ℕ)
  (hw : w > 0)
  (h_ordering : w < x ∧ x < y ∧ y < z) :
  (x + y + z) / (w + x + y) > (w + x + y) / (x + y + z) ∧
  (x + y + z) / (w + x + y) > (w + y + z) / (x + w + z) ∧
  (x + y + z) / (w + x + y) > (x + w + z) / (w + y + z) ∧
  (x + y + z) / (w + x + y) > (y + z + w) / (x + y + z) :=
sorry

end NUMINAMATH_GPT_greatest_fraction_l877_87731


namespace NUMINAMATH_GPT_express_in_scientific_notation_l877_87713

theorem express_in_scientific_notation : (0.0000028 = 2.8 * 10^(-6)) :=
sorry

end NUMINAMATH_GPT_express_in_scientific_notation_l877_87713


namespace NUMINAMATH_GPT_infinitely_many_m_l877_87799

theorem infinitely_many_m (k l : ℕ) (hk : 0 < k) (hl : 0 < l) :
  ∃ᶠ m in Filter.atTop, m ≥ k ∧ Nat.gcd (Nat.choose m k) l = 1 :=
sorry

end NUMINAMATH_GPT_infinitely_many_m_l877_87799


namespace NUMINAMATH_GPT_hip_hop_final_percentage_is_39_l877_87738

noncomputable def hip_hop_percentage (total_songs percentage_country: ℝ):
  ℝ :=
  let percentage_non_country := 1 - percentage_country
  let original_ratio_hip_hop := 0.65
  let original_ratio_pop := 0.35
  let total_non_country := original_ratio_hip_hop + original_ratio_pop
  let hip_hop_percentage := original_ratio_hip_hop / total_non_country * percentage_non_country
  hip_hop_percentage

theorem hip_hop_final_percentage_is_39 (total_songs : ℕ) :
  hip_hop_percentage total_songs 0.40 = 0.39 :=
by
  sorry

end NUMINAMATH_GPT_hip_hop_final_percentage_is_39_l877_87738


namespace NUMINAMATH_GPT_sum_mod_7_l877_87776

/-- Define the six numbers involved. -/
def a := 102345
def b := 102346
def c := 102347
def d := 102348
def e := 102349
def f := 102350

/-- State the theorem to prove the remainder of their sum when divided by 7. -/
theorem sum_mod_7 : 
  (a + b + c + d + e + f) % 7 = 5 := 
by sorry

end NUMINAMATH_GPT_sum_mod_7_l877_87776


namespace NUMINAMATH_GPT_family_chocolate_chip_count_l877_87796

theorem family_chocolate_chip_count
  (batch_cookies : ℕ)
  (total_people : ℕ)
  (batches : ℕ)
  (choco_per_cookie : ℕ)
  (cookie_total : ℕ := batch_cookies * batches)
  (cookies_per_person : ℕ := cookie_total / total_people)
  (choco_per_person : ℕ := cookies_per_person * choco_per_cookie)
  (h1 : batch_cookies = 12)
  (h2 : total_people = 4)
  (h3 : batches = 3)
  (h4 : choco_per_cookie = 2)
  : choco_per_person = 18 := 
by sorry

end NUMINAMATH_GPT_family_chocolate_chip_count_l877_87796


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l877_87773

def repeating_decimal_to_fraction (d: ℚ) (r: ℚ) (p: ℚ): ℚ :=
  d + r

theorem repeating_decimal_fraction :
  repeating_decimal_to_fraction (6 / 10) (1 / 33) (0.6 + (0.03 : ℚ)) = 104 / 165 := 
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l877_87773


namespace NUMINAMATH_GPT_total_fertilizer_usage_l877_87770

theorem total_fertilizer_usage :
  let daily_A : ℝ := 3 / 12
  let daily_B : ℝ := 4 / 10
  let daily_C : ℝ := 5 / 8
  let final_A : ℝ := daily_A + 6
  let final_B : ℝ := daily_B + 5
  let final_C : ℝ := daily_C + 7
  (final_A + final_B + final_C) = 19.275 := by
  sorry

end NUMINAMATH_GPT_total_fertilizer_usage_l877_87770


namespace NUMINAMATH_GPT_whitney_money_leftover_l877_87775

def poster_cost : ℕ := 5
def notebook_cost : ℕ := 4
def bookmark_cost : ℕ := 2

def posters : ℕ := 2
def notebooks : ℕ := 3
def bookmarks : ℕ := 2

def initial_money : ℕ := 2 * 20

def total_cost : ℕ := posters * poster_cost + notebooks * notebook_cost + bookmarks * bookmark_cost

def money_left_over : ℕ := initial_money - total_cost

theorem whitney_money_leftover : money_left_over = 14 := by
  sorry

end NUMINAMATH_GPT_whitney_money_leftover_l877_87775


namespace NUMINAMATH_GPT_average_weight_of_whole_class_l877_87768

theorem average_weight_of_whole_class :
  ∀ (n_a n_b : ℕ) (w_avg_a w_avg_b : ℝ),
    n_a = 60 →
    n_b = 70 →
    w_avg_a = 60 →
    w_avg_b = 80 →
    (n_a * w_avg_a + n_b * w_avg_b) / (n_a + n_b) = 70.77 :=
by
  intros n_a n_b w_avg_a w_avg_b h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end NUMINAMATH_GPT_average_weight_of_whole_class_l877_87768


namespace NUMINAMATH_GPT_option_c_opposites_l877_87746

theorem option_c_opposites : -|3| = -3 ∧ 3 = 3 → ( ∃ x y : ℝ, x = -3 ∧ y = 3 ∧ x = -y) :=
by
  sorry

end NUMINAMATH_GPT_option_c_opposites_l877_87746


namespace NUMINAMATH_GPT_number_divided_by_four_l877_87763

variable (x : ℝ)

theorem number_divided_by_four (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_divided_by_four_l877_87763


namespace NUMINAMATH_GPT_girls_attending_sports_event_l877_87754

theorem girls_attending_sports_event 
  (total_students attending_sports_event : ℕ) 
  (girls boys : ℕ)
  (h1 : total_students = 1500)
  (h2 : attending_sports_event = 900)
  (h3 : girls + boys = total_students)
  (h4 : (1 / 2) * girls + (3 / 5) * boys = attending_sports_event) :
  (1 / 2) * girls = 500 := 
by
  sorry

end NUMINAMATH_GPT_girls_attending_sports_event_l877_87754


namespace NUMINAMATH_GPT_total_pushups_l877_87789

theorem total_pushups (zachary_pushups : ℕ) (david_more_pushups : ℕ) 
  (h1 : zachary_pushups = 44) (h2 : david_more_pushups = 58) : 
  zachary_pushups + (zachary_pushups + david_more_pushups) = 146 :=
by
  sorry

end NUMINAMATH_GPT_total_pushups_l877_87789


namespace NUMINAMATH_GPT_ab_cd_zero_l877_87718

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1)
  (h3 : a * c + b * d = 0) : 
  a * b + c * d = 0 := 
by sorry

end NUMINAMATH_GPT_ab_cd_zero_l877_87718


namespace NUMINAMATH_GPT_uncovered_area_frame_l877_87702

def length_frame : ℕ := 40
def width_frame : ℕ := 32
def length_photo : ℕ := 32
def width_photo : ℕ := 28

def area_frame (l_f w_f : ℕ) : ℕ := l_f * w_f
def area_photo (l_p w_p : ℕ) : ℕ := l_p * w_p

theorem uncovered_area_frame :
  area_frame length_frame width_frame - area_photo length_photo width_photo = 384 :=
by
  sorry

end NUMINAMATH_GPT_uncovered_area_frame_l877_87702


namespace NUMINAMATH_GPT_replace_all_cardio_machines_cost_l877_87729

noncomputable def totalReplacementCost : ℕ :=
  let numGyms := 20
  let bikesPerGym := 10
  let treadmillsPerGym := 5
  let ellipticalsPerGym := 5
  let costPerBike := 700
  let costPerTreadmill := costPerBike * 3 / 2
  let costPerElliptical := costPerTreadmill * 2
  let totalBikes := numGyms * bikesPerGym
  let totalTreadmills := numGyms * treadmillsPerGym
  let totalEllipticals := numGyms * ellipticalsPerGym
  (totalBikes * costPerBike) + (totalTreadmills * costPerTreadmill) + (totalEllipticals * costPerElliptical)

theorem replace_all_cardio_machines_cost :
  totalReplacementCost = 455000 :=
by
  -- All the calculation steps provided as conditions and intermediary results need to be verified here.
  sorry

end NUMINAMATH_GPT_replace_all_cardio_machines_cost_l877_87729


namespace NUMINAMATH_GPT_simplify_expression_l877_87728

theorem simplify_expression (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) :
  (m / n - n / m) / (1 / m - 1 / n) = -(m + n) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l877_87728


namespace NUMINAMATH_GPT_factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l877_87752

-- Problem 1
theorem factorize_polynomial (x y : ℝ) : 
  x^2 - y^2 + 2*x - 2*y = (x - y)*(x + y + 2) := 
sorry

-- Problem 2
theorem triangle_equilateral (a b c : ℝ) (h : a^2 + c^2 - 2*b*(a - b + c) = 0) : 
  a = b ∧ b = c :=
sorry

-- Problem 3
theorem prove_2p_eq_m_plus_n (m n p : ℝ) (h : 1/4*(m - n)^2 = (p - n)*(m - p)) : 
  2*p = m + n :=
sorry

end NUMINAMATH_GPT_factorize_polynomial_triangle_equilateral_prove_2p_eq_m_plus_n_l877_87752


namespace NUMINAMATH_GPT_smallest_number_am_median_largest_l877_87740

noncomputable def smallest_number (a b c : ℕ) : ℕ :=
if a ≤ b ∧ a ≤ c then a
else if b ≤ a ∧ b ≤ c then b
else c

theorem smallest_number_am_median_largest (a b c : ℕ) (h1 : a + b + c = 90) (h2 : b = 28) (h3 : c = b + 6) :
  smallest_number a b c = 28 :=
sorry

end NUMINAMATH_GPT_smallest_number_am_median_largest_l877_87740


namespace NUMINAMATH_GPT_sum_first_20_odds_is_400_l877_87782

-- Define the n-th odd positive integer
def odd_integer (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd positive integers as a function
def sum_first_n_odds (n : ℕ) : ℕ := (n * (2 * n + 1)) / 2

-- Theorem statement: sum of the first 20 odd positive integers is 400
theorem sum_first_20_odds_is_400 : sum_first_n_odds 20 = 400 := 
  sorry

end NUMINAMATH_GPT_sum_first_20_odds_is_400_l877_87782


namespace NUMINAMATH_GPT_solution_l877_87736

noncomputable def problem : Prop :=
  (2 * Real.sin (75 * Real.pi / 180) * Real.cos (75 * Real.pi / 180) = 1 / 2) ∧
  (1 - 2 * Real.sin (Real.pi / 12) ^ 2 ≠ 1 / 2) ∧
  (Real.cos (45 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) - 
   Real.sin (45 * Real.pi / 180) * Real.sin (15 * Real.pi / 180) = 1 / 2) ∧
  ( (Real.tan (77 * Real.pi / 180) - Real.tan (32 * Real.pi / 180)) /
    (2 * (1 + Real.tan (77 * Real.pi / 180) * Real.tan (32 * Real.pi / 180))) = 1 / 2 )

theorem solution : problem :=
  by 
    sorry

end NUMINAMATH_GPT_solution_l877_87736


namespace NUMINAMATH_GPT_compute_value_of_expression_l877_87785

theorem compute_value_of_expression (p q : ℝ) (h₁ : 3 * p ^ 2 - 5 * p - 12 = 0) (h₂ : 3 * q ^ 2 - 5 * q - 12 = 0) :
  (3 * p ^ 2 - 3 * q ^ 2) / (p - q) = 5 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_of_expression_l877_87785


namespace NUMINAMATH_GPT_min_value_inequality_l877_87771

variable {a b c d : ℝ}

theorem min_value_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / b) + (b / c) + (c / d) + (d / a) ≥ 4 :=
sorry

end NUMINAMATH_GPT_min_value_inequality_l877_87771


namespace NUMINAMATH_GPT_solution_set_transformation_l877_87774

variables (a b c α β : ℝ) (h_root : (α : ℝ) > 0)

open Set

def quadratic_inequality (x : ℝ) : Prop :=
  a * x^2 + b * x + c > 0

def transformed_inequality (x : ℝ) : Prop :=
  c * x^2 + b * x + a < 0

theorem solution_set_transformation :
  (∀ x, quadratic_inequality a b c x ↔ (α < x ∧ x < β)) →
  (∃ α β : ℝ, α > 0 ∧ (∀ x, transformed_inequality c b a x ↔ (x < 1/β ∨ x > 1/α))) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_transformation_l877_87774


namespace NUMINAMATH_GPT_change_making_ways_l877_87793

-- Define the conditions
def is_valid_combination (quarters nickels pennies : ℕ) : Prop :=
  quarters ≤ 2 ∧ 25 * quarters + 5 * nickels + pennies = 50

-- Define the main statement
theorem change_making_ways : 
  ∃(num_ways : ℕ), (∀(quarters nickels pennies : ℕ), is_valid_combination quarters nickels pennies → num_ways = 18) :=
sorry

end NUMINAMATH_GPT_change_making_ways_l877_87793


namespace NUMINAMATH_GPT_jennifer_score_l877_87726

theorem jennifer_score 
  (total_questions : ℕ)
  (correct_answers : ℕ)
  (incorrect_answers : ℕ)
  (unanswered_questions : ℕ)
  (points_per_correct : ℤ)
  (points_deduction_incorrect : ℤ)
  (points_per_unanswered : ℤ)
  (h_total : total_questions = 30)
  (h_correct : correct_answers = 15)
  (h_incorrect : incorrect_answers = 10)
  (h_unanswered : unanswered_questions = 5)
  (h_points_correct : points_per_correct = 2)
  (h_deduction_incorrect : points_deduction_incorrect = -1)
  (h_points_unanswered : points_per_unanswered = 0) : 
  ∃ (score : ℤ), score = (correct_answers * points_per_correct 
                          + incorrect_answers * points_deduction_incorrect 
                          + unanswered_questions * points_per_unanswered) 
                        ∧ score = 20 := 
by
  sorry

end NUMINAMATH_GPT_jennifer_score_l877_87726


namespace NUMINAMATH_GPT_completing_square_solution_l877_87715

theorem completing_square_solution (x : ℝ) :
  2 * x^2 + 4 * x - 3 = 0 →
  (x + 1)^2 = 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_completing_square_solution_l877_87715


namespace NUMINAMATH_GPT_gcd_1343_816_l877_87749

theorem gcd_1343_816 : Nat.gcd 1343 816 = 17 := by
  sorry

end NUMINAMATH_GPT_gcd_1343_816_l877_87749


namespace NUMINAMATH_GPT_no_solutions_l877_87723

theorem no_solutions (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hne : a + b ≠ 0) :
  ¬ (1 / a + 2 / b = 3 / (a + b)) :=
by { sorry }

end NUMINAMATH_GPT_no_solutions_l877_87723


namespace NUMINAMATH_GPT_base7_to_base10_and_frac_l877_87739

theorem base7_to_base10_and_frac (c d e : ℕ) 
  (h1 : (761 : ℕ) = 7^2 * 7 + 6 * 7^1 + 1 * 7^0)
  (h2 : (10 * 10 * c + 10 * d + e) = 386)
  (h3 : c = 3)
  (h4 : d = 8)
  (h5 : e = 6) :
  (d * e) / 15 = 48 / 15 := 
sorry

end NUMINAMATH_GPT_base7_to_base10_and_frac_l877_87739


namespace NUMINAMATH_GPT_convert_15_deg_to_rad_l877_87730

theorem convert_15_deg_to_rad (deg_to_rad : ℝ := Real.pi / 180) : 
  15 * deg_to_rad = Real.pi / 12 :=
by sorry

end NUMINAMATH_GPT_convert_15_deg_to_rad_l877_87730


namespace NUMINAMATH_GPT_math_proof_problem_l877_87710

noncomputable def expr : ℚ :=
  ((5 / 8 * (3 / 7) + 1 / 4 * (2 / 6)) - (2 / 3 * (1 / 4) - 1 / 5 * (4 / 9))) * 
  ((7 / 9 * (2 / 5) * (1 / 2) * 5040 + 1 / 3 * (3 / 8) * (9 / 11) * 4230))

theorem math_proof_problem : expr = 336 := 
  by
  sorry

end NUMINAMATH_GPT_math_proof_problem_l877_87710


namespace NUMINAMATH_GPT_data_instances_in_one_hour_l877_87764

-- Definition of the given conditions
def record_interval := 5 -- device records every 5 seconds
def seconds_in_hour := 3600 -- total seconds in one hour

-- Prove that the device records 720 instances in one hour
theorem data_instances_in_one_hour : seconds_in_hour / record_interval = 720 := by
  sorry

end NUMINAMATH_GPT_data_instances_in_one_hour_l877_87764


namespace NUMINAMATH_GPT_solve_trig_inequality_l877_87727

noncomputable def sin_triple_angle_identity (x : ℝ) : ℝ :=
  3 * (Real.sin x) - 4 * (Real.sin x) ^ 3

theorem solve_trig_inequality (x : ℝ) (h1 : 0 < x) (h2 : x < Real.pi) :
  (8 / (3 * Real.sin x - sin_triple_angle_identity x) + 3 * (Real.sin x) ^ 2) ≤ 5 ↔
  x = Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_trig_inequality_l877_87727


namespace NUMINAMATH_GPT_no_prime_ratio_circle_l877_87755

theorem no_prime_ratio_circle (A : Fin 2007 → ℕ) :
  ¬ (∀ i : Fin 2007, (∃ p : ℕ, Nat.Prime p ∧ (p = A i / A ((i + 1) % 2007) ∨ p = A ((i + 1) % 2007) / A i))) := by
  sorry

end NUMINAMATH_GPT_no_prime_ratio_circle_l877_87755


namespace NUMINAMATH_GPT_quadratic_to_square_l877_87741

theorem quadratic_to_square (x h k : ℝ) : 
  (x * x - 4 * x + 3 = 0) →
  ((x + h) * (x + h) = k) →
  k = 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_to_square_l877_87741


namespace NUMINAMATH_GPT_candies_problem_l877_87786

theorem candies_problem (x : ℕ) (Nina : ℕ) (Oliver : ℕ) (total_candies : ℕ) (h1 : 4 * x = Mark) (h2 : 2 * Mark = Nina) (h3 : 6 * Nina = Oliver) (h4 : x + Mark + Nina + Oliver = total_candies) :
  x = 360 / 61 :=
by
  sorry

end NUMINAMATH_GPT_candies_problem_l877_87786


namespace NUMINAMATH_GPT_collinear_vectors_l877_87781

theorem collinear_vectors (m : ℝ) (h_collinear : 1 * m - (-2) * (-3) = 0) : m = 6 :=
by
  sorry

end NUMINAMATH_GPT_collinear_vectors_l877_87781


namespace NUMINAMATH_GPT_seungjun_clay_cost_l877_87733

theorem seungjun_clay_cost (price_per_gram : ℝ) (qty1 qty2 : ℝ) 
  (h1 : price_per_gram = 17.25) 
  (h2 : qty1 = 1000) 
  (h3 : qty2 = 10) :
  (qty1 * price_per_gram + qty2 * price_per_gram) = 17422.5 :=
by
  sorry

end NUMINAMATH_GPT_seungjun_clay_cost_l877_87733


namespace NUMINAMATH_GPT_find_s_base_10_l877_87783

-- Defining the conditions of the problem
def s_in_base_b_equals_42 (b : ℕ) : Prop :=
  let factor_1 := b + 3
  let factor_2 := b + 4
  let factor_3 := b + 5
  let produced_number := factor_1 * factor_2 * factor_3
  produced_number = 2 * b^3 + 3 * b^2 + 2 * b + 5

-- The proof problem as a Lean 4 statement
theorem find_s_base_10 :
  (∃ b : ℕ, s_in_base_b_equals_42 b) →
  13 + 14 + 15 = 42 :=
sorry

end NUMINAMATH_GPT_find_s_base_10_l877_87783


namespace NUMINAMATH_GPT_prasanna_speed_l877_87708

variable (v_L : ℝ) (d t : ℝ)

theorem prasanna_speed (hLaxmiSpeed : v_L = 18) (htime : t = 1) (hdistance : d = 45) : 
  ∃ v_P : ℝ, v_P = 27 :=
  sorry

end NUMINAMATH_GPT_prasanna_speed_l877_87708


namespace NUMINAMATH_GPT_negation_of_existential_l877_87717

theorem negation_of_existential : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 ≤ 0) := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_existential_l877_87717


namespace NUMINAMATH_GPT_best_discount_option_l877_87780

-- Define the original price
def original_price : ℝ := 100

-- Define the discount functions for each option
def option_A : ℝ := original_price * (1 - 0.20)
def option_B : ℝ := (original_price * (1 - 0.10)) * (1 - 0.10)
def option_C : ℝ := (original_price * (1 - 0.15)) * (1 - 0.05)
def option_D : ℝ := (original_price * (1 - 0.05)) * (1 - 0.15)

-- Define the theorem stating that option A gives the best price
theorem best_discount_option : option_A ≤ option_B ∧ option_A ≤ option_C ∧ option_A ≤ option_D :=
by {
  sorry
}

end NUMINAMATH_GPT_best_discount_option_l877_87780


namespace NUMINAMATH_GPT_solve_problem_l877_87757

noncomputable def proof_problem (x y : ℝ) : Prop :=
  (0.65 * x > 26) ∧ (0.40 * y < -3) ∧ ((x - y)^2 ≥ 100) 
  → (x > 40) ∧ (y < -7.5)

theorem solve_problem (x y : ℝ) (h : proof_problem x y) : (x > 40) ∧ (y < -7.5) := 
sorry

end NUMINAMATH_GPT_solve_problem_l877_87757


namespace NUMINAMATH_GPT_dot_product_property_l877_87725

noncomputable def point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

variables (x_P y_P : ℝ) (F1 F2 : ℝ × ℝ)

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0) ∨ F = (-1, 0)

def radius_of_inscribed_circle (r : ℝ) : Prop :=
  r = 1 / 2

theorem dot_product_property (h1 : point_on_ellipse x_P y_P)
  (h2 : is_focus F1) (h3 : is_focus F2) (h4: radius_of_inscribed_circle (1/2)):
  (x_P^2 - 1 + y_P^2) = 9 / 4 :=
sorry

end NUMINAMATH_GPT_dot_product_property_l877_87725


namespace NUMINAMATH_GPT_average_speed_l877_87719

theorem average_speed (speed1 speed2: ℝ) (time1 time2: ℝ) (h1: speed1 = 90) (h2: speed2 = 40) (h3: time1 = 1) (h4: time2 = 1) :
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 65 := by
  sorry

end NUMINAMATH_GPT_average_speed_l877_87719


namespace NUMINAMATH_GPT_minimum_area_l877_87778

-- Define point A
def A : ℝ × ℝ := (-4, 0)

-- Define point B
def B : ℝ × ℝ := (0, 4)

-- Define the circle
def on_circle (C : ℝ × ℝ) : Prop := (C.1 - 2)^2 + C.2^2 = 2

-- Instantiating the proof of the minimum area of △ABC = 8
theorem minimum_area (C : ℝ × ℝ) (h : on_circle C) : 
  ∃ C : ℝ × ℝ, on_circle C ∧ 1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) = 8 := 
sorry

end NUMINAMATH_GPT_minimum_area_l877_87778


namespace NUMINAMATH_GPT_range_of_t_l877_87758

theorem range_of_t (x y a t : ℝ) 
  (h1 : x + 3 * y + a = 4) 
  (h2 : x - y - 3 * a = 0) 
  (h3 : -1 ≤ a ∧ a ≤ 1) 
  (h4 : t = x + y) : 
  1 ≤ t ∧ t ≤ 3 := 
sorry

end NUMINAMATH_GPT_range_of_t_l877_87758


namespace NUMINAMATH_GPT_solve_for_y_l877_87705

theorem solve_for_y (y : ℝ) (h : (7 * y - 2) / (y + 4) - 5 / (y + 4) = 2 / (y + 4)) : 
  y = (9 / 7) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l877_87705


namespace NUMINAMATH_GPT_translation_of_graph_l877_87762

theorem translation_of_graph (f : ℝ → ℝ) (x : ℝ) :
  f x = 2 ^ x →
  f (x - 1) + 2 = 2 ^ (x - 1) + 2 :=
by
  intro
  sorry

end NUMINAMATH_GPT_translation_of_graph_l877_87762


namespace NUMINAMATH_GPT_calories_consummed_l877_87797

-- Definitions based on conditions
def calories_per_strawberry : ℕ := 4
def calories_per_ounce_of_yogurt : ℕ := 17
def strawberries_eaten : ℕ := 12
def yogurt_eaten_in_ounces : ℕ := 6

-- Theorem statement
theorem calories_consummed (c_straw : ℕ) (c_yogurt : ℕ) (straw : ℕ) (yogurt : ℕ) 
  (h1 : c_straw = calories_per_strawberry) 
  (h2 : c_yogurt = calories_per_ounce_of_yogurt) 
  (h3 : straw = strawberries_eaten) 
  (h4 : yogurt = yogurt_eaten_in_ounces) : 
  c_straw * straw + c_yogurt * yogurt = 150 :=
by 
  -- Derived conditions
  rw [h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_calories_consummed_l877_87797


namespace NUMINAMATH_GPT_find_m_l877_87766

-- Conditions given
def ellipse (x y m : ℝ) : Prop := (x^2 / m) + (y^2 / 4) = 1
def eccentricity (e : ℝ) : Prop := e = 2

-- The theorem to prove
theorem find_m (m : ℝ) (h₁ : ellipse 1 1 m) (h₂ : eccentricity 2) : m = 3 ∨ m = 5 :=
  sorry

end NUMINAMATH_GPT_find_m_l877_87766


namespace NUMINAMATH_GPT_slices_eaten_l877_87732

theorem slices_eaten (total_slices : Nat) (slices_left : Nat) (expected_slices_eaten : Nat) :
  total_slices = 32 →
  slices_left = 7 →
  expected_slices_eaten = 25 →
  total_slices - slices_left = expected_slices_eaten :=
by
  intros
  sorry

end NUMINAMATH_GPT_slices_eaten_l877_87732


namespace NUMINAMATH_GPT_find_m_of_cos_alpha_l877_87777

theorem find_m_of_cos_alpha (m : ℝ) (h₁ : (2 * Real.sqrt 5) / 5 = m / Real.sqrt (m ^ 2 + 1)) (h₂ : m > 0) : m = 2 :=
sorry

end NUMINAMATH_GPT_find_m_of_cos_alpha_l877_87777


namespace NUMINAMATH_GPT_sum_even_sub_sum_odd_l877_87784

def sum_arith_seq (a1 an d : ℕ) (n : ℕ) : ℕ :=
  n * (a1 + an) / 2

theorem sum_even_sub_sum_odd :
  let n_even := 50
  let n_odd := 15
  let s_even := sum_arith_seq 2 100 2 n_even
  let s_odd := sum_arith_seq 1 29 2 n_odd
  s_even - s_odd = 2325 :=
by
  sorry

end NUMINAMATH_GPT_sum_even_sub_sum_odd_l877_87784


namespace NUMINAMATH_GPT_sin_minus_cos_third_quadrant_l877_87709

theorem sin_minus_cos_third_quadrant (α : ℝ) (h_tan : Real.tan α = 2) (h_quadrant : π < α ∧ α < 3 * π / 2) : 
  Real.sin α - Real.cos α = -Real.sqrt 5 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_minus_cos_third_quadrant_l877_87709


namespace NUMINAMATH_GPT_line_intersects_circle_l877_87703

theorem line_intersects_circle
    (r : ℝ) (d : ℝ)
    (hr : r = 6) (hd : d = 5) : d < r :=
by
    rw [hr, hd]
    exact by norm_num

end NUMINAMATH_GPT_line_intersects_circle_l877_87703


namespace NUMINAMATH_GPT_min_value_9x_plus_3y_l877_87747

noncomputable def minimum_value_of_expression : ℝ := 6

theorem min_value_9x_plus_3y (x y : ℝ) 
  (h1 : (x - 1) * 4 + 2 * y = 0) 
  (ha : ∃ (a1 a2 : ℝ), (a1, a2) = (x - 1, 2)) 
  (hb : ∃ (b1 b2 : ℝ), (b1, b2) = (4, y)) : 
  9^x + 3^y = minimum_value_of_expression :=
by
  sorry

end NUMINAMATH_GPT_min_value_9x_plus_3y_l877_87747


namespace NUMINAMATH_GPT_ram_krish_together_time_l877_87760

theorem ram_krish_together_time : 
  let t_R := 36
  let t_K := t_R / 2
  let task_per_day_R := 1 / t_R
  let task_per_day_K := 1 / t_K
  let task_per_day_together := task_per_day_R + task_per_day_K
  let T := 1 / task_per_day_together
  T = 12 := 
by
  sorry

end NUMINAMATH_GPT_ram_krish_together_time_l877_87760


namespace NUMINAMATH_GPT_minimum_value_l877_87779

variable {a b : ℝ}

noncomputable def given_conditions (a b : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ a + 2 * b = 2

theorem minimum_value :
  given_conditions a b →
  ∃ x, x = (1 + 4 * a + 3 * b) / (a * b) ∧ x ≥ 25 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l877_87779


namespace NUMINAMATH_GPT_greatest_possible_value_of_squares_l877_87751

theorem greatest_possible_value_of_squares (a b c d : ℝ)
  (h1 : a + b = 15)
  (h2 : ab + c + d = 78)
  (h3 : ad + bc = 160)
  (h4 : cd = 96) :
  a^2 + b^2 + c^2 + d^2 ≤ 717 ∧ ∃ a b c d, a + b = 15 ∧ ab + c + d = 78 ∧ ad + bc = 160 ∧ cd = 96 ∧ a^2 + b^2 + c^2 + d^2 = 717 :=
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_squares_l877_87751


namespace NUMINAMATH_GPT_M_intersection_N_equals_M_l877_87700

variable (x a : ℝ)

def M : Set ℝ := { y | ∃ x, y = x^2 + 1 }
def N : Set ℝ := { y | ∃ a, y = 2 * a^2 - 4 * a + 1 }

theorem M_intersection_N_equals_M : M ∩ N = M := by
  sorry

end NUMINAMATH_GPT_M_intersection_N_equals_M_l877_87700
