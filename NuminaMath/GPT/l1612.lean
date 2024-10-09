import Mathlib

namespace non_degenerate_ellipse_condition_l1612_161218

theorem non_degenerate_ellipse_condition (k : ℝ) :
  (∃ (x y : ℝ), x^2 + 9 * y^2 - 6 * x + 18 * y = k) → k > -9 :=
by
  sorry

end non_degenerate_ellipse_condition_l1612_161218


namespace find_3x2y2_l1612_161207

theorem find_3x2y2 (x y : ℤ) 
  (h1 : y^2 + 3 * x^2 * y^2 = 30 * x^2 + 517) : 
  3 * x^2 * y^2 = 588 := by
  sorry

end find_3x2y2_l1612_161207


namespace sequence_is_increasing_l1612_161280

theorem sequence_is_increasing (a : ℕ → ℝ) (h : ∀ n : ℕ, a (n + 1) - a n = 2) : ∀ n : ℕ, a (n + 1) > a n :=
by
  intro n
  have h2 : a (n + 1) - a n = 2 := h n
  linarith

end sequence_is_increasing_l1612_161280


namespace number_of_small_triangles_l1612_161226

noncomputable def area_of_large_triangle (hypotenuse_large : ℝ) : ℝ :=
  let leg := hypotenuse_large / Real.sqrt 2
  (1 / 2) * (leg * leg)

noncomputable def area_of_small_triangle (hypotenuse_small : ℝ) : ℝ :=
  let leg := hypotenuse_small / Real.sqrt 2
  (1 / 2) * (leg * leg)

theorem number_of_small_triangles (hypotenuse_large : ℝ) (hypotenuse_small : ℝ) :
  hypotenuse_large = 14 → hypotenuse_small = 2 →
  let number_of_triangles := (area_of_large_triangle hypotenuse_large) / (area_of_small_triangle hypotenuse_small)
  number_of_triangles = 49 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end number_of_small_triangles_l1612_161226


namespace Suma_can_complete_in_6_days_l1612_161203

-- Define the rates for Renu and their combined rate
def Renu_rate := (1 : ℚ) / 6
def Combined_rate := (1 : ℚ) / 3

-- Define Suma's time to complete the work alone
def Suma_days := 6

-- defining the work rate Suma is required to achieve given the known rates and combined rate
def Suma_rate := Combined_rate - Renu_rate

-- Require to prove 
theorem Suma_can_complete_in_6_days : (1 / Suma_rate) = Suma_days :=
by
  -- Using the definitions provided and some basic algebra to prove the theorem 
  sorry

end Suma_can_complete_in_6_days_l1612_161203


namespace prime_p_impplies_p_eq_3_l1612_161268

theorem prime_p_impplies_p_eq_3 (p : ℕ) (hp : Prime p) (hp2 : Prime (p^2 + 2)) : p = 3 :=
sorry

end prime_p_impplies_p_eq_3_l1612_161268


namespace graph_of_equation_l1612_161288

theorem graph_of_equation :
  ∀ x y : ℝ, (2 * x - 3 * y) ^ 2 = 4 * x ^ 2 + 9 * y ^ 2 → (x = 0 ∨ y = 0) :=
by
  intros x y h
  sorry

end graph_of_equation_l1612_161288


namespace difference_of_x_y_l1612_161248

theorem difference_of_x_y :
  ∀ (x y : ℤ), x + y = 10 → x = 14 → x - y = 18 :=
by
  intros x y h1 h2
  sorry

end difference_of_x_y_l1612_161248


namespace no_linear_term_l1612_161272

theorem no_linear_term (m : ℤ) : (∀ (x : ℤ), (x^2 - x + m) * (x - 8) = x^3 - 9*x^2 - 8*m → (8 + m) = 0) → m = -8 :=
by
  sorry

end no_linear_term_l1612_161272


namespace sequence_identity_l1612_161262

noncomputable def a_n (n : ℕ) : ℝ := n + 1
noncomputable def b_n (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := (n * (n+1)) / 2  -- Sum of first n terms of arithmetic sequence
noncomputable def T_n (n : ℕ) : ℝ := 2 * (3^n - 1) / 2  -- Sum of first n terms of geometric sequence
noncomputable def c_n (n : ℕ) : ℝ := 2 * a_n n / b_n n
noncomputable def C_n (n : ℕ) : ℝ := (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))

theorem sequence_identity :
  a_n 1 = b_n 1 ∧
  2 * a_n 2 = b_n 2 ∧
  S_n 2 + T_n 2 = 13 ∧
  2 * S_n 3 = b_n 3 →
  (∀ n : ℕ, a_n n = n + 1 ∧ b_n n = 2 * 3^(n-1)) ∧
  (∀ n : ℕ, C_n n = (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))) :=
sorry

end sequence_identity_l1612_161262


namespace sum_of_solutions_eq_zero_l1612_161238

theorem sum_of_solutions_eq_zero :
  ∀ x : ℝ, (-π ≤ x ∧ x ≤ 3 * π ∧ (1 / Real.sin x + 1 / Real.cos x = 4))
  → x = 0 := sorry

end sum_of_solutions_eq_zero_l1612_161238


namespace lcm_of_numbers_l1612_161266

theorem lcm_of_numbers (a b : ℕ) (L : ℕ) 
  (h1 : a + b = 55) 
  (h2 : Nat.gcd a b = 5) 
  (h3 : (1 / (a : ℝ)) + (1 / (b : ℝ)) = 0.09166666666666666) : (Nat.lcm a b = 120) := 
sorry

end lcm_of_numbers_l1612_161266


namespace problem_statement_l1612_161200

variable (a b c : ℝ)

theorem problem_statement (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) ≥ 6 :=
sorry

end problem_statement_l1612_161200


namespace sequence_relation_l1612_161295

theorem sequence_relation
  (a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a^2 + b^2 = 3)
  (h3 : a^3 + b^3 = 4)
  (h4 : a^4 + b^4 = 7)
  (h5 : a^5 + b^5 = 11) :
  a^10 + b^10 = 123 :=
sorry

end sequence_relation_l1612_161295


namespace cost_of_one_dozen_pens_l1612_161213

theorem cost_of_one_dozen_pens
  (x : ℝ)
  (hx : 20 * x = 150) :
  12 * 5 * (150 / 20) = 450 :=
by
  sorry

end cost_of_one_dozen_pens_l1612_161213


namespace work_efficiency_ratio_l1612_161231
noncomputable section

variable (A_eff B_eff : ℚ)

-- Conditions
def efficient_together (A_eff B_eff : ℚ) : Prop := A_eff + B_eff = 1 / 12
def efficient_alone (A_eff : ℚ) : Prop := A_eff = 1 / 16

-- Theorem to prove
theorem work_efficiency_ratio (A_eff B_eff : ℚ) (h1 : efficient_together A_eff B_eff) (h2 : efficient_alone A_eff) : A_eff / B_eff = 3 := by
  sorry

end work_efficiency_ratio_l1612_161231


namespace sufficient_not_necessary_range_l1612_161208

theorem sufficient_not_necessary_range (a : ℝ) (h : ∀ x : ℝ, x > 2 → x^2 > a ∧ ¬(x^2 > a → x > 2)) : a ≤ 4 :=
by
  sorry

end sufficient_not_necessary_range_l1612_161208


namespace evaluate_g_of_h_l1612_161257

def g (x : ℝ) : ℝ := 3 * x^2 - 4

def h (x : ℝ) : ℝ := 5 * x^3 + 2

theorem evaluate_g_of_h : g (h (-2)) = 4328 := 
by
  sorry

end evaluate_g_of_h_l1612_161257


namespace last_month_games_l1612_161279

-- Definitions and conditions
def this_month := 9
def next_month := 7
def total_games := 24

-- Question to prove
theorem last_month_games : total_games - (this_month + next_month) = 8 := 
by 
  sorry

end last_month_games_l1612_161279


namespace hyperbola_range_of_m_l1612_161259

theorem hyperbola_range_of_m (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (1 + m) + (y^2) / (1 - m) = 1) → 
  (m < -1 ∨ m > 1) :=
by 
sorry

end hyperbola_range_of_m_l1612_161259


namespace final_position_relative_total_fuel_needed_l1612_161240

noncomputable def navigation_records : List ℤ := [-7, 11, -6, 10, -5]

noncomputable def fuel_consumption_rate : ℝ := 0.5

theorem final_position_relative (records : List ℤ) : 
  (records.sum = 3) := by 
  sorry

theorem total_fuel_needed (records : List ℤ) (rate : ℝ) : 
  (rate * (records.map Int.natAbs).sum = 19.5) := by 
  sorry

#check final_position_relative navigation_records
#check total_fuel_needed navigation_records fuel_consumption_rate

end final_position_relative_total_fuel_needed_l1612_161240


namespace total_eyes_in_extended_family_l1612_161216

def mom_eyes := 1
def dad_eyes := 3
def kids_eyes := 3 * 4
def moms_previous_child_eyes := 5
def dads_previous_children_eyes := 6 + 2
def dads_ex_wife_eyes := 1
def dads_ex_wifes_new_partner_eyes := 7
def child_of_ex_wife_and_partner_eyes := 8

theorem total_eyes_in_extended_family :
  mom_eyes + dad_eyes + kids_eyes + moms_previous_child_eyes + dads_previous_children_eyes +
  dads_ex_wife_eyes + dads_ex_wifes_new_partner_eyes + child_of_ex_wife_and_partner_eyes = 45 :=
by
  -- add proof here
  sorry

end total_eyes_in_extended_family_l1612_161216


namespace a_congruent_b_mod_1008_l1612_161281

theorem a_congruent_b_mod_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b - b^a = 1008) : a ≡ b [MOD 1008] :=
by
  sorry

end a_congruent_b_mod_1008_l1612_161281


namespace find_least_N_exists_l1612_161209

theorem find_least_N_exists (N : ℕ) :
  (∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℕ), 
    N = (a₁ + 2) * (b₁ + 2) * (c₁ + 2) - 8 ∧ 
    N + 1 = (a₂ + 2) * (b₂ + 2) * (c₂ + 2) - 8) ∧
  N = 55 := 
sorry

end find_least_N_exists_l1612_161209


namespace domain_of_composed_function_l1612_161245

theorem domain_of_composed_function {f : ℝ → ℝ} (h : ∀ x, -1 < x ∧ x < 1 → f x ∈ Set.Ioo (-1:ℝ) 1) :
  ∀ x, 0 < x ∧ x < 1 → f (2*x-1) ∈ Set.Ioo (-1:ℝ) 1 := by
  sorry

end domain_of_composed_function_l1612_161245


namespace circle_area_eq_pi_div_4_l1612_161298

theorem circle_area_eq_pi_div_4 :
  ∀ (x y : ℝ), 3*x^2 + 3*y^2 - 9*x + 12*y + 27 = 0 -> (π * (1 / 2)^2 = π / 4) :=
by
  sorry

end circle_area_eq_pi_div_4_l1612_161298


namespace arithmetic_series_first_term_l1612_161249

theorem arithmetic_series_first_term 
  (a d : ℝ)
  (h1 : 50 * (2 * a + 99 * d) = 1800)
  (h2 : 50 * (2 * a + 199 * d) = 6300) :
  a = -26.55 :=
by
  sorry

end arithmetic_series_first_term_l1612_161249


namespace problem_proof_l1612_161270

theorem problem_proof (x y : ℝ) (h_cond : (x + 3)^2 + |y - 2| = 0) : (x + y)^y = 1 :=
by
  sorry

end problem_proof_l1612_161270


namespace neutral_equilibrium_l1612_161286

noncomputable def equilibrium_ratio (r h : ℝ) : ℝ := r / h

theorem neutral_equilibrium (r h : ℝ) (k : ℝ) : (equilibrium_ratio r h = k) → (k = Real.sqrt 2) :=
by
  intro h1
  have h1' : (r / h = k) := h1
  sorry

end neutral_equilibrium_l1612_161286


namespace total_hens_and_cows_l1612_161250

theorem total_hens_and_cows (H C : ℕ) (hH : H = 28) (h_feet : 2 * H + 4 * C = 136) : H + C = 48 :=
by
  -- Proof goes here 
  sorry

end total_hens_and_cows_l1612_161250


namespace find_sin_2a_l1612_161237

noncomputable def problem_statement (a : ℝ) : Prop :=
a ∈ Set.Ioo (Real.pi / 2) Real.pi ∧
3 * Real.cos (2 * a) = Real.sqrt 2 * Real.sin ((Real.pi / 4) - a)

theorem find_sin_2a (a : ℝ) (h : problem_statement a) : Real.sin (2 * a) = -8 / 9 :=
sorry

end find_sin_2a_l1612_161237


namespace remainder_of_division_l1612_161293

def num : ℤ := 1346584
def divisor : ℤ := 137
def remainder : ℤ := 5

theorem remainder_of_division 
  (h : 0 <= divisor) (h' : divisor ≠ 0) : 
  num % divisor = remainder := 
sorry

end remainder_of_division_l1612_161293


namespace value_of_h_l1612_161285

theorem value_of_h (h : ℤ) : (-1)^3 + h * (-1) - 20 = 0 → h = -21 :=
by
  intro h_cond
  sorry

end value_of_h_l1612_161285


namespace employed_population_percentage_l1612_161267

variable (P : ℝ) -- Total population
variable (percentage_employed_to_population : ℝ) -- Percentage of total population employed
variable (percentage_employed_males_to_population : ℝ := 0.42) -- 42% of population are employed males
variable (percentage_employed_females_to_employed : ℝ := 0.30) -- 30% of employed people are females

theorem employed_population_percentage :
  percentage_employed_to_population = 0.60 :=
sorry

end employed_population_percentage_l1612_161267


namespace range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l1612_161263

def quadratic_has_two_distinct_positive_roots (m : ℝ) : Prop :=
  4 * m^2 - 4 * (m + 2) > 0 ∧ -2 * m > 0 ∧ m + 2 > 0

def hyperbola_with_foci_on_y_axis (m : ℝ) : Prop :=
  m + 3 < 0 ∧ 1 - 2 * m > 0

theorem range_of_m_given_q (m : ℝ) :
  hyperbola_with_foci_on_y_axis m → m < -3 :=
by
  sorry

theorem range_of_m_given_p_or_q_and_not_p_and_q (m : ℝ) :
  (quadratic_has_two_distinct_positive_roots m ∨ hyperbola_with_foci_on_y_axis m) ∧
  ¬(quadratic_has_two_distinct_positive_roots m ∧ hyperbola_with_foci_on_y_axis m) →
  (-2 < m ∧ m < -1) ∨ m < -3 :=
by
  sorry

end range_of_m_given_q_range_of_m_given_p_or_q_and_not_p_and_q_l1612_161263


namespace game_winner_l1612_161223

theorem game_winner (m n : ℕ) (hm : 1 < m) (hn : 1 < n) : 
  (mn % 2 = 1 → first_player_wins) ∧ (mn % 2 = 0 → second_player_wins) :=
sorry

end game_winner_l1612_161223


namespace man_receives_total_amount_l1612_161204
noncomputable def total_amount_received : ℝ := 
  let itemA_price := 1300
  let itemB_price := 750
  let itemC_price := 1800
  
  let itemA_loss := 0.20 * itemA_price
  let itemB_loss := 0.15 * itemB_price
  let itemC_loss := 0.10 * itemC_price

  let itemA_selling_price := itemA_price - itemA_loss
  let itemB_selling_price := itemB_price - itemB_loss
  let itemC_selling_price := itemC_price - itemC_loss

  let vat_rate := 0.12
  let itemA_vat := vat_rate * itemA_selling_price
  let itemB_vat := vat_rate * itemB_selling_price
  let itemC_vat := vat_rate * itemC_selling_price

  let final_itemA := itemA_selling_price + itemA_vat
  let final_itemB := itemB_selling_price + itemB_vat
  let final_itemC := itemC_selling_price + itemC_vat

  final_itemA + final_itemB + final_itemC

theorem man_receives_total_amount :
  total_amount_received = 3693.2 := by
  sorry

end man_receives_total_amount_l1612_161204


namespace total_balloons_after_destruction_l1612_161247

-- Define the initial numbers of balloons
def fredBalloons := 10.0
def samBalloons := 46.0
def destroyedBalloons := 16.0

-- Prove the total number of remaining balloons
theorem total_balloons_after_destruction : fredBalloons + samBalloons - destroyedBalloons = 40.0 :=
by
  sorry

end total_balloons_after_destruction_l1612_161247


namespace quadratic_roots_proof_l1612_161253

theorem quadratic_roots_proof (b c : ℤ) :
  (∀ x : ℤ, x^2 + b * x + c = 0 ↔ (x = 1 ∨ x = -2)) → (b = 1 ∧ c = -2) :=
by
  sorry

end quadratic_roots_proof_l1612_161253


namespace find_k_l1612_161224

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x ^ 2 + (k - 1) * x + 3

theorem find_k (k : ℝ) (h : ∀ x, f k x = f k (-x)) : k = 1 :=
by
  sorry

end find_k_l1612_161224


namespace license_plate_count_l1612_161275

theorem license_plate_count :
  let letters := 26
  let digits := 10
  let total_count := letters * (letters - 1) + letters
  total_count * digits = 6760 :=
by sorry

end license_plate_count_l1612_161275


namespace intersection_M_N_l1612_161294

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N:
  M ∩ N = {1, 2} :=
sorry

end intersection_M_N_l1612_161294


namespace different_tea_packets_or_miscalculation_l1612_161256

theorem different_tea_packets_or_miscalculation : 
  ∀ (n_1 n_2 : ℕ), 3 ≤ t_1 ∧ t_1 ≤ 4 ∧ 3 ≤ t_2 ∧ t_2 ≤ 4 ∧
  (74 = t_1 * x ∧ 105 = t_2 * y → x ≠ y) ∨ 
  (∃ (e_1 e_2 : ℕ), (e_1 + e_2 = 74) ∧ (e_1 + e_2 = 105) → false) :=
by
  -- Construction based on the provided mathematical problem
  sorry

end different_tea_packets_or_miscalculation_l1612_161256


namespace dealer_cash_discount_percentage_l1612_161258

-- Definitions of the given conditions
variable (C : ℝ) (n m : ℕ) (profit_p list_ratio : ℝ)
variable (h_n : n = 25) (h_m : m = 20) (h_profit : profit_p = 1.36) (h_list_ratio : list_ratio = 2)

-- The statement we need to prove
theorem dealer_cash_discount_percentage 
  (h_eff_selling_price : (m : ℝ) / n * C = profit_p * C)
  : ((list_ratio * C - (m / n * C)) / (list_ratio * C) * 100 = 60) :=
by
  sorry

end dealer_cash_discount_percentage_l1612_161258


namespace Jaron_prize_points_l1612_161233

def points_bunnies (bunnies: Nat) (points_per_bunny: Nat) : Nat :=
  bunnies * points_per_bunny

def points_snickers (snickers: Nat) (points_per_snicker: Nat) : Nat :=
  snickers * points_per_snicker

def total_points (bunny_points: Nat) (snicker_points: Nat) : Nat :=
  bunny_points + snicker_points

theorem Jaron_prize_points :
  let bunnies := 8
  let points_per_bunny := 100
  let snickers := 48
  let points_per_snicker := 25
  let bunny_points := points_bunnies bunnies points_per_bunny
  let snicker_points := points_snickers snickers points_per_snicker
  total_points bunny_points snicker_points = 2000 := 
by
  sorry

end Jaron_prize_points_l1612_161233


namespace winter_sales_l1612_161243

theorem winter_sales (T : ℕ) (spring_summer_sales : ℕ) (fall_sales : ℕ) (winter_sales : ℕ) 
  (h1 : T = 20) 
  (h2 : spring_summer_sales = 12) 
  (h3 : fall_sales = 4) 
  (h4 : T = spring_summer_sales + fall_sales + winter_sales) : 
     winter_sales = 4 := 
by 
  rw [h1, h2, h3] at h4
  linarith


end winter_sales_l1612_161243


namespace children_on_bus_l1612_161284

theorem children_on_bus (initial_children additional_children total_children : ℕ) (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = initial_children + additional_children → total_children = 64 :=
by
  -- Proof goes here
  sorry

end children_on_bus_l1612_161284


namespace pencil_length_l1612_161241

theorem pencil_length
  (R P L : ℕ)
  (h1 : P = R + 3)
  (h2 : P = L - 2)
  (h3 : R + P + L = 29) :
  L = 12 :=
by
  sorry

end pencil_length_l1612_161241


namespace polynomial_necessary_but_not_sufficient_l1612_161230

-- Definitions
def polynomial_condition (x : ℝ) : Prop := x^2 - 3 * x + 2 = 0

def specific_value : ℝ := 1

-- Theorem statement
theorem polynomial_necessary_but_not_sufficient :
  (polynomial_condition specific_value ∧ ¬ ∀ x, polynomial_condition x -> x = specific_value) :=
by
  sorry

end polynomial_necessary_but_not_sufficient_l1612_161230


namespace problem_statement_l1612_161277

theorem problem_statement (a : ℝ) (h : (a + 1/a)^2 = 12) : a^3 + 1/a^3 = 18 * Real.sqrt 3 :=
by
  -- We'll skip the proof as per instruction
  sorry

end problem_statement_l1612_161277


namespace min_value_of_sum_l1612_161217

theorem min_value_of_sum (a b : ℝ) (h1 : a > 1) (h2 : b > 0) (h3 : a + b = 2) : 
  ∃ x : ℝ, x = (1 / (a - 1) + 1 / b) ∧ x = 4 :=
by
  sorry

end min_value_of_sum_l1612_161217


namespace track_length_l1612_161210

theorem track_length (x : ℝ) : 
  (∃ B S : ℝ, B + S = x ∧ S = (x / 2 - 75) ∧ B = 75 ∧ S + 100 = x / 2 + 25 ∧ B = x / 2 - 50 ∧ B / S = (x / 2 - 50) / 100) → 
  x = 220 :=
by
  sorry

end track_length_l1612_161210


namespace exists_ellipse_l1612_161225

theorem exists_ellipse (a : ℝ) : ∃ a : ℝ, ∀ x y : ℝ, (x^2 + y^2 / a = 1) → a > 0 ∧ a ≠ 1 := 
by 
  sorry

end exists_ellipse_l1612_161225


namespace quadratic_points_order_l1612_161214

theorem quadratic_points_order (y1 y2 y3 : ℝ) :
  (y1 = -2 * (1:ℝ) ^ 2 + 4) →
  (y2 = -2 * (2:ℝ) ^ 2 + 4) →
  (y3 = -2 * (-3:ℝ) ^ 2 + 4) →
  y1 > y2 ∧ y2 > y3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end quadratic_points_order_l1612_161214


namespace tan_alpha_value_l1612_161202

theorem tan_alpha_value (α : ℝ) (h : 0 < α ∧ α < π / 2) 
  (h_tan2α : Real.tan (2 * α) = (Real.cos α) / (2 - Real.sin α) ) :
  Real.tan α = Real.sqrt 15 / 15 :=
sorry

end tan_alpha_value_l1612_161202


namespace max_perimeter_of_rectangle_with_area_36_l1612_161205

theorem max_perimeter_of_rectangle_with_area_36 :
  ∃ l w : ℕ, l * w = 36 ∧ (∀ l' w' : ℕ, l' * w' = 36 → 2 * (l + w) ≥ 2 * (l' + w')) ∧ 2 * (l + w) = 74 := 
sorry

end max_perimeter_of_rectangle_with_area_36_l1612_161205


namespace woman_waits_time_after_passing_l1612_161297

-- Definitions based only on the conditions in a)
def man_speed : ℝ := 5 -- in miles per hour
def woman_speed : ℝ := 25 -- in miles per hour
def waiting_time_man_minutes : ℝ := 20 -- in minutes

-- Equivalent proof problem statement
theorem woman_waits_time_after_passing :
  let waiting_time_man_hours := waiting_time_man_minutes / 60
  let distance_man : ℝ := man_speed * waiting_time_man_hours
  let relative_speed : ℝ := woman_speed - man_speed
  let time_woman_covers_distance_hours := distance_man / relative_speed
  let time_woman_covers_distance_minutes := time_woman_covers_distance_hours * 60
  time_woman_covers_distance_minutes = 5 :=
by
  sorry

end woman_waits_time_after_passing_l1612_161297


namespace min_value_3x_plus_4y_l1612_161220

variable (x y : ℝ)

theorem min_value_3x_plus_4y (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 
  3 * x + 4 * y ≥ 25 :=
sorry

end min_value_3x_plus_4y_l1612_161220


namespace flour_for_each_cupcake_l1612_161206

noncomputable def flour_per_cupcake (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ) : ℝ :=
  remaining_flour / num_cupcakes

theorem flour_for_each_cupcake :
  ∀ (total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes : ℝ),
    total_flour = 6 →
    remaining_flour = 2 →
    cake_flour_per_cake = 0.5 →
    cake_price = 2.5 →
    cupcake_price = 1 →
    total_revenue = 30 →
    num_cakes = 4 / 0.5 →
    num_cupcakes = 10 →
    flour_per_cupcake total_flour remaining_flour cake_flour_per_cake cake_price cupcake_price total_revenue num_cakes num_cupcakes = 0.2 :=
by intros; sorry

end flour_for_each_cupcake_l1612_161206


namespace arithmetic_sequence_a1_a9_l1612_161211

theorem arithmetic_sequence_a1_a9 
  (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_sum_456 : a 4 + a 5 + a 6 = 36) : 
  a 1 + a 9 = 24 := 
sorry

end arithmetic_sequence_a1_a9_l1612_161211


namespace prime_pattern_l1612_161274

theorem prime_pattern (n x : ℕ) (h1 : x = (10^n - 1) / 9) (h2 : Prime x) : Prime n :=
sorry

end prime_pattern_l1612_161274


namespace problem_statement_l1612_161273

theorem problem_statement (x : ℝ) (h : 7 * x = 3) : 150 * (1 / x) = 350 :=
by
  sorry

end problem_statement_l1612_161273


namespace jamesons_sword_length_l1612_161269

theorem jamesons_sword_length (c j j' : ℕ) (hC: c = 15) 
  (hJ: j = c + 23) (hJJ: j' = j - 5) : 
  j' = 2 * c + 3 := by 
  sorry

end jamesons_sword_length_l1612_161269


namespace inequality_l1612_161278

variable (a b m : ℝ)

theorem inequality (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < m) (h4 : a < b) :
  a / b < (a + m) / (b + m) :=
by
  sorry

end inequality_l1612_161278


namespace min_value_reciprocal_sum_l1612_161215

open Real

theorem min_value_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 20) :
  (1 / a + 1 / b) ≥ 1 / 5 :=
by 
  sorry

end min_value_reciprocal_sum_l1612_161215


namespace sin_2A_cos_C_l1612_161221

theorem sin_2A (A B : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) : 
  Real.sin (2 * A) = 24 / 25 :=
sorry

theorem cos_C (A B C : ℝ) (h1 : Real.sin A = 3 / 5) (h2 : Real.cos B = -5 / 13) 
  (h3 : ∀ x y z : ℝ, x + y + z = π) :
  Real.cos C = 56 / 65 :=
sorry

end sin_2A_cos_C_l1612_161221


namespace solveSystem_l1612_161265

variable {r p q x y z : ℝ}

theorem solveSystem :
  
  -- The given system of equations
  (x + r * y - q * z = 1) ∧
  (-r * x + y + p * z = r) ∧ 
  (q * x - p * y + z = -q) →

  -- Solution equivalence using determined
  x = (1 - r ^ 2 + p ^ 2 - q ^ 2) / (1 + r ^ 2 + p ^ 2 + q ^ 2) :=
by sorry

end solveSystem_l1612_161265


namespace greatest_divisor_of_consecutive_product_l1612_161261

theorem greatest_divisor_of_consecutive_product (n : ℕ) : 
  ∃ k, ∀ n, k = 24 ∧ k ∣ (n * (n + 1) * (n + 2) * (n + 3)) :=
by
  sorry

end greatest_divisor_of_consecutive_product_l1612_161261


namespace multiplication_of_decimals_l1612_161235

theorem multiplication_of_decimals : (0.4 * 0.75 = 0.30) := by
  sorry

end multiplication_of_decimals_l1612_161235


namespace harriet_trip_time_to_B_l1612_161244

variables (D : ℝ) (t1 t2 : ℝ)

-- Definitions based on the given problem
def speed_to_b_town := 100
def speed_to_a_ville := 150
def total_time := 5

-- The condition for the total time for the trip
def total_trip_time_eq := t1 / speed_to_b_town + t2 / speed_to_a_ville = total_time

-- Prove that the time Harriet took to drive from A-ville to B-town is 3 hours.
theorem harriet_trip_time_to_B (h : total_trip_time_eq D D) : t1 = 3 :=
sorry

end harriet_trip_time_to_B_l1612_161244


namespace relationship_between_M_n_and_N_n_plus_2_l1612_161246

theorem relationship_between_M_n_and_N_n_plus_2 (n : ℕ) (h : 2 ≤ n) :
  let M_n := (n * (n + 1)) / 2 + 1
  let N_n_plus_2 := n + 3
  M_n < N_n_plus_2 :=
by
  sorry

end relationship_between_M_n_and_N_n_plus_2_l1612_161246


namespace find_integers_correct_l1612_161255

noncomputable def find_integers (a b c d : ℤ) : Prop :=
  a + b + c = 6 ∧ a + b + d = 7 ∧ a + c + d = 8 ∧ b + c + d = 9

theorem find_integers_correct (a b c d : ℤ) (h : find_integers a b c d) : a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 4 :=
by
  sorry

end find_integers_correct_l1612_161255


namespace geometric_sequence_4th_term_is_2_5_l1612_161292

variables (a r : ℝ) (n : ℕ)

def geometric_term (a r : ℝ) (n : ℕ) : ℝ := a * r^(n-1)

theorem geometric_sequence_4th_term_is_2_5 (a r : ℝ)
  (h1 : a = 125) 
  (h2 : geometric_term a r 8 = 72) :
  geometric_term a r 4 = 5 / 2 := 
sorry

end geometric_sequence_4th_term_is_2_5_l1612_161292


namespace amoeba_count_after_one_week_l1612_161236

/-- An amoeba is placed in a puddle and splits into three amoebas on the same day. Each subsequent
    day, every amoeba in the puddle splits into three new amoebas. -/
theorem amoeba_count_after_one_week : 
  let initial_amoebas := 1
  let daily_split := 3
  let days := 7
  (initial_amoebas * (daily_split ^ days)) = 2187 :=
by
  sorry

end amoeba_count_after_one_week_l1612_161236


namespace extra_food_needed_l1612_161242

theorem extra_food_needed (f1 f2 : ℝ) (h1 : f1 = 0.5) (h2 : f2 = 0.9) :
  f2 - f1 = 0.4 :=
by sorry

end extra_food_needed_l1612_161242


namespace flight_cost_l1612_161287

theorem flight_cost (ground_school_cost flight_portion_addition total_cost flight_portion_cost: ℕ) 
  (h₁ : ground_school_cost = 325)
  (h₂ : flight_portion_addition = 625)
  (h₃ : flight_portion_cost = ground_school_cost + flight_portion_addition):
  flight_portion_cost = 950 :=
by
  -- placeholder for proofs
  sorry

end flight_cost_l1612_161287


namespace value_of_b_minus_a_l1612_161251

theorem value_of_b_minus_a (a b : ℝ) (h1 : |a| = 1) (h2 : |b| = 3) (h3 : a < b) : b - a = 2 ∨ b - a = 4 :=
sorry

end value_of_b_minus_a_l1612_161251


namespace least_number_to_subtract_l1612_161260

theorem least_number_to_subtract {x : ℕ} (h : x = 13604) : 
    ∃ n : ℕ, n = 32 ∧ (13604 - n) % 87 = 0 :=
by
  sorry

end least_number_to_subtract_l1612_161260


namespace tan_of_sine_plus_cosine_eq_neg_4_over_3_l1612_161283

variable {A : ℝ}

theorem tan_of_sine_plus_cosine_eq_neg_4_over_3 
  (h : Real.sin A + Real.cos A = -4/3) : 
  Real.tan A = -4/3 :=
sorry

end tan_of_sine_plus_cosine_eq_neg_4_over_3_l1612_161283


namespace possible_combinations_of_scores_l1612_161232

theorem possible_combinations_of_scores 
    (scores : Set ℕ := {0, 3, 5})
    (total_scores : ℕ := 32)
    (teams : ℕ := 3)
    : (∃ (number_of_combinations : ℕ), number_of_combinations = 255) := by
  sorry

end possible_combinations_of_scores_l1612_161232


namespace find_A_l1612_161276

def divisible_by(a b : ℕ) := b % a = 0

def valid_digit_A (A : ℕ) : Prop := (A = 0 ∨ A = 2 ∨ A = 4 ∨ A = 6 ∨ A = 8) ∧ divisible_by A 75

theorem find_A : ∃! A : ℕ, valid_digit_A A :=
by {
  sorry
}

end find_A_l1612_161276


namespace washer_and_dryer_proof_l1612_161239

noncomputable def washer_and_dryer_problem : Prop :=
  ∃ (price_of_washer price_of_dryer : ℕ),
    price_of_washer + price_of_dryer = 600 ∧
    (∃ (k : ℕ), price_of_washer = k * price_of_dryer) ∧
    price_of_dryer = 150 ∧
    price_of_washer / price_of_dryer = 3

theorem washer_and_dryer_proof : washer_and_dryer_problem :=
sorry

end washer_and_dryer_proof_l1612_161239


namespace smallest_sum_BB_b_l1612_161264

theorem smallest_sum_BB_b (B b : ℕ) (hB : 1 ≤ B ∧ B ≤ 4) (hb : b > 6) (h : 31 * B = 4 * b + 4) : B + b = 8 :=
sorry

end smallest_sum_BB_b_l1612_161264


namespace maria_savings_percentage_is_33_l1612_161296

noncomputable def regular_price : ℝ := 60
noncomputable def second_pair_price : ℝ := regular_price - (0.4 * regular_price)
noncomputable def third_pair_price : ℝ := regular_price - (0.6 * regular_price)
noncomputable def total_regular_price : ℝ := 3 * regular_price
noncomputable def total_discounted_price : ℝ := regular_price + second_pair_price + third_pair_price
noncomputable def savings : ℝ := total_regular_price - total_discounted_price
noncomputable def savings_percentage : ℝ := (savings / total_regular_price) * 100

theorem maria_savings_percentage_is_33 :
  savings_percentage = 33 :=
by
  sorry

end maria_savings_percentage_is_33_l1612_161296


namespace correct_choice_C_l1612_161229

theorem correct_choice_C (x : ℝ) : x^2 ≥ x - 1 := 
sorry

end correct_choice_C_l1612_161229


namespace total_grazing_area_l1612_161291

-- Define the dimensions of the field
def field_width : ℝ := 46
def field_height : ℝ := 20

-- Define the length of the rope
def rope_length : ℝ := 17

-- Define the radius and position of the fenced area
def fenced_radius : ℝ := 5
def fenced_distance_x : ℝ := 25
def fenced_distance_y : ℝ := 10

-- Given the conditions, prove the total grazing area
theorem total_grazing_area (field_width field_height rope_length fenced_radius fenced_distance_x fenced_distance_y : ℝ) :
  (π * rope_length^2 / 4) = 227.07 :=
by
  sorry

end total_grazing_area_l1612_161291


namespace solution_set_of_inequality_l1612_161299

theorem solution_set_of_inequality : { x : ℝ | 0 < x ∧ x < 2 } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end solution_set_of_inequality_l1612_161299


namespace double_neg_eq_pos_l1612_161222

theorem double_neg_eq_pos (n : ℤ) : -(-n) = n := by
  sorry

example : -(-2023) = 2023 :=
  double_neg_eq_pos 2023

end double_neg_eq_pos_l1612_161222


namespace lukas_averages_points_l1612_161254

theorem lukas_averages_points (total_points : ℕ) (num_games : ℕ) (average_points : ℕ)
  (h_total: total_points = 60) (h_games: num_games = 5) : average_points = total_points / num_games :=
sorry

end lukas_averages_points_l1612_161254


namespace value_of_A_l1612_161228

theorem value_of_A 
  (H M A T E: ℤ)
  (H_value: H = 10)
  (MATH_value: M + A + T + H = 35)
  (TEAM_value: T + E + A + M = 42)
  (MEET_value: M + 2*E + T = 38) : 
  A = 21 := 
by 
  sorry

end value_of_A_l1612_161228


namespace total_project_hours_l1612_161219

def research_hours : ℕ := 10
def proposal_hours : ℕ := 2
def report_hours_left : ℕ := 8

theorem total_project_hours :
  research_hours + proposal_hours + report_hours_left = 20 := 
  sorry

end total_project_hours_l1612_161219


namespace marcy_fewer_tickets_l1612_161201

theorem marcy_fewer_tickets (A M : ℕ) (h1 : A = 26) (h2 : M = 5 * A) (h3 : A + M = 150) : M - A = 104 :=
by
  sorry

end marcy_fewer_tickets_l1612_161201


namespace inequality_solution_l1612_161227

theorem inequality_solution (x : ℝ) :
  (1 / (x^2 + 1) > 4 / x + 19 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
sorry

end inequality_solution_l1612_161227


namespace estimate_sqrt_expr_l1612_161234

theorem estimate_sqrt_expr :
  2 < (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) ∧ 
  (Real.sqrt 3 * (Real.sqrt 10 - Real.sqrt 3)) < 3 := 
sorry

end estimate_sqrt_expr_l1612_161234


namespace single_cow_single_bag_l1612_161289

-- Definitions given in the problem conditions
def cows : ℕ := 26
def bags : ℕ := 26
def days : ℕ := 26

-- Statement to be proved
theorem single_cow_single_bag : (1 : ℕ) = 26 := sorry

end single_cow_single_bag_l1612_161289


namespace fraction_increase_each_year_l1612_161271

variable (initial_value : ℝ := 57600)
variable (final_value : ℝ := 72900)
variable (years : ℕ := 2)

theorem fraction_increase_each_year :
  ∃ (f : ℝ), initial_value * (1 + f)^years = final_value ∧ f = 0.125 := by
  sorry

end fraction_increase_each_year_l1612_161271


namespace percentage_increase_sides_l1612_161252

theorem percentage_increase_sides (P : ℝ) :
  (1 + P/100) ^ 2 = 1.3225 → P = 15 := 
by
  sorry

end percentage_increase_sides_l1612_161252


namespace count_ab_bc_ca_l1612_161282

noncomputable def count_ways : ℕ :=
  (Nat.choose 9 3)

theorem count_ab_bc_ca (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ a ≠ c) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) :
  (10 * a + b < 10 * b + c ∧ 10 * b + c < 10 * c + a) → count_ways = 84 :=
sorry

end count_ab_bc_ca_l1612_161282


namespace simplify_expression_l1612_161212

variable (x y : ℝ)

theorem simplify_expression : (x^2 + x * y) / (x * y) * (y^2 / (x + y)) = y := by
  sorry

end simplify_expression_l1612_161212


namespace negation_of_existence_l1612_161290

theorem negation_of_existence (h: ¬ ∃ x : ℝ, x^2 + 1 < 0) : ∀ x : ℝ, x^2 + 1 ≥ 0 :=
by
  sorry

end negation_of_existence_l1612_161290
