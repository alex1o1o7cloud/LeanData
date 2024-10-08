import Mathlib

namespace quadratic_roots_l215_215172

theorem quadratic_roots (x : ℝ) : (x^2 - 8 * x - 2 = 0) ↔ (x = 4 + 3 * Real.sqrt 2) ∨ (x = 4 - 3 * Real.sqrt 2) := by
  sorry

end quadratic_roots_l215_215172


namespace infinite_a_exists_l215_215800

theorem infinite_a_exists (n : ℕ) : ∃ (a : ℕ), ∀ (k : ℕ), ∃ (a : ℕ), n^6 + 3 * a = (n^2 + 3 * k)^3 := 
sorry

end infinite_a_exists_l215_215800


namespace original_number_exists_l215_215266

theorem original_number_exists (x : ℤ) (h1 : x * 16 = 3408) (h2 : 0.016 * 2.13 = 0.03408) : x = 213 := 
by 
  sorry

end original_number_exists_l215_215266


namespace mean_proportional_l215_215834

theorem mean_proportional (a c x : ℝ) (ha : a = 9) (hc : c = 4) (hx : x^2 = a * c) : x = 6 := by
  sorry

end mean_proportional_l215_215834


namespace ferry_speeds_l215_215688

theorem ferry_speeds (v_P v_Q : ℝ) 
  (h1: v_P = v_Q - 1) 
  (h2: 3 * v_P * 3 = v_Q * (3 + 5))
  : v_P = 8 := 
sorry

end ferry_speeds_l215_215688


namespace tenth_term_geometric_sequence_l215_215675

theorem tenth_term_geometric_sequence :
  let a : ℚ := 5
  let r : ℚ := 3 / 4
  let a_n (n : ℕ) : ℚ := a * r ^ (n - 1)
  a_n 10 = 98415 / 262144 :=
by
  sorry

end tenth_term_geometric_sequence_l215_215675


namespace math_problem_statements_l215_215006

theorem math_problem_statements :
  (∀ a : ℝ, (a = -a) → (a = 0)) ∧
  (∀ b : ℝ, (1 / b = b) ↔ (b = 1 ∨ b = -1)) ∧
  (∀ c : ℝ, (c < -1) → (1 / c > c)) ∧
  (∀ d : ℝ, (d > 1) → (1 / d < d)) ∧
  (∃ n : ℕ, n > 0 ∧ ∀ m : ℕ, m > 0 → n ≤ m) :=
by {
  sorry
}

end math_problem_statements_l215_215006


namespace original_number_is_10_l215_215954

theorem original_number_is_10 (x : ℝ) (h : 2 * x + 5 = x / 2 + 20) : x = 10 := 
by {
  sorry
}

end original_number_is_10_l215_215954


namespace fraction_books_sold_l215_215318

theorem fraction_books_sold (B : ℕ) (F : ℚ)
  (hc1 : F * B * 4 = 288)
  (hc2 : F * B + 36 = B) :
  F = 2 / 3 :=
by
  sorry

end fraction_books_sold_l215_215318


namespace gcd_power_diff_l215_215293

theorem gcd_power_diff (m n : ℕ) (h1 : m = 2^2021 - 1) (h2 : n = 2^2000 - 1) :
  Nat.gcd m n = 2097151 :=
by sorry

end gcd_power_diff_l215_215293


namespace hyperbola_eccentricity_l215_215190

theorem hyperbola_eccentricity : 
  let a := Real.sqrt 2
  let b := 1
  let c := Real.sqrt (a^2 + b^2)
  (c / a) = Real.sqrt 6 / 2 := 
by
  sorry

end hyperbola_eccentricity_l215_215190


namespace solution_of_inequality_l215_215694

theorem solution_of_inequality (a : ℝ) :
  (a = 0 → ∀ x : ℝ, ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1) ∧
  (a < 0 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ x > 1 ∨ x < 1/a)) ∧
  (0 < a ∧ a < 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1 < x ∧ x < 1/a)) ∧
  (a > 1 → ∀ x : ℝ, (ax^2 - (a + 1) * x + 1 < 0 ↔ 1/a < x ∧ x < 1)) ∧
  (a = 1 → ∀ x : ℝ, ¬(ax^2 - (a + 1) * x + 1 < 0)) :=
by
  sorry

end solution_of_inequality_l215_215694


namespace inscribed_circle_radius_A_B_D_l215_215679

theorem inscribed_circle_radius_A_B_D (AB CD: ℝ) (angleA acuteAngleD: Prop)
  (M N: Type) (MN: ℝ) (area_trapezoid: ℝ)
  (radius: ℝ) : 
  AB = 2 ∧ CD = 3 ∧ angleA ∧ acuteAngleD ∧ MN = 4 ∧ area_trapezoid = (26 * Real.sqrt 2) / 3 
  → radius = (16 * Real.sqrt 2) / (15 + Real.sqrt 129) :=
by
  intro h
  sorry

end inscribed_circle_radius_A_B_D_l215_215679


namespace find_t_given_conditions_l215_215598

variables (p t j x y : ℝ)

theorem find_t_given_conditions
  (h1 : j = 0.75 * p)
  (h2 : j = 0.80 * t)
  (h3 : t = p * (1 - t / 100))
  (h4 : x = 0.10 * t)
  (h5 : y = 0.50 * j)
  (h6 : x + y = 12) :
  t = 24 :=
by sorry

end find_t_given_conditions_l215_215598


namespace ethan_days_worked_per_week_l215_215992

-- Define the conditions
def hourly_wage : ℕ := 18
def hours_per_day : ℕ := 8
def total_earnings : ℕ := 3600
def weeks_worked : ℕ := 5

-- Compute derived values
def daily_earnings : ℕ := hourly_wage * hours_per_day
def weekly_earnings : ℕ := total_earnings / weeks_worked

-- Define the proposition to be proved
theorem ethan_days_worked_per_week : ∃ d: ℕ, d * daily_earnings = weekly_earnings ∧ d = 5 :=
by
  use 5
  simp [daily_earnings, weekly_earnings]
  sorry

end ethan_days_worked_per_week_l215_215992


namespace melanie_balloons_l215_215083

theorem melanie_balloons (joan_balloons melanie_balloons total_balloons : ℕ)
  (h_joan : joan_balloons = 40)
  (h_total : total_balloons = 81) :
  melanie_balloons = total_balloons - joan_balloons :=
by
  sorry

end melanie_balloons_l215_215083


namespace tangent_values_l215_215058

theorem tangent_values (A : ℝ) (h : A < π) (cos_A : Real.cos A = 3 / 5) :
  Real.tan A = 4 / 3 ∧ Real.tan (A + π / 4) = -7 := 
by
  sorry

end tangent_values_l215_215058


namespace value_of_g_at_x_minus_5_l215_215867

-- Definition of the function g
def g (x : ℝ) : ℝ := -3

-- The theorem we need to prove
theorem value_of_g_at_x_minus_5 (x : ℝ) : g (x - 5) = -3 := by
  sorry

end value_of_g_at_x_minus_5_l215_215867


namespace brian_shoes_l215_215703

theorem brian_shoes (J E B : ℕ) (h1 : J = E / 2) (h2 : E = 3 * B) (h3 : J + E + B = 121) : B = 22 :=
sorry

end brian_shoes_l215_215703


namespace surface_area_ratio_l215_215993

theorem surface_area_ratio (x : ℝ) (hx : x > 0) :
  let SA1 := 6 * (4 * x) ^ 2
  let SA2 := 6 * x ^ 2
  (SA1 / SA2) = 16 := by
  sorry

end surface_area_ratio_l215_215993


namespace simplify_expression_l215_215532

variable (b : ℝ)

theorem simplify_expression (h : b ≠ 2) : (2 - 1 / (1 + b / (2 - b))) = 1 + b / 2 := 
sorry

end simplify_expression_l215_215532


namespace ticket_price_increase_one_day_later_l215_215660

noncomputable def ticket_price : ℝ := 1050
noncomputable def days_before_departure : ℕ := 14
noncomputable def daily_increase_rate : ℝ := 0.05

theorem ticket_price_increase_one_day_later :
  ∀ (price : ℝ) (days : ℕ) (rate : ℝ), price = ticket_price → days = days_before_departure → rate = daily_increase_rate →
  price * rate = 52.50 :=
by
  intros price days rate hprice hdays hrate
  rw [hprice, hrate]
  exact sorry

end ticket_price_increase_one_day_later_l215_215660


namespace smallest_two_digit_number_l215_215497

theorem smallest_two_digit_number :
  ∃ (n : ℕ), 10 ≤ n ∧ n < 100 ∧
            n % 12 = 0 ∧
            n % 5 = 4 ∧
            ∀ m : ℕ, 10 ≤ m ∧ m < 100 ∧ m % 12 = 0 ∧ m % 5 = 4 → n ≤ m :=
  by {
  -- proof shows the mathematical statement is true
  sorry
}

end smallest_two_digit_number_l215_215497


namespace M_is_correct_ab_property_l215_215815

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 1|
def M : Set ℝ := {x | f x < 4}

theorem M_is_correct : M = {x | -2 < x ∧ x < 2} :=
sorry

theorem ab_property (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 2 * |a + b| < |4 + a * b| :=
sorry

end M_is_correct_ab_property_l215_215815


namespace amount_amys_money_l215_215412

def initial_dollars : ℝ := 2
def chores_payment : ℝ := 5 * 13
def birthday_gift : ℝ := 3
def total_after_gift : ℝ := initial_dollars + chores_payment + birthday_gift

def investment_percentage : ℝ := 0.20
def invested_amount : ℝ := investment_percentage * total_after_gift

def interest_rate : ℝ := 0.10
def interest_amount : ℝ := interest_rate * invested_amount
def total_investment : ℝ := invested_amount + interest_amount

def cost_of_toy : ℝ := 12
def remaining_after_toy : ℝ := total_after_gift - cost_of_toy

def grandparents_gift : ℝ := 2 * remaining_after_toy
def total_including_investment : ℝ := grandparents_gift + total_investment

def donation_percentage : ℝ := 0.25
def donated_amount : ℝ := donation_percentage * total_including_investment

def final_amount : ℝ := total_including_investment - donated_amount

theorem amount_amys_money :
  final_amount = 98.55 := by
  sorry

end amount_amys_money_l215_215412


namespace boys_girls_dance_l215_215923

theorem boys_girls_dance (b g : ℕ) 
  (h : ∀ n, (n <= b) → (n + 7) ≤ g) 
  (hb_lasts : b + 7 = g) :
  b = g - 7 := by
  sorry

end boys_girls_dance_l215_215923


namespace integer_solutions_eq_0_or_2_l215_215034

theorem integer_solutions_eq_0_or_2 (a : ℤ) (x : ℤ) : 
  (a * x^2 + 6 = 0) → (a = -6 ∧ (x = 1 ∨ x = -1)) ∨ (¬ (a = -6) ∧ (x ≠ 1) ∧ (x ≠ -1)) :=
by 
sorry

end integer_solutions_eq_0_or_2_l215_215034


namespace possible_length_of_third_side_l215_215951

theorem possible_length_of_third_side (a b c : ℤ) (h1 : a - b = 7) (h2 : (a + b + c) % 2 = 1) : c = 8 :=
sorry

end possible_length_of_third_side_l215_215951


namespace toy_sword_cost_l215_215552

theorem toy_sword_cost (L S : ℕ) (play_dough_cost total_cost : ℕ) :
    L = 250 →
    play_dough_cost = 35 →
    total_cost = 1940 →
    3 * L + 7 * S + 10 * play_dough_cost = total_cost →
    S = 120 :=
by
  intros hL h_play_dough_cost h_total_cost h_eq
  sorry

end toy_sword_cost_l215_215552


namespace choose_one_from_ten_l215_215367

theorem choose_one_from_ten :
  Nat.choose 10 1 = 10 :=
by
  sorry

end choose_one_from_ten_l215_215367


namespace num_adult_tickets_l215_215571

variables (A C : ℕ)

def num_tickets (A C : ℕ) : Prop := A + C = 900
def total_revenue (A C : ℕ) : Prop := 7 * A + 4 * C = 5100

theorem num_adult_tickets : ∃ A, ∃ C, num_tickets A C ∧ total_revenue A C ∧ A = 500 := 
by
  sorry

end num_adult_tickets_l215_215571


namespace line_equation_l215_215138

-- Define the point A(2, 1)
def A : ℝ × ℝ := (2, 1)

-- Define the notion of a line with equal intercepts on the coordinates
def line_has_equal_intercepts (c : ℝ) : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), y = m * x + b ↔ x = y ∧ y = c

-- Define the condition that the line passes through point A
def line_passes_through_A (m b : ℝ) : Prop :=
  A.2 = m * A.1 + b

-- Define the two possible equations for the line
def line_eq1 (x y : ℝ) : Prop :=
  x + y - 3 = 0

def line_eq2 (x y : ℝ) : Prop :=
  2 * x - y = 0

-- Combined conditions in a single theorem
theorem line_equation (m b c x y : ℝ) (h_pass : line_passes_through_A m b) (h_int : line_has_equal_intercepts c) :
  (line_eq1 x y ∨ line_eq2 x y) :=
sorry

end line_equation_l215_215138


namespace beta_still_water_speed_l215_215584

-- Definitions that are used in the conditions
def alpha_speed_still_water : ℝ := 56 
def beta_speed_still_water : ℝ := 52  
def water_current_speed : ℝ := 4

-- The main theorem statement 
theorem beta_still_water_speed : β_speed_still_water = 61 := 
  sorry -- the proof goes here

end beta_still_water_speed_l215_215584


namespace sqrt_inequalities_l215_215637

theorem sqrt_inequalities
  (a b c d e : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hc : 0 ≤ c ∧ c ≤ 1)
  (hd : 0 ≤ d ∧ d ≤ 1)
  (he : 0 ≤ e ∧ e ≤ 1) :
  2 * Real.sqrt 2 ≤ Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ∧
  Real.sqrt (e^2 + a^2) + Real.sqrt (a^2 + b^2) + Real.sqrt (b^2 + c^2) + Real.sqrt (c^2 + d^2) + Real.sqrt (d^2 + e^2) ≤ 5 * Real.sqrt 2 :=
by {
  sorry
}

end sqrt_inequalities_l215_215637


namespace max_constant_C_all_real_numbers_l215_215273

theorem max_constant_C_all_real_numbers:
  ∀ (x1 x2 x3 x4 x5 x6 : ℝ), 
  (x1 + x2 + x3 + x4 + x5 + x6)^2 ≥ 
  3 * (x1 * (x2 + x3) + x2 * (x3 + x4) + x3 * (x4 + x5) + x4 * (x5 + x6) + x5 * (x6 + x1) + x6 * (x1 + x2)) := 
by 
  sorry

end max_constant_C_all_real_numbers_l215_215273


namespace cafeteria_pies_l215_215738

theorem cafeteria_pies (total_apples initial_apples_per_pie held_out_apples : ℕ) (h : total_apples = 150) (g : held_out_apples = 24) (p : initial_apples_per_pie = 15) :
  ((total_apples - held_out_apples) / initial_apples_per_pie) = 8 :=
by
  -- problem-specific proof steps would go here
  sorry

end cafeteria_pies_l215_215738


namespace probability_one_instrument_l215_215876

-- Definitions based on conditions
def total_people : Nat := 800
def play_at_least_one : Nat := total_people / 5
def play_two_or_more : Nat := 32
def play_exactly_one : Nat := play_at_least_one - play_two_or_more

-- Target statement to prove the equivalence
theorem probability_one_instrument: (play_exactly_one : ℝ) / (total_people : ℝ) = 0.16 := by
  sorry

end probability_one_instrument_l215_215876


namespace smallest_abs_sum_l215_215321

open Matrix

noncomputable def matrix_square_eq (a b c d : ℤ) : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![a, b], ![c, d]] * ![![a, b], ![c, d]]

theorem smallest_abs_sum (a b c d : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
    (M_eq : matrix_square_eq a b c d = ![![9, 0], ![0, 9]]) :
    |a| + |b| + |c| + |d| = 8 :=
sorry

end smallest_abs_sum_l215_215321


namespace water_charge_rel_water_usage_from_charge_l215_215589

-- Define the conditions and functional relationship
theorem water_charge_rel (x : ℝ) (hx : x > 5) : y = 3.5 * x - 7.5 :=
  sorry

-- Prove the specific case where the charge y is 17 yuan
theorem water_usage_from_charge (h : 17 = 3.5 * x - 7.5) :
  x = 7 :=
  sorry

end water_charge_rel_water_usage_from_charge_l215_215589


namespace nancy_target_amount_l215_215592

theorem nancy_target_amount {rate : ℝ} {hours : ℝ} (h1 : rate = 28 / 4) (h2 : hours = 10) : 28 / 4 * 10 = 70 :=
by
  sorry

end nancy_target_amount_l215_215592


namespace original_group_size_l215_215539

theorem original_group_size (M : ℕ) (R : ℕ) :
  (M * R * 40 = (M - 5) * R * 50) → M = 25 :=
by
  sorry

end original_group_size_l215_215539


namespace a_and_b_together_finish_in_40_days_l215_215727

theorem a_and_b_together_finish_in_40_days (D : ℕ) 
    (W : ℕ)
    (day_with_b : ℕ)
    (remaining_days_a : ℕ)
    (a_alone_days : ℕ)
    (a_b_together : D = 40)
    (ha : (remaining_days_a = 15) ∧ (a_alone_days = 20) ∧ (day_with_b = 10))
    (work_done_total : 10 * (W / D) + 15 * (W / a_alone_days) = W) :
    D = 40 := 
    sorry

end a_and_b_together_finish_in_40_days_l215_215727


namespace find_f_of_1_over_2016_l215_215004

noncomputable def f (x : ℝ) : ℝ := sorry

lemma f_property_0 : f 0 = 0 := sorry
lemma f_property_1 (x : ℝ) : f x + f (1 - x) = 1 := sorry
lemma f_property_2 (x : ℝ) : f (x / 3) = (1 / 2) * f x := sorry
lemma f_property_3 {x₁ x₂ : ℝ} (h₀ : 0 ≤ x₁) (h₁ : x₁ < x₂) (h₂ : x₂ ≤ 1): f x₁ ≤ f x₂ := sorry

theorem find_f_of_1_over_2016 : f (1 / 2016) = 1 / 128 := sorry

end find_f_of_1_over_2016_l215_215004


namespace weight_loss_l215_215219

def initial_weight : ℕ := 69
def current_weight : ℕ := 34

theorem weight_loss :
  initial_weight - current_weight = 35 :=
by
  sorry

end weight_loss_l215_215219


namespace find_x_l215_215213

-- Define the digits used
def digits : List ℕ := [1, 4, 5]

-- Define the sum of all four-digit numbers formed
def sum_of_digits (x : ℕ) : ℕ :=
  24 * (1 + 4 + 5 + x)

-- State the theorem
theorem find_x (x : ℕ) (h : sum_of_digits x = 288) : x = 2 :=
  by
    sorry

end find_x_l215_215213


namespace distinct_pos_real_ints_l215_215387

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

theorem distinct_pos_real_ints (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) (h4 : ∀ n : ℕ, (floor (n * a)) ∣ (floor (n * b))) : ∃ k l : ℤ, a = k ∧ b = l :=
by
  sorry

end distinct_pos_real_ints_l215_215387


namespace jessica_found_seashells_l215_215669

-- Define the given conditions
def mary_seashells : ℕ := 18
def total_seashells : ℕ := 59

-- Define the goal for the number of seashells Jessica found
def jessica_seashells (mary_seashells total_seashells : ℕ) : ℕ := total_seashells - mary_seashells

-- The theorem stating Jessica found 41 seashells
theorem jessica_found_seashells : jessica_seashells mary_seashells total_seashells = 41 := by
  -- We assume the conditions and skip the proof
  sorry

end jessica_found_seashells_l215_215669


namespace Andrew_has_5_more_goats_than_twice_Adam_l215_215438

-- Definitions based on conditions
def goats_Adam := 7
def goats_Ahmed := 13
def goats_Andrew := goats_Ahmed + 6
def twice_goats_Adam := 2 * goats_Adam

-- Theorem statement
theorem Andrew_has_5_more_goats_than_twice_Adam :
  goats_Andrew - twice_goats_Adam = 5 :=
by
  sorry

end Andrew_has_5_more_goats_than_twice_Adam_l215_215438


namespace otimes_2_1_equals_3_l215_215211

namespace MathProof

-- Define the operation
def otimes (a b : ℝ) : ℝ := a^2 - b

-- The main theorem to prove
theorem otimes_2_1_equals_3 : otimes 2 1 = 3 :=
by
  -- Proof content not needed
  sorry

end MathProof

end otimes_2_1_equals_3_l215_215211


namespace buffalo_weight_rounding_l215_215193

theorem buffalo_weight_rounding
  (weight_kg : ℝ) (conversion_factor : ℝ) (expected_weight_lb : ℕ) :
  weight_kg = 850 →
  conversion_factor = 0.454 →
  expected_weight_lb = 1872 →
  Nat.floor (weight_kg / conversion_factor + 0.5) = expected_weight_lb :=
by
  intro h1 h2 h3
  sorry

end buffalo_weight_rounding_l215_215193


namespace minimum_value_fraction_l215_215963

theorem minimum_value_fraction (x : ℝ) (h : x > 6) : (∃ c : ℝ, c = 12 ∧ ((x = c) → (x^2 / (x - 6) = 18)))
  ∧ (∀ y : ℝ, y > 6 → y^2 / (y - 6) ≥ 18) :=
by {
  sorry
}

end minimum_value_fraction_l215_215963


namespace paint_ratio_l215_215182

theorem paint_ratio
  (blue yellow white : ℕ)
  (ratio_b : ℕ := 4)
  (ratio_y : ℕ := 3)
  (ratio_w : ℕ := 5)
  (total_white : ℕ := 15)
  : yellow = 9 := by
  have ratio := ratio_b + ratio_y + ratio_w
  have white_parts := total_white * ratio_w / ratio_w
  have yellow_parts := white_parts * ratio_y / ratio_w
  exact sorry

end paint_ratio_l215_215182


namespace number_of_height_groups_l215_215924

theorem number_of_height_groups
  (max_height : ℕ) (min_height : ℕ) (class_width : ℕ)
  (h_max : max_height = 186)
  (h_min : min_height = 167)
  (h_class_width : class_width = 3) :
  (max_height - min_height + class_width - 1) / class_width = 7 := by
  sorry

end number_of_height_groups_l215_215924


namespace geom_seq_common_ratio_l215_215333

theorem geom_seq_common_ratio (S_3 S_6 : ℕ) (h1 : S_3 = 7) (h2 : S_6 = 63) : 
  ∃ q : ℕ, q = 2 := 
by
  sorry

end geom_seq_common_ratio_l215_215333


namespace solve_equation_l215_215879

theorem solve_equation (x y : ℤ) (h : 3 * (y - 2) = 5 * (x - 1)) :
  (x = 1 ∧ y = 2) ∨ (x = 4 ∧ y = 7) :=
sorry

end solve_equation_l215_215879


namespace polynomial_coeff_sum_l215_215697

theorem polynomial_coeff_sum (a0 a1 a2 a3 : ℝ) :
  (∀ x : ℝ, (2 * x + 1)^3 = a3 * x^3 + a2 * x^2 + a1 * x + a0) →
  a0 + a1 + a2 + a3 = 27 :=
by
  sorry

end polynomial_coeff_sum_l215_215697


namespace numer_greater_than_denom_iff_l215_215509

theorem numer_greater_than_denom_iff (x : ℝ) (h : -1 ≤ x ∧ x ≤ 3) : 
  (4 * x - 3 > 9 - 2 * x) ↔ (2 < x ∧ x ≤ 3) :=
sorry

end numer_greater_than_denom_iff_l215_215509


namespace shorter_steiner_network_l215_215752

-- Define the variables and inequality
noncomputable def side_length (a : ℝ) : ℝ := a
noncomputable def diagonal_network_length (a : ℝ) : ℝ := 2 * a * Real.sqrt 2
noncomputable def steiner_network_length (a : ℝ) : ℝ := a * (1 + Real.sqrt 3)

theorem shorter_steiner_network {a : ℝ} (h₀ : 0 < a) :
  diagonal_network_length a > steiner_network_length a :=
by
  -- Proof to be provided (skipping it with sorry)
  sorry

end shorter_steiner_network_l215_215752


namespace perp_bisector_eq_l215_215626

noncomputable def C1 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.1 - 7 = 0 }
noncomputable def C2 := { p : ℝ × ℝ | (p.1)^2 + (p.2)^2 - 6 * p.2 - 27 = 0 }

theorem perp_bisector_eq :
  ∃ x y, ( (x, y) ∈ C1 ∧ (x, y) ∈ C2 ) -> ( x - y = 0 ) :=
by
  sorry

end perp_bisector_eq_l215_215626


namespace interval_of_decrease_l215_215125

def quadratic (x : ℝ) := 3 * x^2 - 7 * x + 2

def decreasing_interval (y : ℝ) := y < 2 / 3

theorem interval_of_decrease :
  {x : ℝ | x < (1 / 3)} = {x : ℝ | x < (1 / 3)} :=
by sorry

end interval_of_decrease_l215_215125


namespace xy_equals_nine_l215_215136

theorem xy_equals_nine (x y : ℝ) (h : x * (x + 2 * y) = x ^ 2 + 18) : x * y = 9 :=
by
  sorry

end xy_equals_nine_l215_215136


namespace cost_price_of_watch_l215_215806

/-
Let's state the problem conditions as functions
C represents the cost price
SP1 represents the selling price at 36% loss
SP2 represents the selling price at 4% gain
-/

def cost_price (C : ℝ) : ℝ := C

def selling_price_loss (C : ℝ) : ℝ := 0.64 * C

def selling_price_gain (C : ℝ) : ℝ := 1.04 * C

def price_difference (C : ℝ) : ℝ := (selling_price_gain C) - (selling_price_loss C)

theorem cost_price_of_watch : ∀ C : ℝ, price_difference C = 140 → C = 350 :=
by
   intro C H
   sorry

end cost_price_of_watch_l215_215806


namespace factorize_expression_l215_215757

variable {a b x y : ℝ}

theorem factorize_expression :
  (x^2 - y^2) * a^2 - (x^2 - y^2) * b^2 = (x + y) * (x - y) * (a + b) * (a - b) :=
by
  sorry

end factorize_expression_l215_215757


namespace maximum_cells_covered_at_least_five_times_l215_215133

theorem maximum_cells_covered_at_least_five_times :
  let areas := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let total_covered := List.sum areas
  let exact_coverage := 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4
  let remaining_coverage := total_covered - exact_coverage
  let max_cells_covered_at_least_five := remaining_coverage / 5
  max_cells_covered_at_least_five = 5 :=
by
  sorry

end maximum_cells_covered_at_least_five_times_l215_215133


namespace part1_part2_l215_215994

def traditional_chinese_paintings : ℕ := 6
def oil_paintings : ℕ := 4
def watercolor_paintings : ℕ := 5

theorem part1 :
  traditional_chinese_paintings * oil_paintings * watercolor_paintings = 120 :=
by
  sorry

theorem part2 :
  (traditional_chinese_paintings * oil_paintings) + 
  (traditional_chinese_paintings * watercolor_paintings) + 
  (oil_paintings * watercolor_paintings) = 74 :=
by
  sorry

end part1_part2_l215_215994


namespace sum_of_ages_l215_215102

def Maria_age (E : ℕ) : ℕ := E + 7

theorem sum_of_ages (M E : ℕ) (h1 : M = E + 7) (h2 : M + 10 = 3 * (E - 5)) :
  M + E = 39 :=
by
  sorry

end sum_of_ages_l215_215102


namespace inequality_sum_geq_three_l215_215054

theorem inequality_sum_geq_three
  (a b c : ℝ)
  (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) + 
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := 
sorry

end inequality_sum_geq_three_l215_215054


namespace total_beads_sue_necklace_l215_215165

theorem total_beads_sue_necklace (purple blue green : ℕ) (h1 : purple = 7)
  (h2 : blue = 2 * purple) (h3 : green = blue + 11) : 
  purple + blue + green = 46 := 
by 
  sorry

end total_beads_sue_necklace_l215_215165


namespace maximum_BD_cyclic_quad_l215_215415

theorem maximum_BD_cyclic_quad (AB BC CD : ℤ) (BD : ℝ)
  (h_side_bounds : AB < 15 ∧ BC < 15 ∧ CD < 15)
  (h_distinct_sides : AB ≠ BC ∧ BC ≠ CD ∧ CD ≠ AB)
  (h_AB_value : AB = 13)
  (h_BC_value : BC = 5)
  (h_CD_value : CD = 8)
  (h_sides_product : BC * CD = AB * (10 : ℤ)) :
  BD = Real.sqrt 179 := 
by 
  sorry

end maximum_BD_cyclic_quad_l215_215415


namespace gcd_lcm_of_45_and_150_l215_215341

def GCD (a b : ℕ) : ℕ := Nat.gcd a b
def LCM (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

theorem gcd_lcm_of_45_and_150 :
  GCD 45 150 = 15 ∧ LCM 45 150 = 450 :=
by
  sorry

end gcd_lcm_of_45_and_150_l215_215341


namespace exists_within_distance_l215_215292

theorem exists_within_distance (a : ℝ) (n : ℕ) (h₁ : a > 0) (h₂ : n > 0) :
  ∃ k : ℕ, k < n ∧ ∃ m : ℤ, |k * a - m| < 1 / n :=
by
  sorry

end exists_within_distance_l215_215292


namespace find_root_and_m_l215_215930

theorem find_root_and_m (m x₂ : ℝ) (h₁ : (1 : ℝ) * x₂ = 3) (h₂ : (1 : ℝ) + x₂ = -m) : 
  x₂ = 3 ∧ m = -4 :=
by
  sorry

end find_root_and_m_l215_215930


namespace asha_money_remaining_l215_215327

-- Given conditions as definitions in Lean
def borrowed_from_brother : ℕ := 20
def borrowed_from_father : ℕ := 40
def borrowed_from_mother : ℕ := 30
def gift_from_granny : ℕ := 70
def initial_savings : ℕ := 100

-- Total amount of money Asha has
def total_money : ℕ := borrowed_from_brother + borrowed_from_father + borrowed_from_mother + gift_from_granny + initial_savings

-- Money spent by Asha
def money_spent : ℕ := (3 * total_money) / 4

-- Money remaining with Asha
def money_remaining : ℕ := total_money - money_spent

-- Theorem stating the result
theorem asha_money_remaining : money_remaining = 65 := by
  sorry

end asha_money_remaining_l215_215327


namespace equations_create_24_l215_215943

theorem equations_create_24 :
  ∃ (eq1 eq2 : ℤ),
  ((eq1 = 3 * (-6 + 4 + 10) ∧ eq1 = 24) ∧ 
   (eq2 = 4 - (-6 / 3) * 10 ∧ eq2 = 24)) ∧ 
   eq1 ≠ eq2 := 
by
  sorry

end equations_create_24_l215_215943


namespace daphne_necklaces_l215_215597

/--
Given:
1. Total cost of necklaces and earrings is $240,000.
2. Necklaces are equal in price.
3. Earrings were three times as expensive as any one necklace.
4. Cost of a single necklace is $40,000.

Prove:
Princess Daphne bought 3 necklaces.
-/
theorem daphne_necklaces (total_cost : ℤ) (price_necklace : ℤ) (price_earrings : ℤ) (n : ℤ)
  (h1 : total_cost = 240000)
  (h2 : price_necklace = 40000)
  (h3 : price_earrings = 3 * price_necklace)
  (h4 : total_cost = n * price_necklace + price_earrings) : n = 3 :=
by
  sorry

end daphne_necklaces_l215_215597


namespace g_ab_eq_zero_l215_215585

def g (x : ℤ) : ℤ := x^2 - 2013 * x

theorem g_ab_eq_zero (a b : ℤ) (h1 : g a = g b) (h2 : a ≠ b) : g (a + b) = 0 :=
by
  sorry

end g_ab_eq_zero_l215_215585


namespace closest_square_to_350_l215_215620

def closest_perfect_square (n : ℤ) : ℤ :=
  if (n - 18 * 18) < (19 * 19 - n) then 18 * 18 else 19 * 19

theorem closest_square_to_350 : closest_perfect_square 350 = 361 :=
by
  -- The actual proof would be provided here.
  sorry

end closest_square_to_350_l215_215620


namespace greatest_root_of_g_l215_215748

noncomputable def g (x : ℝ) : ℝ := 10 * x^4 - 16 * x^2 + 6

theorem greatest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → y ≤ x := 
by
  sorry

end greatest_root_of_g_l215_215748


namespace gcd_840_1764_gcd_561_255_l215_215039

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := 
by
  sorry

theorem gcd_561_255 : Nat.gcd 561 255 = 51 :=
by
  sorry

end gcd_840_1764_gcd_561_255_l215_215039


namespace corresponding_angles_equal_l215_215215

theorem corresponding_angles_equal 
  (α β γ : ℝ) 
  (h1 : α + β + γ = 180) 
  (h2 : (180 - α) + β + γ = 180) : 
  α = 90 ∧ β + γ = 90 ∧ (180 - α = 90) :=
by
  sorry

end corresponding_angles_equal_l215_215215


namespace average_rate_l215_215817

theorem average_rate (distance_run distance_swim : ℝ) (rate_run rate_swim : ℝ) 
  (h1 : distance_run = 2) (h2 : distance_swim = 2) (h3 : rate_run = 10) (h4 : rate_swim = 5) : 
  (distance_run + distance_swim) / ((distance_run / rate_run) * 60 + (distance_swim / rate_swim) * 60) = 0.1111 :=
by
  sorry

end average_rate_l215_215817


namespace range_of_b_l215_215908

theorem range_of_b (a b : ℝ) (h₁ : a ≤ -1) (h₂ : a * 2 * b - b - 3 * a ≥ 0) : b ≤ 1 := by
  sorry

end range_of_b_l215_215908


namespace find_x_l215_215707

theorem find_x (x : ℝ) (h_pos : x > 0) (h_area : (1 / 2) * x * (3 * x) = 54) : x = 6 :=
by
  sorry

end find_x_l215_215707


namespace original_faculty_size_l215_215181

theorem original_faculty_size (F : ℝ) (h1 : F * 0.85 * 0.80 = 195) : F = 287 :=
by
  sorry

end original_faculty_size_l215_215181


namespace part1_part2_l215_215899

open Set

variable {U : Type} [TopologicalSpace U]

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def set_B (k : ℝ) : Set ℝ := {x | x ≤ k}

theorem part1 (k : ℝ) (hk : k = 1) :
  A ∩ (univ \ set_B k) = {x | 1 < x ∧ x < 3} :=
by
  sorry

theorem part2 (k : ℝ) (h : set_A ∩ set_B k ≠ ∅) :
  k ≥ -1 :=
by
  sorry

end part1_part2_l215_215899


namespace find_a_value_l215_215169

theorem find_a_value (a a_1 a_2 a_3 a_4 a_5 : ℝ) :
  (∀ x : ℝ, (x + 1)^5 = a + a_1 * (x - 1) + a_2 * (x - 1)^2 + a_3 * (x - 1)^3 + a_4 * (x - 1)^4 + a_5 * (x - 1)^5) → 
  a = 32 :=
by
  sorry

end find_a_value_l215_215169


namespace david_profit_l215_215494

def weight : ℝ := 50
def cost : ℝ := 50
def price_per_kg : ℝ := 1.20
def total_earnings : ℝ := weight * price_per_kg
def profit : ℝ := total_earnings - cost

theorem david_profit : profit = 10 := by
  sorry

end david_profit_l215_215494


namespace productivity_increase_l215_215313

theorem productivity_increase :
  (∃ d : ℝ, 
   (∀ n : ℕ, 0 < n → n ≤ 30 → 
      (5 + (n - 1) * d ≥ 0) ∧ 
      (30 * 5 + (30 * 29 / 2) * d = 390) ∧ 
      1 / 100 < d ∧ d < 1) ∧
      d = 0.52) :=
sorry

end productivity_increase_l215_215313


namespace both_boys_and_girls_selected_probability_l215_215271

theorem both_boys_and_girls_selected_probability :
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) :=
by
  let total_students := 5
  let boys := 2
  let girls := 3
  let selected_students := 3
  let total_ways := Nat.choose total_students selected_students
  let only_girls_ways := Nat.choose girls selected_students
  have h : (only_girls_ways / total_ways : ℚ) = (1 / 10 : ℚ) := sorry
  have h1 : (1 - (only_girls_ways / total_ways : ℚ)) = (9 / 10 : ℚ) := by rw [h]; norm_num
  exact h1

end both_boys_and_girls_selected_probability_l215_215271


namespace solve_y_equation_l215_215890

theorem solve_y_equation :
  ∃ y : ℚ, 4 * (5 * y + 3) - 3 = -3 * (2 - 8 * y) ∧ y = 15 / 4 :=
by
  sorry

end solve_y_equation_l215_215890


namespace coefficient_of_x3_in_expansion_l215_215783

theorem coefficient_of_x3_in_expansion :
  (∀ (x : ℝ), (Polynomial.coeff ((Polynomial.C x - 1)^5) 3) = 10) :=
by
  sorry

end coefficient_of_x3_in_expansion_l215_215783


namespace max_profit_l215_215733

noncomputable def fixed_cost : ℝ := 2.5
noncomputable def var_cost (x : ℕ) : ℝ :=
  if x < 80 then (x^2 + 10 * x) * 1e4
  else (51 * x - 1450) * 1e4
noncomputable def revenue (x : ℕ) : ℝ := 500 * x * 1e4
noncomputable def profit (x : ℕ) : ℝ := revenue x - var_cost x - fixed_cost * 1e4

theorem max_profit (x : ℕ) :
  (∀ y : ℕ, profit y ≤ 43200 * 1e4) ∧ profit 100 = 43200 * 1e4 := by
  sorry

end max_profit_l215_215733


namespace problem1_problem2_problem3_l215_215256

-- Problem 1
theorem problem1 (m : ℝ) (h : -m^2 = m) : m^2 + m + 1 = 1 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) (h : m - n = 2) : 2 * (n - m) - 4 * m + 4 * n - 3 = -15 :=
by sorry

-- Problem 3
theorem problem3 (m n : ℝ) (h1 : m^2 + 2 * m * n = -2) (h2 : m * n - n^2 = -4) : 
  3 * m^2 + (9 / 2) * m * n + (3 / 2) * n^2 = 0 :=
by sorry

end problem1_problem2_problem3_l215_215256


namespace problem_l215_215026

theorem problem (a b : ℝ) (h : a > b) : a / 3 > b / 3 :=
sorry

end problem_l215_215026


namespace steve_family_time_l215_215518

theorem steve_family_time :
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours * assignments_fraction
  let total_activity_hours := sleeping_hours + school_hours + assignments_hours
  day_hours - total_activity_hours = 10 :=
by
  let day_hours := 24
  let sleeping_fraction := 1 / 3
  let school_fraction := 1 / 6
  let assignments_fraction := 1 / 12
  let sleeping_hours := day_hours * sleeping_fraction
  let school_hours := day_hours * school_fraction
  let assignments_hours := day_hours *  assignments_fraction
  let total_activity_hours := sleeping_hours +
                              school_hours + 
                              assignments_hours
  show day_hours - total_activity_hours = 10
  sorry

end steve_family_time_l215_215518


namespace evaluate_g_at_5_l215_215431

def g (x : ℝ) : ℝ := 5 * x + 2

theorem evaluate_g_at_5 : g 5 = 27 := by
  sorry

end evaluate_g_at_5_l215_215431


namespace positive_number_representation_l215_215092

theorem positive_number_representation (a : ℝ) : 
  (a > 0) ↔ (a ≠ 0 ∧ a > 0 ∧ ¬(a < 0)) :=
by 
  sorry

end positive_number_representation_l215_215092


namespace min_checkout_counters_l215_215649

variable (n : ℕ)
variable (x y : ℝ)

-- Conditions based on problem statement
axiom cond1 : 40 * y = 20 * x + n
axiom cond2 : 36 * y = 12 * x + n

theorem min_checkout_counters (m : ℕ) (h : 6 * m * y > 6 * x + n) : m ≥ 6 :=
  sorry

end min_checkout_counters_l215_215649


namespace natural_numbers_equal_power_l215_215347

theorem natural_numbers_equal_power
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n :=
by
  sorry

end natural_numbers_equal_power_l215_215347


namespace original_total_movies_is_293_l215_215051

noncomputable def original_movies (dvd_to_bluray_ratio : ℕ × ℕ) (additional_blurays : ℕ) (new_ratio : ℕ × ℕ) : ℕ :=
  let original_dvds := dvd_to_bluray_ratio.1
  let original_blurays := dvd_to_bluray_ratio.2
  let added_blurays := additional_blurays
  let new_dvds := new_ratio.1
  let new_blurays := new_ratio.2
  let x := (new_dvds * original_blurays - new_blurays * original_dvds) / (new_blurays * original_dvds - added_blurays * original_blurays)
  let total_movies := (original_dvds * x + original_blurays * x)
  let blurays_after_purchase := original_blurays * x + added_blurays

  if (new_dvds * (original_blurays * x + added_blurays) = new_blurays * (original_dvds * x))
  then 
    (original_dvds * x + original_blurays * x)
  else
    0 -- This case should theoretically never happen if the input ratios are consistent.

theorem original_total_movies_is_293 : original_movies (7, 2) 5 (13, 4) = 293 :=
by sorry

end original_total_movies_is_293_l215_215051


namespace jessica_balloon_count_l215_215359

theorem jessica_balloon_count :
  (∀ (joan_initial_balloon_count sally_popped_balloon_count total_balloon_count: ℕ),
  joan_initial_balloon_count = 9 →
  sally_popped_balloon_count = 5 →
  total_balloon_count = 6 →
  ∃ (jessica_balloon_count: ℕ),
    jessica_balloon_count = total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count) →
    jessica_balloon_count = 2) :=
by
  intros joan_initial_balloon_count sally_popped_balloon_count total_balloon_count j1 j2 t1
  use total_balloon_count - (joan_initial_balloon_count - sally_popped_balloon_count)
  sorry

end jessica_balloon_count_l215_215359


namespace plane_eq_unique_l215_215837

open Int 

def plane_eq (A B C D x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

theorem plane_eq_unique (x y z : ℤ) (A B C D : ℤ)
  (h₁ : x = 8) 
  (h₂ : y = -6) 
  (h₃ : z = 2) 
  (h₄ : A > 0)
  (h₅ : gcd (|A|) (gcd (|B|) (gcd (|C|) (|D|))) = 1) :
  plane_eq 4 (-3) 1 (-52) x y z :=
by
  sorry

end plane_eq_unique_l215_215837


namespace average_of_other_two_numbers_l215_215311

theorem average_of_other_two_numbers
  (avg_5_numbers : ℕ → ℚ)
  (sum_3_numbers : ℕ → ℚ)
  (h1 : ∀ n, avg_5_numbers n = 20)
  (h2 : ∀ n, sum_3_numbers n = 48)
  (h3 : ∀ n, ∃ x y z p q : ℚ, avg_5_numbers n = (x + y + z + p + q) / 5)
  (h4 : ∀ n, sum_3_numbers n = x + y + z) :
  ∃ u v : ℚ, ((u + v) / 2 = 26) :=
by sorry

end average_of_other_two_numbers_l215_215311


namespace unique_n_for_50_percent_mark_l215_215966

def exam_conditions (n : ℕ) : Prop :=
  let correct_first_20 : ℕ := 15
  let remaining : ℕ := n - 20
  let correct_remaining : ℕ := remaining / 3
  let total_correct : ℕ := correct_first_20 + correct_remaining
  total_correct * 2 = n

theorem unique_n_for_50_percent_mark : ∃! (n : ℕ), exam_conditions n := sorry

end unique_n_for_50_percent_mark_l215_215966


namespace max_sum_of_digits_l215_215251

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem max_sum_of_digits : ∃ h m : ℕ, h < 24 ∧ m < 60 ∧
  sum_of_digits h + sum_of_digits m = 24 :=
by
  sorry

end max_sum_of_digits_l215_215251


namespace puzzle_solution_l215_215299

-- Definitions for the digits
def K : ℕ := 3
def O : ℕ := 2
def M : ℕ := 4
def R : ℕ := 5
def E : ℕ := 6

-- The main proof statement
theorem puzzle_solution : (10 * K + O : ℕ) + (M / 10 + K / 10 + O / 100) = (10 * K + R : ℕ) + (O / 10 + M / 100) := 
  by 
  sorry

end puzzle_solution_l215_215299


namespace segment_length_eq_ten_l215_215728

theorem segment_length_eq_ten (x : ℝ) (h : |x - 3| = 5) : |8 - (-2)| = 10 :=
by {
  sorry
}

end segment_length_eq_ten_l215_215728


namespace multiplication_333_111_l215_215970

theorem multiplication_333_111: 333 * 111 = 36963 := 
by 
sorry

end multiplication_333_111_l215_215970


namespace abc_less_than_one_l215_215388

variables {a b c : ℝ}

theorem abc_less_than_one (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1: a^2 < b) (h2: b^2 < c) (h3: c^2 < a) : a < 1 ∧ b < 1 ∧ c < 1 := by
  sorry

end abc_less_than_one_l215_215388


namespace find_age_l215_215759

-- Define the age variables
variables (P Q : ℕ)

-- Define the conditions
def condition1 : Prop := (P - 3) * 3 = (Q - 3) * 4
def condition2 : Prop := (P + 6) * 6 = (Q + 6) * 7

-- Prove that, given the conditions, P equals 15
theorem find_age (h1 : condition1 P Q) (h2 : condition2 P Q) : P = 15 :=
sorry

end find_age_l215_215759


namespace fraction_meaningful_iff_l215_215878

theorem fraction_meaningful_iff (m : ℝ) : 
  (∃ (x : ℝ), x = 3 / (m - 4)) ↔ m ≠ 4 :=
by 
  sorry

end fraction_meaningful_iff_l215_215878


namespace product_of_numbers_l215_215198

theorem product_of_numbers (x y : ℝ) (h1 : x - y = 1 * k) (h2 : x + y = 8 * k) (h3 : x * y = 30 * k) : 
  x * y = 400 / 7 := 
sorry

end product_of_numbers_l215_215198


namespace find_wheel_diameter_l215_215687

noncomputable def wheel_diameter (revolutions distance : ℝ) (π_approx : ℝ) : ℝ := 
  distance / (π_approx * revolutions)

theorem find_wheel_diameter : wheel_diameter 47.04276615104641 4136 3.14159 = 27.99 :=
by
  sorry

end find_wheel_diameter_l215_215687


namespace A_inter_B_A_subset_C_l215_215406

namespace MathProof

def A := {x : ℝ | x^2 - 6*x + 8 ≤ 0 }
def B := {x : ℝ | (x - 1)/(x - 3) ≥ 0 }
def C (a : ℝ) := {x : ℝ | x^2 - (2*a + 4)*x + a^2 + 4*a ≤ 0 }

theorem A_inter_B : (A ∩ B) = {x : ℝ | 3 < x ∧ x ≤ 4} := sorry

theorem A_subset_C (a : ℝ) : (A ⊆ C a) ↔ (0 ≤ a ∧ a ≤ 2) := sorry

end MathProof

end A_inter_B_A_subset_C_l215_215406


namespace elroy_more_miles_l215_215950

theorem elroy_more_miles (m_last_year : ℝ) (m_this_year : ℝ) (collect_last_year : ℝ) :
  m_last_year = 4 → m_this_year = 2.75 → collect_last_year = 44 → 
  (collect_last_year / m_this_year - collect_last_year / m_last_year = 5) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end elroy_more_miles_l215_215950


namespace decimal_equiv_half_squared_l215_215793

theorem decimal_equiv_half_squared :
  ((1 / 2 : ℝ) ^ 2) = 0.25 := by
  sorry

end decimal_equiv_half_squared_l215_215793


namespace function_decreasing_iff_l215_215474

theorem function_decreasing_iff (a : ℝ) :
  (0 < a ∧ a < 1) ∧ a ≤ 1/4 ↔ (0 < a ∧ a ≤ 1/4) :=
by
  sorry

end function_decreasing_iff_l215_215474


namespace equivalence_of_equation_and_conditions_l215_215352

open Real
open Set

-- Definitions for conditions
def condition1 (t : ℝ) : Prop := cos t ≠ 0
def condition2 (t : ℝ) : Prop := sin t ≠ 0
def condition3 (t : ℝ) : Prop := cos (2 * t) ≠ 0

-- The main statement to be proved
theorem equivalence_of_equation_and_conditions (t : ℝ) :
  ((sin t / cos t - cos t / sin t + 2 * (sin (2 * t) / cos (2 * t))) * (1 + cos (3 * t))) = 4 * sin (3 * t) ↔
  ((∃ k l : ℤ, t = (π / 5) * (2 * k + 1) ∧ k ≠ 5 * l + 2) ∨ (∃ n l : ℤ, t = (π / 3) * (2 * n + 1) ∧ n ≠ 3 * l + 1))
    ∧ condition1 t
    ∧ condition2 t
    ∧ condition3 t :=
by
  sorry

end equivalence_of_equation_and_conditions_l215_215352


namespace dave_initial_video_games_l215_215938

theorem dave_initial_video_games (non_working_games working_game_price total_earnings : ℕ) 
  (h1 : non_working_games = 2) 
  (h2 : working_game_price = 4) 
  (h3 : total_earnings = 32) : 
  non_working_games + total_earnings / working_game_price = 10 := 
by 
  sorry

end dave_initial_video_games_l215_215938


namespace fresh_fruit_sold_l215_215826

variable (total_fruit frozen_fruit : ℕ)

theorem fresh_fruit_sold (h1 : total_fruit = 9792) (h2 : frozen_fruit = 3513) : 
  total_fruit - frozen_fruit = 6279 :=
by sorry

end fresh_fruit_sold_l215_215826


namespace lines_perpendicular_and_intersect_l215_215745

variable {a b : ℝ}

theorem lines_perpendicular_and_intersect 
  (h_ab_nonzero : a * b ≠ 0)
  (h_orthogonal : a + b = 0) : 
  ∃ p, p ≠ 0 ∧ 
    (∀ x y, x = -y * b^2 → y = 0 → p = (x, y)) ∧ 
    (∀ x y, y = x / a^2 → x = 0 → p = (x, y)) ∧ 
    (∀ x y, x = -y * b^2 ∧ y = x / a^2 → x = 0 ∧ y = 0) := 
sorry

end lines_perpendicular_and_intersect_l215_215745


namespace find_valid_pairs_l215_215558

-- Definitions and conditions:
def satisfies_equation (a b : ℤ) : Prop := a^2 + a * b - b = 2018

-- Correct answers:
def valid_pairs : List (ℤ × ℤ) :=
  [(2, 2014), (0, -2018), (2018, -2018), (-2016, 2014)]

-- Statement to prove:
theorem find_valid_pairs :
  ∀ (a b : ℤ), satisfies_equation a b ↔ (a, b) ∈ valid_pairs.toFinset := by
  sorry

end find_valid_pairs_l215_215558


namespace total_money_needed_l215_215037

-- Declare John's initial amount
def john_has : ℝ := 0.75

-- Declare the additional amount John needs
def john_needs_more : ℝ := 1.75

-- The theorem statement that John needs a total of $2.50
theorem total_money_needed : john_has + john_needs_more = 2.5 :=
  by
  sorry

end total_money_needed_l215_215037


namespace sasha_sequence_eventually_five_to_100_l215_215236

theorem sasha_sequence_eventually_five_to_100 :
  ∃ (n : ℕ), 
  (5 ^ 100) = initial_value + n * (3 ^ 100) - m * (2 ^ 100) ∧ 
  (initial_value + n * (3 ^ 100) - m * (2 ^ 100) > 0) :=
by
  let initial_value := 1
  let threshold := 2 ^ 100
  let increment := 3 ^ 100
  let decrement := 2 ^ 100
  sorry

end sasha_sequence_eventually_five_to_100_l215_215236


namespace total_cost_pencils_l215_215473

theorem total_cost_pencils
  (boxes : ℕ)
  (cost_per_box : ℕ → ℕ → ℕ)
  (price_regular : ℕ)
  (price_bulk : ℕ)
  (box_size : ℕ)
  (bulk_threshold : ℕ)
  (total_pencils : ℕ) :
  total_pencils = 3150 →
  box_size = 150 →
  price_regular = 40 →
  price_bulk = 35 →
  bulk_threshold = 2000 →
  boxes = (total_pencils + box_size - 1) / box_size →
  (total_pencils > bulk_threshold → cost_per_box boxes price_bulk = boxes * price_bulk) →
  (total_pencils ≤ bulk_threshold → cost_per_box boxes price_regular = boxes * price_regular) →
  total_pencils > bulk_threshold →
  cost_per_box boxes price_bulk = 735 :=
by
  intro h_total_pencils
  intro h_box_size
  intro h_price_regular
  intro h_price_bulk
  intro h_bulk_threshold
  intro h_boxes
  intro h_cost_bulk
  intro h_cost_regular
  intro h_bulk_discount_passt
  -- sorry statement as we don't provide the actual proof here
  sorry

end total_cost_pencils_l215_215473


namespace polar_to_cartesian_l215_215231

theorem polar_to_cartesian :
  ∃ (x y : ℝ), x = 2 * Real.cos (Real.pi / 6) ∧ y = 2 * Real.sin (Real.pi / 6) ∧ 
  (x, y) = (Real.sqrt 3, 1) :=
by
  use (2 * Real.cos (Real.pi / 6)), (2 * Real.sin (Real.pi / 6))
  -- The proof will show the necessary steps
  sorry

end polar_to_cartesian_l215_215231


namespace door_X_is_inner_sanctuary_l215_215462

  variable (X Y Z W : Prop)
  variable (A B C D E F G H : Prop)
  variable (is_knight : Prop → Prop)

  -- Each statement according to the conditions in the problem.
  variable (stmt_A : X)
  variable (stmt_B : Y ∨ Z)
  variable (stmt_C : is_knight A ∧ is_knight B)
  variable (stmt_D : X ∧ Y)
  variable (stmt_E : X ∧ Y)
  variable (stmt_F : is_knight D ∨ is_knight E)
  variable (stmt_G : is_knight C → is_knight F)
  variable (stmt_H : is_knight G ∧ is_knight H → is_knight A)

  theorem door_X_is_inner_sanctuary :
    is_knight A → is_knight B → is_knight C → is_knight D → is_knight E → is_knight F → is_knight G → is_knight H → X :=
  sorry
  
end door_X_is_inner_sanctuary_l215_215462


namespace neg_universal_to_existential_l215_215855

theorem neg_universal_to_existential :
  (¬ (∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0)) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by 
  sorry

end neg_universal_to_existential_l215_215855


namespace all_sets_form_right_angled_triangle_l215_215188

theorem all_sets_form_right_angled_triangle :
    (6 * 6 + 8 * 8 = 10 * 10) ∧
    (7 * 7 + 24 * 24 = 25 * 25) ∧
    (3 * 3 + 4 * 4 = 5 * 5) ∧
    (Real.sqrt 2 * Real.sqrt 2 + Real.sqrt 3 * Real.sqrt 3 = Real.sqrt 5 * Real.sqrt 5) :=
by {
  sorry
}

end all_sets_form_right_angled_triangle_l215_215188


namespace bookstore_discount_l215_215654

theorem bookstore_discount (P MP price_paid : ℝ) (h1 : MP = 0.80 * P) (h2 : price_paid = 0.60 * MP) :
  price_paid / P = 0.48 :=
by
  sorry

end bookstore_discount_l215_215654


namespace slices_served_yesterday_l215_215755

theorem slices_served_yesterday
  (lunch_slices : ℕ)
  (dinner_slices : ℕ)
  (total_slices_today : ℕ)
  (h1 : lunch_slices = 7)
  (h2 : dinner_slices = 5)
  (h3 : total_slices_today = 12) :
  (total_slices_today - (lunch_slices + dinner_slices) = 0) :=
by {
  sorry
}

end slices_served_yesterday_l215_215755


namespace geometric_progression_theorem_l215_215656

theorem geometric_progression_theorem 
  (a b c d : ℝ) (q : ℝ) 
  (h1 : b = a * q) 
  (h2 : c = a * q^2) 
  (h3 : d = a * q^3) 
  : (a - d)^2 = (a - c)^2 + (b - c)^2 + (b - d)^2 := 
by sorry

end geometric_progression_theorem_l215_215656


namespace smallest_total_cells_marked_l215_215882

-- Definitions based on problem conditions
def grid_height : ℕ := 8
def grid_width : ℕ := 13

def squares_per_height : ℕ := grid_height / 2
def squares_per_width : ℕ := grid_width / 2

def initial_marked_cells_per_square : ℕ := 1
def additional_marked_cells_per_square : ℕ := 1

def number_of_squares : ℕ := squares_per_height * squares_per_width
def initial_marked_cells : ℕ := number_of_squares * initial_marked_cells_per_square
def additional_marked_cells : ℕ := number_of_squares * additional_marked_cells_per_square

def total_marked_cells : ℕ := initial_marked_cells + additional_marked_cells

-- Statement of the proof problem
theorem smallest_total_cells_marked : total_marked_cells = 48 := by 
    -- Proof is not required as per the instruction
    sorry

end smallest_total_cells_marked_l215_215882


namespace sqrt_meaningful_condition_l215_215249

theorem sqrt_meaningful_condition (x : ℝ) (h : 1 - x ≥ 0) : x ≤ 1 :=
by {
  -- proof steps (omitted)
  sorry
}

end sqrt_meaningful_condition_l215_215249


namespace area_of_rectangle_l215_215405

namespace RectangleArea

variable (l b : ℕ)
variable (h1 : l = 3 * b)
variable (h2 : 2 * (l + b) = 88)

theorem area_of_rectangle : l * b = 363 :=
by
  -- We will prove this in Lean 
  sorry

end RectangleArea

end area_of_rectangle_l215_215405


namespace doug_initial_marbles_l215_215959

theorem doug_initial_marbles (ed_marbles : ℕ) (diff_ed_doug : ℕ) (final_ed_marbles : ed_marbles = 27) (diff : diff_ed_doug = 5) :
  ∃ doug_initial_marbles : ℕ, doug_initial_marbles = 22 :=
by
  sorry

end doug_initial_marbles_l215_215959


namespace age_ratio_l215_215996

noncomputable def ratio_of_ages (A M : ℕ) : ℕ × ℕ :=
if A = 30 ∧ (A + 15 + (M + 15)) / 2 = 50 then
  (A / Nat.gcd A M, M / Nat.gcd A M)
else
  (0, 0)

theorem age_ratio :
  (45 + (40 + 15)) / 2 = 50 → 30 = 3 * 10 ∧ 40 = 4 * 10 →
  ratio_of_ages 30 40 = (3, 4) :=
by
  sorry

end age_ratio_l215_215996


namespace sum_diff_9114_l215_215648

def sum_odd_ints (n : ℕ) := (n + 1) / 2 * (1 + n)
def sum_even_ints (n : ℕ) := n / 2 * (2 + n)

theorem sum_diff_9114 : 
  let m := sum_odd_ints 215
  let t := sum_even_ints 100
  m - t = 9114 :=
by
  sorry

end sum_diff_9114_l215_215648


namespace intersection_point_of_lines_l215_215503

theorem intersection_point_of_lines :
  ∃ x y : ℚ, 
    (y = -3 * x + 4) ∧ 
    (y = (1 / 3) * x + 1) ∧ 
    x = 9 / 10 ∧ 
    y = 13 / 10 :=
by sorry

end intersection_point_of_lines_l215_215503


namespace find_value_of_P_l215_215428

def f (x : ℝ) : ℝ := (x^2 + x - 2)^2002 + 3

theorem find_value_of_P :
  f ( (Real.sqrt 5) / 2 - 1 / 2 ) = 4 := by
  sorry

end find_value_of_P_l215_215428


namespace circle_radius_5_l215_215421

theorem circle_radius_5 (c : ℝ) : 
  (∀ x y : ℝ, x^2 + 10 * x + y^2 + 2 * y + c = 0) → 
  (∀ x y : ℝ, (x + 5)^2 + (y + 1)^2 = 25) → 
  c = 51 :=
sorry

end circle_radius_5_l215_215421


namespace price_change_l215_215903

theorem price_change (P : ℝ) : 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  P4 = P * 0.9216 := 
by 
  let P1 := P * 1.2
  let P2 := P1 * 1.2
  let P3 := P2 * 0.8
  let P4 := P3 * 0.8
  show P4 = P * 0.9216
  sorry

end price_change_l215_215903


namespace repeating_decimal_to_fraction_l215_215838

theorem repeating_decimal_to_fraction : (0.2727272727 : ℝ) = 3 / 11 := 
sorry

end repeating_decimal_to_fraction_l215_215838


namespace inequality_solution_l215_215164

theorem inequality_solution (x : ℝ) (hx : x > 0) : (1 / x > 1) ↔ (0 < x ∧ x < 1) := 
sorry

end inequality_solution_l215_215164


namespace arnolds_total_protein_l215_215113

theorem arnolds_total_protein (collagen_protein_per_two_scoops : ℕ) (protein_per_scoop : ℕ) 
    (steak_protein : ℕ) (scoops_of_collagen : ℕ) (scoops_of_protein : ℕ) :
    collagen_protein_per_two_scoops = 18 →
    protein_per_scoop = 21 →
    steak_protein = 56 →
    scoops_of_collagen = 1 →
    scoops_of_protein = 1 →
    (collagen_protein_per_two_scoops / 2 * scoops_of_collagen + protein_per_scoop * scoops_of_protein + steak_protein = 86) :=
by
  intros hc p s sc sp
  sorry

end arnolds_total_protein_l215_215113


namespace distance_formula_proof_l215_215612

open Real

noncomputable def distance_between_points_on_curve
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  ℝ :=
  |c - a| * sqrt (1 + m^2 * (c + a)^2)

theorem distance_formula_proof
  (a b c d m k : ℝ)
  (h1 : b = m * a^2 + k)
  (h2 : d = m * c^2 + k) :
  distance_between_points_on_curve a b c d m k h1 h2 = |c - a| * sqrt (1 + m^2 * (c + a)^2) :=
by
  sorry

end distance_formula_proof_l215_215612


namespace number_of_elements_in_set_S_l215_215928

-- Define the set S and its conditions
variable (S : Set ℝ) (n : ℝ) (sumS : ℝ)

-- Conditions given in the problem
axiom avg_S : sumS / n = 6.2
axiom new_avg_S : (sumS + 8) / n = 7

-- The statement to be proved
theorem number_of_elements_in_set_S : n = 10 := by 
  sorry

end number_of_elements_in_set_S_l215_215928


namespace Tom_runs_60_miles_in_a_week_l215_215500

variable (days_per_week : ℕ) (hours_per_day : ℝ) (speed : ℝ)
variable (h_days_per_week : days_per_week = 5)
variable (h_hours_per_day : hours_per_day = 1.5)
variable (h_speed : speed = 8)

theorem Tom_runs_60_miles_in_a_week : (days_per_week * hours_per_day * speed) = 60 := by
  sorry

end Tom_runs_60_miles_in_a_week_l215_215500


namespace new_ratio_books_clothes_l215_215012

theorem new_ratio_books_clothes :
  ∀ (B C E : ℝ), (B = 22.5) → (C = 18) → (E = 9) → (C_new = C - 9) → C_new = 9 → B / C_new = 2.5 :=
by
  intros B C E HB HC HE HCnew Hnew
  sorry

end new_ratio_books_clothes_l215_215012


namespace flyers_left_to_hand_out_l215_215653

-- Definitions for given conditions
def total_flyers : Nat := 1236
def jack_handout : Nat := 120
def rose_handout : Nat := 320

-- Statement of the problem
theorem flyers_left_to_hand_out : total_flyers - (jack_handout + rose_handout) = 796 :=
by
  -- proof goes here
  sorry

end flyers_left_to_hand_out_l215_215653


namespace susan_probability_exactly_three_blue_marbles_l215_215708

open ProbabilityTheory

noncomputable def probability_blue_marbles (n_blue n_red : ℕ) (total_trials drawn_blue : ℕ) : ℚ :=
  let total_marbles := n_blue + n_red
  let prob_blue := (n_blue : ℚ) / total_marbles
  let prob_red := (n_red : ℚ) / total_marbles
  let n_comb := Nat.choose total_trials drawn_blue
  (n_comb : ℚ) * (prob_blue ^ drawn_blue) * (prob_red ^ (total_trials - drawn_blue))

theorem susan_probability_exactly_three_blue_marbles :
  probability_blue_marbles 8 7 7 3 = 35 * (1225621 / 171140625) :=
by
  sorry

end susan_probability_exactly_three_blue_marbles_l215_215708


namespace jovana_initial_shells_l215_215861

theorem jovana_initial_shells (x : ℕ) (h₁ : x + 12 = 17) : x = 5 :=
by
  -- Proof omitted
  sorry

end jovana_initial_shells_l215_215861


namespace double_acute_angle_l215_215760

theorem double_acute_angle (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
sorry

end double_acute_angle_l215_215760


namespace metallic_sheet_dimension_l215_215600

theorem metallic_sheet_dimension
  (length_cut : ℕ) (other_dim : ℕ) (volume : ℕ) (x : ℕ)
  (length_cut_eq : length_cut = 8)
  (other_dim_eq : other_dim = 36)
  (volume_eq : volume = 4800)
  (volume_formula : volume = (x - 2 * length_cut) * (other_dim - 2 * length_cut) * length_cut) :
  x = 46 :=
by
  sorry

end metallic_sheet_dimension_l215_215600


namespace walkway_time_stopped_l215_215953

noncomputable def effective_speed_with_walkway (v_p v_w : ℝ) : ℝ := v_p + v_w
noncomputable def effective_speed_against_walkway (v_p v_w : ℝ) : ℝ := v_p - v_w

theorem walkway_time_stopped (v_p v_w : ℝ) (h1 : effective_speed_with_walkway v_p v_w = 2)
                            (h2 : effective_speed_against_walkway v_p v_w = 2 / 3) :
    (200 / v_p) = 150 :=
by sorry

end walkway_time_stopped_l215_215953


namespace max_xy_l215_215362

theorem max_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 16) : 
  xy ≤ 32 :=
sorry

end max_xy_l215_215362


namespace sum_of_a_b_c_l215_215173

theorem sum_of_a_b_c (a b c : ℝ) (h1 : a * b = 24) (h2 : a * c = 36) (h3 : b * c = 54) : a + b + c = 19 :=
by
  -- The proof would go here
  sorry

end sum_of_a_b_c_l215_215173


namespace exponential_sum_sequence_l215_215218

noncomputable def Sn (n : ℕ) : ℝ :=
  Real.log (1 + 1 / n)

theorem exponential_sum_sequence : 
  e^(Sn 9 - Sn 6) = (20 : ℝ) / 21 := by
  sorry

end exponential_sum_sequence_l215_215218


namespace total_coins_is_twenty_l215_215546

def piles_of_quarters := 2
def piles_of_dimes := 3
def coins_per_pile := 4

theorem total_coins_is_twenty : piles_of_quarters * coins_per_pile + piles_of_dimes * coins_per_pile = 20 :=
by sorry

end total_coins_is_twenty_l215_215546


namespace units_digit_13_times_41_l215_215619

def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_13_times_41 :
  units_digit (13 * 41) = 3 :=
sorry

end units_digit_13_times_41_l215_215619


namespace xy_cubed_identity_l215_215967

namespace ProofProblem

theorem xy_cubed_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 :=
by
  sorry

end ProofProblem

end xy_cubed_identity_l215_215967


namespace inequality_solution_l215_215554

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem inequality_solution (a b : ℝ) 
  (h1 : ∀ (x : ℝ), f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ (x : ℝ), f a b (-2 * x) < 0 ↔ x < -3 / 2 ∨ x > 1 / 2 :=
sorry

end inequality_solution_l215_215554


namespace share_pizza_l215_215457

variable (Yoojung_slices Minyoung_slices total_slices : ℕ)
variable (Y : ℕ)

theorem share_pizza :
  Yoojung_slices = Y ∧
  Minyoung_slices = Y + 2 ∧
  total_slices = 10 ∧
  Yoojung_slices + Minyoung_slices = total_slices →
  Y = 4 :=
by
  sorry

end share_pizza_l215_215457


namespace number_equation_l215_215013

variable (x : ℝ)

theorem number_equation :
  5 * x - 2 * x = 10 :=
sorry

end number_equation_l215_215013


namespace triangle_inequality_violation_l215_215952

theorem triangle_inequality_violation (a b c : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 7) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := 
by
  rw [ha, hb, hc]
  simp
  sorry

end triangle_inequality_violation_l215_215952


namespace hyperbola_standard_equation_l215_215784

open Real

noncomputable def distance_from_center_to_focus (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem hyperbola_standard_equation (a b c : ℝ)
  (h1 : a > b) (h2 : b > 0)
  (h3 : b = sqrt 3 * c)
  (h4 : a + c = 3 * sqrt 3) :
  ∃ h : a^2 = 12 ∧ b = 3, y^2 / 12 - x^2 / 9 = 1 :=
sorry

end hyperbola_standard_equation_l215_215784


namespace roger_final_money_is_correct_l215_215758

noncomputable def initial_money : ℝ := 84
noncomputable def birthday_money : ℝ := 56
noncomputable def found_money : ℝ := 20
noncomputable def spent_on_game : ℝ := 35
noncomputable def spent_percentage : ℝ := 0.15

noncomputable def final_money 
  (initial_money birthday_money found_money spent_on_game spent_percentage : ℝ) : ℝ :=
  let total_before_spending := initial_money + birthday_money + found_money
  let remaining_after_game := total_before_spending - spent_on_game
  let spent_on_gift := spent_percentage * remaining_after_game
  remaining_after_game - spent_on_gift

theorem roger_final_money_is_correct :
  final_money initial_money birthday_money found_money spent_on_game spent_percentage = 106.25 :=
by
  sorry

end roger_final_money_is_correct_l215_215758


namespace find_x_l215_215937

theorem find_x (x : ℚ) (h : ∀ (y : ℚ), 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0) : x = 3 / 2 :=
sorry

end find_x_l215_215937


namespace sum_of_series_is_918_l215_215329

-- Define the first term a, common difference d, last term a_n,
-- and the number of terms n calculated from the conditions.
def first_term : Int := -300
def common_difference : Int := 3
def last_term : Int := 309
def num_terms : Int := 204 -- calculated as per the solution

-- Compute the sum of the arithmetic series
def sum_arithmetic_series (a d : Int) (n : Int) : Int :=
  n * (2 * a + (n - 1) * d) / 2

-- Prove that the sum of the series is 918
theorem sum_of_series_is_918 :
  sum_arithmetic_series first_term common_difference num_terms = 918 :=
by
  sorry

end sum_of_series_is_918_l215_215329


namespace wheel_speed_is_12_mph_l215_215464

theorem wheel_speed_is_12_mph
  (r : ℝ) -- speed in miles per hour
  (C : ℝ := 15 / 5280) -- circumference in miles
  (H1 : ∃ t, r * t = C * 3600) -- initial condition that speed times time for one rotation equals 15/5280 miles in seconds
  (H2 : ∃ t, (r + 7) * (t - 1/21600) = C * 3600) -- condition that speed increases by 7 mph when time shortens by 1/6 second
  : r = 12 :=
sorry

end wheel_speed_is_12_mph_l215_215464


namespace choose_starting_team_l215_215330

-- Definitions derived from the conditions
def team_size : ℕ := 18
def selected_goalie (n : ℕ) : ℕ := n
def selected_players (m : ℕ) (k : ℕ) : ℕ := Nat.choose m k

-- The number of ways to choose the starting team
theorem choose_starting_team :
  let n := team_size
  let k := 7
  selected_goalie n * selected_players (n - 1) k = 222768 :=
by
  simp only [team_size, selected_goalie, selected_players]
  sorry

end choose_starting_team_l215_215330


namespace tank_filled_to_depth_l215_215402

noncomputable def tank_volume (R H r d : ℝ) : ℝ := R^2 * H * Real.pi - (r^2 * H * Real.pi)

theorem tank_filled_to_depth (R H r d : ℝ) (h_cond : R = 5 ∧ H = 12 ∧ r = 2 ∧ d = 3) :
  tank_volume R H r d = 110 * Real.pi - 96 :=
sorry

end tank_filled_to_depth_l215_215402


namespace equal_real_roots_of_quadratic_eq_l215_215263

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end equal_real_roots_of_quadratic_eq_l215_215263


namespace find_g_at_1_l215_215859

noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem find_g_at_1 : 
  (∀ x : ℝ, g (2*x + 3) = x^2 - 2*x + 4) → 
  g 1 = 7 := 
by
  intro h
  -- Proof goes here
  sorry

end find_g_at_1_l215_215859


namespace find_number_l215_215661

theorem find_number : ∃ x : ℝ, (x / 5 + 7 = x / 4 - 7) ∧ x = 280 :=
by
  -- Here, we state the existence of a real number x
  -- such that the given condition holds and x = 280.
  sorry

end find_number_l215_215661


namespace find_hourly_wage_l215_215401

noncomputable def hourly_wage_inexperienced (x : ℝ) : Prop :=
  let sailors_total := 17
  let inexperienced_sailors := 5
  let experienced_sailors := sailors_total - inexperienced_sailors
  let wage_experienced := (6 / 5) * x
  let total_hours_month := 240
  let total_monthly_earnings_experienced := 34560
  (experienced_sailors * wage_experienced * total_hours_month) = total_monthly_earnings_experienced

theorem find_hourly_wage (x : ℝ) : hourly_wage_inexperienced x → x = 10 :=
by
  sorry

end find_hourly_wage_l215_215401


namespace radius_increase_of_pizza_l215_215280

/-- 
Prove that the percent increase in radius from a medium pizza to a large pizza is 20% 
given the following conditions:
1. The radius of the large pizza is some percent larger than that of a medium pizza.
2. The percent increase in area between a medium and a large pizza is approximately 44%.
3. The area of a circle is given by the formula A = π * r^2.
--/
theorem radius_increase_of_pizza
  (r R : ℝ) -- r and R are the radii of the medium and large pizza respectively
  (h1 : R = (1 + k) * r) -- The radius of the large pizza is some percent larger than that of a medium pizza
  (h2 : π * R^2 = 1.44 * π * r^2) -- The percent increase in area between a medium and a large pizza is approximately 44%
  : k = 0.2 := 
sorry

end radius_increase_of_pizza_l215_215280


namespace range_of_a_plus_2014b_l215_215525

theorem range_of_a_plus_2014b (a b : ℝ) (h1 : a < b) (h2 : |(Real.log a) / (Real.log 2)| = |(Real.log b) / (Real.log 2)|) :
  ∃ c : ℝ, c > 2015 ∧ ∀ x : ℝ, a + 2014 * b = x → x > 2015 := by
  sorry

end range_of_a_plus_2014b_l215_215525


namespace triangle_area_correct_l215_215397

def line1 (x : ℝ) : ℝ := 8
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define the intersection points
def intersection1 : ℝ × ℝ := (6, line1 6)
def intersection2 : ℝ × ℝ := (-6, line1 (-6))
def intersection3 : ℝ × ℝ := (0, line2 0)

def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * |A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)|

theorem triangle_area_correct :
  triangle_area intersection1 intersection2 intersection3 = 36 :=
by
  sorry

end triangle_area_correct_l215_215397


namespace incorrect_divisor_l215_215340

theorem incorrect_divisor (D x : ℕ) (h1 : D = 24 * x) (h2 : D = 48 * 36) : x = 72 := by
  sorry

end incorrect_divisor_l215_215340


namespace complex_power_difference_l215_215204

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i) ^ 16 - (1 - i) ^ 16 = 0 := by
  sorry

end complex_power_difference_l215_215204


namespace b_payment_l215_215630

theorem b_payment (b_days : ℕ) (a_days : ℕ) (total_wages : ℕ) (b_payment : ℕ) :
  b_days = 10 →
  a_days = 15 →
  total_wages = 5000 →
  b_payment = 3000 :=
by
  intros h1 h2 h3
  -- conditions
  have hb := h1
  have ha := h2
  have ht := h3
  -- skipping proof
  sorry

end b_payment_l215_215630


namespace yearly_return_of_1500_investment_l215_215426

theorem yearly_return_of_1500_investment 
  (combined_return_percent : ℝ)
  (total_investment : ℕ)
  (return_500 : ℕ)
  (investment_500 : ℕ)
  (investment_1500 : ℕ) :
  combined_return_percent = 0.085 →
  total_investment = (investment_500 + investment_1500) →
  return_500 = (investment_500 * 7 / 100) →
  investment_500 = 500 →
  investment_1500 = 1500 →
  total_investment = 2000 →
  (return_500 + investment_1500 * combined_return_percent * 100) = (combined_return_percent * total_investment * 100) →
  ((investment_1500 * (9 : ℝ)) / 100) + return_500 = 0.085 * total_investment →
  (investment_1500 * 7 / 100) = investment_1500 →
  (investment_1500 / investment_1500) = (13500 / 1500) →
  (9 : ℝ) = 9 :=
sorry

end yearly_return_of_1500_investment_l215_215426


namespace esperanza_gross_salary_l215_215308

def rent : ℕ := 600
def food_expenses (rent : ℕ) : ℕ := 3 * rent / 5
def mortgage_bill (food_expenses : ℕ) : ℕ := 3 * food_expenses
def savings : ℕ := 2000
def taxes (savings : ℕ) : ℕ := 2 * savings / 5
def total_expenses (rent food_expenses mortgage_bill taxes : ℕ) : ℕ :=
  rent + food_expenses + mortgage_bill + taxes
def gross_salary (total_expenses savings : ℕ) : ℕ :=
  total_expenses + savings

theorem esperanza_gross_salary : 
  gross_salary (total_expenses rent (food_expenses rent) (mortgage_bill (food_expenses rent)) (taxes savings)) savings = 4840 :=
by
  sorry

end esperanza_gross_salary_l215_215308


namespace negative_number_reciprocal_eq_self_l215_215747

theorem negative_number_reciprocal_eq_self (x : ℝ) (hx : x < 0) (h : 1 / x = x) : x = -1 :=
by
  sorry

end negative_number_reciprocal_eq_self_l215_215747


namespace isosceles_triangle_largest_angle_l215_215724

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_iso : A = B) (h_A : C = 50) :
  max A (max B (180 - A - B)) = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l215_215724


namespace probability_of_three_different_colors_draw_l215_215354

open ProbabilityTheory

def number_of_blue_chips : ℕ := 4
def number_of_green_chips : ℕ := 5
def number_of_red_chips : ℕ := 6
def number_of_yellow_chips : ℕ := 3
def total_number_of_chips : ℕ := 18

def P_B : ℚ := number_of_blue_chips / total_number_of_chips
def P_G : ℚ := number_of_green_chips / total_number_of_chips
def P_R : ℚ := number_of_red_chips / total_number_of_chips
def P_Y : ℚ := number_of_yellow_chips / total_number_of_chips

def P_different_colors : ℚ := 2 * ((P_B * P_G + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G + P_R * P_Y) +
                                    (P_B * P_R + P_B * P_Y + P_G * P_R + P_G * P_Y + P_R * P_B + P_R * P_G + P_Y * P_B + P_Y * P_G))

theorem probability_of_three_different_colors_draw :
  P_different_colors = 141 / 162 :=
by
  -- Placeholder for the actual proof.
  sorry

end probability_of_three_different_colors_draw_l215_215354


namespace least_pos_int_solution_l215_215191

theorem least_pos_int_solution (x : ℤ) : x + 4609 ≡ 2104 [ZMOD 12] → x = 3 := by
  sorry

end least_pos_int_solution_l215_215191


namespace common_tangent_slope_l215_215394

theorem common_tangent_slope (a m : ℝ) : 
  ((∃ a, ∃ m, l = (2 * a) ∧ l = (3 * m^2) ∧ a^2 = 2 * m^3) → (l = 0 ∨ l = 64 / 27)) := 
sorry

end common_tangent_slope_l215_215394


namespace clubsuit_subtraction_l215_215440

def clubsuit (x y : ℕ) := 4 * x + 6 * y

theorem clubsuit_subtraction :
  (clubsuit 5 3) - (clubsuit 1 4) = 10 :=
by
  sorry

end clubsuit_subtraction_l215_215440


namespace total_school_population_220_l215_215217

theorem total_school_population_220 (x B : ℕ) 
  (h1 : 242 = (x * B) / 100) 
  (h2 : B = (50 * x) / 100) : x = 220 := by
  sorry

end total_school_population_220_l215_215217


namespace solution_set_of_inequality_l215_215822

theorem solution_set_of_inequality (a : ℝ) :
  ¬ (∀ x : ℝ, ¬ (a * (x - a) * (a * x + a) ≥ 0)) ∧
  ¬ (∀ x : ℝ, (a - x ≤ 0 ∧ x - (-1) ≤ 0 → a * (x - a) * (a * x + a) ≥ 0)) :=
by
  sorry

end solution_set_of_inequality_l215_215822


namespace sequence_a2018_l215_215956

theorem sequence_a2018 (a : ℕ → ℝ) 
  (h1 : ∀ n, a (n + 2) - 2 * a (n + 1) + a n = 1) 
  (h2 : a 18 = 0) 
  (h3 : a 2017 = 0) :
  a 2018 = 1000 :=
sorry

end sequence_a2018_l215_215956


namespace tan_phi_l215_215808

theorem tan_phi (φ : ℝ) (h1 : Real.cos (π / 2 + φ) = 2 / 3) (h2 : abs φ < π / 2) : 
  Real.tan φ = -2 * Real.sqrt 5 / 5 := 
by 
  sorry

end tan_phi_l215_215808


namespace min_PM_PN_min_PM_squared_PN_squared_l215_215368

noncomputable def min_value_PM_PN := 3 * Real.sqrt 5

noncomputable def min_value_PM_squared_PN_squared := 229 / 10

structure Point :=
  (x : ℝ)
  (y : ℝ)

def M : Point := ⟨2, 5⟩
def N : Point := ⟨-2, 4⟩

def on_line (P : Point) : Prop :=
  P.x - 2 * P.y + 3 = 0

theorem min_PM_PN {P : Point} (h : on_line P) :
  dist (P.x, P.y) (M.x, M.y) + dist (P.x, P.y) (N.x, N.y) = min_value_PM_PN := sorry

theorem min_PM_squared_PN_squared {P : Point} (h : on_line P) :
  (dist (P.x, P.y) (M.x, M.y))^2 + (dist (P.x, P.y) (N.x, N.y))^2 = min_value_PM_squared_PN_squared := sorry

end min_PM_PN_min_PM_squared_PN_squared_l215_215368


namespace students_wearing_other_colors_l215_215360

variable (total_students blue_percentage red_percentage green_percentage : ℕ)
variable (h_total : total_students = 600)
variable (h_blue : blue_percentage = 45)
variable (h_red : red_percentage = 23)
variable (h_green : green_percentage = 15)

theorem students_wearing_other_colors :
  (total_students * (100 - (blue_percentage + red_percentage + green_percentage)) / 100 = 102) :=
by
  sorry

end students_wearing_other_colors_l215_215360


namespace radius_of_circle_l215_215409

variable (r M N : ℝ)

theorem radius_of_circle (h1 : M = Real.pi * r^2) 
  (h2 : N = 2 * Real.pi * r) 
  (h3 : M / N = 15) : 
  r = 30 :=
sorry

end radius_of_circle_l215_215409


namespace minimum_number_of_apples_l215_215587

-- Define the problem conditions and the proof statement
theorem minimum_number_of_apples :
  ∃ p : Fin 6 → ℕ, (∀ i, p i > 0) ∧ (Function.Injective p) ∧ (Finset.univ.sum p * 4 = 100) ∧ (Finset.univ.sum p = 25 / 4) := 
sorry

end minimum_number_of_apples_l215_215587


namespace largest_angle_of_consecutive_integers_hexagon_l215_215088

theorem largest_angle_of_consecutive_integers_hexagon (a b c d e f : ℝ) 
  (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e) (h5 : e < f) 
  (h6 : a + b + c + d + e + f = 720) : 
  ∃ x, f = x + 2 ∧ (x + 2 = 122.5) :=
  sorry

end largest_angle_of_consecutive_integers_hexagon_l215_215088


namespace area_between_circles_of_octagon_l215_215454

-- Define some necessary geometric terms and functions
noncomputable def cot (θ : ℝ) : ℝ := 1 / Real.tan θ
noncomputable def csc (θ : ℝ) : ℝ := 1 / Real.sin θ

/-- The main theorem stating the area between the inscribed and circumscribed circles of a regular octagon is π. -/
theorem area_between_circles_of_octagon :
  let side_length := 2
  let θ := Real.pi / 8 -- 22.5 degrees in radians
  let apothem := cot θ
  let circum_radius := csc θ
  let area_between_circles := π * (circum_radius^2 - apothem^2)
  area_between_circles = π :=
by
  sorry

end area_between_circles_of_octagon_l215_215454


namespace probability_same_color_shoes_l215_215076

theorem probability_same_color_shoes (pairs : ℕ) (total_shoes : ℕ)
  (each_pair_diff_color : pairs * 2 = total_shoes)
  (select_2_without_replacement : total_shoes = 10 ∧ pairs = 5) :
  let successful_outcomes := pairs
  let total_outcomes := (total_shoes * (total_shoes - 1)) / 2
  successful_outcomes / total_outcomes = 1 / 9 :=
by
  sorry

end probability_same_color_shoes_l215_215076


namespace mathematics_equivalent_proof_l215_215282

noncomputable def distinctRealNumbers (a b c d : ℝ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a ≠ d ∧ b ≠ d ∧ c ≠ d

theorem mathematics_equivalent_proof (a b c d : ℝ)
  (H₀ : distinctRealNumbers a b c d)
  (H₁ : (a - d) / (b - c) + (b - d) / (c - a) + (c - d) / (a - b) = 0) :
  (a + d) / (b - c)^3 + (b + d) / (c - a)^3 + (c + d) / (a - b)^3 = 0 :=
sorry

end mathematics_equivalent_proof_l215_215282


namespace points_lie_on_parabola_l215_215658

theorem points_lie_on_parabola (u : ℝ) :
  ∃ (x y : ℝ), x = 3^u - 4 ∧ y = 9^u - 7 * 3^u - 2 ∧ y = x^2 + x - 14 :=
by
  sorry

end points_lie_on_parabola_l215_215658


namespace savings_account_amount_l215_215791

noncomputable def final_amount : ℝ :=
  let initial_deposit : ℝ := 5000
  let first_quarter_rate : ℝ := 0.01
  let second_quarter_rate : ℝ := 0.0125
  let deposit_end_third_month : ℝ := 1000
  let withdrawal_end_fifth_month : ℝ := 500
  let amount_after_first_quarter := initial_deposit * (1 + first_quarter_rate)
  let amount_before_second_quarter := amount_after_first_quarter + deposit_end_third_month
  let amount_after_second_quarter := amount_before_second_quarter * (1 + second_quarter_rate)
  let final_amount := amount_after_second_quarter - withdrawal_end_fifth_month
  final_amount

theorem savings_account_amount :
  final_amount = 5625.625 :=
by
  sorry

end savings_account_amount_l215_215791


namespace three_2x2_squares_exceed_100_l215_215514

open BigOperators

noncomputable def sum_of_1_to_64 : ℕ :=
  (64 * (64 + 1)) / 2

theorem three_2x2_squares_exceed_100 :
  ∀ (s : Fin 16 → ℕ),
    (∑ i, s i = sum_of_1_to_64) →
    (∀ i j, i ≠ j → s i = s j ∨ s i > s j ∨ s i < s j) →
    (∃ i₁ i₂ i₃, i₁ ≠ i₂ ∧ i₂ ≠ i₃ ∧ i₁ ≠ i₃ ∧ s i₁ > 100 ∧ s i₂ > 100 ∧ s i₃ > 100) := sorry

end three_2x2_squares_exceed_100_l215_215514


namespace min_employees_to_hire_l215_215769

-- Definitions of the given conditions
def employees_cust_service : ℕ := 95
def employees_tech_support : ℕ := 80
def employees_both : ℕ := 30

-- The theorem stating the minimum number of new employees to hire
theorem min_employees_to_hire (n : ℕ) :
  n = (employees_cust_service - employees_both) 
    + (employees_tech_support - employees_both) 
    + employees_both → 
  n = 145 := sorry

end min_employees_to_hire_l215_215769


namespace fish_to_rice_equivalence_l215_215242

variable (f : ℚ) (l : ℚ)

theorem fish_to_rice_equivalence (h1 : 5 * f = 3 * l) (h2 : l = 6) : f = 18 / 5 := by
  sorry

end fish_to_rice_equivalence_l215_215242


namespace local_extrema_l215_215452

noncomputable def f (x : ℝ) := 3 * x^3 - 9 * x^2 + 3

theorem local_extrema :
  (∃ x, x = 0 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≤ f x) ∧
  (∃ x, x = 2 ∧ ∀ δ > 0, ∃ ε > 0, ∀ y, abs (y - x) < ε → f y ≥ f x) :=
sorry

end local_extrema_l215_215452


namespace value_to_subtract_l215_215011

variable (x y : ℝ)

theorem value_to_subtract (h1 : (x - 5) / 7 = 7) (h2 : (x - y) / 8 = 6) : y = 6 := by
  sorry

end value_to_subtract_l215_215011


namespace simplify_expression_l215_215437

theorem simplify_expression :
  (1 / (Real.log 3 / Real.log 12 + 1) + 1 / (Real.log 2 / Real.log 8 + 1) + 1 / (Real.log 9 / Real.log 18 + 1)) = 7 / 4 := 
sorry

end simplify_expression_l215_215437


namespace find_X_l215_215045

variable {α : Type} -- considering sets of some type α
variables (A B X : Set α)

theorem find_X (h1 : A ∩ X = B ∩ X ∧ B ∩ X = A ∩ B)
               (h2 : A ∪ B ∪ X = A ∪ B) : X = A ∩ B :=
by {
    sorry
}

end find_X_l215_215045


namespace domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l215_215907

-- For the function y = sqrt(3 + 2x)
theorem domain_sqrt_3_plus_2x (x : ℝ) : 3 + 2 * x ≥ 0 -> x ∈ Set.Ici (-3 / 2) :=
by
  sorry

-- For the function f(x) = 1 + sqrt(9 - x^2)
theorem domain_1_plus_sqrt_9_minus_x2 (x : ℝ) : 9 - x^2 ≥ 0 -> x ∈ Set.Icc (-3) 3 :=
by
  sorry

-- For the function φ(x) = sqrt(log((5x - x^2) / 4))
theorem domain_sqrt_log_5x_minus_x2_over_4 (x : ℝ) : (5 * x - x^2) / 4 > 0 ∧ (5 * x - x^2) / 4 ≥ 1 -> x ∈ Set.Icc 1 4 :=
by
  sorry

-- For the function y = sqrt(3 - x) + arccos((x - 2) / 3)
theorem domain_sqrt_3_minus_x_plus_arccos (x : ℝ) : 3 - x ≥ 0 ∧ -1 ≤ (x - 2) / 3 ∧ (x - 2) / 3 ≤ 1 -> x ∈ Set.Icc (-1) 3 :=
by
  sorry

end domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l215_215907


namespace two_students_follow_all_celebrities_l215_215472

theorem two_students_follow_all_celebrities :
  ∀ (students : Finset ℕ) (celebrities_followers : ℕ → Finset ℕ),
    (students.card = 120) →
    (∀ c : ℕ, c < 10 → (celebrities_followers c).card ≥ 85 ∧ (celebrities_followers c) ⊆ students) →
    ∃ (s1 s2 : ℕ), s1 ∈ students ∧ s2 ∈ students ∧ s1 ≠ s2 ∧
      (∀ c : ℕ, c < 10 → (s1 ∈ celebrities_followers c ∨ s2 ∈ celebrities_followers c)) :=
by
  intros students celebrities_followers h_students_card h_followers_cond
  sorry

end two_students_follow_all_celebrities_l215_215472


namespace pan_dimensions_l215_215031

theorem pan_dimensions (m n : ℕ) : 
  (∃ m n, m * n = 48 ∧ (m-2) * (n-2) = 2 * (2*m + 2*n - 4) ∧ m > 2 ∧ n > 2) → 
  (m = 4 ∧ n = 12) ∨ (m = 12 ∧ n = 4) ∨ (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
by
  sorry

end pan_dimensions_l215_215031


namespace abs_sum_less_abs_diff_l215_215302

theorem abs_sum_less_abs_diff {a b : ℝ} (hab : a * b < 0) : |a + b| < |a - b| :=
sorry

end abs_sum_less_abs_diff_l215_215302


namespace solve_x_from_operation_l215_215961

def operation (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_x_from_operation :
  ∀ x : ℝ, operation (2 * x) 3 3 (-1) = 3 → x = 1 :=
by
  intros x h
  sorry

end solve_x_from_operation_l215_215961


namespace decimal_expansion_2023rd_digit_l215_215377

theorem decimal_expansion_2023rd_digit 
  (x : ℚ) 
  (hx : x = 7 / 26) 
  (decimal_expansion : ℕ → ℕ)
  (hdecimal : ∀ n : ℕ, decimal_expansion n = if n % 12 = 0 
                        then 2 
                        else if n % 12 = 1 
                          then 7 
                          else if n % 12 = 2 
                            then 9 
                            else if n % 12 = 3 
                              then 2 
                              else if n % 12 = 4 
                                then 3 
                                else if n % 12 = 5 
                                  then 0 
                                  else if n % 12 = 6 
                                    then 7 
                                    else if n % 12 = 7 
                                      then 6 
                                      else if n % 12 = 8 
                                        then 9 
                                        else if n % 12 = 9 
                                          then 2 
                                          else if n % 12 = 10 
                                            then 3 
                                            else 0) :
  decimal_expansion 2023 = 0 :=
sorry

end decimal_expansion_2023rd_digit_l215_215377


namespace length_dg_l215_215305

theorem length_dg (a b k l S : ℕ) (h1 : S = 47 * (a + b)) 
                   (h2 : S = a * k) (h3 : S = b * l) (h4 : b = S / l) 
                   (h5 : a = S / k) (h6 : k * l = 47 * k + 47 * l + 2209) : 
  k = 2256 :=
by sorry

end length_dg_l215_215305


namespace find_number_l215_215408

theorem find_number (n : ℝ) (h : n - (1004 / 20.08) = 4970) : n = 5020 := 
by {
  sorry
}

end find_number_l215_215408


namespace state_A_selection_percentage_l215_215595

theorem state_A_selection_percentage
  (candidates_A : ℕ)
  (candidates_B : ℕ)
  (x : ℕ)
  (selected_B_ratio : ℚ)
  (extra_B : ℕ)
  (h1 : candidates_A = 7900)
  (h2 : candidates_B = 7900)
  (h3 : selected_B_ratio = 0.07)
  (h4 : extra_B = 79)
  (h5 : 7900 * (7 / 100) + 79 = 7900 * (x / 100) + 79) :
  x = 7 := by
  sorry

end state_A_selection_percentage_l215_215595


namespace probability_of_first_three_red_cards_l215_215562

theorem probability_of_first_three_red_cards :
  let total_cards := 60
  let red_cards := 36
  let black_cards := total_cards - red_cards
  let total_ways := total_cards * (total_cards - 1) * (total_cards - 2)
  let red_ways := red_cards * (red_cards - 1) * (red_cards - 2)
  (red_ways / total_ways) = 140 / 673 :=
by
  sorry

end probability_of_first_three_red_cards_l215_215562


namespace trapezoid_area_l215_215098

theorem trapezoid_area (AD BC : ℝ) (AD_eq : AD = 18) (BC_eq : BC = 2) (CD : ℝ) (h : CD = 10): 
  ∃ (CH : ℝ), CH = 6 ∧ (1 / 2) * (AD + BC) * CH = 60 :=
by
  sorry

end trapezoid_area_l215_215098


namespace problem_l215_215805

variables (y S : ℝ)

theorem problem (h : 5 * (2 * y + 3 * Real.sqrt 3) = S) : 10 * (4 * y + 6 * Real.sqrt 3) = 4 * S :=
sorry

end problem_l215_215805


namespace like_terms_sum_l215_215480

theorem like_terms_sum (m n : ℕ) (h1 : 2 * m = 2) (h2 : n = 3) : m + n = 4 :=
sorry

end like_terms_sum_l215_215480


namespace find_width_l215_215869

-- Definitions and Conditions
def length : ℝ := 6
def depth : ℝ := 2
def total_surface_area : ℝ := 104

-- Statement to prove the width
theorem find_width (width : ℝ) (h : 12 * width + 4 * width + 24 = total_surface_area) : width = 5 := 
by { 
  -- lean 4 statement only, proof omitted
  sorry 
}

end find_width_l215_215869


namespace train_length_is_correct_l215_215427

noncomputable def length_of_train 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * time_to_cross
  distance_covered - bridge_length

theorem train_length_is_correct :
  length_of_train 23.998080153587715 140 36 = 99.98080153587715 :=
by sorry

end train_length_is_correct_l215_215427


namespace commute_solution_l215_215674

noncomputable def commute_problem : Prop :=
  let t : ℝ := 1                -- 1 hour from 7:00 AM to 8:00 AM
  let late_minutes : ℝ := 5 / 60  -- 5 minutes = 5/60 hours
  let early_minutes : ℝ := 4 / 60 -- 4 minutes = 4/60 hours
  let speed1 : ℝ := 30          -- 30 mph
  let speed2 : ℝ := 70          -- 70 mph
  let d1 : ℝ := speed1 * (t + late_minutes)
  let d2 : ℝ := speed2 * (t - early_minutes)

  ∃ (speed : ℝ), d1 = d2 ∧ speed = d1 / t ∧ speed = 32.5

theorem commute_solution : commute_problem :=
by sorry

end commute_solution_l215_215674


namespace max_quotient_l215_215090

theorem max_quotient (a b : ℝ) (h1 : 300 ≤ a) (h2 : a ≤ 500) (h3 : 900 ≤ b) (h4 : b ≤ 1800) :
  ∃ (q : ℝ), q = 5 / 9 ∧ (∀ (x y : ℝ), (300 ≤ x ∧ x ≤ 500) ∧ (900 ≤ y ∧ y ≤ 1800) → (x / y ≤ q)) :=
by
  use 5 / 9
  sorry

end max_quotient_l215_215090


namespace computation_problem_points_l215_215847

def num_problems : ℕ := 30
def points_per_word_problem : ℕ := 5
def total_points : ℕ := 110
def num_computation_problems : ℕ := 20

def points_per_computation_problem : ℕ := 3

theorem computation_problem_points :
  ∃ x : ℕ, (num_computation_problems * x + (num_problems - num_computation_problems) * points_per_word_problem = total_points) ∧ x = points_per_computation_problem :=
by
  use points_per_computation_problem
  simp
  sorry

end computation_problem_points_l215_215847


namespace inclination_angle_l215_215315

theorem inclination_angle (θ : ℝ) : 
  (∃ (x y : ℝ), x + y - 3 = 0) → θ = 3 * Real.pi / 4 := 
sorry

end inclination_angle_l215_215315


namespace radius_of_cylinder_is_correct_l215_215577

/-- 
  A right circular cylinder is inscribed in a right circular cone such that:
  - The diameter of the cylinder is equal to its height.
  - The cone has a diameter of 8.
  - The cone has an altitude of 10.
  - The axes of the cylinder and cone coincide.
  Prove that the radius of the cylinder is 20/9.
-/
theorem radius_of_cylinder_is_correct :
  ∀ (r : ℚ), 
    (2 * r = 8 - 2 * r ∧ 10 - 2 * r = (10 / 4) * r) → 
    r = 20 / 9 :=
by
  intro r
  intro h
  sorry

end radius_of_cylinder_is_correct_l215_215577


namespace no_integer_solutions_l215_215932

theorem no_integer_solutions (P Q : Polynomial ℤ) (a : ℤ) (hP1 : P.eval a = 0) 
  (hP2 : P.eval (a + 1997) = 0) (hQ : Q.eval 1998 = 2000) : 
  ¬ ∃ x : ℤ, Q.eval (P.eval x) = 1 := 
by
  sorry

end no_integer_solutions_l215_215932


namespace triangle_centroid_eq_l215_215195

-- Define the proof problem
theorem triangle_centroid_eq
  (P Q R G : ℝ × ℝ) -- Points P, Q, R, and G (the centroid of the triangle PQR)
  (centroid_eq : G = ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)) -- Condition that G is the centroid
  (gp_sq_gq_sq_gr_sq_eq : dist G P ^ 2 + dist G Q ^ 2 + dist G R ^ 2 = 22) -- Given GP^2 + GQ^2 + GR^2 = 22
  : dist P Q ^ 2 + dist P R ^ 2 + dist Q R ^ 2 = 66 := -- Prove PQ^2 + PR^2 + QR^2 = 66
sorry -- Proof is omitted

end triangle_centroid_eq_l215_215195


namespace airplane_seats_l215_215285

theorem airplane_seats (F : ℕ) (h : F + 4 * F + 2 = 387) : F = 77 := by
  -- Proof goes here
  sorry

end airplane_seats_l215_215285


namespace evaluate_expression_l215_215583

theorem evaluate_expression : (900^2 / (153^2 - 147^2)) = 450 := by
  sorry

end evaluate_expression_l215_215583


namespace initial_bleach_percentage_l215_215052

-- Define variables and constants
def total_volume : ℝ := 100
def drained_volume : ℝ := 3.0612244898
def desired_percentage : ℝ := 0.05

-- Define the initial percentage (unknown)
variable (P : ℝ)

-- Define the statement to be proved
theorem initial_bleach_percentage :
  ( (total_volume - drained_volume) * P + drained_volume * 1 = total_volume * desired_percentage )
  → P = 0.02 :=
  by
    intro h
    -- skipping the proof as per instructions
    sorry

end initial_bleach_percentage_l215_215052


namespace expected_value_of_expression_is_50_l215_215631

def expected_value_single_digit : ℚ := (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) / 9

def expected_value_expression : ℚ :=
  (expected_value_single_digit + expected_value_single_digit + expected_value_single_digit +
   (expected_value_single_digit + expected_value_single_digit * expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit + expected_value_single_digit) +
   (expected_value_single_digit * expected_value_single_digit * expected_value_single_digit)) / 4

theorem expected_value_of_expression_is_50 :
  expected_value_expression = 50 := sorry

end expected_value_of_expression_is_50_l215_215631


namespace least_x_value_l215_215309

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem least_x_value (x : ℕ) (p : ℕ) (hp : is_prime p) (h : x / (12 * p) = 2) : x = 48 := by
  sorry

end least_x_value_l215_215309


namespace valves_fill_pool_l215_215655

theorem valves_fill_pool
  (a b c d : ℝ)
  (h1 : 1 / a + 1 / b + 1 / c = 1 / 12)
  (h2 : 1 / b + 1 / c + 1 / d = 1 / 15)
  (h3 : 1 / a + 1 / d = 1 / 20) :
  1 / a + 1 / b + 1 / c + 1 / d = 1 / 10 := 
sorry

end valves_fill_pool_l215_215655


namespace max_point_f_l215_215255

-- Define the function f(x) = x^3 - 12x
def f (x : ℝ) : ℝ := x^3 - 12 * x

-- Maximum point of the function f is -2
theorem max_point_f : ∃ m, m = -2 ∧ ∀ x, f x ≤ f (-2) :=
by
  sorry

end max_point_f_l215_215255


namespace move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l215_215085

-- Define the initial conditions
def pointA := (50 : ℝ)
def radius := (1 : ℝ)
def origin := (0 : ℝ)

-- Statement for part (a)
theorem move_point_inside_with_25_reflections :
  ∃ (n : ℕ) (r : ℝ), n = 25 ∧ r = radius + 50 ∧ pointA ≤ r :=
by
  sorry

-- Statement for part (b)
theorem cannot_move_point_inside_with_24_reflections :
  ∀ (n : ℕ) (r : ℝ), n = 24 → r = radius + 48 → pointA > r :=
by
  sorry

end move_point_inside_with_25_reflections_cannot_move_point_inside_with_24_reflections_l215_215085


namespace range_of_x_l215_215527

theorem range_of_x (x y : ℝ) (h1 : 4 * x + y = 3) (h2 : -2 < y ∧ y ≤ 7) :
  -1 ≤ x ∧ x < 5 / 4 :=
sorry

end range_of_x_l215_215527


namespace problem1_problem2_l215_215586

noncomputable def part1 (a : ℝ) : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4} ∩ {x | x ≤ 2 * a - 5}
noncomputable def part2 (a : ℝ) : Prop := ∀ x : ℝ, (-2 ≤ x ∧ x ≤ 4) → (x ≤ 2 * a - 5)

theorem problem1 : part1 3 = {x | -2 ≤ x ∧ x ≤ 1} :=
by { sorry }

theorem problem2 : ∀ a : ℝ, (part2 a) ↔ (a ≥ 9/2) :=
by { sorry }

end problem1_problem2_l215_215586


namespace inequality_proof_l215_215936

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
    (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := 
by 
  sorry

end inequality_proof_l215_215936


namespace max_volume_day1_l215_215414

-- Define volumes of the containers
def volumes : List ℕ := [9, 13, 17, 19, 20, 38]

-- Define conditions: sold containers volumes
def condition_on_first_day (s: List ℕ) := s.length = 3
def condition_on_second_day (s: List ℕ) := s.length = 2

-- Define condition: total and relative volumes sold
def volume_sold_first_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0
def volume_sold_second_day (s: List ℕ) : ℕ := s.foldr (λ x acc => x + acc) 0

def volume_sold_total (s1 s2: List ℕ) := volume_sold_first_day s1 + volume_sold_second_day s2 = 116
def volume_ratio (s1 s2: List ℕ) := volume_sold_first_day s1 = 2 * volume_sold_second_day s2 

-- The goal is to prove the maximum possible volume_sold_first_day
theorem max_volume_day1 (s1 s2: List ℕ) 
  (h1: condition_on_first_day s1)
  (h2: condition_on_second_day s2)
  (h3: volume_sold_total s1 s2)
  (h4: volume_ratio s1 s2) : 
  ∃(max_volume: ℕ), max_volume = 66 :=
sorry

end max_volume_day1_l215_215414


namespace no_integer_k_sq_plus_k_plus_one_divisible_by_101_l215_215599

theorem no_integer_k_sq_plus_k_plus_one_divisible_by_101 (k : ℤ) : 
  (k^2 + k + 1) % 101 ≠ 0 := 
by
  sorry

end no_integer_k_sq_plus_k_plus_one_divisible_by_101_l215_215599


namespace width_of_foil_covered_prism_l215_215851

noncomputable def foil_covered_prism_width : ℕ :=
  let (l, w, h) := (4, 8, 4)
  let inner_width := 2 * l
  let increased_width := w + 2
  increased_width

theorem width_of_foil_covered_prism : foil_covered_prism_width = 10 := 
by
  let l := 4
  let w := 2 * l
  let h := w / 2
  have volume : l * w * h = 128 := by
    sorry
  have width_foil_covered := w + 2
  have : foil_covered_prism_width = width_foil_covered := by
    sorry
  sorry

end width_of_foil_covered_prism_l215_215851


namespace original_list_length_l215_215735

variable (n m : ℕ)   -- number of integers and the mean respectively
variable (l : List ℤ) -- the original list of integers

def mean (l : List ℤ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Condition 1: Appending 25 increases mean by 3
def condition1 (l : List ℤ) : Prop :=
  mean (25 :: l) = mean l + 3

-- Condition 2: Appending -4 to the enlarged list decreases the mean by 1.5
def condition2 (l : List ℤ) : Prop :=
  mean (-4 :: 25 :: l) = mean (25 :: l) - 1.5

theorem original_list_length (l : List ℤ) (h1 : condition1 l) (h2 : condition2 l) : l.length = 4 := by
  sorry

end original_list_length_l215_215735


namespace number_of_minibusses_l215_215465

def total_students := 156
def students_per_van := 10
def students_per_minibus := 24
def number_of_vans := 6

theorem number_of_minibusses : (total_students - number_of_vans * students_per_van) / students_per_minibus = 4 :=
by
  sorry

end number_of_minibusses_l215_215465


namespace total_charts_16_l215_215110

def total_charts_brought (number_of_associate_professors : Int) (number_of_assistant_professors : Int) : Int :=
  number_of_associate_professors * 1 + number_of_assistant_professors * 2

theorem total_charts_16 (A B : Int)
  (h1 : 2 * A + B = 11)
  (h2 : A + B = 9) :
  total_charts_brought A B = 16 :=
by {
  -- the proof will go here
  sorry
}

end total_charts_16_l215_215110


namespace sum_of_cubes_l215_215358

theorem sum_of_cubes (a b : ℕ) (h1 : 2 * x = a) (h2 : 3 * x = b) (h3 : b - a = 3) : a^3 + b^3 = 945 := by
  sorry

end sum_of_cubes_l215_215358


namespace f_sum_positive_l215_215201

noncomputable def f (x : ℝ) : ℝ := x + x^3

theorem f_sum_positive (x₁ x₂ x₃ : ℝ) (h₁₂ : x₁ + x₂ > 0) (h₂₃ : x₂ + x₃ > 0) (h₃₁ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := 
sorry

end f_sum_positive_l215_215201


namespace david_spent_difference_l215_215888

-- Define the initial amount, remaining amount, amount spent and the correct answer
def initial_amount : Real := 1800
def remaining_amount : Real := 500
def spent_amount : Real := initial_amount - remaining_amount
def correct_difference : Real := spent_amount - remaining_amount

-- Prove that the difference between the amount spent and the remaining amount is $800
theorem david_spent_difference : correct_difference = 800 := by
  sorry

end david_spent_difference_l215_215888


namespace abs_inequality_condition_l215_215828

theorem abs_inequality_condition (a : ℝ) : 
  (a < 2) ↔ ∀ x : ℝ, |x - 2| + |x| > a :=
sorry

end abs_inequality_condition_l215_215828


namespace work_day_meeting_percent_l215_215247

open Nat

theorem work_day_meeting_percent :
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  (total_meeting_time : ℚ) / work_day_minutes * 100 = 35 := 
by
  let work_day_minutes := 10 * 60
  let first_meeting := 35
  let second_meeting := 2 * first_meeting
  let third_meeting := first_meeting + second_meeting
  let total_meeting_time := first_meeting + second_meeting + third_meeting
  sorry

end work_day_meeting_percent_l215_215247


namespace cubic_sum_of_roots_l215_215425

theorem cubic_sum_of_roots (r s a b : ℝ) (h1 : r + s = a) (h2 : r * s = b) : 
  r^3 + s^3 = a^3 - 3 * a * b :=
by
  sorry

end cubic_sum_of_roots_l215_215425


namespace number_of_nickels_l215_215084

-- Define the conditions
variable (m : ℕ) -- Total number of coins initially
variable (v : ℕ) -- Total value of coins initially in cents
variable (n : ℕ) -- Number of nickels

-- State the conditions in terms of mathematical equations
-- Condition 1: Average value is 25 cents
axiom avg_value_initial : v = 25 * m

-- Condition 2: Adding one half-dollar (50 cents) results in average of 26 cents
axiom avg_value_after_half_dollar : v + 50 = 26 * (m + 1)

-- Define the relationship between the number of each type of coin and the total value
-- We sum the individual products of the count of each type and their respective values
axiom total_value_definition : v = 5 * n  -- since the problem already validates with total_value == 25m

-- Question to prove
theorem number_of_nickels : n = 30 :=
by
  -- Since we are not providing proof, we will use sorry to indicate the proof is omitted
  sorry

end number_of_nickels_l215_215084


namespace firecracker_confiscation_l215_215238

variables
  (F : ℕ)   -- Total number of firecrackers bought
  (R : ℕ)   -- Number of firecrackers remaining after confiscation
  (D : ℕ)   -- Number of defective firecrackers
  (G : ℕ)   -- Number of good firecrackers before setting off half
  (C : ℕ)   -- Number of firecrackers confiscated

-- Define the conditions:
def conditions := 
  F = 48 ∧
  D = R / 6 ∧
  G = 2 * 15 ∧
  R - D = G ∧
  F - R = C

-- The theorem to prove:
theorem firecracker_confiscation (h : conditions F R D G C) : C = 12 := 
  sorry

end firecracker_confiscation_l215_215238


namespace sum_first_n_terms_l215_215366

-- Define the sequence a_n
def geom_seq (a : ℕ → ℕ) (r : ℕ) : Prop :=
  ∀ n, a (n + 1) = r * a n

-- Define the main conditions from the problem
axiom a7_cond (a : ℕ → ℕ) : a 7 = 8 * a 4
axiom arithmetic_seq_cond (a : ℕ → ℕ) : (1 / 2 : ℝ) * a 2 < (a 3 - 4) ∧ (a 3 - 4) < (a 4 - 12)

-- Define the sequences a_n and b_n using the conditions
def a_n (n : ℕ) : ℕ := 2^(n + 1)
def b_n (n : ℕ) : ℤ := (-1)^n * (Int.ofNat (n + 1))

-- Define the sum of the first n terms of b_n
noncomputable def T_n (n : ℕ) : ℤ :=
  (Finset.range n).sum b_n

-- Main theorem statement
theorem sum_first_n_terms (k : ℕ) : |T_n k| = 20 → k = 40 ∨ k = 37 :=
sorry

end sum_first_n_terms_l215_215366


namespace range_of_a_l215_215574

-- Define the propositions
def Proposition_p (a : ℝ) := ∀ x : ℝ, x > 0 → x + 1/x > a
def Proposition_q (a : ℝ) := ∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0

-- Define the main theorem
theorem range_of_a (a : ℝ) (h1 : ¬ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) 
(h2 : (∀ x : ℝ, x > 0 → x + 1/x > a) ∧ (∃ x : ℝ, x^2 - 2*a*x + 1 ≤ 0) = false) :
a ≥ 2 :=
sorry

end range_of_a_l215_215574


namespace total_number_of_digits_l215_215842

theorem total_number_of_digits (n S S₅ S₃ : ℕ) (h1 : S = 20 * n) (h2 : S₅ = 5 * 12) (h3 : S₃ = 3 * 33) : n = 8 :=
by
  sorry

end total_number_of_digits_l215_215842


namespace no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l215_215559

theorem no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100 :
  ¬ ∃ (a b c d : ℕ), a + b + c + d = 2^100 ∧ a * b * c * d = 17^100 :=
by
  sorry

end no_four_nat_numbers_sum_2_pow_100_prod_17_pow_100_l215_215559


namespace max_value_fraction_l215_215910

theorem max_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∀ z, z = (x / (2 * x + y) + y / (x + 2 * y)) → z ≤ (2 / 3) :=
by
  sorry

end max_value_fraction_l215_215910


namespace cos_alpha_minus_7pi_over_2_l215_215866

-- Given conditions
variable (α : Real) (h : Real.sin α = 3/5)

-- Statement to prove
theorem cos_alpha_minus_7pi_over_2 : Real.cos (α - 7 * Real.pi / 2) = -3/5 :=
by
  sorry

end cos_alpha_minus_7pi_over_2_l215_215866


namespace hexagon_inscribed_in_square_area_l215_215450

noncomputable def hexagon_area (side_length : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * side_length^2

theorem hexagon_inscribed_in_square_area (AB BC : ℝ) (BDEF_square : BDEF_is_square) (hAB : AB = 2) (hBC : BC = 2) :
  hexagon_area (2 * Real.sqrt 2) = 12 * Real.sqrt 3 :=
by
  sorry

-- Definitions to assume the necessary conditions in the theorem (placeholders)
-- Assuming a structure of BDEF_is_square to represent the property that BDEF is a square
structure BDEF_is_square :=
(square : Prop)

end hexagon_inscribed_in_square_area_l215_215450


namespace determine_omega_phi_l215_215555

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem determine_omega_phi (ω φ : ℝ) (x : ℝ)
  (h₁ : 0 < ω) (h₂ : |φ| < Real.pi)
  (h₃ : f ω φ (5 * Real.pi / 8) = 2)
  (h₄ : f ω φ (11 * Real.pi / 8) = 0)
  (h₅ : (2 * Real.pi / ω) > 2 * Real.pi) :
  ω = 2 / 3 ∧ φ = Real.pi / 12 :=
sorry

end determine_omega_phi_l215_215555


namespace Kayla_total_items_l215_215534

-- Define the given conditions
def Theresa_chocolate_bars := 12
def Theresa_soda_cans := 18
def twice (x : Nat) := 2 * x

-- Define the unknowns
def Kayla_chocolate_bars := Theresa_chocolate_bars / 2
def Kayla_soda_cans := Theresa_soda_cans / 2

-- Final proof statement to be shown
theorem Kayla_total_items :
  Kayla_chocolate_bars + Kayla_soda_cans = 15 := 
by
  simp [Kayla_chocolate_bars, Kayla_soda_cans]
  norm_num
  sorry

end Kayla_total_items_l215_215534


namespace find_integers_l215_215773

theorem find_integers (n : ℤ) : (n^2 - 13 * n + 36 < 0) ↔ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 :=
by
  sorry

end find_integers_l215_215773


namespace track_length_is_450_l215_215105

theorem track_length_is_450 (x : ℝ) (d₁ : ℝ) (d₂ : ℝ)
  (h₁ : d₁ = 150)
  (h₂ : x - d₁ = 120)
  (h₃ : d₂ = 200)
  (h₄ : ∀ (d₁ d₂ : ℝ) (t₁ t₂ : ℝ), t₁ / t₂ = d₁ / d₂)
  : x = 450 := by
  sorry

end track_length_is_450_l215_215105


namespace point_transformation_l215_215529

theorem point_transformation : ∀ (P : ℝ×ℝ), P = (1, -2) → P = (-1, 2) :=
by
  sorry

end point_transformation_l215_215529


namespace quiz_passing_condition_l215_215270

theorem quiz_passing_condition (P Q : Prop) :
  (Q → P) → 
    (¬P → ¬Q) ∧ 
    (¬Q → ¬P) ∧ 
    (P → Q) :=
by sorry

end quiz_passing_condition_l215_215270


namespace find_number_of_women_l215_215507

-- Define the work rate variables and the equations from conditions
variables (m w : ℝ) (x : ℝ)

-- Define the first condition
def condition1 : Prop := 3 * m + x * w = 6 * m + 2 * w

-- Define the second condition
def condition2 : Prop := 4 * m + 2 * w = (5 / 7) * (3 * m + x * w)

-- The theorem stating that, given the above conditions, x must be 23
theorem find_number_of_women (hmw : m = 7 * w) (h1 : condition1 m w x) (h2 : condition2 m w x) : x = 23 :=
sorry

end find_number_of_women_l215_215507


namespace linear_function_in_quadrants_l215_215796

section LinearFunctionQuadrants

variable (m : ℝ)

def passesThroughQuadrants (m : ℝ) : Prop :=
  (m + 1 > 0) ∧ (m - 1 > 0)

theorem linear_function_in_quadrants (h : passesThroughQuadrants m) : m > 1 :=
by sorry

end LinearFunctionQuadrants

end linear_function_in_quadrants_l215_215796


namespace uncle_bob_can_park_l215_215883

-- Define the conditions
def total_spaces : Nat := 18
def cars : Nat := 15
def rv_spaces : Nat := 3

-- Define a function to calculate the probability (without implementation)
noncomputable def probability_RV_can_park (total_spaces cars rv_spaces : Nat) : Rat :=
  if h : rv_spaces <= total_spaces - cars then
    -- The probability calculation logic would go here
    16 / 51
  else
    0

-- The theorem stating the desired result
theorem uncle_bob_can_park : probability_RV_can_park total_spaces cars rv_spaces = 16 / 51 :=
  sorry

end uncle_bob_can_park_l215_215883


namespace largest_perfect_square_factor_4410_l215_215370

theorem largest_perfect_square_factor_4410 : ∀ (n : ℕ), n = 441 → (∃ k : ℕ, k^2 ∣ 4410 ∧ ∀ m : ℕ, m^2 ∣ 4410 → m^2 ≤ k^2) := 
by
  sorry

end largest_perfect_square_factor_4410_l215_215370


namespace triangle_height_l215_215245

theorem triangle_height (base height : ℝ) (area : ℝ) (h_base : base = 4) (h_area : area = 12) (h_area_eq : area = (base * height) / 2) :
  height = 6 :=
by
  sorry

end triangle_height_l215_215245


namespace rate_of_current_l215_215825

-- Definitions of the conditions
def downstream_speed : ℝ := 30  -- in kmph
def upstream_speed : ℝ := 10    -- in kmph
def still_water_rate : ℝ := 20  -- in kmph

-- Calculating the rate of the current
def current_rate : ℝ := downstream_speed - still_water_rate

-- Proof statement
theorem rate_of_current :
  current_rate = 10 :=
by
  sorry

end rate_of_current_l215_215825


namespace range_of_x_l215_215275

noncomputable def f (x : ℝ) : ℝ := (5 / (x^2)) - (3 * (x^2)) + 2

theorem range_of_x :
  { x : ℝ | f 1 < f (Real.log x / Real.log 3) } = { x : ℝ | (1 / 3) < x ∧ x < 1 ∨ 1 < x ∧ x < 3 } :=
by
  sorry

end range_of_x_l215_215275


namespace rectangle_dimension_correct_l215_215581

-- Definition of the Width and Length based on given conditions
def width := 3 / 2
def length := 3

-- Perimeter and Area conditions
def perimeter_condition (w l : ℝ) := 2 * (w + l) = 2 * (w * l)
def length_condition (w l : ℝ) := l = 2 * w

-- Main theorem statement
theorem rectangle_dimension_correct :
  ∃ (w l : ℝ), perimeter_condition w l ∧ length_condition w l ∧ w = width ∧ l = length :=
by {
  -- add sorry to skip the proof
  sorry
}

end rectangle_dimension_correct_l215_215581


namespace possible_values_of_a_l215_215572

-- Declare the sets M and N based on given conditions.
def M : Set ℝ := {x | x^2 + x - 6 = 0}
def N (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

-- Define a proof where the set of possible values for a is {-1, 0, 2/3}
theorem possible_values_of_a : 
  {a : ℝ | N a ⊆ M} = {-1, 0, 2 / 3} := 
by 
  sorry

end possible_values_of_a_l215_215572


namespace polynomial_expansion_l215_215356

theorem polynomial_expansion :
  (∀ x : ℝ, (x + 1)^3 * (x + 2)^2 = x^5 + a_1 * x^4 + a_2 * x^3 + a_3 * x^2 + 16 * x + 4) :=
by
  sorry

end polynomial_expansion_l215_215356


namespace correct_system_of_equations_l215_215945

variable (x y : Real)

-- Conditions
def condition1 : Prop := y = x + 4.5
def condition2 : Prop := 0.5 * y = x - 1

-- Main statement representing the correct system of equations
theorem correct_system_of_equations : condition1 x y ∧ condition2 x y :=
  sorry

end correct_system_of_equations_l215_215945


namespace smallest_possible_value_l215_215075

theorem smallest_possible_value (a b c d : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  (a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (b + d)) + (1 / (c + d))) ≥ 8 := 
sorry

end smallest_possible_value_l215_215075


namespace simplify_expr_l215_215942

variable (a b : ℤ)

theorem simplify_expr :
  (22 * a + 60 * b) + (10 * a + 29 * b) - (9 * a + 50 * b) = 23 * a + 39 * b :=
by
  sorry

end simplify_expr_l215_215942


namespace no_solution_pos_integers_l215_215832

theorem no_solution_pos_integers (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a + b + c + d - 3 ≠ a * b + c * d := 
by
  sorry

end no_solution_pos_integers_l215_215832


namespace find_dividend_l215_215650

theorem find_dividend
  (R : ℕ)
  (Q : ℕ)
  (D : ℕ)
  (hR : R = 6)
  (hD_eq_5Q : D = 5 * Q)
  (hD_eq_3R_plus_2 : D = 3 * R + 2) :
  D * Q + R = 86 :=
by
  sorry

end find_dividend_l215_215650


namespace range_of_a_l215_215616

noncomputable def setA : Set ℝ := {x | 3 + 2 * x - x^2 >= 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) : (setA ∩ setB a).Nonempty → a < 3 :=
by
  sorry

end range_of_a_l215_215616


namespace geom_seq_solution_l215_215870

theorem geom_seq_solution (a b x y : ℝ) 
  (h1 : x * (1 + y + y^2) = a) 
  (h2 : x^2 * (1 + y^2 + y^4) = b) :
  x = 1 / (4 * a) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨ 
  x = 1 / (4 * a) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∧
  y = 1 / (2 * (a^2 - b)) * (a^2 + b - Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) ∨
  y = 1 / (2 * (a^2 - b)) * (a^2 + b + Real.sqrt ((3 * a^2 - b) * (3 * b - a^2))) := 
  sorry

end geom_seq_solution_l215_215870


namespace find_ratio_eq_eighty_six_l215_215124

-- Define the set S
def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 45}

-- Define the sum of the first n natural numbers function
def sum_n_nat (n : ℕ) : ℕ := n * (n + 1) / 2

-- Define our specific scenario setup
def selected_numbers (x y : ℕ) : Prop :=
  x ∈ S ∧ y ∈ S ∧ x ≠ y ∧ x * y = sum_n_nat 45 - (x + y)

-- Prove the resulting ratio condition
theorem find_ratio_eq_eighty_six (x y : ℕ) (h : selected_numbers x y) : 
  x < y → y / x = 86 :=
by
  sorry

end find_ratio_eq_eighty_six_l215_215124


namespace max_value_of_f_l215_215175

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 5) * Real.sin (x + Real.pi / 3) + Real.cos (x - Real.pi / 6)

theorem max_value_of_f : ∃ x, f x ≤ 6 / 5 :=
sorry

end max_value_of_f_l215_215175


namespace problem1_problem2_l215_215289

noncomputable def f : ℝ → ℝ := -- we assume f is noncomputable since we know its explicit form in the desired interval
sorry

axiom periodic_f (x : ℝ) : f (x + 5) = f x
axiom odd_f {x : ℝ} (h : -1 ≤ x ∧ x ≤ 1) : f (-x) = -f x
axiom quadratic_f {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * (x - 2) ^ 2 - 5
axiom minimum_f : f 2 = -5

theorem problem1 : f 1 + f 4 = 0 :=
by
  sorry

theorem problem2 {x : ℝ} (h : 1 ≤ x ∧ x ≤ 4) : f x = 2 * x ^ 2 - 8 * x + 3 :=
by
  sorry

end problem1_problem2_l215_215289


namespace correct_dispersion_statements_l215_215339

def statement1 (make_use_of_data : Prop) : Prop :=
make_use_of_data = true

def statement2 (multi_numerical_values : Prop) : Prop :=
multi_numerical_values = true

def statement3 (dispersion_large_value_small : Prop) : Prop :=
dispersion_large_value_small = false

theorem correct_dispersion_statements
  (make_use_of_data : Prop)
  (multi_numerical_values : Prop)
  (dispersion_large_value_small : Prop)
  (h1 : statement1 make_use_of_data)
  (h2 : statement2 multi_numerical_values)
  (h3 : statement3 dispersion_large_value_small) :
  (make_use_of_data ∧ multi_numerical_values ∧ ¬ dispersion_large_value_small) = true :=
by
  sorry

end correct_dispersion_statements_l215_215339


namespace smallest_b_exists_l215_215858

theorem smallest_b_exists :
  ∃ b : ℕ, (∀ r s : ℤ, r * s = 4032 ∧ r + s = b) ∧
    (∀ b' : ℕ, (∀ r' s' : ℤ, r' * s' = 4032 ∧ r' + s' = b') → b ≤ b') :=
sorry

end smallest_b_exists_l215_215858


namespace magic_ink_combinations_l215_215062

def herbs : ℕ := 4
def essences : ℕ := 6
def incompatible_herbs : ℕ := 3

theorem magic_ink_combinations :
  herbs * essences - incompatible_herbs = 21 := 
  by
  sorry

end magic_ink_combinations_l215_215062


namespace original_average_of_numbers_l215_215962

theorem original_average_of_numbers 
  (A : ℝ) 
  (h : (A * 15) + (11 * 15) = 51 * 15) : 
  A = 40 :=
sorry

end original_average_of_numbers_l215_215962


namespace regular_pyramid_cannot_be_hexagonal_l215_215802

theorem regular_pyramid_cannot_be_hexagonal (n : ℕ) (h₁ : n = 6) (base_edge_length slant_height : ℝ) 
  (reg_pyramid : base_edge_length = slant_height) : false :=
by
  sorry

end regular_pyramid_cannot_be_hexagonal_l215_215802


namespace A_salary_l215_215811

theorem A_salary (x y : ℝ) (h1 : x + y = 7000) (h2 : 0.05 * x = 0.15 * y) : x = 5250 :=
by
  sorry

end A_salary_l215_215811


namespace solve_problem_l215_215644
open Complex

noncomputable def problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  (a ≠ -1) ∧ (b ≠ -1) ∧ (c ≠ -1) ∧ (d ≠ -1) ∧ (ω ^ 4 = 1) ∧ (ω ≠ 1) ∧
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 3 / ω ^ 2)
  
theorem solve_problem {a b c d : ℝ} {ω : ℂ} (h : problem a b c d ω) : 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2) :=
sorry

end solve_problem_l215_215644


namespace length_FD_of_folded_square_l215_215386

theorem length_FD_of_folded_square :
  let A := (0, 0)
  let B := (8, 0)
  let D := (0, 8)
  let C := (8, 8)
  let E := (6, 0)
  let F := (8, 8 - (FD : ℝ))
  (ABCD_square : ∀ {x y : ℝ}, (x = 0 ∨ x = 8) ∧ (y = 0 ∨ y = 8)) →  
  let DE := (6 - 0 : ℝ)
  let Pythagorean_statement := (8 - FD) ^ 2 = FD ^ 2 + 6 ^ 2
  ∃ FD : ℝ, FD = 7 / 4 :=
sorry

end length_FD_of_folded_square_l215_215386


namespace train_total_travel_time_l215_215310

noncomputable def totalTravelTime (d1 d2 s1 s2 : ℝ) : ℝ :=
  (d1 / s1) + (d2 / s2)

theorem train_total_travel_time : 
  totalTravelTime 150 200 50 80 = 5.5 :=
by
  sorry

end train_total_travel_time_l215_215310


namespace ratio_monkeys_camels_l215_215469

-- Definitions corresponding to conditions
variables (zebras camels monkeys giraffes : ℕ)
variables (multiple : ℕ)

-- Conditions
def condition1 := zebras = 12
def condition2 := camels = zebras / 2
def condition3 := monkeys = camels * multiple
def condition4 := giraffes = 2
def condition5 := monkeys = giraffes + 22

-- Question: What is the ratio of monkeys to camels? Prove it is 4:1 given the conditions.
theorem ratio_monkeys_camels (zebras camels monkeys giraffes multiple : ℕ) 
  (h1 : condition1 zebras) 
  (h2 : condition2 zebras camels)
  (h3 : condition3 camels monkeys multiple)
  (h4 : condition4 giraffes)
  (h5 : condition5 monkeys giraffes) :
  multiple = 4 :=
sorry

end ratio_monkeys_camels_l215_215469


namespace fuel_first_third_l215_215288

-- Defining constants based on conditions
def total_fuel := 60
def fuel_second_third := total_fuel / 3
def fuel_final_third := fuel_second_third / 2

-- Defining what we need to prove
theorem fuel_first_third :
  total_fuel - (fuel_second_third + fuel_final_third) = 30 :=
by
  sorry

end fuel_first_third_l215_215288


namespace solve_math_problem_l215_215439

noncomputable def math_problem : Prop :=
  ∃ (ω α β : ℂ), (ω^5 = 1) ∧ (ω ≠ 1) ∧ (α = ω + ω^2) ∧ (β = ω^3 + ω^4) ∧
  (∀ x : ℂ, x^2 + x + 3 = 0 → x = α ∨ x = β) ∧ (α + β = -1) ∧ (α * β = 3)

theorem solve_math_problem : math_problem := sorry

end solve_math_problem_l215_215439


namespace middle_number_of_ratio_l215_215364

theorem middle_number_of_ratio (x : ℝ) (h : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862) : 2 * x = 14 :=
sorry

end middle_number_of_ratio_l215_215364


namespace find_n_l215_215456

noncomputable def C (n : ℕ) : ℝ :=
  352 * (1 - 1 / 2 ^ n) / (1 - 1 / 2)

noncomputable def D (n : ℕ) : ℝ :=
  992 * (1 - 1 / (-2) ^ n) / (1 + 1 / 2)

theorem find_n (n : ℕ) (h : 1 ≤ n) : C n = D n ↔ n = 1 := by
  sorry

end find_n_l215_215456


namespace alex_buys_15_pounds_of_wheat_l215_215765

theorem alex_buys_15_pounds_of_wheat (w o : ℝ) (h1 : w + o = 30) (h2 : 72 * w + 36 * o = 1620) : w = 15 :=
by
  sorry

end alex_buys_15_pounds_of_wheat_l215_215765


namespace sample_variance_is_two_l215_215702

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : (1 / 5) * ((-1 - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end sample_variance_is_two_l215_215702


namespace find_2n_plus_m_l215_215016

theorem find_2n_plus_m (n m : ℤ) (h1 : 3 * n - m < 5) (h2 : n + m > 26) (h3 : 3 * m - 2 * n < 46) : 
  2 * n + m = 36 := 
sorry

end find_2n_plus_m_l215_215016


namespace infinite_rel_prime_set_of_form_2n_minus_3_l215_215400

theorem infinite_rel_prime_set_of_form_2n_minus_3 : ∃ S : Set ℕ, (∀ x ∈ S, ∃ n : ℕ, x = 2^n - 3) ∧ 
  (∀ x ∈ S, ∀ y ∈ S, x ≠ y → Nat.gcd x y = 1) ∧ S.Infinite := 
by
  sorry

end infinite_rel_prime_set_of_form_2n_minus_3_l215_215400


namespace part1_part2_l215_215441

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if 2a sin B = sqrt(3) b and A is an acute angle, then A = 60 degrees. -/
theorem part1 {a b : ℝ} {A B : ℝ} (h1 : 2 * a * Real.sin B = Real.sqrt 3 * b)
  (h2 : 0 < A ∧ A < Real.pi / 2) : A = Real.pi / 3 :=
sorry

/-- Given that in triangle ABC, sides opposite to angles A, B, and C are a, b, and c respectively
     if b = 5, c = sqrt(5), and cos C = 9 / 10, then a = 4 or a = 5. -/
theorem part2 {a b c : ℝ} {C : ℝ} (h1 : b = 5) (h2 : c = Real.sqrt 5) 
  (h3 : Real.cos C = 9 / 10) : a = 4 ∨ a = 5 :=
sorry

end part1_part2_l215_215441


namespace max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l215_215200

theorem max_value_y_eq_x_mul_2_minus_x (x : ℝ) (h : 0 < x ∧ x < 3 / 2) : ∃ y : ℝ, y = x * (2 - x) ∧ y ≤ 1 :=
sorry

theorem min_value_y_eq_x_plus_4_div_x_minus_3 (x : ℝ) (h : x > 3) : ∃ y : ℝ, y = x + 4 / (x - 3) ∧ y ≥ 7 :=
sorry

end max_value_y_eq_x_mul_2_minus_x_min_value_y_eq_x_plus_4_div_x_minus_3_l215_215200


namespace total_computers_sold_l215_215131

theorem total_computers_sold (T : ℕ) (h_half_sales_laptops : 2 * T / 2 = T)
        (h_third_sales_netbooks : 3 * T / 3 = T)
        (h_desktop_sales : T - T / 2 - T / 3 = 12) : T = 72 :=
by
  sorry

end total_computers_sold_l215_215131


namespace steps_taken_l215_215114

noncomputable def andrewSpeed : ℝ := 1 -- Let Andrew's speed be represented by 1 feet per minute
noncomputable def benSpeed : ℝ := 3 * andrewSpeed -- Ben's speed is 3 times Andrew's speed
noncomputable def totalDistance : ℝ := 21120 -- Distance between the houses in feet
noncomputable def andrewStep : ℝ := 3 -- Each step of Andrew covers 3 feet

theorem steps_taken : (totalDistance / (andrewSpeed + benSpeed)) * andrewSpeed / andrewStep = 1760 := by
  sorry -- proof to be filled in later

end steps_taken_l215_215114


namespace repeating_decimals_sum_l215_215017

-- Define the repeating decimals as rational numbers
def dec_0_3 : ℚ := 1 / 3
def dec_0_02 : ℚ := 2 / 99
def dec_0_0004 : ℚ := 4 / 9999

-- State the theorem that we need to prove
theorem repeating_decimals_sum :
  dec_0_3 + dec_0_02 + dec_0_0004 = 10581 / 29889 :=
by
  sorry

end repeating_decimals_sum_l215_215017


namespace number_of_girls_is_eleven_l215_215860

-- Conditions transformation
def boys_wear_red_hats : Prop := true
def girls_wear_yellow_hats : Prop := true
def teachers_wear_blue_hats : Prop := true
def cannot_see_own_hat : Prop := true
def little_qiang_sees_hats (x k : ℕ) : Prop := (x + 2) = (x + 2)
def little_hua_sees_hats (x k : ℕ) : Prop := x = 2 * k
def teacher_sees_hats (x k : ℕ) : Prop := k + 2 = (x + 2) + k - 11

-- Proof Statement
theorem number_of_girls_is_eleven (x k : ℕ) (h1 : boys_wear_red_hats)
  (h2 : girls_wear_yellow_hats) (h3 : teachers_wear_blue_hats)
  (h4 : cannot_see_own_hat) (hq : little_qiang_sees_hats x k)
  (hh : little_hua_sees_hats x k) (ht : teacher_sees_hats x k) : x = 11 :=
sorry

end number_of_girls_is_eleven_l215_215860


namespace find_r_l215_215442

-- Declaring the roots of the first polynomial
variables (a b m : ℝ)
-- Declaring the roots of the second polynomial
variables (p r : ℝ)

-- Assumptions based on the given conditions
def roots_of_first_eq : Prop :=
  a + b = m ∧ a * b = 3

def roots_of_second_eq : Prop :=
  ∃ (p : ℝ), (a^2 + 1/b) * (b^2 + 1/a) = r

-- The desired theorem
theorem find_r 
  (h1 : roots_of_first_eq a b m)
  (h2 : (a^2 + 1/b) * (b^2 + 1/a) = r) :
  r = 46/3 := by sorry

end find_r_l215_215442


namespace express_B_using_roster_l215_215265

open Set

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem express_B_using_roster :
  B = {4, 9, 16} := by
  sorry

end express_B_using_roster_l215_215265


namespace phase_shift_of_sine_function_l215_215471

theorem phase_shift_of_sine_function :
  ∀ x : ℝ, y = 3 * Real.sin (3 * x + π / 4) → (∃ φ : ℝ, φ = -π / 12) :=
by sorry

end phase_shift_of_sine_function_l215_215471


namespace pyramid_volume_l215_215153

theorem pyramid_volume (S A : ℝ)
  (h_surface : 3 * S = 432)
  (h_half_triangular : A = 0.5 * S) :
  (1 / 3) * S * (12 * Real.sqrt 3) = 288 * Real.sqrt 3 :=
by
  sorry

end pyramid_volume_l215_215153


namespace difference_in_cents_l215_215487

theorem difference_in_cents (pennies dimes : ℕ) (h : pennies + dimes = 5050) (hpennies : 1 ≤ pennies) (hdimes : 1 ≤ dimes) : 
  let total_value := pennies + 10 * dimes
  let max_value := 50500 - 9 * 1
  let min_value := 50500 - 9 * 5049
  max_value - min_value = 45432 := 
by 
  -- proof goes here
  sorry

end difference_in_cents_l215_215487


namespace sufficient_condition_inequality_l215_215005

theorem sufficient_condition_inequality (k : ℝ) :
  (k = 0 ∨ (-3 < k ∧ k < 0)) → ∀ x : ℝ, 2 * k * x^2 + k * x - 3 / 8 < 0 :=
sorry

end sufficient_condition_inequality_l215_215005


namespace tate_education_ratio_l215_215068

theorem tate_education_ratio
  (n : ℕ)
  (m : ℕ)
  (h1 : n > 1)
  (h2 : (n - 1) + m * (n - 1) = 12)
  (h3 : n = 4) :
  (m * (n - 1)) / (n - 1) = 3 := 
by 
  sorry

end tate_education_ratio_l215_215068


namespace pow_100_mod_18_l215_215077

theorem pow_100_mod_18 : (5 ^ 100) % 18 = 13 := by
  -- Define the conditions
  have h1 : (5 ^ 1) % 18 = 5 := by norm_num
  have h2 : (5 ^ 2) % 18 = 7 := by norm_num
  have h3 : (5 ^ 3) % 18 = 17 := by norm_num
  have h4 : (5 ^ 4) % 18 = 13 := by norm_num
  have h5 : (5 ^ 5) % 18 = 11 := by norm_num
  have h6 : (5 ^ 6) % 18 = 1 := by norm_num
  
  -- The required theorem is based on the conditions mentioned
  sorry

end pow_100_mod_18_l215_215077


namespace necessary_but_not_sufficient_l215_215192

theorem necessary_but_not_sufficient (a : ℝ) (ha : a > 1) : a^2 > a :=
sorry

end necessary_but_not_sufficient_l215_215192


namespace additional_men_required_l215_215214

variables (W_r : ℚ) (W : ℚ) (D : ℚ) (M : ℚ) (E : ℚ)

-- Given variables
def initial_work_rate := (2.5 : ℚ) / (50 * 100)
def remaining_work_length := (12.5 : ℚ)
def remaining_days := (200 : ℚ)
def initial_men := (50 : ℚ)
def additional_men_needed := (75 : ℚ)

-- Calculating the additional men required
theorem additional_men_required
  (calc_wr : W_r = initial_work_rate)
  (calc_wr_remain : W = remaining_work_length)
  (calc_days_remain : D = remaining_days)
  (calc_initial_men : M = initial_men)
  (calc_additional_men : M + E = (125 : ℚ)) :
  E = additional_men_needed :=
sorry

end additional_men_required_l215_215214


namespace find_center_and_tangent_slope_l215_215590

theorem find_center_and_tangent_slope :
  let C := { p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 - 6 * p.1 + 8 = 0 }
  let center := (3, 0)
  let k := - (Real.sqrt 2 / 4)
  (∃ c ∈ C, c = center) ∧
  (∃ q ∈ C, q.2 < 0 ∧ q.2 = k * q.1 ∧
             |3 * k| / Real.sqrt (k ^ 2 + 1) = 1) :=
by
  sorry

end find_center_and_tangent_slope_l215_215590


namespace largest_x_not_defined_l215_215667

theorem largest_x_not_defined : 
  (∀ x, (6 * x ^ 2 - 17 * x + 5 = 0) → x ≤ 2.5) ∧
  (∃ x, (6 * x ^ 2 - 17 * x + 5 = 0) ∧ x = 2.5) :=
by
  sorry

end largest_x_not_defined_l215_215667


namespace college_girls_count_l215_215407

theorem college_girls_count (B G : ℕ) (h1 : B / G = 8 / 5) (h2 : B + G = 546) : G = 210 :=
by
  sorry

end college_girls_count_l215_215407


namespace kids_went_home_l215_215278

theorem kids_went_home (initial_kids : ℝ) (remaining_kids : ℝ) (went_home : ℝ) 
  (h1 : initial_kids = 22.0) 
  (h2 : remaining_kids = 8.0) : went_home = 14.0 :=
by 
  sorry

end kids_went_home_l215_215278


namespace symmetric_point_coordinates_l215_215243

structure Point : Type where
  x : ℝ
  y : ℝ

def symmetric_y (p : Point) : Point :=
  { x := -p.x, y := p.y }

def symmetric_x (p : Point) : Point :=
  { x := p.x, y := -p.y }

def P : Point := { x := -10, y := -1 }

def P1 : Point := symmetric_y P

def P2 : Point := symmetric_x P1

theorem symmetric_point_coordinates :
  P2 = { x := 10, y := 1 } := by
  sorry

end symmetric_point_coordinates_l215_215243


namespace square_of_binomial_l215_215580

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, (x^2 - 18 * x + k) = (x + b)^2) ↔ k = 81 :=
by
  sorry

end square_of_binomial_l215_215580


namespace johns_train_speed_l215_215094

noncomputable def average_speed_of_train (D : ℝ) (V_t : ℝ) : ℝ := D / (0.8 * D / V_t + 0.2 * D / 20)

theorem johns_train_speed (D : ℝ) (V_t : ℝ) (h1 : average_speed_of_train D V_t = 50) : V_t = 64 :=
by
  sorry

end johns_train_speed_l215_215094


namespace ratio_of_polynomials_eq_962_l215_215809

open Real

theorem ratio_of_polynomials_eq_962 :
  (10^4 + 400) * (26^4 + 400) * (42^4 + 400) * (58^4 + 400) /
  ((2^4 + 400) * (18^4 + 400) * (34^4 + 400) * (50^4 + 400)) = 962 := 
sorry

end ratio_of_polynomials_eq_962_l215_215809


namespace problem_l215_215949

theorem problem : (112^2 - 97^2) / 15 = 209 := by
  sorry

end problem_l215_215949


namespace tank_fewer_eggs_in_second_round_l215_215790

variables (T E_total T_r2_diff : ℕ)

theorem tank_fewer_eggs_in_second_round
  (h1 : E_total = 400)
  (h2 : E_total = (T + (T - 10)) + (30 + 60))
  (h3 : T_r2_diff = T - 30) :
  T_r2_diff = 130 := by
    sorry

end tank_fewer_eggs_in_second_round_l215_215790


namespace speed_of_the_stream_l215_215338

theorem speed_of_the_stream (d v_s : ℝ) :
  (∀ (t_up t_down : ℝ), t_up = d / (57 - v_s) ∧ t_down = d / (57 + v_s) ∧ t_up = 2 * t_down) →
  v_s = 19 := by
  sorry

end speed_of_the_stream_l215_215338


namespace shares_difference_l215_215108

theorem shares_difference (x : ℝ) (h_ratio : 2.5 * x + 3.5 * x + 7.5 * x + 9.8 * x = (23.3 * x))
  (h_difference : 7.5 * x - 3.5 * x = 4500) : 9.8 * x - 2.5 * x = 8212.5 :=
by
  sorry

end shares_difference_l215_215108


namespace abc_area_l215_215641

def rectangle_area (length width : ℕ) : ℕ :=
  length * width

theorem abc_area :
  let smaller_side := 7
  let longer_side := 2 * smaller_side
  let length := 3 * longer_side -- since there are 3 identical rectangles placed side by side
  let width := smaller_side
  rectangle_area length width = 294 :=
by
  sorry

end abc_area_l215_215641


namespace roots_ratio_quadratic_l215_215701

theorem roots_ratio_quadratic (p : ℤ) (h : (∃ x1 x2 : ℤ, x1*x2 = -16 ∧ x1 + x2 = -p ∧ x2 = -4 * x1)) :
  p = 6 ∨ p = -6 :=
sorry

end roots_ratio_quadratic_l215_215701


namespace arrange_in_order_l215_215093

noncomputable def a := (Real.sqrt 2 / 2) * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c := Real.sqrt 3 / 2

theorem arrange_in_order : c < a ∧ a < b := 
by
  sorry

end arrange_in_order_l215_215093


namespace initial_bananas_l215_215433

theorem initial_bananas (x B : ℕ) (h1 : 840 * x = B) (h2 : 420 * (x + 2) = B) : x = 2 :=
by
  sorry

end initial_bananas_l215_215433


namespace sheepdog_rounded_up_percentage_l215_215272

/-- Carla's sheepdog rounded up a certain percentage of her sheep. We know the remaining 10% of the sheep  wandered off into the hills, which is 9 sheep out in the wilderness. There are 81 sheep in the pen. We need to prove that the sheepdog rounded up 90% of the total number of sheep. -/
theorem sheepdog_rounded_up_percentage (total_sheep pen_sheep wilderness_sheep : ℕ) 
  (h1 : wilderness_sheep = 9) 
  (h2 : pen_sheep = 81) 
  (h3 : wilderness_sheep = total_sheep / 10) :
  (pen_sheep * 100 / total_sheep) = 90 :=
sorry

end sheepdog_rounded_up_percentage_l215_215272


namespace complement_intersection_l215_215884

open Finset

def U : Finset ℕ := {1, 2, 3, 4, 5}
def A : Finset ℕ := {1, 3, 4}
def B : Finset ℕ := {3, 5}

theorem complement_intersection :
  (U \ (A ∩ B)) = {1, 2, 4, 5} :=
by sorry

end complement_intersection_l215_215884


namespace Tim_age_l215_215845

theorem Tim_age : ∃ (T : ℕ), (T = (3 * T + 2 - 12)) ∧ (T = 5) :=
by
  existsi 5
  sorry

end Tim_age_l215_215845


namespace relationship_between_a_b_c_l215_215892

noncomputable def a : ℝ := 1 / 3
noncomputable def b : ℝ := Real.sin (1 / 3)
noncomputable def c : ℝ := 1 / Real.pi

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l215_215892


namespace simplify_and_evaluate_l215_215350

theorem simplify_and_evaluate (a : ℕ) (h : a = 2023) : (a + 1) / a / (a - 1 / a) = 1 / 2022 :=
by
  sorry

end simplify_and_evaluate_l215_215350


namespace molecular_weight_of_NH4I_l215_215248

-- Define the conditions in Lean
def molecular_weight (moles grams: ℕ) : Prop :=
  grams / moles = 145

-- Statement of the proof problem
theorem molecular_weight_of_NH4I :
  molecular_weight 9 1305 :=
by
  -- Proof is omitted 
  sorry

end molecular_weight_of_NH4I_l215_215248


namespace total_pupils_correct_l215_215880

def number_of_girls : ℕ := 868
def difference_girls_boys : ℕ := 281
def number_of_boys : ℕ := number_of_girls - difference_girls_boys
def total_pupils : ℕ := number_of_girls + number_of_boys

theorem total_pupils_correct : total_pupils = 1455 := by
  sorry

end total_pupils_correct_l215_215880


namespace simplified_value_l215_215458

theorem simplified_value :
  (245^2 - 205^2) / 40 = 450 := by
  sorry

end simplified_value_l215_215458


namespace max_odd_integers_l215_215066

theorem max_odd_integers (chosen : Fin 5 → ℕ) (hpos : ∀ i, chosen i > 0) (heven : ∃ i, chosen i % 2 = 0) : 
  ∃ odd_count, odd_count = 4 ∧ (∀ i, i < 4 → chosen i % 2 = 1) := 
by 
  sorry

end max_odd_integers_l215_215066


namespace sum_T_mod_1000_l215_215964

open Nat

def T (a b : ℕ) : ℕ :=
  if h : a + b ≤ 6 then Nat.choose 6 a * Nat.choose 6 b * Nat.choose 6 (a + b) else 0

def sum_T : ℕ :=
  (Finset.range 7).sum (λ a => (Finset.range (7 - a)).sum (λ b => T a b))

theorem sum_T_mod_1000 : sum_T % 1000 = 564 := by
  sorry

end sum_T_mod_1000_l215_215964


namespace complex_roots_sum_condition_l215_215972

theorem complex_roots_sum_condition 
  (z1 z2 : ℂ) 
  (h1 : ∀ z, z ^ 2 + z + 1 = 0) 
  (h2 : z1 ^ 2 + z1 + 1 = 0)
  (h3 : z2 ^ 2 + z2 + 1 = 0) : 
  (z2 / (z1 + 1)) + (z1 / (z2 + 1)) = -2 := 
 sorry

end complex_roots_sum_condition_l215_215972


namespace club_members_neither_subject_l215_215777

theorem club_members_neither_subject (total members_cs members_bio members_both : ℕ)
  (h_total : total = 150)
  (h_cs : members_cs = 80)
  (h_bio : members_bio = 50)
  (h_both : members_both = 15) :
  total - ((members_cs - members_both) + (members_bio - members_both) + members_both) = 35 := by
  sorry

end club_members_neither_subject_l215_215777


namespace total_balloons_l215_215053

theorem total_balloons (sam_balloons_initial mary_balloons fred_balloons : ℕ) (h1 : sam_balloons_initial = 6)
    (h2 : mary_balloons = 7) (h3 : fred_balloons = 5) : sam_balloons_initial - fred_balloons + mary_balloons = 8 :=
by
  sorry

end total_balloons_l215_215053


namespace arithmetic_sequence_tenth_term_l215_215973

noncomputable def sum_of_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a1 + (n - 1) * d) / 2

def nth_term (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_tenth_term
  (a1 d : ℝ)
  (h1 : a1 + (a1 + d) + (a1 + 2 * d) = (a1 + 3 * d) + (a1 + 4 * d))
  (h2 : sum_of_arithmetic_sequence a1 d 5 = 60) :
  nth_term a1 d 10 = 26 :=
sorry

end arithmetic_sequence_tenth_term_l215_215973


namespace rate_per_meter_for_fencing_l215_215782

theorem rate_per_meter_for_fencing
  (w : ℕ) (length : ℕ) (perimeter : ℕ) (cost : ℕ)
  (h1 : length = w + 10)
  (h2 : perimeter = 2 * (length + w))
  (h3 : perimeter = 340)
  (h4 : cost = 2210) : (cost / perimeter : ℝ) = 6.5 := by
  sorry

end rate_per_meter_for_fencing_l215_215782


namespace eval_expr_at_values_l215_215986

variable (x y : ℝ)

def expr := 2 * (3 * x^2 + x * y^2)- 3 * (2 * x * y^2 - x^2) - 10 * x^2

theorem eval_expr_at_values : x = -1 → y = 0.5 → expr x y = 0 :=
by
  intros hx hy
  rw [hx, hy]
  sorry

end eval_expr_at_values_l215_215986


namespace sum_original_numbers_is_five_l215_215868

noncomputable def sum_original_numbers (a b c d : ℤ) : ℤ :=
  a + b + c + d

theorem sum_original_numbers_is_five (a b c d : ℤ) (hab : 10 * a + b = overline_ab) 
  (h : 100 * (10 * a + b) + 10 * c + 7 * d = 2024) : sum_original_numbers a b c d = 5 :=
sorry

end sum_original_numbers_is_five_l215_215868


namespace problem_f_2009_plus_f_2010_l215_215537

theorem problem_f_2009_plus_f_2010 (f : ℝ → ℝ) 
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (2 * x + 1) = f (2 * (x + 5 / 2) + 1))
  (h_f1 : f 1 = 5) :
  f 2009 + f 2010 = 0 :=
sorry

end problem_f_2009_plus_f_2010_l215_215537


namespace sum_of_coordinates_after_reflections_l215_215383

theorem sum_of_coordinates_after_reflections :
  let A := (3, 2)
  let B := (9, 18)
  let N := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let reflect_y (P : ℤ × ℤ) := (-P.1, P.2)
  let reflect_x (P : ℤ × ℤ) := (P.1, -P.2)
  let N' := reflect_y N
  let N'' := reflect_x N'
  N''.1 + N''.2 = -16 := by sorry

end sum_of_coordinates_after_reflections_l215_215383


namespace range_of_a_l215_215744

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) :=
by
  sorry

end range_of_a_l215_215744


namespace cube_faces_sum_l215_215862

theorem cube_faces_sum (a b c d e f : ℕ) (h1 : a = 12) (h2 : b = 13) (h3 : c = 14)
  (h4 : d = 15) (h5 : e = 16) (h6 : f = 17)
  (h_pairs : a + f = b + e ∧ b + e = c + d) :
  a + b + c + d + e + f = 87 := by
  sorry

end cube_faces_sum_l215_215862


namespace eval_power_81_11_over_4_l215_215933

theorem eval_power_81_11_over_4 : 81^(11/4) = 177147 := by
  sorry

end eval_power_81_11_over_4_l215_215933


namespace betty_watermelons_l215_215997

theorem betty_watermelons :
  ∃ b : ℕ, 
  (b + (b + 10) + (b + 20) + (b + 30) + (b + 40) = 200) ∧
  (b + 40 = 60) :=
by
  sorry

end betty_watermelons_l215_215997


namespace speed_boat_in_still_water_l215_215731

variable (V_b V_s t : ℝ)

def speed_of_boat := V_b

axiom stream_speed : V_s = 26

axiom time_relation : 2 * (t : ℝ) = 2 * t

axiom distance_relation : (V_b + V_s) * t = (V_b - V_s) * (2 * t)

theorem speed_boat_in_still_water : V_b = 78 :=
by {
  sorry
}

end speed_boat_in_still_water_l215_215731


namespace candy_cooking_time_l215_215604

def initial_temperature : ℝ := 60
def peak_temperature : ℝ := 240
def final_temperature : ℝ := 170
def heating_rate : ℝ := 5
def cooling_rate : ℝ := 7

theorem candy_cooking_time : ( (peak_temperature - initial_temperature) / heating_rate + (peak_temperature - final_temperature) / cooling_rate ) = 46 := by
  sorry

end candy_cooking_time_l215_215604


namespace relationship_among_a_b_c_l215_215969

noncomputable def a : ℝ := (Real.sqrt 2) / 2 * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b : ℝ := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c : ℝ := (Real.sqrt 3) / 2

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l215_215969


namespace words_per_hour_after_two_hours_l215_215816

theorem words_per_hour_after_two_hours 
  (total_words : ℕ) (initial_rate : ℕ) (initial_time : ℕ) (start_time_before_deadline : ℕ) 
  (words_written_in_first_phase : ℕ) (remaining_words : ℕ) (remaining_time : ℕ)
  (final_rate_per_hour : ℕ) :
  total_words = 1200 →
  initial_rate = 400 →
  initial_time = 2 →
  start_time_before_deadline = 4 →
  words_written_in_first_phase = initial_rate * initial_time →
  remaining_words = total_words - words_written_in_first_phase →
  remaining_time = start_time_before_deadline - initial_time →
  final_rate_per_hour = remaining_words / remaining_time →
  final_rate_per_hour = 200 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end words_per_hour_after_two_hours_l215_215816


namespace binom_21_13_l215_215693

theorem binom_21_13 : (Nat.choose 21 13) = 203490 :=
by
  have h1 : (Nat.choose 20 13) = 77520 := by sorry
  have h2 : (Nat.choose 20 12) = 125970 := by sorry
  have pascal : (Nat.choose 21 13) = (Nat.choose 20 13) + (Nat.choose 20 12) :=
    by rw [Nat.choose_succ_succ, h1, h2]
  exact pascal

end binom_21_13_l215_215693


namespace discount_equation_l215_215127

variable (P₀ P_f x : ℝ)
variable (h₀ : P₀ = 200)
variable (h₁ : P_f = 164)

theorem discount_equation :
  P₀ * (1 - x)^2 = P_f := by
  sorry

end discount_equation_l215_215127


namespace apples_used_l215_215618

theorem apples_used (x : ℕ) 
  (initial_apples : ℕ := 23) 
  (bought_apples : ℕ := 6) 
  (final_apples : ℕ := 9) 
  (h : (initial_apples - x) + bought_apples = final_apples) : 
  x = 20 :=
by
  sorry

end apples_used_l215_215618


namespace amount_of_first_alloy_used_is_15_l215_215235

-- Definitions of percentages and weights
def chromium_percentage_first_alloy : ℝ := 0.12
def chromium_percentage_second_alloy : ℝ := 0.08
def weight_second_alloy : ℝ := 40
def chromium_percentage_new_alloy : ℝ := 0.0909090909090909
def total_weight_new_alloy (x : ℝ) : ℝ := x + weight_second_alloy
def chromium_content_first_alloy (x : ℝ) : ℝ := chromium_percentage_first_alloy * x
def chromium_content_second_alloy : ℝ := chromium_percentage_second_alloy * weight_second_alloy
def total_chromium_content (x : ℝ) : ℝ := chromium_content_first_alloy x + chromium_content_second_alloy

-- The proof problem
theorem amount_of_first_alloy_used_is_15 :
  ∃ x : ℝ, total_chromium_content x = chromium_percentage_new_alloy * total_weight_new_alloy x ∧ x = 15 :=
by
  sorry

end amount_of_first_alloy_used_is_15_l215_215235


namespace compare_exponential_functions_l215_215570

theorem compare_exponential_functions (x : ℝ) (hx1 : 0 < x) :
  0.4^4 < 1 ∧ 1 < 4^0.4 :=
by sorry

end compare_exponential_functions_l215_215570


namespace five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l215_215695

variable (p q : ℕ)
variable (hp : p % 2 = 1)  -- p is odd
variable (hq : q % 2 = 1)  -- q is odd

theorem five_p_squared_plus_two_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (5 * p^2 + 2 * q^2) % 2 = 1 := 
sorry

theorem p_squared_plus_pq_plus_q_squared_odd 
    (hp : p % 2 = 1) 
    (hq : q % 2 = 1) : 
    (p^2 + p * q + q^2) % 2 = 1 := 
sorry

end five_p_squared_plus_two_q_squared_odd_p_squared_plus_pq_plus_q_squared_odd_l215_215695


namespace jace_total_distance_l215_215283

noncomputable def total_distance (s1 s2 s3 s4 s5 : ℝ) (t1 t2 t3 t4 t5 : ℝ) : ℝ :=
  s1 * t1 + s2 * t2 + s3 * t3 + s4 * t4 + s5 * t5

theorem jace_total_distance :
  total_distance 50 65 60 75 55 3 4.5 2.75 1.8333 2.6667 = 891.67 := by
  sorry

end jace_total_distance_l215_215283


namespace log_eqn_proof_l215_215227

theorem log_eqn_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 4 = 8)
  (h2 : Real.log a / Real.log 4 + Real.log b / Real.log 8 = 2) :
  Real.log a / Real.log 8 + Real.log b / Real.log 2 = -52 / 3 := 
by
  sorry

end log_eqn_proof_l215_215227


namespace work_completion_time_l215_215896

variable (p q : Type)

def efficient (p q : Type) : Prop :=
  ∃ (Wp Wq : ℝ), Wp = 1.5 * Wq ∧ Wp = 1 / 25

def work_done_together (p q : Type) := 1/15

theorem work_completion_time {p q : Type} (h1 : efficient p q) :
  ∃ d : ℝ, d = 15 :=
  sorry

end work_completion_time_l215_215896


namespace smallest_a_for_5880_to_be_cube_l215_215043

theorem smallest_a_for_5880_to_be_cube : ∃ (a : ℕ), a > 0 ∧ (∃ (k : ℕ), 5880 * a = k ^ 3) ∧
  (∀ (b : ℕ), b > 0 ∧ (∃ (k : ℕ), 5880 * b = k ^ 3) → a ≤ b) ∧ a = 1575 :=
sorry

end smallest_a_for_5880_to_be_cube_l215_215043


namespace number_of_pupils_l215_215651

-- Define the number of total people
def total_people : ℕ := 803

-- Define the number of parents
def parents : ℕ := 105

-- We need to prove the number of pupils is 698
theorem number_of_pupils : (total_people - parents) = 698 := 
by
  -- Skip the proof steps
  sorry

end number_of_pupils_l215_215651


namespace complement_intersection_l215_215185

open Set

-- Definitions of the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

-- The theorem we want to prove
theorem complement_intersection :
  (compl M ∩ N) = {4, 5} :=
by
  sorry

end complement_intersection_l215_215185


namespace line_equation_l215_215672

theorem line_equation 
    (passes_through_intersection : ∃ (P : ℝ × ℝ), P ∈ { (x, y) | 11 * x + 3 * y - 7 = 0 } ∧ P ∈ { (x, y) | 12 * x + y - 19 = 0 })
    (equidistant_from_A_and_B : ∃ (P : ℝ × ℝ), dist P (3, -2) = dist P (-1, 6)) :
    ∃ (a b c : ℝ), (a = 7 ∧ b = 1 ∧ c = -9) ∨ (a = 2 ∧ b = 1 ∧ c = 1) ∧ ∀ (x y : ℝ), a * x + b * y + c = 0 := 
sorry

end line_equation_l215_215672


namespace find_k_l215_215345

theorem find_k (x y k : ℤ) (h₁ : x = -3) (h₂ : y = 2) (h₃ : 2 * x + k * y = 6) : k = 6 :=
by
  rw [h₁, h₂] at h₃
  -- Substitute x and y in the equation
  -- 2 * (-3) + k * 2 = 6
  sorry

end find_k_l215_215345


namespace min_days_to_plant_trees_l215_215850

theorem min_days_to_plant_trees (n : ℕ) (h : 2 ≤ n) :
  (2 ^ (n + 1) - 2 ≥ 1000) ↔ (n ≥ 9) :=
by sorry

end min_days_to_plant_trees_l215_215850


namespace green_peaches_more_than_red_l215_215671

theorem green_peaches_more_than_red :
  let red_peaches := 5
  let green_peaches := 11
  (green_peaches - red_peaches) = 6 := by
  sorry

end green_peaches_more_than_red_l215_215671


namespace roots_real_and_equal_l215_215857

theorem roots_real_and_equal (a b c : ℝ) (h_eq : a = 1) (h_b : b = -4 * Real.sqrt 2) (h_c : c = 8) :
  ∃ x : ℝ, (a * x^2 + b * x + c = 0) ∧ (b^2 - 4 * a * c = 0) :=
by
  have h_a : a = 1 := h_eq;
  have h_b : b = -4 * Real.sqrt 2 := h_b;
  have h_c : c = 8 := h_c;
  sorry

end roots_real_and_equal_l215_215857


namespace seventh_term_of_geometric_sequence_l215_215267

theorem seventh_term_of_geometric_sequence (r : ℝ) 
  (h1 : 3 * r^5 = 729) : 3 * r^6 = 2187 :=
sorry

end seventh_term_of_geometric_sequence_l215_215267


namespace f_1996x_l215_215785

noncomputable def f : ℝ → ℝ := sorry

axiom f_equation (x y : ℝ) : f (x^3 + y^3) = (x + y) * ((f x)^2 - f x * f y + (f y)^2)

theorem f_1996x (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end f_1996x_l215_215785


namespace product_of_integers_with_cubes_sum_189_l215_215734

theorem product_of_integers_with_cubes_sum_189 :
  ∃ a b : ℤ, a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  -- The proof is omitted for brevity.
  sorry

end product_of_integers_with_cubes_sum_189_l215_215734


namespace arithmetic_sequence_sum_l215_215901

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)
variable (d : α)

-- Condition definitions
def is_arithmetic_sequence (a : ℕ → α) (d : α) : Prop :=
  ∀ (n : ℕ), a (n + 1) - a n = d

def sum_condition (a : ℕ → α) : Prop :=
  a 2 + a 5 + a 8 = 39

-- The goal statement to prove
theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_sum : sum_condition a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 117 :=
  sorry

end arithmetic_sequence_sum_l215_215901


namespace joan_total_seashells_l215_215877

def seashells_given_to_Sam : ℕ := 43
def seashells_left_with_Joan : ℕ := 27
def total_seashells_found := seashells_given_to_Sam + seashells_left_with_Joan

theorem joan_total_seashells : total_seashells_found = 70 := by
  -- proof goes here, but for now we will use sorry
  sorry

end joan_total_seashells_l215_215877


namespace find_a_l215_215531

theorem find_a (a : ℝ) : 
  (∃ r : ℕ, (10 - 3 * r = 1 ∧ (-a)^r * (Nat.choose 5 r) *  x^(10 - 2 * r - r) = x ∧ -10 = (-a)^3 * (Nat.choose 5 3)))
  → a = 1 :=
sorry

end find_a_l215_215531


namespace sequence_general_formula_l215_215254

theorem sequence_general_formula (a : ℕ → ℕ) (h1 : a 1 = 12)
  (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + 2 * n) :
  ∀ n : ℕ, n ≥ 1 → a n = n^2 - n + 12 :=
sorry

end sequence_general_formula_l215_215254


namespace jerry_initial_candy_l215_215588

theorem jerry_initial_candy
  (total_bags : ℕ)
  (chocolate_hearts_bags : ℕ)
  (chocolate_kisses_bags : ℕ)
  (nonchocolate_pieces : ℕ)
  (remaining_bags : ℕ)
  (pieces_per_bag : ℕ)
  (initial_candy : ℕ)
  (h_total_bags : total_bags = 9)
  (h_chocolate_hearts_bags : chocolate_hearts_bags = 2)
  (h_chocolate_kisses_bags : chocolate_kisses_bags = 3)
  (h_nonchocolate_pieces : nonchocolate_pieces = 28)
  (h_remaining_bags : remaining_bags = total_bags - chocolate_hearts_bags - chocolate_kisses_bags)
  (h_pieces_per_bag : pieces_per_bag = nonchocolate_pieces / remaining_bags)
  (h_initial_candy : initial_candy = total_bags * pieces_per_bag) :
  initial_candy = 63 := by
  sorry

end jerry_initial_candy_l215_215588


namespace inequality_solution_set_range_of_a_l215_215478

def f (x : ℝ) : ℝ := abs (3*x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x < 4 - abs (x - 1) } = { x : ℝ | -5/4 < x ∧ x < 1/2 } :=
by 
  sorry

theorem range_of_a (a : ℝ) (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) 
  (h4 : ∀ x : ℝ, abs (x - a) - f x ≤ 1 / m + 1 / n) : 
  0 < a ∧ a ≤ 10/3 :=
by 
  sorry

end inequality_solution_set_range_of_a_l215_215478


namespace lindy_total_distance_traveled_l215_215307

theorem lindy_total_distance_traveled 
    (initial_distance : ℕ)
    (jack_speed : ℕ)
    (christina_speed : ℕ)
    (lindy_speed : ℕ) 
    (meet_time : ℕ)
    (distance : ℕ) :
    initial_distance = 150 →
    jack_speed = 7 →
    christina_speed = 8 →
    lindy_speed = 10 →
    meet_time = initial_distance / (jack_speed + christina_speed) →
    distance = lindy_speed * meet_time →
    distance = 100 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lindy_total_distance_traveled_l215_215307


namespace Sally_seashells_l215_215081

/- Definitions -/
def Tom_seashells : Nat := 7
def Jessica_seashells : Nat := 5
def total_seashells : Nat := 21

/- Theorem statement -/
theorem Sally_seashells : total_seashells - (Tom_seashells + Jessica_seashells) = 9 := by
  -- Definitions of seashells found by Tom, Jessica and the total should be used here
  -- Proving the theorem
  sorry

end Sally_seashells_l215_215081


namespace num_possible_radii_l215_215460

theorem num_possible_radii :
  ∃ S : Finset ℕ, S.card = 11 ∧ ∀ r ∈ S, (∃ k : ℕ, 150 = k * r) ∧ r ≠ 150 :=
by
  sorry

end num_possible_radii_l215_215460


namespace washing_machines_removed_correct_l215_215042

-- Define the conditions
def crates : ℕ := 10
def boxes_per_crate : ℕ := 6
def washing_machines_per_box : ℕ := 4
def washing_machines_removed_per_box : ℕ := 1

-- Define the initial and final states
def initial_washing_machines_in_crate : ℕ := boxes_per_crate * washing_machines_per_box
def initial_washing_machines_in_container : ℕ := crates * initial_washing_machines_in_crate

def final_washing_machines_in_box : ℕ := washing_machines_per_box - washing_machines_removed_per_box
def final_washing_machines_in_crate : ℕ := boxes_per_crate * final_washing_machines_in_box
def final_washing_machines_in_container : ℕ := crates * final_washing_machines_in_crate

-- Number of washing machines removed
def washing_machines_removed : ℕ := initial_washing_machines_in_container - final_washing_machines_in_container

-- Theorem statement in Lean 4
theorem washing_machines_removed_correct : washing_machines_removed = 60 := by
  sorry

end washing_machines_removed_correct_l215_215042


namespace probability_jack_queen_king_l215_215538

theorem probability_jack_queen_king :
  let deck_size := 52
  let jacks := 4
  let queens := 4
  let kings := 4
  let remaining_after_jack := deck_size - 1
  let remaining_after_queen := deck_size - 2
  (jacks / deck_size) * (queens / remaining_after_jack) * (kings / remaining_after_queen) = 8 / 16575 :=
by
  sorry

end probability_jack_queen_king_l215_215538


namespace integral_sqrt_a_squared_minus_x_squared_l215_215917

open Real

theorem integral_sqrt_a_squared_minus_x_squared (a : ℝ) :
  (∫ x in -a..a, sqrt (a^2 - x^2)) = 1/2 * π * a^2 :=
by
  sorry

end integral_sqrt_a_squared_minus_x_squared_l215_215917


namespace subtract_fractions_l215_215488

theorem subtract_fractions (p q : ℚ) (h₁ : 4 / p = 8) (h₂ : 4 / q = 18) : p - q = 5 / 18 := 
by 
  sorry

end subtract_fractions_l215_215488


namespace calculate_expression_l215_215396

theorem calculate_expression :
  (16^16 * 8^8) / 4^32 = 16777216 := by
  sorry

end calculate_expression_l215_215396


namespace sock_pair_selection_l215_215543

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 5
def num_blue_socks : Nat := 3

def white_odd_positions : List Nat := [1, 3, 5]
def white_even_positions : List Nat := [2, 4]

def brown_odd_positions : List Nat := [1, 3, 5]
def brown_even_positions : List Nat := [2, 4]

def blue_odd_positions : List Nat := [1, 3]
def blue_even_positions : List Nat := [2]

noncomputable def count_pairs : Nat :=
  let white_brown := (white_odd_positions.length * brown_odd_positions.length) +
                     (white_even_positions.length * brown_even_positions.length)
  
  let brown_blue := (brown_odd_positions.length * blue_odd_positions.length) +
                    (brown_even_positions.length * blue_even_positions.length)

  let white_blue := (white_odd_positions.length * blue_odd_positions.length) +
                    (white_even_positions.length * blue_even_positions.length)

  white_brown + brown_blue + white_blue

theorem sock_pair_selection :
  count_pairs = 29 :=
by
  sorry

end sock_pair_selection_l215_215543


namespace base_edge_length_l215_215965

theorem base_edge_length (x : ℕ) :
  (∃ (x : ℕ), 
    (∀ (sum_edges : ℕ), sum_edges = 6 * x + 48 → sum_edges = 120) →
    x = 12) := 
sorry

end base_edge_length_l215_215965


namespace perimeter_of_plot_l215_215196

variable (length breadth : ℝ)
variable (h_ratio : length / breadth = 7 / 5)
variable (h_area : length * breadth = 5040)

theorem perimeter_of_plot (h_ratio : length / breadth = 7 / 5) (h_area : length * breadth = 5040) : 
  (2 * length + 2 * breadth = 288) :=
sorry

end perimeter_of_plot_l215_215196


namespace gambler_initial_games_l215_215224

theorem gambler_initial_games (x : ℕ)
  (h1 : ∀ x, ∃ (wins : ℝ), wins = 0.40 * x) 
  (h2 : ∀ x, ∃ (total_games : ℕ), total_games = x + 30)
  (h3 : ∀ x, ∃ (total_wins : ℝ), total_wins = 0.40 * x + 24)
  (h4 : ∀ x, ∃ (final_win_rate : ℝ), final_win_rate = (0.40 * x + 24) / (x + 30))
  (h5 : ∃ (final_win_rate_target : ℝ), final_win_rate_target = 0.60) :
  x = 30 :=
by
  sorry

end gambler_initial_games_l215_215224


namespace temperature_notation_l215_215398

-- Define what it means to denote temperatures in degrees Celsius
def denote_temperature (t : ℤ) : String :=
  if t < 0 then "-" ++ toString t ++ "°C"
  else if t > 0 then "+" ++ toString t ++ "°C"
  else toString t ++ "°C"

-- Theorem statement
theorem temperature_notation (t : ℤ) (ht : t = 2) : denote_temperature t = "+2°C" :=
by
  -- Proof goes here
  sorry

end temperature_notation_l215_215398


namespace num_broadcasting_methods_l215_215061

theorem num_broadcasting_methods : 
  let n := 6
  let commercials := 4
  let public_services := 2
  (public_services * commercials!) = 48 :=
by
  let n := 6
  let commercials := 4
  let public_services := 2
  have total_methods : (public_services * commercials!) = 48 := sorry
  exact total_methods

end num_broadcasting_methods_l215_215061


namespace multiplication_simplify_l215_215927

theorem multiplication_simplify :
  12 * (1 / 8) * 32 = 48 := 
sorry

end multiplication_simplify_l215_215927


namespace age_difference_l215_215818

theorem age_difference (x y : ℕ) (h1 : 3 * x + 4 * x = 42) (h2 : 18 - y = (24 - y) / 2) : 
  y = 12 :=
  sorry

end age_difference_l215_215818


namespace log_diff_eq_35_l215_215920

theorem log_diff_eq_35 {a b : ℝ} (h₁ : a > b) (h₂ : b > 1)
  (h₃ : (1 / Real.log a / Real.log b) + (1 / (Real.log b / Real.log a)) = Real.sqrt 1229) :
  (1 / (Real.log b / Real.log (a * b))) - (1 / (Real.log a / Real.log (a * b))) = 35 :=
sorry

end log_diff_eq_35_l215_215920


namespace carol_to_cathy_ratio_l215_215021

-- Define the number of cars owned by Cathy, Lindsey, Carol, and Susan
def cathy_cars : ℕ := 5
def lindsey_cars : ℕ := cathy_cars + 4
def carol_cars : ℕ := cathy_cars
def susan_cars : ℕ := carol_cars - 2

-- Define the total number of cars in the problem statement
def total_cars : ℕ := 32

-- Theorem to prove the ratio of Carol's cars to Cathy's cars is 1:1
theorem carol_to_cathy_ratio : carol_cars = cathy_cars := by
  sorry

end carol_to_cathy_ratio_l215_215021


namespace train_length_l215_215563

theorem train_length (L : ℝ) 
  (equal_length : ∀ (A B : ℝ), A = B → L = A)
  (same_direction : ∀ (dir1 dir2 : ℤ), dir1 = 1 → dir2 = 1)
  (speed_faster : ℝ := 50) (speed_slower : ℝ := 36)
  (time_to_pass : ℝ := 36)
  (relative_speed := speed_faster - speed_slower)
  (relative_speed_km_per_sec := relative_speed / 3600)
  (distance_covered := relative_speed_km_per_sec * time_to_pass)
  (total_distance := distance_covered)
  (length_per_train := total_distance / 2)
  (length_in_meters := length_per_train * 1000): 
  L = 70 := 
by 
  sorry

end train_length_l215_215563


namespace machine_A_sprockets_per_hour_l215_215786

-- Definitions based on the problem conditions
def MachineP_time (A : ℝ) (T : ℝ) : ℝ := T + 10
def MachineQ_rate (A : ℝ) : ℝ := 1.1 * A
def MachineP_sprockets (A : ℝ) (T : ℝ) : ℝ := A * (T + 10)
def MachineQ_sprockets (A : ℝ) (T : ℝ) : ℝ := 1.1 * A * T

-- Lean proof statement to prove that Machine A produces 8 sprockets per hour
theorem machine_A_sprockets_per_hour :
  ∀ A T : ℝ, 
  880 = MachineP_sprockets A T ∧
  880 = MachineQ_sprockets A T →
  A = 8 :=
by
  intros A T h
  have h1 : 880 = MachineP_sprockets A T := h.left
  have h2 : 880 = MachineQ_sprockets A T := h.right
  sorry

end machine_A_sprockets_per_hour_l215_215786


namespace max_integer_inequality_l215_215304

theorem max_integer_inequality (a b c: ℝ) (h₀: 0 < a) (h₁: 0 < b) (h₂: 0 < c) :
  (a^2 / (b / 29 + c / 31) + b^2 / (c / 29 + a / 31) + c^2 / (a / 29 + b / 31)) ≥ 14 * (a + b + c) :=
sorry

end max_integer_inequality_l215_215304


namespace terry_spent_total_l215_215714

def total_amount_spent (monday_spent tuesday_spent wednesday_spent : ℕ) : ℕ := 
  monday_spent + tuesday_spent + wednesday_spent

theorem terry_spent_total 
  (monday_spent : ℕ)
  (hmonday : monday_spent = 6)
  (tuesday_spent : ℕ)
  (htuesday : tuesday_spent = 2 * monday_spent)
  (wednesday_spent : ℕ)
  (hwednesday : wednesday_spent = 2 * (monday_spent + tuesday_spent)) :
  total_amount_spent monday_spent tuesday_spent wednesday_spent = 54 :=
by
  sorry

end terry_spent_total_l215_215714


namespace find_b_l215_215881

theorem find_b (a b : ℕ) (h1 : (a + b) % 10 = 5) (h2 : (a + b) % 7 = 4) : b = 2 := 
sorry

end find_b_l215_215881


namespace zero_squared_sum_l215_215918

theorem zero_squared_sum (a b : ℝ) (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 := 
by 
  sorry

end zero_squared_sum_l215_215918


namespace find_a_l215_215413

theorem find_a (a : ℝ) :
  (∃ x y : ℝ, x - 2 * y + 1 = 0 ∧ x + 3 * y - 1 = 0 ∧ ¬(∀ x y : ℝ, ax + 2 * y - 3 = 0)) →
  (∃ p q : ℝ, ax + 2 * q - 3 = 0 ∧ (a = -1 ∨ a = 2 / 3)) :=
by {
  sorry
}

end find_a_l215_215413


namespace is_divisible_by_six_l215_215725

/-- A stingy knight keeps gold coins in six chests. Given that he can evenly distribute the coins by opening any
two chests, any three chests, any four chests, or any five chests, prove that the total number of coins can be 
evenly distributed among all six chests. -/
theorem is_divisible_by_six (n : ℕ) 
  (h2 : ∀ (a b : ℕ), a + b = n → (a % 2 = 0 ∧ b % 2 = 0))
  (h3 : ∀ (a b c : ℕ), a + b + c = n → (a % 3 = 0 ∧ b % 3 = 0 ∧ c % 3 = 0)) 
  (h4 : ∀ (a b c d : ℕ), a + b + c + d = n → (a % 4 = 0 ∧ b % 4 = 0 ∧ c % 4 = 0 ∧ d % 4 = 0))
  (h5 : ∀ (a b c d e : ℕ), a + b + c + d + e = n → (a % 5 = 0 ∧ b % 5 = 0 ∧ c % 5 = 0 ∧ d % 5 = 0 ∧ e % 5 = 0)) :
  n % 6 = 0 :=
sorry

end is_divisible_by_six_l215_215725


namespace twelve_pow_six_mod_eight_l215_215060

theorem twelve_pow_six_mod_eight : ∃ m : ℕ, 0 ≤ m ∧ m < 8 ∧ 12^6 % 8 = m ∧ m = 0 := by
  sorry

end twelve_pow_six_mod_eight_l215_215060


namespace find_x_values_l215_215146

theorem find_x_values (x : ℝ) :
  (2 / (x + 2) + 8 / (x + 4) ≥ 2) ↔ (x ∈ Set.Ici 2 ∨ x ∈ Set.Iic (-4)) := by
sorry

end find_x_values_l215_215146


namespace eq_or_sum_zero_l215_215445

variables (a b c d : ℝ)

theorem eq_or_sum_zero (h : (3 * a + 2 * b) / (2 * b + 4 * c) = (4 * c + 3 * d) / (3 * d + 3 * a)) :
  3 * a = 4 * c ∨ 3 * a + 3 * d + 2 * b + 4 * c = 0 :=
by sorry

end eq_or_sum_zero_l215_215445


namespace average_score_girls_cedar_drake_l215_215171

theorem average_score_girls_cedar_drake
  (C c D d : ℕ)
  (cedar_boys_score cedar_girls_score cedar_combined_score
   drake_boys_score drake_girls_score drake_combined_score combined_boys_score : ℝ)
  (h1 : cedar_boys_score = 68)
  (h2 : cedar_girls_score = 80)
  (h3 : cedar_combined_score = 73)
  (h4 : drake_boys_score = 75)
  (h5 : drake_girls_score = 88)
  (h6 : drake_combined_score = 83)
  (h7 : combined_boys_score = 74)
  (h8 : (68 * C + 80 * c) / (C + c) = 73)
  (h9 : (75 * D + 88 * d) / (D + d) = 83)
  (h10 : (68 * C + 75 * D) / (C + D) = 74) :
  (80 * c + 88 * d) / (c + d) = 87 :=
by
  -- proof is omitted
  sorry

end average_score_girls_cedar_drake_l215_215171


namespace find_m_l215_215334

theorem find_m (x m : ℤ) (h : x = -1 ∧ x - 2 * m = 9) : m = -5 :=
sorry

end find_m_l215_215334


namespace card_probability_l215_215096

theorem card_probability :
  let total_cards := 52
  let hearts := 13
  let clubs := 13
  let spades := 13
  let prob_heart_first := hearts / total_cards
  let remaining_after_heart := total_cards - 1
  let prob_club_second := clubs / remaining_after_heart
  let remaining_after_heart_and_club := remaining_after_heart - 1
  let prob_spade_third := spades / remaining_after_heart_and_club
  (prob_heart_first * prob_club_second * prob_spade_third) = (2197 / 132600) :=
  sorry

end card_probability_l215_215096


namespace smallest_z_value_l215_215145

theorem smallest_z_value :
  ∃ (x z : ℕ), (w = x - 2) ∧ (y = x + 2) ∧ (z = x + 4) ∧ ((x - 2)^3 + x^3 + (x + 2)^3 = (x + 4)^3) ∧ z = 2 := by
  sorry

end smallest_z_value_l215_215145


namespace max_result_of_operation_l215_215628

theorem max_result_of_operation : ∃ n : ℕ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 → 3 * (300 - m) ≤ 870) ∧ 3 * (300 - n) = 870 :=
by
  sorry

end max_result_of_operation_l215_215628


namespace length_of_one_side_of_hexagon_l215_215148

variable (P : ℝ) (n : ℕ)
-- Condition: perimeter P is 60 inches
def hexagon_perimeter_condition : Prop := P = 60
-- Hexagon has six sides
def hexagon_sides_condition : Prop := n = 6
-- The question asks for the side length
noncomputable def side_length_of_hexagon : ℝ := P / n

-- Prove that if a hexagon has a perimeter of 60 inches, then its side length is 10 inches
theorem length_of_one_side_of_hexagon (hP : hexagon_perimeter_condition P) (hn : hexagon_sides_condition n) :
  side_length_of_hexagon P n = 10 := by
  sorry

end length_of_one_side_of_hexagon_l215_215148


namespace number_of_impossible_d_l215_215448

-- Define the problem parameters and conditions
def perimeter_diff (t s : ℕ) : ℕ := 3 * t - 4 * s
def side_diff (t s d : ℕ) : ℕ := t - s - d
def square_perimeter_positive (s : ℕ) : Prop := s > 0

-- Define the proof problem
theorem number_of_impossible_d (t s d : ℕ) (h1 : perimeter_diff t s = 1575) (h2 : side_diff t s d = 0) (h3 : square_perimeter_positive s) : 
    ∃ n, n = 525 ∧ ∀ d, d ≤ 525 → ¬ (3 * d > 1575) :=
    sorry

end number_of_impossible_d_l215_215448


namespace right_triangle_area_l215_215977

theorem right_triangle_area (a b : ℝ) (H₁ : a = 3) (H₂ : b = 5) : 
  1 / 2 * a * b = 7.5 := by
  rw [H₁, H₂]
  norm_num

end right_triangle_area_l215_215977


namespace find_f_at_4_l215_215220

noncomputable def f : ℝ → ℝ := sorry -- We assume such a function exists

theorem find_f_at_4:
  (∀ x : ℝ, f (4^x) + x * f (4^(-x)) = 3) → f (4) = 0 := by
  intro h
  -- Proof would go here, but is omitted as per instructions
  sorry

end find_f_at_4_l215_215220


namespace number_of_possible_values_l215_215291

theorem number_of_possible_values (b : ℕ) (hb4 : 4 ∣ b) (hb24 : b ∣ 24) (hpos : 0 < b) : ∃ n, n = 4 :=
by
  sorry

end number_of_possible_values_l215_215291


namespace initial_kittens_count_l215_215194

-- Let's define the initial conditions first.
def kittens_given_away : ℕ := 2
def kittens_remaining : ℕ := 6

-- The main theorem to prove the initial number of kittens.
theorem initial_kittens_count : (kittens_given_away + kittens_remaining) = 8 :=
by sorry

end initial_kittens_count_l215_215194


namespace volume_parallelepiped_eq_20_l215_215343

theorem volume_parallelepiped_eq_20 (k : ℝ) (h : k > 0) (hvol : abs (3 * k^2 - 7 * k - 6) = 20) :
  k = 13 / 3 :=
sorry

end volume_parallelepiped_eq_20_l215_215343


namespace max_ratio_BO_BM_l215_215160

theorem max_ratio_BO_BM
  (C : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hC : C = (0, -4))
  (hCir : ∃ (P : ℝ × ℝ), (P.1 - 2)^2 + (P.2 - 4)^2 = 1 ∧ A = ((P.1 + C.1) / 2, (P.2 + C.2) / 2))
  (hPar : ∃ (x y : ℝ), B = (x, y) ∧ y^2 = 4 * x) :
  ∃ t, t = (4 * Real.sqrt 7)/7 ∧ t = Real.sqrt ((B.1^2 + 4 * B.1)/((B.1 + 1/2)^2)) := by
  -- Given conditions and definitions
  obtain ⟨P, hP, hA⟩ := hCir
  obtain ⟨x, y, hB⟩ := hPar
  use (4 * Real.sqrt 7) / 7
  sorry

end max_ratio_BO_BM_l215_215160


namespace neg_or_false_implies_or_true_l215_215067

theorem neg_or_false_implies_or_true (p q : Prop) (h : ¬(p ∨ q) = False) : p ∨ q :=
by {
  sorry
}

end neg_or_false_implies_or_true_l215_215067


namespace complement_subset_lemma_l215_215774

-- Definitions for sets P and Q
def P : Set ℝ := {x | 0 < x ∧ x < 1}

def Q : Set ℝ := {x | x^2 + x - 2 ≤ 0}

-- Definition for complement of a set
def C_ℝ (A : Set ℝ) : Set ℝ := {x | ¬(x ∈ A)}

-- Prove the required relationship
theorem complement_subset_lemma : C_ℝ Q ⊆ C_ℝ P :=
by
  -- The proof steps will go here
  sorry

end complement_subset_lemma_l215_215774


namespace distance_focus_to_asymptote_of_hyperbola_l215_215564

open Real

noncomputable def distance_from_focus_to_asymptote_of_hyperbola : ℝ :=
  let a := 2
  let b := 1
  let c := sqrt (a^2 + b^2)
  let foci1 := (sqrt (a^2 + b^2), 0)
  let foci2 := (-sqrt (a^2 + b^2), 0)
  let asymptote_slope := a / b
  let distance_formula := (|abs (sqrt 5)|) / (sqrt (1 + asymptote_slope^2))
  distance_formula

theorem distance_focus_to_asymptote_of_hyperbola :
  distance_from_focus_to_asymptote_of_hyperbola = 1 :=
sorry

end distance_focus_to_asymptote_of_hyperbola_l215_215564


namespace area_of_triangle_l215_215373

theorem area_of_triangle :
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  triangle_area = 34 :=
by {
  -- Definitions
  let A := (1, -3)
  let B := (9, 2)
  let C := (5, 8)
  let v := (A.1 - C.1, A.2 - C.2)
  let w := (B.1 - C.1, B.2 - C.2)
  let parallelogram_area := abs ((v.1 * w.2) - (v.2 * w.1))
  let triangle_area := parallelogram_area / 2
  -- Proof (normally written here, but omitted with 'sorry')
  sorry
}

end area_of_triangle_l215_215373


namespace chairs_problem_l215_215553

theorem chairs_problem (B G W : ℕ) 
  (h1 : G = 3 * B) 
  (h2 : W = B + G - 13) 
  (h3 : B + G + W = 67) : 
  B = 10 :=
by
  sorry

end chairs_problem_l215_215553


namespace clock_hands_overlap_24_hours_l215_215957

theorem clock_hands_overlap_24_hours : 
  (∀ t : ℕ, t < 12 →  ∃ n : ℕ, (n = 11 ∧ (∃ h m : ℕ, h * 60 + m = t * 60 + m))) →
  (∃ k : ℕ, k = 22) :=
by
  sorry

end clock_hands_overlap_24_hours_l215_215957


namespace product_of_solutions_eq_neg_ten_l215_215418

theorem product_of_solutions_eq_neg_ten :
  (∃ x₁ x₂, -20 = -2 * x₁^2 - 6 * x₁ ∧ -20 = -2 * x₂^2 - 6 * x₂ ∧ x₁ ≠ x₂ ∧ x₁ * x₂ = -10) :=
by
  sorry

end product_of_solutions_eq_neg_ten_l215_215418


namespace find_f1_plus_g1_l215_215712

variables (f g : ℝ → ℝ)

-- Conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = -g x
def function_equation (f g : ℝ → ℝ) : Prop := ∀ x : ℝ, f x - g x = x^3 - 2*x^2 + 1

theorem find_f1_plus_g1 
  (hf : even_function f)
  (hg : odd_function g)
  (hfg : function_equation f g):
  f 1 + g 1 = -2 :=
by {
  sorry
}

end find_f1_plus_g1_l215_215712


namespace when_to_sell_goods_l215_215184

variable (a : ℝ) (currentMonthProfit nextMonthProfitWithStorage : ℝ) 
          (interestRate storageFee thisMonthProfit nextMonthProfit : ℝ)
          (hm1 : interestRate = 0.005)
          (hm2 : storageFee = 5)
          (hm3 : thisMonthProfit = 100)
          (hm4 : nextMonthProfit = 120)
          (hm5 : currentMonthProfit = thisMonthProfit + (a + thisMonthProfit) * interestRate)
          (hm6 : nextMonthProfitWithStorage = nextMonthProfit - storageFee)

theorem when_to_sell_goods :
  (a > 2900 → currentMonthProfit > nextMonthProfitWithStorage) ∧
  (a = 2900 → currentMonthProfit = nextMonthProfitWithStorage) ∧
  (a < 2900 → currentMonthProfit < nextMonthProfitWithStorage) := by
  sorry

end when_to_sell_goods_l215_215184


namespace circle_rolling_start_point_l215_215779

theorem circle_rolling_start_point (x : ℝ) (h1 : ∃ x, (x + 2 * Real.pi = -1) ∨ (x - 2 * Real.pi = -1)) :
  x = -1 - 2 * Real.pi ∨ x = -1 + 2 * Real.pi :=
by
  sorry

end circle_rolling_start_point_l215_215779


namespace point_Q_in_first_quadrant_l215_215041

theorem point_Q_in_first_quadrant (a b : ℝ) (h : a < 0 ∧ b < 0) : (0 < -a) ∧ (0 < -b) :=
by
  have ha : -a > 0 := by linarith
  have hb : -b > 0 := by linarith
  exact ⟨ha, hb⟩

end point_Q_in_first_quadrant_l215_215041


namespace train_cross_time_l215_215355

noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 255.03
noncomputable def train_speed_ms : ℝ := 12.5
noncomputable def distance_to_travel : ℝ := train_length + bridge_length
noncomputable def expected_time : ℝ := 30.0024

theorem train_cross_time :
  (distance_to_travel / train_speed_ms) = expected_time :=
by sorry

end train_cross_time_l215_215355


namespace susan_ate_candies_l215_215789

theorem susan_ate_candies (candies_tuesday candies_thursday candies_friday candies_left : ℕ) 
  (h_tuesday : candies_tuesday = 3) 
  (h_thursday : candies_thursday = 5) 
  (h_friday : candies_friday = 2) 
  (h_left : candies_left = 4) : candies_tuesday + candies_thursday + candies_friday - candies_left = 6 := by
  sorry

end susan_ate_candies_l215_215789


namespace probability_sqrt_lt_7_of_random_two_digit_number_l215_215794

theorem probability_sqrt_lt_7_of_random_two_digit_number : 
  (∃ p : ℚ, (∀ n, 10 ≤ n ∧ n ≤ 99 → n < 49 → ∃ k, k = p) ∧ p = 13 / 30) := 
by
  sorry

end probability_sqrt_lt_7_of_random_two_digit_number_l215_215794


namespace total_buttons_l215_215344

-- Defining the given conditions
def green_buttons : ℕ := 90
def yellow_buttons : ℕ := green_buttons + 10
def blue_buttons : ℕ := green_buttons - 5

-- Stating the theorem to prove the total number of buttons
theorem total_buttons : green_buttons + yellow_buttons + blue_buttons = 275 :=
by 
  sorry

end total_buttons_l215_215344


namespace total_items_in_jar_l215_215662

/--
A jar contains 3409.0 pieces of candy and 145.0 secret eggs with a prize.
We aim to prove that the total number of items in the jar is 3554.0.
-/
theorem total_items_in_jar :
  let number_of_pieces_of_candy := 3409.0
  let number_of_secret_eggs := 145.0
  number_of_pieces_of_candy + number_of_secret_eggs = 3554.0 :=
by
  sorry

end total_items_in_jar_l215_215662


namespace minimum_value_of_f_l215_215137

noncomputable def f (x : ℝ) : ℝ := 4 * x + 9 / x

theorem minimum_value_of_f : 
  (∀ (x : ℝ), x > 0 → f x ≥ 12) ∧ (∃ (x : ℝ), x > 0 ∧ f x = 12) :=
by {
  sorry
}

end minimum_value_of_f_l215_215137


namespace range_of_m_satisfying_obtuse_triangle_l215_215874

theorem range_of_m_satisfying_obtuse_triangle (m : ℝ) 
(h_triangle: m > 0 
  → m + (m + 1) > (m + 2) 
  ∧ m + (m + 2) > (m + 1) 
  ∧ (m + 1) + (m + 2) > m
  ∧ (m + 2) ^ 2 > m ^ 2 + (m + 1) ^ 2) : 1 < m ∧ m < 1.5 :=
by
  sorry

end range_of_m_satisfying_obtuse_triangle_l215_215874


namespace contradiction_method_l215_215022

variable (a b : ℝ)

theorem contradiction_method (h1 : a > b) (h2 : 3 * a ≤ 3 * b) : false :=
by sorry

end contradiction_method_l215_215022


namespace largest_multiple_of_7_neg_greater_than_neg_150_l215_215143

theorem largest_multiple_of_7_neg_greater_than_neg_150 : 
  ∃ (k : ℤ), k % 7 = 0 ∧ -k > -150 ∧ (∀ (m : ℤ), m % 7 = 0 ∧ -m > -150 → k ≥ m) ∧ k = 147 :=
by
  sorry

end largest_multiple_of_7_neg_greater_than_neg_150_l215_215143


namespace speed_limit_l215_215403

theorem speed_limit (x : ℝ) (h₀ : 0 < x) :
  (11 / (x + 1.5) + 8 / x ≥ 12 / (x + 2) + 2) → x ≤ 4 := 
sorry

end speed_limit_l215_215403


namespace max_integer_a_real_roots_l215_215719

theorem max_integer_a_real_roots :
  ∀ (a : ℤ), (∃ (x : ℝ), (a + 1 : ℝ) * x^2 - 2 * x + 3 = 0) → a ≤ -2 :=
by
  sorry

end max_integer_a_real_roots_l215_215719


namespace tangent_line_equation_at_1_range_of_a_l215_215320

noncomputable def f (x a : ℝ) : ℝ := (x+1) * Real.log x - a * (x-1)

-- (I) Tangent line equation when a = 4
theorem tangent_line_equation_at_1 (x : ℝ) (hx : x = 1) :
  let a := 4
  2*x + f 1 a - 2 = 0 :=
sorry

-- (II) Range of values for a
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_equation_at_1_range_of_a_l215_215320


namespace Sharmila_hourly_wage_l215_215767

def Sharmila_hours_per_day (day : String) : ℕ :=
  if day = "Monday" ∨ day = "Wednesday" ∨ day = "Friday" then 10
  else if day = "Tuesday" ∨ day = "Thursday" then 8
  else 0

def weekly_total_hours : ℕ :=
  Sharmila_hours_per_day "Monday" + Sharmila_hours_per_day "Tuesday" +
  Sharmila_hours_per_day "Wednesday" + Sharmila_hours_per_day "Thursday" +
  Sharmila_hours_per_day "Friday"

def weekly_earnings : ℤ := 460

def hourly_wage : ℚ :=
  weekly_earnings / weekly_total_hours

theorem Sharmila_hourly_wage :
  hourly_wage = (10 : ℚ) :=
by
  -- proof skipped
  sorry

end Sharmila_hourly_wage_l215_215767


namespace min_value_binom_l215_215591

theorem min_value_binom
  (a b : ℕ → ℕ)
  (n : ℕ) (hn : 0 < n)
  (h1 : ∀ n, a n = 2^n)
  (h2 : ∀ n, b n = 4^n) :
  ∃ n, 2^n + (1 / 2^n) = 5 / 2 :=
sorry

end min_value_binom_l215_215591


namespace angle_sum_l215_215593

theorem angle_sum (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (tan_α : Real.tan α = 3 / 4)
  (sin_β : Real.sin β = 3 / 5) :
  α + 3 * β = 5 * Real.pi / 4 := 
sorry

end angle_sum_l215_215593


namespace rectangle_area_change_l215_215357

theorem rectangle_area_change
  (L B : ℝ)
  (hL : L > 0)
  (hB : B > 0)
  (new_L : ℝ := 1.25 * L)
  (new_B : ℝ := 0.85 * B):
  (new_L * new_B = 1.0625 * (L * B)) :=
by
  sorry

end rectangle_area_change_l215_215357


namespace diameter_of_larger_circle_l215_215568

theorem diameter_of_larger_circle (R r D : ℝ) 
  (h1 : R^2 - r^2 = 25) 
  (h2 : D = 2 * R) : 
  D = Real.sqrt (100 + 4 * r^2) := 
by 
  sorry

end diameter_of_larger_circle_l215_215568


namespace translation_result_l215_215044

-- Define the initial point A
def A : (ℤ × ℤ) := (-2, 3)

-- Define the translation function
def translate (p : (ℤ × ℤ)) (delta_x delta_y : ℤ) : (ℤ × ℤ) :=
  (p.1 + delta_x, p.2 - delta_y)

-- The theorem stating the resulting point after translation
theorem translation_result :
  translate A 3 1 = (1, 2) :=
by
  -- Skipping proof with sorry
  sorry

end translation_result_l215_215044


namespace customers_in_other_countries_l215_215070

-- Given 
def total_customers : ℕ := 7422
def customers_in_us : ℕ := 723

-- To Prove
theorem customers_in_other_countries : (total_customers - customers_in_us) = 6699 := 
by
  sorry

end customers_in_other_countries_l215_215070


namespace units_digit_of_product_of_odds_between_10_and_50_l215_215686

def product_of_odds_units_digit : ℕ :=
  let odds := [11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49]
  let product := odds.foldl (· * ·) 1
  product % 10

theorem units_digit_of_product_of_odds_between_10_and_50 : product_of_odds_units_digit = 5 :=
  sorry

end units_digit_of_product_of_odds_between_10_and_50_l215_215686


namespace inverse_graph_pass_point_l215_215639

variable {f : ℝ → ℝ}
variable {f_inv : ℝ → ℝ}

noncomputable def satisfies_inverse (f f_inv : ℝ → ℝ) : Prop :=
∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

theorem inverse_graph_pass_point
  (hf : satisfies_inverse f f_inv)
  (h_point : (1 : ℝ) - f 1 = 3) :
  f_inv (-2) + 3 = 4 :=
by
  sorry

end inverse_graph_pass_point_l215_215639


namespace tan_value_l215_215170

open Real

theorem tan_value (α : ℝ) 
  (h1 : sin (α + π / 6) = -3 / 5)
  (h2 : -2 * π / 3 < α ∧ α < -π / 6) : 
  tan (4 * π / 3 - α) = -4 / 3 :=
sorry

end tan_value_l215_215170


namespace scale_drawing_represents_line_segment_l215_215545

-- Define the given conditions
def scale_factor : ℝ := 800
def line_segment_length_inch : ℝ := 4.75

-- Prove the length in feet
theorem scale_drawing_represents_line_segment :
  line_segment_length_inch * scale_factor = 3800 :=
by
  sorry

end scale_drawing_represents_line_segment_l215_215545


namespace slope_of_line_l215_215237

theorem slope_of_line (θ : ℝ) (h : θ = 30) :
  ∃ k, k = Real.tan (60 * (π / 180)) ∨ k = Real.tan (120 * (π / 180)) := by
    sorry

end slope_of_line_l215_215237


namespace part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l215_215983

def cost_option1 (x : ℕ) : ℕ :=
  20 * x + 1200

def cost_option2 (x : ℕ) : ℕ :=
  18 * x + 1440

theorem part1_option1_payment (x : ℕ) (h : x > 20) : cost_option1 x = 20 * x + 1200 :=
  by sorry

theorem part1_option2_payment (x : ℕ) (h : x > 20) : cost_option2 x = 18 * x + 1440 :=
  by sorry

theorem part2_cost_effective (x : ℕ) (h : x = 30) : cost_option1 x < cost_option2 x :=
  by sorry

theorem part3_more_cost_effective (x : ℕ) (h : x = 30) : 20 * 80 + 20 * 10 * 9 / 10 = 1780 :=
  by sorry

end part1_option1_payment_part1_option2_payment_part2_cost_effective_part3_more_cost_effective_l215_215983


namespace line_equation_45_deg_through_point_l215_215115

theorem line_equation_45_deg_through_point :
  ∀ (x y : ℝ), 
  (∃ m k: ℝ, m = 1 ∧ k = 5 ∧ y = m * x + k) ∧ (∃ p q : ℝ, p = -2 ∧ q = 3 ∧ y = q ) :=  
  sorry

end line_equation_45_deg_through_point_l215_215115


namespace smallest_n_conditions_l215_215221

theorem smallest_n_conditions :
  ∃ n : ℕ, 0 < n ∧ (∃ k1 : ℕ, 2 * n = k1^2) ∧ (∃ k2 : ℕ, 3 * n = k2^4) ∧ n = 54 :=
by
  sorry

end smallest_n_conditions_l215_215221


namespace find_z_value_l215_215606

theorem find_z_value (z w : ℝ) (hz : z ≠ 0) (hw : w ≠ 0)
  (h1 : z + 1/w = 15) (h2 : w^2 + 1/z = 3) : z = 44/3 := 
by 
  sorry

end find_z_value_l215_215606


namespace range_of_a_minus_b_l215_215839

theorem range_of_a_minus_b {a b : ℝ} (h1 : -2 < a) (h2 : a < 3) (h3 : 1 < b) (h4 : b < 2) : 
  -4 < a - b ∧ a - b < 2 := 
by
  sorry

end range_of_a_minus_b_l215_215839


namespace total_journey_time_l215_215349

def distance_to_post_office : ℝ := 19.999999999999996
def speed_to_post_office : ℝ := 25
def speed_back : ℝ := 4

theorem total_journey_time : 
  (distance_to_post_office / speed_to_post_office) + (distance_to_post_office / speed_back) = 5.8 :=
by
  sorry

end total_journey_time_l215_215349


namespace maximum_smallest_triplet_sum_l215_215520

theorem maximum_smallest_triplet_sum (circle : Fin 10 → ℕ) (h : ∀ i : Fin 10, 1 ≤ circle i ∧ circle i ≤ 10 ∧ ∀ j k, j ≠ k → circle j ≠ circle k):
  ∃ (i : Fin 10), ∀ j ∈ ({i, i + 1, i + 2} : Finset (Fin 10)), circle i + circle (i + 1) + circle (i + 2) ≤ 15 :=
sorry

end maximum_smallest_triplet_sum_l215_215520


namespace regular_train_pass_time_l215_215810

-- Define the lengths of the trains
def high_speed_train_length : ℕ := 400
def regular_train_length : ℕ := 600

-- Define the observation time for the passenger on the high-speed train
def observation_time : ℕ := 3

-- Define the problem to find the time x for the regular train passenger
theorem regular_train_pass_time :
  ∃ (x : ℕ), (regular_train_length / observation_time) * x = high_speed_train_length :=
by 
  sorry

end regular_train_pass_time_l215_215810


namespace transformation_correct_l215_215337

theorem transformation_correct (a b c : ℝ) : a = b → ac = bc :=
by sorry

end transformation_correct_l215_215337


namespace odd_function_condition_l215_215430

noncomputable def f (x a b : ℝ) : ℝ := x * |x + a| + b

theorem odd_function_condition (a b : ℝ) :
  (∀ x : ℝ, f (-x) a b = -f x a b) ↔ a^2 + b^2 = 0 :=
by
  sorry

end odd_function_condition_l215_215430


namespace grandfather_age_l215_215615

variables (M G y z : ℕ)

-- Conditions
def condition1 : Prop := G = 6 * M
def condition2 : Prop := G + y = 5 * (M + y)
def condition3 : Prop := G + y + z = 4 * (M + y + z)

-- Theorem to prove Grandfather's current age is 72
theorem grandfather_age : 
  condition1 M G → 
  condition2 M G y → 
  condition3 M G y z → 
  G = 72 :=
by
  intros h1 h2 h3
  unfold condition1 at h1
  unfold condition2 at h2
  unfold condition3 at h3
  sorry

end grandfather_age_l215_215615


namespace pool_water_volume_after_evaporation_l215_215704

theorem pool_water_volume_after_evaporation :
  let initial_volume := 300
  let evaporation_first_15_days := 1 -- in gallons per day
  let evaporation_next_15_days := 2 -- in gallons per day
  initial_volume - (15 * evaporation_first_15_days + 15 * evaporation_next_15_days) = 255 :=
by
  sorry

end pool_water_volume_after_evaporation_l215_215704


namespace systematic_sampling_second_invoice_l215_215229

theorem systematic_sampling_second_invoice 
  (N : ℕ) 
  (valid_invoice : N ≥ 10)
  (first_invoice : Fin 10) :
  ¬ (∃ k : ℕ, k ≥ 1 ∧ first_invoice.1 + k * 10 = 23) := 
by 
  -- Proof omitted
  sorry

end systematic_sampling_second_invoice_l215_215229


namespace percent_shaded_of_square_l215_215252

theorem percent_shaded_of_square (side_len : ℤ) (first_layer_side : ℤ) 
(second_layer_outer_side : ℤ) (second_layer_inner_side : ℤ)
(third_layer_outer_side : ℤ) (third_layer_inner_side : ℤ)
(h_side : side_len = 7) (h_first : first_layer_side = 2) 
(h_second_outer : second_layer_outer_side = 5) (h_second_inner : second_layer_inner_side = 3) 
(h_third_outer : third_layer_outer_side = 7) (h_third_inner : third_layer_inner_side = 6) : 
  (4 + (25 - 9) + (49 - 36)) / (side_len * side_len : ℝ) = 33 / 49 :=
by
  -- Sorry is used as we are only required to construct the statement, not the proof.
  sorry

end percent_shaded_of_square_l215_215252


namespace sum_of_squares_five_consecutive_not_perfect_square_l215_215378

theorem sum_of_squares_five_consecutive_not_perfect_square 
  (x : ℤ) : ¬ ∃ k : ℤ, (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 = k^2 :=
by 
  sorry

end sum_of_squares_five_consecutive_not_perfect_square_l215_215378


namespace distance_between_points_l215_215611

open Real -- opening real number namespace

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * cos θ, r * sin θ)

theorem distance_between_points :
  let A := polar_to_cartesian 2 (π / 3)
  let B := polar_to_cartesian 2 (2 * π / 3)
  dist A B = 2 :=
by
  sorry

end distance_between_points_l215_215611


namespace y_value_l215_215944

theorem y_value (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + 1 / y = 8) (h4 : y + 1 / x = 7 / 12) (h5 : x + y = 7) : y = 49 / 103 :=
by
  sorry

end y_value_l215_215944


namespace progress_regress_ratio_l215_215314

theorem progress_regress_ratio :
  let progress_rate := 1.2
  let regress_rate := 0.8
  let log2 := 0.3010
  let log3 := 0.4771
  let target_ratio := 10000
  (progress_rate / regress_rate) ^ 23 = target_ratio :=
by
  sorry

end progress_regress_ratio_l215_215314


namespace find_k_l215_215208

variable {x y k : ℝ}

theorem find_k (h1 : 3 * x + 4 * y = k + 2) 
             (h2 : 2 * x + y = 4) 
             (h3 : x + y = 2) :
  k = 4 := 
by
  sorry

end find_k_l215_215208


namespace trig_identity_l215_215710

theorem trig_identity (x : ℝ) (h0 : -3 * Real.pi / 2 < x) (h1 : x < -Real.pi) (h2 : Real.tan x = -3) :
  Real.sin x * Real.cos x = -3 / 10 :=
sorry

end trig_identity_l215_215710


namespace find_original_number_l215_215429

theorem find_original_number (x : ℤ) (h : (x + 5) % 23 = 0) : x = 18 :=
sorry

end find_original_number_l215_215429


namespace find_ordered_pair_l215_215206

noncomputable def discriminant_eq_zero (a c : ℝ) : Prop :=
  a * c = 9

def sum_eq_14 (a c : ℝ) : Prop :=
  a + c = 14

def a_greater_than_c (a c : ℝ) : Prop :=
  a > c

theorem find_ordered_pair : 
  ∃ (a c : ℝ), 
    sum_eq_14 a c ∧ 
    discriminant_eq_zero a c ∧ 
    a_greater_than_c a c ∧ 
    a = 7 + 2 * Real.sqrt 10 ∧ 
    c = 7 - 2 * Real.sqrt 10 :=
by {
  sorry
}

end find_ordered_pair_l215_215206


namespace most_reasonable_sampling_method_l215_215205

-- Define the conditions
axiom significant_differences_in_educational_stages : Prop
axiom insignificant_differences_between_genders : Prop

-- Define the options
inductive SamplingMethod
| SimpleRandomSampling
| StratifiedSamplingByGender
| StratifiedSamplingByEducationalStage
| SystematicSampling

-- State the problem as a theorem
theorem most_reasonable_sampling_method
  (H1 : significant_differences_in_educational_stages)
  (H2 : insignificant_differences_between_genders) :
  SamplingMethod.StratifiedSamplingByEducationalStage = SamplingMethod.StratifiedSamplingByEducationalStage :=
by
  -- Proof is skipped
  sorry

end most_reasonable_sampling_method_l215_215205


namespace new_weights_inequality_l215_215542

theorem new_weights_inequality (W : ℝ) (x y : ℝ) (h_avg_increase : (8 * W - 2 * 68 + x + y) / 8 = W + 5.5)
  (h_sum_new_weights : x + y ≤ 180) : x > W ∧ y > W :=
by {
  sorry
}

end new_weights_inequality_l215_215542


namespace like_terms_exponent_l215_215643

theorem like_terms_exponent (a : ℝ) : (2 * a = a + 3) → a = 3 := 
by
  intros h
  -- Proof here
  sorry

end like_terms_exponent_l215_215643


namespace bacteria_growth_time_l215_215489

theorem bacteria_growth_time (initial_bacteria : ℕ) (final_bacteria : ℕ) (doubling_time : ℕ) :
  initial_bacteria = 1000 →
  final_bacteria = 128000 →
  doubling_time = 3 →
  (∃ t : ℕ, final_bacteria = initial_bacteria * 2 ^ (t / doubling_time) ∧ t = 21) :=
by
  sorry

end bacteria_growth_time_l215_215489


namespace total_get_well_cards_l215_215576

-- Definitions for the number of cards received in each place
def cardsInHospital : ℕ := 403
def cardsAtHome : ℕ := 287

-- Theorem statement:
theorem total_get_well_cards : cardsInHospital + cardsAtHome = 690 := by
  sorry

end total_get_well_cards_l215_215576


namespace minimum_value_of_f_l215_215009

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |3 - x|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 1 ∧ (∃ x₀ : ℝ, f x₀ = 1) := by
  sorry

end minimum_value_of_f_l215_215009


namespace certain_number_l215_215664

theorem certain_number (x : ℝ) (h : (2.28 * x) / 6 = 480.7) : x = 1265.0 := 
by 
  sorry

end certain_number_l215_215664


namespace negate_proposition_l215_215797

theorem negate_proposition : (∀ x : ℝ, x^3 - x^2 + 1 ≤ 1) ↔ ¬ (∃ x : ℝ, x^3 - x^2 + 1 > 1) :=
by
  sorry

end negate_proposition_l215_215797


namespace union_comm_union_assoc_inter_distrib_union_l215_215361

variables {α : Type*} (A B C : Set α)

theorem union_comm : A ∪ B = B ∪ A := sorry

theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry

theorem inter_distrib_union : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry

end union_comm_union_assoc_inter_distrib_union_l215_215361


namespace at_least_3_students_same_score_l215_215436

-- Conditions
def initial_points : ℕ := 6
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def num_questions : ℕ := 6
def num_students : ℕ := 51

-- Question
theorem at_least_3_students_same_score :
  ∃ score : ℤ, ∃ students_with_same_score : ℕ, students_with_same_score ≥ 3 :=
by
  sorry

end at_least_3_students_same_score_l215_215436


namespace radius_squared_of_intersection_circle_l215_215864

def parabola1 (x y : ℝ) := y = (x - 2) ^ 2
def parabola2 (x y : ℝ) := x + 6 = (y - 5) ^ 2

theorem radius_squared_of_intersection_circle
    (x y : ℝ)
    (h₁ : parabola1 x y)
    (h₂ : parabola2 x y) :
    ∃ r, r ^ 2 = 83 / 4 :=
sorry

end radius_squared_of_intersection_circle_l215_215864


namespace students_chose_water_l215_215121

theorem students_chose_water (total_students : ℕ)
  (h1 : 75 * total_students / 100 = 90)
  (h2 : 25 * total_students / 100 = x) :
  x = 30 := 
sorry

end students_chose_water_l215_215121


namespace min_buses_needed_l215_215863

theorem min_buses_needed (total_students : ℕ) (bus45_capacity : ℕ) (bus40_capacity : ℕ) : 
  total_students = 530 ∧ bus45_capacity = 45 ∧ bus40_capacity = 40 → 
  ∃ (n : ℕ), n = 12 :=
by 
  intro h
  obtain ⟨htotal, hbus45, hbus40⟩ := h
  -- Proof would go here...
  sorry

end min_buses_needed_l215_215863


namespace tan_three_halves_pi_sub_alpha_l215_215827

theorem tan_three_halves_pi_sub_alpha (α : ℝ) (h : Real.cos (π - α) = -3/5) :
    Real.tan (3 * π / 2 - α) = 3/4 ∨ Real.tan (3 * π / 2 - α) = -3/4 := by
  sorry

end tan_three_halves_pi_sub_alpha_l215_215827


namespace tenth_pair_in_twentieth_row_l215_215633

def nth_pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if h : n > 0 ∧ k > 0 ∧ n >= k then (k, n + 1 - k)
  else (0, 0) -- define (0,0) as a default for invalid inputs

theorem tenth_pair_in_twentieth_row : nth_pair_in_row 20 10 = (10, 11) :=
by sorry

end tenth_pair_in_twentieth_row_l215_215633


namespace train_cross_time_l215_215608

noncomputable def train_length : ℝ := 130
noncomputable def train_speed_kph : ℝ := 45
noncomputable def total_length : ℝ := 375

noncomputable def speed_mps := train_speed_kph * 1000 / 3600
noncomputable def distance := train_length + total_length

theorem train_cross_time : (distance / speed_mps) = 30 := by
  sorry

end train_cross_time_l215_215608


namespace expression_result_l215_215968

theorem expression_result :
  ( (9 + (1 / 2)) + (7 + (1 / 6)) + (5 + (1 / 12)) + (3 + (1 / 20)) + (1 + (1 / 30)) ) * 12 = 310 := by
  sorry

end expression_result_l215_215968


namespace product_of_powers_l215_215040

theorem product_of_powers (x y : ℕ) (h1 : x = 2) (h2 : y = 3) :
  ((x ^ 1 + y ^ 1) * (x ^ 2 + y ^ 2) * (x ^ 4 + y ^ 4) * 
   (x ^ 8 + y ^ 8) * (x ^ 16 + y ^ 16) * (x ^ 32 + y ^ 32) * 
   (x ^ 64 + y ^ 64)) = y ^ 128 - x ^ 128 :=
by
  rw [h1, h2]
  -- We would proceed with the proof here, but it's not needed per instructions.
  sorry

end product_of_powers_l215_215040


namespace ratio_of_powers_l215_215380

theorem ratio_of_powers (a x : ℝ) (h : a^(2 * x) = Real.sqrt 2 - 1) : (a^(3 * x) + a^(-3 * x)) / (a^x + a^(-x)) = 2 * Real.sqrt 2 - 1 := by
  sorry

end ratio_of_powers_l215_215380


namespace integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l215_215250

theorem integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5 :
  ∀ n : ℤ, ∃ k : ℕ, (n^3 - 3 * n^2 + n + 2 = 5^k) ↔ n = 3 :=
by
  intro n
  exists sorry
  sorry

end integer_n_cubed_minus_3_n_squared_plus_n_plus_2_eq_power_of_5_l215_215250


namespace alex_final_bill_l215_215177

def original_bill : ℝ := 500
def first_late_charge (bill : ℝ) : ℝ := bill * 1.02
def final_bill (bill : ℝ) : ℝ := first_late_charge bill * 1.03

theorem alex_final_bill : final_bill original_bill = 525.30 :=
by sorry

end alex_final_bill_l215_215177


namespace cos_300_eq_one_half_l215_215232

theorem cos_300_eq_one_half :
  Real.cos (300 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_300_eq_one_half_l215_215232


namespace polynomial_root_condition_l215_215233

theorem polynomial_root_condition (a : ℝ) :
  (∃ x1 x2 x3 : ℝ,
    (x1^3 - 6 * x1^2 + a * x1 + a = 0) ∧
    (x2^3 - 6 * x2^2 + a * x2 + a = 0) ∧
    (x3^3 - 6 * x3^2 + a * x3 + a = 0) ∧
    ((x1 - 3)^3 + (x2 - 3)^3 + (x3 - 3)^3 = 0)) →
  a = -9 :=
by
  sorry

end polynomial_root_condition_l215_215233


namespace simplify_and_evaluate_l215_215152

theorem simplify_and_evaluate :
  let x := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := 
by
  let x := 2 * Real.sqrt 3
  sorry

end simplify_and_evaluate_l215_215152


namespace triangle_side_lengths_expression_neg_l215_215063

theorem triangle_side_lengths_expression_neg {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^4 + b^4 + c^4 - 2 * a^2 * b^2 - 2 * b^2 * c^2 - 2 * c^2 * a^2 < 0 := 
by 
  sorry

end triangle_side_lengths_expression_neg_l215_215063


namespace lines_intersect_and_sum_l215_215689

theorem lines_intersect_and_sum (a b : ℝ) :
  (∃ x y : ℝ, x = (1 / 3) * y + a ∧ y = (1 / 3) * x + b ∧ x = 3 ∧ y = 3) →
  a + b = 4 :=
by
  sorry

end lines_intersect_and_sum_l215_215689


namespace sum_of_all_possible_values_of_g_11_l215_215535

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (x : ℝ) : ℝ := 3 * x + 4

theorem sum_of_all_possible_values_of_g_11 :
  (∀ x : ℝ, f x = 11 → g x = 13 ∨ g x = 7) →
  (13 + 7 = 20) := by
  intros h
  sorry

end sum_of_all_possible_values_of_g_11_l215_215535


namespace smallest_YZ_minus_XZ_l215_215536

theorem smallest_YZ_minus_XZ 
  (XZ YZ XY : ℕ)
  (h_sum : XZ + YZ + XY = 3001)
  (h_order : XZ < YZ ∧ YZ ≤ XY)
  (h_triangle_ineq1 : XZ + YZ > XY)
  (h_triangle_ineq2 : XZ + XY > YZ)
  (h_triangle_ineq3 : YZ + XY > XZ) :
  ∃ XZ YZ XY : ℕ, YZ - XZ = 1 := sorry

end smallest_YZ_minus_XZ_l215_215536


namespace necessary_but_not_sufficient_condition_l215_215447

noncomputable def necessary_but_not_sufficient (x : ℝ) : Prop :=
  (3 - x >= 0 → |x - 1| ≤ 2) ∧ ¬(3 - x >= 0 ↔ |x - 1| ≤ 2)

theorem necessary_but_not_sufficient_condition (x : ℝ) :
  necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l215_215447


namespace acceptable_arrangements_correct_l215_215762

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     => 1
| (n+1) => (n + 1) * factorial n

-- Define the total number of people
def total_people := 8

-- Calculate the total arrangements of 8 people
def total_arrangements := factorial total_people

-- Calculate the arrangements where Alice and Bob are together
def reduced_people := total_people - 1
def alice_bob_arrangements := factorial reduced_people * factorial 2

-- Calculate the acceptable arrangements where Alice and Bob are not together
def acceptable_arrangements := total_arrangements - alice_bob_arrangements

-- The theorem statement, asserting the correct answer
theorem acceptable_arrangements_correct : acceptable_arrangements = 30240 :=
by
  sorry

end acceptable_arrangements_correct_l215_215762


namespace consecutive_integers_sum_l215_215605

theorem consecutive_integers_sum (a b : ℤ) (sqrt_33 : ℝ) (h1 : a < sqrt_33) (h2 : sqrt_33 < b) (h3 : b = a + 1) (h4 : sqrt_33 = Real.sqrt 33) : a + b = 11 :=
  sorry

end consecutive_integers_sum_l215_215605


namespace product_of_digits_l215_215187

theorem product_of_digits (A B : ℕ) (h1 : A + B = 14) (h2 : (10 * A + B) % 4 = 0) : A * B = 48 :=
by
  sorry

end product_of_digits_l215_215187


namespace equation_C_is_symmetric_l215_215149

def symm_y_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), f x y ↔ f (-x) y

def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x - y = 1

theorem equation_C_is_symmetric : symm_y_axis equation_C :=
by
  sorry

end equation_C_is_symmetric_l215_215149


namespace positive_number_divisible_by_4_l215_215575

theorem positive_number_divisible_by_4 (N : ℕ) (h1 : N % 4 = 0) (h2 : (2 + 4 + N + 3) % 2 = 1) : N = 4 := 
by 
  sorry

end positive_number_divisible_by_4_l215_215575


namespace ruth_train_track_length_l215_215723

theorem ruth_train_track_length (n : ℕ) (R : ℕ)
  (h_sean : 72 = 8 * n)
  (h_ruth : 72 = R * n) : 
  R = 8 :=
by
  sorry

end ruth_train_track_length_l215_215723


namespace total_worth_of_stock_l215_215897

theorem total_worth_of_stock (W : ℝ) 
    (h1 : 0.2 * W * 0.1 = 0.02 * W)
    (h2 : 0.6 * (0.8 * W) * 0.05 = 0.024 * W)
    (h3 : 0.2 * (0.8 * W) = 0.16 * W)
    (h4 : (0.024 * W) - (0.02 * W) = 400) 
    : W = 100000 := 
sorry

end total_worth_of_stock_l215_215897


namespace calculate_blue_candles_l215_215141

-- Definitions based on identified conditions
def total_candles : Nat := 79
def yellow_candles : Nat := 27
def red_candles : Nat := 14
def blue_candles : Nat := total_candles - (yellow_candles + red_candles)

-- The proof statement
theorem calculate_blue_candles : blue_candles = 38 :=
by
  sorry

end calculate_blue_candles_l215_215141


namespace square_root_area_ratio_l215_215646

theorem square_root_area_ratio 
  (side_C : ℝ) (side_D : ℝ)
  (hC : side_C = 45) 
  (hD : side_D = 60) : 
  Real.sqrt ((side_C^2) / (side_D^2)) = 3 / 4 := by
  -- proof goes here
  sorry

end square_root_area_ratio_l215_215646


namespace polynomial_coeff_sum_l215_215751

theorem polynomial_coeff_sum :
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  a_sum - a_0 = 2555 :=
by
  let f : ℕ → ℕ := λ x => (x^2 + 1) * (x - 1)^8
  let a_0 := f 2
  let a_sum := f 3
  show a_sum - a_0 = 2555
  sorry

end polynomial_coeff_sum_l215_215751


namespace base_seven_sum_l215_215925

def base_seven_to_ten (n : ℕ) : ℕ := 3 * 7^1 + 5 * 7^0   -- Converts 35_7 to base 10
def base_seven_to_ten' (m : ℕ) : ℕ := 1 * 7^1 + 2 * 7^0  -- Converts 12_7 to base 10

noncomputable def base_ten_product (a b : ℕ) : ℕ := (a * b) -- Computes product in base 10

noncomputable def base_ten_to_seven (p : ℕ) : ℕ :=        -- Converts base 10 to base 7
  let p1 := (p / 7 / 7) % 7
  let p2 := (p / 7) % 7
  let p3 := p % 7
  p1 * 100 + p2 * 10 + p3

noncomputable def sum_of_digits (a : ℕ) : ℕ :=             -- Sums digits in base 7
  let d1 := (a / 100) % 10
  let d2 := (a / 10) % 10
  let d3 := a % 10
  d1 + d2 + d3

noncomputable def base_ten_to_seven' (s : ℕ) : ℕ :=        -- Converts sum back to base 7
  let s1 := s / 7
  let s2 := s % 7
  s1 * 10 + s2

theorem base_seven_sum (n m : ℕ) : base_ten_to_seven' (sum_of_digits (base_ten_to_seven (base_ten_product (base_seven_to_ten n) (base_seven_to_ten' m)))) = 15 :=
by
  sorry

end base_seven_sum_l215_215925


namespace bus_speed_kmph_l215_215365

theorem bus_speed_kmph : 
  let distance := 600.048 
  let time := 30
  (distance / time) * 3.6 = 72.006 :=
by
  sorry

end bus_speed_kmph_l215_215365


namespace Linda_total_amount_at_21_years_l215_215578

theorem Linda_total_amount_at_21_years (P : ℝ) (r : ℝ) (n : ℕ) (initial_principal : P = 1500) (annual_rate : r = 0.03) (years : n = 21):
    P * (1 + r)^n = 2709.17 :=
by
  sorry

end Linda_total_amount_at_21_years_l215_215578


namespace max_value_of_operation_l215_215726

theorem max_value_of_operation : 
  ∃ (n : ℤ), (10 ≤ n ∧ n ≤ 99) ∧ 4 * (300 - n) = 1160 := by
  sorry

end max_value_of_operation_l215_215726


namespace max_bag_weight_is_50_l215_215382

noncomputable def max_weight_allowed (people bags_per_person more_bags_allowed total_weight : ℕ) : ℝ := 
  total_weight / ((people * bags_per_person) + more_bags_allowed)

theorem max_bag_weight_is_50 : ∀ (people bags_per_person more_bags_allowed total_weight : ℕ), 
  people = 6 → 
  bags_per_person = 5 → 
  more_bags_allowed = 90 → 
  total_weight = 6000 →
  max_weight_allowed people bags_per_person more_bags_allowed total_weight = 50 := 
by 
  sorry

end max_bag_weight_is_50_l215_215382


namespace minimum_value_expr_min_value_reachable_l215_215038

noncomputable def expr (x y : ℝ) : ℝ :=
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x

theorem minimum_value_expr (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  expr x y ≥ (2 * Real.sqrt 564) / 3 :=
sorry

theorem min_value_reachable :
  ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ expr x y = (2 * Real.sqrt 564) / 3 :=
sorry

end minimum_value_expr_min_value_reachable_l215_215038


namespace total_marbles_l215_215691

namespace MarbleBag

def numBlue : ℕ := 5
def numRed : ℕ := 9
def probRedOrWhite : ℚ := 5 / 6

theorem total_marbles (total_mar : ℕ) (numWhite : ℕ) (h1 : probRedOrWhite = (numRed + numWhite) / total_mar)
                      (h2 : total_mar = numBlue + numRed + numWhite) :
  total_mar = 30 :=
by
  sorry

end MarbleBag

end total_marbles_l215_215691


namespace cos_angle_l215_215916

noncomputable def angle := -19 * Real.pi / 6

theorem cos_angle : Real.cos angle = Real.sqrt 3 / 2 :=
by sorry

end cos_angle_l215_215916


namespace polygon_P_properties_l215_215763

-- Definitions of points A, B, and C
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (1, 0.5, 0)
def C : (ℝ × ℝ × ℝ) := (0, 0.5, 1)

-- Condition of cube intersection and plane containing A, B, and C
def is_corner_of_cube (p : ℝ × ℝ × ℝ) : Prop :=
  p = A

def are_midpoints_of_cube_edges (p₁ p₂ : ℝ × ℝ × ℝ) : Prop :=
  (p₁ = B ∧ p₂ = C)

-- Polygon P resulting from the plane containing A, B, and C intersecting the cube
def num_sides_of_polygon (p : ℝ × ℝ × ℝ) : ℕ := 5 -- Given the polygon is a pentagon

-- Area of triangle ABC
noncomputable def area_triangle_ABC : ℝ :=
  (1/2) * (Real.sqrt 1.5)

-- Area of polygon P
noncomputable def area_polygon_P : ℝ :=
  (11/6) * area_triangle_ABC

-- Theorem stating that polygon P has 5 sides and the ratio of its area to the area of triangle ABC is 11/6
theorem polygon_P_properties (A B C : (ℝ × ℝ × ℝ))
  (hA : is_corner_of_cube A) (hB : are_midpoints_of_cube_edges B C) :
  num_sides_of_polygon A = 5 ∧ area_polygon_P / area_triangle_ABC = (11/6) :=
by sorry

end polygon_P_properties_l215_215763


namespace number_of_workers_in_second_group_l215_215253

theorem number_of_workers_in_second_group (w₁ w₂ d₁ d₂ : ℕ) (total_wages₁ total_wages₂ : ℝ) (daily_wage : ℝ) :
  w₁ = 15 ∧ d₁ = 6 ∧ total_wages₁ = 9450 ∧ 
  w₂ * d₂ * daily_wage = total_wages₂ ∧ d₂ = 5 ∧ total_wages₂ = 9975 ∧ 
  daily_wage = 105 
  → w₂ = 19 :=
by
  sorry

end number_of_workers_in_second_group_l215_215253


namespace box_ratio_l215_215424

theorem box_ratio (h : ℤ) (l : ℤ) (w : ℤ) (v : ℤ)
  (H_height : h = 12)
  (H_length : l = 3 * h)
  (H_volume : l * w * h = 3888)
  (H_length_multiple : ∃ m, l = m * w) :
  l / w = 4 := by
  sorry

end box_ratio_l215_215424


namespace Cary_walked_miles_round_trip_l215_215129

theorem Cary_walked_miles_round_trip : ∀ (m : ℕ), 
  150 * m - 200 = 250 → m = 3 := 
by
  intros m h
  sorry

end Cary_walked_miles_round_trip_l215_215129


namespace six_digit_number_reversed_by_9_l215_215746

-- Hypothetical function to reverse digits of a number
def reverseDigits (n : ℕ) : ℕ := sorry

theorem six_digit_number_reversed_by_9 :
  ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n * 9 = reverseDigits n :=
by
  sorry

end six_digit_number_reversed_by_9_l215_215746


namespace solve_range_m_l215_215234

variable (m : ℝ)
def p := m < 0
def q := ∀ x : ℝ, x^2 + m * x + 1 > 0

theorem solve_range_m (hpq : p m ∧ q m) : -2 < m ∧ m < 0 := 
  sorry

end solve_range_m_l215_215234


namespace mutually_exclusive_events_l215_215002

-- Definitions based on the given conditions
def sample_inspection (n : ℕ) := n = 10
def event_A (defective_products : ℕ) := defective_products ≥ 2
def event_B (defective_products : ℕ) := defective_products ≤ 1

-- The proof statement
theorem mutually_exclusive_events (n : ℕ) (defective_products : ℕ) 
  (h1 : sample_inspection n) (h2 : event_A defective_products) : 
  event_B defective_products = false :=
by
  sorry

end mutually_exclusive_events_l215_215002


namespace max_cables_to_ensure_communication_l215_215913

theorem max_cables_to_ensure_communication
    (A B : ℕ) (n : ℕ) 
    (hA : A = 16) (hB : B = 12) (hn : n = 28) :
    (A * B ≤ 192) ∧ (A * B = 192) :=
by
  sorry

end max_cables_to_ensure_communication_l215_215913


namespace verify_segment_lengths_l215_215003

noncomputable def segment_lengths_proof : Prop :=
  let a := 2
  let b := 3
  let alpha := Real.arccos (5 / 16)
  let segment1 := 4 / 3
  let segment2 := 2 / 3
  let segment3 := 2
  let segment4 := 1
  ∀ (s1 s2 s3 s4 : ℝ), 
    (s1 = segment1 ∧ s2 = segment2 ∧ s3 = segment3 ∧ s4 = segment4) ↔
    -- Parallelogram sides and angle constraints
    (s1 + s2 = a ∧ s3 + s4 = b ∧ 
     -- Mutually perpendicular lines divide into equal areas
     (s1 * s3 * Real.sin alpha / 2 = s2 * s4 * Real.sin alpha / 2) )

-- Placeholder for proof
theorem verify_segment_lengths : segment_lengths_proof :=
  sorry

end verify_segment_lengths_l215_215003


namespace sum_x_coordinates_common_points_l215_215659

theorem sum_x_coordinates_common_points (x y : ℤ) (h1 : y ≡ 3 * x + 5 [ZMOD 13]) (h2 : y ≡ 9 * x + 1 [ZMOD 13]) : x ≡ 5 [ZMOD 13] :=
sorry

end sum_x_coordinates_common_points_l215_215659


namespace circle_equation_range_of_k_l215_215179

theorem circle_equation_range_of_k (k : ℝ) : 
  (∃ x y : ℝ, x^2 + y^2 + 4*k*x - 2*y + 5*k = 0) ↔ (k > 1 ∨ k < 1/4) :=
by
  sorry

end circle_equation_range_of_k_l215_215179


namespace tangent_points_are_on_locus_l215_215792

noncomputable def tangent_points_locus (d : ℝ) : Prop :=
∀ (x y : ℝ), 
((x ≠ 0 ∨ y ≠ 0) ∧ (x-d ≠ 0)) ∧ (y = x) 
→ (y^2 - x*y + d*(x + y) = 0)

theorem tangent_points_are_on_locus (d : ℝ) : 
  tangent_points_locus d :=
by sorry

end tangent_points_are_on_locus_l215_215792


namespace distance_points_l215_215342

theorem distance_points : 
  let P1 := (2, -1)
  let P2 := (7, 6)
  dist P1 P2 = Real.sqrt 74 :=
by
  sorry

end distance_points_l215_215342


namespace alice_bob_total_dollars_l215_215579

-- Define Alice's amount in dollars
def alice_amount : ℚ := 5 / 8

-- Define Bob's amount in dollars
def bob_amount : ℚ := 3 / 5

-- Define the total amount in dollars
def total_amount : ℚ := alice_amount + bob_amount

theorem alice_bob_total_dollars : (alice_amount + bob_amount : ℚ) = 1.225 := by
    sorry

end alice_bob_total_dollars_l215_215579


namespace find_c_plus_d_l215_215496

theorem find_c_plus_d (a b c d : ℤ) (h1 : a + b = 14) (h2 : b + c = 9) (h3 : a + d = 8) : c + d = 3 := 
by
  sorry

end find_c_plus_d_l215_215496


namespace find_y_l215_215887

/-- Given (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10) and x = 12, prove that y = 10 -/
theorem find_y (x y : ℕ) (h : (2 ^ x) - (2 ^ y) = 3 * (2 ^ 10)) (hx : x = 12) : y = 10 :=
by
  sorry

end find_y_l215_215887


namespace marek_sequence_sum_l215_215419

theorem marek_sequence_sum (x : ℝ) :
  let a := x
  let b := (a + 4) / 4 - 4
  let c := (b + 4) / 4 - 4
  let d := (c + 4) / 4 - 4
  (a + 4) / 4 * 4 + (b + 4) / 4 * 4 + (c + 4) / 4 * 4 + (d + 4) / 4 * 4 = 80 →
  x = 38 :=
by
  sorry

end marek_sequence_sum_l215_215419


namespace find_f6_l215_215036

-- Define the function f
variable {f : ℝ → ℝ}
-- The function satisfies f(x + y) = f(x) + f(y) for all real numbers x and y
axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
-- f(4) = 6
axiom f_of_4 : f 4 = 6

theorem find_f6 : f 6 = 9 :=
by
    sorry

end find_f6_l215_215036


namespace divisible_by_8640_l215_215069

theorem divisible_by_8640 (x : ℤ) : 8640 ∣ (x^9 - 6 * x^7 + 9 * x^5 - 4 * x^3) :=
  sorry

end divisible_by_8640_l215_215069


namespace inequality_and_equality_equality_condition_l215_215974

theorem inequality_and_equality (a b : ℕ) (ha : a > 1) (hb : b > 2) : a^b + 1 ≥ b * (a + 1) :=
by sorry

theorem equality_condition (a b : ℕ) : a = 2 ∧ b = 3 → a^b + 1 = b * (a + 1) :=
by
  intro h
  cases h
  sorry

end inequality_and_equality_equality_condition_l215_215974


namespace focus_of_parabola_l215_215634

def parabola (x : ℝ) : ℝ := (x - 3) ^ 2

theorem focus_of_parabola :
  ∃ f : ℝ × ℝ, f = (3, 1 / 4) ∧
  ∀ x : ℝ, parabola x = (x - 3)^2 :=
sorry

end focus_of_parabola_l215_215634


namespace problem_divisible_by_1946_l215_215873

def F (n : ℕ) : ℤ := 1492 ^ n - 1770 ^ n - 1863 ^ n + 2141 ^ n

theorem problem_divisible_by_1946 
  (n : ℕ) 
  (hn : n ≤ 1945) : 
  1946 ∣ F n :=
sorry

end problem_divisible_by_1946_l215_215873


namespace remaining_rice_l215_215627

theorem remaining_rice {q_0 : ℕ} {c : ℕ} {d : ℕ} 
    (h_q0 : q_0 = 52) 
    (h_c : c = 9) 
    (h_d : d = 3) : 
    q_0 - (c * d) = 25 := 
  by 
    -- Proof to be written here
    sorry

end remaining_rice_l215_215627


namespace condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l215_215103

theorem condition_sufficient_but_not_necessary (x y : ℝ) :
  (x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0) :=
sorry

theorem condition_not_necessary (x y : ℝ) :
  ((x + 4) * (x + 3) ≥ 0) → ¬ (x^2 + y^2 + 4*x + 3 ≤ 0) :=
sorry

-- Combine both into a single statement using conjunction
theorem combined_condition (x y : ℝ) :
  ((x^2 + y^2 + 4*x + 3 ≤ 0) → ((x + 4) * (x + 3) ≥ 0))
  ∧ ((x + 4) * (x + 3) ≥ 0 → ¬(x^2 + y^2 + 4*x + 3 ≤ 0)) :=
sorry

end condition_sufficient_but_not_necessary_condition_not_necessary_combined_condition_l215_215103


namespace tangent_line_at_P_l215_215189

noncomputable def y (x : ℝ) : ℝ := 2 * x^2 + 1

def P : ℝ × ℝ := (-1, 3)

theorem tangent_line_at_P :
    ∀ (x y : ℝ), (y = 2*x^2 + 1) →
    (x, y) = P →
    ∃ m b : ℝ, b = -1 ∧ m = -4 ∧ (y = m*x + b) :=
by
  sorry

end tangent_line_at_P_l215_215189


namespace find_largest_number_l215_215476

theorem find_largest_number
  (a b c d : ℕ)
  (h1 : a + b + c = 222)
  (h2 : a + b + d = 208)
  (h3 : a + c + d = 197)
  (h4 : b + c + d = 180) :
  max a (max b (max c d)) = 89 :=
by
  sorry

end find_largest_number_l215_215476


namespace remainder_seven_pow_two_thousand_mod_thirteen_l215_215987

theorem remainder_seven_pow_two_thousand_mod_thirteen :
  7^2000 % 13 = 1 := by
  sorry

end remainder_seven_pow_two_thousand_mod_thirteen_l215_215987


namespace complete_square_proof_l215_215104

def quadratic_eq := ∀ (x : ℝ), x^2 - 6 * x + 5 = 0
def form_completing_square (b c : ℝ) := ∀ (x : ℝ), (x + b)^2 = c

theorem complete_square_proof :
  quadratic_eq → (∃ b c : ℤ, form_completing_square (b : ℝ) (c : ℝ) ∧ b + c = 11) :=
by
  sorry

end complete_square_proof_l215_215104


namespace total_amount_received_correct_l215_215274

variable (total_won : ℝ) (fraction : ℝ) (students : ℕ)
variable (portion_per_student : ℝ := total_won * fraction)
variable (total_given : ℝ := portion_per_student * students)

theorem total_amount_received_correct :
  total_won = 555850 →
  fraction = 3 / 10000 →
  students = 500 →
  total_given = 833775 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_amount_received_correct_l215_215274


namespace range_of_a_if_distinct_zeros_l215_215107

theorem range_of_a_if_distinct_zeros (a : ℝ) :
(∀ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ (x₁^3 - 3*x₁ + a = 0) ∧ (x₂^3 - 3*x₂ + a = 0) ∧ (x₃^3 - 3*x₃ + a = 0)) → -2 < a ∧ a < 2 :=
by
  sorry

end range_of_a_if_distinct_zeros_l215_215107


namespace fractional_eq_range_m_l215_215979

theorem fractional_eq_range_m (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 2) + 1 = x / (2 - x))) → m ≤ 2 ∧ m ≠ -2 :=
by
  sorry

end fractional_eq_range_m_l215_215979


namespace judson_contribution_l215_215526

theorem judson_contribution (J K C : ℝ) (hK : K = 1.20 * J) (hC : C = K + 200) (h_total : J + K + C = 1900) : J = 500 :=
by
  -- This is where the proof would go, but we are skipping it as per the instructions.
  sorry

end judson_contribution_l215_215526


namespace largest_of_four_numbers_l215_215008

theorem largest_of_four_numbers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  max (max (max (a^2 + b^2) (2 * a * b)) a) (1 / 2) = a^2 + b^2 :=
by
  sorry

end largest_of_four_numbers_l215_215008


namespace Nils_has_300_geese_l215_215705

variables (A x k n : ℕ)

def condition1 (A x k n : ℕ) : Prop :=
  A = k * x * n

def condition2 (A x k n : ℕ) : Prop :=
  A = (k + 20) * x * (n - 50)

def condition3 (A x k n : ℕ) : Prop :=
  A = (k - 10) * x * (n + 100)

theorem Nils_has_300_geese (A x k n : ℕ) :
  condition1 A x k n →
  condition2 A x k n →
  condition3 A x k n →
  n = 300 :=
by
  intros h1 h2 h3
  sorry

end Nils_has_300_geese_l215_215705


namespace abc_is_cube_l215_215502

theorem abc_is_cube (a b c : ℤ) (h : (a:ℚ) / (b:ℚ) + (b:ℚ) / (c:ℚ) + (c:ℚ) / (a:ℚ) = 3) : ∃ x : ℤ, abc = x^3 :=
by
  sorry

end abc_is_cube_l215_215502


namespace solve_eq1_solve_eq2_l215_215395

theorem solve_eq1 (y : ℝ) : 6 - 3 * y = 15 + 6 * y ↔ y = -1 := by
  sorry

theorem solve_eq2 (x : ℝ) : (1 - 2 * x) / 3 = (3 * x + 1) / 7 - 2 ↔ x = 2 := by
  sorry

end solve_eq1_solve_eq2_l215_215395


namespace arithmetic_sequence_s10_l215_215998

noncomputable def arithmetic_sequence_sum (n : ℕ) (a d : ℤ) : ℤ :=
  n * (2 * a + (n - 1) * d) / 2

theorem arithmetic_sequence_s10 (a : ℤ) (d : ℤ)
  (h1 : a + (a + 8 * d) = 18)
  (h4 : a + 3 * d = 7) :
  arithmetic_sequence_sum 10 a d = 100 :=
by sorry

end arithmetic_sequence_s10_l215_215998


namespace looms_employed_l215_215168

def sales_value := 500000
def manufacturing_expenses := 150000
def establishment_charges := 75000
def profit_decrease := 5000

def profit_per_loom (L : ℕ) : ℕ := (sales_value / L) - (manufacturing_expenses / L)

theorem looms_employed (L : ℕ) (h : profit_per_loom L = profit_decrease) : L = 70 :=
by
  have h_eq : profit_per_loom L = (sales_value - manufacturing_expenses) / L := by
    sorry
  have profit_expression : profit_per_loom L = profit_decrease := by
    sorry
  have L_value : L = (sales_value - manufacturing_expenses) / profit_decrease := by
    sorry
  have L_is_70 : L = 70 := by
    sorry
  exact L_is_70

end looms_employed_l215_215168


namespace smallest_n_l215_215150

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, b * k = a

def meets_condition (n : ℕ) : Prop :=
  n > 0 ∧
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 ∧ is_divisible (n^2 - n + 1) k ∧
  ∃ l : ℕ, 1 ≤ l ∧ l ≤ n + 1 ∧ ¬ is_divisible (n^2 - n + 1) l

theorem smallest_n : ∃ n : ℕ, meets_condition n ∧ n = 5 :=
by
  sorry

end smallest_n_l215_215150


namespace min_segments_to_erase_l215_215647

noncomputable def nodes (m n : ℕ) : ℕ := (m - 2) * (n - 2)

noncomputable def segments_to_erase (m n : ℕ) : ℕ := (nodes m n + 1) / 2

theorem min_segments_to_erase (m n : ℕ) (hm : m = 11) (hn : n = 11) :
  segments_to_erase m n = 41 := by
  sorry

end min_segments_to_erase_l215_215647


namespace cone_base_radius_l215_215803

/-- A hemisphere of radius 3 rests on the base of a circular cone and is tangent to the cone's lateral surface along a circle. 
Given that the height of the cone is 9, prove that the base radius of the cone is 10.5. -/
theorem cone_base_radius
  (r_h : ℝ) (h : ℝ) (r : ℝ) 
  (hemisphere_tangent_cone : r_h = 3)
  (cone_height : h = 9)
  (tangent_circle_height : r - r_h = 3) :
  r = 10.5 := by
  sorry

end cone_base_radius_l215_215803


namespace line_through_origin_tangent_lines_line_through_tangents_l215_215258

section GeomProblem

variables {A : ℝ × ℝ} {C : ℝ × ℝ → Prop}

def is_circle (C : ℝ × ℝ → Prop) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
∀ (P : ℝ × ℝ), C P ↔ (P.1 - center.1) ^ 2 + (P.2 - center.2) ^ 2 = radius ^ 2

theorem line_through_origin (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ m : ℝ, ∀ P : ℝ × ℝ, C P → abs ((m * P.1 - P.2) / Real.sqrt (m ^ 2 + 1)) = 1)
    ↔ m = 0 :=
sorry

theorem tangent_lines (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P : ℝ × ℝ, C P → (P.2 - 2 * Real.sqrt 3) = k * (P.1 - 1))
    ↔ (∀ P : ℝ × ℝ, C P → (Real.sqrt 3 * P.1 - 3 * P.2 + 5 * Real.sqrt 3 = 0 ∨ P.1 = 1)) :=
sorry

theorem line_through_tangents (C : ℝ × ℝ → Prop) (A : ℝ × ℝ)
  (hC : is_circle C (-1, 0) 2)
  (hA : A = (1, 2 * Real.sqrt 3)) :
  (∃ k : ℝ, ∀ P D E : ℝ × ℝ, C P → (Real.sqrt 3 * D.1 - 3 * D.2 + 5 * Real.sqrt 3 = 0 ∧
                                      (E.1 - 1 = 0 ∨ Real.sqrt 3 * E.1 - 3 * E.2 + 5 * Real.sqrt 3 = 0)) →
    (D.1 + Real.sqrt 3 * D.2 - 1 = 0 ∧ E.1 + Real.sqrt 3 * E.2 - 1 = 0)) :=
sorry

end GeomProblem

end line_through_origin_tangent_lines_line_through_tangents_l215_215258


namespace rectangle_area_l215_215130

theorem rectangle_area (x y : ℝ) (hx : 3 * y = 7 * x) (hp : 2 * (x + y) = 40) :
  x * y = 84 := by
  sorry

end rectangle_area_l215_215130


namespace shoe_price_calculation_l215_215613

theorem shoe_price_calculation :
  let initialPrice : ℕ := 50
  let increasedPrice : ℕ := 60  -- initialPrice * 1.2
  let discountAmount : ℕ := 9    -- increasedPrice * 0.15
  increasedPrice - discountAmount = 51 := 
by
  sorry

end shoe_price_calculation_l215_215613


namespace GCF_LCM_proof_l215_215761

-- Define GCF (greatest common factor)
def GCF (a b : ℕ) : ℕ := Nat.gcd a b

-- Define LCM (least common multiple)
def LCM (a b : ℕ) : ℕ := Nat.lcm a b

theorem GCF_LCM_proof :
  GCF (LCM 9 21) (LCM 14 15) = 21 :=
by
  sorry

end GCF_LCM_proof_l215_215761


namespace find_2a_plus_b_l215_215353

open Real

variables {a b : ℝ}

-- Conditions
def angles_in_first_quadrant (a b : ℝ) : Prop := 
  0 < a ∧ a < π / 2 ∧ 0 < b ∧ b < π / 2

def cos_condition (a b : ℝ) : Prop :=
  5 * cos a ^ 2 + 3 * cos b ^ 2 = 2

def sin_condition (a b : ℝ) : Prop :=
  5 * sin (2 * a) + 3 * sin (2 * b) = 0

-- Problem statement
theorem find_2a_plus_b (a b : ℝ) 
  (h1 : angles_in_first_quadrant a b)
  (h2 : cos_condition a b)
  (h3 : sin_condition a b) :
  2 * a + b = π / 2 := 
sorry

end find_2a_plus_b_l215_215353


namespace negation_of_exists_eq_sin_l215_215729

theorem negation_of_exists_eq_sin : ¬ (∃ x : ℝ, x = Real.sin x) ↔ ∀ x : ℝ, x ≠ Real.sin x :=
by
  sorry

end negation_of_exists_eq_sin_l215_215729


namespace contrapositive_l215_215015

variables (p q : Prop)

theorem contrapositive (hpq : p → q) : ¬ q → ¬ p :=
by sorry

end contrapositive_l215_215015


namespace binomial_probability_X_eq_3_l215_215035

theorem binomial_probability_X_eq_3 :
  let n := 6
  let p := 1 / 2
  let k := 3
  let binom := Nat.choose n k
  (binom * p ^ k * (1 - p) ^ (n - k)) = 5 / 16 := by 
  sorry

end binomial_probability_X_eq_3_l215_215035


namespace find_value_m_sq_plus_2m_plus_n_l215_215819

noncomputable def m_n_roots (x : ℝ) : Prop := x^2 + x - 1001 = 0

theorem find_value_m_sq_plus_2m_plus_n
  (m n : ℝ)
  (hm : m_n_roots m)
  (hn : m_n_roots n)
  (h_sum : m + n = -1)
  (h_prod : m * n = -1001) :
  m^2 + 2 * m + n = 1000 :=
sorry

end find_value_m_sq_plus_2m_plus_n_l215_215819


namespace smallest_blocks_needed_for_wall_l215_215813

noncomputable def smallest_number_of_blocks (wall_length : ℕ) (wall_height : ℕ) (block_length1 : ℕ) (block_length2 : ℕ) (block_length3 : ℝ) : ℕ :=
  let blocks_per_odd_row := wall_length / block_length1
  let blocks_per_even_row := wall_length / block_length1 - 1 + 2
  let odd_rows := wall_height / 2 + 1
  let even_rows := wall_height / 2
  odd_rows * blocks_per_odd_row + even_rows * blocks_per_even_row

theorem smallest_blocks_needed_for_wall :
  smallest_number_of_blocks 120 7 2 1 1.5 = 423 :=
by
  sorry

end smallest_blocks_needed_for_wall_l215_215813


namespace universal_quantifiers_and_propositions_l215_215984

-- Definitions based on conditions
def universal_quantifiers_phrases := ["for all", "for any"]
def universal_quantifier_symbol := "∀"
def universal_proposition := "Universal Proposition"
def universal_proposition_representation := "∀ x ∈ M, p(x)"

-- Main theorem
theorem universal_quantifiers_and_propositions :
  universal_quantifiers_phrases = ["for all", "for any"]
  ∧ universal_quantifier_symbol = "∀"
  ∧ universal_proposition = "Universal Proposition"
  ∧ universal_proposition_representation = "∀ x ∈ M, p(x)" :=
by
  sorry

end universal_quantifiers_and_propositions_l215_215984


namespace even_odd_product_l215_215336

theorem even_odd_product (n : ℕ) (i : Fin n → Fin n) (h_perm : ∀ j : Fin n, ∃ k : Fin n, i k = j) :
  (∃ l, l % 2 = 0) → 
  ∀ (k : Fin n), ¬(i k = k) → 
  (n % 2 = 0 → (∃ m : ℤ, m + 1 % 2 = 1) ∨ (∃ m : ℤ, m + 1 % 2 = 0)) ∧ 
  (n % 2 = 1 → (∃ m : ℤ, m + 1 % 2 = 0)) :=
by
  sorry

end even_odd_product_l215_215336


namespace walls_divided_equally_l215_215711

-- Define the given conditions
def num_people : ℕ := 5
def num_rooms : ℕ := 9
def rooms_with_4_walls : ℕ := 5
def walls_per_room_4 : ℕ := 4
def rooms_with_5_walls : ℕ := 4
def walls_per_room_5 : ℕ := 5

-- Calculate the total number of walls
def total_walls : ℕ := (rooms_with_4_walls * walls_per_room_4) + (rooms_with_5_walls * walls_per_room_5)

-- Define the expected result
def walls_per_person : ℕ := total_walls / num_people

-- Theorem statement: Each person should paint 8 walls.
theorem walls_divided_equally : walls_per_person = 8 := by
  sorry

end walls_divided_equally_l215_215711


namespace solve_abs_inequality_l215_215369

theorem solve_abs_inequality (x : ℝ) (h : 1 < |x - 1| ∧ |x - 1| < 4) : (-3 < x ∧ x < 0) ∨ (2 < x ∧ x < 5) :=
by
  sorry

end solve_abs_inequality_l215_215369


namespace circle_Γ_contains_exactly_one_l215_215829

-- Condition definitions
variables (z1 z2 : ℂ) (Γ : ℂ → ℂ → Prop)
variable (hz1z2 : z1 * z2 = 1)
variable (hΓ_passes : Γ (-1) 1)
variable (hΓ_not_passes : ¬Γ z1 z2)

-- Math proof problem
theorem circle_Γ_contains_exactly_one (hz1z2 : z1 * z2 = 1)
    (hΓ_passes : Γ (-1) 1) (hΓ_not_passes : ¬Γ z1 z2) : 
  (Γ 0 z1 ↔ ¬Γ 0 z2) ∨ (Γ 0 z2 ↔ ¬Γ 0 z1) :=
sorry

end circle_Γ_contains_exactly_one_l215_215829


namespace tangent_line_eqn_l215_215180

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 1
noncomputable def f' (x : ℝ) : ℝ := Real.log x + 1

theorem tangent_line_eqn (h : f' x = 2) : 2 * x - y - Real.exp 1 + 1 = 0 :=
by
  sorry

end tangent_line_eqn_l215_215180


namespace largest_2_digit_number_l215_215154

theorem largest_2_digit_number:
  ∃ (N: ℕ), N >= 10 ∧ N < 100 ∧ N % 4 = 0 ∧ (∀ k: ℕ, k ≥ 1 → (N^k) % 100 = N % 100) ∧ 
  (∀ M: ℕ, M >= 10 → M < 100 → M % 4 = 0 → (∀ k: ℕ, k ≥ 1 → (M^k) % 100 = M % 100) → N ≥ M) :=
sorry

end largest_2_digit_number_l215_215154


namespace width_of_room_l215_215025

noncomputable def roomWidth (length : ℝ) (totalCost : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let area := totalCost / costPerSquareMeter
  area / length

theorem width_of_room :
  roomWidth 5.5 24750 1200 = 3.75 :=
by
  sorry

end width_of_room_l215_215025


namespace chord_bisected_by_point_of_ellipse_l215_215385

theorem chord_bisected_by_point_of_ellipse 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 36 + y^2 / 9 = 1)
  (bisecting_point : ∃ x y : ℝ, x = 4 ∧ y = 2) :
  ∃ a b c : ℝ, a = 1 ∧ b = 2 ∧ c = -8 ∧ ∀ x y : ℝ, a * x + b * y + c = 0 :=
by
   sorry

end chord_bisected_by_point_of_ellipse_l215_215385


namespace moles_of_KOH_combined_l215_215548

theorem moles_of_KOH_combined (H2O_formed : ℕ) (NH4I_used : ℕ) (ratio_KOH_H2O : ℕ) : H2O_formed = 54 → NH4I_used = 3 → ratio_KOH_H2O = 1 → H2O_formed = NH4I_used := 
by 
  intro H2O_formed_eq NH4I_used_eq ratio_eq 
  sorry

end moles_of_KOH_combined_l215_215548


namespace proof_M_inter_N_eq_01_l215_215636
open Set

theorem proof_M_inter_N_eq_01 :
  let M := {x : ℤ | x^2 = x}
  let N := {-1, 0, 1}
  M ∩ N = {0, 1} := by
  sorry

end proof_M_inter_N_eq_01_l215_215636


namespace mr_johnson_total_volunteers_l215_215241

theorem mr_johnson_total_volunteers (students_per_class : ℕ) (classes : ℕ) (teachers : ℕ) (additional_volunteers : ℕ) :
  students_per_class = 5 → classes = 6 → teachers = 13 → additional_volunteers = 7 →
  (students_per_class * classes + teachers + additional_volunteers) = 50 :=
by intros; simp [*]

end mr_johnson_total_volunteers_l215_215241


namespace union_of_A_and_B_l215_215522

-- Definitions for sets A and B
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- Theorem statement to prove the union of A and B
theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_of_A_and_B_l215_215522


namespace rabbit_shape_area_l215_215257

theorem rabbit_shape_area (A_ear : ℝ) (h1 : A_ear = 10) (h2 : A_ear = (1/8) * A_total) :
  A_total = 80 :=
by
  sorry

end rabbit_shape_area_l215_215257


namespace simplify_expression_l215_215046

theorem simplify_expression (x : ℝ) :
  ((3 * x^2 + 2 * x - 1) + 2 * x^2) * 4 + (5 - 2 / 2) * (3 * x^2 + 6 * x - 8) = 32 * x^2 + 32 * x - 36 :=
sorry

end simplify_expression_l215_215046


namespace intersection_points_count_l215_215134

theorem intersection_points_count
  : (∀ n : ℤ, ∃ (x y : ℝ), (x - ⌊x⌋) ^ 2 + y ^ 2 = 2 * (x - ⌊x⌋) ∨ y = 1 / 3 * x) →
    (∃ count : ℕ, count = 12) :=
by
  sorry

end intersection_points_count_l215_215134


namespace part_a_part_b_part_c_l215_215326

def f (n d : ℕ) : ℕ := sorry

theorem part_a (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 ≤ n :=
sorry

theorem part_b (n d : ℕ) (h_even_n_minus_d : (n - d) % 2 = 0) : f n d ≤ (n + d) / (d + 1) :=
sorry

theorem part_c (n : ℕ) (h_even_n : n % 2 = 0) : f n 0 = n :=
sorry

end part_a_part_b_part_c_l215_215326


namespace cross_prod_correct_l215_215176

open Matrix

def vec1 : ℝ × ℝ × ℝ := (3, -1, 4)
def vec2 : ℝ × ℝ × ℝ := (-4, 6, 2)
def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.1 * b.2.2 - a.2.2 * b.2.1,
  a.2.2 * b.1 - a.1 * b.2.2,
  a.1 * b.2.1 - a.2.1 * b.1)

theorem cross_prod_correct :
  cross_product vec1 vec2 = (-26, -22, 14) := by
  -- sorry is used to simplify the proof.
  sorry

end cross_prod_correct_l215_215176


namespace shopkeeper_total_cards_l215_215483

-- Definition of the number of cards in a complete deck
def cards_in_deck : Nat := 52

-- Definition of the number of complete decks the shopkeeper has
def number_of_decks : Nat := 3

-- Definition of the additional cards the shopkeeper has
def additional_cards : Nat := 4

-- The total number of cards the shopkeeper should have
def total_cards : Nat := number_of_decks * cards_in_deck + additional_cards

-- Theorem statement to prove the total number of cards is 160
theorem shopkeeper_total_cards : total_cards = 160 := by
  sorry

end shopkeeper_total_cards_l215_215483


namespace initial_population_l215_215422

theorem initial_population (P : ℝ) (h1 : P * 1.05 * 0.95 = 9975) : P = 10000 :=
by
  sorry

end initial_population_l215_215422


namespace Sheila_attend_probability_l215_215209

noncomputable def prob_rain := 0.3
noncomputable def prob_sunny := 0.4
noncomputable def prob_cloudy := 0.3

noncomputable def prob_attend_if_rain := 0.25
noncomputable def prob_attend_if_sunny := 0.9
noncomputable def prob_attend_if_cloudy := 0.5

noncomputable def prob_attend :=
  prob_rain * prob_attend_if_rain +
  prob_sunny * prob_attend_if_sunny +
  prob_cloudy * prob_attend_if_cloudy

theorem Sheila_attend_probability : prob_attend = 0.585 := by
  sorry

end Sheila_attend_probability_l215_215209


namespace geometric_sequence_nec_suff_l215_215596

theorem geometric_sequence_nec_suff (a b c : ℝ) : (b^2 = a * c) ↔ (∃ r : ℝ, b = a * r ∧ c = b * r) :=
sorry

end geometric_sequence_nec_suff_l215_215596


namespace mike_found_four_more_seashells_l215_215316

/--
Given:
1. Mike initially found 6.0 seashells.
2. The total number of seashells Mike had after finding more is 10.

Prove:
Mike found 4.0 more seashells.
-/
theorem mike_found_four_more_seashells (initial_seashells : ℝ) (total_seashells : ℝ)
  (h1 : initial_seashells = 6.0)
  (h2 : total_seashells = 10.0) :
  total_seashells - initial_seashells = 4.0 :=
by
  sorry

end mike_found_four_more_seashells_l215_215316


namespace proportion_of_salt_correct_l215_215390

def grams_of_salt := 50
def grams_of_water := 1000
def total_solution := grams_of_salt + grams_of_water
def proportion_of_salt : ℚ := grams_of_salt / total_solution

theorem proportion_of_salt_correct :
  proportion_of_salt = 1 / 21 := 
  by {
    sorry
  }

end proportion_of_salt_correct_l215_215390


namespace find_line_equation_l215_215319

theorem find_line_equation 
  (ellipse_eq : ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1)
  (P : ℝ × ℝ) (P_coord : P = (1, 3/2))
  (line_l : ∀ x : ℝ, ℝ)
  (line_eq : ∀ x : ℝ, y = k * x + b) 
  (intersects : ∀ A B : ℝ × ℝ, A ≠ P ∧ B ≠ P)
  (perpendicular : ∀ A B : ℝ × ℝ, (A.1 - 1) * (B.1 - 1) + (A.2 - 3 / 2) * (B.2 - 3 / 2) = 0)
  (bisected_by_y_axis : ∀ A B : ℝ × ℝ, A.1 + B.1 = 0) :
  ∃ k : ℝ, k = 3 / 2 ∨ k = -3 / 2 :=
sorry

end find_line_equation_l215_215319


namespace negation_of_universal_proposition_l215_215135

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, |x| + x^2 ≥ 0)) ↔ (∃ x : ℝ, |x| + x^2 < 0) :=
by
  sorry

end negation_of_universal_proposition_l215_215135


namespace gcd_1037_425_l215_215346

theorem gcd_1037_425 : Int.gcd 1037 425 = 17 :=
by
  sorry

end gcd_1037_425_l215_215346


namespace find_number_eq_36_l215_215841

theorem find_number_eq_36 (n : ℝ) (h : (n / 18) * (n / 72) = 1) : n = 36 :=
sorry

end find_number_eq_36_l215_215841


namespace fifth_scroll_age_l215_215594

def scrolls_age (n : ℕ) : ℕ :=
  match n with
  | 0 => 4080
  | k+1 => (3 * scrolls_age k) / 2

theorem fifth_scroll_age : scrolls_age 4 = 20655 := sorry

end fifth_scroll_age_l215_215594


namespace total_cost_of_cultivating_field_l215_215533

theorem total_cost_of_cultivating_field 
  (base height : ℕ) 
  (cost_per_hectare : ℝ) 
  (base_eq: base = 3 * height) 
  (height_eq: height = 300) 
  (cost_eq: cost_per_hectare = 24.68) 
  : (1/2 : ℝ) * base * height / 10000 * cost_per_hectare = 333.18 :=
by
  sorry

end total_cost_of_cultivating_field_l215_215533


namespace geometric_sequence_alpha5_eq_three_l215_215565

theorem geometric_sequence_alpha5_eq_three (α : ℕ → ℝ) 
  (h1 : ∀ n : ℕ, α (n + 1) = α n * r) 
  (h2 : α 4 * α 5 * α 6 = 27) : α 5 = 3 := 
by
  sorry

end geometric_sequence_alpha5_eq_three_l215_215565


namespace boat_upstream_speed_l215_215737

variable (Vb Vc : ℕ)

def boat_speed_upstream (Vb Vc : ℕ) : ℕ := Vb - Vc

theorem boat_upstream_speed (hVb : Vb = 50) (hVc : Vc = 20) : boat_speed_upstream Vb Vc = 30 :=
by sorry

end boat_upstream_speed_l215_215737


namespace sum_of_solutions_eq_neg2_l215_215101

noncomputable def sum_of_real_solutions (a : ℝ) (h : a > 2) : ℝ :=
  -2

theorem sum_of_solutions_eq_neg2 (a : ℝ) (h : a > 2) :
  sum_of_real_solutions a h = -2 := sorry

end sum_of_solutions_eq_neg2_l215_215101


namespace solve_for_x_l215_215048

variable {x : ℝ}

theorem solve_for_x (h : (4 * x ^ 2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : x = 1 :=
sorry

end solve_for_x_l215_215048


namespace capital_payment_l215_215475

theorem capital_payment (m : ℕ) (hm : m ≥ 3) : 
  ∃ d : ℕ, d = (1000 * (3^m - 2^(m-1))) / (3^m - 2^m) 
  ∧ (∃ a : ℕ, a = 4000 ∧ a = ((3/2)^(m-1) * (3000 - 3 * d) + 2 * d)) := 
by
  sorry

end capital_payment_l215_215475


namespace solve_problem_l215_215673

theorem solve_problem
    (x y z : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : x^2 + x * y + y^2 = 2)
    (h5 : y^2 + y * z + z^2 = 5)
    (h6 : z^2 + z * x + x^2 = 3) :
    x * y + y * z + z * x = 2 * Real.sqrt 2 := 
by
  sorry

end solve_problem_l215_215673


namespace computer_price_increase_l215_215788

theorem computer_price_increase (d : ℝ) (h : 2 * d = 585) : d * 1.2 = 351 := by
  sorry

end computer_price_increase_l215_215788


namespace problem_statement_l215_215905

variables {p q r s : ℝ}

theorem problem_statement 
  (h : (p - q) * (r - s) / (q - r) * (s - p) = 3 / 7) : 
  (p - r) * (q - s) / (p - q) * (r - s) = -4 / 3 :=
by sorry

end problem_statement_l215_215905


namespace collinear_R_S_T_l215_215780

theorem collinear_R_S_T
    (circle : Type)
    (P : circle)
    (A B C D : circle)
    (E F : Type → Type)
    (angle : ∀ (x y z : circle), ℝ)   -- Placeholder for angles
    (quadrilateral_inscribed_in_circle : ∀ (A B C D : circle), Prop)   -- Placeholder for the condition of the quadrilateral
    (extensions_intersect : ∀ (A B C D : circle) (E F : Type → Type), Prop)   -- Placeholder for extensions intersections
    (diagonals_intersect_at : ∀ (A C B D T : circle), Prop)   -- Placeholder for diagonals intersections
    (P_on_circle : ∀ (P : circle), Prop)        -- Point P is on the circle
    (PE_PF_intersect_again : ∀ (P R S : circle) (E F : Type → Type), Prop)   -- PE and PF intersect the circle again at R and S
    (R S T : circle) :
    quadrilateral_inscribed_in_circle A B C D →
    extensions_intersect A B C D E F →
    P_on_circle P →
    PE_PF_intersect_again P R S E F →
    diagonals_intersect_at A C B D T →
    ∃ collinearity : ∀ (R S T : circle), Prop,
    collinearity R S T := 
by
  intro h1 h2 h3 h4 h5
  sorry

end collinear_R_S_T_l215_215780


namespace water_to_milk_ratio_l215_215602

theorem water_to_milk_ratio 
  (V : ℝ) 
  (hV : V > 0) 
  (milk_volume1 : ℝ := (3 / 5) * V) 
  (water_volume1 : ℝ := (2 / 5) * V) 
  (milk_volume2 : ℝ := (4 / 5) * V) 
  (water_volume2 : ℝ := (1 / 5) * V)
  (total_milk_volume : ℝ := milk_volume1 + milk_volume2)
  (total_water_volume : ℝ := water_volume1 + water_volume2) :
  total_water_volume / total_milk_volume = (3 / 7) := 
  sorry

end water_to_milk_ratio_l215_215602


namespace theater_ticket_sales_l215_215560

theorem theater_ticket_sales (R H : ℕ) (h1 : R = 25) (h2 : H = 3 * R + 18) : H = 93 :=
by
  sorry

end theater_ticket_sales_l215_215560


namespace intercept_sum_l215_215523

-- Define the equation of the line and the condition on the intercepts.
theorem intercept_sum (c : ℚ) (x y : ℚ) (h1 : 3 * x + 5 * y + c = 0) (h2 : x + y = 55/4) : 
  c = 825/32 :=
sorry

end intercept_sum_l215_215523


namespace plot_length_60_l215_215781

/-- The length of a rectangular plot is 20 meters more than its breadth. If the cost of fencing the plot at Rs. 26.50 per meter is Rs. 5300, then the length of the plot in meters is 60. -/
theorem plot_length_60 (b l : ℝ) (h1 : l = b + 20) (h2 : 2 * (l + b) * 26.5 = 5300) : l = 60 :=
by
  sorry

end plot_length_60_l215_215781


namespace factorize_x2_plus_2x_l215_215743

theorem factorize_x2_plus_2x (x : ℝ) : x^2 + 2*x = x * (x + 2) :=
by sorry

end factorize_x2_plus_2x_l215_215743


namespace simplify_fraction_expression_l215_215980

variable (d : ℤ)

theorem simplify_fraction_expression : (6 + 4 * d) / 9 + 3 = (33 + 4 * d) / 9 := by
  sorry

end simplify_fraction_expression_l215_215980


namespace part1_part2_l215_215490

-- Definition of the function f
def f (a x : ℝ) : ℝ := |a * x - 2| - |x + 2|

-- First Proof Statement: Inequality for a = 2
theorem part1 : ∀ x : ℝ, - (1 : ℝ) / 3 ≤ x ∧ x ≤ 5 → f 2 x ≤ 1 :=
by
  sorry

-- Second Proof Statement: Range for a such that -4 ≤ f(x) ≤ 4 for all x ∈ ℝ
theorem part2 : ∀ x : ℝ, -4 ≤ f a x ∧ f a x ≤ 4 ↔ a = 1 ∨ a = -1 :=
by
  sorry

end part1_part2_l215_215490


namespace original_price_of_coat_l215_215281

theorem original_price_of_coat (P : ℝ) (h : 0.40 * P = 200) : P = 500 :=
by {
  sorry
}

end original_price_of_coat_l215_215281


namespace correct_choice_l215_215999

theorem correct_choice (x y m : ℝ) (h1 : x > y) (h2 : m > 0) : x - y > 0 := by
  sorry

end correct_choice_l215_215999


namespace arithmetic_sequence_a6_l215_215091

theorem arithmetic_sequence_a6 (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : a 6 = 16 :=
sorry

end arithmetic_sequence_a6_l215_215091


namespace no_int_solutions_5x2_minus_4y2_eq_2017_l215_215484

theorem no_int_solutions_5x2_minus_4y2_eq_2017 :
  ¬ ∃ x y : ℤ, 5 * x^2 - 4 * y^2 = 2017 :=
by
  -- The detailed proof goes here
  sorry

end no_int_solutions_5x2_minus_4y2_eq_2017_l215_215484


namespace intersection_point_on_square_diagonal_l215_215623

theorem intersection_point_on_square_diagonal (a b c : ℝ) (h : c = (a + b) / 2) :
  (b / 2) = (-a / 2) + c :=
by
  sorry

end intersection_point_on_square_diagonal_l215_215623


namespace count_real_numbers_a_with_integer_roots_l215_215750

theorem count_real_numbers_a_with_integer_roots :
  ∃ (S : Finset ℝ), (∀ (a : ℝ), (∃ (x y : ℤ), x^2 + a*x + 9*a = 0 ∧ y^2 + a*y + 9*a = 0) ↔ a ∈ S) ∧ S.card = 8 :=
by
  sorry

end count_real_numbers_a_with_integer_roots_l215_215750


namespace minimal_total_cost_l215_215919

def waterway_length : ℝ := 100
def max_speed : ℝ := 50
def other_costs_per_hour : ℝ := 3240
def speed_at_ten_cost : ℝ := 10
def fuel_cost_at_ten : ℝ := 60
def proportionality_constant : ℝ := 0.06

noncomputable def total_cost (v : ℝ) : ℝ :=
  6 * v^2 + 324000 / v

theorem minimal_total_cost :
  (∃ v : ℝ, 0 < v ∧ v ≤ max_speed ∧ total_cost v = 16200) ∧ 
  (∀ v : ℝ, 0 < v ∧ v ≤ max_speed → total_cost v ≥ 16200) :=
sorry

end minimal_total_cost_l215_215919


namespace smallest_integer_sum_consecutive_l215_215216

theorem smallest_integer_sum_consecutive
  (l m n a : ℤ)
  (h1 : a = 9 * l + 36)
  (h2 : a = 10 * m + 45)
  (h3 : a = 11 * n + 55)
  : a = 495 :=
sorry

end smallest_integer_sum_consecutive_l215_215216


namespace total_area_needed_l215_215652

-- Definitions based on conditions
def oak_trees_first_half := 100
def pine_trees_first_half := 100
def oak_trees_second_half := 150
def pine_trees_second_half := 150
def oak_tree_planting_ratio := 4
def pine_tree_planting_ratio := 2
def oak_tree_space := 4
def pine_tree_space := 2

-- Total area needed for tree planting during the entire year
theorem total_area_needed : (oak_trees_first_half * oak_tree_planting_ratio * oak_tree_space) + ((pine_trees_first_half + pine_trees_second_half) * pine_tree_planting_ratio * pine_tree_space) = 2600 :=
by
  sorry

end total_area_needed_l215_215652


namespace coefficient_a2_l215_215621

theorem coefficient_a2 :
  ∀ (x : ℝ) (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ),
  (x^10 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + a_3 * (x + 1)^3 + 
  a_4 * (x + 1)^4 + a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
  a_7 * (x + 1)^7 + a_8 * (x + 1)^8 + a_9 * (x + 1)^9 + 
  a_10 * (x + 1)^10) →
  a_2 = 45 :=
by
  sorry

end coefficient_a2_l215_215621


namespace solve_for_y_l215_215443

theorem solve_for_y (y : ℝ) (h : 1 / 4 - 1 / 5 = 4 / y) : y = 80 :=
by
  sorry

end solve_for_y_l215_215443


namespace inversely_proportional_y_l215_215556

theorem inversely_proportional_y (k : ℚ) (x y : ℚ) (hx_neg_10 : x = -10) (hy_5 : y = 5) (hprop : y * x = k) (hx_neg_4 : x = -4) : 
  y = 25 / 2 := 
by
  sorry

end inversely_proportional_y_l215_215556


namespace melissa_total_points_l215_215306

-- Definition of the points scored per game and the number of games played.
def points_per_game : ℕ := 7
def number_of_games : ℕ := 3

-- The total points scored by Melissa is defined as the product of points per game and number of games.
def total_points_scored : ℕ := points_per_game * number_of_games

-- The theorem stating the verification of the total points scored by Melissa.
theorem melissa_total_points : total_points_scored = 21 := by
  -- The proof will be given here.
  sorry

end melissa_total_points_l215_215306


namespace problem_proof_l215_215940

-- Formalizing the conditions of the problem
variable {a : ℕ → ℝ}  -- Define the arithmetic sequence
variable (d : ℝ)      -- Common difference of the arithmetic sequence
variable (a₅ a₆ a₇ : ℝ)  -- Specific terms in the sequence

-- The condition given in the problem
axiom cond1 : a 5 + a 6 + a 7 = 15

-- A definition for an arithmetic sequence
noncomputable def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Using the axiom to deduce that a₆ = 5
axiom prop_arithmetic : is_arithmetic_seq a d

-- We want to prove that sum of terms from a₃ to a₉ = 35
theorem problem_proof : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by sorry

end problem_proof_l215_215940


namespace marks_in_physics_l215_215461

-- Definitions of the variables
variables (P C M : ℕ)

-- Conditions
def condition1 : Prop := P + C + M = 210
def condition2 : Prop := P + M = 180
def condition3 : Prop := P + C = 140

-- The statement to prove
theorem marks_in_physics (h1 : condition1 P C M) (h2 : condition2 P M) (h3 : condition3 P C) : P = 110 :=
sorry

end marks_in_physics_l215_215461


namespace sulfuric_acid_reaction_l215_215072

theorem sulfuric_acid_reaction (SO₃ H₂O H₂SO₄ : ℕ) 
  (reaction : SO₃ + H₂O = H₂SO₄)
  (H₂O_eq : H₂O = 2)
  (H₂SO₄_eq : H₂SO₄ = 2) :
  SO₃ = 2 :=
by
  sorry

end sulfuric_acid_reaction_l215_215072


namespace imaginary_unit_multiplication_l215_215603

-- Statement of the problem   
theorem imaginary_unit_multiplication (i : ℂ) (hi : i ^ 2 = -1) : i * (1 + i) = -1 + i :=
by sorry

end imaginary_unit_multiplication_l215_215603


namespace negation_of_p_l215_215470

open Real

-- Define the statement to be negated
def p := ∀ x : ℝ, -π/2 < x ∧ x < π/2 → tan x > 0

-- Define the negation of the statement
def not_p := ∃ x_0 : ℝ, -π/2 < x_0 ∧ x_0 < π/2 ∧ tan x_0 ≤ 0

-- Theorem stating that the negation of p is not_p
theorem negation_of_p : ¬ p ↔ not_p :=
sorry

end negation_of_p_l215_215470


namespace stock_price_end_of_third_year_l215_215112

def first_year_price (initial_price : ℝ) (first_year_increase : ℝ) : ℝ :=
  initial_price + (initial_price * first_year_increase)

def second_year_price (price_end_first : ℝ) (second_year_decrease : ℝ) : ℝ :=
  price_end_first - (price_end_first * second_year_decrease)

def third_year_price (price_end_second : ℝ) (third_year_increase : ℝ) : ℝ :=
  price_end_second + (price_end_second * third_year_increase)

theorem stock_price_end_of_third_year :
  ∀ (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) (third_year_increase : ℝ),
    initial_price = 150 →
    first_year_increase = 0.5 →
    second_year_decrease = 0.3 →
    third_year_increase = 0.2 →
    third_year_price (second_year_price (first_year_price initial_price first_year_increase) second_year_decrease) third_year_increase = 189 :=
by
  intros initial_price first_year_increase second_year_decrease third_year_increase
  sorry

end stock_price_end_of_third_year_l215_215112


namespace digit_sum_is_twelve_l215_215144

theorem digit_sum_is_twelve (n x y : ℕ) (h1 : n = 10 * x + y) (h2 : 0 ≤ x ∧ x ≤ 9) (h3 : 0 ≤ y ∧ y ≤ 9)
  (h4 : (1 / 2 : ℚ) * n = (1 / 4 : ℚ) * n + 3) : x + y = 12 :=
by
  sorry

end digit_sum_is_twelve_l215_215144


namespace angle_B_area_of_triangle_l215_215225

/-
Given a triangle ABC with angle A, B, C and sides a, b, c opposite to these angles respectively.
Consider the conditions:
- A = π/6
- b = (4 + 2 * sqrt 3) * a * cos B
- b = 1

Prove:
1. B = 5 * π / 12
2. The area of triangle ABC = 1 / 4
-/

namespace TriangleProof

open Real

def triangle_conditions (A B C a b c : ℝ) : Prop :=
  A = π / 6 ∧
  b = (4 + 2 * sqrt 3) * a * cos B ∧
  b = 1

theorem angle_B (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  B = 5 * π / 12 :=
sorry

theorem area_of_triangle (A B C a b c : ℝ) 
  (h : triangle_conditions A B C a b c) : 
  1 / 2 * b * c * sin A = 1 / 4 :=
sorry

end TriangleProof

end angle_B_area_of_triangle_l215_215225


namespace ray_reflection_and_distance_l215_215550

-- Define the initial conditions
def pointA : ℝ × ℝ := (-3, 3)
def circleC_eq (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Definitions of the lines for incident and reflected rays
def incident_ray_line (x y : ℝ) : Prop := 4*x + 3*y + 3 = 0
def reflected_ray_line (x y : ℝ) : Prop := 3*x + 4*y - 3 = 0

-- Distance traveled by the ray
def distance_traveled (A T : ℝ × ℝ) := 7

theorem ray_reflection_and_distance :
  ∃ (x₁ y₁ : ℝ), incident_ray_line x₁ y₁ ∧ reflected_ray_line x₁ y₁ ∧ circleC_eq x₁ y₁ ∧ 
  (∀ (P : ℝ × ℝ), P = pointA → distance_traveled P (x₁, y₁) = 7) :=
sorry

end ray_reflection_and_distance_l215_215550


namespace bottles_per_case_correct_l215_215638

-- Define the conditions given in the problem
def daily_bottle_production : ℕ := 120000
def number_of_cases_needed : ℕ := 10000

-- Define the expected answer
def bottles_per_case : ℕ := 12

-- The statement we need to prove
theorem bottles_per_case_correct :
  daily_bottle_production / number_of_cases_needed = bottles_per_case :=
by
  -- Leap of logic: actually solving this for correctness is here considered a leap
  sorry

end bottles_per_case_correct_l215_215638


namespace meeting_point_l215_215886

theorem meeting_point :
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  (Paul_start.1 + Lisa_start.1) / 2 = -2 ∧ (Paul_start.2 + Lisa_start.2) / 2 = 3 :=
by
  let Paul_start := (3, 9)
  let Lisa_start := (-7, -3)
  have x_coord : (Paul_start.1 + Lisa_start.1) / 2 = -2 := sorry
  have y_coord : (Paul_start.2 + Lisa_start.2) / 2 = 3 := sorry
  exact ⟨x_coord, y_coord⟩

end meeting_point_l215_215886


namespace inequality_pos_xy_l215_215904

theorem inequality_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
    (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := 
by {
    sorry
}

end inequality_pos_xy_l215_215904


namespace craig_apples_after_sharing_l215_215159

-- Defining the initial conditions
def initial_apples_craig : ℕ := 20
def shared_apples : ℕ := 7

-- The proof statement
theorem craig_apples_after_sharing : 
  initial_apples_craig - shared_apples = 13 := 
by
  sorry

end craig_apples_after_sharing_l215_215159


namespace avg_weight_of_children_is_138_l215_215515

-- Define the average weight of boys and girls
def average_weight_of_boys := 150
def number_of_boys := 6
def average_weight_of_girls := 120
def number_of_girls := 4

-- Calculate total weights and average weight of all children
noncomputable def total_weight_of_boys := number_of_boys * average_weight_of_boys
noncomputable def total_weight_of_girls := number_of_girls * average_weight_of_girls
noncomputable def total_weight_of_children := total_weight_of_boys + total_weight_of_girls
noncomputable def number_of_children := number_of_boys + number_of_girls
noncomputable def average_weight_of_children := total_weight_of_children / number_of_children

-- Lean statement to prove the average weight of all children is 138 pounds
theorem avg_weight_of_children_is_138 : average_weight_of_children = 138 := by
    sorry

end avg_weight_of_children_is_138_l215_215515


namespace decrement_value_each_observation_l215_215736

theorem decrement_value_each_observation 
  (n : ℕ) 
  (original_mean updated_mean : ℝ) 
  (n_pos : n = 50) 
  (original_mean_value : original_mean = 200)
  (updated_mean_value : updated_mean = 153) :
  (original_mean * n - updated_mean * n) / n = 47 :=
by
  sorry

end decrement_value_each_observation_l215_215736


namespace fraction_add_eq_l215_215696

theorem fraction_add_eq (n : ℤ) :
  (3 + n) = 4 * ((4 + n) - 5) → n = 1 := sorry

end fraction_add_eq_l215_215696


namespace sum_of_integral_c_l215_215095

theorem sum_of_integral_c :
  let discriminant (a b c : ℤ) := b * b - 4 * a * c
  ∃ (valid_c : List ℤ),
    (∀ c ∈ valid_c, c ≤ 30 ∧ ∃ k : ℤ, discriminant 1 (-9) (c) = k * k ∧ k > 0) ∧
    valid_c.sum = 32 := 
by
  sorry

end sum_of_integral_c_l215_215095


namespace proposition_b_proposition_d_l215_215109

-- Proposition B: For a > 0 and b > 0, if ab = 2, then the minimum value of a + 2b is 4
theorem proposition_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = 2) : a + 2 * b ≥ 4 :=
  sorry

-- Proposition D: For a > 0 and b > 0, if a² + b² = 1, then the maximum value of a + b is sqrt(2).
theorem proposition_d (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + b^2 = 1) : a + b ≤ Real.sqrt 2 :=
  sorry

end proposition_b_proposition_d_l215_215109


namespace distinct_xy_values_l215_215264

theorem distinct_xy_values : ∃ (xy_values : Finset ℕ), 
  (∀ (x y : ℕ), (0 < x ∧ 0 < y) → (1 / Real.sqrt x + 1 / Real.sqrt y = 1 / Real.sqrt 20) → (xy_values = {8100, 6400})) ∧
  (xy_values.card = 2) :=
by
  sorry

end distinct_xy_values_l215_215264


namespace scientific_notation_15510000_l215_215295

theorem scientific_notation_15510000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 15510000 = a * 10^n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_15510000_l215_215295


namespace leak_empty_time_l215_215482

-- Define the given conditions
def tank_volume := 2160 -- Tank volume in litres
def inlet_rate := 6 * 60 -- Inlet rate in litres per hour
def combined_empty_time := 12 -- Time in hours to empty the tank with the inlet on

-- Define the derived conditions
def net_rate := tank_volume / combined_empty_time -- Net rate of emptying in litres per hour

-- Define the rate of leakage
def leak_rate := inlet_rate - net_rate -- Rate of leak in litres per hour

-- Prove the main statement
theorem leak_empty_time : (2160 / leak_rate) = 12 :=
by
  unfold leak_rate
  exact sorry

end leak_empty_time_l215_215482


namespace determine_quarters_given_l215_215167

def total_initial_coins (dimes quarters nickels : ℕ) : ℕ :=
  dimes + quarters + nickels

def updated_dimes (original_dimes added_dimes : ℕ) : ℕ :=
  original_dimes + added_dimes

def updated_nickels (original_nickels factor : ℕ) : ℕ :=
  original_nickels + original_nickels * factor

def total_coins_after_addition (dimes quarters nickels : ℕ) (added_dimes added_quarters added_nickels_factor : ℕ) : ℕ :=
  updated_dimes dimes added_dimes +
  (quarters + added_quarters) +
  updated_nickels nickels added_nickels_factor

def quarters_given_by_mother (total_coins initial_dimes initial_quarters initial_nickels added_dimes added_nickels_factor : ℕ) : ℕ :=
  total_coins - total_initial_coins initial_dimes initial_quarters initial_nickels - added_dimes - initial_nickels * added_nickels_factor

theorem determine_quarters_given :
  quarters_given_by_mother 35 2 6 5 2 2 = 10 :=
by
  sorry

end determine_quarters_given_l215_215167


namespace find_expression_l215_215082

variable (a b E : ℝ)

-- Conditions
def condition1 := a / b = 4 / 3
def condition2 := E / (3 * a - 2 * b) = 3

-- Conclusion we want to prove
theorem find_expression : condition1 a b → condition2 a b E → E = 6 * b :=
by
  intro h1 h2
  sorry

end find_expression_l215_215082


namespace P_eq_Q_l215_215298

def P (m : ℝ) : Prop := -1 < m ∧ m < 0

def quadratic_inequality (m : ℝ) (x : ℝ) : Prop := m * x^2 + 4 * m * x - 4 < 0

def Q (m : ℝ) : Prop := ∀ x : ℝ, quadratic_inequality m x

theorem P_eq_Q : ∀ m : ℝ, P m ↔ Q m := 
by 
  sorry

end P_eq_Q_l215_215298


namespace expression_defined_if_x_not_3_l215_215609

theorem expression_defined_if_x_not_3 (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end expression_defined_if_x_not_3_l215_215609


namespace find_sum_f_neg1_f_3_l215_215798

noncomputable def f : ℝ → ℝ := sorry

-- condition: odd function
def odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = - f x

-- condition: symmetry around x=1
def symmetric_around_one (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (1 - x) = f (1 + x)

-- condition: specific value at x=1
def value_at_one (f : ℝ → ℝ) : Prop := f 1 = 2

-- Theorem to prove
theorem find_sum_f_neg1_f_3 (h1 : odd_function f) (h2 : symmetric_around_one f) (h3 : value_at_one f) : f (-1) + f 3 = -4 := by
  sorry

end find_sum_f_neg1_f_3_l215_215798


namespace unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l215_215645

theorem unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2 : ∃! (x : ℤ), x - 9 / (x - 2) = 5 - 9 / (x - 2) := 
by
  sorry

end unique_integral_root_x_minus_9_over_x_minus_2_eq_5_minus_9_over_x_minus_2_l215_215645


namespace gcd_12a_18b_l215_215071

theorem gcd_12a_18b (a b : ℕ) (h : Nat.gcd a b = 12) : Nat.gcd (12 * a) (18 * b) = 72 :=
sorry

end gcd_12a_18b_l215_215071


namespace inverse_variation_example_l215_215269

theorem inverse_variation_example
  (k : ℝ)
  (h1 : ∀ (c d : ℝ), (c^2) * (d^4) = k)
  (h2 : ∃ (c : ℝ), c = 8 ∧ (∀ (d : ℝ), d = 2 → (c^2) * (d^4) = k)) : 
  (∀ (d : ℝ), d = 4 → (∃ (c : ℝ), (c^2) = 4)) := 
by 
  sorry

end inverse_variation_example_l215_215269


namespace cost_of_five_dozen_l215_215468

noncomputable def price_per_dozen (total_cost : ℝ) (num_dozen : ℕ) : ℝ :=
  total_cost / num_dozen

noncomputable def total_cost (price_per_dozen : ℝ) (num_dozen : ℕ) : ℝ :=
  price_per_dozen * num_dozen

theorem cost_of_five_dozen (total_cost_threedozens : ℝ := 28.20) (num_threedozens : ℕ := 3) (num_fivedozens : ℕ := 5) :
  total_cost (price_per_dozen total_cost_threedozens num_threedozens) num_fivedozens = 47.00 :=
  by sorry

end cost_of_five_dozen_l215_215468


namespace positive_difference_l215_215772

theorem positive_difference (x y : ℝ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 14) : y - x = 9.714 :=
sorry

end positive_difference_l215_215772


namespace probability_of_shaded_shape_l215_215106

   def total_shapes : ℕ := 4
   def shaded_shapes : ℕ := 1

   theorem probability_of_shaded_shape : shaded_shapes / total_shapes = 1 / 4 := 
   by
     sorry
   
end probability_of_shaded_shape_l215_215106


namespace difference_pencils_l215_215601

theorem difference_pencils (x : ℕ) (h1 : 162 = x * n_g) (h2 : 216 = x * n_f) : n_f - n_g = 3 :=
by
  sorry

end difference_pencils_l215_215601


namespace priyas_speed_is_30_l215_215981

noncomputable def find_priyas_speed (v : ℝ) : Prop :=
  let riya_speed := 20
  let time := 0.5  -- in hours
  let distance_apart := 25
  (riya_speed + v) * time = distance_apart

theorem priyas_speed_is_30 : ∃ v : ℝ, find_priyas_speed v ∧ v = 30 :=
by
  sorry

end priyas_speed_is_30_l215_215981


namespace inverse_proportion_value_of_m_l215_215375

theorem inverse_proportion_value_of_m (m : ℤ) (x : ℝ) (y : ℝ) : 
  y = (m - 2) * x ^ (m^2 - 5) → (m = -2) := 
by
  sorry

end inverse_proportion_value_of_m_l215_215375


namespace lemonade_sales_l215_215028

theorem lemonade_sales (total_amount small_amount medium_amount large_price sales_price_small sales_price_medium earnings_small earnings_medium : ℕ) (h1 : total_amount = 50) (h2 : sales_price_small = 1) (h3 : sales_price_medium = 2) (h4 : large_price = 3) (h5 : earnings_small = 11) (h6 : earnings_medium = 24) : large_amount = 5 :=
by
  sorry

end lemonade_sales_l215_215028


namespace hot_dogs_left_over_l215_215814

theorem hot_dogs_left_over : 25197629 % 6 = 5 := 
sorry

end hot_dogs_left_over_l215_215814


namespace sqrt_range_real_l215_215000

theorem sqrt_range_real (x : ℝ) (h : 1 - 3 * x ≥ 0) : x ≤ 1 / 3 :=
sorry

end sqrt_range_real_l215_215000


namespace find_train_speed_l215_215078

-- Define the given conditions
def train_length : ℕ := 2500  -- length of the train in meters
def time_to_cross_pole : ℕ := 100  -- time to cross the pole in seconds

-- Define the expected speed
def expected_speed : ℕ := 25  -- expected speed in meters per second

-- The theorem we need to prove
theorem find_train_speed : 
  (train_length / time_to_cross_pole) = expected_speed := 
by 
  sorry

end find_train_speed_l215_215078


namespace total_distance_l215_215156

theorem total_distance (D : ℝ) 
  (h1 : 1/4 * (3/8 * D) = 210) : D = 840 := 
by
  -- proof steps would go here
  sorry

end total_distance_l215_215156


namespace least_number_remainder_l215_215528

theorem least_number_remainder (n : ℕ) (hn : n = 115) : n % 38 = 1 ∧ n % 3 = 1 := by
  sorry

end least_number_remainder_l215_215528


namespace combined_average_age_l215_215147

-- Definitions based on given conditions
def num_fifth_graders : ℕ := 28
def avg_age_fifth_graders : ℝ := 10
def num_parents : ℕ := 45
def avg_age_parents : ℝ := 40

-- The statement to prove
theorem combined_average_age : (num_fifth_graders * avg_age_fifth_graders + num_parents * avg_age_parents) / (num_fifth_graders + num_parents) = 28.49 :=
  by
  sorry

end combined_average_age_l215_215147


namespace harmonic_mean_average_of_x_is_11_l215_215284

theorem harmonic_mean_average_of_x_is_11 :
  let h := (2 * 1008) / (2 + 1008)
  ∃ (x : ℕ), (h + x) / 2 = 11 → x = 18 := by
  sorry

end harmonic_mean_average_of_x_is_11_l215_215284


namespace rectangular_solid_diagonal_l215_215929

theorem rectangular_solid_diagonal (p q r : ℝ) (d : ℝ) :
  p^2 + q^2 + r^2 = d^2 :=
sorry

end rectangular_solid_diagonal_l215_215929


namespace tom_total_expenditure_l215_215906

noncomputable def tom_spent_total : ℝ :=
  let skateboard_price := 9.46
  let skateboard_discount := 0.10 * skateboard_price
  let discounted_skateboard := skateboard_price - skateboard_discount

  let marbles_price := 9.56
  let marbles_discount := 0.10 * marbles_price
  let discounted_marbles := marbles_price - marbles_discount

  let shorts_price := 14.50

  let figures_price := 12.60
  let figures_discount := 0.20 * figures_price
  let discounted_figures := figures_price - figures_discount

  let puzzle_price := 6.35
  let puzzle_discount := 0.15 * puzzle_price
  let discounted_puzzle := puzzle_price - puzzle_discount

  let game_price_eur := 20.50
  let game_discount_eur := 0.05 * game_price_eur
  let discounted_game_eur := game_price_eur - game_discount_eur
  let exchange_rate := 1.12
  let discounted_game_usd := discounted_game_eur * exchange_rate

  discounted_skateboard + discounted_marbles + shorts_price + discounted_figures + discounted_puzzle + discounted_game_usd

theorem tom_total_expenditure : abs (tom_spent_total - 68.91) < 0.01 :=
by norm_num1; sorry

end tom_total_expenditure_l215_215906


namespace hypotenuse_eq_medians_l215_215843

noncomputable def hypotenuse_length_medians (a b : ℝ) (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) : ℝ :=
  3 * Real.sqrt (336 / 13)

-- definition
theorem hypotenuse_eq_medians {a b : ℝ} (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) :
    Real.sqrt (9 * (a^2 + b^2)) = 3 * Real.sqrt (336 / 13) :=
sorry

end hypotenuse_eq_medians_l215_215843


namespace value_division_l215_215207

theorem value_division (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) 
                       (h3 : x / y = n) : n = 4 := 
by 
sorry

end value_division_l215_215207


namespace sqrt_div_add_l215_215203

theorem sqrt_div_add :
  let sqrt_0_81 := 0.9
  let sqrt_1_44 := 1.2
  let sqrt_0_49 := 0.7
  (Real.sqrt 1.1 / sqrt_0_81) + (sqrt_1_44 / sqrt_0_49) = 2.8793 :=
by
  -- Prove equality using the given conditions
  sorry

end sqrt_div_add_l215_215203


namespace average_age_of_adults_l215_215467

theorem average_age_of_adults 
  (total_members : ℕ)
  (avg_age_total : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_adults : ℕ)
  (avg_age_girls : ℕ)
  (avg_age_boys : ℕ)
  (total_sum_ages : ℕ := total_members * avg_age_total)
  (sum_ages_girls : ℕ := num_girls * avg_age_girls)
  (sum_ages_boys : ℕ := num_boys * avg_age_boys)
  (sum_ages_adults : ℕ := total_sum_ages - sum_ages_girls - sum_ages_boys)
  : (num_adults = 10) → (avg_age_total = 20) → (num_girls = 30) → (avg_age_girls = 18) → (num_boys = 20) → (avg_age_boys = 22) → (total_sum_ages = 1200) → (sum_ages_girls = 540) → (sum_ages_boys = 440) → (sum_ages_adults = 220) → (sum_ages_adults / num_adults = 22) :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10
  sorry

end average_age_of_adults_l215_215467


namespace other_asymptote_of_hyperbola_l215_215226

theorem other_asymptote_of_hyperbola (a b : ℝ) :
  (∀ x : ℝ, a * x + b = 2 * x) →
  (∀ p : ℝ × ℝ, (p.1 = 3)) →
  ∀ (c : ℝ × ℝ), (c.1 = 3 ∧ c.2 = 6) ->
  ∃ (m : ℝ), m = -1/2 ∧ (∀ x, c.2 = -1/2 * x + 15/2) :=
by
  sorry

end other_asymptote_of_hyperbola_l215_215226


namespace green_pill_cost_is_21_l215_215513

-- Definitions based on conditions
def number_of_days : ℕ := 21
def total_cost : ℕ := 819
def daily_cost : ℕ := total_cost / number_of_days
def green_pill_cost (pink_pill_cost : ℕ) : ℕ := pink_pill_cost + 3

-- Given pink pill cost is x, then green pill cost is x + 3
-- We need to prove that for some x, the daily cost of the pills equals 39, and thus green pill cost is 21

theorem green_pill_cost_is_21 (pink_pill_cost : ℕ) (h : daily_cost = (green_pill_cost pink_pill_cost) + pink_pill_cost) :
    green_pill_cost pink_pill_cost = 21 :=
by
  sorry

end green_pill_cost_is_21_l215_215513


namespace interesting_seven_digit_numbers_l215_215911

theorem interesting_seven_digit_numbers :
  ∃ n : Fin 2 → ℕ, (∀ i : Fin 2, n i = 128) :=
by sorry

end interesting_seven_digit_numbers_l215_215911


namespace wire_divided_into_quarters_l215_215099

theorem wire_divided_into_quarters
  (l : ℕ) -- length of the wire
  (parts : ℕ) -- number of parts the wire is divided into
  (h_l : l = 28) -- wire is 28 cm long
  (h_parts : parts = 4) -- wire is divided into 4 parts
  : l / parts = 7 := -- each part is 7 cm long
by
  -- use sorry to skip the proof
  sorry

end wire_divided_into_quarters_l215_215099


namespace John_l215_215895

theorem John's_earnings_on_Saturday :
  ∃ S : ℝ, (S + S / 2 + 20 = 47) ∧ (S = 18) := by
    sorry

end John_l215_215895


namespace females_with_advanced_degrees_l215_215567

theorem females_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_advanced_degrees : ℕ)
  (males_college_degree_only : ℕ)
  (h1 : total_employees = 200)
  (h2 : total_females = 120)
  (h3 : total_advanced_degrees = 100)
  (h4 : males_college_degree_only = 40) :
  (total_advanced_degrees - (total_employees - total_females - males_college_degree_only) = 60) :=
by
  -- proof will go here
  sorry

end females_with_advanced_degrees_l215_215567


namespace find_least_positive_n_l215_215435

theorem find_least_positive_n (n : ℕ) : 
  let m := 143
  m = 11 * 13 → 
  (3^5 ≡ 1 [MOD m^2]) →
  (3^39 ≡ 1 [MOD (13^2)]) →
  n = 195 :=
sorry

end find_least_positive_n_l215_215435


namespace B_more_cost_effective_l215_215222

variable (x y : ℝ)
variable (hx : x ≠ y)

theorem B_more_cost_effective (x y : ℝ) (hx : x ≠ y) :
  (1/2 * x + 1/2 * y) > (2 * x * y / (x + y)) :=
by
  sorry

end B_more_cost_effective_l215_215222


namespace max_area_of_rectangle_l215_215001

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 36) : (x * y) ≤ 81 :=
sorry

end max_area_of_rectangle_l215_215001


namespace systems_solution_l215_215142

    theorem systems_solution : 
      (∃ x y : ℝ, 2 * x + 5 * y = -26 ∧ 3 * x - 5 * y = 36 ∧ 
                 (∃ a b : ℝ, a * x - b * y = -4 ∧ b * x + a * y = -8 ∧ 
                 (2 * a + b) ^ 2020 = 1)) := 
    by
      sorry
    
end systems_solution_l215_215142


namespace total_shaded_area_l215_215680

theorem total_shaded_area (S T : ℝ) (h1 : 12 / S = 4) (h2 : S / T = 4) :
  1 * S ^ 2 + 8 * (T ^ 2) = 13.5 := by
  sorry

end total_shaded_area_l215_215680


namespace find_side_length_a_l215_215161

variable {a b c : ℝ}
variable {B : ℝ}

theorem find_side_length_a (h_b : b = 7) (h_c : c = 5) (h_B : B = 2 * Real.pi / 3) :
  a = 3 :=
sorry

end find_side_length_a_l215_215161


namespace maximum_S_n_l215_215223

noncomputable def a_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  a_1 + (n - 1) * d

noncomputable def S_n (a_1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a_1 + (n - 1) * d)

theorem maximum_S_n (a_1 : ℝ) (h : a_1 > 0)
  (h_sequence : 3 * a_n a_1 (2 * a_1 / 39) 8 = 5 * a_n a_1 (2 * a_1 / 39) 13)
  : ∀ n : ℕ, S_n a_1 (2 * a_1 / 39) n ≤ S_n a_1 (2 * a_1 / 39) 20 :=
sorry

end maximum_S_n_l215_215223


namespace closest_integer_to_cube_root_of_150_l215_215519

theorem closest_integer_to_cube_root_of_150 : ∃ (n : ℤ), abs ((n: ℝ)^3 - 150) ≤ abs (((n + 1 : ℤ) : ℝ)^3 - 150) ∧
  abs ((n: ℝ)^3 - 150) ≤ abs (((n - 1 : ℤ) : ℝ)^3 - 150) ∧ n = 5 :=
by
  sorry

end closest_integer_to_cube_root_of_150_l215_215519


namespace smallest_root_of_g_l215_215259

-- Define the function g(x)
def g (x : ℝ) : ℝ := 10 * x^4 - 17 * x^2 + 7

-- The main statement: proving the smallest root of g(x) is -sqrt(7/5)
theorem smallest_root_of_g : ∃ x : ℝ, g x = 0 ∧ ∀ y : ℝ, g y = 0 → x ≤ y := 
sorry

end smallest_root_of_g_l215_215259


namespace transform_cos_to_base_form_l215_215947

theorem transform_cos_to_base_form :
  let f (x : ℝ) := Real.cos (2 * x + (Real.pi / 3))
  let g (x : ℝ) := Real.cos (2 * x)
  ∃ (shift : ℝ), shift = Real.pi / 6 ∧
    (∀ x : ℝ, f (x - shift) = g x) :=
by
  let f := λ x : ℝ => Real.cos (2 * x + (Real.pi / 3))
  let g := λ x : ℝ => Real.cos (2 * x)
  use Real.pi / 6
  sorry

end transform_cos_to_base_form_l215_215947


namespace flight_distance_l215_215477

theorem flight_distance (D : ℝ) :
  let t_out := D / 300
  let t_return := D / 500
  t_out + t_return = 8 -> D = 1500 :=
by
  intro h
  sorry

end flight_distance_l215_215477


namespace problem1_problem2_l215_215086

-- Problem 1
theorem problem1 (x : ℝ) (h1 : 2 * x > 1 - x) (h2 : x + 2 < 4 * x - 1) : x > 1 := 
by
  sorry

-- Problem 2
theorem problem2 (x : ℝ)
  (h1 : (2 / 3) * x + 5 > 1 - x)
  (h2 : x - 1 ≤ (3 / 4) * x - 1 / 8) :
  -12 / 5 < x ∧ x ≤ 7 / 2 := 
by
  sorry

end problem1_problem2_l215_215086


namespace methane_tetrahedron_dot_product_l215_215921

noncomputable def tetrahedron_vectors_dot_product_sum : ℝ :=
  let edge_length := 1
  let dot_product := -1 / 3 * edge_length^2
  let pair_count := 6 -- number of pairs in sum of dot products
  pair_count * dot_product

theorem methane_tetrahedron_dot_product :
  tetrahedron_vectors_dot_product_sum = - (3 / 4) := by
  sorry

end methane_tetrahedron_dot_product_l215_215921


namespace dreamy_bookstore_sales_l215_215351

theorem dreamy_bookstore_sales :
  let total_sales_percent := 100
  let notebooks_percent := 45
  let bookmarks_percent := 25
  let neither_notebooks_nor_bookmarks_percent := total_sales_percent - (notebooks_percent + bookmarks_percent)
  neither_notebooks_nor_bookmarks_percent = 30 :=
by {
  sorry
}

end dreamy_bookstore_sales_l215_215351


namespace five_minus_a_l215_215824

theorem five_minus_a (a b : ℚ) (h1 : 5 + a = 3 - b) (h2 : 3 + b = 8 + a) : 5 - a = 17/2 :=
by
  sorry

end five_minus_a_l215_215824


namespace quadratic_vertex_position_l215_215328

theorem quadratic_vertex_position (a p q m : ℝ) (ha : 0 < a) (hpq : p < q) (hA : p = a * (-1 - m)^2) (hB : q = a * (3 - m)^2) : m ≠ 2 :=
by
  sorry

end quadratic_vertex_position_l215_215328


namespace molecular_weight_acetic_acid_l215_215684

-- Define atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of each atom in acetic acid
def num_C : ℕ := 2
def num_H : ℕ := 4
def num_O : ℕ := 2

-- Define the molecular formula of acetic acid
def molecular_weight_CH3COOH : ℝ :=
  num_C * atomic_weight_C +
  num_H * atomic_weight_H +
  num_O * atomic_weight_O

-- State the proposition
theorem molecular_weight_acetic_acid :
  molecular_weight_CH3COOH = 60.052 := by
  sorry

end molecular_weight_acetic_acid_l215_215684


namespace range_of_a_l215_215678

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - a| + |x - 2| ≥ 1) → (a ≤ 1 ∨ a ≥ 3) :=
sorry

end range_of_a_l215_215678


namespace combined_weight_of_new_students_l215_215607

theorem combined_weight_of_new_students 
  (avg_weight_orig : ℝ) (num_students_orig : ℝ) 
  (new_avg_weight : ℝ) (num_new_students : ℝ) 
  (total_weight_gain_orig : ℝ) (total_weight_loss_orig : ℝ)
  (total_weight_orig : ℝ := avg_weight_orig * num_students_orig) 
  (net_weight_change_orig : ℝ := total_weight_gain_orig - total_weight_loss_orig)
  (total_weight_after_change_orig : ℝ := total_weight_orig + net_weight_change_orig) 
  (total_students_after : ℝ := num_students_orig + num_new_students) 
  (total_weight_class_after : ℝ := new_avg_weight * total_students_after) : 
  total_weight_class_after - total_weight_after_change_orig = 586 :=
by
  sorry

end combined_weight_of_new_students_l215_215607


namespace no_intersection_at_roots_l215_215975

theorem no_intersection_at_roots {f g : ℝ → ℝ} (h : ∀ x, f x = x ∧ g x = x - 3) :
  ¬ (∃ x, (x = 0 ∨ x = 3) ∧ (f x = g x)) :=
by
  intros 
  sorry

end no_intersection_at_roots_l215_215975


namespace abs_diff_of_slopes_l215_215117

theorem abs_diff_of_slopes (k1 k2 b : ℝ) (h : k1 * k2 < 0) (area_cond : (1 / 2) * 3 * |k1 - k2| * 3 = 9) :
  |k1 - k2| = 2 :=
by
  sorry

end abs_diff_of_slopes_l215_215117


namespace distance_proof_l215_215511

-- Define the speeds of Alice and Bob
def aliceSpeed : ℚ := 1 / 20 -- Alice's speed in miles per minute
def bobSpeed : ℚ := 3 / 40 -- Bob's speed in miles per minute

-- Define the time they walk/jog
def time : ℚ := 120 -- Time in minutes (2 hours)

-- Calculate the distances
def aliceDistance : ℚ := aliceSpeed * time -- Distance Alice walked
def bobDistance : ℚ := bobSpeed * time -- Distance Bob jogged

-- The total distance between Alice and Bob after 2 hours
def totalDistance : ℚ := aliceDistance + bobDistance

-- Prove that the total distance is 15 miles
theorem distance_proof : totalDistance = 15 := by
  sorry

end distance_proof_l215_215511


namespace female_salmon_returned_l215_215089

theorem female_salmon_returned :
  let total_salmon : ℕ := 971639
  let male_salmon : ℕ := 712261
  total_salmon - male_salmon = 259378 :=
by
  let total_salmon := 971639
  let male_salmon := 712261
  calc
    971639 - 712261 = 259378 := by norm_num

end female_salmon_returned_l215_215089


namespace Noemi_blackjack_loss_l215_215700

-- Define the conditions
def start_amount : ℕ := 1700
def end_amount : ℕ := 800
def roulette_loss : ℕ := 400

-- Define the total loss calculation
def total_loss : ℕ := start_amount - end_amount

-- Main theorem statement
theorem Noemi_blackjack_loss :
  ∃ (blackjack_loss : ℕ), blackjack_loss = total_loss - roulette_loss := 
by
  -- Start by calculating the total_loss
  let total_loss_eq := start_amount - end_amount
  -- The blackjack loss should be 900 - 400, which we claim to be 500
  use total_loss_eq - roulette_loss
  sorry

end Noemi_blackjack_loss_l215_215700


namespace smallest_largest_multiples_l215_215417

theorem smallest_largest_multiples : 
  ∃ l g, l >= 10 ∧ l < 100 ∧ g >= 100 ∧ g < 1000 ∧
  (2 ∣ l) ∧ (3 ∣ l) ∧ (5 ∣ l) ∧ 
  (2 ∣ g) ∧ (3 ∣ g) ∧ (5 ∣ g) ∧
  (∀ n, n >= 10 ∧ n < 100 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → l ≤ n) ∧
  (∀ n, n >= 100 ∧ n < 1000 ∧ (2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n) → g >= n) ∧
  l = 30 ∧ g = 990 := 
by 
  sorry

end smallest_largest_multiples_l215_215417


namespace find_integer_mod_condition_l215_215835

theorem find_integer_mod_condition (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 4) (h3 : n ≡ -998 [ZMOD 5]) : n = 2 :=
sorry

end find_integer_mod_condition_l215_215835


namespace find_larger_number_l215_215087

theorem find_larger_number (L S : ℤ) (h1 : L - S = 1365) (h2 : L = 6 * S + 5) : L = 1637 :=
by
  sorry

end find_larger_number_l215_215087


namespace dan_initial_money_l215_215116

def money_left : ℕ := 3
def cost_candy : ℕ := 2
def initial_money : ℕ := money_left + cost_candy

theorem dan_initial_money :
  initial_money = 5 :=
by
  -- Definitions according to problem
  let money_left := 3
  let cost_candy := 2

  have h : initial_money = money_left + cost_candy := by rfl
  rw [h]

  -- Show the final equivalence
  show 3 + 2 = 5
  rfl

end dan_initial_money_l215_215116


namespace smallest_value_of_x_l215_215059

theorem smallest_value_of_x :
  ∃ x : Real, (∀ z, (z = (5 * x - 20) / (4 * x - 5)) → (z * z + z = 20)) → x = 0 :=
by
  sorry

end smallest_value_of_x_l215_215059


namespace Lynne_bought_3_magazines_l215_215530

open Nat

def books_about_cats : Nat := 7
def books_about_solar_system : Nat := 2
def book_cost : Nat := 7
def magazine_cost : Nat := 4
def total_spent : Nat := 75

theorem Lynne_bought_3_magazines:
  let total_books := books_about_cats + books_about_solar_system
  let total_cost_books := total_books * book_cost
  let total_cost_magazines := total_spent - total_cost_books
  total_cost_magazines / magazine_cost = 3 :=
by sorry

end Lynne_bought_3_magazines_l215_215530


namespace cookie_portion_l215_215771

theorem cookie_portion :
  ∃ (total_cookies : ℕ) (remaining_cookies : ℕ) (cookies_senior_ate : ℕ) (cookies_senior_took_second_day : ℕ) 
    (cookies_senior_put_back : ℕ) (cookies_junior_took : ℕ),
  total_cookies = 22 ∧
  remaining_cookies = 11 ∧
  cookies_senior_ate = 3 ∧
  cookies_senior_took_second_day = 3 ∧
  cookies_senior_put_back = 2 ∧
  cookies_junior_took = 7 ∧
  4 / 22 = 2 / 11 :=
by
  existsi 22, 11, 3, 3, 2, 7
  sorry

end cookie_portion_l215_215771


namespace cos_sum_formula_l215_215569

open Real

theorem cos_sum_formula (A B C : ℝ) (h1 : sin A + sin B + sin C = 0) (h2 : cos A + cos B + cos C = 0) :
  cos (A - B) + cos (B - C) + cos (C - A) = -3 / 2 :=
by
  sorry

end cos_sum_formula_l215_215569


namespace range_of_x_for_f_ln_x_gt_f_1_l215_215379

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

noncomputable def is_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_x_for_f_ln_x_gt_f_1
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_dec : is_decreasing_on_nonneg f)
  (hf_condition : ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e) :
  ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e := sorry

end range_of_x_for_f_ln_x_gt_f_1_l215_215379


namespace digit_five_occurrences_l215_215872

variable (fives_ones fives_tens fives_hundreds : ℕ)

def count_fives := fives_ones + fives_tens + fives_hundreds

theorem digit_five_occurrences :
  ( ∀ (fives_ones fives_tens fives_hundreds : ℕ), 
    fives_ones = 100 ∧ fives_tens = 100 ∧ fives_hundreds = 100 → 
    count_fives fives_ones fives_tens fives_hundreds = 300 ) :=
by
  sorry

end digit_five_occurrences_l215_215872


namespace green_hats_count_l215_215958

theorem green_hats_count 
  (B G : ℕ)
  (h1 : B + G = 85)
  (h2 : 6 * B + 7 * G = 530) : 
  G = 20 :=
by
  sorry

end green_hats_count_l215_215958


namespace problem_l215_215246

theorem problem (a b c d : ℝ) (h1 : b + c = 7) (h2 : c + d = 5) (h3 : a + d = 2) : a + b = 4 :=
sorry

end problem_l215_215246


namespace factorization_of_polynomial_l215_215978

theorem factorization_of_polynomial : ∀ x : ℝ, x^2 - x - 42 = (x + 6) * (x - 7) :=
by
  sorry

end factorization_of_polynomial_l215_215978


namespace quadratic_roots_range_no_real_k_for_reciprocal_l215_215158

theorem quadratic_roots_range (k : ℝ) (h : 12 * k + 4 > 0) : k > -1 / 3 ∧ k ≠ 0 :=
by
  sorry

theorem no_real_k_for_reciprocal (k : ℝ) : ¬∃ (x1 x2 : ℝ), (kx^2 - 2*(k+1)*x + k-1 = 0) ∧ (1/x1 + 1/x2 = 0) :=
by
  sorry

end quadratic_roots_range_no_real_k_for_reciprocal_l215_215158


namespace propositions_using_logical_connectives_l215_215140

-- Define each of the propositions.
def prop1 := "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def prop2 := "Multiples of 10 are definitely multiples of 5."
def prop3 := "A trapezoid is not a rectangle."
def prop4 := "The solutions to the equation x^2 = 1 are x = ± 1."

-- Define logical connectives usage.
def uses_and (s : String) : Prop := 
  s = "October 1, 2004, is the National Day and also the Mid-Autumn Festival."
def uses_not (s : String) : Prop := 
  s = "A trapezoid is not a rectangle."
def uses_or (s : String) : Prop := 
  s = "The solutions to the equation x^2 = 1 are x = ± 1."

-- The lean theorem stating the propositions that use logical connectives
theorem propositions_using_logical_connectives :
  (uses_and prop1) ∧ (¬ uses_and prop2) ∧ (uses_not prop3) ∧ (uses_or prop4) := 
by
  sorry

end propositions_using_logical_connectives_l215_215140


namespace mosquito_shadow_speed_l215_215444

theorem mosquito_shadow_speed
  (v : ℝ) (t : ℝ) (h : ℝ) (cos_theta : ℝ) (v_shadow : ℝ)
  (hv : v = 0.5) (ht : t = 20) (hh : h = 6) (hcos_theta : cos_theta = 0.6) :
  v_shadow = 0 ∨ v_shadow = 0.8 :=
  sorry

end mosquito_shadow_speed_l215_215444


namespace sum_largest_smallest_prime_factors_1155_l215_215898

theorem sum_largest_smallest_prime_factors_1155 : 
  ∃ smallest largest : ℕ, 
  smallest ∣ 1155 ∧ largest ∣ 1155 ∧ 
  Prime smallest ∧ Prime largest ∧ 
  smallest <= largest ∧ 
  (∀ p : ℕ, p ∣ 1155 → Prime p → (smallest ≤ p ∧ p ≤ largest)) ∧ 
  (smallest + largest = 14) := 
by {
  sorry
}

end sum_largest_smallest_prime_factors_1155_l215_215898


namespace solve_equation_l215_215720

theorem solve_equation:
  ∀ x : ℝ, (x + 1) / 3 - 1 = (5 * x - 1) / 6 → x = -1 :=
by
  intro x
  intro h
  sorry

end solve_equation_l215_215720


namespace find_a_l215_215027

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if h : x < 0 then -(Real.log (-x) / Real.log 2) + a else 0

theorem find_a (a : ℝ) :
  (f a (-2) + f a (-4) = 1) → a = 2 :=
by
  sorry

end find_a_l215_215027


namespace find_x_l215_215547

theorem find_x (x : ℚ) : (8 + 10 + 22) / 3 = (15 + x) / 2 → x = 35 / 3 :=
by
  sorry

end find_x_l215_215547


namespace reporters_not_covering_politics_l215_215372

theorem reporters_not_covering_politics (P_X P_Y P_Z intlPol otherPol econOthers : ℝ)
  (h1 : P_X = 0.15) (h2 : P_Y = 0.10) (h3 : P_Z = 0.08)
  (h4 : otherPol = 0.50) (h5 : intlPol = 0.05) (h6 : econOthers = 0.02) :
  (1 - (P_X + P_Y + P_Z + intlPol + otherPol + econOthers)) = 0.10 := by
  sorry

end reporters_not_covering_politics_l215_215372


namespace value_of_A_is_18_l215_215381

theorem value_of_A_is_18
  (A B C D : ℕ)
  (h1 : A ≠ B)
  (h2 : A ≠ C)
  (h3 : A ≠ D)
  (h4 : B ≠ C)
  (h5 : B ≠ D)
  (h6 : C ≠ D)
  (h7 : A * B = 72)
  (h8 : C * D = 72)
  (h9 : A - B = C + D) : A = 18 :=
sorry

end value_of_A_is_18_l215_215381


namespace count_perfect_cubes_l215_215312

theorem count_perfect_cubes (a b : ℕ) (h1 : a = 200) (h2 : b = 1200) :
  ∃ n, n = 5 ∧ ∀ x, (x^3 > a) ∧ (x^3 < b) → (x = 6 ∨ x = 7 ∨ x = 8 ∨ x = 9 ∨ x = 10) := 
sorry

end count_perfect_cubes_l215_215312


namespace compute_x_plus_y_l215_215795

theorem compute_x_plus_y :
    ∃ (x y : ℕ), 4 * y = 7 * 84 ∧ 4 * 63 = 7 * x ∧ x + y = 183 :=
by
  sorry

end compute_x_plus_y_l215_215795


namespace arnold_and_danny_age_l215_215739

theorem arnold_and_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 9) : x = 4 :=
sorry

end arnold_and_danny_age_l215_215739


namespace find_principal_amount_l215_215742

theorem find_principal_amount 
  (P₁ : ℝ) (r₁ t₁ : ℝ) (S₁ : ℝ)
  (P₂ : ℝ) (r₂ t₂ : ℝ) (C₂ : ℝ) :
  S₁ = (P₁ * r₁ * t₁) / 100 →
  C₂ = P₂ * ( (1 + r₂) ^ t₂ - 1) →
  S₁ = C₂ / 2 →
  P₁ = 2800 :=
by
  sorry

end find_principal_amount_l215_215742


namespace calc_30_exp_l215_215991

theorem calc_30_exp :
  30 * 30 ^ 10 = 30 ^ 11 :=
by sorry

end calc_30_exp_l215_215991


namespace smallest_6_digit_div_by_111_l215_215614

theorem smallest_6_digit_div_by_111 : ∃ n : ℕ, 100000 ≤ n ∧ n < 1000000 ∧ n % 111 = 0 ∧ n = 100011 := by
  sorry

end smallest_6_digit_div_by_111_l215_215614


namespace number_of_shirts_is_20_l215_215376

/-- Given the conditions:
1. The total price for some shirts is 360,
2. The total price for 45 sweaters is 900,
3. The average price of a sweater exceeds that of a shirt by 2,
prove that the number of shirts is 20. -/

theorem number_of_shirts_is_20
  (S : ℕ) (P_shirt P_sweater : ℝ)
  (h1 : S * P_shirt = 360)
  (h2 : 45 * P_sweater = 900)
  (h3 : P_sweater = P_shirt + 2) :
  S = 20 :=
by
  sorry

end number_of_shirts_is_20_l215_215376


namespace more_people_joined_l215_215276

def initial_people : Nat := 61
def final_people : Nat := 83

theorem more_people_joined :
  final_people - initial_people = 22 := by
  sorry

end more_people_joined_l215_215276


namespace common_ratio_l215_215561

variable {a : ℕ → ℝ} -- Define a as a sequence of real numbers

-- Define the conditions as hypotheses
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = q * a n

variables (q : ℝ) (h1 : a 2 = 2) (h2 : a 5 = 1 / 4)

-- Define the theorem to prove the common ratio
theorem common_ratio (h_geom : is_geometric_sequence a q) : q = 1 / 2 :=
  sorry

end common_ratio_l215_215561


namespace ratio_of_numbers_l215_215410

theorem ratio_of_numbers (x y : ℕ) (h1 : x + y = 33) (h2 : x = 22) : y / x = 1 / 2 :=
by
  sorry

end ratio_of_numbers_l215_215410


namespace ticket_savings_l215_215459

def single_ticket_cost : ℝ := 1.50
def package_cost : ℝ := 5.75
def num_tickets_needed : ℝ := 40

theorem ticket_savings :
  (num_tickets_needed * single_ticket_cost) - 
  ((num_tickets_needed / 5) * package_cost) = 14.00 :=
by
  sorry

end ticket_savings_l215_215459


namespace profit_percentage_is_50_l215_215434

/--
Assumption:
- Initial machine cost: Rs 10,000
- Repair cost: Rs 5,000
- Transportation charges: Rs 1,000
- Selling price: Rs 24,000

To prove:
- The profit percentage is 50%
-/

def initial_cost : ℕ := 10000
def repair_cost : ℕ := 5000
def transportation_charges : ℕ := 1000
def selling_price : ℕ := 24000
def total_cost : ℕ := initial_cost + repair_cost + transportation_charges
def profit : ℕ := selling_price - total_cost

theorem profit_percentage_is_50 :
  (profit * 100) / total_cost = 50 :=
by
  -- proof goes here
  sorry

end profit_percentage_is_50_l215_215434


namespace total_area_of_L_shaped_figure_l215_215317

-- Define the specific lengths for each segment
def bottom_rect_length : ℕ := 10
def bottom_rect_width : ℕ := 6
def central_rect_length : ℕ := 4
def central_rect_width : ℕ := 4
def top_rect_length : ℕ := 5
def top_rect_width : ℕ := 1

-- Calculate the area of each rectangle
def bottom_rect_area : ℕ := bottom_rect_length * bottom_rect_width
def central_rect_area : ℕ := central_rect_length * central_rect_width
def top_rect_area : ℕ := top_rect_length * top_rect_width

-- Given the length and width of the rectangles, calculate the total area of the L-shaped figure
theorem total_area_of_L_shaped_figure : 
  bottom_rect_area + central_rect_area + top_rect_area = 81 := by
  sorry

end total_area_of_L_shaped_figure_l215_215317


namespace price_of_paint_models_max_boxes_of_paint_A_l215_215384

-- Define the conditions as hypotheses
def price_paint_model_A_B (x y : ℕ) : Prop :=
  x + 2 * y = 56 ∧ 2 * x + y = 64

def total_cost_constraint (x y m : ℕ) : Prop :=
  24 * m + 16 * (200 - m) ≤ 3920

-- Prove the prices of the paint models given the conditions
theorem price_of_paint_models :
  ∃ (x y : ℕ), price_paint_model_A_B x y ∧ x = 24 ∧ y = 16 :=
by
  sorry

-- Prove the maximum number of boxes of paint model A the school can purchase given the total cost constraint
theorem max_boxes_of_paint_A :
  ∃ (m : ℕ), total_cost_constraint 24 16 m ∧ m = 90 :=
by
  sorry

end price_of_paint_models_max_boxes_of_paint_A_l215_215384


namespace smallest_number_div_by_225_with_digits_0_1_l215_215754

theorem smallest_number_div_by_225_with_digits_0_1 :
  ∃ n : ℕ, (∀ d ∈ n.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ n ∧ (∀ m : ℕ, (∀ d ∈ m.digits 10, d = 0 ∨ d = 1) ∧ 225 ∣ m → n ≤ m) ∧ n = 11111111100 :=
sorry

end smallest_number_div_by_225_with_digits_0_1_l215_215754


namespace girls_first_half_l215_215492

theorem girls_first_half (total_students boys_first_half girls_first_half boys_second_half girls_second_half boys_whole_year : ℕ)
  (h1: total_students = 56)
  (h2: boys_first_half = 25)
  (h3: girls_first_half = 15)
  (h4: boys_second_half = 26)
  (h5: girls_second_half = 25)
  (h6: boys_whole_year = 23) : 
  ∃ girls_first_half_only : ℕ, girls_first_half_only = 3 :=
by {
  sorry
}

end girls_first_half_l215_215492


namespace stratified_sampling_third_year_students_l215_215481

theorem stratified_sampling_third_year_students :
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  (third_year_students : ℚ) * sampling_ratio = 20 :=
by 
  let total_students := 900
  let first_year_students := 300
  let second_year_students := 200
  let third_year_students := 400
  let sample_size := 45
  let sampling_ratio := (sample_size : ℚ) / (total_students : ℚ)
  show (third_year_students : ℚ) * sampling_ratio = 20
  sorry

end stratified_sampling_third_year_students_l215_215481


namespace integer_solutions_set_l215_215262

theorem integer_solutions_set :
  {x : ℤ | 2 * x + 4 > 0 ∧ 1 + x ≥ 2 * x - 1} = {-1, 0, 1, 2} :=
by {
  sorry
}

end integer_solutions_set_l215_215262


namespace exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l215_215286

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := x^2

-- N-gon properties
def is_convex_ngon (N : ℕ) (vertices : List (ℝ × ℝ)) : Prop :=
  -- Placeholder for checking properties; actual implementation would validate convexity and equilateral nature.
  sorry 

-- Statement for 2011-gon
theorem exists_convex_2011_gon_on_parabola :
  ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2011 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

-- Statement for 2012-gon
theorem not_exists_convex_2012_gon_on_parabola :
  ¬ ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2012 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

end exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l215_215286


namespace parabola_intersections_l215_215183

-- Define the first parabola
def parabola1 (x : ℝ) : ℝ :=
  2 * x^2 - 10 * x - 10

-- Define the second parabola
def parabola2 (x : ℝ) : ℝ :=
  x^2 - 4 * x + 6

-- Define the theorem stating the points of intersection
theorem parabola_intersections :
  ∀ (p : ℝ × ℝ), (parabola1 p.1 = p.2) ∧ (parabola2 p.1 = p.2) ↔ (p = (-2, 18) ∨ p = (8, 38)) :=
by
  sorry

end parabola_intersections_l215_215183


namespace number_of_dress_designs_l215_215080

theorem number_of_dress_designs :
  let colors := 5
  let patterns := 4
  let sleeve_designs := 3
  colors * patterns * sleeve_designs = 60 := by
  sorry

end number_of_dress_designs_l215_215080


namespace simplify_fraction_l215_215732

theorem simplify_fraction (m : ℝ) (hm : m ≠ 0) : (3 * m^3) / (6 * m^2) = m / 2 :=
by
  sorry

end simplify_fraction_l215_215732


namespace parking_lot_wheels_l215_215900

-- definitions for the conditions
def num_cars : ℕ := 10
def num_bikes : ℕ := 2
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- statement of the theorem
theorem parking_lot_wheels : (num_cars * wheels_per_car) + (num_bikes * wheels_per_bike) = 44 := by
  sorry

end parking_lot_wheels_l215_215900


namespace find_k_l215_215946

theorem find_k (k : ℤ) (x : ℚ) (h1 : 5 * x + 3 * k = 24) (h2 : 5 * x + 3 = 0) : k = 9 := 
by
  sorry

end find_k_l215_215946


namespace common_divisors_count_l215_215622

-- Given conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8000

-- Prime factorizations from conditions
def fact_num1 : List ℕ := [2^3, 3^1, 5^1, 7^2]
def fact_num2 : List ℕ := [2^6, 5^3]

-- Computing gcd based on factorizations
def gcd : ℕ := 2^3 * 5^1

-- The goal is to prove the number of divisors of gcd is 8
theorem common_divisors_count : 
  ∃ d, d = (3+1)*(1+1) ∧ d = 8 := 
by
  sorry

end common_divisors_count_l215_215622


namespace tyler_remaining_money_l215_215186

def initial_amount : ℕ := 100
def scissors_qty : ℕ := 8
def scissor_cost : ℕ := 5
def erasers_qty : ℕ := 10
def eraser_cost : ℕ := 4

def remaining_amount (initial_amount scissors_qty scissor_cost erasers_qty eraser_cost : ℕ) : ℕ :=
  initial_amount - (scissors_qty * scissor_cost + erasers_qty * eraser_cost)

theorem tyler_remaining_money :
  remaining_amount initial_amount scissors_qty scissor_cost erasers_qty eraser_cost = 20 := by
  sorry

end tyler_remaining_money_l215_215186


namespace initial_mean_corrected_l215_215775

theorem initial_mean_corrected (M : ℝ) (H : 30 * M + 30 = 30 * 151) : M = 150 :=
sorry

end initial_mean_corrected_l215_215775


namespace total_birds_in_marsh_l215_215976

-- Given conditions
def initial_geese := 58
def doubled_geese := initial_geese * 2
def ducks := 37
def swans := 15
def herons := 22

-- Prove that the total number of birds is 190
theorem total_birds_in_marsh : 
  doubled_geese + ducks + swans + herons = 190 := 
by
  sorry

end total_birds_in_marsh_l215_215976


namespace subtraction_and_multiplication_problem_l215_215132

theorem subtraction_and_multiplication_problem :
  (5 / 6 - 1 / 3) * 3 / 4 = 3 / 8 :=
by sorry

end subtraction_and_multiplication_problem_l215_215132


namespace find_principal_l215_215126

def r : ℝ := 0.03
def t : ℝ := 3
def I (P : ℝ) : ℝ := P - 1820
def simple_interest (P : ℝ) : ℝ := P * r * t

theorem find_principal (P : ℝ) : simple_interest P = I P -> P = 2000 :=
by
  sorry

end find_principal_l215_215126


namespace oscar_leap_difference_in_feet_l215_215955

theorem oscar_leap_difference_in_feet 
  (strides_per_gap : ℕ) 
  (leaps_per_gap : ℕ) 
  (total_distance : ℕ) 
  (num_poles : ℕ)
  (h1 : strides_per_gap = 54) 
  (h2 : leaps_per_gap = 15) 
  (h3 : total_distance = 5280) 
  (h4 : num_poles = 51) 
  : (total_distance / (strides_per_gap * (num_poles - 1)) -
       total_distance / (leaps_per_gap * (num_poles - 1)) = 5) :=
by
  sorry

end oscar_leap_difference_in_feet_l215_215955


namespace product_of_first_three_terms_is_960_l215_215392

-- Definitions from the conditions
def a₁ : ℤ := 20 - 6 * 2
def a₂ : ℤ := a₁ + 2
def a₃ : ℤ := a₂ + 2

-- Problem statement
theorem product_of_first_three_terms_is_960 : 
  a₁ * a₂ * a₃ = 960 :=
by
  sorry

end product_of_first_three_terms_is_960_l215_215392


namespace trevor_eggs_left_l215_215100

def gertrude_eggs : Nat := 4
def blanche_eggs : Nat := 3
def nancy_eggs : Nat := 2
def martha_eggs : Nat := 2
def dropped_eggs : Nat := 2

theorem trevor_eggs_left : 
  (gertrude_eggs + blanche_eggs + nancy_eggs + martha_eggs - dropped_eggs) = 9 := 
  by sorry

end trevor_eggs_left_l215_215100


namespace profit_at_15_percent_off_l215_215849

theorem profit_at_15_percent_off 
    (cost_price marked_price : ℝ) 
    (cost_price_eq : cost_price = 2000)
    (marked_price_eq : marked_price = (200 + cost_price) / 0.8) :
    (0.85 * marked_price - cost_price) = 337.5 := by
  sorry

end profit_at_15_percent_off_l215_215849


namespace intersection_complement_l215_215831

def A := {x : ℝ | -1 < x ∧ x < 6}
def B := {x : ℝ | x^2 < 4}
def complement_R (S : Set ℝ) := {x : ℝ | x ∉ S}

theorem intersection_complement :
  A ∩ (complement_R B) = {x : ℝ | 2 ≤ x ∧ x < 6} := by
sorry

end intersection_complement_l215_215831


namespace gcd_180_450_l215_215889

theorem gcd_180_450 : gcd 180 450 = 90 :=
by sorry

end gcd_180_450_l215_215889


namespace distance_to_SFL_l215_215812

def distance_per_hour : ℕ := 27
def hours_travelled : ℕ := 3

theorem distance_to_SFL :
  (distance_per_hour * hours_travelled) = 81 := 
by
  sorry

end distance_to_SFL_l215_215812


namespace second_train_length_l215_215151

theorem second_train_length
  (L1 : ℝ) (V1 : ℝ) (V2 : ℝ) (T : ℝ)
  (h1 : L1 = 300)
  (h2 : V1 = 72 * 1000 / 3600)
  (h3 : V2 = 36 * 1000 / 3600)
  (h4 : T = 79.99360051195904) :
  L1 + (V1 - V2) * T = 799.9360051195904 :=
by
  sorry

end second_train_length_l215_215151


namespace no_base_b_square_of_integer_l215_215290

theorem no_base_b_square_of_integer (b : ℕ) : ¬(∃ n : ℕ, n^2 = b^2 + 3 * b + 1) → b < 4 ∨ b > 8 := by
  sorry

end no_base_b_square_of_integer_l215_215290


namespace rectangle_side_lengths_l215_215166

theorem rectangle_side_lengths:
  ∃ x : ℝ, ∃ y : ℝ, (2 * (x + y) * 2 = x * y) ∧ (y = x + 3) ∧ (x > 0) ∧ (y > 0) ∧ x = 8 ∧ y = 11 :=
by
  sorry

end rectangle_side_lengths_l215_215166


namespace binary_sum_in_base_10_l215_215032

theorem binary_sum_in_base_10 :
  (255 : ℕ) + (63 : ℕ) = 318 :=
sorry

end binary_sum_in_base_10_l215_215032


namespace find_s_squared_l215_215941

-- Define the conditions and entities in Lean
variable (s : ℝ)
def passesThrough (x y : ℝ) (a b : ℝ) : Prop :=
  (y^2 / 9) - (x^2 / a^2) = 1

-- State the given conditions as hypotheses
axiom h₀ : passesThrough 0 3 3 1
axiom h₁ : passesThrough 5 (-3) 25 1
axiom h₂ : passesThrough s (-4) 25 1

-- State the theorem we want to prove
theorem find_s_squared : s^2 = 175 / 9 := by
  sorry

end find_s_squared_l215_215941


namespace odd_number_expression_l215_215698

theorem odd_number_expression (o n : ℤ) (ho : o % 2 = 1) : (o^2 + n * o + 1) % 2 = 1 ↔ n % 2 = 1 := by
  sorry

end odd_number_expression_l215_215698


namespace find_A_for_diamond_l215_215939

def diamond (A B : ℕ) : ℕ := 4 * A + 3 * B + 7

theorem find_A_for_diamond (A : ℕ) (h : diamond A 7 = 76) : A = 12 :=
by
  sorry

end find_A_for_diamond_l215_215939


namespace martha_black_butterflies_l215_215844

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l215_215844


namespace factor_expression_l215_215331

theorem factor_expression (b : ℝ) : 275 * b^2 + 55 * b = 55 * b * (5 * b + 1) := by
  sorry

end factor_expression_l215_215331


namespace total_skips_correct_l215_215517

def S (n : ℕ) : ℕ := n^2 + n

def TotalSkips5 : ℕ :=
  S 1 + S 2 + S 3 + S 4 + S 5

def Skips6 : ℕ :=
  2 * S 6

theorem total_skips_correct : TotalSkips5 + Skips6 = 154 :=
by
  -- proof goes here
  sorry

end total_skips_correct_l215_215517


namespace sum_of_squares_2222_l215_215716

theorem sum_of_squares_2222 :
  ∀ (N : ℕ), (∃ (k : ℕ), N = 2 * 10^k - 1) → (∀ (a b : ℤ), N = a^2 + b^2 ↔ N = 2) :=
by sorry

end sum_of_squares_2222_l215_215716


namespace greatest_x_for_A_is_perfect_square_l215_215931

theorem greatest_x_for_A_is_perfect_square :
  ∃ x : ℕ, x = 2008 ∧ ∀ y : ℕ, (∃ k : ℕ, 2^182 + 4^y + 8^700 = k^2) → y ≤ 2008 :=
by 
  sorry

end greatest_x_for_A_is_perfect_square_l215_215931


namespace solution_set_to_coeff_properties_l215_215261

theorem solution_set_to_coeff_properties 
  (a b c : ℝ) 
  (h : ∀ x, (2 < x ∧ x < 3) → ax^2 + bx + c > 0) 
  : 
  (a < 0) 
  ∧ (b * c < 0) 
  ∧ (b + c = a) :=
sorry

end solution_set_to_coeff_properties_l215_215261


namespace max_min_distance_inequality_l215_215801

theorem max_min_distance_inequality (n : ℕ) (D d : ℝ) (h1 : d > 0) 
    (exists_points : ∃ (points : Fin n → ℝ × ℝ), 
      (∀ i j : Fin n, i ≠ j → dist (points i) (points j) ≥ d) 
      ∧ (∀ i j : Fin n, dist (points i) (points j) ≤ D)) : 
    D / d > (Real.sqrt (n * Real.pi)) / 2 - 1 := 
  sorry

end max_min_distance_inequality_l215_215801


namespace door_height_is_eight_l215_215371

/-- Statement of the problem: given a door with specified dimensions as conditions,
prove that the height of the door is 8 feet. -/
theorem door_height_is_eight (x : ℝ) (h₁ : x^2 = (x - 4)^2 + (x - 2)^2) : (x - 2) = 8 :=
by
  sorry

end door_height_is_eight_l215_215371


namespace jacket_price_equation_l215_215807

theorem jacket_price_equation (x : ℝ) (h : 0.8 * (1 + 0.5) * x - x = 28) : 0.8 * (1 + 0.5) * x = x + 28 :=
by sorry

end jacket_price_equation_l215_215807


namespace problem_part1_problem_part2_l215_215199

-- Definitions based on conditions
def f (x a : ℝ) : ℝ := x^2 + abs (2*x - a)

-- Proof statements
theorem problem_part1 (a : ℝ) (h_even : ∀ x : ℝ, f x a = f (-x) a) : a = 0 := sorry

theorem problem_part2 (a : ℝ) (h_a_gt_two : a > 2) : 
  ∃ x : ℝ, ∀ y : ℝ, f x a ≤ f y a ∧ f x a = a - 1 := sorry

end problem_part1_problem_part2_l215_215199


namespace cube_surface_area_increase_l215_215995

theorem cube_surface_area_increase (s : ℝ) : 
  let original_surface_area := 6 * s^2
  let new_edge := 1.3 * s
  let new_surface_area := 6 * (new_edge)^2
  let percentage_increase := ((new_surface_area - original_surface_area) / original_surface_area) * 100
  percentage_increase = 69 := 
by
  sorry

end cube_surface_area_increase_l215_215995


namespace ice_cream_flavors_l215_215915

theorem ice_cream_flavors : (Nat.choose (4 + 4 - 1) (4 - 1) = 35) :=
by
  sorry

end ice_cream_flavors_l215_215915


namespace range_of_a_l215_215020

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x - 4 < 0) ↔ (-2 < a ∧ a ≤ 2) :=
sorry

end range_of_a_l215_215020


namespace minimize_sum_m_n_l215_215642

-- Definitions of the given conditions
def last_three_digits_equal (a b : ℕ) : Prop :=
  (a % 1000) = (b % 1000)

-- The main statement to prove
theorem minimize_sum_m_n (m n : ℕ) (h1 : n > m) (h2 : 1 ≤ m) 
  (h3 : last_three_digits_equal (1978^n) (1978^m)) : m + n = 106 :=
sorry

end minimize_sum_m_n_l215_215642


namespace fifth_term_binomial_expansion_l215_215049

noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem fifth_term_binomial_expansion (b x : ℝ) :
  let term := (binomial 7 4) * ((b / x)^(7 - 4)) * ((-x^2 * b)^4)
  term = -35 * b^7 * x^5 := 
by
  sorry

end fifth_term_binomial_expansion_l215_215049


namespace xiao_li_first_three_l215_215856

def q1_proba_correct (p1 p2 p3 : ℚ) : ℚ :=
  p1 * p2 * p3 + 
  (1 - p1) * p2 * p3 + 
  p1 * (1 - p2) * p3 + 
  p1 * p2 * (1 - p3)

theorem xiao_li_first_three (p1 p2 p3 : ℚ) (h1 : p1 = 3/4) (h2 : p2 = 1/2) (h3 : p3 = 5/6) :
  q1_proba_correct p1 p2 p3 = 11 / 24 := by
  rw [h1, h2, h3]
  sorry

end xiao_li_first_three_l215_215856


namespace cole_drive_time_l215_215935

theorem cole_drive_time (D T1 T2 : ℝ) (h1 : T1 = D / 75) 
  (h2 : T2 = D / 105) (h3 : T1 + T2 = 6) : 
  (T1 * 60 = 210) :=
by sorry

end cole_drive_time_l215_215935


namespace andrew_age_l215_215056

variables (a g : ℕ)

theorem andrew_age : 
  (g = 16 * a) ∧ (g - a = 60) → a = 4 := by
  sorry

end andrew_age_l215_215056


namespace equation_solution_system_of_inequalities_solution_l215_215230

theorem equation_solution (x : ℝ) : (3 / (x - 1) = 1 / (2 * x + 3)) ↔ (x = -2) :=
by
  sorry

theorem system_of_inequalities_solution (x : ℝ) : ((3 * x - 1 ≥ x + 1) ∧ (x + 3 > 4 * x - 2)) ↔ (1 ≤ x ∧ x < 5 / 3) :=
by
  sorry

end equation_solution_system_of_inequalities_solution_l215_215230


namespace find_x_l215_215854

theorem find_x (x y : ℝ) (h1 : y = 1) (h2 : 4 * x - 2 * y + 3 = 3 * x + 3 * y) : x = 2 :=
by
  sorry

end find_x_l215_215854


namespace soda_cost_per_ounce_l215_215446

/-- 
  Peter brought $2 with him, left with $0.50, and bought 6 ounces of soda.
  Prove that the cost per ounce of soda is $0.25.
-/
theorem soda_cost_per_ounce (initial_money final_money : ℝ) (amount_spent ounces_soda cost_per_ounce : ℝ)
  (h1 : initial_money = 2)
  (h2 : final_money = 0.5)
  (h3 : amount_spent = initial_money - final_money)
  (h4 : amount_spent = 1.5)
  (h5 : ounces_soda = 6)
  (h6 : cost_per_ounce = amount_spent / ounces_soda) :
  cost_per_ounce = 0.25 :=
by sorry

end soda_cost_per_ounce_l215_215446


namespace imo_2007_p6_l215_215495

theorem imo_2007_p6 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  ∃ k : ℕ, (x = 11 * k^2) ∧ (y = 11 * k) ↔
  ∃ k : ℕ, (∃ k₁ : ℤ, k₁ = (x^2 * y + x + y) / (x * y^2 + y + 11)) :=
sorry

end imo_2007_p6_l215_215495


namespace scientists_arrival_probability_l215_215909

open Real

theorem scientists_arrival_probability (x y z : ℕ) (n : ℝ) (h : z ≠ 0)
  (hz : ¬ ∃ p : ℕ, Nat.Prime p ∧ p ^ 2 ∣ z)
  (h1 : n = x - y * sqrt z)
  (h2 : ∃ (a b : ℝ), 0 ≤ a ∧ a ≤ 120 ∧ 0 ≤ b ∧ b ≤ 120 ∧
    |a - b| ≤ n)
  (h3 : (120 - n)^2 / (120 ^ 2) = 0.7) :
  x + y + z = 202 := sorry

end scientists_arrival_probability_l215_215909


namespace brad_red_balloons_l215_215303

theorem brad_red_balloons (total balloons green : ℕ) (h1 : total = 17) (h2 : green = 9) : total - green = 8 := 
by {
  sorry
}

end brad_red_balloons_l215_215303


namespace sarahs_team_mean_score_l215_215014

def mean_score_of_games (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

theorem sarahs_team_mean_score :
  mean_score_of_games [69, 68, 70, 61, 74, 62, 65, 74] = 67.875 :=
by
  sorry

end sarahs_team_mean_score_l215_215014


namespace simplify_division_l215_215922

theorem simplify_division :
  (27 * 10^9) / (9 * 10^5) = 30000 :=
  sorry

end simplify_division_l215_215922


namespace second_number_more_than_first_l215_215830

-- Definitions of A and B based on the given ratio
def A : ℚ := 7 / 56
def B : ℚ := 8 / 56

-- Proof statement
theorem second_number_more_than_first : ((B - A) / A) * 100 = 100 / 7 :=
by
  -- skipped the proof
  sorry

end second_number_more_than_first_l215_215830


namespace minimum_value_of_x_squared_l215_215820

theorem minimum_value_of_x_squared : ∃ x : ℝ, x = 0 ∧ ∀ y : ℝ, y = x^2 → y ≥ 0 :=
by
  sorry

end minimum_value_of_x_squared_l215_215820


namespace inequality_range_a_l215_215821

open Real

theorem inequality_range_a (x : ℝ) (h : x > 1) : x + 1 / (x - 1) ≥ 3 :=
sorry

end inequality_range_a_l215_215821


namespace ratio_of_juniors_to_freshmen_l215_215625

variables (f j : ℕ) 

theorem ratio_of_juniors_to_freshmen (h1 : (1/4 : ℚ) * f = (1/2 : ℚ) * j) :
  j = f / 2 :=
by
  sorry

end ratio_of_juniors_to_freshmen_l215_215625


namespace perpendicular_condition_l215_215721

def line := Type
def plane := Type

variables {α : plane} {a b : line}

-- Conditions: define parallelism and perpendicularity
def parallel (a : line) (α : plane) : Prop := sorry
def perpendicular (a : line) (α : plane) : Prop := sorry
def perpendicular_lines (a b : line) : Prop := sorry

-- Given Hypotheses
variable (h1 : parallel a α)
variable (h2 : perpendicular b α)

-- Statement to prove
theorem perpendicular_condition (h1 : parallel a α) (h2 : perpendicular b α) :
  (perpendicular_lines b a) ∧ (¬ (perpendicular_lines b a → perpendicular b α)) := 
sorry

end perpendicular_condition_l215_215721


namespace perpendicular_vectors_l215_215682

def vector (α : Type) := (α × α)
def dot_product {α : Type} [Add α] [Mul α] (a b : vector α) : α :=
  a.1 * b.1 + a.2 * b.2

theorem perpendicular_vectors
    (a : vector ℝ) (b : vector ℝ)
    (h : dot_product a b = 0)
    (ha : a = (2, 4))
    (hb : b = (-1, n)) : 
    n = 1 / 2 := 
  sorry

end perpendicular_vectors_l215_215682


namespace triangle_ineq_l215_215902

theorem triangle_ineq (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 2 * (a^2 + b^2) > c^2 := 
by 
  sorry

end triangle_ineq_l215_215902


namespace sequence_not_divisible_by_7_l215_215764

theorem sequence_not_divisible_by_7 (n : ℕ) (hn : 1 ≤ n ∧ n ≤ 1200) : ¬ (7 ∣ (9^n + 1)) :=
by
  sorry

end sequence_not_divisible_by_7_l215_215764


namespace range_distance_PQ_l215_215455

noncomputable def point_P (α : ℝ) : ℝ × ℝ × ℝ := (3 * Real.cos α, 3 * Real.sin α, 1)
noncomputable def point_Q (β : ℝ) : ℝ × ℝ × ℝ := (2 * Real.cos β, 2 * Real.sin β, 1)

noncomputable def distance_PQ (α β : ℝ) : ℝ :=
  Real.sqrt ((3 * Real.cos α - 2 * Real.cos β)^2 +
             (3 * Real.sin α - 2 * Real.sin β)^2 +
             (1 - 1)^2)

theorem range_distance_PQ : 
  ∀ α β : ℝ, 1 ≤ distance_PQ α β ∧ distance_PQ α β ≤ 5 := 
by
  intros
  sorry

end range_distance_PQ_l215_215455


namespace instantaneous_velocity_at_3_l215_215451

-- Define the displacement function s(t)
def displacement (t : ℝ) : ℝ := 2 * t^3

-- Define the time at which we want to calculate the instantaneous velocity
def time : ℝ := 3

-- Define the expected instantaneous velocity at t=3
def expected_velocity : ℝ := 54

-- Define the derivative of the displacement function as the velocity function
noncomputable def velocity (t : ℝ) : ℝ := deriv displacement t

-- Theorem: Prove that the instantaneous velocity at t=3 is 54
theorem instantaneous_velocity_at_3 : velocity time = expected_velocity := 
by {
  -- Here the detailed proof should go, but we skip it with sorry
  sorry
}

end instantaneous_velocity_at_3_l215_215451


namespace students_neither_correct_l215_215524

-- Define the total number of students and the numbers for chemistry, biology, and both
def total_students := 75
def chemistry_students := 42
def biology_students := 33
def both_subject_students := 18

-- Define a function to calculate the number of students taking neither chemistry nor biology
def students_neither : ℕ :=
  total_students - ((chemistry_students - both_subject_students) 
                    + (biology_students - both_subject_students) 
                    + both_subject_students)

-- Theorem stating that the number of students taking neither chemistry nor biology is as expected
theorem students_neither_correct : students_neither = 18 :=
  sorry

end students_neither_correct_l215_215524


namespace find_y_l215_215363

-- Declare the variables and conditions
variable (x y : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 1.5 * x = 0.3 * y
def condition2 : Prop := x = 20

-- State the theorem that given these conditions, y must be 100
theorem find_y (h1 : condition1 x y) (h2 : condition2 x) : y = 100 :=
by sorry

end find_y_l215_215363


namespace sum_of_first_2m_terms_l215_215024

variable (m : ℕ)
variable (S : ℕ → ℤ)

-- Conditions
axiom Sm : S m = 100
axiom S3m : S (3 * m) = -150

-- Theorem statement
theorem sum_of_first_2m_terms : S (2 * m) = 50 :=
by
  sorry

end sum_of_first_2m_terms_l215_215024


namespace slope_of_line_l215_215287

theorem slope_of_line
  (k : ℝ) 
  (hk : 0 < k) 
  (h1 : ¬ (2 / Real.sqrt (k^2 + 1) = 3 * 2 * Real.sqrt (1 - 8 * k^2) / Real.sqrt (k^2 + 1))) 
  : k = 1 / 3 :=
sorry

end slope_of_line_l215_215287


namespace real_numbers_inequality_l215_215057

theorem real_numbers_inequality (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c)^2 :=
by
  sorry

end real_numbers_inequality_l215_215057


namespace roots_polynomial_equation_l215_215982

noncomputable def rootsEquation (x y : ℝ) := x + y = 10 ∧ |x - y| = 12

theorem roots_polynomial_equation : ∃ (x y : ℝ), rootsEquation x y ∧ (x^2 - 10 * x - 11 = 0) := sorry

end roots_polynomial_equation_l215_215982


namespace proof_statement_l215_215501

-- Define the initial dimensions and areas
def initial_length : ℕ := 7
def initial_width : ℕ := 5

-- Shortened dimensions by one side and the corresponding area condition
def shortened_new_width : ℕ := 3
def shortened_area : ℕ := 21

-- Define the task
def task_statement : Prop :=
  (initial_length - 2) * initial_width = shortened_area ∧
  (initial_width - 2) * initial_length = shortened_area →
  (initial_length - 2) * (initial_width - 2) = 25

theorem proof_statement : task_statement :=
by {
  sorry -- Proof goes here
}

end proof_statement_l215_215501


namespace fourth_person_height_l215_215840

noncomputable def height_of_fourth_person (H : ℕ) : ℕ := 
  let second_person := H + 2
  let third_person := H + 4
  let fourth_person := H + 10
  fourth_person

theorem fourth_person_height {H : ℕ} 
  (cond1 : 2 = 2)
  (cond2 : 6 = 6)
  (average_height : 76 = 76) 
  (height_sum : H + (H + 2) + (H + 4) + (H + 10) = 304) : 
  height_of_fourth_person H = 82 := sorry

end fourth_person_height_l215_215840


namespace solution_pairs_l215_215663

theorem solution_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x ^ 2 + y ^ 2 - 5 * x * y + 5 = 0 ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2) := by
  sorry

end solution_pairs_l215_215663


namespace min_quotient_l215_215971

def digits_distinct (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def quotient (a b c : ℕ) : ℚ := 
  (100 * a + 10 * b + c : ℚ) / (a + b + c : ℚ)

theorem min_quotient (a b c : ℕ) (h1 : b > 3) (h2 : c ≠ b) (h3: digits_distinct a b c) : 
  quotient a b c ≥ 19.62 :=
sorry

end min_quotient_l215_215971


namespace quadratic_root_range_l215_215393

theorem quadratic_root_range (k : ℝ) (hk : k ≠ 0) (h : (4 + 4 * k) > 0) : k > -1 :=
by sorry

end quadratic_root_range_l215_215393


namespace line_through_A_and_B_l215_215668

variables (x y x₁ y₁ x₂ y₂ : ℝ)

-- Conditions
def condition1 : Prop := 3 * x₁ - 4 * y₁ - 2 = 0
def condition2 : Prop := 3 * x₂ - 4 * y₂ - 2 = 0

-- Proof that the line passing through A(x₁, y₁) and B(x₂, y₂) is 3x - 4y - 2 = 0
theorem line_through_A_and_B (h1 : condition1 x₁ y₁) (h2 : condition2 x₂ y₂) :
    ∀ (x y : ℝ), (∃ k : ℝ, x = x₁ + k * (x₂ - x₁) ∧ y = y₁ + k * (y₂ - y₁)) → 3 * x - 4 * y - 2 = 0 :=
sorry

end line_through_A_and_B_l215_215668


namespace value_of_Y_is_669_l215_215074

theorem value_of_Y_is_669 :
  let A := 3009 / 3
  let B := A / 3
  let Y := A - B
  Y = 669 :=
by
  sorry

end value_of_Y_is_669_l215_215074


namespace fermat_little_theorem_l215_215699

theorem fermat_little_theorem (N p : ℕ) (hp : Nat.Prime p) (hNp : ¬ p ∣ N) : p ∣ (N ^ (p - 1) - 1) := 
sorry

end fermat_little_theorem_l215_215699


namespace decreased_and_divided_l215_215030

theorem decreased_and_divided (x : ℝ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 := by
  sorry

end decreased_and_divided_l215_215030


namespace no_consecutive_squares_l215_215064

open Nat

-- Define a function to get the n-th prime number
def prime (n : ℕ) : ℕ := sorry -- Use an actual function or sequence that generates prime numbers, this is a placeholder.

-- Define the sequence S_n, the sum of the first n prime numbers
def S : ℕ → ℕ
| 0       => 0
| (n + 1) => S n + prime (n + 1)

-- Define a predicate to check if a number is a perfect square
def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

-- The theorem that no two consecutive terms S_{n-1} and S_n can both be perfect squares
theorem no_consecutive_squares (n : ℕ) : ¬ (is_square (S n) ∧ is_square (S (n + 1))) :=
by
  sorry

end no_consecutive_squares_l215_215064


namespace measure_angle_4_l215_215348

theorem measure_angle_4 (m1 m2 m3 m5 m6 m4 : ℝ) 
  (h1 : m1 = 82) 
  (h2 : m2 = 34) 
  (h3 : m3 = 19) 
  (h4 : m5 = m6 + 10) 
  (h5 : m1 + m2 + m3 + m5 + m6 = 180)
  (h6 : m4 + m5 + m6 = 180) : 
  m4 = 135 :=
by
  -- Placeholder for the full proof, omitted due to instructions
  sorry

end measure_angle_4_l215_215348


namespace range_of_k_l215_215510

theorem range_of_k
  (x y k : ℝ)
  (h1 : 3 * x + y = k + 1)
  (h2 : x + 3 * y = 3)
  (h3 : 0 < x + y)
  (h4 : x + y < 1) :
  -4 < k ∧ k < 0 :=
sorry

end range_of_k_l215_215510


namespace price_difference_l215_215391

theorem price_difference (P F : ℝ) (h1 : 0.85 * P = 78.2) (h2 : F = 78.2 * 1.25) : F - P = 5.75 :=
by
  sorry

end price_difference_l215_215391


namespace solution_exists_l215_215875

def age_problem (S F Y : ℕ) : Prop :=
  S = 12 ∧ S = F / 3 ∧ S - Y = (F - Y) / 5 ∧ Y = 6

theorem solution_exists : ∃ (Y : ℕ), ∃ (S F : ℕ), age_problem S F Y :=
by sorry

end solution_exists_l215_215875


namespace total_crayons_l215_215416

theorem total_crayons (crayons_per_child : ℕ) (num_children : ℕ) 
      (h1 : crayons_per_child = 18) (h2 : num_children = 36) : 
        crayons_per_child * num_children = 648 := by
  sorry

end total_crayons_l215_215416


namespace time_per_flash_l215_215240

def minutes_per_hour : ℕ := 60
def seconds_per_minute : ℕ := 60
def light_flashes_in_three_fourths_hour : ℕ := 180

-- Converting ¾ of an hour to minutes and then to seconds
def seconds_in_three_fourths_hour : ℕ := (3 * minutes_per_hour / 4) * seconds_per_minute

-- Proving that the time taken for one flash is 15 seconds
theorem time_per_flash : (seconds_in_three_fourths_hour / light_flashes_in_three_fourths_hour) = 15 :=
by
  sorry

end time_per_flash_l215_215240


namespace opposite_of_2023_l215_215332

theorem opposite_of_2023 :
  -2023 = -2023 :=
by
  sorry

end opposite_of_2023_l215_215332


namespace age_of_25th_student_l215_215197

theorem age_of_25th_student 
(A : ℤ) (B : ℤ) (C : ℤ) (D : ℤ)
(total_students : ℤ)
(total_age : ℤ)
(age_all_students : ℤ)
(avg_age_all_students : ℤ)
(avg_age_7_students : ℤ)
(avg_age_12_students : ℤ)
(avg_age_5_students : ℤ)
:
total_students = 25 →
avg_age_all_students = 18 →
avg_age_7_students = 20 →
avg_age_12_students = 16 →
avg_age_5_students = 19 →
total_age = total_students * avg_age_all_students →
age_all_students = total_age - (7 * avg_age_7_students + 12 * avg_age_12_students + 5 * avg_age_5_students) →
A = 7 * avg_age_7_students →
B = 12 * avg_age_12_students →
C = 5 * avg_age_5_students →
D = total_age - (A + B + C) →
D = 23 :=
by {
  sorry
}

end age_of_25th_student_l215_215197


namespace area_under_pressure_l215_215239

theorem area_under_pressure (F : ℝ) (S : ℝ) (p : ℝ) (hF : F = 100) (hp : p > 1000) (hpressure : p = F / S) :
  S < 0.1 :=
by
  sorry

end area_under_pressure_l215_215239


namespace find_c_l215_215657

theorem find_c (c : ℝ) (h1 : ∃ x : ℝ, (⌊c⌋ : ℝ) = x ∧ 3 * x^2 + 12 * x - 27 = 0)
                      (h2 : ∃ x : ℝ, (c - ⌊c⌋) = x ∧ 4 * x^2 - 12 * x + 5 = 0) :
                      c = -8.5 :=
by
  sorry

end find_c_l215_215657


namespace parabola_tangent_sum_l215_215632

theorem parabola_tangent_sum (m n : ℕ) (hmn_coprime : Nat.gcd m n = 1)
    (h_tangent : ∃ (k : ℝ), ∀ (x y : ℝ), y = 4 * x^2 ↔ x = y^2 + (m / n)) :
    m + n = 19 :=
by
  sorry

end parabola_tangent_sum_l215_215632


namespace find_number_l215_215023

theorem find_number (x n : ℤ) 
  (h1 : 0 < x) (h2 : x < 7) 
  (h3 : x < 15) 
  (h4 : -1 < x) (h5 : x < 5) 
  (h6 : x < 3) (h7 : 0 < x) 
  (h8 : x + n < 4) 
  (hx : x = 1): 
  n < 3 := 
sorry

end find_number_l215_215023


namespace first_player_wins_l215_215778

-- Define the polynomial with placeholders
def P (X : ℤ) (a3 a2 a1 a0 : ℤ) : ℤ :=
  X^4 + a3 * X^3 + a2 * X^2 + a1 * X + a0

-- The statement that the first player can always win
theorem first_player_wins :
  ∀ (a3 a2 a1 a0 : ℤ),
    (a0 ≠ 0) → (a1 ≠ 0) → (a2 ≠ 0) → (a3 ≠ 0) →
    ∃ (strategy : ℕ → ℤ),
      (∀ n, strategy n ≠ 0) ∧
      ¬ ∃ (x y : ℤ), x ≠ y ∧ P x (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 ∧ P y (strategy 0) (strategy 1) (strategy 2) (strategy 3) = 0 :=
by
  sorry

end first_player_wins_l215_215778


namespace items_from_B_l215_215466

noncomputable def totalItems : ℕ := 1200
noncomputable def ratioA : ℕ := 3
noncomputable def ratioB : ℕ := 4
noncomputable def ratioC : ℕ := 5
noncomputable def totalRatio : ℕ := ratioA + ratioB + ratioC
noncomputable def sampledItems : ℕ := 60
noncomputable def numberB := sampledItems * ratioB / totalRatio

theorem items_from_B :
  numberB = 20 :=
by
  sorry

end items_from_B_l215_215466


namespace derivative_at_one_third_l215_215449

noncomputable def f (x : ℝ) : ℝ := Real.log (2 - 3 * x)

theorem derivative_at_one_third : (deriv f (1 / 3) = -3) := by
  sorry

end derivative_at_one_third_l215_215449


namespace kylie_earrings_l215_215118

def number_of_necklaces_monday := 10
def number_of_necklaces_tuesday := 2
def number_of_bracelets_wednesday := 5
def beads_per_necklace := 20
def beads_per_bracelet := 10
def beads_per_earring := 5
def total_beads := 325

theorem kylie_earrings : 
    (total_beads - ((number_of_necklaces_monday + number_of_necklaces_tuesday) * beads_per_necklace + number_of_bracelets_wednesday * beads_per_bracelet)) / beads_per_earring = 7 :=
by
    sorry

end kylie_earrings_l215_215118


namespace expansion_sum_l215_215848

theorem expansion_sum (A B C : ℤ) (h1 : A = (2 - 1)^10) (h2 : B = (2 + 0)^10) (h3 : C = -5120) : 
A + B + C = -4095 :=
by 
  sorry

end expansion_sum_l215_215848


namespace sufficient_but_not_necessary_condition_l215_215741

theorem sufficient_but_not_necessary_condition (x : ℝ) (hx : x > 1) : x > 1 → x^2 > 1 ∧ ∀ y, (y^2 > 1 → ¬(y ≥ 1) → y < -1) := 
by
  sorry

end sufficient_but_not_necessary_condition_l215_215741


namespace manuscript_fee_3800_l215_215551

theorem manuscript_fee_3800 (tax_fee manuscript_fee : ℕ) 
  (h1 : tax_fee = 420) 
  (h2 : (0 < manuscript_fee) ∧ 
        (manuscript_fee ≤ 4000) → 
        tax_fee = (14 * (manuscript_fee - 800)) / 100) 
  (h3 : (manuscript_fee > 4000) → 
        tax_fee = (11 * manuscript_fee) / 100) : manuscript_fee = 3800 :=
by
  sorry

end manuscript_fee_3800_l215_215551


namespace anna_score_correct_l215_215486

-- Given conditions
def correct_answers : ℕ := 17
def incorrect_answers : ℕ := 6
def unanswered_questions : ℕ := 7
def point_per_correct : ℕ := 1
def point_per_incorrect : ℕ := 0
def deduction_per_unanswered : ℤ := -1 / 2

-- Proving the score
theorem anna_score_correct : 
  correct_answers * point_per_correct + incorrect_answers * point_per_incorrect + unanswered_questions * deduction_per_unanswered = 27 / 2 :=
by
  sorry

end anna_score_correct_l215_215486


namespace triangle_pyramid_angle_l215_215033

theorem triangle_pyramid_angle (φ : ℝ) (vertex_angle : ∀ (A B C : ℝ), (A + B + C = φ)) :
  ∃ θ : ℝ, θ = φ :=
by
  sorry

end triangle_pyramid_angle_l215_215033


namespace mod_multiplication_result_l215_215749

theorem mod_multiplication_result :
  ∃ n : ℕ, 507 * 873 ≡ n [MOD 77] ∧ 0 ≤ n ∧ n < 77 ∧ n = 15 := by
  sorry

end mod_multiplication_result_l215_215749


namespace quadrilateral_sides_equality_l215_215065

theorem quadrilateral_sides_equality 
  (a b c d : ℕ) 
  (h1 : (b + c + d) % a = 0) 
  (h2 : (a + c + d) % b = 0) 
  (h3 : (a + b + d) % c = 0) 
  (h4 : (a + b + c) % d = 0) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) :
  a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equality_l215_215065


namespace gcd_n3_plus_16_n_plus_4_l215_215512

/-- For a given positive integer n > 2^4, the greatest common divisor of n^3 + 16 and n + 4 is 1. -/
theorem gcd_n3_plus_16_n_plus_4 (n : ℕ) (h : n > 2^4) : Nat.gcd (n^3 + 16) (n + 4) = 1 := by
  sorry

end gcd_n3_plus_16_n_plus_4_l215_215512


namespace total_pennies_l215_215610

theorem total_pennies (R G K : ℕ) (h1 : R = 180) (h2 : G = R / 2) (h3 : K = G / 3) : R + G + K = 300 := by
  sorry

end total_pennies_l215_215610


namespace calculate_bus_stoppage_time_l215_215722

variable (speed_excl_stoppages speed_incl_stoppages distance_excl_stoppages distance_incl_stoppages distance_diff time_lost_stoppages : ℝ)

def bus_stoppage_time
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  Prop :=
  speed_excl_stoppages = 32 ∧
  speed_incl_stoppages = 16 ∧
  time_stopped = 30

theorem calculate_bus_stoppage_time 
  (speed_excl_stoppages : ℝ)
  (speed_incl_stoppages : ℝ)
  (time_stopped : ℝ) :
  bus_stoppage_time speed_excl_stoppages speed_incl_stoppages time_stopped :=
by
  have h1 : speed_excl_stoppages = 32 := by
    sorry
  have h2 : speed_incl_stoppages = 16 := by
    sorry
  have h3 : time_stopped = 30 := by
    sorry
  exact ⟨h1, h2, h3⟩

end calculate_bus_stoppage_time_l215_215722


namespace lcm_153_180_560_l215_215389

theorem lcm_153_180_560 : Nat.lcm (Nat.lcm 153 180) 560 = 85680 :=
by
  sorry

end lcm_153_180_560_l215_215389


namespace determine_flower_responsibility_l215_215709

-- Define the structure of the grid
structure Grid (m n : ℕ) :=
  (vertices : Fin m → Fin n → Bool) -- True if gardener lives at the vertex

-- Define a function to determine if 3 gardeners are nearest to a flower
def is_nearest (i j fi fj : ℕ) : Bool :=
  -- Assume this function gives true if the gardener at (i, j) is one of the 3 nearest to the flower at (fi, fj)
  sorry

-- The main theorem statement
theorem determine_flower_responsibility 
  {m n : ℕ} 
  (G : Grid m n) 
  (i j : Fin m) 
  (k : Fin n) 
  (h : G.vertices i k = true) 
  : ∃ (fi fj : ℕ), is_nearest (i : ℕ) (k : ℕ) fi fj = true := 
sorry

end determine_flower_responsibility_l215_215709


namespace M_intersect_N_eq_l215_215912

def M : Set ℝ := { y | ∃ x, y = x ^ 2 }
def N : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 2 + y^2 ≤ 1) }

theorem M_intersect_N_eq : M ∩ { y | (y ∈ Set.univ) } = { y | 0 ≤ y ∧ y ≤ Real.sqrt 2 } :=
by
  sorry

end M_intersect_N_eq_l215_215912


namespace find_original_selling_price_l215_215670

variable (x : ℝ) (discount_rate : ℝ) (final_price : ℝ)

def original_selling_price_exists (x : ℝ) (discount_rate : ℝ) (final_price : ℝ) : Prop :=
  (x * (1 - discount_rate) = final_price) → (x = 700)

theorem find_original_selling_price
  (discount_rate : ℝ := 0.20)
  (final_price : ℝ := 560) :
  ∃ x : ℝ, original_selling_price_exists x discount_rate final_price :=
by
  use 700
  sorry

end find_original_selling_price_l215_215670


namespace length_of_square_side_l215_215244

noncomputable def speed_km_per_hr_to_m_per_s (speed : ℝ) : ℝ :=
  speed * 1000 / 3600

noncomputable def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem length_of_square_side
  (time_seconds : ℝ)
  (speed_km_per_hr : ℝ)
  (distance_m : ℝ)
  (side_length : ℝ)
  (h1 : time_seconds = 72)
  (h2 : speed_km_per_hr = 10)
  (h3 : distance_m = speed_km_per_hr_to_m_per_s speed_km_per_hr * time_seconds)
  (h4 : distance_m = perimeter_of_square side_length) :
  side_length = 50 :=
sorry

end length_of_square_side_l215_215244


namespace simplify_and_evaluate_expression_l215_215491

theorem simplify_and_evaluate_expression (x : ℤ) (h : x = 2 ∨ x ≠ -1 ∧ x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1) :
  ((1 / (x:ℚ) - 1 / (x + 1)) / ((x^2 - 1) / (x^2 + 2*x + 1))) = (1 / 2) :=
by
  -- Skipping the proof
  sorry

end simplify_and_evaluate_expression_l215_215491


namespace perpendicular_slope_l215_215521

-- Define the line equation and the result we want to prove about its perpendicular slope
def line_eq (x y : ℝ) := 5 * x - 2 * y = 10

theorem perpendicular_slope : ∀ (m : ℝ), 
  (∀ (x y : ℝ), line_eq x y → y = (5 / 2) * x - 5) →
  m = -(2 / 5) :=
by
  intros m H
  -- Additional logical steps would go here
  sorry

end perpendicular_slope_l215_215521


namespace rhombus_diagonals_ratio_l215_215681

theorem rhombus_diagonals_ratio (a b d1 d2 : ℝ) 
  (h1: a > 0) (h2: b > 0)
  (h3: d1 = 2 * (a / Real.cos θ))
  (h4: d2 = 2 * (b / Real.cos θ)) :
  d1 / d2 = a / b := 
sorry

end rhombus_diagonals_ratio_l215_215681


namespace eval_p_nested_l215_215990

def p (x y : ℤ) : ℤ :=
  if x ≥ 0 ∧ y ≥ 0 then 2 * x + 3 * y
  else if x < 0 ∧ y < 0 then x ^ 2 - y
  else 4 * x + 2 * y

theorem eval_p_nested :
  p (p 2 (-3)) (p (-4) (-3)) = 61 :=
by
  sorry

end eval_p_nested_l215_215990


namespace find_value_of_f_neg_3_over_2_l215_215277

noncomputable def f : ℝ → ℝ := sorry

theorem find_value_of_f_neg_3_over_2 (h1 : ∀ x : ℝ, f (-x) = -f x) 
    (h2 : ∀ x : ℝ, f (x + 3/2) = -f x) : 
    f (- 3 / 2) = 0 := 
sorry

end find_value_of_f_neg_3_over_2_l215_215277


namespace find_x_squared_inv_x_squared_l215_215508

theorem find_x_squared_inv_x_squared (x : ℝ) (h : x^3 + 1/x^3 = 110) : x^2 + 1/x^2 = 23 :=
sorry

end find_x_squared_inv_x_squared_l215_215508


namespace solve_for_x_l215_215988

theorem solve_for_x (x : ℚ) :  (1/2) * (12 * x + 3) = 3 * x + 2 → x = 1/6 := by
  intro h
  sorry

end solve_for_x_l215_215988


namespace angle_sum_equal_l215_215706

theorem angle_sum_equal 
  (AB AC DE DF : ℝ)
  (h_AB_AC : AB = AC)
  (h_DE_DF : DE = DF)
  (angle_BAC angle_EDF : ℝ)
  (h_angle_BAC : angle_BAC = 40)
  (h_angle_EDF : angle_EDF = 50)
  (angle_DAC angle_ADE : ℝ)
  (h_angle_DAC : angle_DAC = 70)
  (h_angle_ADE : angle_ADE = 65) :
  angle_DAC + angle_ADE = 135 := 
sorry

end angle_sum_equal_l215_215706


namespace move_line_down_l215_215268

theorem move_line_down (x y : ℝ) : (y = -3 * x + 5) → (y = -3 * x + 2) :=
by
  sorry

end move_line_down_l215_215268


namespace product_of_two_numbers_l215_215122

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 21) (h2 : x^2 + y^2 = 527) : x * y = -43 :=
sorry

end product_of_two_numbers_l215_215122


namespace relationship_of_y_values_l215_215029

theorem relationship_of_y_values (k : ℝ) (y₁ y₂ y₃ : ℝ) :
  (y₁ = (k^2 + 3) / (-3)) ∧ (y₂ = (k^2 + 3) / (-1)) ∧ (y₃ = (k^2 + 3) / 2) →
  y₂ < y₁ ∧ y₁ < y₃ :=
by
  intro h
  have h₁ : y₁ = (k^2 + 3) / (-3) := h.1
  have h₂ : y₂ = (k^2 + 3) / (-1) := h.2.1
  have h₃ : y₃ = (k^2 + 3) / 2 := h.2.2
  sorry

end relationship_of_y_values_l215_215029


namespace no_integer_solution_for_equation_l215_215047

theorem no_integer_solution_for_equation :
    ¬ ∃ (x y z : ℤ), x^2 + y^2 + z^2 = x * y * z - 1 :=
by
  sorry

end no_integer_solution_for_equation_l215_215047


namespace blue_hat_cost_is_6_l215_215505

-- Total number of hats is 85
def total_hats : ℕ := 85

-- Number of green hats
def green_hats : ℕ := 20

-- Number of blue hats
def blue_hats : ℕ := total_hats - green_hats

-- Cost of each green hat
def cost_per_green_hat : ℕ := 7

-- Total cost for all hats
def total_cost : ℕ := 530

-- Total cost of green hats
def total_cost_green_hats : ℕ := green_hats * cost_per_green_hat

-- Total cost of blue hats
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats

-- Cost per blue hat
def cost_per_blue_hat : ℕ := total_cost_blue_hats / blue_hats 

-- Prove that the cost of each blue hat is $6
theorem blue_hat_cost_is_6 : cost_per_blue_hat = 6 :=
by
  sorry

end blue_hat_cost_is_6_l215_215505


namespace student_solved_correctly_l215_215163

theorem student_solved_correctly (c e : ℕ) (h1 : c + e = 80) (h2 : 5 * c - 3 * e = 8) : c = 31 :=
sorry

end student_solved_correctly_l215_215163


namespace dodecagon_diagonals_l215_215453

def D (n : ℕ) : ℕ := n * (n - 3) / 2

theorem dodecagon_diagonals : D 12 = 54 :=
by
  sorry

end dodecagon_diagonals_l215_215453


namespace find_a1_of_geom_series_l215_215155

noncomputable def geom_series_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem find_a1_of_geom_series (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : S 6 = 9 * S 3)
  (h2 : S 5 = 62)
  (neq1 : q ≠ 1)
  (neqm1 : q ≠ -1) :
  a₁ = 2 :=
by
  have eq1 : S 6 = geom_series_sum a₁ q 6 := sorry
  have eq2 : S 3 = geom_series_sum a₁ q 3 := sorry
  have eq3 : S 5 = geom_series_sum a₁ q 5 := sorry
  sorry

end find_a1_of_geom_series_l215_215155


namespace volume_of_solid_l215_215119

def x_y_relation (x y : ℝ) : Prop := x = (y - 2)^(1/3)
def x1 (x : ℝ) : Prop := x = 1
def y1 (y : ℝ) : Prop := y = 1

theorem volume_of_solid :
  ∀ (x y : ℝ),
    (x_y_relation x y ∧ x1 x ∧ y1 y) →
    ∃ V : ℝ, V = (44 / 7) * Real.pi :=
by
  -- Proof will go here
  sorry

end volume_of_solid_l215_215119


namespace volume_removed_percentage_l215_215479

noncomputable def volume_of_box (length width height : ℝ) : ℝ := 
  length * width * height

noncomputable def volume_of_cube (side : ℝ) : ℝ := 
  side ^ 3

noncomputable def volume_removed (length width height side : ℝ) : ℝ :=
  8 * (volume_of_cube side)

noncomputable def percentage_removed (length width height side : ℝ) : ℝ :=
  (volume_removed length width height side) / (volume_of_box length width height) * 100

theorem volume_removed_percentage :
  percentage_removed 20 15 12 4 = 14.22 := 
by
  sorry

end volume_removed_percentage_l215_215479


namespace perpendicular_distance_l215_215432

structure Vertex :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def S : Vertex := ⟨6, 0, 0⟩
def P : Vertex := ⟨0, 0, 0⟩
def Q : Vertex := ⟨0, 5, 0⟩
def R : Vertex := ⟨0, 0, 4⟩

noncomputable def distance_from_point_to_plane (S P Q R : Vertex) : ℝ := sorry

theorem perpendicular_distance (S P Q R : Vertex) (hS : S = ⟨6, 0, 0⟩) (hP : P = ⟨0, 0, 0⟩) (hQ : Q = ⟨0, 5, 0⟩) (hR : R = ⟨0, 0, 4⟩) :
  distance_from_point_to_plane S P Q R = 6 :=
  sorry

end perpendicular_distance_l215_215432


namespace determine_some_number_l215_215111

theorem determine_some_number (x : ℝ) (n : ℝ) (hx : x = 1.5) (h : (3 + 2 * x)^5 = (1 + n * x)^4) : n = 10 / 3 :=
by {
  sorry
}

end determine_some_number_l215_215111


namespace number_of_girls_l215_215934

theorem number_of_girls (B G : ℕ) (h₁ : B = 6 * G / 5) (h₂ : B + G = 440) : G = 200 :=
by {
  sorry -- Proof steps here
}

end number_of_girls_l215_215934


namespace tangent_curve_line_a_eq_neg1_l215_215823

theorem tangent_curve_line_a_eq_neg1 (a : ℝ) (x : ℝ) : 
  (∀ (x : ℝ), (e^x + a = x) ∧ (e^x = 1) ) → a = -1 :=
by 
  intro h
  sorry

end tangent_curve_line_a_eq_neg1_l215_215823


namespace net_rate_of_pay_l215_215871

/-- The net rate of pay in dollars per hour for a truck driver after deducting gasoline expenses. -/
theorem net_rate_of_pay
  (hrs : ℕ) (speed : ℕ) (miles_per_gallon : ℕ) (pay_per_mile : ℚ) (cost_per_gallon : ℚ) 
  (H1 : hrs = 3)
  (H2 : speed = 50)
  (H3 : miles_per_gallon = 25)
  (H4 : pay_per_mile = 0.6)
  (H5 : cost_per_gallon = 2.50) :
  pay_per_mile * (hrs * speed) - cost_per_gallon * ((hrs * speed) / miles_per_gallon) = 25 * hrs :=
by sorry

end net_rate_of_pay_l215_215871


namespace right_triangular_prism_volume_l215_215322

theorem right_triangular_prism_volume (R a h V : ℝ)
  (h1 : 4 * Real.pi * R^2 = 12 * Real.pi)
  (h2 : h = 2 * R)
  (h3 : (1 / 3) * (Real.sqrt 3 / 2) * a = R)
  (h4 : V = (1 / 2) * a * a * (Real.sin (Real.pi / 3)) * h) :
  V = 54 :=
by sorry

end right_triangular_prism_volume_l215_215322


namespace three_digit_decimal_bounds_l215_215463

def is_rounded_half_up (x : ℝ) (y : ℝ) : Prop :=
  (y - 0.005 ≤ x) ∧ (x < y + 0.005)

theorem three_digit_decimal_bounds :
  ∃ (x : ℝ), (8.725 ≤ x) ∧ (x ≤ 8.734) ∧ is_rounded_half_up x 8.73 :=
by
  sorry

end three_digit_decimal_bounds_l215_215463


namespace trigonometric_expression_value_l215_215411

variable (θ : ℝ)

-- Conditions
axiom tan_theta_eq_two : Real.tan θ = 2

-- Theorem to prove
theorem trigonometric_expression_value : 
  Real.sin θ * Real.sin θ + 
  Real.sin θ * Real.cos θ - 
  2 * Real.cos θ * Real.cos θ = 4 / 5 := 
by
  sorry

end trigonometric_expression_value_l215_215411


namespace annual_rent_per_square_foot_is_156_l215_215296

-- Given conditions
def monthly_rent : ℝ := 1300
def length : ℝ := 10
def width : ℝ := 10
def area : ℝ := length * width
def annual_rent : ℝ := monthly_rent * 12

-- Proof statement: Annual rent per square foot
theorem annual_rent_per_square_foot_is_156 : 
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_is_156_l215_215296


namespace min_value_expression_l215_215624

theorem min_value_expression (m n : ℝ) (h : m > 2 * n) :
  m + (4 * n^2 - 2 * m * n + 9) / (m - 2 * n) ≥ 6 :=
by
  sorry

end min_value_expression_l215_215624


namespace find_pairs_nat_numbers_l215_215073

theorem find_pairs_nat_numbers (a b : ℕ) :
  (a^3 * b - 1) % (a + 1) = 0 ∧ (a * b^3 + 1) % (b - 1) = 0 ↔ 
  (a = 2 ∧ b = 2) ∨ (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 3) :=
by
  sorry

end find_pairs_nat_numbers_l215_215073


namespace thor_jumps_to_exceed_29000_l215_215894

theorem thor_jumps_to_exceed_29000 :
  ∃ (n : ℕ), (3 ^ n) > 29000 ∧ n = 10 := sorry

end thor_jumps_to_exceed_29000_l215_215894


namespace rhombus_diagonals_l215_215666

theorem rhombus_diagonals (x y : ℝ) 
  (h1 : x * y = 234)
  (h2 : x + y = 31) :
  (x = 18 ∧ y = 13) ∨ (x = 13 ∧ y = 18) := by
sorry

end rhombus_diagonals_l215_215666


namespace balls_sum_l215_215123

theorem balls_sum (m n : ℕ) (h₁ : ∀ a, a ∈ ({m, 8, n} : Finset ℕ)) -- condition: balls are identical except for color
  (h₂ : (8 : ℝ) / (m + 8 + n) = (m + n : ℝ) / (m + 8 + n)) : m + n = 8 :=
sorry

end balls_sum_l215_215123


namespace length_error_probability_l215_215212

theorem length_error_probability
  (μ σ : ℝ)
  (X : ℝ → ℝ)
  (h_norm_dist : ∀ x : ℝ, X x = (Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) / (σ * Real.sqrt (2 * Real.pi))))
  (h_max_density : X 0 = 1 / (3 * Real.sqrt (2 * Real.pi)))
  (P : Set ℝ → ℝ)
  (h_prop1 : P {x | μ - σ < x ∧ x < μ + σ} = 0.6826)
  (h_prop2 : P {x | μ - 2 * σ < x ∧ x < μ + 2 * σ} = 0.9544) :
  P {x | 3 < x ∧ x < 6} = 0.1359 :=
sorry

end length_error_probability_l215_215212


namespace calculation_simplifies_l215_215893

theorem calculation_simplifies :
  120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end calculation_simplifies_l215_215893


namespace parabola_points_relation_l215_215485

theorem parabola_points_relation {a b c y1 y2 y3 : ℝ} 
  (hA : y1 = a * (1 / 2)^2 + b * (1 / 2) + c)
  (hB : y2 = a * (0)^2 + b * (0) + c)
  (hC : y3 = a * (-1)^2 + b * (-1) + c)
  (h_cond : 0 < 2 * a ∧ 2 * a < b) : 
  y1 > y2 ∧ y2 > y3 :=
by 
  sorry

end parabola_points_relation_l215_215485


namespace find_f_8_l215_215007

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity : ∀ x : ℝ, f (x + 6) = f x
axiom function_on_interval : ∀ x : ℝ, -3 < x ∧ x < 0 → f x = 2 * x - 5

theorem find_f_8 : f 8 = -9 :=
by
  sorry

end find_f_8_l215_215007


namespace yuna_initial_pieces_l215_215549

variable (Y : ℕ)

theorem yuna_initial_pieces
  (namjoon_initial : ℕ := 250)
  (given_pieces : ℕ := 60)
  (namjoon_after : namjoon_initial - given_pieces = Y + given_pieces - 20) :
  Y = 150 :=
by
  sorry

end yuna_initial_pieces_l215_215549


namespace fixed_point_coordinates_l215_215985

noncomputable def fixed_point (A : Real × Real) : Prop :=
∀ (k : Real), ∃ (x y : Real), A = (x, y) ∧ (3 + k) * x + (1 - 2 * k) * y + 1 + 5 * k = 0

theorem fixed_point_coordinates :
  fixed_point (-1, 2) :=
by
  sorry

end fixed_point_coordinates_l215_215985


namespace find_original_price_l215_215423

-- Defining constants and variables
def original_price (P : ℝ) : Prop :=
  let cost_after_repairs := P + 13000
  let selling_price := 66900
  let profit := selling_price - cost_after_repairs
  let profit_percent := profit / P * 100
  profit_percent = 21.636363636363637

theorem find_original_price : ∃ P : ℝ, original_price P :=
  by
  sorry

end find_original_price_l215_215423


namespace derivative_of_f_is_l215_215228

noncomputable def f (x : ℝ) : ℝ := (x + 1) ^ 2

theorem derivative_of_f_is (x : ℝ) : deriv f x = 2 * x + 2 :=
by
  sorry

end derivative_of_f_is_l215_215228


namespace drink_price_half_promotion_l215_215677

theorem drink_price_half_promotion (P : ℝ) (h : P + (1/2) * P = 13.5) : P = 9 := 
by
  sorry

end drink_price_half_promotion_l215_215677


namespace inequality_solution_l215_215055

theorem inequality_solution (a : ℝ) : (∀ x : ℝ, (a + 1) * x > a + 1 → x < 1) ↔ a < -1 := by
  sorry

end inequality_solution_l215_215055


namespace prove_expression_l215_215324

def otimes (a b : ℚ) : ℚ := a^2 / b

theorem prove_expression : ((otimes (otimes 1 2) 3) - (otimes 1 (otimes 2 3))) = -2/3 :=
by 
  sorry

end prove_expression_l215_215324


namespace find_constants_monotonicity_l215_215787

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f' (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_constants (a b c : ℝ) 
  (h1 : f' (-2/3) a b = 0)
  (h2 : f' 1 a b = 0) :
  a = -1/2 ∧ b = -2 :=
by sorry

theorem monotonicity (a b c : ℝ)
  (h1 : a = -1/2) 
  (h2 : b = -2) : 
  (∀ x : ℝ, x < -2/3 → f' x a b > 0) ∧ 
  (∀ x : ℝ, -2/3 < x ∧ x < 1 → f' x a b < 0) ∧ 
  (∀ x : ℝ, x > 1 → f' x a b > 0) :=
by sorry

end find_constants_monotonicity_l215_215787


namespace fresh_grapes_weight_l215_215926

theorem fresh_grapes_weight :
  ∀ (F : ℝ), (∀ (water_content_fresh : ℝ) (water_content_dried : ℝ) (weight_dried : ℝ),
    water_content_fresh = 0.90 → water_content_dried = 0.20 → weight_dried = 3.125 →
    (F * 0.10 = 0.80 * weight_dried) → F = 78.125) := 
by
  intros F
  intros water_content_fresh water_content_dried weight_dried
  intros h1 h2 h3 h4
  sorry

end fresh_grapes_weight_l215_215926


namespace total_airflow_correct_l215_215516

def airflow_fan_A : ℕ := 10 * 10 * 60 * 7
def airflow_fan_B : ℕ := 15 * 20 * 60 * 5
def airflow_fan_C : ℕ := 25 * 30 * 60 * 5
def airflow_fan_D : ℕ := 20 * 15 * 60 * 2
def airflow_fan_E : ℕ := 30 * 60 * 60 * 6

def total_airflow : ℕ :=
  airflow_fan_A + airflow_fan_B + airflow_fan_C + airflow_fan_D + airflow_fan_E

theorem total_airflow_correct : total_airflow = 1041000 := by
  sorry

end total_airflow_correct_l215_215516


namespace find_positive_Y_for_nine_triangle_l215_215079

def triangle_relation (X Y : ℝ) : ℝ := X^2 + 3 * Y^2

theorem find_positive_Y_for_nine_triangle (Y : ℝ) : (9^2 + 3 * Y^2 = 360) → Y = Real.sqrt 93 := 
by
  sorry

end find_positive_Y_for_nine_triangle_l215_215079


namespace coordinates_of_point_P_l215_215297

theorem coordinates_of_point_P (x y : ℝ) (h1 : x > 0) (h2 : y < 0) (h3 : abs y = 2) (h4 : abs x = 4) : (x, y) = (4, -2) :=
by
  sorry

end coordinates_of_point_P_l215_215297


namespace isosceles_right_triangle_angle_l215_215582

-- Define the conditions given in the problem
def is_isosceles (a b c : ℝ) : Prop := 
(a = b ∨ b = c ∨ c = a)

def is_right_triangle (a b c : ℝ) : Prop := 
(a = 90 ∨ b = 90 ∨ c = 90)

def angles_sum_to_180 (a b c : ℝ) : Prop :=
a + b + c = 180

-- The Proof Problem
theorem isosceles_right_triangle_angle :
  ∀ (a b c x : ℝ), (is_isosceles a b c) → (is_right_triangle a b c) → (angles_sum_to_180 a b c) → (x = a ∨ x = b ∨ x = c) → x = 45 :=
by
  intros a b c x h_isosceles h_right h_sum h_x
  -- Proof is omitted with sorry
  sorry

end isosceles_right_triangle_angle_l215_215582


namespace find_number_l215_215730

theorem find_number (x : ℤ) (h : (85 + x) * 1 = 9637) : x = 9552 :=
by
  sorry

end find_number_l215_215730


namespace intersection_A_B_subset_A_B_l215_215715

-- Definitions for the sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 2 * a - 1 ≤ x ∧ x ≤ a + 3}
def set_B : Set ℝ := {x | x < -1 ∨ x > 5}

-- First proof problem: Intersection
theorem intersection_A_B (a : ℝ) (ha : a = -2) :
  set_A a ∩ set_B = {x | -5 ≤ x ∧ x < -1} :=
sorry

-- Second proof problem: Subset
theorem subset_A_B (a : ℝ) :
  set_A a ⊆ set_B ↔ (a ≤ -4 ∨ a ≥ 3) :=
sorry

end intersection_A_B_subset_A_B_l215_215715


namespace vector_dot_product_l215_215157

theorem vector_dot_product
  (AB : ℝ × ℝ) (BC : ℝ × ℝ)
  (t : ℝ)
  (hAB : AB = (2, 3))
  (hBC : BC = (3, t))
  (ht : t > 0)
  (hmagnitude : (3^2 + t^2).sqrt = (10:ℝ).sqrt) :
  (AB.1 * (AB.1 + BC.1) + AB.2 * (AB.2 + BC.2) = 22) :=
by
  sorry

end vector_dot_product_l215_215157


namespace problem_statement_l215_215050

theorem problem_statement : 2456 + 144 / 12 * 5 - 256 = 2260 := 
by
  -- statements and proof steps would go here
  sorry

end problem_statement_l215_215050


namespace solve_quadratic_equation_l215_215202

theorem solve_quadratic_equation (x : ℝ) : 2 * (x + 1) ^ 2 - 49 = 1 ↔ (x = 4 ∨ x = -6) := 
sorry

end solve_quadratic_equation_l215_215202


namespace find_deducted_salary_l215_215753

noncomputable def dailyWage (weeklySalary : ℝ) (workingDays : ℕ) : ℝ := weeklySalary / workingDays

noncomputable def totalDeduction (dailyWage : ℝ) (absentDays : ℕ) : ℝ := dailyWage * absentDays

noncomputable def deductedSalary (weeklySalary : ℝ) (totalDeduction : ℝ) : ℝ := weeklySalary - totalDeduction

theorem find_deducted_salary
  (weeklySalary : ℝ := 791)
  (workingDays : ℕ := 5)
  (absentDays : ℕ := 4)
  (dW := dailyWage weeklySalary workingDays)
  (tD := totalDeduction dW absentDays)
  (dS := deductedSalary weeklySalary tD) :
  dS = 158.20 := 
  by
    sorry

end find_deducted_salary_l215_215753


namespace simplify_expression_l215_215766

theorem simplify_expression (a : ℝ) (h : a > 0) : 
  (a^2 / (a * (a^3) ^ (1 / 2)) ^ (1 / 3)) = a^(7 / 6) :=
sorry

end simplify_expression_l215_215766


namespace inequality_solution_l215_215174

-- Condition definitions in lean
def numerator (x : ℝ) : ℝ := (x^5 - 13 * x^3 + 36 * x) * (x^4 - 17 * x^2 + 16)
def denominator (y : ℝ) : ℝ := (y^5 - 13 * y^3 + 36 * y) * (y^4 - 17 * y^2 + 16)

-- Given the critical conditions
def is_zero_or_pm1_pm2_pm3_pm4 (y : ℝ) : Prop := 
  y = 0 ∨ y = 1 ∨ y = -1 ∨ y = 2 ∨ y = -2 ∨ y = 3 ∨ y = -3 ∨ y = 4 ∨ y = -4

-- The theorem statement
theorem inequality_solution (x y : ℝ) : 
  (numerator x / denominator y) ≥ 0 ↔ ¬ (is_zero_or_pm1_pm2_pm3_pm4 y) :=
sorry -- proof to be filled in later

end inequality_solution_l215_215174


namespace necessary_but_not_sufficient_condition_l215_215374

variables (p q : Prop)

theorem necessary_but_not_sufficient_condition
  (h : ¬p → q) (hn : ¬q → p) : 
  (p → ¬q) ∧ ¬(¬q → p) :=
sorry

end necessary_but_not_sufficient_condition_l215_215374


namespace koi_fish_multiple_l215_215300

theorem koi_fish_multiple (n m : ℕ) (h1 : n = 39) (h2 : m * n - 64 < n) : m * n = 78 :=
by
  sorry

end koi_fish_multiple_l215_215300


namespace comp_functions_l215_215756

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

theorem comp_functions (x : ℝ) : f (g x) = 6 * x - 7 :=
by
  sorry

end comp_functions_l215_215756


namespace least_positive_integer_x_l215_215640

theorem least_positive_integer_x (x : ℕ) (h1 : x + 3721 ≡ 1547 [MOD 12]) (h2 : x % 2 = 0) : x = 2 :=
sorry

end least_positive_integer_x_l215_215640


namespace equation_of_line_l215_215544

theorem equation_of_line (x_intercept slope : ℝ)
  (hx : x_intercept = 2) (hm : slope = 1) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -1 ∧ c = -2 ∧ (∀ x y : ℝ, y = slope * (x - x_intercept) ↔ a * x + b * y + c = 0) := sorry

end equation_of_line_l215_215544


namespace Jerry_has_36_stickers_l215_215629

variable (FredStickers GeorgeStickers JerryStickers CarlaStickers : ℕ)
variable (h1 : FredStickers = 18)
variable (h2 : GeorgeStickers = FredStickers - 6)
variable (h3 : JerryStickers = 3 * GeorgeStickers)
variable (h4 : CarlaStickers = JerryStickers + JerryStickers / 4)
variable (h5 : GeorgeStickers + FredStickers = CarlaStickers ^ 2)

theorem Jerry_has_36_stickers : JerryStickers = 36 := by
  sorry

end Jerry_has_36_stickers_l215_215629


namespace solve_for_s_l215_215493

theorem solve_for_s (s t : ℚ) (h1 : 15 * s + 7 * t = 210) (h2 : t = 3 * s) : s = 35 / 6 := 
by
  sorry

end solve_for_s_l215_215493


namespace PQRS_value_l215_215833

theorem PQRS_value
  (P Q R S : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q)
  (hR : 0 < R)
  (hS : 0 < S)
  (h1 : Real.log (P * Q) / Real.log 10 + Real.log (P * S) / Real.log 10 = 2)
  (h2 : Real.log (Q * S) / Real.log 10 + Real.log (Q * R) / Real.log 10 = 3)
  (h3 : Real.log (R * P) / Real.log 10 + Real.log (R * S) / Real.log 10 = 5) :
  P * Q * R * S = 100000 := 
sorry

end PQRS_value_l215_215833


namespace Claire_takes_6_photos_l215_215676

-- Define the number of photos Claire has taken
variable (C : ℕ)

-- Define the conditions as stated in the problem
def Lisa_photos := 3 * C
def Robert_photos := C + 12
def same_number_photos := Lisa_photos C = Robert_photos C

-- The goal is to prove that C = 6
theorem Claire_takes_6_photos (h : same_number_photos C) : C = 6 := by
  sorry

end Claire_takes_6_photos_l215_215676


namespace initial_stickers_correct_l215_215018

-- Definitions based on the conditions
def initial_stickers (X : ℕ) : ℕ := X
def after_buying (X : ℕ) : ℕ := X + 26
def after_birthday (X : ℕ) : ℕ := after_buying X + 20
def after_giving (X : ℕ) : ℕ := after_birthday X - 6
def after_decorating (X : ℕ) : ℕ := after_giving X - 58

-- Theorem stating the problem and the expected answer
theorem initial_stickers_correct (X : ℕ) (h : after_decorating X = 2) : initial_stickers X = 26 :=
by {
  sorry
}

end initial_stickers_correct_l215_215018


namespace solve_inequality_l215_215139

theorem solve_inequality (k x : ℝ) :
  (x^2 > (k + 1) * x - k) ↔ 
  (if k > 1 then (x < 1 ∨ x > k)
   else if k = 1 then (x ≠ 1)
   else (x < k ∨ x > 1)) :=
by
  sorry

end solve_inequality_l215_215139


namespace triangle_hypotenuse_segments_l215_215989

theorem triangle_hypotenuse_segments :
  ∀ (x : ℝ) (BC AC : ℝ),
  BC / AC = 3 / 7 →
  ∃ (h : ℝ) (BD AD : ℝ),
    h = 42 ∧
    BD * AD = h^2 ∧
    BD / AD = 9 / 49 ∧
    BD = 18 ∧
    AD = 98 :=
by
  sorry

end triangle_hypotenuse_segments_l215_215989


namespace length_of_c_l215_215260

theorem length_of_c (a b c : ℝ) (h1 : a = 1) (h2 : b = 3) (h_triangle : 0 < c) :
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) → c = 3 :=
by
  intros h_ineq
  sorry

end length_of_c_l215_215260


namespace relationship_ab_c_l215_215573

def a := 0.8 ^ 0.8
def b := 0.8 ^ 0.9
def c := 1.2 ^ 0.8

theorem relationship_ab_c : c > a ∧ a > b := 
by
  -- The proof would go here
  sorry

end relationship_ab_c_l215_215573


namespace harvest_season_weeks_l215_215853

-- Definitions based on given conditions
def weekly_earnings : ℕ := 491
def weekly_rent : ℕ := 216
def total_savings : ℕ := 324775

-- Definition to calculate net earnings per week
def net_earnings_per_week (earnings rent : ℕ) : ℕ :=
  earnings - rent

-- Definition to calculate number of weeks
def number_of_weeks (savings net_earnings : ℕ) : ℕ :=
  savings / net_earnings

theorem harvest_season_weeks :
  number_of_weeks total_savings (net_earnings_per_week weekly_earnings weekly_rent) = 1181 :=
by
  sorry

end harvest_season_weeks_l215_215853


namespace line_MN_parallel_to_y_axis_l215_215665

-- Definition of points
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of vector between two points
def vector_between (P Q : Point) : Point :=
  { x := Q.x - P.x,
    y := Q.y - P.y,
    z := Q.z - P.z }

-- Given points M and N
def M : Point := { x := 3, y := -2, z := 1 }
def N : Point := { x := 3, y := 2, z := 1 }

-- The vector \overrightarrow{MN}
def vec_MN : Point := vector_between M N

-- Theorem: The vector between points M and N is parallel to the y-axis
theorem line_MN_parallel_to_y_axis : vec_MN = {x := 0, y := 4, z := 0} := by
  sorry

end line_MN_parallel_to_y_axis_l215_215665


namespace ellipse_foci_coordinates_l215_215770

theorem ellipse_foci_coordinates :
  ∃ x y : Real, (3 * x^2 + 4 * y^2 = 12) ∧ ((x = 1 ∧ y = 0) ∨ (x = -1 ∧ y = 0)) :=
by
  sorry

end ellipse_foci_coordinates_l215_215770


namespace length_of_chord_l215_215498

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Define the line y = x - 1 with slope 1 passing through the focus (1, 0)
def line (x y : ℝ) : Prop :=
  y = x - 1

-- Prove that the length of the chord |AB| is 8
theorem length_of_chord 
  (x1 y1 x2 y2 : ℝ) 
  (h1 : parabola x1 y1) 
  (h2 : parabola x2 y2) 
  (h3 : line x1 y1) 
  (h4 : line x2 y2) : 
  abs (x2 - x1) = 8 :=
sorry

end length_of_chord_l215_215498


namespace Felicity_used_23_gallons_l215_215865

variable (A Felicity : ℕ)
variable (h1 : Felicity = 4 * A - 5)
variable (h2 : A + Felicity = 30)

theorem Felicity_used_23_gallons : Felicity = 23 := by
  -- Proof steps would go here
  sorry

end Felicity_used_23_gallons_l215_215865


namespace find_the_number_l215_215323

theorem find_the_number (x : ℝ) (h : 100 - x = x + 40) : x = 30 :=
sorry

end find_the_number_l215_215323


namespace num_students_59_l215_215399

theorem num_students_59 (apples : ℕ) (taken_each : ℕ) (students : ℕ) 
  (h_apples : apples = 120) 
  (h_taken_each : taken_each = 2) 
  (h_students_divisors : ∀ d, d = 59 → d ∣ (apples / taken_each)) : students = 59 :=
sorry

end num_students_59_l215_215399


namespace find_f_of_16_l215_215097

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x ^ a

theorem find_f_of_16 : (∃ a : ℝ, f 2 a = Real.sqrt 2) → f 16 (1/2) = 4 :=
by
  intro h
  sorry

end find_f_of_16_l215_215097


namespace employee_y_payment_l215_215718

theorem employee_y_payment (X Y : ℝ) (h1 : X + Y = 616) (h2 : X = 1.2 * Y) : Y = 280 :=
by
  sorry

end employee_y_payment_l215_215718


namespace irrational_of_sqrt_3_l215_215852

noncomputable def is_irritational (x : ℝ) : Prop :=
  ¬ (∃ p q : ℤ, q ≠ 0 ∧ x = p / q)

theorem irrational_of_sqrt_3 :
  is_irritational 0 = false ∧
  is_irritational 3.14 = false ∧
  is_irritational (-1) = false ∧
  is_irritational (Real.sqrt 3) = true := 
by
  -- Proof omitted
  sorry

end irrational_of_sqrt_3_l215_215852


namespace problem_l215_215335

open Real

theorem problem (x y : ℝ) (h1 : 3 * x + 2 * y = 8) (h2 : 2 * x + 3 * y = 11) :
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 2041 / 25 :=
sorry

end problem_l215_215335


namespace graph_inverse_prop_function_quadrants_l215_215914

theorem graph_inverse_prop_function_quadrants :
  ∀ x : ℝ, x ≠ 0 → (x > 0 ∧ y = 4 / x → y > 0) ∨ (x < 0 ∧ y = 4 / x → y < 0) := 
sorry

end graph_inverse_prop_function_quadrants_l215_215914


namespace base4_arithmetic_l215_215120

theorem base4_arithmetic :
  (Nat.ofDigits 4 [2, 3, 1] * Nat.ofDigits 4 [2, 2] / Nat.ofDigits 4 [3]) = Nat.ofDigits 4 [2, 2, 1] := by
sorry

end base4_arithmetic_l215_215120


namespace fraction_simplification_l215_215683

theorem fraction_simplification : (8 : ℝ) / (4 * 25) = 0.08 :=
by
  sorry

end fraction_simplification_l215_215683


namespace sphere_volume_diameter_l215_215019

theorem sphere_volume_diameter {D : ℝ} : 
  (D^3/2 + (1/21) * (D^3/2)) = (π * D^3 / 6) ↔ π = 22 / 7 := 
sorry

end sphere_volume_diameter_l215_215019


namespace odd_square_sum_of_consecutive_l215_215617

theorem odd_square_sum_of_consecutive (n : ℤ) (h_odd : n % 2 = 1) (h_gt : n > 1) : 
  ∃ (j : ℤ), n^2 = j + (j + 1) :=
by
  sorry

end odd_square_sum_of_consecutive_l215_215617


namespace below_sea_level_is_negative_l215_215740
-- Lean 4 statement


theorem below_sea_level_is_negative 
  (above_sea_pos : ∀ x : ℝ, x > 0 → x = x)
  (below_sea_neg : ∀ x : ℝ, x < 0 → x = x) : 
  (-25 = -25) :=
by 
  -- here we are supposed to provide the proof but we are skipping it with sorry
  sorry

end below_sea_level_is_negative_l215_215740


namespace find_X_l215_215128

theorem find_X (X : ℕ) (h1 : 2 + 1 + 3 + X = 3 + 4 + 5) : X = 6 :=
by
  sorry

end find_X_l215_215128


namespace sum_of_fractions_l215_215836

theorem sum_of_fractions : 
  (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8) = 3 / 8) :=
by sorry

end sum_of_fractions_l215_215836


namespace profits_equal_l215_215776

-- Define the profit variables
variables (profitA profitB profitC profitD : ℝ)

-- The conditions
def storeA_profit : profitA = 1.2 * profitB := sorry
def storeB_profit : profitB = 1.2 * profitC := sorry
def storeD_profit : profitD = profitA * 0.6 := sorry

-- The statement to be proven
theorem profits_equal : profitC = profitD :=
by sorry

end profits_equal_l215_215776


namespace spade_to_heart_l215_215768

-- Definition for spade and heart can be abstract geometric shapes
structure Spade := (arcs_top: ℕ) (stem_bottom: ℕ)
structure Heart := (arcs_top: ℕ) (pointed_bottom: ℕ)

-- Condition: the spade symbol must be cut into three parts
def cut_spade (s: Spade) : List (ℕ × ℕ) :=
  [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)]

-- Define a function to verify if the rearranged parts form a heart
def can_form_heart (pieces: List (ℕ × ℕ)) : Prop :=
  pieces = [(1, 0), (0, 1), (0, 1)]

-- The theorem that the spade parts can form a heart
theorem spade_to_heart (s: Spade) (h: Heart):
  (cut_spade s) = [(s.arcs_top, 0), (0, s.stem_bottom), (0, s.stem_bottom)] →
  can_form_heart [(s.arcs_top, 0), (s.stem_bottom, 0), (s.stem_bottom, 0)] := 
by
  sorry


end spade_to_heart_l215_215768


namespace algae_free_day_22_l215_215685

def algae_coverage (day : ℕ) : ℝ :=
if day = 25 then 1 else 2 ^ (25 - day)

theorem algae_free_day_22 :
  1 - algae_coverage 22 = 0.875 :=
by
  -- Proof to be filled in
  sorry

end algae_free_day_22_l215_215685


namespace my_age_now_l215_215846

theorem my_age_now (Y S : ℕ) (h1 : Y - 9 = 5 * (S - 9)) (h2 : Y = 3 * S) : Y = 54 := by
  sorry

end my_age_now_l215_215846


namespace charlie_cookies_l215_215566

theorem charlie_cookies (father_cookies mother_cookies total_cookies charlie_cookies : ℕ)
  (h1 : father_cookies = 10) (h2 : mother_cookies = 5) (h3 : total_cookies = 30) :
  father_cookies + mother_cookies + charlie_cookies = total_cookies → charlie_cookies = 15 :=
by
  intros h
  sorry

end charlie_cookies_l215_215566


namespace simplify_expression_l215_215717

theorem simplify_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + b^3 = 3 * (a + b)) :
  (a / b) + (b / a) - (3 / (a * b)) = 1 := 
sorry

end simplify_expression_l215_215717


namespace expand_binom_l215_215301

theorem expand_binom (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 :=
by
  sorry

end expand_binom_l215_215301


namespace steven_peaches_l215_215690

theorem steven_peaches (jake_peaches : ℕ) (steven_peaches : ℕ) (h1 : jake_peaches = 3) (h2 : jake_peaches + 10 = steven_peaches) : steven_peaches = 13 :=
by
  sorry

end steven_peaches_l215_215690


namespace max_area_basketball_court_l215_215713

theorem max_area_basketball_court : 
  ∃ l w : ℝ, 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l * w = 10000 :=
by {
  -- We are skipping the proof for now
  sorry
}

end max_area_basketball_court_l215_215713


namespace negation_of_implication_l215_215162

variable (a b c : ℝ)

theorem negation_of_implication :
  (¬(a + b + c = 3) → a^2 + b^2 + c^2 < 3) ↔
  ¬((a + b + c = 3) → a^2 + b^2 + c^2 ≥ 3) := by
sorry

end negation_of_implication_l215_215162


namespace decagon_diagonals_intersect_probability_l215_215499

theorem decagon_diagonals_intersect_probability :
  let n := 10  -- number of vertices in decagon
  let diagonals := n * (n - 3) / 2  -- number of diagonals in decagon
  let pairs_diagonals := (diagonals * (diagonals - 1)) / 2  -- ways to choose 2 diagonals from diagonals
  let ways_choose_4 := Nat.choose 10 4  -- ways to choose 4 vertices from 10
  let probability := (4 * ways_choose_4) / pairs_diagonals  -- four vertices chosen determine two intersecting diagonals forming a convex quadrilateral
  probability = (210 / 595) := by
  -- Definitions (diagonals, pairs_diagonals, ways_choose_4) are directly used as hypothesis

  sorry  -- skipping the proof

end decagon_diagonals_intersect_probability_l215_215499


namespace somu_age_ratio_l215_215960

theorem somu_age_ratio (S F : ℕ) (h1 : S = 20) (h2 : S - 10 = (F - 10) / 5) : S / F = 1 / 3 :=
by
  sorry

end somu_age_ratio_l215_215960


namespace problem_extraneous_root_l215_215325

theorem problem_extraneous_root (m : ℤ) :
  (∃ x, x = -4 ∧ (x + 4 = 0) ∧ ((x-1)/(x+4) = m/(x+4)) ∧ (m = -5)) :=
sorry

end problem_extraneous_root_l215_215325


namespace votes_cast_46800_l215_215279

-- Define the election context
noncomputable def total_votes (v : ℕ) : Prop :=
  let percentage_a := 0.35
  let percentage_b := 0.40
  let vote_diff := 2340
  (percentage_b - percentage_a) * (v : ℝ) = (vote_diff : ℝ)

-- Theorem stating the total number of votes cast in the election
theorem votes_cast_46800 : total_votes 46800 :=
by
  sorry

end votes_cast_46800_l215_215279


namespace parabola_translation_l215_215885

theorem parabola_translation :
  ∀ (x : ℝ),
  (∃ x' y' : ℝ, x' = x - 1 ∧ y' = 2 * x' ^ 2 - 3 ∧ y = y' + 3) →
  (y = 2 * x ^ 2) :=
by
  sorry

end parabola_translation_l215_215885


namespace xy_sum_is_one_l215_215540

theorem xy_sum_is_one (x y : ℝ) (h : x^2 + y^2 + x * y = 12 * x - 8 * y + 2) : x + y = 1 :=
sorry

end xy_sum_is_one_l215_215540


namespace geom_seq_product_l215_215420

theorem geom_seq_product (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n, a (n + 1) = r * a n)
  (h_prod : a 5 * a 14 = 5) :
  a 8 * a 9 * a 10 * a 11 = 10 := 
sorry

end geom_seq_product_l215_215420


namespace solution_set_of_inequality_l215_215799

theorem solution_set_of_inequality :
  { x : ℝ | x^2 + x - 12 < 0 } = { x : ℝ | -4 < x ∧ x < 3 } :=
by
  sorry

end solution_set_of_inequality_l215_215799


namespace multiplicative_inverse_correct_l215_215948

def A : ℕ := 123456
def B : ℕ := 654321
def m : ℕ := 1234567
def AB_mod : ℕ := (A * B) % m

def N : ℕ := 513629

theorem multiplicative_inverse_correct (h : AB_mod = 470160) : (470160 * N) % m = 1 := 
by 
  have hN : N = 513629 := rfl
  have hAB : AB_mod = 470160 := h
  sorry

end multiplicative_inverse_correct_l215_215948


namespace sector_angle_l215_215635

theorem sector_angle (r L : ℝ) (h1 : r = 1) (h2 : L = 4) : abs (L - 2 * r) = 2 :=
by 
  -- This is the statement of our proof problem
  -- and does not include the proof itself.
  sorry

end sector_angle_l215_215635


namespace pradeep_passing_percentage_l215_215506

theorem pradeep_passing_percentage (score failed_by max_marks : ℕ) :
  score = 185 → failed_by = 25 → max_marks = 600 →
  ((score + failed_by) / max_marks : ℚ) * 100 = 35 :=
by
  intros h_score h_failed_by h_max_marks
  sorry

end pradeep_passing_percentage_l215_215506


namespace biology_vs_reading_diff_l215_215178

def math_hw_pages : ℕ := 2
def reading_hw_pages : ℕ := 3
def total_hw_pages : ℕ := 15

def biology_hw_pages : ℕ := total_hw_pages - (math_hw_pages + reading_hw_pages)

theorem biology_vs_reading_diff : (biology_hw_pages - reading_hw_pages) = 7 := by
  sorry

end biology_vs_reading_diff_l215_215178


namespace find_f_1_l215_215210

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_1 : (∀ x : ℝ, f x + 3 * f (-x) = Real.logb 2 (x + 3)) → f 1 = 1 / 8 := 
by 
  sorry

end find_f_1_l215_215210


namespace renovation_days_l215_215541

/-
Conditions:
1. Cost to hire a company: 50000 rubles
2. Cost of buying materials: 20000 rubles
3. Husband's daily wage: 2000 rubles
4. Wife's daily wage: 1500 rubles
Question:
How many workdays can they spend on the renovation to make it more cost-effective?
-/

theorem renovation_days (cost_hire_company cost_materials : ℕ) 
  (husband_daily_wage wife_daily_wage : ℕ) 
  (more_cost_effective_days : ℕ) :
  cost_hire_company = 50000 → 
  cost_materials = 20000 → 
  husband_daily_wage = 2000 → 
  wife_daily_wage = 1500 → 
  more_cost_effective_days = 8 :=
by
  intros
  sorry

end renovation_days_l215_215541


namespace sum_of_a_b_l215_215294

variable {a b : ℝ}

theorem sum_of_a_b (h1 : a^2 = 4) (h2 : b^2 = 9) (h3 : a * b < 0) : a + b = 1 ∨ a + b = -1 := 
by 
  sorry

end sum_of_a_b_l215_215294


namespace shorter_leg_of_right_triangle_l215_215557

theorem shorter_leg_of_right_triangle (a b c : ℕ) (h: a^2 + b^2 = c^2) (hn: c = 65): a = 25 ∨ b = 25 :=
by
  sorry

end shorter_leg_of_right_triangle_l215_215557


namespace solve_system_l215_215692

theorem solve_system :
  ∃ (x y : ℝ), (2 * x - y = 1) ∧ (x + y = 2) ∧ (x = 1) ∧ (y = 1) :=
by
  sorry

end solve_system_l215_215692


namespace power_function_solution_l215_215010

def power_function_does_not_pass_through_origin (m : ℝ) : Prop :=
  (m^2 - m - 2) ≤ 0

def condition (m : ℝ) : Prop :=
  m^2 - 3 * m + 3 = 1

theorem power_function_solution (m : ℝ) :
  power_function_does_not_pass_through_origin m ∧ condition m → (m = 1 ∨ m = 2) :=
by sorry

end power_function_solution_l215_215010


namespace katya_female_classmates_l215_215891

theorem katya_female_classmates (g b : ℕ) (h1 : b = 2 * g) (h2 : b = g + 7) :
  g - 1 = 6 :=
by
  sorry

end katya_female_classmates_l215_215891


namespace fiona_first_to_toss_eight_l215_215804

theorem fiona_first_to_toss_eight :
  (∃ p : ℚ, p = 49/169 ∧
    (∀ n:ℕ, (7/8:ℚ)^(3*n) * (1/8) = if n = 0 then (49/512) else (49/512) * (343/512)^n)) :=
sorry

end fiona_first_to_toss_eight_l215_215804


namespace determine_s_value_l215_215504

def f (x : ℚ) : ℚ := abs (x - 1) - abs x

def u : ℚ := f (5 / 16)
def v : ℚ := f u
def s : ℚ := f v

theorem determine_s_value : s = 1 / 2 :=
by
  -- Proof needed here
  sorry

end determine_s_value_l215_215504


namespace focus_of_parabola_x2_eq_neg_4y_l215_215404

theorem focus_of_parabola_x2_eq_neg_4y :
  (∀ x y : ℝ, x^2 = -4 * y → focus = (0, -1)) := 
sorry

end focus_of_parabola_x2_eq_neg_4y_l215_215404
