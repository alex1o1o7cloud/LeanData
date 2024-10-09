import Mathlib

namespace constant_term_correct_l155_15519

theorem constant_term_correct:
    ∀ (a k n : ℤ), 
      (∀ x : ℤ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
      → a - n + k = 7 
      → n = -6 := 
by
    intros a k n h h2
    have h1 := h 0
    sorry

end constant_term_correct_l155_15519


namespace simplify_expression_l155_15536

theorem simplify_expression (x : ℝ) :
  4*x^3 + 5*x + 6*x^2 + 10 - (3 - 6*x^2 - 4*x^3 + 2*x) = 8*x^3 + 12*x^2 + 3*x + 7 :=
by
  sorry

end simplify_expression_l155_15536


namespace annual_income_before_tax_l155_15545

theorem annual_income_before_tax (I : ℝ) (h1 : 0.42 * I - 0.28 * I = 4830) : I = 34500 :=
sorry

end annual_income_before_tax_l155_15545


namespace ratio_of_efficacy_l155_15582

-- Define original conditions
def original_sprigs_of_mint := 3
def green_tea_leaves_per_sprig := 2

-- Define new condition
def new_green_tea_leaves := 12

-- Calculate the number of sprigs of mint corresponding to the new green tea leaves in the new mud
def new_sprigs_of_mint := new_green_tea_leaves / green_tea_leaves_per_sprig

-- Statement of the theorem: ratio of the efficacy of new mud to original mud is 1:2
theorem ratio_of_efficacy : new_sprigs_of_mint = 2 * original_sprigs_of_mint :=
by
    sorry

end ratio_of_efficacy_l155_15582


namespace inequality_one_solution_inequality_two_solution_cases_l155_15571

-- Setting up the problem for the first inequality
theorem inequality_one_solution :
  {x : ℝ | -1 ≤ x ∧ x ≤ 4} = {x : ℝ |  -x ^ 2 + 3 * x + 4 ≥ 0} :=
sorry

-- Setting up the problem for the second inequality with different cases of 'a'
theorem inequality_two_solution_cases (a : ℝ) :
  (a = 0 ∧ {x : ℝ | true} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a > 0 ∧ {x : ℝ | x ≥ a - 1 ∨ x ≤ -a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0})
  ∧ (a < 0 ∧ {x : ℝ | x ≥ -a - 1 ∨ x ≤ a - 1} = {x : ℝ | x ^ 2 + 2 * x + (1 - a) * (1 + a) ≥ 0}) :=
sorry

end inequality_one_solution_inequality_two_solution_cases_l155_15571


namespace john_weekly_loss_is_525000_l155_15500

-- Define the constants given in the problem
def daily_production : ℕ := 1000
def production_cost_per_tire : ℝ := 250
def selling_price_factor : ℝ := 1.5
def potential_daily_sales : ℕ := 1200
def days_in_week : ℕ := 7

-- Define the selling price per tire
def selling_price_per_tire : ℝ := production_cost_per_tire * selling_price_factor

-- Define John's current daily earnings from selling 1000 tires
def current_daily_earnings : ℝ := daily_production * selling_price_per_tire

-- Define John's potential daily earnings from selling 1200 tires
def potential_daily_earnings : ℝ := potential_daily_sales * selling_price_per_tire

-- Define the daily loss by not being able to produce all the tires
def daily_loss : ℝ := potential_daily_earnings - current_daily_earnings

-- Define the weekly loss
def weekly_loss : ℝ := daily_loss * days_in_week

-- Statement: Prove that John's weekly financial loss is $525,000
theorem john_weekly_loss_is_525000 : weekly_loss = 525000 :=
by
  sorry

end john_weekly_loss_is_525000_l155_15500


namespace product_of_g_on_roots_l155_15506

-- Define the given polynomials f and g
def f (x : ℝ) : ℝ := x^5 + 3 * x^2 + 1
def g (x : ℝ) : ℝ := x^2 - 5

-- Define the roots of the polynomial f
axiom roots : ∃ (x1 x2 x3 x4 x5 : ℝ), 
  f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0

theorem product_of_g_on_roots : 
  (∃ x1 x2 x3 x4 x5: ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0 ∧ f x5 = 0) 
  → g x1 * g x2 * g x3 * g x4 * g x5 = 131 := 
by
  sorry

end product_of_g_on_roots_l155_15506


namespace car_speeds_l155_15578

theorem car_speeds (d x : ℝ) (small_car_speed large_car_speed : ℝ) 
  (h1 : d = 135) 
  (h2 : small_car_speed = 5 * x) 
  (h3 : large_car_speed = 2 * x) 
  (h4 : 135 / small_car_speed + (4 + 0.5) = 135 / large_car_speed)
  : small_car_speed = 45 ∧ large_car_speed = 18 := by
  sorry

end car_speeds_l155_15578


namespace largest_common_value_lt_1000_l155_15596

theorem largest_common_value_lt_1000 :
  ∃ a : ℕ, ∃ n m : ℕ, a = 4 + 5 * n ∧ a = 7 + 11 * m ∧ a < 1000 ∧ 
  (∀ b : ℕ, ∀ p q : ℕ, b = 4 + 5 * p ∧ b = 7 + 11 * q ∧ b < 1000 → b ≤ a) :=
sorry

end largest_common_value_lt_1000_l155_15596


namespace range_of_a_l155_15518

variable {x a : ℝ}

def p (a : ℝ) (x : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2*x - 8 > 0

theorem range_of_a (ha : a < 0) 
  (H : (∀ x, ¬ p a x → q x) ∧ ∃ x, q x ∧ ¬ p a x ∧ ¬ q x) : a ≤ -4 := 
sorry

end range_of_a_l155_15518


namespace mean_height_calc_l155_15508

/-- Heights of players on the soccer team -/
def heights : List ℕ := [47, 48, 50, 50, 54, 55, 57, 59, 63, 63, 64, 65]

/-- Total number of players -/
def total_players : ℕ := heights.length

/-- Sum of heights of players -/
def sum_heights : ℕ := heights.sum

/-- Mean height of players on the soccer team -/
def mean_height : ℚ := sum_heights / total_players

/-- Proof that the mean height is correct -/
theorem mean_height_calc : mean_height = 56.25 := by
  sorry

end mean_height_calc_l155_15508


namespace overall_average_of_25_results_l155_15514

theorem overall_average_of_25_results (first_12_avg last_12_avg thirteenth_result : ℝ) 
  (h1 : first_12_avg = 14) (h2 : last_12_avg = 17) (h3 : thirteenth_result = 78) :
  (12 * first_12_avg + thirteenth_result + 12 * last_12_avg) / 25 = 18 :=
by
  sorry

end overall_average_of_25_results_l155_15514


namespace postal_service_revenue_l155_15531

theorem postal_service_revenue 
  (price_colored : ℝ := 0.50)
  (price_bw : ℝ := 0.35)
  (price_golden : ℝ := 2.00)
  (sold_colored : ℕ := 578833)
  (sold_bw : ℕ := 523776)
  (sold_golden : ℕ := 120456) : 
  (price_colored * (sold_colored : ℝ) + 
  price_bw * (sold_bw : ℝ) + 
  price_golden * (sold_golden : ℝ) = 713650.10) :=
by
  sorry

end postal_service_revenue_l155_15531


namespace first_divisor_is_six_l155_15590

theorem first_divisor_is_six {d : ℕ} 
  (h1: (1394 - 14) % d = 0)
  (h2: (2535 - 1929) % d = 0)
  (h3: (40 - 34) % d = 0)
  : d = 6 :=
sorry

end first_divisor_is_six_l155_15590


namespace intersection_point_of_lines_l155_15553

theorem intersection_point_of_lines :
  let line1 (x : ℝ) := 3 * x - 4
  let line2 (x : ℝ) := - (1 / 3) * x + 5
  (∃ x y : ℝ, line1 x = y ∧ line2 x = y ∧ x = 2.7 ∧ y = 4.1) :=
by {
    sorry
}

end intersection_point_of_lines_l155_15553


namespace milk_exchange_l155_15595

theorem milk_exchange (initial_empty_bottles : ℕ) (exchange_rate : ℕ) (start_full_bottles : ℕ) : initial_empty_bottles = 43 → exchange_rate = 4 → start_full_bottles = 0 → ∃ liters_of_milk : ℕ, liters_of_milk = 14 :=
by
  intro h1 h2 h3
  sorry

end milk_exchange_l155_15595


namespace digital_earth_sustainable_development_l155_15520

theorem digital_earth_sustainable_development :
  (after_realization_digital_earth : Prop) → (scientists_can : Prop) :=
sorry

end digital_earth_sustainable_development_l155_15520


namespace discount_percentage_l155_15592

variable (P : ℝ)  -- Original price of the car
variable (D : ℝ)  -- Discount percentage in decimal form
variable (S : ℝ)  -- Selling price of the car

theorem discount_percentage
  (h1 : S = P * (1 - D) * 1.70)
  (h2 : S = P * 1.1899999999999999) :
  D = 0.3 :=
by
  -- The proof goes here
  sorry

end discount_percentage_l155_15592


namespace adjusted_retail_price_l155_15560

variable {a : ℝ} {m n : ℝ}

theorem adjusted_retail_price (h : 0 ≤ m ∧ 0 ≤ n) : (a * (1 + m / 100) * (n / 100)) = a * (1 + m / 100) * (n / 100) :=
by
  sorry

end adjusted_retail_price_l155_15560


namespace fraction_subtraction_l155_15589

theorem fraction_subtraction :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end fraction_subtraction_l155_15589


namespace Henry_age_l155_15541

-- Define the main proof statement
theorem Henry_age (h s : ℕ) 
(h1 : h + 8 = 3 * (s - 1))
(h2 : (h - 25) + (s - 25) = 83) : h = 97 :=
by
  sorry

end Henry_age_l155_15541


namespace difference_between_sums_l155_15539

open Nat

-- Sum of the first 'n' positive odd integers formula: n^2
def sum_of_first_odd (n : ℕ) : ℕ := n * n

-- Sum of the first 'n' positive even integers formula: n(n+1)
def sum_of_first_even (n : ℕ) : ℕ := n * (n + 1)

-- The main theorem stating the difference between the sums
theorem difference_between_sums (n : ℕ) (h : n = 3005) :
  sum_of_first_even n - sum_of_first_odd n = 3005 :=
by
  sorry

end difference_between_sums_l155_15539


namespace mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l155_15547

def molar_mass_carbon : Float := 12.01
def molar_mass_hydrogen : Float := 1.008
def moles_of_nonane : Float := 23.0
def num_carbons_in_nonane : Float := 9.0
def num_hydrogens_in_nonane : Float := 20.0

theorem mass_of_23_moles_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  mass_23_moles = 2950.75 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let mass_23_moles := moles_of_nonane * molar_mass_C9H20
  have molar_mass_C9H20_val : molar_mass_C9H20 = 128.25 := sorry
  have mass_23_moles_val : mass_23_moles = 2950.75 := sorry
  exact mass_23_moles_val

theorem percentage_composition_C_H_O_in_C9H20 :
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  percentage_carbon = 84.27 ∧ percentage_hydrogen = 15.73 ∧ percentage_oxygen = 0 :=
by
  let molar_mass_C9H20 := (num_carbons_in_nonane * molar_mass_carbon) + (num_hydrogens_in_nonane * molar_mass_hydrogen)
  let percentage_carbon := (num_carbons_in_nonane * molar_mass_carbon / molar_mass_C9H20) * 100
  let percentage_hydrogen := (num_hydrogens_in_nonane * molar_mass_hydrogen / molar_mass_C9H20) * 100
  let percentage_oxygen := 0
  have percentage_carbon_val : percentage_carbon = 84.27 := sorry
  have percentage_hydrogen_val : percentage_hydrogen = 15.73 := sorry
  have percentage_oxygen_val : percentage_oxygen = 0 := by rfl
  exact ⟨percentage_carbon_val, percentage_hydrogen_val, percentage_oxygen_val⟩

end mass_of_23_moles_C9H20_percentage_composition_C_H_O_in_C9H20_l155_15547


namespace exist_odd_a_b_k_l155_15525

theorem exist_odd_a_b_k (m : ℤ) : 
  ∃ (a b k : ℤ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (k ≥ 0) ∧ (2 * m = a^19 + b^99 + k * 2^1999) :=
by {
  sorry
}

end exist_odd_a_b_k_l155_15525


namespace exists_unique_xy_l155_15532

theorem exists_unique_xy (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end exists_unique_xy_l155_15532


namespace regular_polygon_sides_l155_15546

theorem regular_polygon_sides (n : ℕ) (h : (180 * (n - 2) = 135 * n)) : n = 8 := by
  sorry

end regular_polygon_sides_l155_15546


namespace arithmetic_expression_l155_15512

theorem arithmetic_expression : 125 - 25 * 4 = 25 := 
by
  sorry

end arithmetic_expression_l155_15512


namespace total_interest_l155_15599

def P : ℝ := 1000
def r : ℝ := 0.1
def n : ℕ := 3

theorem total_interest : (P * (1 + r)^n) - P = 331 := by
  sorry

end total_interest_l155_15599


namespace cubic_difference_l155_15559

theorem cubic_difference (a b : ℝ) 
  (h₁ : a - b = 7)
  (h₂ : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := 
by 
  sorry

end cubic_difference_l155_15559


namespace range_of_x_for_a_range_of_a_l155_15585

-- Define propositions p and q
def prop_p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def prop_q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- Part (I)
theorem range_of_x_for_a (a x : ℝ) (ha : a = 1) (hpq : prop_p a x ∧ prop_q x) : 2 < x ∧ x < 3 :=
by
  sorry

-- Part (II)
theorem range_of_a (p q : ℝ → Prop) (hpq : ∀ x : ℝ, ¬p x → ¬q x) :
  1 < a ∧ a ≤ 2 :=
by
  sorry

end range_of_x_for_a_range_of_a_l155_15585


namespace functional_equation_solution_l155_15538

-- The mathematical problem statement in Lean 4

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_monotonic : ∀ x y : ℝ, (f x) * (f y) = f (x + y))
  (h_mono : ∀ x y : ℝ, x < y → f x < f y ∨ f x > f y) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end functional_equation_solution_l155_15538


namespace sally_cost_is_42000_l155_15503

-- Definitions for conditions
def lightningCost : ℕ := 140000
def materCost : ℕ := (10 * lightningCost) / 100
def sallyCost : ℕ := 3 * materCost

-- Theorem statement
theorem sally_cost_is_42000 : sallyCost = 42000 := by
  sorry

end sally_cost_is_42000_l155_15503


namespace find_red_peaches_l155_15593

def num_red_peaches (red yellow green : ℕ) : Prop :=
  (green = red + 1) ∧ yellow = 71 ∧ green = 8

theorem find_red_peaches (red : ℕ) :
  num_red_peaches red 71 8 → red = 7 :=
by
  sorry

end find_red_peaches_l155_15593


namespace relationship_between_x_y_l155_15561

theorem relationship_between_x_y (x y : ℝ) (h1 : x^2 - y^2 > 2 * x) (h2 : x * y < y) : x < y ∧ y < 0 := 
sorry

end relationship_between_x_y_l155_15561


namespace necessary_but_not_sufficient_l155_15524

def p (x : ℝ) : Prop := x < 3
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

theorem necessary_but_not_sufficient : 
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) :=
by sorry

end necessary_but_not_sufficient_l155_15524


namespace sin_minus_cos_eq_l155_15579

variable {α : ℝ} (h₁ : 0 < α ∧ α < π) (h₂ : Real.sin α + Real.cos α = 1/3)

theorem sin_minus_cos_eq : Real.sin α - Real.cos α = Real.sqrt 17 / 3 :=
by 
  -- Proof goes here
  sorry

end sin_minus_cos_eq_l155_15579


namespace Susan_initial_amount_l155_15581

def initial_amount (S : ℝ) : Prop :=
  let Spent_in_September := (1/6) * S
  let Spent_in_October := (1/8) * S
  let Spent_in_November := 0.3 * S
  let Spent_in_December := 100
  let Remaining := 480
  S - (Spent_in_September + Spent_in_October + Spent_in_November + Spent_in_December) = Remaining

theorem Susan_initial_amount : ∃ S : ℝ, initial_amount S ∧ S = 1420 :=
by
  sorry

end Susan_initial_amount_l155_15581


namespace ratio_bee_eaters_leopards_l155_15598

variables (s f l c a t e r : ℕ)

-- Define the conditions from the problem.
def conditions : Prop :=
  s = 100 ∧
  f = 80 ∧
  l = 20 ∧
  c = s / 2 ∧
  a = 2 * (f + l) ∧
  t = 670 ∧
  e = t - (s + f + l + c + a)

-- The theorem statement proving the ratio.
theorem ratio_bee_eaters_leopards (h : conditions s f l c a t e) : r = (e / l) := by
  sorry

end ratio_bee_eaters_leopards_l155_15598


namespace sum_of_products_two_at_a_time_l155_15586

theorem sum_of_products_two_at_a_time
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 222)
  (h2 : a + b + c = 22) :
  a * b + b * c + c * a = 131 := 
sorry

end sum_of_products_two_at_a_time_l155_15586


namespace no_unique_symbols_for_all_trains_l155_15530

def proposition (a b c d : Prop) : Prop :=
  (¬a ∧  b ∧ ¬c ∧  d)
∨ ( a ∧ ¬b ∧ ¬c ∧ ¬d)

theorem no_unique_symbols_for_all_trains 
    (a b c d : Prop)
    (p : proposition a b c d)
    (s1 : ¬a ∧  b ∧ ¬c ∧  d)
    (s2 :  a ∧ ¬b ∧ ¬c ∧ ¬d) : 
    False :=
by {cases s1; cases s2; contradiction}

end no_unique_symbols_for_all_trains_l155_15530


namespace ratio_of_cows_sold_l155_15569

-- Condition 1: The farmer originally has 51 cows.
def original_cows : ℕ := 51

-- Condition 2: The farmer adds 5 new cows to the herd.
def new_cows : ℕ := 5

-- Condition 3: The farmer has 42 cows left after selling a portion of the herd.
def remaining_cows : ℕ := 42

-- Defining total cows after adding new cows
def total_cows_after_addition : ℕ := original_cows + new_cows

-- Defining cows sold
def cows_sold : ℕ := total_cows_after_addition - remaining_cows

-- The theorem states the ratio of 'cows sold' to 'total cows after addition' is 1 : 4
theorem ratio_of_cows_sold : (cows_sold : ℚ) / (total_cows_after_addition : ℚ) = 1 / 4 := by
  -- Proof would go here
  sorry


end ratio_of_cows_sold_l155_15569


namespace cars_minus_trucks_l155_15505

theorem cars_minus_trucks (total : ℕ) (trucks : ℕ) (h_total : total = 69) (h_trucks : trucks = 21) :
  (total - trucks) - trucks = 27 :=
by
  sorry

end cars_minus_trucks_l155_15505


namespace quadratic_passes_through_point_l155_15554

theorem quadratic_passes_through_point (a b : ℝ) (h : a ≠ 0) (h₁ : ∃ y : ℝ, y = a * 1^2 + b * 1 - 1 ∧ y = 1) : a + b + 1 = 3 :=
by
  obtain ⟨y, hy1, hy2⟩ := h₁
  sorry

end quadratic_passes_through_point_l155_15554


namespace evaluated_result_l155_15556

noncomputable def evaluate_expression (y : ℝ) (hy : y ≠ 0) : ℝ :=
  (18 * y^3) * (4 * y^2) * (1 / (2 * y)^3)

theorem evaluated_result (y : ℝ) (hy : y ≠ 0) : evaluate_expression y hy = 9 * y^2 :=
by
  sorry

end evaluated_result_l155_15556


namespace initial_guppies_l155_15529

theorem initial_guppies (total_gups : ℕ) (dozen_gups : ℕ) (extra_gups : ℕ) (baby_gups_initial : ℕ) (baby_gups_later : ℕ) :
  total_gups = 52 → dozen_gups = 12 → extra_gups = 3 → baby_gups_initial = 3 * 12 → baby_gups_later = 9 → 
  total_gups - (baby_gups_initial + baby_gups_later) = 7 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end initial_guppies_l155_15529


namespace four_digit_integer_product_l155_15502

theorem four_digit_integer_product :
  ∃ (a b c d : ℕ), a < b ∧ b < c ∧ c < d ∧ 
  a^2 + b^2 + c^2 + d^2 = 65 ∧ a * b * c * d = 140 :=
by
  sorry

end four_digit_integer_product_l155_15502


namespace dilation_transformation_result_l155_15584

theorem dilation_transformation_result
  (x y x' y' : ℝ)
  (h₀ : x'^2 / 4 + y'^2 / 9 = 1) 
  (h₁ : x' = 2 * x)
  (h₂ : y' = 3 * y)
  (h₃ : x^2 + y^2 = 1)
  : x'^2 / 4 + y'^2 / 9 = 1 := 
by
  sorry

end dilation_transformation_result_l155_15584


namespace percentage_in_excess_l155_15567

theorem percentage_in_excess 
  (A B : ℝ) (x : ℝ)
  (h1 : ∀ A',  A' = A * (1 + x / 100))
  (h2 : ∀ B',  B' = 0.94 * B)
  (h3 : ∀ A' B', A' * B' = A * B * (1 + 0.0058)) :
  x = 7 :=
by
  sorry

end percentage_in_excess_l155_15567


namespace range_of_a_l155_15501

variable (a x : ℝ)
def A (a : ℝ) := {x : ℝ | 2 * a ≤ x ∧ x ≤ a ^ 2 + 1}
def B (a : ℝ) := {x : ℝ | (x - 2) * (x - (3 * a + 1)) ≤ 0}

theorem range_of_a (a : ℝ) : (∀ x, x ∈ A a → x ∈ B a) ↔ (1 ≤ a ∧ a ≤ 3) ∨ (a = -1) := by sorry

end range_of_a_l155_15501


namespace exists_2016_integers_with_product_9_and_sum_0_l155_15591

theorem exists_2016_integers_with_product_9_and_sum_0 :
  ∃ (L : List ℤ), L.length = 2016 ∧ L.prod = 9 ∧ L.sum = 0 := by
  sorry

end exists_2016_integers_with_product_9_and_sum_0_l155_15591


namespace closest_integer_to_cbrt_250_l155_15543

theorem closest_integer_to_cbrt_250 (a b : ℤ)
  (h₁ : a = 6) (h₂ : b = 7)
  (h₃ : a^3 = 216) (h₄ : b^3 = 343) :
  abs ((6 : ℤ)^3 - 250) < abs ((7 : ℤ)^3 - 250) :=
by
  sorry

end closest_integer_to_cbrt_250_l155_15543


namespace picture_area_l155_15510

theorem picture_area (x y : ℕ) (hx : 1 < x) (hy : 1 < y) 
  (h_area : (3 * x + 4) * (y + 3) = 60) : x * y = 15 := 
by 
  sorry

end picture_area_l155_15510


namespace negative_expression_P_minus_Q_l155_15551

theorem negative_expression_P_minus_Q :
  ∀ (P Q R S T : ℝ), 
    P = -4.0 → 
    Q = -2.0 → 
    R = 0.2 → 
    S = 1.1 → 
    T = 1.7 → 
    P - Q < 0 := 
by 
  intros P Q R S T hP hQ hR hS hT
  rw [hP, hQ]
  sorry

end negative_expression_P_minus_Q_l155_15551


namespace number_of_ideal_subsets_l155_15537

def is_ideal_subset (p q : ℕ) (S : Set ℕ) : Prop :=
  0 ∈ S ∧ ∀ n ∈ S, n + p ∈ S ∧ n + q ∈ S

theorem number_of_ideal_subsets (p q : ℕ) (hpq : Nat.Coprime p q) :
  ∃ n, n = Nat.choose (p + q) p / (p + q) :=
sorry

end number_of_ideal_subsets_l155_15537


namespace max_chords_through_line_l155_15574

noncomputable def maxChords (n : ℕ) : ℕ :=
  let k := n / 2
  k * k + n

theorem max_chords_through_line (points : ℕ) (h : points = 2017) : maxChords 2016 = 1018080 :=
by
  have h1 : (2016 / 2) * (2016 / 2) + 2016 = 1018080 := by norm_num
  rw [← h1]; sorry

end max_chords_through_line_l155_15574


namespace sugar_percentage_l155_15587

theorem sugar_percentage 
  (initial_volume : ℝ) (initial_water_perc : ℝ) (initial_kola_perc: ℝ) (added_sugar : ℝ) (added_water : ℝ) (added_kola : ℝ)
  (initial_solution: initial_volume = 340) 
  (perc_water : initial_water_perc = 0.75) 
  (perc_kola: initial_kola_perc = 0.05)
  (added_sugar_amt : added_sugar = 3.2) 
  (added_water_amt : added_water = 12) 
  (added_kola_amt : added_kola = 6.8) : 
  (71.2 / 362) * 100 = 19.67 := 
by 
  sorry

end sugar_percentage_l155_15587


namespace parabola_focus_equals_hyperbola_focus_l155_15568

noncomputable def hyperbola_right_focus : (Float × Float) := (2, 0)

noncomputable def parabola_focus (p : Float) : (Float × Float) := (p / 2, 0)

theorem parabola_focus_equals_hyperbola_focus (p : Float) :
  parabola_focus p = hyperbola_right_focus → p = 4 := by
  intro h
  sorry

end parabola_focus_equals_hyperbola_focus_l155_15568


namespace exponent_multiplication_l155_15580

theorem exponent_multiplication :
  (-1 / 2 : ℝ) ^ 2022 * (2 : ℝ) ^ 2023 = 2 :=
by sorry

end exponent_multiplication_l155_15580


namespace sum_of_arithmetic_sequence_has_remainder_2_l155_15504

def arithmetic_sequence_remainder : ℕ := 
  let first_term := 1
  let common_difference := 6
  let last_term := 259
  -- Calculate number of terms
  let n := (last_term + 5) / common_difference
  -- Sum of remainders of each term when divided by 6
  let sum_of_remainders := n * 1
  -- The remainder when this sum is divided by 6
  sum_of_remainders % 6 
theorem sum_of_arithmetic_sequence_has_remainder_2 : 
  arithmetic_sequence_remainder = 2 := by 
  sorry

end sum_of_arithmetic_sequence_has_remainder_2_l155_15504


namespace mean_is_six_greater_than_median_l155_15565

theorem mean_is_six_greater_than_median (x a : ℕ) 
  (h1 : (x + a) + (x + 4) + (x + 7) + (x + 37) + x == 5 * (x + 10)) :
  a = 2 :=
by
  -- proof goes here
  sorry

end mean_is_six_greater_than_median_l155_15565


namespace inequality_Cauchy_Schwarz_l155_15523

theorem inequality_Cauchy_Schwarz (a b : ℝ) : 
  (a^4 + b^4) * (a^2 + b^2) ≥ (a^3 + b^3)^2 :=
by
  sorry

end inequality_Cauchy_Schwarz_l155_15523


namespace cheryl_material_used_l155_15550

noncomputable def total_material_needed : ℚ :=
  (5 / 11) + (2 / 3)

noncomputable def material_left : ℚ :=
  25 / 55

noncomputable def material_used : ℚ :=
  total_material_needed - material_left

theorem cheryl_material_used :
  material_used = 22 / 33 :=
by
  sorry

end cheryl_material_used_l155_15550


namespace cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l155_15542

-- Problem Part 1
theorem cos_alpha_implies_sin_alpha (alpha : ℝ) (h1 : Real.cos alpha = -4/5) (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin alpha = -3/5 := sorry

-- Problem Part 2
theorem tan_theta_implies_expr (theta : ℝ) (h1 : Real.tan theta = 3) : 
  (Real.sin theta + Real.cos theta) / (2 * Real.sin theta + Real.cos theta) = 4 / 7 := sorry

end cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l155_15542


namespace tire_circumference_constant_l155_15534

/--
Given the following conditions:
1. Car speed v = 120 km/h
2. Tire rotation rate n = 400 rpm
3. Tire pressure P = 32 psi
4. Tire radius changes according to the formula R = R_0(1 + kP)
5. R_0 is the initial tire radius
6. k is a constant relating to the tire's elasticity
7. Change in tire pressure due to the incline is negligible

Prove that the circumference C of the tire is 5 meters.
-/
theorem tire_circumference_constant (v : ℝ) (n : ℝ) (P : ℝ) (R_0 : ℝ) (k : ℝ) 
  (h1 : v = 120 * 1000 / 3600) -- Car speed in m/s
  (h2 : n = 400 / 60)           -- Tire rotation rate in rps
  (h3 : P = 32)                 -- Tire pressure in psi
  (h4 : ∀ R P, R = R_0 * (1 + k * P)) -- Tire radius formula
  (h5 : ∀ P, P = 0)             -- Negligible change in tire pressure
  : C = 5 :=
  sorry

end tire_circumference_constant_l155_15534


namespace NicoleEndsUpWith36Pieces_l155_15507

namespace ClothingProblem

noncomputable def NicoleClothesStart := 10
noncomputable def FirstOlderSisterClothes := NicoleClothesStart / 2
noncomputable def NextOldestSisterClothes := NicoleClothesStart + 2
noncomputable def OldestSisterClothes := (NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes) / 3

theorem NicoleEndsUpWith36Pieces : 
  NicoleClothesStart + FirstOlderSisterClothes + NextOldestSisterClothes + OldestSisterClothes = 36 :=
  by
    sorry

end ClothingProblem

end NicoleEndsUpWith36Pieces_l155_15507


namespace john_remaining_amount_l155_15583

theorem john_remaining_amount (initial_amount games: ℕ) (food souvenirs: ℕ) :
  initial_amount = 100 →
  games = 20 →
  food = 3 * games →
  souvenirs = (1 / 2 : ℚ) * games →
  initial_amount - (games + food + souvenirs) = 10 :=
by
  sorry

end john_remaining_amount_l155_15583


namespace sandwiches_per_person_l155_15511

-- Definitions derived from conditions
def cost_of_12_croissants := 8.0
def number_of_people := 24
def total_spending := 32.0
def croissants_per_set := 12

-- Statement to be proved
theorem sandwiches_per_person :
  ∀ (cost_of_12_croissants total_spending croissants_per_set number_of_people : ℕ),
  total_spending / cost_of_12_croissants * croissants_per_set / number_of_people = 2 :=
by
  sorry

end sandwiches_per_person_l155_15511


namespace problem_I_II_l155_15509

noncomputable def f (x : ℝ) : ℝ := x - Real.sin x

def seq_a (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 0 = a1 ∧ (∀ n, a (n + 1) = f (a n))

theorem problem_I_II (a : ℕ → ℝ) (a1 : ℝ) (h_a1 : 0 < a1 ∧ a1 < 1) (h_seq : seq_a a a1) :
  (∀ n, 0 < a (n + 1) ∧ a (n + 1) < a n ∧ a n < 1) ∧
  (∀ n, a (n + 1) < (1 / 6) * (a n) ^ 3) :=
  sorry

end problem_I_II_l155_15509


namespace intersection_complement_l155_15515

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3, 5}

theorem intersection_complement : A ∩ (U \ B) = {0, 1} := by
  sorry

end intersection_complement_l155_15515


namespace distance_to_grandmas_house_is_78_l155_15535

-- Define the conditions
def miles_to_pie_shop : ℕ := 35
def miles_to_gas_station : ℕ := 18
def miles_remaining : ℕ := 25

-- Define the mathematical claim
def total_distance_to_grandmas_house : ℕ :=
  miles_to_pie_shop + miles_to_gas_station + miles_remaining

-- Prove the claim
theorem distance_to_grandmas_house_is_78 :
  total_distance_to_grandmas_house = 78 :=
by
  sorry

end distance_to_grandmas_house_is_78_l155_15535


namespace inverse_of_B_squared_l155_15566

theorem inverse_of_B_squared (B_inv : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B_inv = ![![3, -2], ![0, 5]]) : 
  (B_inv * B_inv) = ![![9, -16], ![0, 25]] :=
by
  sorry

end inverse_of_B_squared_l155_15566


namespace percent_children_with_both_colors_l155_15548

theorem percent_children_with_both_colors
  (F : ℕ) (C : ℕ) 
  (even_F : F % 2 = 0)
  (children_pick_two_flags : C = F / 2)
  (sixty_percent_blue : 6 * C / 10 = 6 * C / 10)
  (fifty_percent_red : 5 * C / 10 = 5 * C / 10)
  : (6 * C / 10) + (5 * C / 10) - C = C / 10 :=
by
  sorry

end percent_children_with_both_colors_l155_15548


namespace min_a2_plus_b2_l155_15564

theorem min_a2_plus_b2 (a b : ℝ) (h : a * b = 1) : a^2 + b^2 ≥ 2 :=
sorry

end min_a2_plus_b2_l155_15564


namespace tan_sub_pi_over_4_l155_15570

-- Define the conditions and the problem statement
variable (α : ℝ) (h : Real.tan α = 2)

-- State the problem as a theorem
theorem tan_sub_pi_over_4 : Real.tan (α - Real.pi / 4) = 1 / 3 :=
by
  sorry

end tan_sub_pi_over_4_l155_15570


namespace find_pink_highlighters_l155_15552

def yellow_highlighters : ℕ := 7
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 15

theorem find_pink_highlighters : (total_highlighters - (yellow_highlighters + blue_highlighters)) = 3 :=
by
  sorry

end find_pink_highlighters_l155_15552


namespace find_11th_place_l155_15573

def placement_problem (Amara Bindu Carlos Devi Eshan Farel: ℕ): Prop :=
  (Carlos + 5 = Amara) ∧
  (Bindu = Eshan + 3) ∧
  (Carlos = Devi + 2) ∧
  (Devi = 6) ∧
  (Eshan + 1 = Farel) ∧
  (Bindu + 4 = Amara) ∧
  (Farel = 9)

theorem find_11th_place (Amara Bindu Carlos Devi Eshan Farel: ℕ) 
  (h : placement_problem Amara Bindu Carlos Devi Eshan Farel) : 
  Eshan = 11 := 
sorry

end find_11th_place_l155_15573


namespace find_fraction_value_l155_15575

variable (a b : ℝ)

theorem find_fraction_value (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : (4 * a + b) / (a - 4 * b) = 3) :
  (a + 4 * b) / (4 * a - b) = 9 / 53 := 
  sorry

end find_fraction_value_l155_15575


namespace union_of_A_and_B_l155_15555

-- Define set A
def A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

-- Define set B
def B := {x : ℝ | x < 1}

-- The proof problem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≤ 2} :=
by sorry

end union_of_A_and_B_l155_15555


namespace solve_abs_quadratic_eq_l155_15533

theorem solve_abs_quadratic_eq (x : ℝ) (h : |2 * x + 4| = 1 - 3 * x + x ^ 2) :
    x = (5 + Real.sqrt 37) / 2 ∨ x = (5 - Real.sqrt 37) / 2 := by
  sorry

end solve_abs_quadratic_eq_l155_15533


namespace product_of_total_points_l155_15572

def f (n : ℕ) : ℕ :=
  if n % 3 = 0 then 5
  else if n % 2 = 0 then 3
  else 0

def Allie_rolls : List ℕ := [3, 5, 6, 2, 4]
def Betty_rolls : List ℕ := [3, 2, 1, 6, 4]

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

theorem product_of_total_points :
  total_points Allie_rolls * total_points Betty_rolls = 256 :=
by
  sorry

end product_of_total_points_l155_15572


namespace total_sacks_after_6_days_l155_15528

-- Define the conditions
def sacks_per_day : ℕ := 83
def days : ℕ := 6

-- Prove the total number of sacks after 6 days is 498
theorem total_sacks_after_6_days : sacks_per_day * days = 498 := by
  -- Proof Content Placeholder
  sorry

end total_sacks_after_6_days_l155_15528


namespace rolls_for_mode_of_two_l155_15597

theorem rolls_for_mode_of_two (n : ℕ) (p : ℚ := 1/6) (m0 : ℕ := 32) : 
  (n : ℚ) * p - (1 - p) ≤ m0 ∧ m0 ≤ (n : ℚ) * p + p ↔ 191 ≤ n ∧ n ≤ 197 := 
by
  sorry

end rolls_for_mode_of_two_l155_15597


namespace fraction_addition_l155_15527

variable (a : ℝ)

theorem fraction_addition (ha : a ≠ 0) : (3 / a) + (2 / a) = (5 / a) :=
by
  sorry

end fraction_addition_l155_15527


namespace maria_should_buy_more_l155_15557

-- Define the conditions as assumptions.
variables (needs total_cartons : ℕ) (strawberries blueberries : ℕ)

-- Specify the given conditions.
def maria_conditions (needs total_cartons strawberries blueberries : ℕ) : Prop :=
  needs = 21 ∧ strawberries = 4 ∧ blueberries = 8 ∧ total_cartons = strawberries + blueberries

-- State the theorem to be proven.
theorem maria_should_buy_more
  (needs total_cartons : ℕ) (strawberries blueberries : ℕ)
  (h : maria_conditions needs total_cartons strawberries blueberries) :
  needs - total_cartons = 9 :=
sorry

end maria_should_buy_more_l155_15557


namespace A_completion_time_l155_15526

theorem A_completion_time :
  ∃ A : ℝ, (A > 0) ∧ (
    (2 * (1 / A + 1 / 10) + 3.0000000000000004 * (1 / 10) = 1) ↔ A = 4
  ) :=
by
  have B_workday := 10
  sorry -- proof would go here

end A_completion_time_l155_15526


namespace carrot_price_l155_15588

variables (total_tomatoes : ℕ) (total_carrots : ℕ) (price_per_tomato : ℝ) (total_revenue : ℝ)

theorem carrot_price :
  total_tomatoes = 200 →
  total_carrots = 350 →
  price_per_tomato = 1 →
  total_revenue = 725 →
  (total_revenue - total_tomatoes * price_per_tomato) / total_carrots = 1.5 :=
by
  intros h1 h2 h3 h4
  sorry

end carrot_price_l155_15588


namespace percentage_failed_hindi_l155_15562

theorem percentage_failed_hindi 
  (F_E F_B P_BE : ℕ) 
  (h₁ : F_E = 42) 
  (h₂ : F_B = 28) 
  (h₃ : P_BE = 56) :
  ∃ F_H, F_H = 30 := 
by
  sorry

end percentage_failed_hindi_l155_15562


namespace sufficient_condition_l155_15558

theorem sufficient_condition (a b c : ℤ) (h1 : a = c) (h2 : b - 1 = a) : 
  a * (a - b) + b * (b - c) + c * (c - a) = 2 :=
by
  sorry

end sufficient_condition_l155_15558


namespace right_triangle_acute_angles_l155_15563

theorem right_triangle_acute_angles (a b : ℝ)
  (h_right_triangle : a + b = 90)
  (h_ratio : a / b = 3 / 2) :
  (a = 54) ∧ (b = 36) :=
by
  sorry

end right_triangle_acute_angles_l155_15563


namespace binomial_sum_eval_l155_15594

theorem binomial_sum_eval :
  (Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 5)) +
  (Nat.factorial 6 / (Nat.factorial 4 * Nat.factorial 2)) = 36 := by
sorry

end binomial_sum_eval_l155_15594


namespace chord_square_length_eq_512_l155_15540

open Real

/-
The conditions are:
1. The radii of two smaller circles are 4 and 8.
2. These circles are externally tangent to each other.
3. Both smaller circles are internally tangent to a larger circle with radius 12.
4. A common external tangent to the two smaller circles serves as a chord of the larger circle.
-/

noncomputable def radius_small1 : ℝ := 4
noncomputable def radius_small2 : ℝ := 8
noncomputable def radius_large : ℝ := 12

/-- Show that the square of the length of the chord formed by the common external tangent of two smaller circles 
which are externally tangent to each other and internally tangent to a larger circle is 512. -/
theorem chord_square_length_eq_512 : ∃ (PQ : ℝ), PQ^2 = 512 := by
  sorry

end chord_square_length_eq_512_l155_15540


namespace volume_increase_l155_15577

theorem volume_increase (l w h: ℕ) 
(h1: l * w * h = 4320) 
(h2: l * w + w * h + h * l = 852) 
(h3: l + w + h = 52) : 
(l + 1) * (w + 1) * (h + 1) = 5225 := 
by 
  sorry

end volume_increase_l155_15577


namespace dwarf_diamond_distribution_l155_15549

-- Definitions for conditions
def dwarves : Type := Fin 8
structure State :=
  (diamonds : dwarves → ℕ)

-- Initial condition: Each dwarf has 3 diamonds
def initial_state : State := 
  { diamonds := fun _ => 3 }

-- Transition function: Each dwarf divides diamonds into two piles and passes them to neighbors
noncomputable def transition (s : State) : State := sorry

-- Proof goal: At a certain point in time, 3 specific dwarves have 24 diamonds in total,
-- with one dwarf having 7 diamonds, then prove the other two dwarves have 12 and 5 diamonds.
theorem dwarf_diamond_distribution (s : State)
  (h1 : ∃ t, s = (transition^[t]) initial_state ∧ ∃ i j k : dwarves, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧
    s.diamonds i + s.diamonds j + s.diamonds k = 24 ∧
    s.diamonds i = 7)
  : ∃ a b : dwarves, a ≠ b ∧ s.diamonds a = 12 ∧ s.diamonds b = 5 := sorry

end dwarf_diamond_distribution_l155_15549


namespace solve_inequality_l155_15522

open Set

theorem solve_inequality :
  { x : ℝ | (2 * x - 2) / (x^2 - 5*x + 6) ≤ 3 } = Ioo (5/3) 2 ∪ Icc 3 4 :=
by
  sorry

end solve_inequality_l155_15522


namespace geom_seq_a7_a10_sum_l155_15516

theorem geom_seq_a7_a10_sum (a_n : ℕ → ℝ) (q a1 : ℝ)
  (h_seq : ∀ n, a_n (n + 1) = a1 * (q ^ n))
  (h1 : a1 + a1 * q = 2)
  (h2 : a1 * (q ^ 2) + a1 * (q ^ 3) = 4) :
  a_n 7 + a_n 8 + a_n 9 + a_n 10 = 48 := 
sorry

end geom_seq_a7_a10_sum_l155_15516


namespace least_possible_number_l155_15517

theorem least_possible_number {x : ℕ} (h1 : x % 6 = 2) (h2 : x % 4 = 3) : x = 50 :=
sorry

end least_possible_number_l155_15517


namespace original_number_is_fraction_l155_15544

theorem original_number_is_fraction (x : ℚ) (h : 1 + 1/x = 7/3) : x = 3/4 :=
sorry

end original_number_is_fraction_l155_15544


namespace knights_round_table_l155_15521

theorem knights_round_table (n : ℕ) (h : ∃ (f e : ℕ), f = e ∧ f + e = n) : n % 4 = 0 :=
sorry

end knights_round_table_l155_15521


namespace value_of_f_at_2_l155_15576

def f (x : ℤ) : ℤ := x^3 - x

theorem value_of_f_at_2 : f 2 = 6 := by
  sorry

end value_of_f_at_2_l155_15576


namespace find_integer_n_l155_15513

theorem find_integer_n : ∃ n, 5 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ n = 9 :=   
by 
  -- The proof will be written here.
  sorry

end find_integer_n_l155_15513
