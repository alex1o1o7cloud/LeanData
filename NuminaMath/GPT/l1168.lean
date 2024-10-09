import Mathlib

namespace mark_sold_8_boxes_less_l1168_116863

theorem mark_sold_8_boxes_less (T M A x : ℕ) (hT : T = 9) 
    (hM : M = T - x) (hA : A = T - 2) 
    (hM_ge_1 : 1 ≤ M) (hA_ge_1 : 1 ≤ A) 
    (h_sum_lt_T : M + A < T) : x = 8 := 
by
  sorry

end mark_sold_8_boxes_less_l1168_116863


namespace price_per_half_pound_of_basil_l1168_116867

theorem price_per_half_pound_of_basil
    (cost_per_pound_eggplant : ℝ)
    (pounds_eggplant : ℝ)
    (cost_per_pound_zucchini : ℝ)
    (pounds_zucchini : ℝ)
    (cost_per_pound_tomato : ℝ)
    (pounds_tomato : ℝ)
    (cost_per_pound_onion : ℝ)
    (pounds_onion : ℝ)
    (quarts_ratatouille : ℝ)
    (cost_per_quart : ℝ) :
    pounds_eggplant = 5 → cost_per_pound_eggplant = 2 →
    pounds_zucchini = 4 → cost_per_pound_zucchini = 2 →
    pounds_tomato = 4 → cost_per_pound_tomato = 3.5 →
    pounds_onion = 3 → cost_per_pound_onion = 1 →
    quarts_ratatouille = 4 → cost_per_quart = 10 →
    (cost_per_quart * quarts_ratatouille - 
    (cost_per_pound_eggplant * pounds_eggplant + 
    cost_per_pound_zucchini * pounds_zucchini + 
    cost_per_pound_tomato * pounds_tomato + 
    cost_per_pound_onion * pounds_onion)) / 2 = 2.5 :=
by
    intros h₁ h₂ h₃ h₄ h₅ h₆ h₇ h₈ h₉ h₀
    rw [h₁, h₂, h₃, h₄, h₅, h₆, h₇, h₈, h₉, h₀]
    sorry

end price_per_half_pound_of_basil_l1168_116867


namespace fewer_cans_l1168_116819

theorem fewer_cans (sarah_yesterday lara_more alex_yesterday sarah_today lara_today alex_today : ℝ)
  (H1 : sarah_yesterday = 50.5)
  (H2 : lara_more = 30.3)
  (H3 : alex_yesterday = 90.2)
  (H4 : sarah_today = 40.7)
  (H5 : lara_today = 70.5)
  (H6 : alex_today = 55.3) :
  (sarah_yesterday + (sarah_yesterday + lara_more) + alex_yesterday) - (sarah_today + lara_today + alex_today) = 55 :=
by {
  -- Sorry to skip the proof
  sorry
}

end fewer_cans_l1168_116819


namespace sum_first_100_odd_l1168_116845

-- Define the sequence of odd numbers.
def odd (n : ℕ) : ℕ := 2 * n + 1

-- Define the sum of the first n odd natural numbers.
def sumOdd (n : ℕ) : ℕ := (n * (n + 1))

-- State the theorem.
theorem sum_first_100_odd : sumOdd 100 = 10000 :=
by
  -- Skipping the proof as per the instructions
  sorry

end sum_first_100_odd_l1168_116845


namespace exists_c_same_digit_occurrences_l1168_116850

theorem exists_c_same_digit_occurrences (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ c : ℕ, c > 0 ∧ ∀ d : ℕ, d ≠ 0 → 
    (Nat.digits 10 (c * m)).count d = (Nat.digits 10 (c * n)).count d := sorry

end exists_c_same_digit_occurrences_l1168_116850


namespace ticket_sales_total_cost_l1168_116806

noncomputable def total_ticket_cost (O B : ℕ) : ℕ :=
  12 * O + 8 * B

theorem ticket_sales_total_cost (O B : ℕ) (h1 : O + B = 350) (h2 : B = O + 90) :
  total_ticket_cost O B = 3320 :=
by
  -- the proof steps calculating the total cost will go here
  sorry

end ticket_sales_total_cost_l1168_116806


namespace original_volume_of_cube_l1168_116875

theorem original_volume_of_cube (a : ℕ) 
  (h1 : (a + 2) * (a - 2) * (a + 3) = a^3 - 7) : 
  a = 3 :=
by sorry

end original_volume_of_cube_l1168_116875


namespace x_range_l1168_116869

theorem x_range (x : ℝ) : (x + 2) > 0 → (3 - x) ≥ 0 → (-2 < x ∧ x ≤ 3) :=
by
  intro h1 h2
  constructor
  { linarith }
  { linarith }

end x_range_l1168_116869


namespace freight_cost_minimization_l1168_116820

-- Define the main parameters: tonnage and costs for the trucks.
def freight_cost (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  65 * num_seven_ton_trucks + 50 * num_five_ton_trucks

-- Define the total transported capacity by the two types of trucks.
def total_capacity (num_seven_ton_trucks : ℕ) (num_five_ton_trucks : ℕ) : ℕ :=
  7 * num_seven_ton_trucks + 5 * num_five_ton_trucks

-- Define the minimum freight cost given the conditions.
def minimum_freight_cost := 685

-- The theorem we want to prove.
theorem freight_cost_minimization : ∃ x y : ℕ, total_capacity x y ≥ 73 ∧
  (freight_cost x y = minimum_freight_cost) :=
by
  sorry

end freight_cost_minimization_l1168_116820


namespace merchant_discount_l1168_116880

-- Definitions based on conditions
def original_price : ℝ := 1
def increased_price : ℝ := original_price * 1.2
def final_price : ℝ := increased_price * 0.8
def actual_discount : ℝ := original_price - final_price

-- The theorem to be proved
theorem merchant_discount : actual_discount = 0.04 :=
by
  -- Proof goes here
  sorry

end merchant_discount_l1168_116880


namespace problem_sequence_k_term_l1168_116817

theorem problem_sequence_k_term (a : ℕ → ℤ) (S : ℕ → ℤ) (h₀ : ∀ n, S n = n^2 - 9 * n)
    (h₁ : ∀ n, a n = S n - S (n - 1)) (h₂ : 5 < a 8 ∧ a 8 < 8) : 8 = 8 :=
sorry

end problem_sequence_k_term_l1168_116817


namespace price_difference_is_correct_l1168_116893

noncomputable def total_cost : ℝ := 70.93
noncomputable def cost_of_pants : ℝ := 34.0
noncomputable def cost_of_belt : ℝ := total_cost - cost_of_pants
noncomputable def price_difference : ℝ := cost_of_belt - cost_of_pants

theorem price_difference_is_correct :
  price_difference = 2.93 := by
  sorry

end price_difference_is_correct_l1168_116893


namespace tom_mowing_lawn_l1168_116891

theorem tom_mowing_lawn (hours_to_mow : ℕ) (time_worked : ℕ) (fraction_mowed_per_hour : ℚ) : 
  (hours_to_mow = 6) → 
  (time_worked = 3) → 
  (fraction_mowed_per_hour = (1 : ℚ) / hours_to_mow) → 
  (1 - (time_worked * fraction_mowed_per_hour) = (1 : ℚ) / 2) :=
by
  intros h1 h2 h3
  sorry

end tom_mowing_lawn_l1168_116891


namespace value_of_a_l1168_116815

theorem value_of_a (a : ℚ) (h : 2 * a + a / 2 = 9 / 2) : a = 9 / 5 :=
by
  sorry

end value_of_a_l1168_116815


namespace primes_with_consecutives_l1168_116803

-- Define what it means for a number to be prime
def is_prime (n : Nat) := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬ (n % m = 0)

-- Define the main theorem to prove
theorem primes_with_consecutives (p : Nat) : is_prime p ∧ is_prime (p + 2) ∧ is_prime (p + 4) → p = 3 :=
by
  sorry

end primes_with_consecutives_l1168_116803


namespace students_who_won_first_prize_l1168_116866

theorem students_who_won_first_prize :
  ∃ x : ℤ, 30 ≤ x ∧ x ≤ 55 ∧ (x % 3 = 2) ∧ (x % 5 = 4) ∧ (x % 7 = 2) ∧ x = 44 :=
by
  sorry

end students_who_won_first_prize_l1168_116866


namespace range_of_a_l1168_116834

-- Define the quadratic inequality
def quadratic_inequality (a x : ℝ) : ℝ := (a-1)*x^2 + (a-1)*x + 1

theorem range_of_a :
  (∀ x : ℝ, quadratic_inequality a x > 0) ↔ (1 ≤ a ∧ a < 5) :=
by
  sorry

end range_of_a_l1168_116834


namespace fewer_miles_per_gallon_city_l1168_116889

-- Define the given conditions.
def miles_per_tankful_highway : ℕ := 420
def miles_per_tankful_city : ℕ := 336
def miles_per_gallon_city : ℕ := 24

-- Define the question as a theorem that proves how many fewer miles per gallon in the city compared to the highway.
theorem fewer_miles_per_gallon_city (G : ℕ) (hG : G = miles_per_tankful_city / miles_per_gallon_city) :
  miles_per_tankful_highway / G - miles_per_gallon_city = 6 :=
by
  -- The proof will be provided here.
  sorry

end fewer_miles_per_gallon_city_l1168_116889


namespace compute_expression_l1168_116899

noncomputable def quadratic_roots (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α * β = -2) ∧ (α + β = -p) ∧ (γ * δ = -2) ∧ (γ + δ = -q)

theorem compute_expression (p q α β γ δ : ℝ) 
  (h₁ : quadratic_roots p q α β γ δ) :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = -(p^2 - q^2) :=
by
  -- We will provide the proof here
  sorry

end compute_expression_l1168_116899


namespace rectangle_side_multiple_of_6_l1168_116822

theorem rectangle_side_multiple_of_6 (a b : ℕ) (h : ∃ n : ℕ, a * b = n * 6) : a % 6 = 0 ∨ b % 6 = 0 :=
sorry

end rectangle_side_multiple_of_6_l1168_116822


namespace rectangle_perimeter_l1168_116859

variable (a b : ℝ)
variable (h1 : a * b = 24)
variable (h2 : a^2 + b^2 = 121)

theorem rectangle_perimeter : 2 * (a + b) = 26 := 
by
  sorry

end rectangle_perimeter_l1168_116859


namespace remaining_movies_to_watch_l1168_116812

theorem remaining_movies_to_watch (total_movies watched_movies remaining_movies : ℕ) 
  (h1 : total_movies = 8) 
  (h2 : watched_movies = 4) 
  (h3 : remaining_movies = total_movies - watched_movies) : 
  remaining_movies = 4 := 
by
  sorry

end remaining_movies_to_watch_l1168_116812


namespace original_rectangle_length_l1168_116879

-- Define the problem conditions
def length_three_times_width (l w : ℕ) : Prop :=
  l = 3 * w

def length_decreased_width_increased (l w : ℕ) : Prop :=
  l - 5 = w + 5

-- Define the proof problem
theorem original_rectangle_length (l w : ℕ) (H1 : length_three_times_width l w) (H2 : length_decreased_width_increased l w) : l = 15 :=
sorry

end original_rectangle_length_l1168_116879


namespace circles_are_externally_tangent_l1168_116816

-- Conditions given in the problem
def r1 (r2 : ℝ) : Prop := ∃ r1 : ℝ, r1 * r2 = 10 ∧ r1 + r2 = 7
def distance := 7

-- The positional relationship proof problem statement
theorem circles_are_externally_tangent (r1 r2 : ℝ) (h : r1 * r2 = 10 ∧ r1 + r2 = 7) (d : ℝ) (h_d : d = distance) : 
  d = r1 + r2 :=
sorry

end circles_are_externally_tangent_l1168_116816


namespace square_pyramid_sum_l1168_116811

def square_pyramid_faces : Nat := 5
def square_pyramid_edges : Nat := 8
def square_pyramid_vertices : Nat := 5

theorem square_pyramid_sum : square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18 := by
  sorry

end square_pyramid_sum_l1168_116811


namespace arithmetic_mean_124_4_31_l1168_116836

theorem arithmetic_mean_124_4_31 :
  let numbers := [12, 25, 39, 48]
  let total := 124
  let count := 4
  (total / count : ℝ) = 31 := by
  sorry

end arithmetic_mean_124_4_31_l1168_116836


namespace Skylar_chickens_less_than_triple_Colten_l1168_116897

def chickens_count (S Q C : ℕ) : Prop := 
  Q + S + C = 383 ∧ 
  Q = 2 * S + 25 ∧ 
  C = 37

theorem Skylar_chickens_less_than_triple_Colten (S Q C : ℕ) 
  (h : chickens_count S Q C) : (3 * C - S = 4) := 
sorry

end Skylar_chickens_less_than_triple_Colten_l1168_116897


namespace find_abc_l1168_116877

theorem find_abc (a b c : ℕ) (h_coprime_ab : gcd a b = 1) (h_coprime_ac : gcd a c = 1) 
  (h_coprime_bc : gcd b c = 1) (h1 : ab + bc + ac = 431) (h2 : a + b + c = 39) 
  (h3 : a + b + (ab / c) = 18) : 
  a = 7 ∧ b = 9 ∧ c = 23 := 
sorry

end find_abc_l1168_116877


namespace cost_price_of_cloth_l1168_116898

-- Definitions for conditions
def sellingPrice (totalMeters : ℕ) : ℕ := 8500
def profitPerMeter : ℕ := 15
def totalMeters : ℕ := 85

-- Proof statement with conditions and expected proof
theorem cost_price_of_cloth : 
  (sellingPrice totalMeters) = 8500 -> 
  profitPerMeter = 15 -> 
  totalMeters = 85 -> 
  (8500 - (profitPerMeter * totalMeters)) / totalMeters = 85 := 
by 
  sorry

end cost_price_of_cloth_l1168_116898


namespace fourth_number_is_2_eighth_number_is_2_l1168_116853

-- Conditions as given in the problem
def initial_board := [1]

/-- Medians recorded in Mitya's notebook for the first 10 numbers -/
def medians := [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5]

/-- Prove that the fourth number written on the board is 2 given initial conditions. -/
theorem fourth_number_is_2 (board : ℕ → ℤ)  
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 3 = 2 :=
sorry

/-- Prove that the eighth number written on the board is 2 given initial conditions. -/
theorem eighth_number_is_2 (board : ℕ → ℤ) 
  (h1 : board 0 = 1)
  (h2 : medians = [1, 2, 3, 2.5, 3, 2.5, 2, 2, 2, 2.5])
  : board 7 = 2 :=
sorry

end fourth_number_is_2_eighth_number_is_2_l1168_116853


namespace value_of_e_over_f_l1168_116824

theorem value_of_e_over_f 
    (a b c d e f : ℝ) 
    (h1 : a * b * c = 1.875 * d * e * f)
    (h2 : a / b = 5 / 2)
    (h3 : b / c = 1 / 2)
    (h4 : c / d = 1)
    (h5 : d / e = 3 / 2) : 
    e / f = 1 / 3 :=
by
  sorry

end value_of_e_over_f_l1168_116824


namespace raghu_investment_l1168_116890

-- Define the conditions as Lean definitions
def invest_raghu : Real := sorry
def invest_trishul := 0.90 * invest_raghu
def invest_vishal := 1.10 * invest_trishul
def invest_chandni := 1.15 * invest_vishal
def total_investment := invest_raghu + invest_trishul + invest_vishal + invest_chandni

-- State the proof problem
theorem raghu_investment (h : total_investment = 10700) : invest_raghu = 2656.25 :=
by
  sorry

end raghu_investment_l1168_116890


namespace parabola_vertex_example_l1168_116844

noncomputable def parabola_vertex (a b c : ℝ) := (-b / (2 * a), (4 * a * c - b^2) / (4 * a))

theorem parabola_vertex_example : parabola_vertex (-4) (-16) (-20) = (-2, -4) :=
by
  sorry

end parabola_vertex_example_l1168_116844


namespace part1_daily_sales_profit_final_max_daily_sales_profit_l1168_116833

-- Conditions from part (a)
def original_selling_price : ℚ := 30
def cost_price : ℚ := 15
def original_sales_volume : ℚ := 60
def sales_increase_per_yuan : ℚ := 10

-- Part (1): Daily sales profit if the price is reduced by 2 yuan
def new_selling_price1 : ℚ := original_selling_price - 2
def new_sales_volume1 : ℚ := original_sales_volume + (2 * sales_increase_per_yuan)
def profit_per_kilogram1 : ℚ := new_selling_price1 - cost_price
def daily_sales_profit1 : ℚ := profit_per_kilogram1 * new_sales_volume1

theorem part1_daily_sales_profit : daily_sales_profit1 = 1040 := by
  sorry

-- Part (2): Maximum daily sales profit and corresponding selling price
def selling_price_at_max_profit : ℚ := 51 / 2

def daily_profit (x : ℚ) : ℚ :=
  (x - cost_price) * (original_sales_volume + (original_selling_price - x) * sales_increase_per_yuan)

theorem final_max_daily_sales_profit :
  (∀ x : ℚ, daily_profit x ≤ daily_profit selling_price_at_max_profit) ∧ daily_profit selling_price_at_max_profit = 1102.5 := by
  sorry

end part1_daily_sales_profit_final_max_daily_sales_profit_l1168_116833


namespace ratio_P_K_is_2_l1168_116885

theorem ratio_P_K_is_2 (P K M : ℝ) (r : ℝ)
  (h1: P + K + M = 153)
  (h2: P = r * K)
  (h3: P = (1/3) * M)
  (h4: M = K + 85) : r = 2 :=
  sorry

end ratio_P_K_is_2_l1168_116885


namespace cube_root_of_sum_powers_l1168_116821

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem cube_root_of_sum_powers :
  cube_root (2^7 + 2^7 + 2^7) = 4 * cube_root 2 :=
by
  sorry

end cube_root_of_sum_powers_l1168_116821


namespace simplify_expression_l1168_116814

variable (x : ℝ)

theorem simplify_expression :
  3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 = -x^2 + 23 * x - 3 :=
sorry

end simplify_expression_l1168_116814


namespace sin_B_value_l1168_116813

variable {A B C : Real}
variable {a b c : Real}
variable {sin_A sin_B sin_C : Real}

-- Given conditions as hypotheses
axiom h1 : c = 2 * a
axiom h2 : b * sin_B - a * sin_A = (1 / 2) * a * sin_C

-- The statement to prove
theorem sin_B_value : sin_B = Real.sqrt 7 / 4 :=
by
  -- Proof omitted
  sorry

end sin_B_value_l1168_116813


namespace calculate_total_interest_rate_l1168_116800

noncomputable def total_investment : ℝ := 10000
noncomputable def amount_invested_11_percent : ℝ := 3750
noncomputable def amount_invested_9_percent : ℝ := total_investment - amount_invested_11_percent
noncomputable def interest_rate_9_percent : ℝ := 0.09
noncomputable def interest_rate_11_percent : ℝ := 0.11

noncomputable def interest_from_9_percent : ℝ := interest_rate_9_percent * amount_invested_9_percent
noncomputable def interest_from_11_percent : ℝ := interest_rate_11_percent * amount_invested_11_percent

noncomputable def total_interest : ℝ := interest_from_9_percent + interest_from_11_percent

noncomputable def total_interest_rate : ℝ := (total_interest / total_investment) * 100

theorem calculate_total_interest_rate :
  total_interest_rate = 9.75 :=
by 
  sorry

end calculate_total_interest_rate_l1168_116800


namespace factorize_expression_l1168_116895

theorem factorize_expression (x : ℝ) : 2 * x^3 - 8 * x^2 + 8 * x = 2 * x * (x - 2) ^ 2 := 
sorry

end factorize_expression_l1168_116895


namespace sqrt_expression_l1168_116831

theorem sqrt_expression :
  Real.sqrt 18 - 3 * Real.sqrt (1 / 2) + Real.sqrt 2 = (5 * Real.sqrt 2) / 2 :=
by
  sorry

end sqrt_expression_l1168_116831


namespace problem_equivalence_l1168_116873

theorem problem_equivalence : (7^2 - 3^2)^4 = 2560000 :=
by
  sorry

end problem_equivalence_l1168_116873


namespace floor_expression_equality_l1168_116849

theorem floor_expression_equality :
  ⌊((2023^3 : ℝ) / (2021 * 2022) - (2021^3 : ℝ) / (2022 * 2023))⌋ = 8 := 
sorry

end floor_expression_equality_l1168_116849


namespace weight_mixture_is_correct_l1168_116825

noncomputable def weight_mixture_in_kg (weight_a_per_liter weight_b_per_liter : ℝ)
  (ratio_a ratio_b total_volume_liters weight_conversion : ℝ) : ℝ :=
  let total_parts := ratio_a + ratio_b
  let volume_per_part := total_volume_liters / total_parts
  let volume_a := ratio_a * volume_per_part
  let volume_b := ratio_b * volume_per_part
  let weight_a := volume_a * weight_a_per_liter
  let weight_b := volume_b * weight_b_per_liter
  let total_weight_gm := weight_a + weight_b
  total_weight_gm / weight_conversion

theorem weight_mixture_is_correct :
  weight_mixture_in_kg 900 700 3 2 4 1000 = 3.280 :=
by
  -- Calculation should follow from the def
  sorry

end weight_mixture_is_correct_l1168_116825


namespace find_unknown_l1168_116855

theorem find_unknown (x : ℝ) :
  300 * 2 + (x + 4) * (1 / 8) = 602 → x = 12 :=
by 
  sorry

end find_unknown_l1168_116855


namespace problem_statement_l1168_116837

theorem problem_statement (a b : ℝ) (h0 : 0 < b) (h1 : b < 1/2) (h2 : 1/2 < a) (h3 : a < 1) :
  (0 < a - b) ∧ (a - b < 1) ∧ (ab < a^2) ∧ (a - 1/b < b - 1/a) :=
by 
  sorry

end problem_statement_l1168_116837


namespace total_tubes_in_consignment_l1168_116847

theorem total_tubes_in_consignment (N : ℕ) 
  (h : (5 / (N : ℝ)) * (4 / (N - 1 : ℝ)) = 0.05263157894736842) : 
  N = 20 := 
sorry

end total_tubes_in_consignment_l1168_116847


namespace paperclip_day_l1168_116854

theorem paperclip_day:
  ∃ k : ℕ, 5 * 3 ^ k > 500 ∧ ∀ m : ℕ, m < k → 5 * 3 ^ m ≤ 500 ∧ k % 7 = 5 :=
sorry

end paperclip_day_l1168_116854


namespace insurance_covers_80_percent_of_medical_bills_l1168_116868

theorem insurance_covers_80_percent_of_medical_bills 
    (vaccine_cost : ℕ) (num_vaccines : ℕ) (doctor_visit_cost trip_cost : ℕ) (amount_tom_pays : ℕ) 
    (total_cost := num_vaccines * vaccine_cost + doctor_visit_cost) 
    (total_trip_cost := trip_cost + total_cost)
    (insurance_coverage := total_trip_cost - amount_tom_pays)
    (percent_covered := (insurance_coverage * 100) / total_cost) :
    vaccine_cost = 45 → num_vaccines = 10 → doctor_visit_cost = 250 → trip_cost = 1200 → amount_tom_pays = 1340 →
    percent_covered = 80 := 
by
  sorry

end insurance_covers_80_percent_of_medical_bills_l1168_116868


namespace find_a_of_even_function_l1168_116860

-- Define the function f
def f (x a : ℝ) := (x + 1) * (x + a)

-- State the theorem to be proven
theorem find_a_of_even_function (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  -- The actual proof goes here
  sorry

end find_a_of_even_function_l1168_116860


namespace remainder_eq_one_l1168_116870

theorem remainder_eq_one (n : ℤ) (h : n % 6 = 1) : (n + 150) % 6 = 1 := 
by
  sorry

end remainder_eq_one_l1168_116870


namespace alice_cell_phone_cost_l1168_116872

theorem alice_cell_phone_cost
  (base_cost : ℕ)
  (included_hours : ℕ)
  (text_cost_per_message : ℕ)
  (extra_minute_cost : ℕ)
  (messages_sent : ℕ)
  (hours_spent : ℕ) :
  base_cost = 25 →
  included_hours = 40 →
  text_cost_per_message = 4 →
  extra_minute_cost = 5 →
  messages_sent = 150 →
  hours_spent = 42 →
  (base_cost + (messages_sent * text_cost_per_message) / 100 + ((hours_spent - included_hours) * 60 * extra_minute_cost) / 100) = 37 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end alice_cell_phone_cost_l1168_116872


namespace chord_slope_of_ellipse_l1168_116810

theorem chord_slope_of_ellipse :
  (∃ (x1 y1 x2 y2 : ℝ), (x1 + x2)/2 = 4 ∧ (y1 + y2)/2 = 2 ∧
    (x1^2)/36 + (y1^2)/9 = 1 ∧ (x2^2)/36 + (y2^2)/9 = 1) →
    (∃ k : ℝ, k = (y1 - y2)/(x1 - x2) ∧ k = -1/2) :=
sorry

end chord_slope_of_ellipse_l1168_116810


namespace probability_same_color_opposite_foot_l1168_116829

def total_shoes := 28

def black_pairs := 7
def brown_pairs := 4
def gray_pairs := 2
def red_pair := 1

def total_pairs := black_pairs + brown_pairs + gray_pairs + red_pair

theorem probability_same_color_opposite_foot : 
  (7 + 4 + 2 + 1) * 2 = total_shoes →
  (14 / 28 * (7 / 27) + 8 / 28 * (4 / 27) + 4 / 28 * (2 / 27) + 2 / 28 * (1 / 27)) = (20 / 63) :=
by
  sorry

end probability_same_color_opposite_foot_l1168_116829


namespace johnnys_age_l1168_116808

theorem johnnys_age (x : ℤ) (h : x + 2 = 2 * (x - 3)) : x = 8 := sorry

end johnnys_age_l1168_116808


namespace simple_interest_borrowed_rate_l1168_116857

theorem simple_interest_borrowed_rate
  (P_borrowed P_lent : ℝ)
  (n_years : ℕ)
  (gain_per_year : ℝ)
  (simple_interest_lent_rate : ℝ)
  (SI_lending : ℝ := P_lent * simple_interest_lent_rate * n_years / 100)
  (total_gain : ℝ := gain_per_year * n_years) :
  SI_lending = 1000 →
  total_gain = 100 →
  ∀ (SI_borrowing : ℝ), SI_borrowing = SI_lending - total_gain →
  ∀ (R_borrowed : ℝ), SI_borrowing = P_borrowed * R_borrowed * n_years / 100 →
  R_borrowed = 9 := 
by
  sorry

end simple_interest_borrowed_rate_l1168_116857


namespace range_of_a_l1168_116842

open Real

theorem range_of_a (a : ℝ) :
  (∀ x, |x - 1| < 3 → (x + 2) * (x + a) < 0) ∧ ¬ (∀ x, (x + 2) * (x + a) < 0 → |x - 1| < 3) →
  a < -4 :=
by
  sorry

end range_of_a_l1168_116842


namespace intersection_of_A_and_B_range_of_a_l1168_116823

open Set

namespace ProofProblem

def A : Set ℝ := {x | -1 ≤ x ∧ x < 3}
def B : Set ℝ := {x | x ≥ 2}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

theorem intersection_of_A_and_B :
  A ∩ B = {x | 2 ≤ x ∧ x < 3} := 
sorry

theorem range_of_a (a : ℝ) :
  (B ∪ C a) = C a → a ≤ 3 :=
sorry

end ProofProblem

end intersection_of_A_and_B_range_of_a_l1168_116823


namespace supplement_of_complement_of_75_degree_angle_l1168_116830

def angle : ℕ := 75
def complement_angle (a : ℕ) := 90 - a
def supplement_angle (a : ℕ) := 180 - a

theorem supplement_of_complement_of_75_degree_angle : supplement_angle (complement_angle angle) = 165 :=
by
  sorry

end supplement_of_complement_of_75_degree_angle_l1168_116830


namespace find_polynomial_l1168_116835

theorem find_polynomial
  (M : ℝ → ℝ)
  (h : ∀ x, M x + 5 * x^2 - 4 * x - 3 = -1 * x^2 - 3 * x) :
  ∀ x, M x = -6 * x^2 + x + 3 :=
sorry

end find_polynomial_l1168_116835


namespace pool_length_l1168_116861

theorem pool_length (r : ℕ) (t : ℕ) (w : ℕ) (d : ℕ) (L : ℕ) 
  (H1 : r = 60)
  (H2 : t = 2000)
  (H3 : w = 80)
  (H4 : d = 10)
  (H5 : L = (r * t) / (w * d)) : L = 150 :=
by
  rw [H1, H2, H3, H4] at H5
  exact H5


end pool_length_l1168_116861


namespace jane_sandwich_count_l1168_116838

noncomputable def total_sandwiches : ℕ := 5 * 7 * 4

noncomputable def turkey_swiss_reduction : ℕ := 5 * 1 * 1

noncomputable def salami_bread_reduction : ℕ := 5 * 1 * 4

noncomputable def correct_sandwich_count : ℕ := 115

theorem jane_sandwich_count : total_sandwiches - turkey_swiss_reduction - salami_bread_reduction = correct_sandwich_count :=
by
  sorry

end jane_sandwich_count_l1168_116838


namespace Margo_James_pairs_probability_l1168_116802

def total_students : ℕ := 32
def Margo_pairs_prob : ℚ := 1 / 31
def James_pairs_prob : ℚ := 1 / 30
def total_prob : ℚ := Margo_pairs_prob * James_pairs_prob

theorem Margo_James_pairs_probability :
  total_prob = 1 / 930 := 
by
  -- sorry allows us to skip the proof steps, only statement needed
  sorry

end Margo_James_pairs_probability_l1168_116802


namespace Mille_suckers_l1168_116876

theorem Mille_suckers:
  let pretzels := 64
  let goldfish := 4 * pretzels
  let baggies := 16
  let items_per_baggie := 22
  let total_items_needed := baggies * items_per_baggie
  let total_pretzels_and_goldfish := pretzels + goldfish
  let suckers := total_items_needed - total_pretzels_and_goldfish
  suckers = 32 := 
by sorry

end Mille_suckers_l1168_116876


namespace g_symmetry_solutions_l1168_116896

noncomputable def g : ℝ → ℝ := sorry

theorem g_symmetry_solutions (g_def: ∀ (x : ℝ), x ≠ 0 → g x + 3 * g (1 / x) = 6 * x^2) :
  ∀ (x : ℝ), g x = g (-x) → x = 1 ∨ x = -1 :=
by
  sorry

end g_symmetry_solutions_l1168_116896


namespace find_b_perpendicular_lines_l1168_116826

theorem find_b_perpendicular_lines (b : ℚ)
  (line1 : (3 : ℚ) * x + 4 * y - 6 = 0)
  (line2 : b * x + 4 * y - 6 = 0)
  (perpendicular : ( - (3 : ℚ) / 4 ) * ( - (b / 4) ) = -1) :
  b = - (16 : ℚ) / 3 := 
sorry

end find_b_perpendicular_lines_l1168_116826


namespace relationship_of_variables_l1168_116839

variable {a b c d : ℝ}

theorem relationship_of_variables 
  (h1 : d - a < c - b) 
  (h2 : c - b < 0) 
  (h3 : d - b = c - a) : 
  d < c ∧ c < b ∧ b < a := 
sorry

end relationship_of_variables_l1168_116839


namespace Mike_exercises_l1168_116856

theorem Mike_exercises :
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490 :=
by
  let pull_ups_per_visit := 2
  let push_ups_per_visit := 5
  let squats_per_visit := 10
  let office_visits_per_day := 5
  let kitchen_visits_per_day := 8
  let living_room_visits_per_day := 7
  let days_per_week := 7
  let total_pull_ups := pull_ups_per_visit * office_visits_per_day * days_per_week
  let total_push_ups := push_ups_per_visit * kitchen_visits_per_day * days_per_week
  let total_squats := squats_per_visit * living_room_visits_per_day * days_per_week
  have h1 : total_pull_ups = 2 * 5 * 7 := rfl
  have h2 : total_push_ups = 5 * 8 * 7 := rfl
  have h3 : total_squats = 10 * 7 * 7 := rfl
  show total_pull_ups = 70 ∧ total_push_ups = 280 ∧ total_squats = 490
  sorry

end Mike_exercises_l1168_116856


namespace pen_price_relationship_l1168_116871

variable (x : ℕ) -- x represents the number of pens
variable (y : ℝ) -- y represents the total selling price in dollars
variable (p : ℝ) -- p represents the price per pen

-- Each box contains 10 pens
def pens_per_box := 10

-- Each box is sold for $16
def price_per_box := 16

-- Given the conditions, prove the relationship between y and x
theorem pen_price_relationship (hx : x = 10) (hp : p = 16) :
  y = 1.6 * x := sorry

end pen_price_relationship_l1168_116871


namespace problem_1_problem_2_l1168_116892

variable {a : ℕ → ℝ}
variable (n : ℕ)

-- Conditions of the problem
def seq_positive : ∀ (k : ℕ), a k > 0 := sorry
def a1 : a 1 = 1 := sorry
def recurrence (n : ℕ) : a (n + 1) = (a n + 1) / (12 * a n) := sorry

-- Proofs to be provided
theorem problem_1 : ∀ n : ℕ, a (2 * n + 1) < a (2 * n - 1) := 
by 
  apply sorry 

theorem problem_2 : ∀ n : ℕ, 1 / 6 ≤ a n ∧ a n ≤ 1 := 
by 
  apply sorry 

end problem_1_problem_2_l1168_116892


namespace smallest_common_multiple_of_10_11_18_l1168_116805

theorem smallest_common_multiple_of_10_11_18 : 
  ∃ (n : ℕ), (n % 10 = 0) ∧ (n % 11 = 0) ∧ (n % 18 = 0) ∧ (n = 990) :=
by
  sorry

end smallest_common_multiple_of_10_11_18_l1168_116805


namespace least_common_addition_of_primes_l1168_116840

theorem least_common_addition_of_primes (x y : ℕ) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < y) (h : 4 * x + y = 87) : x + y = 81 := 
sorry

end least_common_addition_of_primes_l1168_116840


namespace find_a_value_l1168_116878

theorem find_a_value
  (a : ℝ)
  (h : ∀ x, 0 ≤ x ∧ x ≤ (π / 2) → a * Real.sin x + Real.cos x ≤ 2)
  (h_max : ∃ x, 0 ≤ x ∧ x ≤ (π / 2) ∧ a * Real.sin x + Real.cos x = 2) :
  a = Real.sqrt 3 :=
sorry

end find_a_value_l1168_116878


namespace solution_m_in_interval_l1168_116841

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
if x < 1 then -x^2 + 2 * m * x - 2 else 1 + Real.log x

theorem solution_m_in_interval :
  ∃ m : ℝ, (1 ≤ m ∧ m ≤ 2) ∧
  (∀ x < 1, ∀ y < 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x ≥ 1, ∀ y ≥ 1, x ≤ y → f x m ≤ f y m) ∧
  (∀ x < 1, ∀ y ≥ 1, f x m ≤ f y m) :=
by
  sorry

end solution_m_in_interval_l1168_116841


namespace factory_output_increase_l1168_116881

theorem factory_output_increase (x : ℝ) (h : (1 + x / 100) ^ 4 = 4) : x = 41.4 :=
by
  -- Given (1 + x / 100) ^ 4 = 4
  sorry

end factory_output_increase_l1168_116881


namespace avg_salary_l1168_116809

-- Conditions as definitions
def number_of_technicians : Nat := 7
def salary_per_technician : Nat := 10000
def number_of_workers : Nat := 14
def salary_per_non_technician : Nat := 6000

-- Total salary of technicians
def total_salary_technicians : Nat := number_of_technicians * salary_per_technician

-- Number of non-technicians
def number_of_non_technicians : Nat := number_of_workers - number_of_technicians

-- Total salary of non-technicians
def total_salary_non_technicians : Nat := number_of_non_technicians * salary_per_non_technician

-- Total salary
def total_salary_all_workers : Nat := total_salary_technicians + total_salary_non_technicians

-- Average salary of all workers
def avg_salary_all_workers : Nat := total_salary_all_workers / number_of_workers

-- Theorem to prove
theorem avg_salary (A : Nat) (h : A = avg_salary_all_workers) : A = 8000 := by
  sorry

end avg_salary_l1168_116809


namespace pizza_left_for_Wally_l1168_116801

theorem pizza_left_for_Wally (a b c : ℚ) (ha : a = 1/3) (hb : b = 1/6) (hc : c = 1/4) :
  1 - (a + b + c) = 1/4 :=
by
  sorry

end pizza_left_for_Wally_l1168_116801


namespace initial_provisions_last_l1168_116865

theorem initial_provisions_last (x : ℕ) (h : 2000 * (x - 20) = 4000 * 10) : x = 40 :=
by sorry

end initial_provisions_last_l1168_116865


namespace books_brought_back_l1168_116858

def initial_books : ℕ := 235
def taken_out_tuesday : ℕ := 227
def taken_out_friday : ℕ := 35
def books_remaining : ℕ := 29

theorem books_brought_back (B : ℕ) :
  B = 56 ↔ (initial_books - taken_out_tuesday + B - taken_out_friday = books_remaining) :=
by
  -- proof steps would go here
  sorry

end books_brought_back_l1168_116858


namespace distance_eq_l1168_116818

open Real

variables (a b c d p q: ℝ)

-- Conditions from step a)
def onLine1 : Prop := b = (p-1)*a + q
def onLine2 : Prop := d = (p-1)*c + q

-- Theorem about the distance between points (a, b) and (c, d)
theorem distance_eq : 
  onLine1 a b p q → 
  onLine2 c d p q → 
  dist (a, b) (c, d) = abs (a - c) * sqrt (1 + (p - 1)^2) := 
by
  intros h1 h2
  sorry

end distance_eq_l1168_116818


namespace num_repeating_decimals_between_1_and_20_l1168_116883

def is_repeating_decimal (a b : ℕ) : Prop :=
  ∀ p q : ℕ, ¬ b = 2^p * 5^q

theorem num_repeating_decimals_between_1_and_20 :
  ∃ (cnt : ℕ), cnt = 20 ∧
  ∀ n : ℕ, (1 ≤ n ∧ n ≤ 20) → is_repeating_decimal n 18 := 
by
  sorry

end num_repeating_decimals_between_1_and_20_l1168_116883


namespace trailing_zeros_of_9_pow_999_plus_1_l1168_116862

theorem trailing_zeros_of_9_pow_999_plus_1 :
  ∃ n : ℕ, n = 999 ∧ (9^n + 1) % 10 = 0 ∧ (9^n + 1) % 100 ≠ 0 :=
by
  sorry

end trailing_zeros_of_9_pow_999_plus_1_l1168_116862


namespace fewer_students_played_thursday_l1168_116886

variable (w t : ℕ)

theorem fewer_students_played_thursday (h1 : w = 37) (h2 : w + t = 65) : w - t = 9 :=
by
  sorry

end fewer_students_played_thursday_l1168_116886


namespace find_xyz_l1168_116851

theorem find_xyz (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49) 
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 17) 
  (h3 : x^3 + y^3 + z^3 = 27) : 
  x * y * z = 32 / 3 :=
  sorry

end find_xyz_l1168_116851


namespace find_a_and_an_l1168_116894

-- Given Sequences
def S (n : ℕ) (a : ℝ) : ℝ := 3^n - a

def is_geometric_sequence (a_n : ℕ → ℝ) : Prop := ∃ a1 q, q ≠ 1 ∧ ∀ n, a_n n = a1 * q^n

-- The main statement
theorem find_a_and_an (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (a : ℝ) :
  (∀ n, S_n n = 3^n - a) ∧ is_geometric_sequence a_n →
  ∃ a, a = 1 ∧ ∀ n, a_n n = 2 * 3^(n-1) :=
by
  sorry

end find_a_and_an_l1168_116894


namespace intersection_is_2_l1168_116827

-- Define the sets A and B
def A : Set ℝ := { x | x < 1 }
def B : Set ℝ := { -1, 0, 2 }

-- Define the complement of A
def A_complement : Set ℝ := { x | x ≥ 1 }

-- Define the intersection of the complement of A and B
def intersection : Set ℝ := A_complement ∩ B

-- Prove that the intersection is {2}
theorem intersection_is_2 : intersection = {2} := by
  sorry

end intersection_is_2_l1168_116827


namespace bill_picked_apples_l1168_116888

-- Definitions from conditions
def children := 2
def apples_per_child_per_teacher := 3
def favorite_teachers := 2
def apples_per_pie := 10
def pies_baked := 2
def apples_left := 24

-- Number of apples given to teachers
def apples_for_teachers := children * apples_per_child_per_teacher * favorite_teachers

-- Number of apples used for pies
def apples_for_pies := pies_baked * apples_per_pie

-- The final theorem to be stated
theorem bill_picked_apples :
  apples_for_teachers + apples_for_pies + apples_left = 56 := 
sorry

end bill_picked_apples_l1168_116888


namespace farmer_plough_remaining_area_l1168_116807

theorem farmer_plough_remaining_area :
  ∀ (x R : ℕ),
  (90 * x = 3780) →
  (85 * (x + 2) + R = 3780) →
  R = 40 :=
by
  intros x R h1 h2
  sorry

end farmer_plough_remaining_area_l1168_116807


namespace expected_interval_is_correct_l1168_116884

-- Define the travel times via northern and southern routes
def travel_time_north : ℝ := 17
def travel_time_south : ℝ := 11

-- Define the average time difference between train arrivals
noncomputable def avg_time_diff : ℝ := 1.25

-- The average time difference for traveling from home to work versus work to home
noncomputable def time_diff_home_to_work : ℝ := 1

-- Define the expected interval between trains
noncomputable def expected_interval_between_trains := 3

-- Proof problem statement
theorem expected_interval_is_correct :
  ∃ (T : ℝ), (T = expected_interval_between_trains)
  → (travel_time_north - travel_time_south + 2 * avg_time_diff = time_diff_home_to_work)
  → (T = 3) := 
by
  use 3 
  intro h1 h2
  sorry

end expected_interval_is_correct_l1168_116884


namespace proof_problem_l1168_116843

noncomputable def problem : ℕ :=
  let p := 588
  let q := 0
  let r := 1
  p + q + r

theorem proof_problem
  (AB : ℝ) (P Q : ℝ) (AP BP PQ : ℝ) (angle_POQ : ℝ) 
  (h1 : AB = 1200)
  (h2 : AP + PQ = BP)
  (h3 : BP - Q = 600)
  (h4 : angle_POQ = 30)
  (h5 : PQ = 500)
  : problem = 589 := by
    sorry

end proof_problem_l1168_116843


namespace inequality_abc_l1168_116804

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2) * (b^2 + 2) * (c^2 + 2) ≥ 9 * (a * b + b * c + c * a) :=
by
  sorry

end inequality_abc_l1168_116804


namespace movie_ticket_cost_l1168_116864

variable (x : ℝ)
variable (h1 : x * 2 + 1.59 + 13.95 = 36.78)

theorem movie_ticket_cost : x = 10.62 :=
by
  sorry

end movie_ticket_cost_l1168_116864


namespace cos_squared_value_l1168_116832

theorem cos_squared_value (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) :
  Real.cos (α + π / 4) ^ 2 = 1 / 6 :=
sorry

end cos_squared_value_l1168_116832


namespace maximum_achievable_score_l1168_116852

def robot_initial_iq : Nat := 25
def problem_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem maximum_achievable_score 
  (initial_iq : Nat := robot_initial_iq) 
  (scores : List Nat := problem_scores) 
  : Nat :=
  31

end maximum_achievable_score_l1168_116852


namespace math_problem_l1168_116887

noncomputable def problem_statement : Prop :=
  ∃ b c : ℝ, 
  (∀ x : ℝ, (x^2 - b * x + c < 0) ↔ (-3 < x ∧ x < 2)) ∧ 
  (b + c = -7)

theorem math_problem : problem_statement := 
by
  sorry

end math_problem_l1168_116887


namespace total_distance_traveled_l1168_116828

noncomputable def row_speed_still_water : ℝ := 8
noncomputable def river_speed : ℝ := 2

theorem total_distance_traveled (h : (3.75 / (row_speed_still_water - river_speed)) + (3.75 / (row_speed_still_water + river_speed)) = 1) : 
  2 * 3.75 = 7.5 :=
by
  sorry

end total_distance_traveled_l1168_116828


namespace hexagon_side_lengths_l1168_116874

theorem hexagon_side_lengths (n m : ℕ) (AB BC : ℕ) (P : ℕ) :
  n + m = 6 ∧ n * 4 + m * 7 = 38 ∧ AB = 4 ∧ BC = 7 → m = 4 :=
by
  sorry

end hexagon_side_lengths_l1168_116874


namespace false_statement_of_quadratic_l1168_116848

-- Define the function f and the conditions
def f (a b c x : ℝ) := a * x^2 + b * x + c

theorem false_statement_of_quadratic (a b c x0 : ℝ) (h₀ : a > 0) (h₁ : 2 * a * x0 + b = 0) :
  ¬ ∀ x : ℝ, f a b c x ≤ f a b c x0 := by
  sorry

end false_statement_of_quadratic_l1168_116848


namespace truck_travel_distance_l1168_116846

theorem truck_travel_distance (b t : ℝ) (ht : t > 0) (ht30 : t + 30 > 0) : 
  let converted_feet := 4 * 60
  let time_half := converted_feet / 2
  let speed_first_half := b / 4
  let speed_second_half := b / 4
  let distance_first_half := speed_first_half * time_half / t
  let distance_second_half := speed_second_half * time_half / (t + 30)
  let total_distance_feet := distance_first_half + distance_second_half
  let result_yards := total_distance_feet / 3
  result_yards = (10 * b / t) + (10 * b / (t + 30))
:= by
  -- proof skipped
  sorry

end truck_travel_distance_l1168_116846


namespace calculate_seven_a_sq_minus_four_a_sq_l1168_116882

variable (a : ℝ)

theorem calculate_seven_a_sq_minus_four_a_sq : 7 * a^2 - 4 * a^2 = 3 * a^2 := 
by
  sorry

end calculate_seven_a_sq_minus_four_a_sq_l1168_116882
