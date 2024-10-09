import Mathlib

namespace M_intersection_N_l1130_113038

-- Definition of sets M and N
def M : Set ℝ := { x | x^2 + 2 * x - 8 < 0 }
def N : Set ℝ := { y | ∃ x : ℝ, y = 2^x }

-- Goal: Prove that M ∩ N = (0, 2)
theorem M_intersection_N :
  M ∩ N = { y | 0 < y ∧ y < 2 } :=
sorry

end M_intersection_N_l1130_113038


namespace find_vector_at_t_0_l1130_113094

def vec2 := ℝ × ℝ

def line_at_t (a d : vec2) (t : ℝ) : vec2 :=
  (a.1 + t * d.1, a.2 + t * d.2)

-- Given conditions
def vector_at_t_1 (v : vec2) : Prop :=
  v = (2, 3)

def vector_at_t_4 (v : vec2) : Prop :=
  v = (8, -5)

-- Prove that the vector at t = 0 is (0, 17/3)
theorem find_vector_at_t_0 (a d: vec2) (h1: line_at_t a d 1 = (2, 3)) (h4: line_at_t a d 4 = (8, -5)) :
  line_at_t a d 0 = (0, 17 / 3) :=
sorry

end find_vector_at_t_0_l1130_113094


namespace sixth_root_of_large_number_l1130_113053

theorem sixth_root_of_large_number : 
  ∃ (x : ℕ), x = 51 ∧ x ^ 6 = 24414062515625 :=
by
  sorry

end sixth_root_of_large_number_l1130_113053


namespace sum_of_roots_l1130_113056

theorem sum_of_roots : ∀ x : ℝ, x^2 - 2004 * x + 2021 = 0 → x = 2004 := by
  sorry

end sum_of_roots_l1130_113056


namespace sqrt_ineq_l1130_113080

open Real

theorem sqrt_ineq (α β : ℝ) (hα : 1 ≤ α) (hβ : 1 ≤ β) :
  Int.floor (sqrt α) + Int.floor (sqrt (α + β)) + Int.floor (sqrt β) ≥
    Int.floor (sqrt (2 * α)) + Int.floor (sqrt (2 * β)) := by sorry

end sqrt_ineq_l1130_113080


namespace adults_at_zoo_l1130_113032

theorem adults_at_zoo (A K : ℕ) (h1 : A + K = 254) (h2 : 28 * A + 12 * K = 3864) : A = 51 :=
sorry

end adults_at_zoo_l1130_113032


namespace cookies_eaten_is_correct_l1130_113084

-- Define initial and remaining cookies
def initial_cookies : ℕ := 7
def remaining_cookies : ℕ := 5
def cookies_eaten : ℕ := initial_cookies - remaining_cookies

-- The theorem we need to prove
theorem cookies_eaten_is_correct : cookies_eaten = 2 :=
by
  -- Here we would provide the proof
  sorry

end cookies_eaten_is_correct_l1130_113084


namespace two_digit_number_difference_perfect_square_l1130_113073

theorem two_digit_number_difference_perfect_square (N : ℕ) (a b : ℕ)
  (h1 : N = 10 * a + b)
  (h2 : N % 100 = N)
  (h3 : 1 ≤ a ∧ a ≤ 9)
  (h4 : 0 ≤ b ∧ b ≤ 9)
  (h5 : (N - (10 * b + a : ℕ)) = 64) : 
  N = 90 := 
sorry

end two_digit_number_difference_perfect_square_l1130_113073


namespace find_theta_l1130_113050

theorem find_theta (theta : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ 2 * π) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → x^3 * Real.cos θ - x^2 * (1 - x) + (1 - x)^3 * Real.sin θ > 0) →
  θ > π / 12 ∧ θ < 5 * π / 12 :=
by
  sorry

end find_theta_l1130_113050


namespace speed_of_stream_l1130_113087

variables (V_d V_u V_m V_s : ℝ)
variables (h1 : V_d = V_m + V_s) (h2 : V_u = V_m - V_s) (h3 : V_d = 18) (h4 : V_u = 6) (h5 : V_m = 12)

theorem speed_of_stream : V_s = 6 :=
by
  sorry

end speed_of_stream_l1130_113087


namespace roots_reciprocal_sum_l1130_113005

theorem roots_reciprocal_sum (x₁ x₂ : ℝ) 
    (h_roots : x₁ * x₁ + x₁ - 2 = 0 ∧ x₂ * x₂ + x₂ - 2 = 0):
    x₁ ≠ x₂ → (1 / x₁ + 1 / x₂ = 1 / 2) :=
by
  intro h_neq
  sorry

end roots_reciprocal_sum_l1130_113005


namespace altered_solution_detergent_volume_l1130_113090

theorem altered_solution_detergent_volume 
  (bleach : ℕ)
  (detergent : ℕ)
  (water : ℕ)
  (h1 : bleach / detergent = 4 / 40)
  (h2 : detergent / water = 40 / 100)
  (ratio_tripled : 3 * (bleach / detergent) = bleach / detergent)
  (ratio_halved : (detergent / water) / 2 = (detergent / water))
  (altered_water : water = 300) : 
  detergent = 60 := 
  sorry

end altered_solution_detergent_volume_l1130_113090


namespace common_element_exists_l1130_113010

theorem common_element_exists {S : Fin 2011 → Set ℤ}
  (h_nonempty : ∀ (i : Fin 2011), (S i).Nonempty)
  (h_consecutive : ∀ (i : Fin 2011), ∃ a b : ℤ, S i = Set.Icc a b)
  (h_common : ∀ (i j : Fin 2011), (S i ∩ S j).Nonempty) :
  ∃ a : ℤ, 0 < a ∧ ∀ (i : Fin 2011), a ∈ S i := sorry

end common_element_exists_l1130_113010


namespace solve_abs_quadratic_eq_and_properties_l1130_113088

theorem solve_abs_quadratic_eq_and_properties :
  ∃ x1 x2 : ℝ, (|x1|^2 + 2 * |x1| - 8 = 0) ∧ (|x2|^2 + 2 * |x2| - 8 = 0) ∧
               (x1 = 2 ∨ x1 = -2) ∧ (x2 = 2 ∨ x2 = -2) ∧
               (x1 + x2 = 0) ∧ (x1 * x2 = -4) :=
by
  sorry

end solve_abs_quadratic_eq_and_properties_l1130_113088


namespace cyclic_quadrilateral_iff_condition_l1130_113012

theorem cyclic_quadrilateral_iff_condition
  (α β γ δ : ℝ)
  (h : α + β + γ + δ = 2 * π) :
  (α * β + α * δ + γ * β + γ * δ = π^2) ↔ (α + γ = π ∧ β + δ = π) :=
by
  sorry

end cyclic_quadrilateral_iff_condition_l1130_113012


namespace sum_of_cubes_consecutive_integers_divisible_by_9_l1130_113092

theorem sum_of_cubes_consecutive_integers_divisible_by_9 (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 :=
sorry

end sum_of_cubes_consecutive_integers_divisible_by_9_l1130_113092


namespace horse_drinking_water_l1130_113069

-- Definitions and conditions

def initial_horses : ℕ := 3
def added_horses : ℕ := 5
def total_horses : ℕ := initial_horses + added_horses
def bathing_water_per_day : ℕ := 2
def total_water_28_days : ℕ := 1568
def days : ℕ := 28
def daily_water_total : ℕ := total_water_28_days / days

-- The statement looking to prove
theorem horse_drinking_water (D : ℕ) : 
  (total_horses * (D + bathing_water_per_day) = daily_water_total) → 
  D = 5 := 
by
  -- Add proof steps here
  sorry

end horse_drinking_water_l1130_113069


namespace sphere_cylinder_surface_area_difference_l1130_113001

theorem sphere_cylinder_surface_area_difference (R : ℝ) :
  let S_sphere := 4 * Real.pi * R^2
  let S_lateral := 4 * Real.pi * R^2
  S_sphere - S_lateral = 0 :=
by
  sorry

end sphere_cylinder_surface_area_difference_l1130_113001


namespace cubed_identity_l1130_113060

theorem cubed_identity (x : ℝ) (h : x + 1/x = 7) : x^3 + 1/x^3 = 322 := 
sorry

end cubed_identity_l1130_113060


namespace divisor_of_a_l1130_113071

namespace MathProofProblem

-- Define the given problem
variable (a b c d : ℕ) -- Variables representing positive integers

-- Given conditions
variables (h_gcd_ab : Nat.gcd a b = 30)
variables (h_gcd_bc : Nat.gcd b c = 42)
variables (h_gcd_cd : Nat.gcd c d = 66)
variables (h_lcm_cd : Nat.lcm c d = 2772)
variables (h_gcd_da : 100 < Nat.gcd d a ∧ Nat.gcd d a < 150)

-- Target statement to prove
theorem divisor_of_a : 13 ∣ a :=
by
  sorry

end MathProofProblem

end divisor_of_a_l1130_113071


namespace number_of_chocolate_bars_by_theresa_l1130_113013

-- Define the number of chocolate bars and soda cans that Kayla bought
variables (C S : ℕ)

-- Assume the total number of chocolate bars and soda cans Kayla bought is 15
axiom total_purchased_by_kayla : C + S = 15

-- Define the number of chocolate bars Theresa bought as twice the number Kayla bought
def chocolate_bars_purchased_by_theresa := 2 * C

-- The theorem to prove
theorem number_of_chocolate_bars_by_theresa : chocolate_bars_purchased_by_theresa = 2 * C :=
by
  -- The proof is omitted as instructed
  sorry

end number_of_chocolate_bars_by_theresa_l1130_113013


namespace height_of_boxes_l1130_113025

theorem height_of_boxes
  (volume_required : ℝ)
  (price_per_box : ℝ)
  (min_expenditure : ℝ)
  (volume_per_box : ∀ n : ℕ, n = min_expenditure / price_per_box -> ℝ) :
  volume_required = 3060000 ->
  price_per_box = 0.50 ->
  min_expenditure = 255 ->
  ∃ h : ℝ, h = 19 := by
  sorry

end height_of_boxes_l1130_113025


namespace curtain_additional_material_l1130_113089

theorem curtain_additional_material
  (room_height_feet : ℕ)
  (curtain_length_inches : ℕ)
  (height_conversion_factor : ℕ)
  (desired_length : ℕ)
  (h_room_height_conversion : room_height_feet * height_conversion_factor = 96)
  (h_desired_length : desired_length = 101) :
  curtain_length_inches = desired_length - (room_height_feet * height_conversion_factor) :=
by
  sorry

end curtain_additional_material_l1130_113089


namespace renovation_services_are_credence_goods_and_choice_arguments_l1130_113017

-- Define what credence goods are and the concept of information asymmetry
structure CredenceGood where
  information_asymmetry : Prop
  unobservable_quality  : Prop

-- Define renovation service as an instance of CredenceGood
def RenovationService : CredenceGood := {
  information_asymmetry := true,
  unobservable_quality := true
}

-- Primary conditions for choosing between construction company and private repair crew
structure ChoiceArgument where
  information_availability     : Prop
  warranty_and_accountability  : Prop
  higher_costs                 : Prop
  potential_bias_in_reviews    : Prop

-- Arguments for using construction company
def ConstructionCompanyArguments : ChoiceArgument := {
  information_availability := true,
  warranty_and_accountability := true,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Arguments against using construction company
def PrivateRepairCrewArguments : ChoiceArgument := {
  information_availability := false,
  warranty_and_accountability := false,
  higher_costs := true,
  potential_bias_in_reviews := true
}

-- Proof statement to show renovation services are credence goods and economically reasoned arguments for/against
theorem renovation_services_are_credence_goods_and_choice_arguments:
  RenovationService = {
    information_asymmetry := true,
    unobservable_quality := true
  } ∧
  (ConstructionCompanyArguments.information_availability = true ∧
   ConstructionCompanyArguments.warranty_and_accountability = true) ∧
  (ConstructionCompanyArguments.higher_costs = true ∧
   ConstructionCompanyArguments.potential_bias_in_reviews = true) ∧
  (PrivateRepairCrewArguments.higher_costs = true ∧
   PrivateRepairCrewArguments.potential_bias_in_reviews = true) :=
by sorry

end renovation_services_are_credence_goods_and_choice_arguments_l1130_113017


namespace RahulPlayedMatchesSolver_l1130_113037

noncomputable def RahulPlayedMatches (current_average new_average runs_in_today current_matches : ℕ) : ℕ :=
  let total_runs_before := current_average * current_matches
  let total_runs_after := total_runs_before + runs_in_today
  let total_matches_after := current_matches + 1
  total_runs_after / new_average

theorem RahulPlayedMatchesSolver:
  RahulPlayedMatches 52 54 78 12 = 12 :=
by
  sorry

end RahulPlayedMatchesSolver_l1130_113037


namespace meeting_point_distance_l1130_113009

theorem meeting_point_distance
  (distance_to_top : ℝ)
  (total_distance : ℝ)
  (jack_start_time : ℝ)
  (jack_uphill_speed : ℝ)
  (jack_downhill_speed : ℝ)
  (jill_uphill_speed : ℝ)
  (jill_downhill_speed : ℝ)
  (meeting_point_distance : ℝ):
  distance_to_top = 5 -> total_distance = 10 -> jack_start_time = 10 / 60 ->
  jack_uphill_speed = 15 -> jack_downhill_speed = 20 ->
  jill_uphill_speed = 16 -> jill_downhill_speed = 22 ->
  meeting_point_distance = 35 / 27 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end meeting_point_distance_l1130_113009


namespace like_terms_satisfy_conditions_l1130_113007

theorem like_terms_satisfy_conditions (m n : ℤ) (h1 : m - 1 = n) (h2 : m + n = 3) :
  m = 2 ∧ n = 1 := by
  sorry

end like_terms_satisfy_conditions_l1130_113007


namespace intersection_M_N_l1130_113033

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = {0, 1} := 
by
  sorry

end intersection_M_N_l1130_113033


namespace units_digit_2749_987_l1130_113077

def mod_units_digit (base : ℕ) (exp : ℕ) : ℕ :=
  (base % 10)^(exp % 2) % 10

theorem units_digit_2749_987 : mod_units_digit 2749 987 = 9 := 
by 
  sorry

end units_digit_2749_987_l1130_113077


namespace quadratic_roots_real_and_values_l1130_113098

theorem quadratic_roots_real_and_values (m : ℝ) (x : ℝ) :
  (x ^ 2 - x + 2 * m - 2 = 0) → (m ≤ 9 / 8) ∧ (m = 1 → (x = 0 ∨ x = 1)) :=
by
  sorry

end quadratic_roots_real_and_values_l1130_113098


namespace final_price_correct_l1130_113011

variable (original_price first_discount second_discount third_discount sales_tax : ℝ)
variable (final_discounted_price final_price: ℝ)

-- Define original price and discounts
def initial_price : ℝ := 20000
def discount1      : ℝ := 0.12
def discount2      : ℝ := 0.10
def discount3      : ℝ := 0.05
def tax_rate       : ℝ := 0.08

def price_after_first_discount : ℝ := initial_price * (1 - discount1)
def price_after_second_discount : ℝ := price_after_first_discount * (1 - discount2)
def price_after_third_discount : ℝ := price_after_second_discount * (1 - discount3)
def final_sale_price : ℝ := price_after_third_discount * (1 + tax_rate)

-- Prove final sale price is 16251.84
theorem final_price_correct : final_sale_price = 16251.84 := by
  sorry

end final_price_correct_l1130_113011


namespace median_number_of_children_l1130_113061

-- Define the given conditions
def number_of_data_points : Nat := 13
def median_position : Nat := (number_of_data_points + 1) / 2

-- We assert the median value based on information given in the problem
def median_value : Nat := 4

-- Statement to prove the problem
theorem median_number_of_children (h1: median_position = 7) (h2: median_value = 4) : median_value = 4 := 
by
  sorry

end median_number_of_children_l1130_113061


namespace exp_product_correct_l1130_113028

def exp_1 := (2 : ℕ) ^ 4
def exp_2 := (3 : ℕ) ^ 2
def exp_3 := (5 : ℕ) ^ 2
def exp_4 := (7 : ℕ)
def exp_5 := (11 : ℕ)
def final_value := exp_1 * exp_2 * exp_3 * exp_4 * exp_5

theorem exp_product_correct : final_value = 277200 := by
  sorry

end exp_product_correct_l1130_113028


namespace general_term_formula_T_n_less_than_one_sixth_l1130_113082

noncomputable def S (n : ℕ) : ℕ := n^2 + 2*n

def a (n : ℕ) : ℕ := if n = 0 then 0 else 2*n + 1

def b (n : ℕ) : ℕ := if n = 0 then 0 else 1 / (a n) * (a (n+1))

def T (n : ℕ) : ℝ := (Finset.range n).sum (λ k => (b k : ℝ))

theorem general_term_formula (n : ℕ) (hn : n ≠ 0) : 
  a n = 2*n + 1 :=
by sorry

theorem T_n_less_than_one_sixth (n : ℕ) : 
  T n < (1 / 6 : ℝ) :=
by sorry

end general_term_formula_T_n_less_than_one_sixth_l1130_113082


namespace avg_meal_cost_per_individual_is_72_l1130_113059

theorem avg_meal_cost_per_individual_is_72
  (total_bill : ℝ)
  (gratuity_percent : ℝ)
  (num_investment_bankers num_clients : ℕ)
  (total_individuals := num_investment_bankers + num_clients)
  (meal_cost_before_gratuity : ℝ := total_bill / (1 + gratuity_percent))
  (average_cost := meal_cost_before_gratuity / total_individuals) :
  total_bill = 1350 ∧ gratuity_percent = 0.25 ∧ num_investment_bankers = 7 ∧ num_clients = 8 →
  average_cost = 72 := by
  sorry

end avg_meal_cost_per_individual_is_72_l1130_113059


namespace problem1_xy_value_problem2_min_value_l1130_113067

-- Define the first problem conditions
def problem1 (x y : ℝ) : Prop :=
  x^2 - 2 * x * y + 2 * y^2 + 6 * y + 9 = 0

-- Prove that xy = 9 given the above condition
theorem problem1_xy_value (x y : ℝ) (h : problem1 x y) : x * y = 9 :=
  sorry

-- Define the second problem conditions
def expression (m : ℝ) : ℝ :=
  m^2 + 6 * m + 13

-- Prove that the minimum value of the expression is 4
theorem problem2_min_value : ∃ m, expression m = 4 :=
  sorry

end problem1_xy_value_problem2_min_value_l1130_113067


namespace compare_m_n_l1130_113003

noncomputable def m (a : ℝ) : ℝ := 6^a / (36^(a + 1) + 1)
noncomputable def n (b : ℝ) : ℝ := (1/3) * b^2 - b + (5/6)

theorem compare_m_n (a b : ℝ) : m a ≤ n b := sorry

end compare_m_n_l1130_113003


namespace num_envelopes_requiring_charge_l1130_113048

structure Envelope where
  length : ℕ
  height : ℕ

def requiresExtraCharge (env : Envelope) : Bool :=
  let ratio := env.length / env.height
  ratio < 3/2 ∨ ratio > 3

def envelopes : List Envelope :=
  [{ length := 7, height := 5 },  -- E
   { length := 10, height := 2 }, -- F
   { length := 8, height := 8 },  -- G
   { length := 12, height := 3 }] -- H

def countExtraChargedEnvelopes : ℕ :=
  envelopes.filter requiresExtraCharge |>.length

theorem num_envelopes_requiring_charge : countExtraChargedEnvelopes = 4 := by
  sorry

end num_envelopes_requiring_charge_l1130_113048


namespace voldemort_lunch_calories_l1130_113093

def dinner_cake_calories : Nat := 110
def chips_calories : Nat := 310
def coke_calories : Nat := 215
def breakfast_calories : Nat := 560
def daily_intake_limit : Nat := 2500
def remaining_calories : Nat := 525

def total_dinner_snacks_breakfast : Nat :=
  dinner_cake_calories + chips_calories + coke_calories + breakfast_calories

def total_remaining_allowance : Nat :=
  total_dinner_snacks_breakfast + remaining_calories

def lunch_calories : Nat :=
  daily_intake_limit - total_remaining_allowance

theorem voldemort_lunch_calories:
  lunch_calories = 780 := by
  sorry

end voldemort_lunch_calories_l1130_113093


namespace largest_integer_le_1_l1130_113066

theorem largest_integer_le_1 (x : ℤ) (h : (2 * x : ℚ) / 7 + 3 / 4 < 8 / 7) : x ≤ 1 :=
sorry

end largest_integer_le_1_l1130_113066


namespace boat_cost_per_foot_l1130_113086

theorem boat_cost_per_foot (total_savings : ℝ) (license_cost : ℝ) (docking_fee_multiplier : ℝ) (max_boat_length : ℝ) 
  (h1 : total_savings = 20000) 
  (h2 : license_cost = 500) 
  (h3 : docking_fee_multiplier = 3) 
  (h4 : max_boat_length = 12) 
  : (total_savings - (license_cost + docking_fee_multiplier * license_cost)) / max_boat_length = 1500 :=
by
  sorry

end boat_cost_per_foot_l1130_113086


namespace pumpkin_count_sunshine_orchard_l1130_113079

def y (x : ℕ) : ℕ := 3 * x^2 + 12

theorem pumpkin_count_sunshine_orchard :
  y 14 = 600 :=
by
  sorry

end pumpkin_count_sunshine_orchard_l1130_113079


namespace integer_ratio_l1130_113058

theorem integer_ratio (A B C D : ℕ) (h1 : (A + B + C + D) / 4 = 16)
  (h2 : A % B = 0) (h3 : B = C - 2) (h4 : D = 2) (h5 : A ≠ B) (h6 : B ≠ C) (h7 : C ≠ D) (h8 : D ≠ A)
  (h9: 0 < A) (h10: 0 < B) (h11: 0 < C):
  A / B = 28 := 
sorry

end integer_ratio_l1130_113058


namespace blowfish_stayed_own_tank_l1130_113091

def number_clownfish : ℕ := 50
def number_blowfish : ℕ := 50
def number_clownfish_display_initial : ℕ := 24
def number_clownfish_display_final : ℕ := 16

theorem blowfish_stayed_own_tank : 
    (number_clownfish + number_blowfish = 100) ∧ 
    (number_clownfish = number_blowfish) ∧ 
    (number_clownfish_display_final = 2 / 3 * number_clownfish_display_initial) →
    ∀ (blowfish : ℕ), 
    blowfish = number_blowfish - number_clownfish_display_initial → 
    blowfish = 26 :=
sorry

end blowfish_stayed_own_tank_l1130_113091


namespace find_a9_l1130_113018

theorem find_a9 (a : ℕ → ℕ) 
  (h_add : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
  (h_a2 : a 2 = 4) 
  : a 9 = 18 :=
sorry

end find_a9_l1130_113018


namespace sum_four_digit_even_numbers_l1130_113031

-- Define the digits set
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

-- Define the set of valid units digits for even numbers
def even_units : Finset ℕ := {0, 2, 4}

-- Define the set of all four-digit numbers using the provided digits
def four_digit_even_numbers : Finset ℕ :=
  (Finset.range (10000) \ Finset.range (1000)).filter (λ n =>
    n % 10 ∈ even_units ∧
    (n / 1000) ∈ digits ∧
    ((n / 100) % 10) ∈ digits ∧
    ((n / 10) % 10) ∈ digits)

theorem sum_four_digit_even_numbers :
  (four_digit_even_numbers.sum (λ x => x)) = 1769580 :=
  sorry

end sum_four_digit_even_numbers_l1130_113031


namespace hadassah_painting_time_l1130_113039

noncomputable def time_to_paint_all_paintings (time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings : ℝ) : ℝ :=
  time_small_paintings + time_large_paintings + time_additional_small_paintings + time_additional_large_paintings

theorem hadassah_painting_time :
  let time_small_paintings := 6
  let time_large_paintings := 8
  let time_per_small_painting := 6 / 12 -- = 0.5
  let time_per_large_painting := 8 / 6 -- ≈ 1.33
  let time_additional_small_paintings := 15 * time_per_small_painting -- = 7.5
  let time_additional_large_paintings := 10 * time_per_large_painting -- ≈ 13.3
  time_to_paint_all_paintings time_small_paintings time_large_paintings time_additional_small_paintings time_additional_large_paintings = 34.8 :=
by
  sorry

end hadassah_painting_time_l1130_113039


namespace impossible_divide_into_three_similar_l1130_113074

noncomputable def sqrt2 : ℝ := Real.sqrt 2

def similar (x y : ℝ) : Prop :=
  x ≤ sqrt2 * y

theorem impossible_divide_into_three_similar (N : ℝ) :
  ¬ ∃ (x y z : ℝ), x + y + z = N ∧ similar x y ∧ similar y z ∧ similar x z := 
by
  sorry

end impossible_divide_into_three_similar_l1130_113074


namespace g_eq_l1130_113044

noncomputable def g (n : ℕ) : ℝ :=
  (7 + 4 * Real.sqrt 7) / 14 * ((1 + Real.sqrt 7) / 2) ^ n +
  (7 - 4 * Real.sqrt 7) / 14 * ((1 - Real.sqrt 7) / 2) ^ n

theorem g_eq (n : ℕ) : g (n + 2) - g (n - 2) = 3 * g n := by
  sorry

end g_eq_l1130_113044


namespace positive_solutions_count_l1130_113020

theorem positive_solutions_count :
  ∃ n : ℕ, n = 9 ∧
  (∀ (x y : ℕ), 5 * x + 10 * y = 100 → 0 < x ∧ 0 < y → (∃ k : ℕ, k < 10 ∧ n = 9)) :=
sorry

end positive_solutions_count_l1130_113020


namespace length_of_floor_is_10_l1130_113099

variable (L : ℝ) -- Declare the variable representing the length of the floor

-- Conditions as definitions
def width_of_floor := 8
def strip_width := 2
def area_of_rug := 24
def rug_length := L - 2 * strip_width
def rug_width := width_of_floor - 2 * strip_width

-- Math proof problem statement
theorem length_of_floor_is_10
  (h1 : rug_length * rug_width = area_of_rug)
  (h2 : width_of_floor = 8)
  (h3 : strip_width = 2) :
  L = 10 :=
by
  -- Placeholder for the actual proof
  sorry

end length_of_floor_is_10_l1130_113099


namespace number_of_schools_l1130_113045

-- Define the conditions
def is_median (a : ℕ) (n : ℕ) : Prop := 2 * a - 1 = n
def high_team_score (a b c : ℕ) : Prop := a > b ∧ a > c
def ranks (b c : ℕ) : Prop := b = 39 ∧ c = 67

-- Define the main problem
theorem number_of_schools (a n b c : ℕ) :
  is_median a n →
  high_team_score a b c →
  ranks b c →
  34 ≤ a ∧ a < 39 →
  2 * a ≡ 1 [MOD 3] →
  (n = 67 → a = 35) →
  (∀ m : ℕ, n = 3 * m + 1) →
  m = 23 :=
by
  sorry

end number_of_schools_l1130_113045


namespace yuna_survey_l1130_113051

theorem yuna_survey :
  let M := 27
  let K := 28
  let B := 22
  M + K - B = 33 :=
by
  sorry

end yuna_survey_l1130_113051


namespace power_function_through_point_l1130_113085

noncomputable def f (x k α : ℝ) : ℝ := k * x ^ α

theorem power_function_through_point (k α : ℝ) (h : f (1/2) k α = Real.sqrt 2) : 
  k + α = 1/2 := 
by 
  sorry

end power_function_through_point_l1130_113085


namespace zoe_earns_per_candy_bar_l1130_113042

-- Given conditions
def cost_of_trip : ℝ := 485
def grandma_contribution : ℝ := 250
def candy_bars_to_sell : ℝ := 188

-- Derived condition
def additional_amount_needed : ℝ := cost_of_trip - grandma_contribution

-- Assertion to prove
theorem zoe_earns_per_candy_bar :
  (additional_amount_needed / candy_bars_to_sell) = 1.25 :=
by
  sorry

end zoe_earns_per_candy_bar_l1130_113042


namespace plane_equation_l1130_113095

variable (x y z : ℝ)

def line1 := 3 * x - 2 * y + 5 * z + 3 = 0
def line2 := x + 2 * y - 3 * z - 11 = 0
def origin_plane := 18 * x - 8 * y + 23 * z = 0

theorem plane_equation : 
  (∀ x y z, line1 x y z → line2 x y z → origin_plane x y z) :=
by
  sorry

end plane_equation_l1130_113095


namespace no_possible_stack_of_1997_sum_l1130_113043

theorem no_possible_stack_of_1997_sum :
  ¬ ∃ k : ℕ, 6 * k = 3 * 1997 := by
  sorry

end no_possible_stack_of_1997_sum_l1130_113043


namespace boat_speed_in_still_water_l1130_113021

-- Definitions for conditions
variables (V_b V_s : ℝ)

-- The conditions provided for the problem
def along_stream := V_b + V_s = 13
def against_stream := V_b - V_s = 5

-- The theorem we want to prove
theorem boat_speed_in_still_water (h1 : along_stream V_b V_s) (h2 : against_stream V_b V_s) : V_b = 9 :=
sorry

end boat_speed_in_still_water_l1130_113021


namespace fido_yard_area_fraction_l1130_113046

theorem fido_yard_area_fraction (r : ℝ) (h : r > 0) :
  let square_area := (2 * r)^2
  let reachable_area := π * r^2
  let fraction := reachable_area / square_area
  ∃ a b : ℕ, (fraction = (Real.sqrt a) / b * π) ∧ (a * b = 4) := by
  sorry

end fido_yard_area_fraction_l1130_113046


namespace find_a_l1130_113083

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (3 - a) * x - a else Real.log x / Real.log a

theorem find_a (a : ℝ) (h : f a (f a 1) = 2) : a = -2 := by
  sorry

end find_a_l1130_113083


namespace pens_more_than_notebooks_l1130_113002

theorem pens_more_than_notebooks
  (N P : ℕ) 
  (h₁ : N = 30) 
  (h₂ : N + P = 110) :
  P - N = 50 := 
by
  sorry

end pens_more_than_notebooks_l1130_113002


namespace tom_total_calories_l1130_113030

-- Define the conditions
def c_weight : ℕ := 1
def c_calories_per_pound : ℕ := 51
def b_weight : ℕ := 2 * c_weight
def b_calories_per_pound : ℕ := c_calories_per_pound / 3

-- Define the total calories
def total_calories : ℕ := (c_weight * c_calories_per_pound) + (b_weight * b_calories_per_pound)

-- Prove the total calories Tom eats
theorem tom_total_calories : total_calories = 85 := by
  sorry

end tom_total_calories_l1130_113030


namespace elongation_rate_significantly_improved_l1130_113049

noncomputable def elongation_improvement : Prop :=
  let x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
  let y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
  let z := List.zipWith (λ xi yi => xi - yi) x y
  let n : ℝ := 10
  let mean_z := (List.sum z) / n
  let variance_z := (List.sum (List.map (λ zi => (zi - mean_z)^2) z)) / n
  mean_z = 11 ∧ 
  variance_z = 61 ∧ 
  mean_z ≥ 2 * Real.sqrt (variance_z / n)

-- We state the theorem without proof
theorem elongation_rate_significantly_improved : elongation_improvement :=
by
  -- Proof can be written here
  sorry

end elongation_rate_significantly_improved_l1130_113049


namespace cereal_original_price_l1130_113034

-- Define the known conditions as constants
def initial_money : ℕ := 60
def celery_price : ℕ := 5
def bread_price : ℕ := 8
def milk_full_price : ℕ := 10
def milk_discount : ℕ := 10
def milk_price : ℕ := milk_full_price - (milk_full_price * milk_discount / 100)
def potato_price : ℕ := 1
def potato_quantity : ℕ := 6
def potatoes_total_price : ℕ := potato_price * potato_quantity
def coffee_remaining_money : ℕ := 26
def total_spent_exclude_coffee : ℕ := initial_money - coffee_remaining_money
def spent_on_other_items : ℕ := celery_price + bread_price + milk_price + potatoes_total_price
def spent_on_cereal : ℕ := total_spent_exclude_coffee - spent_on_other_items
def cereal_discount : ℕ := 50

theorem cereal_original_price :
  (spent_on_other_items = celery_price + bread_price + milk_price + potatoes_total_price) →
  (total_spent_exclude_coffee = initial_money - coffee_remaining_money) →
  (spent_on_cereal = total_spent_exclude_coffee - spent_on_other_items) →
  (spent_on_cereal * 2 = 12) :=
by {
  -- proof here
  sorry
}

end cereal_original_price_l1130_113034


namespace comparison_M_N_l1130_113014

def M (x : ℝ) : ℝ := x^2 - 3*x + 7
def N (x : ℝ) : ℝ := -x^2 + x + 1

theorem comparison_M_N (x : ℝ) : M x > N x :=
  by sorry

end comparison_M_N_l1130_113014


namespace original_acid_concentration_l1130_113008

theorem original_acid_concentration (P : ℝ) (h1 : 0.5 * P + 0.5 * 20 = 35) : P = 50 :=
by
  sorry

end original_acid_concentration_l1130_113008


namespace sum_of_reciprocals_of_roots_l1130_113072

theorem sum_of_reciprocals_of_roots (p q r : ℝ) (h : ∀ x : ℝ, (x^3 - x - 6 = 0) → (x = p ∨ x = q ∨ x = r)) :
  1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 11 / 12 :=
sorry

end sum_of_reciprocals_of_roots_l1130_113072


namespace cheapest_candle_cost_to_measure_1_minute_l1130_113040

-- Definitions

def big_candle_cost := 16 -- cost of a big candle in cents
def big_candle_burn_time := 16 -- burn time of a big candle in minutes
def small_candle_cost := 7 -- cost of a small candle in cents
def small_candle_burn_time := 7 -- burn time of a small candle in minutes

-- Problem statement
theorem cheapest_candle_cost_to_measure_1_minute :
  (∃ (n m : ℕ), n * big_candle_burn_time - m * small_candle_burn_time = 1 ∧
                 n * big_candle_cost + m * small_candle_cost = 97) :=
sorry

end cheapest_candle_cost_to_measure_1_minute_l1130_113040


namespace gcd_105_490_l1130_113057

theorem gcd_105_490 : Nat.gcd 105 490 = 35 := by
sorry

end gcd_105_490_l1130_113057


namespace asymptote_equation_of_hyperbola_l1130_113026

def hyperbola_eccentricity (a : ℝ) (h : a > 0) : Prop :=
  let e := Real.sqrt 2
  e = Real.sqrt (1 + a^2) / a

theorem asymptote_equation_of_hyperbola :
  ∀ (a : ℝ) (h : a > 0), hyperbola_eccentricity a h → (∀ x y : ℝ, (x^2 - y^2 = 1 → y = x ∨ y = -x)) :=
by
  intro a h he
  sorry

end asymptote_equation_of_hyperbola_l1130_113026


namespace sequence_geq_four_l1130_113041

theorem sequence_geq_four (a : ℕ → ℝ) (h0 : a 1 = 5) 
    (h1 : ∀ n ≥ 1, a (n+1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)) : 
    ∀ n ≥ 1, a n ≥ 4 := 
by
  sorry

end sequence_geq_four_l1130_113041


namespace electronics_weight_l1130_113000

theorem electronics_weight (B C E : ℝ) (h1 : B / C = 5 / 4) (h2 : B / E = 5 / 2) (h3 : B / (C - 9) = 10 / 4) : E = 9 := 
by 
  sorry

end electronics_weight_l1130_113000


namespace angle_subtraction_correct_polynomial_simplification_correct_l1130_113015

noncomputable def angleSubtraction : Prop :=
  let a1 := 34 * 60 + 26 -- Convert 34°26' to total minutes
  let a2 := 25 * 60 + 33 -- Convert 25°33' to total minutes
  let diff := a1 - a2 -- Subtract in minutes
  let degrees := diff / 60 -- Convert back to degrees
  let minutes := diff % 60 -- Remainder in minutes
  degrees = 8 ∧ minutes = 53 -- Expected result in degrees and minutes

noncomputable def polynomialSimplification (m : Int) : Prop :=
  let expr := 5 * m^2 - (m^2 - 6 * m) - 2 * (-m + 3 * m^2)
  expr = -2 * m^2 + 8 * m -- Simplified form

-- Statements needing proof
theorem angle_subtraction_correct : angleSubtraction := by
  sorry

theorem polynomial_simplification_correct (m : Int) : polynomialSimplification m := by
  sorry

end angle_subtraction_correct_polynomial_simplification_correct_l1130_113015


namespace boys_neither_happy_nor_sad_l1130_113006

theorem boys_neither_happy_nor_sad (total_children : ℕ)
  (happy_children sad_children neither_happy_nor_sad total_boys total_girls : ℕ)
  (happy_boys sad_girls : ℕ)
  (h_total : total_children = 60)
  (h_happy : happy_children = 30)
  (h_sad : sad_children = 10)
  (h_neither : neither_happy_nor_sad = 20)
  (h_boys : total_boys = 17)
  (h_girls : total_girls = 43)
  (h_happy_boys : happy_boys = 6)
  (h_sad_girls : sad_girls = 4) :
  ∃ (boys_neither_happy_nor_sad : ℕ), boys_neither_happy_nor_sad = 5 := by
  sorry

end boys_neither_happy_nor_sad_l1130_113006


namespace product_evaluation_l1130_113036

theorem product_evaluation :
  (6 * 27^12 + 2 * 81^9) / 8000000^2 * (80 * 32^3 * 125^4) / (9^19 - 729^6) = 10 :=
by sorry

end product_evaluation_l1130_113036


namespace coeff_sum_eq_32_l1130_113047

theorem coeff_sum_eq_32 (n : ℕ) (h : (2 : ℕ)^n = 32) : n = 5 :=
sorry

end coeff_sum_eq_32_l1130_113047


namespace bus_departure_interval_l1130_113097

theorem bus_departure_interval
  (v : ℝ) -- speed of B (per minute)
  (t_A : ℝ := 10) -- A is overtaken every 10 minutes
  (t_B : ℝ := 6) -- B is overtaken every 6 minutes
  (v_A : ℝ := 3 * v) -- speed of A
  (d_A : ℝ := v_A * t_A) -- distance covered by A in 10 minutes
  (d_B : ℝ := v * t_B) -- distance covered by B in 6 minutes
  (v_bus_minus_vA : ℝ := d_A / t_A) -- bus speed relative to A
  (v_bus_minus_vB : ℝ := d_B / t_B) -- bus speed relative to B) :
  (t : ℝ) -- time interval between bus departures
  : t = 5 := sorry

end bus_departure_interval_l1130_113097


namespace length_of_train_correct_l1130_113029

noncomputable def length_of_train (time_pass_man : ℝ) (train_speed_kmh : ℝ) (man_speed_kmh : ℝ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh - man_speed_kmh
  let relative_speed_ms := (relative_speed_kmh * 1000) / 3600
  relative_speed_ms * time_pass_man

theorem length_of_train_correct :
  length_of_train 29.997600191984642 60 6 = 449.96400287976963 := by
  sorry

end length_of_train_correct_l1130_113029


namespace most_followers_after_three_weeks_l1130_113027

def initial_followers_susy := 100
def initial_followers_sarah := 50
def first_week_gain_susy := 40
def second_week_gain_susy := first_week_gain_susy / 2
def third_week_gain_susy := second_week_gain_susy / 2
def first_week_gain_sarah := 90
def second_week_gain_sarah := first_week_gain_sarah / 3
def third_week_gain_sarah := second_week_gain_sarah / 3

def total_followers_susy := initial_followers_susy + first_week_gain_susy + second_week_gain_susy + third_week_gain_susy
def total_followers_sarah := initial_followers_sarah + first_week_gain_sarah + second_week_gain_sarah + third_week_gain_sarah

theorem most_followers_after_three_weeks : max total_followers_susy total_followers_sarah = 180 :=
by
  sorry

end most_followers_after_three_weeks_l1130_113027


namespace rhombus_area_is_correct_l1130_113055

def calculate_rhombus_area (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

theorem rhombus_area_is_correct :
  calculate_rhombus_area (3 * 6) (3 * 4) = 108 := by
  sorry

end rhombus_area_is_correct_l1130_113055


namespace bases_for_final_digit_one_l1130_113016

noncomputable def numberOfBases : ℕ :=
  (Finset.filter (λ b => ((625 - 1) % b = 0)) (Finset.range 11)).card - 
  (Finset.filter (λ b => b ≤ 2) (Finset.range 11)).card

theorem bases_for_final_digit_one : numberOfBases = 4 :=
by sorry

end bases_for_final_digit_one_l1130_113016


namespace simple_interest_is_correct_l1130_113065

-- Define the principal amount, rate of interest, and time
def P : ℕ := 400
def R : ℚ := 22.5
def T : ℕ := 2

-- Define the formula for simple interest
def simple_interest (P : ℕ) (R : ℚ) (T : ℕ) : ℚ :=
  (P * R * T) / 100

-- The statement we need to prove
theorem simple_interest_is_correct : simple_interest P R T = 90 :=
by
  sorry

end simple_interest_is_correct_l1130_113065


namespace point_B_number_l1130_113068

theorem point_B_number (A B : ℤ) (hA : A = -2) (hB : abs (B - A) = 3) : B = 1 ∨ B = -5 :=
sorry

end point_B_number_l1130_113068


namespace stratified_sampling_l1130_113064

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem stratified_sampling :
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  combination junior_students junior_sample_size * combination senior_students senior_sample_size =
    combination 400 40 * combination 200 20 :=
by
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  exact sorry

end stratified_sampling_l1130_113064


namespace problem1_problem2_l1130_113024

-- Definitions for sets A and S
def setA (x : ℝ) : Prop := -7 ≤ 2 * x - 5 ∧ 2 * x - 5 ≤ 9
def setS (x k : ℝ) : Prop := k + 1 ≤ x ∧ x ≤ 2 * k - 1

-- Preliminary ranges for x
lemma range_A : ∀ x, setA x ↔ -1 ≤ x ∧ x ≤ 7 := sorry

noncomputable def k_range1 (k : ℝ) : Prop := 2 ≤ k ∧ k ≤ 4
noncomputable def k_range2 (k : ℝ) : Prop := k < 2 ∨ k > 6

-- Proof problems in Lean 4

-- First problem statement
theorem problem1 (k : ℝ) : (∀ x, setS x k → setA x) ∧ (∃ x, setS x k) → k_range1 k := sorry

-- Second problem statement
theorem problem2 (k : ℝ) : (∀ x, ¬(setA x ∧ setS x k)) → k_range2 k := sorry

end problem1_problem2_l1130_113024


namespace min_value_a_decreasing_range_of_a_x1_x2_l1130_113035

noncomputable def f (a x : ℝ) := x / Real.log x - a * x

theorem min_value_a_decreasing :
  ∀ (a : ℝ), (∀ (x : ℝ), 1 < x → f a x <= 0) → a ≥ 1 / 4 :=
sorry

theorem range_of_a_x1_x2 :
  ∀ (a : ℝ), (∃ (x₁ x₂ : ℝ), e ≤ x₁ ∧ x₁ ≤ e^2 ∧ e ≤ x₂ ∧ x₂ ≤ e^2 ∧ f a x₁ ≤ f a x₂ + a)
  → a ≥ 1 / 2 - 1 / (4 * e^2) :=
sorry

end min_value_a_decreasing_range_of_a_x1_x2_l1130_113035


namespace no_repetition_five_digit_count_l1130_113023

theorem no_repetition_five_digit_count (digits : Finset ℕ) (count : Nat) :
  digits = {0, 1, 2, 3, 4, 5} →
  (∀ n ∈ digits, 0 ≤ n ∧ n ≤ 5) →
  (∃ numbers : Finset ℕ, 
    (∀ x ∈ numbers, (x / 100) % 10 ≠ 3 ∧ x % 5 = 0 ∧ x < 100000 ∧ x ≥ 10000) ∧
    (numbers.card = count)) →
  count = 174 :=
by
  sorry

end no_repetition_five_digit_count_l1130_113023


namespace second_parentheses_expression_eq_zero_l1130_113022

def custom_op (x y : ℝ) : ℝ := (x + y)^2 - (x - y)^2

theorem second_parentheses_expression_eq_zero :
  custom_op (Real.sqrt 6) (Real.sqrt 6) = 0 := by
  sorry

end second_parentheses_expression_eq_zero_l1130_113022


namespace isosceles_triangle_possible_values_of_x_l1130_113075

open Real

-- Define the main statement
theorem isosceles_triangle_possible_values_of_x :
  ∀ x : ℝ, 
  (0 < x ∧ x < 90) ∧ 
  (sin (3*x) = sin (2*x) ∧ 
   sin (9*x) = sin (2*x)) 
  → x = 0 ∨ x = 180/11 ∨ x = 540/11 :=
by
  sorry

end isosceles_triangle_possible_values_of_x_l1130_113075


namespace tan_alpha_frac_l1130_113063

theorem tan_alpha_frac (α : ℝ) (h : Real.tan α = 2) : (Real.sin α - Real.cos α) / (3 * Real.sin α + 5 * Real.cos α) = 1 / 11 := by
  sorry

end tan_alpha_frac_l1130_113063


namespace ratio_of_squares_l1130_113096

theorem ratio_of_squares : (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3 / 10 := 
sorry

end ratio_of_squares_l1130_113096


namespace range_of_b_l1130_113054

noncomputable def f (x a b : ℝ) : ℝ :=
  x + a / x + b

theorem range_of_b (b : ℝ) :
  (∀ (a x : ℝ), (1/2 ≤ a ∧ a ≤ 2) ∧ (1/4 ≤ x ∧ x ≤ 1) → f x a b ≤ 10) →
  b ≤ 7 / 4 :=
by
  sorry

end range_of_b_l1130_113054


namespace probability_same_color_l1130_113004

-- Define the total number of plates
def totalPlates : ℕ := 6 + 5 + 3

-- Define the number of red plates, blue plates, and green plates
def redPlates : ℕ := 6
def bluePlates : ℕ := 5
def greenPlates : ℕ := 3

-- Define the total number of ways to choose 3 plates from 14
def totalWaysChoose3 : ℕ := Nat.choose totalPlates 3

-- Define the number of ways to choose 3 red plates, 3 blue plates, and 3 green plates
def redWaysChoose3 : ℕ := Nat.choose redPlates 3
def blueWaysChoose3 : ℕ := Nat.choose bluePlates 3
def greenWaysChoose3 : ℕ := Nat.choose greenPlates 3

-- Calculate the total number of favorable combinations (all plates being the same color)
def favorableCombinations : ℕ := redWaysChoose3 + blueWaysChoose3 + greenWaysChoose3

-- State the theorem: the probability that all plates are of the same color.
theorem probability_same_color : (favorableCombinations : ℚ) / (totalWaysChoose3 : ℚ) = 31 / 364 := by sorry

end probability_same_color_l1130_113004


namespace nonneg_int_solutions_eqn_l1130_113070

theorem nonneg_int_solutions_eqn :
  { (x, y, z, w) : ℕ × ℕ × ℕ × ℕ | 2^x * 3^y - 5^z * 7^w = 1 } =
  {(1, 0, 0, 0), (3, 0, 0, 1), (1, 1, 1, 0), (2, 2, 1, 1)} :=
by {
  sorry
}

end nonneg_int_solutions_eqn_l1130_113070


namespace calculate_expression_l1130_113078

theorem calculate_expression (x y : ℕ) (hx : x = 3) (hy : y = 4) : 
  (1 / (y + 1)) / (1 / (x + 2)) = 1 := by
  sorry

end calculate_expression_l1130_113078


namespace sum_50th_set_correct_l1130_113076

noncomputable def sum_of_fiftieth_set : ℕ := 195 + 197

theorem sum_50th_set_correct : sum_of_fiftieth_set = 392 :=
by 
  -- The proof would go here
  sorry

end sum_50th_set_correct_l1130_113076


namespace storm_deposit_eq_120_billion_gallons_l1130_113019

theorem storm_deposit_eq_120_billion_gallons :
  ∀ (initial_content : ℝ) (full_percentage_pre_storm : ℝ) (full_percentage_post_storm : ℝ) (reservoir_capacity : ℝ),
  initial_content = 220 * 10^9 → 
  full_percentage_pre_storm = 0.55 →
  full_percentage_post_storm = 0.85 →
  reservoir_capacity = initial_content / full_percentage_pre_storm →
  (full_percentage_post_storm * reservoir_capacity - initial_content) = 120 * 10^9 :=
by
  intro initial_content full_percentage_pre_storm full_percentage_post_storm reservoir_capacity
  intros h_initial_content h_pre_storm h_post_storm h_capacity
  sorry

end storm_deposit_eq_120_billion_gallons_l1130_113019


namespace cost_of_pen_is_30_l1130_113052

noncomputable def mean_expenditure_per_day : ℕ := 500
noncomputable def days_in_week : ℕ := 7
noncomputable def total_expenditure : ℕ := mean_expenditure_per_day * days_in_week

noncomputable def mon_expenditure : ℕ := 450
noncomputable def tue_expenditure : ℕ := 600
noncomputable def wed_expenditure : ℕ := 400
noncomputable def thurs_expenditure : ℕ := 500
noncomputable def sat_expenditure : ℕ := 550
noncomputable def sun_expenditure : ℕ := 300

noncomputable def fri_notebook_cost : ℕ := 50
noncomputable def fri_earphone_cost : ℕ := 620

noncomputable def total_non_fri_expenditure : ℕ := 
  mon_expenditure + tue_expenditure + wed_expenditure + 
  thurs_expenditure + sat_expenditure + sun_expenditure

noncomputable def fri_expenditure : ℕ := 
  total_expenditure - total_non_fri_expenditure

noncomputable def fri_pen_cost : ℕ := 
  fri_expenditure - (fri_earphone_cost + fri_notebook_cost)

theorem cost_of_pen_is_30 : fri_pen_cost = 30 :=
  sorry

end cost_of_pen_is_30_l1130_113052


namespace permutation_inequality_l1130_113081

theorem permutation_inequality (a b c d : ℝ) (h : a * b * c * d > 0) :
  ∃ (x y z w : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d) ∧ (y = a ∨ y = b ∨ y = c ∨ y = d) ∧
  (z = a ∨ z = b ∨ z = c ∨ z = d) ∧ (w = a ∨ w = b ∨ w = c ∨ w = d) ∧ 
  2 * (x * y + z * w)^2 > (x^2 + y^2) * (z^2 + w^2) := 
sorry

end permutation_inequality_l1130_113081


namespace angle_ACB_33_l1130_113062

noncomputable def triangle_ABC : Type := sorry  -- Define the triangle ABC
noncomputable def ω : Type := sorry  -- Define the circumcircle of ABC
noncomputable def M : Type := sorry  -- Define the midpoint of arc BC not containing A
noncomputable def D : Type := sorry  -- Define the point D such that DM is tangent to ω
def AM_eq_AC : Prop := sorry  -- Define the equality AM = AC
def angle_DMC := (38 : ℝ)  -- Define angle DMC = 38 degrees

theorem angle_ACB_33 (h1 : triangle_ABC) 
                      (h2 : ω) 
                      (h3 : M) 
                      (h4 : D) 
                      (h5 : AM_eq_AC)
                      (h6 : angle_DMC = 38) : ∃ θ, (θ = 33) ∧ (angle_ACB = θ) :=
sorry  -- Proof goes here

end angle_ACB_33_l1130_113062
