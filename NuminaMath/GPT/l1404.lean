import Mathlib

namespace find_omega_l1404_140421

noncomputable def omega_solution (ω : ℝ) : Prop :=
  ω > 0 ∧
  (∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) > 2 * Real.cos (ω * y)) ∧
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 * Real.pi / 3 → 2 * Real.cos (ω * x) ≥ 1)

theorem find_omega : omega_solution (1 / 2) :=
sorry

end find_omega_l1404_140421


namespace find_k_l1404_140427

variable (k : ℝ)
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, 2)

theorem find_k 
  (h : (k * a.1 - b.1, k * a.2 - b.2) = (k - 1, k - 2)) 
  (perp_cond : (k * a.1 - b.1, k * a.2 - b.2).fst * (b.1 + a.1) + (k * a.1 - b.1, k * a.2 - b.2).snd * (b.2 + a.2) = 0) :
  k = 8 / 5 :=
sorry

end find_k_l1404_140427


namespace total_donation_correct_l1404_140441

def carwash_earnings : ℝ := 100
def carwash_donation : ℝ := carwash_earnings * 0.90

def bakesale_earnings : ℝ := 80
def bakesale_donation : ℝ := bakesale_earnings * 0.75

def mowinglawns_earnings : ℝ := 50
def mowinglawns_donation : ℝ := mowinglawns_earnings * 1.00

def total_donation : ℝ := carwash_donation + bakesale_donation + mowinglawns_donation

theorem total_donation_correct : total_donation = 200 := by
  -- the proof will be written here
  sorry

end total_donation_correct_l1404_140441


namespace curve_intersection_four_points_l1404_140412

theorem curve_intersection_four_points (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = 4 * a^2 ∧ y = a * x^2 - 2 * a) ∧ 
  (∃! (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ), 
    x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧ 
    y1 ≠ y2 ∧ y1 ≠ y3 ∧ y1 ≠ y4 ∧ y2 ≠ y3 ∧ y2 ≠ y4 ∧ y3 ≠ y4 ∧
    x1^2 + y1^2 = 4 * a^2 ∧ y1 = a * x1^2 - 2 * a ∧
    x2^2 + y2^2 = 4 * a^2 ∧ y2 = a * x2^2 - 2 * a ∧
    x3^2 + y3^2 = 4 * a^2 ∧ y3 = a * x3^2 - 2 * a ∧
    x4^2 + y4^2 = 4 * a^2 ∧ y4 = a * x4^2 - 2 * a) ↔ 
  a > 1 / 2 :=
by 
  sorry

end curve_intersection_four_points_l1404_140412


namespace find_starting_number_of_range_l1404_140418

theorem find_starting_number_of_range :
  ∃ x, (∀ n, 0 ≤ n ∧ n < 10 → 65 - 5 * n = x + 5 * (9 - n)) ∧ x = 15 := 
by
  sorry

end find_starting_number_of_range_l1404_140418


namespace find_a_plus_b_l1404_140490

noncomputable def A : ℝ := 3
noncomputable def B : ℝ := -1

noncomputable def l : ℝ := -1 -- Slope of line l (since angle is 3π/4)

noncomputable def l1_slope : ℝ := 1 -- Slope of line l1 which is perpendicular to l

noncomputable def a : ℝ := 0 -- Calculated from k_{AB} = 1

noncomputable def b : ℝ := -2 -- Calculated from line parallel condition

theorem find_a_plus_b : a + b = -2 :=
by
  sorry

end find_a_plus_b_l1404_140490


namespace completion_time_l1404_140491

theorem completion_time (total_work : ℕ) (initial_num_men : ℕ) (initial_efficiency : ℝ)
  (new_num_men : ℕ) (new_efficiency : ℝ) :
  total_work = 12 ∧ initial_num_men = 4 ∧ initial_efficiency = 1.5 ∧
  new_num_men = 6 ∧ new_efficiency = 2.0 →
  total_work / (new_num_men * new_efficiency) = 1 :=
by
  sorry

end completion_time_l1404_140491


namespace quadratic_completing_the_square_l1404_140443

theorem quadratic_completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 → (x - 2)^2 = 6 :=
by sorry

end quadratic_completing_the_square_l1404_140443


namespace fourth_person_height_l1404_140489

theorem fourth_person_height 
  (H : ℕ) 
  (h_avg : (H + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) : 
  H + 10 = 85 :=
by
  sorry

end fourth_person_height_l1404_140489


namespace jordan_field_area_l1404_140437

theorem jordan_field_area
  (s l : ℕ)
  (h1 : 2 * (s + l) = 24)
  (h2 : l + 1 = 2 * (s + 1)) :
  3 * s * 3 * l = 189 := 
by
  sorry

end jordan_field_area_l1404_140437


namespace arithmetic_sequence_sums_l1404_140480

variable (a : ℕ → ℕ)

-- Conditions
def condition1 := a 1 + a 4 + a 7 = 39
def condition2 := a 2 + a 5 + a 8 = 33

-- Question and expected answer
def result := a 3 + a 6 + a 9 = 27

theorem arithmetic_sequence_sums (h1 : condition1 a) (h2 : condition2 a) : result a := 
sorry

end arithmetic_sequence_sums_l1404_140480


namespace remaining_pencils_l1404_140415

-- Define the initial conditions
def initial_pencils : Float := 56.0
def pencils_given : Float := 9.0

-- Formulate the theorem stating that the remaining pencils = 47.0
theorem remaining_pencils : initial_pencils - pencils_given = 47.0 := by
  sorry

end remaining_pencils_l1404_140415


namespace best_approximation_of_x_squared_l1404_140424

theorem best_approximation_of_x_squared
  (x : ℝ) (A B C D E : ℝ)
  (h1 : -2 < -1)
  (h2 : -1 < 0)
  (h3 : 0 < 1)
  (h4 : 1 < 2)
  (hx : -1 < x ∧ x < 0)
  (hC : 0 < C ∧ C < 1) :
  x^2 = C :=
sorry

end best_approximation_of_x_squared_l1404_140424


namespace proposition_a_proposition_b_proposition_c_proposition_d_l1404_140403

variable (a b c : ℝ)

-- Proposition A: If ac^2 > bc^2, then a > b
theorem proposition_a (h : a * c^2 > b * c^2) : a > b := sorry

-- Proposition B: If a > b, then ac^2 > bc^2
theorem proposition_b (h : a > b) : ¬ (a * c^2 > b * c^2) := sorry

-- Proposition C: If a > b, then 1/a < 1/b
theorem proposition_c (h : a > b) : ¬ (1/a < 1/b) := sorry

-- Proposition D: If a > b > 0, then a^2 > ab > b^2
theorem proposition_d (h1 : a > b) (h2 : b > 0) : a^2 > a * b ∧ a * b > b^2 := sorry

end proposition_a_proposition_b_proposition_c_proposition_d_l1404_140403


namespace force_required_for_bolt_b_20_inch_l1404_140465

noncomputable def force_inversely_proportional (F L : ℝ) : ℝ := F * L

theorem force_required_for_bolt_b_20_inch (F L : ℝ) :
  let handle_length_10 := 10
  let force_length_product_bolt_a := 3000
  let force_length_product_bolt_b := 4000
  let new_handle_length := 20
  (F * handle_length_10 = 400)
  ∧ (F * new_handle_length = 200)
  → force_inversely_proportional 400 10 = 4000
  ∧ force_inversely_proportional 200 20 = 4000
:=
by
  sorry

end force_required_for_bolt_b_20_inch_l1404_140465


namespace fish_ratio_bobby_sarah_l1404_140402

-- Defining the conditions
variables (bobby sarah tony billy : ℕ)

-- Condition: Billy has 10 fish.
def billy_has_10_fish : billy = 10 := by sorry

-- Condition: Tony has 3 times as many fish as Billy.
def tony_has_3_times_billy : tony = 3 * billy := by sorry

-- Condition: Sarah has 5 more fish than Tony.
def sarah_has_5_more_than_tony : sarah = tony + 5 := by sorry

-- Condition: All 4 people have 145 fish together.
def total_fish : bobby + sarah + tony + billy = 145 := by sorry

-- The theorem we want to prove
theorem fish_ratio_bobby_sarah : (bobby : ℚ) / sarah = 2 / 1 := by
  -- You can write out the entire proof step by step here, but initially, we'll just put sorry.
  sorry

end fish_ratio_bobby_sarah_l1404_140402


namespace transport_capacity_l1404_140413

-- Declare x and y as the amount of goods large and small trucks can transport respectively
variables (x y : ℝ)

-- Given conditions
def condition1 : Prop := 2 * x + 3 * y = 15.5
def condition2 : Prop := 5 * x + 6 * y = 35

-- The goal to prove
def goal : Prop := 3 * x + 5 * y = 24.5

-- Main theorem stating that given the conditions, the goal follows
theorem transport_capacity (h1 : condition1 x y) (h2 : condition2 x y) : goal x y :=
by sorry

end transport_capacity_l1404_140413


namespace failed_students_calculation_l1404_140425

theorem failed_students_calculation (total_students : ℕ) (percentage_passed : ℕ)
  (h_total : total_students = 840) (h_passed : percentage_passed = 35) :
  (total_students * (100 - percentage_passed) / 100) = 546 :=
by
  sorry

end failed_students_calculation_l1404_140425


namespace factor_expression_l1404_140446

theorem factor_expression (y : ℝ) : 49 - 16*y^2 + 8*y = (7 - 4*y)*(7 + 4*y) := 
sorry

end factor_expression_l1404_140446


namespace volume_calc_l1404_140456

noncomputable
def volume_of_open_box {l w : ℕ} (sheet_length : l = 48) (sheet_width : w = 38) (cut_length : ℕ) (cut_length_eq : cut_length = 8) : ℕ :=
  let new_length := l - 2 * cut_length
  let new_width := w - 2 * cut_length
  let height := cut_length
  new_length * new_width * height

theorem volume_calc : volume_of_open_box (sheet_length := rfl) (sheet_width := rfl) (cut_length := 8) (cut_length_eq := rfl) = 5632 :=
sorry

end volume_calc_l1404_140456


namespace triangular_array_sum_digits_l1404_140468

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 3780) : (N / 10 + N % 10) = 15 :=
sorry

end triangular_array_sum_digits_l1404_140468


namespace value_of_x7_plus_64x2_l1404_140400

-- Let x be a real number such that x^3 + 4x = 8.
def x_condition (x : ℝ) : Prop := x^3 + 4 * x = 8

-- We need to determine the value of x^7 + 64x^2.
theorem value_of_x7_plus_64x2 (x : ℝ) (h : x_condition x) : x^7 + 64 * x^2 = 128 :=
by
  sorry

end value_of_x7_plus_64x2_l1404_140400


namespace next_perfect_cube_l1404_140470

theorem next_perfect_cube (x : ℕ) (h : ∃ k : ℕ, x = k^3) : 
  ∃ m : ℕ, m^3 = x + 3 * (x^(1/3))^2 + 3 * x^(1/3) + 1 :=
by
  sorry

end next_perfect_cube_l1404_140470


namespace bens_car_costs_l1404_140401

theorem bens_car_costs :
  (∃ C_old C_2nd : ℕ,
    (2 * C_old = 4 * C_2nd) ∧
    (C_old = 1800) ∧
    (C_2nd = 900) ∧
    (2 * C_old = 3600) ∧
    (4 * C_2nd = 3600) ∧
    (1800 + 900 = 2700) ∧
    (3600 - 2700 = 900) ∧
    (2000 - 900 = 1100) ∧
    (900 * 0.05 = 45) ∧
    (45 * 2 = 90))
  :=
sorry

end bens_car_costs_l1404_140401


namespace tiled_floor_area_correct_garden_area_correct_seating_area_correct_l1404_140488

noncomputable def length_room : ℝ := 20
noncomputable def width_room : ℝ := 12
noncomputable def width_veranda : ℝ := 2
noncomputable def length_pool : ℝ := 15
noncomputable def width_pool : ℝ := 6

noncomputable def area (length width : ℝ) : ℝ := length * width

noncomputable def area_room : ℝ := area length_room width_room
noncomputable def area_pool : ℝ := area length_pool width_pool
noncomputable def area_tiled_floor : ℝ := area_room - area_pool

noncomputable def total_length : ℝ := length_room + 2 * width_veranda
noncomputable def total_width : ℝ := width_room + 2 * width_veranda
noncomputable def area_total : ℝ := area total_length total_width
noncomputable def area_veranda : ℝ := area_total - area_room
noncomputable def area_garden : ℝ := area_veranda / 2
noncomputable def area_seating : ℝ := area_veranda / 2

theorem tiled_floor_area_correct : area_tiled_floor = 150 := by
  sorry

theorem garden_area_correct : area_garden = 72 := by
  sorry

theorem seating_area_correct : area_seating = 72 := by
  sorry

end tiled_floor_area_correct_garden_area_correct_seating_area_correct_l1404_140488


namespace sum_of_three_integers_l1404_140477

theorem sum_of_three_integers :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = 125 ∧ a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_l1404_140477


namespace find_a_l1404_140454

theorem find_a (k x y a : ℝ) (hkx : k ≤ x) (hx3 : x ≤ 3) (hy7 : a ≤ y) (hy7' : y ≤ 7) (hy : y = k * x + 1) :
  a = 5 ∨ a = 1 - 3 * Real.sqrt 6 :=
sorry

end find_a_l1404_140454


namespace speed_ratio_l1404_140493

variables (H D : ℝ)
variables (duck_leaps hen_leaps : ℕ)
-- hen_leaps and duck_leaps denote the leaps taken by hen and duck respectively

-- conditions given
axiom cond1 : hen_leaps = 6 ∧ duck_leaps = 8
axiom cond2 : 4 * D = 3 * H

-- goal to prove
theorem speed_ratio (H D : ℝ) (hen_leaps duck_leaps : ℕ) (cond1 : hen_leaps = 6 ∧ duck_leaps = 8) (cond2 : 4 * D = 3 * H) : 
  (6 * H) = (8 * D) :=
by
  intros
  sorry

end speed_ratio_l1404_140493


namespace cos_double_angle_l1404_140475

variable (α : ℝ)

theorem cos_double_angle (h1 : 0 < α ∧ α < π / 2) 
                         (h2 : Real.cos ( α + π / 4) = 3 / 5) : 
    Real.cos (2 * α) = 24 / 25 :=
by
  sorry

end cos_double_angle_l1404_140475


namespace g_zero_eq_zero_l1404_140408

noncomputable def g : ℝ → ℝ :=
  sorry

axiom functional_equation (a b : ℝ) :
  g (3 * a + 2 * b) + g (3 * a - 2 * b) = 2 * g (3 * a) + 2 * g (2 * b)

theorem g_zero_eq_zero : g 0 = 0 :=
by
  let a := 0
  let b := 0
  have eqn := functional_equation a b
  sorry

end g_zero_eq_zero_l1404_140408


namespace number_of_workers_in_original_scenario_l1404_140429

-- Definitions based on the given conditions
def original_days := 70
def alternative_days := 42
def alternative_workers := 50

-- The statement we want to prove
theorem number_of_workers_in_original_scenario : 
  (∃ (W : ℕ), W * original_days = alternative_workers * alternative_days) → ∃ (W : ℕ), W = 30 :=
by
  sorry

end number_of_workers_in_original_scenario_l1404_140429


namespace determine_k_for_intersection_l1404_140423

theorem determine_k_for_intersection (k : ℝ) :
  (∃ x : ℝ, k * x^2 + 2 * x + 3 = 2 * x + 5) ∧ 
  (∀ x₁ x₂ : ℝ, (k * x₁^2 + 2 * x₁ + 3 = 2 * x₁ + 5) ∧ 
                (k * x₂^2 + 2 * x₂ + 3 = 2 * x₂ + 5) → 
              x₁ = x₂) ↔ k = -1/2 :=
by
  sorry

end determine_k_for_intersection_l1404_140423


namespace function_decomposition_l1404_140495

open Real

noncomputable def f (x : ℝ) : ℝ := log (10^x + 1)
noncomputable def g (x : ℝ) : ℝ := x / 2
noncomputable def h (x : ℝ) : ℝ := log (10^x + 1) - x / 2

theorem function_decomposition :
  ∀ x : ℝ, f x = g x + h x ∧ (∀ x, g (-x) = -g x) ∧ (∀ x, h (-x) = h x) :=
by
  intro x
  sorry

end function_decomposition_l1404_140495


namespace smallest_x_exists_l1404_140447

theorem smallest_x_exists {M : ℤ} (h : 2520 = 2^3 * 3^2 * 5 * 7) : 
  ∃ x : ℕ, 2520 * x = M^3 ∧ x = 3675 := 
by {
  sorry
}

end smallest_x_exists_l1404_140447


namespace sum_of_arithmetic_sequence_l1404_140472

-- Given conditions in the problem
axiom arithmetic_sequence (a : ℕ → ℤ): Prop
axiom are_roots (a b : ℤ): ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a = p ∨ a = q) ∧ (b = p ∨ b = q)

-- The equivalent proof problem statement
theorem sum_of_arithmetic_sequence (a : ℕ → ℤ) 
  (h1 : arithmetic_sequence a)
  (h2 : ∃ p q : ℤ, p * q = -5 ∧ p + q = 3 ∧ (a 2 = p ∨ a 2 = q) ∧ (a 11 = p ∨ a 11 = q)):
  a 5 + a 8 = 3 :=
sorry

end sum_of_arithmetic_sequence_l1404_140472


namespace first_player_wins_if_take_one_initial_l1404_140461

theorem first_player_wins_if_take_one_initial :
  ∃ strategy : ℕ → ℕ, 
    (∀ n, strategy n = if n % 3 = 0 then 1 else 2) ∧ 
    strategy 99 = 1 ∧ 
    strategy 100 = 1 :=
sorry

end first_player_wins_if_take_one_initial_l1404_140461


namespace greatest_m_value_l1404_140492

theorem greatest_m_value (x y z u : ℕ) (hx : x ≥ y) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) : 
  ∃ m, m = 3 + 2 * Real.sqrt 2 ∧ m ≤ x / y :=
sorry

end greatest_m_value_l1404_140492


namespace Kylie_US_coins_left_l1404_140438

-- Define the given conditions
def initial_US_coins : ℝ := 15
def Euro_coins : ℝ := 13
def Canadian_coins : ℝ := 8
def US_coins_given_to_Laura : ℝ := 21
def Euro_to_US_rate : ℝ := 1.18
def Canadian_to_US_rate : ℝ := 0.78

-- Define the conversions
def Euro_to_US : ℝ := Euro_coins * Euro_to_US_rate
def Canadian_to_US : ℝ := Canadian_coins * Canadian_to_US_rate
def total_US_before_giving : ℝ := initial_US_coins + Euro_to_US + Canadian_to_US
def US_left_with : ℝ := total_US_before_giving - US_coins_given_to_Laura

-- Statement of the problem to be proven
theorem Kylie_US_coins_left :
  US_left_with = 15.58 := by
  sorry

end Kylie_US_coins_left_l1404_140438


namespace total_pay_is_186_l1404_140485

-- Define the conditions
def regular_rate : ℕ := 3 -- dollars per hour
def regular_hours : ℕ := 40 -- hours
def overtime_rate_multiplier : ℕ := 2
def overtime_hours : ℕ := 11

-- Calculate the regular pay
def regular_pay : ℕ := regular_hours * regular_rate

-- Calculate the overtime pay
def overtime_rate : ℕ := regular_rate * overtime_rate_multiplier
def overtime_pay : ℕ := overtime_hours * overtime_rate

-- Calculate the total pay
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem total_pay_is_186 : total_pay = 186 :=
by 
  sorry

end total_pay_is_186_l1404_140485


namespace complement_of_log_set_l1404_140458

-- Define the set A based on the logarithmic inequality condition
def A : Set ℝ := { x : ℝ | Real.log x / Real.log (1 / 2) ≥ 2 }

-- Define the complement of A in the real numbers
noncomputable def complement_A : Set ℝ := { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 }

-- The goal is to prove the equivalence
theorem complement_of_log_set :
  complement_A = { x : ℝ | x ≤ 0 } ∪ { x : ℝ | x > 1 / 4 } :=
by
  sorry

end complement_of_log_set_l1404_140458


namespace monthly_income_of_p_l1404_140487

theorem monthly_income_of_p (P Q R : ℕ) 
    (h1 : (P + Q) / 2 = 5050)
    (h2 : (Q + R) / 2 = 6250)
    (h3 : (P + R) / 2 = 5200) :
    P = 4000 :=
by
  -- proof would go here
  sorry

end monthly_income_of_p_l1404_140487


namespace probability_six_distinct_numbers_l1404_140410

theorem probability_six_distinct_numbers :
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  probability = (35 / 648) := 
by
  let outcomes := 6^7
  let favorable_outcomes := 1 * 6 * (Nat.choose 7 2) * (Nat.factorial 5)
  let probability := favorable_outcomes / outcomes
  have h : favorable_outcomes = 15120 := by sorry
  have h2 : outcomes = 279936 := by sorry
  have prob : probability = (15120 / 279936) := by sorry
  have gcd_calc : gcd 15120 279936 = 432 := by sorry
  have simplified_prob : (15120 / 279936) = (35 / 648) := by sorry
  exact simplified_prob

end probability_six_distinct_numbers_l1404_140410


namespace dogs_left_l1404_140474

-- Define the conditions
def total_dogs : ℕ := 50
def dog_houses : ℕ := 17

-- Statement to prove the number of dogs left
theorem dogs_left : (total_dogs % dog_houses) = 16 :=
by sorry

end dogs_left_l1404_140474


namespace sixteenth_term_l1404_140484

theorem sixteenth_term :
  (-1)^(16+1) * Real.sqrt (3 * (16 - 1)) = -3 * Real.sqrt 5 :=
by sorry

end sixteenth_term_l1404_140484


namespace negation_of_universal_l1404_140406

theorem negation_of_universal {x : ℝ} : ¬ (∀ x > 0, x^2 - x ≤ 0) ↔ ∃ x > 0, x^2 - x > 0 :=
by
  sorry

end negation_of_universal_l1404_140406


namespace paint_snake_l1404_140463

theorem paint_snake (num_cubes : ℕ) (paint_per_cube : ℕ) (end_paint : ℕ) (total_paint : ℕ) 
  (h_cubes : num_cubes = 2016)
  (h_paint_per_cube : paint_per_cube = 60)
  (h_end_paint : end_paint = 20)
  (h_total_paint : total_paint = 121000) :
  total_paint = (num_cubes * paint_per_cube) + 2 * end_paint :=
by
  rw [h_cubes, h_paint_per_cube, h_end_paint]
  sorry

end paint_snake_l1404_140463


namespace football_team_total_players_l1404_140457

variable (P : ℕ)
variable (throwers : ℕ := 52)
variable (total_right_handed : ℕ := 64)
variable (remaining := P - throwers)
variable (left_handed := remaining / 3)
variable (right_handed_non_throwers := 2 * remaining / 3)

theorem football_team_total_players:
  right_handed_non_throwers + throwers = total_right_handed →
  P = 70 :=
by
  sorry

end football_team_total_players_l1404_140457


namespace hyperbola_triangle_area_l1404_140483

/-- The relationship between the hyperbola's asymptotes, tangent, and area proportion -/
theorem hyperbola_triangle_area (a b x0 y0 : ℝ) 
  (h_asymptote1 : ∀ x, y = (b / a) * x)
  (h_asymptote2 : ∀ x, y = -(b / a) * x)
  (h_tangent    : ∀ x y, (x0 * x) / (a ^ 2) - (y0 * y) / (b ^ 2) = 1)
  (h_condition  : (x0 ^ 2) * (a ^ 2) - (y0 ^ 2) * (b ^ 2) = (a ^ 2) * (b ^ 2)) :
  ∃ k : ℝ, k = a ^ 4 :=
sorry

end hyperbola_triangle_area_l1404_140483


namespace square_of_negative_eq_square_l1404_140426

theorem square_of_negative_eq_square (a : ℝ) : (-a)^2 = a^2 :=
sorry

end square_of_negative_eq_square_l1404_140426


namespace arithmetic_sequence_eightieth_term_l1404_140498

open BigOperators

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

theorem arithmetic_sequence_eightieth_term :
  ∀ (d : ℝ),
  arithmetic_sequence 3 d 21 = 41 →
  arithmetic_sequence 3 d 80 = 153.1 :=
by
  intros
  sorry

end arithmetic_sequence_eightieth_term_l1404_140498


namespace circular_garden_radius_l1404_140444

theorem circular_garden_radius (r : ℝ) (h1 : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) : r = 12 :=
by sorry

end circular_garden_radius_l1404_140444


namespace correct_propositions_l1404_140416

-- Definitions based on the propositions
def prop1 := 
"Sampling every 20 minutes from a uniformly moving production line is stratified sampling."

def prop2 := 
"The stronger the correlation between two random variables, the closer the absolute value of the correlation coefficient is to 1."

def prop3 := 
"In the regression line equation hat_y = 0.2 * x + 12, the forecasted variable hat_y increases by 0.2 units on average for each unit increase in the explanatory variable x."

def prop4 := 
"For categorical variables X and Y, the smaller the observed value k of their statistic K², the greater the certainty of the relationship between X and Y."

-- Mathematical statements for propositions
def p1 : Prop := false -- Proposition ① is incorrect
def p2 : Prop := true  -- Proposition ② is correct
def p3 : Prop := true  -- Proposition ③ is correct
def p4 : Prop := false -- Proposition ④ is incorrect

-- The theorem we need to prove
theorem correct_propositions : (p2 = true) ∧ (p3 = true) :=
by 
  -- Details of the proof here
  sorry

end correct_propositions_l1404_140416


namespace student_percentage_to_pass_l1404_140430

/-- A student needs to obtain 50% of the total marks to pass given the conditions:
    1. The student got 200 marks.
    2. The student failed by 20 marks.
    3. The maximum marks are 440. -/
theorem student_percentage_to_pass : 
  ∀ (student_marks : ℕ) (failed_by : ℕ) (max_marks : ℕ),
  student_marks = 200 → failed_by = 20 → max_marks = 440 →
  (student_marks + failed_by) / max_marks * 100 = 50 := 
by
  intros student_marks failed_by max_marks h1 h2 h3
  sorry

end student_percentage_to_pass_l1404_140430


namespace solve_trig_equation_l1404_140409
open Real

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (1 / 2) * abs (cos (2 * x) + (1 / 2)) = (sin (3 * x))^2 - (sin x) * (sin (3 * x))

-- Define the correct solution set 
def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (π / 6) + (k * (π / 2)) ∨ x = -(π / 6) + (k * (π / 2))

-- The theorem we need to prove
theorem solve_trig_equation : ∀ x : ℝ, original_equation x ↔ solution_set x :=
by sorry

end solve_trig_equation_l1404_140409


namespace remainder_product_div_17_l1404_140486

theorem remainder_product_div_17 :
  (2357 ≡ 6 [MOD 17]) → (2369 ≡ 4 [MOD 17]) → (2384 ≡ 0 [MOD 17]) →
  (2391 ≡ 9 [MOD 17]) → (3017 ≡ 9 [MOD 17]) → (3079 ≡ 0 [MOD 17]) →
  (3082 ≡ 3 [MOD 17]) →
  ((2357 * 2369 * 2384 * 2391) * (3017 * 3079 * 3082) ≡ 0 [MOD 17]) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end remainder_product_div_17_l1404_140486


namespace sum_of_first_10_terms_l1404_140455

noncomputable def a (n : ℕ) : ℕ := sorry
noncomputable def S (n : ℕ) : ℕ := sorry

variable {n : ℕ}

-- Conditions
axiom h1 : ∀ n, S (n + 1) = S n + a n + 3
axiom h2 : a 5 + a 6 = 29

-- Statement to prove
theorem sum_of_first_10_terms : S 10 = 145 := 
sorry

end sum_of_first_10_terms_l1404_140455


namespace number_of_birds_is_400_l1404_140462

-- Definitions of the problem
def num_stones : ℕ := 40
def num_trees : ℕ := 3 * num_stones + num_stones
def combined_trees_stones : ℕ := num_trees + num_stones
def num_birds : ℕ := 2 * combined_trees_stones

-- Statement to prove
theorem number_of_birds_is_400 : num_birds = 400 := by
  sorry

end number_of_birds_is_400_l1404_140462


namespace max_roses_purchasable_l1404_140442

theorem max_roses_purchasable 
  (price_individual : ℝ) (price_dozen : ℝ) (price_two_dozen : ℝ) (price_five_dozen : ℝ) 
  (discount_threshold : ℕ) (discount_rate : ℝ) (total_money : ℝ) : 
  (price_individual = 4.50) →
  (price_dozen = 36) →
  (price_two_dozen = 50) →
  (price_five_dozen = 110) →
  (discount_threshold = 36) →
  (discount_rate = 0.10) →
  (total_money = 680) →
  ∃ (roses : ℕ), roses = 364 :=
by
  -- Definitions based on conditions
  intros
  -- The proof steps have been omitted for brevity
  sorry

end max_roses_purchasable_l1404_140442


namespace line_length_l1404_140422

theorem line_length (n : ℕ) (d : ℤ) (h1 : n = 51) (h2 : d = 3) : 
  (n - 1) * d = 150 := sorry

end line_length_l1404_140422


namespace friend_balloons_count_l1404_140499

-- Definitions of the conditions
def balloons_you_have : ℕ := 7
def balloons_difference : ℕ := 2

-- Proof problem statement
theorem friend_balloons_count : (balloons_you_have - balloons_difference) = 5 :=
by
  sorry

end friend_balloons_count_l1404_140499


namespace rationalize_denominator_l1404_140494

theorem rationalize_denominator : (7 / Real.sqrt 147) = (Real.sqrt 3 / 3) :=
by
  sorry

end rationalize_denominator_l1404_140494


namespace repeating_decimals_sum_is_fraction_l1404_140476

-- Define the repeating decimals as fractions
def x : ℚ := 1 / 3
def y : ℚ := 2 / 99

-- Define the sum of the repeating decimals
def sum := x + y

-- State the theorem
theorem repeating_decimals_sum_is_fraction :
  sum = 35 / 99 := sorry

end repeating_decimals_sum_is_fraction_l1404_140476


namespace max_f_value_l1404_140440

open Real

noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

theorem max_f_value : ∃ (x₀ y₀ : ℝ), (0 < x₀) ∧ (0 < y₀) ∧ (∀ (x y : ℝ), (0 < x) → (0 < y) → f x y ≤ f x₀ y₀) ∧ f x₀ y₀ = 1 / sqrt 2 :=
by 
  sorry

end max_f_value_l1404_140440


namespace inequality_of_cubic_powers_l1404_140481

theorem inequality_of_cubic_powers 
  (a b: ℝ) (h : a ≠ 0 ∧ b ≠ 0) 
  (h_cond : a * |a| > b * |b|) : 
  a^3 > b^3 := by
  sorry

end inequality_of_cubic_powers_l1404_140481


namespace original_price_of_shoes_l1404_140471

-- Define the conditions.
def discount_rate : ℝ := 0.20
def amount_paid : ℝ := 480

-- Statement of the theorem.
theorem original_price_of_shoes (P : ℝ) (h₀ : P * (1 - discount_rate) = amount_paid) : 
  P = 600 :=
by
  sorry

end original_price_of_shoes_l1404_140471


namespace tangent_planes_of_surface_and_given_plane_l1404_140404

-- Define the surface and the given plane
def surface (x y z : ℝ) := (x^2 + 4 * y^2 + 9 * z^2 = 1)
def given_plane (x y z : ℝ) := (x + y + 2 * z = 1)

-- Define the tangent plane equations to be proved
def tangent_plane_1 (x y z : ℝ) := (x + y + 2 * z - (109 / (6 * Real.sqrt 61)) = 0)
def tangent_plane_2 (x y z : ℝ) := (x + y + 2 * z + (109 / (6 * Real.sqrt 61)) = 0)

-- The statement to be proved
theorem tangent_planes_of_surface_and_given_plane :
  ∀ x y z, surface x y z ∧ given_plane x y z →
    tangent_plane_1 x y z ∨ tangent_plane_2 x y z :=
sorry

end tangent_planes_of_surface_and_given_plane_l1404_140404


namespace jane_performance_l1404_140436

theorem jane_performance :
  ∃ (p w e : ℕ), 
  p + w + e = 15 ∧ 
  2 * p + 4 * w + 6 * e = 66 ∧ 
  e = p + 4 ∧ 
  w = 11 :=
by
  sorry

end jane_performance_l1404_140436


namespace gcd_98_140_245_l1404_140451

theorem gcd_98_140_245 : Nat.gcd (Nat.gcd 98 140) 245 = 7 := 
by 
  sorry

end gcd_98_140_245_l1404_140451


namespace find_expression_value_l1404_140482

theorem find_expression_value : 1 + 2 * 3 - 4 + 5 = 8 :=
by
  sorry

end find_expression_value_l1404_140482


namespace xy_value_l1404_140449

theorem xy_value (x y : ℝ) (h : (|x| - 1)^2 + (2 * y + 1)^2 = 0) : xy = 1/2 ∨ xy = -1/2 :=
by {
  sorry
}

end xy_value_l1404_140449


namespace find_abc_sum_l1404_140411

theorem find_abc_sum :
  ∃ (a b c : ℤ), 2 * a + 3 * b = 52 ∧ 3 * b + c = 41 ∧ b * c = 60 ∧ a + b + c = 25 :=
by
  use 8, 12, 5
  sorry

end find_abc_sum_l1404_140411


namespace nth_term_closed_form_arithmetic_sequence_l1404_140459

open Nat

noncomputable def S (n : ℕ) : ℕ := 3 * n^2 + 4 * n
noncomputable def a (n : ℕ) : ℕ := if h : n > 0 then S n - S (n-1) else S n

theorem nth_term_closed_form (n : ℕ) (h : n > 0) : a n = 6 * n + 1 :=
by
  sorry

theorem arithmetic_sequence (n : ℕ) (h : n > 1) : a n - a (n - 1) = 6 :=
by
  sorry

end nth_term_closed_form_arithmetic_sequence_l1404_140459


namespace cube_volume_in_pyramid_l1404_140479

-- Definition for the conditions and parameters of the problem
def pyramid_condition (base_length : ℝ) (triangle_side : ℝ) : Prop :=
  base_length = 2 ∧ triangle_side = 2 * Real.sqrt 2

-- Definition for the cube's placement and side length condition inside the pyramid
def cube_side_length (s : ℝ) : Prop :=
  s = (Real.sqrt 6 / 3)

-- The final Lean statement proving the volume of the cube
theorem cube_volume_in_pyramid (base_length triangle_side s : ℝ) 
  (h_base_length : base_length = 2)
  (h_triangle_side : triangle_side = 2 * Real.sqrt 2)
  (h_cube_side_length : s = (Real.sqrt 6 / 3)) :
  (s ^ 3) = (2 * Real.sqrt 6 / 9) := 
by
  -- Using the given conditions to assert the conclusion
  rw [h_cube_side_length]
  have : (Real.sqrt 6 / 3) ^ 3 = 2 * Real.sqrt 6 / 9 := sorry
  exact this

end cube_volume_in_pyramid_l1404_140479


namespace percent_of_x_is_y_l1404_140478

theorem percent_of_x_is_y (x y : ℝ) (h : 0.60 * (x - y) = 0.30 * (x + y)) : y = 0.3333 * x :=
by
  sorry

end percent_of_x_is_y_l1404_140478


namespace angle_sum_property_l1404_140450

theorem angle_sum_property
  (angle1 angle2 angle3 : ℝ) 
  (h1 : angle1 = 58) 
  (h2 : angle2 = 35) 
  (h3 : angle3 = 42) : 
  angle1 + angle2 + angle3 + (180 - (angle1 + angle2 + angle3)) = 180 := 
by 
  sorry

end angle_sum_property_l1404_140450


namespace wire_cutting_l1404_140428

theorem wire_cutting : 
  ∃ (n : ℕ), n = 33 ∧ (∀ (x y : ℕ), 3 * x + y = 100 → x > 0 ∧ y > 0 → ∃ m : ℕ, m = n) :=
by {
  sorry
}

end wire_cutting_l1404_140428


namespace george_and_hannah_received_A_grades_l1404_140467

-- Define students as propositions
variables (Elena Fred George Hannah : Prop)

-- Define the conditions
def condition1 : Prop := Elena → Fred
def condition2 : Prop := Fred → George
def condition3 : Prop := George → Hannah
def condition4 : Prop := ∃ A1 A2 : Prop, A1 ∧ A2 ∧ (A1 ≠ A2) ∧ (A1 = George ∨ A1 = Hannah) ∧ (A2 = George ∨ A2 = Hannah)

-- The theorem to be proven: George and Hannah received A grades
theorem george_and_hannah_received_A_grades :
  condition1 Elena Fred →
  condition2 Fred George →
  condition3 George Hannah →
  condition4 George Hannah :=
by
  sorry

end george_and_hannah_received_A_grades_l1404_140467


namespace range_of_mn_l1404_140469

noncomputable def f (x : ℝ) : ℝ := -x^2 + 4 * x

theorem range_of_mn (m n : ℝ)
  (h₁ : ∀ x, m ≤ x ∧ x ≤ n → -5 ≤ f x ∧ f x ≤ 4)
  (h₂ : ∀ z, -5 ≤ z ∧ z ≤ 4 → ∃ x, f x = z ∧ m ≤ x ∧ x ≤ n) :
  1 ≤ m + n ∧ m + n ≤ 7 :=
by
  sorry

end range_of_mn_l1404_140469


namespace Samanta_points_diff_l1404_140407

variables (Samanta Mark Eric : ℕ)

/-- In a game, Samanta has some more points than Mark, Mark has 50% more points than Eric,
Eric has 6 points, and Samanta, Mark, and Eric have a total of 32 points. Prove that Samanta
has 8 more points than Mark. -/
theorem Samanta_points_diff 
    (h1 : Mark = Eric + Eric / 2) 
    (h2 : Eric = 6) 
    (h3 : Samanta + Mark + Eric = 32)
    : Samanta - Mark = 8 :=
sorry

end Samanta_points_diff_l1404_140407


namespace simplify_product_l1404_140432

theorem simplify_product (x y : ℝ) : 
  (x - 3 * y + 2) * (x + 3 * y + 2) = (x^2 + 4 * x + 4 - 9 * y^2) :=
by
  sorry

end simplify_product_l1404_140432


namespace primes_between_30_and_50_l1404_140460

theorem primes_between_30_and_50 : (Finset.card (Finset.filter Nat.Prime (Finset.Ico 30 51))) = 5 :=
by
  sorry

end primes_between_30_and_50_l1404_140460


namespace walking_time_difference_at_slower_speed_l1404_140439

theorem walking_time_difference_at_slower_speed (T : ℕ) (v_s: ℚ) (h1: T = 32) (h2: v_s = 4/5) : 
  (T * (5/4) - T) = 8 :=
by
  sorry

end walking_time_difference_at_slower_speed_l1404_140439


namespace geometric_sum_S30_l1404_140419

theorem geometric_sum_S30 (S : ℕ → ℝ) (h1 : S 10 = 10) (h2 : S 20 = 30) : S 30 = 70 := 
by 
  sorry

end geometric_sum_S30_l1404_140419


namespace graphs_symmetric_l1404_140414

noncomputable def exp2 : ℝ → ℝ := λ x => 2^x
noncomputable def log2 : ℝ → ℝ := λ x => Real.log x / Real.log 2

theorem graphs_symmetric :
  ∀ (x y : ℝ), (y = exp2 x) ↔ (x = log2 y) := sorry

end graphs_symmetric_l1404_140414


namespace find_four_digit_number_l1404_140435

theorem find_four_digit_number :
  ∃ (N : ℕ), 1000 ≤ N ∧ N < 10000 ∧ 
    (N % 131 = 112) ∧ 
    (N % 132 = 98) ∧ 
    N = 1946 :=
by
  sorry

end find_four_digit_number_l1404_140435


namespace inverse_proportion_k_value_l1404_140405

theorem inverse_proportion_k_value (k m : ℝ) 
  (h1 : m = k / 3) 
  (h2 : 6 = k / (m - 1)) 
  : k = 6 :=
by
  sorry

end inverse_proportion_k_value_l1404_140405


namespace inequality_holds_iff_b_lt_a_l1404_140466

theorem inequality_holds_iff_b_lt_a (a b : ℝ) :
  (∀ x : ℝ, (a + 1) * x^2 + a * x + a > b * (x^2 + x + 1)) ↔ b < a :=
by
  sorry

end inequality_holds_iff_b_lt_a_l1404_140466


namespace inequality_proof_l1404_140496

theorem inequality_proof (a b x y : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : x / a < y / b) :
  (1 / 2) * (x / a + y / b) > (x + y) / (a + b) := by
  sorry

end inequality_proof_l1404_140496


namespace correct_word_for_blank_l1404_140464

theorem correct_word_for_blank :
  (∀ (word : String), word = "that" ↔ word = "whoever" ∨ word = "someone" ∨ word = "that" ∨ word = "any") :=
by
  sorry

end correct_word_for_blank_l1404_140464


namespace equivalent_resistance_is_15_l1404_140417

-- Definitions based on conditions
def R : ℝ := 5 -- Resistance of each resistor in Ohms
def num_resistors : ℕ := 4

-- The equivalent resistance due to the short-circuit path removing one resistor
def simplified_circuit_resistance : ℝ := (num_resistors - 1) * R

-- The statement to prove
theorem equivalent_resistance_is_15 :
  simplified_circuit_resistance = 15 :=
by
  sorry

end equivalent_resistance_is_15_l1404_140417


namespace exterior_angle_of_triangle_cond_40_degree_l1404_140497

theorem exterior_angle_of_triangle_cond_40_degree (A B C : ℝ)
  (h1 : (A = 40 ∨ B = 40 ∨ C = 40))
  (h2 : A = B)
  (h3 : A + B + C = 180) :
  ((180 - C) = 80 ∨ (180 - C) = 140) :=
by
  sorry

end exterior_angle_of_triangle_cond_40_degree_l1404_140497


namespace total_highlighters_l1404_140433

-- Define the number of highlighters of each color
def pink_highlighters : ℕ := 10
def yellow_highlighters : ℕ := 15
def blue_highlighters : ℕ := 8

-- Prove the total number of highlighters
theorem total_highlighters : pink_highlighters + yellow_highlighters + blue_highlighters = 33 :=
by
  sorry

end total_highlighters_l1404_140433


namespace no_negative_roots_l1404_140473

theorem no_negative_roots (x : ℝ) :
  x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 = 0 → 0 ≤ x :=
by
  sorry

end no_negative_roots_l1404_140473


namespace tangents_parallel_l1404_140420

variable {R : Type*} [Field R]

-- Let f be a function from ratios to slopes
variable (φ : R -> R)

-- Given points (x, y) and (x₁, y₁) with corresponding conditions
variable (x x₁ y y₁ : R)

-- Conditions
def corresponding_points := y / x = y₁ / x₁
def homogeneous_diff_eqn := ∀ x y, (y / x) = φ (y / x)

-- Prove that the tangents are parallel
theorem tangents_parallel (h_corr : corresponding_points x x₁ y y₁)
  (h_diff_eqn : ∀ (x x₁ y y₁ : R), y' = φ (y / x) ∧ y₁' = φ (y₁ / x₁)) :
  y' = y₁' :=
by
  sorry

end tangents_parallel_l1404_140420


namespace cost_price_of_each_watch_l1404_140453

-- Define the given conditions.
def sold_at_loss (C : ℝ) := 0.925 * C
def total_transaction_price (C : ℝ) := 3 * C * 1.053
def sold_for_more (C : ℝ) := 0.925 * C + 265

-- State the theorem to prove the cost price of each watch.
theorem cost_price_of_each_watch (C : ℝ) :
  3 * sold_for_more C = total_transaction_price C → C = 2070.31 :=
by
  intros h
  sorry

end cost_price_of_each_watch_l1404_140453


namespace flour_more_than_sugar_l1404_140452

/-
  Mary is baking a cake. The recipe calls for 6 cups of sugar and 9 cups of flour. 
  She already put in 2 cups of flour. 
  Prove that the number of additional cups of flour Mary needs is 1 more than the number of additional cups of sugar she needs.
-/

theorem flour_more_than_sugar (s f a : ℕ) (h_s : s = 6) (h_f : f = 9) (h_a : a = 2) :
  (f - a) - s = 1 :=
by
  sorry

end flour_more_than_sugar_l1404_140452


namespace total_books_l1404_140448

theorem total_books (b1 b2 b3 b4 b5 b6 b7 b8 b9 : ℕ) :
  b1 = 56 →
  b2 = b1 + 2 →
  b3 = b2 + 2 →
  b4 = b3 + 2 →
  b5 = b4 + 2 →
  b6 = b5 + 2 →
  b7 = b6 - 4 →
  b8 = b7 - 4 →
  b9 = b8 - 4 →
  b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 = 490 :=
by
  sorry

end total_books_l1404_140448


namespace find_certain_number_l1404_140431

open Real

noncomputable def certain_number (x : ℝ) : Prop :=
  0.75 * x = 0.50 * 900

theorem find_certain_number : certain_number 600 :=
by
  dsimp [certain_number]
  -- We need to show that 0.75 * 600 = 0.50 * 900
  sorry

end find_certain_number_l1404_140431


namespace billy_distance_l1404_140445

noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := 
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem billy_distance :
  distance 0 0 (7 + 4 * Real.sqrt 2) (4 * (Real.sqrt 2 + 1)) = Real.sqrt (129 + 88 * Real.sqrt 2) :=
by
  -- proof goes here
  sorry

end billy_distance_l1404_140445


namespace matrix_multiplication_correct_l1404_140434

-- Define the matrices
def A : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, 0, -3],
    ![1, 3, -2],
    ![0, 2, 4]
  ]

def B : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![1, -1, 0],
    ![0, 2, -1],
    ![3, 0, 1]
  ]

def C : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![-7, -2, -3],
    ![-5, 5, -5],
    ![12, 4, 2]
  ]

-- Proof statement that multiplication of A and B gives C
theorem matrix_multiplication_correct : A * B = C := 
by
  sorry

end matrix_multiplication_correct_l1404_140434
