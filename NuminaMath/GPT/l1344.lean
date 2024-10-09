import Mathlib

namespace grisha_cross_coloring_l1344_134407

open Nat

theorem grisha_cross_coloring :
  let grid_size := 40
  let cutout_rect_width := 36
  let cutout_rect_height := 37
  let total_cells := grid_size * grid_size
  let cutout_cells := cutout_rect_width * cutout_rect_height
  let remaining_cells := total_cells - cutout_cells
  let cross_cells := 5
  -- the result we need to prove is 113
  (remaining_cells - cross_cells - ((cutout_rect_width + cutout_rect_height - 1) - 1)) = 113 := by
  sorry

end grisha_cross_coloring_l1344_134407


namespace conclusion_2_conclusion_3_conclusion_4_l1344_134408

variable (b : ℝ)

def f (x : ℝ) : ℝ := x^2 - |b| * x - 3

theorem conclusion_2 (h_min : ∃ x, f b x = -3) : b = 0 :=
  sorry

theorem conclusion_3 (h_b : b = -2) (x : ℝ) (hx : -2 < x ∧ x < 2) :
    -4 ≤ f b x ∧ f b x ≤ -3 :=
  sorry

theorem conclusion_4 (hb_ne : b ≠ 0) (m : ℝ) (h_roots : ∃ x1 x2, f b x1 = m ∧ f b x2 = m ∧ x1 ≠ x2) :
    m > -3 ∨ b^2 = -4 * m - 12 :=
  sorry

end conclusion_2_conclusion_3_conclusion_4_l1344_134408


namespace solve_system_l1344_134456

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + y - z = 4 ∧ x^2 - y^2 + z^2 = -4 ∧ xyz = 6) ↔ 
    (x, y, z) = (2, 3, 1) ∨ (x, y, z) = (-1, 3, -2) :=
by
  sorry

end solve_system_l1344_134456


namespace ratio_2006_to_2005_l1344_134446

-- Conditions
def kids_in_2004 : ℕ := 60
def kids_in_2005 : ℕ := kids_in_2004 / 2
def kids_in_2006 : ℕ := 20

-- The statement to prove
theorem ratio_2006_to_2005 : 
  (kids_in_2006 : ℚ) / kids_in_2005 = 2 / 3 :=
sorry

end ratio_2006_to_2005_l1344_134446


namespace marbles_distribution_l1344_134464

theorem marbles_distribution (marbles children : ℕ) (h1 : marbles = 60) (h2 : children = 7) :
  ∃ k, k = 3 → (∀ i < children, marbles / children + (if i < marbles % children then 1 else 0) < 9) → k = 3 :=
by
  sorry

end marbles_distribution_l1344_134464


namespace anita_gave_apples_l1344_134491

theorem anita_gave_apples (initial_apples needed_for_pie apples_left_after_pie : ℝ)
  (h_initial : initial_apples = 10.0)
  (h_needed : needed_for_pie = 4.0)
  (h_left : apples_left_after_pie = 11.0) :
  ∃ (anita_apples : ℝ), anita_apples = 5 :=
by
  sorry

end anita_gave_apples_l1344_134491


namespace part_a_l1344_134468

theorem part_a (α β : ℝ) (h₁ : α = 1.0000000004) (h₂ : β = 1.00000000002) (h₃ : α > β) :
  2.00000000002 / (β * β + 2.00000000002) > 2.00000000004 / α := 
sorry

end part_a_l1344_134468


namespace girls_insects_collected_l1344_134487

theorem girls_insects_collected (boys_insects groups insects_per_group : ℕ) :
  boys_insects = 200 →
  groups = 4 →
  insects_per_group = 125 →
  (groups * insects_per_group) - boys_insects = 300 :=
by
  intros h1 h2 h3
  -- Prove the statement
  sorry

end girls_insects_collected_l1344_134487


namespace negative_number_among_options_l1344_134426

theorem negative_number_among_options :
  let A := abs (-1)
  let B := -(2^2)
  let C := (-(Real.sqrt 3))^2
  let D := (-3)^0
  B < 0 ∧ A > 0 ∧ C > 0 ∧ D > 0 :=
by
  sorry

end negative_number_among_options_l1344_134426


namespace employee_payment_correct_l1344_134430

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price increase percentage
def retail_increase_percentage : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount_percentage : ℝ := 0.30

-- Define the retail price as wholesale cost increased by the retail increase percentage
def retail_price : ℝ := wholesale_cost * (1 + retail_increase_percentage)

-- Define the discount amount as the retail price multiplied by the discount percentage
def discount_amount : ℝ := retail_price * employee_discount_percentage

-- Define the final employee payment as retail price minus the discount amount
def employee_final_payment : ℝ := retail_price - discount_amount

-- Theorem statement: Prove that the employee final payment equals $168
theorem employee_payment_correct : employee_final_payment = 168 := by
  sorry

end employee_payment_correct_l1344_134430


namespace part1_inequality_part2_range_of_a_l1344_134402

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 1)

-- Part (1)
theorem part1_inequality (x : ℝ) (h : f x 2 < 5) : -2 < x ∧ x < 3 := sorry

-- Part (2)
theorem part2_range_of_a (x a : ℝ) (h : ∀ x, f x a ≥ 4 - abs (a - 1)) : a ≤ -2 ∨ a ≥ 2 := sorry

end part1_inequality_part2_range_of_a_l1344_134402


namespace find_ab_l1344_134421

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
by
  sorry

end find_ab_l1344_134421


namespace angle_at_3_40_pm_is_130_degrees_l1344_134406

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end angle_at_3_40_pm_is_130_degrees_l1344_134406


namespace min_surface_area_of_stacked_solids_l1344_134485

theorem min_surface_area_of_stacked_solids :
  ∀ (l w h : ℕ), l = 3 → w = 2 → h = 1 → 
  (2 * (l * w + l * h + w * h) - 2 * l * w = 32) :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  sorry

end min_surface_area_of_stacked_solids_l1344_134485


namespace smallest_number_l1344_134484

-- Definitions of conditions for H, P, and S
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5
def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def satisfies_conditions_H (H : ℕ) : Prop :=
  is_cube (H / 2) ∧ is_fifth_power (H / 3) ∧ is_square (H / 5)

def satisfies_conditions_P (P A B C : ℕ) : Prop :=
  P / 2 = A^2 ∧ P / 3 = B^3 ∧ P / 5 = C^5

def satisfies_conditions_S (S D E F : ℕ) : Prop :=
  S / 2 = D^5 ∧ S / 3 = E^2 ∧ S / 5 = F^3

-- Main statement: Prove that P is the smallest number satisfying the conditions
theorem smallest_number (H P S A B C D E F : ℕ)
  (hH : satisfies_conditions_H H)
  (hP : satisfies_conditions_P P A B C)
  (hS : satisfies_conditions_S S D E F) :
  P ≤ H ∧ P ≤ S :=
  sorry

end smallest_number_l1344_134484


namespace construct_segment_eq_abc_div_de_l1344_134433

theorem construct_segment_eq_abc_div_de 
(a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
  ∃ x : ℝ, x = (a * b * c) / (d * e) :=
by sorry

end construct_segment_eq_abc_div_de_l1344_134433


namespace g_increasing_on_interval_l1344_134410

noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/5) * x + 13 * Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin ((1/5) * (x - 10 * Real.pi / 3) + 13 * Real.pi / 6)

theorem g_increasing_on_interval : ∀ x y : ℝ, (π ≤ x ∧ x < y ∧ y ≤ 2 * π) → g x < g y :=
by
  intro x y h
  -- Mathematical steps to prove this
  sorry

end g_increasing_on_interval_l1344_134410


namespace domain_of_k_l1344_134448

noncomputable def k (x : ℝ) := (1 / (x + 9)) + (1 / (x^2 + 9)) + (1 / (x^5 + 9)) + (1 / (x - 9))

theorem domain_of_k :
  ∀ x : ℝ, x ≠ -9 ∧ x ≠ -1.551 ∧ x ≠ 9 → ∃ y, y = k x := 
by
  sorry

end domain_of_k_l1344_134448


namespace focus_of_parabola_l1344_134477

theorem focus_of_parabola (focus : ℝ × ℝ) : 
  (∃ p : ℝ, y = p * x^2 / 2 → focus = (0, 1 / 2)) :=
by
  sorry

end focus_of_parabola_l1344_134477


namespace value_of_a_l1344_134400

theorem value_of_a (a : ℝ) (A : Set ℝ) (hA : A = {a^2, 1}) (h : 3 ∈ A) : 
  a = Real.sqrt 3 ∨ a = -Real.sqrt 3 :=
by
  sorry

end value_of_a_l1344_134400


namespace rate_of_current_l1344_134440

variable (c : ℝ)
def effective_speed_downstream (c : ℝ) : ℝ := 4.5 + c
def effective_speed_upstream (c : ℝ) : ℝ := 4.5 - c

theorem rate_of_current
  (h1 : ∀ d : ℝ, d / (4.5 - c) = 2 * (d / (4.5 + c)))
  : c = 1.5 :=
by
  sorry

end rate_of_current_l1344_134440


namespace avg_cans_used_per_game_l1344_134452

theorem avg_cans_used_per_game (total_rounds : ℕ) (games_first_round : ℕ) (games_second_round : ℕ)
  (games_third_round : ℕ) (games_finals : ℕ) (total_tennis_balls : ℕ) (balls_per_can : ℕ)
  (h1 : total_rounds = 4) (h2 : games_first_round = 8) (h3 : games_second_round = 4) 
  (h4 : games_third_round = 2) (h5 : games_finals = 1) (h6 : total_tennis_balls = 225) 
  (h7 : balls_per_can = 3) :
  let total_games := games_first_round + games_second_round + games_third_round + games_finals
  let total_cans_used := total_tennis_balls / balls_per_can
  let avg_cans_per_game := total_cans_used / total_games
  avg_cans_per_game = 5 :=
by {
  -- proof steps here
  sorry
}

end avg_cans_used_per_game_l1344_134452


namespace value_of_a9_l1344_134453

variables (a : ℕ → ℤ) (d : ℤ)
noncomputable def arithmetic_sequence : Prop :=
(a 1 + (a 1 + 10 * d)) / 2 = 15 ∧
a 1 + (a 1 + d) + (a 1 + 2 * d) = 9

theorem value_of_a9 (h : arithmetic_sequence a d) : a 9 = 24 :=
by sorry

end value_of_a9_l1344_134453


namespace original_decimal_number_l1344_134499

theorem original_decimal_number (x : ℝ) (h : 0.375 = (x / 1000) * 10) : x = 37.5 :=
sorry

end original_decimal_number_l1344_134499


namespace ratio_of_a_to_c_l1344_134461

theorem ratio_of_a_to_c
  {a b c : ℕ}
  (h1 : a / b = 11 / 3)
  (h2 : b / c = 1 / 5) :
  a / c = 11 / 15 :=
by 
  sorry

end ratio_of_a_to_c_l1344_134461


namespace infinitely_many_n_divisible_by_prime_l1344_134429

theorem infinitely_many_n_divisible_by_prime (p : ℕ) (hp : Prime p) : 
  ∃ᶠ n in at_top, p ∣ (2^n - n) :=
by {
  sorry
}

end infinitely_many_n_divisible_by_prime_l1344_134429


namespace number_of_bottle_caps_l1344_134466

def total_cost : ℝ := 25
def cost_per_bottle_cap : ℝ := 5

theorem number_of_bottle_caps : total_cost / cost_per_bottle_cap = 5 := 
by 
  sorry

end number_of_bottle_caps_l1344_134466


namespace smallest_possible_other_integer_l1344_134419

theorem smallest_possible_other_integer (n : ℕ) (h1 : Nat.lcm 60 n / Nat.gcd 60 n = 84) : n = 35 :=
sorry

end smallest_possible_other_integer_l1344_134419


namespace expand_expression_l1344_134489

theorem expand_expression (x : ℝ) : (2 * x - 3) * (2 * x + 3) * (4 * x ^ 2 + 9) = 4 * x ^ 4 - 81 := by
  sorry

end expand_expression_l1344_134489


namespace smallest_cost_l1344_134417

def gift1_choc := 3
def gift1_caramel := 15
def price1 := 350

def gift2_choc := 20
def gift2_caramel := 5
def price2 := 500

def equal_candies (m n : ℕ) : Prop :=
  gift1_choc * m + gift2_choc * n = gift1_caramel * m + gift2_caramel * n

def total_cost (m n : ℕ) : ℕ :=
  price1 * m + price2 * n

theorem smallest_cost :
  ∃ m n : ℕ, equal_candies m n ∧ total_cost m n = 3750 :=
by {
  sorry
}

end smallest_cost_l1344_134417


namespace necessary_but_not_sufficient_l1344_134488

-- Variables for the conditions
variables (x y : ℝ)

-- Conditions
def cond1 : Prop := x ≠ 1 ∨ y ≠ 4
def cond2 : Prop := x + y ≠ 5

-- Statement to prove the type of condition
theorem necessary_but_not_sufficient :
  cond2 x y → cond1 x y ∧ ¬(cond1 x y → cond2 x y) :=
sorry

end necessary_but_not_sufficient_l1344_134488


namespace probability_at_least_6_heads_8_flips_l1344_134470

-- Define the probability calculation of getting at least 6 heads in 8 coin flips.
def probability_at_least_6_heads (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k + Nat.choose n (k + 1) + Nat.choose n (k + 2)) / 2^n

theorem probability_at_least_6_heads_8_flips : 
  probability_at_least_6_heads 8 6 = 37 / 256 := 
by
  sorry

end probability_at_least_6_heads_8_flips_l1344_134470


namespace trigonometric_identity_proof_l1344_134492

theorem trigonometric_identity_proof (α : ℝ) (h : Real.tan α = 3) : (Real.sin (2 * α)) / ((Real.cos α) ^ 2) = 6 :=
by
  sorry

end trigonometric_identity_proof_l1344_134492


namespace candy_lollipops_l1344_134497

theorem candy_lollipops (κ c l : ℤ) 
  (h1 : κ = l + c - 8)
  (h2 : c = l + κ - 14) :
  l = 11 :=
by
  sorry

end candy_lollipops_l1344_134497


namespace greatest_integer_solution_l1344_134435

theorem greatest_integer_solution :
  ∃ x : ℤ, (∃ (k : ℤ), (8 : ℚ) / 11 > k / 15 ∧ k = 10) ∧ x = 10 :=
by {
  sorry
}

end greatest_integer_solution_l1344_134435


namespace nines_appear_600_times_l1344_134414

-- Define a function to count the number of nines in a single digit place
def count_nines_in_place (place_value : Nat) (range_start : Nat) (range_end : Nat) : Nat :=
  (range_end - range_start + 1) / 10 * place_value

-- Define a function to count the total number of nines in all places from 1 to 1000
def total_nines_1_to_1000 : Nat :=
  let counts_per_place := 3 * count_nines_in_place 100 1 1000
  counts_per_place

theorem nines_appear_600_times :
  total_nines_1_to_1000 = 600 :=
by
  -- Using specific step counts calculated in the problem
  have units_place := count_nines_in_place 100 1 1000
  have tens_place := units_place
  have hundreds_place := units_place
  have thousands_place := 0
  have total_nines := units_place + tens_place + hundreds_place + thousands_place
  sorry

end nines_appear_600_times_l1344_134414


namespace second_interest_rate_exists_l1344_134443

theorem second_interest_rate_exists (X Y : ℝ) (H : 0 < X ∧ X ≤ 10000) : ∃ Y, 8 * X + Y * (10000 - X) = 85000 :=
by
  sorry

end second_interest_rate_exists_l1344_134443


namespace number_of_distinct_real_roots_l1344_134465

theorem number_of_distinct_real_roots (f : ℝ → ℝ) (h : ∀ x, f x = |x| - (4 / x) - (3 * |x| / x)) : ∃ k, k = 1 :=
by
  sorry

end number_of_distinct_real_roots_l1344_134465


namespace inequality_transformation_l1344_134422

theorem inequality_transformation (x : ℝ) :
  x - 2 > 1 → x > 3 :=
by
  intro h
  linarith

end inequality_transformation_l1344_134422


namespace find_x_l1344_134413

theorem find_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) (h3 : x + y = 5) : 
  x = (7 + Real.sqrt 5) / 2 :=
by 
  sorry

end find_x_l1344_134413


namespace george_monthly_income_l1344_134437

theorem george_monthly_income (I : ℝ) (h : I / 2 - 20 = 100) : I = 240 := 
by sorry

end george_monthly_income_l1344_134437


namespace intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l1344_134415

variables (x y z : ℕ)

-- Conditions
axiom condition_1 : ∀ (t: ℕ), t = (6 : ℕ) → y * z = 6 * (y - x)
axiom condition_2 : ∀ (t: ℕ), t = (3 : ℕ) → y * z = 3 * (y + x)

-- Proof statements
theorem intervals_between_trolleybuses : z = 4 :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

theorem sportsman_slower_than_trolleybus : y = 3 * x :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

end intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l1344_134415


namespace frictional_force_is_correct_l1344_134418

-- Definitions
def m1 := 2.0 -- mass of the tank in kg
def m2 := 10.0 -- mass of the cart in kg
def a := 5.0 -- acceleration of the cart in m/s^2
def mu := 0.6 -- coefficient of friction between the tank and the cart
def g := 9.8 -- acceleration due to gravity in m/s^2

-- Frictional force acting on the tank
def frictional_force := mu * (m1 * g)

-- Required force to accelerate the tank with the cart
def required_force := m1 * a

-- Proof statement
theorem frictional_force_is_correct : required_force = 10 := 
by
  -- skipping the proof as specified
  sorry

end frictional_force_is_correct_l1344_134418


namespace total_distance_covered_l1344_134467

theorem total_distance_covered (d : ℝ) :
  (d / 5 + d / 10 + d / 15 + d / 20 + d / 25 = 15 / 60) → (5 * d = 375 / 137) :=
by
  intro h
  -- proof will go here
  sorry

end total_distance_covered_l1344_134467


namespace lcm_72_108_2100_l1344_134447

theorem lcm_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end lcm_72_108_2100_l1344_134447


namespace solve_system_l1344_134486

theorem solve_system :
  ∃ x y : ℚ, 3 * x - 2 * y = 5 ∧ 4 * x + 5 * y = 16 ∧ x = 57 / 23 ∧ y = 28 / 23 :=
by {
  sorry
}

end solve_system_l1344_134486


namespace set_intersection_l1344_134409

-- defining universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- defining set A
def A : Set ℕ := {1, 5, 9}

-- defining set B
def B : Set ℕ := {3, 7, 9}

-- complement of A in U
def complU (s : Set ℕ) := {x ∈ U | x ∉ s}

-- defining the intersection of complement of A with B
def intersection := complU A ∩ B

-- statement to be proved
theorem set_intersection : intersection = {3, 7} :=
by
  sorry

end set_intersection_l1344_134409


namespace lowest_possible_price_l1344_134476

theorem lowest_possible_price
  (MSRP : ℕ) (max_initial_discount_percent : ℕ) (platinum_discount_percent : ℕ)
  (h1 : MSRP = 35) (h2 : max_initial_discount_percent = 40) (h3 : platinum_discount_percent = 30) :
  let initial_discount := max_initial_discount_percent * MSRP / 100
  let price_after_initial_discount := MSRP - initial_discount
  let platinum_discount := platinum_discount_percent * price_after_initial_discount / 100
  let lowest_price := price_after_initial_discount - platinum_discount
  lowest_price = 147 / 10 :=
by
  sorry

end lowest_possible_price_l1344_134476


namespace multiple_of_3_iff_has_odd_cycle_l1344_134411

-- Define the undirected simple graph G
variable {V : Type} (G : SimpleGraph V)

-- Define the function f(G) which counts the number of acyclic orientations
def f (G : SimpleGraph V) : ℕ := sorry

-- Define what it means for a graph to have an odd-length cycle
def has_odd_cycle (G : SimpleGraph V) : Prop := sorry

-- The theorem statement
theorem multiple_of_3_iff_has_odd_cycle (G : SimpleGraph V) : 
  (f G) % 3 = 0 ↔ has_odd_cycle G := 
sorry

end multiple_of_3_iff_has_odd_cycle_l1344_134411


namespace fractional_cake_eaten_l1344_134472

def total_cake_eaten : ℚ :=
  1 / 3 + 1 / 3 + 1 / 6 + 1 / 12 + 1 / 24 + 1 / 48

theorem fractional_cake_eaten :
  total_cake_eaten = 47 / 48 := by
  sorry

end fractional_cake_eaten_l1344_134472


namespace total_cleaning_time_l1344_134439

def hose_time : ℕ := 10
def shampoos : ℕ := 3
def shampoo_time : ℕ := 15

theorem total_cleaning_time : hose_time + shampoos * shampoo_time = 55 := by
  sorry

end total_cleaning_time_l1344_134439


namespace initial_dimes_l1344_134434

theorem initial_dimes (x : ℕ) (h1 : x + 7 = 16) : x = 9 := by
  sorry

end initial_dimes_l1344_134434


namespace m_mul_m_add_1_not_power_of_integer_l1344_134463

theorem m_mul_m_add_1_not_power_of_integer (m n k : ℕ) : m * (m + 1) ≠ n^k :=
by
  sorry

end m_mul_m_add_1_not_power_of_integer_l1344_134463


namespace factorize_poly_l1344_134427

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end factorize_poly_l1344_134427


namespace square_perimeter_of_N_l1344_134450

theorem square_perimeter_of_N (area_M : ℝ) (area_N : ℝ) (side_N : ℝ) (perimeter_N : ℝ)
  (h1 : area_M = 100)
  (h2 : area_N = 4 * area_M)
  (h3 : area_N = side_N * side_N)
  (h4 : perimeter_N = 4 * side_N) :
  perimeter_N = 80 := 
sorry

end square_perimeter_of_N_l1344_134450


namespace volleyball_team_geography_l1344_134459

theorem volleyball_team_geography (total_players history_players both_subjects : ℕ) 
  (H1 : total_players = 15) 
  (H2 : history_players = 9) 
  (H3 : both_subjects = 4) : 
  ∃ (geography_players : ℕ), geography_players = 10 :=
by
  -- Definitions / Calculations
  -- Using conditions to derive the number of geography players
  let only_geography_players : ℕ := total_players - history_players
  let geography_players : ℕ := only_geography_players + both_subjects

  -- Prove the statement
  use geography_players
  sorry

end volleyball_team_geography_l1344_134459


namespace initial_customers_l1344_134431

theorem initial_customers (S : ℕ) (initial : ℕ) (H1 : initial = S + (S + 5)) (H2 : S = 3) : initial = 11 := 
by
  sorry

end initial_customers_l1344_134431


namespace pentagon_total_area_l1344_134412

-- Conditions definition
variables {a b c d e : ℕ}
variables {side1 side2 side3 side4 side5 : ℕ} 
variables {h : ℕ}
variables {triangle_area : ℕ}
variables {trapezoid_area : ℕ}
variables {total_area : ℕ}

-- Specific conditions given in the problem
def pentagon_sides (a b c d e : ℕ) : Prop :=
  a = 18 ∧ b = 25 ∧ c = 30 ∧ d = 28 ∧ e = 25

def can_be_divided (triangle_area trapezoid_area total_area : ℕ) : Prop :=
  triangle_area = 225 ∧ trapezoid_area = 770 ∧ total_area = 995

-- Total area of the pentagon under given conditions
theorem pentagon_total_area 
  (h_div: can_be_divided triangle_area trapezoid_area total_area) 
  (h_sides: pentagon_sides a b c d e)
  (h: triangle_area + trapezoid_area = total_area) :
  total_area = 995 := 
by
  sorry

end pentagon_total_area_l1344_134412


namespace remainder_of_2n_divided_by_11_l1344_134425

theorem remainder_of_2n_divided_by_11
  (n k : ℤ)
  (h : n = 22 * k + 12) :
  (2 * n) % 11 = 2 :=
by
  -- This is where the proof would go
  sorry

end remainder_of_2n_divided_by_11_l1344_134425


namespace range_of_a_l1344_134401

theorem range_of_a 
  (x1 x2 a : ℝ) 
  (h1 : x1 + x2 = 4) 
  (h2 : x1 * x2 = a) 
  (h3 : x1 > 1) 
  (h4 : x2 > 1) : 
  3 < a ∧ a ≤ 4 := 
sorry

end range_of_a_l1344_134401


namespace quadratic_function_value_l1344_134442

theorem quadratic_function_value (x1 x2 a b : ℝ) (h1 : a ≠ 0)
  (h2 : 2012 = a * x1^2 + b * x1 + 2009)
  (h3 : 2012 = a * x2^2 + b * x2 + 2009) :
  (a * (x1 + x2)^2 + b * (x1 + x2) + 2009) = 2009 :=
by
  sorry

end quadratic_function_value_l1344_134442


namespace functional_relationship_remaining_oil_after_4_hours_l1344_134490

-- Define the initial conditions and the functional form
def initial_oil : ℝ := 50
def consumption_rate : ℝ := 8
def remaining_oil (t : ℝ) : ℝ := initial_oil - consumption_rate * t

-- Prove the functional relationship and the remaining oil after 4 hours
theorem functional_relationship : ∀ (t : ℝ), remaining_oil t = 50 - 8 * t :=
by intros t
   exact rfl

theorem remaining_oil_after_4_hours : remaining_oil 4 = 18 :=
by simp [remaining_oil]
   norm_num
   sorry

end functional_relationship_remaining_oil_after_4_hours_l1344_134490


namespace find_z_l1344_134457

theorem find_z (x y z : ℚ) (h1 : x / (y + 1) = 4 / 5) (h2 : 3 * z = 2 * x + y) (h3 : y = 10) : 
  z = 46 / 5 := 
sorry

end find_z_l1344_134457


namespace rectangle_area_l1344_134436

-- Definitions of conditions
def width : ℝ := 5
def length : ℝ := 2 * width

-- The goal is to prove the area is 50 square inches given the length and width
theorem rectangle_area : length * width = 50 := by
  have h_length : length = 2 * width := by rfl
  have h_width : width = 5 := by rfl
  sorry

end rectangle_area_l1344_134436


namespace exponentiation_problem_l1344_134420

variable (x : ℝ) (m n : ℝ)

theorem exponentiation_problem (h1 : x ^ m = 5) (h2 : x ^ n = 1 / 4) :
  x ^ (2 * m - n) = 100 :=
sorry

end exponentiation_problem_l1344_134420


namespace polynomial_one_negative_root_iff_l1344_134441

noncomputable def polynomial_has_one_negative_real_root (p : ℝ) : Prop :=
  ∃ (x : ℝ), (x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1 = 0) ∧
  ∀ (y : ℝ), y < x → y^4 + 3*p*y^3 + 6*y^2 + 3*p*y + 1 ≠ 0

theorem polynomial_one_negative_root_iff (p : ℝ) :
  polynomial_has_one_negative_real_root p ↔ p ≥ 4 / 3 :=
sorry

end polynomial_one_negative_root_iff_l1344_134441


namespace hyperbola_parabola_shared_focus_l1344_134416

theorem hyperbola_parabola_shared_focus (a : ℝ) (h : a > 0) :
  (∃ b c : ℝ, b^2 = 3 ∧ c = 2 ∧ a^2 = c^2 - b^2 ∧ b ≠ 0) →
  a = 1 :=
by
  intro h_shared_focus
  sorry

end hyperbola_parabola_shared_focus_l1344_134416


namespace number_of_boxes_needed_l1344_134438

def total_bananas : ℕ := 40
def bananas_per_box : ℕ := 5

theorem number_of_boxes_needed : (total_bananas / bananas_per_box) = 8 := by
  sorry

end number_of_boxes_needed_l1344_134438


namespace discriminant_of_quadratic_is_321_l1344_134475

-- Define the quadratic equation coefficients
def a : ℝ := 4
def b : ℝ := -9
def c : ℝ := -15

-- Define the discriminant formula
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- The proof statement
theorem discriminant_of_quadratic_is_321 : discriminant a b c = 321 := by
  sorry

end discriminant_of_quadratic_is_321_l1344_134475


namespace smallest_even_number_l1344_134428

theorem smallest_even_number (n1 n2 n3 n4 n5 n6 n7 : ℤ) 
  (h_sum_seven : n1 + n2 + n3 + n4 + n5 + n6 + n7 = 700)
  (h_sum_first_three : n1 + n2 + n3 > 200)
  (h_consecutive : n2 = n1 + 2 ∧ n3 = n2 + 2 ∧ n4 = n3 + 2 ∧ n5 = n4 + 2 ∧ n6 = n5 + 2 ∧ n7 = n6 + 2) :
  n1 = 94 := 
sorry

end smallest_even_number_l1344_134428


namespace sum_of_consecutive_integers_345_l1344_134432

-- Definition of the conditions
def is_consecutive_sum (n : ℕ) (k : ℕ) (s : ℕ) : Prop :=
  s = k * n + k * (k - 1) / 2

-- Problem statement
theorem sum_of_consecutive_integers_345 :
  ∃ k_set : Finset ℕ, (∀ k ∈ k_set, k ≥ 2 ∧ ∃ n : ℕ, is_consecutive_sum n k 345) ∧ k_set.card = 6 :=
sorry

end sum_of_consecutive_integers_345_l1344_134432


namespace initial_money_l1344_134424

theorem initial_money (M : ℝ) (h1 : M - (1/4 * M) - (1/3 * (M - (1/4 * M))) = 1600) : M = 3200 :=
sorry

end initial_money_l1344_134424


namespace time_to_cross_tree_l1344_134482

def train_length : ℕ := 600
def platform_length : ℕ := 450
def time_to_pass_platform : ℕ := 105

-- Definition of the condition that leads to the speed of the train
def speed_of_train : ℚ := (train_length + platform_length) / time_to_pass_platform

-- Statement to prove the time to cross the tree
theorem time_to_cross_tree :
  (train_length : ℚ) / speed_of_train = 60 :=
by
  sorry

end time_to_cross_tree_l1344_134482


namespace total_water_capacity_of_coolers_l1344_134493

theorem total_water_capacity_of_coolers :
  ∀ (first_cooler second_cooler third_cooler : ℕ), 
  first_cooler = 100 ∧ 
  second_cooler = first_cooler + first_cooler / 2 ∧ 
  third_cooler = second_cooler / 2 -> 
  first_cooler + second_cooler + third_cooler = 325 := 
by
  intros first_cooler second_cooler third_cooler H
  cases' H with H1 H2
  cases' H2 with H3 H4
  sorry

end total_water_capacity_of_coolers_l1344_134493


namespace ada_originally_in_seat2_l1344_134481

inductive Seat
| S1 | S2 | S3 | S4 | S5 deriving Inhabited, DecidableEq

def moveRight : Seat → Option Seat
| Seat.S1 => some Seat.S2
| Seat.S2 => some Seat.S3
| Seat.S3 => some Seat.S4
| Seat.S4 => some Seat.S5
| Seat.S5 => none

def moveLeft : Seat → Option Seat
| Seat.S1 => none
| Seat.S2 => some Seat.S1
| Seat.S3 => some Seat.S2
| Seat.S4 => some Seat.S3
| Seat.S5 => some Seat.S4

structure FriendState :=
  (bea ceci dee edie : Seat)
  (ada_left : Bool) -- Ada is away for snacks, identified by her not being in the seat row.

def initial_seating := FriendState.mk Seat.S2 Seat.S3 Seat.S4 Seat.S5 true

def final_seating (init : FriendState) : FriendState :=
  let bea' := match moveRight init.bea with
              | some pos => pos
              | none => init.bea
  let ceci' := init.ceci -- Ceci moves left then back, net zero movement
  let (dee', edie') := match moveRight init.dee, init.dee with
                      | some new_ee, ed => (new_ee, ed) -- Dee and Edie switch and Edie moves right
                      | _, _ => (init.dee, init.edie) -- If moves are invalid
  FriendState.mk bea' ceci' dee' edie' init.ada_left

theorem ada_originally_in_seat2 (init : FriendState) : init = initial_seating → final_seating init ≠ initial_seating → init.bea = Seat.S2 :=
by
  intro h_init h_finalne
  sorry -- Proof steps go here

end ada_originally_in_seat2_l1344_134481


namespace number_of_ways_2020_l1344_134494

-- We are defining b_i explicitly restricted by the conditions in the problem.
def b (i : ℕ) : ℕ :=
  sorry

-- Given conditions
axiom h_bounds : ∀ i, 0 ≤ b i ∧ b i ≤ 99
axiom h_indices : ∀ (i : ℕ), i < 4

-- Main theorem statement
theorem number_of_ways_2020 (M : ℕ) 
  (h : 2020 = b 3 * 1000 + b 2 * 100 + b 1 * 10 + b 0) 
  (htotal : M = 203) : 
  M = 203 :=
  by 
    sorry

end number_of_ways_2020_l1344_134494


namespace min_n_probability_l1344_134479

-- Define the number of members in teams
def num_members (n : ℕ) : ℕ := n

-- Define the total number of handshakes
def total_handshakes (n : ℕ) : ℕ := n * n

-- Define the number of ways to choose 2 handshakes from total handshakes
def choose_two_handshakes (n : ℕ) : ℕ := (total_handshakes n).choose 2

-- Define the number of ways to choose event A (involves exactly 3 different members)
def event_a_count (n : ℕ) : ℕ := 2 * n.choose 1 * (n - 1).choose 1

-- Define the probability of event A
def probability_event_a (n : ℕ) : ℚ := (event_a_count n : ℚ) / (choose_two_handshakes n : ℚ)

-- The minimum value of n such that the probability of event A is less than 1/10
theorem min_n_probability :
  ∃ n : ℕ, (probability_event_a n < (1 : ℚ) / 10) ∧ n ≥ 20 :=
by {
  sorry
}

end min_n_probability_l1344_134479


namespace distance_from_point_to_origin_l1344_134458

theorem distance_from_point_to_origin (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  (Real.sqrt (x^2 + y^2)) = 5 := by
  sorry

end distance_from_point_to_origin_l1344_134458


namespace negative_two_squared_l1344_134483

theorem negative_two_squared :
  (-2 : ℤ)^2 = 4 := 
sorry

end negative_two_squared_l1344_134483


namespace circle_diameter_length_l1344_134469

theorem circle_diameter_length (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 :=
by
  -- Placeholder for proof
  sorry

end circle_diameter_length_l1344_134469


namespace divisibility_criterion_l1344_134423

theorem divisibility_criterion (n : ℕ) : 
  (20^n - 13^n - 7^n) % 309 = 0 ↔ 
  ∃ k : ℕ, n = 1 + 6 * k ∨ n = 5 + 6 * k := 
  sorry

end divisibility_criterion_l1344_134423


namespace fence_cost_l1344_134495

theorem fence_cost (area : ℝ) (price_per_foot : ℝ) (side_length perimeter cost : ℝ) 
  (h1 : area = 289) 
  (h2 : price_per_foot = 55)
  (h3 : side_length = Real.sqrt area)
  (h4 : perimeter = 4 * side_length)
  (h5 : cost = perimeter * price_per_foot) :
  cost = 3740 := 
sorry

end fence_cost_l1344_134495


namespace ratio_of_boys_l1344_134462

variables {b g o : ℝ}

theorem ratio_of_boys (h1 : b = (1/2) * o)
  (h2 : g = o - b)
  (h3 : b + g + o = 1) :
  b = 1 / 4 :=
by
  sorry

end ratio_of_boys_l1344_134462


namespace tan_alpha_l1344_134405

theorem tan_alpha {α : ℝ} (h : Real.tan (α + π / 4) = 9) : Real.tan α = 4 / 5 :=
sorry

end tan_alpha_l1344_134405


namespace betta_fish_count_l1344_134444

theorem betta_fish_count 
  (total_guppies_per_day : ℕ) 
  (moray_eel_consumption : ℕ) 
  (betta_fish_consumption : ℕ) 
  (betta_fish_count : ℕ) 
  (h_total : total_guppies_per_day = 55)
  (h_eel : moray_eel_consumption = 20)
  (h_betta : betta_fish_consumption = 7) 
  (h_eq : total_guppies_per_day - moray_eel_consumption = betta_fish_consumption * betta_fish_count) : 
  betta_fish_count = 5 :=
by 
  sorry

end betta_fish_count_l1344_134444


namespace find_x_l1344_134460

noncomputable def x : ℝ := 10.3

theorem find_x (h1 : x + (⌈x⌉ : ℝ) = 21.3) (h2 : x > 0) : x = 10.3 :=
sorry

end find_x_l1344_134460


namespace cost_of_article_l1344_134404

theorem cost_of_article (C: ℝ) (G: ℝ) (h1: 380 = C + G) (h2: 420 = C + G + 0.05 * C) : C = 800 :=
by
  sorry

end cost_of_article_l1344_134404


namespace jennifer_book_spending_l1344_134451

variable (initial_total : ℕ)
variable (spent_sandwich : ℚ)
variable (spent_museum : ℚ)
variable (money_left : ℕ)

theorem jennifer_book_spending :
  initial_total = 90 → 
  spent_sandwich = 1/5 * 90 → 
  spent_museum = 1/6 * 90 → 
  money_left = 12 →
  (initial_total - money_left - (spent_sandwich + spent_museum)) / initial_total = 1/2 :=
by
  intros h_initial_total h_spent_sandwich h_spent_museum h_money_left
  sorry

end jennifer_book_spending_l1344_134451


namespace percent_difference_z_w_l1344_134473

theorem percent_difference_z_w (w x y z : ℝ)
  (h1 : w = 0.60 * x)
  (h2 : x = 0.60 * y)
  (h3 : z = 0.54 * y) :
  (z - w) / w * 100 = 50 := by
sorry

end percent_difference_z_w_l1344_134473


namespace polygon_sides_l1344_134471

theorem polygon_sides (h : ∀ (n : ℕ), (180 * (n - 2)) / n = 150) : n = 12 :=
by
  sorry

end polygon_sides_l1344_134471


namespace triangle_AB_C_min_perimeter_l1344_134474

noncomputable def minimum_perimeter (a b c : ℕ) (A B C : ℝ) : ℝ := a + b + c

theorem triangle_AB_C_min_perimeter
  (a b c : ℕ)
  (A B C : ℝ)
  (h1 : A = 2 * B)
  (h2 : C > π / 2)
  (h3 : a^2 = b * (b + c))
  (h4 : ∀ x : ℕ, x > 0 → a ≠ 0)
  (h5 :  a + b > c ∧ a + c > b ∧ b + c > a) :
  minimum_perimeter a b c A B C = 77 := 
sorry

end triangle_AB_C_min_perimeter_l1344_134474


namespace geometric_sequence_common_ratio_l1344_134449

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_relation : ∀ n, a n = (1 / 2) * (a (n + 1) + a (n + 2))) (h_positive : ∀ n, a n > 0) : r = 1 :=
sorry

end geometric_sequence_common_ratio_l1344_134449


namespace all_numbers_rational_l1344_134498

-- Define the mathematical operations for the problem
def fourth_root (x : ℝ) : ℝ := x ^ (1 / 4)
def square_root (x : ℝ) : ℝ := x ^ (1 / 2)
def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem all_numbers_rational :
    (∃ x1 : ℚ, fourth_root 81 = x1) ∧
    (∃ x2 : ℚ, square_root 0.64 = x2) ∧
    (∃ x3 : ℚ, cube_root 0.001 = x3) ∧
    (∃ x4 : ℚ, (cube_root 8) * (square_root ((0.25)⁻¹)) = x4) :=
  sorry

end all_numbers_rational_l1344_134498


namespace compute_expression_l1344_134478

theorem compute_expression (x : ℝ) (h : x + (1 / x) = 7) :
  (x - 3)^2 + (49 / (x - 3)^2) = 23 :=
by
  sorry

end compute_expression_l1344_134478


namespace spinner_prob_l1344_134455

theorem spinner_prob (PD PE PF_PG : ℚ) (hD : PD = 1/4) (hE : PE = 1/3) 
  (hTotal : PD + PE + PF_PG = 1) : PF_PG = 5/12 := by
  sorry

end spinner_prob_l1344_134455


namespace compare_sine_values_1_compare_sine_values_2_l1344_134480

theorem compare_sine_values_1 (h1 : 0 < Real.pi / 10) (h2 : Real.pi / 10 < Real.pi / 8) (h3 : Real.pi / 8 < Real.pi / 2) :
  Real.sin (- Real.pi / 10) > Real.sin (- Real.pi / 8) :=
by
  sorry

theorem compare_sine_values_2 (h1 : 0 < Real.pi / 8) (h2 : Real.pi / 8 < 3 * Real.pi / 8) (h3 : 3 * Real.pi / 8 < Real.pi / 2) :
  Real.sin (7 * Real.pi / 8) < Real.sin (5 * Real.pi / 8) :=
by
  sorry

end compare_sine_values_1_compare_sine_values_2_l1344_134480


namespace range_of_x_satisfying_inequality_l1344_134496

def f (x : ℝ) : ℝ := (x - 1) ^ 4 + 2 * |x - 1|

theorem range_of_x_satisfying_inequality :
  {x : ℝ | f x > f (2 * x)} = {x : ℝ | 0 < x ∧ x < (2 : ℝ) / 3} :=
by
  sorry

end range_of_x_satisfying_inequality_l1344_134496


namespace total_cost_898_8_l1344_134403

theorem total_cost_898_8 :
  ∀ (M R F : ℕ → ℝ), 
    (10 * M 1 = 24 * R 1) →
    (6 * F 1 = 2 * R 1) →
    (F 1 = 21) →
    (4 * M 1 + 3 * R 1 + 5 * F 1 = 898.8) :=
by
  intros M R F h1 h2 h3
  sorry

end total_cost_898_8_l1344_134403


namespace find_a_plus_b_l1344_134454

theorem find_a_plus_b (a b : ℝ) (x y : ℝ) 
  (h1 : x = 2) 
  (h2 : y = -1) 
  (h3 : a * x - 2 * y = 4) 
  (h4 : 3 * x + b * y = -7) : a + b = 14 := 
by 
  -- Begin the proof
  sorry

end find_a_plus_b_l1344_134454


namespace abs_diff_of_two_numbers_l1344_134445

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) : |x - y| = 2 * Real.sqrt 65 :=
by
  sorry

end abs_diff_of_two_numbers_l1344_134445
