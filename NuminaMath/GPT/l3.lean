import Mathlib

namespace sum_ineq_l3_3704

theorem sum_ineq (x y z t : ℝ) (h₁ : x + y + z + t = 0) (h₂ : x^2 + y^2 + z^2 + t^2 = 1) :
  -1 ≤ x * y + y * z + z * t + t * x ∧ x * y + y * z + z * t + t * x ≤ 0 :=
by
  sorry

end sum_ineq_l3_3704


namespace surface_area_of_figure_l3_3291

theorem surface_area_of_figure 
  (block_surface_area : ℕ) 
  (loss_per_block : ℕ) 
  (number_of_blocks : ℕ) 
  (effective_surface_area : ℕ)
  (total_surface_area : ℕ) 
  (h_block : block_surface_area = 18) 
  (h_loss : loss_per_block = 2) 
  (h_blocks : number_of_blocks = 4) 
  (h_effective : effective_surface_area = block_surface_area - loss_per_block) 
  (h_total : total_surface_area = number_of_blocks * effective_surface_area) : 
  total_surface_area = 64 :=
by
  sorry

end surface_area_of_figure_l3_3291


namespace total_cost_of_hats_l3_3547

-- Definition of conditions
def weeks := 2
def days_per_week := 7
def cost_per_hat := 50

-- Definition of the number of hats
def num_hats := weeks * days_per_week

-- Statement of the problem
theorem total_cost_of_hats : num_hats * cost_per_hat = 700 := 
by sorry

end total_cost_of_hats_l3_3547


namespace f_decreasing_ln_inequality_limit_inequality_l3_3877

-- Definitions of the given conditions
noncomputable def f (x : ℝ) : ℝ := (Real.log (1 + x)) / x

-- Statements we need to prove

-- (I) Prove that f(x) is decreasing on (0, +∞)
theorem f_decreasing : ∀ x y : ℝ, 0 < x → x < y → f y < f x := sorry

-- (II) Prove that for the inequality ln(1 + x) < ax to hold for all x in (0, +∞), a must be at least 1
theorem ln_inequality (a : ℝ) : (∀ x : ℝ, 0 < x → Real.log (1 + x) < a * x) ↔ 1 ≤ a := sorry

-- (III) Prove that (1 + 1/n)^n < e for all n in ℕ*
theorem limit_inequality (n : ℕ) (h : n ≠ 0) : (1 + 1 / n) ^ n < Real.exp 1 := sorry

end f_decreasing_ln_inequality_limit_inequality_l3_3877


namespace max_value_x2_y3_z_l3_3948

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  if x + y + z = 3 then x^2 * y^3 * z else 0

theorem max_value_x2_y3_z
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x + y + z = 3) :
  maximum_value x y z ≤ 9 / 16 := sorry

end max_value_x2_y3_z_l3_3948


namespace smallest_M_exists_l3_3742

theorem smallest_M_exists :
  ∃ M : ℕ, M = 249 ∧
  (∃ k1 : ℕ, (M + k1 = 8 * k1 ∨ M + k1 + 1 = 8 * k1 ∨ M + k1 + 2 = 8 * k1)) ∧
  (∃ k2 : ℕ, (M + k2 = 27 * k2 ∨ M + k2 + 1 = 27 * k2 ∨ M + k2 + 2 = 27 * k2)) ∧
  (∃ k3 : ℕ, (M + k3 = 125 * k3 ∨ M + k3 + 1 = 125 * k3 ∨ M + k3 + 2 = 125 * k3)) :=
by
  sorry

end smallest_M_exists_l3_3742


namespace f_minimum_positive_period_and_max_value_l3_3240

noncomputable def f (x : ℝ) : ℝ := (Real.sin x * Real.cos x) + (1 + (Real.tan x)^2) * (Real.cos x)^2

theorem f_minimum_positive_period_and_max_value :
  (∀ T > 0, (∀ x : ℝ, f (x + T) = f x) → T ≥ π) ∧ (∃ M, ∀ x : ℝ, f x ≤ M ∧ M = 3 / 2) := by
  sorry

end f_minimum_positive_period_and_max_value_l3_3240


namespace largest_prime_number_largest_composite_number_l3_3605

-- Definitions of prime and composite
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

-- Largest prime and composite numbers less than 20
def largest_prime_less_than_20 := 19
def largest_composite_less_than_20 := 18

theorem largest_prime_number : 
  largest_prime_less_than_20 = 19 ∧ is_prime 19 ∧ 
  (∀ n : ℕ, n < 20 → is_prime n → n < 19) := 
by sorry

theorem largest_composite_number : 
  largest_composite_less_than_20 = 18 ∧ is_composite 18 ∧ 
  (∀ n : ℕ, n < 20 → is_composite n → n < 18) := 
by sorry

end largest_prime_number_largest_composite_number_l3_3605


namespace find_value_l3_3059

theorem find_value (x y : ℝ) (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 5 * x^2 + 11 * x * y + 5 * y^2 = 89 :=
by
  sorry

end find_value_l3_3059


namespace g_at_2_l3_3983

def g (x : ℝ) : ℝ := x^3 - x

theorem g_at_2 : g 2 = 6 :=
by
  sorry

end g_at_2_l3_3983


namespace total_rainfall_November_l3_3735

def rain_first_15_days : ℕ := 4

def days_first_15 : ℕ := 15

def rain_last_15_days : ℕ := 2 * rain_first_15_days

def days_last_15 : ℕ := 15

def total_rainfall : ℕ := 
  (rain_first_15_days * days_first_15) + (rain_last_15_days * days_last_15)

theorem total_rainfall_November : total_rainfall = 180 := by
  sorry

end total_rainfall_November_l3_3735


namespace max_a4b2c_l3_3852

-- Define the conditions and required statement
theorem max_a4b2c (a b c : ℝ) (h1: 0 < a) (h2: 0 < b) (h3: 0 < c) (h4: a + b + c = 1) :
    a^4 * b^2 * c ≤ 1024 / 117649 :=
sorry

end max_a4b2c_l3_3852


namespace cost_of_saddle_l3_3922

theorem cost_of_saddle (S : ℝ) (H : 4 * S + S = 5000) : S = 1000 :=
by sorry

end cost_of_saddle_l3_3922


namespace shift_upwards_l3_3048

theorem shift_upwards (a : ℝ) :
  (∀ x : ℝ, y = -2 * x + a) -> (a = 1) :=
by
  sorry

end shift_upwards_l3_3048


namespace x_intercept_of_line_l3_3498

theorem x_intercept_of_line :
  (∃ x : ℝ, 5 * x - 7 * 0 = 35 ∧ (x, 0) = (7, 0)) :=
by
  use 7
  simp
  sorry

end x_intercept_of_line_l3_3498


namespace arithmetic_sequence_b1_l3_3708

theorem arithmetic_sequence_b1 
  (b : ℕ → ℝ) 
  (U : ℕ → ℝ)
  (U2023 : ℝ) 
  (b2023 : ℝ)
  (hb2023 : b 2023 = b 1 + 2022 * (b 2 - b 1))
  (hU2023 : U 2023 = 2023 * (b 1 + 1011 * (b 2 - b 1))) 
  (hUn : ∀ n, U n = (n * (2 * b 1 + (n - 1) * (b 2 - b 1)) / 2)) :
  b 1 = (U 2023 - 2023 * b 2023) / 2023 :=
by
  sorry

end arithmetic_sequence_b1_l3_3708


namespace correct_division_l3_3611

theorem correct_division (x : ℝ) (h : 8 * x + 8 = 56) : x / 8 = 0.75 :=
by
  sorry

end correct_division_l3_3611


namespace charles_housesitting_hours_l3_3606

theorem charles_housesitting_hours :
  ∀ (earnings_per_hour_housesitting earnings_per_hour_walking_dog number_of_dogs_walked total_earnings : ℕ),
  earnings_per_hour_housesitting = 15 →
  earnings_per_hour_walking_dog = 22 →
  number_of_dogs_walked = 3 →
  total_earnings = 216 →
  ∃ h : ℕ, 15 * h + 22 * 3 = 216 ∧ h = 10 :=
by
  intros
  sorry

end charles_housesitting_hours_l3_3606


namespace jellybean_probability_l3_3249

theorem jellybean_probability :
  let total_jellybeans := 15
  let green_jellybeans := 6
  let purple_jellybeans := 2
  let yellow_jellybeans := 7
  let total_picked := 4
  let total_ways := Nat.choose total_jellybeans total_picked
  let ways_to_pick_two_yellow := Nat.choose yellow_jellybeans 2
  let ways_to_pick_two_non_yellow := Nat.choose (total_jellybeans - yellow_jellybeans) 2
  let successful_outcomes := ways_to_pick_two_yellow * ways_to_pick_two_non_yellow
  let probability := successful_outcomes / total_ways
  probability = 4 / 9 := by
sorry

end jellybean_probability_l3_3249


namespace average_marks_is_70_l3_3468

variable (P C M : ℕ)

-- Condition: The total marks in physics, chemistry, and mathematics is 140 more than the marks in physics
def total_marks_condition : Prop := P + C + M = P + 140

-- Definition of the average marks in chemistry and mathematics
def average_marks_C_M : ℕ := (C + M) / 2

theorem average_marks_is_70 (h : total_marks_condition P C M) : average_marks_C_M C M = 70 :=
sorry

end average_marks_is_70_l3_3468


namespace folded_strip_fit_l3_3554

open Classical

noncomputable def canFitAfterFolding (r : ℝ) (strip : Set (ℝ × ℝ)) (folded_strip : Set (ℝ × ℝ)) : Prop :=
  ∀ p : ℝ × ℝ, p ∈ folded_strip → (p.1^2 + p.2^2 ≤ r^2)

theorem folded_strip_fit {r : ℝ} {strip folded_strip : Set (ℝ × ℝ)} :
  (∀ p : ℝ × ℝ, p ∈ strip → (p.1^2 + p.2^2 ≤ r^2)) →
  (∀ q : ℝ × ℝ, q ∈ folded_strip → (∃ p : ℝ × ℝ, p ∈ strip ∧ q = p)) →
  canFitAfterFolding r strip folded_strip :=
by
  intros hs hf
  sorry

end folded_strip_fit_l3_3554


namespace wrapping_paper_area_correct_l3_3909

noncomputable def wrapping_paper_area (l w h : ℝ) (hlw : l ≥ w) : ℝ :=
  (l + 2*h)^2

theorem wrapping_paper_area_correct (l w h : ℝ) (hlw : l ≥ w) :
  wrapping_paper_area l w h hlw = (l + 2*h)^2 :=
by
  sorry

end wrapping_paper_area_correct_l3_3909


namespace find_x8_l3_3968

theorem find_x8 (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 :=
by sorry

end find_x8_l3_3968


namespace find_number_l3_3914

theorem find_number 
  (a b c d : ℤ)
  (h1 : 0 ≤ a ∧ a ≤ 9)
  (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : 0 ≤ c ∧ c ≤ 9)
  (h4 : 0 ≤ d ∧ d ≤ 9)
  (h5 : 6 * a + 9 * b + 3 * c + d = 88)
  (h6 : a - b + c - d = -6)
  (h7 : a - 9 * b + 3 * c - d = -46) : 
  1000 * a + 100 * b + 10 * c + d = 6507 := 
sorry

end find_number_l3_3914


namespace isosceles_triangle_area_l3_3039

theorem isosceles_triangle_area {a b h : ℝ} (h1 : a = 13) (h2 : b = 13) (h3 : h = 10) :
  ∃ (A : ℝ), A = 60 ∧ A = (1 / 2) * h * 12 :=
by
  sorry

end isosceles_triangle_area_l3_3039


namespace cos2alpha_plus_sin2alpha_l3_3769

theorem cos2alpha_plus_sin2alpha (α : Real) (h : Real.tan (Real.pi + α) = 2) : 
  Real.cos (2 * α) + Real.sin (2 * α) = 1 / 5 :=
sorry

end cos2alpha_plus_sin2alpha_l3_3769


namespace find_a_l3_3296

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 * Real.exp x

theorem find_a (a : ℝ) : (∀ x : ℝ, -1 < x ∧ x < 1 → (x - a) * (x - a + 2) ≤ 0) → a = 1 :=
by
  intro h
  sorry 

end find_a_l3_3296


namespace sequence_properties_l3_3174

theorem sequence_properties (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, (a n)^2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0) :
  a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_properties_l3_3174


namespace factorize_expression_l3_3974

theorem factorize_expression (m : ℝ) : m^2 + 3 * m = m * (m + 3) :=
by
  sorry

end factorize_expression_l3_3974


namespace water_for_1200ml_flour_l3_3042

-- Define the condition of how much water is mixed with a specific amount of flour
def water_per_flour (flour water : ℕ) : Prop :=
  water = (flour / 400) * 100

-- Given condition: Maria uses 100 mL of water for every 400 mL of flour
def condition : Prop := water_per_flour 400 100

-- Problem Statement: How many mL of water for 1200 mL of flour?
theorem water_for_1200ml_flour (h : condition) : water_per_flour 1200 300 :=
sorry

end water_for_1200ml_flour_l3_3042


namespace cupboard_cost_price_l3_3515

theorem cupboard_cost_price (C SP NSP : ℝ) (h1 : SP = 0.84 * C) (h2 : NSP = 1.16 * C) (h3 : NSP = SP + 1200) : C = 3750 :=
by
  sorry

end cupboard_cost_price_l3_3515


namespace students_came_to_school_l3_3889

theorem students_came_to_school (F M T A : ℕ) 
    (hF : F = 658)
    (hM : M = F - 38)
    (hA : A = 17)
    (hT : T = M + F - A) :
    T = 1261 := by 
sorry

end students_came_to_school_l3_3889


namespace part1_part2_l3_3791

-- Definitions of sets A and B
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 3 - 2 * a }

-- Part 1: Prove that (complement of A union B = Universal Set) implies a in (-∞, 0]
theorem part1 (U : Set ℝ) (hU : (Aᶜ ∪ B a) = U) : a ≤ 0 := sorry

-- Part 2: Prove that (A intersection B = B) implies a in [1/2, ∞)
theorem part2 (h : (A ∩ B a) = B a) : 1/2 ≤ a := sorry

end part1_part2_l3_3791


namespace rectangle_area_l3_3406

theorem rectangle_area (w l : ℝ) (hw : w = 2) (hl : l = 3) : w * l = 6 := by
  sorry

end rectangle_area_l3_3406


namespace submarine_rise_l3_3206

theorem submarine_rise (initial_depth final_depth : ℤ) (h_initial : initial_depth = -27) (h_final : final_depth = -18) :
  final_depth - initial_depth = 9 :=
by
  rw [h_initial, h_final]
  norm_num 

end submarine_rise_l3_3206


namespace john_average_speed_l3_3211

theorem john_average_speed :
  let distance_uphill := 2 -- distance in km
  let distance_downhill := 2 -- distance in km
  let time_uphill := 45 / 60 -- time in hours (45 minutes)
  let time_downhill := 15 / 60 -- time in hours (15 minutes)
  let total_distance := distance_uphill + distance_downhill -- total distance in km
  let total_time := time_uphill + time_downhill -- total time in hours
  total_distance / total_time = 4 := by
  sorry

end john_average_speed_l3_3211


namespace Leah_coins_value_in_cents_l3_3466

theorem Leah_coins_value_in_cents (p n : ℕ) (h₁ : p + n = 15) (h₂ : p = n + 2) : p + 5 * n = 44 :=
by
  sorry

end Leah_coins_value_in_cents_l3_3466


namespace factor_t_squared_minus_144_l3_3164

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end factor_t_squared_minus_144_l3_3164


namespace equal_area_division_l3_3998

theorem equal_area_division (d : ℝ) : 
  (∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 
   (x = d ∨ x = 4) ∧ (y = 4 ∨ y = 0) ∧ 
   (2 : ℝ) * (4 - d) = 4) ↔ d = 2 :=
by
  sorry

end equal_area_division_l3_3998


namespace triangle_incenter_equilateral_l3_3308

theorem triangle_incenter_equilateral (a b c : ℝ) (h : (b + c) / a = (a + c) / b ∧ (a + c) / b = (a + b) / c) : a = b ∧ b = c :=
by
  sorry

end triangle_incenter_equilateral_l3_3308


namespace parts_processed_per_day_l3_3693

-- Given conditions
variable (a : ℕ)

-- Goal: Prove the daily productivity of Master Wang given the conditions
theorem parts_processed_per_day (h1 : ∀ n, n = 8) (h2 : ∃ m, m = a + 3):
  (a + 3) / 8 = (a + 3) / 8 :=
by
  sorry

end parts_processed_per_day_l3_3693


namespace describe_S_l3_3177

def S : Set (ℝ × ℝ) :=
  { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) }

theorem describe_S :
  S = { p : ℝ × ℝ | (p.2 ≤ 11 ∧ p.1 = 2) ∨ (p.1 ≤ 2 ∧ p.2 = 11) ∨ (p.1 ≥ 2 ∧ p.2 = p.1 + 9) } := 
by
  -- proof is omitted
  sorry

end describe_S_l3_3177


namespace solve_equation_1_solve_equation_2_l3_3621

theorem solve_equation_1 (x : ℚ) : 1 - (1 / (x - 5)) = (x / (x + 5)) → x = 15 / 2 := 
by
  sorry

theorem solve_equation_2 (x : ℚ) : (3 / (x - 1)) - (2 / (x + 1)) = (1 / (x^2 - 1)) → x = -4 := 
by
  sorry

end solve_equation_1_solve_equation_2_l3_3621


namespace monotonic_f_on_interval_l3_3930

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x + Real.pi / 10) - 2

theorem monotonic_f_on_interval : 
  ∀ x y : ℝ, 
    x ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    y ∈ Set.Icc (Real.pi / 2) (7 * Real.pi / 5) → 
    x ≤ y → 
    f x ≤ f y :=
sorry

end monotonic_f_on_interval_l3_3930


namespace soccer_team_players_l3_3979

theorem soccer_team_players
  (first_half_starters : ℕ)
  (first_half_subs : ℕ)
  (second_half_mult : ℕ)
  (did_not_play : ℕ)
  (players_prepared : ℕ) :
  first_half_starters = 11 →
  first_half_subs = 2 →
  second_half_mult = 2 →
  did_not_play = 7 →
  players_prepared = 20 :=
by
  -- Proof steps go here
  sorry

end soccer_team_players_l3_3979


namespace work_completion_days_l3_3580

-- Definitions based on the conditions
def A_work_days : ℕ := 20
def B_work_days : ℕ := 30
def C_work_days : ℕ := 10  -- Twice as fast as A, and A can do it in 20 days, hence 10 days.
def together_work_days : ℕ := 12
def B_C_half_day_rate : ℚ := (1 / B_work_days) / 2 + (1 / C_work_days) / 2  -- rate per half day for both B and C
def A_full_day_rate : ℚ := 1 / A_work_days  -- rate per full day for A

-- Converting to rate per day when B and C work only half day daily
def combined_rate_per_day_with_BC_half : ℚ := A_full_day_rate + B_C_half_day_rate

-- The main theorem to prove
theorem work_completion_days 
  (A_work_days B_work_days C_work_days together_work_days : ℕ)
  (C_work_days_def : C_work_days = A_work_days / 2) 
  (total_days_def : 1 / combined_rate_per_day_with_BC_half = 60 / 7) :
  (1 / combined_rate_per_day_with_BC_half) = 60 / 7 :=
sorry

end work_completion_days_l3_3580


namespace shifted_sine_odd_function_l3_3680

theorem shifted_sine_odd_function (ϕ : ℝ) (hϕ : 0 < ϕ ∧ ϕ < π) :
  ∃ k : ℤ, ϕ = (2 * π / 3) + k * π ∧ 0 < (2 * π / 3) + k * π ∧ (2 * π / 3) + k * π < π :=
sorry

end shifted_sine_odd_function_l3_3680


namespace dogwood_trees_initial_count_l3_3100

theorem dogwood_trees_initial_count 
  (dogwoods_today : ℕ) 
  (dogwoods_tomorrow : ℕ) 
  (final_dogwoods : ℕ)
  (total_planted : ℕ := dogwoods_today + dogwoods_tomorrow)
  (initial_dogwoods := final_dogwoods - total_planted)
  (h : dogwoods_today = 41)
  (h1 : dogwoods_tomorrow = 20)
  (h2 : final_dogwoods = 100) : 
  initial_dogwoods = 39 := 
by sorry

end dogwood_trees_initial_count_l3_3100


namespace no_positive_integer_makes_expression_integer_l3_3717

theorem no_positive_integer_makes_expression_integer : 
  ∀ n : ℕ, n > 0 → ¬ ∃ k : ℤ, (n^(3 * n - 2) - 3 * n + 1) = k * (3 * n - 2) := 
by 
  intro n hn
  sorry

end no_positive_integer_makes_expression_integer_l3_3717


namespace green_apples_count_l3_3907

-- Definitions for the conditions in the problem
def total_apples : ℕ := 19
def red_apples : ℕ := 3
def yellow_apples : ℕ := 14

-- Statement expressing that the number of green apples on the table is 2
theorem green_apples_count : (total_apples - red_apples - yellow_apples = 2) :=
by
  sorry

end green_apples_count_l3_3907


namespace value_of_k_l3_3815

theorem value_of_k :
  ∃ k, k = 2 ∧ (∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 5 ∧
                ∀ (s t : ℕ), (s, t) ∈ pairs → s = k * t) :=
by 
sorry

end value_of_k_l3_3815


namespace triangle_area_is_integer_l3_3714

theorem triangle_area_is_integer (x1 x2 x3 y1 y2 y3 : ℤ) 
  (hx_even : (x1 + x2 + x3) % 2 = 0) 
  (hy_even : (y1 + y2 + y3) % 2 = 0) : 
  ∃ k : ℤ, 
    abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) = 2 * k := 
sorry

end triangle_area_is_integer_l3_3714


namespace number_of_ways_to_feed_animals_l3_3990

-- Definitions for the conditions
def pairs_of_animals := 5
def alternating_feeding (start_with_female : Bool) (remaining_pairs : ℕ) : ℕ :=
if start_with_female then
  (pairs_of_animals.factorial / 2 ^ pairs_of_animals)
else
  0 -- we can ignore this case as it is not needed

-- Theorem statement
theorem number_of_ways_to_feed_animals :
  alternating_feeding true pairs_of_animals = 2880 :=
sorry

end number_of_ways_to_feed_animals_l3_3990


namespace max_value_2ab_2bc_root_3_l3_3389

theorem max_value_2ab_2bc_root_3 (a b c : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c) (h_sum : a^2 + b^2 + c^2 = 3) :
  2 * a * b + 2 * b * c * Real.sqrt 3 ≤ 6 := by
sorry

end max_value_2ab_2bc_root_3_l3_3389


namespace smallest_positive_multiple_of_45_l3_3473

theorem smallest_positive_multiple_of_45 : ∃ (n : ℕ), n > 0 ∧ ∃ (x : ℕ), x > 0 ∧ n = 45 * x ∧ n = 45 :=
by
  sorry

end smallest_positive_multiple_of_45_l3_3473


namespace gcd_g50_g51_l3_3008

-- Define the polynomial g(x)
def g (x : ℤ) : ℤ := x^2 + x + 2023

-- State the theorem with necessary conditions
theorem gcd_g50_g51 : Int.gcd (g 50) (g 51) = 17 :=
by
  -- Goals and conditions stated
  sorry  -- Placeholder for the proof

end gcd_g50_g51_l3_3008


namespace choose_president_and_secretary_same_gender_l3_3121

theorem choose_president_and_secretary_same_gender :
  let total_members := 25
  let boys := 15
  let girls := 10
  ∃ (total_ways : ℕ), total_ways = (boys * (boys - 1)) + (girls * (girls - 1)) := sorry

end choose_president_and_secretary_same_gender_l3_3121


namespace min_value_expression_l3_3184

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + 1 / b) * (b + 4 / a) ≥ 9 :=
by
  sorry

end min_value_expression_l3_3184


namespace work_rate_l3_3514

theorem work_rate (x : ℕ) (hx : 2 * x = 30) : x = 15 := by
  -- We assume the prerequisite 2 * x = 30
  sorry

end work_rate_l3_3514


namespace largest_pies_without_any_ingredients_l3_3485

-- Define the conditions
def total_pies : ℕ := 60
def pies_with_strawberries : ℕ := total_pies / 4
def pies_with_bananas : ℕ := total_pies * 3 / 8
def pies_with_cherries : ℕ := total_pies / 2
def pies_with_pecans : ℕ := total_pies / 10

-- State the theorem to prove
theorem largest_pies_without_any_ingredients : (total_pies - pies_with_cherries) = 30 := by
  sorry

end largest_pies_without_any_ingredients_l3_3485


namespace football_game_spectators_l3_3490

-- Define the conditions and the proof goals
theorem football_game_spectators 
  (A C : ℕ) 
  (h_condition_1 : 2 * A + 2 * C + 40 = 310) 
  (h_condition_2 : C = A / 2) : 
  A = 90 ∧ C = 45 ∧ (A + C + 20) = 155 := 
by 
  sorry

end football_game_spectators_l3_3490


namespace ganesh_average_speed_l3_3159

variable (D : ℝ) -- the distance between towns X and Y

theorem ganesh_average_speed :
  let time_x_to_y := D / 43
  let time_y_to_x := D / 34
  let total_distance := 2 * D
  let total_time := time_x_to_y + time_y_to_x
  let avg_speed := total_distance / total_time
  avg_speed = 37.97 := by
    sorry

end ganesh_average_speed_l3_3159


namespace percentage_of_cars_in_accident_l3_3258

-- Define probabilities of each segment of the rally
def prob_fall_bridge := 1 / 5
def prob_off_turn := 3 / 10
def prob_crash_tunnel := 1 / 10
def prob_stuck_sand := 2 / 5

-- Define complement probabilities (successful completion)
def prob_success_bridge := 1 - prob_fall_bridge
def prob_success_turn := 1 - prob_off_turn
def prob_success_tunnel := 1 - prob_crash_tunnel
def prob_success_sand := 1 - prob_stuck_sand

-- Define overall success probability
def prob_success_total := prob_success_bridge * prob_success_turn * prob_success_tunnel * prob_success_sand

-- Define percentage function
def percentage (p: ℚ) : ℚ := p * 100

-- Prove the percentage of cars involved in accidents
theorem percentage_of_cars_in_accident : percentage (1 - prob_success_total) = 70 := by sorry

end percentage_of_cars_in_accident_l3_3258


namespace circle_equation_l3_3768

theorem circle_equation (x y : ℝ) :
  (∀ (C P : ℝ × ℝ), C = (8, -3) ∧ P = (5, 1) →
    ∃ R : ℝ, (x - 8)^2 + (y + 3)^2 = R^2 ∧ R^2 = 25) :=
sorry

end circle_equation_l3_3768


namespace water_in_pool_after_35_days_l3_3568

theorem water_in_pool_after_35_days :
  ∀ (initial_amount : ℕ) (evap_rate : ℕ) (cycle_days : ℕ) (add_amount : ℕ) (total_days : ℕ),
  initial_amount = 300 → evap_rate = 1 → cycle_days = 5 → add_amount = 5 → total_days = 35 →
  initial_amount - evap_rate * total_days + (total_days / cycle_days) * add_amount = 300 :=
by
  intros initial_amount evap_rate cycle_days add_amount total_days h₁ h₂ h₃ h₄ h₅
  sorry

end water_in_pool_after_35_days_l3_3568


namespace cricket_game_initial_overs_l3_3166

theorem cricket_game_initial_overs
    (run_rate_initial : ℝ)
    (run_rate_remaining : ℝ)
    (remaining_overs : ℕ)
    (target_score : ℝ)
    (initial_overs : ℕ) :
    run_rate_initial = 3.2 →
    run_rate_remaining = 5.25 →
    remaining_overs = 40 →
    target_score = 242 →
    initial_overs = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end cricket_game_initial_overs_l3_3166


namespace average_other_marbles_l3_3074

def total_marbles : ℕ := 10 -- Define a hypothetical total number for computation
def clear_marbles : ℕ := total_marbles * 40 / 100
def black_marbles : ℕ := total_marbles * 20 / 100
def other_marbles : ℕ := total_marbles - clear_marbles - black_marbles
def marbles_taken : ℕ := 5

theorem average_other_marbles :
  marbles_taken * other_marbles / total_marbles = 2 := by
  sorry

end average_other_marbles_l3_3074


namespace math_club_partition_l3_3635

def is_played (team : Finset ℕ) (A B C : ℕ) : Bool :=
(A ∈ team ∧ B ∉ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∈ team ∧ C ∉ team) ∨ 
(A ∉ team ∧ B ∉ team ∧ C ∈ team) ∨ 
(A ∈ team ∧ B ∈ team ∧ C ∈ team)

theorem math_club_partition 
  (students : Finset ℕ) (A B C : ℕ) 
  (h_size : students.card = 24)
  (teams : List (Finset ℕ))
  (h_teams : teams.length = 4)
  (h_team_size : ∀ t ∈ teams, t.card = 6)
  (h_partition : ∀ t ∈ teams, t ⊆ students) :
  ∃ (teams_played : List (Finset ℕ)), teams_played.length = 1 ∨ teams_played.length = 3 :=
sorry

end math_club_partition_l3_3635


namespace find_annual_interest_rate_l3_3641

theorem find_annual_interest_rate (A P : ℝ) (n t : ℕ) (r : ℝ) :
  A = P * (1 + r / n)^(n * t) →
  A = 5292 →
  P = 4800 →
  n = 1 →
  t = 2 →
  r = 0.05 :=
by
  intros h₁ h₂ h₃ h₄ h₅
  sorry

end find_annual_interest_rate_l3_3641


namespace solve_for_a_l3_3431

theorem solve_for_a (a : ℝ) (h : 50 - |a - 2| = |4 - a|) :
  a = -22 ∨ a = 28 :=
sorry

end solve_for_a_l3_3431


namespace problem_proof_l3_3489

def mixed_to_improper (a b c : ℚ) : ℚ := a + b / c

noncomputable def evaluate_expression : ℚ :=
  100 - (mixed_to_improper 3 1 8) / (mixed_to_improper 2 1 12 - 5 / 8) * (8 / 5 + mixed_to_improper 2 2 3)

theorem problem_proof : evaluate_expression = 636 / 7 := 
  sorry

end problem_proof_l3_3489


namespace radius_ratio_l3_3988

variable (VL VS rL rS : ℝ)
variable (hVL : VL = 432 * Real.pi)
variable (hVS : VS = 0.275 * VL)

theorem radius_ratio (h1 : (4 / 3) * Real.pi * rL^3 = VL)
                     (h2 : (4 / 3) * Real.pi * rS^3 = VS) :
  rS / rL = 2 / 3 := by
  sorry

end radius_ratio_l3_3988


namespace vertex_of_parabola_l3_3358

theorem vertex_of_parabola : 
  ∀ x, (3 * (x - 1)^2 + 2) = ((x - 1)^2 * 3 + 2) := 
by {
  -- The proof steps would go here
  sorry -- Placeholder to signify the proof steps are omitted
}

end vertex_of_parabola_l3_3358


namespace no_solution_a4_plus_6_eq_b3_mod_13_l3_3158

theorem no_solution_a4_plus_6_eq_b3_mod_13 :
  ¬ ∃ (a b : ℤ), (a^4 + 6) % 13 = b^3 % 13 :=
by
  sorry

end no_solution_a4_plus_6_eq_b3_mod_13_l3_3158


namespace find_value_of_f3_l3_3999

variable {R : Type} [LinearOrderedField R]

/-- f is an odd function -/
def is_odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

/-- f is symmetric about the line x = 1 -/
def is_symmetric_about (f : R → R) (a : R) : Prop := ∀ x : R, f (a + x) = f (a - x)

variable (f : R → R)
variable (Hodd : is_odd_function f)
variable (Hsymmetric : is_symmetric_about f 1)
variable (Hf1 : f 1 = 2)

theorem find_value_of_f3 : f 3 = -2 :=
by
  sorry

end find_value_of_f3_l3_3999


namespace average_age_of_club_l3_3281

theorem average_age_of_club (S_f S_m S_c : ℕ) (females males children : ℕ) (avg_females avg_males avg_children : ℕ) :
  females = 12 →
  males = 20 →
  children = 8 →
  avg_females = 28 →
  avg_males = 40 →
  avg_children = 10 →
  S_f = avg_females * females →
  S_m = avg_males * males →
  S_c = avg_children * children →
  (S_f + S_m + S_c) / (females + males + children) = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end average_age_of_club_l3_3281


namespace average_marks_l3_3858

/-- Given that the total marks in physics, chemistry, and mathematics is 110 more than the marks obtained in physics. -/
theorem average_marks (P C M : ℕ) (h : P + C + M = P + 110) : (C + M) / 2 = 55 :=
by
  -- The proof goes here.
  sorry

end average_marks_l3_3858


namespace carrie_weekly_earning_l3_3421

-- Definitions and conditions
def iphone_cost : ℕ := 800
def trade_in_value : ℕ := 240
def weeks_needed : ℕ := 7

-- Calculate the required weekly earning
def weekly_earning : ℕ := (iphone_cost - trade_in_value) / weeks_needed

-- Problem statement: Prove that Carrie makes $80 per week babysitting
theorem carrie_weekly_earning :
  weekly_earning = 80 := by
  sorry

end carrie_weekly_earning_l3_3421


namespace probability_of_Y_l3_3588

theorem probability_of_Y (P_X P_both : ℝ) (h1 : P_X = 1/5) (h2 : P_both = 0.13333333333333333) : 
    (0.13333333333333333 / (1 / 5)) = 0.6666666666666667 :=
by sorry

end probability_of_Y_l3_3588


namespace arnel_kept_fifty_pencils_l3_3219

theorem arnel_kept_fifty_pencils
    (num_boxes : ℕ) (pencils_each_box : ℕ) (friends : ℕ) (pencils_each_friend : ℕ) (total_pencils : ℕ)
    (boxes_pencils : ℕ) (friends_pencils : ℕ) :
    num_boxes = 10 →
    pencils_each_box = 5 →
    friends = 5 →
    pencils_each_friend = 8 →
    friends_pencils = friends * pencils_each_friend →
    boxes_pencils = num_boxes * pencils_each_box →
    total_pencils = boxes_pencils + friends_pencils →
    (total_pencils - friends_pencils) = 50 :=
by
    sorry

end arnel_kept_fifty_pencils_l3_3219


namespace necessary_but_not_sufficient_condition_l3_3686

-- Definitions
def represents_ellipse (m n : ℝ) (x y : ℝ) : Prop := 
  (x^2 / m + y^2 / n = 1)

-- Main theorem statement
theorem necessary_but_not_sufficient_condition 
    (m n x y : ℝ) (h_mn_pos : m * n > 0) :
    (represents_ellipse m n x y) → 
    (m ≠ n ∧ m > 0 ∧ n > 0 ∧ represents_ellipse m n x y) → 
    (m * n > 0) ∧ ¬(
    ∀ m n : ℝ, (m ≠ n ∧ m > 0 ∧ n > 0) →
    represents_ellipse m n x y
    ) :=
by
  sorry

end necessary_but_not_sufficient_condition_l3_3686


namespace ball_radius_l3_3252

theorem ball_radius 
  (r_cylinder : ℝ) (h_rise : ℝ) (v_approx : ℝ)
  (r_cylinder_value : r_cylinder = 12)
  (h_rise_value : h_rise = 6.75)
  (v_approx_value : v_approx = 3053.628) :
  ∃ (r_ball : ℝ), (4 / 3) * Real.pi * r_ball^3 = v_approx ∧ r_ball = 9 := 
by 
  use 9
  sorry

end ball_radius_l3_3252


namespace percentage_increase_school_B_l3_3598

theorem percentage_increase_school_B (A B Q_A Q_B : ℝ) 
  (h1 : Q_A = 0.7 * A) 
  (h2 : Q_B = 1.5 * Q_A) 
  (h3 : Q_B = 0.875 * B) :
  (B - A) / A * 100 = 20 :=
by
  sorry

end percentage_increase_school_B_l3_3598


namespace neg_p_sufficient_but_not_necessary_for_q_l3_3977

variable {x : ℝ}

def p (x : ℝ) : Prop := (1 - x) * (x + 3) < 0
def q (x : ℝ) : Prop := 5 * x - 6 ≤ x^2

theorem neg_p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, ¬ p x → q x) ∧ ¬ (∀ x : ℝ, q x → ¬ p x) :=
by
  sorry

end neg_p_sufficient_but_not_necessary_for_q_l3_3977


namespace white_balls_probability_l3_3411

noncomputable def probability_all_white (total_balls white_balls draw_count : ℕ) : ℚ :=
  if h : total_balls >= draw_count ∧ white_balls >= draw_count then
    (Nat.choose white_balls draw_count : ℚ) / (Nat.choose total_balls draw_count : ℚ)
  else
    0

theorem white_balls_probability :
  probability_all_white 11 5 5 = 1 / 462 :=
by
  sorry

end white_balls_probability_l3_3411


namespace abs_neg_2023_eq_2023_l3_3418

theorem abs_neg_2023_eq_2023 : abs (-2023) = 2023 := by
  sorry

end abs_neg_2023_eq_2023_l3_3418


namespace negation_of_existence_l3_3040

theorem negation_of_existence :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) :=
by sorry

end negation_of_existence_l3_3040


namespace exit_time_correct_l3_3835

def time_to_exit_wide : ℝ := 6
def time_to_exit_narrow : ℝ := 10

theorem exit_time_correct :
  ∃ x y : ℝ, x = 6 ∧ y = 10 ∧ 
  (1 / x + 1 / y = 4 / 15) ∧ 
  (y = x + 4) ∧ 
  (3.75 * (1 / x + 1 / y) = 1) :=
by
  use time_to_exit_wide
  use time_to_exit_narrow
  sorry

end exit_time_correct_l3_3835


namespace ways_to_place_7_balls_into_3_boxes_l3_3231

theorem ways_to_place_7_balls_into_3_boxes :
  (Nat.choose (7 + 3 - 1) (3 - 1)) = 36 :=
by
  sorry

end ways_to_place_7_balls_into_3_boxes_l3_3231


namespace max_gcd_13n_plus_4_8n_plus_3_l3_3265

theorem max_gcd_13n_plus_4_8n_plus_3 : ∃ n : ℕ, n > 0 ∧ Int.gcd (13 * n + 4) (8 * n + 3) = 11 := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l3_3265


namespace problem1_solution_problem2_solution_l3_3872

noncomputable def problem1 : ℝ :=
  (Real.sqrt (1 / 3) + Real.sqrt 6) / Real.sqrt 3

noncomputable def problem2 : ℝ :=
  (Real.sqrt 3)^2 - Real.sqrt 4 + Real.sqrt ((-2)^2)

theorem problem1_solution :
  problem1 = 1 + 3 * Real.sqrt 2 :=
by
  sorry

theorem problem2_solution :
  problem2 = 3 :=
by
  sorry

end problem1_solution_problem2_solution_l3_3872


namespace box_volume_l3_3759

structure Box where
  L : ℝ  -- Length
  W : ℝ  -- Width
  H : ℝ  -- Height

def front_face_area (box : Box) : ℝ := box.L * box.H
def top_face_area (box : Box) : ℝ := box.L * box.W
def side_face_area (box : Box) : ℝ := box.H * box.W

noncomputable def volume (box : Box) : ℝ := box.L * box.W * box.H

theorem box_volume (box : Box)
  (h1 : front_face_area box = 0.5 * top_face_area box)
  (h2 : top_face_area box = 1.5 * side_face_area box)
  (h3 : side_face_area box = 72) :
  volume box = 648 := by
  sorry

end box_volume_l3_3759


namespace range_of_a_l3_3404

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - x + a = 0

theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a < 0 ∨ (1/4 < a ∧ a < 4) := 
sorry

end range_of_a_l3_3404


namespace clean_room_to_homework_ratio_l3_3423

-- Define the conditions
def timeHomework : ℕ := 30
def timeWalkDog : ℕ := timeHomework + 5
def timeTrash : ℕ := timeHomework / 6
def totalTimeAvailable : ℕ := 120
def remainingTime : ℕ := 35

-- Definition to calculate total time spent on other tasks
def totalTimeOnOtherTasks : ℕ := timeHomework + timeWalkDog + timeTrash

-- Definition to calculate the time to clean the room
def timeCleanRoom : ℕ := totalTimeAvailable - remainingTime - totalTimeOnOtherTasks

-- The theorem to prove the ratio
theorem clean_room_to_homework_ratio : (timeCleanRoom : ℚ) / (timeHomework : ℚ) = 1 / 2 :=
by
  -- Proof steps would go here
  sorry

end clean_room_to_homework_ratio_l3_3423


namespace solve_inequality_l3_3656

theorem solve_inequality (x : ℝ) : (x^2 - 50 * x + 625 ≤ 25) = (20 ≤ x ∧ x ≤ 30) :=
sorry

end solve_inequality_l3_3656


namespace g_2187_value_l3_3251

-- Define the function properties and the goal
theorem g_2187_value (g : ℕ → ℝ) (h : ∀ x y m : ℕ, x + y = 3^m → g x + g y = m^3) :
  g 2187 = 343 :=
sorry

end g_2187_value_l3_3251


namespace elevator_travel_time_l3_3381

noncomputable def total_time_in_hours (floors : ℕ) (time_first_half : ℕ) (time_next_floors_per_floor : ℕ) (next_floors : ℕ) (time_final_floors_per_floor : ℕ) (final_floors : ℕ) : ℕ :=
  let time_first_part := time_first_half
  let time_next_part := time_next_floors_per_floor * next_floors
  let time_final_part := time_final_floors_per_floor * final_floors
  (time_first_part + time_next_part + time_final_part) / 60

theorem elevator_travel_time :
  total_time_in_hours 20 15 5 5 16 5 = 2 := 
by
  sorry

end elevator_travel_time_l3_3381


namespace inequalities_not_simultaneous_l3_3891

theorem inequalities_not_simultaneous (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (ineq1 : a + b < c + d) (ineq2 : (a + b) * (c + d) < a * b + c * d) (ineq3 : (a + b) * c * d < (c + d) * a * b) :
  false := 
sorry

end inequalities_not_simultaneous_l3_3891


namespace coordinates_C_on_segment_AB_l3_3529

theorem coordinates_C_on_segment_AB :
  ∃ C : (ℝ × ℝ), 
  (C.1 = 2 ∧ C.2 = 6) ∧
  ∃ A B : (ℝ × ℝ), 
  (A = (-1, 0)) ∧ 
  (B = (3, 8)) ∧ 
  (∃ k : ℝ, (k = 3) ∧ dist (C) (A) = k * dist (C) (B)) :=
by
  sorry

end coordinates_C_on_segment_AB_l3_3529


namespace angle_B_is_40_degrees_l3_3155

theorem angle_B_is_40_degrees (angle_A angle_B angle_C : ℝ)
  (h1 : angle_A = 3 * angle_B)
  (h2 : angle_B = 2 * angle_C)
  (triangle_sum : angle_A + angle_B + angle_C = 180) :
  angle_B = 40 :=
by
  sorry

end angle_B_is_40_degrees_l3_3155


namespace trig_expression_equality_l3_3484

theorem trig_expression_equality :
  (1 - 1 / (Real.cos (Real.pi / 6))) *
  (1 + 1 / (Real.sin (Real.pi / 3))) *
  (1 - 1 / (Real.sin (Real.pi / 6))) *
  (1 + 1 / (Real.cos (Real.pi / 3))) = 1 :=
by
  have hcos30 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 := sorry
  have hsin60 : Real.sin (Real.pi / 3) = (Real.sqrt 3) / 2 := sorry
  have hsin30 : Real.sin (Real.pi / 6) = 1 / 2 := sorry
  have hcos60 : Real.cos (Real.pi / 3) = 1 / 2 := sorry
  sorry

end trig_expression_equality_l3_3484


namespace find_OC_l3_3673

noncomputable section

open Real

structure Point where
  x : ℝ
  y : ℝ

def OA (A : Point) : ℝ := sqrt (A.x^2 + A.y^2)
def OB (B : Point) : ℝ := sqrt (B.x^2 + B.y^2)
def OD (D : Point) : ℝ := sqrt (D.x^2 + D.y^2)
def ratio_of_lengths (A B : Point) : ℝ := OA A / OB B

def find_D (A B : Point) : Point :=
  let ratio := ratio_of_lengths A B
  { x := (A.x + ratio * B.x) / (1 + ratio),
    y := (A.y + ratio * B.y) / (1 + ratio) }

-- Given conditions
def A : Point := ⟨0, 1⟩
def B : Point := ⟨-3, 4⟩
def C_magnitude : ℝ := 2

-- Goal to prove
theorem find_OC : Point :=
  let D := find_D A B
  let D_length := OD D
  let scale := C_magnitude / D_length
  { x := D.x * scale,
    y := D.y * scale }

example : find_OC = ⟨-sqrt 10 / 5, 3 * sqrt 10 / 5⟩ := by
  sorry

end find_OC_l3_3673


namespace sin_double_angle_l3_3067

theorem sin_double_angle (k α : ℝ) (h : Real.cos (π / 4 - α) = k) : Real.sin (2 * α) = 2 * k^2 - 1 := 
by
  sorry

end sin_double_angle_l3_3067


namespace calc1_calc2_calc3_calc4_l3_3902

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end calc1_calc2_calc3_calc4_l3_3902


namespace quadratic_decomposition_l3_3096

theorem quadratic_decomposition (a b c : ℝ) :
  (∀ x : ℝ, 6 * x^2 + 72 * x + 432 = a * (x + b)^2 + c) → a + b + c = 228 :=
sorry

end quadratic_decomposition_l3_3096


namespace golden_chest_diamonds_rubies_l3_3961

theorem golden_chest_diamonds_rubies :
  ∀ (diamonds rubies : ℕ), diamonds = 421 → rubies = 377 → diamonds - rubies = 44 :=
by
  intros diamonds rubies
  sorry

end golden_chest_diamonds_rubies_l3_3961


namespace cloth_gain_percentage_l3_3212

theorem cloth_gain_percentage 
  (x : ℝ) -- x represents the cost price of 1 meter of cloth
  (CP : ℝ := 30 * x) -- CP of 30 meters of cloth
  (profit : ℝ := 10 * x) -- profit from selling 30 meters of cloth
  (SP : ℝ := CP + profit) -- selling price of 30 meters of cloth
  (gain_percentage : ℝ := (profit / CP) * 100) : 
  gain_percentage = 33.33 := 
sorry

end cloth_gain_percentage_l3_3212


namespace b8_expression_l3_3881

theorem b8_expression (a b : ℕ → ℚ)
  (ha0 : a 0 = 2)
  (hb0 : b 0 = 3)
  (ha : ∀ n, a (n + 1) = (a n) ^ 2 / (b n))
  (hb : ∀ n, b (n + 1) = (b n) ^ 2 / (a n)) :
  b 8 = 3 ^ 3281 / 2 ^ 3280 :=
by
  sorry

end b8_expression_l3_3881


namespace k_equals_10_l3_3234

variable {α : Type*} [LinearOrderedField α]

def arithmetic_sequence (a d : α) : ℕ → α
  | 0     => a
  | (n+1) => a + (n+1) * d

noncomputable def sum_of_first_n_terms (a d : α) (n : ℕ) : α :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem k_equals_10
  (a d : α)
  (h1 : sum_of_first_n_terms a d 9 = sum_of_first_n_terms a d 4)
  (h2 : arithmetic_sequence a d 4 + arithmetic_sequence a d 10 = 0) :
  k = 10 :=
sorry

end k_equals_10_l3_3234


namespace simplify_fraction_l3_3599

theorem simplify_fraction :
  (5 : ℝ) / (Real.sqrt 75 + 3 * Real.sqrt 3 + Real.sqrt 48) = (5 * Real.sqrt 3) / 36 :=
by
  have h1 : Real.sqrt 75 = 5 * Real.sqrt 3 := by sorry
  have h2 : Real.sqrt 48 = 4 * Real.sqrt 3 := by sorry
  sorry

end simplify_fraction_l3_3599


namespace factor_x4_minus_81_l3_3068

variable (x : ℝ)

theorem factor_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
  by { -- proof steps would go here 
    sorry 
}

end factor_x4_minus_81_l3_3068


namespace solved_fraction_equation_l3_3901

theorem solved_fraction_equation :
  ∀ (x : ℚ),
    x ≠ 2 →
    x ≠ 7 →
    x ≠ -5 →
    (x^2 - 6*x + 8) / (x^2 - 9*x + 14) = (x^2 - 4*x - 5) / (x^2 - 2*x - 35) →
    x = 55 / 13 := by
  sorry

end solved_fraction_equation_l3_3901


namespace distance_traveled_by_second_hand_l3_3287

def second_hand_length : ℝ := 8
def time_period_minutes : ℝ := 45
def rotations_per_minute : ℝ := 1

theorem distance_traveled_by_second_hand :
  let circumference := 2 * Real.pi * second_hand_length
  let rotations := time_period_minutes * rotations_per_minute
  let total_distance := rotations * circumference
  total_distance = 720 * Real.pi := by
  sorry

end distance_traveled_by_second_hand_l3_3287


namespace length_of_EC_l3_3268

theorem length_of_EC
  (AB CD AC : ℝ)
  (h1 : AB = 3 * CD)
  (h2 : AC = 15)
  (EC : ℝ)
  (h3 : AC = 4 * EC)
  : EC = 15 / 4 := 
sorry

end length_of_EC_l3_3268


namespace apples_problem_l3_3561

theorem apples_problem :
  ∃ (jackie rebecca : ℕ), (rebecca = 2 * jackie) ∧ (∃ (adam : ℕ), (adam = jackie + 3) ∧ (adam = 9) ∧ jackie = 6 ∧ rebecca = 12) :=
by
  sorry

end apples_problem_l3_3561


namespace max_perimeter_triangle_l3_3685

theorem max_perimeter_triangle (y : ℤ) (h1 : y < 16) (h2 : y > 2) : 
    7 + 9 + y = 31 → y = 15 := by
  sorry

end max_perimeter_triangle_l3_3685


namespace rectangle_extraction_l3_3127

theorem rectangle_extraction (m : ℤ) (h1 : m > 12) : 
  ∃ (x y : ℤ), x ≤ y ∧ x * y > m ∧ x * (y - 1) < m :=
by
  sorry

end rectangle_extraction_l3_3127


namespace sqrt_domain_l3_3603

theorem sqrt_domain (x : ℝ) : (∃ y, y * y = x - 2) ↔ (x ≥ 2) :=
by sorry

end sqrt_domain_l3_3603


namespace distance_between_parallel_lines_l3_3187

theorem distance_between_parallel_lines :
  let A := 3
  let B := 2
  let C1 := -1
  let C2 := 1 / 2
  let d := |C2 - C1| / Real.sqrt (A^2 + B^2)
  d = 3 / Real.sqrt 13 :=
by
  -- Proof goes here
  sorry

end distance_between_parallel_lines_l3_3187


namespace dot_product_is_one_l3_3399

variable (a : ℝ × ℝ := (1, 1))
variable (b : ℝ × ℝ := (-1, 2))

theorem dot_product_is_one : (a.1 * b.1 + a.2 * b.2) = 1 := by
  sorry

end dot_product_is_one_l3_3399


namespace expand_and_simplify_l3_3037

theorem expand_and_simplify :
  ∀ (x : ℝ), 2 * x * (3 * x ^ 2 - 4 * x + 5) - (x ^ 2 - 3 * x) * (4 * x + 5) = 2 * x ^ 3 - x ^ 2 + 25 * x :=
by
  intro x
  sorry

end expand_and_simplify_l3_3037


namespace description_of_T_l3_3233

-- Define the conditions
def T := { p : ℝ × ℝ | (∃ (c : ℝ), ((c = 5 ∨ c = p.1 + 3 ∨ c = p.2 - 6) ∧ (5 ≥ p.1 + 3) ∧ (5 ≥ p.2 - 6))) }

-- The main theorem
theorem description_of_T : 
  ∃ p : ℝ × ℝ, 
    (p = (2, 11)) ∧ 
    ∀ q ∈ T, 
      (q.fst = 2 ∧ q.snd ≤ 11) ∨ 
      (q.snd = 11 ∧ q.fst ≤ 2) ∨ 
      (q.snd = q.fst + 9 ∧ q.fst ≤ 2) :=
sorry

end description_of_T_l3_3233


namespace prove_inequality_l3_3216

open Real

noncomputable def inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : Prop :=
  3 + (a + b + c) + (1/a + 1/b + 1/c) + (a/b + b/c + c/a) ≥ 
  3 * (a + 1) * (b + 1) * (c + 1) / (a * b * c + 1)

theorem prove_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) : 
  inequality a b c h1 h2 h3 := 
  sorry

end prove_inequality_l3_3216


namespace Tino_jellybeans_l3_3558

variable (L T A : ℕ)
variable (h1 : T = L + 24)
variable (h2 : A = L / 2)
variable (h3 : A = 5)

theorem Tino_jellybeans : T = 34 :=
by
  sorry

end Tino_jellybeans_l3_3558


namespace harvest_unripe_oranges_l3_3298

theorem harvest_unripe_oranges (R T D U: ℕ) (h1: R = 28) (h2: T = 2080) (h3: D = 26)
  (h4: T = D * (R + U)) :
  U = 52 :=
by
  sorry

end harvest_unripe_oranges_l3_3298


namespace percent_increase_decrease_l3_3653

theorem percent_increase_decrease (P y : ℝ) (h : (P * (1 + y / 100) * (1 - y / 100) = 0.90 * P)) :
    y = 31.6 :=
by
  sorry

end percent_increase_decrease_l3_3653


namespace hostel_provisions_l3_3776

theorem hostel_provisions (x : ℕ) (h1 : 250 * x = 200 * 40) : x = 32 :=
by
  sorry

end hostel_provisions_l3_3776


namespace playback_methods_proof_l3_3009

/-- A TV station continuously plays 5 advertisements, consisting of 3 different commercial advertisements
and 2 different Olympic promotional advertisements. The requirements are:
  1. The last advertisement must be an Olympic promotional advertisement.
  2. The 2 Olympic promotional advertisements can be played consecutively.
-/
def number_of_playback_methods (commercials olympics: ℕ) (last_ad_olympic: Bool) (olympics_consecutive: Bool) : ℕ :=
  if commercials = 3 ∧ olympics = 2 ∧ last_ad_olympic ∧ olympics_consecutive then 36 else 0

theorem playback_methods_proof :
  number_of_playback_methods 3 2 true true = 36 := by
  sorry

end playback_methods_proof_l3_3009


namespace find_roots_of_parabola_l3_3230

-- Define the conditions given in the problem
variables (a b c : ℝ)
variable (a_nonzero : a ≠ 0)
variable (passes_through_1_0 : a * 1^2 + b * 1 + c = 0)
variable (axis_of_symmetry : -b / (2 * a) = -2)

-- Lean theorem statement
theorem find_roots_of_parabola (a b c : ℝ) (a_nonzero : a ≠ 0)
(passes_through_1_0 : a * 1^2 + b * 1 + c = 0) (axis_of_symmetry : -b / (2 * a) = -2) :
  (a * (-5)^2 + b * (-5) + c = 0) ∧ (a * 1^2 + b * 1 + c = 0) :=
by
  -- Placeholder for the proof
  sorry

end find_roots_of_parabola_l3_3230


namespace sufficient_but_not_necessary_l3_3821

def P (x : ℝ) : Prop := 2 < x ∧ x < 4
def Q (x : ℝ) : Prop := Real.log x < Real.exp 1

theorem sufficient_but_not_necessary (x : ℝ) : P x → Q x ∧ (¬ ∀ x, Q x → P x) := by
  sorry

end sufficient_but_not_necessary_l3_3821


namespace quadratic_y1_gt_y2_l3_3388

theorem quadratic_y1_gt_y2 (a b c y1 y2 : ℝ) (h_a_pos : a > 0) (h_sym : ∀ x, a * (x - 1)^2 + c = a * (1 - x)^2 + c) (h1 : y1 = a * (-1)^2 + b * (-1) + c) (h2 : y2 = a * 2^2 + b * 2 + c) : y1 > y2 :=
sorry

end quadratic_y1_gt_y2_l3_3388


namespace compute_pqr_l3_3521

theorem compute_pqr
  (p q r : ℤ)
  (hp : p ≠ 0)
  (hq : q ≠ 0)
  (hr : r ≠ 0)
  (h_sum : p + q + r = 30)
  (h_eq : 1 / p + 1 / q + 1 / r + 240 / (p * q * r) = 1) :
  p * q * r = 1080 := by
  sorry

end compute_pqr_l3_3521


namespace converse_l3_3373

theorem converse (x y : ℝ) (h : x + y ≥ 5) : x ≥ 2 ∧ y ≥ 3 := 
sorry

end converse_l3_3373


namespace value_of_expression_l3_3675

theorem value_of_expression : 4 * (8 - 6) - 7 = 1 := by
  -- Calculation steps would go here
  sorry

end value_of_expression_l3_3675


namespace find_average_after_17th_inning_l3_3586

def initial_average_after_16_inns (A : ℕ) : Prop :=
  let total_runs := 16 * A
  let new_total_runs := total_runs + 87
  let new_average := new_total_runs / 17
  new_average = A + 4

def runs_in_17th_inning := 87

noncomputable def average_after_17th_inning (A : ℕ) : Prop :=
  A + 4 = 23

theorem find_average_after_17th_inning (A : ℕ) :
  initial_average_after_16_inns A →
  average_after_17th_inning A :=
  sorry

end find_average_after_17th_inning_l3_3586


namespace cylinder_volume_ratio_l3_3739

theorem cylinder_volume_ratio (h1 h2 r1 r2 V1 V2 : ℝ)
  (h1_eq : h1 = 9)
  (h2_eq : h2 = 6)
  (circumference1_eq : 2 * π * r1 = 6)
  (circumference2_eq : 2 * π * r2 = 9)
  (V1_eq : V1 = π * r1^2 * h1)
  (V2_eq : V2 = π * r2^2 * h2)
  (V1_calculated : V1 = 81 / π)
  (V2_calculated : V2 = 243 / (4 * π)) :
  (max V1 V2) / (min V1 V2) = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l3_3739


namespace sum_first_15_terms_l3_3438

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers

-- Define the conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0

def a1_plus_a15_eq_three (a : ℕ → ℝ) : Prop :=
  a 1 + a 15 = 3

noncomputable def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (a 1 + a n)

theorem sum_first_15_terms (a : ℕ → ℝ) (h_arith: arithmetic_sequence a) (h_sum: a1_plus_a15_eq_three a) :
  sum_first_n_terms a 15 = 22.5 := by
  sorry

end sum_first_15_terms_l3_3438


namespace ratio_of_3_numbers_l3_3727

variable (A B C : ℕ)
variable (k : ℕ)

theorem ratio_of_3_numbers (h₁ : A = 5 * k) (h₂ : B = k) (h₃ : C = 4 * k) (h_sum : A + B + C = 1000) : C = 400 :=
  sorry

end ratio_of_3_numbers_l3_3727


namespace percent_daisies_l3_3311

theorem percent_daisies 
    (total_flowers : ℕ)
    (yellow_flowers : ℕ)
    (yellow_tulips : ℕ)
    (blue_flowers : ℕ)
    (blue_daisies : ℕ)
    (h1 : 2 * yellow_tulips = yellow_flowers) 
    (h2 : 3 * blue_daisies = blue_flowers)
    (h3 : 10 * yellow_flowers = 7 * total_flowers) : 
    100 * (yellow_flowers / 2 + blue_daisies) = 45 * total_flowers :=
by
  sorry

end percent_daisies_l3_3311


namespace suzy_twice_mary_l3_3888

def suzy_current_age : ℕ := 20
def mary_current_age : ℕ := 8

theorem suzy_twice_mary (x : ℕ) : suzy_current_age + x = 2 * (mary_current_age + x) ↔ x = 4 := by
  sorry

end suzy_twice_mary_l3_3888


namespace binom_10_8_equals_45_l3_3873

theorem binom_10_8_equals_45 : Nat.choose 10 8 = 45 := 
by
  sorry

end binom_10_8_equals_45_l3_3873


namespace compute_expression_l3_3966

theorem compute_expression : 2 + 7 * 3 - 4 + 8 / 2 = 23 := by
  sorry

end compute_expression_l3_3966


namespace acute_triangle_sin_sum_gt_2_l3_3566

open Real

theorem acute_triangle_sin_sum_gt_2 (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (hγ : 0 < γ ∧ γ < π / 2) (h_sum : α + β + γ = π) :
  sin α + sin β + sin γ > 2 :=
sorry

end acute_triangle_sin_sum_gt_2_l3_3566


namespace common_ratio_of_geometric_sequence_l3_3779

theorem common_ratio_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_nonzero : d ≠ 0) 
  (h_geom : (a 1)^2 = a 0 * a 2) :
  (a 2) / (a 0) = 3 / 2 := 
sorry

end common_ratio_of_geometric_sequence_l3_3779


namespace least_positive_integer_divisible_by_primes_gt_5_l3_3854

theorem least_positive_integer_divisible_by_primes_gt_5 : ∃ n : ℕ, n = 7 * 11 * 13 ∧ ∀ k : ℕ, (k > 0 ∧ (k % 7 = 0) ∧ (k % 11 = 0) ∧ (k % 13 = 0)) → k ≥ 1001 := 
sorry

end least_positive_integer_divisible_by_primes_gt_5_l3_3854


namespace sports_club_membership_l3_3665

theorem sports_club_membership :
  (17 + 21 - 10 + 2 = 30) :=
by
  sorry

end sports_club_membership_l3_3665


namespace problem_statement_l3_3850

theorem problem_statement (p q : ℚ) (hp : 3 / p = 4) (hq : 3 / q = 18) : p - q = 7 / 12 := 
by
  sorry

end problem_statement_l3_3850


namespace Kevin_lost_cards_l3_3361

theorem Kevin_lost_cards (initial_cards final_cards : ℝ) (h1 : initial_cards = 47.0) (h2 : final_cards = 40) :
  initial_cards - final_cards = 7 :=
by
  sorry

end Kevin_lost_cards_l3_3361


namespace expected_value_is_minus_one_half_l3_3417

def prob_heads := 1 / 4
def prob_tails := 2 / 4
def prob_edge := 1 / 4
def win_heads := 4
def win_tails := -3
def win_edge := 0

theorem expected_value_is_minus_one_half :
  (prob_heads * win_heads + prob_tails * win_tails + prob_edge * win_edge) = -1 / 2 :=
by
  sorry

end expected_value_is_minus_one_half_l3_3417


namespace solution_pairs_l3_3167

def equation (r p : ℤ) : Prop := r^2 - r * (p + 6) + p^2 + 5 * p + 6 = 0

theorem solution_pairs :
  ∀ (r p : ℤ),
    equation r p ↔ (r = 3 ∧ p = 1) ∨ (r = 4 ∧ p = 1) ∨ 
                    (r = 0 ∧ p = -2) ∨ (r = 4 ∧ p = -2) ∨ 
                    (r = 0 ∧ p = -3) ∨ (r = 3 ∧ p = -3) :=
by
  sorry

end solution_pairs_l3_3167


namespace sara_grew_4_onions_l3_3865

def onions_sally := 5
def onions_fred := 9
def total_onions := 18

def onions_sara : ℕ := total_onions - (onions_sally + onions_fred)

theorem sara_grew_4_onions : onions_sara = 4 := by
  -- proof here
  sorry

end sara_grew_4_onions_l3_3865


namespace cos_alpha_value_cos_2alpha_value_l3_3766

noncomputable def x : ℤ := -3
noncomputable def y : ℤ := 4
noncomputable def r : ℝ := Real.sqrt (x^2 + y^2)
noncomputable def cos_alpha : ℝ := x / r
noncomputable def cos_2alpha : ℝ := 2 * cos_alpha^2 - 1

theorem cos_alpha_value : cos_alpha = -3 / 5 := by
  sorry

theorem cos_2alpha_value : cos_2alpha = -7 / 25 := by
  sorry

end cos_alpha_value_cos_2alpha_value_l3_3766


namespace sausages_fried_l3_3784

def num_eggs : ℕ := 6
def time_per_sausage : ℕ := 5
def time_per_egg : ℕ := 4
def total_time : ℕ := 39
def time_per_sauteurs (S : ℕ) : ℕ := S * time_per_sausage

theorem sausages_fried (S : ℕ) (h : num_eggs * time_per_egg + S * time_per_sausage = total_time) : S = 3 :=
by
  sorry

end sausages_fried_l3_3784


namespace age_ratio_in_ten_years_l3_3718

-- Definitions of given conditions
variable (A : ℕ) (B : ℕ)
axiom age_condition : A = 20
axiom sum_of_ages : A + 10 + (B + 10) = 45

-- Theorem and proof skeleton for the ratio of ages in ten years.
theorem age_ratio_in_ten_years (A B : ℕ) (hA : A = 20) (hSum : A + 10 + (B + 10) = 45) :
  (A + 10) / (B + 10) = 2 := by
  sorry

end age_ratio_in_ten_years_l3_3718


namespace fox_cub_distribution_l3_3658

variable (m a x y : ℕ)
-- Assuming the system of equations given in the problem:
def fox_cub_system_of_equations (n : ℕ) : Prop :=
  ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n →
    ((k * (m - 1) * a + x) = ((m + k - 1) * y))

theorem fox_cub_distribution (m a x y : ℕ) (h : fox_cub_system_of_equations m a x y n) :
  y = ((m-1) * a) ∧ x = ((m-1)^2 * a) :=
by
  sorry

end fox_cub_distribution_l3_3658


namespace count_valid_pairs_is_7_l3_3482

def valid_pairs_count : Nat :=
  let pairs := [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3), (4, 4)]
  List.length pairs

theorem count_valid_pairs_is_7 (b c : ℕ) (hb : b > 0) (hc : c > 0) :
  (b^2 - 4 * c ≤ 0) → (c^2 - 4 * b ≤ 0) → valid_pairs_count = 7 :=
by
  sorry

end count_valid_pairs_is_7_l3_3482


namespace find_B_l3_3148

theorem find_B (N : ℕ) (A B : ℕ) (H1 : N = 757000000 + A * 10000 + B * 1000 + 384) (H2 : N % 357 = 0) : B = 5 :=
sorry

end find_B_l3_3148


namespace find_p_l3_3738

theorem find_p
  (p : ℝ)
  (h1 : ∃ (x y : ℝ), p * (x^2 - y^2) = (p^2 - 1) * x * y ∧ |x - 1| + |y| = 1)
  (h2 : ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ), 
         x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧ 
         p * (x₁^2 - y₁^2) = (p^2 - 1) * x₁ * y₁ ∧ |x₁ - 1| + |y₁| = 1 ∧
         p * (x₂^2 - y₂^2) = (p^2 - 1) * x₂ * y₂ ∧ |x₂ - 1| + |y₂| = 1 ∧
         p * (x₃^2 - y₃^2) = (p^2 - 1) * x₃ * y₃ ∧ |x₃ - 1| + |y₃| = 1) :
  p = 1 ∨ p = -1 :=
by sorry

end find_p_l3_3738


namespace apples_to_pears_l3_3181

theorem apples_to_pears (a o p : ℕ) 
  (h1 : 10 * a = 5 * o) 
  (h2 : 3 * o = 4 * p) : 
  (20 * a) = 40 / 3 * p :=
sorry

end apples_to_pears_l3_3181


namespace find_a_l3_3866

theorem find_a (a : ℝ) : (∀ x : ℝ, |x - a| ≤ 3 ↔ -1 ≤ x ∧ x ≤ 5) → a = 2 :=
by
  intro h
  have h1 : |(-1 : ℝ) - a| = 3 := sorry
  have h2 : |(5 : ℝ) - a| = 3 := sorry
  sorry

end find_a_l3_3866


namespace isabella_paintable_area_l3_3014

def total_paintable_area : ℕ :=
  let room1_area := 2 * (14 * 9) + 2 * (12 * 9) - 70
  let room2_area := 2 * (13 * 9) + 2 * (11 * 9) - 70
  let room3_area := 2 * (15 * 9) + 2 * (10 * 9) - 70
  let room4_area := 4 * (12 * 9) - 70
  room1_area + room2_area + room3_area + room4_area

theorem isabella_paintable_area : total_paintable_area = 1502 := by
  sorry

end isabella_paintable_area_l3_3014


namespace percent_profit_l3_3737

-- Definitions based on given conditions
variables (P : ℝ) -- original price of the car

def discounted_price := 0.90 * P
def first_year_value := 0.945 * P
def second_year_value := 0.9828 * P
def third_year_value := 1.012284 * P
def selling_price := 1.62 * P

-- Theorem statement
theorem percent_profit : (selling_price P - P) / P * 100 = 62 := by
  sorry

end percent_profit_l3_3737


namespace perimeter_original_rectangle_l3_3667

variable {L W : ℕ}

axiom area_original : L * W = 360
axiom area_changed : (L + 10) * (W - 6) = 360

theorem perimeter_original_rectangle : 2 * (L + W) = 76 :=
by
  sorry

end perimeter_original_rectangle_l3_3667


namespace square_difference_l3_3679

theorem square_difference (x y : ℝ) 
  (h₁ : (x + y)^2 = 36) 
  (h₂ : x * y = 8) : 
  (x - y)^2 = 4 :=
by
  sorry

end square_difference_l3_3679


namespace inscribed_circle_radius_l3_3016

theorem inscribed_circle_radius (A p r s : ℝ) (h₁ : A = 2 * p) (h₂ : p = 2 * s) (h₃ : A = r * s) : r = 4 :=
by sorry

end inscribed_circle_radius_l3_3016


namespace selected_people_take_B_l3_3894

def arithmetic_sequence (a d n : Nat) : Nat := a + (n - 1) * d

theorem selected_people_take_B (a d total sampleCount start n_upper n_lower : Nat) :
  a = 9 →
  d = 30 →
  total = 960 →
  sampleCount = 32 →
  start = 451 →
  n_upper = 25 →
  n_lower = 16 →
  (960 / 32) = d → 
  (10 = n_upper - n_lower + 1) ∧ 
  ∀ n, (n_lower ≤ n ∧ n ≤ n_upper) → (start ≤ arithmetic_sequence a d n ∧ arithmetic_sequence a d n ≤ 750) :=
by sorry

end selected_people_take_B_l3_3894


namespace rectangle_area_l3_3496

theorem rectangle_area (L B : ℝ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 :=
by sorry

end rectangle_area_l3_3496


namespace line_through_points_l3_3757

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : x1 ≠ x2) (hx1 : x1 = -3) (hy1 : y1 = 1) (hx2 : x2 = 1) (hy2 : y2 = 5) :
  ∃ (m b : ℝ), (m + b = 5) ∧ (y1 = m * x1 + b) ∧ (y2 = m * x2 + b) :=
by
  sorry

end line_through_points_l3_3757


namespace bacteria_growth_rate_l3_3522

theorem bacteria_growth_rate
  (r : ℝ) 
  (h1 : ∃ B D : ℝ, B * r^30 = D) 
  (h2 : ∃ B D : ℝ, B * r^25 = D / 32) :
  r = 2 := 
by 
  sorry

end bacteria_growth_rate_l3_3522


namespace unit_prices_max_toys_l3_3951

-- For question 1
theorem unit_prices (x y : ℕ)
  (h₁ : y = x + 25)
  (h₂ : 2*y + x = 200) : x = 50 ∧ y = 75 :=
by {
  sorry
}

-- For question 2
theorem max_toys (cost_a cost_b q_a q_b : ℕ)
  (h₁ : cost_a = 50)
  (h₂ : cost_b = 75)
  (h₃ : q_b = 2 * q_a)
  (h₄ : 50 * q_a + 75 * q_b ≤ 20000) : q_a ≤ 100 :=
by {
  sorry
}

end unit_prices_max_toys_l3_3951


namespace problem1_problem2_l3_3626

-- Problem 1
theorem problem1 : (2 * Real.sqrt 12 - 3 * Real.sqrt (1 / 3)) * Real.sqrt 6 = 9 * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 (x : ℝ) (h1 : x / (2 * x - 1) = 2 - 3 / (1 - 2 * x)) : x = -1 / 3 := by
  sorry

end problem1_problem2_l3_3626


namespace product_of_two_numbers_l3_3507

theorem product_of_two_numbers (x y : ℕ) (h1 : x + y = 50) (h2 : x - y = 6) : x * y = 616 :=
sorry

end product_of_two_numbers_l3_3507


namespace ratio_of_x_to_y_l3_3228

variable (x y : ℝ)

theorem ratio_of_x_to_y (h : 0.10 * x = 0.20 * y) : x / y = 2 :=
by sorry

end ratio_of_x_to_y_l3_3228


namespace scienceStudyTime_l3_3301

def totalStudyTime : ℕ := 60
def mathStudyTime : ℕ := 35

theorem scienceStudyTime : totalStudyTime - mathStudyTime = 25 :=
by sorry

end scienceStudyTime_l3_3301


namespace triangle_inequality_l3_3689

variables {l_a l_b l_c m_a m_b m_c h_n m_n h_h_n m_m_p : ℝ}

-- Assuming some basic properties for the variables involved (all are positive in their respective triangle context)
axiom pos_l_a : 0 < l_a
axiom pos_l_b : 0 < l_b
axiom pos_l_c : 0 < l_c
axiom pos_m_a : 0 < m_a
axiom pos_m_b : 0 < m_b
axiom pos_m_c : 0 < m_c
axiom pos_h_n : 0 < h_n
axiom pos_m_n : 0 < m_n
axiom pos_h_h_n : 0 < h_h_n
axiom pos_m_m_p : 0 < m_m_p

theorem triangle_inequality :
  (h_n / m_n) + (h_n / h_h_n) + (l_c / m_m_p) > 1 :=
sorry

end triangle_inequality_l3_3689


namespace nonneg_int_solutions_to_ineq_system_l3_3634

open Set

theorem nonneg_int_solutions_to_ineq_system :
  {x : ℤ | (5 * x - 6 ≤ 2 * (x + 3)) ∧ ((x / 4 : ℚ) - 1 < (x - 2) / 3)} = {0, 1, 2, 3, 4} :=
by
  sorry

end nonneg_int_solutions_to_ineq_system_l3_3634


namespace seating_arrangements_l3_3710

def total_seats_front := 11
def total_seats_back := 12
def middle_seats_front := 3

def number_of_arrangements := 334

theorem seating_arrangements: 
  (total_seats_front - middle_seats_front) * (total_seats_front - middle_seats_front - 1) / 2 +
  (total_seats_back * (total_seats_back - 1)) / 2 +
  (total_seats_front - middle_seats_front) * total_seats_back +
  total_seats_back * (total_seats_front - middle_seats_front) = number_of_arrangements := 
sorry

end seating_arrangements_l3_3710


namespace quadratic_negativity_cond_l3_3533

theorem quadratic_negativity_cond {x m k : ℝ} :
  (∀ x, x^2 - m * x - k + m < 0) ↔ k > m - (m^2 / 4) :=
sorry

end quadratic_negativity_cond_l3_3533


namespace bowls_remaining_l3_3118

def initial_bowls : ℕ := 250

def customers_purchases : List (ℕ × ℕ) :=
  [(5, 7), (10, 15), (15, 22), (5, 36), (7, 46), (8, 0)]

def reward_ranges (bought : ℕ) : ℕ :=
  if bought >= 5 && bought <= 9 then 1
  else if bought >= 10 && bought <= 19 then 3
  else if bought >= 20 && bought <= 29 then 6
  else if bought >= 30 && bought <= 39 then 8
  else if bought >= 40 then 12
  else 0

def total_free_bowls : ℕ :=
  List.foldl (λ acc (n, b) => acc + n * reward_ranges b) 0 customers_purchases

theorem bowls_remaining :
  initial_bowls - total_free_bowls = 1 := by
  sorry

end bowls_remaining_l3_3118


namespace cost_of_27_lilies_l3_3320

theorem cost_of_27_lilies
  (cost_18 : ℕ)
  (price_ratio : ℕ → ℕ → Prop)
  (h_cost_18 : cost_18 = 30)
  (h_price_ratio : ∀ n m c : ℕ, price_ratio n m ↔ c = n * 5 / 3 ∧ m = c * 3 / 5) :
  ∃ c : ℕ, price_ratio 27 c ∧ c = 45 := 
by
  sorry

end cost_of_27_lilies_l3_3320


namespace n_cubed_plus_5_div_by_6_l3_3594

theorem n_cubed_plus_5_div_by_6  (n : ℤ) : 6 ∣ n * (n^2 + 5) :=
sorry

end n_cubed_plus_5_div_by_6_l3_3594


namespace find_m_l3_3926

open Real

noncomputable def vec_a : ℝ × ℝ := (-1, 2)
noncomputable def vec_b (m : ℝ) : ℝ × ℝ := (m, 3)

theorem find_m (m : ℝ) (h : -1 * m + 2 * 3 = 0) : m = 6 :=
sorry

end find_m_l3_3926


namespace bonnie_roark_wire_length_ratio_l3_3200

noncomputable def ratio_of_wire_lengths : ℚ :=
let bonnie_wire_per_piece := 8
let bonnie_pieces := 12
let bonnie_total_wire := bonnie_pieces * bonnie_wire_per_piece

let bonnie_side := bonnie_wire_per_piece
let bonnie_volume := bonnie_side^3

let roark_side := 2
let roark_volume := roark_side^3
let roark_cubes := bonnie_volume / roark_volume

let roark_wire_per_piece := 2
let roark_pieces_per_cube := 12
let roark_wire_per_cube := roark_pieces_per_cube * roark_wire_per_piece
let roark_total_wire := roark_cubes * roark_wire_per_cube

let ratio := bonnie_total_wire / roark_total_wire
ratio 

theorem bonnie_roark_wire_length_ratio :
  ratio_of_wire_lengths = (1 : ℚ) / 16 := 
sorry

end bonnie_roark_wire_length_ratio_l3_3200


namespace ratio_of_expenditures_l3_3604

theorem ratio_of_expenditures 
  (income_Uma : ℕ) (income_Bala : ℕ) (expenditure_Uma : ℕ) (expenditure_Bala : ℕ)
  (h_ratio_incomes : income_Uma / income_Bala = 4 / 3)
  (h_savings_Uma : income_Uma - expenditure_Uma = 5000)
  (h_savings_Bala : income_Bala - expenditure_Bala = 5000)
  (h_income_Uma : income_Uma = 20000) :
  expenditure_Uma / expenditure_Bala = 3 / 2 :=
sorry

end ratio_of_expenditures_l3_3604


namespace analysis_hours_l3_3349

theorem analysis_hours (n t : ℕ) (h1 : n = 206) (h2 : t = 1) : n * t = 206 := by
  sorry

end analysis_hours_l3_3349


namespace range_of_expression_l3_3280

theorem range_of_expression (x : ℝ) : (x + 2 ≥ 0 ∧ x - 1 ≠ 0) ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  sorry

end range_of_expression_l3_3280


namespace ticTacToeWinningDiagonals_l3_3996

-- Define the tic-tac-toe board and the conditions
def ticTacToeBoard : Type := Fin 3 × Fin 3
inductive Player | X | O

def isWinningDiagonal (board : ticTacToeBoard → Option Player) : Prop :=
  (board (0, 0) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 2) = some Player.O) ∨
  (board (0, 2) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 0) = some Player.O)

-- Define the main problem statement
theorem ticTacToeWinningDiagonals : ∃ (n : ℕ), n = 40 :=
  sorry

end ticTacToeWinningDiagonals_l3_3996


namespace tangent_line_equation_range_of_k_l3_3581

noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x

-- Part (I): Tangent line equation
theorem tangent_line_equation :
  let f (x : ℝ) := x^2 - x * Real.log x
  let p := (1 : ℝ)
  let y := f p
  (∀ x, y = x) :=
sorry

-- Part (II): Range of k
theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → (k / x + x / 2 - f x / x < 0)) → k ≤ 1 / 2 :=
sorry

end tangent_line_equation_range_of_k_l3_3581


namespace no_such_integers_exists_l3_3092

theorem no_such_integers_exists :
  ∀ (P : ℕ → ℕ), (∀ x, P x = x ^ 2000 - x ^ 1000 + 1) →
  ¬(∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
  (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k))) := 
by
  intro P hP notExists
  have contra : ∃ (a : Fin 8002 → ℕ), Function.Injective a ∧
    (∀ i j k : Fin 8002, i ≠ j → j ≠ k → i ≠ k → a i * a j * a k ∣ P (a i) * P (a j) * P (a k)) := notExists
  sorry

end no_such_integers_exists_l3_3092


namespace exists_perfect_square_intersection_l3_3651

theorem exists_perfect_square_intersection : ∃ n : ℕ, n > 1 ∧ ∃ k : ℕ, (2^n - n) = k^2 :=
by sorry

end exists_perfect_square_intersection_l3_3651


namespace solve_for_xy_l3_3944

theorem solve_for_xy (x y : ℕ) : 
  (4^x / 2^(x + y) = 16) ∧ (9^(x + y) / 3^(5 * y) = 81) → x * y = 32 :=
by
  sorry

end solve_for_xy_l3_3944


namespace simplify_and_evaluate_expr_l3_3919

theorem simplify_and_evaluate_expr 
  (x : ℝ) 
  (h : x = 1/2) : 
  (2 * x - 1) ^ 2 - (3 * x + 1) * (3 * x - 1) + 5 * x * (x - 1) = -5 / 2 := 
by
  sorry

end simplify_and_evaluate_expr_l3_3919


namespace negation_of_quadratic_statement_l3_3119

variable {x a b : ℝ}

theorem negation_of_quadratic_statement (h : x = a ∨ x = b) : x^2 - (a + b) * x + ab = 0 := sorry

end negation_of_quadratic_statement_l3_3119


namespace sin_log_infinite_zeros_in_01_l3_3340

noncomputable def f (x : ℝ) : ℝ := Real.sin (Real.log x)

theorem sin_log_infinite_zeros_in_01 : ∃ (S : Set ℝ), S = {x | 0 < x ∧ x < 1 ∧ f x = 0} ∧ Set.Infinite S := 
sorry

end sin_log_infinite_zeros_in_01_l3_3340


namespace cd_player_percentage_l3_3574

-- Define the percentage variables
def powerWindowsAndAntiLock : ℝ := 0.10
def antiLockAndCdPlayer : ℝ := 0.15
def powerWindowsAndCdPlayer : ℝ := 0.22
def cdPlayerAlone : ℝ := 0.38

-- Define the problem statement
theorem cd_player_percentage : 
  powerWindowsAndAntiLock = 0.10 → 
  antiLockAndCdPlayer = 0.15 → 
  powerWindowsAndCdPlayer = 0.22 → 
  cdPlayerAlone = 0.38 → 
  (antiLockAndCdPlayer + powerWindowsAndCdPlayer + cdPlayerAlone) = 0.75 :=
by
  intros
  sorry

end cd_player_percentage_l3_3574


namespace boys_less_than_two_fifths_total_l3_3871

theorem boys_less_than_two_fifths_total
  (n b g n1 n2 b1 b2 : ℕ)
  (h_total: n = b + g)
  (h_first_trip: b1 < 2 * n1 / 5)
  (h_second_trip: b2 < 2 * n2 / 5)
  (h_participation: b ≤ b1 + b2)
  (h_total_participants: n ≤ n1 + n2) :
  b < 2 * n / 5 := 
sorry

end boys_less_than_two_fifths_total_l3_3871


namespace negation_of_exists_l3_3444

open Classical

theorem negation_of_exists (p : Prop) : 
  (∃ x : ℝ, 2^x ≥ 2 * x + 1) ↔ ¬ ∀ x : ℝ, 2^x < 2 * x + 1 :=
by
  sorry

end negation_of_exists_l3_3444


namespace flower_bee_difference_proof_l3_3832

variable (flowers bees : ℕ)

def flowers_bees_difference (flowers bees : ℕ) : ℕ :=
  flowers - bees

theorem flower_bee_difference_proof : flowers_bees_difference 5 3 = 2 :=
by
  sorry

end flower_bee_difference_proof_l3_3832


namespace min_val_of_f_l3_3938

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

-- Theorem stating the minimum value of f(x) for x > 0 is 5.5
theorem min_val_of_f : ∀ x : ℝ, x > 0 → f x ≥ 5.5 :=
by sorry

end min_val_of_f_l3_3938


namespace main_problem_proof_l3_3355

def main_problem : Prop :=
  (1 : ℤ)^10 + (-1 : ℤ)^8 + (-1 : ℤ)^7 + (1 : ℤ)^5 = 2

theorem main_problem_proof : main_problem :=
by {
  sorry
}

end main_problem_proof_l3_3355


namespace labor_union_tree_equation_l3_3803

theorem labor_union_tree_equation (x : ℕ) : 2 * x + 21 = 3 * x - 24 := 
sorry

end labor_union_tree_equation_l3_3803


namespace quotient_of_division_l3_3628

theorem quotient_of_division (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 271) (h_divisor : divisor = 30) 
  (h_remainder : remainder = 1) (h_division : dividend = divisor * quotient + remainder) : 
  quotient = 9 := 
by 
  sorry

end quotient_of_division_l3_3628


namespace arithmetic_sequence_terms_count_l3_3052

theorem arithmetic_sequence_terms_count (a d l : Int) (h1 : a = 20) (h2 : d = -3) (h3 : l = -5) :
  ∃ n : Int, l = a + (n - 1) * d ∧ n = 8 :=
by
  sorry

end arithmetic_sequence_terms_count_l3_3052


namespace Paul_dig_days_alone_l3_3905

/-- Jake's daily work rate -/
def Jake_work_rate : ℚ := 1 / 16

/-- Hari's daily work rate -/
def Hari_work_rate : ℚ := 1 / 48

/-- Combined work rate of Jake, Paul, and Hari, when they work together they can dig the well in 8 days -/
def combined_work_rate (Paul_work_rate : ℚ) : Prop :=
  Jake_work_rate + Paul_work_rate + Hari_work_rate = 1 / 8

/-- Theorem stating that Paul can dig the well alone in 24 days -/
theorem Paul_dig_days_alone : ∃ (P : ℚ), combined_work_rate (1 / P) ∧ P = 24 :=
by
  use 24
  unfold combined_work_rate
  sorry

end Paul_dig_days_alone_l3_3905


namespace no_such_function_exists_l3_3080

theorem no_such_function_exists :
  ¬ ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = x ^ 2 - 1996 :=
by
  sorry

end no_such_function_exists_l3_3080


namespace prism_volume_l3_3684

open Real

theorem prism_volume :
  ∃ (a b c : ℝ), a * b = 15 ∧ b * c = 10 ∧ c * a = 30 ∧ a * b * c = 30 * sqrt 5 :=
by
  sorry

end prism_volume_l3_3684


namespace area_of_inscribed_rectangle_l3_3245

theorem area_of_inscribed_rectangle (r : ℝ) (h : r = 6) (ratio : ℝ) (hr : ratio = 3 / 1) :
  ∃ (length width : ℝ), (width = 2 * r) ∧ (length = ratio * width) ∧ (length * width = 432) :=
by
  sorry

end area_of_inscribed_rectangle_l3_3245


namespace father_cards_given_l3_3109

-- Defining the conditions
def Janessa_initial_cards : Nat := 4
def eBay_cards : Nat := 36
def bad_cards : Nat := 4
def dexter_cards : Nat := 29
def janessa_kept_cards : Nat := 20

-- Proving the number of cards father gave her
theorem father_cards_given : ∃ n : Nat, n = 13 ∧ (Janessa_initial_cards + eBay_cards - bad_cards + n = dexter_cards + janessa_kept_cards) := 
by
  sorry

end father_cards_given_l3_3109


namespace find_certain_number_l3_3954

theorem find_certain_number (x : ℝ) 
  (h : 3889 + x - 47.95000000000027 = 3854.002) : x = 12.95200000000054 :=
by
  sorry

end find_certain_number_l3_3954


namespace fraction_cal_handled_l3_3517

theorem fraction_cal_handled (Mabel Anthony Cal Jade : ℕ) 
  (h_Mabel : Mabel = 90)
  (h_Anthony : Anthony = Mabel + Mabel / 10)
  (h_Jade : Jade = 80)
  (h_Cal : Cal = Jade - 14) :
  (Cal : ℚ) / (Anthony : ℚ) = 2 / 3 :=
by
  sorry

end fraction_cal_handled_l3_3517


namespace tanya_dan_error_l3_3987

theorem tanya_dan_error 
  (a b c d e f g : ℤ)
  (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g)
  (h₇ : a % 2 = 1) (h₈ : b % 2 = 1) (h₉ : c % 2 = 1) (h₁₀ : d % 2 = 1) 
  (h₁₁ : e % 2 = 1) (h₁₂ : f % 2 = 1) (h₁₃ : g % 2 = 1)
  (h₁₄ : (a + b + c + d + e + f + g) / 7 - d = 3 / 7) :
  false :=
by sorry

end tanya_dan_error_l3_3987


namespace at_least_one_not_less_than_two_l3_3209

theorem at_least_one_not_less_than_two (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + 1/y ≥ 2) ∨ (y + 1/z ≥ 2) ∨ (z + 1/x ≥ 2) := sorry

end at_least_one_not_less_than_two_l3_3209


namespace sum_of_nonnegative_reals_l3_3261

theorem sum_of_nonnegative_reals (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 52) (h2 : a * b + b * c + c * a = 24) (h3 : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) :
  a + b + c = 10 :=
sorry

end sum_of_nonnegative_reals_l3_3261


namespace password_lock_probability_l3_3553

theorem password_lock_probability :
  (∃ (A B C : ℕ), 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 
   (∃ password : ℕ × ℕ × ℕ, password = (A, B, C) ∧ 
   (∃ chosen_middle_digit : ℕ, 0 <= chosen_middle_digit ∧ chosen_middle_digit <= 9 ∧
   chosen_middle_digit = B))) →
  (1/10 : ℝ) = 0.1 :=
by 
  sorry

end password_lock_probability_l3_3553


namespace stratified_sampling_sum_l3_3407

theorem stratified_sampling_sum :
  let grains := 40
  let vegetable_oils := 10
  let animal_foods := 30
  let fruits_and_vegetables := 20
  let sample_size := 20
  let total_food_types := grains + vegetable_oils + animal_foods + fruits_and_vegetables
  let sampling_fraction := sample_size / total_food_types
  let number_drawn := sampling_fraction * (vegetable_oils + fruits_and_vegetables)
  number_drawn = 6 :=
by
  sorry

end stratified_sampling_sum_l3_3407


namespace mean_home_runs_l3_3032

theorem mean_home_runs :
  let players_with_5 := 3
  let players_with_6 := 4
  let players_with_8 := 2
  let players_with_9 := 1
  let players_with_11 := 1
  let total_home_runs := (5 * players_with_5) + (6 * players_with_6) + (8 * players_with_8) + (9 * players_with_9) + (11 * players_with_11)
  let total_players := players_with_5 + players_with_6 + players_with_8 + players_with_9 + players_with_11
  (total_home_runs / total_players : ℚ) = 75 / 11 :=
by
  sorry

end mean_home_runs_l3_3032


namespace widget_cost_reduction_l3_3276

theorem widget_cost_reduction (W R : ℝ) (h1 : 6 * W = 36) (h2 : 8 * (W - R) = 36) : R = 1.5 :=
by
  sorry

end widget_cost_reduction_l3_3276


namespace four_fold_application_of_f_l3_3879

def f (x : ℕ) : ℕ :=
  if x % 3 = 0 then
    x / 3
  else
    5 * x + 2

theorem four_fold_application_of_f : f (f (f (f 3))) = 187 := 
  by
    sorry

end four_fold_application_of_f_l3_3879


namespace quadrilateral_with_equal_sides_is_rhombus_l3_3470

theorem quadrilateral_with_equal_sides_is_rhombus (a b c d : ℝ) (h1 : a = b) (h2 : b = c) (h3 : c = d) : a = d :=
by
  sorry

end quadrilateral_with_equal_sides_is_rhombus_l3_3470


namespace evaluate_expression_l3_3546

theorem evaluate_expression :
  (4 * 6) / (12 * 16) * (8 * 12 * 16) / (4 * 6 * 8) = 1 :=
by
  sorry

end evaluate_expression_l3_3546


namespace evaluate_expression_l3_3262

theorem evaluate_expression : (2^(2 + 1) - 4 * (2 - 1)^2)^2 = 16 :=
by
  sorry

end evaluate_expression_l3_3262


namespace students_wearing_other_colors_l3_3022

-- Definitions according to the problem conditions
def total_students : ℕ := 900
def percentage_blue : ℕ := 44
def percentage_red : ℕ := 28
def percentage_green : ℕ := 10

-- Goal: Prove the number of students who wear other colors
theorem students_wearing_other_colors :
  (total_students * (100 - (percentage_blue + percentage_red + percentage_green))) / 100 = 162 :=
by
  -- Skipping the proof steps with sorry
  sorry

end students_wearing_other_colors_l3_3022


namespace problem_a_l3_3075

theorem problem_a (nums : Fin 101 → ℤ) : ∃ i j : Fin 101, i ≠ j ∧ (nums i - nums j) % 100 = 0 := sorry

end problem_a_l3_3075


namespace distance_center_to_point_l3_3787

theorem distance_center_to_point : 
  let center := (2, 3)
  let point  := (5, -2)
  let distance := Real.sqrt ((5 - 2)^2 + (-2 - 3)^2)
  distance = Real.sqrt 34 := by
  sorry

end distance_center_to_point_l3_3787


namespace expression_equals_36_l3_3382

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l3_3382


namespace intersection_line_l3_3464

-- Define the equations of the circles in Cartesian coordinates.
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 * x = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y = 0

-- The theorem to prove.
theorem intersection_line (x y : ℝ) : circle1 x y ∧ circle2 x y → y + 4 * x = 0 :=
by
  sorry

end intersection_line_l3_3464


namespace time_difference_is_16_point_5_l3_3102

noncomputable def time_difference : ℝ :=
  let danny_to_steve : ℝ := 33
  let steve_to_danny := 2 * danny_to_steve -- Steve takes twice the time as Danny
  let emma_to_houses : ℝ := 40
  let danny_halfway := danny_to_steve / 2 -- Halfway point for Danny
  let steve_halfway := steve_to_danny / 2 -- Halfway point for Steve
  let emma_halfway := emma_to_houses / 2 -- Halfway point for Emma
  -- Additional times to the halfway point
  let steve_additional := steve_halfway - danny_halfway
  let emma_additional := emma_halfway - danny_halfway
  -- The final result is the maximum of these times
  max steve_additional emma_additional

theorem time_difference_is_16_point_5 : time_difference = 16.5 :=
  by
  sorry

end time_difference_is_16_point_5_l3_3102


namespace trade_in_value_of_old_phone_l3_3530

-- Define the given conditions
def cost_of_iphone : ℕ := 800
def earnings_per_week : ℕ := 80
def weeks_worked : ℕ := 7

-- Define the total earnings from babysitting
def total_earnings : ℕ := earnings_per_week * weeks_worked

-- Define the final proof statement
theorem trade_in_value_of_old_phone : cost_of_iphone - total_earnings = 240 :=
by
  unfold cost_of_iphone
  unfold total_earnings
  -- Substitute in the values
  have h1 : 800 - (80 * 7) = 240 := sorry
  exact h1

end trade_in_value_of_old_phone_l3_3530


namespace factorize_expression_l3_3443

variable (a x y : ℝ)

theorem factorize_expression : a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l3_3443


namespace find_Gary_gold_l3_3246

variable (G : ℕ) -- G represents the number of grams of gold Gary has.
variable (cost_Gary_gold_per_gram : ℕ) -- The cost per gram of Gary's gold.
variable (grams_Anna_gold : ℕ) -- The number of grams of gold Anna has.
variable (cost_Anna_gold_per_gram : ℕ) -- The cost per gram of Anna's gold.
variable (combined_cost : ℕ) -- The combined cost of both Gary's and Anna's gold.

theorem find_Gary_gold (h1 : cost_Gary_gold_per_gram = 15)
                       (h2 : grams_Anna_gold = 50)
                       (h3 : cost_Anna_gold_per_gram = 20)
                       (h4 : combined_cost = 1450)
                       (h5 : combined_cost = cost_Gary_gold_per_gram * G + grams_Anna_gold * cost_Anna_gold_per_gram) :
  G = 30 :=
by 
  sorry

end find_Gary_gold_l3_3246


namespace gcd_lcm_problem_l3_3627

theorem gcd_lcm_problem (b : ℤ) (x : ℕ) (hx_pos : 0 < x) (hx : x = 12) :
  gcd 30 b = x + 3 ∧ lcm 30 b = x * (x + 3) → b = 90 := 
by
  sorry

end gcd_lcm_problem_l3_3627


namespace arithmetic_progression_number_of_terms_l3_3612

variable (a d : ℕ)
variable (n : ℕ) (h_n_even : n % 2 = 0)
variable (h_sum_odd : (n / 2) * (2 * a + (n - 2) * d) = 60)
variable (h_sum_even : (n / 2) * (2 * (a + d) + (n - 2) * d) = 80)
variable (h_diff : (n - 1) * d = 16)

theorem arithmetic_progression_number_of_terms : n = 8 :=
by
  sorry

end arithmetic_progression_number_of_terms_l3_3612


namespace relationship_between_roses_and_total_flowers_l3_3570

variables (C V T R F : ℝ)
noncomputable def F_eq_64_42376521116678_percent_of_C := 
  C = 0.6442376521116678 * F

def V_eq_one_third_of_C := 
  V = (1 / 3) * C

def T_eq_one_ninth_of_C := 
  T = (1 / 9) * C

def F_eq_C_plus_V_plus_T_plus_R := 
  F = C + V + T + R

theorem relationship_between_roses_and_total_flowers (C V T R F : ℝ) 
    (h1 : C = 0.6442376521116678 * F)
    (h2 : V = 1 / 3 * C)
    (h3 : T = 1 / 9 * C)
    (h4 : F = C + V + T + R) :
    R = F - 13 / 9 * C := 
  by sorry

end relationship_between_roses_and_total_flowers_l3_3570


namespace yesterday_tomorrow_is_friday_l3_3870

-- Defining the days of the week
inductive Day
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to go to the next day
def next_day : Day → Day
| Sunday    => Monday
| Monday    => Tuesday
| Tuesday   => Wednesday
| Wednesday => Thursday
| Thursday  => Friday
| Friday    => Saturday
| Saturday  => Sunday

-- Function to go to the previous day
def previous_day : Day → Day
| Sunday    => Saturday
| Monday    => Sunday
| Tuesday   => Monday
| Wednesday => Tuesday
| Thursday  => Wednesday
| Friday    => Thursday
| Saturday  => Friday

-- Proving the statement
theorem yesterday_tomorrow_is_friday (T : Day) (H : next_day (previous_day T) = Thursday) : previous_day (next_day (next_day T)) = Friday :=
by
  sorry

end yesterday_tomorrow_is_friday_l3_3870


namespace part_I_part_II_l3_3346

noncomputable def M : Set ℝ := { x | |x + 1| + |x - 1| ≤ 2 }

theorem part_I : M = Set.Icc (-1 : ℝ) (1 : ℝ) := 
sorry

theorem part_II (x y z : ℝ) (hx : x ∈ M) (hy : |y| ≤ (1/6)) (hz : |z| ≤ (1/9)) :
  |x + 2 * y - 3 * z| ≤ (5/3) :=
by
  sorry

end part_I_part_II_l3_3346


namespace field_length_l3_3571

theorem field_length (w l : ℕ) (Pond_Area : ℕ) (Pond_Field_Ratio : ℚ) (Field_Length_Ratio : ℕ) 
  (h1 : Length = 2 * Width)
  (h2 : Pond_Area = 8 * 8)
  (h3 : Pond_Field_Ratio = 1 / 50)
  (h4 : Pond_Area = Pond_Field_Ratio * Field_Area)
  : l = 80 := 
by
  -- begin solution
  sorry

end field_length_l3_3571


namespace four_transformations_of_1989_l3_3993

-- Definition of the sum of the digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Initial number
def initial_number : ℕ := 1989

-- Theorem statement
theorem four_transformations_of_1989 : 
  sum_of_digits (sum_of_digits (sum_of_digits (sum_of_digits initial_number))) = 9 :=
by
  sorry

end four_transformations_of_1989_l3_3993


namespace polynomial_condition_l3_3994

noncomputable def polynomial_of_degree_le (n : ℕ) (P : Polynomial ℝ) :=
  P.degree ≤ n

noncomputable def has_nonneg_coeff (P : Polynomial ℝ) :=
  ∀ i, 0 ≤ P.coeff i

theorem polynomial_condition
  (n : ℕ) (P : Polynomial ℝ)
  (h1 : polynomial_of_degree_le n P)
  (h2 : has_nonneg_coeff P)
  (h3 : ∀ x : ℝ, x > 0 → P.eval x * P.eval (1 / x) ≤ (P.eval 1) ^ 2) : 
  ∃ a_n : ℝ, 0 ≤ a_n ∧ P = Polynomial.C a_n * Polynomial.X^n :=
sorry

end polynomial_condition_l3_3994


namespace work_days_l3_3725

theorem work_days (A B : ℝ) (h1 : A = 2 * B) (h2 : B = 1 / 18) :
    1 / (A + B) = 6 :=
by
  sorry

end work_days_l3_3725


namespace algebraic_expression_value_l3_3838

theorem algebraic_expression_value (m: ℝ) (h: m^2 + m - 1 = 0) : 2023 - m^2 - m = 2022 := 
by 
  sorry

end algebraic_expression_value_l3_3838


namespace set_intersection_complement_eq_l3_3012

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 4, 6}
def B : Set ℕ := {2, 4, 5, 6}

noncomputable def complement (U B : Set ℕ) : Set ℕ := { x ∈ U | x ∉ B }

theorem set_intersection_complement_eq : (A ∩ (complement U B)) = {1, 3} := 
by 
  sorry

end set_intersection_complement_eq_l3_3012


namespace right_triangle_hypotenuse_length_l3_3427

theorem right_triangle_hypotenuse_length
  (a b : ℝ)
  (ha : a = 12)
  (hb : b = 16) :
  c = 20 :=
by
  -- Placeholder for the proof
  sorry

end right_triangle_hypotenuse_length_l3_3427


namespace complement_of_A_is_correct_l3_3385

-- Define the universal set U and the set A.
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3}

-- Define the complement of A with respect to U.
def A_complement : Set ℕ := {x ∈ U | x ∉ A}

-- The theorem statement that the complement of A in U is {2, 4}.
theorem complement_of_A_is_correct : A_complement = {2, 4} :=
sorry

end complement_of_A_is_correct_l3_3385


namespace units_digit_quotient_eq_one_l3_3088

theorem units_digit_quotient_eq_one :
  (2^2023 + 3^2023) / 5 % 10 = 1 := by
  sorry

end units_digit_quotient_eq_one_l3_3088


namespace find_N_l3_3055

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem find_N (N : ℕ) (hN1 : N < 10000)
  (hN2 : N = 26 * sum_of_digits N) : N = 234 ∨ N = 468 := 
  sorry

end find_N_l3_3055


namespace line_bisects_circle_l3_3491

theorem line_bisects_circle
  (C : Type)
  [MetricSpace C]
  (x y : ℝ)
  (h : ∀ {x y : ℝ}, x^2 + y^2 - 2*x - 4*y + 1 = 0) : 
  x - y + 1 = 0 → True :=
by
  intro h_line
  sorry

end line_bisects_circle_l3_3491


namespace y_capital_l3_3840

theorem y_capital (X Y Z : ℕ) (Pz : ℕ) (Z_months_after_start : ℕ) (total_profit Z_share : ℕ)
    (hx : X = 20000)
    (hz : Z = 30000)
    (hz_profit : Z_share = 14000)
    (htotal_profit : total_profit = 50000)
    (hZ_months : Z_months_after_start = 5)
  : Y = 25000 := 
by
  -- Here we would have a proof, skipped with sorry for now
  sorry

end y_capital_l3_3840


namespace distinctPaintedCubeConfigCount_l3_3897

-- Define a painted cube with given face colors
structure PaintedCube where
  blue_face : ℤ
  yellow_faces : Finset ℤ
  red_faces : Finset ℤ
  -- Ensure logical conditions about faces
  face_count : blue_face ∉ yellow_faces ∧ blue_face ∉ red_faces ∧
               yellow_faces ∩ red_faces = ∅ ∧ yellow_faces.card = 2 ∧
               red_faces.card = 3

-- There are no orientation-invariant rotations that change the configuration
def equivPaintedCube (c1 c2 : PaintedCube) : Prop :=
  ∃ (r: ℤ), 
    -- rotate c1 by r to get c2
    true -- placeholder for rotation logic

-- The set of all possible distinct painted cubes under rotation constraints is defined
def possibleConfigurations : Finset PaintedCube :=
  sorry  -- construct this set considering rotations

-- The main proposition
theorem distinctPaintedCubeConfigCount : (possibleConfigurations.card = 4) :=
  sorry

end distinctPaintedCubeConfigCount_l3_3897


namespace fraction_of_shoppers_avoiding_checkout_l3_3705

theorem fraction_of_shoppers_avoiding_checkout 
  (total_shoppers : ℕ) 
  (shoppers_at_checkout : ℕ) 
  (h1 : total_shoppers = 480) 
  (h2 : shoppers_at_checkout = 180) : 
  (total_shoppers - shoppers_at_checkout) / total_shoppers = 5 / 8 :=
by
  sorry

end fraction_of_shoppers_avoiding_checkout_l3_3705


namespace dexter_filled_fewer_boxes_with_football_cards_l3_3502

-- Conditions
def boxes_with_basketball_cards : ℕ := 9
def cards_per_basketball_box : ℕ := 15
def cards_per_football_box : ℕ := 20
def total_cards : ℕ := 255

-- Definition of the main problem statement
def fewer_boxes_with_football_cards : Prop :=
  let basketball_cards := boxes_with_basketball_cards * cards_per_basketball_box
  let football_cards := total_cards - basketball_cards
  let boxes_with_football_cards := football_cards / cards_per_football_box
  boxes_with_basketball_cards - boxes_with_football_cards = 3

theorem dexter_filled_fewer_boxes_with_football_cards : fewer_boxes_with_football_cards :=
by
  sorry

end dexter_filled_fewer_boxes_with_football_cards_l3_3502


namespace ellipse_focus_distance_l3_3927

theorem ellipse_focus_distance : ∀ (x y : ℝ), 9 * x^2 + y^2 = 900 → 2 * Real.sqrt (10^2 - 30^2) = 40 * Real.sqrt 2 :=
by
  intros x y h
  sorry

end ellipse_focus_distance_l3_3927


namespace three_xy_eq_24_l3_3130

variable {x y : ℝ}

theorem three_xy_eq_24 (h : x * (x + 3 * y) = x^2 + 24) : 3 * x * y = 24 :=
sorry

end three_xy_eq_24_l3_3130


namespace julia_cookies_l3_3254

theorem julia_cookies (N : ℕ) 
  (h1 : N % 6 = 5) 
  (h2 : N % 8 = 7) 
  (h3 : N < 100) : 
  N = 17 ∨ N = 41 ∨ N = 65 ∨ N = 89 → 17 + 41 + 65 + 89 = 212 :=
sorry

end julia_cookies_l3_3254


namespace gcd_squares_example_l3_3715

noncomputable def gcd_of_squares : ℕ :=
  Nat.gcd (101 ^ 2 + 202 ^ 2 + 303 ^ 2) (100 ^ 2 + 201 ^ 2 + 304 ^ 2)

theorem gcd_squares_example : gcd_of_squares = 3 :=
by
  sorry

end gcd_squares_example_l3_3715


namespace divisible_by_3_l3_3790

theorem divisible_by_3 (x y : ℤ) (h : (x^2 + y^2) % 3 = 0) : x % 3 = 0 ∧ y % 3 = 0 :=
sorry

end divisible_by_3_l3_3790


namespace sawyer_saw_octopuses_l3_3162

def number_of_legs := 40
def legs_per_octopus := 8

theorem sawyer_saw_octopuses : number_of_legs / legs_per_octopus = 5 := 
by
  sorry

end sawyer_saw_octopuses_l3_3162


namespace triangle_perimeters_sum_l3_3916

theorem triangle_perimeters_sum :
  ∃ (t : ℕ),
    (∀ (A B C D : Type) (x y : ℕ), 
      (AB = 7 ∧ BC = 17 ∧ AD = x ∧ CD = x ∧ BD = y ∧ x^2 - y^2 = 240) →
      t = 114) :=
sorry

end triangle_perimeters_sum_l3_3916


namespace machine_value_after_two_years_l3_3242

noncomputable def machine_market_value (initial_value : ℝ) (years : ℕ) (decrease_rate : ℝ) : ℝ :=
  initial_value * (1 - decrease_rate) ^ years

theorem machine_value_after_two_years :
  machine_market_value 8000 2 0.2 = 5120 := by
  sorry

end machine_value_after_two_years_l3_3242


namespace width_to_length_ratio_l3_3300

variables {w l P : ℕ}

theorem width_to_length_ratio :
  l = 10 → P = 30 → P = 2 * (l + w) → (w : ℚ) / l = 1 / 2 :=
by
  intro h1 h2 h3
  -- Noncomputable definition for rational division
  -- (ℚ is used for exact rational division)
  sorry

#check width_to_length_ratio

end width_to_length_ratio_l3_3300


namespace barbara_total_candies_l3_3426

theorem barbara_total_candies :
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855 := 
by
  let boxes1 := 9
  let candies_per_box1 := 25
  let boxes2 := 18
  let candies_per_box2 := 35
  show boxes1 * candies_per_box1 + boxes2 * candies_per_box2 = 855
  sorry

end barbara_total_candies_l3_3426


namespace min_positive_period_cos_2x_l3_3746

noncomputable def f (x : ℝ) : ℝ := Real.cos (2 * x)

theorem min_positive_period_cos_2x :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ ∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T := 
sorry

end min_positive_period_cos_2x_l3_3746


namespace polynomial_evaluation_l3_3326

-- Define the polynomial p(x) and the conditions
noncomputable def p (x : ℝ) (a b c d : ℝ) : ℝ := x^4 + a * x^3 + b * x^2 + c * x + d

-- Given conditions for p(1), p(2), p(3)
variables (a b c d : ℝ)
axiom h₁ : p 1 a b c d = 1993
axiom h₂ : p 2 a b c d = 3986
axiom h₃ : p 3 a b c d = 5979

-- The final proof statement
theorem polynomial_evaluation :
  (1 / 4) * (p 11 a b c d + p (-7) a b c d) = 5233 :=
sorry

end polynomial_evaluation_l3_3326


namespace ratio_circle_to_triangle_area_l3_3161

theorem ratio_circle_to_triangle_area 
  (h d : ℝ) 
  (h_pos : 0 < h) 
  (d_pos : 0 < d) 
  (R : ℝ) 
  (R_def : R = h / 2) :
  (π * R^2) / (1/2 * h * d) = (π * h) / (2 * d) :=
by sorry

end ratio_circle_to_triangle_area_l3_3161


namespace percentage_A_of_B_l3_3248

variable {A B C D : ℝ}

theorem percentage_A_of_B (
  h1: A = 0.125 * C)
  (h2: B = 0.375 * D)
  (h3: D = 1.225 * C)
  (h4: C = 0.805 * B) :
  A = 0.100625 * B := by
  -- Sufficient proof steps would go here
  sorry

end percentage_A_of_B_l3_3248


namespace segment_length_of_points_A_l3_3946

-- Define the basic setup
variable (d BA CA : ℝ)
variable {A B C : Point} -- Assume we have a type Point for the geometric points

-- Establish some conditions: A right triangle with given lengths
def is_right_triangle (A B C : Point) : Prop := sorry -- Placeholder for definition

def distance (P Q : Point) : ℝ := sorry -- Placeholder for the distance function

-- Conditions
variables (h_right_triangle : is_right_triangle A B C)
variables (h_hypotenuse : distance B C = d)
variables (h_smallest_leg : min (distance B A) (distance C A) = min BA CA)

-- The theorem statement
theorem segment_length_of_points_A (h_right_triangle : is_right_triangle A B C)
                                    (h_hypotenuse : distance B C = d)
                                    (h_smallest_leg : min (distance B A) (distance C A) = min BA CA) :
  ∃ A, (∀ t : ℝ, distance O A = d - min BA CA) :=
sorry -- Proof to be provided

end segment_length_of_points_A_l3_3946


namespace sum_modulo_9_l3_3639

theorem sum_modulo_9 :
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := 
by
  -- Skipping the detailed proof steps
  sorry

end sum_modulo_9_l3_3639


namespace percent_of_x_is_y_l3_3692

variables (x y : ℝ)

theorem percent_of_x_is_y (h : 0.30 * (x - y) = 0.20 * (x + y)) : y = 0.20 * x :=
by sorry

end percent_of_x_is_y_l3_3692


namespace minutes_per_mile_l3_3259

-- Define the total distance Peter needs to walk
def total_distance : ℝ := 2.5

-- Define the distance Peter has already walked
def walked_distance : ℝ := 1.0

-- Define the remaining time Peter needs to walk to reach the grocery store
def remaining_time : ℝ := 30.0

-- Define the remaining distance Peter needs to walk
def remaining_distance : ℝ := total_distance - walked_distance

-- The desired statement to prove: it takes Peter 20 minutes to walk one mile
theorem minutes_per_mile : remaining_distance / remaining_time = 1.0 / 20.0 := by
  sorry

end minutes_per_mile_l3_3259


namespace base9_square_multiple_of_3_ab4c_l3_3867

theorem base9_square_multiple_of_3_ab4c (a b c : ℕ) (N : ℕ) (h1 : a ≠ 0)
  (h2 : N = a * 9^3 + b * 9^2 + 4 * 9 + c)
  (h3 : ∃ k : ℕ, N = k^2)
  (h4 : N % 3 = 0) :
  c = 0 :=
sorry

end base9_square_multiple_of_3_ab4c_l3_3867


namespace utility_bill_amount_l3_3057

/-- Mrs. Brown's utility bill amount given her payments in specific denominations. -/
theorem utility_bill_amount : 
  let fifty_bills := 3 * 50
  let ten_bills := 2 * 10
  fifty_bills + ten_bills = 170 := 
by
  rfl

end utility_bill_amount_l3_3057


namespace number_of_trailing_zeros_l3_3707

def trailing_zeros (n : Nat) : Nat :=
  let powers_of_two := 2 * 52^5
  let powers_of_five := 2 * 25^2
  min powers_of_two powers_of_five

theorem number_of_trailing_zeros : trailing_zeros (525^(25^2) * 252^(52^5)) = 1250 := 
by sorry

end number_of_trailing_zeros_l3_3707


namespace quadratic_part_of_equation_l3_3745

theorem quadratic_part_of_equation (x: ℝ) :
  (x^2 - 8*x + 21 = |x - 5| + 4) → (x^2 - 8*x + 21) = x^2 - 8*x + 21 :=
by
  intros h
  sorry

end quadratic_part_of_equation_l3_3745


namespace ratio_sum_ineq_l3_3367

theorem ratio_sum_ineq 
  (a b α β : ℝ) 
  (hαβ : 0 < α ∧ 0 < β) 
  (h_range : α ≤ a ∧ a ≤ β ∧ α ≤ b ∧ b ≤ β) : 
  (b / a + a / b ≤ β / α + α / β) ∧ 
  (b / a + a / b = β / α + α / β ↔ (a = α ∧ b = β ∨ a = β ∧ b = α)) :=
by
  sorry

end ratio_sum_ineq_l3_3367


namespace train_speed_l3_3788

theorem train_speed (v : ℝ) (d : ℝ) : 
  (v > 0) →
  (d > 0) →
  (d + (d - 55) = 495) →
  (d / v = (d - 55) / 25) →
  v = 31.25 := 
by
  intros hv hd hdist heqn
  -- We can leave the proof part out because we only need the statement
  sorry

end train_speed_l3_3788


namespace minValue_Proof_l3_3483

noncomputable def minValue (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) : Prop :=
  ∃ m : ℝ, m = 4.5 ∧ (∀ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ a + b + c = 2 → (1/a + 1/b + 1/c) ≥ 9/2)

theorem minValue_Proof :
  ∀ (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2), 
    minValue x y z h1 h2 h3 h4 := by
  sorry

end minValue_Proof_l3_3483


namespace maximum_area_of_rectangle_l3_3781

theorem maximum_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 40) : ∃ A, A = 100 ∧ ∀ x' y', 2 * x' + 2 * y' = 40 → x' * y' ≤ A := by
  sorry

end maximum_area_of_rectangle_l3_3781


namespace laura_rental_cost_l3_3696

def rental_cost_per_day : ℝ := 30
def driving_cost_per_mile : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 300

theorem laura_rental_cost : rental_cost_per_day * days_rented + driving_cost_per_mile * miles_driven = 165 := by
  sorry

end laura_rental_cost_l3_3696


namespace train_length_calculation_l3_3843

theorem train_length_calculation (L : ℝ) (t : ℝ) (v_faster : ℝ) (v_slower : ℝ) (relative_speed : ℝ) (total_distance : ℝ) :
  (v_faster = 60) →
  (v_slower = 40) →
  (relative_speed = (v_faster - v_slower) * 1000 / 3600) →
  (t = 48) →
  (total_distance = relative_speed * t) →
  (2 * L = total_distance) →
  L = 133.44 :=
by
  intros
  sorry

end train_length_calculation_l3_3843


namespace largest_possible_cylindrical_tank_radius_in_crate_l3_3947

theorem largest_possible_cylindrical_tank_radius_in_crate
  (crate_length : ℝ) (crate_width : ℝ) (crate_height : ℝ)
  (cylinder_height : ℝ) (cylinder_radius : ℝ)
  (h_cube : crate_length = 20 ∧ crate_width = 20 ∧ crate_height = 20)
  (h_cylinder_in_cube : cylinder_height = 20 ∧ 2 * cylinder_radius ≤ 20) :
  cylinder_radius = 10 :=
sorry

end largest_possible_cylindrical_tank_radius_in_crate_l3_3947


namespace solve_equation_l3_3359

theorem solve_equation (x : ℝ) : (x + 2) * (x + 1) = 3 * (x + 1) ↔ (x = -1 ∨ x = 1) :=
by sorry

end solve_equation_l3_3359


namespace frosting_problem_equivalent_l3_3142

/-
Problem:
Cagney can frost a cupcake every 15 seconds.
Lacey can frost a cupcake every 40 seconds.
Mack can frost a cupcake every 25 seconds.
Prove that together they can frost 79 cupcakes in 10 minutes.
-/

def cupcakes_frosted_together_in_10_minutes (rate_cagney rate_lacey rate_mack : ℕ) (time_minutes : ℕ) : ℕ :=
  let time_seconds := time_minutes * 60
  let rate_cagney := 1 / 15
  let rate_lacey := 1 / 40
  let rate_mack := 1 / 25
  let combined_rate := rate_cagney + rate_lacey + rate_mack
  combined_rate * time_seconds

theorem frosting_problem_equivalent:
  cupcakes_frosted_together_in_10_minutes 1 1 1 10 = 79 := by
  sorry

end frosting_problem_equivalent_l3_3142


namespace range_of_m_l3_3006

open Set

variable {m : ℝ}

def A : Set ℝ := { x | x^2 < 16 }
def B (m : ℝ) : Set ℝ := { x | x < m }

theorem range_of_m (h : A ∩ B m = A) : 4 ≤ m :=
by
  sorry

end range_of_m_l3_3006


namespace vectors_parallel_l3_3963

theorem vectors_parallel (m : ℝ) (a : ℝ × ℝ := (m, -1)) (b : ℝ × ℝ := (1, m + 2)) :
  (∃ k : ℝ, a = (k * b.1, k * b.2)) → m = -1 := by
  sorry

end vectors_parallel_l3_3963


namespace find_ab_from_conditions_l3_3357

theorem find_ab_from_conditions (a b : ℝ) (h1 : a^2 + b^2 = 5) (h2 : a + b = 3) : a * b = 2 := 
by
  sorry

end find_ab_from_conditions_l3_3357


namespace product_eq_one_l3_3190

noncomputable def f (x : ℝ) : ℝ := |Real.logb 3 x|

theorem product_eq_one (a b : ℝ) (h_diff : a ≠ b) (h_eq : f a = f b) : a * b = 1 := by
  sorry

end product_eq_one_l3_3190


namespace millet_exceeds_half_l3_3539

noncomputable def seeds_millet_day (n : ℕ) : ℝ :=
  0.2 * (1 - 0.7 ^ n) / (1 - 0.7) + 0.2 * 0.7 ^ n

noncomputable def seeds_other_day (n : ℕ) : ℝ :=
  0.3 * (1 - 0.1 ^ n) / (1 - 0.1) + 0.3 * 0.1 ^ n

noncomputable def prop_millet (n : ℕ) : ℝ :=
  seeds_millet_day n / (seeds_millet_day n + seeds_other_day n)

theorem millet_exceeds_half : ∃ n : ℕ, prop_millet n > 0.5 ∧ n = 3 :=
by sorry

end millet_exceeds_half_l3_3539


namespace man_walking_time_l3_3344

section TrainProblem

variables {T W : ℕ}

/-- Each day a man meets his wife at the train station after work,
    and then she drives him home. She always arrives exactly on time to pick him up.
    One day he catches an earlier train and arrives at the station an hour early.
    He immediately begins walking home along the same route the wife drives.
    Eventually, his wife sees him on her way to the station and drives him the rest of the way home.
    When they arrive home, the man notices that they arrived 30 minutes earlier than usual.
    How much time did the man spend walking? -/
theorem man_walking_time : 
    (∃ (T : ℕ), T > 30 ∧ (W = T - 30) ∧ (W + 30 = T)) → W = 30 :=
sorry

end TrainProblem

end man_walking_time_l3_3344


namespace check_conditions_l3_3304

noncomputable def f (x a b : ℝ) : ℝ := |x^2 - 2 * a * x + b|

theorem check_conditions (a b : ℝ) :
  ¬ (∀ x : ℝ, f x a b = f (-x) a b) ∧         -- f(x) is not necessarily an even function
  ¬ (∀ x : ℝ, (f 0 a b = f 2 a b → (f x a b = f (2 - x) a b))) ∧ -- No guaranteed symmetry about x=1
  (a^2 - b^2 ≤ 0 → ∀ x : ℝ, x ≥ a → ∀ y : ℝ, y ≥ x → f y a b ≥ f x a b) ∧ -- f(x) is increasing on [a, +∞) if a^2 - b^2 ≤ 0
  ¬ (∀ x : ℝ, f x a b ≤ |a^2 - b|)         -- f(x) does not necessarily have a max value of |a^2 - b|
:= sorry

end check_conditions_l3_3304


namespace marbles_in_jar_l3_3528

theorem marbles_in_jar (M : ℕ) (h1 : M / 24 = 24 * 26 / 26) (h2 : M / 26 + 1 = M / 24) : M = 312 := by
  sorry

end marbles_in_jar_l3_3528


namespace fraction_inequality_l3_3191

theorem fraction_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a < b) (h2 : c < d) : (a + c) / (b + c) < (a + d) / (b + d) :=
by
  sorry

end fraction_inequality_l3_3191


namespace diminished_value_is_seven_l3_3929

theorem diminished_value_is_seven (x y : ℕ) (hx : x = 280)
  (h_eq : x / 5 + 7 = x / 4 - y) : y = 7 :=
by {
  sorry
}

end diminished_value_is_seven_l3_3929


namespace find_matrix_triples_elements_l3_3023

theorem find_matrix_triples_elements (M A : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ (a b c d : ℝ), A = ![![a, b], ![c, d]] -> M * A = ![![3 * a, 3 * b], ![3 * c, 3 * d]]) :
  M = ![![3, 0], ![0, 3]] :=
by
  sorry

end find_matrix_triples_elements_l3_3023


namespace Bert_sandwiches_left_l3_3294

theorem Bert_sandwiches_left : (Bert:Type) → 
  (sandwiches_made : ℕ) → 
  sandwiches_made = 12 → 
  (sandwiches_eaten_day1 : ℕ) → 
  sandwiches_eaten_day1 = sandwiches_made / 2 → 
  (sandwiches_eaten_day2 : ℕ) → 
  sandwiches_eaten_day2 = sandwiches_eaten_day1 - 2 →
  (sandwiches_left : ℕ) → 
  sandwiches_left = sandwiches_made - (sandwiches_eaten_day1 + sandwiches_eaten_day2) → 
  sandwiches_left = 2 := 
  sorry

end Bert_sandwiches_left_l3_3294


namespace equal_sets_d_l3_3520

theorem equal_sets_d : 
  (let M := {x | x^2 - 3*x + 2 = 0}
   let N := {1, 2}
   M = N) :=
by 
  sorry

end equal_sets_d_l3_3520


namespace condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l3_3819

variables {a b : ℝ}

theorem condition_3_implies_at_least_one_gt_one (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

theorem condition_5_implies_at_least_one_gt_one (h : ab > 1) : a > 1 ∨ b > 1 :=
sorry

end condition_3_implies_at_least_one_gt_one_condition_5_implies_at_least_one_gt_one_l3_3819


namespace find_distance_l3_3915

-- Definitions based on conditions
def speed : ℝ := 75 -- in km/hr
def time : ℝ := 4 -- in hr

-- Statement to be proved
theorem find_distance : speed * time = 300 := by
  sorry

end find_distance_l3_3915


namespace candy_cost_55_cents_l3_3297

theorem candy_cost_55_cents
  (paid: ℕ) (change: ℕ) (num_coins: ℕ)
  (coin1 coin2 coin3 coin4: ℕ)
  (h1: paid = 100)
  (h2: num_coins = 4)
  (h3: coin1 = 25)
  (h4: coin2 = 10)
  (h5: coin3 = 10)
  (h6: coin4 = 0)
  (h7: change = coin1 + coin2 + coin3 + coin4) :
  paid - change = 55 :=
by
  -- The proof can be provided here.
  sorry

end candy_cost_55_cents_l3_3297


namespace bc_possible_values_l3_3370

theorem bc_possible_values (a b c : ℝ) 
  (h1 : a + b + c = 100) 
  (h2 : ab + bc + ca = 20) 
  (h3 : (a + b) * (a + c) = 24) : 
  bc = -176 ∨ bc = 224 :=
by
  sorry

end bc_possible_values_l3_3370


namespace chocolate_cost_l3_3377

theorem chocolate_cost (Ccb Cc : ℝ) (h1 : Ccb = 6) (h2 : Ccb = Cc + 3) : Cc = 3 :=
by
  sorry

end chocolate_cost_l3_3377


namespace tangent_product_power_l3_3446

noncomputable def tangent_product : ℝ :=
  (1 + Real.tan (1 * Real.pi / 180))
  * (1 + Real.tan (2 * Real.pi / 180))
  * (1 + Real.tan (3 * Real.pi / 180))
  * (1 + Real.tan (4 * Real.pi / 180))
  * (1 + Real.tan (5 * Real.pi / 180))
  * (1 + Real.tan (6 * Real.pi / 180))
  * (1 + Real.tan (7 * Real.pi / 180))
  * (1 + Real.tan (8 * Real.pi / 180))
  * (1 + Real.tan (9 * Real.pi / 180))
  * (1 + Real.tan (10 * Real.pi / 180))
  * (1 + Real.tan (11 * Real.pi / 180))
  * (1 + Real.tan (12 * Real.pi / 180))
  * (1 + Real.tan (13 * Real.pi / 180))
  * (1 + Real.tan (14 * Real.pi / 180))
  * (1 + Real.tan (15 * Real.pi / 180))
  * (1 + Real.tan (16 * Real.pi / 180))
  * (1 + Real.tan (17 * Real.pi / 180))
  * (1 + Real.tan (18 * Real.pi / 180))
  * (1 + Real.tan (19 * Real.pi / 180))
  * (1 + Real.tan (20 * Real.pi / 180))
  * (1 + Real.tan (21 * Real.pi / 180))
  * (1 + Real.tan (22 * Real.pi / 180))
  * (1 + Real.tan (23 * Real.pi / 180))
  * (1 + Real.tan (24 * Real.pi / 180))
  * (1 + Real.tan (25 * Real.pi / 180))
  * (1 + Real.tan (26 * Real.pi / 180))
  * (1 + Real.tan (27 * Real.pi / 180))
  * (1 + Real.tan (28 * Real.pi / 180))
  * (1 + Real.tan (29 * Real.pi / 180))
  * (1 + Real.tan (30 * Real.pi / 180))
  * (1 + Real.tan (31 * Real.pi / 180))
  * (1 + Real.tan (32 * Real.pi / 180))
  * (1 + Real.tan (33 * Real.pi / 180))
  * (1 + Real.tan (34 * Real.pi / 180))
  * (1 + Real.tan (35 * Real.pi / 180))
  * (1 + Real.tan (36 * Real.pi / 180))
  * (1 + Real.tan (37 * Real.pi / 180))
  * (1 + Real.tan (38 * Real.pi / 180))
  * (1 + Real.tan (39 * Real.pi / 180))
  * (1 + Real.tan (40 * Real.pi / 180))
  * (1 + Real.tan (41 * Real.pi / 180))
  * (1 + Real.tan (42 * Real.pi / 180))
  * (1 + Real.tan (43 * Real.pi / 180))
  * (1 + Real.tan (44 * Real.pi / 180))
  * (1 + Real.tan (45 * Real.pi / 180))
  * (1 + Real.tan (46 * Real.pi / 180))
  * (1 + Real.tan (47 * Real.pi / 180))
  * (1 + Real.tan (48 * Real.pi / 180))
  * (1 + Real.tan (49 * Real.pi / 180))
  * (1 + Real.tan (50 * Real.pi / 180))
  * (1 + Real.tan (51 * Real.pi / 180))
  * (1 + Real.tan (52 * Real.pi / 180))
  * (1 + Real.tan (53 * Real.pi / 180))
  * (1 + Real.tan (54 * Real.pi / 180))
  * (1 + Real.tan (55 * Real.pi / 180))
  * (1 + Real.tan (56 * Real.pi / 180))
  * (1 + Real.tan (57 * Real.pi / 180))
  * (1 + Real.tan (58 * Real.pi / 180))
  * (1 + Real.tan (59 * Real.pi / 180))
  * (1 + Real.tan (60 * Real.pi / 180))

theorem tangent_product_power : tangent_product = 2^30 := by
  sorry

end tangent_product_power_l3_3446


namespace value_of_sum_l3_3512

theorem value_of_sum (a b c : ℚ) (h1 : 2 * a + 3 * b + c = 27) (h2 : 4 * a + 6 * b + 5 * c = 71) :
  a + b + c = 115 / 9 :=
sorry

end value_of_sum_l3_3512


namespace simplify_expression_l3_3380

theorem simplify_expression (x y : ℝ) : 3 * x + 2 * y + 4 * x + 5 * y + 7 = 7 * x + 7 * y + 7 := 
by sorry

end simplify_expression_l3_3380


namespace a_n_plus_1_is_geometric_general_term_formula_l3_3465

-- Define the sequence a_n.
def a : ℕ → ℤ
| 0       => 0  -- a_0 is not given explicitly, we start the sequence from 1.
| (n + 1) => if n = 0 then 1 else 2 * a n + 1

-- Prove that the sequence {a_n + 1} is a geometric sequence.
theorem a_n_plus_1_is_geometric : ∃ r : ℤ, ∀ n : ℕ, (a (n + 1) + 1) / (a n + 1) = r := by
  sorry

-- Find the general formula for a_n.
theorem general_term_formula : ∃ f : ℕ → ℤ, ∀ n : ℕ, a n = f n := by
  sorry

end a_n_plus_1_is_geometric_general_term_formula_l3_3465


namespace exist_positive_integers_for_perfect_squares_l3_3767

theorem exist_positive_integers_for_perfect_squares :
  ∃ (x y : ℕ), (0 < x ∧ 0 < y) ∧ (∃ a b c : ℕ, x + y = a^2 ∧ x^2 + y^2 = b^2 ∧ x^3 + y^3 = c^2) :=
by
  sorry

end exist_positive_integers_for_perfect_squares_l3_3767


namespace find_first_factor_of_lcm_l3_3452

theorem find_first_factor_of_lcm (hcf : ℕ) (A : ℕ) (X : ℕ) (B : ℕ) (lcm_val : ℕ) 
  (h_hcf : hcf = 59)
  (h_A : A = 944)
  (h_lcm_val : lcm_val = 59 * X * 16)
  (h_A_lcm : A = lcm_val) :
  X = 1 := 
by
  sorry

end find_first_factor_of_lcm_l3_3452


namespace books_in_shipment_l3_3548

theorem books_in_shipment (B : ℕ) (h : 3 / 4 * B = 180) : B = 240 :=
sorry

end books_in_shipment_l3_3548


namespace number_of_buses_l3_3315

theorem number_of_buses (vans people_per_van buses people_per_bus extra_people_in_buses : ℝ) 
  (h_vans : vans = 6.0) 
  (h_people_per_van : people_per_van = 6.0) 
  (h_people_per_bus : people_per_bus = 18.0) 
  (h_extra_people_in_buses : extra_people_in_buses = 108.0) 
  (h_eq : people_per_bus * buses = vans * people_per_van + extra_people_in_buses) : 
  buses = 8.0 :=
by
  sorry

end number_of_buses_l3_3315


namespace dodecahedron_interior_diagonals_l3_3400

-- Define the structure and properties of a dodecahedron
structure Dodecahedron :=
  (faces: ℕ := 12)
  (vertices: ℕ := 20)
  (vertices_per_face: ℕ := 5)
  (faces_per_vertex: ℕ := 3)

-- Total number of potential vertices to connect
def total_vertices (d: Dodecahedron) : ℕ := d.vertices - 1

-- Number of connected neighbors per vertex
def connected_neighbors (d: Dodecahedron) : ℕ := d.faces_per_vertex

-- Number of interior diagonals from one vertex
def interior_diagonals_per_vertex (d: Dodecahedron) : ℕ :=
  total_vertices d - connected_neighbors d

-- Total initial count of interior diagonals
def total_initial_interiors (d: Dodecahedron) : ℕ :=
  d.vertices * interior_diagonals_per_vertex d

-- Correct count of interior diagonals by accounting for overcounting
def correct_interior_diagonals (d: Dodecahedron) : ℕ :=
  total_initial_interiors d / 2

-- The theorem to prove
theorem dodecahedron_interior_diagonals (d: Dodecahedron) :
  correct_interior_diagonals d = 160 := by
  sorry

end dodecahedron_interior_diagonals_l3_3400


namespace lottery_win_probability_l3_3937

theorem lottery_win_probability :
  let MegaBall_prob := 1 / 30
  let WinnerBall_prob := 1 / Nat.choose 50 5
  let BonusBall_prob := 1 / 15
  let Total_prob := MegaBall_prob * WinnerBall_prob * BonusBall_prob
  Total_prob = 1 / 953658000 :=
by
  sorry

end lottery_win_probability_l3_3937


namespace ratio_of_gilled_to_spotted_l3_3369

theorem ratio_of_gilled_to_spotted (total_mushrooms gilled_mushrooms spotted_mushrooms : ℕ) 
  (h1 : total_mushrooms = 30) 
  (h2 : gilled_mushrooms = 3) 
  (h3 : spotted_mushrooms = total_mushrooms - gilled_mushrooms) :
  gilled_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 1 ∧ 
  spotted_mushrooms / gcd gilled_mushrooms spotted_mushrooms = 9 := 
by
  sorry

end ratio_of_gilled_to_spotted_l3_3369


namespace total_different_books_l3_3277

def tony_books : ℕ := 23
def dean_books : ℕ := 12
def breanna_books : ℕ := 17
def tony_dean_shared_books : ℕ := 3
def all_three_shared_book : ℕ := 1

theorem total_different_books :
  tony_books + dean_books + breanna_books - tony_dean_shared_books - 2 * all_three_shared_book = 47 := 
by
  sorry 

end total_different_books_l3_3277


namespace second_person_days_l3_3503

theorem second_person_days (P1 P2 : ℝ) (h1 : P1 = 1 / 24) (h2 : P1 + P2 = 1 / 8) : 1 / P2 = 12 :=
by
  sorry

end second_person_days_l3_3503


namespace ratio_of_side_lengths_l3_3619

theorem ratio_of_side_lengths
  (pentagon_perimeter square_perimeter : ℕ)
  (pentagon_sides square_sides : ℕ)
  (pentagon_perimeter_eq : pentagon_perimeter = 100)
  (square_perimeter_eq : square_perimeter = 100)
  (pentagon_sides_eq : pentagon_sides = 5)
  (square_sides_eq : square_sides = 4) :
  (pentagon_perimeter / pentagon_sides) / (square_perimeter / square_sides) = 4 / 5 :=
by
  sorry

end ratio_of_side_lengths_l3_3619


namespace lcm_105_360_eq_2520_l3_3114

theorem lcm_105_360_eq_2520 :
  Nat.lcm 105 360 = 2520 :=
by
  have h1 : 105 = 3 * 5 * 7 := by norm_num
  have h2 : 360 = 2^3 * 3^2 * 5 := by norm_num
  rw [h1, h2]
  sorry

end lcm_105_360_eq_2520_l3_3114


namespace douglas_won_in_Y_l3_3241

theorem douglas_won_in_Y (percent_total_vote : ℕ) (percent_vote_X : ℕ) (ratio_XY : ℕ) (P : ℕ) :
  percent_total_vote = 54 →
  percent_vote_X = 62 →
  ratio_XY = 2 →
  P = 38 :=
by
  sorry

end douglas_won_in_Y_l3_3241


namespace range_of_a_if_inequality_holds_l3_3997

noncomputable def satisfies_inequality_for_all_xy_pos (a : ℝ) :=
  ∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9

theorem range_of_a_if_inequality_holds :
  (∀ (x y : ℝ), (x > 0) → (y > 0) → (x + y) * (1 / x + a / y) ≥ 9) → (a ≥ 4) :=
by
  sorry

end range_of_a_if_inequality_holds_l3_3997


namespace prob_neither_A_nor_B_l3_3337

theorem prob_neither_A_nor_B
  (P_A : ℝ) (P_B : ℝ) (P_A_and_B : ℝ)
  (h1 : P_A = 0.25) (h2 : P_B = 0.30) (h3 : P_A_and_B = 0.15) : 
  1 - (P_A + P_B - P_A_and_B) = 0.60 :=
by
  sorry

end prob_neither_A_nor_B_l3_3337


namespace find_m_l3_3495

theorem find_m (n m : ℕ) (h1 : m = 13 * n + 8) (h2 : m = 15 * n) : m = 60 :=
  sorry

end find_m_l3_3495


namespace simplify_expression_l3_3829

theorem simplify_expression (x : ℝ) : (3 * x + 6 - 5 * x) / 3 = - (2 / 3) * x + 2 :=
by
  sorry

end simplify_expression_l3_3829


namespace great_circle_bisects_angle_l3_3917

noncomputable def north_pole : Point := sorry
noncomputable def equator_point (C : Point) : Prop := sorry
noncomputable def great_circle_through (P Q : Point) : Circle := sorry
noncomputable def equidistant_from_N (A B N : Point) : Prop := sorry
noncomputable def spherical_triangle (A B C : Point) : Triangle := sorry
noncomputable def bisects_angle (C N A B : Point) : Prop := sorry

theorem great_circle_bisects_angle
  (N A B C: Point)
  (hN: N = north_pole)
  (hA: equidistant_from_N A B N)
  (hC: equator_point C)
  (hTriangle: spherical_triangle A B C)
  : bisects_angle C N A B :=
sorry

end great_circle_bisects_angle_l3_3917


namespace unknown_card_value_l3_3451

theorem unknown_card_value (cards_total : ℕ)
  (p1_hand : ℕ) (p1_hand_extra : ℕ) (table_card1 : ℕ) (total_card_values : ℕ)
  (sum_removed_cards_sets : ℕ)
  (n : ℕ) :
  cards_total = 40 ∧ 
  p1_hand = 5 ∧ 
  p1_hand_extra = 3 ∧ 
  table_card1 = 9 ∧ 
  total_card_values = 220 ∧ 
  sum_removed_cards_sets = 15 * n → 
  ∃ x : ℕ, 1 ≤ x ∧ x ≤ 10 ∧ total_card_values = p1_hand + p1_hand_extra + table_card1 + x + sum_removed_cards_sets → 
  x = 8 := 
sorry

end unknown_card_value_l3_3451


namespace even_num_students_count_l3_3270

-- Define the number of students in each school
def num_students_A : Nat := 786
def num_students_B : Nat := 777
def num_students_C : Nat := 762
def num_students_D : Nat := 819
def num_students_E : Nat := 493

-- Define a predicate to check if a number is even
def is_even (n : Nat) : Prop := n % 2 = 0

-- The theorem to state the problem
theorem even_num_students_count :
  (is_even num_students_A ∧ is_even num_students_C) ∧ ¬(is_even num_students_B ∧ is_even num_students_D ∧ is_even num_students_E) →
  2 = 2 :=
by
  sorry

end even_num_students_count_l3_3270


namespace sheila_tue_thu_hours_l3_3133

def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def total_hours_mwf : ℕ := hours_mwf * days_mwf

def weekly_earnings : ℕ := 360
def hourly_rate : ℕ := 10
def earnings_mwf : ℕ := total_hours_mwf * hourly_rate

def earnings_tue_thu : ℕ := weekly_earnings - earnings_mwf
def hours_tue_thu : ℕ := earnings_tue_thu / hourly_rate

theorem sheila_tue_thu_hours : hours_tue_thu = 12 := by
  -- proof omitted
  sorry

end sheila_tue_thu_hours_l3_3133


namespace four_diff_digits_per_day_l3_3876

def valid_time_period (start_hour : ℕ) (end_hour : ℕ) : ℕ :=
  let total_minutes := (end_hour - start_hour + 1) * 60
  let valid_combinations :=
    match start_hour with
    | 0 => 0  -- start with appropriate calculation logic
    | 2 => 0  -- start with appropriate calculation logic
    | _ => 0  -- for general case, replace with correct logic
  total_minutes + valid_combinations  -- use proper aggregation

theorem four_diff_digits_per_day :
  valid_time_period 0 19 + valid_time_period 20 23 = 588 :=
by
  sorry

end four_diff_digits_per_day_l3_3876


namespace expected_sample_size_l3_3383

noncomputable def highSchoolTotalStudents (f s j : ℕ) : ℕ :=
  f + s + j

noncomputable def expectedSampleSize (total : ℕ) (p : ℝ) : ℝ :=
  total * p

theorem expected_sample_size :
  let f := 400
  let s := 320
  let j := 280
  let p := 0.2
  let total := highSchoolTotalStudents f s j
  expectedSampleSize total p = 200 :=
by
  sorry

end expected_sample_size_l3_3383


namespace no_polyhedron_with_area_ratio_ge_two_l3_3814

theorem no_polyhedron_with_area_ratio_ge_two (n : ℕ) (areas : Fin n → ℝ)
  (h : ∀ (i j : Fin n), i < j → (areas j) / (areas i) ≥ 2) : False := by
  sorry

end no_polyhedron_with_area_ratio_ge_two_l3_3814


namespace couple_slices_each_l3_3911

noncomputable def slices_for_couple (total_slices children_slices people_in_couple : ℕ) : ℕ :=
  (total_slices - children_slices) / people_in_couple

theorem couple_slices_each (people_in_couple children slices_per_pizza num_pizzas : ℕ) (H1 : people_in_couple = 2) (H2 : children = 6) (H3 : slices_per_pizza = 4) (H4 : num_pizzas = 3) :
  slices_for_couple (num_pizzas * slices_per_pizza) (children * 1) people_in_couple = 3 := 
  by
  rw [H1, H2, H3, H4]
  show slices_for_couple (3 * 4) (6 * 1) 2 = 3
  rfl

end couple_slices_each_l3_3911


namespace power_of_two_representation_l3_3733

/-- Prove that any number 2^n, where n = 3,4,5,..., can be represented 
as 7x^2 + y^2 where x and y are odd numbers. -/
theorem power_of_two_representation (n : ℕ) (hn : n ≥ 3) : 
  ∃ x y : ℤ, (2*x ≠ 0 ∧ 2*y ≠ 0) ∧ 2^n = 7 * x^2 + y^2 :=
by
  sorry

end power_of_two_representation_l3_3733


namespace x_y_iff_pos_l3_3792

theorem x_y_iff_pos (x y : ℝ) : x + y > |x - y| ↔ x > 0 ∧ y > 0 := by
  sorry

end x_y_iff_pos_l3_3792


namespace pizza_slices_left_l3_3124

-- Lean definitions for conditions
def total_slices : ℕ := 24
def slices_eaten_dinner : ℕ := total_slices / 3
def slices_after_dinner : ℕ := total_slices - slices_eaten_dinner

def slices_eaten_yves : ℕ := slices_after_dinner / 5
def slices_after_yves : ℕ := slices_after_dinner - slices_eaten_yves

def slices_eaten_oldest_siblings : ℕ := 3 * 3
def slices_after_oldest_siblings : ℕ := slices_after_yves - slices_eaten_oldest_siblings

def num_remaining_siblings : ℕ := 7 - 3
def slices_eaten_remaining_siblings : ℕ := num_remaining_siblings * 2
def slices_final : ℕ := if slices_after_oldest_siblings < slices_eaten_remaining_siblings then 0 else slices_after_oldest_siblings - slices_eaten_remaining_siblings

-- Proposition to prove
theorem pizza_slices_left : slices_final = 0 := by sorry

end pizza_slices_left_l3_3124


namespace chi_square_confidence_l3_3139

theorem chi_square_confidence (chi_square : ℝ) (df : ℕ) (critical_value : ℝ) :
  chi_square = 6.825 ∧ df = 1 ∧ critical_value = 6.635 → confidence_level = 0.99 := 
by
  sorry

end chi_square_confidence_l3_3139


namespace max_value_3x_plus_4y_l3_3120

theorem max_value_3x_plus_4y (x y : ℝ) : x^2 + y^2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73 :=
sorry

end max_value_3x_plus_4y_l3_3120


namespace two_dice_sum_greater_than_four_l3_3596
open Classical

def probability_sum_greater_than_four : ℚ := by sorry

theorem two_dice_sum_greater_than_four :
  probability_sum_greater_than_four = 5 / 6 :=
sorry

end two_dice_sum_greater_than_four_l3_3596


namespace Luca_milk_water_needed_l3_3711

def LucaMilk (flour : ℕ) : ℕ := (flour / 250) * 50
def LucaWater (flour : ℕ) : ℕ := (flour / 250) * 30

theorem Luca_milk_water_needed (flour : ℕ) (h : flour = 1250) : LucaMilk flour = 250 ∧ LucaWater flour = 150 := by
  rw [h]
  sorry

end Luca_milk_water_needed_l3_3711


namespace find_number_l3_3729

theorem find_number :
  ∃ x : ℕ, (8 * x + 5400) / 12 = 530 ∧ x = 120 :=
by
  sorry

end find_number_l3_3729


namespace choose_three_consecutive_circles_l3_3429

theorem choose_three_consecutive_circles (n : ℕ) (hn : n = 33) : 
  ∃ (ways : ℕ), ways = 57 :=
by
  sorry

end choose_three_consecutive_circles_l3_3429


namespace exists_d_for_m_divides_f_of_f_n_l3_3237

noncomputable def f : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => 23 * f (n + 1) + f n

theorem exists_d_for_m_divides_f_of_f_n (m : ℕ) : 
  ∃ (d : ℕ), ∀ (n : ℕ), m ∣ f (f n) ↔ d ∣ n := 
sorry

end exists_d_for_m_divides_f_of_f_n_l3_3237


namespace find_C_l3_3331

theorem find_C (A B C : ℕ)
  (hA : A = 348)
  (hB : B = A + 173)
  (hC : C = B + 299) :
  C = 820 :=
sorry

end find_C_l3_3331


namespace find_numbers_l3_3845

theorem find_numbers (S P : ℝ) (x y : ℝ) : 
  (x + y = S ∧ xy = P) ↔ 
  (x = (S + Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S - Real.sqrt (S^2 - 4 * P)) / 2) ∨ 
  (x = (S - Real.sqrt (S^2 - 4 * P)) / 2 ∧ y = (S + Real.sqrt (S^2 - 4 * P)) / 2) :=
by sorry

end find_numbers_l3_3845


namespace rita_bought_4_jackets_l3_3584

/-
Given:
  - Rita bought 5 short dresses costing $20 each.
  - Rita bought 3 pairs of pants costing $12 each.
  - The jackets cost $30 each.
  - She spent an additional $5 on transportation.
  - Rita had $400 initially.
  - Rita now has $139.

Prove that the number of jackets Rita bought is 4.
-/

theorem rita_bought_4_jackets :
  let dresses_cost := 5 * 20
  let pants_cost := 3 * 12
  let transportation_cost := 5
  let initial_amount := 400
  let remaining_amount := 139
  let jackets_cost_per_unit := 30
  let total_spent := initial_amount - remaining_amount
  let total_clothes_transportation_cost := dresses_cost + pants_cost + transportation_cost
  let jackets_cost := total_spent - total_clothes_transportation_cost
  let number_of_jackets := jackets_cost / jackets_cost_per_unit
  number_of_jackets = 4 :=
by
  sorry

end rita_bought_4_jackets_l3_3584


namespace sector_angle_l3_3185

theorem sector_angle (r α : ℝ) (h₁ : 2 * r + α * r = 4) (h₂ : (1 / 2) * α * r^2 = 1) : α = 2 :=
sorry

end sector_angle_l3_3185


namespace max_cos_product_l3_3555

open Real

theorem max_cos_product (α β γ : ℝ) (hα : 0 < α ∧ α < π / 2)
                                       (hβ : 0 < β ∧ β < π / 2)
                                       (hγ : 0 < γ ∧ γ < π / 2)
                                       (h : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) : 
  cos α * cos β * cos γ ≤ 2 * Real.sqrt 6 / 9 := 
by sorry

end max_cos_product_l3_3555


namespace michael_max_correct_answers_l3_3649

theorem michael_max_correct_answers (c w b : ℕ) 
  (h1 : c + w + b = 30) 
  (h2 : 4 * c - 3 * w = 72) : 
  c ≤ 21 := 
sorry

end michael_max_correct_answers_l3_3649


namespace thalassa_population_2050_l3_3722

def population_in_2000 : ℕ := 250

def population_doubling_interval : ℕ := 20

def population_linear_increase_interval : ℕ := 10

def linear_increase_amount : ℕ := 500

noncomputable def population_in_2050 : ℕ :=
  let double1 := population_in_2000 * 2
  let double2 := double1 * 2
  double2 + linear_increase_amount

theorem thalassa_population_2050 : population_in_2050 = 1500 := by
  sorry

end thalassa_population_2050_l3_3722


namespace largest_number_is_870_l3_3375

-- Define the set of digits {8, 7, 0}
def digits : Set ℕ := {8, 7, 0}

-- Define the largest number that can be made by arranging these digits
def largest_number (s : Set ℕ) : ℕ := 870

-- Statement to prove
theorem largest_number_is_870 : largest_number digits = 870 :=
by
  -- Proof is omitted
  sorry

end largest_number_is_870_l3_3375


namespace complement_of_M_in_U_l3_3132

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5}
noncomputable def M : Set ℕ := {0, 1}

theorem complement_of_M_in_U : (U \ M) = {2, 3, 4, 5} :=
by
  -- The proof is omitted here.
  sorry

end complement_of_M_in_U_l3_3132


namespace second_workshop_production_l3_3356

theorem second_workshop_production (a b c : ℕ) (h₁ : a + b + c = 3600) (h₂ : a + c = 2 * b) : b * 3 = 3600 := 
by 
  sorry

end second_workshop_production_l3_3356


namespace probability_of_all_heads_or_tails_l3_3723

theorem probability_of_all_heads_or_tails :
  let possible_outcomes := 256
  let favorable_outcomes := 2
  favorable_outcomes / possible_outcomes = 1 / 128 := by
  sorry

end probability_of_all_heads_or_tails_l3_3723


namespace complement_union_l3_3192

variable (x : ℝ)

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def P : Set ℝ := {x | x ≥ 2}

theorem complement_union (x : ℝ) : x ∈ U → (¬ (x ∈ M ∨ x ∈ P)) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complement_union_l3_3192


namespace quadratic_solution_transformation_l3_3015

theorem quadratic_solution_transformation
  (m h k : ℝ)
  (h_nonzero : m ≠ 0)
  (x1 x2 : ℝ)
  (h_sol1 : m * (x1 - h)^2 - k = 0)
  (h_sol2 : m * (x2 - h)^2 - k = 0)
  (h_x1 : x1 = 2)
  (h_x2 : x2 = 5) :
  (∃ x1' x2', x1' = 1 ∧ x2' = 4 ∧ m * (x1' - h + 1)^2 = k ∧ m * (x2' - h + 1)^2 = k) :=
by 
  -- Proof here
  sorry

end quadratic_solution_transformation_l3_3015


namespace computer_price_decrease_l3_3900

theorem computer_price_decrease 
  (initial_price : ℕ) 
  (decrease_factor : ℚ)
  (years : ℕ) 
  (final_price : ℕ) 
  (h1 : initial_price = 8100)
  (h2 : decrease_factor = 1/3)
  (h3 : years = 6)
  (h4 : final_price = 2400) : 
  initial_price * (1 - decrease_factor) ^ (years / 2) = final_price :=
by
  sorry

end computer_price_decrease_l3_3900


namespace least_multiple_greater_than_500_l3_3878

theorem least_multiple_greater_than_500 : ∃ n : ℕ, n > 0 ∧ 35 * n > 500 ∧ 35 * n = 525 :=
by
  sorry

end least_multiple_greater_than_500_l3_3878


namespace distinct_students_count_l3_3908

-- Definition of the initial parameters
def num_gauss : Nat := 12
def num_euler : Nat := 10
def num_fibonnaci : Nat := 7
def overlap : Nat := 1

-- The main theorem to prove
theorem distinct_students_count : num_gauss + num_euler + num_fibonnaci - overlap = 28 := by
  sorry

end distinct_students_count_l3_3908


namespace distance_between_QY_l3_3687

theorem distance_between_QY 
  (m_rate : ℕ) (j_rate : ℕ) (j_distance : ℕ) (headstart : ℕ) 
  (t : ℕ) 
  (h1 : m_rate = 3) 
  (h2 : j_rate = 4) 
  (h3 : j_distance = 24) 
  (h4 : headstart = 1) 
  (h5 : j_distance = j_rate * (t - headstart)) 
  (h6 : t = 7) 
  (distance_m : ℕ := m_rate * t) 
  (distance_j : ℕ := j_distance) :
  distance_j + distance_m = 45 :=
by 
  sorry

end distance_between_QY_l3_3687


namespace value_of_d_l3_3786

theorem value_of_d (y : ℝ) (d : ℝ) (h1 : y > 0) (h2 : (4 * y) / 20 + (3 * y) / d = 0.5 * y) : d = 10 :=
by
  sorry

end value_of_d_l3_3786


namespace find_m_for_parallel_lines_l3_3292

theorem find_m_for_parallel_lines (m : ℝ) :
  (∀ x y : ℝ, 6 * x + m * y - 1 = 0 ↔ 2 * x - y + 1 = 0) → m = -3 :=
by
  sorry

end find_m_for_parallel_lines_l3_3292


namespace snow_volume_l3_3614

theorem snow_volume
  (length : ℝ) (width : ℝ) (depth : ℝ)
  (h_length : length = 15)
  (h_width : width = 3)
  (h_depth : depth = 0.6) :
  length * width * depth = 27 := 
by
  -- placeholder for proof
  sorry

end snow_volume_l3_3614


namespace calculate_polygon_sides_l3_3201

-- Let n be the number of sides of the regular polygon with each exterior angle of 18 degrees
def regular_polygon_sides (n : ℕ) (exterior_angle : ℝ) : Prop :=
  exterior_angle = 18 ∧ n * exterior_angle = 360

theorem calculate_polygon_sides (n : ℕ) (exterior_angle : ℝ) :
  regular_polygon_sides n exterior_angle → n = 20 :=
by
  intro h
  sorry

end calculate_polygon_sides_l3_3201


namespace inappropriate_survey_method_l3_3631

def survey_method_appropriate (method : String) : Bool :=
  method = "sampling" -- only sampling is considered appropriate in this toy model

def survey_approps : Bool :=
  let A := survey_method_appropriate "sampling"
  let B := survey_method_appropriate "sampling"
  let C := ¬ survey_method_appropriate "census"
  let D := survey_method_appropriate "census"
  C

theorem inappropriate_survey_method :
  survey_approps = true :=
by
  sorry

end inappropriate_survey_method_l3_3631


namespace rhombus_difference_l3_3309

theorem rhombus_difference (n : ℕ) (h : n > 3)
    (m : ℕ := 3 * (n - 1) * n / 2)
    (d : ℕ := 3 * (n - 3) * (n - 2) / 2) :
    m - d = 6 * n - 9 := by {
  -- Proof omitted
  sorry
}

end rhombus_difference_l3_3309


namespace diana_owes_l3_3720

-- Define the conditions
def initial_charge : ℝ := 60
def annual_interest_rate : ℝ := 0.06
def time_in_years : ℝ := 1

-- Define the simple interest calculation
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

-- Define the total amount owed calculation
def total_amount_owed (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

-- State the theorem: Diana will owe $63.60 after one year
theorem diana_owes : total_amount_owed initial_charge (simple_interest initial_charge annual_interest_rate time_in_years) = 63.60 :=
by sorry

end diana_owes_l3_3720


namespace lemon_heads_per_package_l3_3913

theorem lemon_heads_per_package (total_lemon_heads boxes : ℕ)
  (H : total_lemon_heads = 54)
  (B : boxes = 9)
  (no_leftover : total_lemon_heads % boxes = 0) :
  total_lemon_heads / boxes = 6 :=
sorry

end lemon_heads_per_package_l3_3913


namespace hyperbola_asymptotes_l3_3682

theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), (x^2 - 4 * y^2 = 1) → (x = 2 * y ∨ x = -2 * y) :=
by
  intros x y h
  sorry

end hyperbola_asymptotes_l3_3682


namespace number_is_seven_l3_3747

theorem number_is_seven (x : ℝ) (h : x^2 + 120 = (x - 20)^2) : x = 7 := 
by
  sorry

end number_is_seven_l3_3747


namespace birds_reduction_on_third_day_l3_3003

theorem birds_reduction_on_third_day
  {a b c : ℕ} 
  (h1 : a = 300)
  (h2 : b = 2 * a)
  (h3 : c = 1300)
  : (b - (c - (a + b))) = 200 :=
by sorry

end birds_reduction_on_third_day_l3_3003


namespace maximum_ab_is_40_l3_3480

noncomputable def maximum_ab (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : ℝ :=
  max (a * b) 40

theorem maximum_ab_is_40 {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : 5 * a + 8 * b = 80) : maximum_ab a b h₀ h₁ h₂ = 40 := 
by 
  sorry

end maximum_ab_is_40_l3_3480


namespace outer_boundary_diameter_l3_3928

def width_jogging_path : ℝ := 4
def width_garden_ring : ℝ := 10
def diameter_pond : ℝ := 12

theorem outer_boundary_diameter : 2 * (diameter_pond / 2 + width_garden_ring + width_jogging_path) = 40 := by
  sorry

end outer_boundary_diameter_l3_3928


namespace composite_sum_of_four_integers_l3_3494

theorem composite_sum_of_four_integers 
  (a b c d : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h_pos_d : 0 < d) 
  (h_eq : a^2 + b^2 + a * b = c^2 + d^2 + c * d) : 
  ∃ n m : ℕ, 1 < a + b + c + d ∧ a + b + c + d = n * m ∧ 1 < n ∧ 1 < m := 
sorry

end composite_sum_of_four_integers_l3_3494


namespace probability_of_two_pairs_of_same_value_is_correct_l3_3690

def total_possible_outcomes := 6^6
def number_of_ways_to_form_pairs := 15
def choose_first_pair := 6
def choose_second_pair := 15
def choose_third_pair := 6
def choose_fourth_die := 4
def choose_fifth_die := 3

def successful_outcomes := number_of_ways_to_form_pairs *
                           choose_first_pair *
                           choose_second_pair *
                           choose_third_pair *
                           choose_fourth_die *
                           choose_fifth_die

def probability_of_two_pairs_of_same_value := (successful_outcomes : ℚ) / total_possible_outcomes

theorem probability_of_two_pairs_of_same_value_is_correct :
  probability_of_two_pairs_of_same_value = 25 / 72 :=
by
  -- proof omitted
  sorry

end probability_of_two_pairs_of_same_value_is_correct_l3_3690


namespace circle_bisection_relation_l3_3282

theorem circle_bisection_relation (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = b^2 + 1 → (x + 1)^2 + (y + 1)^2 = 4) ↔ 
  a^2 + 2 * a + 2 * b + 5 = 0 :=
by sorry

end circle_bisection_relation_l3_3282


namespace polynomial_divisibility_l3_3856

open Polynomial

variables {R : Type*} [CommRing R]
variables {f g h k : R[X]}

theorem polynomial_divisibility (h1 : (X^2 + 1) * h + (X - 1) * f + (X - 2) * g = 0)
    (h2 : (X^2 + 1) * k + (X + 1) * f + (X + 2) * g = 0) :
    (X^2 + 1) ∣ (f * g) :=
sorry

end polynomial_divisibility_l3_3856


namespace speed_of_man_upstream_l3_3196

def speed_of_man_in_still_water : ℝ := 32
def speed_of_man_downstream : ℝ := 39

theorem speed_of_man_upstream (V_m V_s : ℝ) :
  V_m = speed_of_man_in_still_water →
  V_m + V_s = speed_of_man_downstream →
  V_m - V_s = 25 :=
sorry

end speed_of_man_upstream_l3_3196


namespace reciprocal_of_neg3_l3_3525

theorem reciprocal_of_neg3 : 1 / (-3 : ℝ) = - (1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l3_3525


namespace mark_initial_kept_percentage_l3_3025

-- Defining the conditions
def initial_friends : Nat := 100
def remaining_friends : Nat := 70
def percentage_contacted (P : ℝ) := 100 - P
def percentage_responded : ℝ := 0.5

-- Theorem statement: Mark initially kept 40% of his friends
theorem mark_initial_kept_percentage (P : ℝ) : 
  (P / 100 * initial_friends) + (percentage_contacted P / 100 * initial_friends * percentage_responded) = remaining_friends → 
  P = 40 := by
  sorry

end mark_initial_kept_percentage_l3_3025


namespace complement_of_A_in_U_l3_3027

variable {U : Set ℤ}
variable {A : Set ℤ}

theorem complement_of_A_in_U (hU : U = {-1, 0, 1}) (hA : A = {0, 1}) : U \ A = {-1} := by
  sorry

end complement_of_A_in_U_l3_3027


namespace find_d_l3_3053

def f (x : ℝ) (c : ℝ) : ℝ := 5 * x + c
def g (x : ℝ) (c : ℝ) : ℝ := c * x + 3

theorem find_d (c d : ℝ) (h : ∀ x : ℝ, f (g x c) c = 15 * x + d) : d = 18 :=
by
  sorry

end find_d_l3_3053


namespace monotonicity_of_f_range_of_a_l3_3072

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x

theorem monotonicity_of_f (a : ℝ) : 
  (a ≤ 0 → ∀ x y : ℝ, x < y → f x a < f y a) ∧ 
  (a > 0 → ∀ x y : ℝ, 
    (x < y ∧ y ≤ Real.log a → f x a > f y a) ∧ 
    (x > Real.log a → f x a < f y a)) :=
by
  sorry

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 0) ↔ 0 ≤ a ∧ a ≤ Real.exp 1 :=
by
  sorry

end monotonicity_of_f_range_of_a_l3_3072


namespace unique_zero_f_x1_minus_2x2_l3_3807

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := a * (Real.exp x - x - 1) - Real.log (x + 1) + x

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a * Real.exp x + x

-- Condition a ≥ 0
variable (a : ℝ) (a_nonneg : 0 ≤ a)

-- Define the first part of the problem
theorem unique_zero_f : ∃! x, f a x = 0 :=
  sorry

-- Variables for the second part of the problem
variable (x₁ x₂ : ℝ)
variable (cond : f a x₁ = g a x₁ - g a x₂)

-- Define the second part of the problem
theorem x1_minus_2x2 : x₁ - 2 * x₂ ≥ 1 - 2 * Real.log 2 :=
  sorry

end unique_zero_f_x1_minus_2x2_l3_3807


namespace probability_average_is_five_l3_3556

-- Definitions and conditions
def numbers : List ℕ := [1, 3, 4, 6, 7, 9]

def average_is_five (a b : ℕ) : Prop := (a + b) / 2 = 5

-- Desired statement
theorem probability_average_is_five : 
  ∃ p : ℚ, p = 1 / 5 ∧ (∃ a b : ℕ, a ∈ numbers ∧ b ∈ numbers ∧ average_is_five a b) := 
sorry

end probability_average_is_five_l3_3556


namespace total_students_in_Lansing_l3_3936

def n_schools : Nat := 25
def students_per_school : Nat := 247
def total_students : Nat := n_schools * students_per_school

theorem total_students_in_Lansing :
  total_students = 6175 :=
  by
    -- we can either compute manually or just put sorry for automated assistance
    sorry

end total_students_in_Lansing_l3_3936


namespace solution_set_of_inequality_l3_3721

theorem solution_set_of_inequality :
  {x : ℝ | (x-2)*(3-x) > 0} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solution_set_of_inequality_l3_3721


namespace nap_time_left_l3_3643

def train_ride_duration : ℕ := 9
def reading_time : ℕ := 2
def eating_time : ℕ := 1
def watching_movie_time : ℕ := 3

theorem nap_time_left :
  train_ride_duration - (reading_time + eating_time + watching_movie_time) = 3 :=
by
  -- Insert proof here
  sorry

end nap_time_left_l3_3643


namespace unique_nonneg_sequence_l3_3875

theorem unique_nonneg_sequence (a : List ℝ) (h_sum : 0 < a.sum) :
  ∃ b : List ℝ, (∀ x ∈ b, 0 ≤ x) ∧ 
                (∃ f : List ℝ → List ℝ, (f a = b) ∧ (∀ x y z, f (x :: y :: z :: tl) = (x + y) :: (-y) :: (z + y) :: tl)) :=
sorry

end unique_nonneg_sequence_l3_3875


namespace probability_of_selection_l3_3724

noncomputable def probability_selected (total_students : ℕ) (excluded_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / (total_students - excluded_students)

theorem probability_of_selection :
  probability_selected 2008 8 50 = 25 / 1004 :=
by
  sorry

end probability_of_selection_l3_3724


namespace batsman_sixes_l3_3474

theorem batsman_sixes 
(scorer_runs : ℕ)
(boundaries : ℕ)
(run_contrib : ℕ → ℚ)
(score_by_boundary : ℕ)
(score : ℕ)
(h1 : scorer_runs = 125)
(h2 : boundaries = 5)
(h3 : ∀ (x : ℕ), run_contrib x = (0.60 * scorer_runs : ℚ))
(h4 : score_by_boundary = boundaries * 4)
(h5 : score = scorer_runs - score_by_boundary) : 
∃ (x : ℕ), x = 5 ∧ (scorer_runs = score + (x * 6)) :=
by
  sorry

end batsman_sixes_l3_3474


namespace distinct_integers_real_roots_l3_3402

theorem distinct_integers_real_roots (a b c : ℤ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : a > b) (h5 : b > c) :
    (∃ x : ℝ, x^2 + 2 * a * x + 3 * (b + c) = 0) :=
sorry

end distinct_integers_real_roots_l3_3402


namespace problem_solution_l3_3435

noncomputable def f (x a : ℝ) : ℝ :=
  2 * (Real.cos x)^2 - 2 * a * Real.cos x - (2 * a + 1)

noncomputable def g (a : ℝ) : ℝ :=
  if a < -2 then 1
  else if a < 2 then -a^2 / 2 - 2 * a - 1
  else 1 - 4 * a

theorem problem_solution :
  g a = 1 ∨ g a = (-a^2 / 2 - 2 * a - 1) ∨ g a = 1 - 4 * a →
  (∀ a, g a = 1 / 2 → a = -1) ∧ (f x (-1) ≤ 5) :=
sorry

end problem_solution_l3_3435


namespace intersection_M_N_l3_3083

open Set Int

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by
  sorry

end intersection_M_N_l3_3083


namespace rectangle_dimensions_l3_3471

variable (w l : ℝ)
variable (h1 : l = w + 15)
variable (h2 : 2 * w + 2 * l = 150)

theorem rectangle_dimensions :
  w = 30 ∧ l = 45 :=
by
  sorry

end rectangle_dimensions_l3_3471


namespace sun_city_population_correct_l3_3613

noncomputable def willowdale_population : Nat := 2000
noncomputable def roseville_population : Nat := 3 * willowdale_population - 500
noncomputable def sun_city_population : Nat := 2 * roseville_population + 1000

theorem sun_city_population_correct : sun_city_population = 12000 := by
  sorry

end sun_city_population_correct_l3_3613


namespace batsman_average_increase_l3_3151

theorem batsman_average_increase
  (A : ℤ)
  (h1 : (16 * A + 85) / 17 = 37) :
  37 - A = 3 :=
by
  sorry

end batsman_average_increase_l3_3151


namespace jellybeans_in_jar_now_l3_3650

def initial_jellybeans : ℕ := 90
def samantha_takes : ℕ := 24
def shelby_takes : ℕ := 12
def scarlett_takes : ℕ := 2 * shelby_takes
def scarlett_returns : ℕ := scarlett_takes / 2
def shannon_refills : ℕ := (samantha_takes + shelby_takes) / 2

theorem jellybeans_in_jar_now : 
  initial_jellybeans 
  - samantha_takes 
  - shelby_takes 
  + scarlett_returns
  + shannon_refills 
  = 84 := by
  sorry

end jellybeans_in_jar_now_l3_3650


namespace sum_of_consecutive_integers_product_2730_eq_42_l3_3138

theorem sum_of_consecutive_integers_product_2730_eq_42 :
  ∃ x : ℤ, x * (x + 1) * (x + 2) = 2730 ∧ x + (x + 1) + (x + 2) = 42 :=
by
  sorry

end sum_of_consecutive_integers_product_2730_eq_42_l3_3138


namespace women_count_l3_3660

/-- 
Initially, the men and women in a room were in the ratio of 4:5.
Then, 2 men entered the room and 3 women left the room.
The number of women then doubled.
There are now 14 men in the room.
Prove that the number of women currently in the room is 24.
-/
theorem women_count (x : ℕ) (h1 : 4 * x + 2 = 14) (h2 : 2 * (5 * x - 3) = n) : 
  n = 24 :=
by
  sorry

end women_count_l3_3660


namespace remainder_when_2519_divided_by_3_l3_3846

theorem remainder_when_2519_divided_by_3 :
  2519 % 3 = 2 :=
by
  sorry

end remainder_when_2519_divided_by_3_l3_3846


namespace min_value_ab_l3_3824

theorem min_value_ab (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : (a / 2) + b = 1) :
  (1 / a) + (1 / b) = (3 / 2) + Real.sqrt 2 :=
by sorry

end min_value_ab_l3_3824


namespace least_positive_n_l3_3488

theorem least_positive_n : ∃ n : ℕ, (1 / (n : ℝ) - 1 / (n + 1 : ℝ) < 1 / 12) ∧ (∀ m : ℕ, (1 / (m : ℝ) - 1 / (m + 1 : ℝ) < 1 / 12) → n ≤ m) :=
by {
  sorry
}

end least_positive_n_l3_3488


namespace track_width_l3_3391

theorem track_width (r1 r2 : ℝ) (h : 2 * π * r1 - 2 * π * r2 = 10 * π) : r1 - r2 = 5 :=
sorry

end track_width_l3_3391


namespace sn_values_l3_3146

noncomputable def s (x1 x2 x3 : ℂ) (n : ℕ) : ℂ :=
  x1^n + x2^n + x3^n

theorem sn_values (p q x1 x2 x3 : ℂ) (h_root1 : x1^3 + p * x1 + q = 0)
                    (h_root2 : x2^3 + p * x2 + q = 0)
                    (h_root3 : x3^3 + p * x3 + q = 0) :
  s x1 x2 x3 2 = -3 * q ∧
  s x1 x2 x3 3 = 3 * q^2 ∧
  s x1 x2 x3 4 = 2 * p^2 ∧
  s x1 x2 x3 5 = 5 * p * q ∧
  s x1 x2 x3 6 = -2 * p^3 + 3 * q^2 ∧
  s x1 x2 x3 7 = -7 * p^2 * q ∧
  s x1 x2 x3 8 = 2 * p^4 - 8 * p * q^2 ∧
  s x1 x2 x3 9 = 9 * p^3 * q - 3 * q^3 ∧
  s x1 x2 x3 10 = -2 * p^5 + 15 * p^2 * q^2 :=
by {
  sorry
}

end sn_values_l3_3146


namespace number_of_pencils_l3_3420

theorem number_of_pencils (P L : ℕ) (h1 : (P : ℚ) / L = 5 / 6) (h2 : L = P + 6) : L = 36 :=
sorry

end number_of_pencils_l3_3420


namespace exists_positive_integers_for_equation_l3_3226

theorem exists_positive_integers_for_equation :
  ∃ (a b c : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a^4 = b^3 + c^2 :=
by
  sorry

end exists_positive_integers_for_equation_l3_3226


namespace problem_solution_l3_3885

noncomputable def proof_problem : Prop :=
∀ x y : ℝ, y = (x + 1)^2 ∧ (x * y^2 + y = 1) → false

theorem problem_solution : proof_problem :=
by
  sorry

end problem_solution_l3_3885


namespace bus_travel_fraction_l3_3086

theorem bus_travel_fraction :
  ∃ D : ℝ, D = 30.000000000000007 ∧
            (1 / 3) * D + 2 + (18 / 30) * D = D ∧
            (18 / 30) = (3 / 5) :=
by
  sorry

end bus_travel_fraction_l3_3086


namespace rectangle_length_l3_3350

theorem rectangle_length {width length : ℝ} (h1 : (3 : ℝ) * 3 = 9) (h2 : width = 3) (h3 : width * length = 9) : 
  length = 3 :=
by
  sorry

end rectangle_length_l3_3350


namespace flat_rate_first_night_l3_3045

theorem flat_rate_first_night
  (f n : ℚ)
  (h1 : f + 3 * n = 210)
  (h2 : f + 6 * n = 350)
  : f = 70 :=
by
  sorry

end flat_rate_first_night_l3_3045


namespace equation_of_the_line_l3_3645

noncomputable def line_equation (t : ℝ) : (ℝ × ℝ) := (3 * t + 6, 5 * t - 7)

theorem equation_of_the_line : ∃ m b : ℝ, (∀ t : ℝ, ∃ (x y : ℝ), line_equation t = (x, y) ∧ y = m * x + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  sorry

end equation_of_the_line_l3_3645


namespace probability_both_red_l3_3589

-- Definitions for the problem conditions
def total_balls := 16
def red_balls := 7
def blue_balls := 5
def green_balls := 4
def first_red_prob := (red_balls : ℚ) / total_balls
def second_red_given_first_red_prob := (red_balls - 1 : ℚ) / (total_balls - 1)

-- The statement to be proved
theorem probability_both_red : (first_red_prob * second_red_given_first_red_prob) = (7 : ℚ) / 40 :=
by 
  -- Proof goes here
  sorry

end probability_both_red_l3_3589


namespace average_goods_per_hour_l3_3955

-- Define the conditions
def morning_goods : ℕ := 64
def morning_hours : ℕ := 4
def afternoon_rate : ℕ := 23
def afternoon_hours : ℕ := 3

-- Define the target statement to be proven
theorem average_goods_per_hour : (morning_goods + afternoon_rate * afternoon_hours) / (morning_hours + afternoon_hours) = 19 := by
  -- Add proof steps here
  sorry

end average_goods_per_hour_l3_3955


namespace sin_identity_l3_3869

theorem sin_identity (α : ℝ) (h : Real.sin (2 * Real.pi / 3 - α) + Real.sin α = 4 * Real.sqrt 3 / 5) :
  Real.sin (α + 7 * Real.pi / 6) = -4 / 5 := sorry

end sin_identity_l3_3869


namespace choose_officers_from_six_l3_3303

/--
In how many ways can a President, a Vice-President, and a Secretary be chosen from a group of 6 people 
(assuming that all positions must be held by different individuals)?
-/
theorem choose_officers_from_six : (6 * 5 * 4 = 120) := 
by sorry

end choose_officers_from_six_l3_3303


namespace positive_difference_solutions_of_abs_eq_l3_3610

theorem positive_difference_solutions_of_abs_eq (x1 x2 : ℝ) (h1 : 2 * x1 - 3 = 15) (h2 : 2 * x2 - 3 = -15) : |x1 - x2| = 15 := by
  sorry

end positive_difference_solutions_of_abs_eq_l3_3610


namespace odd_expression_l3_3253

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1

theorem odd_expression (p q : ℕ) (hp : is_odd p) (hq : is_odd q) : is_odd (2 * p * p - q) :=
by
  sorry

end odd_expression_l3_3253


namespace sum_of_solutions_l3_3972

theorem sum_of_solutions (a b c : ℚ) (h : a ≠ 0) (eq : 2 * x^2 - 7 * x - 9 = 0) : 
  (-b / a) = (7 / 2) := 
sorry

end sum_of_solutions_l3_3972


namespace calculate_fraction_l3_3654

-- Define the fractions we are working with
def fraction1 : ℚ := 3 / 4
def fraction2 : ℚ := 15 / 5
def one_half : ℚ := 1 / 2

-- Define the main calculation
def main_fraction (f1 f2 one_half : ℚ) : ℚ := f1 * f2 - one_half

-- State the theorem
theorem calculate_fraction : main_fraction fraction1 fraction2 one_half = (7 / 4) := by
  sorry

end calculate_fraction_l3_3654


namespace average_sleep_time_l3_3775

def sleep_times : List ℕ := [10, 9, 10, 8, 8]

theorem average_sleep_time : (sleep_times.sum / sleep_times.length) = 9 := by
  sorry

end average_sleep_time_l3_3775


namespace intersection_of_A_and_B_is_2_l3_3413

-- Define the sets A and B based on the given conditions
def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

-- State the theorem that needs to be proved
theorem intersection_of_A_and_B_is_2 : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_is_2_l3_3413


namespace right_triangle_legs_l3_3672

theorem right_triangle_legs (m r x y : ℝ) 
  (h1 : m^2 = x^2 + y^2) 
  (h2 : r = (x + y - m) / 2) 
  (h3 : r ≤ m * (Real.sqrt 2 - 1) / 2) : 
  (x = (2 * r + m + Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) ∧ 
  (y = (2 * r + m - Real.sqrt (m^2 - 4 * r^2 - 4 * r * m)) / 2) :=
by 
  sorry

end right_triangle_legs_l3_3672


namespace bob_initial_pennies_l3_3992

-- Definitions of conditions
variables (a b : ℕ)
def condition1 : Prop := b + 2 = 4 * (a - 2)
def condition2 : Prop := b - 2 = 3 * (a + 2)

-- Goal: Proving that b = 62
theorem bob_initial_pennies (h1 : condition1 a b) (h2 : condition2 a b) : b = 62 :=
by {
  sorry
}

end bob_initial_pennies_l3_3992


namespace sum_second_largest_smallest_l3_3505

theorem sum_second_largest_smallest (a b c : ℕ) (order_cond : a < b ∧ b < c) : a + b = 21 :=
by
  -- Following the correct answer based on the provided conditions:
  -- 10, 11, and 12 with their ordering, we have the smallest a and the second largest b.
  sorry

end sum_second_largest_smallest_l3_3505


namespace sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l3_3330

-- Problem (1)
theorem sector_angle_given_circumference_and_area :
  (∀ (r l : ℝ), 2 * r + l = 10 ∧ (1 / 2) * l * r = 4 → l / r = (1 / 2)) := by
  sorry

-- Problem (2)
theorem max_sector_area_given_circumference :
  (∀ (r l : ℝ), 2 * r + l = 40 → (r = 10 ∧ l = 20 ∧ (1 / 2) * l * r = 100 ∧ l / r = 2)) := by
  sorry

end sector_angle_given_circumference_and_area_max_sector_area_given_circumference_l3_3330


namespace remaining_string_length_l3_3202

theorem remaining_string_length (original_length : ℝ) (given_to_Minyoung : ℝ) (fraction_used : ℝ) :
  original_length = 70 →
  given_to_Minyoung = 27 →
  fraction_used = 7/9 →
  abs (original_length - given_to_Minyoung - fraction_used * (original_length - given_to_Minyoung) - 9.56) < 0.01 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end remaining_string_length_l3_3202


namespace prove_sequences_and_sum_l3_3730

theorem prove_sequences_and_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (a 1 = 5) →
  (a 2 = 2) →
  (∀ n, 2 * (a n + a (n + 2)) = 5 * a (n + 1)) →
  (∀ n, ∃ r1, (a (n + 1) - 2 * a n) = (a 2 - 2 * a 1) * r1 ^ n) ∧
  (∀ n, ∃ r2, (a (n + 1) - (1 / 2) * a n) = (a 2 - (1 / 2) * a 1) * r2 ^ n) ∧
  (∀ n, S n = (4 * n) / 3 + (4 ^ n) / 36 - 1 / 36) :=
by
  sorry

end prove_sequences_and_sum_l3_3730


namespace geometric_N_digit_not_20_l3_3368

-- Variables and definitions
variables (a b c : ℕ)

-- Given conditions
def geometric_progression (a b c : ℕ) : Prop :=
  ∃ q : ℚ, (b = q * a) ∧ (c = q * b)

def ends_with_20 (N : ℕ) : Prop := N % 100 = 20

-- Prove the main theorem
theorem geometric_N_digit_not_20 (h1 : geometric_progression a b c) (h2 : ends_with_20 (a^3 + b^3 + c^3 - 3 * a * b * c)) :
  False :=
sorry

end geometric_N_digit_not_20_l3_3368


namespace factor_poly_l3_3285

theorem factor_poly (x : ℤ) :
  (x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 - x^9 + x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1)) :=
by
  sorry

end factor_poly_l3_3285


namespace pure_imaginary_solutions_l3_3582

theorem pure_imaginary_solutions:
  ∀ (x : ℂ), (x.im ≠ 0 ∧ x.re = 0) → (x ^ 4 - 5 * x ^ 3 + 10 * x ^ 2 - 50 * x - 75 = 0)
         → (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by
  sorry

end pure_imaginary_solutions_l3_3582


namespace david_age_l3_3419

theorem david_age (x : ℕ) (y : ℕ) (h1 : y = x + 7) (h2 : y = 2 * x) : x = 7 :=
by
  sorry

end david_age_l3_3419


namespace min_value_96_l3_3770

noncomputable def min_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) : ℝ :=
x^2 + 4 * x * y + 4 * y^2 + 2 * z^2

theorem min_value_96 (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_xyz : x * y * z = 32) :
  min_value x y z h_pos h_xyz = 96 :=
sorry

end min_value_96_l3_3770


namespace real_solution_exists_l3_3760

theorem real_solution_exists : ∃ x : ℝ, x^3 + (x+1)^4 + (x+2)^3 = (x+3)^4 :=
sorry

end real_solution_exists_l3_3760


namespace area_under_the_curve_l3_3671

theorem area_under_the_curve : 
  ∫ x in (0 : ℝ)..1, (x^2 + 1) = 4 / 3 := 
by
  sorry

end area_under_the_curve_l3_3671


namespace total_percent_decrease_l3_3931

theorem total_percent_decrease (initial_value : ℝ) (val1 val2 : ℝ) :
  initial_value > 0 →
  val1 = initial_value * (1 - 0.60) →
  val2 = val1 * (1 - 0.10) →
  (initial_value - val2) / initial_value * 100 = 64 :=
by
  intros h_initial h_val1 h_val2
  sorry

end total_percent_decrease_l3_3931


namespace largest_angle_right_triangle_l3_3883

theorem largest_angle_right_triangle
  (a b c : ℝ)
  (h₁ : ∃ x : ℝ, x^2 + 4 * (c + 2) = (c + 4) * x)
  (h₂ : a + b = c + 4)
  (h₃ : a * b = 4 * (c + 2))
  : ∃ x : ℝ, x = 90 :=
by {
  sorry
}

end largest_angle_right_triangle_l3_3883


namespace assignment_statement_increases_l3_3844

theorem assignment_statement_increases (N : ℕ) : (N + 1 = N + 1) :=
sorry

end assignment_statement_increases_l3_3844


namespace ashok_avg_first_five_l3_3171

-- Define the given conditions 
def avg (n : ℕ) (s : ℕ) : ℕ := s / n

def total_marks (average : ℕ) (num_subjects : ℕ) : ℕ := average * num_subjects

variables (avg_six_subjects : ℕ := 76)
variables (sixth_subject_marks : ℕ := 86)
variables (total_six_subjects : ℕ := total_marks avg_six_subjects 6)
variables (total_first_five_subjects : ℕ := total_six_subjects - sixth_subject_marks)
variables (avg_first_five_subjects : ℕ := avg 5 total_first_five_subjects)

-- State the theorem
theorem ashok_avg_first_five 
  (h1 : avg_six_subjects = 76)
  (h2 : sixth_subject_marks = 86)
  (h3 : avg_first_five_subjects = 74)
  : avg 5 (total_marks 76 6 - 86) = 74 := 
sorry

end ashok_avg_first_five_l3_3171


namespace ravi_refrigerator_purchase_price_l3_3959

theorem ravi_refrigerator_purchase_price (purchase_price_mobile : ℝ) (sold_mobile : ℝ)
  (profit : ℝ) (loss : ℝ) (overall_profit : ℝ)
  (H1 : purchase_price_mobile = 8000)
  (H2 : loss = 0.04)
  (H3 : profit = 0.10)
  (H4 : overall_profit = 200) :
  ∃ R : ℝ, 0.96 * R + sold_mobile = R + purchase_price_mobile + overall_profit ∧ R = 15000 :=
by
  use 15000
  sorry

end ravi_refrigerator_purchase_price_l3_3959


namespace hyperbola_equation_l3_3457

noncomputable def sqrt_cubed := Real.sqrt 3

theorem hyperbola_equation
  (P : ℝ × ℝ)
  (a b : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (hP : P = (1, sqrt_cubed))
  (hAsymptote : (1 / a)^2 - (sqrt_cubed / b)^2 = 0)
  (hAngle : ∀ F : ℝ × ℝ, ∀ O : ℝ × ℝ, (F.1 - 1)^2 + (F.2 - sqrt_cubed)^2 + F.1^2 + F.2^2 = 16) :
  (a^2 = 4) ∧ (b^2 = 12) ∧ (c = 4) →
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 :=
by
  sorry

end hyperbola_equation_l3_3457


namespace inequality_x_y_l3_3236

theorem inequality_x_y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) : 
  (x / (x + 5 * y)) + (y / (y + 5 * x)) ≤ 1 := 
by 
  sorry

end inequality_x_y_l3_3236


namespace largest_integral_x_l3_3445

theorem largest_integral_x (x : ℤ) : (1 / 4 : ℚ) < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ↔ x = 4 :=
by 
  sorry

end largest_integral_x_l3_3445


namespace monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l3_3874

-- Proof Problem I
noncomputable def f1 (x : ℝ) := x^2 + x - Real.log x

theorem monotonic_intervals_a1 : 
  (∀ x, 0 < x ∧ x < 1 / 2 → f1 x < 0) ∧ (∀ x, 1 / 2 < x → f1 x > 0) := 
sorry

-- Proof Problem II
noncomputable def f2 (x : ℝ) (a : ℝ) := x^2 + a * x - Real.log x

theorem decreasing_on_1_to_2 (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ 2 → f2 x a ≤ 0) → a ≤ -7 / 2 :=
sorry

-- Proof Problem III
noncomputable def g (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem exists_a_for_minimum_value :
  ∃ a : ℝ, (∀ x, 0 < x ∧ x ≤ Real.exp 1 → g x a = 3) ∧ a = Real.exp 2 :=
sorry

end monotonic_intervals_a1_decreasing_on_1_to_2_exists_a_for_minimum_value_l3_3874


namespace crystal_meal_combinations_l3_3608

-- Definitions for conditions:
def entrees := 4
def drinks := 4
def desserts := 3 -- includes two desserts and the option of no dessert

-- Statement of the problem as a theorem:
theorem crystal_meal_combinations : entrees * drinks * desserts = 48 := by
  sorry

end crystal_meal_combinations_l3_3608


namespace determine_m_ratio_l3_3932

def ratio_of_C_to_A_investment (x : ℕ) (m : ℕ) (total_gain : ℕ) (a_share : ℕ) : Prop :=
  total_gain = 18000 ∧ a_share = 6000 ∧
  (12 * x / (12 * x + 4 * m * x) = 1 / 3)

theorem determine_m_ratio (x : ℕ) (m : ℕ) (h : ratio_of_C_to_A_investment x m 18000 6000) :
  m = 6 :=
by
  sorry

end determine_m_ratio_l3_3932


namespace max_min_M_l3_3371

noncomputable def M (x y : ℝ) : ℝ :=
  abs (x + y) + abs (y + 1) + abs (2 * y - x - 4)

theorem max_min_M (x y : ℝ) (hx : abs x ≤ 1) (hy : abs y ≤ 1) :
  3 ≤ M x y ∧ M x y ≤ 7 :=
sorry

end max_min_M_l3_3371


namespace remaining_nap_time_is_three_hours_l3_3305

-- Define the flight time and the times spent on various activities
def flight_time_minutes := 11 * 60 + 20
def reading_time_minutes := 2 * 60
def movie_time_minutes := 4 * 60
def dinner_time_minutes := 30
def radio_time_minutes := 40
def game_time_minutes := 60 + 10

-- Calculate the total time spent on activities
def total_activity_time_minutes :=
  reading_time_minutes + movie_time_minutes + dinner_time_minutes + radio_time_minutes + game_time_minutes

-- Calculate the remaining time for a nap
def remaining_nap_time_minutes :=
  flight_time_minutes - total_activity_time_minutes

-- Convert the remaining nap time to hours
def remaining_nap_time_hours :=
  remaining_nap_time_minutes / 60

-- The statement to be proved
theorem remaining_nap_time_is_three_hours :
  remaining_nap_time_hours = 3 := by
  sorry

#check remaining_nap_time_is_three_hours -- This will check if the theorem statement is correct

end remaining_nap_time_is_three_hours_l3_3305


namespace latest_time_for_60_degrees_l3_3019

def temperature_at_time (t : ℝ) : ℝ :=
  -2 * t^2 + 16 * t + 40

theorem latest_time_for_60_degrees (t : ℝ) :
  temperature_at_time t = 60 → t = 5 :=
sorry

end latest_time_for_60_degrees_l3_3019


namespace partI_partII_l3_3848

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x
noncomputable def f' (x m : ℝ) : ℝ := (1 / x) - m

theorem partI (m : ℝ) : (∃ x : ℝ, x > 0 ∧ f x m = -1) → m = 1 := by
  sorry

theorem partII (x1 x2 : ℝ) (h1 : e ^ x1 ≤ x2) (h2 : f x1 1 = 0) (h3 : f x2 1 = 0) :
  ∃ y : ℝ, y = (x1 - x2) * f' (x1 + x2) 1 ∧ y = 2 / (1 + Real.exp 1) := by
  sorry

end partI_partII_l3_3848


namespace find_b_l3_3203

noncomputable def a_and_b_integers_and_factor (a b : ℤ) : Prop :=
  ∀ (x : ℝ), (x^2 - x - 1) * (a*x^3 + b*x^2 - x + 1) = 0

theorem find_b (a b : ℤ) (h : a_and_b_integers_and_factor a b) : b = -1 :=
by 
  sorry

end find_b_l3_3203


namespace cube_tangent_ratio_l3_3694

theorem cube_tangent_ratio 
  (edge_length : ℝ) 
  (midpoint K : ℝ) 
  (tangent E : ℝ) 
  (intersection F : ℝ) 
  (radius R : ℝ)
  (h1 : edge_length = 2)
  (h2 : radius = 1)
  (h3 : K = midpoint)
  (h4 : ∃ E F, tangent = E ∧ intersection = F) :
  (K - E) / (F - E) = 4 / 5 :=
sorry

end cube_tangent_ratio_l3_3694


namespace find_r_l3_3487

theorem find_r : ∃ r : ℕ, (5 + 7 * 8 + 1 * 8^2) = 120 + r ∧ r = 5 := 
by
  use 5
  sorry

end find_r_l3_3487


namespace sum_q_p_eq_zero_l3_3442

def p (x : Int) : Int := x^2 - 4

def q (x : Int) : Int := 
  if x ≥ 0 then -x
  else x

def q_p (x : Int) : Int := q (p x)

#eval List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0

theorem sum_q_p_eq_zero :
  List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0 :=
sorry

end sum_q_p_eq_zero_l3_3442


namespace fox_jeans_price_l3_3749

theorem fox_jeans_price (pony_price : ℝ)
                        (total_savings : ℝ)
                        (total_discount_rate : ℝ)
                        (pony_discount_rate : ℝ)
                        (fox_discount_rate : ℝ)
                        (fox_price : ℝ) :
    pony_price = 18 ∧
    total_savings = 8.91 ∧
    total_discount_rate = 0.22 ∧
    pony_discount_rate = 0.1099999999999996 ∧
    fox_discount_rate = 0.11 →
    (3 * fox_discount_rate * fox_price + 2 * pony_discount_rate * pony_price = total_savings) →
    fox_price = 15 :=
by
  intros h h_eq
  rcases h with ⟨h_pony, h_savings, h_total_rate, h_pony_rate, h_fox_rate⟩
  sorry

end fox_jeans_price_l3_3749


namespace subtraction_of_fractions_l3_3664

theorem subtraction_of_fractions :
  1 + 1 / 2 - 3 / 5 = 9 / 10 := by
  sorry

end subtraction_of_fractions_l3_3664


namespace problem1_problem2_problem3_l3_3703

-- Definitions of transformations and final sequence S
def transformation (A : List ℕ) : List ℕ := 
  match A with
  | x :: y :: xs => (x + y) :: transformation (y :: xs)
  | _ => []

def nth_transform (A : List ℕ) (n : ℕ) : List ℕ :=
  Nat.iterate (λ L => transformation L) n A

def final_sequence (A : List ℕ) : ℕ :=
  match nth_transform A (A.length - 1) with
  | [x] => x
  | _ => 0

-- Proof Statements

theorem problem1 : final_sequence [1, 2, 3] = 8 := sorry

theorem problem2 (n : ℕ) : final_sequence (List.range (n+1)) = (n + 2) * 2 ^ (n - 1) := sorry

theorem problem3 (A B : List ℕ) (h : A = List.range (B.length)) (h_perm : B.permutations.contains A) : 
  final_sequence B = final_sequence A := by
  sorry

end problem1_problem2_problem3_l3_3703


namespace test_takers_percent_correct_l3_3975

theorem test_takers_percent_correct 
  (n : Set ℕ → ℝ) 
  (A B : Set ℕ) 
  (hB : n B = 0.75) 
  (hAB : n (A ∩ B) = 0.60) 
  (hneither : n (Set.univ \ (A ∪ B)) = 0.05) 
  : n A = 0.80 := by
  sorry

end test_takers_percent_correct_l3_3975


namespace inequality_always_holds_l3_3578

theorem inequality_always_holds (a b c : ℝ) (h : a > b) : (a - b) * c^2 ≥ 0 :=
by 
  sorry

end inequality_always_holds_l3_3578


namespace part1_part2_l3_3327

noncomputable def point_M (m : ℝ) : ℝ × ℝ := (2 * m + 1, m - 4)
def point_N : ℝ × ℝ := (5, 2)

theorem part1 (m : ℝ) (h : m - 4 = 2) : point_M m = (13, 2) := by
  sorry

theorem part2 (m : ℝ) (h : 2 * m + 1 = 3) : point_M m = (3, -3) := by
  sorry

end part1_part2_l3_3327


namespace prism_volume_l3_3453

theorem prism_volume (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 50) (h3 : b * c = 75) :
  a * b * c = 150 * Real.sqrt 5 :=
by
  sorry

end prism_volume_l3_3453


namespace max_students_in_auditorium_l3_3597

def increment (i : ℕ) : ℕ :=
  (i * (i + 1)) / 2

def seats_in_row (i : ℕ) : ℕ :=
  10 + increment i

def max_students_in_row (n : ℕ) : ℕ :=
  (n + 1) / 2

def total_max_students_up_to_row (n : ℕ) : ℕ :=
  (Finset.range n).sum (λ i => max_students_in_row (seats_in_row (i + 1)))

theorem max_students_in_auditorium : total_max_students_up_to_row 20 = 335 := 
sorry

end max_students_in_auditorium_l3_3597


namespace inconsistent_proportion_l3_3193

theorem inconsistent_proportion (a b : ℝ) (h1 : 3 * a = 5 * b) (ha : a ≠ 0) (hb : b ≠ 0) : ¬ (a / b = 3 / 5) :=
sorry

end inconsistent_proportion_l3_3193


namespace shopkeeper_profit_percentage_l3_3741

theorem shopkeeper_profit_percentage (C : ℝ) (hC : C > 0) :
  let selling_price := 12 * C
  let cost_price := 10 * C
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by
  sorry

end shopkeeper_profit_percentage_l3_3741


namespace geometric_sequence_sum_l3_3785

variable {a : ℕ → ℕ}

-- Defining the geometric sequence and the conditions
def is_geometric_sequence (a : ℕ → ℕ) (q : ℕ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 (a : ℕ → ℕ) : Prop :=
  a 1 = 3

def condition2 (a : ℕ → ℕ) : Prop :=
  a 1 + a 3 + a 5 = 21

-- The main theorem
theorem geometric_sequence_sum (a : ℕ → ℕ) (q : ℕ) 
  (h1 : condition1 a) (h2: condition2 a) (hq : is_geometric_sequence a q) : 
  a 3 + a 5 + a 7 = 42 := 
sorry

end geometric_sequence_sum_l3_3785


namespace average_time_per_stop_l3_3511

theorem average_time_per_stop (pizzas : ℕ) 
                              (stops_for_two_pizzas : ℕ) 
                              (pizzas_per_stop_for_two : ℕ) 
                              (remaining_pizzas : ℕ) 
                              (total_stops : ℕ) 
                              (total_time : ℕ) 
                              (H1: pizzas = 12) 
                              (H2: stops_for_two_pizzas = 2) 
                              (H3: pizzas_per_stop_for_two = 2) 
                              (H4: remaining_pizzas = pizzas - stops_for_two_pizzas * pizzas_per_stop_for_two)
                              (H5: total_stops = stops_for_two_pizzas + remaining_pizzas)
                              (H6: total_time = 40) :
                              total_time / total_stops = 4 :=
by
  sorry

end average_time_per_stop_l3_3511


namespace sum_of_squares_l3_3978

theorem sum_of_squares (a d : Int) : 
  ∃ y1 y2 : Int, a^2 + 2*(a+d)^2 + 3*(a+2*d)^2 + 4*(a+3*d)^2 = (3*a + y1*d)^2 + (a + y2*d)^2 :=
by
  sorry

end sum_of_squares_l3_3978


namespace value_of_m_l3_3450

theorem value_of_m (m : ℕ) : (5^m = 5 * 25^2 * 125^3) → m = 14 :=
by
  sorry

end value_of_m_l3_3450


namespace simplify_and_evaluate_expression_l3_3065

def a : ℚ := 1 / 3
def b : ℚ := -1
def expr : ℚ := 4 * (3 * a^2 * b - a * b^2) - (2 * a * b^2 + 3 * a^2 * b)

theorem simplify_and_evaluate_expression : expr = -3 := 
by
  sorry

end simplify_and_evaluate_expression_l3_3065


namespace solve_system_l3_3569

theorem solve_system : 
  ∀ (a b c : ℝ), 
  (a * (b^2 + c) = c * (c + a * b) ∧ 
   b * (c^2 + a) = a * (a + b * c) ∧ 
   c * (a^2 + b) = b * (b + c * a)) 
   → (∃ t : ℝ, a = t ∧ b = t ∧ c = t) :=
by
  intros a b c h
  sorry

end solve_system_l3_3569


namespace percentage_disliked_by_both_l3_3841

theorem percentage_disliked_by_both (total_comics liked_by_females liked_by_males disliked_by_both : ℕ) 
  (total_comics_eq : total_comics = 300)
  (liked_by_females_eq : liked_by_females = 30 * total_comics / 100)
  (liked_by_males_eq : liked_by_males = 120)
  (disliked_by_both_eq : disliked_by_both = total_comics - (liked_by_females + liked_by_males)) :
  (disliked_by_both * 100 / total_comics) = 30 := by
  sorry

end percentage_disliked_by_both_l3_3841


namespace fill_blanks_l3_3229

/-
Given the following conditions:
1. 20 * (x1 - 8) = 20
2. x2 / 2 + 17 = 20
3. 3 * x3 - 4 = 20
4. (x4 + 8) / 12 = y4
5. 4 * x5 = 20
6. 20 * (x6 - y6) = 100

Prove that:
1. x1 = 9
2. x2 = 6
3. x3 = 8
4. x4 = 4 and y4 = 1
5. x5 = 5
6. x6 = 7 and y6 = 2
-/
theorem fill_blanks (x1 x2 x3 x4 y4 x5 x6 y6 : ℕ) :
  20 * (x1 - 8) = 20 →
  x2 / 2 + 17 = 20 →
  3 * x3 - 4 = 20 →
  (x4 + 8) / 12 = y4 →
  4 * x5 = 20 →
  20 * (x6 - y6) = 100 →
  x1 = 9 ∧
  x2 = 6 ∧
  x3 = 8 ∧
  x4 = 4 ∧
  y4 = 1 ∧
  x5 = 5 ∧
  x6 = 7 ∧
  y6 = 2 :=
by
  sorry

end fill_blanks_l3_3229


namespace find_a_n_l3_3572

theorem find_a_n (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n, S n = 3^n + 2) :
  ∀ n, a n = if n = 1 then 5 else 2 * 3^(n - 1) := by
  sorry

end find_a_n_l3_3572


namespace actual_distance_traveled_l3_3266

-- Given conditions
variables (D : ℝ)
variables (H : D / 5 = (D + 20) / 15)

-- The proof problem statement
theorem actual_distance_traveled : D = 10 :=
by
  sorry

end actual_distance_traveled_l3_3266


namespace snow_at_least_once_prob_l3_3060

-- Define the conditions for the problem
def prob_snow_day1_to_day4 : ℚ := 1 / 2
def prob_no_snow_day1_to_day4 : ℚ := 1 - prob_snow_day1_to_day4

def prob_snow_day5_to_day7 : ℚ := 1 / 3
def prob_no_snow_day5_to_day7 : ℚ := 1 - prob_snow_day5_to_day7

-- Define the probability of no snow during the first week of February
def prob_no_snow_week : ℚ := (prob_no_snow_day1_to_day4 ^ 4) * (prob_no_snow_day5_to_day7 ^ 3)

-- Define the probability that it snows at least once during the first week of February
def prob_snow_at_least_once : ℚ := 1 - prob_no_snow_week

-- The theorem we want to prove
theorem snow_at_least_once_prob : prob_snow_at_least_once = 53 / 54 :=
by
  sorry

end snow_at_least_once_prob_l3_3060


namespace p_minus_q_value_l3_3798

theorem p_minus_q_value (p q : ℝ) (h1 : (x - 4) * (x + 4) = 24 * x - 96) (h2 : x^2 - 24 * x + 80 = 0) (h3 : p = 20) (h4 : q = 4) : p - q = 16 :=
by
  sorry

end p_minus_q_value_l3_3798


namespace ratio_final_to_original_l3_3903

-- Given conditions
variable (d : ℝ)
variable (h1 : 364 = d * 1.30)

-- Problem statement
theorem ratio_final_to_original : (364 / d) = 1.3 := 
by sorry

end ratio_final_to_original_l3_3903


namespace infinite_series_sum_l3_3794

theorem infinite_series_sum :
  ∑' n : ℕ, n / (8 : ℝ) ^ n = (8 / 49 : ℝ) :=
sorry

end infinite_series_sum_l3_3794


namespace tan_alpha_value_l3_3147

open Real

theorem tan_alpha_value 
  (α : ℝ) 
  (hα_range : 0 < α ∧ α < π) 
  (h_cos_alpha : cos α = -3/5) :
  tan α = -4/3 := 
by
  sorry

end tan_alpha_value_l3_3147


namespace diane_total_loss_l3_3513

-- Define the starting amount of money Diane had.
def starting_amount : ℤ := 100

-- Define the amount of money Diane won.
def winnings : ℤ := 65

-- Define the amount of money Diane owed at the end.
def debt : ℤ := 50

-- Define the total amount of money Diane had after winnings.
def mid_game_total : ℤ := starting_amount + winnings

-- Define the total amount Diane lost.
def total_loss : ℤ := mid_game_total + debt

-- Theorem stating the total amount Diane lost is 215 dollars.
theorem diane_total_loss : total_loss = 215 := by
  sorry

end diane_total_loss_l3_3513


namespace area_of_region_is_12_l3_3097

def region_area : ℝ :=
  let f1 (x : ℝ) : ℝ := |x - 2|
  let f2 (x : ℝ) : ℝ := 5 - |x + 1|
  let valid_region (x y : ℝ) : Prop := f1 x ≤ y ∧ y ≤ f2 x
  12

theorem area_of_region_is_12 :
  ∃ (area : ℝ), region_area = 12 := by
  use 12
  sorry

end area_of_region_is_12_l3_3097


namespace gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l3_3956

theorem gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4 (k : Int) :
  Int.gcd ((360 * k)^2 + 6 * (360 * k) + 8) (360 * k + 4) = 4 := 
sorry

end gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l3_3956


namespace find_k_l3_3215

theorem find_k (x₁ x₂ k : ℝ) (hx : x₁ + x₂ = 3) (h_prod : x₁ * x₂ = k) (h_cond : x₁ * x₂ + 2 * x₁ + 2 * x₂ = 1) : k = -5 :=
by
  sorry

end find_k_l3_3215


namespace find_third_number_l3_3387

-- Definitions and conditions for the problem
def x : ℚ := 1.35
def third_number := 5
def proportion (a b c d : ℚ) := a * d = b * c 

-- Proposition to prove
theorem find_third_number : proportion 0.75 x third_number 9 := 
by
  -- It's advisable to split the proof steps here, but the proof itself is condensed.
  sorry

end find_third_number_l3_3387


namespace union_M_N_is_U_l3_3136

-- Defining the universal set as the set of real numbers
def U : Set ℝ := Set.univ

-- Defining the set M
def M : Set ℝ := {x | x > 0}

-- Defining the set N
def N : Set ℝ := {x | x^2 >= x}

-- Stating the theorem that M ∪ N = U
theorem union_M_N_is_U : M ∪ N = U :=
  sorry

end union_M_N_is_U_l3_3136


namespace cos_of_angle_in_third_quadrant_l3_3295

theorem cos_of_angle_in_third_quadrant (A : ℝ) (hA : π < A ∧ A < 3 * π / 2) (h_sin : Real.sin A = -1 / 3) :
  Real.cos A = -2 * Real.sqrt 2 / 3 :=
by
  sorry

end cos_of_angle_in_third_quadrant_l3_3295


namespace difference_of_squares_l3_3182

-- Definition of the constants a and b as given in the problem
def a := 502
def b := 498

theorem difference_of_squares : a^2 - b^2 = 4000 := by
  sorry

end difference_of_squares_l3_3182


namespace smallest_k_for_sum_of_squares_multiple_of_360_l3_3895

theorem smallest_k_for_sum_of_squares_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (k * (k + 1) * (2 * k + 1)) / 6 % 360 = 0 ∧ ∀ n : ℕ, n > 0 → (n * (n + 1) * (2 * n + 1)) / 6 % 360 = 0 → k ≤ n :=
by sorry

end smallest_k_for_sum_of_squares_multiple_of_360_l3_3895


namespace fencing_required_l3_3098

theorem fencing_required {length width : ℝ} 
  (uncovered_side : length = 20)
  (field_area : length * width = 50) :
  2 * width + length = 25 :=
by
  sorry

end fencing_required_l3_3098


namespace smallest_marble_count_l3_3455

theorem smallest_marble_count (N : ℕ) (a b c : ℕ) (h1 : N > 1)
  (h2 : N ≡ 2 [MOD 5])
  (h3 : N ≡ 2 [MOD 7])
  (h4 : N ≡ 2 [MOD 9]) : N = 317 :=
sorry

end smallest_marble_count_l3_3455


namespace product_of_sums_of_squares_l3_3336

theorem product_of_sums_of_squares (a b : ℤ) 
  (h1 : ∃ x1 y1 : ℤ, a = x1^2 + y1^2)
  (h2 : ∃ x2 y2 : ℤ, b = x2^2 + y2^2) : 
  ∃ x y : ℤ, a * b = x^2 + y^2 :=
by
  sorry

end product_of_sums_of_squares_l3_3336


namespace total_members_in_club_l3_3372

theorem total_members_in_club (females : ℕ) (males : ℕ) (total : ℕ) : 
  (females = 12) ∧ (females = 2 * males) ∧ (total = females + males) → total = 18 := 
by
  sorry

end total_members_in_club_l3_3372


namespace max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l3_3269

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x
noncomputable def f' (x : ℝ) : ℝ := Real.cos x - Real.sin x
noncomputable def g (x : ℝ) : ℝ := f x * f' x - f x ^ 2

theorem max_value_and_period_of_g :
  ∃ (M : ℝ) (T : ℝ), (∀ x, g x ≤ M) ∧ (∀ x, g (x + T) = g x) ∧ M = 2 ∧ T = Real.pi :=
sorry

theorem value_of_expression_if_fx_eq_2f'x (x : ℝ) :
  f x = 2 * f' x → (1 + Real.sin x ^ 2) / (Real.cos x ^ 2 - Real.sin x * Real.cos x) = 11 / 6 :=
sorry

end max_value_and_period_of_g_value_of_expression_if_fx_eq_2f_l3_3269


namespace triangle_angle_A_l3_3783

theorem triangle_angle_A (a c C A : Real) (h1 : a = 1) (h2 : c = Real.sqrt 3) (h3 : C = 2 * Real.pi / 3) 
(h4 : Real.sin A = 1 / 2) : A = Real.pi / 6 :=
sorry

end triangle_angle_A_l3_3783


namespace zach_needs_more_money_zach_more_money_needed_l3_3094

/-!
# Zach's Bike Savings Problem
Zach needs $100 to buy a brand new bike.
Weekly allowance: $5.
Earnings from mowing the lawn: $10.
Earnings from babysitting: $7 per hour.
Zach has already saved $65.
He will receive weekly allowance on Friday.
He will mow the lawn and babysit for 2 hours this Saturday.
Prove that Zach needs $6 more to buy the bike.
-/

def zach_current_savings : ℕ := 65
def bike_cost : ℕ := 100
def weekly_allowance : ℕ := 5
def mowing_earnings : ℕ := 10
def babysitting_rate : ℕ := 7
def babysitting_hours : ℕ := 2

theorem zach_needs_more_money : zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours)) = 94 :=
by sorry

theorem zach_more_money_needed : bike_cost - (zach_current_savings + (weekly_allowance + mowing_earnings + (babysitting_rate * babysitting_hours))) = 6 :=
by sorry

end zach_needs_more_money_zach_more_money_needed_l3_3094


namespace Bobby_paycheck_final_amount_l3_3360

theorem Bobby_paycheck_final_amount :
  let salary := 450
  let federal_tax := (1 / 3 : ℚ) * salary
  let state_tax := 0.08 * salary
  let health_insurance := 50
  let life_insurance := 20
  let city_fee := 10
  let total_deductions := federal_tax + state_tax + health_insurance + life_insurance + city_fee
  salary - total_deductions = 184 :=
by
  -- We put sorry here to skip the proof step
  sorry

end Bobby_paycheck_final_amount_l3_3360


namespace triangle_side_ratio_l3_3934

theorem triangle_side_ratio (a b c : ℝ) (h1 : a + b ≤ 2 * c) (h2 : b + c ≤ 3 * a) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  2 / 3 < c / a ∧ c / a < 2 :=
by
  sorry

end triangle_side_ratio_l3_3934


namespace consecutive_numbers_product_l3_3647

theorem consecutive_numbers_product (a b c d : ℤ) 
  (h1 : b = a + 1) 
  (h2 : c = a + 2) 
  (h3 : d = a + 3) 
  (h4 : a + d = 109) : 
  b * c = 2970 := by
  sorry

end consecutive_numbers_product_l3_3647


namespace negation_example_l3_3623

theorem negation_example :
  (¬ (∀ x : ℝ, x^2 - 2 * x + 1 > 0)) ↔ (∃ x : ℝ, x^2 - 2 * x + 1 ≤ 0) :=
sorry

end negation_example_l3_3623


namespace possible_division_l3_3602

theorem possible_division (side_length : ℕ) (areas : Fin 5 → Set (Fin side_length × Fin side_length))
  (h1 : side_length = 5)
  (h2 : ∀ i, ∃ cells : Finset (Fin side_length × Fin side_length), areas i = cells ∧ Finset.card cells = 5)
  (h3 : ∀ i j, i ≠ j → Disjoint (areas i) (areas j))
  (total_cut_length : ℕ)
  (h4 : total_cut_length ≤ 16) :
  
  ∃ cuts : Finset (Fin side_length × Fin side_length) × Finset (Fin side_length × Fin side_length),
    total_cut_length = (cuts.1.card + cuts.2.card) :=
sorry

end possible_division_l3_3602


namespace fewest_handshakes_is_zero_l3_3697

noncomputable def fewest_handshakes (n k : ℕ) : ℕ :=
  if h : (n * (n - 1)) / 2 + k = 325 then k else 325

theorem fewest_handshakes_is_zero :
  ∃ n k : ℕ, (n * (n - 1)) / 2 + k = 325 ∧ 0 = fewest_handshakes n k :=
by
  sorry

end fewest_handshakes_is_zero_l3_3697


namespace generate_sequence_next_three_members_l3_3410

-- Define the function that generates the sequence
def f (n : ℕ) : ℕ := 2 * (n + 1) ^ 2 * (n + 2) ^ 2

-- Define the predicate that checks if a number can be expressed as the sum of squares of two positive integers
def is_sum_of_squares_of_two_positives (k : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ a^2 + b^2 = k

-- The problem statement to prove the equivalence
theorem generate_sequence_next_three_members :
  is_sum_of_squares_of_two_positives (f 1) ∧
  is_sum_of_squares_of_two_positives (f 2) ∧
  is_sum_of_squares_of_two_positives (f 3) ∧
  is_sum_of_squares_of_two_positives (f 4) ∧
  is_sum_of_squares_of_two_positives (f 5) ∧
  is_sum_of_squares_of_two_positives (f 6) ∧
  f 1 = 72 ∧
  f 2 = 288 ∧
  f 3 = 800 ∧
  f 4 = 1800 ∧
  f 5 = 3528 ∧
  f 6 = 6272 :=
sorry

end generate_sequence_next_three_members_l3_3410


namespace a_n_general_term_b_n_general_term_l3_3107

noncomputable def seq_a (n : ℕ) : ℕ :=
  2 * n - 1

theorem a_n_general_term (n : ℕ) (Sn : ℕ → ℕ) (S_property : ∀ n : ℕ, 4 * Sn n = (seq_a n) ^ 2 + 2 * seq_a n + 1) :
  seq_a n = 2 * n - 1 :=
sorry

noncomputable def geom_seq (q : ℕ) (n : ℕ) : ℕ :=
  q ^ (n - 1)

theorem b_n_general_term (n m q : ℕ) (a1 am am3 : ℕ) (b_property : ∀ n : ℕ, geom_seq q n = q ^ (n - 1))
  (a_property : ∀ n : ℕ, seq_a n = 2 * n - 1)
  (b1_condition : geom_seq q 1 = seq_a 1) (bm_condition : geom_seq q m = seq_a m)
  (bm1_condition : geom_seq q (m + 1) = seq_a (m + 3)) :
  q = 3 ∨ q = 7 ∧ (∀ n : ℕ, geom_seq q n = 3 ^ (n - 1) ∨ geom_seq q n = 7 ^ (n - 1)) :=
sorry

end a_n_general_term_b_n_general_term_l3_3107


namespace last_digit_of_expression_l3_3329

-- Conditions
def a : ℤ := 25
def b : ℤ := -3

-- Statement to be proved
theorem last_digit_of_expression :
  (a ^ 1999 + b ^ 2002) % 10 = 4 :=
by
  -- proof would go here
  sorry

end last_digit_of_expression_l3_3329


namespace quadratic_root_form_eq_l3_3056

theorem quadratic_root_form_eq (c : ℚ) : 
  (∀ x : ℚ, x^2 - 7 * x + c = 0 → x = (7 + Real.sqrt (9 * c)) / 2 ∨ x = (7 - Real.sqrt (9 * c)) / 2) →
  c = 49 / 13 := 
by
  sorry

end quadratic_root_form_eq_l3_3056


namespace solve_system_of_equations_l3_3524

theorem solve_system_of_equations (x y z : ℝ) :
  x + y + z = 2 →
  x * y * z = 2 * (x * y + y * z + z * x) →
  ((x = -y ∧ z = 2) ∨ (y = -z ∧ x = 2) ∨ (z = -x ∧ y = 2)) :=
by
  intros h1 h2
  sorry

end solve_system_of_equations_l3_3524


namespace M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l3_3314

-- Definition of the curve using parametric equations
def curve (t : ℝ) : ℝ × ℝ :=
  (3 * t, 2 * t^2 + 1)

-- Questions and proof statements
theorem M1_on_curve_C : ∃ t : ℝ, curve t = (0, 1) :=
by { 
  sorry 
}

theorem M2_not_on_curve_C : ¬ (∃ t : ℝ, curve t = (5, 4)) :=
by { 
  sorry 
}

theorem M3_on_curve_C_a_eq_9 (a : ℝ) : (∃ t : ℝ, curve t = (6, a)) → a = 9 :=
by { 
  sorry 
}

end M1_on_curve_C_M2_not_on_curve_C_M3_on_curve_C_a_eq_9_l3_3314


namespace berry_ratio_l3_3076

-- Define the conditions
variables (S V R : ℕ) -- Number of berries Stacy, Steve, and Sylar have
axiom h1 : S + V + R = 1100
axiom h2 : S = 800
axiom h3 : V = 2 * R

-- Define the theorem to be proved
theorem berry_ratio (h1 : S + V + R = 1100) (h2 : S = 800) (h3 : V = 2 * R) : S / V = 4 :=
by
  sorry

end berry_ratio_l3_3076


namespace mul_inv_mod_391_l3_3808

theorem mul_inv_mod_391 (a : ℤ) (ha : 143 * a % 391 = 1) : a = 28 := by
  sorry

end mul_inv_mod_391_l3_3808


namespace probability_N_taller_than_L_l3_3962

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l3_3962


namespace coloring_equilateral_triangle_l3_3110

theorem coloring_equilateral_triangle :
  ∀ (A B C : Type) (color : A → Type) (d : A → A → ℝ),
  (∀ x y, d x y = 1 → color x = color y) :=
by sorry

end coloring_equilateral_triangle_l3_3110


namespace ten_percent_of_number_l3_3510

theorem ten_percent_of_number (x : ℝ)
  (h : x - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 3.325 :=
sorry

end ten_percent_of_number_l3_3510


namespace number_of_recipes_needed_l3_3398

noncomputable def cookies_per_student : ℕ := 3
noncomputable def total_students : ℕ := 150
noncomputable def recipe_yield : ℕ := 20
noncomputable def attendance_drop_rate : ℝ := 0.30

theorem number_of_recipes_needed : 
  ⌈ (total_students * (1 - attendance_drop_rate) * cookies_per_student) / recipe_yield ⌉ = 16 := by
  sorry

end number_of_recipes_needed_l3_3398


namespace count_points_l3_3070

theorem count_points (a b : ℝ) :
  (abs b = 2) ∧ (abs a = 4) → (∃ (P : ℝ × ℝ), P = (a, b) ∧ (abs b = 2) ∧ (abs a = 4) ∧
    ((a = 4 ∨ a = -4) ∧ (b = 2 ∨ b = -2)) ∧
    (P = (4, 2) ∨ P = (4, -2) ∨ P = (-4, 2) ∨ P = (-4, -2)) ∧
    ∃ n, n = 4) :=
sorry

end count_points_l3_3070


namespace boxes_in_case_number_of_boxes_in_case_l3_3661

-- Definitions based on the conditions
def boxes_of_eggs : Nat := 5
def eggs_per_box : Nat := 3
def total_eggs : Nat := 15

-- Proposition
theorem boxes_in_case (boxes_of_eggs : Nat) (eggs_per_box : Nat) (total_eggs : Nat) : Nat :=
  if boxes_of_eggs * eggs_per_box = total_eggs then boxes_of_eggs else 0

-- Assertion that needs to be proven
theorem number_of_boxes_in_case : boxes_in_case boxes_of_eggs eggs_per_box total_eggs = 5 :=
by sorry

end boxes_in_case_number_of_boxes_in_case_l3_3661


namespace ratio_S15_S5_l3_3508

variable {α : Type*} [LinearOrderedField α]

namespace ArithmeticSequence

def sum_of_first_n_terms (a : α) (d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem ratio_S15_S5
  {a d : α}
  {S5 S10 S15 : α}
  (h1 : S5 = sum_of_first_n_terms a d 5)
  (h2 : S10 = sum_of_first_n_terms a d 10)
  (h3 : S15 = sum_of_first_n_terms a d 15)
  (h_ratio : S5 / S10 = 2 / 3) :
  S15 / S5 = 3 / 2 := 
sorry

end ArithmeticSequence

end ratio_S15_S5_l3_3508


namespace positive_value_of_A_l3_3493

def my_relation (A B k : ℝ) : ℝ := A^2 + k * B^2

theorem positive_value_of_A (A : ℝ) (h1 : ∀ A B, my_relation A B 3 = A^2 + 3 * B^2) (h2 : my_relation A 7 3 = 196) :
  A = 7 := by
  sorry

end positive_value_of_A_l3_3493


namespace original_weight_of_apples_l3_3925

theorem original_weight_of_apples (x : ℕ) (h1 : 5 * (x - 30) = 2 * x) : x = 50 :=
by
  sorry

end original_weight_of_apples_l3_3925


namespace base_eight_to_base_ten_l3_3893

theorem base_eight_to_base_ten {d1 d2 d3 : ℕ} (h1 : d1 = 1) (h2 : d2 = 5) (h3 : d3 = 7) :
  d3 * 8^0 + d2 * 8^1 + d1 * 8^2 = 111 := 
by
  sorry

end base_eight_to_base_ten_l3_3893


namespace cones_slant_height_angle_l3_3861

theorem cones_slant_height_angle :
  ∀ (α: ℝ),
  α = 2 * Real.arccos (Real.sqrt (2 / (2 + Real.sqrt 2))) :=
by
  sorry

end cones_slant_height_angle_l3_3861


namespace geometric_sequence_an_l3_3923

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 3 else 3 * (2:ℝ)^(n - 1)

noncomputable def S (n : ℕ) : ℝ :=
  if n = 1 then 3 else (3 * (2:ℝ)^n - 3)

theorem geometric_sequence_an (n : ℕ) (h1 : a 1 = 3) (h2 : S 2 = 9) :
  a n = 3 * 2^(n-1) ∧ S n = 3 * (2^n - 1) :=
by
  sorry

end geometric_sequence_an_l3_3923


namespace age_difference_l3_3302

variable (A B C D : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 16) : (A + B) - (B + C) = 16 :=
by
  sorry

end age_difference_l3_3302


namespace range_of_a_l3_3719

theorem range_of_a (x y a : ℝ) :
  (2 * x + y ≥ 4) → 
  (x - y ≥ 1) → 
  (x - 2 * y ≤ 2) → 
  (x = 2) → 
  (y = 0) → 
  (z = a * x + y) → 
  (Ax = 2) → 
  (Ay = 0) → 
  (-1/2 < a ∧ a < 2) := sorry

end range_of_a_l3_3719


namespace clothing_prices_and_purchase_plans_l3_3702

theorem clothing_prices_and_purchase_plans :
  ∃ (x y : ℕ) (a : ℤ), 
  x + y = 220 ∧
  6 * x = 5 * y ∧
  120 * a + 100 * (150 - a) ≤ 17000 ∧
  (90 ≤ a ∧ a ≤ 100) ∧
  x = 100 ∧
  y = 120 ∧
  (∀ b : ℤ, (90 ≤ b ∧ b ≤ 100) → 120 * b + 100 * (150 - b) ≥ 16800)
  :=
sorry

end clothing_prices_and_purchase_plans_l3_3702


namespace coat_price_proof_l3_3449

variable (W : ℝ) -- wholesale price
variable (currentPrice : ℝ) -- current price of the coat

-- Condition 1: The retailer marked up the coat by 90%.
def markup_90 : Prop := currentPrice = 1.9 * W

-- Condition 2: Further $4 increase achieves a 100% markup.
def increase_4 : Prop := 2 * W - currentPrice = 4

-- Theorem: The current price of the coat is $76.
theorem coat_price_proof (h1 : markup_90 W currentPrice) (h2 : increase_4 W currentPrice) : currentPrice = 76 :=
sorry

end coat_price_proof_l3_3449


namespace points_on_ellipse_l3_3540

-- Definitions of the conditions
def ellipse (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ x^2 / a^2 + y^2 / b^2 = 1

def passes_through_point (a b : ℝ) : Prop :=
  ellipse a b 2 1

-- Target set of points
def target_set (x y : ℝ) : Prop :=
  x^2 + y^2 < 5 ∧ |y| > 1

-- Main theorem to prove
theorem points_on_ellipse (a b x y : ℝ) (h₁ : passes_through_point a b) (h₂ : |y| > 1) :
  ellipse a b x y → target_set x y :=
sorry

end points_on_ellipse_l3_3540


namespace frequency_of_sixth_group_l3_3880

theorem frequency_of_sixth_group :
  ∀ (total_data_points : ℕ)
    (freq1 freq2 freq3 freq4 : ℕ)
    (freq5_ratio : ℝ),
    total_data_points = 40 →
    freq1 = 10 →
    freq2 = 5 →
    freq3 = 7 →
    freq4 = 6 →
    freq5_ratio = 0.10 →
    (total_data_points - (freq1 + freq2 + freq3 + freq4) - (total_data_points * freq5_ratio)) = 8 :=
by
  sorry

end frequency_of_sixth_group_l3_3880


namespace baby_guppies_calculation_l3_3150

-- Define the problem in Lean
theorem baby_guppies_calculation :
  ∀ (initial_guppies first_sighting two_days_gups total_guppies_after_two_days : ℕ), 
  initial_guppies = 7 →
  first_sighting = 36 →
  total_guppies_after_two_days = 52 →
  total_guppies_after_two_days = initial_guppies + first_sighting + two_days_gups →
  two_days_gups = 9 :=
by
  intros initial_guppies first_sighting two_days_gups total_guppies_after_two_days
  intros h_initial h_first h_total h_eq
  sorry

end baby_guppies_calculation_l3_3150


namespace solve_problem_l3_3186

noncomputable def problem_statement : Prop :=
  (2015 : ℝ) / (2015^2 - 2016 * 2014) = 2015

theorem solve_problem : problem_statement := by
  -- Proof steps will be filled in here.
  sorry

end solve_problem_l3_3186


namespace derivatives_at_zero_l3_3078

noncomputable def f : ℝ → ℝ := sorry

axiom diff_f : ∀ n : ℕ, f (1 / (n + 1)) = (n + 1)^2 / ((n + 1)^2 + 1)

theorem derivatives_at_zero :
  f 0 = 1 ∧ 
  deriv f 0 = 0 ∧ 
  deriv (deriv f) 0 = -2 ∧ 
  ∀ k : ℕ, k ≥ 3 → deriv^[k] f 0 = 0 :=
by
  sorry

end derivatives_at_zero_l3_3078


namespace area_of_inscribed_square_in_ellipse_l3_3267

open Real

noncomputable def inscribed_square_area : ℝ := 32

theorem area_of_inscribed_square_in_ellipse :
  ∀ (x y : ℝ),
  (x^2 / 4 + y^2 / 8 = 1) →
  (x = t - t) ∧ (y = (t + t) / sqrt 2) ∧ 
  (t = sqrt 4) → inscribed_square_area = 32 :=
  sorry

end area_of_inscribed_square_in_ellipse_l3_3267


namespace expected_sufferers_l3_3321

theorem expected_sufferers 
  (fraction_condition : ℚ := 1 / 4)
  (sample_size : ℕ := 400) 
  (expected_number : ℕ := 100) : 
  fraction_condition * sample_size = expected_number := 
by 
  sorry

end expected_sufferers_l3_3321


namespace relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l3_3780

theorem relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the sufficiency part
theorem sufficiency_x_lt_1 (x : ℝ) :
  (x < 1) → (x^2 - 4 * x + 3 > 0) :=
by sorry

-- Define the necessity part
theorem necessity_x_lt_1 (x : ℝ) :
  (x^2 - 4 * x + 3 > 0) → (x < 1 ∨ x > 3) :=
by sorry

end relation_x_lt_1_and_x_sq_sub_4x_add_3_gt_0_sufficiency_x_lt_1_necessity_x_lt_1_l3_3780


namespace no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l3_3386
-- Bringing in the entirety of Mathlib

-- Problem (a): There are no non-zero integers that increase by 7 or 9 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_7_or_9 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ (10 * X + d = 7 * n ∨ 10 * X + d = 9 * n)) :=
by sorry

-- Problem (b): There are no non-zero integers that increase by 4 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_4 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ 10 * X + d = 4 * n) :=
by sorry

end no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l3_3386


namespace quadrilateral_equality_l3_3545

-- Variables definitions for points and necessary properties
variables {A B C D : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Assumptions based on given conditions
variables (AB : ℝ) (AD : ℝ) (BC : ℝ) (DC : ℝ) (beta : ℝ)
variables {angleB : ℝ} {angleD : ℝ}

-- Given conditions
axiom AB_eq_AD : AB = AD
axiom angleB_eq_angleD : angleB = angleD

-- The statement to be proven
theorem quadrilateral_equality (h1 : AB = AD) (h2 : angleB = angleD) : BC = DC :=
by
  sorry

end quadrilateral_equality_l3_3545


namespace max_integer_value_l3_3364

theorem max_integer_value (x : ℝ) : ∃ (m : ℤ), m = 53 ∧ ∀ y : ℝ, (1 + 13 / (3 * y^2 + 9 * y + 7) ≤ m) := 
sorry

end max_integer_value_l3_3364


namespace how_many_oxen_c_put_l3_3509

variables (oxen_a oxen_b months_a months_b rent total_rent c_share x : ℕ)
variable (H : 10 * 7 = oxen_a)
variable (H1 : 12 * 5 = oxen_b)
variable (H2 : 3 * x = months_a)
variable (H3 : 70 + 60 + 3 * x = months_b)
variable (H4 : 280 = total_rent)
variable (H5 : 72 = c_share)

theorem how_many_oxen_c_put : x = 15 :=
  sorry

end how_many_oxen_c_put_l3_3509


namespace median_eq_range_le_l3_3476

variables (x₁ x₂ x₃ x₄ x₅ x₆ : ℝ)
-- x₁ is the minimum value
-- x₆ is the maximum value
-- Assume x₁ ≤ x₂ ≤ x₃ ≤ x₄ ≤ x₅ ≤ x₆

theorem median_eq : (x₃ + x₄) / 2 = (x₃ + x₄) / 2 := 
by sorry

theorem range_le : (x₅ - x₂) ≤ (x₆ - x₁) := 
by sorry

end median_eq_range_le_l3_3476


namespace part_i_part_ii_l3_3551

-- Define the function f
def f (x a : ℝ) : ℝ := abs (x - 2 * a) + abs (x - a)

-- Part I: Prove solution to the inequality.
theorem part_i (x : ℝ) : f x 1 > 3 ↔ x ∈ {x | x < 0} ∪ {x | x > 3} :=
sorry

-- Part II: Prove the inequality for general a and b with condition for equality.
theorem part_ii (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  f b a ≥ f a a ∧ ((2 * a - b = 0 ∨ b - a = 0) ∨ (2 * a - b > 0 ∧ b - a > 0) ∨ (2 * a - b < 0 ∧ b - a < 0)) ↔ f b a = f a a :=
sorry

end part_i_part_ii_l3_3551


namespace triangle_area_is_9_l3_3108

-- Define the vertices of the triangle
def x1 : ℝ := 1
def y1 : ℝ := 2
def x2 : ℝ := 4
def y2 : ℝ := 5
def x3 : ℝ := 6
def y3 : ℝ := 1

-- Define the area calculation formula for the triangle
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The proof statement
theorem triangle_area_is_9 :
  triangle_area x1 y1 x2 y2 x3 y3 = 9 :=
by
  sorry

end triangle_area_is_9_l3_3108


namespace min_people_wearing_both_hat_and_glove_l3_3043

theorem min_people_wearing_both_hat_and_glove (n : ℕ) (x : ℕ) 
  (h1 : 2 * n = 5 * (8 : ℕ)) -- 2/5 of n people wear gloves
  (h2 : 3 * n = 4 * (15 : ℕ)) -- 3/4 of n people wear hats
  (h3 : n = 20): -- total number of people is 20
  x = 3 := -- minimum number of people wearing both a hat and a glove is 3
by sorry

end min_people_wearing_both_hat_and_glove_l3_3043


namespace red_balls_count_l3_3542

theorem red_balls_count (y : ℕ) (p_yellow : ℚ) (h1 : y = 10)
  (h2 : p_yellow = 5/8) (total_balls_le : ∀ r : ℕ, y + r ≤ 32) :
  ∃ r : ℕ, 10 + r > 0 ∧ p_yellow = 10 / (10 + r) ∧ r = 6 :=
by
  sorry

end red_balls_count_l3_3542


namespace area_of_triangle_XYZ_l3_3051

noncomputable def centroid (p1 p2 p3 : (ℚ × ℚ)) : (ℚ × ℚ) :=
((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

noncomputable def triangle_area (p1 p2 p3 : (ℚ × ℚ)) : ℚ :=
abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2 - p1.2 * p2.1 - p2.2 * p3.1 - p3.2 * p1.1) / 2)

noncomputable def point_A : (ℚ × ℚ) := (5, 12)
noncomputable def point_B : (ℚ × ℚ) := (0, 0)
noncomputable def point_C : (ℚ × ℚ) := (14, 0)

noncomputable def point_X : (ℚ × ℚ) :=
(109 / 13, 60 / 13)
noncomputable def point_Y : (ℚ × ℚ) :=
centroid point_A point_B point_X
noncomputable def point_Z : (ℚ × ℚ) :=
centroid point_B point_C point_Y

theorem area_of_triangle_XYZ : triangle_area point_X point_Y point_Z = 84 / 13 :=
sorry

end area_of_triangle_XYZ_l3_3051


namespace two_positive_roots_condition_l3_3342

theorem two_positive_roots_condition (a : ℝ) :
  (1 < a ∧ a ≤ 2) ∨ (a ≥ 10) ↔
  ∃ x1 x2 : ℝ, (1-a) * x1^2 + (a+2) * x1 - 4 = 0 ∧ 
               (1-a) * x2^2 + (a+2) * x2 - 4 = 0 ∧ 
               x1 > 0 ∧ x2 > 0 :=
sorry

end two_positive_roots_condition_l3_3342


namespace lengths_of_trains_l3_3218

noncomputable def km_per_hour_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def length_of_train (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem lengths_of_trains (Va Vb : ℝ) : Va = 60 ∧ Vb < Va ∧ length_of_train (km_per_hour_to_m_per_s Va) 42 = (700 : ℝ) 
    → length_of_train (km_per_hour_to_m_per_s Vb * (42 / 56)) 56 = (700 : ℝ) :=
by
  intros h
  sorry

end lengths_of_trains_l3_3218


namespace starting_number_of_range_l3_3144

theorem starting_number_of_range (multiples: ℕ) (end_of_range: ℕ) (span: ℕ)
  (h1: multiples = 991) (h2: end_of_range = 10000) (h3: span = multiples * 10) :
  end_of_range - span = 90 := 
by 
  sorry

end starting_number_of_range_l3_3144


namespace total_tickets_sold_l3_3440

/-
Problem: Prove that the total number of tickets sold is 65 given the conditions.
Conditions:
1. Senior citizen tickets cost 10 dollars each.
2. Regular tickets cost 15 dollars each.
3. Total sales were 855 dollars.
4. 24 senior citizen tickets were sold.
-/

def senior_tickets_sold : ℕ := 24
def senior_ticket_cost : ℕ := 10
def regular_ticket_cost : ℕ := 15
def total_sales : ℕ := 855

theorem total_tickets_sold (R : ℕ) (H : total_sales = senior_tickets_sold * senior_ticket_cost + R * regular_ticket_cost) :
  senior_tickets_sold + R = 65 :=
by
  sorry

end total_tickets_sold_l3_3440


namespace probability_of_red_ball_l3_3823

theorem probability_of_red_ball :
  let total_balls := 9
  let red_balls := 6
  let probability := (red_balls : ℚ) / total_balls
  probability = (2 : ℚ) / 3 :=
by
  sorry

end probability_of_red_ball_l3_3823


namespace change_received_l3_3323

theorem change_received (cost_cat_toy : ℝ) (cost_cage : ℝ) (total_paid : ℝ) (change : ℝ) :
  cost_cat_toy = 8.77 →
  cost_cage = 10.97 →
  total_paid = 20.00 →
  change = 0.26 →
  total_paid - (cost_cat_toy + cost_cage) = change := by
sorry

end change_received_l3_3323


namespace tangent_segment_length_l3_3199

-- Setting up the necessary definitions and theorem.
def radius := 10
def seg1 := 4
def seg2 := 2

theorem tangent_segment_length :
  ∃ X : ℝ, X = 8 ∧
  (radius^2 = X^2 + ((X + seg1 + seg2) / 2)^2) :=
by
  sorry

end tangent_segment_length_l3_3199


namespace calculate_initial_budget_l3_3855

-- Definitions based on conditions
def cost_of_chicken := 12
def cost_per_pound_beef := 3
def pounds_of_beef := 5
def amount_left := 53

-- Derived definition for total cost of beef
def cost_of_beef := cost_per_pound_beef * pounds_of_beef
-- Derived definition for total spent
def total_spent := cost_of_chicken + cost_of_beef
-- Final calculation for initial budget
def initial_budget := total_spent + amount_left

-- Statement to prove
theorem calculate_initial_budget : initial_budget = 80 :=
by
  sorry

end calculate_initial_budget_l3_3855


namespace delegate_arrangement_probability_l3_3034

theorem delegate_arrangement_probability :
  let delegates := 10
  let countries := 3
  let independent_delegate := 1
  let total_seats := 10
  let m := 379
  let n := 420
  delegates = 10 ∧ countries = 3 ∧ independent_delegate = 1 ∧ total_seats = 10 →
  Nat.gcd m n = 1 →
  m + n = 799 :=
by
  sorry

end delegate_arrangement_probability_l3_3034


namespace fill_bathtub_time_l3_3207

theorem fill_bathtub_time (V : ℝ) (cold_rate hot_rate drain_rate net_rate : ℝ) 
  (hcold : cold_rate = V / 10) 
  (hhot : hot_rate = V / 15) 
  (hdrain : drain_rate = -V / 12) 
  (hnet : net_rate = cold_rate + hot_rate + drain_rate) 
  (V_eq : V = 1) : 
  1 / net_rate = 12 :=
by {
  -- placeholder for proof steps
  sorry
}

end fill_bathtub_time_l3_3207


namespace number_of_pairs_x_y_l3_3706

theorem number_of_pairs_x_y (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x^2 - y^2 = 85) : 
    (1 : ℕ) + (1 : ℕ) = 2 := 
by 
  sorry

end number_of_pairs_x_y_l3_3706


namespace expected_coins_basilio_per_day_l3_3284

/-- The expected number of gold coins received by Basilio per day is 5.25 -/
def expected_coins_received_by_basilio (n : ℕ) (p : ℚ) : ℚ :=
  if n = 20 ∧ p = (1 / 2 : ℚ) then 5.25 else 0

theorem expected_coins_basilio_per_day :
  expected_coins_received_by_basilio 20 (1 / 2) = 5.25 :=
by {
  -- proof goes here
  sorry
}

end expected_coins_basilio_per_day_l3_3284


namespace ted_age_proof_l3_3479

theorem ted_age_proof (s t : ℝ) (h1 : t = 3 * s - 20) (h2 : t + s = 78) : t = 53.5 :=
by
  sorry  -- Proof steps are not required, hence using sorry.

end ted_age_proof_l3_3479


namespace x_minus_p_eq_2_minus_2p_l3_3273

theorem x_minus_p_eq_2_minus_2p (x p : ℝ) (h1 : |x - 3| = p + 1) (h2 : x < 3) : x - p = 2 - 2 * p := 
sorry

end x_minus_p_eq_2_minus_2p_l3_3273


namespace convert_speed_kmh_to_ms_l3_3620

-- Define the given speed in km/h
def speed_kmh : ℝ := 1.1076923076923078

-- Define the conversion factor from km/h to m/s
def conversion_factor : ℝ := 3.6

-- State the theorem
theorem convert_speed_kmh_to_ms (s : ℝ) (h : s = speed_kmh) : (s / conversion_factor) = 0.3076923076923077 := by
  -- Skip the proof as instructed
  sorry

end convert_speed_kmh_to_ms_l3_3620


namespace value_of_y_l3_3552

theorem value_of_y (y : ℕ) (h : 9 / (y^2) = y / 81) : y = 9 :=
by
-- Since we are only required to state the theorem, we leave the proof out for now.
sorry

end value_of_y_l3_3552


namespace additional_discount_during_sale_l3_3958

theorem additional_discount_during_sale:
  ∀ (list_price : ℝ) (max_typical_discount_pct : ℝ) (lowest_possible_sale_pct : ℝ),
  30 ≤ max_typical_discount_pct ∧ max_typical_discount_pct ≤ 50 ∧
  lowest_possible_sale_pct = 40 ∧ 
  list_price = 80 →
  ((max_typical_discount_pct * list_price / 100) - (lowest_possible_sale_pct * list_price / 100)) * 100 / 
    (max_typical_discount_pct * list_price / 100) = 20 :=
by
  sorry

end additional_discount_during_sale_l3_3958


namespace values_of_a_l3_3683

open Set

noncomputable def A : Set ℝ := { x | x^2 - 2*x - 3 = 0 }
noncomputable def B (a : ℝ) : Set ℝ := if a = 0 then ∅ else { x | a * x = 1 }

theorem values_of_a (a : ℝ) : (B a ⊆ A) ↔ (a = -1 ∨ a = 0 ∨ a = 1/3) :=
by 
  sorry

end values_of_a_l3_3683


namespace sum_of_first_six_terms_geometric_sequence_l3_3106

theorem sum_of_first_six_terms_geometric_sequence :
  let a := (1 : ℚ) / 4
  let r := (1 : ℚ) / 4
  let n := 6
  let S_n := a * (1 - r ^ n) / (1 - r)
  S_n = 1365 / 4096 := by
  sorry

end sum_of_first_six_terms_geometric_sequence_l3_3106


namespace max_value_m_l3_3862

theorem max_value_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 2 * x - 8 > 0) -> (x < m)) -> m = -2 :=
by
  sorry

end max_value_m_l3_3862


namespace num_int_solutions_l3_3122

theorem num_int_solutions (x : ℤ) : 
  (x^4 - 39 * x^2 + 140 < 0) ↔ (x = 3 ∨ x = -3 ∨ x = 4 ∨ x = -4 ∨ x = 5 ∨ x = -5) := 
sorry

end num_int_solutions_l3_3122


namespace product_of_three_integers_sum_l3_3624
-- Import necessary libraries

-- Define the necessary conditions and the goal
theorem product_of_three_integers_sum (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
(h4 : a * b * c = 11^3) : a + b + c = 133 :=
sorry

end product_of_three_integers_sum_l3_3624


namespace number_of_integer_terms_l3_3178

noncomputable def count_integer_terms_in_sequence (n : ℕ) (k : ℕ) (a : ℕ) : ℕ :=
  if h : a = k * 3 ^ n then n + 1 else 0

theorem number_of_integer_terms :
  count_integer_terms_in_sequence 5 (2^3 * 5) 9720 = 6 :=
by sorry

end number_of_integer_terms_l3_3178


namespace total_chewing_gums_l3_3347

-- Definitions for the conditions
def mary_gums : Nat := 5
def sam_gums : Nat := 10
def sue_gums : Nat := 15

-- Lean 4 Theorem statement to prove the total chewing gums
theorem total_chewing_gums : mary_gums + sam_gums + sue_gums = 30 := by
  sorry

end total_chewing_gums_l3_3347


namespace cory_chairs_l3_3674

theorem cory_chairs (total_cost table_cost chair_cost C : ℕ) (h1 : total_cost = 135) (h2 : table_cost = 55) (h3 : chair_cost = 20) (h4 : total_cost = table_cost + chair_cost * C) : C = 4 := 
by 
  sorry

end cory_chairs_l3_3674


namespace prob_equals_two_yellow_marbles_l3_3099

noncomputable def probability_two_yellow_marbles : ℚ :=
  let total_marbles : ℕ := 3 + 4 + 8
  let yellow_marbles : ℕ := 4
  let first_draw_prob : ℚ := yellow_marbles / total_marbles
  let second_total_marbles : ℕ := total_marbles - 1
  let second_yellow_marbles : ℕ := yellow_marbles - 1
  let second_draw_prob : ℚ := second_yellow_marbles / second_total_marbles
  first_draw_prob * second_draw_prob

theorem prob_equals_two_yellow_marbles :
  probability_two_yellow_marbles = 2 / 35 :=
by
  sorry

end prob_equals_two_yellow_marbles_l3_3099


namespace symm_central_origin_l3_3077

noncomputable def f₁ (x : ℝ) : ℝ := 3^x

noncomputable def f₂ (x : ℝ) : ℝ := -3^(-x)

theorem symm_central_origin :
  ∀ x : ℝ, ∃ x' y y' : ℝ, (f₁ x = y) ∧ (f₂ x' = y') ∧ (x' = -x) ∧ (y' = -y) :=
by
  sorry

end symm_central_origin_l3_3077


namespace travel_time_l3_3986

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end travel_time_l3_3986


namespace train_speed_and_length_l3_3143

theorem train_speed_and_length 
  (x y : ℝ)
  (h1 : 60 * x = 1000 + y)
  (h2 : 40 * x = 1000 - y) :
  x = 20 ∧ y = 200 :=
by
  sorry

end train_speed_and_length_l3_3143


namespace gcd_pow_minus_one_l3_3921

theorem gcd_pow_minus_one (m n : ℕ) (hm : 0 < m) (hn : 0 < n) : 
  Nat.gcd (2^m - 1) (2^n - 1) = 2^(Nat.gcd m n) - 1 :=
by
  sorry

end gcd_pow_minus_one_l3_3921


namespace find_c_l3_3989

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x, g_inv (g x c) = x) ↔ c = 3 / 2 := by
  sorry

end find_c_l3_3989


namespace fill_cistern_7_2_hours_l3_3772

theorem fill_cistern_7_2_hours :
  let R_fill := 1 / 4
  let R_empty := 1 / 9
  R_fill - R_empty = 5 / 36 →
  1 / (R_fill - R_empty) = 7.2 := 
by
  intros
  sorry

end fill_cistern_7_2_hours_l3_3772


namespace max_k_range_minus_five_l3_3123

theorem max_k_range_minus_five :
  ∃ k : ℝ, (∀ x : ℝ, x^2 + 5 * x + k = -5) → k = 5 / 4 :=
by
  sorry

end max_k_range_minus_five_l3_3123


namespace equilateral_triangle_l3_3859

theorem equilateral_triangle
  (a b c : ℝ) (α β γ : ℝ)
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : 0 < α ∧ α < π)
  (h5 : 0 < β ∧ β < π)
  (h6 : 0 < γ ∧ γ < π)
  (h7 : α + β + γ = π)
  (h8 : a * (1 - 2 * Real.cos α) + b * (1 - 2 * Real.cos β) + c * (1 - 2 * Real.cos γ) = 0) :
  α = β ∧ β = γ ∧ γ = α :=
by
  sorry

end equilateral_triangle_l3_3859


namespace part1_even_function_part2_min_value_l3_3625

variable {a x : ℝ}

def f (x a : ℝ) : ℝ := x^2 + |x - a| + 1

theorem part1_even_function (h : a = 0) : 
  ∀ x : ℝ, f x 0 = f (-x) 0 :=
by
  -- This statement needs to be proved to show that f(x) is even when a = 0
  sorry

theorem part2_min_value (h : true) : 
  (a > (1/2) → ∃ x : ℝ, f x a = a + (3/4)) ∧
  (a ≤ -(1/2) → ∃ x : ℝ, f x a = -a + (3/4)) ∧
  ((- (1/2) < a ∧ a ≤ (1/2)) → ∃ x : ℝ, f x a = a^2 + 1) :=
by
  -- This statement needs to be proved to show the different minimum values of the function
  sorry

end part1_even_function_part2_min_value_l3_3625


namespace books_per_shelf_l3_3343

theorem books_per_shelf :
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  remaining_books / shelves = 3 :=
by
  let total_books := 14
  let taken_books := 2
  let shelves := 4
  let remaining_books := total_books - taken_books
  have h1 : remaining_books = 12 := by simp [remaining_books]
  have h2 : remaining_books / shelves = 3 := by norm_num [remaining_books, shelves]
  exact h2

end books_per_shelf_l3_3343


namespace value_of_star_l3_3189

theorem value_of_star :
  ∀ x : ℕ, 45 - (28 - (37 - (15 - x))) = 55 → x = 16 :=
by
  intro x
  intro h
  sorry

end value_of_star_l3_3189


namespace infinite_points_in_region_l3_3782

theorem infinite_points_in_region : 
  ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → ¬(∃ n : ℕ, ∀ x y : ℚ, 0 < x → 0 < y → x + 2 * y ≤ 6 → sorry) :=
sorry

end infinite_points_in_region_l3_3782


namespace sally_money_l3_3335

def seashells_monday : ℕ := 30
def seashells_tuesday : ℕ := seashells_monday / 2
def total_seashells : ℕ := seashells_monday + seashells_tuesday
def price_per_seashell : ℝ := 1.20

theorem sally_money : total_seashells * price_per_seashell = 54 :=
by
  sorry

end sally_money_l3_3335


namespace concentration_after_5_days_l3_3084

noncomputable def ozverin_concentration_after_iterations 
    (initial_volume : ℝ) (initial_concentration : ℝ)
    (drunk_volume : ℝ) (iterations : ℕ) : ℝ :=
initial_concentration * (1 - drunk_volume / initial_volume)^iterations

theorem concentration_after_5_days : 
  ozverin_concentration_after_iterations 0.5 0.4 0.05 5 = 0.236 :=
by
  sorry

end concentration_after_5_days_l3_3084


namespace max_coefficient_terms_l3_3460

theorem max_coefficient_terms (x : ℝ) :
  let n := 8
  let T_3 := 7 * x^2
  let T_4 := 7 * x
  true := by
  sorry

end max_coefficient_terms_l3_3460


namespace ceil_floor_subtraction_l3_3322

theorem ceil_floor_subtraction :
  ⌈(7:ℝ) / 3⌉ + ⌊- (7:ℝ) / 3⌋ - 3 = -3 := 
by
  sorry   -- Placeholder for the proof

end ceil_floor_subtraction_l3_3322


namespace evening_temperature_l3_3892

-- Definitions based on conditions
def noon_temperature : ℤ := 2
def temperature_drop : ℤ := 3

-- The theorem statement
theorem evening_temperature : noon_temperature - temperature_drop = -1 := 
by
  -- The proof is omitted
  sorry

end evening_temperature_l3_3892


namespace quilt_patch_cost_is_correct_l3_3111

noncomputable def quilt_area : ℕ := 16 * 20

def patch_area : ℕ := 4

def first_10_patch_cost : ℕ := 10

def discount_patch_cost : ℕ := 5

def total_patches (quilt_area patch_area : ℕ) : ℕ := quilt_area / patch_area

def cost_for_first_10 (first_10_patch_cost : ℕ) : ℕ := 10 * first_10_patch_cost

def cost_for_discounted (total_patches first_10_patch_cost discount_patch_cost : ℕ) : ℕ :=
  (total_patches - 10) * discount_patch_cost

def total_cost (cost_for_first_10 cost_for_discounted : ℕ) : ℕ :=
  cost_for_first_10 + cost_for_discounted

theorem quilt_patch_cost_is_correct :
  total_cost (cost_for_first_10 first_10_patch_cost)
             (cost_for_discounted (total_patches quilt_area patch_area) first_10_patch_cost discount_patch_cost) = 450 :=
by
  sorry

end quilt_patch_cost_is_correct_l3_3111


namespace largest_whole_number_lt_div_l3_3960

theorem largest_whole_number_lt_div {x : ℕ} (hx : 8 * x < 80) : x ≤ 9 :=
by
  sorry

end largest_whole_number_lt_div_l3_3960


namespace eq_implies_sq_eq_l3_3224

theorem eq_implies_sq_eq (a b : ℝ) (h : a = b) : a^2 = b^2 :=
sorry

end eq_implies_sq_eq_l3_3224


namespace find_number_l3_3527

-- Define the problem conditions
def problem_condition (x : ℝ) : Prop := 2 * x - x / 2 = 45

-- Main theorem statement
theorem find_number : ∃ (x : ℝ), problem_condition x ∧ x = 30 :=
by
  existsi 30
  -- Include the problem condition and the solution check
  unfold problem_condition
  -- We are skipping the proof using sorry to just provide the statement
  sorry

end find_number_l3_3527


namespace initial_money_is_correct_l3_3755

-- Given conditions
def spend_per_trip : ℕ := 2
def trips_per_month : ℕ := 4
def months_per_year : ℕ := 12
def money_left_after_year : ℕ := 104

-- Define the initial amount of money
def initial_amount_of_money (spend_per_trip trips_per_month months_per_year money_left_after_year : ℕ) : ℕ :=
  money_left_after_year + (spend_per_trip * trips_per_month * months_per_year)

-- Theorem stating that under the given conditions, the initial amount of money is 200
theorem initial_money_is_correct :
  initial_amount_of_money spend_per_trip trips_per_month months_per_year money_left_after_year = 200 :=
  sorry

end initial_money_is_correct_l3_3755


namespace largest_divisor_of_consecutive_even_product_l3_3031

theorem largest_divisor_of_consecutive_even_product :
  ∀ (n : ℕ), ∃ k : ℤ, k = 24 ∧ 
  (2 * n) * (2 * n + 2) * (2 * n + 4) % k = 0 :=
by
  sorry

end largest_divisor_of_consecutive_even_product_l3_3031


namespace circle_area_from_points_l3_3325

theorem circle_area_from_points (C D : ℝ × ℝ) (hC : C = (2, 3)) (hD : D = (8, 9)) : 
  ∃ A : ℝ, A = 18 * Real.pi :=
by
  sorry

end circle_area_from_points_l3_3325


namespace vector_parallel_l3_3018

theorem vector_parallel (x : ℝ) :
  let a : ℝ × ℝ := (2 * x + 1, 4)
  let b : ℝ × ℝ := (2 - x, 3)
  (3 * (2 * x + 1) - 4 * (2 - x) = 0) → (x = 1 / 2) :=
by
  intros a b h
  sorry

end vector_parallel_l3_3018


namespace pregnant_fish_in_each_tank_l3_3050

/-- Mark has 3 tanks for pregnant fish. Each tank has a certain number of pregnant fish and each fish
gives birth to 20 young. Mark has 240 young fish at the end. Prove that there are 4 pregnant fish in
each tank. -/
theorem pregnant_fish_in_each_tank (x : ℕ) (h1 : 3 * 20 * x = 240) : x = 4 := by
  sorry

end pregnant_fish_in_each_tank_l3_3050


namespace horner_multiplications_additions_l3_3601

-- Define the polynomial
def f (x : ℤ) : ℤ := x^7 + 2 * x^5 + 3 * x^4 + 4 * x^3 + 5 * x^2 + 6 * x + 7

-- Define the number of multiplications and additions required by Horner's method
def horner_method_mults (n : ℕ) : ℕ := n
def horner_method_adds (n : ℕ) : ℕ := n - 1

-- Define the value of x
def x : ℤ := 3

-- Define the degree of the polynomial
def degree_of_polynomial : ℕ := 7

-- Define the statements for the proof
theorem horner_multiplications_additions :
  horner_method_mults degree_of_polynomial = 7 ∧
  horner_method_adds degree_of_polynomial = 6 :=
by
  sorry

end horner_multiplications_additions_l3_3601


namespace inequality_holds_for_all_real_l3_3506

open Real -- Open the real numbers namespace

theorem inequality_holds_for_all_real (x : ℝ) : 
  2^((sin x)^2) + 2^((cos x)^2) ≥ 2 * sqrt 2 :=
by
  sorry

end inequality_holds_for_all_real_l3_3506


namespace quadratic_algebraic_expression_l3_3424

theorem quadratic_algebraic_expression (a b : ℝ) (h₁ : a^2 - 3 * a + 1 = 0) (h₂ : b^2 - 3 * b + 1 = 0) :
    a + b - a * b = 2 := by
  sorry

end quadratic_algebraic_expression_l3_3424


namespace relationship_between_a_b_c_l3_3810

theorem relationship_between_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h₁ : a = (10 ^ 1988 + 1) / (10 ^ 1989 + 1))
  (h₂ : b = (10 ^ 1987 + 1) / (10 ^ 1988 + 1))
  (h₃ : c = (10 ^ 1987 + 9) / (10 ^ 1988 + 9)) :
  a < b ∧ b < c := 
sorry

end relationship_between_a_b_c_l3_3810


namespace perpendicular_lines_l3_3021

theorem perpendicular_lines (a : ℝ) :
  (∀ x y : ℝ, 2 * x + y + 1 = 0) ∧ (∀ x y : ℝ, x + a * y + 3 = 0) ∧ (∀ A1 B1 A2 B2 : ℝ, A1 * A2 + B1 * B2 = 0) →
  a = -2 :=
by
  intros h
  sorry

end perpendicular_lines_l3_3021


namespace min_overlap_percent_l3_3504

theorem min_overlap_percent
  (M S : ℝ)
  (hM : M = 0.9)
  (hS : S = 0.85) :
  ∃ x, x = 0.75 ∧ (M + S - 1 ≤ x ∧ x ≤ min M S ∧ x = M + S - 1) :=
by
  sorry

end min_overlap_percent_l3_3504


namespace isosceles_triangle_largest_angle_l3_3849

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h : A = B) (h₁ : C + A + B = 180) (h₂ : C = 30) : 
  180 - 2 * 30 = 120 :=
by sorry

end isosceles_triangle_largest_angle_l3_3849


namespace distributive_property_example_l3_3307

theorem distributive_property_example :
  (3/4 + 7/12 - 5/9) * (-36) = (3/4) * (-36) + (7/12) * (-36) - (5/9) * (-36) :=
by
  sorry

end distributive_property_example_l3_3307


namespace jake_peaches_calculation_l3_3898

variable (S_p : ℕ) (J_p : ℕ)

-- Given that Steven has 19 peaches
def steven_peaches : ℕ := 19

-- Jake has 12 fewer peaches than Steven
def jake_peaches : ℕ := S_p - 12

theorem jake_peaches_calculation (h1 : S_p = steven_peaches) (h2 : S_p = 19) :
  J_p = jake_peaches := 
by
  sorry

end jake_peaches_calculation_l3_3898


namespace steve_travel_time_l3_3942

theorem steve_travel_time :
  ∀ (d : ℕ) (v_back : ℕ) (v_to : ℕ),
  d = 20 →
  v_back = 10 →
  v_to = v_back / 2 →
  d / v_to + d / v_back = 6 := 
by
  intros d v_back v_to h1 h2 h3
  sorry

end steve_travel_time_l3_3942


namespace new_student_weight_l3_3188

theorem new_student_weight :
  ∀ (W : ℝ) (total_weight_19 : ℝ) (total_weight_20 : ℝ),
    total_weight_19 = 19 * 15 →
    total_weight_20 = 20 * 14.8 →
    total_weight_19 + W = total_weight_20 →
    W = 11 :=
by
  intros W total_weight_19 total_weight_20 h1 h2 h3
  -- Skipping the proof as instructed
  sorry

end new_student_weight_l3_3188


namespace possible_value_of_b_l3_3223

theorem possible_value_of_b (a b : ℕ) (H1 : b ∣ (5 * a - 1)) (H2 : b ∣ (a - 10)) (H3 : ¬ b ∣ (3 * a + 5)) : 
  b = 49 :=
sorry

end possible_value_of_b_l3_3223


namespace greatest_q_minus_r_l3_3221

theorem greatest_q_minus_r :
  ∃ q r : ℤ, q > 0 ∧ r > 0 ∧ 975 = 23 * q + r ∧ q - r = 33 := sorry

end greatest_q_minus_r_l3_3221


namespace trigonometric_identity_l3_3422

theorem trigonometric_identity : (1 / 4) * Real.sin (15 * Real.pi / 180) * Real.cos (15 * Real.pi / 180) = 1 / 16 := by
  sorry

end trigonometric_identity_l3_3422


namespace lorelei_roses_l3_3232

theorem lorelei_roses :
  let red_flowers := 12
  let pink_flowers := 18
  let yellow_flowers := 20
  let orange_flowers := 8
  let lorelei_red := (50 / 100) * red_flowers
  let lorelei_pink := (50 / 100) * pink_flowers
  let lorelei_yellow := (25 / 100) * yellow_flowers
  let lorelei_orange := (25 / 100) * orange_flowers
  lorelei_red + lorelei_pink + lorelei_yellow + lorelei_orange = 22 :=
by
  sorry

end lorelei_roses_l3_3232


namespace solve_quadratic_inequality_l3_3646

theorem solve_quadratic_inequality (x : ℝ) :
  (-3 * x^2 + 8 * x + 5 > 0) ↔ (x < -1 / 3) :=
by
  sorry

end solve_quadratic_inequality_l3_3646


namespace johns_new_weekly_earnings_l3_3140

-- Definition of the initial weekly earnings
def initial_weekly_earnings := 40

-- Definition of the percent increase in earnings
def percent_increase := 100

-- Definition for the final weekly earnings after the raise
def final_weekly_earnings (initial_earnings : Nat) (percentage : Nat) := 
  initial_earnings + (initial_earnings * percentage / 100)

-- Theorem stating John’s final weekly earnings after the raise
theorem johns_new_weekly_earnings : final_weekly_earnings initial_weekly_earnings percent_increase = 80 :=
  by
  sorry

end johns_new_weekly_earnings_l3_3140


namespace value_divided_by_is_three_l3_3518

theorem value_divided_by_is_three (x : ℝ) (h : 72 / x = 24) : x = 3 := 
by
  sorry

end value_divided_by_is_three_l3_3518


namespace inequality_proof_l3_3312

open Real

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  1 / (a ^ 3 * (b + c)) + 1 / (b ^ 3 * (a + c)) + 1 / (c ^ 3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l3_3312


namespace domain_ln_x_minus_x_sq_l3_3716

noncomputable def f (x : ℝ) : ℝ := Real.log (x - x^2)

theorem domain_ln_x_minus_x_sq : { x : ℝ | x - x^2 > 0 } = { x : ℝ | 0 < x ∧ x < 1 } :=
by {
  -- These are placeholders for conditions needed in the proof
  sorry
}

end domain_ln_x_minus_x_sq_l3_3716


namespace geometric_sequence_fourth_term_l3_3637

theorem geometric_sequence_fourth_term :
  let a₁ := 3^(3/4)
  let a₂ := 3^(2/4)
  let a₃ := 3^(1/4)
  ∃ a₄, a₄ = 1 ∧ a₂ = a₁ * (a₃ / a₂) ∧ a₃ = a₂ * (a₄ / a₃) :=
by
  sorry

end geometric_sequence_fourth_term_l3_3637


namespace triangle_is_right_angled_l3_3573

noncomputable def median (a b c : ℝ) : ℝ := (1 / 2) * (Real.sqrt (2 * b^2 + 2 * c^2 - a^2))

theorem triangle_is_right_angled (a b c : ℝ) (ha : median a b c = 5) (hb : median b c a = Real.sqrt 52) (hc : median c a b = Real.sqrt 73) :
  a^2 = b^2 + c^2 :=
sorry

end triangle_is_right_angled_l3_3573


namespace pencils_left_l3_3499

def ashton_boxes : Nat := 3
def pencils_per_box : Nat := 14
def pencils_given_to_brother : Nat := 6
def pencils_given_to_friends : Nat := 12

theorem pencils_left (h₁ : ashton_boxes = 3) 
                     (h₂ : pencils_per_box = 14)
                     (h₃ : pencils_given_to_brother = 6)
                     (h₄ : pencils_given_to_friends = 12) :
  (ashton_boxes * pencils_per_box - pencils_given_to_brother - pencils_given_to_friends) = 24 :=
by
  sorry

end pencils_left_l3_3499


namespace more_people_attended_l3_3549

def saturday_attendance := 80
def monday_attendance := saturday_attendance - 20
def wednesday_attendance := monday_attendance + 50
def friday_attendance := saturday_attendance + monday_attendance
def expected_audience := 350

theorem more_people_attended :
  saturday_attendance + monday_attendance + wednesday_attendance + friday_attendance - expected_audience = 40 :=
by
  sorry

end more_people_attended_l3_3549


namespace committee_count_l3_3576

theorem committee_count (students : Finset ℕ) (Alice : ℕ) (hAlice : Alice ∈ students) (hCard : students.card = 7) :
  ∃ committees : Finset (Finset ℕ), (∀ c ∈ committees, Alice ∈ c ∧ c.card = 4) ∧ committees.card = 20 :=
sorry

end committee_count_l3_3576


namespace distinct_paths_from_C_to_D_l3_3033

-- Definitions based on conditions
def grid_rows : ℕ := 7
def grid_columns : ℕ := 8
def total_steps : ℕ := grid_rows + grid_columns -- 15 in this case
def steps_right : ℕ := grid_columns -- 8 in this case

-- Theorem statement
theorem distinct_paths_from_C_to_D :
  Nat.choose total_steps steps_right = 6435 :=
by
  -- The proof itself
  sorry

end distinct_paths_from_C_to_D_l3_3033


namespace compute_xy_l3_3633

variable (x y : ℝ)
variable (h1 : x - y = 6)
variable (h2 : x^3 - y^3 = 108)

theorem compute_xy : x * y = 0 := by
  sorry

end compute_xy_l3_3633


namespace find_number_l3_3699

theorem find_number (x : ℝ) (h : 0.60 * x - 40 = 50) : x = 150 := 
by
  sorry

end find_number_l3_3699


namespace triangle_circle_area_relation_l3_3943

theorem triangle_circle_area_relation (A B C : ℝ) (h : 15^2 + 20^2 = 25^2) (A_area_eq : A + B + 150 = C) :
  A + B + 150 = C :=
by
  -- The proof has been omitted.
  sorry

end triangle_circle_area_relation_l3_3943


namespace rectangle_perimeter_l3_3910

-- Definitions based on conditions
def length (w : ℝ) : ℝ := 2 * w
def width (w : ℝ) : ℝ := w
def area (w : ℝ) : ℝ := length w * width w
def perimeter (w : ℝ) : ℝ := 2 * (length w + width w)

-- Problem statement: Prove that the perimeter is 120 cm given area is 800 cm² and length is twice the width
theorem rectangle_perimeter (w : ℝ) (h : area w = 800) : perimeter w = 120 := by
  sorry

end rectangle_perimeter_l3_3910


namespace coprime_exist_m_n_l3_3536

theorem coprime_exist_m_n (a b : ℕ) (h_coprime : Nat.gcd a b = 1) (h_a : a ≥ 1) (h_b : b ≥ 1) :
  ∃ (m n : ℕ), m ≥ 1 ∧ n ≥ 1 ∧ a^m + b^n ≡ 1 [MOD a * b] :=
by
  use Nat.totient b, Nat.totient a
  sorry

end coprime_exist_m_n_l3_3536


namespace cistern_fill_time_l3_3764

theorem cistern_fill_time (A B : ℝ) (hA : A = 1/60) (hB : B = 1/45) : (|A - B|)⁻¹ = 180 := by
  sorry

end cistern_fill_time_l3_3764


namespace set_equality_implies_sum_zero_l3_3501

theorem set_equality_implies_sum_zero
  (x y : ℝ)
  (A : Set ℝ := {x, y, x + y})
  (B : Set ℝ := {0, x^2, x * y}) :
  A = B → x + y = 0 :=
by
  sorry

end set_equality_implies_sum_zero_l3_3501


namespace cost_of_candy_bar_l3_3036

def initial_amount : ℝ := 3.0
def remaining_amount : ℝ := 2.0

theorem cost_of_candy_bar :
  initial_amount - remaining_amount = 1.0 :=
by
  sorry

end cost_of_candy_bar_l3_3036


namespace root_count_sqrt_eq_l3_3467

open Real

theorem root_count_sqrt_eq (x : ℝ) :
  (∀ y, (y = sqrt (7 - 2 * x)) → y = x * y → (∃ x, x = 7 / 2 ∨ x = 1)) ∧
  (7 - 2 * x ≥ 0) →
  ∃ s, s = 1 ∧ (7 - 2 * s = 0) → x = 1 ∨ x = 7 / 2 :=
sorry

end root_count_sqrt_eq_l3_3467


namespace total_pairs_of_shoes_equivalence_l3_3957

variable (Scott Anthony Jim Melissa Tim: ℕ)

theorem total_pairs_of_shoes_equivalence
    (h1 : Scott = 7)
    (h2 : Anthony = 3 * Scott)
    (h3 : Jim = Anthony - 2)
    (h4 : Jim = 2 * Melissa)
    (h5 : Tim = (Anthony + Melissa) / 2):

  Scott + Anthony + Jim + Melissa + Tim = 71 :=
  by
  sorry

end total_pairs_of_shoes_equivalence_l3_3957


namespace log_function_passes_through_point_l3_3126

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  Real.log (x - 1) / Real.log a - 1

theorem log_function_passes_through_point {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -1 :=
by
  -- To complete the proof, one would argue about the properties of logarithms in specific bases.
  sorry

end log_function_passes_through_point_l3_3126


namespace fraction_simplification_l3_3007

theorem fraction_simplification :
  (36 / 19) * (57 / 40) * (95 / 171) = (3 / 2) :=
by
  sorry

end fraction_simplification_l3_3007


namespace sophia_daily_saving_l3_3263

theorem sophia_daily_saving (total_days : ℕ) (total_saving : ℝ) (h1 : total_days = 20) (h2 : total_saving = 0.20) : 
  (total_saving / total_days) = 0.01 :=
by
  sorry

end sophia_daily_saving_l3_3263


namespace cost_price_of_watch_l3_3081

theorem cost_price_of_watch (CP SP_loss SP_gain : ℝ) (h1 : SP_loss = 0.79 * CP)
  (h2 : SP_gain = 1.04 * CP) (h3 : SP_gain - SP_loss = 140) : CP = 560 := by
  sorry

end cost_price_of_watch_l3_3081


namespace number_of_teams_l3_3250

-- Total number of players
def total_players : Nat := 12

-- Number of ways to choose one captain
def ways_to_choose_captain : Nat := total_players

-- Number of remaining players after choosing the captain
def remaining_players : Nat := total_players - 1

-- Number of players needed to form a team (excluding the captain)
def team_size : Nat := 5

-- Number of ways to choose 5 players from the remaining 11
def ways_to_choose_team (n k : Nat) : Nat := Nat.choose n k

-- Total number of different teams
def total_teams : Nat := ways_to_choose_captain * ways_to_choose_team remaining_players team_size

theorem number_of_teams : total_teams = 5544 := by
  sorry

end number_of_teams_l3_3250


namespace problem_statement_l3_3105

theorem problem_statement (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : 
  x^12 - 7 * x^8 + x^4 = 343 :=
sorry

end problem_statement_l3_3105


namespace difference_of_squares_l3_3160

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
by
  sorry

end difference_of_squares_l3_3160


namespace average_income_B_and_C_l3_3362

variables (A_income B_income C_income : ℝ)

noncomputable def average_monthly_income_B_and_C (A_income : ℝ) :=
  (B_income + C_income) / 2

theorem average_income_B_and_C
  (h1 : (A_income + B_income) / 2 = 5050)
  (h2 : (A_income + C_income) / 2 = 5200)
  (h3 : A_income = 4000) :
  average_monthly_income_B_and_C 4000 = 6250 :=
by
  sorry

end average_income_B_and_C_l3_3362


namespace binary_to_decimal_1100_l3_3002

-- Define the binary number 1100
def binary_1100 : ℕ := 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 0 * 2^0

-- State the theorem that we need to prove
theorem binary_to_decimal_1100 : binary_1100 = 12 := by
  rw [binary_1100]
  sorry

end binary_to_decimal_1100_l3_3002


namespace quadratic_one_real_root_l3_3592

theorem quadratic_one_real_root (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h : ∀ x : ℝ, x^2 + 6*m*x - n = 0 → x * x = 0) : n = 9*m^2 := 
by 
  sorry

end quadratic_one_real_root_l3_3592


namespace total_number_of_ways_is_144_l3_3095

def count_ways_to_place_letters_on_grid : Nat :=
  16 * 9

theorem total_number_of_ways_is_144 :
  count_ways_to_place_letters_on_grid = 144 :=
  by
    sorry

end total_number_of_ways_is_144_l3_3095


namespace a_eq_zero_l3_3804

theorem a_eq_zero (a b : ℤ) (h : ∀ n : ℕ, ∃ k : ℤ, 2^n * a + b = k^2) : a = 0 :=
sorry

end a_eq_zero_l3_3804


namespace sandwiches_bought_is_2_l3_3712

-- The given costs and totals
def sandwich_cost : ℝ := 3.49
def soda_cost : ℝ := 0.87
def total_cost : ℝ := 10.46
def sodas_bought : ℕ := 4

-- We need to prove that the number of sandwiches bought, S, is 2
theorem sandwiches_bought_is_2 (S : ℕ) :
  sandwich_cost * S + soda_cost * sodas_bought = total_cost → S = 2 :=
by
  intros h
  sorry

end sandwiches_bought_is_2_l3_3712


namespace cubic_increasing_l3_3456

-- The definition of an increasing function
def increasing_function (f : ℝ → ℝ) := ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2

-- The function y = x^3
def cubic_function (x : ℝ) : ℝ := x^3

-- The statement we want to prove
theorem cubic_increasing : increasing_function cubic_function :=
sorry

end cubic_increasing_l3_3456


namespace sufficient_not_necessary_condition_l3_3235

theorem sufficient_not_necessary_condition (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (x^2 - 1 > 0 → x < -1 ∨ x > 1) :=
by
  sorry

end sufficient_not_necessary_condition_l3_3235


namespace exists_same_color_rectangle_l3_3163

variable (coloring : ℕ × ℕ → Fin 3)

theorem exists_same_color_rectangle :
  (∃ (r1 r2 r3 r4 c1 c2 c3 c4 : ℕ), 
    r1 ≠ r2 ∧ r2 ≠ r3 ∧ r3 ≠ r4 ∧ r1 ≠ r3 ∧ r1 ≠ r4 ∧ r2 ≠ r4 ∧ 
    c1 ≠ c2 ∧ 
    coloring (4, 82) = 4 ∧ 
    coloring (r1, c1) = coloring (r1, c2) ∧ coloring (r1, c2) = coloring (r2, c1) ∧ 
    coloring (r2, c1) = coloring (r2, c2)) :=
sorry

end exists_same_color_rectangle_l3_3163


namespace abc_values_l3_3000

theorem abc_values (a b c : ℝ) 
  (ha : |a| > 1) 
  (hb : |b| > 1) 
  (hc : |c| > 1) 
  (hab : b = a^2 / (2 - a^2)) 
  (hbc : c = b^2 / (2 - b^2)) 
  (hca : a = c^2 / (2 - c^2)) : 
  a + b + c = 6 ∨ a + b + c = -4 ∨ a + b + c = -6 :=
sorry

end abc_values_l3_3000


namespace sine_product_identity_l3_3306

open Real

theorem sine_product_identity :
  sin 12 * sin 36 * sin 54 * sin 72 = 1 / 16 := by
  have h1 : sin 72 = cos 18 := by sorry
  have h2 : sin 54 = cos 36 := by sorry
  have h3 : ∀ θ, sin θ * cos θ = 1 / 2 * sin (2 * θ) := by sorry
  have h4 : ∀ θ, cos (2 * θ) = 2 * cos θ ^ 2 - 1 := by sorry
  have h5 : cos 36 = 1 - 2 * (sin 18) ^ 2 := by sorry
  have h6 : ∀ θ, sin (180 - θ) = sin θ := by sorry
  sorry

end sine_product_identity_l3_3306


namespace number_of_boys_l3_3011

theorem number_of_boys
  (M W B : Nat)
  (total_earnings wages_of_men earnings_of_men : Nat)
  (num_men_eq_women : 5 * M = W)
  (num_men_eq_boys : 5 * M = B)
  (earnings_eq_90 : total_earnings = 90)
  (men_wages_6 : wages_of_men = 6)
  (men_earnings_eq_30 : earnings_of_men = M * wages_of_men) : 
  B = 5 := 
by
  sorry

end number_of_boys_l3_3011


namespace length_of_bridge_l3_3030

theorem length_of_bridge (speed : ℝ) (time_min : ℝ) (length : ℝ)
  (h_speed : speed = 5) (h_time : time_min = 15) :
  length = 1250 :=
sorry

end length_of_bridge_l3_3030


namespace inequality_first_inequality_second_l3_3500

theorem inequality_first (x : ℝ) : 4 * x - 2 < 1 - 2 * x → x < 1 / 2 := 
sorry

theorem inequality_second (x : ℝ) : (3 - 2 * x ≥ x - 6) ∧ ((3 * x + 1) / 2 < 2 * x) → 1 < x ∧ x ≤ 3 :=
sorry

end inequality_first_inequality_second_l3_3500


namespace meryll_questions_l3_3941

/--
Meryll wants to write a total of 35 multiple-choice questions and 15 problem-solving questions. 
She has written \(\frac{2}{5}\) of the multiple-choice questions and \(\frac{1}{3}\) of the problem-solving questions.
We need to prove that she needs to write 31 more questions in total.
-/
theorem meryll_questions : (35 - (2 / 5) * 35) + (15 - (1 / 3) * 15) = 31 := by
  sorry

end meryll_questions_l3_3941


namespace distance_post_office_l3_3405

theorem distance_post_office 
  (D : ℝ)
  (speed_to_post_office : ℝ := 25)
  (speed_back : ℝ := 4)
  (total_time : ℝ := 5 + (48 / 60)) :
  (D / speed_to_post_office + D / speed_back = total_time) → D = 20 :=
by
  sorry

end distance_post_office_l3_3405


namespace common_ratio_of_geometric_sequence_l3_3393

theorem common_ratio_of_geometric_sequence (a_1 a_2 a_3 a_4 q : ℝ)
  (h1 : a_1 * a_2 * a_3 = 27)
  (h2 : a_2 + a_4 = 30)
  (geometric_sequence : a_2 = a_1 * q ∧ a_3 = a_1 * q^2 ∧ a_4 = a_1 * q^3) :
  q = 3 ∨ q = -3 :=
sorry

end common_ratio_of_geometric_sequence_l3_3393


namespace identify_conic_section_is_hyperbola_l3_3657

theorem identify_conic_section_is_hyperbola :
  ∀ x y : ℝ, x^2 - 16 * y^2 - 10 * x + 4 * y + 36 = 0 →
  (∃ a b h c d k : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ h = 0 ∧ (x - c)^2 / a^2 - (y - d)^2 / b^2 = k) :=
by
  sorry

end identify_conic_section_is_hyperbola_l3_3657


namespace largest_visits_l3_3886

theorem largest_visits (stores : ℕ) (total_visits : ℕ) (unique_visitors : ℕ) 
  (visits_two_stores : ℕ) (remaining_visitors : ℕ) : 
  stores = 7 ∧ total_visits = 21 ∧ unique_visitors = 11 ∧ visits_two_stores = 7 ∧ remaining_visitors = (unique_visitors - visits_two_stores) →
  (remaining_visitors * 2 <= total_visits - visits_two_stores * 2) → (∀ v : ℕ, v * unique_visitors = total_visits) →
  (∃ v_max : ℕ, v_max = 4) :=
by
  sorry

end largest_visits_l3_3886


namespace initial_bottles_count_l3_3537

theorem initial_bottles_count
  (players : ℕ)
  (bottles_per_player_first_break : ℕ)
  (bottles_per_player_end_game : ℕ)
  (remaining_bottles : ℕ)
  (total_bottles_taken_first_break : bottles_per_player_first_break * players = 22)
  (total_bottles_taken_end_game : bottles_per_player_end_game * players = 11)
  (total_remaining_bottles : remaining_bottles = 15) :
  players * bottles_per_player_first_break + players * bottles_per_player_end_game + remaining_bottles = 48 :=
by 
  -- skipping the proof
  sorry

end initial_bottles_count_l3_3537


namespace fraction_equiv_ratio_equiv_percentage_equiv_l3_3851

-- Define the problem's components and conditions.
def frac_1 : ℚ := 3 / 5
def frac_2 (a b : ℚ) : Prop := 3 / 5 = a / b
def ratio_1 (a b : ℚ) : Prop := 10 / a = b / 100
def percentage_1 (a b : ℚ) : Prop := (a / b) * 100 = 60

-- Problem statement 1: Fraction equality
theorem fraction_equiv : frac_2 12 20 := 
by sorry

-- Problem statement 2: Ratio equality
theorem ratio_equiv : ratio_1 (50 / 3) 60 := 
by sorry

-- Problem statement 3: Percentage equality
theorem percentage_equiv : percentage_1 60 100 := 
by sorry

end fraction_equiv_ratio_equiv_percentage_equiv_l3_3851


namespace sum_of_first_n_terms_l3_3834

variable (a : ℕ → ℤ) (b : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Given conditions
axiom a_n_arith : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom a_3 : a 3 = -6
axiom a_6 : a 6 = 0
axiom b_1 : b 1 = -8
axiom b_2 : b 2 = a 1 + a 2 + a 3

-- Correct answer to prove
theorem sum_of_first_n_terms : S n = 4 * (1 - 3^n) := sorry

end sum_of_first_n_terms_l3_3834


namespace part1_part2_l3_3194
-- Importing the entire Mathlib library for required definitions

-- Define the sequence a_n with the conditions given in the problem
def a : ℕ → ℚ
| 0       => 1
| (n + 1) => a n / (2 * a n + 1)

-- Prove the given claims
theorem part1 (n : ℕ) : a n = (1 : ℚ) / (2 * n + 1) :=
sorry

def b (n : ℕ) : ℚ := a n * a (n + 1)

-- The sum of the first n terms of the sequence b_n is denoted as T_n
def T : ℕ → ℚ
| 0       => 0
| (n + 1) => T n + b n

-- Prove the given sum
theorem part2 (n : ℕ) : T n = (n : ℚ) / (2 * n + 1) :=
sorry

end part1_part2_l3_3194


namespace two_buckets_have_40_liters_l3_3004

def liters_in_jug := 5
def jugs_in_bucket := 4
def liters_in_bucket := liters_in_jug * jugs_in_bucket
def buckets := 2

theorem two_buckets_have_40_liters :
  buckets * liters_in_bucket = 40 :=
by
  sorry

end two_buckets_have_40_liters_l3_3004


namespace leak_empties_tank_in_8_hours_l3_3317

theorem leak_empties_tank_in_8_hours (capacity : ℕ) (inlet_rate_per_minute : ℕ) (time_with_inlet_open : ℕ) (time_without_inlet_open : ℕ) : 
  capacity = 8640 ∧ inlet_rate_per_minute = 6 ∧ time_with_inlet_open = 12 ∧ time_without_inlet_open = 8 := 
by 
  sorry

end leak_empties_tank_in_8_hours_l3_3317


namespace find_x_l3_3532

theorem find_x :
  (12^3 * 6^3) / x = 864 → x = 432 :=
by
  sorry

end find_x_l3_3532


namespace largest_value_l3_3809

noncomputable def largest_possible_4x_3y (x y : ℝ) : ℝ :=
  4 * x + 3 * y

theorem largest_value (x y : ℝ) :
  x^2 + y^2 = 16 * x + 8 * y + 8 → (∃ x y, largest_possible_4x_3y x y = 9.64) :=
by
  sorry

end largest_value_l3_3809


namespace pq_conditions_l3_3010

theorem pq_conditions (p q : ℝ) (hp : p > 1) (hq : q > 1) (hq_inverse : 1 / p + 1 / q = 1) (hpq : p * q = 9) :
  (p = (9 + 3 * Real.sqrt 5) / 2 ∧ q = (9 - 3 * Real.sqrt 5) / 2) ∨ (p = (9 - 3 * Real.sqrt 5) / 2 ∧ q = (9 + 3 * Real.sqrt 5) / 2) :=
  sorry

end pq_conditions_l3_3010


namespace number_is_28_l3_3437

-- Definitions from conditions in part a
def inner_expression := 15 - 15
def middle_expression := 37 - inner_expression
def outer_expression (some_number : ℕ) := 45 - (some_number - middle_expression)

-- Lean 4 statement to state the proof problem
theorem number_is_28 (some_number : ℕ) (h : outer_expression some_number = 54) : some_number = 28 := by
  sorry

end number_is_28_l3_3437


namespace prob_same_color_seven_red_and_five_green_l3_3152

noncomputable def probability_same_color (red_plat : ℕ) (green_plat : ℕ) : ℚ :=
  let total_plates := red_plat + green_plat
  let total_pairs := (total_plates.choose 2) -- total ways to select 2 plates
  let red_pairs := (red_plat.choose 2) -- ways to select 2 red plates
  let green_pairs := (green_plat.choose 2) -- ways to select 2 green plates
  (red_pairs + green_pairs) / total_pairs

theorem prob_same_color_seven_red_and_five_green :
  probability_same_color 7 5 = 31 / 66 :=
by
  sorry

end prob_same_color_seven_red_and_five_green_l3_3152


namespace B_coordinates_when_A_is_origin_l3_3882

-- Definitions based on the conditions
def A_coordinates_when_B_is_origin := (2, 5)

-- Theorem to prove the coordinates of B when A is the origin
theorem B_coordinates_when_A_is_origin (x y : ℤ) :
    A_coordinates_when_B_is_origin = (2, 5) →
    (x, y) = (-2, -5) :=
by
  intro h
  -- skipping the proof steps
  sorry

end B_coordinates_when_A_is_origin_l3_3882


namespace probability_five_chords_form_convex_pentagon_l3_3058

-- Definitions of problem conditions
variable (n : ℕ) (k : ℕ)

-- Eight points on a circle
def points_on_circle : ℕ := 8

-- Number of chords selected
def selected_chords : ℕ := 5

-- Total number of ways to select 5 chords from 28 possible chords
def total_ways : ℕ := Nat.choose 28 5

-- Number of ways to select 5 points from 8, forming a convex pentagon
def favorable_ways : ℕ := Nat.choose 8 5

-- The probability computation
def probability_pentagon (total : ℕ) (favorable : ℕ) : ℚ :=
  favorable / total

theorem probability_five_chords_form_convex_pentagon :
  probability_pentagon total_ways favorable_ways = 1 / 1755 :=
by
  sorry

end probability_five_chords_form_convex_pentagon_l3_3058


namespace base10_to_base4_addition_l3_3415

-- Define the base 10 numbers
def n1 : ℕ := 45
def n2 : ℕ := 28

-- Define the base 4 representations
def n1_base4 : ℕ := 2 * 4^2 + 3 * 4^1 + 1 * 4^0
def n2_base4 : ℕ := 1 * 4^2 + 3 * 4^1 + 0 * 4^0

-- The sum of the base 10 numbers
def sum_base10 : ℕ := n1 + n2

-- The expected sum in base 4
def sum_base4 : ℕ := 1 * 4^3 + 0 * 4^2 + 2 * 4^1 + 1 * 4^0

-- Prove the equivalence
theorem base10_to_base4_addition :
  (n1 + n2 = n1_base4  + n2_base4) →
  (sum_base10 = sum_base4) :=
by
  sorry

end base10_to_base4_addition_l3_3415


namespace max_T_n_at_2_l3_3818

noncomputable def geom_seq (a n : ℕ) : ℕ :=
  a * 2 ^ n

noncomputable def S_n (a n : ℕ) : ℕ :=
  a * (2 ^ n - 1)

noncomputable def T_n (a n : ℕ) : ℕ :=
  (17 * S_n a n - S_n a (2 * n)) / geom_seq a n

theorem max_T_n_at_2 (a : ℕ) : (∀ n > 0, T_n a n ≤ T_n a 2) :=
by
  -- proof omitted
  sorry

end max_T_n_at_2_l3_3818


namespace car_selling_price_l3_3681

def car_material_cost : ℕ := 100
def car_production_per_month : ℕ := 4
def motorcycle_material_cost : ℕ := 250
def motorcycles_sold_per_month : ℕ := 8
def motorcycle_selling_price : ℤ := 50
def additional_motorcycle_profit : ℤ := 50

theorem car_selling_price (x : ℤ) :
  (motorcycles_sold_per_month * motorcycle_selling_price - motorcycle_material_cost)
  = (car_production_per_month * x - car_material_cost + additional_motorcycle_profit) →
  x = 50 :=
by
  sorry

end car_selling_price_l3_3681


namespace tied_part_length_l3_3475

theorem tied_part_length (length_of_each_string : ℕ) (num_strings : ℕ) (total_tied_length : ℕ) 
  (H1 : length_of_each_string = 217) (H2 : num_strings = 3) (H3 : total_tied_length = 627) : 
  (length_of_each_string * num_strings - total_tied_length) / (num_strings - 1) = 12 :=
by
  sorry

end tied_part_length_l3_3475


namespace maximum_gel_pens_l3_3066

theorem maximum_gel_pens 
  (x y z : ℕ) 
  (h1 : x + y + z = 20)
  (h2 : 10 * x + 50 * y + 80 * z = 1000)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) 
  : y ≤ 13 :=
sorry

end maximum_gel_pens_l3_3066


namespace single_train_car_passenger_count_l3_3255

theorem single_train_car_passenger_count (P : ℕ) 
  (h1 : ∀ (plane_capacity train_capacity : ℕ), plane_capacity = 366 →
    train_capacity = 16 * P →
      (train_capacity = (2 * plane_capacity) + 228)) : 
  P = 60 :=
by
  sorry

end single_train_car_passenger_count_l3_3255


namespace tangency_condition_l3_3374

def ellipse (x y : ℝ) : Prop := x^2 + 9 * y^2 = 9
def hyperbola (x y m : ℝ) : Prop := x^2 - m * (y + 3)^2 = 4

theorem tangency_condition (m : ℝ) :
  (∀ x y : ℝ, ellipse x y → hyperbola x y m → x^2 = 9 - 9 * y^2 ∧ x^2 = 4 + m * (y + 3)^2 → ((m - 9) * y^2 + 6 * m * y + (9 * m - 5) = 0 → 36 * m^2 - 4 * (m - 9) * (9 * m - 5) = 0 ) ) → 
  m = 5 / 54 :=
by
  sorry

end tangency_condition_l3_3374


namespace time_to_reach_ticket_window_l3_3154

-- Define the conditions as per the problem
def rate_kit : ℕ := 2 -- feet per minute (rate)
def remaining_distance : ℕ := 210 -- feet

-- Goal: To prove the time required to reach the ticket window is 105 minutes
theorem time_to_reach_ticket_window : remaining_distance / rate_kit = 105 :=
by sorry

end time_to_reach_ticket_window_l3_3154


namespace how_many_children_got_on_l3_3461

noncomputable def initial_children : ℝ := 42.5
noncomputable def children_got_off : ℝ := 21.3
noncomputable def final_children : ℝ := 35.8

theorem how_many_children_got_on : initial_children - children_got_off + (final_children - (initial_children - children_got_off)) = final_children := by
  sorry

end how_many_children_got_on_l3_3461


namespace beautiful_39th_moment_l3_3918

def is_beautiful (h : ℕ) (mm : ℕ) : Prop :=
  (h + mm) % 12 = 0

def start_time := (7, 49)

noncomputable def find_39th_beautiful_moment : ℕ × ℕ :=
  (15, 45)

theorem beautiful_39th_moment :
  find_39th_beautiful_moment = (15, 45) :=
by
  sorry

end beautiful_39th_moment_l3_3918


namespace mrs_heine_dogs_l3_3973

theorem mrs_heine_dogs (total_biscuits biscuits_per_dog : ℕ) (h1 : total_biscuits = 6) (h2 : biscuits_per_dog = 3) :
  total_biscuits / biscuits_per_dog = 2 :=
by
  sorry

end mrs_heine_dogs_l3_3973


namespace find_a_in_terms_of_x_l3_3115

theorem find_a_in_terms_of_x (a b x : ℝ) (h₁ : a ≠ b) (h₂ : a^3 - b^3 = 22 * x^3) (h₃ : a - b = 2 * x) : 
  a = x * (1 + (Real.sqrt (40 / 3)) / 2) ∨ a = x * (1 - (Real.sqrt (40 / 3)) / 2) :=
by
  sorry

end find_a_in_terms_of_x_l3_3115


namespace triangle_area_ABC_l3_3001

variable {A : Prod ℝ ℝ}
variable {B : Prod ℝ ℝ}
variable {C : Prod ℝ ℝ}

noncomputable def area_of_triangle (A B C : Prod ℝ ℝ ) : ℝ :=
  (1 / 2) * (abs ((A.1 * (B.2 - C.2)) + (B.1 * (C.2 - A.2)) + (C.1 * (A.2 - B.2))))

theorem triangle_area_ABC : 
  ∀ {A B C : Prod ℝ ℝ}, 
  A = (2, 3) → 
  B = (5, 7) → 
  C = (6, 1) → 
  area_of_triangle A B C = 11 
:= by
  intros
  subst_vars
  simp [area_of_triangle]
  sorry

end triangle_area_ABC_l3_3001


namespace count_integers_with_factors_12_and_7_l3_3744

theorem count_integers_with_factors_12_and_7 :
  ∃ k : ℕ, k = 4 ∧
    (∀ (n : ℕ), 500 ≤ n ∧ n ≤ 800 ∧ 12 ∣ n ∧ 7 ∣ n ↔ (84 ∣ n ∧
      n = 504 ∨ n = 588 ∨ n = 672 ∨ n = 756)) :=
sorry

end count_integers_with_factors_12_and_7_l3_3744


namespace required_raise_percentage_l3_3137

theorem required_raise_percentage (S : ℝ) (hS : S > 0) : 
  ((S - (0.85 * S - 50)) / (0.85 * S - 50) = 0.1875) :=
by
  -- Proof of this theorem can be carried out here
  sorry

end required_raise_percentage_l3_3137


namespace total_value_is_155_l3_3860

def coin_count := 20
def silver_coin_count := 10
def silver_coin_value_total := 30
def gold_coin_count := 5
def regular_coin_value := 1

def silver_coin_value := silver_coin_value_total / 4
def gold_coin_value := 2 * silver_coin_value

def total_silver_value := silver_coin_count * silver_coin_value
def total_gold_value := gold_coin_count * gold_coin_value
def regular_coin_count := coin_count - (silver_coin_count + gold_coin_count)
def total_regular_value := regular_coin_count * regular_coin_value

def total_collection_value := total_silver_value + total_gold_value + total_regular_value

theorem total_value_is_155 : total_collection_value = 155 := 
by
  sorry

end total_value_is_155_l3_3860


namespace parabola_focus_to_equation_l3_3864

-- Define the focus of the parabola
def F : (ℝ × ℝ) := (5, 0)

-- Define the standard equation of the parabola
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 20 * x

-- State the problem in Lean
theorem parabola_focus_to_equation : 
  (F = (5, 0)) → ∀ x y, parabola_equation x y :=
by
  intro h_focus_eq
  sorry

end parabola_focus_to_equation_l3_3864


namespace vitamin_C_relationship_l3_3179

variables (A O G : ℝ)

-- Conditions given in the problem
def condition1 : Prop := A + O + G = 275
def condition2 : Prop := 2 * A + 3 * O + 4 * G = 683

-- Rewrite the math proof problem statement
theorem vitamin_C_relationship (h1 : condition1 A O G) (h2 : condition2 A O G) : O + 2 * G = 133 :=
by {
  sorry
}

end vitamin_C_relationship_l3_3179


namespace maximum_f_value_l3_3091

noncomputable def otimes (a b : ℝ) : ℝ :=
if a ≤ b then a else b

noncomputable def f (x : ℝ) : ℝ :=
otimes (3 * x^2 + 6) (23 - x^2)

theorem maximum_f_value : ∃ x : ℝ, ∀ y : ℝ, f x ≤ f y ∧ f x = 4 :=
sorry

end maximum_f_value_l3_3091


namespace problem_l3_3220

theorem problem (x : ℝ) (h : x + 1 / x = 5) : x ^ 2 + (1 / x) ^ 2 = 23 := 
sorry

end problem_l3_3220


namespace equation_solution_l3_3041

theorem equation_solution (x y : ℕ) :
  (x^2 + 1)^y - (x^2 - 1)^y = 2 * x^y ↔ 
  (x = 1 ∧ y = 1) ∨ (x = 0 ∧ ∃ k : ℕ, y = 2 * k ∧ k > 0) :=
by sorry

end equation_solution_l3_3041


namespace sum_even_deg_coeff_l3_3477

theorem sum_even_deg_coeff (x : ℕ) : 
  (3 - 2*x)^3 * (2*x + 1)^4 = (3 - 2*x)^3 * (2*x + 1)^4 →
  (∀ (x : ℕ), (3 - 2*x)^3 * (2*1 + 1)^4 =  81 ∧ 
  (3 - 2*(-1))^3 * (2*(-1) + 1)^4 = 125 → 
  (81 + 125) / 2 = 103) :=
by
  sorry

end sum_even_deg_coeff_l3_3477


namespace eval_expression_correct_l3_3169

def eval_expression : ℤ :=
  -(-1) + abs (-1)

theorem eval_expression_correct : eval_expression = 2 :=
  by
    sorry

end eval_expression_correct_l3_3169


namespace simplify_fractions_l3_3616

theorem simplify_fractions :
  (270 / 18) * (7 / 210) * (9 / 4) = 9 / 8 :=
by sorry

end simplify_fractions_l3_3616


namespace parabola_equivalence_l3_3663

theorem parabola_equivalence :
  ∃ (a : ℝ) (h k : ℝ),
    (a = -3 ∧ h = -1 ∧ k = 2) ∧
    ∀ (x : ℝ), (y = -3 * x^2 + 1) → (y = -3 * (x + 1)^2 + 2) :=
sorry

end parabola_equivalence_l3_3663


namespace crayons_initially_l3_3101

theorem crayons_initially (crayons_left crayons_lost : ℕ) (h_left : crayons_left = 134) (h_lost : crayons_lost = 345) :
  crayons_left + crayons_lost = 479 :=
by
  sorry

end crayons_initially_l3_3101


namespace ratio_of_boys_to_girls_l3_3286

-- Variables for the number of boys, girls, and teachers
variables (B G T : ℕ)

-- Conditions from the problem
def number_of_girls := G = 60
def number_of_teachers := T = (20 * B) / 100
def total_people := B + G + T = 114

-- Proving the ratio of boys to girls is 3:4 given the conditions
theorem ratio_of_boys_to_girls 
  (hG : number_of_girls G)
  (hT : number_of_teachers B T)
  (hTotal : total_people B G T) :
  B / 15 = 3 ∧ G / 15 = 4 :=
by {
  sorry
}

end ratio_of_boys_to_girls_l3_3286


namespace prime_divisors_of_1890_l3_3197

theorem prime_divisors_of_1890 : ∃ (S : Finset ℕ), (S.card = 4) ∧ (∀ p ∈ S, Nat.Prime p) ∧ 1890 = S.prod id :=
by
  sorry

end prime_divisors_of_1890_l3_3197


namespace least_value_expression_l3_3732

-- Definition of the expression
def expression (x y : ℝ) := (x * y - 2) ^ 2 + (x - 1 + y) ^ 2

-- Statement to prove the least possible value of the expression
theorem least_value_expression : ∃ x y : ℝ, expression x y = 2 := 
sorry

end least_value_expression_l3_3732


namespace radius_wheel_l3_3447

noncomputable def pi : ℝ := 3.14159

theorem radius_wheel (D : ℝ) (N : ℕ) (r : ℝ) (h1 : D = 760.57) (h2 : N = 500) :
  r = (D / N) / (2 * pi) :=
sorry

end radius_wheel_l3_3447


namespace yellow_balls_count_l3_3830

theorem yellow_balls_count (R B G Y : ℕ) 
  (h1 : R = 2 * B) 
  (h2 : B = 2 * G) 
  (h3 : Y > 7) 
  (h4 : R + B + G + Y = 27) : 
  Y = 20 := by
  sorry

end yellow_balls_count_l3_3830


namespace value_of_a8_l3_3195

theorem value_of_a8 (a : ℕ → ℝ) :
  (1 + x) ^ 10 = a 0 + a 1 * (1 - x) + a 2 * (1 - x) ^ 2 + a 3 * (1 - x) ^ 3 +
  a 4 * (1 - x) ^ 4 + a 5 * (1 - x) ^ 5 + a 6 * (1 - x) ^ 6 + a 7 * (1 - x) ^ 7 + 
  a 8 * (1 - x) ^ 8 + a 9 * (1 - x) ^ 9 + a 10 * (1 - x) ^ 10 → 
  a 8 = 180 :=
by
  sorry

end value_of_a8_l3_3195


namespace find_a4_in_geometric_seq_l3_3157

variable {q : ℝ} -- q is the common ratio of the geometric sequence

noncomputable def geometric_seq (q : ℝ) (n : ℕ) : ℝ := 16 * q ^ (n - 1)

theorem find_a4_in_geometric_seq (h1 : geometric_seq q 1 = 16)
  (h2 : geometric_seq q 6 = 2 * geometric_seq q 5 * geometric_seq q 7) :
  geometric_seq q 4 = 2 := 
  sorry

end find_a4_in_geometric_seq_l3_3157


namespace exceed_1000_cents_l3_3953

def total_amount (n : ℕ) : ℕ :=
  3 * (3 ^ n - 1) / (3 - 1)

theorem exceed_1000_cents : 
  ∃ n : ℕ, total_amount n ≥ 1000 ∧ (n + 7) % 7 = 6 := 
by
  sorry

end exceed_1000_cents_l3_3953


namespace total_points_l3_3789

theorem total_points (total_players : ℕ) (paige_points : ℕ) (other_points : ℕ) (points_per_other_player : ℕ) :
  total_players = 5 →
  paige_points = 11 →
  points_per_other_player = 6 →
  other_points = (total_players - 1) * points_per_other_player →
  paige_points + other_points = 35 :=
by
  intro h_total_players h_paige_points h_points_per_other_player h_other_points
  sorry

end total_points_l3_3789


namespace geometric_sequence_sum_eq_80_243_l3_3695

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_eq_80_243 {n : ℕ} :
  let a := (1 / 3 : ℝ)
  let r := (1 / 3 : ℝ)
  geometric_sum a r n = 80 / 243 ↔ n = 3 :=
by
  intros a r
  sorry

end geometric_sequence_sum_eq_80_243_l3_3695


namespace clock_correct_time_fraction_l3_3104

/-- A 12-hour digital clock problem:
A 12-hour digital clock displays the hour and minute of a day.
Whenever it is supposed to display a '1' or a '2', it mistakenly displays a '9'.
The fraction of the day during which the clock shows the correct time is 7/24.
-/
theorem clock_correct_time_fraction : (7 : ℚ) / 24 = 7 / 24 :=
by sorry

end clock_correct_time_fraction_l3_3104


namespace simplify_expression_l3_3590

-- Definitions derived from the problem statement
variable (x : ℝ)

-- Theorem statement
theorem simplify_expression : 1 - (1 + (1 - (1 + (1 - x)))) = 1 - x :=
sorry

end simplify_expression_l3_3590


namespace maize_donation_amount_l3_3379

-- Definitions and Conditions
def monthly_storage : ℕ := 1
def months_in_year : ℕ := 12
def years : ℕ := 2
def stolen_tonnes : ℕ := 5
def total_tonnes_at_end : ℕ := 27

-- Theorem statement
theorem maize_donation_amount :
  let total_stored := monthly_storage * (months_in_year * years)
  let remaining_after_theft := total_stored - stolen_tonnes
  total_tonnes_at_end - remaining_after_theft = 8 :=
by
  -- This part is just the statement, hence we use sorry to omit the proof.
  sorry

end maize_donation_amount_l3_3379


namespace units_digit_n_squared_plus_two_pow_n_l3_3765

theorem units_digit_n_squared_plus_two_pow_n
  (n : ℕ)
  (h : n = 2018^2 + 2^2018) : 
  (n^2 + 2^n) % 10 = 5 := by
  sorry

end units_digit_n_squared_plus_two_pow_n_l3_3765


namespace find_a_l3_3414

theorem find_a (a : ℤ) (h₀ : 0 ≤ a ∧ a ≤ 13) (h₁ : 13 ∣ (51 ^ 2016 - a)) : a = 1 := sorry

end find_a_l3_3414


namespace number_of_boxes_needed_l3_3896

theorem number_of_boxes_needed 
  (students : ℕ) (cookies_per_student : ℕ) (cookies_per_box : ℕ) 
  (total_students : students = 134) 
  (cookies_each : cookies_per_student = 7) 
  (cookies_in_box : cookies_per_box = 28) 
  (total_cookies : students * cookies_per_student = 938)
  : Nat.ceil (938 / 28) = 34 := 
by
  sorry

end number_of_boxes_needed_l3_3896


namespace father_current_age_l3_3165

theorem father_current_age (F S : ℕ) 
  (h₁ : F - 6 = 5 * (S - 6)) 
  (h₂ : (F + 6) + (S + 6) = 78) : 
  F = 51 := 
sorry

end father_current_age_l3_3165


namespace percentage_invalid_l3_3433

theorem percentage_invalid (total_votes valid_votes_A : ℕ) (percent_A : ℝ) (total_valid_votes : ℝ) (percent_invalid : ℝ) :
  total_votes = 560000 →
  valid_votes_A = 333200 →
  percent_A = 0.70 →
  (1 - percent_invalid / 100) * total_votes = total_valid_votes →
  percent_A * total_valid_votes = valid_votes_A →
  percent_invalid = 15 :=
by
  intros h_total_votes h_valid_votes_A h_percent_A h_total_valid_votes h_valid_poll_A
  sorry

end percentage_invalid_l3_3433


namespace basketball_free_throws_l3_3668

theorem basketball_free_throws
  (a b x : ℕ)
  (h1 : 3 * b = 2 * a)
  (h2 : x = 2 * a)
  (h3 : 2 * a + 3 * b + x = 72)
  : x = 24 := by
  sorry

end basketball_free_throws_l3_3668


namespace exists_identical_coordinates_l3_3135

theorem exists_identical_coordinates
  (O O' : ℝ × ℝ)
  (Ox Oy O'x' O'y' : ℝ → ℝ)
  (units_different : ∃ u v : ℝ, u ≠ v)
  (O_ne_O' : O ≠ O')
  (Ox_not_parallel_O'x' : ∃ θ : ℝ, θ ≠ 0 ∧ θ ≠ π) :
  ∃ S : ℝ × ℝ, (S.1 = Ox S.1 ∧ S.2 = Oy S.2) ∧ (S.1 = O'x' S.1 ∧ S.2 = O'y' S.2) :=
sorry

end exists_identical_coordinates_l3_3135


namespace james_coursework_materials_expense_l3_3924

-- Definitions based on conditions
def james_budget : ℝ := 1000
def food_percentage : ℝ := 0.30
def accommodation_percentage : ℝ := 0.15
def entertainment_percentage : ℝ := 0.25

-- Calculate expenditures based on percentages
def food_expense : ℝ := food_percentage * james_budget
def accommodation_expense : ℝ := accommodation_percentage * james_budget
def entertainment_expense : ℝ := entertainment_percentage * james_budget
def total_other_expenses : ℝ := food_expense + accommodation_expense + entertainment_expense

-- Prove that the amount spent on coursework materials is $300
theorem james_coursework_materials_expense : james_budget - total_other_expenses = 300 := 
by 
  sorry

end james_coursework_materials_expense_l3_3924


namespace evaluate_expression_l3_3214

theorem evaluate_expression : 
  (1 / 10 : ℝ) + (2 / 20 : ℝ) - (3 / 60 : ℝ) = 0.15 :=
by
  sorry

end evaluate_expression_l3_3214


namespace patio_total_tiles_l3_3274

theorem patio_total_tiles (s : ℕ) (red_tiles : ℕ) (h1 : s % 2 = 1) (h2 : red_tiles = 2 * s - 1) (h3 : red_tiles = 61) :
  s * s = 961 :=
by
  sorry

end patio_total_tiles_l3_3274


namespace xiaomings_mother_money_l3_3430

-- Definitions for the conditions
def price_A : ℕ := 6
def price_B : ℕ := 9
def units_more_A := 2

-- Main statement to prove
theorem xiaomings_mother_money (x : ℕ) (M : ℕ) :
  M = 6 * x ∧ M = 9 * (x - 2) → M = 36 :=
by
  -- Assuming the conditions are given
  rintro ⟨hA, hB⟩
  -- The proof is omitted
  sorry

end xiaomings_mother_money_l3_3430


namespace hyperbola_sqrt3_eccentricity_l3_3020

noncomputable def hyperbola_eccentricity (m : ℝ) : ℝ :=
  let a := 2
  let b := m
  let c := Real.sqrt (a^2 + b^2)
  c / a

theorem hyperbola_sqrt3_eccentricity (m : ℝ) (h_m_pos : 0 < m) (h_slope : m = 2 * Real.sqrt 2) :
  hyperbola_eccentricity m = Real.sqrt 3 :=
by
  unfold hyperbola_eccentricity
  rw [h_slope]
  simp
  sorry

end hyperbola_sqrt3_eccentricity_l3_3020


namespace root_polynomial_value_l3_3047

theorem root_polynomial_value (m : ℝ) (h : m^2 + 3 * m - 2022 = 0) : m^3 + 4 * m^2 - 2019 * m - 2023 = -1 :=
  sorry

end root_polynomial_value_l3_3047


namespace start_time_is_10_am_l3_3134

-- Definitions related to the problem statements
def distance_AB : ℝ := 600
def speed_A_to_B : ℝ := 70
def speed_B_to_A : ℝ := 80
def meeting_time : ℝ := 14  -- using 24-hour format, 2 pm as 14

-- Prove that the starting time is 10 am given the conditions
theorem start_time_is_10_am (t : ℝ) :
  (speed_A_to_B * t + speed_B_to_A * t = distance_AB) →
  (meeting_time - t = 10) :=
sorry

end start_time_is_10_am_l3_3134


namespace average_age_decrease_l3_3271

-- Define the conditions as given in the problem
def original_strength : ℕ := 12
def new_students : ℕ := 12

def original_avg_age : ℕ := 40
def new_students_avg_age : ℕ := 32

def decrease_in_avg_age (O N : ℕ) (OA NA : ℕ) : ℕ :=
  let total_original_age := O * OA
  let total_new_students_age := N * NA
  let total_students := O + N
  let new_avg_age := (total_original_age + total_new_students_age) / total_students
  OA - new_avg_age

theorem average_age_decrease :
  decrease_in_avg_age original_strength new_students original_avg_age new_students_avg_age = 4 :=
sorry

end average_age_decrease_l3_3271


namespace troy_initial_straws_l3_3439

theorem troy_initial_straws (total_piglets : ℕ) (straws_per_piglet : ℕ)
  (fraction_adult_pigs : ℚ) (fraction_piglets : ℚ) 
  (adult_pigs_straws : ℕ) (piglets_straws : ℕ) 
  (total_straws : ℕ) (initial_straws : ℚ) :
  total_piglets = 20 →
  straws_per_piglet = 6 →
  fraction_adult_pigs = 3 / 5 →
  fraction_piglets = 3 / 5 →
  piglets_straws = total_piglets * straws_per_piglet →
  adult_pigs_straws = piglets_straws →
  total_straws = piglets_straws + adult_pigs_straws →
  (fraction_adult_pigs + fraction_piglets) * initial_straws = total_straws →
  initial_straws = 200 := 
by 
  sorry

end troy_initial_straws_l3_3439


namespace union_card_ge_165_l3_3089

open Finset

variable (A : Finset ℕ) (A_i : Fin (11) → Finset ℕ)
variable (hA : A.card = 225)
variable (hA_i_card : ∀ i, (A_i i).card = 45)
variable (hA_i_intersect : ∀ i j, i < j → ((A_i i) ∩ (A_i j)).card = 9)

theorem union_card_ge_165 : (Finset.biUnion Finset.univ A_i).card ≥ 165 := by sorry

end union_card_ge_165_l3_3089


namespace m_in_A_l3_3079

variable (x : ℝ)
variable (A : Set ℝ := {x | x ≤ 2})
noncomputable def m : ℝ := Real.sqrt 2

theorem m_in_A : m ∈ A :=
sorry

end m_in_A_l3_3079


namespace inequality_A_only_inequality_B_not_always_l3_3243

theorem inequality_A_only (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  a < c / 3 := 
sorry

theorem inequality_B_not_always (a b c : ℝ) (h1 : 2 * b > c) (h2 : c > a) (h3 : c > b) :
  ¬ (b < c / 3) := 
sorry

end inequality_A_only_inequality_B_not_always_l3_3243


namespace gasoline_price_increase_l3_3935

theorem gasoline_price_increase 
  (highest_price : ℝ) (lowest_price : ℝ) 
  (h_high : highest_price = 17) 
  (h_low : lowest_price = 10) : 
  (highest_price - lowest_price) / lowest_price * 100 = 70 := 
by
  /- proof can go here -/
  sorry

end gasoline_price_increase_l3_3935


namespace simplify_polynomial_l3_3904

variable (x : ℝ)

theorem simplify_polynomial :
  (2 * x^10 + 8 * x^9 + 3 * x^8) + (5 * x^12 - x^10 + 2 * x^9 - 5 * x^8 + 4 * x^5 + 6)
  = 5 * x^12 + x^10 + 10 * x^9 - 2 * x^8 + 4 * x^5 + 6 := by
  sorry

end simplify_polynomial_l3_3904


namespace three_digit_sum_reverse_eq_l3_3648

theorem three_digit_sum_reverse_eq :
  ∃ (a b c : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9),
    101 * (a + c) + 20 * b = 1777 ∧ (a, b, c) = (8, 7, 9) ∨ (a, b, c) = (9, 7, 8) :=
by
  sorry

end three_digit_sum_reverse_eq_l3_3648


namespace total_customers_l3_3828

namespace math_proof

-- Definitions based on the problem's conditions.
def tables : ℕ := 9
def women_per_table : ℕ := 7
def men_per_table : ℕ := 3

-- The theorem stating the problem's question and correct answer.
theorem total_customers : tables * (women_per_table + men_per_table) = 90 := 
by
  -- This would be expanded into a proof, but we use sorry to bypass it here.
  sorry

end math_proof

end total_customers_l3_3828


namespace sqrt_sum_eq_l3_3920

theorem sqrt_sum_eq :
  (Real.sqrt (9 / 2) + Real.sqrt (2 / 9)) = (11 * Real.sqrt 2 / 6) :=
sorry

end sqrt_sum_eq_l3_3920


namespace range_of_x_l3_3562

def f (x a : ℝ) : ℝ := x^3 + 3 * a * x - 1

def g (x a : ℝ) : ℝ := 3 * x^2 - a * x + 3 * a - 5

def condition (a : ℝ) : Prop := -1 ≤ a ∧ a ≤ 1

theorem range_of_x (x a : ℝ) (h : condition a) : g x a < 0 → -2/3 < x ∧ x < 1 := 
sorry

end range_of_x_l3_3562


namespace area_quadrilateral_EFGH_l3_3758

-- Define the rectangles ABCD and XYZR
def area_rectangle_ABCD : ℝ := 60 
def area_rectangle_XYZR : ℝ := 4

-- Define what needs to be proven: the area of quadrilateral EFGH
theorem area_quadrilateral_EFGH (a b c d : ℝ) :
  (area_rectangle_ABCD = area_rectangle_XYZR + 2 * (a + b + c + d)) →
  (a + b + c + d = 28) →
  (area_rectangle_XYZR = 4) →
  (area_rectangle_ABCD = 60) →
  (a + b + c + d + area_rectangle_XYZR = 32) :=
by
  intros h1 h2 h3 h4
  sorry

end area_quadrilateral_EFGH_l3_3758


namespace min_value_of_sequence_l3_3557

variable (b1 b2 b3 : ℝ)

def geometric_sequence (b1 b2 b3 : ℝ) : Prop :=
  ∃ s : ℝ, b2 = b1 * s ∧ b3 = b1 * s^2 

theorem min_value_of_sequence (h1 : b1 = 2) (h2 : geometric_sequence b1 b2 b3) :
  ∃ s : ℝ, 3 * b2 + 4 * b3 = -9 / 8 :=
sorry

end min_value_of_sequence_l3_3557


namespace largest_fraction_l3_3796

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (7 : ℚ) / 15
  let C := (29 : ℚ) / 59
  let D := (200 : ℚ) / 399
  let E := (251 : ℚ) / 501
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_fraction_l3_3796


namespace power_of_m_l3_3049

theorem power_of_m (m : ℕ) (h₁ : ∀ k : ℕ, m^k % 24 = 0) (h₂ : ∀ d : ℕ, d ∣ m → d ≤ 8) : ∃ k : ℕ, m^k = 24 :=
sorry

end power_of_m_l3_3049


namespace find_general_term_l3_3149

theorem find_general_term (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n, a (n + 1) = 2 * a n + n^2) :
  ∀ n, a n = 7 * 2^(n - 1) - n^2 - 2 * n - 3 :=
by
  sorry

end find_general_term_l3_3149


namespace principal_amount_l3_3995

/-- Given:
 - 820 = P + (P * R * 2) / 100
 - 1020 = P + (P * R * 6) / 100
Prove:
 - P = 720
--/

theorem principal_amount (P R : ℝ) (h1 : 820 = P + (P * R * 2) / 100) (h2 : 1020 = P + (P * R * 6) / 100) : P = 720 :=
by
  sorry

end principal_amount_l3_3995


namespace turban_price_l3_3339

theorem turban_price (T : ℝ) (total_salary : ℝ) (received_salary : ℝ)
  (cond1 : total_salary = 90 + T)
  (cond2 : received_salary = 65 + T)
  (cond3 : received_salary = (3 / 4) * total_salary) :
  T = 10 :=
by
  sorry

end turban_price_l3_3339


namespace baseball_card_problem_l3_3063

theorem baseball_card_problem:
  let initial_cards := 15
  let maria_takes := (initial_cards + 1) / 2
  let cards_after_maria := initial_cards - maria_takes
  let cards_after_peter := cards_after_maria - 1
  let final_cards := cards_after_peter * 3
  final_cards = 18 :=
by
  sorry

end baseball_card_problem_l3_3063


namespace find_S10_value_l3_3313

noncomputable def sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
∀ n : ℕ, 4 * S n = n * (a n + a (n + 1))

theorem find_S10_value (a S : ℕ → ℕ) (h1 : a 4 = 7) (h2 : sequence_sum a S) :
  S 10 = 100 :=
sorry

end find_S10_value_l3_3313


namespace sum_of_digits_of_square_99999_l3_3622

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_of_digits_of_square_99999 : sum_of_digits ((99999 : ℕ)^2) = 45 := by
  sorry

end sum_of_digits_of_square_99999_l3_3622


namespace problem_solution_l3_3771

noncomputable def given_problem : ℝ := (Real.pi - 3)^0 - Real.sqrt 8 + 2 * Real.sin (45 * Real.pi / 180) + (1 / 2)⁻¹

theorem problem_solution : given_problem = 3 - Real.sqrt 2 := by
  sorry

end problem_solution_l3_3771


namespace number_of_cars_l3_3579

theorem number_of_cars (n s t C : ℕ) (h1 : n = 9) (h2 : s = 4) (h3 : t = 3) (h4 : n * s = t * C) : C = 12 :=
by
  sorry

end number_of_cars_l3_3579


namespace value_of_x_squared_add_reciprocal_squared_l3_3129

theorem value_of_x_squared_add_reciprocal_squared (x : ℝ) (h : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end value_of_x_squared_add_reciprocal_squared_l3_3129


namespace find_m_value_l3_3969

theorem find_m_value (m : ℤ) (h : (∀ x : ℤ, (x-5)*(x+7) = x^2 - mx - 35)) : m = -2 :=
by sorry

end find_m_value_l3_3969


namespace jacket_total_selling_price_l3_3950

theorem jacket_total_selling_price :
  let original_price := 120
  let discount_rate := 0.30
  let tax_rate := 0.08
  let processing_fee := 5
  let discounted_price := original_price * (1 - discount_rate)
  let tax := discounted_price * tax_rate
  let total_price := discounted_price + tax + processing_fee
  total_price = 95.72 := by
  sorry

end jacket_total_selling_price_l3_3950


namespace div_sub_mult_exp_eq_l3_3531

-- Lean 4 statement for the mathematical proof problem
theorem div_sub_mult_exp_eq :
  8 / 4 - 3 - 9 + 3 * 7 - 2^2 = 7 := 
sorry

end div_sub_mult_exp_eq_l3_3531


namespace find_m_if_parallel_l3_3462

-- Given vectors
def a (m : ℝ) : ℝ × ℝ := (2, m)
def b (m : ℝ) : ℝ × ℝ := (m, 2)

-- Parallel condition and the result that m must be -2 or 2
theorem find_m_if_parallel (m : ℝ) (h : ∃ k : ℝ, a m = (k * (b m).fst, k * (b m).snd)) : 
  m = -2 ∨ m = 2 :=
sorry

end find_m_if_parallel_l3_3462


namespace appears_every_number_smallest_triplicate_number_l3_3842

open Nat

/-- Pascal's triangle is constructed such that each number 
    is the sum of the two numbers directly above it in the 
    previous row -/
def pascal (r k : ℕ) : ℕ :=
  if k > r then 0 else Nat.choose r k

/-- Every positive integer does appear at least once, but not 
    necessarily more than once for smaller numbers -/
theorem appears_every_number (n : ℕ) : ∃ r k : ℕ, pascal r k = n := sorry

/-- The smallest three-digit number in Pascal's triangle 
    that appears more than once is 102 -/
theorem smallest_triplicate_number : ∃ r1 k1 r2 k2 : ℕ, 
  100 ≤ pascal r1 k1 ∧ pascal r1 k1 < 1000 ∧ 
  pascal r1 k1 = 102 ∧ 
  r1 ≠ r2 ∧ k1 ≠ k2 ∧ 
  pascal r1 k1 = pascal r2 k2 := sorry

end appears_every_number_smallest_triplicate_number_l3_3842


namespace parabola_transform_l3_3801

theorem parabola_transform (b c : ℝ) : 
  (∀ x : ℝ, x^2 + b * x + c = (x - 4)^2 - 3) → 
  b = 4 ∧ c = 6 := 
by
  sorry

end parabola_transform_l3_3801


namespace n_minus_m_l3_3469

theorem n_minus_m (m n : ℝ) (h1 : m^2 - n^2 = 6) (h2 : m + n = 3) : n - m = -2 :=
by
  sorry

end n_minus_m_l3_3469


namespace percentage_of_tip_is_25_l3_3180

-- Definitions of the costs
def cost_samosas : ℕ := 3 * 2
def cost_pakoras : ℕ := 4 * 3
def cost_mango_lassi : ℕ := 2

-- Definition of total food cost
def total_food_cost : ℕ := cost_samosas + cost_pakoras + cost_mango_lassi

-- Definition of the total meal cost including tax
def total_meal_cost_with_tax : ℕ := 25

-- Definition of the tip
def tip : ℕ := total_meal_cost_with_tax - total_food_cost

-- Definition of the percentage of the tip
def percentage_tip : ℕ := (tip * 100) / total_food_cost

-- The theorem to be proved
theorem percentage_of_tip_is_25 :
  percentage_tip = 25 :=
by
  sorry

end percentage_of_tip_is_25_l3_3180


namespace circle_eq_focus_tangent_directrix_l3_3559

theorem circle_eq_focus_tangent_directrix (x y : ℝ) :
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  ((x - focus.1)^2 + (y - focus.2)^2 = radius^2) :=
by
  let focus := (0, 4)
  let directrix := -4
  let radius := 8
  sorry

end circle_eq_focus_tangent_directrix_l3_3559


namespace purely_imaginary_complex_number_l3_3116

theorem purely_imaginary_complex_number (a : ℝ) (i : ℂ)
  (h₁ : i * i = -1)
  (h₂ : ∃ z : ℂ, z = (a + i) / (1 - i) ∧ z.im ≠ 0 ∧ z.re = 0) :
  a = 1 :=
sorry

end purely_imaginary_complex_number_l3_3116


namespace product_is_correct_l3_3125

noncomputable def IKS := 521
noncomputable def KSI := 215
def product := 112015

theorem product_is_correct : IKS * KSI = product :=
by
  -- Proof yet to be constructed
  sorry

end product_is_correct_l3_3125


namespace part_a_part_b_l3_3131

-- Assuming existence of function S satisfying certain properties
variable (S : Type → Type → Type → ℝ)

-- Part (a)
theorem part_a (A B C : Type) : 
  S A B C = -S B A C ∧ S A B C = S B C A :=
sorry

-- Part (b)
theorem part_b (A B C D : Type) : 
  S A B C = S D A B + S D B C + S D C A :=
sorry

end part_a_part_b_l3_3131


namespace points_on_hyperbola_l3_3376

theorem points_on_hyperbola {s : ℝ} :
  let x := Real.exp s - Real.exp (-s)
  let y := 5 * (Real.exp s + Real.exp (-s))
  (y^2 / 100 - x^2 / 4 = 1) :=
by
  sorry

end points_on_hyperbola_l3_3376


namespace integer_solutions_system_inequalities_l3_3175

theorem integer_solutions_system_inequalities:
  {x : ℤ} → (2 * x - 1 < x + 1) → (1 - 2 * (x - 1) ≤ 3) → x = 0 ∨ x = 1 := 
by
  intros x h1 h2
  sorry

end integer_solutions_system_inequalities_l3_3175


namespace tom_books_after_transactions_l3_3585

-- Define the initial conditions as variables
def initial_books : ℕ := 5
def sold_books : ℕ := 4
def new_books : ℕ := 38

-- Define the property we need to prove
theorem tom_books_after_transactions : initial_books - sold_books + new_books = 39 := by
  sorry

end tom_books_after_transactions_l3_3585


namespace coordinates_of_point_on_x_axis_l3_3817

theorem coordinates_of_point_on_x_axis (m : ℤ) 
  (h : 2 * m + 8 = 0) : (m + 5, 2 * m + 8) = (1, 0) :=
sorry

end coordinates_of_point_on_x_axis_l3_3817


namespace intersection_A_B_l3_3778

open Set

def A : Set ℝ := Icc 1 2

def B : Set ℤ := {x : ℤ | x^2 - 2 * x - 3 < 0}

theorem intersection_A_B :
  (A ∩ (coe '' B) : Set ℝ) = {1, 2} :=
sorry

end intersection_A_B_l3_3778


namespace f_decreasing_max_k_value_l3_3472

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_decreasing : ∀ x > 0, ∀ y > 0, x < y → f x > f y := by
  sorry

theorem max_k_value : ∀ x > 0, f x > k / (x + 1) → k ≤ 3 := by
  sorry

end f_decreasing_max_k_value_l3_3472


namespace sum_six_consecutive_integers_l3_3577

theorem sum_six_consecutive_integers (n : ℤ) : 
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l3_3577


namespace solve_quadratic_inequality_l3_3795

theorem solve_quadratic_inequality :
  { x : ℝ | -3 * x^2 + 8 * x + 5 < 0 } = { x : ℝ | x < -1 ∨ x > 5 / 3 } :=
sorry

end solve_quadratic_inequality_l3_3795


namespace ratio_of_b_to_sum_a_c_l3_3662

theorem ratio_of_b_to_sum_a_c (a b c : ℕ) (h1 : a + b + c = 60) (h2 : a = 1/3 * (b + c)) (h3 : c = 35) : b = 1/5 * (a + c) :=
by
  sorry

end ratio_of_b_to_sum_a_c_l3_3662


namespace last_digit_of_3_pow_2012_l3_3640

-- Theorem: The last digit of 3^2012 is 1 given the cyclic pattern of last digits for powers of 3.
theorem last_digit_of_3_pow_2012 : (3 ^ 2012) % 10 = 1 :=
by
  sorry

end last_digit_of_3_pow_2012_l3_3640


namespace bins_of_soup_l3_3378

theorem bins_of_soup (total_bins : ℝ) (bins_of_vegetables : ℝ) (bins_of_pasta : ℝ) 
(h1 : total_bins = 0.75) (h2 : bins_of_vegetables = 0.125) (h3 : bins_of_pasta = 0.5) :
  total_bins - (bins_of_vegetables + bins_of_pasta) = 0.125 := by
  -- proof
  sorry

end bins_of_soup_l3_3378


namespace supremum_neg_frac_l3_3210

noncomputable def supremum_expression (a b : ℝ) : ℝ :=
  - (1 / (2 * a) + 2 / b)

theorem supremum_neg_frac {a b : ℝ} (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  ∃ M : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ M)
  ∧ (∀ N : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → x + y = 1 → supremum_expression x y ≤ N) → M ≤ N)
  ∧ M = -9 / 2 :=
sorry

end supremum_neg_frac_l3_3210


namespace find_coordinates_of_P_l3_3756

-- Definitions based on the conditions:
-- Point P has coordinates (a, 2a-1) and lies on the line y = x.

def lies_on_bisector (a : ℝ) : Prop :=
  (2 * a - 1) = a -- This is derived from the line y = x for the given point coordinates.

-- The final statement to prove:
theorem find_coordinates_of_P (a : ℝ) (P : ℝ × ℝ) (h1 : P = (a, 2 * a - 1)) (h2 : lies_on_bisector a) :
  P = (1, 1) :=
by
  -- Proof steps are omitted and replaced with sorry.
  sorry

end find_coordinates_of_P_l3_3756


namespace evaluate_expression_l3_3638

theorem evaluate_expression : (3 + 5 + 7) / (2 + 4 + 6) - (2 + 4 + 6) / (3 + 5 + 7) = 9 / 20 := 
by 
  sorry

end evaluate_expression_l3_3638


namespace find_value_of_t_l3_3912

variable (a b v d t r : ℕ)

-- All variables are non-zero digits (1-9)
axiom non_zero_a : 0 < a ∧ a < 10
axiom non_zero_b : 0 < b ∧ b < 10
axiom non_zero_v : 0 < v ∧ v < 10
axiom non_zero_d : 0 < d ∧ d < 10
axiom non_zero_t : 0 < t ∧ t < 10
axiom non_zero_r : 0 < r ∧ r < 10

-- Given conditions
axiom condition1 : a + b = v
axiom condition2 : v + d = t
axiom condition3 : t + a = r
axiom condition4 : b + d + r = 18

theorem find_value_of_t : t = 9 :=
by sorry

end find_value_of_t_l3_3912


namespace find_x_of_orthogonal_vectors_l3_3112

variable (x : ℝ)

def a : ℝ × ℝ × ℝ := (2, -3, 1)
def b : ℝ × ℝ × ℝ := (-4, 2, x)

theorem find_x_of_orthogonal_vectors (h : (2 * -4 + -3 * 2 + 1 * x) = 0) : x = 14 := by
  sorry

end find_x_of_orthogonal_vectors_l3_3112


namespace choose_students_l3_3345

/-- There are 50 students in the class, including one class president and one vice-president. 
    We want to select 5 students to participate in an activity such that at least one of 
    the class president or vice-president is included. We assert that there are exactly 2 
    distinct methods for making this selection. -/
theorem choose_students (students : Finset ℕ) (class_president vice_president : ℕ) (students_card : students.card = 50)
  (students_ex : class_president ∈ students ∧ vice_president ∈ students) : 
  ∃ valid_methods : Finset (Finset ℕ), valid_methods.card = 2 :=
by
  sorry

end choose_students_l3_3345


namespace max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l3_3534

noncomputable def point_in_circle : Prop :=
  let P := (-Real.sqrt 3, 2)
  ∃ (x y : ℝ), x^2 + y^2 = 12 ∧ x = -Real.sqrt 3 ∧ y = 2

theorem max_min_AB_length (α : ℝ) (h1 : -Real.sqrt 3 ≤ α ∧ α ≤ Real.pi / 2) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let R := Real.sqrt 12
  ∀ (A B : ℝ × ℝ), (A.1^2 + A.2^2 = 12 ∧ B.1^2 + B.2^2 = 12 ∧ (P.1, P.2) = (-Real.sqrt 3, 2)) →
    ((max (dist A B) (dist P P)) = 4 * Real.sqrt 3 ∧ (min (dist A B) (dist P P)) = 2 * Real.sqrt 5) :=
sorry

theorem chord_length_at_angle (α : ℝ) (h2 : α = 120 / 180 * Real.pi) :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  let A := (Real.sqrt 12, 0)
  let B := (-Real.sqrt 12, 0)
  let AB := (dist A B)
  AB = Real.sqrt 47 :=
sorry

theorem trajectory_midpoint_chord :
  let P : ℝ × ℝ := (-Real.sqrt 3, 2)
  ∀ (M : ℝ × ℝ), (∀ k : ℝ, P.2 - 2 = k * (P.1 + Real.sqrt 3) ∧ M.2 = - 1 / k * M.1) → 
  (M.1^2 + M.2^2 + Real.sqrt 3 * M.1 + 2 * M.2 = 0) :=
sorry

end max_min_AB_length_chord_length_at_angle_trajectory_midpoint_chord_l3_3534


namespace solve_equation_integers_l3_3698

theorem solve_equation_integers :
  ∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  (1 + 1 / (x : ℚ)) * (1 + 1 / (y : ℚ)) * (1 + 1 / (z : ℚ)) = 2 ∧
  (x = 2 ∧ y = 4 ∧ z = 15 ∨
   x = 2 ∧ y = 5 ∧ z = 9 ∨
   x = 2 ∧ y = 6 ∧ z = 7 ∨
   x = 3 ∧ y = 4 ∧ z = 5 ∨
   x = 3 ∧ y = 3 ∧ z = 8 ∨
   x = 2 ∧ y = 15 ∧ z = 4 ∨
   x = 2 ∧ y = 9 ∧ z = 5 ∨
   x = 2 ∧ y = 7 ∧ z = 6 ∨
   x = 3 ∧ y = 5 ∧ z = 4 ∨
   x = 3 ∧ y = 8 ∧ z = 3) ∧
  (y = 2 ∧ x = 4 ∧ z = 15 ∨
   y = 2 ∧ x = 5 ∧ z = 9 ∨
   y = 2 ∧ x = 6 ∧ z = 7 ∨
   y = 3 ∧ x = 4 ∧ z = 5 ∨
   y = 3 ∧ x = 3 ∧ z = 8 ∨
   y = 15 ∧ x = 4 ∧ z = 2 ∨
   y = 9 ∧ x = 5 ∧ z = 2 ∨
   y = 7 ∧ x = 6 ∧ z = 2 ∨
   y = 5 ∧ x = 4 ∧ z = 3 ∨
   y = 8 ∧ x = 3 ∧ z = 3) ∧
  (z = 2 ∧ x = 4 ∧ y = 15 ∨
   z = 2 ∧ x = 5 ∧ y = 9 ∨
   z = 2 ∧ x = 6 ∧ y = 7 ∨
   z = 3 ∧ x = 4 ∧ y = 5 ∨
   z = 3 ∧ x = 3 ∧ y = 8 ∨
   z = 15 ∧ x = 4 ∧ y = 2 ∨
   z = 9 ∧ x = 5 ∧ y = 2 ∨
   z = 7 ∧ x = 6 ∧ y = 2 ∨
   z = 5 ∧ x = 4 ∧ y = 3 ∨
   z = 8 ∧ x = 3 ∧ y = 3)
:= sorry

end solve_equation_integers_l3_3698


namespace function_properties_l3_3394

theorem function_properties
  (f : ℝ → ℝ)
  (h1 : ∀ (x1 x2 : ℝ), x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0)
  (h2 : ∀ x, f (x - t) = f (x + t)) 
  (h3_even : ∀ x, f (-x) = f x)
  (h3_decreasing : ∀ x1 x2, x1 < x2 ∧ x2 < 0 → f x1 > f x2)
  (h3_at_neg2 : f (-2) = 0)
  (h4_odd : ∀ x, f (-x) = -f x) : 
  ((∀ x1 x2, x1 < x2 → f x1 > f x2) ∧
   (¬∀ x, (f x > 0) ↔ (-2 < x ∧ x < 2)) ∧
   (∀ x, f (x) * f (|x|) = - f (-x) * f |x|) ∧
   (¬∀ x, f (x) = f (x + 2 * t))) :=
by 
  sorry

end function_properties_l3_3394


namespace dave_deleted_apps_l3_3535

def apps_initial : ℕ := 23
def apps_left : ℕ := 5
def apps_deleted : ℕ := apps_initial - apps_left

theorem dave_deleted_apps : apps_deleted = 18 := 
by
  sorry

end dave_deleted_apps_l3_3535


namespace total_ounces_of_coffee_l3_3691

/-
Defining the given conditions
-/
def num_packages_10_oz : Nat := 5
def num_packages_5_oz : Nat := num_packages_10_oz + 2
def ounces_per_10_oz_pkg : Nat := 10
def ounces_per_5_oz_pkg : Nat := 5

/-
Statement to prove the total ounces of coffee
-/
theorem total_ounces_of_coffee :
  (num_packages_10_oz * ounces_per_10_oz_pkg + num_packages_5_oz * ounces_per_5_oz_pkg) = 85 := by
  sorry

end total_ounces_of_coffee_l3_3691


namespace loan_amount_calculation_l3_3352

theorem loan_amount_calculation
  (annual_interest : ℝ) (interest_rate : ℝ) (time : ℝ) (loan_amount : ℝ)
  (h1 : annual_interest = 810)
  (h2 : interest_rate = 0.09)
  (h3 : time = 1)
  (h4 : loan_amount = annual_interest / (interest_rate * time)) :
  loan_amount = 9000 := by
sorry

end loan_amount_calculation_l3_3352


namespace cathy_total_money_l3_3082

variable (i d m : ℕ)
variable (h1 : i = 12)
variable (h2 : d = 25)
variable (h3 : m = 2 * d)

theorem cathy_total_money : i + d + m = 87 :=
by
  rw [h1, h2, h3]
  -- Continue proof steps here if necessary
  sorry

end cathy_total_money_l3_3082


namespace part_i_solution_set_part_ii_minimum_value_l3_3478

-- Part (I)
theorem part_i_solution_set :
  (∀ (x : ℝ), 1 = 1 ∧ 2 = 2 → |x - 1| + |x + 2| ≤ 5 ↔ -3 ≤ x ∧ x ≤ 2) :=
by { sorry }

-- Part (II)
theorem part_ii_minimum_value (a b x : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 2 * a * b) :
  |x - a| + |x + b| ≥ 9 / 2 :=
by { sorry }

end part_i_solution_set_part_ii_minimum_value_l3_3478


namespace remainder_twice_sum_first_150_mod_10000_eq_2650_l3_3441

theorem remainder_twice_sum_first_150_mod_10000_eq_2650 :
  let n := 150
  let S := n * (n + 1) / 2  -- Sum of first 150 numbers
  let result := 2 * S
  result % 10000 = 2650 :=
by
  sorry -- proof not required

end remainder_twice_sum_first_150_mod_10000_eq_2650_l3_3441


namespace simplify_expression_correct_l3_3172

noncomputable def simplify_expression (m n : ℝ) : ℝ :=
  ( (2 - n) / (n - 1) + 4 * ((m - 1) / (m - 2)) ) /
  ( n^2 * ((m - 1) / (n - 1)) + m^2 * ((2 - n) / (m - 2)) )

theorem simplify_expression_correct :
  simplify_expression (Real.rpow 400 (1/4)) (Real.sqrt 5) = (Real.sqrt 5) / 5 := 
sorry

end simplify_expression_correct_l3_3172


namespace GP_length_l3_3752

theorem GP_length (X Y Z G P Q : Type) 
  (XY XZ YZ : ℝ) 
  (hXY : XY = 12) 
  (hXZ : XZ = 9) 
  (hYZ : YZ = 15) 
  (hG_centroid : true)  -- Medians intersect at G (Centroid property)
  (hQ_altitude : true)  -- Q is the foot of the altitude from X to YZ
  (hP_below_G : true)  -- P is the point on YZ directly below G
  : GP = 2.4 := 
sorry

end GP_length_l3_3752


namespace number_of_integers_inequality_l3_3204

theorem number_of_integers_inequality : (∃ s : Finset ℤ, (∀ x ∈ s, 10 ≤ x ∧ x ≤ 24) ∧ s.card = 15) :=
by
  sorry

end number_of_integers_inequality_l3_3204


namespace balls_per_bag_l3_3401

theorem balls_per_bag (total_balls : ℕ) (total_bags : ℕ) (h1 : total_balls = 36) (h2 : total_bags = 9) : total_balls / total_bags = 4 :=
by
  sorry

end balls_per_bag_l3_3401


namespace product_of_two_numbers_l3_3700

theorem product_of_two_numbers :
  ∃ x y : ℝ, x + y = 16 ∧ x^2 + y^2 = 200 ∧ x * y = 28 :=
by
  sorry

end product_of_two_numbers_l3_3700


namespace combined_weight_of_jake_and_sister_l3_3244

theorem combined_weight_of_jake_and_sister (j s : ℕ) (h1 : j = 188) (h2 : j - 8 = 2 * s) : j + s = 278 :=
sorry

end combined_weight_of_jake_and_sister_l3_3244


namespace math_problem_l3_3827

-- Definition of ⊕
def opp (a b : ℝ) : ℝ := a * b + a - b

-- Definition of ⊗
def tensor (a b : ℝ) : ℝ := (a * b) + a - b

theorem math_problem (a b : ℝ) :
  opp a b + tensor (b - a) b = b^2 - b := 
by
  sorry

end math_problem_l3_3827


namespace susan_avg_speed_l3_3813

theorem susan_avg_speed 
  (speed1 : ℕ)
  (distance1 : ℕ)
  (speed2 : ℕ)
  (distance2 : ℕ)
  (no_stops : Prop) 
  (H1 : speed1 = 15)
  (H2 : distance1 = 40)
  (H3 : speed2 = 60)
  (H4 : distance2 = 20)
  (H5 : no_stops) :
  let total_distance := distance1 + distance2
  let time1 := distance1 / speed1
  let time2 := distance2 / speed2
  let total_time := time1 + time2
  let avg_speed := total_distance / total_time
  avg_speed = 20 := by
  sorry

end susan_avg_speed_l3_3813


namespace inheritance_calculation_l3_3761

theorem inheritance_calculation
  (x : ℝ)
  (h1 : 0.25 * x + 0.15 * (0.75 * x) = 14000) :
  x = 38600 := by
  sorry

end inheritance_calculation_l3_3761


namespace denomination_of_four_bills_l3_3069

theorem denomination_of_four_bills (X : ℕ) (h1 : 10 * 20 + 8 * 10 + 4 * X = 300) : X = 5 :=
by
  -- proof goes here
  sorry

end denomination_of_four_bills_l3_3069


namespace chloe_sold_strawberries_l3_3583

noncomputable section

def cost_per_dozen : ℕ := 50
def sale_price_per_half_dozen : ℕ := 30
def total_profit : ℕ := 500
def profit_per_half_dozen := sale_price_per_half_dozen - (cost_per_dozen / 2)
def half_dozens_sold := total_profit / profit_per_half_dozen

theorem chloe_sold_strawberries : half_dozens_sold / 2 = 50 :=
by
  -- proof would go here
  sorry

end chloe_sold_strawberries_l3_3583


namespace veranda_width_l3_3432

theorem veranda_width (w : ℝ) (h_room : 18 * 12 = 216) (h_veranda : 136 = 136) : 
  (18 + 2*w) * (12 + 2*w) = 352 → w = 2 :=
by
  sorry

end veranda_width_l3_3432


namespace midpoint_one_seventh_one_ninth_l3_3659

theorem midpoint_one_seventh_one_ninth : 
  let a := (1 : ℚ) / 7
  let b := (1 : ℚ) / 9
  (a + b) / 2 = 8 / 63 := 
by
  sorry

end midpoint_one_seventh_one_ninth_l3_3659


namespace truncated_cone_surface_area_l3_3688

theorem truncated_cone_surface_area (R r : ℝ) (S : ℝ)
  (h1: S = 4 * Real.pi * (R^2 + R * r + r^2)) :
  2 * Real.pi * (R^2 + R * r + r^2) = S / 2 :=
by
  sorry

end truncated_cone_surface_area_l3_3688


namespace simplify_fraction_l3_3731

variable {x y : ℝ}

theorem simplify_fraction (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end simplify_fraction_l3_3731


namespace fraction_covered_by_pepperoni_l3_3492

theorem fraction_covered_by_pepperoni 
  (d_pizza : ℝ) (n_pepperoni_diameter : ℕ) (n_pepperoni : ℕ) (diameter_pepperoni : ℝ) 
  (radius_pepperoni : ℝ) (radius_pizza : ℝ)
  (area_one_pepperoni : ℝ) (total_area_pepperoni : ℝ) (area_pizza : ℝ)
  (fraction_covered : ℝ)
  (h1 : d_pizza = 16)
  (h2 : n_pepperoni_diameter = 14)
  (h3 : n_pepperoni = 42)
  (h4 : diameter_pepperoni = d_pizza / n_pepperoni_diameter)
  (h5 : radius_pepperoni = diameter_pepperoni / 2)
  (h6 : radius_pizza = d_pizza / 2)
  (h7 : area_one_pepperoni = π * radius_pepperoni ^ 2)
  (h8 : total_area_pepperoni = n_pepperoni * area_one_pepperoni)
  (h9 : area_pizza = π * radius_pizza ^ 2)
  (h10 : fraction_covered = total_area_pepperoni / area_pizza) :
  fraction_covered = 3 / 7 :=
sorry

end fraction_covered_by_pepperoni_l3_3492


namespace imaginary_part_of_z_l3_3762

open Complex

theorem imaginary_part_of_z (z : ℂ) (h : I * z = 1 + I) : z.im = -1 := 
sorry

end imaginary_part_of_z_l3_3762


namespace system_solution_find_a_l3_3035

theorem system_solution (x y : ℝ) (a : ℝ) :
  (|16 + 6 * x - x ^ 2 - y ^ 2| + |6 * x| = 16 + 12 * x - x ^ 2 - y ^ 2)
  ∧ ((a + 15) * y + 15 * x - a = 0) →
  ( (x - 3) ^ 2 + y ^ 2 ≤ 25 ∧ x ≥ 0 ) :=
sorry

theorem find_a (a : ℝ) :
  ∃ (x y : ℝ), 
  ((a + 15) * y + 15 * x - a = 0 ∧ x ≥ 0 ∧ (x - 3) ^ 2 + y ^ 2 ≤ 25) ↔ 
  (a = -20 ∨ a = -12) :=
sorry

end system_solution_find_a_l3_3035


namespace solution_set_of_inequality_system_l3_3459

theorem solution_set_of_inequality_system :
  (6 - 2 * x ≥ 0) ∧ (2 * x + 4 > 0) ↔ (-2 < x ∧ x ≤ 3) := 
sorry

end solution_set_of_inequality_system_l3_3459


namespace pasta_sauce_cost_l3_3044

theorem pasta_sauce_cost :
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  pasta_sauce_cost = 5 :=
by
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  sorry

end pasta_sauce_cost_l3_3044


namespace exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l3_3736

def small_numbers (n : ℕ) : Prop := n ≤ 150

theorem exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest :
  ∃ (N : ℕ), (∃ (a b : ℕ), small_numbers a ∧ small_numbers b ∧ (a + 1 = b) ∧ ¬(N % a = 0) ∧ ¬(N % b = 0))
  ∧ (∀ (m : ℕ), small_numbers m → ¬(m = a ∨ m = b) → N % m = 0) :=
sorry

end exists_n_not_divisible_by_2_consecutive_but_divisible_by_rest_l3_3736


namespace percent_profit_l3_3183

theorem percent_profit (C S : ℝ) (h : 55 * C = 50 * S) : 
  100 * ((S - C) / C) = 10 :=
by
  sorry

end percent_profit_l3_3183


namespace T_shaped_area_l3_3365

theorem T_shaped_area (a b c d : ℕ) (side1 side2 side3 large_side : ℕ)
  (h_side1: side1 = 2)
  (h_side2: side2 = 2)
  (h_side3: side3 = 4)
  (h_large_side: large_side = 6)
  (h_area_large_square : a = large_side * large_side)
  (h_area_square1 : b = side1 * side1)
  (h_area_square2 : c = side2 * side2)
  (h_area_square3 : d = side3 * side3) :
  a - (b + c + d) = 12 := by
  sorry

end T_shaped_area_l3_3365


namespace simplify_sum_of_polynomials_l3_3290

-- Definitions of the given polynomials
def P (x : ℝ) : ℝ := 2 * x^5 - 3 * x^4 + x^3 + 5 * x^2 - 8 * x + 15
def Q (x : ℝ) : ℝ := -5 * x^4 - 2 * x^3 + 3 * x^2 + 8 * x + 9

-- Statement to prove that the sum of P and Q equals the simplified polynomial
theorem simplify_sum_of_polynomials (x : ℝ) : 
  P x + Q x = 2 * x^5 - 8 * x^4 - x^3 + 8 * x^2 + 24 := 
sorry

end simplify_sum_of_polynomials_l3_3290


namespace chickens_and_sheep_are_ten_l3_3847

noncomputable def chickens_and_sheep_problem (C S : ℕ) : Prop :=
  (C + 4 * S = 2 * C) ∧ (2 * C + 4 * (S - 4) = 16 * (S - 4)) → (S + 2 = 10)

theorem chickens_and_sheep_are_ten (C S : ℕ) : chickens_and_sheep_problem C S :=
sorry

end chickens_and_sheep_are_ten_l3_3847


namespace b101_mod_49_l3_3117

-- Definitions based on conditions
def b (n : ℕ) : ℕ := 5^n + 7^n

-- The formal statement of the proof problem
theorem b101_mod_49 : b 101 % 49 = 12 := by
  sorry

end b101_mod_49_l3_3117


namespace inequality_sum_squares_l3_3831

theorem inequality_sum_squares (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a + b + c = 1) :
  (a + 1/a)^2 + (b + 1/b)^2 + (c + 1/c)^2 ≥ 100/3 :=
sorry

end inequality_sum_squares_l3_3831


namespace distance_to_school_l3_3278

def jerry_one_way_time : ℝ := 15  -- Jerry's one-way time in minutes
def carson_speed_mph : ℝ := 8  -- Carson's speed in miles per hour
def minutes_per_hour : ℝ := 60  -- Number of minutes in one hour

noncomputable def carson_speed_mpm : ℝ := carson_speed_mph / minutes_per_hour -- Carson's speed in miles per minute
def carson_one_way_time : ℝ := jerry_one_way_time -- Carson's one-way time is the same as Jerry's round trip time / 2

-- Prove that the distance to the school is 2 miles.
theorem distance_to_school : carson_speed_mpm * carson_one_way_time = 2 := by
  sorry

end distance_to_school_l3_3278


namespace baron_munchausen_incorrect_l3_3333

theorem baron_munchausen_incorrect : 
  ∀ (n : ℕ) (ab : ℕ), 10 ≤ n → n ≤ 99 → 0 ≤ ab → ab ≤ 99 
  → ¬ (∃ (m : ℕ), n * 100 + ab = m * m) := 
by
  intros n ab n_lower_bound n_upper_bound ab_lower_bound ab_upper_bound
  sorry

end baron_munchausen_incorrect_l3_3333


namespace arithmetic_sequence_property_l3_3087

theorem arithmetic_sequence_property 
  (a : ℕ → ℤ) 
  (h₁ : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_sequence_property_l3_3087


namespace kims_total_points_l3_3868

theorem kims_total_points :
  let points_easy := 2
  let points_average := 3
  let points_hard := 5
  let answers_easy := 6
  let answers_average := 2
  let answers_hard := 4
  let total_points := (answers_easy * points_easy) + (answers_average * points_average) + (answers_hard * points_hard)
  total_points = 38 :=
by
  -- This is a placeholder to indicate that the proof is not included.
  sorry

end kims_total_points_l3_3868


namespace total_scoops_l3_3463

-- Definitions
def single_cone_scoops : ℕ := 1
def double_cone_scoops : ℕ := 2 * single_cone_scoops
def banana_split_scoops : ℕ := 3 * single_cone_scoops
def waffle_bowl_scoops : ℕ := banana_split_scoops + 1

-- Theorem statement: Prove the total number of scoops is 10.
theorem total_scoops : single_cone_scoops + double_cone_scoops + banana_split_scoops + waffle_bowl_scoops = 10 :=
by
  -- Proof steps would go here
  sorry

end total_scoops_l3_3463


namespace difference_high_low_score_l3_3800

theorem difference_high_low_score :
  ∀ (num_innings : ℕ) (total_runs : ℕ) (exc_total_runs : ℕ) (high_score : ℕ) (low_score : ℕ),
  num_innings = 46 →
  total_runs = 60 * 46 →
  exc_total_runs = 58 * 44 →
  high_score = 194 →
  total_runs - exc_total_runs = high_score + low_score →
  high_score - low_score = 180 :=
by
  intros num_innings total_runs exc_total_runs high_score low_score h_innings h_total h_exc_total h_high_sum h_difference
  sorry

end difference_high_low_score_l3_3800


namespace prob_rain_both_days_l3_3319

-- Declare the probabilities involved
def P_Monday : ℝ := 0.40
def P_Tuesday : ℝ := 0.30
def P_Tuesday_given_Monday : ℝ := 0.30

-- Prove the probability of it raining on both days
theorem prob_rain_both_days : P_Monday * P_Tuesday_given_Monday = 0.12 :=
by
  sorry

end prob_rain_both_days_l3_3319


namespace trapezoid_area_l3_3563

noncomputable def area_trapezoid (B1 B2 h : ℝ) : ℝ := (1 / 2 * (B1 + B2) * h)

theorem trapezoid_area
    (h1 : ∀ x : ℝ, 3 * x = 10 → x = 10 / 3)
    (h2 : ∀ x : ℝ, 3 * x = 5 → x = 5 / 3)
    (h3 : B1 = 10 / 3)
    (h4 : B2 = 5 / 3)
    (h5 : h = 5)
    : area_trapezoid B1 B2 h = 12.5 := by
  sorry

end trapezoid_area_l3_3563


namespace A_inter_B_eq_C_l3_3256

noncomputable def A : Set ℝ := { x | ∃ α β : ℤ, α ≥ 0 ∧ β ≥ 0 ∧ x = 2^α * 3^β }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 5 }
def C : Set ℝ := {1, 2, 3, 4}

theorem A_inter_B_eq_C : A ∩ B = C :=
by
  sorry

end A_inter_B_eq_C_l3_3256


namespace mass_percentage_Ba_in_BaI2_l3_3538

noncomputable def molar_mass_Ba : ℝ := 137.33
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_BaI2 : ℝ := molar_mass_Ba + 2 * molar_mass_I

theorem mass_percentage_Ba_in_BaI2 : 
  (molar_mass_Ba / molar_mass_BaI2) * 100 = 35.11 := 
  by 
    -- implementing the proof here would demonstrate that (137.33 / 391.13) * 100 = 35.11
    sorry

end mass_percentage_Ba_in_BaI2_l3_3538


namespace maple_trees_remaining_l3_3238

-- Define the initial number of maple trees in the park
def initial_maple_trees : ℝ := 9.0

-- Define the number of maple trees that will be cut down
def cut_down_maple_trees : ℝ := 2.0

-- Define the expected number of maple trees left after cutting down
def remaining_maple_trees : ℝ := 7.0

-- Theorem to prove the remaining number of maple trees is correct
theorem maple_trees_remaining :
  initial_maple_trees - cut_down_maple_trees = remaining_maple_trees := by
  admit -- sorry can be used alternatively

end maple_trees_remaining_l3_3238


namespace boat_downstream_distance_l3_3644

-- Given conditions
def speed_boat_still_water : ℕ := 25
def speed_stream : ℕ := 5
def travel_time_downstream : ℕ := 3

-- Proof statement: The distance travelled downstream is 90 km
theorem boat_downstream_distance :
  speed_boat_still_water + speed_stream * travel_time_downstream = 90 :=
by
  -- omitting the actual proof steps
  sorry

end boat_downstream_distance_l3_3644


namespace union_A_B_l3_3550

-- Definitions for the sets A and B
def A : Set ℝ := {1, 2, 3, 4}
def B : Set ℝ := {x | -1 < x ∧ x < 3}

-- The statement to be proven
theorem union_A_B :
  A ∪ B = {x | (-1 < x ∧ x ≤ 3) ∨ x = 4} :=
sorry

end union_A_B_l3_3550


namespace find_R_when_S_eq_5_l3_3264

theorem find_R_when_S_eq_5
  (g : ℚ)
  (h1 : ∀ S, R = g * S^2 - 6)
  (h2 : R = 15 ∧ S = 3) :
  R = 157 / 3 := by
    sorry

end find_R_when_S_eq_5_l3_3264


namespace parabola_translation_vertex_l3_3170

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Define the translation of the parabola
def translated_parabola (x : ℝ) : ℝ := (x + 3)^2 - 4*(x + 3) + 2 - 2 -- Adjust x + 3 for shift left and subtract 2 for shift down

-- The vertex coordinates function
def vertex_coords (f : ℝ → ℝ) (x_vertex : ℝ) : ℝ × ℝ := (x_vertex, f x_vertex)

-- Define the original vertex
def original_vertex : ℝ × ℝ := vertex_coords original_parabola 2

-- Define the translated vertex we expect
def expected_translated_vertex : ℝ × ℝ := vertex_coords translated_parabola (-1)

-- Statement of the problem
theorem parabola_translation_vertex :
  expected_translated_vertex = (-1, -4) :=
  sorry

end parabola_translation_vertex_l3_3170


namespace amount_distributed_l3_3982

theorem amount_distributed (A : ℕ) (h : A / 14 = A / 18 + 80) : A = 5040 :=
sorry

end amount_distributed_l3_3982


namespace binom_8_3_eq_56_and_2_pow_56_l3_3384

theorem binom_8_3_eq_56_and_2_pow_56 :
  (Nat.choose 8 3 = 56) ∧ (2 ^ (Nat.choose 8 3) = 2 ^ 56) :=
by
  sorry

end binom_8_3_eq_56_and_2_pow_56_l3_3384


namespace product_of_reverse_numbers_l3_3630

def reverse (n : Nat) : Nat :=
  Nat.ofDigits 10 (List.reverse (Nat.digits 10 n))

theorem product_of_reverse_numbers : 
  ∃ (a b : ℕ), a * b = 92565 ∧ b = reverse a ∧ ((a = 165 ∧ b = 561) ∨ (a = 561 ∧ b = 165)) :=
by
  sorry

end product_of_reverse_numbers_l3_3630


namespace units_digit_of_sequence_l3_3064

theorem units_digit_of_sequence : 
  (2 + 2 * 3 + 2 * 3^2 + 2 * 3^3 + 2 * 3^4 + 2 * 3^5 + 2 * 3^6 + 2 * 3^7 + 2 * 3^8 + 2 * 3^9) % 10 = 8 := 
by 
  sorry

end units_digit_of_sequence_l3_3064


namespace tony_graduate_degree_years_l3_3750

-- Define the years spent for each degree and the total time
def D1 := 4 -- years for the first degree in science
def D2 := 4 -- years for each of the two additional degrees
def T := 14 -- total years spent in school
def G := 2 -- years spent for the graduate degree in physics

-- Theorem: Given the conditions, prove that Tony spent 2 years on his graduate degree in physics
theorem tony_graduate_degree_years : 
  D1 + 2 * D2 + G = T :=
by
  sorry

end tony_graduate_degree_years_l3_3750


namespace students_spend_185_minutes_in_timeout_l3_3103

variable (tR tF tS t_total : ℕ)

-- Conditions
def running_timeouts : ℕ := 5
def food_timeouts : ℕ := 5 * running_timeouts - 1
def swearing_timeouts : ℕ := food_timeouts / 3
def total_timeouts : ℕ := running_timeouts + food_timeouts + swearing_timeouts
def timeout_duration : ℕ := 5

-- Total time spent in time-out
def total_timeout_minutes : ℕ := total_timeouts * timeout_duration

theorem students_spend_185_minutes_in_timeout :
  total_timeout_minutes = 185 :=
by
  -- The answer is directly given by the conditions and the correct answer identified.
  sorry

end students_spend_185_minutes_in_timeout_l3_3103


namespace vasya_number_l3_3734

theorem vasya_number (a b c : ℕ) (h1 : 100 ≤ 100*a + 10*b + c) (h2 : 100*a + 10*b + c < 1000) 
  (h3 : a + c = 1) (h4 : a * b = 4) (h5 : a ≠ 0) : 100*a + 10*b + c = 140 :=
by
  sorry

end vasya_number_l3_3734


namespace box_contains_1600_calories_l3_3093

theorem box_contains_1600_calories :
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  total_calories = 1600 :=
by
  let cookies_per_bag := 20
  let bags_per_box := 4
  let calories_per_cookie := 20
  let total_cookies := cookies_per_bag * bags_per_box
  let total_calories := total_cookies * calories_per_cookie
  show total_calories = 1600
  sorry

end box_contains_1600_calories_l3_3093


namespace remaining_marbles_l3_3113

theorem remaining_marbles (initial_marbles : ℕ) (num_customers : ℕ) (marble_range : List ℕ)
  (h_initial : initial_marbles = 2500)
  (h_customers : num_customers = 50)
  (h_range : marble_range = List.range' 1 50)
  (disjoint_range : ∀ (a b : ℕ), a ∈ marble_range → b ∈ marble_range → a ≠ b → a + b ≤ 50) :
  initial_marbles - (num_customers * (50 + 1) / 2) = 1225 :=
by
  sorry

end remaining_marbles_l3_3113


namespace union_sets_l3_3454

noncomputable def setA : Set ℝ := { x | x^2 - 3*x - 4 ≤ 0 }
noncomputable def setB : Set ℝ := { x | 1 < x ∧ x < 5 }

theorem union_sets :
  (setA ∪ setB) = { x | -1 ≤ x ∧ x < 5 } :=
by
  sorry

end union_sets_l3_3454


namespace age_problem_l3_3288

theorem age_problem (S Sh K : ℕ) 
  (h1 : S / Sh = 4 / 3)
  (h2 : S / K = 4 / 2)
  (h3 : K + 10 = S)
  (h4 : S + 8 = 30) :
  S = 22 ∧ Sh = 17 ∧ K = 10 := 
sorry

end age_problem_l3_3288


namespace bonus_implies_completion_l3_3939

variable (John : Type)
variable (completes_all_tasks_perfectly : John → Prop)
variable (receives_bonus : John → Prop)

theorem bonus_implies_completion :
  (∀ e : John, completes_all_tasks_perfectly e → receives_bonus e) →
  (∀ e : John, receives_bonus e → completes_all_tasks_perfectly e) :=
by
  intros h e
  sorry

end bonus_implies_completion_l3_3939


namespace largest_digit_7182N_divisible_by_6_l3_3666

noncomputable def largest_digit_divisible_by_6 : ℕ := 6

theorem largest_digit_7182N_divisible_by_6 (N : ℕ) : 
  (N % 2 = 0) ∧ ((18 + N) % 3 = 0) ↔ (N ≤ 9) ∧ (N = 6) :=
by
  sorry

end largest_digit_7182N_divisible_by_6_l3_3666


namespace trajectory_equation_l3_3906

-- Define the fixed points F1 and F2
structure Point where
  x : ℝ
  y : ℝ

def F1 : Point := ⟨-2, 0⟩
def F2 : Point := ⟨2, 0⟩

-- Define the moving point M and the condition it must satisfy
def satisfies_condition (M : Point) : Prop :=
  (Real.sqrt ((M.x + 2)^2 + M.y^2) - Real.sqrt ((M.x - 2)^2 + M.y^2)) = 4

-- The trajectory of the point M must satisfy y = 0 and x >= 2
def on_trajectory (M : Point) : Prop :=
  M.y = 0 ∧ M.x ≥ 2

-- The final theorem to be proved
theorem trajectory_equation (M : Point) (h : satisfies_condition M) : on_trajectory M := by
  sorry

end trajectory_equation_l3_3906


namespace max_expression_sum_l3_3227

open Real

theorem max_expression_sum :
  ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ 
  (2 * x^2 - 3 * x * y + 4 * y^2 = 15 ∧ 
  (3 * x^2 + 2 * x * y + y^2 = 50 * sqrt 3 + 65)) :=
sorry

#eval 65 + 50 + 3 + 1 -- this should output 119

end max_expression_sum_l3_3227


namespace quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l3_3324

open Real

-- Mathematical translations of conditions and proofs
theorem quadratic_real_roots_range_of_m (m : ℝ) (h1 : ∃ x : ℝ, x^2 + 2 * x - (m - 2) = 0) :
  m ≥ 1 := by
  sorry

theorem quadratic_root_and_other_m (h1 : (1:ℝ) ^ 2 + 2 * 1 - (m - 2) = 0) :
  m = 3 ∧ ∃ x : ℝ, (x = -3) ∧ (x^2 + 2 * x - 3 = 0) := by
  sorry

end quadratic_real_roots_range_of_m_quadratic_root_and_other_m_l3_3324


namespace total_investment_with_interest_l3_3516

def principal : ℝ := 1000
def part3Percent : ℝ := 199.99999999999983
def rate3Percent : ℝ := 0.03
def rate5Percent : ℝ := 0.05
def interest3Percent : ℝ := part3Percent * rate3Percent
def part5Percent : ℝ := principal - part3Percent
def interest5Percent : ℝ := part5Percent * rate5Percent
def totalWithInterest : ℝ := principal + interest3Percent + interest5Percent

theorem total_investment_with_interest :
  totalWithInterest = 1046.00 :=
by
  unfold totalWithInterest interest5Percent part5Percent interest3Percent
  sorry

end total_investment_with_interest_l3_3516


namespace boxes_left_l3_3565

theorem boxes_left (boxes_sat : ℕ) (boxes_sun : ℕ) (apples_per_box : ℕ) (apples_sold : ℕ)
  (h1 : boxes_sat = 50) (h2 : boxes_sun = 25) (h3 : apples_per_box = 10) (h4 : apples_sold = 720) :
  (boxes_sat * apples_per_box + boxes_sun * apples_per_box - apples_sold) / apples_per_box = 3 :=
by
  sorry

end boxes_left_l3_3565


namespace increasing_function_range_b_l3_3543

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x > 0 then (b - 3 / 2) * x + b - 1 else -x^2 + (2 - b) * x

theorem increasing_function_range_b :
  (∀ x y, x < y → f b x ≤ f b y) ↔ (3 / 2 < b ∧ b ≤ 2 ) := 
by
  sorry

end increasing_function_range_b_l3_3543


namespace solution_set_f_x_minus_2_pos_l3_3887

noncomputable def f (x : ℝ) : ℝ :=
  if x >= 0 then 2 * x - 4 else 2 * (-x) - 4

theorem solution_set_f_x_minus_2_pos :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
by
  sorry

end solution_set_f_x_minus_2_pos_l3_3887


namespace Dawn_has_10_CDs_l3_3523

-- Lean definition of the problem conditions
def Kristine_more_CDs (D K : ℕ) : Prop :=
  K = D + 7

def Total_CDs (D K : ℕ) : Prop :=
  D + K = 27

-- Lean statement of the proof
theorem Dawn_has_10_CDs (D K : ℕ) (h1 : Kristine_more_CDs D K) (h2 : Total_CDs D K) : D = 10 :=
by
  sorry

end Dawn_has_10_CDs_l3_3523


namespace monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l3_3678

/- Part 1: Monthly Average Growth Rate -/
theorem monthly_average_growth_rate (m : ℝ) (sale_april sale_june : ℝ) (h_apr_val : sale_april = 256) (h_june_val : sale_june = 400) :
  256 * (1 + m) ^ 2 = 400 → m = 0.25 :=
sorry

/- Part 2: Optimal Selling Price for Desired Profit -/
theorem optimal_selling_price_for_desired_profit (y : ℝ) (initial_price selling_price : ℝ) (sale_june : ℝ) (h_june_sale : sale_june = 400) (profit : ℝ) (h_profit : profit = 8400) :
  (y - 35) * (1560 - 20 * y) = 8400 → y = 50 :=
sorry

end monthly_average_growth_rate_optimal_selling_price_for_desired_profit_l3_3678


namespace num_men_in_boat_l3_3991

theorem num_men_in_boat 
  (n : ℕ) (W : ℝ)
  (h1 : (W / n : ℝ) = W / n)
  (h2 : (W + 8) / n = W / n + 1)
  : n = 8 := 
sorry

end num_men_in_boat_l3_3991


namespace valid_combination_exists_l3_3013

def exists_valid_combination : Prop :=
  ∃ (a: Fin 7 → ℤ), (a 0 = 1) ∧
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 4) ∧ 
  (a 4 = 5) ∧ (a 5 = 6) ∧ (a 6 = 7) ∧
  ((a 0 = a 1 + a 2 + a 3 + a 4 - a 5 - a 6))

theorem valid_combination_exists :
  exists_valid_combination :=
by
  sorry

end valid_combination_exists_l3_3013


namespace marbles_lost_correct_l3_3933

-- Define the initial number of marbles
def initial_marbles : ℕ := 16

-- Define the current number of marbles
def current_marbles : ℕ := 9

-- Define the number of marbles lost
def marbles_lost (initial current : ℕ) : ℕ := initial - current

-- State the proof problem: Given the conditions, prove the number of marbles lost is 7
theorem marbles_lost_correct : marbles_lost initial_marbles current_marbles = 7 := by
  sorry

end marbles_lost_correct_l3_3933


namespace determine_a_l3_3310

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then x ^ 2 + a else 2 ^ x

theorem determine_a (a : ℝ) (h1 : a > -1) (h2 : f a (f a (-1)) = 4) : a = 1 :=
sorry

end determine_a_l3_3310


namespace find_circle_center_l3_3026

theorem find_circle_center
  (x y : ℝ)
  (h1 : 5 * x - 4 * y = 10)
  (h2 : 3 * x - y = 0)
  : x = -10 / 7 ∧ y = -30 / 7 :=
by {
  sorry
}

end find_circle_center_l3_3026


namespace find_real_solution_to_given_equation_l3_3071

noncomputable def sqrt_96_minus_sqrt_84 : ℝ := Real.sqrt 96 - Real.sqrt 84

theorem find_real_solution_to_given_equation (x : ℝ) (hx : x + 4 ≥ 0) :
  x^2 + 6 * x + 6 * x * Real.sqrt (x + 4) = 60 ↔ x = sqrt_96_minus_sqrt_84 := 
by
  sorry

end find_real_solution_to_given_equation_l3_3071


namespace sugar_required_in_new_recipe_l3_3655

theorem sugar_required_in_new_recipe
  (ratio_flour_water_sugar : ℕ × ℕ × ℕ)
  (double_ratio_flour_water : (ℕ → ℕ))
  (half_ratio_flour_sugar : (ℕ → ℕ))
  (new_water_cups : ℕ) :
  ratio_flour_water_sugar = (7, 2, 1) →
  double_ratio_flour_water 7 = 14 → 
  double_ratio_flour_water 2 = 4 →
  half_ratio_flour_sugar 7 = 7 →
  half_ratio_flour_sugar 1 = 2 →
  new_water_cups = 2 →
  (∃ sugar_cups : ℕ, sugar_cups = 1) :=
by
  sorry

end sugar_required_in_new_recipe_l3_3655


namespace remainder_when_divided_by_7_l3_3890

theorem remainder_when_divided_by_7 (n : ℕ) (h : (2 * n) % 7 = 4) : n % 7 = 2 :=
  by sorry

end remainder_when_divided_by_7_l3_3890


namespace find_k_for_perfect_square_l3_3701

theorem find_k_for_perfect_square :
  ∃ k : ℤ, (k = 12 ∨ k = -12) ∧ (∀ n : ℤ, ∃ a b : ℤ, 4 * n^2 + k * n + 9 = (a * n + b)^2) :=
sorry

end find_k_for_perfect_square_l3_3701


namespace seashells_after_giving_away_l3_3316

-- Define the given conditions
def initial_seashells : ℕ := 79
def given_away_seashells : ℕ := 63

-- State the proof problem
theorem seashells_after_giving_away : (initial_seashells - given_away_seashells) = 16 :=
  by 
    sorry

end seashells_after_giving_away_l3_3316


namespace find_theta_l3_3853

noncomputable def P := (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4))

theorem find_theta
  (theta : ℝ)
  (h_theta_range : 0 ≤ theta ∧ theta < 2 * Real.pi)
  (h_P_theta : P = (Real.sin theta, Real.cos theta)) :
  theta = 7 * Real.pi / 4 :=
sorry

end find_theta_l3_3853


namespace max_value_of_expression_l3_3822

theorem max_value_of_expression (x y z : ℝ) (h : 0 < x) (h' : 0 < y) (h'' : 0 < z) (hxyz : x * y * z = 1) :
  (∃ s, s = x ∧ ∃ t, t = y ∧ ∃ u, u = z ∧ 
  (x^2 * y / (x + y) + y^2 * z / (y + z) + z^2 * x / (z + x) ≤ 3 / 2)) :=
sorry

end max_value_of_expression_l3_3822


namespace find_x_y_l3_3225

theorem find_x_y (x y : ℝ) 
  (h1 : 3 * x = 0.75 * y)
  (h2 : x + y = 30) : x = 6 ∧ y = 24 := 
by
  sorry  -- Proof is omitted

end find_x_y_l3_3225


namespace integral_cos_2x_eq_half_l3_3090

theorem integral_cos_2x_eq_half :
  ∫ x in (0:ℝ)..(Real.pi / 4), Real.cos (2 * x) = 1 / 2 := by
sorry

end integral_cos_2x_eq_half_l3_3090


namespace outdoor_tables_count_l3_3833

variable (numIndoorTables : ℕ) (chairsPerIndoorTable : ℕ) (totalChairs : ℕ)
variable (chairsPerOutdoorTable : ℕ)

theorem outdoor_tables_count 
  (h1 : numIndoorTables = 8) 
  (h2 : chairsPerIndoorTable = 3) 
  (h3 : totalChairs = 60) 
  (h4 : chairsPerOutdoorTable = 3) :
  ∃ (numOutdoorTables : ℕ), numOutdoorTables = 12 := by
  admit

end outdoor_tables_count_l3_3833


namespace trigonometric_identity_l3_3713

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  Real.sin α * Real.sin (3 * Real.pi / 2 - α) = -3 / 10 :=
by
  sorry

end trigonometric_identity_l3_3713


namespace function_C_is_quadratic_l3_3884

def isQuadratic (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

def function_C (x : ℝ) : ℝ := (x + 1)^2 - 5

theorem function_C_is_quadratic : isQuadratic function_C :=
by
  sorry

end function_C_is_quadratic_l3_3884


namespace range_of_a_l3_3366

noncomputable def isNotPurelyImaginary (a : ℝ) : Prop :=
  let re := a^2 - a - 2
  re ≠ 0

theorem range_of_a (a : ℝ) (h : isNotPurelyImaginary a) : a ≠ -1 :=
  sorry

end range_of_a_l3_3366


namespace percent_of_a_l3_3260

theorem percent_of_a (a b : ℝ) (h : a = 1.2 * b) : 4 * b = (10 / 3) * a :=
sorry

end percent_of_a_l3_3260


namespace find_d_q_l3_3575

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

noncomputable def b_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  b1 * q^(n - 1)

-- Sum of the first n terms of an arithmetic sequence
noncomputable def S_n (a1 d : ℕ) (n : ℕ) : ℕ :=
  n * a1 + (n * (n - 1) / 2) * d

-- Sum of the first n terms of a geometric sequence
noncomputable def T_n (b1 q : ℕ) (n : ℕ) : ℕ :=
  if q = 1 then n * b1
  else b1 * (1 - q^n) / (1 - q)

theorem find_d_q (a1 b1 d q : ℕ) (h1 : ∀ n : ℕ, n > 0 →
  n^2 * (T_n b1 q n + 1) = 2^n * S_n a1 d n) : d = 2 ∧ q = 2 :=
by
  sorry

end find_d_q_l3_3575


namespace cattle_train_speed_is_56_l3_3541

variable (v : ℝ)

def cattle_train_speed :=
  let cattle_distance_until_diesel_starts := 6 * v
  let diesel_speed := v - 33
  let diesel_distance := 12 * diesel_speed
  let cattle_additional_distance := 12 * v
  let total_distance := cattle_distance_until_diesel_starts + diesel_distance + cattle_additional_distance
  total_distance = 1284

theorem cattle_train_speed_is_56 (h : cattle_train_speed v) : v = 56 :=
  sorry

end cattle_train_speed_is_56_l3_3541


namespace jellyfish_cost_l3_3945

theorem jellyfish_cost (J E : ℝ) (h1 : E = 9 * J) (h2 : J + E = 200) : J = 20 := by
  sorry

end jellyfish_cost_l3_3945


namespace omino_tilings_2_by_10_l3_3061

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n+2) => fib n + fib (n+1)

def omino_tilings (n : ℕ) : ℕ :=
  fib (n + 1)

theorem omino_tilings_2_by_10 : omino_tilings 10 = 3025 := by
  sorry

end omino_tilings_2_by_10_l3_3061


namespace eel_species_count_l3_3205

theorem eel_species_count (sharks eels whales total : ℕ)
    (h_sharks : sharks = 35)
    (h_whales : whales = 5)
    (h_total : total = 55)
    (h_species_sum : sharks + eels + whales = total) : eels = 15 :=
by
  -- Proof goes here
  sorry

end eel_species_count_l3_3205


namespace fraction_paint_remaining_l3_3567

theorem fraction_paint_remaining 
  (original_paint : ℝ)
  (h_original : original_paint = 2) 
  (used_first_day : ℝ)
  (h_used_first_day : used_first_day = (1 / 4) * original_paint) 
  (remaining_after_first : ℝ)
  (h_remaining_first : remaining_after_first = original_paint - used_first_day) 
  (used_second_day : ℝ)
  (h_used_second_day : used_second_day = (1 / 3) * remaining_after_first) 
  (remaining_after_second : ℝ)
  (h_remaining_second : remaining_after_second = remaining_after_first - used_second_day) : 
  remaining_after_second / original_paint = 1 / 2 :=
by
  -- Proof goes here.
  sorry

end fraction_paint_remaining_l3_3567


namespace negation_of_P_l3_3217

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- Define the negation of P
def not_P : Prop := ∀ x : ℝ, Real.exp x - x - 1 > 0

-- The theorem statement
theorem negation_of_P : ¬ P = not_P :=
by
  sorry

end negation_of_P_l3_3217


namespace find_k_l3_3351

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem find_k (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 :=
by
  sorry

end find_k_l3_3351


namespace rational_m_abs_nonneg_l3_3816

theorem rational_m_abs_nonneg (m : ℚ) : m + |m| ≥ 0 :=
by sorry

end rational_m_abs_nonneg_l3_3816


namespace first_alloy_mass_l3_3054

theorem first_alloy_mass (x : ℝ) : 
  (0.12 * x + 2.8) / (x + 35) = 9.454545454545453 / 100 → 
  x = 20 :=
by
  intro h
  sorry

end first_alloy_mass_l3_3054


namespace sin_alpha_eq_sqrt_5_div_3_l3_3299

variable (α : ℝ)

theorem sin_alpha_eq_sqrt_5_div_3
  (hα : 0 < α ∧ α < Real.pi)
  (h : 3 * Real.cos (2 * α) - 8 * Real.cos α = 5) :
  Real.sin α = Real.sqrt 5 / 3 := 
by 
  sorry

end sin_alpha_eq_sqrt_5_div_3_l3_3299


namespace sum_valid_two_digit_integers_l3_3949

theorem sum_valid_two_digit_integers :
  ∃ S : ℕ, S = 36 ∧ (∀ n, 10 ≤ n ∧ n < 100 →
    (∃ a b, n = 10 * a + b ∧ a + b ∣ n ∧ 2 * a * b ∣ n → n = 36)) :=
by
  sorry

end sum_valid_two_digit_integers_l3_3949


namespace integer_values_of_n_satisfy_inequality_l3_3208

theorem integer_values_of_n_satisfy_inequality :
  ∃ S : Finset ℤ, (∀ n ∈ S, -100 < n^3 ∧ n^3 < 100) ∧ S.card = 9 :=
by
  -- Sorry provides the placeholder for where the proof would go
  sorry

end integer_values_of_n_satisfy_inequality_l3_3208


namespace average_sales_is_167_5_l3_3726

def sales_january : ℝ := 150
def sales_february : ℝ := 90
def sales_march : ℝ := 1.5 * sales_february
def sales_april : ℝ := 180
def sales_may : ℝ := 210
def sales_june : ℝ := 240
def total_sales : ℝ := sales_january + sales_february + sales_march + sales_april + sales_may + sales_june
def number_of_months : ℝ := 6

theorem average_sales_is_167_5 :
  total_sales / number_of_months = 167.5 :=
sorry

end average_sales_is_167_5_l3_3726


namespace gary_money_left_l3_3222

variable (initialAmount : Nat)
variable (amountSpent : Nat)

theorem gary_money_left (h1 : initialAmount = 73) (h2 : amountSpent = 55) : initialAmount - amountSpent = 18 :=
by
  sorry

end gary_money_left_l3_3222


namespace number_of_pairs_is_2_pow_14_l3_3837

noncomputable def number_of_pairs_satisfying_conditions : ℕ :=
  let fact5 := Nat.factorial 5
  let fact50 := Nat.factorial 50
  Nat.card {p : ℕ × ℕ | Nat.gcd p.1 p.2 = fact5 ∧ Nat.lcm p.1 p.2 = fact50}

theorem number_of_pairs_is_2_pow_14 :
  number_of_pairs_satisfying_conditions = 2^14 := by
  sorry

end number_of_pairs_is_2_pow_14_l3_3837


namespace solve_equation_l3_3985

theorem solve_equation (a b : ℚ) : 
  ((b = 0) → false) ∧ 
  ((4 * a - 3 = 0) → ((5 * b - 1 = 0) → a = 3 / 4 ∧ b = 1 / 5)) ∧ 
  ((4 * a - 3 ≠ 0) → (∃ x : ℚ, x = (5 * b - 1) / (4 * a - 3))) :=
by
  sorry

end solve_equation_l3_3985


namespace lines_perpendicular_to_same_plane_are_parallel_l3_3564

variables {Point Line Plane : Type*}
variables [MetricSpace Point] [LinearOrder Line]

def line_parallel_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def line_perpendicular_to_plane (a : Line) (M : Plane) : Prop := sorry -- Define the formal condition
def lines_parallel (a b : Line) : Prop := sorry -- Define the formal condition

theorem lines_perpendicular_to_same_plane_are_parallel 
  (a b : Line) (M : Plane) 
  (h₁ : line_perpendicular_to_plane a M) 
  (h₂ : line_perpendicular_to_plane b M) : 
  lines_parallel a b :=
sorry

end lines_perpendicular_to_same_plane_are_parallel_l3_3564


namespace average_of_five_quantities_l3_3434

theorem average_of_five_quantities (a b c d e : ℝ) 
  (h1 : (a + b + c) / 3 = 4) 
  (h2 : (d + e) / 2 = 33) : 
  ((a + b + c + d + e) / 5) = 15.6 := 
sorry

end average_of_five_quantities_l3_3434


namespace max_minus_min_l3_3793

noncomputable def f (x : ℝ) := if x > 0 then (x - 1) ^ 2 else (x + 1) ^ 2

theorem max_minus_min (n m : ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ (-1 / 2) → n ≤ f x ∧ f x ≤ m) →
  m - n = 1 :=
by { sorry }

end max_minus_min_l3_3793


namespace batsman_average_after_25th_innings_l3_3797

theorem batsman_average_after_25th_innings :
  ∃ A : ℝ, 
    (∀ s : ℝ, s = 25 * A + 62.5 → 24 * A + 95 = s) →
    A + 2.5 = 35 :=
by
  sorry

end batsman_average_after_25th_innings_l3_3797


namespace largest_element_in_A_inter_B_l3_3272

def A : Set ℕ := { n | 1 ≤ n ∧ n ≤ 2023 }
def B : Set ℕ := { n | ∃ k : ℤ, n = 3 * k + 2 ∧ n > 0 }

theorem largest_element_in_A_inter_B : ∃ x ∈ (A ∩ B), ∀ y ∈ (A ∩ B), y ≤ x ∧ x = 2021 := by
  sorry

end largest_element_in_A_inter_B_l3_3272


namespace quadratic_inequality_l3_3632

theorem quadratic_inequality (x : ℝ) (h : x^2 - 8 * x + 12 < 0) : 2 < x ∧ x < 6 :=
sorry

end quadratic_inequality_l3_3632


namespace min_value_of_fraction_l3_3676

theorem min_value_of_fraction (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) : 
  (∀ n m, 2 * n + m = 4 ∧ m > 0 ∧ n > 0 → ∀ y, y = 2 / m + 1 / n → y ≥ 2) :=
by sorry

end min_value_of_fraction_l3_3676


namespace lower_percentage_increase_l3_3458

theorem lower_percentage_increase (E P : ℝ) (h1 : 1.26 * E = 693) (h2 : (1 + P) * E = 660) : P = 0.2 := by
  sorry

end lower_percentage_increase_l3_3458


namespace like_terms_exponents_product_l3_3338

theorem like_terms_exponents_product (m n : ℤ) (a b : ℝ) 
  (h1 : 3 * a^m * b^2 = -1 * a^2 * b^(n+3)) : m * n = -2 :=
  sorry

end like_terms_exponents_product_l3_3338


namespace series_converges_l3_3857

theorem series_converges :
  ∑' n, (2^n) / (3^(2^n) + 1) = 1 / 2 :=
sorry

end series_converges_l3_3857


namespace base_of_exponent_l3_3617

theorem base_of_exponent (b x y : ℕ) (h1 : x - y = 12) (h2 : x = 12) (h3 : b^x * 4^y = 531441) : b = 3 :=
by
  sorry

end base_of_exponent_l3_3617


namespace sum_of_ages_l3_3669

variable (S F : ℕ)

theorem sum_of_ages (h1 : F - 18 = 3 * (S - 18)) (h2 : F = 2 * S) : S + F = 108 := by
  sorry

end sum_of_ages_l3_3669


namespace sum_of_six_digits_is_31_l3_3481

-- Problem constants and definitions
def digits : Set ℕ := {0, 2, 3, 4, 5, 7, 8, 9}

-- Problem conditions expressed as hypotheses
variables (a b c d e f g : ℕ)
variables (h1 : a ∈ digits) (h2 : b ∈ digits) (h3 : c ∈ digits) 
          (h4 : d ∈ digits) (h5 : e ∈ digits) (h6 : f ∈ digits) (h7 : g ∈ digits)
          (h8 : a ≠ b) (h9 : a ≠ c) (h10 : a ≠ d) (h11 : a ≠ e) (h12 : a ≠ f) (h13 : a ≠ g)
          (h14 : b ≠ c) (h15 : b ≠ d) (h16 : b ≠ e) (h17 : b ≠ f) (h18 : b ≠ g)
          (h19 : c ≠ d) (h20 : c ≠ e) (h21 : c ≠ f) (h22 : c ≠ g)
          (h23 : d ≠ e) (h24 : d ≠ f) (h25 : d ≠ g)
          (h26 : e ≠ f) (h27 : e ≠ g) (h28 : f ≠ g)
variable (shared : b = e)
variables (h29 : a + b + c = 24) (h30 : d + e + f + g = 14)

-- Proposition to be proved
theorem sum_of_six_digits_is_31 : a + b + c + d + e + f = 31 :=
by 
  sorry

end sum_of_six_digits_is_31_l3_3481


namespace f_at_3_l3_3805

variable {R : Type} [LinearOrderedField R]

-- Define odd function
def is_odd_function (f : R → R) := ∀ x : R, f (-x) = -f x

-- Define the given function f and its properties
variables (f : R → R)
  (h_odd : is_odd_function f)
  (h_domain : ∀ x : R, true) -- domain is R implicitly
  (h_eq : ∀ x : R, f x + f (2 - x) = 4)

-- Prove that f(3) = 6
theorem f_at_3 : f 3 = 6 :=
  sorry

end f_at_3_l3_3805


namespace b_distance_behind_proof_l3_3334

-- Given conditions
def race_distance : ℕ := 1000
def a_time : ℕ := 40
def b_delay : ℕ := 10

def a_speed : ℕ := race_distance / a_time
def b_distance_behind : ℕ := a_speed * b_delay

theorem b_distance_behind_proof : b_distance_behind = 250 := by
  -- Prove that b_distance_behind = 250
  sorry

end b_distance_behind_proof_l3_3334


namespace price_difference_proof_l3_3753

theorem price_difference_proof (y : ℝ) (n : ℕ) :
  ∃ n : ℕ, (4.20 + 0.45 * n) = (6.30 + 0.01 * y * n + 0.65) → 
  n = (275 / (45 - y)) :=
by
  sorry

end price_difference_proof_l3_3753


namespace find_constant_t_l3_3328

theorem find_constant_t :
  (exists t : ℚ,
  ∀ x : ℚ,
    (5 * x ^ 2 - 6 * x + 7) * (4 * x ^ 2 + t * x + 10) =
      20 * x ^ 4 - 48 * x ^ 3 + 114 * x ^ 2 - 102 * x + 70) :=
sorry

end find_constant_t_l3_3328


namespace initial_bottles_count_l3_3899

theorem initial_bottles_count : 
  ∀ (jason_buys harry_buys bottles_left initial_bottles : ℕ), 
  jason_buys = 5 → 
  harry_buys = 6 → 
  bottles_left = 24 → 
  initial_bottles = bottles_left + jason_buys + harry_buys → 
  initial_bottles = 35 :=
by
  intros jason_buys harry_buys bottles_left initial_bottles
  intro h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end initial_bottles_count_l3_3899


namespace M_inter_N_empty_l3_3392

def M : Set ℝ := {a : ℝ | (1 / 2 < a ∧ a < 1) ∨ (1 < a)}
def N : Set ℝ := {a : ℝ | 0 < a ∧ a ≤ 1 / 2}

theorem M_inter_N_empty : M ∩ N = ∅ :=
sorry

end M_inter_N_empty_l3_3392


namespace power_of_two_l3_3141

theorem power_of_two (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_prime : Prime (m^(4^n + 1) - 1)) : 
  ∃ t : ℕ, n = 2^t :=
sorry

end power_of_two_l3_3141


namespace factorization_of_difference_of_squares_l3_3497

variable {R : Type} [CommRing R]

theorem factorization_of_difference_of_squares (m : R) : m^2 - 4 = (m + 2) * (m - 2) :=
by sorry

end factorization_of_difference_of_squares_l3_3497


namespace rachels_game_final_configurations_l3_3600

-- Define the number of cells in the grid
def n : ℕ := 2011

-- Define the number of moves needed
def moves_needed : ℕ := n - 3

-- Define a function that counts the number of distinct final configurations
-- based on the number of fights (f) possible in the given moves.
def final_configurations : ℕ := moves_needed + 1

theorem rachels_game_final_configurations : final_configurations = 2009 :=
by
  -- Calculation shows that moves_needed = 2008 and therefore final_configurations = 2008 + 1 = 2009.
  sorry

end rachels_game_final_configurations_l3_3600


namespace meal_combinations_l3_3642

theorem meal_combinations (n : ℕ) (h : n = 12) : ∃ m : ℕ, m = 132 :=
by
  -- Initialize the variables for dishes chosen by Yann and Camille
  let yann_choices := n
  let camille_choices := n - 1
  
  -- Calculate the total number of combinations
  let total_combinations := yann_choices * camille_choices
  
  -- Assert the number of combinations is equal to 132
  use total_combinations
  exact sorry

end meal_combinations_l3_3642


namespace barycentric_vector_identity_l3_3239

variables {A B C X : Type} [AddCommGroup X] [Module ℝ X]
variables (α β γ : ℝ) (A B C X : X)

-- Defining the barycentric coordinates condition
axiom barycentric_coords : α • A + β • B + γ • C = X

-- Additional condition that sum of coordinates is 1
axiom sum_coords : α + β + γ = 1

-- The theorem to prove
theorem barycentric_vector_identity :
  (X - A) = β • (B - A) + γ • (C - A) :=
sorry

end barycentric_vector_identity_l3_3239


namespace min_neg_condition_l3_3318

theorem min_neg_condition (a : ℝ) (x : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) → a < -7 :=
sorry

end min_neg_condition_l3_3318


namespace jumping_contest_l3_3448

theorem jumping_contest (grasshopper_jump frog_jump : ℕ) (h_grasshopper : grasshopper_jump = 9) (h_frog : frog_jump = 12) : frog_jump - grasshopper_jump = 3 := by
  ----- h_grasshopper and h_frog are our conditions -----
  ----- The goal is to prove frog_jump - grasshopper_jump = 3 -----
  sorry

end jumping_contest_l3_3448


namespace worth_of_each_gift_l3_3038

def workers_per_block : Nat := 200
def total_amount_for_gifts : Nat := 6000
def number_of_blocks : Nat := 15

theorem worth_of_each_gift (workers_per_block : Nat) (total_amount_for_gifts : Nat) (number_of_blocks : Nat) : 
  (total_amount_for_gifts / (workers_per_block * number_of_blocks)) = 2 := 
by 
  sorry

end worth_of_each_gift_l3_3038


namespace total_points_team_l3_3636

def T : ℕ := 4
def J : ℕ := 2 * T + 6
def S : ℕ := J / 2
def R : ℕ := T + J - 3
def A : ℕ := S + R + 4

theorem total_points_team : T + J + S + R + A = 66 := by
  sorry

end total_points_team_l3_3636


namespace solve_equation_l3_3028

theorem solve_equation : ∃ x : ℝ, 3 * x + 2 * (x - 2) = 6 ↔ x = 2 :=
by
  sorry

end solve_equation_l3_3028


namespace custom_star_calc_l3_3825

-- defining the custom operation "*"
def custom_star (a b : ℤ) : ℤ :=
  a * b - (b-1) * b

-- providing the theorem statement
theorem custom_star_calc : custom_star 2 (-3) = -18 :=
  sorry

end custom_star_calc_l3_3825


namespace plane_crash_probabilities_eq_l3_3486

noncomputable def crashing_probability_3_engines (p : ℝ) : ℝ :=
  3 * p^2 * (1 - p) + p^3

noncomputable def crashing_probability_5_engines (p : ℝ) : ℝ :=
  10 * p^3 * (1 - p)^2 + 5 * p^4 * (1 - p) + p^5

theorem plane_crash_probabilities_eq (p : ℝ) :
  crashing_probability_3_engines p = crashing_probability_5_engines p ↔ p = 0 ∨ p = 1/2 ∨ p = 1 :=
by
  sorry

end plane_crash_probabilities_eq_l3_3486


namespace digit_B_divisibility_l3_3799

theorem digit_B_divisibility :
  ∃ B : ℕ, B < 10 ∧
    (∃ n : ℕ, 658274 * 10 + B = 2 * n) ∧
    (∃ m : ℕ, 6582740 + B = 4 * m) ∧
    (B = 0 ∨ B = 5) ∧
    (∃ k : ℕ, 658274 * 10 + B = 7 * k) ∧
    (∃ p : ℕ, 6582740 + B = 8 * p) :=
sorry

end digit_B_divisibility_l3_3799


namespace solve_eq_l3_3629

theorem solve_eq : ∀ x : ℝ, -2 * (x - 1) = 4 → x = -1 := 
by
  intro x
  intro h
  sorry

end solve_eq_l3_3629


namespace find_a_b_max_profit_allocation_l3_3145

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) := (a * Real.log x) / x + 5 / x - b

theorem find_a_b :
  (∃ (a b : ℝ), f 1 a b = 5 ∧ f 10 a b = 16.515) :=
sorry

noncomputable def g (x : ℝ) := 2 * Real.sqrt x / x

noncomputable def profit (x : ℝ) := x * (5 * Real.log x / x + 5 / x) + (50 - x) * (2 * Real.sqrt (50 - x) / (50 - x))

theorem max_profit_allocation :
  (∃ (x : ℝ), 10 ≤ x ∧ x ≤ 40 ∧ ∀ y, (10 ≤ y ∧ y ≤ 40) → profit x ≥ profit y)
  ∧ profit 25 = 31.09 :=
sorry

end find_a_b_max_profit_allocation_l3_3145


namespace triangle_is_isosceles_right_triangle_l3_3425

theorem triangle_is_isosceles_right_triangle
  (a b c : ℝ)
  (h1 : (a - b)^2 + (Real.sqrt (2 * a - b - 3)) + (abs (c - 3 * Real.sqrt 2)) = 0) :
  (a = 3) ∧ (b = 3) ∧ (c = 3 * Real.sqrt 2) :=
by
  sorry

end triangle_is_isosceles_right_triangle_l3_3425


namespace part1_solution_part2_solution_l3_3544

-- Proof problem for Part (1)
theorem part1_solution (x : ℝ) (a : ℝ) (h : a = 1) : 
  (|x - 1| + |x + 3|) ≥ 6 ↔ (x ≤ -4) ∨ (x ≥ 2) :=
by
  sorry

-- Proof problem for Part (2)
theorem part2_solution (a : ℝ) : 
  (∀ x : ℝ, (|x - a| + |x + 3|) > -a) ↔ a > -3 / 2 :=
by
  sorry

end part1_solution_part2_solution_l3_3544


namespace no_tiling_10x10_1x4_l3_3168

-- Define the problem using the given conditions
def checkerboard_tiling (n k : ℕ) : Prop :=
  ∃ t : ℕ, t * k = n * n ∧ n % k = 0

-- Prove that it is impossible to tile a 10x10 board with 1x4 tiles
theorem no_tiling_10x10_1x4 : ¬ checkerboard_tiling 10 4 :=
sorry

end no_tiling_10x10_1x4_l3_3168


namespace petya_series_sum_l3_3952

theorem petya_series_sum (n k : ℕ) (h1 : (n + k) * (k + 1) = 20 * (n + 2 * k)) 
                                      (h2 : (n + k) * (k + 1) = 60 * n) :
  n = 29 ∧ k = 29 :=
by
  sorry

end petya_series_sum_l3_3952


namespace inequality_solution_l3_3609

theorem inequality_solution (x : ℝ) : 
  (x / (x + 5) ≥ 0) ↔ (x ∈ (Set.Iio (-5)).union (Set.Ici 0)) :=
by
  sorry

end inequality_solution_l3_3609


namespace solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l3_3332

-- Problem 1: Solution Set of the Inequality
theorem solution_set_x2_minus_5x_plus_4 : 
  {x : ℝ | x^2 - 5 * x + 4 > 0} = {x : ℝ | x < 1 ∨ x > 4} :=
sorry

-- Problem 2: Range of Values for a
theorem range_of_a_if_x2_plus_ax_plus_4_gt_0 (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + 4 > 0) :
  -4 < a ∧ a < 4 :=
sorry

end solution_set_x2_minus_5x_plus_4_range_of_a_if_x2_plus_ax_plus_4_gt_0_l3_3332


namespace distance_difference_l3_3353

-- Given conditions
def speed_train1 : ℕ := 20
def speed_train2 : ℕ := 25
def total_distance : ℕ := 675

-- Define the problem statement
theorem distance_difference : ∃ t : ℝ, (speed_train2 * t - speed_train1 * t) = 75 ∧ (speed_train1 * t + speed_train2 * t) = total_distance := by 
  sorry

end distance_difference_l3_3353


namespace right_triangle_hypotenuse_l3_3595

def is_nat (n : ℕ) : Prop := n > 0

theorem right_triangle_hypotenuse (x : ℕ) (x_pos : is_nat x) (consec : x + 1 > x) (h : 11^2 + x^2 = (x + 1)^2) : x + 1 = 61 :=
by
  sorry

end right_triangle_hypotenuse_l3_3595


namespace alexander_eq_alice_l3_3397

-- Definitions and conditions
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25
def sales_tax_rate : ℝ := 0.07

-- Calculation functions for Alexander and Alice
def alexander_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let taxed_price := price * (1 + tax)
  let discounted_price := taxed_price * (1 - discount)
  discounted_price

def alice_total (price : ℝ) (discount : ℝ) (tax : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  taxed_price

-- Proof that the difference between Alexander's and Alice's total is 0
theorem alexander_eq_alice : 
  alexander_total original_price discount_rate sales_tax_rate = 
  alice_total original_price discount_rate sales_tax_rate :=
by
  sorry

end alexander_eq_alice_l3_3397


namespace reduced_price_per_kg_l3_3820

variable (P : ℝ)
variable (R : ℝ)
variable (Q : ℝ)

theorem reduced_price_per_kg
  (h1 : R = 0.75 * P)
  (h2 : 500 = Q * P)
  (h3 : 500 = (Q + 5) * R)
  (h4 : Q = 500 / P) :
  R = 25 :=
by
  sorry

end reduced_price_per_kg_l3_3820


namespace right_triangle_30_60_90_l3_3412

theorem right_triangle_30_60_90 (a b : ℝ) (h : a = 15) :
  (b = 30) ∧ (b = 15 * Real.sqrt 3) :=
by
  sorry

end right_triangle_30_60_90_l3_3412


namespace gcd_884_1071_l3_3363

theorem gcd_884_1071 : Nat.gcd 884 1071 = 17 := by
  sorry

end gcd_884_1071_l3_3363


namespace fill_cistern_time_l3_3526

theorem fill_cistern_time (F E : ℝ) (hF : F = 1/2) (hE : E = 1/4) : 
  (1 / (F - E)) = 4 :=
by
  -- Definitions of F and E are used as hypotheses hF and hE
  -- Prove the actual theorem stating the time to fill the cistern is 4 hours
  sorry

end fill_cistern_time_l3_3526


namespace incorrect_operation_l3_3971

theorem incorrect_operation (a : ℝ) : ¬ (a^3 + a^3 = 2 * a^6) :=
by
  sorry

end incorrect_operation_l3_3971


namespace length_of_BC_l3_3428

theorem length_of_BC (a : ℝ) (b_x b_y c_x c_y area : ℝ) 
  (h1 : b_y = b_x ^ 2)
  (h2 : c_y = c_x ^ 2)
  (h3 : b_y = c_y)
  (h4 : area = 64) :
  c_x - b_x = 8 := by
sorry

end length_of_BC_l3_3428


namespace range_of_a_l3_3247

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 * x^2 + (a - 1) * x + 1/2 > 0) ↔ (-1 < a ∧ a < 3) := 
sorry

end range_of_a_l3_3247


namespace sugar_recipes_l3_3176

theorem sugar_recipes (container_sugar recipe_sugar : ℚ) 
  (h1 : container_sugar = 56 / 3) 
  (h2 : recipe_sugar = 3 / 2) :
  container_sugar / recipe_sugar = 112 / 9 := sorry

end sugar_recipes_l3_3176


namespace profit_calculation_correct_l3_3811

def main_actor_fee : ℕ := 500
def supporting_actor_fee : ℕ := 100
def extra_fee : ℕ := 50
def main_actor_food : ℕ := 10
def supporting_actor_food : ℕ := 5
def remaining_member_food : ℕ := 3
def post_production_cost : ℕ := 850
def revenue : ℕ := 10000

def total_actor_fees : ℕ := 2 * main_actor_fee + 3 * supporting_actor_fee + extra_fee
def total_food_cost : ℕ := 2 * main_actor_food + 4 * supporting_actor_food + 44 * remaining_member_food
def total_equipment_rental : ℕ := 2 * (total_actor_fees + total_food_cost)
def total_cost : ℕ := total_actor_fees + total_food_cost + total_equipment_rental + post_production_cost
def profit : ℕ := revenue - total_cost

theorem profit_calculation_correct : profit = 4584 :=
by
  -- proof omitted
  sorry

end profit_calculation_correct_l3_3811


namespace two_talents_students_l3_3390

-- Definitions and conditions
def total_students : ℕ := 120
def cannot_sing : ℕ := 50
def cannot_dance : ℕ := 75
def cannot_act : ℕ := 35

-- Definitions based on conditions
def can_sing : ℕ := total_students - cannot_sing
def can_dance : ℕ := total_students - cannot_dance
def can_act : ℕ := total_students - cannot_act

-- The main theorem statement
theorem two_talents_students : can_sing + can_dance + can_act - total_students = 80 :=
by
  -- substituting actual numbers to prove directly
  have h_can_sing : can_sing = 70 := rfl
  have h_can_dance : can_dance = 45 := rfl
  have h_can_act : can_act = 85 := rfl
  sorry

end two_talents_students_l3_3390


namespace not_odd_iff_exists_ne_l3_3046

open Function

variable {f : ℝ → ℝ}

theorem not_odd_iff_exists_ne : (∃ x : ℝ, f (-x) ≠ -f x) ↔ ¬ (∀ x : ℝ, f (-x) = -f x) :=
by
  sorry

end not_odd_iff_exists_ne_l3_3046


namespace goldie_worked_hours_last_week_l3_3409

variable (H : ℕ)
variable (money_per_hour : ℕ := 5)
variable (hours_this_week : ℕ := 30)
variable (total_earnings : ℕ := 250)

theorem goldie_worked_hours_last_week :
  H = (total_earnings - hours_this_week * money_per_hour) / money_per_hour :=
sorry

end goldie_worked_hours_last_week_l3_3409


namespace ji_hoon_original_answer_l3_3812

-- Define the conditions: Ji-hoon's mistake
def ji_hoon_mistake (x : ℝ) := x - 7 = 0.45

-- The theorem statement
theorem ji_hoon_original_answer (x : ℝ) (h : ji_hoon_mistake x) : x * 7 = 52.15 :=
by
  sorry

end ji_hoon_original_answer_l3_3812


namespace find_f_ln2_l3_3436

variable (f : ℝ → ℝ)

-- Condition: f is an odd function
axiom odd_fn : ∀ x : ℝ, f (-x) = -f x

-- Condition: f(x) = e^(-x) - 2 for x < 0
axiom def_fn : ∀ x : ℝ, x < 0 → f x = Real.exp (-x) - 2

-- Problem: Find f(ln 2)
theorem find_f_ln2 : f (Real.log 2) = 0 := by
  sorry

end find_f_ln2_l3_3436


namespace arithmetic_sequence_problem_l3_3748

theorem arithmetic_sequence_problem (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) -- condition for arithmetic sequence
  (h_condition : a 3 + a 5 + a 7 + a 9 + a 11 = 100) : 
  3 * a 9 - a 13 = 40 :=
sorry

end arithmetic_sequence_problem_l3_3748


namespace problem_1_problem_2_l3_3341

noncomputable def f (x : ℝ) : ℝ := x / Real.log x

theorem problem_1 (h₁ : ∀ x, x > 0 → x ≠ 1 → f x = x / Real.log x) :
  (∀ x, 1 < x ∧ x < Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) ∧
  (∀ x, x > Real.exp 1 → (Real.log x - 1) / (Real.log x * Real.log x) > 0) :=
sorry

theorem problem_2 (h₁ : f x₁ = 1) (h₂ : f x₂ = 1) (h₃ : x₁ ≠ x₂) (h₄ : x₁ > 0) (h₅ : x₂ > 0):
  x₁ + x₂ > 2 * Real.exp 1 :=
sorry

end problem_1_problem_2_l3_3341


namespace fish_caught_in_second_catch_l3_3984

theorem fish_caught_in_second_catch {N x : ℕ} (hN : N = 1750) (hx1 : 70 * x = 2 * N) : x = 50 :=
by
  sorry

end fish_caught_in_second_catch_l3_3984


namespace completing_square_16x2_32x_512_eq_33_l3_3751

theorem completing_square_16x2_32x_512_eq_33:
  (∃ p q : ℝ, (16 * x ^ 2 + 32 * x - 512 = 0) → (x + p) ^ 2 = q ∧ q = 33) :=
by
  sorry

end completing_square_16x2_32x_512_eq_33_l3_3751


namespace chantel_bracelets_at_end_l3_3677

-- Definitions based on conditions
def bracelets_day1 := 4
def days1 := 7
def given_away1 := 8

def bracelets_day2 := 5
def days2 := 10
def given_away2 := 12

-- Computation based on conditions
def total_bracelets := days1 * bracelets_day1 - given_away1 + days2 * bracelets_day2 - given_away2

-- The proof statement
theorem chantel_bracelets_at_end : total_bracelets = 58 := by
  sorry

end chantel_bracelets_at_end_l3_3677


namespace true_statements_proved_l3_3802

-- Conditions
def A : Prop := ∃ n : ℕ, 25 = 5 * n
def B : Prop := (∃ m1 : ℕ, 209 = 19 * m1) ∧ (¬ ∃ m2 : ℕ, 63 = 19 * m2)
def C : Prop := (¬ ∃ k1 : ℕ, 90 = 30 * k1) ∧ (¬ ∃ k2 : ℕ, 49 = 30 * k2)
def D : Prop := (∃ l1 : ℕ, 34 = 17 * l1) ∧ (¬ ∃ l2 : ℕ, 68 = 17 * l2)
def E : Prop := ∃ q : ℕ, 140 = 7 * q

-- Correct statements
def TrueStatements : Prop := A ∧ B ∧ E ∧ ¬C ∧ ¬D

-- Lean statement to prove
theorem true_statements_proved : TrueStatements := 
by
  sorry

end true_statements_proved_l3_3802


namespace cost_price_of_one_toy_l3_3354

theorem cost_price_of_one_toy (C : ℝ) (h : 21 * C = 21000) : C = 1000 :=
by sorry

end cost_price_of_one_toy_l3_3354


namespace genevieve_errors_fixed_l3_3024

theorem genevieve_errors_fixed (total_lines : ℕ) (lines_per_debug : ℕ) (errors_per_debug : ℕ)
  (h_total_lines : total_lines = 4300)
  (h_lines_per_debug : lines_per_debug = 100)
  (h_errors_per_debug : errors_per_debug = 3) :
  (total_lines / lines_per_debug) * errors_per_debug = 129 :=
by
  -- Placeholder proof to indicate the theorem should be true
  sorry

end genevieve_errors_fixed_l3_3024


namespace dr_reeds_statement_l3_3806

variables (P Q : Prop)

theorem dr_reeds_statement (h : P → Q) : ¬Q → ¬P :=
by sorry

end dr_reeds_statement_l3_3806


namespace number_of_books_about_trains_l3_3670

theorem number_of_books_about_trains
  (books_animals : ℕ)
  (books_outer_space : ℕ)
  (book_cost : ℕ)
  (total_spent : ℕ)
  (T : ℕ)
  (hyp1 : books_animals = 8)
  (hyp2 : books_outer_space = 6)
  (hyp3 : book_cost = 6)
  (hyp4 : total_spent = 102)
  (hyp5 : total_spent = (books_animals + books_outer_space + T) * book_cost)
  : T = 3 := by
  sorry

end number_of_books_about_trains_l3_3670


namespace total_price_of_books_l3_3980

theorem total_price_of_books (total_books: ℕ) (math_books: ℕ) (math_book_cost: ℕ) (history_book_cost: ℕ) (price: ℕ) 
  (h1 : total_books = 90) 
  (h2 : math_books = 54) 
  (h3 : math_book_cost = 4) 
  (h4 : history_book_cost = 5)
  (h5 : price = 396) :
  let history_books := total_books - math_books
  let math_books_price := math_books * math_book_cost
  let history_books_price := history_books * history_book_cost
  let total_price := math_books_price + history_books_price
  total_price = price := 
  by
    sorry

end total_price_of_books_l3_3980


namespace safety_rent_a_car_cost_per_mile_l3_3085

/-
Problem:
Prove that the cost per mile for Safety Rent-a-Car is 0.177 dollars, given that the total cost of renting an intermediate-size car for 150 miles is the same for Safety Rent-a-Car and City Rentals, with their respective pricing schemes.
-/

theorem safety_rent_a_car_cost_per_mile :
  let x := 21.95
  let y := 18.95
  let z := 0.21
  (x + 150 * real_safety_per_mile) = (y + 150 * z) ↔ real_safety_per_mile = 0.177 :=
by
  sorry

end safety_rent_a_car_cost_per_mile_l3_3085


namespace street_trees_one_side_number_of_street_trees_l3_3965

-- Conditions
def road_length : ℕ := 2575
def interval : ℕ := 25
def trees_at_endpoints : ℕ := 2

-- Question: number of street trees on one side of the road
theorem street_trees_one_side (road_length interval : ℕ) (trees_at_endpoints : ℕ) : ℕ :=
  (road_length / interval) + 1

-- Proof of the provided problem
theorem number_of_street_trees : street_trees_one_side road_length interval trees_at_endpoints = 104 :=
by
  sorry

end street_trees_one_side_number_of_street_trees_l3_3965


namespace student_monthly_earnings_l3_3740

theorem student_monthly_earnings :
  let daily_rate := 1250
  let days_per_week := 4
  let weeks_per_month := 4
  let income_tax_rate := 0.13
  let weekly_earnings := daily_rate * days_per_week
  let monthly_earnings_before_tax := weekly_earnings * weeks_per_month
  let income_tax_amount := monthly_earnings_before_tax * income_tax_rate
  let monthly_earnings_after_tax := monthly_earnings_before_tax - income_tax_amount
  monthly_earnings_after_tax = 17400 := by
  -- Proof steps here
  sorry

end student_monthly_earnings_l3_3740


namespace gibraltar_initial_population_stable_l3_3836
-- Import necessary libraries

-- Define constants based on conditions
def full_capacity := 300 * 4
def initial_population := (full_capacity / 3) - 100
def population := 300 -- This is the final answer we need to validate

-- The main theorem to prove
theorem gibraltar_initial_population_stable : initial_population = population :=
by 
  -- Proof is skipped as requested
  sorry

end gibraltar_initial_population_stable_l3_3836


namespace volume_parallelepiped_l3_3395

open Real

theorem volume_parallelepiped :
  ∃ (a h : ℝ), 
    let S_base := (4 : ℝ)
    let AB := a
    let AD := 2 * a
    let lateral_face1 := (6 : ℝ)
    let lateral_face2 := (12 : ℝ)
    (AB * h = lateral_face1) ∧
    (AD * h = lateral_face2) ∧
    (1 / 2 * AD * S_base = AB * (1 / 2 * AD)) ∧ 
    (AB^2 + AD^2 - 2 * AB * AD * (cos (π / 6)) = S_base) ∧
    (a = 2) ∧
    (h = 3) ∧ 
    (S_base * h = 12) :=
sorry

end volume_parallelepiped_l3_3395


namespace max_diff_distance_l3_3153

def hyperbola_right_branch (x y : ℝ) : Prop := 
  (x^2 / 9) - (y^2 / 16) = 1 ∧ x > 0

def circle_1 (x y : ℝ) : Prop := 
  (x + 5)^2 + y^2 = 4

def circle_2 (x y : ℝ) : Prop := 
  (x - 5)^2 + y^2 = 1

theorem max_diff_distance 
  (P M N : ℝ × ℝ) 
  (hp : hyperbola_right_branch P.fst P.snd) 
  (hm : circle_1 M.fst M.snd) 
  (hn : circle_2 N.fst N.snd) :
  |dist P M - dist P N| ≤ 9 := 
sorry

end max_diff_distance_l3_3153


namespace gain_percent_is_30_l3_3967

-- Given conditions
def CostPrice : ℕ := 100
def SellingPrice : ℕ := 130
def Gain : ℕ := SellingPrice - CostPrice
def GainPercent : ℕ := (Gain * 100) / CostPrice

-- The theorem to be proven
theorem gain_percent_is_30 :
  GainPercent = 30 := sorry

end gain_percent_is_30_l3_3967


namespace sum_of_fractions_bounds_l3_3607

theorem sum_of_fractions_bounds (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h_sum_numerators : a + c = 1000) (h_sum_denominators : b + d = 1000) :
  (999 / 969 + 1 / 31) ≤ (a / b + c / d) ∧ (a / b + c / d) ≤ (999 + 1 / 999) :=
by
  sorry

end sum_of_fractions_bounds_l3_3607


namespace equal_roots_quadratic_k_eq_one_l3_3017

theorem equal_roots_quadratic_k_eq_one
  (k : ℝ)
  (h : ∃ x : ℝ, x^2 - 2 * x + k == 0 ∧ x^2 - 2 * x + k == 0) :
  k = 1 :=
by {
  sorry
}

end equal_roots_quadratic_k_eq_one_l3_3017


namespace gcd_bc_minimum_l3_3863

theorem gcd_bc_minimum
  (a b c : ℕ)
  (h1 : Nat.gcd a b = 360)
  (h2 : Nat.gcd a c = 1170)
  (h3 : ∃ k1 : ℕ, b = 5 * k1)
  (h4 : ∃ k2 : ℕ, c = 13 * k2) : Nat.gcd b c = 90 :=
by
  sorry

end gcd_bc_minimum_l3_3863


namespace remainder_of_7_pow_4_div_100_l3_3213

theorem remainder_of_7_pow_4_div_100 :
  (7^4) % 100 = 1 := 
sorry

end remainder_of_7_pow_4_div_100_l3_3213


namespace quadratic_inequality_l3_3652

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, a * x^2 + 2 * x + 1 > 0) : a ≥ 1 :=
sorry

end quadratic_inequality_l3_3652


namespace system_solution_l3_3976

theorem system_solution (x y : ℚ) (h1 : 2 * x - 3 * y = 1) (h2 : (y + 1) / 4 + 1 = (x + 2) / 3) : x = 3 ∧ y = 5 / 3 :=
by
  sorry

end system_solution_l3_3976


namespace problem_condition_problem_statement_l3_3396

noncomputable def a : ℕ → ℕ 
| 0     => 2
| (n+1) => 3 * a n

noncomputable def S : ℕ → ℕ
| 0     => 0
| (n+1) => S n + a n

theorem problem_condition : ∀ n, 3 * a n - 2 * S n = 2 :=
by
  sorry

theorem problem_statement (n : ℕ) (h : ∀ n, 3 * a n - 2 * S n = 2) :
  (S (n+1))^2 - (S n) * (S (n+2)) = 4 * 3^n :=
by
  sorry

end problem_condition_problem_statement_l3_3396


namespace first_month_sale_eq_6435_l3_3198

theorem first_month_sale_eq_6435 (s2 s3 s4 s5 s6 : ℝ)
  (h2 : s2 = 6927) (h3 : s3 = 6855) (h4 : s4 = 7230) (h5 : s5 = 6562) (h6 : s6 = 7391)
  (avg : ℝ) (h_avg : avg = 6900) :
  let total_sales := 6 * avg
  let other_months_sales := s2 + s3 + s4 + s5 + s6
  let first_month_sale := total_sales - other_months_sales
  first_month_sale = 6435 :=
by
  sorry

end first_month_sale_eq_6435_l3_3198


namespace equilateral_triangles_circle_l3_3728

-- Definitions and conditions
structure Triangle :=
  (A B C : ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 12)

structure Circle :=
  (S : ℝ)

def PointOnArc (P1 P2 P : ℝ) : Prop :=
  -- Definition to describe P lies on the arc P1P2
  sorry

-- Theorem stating the proof problem
theorem equilateral_triangles_circle
  (S : Circle)
  (T1 T2 : Triangle)
  (H1 : T1.side_length = 12)
  (H2 : T2.side_length = 12)
  (HAonArc : PointOnArc T2.B T2.C T1.A)
  (HBonArc : PointOnArc T2.A T2.B T1.B) :
  (T1.A - T2.A) ^ 2 + (T1.B - T2.B) ^ 2 + (T1.C - T2.C) ^ 2 = 288 :=
sorry

end equilateral_triangles_circle_l3_3728


namespace not_right_triangle_condition_C_l3_3408

theorem not_right_triangle_condition_C :
  ∀ (a b c : ℝ), 
    (a^2 = b^2 + c^2) ∨
    (∀ (angleA angleB angleC : ℝ), angleA = angleB + angleC ∧ angleA + angleB + angleC = 180) ∨
    (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5) ∨
    (a^2 / b^2 = 1 / 2 ∧ b^2 / c^2 = 2 / 3) ->
    ¬ (∀ (angleA angleB angleC : ℝ), angleA / angleB = 3 / 4 ∧ angleB / angleC = 4 / 5 -> angleA = 90 ∨ angleB = 90 ∨ angleC = 90) :=
by
  intro a b c h
  cases h
  case inl h1 =>
    -- Option A: b^2 = a^2 - c^2
    sorry
  case inr h2 =>
    cases h2
    case inl h3 => 
      -- Option B: angleA = angleB + angleC
      sorry
    case inr h4 =>
      cases h4
      case inl h5 =>
        -- Option C: angleA : angleB : angleC = 3 : 4 : 5
        sorry
      case inr h6 =>
        -- Option D: a^2 : b^2 : c^2 = 1 : 2 : 3
        sorry

end not_right_triangle_condition_C_l3_3408


namespace factory_a_min_hours_l3_3128

theorem factory_a_min_hours (x : ℕ) :
  (550 * x + (700 - 55 * x) / 45 * 495 ≤ 7260) → (8 ≤ x) :=
by
  sorry

end factory_a_min_hours_l3_3128


namespace square_is_six_l3_3279

def represents_digit (square triangle circle : ℕ) : Prop :=
  square < 10 ∧ triangle < 10 ∧ circle < 10 ∧
  square ≠ triangle ∧ square ≠ circle ∧ triangle ≠ circle

theorem square_is_six :
  ∃ (square triangle circle : ℕ), represents_digit square triangle circle ∧ triangle = 1 ∧ circle = 9 ∧ (square + triangle + 100 * 1 + 10 * 9) = 117 ∧ square = 6 :=
by {
  sorry
}

end square_is_six_l3_3279


namespace circle_tangent_unique_point_l3_3073

theorem circle_tangent_unique_point (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 1 → (x+4)^2 + (y-a)^2 = 25 → false) →
  (a = 2 * Real.sqrt 5 ∨ a = -2 * Real.sqrt 5 ∨ a = 0) :=
by
  sorry

end circle_tangent_unique_point_l3_3073


namespace part1_part2_l3_3403

theorem part1 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) = 0) → 
  a = -1 ∧ ∀ x : ℝ, (4 * x^2 + 4 * x + 1 = 0 ↔ x = -1/2) :=
sorry

theorem part2 (a : ℝ) : (a - 3 ≠ 0) ∧ (16 - 4 * (a-3) * (-1) > 0) → 
  a > -1 ∧ a ≠ 3 :=
sorry

end part1_part2_l3_3403


namespace projection_matrix_solution_l3_3618

theorem projection_matrix_solution (a c : ℚ) (Q : Matrix (Fin 2) (Fin 2) ℚ) 
  (hQ : Q = !![a, 18/45; c, 27/45] ) 
  (proj_Q : Q * Q = Q) : 
  (a, c) = (2/5, 3/5) :=
by
  sorry

end projection_matrix_solution_l3_3618


namespace find_function_solution_l3_3591

noncomputable def function_solution (f : ℤ → ℤ) : Prop :=
∀ x y : ℤ, x ≠ 0 → x * f (2 * f y - x) + y^2 * f (2 * x - f y) = f x ^ 2 / x + f (y * f y)

theorem find_function_solution : 
  ∀ f : ℤ → ℤ, function_solution f → (∀ x : ℤ, f x = 0) ∨ (∀ x : ℤ, f x = x^2) :=
sorry

end find_function_solution_l3_3591


namespace bridge_length_l3_3275

theorem bridge_length (train_length : ℕ) (train_cross_bridge_time : ℕ) (train_cross_lamp_time : ℕ) (bridge_length : ℕ) :
  train_length = 600 →
  train_cross_bridge_time = 70 →
  train_cross_lamp_time = 20 →
  bridge_length = 1500 :=
by
  intro h1 h2 h3
  sorry

end bridge_length_l3_3275


namespace determine_m_l3_3774

variables (m x : ℝ)
noncomputable def f (x : ℝ) := x^2 - 3*x + m
noncomputable def g (x : ℝ) := x^2 - 3*x + 5*m

theorem determine_m (h : 3 * f 5 = 2 * g 5) : m = 10 / 7 :=
by
  sorry

end determine_m_l3_3774


namespace complex_multiplication_l3_3283

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (1 - 2 * i) = 2 + i :=
by
  sorry

end complex_multiplication_l3_3283


namespace find_a_l3_3777

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 := 
sorry

end find_a_l3_3777


namespace fraction_integer_l3_3826

theorem fraction_integer (x y : ℤ) (h₁ : ∃ k : ℤ, 3 * x + 4 * y = 5 * k) : ∃ m : ℤ, 4 * x - 3 * y = 5 * m :=
by
  sorry

end fraction_integer_l3_3826


namespace manufacturing_sector_angle_l3_3062

theorem manufacturing_sector_angle (h1 : 50 ≤ 100) (h2 : 360 = 4 * 90) : 0.50 * 360 = 180 := 
by
  sorry

end manufacturing_sector_angle_l3_3062


namespace rectangle_width_eq_six_l3_3560

theorem rectangle_width_eq_six (w : ℝ) :
  ∃ w, (3 * w = 25 - 7) ↔ w = 6 :=
by
  -- Given the conditions as stated:
  -- Length of the rectangle: 3 inches
  -- Width of the square: 5 inches
  -- Difference in area between the square and the rectangle: 7 square inches
  -- We can show that the width of the rectangle is 6 inches.
  sorry

end rectangle_width_eq_six_l3_3560


namespace panthers_score_l3_3940

-- Definitions as per the conditions
def total_points (C P : ℕ) : Prop := C + P = 48
def margin (C P : ℕ) : Prop := C = P + 20

-- Theorem statement proving Panthers score 14 points
theorem panthers_score (C P : ℕ) (h1 : total_points C P) (h2 : margin C P) : P = 14 :=
sorry

end panthers_score_l3_3940


namespace problem_statement_l3_3964

theorem problem_statement : 2009 * 20082008 - 2008 * 20092009 = 0 := by
  sorry

end problem_statement_l3_3964


namespace cost_per_lunch_is_7_l3_3773

-- Definitions of the conditions
def total_children := 35
def total_chaperones := 5
def janet := 1
def additional_lunches := 3
def total_cost := 308

-- Calculate the total number of lunches
def total_lunches : Int :=
  total_children + total_chaperones + janet + additional_lunches

-- Statement to prove that the cost per lunch is 7
theorem cost_per_lunch_is_7 : total_cost / total_lunches = 7 := by
  sorry

end cost_per_lunch_is_7_l3_3773


namespace y_relationship_l3_3519

theorem y_relationship (x1 x2 x3 y1 y2 y3 : ℝ) 
  (h1: y1 = -4 / x1) (h2: y2 = -4 / x2) (h3: y3 = -4 / x3)
  (h4: x1 < 0) (h5: 0 < x2) (h6: x2 < x3) :
  y1 > y3 ∧ y3 > y2 :=
by
  sorry

end y_relationship_l3_3519


namespace math_proof_l3_3743

-- Definitions
def U := Set ℝ
def A : Set ℝ := {x | x ≥ 3}
def B : Set ℝ := {x | x^2 - 8*x + 7 ≤ 0}
def C (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Theorem
theorem math_proof (a : ℝ) :
  (A ∩ B = {x | 3 ≤ x ∧ x ≤ 7}) ∧
  (A ∪ B = {x | x ≥ 1}) ∧
  (C a ∪ A = A → a ≥ 4) :=
by
  sorry

end math_proof_l3_3743


namespace lower_bound_for_expression_l3_3615

theorem lower_bound_for_expression :
  ∃ L: ℤ, (∀ n: ℤ, L < 4 * n + 7 ∧ 4 * n + 7 < 120) → L = 5 :=
sorry

end lower_bound_for_expression_l3_3615


namespace additional_savings_l3_3257

-- Defining the conditions
def initial_price : ℝ := 50
def discount_one : ℝ := 6
def discount_percentage : ℝ := 0.15

-- Defining the final prices according to the two methods
def first_method : ℝ := (1 - discount_percentage) * (initial_price - discount_one)
def second_method : ℝ := (1 - discount_percentage) * initial_price - discount_one

-- Defining the savings for the two methods
def savings_first_method : ℝ := initial_price - first_method
def savings_second_method : ℝ := initial_price - second_method

-- Proving that the second method results in an additional 0.90 savings
theorem additional_savings : (savings_second_method - savings_first_method) = 0.90 :=
by
  sorry

end additional_savings_l3_3257


namespace smallest_b_for_quadratic_inequality_l3_3587

theorem smallest_b_for_quadratic_inequality : 
  ∃ b : ℝ, (b^2 - 16 * b + 63 ≤ 0) ∧ ∀ b' : ℝ, (b'^2 - 16 * b' + 63 ≤ 0) → b ≤ b' := sorry

end smallest_b_for_quadratic_inequality_l3_3587


namespace half_angle_quadrant_l3_3839

theorem half_angle_quadrant
  (α : ℝ) (k : ℤ)
  (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
  (∃ m : ℤ, m * π < α / 2 ∧ α / 2 < m * π + π / 2) :=
by
  sorry

end half_angle_quadrant_l3_3839


namespace driver_net_rate_of_pay_l3_3156

theorem driver_net_rate_of_pay
  (hours : ℕ)
  (speed : ℕ)
  (fuel_efficiency : ℕ)
  (pay_per_mile : ℚ)
  (gas_cost_per_gallon : ℚ)
  (net_rate_of_pay : ℚ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : fuel_efficiency = 25)
  (h4 : pay_per_mile = 0.60)
  (h5 : gas_cost_per_gallon = 2.50)
  (h6 : net_rate_of_pay = 25) :
  net_rate_of_pay = (hours * speed * pay_per_mile - (hours * speed / fuel_efficiency) * gas_cost_per_gallon) / hours := 
by sorry

end driver_net_rate_of_pay_l3_3156


namespace shaded_area_l3_3005

theorem shaded_area (d : ℝ) (k : ℝ) (π : ℝ) (r : ℝ)
  (h_diameter : d = 6) 
  (h_radius_large : k = 5)
  (h_small_radius: r = d / 2) :
  ((π * (k * r)^2) - (π * r^2)) = 216 * π :=
by
  sorry

end shaded_area_l3_3005


namespace factorize_expression_l3_3763

-- Define the variables a and b
variables (a b : ℝ)

-- State the theorem
theorem factorize_expression : 5*a^2*b - 20*b^3 = 5*b*(a + 2*b)*(a - 2*b) :=
by sorry

end factorize_expression_l3_3763


namespace height_average_inequality_l3_3709

theorem height_average_inequality 
    (a b c d : ℝ)
    (h1 : 3 * a + 2 * b = 2 * c + 3 * d)
    (h2 : a > d) : 
    (|c + d| / 2 > |a + b| / 2) :=
sorry

end height_average_inequality_l3_3709


namespace find_a4_l3_3593

variable {a_n : ℕ → ℝ}
variable (S_n : ℕ → ℝ)

noncomputable def Sn := 1/2 * 5 * (a_n 1 + a_n 5)

axiom h1 : S_n 5 = 25
axiom h2 : a_n 2 = 3

theorem find_a4 : a_n 4 = 5 := sorry

end find_a4_l3_3593


namespace cost_per_top_l3_3293
   
   theorem cost_per_top 
     (total_spent : ℕ) 
     (short_pairs : ℕ) 
     (short_cost_per_pair : ℕ) 
     (shoe_pairs : ℕ) 
     (shoe_cost_per_pair : ℕ) 
     (top_count : ℕ)
     (remaining_cost : ℕ)
     (total_short_cost : ℕ) 
     (total_shoe_cost : ℕ) 
     (total_short_shoe_cost : ℕ)
     (total_top_cost : ℕ) :
     total_spent = 75 →
     short_pairs = 5 →
     short_cost_per_pair = 7 →
     shoe_pairs = 2 →
     shoe_cost_per_pair = 10 →
     top_count = 4 →
     total_short_cost = short_pairs * short_cost_per_pair →
     total_shoe_cost = shoe_pairs * shoe_cost_per_pair →
     total_short_shoe_cost = total_short_cost + total_shoe_cost →
     total_top_cost = total_spent - total_short_shoe_cost →
     remaining_cost = total_top_cost / top_count →
     remaining_cost = 5 :=
   by
     intros
     sorry
   
end cost_per_top_l3_3293


namespace necessary_but_not_sufficient_condition_l3_3029

theorem necessary_but_not_sufficient_condition (x y : ℝ) : 
  ((x > 1) ∨ (y > 2)) → (x + y > 3) ∧ ¬((x > 1) ∨ (y > 2) ↔ (x + y > 3)) :=
by
  sorry

end necessary_but_not_sufficient_condition_l3_3029


namespace cos_sin_15_deg_l3_3754

theorem cos_sin_15_deg :
  400 * (Real.cos (15 * Real.pi / 180))^5 +  (Real.sin (15 * Real.pi / 180))^5 / (Real.cos (15 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 100 := 
sorry

end cos_sin_15_deg_l3_3754


namespace annual_interest_approx_l3_3970

noncomputable def P : ℝ := 10000
noncomputable def r : ℝ := 0.05
noncomputable def t : ℝ := 1
noncomputable def e : ℝ := Real.exp 1

theorem annual_interest_approx :
  let A := P * Real.exp (r * t)
  let interest := A - P
  abs (interest - 512.71) < 0.01 := sorry

end annual_interest_approx_l3_3970


namespace journey_total_distance_l3_3348

def miles_driven : ℕ := 923
def miles_to_go : ℕ := 277
def total_distance : ℕ := 1200

theorem journey_total_distance : miles_driven + miles_to_go = total_distance := by
  sorry

end journey_total_distance_l3_3348


namespace total_robodinos_in_shipment_l3_3981

-- Definitions based on the conditions:
def percentage_on_display : ℝ := 0.30
def percentage_in_storage : ℝ := 0.70
def stored_robodinos : ℕ := 168

-- The main statement to prove:
theorem total_robodinos_in_shipment (T : ℝ) : (percentage_in_storage * T = stored_robodinos) → T = 240 := by
  sorry

end total_robodinos_in_shipment_l3_3981


namespace valid_integer_pairs_l3_3416

theorem valid_integer_pairs :
  { (x, y) : ℤ × ℤ |
    (∃ α β : ℝ, α^2 + β^2 < 4 ∧ α + β = (-x : ℝ) ∧ α * β = y ∧ x^2 - 4 * y ≥ 0) } =
  {(-2,1), (-1,-1), (-1,0), (0, -1), (0,0), (1,0), (1,-1), (2,1)} :=
sorry

end valid_integer_pairs_l3_3416


namespace height_of_fourth_person_l3_3289

theorem height_of_fourth_person 
  (H : ℕ) 
  (h_avg : ((H) + (H + 2) + (H + 4) + (H + 10)) / 4 = 79) :
  (H + 10 = 85) :=
by
  sorry

end height_of_fourth_person_l3_3289


namespace rectangle_width_l3_3173

theorem rectangle_width (w : ℝ) (h_length : w * 2 = l) (h_area : w * l = 50) : w = 5 :=
by
  sorry

end rectangle_width_l3_3173
