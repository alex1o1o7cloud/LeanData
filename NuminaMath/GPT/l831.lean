import Mathlib

namespace events_mutually_exclusive_not_complementary_l831_83116

-- Define the set of balls and people
inductive Ball : Type
| b1 | b2 | b3 | b4

inductive Person : Type
| A | B | C | D

-- Define the event types
structure Event :=
  (p : Person)
  (b : Ball)

-- Define specific events as follows
def EventA : Event := { p := Person.A, b := Ball.b1 }
def EventB : Event := { p := Person.B, b := Ball.b1 }

-- We want to prove the relationship between two specific events:
-- "Person A gets ball number 1" and "Person B gets ball number 1"
-- Namely, that they are mutually exclusive but not complementary.

theorem events_mutually_exclusive_not_complementary :
  (∀ e : Event, (e = EventA → ¬ (e = EventB)) ∧ ¬ (e = EventA ∨ e = EventB)) :=
sorry

end events_mutually_exclusive_not_complementary_l831_83116


namespace lena_calculation_l831_83113

def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + (10 - n % 10)

theorem lena_calculation :
  round_to_nearest_ten (63 + 2 * 29) = 120 :=
by
  sorry

end lena_calculation_l831_83113


namespace gunny_bag_capacity_in_tons_l831_83153

def ton_to_pounds := 2200
def pound_to_ounces := 16
def packets := 1760
def packet_weight_pounds := 16
def packet_weight_ounces := 4

theorem gunny_bag_capacity_in_tons :
  ((packets * (packet_weight_pounds + (packet_weight_ounces / pound_to_ounces))) / ton_to_pounds) = 13 :=
sorry

end gunny_bag_capacity_in_tons_l831_83153


namespace initial_machines_l831_83121

theorem initial_machines (r : ℝ) (x : ℕ) (h1 : x * 42 * r = 7 * 36 * r) : x = 6 :=
by
  sorry

end initial_machines_l831_83121


namespace find_b_l831_83147

variables (U : Set ℝ) (A : Set ℝ) (b : ℝ)

theorem find_b (hU : U = Set.univ)
               (hA : A = {x | 1 ≤ x ∧ x < b})
               (hComplA : U \ A = {x | x < 1 ∨ x ≥ 2}) :
  b = 2 :=
sorry

end find_b_l831_83147


namespace mapping_problem_l831_83196

open Set

noncomputable def f₁ (x : ℝ) : ℝ := Real.sqrt x
noncomputable def f₂ (x : ℝ) : ℝ := 1 / x
def f₃ (x : ℝ) : ℝ := x^2 - 2
def f₄ (x : ℝ) : ℝ := x^2

def A₁ : Set ℝ := {1, 4, 9}
def B₁ : Set ℝ := {-3, -2, -1, 1, 2, 3}
def A₂ : Set ℝ := univ
def B₂ : Set ℝ := univ
def A₃ : Set ℝ := univ
def B₃ : Set ℝ := univ
def A₄ : Set ℝ := {-1, 0, 1}
def B₄ : Set ℝ := {-1, 0, 1}

theorem mapping_problem : 
  ¬ (∀ x ∈ A₁, f₁ x ∈ B₁) ∧
  ¬ (∀ x ∈ A₂, x ≠ 0 → f₂ x ∈ B₂) ∧
  (∀ x ∈ A₃, f₃ x ∈ B₃) ∧
  (∀ x ∈ A₄, f₄ x ∈ B₄) :=
by
  sorry

end mapping_problem_l831_83196


namespace equation_solutions_equiv_l831_83137

theorem equation_solutions_equiv (p : ℕ) (hp : p.Prime) :
  (∃ x s : ℤ, x^2 - x + 3 - p * s = 0) ↔ 
  (∃ y t : ℤ, y^2 - y + 25 - p * t = 0) :=
by { sorry }

end equation_solutions_equiv_l831_83137


namespace value_of_p_l831_83155

theorem value_of_p (m n p : ℝ) (h₁ : m = 8 * n + 5) (h₂ : m + 2 = 8 * (n + p) + 5) : p = 1 / 4 :=
by {
  sorry
}

end value_of_p_l831_83155


namespace cone_volume_l831_83193

noncomputable def volume_of_cone_from_lateral_surface (radius_semicircle : ℝ) 
  (circumference_base : ℝ := 2 * radius_semicircle * Real.pi) 
  (radius_base : ℝ := circumference_base / (2 * Real.pi)) 
  (height_cone : ℝ := Real.sqrt ((radius_semicircle:ℝ) ^ 2 - (radius_base:ℝ) ^ 2)) : ℝ := 
  (1 / 3) * Real.pi * (radius_base ^ 2) * height_cone

theorem cone_volume (h_semicircle : 2 = 2) : volume_of_cone_from_lateral_surface 2 = (Real.sqrt 3) / 3 * Real.pi := 
by
  -- Importing Real.sqrt and Real.pi to bring them into scope
  sorry

end cone_volume_l831_83193


namespace raisin_cost_fraction_l831_83178

theorem raisin_cost_fraction
  (R : ℚ) -- cost of a pound of raisins in dollars
  (cost_of_nuts : ℚ)
  (total_cost_raisins : ℚ)
  (total_cost_nuts : ℚ) :
  cost_of_nuts = 3 * R →
  total_cost_raisins = 5 * R →
  total_cost_nuts = 4 * cost_of_nuts →
  (total_cost_raisins / (total_cost_raisins + total_cost_nuts)) = 5 / 17 :=
by
  sorry

end raisin_cost_fraction_l831_83178


namespace runner_overtake_time_l831_83187

theorem runner_overtake_time
  (L : ℝ)
  (v1 v2 v3 : ℝ)
  (h1 : v1 = v2 + L / 6)
  (h2 : v1 = v3 + L / 10) :
  L / (v3 - v2) = 15 := by
  sorry

end runner_overtake_time_l831_83187


namespace fruit_salad_cherries_l831_83166

variable (b r g c : ℕ)

theorem fruit_salad_cherries :
  (b + r + g + c = 350) ∧
  (r = 3 * b) ∧
  (g = 4 * c) ∧
  (c = 5 * r) →
  c = 66 :=
by
  sorry

end fruit_salad_cherries_l831_83166


namespace factorized_expression_l831_83160

variable {a b c : ℝ}

theorem factorized_expression :
  ( ((a^2 - b^2)^3 + (b^2 - c^2)^3 + (c^2 - a^2)^3) / 
    ((a - b)^3 + (b - c)^3 + (c - a)^3) ) 
  = (a + b) * (a + c) * (b + c) := 
  sorry

end factorized_expression_l831_83160


namespace min_throws_for_repeated_sum_l831_83179

theorem min_throws_for_repeated_sum (n : ℕ) (h1 : 2 ≤ n) (h2 : n ≤ 16) : 
  ∃ m, m = 16 ∧ (∀ (k : ℕ), k < 16 → ∃ i < 16, ∃ j < 16, i ≠ j ∧ i + j = k) :=
by
  sorry

end min_throws_for_repeated_sum_l831_83179


namespace radius_of_circle_l831_83194

theorem radius_of_circle :
  ∀ (r : ℝ), (π * r^2 = 2.5 * 2 * π * r) → r = 5 :=
by sorry

end radius_of_circle_l831_83194


namespace scaled_system_solution_l831_83182

theorem scaled_system_solution (a1 b1 c1 a2 b2 c2 x y : ℝ) 
  (h1 : a1 * 8 + b1 * 3 = c1) 
  (h2 : a2 * 8 + b2 * 3 = c2) : 
  4 * a1 * 10 + 3 * b1 * 5 = 5 * c1 ∧ 4 * a2 * 10 + 3 * b2 * 5 = 5 * c2 := 
by 
  sorry

end scaled_system_solution_l831_83182


namespace parallel_lines_solution_l831_83109

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y + 1 = 0 → 2 * x + a * y + 2 = 0 → (a = 1 ∨ a = -2)) :=
by
  sorry

end parallel_lines_solution_l831_83109


namespace fence_pole_count_l831_83136

-- Define the conditions
def path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the goal
def total_poles : ℕ := 286

-- The statement to prove
theorem fence_pole_count :
  let total_length_to_fence := (path_length - bridge_length)
  let poles_per_side := total_length_to_fence / pole_spacing
  let total_poles_needed := 2 * poles_per_side
  total_poles_needed = total_poles :=
by
  sorry

end fence_pole_count_l831_83136


namespace max_distance_between_bus_stops_l831_83117

theorem max_distance_between_bus_stops 
  (v_m : ℝ) (v_b : ℝ) (dist : ℝ) 
  (h1 : v_m = v_b / 3) (h2 : dist = 2) : 
  ∀ d : ℝ, d = 1.5 := sorry

end max_distance_between_bus_stops_l831_83117


namespace xiao_wang_exam_grades_l831_83184

theorem xiao_wang_exam_grades 
  (x y : ℕ) 
  (h1 : (x * y + 98) / (x + 1) = y + 1)
  (h2 : (x * y + 98 + 70) / (x + 2) = y - 1) : 
  x + 2 = 10 ∧ y - 1 = 88 := 
by
  sorry

end xiao_wang_exam_grades_l831_83184


namespace samantha_birth_year_l831_83192

theorem samantha_birth_year (first_kangaroo_year birth_year kangaroo_freq : ℕ)
  (h_first_kangaroo: first_kangaroo_year = 1991)
  (h_kangaroo_freq: kangaroo_freq = 1)
  (h_samantha_age: ∃ y, y = (first_kangaroo_year + 9 * kangaroo_freq) ∧ 2000 - 14 = y) :
  birth_year = 1986 :=
by sorry

end samantha_birth_year_l831_83192


namespace michael_water_left_l831_83185

theorem michael_water_left :
  let initial_water := 5
  let given_water := (18 / 7 : ℚ) -- using rational number to represent the fractions
  let remaining_water := initial_water - given_water
  remaining_water = 17 / 7 :=
by
  sorry

end michael_water_left_l831_83185


namespace gcd_max_1001_l831_83151

theorem gcd_max_1001 (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1001) : 
  ∃ d, d = Nat.gcd a b ∧ d ≤ 143 := 
sorry

end gcd_max_1001_l831_83151


namespace manny_original_marbles_l831_83183

/-- 
Let total marbles be 120, and the marbles are divided between Mario, Manny, and Mike in the ratio 4:5:6. 
Let x be the number of marbles Manny is left with after giving some marbles to his brother.
Prove that Manny originally had 40 marbles. 
-/
theorem manny_original_marbles (total_marbles : ℕ) (ratio_mario ratio_manny ratio_mike : ℕ)
    (present_marbles : ℕ) (total_parts : ℕ)
    (h_marbles : total_marbles = 120) 
    (h_ratio : ratio_mario = 4 ∧ ratio_manny = 5 ∧ ratio_mike = 6) 
    (h_total_parts : total_parts = ratio_mario + ratio_manny + ratio_mike)
    (h_manny_parts : total_marbles/total_parts * ratio_manny = 40) : 
  present_marbles = 40 := 
sorry

end manny_original_marbles_l831_83183


namespace Mickey_less_than_twice_Minnie_l831_83114

def Minnie_horses_per_day : ℕ := 10
def Mickey_horses_per_day : ℕ := 14

theorem Mickey_less_than_twice_Minnie :
  2 * Minnie_horses_per_day - Mickey_horses_per_day = 6 := by
  sorry

end Mickey_less_than_twice_Minnie_l831_83114


namespace rabbit_count_l831_83190

-- Define the conditions
def original_rabbits : ℕ := 8
def new_rabbits_born : ℕ := 5

-- Define the total rabbits based on the conditions
def total_rabbits : ℕ := original_rabbits + new_rabbits_born

-- The statement to prove that the total number of rabbits is 13
theorem rabbit_count : total_rabbits = 13 :=
by
  -- Proof not needed, hence using sorry
  sorry

end rabbit_count_l831_83190


namespace g_of_900_eq_34_l831_83177

theorem g_of_900_eq_34 (g : ℕ+ → ℝ) 
  (h_mul : ∀ x y : ℕ+, g (x * y) = g x + g y)
  (h_30 : g 30 = 17)
  (h_60 : g 60 = 21) :
  g 900 = 34 :=
sorry

end g_of_900_eq_34_l831_83177


namespace fraction_of_apples_consumed_l831_83131

theorem fraction_of_apples_consumed (f : ℚ) 
  (bella_eats_per_day : ℚ := 6) 
  (days_per_week : ℕ := 7) 
  (grace_remaining_apples : ℚ := 504) 
  (weeks_passed : ℕ := 6) 
  (total_apples_picked : ℚ := 42 / f) :
  (total_apples_picked - (bella_eats_per_day * days_per_week * weeks_passed) = grace_remaining_apples) 
  → f = 1 / 18 :=
by
  intro h
  sorry

end fraction_of_apples_consumed_l831_83131


namespace fraction_identity_l831_83111

theorem fraction_identity (f : ℚ) (h : 32 * f^2 = 2^3) : f = 1 / 2 :=
sorry

end fraction_identity_l831_83111


namespace interest_rate_l831_83104

theorem interest_rate (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) : 
  P = 8000.000000000171 → t = 2 → d = 20 →
  (P * (1 + r/100)^2 - P - (P * r * t / 100) = d) → r = 5 :=
by
  intros hP ht hd heq
  sorry

end interest_rate_l831_83104


namespace greatest_four_digit_divisible_by_conditions_l831_83167

-- Definitions based on the conditions
def is_divisible_by (a b : ℕ) : Prop := ∃ k, a = k * b

def is_four_digit (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

-- Problem statement: Finding the greatest 4-digit number divisible by 15, 25, 40, and 75
theorem greatest_four_digit_divisible_by_conditions :
  ∃ n, is_four_digit n ∧ is_divisible_by n 15 ∧ is_divisible_by n 25 ∧ is_divisible_by n 40 ∧ is_divisible_by n 75 ∧ n = 9600 :=
  sorry

end greatest_four_digit_divisible_by_conditions_l831_83167


namespace calculate_total_cost_l831_83140

theorem calculate_total_cost :
  let sandwich_cost := 4
  let soda_cost := 3
  let num_sandwiches := 6
  let num_sodas := 5
  sandwich_cost * num_sandwiches + soda_cost * num_sodas = 39 := by
  sorry

end calculate_total_cost_l831_83140


namespace option_not_equal_to_three_halves_l831_83141

theorem option_not_equal_to_three_halves (d : ℚ) (h1 : d = 3/2) 
    (hA : 9/6 = 3/2) 
    (hB : 1 + 1/2 = 3/2) 
    (hC : 1 + 2/4 = 3/2)
    (hE : 1 + 6/12 = 3/2) :
  1 + 2/3 ≠ 3/2 :=
by
  sorry

end option_not_equal_to_three_halves_l831_83141


namespace painter_completes_at_9pm_l831_83139

noncomputable def mural_completion_time (start_time : Nat) (fraction_completed_time : Nat)
    (fraction_completed : ℚ) : Nat :=
  let fraction_per_hour := fraction_completed / fraction_completed_time
  start_time + Nat.ceil (1 / fraction_per_hour)

theorem painter_completes_at_9pm :
  mural_completion_time 9 3 (1/4) = 21 := by
  sorry

end painter_completes_at_9pm_l831_83139


namespace parameterized_line_solution_l831_83108

theorem parameterized_line_solution :
  ∃ s l : ℝ, s = 1 / 2 ∧ l = -10 ∧
    ∀ t : ℝ, ∃ x y : ℝ,
      (x = -7 + t * l → y = s + t * (-5)) ∧ (y = (1 / 2) * x + 4) :=
by
  sorry

end parameterized_line_solution_l831_83108


namespace room_area_ratio_l831_83172

theorem room_area_ratio (total_squares overlapping_squares : ℕ) 
  (h_total : total_squares = 16) 
  (h_overlap : overlapping_squares = 4) : 
  total_squares / overlapping_squares = 4 := 
by 
  sorry

end room_area_ratio_l831_83172


namespace distance_between_city_centers_l831_83158

theorem distance_between_city_centers (d_map : ℝ) (scale : ℝ) (d_real : ℝ) (h1 : d_map = 112) (h2 : scale = 10) (h3 : d_real = d_map * scale) : d_real = 1120 := by
  sorry

end distance_between_city_centers_l831_83158


namespace fishAddedIs15_l831_83186

-- Define the number of fish Jason starts with
def initialNumberOfFish : ℕ := 6

-- Define the fish counts on each day
def fishOnDay2 := 2 * initialNumberOfFish
def fishOnDay3 := 2 * fishOnDay2 - (1 / 3 : ℚ) * (2 * fishOnDay2)
def fishOnDay4 := 2 * fishOnDay3
def fishOnDay5 := 2 * fishOnDay4 - (1 / 4 : ℚ) * (2 * fishOnDay4)
def fishOnDay6 := 2 * fishOnDay5
def fishOnDay7 := 2 * fishOnDay6

-- Define the total fish on the seventh day after adding some fish
def totalFishOnDay7 := 207

-- Define the number of fish Jason added on the seventh day
def fishAddedOnDay7 := totalFishOnDay7 - fishOnDay7

-- Prove that the number of fish Jason added on the seventh day is 15
theorem fishAddedIs15 : fishAddedOnDay7 = 15 := sorry

end fishAddedIs15_l831_83186


namespace y1_lt_y2_of_linear_function_l831_83154

theorem y1_lt_y2_of_linear_function (y1 y2 : ℝ) (h1 : y1 = 2 * (-3) + 1) (h2 : y2 = 2 * 2 + 1) : y1 < y2 :=
by
  sorry

end y1_lt_y2_of_linear_function_l831_83154


namespace smallest_n_l831_83165

theorem smallest_n (n : ℕ) (h : 23 * n ≡ 789 [MOD 11]) : n = 9 :=
sorry

end smallest_n_l831_83165


namespace defective_rate_worker_y_l831_83126

theorem defective_rate_worker_y (d_x d_y : ℝ) (f_y : ℝ) (total_defective_rate : ℝ) :
  d_x = 0.005 → f_y = 0.8 → total_defective_rate = 0.0074 → 
  (0.2 * d_x + f_y * d_y = total_defective_rate) → d_y = 0.008 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end defective_rate_worker_y_l831_83126


namespace group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l831_83107

def mats_weaved (weavers mats days : ℕ) : ℕ :=
  (mats / days) * weavers

theorem group_a_mats_in_12_days (mats_req : ℕ) :
  let weavers := 4
  let mats_per_period := 4
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_b_mats_in_12_days (mats_req : ℕ) :
  let weavers := 6
  let mats_per_period := 9
  let period_days := 3
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_c_mats_in_12_days (mats_req : ℕ) :
  let weavers := 8
  let mats_per_period := 16
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

end group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l831_83107


namespace find_f_2012_l831_83146

noncomputable def f : ℝ → ℝ := sorry

axiom f_condition1 : f 1 = 1 / 4
axiom f_condition2 : ∀ x y : ℝ, 4 * f x * f y = f (x + y) + f (x - y)

theorem find_f_2012 : f 2012 = -1 / 4 := 
sorry

end find_f_2012_l831_83146


namespace integer_solutions_exist_l831_83132

theorem integer_solutions_exist (x y : ℤ) : 
  12 * x^2 + 7 * y^2 = 4620 ↔ 
  (x = 7 ∧ y = 24) ∨ 
  (x = -7 ∧ y = 24) ∨
  (x = 7 ∧ y = -24) ∨
  (x = -7 ∧ y = -24) ∨
  (x = 14 ∧ y = 18) ∨
  (x = -14 ∧ y = 18) ∨
  (x = 14 ∧ y = -18) ∨
  (x = -14 ∧ y = -18) :=
sorry

end integer_solutions_exist_l831_83132


namespace sum_of_digits_of_smallest_divisible_is_6_l831_83188

noncomputable def smallest_divisible (n : ℕ) : ℕ :=
Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7

def sum_of_digits (n : ℕ) : ℕ :=
n.digits 10 |>.sum

theorem sum_of_digits_of_smallest_divisible_is_6 : sum_of_digits (smallest_divisible 7) = 6 := 
by
  simp [smallest_divisible, sum_of_digits]
  sorry

end sum_of_digits_of_smallest_divisible_is_6_l831_83188


namespace janet_saving_l831_83189

def tile_cost_difference_saving : ℕ :=
  let turquoise_cost_per_tile := 13
  let purple_cost_per_tile := 11
  let area_wall1 := 5 * 8
  let area_wall2 := 7 * 8
  let total_area := area_wall1 + area_wall2
  let tiles_per_square_foot := 4
  let number_of_tiles := total_area * tiles_per_square_foot
  let cost_difference_per_tile := turquoise_cost_per_tile - purple_cost_per_tile
  number_of_tiles * cost_difference_per_tile

theorem janet_saving : tile_cost_difference_saving = 768 := by
  sorry

end janet_saving_l831_83189


namespace dealer_car_ratio_calculation_l831_83181

theorem dealer_car_ratio_calculation (X Y : ℝ) 
  (cond1 : 1.4 * X = 1.54 * (X + Y) - 1.6 * Y) :
  let a := 3
  let b := 7
  ((X / Y) = (3 / 7) ∧ (11 * a + 13 * b = 124)) :=
by
  sorry

end dealer_car_ratio_calculation_l831_83181


namespace find_possible_values_of_a_l831_83199

noncomputable def P : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_possible_values_of_a (a : ℝ) (h : Q a ⊆ P) :
  a = 0 ∨ a = -1/2 ∨ a = 1/3 := by
  sorry

end find_possible_values_of_a_l831_83199


namespace tan_seven_pi_over_four_l831_83100

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end tan_seven_pi_over_four_l831_83100


namespace expression_evaluation_l831_83174

theorem expression_evaluation : |(-7: ℤ)| / ((2 / 3) - (1 / 5)) - (1 / 2) * ((-4)^2) = 7 := by
  sorry

end expression_evaluation_l831_83174


namespace units_digit_6_l831_83130

theorem units_digit_6 (p : ℤ) (hp : 0 < p % 10) (h1 : (p^3 % 10) = (p^2 % 10)) (h2 : (p + 2) % 10 = 8) : p % 10 = 6 :=
by
  sorry

end units_digit_6_l831_83130


namespace triangle_side_range_l831_83171

theorem triangle_side_range (x : ℝ) (h1 : x > 0) (h2 : x + (x + 1) + (x + 2) ≤ 12) :
  1 < x ∧ x ≤ 3 :=
by
  sorry

end triangle_side_range_l831_83171


namespace ron_chocolate_bar_cost_l831_83128

-- Definitions of the conditions given in the problem
def cost_per_chocolate_bar : ℝ := 1.50
def sections_per_chocolate_bar : ℕ := 3
def scouts : ℕ := 15
def s'mores_needed_per_scout : ℕ := 2
def total_s'mores_needed : ℕ := scouts * s'mores_needed_per_scout
def chocolate_bars_needed : ℕ := total_s'mores_needed / sections_per_chocolate_bar
def total_cost_of_chocolate_bars : ℝ := chocolate_bars_needed * cost_per_chocolate_bar

-- Proving the question equals the answer given conditions
theorem ron_chocolate_bar_cost : total_cost_of_chocolate_bars = 15.00 := by
  sorry

end ron_chocolate_bar_cost_l831_83128


namespace smallest_factor_of_36_sum_4_l831_83129

theorem smallest_factor_of_36_sum_4 : ∃ a b c : ℤ, (a * b * c = 36) ∧ (a + b + c = 4) ∧ (a = -4 ∨ b = -4 ∨ c = -4) :=
by
  sorry

end smallest_factor_of_36_sum_4_l831_83129


namespace ice_cream_flavors_l831_83120

theorem ice_cream_flavors (F : ℕ) (h1 : F / 4 + F / 2 + 25 = F) : F = 100 :=
by
  sorry

end ice_cream_flavors_l831_83120


namespace problem_I3_1_l831_83134

theorem problem_I3_1 (w x y z : ℝ) (h1 : w * x * y * z = 4) (h2 : w - x * y * z = 3) (h3 : w > 0) : 
  w = 4 :=
by
  sorry

end problem_I3_1_l831_83134


namespace proof_average_l831_83110

def average_two (x y : ℚ) : ℚ := (x + y) / 2
def average_three (x y z : ℚ) : ℚ := (x + y + z) / 3

theorem proof_average :
  average_three (2 * average_three 3 2 0) (average_two 0 3) (1 * 3) = 47 / 18 :=
by
  sorry

end proof_average_l831_83110


namespace sample_size_l831_83157

theorem sample_size (k n : ℕ) (h_ratio : 3 * n / (3 + 4 + 7) = 9) : n = 42 :=
by
  sorry

end sample_size_l831_83157


namespace cubic_identity_l831_83115

theorem cubic_identity (a b c : ℝ) 
  (h1 : a + b + c = 12)
  (h2 : ab + ac + bc = 30)
  : a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end cubic_identity_l831_83115


namespace regular_price_adult_ticket_l831_83180

theorem regular_price_adult_ticket : 
  ∀ (concessions_cost_children cost_adult1 cost_adult2 cost_adult3 cost_adult4 cost_adult5
       ticket_cost_child cost_discount1 cost_discount2 cost_discount3 total_cost : ℝ),
  (concessions_cost_children = 3) → 
  (cost_adult1 = 5) → 
  (cost_adult2 = 6) → 
  (cost_adult3 = 7) → 
  (cost_adult4 = 4) → 
  (cost_adult5 = 9) → 
  (ticket_cost_child = 7) → 
  (cost_discount1 = 3) → 
  (cost_discount2 = 2) → 
  (cost_discount3 = 1) → 
  (total_cost = 139) → 
  (∀ A : ℝ, total_cost = 
    (2 * concessions_cost_children + cost_adult1 + cost_adult2 + cost_adult3 + cost_adult4 + cost_adult5) + 
    (2 * ticket_cost_child + (2 * A + (A - cost_discount1) + (A - cost_discount2) + (A - cost_discount3))) → 
    5 * A - 6 = 88 →
    A = 18.80) :=
by
  intros
  sorry

end regular_price_adult_ticket_l831_83180


namespace evaluate_at_5_l831_83122

def f(x: ℝ) : ℝ := 3 * x^5 - 15 * x^4 + 27 * x^3 - 20 * x^2 - 72 * x + 40

theorem evaluate_at_5 : f 5 = 2515 :=
by
  sorry

end evaluate_at_5_l831_83122


namespace coin_flips_137_l831_83149

-- Definitions and conditions
def steph_transformation_heads (x : ℤ) : ℤ := 2 * x - 1
def steph_transformation_tails (x : ℤ) : ℤ := (x + 1) / 2
def jeff_transformation_heads (y : ℤ) : ℤ := y + 8
def jeff_transformation_tails (y : ℤ) : ℤ := y - 3

-- The problem statement
theorem coin_flips_137
  (a b : ℤ)
  (h₁ : a - b = 7)
  (h₂ : 8 * a - 3 * b = 381)
  (steph_initial jeff_initial : ℤ)
  (h₃ : steph_initial = 4)
  (h₄ : jeff_initial = 4) : a + b = 137 := 
by
  sorry

end coin_flips_137_l831_83149


namespace factorial_div_add_two_l831_83106

def factorial (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_div_add_two :
  (factorial 50) / (factorial 48) + 2 = 2452 :=
by
  sorry

end factorial_div_add_two_l831_83106


namespace triangle_side_and_altitude_sum_l831_83156

theorem triangle_side_and_altitude_sum 
(x y : ℕ) (h1 : x < 75) (h2 : y < 28)
(h3 : x * 60 = 75 * 28) (h4 : 100 * y = 75 * 28) : 
x + y = 56 := 
sorry

end triangle_side_and_altitude_sum_l831_83156


namespace proof_problem_l831_83135

variables {R : Type*} [CommRing R]

-- f is a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Variable definitions for the conditions
variables (h_odd : is_odd f)
(h_f1 : f 1 = 1)
(h_period : ∀ x, f (x + 6) = f x + f 3)

-- The proof problem statement
theorem proof_problem : f 2015 + f 2016 = -1 :=
by
  sorry

end proof_problem_l831_83135


namespace bounded_sequence_l831_83176

theorem bounded_sequence (a : ℕ → ℕ) (h1 : a 1 = 2) (h2 : a 2 = 2)
  (h_rec : ∀ n : ℕ, a (n + 2) = (a (n + 1) + a n) / Nat.gcd (a n) (a (n + 1))) :
  ∃ M : ℕ, ∀ n : ℕ, a n ≤ M := 
sorry

end bounded_sequence_l831_83176


namespace landlord_packages_l831_83125

def label_packages_required (start1 end1 start2 end2 start3 end3 : ℕ) : ℕ :=
  let digit_count := 1
  let hundreds_first := (end1 - start1 + 1)
  let hundreds_second := (end2 - start2 + 1)
  let hundreds_third := (end3 - start3 + 1)
  let total_hundreds := hundreds_first + hundreds_second + hundreds_third
  
  let tens_first := ((end1 - start1 + 1) / 10) 
  let tens_second := ((end2 - start2 + 1) / 10) 
  let tens_third := ((end3 - start3 + 1) / 10)
  let total_tens := tens_first + tens_second + tens_third

  let units_per_floor := 5
  let total_units := units_per_floor * 3
  
  let total_ones := total_hundreds + total_tens + total_units
  
  let packages_required := total_ones

  packages_required

theorem landlord_packages : label_packages_required 100 150 200 250 300 350 = 198 := 
  by sorry

end landlord_packages_l831_83125


namespace find_s_for_g3_eq_0_l831_83162

def g (x s : ℝ) : ℝ := 3 * x^5 - 2 * x^4 + x^3 - 4 * x^2 + 5 * x + s

theorem find_s_for_g3_eq_0 : (g 3 s = 0) ↔ (s = -573) :=
by
  sorry

end find_s_for_g3_eq_0_l831_83162


namespace overtime_percentage_increase_l831_83142

-- Define the conditions.
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def total_compensation : ℝ := 1116
def total_hours_worked : ℕ := 57
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Define the question and the answer as a proof problem.
theorem overtime_percentage_increase :
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  overtime_rate > regular_rate →
  ((overtime_rate - regular_rate) / regular_rate) * 100 = 75 := 
by
  sorry

end overtime_percentage_increase_l831_83142


namespace find_tangent_point_and_slope_l831_83170

theorem find_tangent_point_and_slope :
  ∃ m n : ℝ, (m = 1 ∧ n = Real.exp 1 ∧ 
    (∀ x y : ℝ, y - n = (Real.exp m) * (x - m) → x = 0 ∧ y = 0) ∧ 
    (Real.exp m = Real.exp 1)) :=
sorry

end find_tangent_point_and_slope_l831_83170


namespace find_C_l831_83144

theorem find_C (A B C : ℕ) (h1 : 3 * A - A = 10) (h2 : B + A = 12) (h3 : C - B = 6) : C = 13 :=
by
  sorry

end find_C_l831_83144


namespace red_balls_count_l831_83124

theorem red_balls_count (r y b : ℕ) (total_balls : ℕ := 15) (prob_neither_red : ℚ := 2/7) :
    y + b = total_balls - r → (15 - r) * (14 - r) = 60 → r = 5 :=
by
  intros h1 h2
  sorry

end red_balls_count_l831_83124


namespace smallest_even_number_of_sum_1194_l831_83112

-- Defining the given condition
def sum_of_three_consecutive_even_numbers (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) = 1194

-- Stating the theorem to prove the smallest even number
theorem smallest_even_number_of_sum_1194 :
  ∃ x : ℕ, sum_of_three_consecutive_even_numbers x ∧ x = 396 :=
by
  sorry

end smallest_even_number_of_sum_1194_l831_83112


namespace triangle_cosine_l831_83102

theorem triangle_cosine (LM : ℝ) (cos_N : ℝ) (LN : ℝ) (h1 : LM = 20) (h2 : cos_N = 3/5) :
  LM / LN = cos_N → LN = 100 / 3 :=
by
  intro h3
  sorry

end triangle_cosine_l831_83102


namespace min_value_of_quadratic_expression_l831_83152

theorem min_value_of_quadratic_expression (x y z : ℝ) (h : x + y + z = 1) :
  ∃ (u : ℝ), (2 * x^2 + 3 * y^2 + z^2 = u) ∧ u = 6 / 11 :=
sorry

end min_value_of_quadratic_expression_l831_83152


namespace remainder_when_6n_divided_by_4_l831_83168

theorem remainder_when_6n_divided_by_4 (n : ℤ) (h : n % 4 = 1) : 6 * n % 4 = 2 := by
  sorry

end remainder_when_6n_divided_by_4_l831_83168


namespace x1_mul_x2_l831_83143

open Real

theorem x1_mul_x2 (x1 x2 : ℝ) (h1 : x1 + x2 = 2 * sqrt 1703) (h2 : abs (x1 - x2) = 90) : x1 * x2 = -322 := by
  sorry

end x1_mul_x2_l831_83143


namespace work_problem_correct_l831_83123

noncomputable def work_problem : Prop :=
  let A := 1 / 36
  let C := 1 / 6
  let total_rate := 1 / 4
  ∃ B : ℝ, (A + B + C = total_rate) ∧ (B = 1 / 18)

-- Create the theorem statement which says if the conditions are met,
-- then the rate of b must be 1/18 and the number of days b alone takes to
-- finish the work is 18.
theorem work_problem_correct (A C total_rate B : ℝ) (h1 : A = 1 / 36) (h2 : C = 1 / 6) (h3 : total_rate = 1 / 4) 
(h4 : A + B + C = total_rate) : B = 1 / 18 ∧ (1 / B = 18) :=
  by
  sorry

end work_problem_correct_l831_83123


namespace people_not_in_any_club_l831_83133

def num_people_company := 120
def num_people_club_A := 25
def num_people_club_B := 34
def num_people_club_C := 21
def num_people_club_D := 16
def num_people_club_E := 10
def overlap_C_D := 8
def overlap_D_E := 4

theorem people_not_in_any_club :
  num_people_company - 
  (num_people_club_A + num_people_club_B + 
  (num_people_club_C + (num_people_club_D - overlap_C_D) + (num_people_club_E - overlap_D_E))) = 26 :=
by
  unfold num_people_company num_people_club_A num_people_club_B num_people_club_C num_people_club_D num_people_club_E overlap_C_D overlap_D_E
  sorry

end people_not_in_any_club_l831_83133


namespace isosceles_obtuse_triangle_smallest_angle_l831_83119

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β γ : ℝ), α = 1.8 * 90 ∧ β = γ ∧ α + β + γ = 180 → β = 9 :=
by
  intros α β γ h
  sorry

end isosceles_obtuse_triangle_smallest_angle_l831_83119


namespace calc_2002_sq_minus_2001_mul_2003_l831_83145

theorem calc_2002_sq_minus_2001_mul_2003 : 2002 ^ 2 - 2001 * 2003 = 1 := 
by
  sorry

end calc_2002_sq_minus_2001_mul_2003_l831_83145


namespace cost_of_cd_l831_83127

theorem cost_of_cd 
  (cost_film : ℕ) (cost_book : ℕ) (total_spent : ℕ) (num_cds : ℕ) (total_cost_films : ℕ)
  (total_cost_books : ℕ) (cost_cd : ℕ) : 
  cost_film = 5 → cost_book = 4 → total_spent = 79 →
  total_cost_films = 9 * cost_film → total_cost_books = 4 * cost_book →
  total_spent = total_cost_films + total_cost_books + num_cds * cost_cd →
  num_cds = 6 →
  cost_cd = 3 := 
by {
  -- proof would go here
  sorry
}

end cost_of_cd_l831_83127


namespace total_bars_is_7_l831_83175

variable (x : ℕ)

-- Each chocolate bar costs $3
def cost_per_bar := 3

-- Olivia sold all but 4 bars
def bars_sold (total_bars : ℕ) := total_bars - 4

-- Olivia made $9
def amount_made (total_bars : ℕ) := cost_per_bar * bars_sold total_bars

-- Given conditions
def condition1 (total_bars : ℕ) := amount_made total_bars = 9

-- Proof that the total number of bars is 7
theorem total_bars_is_7 : condition1 x -> x = 7 := by
  sorry

end total_bars_is_7_l831_83175


namespace number_of_rows_l831_83195

theorem number_of_rows (total_pencils : ℕ) (pencils_per_row : ℕ) (h1 : total_pencils = 30) (h2 : pencils_per_row = 5) : total_pencils / pencils_per_row = 6 :=
by
  sorry

end number_of_rows_l831_83195


namespace proof_MrLalandeInheritance_l831_83103

def MrLalandeInheritance : Nat := 18000
def initialPayment : Nat := 3000
def monthlyInstallment : Nat := 2500
def numInstallments : Nat := 6

theorem proof_MrLalandeInheritance :
  initialPayment + numInstallments * monthlyInstallment = MrLalandeInheritance := 
by 
  sorry

end proof_MrLalandeInheritance_l831_83103


namespace find_other_root_l831_83138

theorem find_other_root (x y : ℚ) (h : 48 * x^2 - 77 * x + 21 = 0) (hx : x = 3 / 4) : y = 7 / 12 → 48 * y^2 - 77 * y + 21 = 0 := by
  sorry

end find_other_root_l831_83138


namespace number_of_integers_x_l831_83173

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def is_acute_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 > c^2 ∧ a^2 + c^2 > b^2 ∧ b^2 + c^2 > a^2

def valid_range_x (x : ℝ) : Prop :=
  13 < x ∧ x < 43

def conditions_for_acute_triangle (x : ℝ) : Prop :=
  (x > 28 ∧ x^2 < 1009) ∨ (x ≤ 28 ∧ x > 23.64)

theorem number_of_integers_x (count : ℤ) :
  (∃ (x : ℤ), valid_range_x x ∧ is_triangle 15 28 x ∧ is_acute_triangle 15 28 x ∧ conditions_for_acute_triangle x) →
  count = 8 :=
sorry

end number_of_integers_x_l831_83173


namespace minimum_k_condition_l831_83198

def is_acute_triangle (a b c : ℕ) : Prop :=
  a * a + b * b > c * c

def any_subset_with_three_numbers_construct_acute_triangle (s : Finset ℕ) : Prop :=
  ∀ t : Finset ℕ, t.card = 3 → 
    (∃ a b c : ℕ, a ∈ t ∧ b ∈ t ∧ c ∈ t ∧ 
      is_acute_triangle a b c ∨
      is_acute_triangle a c b ∨
      is_acute_triangle b c a)

theorem minimum_k_condition (k : ℕ) :
  (∀ s : Finset ℕ, s.card = k → any_subset_with_three_numbers_construct_acute_triangle s) ↔ (k = 29) :=
  sorry

end minimum_k_condition_l831_83198


namespace min_value_of_expression_l831_83169

theorem min_value_of_expression (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  25 ≤ (4 / a) + (9 / b) :=
sorry

end min_value_of_expression_l831_83169


namespace find_fourth_speed_l831_83197

theorem find_fourth_speed 
  (avg_speed : ℝ)
  (speed1 speed2 speed3 fourth_speed : ℝ)
  (h_avg_speed : avg_speed = 11.52)
  (h_speed1 : speed1 = 6.0)
  (h_speed2 : speed2 = 12.0)
  (h_speed3 : speed3 = 18.0)
  (expected_avg_speed_eq : avg_speed = 4 / ((1 / speed1) + (1 / speed2) + (1 / speed3) + (1 / fourth_speed))) :
  fourth_speed = 2.095 :=
by 
  sorry

end find_fourth_speed_l831_83197


namespace range_of_2a_plus_3b_l831_83101

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 :=
sorry

end range_of_2a_plus_3b_l831_83101


namespace largest_area_of_triangle_DEF_l831_83105

noncomputable def maxAreaTriangleDEF : Real :=
  let DE := 16.0
  let EF_to_FD := 25.0 / 24.0
  let max_area := 446.25
  max_area

theorem largest_area_of_triangle_DEF :
  ∀ (DE : Real) (EF FD : Real),
    DE = 16 ∧ EF / FD = 25 / 24 → 
    (∃ (area : Real), area ≤ maxAreaTriangleDEF) :=
by 
  sorry

end largest_area_of_triangle_DEF_l831_83105


namespace factorize_expression_l831_83161

theorem factorize_expression (x y a : ℝ) : x * (a - y) - y * (y - a) = (x + y) * (a - y) := 
by 
  sorry

end factorize_expression_l831_83161


namespace business_total_profit_l831_83163

noncomputable def total_profit (spending_ratio income_ratio total_income : ℕ) : ℕ :=
  let total_parts := spending_ratio + income_ratio
  let one_part_value := total_income / income_ratio
  let spending := spending_ratio * one_part_value
  total_income - spending

theorem business_total_profit :
  total_profit 5 9 108000 = 48000 :=
by
  -- We omit the proof steps, as instructed.
  sorry

end business_total_profit_l831_83163


namespace find_value_of_expression_l831_83150

-- Conditions as provided
axiom given_condition : ∃ (x : ℕ), 3^x + 3^x + 3^x + 3^x = 2187

-- Proof statement
theorem find_value_of_expression : (exists (x : ℕ), (3^x + 3^x + 3^x + 3^x = 2187) ∧ ((x + 2) * (x - 2) = 21)) :=
sorry

end find_value_of_expression_l831_83150


namespace probability_of_at_least_40_cents_l831_83148

-- Definitions for each type of coin and their individual values in cents.
def penny := 1
def nickel := 5
def dime := 10
def quarter := 25
def half_dollar := 50

-- The total value needed for a successful outcome
def minimum_success_value := 40

-- Total number of possible outcomes from flipping 5 coins independently
def total_outcomes := 2^5

-- Count the successful outcomes that result in at least 40 cents
-- This is a placeholder for the actual successful counting method
noncomputable def successful_outcomes := 18

-- Calculate the probability of successful outcomes
noncomputable def probability := (successful_outcomes : ℚ) / total_outcomes

-- Proof statement to show the probability is 9/16
theorem probability_of_at_least_40_cents : probability = 9 / 16 := 
by
  sorry

end probability_of_at_least_40_cents_l831_83148


namespace greatest_num_consecutive_integers_sum_eq_36_l831_83159

theorem greatest_num_consecutive_integers_sum_eq_36 :
    ∃ a : ℤ, ∃ N : ℕ, N > 0 ∧ (N = 9) ∧ (N * (2 * a + N - 1) = 72) :=
sorry

end greatest_num_consecutive_integers_sum_eq_36_l831_83159


namespace natasha_can_achieve_plan_l831_83191

noncomputable def count_ways : Nat :=
  let num_1x1 := 4
  let num_1x2 := 24
  let target := 2021
  6517

theorem natasha_can_achieve_plan (num_1x1 num_1x2 target : Nat) (h1 : num_1x1 = 4) (h2 : num_1x2 = 24) (h3 : target = 2021) :
  count_ways = 6517 :=
by
  sorry

end natasha_can_achieve_plan_l831_83191


namespace relationship_among_a_b_and_ab_l831_83164

noncomputable def a : ℝ := Real.log 0.4 / Real.log 0.2
noncomputable def b : ℝ := 1 - (1 / (Real.log 4 / Real.log 10))

theorem relationship_among_a_b_and_ab : a * b < a + b ∧ a + b < 0 := by
  sorry

end relationship_among_a_b_and_ab_l831_83164


namespace sin_of_alpha_l831_83118

theorem sin_of_alpha 
  (α : ℝ) 
  (h : Real.cos (α - Real.pi / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := 
by 
  sorry

end sin_of_alpha_l831_83118
