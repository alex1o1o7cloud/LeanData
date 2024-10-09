import Mathlib

namespace average_weight_of_whole_class_l2170_217021

def num_students_a : ℕ := 50
def num_students_b : ℕ := 70
def avg_weight_a : ℚ := 50
def avg_weight_b : ℚ := 70

theorem average_weight_of_whole_class :
  (num_students_a * avg_weight_a + num_students_b * avg_weight_b) / (num_students_a + num_students_b) = 61.67 := by
  sorry

end average_weight_of_whole_class_l2170_217021


namespace puppies_adopted_each_day_l2170_217032

variable (initial_puppies additional_puppies days total_puppies puppies_per_day : ℕ)

axiom initial_puppies_ax : initial_puppies = 9
axiom additional_puppies_ax : additional_puppies = 12
axiom days_ax : days = 7
axiom total_puppies_ax : total_puppies = initial_puppies + additional_puppies
axiom adoption_rate_ax : total_puppies / days = puppies_per_day

theorem puppies_adopted_each_day : 
  initial_puppies = 9 → additional_puppies = 12 → days = 7 → total_puppies = initial_puppies + additional_puppies → total_puppies / days = puppies_per_day → puppies_per_day = 3 :=
by
  intro initial_puppies_ax additional_puppies_ax days_ax total_puppies_ax adoption_rate_ax
  sorry

end puppies_adopted_each_day_l2170_217032


namespace max_value_a_l2170_217040

def no_lattice_points (m : ℚ) : Prop :=
  ∀ (x : ℤ), 0 < x ∧ x ≤ 150 → ¬∃ (y : ℤ), y = m * x + 3

def valid_m (m : ℚ) (a : ℚ) : Prop :=
  (2 : ℚ) / 3 < m ∧ m < a

theorem max_value_a (a : ℚ) : (a = 101 / 151) ↔ 
  ∀ (m : ℚ), valid_m m a → no_lattice_points m :=
sorry

end max_value_a_l2170_217040


namespace find_a_of_odd_function_l2170_217063

noncomputable def f (a : ℝ) (x : ℝ) := 1 + a / (2^x + 1)

theorem find_a_of_odd_function (a : ℝ) (h : ∀ x : ℝ, f a x = -f a (-x)) : a = -2 :=
by
  sorry

end find_a_of_odd_function_l2170_217063


namespace evaluate_f_5_minus_f_neg_5_l2170_217066

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x + 3

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := 
  by
    sorry

end evaluate_f_5_minus_f_neg_5_l2170_217066


namespace base_eight_to_base_ten_l2170_217009

theorem base_eight_to_base_ten (n : ℕ) : 
  n = 3 * 8^1 + 1 * 8^0 → n = 25 :=
by
  intro h
  rw [mul_comm 3 (8^1), pow_one, mul_comm 1 (8^0), pow_zero, mul_one] at h
  exact h

end base_eight_to_base_ten_l2170_217009


namespace minimum_pyramid_volume_proof_l2170_217046

noncomputable def minimum_pyramid_volume (side_length : ℝ) (apex_angle : ℝ) : ℝ :=
  if side_length = 6 ∧ apex_angle = 2 * Real.arcsin (1 / 3 : ℝ) then 5 * Real.sqrt 23 else 0

theorem minimum_pyramid_volume_proof : 
  minimum_pyramid_volume 6 (2 * Real.arcsin (1 / 3)) = 5 * Real.sqrt 23 :=
by
  sorry

end minimum_pyramid_volume_proof_l2170_217046


namespace green_chips_count_l2170_217008

def total_chips : ℕ := 60
def fraction_blue_chips : ℚ := 1 / 6
def num_red_chips : ℕ := 34

theorem green_chips_count :
  let num_blue_chips := total_chips * fraction_blue_chips
  let chips_not_green := num_blue_chips + num_red_chips
  let num_green_chips := total_chips - chips_not_green
  num_green_chips = 16 := by
    let num_blue_chips := total_chips * fraction_blue_chips
    let chips_not_green := num_blue_chips + num_red_chips
    let num_green_chips := total_chips - chips_not_green
    show num_green_chips = 16
    sorry

end green_chips_count_l2170_217008


namespace benjamin_earns_more_l2170_217012

noncomputable def additional_earnings : ℝ :=
  let P : ℝ := 75000
  let r : ℝ := 0.05
  let t_M : ℝ := 3
  let r_m : ℝ := r / 12
  let t_B : ℝ := 36
  let A_M : ℝ := P * (1 + r)^t_M
  let A_B : ℝ := P * (1 + r_m)^t_B
  A_B - A_M

theorem benjamin_earns_more : additional_earnings = 204 := by
  sorry

end benjamin_earns_more_l2170_217012


namespace find_sticker_price_l2170_217011

-- Define the conditions
def storeX_discount (x : ℝ) : ℝ := 0.80 * x - 70
def storeY_discount (x : ℝ) : ℝ := 0.70 * x

-- Define the main statement
theorem find_sticker_price (x : ℝ) (h : storeX_discount x = storeY_discount x - 20) : x = 500 :=
sorry

end find_sticker_price_l2170_217011


namespace parallel_lines_slope_l2170_217041

theorem parallel_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2 * y + 1 = 0 → ∀ x y : ℝ, x + y - 2 = 0 → True) → 
  a = 2 :=
by
  sorry

end parallel_lines_slope_l2170_217041


namespace julia_stairs_less_than_third_l2170_217067

theorem julia_stairs_less_than_third (J1 : ℕ) (T : ℕ) (T_total : ℕ) (J : ℕ) 
  (hJ1 : J1 = 1269) (hT : T = 1269 / 3) (hT_total : T_total = 1685) (hTotal : J1 + J = T_total) : 
  T - J = 7 := 
by
  sorry

end julia_stairs_less_than_third_l2170_217067


namespace multiplication_equation_l2170_217089

-- Define the given conditions
def multiplier : ℕ := 6
def product : ℕ := 168
def multiplicand : ℕ := product - 140

-- Lean statement for the proof
theorem multiplication_equation : multiplier * multiplicand = product := by
  sorry

end multiplication_equation_l2170_217089


namespace john_needs_2_sets_l2170_217010

-- Definition of the conditions
def num_bars_per_set : ℕ := 7
def total_bars : ℕ := 14

-- The corresponding proof problem statement
theorem john_needs_2_sets : total_bars / num_bars_per_set = 2 :=
by
  sorry

end john_needs_2_sets_l2170_217010


namespace donuts_purchased_l2170_217017

/-- John goes to a bakery every day for a four-day workweek and chooses between a 
    60-cent croissant or a 90-cent donut. At the end of the week, he spent a whole 
    number of dollars. Prove that he must have purchased 2 donuts. -/
theorem donuts_purchased (d c : ℕ) (h1 : d + c = 4) (h2 : 90 * d + 60 * c % 100 = 0) : d = 2 :=
sorry

end donuts_purchased_l2170_217017


namespace zebra_difference_is_zebra_l2170_217018

/-- 
A zebra number is a non-negative integer in which the digits strictly alternate between even and odd.
Given two 100-digit zebra numbers, prove that their difference is still a 100-digit zebra number.
-/
theorem zebra_difference_is_zebra 
  (A B : ℕ) 
  (hA : (∀ i, (A / 10^i % 10) % 2 = i % 2) ∧ (A / 10^100 = 0) ∧ (A > 10^99))
  (hB : (∀ i, (B / 10^i % 10) % 2 = i % 2) ∧ (B / 10^100 = 0) ∧ (B > 10^99)) 
  : (∀ j, (((A - B) / 10^j) % 10) % 2 = j % 2) ∧ ((A - B) / 10^100 = 0) ∧ ((A - B) > 10^99) :=
sorry

end zebra_difference_is_zebra_l2170_217018


namespace min_value_of_y_l2170_217065

theorem min_value_of_y {y : ℤ} (h : ∃ x : ℤ, y^2 = (0 ^ 2 + 1 ^ 2 + 2 ^ 2 + 3 ^ 2 + 4 ^ 2 + 5 ^ 2 + (-1) ^ 2 + (-2) ^ 2 + (-3) ^ 2 + (-4) ^ 2 + (-5) ^ 2)) :
  y = -11 :=
by sorry

end min_value_of_y_l2170_217065


namespace length_of_room_l2170_217069

theorem length_of_room (L : ℝ) 
  (h_width : 12 > 0) 
  (h_veranda_width : 2 > 0) 
  (h_area_veranda : (L + 4) * 16 - L * 12 = 140) : 
  L = 19 := 
by
  sorry

end length_of_room_l2170_217069


namespace common_divisor_seven_l2170_217072

-- Definition of numbers A, B, and C based on given conditions
def A (m n : ℤ) : ℤ := n^2 + 2 * m * n + 3 * m^2 + 2
def B (m n : ℤ) : ℤ := 2 * n^2 + 3 * m * n + m^2 + 2
def C (m n : ℤ) : ℤ := 3 * n^2 + m * n + 2 * m^2 + 1

-- The proof statement ensuring A, B and C have a common divisor of 7
theorem common_divisor_seven (m n : ℤ) : ∃ d : ℤ, d > 1 ∧ d ∣ A m n ∧ d ∣ B m n ∧ d ∣ C m n → d = 7 :=
by
  sorry

end common_divisor_seven_l2170_217072


namespace math_problem_solution_l2170_217068

noncomputable def math_problem (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) : ℝ :=
  (a * b + c * d) / (b * c + a * d)

theorem math_problem_solution (a b c d : ℝ) (h1 : a^2 + b^2 - c^2 - d^2 = 0) (h2 : a^2 - b^2 - c^2 + d^2 = (56 / 53) * (b * c + a * d)) :
  math_problem a b c d h1 h2 = 45 / 53 := sorry

end math_problem_solution_l2170_217068


namespace math_problem_l2170_217043

def letters := "MATHEMATICS".toList

def vowels := "AAEII".toList
def consonants := "MTHMTCS".toList
def fixed_t := 'T'

def factorial (n : Nat) : Nat := 
  if n = 0 then 1 
  else n * factorial (n - 1)

def arrangements (n : Nat) (reps : List Nat) : Nat := 
  factorial n / reps.foldr (fun r acc => factorial r * acc) 1

noncomputable def vowel_arrangements := arrangements 5 [2, 2]
noncomputable def consonant_arrangements := arrangements 6 [2]

noncomputable def total_arrangements := vowel_arrangements * consonant_arrangements

theorem math_problem : total_arrangements = 10800 := by
  sorry

end math_problem_l2170_217043


namespace rhombus_area_l2170_217078

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 6) (h2 : d2 = 8) : 
  (1 / 2) * d1 * d2 = 24 :=
by {
  sorry
}

end rhombus_area_l2170_217078


namespace triangle_is_isosceles_l2170_217052

theorem triangle_is_isosceles 
  (a b c : ℝ)
  (h : a^2 - b^2 + a * c - b * c = 0)
  (h_tri : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  : a = b := 
sorry

end triangle_is_isosceles_l2170_217052


namespace polynomial_sum_is_integer_l2170_217002

-- Define the integer polynomial and the integers a and b
variables (f : ℤ[X]) (a b : ℤ)

-- The theorem statement
theorem polynomial_sum_is_integer :
  ∃ c : ℤ, f.eval (a - real.sqrt b) + f.eval (a + real.sqrt b) = c :=
sorry

end polynomial_sum_is_integer_l2170_217002


namespace fraction_identity_l2170_217038

theorem fraction_identity (a b : ℝ) (h : 2 * a = 5 * b) : a / b = 5 / 2 := by
  sorry

end fraction_identity_l2170_217038


namespace find_angle_at_A_l2170_217085

def triangle_angles_sum_to_180 (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def ab_lt_bc_lt_ac (AB BC AC : ℝ) : Prop :=
  AB < BC ∧ BC < AC

def angles_relation (α β γ : ℝ) : Prop :=
  (α = 2 * γ) ∧ (β = 3 * γ)

theorem find_angle_at_A
  (AB BC AC : ℝ)
  (α β γ : ℝ)
  (h1 : ab_lt_bc_lt_ac AB BC AC)
  (h2 : angles_relation α β γ)
  (h3 : triangle_angles_sum_to_180 α β γ) :
  α = 60 :=
sorry

end find_angle_at_A_l2170_217085


namespace area_of_triangle_BXC_l2170_217019

/-
  Given:
  - AB = 15 units
  - CD = 40 units
  - The area of trapezoid ABCD = 550 square units

  To prove:
  - The area of triangle BXC = 1200 / 11 square units
-/
theorem area_of_triangle_BXC 
  (AB CD : ℝ) 
  (hAB : AB = 15) 
  (hCD : CD = 40) 
  (area_ABCD : ℝ)
  (hArea_ABCD : area_ABCD = 550) 
  : ∃ (area_BXC : ℝ), area_BXC = 1200 / 11 :=
by
  sorry

end area_of_triangle_BXC_l2170_217019


namespace squirrel_rise_per_circuit_l2170_217062

noncomputable def rise_per_circuit
    (height : ℕ)
    (circumference : ℕ)
    (distance : ℕ) :=
    height / (distance / circumference)

theorem squirrel_rise_per_circuit : rise_per_circuit 25 3 15 = 5 :=
by
  sorry

end squirrel_rise_per_circuit_l2170_217062


namespace inclination_angle_range_l2170_217091

theorem inclination_angle_range (k : ℝ) (α : ℝ) (h1 : -1 ≤ k) (h2 : k < 1)
  (h3 : k = Real.tan α) (h4 : 0 ≤ α) (h5 : α < 180) :
  (0 ≤ α ∧ α < 45) ∨ (135 ≤ α ∧ α < 180) :=
sorry

end inclination_angle_range_l2170_217091


namespace pages_per_comic_l2170_217073

variable {comics_initial : ℕ} -- initially 5 untorn comics in the box
variable {comics_final : ℕ}   -- now there are 11 comics in the box
variable {pages_found : ℕ}    -- found 150 pages on the floor
variable {comics_assembled : ℕ} -- comics assembled from the found pages

theorem pages_per_comic (h1 : comics_initial = 5) (h2 : comics_final = 11) 
      (h3 : pages_found = 150) (h4 : comics_assembled = comics_final - comics_initial) :
      (pages_found / comics_assembled = 25) := 
sorry

end pages_per_comic_l2170_217073


namespace range_of_a_l2170_217083

open Real

theorem range_of_a (a b c : ℝ) (h₁ : a ≠ 0) (h₂ : a - b + c = 3) (h₃ : a + b + c = 1) (h₄ : 0 < c ∧ c < 1) : 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l2170_217083


namespace both_sports_l2170_217055

-- Definitions based on the given conditions
def total_members := 80
def badminton_players := 48
def tennis_players := 46
def neither_players := 7

-- The theorem to be proved
theorem both_sports : (badminton_players + tennis_players - (total_members - neither_players)) = 21 := by
  sorry

end both_sports_l2170_217055


namespace container_solution_exists_l2170_217030

theorem container_solution_exists (x y : ℕ) (h : 130 * x + 160 * y = 3000) : 
  (x = 12) ∧ (y = 9) :=
by sorry

end container_solution_exists_l2170_217030


namespace prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l2170_217035

-- Define the conditions
def quadratic_has_two_different_negative_roots (a : ℝ) : Prop :=
  a^2 - 1/4 > 0 ∧ -a < 0 ∧ 1/16 > 0

def inequality_q (a : ℝ) : Prop :=
  0 < a ∧ a < 1

-- Prove the results based on the conditions
theorem prove_a_range_if_p (a : ℝ) (hp : quadratic_has_two_different_negative_roots a) : a > 1/2 :=
  sorry

theorem prove_a_range_if_p_or_q_and_not_and (a : ℝ) (hp_or_q : quadratic_has_two_different_negative_roots a ∨ inequality_q a) 
  (hnot_p_and_q : ¬ (quadratic_has_two_different_negative_roots a ∧ inequality_q a)) :
  a ≥ 1 ∨ (0 < a ∧ a ≤ 1/2) :=
  sorry

end prove_a_range_if_p_prove_a_range_if_p_or_q_and_not_and_l2170_217035


namespace millet_percentage_in_mix_l2170_217099

def contribution_millet_brandA (percA mixA : ℝ) := percA * mixA
def contribution_millet_brandB (percB mixB : ℝ) := percB * mixB

theorem millet_percentage_in_mix
  (percA : ℝ) (percB : ℝ) (mixA : ℝ) (mixB : ℝ)
  (h1 : percA = 0.40) (h2 : percB = 0.65) (h3 : mixA = 0.60) (h4 : mixB = 0.40) :
  (contribution_millet_brandA percA mixA + contribution_millet_brandB percB mixB = 0.50) :=
by
  sorry

end millet_percentage_in_mix_l2170_217099


namespace train_length_l2170_217059

theorem train_length
    (V : ℝ) -- train speed in m/s
    (L : ℝ) -- length of the train in meters
    (H1 : L = V * 18) -- condition: train crosses signal pole in 18 sec
    (H2 : L + 333.33 = V * 38) -- condition: train crosses platform in 38 sec
    (V_pos : 0 < V) -- additional condition: speed must be positive
    : L = 300 :=
by
-- here goes the proof which is not required for our task
sorry

end train_length_l2170_217059


namespace rowers_voted_l2170_217024

variable (R : ℕ)

/-- Each rower votes for exactly 4 coaches out of 50 coaches,
and each coach receives exactly 7 votes.
Prove that the number of rowers is 88. -/
theorem rowers_voted (h1 : 50 * 7 = 4 * R) : R = 88 := by 
  sorry

end rowers_voted_l2170_217024


namespace green_pill_cost_l2170_217056

-- Given conditions
def days := 21
def total_cost := 903
def cost_difference := 2
def daily_cost := total_cost / days

-- Statement to prove
theorem green_pill_cost : (∃ (y : ℝ), y + (y - cost_difference) = daily_cost ∧ y = 22.5) :=
by
  sorry

end green_pill_cost_l2170_217056


namespace correct_subtraction_result_l2170_217080

-- Definitions based on the problem conditions
def initial_two_digit_number (X Y : ℕ) : ℕ := X * 10 + Y

-- Lean statement that expresses the proof problem
theorem correct_subtraction_result (X Y : ℕ) (H1 : initial_two_digit_number X Y = 99) (H2 : 57 = 57) :
  99 - 57 = 42 :=
by
  sorry

end correct_subtraction_result_l2170_217080


namespace factor_expression_l2170_217014

theorem factor_expression (a : ℝ) : 74 * a^2 + 222 * a + 148 = 74 * (a + 2) * (a + 1) :=
by
  sorry

end factor_expression_l2170_217014


namespace ratio_of_areas_of_concentric_circles_l2170_217064

theorem ratio_of_areas_of_concentric_circles
  (Q : Type)
  (r₁ r₂ : ℝ)
  (C₁ C₂ : ℝ)
  (h₀ : r₁ > 0 ∧ r₂ > 0)
  (h₁ : C₁ = 2 * π * r₁)
  (h₂ : C₂ = 2 * π * r₂)
  (h₃ : (60 / 360) * C₁ = (30 / 360) * C₂) :
  (π * r₁^2) / (π * r₂^2) = 1 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_l2170_217064


namespace physics_marks_l2170_217048

theorem physics_marks (P C M : ℕ) 
  (h1 : P + C + M = 195)
  (h2 : P + M = 180)
  (h3 : P + C = 140) :
  P = 125 :=
by {
  sorry
}

end physics_marks_l2170_217048


namespace ball_hits_ground_in_2_72_seconds_l2170_217087

noncomputable def height_at_time (t : ℝ) : ℝ :=
  -16 * t^2 - 30 * t + 200

theorem ball_hits_ground_in_2_72_seconds :
  ∃ t : ℝ, t = 2.72 ∧ height_at_time t = 0 :=
by
  use 2.72
  sorry

end ball_hits_ground_in_2_72_seconds_l2170_217087


namespace sum_first_95_odds_equals_9025_l2170_217096

-- Define the nth odd positive integer
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of the first n odd positive integers
def sum_first_n_odds (n : ℕ) : ℕ := n^2

-- State the theorem to be proved
theorem sum_first_95_odds_equals_9025 : sum_first_n_odds 95 = 9025 :=
by
  -- We provide a placeholder for the proof
  sorry

end sum_first_95_odds_equals_9025_l2170_217096


namespace rate_of_rainfall_on_Monday_l2170_217034

theorem rate_of_rainfall_on_Monday (R : ℝ) :
  7 * R + 4 * 2 + 2 * (2 * 2) = 23 → R = 1 := 
by
  sorry

end rate_of_rainfall_on_Monday_l2170_217034


namespace twelve_div_one_fourth_eq_48_l2170_217060

theorem twelve_div_one_fourth_eq_48 : 12 / (1 / 4) = 48 := by
  -- We know that dividing by a fraction is equivalent to multiplying by its reciprocal
  sorry

end twelve_div_one_fourth_eq_48_l2170_217060


namespace certain_number_is_47_l2170_217031

theorem certain_number_is_47 (x : ℤ) (h : 34 + x - 53 = 28) : x = 47 :=
by
  sorry

end certain_number_is_47_l2170_217031


namespace num_solutions_congruence_l2170_217028

-- Define the problem context and conditions
def is_valid_solution (y : ℕ) : Prop :=
  y < 150 ∧ (y + 21) % 46 = 79 % 46

-- Define the proof problem
theorem num_solutions_congruence : ∃ (s : Finset ℕ), s.card = 3 ∧ ∀ y ∈ s, is_valid_solution y := by
  sorry

end num_solutions_congruence_l2170_217028


namespace area_new_rectangle_greater_than_square_l2170_217084

theorem area_new_rectangle_greater_than_square (a b : ℝ) (h : a > b) : 
  (2 * (a + b) * (2 * b + a) / 3) > ((a + b) * (a + b)) := 
sorry

end area_new_rectangle_greater_than_square_l2170_217084


namespace new_boarders_l2170_217047

theorem new_boarders (init_boarders : ℕ) (init_day_students : ℕ) (ratio_b : ℕ) (ratio_d : ℕ) (ratio_new_b : ℕ) (ratio_new_d : ℕ) (x : ℕ) :
    init_boarders = 240 →
    ratio_b = 8 →
    ratio_d = 17 →
    ratio_new_b = 3 →
    ratio_new_d = 7 →
    init_day_students = (init_boarders * ratio_d) / ratio_b →
    (ratio_new_b * init_day_students) = ratio_new_d * (init_boarders + x) →
    x = 21 :=
by sorry

end new_boarders_l2170_217047


namespace intersection_domain_range_l2170_217094

-- Define domain and function
def domain : Set ℝ := {-1, 0, 1}
def f (x : ℝ) : ℝ := |x|

-- Prove the theorem
theorem intersection_domain_range :
  let range : Set ℝ := {y | ∃ x ∈ domain, f x = y}
  let A : Set ℝ := domain
  let B : Set ℝ := range 
  A ∩ B = {0, 1} :=
by
  -- The proof is skipped with sorry
  sorry

end intersection_domain_range_l2170_217094


namespace tan_105_degree_l2170_217020

theorem tan_105_degree : Real.tan (Real.pi * 105 / 180) = -2 - Real.sqrt 3 :=
by
  sorry

end tan_105_degree_l2170_217020


namespace arithmetic_sequence_S22_zero_l2170_217037

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ :=
  a + (n - 1) * d

noncomputable def sum_of_first_n_terms (a d : ℝ) (n : ℕ) : ℝ :=
  (n / 2) * (2 * a + (n - 1) * d)

theorem arithmetic_sequence_S22_zero (a d : ℝ) (S : ℕ → ℝ) (h_arith_seq : ∀ n, S n = sum_of_first_n_terms a d n)
  (h1 : a > 0) (h2 : S 5 = S 17) :
  S 22 = 0 :=
by
  sorry

end arithmetic_sequence_S22_zero_l2170_217037


namespace vasya_average_not_exceed_4_l2170_217075

variable (a b c d e : ℕ) 

-- Total number of grades
def total_grades : ℕ := a + b + c + d + e

-- Initial average condition
def initial_condition : Prop := 
  (a + 2 * b + 3 * c + 4 * d + 5 * e) < 3 * (total_grades a b c d e)

-- New average condition after grade changes
def changed_average (a b c d e : ℕ) : ℚ := 
  ((2 * b + 3 * (a + c) + 4 * d + 5 * e) : ℚ) / (total_grades a b c d e)

-- Proof problem to show the new average grade does not exceed 4
theorem vasya_average_not_exceed_4 (h : initial_condition a b c d e) : 
  (changed_average 0 b (c + a) d e) ≤ 4 := 
sorry

end vasya_average_not_exceed_4_l2170_217075


namespace statement_C_is_incorrect_l2170_217079

noncomputable def g (x : ℝ) : ℝ := (2 * x + 3) / (x - 2)

theorem statement_C_is_incorrect : g (-2) ≠ 0 :=
by
  sorry

end statement_C_is_incorrect_l2170_217079


namespace fraction_value_l2170_217054

theorem fraction_value (x : ℝ) (h : x + 1/x = 3) : x^2 / (x^4 + x^2 + 1) = 1/8 :=
by sorry

end fraction_value_l2170_217054


namespace simplify_sqrt_product_l2170_217077

theorem simplify_sqrt_product (x : ℝ) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (28 * x) * Real.sqrt (5 * x) =
  60 * x^2 * Real.sqrt 35 :=
by
  sorry

end simplify_sqrt_product_l2170_217077


namespace book_page_count_l2170_217057

theorem book_page_count:
  (∃ (total_pages : ℕ), 
    (∃ (days_read : ℕ) (pages_per_day : ℕ), 
      days_read = 12 ∧ 
      pages_per_day = 8 ∧ 
      (days_read * pages_per_day) = 2 * (total_pages / 3)) 
  ↔ total_pages = 144) :=
by 
  sorry

end book_page_count_l2170_217057


namespace sqrt_of_square_neg_five_eq_five_l2170_217015

theorem sqrt_of_square_neg_five_eq_five :
  Real.sqrt ((-5 : ℝ)^2) = 5 := 
by
  sorry

end sqrt_of_square_neg_five_eq_five_l2170_217015


namespace marble_probability_l2170_217023

theorem marble_probability (g w r b : ℕ) (h_g : g = 4) (h_w : w = 3) (h_r : r = 5) (h_b : b = 6) :
  (g + w + r + b = 18) → (g + w = 7) → (7 / 18 = 7 / 18) :=
by
  sorry

end marble_probability_l2170_217023


namespace lcm_first_ten_integers_l2170_217061

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l2170_217061


namespace balls_in_base_l2170_217044

theorem balls_in_base (n k : ℕ) (h1 : 165 = (n * (n + 1) * (n + 2)) / 6) (h2 : k = n * (n + 1) / 2) : k = 45 := 
by 
  sorry

end balls_in_base_l2170_217044


namespace find_x_eq_l2170_217050

-- Given conditions
variables (c b θ : ℝ)

-- The proof problem
theorem find_x_eq :
  ∃ x : ℝ, x^2 + c^2 * (Real.sin θ)^2 = (b - x)^2 ∧
          x = (b^2 - c^2 * (Real.sin θ)^2) / (2 * b) :=
by
    sorry

end find_x_eq_l2170_217050


namespace minimum_balls_ensure_20_single_color_l2170_217039

def num_balls_to_guarantee_color (r g y b w k : ℕ) : ℕ :=
  let max_without_20 := 19 + 19 + 19 + 18 + 15 + 12
  max_without_20 + 1

theorem minimum_balls_ensure_20_single_color :
  num_balls_to_guarantee_color 30 25 25 18 15 12 = 103 := by
  sorry

end minimum_balls_ensure_20_single_color_l2170_217039


namespace farmer_cages_l2170_217006

theorem farmer_cages (c : ℕ) (h1 : 164 + 6 = 170) (h2 : ∃ r : ℕ, c * r = 170) (h3 : ∃ r : ℕ, c * r > 164) :
  c = 10 :=
by
  sorry

end farmer_cages_l2170_217006


namespace total_points_correct_l2170_217076

-- Define the scores
def Marius (Darius : ℕ) : ℕ := Darius + 3
def Matt (Darius : ℕ) : ℕ := Darius + 5

-- Define the total points function
def total_points (Darius : ℕ) : ℕ :=
  Darius + Marius Darius + Matt Darius

-- Specific value for Darius's score
def Darius_score : ℕ := 10

-- The theorem that proves the total score is 38 given Darius's score
theorem total_points_correct :
  total_points Darius_score = 38 :=
by
  sorry

end total_points_correct_l2170_217076


namespace winner_won_by_324_votes_l2170_217005

theorem winner_won_by_324_votes
  (total_votes : ℝ)
  (winner_percentage : ℝ)
  (winner_votes : ℝ)
  (h1 : winner_percentage = 0.62)
  (h2 : winner_votes = 837) :
  (winner_votes - (0.38 * total_votes) = 324) :=
by
  sorry

end winner_won_by_324_votes_l2170_217005


namespace heptagon_divisibility_impossible_l2170_217000

theorem heptagon_divisibility_impossible (a b c d e f g : ℕ) :
  (b ∣ a ∨ a ∣ b) ∧ (c ∣ b ∨ b ∣ c) ∧ (d ∣ c ∨ c ∣ d) ∧ (e ∣ d ∨ d ∣ e) ∧
  (f ∣ e ∨ e ∣ f) ∧ (g ∣ f ∨ f ∣ g) ∧ (a ∣ g ∨ g ∣ a) →
  ¬((a ∣ c ∨ c ∣ a) ∧ (a ∣ d ∨ d ∣ a) ∧ (a ∣ e ∨ e ∣ a) ∧ (a ∣ f ∨ f ∣ a) ∧
    (a ∣ g ∨ g ∣ a) ∧ (b ∣ d ∨ d ∣ b) ∧ (b ∣ e ∨ e ∣ b) ∧ (b ∣ f ∨ f ∣ b) ∧
    (b ∣ g ∨ g ∣ b) ∧ (c ∣ e ∨ e ∣ c) ∧ (c ∣ f ∨ f ∣ c) ∧ (c ∣ g ∨ g ∣ c) ∧
    (d ∣ f ∨ f ∣ d) ∧ (d ∣ g ∨ g ∣ d) ∧ (e ∣ g ∨ g ∣ e)) :=
 by
  sorry

end heptagon_divisibility_impossible_l2170_217000


namespace easter_egg_problem_l2170_217007

-- Define the conditions as assumptions
def total_eggs : Nat := 63
def helen_eggs (H : Nat) := H
def hannah_eggs (H : Nat) := 2 * H
def harry_eggs (H : Nat) := 2 * H + 3

-- The theorem stating the proof problem
theorem easter_egg_problem (H : Nat) (hh : hannah_eggs H + helen_eggs H + harry_eggs H = total_eggs) : 
    helen_eggs H = 12 ∧ hannah_eggs H = 24 ∧ harry_eggs H = 27 :=
sorry -- Proof is omitted

end easter_egg_problem_l2170_217007


namespace multiply_of_Mari_buttons_l2170_217001

-- Define the variables and constants from the problem
def Mari_buttons : ℕ := 8
def Sue_buttons : ℕ := 22
def Kendra_buttons : ℕ := 2 * Sue_buttons

-- Statement that we need to prove
theorem multiply_of_Mari_buttons : ∃ (x : ℕ), Kendra_buttons = 8 * x + 4 ∧ x = 5 := by
  sorry

end multiply_of_Mari_buttons_l2170_217001


namespace area_of_pentagon_correct_l2170_217082

noncomputable def area_of_pentagon : ℝ :=
  let AB := 5
  let BC := 3
  let BD := 3
  let AC := Real.sqrt (AB^2 - BC^2)
  let AD := Real.sqrt (AB^2 - BD^2)
  let EC := 1
  let FD := 2
  let AE := AC - EC
  let AF := AD - FD
  let sin_alpha := BC / AB
  let cos_alpha := AC / AB
  let sin_2alpha := 2 * sin_alpha * cos_alpha
  let area_ABC := 0.5 * AB * BC
  let area_AEF := 0.5 * AE * AF * sin_2alpha
  2 * area_ABC - area_AEF

theorem area_of_pentagon_correct :
  area_of_pentagon = 9.12 := sorry

end area_of_pentagon_correct_l2170_217082


namespace construct_80_construct_160_construct_20_l2170_217022

-- Define the notion of constructibility from an angle
inductive Constructible : ℝ → Prop
| base (a : ℝ) : a = 40 → Constructible a
| add (a b : ℝ) : Constructible a → Constructible b → Constructible (a + b)
| sub (a b : ℝ) : Constructible a → Constructible b → Constructible (a - b)

-- Lean statements for proving the constructibility
theorem construct_80 : Constructible 80 :=
sorry

theorem construct_160 : Constructible 160 :=
sorry

theorem construct_20 : Constructible 20 :=
sorry

end construct_80_construct_160_construct_20_l2170_217022


namespace crossed_out_number_is_29_l2170_217042

theorem crossed_out_number_is_29 : 
  ∀ n : ℕ, (11 * n + 66 - (325 - (12 * n + 66 - 325))) = 29 :=
by sorry

end crossed_out_number_is_29_l2170_217042


namespace maximum_value_l2170_217090

noncomputable def p : ℝ := 1 + 1/2 + 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5

theorem maximum_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h_constraint : (x - 1)^2 + (y - 1)^2 + (z - 1)^2 = 27) : 
  x^p + y^p + z^p ≤ 40.4 :=
sorry

end maximum_value_l2170_217090


namespace exists_y_with_7_coprimes_less_than_20_l2170_217051

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1
def connection (a b : ℕ) : ℚ := Nat.lcm a b / (a * b)

theorem exists_y_with_7_coprimes_less_than_20 :
  ∃ y : ℕ, y < 20 ∧ (∃ x : ℕ, connection y x = 1) ∧ (Nat.totient y = 7) :=
by
  sorry

end exists_y_with_7_coprimes_less_than_20_l2170_217051


namespace swans_in_10_years_l2170_217088

def doubling_time := 2
def initial_swans := 15
def periods := 10 / doubling_time

theorem swans_in_10_years : 
  (initial_swans * 2 ^ periods) = 480 := 
by
  sorry

end swans_in_10_years_l2170_217088


namespace quadratic_function_increases_l2170_217074

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + 5

-- Prove that for x > 1, the function value y increases as x increases
theorem quadratic_function_increases (x : ℝ) (h : x > 1) : 
  quadratic_function x > quadratic_function 1 :=
sorry

end quadratic_function_increases_l2170_217074


namespace charlie_paints_60_sqft_l2170_217029

theorem charlie_paints_60_sqft (A B C : ℕ) (total_sqft : ℕ) (h_ratio : A = 3 ∧ B = 5 ∧ C = 2) (h_total : total_sqft = 300) : 
  C * (total_sqft / (A + B + C)) = 60 :=
by
  rcases h_ratio with ⟨rfl, rfl, rfl⟩
  rcases h_total with rfl
  sorry

end charlie_paints_60_sqft_l2170_217029


namespace pentagon_vertex_assignment_l2170_217027

theorem pentagon_vertex_assignment :
  ∃ (x_A x_B x_C x_D x_E : ℝ),
    x_A + x_B = 1 ∧
    x_B + x_C = 2 ∧
    x_C + x_D = 3 ∧
    x_D + x_E = 4 ∧
    x_E + x_A = 5 ∧
    (x_A, x_B, x_C, x_D, x_E) = (1.5, -0.5, 2.5, 0.5, 3.5) := by
  sorry

end pentagon_vertex_assignment_l2170_217027


namespace rohan_monthly_salary_l2170_217097

theorem rohan_monthly_salary :
  ∃ S : ℝ, 
    (0.4 * S) + (0.2 * S) + (0.1 * S) + (0.1 * S) + 1000 = S :=
by
  sorry

end rohan_monthly_salary_l2170_217097


namespace floor_area_difference_l2170_217086

noncomputable def area_difference (r_outer : ℝ) (n : ℕ) (r_inner : ℝ) : ℝ :=
  let outer_area := Real.pi * r_outer^2
  let inner_area := n * Real.pi * r_inner^2
  outer_area - inner_area

theorem floor_area_difference :
  ∀ (r_outer : ℝ) (n : ℕ) (r_inner : ℝ), 
  n = 8 ∧ r_outer = 40 ∧ r_inner = 40 / (2*Real.sqrt 2 + 1) →
  ⌊area_difference r_outer n r_inner⌋ = 1150 :=
by
  intros
  sorry

end floor_area_difference_l2170_217086


namespace square_side_length_l2170_217004

theorem square_side_length (x : ℝ) (h : x^2 = 12) : x = 2 * Real.sqrt 3 :=
sorry

end square_side_length_l2170_217004


namespace sum_exterior_angles_const_l2170_217033

theorem sum_exterior_angles_const (n : ℕ) (h : n ≥ 3) : 
  ∃ s : ℝ, s = 360 :=
by
  sorry

end sum_exterior_angles_const_l2170_217033


namespace triangle_sum_correct_l2170_217026

def triangle_op (a b c : ℕ) : ℕ :=
  a * b / c

theorem triangle_sum_correct :
  triangle_op 4 8 2 + triangle_op 5 10 5 = 26 :=
by
  sorry

end triangle_sum_correct_l2170_217026


namespace find_initial_number_l2170_217045

theorem find_initial_number (x : ℤ) (h : (x + 2)^2 = x^2 - 2016) : x = -505 :=
by {
  sorry
}

end find_initial_number_l2170_217045


namespace ryan_total_commuting_time_l2170_217081

def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def bus_commutes : ℕ := 3
def total_bus_time : ℕ := bus_time * bus_commutes
def friend_time : ℕ := biking_time - (2 * biking_time / 3)
def total_commuting_time : ℕ := biking_time + total_bus_time + friend_time

theorem ryan_total_commuting_time :
  total_commuting_time = 160 :=
by
  sorry

end ryan_total_commuting_time_l2170_217081


namespace gray_region_area_l2170_217070

theorem gray_region_area (r : ℝ) (h1 : r > 0) (h2 : 3 * r - r = 3) : 
  (π * (3 * r) * (3 * r) - π * r * r) = 18 * π := by
  sorry

end gray_region_area_l2170_217070


namespace percentage_proof_l2170_217053

/-- Lean 4 statement proving the percentage -/
theorem percentage_proof :
  ∃ P : ℝ, (800 - (P / 100) * 8000) = 796 ∧ P = 0.05 :=
by
  use 0.05
  sorry

end percentage_proof_l2170_217053


namespace gambler_target_win_percentage_l2170_217093

-- Define the initial conditions
def initial_games_played : ℕ := 20
def initial_win_rate : ℚ := 0.40

def additional_games_played : ℕ := 20
def additional_win_rate : ℚ := 0.80

-- Define the proof problem statement
theorem gambler_target_win_percentage 
  (initial_wins : ℚ := initial_win_rate * initial_games_played)
  (additional_wins : ℚ := additional_win_rate * additional_games_played)
  (total_games_played : ℕ := initial_games_played + additional_games_played)
  (total_wins : ℚ := initial_wins + additional_wins) :
  ((total_wins / total_games_played) * 100 : ℚ) = 60 := 
by
  -- Skipping the proof steps
  sorry

end gambler_target_win_percentage_l2170_217093


namespace sphere_radius_geometric_mean_l2170_217095

-- Definitions from conditions
variable (r R ρ : ℝ)
variable (r_nonneg : 0 ≤ r)
variable (R_relation : R = 3 * r)
variable (ρ_relation : ρ = Real.sqrt 3 * r)

-- Problem statement
theorem sphere_radius_geometric_mean (tetrahedron : Prop):
  ρ * ρ = R * r :=
by
  sorry

end sphere_radius_geometric_mean_l2170_217095


namespace sum_of_first_five_terms_l2170_217016

theorem sum_of_first_five_terms 
  (a₂ a₃ a₄ : ℤ)
  (h1 : a₂ = 4)
  (h2 : a₃ = 7)
  (h3 : a₄ = 10) :
  ∃ a1 a5, a1 + a₂ + a₃ + a₄ + a5 = 35 :=
by
  sorry

end sum_of_first_five_terms_l2170_217016


namespace range_of_a_if_proposition_l2170_217003

theorem range_of_a_if_proposition :
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → -4 < a ∧ a < 2 := by
  sorry

end range_of_a_if_proposition_l2170_217003


namespace lily_spent_on_shirt_l2170_217049

theorem lily_spent_on_shirt (S : ℝ) (initial_balance : ℝ) (final_balance : ℝ) : 
  initial_balance = 55 → 
  final_balance = 27 → 
  55 - S - 3 * S = 27 → 
  S = 7 := 
by
  intros h1 h2 h3
  sorry

end lily_spent_on_shirt_l2170_217049


namespace final_sum_l2170_217036

def Q (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 4

noncomputable def probability_condition_holds : ℝ :=
  by sorry

theorem final_sum :
  let m := 1
  let n := 1
  let o := 1
  let p := 0
  let q := 8
  (m + n + o + p + q) = 11 :=
  by
    sorry

end final_sum_l2170_217036


namespace minimum_value_l2170_217013

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y / x = 1) :
  ∃ (m : ℝ), m = 4 ∧ ∀ z, z = (1 / x + x / y) → z ≥ m :=
sorry

end minimum_value_l2170_217013


namespace area_of_pentagon_eq_fraction_l2170_217071

theorem area_of_pentagon_eq_fraction (w : ℝ) (h : ℝ) (fold_x : ℝ) (fold_y : ℝ)
    (hw3 : h = 3 * w)
    (hfold : fold_x = fold_y)
    (hx : fold_x ^ 2 + fold_y ^ 2 = 3 ^ 2)
    (hx_dist : fold_x = 4 / 3) :
  (3 * (1 / 2) + fold_x / 2) / (3 * w) = 13 / 18 := 
by 
  sorry

end area_of_pentagon_eq_fraction_l2170_217071


namespace alex_needs_additional_coins_l2170_217058

theorem alex_needs_additional_coins : 
  let friends := 12
  let coins := 63
  let total_coins_needed := (friends * (friends + 1)) / 2 
  let additional_coins_needed := total_coins_needed - coins
  additional_coins_needed = 15 :=
by sorry

end alex_needs_additional_coins_l2170_217058


namespace tan_alpha_minus_pi_six_l2170_217098

variable (α β : Real)

axiom tan_alpha_minus_beta : Real.tan (α - β) = 2 / 3
axiom tan_pi_six_minus_beta : Real.tan ((Real.pi / 6) - β) = 1 / 2

theorem tan_alpha_minus_pi_six : Real.tan (α - (Real.pi / 6)) = 1 / 8 :=
by
  sorry

end tan_alpha_minus_pi_six_l2170_217098


namespace expression_equals_5000_l2170_217092

theorem expression_equals_5000 :
  12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 :=
by
  sorry

end expression_equals_5000_l2170_217092


namespace time_at_simple_interest_l2170_217025

theorem time_at_simple_interest 
  (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * (R + 5) / 100) * T = (P * (R / 100) * T) + 150) : 
  T = 10 := 
by 
  -- Proof is omitted.
  sorry

end time_at_simple_interest_l2170_217025
