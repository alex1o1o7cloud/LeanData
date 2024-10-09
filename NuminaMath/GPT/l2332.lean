import Mathlib

namespace sequence_general_term_l2332_233280

theorem sequence_general_term (a : ℕ → ℕ) (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, 3 * (Finset.range (n + 1)).sum a = (n + 2) * a n) :
  ∀ n : ℕ, a n = n :=
by
  sorry

end sequence_general_term_l2332_233280


namespace work_problem_l2332_233218

theorem work_problem (P Q R W t_q : ℝ) (h1 : P = Q + R) 
    (h2 : (P + Q) * 10 = W) 
    (h3 : R * 35 = W) 
    (h4 : Q * t_q = W) : 
    t_q = 28 := 
by
    sorry

end work_problem_l2332_233218


namespace perimeter_of_large_rectangle_l2332_233272

-- We are bringing in all necessary mathematical libraries, no specific submodules needed.
theorem perimeter_of_large_rectangle
  (small_rectangle_longest_side : ℝ)
  (number_of_small_rectangles : ℕ)
  (length_of_large_rectangle : ℝ)
  (height_of_large_rectangle : ℝ)
  (perimeter_of_large_rectangle : ℝ) :
  small_rectangle_longest_side = 10 ∧ number_of_small_rectangles = 9 →
  length_of_large_rectangle = 2 * small_rectangle_longest_side →
  height_of_large_rectangle = 5 * (small_rectangle_longest_side / 2) →
  perimeter_of_large_rectangle = 2 * (length_of_large_rectangle + height_of_large_rectangle) →
  perimeter_of_large_rectangle = 76 := by
  sorry

end perimeter_of_large_rectangle_l2332_233272


namespace infinite_zeros_in_S_l2332_233228

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n % 4 = 0 then -↑(n + 1) else
  if n % 4 = 1 then ↑n else
  if n % 4 = 2 then ↑n else
  -↑(n + 1)

-- Define the sequence S_k as partial sum of a_n
def S : ℕ → ℤ
| 0       => a 0
| (n + 1) => S n + a (n + 1)

-- Proposition: S_k contains infinitely many zeros
theorem infinite_zeros_in_S : ∀ n : ℕ, ∃ m > n, S m = 0 := sorry

end infinite_zeros_in_S_l2332_233228


namespace sin_cos_identity_l2332_233204

theorem sin_cos_identity (x : ℝ) (h : Real.sin x = 4 * Real.cos x) : 
  Real.sin x * Real.cos x = 4 / 17 := by
  sorry

end sin_cos_identity_l2332_233204


namespace profit_share_of_B_l2332_233246

theorem profit_share_of_B (P : ℝ) (A_share B_share C_share : ℝ) :
  let A_initial := 8000
  let B_initial := 10000
  let C_initial := 12000
  let total_capital := A_initial + B_initial + C_initial
  let investment_ratio_A := A_initial / total_capital
  let investment_ratio_B := B_initial / total_capital
  let investment_ratio_C := C_initial / total_capital
  let total_profit := 4200
  let diff_AC := 560
  A_share = (investment_ratio_A * total_profit) →
  B_share = (investment_ratio_B * total_profit) →
  C_share = (investment_ratio_C * total_profit) →
  C_share - A_share = diff_AC →
  B_share = 1400 :=
by
  intros
  sorry

end profit_share_of_B_l2332_233246


namespace percentage_increase_l2332_233263

-- defining the given values
def Z := 150
def total := 555
def x_from_y (Y : ℝ) := 1.25 * Y

-- defining the condition that x gets 25% more than y and z out of 555 is Rs. 150
def condition1 (X Y : ℝ) := X = x_from_y Y
def condition2 (X Y : ℝ) := X + Y + Z = total

-- theorem to prove
theorem percentage_increase (Y : ℝ) :
  condition1 (x_from_y Y) Y →
  condition2 (x_from_y Y) Y →
  ((Y - Z) / Z) * 100 = 20 :=
by
  sorry

end percentage_increase_l2332_233263


namespace simplify_expression_l2332_233223

theorem simplify_expression :
  15 * (18 / 5) * (-42 / 45) = -50.4 :=
by
  sorry

end simplify_expression_l2332_233223


namespace find_P_l2332_233270

theorem find_P (P Q R S : ℕ) (h1: P ≠ Q) (h2: R ≠ S) (h3: P * Q = 72) (h4: R * S = 72) (h5: P - Q = R + S) :
  P = 18 := 
  sorry

end find_P_l2332_233270


namespace negation_of_at_most_three_l2332_233251

theorem negation_of_at_most_three (x : ℕ) : ¬ (x ≤ 3) ↔ x > 3 :=
by sorry

end negation_of_at_most_three_l2332_233251


namespace distribution_of_balls_l2332_233299

theorem distribution_of_balls :
  ∃ (P : ℕ → ℕ → ℕ), P 6 4 = 9 := 
by
  sorry

end distribution_of_balls_l2332_233299


namespace remainder_sum_first_six_primes_div_seventh_prime_l2332_233224

-- Define the first six prime numbers
def firstSixPrimes : List ℕ := [2, 3, 5, 7, 11, 13]

-- Define the sum of the first six prime numbers
def sumOfFirstSixPrimes : ℕ := firstSixPrimes.sum

-- Define the seventh prime number
def seventhPrime : ℕ := 17

-- Proof statement that the remainder of the division is 7
theorem remainder_sum_first_six_primes_div_seventh_prime :
  (sumOfFirstSixPrimes % seventhPrime) = 7 :=
by
  sorry

end remainder_sum_first_six_primes_div_seventh_prime_l2332_233224


namespace inverse_proportion_value_scientific_notation_l2332_233200

-- Statement to prove for Question 1:
theorem inverse_proportion_value (m : ℤ) (x : ℝ) :
  (m - 2) * x ^ (m ^ 2 - 5) = 0 ↔ m = -2 := by
  sorry

-- Statement to prove for Question 2:
theorem scientific_notation : -0.00000032 = -3.2 * 10 ^ (-7) := by
  sorry

end inverse_proportion_value_scientific_notation_l2332_233200


namespace find_value_of_n_l2332_233278

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem find_value_of_n
  (a b c n : ℕ)
  (ha : is_prime a)
  (hb : is_prime b)
  (hc : is_prime c)
  (h1 : 2 * a + 3 * b = c)
  (h2 : 4 * a + c + 1 = 4 * b)
  (h3 : n = a * b * c)
  (h4 : n < 10000) :
  n = 1118 :=
by
  sorry

end find_value_of_n_l2332_233278


namespace weight_of_substance_l2332_233238

variable (k W1 W2 : ℝ)

theorem weight_of_substance (h1 : ∃ (k : ℝ), ∀ (V W : ℝ), V = k * W)
  (h2 : 48 = k * W1) (h3 : 36 = k * 84) : 
  (∃ (W2 : ℝ), 48 = (36 / 84) * W2) → W2 = 112 := 
by
  sorry

end weight_of_substance_l2332_233238


namespace div_by_3kp1_iff_div_by_3k_l2332_233230

theorem div_by_3kp1_iff_div_by_3k (m n k : ℕ) (h1 : m > n) :
  (3 ^ (k + 1)) ∣ (4 ^ m - 4 ^ n) ↔ (3 ^ k) ∣ (m - n) := 
sorry

end div_by_3kp1_iff_div_by_3k_l2332_233230


namespace part1_part2_l2332_233245

-- Part (1)
theorem part1 (a : ℕ → ℕ) (d : ℕ) (S_3 T_3 : ℕ) (h₁ : 3 * a 2 = 3 * a 1 + a 3) (h₂ : S_3 + T_3 = 21) :
  (∀ n, a n = 3 * n) :=
sorry

-- Part (2)
theorem part2 (b : ℕ → ℕ) (d : ℕ) (S_99 T_99 : ℕ) (h₁ : ∀ m n : ℕ, b (m + n) - b m = d * n)
  (h₂ : S_99 - T_99 = 99) : d = 51 / 50 :=
sorry

end part1_part2_l2332_233245


namespace percentage_decrease_increase_l2332_233216

theorem percentage_decrease_increase (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 ↔ x = 50 :=
by
  sorry

end percentage_decrease_increase_l2332_233216


namespace owls_joined_l2332_233276

theorem owls_joined (initial_owls : ℕ) (total_owls : ℕ) (join_owls : ℕ) 
  (h_initial : initial_owls = 3) (h_total : total_owls = 5) : join_owls = 2 :=
by {
  -- Sorry is used to skip the proof
  sorry
}

end owls_joined_l2332_233276


namespace least_value_of_fourth_integer_l2332_233281

theorem least_value_of_fourth_integer :
  ∃ (A B C D : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
    A + B + C + D = 64 ∧ 
    A = 3 * B ∧ B = C - 2 ∧ 
    D = 52 := sorry

end least_value_of_fourth_integer_l2332_233281


namespace student_scores_l2332_233221

def weighted_average (math history science geography : ℝ) : ℝ :=
  (math * 0.30) + (history * 0.30) + (science * 0.20) + (geography * 0.20)

theorem student_scores :
  ∀ (math history science geography : ℝ),
    math = 74 →
    history = 81 →
    science = geography + 5 →
    science ≥ 75 →
    weighted_average math history science geography = 80 →
    science = 86.25 ∧ geography = 81.25 :=
by
  intros math history science geography h_math h_history h_science h_min_sci h_avg
  sorry

end student_scores_l2332_233221


namespace fish_lifespan_is_12_l2332_233242

def hamster_lifespan : ℝ := 2.5
def dog_lifespan : ℝ := 4 * hamster_lifespan
def fish_lifespan : ℝ := dog_lifespan + 2

theorem fish_lifespan_is_12 : fish_lifespan = 12 := by
  sorry

end fish_lifespan_is_12_l2332_233242


namespace parabola_focus_distance_l2332_233209

theorem parabola_focus_distance (M : ℝ × ℝ) (h1 : (M.2)^2 = 4 * M.1) (h2 : dist M (1, 0) = 4) : M.1 = 3 :=
sorry

end parabola_focus_distance_l2332_233209


namespace jeans_to_tshirt_ratio_l2332_233231

noncomputable def socks_price := 5
noncomputable def tshirt_price := socks_price + 10
noncomputable def jeans_price := 30

theorem jeans_to_tshirt_ratio :
  jeans_price / tshirt_price = (2 : ℝ) :=
by sorry

end jeans_to_tshirt_ratio_l2332_233231


namespace greatest_value_q_minus_r_l2332_233237

theorem greatest_value_q_minus_r {x y : ℕ} (hx : x < 10) (hy : y < 10) (hqr : 9 * (x - y) < 70) :
  9 * (x - y) = 63 :=
sorry

end greatest_value_q_minus_r_l2332_233237


namespace can_cabinet_be_moved_out_through_door_l2332_233260

/-
Definitions for the problem:
- Length, width, and height of the room
- Width, height, and depth of the cabinet
- Width and height of the door
-/

structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

def room : Dimensions := { length := 4, width := 2.5, height := 2.3 }
def cabinet : Dimensions := { length := 0.6, width := 1.8, height := 2.1 }
def door : Dimensions := { length := 0.8, height := 1.9, width := 0 }

theorem can_cabinet_be_moved_out_through_door : 
  (cabinet.length ≤ door.length ∧ cabinet.width ≤ door.height) ∨ 
  (cabinet.width ≤ door.length ∧ cabinet.length ≤ door.height) 
∧ 
cabinet.height ≤ room.height ∧ cabinet.width ≤ room.width ∧ 
cabinet.length ≤ room.length → True :=
by
  sorry

end can_cabinet_be_moved_out_through_door_l2332_233260


namespace solve_a_for_pure_imaginary_l2332_233244

theorem solve_a_for_pure_imaginary (a : ℝ) : (1 - a^2 = 0) ∧ (2 * a ≠ 0) → (a = 1 ∨ a = -1) :=
by
  sorry

end solve_a_for_pure_imaginary_l2332_233244


namespace find_triples_l2332_233287

theorem find_triples (a m n : ℕ) (h1 : a ≥ 2) (h2 : m ≥ 2) :
  a^n + 203 ∣ a^(m * n) + 1 → ∃ (k : ℕ), (k ≥ 1) := 
sorry

end find_triples_l2332_233287


namespace cuboid_volume_l2332_233279

/-- Given a cuboid with edges 6 cm, 5 cm, and 6 cm, the volume of the cuboid
    is 180 cm³. -/
theorem cuboid_volume (a b c : ℕ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 6) :
  a * b * c = 180 := by
  sorry

end cuboid_volume_l2332_233279


namespace change_is_4_25_l2332_233256

-- Define the conditions
def apple_cost : ℝ := 0.75
def amount_paid : ℝ := 5.00

-- State the theorem
theorem change_is_4_25 : amount_paid - apple_cost = 4.25 :=
by
  sorry

end change_is_4_25_l2332_233256


namespace problem_statement_l2332_233261

noncomputable def angle_between_vectors (a b : EuclideanSpace ℝ (Fin 3)) : ℝ :=
Real.arccos (inner a b / (‖a‖ * ‖b‖))

theorem problem_statement
  (a b : EuclideanSpace ℝ (Fin 3))
  (h_angle_ab : angle_between_vectors a b = Real.pi / 3)
  (h_norm_a : ‖a‖ = 2)
  (h_norm_b : ‖b‖ = 1) :
  angle_between_vectors a (a + 2 • b) = Real.pi / 6 :=
sorry

end problem_statement_l2332_233261


namespace min_rows_512_l2332_233225

theorem min_rows_512 (n : ℕ) (table : ℕ → ℕ → ℕ) 
  (H : ∀ A (i j : ℕ), i < 10 → j < 10 → i ≠ j → ∃ B, B < n ∧ (table B i ≠ table A i) ∧ (table B j ≠ table A j) ∧ ∀ k, k ≠ i ∧ k ≠ j → table B k = table A k) : 
  n ≥ 512 :=
sorry

end min_rows_512_l2332_233225


namespace find_a_l2332_233255

theorem find_a {a : ℝ} (h : {x : ℝ | (1/2 : ℝ) < x ∧ x < 2} = {x : ℝ | 0 < ax^2 + 5 * x - 2}) : a = -2 :=
sorry

end find_a_l2332_233255


namespace johns_percentage_increase_l2332_233217

def original_amount : ℕ := 60
def new_amount : ℕ := 84

def percentage_increase (original new : ℕ) := ((new - original : ℕ) * 100) / original 

theorem johns_percentage_increase : percentage_increase original_amount new_amount = 40 :=
by
  sorry

end johns_percentage_increase_l2332_233217


namespace paint_cost_decrease_l2332_233295

variables (C P : ℝ)
variable (cost_decrease_canvas : ℝ := 0.40)
variable (total_cost_decrease : ℝ := 0.56)
variable (paint_to_canvas_ratio : ℝ := 4)

theorem paint_cost_decrease (x : ℝ) : 
  P = 4 * C ∧ 
  P * (1 - x) + C * (1 - cost_decrease_canvas) = (1 - total_cost_decrease) * (P + C) → 
  x = 0.60 :=
by
  intro h
  sorry

end paint_cost_decrease_l2332_233295


namespace correct_option_is_B_l2332_233266

-- Definitions and conditions based on the problem
def is_monomial (t : String) : Prop :=
  t = "1"

def coefficient (expr : String) : Int :=
  if expr = "x" then 1
  else if expr = "-3x" then -3
  else 0

def degree (term : String) : Int :=
  if term = "5x^2y" then 3
  else 0

-- Proof statement
theorem correct_option_is_B : 
  is_monomial "1" ∧ ¬ (coefficient "x" = 0) ∧ ¬ (coefficient "-3x" = 3) ∧ ¬ (degree "5x^2y" = 2) := 
by
  -- Proof steps will go here
  sorry

end correct_option_is_B_l2332_233266


namespace cone_base_circumference_l2332_233268

theorem cone_base_circumference (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) :
  V = 18 * Real.pi →
  h = 6 →
  (V = (1 / 3) * Real.pi * r^2 * h) →
  C = 2 * Real.pi * r →
  C = 6 * Real.pi :=
by
  intros h1 h2 h3 h4
  sorry

end cone_base_circumference_l2332_233268


namespace tangent_condition_sum_f_l2332_233205

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

theorem tangent_condition (a : ℝ) (h : f a 1 = f a 1) (m : ℝ) : 
    (3 * a + 1 = (7 - (f a 1)) / 2) := 
    sorry

theorem sum_f (a : ℝ) (h : a = 3/7) : 
    f a (-4) + f a (-3) + f a (-2) + f a (-1) + f a 0 + 
    f a 1 + f a 2 + f a 3 + f a 4 = 9 := 
    sorry

end tangent_condition_sum_f_l2332_233205


namespace volunteer_selection_probability_l2332_233259

theorem volunteer_selection_probability :
  ∀ (students total_students remaining_students selected_volunteers : ℕ),
    total_students = 2018 →
    remaining_students = total_students - 18 →
    selected_volunteers = 50 →
    (selected_volunteers : ℚ) / total_students = (25 : ℚ) / 1009 :=
by
  intros students total_students remaining_students selected_volunteers
  intros h1 h2 h3
  sorry

end volunteer_selection_probability_l2332_233259


namespace simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l2332_233297

theorem simplify_175_sub_57_sub_43 : 175 - 57 - 43 = 75 :=
by
  sorry

theorem simplify_128_sub_64_sub_36 : 128 - 64 - 36 = 28 :=
by
  sorry

theorem simplify_156_sub_49_sub_51 : 156 - 49 - 51 = 56 :=
by
  sorry

end simplify_175_sub_57_sub_43_simplify_128_sub_64_sub_36_simplify_156_sub_49_sub_51_l2332_233297


namespace total_amount_shared_l2332_233284

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
axiom condition1 : a = (1 / 3 : ℝ) * (b + c)
axiom condition2 : b = (2 / 7 : ℝ) * (a + c)
axiom condition3 : a = b + 15

-- The proof statement
theorem total_amount_shared : a + b + c = 540 :=
by
  -- We assume these axioms are declared and noncontradictory
  sorry

end total_amount_shared_l2332_233284


namespace option_c_same_function_l2332_233277

theorem option_c_same_function :
  ∀ (x : ℝ), x ≠ 0 → (1 + (1 / x) = u ↔ u = 1 + (1 / (1 + 1 / x))) :=
by sorry

end option_c_same_function_l2332_233277


namespace solve_equation_l2332_233282

theorem solve_equation (x : ℝ) (h : x * (x - 3) = 10) : x = 5 ∨ x = -2 :=
by sorry

end solve_equation_l2332_233282


namespace tangent_lengths_identity_l2332_233293

theorem tangent_lengths_identity
  (a b c BC AC AB : ℝ)
  (sqrt_a sqrt_b sqrt_c : ℝ)
  (h1 : sqrt_a^2 = a)
  (h2 : sqrt_b^2 = b)
  (h3 : sqrt_c^2 = c) :
  a * BC + c * AB - b * AC = BC * AC * AB :=
sorry

end tangent_lengths_identity_l2332_233293


namespace gcd_of_powers_of_two_l2332_233267

def m : ℕ := 2^2100 - 1
def n : ℕ := 2^2000 - 1

theorem gcd_of_powers_of_two :
  Nat.gcd m n = 2^100 - 1 := sorry

end gcd_of_powers_of_two_l2332_233267


namespace min_frac_a_n_over_n_l2332_233212

open Nat

def a : ℕ → ℕ
| 0     => 60
| (n+1) => a n + 2 * n

theorem min_frac_a_n_over_n : ∃ n : ℕ, n > 0 ∧ (a n / n = (29 / 2) ∧ ∀ m : ℕ, m > 0 → a m / m ≥ (29 / 2)) :=
by
  sorry

end min_frac_a_n_over_n_l2332_233212


namespace functional_square_for_all_n_l2332_233275

theorem functional_square_for_all_n (f : ℕ → ℕ) :
  (∀ m n : ℕ, ∃ k : ℕ, (f m + n) * (m + f n) = k ^ 2) ↔ ∃ c : ℕ, ∀ n : ℕ, f n = n + c := 
sorry

end functional_square_for_all_n_l2332_233275


namespace total_points_l2332_233220

theorem total_points (Jon Jack Tom : ℕ) (h1 : Jon = 3) (h2 : Jack = Jon + 5) (h3 : Tom = Jon + Jack - 4) : Jon + Jack + Tom = 18 := by
  sorry

end total_points_l2332_233220


namespace burger_meal_cost_l2332_233264

theorem burger_meal_cost 
  (x : ℝ) 
  (h : 5 * (x + 1) = 35) : 
  x = 6 := 
sorry

end burger_meal_cost_l2332_233264


namespace evaluate_expression_l2332_233274

theorem evaluate_expression (b : ℕ) (h : b = 2) : b^3 * b^4 = 128 :=
by
  -- sorry is used to skip the proof
  sorry

end evaluate_expression_l2332_233274


namespace insufficient_info_for_pumpkins_l2332_233219

variable (jason_watermelons : ℕ) (sandy_watermelons : ℕ) (total_watermelons : ℕ)

theorem insufficient_info_for_pumpkins (h1 : jason_watermelons = 37)
  (h2 : sandy_watermelons = 11)
  (h3 : jason_watermelons + sandy_watermelons = total_watermelons)
  (h4 : total_watermelons = 48) : 
  ¬∃ (jason_pumpkins : ℕ), true
:= by
  sorry

end insufficient_info_for_pumpkins_l2332_233219


namespace Kira_was_away_for_8_hours_l2332_233234

theorem Kira_was_away_for_8_hours
  (kibble_rate: ℕ)
  (initial_kibble: ℕ)
  (remaining_kibble: ℕ)
  (hours_per_pound: ℕ) 
  (kibble_eaten: ℕ)
  (kira_was_away: ℕ)
  (h1: kibble_rate = 1)
  (h2: initial_kibble = 3)
  (h3: remaining_kibble = 1)
  (h4: hours_per_pound = 4)
  (h5: kibble_eaten = initial_kibble - remaining_kibble)
  (h6: kira_was_away = hours_per_pound * kibble_eaten) : 
  kira_was_away = 8 :=
by
  sorry

end Kira_was_away_for_8_hours_l2332_233234


namespace rectangle_area_decrease_l2332_233289

noncomputable def rectangle_area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) : ℝ :=
  let L' := 1.10 * L
  let B' := 0.90 * B
  let A  := L * B
  let A' := L' * B'
  A'

theorem rectangle_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  rectangle_area_change L B hL hB = 0.99 * (L * B) := by
  sorry

end rectangle_area_decrease_l2332_233289


namespace max_difference_in_flour_masses_l2332_233292

/--
Given three brands of flour with the following mass ranges:
1. Brand A: (48 ± 0.1) kg
2. Brand B: (48 ± 0.2) kg
3. Brand C: (48 ± 0.3) kg

Prove that the maximum difference in mass between any two bags of these different brands is 0.5 kg.
-/
theorem max_difference_in_flour_masses :
  (∀ (a b : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.8 ≤ b ∧ b ≤ 48.2)) →
    |a - b| ≤ 0.5) ∧
  (∀ (a c : ℝ), ((47.9 ≤ a ∧ a ≤ 48.1) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |a - c| ≤ 0.5) ∧
  (∀ (b c : ℝ), ((47.8 ≤ b ∧ b ≤ 48.2) ∧ (47.7 ≤ c ∧ c ≤ 48.3)) →
    |b - c| ≤ 0.5) := 
sorry

end max_difference_in_flour_masses_l2332_233292


namespace shop_owner_percentage_profit_l2332_233215

theorem shop_owner_percentage_profit :
  let cost_price_per_kg := 100
  let buy_cheat_percent := 18.5 / 100
  let sell_cheat_percent := 22.3 / 100
  let amount_bought := 1 / (1 + buy_cheat_percent)
  let amount_sold := 1 - sell_cheat_percent
  let effective_cost_price := cost_price_per_kg * amount_sold / amount_bought
  let selling_price := cost_price_per_kg
  let profit := selling_price - effective_cost_price
  let percentage_profit := (profit / effective_cost_price) * 100
  percentage_profit = 52.52 :=
by
  sorry

end shop_owner_percentage_profit_l2332_233215


namespace division_by_reciprocal_l2332_233294

theorem division_by_reciprocal :
  (10 / 3) / (1 / 5) = 50 / 3 := 
sorry

end division_by_reciprocal_l2332_233294


namespace unique_ordered_pair_satisfies_equation_l2332_233240

theorem unique_ordered_pair_satisfies_equation :
  ∃! (m n : ℕ), 0 < m ∧ 0 < n ∧ (6 / m + 3 / n + 1 / (m * n) = 1) :=
by
  sorry

end unique_ordered_pair_satisfies_equation_l2332_233240


namespace interest_received_l2332_233213

theorem interest_received
  (total_investment : ℝ)
  (part_invested_6 : ℝ)
  (rate_6 : ℝ)
  (rate_9 : ℝ) :
  part_invested_6 = 7200 →
  rate_6 = 0.06 →
  rate_9 = 0.09 →
  total_investment = 10000 →
  (total_investment - part_invested_6) * rate_9 + part_invested_6 * rate_6 = 684 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end interest_received_l2332_233213


namespace total_yellow_leaves_l2332_233286

noncomputable def calculate_yellow_leaves (total : ℕ) (percent_brown : ℕ) (percent_green : ℕ) : ℕ :=
  let brown_leaves := (total * percent_brown + 50) / 100
  let green_leaves := (total * percent_green + 50) / 100
  total - (brown_leaves + green_leaves)

theorem total_yellow_leaves :
  let t_yellow := calculate_yellow_leaves 15 25 40
  let f_yellow := calculate_yellow_leaves 22 30 20
  let s_yellow := calculate_yellow_leaves 30 15 50
  t_yellow + f_yellow + s_yellow = 26 :=
by
  sorry

end total_yellow_leaves_l2332_233286


namespace fewest_tiles_needed_l2332_233239

theorem fewest_tiles_needed 
  (tile_len : ℝ) (tile_wid : ℝ) (region_len : ℝ) (region_wid : ℝ)
  (h_tile_dims : tile_len = 2 ∧ tile_wid = 3)
  (h_region_dims : region_len = 48 ∧ region_wid = 72) :
  (region_len * region_wid) / (tile_len * tile_wid) = 576 :=
by {
  sorry
}

end fewest_tiles_needed_l2332_233239


namespace village_population_l2332_233253

-- Defining the variables and the condition
variable (P : ℝ) (h : 0.9 * P = 36000)

-- Statement of the theorem to prove
theorem village_population : P = 40000 :=
by sorry

end village_population_l2332_233253


namespace Calvin_mistake_correct_l2332_233291

theorem Calvin_mistake_correct (a : ℕ) : 37 + 31 * a = 37 * 31 + a → a = 37 :=
sorry

end Calvin_mistake_correct_l2332_233291


namespace exists_3x3_grid_l2332_233285

theorem exists_3x3_grid : 
  ∃ (a₁₂ a₂₁ a₂₃ a₃₂ : ℕ), 
  a₁₂ ≠ a₂₁ ∧ a₁₂ ≠ a₂₃ ∧ a₁₂ ≠ a₃₂ ∧ 
  a₂₁ ≠ a₂₃ ∧ a₂₁ ≠ a₃₂ ∧ 
  a₂₃ ≠ a₃₂ ∧ 
  a₁₂ ≤ 25 ∧ a₂₁ ≤ 25 ∧ a₂₃ ≤ 25 ∧ a₃₂ ≤ 25 ∧ 
  a₁₂ > 0 ∧ a₂₁ > 0 ∧ a₂₃ > 0 ∧ a₃₂ > 0 ∧
  (∃ (a₁₁ a₁₃ a₃₁ a₃₃ a₂₂ : ℕ),
  a₁₁ ≤ 25 ∧ a₁₃ ≤ 25 ∧ a₃₁ ≤ 25 ∧ a₃₃ ≤ 25 ∧ a₂₂ ≤ 25 ∧
  a₁₁ > 0 ∧ a₁₃ > 0 ∧ a₃₁ > 0 ∧ a₃₃ > 0 ∧ a₂₂ > 0 ∧
  a₁₁ ≠ a₁₂ ∧ a₁₁ ≠ a₂₁ ∧ a₁₁ ≠ a₁₃ ∧ a₁₁ ≠ a₃₁ ∧ 
  a₁₃ ≠ a₃₃ ∧ a₁₃ ≠ a₂₃ ∧ a₂₁ ≠ a₃₁ ∧ a₃₁ ≠ a₃₂ ∧ 
  a₃₃ ≠ a₂₂ ∧ a₃₃ ≠ a₃₂ ∧ a₂₂ = 1 ∧
  (a₁₂ % a₂₂ = 0 ∨ a₂₂ % a₁₂ = 0) ∧
  (a₂₁ % a₂₂ = 0 ∨ a₂₂ % a₂₁ = 0) ∧
  (a₂₃ % a₂₂ = 0 ∨ a₂₂ % a₂₃ = 0) ∧
  (a₃₂ % a₂₂ = 0 ∨ a₂₂ % a₃₂ = 0) ∧
  (a₁₁ % a₁₂ = 0 ∨ a₁₂ % a₁₁ = 0) ∧
  (a₁₁ % a₂₁ = 0 ∨ a₂₁ % a₁₁ = 0) ∧
  (a₁₃ % a₁₂ = 0 ∨ a₁₂ % a₁₃ = 0) ∧
  (a₁₃ % a₂₃ = 0 ∨ a₂₃ % a₁₃ = 0) ∧
  (a₃₁ % a₂₁ = 0 ∨ a₂₁ % a₃₁ = 0) ∧
  (a₃₁ % a₃₂ = 0 ∨ a₃₂ % a₃₁ = 0) ∧
  (a₃₃ % a₂₃ = 0 ∨ a₂₃ % a₃₃ = 0) ∧
  (a₃₃ % a₃₂ = 0 ∨ a₃₂ % a₃₃ = 0)) 
  :=
sorry

end exists_3x3_grid_l2332_233285


namespace baseball_card_decrease_l2332_233210

theorem baseball_card_decrease (x : ℝ) :
  (0 < x) ∧ (x < 100) ∧ (100 - x) * 0.9 = 45 → x = 50 :=
by
  intros h
  sorry

end baseball_card_decrease_l2332_233210


namespace geometric_sequence_value_l2332_233250

variable {α : Type*} [LinearOrderedField α] (a : ℕ → α)
variable (r : α)
variable (a_pos : ∀ n, a n > 0)
variable (h1 : a 1 = 2)
variable (h99 : a 99 = 8)
variable (geom_seq : ∀ n, a (n + 1) = r * a n)

theorem geometric_sequence_value :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end geometric_sequence_value_l2332_233250


namespace b_95_mod_49_l2332_233262

-- Define the sequence b_n
def b (n : ℕ) : ℕ := 7^n + 9^n

-- Goal: Prove that the remainder when b 95 is divided by 49 is 28
theorem b_95_mod_49 : b 95 % 49 = 28 := 
by
  sorry

end b_95_mod_49_l2332_233262


namespace find_e_l2332_233235

-- Define values for a, b, c, d
def a := 2
def b := 3
def c := 4
def d := 5

-- State the problem
theorem find_e (e : ℚ) : a + b + c + d + e = a + (b + (c - (d * e))) → e = -5/6 :=
by
  sorry

end find_e_l2332_233235


namespace sum_of_cubes_of_consecutive_integers_l2332_233226

theorem sum_of_cubes_of_consecutive_integers :
  ∃ (a b c d : ℕ), a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a^2 + b^2 + c^2 + d^2 = 9340) ∧ (a^3 + b^3 + c^3 + d^3 = 457064) :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_l2332_233226


namespace find_x_l2332_233203

theorem find_x (x : ℕ) (h : x * 5^4 = 75625) : x = 121 :=
by
  sorry

end find_x_l2332_233203


namespace fraction_diff_equals_7_over_12_l2332_233241

noncomputable def fraction_diff : ℚ :=
  (2 + 4 + 6) / (1 + 3 + 5) - (1 + 3 + 5) / (2 + 4 + 6)

theorem fraction_diff_equals_7_over_12 : fraction_diff = 7 / 12 := by
  sorry

end fraction_diff_equals_7_over_12_l2332_233241


namespace factorize_poly1_min_value_poly2_l2332_233298

-- Define the polynomials
def poly1 := fun (x : ℝ) => x^2 + 2 * x - 3
def factored_poly1 := fun (x : ℝ) => (x - 1) * (x + 3)

def poly2 := fun (x : ℝ) => x^2 + 4 * x + 5
def min_value := 1

-- State the theorems without providing proofs
theorem factorize_poly1 : ∀ x : ℝ, poly1 x = factored_poly1 x := 
by { sorry }

theorem min_value_poly2 : ∀ x : ℝ, poly2 x ≥ min_value := 
by { sorry }

end factorize_poly1_min_value_poly2_l2332_233298


namespace probability_standard_bulb_l2332_233232

structure FactoryConditions :=
  (P_H1 : ℝ)
  (P_H2 : ℝ)
  (P_H3 : ℝ)
  (P_A_H1 : ℝ)
  (P_A_H2 : ℝ)
  (P_A_H3 : ℝ)

theorem probability_standard_bulb (conditions : FactoryConditions) : 
  conditions.P_H1 = 0.45 → 
  conditions.P_H2 = 0.40 → 
  conditions.P_H3 = 0.15 →
  conditions.P_A_H1 = 0.70 → 
  conditions.P_A_H2 = 0.80 → 
  conditions.P_A_H3 = 0.81 → 
  (conditions.P_H1 * conditions.P_A_H1 + 
   conditions.P_H2 * conditions.P_A_H2 + 
   conditions.P_H3 * conditions.P_A_H3) = 0.7565 :=
by 
  intros h1 h2 h3 a_h1 a_h2 a_h3 
  sorry

end probability_standard_bulb_l2332_233232


namespace five_digit_number_l2332_233229

open Nat

noncomputable def problem_statement : Prop :=
  ∃ A B C D E F : ℕ,
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
    D ≠ E ∧ D ≠ F ∧
    E ≠ F ∧
    A + B + C + D + E + F = 25 ∧
    (A, B, C, D, E, F) = (3, 4, 2, 1, 6, 9)

theorem five_digit_number : problem_statement := 
  sorry

end five_digit_number_l2332_233229


namespace find_J_l2332_233290

variables (J S B : ℕ)

-- Conditions
def condition1 : Prop := J - 20 = 2 * S
def condition2 : Prop := B = J / 2
def condition3 : Prop := J + S + B = 330
def condition4 : Prop := (J - 20) + S + B = 318

-- Theorem to prove
theorem find_J (h1 : condition1 J S) (h2 : condition2 J B) (h3 : condition3 J S B) (h4 : condition4 J S B) :
  J = 170 :=
sorry

end find_J_l2332_233290


namespace area_of_triangle_is_24_l2332_233258

open Real

-- Define the coordinates of the vertices
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (10, 6)

-- Define the vectors from point C
def v : ℝ × ℝ := (A.1 - C.1, A.2 - C.2)
def w : ℝ × ℝ := (B.1 - C.1, B.2 - C.2)

-- Define the determinant for the parallelogram area
def parallelogram_area : ℝ :=
  abs (v.1 * w.2 - v.2 * w.1)

-- Prove the area of the triangle
theorem area_of_triangle_is_24 : (parallelogram_area / 2) = 24 := by
  sorry

end area_of_triangle_is_24_l2332_233258


namespace sum_abc_is_eight_l2332_233236

theorem sum_abc_is_eight (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h : (a + b + c)^3 - a^3 - b^3 - c^3 = 294) : a + b + c = 8 :=
by
  sorry

end sum_abc_is_eight_l2332_233236


namespace jordan_weight_after_exercise_l2332_233247

theorem jordan_weight_after_exercise :
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  current_weight = 222 :=
by
  let initial_weight := 250
  let weight_loss_4_weeks := 3 * 4
  let weight_loss_8_weeks := 2 * 8
  let total_weight_loss := weight_loss_4_weeks + weight_loss_8_weeks
  let current_weight := initial_weight - total_weight_loss
  show current_weight = 222
  sorry

end jordan_weight_after_exercise_l2332_233247


namespace median_length_of_right_triangle_l2332_233283

noncomputable def length_of_median (a b c : ℕ) : ℝ := 
  if a * a + b * b = c * c then c / 2 else 0

theorem median_length_of_right_triangle :
  length_of_median 9 12 15 = 7.5 :=
by
  -- Insert the proof here
  sorry

end median_length_of_right_triangle_l2332_233283


namespace least_four_digit_with_factors_3_5_7_l2332_233208

open Nat

-- Definitions for the conditions
def has_factors (n : ℕ) (factors : List ℕ) : Prop :=
  ∀ f ∈ factors, f ∣ n

-- Main theorem statement
theorem least_four_digit_with_factors_3_5_7
  (n : ℕ) 
  (h1 : 1000 ≤ n) 
  (h2 : n < 10000)
  (h3 : has_factors n [3, 5, 7]) :
  n = 1050 :=
sorry

end least_four_digit_with_factors_3_5_7_l2332_233208


namespace coat_price_reduction_l2332_233252

theorem coat_price_reduction (original_price reduction_amount : ℝ) (h : original_price = 500) (h_red : reduction_amount = 150) :
  ((reduction_amount / original_price) * 100) = 30 :=
by
  rw [h, h_red]
  norm_num

end coat_price_reduction_l2332_233252


namespace square_root_condition_l2332_233202

-- Define the condition under which the square root of an expression is defined
def is_square_root_defined (x : ℝ) : Prop := (x + 3) ≥ 0

-- Prove that the condition for the square root of x + 3 to be defined is x ≥ -3
theorem square_root_condition (x : ℝ) : is_square_root_defined x ↔ x ≥ -3 := 
sorry

end square_root_condition_l2332_233202


namespace magnitude_of_z_l2332_233227

open Complex

theorem magnitude_of_z {z : ℂ} (h : z * (1 + I) = 1 - I) : abs z = 1 :=
sorry

end magnitude_of_z_l2332_233227


namespace minimum_s_value_l2332_233233

theorem minimum_s_value (x y z : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z) (h_cond : 3 * x^2 + 2 * y^2 + z^2 = 1) :
  ∃ (s : ℝ), s = 8 * Real.sqrt 6 ∧ ∀ (x' y' z' : ℝ), (0 < x' ∧ 0 < y' ∧ 0 < z' ∧ 3 * x'^2 + 2 * y'^2 + z'^2 = 1) → 
      s ≤ (1 + z') / (x' * y' * z') :=
sorry

end minimum_s_value_l2332_233233


namespace side_of_rhombus_l2332_233296

variable (d : ℝ) (K : ℝ) 

-- Conditions
def shorter_diagonal := d
def longer_diagonal := 3 * d
def area_rhombus := K = (1 / 2) * d * (3 * d)

-- Proof Statement
theorem side_of_rhombus (h1 : K = (3 / 2) * d^2) : (∃ s : ℝ, s = Real.sqrt (5 * K / 3)) := 
  sorry

end side_of_rhombus_l2332_233296


namespace candidate_percentage_l2332_233206

theorem candidate_percentage (P : ℝ) (h : (P / 100) * 7800 + 2340 = 7800) : P = 70 :=
sorry

end candidate_percentage_l2332_233206


namespace solve_floor_equation_l2332_233248

theorem solve_floor_equation (x : ℝ) (h : ⌊x * ⌊x⌋⌋ = 20) : 5 ≤ x ∧ x < 5.25 := by
  sorry

end solve_floor_equation_l2332_233248


namespace john_gives_to_stud_owner_l2332_233269

variable (initial_puppies : ℕ) (puppies_given_away : ℕ) (puppies_kept : ℕ) (price_per_puppy : ℕ) (profit : ℕ)

theorem john_gives_to_stud_owner
  (h1 : initial_puppies = 8)
  (h2 : puppies_given_away = initial_puppies / 2)
  (h3 : puppies_kept = 1)
  (h4 : price_per_puppy = 600)
  (h5 : profit = 1500) :
  let puppies_left_to_sell := initial_puppies - puppies_given_away - puppies_kept
  let total_sales := puppies_left_to_sell * price_per_puppy
  total_sales - profit = 300 :=
by
  intro puppies_left_to_sell
  intro total_sales
  sorry

end john_gives_to_stud_owner_l2332_233269


namespace linear_combination_of_matrices_l2332_233201

variable (A B : Matrix (Fin 3) (Fin 3) ℤ) 

def matrixA : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![2, -4, 0],
    ![-1, 5, 1],
    ![0, 3, -7]
  ]

def matrixB : Matrix (Fin 3) (Fin 3) ℤ := 
  ![
    ![4, -1, -2],
    ![0, -3, 5],
    ![2, 0, -4]
  ]

theorem linear_combination_of_matrices :
  3 • matrixA - 2 • matrixB = 
  ![
    ![-2, -10, 4],
    ![-3, 21, -7],
    ![-4, 9, -13]
  ] :=
sorry

end linear_combination_of_matrices_l2332_233201


namespace smallest_n_for_divisibility_l2332_233271

theorem smallest_n_for_divisibility (a₁ a₂ : ℕ) (n : ℕ) (h₁ : a₁ = 5 / 8) (h₂ : a₂ = 25) :
  (∃ n : ℕ, n ≥ 1 ∧ (a₁ * (40 ^ (n - 1)) % 2000000 = 0)) → (n = 7) :=
by
  sorry

end smallest_n_for_divisibility_l2332_233271


namespace find_principal_l2332_233257

variable (SI : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)

theorem find_principal (h1 : SI = 4020.75) (h2 : R = 0.0875) (h3 : T = 5.5) (h4 : SI = P * R * T) : 
  P = 8355.00 :=
sorry

end find_principal_l2332_233257


namespace percentage_seeds_germinated_l2332_233243

/-- There were 300 seeds planted in the first plot and 200 seeds planted in the second plot. 
    30% of the seeds in the first plot germinated and 32% of the total seeds germinated.
    Prove that 35% of the seeds in the second plot germinated. -/
theorem percentage_seeds_germinated 
  (s1 s2 : ℕ) (p1 p2 t : ℚ)
  (h1 : s1 = 300) 
  (h2 : s2 = 200) 
  (h3 : p1 = 30) 
  (h4 : t = 32) 
  (h5 : 0.30 * s1 + p2 * s2 = 0.32 * (s1 + s2)) :
  p2 = 35 :=
by 
  -- Proof goes here
  sorry

end percentage_seeds_germinated_l2332_233243


namespace correct_statements_l2332_233254

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - 13 / 4 * Real.pi)

theorem correct_statements :
    (f (Real.pi / 8) = 0) ∧ 
    (∀ x, 2 * Real.sin (2 * (x - 5 / 8 * Real.pi)) = f x) :=
by
  sorry

end correct_statements_l2332_233254


namespace count_coin_distributions_l2332_233249

-- Mathematical conditions
def coin_denominations : Finset ℕ := {1, 2, 3, 5}
def number_of_boys : ℕ := 6

-- Theorem statement
theorem count_coin_distributions : (coin_denominations.card ^ number_of_boys) = 4096 :=
by
  sorry

end count_coin_distributions_l2332_233249


namespace marge_funds_for_fun_l2332_233211

-- Definitions based on given conditions
def lottery_amount : ℕ := 12006
def taxes_paid : ℕ := lottery_amount / 2
def remaining_after_taxes : ℕ := lottery_amount - taxes_paid
def student_loans_paid : ℕ := remaining_after_taxes / 3
def remaining_after_loans : ℕ := remaining_after_taxes - student_loans_paid
def savings : ℕ := 1000
def remaining_after_savings : ℕ := remaining_after_loans - savings
def stock_market_investment : ℕ := savings / 5
def remaining_after_investment : ℕ := remaining_after_savings - stock_market_investment

-- The proof goal
theorem marge_funds_for_fun : remaining_after_investment = 2802 :=
sorry

end marge_funds_for_fun_l2332_233211


namespace expression_evaluation_l2332_233288

theorem expression_evaluation :
  (8 / 4 - 3^2 + 4 * 5) = 13 :=
by sorry

end expression_evaluation_l2332_233288


namespace ed_marbles_l2332_233273

theorem ed_marbles (doug_initial_marbles : ℕ) (marbles_lost : ℕ) (ed_doug_difference : ℕ) 
  (h1 : doug_initial_marbles = 22) (h2 : marbles_lost = 3) (h3 : ed_doug_difference = 5) : 
  (doug_initial_marbles + ed_doug_difference) = 27 :=
by
  sorry

end ed_marbles_l2332_233273


namespace negation_P_l2332_233214

-- Define the proposition P
def P (m : ℤ) : Prop := ∃ x : ℤ, 2 * x^2 + x + m ≤ 0

-- Define the negation of the proposition P
theorem negation_P (m : ℤ) : ¬P m ↔ ∀ x : ℤ, 2 * x^2 + x + m > 0 :=
by
  sorry

end negation_P_l2332_233214


namespace solve_remainder_l2332_233222

theorem solve_remainder (y : ℤ) 
  (hc1 : y + 4 ≡ 9 [ZMOD 3^3])
  (hc2 : y + 4 ≡ 16 [ZMOD 5^3])
  (hc3 : y + 4 ≡ 36 [ZMOD 7^3]) : 
  y ≡ 32 [ZMOD 105] :=
by
  sorry

end solve_remainder_l2332_233222


namespace gcd_linear_combination_l2332_233207

theorem gcd_linear_combination (a b : ℤ) (h : Int.gcd a b = 1) : 
    Int.gcd (11 * a + 2 * b) (18 * a + 5 * b) = 1 := 
by
  sorry

end gcd_linear_combination_l2332_233207


namespace earnings_correct_l2332_233265

def price_8inch : ℝ := 5
def price_12inch : ℝ := 2.5 * price_8inch
def price_16inch : ℝ := 3 * price_8inch
def price_20inch : ℝ := 4 * price_8inch
def price_24inch : ℝ := 5.5 * price_8inch

noncomputable def earnings_monday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 1 * price_16inch + 2 * price_20inch + 1 * price_24inch

noncomputable def earnings_tuesday : ℝ :=
  5 * price_8inch + 1 * price_12inch + 4 * price_16inch + 2 * price_24inch

noncomputable def earnings_wednesday : ℝ :=
  4 * price_8inch + 3 * price_12inch + 3 * price_16inch + 1 * price_20inch

noncomputable def earnings_thursday : ℝ :=
  2 * price_8inch + 2 * price_12inch + 2 * price_16inch + 1 * price_20inch + 3 * price_24inch

noncomputable def earnings_friday : ℝ :=
  6 * price_8inch + 4 * price_12inch + 2 * price_16inch + 2 * price_20inch

noncomputable def earnings_saturday : ℝ :=
  1 * price_8inch + 3 * price_12inch + 3 * price_16inch + 4 * price_20inch + 2 * price_24inch

noncomputable def earnings_sunday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 4 * price_16inch + 3 * price_20inch + 1 * price_24inch

noncomputable def total_earnings : ℝ :=
  earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday + earnings_saturday + earnings_sunday

theorem earnings_correct : total_earnings = 1025 := by
  -- proof goes here
  sorry

end earnings_correct_l2332_233265
