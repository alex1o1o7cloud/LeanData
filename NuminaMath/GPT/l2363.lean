import Mathlib

namespace number_of_bananas_in_bowl_l2363_236366

theorem number_of_bananas_in_bowl (A P B : Nat) (h1 : P = A + 2) (h2 : B = P + 3) (h3 : A + P + B = 19) : B = 9 :=
sorry

end number_of_bananas_in_bowl_l2363_236366


namespace range_of_a_l2363_236398

theorem range_of_a (a : ℝ) (h : ¬ ∃ x : ℝ, x^2 + (a + 1) * x + 1 ≤ 0) : -3 < a ∧ a < 1 :=
sorry

end range_of_a_l2363_236398


namespace five_coins_no_105_cents_l2363_236329

theorem five_coins_no_105_cents :
  ¬ ∃ (a b c d e : ℕ), a + b + c + d + e = 5 ∧
    (a * 1 + b * 5 + c * 10 + d * 25 + e * 50 = 105) :=
sorry

end five_coins_no_105_cents_l2363_236329


namespace find_A_plus_B_l2363_236350

/-- Let A, B, C, and D be distinct digits such that 0 ≤ A, B, C, D ≤ 9.
    C and D are non-zero, and A ≠ B ≠ C ≠ D.
    If (A+B)/(C+D) is an integer and C+D is minimized,
    then prove that A + B = 15. -/
theorem find_A_plus_B
  (A B C D : ℕ)
  (h_digits : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_range : 0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9)
  (h_nonzero_CD : C ≠ 0 ∧ D ≠ 0)
  (h_integer : (A + B) % (C + D) = 0)
  (h_min_CD : ∀ C' D', (C' ≠ C ∨ D' ≠ D) → (C' ≠ 0 ∧ D' ≠ 0 → (C + D ≤ C' + D'))) :
  A + B = 15 := 
sorry

end find_A_plus_B_l2363_236350


namespace probability_heart_spade_queen_l2363_236351

theorem probability_heart_spade_queen (h_cards : ℕ) (s_cards : ℕ) (q_cards : ℕ) (total_cards : ℕ) 
    (h_not_q : ℕ) (remaining_cards_after_2 : ℕ) (remaining_spades : ℕ) 
    (queen_remaining_after_2 : ℕ) (remaining_cards_after_1 : ℕ) :
    h_cards = 13 ∧ s_cards = 13 ∧ q_cards = 4 ∧ total_cards = 52 ∧ h_not_q = 12 ∧ remaining_cards_after_2 = 50 ∧
    remaining_spades = 13 ∧ queen_remaining_after_2 = 3 ∧ remaining_cards_after_1 = 51 →
    (h_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (q_cards / remaining_cards_after_2) + 
    (q_cards / total_cards) * (remaining_spades / remaining_cards_after_1) * (queen_remaining_after_2 / remaining_cards_after_2) = 
    221 / 44200 := by 
  sorry

end probability_heart_spade_queen_l2363_236351


namespace smarties_division_l2363_236382

theorem smarties_division (m : ℕ) (h : m % 7 = 5) : (4 * m) % 7 = 6 := by
  sorry

end smarties_division_l2363_236382


namespace parabola_equation_l2363_236360

theorem parabola_equation (p : ℝ) (hp : 0 < p)
  (F : ℝ × ℝ) (hF : F = (p / 2, 0))
  (A B : ℝ × ℝ)
  (hA : A = (x1, y1)) (hB : B = (x2, y2))
  (h_intersect : y1^2 = 2*p*x1 ∧ y2^2 = 2*p*x2)
  (M : ℝ × ℝ) (hM : M = ((x1 + x2) / 2, (y1 + y2) / 2))
  (hM_coords : M = (3, 2)) :
  p = 2 ∨ p = 4 :=
sorry

end parabola_equation_l2363_236360


namespace smallest_divisor_of_2880_that_results_in_perfect_square_l2363_236365

theorem smallest_divisor_of_2880_that_results_in_perfect_square : 
  ∃ (n : ℕ), (n ∣ 2880) ∧ (∃ m : ℕ, 2880 / n = m * m) ∧ (∀ k : ℕ, (k ∣ 2880) ∧ (∃ m' : ℕ, 2880 / k = m' * m') → n ≤ k) ∧ n = 10 :=
sorry

end smallest_divisor_of_2880_that_results_in_perfect_square_l2363_236365


namespace josh_initial_wallet_l2363_236359

noncomputable def initial_wallet_amount (investment final_wallet: ℕ) (stock_increase_percentage: ℕ): ℕ :=
  let investment_value_after_rise := investment + (investment * stock_increase_percentage / 100)
  final_wallet - investment_value_after_rise

theorem josh_initial_wallet : initial_wallet_amount 2000 2900 30 = 300 :=
by
  sorry

end josh_initial_wallet_l2363_236359


namespace ratio_bones_child_to_adult_woman_l2363_236325

noncomputable def num_skeletons : ℕ := 20
noncomputable def num_adult_women : ℕ := num_skeletons / 2
noncomputable def num_adult_men_and_children : ℕ := num_skeletons - num_adult_women
noncomputable def num_adult_men : ℕ := num_adult_men_and_children / 2
noncomputable def num_children : ℕ := num_adult_men_and_children / 2
noncomputable def bones_per_adult_woman : ℕ := 20
noncomputable def bones_per_adult_man : ℕ := bones_per_adult_woman + 5
noncomputable def total_bones : ℕ := 375
noncomputable def bones_per_child : ℕ := (total_bones - (num_adult_women * bones_per_adult_woman + num_adult_men * bones_per_adult_man)) / num_children

theorem ratio_bones_child_to_adult_woman : 
  (bones_per_child : ℚ) / (bones_per_adult_woman : ℚ) = 1 / 2 := by
sorry

end ratio_bones_child_to_adult_woman_l2363_236325


namespace rearrange_letters_no_adjacent_repeats_l2363_236372

-- Factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Problem conditions
def distinct_permutations (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  factorial (String.length word) / (factorial freq_I * factorial freq_L)

-- No-adjacent-repeated permutations
def no_adjacent_repeats (word : String) (freq_I : ℕ) (freq_L : ℕ) : ℕ :=
  let total_permutations := distinct_permutations word freq_I freq_L
  let i_superletter_permutations := distinct_permutations (String.dropRight word 1) (freq_I - 1) freq_L
  let l_superletter_permutations := distinct_permutations (String.dropRight word 1) freq_I (freq_L - 1)
  let both_superletter_permutations := factorial (String.length word - 2)
  total_permutations - (i_superletter_permutations + l_superletter_permutations - both_superletter_permutations)

-- Given problem definition
def word := "BRILLIANT"
def freq_I := 2
def freq_L := 2

-- Proof problem statement
theorem rearrange_letters_no_adjacent_repeats :
  no_adjacent_repeats word freq_I freq_L = 55440 := by
  sorry

end rearrange_letters_no_adjacent_repeats_l2363_236372


namespace solve_system_l2363_236323

theorem solve_system :
  ∃ (x y z : ℝ), x + y + z = 9 ∧ (1/x + 1/y + 1/z = 1) ∧ (x * y + x * z + y * z = 27) ∧ x = 3 ∧ y = 3 ∧ z = 3 := by
sorry

end solve_system_l2363_236323


namespace number_of_bags_proof_l2363_236381

def total_flight_time_hours : ℕ := 2
def minutes_per_hour : ℕ := 60
def total_minutes := total_flight_time_hours * minutes_per_hour

def peanuts_per_minute : ℕ := 1
def total_peanuts_eaten := total_minutes * peanuts_per_minute

def peanuts_per_bag : ℕ := 30
def number_of_bags : ℕ := total_peanuts_eaten / peanuts_per_bag

theorem number_of_bags_proof : number_of_bags = 4 := by
  -- proof goes here
  sorry

end number_of_bags_proof_l2363_236381


namespace ethanol_in_fuel_A_l2363_236341

def fuel_tank_volume : ℝ := 208
def fuel_A_volume : ℝ := 82
def fuel_B_volume : ℝ := fuel_tank_volume - fuel_A_volume
def ethanol_in_fuel_B : ℝ := 0.16
def total_ethanol : ℝ := 30

theorem ethanol_in_fuel_A 
  (x : ℝ) 
  (H_fuel_tank_capacity : fuel_tank_volume = 208) 
  (H_fuel_A_volume : fuel_A_volume = 82) 
  (H_fuel_B_volume : fuel_B_volume = 126) 
  (H_ethanol_in_fuel_B : ethanol_in_fuel_B = 0.16) 
  (H_total_ethanol : total_ethanol = 30) 
  : 82 * x + 0.16 * 126 = 30 → x = 0.12 := by
  sorry

end ethanol_in_fuel_A_l2363_236341


namespace find_number_l2363_236364

theorem find_number (x : ℝ) (h : x - (3/5) * x = 56) : x = 140 :=
sorry

end find_number_l2363_236364


namespace neg_p_implies_neg_q_sufficient_but_not_necessary_l2363_236328

variables (x : ℝ) (p : Prop) (q : Prop)

def p_condition := (1 < x ∨ x < -3)
def q_condition := (5 * x - 6 > x ^ 2)

theorem neg_p_implies_neg_q_sufficient_but_not_necessary :
  p_condition x → q_condition x → ((¬ p_condition x) → (¬ q_condition x)) :=
by 
  intro h1 h2
  sorry

end neg_p_implies_neg_q_sufficient_but_not_necessary_l2363_236328


namespace range_of_a_l2363_236375

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3 * x + 1) ≤ a) → a ≥ 1/5 :=
by
  intro h
  sorry

end range_of_a_l2363_236375


namespace find_m_for_local_minimum_l2363_236319

noncomputable def f (x m : ℝ) := x * (x - m) ^ 2

theorem find_m_for_local_minimum :
  ∃ m : ℝ, (∀ x : ℝ, (x = 1 → deriv (λ x => f x m) x = 0) ∧ 
                  (x = 1 → deriv (deriv (λ x => f x m)) x > 0)) ∧ 
            m = 1 :=
by
  sorry

end find_m_for_local_minimum_l2363_236319


namespace smaller_root_of_quadratic_l2363_236322

theorem smaller_root_of_quadratic :
  ∃ (x₁ x₂ : ℝ), (x₁ ≠ x₂) ∧ (x₁^2 - 14 * x₁ + 45 = 0) ∧ (x₂^2 - 14 * x₂ + 45 = 0) ∧ (min x₁ x₂ = 5) :=
sorry

end smaller_root_of_quadratic_l2363_236322


namespace simple_interest_rate_l2363_236333

theorem simple_interest_rate (P : ℝ) (T : ℝ) (R : ℝ) (SI : ℝ) (hT : T = 8) 
  (hSI : SI = P / 5) : SI = (P * R * T) / 100 → R = 2.5 :=
by
  intro
  sorry

end simple_interest_rate_l2363_236333


namespace transform_identity_l2363_236305

theorem transform_identity (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) = (a * c + b * d)^2 + (a * d - b * c)^2 := 
sorry

end transform_identity_l2363_236305


namespace calculate_a_mul_a_sub_3_l2363_236303

variable (a : ℝ)

theorem calculate_a_mul_a_sub_3 : a * (a - 3) = a^2 - 3 * a := 
by
  sorry

end calculate_a_mul_a_sub_3_l2363_236303


namespace polynomial_multiplication_l2363_236391

theorem polynomial_multiplication (x y : ℝ) : 
  (2 * x - 3 * y + 1) * (2 * x + 3 * y - 1) = 4 * x^2 - 9 * y^2 + 6 * y - 1 := by
  sorry

end polynomial_multiplication_l2363_236391


namespace problem_l2363_236376

variable {m n r t : ℚ}

theorem problem (h1 : m / n = 5 / 4) (h2 : r / t = 8 / 15) : (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 2 :=
by
  sorry

end problem_l2363_236376


namespace problem_statement_l2363_236368

theorem problem_statement (a b c : ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) : 
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 :=
by
  sorry

end problem_statement_l2363_236368


namespace problem_statement_l2363_236352

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 2013 a b α β = 5) :
  f 2014 a b α β = 3 :=
by
  sorry

end problem_statement_l2363_236352


namespace prove_x_plus_y_leq_zero_l2363_236317

-- Definitions of the conditions
def valid_powers (a b : ℝ) (x y : ℝ) : Prop :=
  1 < a ∧ a < b ∧ a^x + b^y ≤ a^(-x) + b^(-y)

-- The theorem statement
theorem prove_x_plus_y_leq_zero (a b x y : ℝ) (h : valid_powers a b x y) : 
  x + y ≤ 0 :=
by
  sorry

end prove_x_plus_y_leq_zero_l2363_236317


namespace fraction_flower_beds_l2363_236355

theorem fraction_flower_beds (length1 length2 height triangle_area yard_area : ℝ) (h1 : length1 = 18) (h2 : length2 = 30) (h3 : height = 10) (h4 : triangle_area = 2 * (1 / 2 * (6 ^ 2))) (h5 : yard_area = ((length1 + length2) / 2) * height) : 
  (triangle_area / yard_area) = 3 / 20 :=
by 
  sorry

end fraction_flower_beds_l2363_236355


namespace negation_of_universal_quadratic_l2363_236374

theorem negation_of_universal_quadratic (P : ∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  ¬(∀ a b c : ℝ, a ≠ 0 → ∃ x : ℝ, a * x^2 + b * x + c = 0) ↔ ∃ a b c : ℝ, a ≠ 0 ∧ ¬(∃ x : ℝ, a * x^2 + b * x + c = 0) :=
by
  sorry

end negation_of_universal_quadratic_l2363_236374


namespace devin_biked_more_l2363_236337

def cyra_distance := 77
def cyra_time := 7
def cyra_speed := cyra_distance / cyra_time
def devin_speed := cyra_speed + 3
def marathon_time := 7
def devin_distance := devin_speed * marathon_time
def distance_difference := devin_distance - cyra_distance

theorem devin_biked_more : distance_difference = 21 := 
  by
    sorry

end devin_biked_more_l2363_236337


namespace units_digit_of_7_pow_2500_l2363_236357

theorem units_digit_of_7_pow_2500 : (7^2500) % 10 = 1 :=
by
  -- Variables and constants can be used to formalize steps if necessary, 
  -- but focus is on the statement itself.
  -- Sorry is used to skip the proof part.
  sorry

end units_digit_of_7_pow_2500_l2363_236357


namespace eighty_five_squared_l2363_236397

theorem eighty_five_squared : 85^2 = 7225 := by
  sorry

end eighty_five_squared_l2363_236397


namespace solution_in_Quadrant_III_l2363_236393

theorem solution_in_Quadrant_III {c x y : ℝ} 
    (h1 : x - y = 4) 
    (h2 : c * x + y = 5) 
    (hx : x < 0) 
    (hy : y < 0) : 
    c < -1 := 
sorry

end solution_in_Quadrant_III_l2363_236393


namespace trajectory_midpoints_l2363_236399

variables (a b c x y : ℝ)

def arithmetic_sequence (a b c : ℝ) : Prop := c = 2 * b - a

def line_eq (b a c x y : ℝ) : Prop := b * x + a * y + c = 0

def parabola_eq (x y : ℝ) : Prop := y^2 = -0.5 * x

theorem trajectory_midpoints
  (hac : arithmetic_sequence a b c)
  (line_cond : line_eq b a c x y)
  (parabola_cond : parabola_eq x y) :
  (x + 1 = -(2 * y - 1)^2) ∧ (y ≠ 1) :=
sorry

end trajectory_midpoints_l2363_236399


namespace fourth_number_is_57_l2363_236373

noncomputable def digit_sum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

def sum_list (l : List ℕ) : ℕ :=
  l.foldr (.+.) 0

theorem fourth_number_is_57 : 
  ∃ (N : ℕ), N < 100 ∧ 177 + N = 4 * (33 + digit_sum N) ∧ N = 57 :=
by {
  sorry
}

end fourth_number_is_57_l2363_236373


namespace tan_sum_l2363_236379

theorem tan_sum (x y : ℝ) (h1 : Real.sin x + Real.sin y = 85 / 65) (h2 : Real.cos x + Real.cos y = 60 / 65) :
  Real.tan x + Real.tan y = 17 / 12 :=
sorry

end tan_sum_l2363_236379


namespace smallest_multiple_of_seven_l2363_236392

/-- The definition of the six-digit number formed by digits a, b, and c followed by "321". -/
def form_number (a b c : ℕ) : ℕ := 100000 * a + 10000 * b + 1000 * c + 321

/-- The condition that a, b, and c are distinct and greater than 3. -/
def valid_digits (a b c : ℕ) : Prop := a > 3 ∧ b > 3 ∧ c > 3 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem smallest_multiple_of_seven (a b c : ℕ)
  (h_valid : valid_digits a b c)
  (h_mult_seven : form_number a b c % 7 = 0) :
  form_number a b c = 468321 :=
sorry

end smallest_multiple_of_seven_l2363_236392


namespace find_x_l2363_236342

-- Defining the number x and the condition
variable (x : ℝ) 

-- The condition given in the problem
def condition := x / 3 = x - 3

-- The theorem to be proved
theorem find_x (h : condition x) : x = 4.5 := 
by 
  sorry

end find_x_l2363_236342


namespace number_of_boys_is_320_l2363_236345

-- Definition of the problem's conditions
variable (B G : ℕ)
axiom condition1 : B + G = 400
axiom condition2 : G = (B / 400) * 100

-- Stating the theorem to prove number of boys is 320
theorem number_of_boys_is_320 : B = 320 :=
by
  sorry

end number_of_boys_is_320_l2363_236345


namespace partition_exists_l2363_236310
open Set Real

theorem partition_exists (r : ℚ) (hr : r > 1) :
  ∃ (A B : ℕ → Prop), (∀ n, A n ∨ B n) ∧ (∀ n, ¬(A n ∧ B n)) ∧ 
  (∀ k l, A k → A l → (k : ℚ) / (l : ℚ) ≠ r) ∧ 
  (∀ k l, B k → B l → (k : ℚ) / (l : ℚ) ≠ r) :=
sorry

end partition_exists_l2363_236310


namespace ned_weekly_sales_l2363_236311

-- Define the conditions given in the problem
def normal_mouse_price : ℝ := 120
def normal_keyboard_price : ℝ := 80
def normal_scissor_price : ℝ := 30

def lt_hand_mouse_price := normal_mouse_price * 1.3
def lt_hand_keyboard_price := normal_keyboard_price * 1.2
def lt_hand_scissor_price := normal_scissor_price * 1.5

def lt_hand_mouse_daily_sales : ℝ := 25 * lt_hand_mouse_price
def lt_hand_keyboard_daily_sales : ℝ := 10 * lt_hand_keyboard_price
def lt_hand_scissor_daily_sales : ℝ := 15 * lt_hand_scissor_price

def total_daily_sales := lt_hand_mouse_daily_sales + lt_hand_keyboard_daily_sales + lt_hand_scissor_daily_sales
def days_open_per_week : ℝ := 4

def weekly_sales := total_daily_sales * days_open_per_week

-- The theorem to prove
theorem ned_weekly_sales : weekly_sales = 22140 := by
  -- The proof is omitted
  sorry

end ned_weekly_sales_l2363_236311


namespace restore_triangle_Nagel_point_l2363_236321

-- Define the variables and types involved
variables {Point : Type}

-- Assume a structure to capture the properties of a triangle
structure Triangle (Point : Type) :=
(A B C : Point)

-- Define the given conditions
variables (N B E : Point)

-- Statement of the main Lean theorem to reconstruct the triangle ABC
theorem restore_triangle_Nagel_point 
    (N B E : Point) :
    ∃ (ABC : Triangle Point), 
      (ABC).B = B ∧
      -- Additional properties of the triangle to be stated here
      sorry
    :=
sorry

end restore_triangle_Nagel_point_l2363_236321


namespace simplify_correct_l2363_236367

open Polynomial

noncomputable def simplify_expression (y : ℚ) : Polynomial ℚ :=
  (3 * (Polynomial.C y) + 2) * (2 * (Polynomial.C y)^12 + 3 * (Polynomial.C y)^11 - (Polynomial.C y)^9 - (Polynomial.C y)^8)

theorem simplify_correct (y : ℚ) : 
  simplify_expression y = 6 * (Polynomial.C y)^13 + 13 * (Polynomial.C y)^12 + 6 * (Polynomial.C y)^11 - 3 * (Polynomial.C y)^10 - 5 * (Polynomial.C y)^9 - 2 * (Polynomial.C y)^8 := 
by 
  simp [simplify_expression]
  sorry

end simplify_correct_l2363_236367


namespace part_a_part_b_l2363_236387

-- Part (a)
theorem part_a (a b : ℕ) (h : (3 * a + b) % 10 = (3 * b + a) % 10) : ¬(a % 10 = b % 10) :=
by sorry

-- Part (b)
theorem part_b (a b c : ℕ)
  (h1 : (2 * a + b) % 10 = (2 * b + c) % 10)
  (h2 : (2 * b + c) % 10 = (2 * c + a) % 10)
  (h3 : (2 * c + a) % 10 = (2 * a + b) % 10) :
  (a % 10 = b % 10) ∧ (b % 10 = c % 10) ∧ (c % 10 = a % 10) :=
by sorry

end part_a_part_b_l2363_236387


namespace shaded_area_l2363_236356

-- Defining the conditions
def small_square_side := 4
def large_square_side := 12
def half_large_square_side := large_square_side / 2

-- DG is calculated as (12 / 16) * small_square_side = 3
def DG := (large_square_side / (half_large_square_side + small_square_side)) * small_square_side

-- Calculating area of triangle DGF
def area_triangle_DGF := (DG * small_square_side) / 2

-- Area of the smaller square
def area_small_square := small_square_side * small_square_side

-- Area of the shaded region
def area_shaded_region := area_small_square - area_triangle_DGF

-- The theorem stating the question
theorem shaded_area : area_shaded_region = 10 := by
  sorry

end shaded_area_l2363_236356


namespace exist_prime_not_dividing_l2363_236386

theorem exist_prime_not_dividing (p : ℕ) (hp : Prime p) : 
  ∃ q : ℕ, Prime q ∧ ∀ n : ℕ, 0 < n → ¬ (q ∣ n^p - p) := 
sorry

end exist_prime_not_dividing_l2363_236386


namespace avg_height_eq_61_l2363_236385

-- Define the constants and conditions
def Brixton : ℕ := 64
def Zara : ℕ := 64
def Zora := Brixton - 8
def Itzayana := Zora + 4

-- Define the total height of the four people
def total_height := Brixton + Zara + Zora + Itzayana

-- Define the average height
def average_height := total_height / 4

-- Theorem stating that the average height is 61 inches
theorem avg_height_eq_61 : average_height = 61 := by
  sorry

end avg_height_eq_61_l2363_236385


namespace area_of_rectangle_is_correct_l2363_236340

-- Given Conditions
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- Question: Find the area of the rectangle
def area := length * width

-- The theorem to prove
theorem area_of_rectangle_is_correct : area = 588 :=
by
  -- Proof steps can go here.
  sorry

end area_of_rectangle_is_correct_l2363_236340


namespace standard_eq_of_largest_circle_l2363_236348

theorem standard_eq_of_largest_circle 
  (m : ℝ)
  (hm : 0 < m) :
  ∃ r : ℝ, 
  (∀ x y : ℝ, (x^2 + (y - 1)^2 = 8) ↔ 
      (x^2 + (y - 1)^2 = r)) :=
sorry

end standard_eq_of_largest_circle_l2363_236348


namespace TamekaBoxesRelation_l2363_236318

theorem TamekaBoxesRelation 
  (S : ℤ)
  (h1 : 40 + S + S / 2 = 145) :
  S - 40 = 30 :=
by
  sorry

end TamekaBoxesRelation_l2363_236318


namespace collective_earnings_l2363_236353

theorem collective_earnings:
  let lloyd_hours := 10.5
  let mary_hours := 12.0
  let tom_hours := 7.0
  let lloyd_normal_hours := 7.5
  let mary_normal_hours := 8.0
  let tom_normal_hours := 9.0
  let lloyd_rate := 4.5
  let mary_rate := 5.0
  let tom_rate := 6.0
  let lloyd_overtime_rate := 2.5 * lloyd_rate
  let mary_overtime_rate := 3.0 * mary_rate
  let tom_overtime_rate := 2.0 * tom_rate
  let lloyd_earnings := (lloyd_normal_hours * lloyd_rate) + ((lloyd_hours - lloyd_normal_hours) * lloyd_overtime_rate)
  let mary_earnings := (mary_normal_hours * mary_rate) + ((mary_hours - mary_normal_hours) * mary_overtime_rate)
  let tom_earnings := (tom_hours * tom_rate)
  let total_earnings := lloyd_earnings + mary_earnings + tom_earnings
  total_earnings = 209.50 := by
  sorry

end collective_earnings_l2363_236353


namespace S_range_l2363_236395

theorem S_range (x : ℝ) (y : ℝ) (S : ℝ) 
  (h1 : y = 2 * x - 1) 
  (h2 : 0 ≤ x) 
  (h3 : x ≤ 1 / 2) 
  (h4 : S = x * y) : 
  -1 / 8 ≤ S ∧ S ≤ 0 := 
sorry

end S_range_l2363_236395


namespace cylinder_height_l2363_236313

theorem cylinder_height (r₁ r₂ : ℝ) (S : ℝ) (hR : r₁ = 3) (hL : r₂ = 4) (hS : S = 100 * Real.pi) : 
  (∃ h : ℝ, h = 7 ∨ h = 1) :=
by 
  sorry

end cylinder_height_l2363_236313


namespace triangle_side_ineq_l2363_236331

theorem triangle_side_ineq (a b c : ℝ) 
  (h1 : a + b > c) 
  (h2 : b + c > a) 
  (h3 : c + a > b) :
  (a - b) / (a + b) + (b - c) / (b + c) + (c - a) / (a + c) < 1 / 16 :=
  sorry

end triangle_side_ineq_l2363_236331


namespace online_sale_discount_l2363_236327

theorem online_sale_discount (purchase_amount : ℕ) (discount_per_100 : ℕ) (total_paid : ℕ) : 
  purchase_amount = 250 → 
  discount_per_100 = 10 → 
  total_paid = purchase_amount - (purchase_amount / 100) * discount_per_100 → 
  total_paid = 230 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end online_sale_discount_l2363_236327


namespace tan_alpha_plus_pi_div_4_l2363_236314

theorem tan_alpha_plus_pi_div_4 (α : ℝ) (hcos : Real.cos α = 3 / 5) (h0 : 0 < α) (hpi : α < Real.pi) :
  Real.tan (α + Real.pi / 4) = -7 :=
by
  sorry

end tan_alpha_plus_pi_div_4_l2363_236314


namespace prob_all_meet_standard_prob_at_least_one_meets_standard_l2363_236304

def P_meeting_standard_A := 0.8
def P_meeting_standard_B := 0.6
def P_meeting_standard_C := 0.5

theorem prob_all_meet_standard :
  (P_meeting_standard_A * P_meeting_standard_B * P_meeting_standard_C) = 0.24 :=
by
  sorry

theorem prob_at_least_one_meets_standard :
  (1 - ((1 - P_meeting_standard_A) * (1 - P_meeting_standard_B) * (1 - P_meeting_standard_C))) = 0.96 :=
by
  sorry

end prob_all_meet_standard_prob_at_least_one_meets_standard_l2363_236304


namespace polynomial_real_roots_l2363_236302

theorem polynomial_real_roots :
  (∃ x : ℝ, x^4 - 3*x^3 - 2*x^2 + 6*x + 9 = 0) ↔ (x = 1 ∨ x = 3) := 
by
  sorry

end polynomial_real_roots_l2363_236302


namespace rolling_dice_probability_l2363_236309

-- Defining variables and conditions
def total_outcomes : Nat := 6^7

def favorable_outcomes : Nat :=
  Nat.choose 7 2 * 6 * (Nat.factorial 5) -- Calculation for exactly one pair of identical numbers

def probability : Rat :=
  favorable_outcomes / total_outcomes

-- The main theorem to prove the probability is 5/18
theorem rolling_dice_probability :
  probability = 5 / 18 := by
  sorry

end rolling_dice_probability_l2363_236309


namespace find_number_l2363_236377

theorem find_number (n : ℝ) (h : 3 / 5 * ((2 / 3 + 3 / 8) / n) - 1 / 16 = 0.24999999999999994) : n = 48 :=
  sorry

end find_number_l2363_236377


namespace min_length_M_inter_N_l2363_236389

def setM (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3 / 4}
def setN (n : ℝ) : Set ℝ := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
def setP : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem min_length_M_inter_N (m n : ℝ) 
  (hm : 0 ≤ m ∧ m + 3 / 4 ≤ 1) 
  (hn : 1 / 3 ≤ n ∧ n ≤ 1) : 
  let I := (setM m ∩ setN n)
  ∃ Iinf Isup : ℝ, I = {x | Iinf ≤ x ∧ x ≤ Isup} ∧ Isup - Iinf = 1 / 12 :=
  sorry

end min_length_M_inter_N_l2363_236389


namespace sin_70_given_sin_10_l2363_236343

theorem sin_70_given_sin_10 (k : ℝ) (h : Real.sin 10 = k) : Real.sin 70 = 1 - 2 * k^2 := 
by 
  sorry

end sin_70_given_sin_10_l2363_236343


namespace geometrical_shapes_OABC_l2363_236334

/-- Given distinct points A(x₁, y₁), B(x₂, y₂), and C(2x₁ - x₂, 2y₁ - y₂) on a coordinate plane
    and the origin O(0,0), determine the possible geometrical shapes that the figure OABC can form
    among these three possibilities: (1) parallelogram (2) straight line (3) rhombus.
    
    Prove that the figure OABC can form either a parallelogram or a straight line,
    but not a rhombus.
-/
theorem geometrical_shapes_OABC (x₁ y₁ x₂ y₂ : ℝ) (h_distinct : (x₁, y₁) ≠ (x₂, y₂) ∧ (x₁, y₁) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂) ∧ (x₂, y₂) ≠ (2 * x₁ - x₂, 2 * y₁ - y₂)) :
  (∃ t : ℝ, t ≠ 0 ∧ t ≠ 1 ∧ x₂ = t * x₁ ∧ y₂ = t * y₁) ∨
  (2 * x₁ = x₁ + x₂ ∧ 2 * y₁ = y₁ + y₂) :=
sorry

end geometrical_shapes_OABC_l2363_236334


namespace base_eight_to_base_ten_642_l2363_236332

theorem base_eight_to_base_ten_642 :
  let d0 := 2
  let d1 := 4
  let d2 := 6
  let base := 8
  d0 * base^0 + d1 * base^1 + d2 * base^2 = 418 := 
by
  sorry

end base_eight_to_base_ten_642_l2363_236332


namespace power_division_identity_l2363_236339

theorem power_division_identity : 
  ∀ (a b c : ℕ), a = 3 → b = 12 → c = 2 → (3 ^ 12 / (3 ^ 2) ^ 2 = 6561) :=
by
  intros a b c h1 h2 h3
  sorry

end power_division_identity_l2363_236339


namespace total_distance_of_bus_rides_l2363_236346

theorem total_distance_of_bus_rides :
  let vince_distance   := 5 / 8
  let zachary_distance := 1 / 2
  let alice_distance   := 17 / 20
  let rebecca_distance := 2 / 5
  let total_distance   := vince_distance + zachary_distance + alice_distance + rebecca_distance
  total_distance = 19/8 := by
  sorry

end total_distance_of_bus_rides_l2363_236346


namespace value_of_x_plus_y_l2363_236330

theorem value_of_x_plus_y (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : 1/x - 1/y = 2) : x + y = 4/3 :=
sorry

end value_of_x_plus_y_l2363_236330


namespace problem_statement_l2363_236380

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := λ x => f (x - 1)

theorem problem_statement :
  (∀ x : ℝ, f (-x) = f x) →  -- Condition: f is an even function.
  (∀ x : ℝ, g (-x) = -g x) → -- Condition: g is an odd function.
  (g 1 = 3) →                -- Condition: g passes through (1,3).
  (f 2012 + g 2013 = 6) :=   -- Statement to prove.
by
  sorry

end problem_statement_l2363_236380


namespace linear_correlation_l2363_236347

variable (r : ℝ) (r_critical : ℝ)

theorem linear_correlation (h1 : r = -0.9362) (h2 : r_critical = 0.8013) :
  |r| > r_critical :=
by
  sorry

end linear_correlation_l2363_236347


namespace cauchy_problem_solution_l2363_236383

noncomputable def solution (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = (x^2) / 2 + (x^3) / 6 + (x^4) / 12 + (x^5) / 20 + x + 1

theorem cauchy_problem_solution (y : ℝ → ℝ) (x : ℝ) 
  (h1: ∀ x, (deriv^[2] y) x = 1 + x + x^2 + x^3)
  (h2: y 0 = 1)
  (h3: deriv y 0 = 1) : 
  solution y x := 
by
  -- Proof Steps
  sorry

end cauchy_problem_solution_l2363_236383


namespace expand_product_l2363_236336

theorem expand_product (x : ℝ) : (x + 5) * (x - 16) = x^2 - 11 * x - 80 :=
by sorry

end expand_product_l2363_236336


namespace fractions_product_simplified_l2363_236349

theorem fractions_product_simplified : (2/3 : ℚ) * (4/7) * (9/11) = 24/77 := by
  sorry

end fractions_product_simplified_l2363_236349


namespace unique_solution_l2363_236371

-- Definitions of the problem
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ (m : ℕ), m ∣ n → m = 1 ∨ m = n

def satisfies_conditions (p q r : ℕ) : Prop :=
  is_prime p ∧ is_prime q ∧ is_prime r ∧ is_prime (4 * q - 1) ∧ (p + q) * (r - p) = p + r

theorem unique_solution (p q r : ℕ) (h : satisfies_conditions p q r) : (p, q, r) = (2, 3, 3) :=
  sorry

end unique_solution_l2363_236371


namespace number_of_non_symmetric_letters_is_3_l2363_236354

def letters_in_JUNIOR : List Char := ['J', 'U', 'N', 'I', 'O', 'R']

def axis_of_symmetry (c : Char) : Bool :=
  match c with
  | 'J' => false
  | 'U' => true
  | 'N' => false
  | 'I' => true
  | 'O' => true
  | 'R' => false
  | _   => false

def letters_with_no_symmetry : List Char :=
  letters_in_JUNIOR.filter (λ c => ¬axis_of_symmetry c)

theorem number_of_non_symmetric_letters_is_3 :
  letters_with_no_symmetry.length = 3 :=
by
  sorry

end number_of_non_symmetric_letters_is_3_l2363_236354


namespace num_of_chairs_per_row_l2363_236362

theorem num_of_chairs_per_row (total_chairs : ℕ) (num_rows : ℕ) (chairs_per_row : ℕ)
  (h1 : total_chairs = 432)
  (h2 : num_rows = 27) :
  total_chairs = num_rows * chairs_per_row ↔ chairs_per_row = 16 :=
by
  sorry

end num_of_chairs_per_row_l2363_236362


namespace domain_of_f_l2363_236388

def domain_condition1 (x : ℝ) : Prop := 1 - |x - 1| > 0
def domain_condition2 (x : ℝ) : Prop := x - 1 ≠ 0

theorem domain_of_f :
  (∀ x : ℝ, domain_condition1 x ∧ domain_condition2 x → 0 < x ∧ x < 2 ∧ x ≠ 1) ↔
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1 ∪ Set.Ioo 1 2)) :=
by
  sorry

end domain_of_f_l2363_236388


namespace line_bisects_circle_and_perpendicular_l2363_236396

   def line_bisects_circle_and_is_perpendicular (x y : ℝ) : Prop :=
     (∃ (b : ℝ), ((2 * x - y + b = 0) ∧ (x^2 + y^2 - 2 * x - 4 * y = 0))) ∧
     ∀ b, (2 * 1 - 2 + b = 0) → b = 0 → (2 * x - y = 0)

   theorem line_bisects_circle_and_perpendicular :
     line_bisects_circle_and_is_perpendicular 1 2 :=
   by
     sorry
   
end line_bisects_circle_and_perpendicular_l2363_236396


namespace product_modulo_10_l2363_236358

-- Define the numbers involved
def a := 2457
def b := 7623
def c := 91309

-- Define the modulo operation we're interested in
def modulo_10 (n : Nat) : Nat := n % 10

-- State the theorem we want to prove
theorem product_modulo_10 :
  modulo_10 (a * b * c) = 9 :=
sorry

end product_modulo_10_l2363_236358


namespace sin_five_pi_over_six_l2363_236344

theorem sin_five_pi_over_six : Real.sin (5 * Real.pi / 6) = 1 / 2 := 
  sorry

end sin_five_pi_over_six_l2363_236344


namespace min_points_on_dodecahedron_min_points_on_icosahedron_l2363_236324

-- Definitions for the dodecahedron problem
def dodecahedron_has_12_faces : Prop := true
def each_vertex_in_dodecahedron_belongs_to_3_faces : Prop := true

-- Proof statement for dodecahedron
theorem min_points_on_dodecahedron : dodecahedron_has_12_faces ∧ each_vertex_in_dodecahedron_belongs_to_3_faces → ∃ n, n = 4 :=
by
  sorry

-- Definitions for the icosahedron problem
def icosahedron_has_20_faces : Prop := true
def icosahedron_has_12_vertices : Prop := true
def each_vertex_in_icosahedron_belongs_to_5_faces : Prop := true
def vertices_of_icosahedron_grouped_into_6_pairs : Prop := true

-- Proof statement for icosahedron
theorem min_points_on_icosahedron : 
  icosahedron_has_20_faces ∧ icosahedron_has_12_vertices ∧ each_vertex_in_icosahedron_belongs_to_5_faces ∧ vertices_of_icosahedron_grouped_into_6_pairs → ∃ n, n = 6 :=
by
  sorry

end min_points_on_dodecahedron_min_points_on_icosahedron_l2363_236324


namespace inequality_always_true_l2363_236307

theorem inequality_always_true 
  (a b : ℝ) 
  (h1 : ab > 0) : 
  (b / a) + (a / b) ≥ 2 := 
by sorry

end inequality_always_true_l2363_236307


namespace joes_current_weight_l2363_236394

theorem joes_current_weight (W : ℕ) (R : ℕ) : 
  (W = 222 - 4 * R) →
  (W - 3 * R = 180) →
  W = 198 :=
by
  intros h1 h2
  -- Skip the proof for now
  sorry

end joes_current_weight_l2363_236394


namespace part_a_part_b_l2363_236308

noncomputable def triangle_exists (h1 h2 h3 : ℕ) : Prop :=
  ∃ a b c, 2 * a = h1 * (b + c) ∧ 2 * b = h2 * (a + c) ∧ 2 * c = h3 * (a + b)

theorem part_a : ¬ triangle_exists 2 3 6 :=
sorry

theorem part_b : triangle_exists 2 3 5 :=
sorry

end part_a_part_b_l2363_236308


namespace amoeba_population_at_11am_l2363_236338

/-- Sarah observes an amoeba colony where initially there are 50 amoebas at 10:00 a.m. The population triples every 10 minutes and there are no deaths among the amoebas. Prove that the number of amoebas at 11:00 a.m. is 36450. -/
theorem amoeba_population_at_11am : 
  let initial_population := 50
  let growth_rate := 3
  let increments := 6  -- since 60 minutes / 10 minutes per increment = 6
  initial_population * (growth_rate ^ increments) = 36450 :=
by
  sorry

end amoeba_population_at_11am_l2363_236338


namespace calculation_correct_l2363_236335

theorem calculation_correct : (5 * 7 + 9 * 4 - 36 / 3 : ℤ) = 59 := by
  sorry

end calculation_correct_l2363_236335


namespace percentage_of_alcohol_in_new_mixture_l2363_236306

def original_solution_volume : ℕ := 11
def added_water_volume : ℕ := 3
def alcohol_percentage_original : ℝ := 0.42

def total_volume : ℕ := original_solution_volume + added_water_volume
def amount_of_alcohol : ℝ := alcohol_percentage_original * original_solution_volume

theorem percentage_of_alcohol_in_new_mixture :
  (amount_of_alcohol / total_volume) * 100 = 33 := by
  sorry

end percentage_of_alcohol_in_new_mixture_l2363_236306


namespace ball_height_less_than_10_after_16_bounces_l2363_236384

noncomputable def bounce_height (initial : ℝ) (ratio : ℝ) (bounces : ℕ) : ℝ :=
  initial * ratio^bounces

theorem ball_height_less_than_10_after_16_bounces :
  let initial_height := 800
  let bounce_ratio := 3 / 4
  ∃ k : ℕ, k = 16 ∧ bounce_height initial_height bounce_ratio k < 10 := by
  let initial_height := 800
  let bounce_ratio := 3 / 4
  use 16
  sorry

end ball_height_less_than_10_after_16_bounces_l2363_236384


namespace lines_through_point_l2363_236369

theorem lines_through_point {a b c : ℝ} :
  (3 = a + b) ∧ (3 = b + c) ∧ (3 = c + a) → (a = 1.5 ∧ b = 1.5 ∧ c = 1.5) :=
by
  intros h
  sorry

end lines_through_point_l2363_236369


namespace table_price_l2363_236361

theorem table_price :
  ∃ C T : ℝ, (2 * C + T = 0.6 * (C + 2 * T)) ∧ (C + T = 72) ∧ (T = 63) :=
by
  sorry

end table_price_l2363_236361


namespace volume_of_triangular_prism_l2363_236316

theorem volume_of_triangular_prism (S_side_face : ℝ) (distance : ℝ) :
  ∃ (Volume_prism : ℝ), Volume_prism = 1/2 * (S_side_face * distance) :=
by sorry

end volume_of_triangular_prism_l2363_236316


namespace dan_bought_one_candy_bar_l2363_236320

-- Define the conditions
def initial_money : ℕ := 4
def cost_per_candy_bar : ℕ := 3
def money_left : ℕ := 1

-- Define the number of candy bars Dan bought
def number_of_candy_bars_bought : ℕ := (initial_money - money_left) / cost_per_candy_bar

-- Prove the number of candy bars bought is equal to 1
theorem dan_bought_one_candy_bar : number_of_candy_bars_bought = 1 := by
  sorry

end dan_bought_one_candy_bar_l2363_236320


namespace people_came_later_l2363_236315

theorem people_came_later (lollipop_ratio initial_people lollipops : ℕ) 
  (h1 : lollipop_ratio = 5) 
  (h2 : initial_people = 45) 
  (h3 : lollipops = 12) : 
  (lollipops * lollipop_ratio - initial_people) = 15 := by 
  sorry

end people_came_later_l2363_236315


namespace number_of_zeros_of_f_l2363_236300

noncomputable def f (x : ℝ) := (1 / 3) * x ^ 3 - x ^ 2 - 3 * x + 9

theorem number_of_zeros_of_f : ∃ (z : ℕ), z = 2 ∧ ∀ x : ℝ, (f x = 0 → x = -3 ∨ x = -2 / 3 ∨ x = 1 ∨ x = 3) := 
sorry

end number_of_zeros_of_f_l2363_236300


namespace range_of_b_div_a_l2363_236326

theorem range_of_b_div_a (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
(h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  (2 / 3 : ℝ) ≤ b / a ∧ b / a ≤ (3 / 2 : ℝ) :=
sorry

end range_of_b_div_a_l2363_236326


namespace polynomial_identity_l2363_236370

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l2363_236370


namespace quadrilateral_EFGH_area_l2363_236301

-- Definitions based on conditions
def quadrilateral_EFGH_right_angles (F H : ℝ) : Prop :=
  ∃ E G, E - F = 0 ∧ H - G = 0

def quadrilateral_length_hypotenuse (E G : ℝ) : Prop :=
  E - G = 5

def distinct_integer_lengths (EF FG EH HG : ℝ) : Prop :=
  EF ≠ FG ∧ EH ≠ HG ∧ ∃ a b : ℕ, EF = a ∧ FG = b ∧ EH = b ∧ HG = a ∧ a * a + b * b = 25

-- Proof statement
theorem quadrilateral_EFGH_area (F H : ℝ) 
  (EF FG EH HG E G : ℝ) 
  (h1 : quadrilateral_EFGH_right_angles F H) 
  (h2 : quadrilateral_length_hypotenuse E G)
  (h3 : distinct_integer_lengths EF FG EH HG) 
: 
  EF * FG / 2 + EH * HG / 2 = 12 := 
sorry

end quadrilateral_EFGH_area_l2363_236301


namespace express_x_n_prove_inequality_l2363_236363

variable (a b n : Real)
variable (x : ℕ → Real)

def trapezoid_conditions : Prop :=
  ∀ n, x 1 = a * b / (a + b) ∧ (x (n + 1) / x n = x (n + 1) / a)

theorem express_x_n (h : trapezoid_conditions a b x) : 
  ∀ n, x n = a * b / (a + n * b) := 
by
  sorry

theorem prove_inequality (h : trapezoid_conditions a b x) : 
  ∀ n, x n ≤ (a + n * b) / (4 * n) := 
by
  sorry

end express_x_n_prove_inequality_l2363_236363


namespace average_of_numbers_l2363_236390

theorem average_of_numbers : 
  (12 + 13 + 14 + 510 + 520 + 530 + 1115 + 1120 + 1 + 1252140 + 2345) / 11 = 114391 :=
by
  sorry

end average_of_numbers_l2363_236390


namespace correct_weights_swapped_l2363_236378

theorem correct_weights_swapped 
  (W X Y Z : ℝ) 
  (h1 : Z > Y) 
  (h2 : X > W) 
  (h3 : Y + Z > W + X) :
  (W, Z) = (Z, W) :=
sorry

end correct_weights_swapped_l2363_236378


namespace outfits_count_l2363_236312

-- Definitions of the counts of each type of clothing item
def num_blue_shirts : Nat := 6
def num_green_shirts : Nat := 4
def num_pants : Nat := 7
def num_blue_hats : Nat := 9
def num_green_hats : Nat := 7

-- Statement of the problem to prove
theorem outfits_count :
  (num_blue_shirts * num_pants * num_green_hats) + (num_green_shirts * num_pants * num_blue_hats) = 546 :=
by
  sorry

end outfits_count_l2363_236312
