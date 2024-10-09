import Mathlib

namespace classroom_student_count_l884_88459

-- Define the conditions and the question
theorem classroom_student_count (B G : ℕ) (h1 : B / G = 3 / 5) (h2 : G = B + 4) : B + G = 16 := by
  sorry

end classroom_student_count_l884_88459


namespace complement_M_intersect_N_l884_88447

def M : Set ℤ := {m | m ≤ -3 ∨ m ≥ 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}
def complement_M : Set ℤ := {m | -3 < m ∧ m < 2} 

theorem complement_M_intersect_N : (complement_M ∩ N) = {-1, 0, 1} := by
  sorry

end complement_M_intersect_N_l884_88447


namespace cost_per_gumball_l884_88427

theorem cost_per_gumball (total_money : ℕ) (num_gumballs : ℕ) (cost_each : ℕ) 
  (h1 : total_money = 32) (h2 : num_gumballs = 4) : cost_each = 8 :=
by
  sorry -- Proof omitted

end cost_per_gumball_l884_88427


namespace find_second_number_l884_88487

theorem find_second_number (a b c : ℚ) (h1 : a + b + c = 98) (h2 : a = (2 / 3) * b) (h3 : c = (8 / 5) * b) : b = 30 :=
by sorry

end find_second_number_l884_88487


namespace square_side_length_l884_88496

theorem square_side_length :
  ∀ (s : ℝ), (∃ w l : ℝ, w = 6 ∧ l = 24 ∧ s^2 = w * l) → s = 12 := by 
  sorry

end square_side_length_l884_88496


namespace number_of_friends_l884_88438

-- Define the conditions
def initial_apples := 55
def apples_given_to_father := 10
def apples_per_person := 9

-- Define the formula to calculate the number of friends
def friends (initial_apples apples_given_to_father apples_per_person : ℕ) : ℕ :=
  (initial_apples - apples_given_to_father - apples_per_person) / apples_per_person

-- State the Lean theorem
theorem number_of_friends :
  friends initial_apples apples_given_to_father apples_per_person = 4 :=
by
  sorry

end number_of_friends_l884_88438


namespace tank_dimension_l884_88420

theorem tank_dimension (cost_per_sf : ℝ) (total_cost : ℝ) (length1 length3 : ℝ) (surface_area : ℝ) (dimension : ℝ) :
  cost_per_sf = 20 ∧ total_cost = 1520 ∧ 
  length1 = 4 ∧ length3 = 2 ∧ 
  surface_area = total_cost / cost_per_sf ∧
  12 * dimension + 16 = surface_area → dimension = 5 :=
by
  intro h
  obtain ⟨hcps, htac, hl1, hl3, hsa, heq⟩ := h
  sorry

end tank_dimension_l884_88420


namespace problem1_problem2_l884_88448

-- Problem 1: Prove the expression equals 5
theorem problem1 : (1 : ℚ) * ((1/3 : ℚ) - (3/4) + (5/6)) / (1/12) = 5 := by
  sorry

-- Problem 2: Prove the expression equals 7
theorem problem2 : ((-1 : ℤ)^2023 + |(1 - 0.5 : ℚ)| * ((-4)^2)) = 7 := by
  sorry

end problem1_problem2_l884_88448


namespace merchant_spent_initially_500_rubles_l884_88469

theorem merchant_spent_initially_500_rubles
  (x : ℕ)
  (h1 : x + 100 > x)
  (h2 : x + 220 > x + 100)
  (h3 : x * (x + 220) = (x + 100) * (x + 100))
  : x = 500 := sorry

end merchant_spent_initially_500_rubles_l884_88469


namespace determinant_roots_cubic_eq_l884_88480

noncomputable def determinant_of_matrix (a b c : ℝ) : ℝ :=
  a * (b * c - 1) - (c - 1) + (1 - b)

theorem determinant_roots_cubic_eq {a b c p q r : ℝ}
  (h1 : a + b + c = p)
  (h2 : a * b + b * c + c * a = q)
  (h3 : a * b * c = r) :
  determinant_of_matrix a b c = r - p + 2 :=
by {
  sorry
}

end determinant_roots_cubic_eq_l884_88480


namespace number_of_children_on_bus_l884_88453

theorem number_of_children_on_bus (initial_children : ℕ) (additional_children : ℕ) (total_children : ℕ) 
  (h1 : initial_children = 26) (h2 : additional_children = 38) : total_children = 64 :=
by
  sorry

end number_of_children_on_bus_l884_88453


namespace all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l884_88455

def coin_values : Set ℤ := {1, 5, 10, 25}

theorem all_values_achievable (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 30) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_1 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 40) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_2 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 50) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_3 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 60) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

theorem all_values_achievable_4 (a b c d: ℕ) (h: a + b + c + d = 6) (h_a: a * 1 + b * 5 + c * 10 + d * 25 = 70) 
  (coins: Set ℤ := coin_values) : 
  ∃ (x y z w: ℕ), x + y + z + w = 6 ∧ x * 1 + y * 5 + z * 10 + w * 25 = a * 1 + b * 5 + c * 10 + d * 25 :=
by sorry

end all_values_achievable_all_values_achievable_1_all_values_achievable_2_all_values_achievable_3_all_values_achievable_4_l884_88455


namespace doubled_volume_l884_88430

theorem doubled_volume (V : ℕ) (h : V = 4) : 8 * V = 32 := by
  sorry

end doubled_volume_l884_88430


namespace find_radius_of_circle_l884_88424

theorem find_radius_of_circle
  (a b R : ℝ)
  (h1 : R^2 = a * b) :
  R = Real.sqrt (a * b) :=
by
  sorry

end find_radius_of_circle_l884_88424


namespace count_neither_multiples_of_2_nor_3_l884_88404

theorem count_neither_multiples_of_2_nor_3 : 
  let count_multiples (k n : ℕ) : ℕ := n / k
  let total_numbers := 100
  let multiples_of_2 := count_multiples 2 total_numbers
  let multiples_of_3 := count_multiples 3 total_numbers
  let multiples_of_6 := count_multiples 6 total_numbers
  let multiples_of_2_or_3 := multiples_of_2 + multiples_of_3 - multiples_of_6
  total_numbers - multiples_of_2_or_3 = 33 :=
by 
  sorry

end count_neither_multiples_of_2_nor_3_l884_88404


namespace pow_two_grows_faster_than_square_l884_88458

theorem pow_two_grows_faster_than_square (n : ℕ) (h : n ≥ 5) : 2^n > n^2 := sorry

end pow_two_grows_faster_than_square_l884_88458


namespace squares_area_relation_l884_88462

/-- 
Given:
1. $\alpha$ such that $\angle 1 = \angle 2 = \angle 3 = \alpha$
2. The areas of the squares are given by:
   - $S_A = \cos^4 \alpha$
   - $S_D = \sin^4 \alpha$
   - $S_B = \cos^2 \alpha \sin^2 \alpha$
   - $S_C = \cos^2 \alpha \sin^2 \alpha$

Prove that:
$S_A \cdot S_D = S_B \cdot S_C$
--/

theorem squares_area_relation (α : ℝ) :
  (Real.cos α)^4 * (Real.sin α)^4 = (Real.cos α)^2 * (Real.sin α)^2 * (Real.cos α)^2 * (Real.sin α)^2 :=
by sorry

end squares_area_relation_l884_88462


namespace cricket_run_target_l884_88450

/-- Assuming the run rate in the first 15 overs and the required run rate for the next 35 overs to
reach a target, prove that the target number of runs is 275. -/
theorem cricket_run_target
  (run_rate_first_15 : ℝ := 3.2)
  (overs_first_15 : ℝ := 15)
  (run_rate_remaining_35 : ℝ := 6.485714285714286)
  (overs_remaining_35 : ℝ := 35)
  (runs_first_15 := run_rate_first_15 * overs_first_15)
  (runs_remaining_35 := run_rate_remaining_35 * overs_remaining_35)
  (target_runs := runs_first_15 + runs_remaining_35) :
  target_runs = 275 := by
  sorry

end cricket_run_target_l884_88450


namespace regular_polygon_is_octagon_l884_88445

theorem regular_polygon_is_octagon (n : ℕ) (interior_angle exterior_angle : ℝ) :
  interior_angle = 3 * exterior_angle ∧ interior_angle + exterior_angle = 180 → n = 8 :=
by
  intros h
  sorry

end regular_polygon_is_octagon_l884_88445


namespace distance_Q_to_EH_l884_88475

noncomputable def N : ℝ × ℝ := (3, 0)
noncomputable def E : ℝ × ℝ := (0, 6)
noncomputable def circle1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16
noncomputable def circle2 (x y : ℝ) : Prop := x^2 + (y - 6)^2 = 9
noncomputable def EH_line (y : ℝ) : Prop := y = 6

theorem distance_Q_to_EH :
  ∃ (Q : ℝ × ℝ), circle1 Q.1 Q.2 ∧ circle2 Q.1 Q.2 ∧ Q ≠ (0, 0) ∧ abs (Q.2 - 6) = 19 / 3 := sorry

end distance_Q_to_EH_l884_88475


namespace total_points_scored_l884_88433

theorem total_points_scored
    (Bailey_points Chandra_points Akiko_points Michiko_points : ℕ)
    (h1 : Bailey_points = 14)
    (h2 : Michiko_points = Bailey_points / 2)
    (h3 : Akiko_points = Michiko_points + 4)
    (h4 : Chandra_points = 2 * Akiko_points) :
  Bailey_points + Michiko_points + Akiko_points + Chandra_points = 54 := by
  sorry

end total_points_scored_l884_88433


namespace smallest_n_l884_88498

theorem smallest_n (n : ℕ) : 
  (2^n + 5^n - n) % 1000 = 0 ↔ n = 797 :=
sorry

end smallest_n_l884_88498


namespace jenny_correct_number_l884_88484

theorem jenny_correct_number (x : ℤ) (h : x - 26 = -14) : x + 26 = 38 :=
by
  sorry

end jenny_correct_number_l884_88484


namespace millie_initial_bracelets_l884_88417

theorem millie_initial_bracelets (n : ℕ) (h1 : n - 2 = 7) : n = 9 :=
sorry

end millie_initial_bracelets_l884_88417


namespace building_time_l884_88499

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end building_time_l884_88499


namespace thirteenth_term_geometric_sequence_l884_88485

theorem thirteenth_term_geometric_sequence 
  (a : ℕ → ℕ) 
  (r : ℝ)
  (h₁ : a 7 = 7) 
  (h₂ : a 10 = 21)
  (h₃ : ∀ (n : ℕ), a (n + 1) = a n * r) : 
  a 13 = 63 := 
by
  -- proof needed
  sorry

end thirteenth_term_geometric_sequence_l884_88485


namespace ratio_eq_thirteen_fifths_l884_88446

theorem ratio_eq_thirteen_fifths
  (a b c : ℝ)
  (h₁ : b / a = 4)
  (h₂ : c / b = 2) :
  (a + b + c) / (a + b) = 13 / 5 :=
sorry

end ratio_eq_thirteen_fifths_l884_88446


namespace tan_beta_identity_l884_88466

theorem tan_beta_identity (α β : ℝ) (h1 : Real.tan α = 1/3) (h2 : Real.tan (α + β) = 1/2) :
  Real.tan β = 1/7 :=
sorry

end tan_beta_identity_l884_88466


namespace product_b6_b8_is_16_l884_88476

-- Given conditions
variable (a : ℕ → ℝ) -- Sequence a_n
variable (b : ℕ → ℝ) -- Sequence b_n

-- Condition 1: Arithmetic sequence a_n and non-zero
axiom a_is_arithmetic : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d
axiom a_non_zero : ∃ n, a n ≠ 0

-- Condition 2: Equation 2a_3 - a_7^2 + 2a_n = 0
axiom a_satisfies_eq : ∀ n : ℕ, 2 * a 3 - (a 7) ^ 2 + 2 * a n = 0

-- Condition 3: Geometric sequence b_n with b_7 = a_7
axiom b_is_geometric : ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n
axiom b7_equals_a7 : b 7 = a 7

-- Prove statement
theorem product_b6_b8_is_16 : b 6 * b 8 = 16 := sorry

end product_b6_b8_is_16_l884_88476


namespace MitchWorks25Hours_l884_88411

noncomputable def MitchWorksHours : Prop :=
  let weekday_earnings_rate := 3
  let weekend_earnings_rate := 6
  let weekly_earnings := 111
  let weekend_hours := 6
  let weekday_hours (x : ℕ) := 5 * x
  let weekend_earnings := weekend_hours * weekend_earnings_rate
  let weekday_earnings (x : ℕ) := x * weekday_earnings_rate
  let total_weekday_earnings (x : ℕ) := weekly_earnings - weekend_earnings
  ∀ (x : ℕ), weekday_earnings x = total_weekday_earnings x → x = 25

theorem MitchWorks25Hours : MitchWorksHours := by
  sorry

end MitchWorks25Hours_l884_88411


namespace last_digit_of_sum_of_powers_l884_88468

theorem last_digit_of_sum_of_powers {a b c d : ℕ} 
  (h1 : a = 2311) (h2 : b = 5731) (h3 : c = 3467) (h4 : d = 6563) 
  : (a^b + c^d) % 10 = 4 := by
  sorry

end last_digit_of_sum_of_powers_l884_88468


namespace trishul_invested_percentage_less_than_raghu_l884_88461

variable {T V R : ℝ}

def vishal_invested_more (T V : ℝ) : Prop :=
  V = 1.10 * T

def total_sum_of_investments (T V : ℝ) : Prop :=
  T + V + 2300 = 6647

def raghu_investment : ℝ := 2300

theorem trishul_invested_percentage_less_than_raghu
  (h1 : vishal_invested_more T V)
  (h2 : total_sum_of_investments T V) :
  ((raghu_investment - T) / raghu_investment) * 100 = 10 :=
  sorry

end trishul_invested_percentage_less_than_raghu_l884_88461


namespace correct_divisor_l884_88470

theorem correct_divisor (X : ℕ) (D : ℕ) (H1 : X = 24 * 87) (H2 : X / D = 58) : D = 36 :=
by
  sorry

end correct_divisor_l884_88470


namespace auction_theorem_l884_88444

def auctionProblem : Prop :=
  let starting_value := 300
  let harry_bid_round1 := starting_value + 200
  let alice_bid_round1 := harry_bid_round1 * 2
  let bob_bid_round1 := harry_bid_round1 * 3
  let highest_bid_round1 := bob_bid_round1
  let carol_bid_round2 := highest_bid_round1 * 1.5
  let sum_previous_increases := (harry_bid_round1 - starting_value) + 
                                 (alice_bid_round1 - harry_bid_round1) + 
                                 (bob_bid_round1 - harry_bid_round1)
  let dave_bid_round2 := carol_bid_round2 + sum_previous_increases
  let highest_other_bid_round3 := dave_bid_round2
  let harry_final_bid_round3 := 6000
  let difference := harry_final_bid_round3 - highest_other_bid_round3
  difference = 2050

theorem auction_theorem : auctionProblem :=
by
  sorry

end auction_theorem_l884_88444


namespace lunks_needed_for_12_apples_l884_88464

/-- 
  Given:
  1. 7 lunks can be traded for 4 kunks.
  2. 3 kunks will buy 5 apples.

  Prove that the number of lunks needed to purchase one dozen (12) apples is equal to 14.
-/
theorem lunks_needed_for_12_apples (L K : ℕ)
  (h1 : 7 * L = 4 * K)
  (h2 : 3 * K = 5) :
  (8 * K = 14 * L) :=
by
  sorry

end lunks_needed_for_12_apples_l884_88464


namespace sum_of_zeros_of_even_function_is_zero_l884_88422

open Function

theorem sum_of_zeros_of_even_function_is_zero (f : ℝ → ℝ) (hf: Even f) (hx: ∃ x1 x2 x3 x4 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) :
  x1 + x2 + x3 + x4 = 0 := by
  sorry

end sum_of_zeros_of_even_function_is_zero_l884_88422


namespace simplify_fraction_multiplication_l884_88432

theorem simplify_fraction_multiplication :
  8 * (15 / 4) * (-40 / 45) = -64 / 9 :=
by
  sorry

end simplify_fraction_multiplication_l884_88432


namespace cycling_speed_l884_88402

-- Definitions based on given conditions.
def ratio_L_B : ℕ := 1
def ratio_B_L : ℕ := 2
def area_of_park : ℕ := 20000
def time_in_minutes : ℕ := 6

-- The question translated to Lean 4 statement.
theorem cycling_speed (L B : ℕ) (h1 : ratio_L_B * B = ratio_B_L * L)
  (h2 : L * B = area_of_park)
  (h3 : B = 2 * L) :
  (2 * L + 2 * B) / (time_in_minutes / 60) = 6000 := by
  sorry

end cycling_speed_l884_88402


namespace sum_of_probability_fractions_l884_88425

def total_tree_count := 15
def non_birch_count := 9
def birch_count := 6
def total_arrangements := Nat.choose 15 6
def non_adjacent_birch_arrangements := Nat.choose 10 6
def birch_probability := non_adjacent_birch_arrangements / total_arrangements
def simplified_probability_numerator := 6
def simplified_probability_denominator := 143
def answer := simplified_probability_numerator + simplified_probability_denominator

theorem sum_of_probability_fractions :
  answer = 149 := by
  sorry

end sum_of_probability_fractions_l884_88425


namespace fraction_not_integer_l884_88486

def containsExactlyTwoOccurrences (d : List ℕ) : Prop :=
  ∀ n ∈ [1, 2, 3, 4, 5, 6, 7], d.count n = 2

theorem fraction_not_integer
  (k m : ℕ)
  (hk : 14 = (List.length (Nat.digits 10 k)))
  (hm : 14 = (List.length (Nat.digits 10 m)))
  (hkd : containsExactlyTwoOccurrences (Nat.digits 10 k))
  (hmd : containsExactlyTwoOccurrences (Nat.digits 10 m))
  (hkm : k ≠ m) :
  ¬ ∃ d : ℕ, k = m * d := 
sorry

end fraction_not_integer_l884_88486


namespace proof_stmt_l884_88452

variable (a x y : ℝ)
variable (ha : a > 0) (hneq : a ≠ 1)

noncomputable def S (x : ℝ) := a^x - a^(-x)
noncomputable def C (x : ℝ) := a^x + a^(-x)

theorem proof_stmt :
  2 * S a (x + y) = S a x * C a y + C a x * S a y ∧
  2 * S a (x - y) = S a x * C a y - C a x * S a y :=
by sorry

end proof_stmt_l884_88452


namespace temperature_difference_l884_88429

theorem temperature_difference (T_south T_north : ℤ) (h1 : T_south = -7) (h2 : T_north = -15) :
  T_south - T_north = 8 :=
by
  sorry

end temperature_difference_l884_88429


namespace mr_brown_selling_price_l884_88413

noncomputable def initial_price : ℝ := 100000
noncomputable def profit_percentage : ℝ := 0.10
noncomputable def loss_percentage : ℝ := 0.10

def selling_price_mr_brown (initial_price profit_percentage : ℝ) : ℝ :=
  initial_price * (1 + profit_percentage)

def selling_price_to_friend (selling_price_mr_brown loss_percentage : ℝ) : ℝ :=
  selling_price_mr_brown * (1 - loss_percentage)

theorem mr_brown_selling_price :
  selling_price_to_friend (selling_price_mr_brown initial_price profit_percentage) loss_percentage = 99000 :=
by
  sorry

end mr_brown_selling_price_l884_88413


namespace petya_vacation_days_l884_88412

-- Defining the conditions
def total_days : ℕ := 90

def swims (d : ℕ) : Prop := d % 2 = 0
def shops (d : ℕ) : Prop := d % 3 = 0
def solves_math (d : ℕ) : Prop := d % 5 = 0

def does_all (d : ℕ) : Prop := swims d ∧ shops d ∧ solves_math d

def does_any_task (d : ℕ) : Prop := swims d ∨ shops d ∨ solves_math d

-- "Pleasant" days definition: swims, not shops, not solves math
def is_pleasant_day (d : ℕ) : Prop := swims d ∧ ¬shops d ∧ ¬solves_math d
-- "Boring" days definition: does nothing
def is_boring_day (d : ℕ) : Prop := ¬does_any_task d

-- Theorem stating the number of pleasant and boring days
theorem petya_vacation_days :
  (∃ pleasant_days : Finset ℕ, pleasant_days.card = 24 ∧ ∀ d ∈ pleasant_days, is_pleasant_day d)
  ∧ (∃ boring_days : Finset ℕ, boring_days.card = 24 ∧ ∀ d ∈ boring_days, is_boring_day d) :=
by
  sorry

end petya_vacation_days_l884_88412


namespace evaluate_difference_of_squares_l884_88409

theorem evaluate_difference_of_squares :
  (50^2 - 30^2 = 1600) :=
by sorry

end evaluate_difference_of_squares_l884_88409


namespace max_correct_answers_l884_88437

variable (c w b : ℕ) -- Define c, w, b as natural numbers

theorem max_correct_answers (h1 : c + w + b = 30) (h2 : 4 * c - w = 70) : c ≤ 20 := by
  sorry

end max_correct_answers_l884_88437


namespace total_revenue_l884_88473

-- Definitions based on the conditions
def ticket_price : ℕ := 25
def first_show_tickets : ℕ := 200
def second_show_tickets : ℕ := 3 * first_show_tickets

-- Statement to prove the problem
theorem total_revenue : (first_show_tickets * ticket_price + second_show_tickets * ticket_price) = 20000 :=
by
  sorry

end total_revenue_l884_88473


namespace sandbox_area_l884_88474

def length : ℕ := 312
def width : ℕ := 146
def area : ℕ := 45552

theorem sandbox_area : length * width = area := by
  sorry

end sandbox_area_l884_88474


namespace max_number_soap_boxes_l884_88490

-- Definition of dimensions and volumes
def carton_length : ℕ := 25
def carton_width : ℕ := 42
def carton_height : ℕ := 60
def soap_box_length : ℕ := 7
def soap_box_width : ℕ := 12
def soap_box_height : ℕ := 5

def volume (l w h : ℕ) : ℕ := l * w * h

-- Volumes of the carton and soap box
def carton_volume : ℕ := volume carton_length carton_width carton_height
def soap_box_volume : ℕ := volume soap_box_length soap_box_width soap_box_height

-- The maximum number of soap boxes that can be placed in the carton
def max_soap_boxes : ℕ := carton_volume / soap_box_volume

theorem max_number_soap_boxes :
  max_soap_boxes = 150 :=
by
  -- Proof here
  sorry

end max_number_soap_boxes_l884_88490


namespace marcus_baseball_cards_l884_88477

/-- 
Marcus initially has 210.0 baseball cards.
Carter gives Marcus 58.0 more baseball cards.
Prove that Marcus now has 268.0 baseball cards.
-/
theorem marcus_baseball_cards (initial_cards : ℝ) (additional_cards : ℝ) 
  (h_initial : initial_cards = 210.0) (h_additional : additional_cards = 58.0) : 
  initial_cards + additional_cards = 268.0 :=
  by
    sorry

end marcus_baseball_cards_l884_88477


namespace prove_y_eq_x_l884_88441

theorem prove_y_eq_x
  (x y : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (h1 : x = 2 + 1 / y)
  (h2 : y = 2 + 1 / x) : y = x :=
sorry

end prove_y_eq_x_l884_88441


namespace max_prime_area_of_rectangle_with_perimeter_40_is_19_l884_88465

-- Predicate to check if a number is prime
def is_prime (n : ℕ) : Prop := sorry

-- Given conditions: perimeter of 40 units; perimeter condition and area as prime number.
def max_prime_area_of_rectangle_with_perimeter_40 : Prop :=
  ∃ (l w : ℕ), l + w = 20 ∧ is_prime (l * (20 - l)) ∧
  ∀ (l' w' : ℕ), l' + w' = 20 → is_prime (l' * (20 - l')) → (l * (20 - l)) ≥ (l' * (20 - l'))

theorem max_prime_area_of_rectangle_with_perimeter_40_is_19 :
  max_prime_area_of_rectangle_with_perimeter_40 :=
sorry

end max_prime_area_of_rectangle_with_perimeter_40_is_19_l884_88465


namespace parabola_ellipse_focus_l884_88435

theorem parabola_ellipse_focus (p : ℝ) :
  (∃ (x y : ℝ), x^2 = 2 * p * y ∧ y = -1 ∧ x = 0) →
  p = -2 :=
by
  sorry

end parabola_ellipse_focus_l884_88435


namespace trajectory_C_find_m_l884_88414

noncomputable def trajectory_C_eq (x y : ℝ) : Prop :=
  (x - 3)^2 + y^2 = 7

theorem trajectory_C (x y : ℝ) (hx : trajectory_C_eq x y) :
  (x - 3)^2 + y^2 = 7 := by
  sorry

theorem find_m (m : ℝ) : (∃ x1 x2 y1 y2 : ℝ, x1 + x2 = 3 + m ∧ x1 * x2 + (1/(2:ℝ)) * ((m^2 + 2)/(2:ℝ)) = 0 ∧ x1 * x2 + (x1 - m) * (x2 - m) = 0) → m = 1 ∨ m = 2 := by
  sorry

end trajectory_C_find_m_l884_88414


namespace min_goals_in_previous_three_matches_l884_88416

theorem min_goals_in_previous_three_matches 
  (score1 score2 score3 score4 : ℕ)
  (total_after_seven_matches : ℕ)
  (previous_three_goal_sum : ℕ) :
  score1 = 18 →
  score2 = 12 →
  score3 = 15 →
  score4 = 14 →
  total_after_seven_matches ≥ 100 →
  previous_three_goal_sum = total_after_seven_matches - (score1 + score2 + score3 + score4) →
  (previous_three_goal_sum / 3 : ℝ) < ((score1 + score2 + score3 + score4) / 4 : ℝ) →
  previous_three_goal_sum ≥ 41 :=
by
  sorry

end min_goals_in_previous_three_matches_l884_88416


namespace clock_four_different_digits_l884_88481

noncomputable def total_valid_minutes : ℕ :=
  let minutes_from_00_00_to_19_59 := 20 * 60
  let valid_minutes_1 := 2 * 9 * 4 * 7
  let minutes_from_20_00_to_23_59 := 4 * 60
  let valid_minutes_2 := 1 * 3 * 4 * 7
  valid_minutes_1 + valid_minutes_2

theorem clock_four_different_digits : total_valid_minutes = 588 :=
by
  sorry

end clock_four_different_digits_l884_88481


namespace parallelogram_side_lengths_l884_88460

theorem parallelogram_side_lengths (x y : ℚ) 
  (h1 : 12 * x - 2 = 10) 
  (h2 : 5 * y + 5 = 4) : 
  x + y = 4 / 5 := 
by 
  sorry

end parallelogram_side_lengths_l884_88460


namespace solution_set_even_function_l884_88405

/-- Let f be an even function, and for x in [0, ∞), f(x) = x - 1. Determine the solution set for the inequality f(x) > 1.
We prove that the solution set is {x | x < -2 or x > 2}. -/
theorem solution_set_even_function (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_def : ∀ x, 0 ≤ x → f x = x - 1) :
  {x : ℝ | f x > 1} = {x | x < -2 ∨ x > 2} :=
by
  sorry  -- Proof steps go here.

end solution_set_even_function_l884_88405


namespace quadratic_equation_statements_l884_88479

theorem quadratic_equation_statements (a b c : ℝ) (h₀ : a ≠ 0) :
  (if -4 * a * c > 0 then (b^2 - 4 * a * c) > 0 else false) ∧
  ¬((b^2 - 4 * a * c > 0) → (b^2 - 4 * c * a > 0)) ∧
  ¬((c^2 * a + c * b + c = 0) → (a * c + b + 1 = 0)) ∧
  ¬(∀ (x₀ : ℝ), (a * x₀^2 + b * x₀ + c = 0) → (b^2 - 4 * a * c = (2 * a * x₀ - b)^2)) :=
by
    sorry

end quadratic_equation_statements_l884_88479


namespace distinct_arrays_for_48_chairs_with_conditions_l884_88483

theorem distinct_arrays_for_48_chairs_with_conditions : 
  ∃ n : ℕ, n = 7 ∧ 
    ∀ (m r c : ℕ), 
      m = 48 ∧ 
      2 ≤ r ∧ 
      2 ≤ c ∧ 
      r * c = m ↔ 
      (∃ (k : ℕ), 
         ((k = (m / r) ∧ r * (m / r) = m) ∨ (k = (m / c) ∧ c * (m / c) = m)) ∧ 
         r * c = m) → 
    n = 7 :=
by
  sorry

end distinct_arrays_for_48_chairs_with_conditions_l884_88483


namespace pure_imaginary_denom_rationalization_l884_88495

theorem pure_imaginary_denom_rationalization (a : ℝ) : 
  (∃ b : ℝ, 1 - a * Complex.I * Complex.I = b * Complex.I) → a = 0 :=
by
  sorry

end pure_imaginary_denom_rationalization_l884_88495


namespace paint_total_gallons_l884_88418

theorem paint_total_gallons
  (white_paint_gallons : ℕ)
  (blue_paint_gallons : ℕ)
  (h_wp : white_paint_gallons = 660)
  (h_bp : blue_paint_gallons = 6029) :
  white_paint_gallons + blue_paint_gallons = 6689 := 
by
  sorry

end paint_total_gallons_l884_88418


namespace petya_friends_count_l884_88431

theorem petya_friends_count (x : ℕ) (total_stickers : ℤ)
    (h1 : total_stickers = 5 * x + 8)
    (h2 : total_stickers = 6 * x - 11) :
    x = 19 :=
by
  -- Here we use the given conditions h1 and h2 to prove x = 19.
  sorry

end petya_friends_count_l884_88431


namespace range_of_f_on_nonneg_reals_l884_88408

theorem range_of_f_on_nonneg_reals (k : ℕ) (h_even : k % 2 = 0) (h_pos : 0 < k) :
    ∀ y : ℝ, 0 ≤ y ↔ ∃ x : ℝ, 0 ≤ x ∧ x^k = y :=
by
  sorry

end range_of_f_on_nonneg_reals_l884_88408


namespace gcf_120_180_240_l884_88403

def gcf (a b : ℕ) : ℕ :=
  Nat.gcd a b

theorem gcf_120_180_240 : gcf (gcf 120 180) 240 = 60 := by
  have h₁ : 120 = 2^3 * 3 * 5 := by norm_num
  have h₂ : 180 = 2^2 * 3^2 * 5 := by norm_num
  have h₃ : 240 = 2^4 * 3 * 5 := by norm_num
  have gcf_120_180 : gcf 120 180 = 60 := by
    -- Proof of GCF for 120 and 180
    sorry  -- Placeholder for the specific proof steps
  have gcf_60_240 : gcf 60 240 = 60 := by
    -- Proof of GCF for 60 and 240
    sorry  -- Placeholder for the specific proof steps
  -- Conclude the overall GCF
  exact gcf_60_240

end gcf_120_180_240_l884_88403


namespace compare_xyz_l884_88456

open Real

noncomputable def x : ℝ := 6 * log 3 / log 64
noncomputable def y : ℝ := (1 / 3) * log 64 / log 3
noncomputable def z : ℝ := (3 / 2) * log 3 / log 8

theorem compare_xyz : x > y ∧ y > z := 
by {
  sorry
}

end compare_xyz_l884_88456


namespace sum_of_integers_between_60_and_460_ending_in_2_is_10280_l884_88406

-- We define the sequence.
def endsIn2Seq : List Int := List.range' 62 (452 + 1 - 62) 10  -- Generates [62, 72, ..., 452]

-- The sum of the sequence.
def sumEndsIn2Seq : Int := endsIn2Seq.sum

-- The theorem to prove the desired sum.
theorem sum_of_integers_between_60_and_460_ending_in_2_is_10280 :
  sumEndsIn2Seq = 10280 := by
  -- Proof is omitted
  sorry

end sum_of_integers_between_60_and_460_ending_in_2_is_10280_l884_88406


namespace inequality_proof_l884_88497

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + (33 * b^2 - b) / (33 * b^2 + 1) + (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 :=
sorry

end inequality_proof_l884_88497


namespace mouse_lives_count_l884_88419

-- Define the basic conditions
def catLives : ℕ := 9
def dogLives : ℕ := catLives - 3
def mouseLives : ℕ := dogLives + 7

-- The main theorem to prove
theorem mouse_lives_count : mouseLives = 13 :=
by
  -- proof steps go here
  sorry

end mouse_lives_count_l884_88419


namespace pet_snake_cost_l884_88457

theorem pet_snake_cost (original_amount left_amount snake_cost : ℕ) 
  (h1 : original_amount = 73) 
  (h2 : left_amount = 18)
  (h3 : snake_cost = original_amount - left_amount) : 
  snake_cost = 55 := 
by 
  sorry

end pet_snake_cost_l884_88457


namespace find_alpha_l884_88489

open Real

def alpha_is_acute (α : ℝ) : Prop := 0 < α ∧ α < π / 2

theorem find_alpha (α : ℝ) (h1 : alpha_is_acute α) (h2 : sin (α - 10 * (pi / 180)) = sqrt 3 / 2) : α = 70 * (pi / 180) :=
sorry

end find_alpha_l884_88489


namespace emily_page_production_difference_l884_88407

variables (p h : ℕ)

def first_day_pages (p h : ℕ) : ℕ := p * h
def second_day_pages (p h : ℕ) : ℕ := (p - 3) * (h + 3)
def page_difference (p h : ℕ) : ℕ := second_day_pages p h - first_day_pages p h

theorem emily_page_production_difference (h : ℕ) (p_eq_3h : p = 3 * h) :
  page_difference p h = 6 * h - 9 :=
by sorry

end emily_page_production_difference_l884_88407


namespace max_area_of_sector_l884_88472

theorem max_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 12) : 
  (1 / 2) * l * r ≤ 9 :=
by sorry

end max_area_of_sector_l884_88472


namespace cos_double_angle_l884_88454

theorem cos_double_angle (α : ℝ) (h : Real.sin ((Real.pi / 6) + α) = 1 / 3) :
  Real.cos ((2 * Real.pi / 3) - 2 * α) = -7 / 9 := by
  sorry

end cos_double_angle_l884_88454


namespace good_numbers_product_sum_digits_not_equal_l884_88463

def is_good_number (n : ℕ) : Prop :=
  n.digits 10 ⊆ [0, 1]

theorem good_numbers_product_sum_digits_not_equal (A B : ℕ) (hA : is_good_number A) (hB : is_good_number B) (hAB : is_good_number (A * B)) :
  ¬ ( (A.digits 10).sum * (B.digits 10).sum = ((A * B).digits 10).sum ) :=
sorry

end good_numbers_product_sum_digits_not_equal_l884_88463


namespace arithmetic_sequence_root_sum_l884_88436

theorem arithmetic_sequence_root_sum (a : ℕ → ℝ) (h_arith : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) 
    (h_roots : (a 3) * (a 8) + 3 * (a 3) + 3 * (a 8) - 18 = 0) : a 5 + a 6 = 3 := by
  sorry

end arithmetic_sequence_root_sum_l884_88436


namespace movie_theater_total_revenue_l884_88415

noncomputable def revenue_from_matinee_tickets : ℕ := 20 * 5 * 1 / 2 + 180 * 5
noncomputable def revenue_from_evening_tickets : ℕ := 150 * 12 * 9 / 10 + 75 * 12 * 75 / 100 + 75 * 12
noncomputable def revenue_from_3d_tickets : ℕ := 60 * 23 + 25 * 20 * 85 / 100 + 15 * 20
noncomputable def revenue_from_late_night_tickets : ℕ := 30 * 10 * 12 / 10 + 20 * 10

noncomputable def total_revenue : ℕ :=
  revenue_from_matinee_tickets + revenue_from_evening_tickets +
  revenue_from_3d_tickets + revenue_from_late_night_tickets

theorem movie_theater_total_revenue : total_revenue = 6810 := by
  sorry

end movie_theater_total_revenue_l884_88415


namespace solve_for_x_l884_88493

theorem solve_for_x (x y : ℚ) (h1 : 2 * x - 3 * y = 15) (h2 : x + 2 * y = 8) : x = 54 / 7 :=
sorry

end solve_for_x_l884_88493


namespace new_person_weight_l884_88492

theorem new_person_weight (W : ℝ) (N : ℝ) (old_weight : ℝ) (average_increase : ℝ) (num_people : ℕ)
  (h1 : num_people = 8)
  (h2 : old_weight = 45)
  (h3 : average_increase = 6)
  (h4 : (W - old_weight + N) / num_people = W / num_people + average_increase) :
  N = 93 :=
by
  sorry

end new_person_weight_l884_88492


namespace temperature_at_80_degrees_l884_88443

theorem temperature_at_80_degrees (t : ℝ) :
  (-t^2 + 10 * t + 60 = 80) ↔ (t = 5 + 3 * Real.sqrt 5 ∨ t = 5 - 3 * Real.sqrt 5) := by
  sorry

end temperature_at_80_degrees_l884_88443


namespace domain_of_sqrt_l884_88400

theorem domain_of_sqrt (x : ℝ) : (x - 1 ≥ 0) → (x ≥ 1) :=
by
  sorry

end domain_of_sqrt_l884_88400


namespace simplify_eq_neg_one_l884_88421

variable (a b c : ℝ)

noncomputable def simplify_expression := 
  1 / (b^2 + c^2 - a^2) + 1 / (a^2 + c^2 - b^2) + 1 / (a^2 + b^2 - c^2)

theorem simplify_eq_neg_one 
  (a_ne_zero : a ≠ 0) 
  (b_ne_zero : b ≠ 0) 
  (c_ne_zero : c ≠ 0) 
  (sum_eq_one : a + b + c = 1) 
  : simplify_expression a b c = -1 :=
by sorry

end simplify_eq_neg_one_l884_88421


namespace polynomial_relation_l884_88471

theorem polynomial_relation (x y : ℕ) :
  (x = 1 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 4) ∨ 
  (x = 3 ∧ y = 9) ∨ 
  (x = 4 ∧ y = 16) ∨ 
  (x = 5 ∧ y = 25) → 
  y = x^2 := 
by
  sorry

end polynomial_relation_l884_88471


namespace min_value_112_l884_88434

noncomputable def min_value_expr (a b c d : ℝ) : ℝ :=
  20 * (a^2 + b^2 + c^2 + d^2) - (a^3 * b + a^3 * c + a^3 * d + b^3 * a + b^3 * c + b^3 * d +
                                c^3 * a + c^3 * b + c^3 * d + d^3 * a + d^3 * b + d^3 * c)

theorem min_value_112 (a b c d : ℝ) (h : a + b + c + d = 8) : min_value_expr a b c d = 112 :=
  sorry

end min_value_112_l884_88434


namespace wheres_waldo_books_published_l884_88439

theorem wheres_waldo_books_published (total_minutes : ℕ) (minutes_per_puzzle : ℕ) (puzzles_per_book : ℕ)
  (h1 : total_minutes = 1350) (h2 : minutes_per_puzzle = 3) (h3 : puzzles_per_book = 30) :
  total_minutes / minutes_per_puzzle / puzzles_per_book = 15 :=
by
  sorry

end wheres_waldo_books_published_l884_88439


namespace geometric_common_ratio_eq_three_l884_88428

theorem geometric_common_ratio_eq_three 
  (a : ℕ → ℤ) 
  (d : ℤ) 
  (h_arithmetic_seq : ∀ n, a (n + 1) = a n + d)
  (h_nonzero_d : d ≠ 0) 
  (h_geom_seq : (a 2 + 2 * d) ^ 2 = (a 2 + d) * (a 2 + 5 * d)) : 
  (a 3) / (a 2) = 3 :=
by 
  sorry

end geometric_common_ratio_eq_three_l884_88428


namespace smallest_value_36k_minus_5l_l884_88401

theorem smallest_value_36k_minus_5l (k l : ℕ) :
  ∃ k l, 0 < 36^k - 5^l ∧ (∀ k' l', (0 < 36^k' - 5^l' → 36^k - 5^l ≤ 36^k' - 5^l')) ∧ 36^k - 5^l = 11 :=
by sorry

end smallest_value_36k_minus_5l_l884_88401


namespace calculate_expression_l884_88440

theorem calculate_expression :
  (5 / 19) * ((19 / 5) * (16 / 3) + (14 / 3) * (19 / 5)) = 10 :=
by
  sorry

end calculate_expression_l884_88440


namespace smallest_number_condition_l884_88494

theorem smallest_number_condition :
  ∃ n : ℕ, (n + 1) % 12 = 0 ∧
           (n + 1) % 18 = 0 ∧
           (n + 1) % 24 = 0 ∧
           (n + 1) % 32 = 0 ∧
           (n + 1) % 40 = 0 ∧
           n = 2879 :=
sorry

end smallest_number_condition_l884_88494


namespace add_fractions_l884_88451

-- Define the two fractions
def frac1 := 7 / 8
def frac2 := 9 / 12

-- The problem: addition of the two fractions and expressing in simplest form
theorem add_fractions : frac1 + frac2 = (13 : ℚ) / 8 := 
by 
  sorry

end add_fractions_l884_88451


namespace alcohol_percentage_in_new_mixture_l884_88491

theorem alcohol_percentage_in_new_mixture :
  let afterShaveLotionVolume := 200
  let afterShaveLotionConcentration := 0.35
  let solutionVolume := 75
  let solutionConcentration := 0.15
  let waterVolume := 50
  let totalVolume := afterShaveLotionVolume + solutionVolume + waterVolume
  let alcoholVolume := (afterShaveLotionVolume * afterShaveLotionConcentration) + (solutionVolume * solutionConcentration)
  let alcoholPercentage := (alcoholVolume / totalVolume) * 100
  alcoholPercentage = 25 := 
  sorry

end alcohol_percentage_in_new_mixture_l884_88491


namespace ratio_of_hexagon_areas_l884_88410

open Real

-- Define the given conditions about the hexagon and the midpoints
structure Hexagon :=
  (s : ℝ)
  (regular : True)
  (midpoints : True)

theorem ratio_of_hexagon_areas (h : Hexagon) : 
  let s := 2
  ∃ (area_ratio : ℝ), area_ratio = 4 / 7 :=
by
  sorry

end ratio_of_hexagon_areas_l884_88410


namespace count_valid_propositions_is_zero_l884_88426

theorem count_valid_propositions_is_zero :
  (∀ (a b : ℝ), (a > b → a^2 > b^2) = false) ∧
  (∀ (a b : ℝ), (a^2 > b^2 → a > b) = false) ∧
  (∀ (a b : ℝ), (a > b → b / a < 1) = false) ∧
  (∀ (a b : ℝ), (a > b → 1 / a < 1 / b) = false) :=
by
  sorry

end count_valid_propositions_is_zero_l884_88426


namespace sqrt_factorial_product_squared_l884_88488

open Nat

theorem sqrt_factorial_product_squared :
  (Real.sqrt ((factorial 5) * (factorial 4))) ^ 2 = 2880 := by
sorry

end sqrt_factorial_product_squared_l884_88488


namespace survived_trees_difference_l884_88478

theorem survived_trees_difference {original_trees died_trees survived_trees: ℕ} 
  (h1 : original_trees = 13) 
  (h2 : died_trees = 6)
  (h3 : survived_trees = original_trees - died_trees) :
  survived_trees - died_trees = 1 :=
by
  sorry

end survived_trees_difference_l884_88478


namespace find_x_l884_88449

theorem find_x (x y : ℝ) (h1 : x + 2 * y = 10) (h2 : y = 4) : x = 2 :=
by
  sorry

end find_x_l884_88449


namespace area_enclosed_by_graph_l884_88482

theorem area_enclosed_by_graph : 
  ∃ A : ℝ, (∀ x y : ℝ, |x| + |3 * y| = 9 ↔ (x = 9 ∨ x = -9 ∨ y = 3 ∨ y = -3)) → A = 54 :=
by
  sorry

end area_enclosed_by_graph_l884_88482


namespace no_positive_integer_solutions_l884_88423

theorem no_positive_integer_solutions (x n r : ℕ) (h1 : x > 1) (h2 : x > 0) (h3 : n > 0) (h4 : r > 0) :
  ¬(x^(2*n + 1) = 2^r + 1 ∨ x^(2*n + 1) = 2^r - 1) :=
sorry

end no_positive_integer_solutions_l884_88423


namespace domain_of_f_l884_88467

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3)^2 + (x - 6))

theorem domain_of_f :
  ∀ x : ℝ, x ≠ (5 + Real.sqrt 13) / 2 ∧ x ≠ (5 - Real.sqrt 13) / 2 → ∃ y : ℝ, y = f x :=
by
  sorry

end domain_of_f_l884_88467


namespace coordinates_of_point_P_in_third_quadrant_l884_88442

noncomputable def distance_from_y_axis (P : ℝ × ℝ) : ℝ := abs P.1
noncomputable def distance_from_x_axis (P : ℝ × ℝ) : ℝ := abs P.2

theorem coordinates_of_point_P_in_third_quadrant : 
  ∃ P : ℝ × ℝ, P.1 < 0 ∧ P.2 < 0 ∧ distance_from_x_axis P = 2 ∧ distance_from_y_axis P = 5 ∧ P = (-5, -2) :=
by
  sorry

end coordinates_of_point_P_in_third_quadrant_l884_88442
