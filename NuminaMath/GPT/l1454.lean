import Mathlib

namespace tan_alpha_minus_beta_value_l1454_145489

theorem tan_alpha_minus_beta_value (α β : Real) 
  (h1 : Real.sin α = 3 / 5) 
  (h2 : α ∈ Set.Ioo (π / 2) π) 
  (h3 : Real.tan (π - β) = 1 / 2) : 
  Real.tan (α - β) = -2 / 11 :=
by
  sorry

end tan_alpha_minus_beta_value_l1454_145489


namespace train_speed_l1454_145447

theorem train_speed
    (train_length : ℕ := 800)
    (tunnel_length : ℕ := 500)
    (time_minutes : ℕ := 1)
    : (train_length + tunnel_length) * (60 / time_minutes) / 1000 = 78 := by
  sorry

end train_speed_l1454_145447


namespace solve_floor_eq_l1454_145477

theorem solve_floor_eq (x : ℝ) (hx_pos : 0 < x) (h : (⌊x⌋ : ℝ) * x = 110) : x = 11 := 
sorry

end solve_floor_eq_l1454_145477


namespace third_number_in_sequence_l1454_145441

theorem third_number_in_sequence (n : ℕ) (h_sum : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 63) : n + 2 = 8 :=
by
  -- the proof would be written here
  sorry

end third_number_in_sequence_l1454_145441


namespace ken_ride_time_l1454_145478

variables (x y k t : ℝ)

-- Condition 1: It takes Ken 80 seconds to walk down an escalator when it is not moving.
def condition1 : Prop := 80 * x = y

-- Condition 2: It takes Ken 40 seconds to walk down an escalator when it is moving with a 10-second delay.
def condition2 : Prop := 50 * (x + k) = y

-- Condition 3: There is a 10-second delay before the escalator starts moving.
def condition3 : Prop := t = y / k + 10

-- Related Speed
def condition4 : Prop := k = 0.6 * x

-- Proposition: The time Ken takes to ride the escalator down without walking, including the delay, is 143 seconds.
theorem ken_ride_time {x y k t : ℝ} (h1 : condition1 x y) (h2 : condition2 x y k) (h3 : condition3 y k t) (h4 : condition4 x k) :
  t = 143 :=
by sorry

end ken_ride_time_l1454_145478


namespace problem_solution_l1454_145468

noncomputable def problem_expr : ℝ :=
  (64 + 5 * 12) / (180 / 3) + Real.sqrt 49 - 2^3 * Nat.factorial 4

theorem problem_solution : problem_expr = -182.93333333 :=
by 
  sorry

end problem_solution_l1454_145468


namespace milk_price_same_after_reductions_l1454_145438

theorem milk_price_same_after_reductions (x : ℝ) (h1 : 0 < x) :
  (x - 0.4 * x) = ((x - 0.2 * x) - 0.25 * (x - 0.2 * x)) :=
by
  sorry

end milk_price_same_after_reductions_l1454_145438


namespace find_difference_l1454_145454

noncomputable def expr (a b : ℝ) : ℝ :=
  |a - b| / (|a| + |b|)

def min_val (a b : ℝ) : ℝ := 0

def max_val (a b : ℝ) : ℝ := 1

theorem find_difference (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) :
  max_val a b - min_val a b = 1 :=
by
  sorry

end find_difference_l1454_145454


namespace problem1_solution_problem2_solution_l1454_145467

def f (c a b : ℝ) (x : ℝ) : ℝ := |(c * x + a)| + |(c * x - b)|
def g (c : ℝ) (x : ℝ) : ℝ := |(x - 2)| + c

noncomputable def sol_set_eq1 := {x : ℝ | -1 / 2 ≤ x ∧ x ≤ 3 / 2}
noncomputable def range_a_eq2 := {a : ℝ | a ≤ -2 ∨ a ≥ 0}

-- Problem (1)
theorem problem1_solution : ∀ (x : ℝ), f 2 1 3 x - 4 = 0 ↔ x ∈ sol_set_eq1 := 
by
  intro x
  sorry -- Proof to be filled in

-- Problem (2)
theorem problem2_solution : 
  ∀ x_1 : ℝ, ∃ x_2 : ℝ, g 1 x_2 = f 1 0 1 x_1 ↔ a ∈ range_a_eq2 :=
by
  intro x_1
  sorry -- Proof to be filled in

end problem1_solution_problem2_solution_l1454_145467


namespace final_concentration_of_milk_l1454_145421

variable (x : ℝ) (total_vol : ℝ) (initial_milk : ℝ)
axiom x_value : x = 33.333333333333336
axiom total_volume : total_vol = 100
axiom initial_milk_vol : initial_milk = 36

theorem final_concentration_of_milk :
  let first_removal := x / total_vol * initial_milk
  let remaining_milk_after_first := initial_milk - first_removal
  let second_removal := x / total_vol * remaining_milk_after_first
  let final_milk := remaining_milk_after_first - second_removal
  (final_milk / total_vol) * 100 = 16 :=
by {
  sorry
}

end final_concentration_of_milk_l1454_145421


namespace second_root_l1454_145445

variables {a b c x : ℝ}

theorem second_root (h : a * (b + c) * x ^ 2 - b * (c + a) * x + c * (a + b) = 0)
(hroot : a * (b + c) * (-1) ^ 2 - b * (c + a) * (-1) + c * (a + b) = 0) :
  ∃ k : ℝ, k = - c * (a + b) / (a * (b + c)) ∧ a * (b + c) * k ^ 2 - b * (c + a) * k + c * (a + b) = 0 :=
sorry

end second_root_l1454_145445


namespace relation_xy_l1454_145452

theorem relation_xy (a c b d : ℝ) (x y p : ℝ) 
  (h1 : a^x = c^(3 * p))
  (h2 : c^(3 * p) = b^2)
  (h3 : c^y = b^p)
  (h4 : b^p = d^3) :
  y = 3 * p^2 / 2 :=
by
  sorry

end relation_xy_l1454_145452


namespace ellipse_equation_l1454_145455

theorem ellipse_equation 
  (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m ≠ n)
  (h4 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → dist A B = 2 * (2:ℝ).sqrt)
  (h5 : ∀ A B : ℝ × ℝ, (m * A.1^2 + n * A.2^2 = 1) ∧ (m * B.1^2 + n * B.2^2 = 1) ∧ (A.1 + A.2 = 1) ∧ (B.1 + B.2 = 1) → 
    (A.2 + B.2) / (A.1 + B.1) = (2:ℝ).sqrt / 2) :
  m = 1 / 3 → n = (2:ℝ).sqrt / 3 → 
  (∀ x y : ℝ, (1 / 3) * x^2 + ((2:ℝ).sqrt / 3) * y^2 = 1) :=
by
  sorry

end ellipse_equation_l1454_145455


namespace max_m_l1454_145409

theorem max_m : ∃ m A B : ℤ, (AB = 90 ∧ m = 5 * B + A) ∧ (∀ m' A' B', (A' * B' = 90 ∧ m' = 5 * B' + A') → m' ≤ 451) ∧ m = 451 :=
by
  sorry

end max_m_l1454_145409


namespace Peter_total_distance_l1454_145404

theorem Peter_total_distance 
  (total_time : ℝ) 
  (speed1 speed2 fraction1 fraction2 : ℝ) 
  (h_time : total_time = 1.4) 
  (h_speed1 : speed1 = 4) 
  (h_speed2 : speed2 = 5) 
  (h_fraction1 : fraction1 = 2/3) 
  (h_fraction2 : fraction2 = 1/3) 
  (D : ℝ) : 
  (fraction1 * D / speed1 + fraction2 * D / speed2 = total_time) → D = 6 :=
by
  intros h_eq
  sorry

end Peter_total_distance_l1454_145404


namespace total_cats_received_l1454_145420

-- Defining the constants and conditions
def total_adult_cats := 150
def fraction_female_cats := 2 / 3
def fraction_litters := 2 / 5
def kittens_per_litter := 5

-- Defining the proof problem
theorem total_cats_received :
  let number_female_cats := (fraction_female_cats * total_adult_cats : ℤ)
  let number_litters := (fraction_litters * number_female_cats : ℤ)
  let number_kittens := number_litters * kittens_per_litter
  number_female_cats + number_kittens + (total_adult_cats - number_female_cats) = 350 := 
by
  sorry

end total_cats_received_l1454_145420


namespace bill_score_l1454_145472

variable {J B S : ℕ}

theorem bill_score (h1 : B = J + 20) (h2 : B = S / 2) (h3 : J + B + S = 160) : B = 45 :=
sorry

end bill_score_l1454_145472


namespace find_x2_plus_y2_l1454_145481

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := 
sorry

end find_x2_plus_y2_l1454_145481


namespace vip_seat_cost_l1454_145491

theorem vip_seat_cost
  (V : ℝ)
  (G V_T : ℕ)
  (h1 : 20 * G + V * V_T = 7500)
  (h2 : G + V_T = 320)
  (h3 : V_T = G - 276) :
  V = 70 := by
sorry

end vip_seat_cost_l1454_145491


namespace ratio_first_part_l1454_145460

theorem ratio_first_part (x : ℝ) (h1 : 180 / 100 * 5 = x) : x = 9 :=
by sorry

end ratio_first_part_l1454_145460


namespace fraction_given_to_emma_is_7_over_36_l1454_145461

-- Define initial quantities
def stickers_noah : ℕ := sorry
def stickers_emma : ℕ := 3 * stickers_noah
def stickers_liam : ℕ := 12 * stickers_noah

-- Define required number of stickers for equal distribution
def total_stickers := stickers_noah + stickers_emma + stickers_liam
def equal_stickers := total_stickers / 3

-- Define the number of stickers to be given to Emma and the fraction of Liam's stickers he should give to Emma
def stickers_given_to_emma := equal_stickers - stickers_emma
def fraction_liams_stickers_given_to_emma := stickers_given_to_emma / stickers_liam

-- Theorem statement
theorem fraction_given_to_emma_is_7_over_36 :
  fraction_liams_stickers_given_to_emma = 7 / 36 :=
sorry

end fraction_given_to_emma_is_7_over_36_l1454_145461


namespace banana_cost_is_2_l1454_145482

noncomputable def bananas_cost (B : ℝ) : Prop :=
  let cost_oranges : ℝ := 10 * 1.5
  let total_cost : ℝ := 25
  let cost_bananas : ℝ := total_cost - cost_oranges
  let num_bananas : ℝ := 5
  B = cost_bananas / num_bananas

theorem banana_cost_is_2 : bananas_cost 2 :=
by
  unfold bananas_cost
  sorry

end banana_cost_is_2_l1454_145482


namespace probability_at_most_one_incorrect_l1454_145435

variable (p : ℝ)

theorem probability_at_most_one_incorrect (h : 0 ≤ p ∧ p ≤ 1) :
  p^9 * (10 - 9*p) = p^10 + 10 * (1 - p) * p^9 := by
  sorry

end probability_at_most_one_incorrect_l1454_145435


namespace undefined_count_expression_l1454_145453

theorem undefined_count_expression : 
  let expr (x : ℝ) := (x^2 - 16) / ((x^2 + 3*x - 10) * (x - 4))
  ∃ u v w : ℝ, (u = 2 ∨ v = -5 ∨ w = 4) ∧
  (u ≠ v ∧ u ≠ w ∧ v ≠ w) :=
by
  sorry

end undefined_count_expression_l1454_145453


namespace down_payment_l1454_145428

theorem down_payment {total_loan : ℕ} {monthly_payment : ℕ} {years : ℕ} (h1 : total_loan = 46000) (h2 : monthly_payment = 600) (h3 : years = 5):
  total_loan - (years * 12 * monthly_payment) = 10000 := by
  sorry

end down_payment_l1454_145428


namespace part1_zero_of_f_a_neg1_part2_range_of_a_l1454_145403

noncomputable def f (a x : ℝ) := a * x^2 + 2 * x - 2 - a

theorem part1_zero_of_f_a_neg1 : 
  f (-1) 1 = 0 :=
by 
  sorry

theorem part2_range_of_a (a : ℝ) :
  a ≤ 0 →
  (∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a x = 0) ∧ (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x = 0 → x = 1) ↔ 
  (-1 ≤ a ∧ a ≤ 0) ∨ (a ≤ -2) :=
by 
  sorry

end part1_zero_of_f_a_neg1_part2_range_of_a_l1454_145403


namespace squares_equal_l1454_145490

theorem squares_equal (a b c : ℤ) 
  (h : (a - 5)^2 + (b - 12)^2 - (c - 13)^2 = a^2 + b^2 - c^2) 
    : ∃ (k : ℤ), a^2 + b^2 - c^2 = k^2 := 
by 
  sorry

end squares_equal_l1454_145490


namespace bobby_initial_pieces_l1454_145483

-- Definitions based on the conditions
def pieces_eaten_1 := 17
def pieces_eaten_2 := 15
def pieces_left := 4

-- Definition based on the question and answer
def initial_pieces (pieces_eaten_1 pieces_eaten_2 pieces_left : ℕ) : ℕ :=
  pieces_eaten_1 + pieces_eaten_2 + pieces_left

-- Theorem stating the problem and the expected answer
theorem bobby_initial_pieces : 
  initial_pieces pieces_eaten_1 pieces_eaten_2 pieces_left = 36 :=
by 
  sorry

end bobby_initial_pieces_l1454_145483


namespace gcd_102_238_l1454_145476

theorem gcd_102_238 : Nat.gcd 102 238 = 34 :=
by
  sorry

end gcd_102_238_l1454_145476


namespace original_acid_percentage_zero_l1454_145458

theorem original_acid_percentage_zero (a w : ℝ) 
  (h1 : (a + 1) / (a + w + 1) = 1 / 4) 
  (h2 : (a + 2) / (a + w + 2) = 2 / 5) : 
  a / (a + w) = 0 := 
by
  sorry

end original_acid_percentage_zero_l1454_145458


namespace number_of_days_worked_l1454_145442

-- Define the conditions
def hours_per_day := 8
def total_hours := 32

-- Define the proof statement
theorem number_of_days_worked : total_hours / hours_per_day = 4 :=
by
  -- Placeholder for the proof
  sorry

end number_of_days_worked_l1454_145442


namespace number_of_balloons_Allan_bought_l1454_145448

theorem number_of_balloons_Allan_bought 
  (initial_balloons final_balloons : ℕ) 
  (h1 : initial_balloons = 5) 
  (h2 : final_balloons = 8) : 
  final_balloons - initial_balloons = 3 := 
  by 
  sorry

end number_of_balloons_Allan_bought_l1454_145448


namespace greatest_three_digit_multiple_of_17_l1454_145414

theorem greatest_three_digit_multiple_of_17 : ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ n) :=
sorry

end greatest_three_digit_multiple_of_17_l1454_145414


namespace Hannah_total_spent_l1454_145427

def rides_cost (total_money : ℝ) : ℝ :=
  0.35 * total_money

def games_cost (total_money : ℝ) : ℝ :=
  0.25 * total_money

def food_and_souvenirs_cost : ℝ :=
  7 + 4 + 5 + 6

def total_spent (total_money : ℝ) : ℝ :=
  rides_cost total_money + games_cost total_money + food_and_souvenirs_cost

theorem Hannah_total_spent (total_money : ℝ) (h : total_money = 80) :
  total_spent total_money = 70 :=
by
  rw [total_spent, h, rides_cost, games_cost]
  norm_num
  sorry

end Hannah_total_spent_l1454_145427


namespace sin_squared_equiv_cosine_l1454_145465

theorem sin_squared_equiv_cosine :
  1 - 2 * (Real.sin (Real.pi / 8))^2 = (Real.sqrt 2) / 2 :=
by sorry

end sin_squared_equiv_cosine_l1454_145465


namespace find_a_l1454_145473

theorem find_a (a : ℝ) : (∃ p : ℝ × ℝ, p = (2 - a, a - 3) ∧ p.fst = 0) → a = 2 := by
  sorry

end find_a_l1454_145473


namespace factor_tree_value_l1454_145486

theorem factor_tree_value :
  let Q := 5 * 3
  let R := 11 * 2
  let Y := 2 * Q
  let Z := 7 * R
  let X := Y * Z
  X = 4620 :=
by
  sorry

end factor_tree_value_l1454_145486


namespace train_speed_approximation_l1454_145426

theorem train_speed_approximation (train_speed_mph : ℝ) (seconds : ℝ) :
  (40 : ℝ) * train_speed_mph * 1 / 60 = seconds → seconds = 27 := 
  sorry

end train_speed_approximation_l1454_145426


namespace rug_overlap_area_l1454_145419

theorem rug_overlap_area (A S S2 S3 : ℝ) 
  (hA : A = 200)
  (hS : S = 138)
  (hS2 : S2 = 24)
  (h1 : ∃ (S1 : ℝ), S1 + S2 + S3 = S)
  (h2 : ∃ (S1 : ℝ), S1 + 2 * S2 + 3 * S3 = A) : S3 = 19 :=
by
  sorry

end rug_overlap_area_l1454_145419


namespace problem1_1_problem1_2_problem2_l1454_145499

open Set

/-
Given sets U, A, and B, derived from the provided conditions:
  U : Set ℝ
  A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
  B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}
-/

def U : Set ℝ := univ
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | x < 2 * m - 3}

theorem problem1_1 (m : ℝ) (h : m = 5) : A ∩ B m = {x | -3 ≤ x ∧ x ≤ 5} :=
sorry

theorem problem1_2 (m : ℝ) (h : m = 5) : (compl A) ∪ B m = univ :=
sorry

theorem problem2 (m : ℝ) : A ⊆ B m → 4 < m :=
sorry

end problem1_1_problem1_2_problem2_l1454_145499


namespace sufficient_and_necessary_condition_for_positive_sum_l1454_145459

variable (q : ℤ) (a1 : ℤ)

def geometric_sequence (n : ℕ) : ℤ := a1 * q ^ (n - 1)

def sum_of_first_n_terms (n : ℕ) : ℤ :=
  if q = 1 then a1 * n else (a1 * (1 - q ^ n)) / (1 - q)

theorem sufficient_and_necessary_condition_for_positive_sum :
  (a1 > 0) ↔ (sum_of_first_n_terms q a1 2017 > 0) :=
sorry

end sufficient_and_necessary_condition_for_positive_sum_l1454_145459


namespace greatest_number_of_rented_trucks_l1454_145433

-- Define the conditions
def total_trucks_on_monday : ℕ := 24
def trucks_returned_percentage : ℕ := 50
def trucks_on_lot_saturday (R : ℕ) (P : ℕ) : ℕ := (R * P) / 100
def min_trucks_on_lot_saturday : ℕ := 12

-- Define the theorem
theorem greatest_number_of_rented_trucks : ∃ R, R = total_trucks_on_monday ∧ trucks_returned_percentage = 50 ∧ min_trucks_on_lot_saturday = 12 → R = 24 :=
by
  sorry

end greatest_number_of_rented_trucks_l1454_145433


namespace distance_from_circle_center_to_line_l1454_145425

-- Define the center of the circle
def circle_center : ℝ × ℝ := (1, -2)

-- Define the equation of the line
def line_eq (x y : ℝ) : ℝ := 2 * x + y - 5

-- Define the distance function from a point to a line
noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  |a * p.1 + b * p.2 + c| / Real.sqrt (a ^ 2 + b ^ 2)

-- Define the actual proof problem
theorem distance_from_circle_center_to_line : 
  distance_to_line circle_center 2 1 (-5) = Real.sqrt 5 :=
by
  sorry

end distance_from_circle_center_to_line_l1454_145425


namespace none_of_the_choices_sum_of_150_consecutive_integers_l1454_145434

theorem none_of_the_choices_sum_of_150_consecutive_integers :
  ¬(∃ k : ℕ, 678900 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1136850 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1000000 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 2251200 = 150 * k + 11325) ∧
  ¬(∃ k : ℕ, 1876800 = 150 * k + 11325) :=
by
  sorry

end none_of_the_choices_sum_of_150_consecutive_integers_l1454_145434


namespace find_m_l1454_145498

theorem find_m (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 2) : m = Real.sqrt 10 :=
by
  sorry

end find_m_l1454_145498


namespace distinct_positive_integer_triplets_l1454_145418

theorem distinct_positive_integer_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) (hprod : a * b * c = 72^3) : 
  ∃ n, n = 1482 :=
by
  sorry

end distinct_positive_integer_triplets_l1454_145418


namespace arithmetic_seq_problem_l1454_145437

variable {a : Nat → ℝ}  -- a_n represents the value at index n
variable {d : ℝ} -- The common difference in the arithmetic sequence

-- Define the general term of the arithmetic sequence
def arithmeticSeq (a : Nat → ℝ) (a1 : ℝ) (d : ℝ) : Prop :=
  ∀ n : Nat, a n = a1 + n * d

-- The main proof problem
theorem arithmetic_seq_problem
  (a1 : ℝ)
  (d : ℝ)
  (a : Nat → ℝ)
  (h_arithmetic: arithmeticSeq a a1 d)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) : 
  a 7 - (1 / 2) * a 8 = 8 := 
  by
  sorry

end arithmetic_seq_problem_l1454_145437


namespace largest_fraction_l1454_145493

theorem largest_fraction (p q r s : ℕ) (hp : 0 < p) (hpq : p < q) (hqr : q < r) (hrs : r < s) : 
  max (max (max (max (↑(p + q) / ↑(r + s)) (↑(p + s) / ↑(q + r))) 
              (↑(q + r) / ↑(p + s))) 
          (↑(q + s) / ↑(p + r))) 
      (↑(r + s) / ↑(p + q)) = (↑(r + s) / ↑(p + q)) :=
sorry

end largest_fraction_l1454_145493


namespace factorization_eq_l1454_145466

theorem factorization_eq :
  ∀ (a : ℝ), a^2 + 4 * a - 21 = (a - 3) * (a + 7) := by
  intro a
  sorry

end factorization_eq_l1454_145466


namespace max_log_sum_l1454_145495

noncomputable def log (x : ℝ) : ℝ := Real.log x

theorem max_log_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4) : 
  ∃ L, (∀ x y, x > 0 → y > 0 → x + y = 4 → log x + log y ≤ L) ∧ L = log 4 :=
by
  sorry

end max_log_sum_l1454_145495


namespace double_transmission_yellow_twice_double_transmission_less_single_l1454_145430

variables {α : ℝ} (hα : 0 < α ∧ α < 1)

-- Statement B
theorem double_transmission_yellow_twice (hα : 0 < α ∧ α < 1) :
  probability_displays_yellow_twice = α^2 :=
sorry

-- Statement D
theorem double_transmission_less_single (hα : 0 < α ∧ α < 1) :
  (1 - α)^2 < (1 - α) :=
sorry

end double_transmission_yellow_twice_double_transmission_less_single_l1454_145430


namespace grape_ratio_new_new_cans_from_grape_l1454_145487

-- Definitions derived from the problem conditions
def apple_ratio_initial : ℚ := 1 / 6
def grape_ratio_initial : ℚ := 1 / 10
def apple_ratio_new : ℚ := 1 / 5

-- Prove the new grape_ratio
theorem grape_ratio_new : ℚ :=
  let total_volume_per_can := apple_ratio_initial + grape_ratio_initial
  let grape_ratio_new_reciprocal := (total_volume_per_can - apple_ratio_new)
  1 / grape_ratio_new_reciprocal

-- Required final quantity of cans
theorem new_cans_from_grape : 
  (1 / grape_ratio_new) = 15 :=
sorry

end grape_ratio_new_new_cans_from_grape_l1454_145487


namespace part1_part2_l1454_145471

theorem part1 (x : ℝ) : -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 :=
by sorry

theorem part2 (a x : ℝ) (h : 0 < a) :
  (a * x^2 + (a + 3) * x + 3 > 0 ↔
    (
      (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
      (a = 3 ∧ x ≠ -1) ∨
      (a > 3 ∧ (x < -1 ∨ x > -3/a))
    )
  ) :=
by sorry

end part1_part2_l1454_145471


namespace solve_y_l1454_145440

theorem solve_y (x y : ℤ) (h₁ : x = 3) (h₂ : x^3 - x - 2 = y + 2) : y = 20 :=
by
  -- Proof goes here
  sorry

end solve_y_l1454_145440


namespace percentage_decrease_l1454_145464

noncomputable def original_fraction (N D : ℝ) : Prop := N / D = 0.75
noncomputable def new_fraction (N D x : ℝ) : Prop := (1.15 * N) / (D * (1 - x / 100)) = 15 / 16

theorem percentage_decrease (N D x : ℝ) (h1 : original_fraction N D) (h2 : new_fraction N D x) : 
  x = 22.67 := 
sorry

end percentage_decrease_l1454_145464


namespace min_value_expression_l1454_145413

theorem min_value_expression : 
  ∃ x y : ℝ, (∀ a b : ℝ, 2 * a^2 + 3 * a * b + 4 * b^2 + 5 ≥ 5) ∧ (2 * x^2 + 3 * x * y + 4 * y^2 + 5 = 5) := 
by 
sorry

end min_value_expression_l1454_145413


namespace first_ant_arrives_first_l1454_145451

noncomputable def time_crawling (d v : ℝ) : ℝ := d / v

noncomputable def time_riding_caterpillar (d v : ℝ) : ℝ := (d / 2) / (v / 2)

noncomputable def time_riding_grasshopper (d v : ℝ) : ℝ := (d / 2) / (10 * v)

noncomputable def time_ant1 (d v : ℝ) : ℝ := time_crawling d v

noncomputable def time_ant2 (d v : ℝ) : ℝ := time_riding_caterpillar d v + time_riding_grasshopper d v

theorem first_ant_arrives_first (d v : ℝ) (h_v_pos : 0 < v): time_ant1 d v < time_ant2 d v := by
  -- provide the justification for the theorem here
  sorry

end first_ant_arrives_first_l1454_145451


namespace joan_dimes_l1454_145488

theorem joan_dimes :
  ∀ (total_dimes_jacket : ℕ) (total_money : ℝ) (value_per_dime : ℝ),
    total_dimes_jacket = 15 →
    total_money = 1.90 →
    value_per_dime = 0.10 →
    ((total_money - (total_dimes_jacket * value_per_dime)) / value_per_dime) = 4 :=
by
  intros total_dimes_jacket total_money value_per_dime h1 h2 h3
  sorry

end joan_dimes_l1454_145488


namespace ratio_AD_DC_l1454_145449

-- Definitions based on conditions
variable (A B C D : Point)
variable (AB BC AD DB : ℝ)
variable (h1 : AB = 2 * BC)
variable (h2 : AD = 3 / 5 * AB)
variable (h3 : DB = 2 / 5 * AB)

-- Lean statement for the problem
theorem ratio_AD_DC (h1 : AB = 2 * BC) (h2 : AD = 3 / 5 * AB) (h3 : DB = 2 / 5 * AB) :
  AD / (DB + BC) = 2 / 3 := 
by
  sorry

end ratio_AD_DC_l1454_145449


namespace game_C_more_likely_than_game_D_l1454_145429

-- Definitions for the probabilities
def p_heads : ℚ := 3 / 4
def p_tails : ℚ := 1 / 4

-- Game C probability
def p_game_C : ℚ := p_heads ^ 4

-- Game D probabilities for each scenario
def p_game_D_scenario1 : ℚ := (p_heads ^ 3) * (p_heads ^ 2)
def p_game_D_scenario2 : ℚ := (p_heads ^ 3) * (p_tails ^ 2)
def p_game_D_scenario3 : ℚ := (p_tails ^ 3) * (p_heads ^ 2)
def p_game_D_scenario4 : ℚ := (p_tails ^ 3) * (p_tails ^ 2)

-- Total probability for Game D
def p_game_D : ℚ :=
  p_game_D_scenario1 + p_game_D_scenario2 + p_game_D_scenario3 + p_game_D_scenario4

-- Proof statement
theorem game_C_more_likely_than_game_D : (p_game_C - p_game_D) = 11 / 256 := by
  sorry

end game_C_more_likely_than_game_D_l1454_145429


namespace problem1_problem2_problem3_l1454_145469

-- Problem (I)
theorem problem1 (x : ℝ) (hx : x > 1) : 2 * Real.log x < x - 1/x :=
sorry

-- Problem (II)
theorem problem2 (a : ℝ) : (∀ t : ℝ, t > 0 → (1 + a / t) * Real.log (1 + t) > a) → 0 < a ∧ a ≤ 2 :=
sorry

-- Problem (III)
theorem problem3 : (9/10 : ℝ)^19 < 1 / (Real.exp 2) :=
sorry

end problem1_problem2_problem3_l1454_145469


namespace max_volume_cube_max_volume_parallelepiped_l1454_145417

variables {a b c : ℝ}

-- Problem (a): Cube with the maximum volume entirely contained in the tetrahedron
theorem max_volume_cube (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ s : ℝ, s = (a * b * c) / (a * b + b * c + a * c) := sorry

-- Problem (b): Rectangular parallelepiped with the maximum volume entirely contained in the tetrahedron
theorem max_volume_parallelepiped (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ (x y z : ℝ),
  (x = a / 3 ∧ y = b / 3 ∧ z = c / 3) ∧
  (x * y * z = (a * b * c) / 27) := sorry

end max_volume_cube_max_volume_parallelepiped_l1454_145417


namespace jayden_half_of_ernesto_in_some_years_l1454_145439

theorem jayden_half_of_ernesto_in_some_years :
  ∃ x : ℕ, (4 + x = (1 : ℝ) / 2 * (11 + x)) ∧ x = 3 := by
  sorry

end jayden_half_of_ernesto_in_some_years_l1454_145439


namespace valerie_light_bulbs_deficit_l1454_145422

theorem valerie_light_bulbs_deficit :
  let small_price := 8.75
  let medium_price := 11.25
  let large_price := 15.50
  let xsmall_price := 6.10
  let budget := 120
  
  let lamp_A_cost := 2 * small_price
  let lamp_B_cost := 3 * medium_price
  let lamp_C_cost := large_price
  let lamp_D_cost := 4 * xsmall_price
  let lamp_E_cost := 2 * large_price
  let lamp_F_cost := small_price + medium_price

  let total_cost := lamp_A_cost + lamp_B_cost + lamp_C_cost + lamp_D_cost + lamp_E_cost + lamp_F_cost

  total_cost - budget = 22.15 :=
by
  sorry

end valerie_light_bulbs_deficit_l1454_145422


namespace bond_paper_cost_l1454_145412

/-!
# Bond Paper Cost Calculation

This theorem calculates the total cost to buy the required amount of each type of bond paper, given the specified conditions.
-/

def cost_of_ream (sheets_per_ream : ℤ) (cost_per_ream : ℤ) (required_sheets : ℤ) : ℤ :=
  let reams_needed := (required_sheets + sheets_per_ream - 1) / sheets_per_ream
  reams_needed * cost_per_ream

theorem bond_paper_cost :
  let total_sheets := 5000
  let required_A := 2500
  let required_B := 1500
  let remaining_sheets := total_sheets - required_A - required_B
  let cost_A := cost_of_ream 500 27 required_A
  let cost_B := cost_of_ream 400 24 required_B
  let cost_C := cost_of_ream 300 18 remaining_sheets
  cost_A + cost_B + cost_C = 303 := 
by
  sorry

end bond_paper_cost_l1454_145412


namespace proof1_proof2_l1454_145485

-- Definitions based on the conditions given in the problem description.

def f1 (x : ℝ) (a : ℝ) : ℝ := abs (3 * x - 1) + a * x + 3

theorem proof1 (x : ℝ) : (f1 x 1 ≤ 4) ↔ 0 ≤ x ∧ x ≤ 1 / 2 := 
by
  sorry

theorem proof2 (a : ℝ) : (-3 ≤ a ∧ a ≤ 3) ↔ 
  ∃ (x : ℝ), ∀ y : ℝ, f1 x a ≤ f1 y a := 
by
  sorry

end proof1_proof2_l1454_145485


namespace find_n_l1454_145424

-- Define that Amy bought and sold 15n avocados.
def bought_sold_avocados (n : ℕ) := 15 * n

-- Define the profit function.
def calculate_profit (n : ℕ) : ℤ := 
  let total_cost := 10 * n
  let total_earnings := 12 * n
  total_earnings - total_cost

theorem find_n (n : ℕ) (profit : ℤ) (h1 : profit = 100) (h2 : profit = calculate_profit n) : n = 50 := 
by 
  sorry

end find_n_l1454_145424


namespace tangent_line_equation_l1454_145496

theorem tangent_line_equation :
  (∃ l : ℝ → ℝ, 
   (∀ x, l x = (1 / (4 + 2 * Real.sqrt 3)) * x + (2 + Real.sqrt 3) / 2 ∨ 
         l x = (1 / (4 - 2 * Real.sqrt 3)) * x + (2 - Real.sqrt 3) / 2) ∧ 
   (l 1 = 2) ∧ 
   (∀ x, l x = Real.sqrt x)
  ) →
  (∀ x y, 
   (y = (1 / 4 + Real.sqrt 3) * x + (2 + Real.sqrt 3) / 2 ∨ 
    y = (1 / 4 - Real.sqrt 3) * x + (2 - Real.sqrt 3) / 2) ∨ 
   (x - (4 + 2 * Real.sqrt 3) * y + (7 + 4 * Real.sqrt 3) = 0 ∨ 
    x - (4 - 2 * Real.sqrt 3) * y + (7 - 4 * Real.sqrt 3) = 0)
) :=
sorry

end tangent_line_equation_l1454_145496


namespace find_c_l1454_145444

theorem find_c (c : ℝ) 
    (h : ∀ x, (x - 4) ∣ (c * x^3 + 16 * x^2 - 5 * c * x + 40)) : 
    c = -74 / 11 :=
by
  sorry

end find_c_l1454_145444


namespace how_many_lassis_l1454_145431

def lassis_per_mango : ℕ := 15 / 3

def lassis15mangos : ℕ := 15

theorem how_many_lassis (H : lassis_per_mango = 5) : lassis15mangos * lassis_per_mango = 75 :=
by
  rw [H]
  sorry

end how_many_lassis_l1454_145431


namespace calculate_product1_calculate_square_l1454_145406

theorem calculate_product1 : 100.2 * 99.8 = 9999.96 :=
by
  sorry

theorem calculate_square : 103^2 = 10609 :=
by
  sorry

end calculate_product1_calculate_square_l1454_145406


namespace max_value_of_P_l1454_145470

noncomputable def P (a b c : ℝ) : ℝ :=
  (2 / (a^2 + 1)) - (2 / (b^2 + 1)) + (3 / (c^2 + 1))

theorem max_value_of_P (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a * b * c + a + c = b) :
  ∃ x, x = 1 ∧ ∀ y, (y = P a b c) → y ≤ x :=
sorry

end max_value_of_P_l1454_145470


namespace ed_more_marbles_than_doug_initially_l1454_145408

noncomputable def ed_initial_marbles := 37
noncomputable def doug_marbles := 5

theorem ed_more_marbles_than_doug_initially :
  ed_initial_marbles - doug_marbles = 32 := by
  sorry

end ed_more_marbles_than_doug_initially_l1454_145408


namespace remainder_when_7x_div_9_l1454_145401

theorem remainder_when_7x_div_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 :=
sorry

end remainder_when_7x_div_9_l1454_145401


namespace triangle_inequality_l1454_145410

theorem triangle_inequality 
  (a b c : ℝ) -- lengths of the sides of the triangle
  (α β γ : ℝ) -- angles of the triangle in radians opposite to sides a, b, c
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)  -- positivity of sides
  (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π) (hγ : 0 < γ ∧ γ < π) -- positivity and range of angles
  (h_sum : α + β + γ = π) -- angle sum property of a triangle
: 
  b / Real.sin (γ + α / 3) + c / Real.sin (β + α / 3) > (2 / 3) * (a / Real.sin (α / 3)) :=
sorry

end triangle_inequality_l1454_145410


namespace cos_2theta_plus_pi_l1454_145463

-- Given condition
def tan_theta_eq_2 (θ : ℝ) : Prop := Real.tan θ = 2

-- The mathematical statement to prove
theorem cos_2theta_plus_pi (θ : ℝ) (h : tan_theta_eq_2 θ) : Real.cos (2 * θ + Real.pi) = 3 / 5 := 
sorry

end cos_2theta_plus_pi_l1454_145463


namespace max_piece_length_l1454_145416

theorem max_piece_length (a b c : ℕ) (h1 : a = 60) (h2 : b = 75) (h3 : c = 90) :
  Nat.gcd (Nat.gcd a b) c = 15 :=
by 
  sorry

end max_piece_length_l1454_145416


namespace sum_of_y_values_l1454_145492

def g (x : ℚ) : ℚ := 2 * x^2 - x + 3

theorem sum_of_y_values (y1 y2 : ℚ) (hy : g (4 * y1) = 10 ∧ g (4 * y2) = 10) :
  y1 + y2 = 1 / 16 :=
sorry

end sum_of_y_values_l1454_145492


namespace find_n_l1454_145432

-- Definitions based on conditions
variables (x n y : ℕ)
variable (h1 : x / n = 3 / 2)
variable (h2 : (7 * x + n * y) / (x - n * y) = 23)

-- Proof that n is equivalent to 1 given the conditions.
theorem find_n : n = 1 :=
sorry

end find_n_l1454_145432


namespace min_value_1_a_plus_2_b_l1454_145484

open Real

theorem min_value_1_a_plus_2_b (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (∀ a b, 0 < a → 0 < b → a + b = 1 → 3 + 2 * sqrt 2 ≤ 1 / a + 2 / b) := sorry

end min_value_1_a_plus_2_b_l1454_145484


namespace union_of_sets_l1454_145423

theorem union_of_sets (x y : ℕ) (A B : Set ℕ) (h1 : A = {x, y}) (h2 : B = {x + 1, 5}) (h3 : A ∩ B = {2}) : A ∪ B = {1, 2, 5} :=
sorry

end union_of_sets_l1454_145423


namespace john_exactly_three_green_marbles_l1454_145450

-- Definitions based on the conditions
def total_marbles : ℕ := 15
def green_marbles : ℕ := 8
def purple_marbles : ℕ := 7
def trials : ℕ := 7
def green_prob : ℚ := 8 / 15
def purple_prob : ℚ := 7 / 15
def binom_coeff : ℕ := Nat.choose 7 3 

-- Theorem Statement
theorem john_exactly_three_green_marbles :
  (binom_coeff : ℚ) * (green_prob^3 * purple_prob^4) = 8604112 / 15946875 :=
by
  sorry

end john_exactly_three_green_marbles_l1454_145450


namespace find_number_l1454_145457

theorem find_number :
  ∃ x : ℤ, 27 * (x + 143) = 9693 ∧ x = 216 :=
by 
  existsi 216
  sorry

end find_number_l1454_145457


namespace decrypt_probability_l1454_145475

theorem decrypt_probability (p1 p2 p3 : ℚ) (h1 : p1 = 1/5) (h2 : p2 = 2/5) (h3 : p3 = 1/2) : 
  1 - ((1 - p1) * (1 - p2) * (1 - p3)) = 19/25 :=
by
  sorry

end decrypt_probability_l1454_145475


namespace sum_of_six_numbers_l1454_145480

theorem sum_of_six_numbers:
  ∃ (A B C D E F : ℕ), 
    A > B ∧ B > C ∧ C > D ∧ D > E ∧ E > F ∧
    E > F ∧ C > F ∧ D > F ∧ A + B + C + D + E + F = 141 := 
sorry

end sum_of_six_numbers_l1454_145480


namespace custom_op_4_3_equals_37_l1454_145497

def custom_op (a b : ℕ) : ℕ := a^2 + a*b + b^2

theorem custom_op_4_3_equals_37 : custom_op 4 3 = 37 := by
  sorry

end custom_op_4_3_equals_37_l1454_145497


namespace simplify_expression_l1454_145402

theorem simplify_expression :
  (2021^3 - 3 * 2021^2 * 2022 + 4 * 2021 * 2022^2 - 2022^3 + 2) / (2021 * 2022) = 
  1 + (1 / 2021) :=
by
  sorry

end simplify_expression_l1454_145402


namespace solve_3x_5y_eq_7_l1454_145411

theorem solve_3x_5y_eq_7 :
  ∃ (x y k : ℤ), (3 * x + 5 * y = 7) ∧ (x = 4 + 5 * k) ∧ (y = -1 - 3 * k) :=
by 
  sorry

end solve_3x_5y_eq_7_l1454_145411


namespace bowls_per_minute_l1454_145494

def ounces_per_bowl : ℕ := 10
def gallons_of_soup : ℕ := 6
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem bowls_per_minute :
  (gallons_of_soup * ounces_per_gallon / servings_time_minutes) / ounces_per_bowl = 5 :=
by
  sorry

end bowls_per_minute_l1454_145494


namespace tom_books_read_in_may_l1454_145456

def books_read_in_june := 6
def books_read_in_july := 10
def total_books_read := 18

theorem tom_books_read_in_may : total_books_read - (books_read_in_june + books_read_in_july) = 2 :=
by sorry

end tom_books_read_in_may_l1454_145456


namespace initial_birds_l1454_145407

theorem initial_birds (B : ℕ) (h : B + 13 = 42) : B = 29 :=
sorry

end initial_birds_l1454_145407


namespace math_score_is_75_l1454_145405

def average_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := (s1 + s2 + s3 + s4) / 4
def total_of_four_subjects (s1 s2 s3 s4 : ℕ) : ℕ := s1 + s2 + s3 + s4
def average_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := (s1 + s2 + s3 + s4 + s5) / 5
def total_of_five_subjects (s1 s2 s3 s4 s5 : ℕ) : ℕ := s1 + s2 + s3 + s4 + s5

theorem math_score_is_75 (s1 s2 s3 s4 : ℕ) (h1 : average_of_four_subjects s1 s2 s3 s4 = 90)
                            (h2 : average_of_five_subjects s1 s2 s3 s4 s5 = 87) :
  s5 = 75 :=
by
  sorry

end math_score_is_75_l1454_145405


namespace judgement_only_b_correct_l1454_145479

theorem judgement_only_b_correct
  (A_expr : Int := 11 + (-14) + 19 - (-6))
  (A_computed : Int := 11 + 19 + ((-14) + (-6)))
  (A_result_incorrect : A_computed ≠ 10)
  (B_expr : ℚ := -2/3 - 1/5 + (-1/3))
  (B_computed : ℚ := (-2/3 + -1/3) + -1/5)
  (B_result_correct : B_computed = -6/5) :
  (A_computed ≠ 10 ∧ B_computed = -6/5) :=
by
  sorry

end judgement_only_b_correct_l1454_145479


namespace shaded_area_correct_l1454_145415

-- Definition of the grid dimensions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

-- Definition of the heights of the shaded regions in segments
def shaded_height (x : ℕ) : ℕ :=
if x < 4 then 2
else if x < 9 then 3
else if x < 13 then 4
else if x < 15 then 5
else 0

-- Definition for the area of the entire grid
def grid_area : ℝ := grid_width * grid_height

-- Definition for the area of the unshaded triangle
def unshaded_triangle_area : ℝ := 0.5 * grid_width * grid_height

-- Definition for the area of the shaded region
def shaded_area : ℝ := grid_area - unshaded_triangle_area

-- The theorem to be proved
theorem shaded_area_correct : shaded_area = 37.5 :=
by
  sorry

end shaded_area_correct_l1454_145415


namespace weight_of_each_bag_of_flour_l1454_145462

theorem weight_of_each_bag_of_flour
  (flour_weight_needed : ℕ)
  (cost_per_bag : ℕ)
  (salt_weight_needed : ℕ)
  (salt_cost_per_pound : ℚ)
  (promotion_cost : ℕ)
  (ticket_price : ℕ)
  (tickets_sold : ℕ)
  (total_made : ℕ)
  (profit : ℕ)
  (total_flour_cost : ℕ)
  (num_bags : ℕ)
  (weight_per_bag : ℕ)
  (calc_salt_cost : ℚ := salt_weight_needed * salt_cost_per_pound)
  (calc_total_earnings : ℕ := tickets_sold * ticket_price)
  (calc_total_cost : ℚ := calc_total_earnings - profit)
  (calc_flour_cost : ℚ := calc_total_cost - calc_salt_cost - promotion_cost)
  (calc_num_bags : ℚ := calc_flour_cost / cost_per_bag)
  (calc_weight_per_bag : ℚ := flour_weight_needed / calc_num_bags) :
  flour_weight_needed = 500 ∧
  cost_per_bag = 20 ∧
  salt_weight_needed = 10 ∧
  salt_cost_per_pound = 0.2 ∧
  promotion_cost = 1000 ∧
  ticket_price = 20 ∧
  tickets_sold = 500 ∧
  total_made = 8798 ∧
  profit = 10000 - total_made ∧
  calc_salt_cost = 2 ∧
  calc_total_earnings = 10000 ∧
  calc_total_cost = 1202 ∧
  calc_flour_cost = 200 ∧
  calc_num_bags = 10 ∧
  calc_weight_per_bag = 50 :=
by {
  sorry
}

end weight_of_each_bag_of_flour_l1454_145462


namespace alexander_first_gallery_pictures_l1454_145446

def pictures_for_new_galleries := 5 * 2
def pencils_for_new_galleries := pictures_for_new_galleries * 4
def total_exhibitions := 1 + 5
def pencils_for_signing := total_exhibitions * 2
def total_pencils := 88
def pencils_for_first_gallery := total_pencils - pencils_for_new_galleries - pencils_for_signing
def pictures_for_first_gallery := pencils_for_first_gallery / 4

theorem alexander_first_gallery_pictures : pictures_for_first_gallery = 9 :=
by
  sorry

end alexander_first_gallery_pictures_l1454_145446


namespace set_equiv_l1454_145400

-- Definition of the set A according to the conditions
def A : Set ℚ := { z : ℚ | ∃ p q : ℕ, z = p / (q : ℚ) ∧ p + q = 5 ∧ p > 0 ∧ q > 0 }

-- The target set we want to prove A is equal to
def target_set : Set ℚ := { 1/4, 2/3, 3/2, 4 }

-- The theorem to prove that both sets are equal
theorem set_equiv : A = target_set :=
by
  sorry -- Proof goes here

end set_equiv_l1454_145400


namespace trigonometric_identity_l1454_145474

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (1 - Real.sin (2 * θ)) / (2 * (Real.cos θ)^2) = 1 / 2 :=
sorry

end trigonometric_identity_l1454_145474


namespace problem_l1454_145443

theorem problem 
  {a1 a2 : ℝ}
  (h1 : 0 ≤ a1)
  (h2 : 0 ≤ a2)
  (h3 : a1 + a2 = 1) :
  ∃ (b1 b2 : ℝ), 0 ≤ b1 ∧ 0 ≤ b2 ∧ b1 + b2 = 1 ∧ ((5/4 - a1) * b1 + 3 * (5/4 - a2) * b2 > 1) :=
by
  sorry

end problem_l1454_145443


namespace number_multiplies_a_l1454_145436

theorem number_multiplies_a (a b x : ℝ) (h₀ : x * a = 8 * b) (h₁ : a ≠ 0 ∧ b ≠ 0) (h₂ : (a / 8) / (b / 7) = 1) : x = 7 :=
by
  sorry

end number_multiplies_a_l1454_145436
