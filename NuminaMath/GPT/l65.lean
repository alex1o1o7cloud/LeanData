import Mathlib

namespace system_linear_eq_sum_l65_6551

theorem system_linear_eq_sum (x y : ℝ) (h₁ : 3 * x + 2 * y = 2) (h₂ : 2 * x + 3 * y = 8) : x + y = 2 :=
sorry

end system_linear_eq_sum_l65_6551


namespace range_of_a_l65_6580

theorem range_of_a (a : ℝ) (hx : ∀ x : ℝ, (a - 1) * x > a - 1 → x < 1) : a < 1 :=
sorry

end range_of_a_l65_6580


namespace complex_modulus_square_l65_6536

open Complex

theorem complex_modulus_square (z : ℂ) (h : z^2 + abs z ^ 2 = 7 + 6 * I) : abs z ^ 2 = 85 / 14 :=
sorry

end complex_modulus_square_l65_6536


namespace negate_existential_l65_6532

theorem negate_existential (p : Prop) : (¬(∃ x : ℝ, x^2 - 2 * x + 2 ≤ 0)) ↔ ∀ x : ℝ, x^2 - 2 * x + 2 > 0 :=
by sorry

end negate_existential_l65_6532


namespace jean_jail_time_l65_6502

/-- Jean has 3 counts of arson -/
def arson_count : ℕ := 3

/-- Each arson count has a 36-month sentence -/
def arson_sentence : ℕ := 36

/-- Jean has 2 burglary charges -/
def burglary_charges : ℕ := 2

/-- Each burglary charge has an 18-month sentence -/
def burglary_sentence : ℕ := 18

/-- Jean has six times as many petty larceny charges as burglary charges -/
def petty_larceny_multiplier : ℕ := 6

/-- Each petty larceny charge is 1/3 as long as a burglary charge -/
def petty_larceny_sentence : ℕ := burglary_sentence / 3

/-- Calculate all charges in months -/
def total_charges : ℕ :=
  (arson_count * arson_sentence) +
  (burglary_charges * burglary_sentence) +
  (petty_larceny_multiplier * burglary_charges * petty_larceny_sentence)

/-- Prove the total jail time for Jean is 216 months -/
theorem jean_jail_time : total_charges = 216 := by
  sorry

end jean_jail_time_l65_6502


namespace smallest_total_marbles_l65_6534

-- Definitions based on conditions in a)
def urn_contains_marbles : Type := ℕ → ℕ
def red_marbles (u : urn_contains_marbles) := u 0
def white_marbles (u : urn_contains_marbles) := u 1
def blue_marbles (u : urn_contains_marbles) := u 2
def green_marbles (u : urn_contains_marbles) := u 3
def yellow_marbles (u : urn_contains_marbles) := u 4
def total_marbles (u : urn_contains_marbles) := u 0 + u 1 + u 2 + u 3 + u 4

-- Probabilities of selection events
def prob_event_a (u : urn_contains_marbles) := (red_marbles u).choose 5
def prob_event_b (u : urn_contains_marbles) := (white_marbles u).choose 1 * (red_marbles u).choose 4
def prob_event_c (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (red_marbles u).choose 3
def prob_event_d (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (red_marbles u).choose 2
def prob_event_e (u : urn_contains_marbles) := (white_marbles u).choose 1 * (blue_marbles u).choose 1 * (green_marbles u).choose 1 * (yellow_marbles u).choose 1 * (red_marbles u).choose 1

-- Proof that the smallest total number of marbles satisfying the conditions is 33
theorem smallest_total_marbles : ∃ u : urn_contains_marbles, 
    (prob_event_a u = prob_event_b u) ∧ 
    (prob_event_b u = prob_event_c u) ∧ 
    (prob_event_c u = prob_event_d u) ∧ 
    (prob_event_d u = prob_event_e u) ∧ 
    total_marbles u = 33 := sorry

end smallest_total_marbles_l65_6534


namespace positive_3_digit_numbers_divisible_by_13_count_l65_6596

theorem positive_3_digit_numbers_divisible_by_13_count : 
  (∃ k : ℕ, 100 ≤ 13 * k ∧ 13 * k < 1000) ∧
  (∀ n : ℕ, 100 ≤ 13 * n → 13 * n < 1000 → (n = 8 + (76 - 8) - 0)) :=
sorry

end positive_3_digit_numbers_divisible_by_13_count_l65_6596


namespace math_problem_l65_6554

theorem math_problem
  (numerator : ℕ := (Nat.factorial 10))
  (denominator : ℕ := (10 * 11 / 2)) :
  (numerator / denominator : ℚ) = 66069 + 1 / 11 := by
  sorry

end math_problem_l65_6554


namespace part_one_solution_part_two_solution_l65_6560

-- (I) Prove the solution set for the given inequality with m = 2.
theorem part_one_solution (x : ℝ) : 
  (|x - 2| > 7 - |x - 1|) ↔ (x < -4 ∨ x > 5) :=
sorry

-- (II) Prove the range of m given the condition.
theorem part_two_solution (m : ℝ) : 
  (∃ x : ℝ, |x - m| > 7 + |x - 1|) ↔ (m ∈ Set.Iio (-6) ∪ Set.Ioi (8)) :=
sorry

end part_one_solution_part_two_solution_l65_6560


namespace arithmetic_sequence_a5_value_l65_6569

theorem arithmetic_sequence_a5_value 
  (a : ℕ → ℝ) 
  (h1 : a 2 + a 4 = 16) 
  (h2 : a 1 = 1) : 
  a 5 = 15 := 
by 
  sorry

end arithmetic_sequence_a5_value_l65_6569


namespace eraser_difference_l65_6579

theorem eraser_difference
  (hanna_erasers rachel_erasers tanya_erasers tanya_red_erasers : ℕ)
  (h1 : hanna_erasers = 2 * rachel_erasers)
  (h2 : rachel_erasers = tanya_red_erasers)
  (h3 : tanya_erasers = 20)
  (h4 : tanya_red_erasers = tanya_erasers / 2)
  (h5 : hanna_erasers = 4) :
  rachel_erasers - (tanya_red_erasers / 2) = 5 :=
sorry

end eraser_difference_l65_6579


namespace largest_three_digit_base7_to_decimal_l65_6512

theorem largest_three_digit_base7_to_decimal :
  (6 * 7^2 + 6 * 7^1 + 6 * 7^0) = 342 :=
by
  sorry

end largest_three_digit_base7_to_decimal_l65_6512


namespace product_of_radii_l65_6511

-- Definitions based on the problem conditions
def passes_through (a : ℝ) (C : ℝ × ℝ) : Prop :=
  (C.1 - a)^2 + (C.2 - a)^2 = a^2

def tangent_to_axes (a : ℝ) : Prop :=
  a > 0

def circle_radii_roots (a b : ℝ) : Prop :=
  a^2 - 14 * a + 25 = 0 ∧ b^2 - 14 * b + 25 = 0

-- Theorem statement to prove the product of the radii
theorem product_of_radii (a r1 r2 : ℝ) (h1 : passes_through a (3, 4)) (h2 : tangent_to_axes a) (h3 : circle_radii_roots r1 r2) : r1 * r2 = 25 :=
by
  sorry

end product_of_radii_l65_6511


namespace max_marks_l65_6530

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 92 + 40) : M = 400 :=
by
  sorry

end max_marks_l65_6530


namespace marble_draw_probability_l65_6591

def marble_probabilities : ℚ :=
  let prob_white_a := 5 / 10
  let prob_black_a := 5 / 10
  let prob_yellow_b := 8 / 15
  let prob_yellow_c := 3 / 10
  let prob_green_d := 6 / 10
  let prob_white_then_yellow_then_green := prob_white_a * prob_yellow_b * prob_green_d
  let prob_black_then_yellow_then_green := prob_black_a * prob_yellow_c * prob_green_d
  prob_white_then_yellow_then_green + prob_black_then_yellow_then_green

theorem marble_draw_probability :
  marble_probabilities = 17 / 50 := by
  sorry

end marble_draw_probability_l65_6591


namespace simplify_expression_l65_6562

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

noncomputable def x : ℝ := (b / c) * (c / b)
noncomputable def y : ℝ := (a / c) * (c / a)
noncomputable def z : ℝ := (a / b) * (b / a)

theorem simplify_expression : x^2 + y^2 + z^2 + x^2 * y^2 * z^2 = 4 := 
by {
  sorry
}

end simplify_expression_l65_6562


namespace hamburger_price_l65_6561

theorem hamburger_price (P : ℝ) 
    (h1 : 2 * 4 + 2 * 2 = 12) 
    (h2 : 12 * P + 4 * P = 50) : 
    P = 3.125 := 
by
  -- sorry added to skip the proof.
  sorry

end hamburger_price_l65_6561


namespace regular_price_of_tire_l65_6500

theorem regular_price_of_tire (p : ℝ) (h : 2 * p + p / 2 = 270) : p = 108 :=
sorry

end regular_price_of_tire_l65_6500


namespace express_in_scientific_notation_l65_6539

-- Definitions based on problem conditions
def GDP_first_quarter : ℝ := 27017800000000

-- Main theorem statement that needs to be proved
theorem express_in_scientific_notation :
  ∃ (a : ℝ) (b : ℤ), (GDP_first_quarter = a * 10 ^ b) ∧ (a = 2.70178) ∧ (b = 13) :=
by
  sorry -- Placeholder to indicate the proof is omitted

end express_in_scientific_notation_l65_6539


namespace right_triangle_ratio_l65_6523

theorem right_triangle_ratio (a b c r s : ℝ) (h : a / b = 2 / 5)
  (h_c : c^2 = a^2 + b^2)
  (h_r : r = a^2 / c)
  (h_s : s = b^2 / c) :
  r / s = 4 / 25 := by
  sorry

end right_triangle_ratio_l65_6523


namespace minimum_value_l65_6510

theorem minimum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_condition : 2 * a + 3 * b = 1) :
  ∃ min_value : ℝ, min_value = 65 / 6 ∧ (∀ c d : ℝ, (0 < c) → (0 < d) → (2 * c + 3 * d = 1) → (1 / c + 1 / d ≥ min_value)) :=
sorry

end minimum_value_l65_6510


namespace smallest_common_multiple_gt_50_l65_6533

theorem smallest_common_multiple_gt_50 (a b : ℕ) (h1 : a = 15) (h2 : b = 20) : 
    ∃ x, x > 50 ∧ Nat.lcm a b = x := by
  have h_lcm : Nat.lcm a b = 60 := by sorry
  use 60
  exact ⟨by decide, h_lcm⟩

end smallest_common_multiple_gt_50_l65_6533


namespace problem_condition_l65_6598

variable {f : ℝ → ℝ}
variable {a b : ℝ}

noncomputable def fx_condition (f : ℝ → ℝ) :=
  ∀ x : ℝ, f x + x * (deriv f x) < 0

theorem problem_condition {f : ℝ → ℝ} {a b : ℝ} (h1 : fx_condition f) (h2 : a < b) :
  a * f a > b * f b :=
sorry

end problem_condition_l65_6598


namespace n_minus_m_eq_singleton_6_l65_6577

def set_difference (A B : Set α) : Set α :=
  {x | x ∈ A ∧ x ∉ B}

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 6}

theorem n_minus_m_eq_singleton_6 : set_difference N M = {6} :=
by
  sorry

end n_minus_m_eq_singleton_6_l65_6577


namespace last_digit_of_3_to_2010_is_9_l65_6549

theorem last_digit_of_3_to_2010_is_9 : (3^2010 % 10) = 9 := by
  -- Given that the last digits of powers of 3 cycle through 3, 9, 7, 1
  -- We need to prove that the last digit of 3^2010 is 9
  sorry

end last_digit_of_3_to_2010_is_9_l65_6549


namespace greatest_four_digit_n_l65_6595

theorem greatest_four_digit_n :
  ∃ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) ∧ (∃ m : ℕ, n + 1 = m^2) ∧ ¬(n! % (n * (n + 1) / 2) = 0) ∧ n = 9999 :=
by sorry

end greatest_four_digit_n_l65_6595


namespace alice_savings_l65_6564

variable (B : ℝ)

def savings (B : ℝ) : ℝ :=
  let first_month := 10
  let second_month := first_month + 30 + B
  let third_month := first_month + 30 + 30
  first_month + second_month + third_month

theorem alice_savings (B : ℝ) : savings B = 120 + B :=
by
  sorry

end alice_savings_l65_6564


namespace steven_more_peaches_l65_6526

variable (Jake Steven Jill : ℕ)

-- Conditions
axiom h1 : Jake + 6 = Steven
axiom h2 : Jill = 5
axiom h3 : Jake = 17

-- Goal
theorem steven_more_peaches : Steven - Jill = 18 := by
  sorry

end steven_more_peaches_l65_6526


namespace custom_operator_example_l65_6559

def custom_operator (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem custom_operator_example : custom_operator 5 3 = 4 := by
  sorry

end custom_operator_example_l65_6559


namespace profit_percentage_l65_6592

theorem profit_percentage (selling_price profit : ℝ) (h1 : selling_price = 900) (h2 : profit = 300) : 
  (profit / (selling_price - profit)) * 100 = 50 :=
by
  sorry

end profit_percentage_l65_6592


namespace root_in_interval_l65_6565

noncomputable def f (x: ℝ) : ℝ := x^2 + (Real.log x) - 4

theorem root_in_interval : 
  (∃ ξ ∈ Set.Ioo 1 2, f ξ = 0) :=
by
  sorry

end root_in_interval_l65_6565


namespace count_valid_triples_l65_6575

-- Define the necessary conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def is_positive (n : ℕ) : Prop := n > 0
def valid_triple (p q n : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_positive n ∧ (1/p + 2013/q = n/5)

-- Lean statement for the proof problem
theorem count_valid_triples : 
  ∃ c : ℕ, c = 5 ∧ 
  (∀ p q n : ℕ, valid_triple p q n → true) :=
sorry

end count_valid_triples_l65_6575


namespace necessary_but_not_sufficient_l65_6509

def p (a : ℝ) : Prop := ∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0

def q (a : ℝ) : Prop := a > 0 ∨ a < -1

theorem necessary_but_not_sufficient (a : ℝ) : (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0) → (a > 0 ∨ a < -1) ∧ ¬((a > 0 ∨ a < -1) → (∃ (x : ℝ), x^2 + 2 * a * x - a ≤ 0)) :=
by
  sorry

end necessary_but_not_sufficient_l65_6509


namespace both_players_score_same_points_l65_6508

theorem both_players_score_same_points :
  let P_A_score := 0.5 
  let P_B_score := 0.8 
  let P_A_miss := 1 - P_A_score
  let P_B_miss := 1 - P_B_score
  let P_both_miss := P_A_miss * P_B_miss
  let P_both_score := P_A_score * P_B_score
  let P_same_points := P_both_miss + P_both_score
  P_same_points = 0.5 := 
by {
  -- Actual proof should be here
  sorry
}

end both_players_score_same_points_l65_6508


namespace find_value_of_expression_l65_6522

open Real

theorem find_value_of_expression (x y z w : ℝ) (h1 : x + y + z + w = 0) (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 := by
  sorry

end find_value_of_expression_l65_6522


namespace adam_paper_tearing_l65_6525

theorem adam_paper_tearing (n : ℕ) :
  let starts_with_one_piece : ℕ := 1
  let increment_to_four : ℕ := 3
  let increment_to_ten : ℕ := 9
  let target_pieces : ℕ := 20000
  let start_modulo : ℤ := 1

  -- Modulo 3 analysis
  starts_with_one_piece % 3 = start_modulo ∧
  increment_to_four % 3 = 0 ∧ 
  increment_to_ten % 3 = 0 ∧ 
  target_pieces % 3 = 2 → 
  n % 3 = start_modulo ∧ ∀ m, m % 3 = 0 → n + m ≠ target_pieces :=
sorry

end adam_paper_tearing_l65_6525


namespace coin_tails_probability_l65_6578

theorem coin_tails_probability (p : ℝ) (h : p = 0.5) (n : ℕ) (h_n : n = 3) :
  ∃ k : ℕ, k ≤ n ∧ (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k) = 0.375 :=
by
  sorry

end coin_tails_probability_l65_6578


namespace factorize_expression_l65_6501

variable (a : ℝ) -- assuming a is a real number

theorem factorize_expression (a : ℝ) : a^2 + 3 * a = a * (a + 3) :=
by
  -- proof goes here
  sorry

end factorize_expression_l65_6501


namespace tangents_parallel_l65_6589

-- Definitions based on the conditions in part A
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_line (c : Circle) (p : ℝ × ℝ) : ℝ := sorry

def secant_intersection (c1 c2 : Circle) (A : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) := 
  sorry

-- Main theorem statement
theorem tangents_parallel 
  (c1 c2 : Circle) (A B C : ℝ × ℝ) 
  (h1 : c1.center ≠ c2.center) 
  (h2 : dist c1.center c2.center = c1.radius + c2.radius) 
  (h3 : (B, C) = secant_intersection c1 c2 A) 
  (h4 : tangent_line c1 B ≠ tangent_line c2 C) :
  tangent_line c1 B = tangent_line c2 C :=
sorry

end tangents_parallel_l65_6589


namespace triangle_sum_is_19_l65_6513

-- Defining the operation on a triangle
def triangle_op (a b c : ℕ) := a * b - c

-- Defining the vertices of the two triangles
def triangle1 := (4, 2, 3)
def triangle2 := (3, 5, 1)

-- Statement that the sum of the operation results is 19
theorem triangle_sum_is_19 :
  triangle_op (4) (2) (3) + triangle_op (3) (5) (1) = 19 :=
by
  -- Triangle 1 calculation: 4 * 2 - 3 = 8 - 3 = 5
  -- Triangle 2 calculation: 3 * 5 - 1 = 15 - 1 = 14
  -- Sum of calculations: 5 + 14 = 19
  sorry

end triangle_sum_is_19_l65_6513


namespace divide_bill_evenly_l65_6590

variable (totalBill amountPaid : ℕ)
variable (numberOfFriends : ℕ)

theorem divide_bill_evenly (h1 : totalBill = 135) (h2 : amountPaid = 45) (h3 : numberOfFriends * amountPaid = totalBill) :
  numberOfFriends = 3 := by
  sorry

end divide_bill_evenly_l65_6590


namespace age_twice_in_2_years_l65_6529

/-
Conditions:
1. The man is 24 years older than his son.
2. The present age of the son is 22 years.
3. In a certain number of years, the man's age will be twice the age of his son.
-/
def man_is_24_years_older (S M : ℕ) : Prop := M = S + 24
def present_age_son : ℕ := 22
def age_twice_condition (Y S M : ℕ) : Prop := M + Y = 2 * (S + Y)

/-
Prove that in 2 years, the man's age will be twice the age of his son.
-/
theorem age_twice_in_2_years : ∃ (Y : ℕ), 
  (man_is_24_years_older present_age_son M) → 
  (age_twice_condition Y present_age_son M) →
  Y = 2 :=
by
  sorry

end age_twice_in_2_years_l65_6529


namespace toby_friends_girls_count_l65_6593

noncomputable def percentage_of_boys : ℚ := 55 / 100
noncomputable def boys_count : ℕ := 33
noncomputable def total_friends : ℚ := boys_count / percentage_of_boys
noncomputable def percentage_of_girls : ℚ := 1 - percentage_of_boys
noncomputable def girls_count : ℚ := percentage_of_girls * total_friends

theorem toby_friends_girls_count : girls_count = 27 := by
  sorry

end toby_friends_girls_count_l65_6593


namespace last_digit_of_189_in_base_3_is_0_l65_6531

theorem last_digit_of_189_in_base_3_is_0 : 
  (189 % 3 = 0) :=
sorry

end last_digit_of_189_in_base_3_is_0_l65_6531


namespace range_of_a_part1_range_of_a_part2_l65_6599

theorem range_of_a_part1 (a : ℝ) :
  (∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) → 0 < a ∧ a < 4 :=
sorry

theorem range_of_a_part2 (a : ℝ) :
  ((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∨ (∃ x : ℝ, x^2 - x + a = 0)) ∧ ¬((∃ x : ℝ, y^2 = (a^2 - 4 * a) * x ∧ x < 0) ∧ (∃ x : ℝ, x^2 - x + a = 0)) →
  a ≤ 0 ∨ (1 / 4) < a ∧ a < 4 :=
sorry

end range_of_a_part1_range_of_a_part2_l65_6599


namespace find_q_l65_6507

def Q (x p q d : ℝ) : ℝ := x^3 + p * x^2 + q * x + d

theorem find_q (p q d : ℝ) (h: d = 3) (h1: -p / 3 = -d) (h2: -p / 3 = 1 + p + q + d) : q = -16 :=
by
  sorry

end find_q_l65_6507


namespace solve_quadratic_l65_6506

theorem solve_quadratic (h₁ : 48 * (3/4:ℚ)^2 - 74 * (3/4:ℚ) + 47 = 0) :
  ∃ x : ℚ, x ≠ 3/4 ∧ 48 * x^2 - 74 * x + 47 = 0 ∧ x = 11/12 := 
by
  sorry

end solve_quadratic_l65_6506


namespace number_of_buses_l65_6546

-- Define the conditions
def vans : ℕ := 9
def people_per_van : ℕ := 8
def people_per_bus : ℕ := 27
def total_people : ℕ := 342

-- Translate the mathematical proof problem
theorem number_of_buses : ∃ buses : ℕ, (vans * people_per_van + buses * people_per_bus = total_people) ∧ (buses = 10) :=
by
  -- calculations to prove the theorem
  sorry

end number_of_buses_l65_6546


namespace chandra_valid_pairings_l65_6520

noncomputable def valid_pairings (total_items : Nat) (invalid_pairing : Nat) : Nat :=
total_items * total_items - invalid_pairing

theorem chandra_valid_pairings : valid_pairings 5 1 = 24 := by
  sorry

end chandra_valid_pairings_l65_6520


namespace rotated_line_eq_l65_6571

theorem rotated_line_eq :
  ∀ (x y : ℝ), 
  (x - y + 4 = 0) ∨ (x - y - 4 = 0) ↔ 
  ∃ (x' y' : ℝ), (-x', -y') = (x, y) ∧ (x' - y' + 4 = 0) :=
by
  sorry

end rotated_line_eq_l65_6571


namespace average_visitors_per_day_in_month_l65_6550

theorem average_visitors_per_day_in_month (avg_visitors_sunday : ℕ) (avg_visitors_other_days : ℕ) (days_in_month : ℕ) (starts_sunday : Bool) :
  avg_visitors_sunday = 140 → avg_visitors_other_days = 80 → days_in_month = 30 → starts_sunday = true → 
  (∀ avg_visitors, avg_visitors = (4 * avg_visitors_sunday + 26 * avg_visitors_other_days) / days_in_month → avg_visitors = 88) :=
by
  intros h1 h2 h3 h4
  have total_visitors : ℕ := 4 * avg_visitors_sunday + 26 * avg_visitors_other_days
  have avg := total_visitors / days_in_month
  have visitors : ℕ := 2640
  sorry

end average_visitors_per_day_in_month_l65_6550


namespace isosceles_triangle_base_length_l65_6566

open Real

noncomputable def average_distance_sun_earth : ℝ := 1.5 * 10^8 -- in kilometers
noncomputable def base_length_given_angle_one_second (legs_length : ℝ) : ℝ := 4.848 -- in millimeters when legs are 1 kilometer

theorem isosceles_triangle_base_length 
  (vertex_angle : ℝ) (legs_length : ℝ) 
  (h1 : vertex_angle = 1 / 3600) 
  (h2 : legs_length = average_distance_sun_earth) : 
  ∃ base_length: ℝ, base_length = 727.2 := 
by 
  sorry

end isosceles_triangle_base_length_l65_6566


namespace mappings_count_A_to_B_l65_6541

open Finset

def A : Finset ℕ := {1, 2}
def B : Finset ℕ := {3, 4}

theorem mappings_count_A_to_B : (card B) ^ (card A) = 4 :=
by
  -- This line will state that the proof is skipped for now.
  sorry

end mappings_count_A_to_B_l65_6541


namespace modulus_of_complex_l65_6504

-- Some necessary imports for complex numbers and proofs in Lean
open Complex

theorem modulus_of_complex (x y : ℝ) (h : (1 + I) * x = 1 + y * I) : abs (x + y * I) = Real.sqrt 2 :=
by
  sorry

end modulus_of_complex_l65_6504


namespace largest_square_area_l65_6535

theorem largest_square_area (a b c : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (square_area_sum : a^2 + b^2 + c^2 = 450)
  (area_a : a^2 = 100) :
  c^2 = 225 :=
by
  sorry

end largest_square_area_l65_6535


namespace ratio_of_speeds_l65_6527

theorem ratio_of_speeds (v_A v_B : ℝ) (d_A d_B t : ℝ) (h1 : d_A = 100) (h2 : d_B = 50) (h3 : v_A = d_A / t) (h4 : v_B = d_B / t) : 
  v_A / v_B = 2 := 
by sorry

end ratio_of_speeds_l65_6527


namespace train_cross_bridge_time_l65_6515

-- Length of the train in meters
def train_length : ℕ := 165

-- Length of the bridge in meters
def bridge_length : ℕ := 660

-- Speed of the train in kmph
def train_speed_kmph : ℕ := 54

-- Conversion factor from kmph to m/s
def kmph_to_mps : ℚ := 5 / 18

-- Total distance to be traveled by the train to cross the bridge
def total_distance : ℕ := train_length + bridge_length

-- Speed of the train in meters per second (m/s)
def train_speed_mps : ℚ := train_speed_kmph * kmph_to_mps

-- Time taken for the train to cross the bridge (in seconds)
def time_to_cross_bridge : ℚ := total_distance / train_speed_mps

-- Prove that the time taken for the train to cross the bridge is 55 seconds
theorem train_cross_bridge_time : time_to_cross_bridge = 55 := by
  -- Proof goes here
  sorry

end train_cross_bridge_time_l65_6515


namespace log_expression_l65_6567

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression : 
  log 2 * log 50 + log 25 - log 5 * log 20 = 1 := 
by 
  sorry

end log_expression_l65_6567


namespace inequality_proof_l65_6570

theorem inequality_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_proof_l65_6570


namespace probability_even_in_5_of_7_rolls_is_21_over_128_l65_6542

noncomputable def probability_even_in_5_of_7_rolls : ℚ :=
  let n := 7
  let k := 5
  let p := (1:ℚ) / 2
  let binomial (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  (binomial n k) * (p^k) * ((1 - p)^(n - k))

theorem probability_even_in_5_of_7_rolls_is_21_over_128 :
  probability_even_in_5_of_7_rolls = 21 / 128 :=
by
  sorry

end probability_even_in_5_of_7_rolls_is_21_over_128_l65_6542


namespace binary_addition_to_decimal_l65_6581

theorem binary_addition_to_decimal : (2^8 + 2^7 + 2^6 + 2^5 + 2^4 + 2^3 + 2^2 + 2^1 + 2^0)
                                     + (2^5 + 2^4 + 2^3 + 2^2) = 571 := by
  sorry

end binary_addition_to_decimal_l65_6581


namespace cashier_total_bills_l65_6528

theorem cashier_total_bills
  (total_value : ℕ)
  (num_ten_bills : ℕ)
  (num_twenty_bills : ℕ)
  (h1 : total_value = 330)
  (h2 : num_ten_bills = 27)
  (h3 : num_twenty_bills = 3) :
  num_ten_bills + num_twenty_bills = 30 :=
by
  -- Proof goes here
  sorry

end cashier_total_bills_l65_6528


namespace car_miles_traveled_actual_miles_l65_6586

noncomputable def count_skipped_numbers (n : ℕ) : ℕ :=
  let count_digit7 (x : ℕ) : Bool := x = 7
  -- Function to count the number of occurrences of digit 7 in each place value
  let rec count (x num_skipped : ℕ) : ℕ :=
    if x = 0 then num_skipped else
    let digit := x % 10
    let new_count := if count_digit7 digit then num_skipped + 1 else num_skipped
    count (x / 10) new_count
  count n 0

theorem car_miles_traveled (odometer_reading : ℕ) : ℕ :=
  let num_skipped := count_skipped_numbers 3008
  odometer_reading - num_skipped

theorem actual_miles {odometer_reading : ℕ} (h : odometer_reading = 3008) : car_miles_traveled odometer_reading = 2194 :=
by sorry

end car_miles_traveled_actual_miles_l65_6586


namespace minimize_square_sum_l65_6574

theorem minimize_square_sum (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : 
  ∃ x y z, (x + 2 * y + 3 * z = 1) ∧ (x^2 + y^2 + z^2 ≥ 0) ∧ ((x^2 + y^2 + z^2) = 1 / 14) :=
sorry

end minimize_square_sum_l65_6574


namespace donna_additional_flyers_l65_6545

theorem donna_additional_flyers (m d a : ℕ) (h1 : m = 33) (h2 : d = 2 * m + a) (h3 : d = 71) : a = 5 :=
by
  have m_val : m = 33 := h1
  rw [m_val] at h2
  linarith [h3, h2]

end donna_additional_flyers_l65_6545


namespace solve_for_x_l65_6517

variables (x y z : ℝ)

def condition : Prop :=
  1 / (x + y) + 1 / (x - y) = z / (x - y)

theorem solve_for_x (h : condition x y z) : x = z / 2 :=
by
  sorry

end solve_for_x_l65_6517


namespace number_of_cows_l65_6576

def each_cow_milk_per_day : ℕ := 1000
def total_milk_per_week : ℕ := 364000
def days_in_week : ℕ := 7

theorem number_of_cows : 
  (total_milk_per_week = 364000) →
  (each_cow_milk_per_day = 1000) →
  (days_in_week = 7) →
  (total_milk_per_week / (each_cow_milk_per_day * days_in_week)) = 52 :=
by
  sorry

end number_of_cows_l65_6576


namespace sum_of_base_radii_l65_6553

theorem sum_of_base_radii (R : ℝ) (hR : R = 5) (a b c : ℝ) 
  (h_ratios : a = 1 ∧ b = 2 ∧ c = 3) 
  (r1 r2 r3 : ℝ) 
  (h_r1 : r1 = (a / (a + b + c)) * R)
  (h_r2 : r2 = (b / (a + b + c)) * R)
  (h_r3 : r3 = (c / (a + b + c)) * R) : 
  r1 + r2 + r3 = 5 := 
by
  subst hR
  simp [*, ←add_assoc, add_comm]
  sorry

end sum_of_base_radii_l65_6553


namespace total_cost_of_fencing_l65_6552

def diameter : ℝ := 28
def cost_per_meter : ℝ := 1.50
def pi_approx : ℝ := 3.14159

noncomputable def circumference : ℝ := pi_approx * diameter
noncomputable def total_cost : ℝ := circumference * cost_per_meter

theorem total_cost_of_fencing : total_cost = 131.94 :=
by
  sorry

end total_cost_of_fencing_l65_6552


namespace complement_of_A_in_B_l65_6548

def set_A : Set ℤ := {x | 2 * x = x^2}
def set_B : Set ℤ := {x | -x^2 + x + 2 ≥ 0}

theorem complement_of_A_in_B :
  (set_B \ set_A) = {-1, 1} :=
by
  sorry

end complement_of_A_in_B_l65_6548


namespace find_p_l65_6538

variable (p q : ℝ) (k : ℕ)

theorem find_p (h_sum : ∀ (α β : ℝ), α + β = 2) (h_prod : ∀ (α β : ℝ), α * β = k) (hk : k > 0) :
  p = -2 := by
  sorry

end find_p_l65_6538


namespace no_positive_reals_satisfy_conditions_l65_6543

theorem no_positive_reals_satisfy_conditions :
  ¬ ∃ (a b c : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  4 * (a * b + b * c + c * a) - 1 ≥ a^2 + b^2 + c^2 ∧ 
  a^2 + b^2 + c^2 ≥ 3 * (a^3 + b^3 + c^3) :=
by
  sorry

end no_positive_reals_satisfy_conditions_l65_6543


namespace percentage_students_passed_l65_6594

theorem percentage_students_passed
    (total_students : ℕ)
    (students_failed : ℕ)
    (students_passed : ℕ)
    (percentage_passed : ℕ)
    (h1 : total_students = 840)
    (h2 : students_failed = 546)
    (h3 : students_passed = total_students - students_failed)
    (h4 : percentage_passed = (students_passed * 100) / total_students) :
    percentage_passed = 35 := by
  sorry

end percentage_students_passed_l65_6594


namespace citizen_income_l65_6519

noncomputable def income (I : ℝ) : Prop :=
  let P := 0.11 * 40000
  let A := I - 40000
  P + 0.20 * A = 8000

theorem citizen_income (I : ℝ) (h : income I) : I = 58000 := 
by
  -- proof steps go here
  sorry

end citizen_income_l65_6519


namespace sequence_formula_minimum_m_l65_6503

variable (a_n : ℕ → ℕ) (S_n : ℕ → ℕ)

/-- The sequence a_n with sum of its first n terms S_n, the first term a_1 = 1, and the terms
   1, a_n, S_n forming an arithmetic sequence, satisfies a_n = 2^(n-1). -/
theorem sequence_formula (h1 : a_n 1 = 1)
    (h2 : ∀ n : ℕ, 1 + n * (a_n n - 1) = S_n n) :
    ∀ n : ℕ, a_n n = 2 ^ (n - 1) := by
  sorry

/-- T_n being the sum of the sequence {n / a_n}, if T_n < (m - 4) / 3 for all n in ℕ*, 
    then the minimum value of m is 16. -/
theorem minimum_m (T_n : ℕ → ℝ) (m : ℕ)
    (hT : ∀ n : ℕ, n > 0 → T_n n < (m - 4) / 3) :
    m ≥ 16 := by
  sorry

end sequence_formula_minimum_m_l65_6503


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l65_6582

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l65_6582


namespace gcd_in_range_l65_6563

theorem gcd_in_range :
  ∃ n, 70 ≤ n ∧ n ≤ 80 ∧ Int.gcd n 30 = 10 :=
sorry

end gcd_in_range_l65_6563


namespace find_width_of_sheet_of_paper_l65_6547

def width_of_sheet_of_paper (W : ℝ) : Prop :=
  let margin := 1.5
  let length_of_paper := 10
  let area_covered := 38.5
  let width_of_picture := W - 2 * margin
  let length_of_picture := length_of_paper - 2 * margin
  width_of_picture * length_of_picture = area_covered

theorem find_width_of_sheet_of_paper : ∃ W : ℝ, width_of_sheet_of_paper W ∧ W = 8.5 :=
by
  -- Placeholder for the actual proof
  sorry

end find_width_of_sheet_of_paper_l65_6547


namespace smallest_four_digit_multiple_of_15_l65_6583

theorem smallest_four_digit_multiple_of_15 :
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 15 = 0) ∧ (∀ m : ℕ, (1000 ≤ m ∧ m ≤ 9999) ∧ (m % 15 = 0) → n ≤ m) ∧ n = 1005 :=
sorry

end smallest_four_digit_multiple_of_15_l65_6583


namespace sum_y_coordinates_of_intersection_with_y_axis_l65_6585

-- Define the center and radius of the circle
def center : ℝ × ℝ := (-4, 5)
def radius : ℝ := 9

-- Define the equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  (x + center.1)^2 + (y - center.2)^2 = radius^2

theorem sum_y_coordinates_of_intersection_with_y_axis : 
  ∃ y1 y2 : ℝ, circle_eq 0 y1 ∧ circle_eq 0 y2 ∧ y1 + y2 = 10 :=
by
  sorry

end sum_y_coordinates_of_intersection_with_y_axis_l65_6585


namespace solution_Y_required_l65_6573

theorem solution_Y_required (V_total V_ratio_Y : ℝ) (h_total : V_total = 0.64) (h_ratio : V_ratio_Y = 3 / 8) : 
  (0.64 * (3 / 8) = 0.24) :=
by
  sorry

end solution_Y_required_l65_6573


namespace find_x_plus_y_l65_6505

theorem find_x_plus_y
  (x y : ℝ)
  (h1 : x + Real.cos y = 2010)
  (h2 : x + 2010 * Real.sin y = 2009)
  (h3 : (π / 2) ≤ y ∧ y ≤ π) :
  x + y = 2011 + π :=
sorry

end find_x_plus_y_l65_6505


namespace min_value_problem_l65_6597

noncomputable def minValueOfExpression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) : ℝ :=
  (x + 2 * y) * (y + 2 * z) * (x * z + 1)

theorem min_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  minValueOfExpression x y z hx hy hz hxyz = 16 :=
  sorry

end min_value_problem_l65_6597


namespace limit_expr_at_pi_l65_6516

theorem limit_expr_at_pi :
  (Real.exp π - Real.exp x) / (Real.sin (5*x) - Real.sin (3*x)) = 1 / 2 * Real.exp π :=
by
  sorry

end limit_expr_at_pi_l65_6516


namespace tangent_normal_lines_l65_6537

noncomputable def x (t : ℝ) : ℝ := (1 / 2) * t^2 - (1 / 4) * t^4
noncomputable def y (t : ℝ) : ℝ := (1 / 2) * t^2 + (1 / 3) * t^3
def t0 : ℝ := 0

theorem tangent_normal_lines :
  (∃ m : ℝ, ∀ t : ℝ, t = t0 → y t = m * x t) ∧
  (∃ n : ℝ, ∀ t : ℝ, t = t0 → y t = n * x t ∧ n = -1 / m) :=
sorry

end tangent_normal_lines_l65_6537


namespace largest_angle_of_quadrilateral_l65_6584

open Real

theorem largest_angle_of_quadrilateral 
  (PQ QR RS : ℝ)
  (angle_RQP angle_SRQ largest_angle : ℝ)
  (h1: PQ = QR) 
  (h2: QR = RS) 
  (h3: angle_RQP = 60)
  (h4: angle_SRQ = 100)
  (h5: largest_angle = 130) : 
  largest_angle = 130 := by
  sorry

end largest_angle_of_quadrilateral_l65_6584


namespace pencil_distribution_l65_6587

-- Formalize the problem in Lean
theorem pencil_distribution (x1 x2 x3 x4 : ℕ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 5) (hx2 : 1 ≤ x2 ∧ x2 ≤ 5) (hx3 : 1 ≤ x3 ∧ x3 ≤ 5) (hx4 : 1 ≤ x4 ∧ x4 ≤ 5) :
  x1 + x2 + x3 + x4 = 10 → 64 = 64 :=
by {
  sorry
}

end pencil_distribution_l65_6587


namespace sample_size_is_50_l65_6521

theorem sample_size_is_50 (n : ℕ) :
  (n > 0) → 
  (10 / n = 2 / (2 + 3 + 5)) → 
  n = 50 := 
by
  sorry

end sample_size_is_50_l65_6521


namespace range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l65_6588

variable {a b : ℝ}

theorem range_of_2a_plus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -10 < 2*a + b ∧ 2*a + b < 19 :=
by
  sorry

theorem range_of_a_minus_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -9 < a - b ∧ a - b < 6 :=
by
  sorry

theorem range_of_a_div_b (h1 : -6 < a) (h2 : a < 8) (h3 : 2 < b) (h4 : b < 3) : -2 < a / b ∧ a / b < 4 :=
by
  sorry

end range_of_2a_plus_b_range_of_a_minus_b_range_of_a_div_b_l65_6588


namespace min_value_xyz_l65_6514

theorem min_value_xyz (x y z : ℝ) (h : x + 2 * y + 3 * z = 1) : x^2 + y^2 + z^2 ≥ 1 / 14 := 
by
  sorry

end min_value_xyz_l65_6514


namespace total_amount_l65_6558

theorem total_amount (N50 N: ℕ) (h1: N = 90) (h2: N50 = 77) : 
  (N50 * 50 + (N - N50) * 500) = 10350 :=
by
  sorry

end total_amount_l65_6558


namespace intersection_M_N_l65_6572

-- Define sets M and N
def M := {x : ℝ | x^2 - 2*x ≤ 0}
def N := {x : ℝ | -2 < x ∧ x < 1}

-- The theorem stating the intersection of M and N equals [0, 1)
theorem intersection_M_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := 
by
  sorry

end intersection_M_N_l65_6572


namespace math_problem_l65_6540

variables {x y : ℝ}

theorem math_problem (h1 : x * y = 16) (h2 : 1 / x = 3 * (1 / y)) (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x < y) : 
  (2 * y - x) = 24 - (4 * Real.sqrt 3) / 3 :=
by sorry

end math_problem_l65_6540


namespace det_matrix_example_l65_6568

def det_2x2 (a b c d : ℤ) : ℤ := a * d - b * c

theorem det_matrix_example : det_2x2 4 5 2 3 = 2 :=
by
  sorry

end det_matrix_example_l65_6568


namespace only_prime_such_that_2p_plus_one_is_perfect_power_l65_6556

theorem only_prime_such_that_2p_plus_one_is_perfect_power :
  ∃ (p : ℕ), p ≤ 1000 ∧ Prime p ∧ ∃ (m n : ℕ), n ≥ 2 ∧ 2 * p + 1 = m^n ∧ p = 13 :=
by
  sorry

end only_prime_such_that_2p_plus_one_is_perfect_power_l65_6556


namespace inequality_problem_l65_6524

theorem inequality_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : 
  (a + 1 / (2 * b^2)) * (b + 1 / (2 * a^2)) ≥ 25 / 4 := 
sorry

end inequality_problem_l65_6524


namespace intersection_complement_eq_l65_6518

def P : Set ℝ := {1, 2, 3, 4}
def Q : Set ℝ := {3, 4, 5}
def U : Set ℝ := Set.univ  -- Universal set U is the set of all real numbers

theorem intersection_complement_eq : P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end intersection_complement_eq_l65_6518


namespace percentage_of_students_on_trip_l65_6544

variable (students : ℕ) -- Total number of students at the school
variable (students_trip_and_more_than_100 : ℕ) -- Number of students who went to the camping trip and took more than $100
variable (percent_trip_and_more_than_100 : ℚ) -- Percent of students who went to camping trip and took more than $100

-- Given Conditions
def cond1 : students_trip_and_more_than_100 = (percent_trip_and_more_than_100 * students) := 
  by
    sorry  -- This will represent the first condition: 18% of students went to a camping trip and took more than $100.

variable (percent_did_not_take_more_than_100 : ℚ) -- Percent of students who went to camping trip and did not take more than $100

-- second condition
def cond2 : percent_did_not_take_more_than_100 = 0.75 := 
  by
    sorry  -- Represent the second condition: 75% of students who went to the camping trip did not take more than $100.

-- Prove
theorem percentage_of_students_on_trip : 
  (students_trip_and_more_than_100 / (0.25 * students)) * 100 = (72 : ℚ) := 
  by
    sorry

end percentage_of_students_on_trip_l65_6544


namespace percentage_cut_third_week_l65_6555

noncomputable def initial_weight : ℝ := 300
noncomputable def first_week_percentage : ℝ := 0.30
noncomputable def second_week_percentage : ℝ := 0.30
noncomputable def final_weight : ℝ := 124.95

theorem percentage_cut_third_week :
  let remaining_after_first_week := initial_weight * (1 - first_week_percentage)
  let remaining_after_second_week := remaining_after_first_week * (1 - second_week_percentage)
  let cut_weight_third_week := remaining_after_second_week - final_weight
  let percentage_cut_third_week := (cut_weight_third_week / remaining_after_second_week) * 100
  percentage_cut_third_week = 15 :=
by
  sorry

end percentage_cut_third_week_l65_6555


namespace candidate_percentage_l65_6557

theorem candidate_percentage (P : ℕ) (total_votes : ℕ) (vote_diff : ℕ)
  (h1 : total_votes = 7000)
  (h2 : vote_diff = 2100)
  (h3 : (P * total_votes / 100) + (P * total_votes / 100) + vote_diff = total_votes) :
  P = 35 :=
by
  sorry

end candidate_percentage_l65_6557
