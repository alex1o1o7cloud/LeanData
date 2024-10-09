import Mathlib

namespace cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l1414_141498

theorem cos_2alpha_plus_pi_div_2_eq_neg_24_div_25
  (α : ℝ) (hα1 : 0 < α) (hα2 : α < π / 2) (h_tanα : Real.tan α = 4 / 3) :
  Real.cos (2 * α + π / 2) = - 24 / 25 :=
by sorry

end cos_2alpha_plus_pi_div_2_eq_neg_24_div_25_l1414_141498


namespace find_special_two_digit_numbers_l1414_141489

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_special (A : ℕ) : Prop :=
  let sum_A := sum_digits A
  sum_A^2 = sum_digits (A^2)

theorem find_special_two_digit_numbers :
  {A : ℕ | 10 ≤ A ∧ A < 100 ∧ is_special A} = {10, 11, 12, 13, 20, 21, 22, 30, 31} :=
by 
  sorry

end find_special_two_digit_numbers_l1414_141489


namespace positive_value_of_A_l1414_141464

theorem positive_value_of_A (A : ℝ) (h : A^2 + 3^2 = 130) : A = 11 :=
sorry

end positive_value_of_A_l1414_141464


namespace ralph_total_cost_correct_l1414_141459

noncomputable def calculate_total_cost : ℝ :=
  let original_cart_cost := 54.00
  let small_issue_item_original := 20.00
  let additional_item_original := 15.00
  let small_issue_discount := 0.20
  let additional_item_discount := 0.25
  let coupon_discount := 0.10
  let sales_tax := 0.07

  -- Calculate the discounted prices
  let small_issue_discounted := small_issue_item_original * (1 - small_issue_discount)
  let additional_item_discounted := additional_item_original * (1 - additional_item_discount)

  -- Total cost before the coupon and tax
  let total_before_coupon := original_cart_cost + small_issue_discounted + additional_item_discounted

  -- Apply the coupon discount
  let total_after_coupon := total_before_coupon * (1 - coupon_discount)

  -- Apply the sales tax
  total_after_coupon * (1 + sales_tax)

-- Define the problem statement
theorem ralph_total_cost_correct : calculate_total_cost = 78.24 :=
by sorry

end ralph_total_cost_correct_l1414_141459


namespace expression_evaluation_l1414_141495

theorem expression_evaluation : 
  54 + (42 / 14) + (27 * 17) - 200 - (360 / 6) + 2^4 = 272 := by 
  sorry

end expression_evaluation_l1414_141495


namespace range_of_a_l1414_141449

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ a < -1 ∨ a > 3 :=
sorry

end range_of_a_l1414_141449


namespace smallest_possible_input_l1414_141426

def F (n : ℕ) := 9 * n + 120

theorem smallest_possible_input : ∃ n : ℕ, n > 0 ∧ F n = 129 :=
by {
  -- Here we would provide the proof steps, but we use sorry for now.
  sorry
}

end smallest_possible_input_l1414_141426


namespace determine_a_square_binomial_l1414_141479

theorem determine_a_square_binomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, (4 * x^2 + 16 * x + a) = (2 * x + b)^2) → a = 16 := 
by
  sorry

end determine_a_square_binomial_l1414_141479


namespace largest_n_employees_in_same_quarter_l1414_141439

theorem largest_n_employees_in_same_quarter (n : ℕ) (h1 : 72 % 4 = 0) (h2 : 72 / 4 = 18) : 
  n = 18 :=
sorry

end largest_n_employees_in_same_quarter_l1414_141439


namespace percent_decrease_area_square_l1414_141496

/-- 
In a configuration, two figures, an equilateral triangle and a square, are initially given. 
The equilateral triangle has an area of 27√3 square inches, and the square has an area of 27 square inches.
If the side length of the square is decreased by 10%, prove that the percent decrease in the area of the square is 19%.
-/
theorem percent_decrease_area_square 
  (triangle_area : ℝ := 27 * Real.sqrt 3)
  (square_area : ℝ := 27)
  (percentage_decrease : ℝ := 0.10) : 
  let new_square_side := Real.sqrt square_area * (1 - percentage_decrease)
  let new_square_area := new_square_side ^ 2
  let area_decrease := square_area - new_square_area
  let percent_decrease := (area_decrease / square_area) * 100
  percent_decrease = 19 := 
by
  sorry

end percent_decrease_area_square_l1414_141496


namespace num_roots_l1414_141413

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 2

theorem num_roots : ∃! x : ℝ, f x = 0 := 
sorry

end num_roots_l1414_141413


namespace quadratic_single_solution_positive_n_l1414_141493

variables (n : ℝ)

theorem quadratic_single_solution_positive_n :
  (∃ x : ℝ, 9 * x^2 + n * x + 36 = 0) ∧ (∀ x1 x2 : ℝ, 9 * x1^2 + n * x1 + 36 = 0 ∧ 9 * x2^2 + n * x2 + 36 = 0 → x1 = x2) →
  (n = 36) :=
sorry

end quadratic_single_solution_positive_n_l1414_141493


namespace count_multiples_5_or_7_but_not_both_l1414_141461

-- Definitions based on the given problem conditions
def multiples_of_five (n : Nat) : Nat :=
  (n - 1) / 5

def multiples_of_seven (n : Nat) : Nat :=
  (n - 1) / 7

def multiples_of_thirty_five (n : Nat) : Nat :=
  (n - 1) / 35

def count_multiples (n : Nat) : Nat :=
  (multiples_of_five n) + (multiples_of_seven n) - 2 * (multiples_of_thirty_five n)

-- The main statement to be proved
theorem count_multiples_5_or_7_but_not_both : count_multiples 101 = 30 :=
by
  sorry

end count_multiples_5_or_7_but_not_both_l1414_141461


namespace find_triples_l1414_141424

-- Definitions of the problem conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def satisfies_equation (a b p : ℕ) : Prop := a^p = factorial b + p

-- The main theorem statement based on the problem conditions
theorem find_triples :
  (satisfies_equation 2 2 2 ∧ is_prime 2) ∧
  (satisfies_equation 3 4 3 ∧ is_prime 3) ∧
  (∀ (a b p : ℕ), (satisfies_equation a b p ∧ is_prime p) → (a, b, p) = (2, 2, 2) ∨ (a, b, p) = (3, 4, 3)) :=
by
  -- Proof to be filled
  sorry

end find_triples_l1414_141424


namespace quadratic_inequality_solution_l1414_141406

theorem quadratic_inequality_solution (x : ℝ) : 
    (x^2 - 3*x - 4 > 0) ↔ (x < -1 ∨ x > 4) :=
sorry

end quadratic_inequality_solution_l1414_141406


namespace bakery_combinations_l1414_141451

theorem bakery_combinations (h : ∀ (a b c : ℕ), a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ count : ℕ, count = 25 := 
sorry

end bakery_combinations_l1414_141451


namespace cost_per_board_game_is_15_l1414_141477

-- Definitions of the conditions
def number_of_board_games : ℕ := 6
def bill_paid : ℕ := 100
def bill_value : ℕ := 5
def bills_received : ℕ := 2

def total_change := bills_received * bill_value
def total_cost := bill_paid - total_change
def cost_per_board_game := total_cost / number_of_board_games

-- The theorem stating that the cost of each board game is $15
theorem cost_per_board_game_is_15 : cost_per_board_game = 15 := 
by
  -- Omitted proof steps
  sorry

end cost_per_board_game_is_15_l1414_141477


namespace max_marks_l1414_141488

theorem max_marks (M : ℝ) (h1 : 0.40 * M = 200) : M = 500 := by
  sorry

end max_marks_l1414_141488


namespace contrapositive_of_zero_implication_l1414_141419

theorem contrapositive_of_zero_implication (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) → (a * b ≠ 0 → (a ≠ 0 ∧ b ≠ 0)) :=
by
  intro h
  sorry

end contrapositive_of_zero_implication_l1414_141419


namespace ms_walker_drives_24_miles_each_way_l1414_141405

theorem ms_walker_drives_24_miles_each_way
  (D : ℝ)
  (H1 : 1 / 60 * D + 1 / 40 * D = 1) :
  D = 24 := 
sorry

end ms_walker_drives_24_miles_each_way_l1414_141405


namespace value_of_x2_plus_y2_l1414_141499

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l1414_141499


namespace rectangular_prism_diagonal_inequality_l1414_141425

variable (a b c l : ℝ)

theorem rectangular_prism_diagonal_inequality (h : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := sorry

end rectangular_prism_diagonal_inequality_l1414_141425


namespace Elberta_has_23_dollars_l1414_141444

theorem Elberta_has_23_dollars :
  let granny_smith_amount := 63
  let anjou_amount := 1 / 3 * granny_smith_amount
  let elberta_amount := anjou_amount + 2
  elberta_amount = 23 := by
  sorry

end Elberta_has_23_dollars_l1414_141444


namespace min_tiles_for_square_l1414_141457

theorem min_tiles_for_square (a b : ℕ) (ha : a = 6) (hb : b = 4) (harea_tile : a * b = 24)
  (h_lcm : Nat.lcm a b = 12) : 
  let area_square := (Nat.lcm a b) * (Nat.lcm a b) 
  let num_tiles_required := area_square / (a * b)
  num_tiles_required = 6 :=
by
  sorry

end min_tiles_for_square_l1414_141457


namespace find_total_tennis_balls_l1414_141469

noncomputable def original_white_balls : ℕ := sorry
noncomputable def original_yellow_balls : ℕ := sorry
noncomputable def dispatched_yellow_balls : ℕ := original_yellow_balls + 20

theorem find_total_tennis_balls
  (white_balls_eq : original_white_balls = original_yellow_balls)
  (ratio_eq : original_white_balls / dispatched_yellow_balls = 8 / 13) :
  original_white_balls + original_yellow_balls = 64 := sorry

end find_total_tennis_balls_l1414_141469


namespace two_digit_number_l1414_141412

theorem two_digit_number (a : ℕ) (N M : ℕ) :
  (10 ≤ a) ∧ (a ≤ 99) ∧ (2 * a + 1 = N^2) ∧ (3 * a + 1 = M^2) → a = 40 :=
by
  sorry

end two_digit_number_l1414_141412


namespace sum_consecutive_integers_l1414_141408

theorem sum_consecutive_integers (S : ℕ) (hS : S = 221) :
  ∃ (k : ℕ) (hk : k ≥ 2) (n : ℕ), 
    (S = k * n + (k * (k - 1)) / 2) → k = 2 := sorry

end sum_consecutive_integers_l1414_141408


namespace carson_clawed_39_times_l1414_141416

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end carson_clawed_39_times_l1414_141416


namespace no_prime_factor_congruent_to_7_mod_8_l1414_141497

open Nat

theorem no_prime_factor_congruent_to_7_mod_8 (n : ℕ) (hn : 0 < n) : 
  ¬ (∃ p : ℕ, p.Prime ∧ p ∣ 2^n + 1 ∧ p % 8 = 7) :=
sorry

end no_prime_factor_congruent_to_7_mod_8_l1414_141497


namespace find_n_equiv_l1414_141490

theorem find_n_equiv :
  ∃ (n : ℕ), 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ (n = 3 ∨ n = 9) :=
by
  sorry

end find_n_equiv_l1414_141490


namespace john_profit_l1414_141429

-- Definitions based on given conditions
def total_newspapers := 500
def selling_price_per_newspaper : ℝ := 2
def discount_percentage : ℝ := 0.75
def percentage_sold : ℝ := 0.80

-- Derived basic definitions
def cost_price_per_newspaper := selling_price_per_newspaper * (1 - discount_percentage)
def total_cost_price := cost_price_per_newspaper * total_newspapers
def newspapers_sold := total_newspapers * percentage_sold
def revenue := selling_price_per_newspaper * newspapers_sold
def profit := revenue - total_cost_price

-- Theorem stating the profit
theorem john_profit : profit = 550 := by
  sorry

#check john_profit

end john_profit_l1414_141429


namespace find_floor_abs_S_l1414_141427

-- Conditions
-- For integers from 1 to 1500, x_1 + 2 = x_2 + 4 = x_3 + 6 = ... = x_1500 + 3000 = ∑(n=1 to 1500) x_n + 3001
def condition (x : ℕ → ℤ) (S : ℤ) : Prop :=
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 1500 →
    x a + 2 * a = S + 3001

-- Problem statement
theorem find_floor_abs_S (x : ℕ → ℤ) (S : ℤ)
  (h : condition x S) :
  (⌊|S|⌋ : ℤ) = 1500 :=
sorry

end find_floor_abs_S_l1414_141427


namespace range_of_f_t_l1414_141460

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (Real.exp x) + Real.log x - x

theorem range_of_f_t (a : ℝ) (t : ℝ) 
  (h_unique_critical : ∀ x, f a x = 0 → x = t) : 
  ∃ y : ℝ, y ≥ -2 ∧ ∀ z : ℝ, y = f a t :=
sorry

end range_of_f_t_l1414_141460


namespace sticks_per_stool_is_two_l1414_141428

-- Conditions
def sticks_from_chair := 6
def sticks_from_table := 9
def sticks_needed_per_hour := 5
def num_chairs := 18
def num_tables := 6
def num_stools := 4
def hours_to_keep_warm := 34

-- Question and Answer in Lean 4 statement
theorem sticks_per_stool_is_two : 
  (hours_to_keep_warm * sticks_needed_per_hour) - (num_chairs * sticks_from_chair + num_tables * sticks_from_table) = 2 * num_stools := 
  by
    sorry

end sticks_per_stool_is_two_l1414_141428


namespace ab_product_l1414_141465

theorem ab_product (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (2 * a + b) * (2 * b + a) = 4752) : a * b = 520 := 
by
  sorry

end ab_product_l1414_141465


namespace solve_for_x_l1414_141443

theorem solve_for_x (A B C D: Type) 
(y z w x : ℝ) 
(h_triangle : ∃ a b c : Type, True) 
(h_D_on_extension : ∃ D_on_extension : Type, True)
(h_AD_GT_BD : ∃ s : Type, True) 
(h_x_at_D : ∃ t : Type, True) 
(h_y_at_A : ∃ u : Type, True) 
(h_z_at_B : ∃ v : Type, True) 
(h_w_at_C : ∃ w : Type, True)
(h_triangle_angle_sum : y + z + w = 180):
x = 180 - z - w := by
  sorry

end solve_for_x_l1414_141443


namespace smallest_four_digit_multiple_of_13_l1414_141473

theorem smallest_four_digit_multiple_of_13 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (n % 13 = 0) ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < n → m % 13 ≠ 0 :=
by
  sorry

end smallest_four_digit_multiple_of_13_l1414_141473


namespace males_listen_l1414_141481

theorem males_listen (total_listen : ℕ) (females_listen : ℕ) (known_total_listen : total_listen = 160)
  (known_females_listen : females_listen = 75) : (total_listen - females_listen) = 85 :=
by 
  sorry

end males_listen_l1414_141481


namespace sum_of_extreme_numbers_is_846_l1414_141404

theorem sum_of_extreme_numbers_is_846 :
  let digits := [0, 2, 4, 6]
  let is_valid_hundreds_digit (d : Nat) := d ≠ 0
  let create_three_digit_number (h t u : Nat) := h * 100 + t * 10 + u
  let max_num := create_three_digit_number 6 4 2
  let min_num := create_three_digit_number 2 0 4
  max_num + min_num = 846 := by
  sorry

end sum_of_extreme_numbers_is_846_l1414_141404


namespace math_problem_l1414_141492

noncomputable def A (k : ℝ) : ℝ := k - 5
noncomputable def B (k : ℝ) : ℝ := k + 2
noncomputable def C (k : ℝ) : ℝ := k / 2
noncomputable def D (k : ℝ) : ℝ := 2 * k

theorem math_problem (k : ℝ) (h : A k + B k + C k + D k = 100) : 
  (A k) * (B k) * (C k) * (D k) =  (161 * 224 * 103 * 412) / 6561 :=
by
  sorry

end math_problem_l1414_141492


namespace find_divisor_l1414_141422

theorem find_divisor (n x : ℕ) (hx : x ≠ 11) (hn : n = 386) 
  (h1 : ∃ k : ℤ, n = k * x + 1) (h2 : ∀ m : ℤ, n = 11 * m + 1 → n = 386) : x = 5 :=
  sorry

end find_divisor_l1414_141422


namespace cake_flour_amount_l1414_141420

theorem cake_flour_amount (sugar_cups : ℕ) (flour_already_in : ℕ) (extra_flour_needed : ℕ) (total_flour : ℕ) 
  (h1 : sugar_cups = 7) 
  (h2 : flour_already_in = 2)
  (h3 : extra_flour_needed = 2)
  (h4 : total_flour = sugar_cups + extra_flour_needed) : 
  total_flour = 9 := 
sorry

end cake_flour_amount_l1414_141420


namespace arithmetic_sequence_sum_l1414_141486

variable {a : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n m, a (n + m) = a n + m * (a 1 - a 0)

theorem arithmetic_sequence_sum
  (h_arith : is_arithmetic_sequence a)
  (h_sum : a 5 + a 6 + a 7 = 15) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 35 :=
by
  sorry

end arithmetic_sequence_sum_l1414_141486


namespace contrapositive_proposition_contrapositive_equiv_l1414_141452

theorem contrapositive_proposition (x : ℝ) (h : -1 < x ∧ x < 1) : (x^2 < 1) :=
sorry

theorem contrapositive_equiv (x : ℝ) (h : x^2 ≥ 1) : x ≥ 1 ∨ x ≤ -1 :=
sorry

end contrapositive_proposition_contrapositive_equiv_l1414_141452


namespace permutation_6_2_eq_30_l1414_141448

theorem permutation_6_2_eq_30 :
  (Nat.factorial 6) / (Nat.factorial (6 - 2)) = 30 :=
by
  sorry

end permutation_6_2_eq_30_l1414_141448


namespace power_sum_l1414_141441

theorem power_sum (a b c : ℝ) (h1 : a + b + c = 1)
                  (h2 : a^2 + b^2 + c^2 = 3)
                  (h3 : a^3 + b^3 + c^3 = 4)
                  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 :=
  sorry

end power_sum_l1414_141441


namespace laundry_per_hour_l1414_141411

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end laundry_per_hour_l1414_141411


namespace intersecting_lines_sum_c_d_l1414_141484

theorem intersecting_lines_sum_c_d 
  (c d : ℚ)
  (h1 : 2 = 1 / 5 * (3 : ℚ) + c)
  (h2 : 3 = 1 / 5 * (2 : ℚ) + d) : 
  c + d = 4 :=
by sorry

end intersecting_lines_sum_c_d_l1414_141484


namespace max_k_value_l1414_141447

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def B (i : ℕ) := {b : Finset ℕ // b ⊆ A ∧ b ≠ ∅ ∧ ∀ j ≠ i, ∃ k : Finset ℕ, k ⊆ A ∧ k ≠ ∅ ∧ (b ∩ k).card ≤ 2}

theorem max_k_value : ∃ k, k = 175 :=
  by
    sorry

end max_k_value_l1414_141447


namespace bill_age_l1414_141430

theorem bill_age (C : ℕ) (h1 : ∀ B : ℕ, B = 2 * C - 1) (h2 : C + (2 * C - 1) = 26) : 
  ∃ B : ℕ, B = 17 := 
by
  sorry

end bill_age_l1414_141430


namespace range_g_l1414_141463

noncomputable def g (x : ℝ) : ℝ := x / (x^2 - 2 * x + 2)

theorem range_g : Set.Icc (-(1:ℝ)/2) (1/2) = {y : ℝ | ∃ x : ℝ, g x = y} := 
by
  sorry

end range_g_l1414_141463


namespace exists_three_distinct_div_l1414_141437

theorem exists_three_distinct_div (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ m : ℕ, ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ abc ∣ (x * y * z) ∧ m ≤ x ∧ x < m + 2*c ∧ m ≤ y ∧ y < m + 2*c ∧ m ≤ z ∧ z < m + 2*c :=
by
  sorry

end exists_three_distinct_div_l1414_141437


namespace inequality_proof_l1414_141436

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxyz : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) +
  (y^5 - y^2) / (y^5 + z^2 + x^2) +
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := 
sorry

end inequality_proof_l1414_141436


namespace min_value_proof_l1414_141454

noncomputable def min_value (α γ : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2

theorem min_value_proof (α γ : ℝ) : ∃ α γ : ℝ, min_value α γ = 36 :=
by
  use (Real.arcsin 12/13), (Real.pi/2 - Real.arcsin 12/13)
  sorry

end min_value_proof_l1414_141454


namespace container_capacity_l1414_141403

theorem container_capacity
  (C : ℝ)  -- Total capacity of the container in liters
  (h1 : C / 2 + 20 = 3 * C / 4)  -- Condition combining the water added and the fractional capacities
  : C = 80 := 
sorry

end container_capacity_l1414_141403


namespace square_roots_sum_eq_zero_l1414_141485

theorem square_roots_sum_eq_zero (x y : ℝ) (h1 : x^2 = 2011) (h2 : y^2 = 2011) : x + y = 0 :=
by sorry

end square_roots_sum_eq_zero_l1414_141485


namespace inequality_1_minimum_value_l1414_141468

-- Definition for part (1)
theorem inequality_1 (a b m n : ℝ) (hm : m > 0) (hn : n > 0) : 
  (a^2 / m + b^2 / n) ≥ ((a + b)^2 / (m + n)) :=
sorry

-- Definition for part (2)
theorem minimum_value (x : ℝ) (hx : 0 < x) (hx' : x < 1) : 
  (∃ (y : ℝ), y = (1 / x + 4 / (1 - x)) ∧ y = 9) :=
sorry

end inequality_1_minimum_value_l1414_141468


namespace white_area_correct_l1414_141414

/-- The dimensions of the sign and the letter components -/
def sign_width : ℕ := 18
def sign_height : ℕ := 6
def vertical_bar_height : ℕ := 6
def vertical_bar_width : ℕ := 1
def horizontal_bar_length : ℕ := 4
def horizontal_bar_width : ℕ := 1

/-- The areas of the components of each letter -/
def area_C : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)
def area_O : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + 2 * (horizontal_bar_length * horizontal_bar_width)
def area_L : ℕ := (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)

/-- The total area of the sign -/
def total_sign_area : ℕ := sign_height * sign_width

/-- The total black area covered by the letters "COOL" -/
def total_black_area : ℕ := area_C + 2 * area_O + area_L

/-- The area of the white portion of the sign -/
def white_area : ℕ := total_sign_area - total_black_area

/-- Proof that the area of the white portion of the sign is 42 square units -/
theorem white_area_correct : white_area = 42 := by
  -- Calculation steps (skipped, though the result is expected to be 42)
  sorry

end white_area_correct_l1414_141414


namespace not_odd_not_even_min_value_3_l1414_141440

def f (x : ℝ) : ℝ := x^2 + abs (x - 2) - 1

-- Statement 1: Prove that the function is neither odd nor even.
theorem not_odd_not_even : 
  ¬(∀ x, f (-x) = -f x) ∧ ¬(∀ x, f (-x) = f x) :=
sorry

-- Statement 2: Prove that the minimum value of the function is 3.
theorem min_value_3 : ∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≥ 3 :=
sorry

end not_odd_not_even_min_value_3_l1414_141440


namespace triangle_expression_simplification_l1414_141453

variable (a b c : ℝ)

theorem triangle_expression_simplification (h1 : a + b > c) 
                                           (h2 : a + c > b) 
                                           (h3 : b + c > a) :
  |a - b - c| + |b - a - c| - |c - a + b| = a - b + c :=
sorry

end triangle_expression_simplification_l1414_141453


namespace monotonic_increasing_interval_l1414_141487

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 - Real.log x

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 0 < x → (1 / 2 < x → (f (x + 0.1) > f x)) :=
by
  intro x hx h
  sorry

end monotonic_increasing_interval_l1414_141487


namespace determine_b_l1414_141458

theorem determine_b (b : ℤ) : (x - 5) ∣ (x^3 + 3 * x^2 + b * x + 5) → b = -41 :=
by
  sorry

end determine_b_l1414_141458


namespace min_triangular_faces_l1414_141431

theorem min_triangular_faces (l c e m n k : ℕ) (h1 : l > c) (h2 : l + c = e + 2) (h3 : l = c + k) (h4 : e ≥ (3 * m + 4 * n) / 2) :
  m ≥ 6 := sorry

end min_triangular_faces_l1414_141431


namespace eq_m_neg_one_l1414_141418

theorem eq_m_neg_one (m : ℝ) (x : ℝ) (h1 : (m-1) * x^(m^2 + 1) + 2*x - 3 = 0) (h2 : m - 1 ≠ 0) (h3 : m^2 + 1 = 2) : 
  m = -1 :=
sorry

end eq_m_neg_one_l1414_141418


namespace circle_center_and_radius_l1414_141442

noncomputable def circle_eq : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 + 2 * x - 2 * y - 2 = 0) ↔ (x + 1)^2 + (y - 1)^2 = 4

theorem circle_center_and_radius :
  ∃ center : ℝ × ℝ, ∃ r : ℝ, 
  center = (-1, 1) ∧ r = 2 ∧ circle_eq :=
by
  sorry

end circle_center_and_radius_l1414_141442


namespace toms_investment_l1414_141478

theorem toms_investment 
  (P : ℝ)
  (rA : ℝ := 0.06)
  (nA : ℝ := 1)
  (tA : ℕ := 4)
  (rB : ℝ := 0.08)
  (nB : ℕ := 2)
  (tB : ℕ := 4)
  (delta : ℝ := 100)
  (A_A := P * (1 + rA / nA) ^ (nA * tA))
  (A_B := P * (1 + rB / nB) ^ (nB * tB))
  (h : A_B - A_A = delta) : 
  P = 942.59 := by
sorry

end toms_investment_l1414_141478


namespace seven_y_minus_x_eq_three_l1414_141483

-- Definitions for the conditions
variables (x y : ℤ)
variables (hx : x > 0)
variables (h1 : x = 11 * y + 4)
variables (h2 : 2 * x = 18 * y + 1)

-- The theorem we want to prove
theorem seven_y_minus_x_eq_three : 7 * y - x = 3 :=
by
  -- Placeholder for the proof.
  sorry

end seven_y_minus_x_eq_three_l1414_141483


namespace picked_balls_correct_l1414_141450

-- Conditions
def initial_balls := 6
def final_balls := 24

-- The task is to find the number of picked balls
def picked_balls : Nat := final_balls - initial_balls

-- The proof goal
theorem picked_balls_correct : picked_balls = 18 :=
by
  -- We declare, but the proof is not required
  sorry

end picked_balls_correct_l1414_141450


namespace find_length_AB_l1414_141475

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line y = x - 1
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection length |AB|
noncomputable def length_AB (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

-- Main theorem statement
theorem find_length_AB (x1 x2 : ℝ)
  (h₁ : parabola x1 (x1 - 1))
  (h₂ : parabola x2 (x2 - 1))
  (hx : x1 + x2 = 6) :
  length_AB x1 x2 = 8 := sorry

end find_length_AB_l1414_141475


namespace less_than_reciprocal_l1414_141494

theorem less_than_reciprocal (a b c d e : ℝ) (ha : a = -3) (hb : b = -1/2) (hc : c = 0.5) (hd : d = 1) (he : e = 3) :
  (a < 1 / a) ∧ (c < 1 / c) ∧ ¬(b < 1 / b) ∧ ¬(d < 1 / d) ∧ ¬(e < 1 / e) :=
by
  sorry

end less_than_reciprocal_l1414_141494


namespace june_found_total_eggs_l1414_141462

def eggs_in_tree_1 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_tree_2 (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest
def eggs_in_yard (nests : ℕ) (eggs_per_nest : ℕ) : ℕ := nests * eggs_per_nest

def total_eggs (eggs_tree_1 : ℕ) (eggs_tree_2 : ℕ) (eggs_yard : ℕ) : ℕ :=
eggs_tree_1 + eggs_tree_2 + eggs_yard

theorem june_found_total_eggs :
  total_eggs (eggs_in_tree_1 2 5) (eggs_in_tree_2 1 3) (eggs_in_yard 1 4) = 17 :=
by
  sorry

end june_found_total_eggs_l1414_141462


namespace remainder_of_power_mod_l1414_141446

theorem remainder_of_power_mod :
  ∀ (x n m : ℕ), 
  x = 5 → n = 2021 → m = 17 →
  x^n % m = 11 := by
sorry

end remainder_of_power_mod_l1414_141446


namespace monthly_pool_cost_is_correct_l1414_141480

def cost_of_cleaning : ℕ := 150
def tip_percentage : ℕ := 10
def number_of_cleanings_in_a_month : ℕ := 30 / 3
def cost_of_chemicals_per_use : ℕ := 200
def number_of_chemical_uses_in_a_month : ℕ := 2

def monthly_cost_of_pool : ℕ :=
  let cost_per_cleaning := cost_of_cleaning + (cost_of_cleaning * tip_percentage / 100)
  let total_cleaning_cost := number_of_cleanings_in_a_month * cost_per_cleaning
  let total_chemical_cost := number_of_chemical_uses_in_a_month * cost_of_chemicals_per_use
  total_cleaning_cost + total_chemical_cost

theorem monthly_pool_cost_is_correct : monthly_cost_of_pool = 2050 :=
by
  sorry

end monthly_pool_cost_is_correct_l1414_141480


namespace units_digit_G_2000_l1414_141423

-- Define the sequence G
def G (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 5 ^ (5 ^ n)

-- The main goal is to show that the units digit of G 2000 is 1
theorem units_digit_G_2000 : (G 2000) % 10 = 1 :=
by
  sorry

end units_digit_G_2000_l1414_141423


namespace exists_duplicate_parenthesizations_l1414_141434

def expr : List Int := List.range' 1 (1991 + 1)

def num_parenthesizations : Nat := 2 ^ 995

def num_distinct_results : Nat := 3966067

theorem exists_duplicate_parenthesizations :
  num_parenthesizations > num_distinct_results :=
sorry

end exists_duplicate_parenthesizations_l1414_141434


namespace basketball_volleyball_problem_l1414_141472

-- Define variables and conditions
variables (x y : ℕ) (m : ℕ)

-- Conditions
def price_conditions : Prop :=
  2 * x + 3 * y = 190 ∧ 3 * x = 5 * y

def price_solutions : Prop :=
  x = 50 ∧ y = 30

def purchase_conditions : Prop :=
  8 ≤ m ∧ m ≤ 10 ∧ 50 * m + 30 * (20 - m) ≤ 800

-- The most cost-effective plan
def cost_efficient_plan : Prop :=
  m = 8 ∧ (20 - m) = 12

-- Conjecture for the problem
theorem basketball_volleyball_problem :
  price_conditions x y ∧ purchase_conditions m →
  price_solutions x y ∧ cost_efficient_plan m :=
by {
  sorry
}

end basketball_volleyball_problem_l1414_141472


namespace minimum_possible_sum_of_4x4x4_cube_l1414_141471

theorem minimum_possible_sum_of_4x4x4_cube: 
  (∀ die: ℕ, (1 ≤ die) ∧ (die ≤ 6) ∧ (∃ opposite, die + opposite = 7)) → 
  (∃ sum, sum = 304) :=
by
  sorry

end minimum_possible_sum_of_4x4x4_cube_l1414_141471


namespace final_purchase_price_correct_l1414_141410

-- Definitions
def initial_house_value : ℝ := 100000
def profit_percentage_Mr_Brown : ℝ := 0.10
def renovation_percentage : ℝ := 0.05
def profit_percentage_Mr_Green : ℝ := 0.07
def loss_percentage_Mr_Brown : ℝ := 0.10

-- Calculations
def purchase_price_mr_brown : ℝ := initial_house_value * (1 + profit_percentage_Mr_Brown)
def total_cost_mr_brown : ℝ := purchase_price_mr_brown * (1 + renovation_percentage)
def purchase_price_mr_green : ℝ := total_cost_mr_brown * (1 + profit_percentage_Mr_Green)
def final_purchase_price_mr_brown : ℝ := purchase_price_mr_green * (1 - loss_percentage_Mr_Brown)

-- Statement to prove
theorem final_purchase_price_correct : 
  final_purchase_price_mr_brown = 111226.50 :=
by
  sorry -- Proof is omitted

end final_purchase_price_correct_l1414_141410


namespace solve_for_x_l1414_141482

theorem solve_for_x (x y z : ℕ) 
  (h1 : 3^x * 4^y / 2^z = 59049)
  (h2 : x - y + 2 * z = 10) : 
  x = 10 :=
sorry

end solve_for_x_l1414_141482


namespace no_solution_for_a_l1414_141470

theorem no_solution_for_a {a : ℝ} :
  (a ∈ Set.Iic (-32) ∪ Set.Ici 0) →
  ¬ ∃ x : ℝ,  9 * |x - 4 * a| + |x - a^2| + 8 * x - 4 * a = 0 :=
by
  intro h
  sorry

end no_solution_for_a_l1414_141470


namespace area_of_polygon_ABHFGD_l1414_141476

noncomputable def total_area_ABHFGD : ℝ :=
  let side_ABCD := 3
  let side_EFGD := 5
  let area_ABCD := side_ABCD * side_ABCD
  let area_EFGD := side_EFGD * side_EFGD
  let area_DBH := 0.5 * 3 * (3 / 2 : ℝ) -- Area of triangle DBH
  let area_DFH := 0.5 * 5 * (5 / 2 : ℝ) -- Area of triangle DFH
  area_ABCD + area_EFGD - (area_DBH + area_DFH)

theorem area_of_polygon_ABHFGD : total_area_ABHFGD = 25.5 := by
  sorry

end area_of_polygon_ABHFGD_l1414_141476


namespace problem1_problem2_l1414_141409

-- Problem 1
theorem problem1 (a : ℝ) : 3 * a ^ 2 - 2 * a + 1 + (3 * a - a ^ 2 + 2) = 2 * a ^ 2 + a + 3 :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : x - 2 * (x - 3 / 2 * y) + 3 * (x - x * y) = 2 * x + 3 * y - 3 * x * y :=
by
  sorry

end problem1_problem2_l1414_141409


namespace probability_part_not_scrap_l1414_141438

noncomputable def probability_not_scrap : Prop :=
  let p_scrap_first := 0.01
  let p_scrap_second := 0.02
  let p_not_scrap_first := 1 - p_scrap_first
  let p_not_scrap_second := 1 - p_scrap_second
  let p_not_scrap := p_not_scrap_first * p_not_scrap_second
  p_not_scrap = 0.9702

theorem probability_part_not_scrap : probability_not_scrap :=
by simp [probability_not_scrap] ; sorry

end probability_part_not_scrap_l1414_141438


namespace temperature_at_midnight_is_minus4_l1414_141456

-- Definitions of initial temperature and changes
def initial_temperature : ℤ := -2
def temperature_rise_noon : ℤ := 6
def temperature_drop_midnight : ℤ := 8

-- Temperature at midnight
def temperature_midnight : ℤ :=
  initial_temperature + temperature_rise_noon - temperature_drop_midnight

theorem temperature_at_midnight_is_minus4 :
  temperature_midnight = -4 := by
  sorry

end temperature_at_midnight_is_minus4_l1414_141456


namespace path_traveled_by_A_l1414_141402

-- Define the initial conditions
def RectangleABCD (A B C D : ℝ × ℝ) :=
  dist A B = 3 ∧ dist C D = 3 ∧ dist B C = 5 ∧ dist D A = 5

-- Define the transformations
def rotated90Clockwise (D : ℝ × ℝ) (A : ℝ × ℝ) (A' : ℝ × ℝ) : Prop :=
  -- 90-degree clockwise rotation moves point A to A'
  A' = (D.1 + D.2 - A.2, D.2 - D.1 + A.1)

def translated3AlongDC (D C A' : ℝ × ℝ) (A'' : ℝ × ℝ) : Prop :=
  -- Translation by 3 units along line DC moves point A' to A''
  A'' = (A'.1 - 3, A'.2)

-- Define the total path traveled
noncomputable def totalPathTraveled (rotatedPath translatedPath : ℝ) : ℝ :=
  rotatedPath + translatedPath

-- Prove the total path is 2.5*pi + 3
theorem path_traveled_by_A (A B C D A' A'' : ℝ × ℝ) (hRect : RectangleABCD A B C D) (hRotate : rotated90Clockwise D A A') (hTranslate : translated3AlongDC D C A' A'') :
  totalPathTraveled (2.5 * Real.pi) 3 = (2.5 * Real.pi + 3) := by
  sorry

end path_traveled_by_A_l1414_141402


namespace find_m_l1414_141417

theorem find_m (m : ℕ) (h : 8 ^ 36 * 6 ^ 21 = 3 * 24 ^ m) : m = 43 :=
sorry

end find_m_l1414_141417


namespace has_only_one_minimum_point_and_no_maximum_point_l1414_141467

noncomputable def f (x : ℝ) : ℝ := 3 * x^4 - 4 * x^3

theorem has_only_one_minimum_point_and_no_maximum_point :
  ∃! c : ℝ, (deriv f c = 0 ∧ ∀ x < c, deriv f x < 0 ∧ ∀ x > c, deriv f x > 0) ∧
  ∀ x, f x ≥ f c ∧ (∀ x, deriv f x > 0 ∨ deriv f x < 0) := sorry

end has_only_one_minimum_point_and_no_maximum_point_l1414_141467


namespace find_m_n_l1414_141407

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end find_m_n_l1414_141407


namespace eval_expression_l1414_141400

theorem eval_expression :
  let a := 3
  let b := 2
  (2 ^ a ∣ 200) ∧ ¬(2 ^ (a + 1) ∣ 200) ∧ (5 ^ b ∣ 200) ∧ ¬(5 ^ (b + 1) ∣ 200)
→ (1 / 3)^(b - a) = 3 :=
by sorry

end eval_expression_l1414_141400


namespace lisa_pizza_l1414_141435

theorem lisa_pizza (P H S : ℕ) 
  (h1 : H = 2 * P) 
  (h2 : S = P + 12) 
  (h3 : P + H + S = 132) : 
  P = 30 := 
by
  sorry

end lisa_pizza_l1414_141435


namespace complement_B_range_a_l1414_141445

open Set

variable (A B : Set ℝ) (a : ℝ)

def mySetA : Set ℝ := {x | 2 * a - 2 < x ∧ x < a}
def mySetB : Set ℝ := {x | 3 / (x - 1) ≥ 1}

theorem complement_B_range_a (h : mySetA a ⊆ compl mySetB) : 
  compl mySetB = {x | x ≤ 1} ∪ {x | x > 4} ∧ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end complement_B_range_a_l1414_141445


namespace acute_angle_inequality_l1414_141433

theorem acute_angle_inequality (a b : ℝ) (α β : ℝ) (γ : ℝ) (h : γ < π / 2) :
  (a^2 + b^2) * Real.cos (α - β) ≤ 2 * a * b :=
sorry

end acute_angle_inequality_l1414_141433


namespace inequality_solution_range_l1414_141474

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, 2 * x - 6 + m < 0 ∧ 4 * x - m > 0) → m < 4 :=
by
  intro h
  sorry

end inequality_solution_range_l1414_141474


namespace part1_part2_l1414_141415

def f (x a : ℝ) := |x - a| + x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  {x : ℝ | f x a ≥ x + 2} = {x | x ≥ 3} ∪ {x | x ≤ -1} := 
by
  sorry

theorem part2 (a : ℝ) (h : {x : ℝ | f x a ≤ 3 * x} = {x | x ≥ 2}) : 
  a = 6 := 
by
  sorry

end part1_part2_l1414_141415


namespace ducks_remaining_after_three_nights_l1414_141421

def initial_ducks : ℕ := 320
def first_night_ducks_eaten (ducks : ℕ) : ℕ := ducks * 1 / 4
def after_first_night (ducks : ℕ) : ℕ := ducks - first_night_ducks_eaten ducks
def second_night_ducks_fly_away (ducks : ℕ) : ℕ := ducks * 1 / 6
def after_second_night (ducks : ℕ) : ℕ := ducks - second_night_ducks_fly_away ducks
def third_night_ducks_stolen (ducks : ℕ) : ℕ := ducks * 30 / 100
def after_third_night (ducks : ℕ) : ℕ := ducks - third_night_ducks_stolen ducks

theorem ducks_remaining_after_three_nights : after_third_night (after_second_night (after_first_night initial_ducks)) = 140 :=
by 
  -- replace the following sorry with the actual proof steps
  sorry

end ducks_remaining_after_three_nights_l1414_141421


namespace car_speed_first_hour_l1414_141401

theorem car_speed_first_hour (x : ℕ) (h1 : 60 > 0) (h2 : 40 > 0) (h3 : 2 > 0) (avg_speed : 40 = (x + 60) / 2) : x = 20 := 
by
  sorry

end car_speed_first_hour_l1414_141401


namespace find_p_if_geometric_exists_p_arithmetic_sequence_l1414_141466

variable (a : ℕ → ℝ) (p : ℝ)

-- Condition 1: a_1 = 1
axiom a1_eq_1 : a 1 = 1

-- Condition 2: a_n + a_{n+1} = pn + 1
axiom a_recurrence : ∀ n : ℕ, a n + a (n + 1) = p * n + 1

-- Question 1: If a_1, a_2, a_4 form a geometric sequence, find p
theorem find_p_if_geometric (h_geometric : (a 2)^2 = (a 1) * (a 4)) : p = 2 := by
  -- Proof goes here
  sorry

-- Question 2: Does there exist a p such that the sequence {a_n} is an arithmetic sequence?
theorem exists_p_arithmetic_sequence : ∃ p : ℝ, (∀ n : ℕ, a n + a (n + 1) = p * n + 1) ∧ 
                                         (∀ m n : ℕ, a (m + n) - a n = m * p) := by
  -- Proof goes here
  exists 2
  sorry

end find_p_if_geometric_exists_p_arithmetic_sequence_l1414_141466


namespace set_difference_equals_six_l1414_141432

-- Set Operations definitions used
def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

-- Define sets M and N
def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 3, 6}

-- Problem statement to prove
theorem set_difference_equals_six : set_difference N M = {6} :=
  sorry

end set_difference_equals_six_l1414_141432


namespace number_of_students_l1414_141455

-- Define John's total winnings
def john_total_winnings : ℤ := 155250

-- Define the proportion of winnings given to each student
def proportion_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received_by_students : ℚ := 15525

-- Calculate the amount each student received
def amount_per_student : ℚ := john_total_winnings * proportion_per_student

-- Theorem to prove the number of students
theorem number_of_students : total_received_by_students / amount_per_student = 100 :=
by
  -- Lean will be expected to fill in this proof
  sorry

end number_of_students_l1414_141455


namespace temperature_on_April_15_and_19_l1414_141491

/-
We define the daily temperatures as functions of the temperature on April 15 (T_15) with the given increment of 1.5 degrees each day. 
T_15 represents the temperature on April 15.
-/
theorem temperature_on_April_15_and_19 (T : ℕ → ℝ) (T_avg : ℝ) (inc : ℝ) 
  (h1 : inc = 1.5)
  (h2 : T_avg = 17.5)
  (h3 : ∀ n, T (15 + n) = T 15 + inc * n)
  (h4 : (T 15 + T 16 + T 17 + T 18 + T 19) / 5 = T_avg) :
  T 15 = 14.5 ∧ T 19 = 20.5 :=
by
  sorry

end temperature_on_April_15_and_19_l1414_141491
