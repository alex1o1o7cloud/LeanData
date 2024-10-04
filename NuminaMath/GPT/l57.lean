import Mathlib

namespace smallest_base_for_100_l57_57716

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end smallest_base_for_100_l57_57716


namespace pascal_triangle_fifth_number_l57_57515

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57515


namespace sum_of_terms_l57_57177

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l57_57177


namespace luna_budget_l57_57658

variable {H F P : ℝ}

theorem luna_budget (h1: F = 0.60 * H) (h2: P = 0.10 * F) (h3: H + F + P = 249) :
  H + F = 240 :=
by
  -- The proof will be filled in here. For now, we use sorry.
  sorry

end luna_budget_l57_57658


namespace mary_thought_animals_l57_57011

-- Definitions based on conditions
def double_counted_sheep : ℕ := 7
def forgotten_pigs : ℕ := 3
def actual_animals : ℕ := 56

-- Statement to be proven
theorem mary_thought_animals (double_counted_sheep forgotten_pigs actual_animals : ℕ) :
  (actual_animals + double_counted_sheep - forgotten_pigs) = 60 := 
by 
  -- Proof goes here
  sorry

end mary_thought_animals_l57_57011


namespace find_f_ln_log_52_l57_57825

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

axiom given_condition (a : ℝ) : f a (Real.log (Real.log 5 / Real.log 2)) = 5

theorem find_f_ln_log_52 (a : ℝ) : f a (Real.log (Real.log 2 / Real.log 5)) = 3 :=
by
  -- The details of the proof are omitted
  sorry

end find_f_ln_log_52_l57_57825


namespace sqrt_of_sum_of_powers_l57_57044

theorem sqrt_of_sum_of_powers :
  sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 :=
sorry

end sqrt_of_sum_of_powers_l57_57044


namespace pascal_fifteen_four_l57_57574

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57574


namespace gcd_elements_of_B_l57_57636

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l57_57636


namespace pascal_fifth_number_l57_57486

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57486


namespace problem1_problem2_l57_57981

def setA : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | (x - m + 2) * (x - m - 2) ≤ 0}

-- Problem 1: prove that if A ∩ B = {x | 0 ≤ x ≤ 3}, then m = 2
theorem problem1 (m : ℝ) : (setA ∩ setB m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2 :=
by
  sorry

-- Problem 2: prove that if A ⊆ complement of B, then m ∈ (-∞, -3) ∪ (5, +∞)
theorem problem2 (m : ℝ) : (setA ⊆ (fun x => x ∉ setB m)) → (m < -3 ∨ m > 5) :=
by
  sorry

end problem1_problem2_l57_57981


namespace john_total_payment_in_month_l57_57164

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end john_total_payment_in_month_l57_57164


namespace point_coordinates_l57_57965

noncomputable def parametric_curve (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) := (3 * Real.cos θ, 4 * Real.sin θ)

theorem point_coordinates (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) : 
  (Real.arcsin (4 * (Real.tan θ)) = π/4) → (3 * Real.cos θ, 4 * Real.sin θ) = (12 / 5, 12 / 5) :=
by
  sorry

end point_coordinates_l57_57965


namespace more_whistles_sean_than_charles_l57_57334

def whistles_sean : ℕ := 223
def whistles_charles : ℕ := 128

theorem more_whistles_sean_than_charles : (whistles_sean - whistles_charles) = 95 :=
by
  sorry

end more_whistles_sean_than_charles_l57_57334


namespace evaluate_expression_l57_57430

theorem evaluate_expression :
  (3 + 6 + 9 : ℚ) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end evaluate_expression_l57_57430


namespace moving_circle_trajectory_is_ellipse_l57_57919

noncomputable def trajectory_of_center (x y : ℝ) : Prop :=
  let ellipse_eq := x^2 / 4 + y^2 / 3 = 1 
  ellipse_eq ∧ x ≠ -2

theorem moving_circle_trajectory_is_ellipse
  (M_1 M_2 center : ℝ × ℝ)
  (r1 r2 R : ℝ)
  (h1 : M_1 = (-1, 0))
  (h2 : M_2 = (1, 0))
  (h3 : r1 = 1)
  (h4 : r2 = 3)
  (h5 : (center.1 + 1)^2 + center.2^2 = (1 + R)^2)
  (h6 : (center.1 - 1)^2 + center.2^2 = (3 - R)^2) :
  trajectory_of_center center.1 center.2 :=
by sorry

end moving_circle_trajectory_is_ellipse_l57_57919


namespace rectangle_area_3650_l57_57994

variables (L B : ℕ)

-- Conditions given in the problem
def condition1 : Prop := L - B = 23
def condition2 : Prop := 2 * (L + B) = 246

-- Prove that the area of the rectangle is 3650 m² given the conditions
theorem rectangle_area_3650 (h1 : condition1 L B) (h2 : condition2 L B) : L * B = 3650 := by
  sorry

end rectangle_area_3650_l57_57994


namespace difference_of_interchanged_digits_l57_57026

theorem difference_of_interchanged_digits {x y : ℕ} (h : x - y = 4) :
  (10 * x + y) - (10 * y + x) = 36 :=
by sorry

end difference_of_interchanged_digits_l57_57026


namespace fraction_meaningful_l57_57034

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ∃ y, y = 1 / (x - 1) :=
by
  sorry

end fraction_meaningful_l57_57034


namespace count_possible_integer_values_l57_57880

theorem count_possible_integer_values :
  ∃ n : ℕ, (∀ x : ℤ, (25 < x ∧ x < 55) ↔ (26 ≤ x ∧ x ≤ 54)) ∧ n = 29 := by
  sorry

end count_possible_integer_values_l57_57880


namespace ella_max_book_price_l57_57427

/--
Given that Ella needs to buy 20 identical books and her total budget, 
after deducting the $5 entry fee, is $195. Each book has the same 
cost in whole dollars, and an 8% sales tax is applied to the price of each book. 
Prove that the highest possible price per book that Ella can afford is $9.
-/
theorem ella_max_book_price : 
  ∀ (n : ℕ) (B T : ℝ), n = 20 → B = 195 → T = 1.08 → 
  ∃ (p : ℕ), (↑p ≤ B / T / n) → (9 ≤ p) := 
by 
  sorry

end ella_max_book_price_l57_57427


namespace evaluate_expr_l57_57392

-- Define the imaginary unit i
def i := Complex.I

-- Define the expressions for the proof
def expr1 := (1 + 2 * i) * i ^ 3
def expr2 := 2 * i ^ 2

-- The main statement we need to prove
theorem evaluate_expr : expr1 + expr2 = -i :=
by 
  sorry

end evaluate_expr_l57_57392


namespace correct_average_is_26_l57_57202

noncomputable def initial_average : ℕ := 20
noncomputable def number_of_numbers : ℕ := 10
noncomputable def incorrect_number : ℕ := 26
noncomputable def correct_number : ℕ := 86
noncomputable def incorrect_total_sum : ℕ := initial_average * number_of_numbers
noncomputable def correct_total_sum : ℕ := incorrect_total_sum + (correct_number - incorrect_number)
noncomputable def correct_average : ℕ := correct_total_sum / number_of_numbers

theorem correct_average_is_26 :
  correct_average = 26 := by
  sorry

end correct_average_is_26_l57_57202


namespace length_of_room_l57_57211

theorem length_of_room {L : ℝ} (width : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) 
  (h1 : width = 4)
  (h2 : cost_per_sqm = 750)
  (h3 : total_cost = 16500) :
  L = 5.5 ↔ (L * width) * cost_per_sqm = total_cost := 
by
  sorry

end length_of_room_l57_57211


namespace pascal_triangle_15_4_l57_57509

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57509


namespace fifth_number_in_pascal_row_l57_57611

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57611


namespace pascal_fifth_number_in_row_15_l57_57550

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57550


namespace pascal_triangle_fifth_number_l57_57606

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57606


namespace smallest_base_to_express_100_with_three_digits_l57_57719

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end smallest_base_to_express_100_with_three_digits_l57_57719


namespace tangent_k_value_one_common_point_range_l57_57451

namespace Geometry

-- Definitions:
def line (k : ℝ) : ℝ → ℝ := λ x => k * x - 3 * k + 2
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 4
def is_tangent (k : ℝ) : Prop := |-2 * k + 3| / (Real.sqrt (k^2 + 1)) = 2
def has_only_one_common_point (k : ℝ) : Prop :=
  (1 / 2 < k ∧ k <= 5 / 2) ∨ (k = 5 / 12)

-- Theorem statements:
theorem tangent_k_value : ∀ k : ℝ, is_tangent k → k = 5 / 12 := sorry

theorem one_common_point_range : ∀ k : ℝ, has_only_one_common_point k → k ∈
  Set.union (Set.Ioc (1 / 2) (5 / 2)) {5 / 12} := sorry

end Geometry

end tangent_k_value_one_common_point_range_l57_57451


namespace pascal_triangle_fifth_number_l57_57610

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57610


namespace find_common_ratio_l57_57446

noncomputable def a_n (n : ℕ) (q : ℚ) : ℚ :=
  if n = 1 then 1 / 8 else (q^(n - 1)) * (1 / 8)

theorem find_common_ratio (q : ℚ) :
  (a_n 4 q = -1) ↔ (q = -2) :=
by
  sorry

end find_common_ratio_l57_57446


namespace pascal_fifth_element_15th_row_l57_57599

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57599


namespace largest_divisor_of_n_pow4_minus_n_for_composites_l57_57793

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end largest_divisor_of_n_pow4_minus_n_for_composites_l57_57793


namespace area_of_triangle_ABC_l57_57145

theorem area_of_triangle_ABC 
  (ABCD_is_trapezoid : ∀ {a b c d : ℝ}, a + d = b + c)
  (area_ABCD : ∀ {a b : ℝ}, a * b = 24)
  (CD_three_times_AB : ∀ {a : ℝ}, a * 3 = 24) :
  ∃ (area_ABC : ℝ), area_ABC = 6 :=
by 
  sorry

end area_of_triangle_ABC_l57_57145


namespace pascal_triangle_fifth_number_l57_57581

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57581


namespace Pascal_triangle_fifth_number_l57_57565

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57565


namespace gcd_of_B_is_2_l57_57627

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l57_57627


namespace reciprocal_neg_one_over_2023_l57_57884

theorem reciprocal_neg_one_over_2023 : 1 / (- (1 / 2023 : ℝ)) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_l57_57884


namespace ivan_expected_shots_l57_57158

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l57_57158


namespace pascal_triangle_fifth_number_l57_57496

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57496


namespace problem_exist_formula_and_monotonic_intervals_l57_57302

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 6 * x + 1

theorem problem_exist_formula_and_monotonic_intervals :
  (∀ a b : ℝ, f(1) = a * (1:ℝ)^3 + b * (1:ℝ) + 1 → ∃ a b : ℝ, (a = 2 ∧ b = -6)) ∧
  (∀ x : ℝ, (∀ y : ℝ, f y = 2 * y^3 - 6 * y + 1) → deriv f x = 6 * x^2 - 6 →  (∀ x : ℝ, x < -1 ∨ x > 1 → deriv f x > 0)) :=
by
  sorry

end problem_exist_formula_and_monotonic_intervals_l57_57302


namespace abc_plus_2_gt_a_plus_b_plus_c_l57_57816

theorem abc_plus_2_gt_a_plus_b_plus_c (a b c : ℝ) (h1 : |a| < 1) (h2 : |b| < 1) (h3 : |c| < 1) : abc + 2 > a + b + c :=
by
  sorry

end abc_plus_2_gt_a_plus_b_plus_c_l57_57816


namespace largest_divisor_composite_difference_l57_57798

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_composite_difference_l57_57798


namespace pascal_15_5th_number_l57_57521

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57521


namespace rain_probability_tel_aviv_l57_57348

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l57_57348


namespace pascal_triangle_fifth_number_l57_57602

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57602


namespace notebooks_last_days_l57_57320

theorem notebooks_last_days (n p u : Nat) (total_pages days : Nat) 
  (h1 : n = 5)
  (h2 : p = 40)
  (h3 : u = 4)
  (h_total : total_pages = n * p)
  (h_days  : days = total_pages / u) :
  days = 50 := 
by
  sorry

end notebooks_last_days_l57_57320


namespace distinct_products_count_l57_57986

theorem distinct_products_count : 
  let s := {2, 3, 5, 7, 11}
  in (finset.powerset s).filter (λ t, 2 ≤ t.card).image (λ t, t.prod).card = 26 :=
by
  sorry

end distinct_products_count_l57_57986


namespace rain_probability_tel_aviv_l57_57346

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l57_57346


namespace oscar_leap_longer_than_elmer_stride_l57_57275

theorem oscar_leap_longer_than_elmer_stride :
  ∀ (elmer_strides_per_gap oscar_leaps_per_gap gaps_between_poles : ℕ)
    (total_distance : ℝ),
  elmer_strides_per_gap = 60 →
  oscar_leaps_per_gap = 16 →
  gaps_between_poles = 60 →
  total_distance = 7920 →
  let elmer_stride_length := total_distance / (elmer_strides_per_gap * gaps_between_poles)
  let oscar_leap_length := total_distance / (oscar_leaps_per_gap * gaps_between_poles)
  oscar_leap_length - elmer_stride_length = 6.05 :=
by
  intros
  sorry

end oscar_leap_longer_than_elmer_stride_l57_57275


namespace oreo_solution_l57_57001

noncomputable def oreo_problem : Prop :=
∃ (m : ℤ), (11 + m * 11 + 3 = 36) → m = 2

theorem oreo_solution : oreo_problem :=
sorry

end oreo_solution_l57_57001


namespace team_a_games_played_l57_57023

theorem team_a_games_played (a b: ℕ) (hA_wins : 3 * a = 4 * wins_A)
(hB_wins : 2 * b = 3 * wins_B)
(hB_more_wins : wins_B = wins_A + 8)
(hB_more_loss : b - wins_B = a - wins_A + 8) :
  a = 192 := 
by
  sorry

end team_a_games_played_l57_57023


namespace pascal_fifth_number_l57_57488

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57488


namespace inequality_a_b_l57_57323

theorem inequality_a_b (a b : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) :
    a / (b + 1) + b / (a + 1) ≤ 1 :=
  sorry

end inequality_a_b_l57_57323


namespace Ponchik_week_day_l57_57014

theorem Ponchik_week_day (n s : ℕ) (h1 : s = 20) (h2 : s * (4 * n + 1) = 1360) : n = 4 :=
by
  sorry

end Ponchik_week_day_l57_57014


namespace plane_through_intersection_l57_57958

def plane1 (x y z : ℝ) : Prop := x + y + 5 * z - 1 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 3 * y - z + 2 = 0
def pointM (x y z : ℝ) : Prop := (x, y, z) = (3, 2, 1)

theorem plane_through_intersection (x y z : ℝ) :
  plane1 x y z ∧ plane2 x y z ∧ pointM x y z → 5 * x + 14 * y - 74 * z + 31 = 0 := by
  intro h
  sorry

end plane_through_intersection_l57_57958


namespace boat_speed_still_water_l57_57402

variable (V_b V_s t : ℝ)

-- Conditions given in the problem
axiom speedOfStream : V_s = 13
axiom timeRelation : ∀ t, (V_b + V_s) * t = 2 * (V_b - V_s) * t

-- The statement to be proved
theorem boat_speed_still_water : V_b = 39 :=
by
  sorry

end boat_speed_still_water_l57_57402


namespace mr_willam_land_percentage_over_taxable_land_l57_57946

def total_tax_collected : ℝ := 3840
def tax_paid_by_mr_willam : ℝ := 480
def farm_tax_percentage : ℝ := 0.45

theorem mr_willam_land_percentage_over_taxable_land :
  (tax_paid_by_mr_willam / total_tax_collected) * 100 = 5.625 :=
by
  sorry

end mr_willam_land_percentage_over_taxable_land_l57_57946


namespace starting_number_of_sequence_l57_57692

theorem starting_number_of_sequence :
  ∃ (start : ℤ), 
    (∀ n, 0 ≤ n ∧ n < 8 → start + n * 11 ≤ 119) ∧ 
    (∃ k, 1 ≤ k ∧ k ≤ 8 ∧ 119 = start + (k - 1) * 11) ↔ start = 33 :=
by
  sorry

end starting_number_of_sequence_l57_57692


namespace total_baseball_cards_l57_57184
-- Import the broad Mathlib library

-- The conditions stating the number of cards each person has
def melanie_cards : ℕ := 3
def benny_cards : ℕ := 3
def sally_cards : ℕ := 3
def jessica_cards : ℕ := 3

-- The theorem to prove the total number of cards they have is 12
theorem total_baseball_cards : melanie_cards + benny_cards + sally_cards + jessica_cards = 12 := by
  sorry

end total_baseball_cards_l57_57184


namespace eval_f_a_plus_1_l57_57448

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the condition
axiom a : ℝ

-- State the theorem to be proven
theorem eval_f_a_plus_1 : f (a + 1) = a^2 + 2*a + 1 :=
by
  sorry

end eval_f_a_plus_1_l57_57448


namespace abs_sum_div_diff_sqrt_7_5_l57_57133

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end abs_sum_div_diff_sqrt_7_5_l57_57133


namespace weight_of_new_person_l57_57734

/-- 
The average weight of 10 persons increases by 6.3 kg when a new person replaces one of them. 
The weight of the replaced person is 65 kg. 
Prove that the weight of the new person is 128 kg. 
-/
theorem weight_of_new_person (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) : 
  (avg_increase = 6.3) → 
  (old_weight = 65) → 
  (new_weight = old_weight + 10 * avg_increase) → 
  new_weight = 128 := 
by
  intros
  sorry

end weight_of_new_person_l57_57734


namespace math_problem_l57_57040

theorem math_problem : 2 - (-3)^2 - 4 - (-5) - 6^2 - (-7) = -35 := 
by
  sorry

end math_problem_l57_57040


namespace price_reduction_l57_57075

theorem price_reduction (C : ℝ) (h1 : C > 0) :
  let first_discounted_price := 0.7 * C
  let final_discounted_price := 0.8 * first_discounted_price
  let reduction := 1 - final_discounted_price / C
  reduction = 0.44 :=
by
  sorry

end price_reduction_l57_57075


namespace largest_integer_divides_difference_l57_57790

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end largest_integer_divides_difference_l57_57790


namespace angle_of_isosceles_trapezoid_in_monument_l57_57252

-- Define the larger interior angle x of an isosceles trapezoid in the monument
def larger_interior_angle_of_trapezoid (x : ℝ) : Prop :=
  ∃ n : ℕ, 
    n = 12 ∧
    ∃ α : ℝ, 
      α = 360 / (2 * n) ∧
      ∃ θ : ℝ, 
        θ = (180 - α) / 2 ∧
        x = 180 - θ

-- The theorem stating the larger interior angle x is 97.5 degrees
theorem angle_of_isosceles_trapezoid_in_monument : larger_interior_angle_of_trapezoid 97.5 :=
by 
  sorry

end angle_of_isosceles_trapezoid_in_monument_l57_57252


namespace square_park_area_l57_57274

theorem square_park_area (side_length : ℝ) (h : side_length = 200) : side_length * side_length = 40000 := by
  sorry

end square_park_area_l57_57274


namespace time_to_school_building_l57_57012

theorem time_to_school_building 
  (total_time : ℕ := 30) 
  (time_to_gate : ℕ := 15) 
  (time_to_room : ℕ := 9)
  (remaining_time := total_time - time_to_gate - time_to_room) : 
  remaining_time = 6 :=
by
  sorry

end time_to_school_building_l57_57012


namespace expected_shots_l57_57151

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l57_57151


namespace find_k_l57_57820

-- Definitions for arithmetic sequence properties
noncomputable def sum_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := n * a₁ + (n * (n-1) / 2) * d

noncomputable def term_arith_seq (a₁ d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

-- Given Conditions
variables (a₁ d : ℝ)
variables (k : ℕ)

axiom sum_condition : sum_arith_seq a₁ d 9 = sum_arith_seq a₁ d 4
axiom term_condition : term_arith_seq a₁ d 4 + term_arith_seq a₁ d k = 0

-- Prove k = 10
theorem find_k : k = 10 :=
by
  sorry

end find_k_l57_57820


namespace product_of_m_l57_57785

theorem product_of_m (m n : ℤ) (h_cond : m^2 + m + 8 = n^2) (h_nonneg : n ≥ 0) : 
  (∀ m, (∃ n, m^2 + m + 8 = n^2 ∧ n ≥ 0) → m = 7 ∨ m = -8) ∧ 
  (∃ m1 m2 : ℤ, m1 = 7 ∧ m2 = -8 ∧ (m1 * m2 = -56)) :=
by
  sorry

end product_of_m_l57_57785


namespace cos_5theta_l57_57833

theorem cos_5theta (θ : ℝ) (h : Real.cos θ = 1/3) : Real.cos (5 * θ) = 241/243 :=
by
  sorry

end cos_5theta_l57_57833


namespace gcd_polynomial_l57_57298

theorem gcd_polynomial (b : ℤ) (h : ∃ k : ℤ, b = 2 * 997 * k) : 
  Int.gcd (3 * b^2 + 34 * b + 102) (b + 21) = 21 := 
by
  -- Proof would go here, but is omitted as instructed
  sorry

end gcd_polynomial_l57_57298


namespace f_neg_expression_l57_57650

noncomputable def f : ℝ → ℝ :=
  λ x => if x > 0 then x^2 - 2*x + 3 else sorry

-- Define f by cases: for x > 0 and use the property of odd functions to conclude the expression for x < 0.

theorem f_neg_expression (x : ℝ) (h : x < 0) : f x = -x^2 - 2*x - 3 :=
by
  sorry

end f_neg_expression_l57_57650


namespace intersection_complements_l57_57982

open Set

variable (U : Set (ℝ × ℝ))
variable (M : Set (ℝ × ℝ))
variable (N : Set (ℝ × ℝ))

noncomputable def complementU (A : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ A

theorem intersection_complements :
  let U := {p : ℝ × ℝ | True}
  let M := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y + 2 = x - 2 ∧ x ≠ 2)}
  let N := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y ≠ x - 4)}
  ((complementU U M) ∩ (complementU U N)) = {(2, -2)} :=
by
  let U := {(x, y) : ℝ × ℝ | True}
  let M := {(x, y) : ℝ × ℝ | (y + 2) = (x - 2) ∧ x ≠ 2}
  let N := {(x, y) : ℝ × ℝ | y ≠ (x - 4)}
  have complement_M := U \ M
  have complement_N := U \ N
  sorry

end intersection_complements_l57_57982


namespace gcd_factorial_8_and_6_squared_l57_57767

-- We will use nat.factorial and other utility functions from Mathlib

noncomputable def factorial_8 : ℕ := nat.factorial 8
noncomputable def factorial_6_squared : ℕ := (nat.factorial 6) ^ 2

theorem gcd_factorial_8_and_6_squared : nat.gcd factorial_8 factorial_6_squared = 1440 := by
  -- We'll need Lean to compute prime factorizations
  sorry

end gcd_factorial_8_and_6_squared_l57_57767


namespace Irene_age_is_46_l57_57941

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end Irene_age_is_46_l57_57941


namespace greatest_common_divisor_of_B_l57_57631

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l57_57631


namespace f_even_l57_57108

-- Define E_x^n as specified
def E_x (n : ℕ) (x : ℝ) : ℝ := List.prod (List.map (λ i => x + i) (List.range n))

-- Define the function f(x)
def f (x : ℝ) : ℝ := x * E_x 5 (x - 2)

-- Define the statement to prove f(x) is even
theorem f_even (x : ℝ) : f x = f (-x) := by
  sorry

end f_even_l57_57108


namespace twice_perimeter_is_72_l57_57670

def twice_perimeter_of_square_field (s : ℝ) : ℝ := 2 * 4 * s

theorem twice_perimeter_is_72 (a P : ℝ) (h1 : a = s^2) (h2 : P = 36) 
    (h3 : 6 * a = 6 * (2 * P + 9)) : twice_perimeter_of_square_field s = 72 := 
by
  sorry

end twice_perimeter_is_72_l57_57670


namespace pipe_fills_tank_in_10_hours_l57_57921

variables (pipe_rate leak_rate : ℝ)

-- Conditions
def combined_rate := pipe_rate - leak_rate
def leak_time := 30
def combined_time := 15

-- Express leak_rate from leak_time
noncomputable def leak_rate_def : ℝ := 1 / leak_time

-- Express pipe_rate from combined_time with leak_rate considered
noncomputable def pipe_rate_def : ℝ := 1 / combined_time + leak_rate_def

-- Theorem to be proved
theorem pipe_fills_tank_in_10_hours :
  (1 / pipe_rate_def) = 10 :=
by
  sorry

end pipe_fills_tank_in_10_hours_l57_57921


namespace flour_more_than_sugar_l57_57010

/-
  Mary is baking a cake. The recipe calls for 6 cups of sugar and 9 cups of flour. 
  She already put in 2 cups of flour. 
  Prove that the number of additional cups of flour Mary needs is 1 more than the number of additional cups of sugar she needs.
-/

theorem flour_more_than_sugar (s f a : ℕ) (h_s : s = 6) (h_f : f = 9) (h_a : a = 2) :
  (f - a) - s = 1 :=
by
  sorry

end flour_more_than_sugar_l57_57010


namespace sum_of_coefficients_eq_one_l57_57847

theorem sum_of_coefficients_eq_one (a a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x - 3) ^ 4 = a + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  a + a₁ + a₂ + a₃ + a₄ = 1 :=
by
  intros h
  specialize h 1
  -- Specific calculation steps would go here
  sorry

end sum_of_coefficients_eq_one_l57_57847


namespace parabola_focus_coordinates_l57_57025

theorem parabola_focus_coordinates (x y : ℝ) (h : y = -2 * x^2) : (0, -1 / 8) = (0, (-1 / 2) * (y: ℝ)) :=
sorry

end parabola_focus_coordinates_l57_57025


namespace multiples_six_or_eight_not_both_l57_57468

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l57_57468


namespace arithmetic_seq_sum_l57_57174

theorem arithmetic_seq_sum (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S(3) = 9) 
  (h₂ : S(6) = 36) 
  (h₃ : ∀ n, S(n + 1) = S(n) + a(n + 1)) :
  a(7) + a(8) + a(9) = 45 :=
by
  sorry

end arithmetic_seq_sum_l57_57174


namespace find_monic_polynomial_of_shifted_roots_l57_57180

theorem find_monic_polynomial_of_shifted_roots (a b c : ℝ) (h : ∀ x : ℝ, (x - a) * (x - b) * (x - c) = x^3 - 5 * x + 7) : 
  (x : ℝ) → (x - (a - 3)) * (x - (b - 3)) * (x - (c - 3)) = x^3 + 9 * x^2 + 22 * x + 19 :=
by
  -- Proof will be provided here.
  sorry

end find_monic_polynomial_of_shifted_roots_l57_57180


namespace seating_sessions_l57_57694

theorem seating_sessions (num_parents num_pupils morning_parents afternoon_parents morning_pupils mid_day_pupils evening_pupils session_capacity total_sessions : ℕ) 
  (h1 : num_parents = 61)
  (h2 : num_pupils = 177)
  (h3 : session_capacity = 44)
  (h4 : morning_parents = 35)
  (h5 : afternoon_parents = 26)
  (h6 : morning_pupils = 65)
  (h7 : mid_day_pupils = 57)
  (h8 : evening_pupils = 55)
  (h9 : total_sessions = 8) :
  ∃ (parent_sessions pupil_sessions : ℕ), 
    parent_sessions + pupil_sessions = total_sessions ∧
    parent_sessions = (morning_parents + session_capacity - 1) / session_capacity + (afternoon_parents + session_capacity - 1) / session_capacity ∧
    pupil_sessions = (morning_pupils + session_capacity - 1) / session_capacity + (mid_day_pupils + session_capacity - 1) / session_capacity + (evening_pupils + session_capacity - 1) / session_capacity := 
by
  sorry

end seating_sessions_l57_57694


namespace multiplication_factor_l57_57203

theorem multiplication_factor
  (n : ℕ) (avg_orig avg_new : ℝ) (F : ℝ)
  (H1 : n = 7)
  (H2 : avg_orig = 24)
  (H3 : avg_new = 120)
  (H4 : (n * avg_new) = F * (n * avg_orig)) :
  F = 5 :=
by {
  sorry
}

end multiplication_factor_l57_57203


namespace Pascal_triangle_fifth_number_l57_57559

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57559


namespace radius_of_tangent_circle_l57_57398

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end radius_of_tangent_circle_l57_57398


namespace dependent_variable_is_temperature_l57_57000

-- Define the variables involved in the problem
variables (intensity_of_sunlight : ℝ)
variables (temperature_of_water : ℝ)
variables (duration_of_exposure : ℝ)
variables (capacity_of_heater : ℝ)

-- Define the conditions
def changes_with_duration (temp: ℝ) (duration: ℝ) : Prop :=
  ∃ f : ℝ → ℝ, (∀ d, temp = f d) ∧ ∀ d₁ d₂, d₁ ≠ d₂ → f d₁ ≠ f d₂

-- The theorem we need to prove
theorem dependent_variable_is_temperature :
  changes_with_duration temperature_of_water duration_of_exposure → 
  (∀ t, ∃ d, temperature_of_water = t → duration_of_exposure = d) :=
sorry

end dependent_variable_is_temperature_l57_57000


namespace feta_price_calculation_l57_57199

noncomputable def feta_price_per_pound (sandwiches_price : ℝ) (sandwiches_count : ℕ) 
  (salami_price : ℝ) (brie_factor : ℝ) (olive_price_per_pound : ℝ) 
  (olive_weight : ℝ) (bread_price : ℝ) (total_spent : ℝ)
  (feta_weight : ℝ) :=
  (total_spent - (sandwiches_count * sandwiches_price + salami_price + brie_factor * salami_price + olive_price_per_pound * olive_weight + bread_price)) / feta_weight

theorem feta_price_calculation : 
  feta_price_per_pound 7.75 2 4.00 3 10.00 0.25 2.00 40.00 0.5 = 8.00 := 
by
  sorry

end feta_price_calculation_l57_57199


namespace original_savings_l57_57911

-- Define original savings as a variable
variable (S : ℝ)

-- Define the condition that 1/4 of the savings equals 200
def tv_cost_condition : Prop := (1 / 4) * S = 200

-- State the theorem that if the condition is satisfied, then the original savings are 800
theorem original_savings (h : tv_cost_condition S) : S = 800 :=
by
  sorry

end original_savings_l57_57911


namespace gcd_of_B_is_2_l57_57626

def is_in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = 4 * x + 2

theorem gcd_of_B_is_2 : gcd_upto is_in_B = 2 := by
  sorry

end gcd_of_B_is_2_l57_57626


namespace pascal_15_5th_number_l57_57529

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57529


namespace average_of_remaining_three_numbers_l57_57673

noncomputable def avg_remaining_three_numbers (avg_12 : ℝ) (avg_4 : ℝ) (avg_3 : ℝ) (avg_2 : ℝ) : ℝ :=
  let sum_12 := 12 * avg_12
  let sum_4 := 4 * avg_4
  let sum_3 := 3 * avg_3
  let sum_2 := 2 * avg_2
  let sum_9 := sum_4 + sum_3 + sum_2
  let sum_remaining_3 := sum_12 - sum_9
  sum_remaining_3 / 3

theorem average_of_remaining_three_numbers :
  avg_remaining_three_numbers 6.30 5.60 4.90 7.25 = 8 :=
by {
  sorry
}

end average_of_remaining_three_numbers_l57_57673


namespace find_A_max_min_l57_57253

theorem find_A_max_min :
  ∃ (A_max A_min : ℕ), 
    (A_max = 99999998 ∧ A_min = 17777779) ∧
    (∀ B A, 
      (B > 77777777) ∧
      (Nat.coprime B 36) ∧
      (A = (B % 10) * 10000000 + B / 10) →
      (A ≤ 99999998 ∧ A ≥ 17777779)) :=
by 
  existsi 99999998
  existsi 17777779
  split
  { 
    split 
    { 
      refl 
    }
    refl 
  }
  intros B A h
  sorry

end find_A_max_min_l57_57253


namespace combined_weight_l57_57311

noncomputable def Jake_weight : ℕ := 196
noncomputable def Kendra_weight : ℕ := 94

-- Condition: If Jake loses 8 pounds, he will weigh twice as much as Kendra
axiom lose_8_pounds (j k : ℕ) : (j - 8 = 2 * k) → j = Jake_weight → k = Kendra_weight

-- To Prove: The combined weight of Jake and Kendra is 290 pounds
theorem combined_weight (j k : ℕ) (h₁ : j = Jake_weight) (h₂ : k = Kendra_weight) : j + k = 290 := 
by  sorry

end combined_weight_l57_57311


namespace pascal_fifth_number_l57_57491

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57491


namespace xy_condition_l57_57007

theorem xy_condition (x y : ℝ) (h : x * y + x / y + y / x = -3) : (x - 2) * (y - 2) = 3 :=
sorry

end xy_condition_l57_57007


namespace pascal_fifth_element_15th_row_l57_57593

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57593


namespace num_special_fractions_eq_one_l57_57266

-- Definitions of relatively prime and positive
def are_rel_prime (a b : ℕ) : Prop := Nat.gcd a b = 1
def is_positive (n : ℕ) : Prop := n > 0

-- Statement to prove the number of such fractions
theorem num_special_fractions_eq_one : 
  (∀ (x y : ℕ), is_positive x → is_positive y → are_rel_prime x y → 
    (x + 1) * 10 * y = (y + 1) * 11 * x →
    ((x = 5 ∧ y = 11) ∨ False)) := sorry

end num_special_fractions_eq_one_l57_57266


namespace calc_exponent_l57_57260

theorem calc_exponent (a b : ℕ) : 1^345 + 5^7 / 5^5 = 26 := by
  sorry

end calc_exponent_l57_57260


namespace find_a2_l57_57144

variable {a_n : ℕ → ℚ}

def arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ a1 d, ∀ n, a n = a1 + (n-1) * d

theorem find_a2 (h_seq : arithmetic_seq a_n) (h3_5 : a_n 3 + a_n 5 = 15) (h6 : a_n 6 = 7) :
  a_n 2 = 8 := 
sorry

end find_a2_l57_57144


namespace translation_is_elevator_l57_57251

-- Definitions representing the conditions
def P_A : Prop := true  -- The movement of elevators constitutes translation.
def P_B : Prop := false -- Swinging on a swing does not constitute translation.
def P_C : Prop := false -- Closing an open textbook does not constitute translation.
def P_D : Prop := false -- The swinging of a pendulum does not constitute translation.

-- The goal is to prove that Option A is the phenomenon that belongs to translation
theorem translation_is_elevator : P_A ∧ ¬P_B ∧ ¬P_C ∧ ¬P_D :=
by
  sorry -- proof not required

end translation_is_elevator_l57_57251


namespace relationship_among_a_b_c_l57_57270

noncomputable def f (x : ℝ) : ℝ := sorry  -- The actual function definition is not necessary for this statement.

-- Lean statements for the given conditions
variables {f : ℝ → ℝ}

-- f is even
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- f(x+1) = -f(x)
def periodic_property (f : ℝ → ℝ) := ∀ x, f (x + 1) = - f x

-- f is monotonically increasing on [-1, 0]
def monotonically_increasing_on (f : ℝ → ℝ) := ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the relationship statement
theorem relationship_among_a_b_c (h1 : even_function f) (h2 : periodic_property f) 
  (h3 : monotonically_increasing_on f) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end relationship_among_a_b_c_l57_57270


namespace carmen_counting_cars_l57_57093

theorem carmen_counting_cars 
  (num_trucks : ℕ)
  (num_cars : ℕ)
  (red_trucks : ℕ)
  (black_trucks : ℕ)
  (white_trucks : ℕ)
  (total_vehicles : ℕ)
  (percent_white_trucks : ℚ) :
  num_trucks = 50 →
  num_cars = 40 →
  red_trucks = num_trucks / 2 →
  black_trucks = (20 * num_trucks) / 100 →
  white_trucks = num_trucks - red_trucks - black_trucks →
  total_vehicles = num_trucks + num_cars →
  percent_white_trucks = (white_trucks : ℚ) / total_vehicles * 100 →
  percent_white_trucks ≈ 17 :=
sorry

end carmen_counting_cars_l57_57093


namespace min_a2_plus_b2_l57_57918

-- Define circle and line intercept conditions
def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 2
def line_eq (a b x y : ℝ) : Prop := a * x + 2 * b * y - 4 = 0
def chord_length (chord_len : ℝ) : Prop := chord_len = 4

-- Define the final minimum value to prove
def min_value (a b : ℝ) : ℝ := a^2 + b^2

-- Proving the specific value considering the conditions
theorem min_a2_plus_b2 (a b : ℝ) (h1 : b = a + 2) (h2 : chord_length 4) : min_value a b = 2 := by
  sorry

end min_a2_plus_b2_l57_57918


namespace polynomial_inequality_l57_57332

theorem polynomial_inequality (x : ℝ) : x * (x + 1) * (x + 2) * (x + 3) ≥ -1 :=
sorry

end polynomial_inequality_l57_57332


namespace hardcover_volumes_l57_57100

theorem hardcover_volumes (h p : ℕ) (h_condition : h + p = 12) (cost_condition : 25 * h + 15 * p = 240) : h = 6 :=
by
  -- omitted proof steps for brevity
  sorry

end hardcover_volumes_l57_57100


namespace gcd_B_is_2_l57_57629

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l57_57629


namespace Kim_morning_routine_time_l57_57165

theorem Kim_morning_routine_time :
  let senior_employees := 3
  let junior_employees := 3
  let interns := 3

  let senior_overtime := 2
  let junior_overtime := 3
  let intern_overtime := 1
  let senior_not_overtime := senior_employees - senior_overtime
  let junior_not_overtime := junior_employees - junior_overtime
  let intern_not_overtime := interns - intern_overtime

  let coffee_time := 5
  let email_time := 10
  let supplies_time := 8
  let meetings_time := 6
  let reports_time := 5

  let status_update_time := 3 * senior_employees + 2 * junior_employees + 1 * interns
  let payroll_update_time := 
    4 * senior_overtime + 2 * senior_not_overtime +
    3 * junior_overtime + 1 * junior_not_overtime +
    2 * intern_overtime + 0.5 * intern_not_overtime
  let daily_tasks_time :=
    4 * senior_employees + 3 * junior_employees + 2 * interns

  let total_time := coffee_time + status_update_time + payroll_update_time + daily_tasks_time + email_time + supplies_time + meetings_time + reports_time
  total_time = 101 := by
  sorry

end Kim_morning_routine_time_l57_57165


namespace book_shelf_arrangement_l57_57129

-- Definitions for the problem conditions
def math_books := 3
def english_books := 4
def science_books := 2

-- The total number of ways to arrange the books
def total_arrangements :=
  (Nat.factorial (math_books + english_books + science_books - 6)) * -- For the groups
  (Nat.factorial math_books) * -- For math books within the group
  (Nat.factorial english_books) * -- For English books within the group
  (Nat.factorial science_books) -- For science books within the group

theorem book_shelf_arrangement :
  total_arrangements = 1728 := by
  -- Proof starts here
  sorry

end book_shelf_arrangement_l57_57129


namespace calculate_expression_l57_57091

theorem calculate_expression : 5 * 7 + 10 * 4 - 36 / 3 + 6 * 3 = 81 :=
by
  -- Proof steps would be included here if they were needed, but the proof is left as sorry for now.
  sorry

end calculate_expression_l57_57091


namespace geometric_sequence_product_l57_57484

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) (h_seq : geometric_sequence a) 
  (h_cond : a 2 * a 4 = 16) : a 2 * a 3 * a 4 = 64 ∨ a 2 * a 3 * a 4 = -64 :=
by
  sorry

end geometric_sequence_product_l57_57484


namespace toby_total_time_l57_57891

theorem toby_total_time (d1 d2 d3 d4 : ℕ)
  (speed_loaded speed_unloaded : ℕ)
  (time1 time2 time3 time4 total_time : ℕ)
  (h1 : d1 = 180)
  (h2 : d2 = 120)
  (h3 : d3 = 80)
  (h4 : d4 = 140)
  (h5 : speed_loaded = 10)
  (h6 : speed_unloaded = 20)
  (h7 : time1 = d1 / speed_loaded)
  (h8 : time2 = d2 / speed_unloaded)
  (h9 : time3 = d3 / speed_loaded)
  (h10 : time4 = d4 / speed_unloaded)
  (h11 : total_time = time1 + time2 + time3 + time4) :
  total_time = 39 := by
  sorry

end toby_total_time_l57_57891


namespace amount_of_money_C_l57_57250

theorem amount_of_money_C (a b c d : ℤ) 
  (h1 : a + b + c + d = 600)
  (h2 : a + c = 200)
  (h3 : b + c = 350)
  (h4 : a + d = 300)
  (h5 : a ≥ 2 * b) : c = 150 := 
by
  sorry

end amount_of_money_C_l57_57250


namespace sam_original_seashells_count_l57_57868

-- Definitions representing the conditions
def seashells_given_to_joan : ℕ := 18
def seashells_sam_has_now : ℕ := 17

-- The question and the answer translated to a proof problem
theorem sam_original_seashells_count :
  seashells_given_to_joan + seashells_sam_has_now = 35 :=
by
  sorry

end sam_original_seashells_count_l57_57868


namespace find_y_l57_57390

theorem find_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hrem : x % y = 5) (hdiv : (x : ℝ) / y = 96.2) : y = 25 := by
  sorry

end find_y_l57_57390


namespace chinese_medicine_excess_purchased_l57_57701

-- Define the conditions of the problem

def total_plan : ℕ := 1500

def first_half_percentage : ℝ := 0.55
def second_half_percentage : ℝ := 0.65

-- State the theorem to prove the amount purchased in excess
theorem chinese_medicine_excess_purchased :
    first_half_percentage * total_plan + second_half_percentage * total_plan - total_plan = 300 :=
by 
  sorry

end chinese_medicine_excess_purchased_l57_57701


namespace pascal_row_fifth_number_l57_57534

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57534


namespace gcd_factorial_eight_six_sq_l57_57781

/-- Prove that the gcd of 8! and (6!)^2 is 11520 -/
theorem gcd_factorial_eight_six_sq : Int.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 11520 :=
by
  -- We'll put the proof here, but for now, we use sorry to omit it.
  sorry

end gcd_factorial_eight_six_sq_l57_57781


namespace largest_divisor_of_n_pow4_minus_n_l57_57804

theorem largest_divisor_of_n_pow4_minus_n (n : ℕ) (h : ¬ prime n ∧ n > 1) : ∃ d, d = 6 ∧ ∀ k, k ∣ n * (n - 1) * (n + 1) * (n^2 + n + 1) → k ≤ 6 := sorry

end largest_divisor_of_n_pow4_minus_n_l57_57804


namespace max_a_value_l57_57304

noncomputable def f (x k a : ℝ) : ℝ := x^2 - (k^2 - 5 * a * k + 3) * x + 7

theorem max_a_value : ∀ (k a : ℝ), (0 <= k) → (k <= 2) →
  (∀ (x1 : ℝ), (k <= x1) → (x1 <= k + a) →
  ∀ (x2 : ℝ), (k + 2 * a <= x2) → (x2 <= k + 4 * a) →
  f x1 k a >= f x2 k a) → 
  a <= (2 * Real.sqrt 6 - 4) / 5 := 
sorry

end max_a_value_l57_57304


namespace speed_of_stream_l57_57731

theorem speed_of_stream (v : ℝ) :
  (∀ s : ℝ, s = 3 → (3 + v) / (3 - v) = 2) → v = 1 :=
by 
  intro h
  sorry

end speed_of_stream_l57_57731


namespace gcd_factorials_l57_57766


noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials :
  gcd (factorial 8) (factorial 6 * factorial 6) = 5760 := by
  sorry

end gcd_factorials_l57_57766


namespace quad_equiv_proof_l57_57312

theorem quad_equiv_proof (a b : ℝ) (h : a ≠ 0) (hroot : a * 2019^2 + b * 2019 + 2 = 0) :
  ∃ x : ℝ, a * (x - 1)^2 + b * (x - 1) = -2 ∧ x = 2019 :=
sorry

end quad_equiv_proof_l57_57312


namespace greatest_integer_not_exceeding_100y_l57_57652

noncomputable def y : ℝ := (∑ n in Finset.range 30, Real.cos (n + 1) * (Real.pi / 180)) / (∑ n in Finset.range 30, Real.sin (n + 1) * (Real.pi / 180))

theorem greatest_integer_not_exceeding_100y :
  ⌊100 * y⌋ = 173 :=
by
  sorry

end greatest_integer_not_exceeding_100y_l57_57652


namespace evaluate_polynomial_at_6_l57_57200

def polynomial (x : ℝ) : ℝ := 2 * x^4 + 5 * x^3 - x^2 + 3 * x + 4

theorem evaluate_polynomial_at_6 : polynomial 6 = 3658 :=
by 
  sorry

end evaluate_polynomial_at_6_l57_57200


namespace count_multiples_6_or_8_not_both_l57_57462

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l57_57462


namespace total_tweets_l57_57860

-- Conditions and Definitions
def tweets_happy_per_minute := 18
def tweets_hungry_per_minute := 4
def tweets_reflection_per_minute := 45
def minutes_each_period := 20

-- Proof Problem Statement
theorem total_tweets : 
  (minutes_each_period * tweets_happy_per_minute) + 
  (minutes_each_period * tweets_hungry_per_minute) + 
  (minutes_each_period * tweets_reflection_per_minute) = 1340 :=
by
  sorry

end total_tweets_l57_57860


namespace calculate_savings_l57_57732

/-- Given the income is 19000 and the income to expenditure ratio is 5:4, prove the savings of 3800. -/
theorem calculate_savings (i : ℕ) (exp : ℕ) (rat : ℕ → ℕ → Prop)
  (h_income : i = 19000)
  (h_ratio : rat 5 4)
  (h_exp_eq : ∃ x, i = 5 * x ∧ exp = 4 * x) :
  i - exp = 3800 :=
by 
  sorry

end calculate_savings_l57_57732


namespace circle_radius_l57_57397

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end circle_radius_l57_57397


namespace pascal_triangle_row_fifth_number_l57_57545

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57545


namespace AB_passes_fixed_point_locus_of_N_l57_57979

-- Definition of the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Definition of the point M which is the right-angle vertex
def M : ℝ × ℝ := (1, 2)

-- Statement for Part 1: Prove line AB passes through a fixed point
theorem AB_passes_fixed_point 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) :
    ∃ P : ℝ × ℝ, P = (5, -2) := sorry

-- Statement for Part 2: Find the locus of point N
theorem locus_of_N 
    (A B : ℝ × ℝ)
    (hA : parabola A.1 A.2)
    (hB : parabola B.1 B.2)
    (hM : M = (1, 2))
    (hRightAngle : (A.1 - 1) * (B.1 - 1) + (A.2 - 2) * (B.2 - 2) = 0) 
    (N : ℝ × ℝ)
    (hN : ∃ t : ℝ, N = (t, -(t - 3))) :
    (N.1 - 3)^2 + N.2^2 = 8 ∧ N.1 ≠ 1 := sorry

end AB_passes_fixed_point_locus_of_N_l57_57979


namespace smallest_base_for_100_l57_57715

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end smallest_base_for_100_l57_57715


namespace find_a7_a8_a9_l57_57176

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l57_57176


namespace quadratic_conversion_l57_57267

def quadratic_to_vertex_form (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 1

theorem quadratic_conversion :
  (∀ x : ℝ, quadratic_to_vertex_form x = 2 * (x - 2)^2 - 9) :=
by
  sorry

end quadratic_conversion_l57_57267


namespace find_a_l57_57047

theorem find_a : 
  ∃ a : ℝ, (a > 0) ∧ (1 / Real.logb 5 a + 1 / Real.logb 6 a + 1 / Real.logb 7 a = 1) ∧ a = 210 :=
by
  sorry

end find_a_l57_57047


namespace dhoni_initial_toys_l57_57099

theorem dhoni_initial_toys (x : ℕ) (T : ℕ) 
    (h1 : T = 10 * x) 
    (h2 : T + 16 = 66) : x = 5 := by
  sorry

end dhoni_initial_toys_l57_57099


namespace last_matching_date_2008_l57_57892

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The last date in 2008 when the sum of the first four digits equals the sum of the last four digits is 25 December 2008. -/
theorem last_matching_date_2008 :
  ∃ d m y, d = 25 ∧ m = 12 ∧ y = 2008 ∧
            sum_of_digits 2512 = sum_of_digits 2008 :=
by {
  sorry
}

end last_matching_date_2008_l57_57892


namespace pascal_triangle_fifth_number_l57_57501

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57501


namespace total_consultation_time_l57_57695

-- Define the times in which each chief finishes a pipe
def chief1_time := 10
def chief2_time := 30
def chief3_time := 60

theorem total_consultation_time : 
  ∃ (t : ℕ), (∃ x, ((x / chief1_time) + (x / chief2_time) + (x / chief3_time) = 1) ∧ t = 3 * x) ∧ t = 20 :=
sorry

end total_consultation_time_l57_57695


namespace three_digit_number_second_digit_l57_57410

theorem three_digit_number_second_digit (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : a < 10) (h₂ : b < 10) (h₃ : c < 10) :
  (100 * a + 10 * b + c) - (a + b + c) = 261 → b = 7 :=
by sorry

end three_digit_number_second_digit_l57_57410


namespace pascal_triangle_fifth_number_l57_57512

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57512


namespace pascal_triangle_row_fifth_number_l57_57544

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57544


namespace projection_matrix_solution_l57_57683

theorem projection_matrix_solution (a c : ℚ) (Q : Matrix (Fin 2) (Fin 2) ℚ) 
  (hQ : Q = !![a, 18/45; c, 27/45] ) 
  (proj_Q : Q * Q = Q) : 
  (a, c) = (2/5, 3/5) :=
by
  sorry

end projection_matrix_solution_l57_57683


namespace max_x_for_lcm_120_l57_57209

def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem max_x_for_lcm_120 (x : ℕ) (h : lcm (lcm x 8) 12 = 120) : x ≤ 120 :=
by
-- sorry proof steps not required
sorry

end max_x_for_lcm_120_l57_57209


namespace greatest_common_divisor_of_B_l57_57630

def B : Set ℤ := {n | ∃ x : ℤ, n = 4*x + 2}

theorem greatest_common_divisor_of_B : Nat.gcd (4*x + 2) = 2 :=
by
  sorry

end greatest_common_divisor_of_B_l57_57630


namespace sara_grew_4_onions_l57_57869

def onions_sally := 5
def onions_fred := 9
def total_onions := 18

def onions_sara : ℕ := total_onions - (onions_sally + onions_fred)

theorem sara_grew_4_onions : onions_sara = 4 := by
  -- proof here
  sorry

end sara_grew_4_onions_l57_57869


namespace range_of_m_l57_57677

/-- The quadratic equation x^2 + (2m - 1)x + 4 - 2m = 0 has one root 
greater than 2 and the other less than 2 if and only if m < -3. -/
theorem range_of_m (m : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 > 2 ∧ x2 < 2 ∧ x1 ^ 2 + (2 * m - 1) * x1 + 4 - 2 * m = 0 ∧
    x2 ^ 2 + (2 * m - 1) * x2 + 4 - 2 * m = 0) ↔
    m < -3 := by
  sorry

end range_of_m_l57_57677


namespace smallest_sum_of_xy_l57_57973

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end smallest_sum_of_xy_l57_57973


namespace gcd_factorial_8_and_6_squared_l57_57769

-- We will use nat.factorial and other utility functions from Mathlib

noncomputable def factorial_8 : ℕ := nat.factorial 8
noncomputable def factorial_6_squared : ℕ := (nat.factorial 6) ^ 2

theorem gcd_factorial_8_and_6_squared : nat.gcd factorial_8 factorial_6_squared = 1440 := by
  -- We'll need Lean to compute prime factorizations
  sorry

end gcd_factorial_8_and_6_squared_l57_57769


namespace largest_multiple_of_7_less_than_neg_100_l57_57381

theorem largest_multiple_of_7_less_than_neg_100 : 
  ∃ (x : ℤ), (∃ n : ℤ, x = 7 * n) ∧ x < -100 ∧ ∀ y : ℤ, (∃ m : ℤ, y = 7 * m) ∧ y < -100 → y ≤ x :=
by
  sorry

end largest_multiple_of_7_less_than_neg_100_l57_57381


namespace inscribed_circle_diameter_l57_57423

noncomputable def diameter_inscribed_circle (side_length : ℝ) : ℝ :=
  let s := (3 * side_length) / 2
  let K := (Real.sqrt 3 / 4) * (side_length ^ 2)
  let r := K / s
  2 * r

theorem inscribed_circle_diameter (side_length : ℝ) (h : side_length = 10) :
  diameter_inscribed_circle side_length = (10 * Real.sqrt 3) / 3 :=
by
  rw [h]
  simp [diameter_inscribed_circle]
  sorry

end inscribed_circle_diameter_l57_57423


namespace calculation_is_one_l57_57262

noncomputable def calc_expression : ℝ :=
  (1/2)⁻¹ - (2021 + Real.pi)^0 + 4 * Real.sin (Real.pi / 3) - Real.sqrt 12

theorem calculation_is_one : calc_expression = 1 :=
by
  -- Each of the steps involved in calculating should match the problem's steps
  -- 1. (1/2)⁻¹ = 2
  -- 2. (2021 + π)^0 = 1
  -- 3. 4 * sin(π / 3) = 2√3 with sin(60°) = √3/2
  -- 4. sqrt(12) = 2√3
  -- Hence 2 - 1 + 2√3 - 2√3 = 1
  sorry

end calculation_is_one_l57_57262


namespace expand_polynomial_l57_57277

variable (x : ℝ)

theorem expand_polynomial :
  2 * (5 * x^2 - 3 * x + 4 - x^3) = -2 * x^3 + 10 * x^2 - 6 * x + 8 :=
by
  sorry

end expand_polynomial_l57_57277


namespace john_total_payment_in_month_l57_57163

def daily_pills : ℕ := 2
def cost_per_pill : ℝ := 1.5
def insurance_coverage : ℝ := 0.4
def days_in_month : ℕ := 30

theorem john_total_payment_in_month : john_payment = 54 :=
  let daily_cost := daily_pills * cost_per_pill
  let monthly_cost := daily_cost * days_in_month
  let insurance_paid := monthly_cost * insurance_coverage
  let john_payment := monthly_cost - insurance_paid
  sorry

end john_total_payment_in_month_l57_57163


namespace correct_scientific_notation_l57_57417

def scientific_notation (n : ℝ) : ℝ × ℝ := 
  (4, 5)

theorem correct_scientific_notation : scientific_notation 400000 = (4, 5) :=
by {
  sorry
}

end correct_scientific_notation_l57_57417


namespace GCD_of_set_B_is_2_l57_57633

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l57_57633


namespace initial_length_proof_l57_57697

variables (L : ℕ)

-- Conditions from the problem statement
def condition1 (L : ℕ) : Prop := L - 25 > 118
def condition2 : Prop := 125 - 7 = 118
def initial_length : Prop := L = 143

-- Proof statement
theorem initial_length_proof (L : ℕ) (h1 : condition1 L) (h2 : condition2) : initial_length L :=
sorry

end initial_length_proof_l57_57697


namespace tiffany_bags_l57_57696

/-!
## Problem Statement
Tiffany was collecting cans for recycling. On Monday she had some bags of cans. 
She found 3 bags of cans on the next day and 7 bags of cans the day after that. 
She had altogether 20 bags of cans. Prove that the number of bags of cans she had on Monday is 10.
-/

theorem tiffany_bags (M : ℕ) (h1 : M + 3 + 7 = 20) : M = 10 :=
by {
  sorry
}

end tiffany_bags_l57_57696


namespace thirds_side_length_valid_l57_57140

theorem thirds_side_length_valid (x : ℝ) (h1 : x > 5) (h2 : x < 13) : x = 12 :=
sorry

end thirds_side_length_valid_l57_57140


namespace plan_a_monthly_fee_l57_57904

-- This is the statement for the mathematically equivalent proof problem:
theorem plan_a_monthly_fee (F : ℝ)
  (h1 : ∀ n : ℕ, n = 60 → PlanACost : ℝ := 0.25 * n + F)
  (h2 : ∀ n : ℕ, n = 60 → PlanBCost : ℝ := 0.40 * n)
  (h3 : ∀ n : ℕ, n = 60 → PlanACost = PlanBCost) : F = 9 :=
begin
  sorry
end

end plan_a_monthly_fee_l57_57904


namespace find_tricias_age_l57_57373

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l57_57373


namespace gcd_of_B_is_two_l57_57644

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l57_57644


namespace sum_of_four_circles_l57_57088

open Real

theorem sum_of_four_circles:
  ∀ (s c : ℝ), 
  (2 * s + 3 * c = 26) → 
  (3 * s + 2 * c = 23) → 
  (4 * c = 128 / 5) :=
by
  intros s c h1 h2
  sorry

end sum_of_four_circles_l57_57088


namespace fifth_number_in_pascal_row_l57_57612

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57612


namespace sum_of_terms_l57_57178

-- Given the condition that the sequence a_n is an arithmetic sequence
-- with Sum S_n of first n terms such that S_3 = 9 and S_6 = 36,
-- prove that a_7 + a_8 + a_9 is 45.

variable (a : ℕ → ℝ) -- arithmetic sequence
variable (S : ℕ → ℝ) -- sum of the first n terms of the sequence

axiom sum_3 : S 3 = 9
axiom sum_6 : S 6 = 36
axiom sum_seq_arith : ∀ n : ℕ, S n = n * (a 1) + (n - 1) * n / 2 * (a 2 - a 1)

theorem sum_of_terms : a 7 + a 8 + a 9 = 45 :=
by {
  sorry
}

end sum_of_terms_l57_57178


namespace two_square_numbers_difference_133_l57_57702

theorem two_square_numbers_difference_133 : 
  ∃ (x y : ℤ), x^2 - y^2 = 133 ∧ ((x = 67 ∧ y = 66) ∨ (x = 13 ∧ y = 6)) :=
by {
  sorry
}

end two_square_numbers_difference_133_l57_57702


namespace pascal_15_5th_number_l57_57523

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57523


namespace infinite_solutions_l57_57183

theorem infinite_solutions (a : ℤ) (h_a : a > 1) 
  (h_sol : ∃ x y : ℤ, x^2 - a * y^2 = -1) : 
  ∃ f : ℕ → ℤ × ℤ, ∀ n : ℕ, (f n).fst^2 - a * (f n).snd^2 = -1 :=
sorry

end infinite_solutions_l57_57183


namespace train_passing_time_l57_57036

theorem train_passing_time
  (length_A : ℝ) (length_B : ℝ) (time_A : ℝ) (speed_B : ℝ) 
  (Dir_opposite : true) 
  (passenger_on_A_time : time_A = 10)
  (length_of_A : length_A = 150)
  (length_of_B : length_B = 200)
  (relative_speed : speed_B = length_B / time_A) :
  ∃ x : ℝ, length_A / x = length_B / time_A ∧ x = 7.5 :=
by
  -- conditions stated
  sorry

end train_passing_time_l57_57036


namespace graph_n_plus_k_odd_l57_57483

-- Definitions and assumptions
variable {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V)
variable (n k : ℕ)
variable (hG : Fintype.card V = n)
variable (hCond : ∀ (S : Finset V), S.card = k → (G.commonNeighborsFinset S).card % 2 = 1)

-- Goal
theorem graph_n_plus_k_odd :
  (n + k) % 2 = 1 :=
sorry

end graph_n_plus_k_odd_l57_57483


namespace greatest_common_divisor_of_B_l57_57624

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l57_57624


namespace set_inter_complement_l57_57983

open Set

variable {α : Type*}
variable (U A B : Set α)

theorem set_inter_complement (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 2, 3})
  (hB : B = {1, 4}) :
  ((U \ A) ∩ B) = {4} := 
by
  sorry

end set_inter_complement_l57_57983


namespace multiples_six_or_eight_not_both_l57_57469

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l57_57469


namespace amount_spent_on_tracksuit_l57_57328

-- Definitions based on the conditions
def original_price (x : ℝ) := x
def discount_rate : ℝ := 0.20
def savings : ℝ := 30
def actual_spent (x : ℝ) := 0.8 * x

-- Theorem statement derived from the proof translation
theorem amount_spent_on_tracksuit (x : ℝ) (h : (original_price x) * discount_rate = savings) :
  actual_spent x = 120 :=
by
  sorry

end amount_spent_on_tracksuit_l57_57328


namespace fifth_number_in_pascals_triangle_l57_57585

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57585


namespace complex_conjugate_x_l57_57116

theorem complex_conjugate_x (x : ℝ) (h : x^2 + x - 2 + (x^2 - 3 * x + 2 : ℂ) * Complex.I = 4 + 20 * Complex.I) : x = -3 := sorry

end complex_conjugate_x_l57_57116


namespace geometric_sequence_b_value_l57_57216

theorem geometric_sequence_b_value
  (b : ℝ)
  (hb_pos : b > 0)
  (hgeom : ∃ r : ℝ, 30 * r = b ∧ b * r = 3 / 8) :
  b = 7.5 := by
  sorry

end geometric_sequence_b_value_l57_57216


namespace pascal_triangle_fifth_number_l57_57498

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57498


namespace pascal_row_fifth_number_l57_57531

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57531


namespace jersey_cost_difference_l57_57344

theorem jersey_cost_difference :
  let jersey_cost := 115
  let tshirt_cost := 25
  jersey_cost - tshirt_cost = 90 :=
by
  -- proof goes here
  sorry

end jersey_cost_difference_l57_57344


namespace pascal_triangle_fifth_number_l57_57518

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57518


namespace find_k_l57_57657

-- Define the equation of line m
def line_m (x : ℝ) : ℝ := 2 * x + 8

-- Define the equation of line n with an unknown slope k
def line_n (k : ℝ) (x : ℝ) : ℝ := k * x - 9

-- Define the point of intersection
def intersection_point := (-4, 0)

-- The proof statement
theorem find_k : ∃ k : ℝ, k = -9 / 4 ∧ line_m (-4) = 0 ∧ line_n k (-4) = 0 :=
by
  exists (-9 / 4)
  simp [line_m, line_n, intersection_point]
  sorry

end find_k_l57_57657


namespace no_square_has_units_digit_seven_l57_57988

theorem no_square_has_units_digit_seven :
  ¬ ∃ n : ℕ, n ≤ 9 ∧ (n^2 % 10) = 7 := by
  sorry

end no_square_has_units_digit_seven_l57_57988


namespace gcd_factorials_l57_57763


noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials :
  gcd (factorial 8) (factorial 6 * factorial 6) = 5760 := by
  sorry

end gcd_factorials_l57_57763


namespace pascal_triangle_fifth_number_l57_57575

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57575


namespace one_eighth_of_two_pow_36_eq_two_pow_y_l57_57138

theorem one_eighth_of_two_pow_36_eq_two_pow_y (y : ℕ) : (2^36 / 8 = 2^y) → (y = 33) :=
by
  sorry

end one_eighth_of_two_pow_36_eq_two_pow_y_l57_57138


namespace vertex_coordinates_l57_57675

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 8

-- State the theorem for the coordinates of the vertex
theorem vertex_coordinates : 
  (∃ h k : ℝ, ∀ x : ℝ, parabola x = 2 * (x - h)^2 + k) ∧ h = 1 ∧ k = 8 :=
sorry

end vertex_coordinates_l57_57675


namespace solve_fractional_equation_l57_57341

theorem solve_fractional_equation : 
  ∀ x : ℝ, x = 2 → (2 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by
  intro x hx
  rw hx
  simp
  sorry

end solve_fractional_equation_l57_57341


namespace pascal_fifteen_four_l57_57571

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57571


namespace find_N_l57_57102

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

theorem find_N (N : ℕ) (hN1 : N < 10000)
  (hN2 : N = 26 * sum_of_digits N) : N = 234 ∨ N = 468 := 
  sorry

end find_N_l57_57102


namespace leak_empty_time_l57_57074

theorem leak_empty_time
  (R : ℝ) (L : ℝ)
  (hR : R = 1 / 8)
  (hRL : R - L = 1 / 10) :
  1 / L = 40 :=
by
  sorry

end leak_empty_time_l57_57074


namespace sin_pi_minus_alpha_l57_57445

theorem sin_pi_minus_alpha (α : ℝ) (h : Real.sin (Real.pi - α) = -1/3) : Real.sin α = -1/3 :=
sorry

end sin_pi_minus_alpha_l57_57445


namespace smallest_integer_in_range_l57_57382

theorem smallest_integer_in_range :
  ∃ (n : ℕ), n > 1 ∧ n % 3 = 2 ∧ n % 7 = 2 ∧ n % 8 = 2 ∧ 131 ≤ n ∧ n ≤ 170 :=
by
  sorry

end smallest_integer_in_range_l57_57382


namespace square_problem_solution_l57_57234

theorem square_problem_solution
  (x : ℝ)
  (h1 : ∃ s1 : ℝ, s1^2 = x^2 + 12*x + 36)
  (h2 : ∃ s2 : ℝ, s2^2 = 4*x^2 - 12*x + 9)
  (h3 : 4 * (s1 + s2) = 64) :
  x = 13 / 3 :=
by
  sorry

end square_problem_solution_l57_57234


namespace log_order_l57_57066

theorem log_order {x y z : ℝ} (h1 : 0 < x ∧ x < 1) (h2 : 1 < y) (h3 : ∀ x, log x < 0) : log x < x ∧ x < y :=
by
  sorry

end log_order_l57_57066


namespace tel_aviv_rain_probability_l57_57352

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l57_57352


namespace seven_digit_palindromes_l57_57454

theorem seven_digit_palindromes : 
  let digits := {1, 1, 4, 4, 6, 6, 6} in 
  (count_palindromes digits = 6) :=
by
  sorry

end seven_digit_palindromes_l57_57454


namespace plan_A_fee_eq_nine_l57_57905

theorem plan_A_fee_eq_nine :
  ∃ F : ℝ, (0.25 * 60 + F = 0.40 * 60) ∧ (F = 9) :=
by
  sorry

end plan_A_fee_eq_nine_l57_57905


namespace gcd_of_B_l57_57635

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l57_57635


namespace gcd_B_is_2_l57_57628

-- Definition for the set B given as the condition
def B := {n : ℕ | ∃ x : ℕ, n = (x - 1) + x + (x + 1) + (x + 2)}

-- Lean statement to prove
theorem gcd_B_is_2 : gcd_set B = 2 :=
sorry

end gcd_B_is_2_l57_57628


namespace number_of_cherries_l57_57076

-- Definitions for the problem conditions
def total_fruits : ℕ := 580
def raspberries (b : ℕ) : ℕ := 2 * b
def grapes (c : ℕ) : ℕ := 3 * c
def cherries (r : ℕ) : ℕ := 3 * r

-- Theorem to prove the number of cherries
theorem number_of_cherries (b r g c : ℕ) 
  (H1 : b + r + g + c = total_fruits)
  (H2 : r = raspberries b)
  (H3 : g = grapes c)
  (H4 : c = cherries r) :
  c = 129 :=
by sorry

end number_of_cherries_l57_57076


namespace chessboard_grains_difference_l57_57078

open BigOperators

def grains_on_square (k : ℕ) : ℕ := 2^k

def sum_of_first_n_squares (n : ℕ) : ℕ := ∑ k in Finset.range (n + 1), grains_on_square k

theorem chessboard_grains_difference : 
  grains_on_square 12 - sum_of_first_n_squares 10 = 2050 := 
by 
  -- Proof of the statement goes here.
  sorry

end chessboard_grains_difference_l57_57078


namespace carnations_in_first_bouquet_l57_57698

theorem carnations_in_first_bouquet 
  (c2 : ℕ) (c3 : ℕ) (avg : ℕ) (n : ℕ) (total_carnations : ℕ) : 
  c2 = 14 → c3 = 13 → avg = 12 → n = 3 → total_carnations = avg * n →
  (total_carnations - (c2 + c3) = 9) :=
by
  sorry

end carnations_in_first_bouquet_l57_57698


namespace polynomial_divisibility_l57_57854

open Polynomial

variables {R : Type*} [CommRing R]
variables {f g h k : R[X]}

theorem polynomial_divisibility (h1 : (X^2 + 1) * h + (X - 1) * f + (X - 2) * g = 0)
    (h2 : (X^2 + 1) * k + (X + 1) * f + (X + 2) * g = 0) :
    (X^2 + 1) ∣ (f * g) :=
sorry

end polynomial_divisibility_l57_57854


namespace pascal_fifth_number_l57_57493

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57493


namespace min_students_blue_shirt_red_shoes_l57_57313

theorem min_students_blue_shirt_red_shoes
    (n : ℕ)
    (hn : n % 63 = 0)
    (h_blue_shirt : ∃ k : ℕ, (3 : ℤ) * n = 7 * k)
    (h_red_shoes : ∃ m : ℕ, (4 : ℤ) * n = 9 * m) :
    ∃ x : ℕ, x = 8 :=
by
  -- Place the proof here
  sorry

end min_students_blue_shirt_red_shoes_l57_57313


namespace emily_journey_length_l57_57101

theorem emily_journey_length
  (y : ℝ)
  (h1 : y / 5 + 30 + y / 3 + y / 6 = y) :
  y = 100 :=
by
  sorry

end emily_journey_length_l57_57101


namespace michael_and_emma_dig_time_correct_l57_57660

noncomputable def michael_and_emma_digging_time : ℝ :=
let father_rate := 4
let father_time := 450
let father_depth := father_rate * father_time
let mother_rate := 5
let mother_time := 300
let mother_depth := mother_rate * mother_time
let michael_desired_depth := 3 * father_depth - 600
let emma_desired_depth := 2 * mother_depth + 300
let desired_depth := max michael_desired_depth emma_desired_depth
let michael_rate := 3
let emma_rate := 6
let combined_rate := michael_rate + emma_rate
desired_depth / combined_rate

theorem michael_and_emma_dig_time_correct :
  michael_and_emma_digging_time = 533.33 := 
sorry

end michael_and_emma_dig_time_correct_l57_57660


namespace line_intersects_x_axis_at_3_0_l57_57415

theorem line_intersects_x_axis_at_3_0 : ∃ (x : ℝ), ∃ (y : ℝ), 2 * y + 5 * x = 15 ∧ y = 0 ∧ (x, y) = (3, 0) :=
by
  sorry

end line_intersects_x_axis_at_3_0_l57_57415


namespace zero_ordered_triples_non_zero_satisfy_conditions_l57_57832

theorem zero_ordered_triples_non_zero_satisfy_conditions :
  ∀ (a b c : ℝ), a ≠ 0 → b ≠ 0 → c ≠ 0 → a = b + c → b = c + a → c = a + b → a + b + c ≠ 0 :=
by
  sorry

end zero_ordered_triples_non_zero_satisfy_conditions_l57_57832


namespace bread_rise_times_l57_57858

-- Defining the conditions
def rise_time : ℕ := 120
def kneading_time : ℕ := 10
def baking_time : ℕ := 30
def total_time : ℕ := 280

-- The proof statement
theorem bread_rise_times (n : ℕ) 
  (h1 : rise_time * n + kneading_time + baking_time = total_time) 
  : n = 2 :=
sorry

end bread_rise_times_l57_57858


namespace find_second_expression_l57_57201

theorem find_second_expression (a : ℕ) (x : ℕ) (h1 : (2 * a + 16 + x) / 2 = 69) (h2 : a = 26) : x = 70 := 
by
  sorry

end find_second_expression_l57_57201


namespace roots_solution_l57_57006

theorem roots_solution (p q : ℝ) (h1 : (∀ x : ℝ, (x - 3) * (3 * x + 8) = x^2 - 5 * x + 6 → (x = p ∨ x = q)))
  (h2 : p + q = 0) (h3 : p * q = -9) : (p + 4) * (q + 4) = 7 :=
by
  sorry

end roots_solution_l57_57006


namespace trigonometric_identity_l57_57434

theorem trigonometric_identity
  (θ : ℝ)
  (h1 : θ > -π/2)
  (h2 : θ < 0)
  (h3 : Real.tan θ = -2) :
  (Real.sin θ)^2 / (Real.cos (2 * θ) + 2) = 4 / 7 :=
sorry

end trigonometric_identity_l57_57434


namespace john_pays_in_30_day_month_l57_57161

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end john_pays_in_30_day_month_l57_57161


namespace probability_first_die_l57_57377

theorem probability_first_die (n : ℕ) (n_pos : n = 4025) (m : ℕ) (m_pos : m = 2012) : 
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  (favorable_outcomes / total_outcomes : ℚ) = 1006 / 4025 :=
by
  have h_n : n = 4025 := n_pos
  have h_m : m = 2012 := m_pos
  let total_outcomes := (n * (n + 1)) / 2
  let favorable_outcomes := (m * (m + 1)) / 2
  sorry

end probability_first_die_l57_57377


namespace number_divided_by_five_is_same_as_three_added_l57_57053

theorem number_divided_by_five_is_same_as_three_added :
  ∃ x : ℚ, x / 5 = x + 3 ∧ x = -15 / 4 :=
by
  sorry

end number_divided_by_five_is_same_as_three_added_l57_57053


namespace andreas_living_room_floor_area_l57_57739

-- Definitions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_coverage_percentage : ℝ := 0.30
def carpet_area : ℝ := carpet_length * carpet_width

-- Theorem statement
theorem andreas_living_room_floor_area (A : ℝ) 
  (h1 : carpet_coverage_percentage * A = carpet_area) :
  A = 120 :=
by
  sorry

end andreas_living_room_floor_area_l57_57739


namespace pascal_fifth_element_15th_row_l57_57594

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57594


namespace more_oranges_than_apples_l57_57166

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l57_57166


namespace gcd_factorial_eight_squared_six_factorial_squared_l57_57775

theorem gcd_factorial_eight_squared_six_factorial_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 2880 := by
  sorry

end gcd_factorial_eight_squared_six_factorial_squared_l57_57775


namespace unique_integers_exist_l57_57418

theorem unique_integers_exist (p : ℕ) (hp : p > 1) : 
  ∃ (a b c : ℤ), b^2 - 4*a*c = 1 - 4*p ∧ 0 < a ∧ a ≤ c ∧ -a ≤ b ∧ b < a :=
sorry

end unique_integers_exist_l57_57418


namespace spent_amount_l57_57384

def initial_amount : ℕ := 15
def final_amount : ℕ := 11

theorem spent_amount : initial_amount - final_amount = 4 :=
by
  sorry

end spent_amount_l57_57384


namespace max_path_length_is_32_l57_57070
-- Import the entire Mathlib library to use its definitions and lemmas

-- Definition of the problem setup
def number_of_edges_4x4_grid : Nat := 
  let total_squares := 4 * 4
  let total_edges_per_square := 4
  total_squares * total_edges_per_square

-- Definitions of internal edges shared by adjacent squares
def distinct_edges_4x4_grid : Nat := 
  let horizontal_lines := 5 * 4
  let vertical_lines := 5 * 4
  horizontal_lines + vertical_lines

-- Calculate the maximum length of the path
def max_length_of_path_4x4_grid : Nat := 
  let degree_3_nodes := 8
  distinct_edges_4x4_grid - degree_3_nodes

-- Main statement: Prove that the maximum length of the path is 32
theorem max_path_length_is_32 : max_length_of_path_4x4_grid = 32 := by
  -- Definitions for clarity and correctness
  have h1 : number_of_edges_4x4_grid = 64 := rfl
  have h2 : distinct_edges_4x4_grid = 40 := rfl
  have h3 : max_length_of_path_4x4_grid = 32 := rfl
  exact h3

end max_path_length_is_32_l57_57070


namespace odd_function_at_zero_l57_57963

theorem odd_function_at_zero
  (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  f 0 = 0 :=
by
  sorry

end odd_function_at_zero_l57_57963


namespace pascal_fifth_element_15th_row_l57_57600

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57600


namespace min_cosine_largest_angle_l57_57289

theorem min_cosine_largest_angle (a b c : ℕ → ℝ) 
  (triangle_inequality: ∀ i, a i ≤ b i ∧ b i ≤ c i)
  (pythagorean_inequality: ∀ i, (a i)^2 + (b i)^2 ≥ (c i)^2)
  (A : ℝ := ∑' i, a i)
  (B : ℝ := ∑' i, b i)
  (C : ℝ := ∑' i, c i) :
  (A^2 + B^2 - C^2) / (2 * A * B) ≥ 1 - (Real.sqrt 2) :=
sorry

end min_cosine_largest_angle_l57_57289


namespace gcd_B_eq_two_l57_57646

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l57_57646


namespace equidistant_cyclist_l57_57811

-- Definition of key parameters
def speed_car := 60  -- in km/h
def speed_cyclist := 18  -- in km/h
def speed_pedestrian := 6  -- in km/h
def distance_AC := 10  -- in km
def angle_ACB := 60  -- in degrees
def time_car_start := (7, 58)  -- 7:58 AM
def time_cyclist_start := (8, 0)  -- 8:00 AM
def time_pedestrian_start := (6, 44) -- 6:44 AM
def time_solution := (8, 6)  -- 8:06 AM

-- Time difference function
def time_diff (t1 t2 : Nat × Nat) : Nat :=
  (t2.1 - t1.1) * 60 + (t2.2 - t1.2)  -- time difference in minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (m : Nat) : ℝ :=
  m / 60.0

-- Distances traveled by car, cyclist, and pedestrian by the given time
noncomputable def distance_car (t1 t2 : Nat × Nat) : ℝ :=
  speed_car * (minutes_to_hours (time_diff t1 t2) + 2 / 60.0)

noncomputable def distance_cyclist (t1 t2 : Nat × Nat) : ℝ :=
  speed_cyclist * minutes_to_hours (time_diff t1 t2)

noncomputable def distance_pedestrian (t1 t2 : Nat × Nat) : ℝ :=
  speed_pedestrian * (minutes_to_hours (time_diff t1 t2) + 136 / 60.0)

-- Verification statement
theorem equidistant_cyclist :
  distance_car time_car_start time_solution = distance_pedestrian time_pedestrian_start time_solution → 
  distance_cyclist time_cyclist_start time_solution = 
  distance_car time_car_start time_solution ∧
  distance_cyclist time_cyclist_start time_solution = 
  distance_pedestrian time_pedestrian_start time_solution :=
by
  -- Given conditions and the correctness to be shown
  sorry

end equidistant_cyclist_l57_57811


namespace gcd_factorial_8_6_squared_l57_57773

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end gcd_factorial_8_6_squared_l57_57773


namespace domain_of_k_l57_57038

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 6)) + (1 / (x^2 + 2*x + 9)) + (1 / (x^3 - 27))

theorem domain_of_k : {x : ℝ | k x ≠ 0} = {x : ℝ | x ≠ -6 ∧ x ≠ 3} :=
by
  sorry

end domain_of_k_l57_57038


namespace product_of_three_consecutive_integers_is_multiple_of_6_l57_57056

theorem product_of_three_consecutive_integers_is_multiple_of_6 (n : ℕ) (h : n > 0) :
    ∃ k : ℕ, n * (n + 1) * (n + 2) = 6 * k :=
by
  sorry

end product_of_three_consecutive_integers_is_multiple_of_6_l57_57056


namespace gcd_factorial_eight_squared_six_factorial_squared_l57_57776

theorem gcd_factorial_eight_squared_six_factorial_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 2880 := by
  sorry

end gcd_factorial_eight_squared_six_factorial_squared_l57_57776


namespace irene_age_is_46_l57_57944

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end irene_age_is_46_l57_57944


namespace pascal_row_fifth_number_l57_57535

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57535


namespace race_problem_l57_57996

theorem race_problem 
    (d : ℕ) (a1 : ℕ) (a2 : ℕ) 
    (h1 : d = 60)
    (h2 : a1 = 10)
    (h3 : a2 = 20) 
    (const_speed : ∀ (x y z : ℕ), x * y = z → y ≠ 0 → x = z / y) :
  (d - d * (d - a1) / (d - a2) = 12) := 
by {
  sorry
}

end race_problem_l57_57996


namespace prism_surface_area_l57_57065

theorem prism_surface_area (a : ℝ) : 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  surface_area_cubes - surface_area_shared_faces = 14 * a^2 := 
by 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  have : surface_area_cubes - surface_area_shared_faces = 14 * a^2 := sorry
  exact this

end prism_surface_area_l57_57065


namespace cylinder_volume_transformation_l57_57360

noncomputable def original_volume := 20
noncomputable def original_radius := r
noncomputable def original_height := h
noncomputable def new_radius := 3 * original_radius
noncomputable def new_height := 2 * original_height
noncomputable def volume (radius : ℝ) (height : ℝ) := π * radius ^ 2 * height
noncomputable def new_volume := volume new_radius new_height

theorem cylinder_volume_transformation :
  (original_volume = volume original_radius original_height) →
  new_volume = 360 := by
  sorry

end cylinder_volume_transformation_l57_57360


namespace inverse_is_correct_l57_57356

-- Definitions
def original_proposition (n : ℤ) : Prop := n < 0 → n ^ 2 > 0
def inverse_proposition (n : ℤ) : Prop := n ^ 2 > 0 → n < 0

-- Theorem stating the inverse
theorem inverse_is_correct : 
  (∀ n : ℤ, original_proposition n) → (∀ n : ℤ, inverse_proposition n) :=
by
  sorry

end inverse_is_correct_l57_57356


namespace roots_eq_squares_l57_57009

theorem roots_eq_squares (p q : ℝ) (h1 : p^2 - 5 * p + 6 = 0) (h2 : q^2 - 5 * q + 6 = 0) :
  p^2 + q^2 = 13 :=
sorry

end roots_eq_squares_l57_57009


namespace smallest_base_to_express_100_with_three_digits_l57_57722

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end smallest_base_to_express_100_with_three_digits_l57_57722


namespace sqrt_sum_of_cubes_l57_57045

theorem sqrt_sum_of_cubes :
  √(4^3 + 4^3 + 4^3 + 4^3) = 16 :=
by
  sorry

end sqrt_sum_of_cubes_l57_57045


namespace leif_has_more_oranges_than_apples_l57_57170

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l57_57170


namespace combined_percentage_increase_l57_57938

def initial_interval_days : ℝ := 50
def additive_A_effect : ℝ := 0.20
def additive_B_effect : ℝ := 0.30
def additive_C_effect : ℝ := 0.40

theorem combined_percentage_increase :
  ((1 + additive_A_effect) * (1 + additive_B_effect) * (1 + additive_C_effect) - 1) * 100 = 118.4 :=
by
  norm_num
  sorry

end combined_percentage_increase_l57_57938


namespace anna_baked_60_cupcakes_l57_57751

variable (C : ℕ)
variable (h1 : (1/5 : ℚ) * C - 3 = 9)

theorem anna_baked_60_cupcakes (h1 : (1/5 : ℚ) * C - 3 = 9) : C = 60 :=
sorry

end anna_baked_60_cupcakes_l57_57751


namespace find_d_from_sine_wave_conditions_l57_57416

theorem find_d_from_sine_wave_conditions (a b d : ℝ) (h1 : d + a = 4) (h2 : d - a = -2) : d = 1 :=
by {
  sorry
}

end find_d_from_sine_wave_conditions_l57_57416


namespace parallel_tangent_line_l57_57205

theorem parallel_tangent_line (b : ℝ) :
  (∃ b : ℝ, (∀ x y : ℝ, x + 2 * y + b = 0 → (x^2 + y^2 = 5))) →
  (b = 5 ∨ b = -5) :=
by
  sorry

end parallel_tangent_line_l57_57205


namespace triangular_25_l57_57222

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end triangular_25_l57_57222


namespace gcd_factorials_l57_57764


noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials :
  gcd (factorial 8) (factorial 6 * factorial 6) = 5760 := by
  sorry

end gcd_factorials_l57_57764


namespace problem_statement_l57_57853

noncomputable def f (x : ℝ) (b : ℝ) : ℝ :=
  if x ≥ 0 then 2^x + 2*x + b else -(2^(-x) + 2*(-x) + b)

theorem problem_statement (b : ℝ) (hb : 2^0 + 2*0 + b = 0) : f (-1) b = -3 :=
by
  sorry

end problem_statement_l57_57853


namespace leif_apples_oranges_l57_57169

theorem leif_apples_oranges : 
  let apples := 14
  let dozens_of_oranges := 2 
  let oranges := dozens_of_oranges * 12
  in oranges - apples = 10 :=
by 
  let apples := 14
  let dozens_of_oranges := 2
  let oranges := dozens_of_oranges * 12
  show oranges - apples = 10
  sorry

end leif_apples_oranges_l57_57169


namespace triangular_angles_l57_57977

noncomputable def measure_of_B (A : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3))

noncomputable def length_of_c (A : ℝ) : ℝ := 
  Real.sqrt (22 - 6 * Real.sqrt 13 * Real.cos (measure_of_B A))

noncomputable def area_of_triangle_ABC (A : ℝ) : ℝ := 
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * A)) / 3)

theorem triangular_angles 
  (a b c : ℝ) (b_pos : b = Real.sqrt 13) (a_pos : a = 3) (h : b * Real.cos c = (2 * a - c) * Real.cos (measure_of_B c)) :
  c = length_of_c c ∧
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * c)) / 3) = area_of_triangle_ABC c :=
by
  sorry

end triangular_angles_l57_57977


namespace total_tweets_l57_57859

-- Conditions and Definitions
def tweets_happy_per_minute := 18
def tweets_hungry_per_minute := 4
def tweets_reflection_per_minute := 45
def minutes_each_period := 20

-- Proof Problem Statement
theorem total_tweets : 
  (minutes_each_period * tweets_happy_per_minute) + 
  (minutes_each_period * tweets_hungry_per_minute) + 
  (minutes_each_period * tweets_reflection_per_minute) = 1340 :=
by
  sorry

end total_tweets_l57_57859


namespace radius_of_tangent_circle_l57_57399

theorem radius_of_tangent_circle (k r : ℝ) (hk : k > 8) (h1 : k - 8 = r) (h2 : r * Real.sqrt 2 = k) : 
  r = 8 * (Real.sqrt 2 + 1) := 
sorry

end radius_of_tangent_circle_l57_57399


namespace calculate_nested_expression_l57_57261

theorem calculate_nested_expression :
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2 = 1457 :=
by
  sorry

end calculate_nested_expression_l57_57261


namespace probability_of_same_color_correct_l57_57131

def total_plates : ℕ := 13
def red_plates : ℕ := 7
def blue_plates : ℕ := 6

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def total_ways_to_choose_two : ℕ := choose total_plates 2
noncomputable def ways_to_choose_two_red : ℕ := choose red_plates 2
noncomputable def ways_to_choose_two_blue : ℕ := choose blue_plates 2

noncomputable def ways_to_choose_two_same_color : ℕ :=
  ways_to_choose_two_red + ways_to_choose_two_blue

noncomputable def probability_same_color : ℚ :=
  ways_to_choose_two_same_color / total_ways_to_choose_two

theorem probability_of_same_color_correct :
  probability_same_color = 4 / 9 := by
  sorry

end probability_of_same_color_correct_l57_57131


namespace evaluate_f_at_2_l57_57134

def f (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem evaluate_f_at_2 : f 2 = 4 :=
by
  -- Proof goes here
  sorry

end evaluate_f_at_2_l57_57134


namespace pascal_triangle_row_fifth_number_l57_57547

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57547


namespace find_a_l57_57004

noncomputable def f (x a : ℝ) : ℝ := -x^3 + 3*x + a

theorem find_a (a : ℝ) :
  (∃! x : ℝ, f x a = 0) → (a = -2 ∨ a = 2) :=
sorry

end find_a_l57_57004


namespace train_meeting_distance_l57_57736

theorem train_meeting_distance
  (d : ℝ) (tx ty: ℝ) (dx dy: ℝ)
  (hx : dx = 140) 
  (hy : dy = 140)
  (hx_speed : dx / tx = 35) 
  (hy_speed : dy / ty = 46.67) 
  (meet : tx = ty) :
  d = 60 := 
sorry

end train_meeting_distance_l57_57736


namespace smallest_base_for_100_l57_57717

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end smallest_base_for_100_l57_57717


namespace shifted_line_does_not_pass_third_quadrant_l57_57725

def line_eq (x: ℝ) : ℝ := -2 * x - 1
def shifted_line_eq (x: ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_third_quadrant :
  ¬∃ x y : ℝ, shifted_line_eq x = y ∧ x < 0 ∧ y < 0 :=
sorry

end shifted_line_does_not_pass_third_quadrant_l57_57725


namespace multiples_of_6_or_8_under_201_not_both_l57_57460

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l57_57460


namespace gcd_B_eq_two_l57_57647

def sum_of_four_consecutive (x : ℕ) : ℕ := (x - 1) + x + (x + 1) + (x + 2)

def in_B (n : ℕ) : Prop :=
  ∃ x : ℕ, n = sum_of_four_consecutive x

theorem gcd_B_eq_two : ∀ n ∈ B, gcd n = 2 :=
by
  -- Proof is skipped intentionally.
  sorry

end gcd_B_eq_two_l57_57647


namespace future_ages_equation_l57_57623

-- Defining the ages of Joe and James with given conditions
def joe_current_age : ℕ := 22
def james_current_age : ℕ := 12

-- Defining the condition that Joe is 10 years older than James
lemma joe_older_than_james : joe_current_age = james_current_age + 10 := by
  unfold joe_current_age james_current_age
  simp

-- Defining the future age condition equation and the target years y.
theorem future_ages_equation (y : ℕ) :
  2 * (joe_current_age + y) = 3 * (james_current_age + y) → y = 8 := by
  unfold joe_current_age james_current_age
  intro h
  linarith

end future_ages_equation_l57_57623


namespace shifted_parabola_sum_constants_l57_57055

theorem shifted_parabola_sum_constants :
  let a := 2
  let b := -17
  let c := 43
  a + b + c = 28 := sorry

end shifted_parabola_sum_constants_l57_57055


namespace determine_monotonically_increasing_interval_l57_57931

noncomputable theory
open Real

/-- Problem Statement: Given the function f(x) = 7 * sin(x - π / 6), prove that it is monotonically increasing in the interval (0, π/2). -/
def monotonically_increasing_interval (x : ℝ) : Prop :=
  0 < x ∧ x < π / 2

def function_f (x : ℝ) : ℝ :=
  7 * sin (x - π / 6)

theorem determine_monotonically_increasing_interval :
  ∀ x : ℝ, monotonically_increasing_interval x → monotone_on function_f (set.Ioo 0 (π / 2)) :=
sorry

end determine_monotonically_increasing_interval_l57_57931


namespace intersection_empty_l57_57293

open Set

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := by
  sorry

end intersection_empty_l57_57293


namespace seating_arrangement_l57_57316

-- We define the conditions under which we will prove our theorem.
def chairs : ℕ := 7
def people : ℕ := 5

/-- Prove that there are exactly 1800 ways to seat five people in seven chairs such that the first person cannot sit in the first or last chair. -/
theorem seating_arrangement : (5 * 6 * 5 * 4 * 3) = 1800 :=
by
  sorry

end seating_arrangement_l57_57316


namespace a_perp_a_add_b_l57_57452

def vector (α : Type*) := α × α

def a : vector ℤ := (2, -1)
def b : vector ℤ := (1, 7)

def dot_product (v1 v2 : vector ℤ) : ℤ :=
  v1.1 * v2.1 + v1.2 * v2.2

def add_vector (v1 v2 : vector ℤ) : vector ℤ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def perpendicular (v1 v2 : vector ℤ) : Prop :=
  dot_product v1 v2 = 0

theorem a_perp_a_add_b :
  perpendicular a (add_vector a b) :=
by {
  sorry
}

end a_perp_a_add_b_l57_57452


namespace distance_is_half_volume_is_pi_squared_l57_57315

namespace SphereRotation

-- Define the radius of the sphere and the length of the chord
def radius : ℝ := 1
def chord_length : ℝ := real.sqrt 3

-- Define the distance from the center of the sphere to the line
def distance_center_to_line (r l : ℝ) : ℝ := real.sqrt (r^2 - (l / 2)^2)

-- Prove that the distance is 1/2 given the conditions
theorem distance_is_half : distance_center_to_line radius chord_length = 1/2 := by
  sorry

-- Define the volume of the torus formed by rotating the sphere about a line
def torus_volume (r R : ℝ) : ℝ := 2 * real.pi^2 * R * r^2

-- Prove that the volume is pi^2 given the conditions
theorem volume_is_pi_squared : torus_volume radius (1/2) = real.pi^2 := by
  sorry

end SphereRotation

end distance_is_half_volume_is_pi_squared_l57_57315


namespace exponent_multiplication_correct_l57_57057

theorem exponent_multiplication_correct (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end exponent_multiplication_correct_l57_57057


namespace pascal_triangle_fifth_number_l57_57578

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57578


namespace Tricia_is_five_years_old_l57_57369

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l57_57369


namespace elem_of_M_l57_57655

variable (U M : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5})
variable (hM : U \ M = {1, 3})

theorem elem_of_M : 2 ∈ M :=
by {
  sorry
}

end elem_of_M_l57_57655


namespace multiples_count_l57_57457

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l57_57457


namespace single_elimination_games_l57_57143

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = n - 1 :=
by
  have h1 : n = 512 := h
  use 511
  sorry

end single_elimination_games_l57_57143


namespace sqrt_of_sum_of_powers_l57_57041

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l57_57041


namespace fifth_number_in_pascal_row_l57_57614

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57614


namespace simplify_and_evaluate_l57_57871

theorem simplify_and_evaluate (x : ℝ) (h : x = real.sqrt 3 - 1) : (1 - 2 / (x + 1)) / ((x^2 - 1) / (3 * x + 3)) = real.sqrt 3 :=
by {
  rw h,
  sorry
}

end simplify_and_evaluate_l57_57871


namespace pascal_triangle_row_fifth_number_l57_57539

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57539


namespace man_half_father_age_in_years_l57_57404

theorem man_half_father_age_in_years
  (M F Y : ℕ) 
  (h1: M = (2 * F) / 5) 
  (h2: F = 25) 
  (h3: M + Y = (F + Y) / 2) : 
  Y = 5 := by 
  sorry

end man_half_father_age_in_years_l57_57404


namespace negation_cube_of_every_odd_is_odd_l57_57882

def odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

def cube (n : ℤ) : ℤ := n * n * n

def cube_of_odd_is_odd (n : ℤ) : Prop := odd n → odd (cube n)

theorem negation_cube_of_every_odd_is_odd :
  ¬ (∀ n : ℤ, odd n → odd (cube n)) ↔ ∃ n : ℤ, odd n ∧ ¬ odd (cube n) :=
sorry

end negation_cube_of_every_odd_is_odd_l57_57882


namespace youngest_child_age_l57_57067

variable (Y : ℕ) (O : ℕ) -- Y: the youngest child's present age
variable (P₀ P₁ P₂ P₃ : ℕ) -- P₀, P₁, P₂, P₃: the present ages of the 4 original family members

-- Conditions translated to Lean
variable (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
variable (h₂ : O = Y + 2)
variable (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24)

theorem youngest_child_age (h₁ : ((P₀ - 10) + (P₁ - 10) + (P₂ - 10) + (P₃ - 10)) / 4 = 24)
                       (h₂ : O = Y + 2)
                       (h₃ : ((P₀ + P₁ + P₂ + P₃) + Y + O) / 6 = 24) :
  Y = 3 := by 
  sorry

end youngest_child_age_l57_57067


namespace max_value_of_a_exists_max_value_of_a_l57_57980

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  a ≤ (Real.sqrt 6 / 3) :=
sorry

theorem exists_max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ∃ a_max: ℝ, a_max = (Real.sqrt 6 / 3) ∧ (∀ a', (a' ≤ a_max)) :=
sorry

end max_value_of_a_exists_max_value_of_a_l57_57980


namespace pascal_triangle_fifth_number_l57_57502

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57502


namespace parking_lot_total_spaces_l57_57741

-- Given conditions
def section1_spaces := 320
def section2_spaces := 440
def section3_spaces := section2_spaces - 200
def total_spaces := section1_spaces + section2_spaces + section3_spaces

-- Problem statement to be proved
theorem parking_lot_total_spaces : total_spaces = 1000 :=
by
  sorry

end parking_lot_total_spaces_l57_57741


namespace tan_of_x_is_3_l57_57296

theorem tan_of_x_is_3 (x : ℝ) (h : Real.tan x = 3) (hx : Real.cos x ≠ 0) : 
  (Real.sin x + 3 * Real.cos x) / (2 * Real.sin x - 3 * Real.cos x) = 2 :=
by
  sorry

end tan_of_x_is_3_l57_57296


namespace simplify_expression_l57_57336

theorem simplify_expression (x : ℝ) :
  (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * (x - 1) + 1 = x^4 :=
sorry

end simplify_expression_l57_57336


namespace pascal_fifth_number_in_row_15_l57_57552

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57552


namespace average_marks_combined_l57_57235

theorem average_marks_combined (avg1 : ℝ) (students1 : ℕ) (avg2 : ℝ) (students2 : ℕ) :
  avg1 = 30 → students1 = 30 → avg2 = 60 → students2 = 50 →
  (students1 * avg1 + students2 * avg2) / (students1 + students2) = 48.75 := 
by
  intros h_avg1 h_students1 h_avg2 h_students2
  sorry

end average_marks_combined_l57_57235


namespace sector_central_angle_l57_57119

theorem sector_central_angle (r l θ : ℝ) (h_perimeter : 2 * r + l = 8) (h_area : (1 / 2) * l * r = 4) : θ = 2 :=
by
  sorry

end sector_central_angle_l57_57119


namespace race_distance_l57_57314

theorem race_distance (x : ℝ) (D : ℝ) (vA vB : ℝ) (head_start win_margin : ℝ):
  vA = 5 * x →
  vB = 4 * x →
  head_start = 100 →
  win_margin = 200 →
  (D - win_margin) / vB = (D - head_start) / vA →
  D = 600 :=
by 
  sorry

end race_distance_l57_57314


namespace find_radius_l57_57699

theorem find_radius 
  (r : ℝ)
  (h1 : ∀ (x y : ℝ), ((x - r) ^ 2 + y ^ 2 = r ^ 2) → (4 * x ^ 2 + 9 * y ^ 2 = 36)) 
  (h2 : (4 * r ^ 2 + 9 * 0 ^ 2 = 36)) 
  (h3 : ∃ r : ℝ, r > 0) : 
  r = (2 * Real.sqrt 5) / 3 :=
sorry

end find_radius_l57_57699


namespace largest_divisor_of_difference_between_n_and_n4_l57_57801

theorem largest_divisor_of_difference_between_n_and_n4 (n : ℤ) (h_composite : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ p * q = n) :
  ∃ k : ℤ, k = 6 ∧ k ∣ (n^4 - n) :=
begin
  sorry
end

end largest_divisor_of_difference_between_n_and_n4_l57_57801


namespace common_points_count_l57_57949

noncomputable def eq1 (x y : ℝ) : Prop := (x - 2 * y + 3) * (4 * x + y - 5) = 0
noncomputable def eq2 (x y : ℝ) : Prop := (x + 2 * y - 5) * (3 * x - 4 * y + 6) = 0

theorem common_points_count : 
  (∃ x1 y1 : ℝ, eq1 x1 y1 ∧ eq2 x1 y1) ∧
  (∃ x2 y2 : ℝ, eq1 x2 y2 ∧ eq2 x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)) ∧
  (∃ x3 y3 : ℝ, eq1 x3 y3 ∧ eq2 x3 y3 ∧ (x3 ≠ x1 ∧ x3 ≠ x2 ∧ y3 ≠ y1 ∧ y3 ≠ y2)) ∧ 
  (∃ x4 y4 : ℝ, eq1 x4 y4 ∧ eq2 x4 y4 ∧ (x4 ≠ x1 ∧ x4 ≠ x2 ∧ x4 ≠ x3 ∧ y4 ≠ y1 ∧ y4 ≠ y2 ∧ y4 ≠ y3)) ∧ 
  ∀ x y : ℝ, (eq1 x y ∧ eq2 x y) → (((x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2) ∨ (x = x3 ∧ y = y3) ∨ (x = x4 ∧ y = y4))) :=
by
  sorry

end common_points_count_l57_57949


namespace Pascal_triangle_fifth_number_l57_57563

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57563


namespace unobstructed_sight_l57_57447

-- Define the curve C as y = 2x^2
def curve (x : ℝ) : ℝ := 2 * x^2

-- Define point A and point B
def pointA : ℝ × ℝ := (0, -2)
def pointB (a : ℝ) : ℝ × ℝ := (3, a)

-- Statement of the problem
theorem unobstructed_sight {a : ℝ} (h : ∀ x : ℝ, 0 ≤ x → x ≤ 3 → 4 * x - 2 ≥ 2 * x^2) : a < 10 :=
sorry

end unobstructed_sight_l57_57447


namespace find_pq_l57_57441

noncomputable def p_and_q (p q : ℝ) := 
  (Complex.I * 2 - 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0} ∧ 
  - (Complex.I * 2 + 3) ∈ {z : Complex | z^2 * 2 + z * (p : Complex) + (q : Complex) = 0}

theorem find_pq : ∃ (p q : ℝ), p_and_q p q ∧ p + q = 38 :=
by
  sorry

end find_pq_l57_57441


namespace fair_die_multiple_of_2_probability_l57_57902

theorem fair_die_multiple_of_2_probability :
  let outcomes := {1, 2, 3, 4, 5, 6} in
  let favorable := {n ∈ outcomes | n % 2 = 0} in
  let total_outcomes := outcomes.card in
  let favorable_outcomes := favorable.size in
  total_outcomes = 6 -> 
  favorable_outcomes = 3 ->
  (favorable_outcomes / total_outcomes : ℚ) = 1 / 2 :=
by
  intros outcomes favorable total_outcomes favorable_outcomes h1 h2
  sorry

end fair_die_multiple_of_2_probability_l57_57902


namespace simplify_subtracted_terms_l57_57870

theorem simplify_subtracted_terms (r : ℝ) : 180 * r - 88 * r = 92 * r := 
by 
  sorry

end simplify_subtracted_terms_l57_57870


namespace smallest_positive_real_l57_57433

theorem smallest_positive_real (x : ℝ) (h₁ : ∃ y : ℝ, y > 0 ∧ ⌊y^2⌋ - y * ⌊y⌋ = 4) : x = 29 / 5 :=
by
  sorry

end smallest_positive_real_l57_57433


namespace vertex_of_parabola_l57_57032

theorem vertex_of_parabola : 
  (exists (a b: ℝ), ∀ x: ℝ, (a * (x - 1)^2 + b = (x - 1)^2 - 2)) → (1, -2) = (1, -2) :=
by
  intro h
  sorry

end vertex_of_parabola_l57_57032


namespace johns_subtraction_l57_57889

theorem johns_subtraction : 
  ∀ (a : ℕ), 
  a = 40 → 
  (a - 1)^2 = a^2 - 79 := 
by 
  -- The proof is omitted as per instruction
  sorry

end johns_subtraction_l57_57889


namespace midpoint_in_segment_l57_57172

open Set Metric

theorem midpoint_in_segment (S : Set ℝ^2) (D : Set ℝ^2)
  (hS_nonempty : S.Nonempty) (hS_closed : IsClosed S)
  (hD_closed : IsClosed D) (hS_in_D : S ⊆ D)
  (hD_property : ∀ D', IsClosed D' → (S ⊆ D' → D ⊆ D')) :
  ∀ y ∈ D, ∃ z1 z2 ∈ S, y = (z1 + z2) / 2 :=
sorry

end midpoint_in_segment_l57_57172


namespace union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l57_57276

open Set

variables {α : Type*} (A B C : Set α)

-- Commutativity
theorem union_comm : A ∪ B = B ∪ A := sorry
theorem inter_comm : A ∩ B = B ∩ A := sorry

-- Associativity
theorem union_assoc : A ∪ (B ∪ C) = (A ∪ B) ∪ C := sorry
theorem inter_assoc : A ∩ (B ∩ C) = (A ∩ B) ∩ C := sorry

-- Distributivity
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) := sorry
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) := sorry

-- Idempotence
theorem union_idem : A ∪ A = A := sorry
theorem inter_idem : A ∩ A = A := sorry

-- De Morgan's Laws
theorem de_morgan_union : compl (A ∪ B) = compl A ∩ compl B := sorry
theorem de_morgan_inter : compl (A ∩ B) = compl A ∪ compl B := sorry

end union_comm_inter_comm_union_assoc_inter_assoc_inter_union_distrib_union_inter_distrib_union_idem_inter_idem_de_morgan_union_de_morgan_inter_l57_57276


namespace greatest_possible_integer_radius_l57_57837

theorem greatest_possible_integer_radius (r : ℕ) (h : ∀ (A : ℝ), A = Real.pi * (r : ℝ)^2 → A < 75 * Real.pi) : r ≤ 8 :=
by sorry

end greatest_possible_integer_radius_l57_57837


namespace smallest_sum_of_xy_l57_57974

namespace MathProof

theorem smallest_sum_of_xy (x y : ℕ) (hx : x ≠ y) (hxy : (1 : ℝ) / x + (1 : ℝ) / y = 1 / 12) : x + y = 49 :=
sorry

end MathProof

end smallest_sum_of_xy_l57_57974


namespace abs_sub_eq_five_l57_57182

theorem abs_sub_eq_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 :=
sorry

end abs_sub_eq_five_l57_57182


namespace count_multiples_6_or_8_not_both_l57_57463

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l57_57463


namespace isosceles_triangle_height_eq_four_times_base_l57_57357

theorem isosceles_triangle_height_eq_four_times_base (b h : ℝ) 
    (same_area : (b * 2 * b) = (1/2 * b * h)) : 
    h = 4 * b :=
by 
  -- sorry allows us to skip the proof steps
  sorry

end isosceles_triangle_height_eq_four_times_base_l57_57357


namespace ninth_grade_class_notification_l57_57212

theorem ninth_grade_class_notification (n : ℕ) (h1 : 1 + n + n * n = 43) : n = 6 :=
by
  sorry

end ninth_grade_class_notification_l57_57212


namespace custom_op_4_8_l57_57029

-- Definition of the custom operation
def custom_op (a b : ℕ) : ℕ := b + b / a

-- Theorem stating the desired equality
theorem custom_op_4_8 : custom_op 4 8 = 10 :=
by
  -- Proof is omitted
  sorry

end custom_op_4_8_l57_57029


namespace total_trash_pieces_l57_57220

theorem total_trash_pieces (classroom_trash : ℕ) (outside_trash : ℕ)
  (h1 : classroom_trash = 344) (h2 : outside_trash = 1232) : 
  classroom_trash + outside_trash = 1576 :=
by
  sorry

end total_trash_pieces_l57_57220


namespace train_stop_time_l57_57908

theorem train_stop_time : 
  let speed_exc_stoppages := 45.0
  let speed_inc_stoppages := 31.0
  let speed_diff := speed_exc_stoppages - speed_inc_stoppages
  let km_per_minute := speed_exc_stoppages / 60.0
  let stop_time := speed_diff / km_per_minute
  stop_time = 18.67 :=
  by
    sorry

end train_stop_time_l57_57908


namespace polynomials_divisibility_l57_57855

variable (R : Type*) [CommRing R]
variable (f g h k : R[X])

theorem polynomials_divisibility
  (H1 : (X^2 + 1) * h + (X - 1) * f + (X - 2) * g = 0)
  (H2 : (X^2 + 1) * k + (X + 1) * f + (X + 2) * g = 0) :
  (X^2 + 1) ∣ (f * g) :=
by
  sorry

end polynomials_divisibility_l57_57855


namespace pascal_fifth_number_l57_57492

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57492


namespace intersection_M_N_l57_57856

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x * x = x}

theorem intersection_M_N :
  M ∩ N = {0, 1} :=
sorry

end intersection_M_N_l57_57856


namespace Mark_bill_total_l57_57069

theorem Mark_bill_total
  (original_bill : ℝ)
  (first_late_charge_rate : ℝ)
  (second_late_charge_rate : ℝ)
  (after_first_late_charge : ℝ)
  (final_total : ℝ) :
  original_bill = 500 ∧
  first_late_charge_rate = 0.02 ∧
  second_late_charge_rate = 0.02 ∧
  after_first_late_charge = original_bill * (1 + first_late_charge_rate) ∧
  final_total = after_first_late_charge * (1 + second_late_charge_rate) →
  final_total = 520.20 := by
  sorry

end Mark_bill_total_l57_57069


namespace solution_set_of_f_greater_than_one_l57_57978

theorem solution_set_of_f_greater_than_one (f : ℝ → ℝ) (h_inv : ∀ x, f (x / (x + 3)) = x) :
  {x | f x > 1} = {x | 1 / 4 < x ∧ x < 1} :=
by
  sorry

end solution_set_of_f_greater_than_one_l57_57978


namespace white_truck_percentage_is_17_l57_57094

-- Define the conditions
def total_trucks : ℕ := 50
def total_cars : ℕ := 40
def total_vehicles : ℕ := total_trucks + total_cars

def red_trucks : ℕ := total_trucks / 2
def black_trucks : ℕ := (total_trucks * 20) / 100
def white_trucks : ℕ := total_trucks - red_trucks - black_trucks

def percentage_white_trucks : ℕ := (white_trucks * 100) / total_vehicles

theorem white_truck_percentage_is_17 :
  percentage_white_trucks = 17 :=
  by sorry

end white_truck_percentage_is_17_l57_57094


namespace reporters_percentage_l57_57945

theorem reporters_percentage (total_reporters : ℕ) (local_politics_percentage : ℝ) (non_politics_percentage : ℝ) :
  local_politics_percentage = 28 → non_politics_percentage = 60 → 
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  100 * (non_local_political_reporters / political_reporters) = 30 :=
by
  intros
  let political_reporters := total_reporters * ((100 - non_politics_percentage) / 100)
  let local_political_reporters := total_reporters * (local_politics_percentage / 100)
  let non_local_political_reporters := political_reporters - local_political_reporters
  sorry

end reporters_percentage_l57_57945


namespace brooke_social_studies_problems_l57_57090

theorem brooke_social_studies_problems :
  ∀ (math_problems science_problems total_minutes : Nat) 
    (math_time_per_problem science_time_per_problem soc_studies_time_per_problem : Nat)
    (soc_studies_problems : Nat),
  math_problems = 15 →
  science_problems = 10 →
  total_minutes = 48 →
  math_time_per_problem = 2 →
  science_time_per_problem = 3 / 2 → -- converting 1.5 minutes to a fraction
  soc_studies_time_per_problem = 1 / 2 → -- converting 30 seconds to a fraction
  math_problems * math_time_per_problem + science_problems * science_time_per_problem + soc_studies_problems * soc_studies_time_per_problem = 48 →
  soc_studies_problems = 6 :=
by
  intros math_problems science_problems total_minutes math_time_per_problem science_time_per_problem soc_studies_time_per_problem soc_studies_problems
  intros h_math_problems h_science_problems h_total_minutes h_math_time_per_problem h_science_time_per_problem h_soc_studies_time_per_problem h_eq
  sorry

end brooke_social_studies_problems_l57_57090


namespace sales_tax_rate_l57_57661

-- Given conditions
def cost_of_video_game : ℕ := 50
def weekly_allowance : ℕ := 10
def weekly_savings : ℕ := weekly_allowance / 2
def weeks_to_save : ℕ := 11
def total_savings : ℕ := weeks_to_save * weekly_savings

-- Proof problem statement
theorem sales_tax_rate : 
  total_savings - cost_of_video_game = (cost_of_video_game * 10) / 100 := by
  sorry

end sales_tax_rate_l57_57661


namespace complement_of_P_in_U_l57_57442

/-- Definitions of sets U and P -/
def U := { y : ℝ | ∃ x : ℝ, x > 1 ∧ y = Real.log x / Real.log 2 }
def P := { y : ℝ | ∃ x : ℝ, x > 2 ∧ y = 1 / x }

/-- The complement of P in U -/
def complement_U_P := { y : ℝ | y = 0 ∨ y ≥ 1 / 2 }

/-- Proving the complement of P in U is as expected -/
theorem complement_of_P_in_U : { y : ℝ | y ∈ U ∧ y ∉ P } = complement_U_P := by
  sorry

end complement_of_P_in_U_l57_57442


namespace no_ordered_triples_l57_57245

theorem no_ordered_triples (x y z : ℕ)
  (h1 : 1 ≤ x) (h2 : x ≤ y) (h3 : y ≤ z) :
  x * y * z + 2 * (x * y + y * z + z * x) ≠ 2 * (2 * (x * y + y * z + z * x)) + 12 :=
by {
  sorry
}

end no_ordered_triples_l57_57245


namespace more_oranges_than_apples_l57_57167

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l57_57167


namespace composite_divisible_by_six_l57_57810

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end composite_divisible_by_six_l57_57810


namespace whale_crossing_time_l57_57897

theorem whale_crossing_time
  (speed_fast : ℝ)
  (speed_slow : ℝ)
  (length_slow : ℝ)
  (h_fast : speed_fast = 18)
  (h_slow : speed_slow = 15)
  (h_length : length_slow = 45) :
  (length_slow / (speed_fast - speed_slow) = 15) :=
by
  sorry

end whale_crossing_time_l57_57897


namespace composite_divisible_by_six_l57_57809

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end composite_divisible_by_six_l57_57809


namespace gcd_of_B_is_2_l57_57640

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l57_57640


namespace find_extrema_A_l57_57256

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end find_extrema_A_l57_57256


namespace gate_perimeter_l57_57400

theorem gate_perimeter (r : ℝ) (theta : ℝ) (h1 : r = 2) (h2 : theta = π / 2) :
  let arc_length := (3 / 4) * (2 * π * r)
  let radii_length := 2 * r
  arc_length + radii_length = 3 * π + 4 :=
by
  simp [h1, h2]
  sorry

end gate_perimeter_l57_57400


namespace pascal_triangle_fifth_number_l57_57608

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57608


namespace total_number_of_red_and_white_jelly_beans_in_fishbowl_l57_57621

def number_of_red_jelly_beans_in_bag := 24
def number_of_white_jelly_beans_in_bag := 18
def number_of_bags := 3

theorem total_number_of_red_and_white_jelly_beans_in_fishbowl :
  number_of_red_jelly_beans_in_bag * number_of_bags + number_of_white_jelly_beans_in_bag * number_of_bags = 126 := by
  sorry

end total_number_of_red_and_white_jelly_beans_in_fishbowl_l57_57621


namespace universal_proposition_example_l57_57229

theorem universal_proposition_example :
  (∀ n : ℕ, n % 2 = 0 → ∃ k : ℕ, n = 2 * k) :=
sorry

end universal_proposition_example_l57_57229


namespace pascal_row_fifth_number_l57_57536

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57536


namespace peter_total_pizza_eaten_l57_57331

def slices_total : Nat := 16
def peter_slices_eaten_alone : ℚ := 2 / 16
def shared_slice_total : ℚ := 1 / (3 * 16)

theorem peter_total_pizza_eaten : peter_slices_eaten_alone + shared_slice_total = 7 / 48 := by
  sorry

end peter_total_pizza_eaten_l57_57331


namespace card_problem_l57_57071

-- Define the variables
variables (x y : ℕ)

-- Conditions given in the problem
theorem card_problem 
  (h1 : x - 1 = y + 1) 
  (h2 : x + 1 = 2 * (y - 1)) : 
  x + y = 12 :=
sorry

end card_problem_l57_57071


namespace no_minus_three_in_range_l57_57957

theorem no_minus_three_in_range (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 3 ≠ -3) ↔ b^2 < 24 :=
by
  sorry

end no_minus_three_in_range_l57_57957


namespace largest_pies_without_ingredients_l57_57269

variable (total_pies : ℕ) (chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
variable (b : total_pies = 36)
variable (c : chocolate_pies = total_pies / 2)
variable (m : marshmallow_pies = 2 * total_pies / 3)
variable (k : cayenne_pies = 3 * total_pies / 4)
variable (s : soy_nut_pies = total_pies / 6)

theorem largest_pies_without_ingredients (total_pies chocolate_pies marshmallow_pies cayenne_pies soy_nut_pies : ℕ)
  (b : total_pies = 36)
  (c : chocolate_pies = total_pies / 2)
  (m : marshmallow_pies = 2 * total_pies / 3)
  (k : cayenne_pies = 3 * total_pies / 4)
  (s : soy_nut_pies = total_pies / 6) :
  9 = total_pies - chocolate_pies - marshmallow_pies - cayenne_pies - soy_nut_pies + 3 * cayenne_pies := 
by
  sorry

end largest_pies_without_ingredients_l57_57269


namespace pascal_triangle_fifth_number_l57_57604

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57604


namespace wire_weight_l57_57081

theorem wire_weight (w : ℕ → ℕ) (h_proportional : ∀ (x y : ℕ), w (x + y) = w x + w y) : 
  (w 25 = 5) → w 75 = 15 :=
by
  intro h1
  sorry

end wire_weight_l57_57081


namespace solve_for_p_l57_57215

variable (p q : ℝ)
noncomputable def binomial_third_term : ℝ := 55 * p^9 * q^2
noncomputable def binomial_fourth_term : ℝ := 165 * p^8 * q^3

theorem solve_for_p (h1 : p + q = 1) (h2 : binomial_third_term p q = binomial_fourth_term p q) : p = 3 / 4 :=
by sorry

end solve_for_p_l57_57215


namespace smallest_base_for_100_l57_57707

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end smallest_base_for_100_l57_57707


namespace max_number_of_small_boxes_l57_57062

def volume_of_large_box (length width height : ℕ) : ℕ :=
  length * width * height

def volume_of_small_box (length width height : ℕ) : ℕ :=
  length * width * height

def number_of_small_boxes (large_volume small_volume : ℕ) : ℕ :=
  large_volume / small_volume

theorem max_number_of_small_boxes :
  let large_box_length := 4 * 100  -- in cm
  let large_box_width := 2 * 100  -- in cm
  let large_box_height := 4 * 100  -- in cm
  let small_box_length := 4  -- in cm
  let small_box_width := 2  -- in cm
  let small_box_height := 2  -- in cm
  let large_volume := volume_of_large_box large_box_length large_box_width large_box_height
  let small_volume := volume_of_small_box small_box_length small_box_width small_box_height
  number_of_small_boxes large_volume small_volume = 2000000 := by
  -- Prove the statement
  sorry

end max_number_of_small_boxes_l57_57062


namespace passengers_landed_in_newberg_last_year_l57_57846

theorem passengers_landed_in_newberg_last_year :
  let airport_a_on_time : ℕ := 16507
  let airport_a_late : ℕ := 256
  let airport_b_on_time : ℕ := 11792
  let airport_b_late : ℕ := 135
  airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690 :=
by
  let airport_a_on_time := 16507
  let airport_a_late := 256
  let airport_b_on_time := 11792
  let airport_b_late := 135
  show airport_a_on_time + airport_a_late + airport_b_on_time + airport_b_late = 28690
  sorry

end passengers_landed_in_newberg_last_year_l57_57846


namespace proof_math_problem_lean_l57_57465

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l57_57465


namespace factorial_multiple_of_3_l57_57282

theorem factorial_multiple_of_3 (n : ℤ) (h : n ≥ 9) : 3 ∣ (n+1) * (n+3) :=
sorry

end factorial_multiple_of_3_l57_57282


namespace opposite_neg_two_is_two_l57_57358

theorem opposite_neg_two_is_two : -(-2) = 2 :=
by
  sorry

end opposite_neg_two_is_two_l57_57358


namespace find_tricias_age_l57_57374

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l57_57374


namespace find_percentage_l57_57136

theorem find_percentage (P N : ℝ) (h1 : (P / 100) * N = 60) (h2 : 0.80 * N = 240) : P = 20 :=
sorry

end find_percentage_l57_57136


namespace find_x_value_l57_57281

theorem find_x_value (x : ℝ) (h : (55 + 113 / x) * x = 4403) : x = 78 :=
sorry

end find_x_value_l57_57281


namespace area_of_quadrilateral_is_correct_l57_57913

noncomputable def area_of_quadrilateral_BGFAC : ℝ :=
  let a := 3 -- side of the equilateral triangle
  let triangle_area := (a^2 * Real.sqrt 3) / 4 -- area of ABC
  let ratio_AG_GC := 2 -- ratio AG:GC = 2:1
  let area_AGC := triangle_area / 3 -- area of triangle AGC
  let area_BGC := triangle_area / 3 -- area of triangle BGC
  let area_BFC := (2 : ℝ) * triangle_area / 3 -- area of triangle BFC
  let area_BGFC := area_BGC + area_BFC -- area of quadrilateral BGFC
  area_BGFC

theorem area_of_quadrilateral_is_correct :
  area_of_quadrilateral_BGFAC = (3 * Real.sqrt 3) / 2 :=
by
  -- Proof will be provided here
  sorry

end area_of_quadrilateral_is_correct_l57_57913


namespace sufficient_but_not_necessary_condition_l57_57237

-- Define the condition
variable (a : ℝ)

-- Theorem statement: $a > 0$ is a sufficient but not necessary condition for $a^2 > 0$
theorem sufficient_but_not_necessary_condition : 
  (a > 0 → a^2 > 0) ∧ (¬ (a > 0) → a^2 > 0) :=
  by
    sorry

end sufficient_but_not_necessary_condition_l57_57237


namespace gcd_of_B_is_two_l57_57645

-- Definition of the set B
def B : Set ℤ := { n | ∃ x : ℤ, n = 4 * x + 2 }

-- Function to find the gcd of all elements in B
noncomputable def gcd_B : ℤ := Nat.gcd 2 -- gcd of 2(2x + 1) and any integer factorable by 2

-- Lean statement to prove gcd_B equals 2
theorem gcd_of_B_is_two : gcd_B = 2 := by
  sorry

end gcd_of_B_is_two_l57_57645


namespace smallest_base_for_100_l57_57709

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end smallest_base_for_100_l57_57709


namespace longer_string_length_l57_57724

theorem longer_string_length 
  (total_length : ℕ) 
  (length_diff : ℕ)
  (h_total_length : total_length = 348)
  (h_length_diff : length_diff = 72) :
  ∃ (L S : ℕ), 
  L - S = length_diff ∧
  L + S = total_length ∧ 
  L = 210 :=
by
  sorry

end longer_string_length_l57_57724


namespace johann_ate_ten_oranges_l57_57845

variable (x : ℕ)
variable (y : ℕ)

def johann_initial_oranges := 60

def johann_remaining_after_eating := johann_initial_oranges - x

def johann_remaining_after_theft := (johann_remaining_after_eating / 2)

def johann_remaining_after_return := johann_remaining_after_theft + 5

theorem johann_ate_ten_oranges (h : johann_remaining_after_return = 30) : x = 10 :=
by
  sorry

end johann_ate_ten_oranges_l57_57845


namespace probability_red_red_red_l57_57386

-- Definition of probability for picking three red balls without replacement
def total_balls := 21
def red_balls := 7
def blue_balls := 9
def green_balls := 5

theorem probability_red_red_red : 
  (red_balls / total_balls) * ((red_balls - 1) / (total_balls - 1)) * ((red_balls - 2) / (total_balls - 2)) = 1 / 38 := 
by sorry

end probability_red_red_red_l57_57386


namespace car_arrives_first_and_earlier_l57_57107

-- Define the conditions
def total_intersections : ℕ := 11
def total_blocks : ℕ := 12
def green_time : ℕ := 3
def red_time : ℕ := 1
def car_block_time : ℕ := 1
def bus_block_time : ℕ := 2

-- Define the functions that compute the travel times
def car_travel_time (blocks : ℕ) : ℕ :=
  (blocks / 3) * (green_time + red_time) + (blocks % 3 * car_block_time)

def bus_travel_time (blocks : ℕ) : ℕ :=
  blocks * bus_block_time

-- Define the theorem to prove
theorem car_arrives_first_and_earlier :
  car_travel_time total_blocks < bus_travel_time total_blocks ∧
  bus_travel_time total_blocks - car_travel_time total_blocks = 9 := 
by
  sorry

end car_arrives_first_and_earlier_l57_57107


namespace third_beats_seventh_l57_57998

-- Definitions and conditions
variable (points : Fin 8 → ℕ)
variable (distinct_points : Function.Injective points)
variable (sum_last_four : points 1 = points 4 + points 5 + points 6 + points 7)

-- Proof statement
theorem third_beats_seventh 
  (h_distinct : ∀ i j, i ≠ j → points i ≠ points j)
  (h_sum : points 1 = points 4 + points 5 + points 6 + points 7) :
  points 2 > points 6 :=
sorry

end third_beats_seventh_l57_57998


namespace sin_double_angle_l57_57989

theorem sin_double_angle {θ : ℝ} (h : Real.tan θ = 1 / 3) : 
  Real.sin (2 * θ) = 3 / 5 := 
  sorry

end sin_double_angle_l57_57989


namespace pascal_fifth_number_in_row_15_l57_57554

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57554


namespace fifth_number_in_pascal_row_l57_57617

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57617


namespace bn_six_eight_product_l57_57843

noncomputable def sequence_an (n : ℕ) : ℝ := sorry  -- given that an is an arithmetic sequence and an ≠ 0
noncomputable def sequence_bn (n : ℕ) : ℝ := sorry  -- given that bn is a geometric sequence

theorem bn_six_eight_product :
  (∀ n : ℕ, sequence_an n ≠ 0) →
  2 * sequence_an 3 - sequence_an 7 ^ 2 + 2 * sequence_an 11 = 0 →
  sequence_bn 7 = sequence_an 7 →
  sequence_bn 6 * sequence_bn 8 = 16 :=
sorry

end bn_six_eight_product_l57_57843


namespace number_of_10_digit_integers_with_consecutive_twos_l57_57128

open Nat

-- Define the total number of 10-digit integers using only '1' and '2's
def total_10_digit_numbers : ℕ := 2^10

-- Define the Fibonacci function
def fibonacci : ℕ → ℕ
| 0    => 1
| 1    => 2
| n+2  => fibonacci (n+1) + fibonacci n

-- Calculate the 10th Fibonacci number for the problem context
def F_10 : ℕ := fibonacci 9 + fibonacci 8

-- Prove that the number of 10-digit integers with at least one pair of consecutive '2's is 880
theorem number_of_10_digit_integers_with_consecutive_twos :
  total_10_digit_numbers - F_10 = 880 :=
by
  sorry

end number_of_10_digit_integers_with_consecutive_twos_l57_57128


namespace gcd_45123_32768_l57_57948

theorem gcd_45123_32768 : Nat.gcd 45123 32768 = 1 := by
  sorry

end gcd_45123_32768_l57_57948


namespace solve_fractional_equation_l57_57342

theorem solve_fractional_equation : 
  ∀ x : ℝ, x = 2 → (2 - x) / (x - 3) + 1 / (3 - x) = 1 :=
by
  intro x hx
  rw hx
  simp
  sorry

end solve_fractional_equation_l57_57342


namespace sum_of_consecutive_integers_eq_pow_of_two_l57_57947

theorem sum_of_consecutive_integers_eq_pow_of_two (n : ℕ) : 
  (∀ a b : ℕ, a < b → 2 * n ≠ (a + b) * (b - a + 1)) ↔ ∃ k : ℕ, n = 2 ^ k := 
sorry

end sum_of_consecutive_integers_eq_pow_of_two_l57_57947


namespace proof_math_problem_lean_l57_57466

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l57_57466


namespace multiples_count_l57_57455

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l57_57455


namespace gcd_factorial_8_6_squared_l57_57774

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end gcd_factorial_8_6_squared_l57_57774


namespace cuboid_length_l57_57676

variable (L W H V : ℝ)

theorem cuboid_length (W_eq : W = 4) (H_eq : H = 6) (V_eq : V = 96) (Volume_eq : V = L * W * H) : L = 4 :=
by
  sorry

end cuboid_length_l57_57676


namespace fractional_eq_solution_l57_57337

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l57_57337


namespace sequence_term_is_correct_l57_57122

theorem sequence_term_is_correct : ∀ (n : ℕ), (n = 7) → (2 * Real.sqrt 5 = Real.sqrt (3 * n - 1)) :=
by
  sorry

end sequence_term_is_correct_l57_57122


namespace range_of_m_l57_57830

open Classical

variable {m : ℝ}

def p (m : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * x + m ≤ 0
def q (m : ℝ) : Prop := ∀ x : ℝ, (3 - m) > 1 → ((3 - m) ^ x > 0)

theorem range_of_m (hm : (p m ∨ q m) ∧ ¬(p m ∧ q m)) : 1 < m ∧ m < 2 :=
  sorry

end range_of_m_l57_57830


namespace arithmetic_sequence_120th_term_l57_57703

theorem arithmetic_sequence_120th_term :
  let a1 := 6
  let d := 6
  let n := 120
  let a_n := a1 + (n - 1) * d
  a_n = 720 := by
  sorry

end arithmetic_sequence_120th_term_l57_57703


namespace nat_diff_same_prime_divisors_l57_57018

/-- Every natural number can be expressed as the difference of two natural numbers that have the same number of prime divisors. -/
theorem nat_diff_same_prime_divisors (n : ℕ) : 
  ∃ a b : ℕ, (a - b = n) ∧ (card a.prime_divisors = card b.prime_divisors) := 
sorry

end nat_diff_same_prime_divisors_l57_57018


namespace pyramid_edges_l57_57364

-- Define the conditions
def isPyramid (n : ℕ) : Prop :=
  (n + 1) + (n + 1) = 16

-- Statement to be proved
theorem pyramid_edges : ∃ (n : ℕ), isPyramid n ∧ 2 * n = 14 :=
by {
  sorry
}

end pyramid_edges_l57_57364


namespace pascal_fifth_element_15th_row_l57_57595

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57595


namespace pascal_15_5th_number_l57_57526

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57526


namespace composite_divisible_by_six_l57_57807

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end composite_divisible_by_six_l57_57807


namespace multiples_six_or_eight_not_both_l57_57467

def countMultiples (n m : ℕ) : ℕ := n / m

def LCM (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem multiples_six_or_eight_not_both : 
  let multiplesSix := countMultiples 200 6
  let multiplesEight := countMultiples 200 8
  let commonMultiple := countMultiples 200 (LCM 6 8)
  multiplesSix - commonMultiple + multiplesEight - commonMultiple = 42 := 
by
  sorry

end multiples_six_or_eight_not_both_l57_57467


namespace remainder_of_122_div_20_l57_57330

theorem remainder_of_122_div_20 :
  (∃ (q r : ℕ), 122 = 20 * q + r ∧ r < 20 ∧ q = 6) →
  r = 2 :=
by
  sorry

end remainder_of_122_div_20_l57_57330


namespace find_a8_l57_57206

noncomputable def a (n : ℕ) : ℤ := sorry

noncomputable def b (n : ℕ) : ℤ := a (n + 1) - a n

theorem find_a8 :
  (a 1 = 3) ∧
  (∀ n : ℕ, b n = b 1 + n * 2) ∧
  (b 3 = -2) ∧
  (b 10 = 12) →
  a 8 = 3 :=
by sorry

end find_a8_l57_57206


namespace triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l57_57287

variable {a b c x : ℝ}

-- Part (1)
theorem triangle_ABC_is_isosceles (h : (a + b) * 1 ^ 2 - 2 * c * 1 + (a - b) = 0) : a = c :=
by 
  -- Proof omitted
  sorry

-- Part (2)
theorem roots_of_quadratic_for_equilateral (h_eq : a = b ∧ b = c ∧ c = a) : 
  (∀ x : ℝ, (a + a) * x ^ 2 - 2 * a * x + (a - a) = 0 → (x = 0 ∨ x = 1)) :=
by 
  -- Proof omitted
  sorry

end triangle_ABC_is_isosceles_roots_of_quadratic_for_equilateral_l57_57287


namespace largest_divisor_of_n_pow4_minus_n_for_composites_l57_57792

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end largest_divisor_of_n_pow4_minus_n_for_composites_l57_57792


namespace man_l57_57403

theorem man's_age (x : ℕ) : 6 * (x + 6) - 6 * (x - 6) = x → x = 72 :=
by
  sorry

end man_l57_57403


namespace pascal_fifth_number_in_row_15_l57_57553

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57553


namespace gcd_factorial_8_and_6_squared_l57_57770

-- We will use nat.factorial and other utility functions from Mathlib

noncomputable def factorial_8 : ℕ := nat.factorial 8
noncomputable def factorial_6_squared : ℕ := (nat.factorial 6) ^ 2

theorem gcd_factorial_8_and_6_squared : nat.gcd factorial_8 factorial_6_squared = 1440 := by
  -- We'll need Lean to compute prime factorizations
  sorry

end gcd_factorial_8_and_6_squared_l57_57770


namespace resized_height_l57_57426

-- Define original dimensions
def original_width : ℝ := 4.5
def original_height : ℝ := 3

-- Define new width
def new_width : ℝ := 13.5

-- Define new height to be proven
def new_height : ℝ := 9

-- Theorem statement
theorem resized_height :
  (new_width / original_width) * original_height = new_height :=
by
  -- The statement that equates the new height calculated proportionately to 9
  sorry

end resized_height_l57_57426


namespace problem_solution_includes_024_l57_57878

theorem problem_solution_includes_024 (x : ℝ) :
  (2 * 88 * (abs (abs (abs (abs (x - 1) - 1) - 1) - 1)) = 0) →
  x = 0 ∨ x = 2 ∨ x = 4 :=
by
  sorry

end problem_solution_includes_024_l57_57878


namespace pascal_row_fifth_number_l57_57532

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57532


namespace pascal_triangle_fifth_number_l57_57583

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57583


namespace solution_set_ineq_l57_57952

theorem solution_set_ineq (x : ℝ) : 
  (x - 1) / (2 * x + 3) > 1 ↔ -4 < x ∧ x < -3 / 2 :=
by
  sorry

end solution_set_ineq_l57_57952


namespace minimum_value_of_a_l57_57966

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (Real.exp (3 * Real.log x - x)) - x^2 - (a - 4) * x - 4

theorem minimum_value_of_a (h : ∀ x > 0, f x ≤ 0) : a ≥ 4 / Real.exp 2 := by
  sorry

end minimum_value_of_a_l57_57966


namespace Pascal_triangle_fifth_number_l57_57561

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57561


namespace notebooks_last_days_l57_57321

/-- John buys 5 notebooks, each with 40 pages. 
    He uses 4 pages per day. 
    Prove the notebooks last 50 days. -/
theorem notebooks_last_days : 
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  (notebooks * pages_per_notebook) / pages_per_day = 50 := 
by
  -- Definitions
  let notebooks := 5
  let pages_per_notebook := 40
  let pages_per_day := 4
  calc
    (notebooks * pages_per_notebook) / pages_per_day
      = (5 * 40) / 4 : by rfl
      ... = 200 / 4 : by rfl
      ... = 50 : by rfl

end notebooks_last_days_l57_57321


namespace Tricia_is_five_years_old_l57_57370

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l57_57370


namespace circle_radius_l57_57396

-- Parameters of the problem
variables (k : ℝ) (r : ℝ)
-- Conditions
axiom cond_k_positive : k > 8
axiom tangency_y_8 : r = k - 8
axiom tangency_y_x : r = k / (Real.sqrt 2)

-- Statement to prove
theorem circle_radius (k : ℝ) (hk : k > 8) (r : ℝ) (hr1 : r = k - 8) (hr2 : r = k / (Real.sqrt 2)) : r = 8 * Real.sqrt 2 + 8 :=
sorry

end circle_radius_l57_57396


namespace largest_divisor_of_difference_between_n_and_n4_l57_57802

theorem largest_divisor_of_difference_between_n_and_n4 (n : ℤ) (h_composite : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ p * q = n) :
  ∃ k : ℤ, k = 6 ∧ k ∣ (n^4 - n) :=
begin
  sorry
end

end largest_divisor_of_difference_between_n_and_n4_l57_57802


namespace solve_for_a_l57_57821

theorem solve_for_a (x y a : ℤ) (h1 : x = 1) (h2 : y = 2) (h3 : x - a * y = 3) : a = -1 :=
by
  -- Proof is skipped
  sorry

end solve_for_a_l57_57821


namespace sequence_B_is_arithmetic_l57_57903

-- Definitions of the sequences
def S_n (n : ℕ) : ℕ := 2*n + 1

-- Theorem statement
theorem sequence_B_is_arithmetic : ∀ n : ℕ, S_n (n + 1) - S_n n = 2 :=
by
  intro n
  sorry

end sequence_B_is_arithmetic_l57_57903


namespace n_squared_plus_d_not_square_l57_57906

theorem n_squared_plus_d_not_square 
  (n : ℕ) (d : ℕ)
  (h_pos_n : n > 0) 
  (h_pos_d : d > 0) 
  (h_div : d ∣ 2 * n^2) : 
  ¬ ∃ m : ℕ, n^2 + d = m^2 := 
sorry

end n_squared_plus_d_not_square_l57_57906


namespace concentric_circles_false_statement_l57_57305

theorem concentric_circles_false_statement
  (a b c : ℝ)
  (h1 : a < b)
  (h2 : b < c) :
  ¬ (b + a = c + b) :=
sorry

end concentric_circles_false_statement_l57_57305


namespace max_elements_set_M_l57_57121

theorem max_elements_set_M (n : ℕ) (hn : n ≥ 2) (M : Finset (ℕ × ℕ))
  (hM : ∀ {i k}, (i, k) ∈ M → i < k → ∀ {m}, k < m → (k, m) ∉ M) :
  M.card ≤ n^2 / 4 :=
sorry

end max_elements_set_M_l57_57121


namespace tel_aviv_rain_probability_l57_57354

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l57_57354


namespace cape_may_multiple_l57_57420

theorem cape_may_multiple :
  ∃ x : ℕ, 26 = x * 7 + 5 ∧ x = 3 :=
by
  sorry

end cape_may_multiple_l57_57420


namespace gcd_factorial_eight_six_sq_l57_57782

/-- Prove that the gcd of 8! and (6!)^2 is 11520 -/
theorem gcd_factorial_eight_six_sq : Int.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 11520 :=
by
  -- We'll put the proof here, but for now, we use sorry to omit it.
  sorry

end gcd_factorial_eight_six_sq_l57_57782


namespace ratio_of_money_with_Gopal_and_Krishan_l57_57362

theorem ratio_of_money_with_Gopal_and_Krishan 
  (R G K : ℕ) 
  (h1 : R = 735) 
  (h2 : K = 4335) 
  (h3 : R * 17 = G * 7) :
  G * 4335 = 1785 * K :=
by
  sorry

end ratio_of_money_with_Gopal_and_Krishan_l57_57362


namespace polynomial_coeff_sum_abs_l57_57393

theorem polynomial_coeff_sum_abs (a a_1 a_2 a_3 a_4 a_5 : ℤ) :
    (2 * x - 1)^5 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 →
    |a_1| + |a_2| + |a_3| + |a_4| + |a_5| = 242 := by 
  sorry

end polynomial_coeff_sum_abs_l57_57393


namespace a_le_neg4_l57_57828

noncomputable def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x
noncomputable def g (a x : ℝ) : ℝ := a * Real.log x

noncomputable def h (a x : ℝ) : ℝ := f x - g a x

-- Theorem
theorem a_le_neg4 (a : ℝ) : 
  (∀ (x1 x2 : ℝ), x1 ≠ x2 → x1 > 0 → x2 > 0 → (h a x1 - h a x2) / (x1 - x2) > 2) →
  a ≤ -4 :=
by
  sorry

end a_le_neg4_l57_57828


namespace final_number_correct_l57_57915

noncomputable def initial_number : ℝ := 1256
noncomputable def first_increase_rate : ℝ := 3.25
noncomputable def second_increase_rate : ℝ := 1.47

theorem final_number_correct :
  initial_number * first_increase_rate * second_increase_rate = 6000.54 := 
by
  sorry

end final_number_correct_l57_57915


namespace fifth_number_in_pascals_triangle_l57_57592

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57592


namespace citizen_income_l57_57231

theorem citizen_income (I : ℝ) 
  (h1 : I > 0)
  (h2 : 0.12 * 40000 + 0.20 * (I - 40000) = 8000) : 
  I = 56000 := 
sorry

end citizen_income_l57_57231


namespace min_red_chips_l57_57482

theorem min_red_chips (w b r : ℕ) 
  (h1 : b ≥ w / 3) 
  (h2 : b ≤ r / 4) 
  (h3 : w + b ≥ 72) :
  72 ≤ r :=
by
  sorry

end min_red_chips_l57_57482


namespace pascal_fifteen_four_l57_57572

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57572


namespace discriminant_of_quad_eq_l57_57379

def a : ℕ := 5
def b : ℕ := 8
def c : ℤ := -6

def discriminant (a b c : ℤ) : ℤ := b^2 - 4 * a * c

theorem discriminant_of_quad_eq : discriminant 5 8 (-6) = 184 :=
by
  -- The proof is skipped
  sorry

end discriminant_of_quad_eq_l57_57379


namespace large_block_volume_correct_l57_57073

def normal_block_volume (w d l : ℝ) : ℝ := w * d * l

def large_block_volume (w d l : ℝ) : ℝ := (2 * w) * (2 * d) * (3 * l)

theorem large_block_volume_correct (w d l : ℝ) (h : normal_block_volume w d l = 3) :
  large_block_volume w d l = 36 :=
by sorry

end large_block_volume_correct_l57_57073


namespace total_songs_time_l57_57690

-- Definitions of durations for each radio show
def duration_show1 : ℕ := 180
def duration_show2 : ℕ := 240
def duration_show3 : ℕ := 120

-- Definitions of talking segments for each show
def talking_segments_show1 : ℕ := 3 * 10  -- 3 segments, 10 minutes each
def talking_segments_show2 : ℕ := 4 * 15  -- 4 segments, 15 minutes each
def talking_segments_show3 : ℕ := 2 * 8   -- 2 segments, 8 minutes each

-- Definitions of ad breaks for each show
def ad_breaks_show1 : ℕ := 5 * 5  -- 5 breaks, 5 minutes each
def ad_breaks_show2 : ℕ := 6 * 4  -- 6 breaks, 4 minutes each
def ad_breaks_show3 : ℕ := 3 * 6  -- 3 breaks, 6 minutes each

-- Function to calculate time spent on songs for a given show
def time_spent_on_songs (duration talking ad_breaks : ℕ) : ℕ :=
  duration - talking - ad_breaks

-- Total time spent on songs for all three shows
def total_time_spent_on_songs : ℕ :=
  time_spent_on_songs duration_show1 talking_segments_show1 ad_breaks_show1 +
  time_spent_on_songs duration_show2 talking_segments_show2 ad_breaks_show2 +
  time_spent_on_songs duration_show3 talking_segments_show3 ad_breaks_show3

-- The theorem we want to prove
theorem total_songs_time : total_time_spent_on_songs = 367 := 
  sorry

end total_songs_time_l57_57690


namespace intersection_A_B_l57_57962

def A (y : ℝ) : Prop := ∃ x : ℝ, y = -x^2 + 2*x - 1
def B (y : ℝ) : Prop := ∃ x : ℝ, y = 2*x + 1

theorem intersection_A_B :
  {y : ℝ | A y} ∩ {y : ℝ | B y} = {y : ℝ | y ≤ 0} :=
sorry

end intersection_A_B_l57_57962


namespace find_order_amount_l57_57411

noncomputable def unit_price : ℝ := 100

def discount_rate (x : ℕ) : ℝ :=
  if x < 250 then 0
  else if x < 500 then 0.05
  else if x < 1000 then 0.10
  else 0.15

theorem find_order_amount (T : ℝ) (x : ℕ)
    (hx : x = 980) (hT : T = 88200) :
  T = unit_price * x * (1 - discount_rate x) :=
by
  rw [hx, hT]
  sorry

end find_order_amount_l57_57411


namespace pascal_fifteen_four_l57_57573

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57573


namespace positive_difference_for_6_points_l57_57240

def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def positiveDifferenceTrianglesAndQuadrilaterals (n : ℕ) : ℕ :=
  combinations n 3 - combinations n 4

theorem positive_difference_for_6_points : positiveDifferenceTrianglesAndQuadrilaterals 6 = 5 :=
by
  sorry

end positive_difference_for_6_points_l57_57240


namespace hcf_of_two_numbers_l57_57355

theorem hcf_of_two_numbers (H : ℕ) 
(lcm_def : lcm a b = H * 13 * 14) 
(h : a = 280 ∨ b = 280) 
(is_factor_h : H ∣ 280) : 
H = 5 :=
sorry

end hcf_of_two_numbers_l57_57355


namespace gcd_factorial_l57_57759

theorem gcd_factorial (a b : ℕ) (h : a = 8! ∧ b = (6!)^2) : Nat.gcd a b = 5760 :=
by sorry

end gcd_factorial_l57_57759


namespace product_remainder_l57_57035

theorem product_remainder (a b : ℕ) (m n : ℤ) (ha : a = 3 * m + 2) (hb : b = 3 * n + 2) : 
  (a * b) % 3 = 1 := 
by 
  sorry

end product_remainder_l57_57035


namespace abs_neg_two_eq_two_l57_57885

theorem abs_neg_two_eq_two : abs (-2) = 2 :=
sorry

end abs_neg_two_eq_two_l57_57885


namespace yard_length_l57_57917

theorem yard_length (father_step : ℝ) (son_step : ℝ) (total_footprints : ℕ) 
  (h_father_step : father_step = 0.72) 
  (h_son_step : son_step = 0.54) 
  (h_total_footprints : total_footprints = 61) : 
  ∃ length : ℝ, length = 21.6 :=
by
  sorry

end yard_length_l57_57917


namespace multiples_of_6_or_8_under_201_not_both_l57_57459

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l57_57459


namespace pet_snake_cost_l57_57960

theorem pet_snake_cost (original_amount left_amount snake_cost : ℕ) 
  (h1 : original_amount = 73) 
  (h2 : left_amount = 18)
  (h3 : snake_cost = original_amount - left_amount) : 
  snake_cost = 55 := 
by 
  sorry

end pet_snake_cost_l57_57960


namespace solve_fractional_equation_l57_57339

noncomputable def fractional_equation (x : ℝ) : Prop :=
  (2 - x) / (x - 3) + 1 / (3 - x) = 1

theorem solve_fractional_equation :
  ∀ x : ℝ, x ≠ 3 → fractional_equation x ↔ x = 2 :=
by
  intros x h
  unfold fractional_equation
  sorry

end solve_fractional_equation_l57_57339


namespace shortest_chord_length_l57_57016

/-- The shortest chord passing through point D given the conditions provided. -/
theorem shortest_chord_length
  (O : Point) (D : Point) (r : ℝ) (OD : ℝ)
  (h_or : r = 5) (h_od : OD = 3) :
  ∃ (AB : ℝ), AB = 8 := 
  sorry

end shortest_chord_length_l57_57016


namespace arithmetic_seq_sum_l57_57173

theorem arithmetic_seq_sum (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (h₁ : S(3) = 9) 
  (h₂ : S(6) = 36) 
  (h₃ : ∀ n, S(n + 1) = S(n) + a(n + 1)) :
  a(7) + a(8) + a(9) = 45 :=
by
  sorry

end arithmetic_seq_sum_l57_57173


namespace solve_inequality_l57_57194

theorem solve_inequality (x : ℝ) : (x - 2) / (x + 5) ≥ 0 ↔ x ∈ set.Iio (-5) ∪ set.Ici 2 :=
by {
  sorry
}

end solve_inequality_l57_57194


namespace pascal_triangle_15_4_l57_57510

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57510


namespace tricia_age_l57_57368

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l57_57368


namespace find_M_plus_N_l57_57120

theorem find_M_plus_N (M N : ℕ)
  (h1 : 4 * 63 = 7 * M)
  (h2 : 4 * N = 7 * 84) :
  M + N = 183 :=
by sorry

end find_M_plus_N_l57_57120


namespace tan_ratio_l57_57309

open Real

theorem tan_ratio (x y : ℝ) (h1 : sin x / cos y + sin y / cos x = 2) (h2 : cos x / sin y + cos y / sin x = 4) : 
  tan x / tan y + tan y / tan x = 2 :=
sorry

end tan_ratio_l57_57309


namespace train_length_l57_57681

theorem train_length {L : ℝ} (h_equal_lengths : ∃ (L: ℝ), L = L) (h_cross_time : ∃ (t : ℝ), t = 60) (h_speed : ∃ (v : ℝ), v = 20) : L = 600 :=
by
  sorry

end train_length_l57_57681


namespace multiples_of_6_or_8_under_201_not_both_l57_57458

theorem multiples_of_6_or_8_under_201_not_both : 
  ∃ (n : ℕ), n = 42 ∧ 
    (∀ x : ℕ, x < 201 → ((x % 6 = 0 ∨ x % 8 = 0) ∧ x % 24 ≠ 0) → x ∈ Finset.range 201) :=
by
  sorry

end multiples_of_6_or_8_under_201_not_both_l57_57458


namespace loan_difference_eq_1896_l57_57192

/-- 
  Samantha borrows $12,000 with two repayment schemes:
  1. A twelve-year loan with an annual interest rate of 8% compounded semi-annually. 
     At the end of 6 years, she must make a payment equal to half of what she owes, 
     and the remaining balance accrues interest until the end of 12 years.
  2. A twelve-year loan with a simple annual interest rate of 10%, paid as a lump-sum at the end.

  Prove that the positive difference between the total amounts to be paid back 
  under the two schemes is $1,896, rounded to the nearest dollar.
-/
theorem loan_difference_eq_1896 :
  let P := 12000
  let r1 := 0.08
  let r2 := 0.10
  let n := 2
  let t := 12
  let t1 := 6
  let A1 := P * (1 + r1 / n) ^ (n * t1)
  let payment_after_6_years := A1 / 2
  let remaining_balance := A1 / 2
  let compounded_remaining := remaining_balance * (1 + r1 / n) ^ (n * t1)
  let total_compound := payment_after_6_years + compounded_remaining
  let total_simple := P * (1 + r2 * t)
  (total_simple - total_compound).round = 1896 := 
by
  sorry

end loan_difference_eq_1896_l57_57192


namespace pascal_triangle_row_fifth_number_l57_57540

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57540


namespace num_valid_sequences_10_transformations_l57_57247

/-- Define the transformations: 
    L: 90° counterclockwise rotation,
    R: 90° clockwise rotation,
    H: reflection across the x-axis,
    V: reflection across the y-axis. -/
inductive Transformation
| L | R | H | V

/-- Define a function to get the number of valid sequences of transformations
    that bring the vertices E, F, G, H back to their original positions.-/
def countValidSequences : ℕ :=
  56

/-- The theorem to prove that the number of valid sequences
    of 10 transformations resulting in the identity transformation is 56. -/
theorem num_valid_sequences_10_transformations : 
  countValidSequences = 56 :=
sorry

end num_valid_sequences_10_transformations_l57_57247


namespace Sue_chewing_gums_count_l57_57028

theorem Sue_chewing_gums_count (S : ℕ) 
  (hMary : 5 = 5) 
  (hSam : 10 = 10) 
  (hTotal : 5 + 10 + S = 30) : S = 15 := 
by {
  sorry
}

end Sue_chewing_gums_count_l57_57028


namespace leif_has_more_oranges_than_apples_l57_57171

-- We are given that Leif has 14 apples and 24 oranges.
def number_of_apples : ℕ := 14
def number_of_oranges : ℕ := 24

-- We need to show how many more oranges he has than apples.
theorem leif_has_more_oranges_than_apples :
  number_of_oranges - number_of_apples = 10 :=
by
  -- The proof would go here, but we are skipping it.
  sorry

end leif_has_more_oranges_than_apples_l57_57171


namespace largest_integer_divides_difference_l57_57789

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end largest_integer_divides_difference_l57_57789


namespace apples_added_l57_57691

theorem apples_added (initial_apples added_apples final_apples : ℕ) 
  (h1 : initial_apples = 8) 
  (h2 : final_apples = 13) 
  (h3 : final_apples = initial_apples + added_apples) : 
  added_apples = 5 :=
by
  sorry

end apples_added_l57_57691


namespace one_third_of_four_l57_57839

theorem one_third_of_four : (1/3) * 4 = 2 :=
by
  sorry

end one_third_of_four_l57_57839


namespace pascal_fifteen_four_l57_57567

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57567


namespace value_of_b_add_c_l57_57823

variables {a b c d : ℝ}

theorem value_of_b_add_c (h1 : a + b = 5) (h2 : c + d = 3) (h3 : a + d = 2) : b + c = 6 :=
sorry

end value_of_b_add_c_l57_57823


namespace find_abs_ab_l57_57027

def ellipse_foci_distance := 5
def hyperbola_foci_distance := 7

def ellipse_condition (a b : ℝ) := b^2 - a^2 = ellipse_foci_distance^2
def hyperbola_condition (a b : ℝ) := a^2 + b^2 = hyperbola_foci_distance^2

theorem find_abs_ab (a b : ℝ) (h_ellipse : ellipse_condition a b) (h_hyperbola : hyperbola_condition a b) :
  |a * b| = 2 * Real.sqrt 111 :=
by
  sorry

end find_abs_ab_l57_57027


namespace pascal_triangle_fifth_number_l57_57516

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57516


namespace find_a7_a8_a9_l57_57175

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def is_arithmetic_sequence (a : ℕ → α) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → α) (n : ℕ) : α :=
  n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

variables {a : ℕ → α}
variables (S : ℕ → α)
variables (S_3 S_6 : α)

-- Given conditions
axiom is_arith_seq : is_arithmetic_sequence a
axiom S_def : ∀ n, S n = sum_of_arithmetic_sequence a n
axiom S_3_eq : S 3 = 9
axiom S_6_eq : S 6 = 36

-- Theorem to prove
theorem find_a7_a8_a9 : a 7 + a 8 + a 9 = 45 :=
sorry

end find_a7_a8_a9_l57_57175


namespace ratio_bound_exceeds_2023_power_l57_57863

theorem ratio_bound_exceeds_2023_power (a b : ℕ → ℝ) (h_pos : ∀ n, 0 < a n ∧ 0 < b n)
  (h1 : ∀ n, (a (n + 1)) * (b (n + 1)) = (a n)^2 + (b n)^2)
  (h2 : ∀ n, (a (n + 1)) + (b (n + 1)) = (a n) * (b n))
  (h3 : ∀ n, a n ≥ b n) :
  ∃ n, (a n) / (b n) > 2023^2023 :=
by
  sorry

end ratio_bound_exceeds_2023_power_l57_57863


namespace weight_of_each_piece_l57_57187

theorem weight_of_each_piece 
  (x : ℝ)
  (h : 2 * x + 0.08 = 0.75) : 
  x = 0.335 :=
by
  sorry

end weight_of_each_piece_l57_57187


namespace pascal_triangle_15_4_l57_57507

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57507


namespace units_digit_of_j_squared_plus_3_power_j_l57_57181

def j : ℕ := 2023^3 + 3^2023 + 2023

theorem units_digit_of_j_squared_plus_3_power_j (j : ℕ) (h : j = 2023^3 + 3^2023 + 2023) : 
  ((j^2 + 3^j) % 10) = 6 := 
  sorry

end units_digit_of_j_squared_plus_3_power_j_l57_57181


namespace problem_I_problem_II_l57_57123

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 ≥ 0}
def B : Set ℝ := {x : ℝ | x ≥ 1}

-- Define the complement of A in the universal set U which is ℝ
def complement_U_A : Set ℝ := {x : ℝ | -1 < x ∧ x < 3}

-- Define the union of complement_U_A and B
def union_complement_U_A_B : Set ℝ := complement_U_A ∪ B

-- Proof Problem I: Prove that the set A is as specified
theorem problem_I : A = {x : ℝ | x ≤ -1 ∨ x ≥ 3} := sorry

-- Proof Problem II: Prove that the union of the complement of A and B is as specified
theorem problem_II : union_complement_U_A_B = {x : ℝ | x > -1} := sorry

end problem_I_problem_II_l57_57123


namespace radius_of_circle_area_of_sector_l57_57969

theorem radius_of_circle (L : ℝ) (θ : ℝ) (hL : L = 50) (hθ : θ = 200) : 
  ∃ r : ℝ, r = 45 / Real.pi := 
by
  sorry

theorem area_of_sector (L : ℝ) (r : ℝ) (hL : L = 50) (hr : r = 45 / Real.pi) : 
  ∃ S : ℝ, S = 1125 / Real.pi := 
by
  sorry

end radius_of_circle_area_of_sector_l57_57969


namespace positional_relationship_l57_57914

variables {Point Line Plane : Type}
variables (a b : Line) (α : Plane)

-- Condition 1: Line a is parallel to Plane α
def line_parallel_to_plane (a : Line) (α : Plane) : Prop := sorry

-- Condition 2: Line b is contained within Plane α
def line_contained_within_plane (b : Line) (α : Plane) : Prop := sorry

-- The positional relationship between line a and line b is either parallel or skew
def lines_parallel_or_skew (a b : Line) : Prop := sorry

theorem positional_relationship (ha : line_parallel_to_plane a α) (hb : line_contained_within_plane b α) :
  lines_parallel_or_skew a b :=
sorry

end positional_relationship_l57_57914


namespace corn_growth_ratio_l57_57752

theorem corn_growth_ratio 
  (growth_first_week : ℕ := 2) 
  (growth_second_week : ℕ) 
  (growth_third_week : ℕ) 
  (total_height : ℕ := 22) 
  (r : ℕ) 
  (h1 : growth_second_week = 2 * r) 
  (h2 : growth_third_week = 4 * (2 * r)) 
  (h3 : growth_first_week + growth_second_week + growth_third_week = total_height) 
  : r = 2 := 
by 
  sorry

end corn_growth_ratio_l57_57752


namespace ken_summit_time_l57_57333

variables (t : ℕ) (s : ℕ)

/--
Sari and Ken climb up a mountain. 
Ken climbs at a constant pace of 500 meters per hour,
and reaches the summit after \( t \) hours starting from 10:00.
Sari starts climbing 2 hours before Ken at 08:00 and is 50 meters behind Ken when he reaches the summit.
Sari is already 700 meters ahead of Ken when he starts climbing.
Prove that Ken reaches the summit at 15:00.
-/
theorem ken_summit_time (h1 : 500 * t = s * (t + 2) + 50)
  (h2 : s * 2 = 700) : t + 10 = 15 :=

sorry

end ken_summit_time_l57_57333


namespace max_area_of_house_l57_57742

-- Definitions for conditions
def height_of_plates : ℝ := 2.5
def price_per_meter_colored : ℝ := 450
def price_per_meter_composite : ℝ := 200
def roof_cost_per_sqm : ℝ := 200
def cost_limit : ℝ := 32000

-- Definitions for the variables
variables (x y : ℝ) (P S : ℝ)

-- Definition for the material cost P
def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

-- Maximum area S and corresponding x
theorem max_area_of_house (x y : ℝ) (h : material_cost x y ≤ cost_limit) :
  S = 100 ∧ x = 20 / 3 :=
sorry

end max_area_of_house_l57_57742


namespace park_area_calculation_l57_57407

def scale := 300 -- miles per inch
def short_diagonal := 10 -- inches
def real_length := short_diagonal * scale -- miles
def park_area := (1/2) * real_length * real_length -- square miles

theorem park_area_calculation : park_area = 4500000 := by
  sorry

end park_area_calculation_l57_57407


namespace pascal_triangle_fifth_number_l57_57577

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57577


namespace inequality_reciprocal_l57_57297

theorem inequality_reciprocal (a b : ℝ) (hab : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
by
  sorry

end inequality_reciprocal_l57_57297


namespace gcd_elements_of_B_l57_57637

def is_element_of_B (n : ℕ) : Prop := ∃ y : ℕ, y > 0 ∧ n = 4 * y + 2

theorem gcd_elements_of_B : ∀ n, is_element_of_B n → (∃ d, ∀ m, is_element_of_B m → d ∣ m) ∧ gcd_elements_of_B d = 2 :=
by
  sorry

end gcd_elements_of_B_l57_57637


namespace stationary_train_length_l57_57750

-- Definitions
def speed_km_per_h := 72
def speed_m_per_s := speed_km_per_h * (1000 / 3600) -- conversion from km/h to m/s
def time_to_pass_pole := 10 -- in seconds
def time_to_cross_stationary_train := 35 -- in seconds
def speed := 20 -- speed in m/s, 72 km/h = 20 m/s, can be inferred from conversion

-- Length of moving train
def length_of_moving_train := speed * time_to_pass_pole

-- Total distance in crossing stationary train
def total_distance := speed * time_to_cross_stationary_train

-- Length of stationary train
def length_of_stationary_train := total_distance - length_of_moving_train

-- Proof statement
theorem stationary_train_length :
  length_of_stationary_train = 500 := by
  sorry

end stationary_train_length_l57_57750


namespace flags_count_l57_57268

-- Define the colors available
inductive Color
| purple | gold | silver

-- Define the number of stripes on the flag
def number_of_stripes : Nat := 3

-- Define a function to calculate the total number of combinations
def total_flags (colors : Nat) (stripes : Nat) : Nat :=
  colors ^ stripes

-- The main theorem we want to prove
theorem flags_count : total_flags 3 number_of_stripes = 27 :=
by
  -- This is the statement only, and the proof is omitted
  sorry

end flags_count_l57_57268


namespace intersection_empty_l57_57291

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l57_57291


namespace sphere_cooling_time_l57_57077

theorem sphere_cooling_time :
  ∃ t : ℝ, t = 15 ∧
  (∀ (k : ℝ) (T : ℝ → ℝ),
    (∀ t, T t = 12 * real.exp (-k * t)) ∧
    (T 0 = 12) ∧ 
    (T 8 = 9) ∧
    (T t = 7) ∧
    ∃ k > 0, (9 = 12 * real.exp (-k * 8)) ∧ (T t = 7)) :=
begin
  sorry
end

end sphere_cooling_time_l57_57077


namespace last_even_distribution_l57_57395

theorem last_even_distribution (n : ℕ) (h : n = 590490) :
  ∃ k : ℕ, (k ≤ n ∧ (n = 3^k + 3^k + 3^k) ∧ (∀ m : ℕ, m < k → ¬(n = 3^m + 3^m + 3^m))) ∧ k = 1 := 
by 
  sorry

end last_even_distribution_l57_57395


namespace Pascal_triangle_fifth_number_l57_57562

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57562


namespace pascal_triangle_fifth_number_l57_57520

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57520


namespace square_of_cube_plus_11_l57_57039

def third_smallest_prime : ℕ := 5

theorem square_of_cube_plus_11 : (third_smallest_prime ^ 3)^2 + 11 = 15636 := by
  -- We will provide a proof later
  sorry

end square_of_cube_plus_11_l57_57039


namespace total_balloons_are_48_l57_57991

theorem total_balloons_are_48 
  (brooke_initial : ℕ) (brooke_add : ℕ) (tracy_initial : ℕ) (tracy_add : ℕ)
  (brooke_half_given : ℕ) (tracy_third_popped : ℕ) : 
  brooke_initial = 20 →
  brooke_add = 15 →
  tracy_initial = 10 →
  tracy_add = 35 →
  brooke_half_given = (brooke_initial + brooke_add) / 2 →
  tracy_third_popped = (tracy_initial + tracy_add) / 3 →
  (brooke_initial + brooke_add - brooke_half_given) + (tracy_initial + tracy_add - tracy_third_popped) = 48 := 
by
  intros
  sorry

end total_balloons_are_48_l57_57991


namespace areas_of_geometric_figures_with_equal_perimeter_l57_57970

theorem areas_of_geometric_figures_with_equal_perimeter (l : ℝ) (h : (l > 0)) :
  let s1 := l^2 / (4 * Real.pi)
  let s2 := l^2 / 16
  let s3 := (Real.sqrt 3) * l^2 / 36
  s1 > s2 ∧ s2 > s3 := by
  sorry

end areas_of_geometric_figures_with_equal_perimeter_l57_57970


namespace acute_triangle_and_angle_relations_l57_57886

theorem acute_triangle_and_angle_relations (a b c u v w : ℝ) (A B C : ℝ)
  (h₁ : a^2 = u * (v + w - u))
  (h₂ : b^2 = v * (w + u - v))
  (h₃ : c^2 = w * (u + v - w)) :
  (a^2 + b^2 > c^2 ∧ b^2 + c^2 > a^2 ∧ c^2 + a^2 > b^2) ∧
  (∀ U V W : ℝ, U = 180 - 2 * A ∧ V = 180 - 2 * B ∧ W = 180 - 2 * C) :=
by sorry

end acute_triangle_and_angle_relations_l57_57886


namespace tangent_slope_at_1_0_l57_57031

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_slope_at_1_0 : (deriv f 1) = 3 := by
  sorry

end tangent_slope_at_1_0_l57_57031


namespace tan_double_angle_l57_57822

open Real

-- Given condition
def condition (x : ℝ) : Prop := tan x - 1 / tan x = 3 / 2

-- Main theorem to prove
theorem tan_double_angle (x : ℝ) (h : condition x) : tan (2 * x) = -4 / 3 := by
  sorry

end tan_double_angle_l57_57822


namespace range_of_a_l57_57738

-- Definitions of the propositions in Lean terms
def proposition_p (a : ℝ) := 
  ∃ x : ℝ, x ∈ [-1, 1] ∧ x^2 - (2 + a) * x + 2 * a = 0

def proposition_q (a : ℝ) := 
  ∃ x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The main theorem to prove that the range of values for a is [-1, 0]
theorem range_of_a {a : ℝ} (h : proposition_p a ∧ proposition_q a) : 
  -1 ≤ a ∧ a ≤ 0 := sorry

end range_of_a_l57_57738


namespace pascal_triangle_15_4_l57_57511

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57511


namespace y_intercept_of_line_l57_57824

theorem y_intercept_of_line 
  (point : ℝ × ℝ)
  (slope_angle : ℝ)
  (h1 : point = (2, -5))
  (h2 : slope_angle = 135) :
  ∃ b : ℝ, (∀ x y : ℝ, y = -x + b ↔ ((y - (-5)) = (-1) * (x - 2))) ∧ b = -3 := 
sorry

end y_intercept_of_line_l57_57824


namespace pascal_15_5th_number_l57_57528

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57528


namespace slope_product_l57_57450

   -- Define the hyperbola
   def hyperbola (x y : ℝ) : Prop := x^2 - (2 * y^2) / (Real.sqrt 5 + 1) = 1

   -- Define the slope calculation for points P, M, N on the hyperbola
   def slopes (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (Real.sqrt 5 + 1) / 2 = ((yP - y0) * (yP + y0)) / ((xP - x0) * (xP + x0)) := sorry
  
   -- Theorem to show the required relationship
   theorem slope_product (xP yP x0 y0 : ℝ) (hP : hyperbola xP yP) (hM : hyperbola x0 y0) (hN : hyperbola (-x0) (-y0)) :
     (yP^2 - y0^2) / (xP^2 - x0^2) = (Real.sqrt 5 + 1) / 2 := sorry
   
end slope_product_l57_57450


namespace Ivan_expected_shots_l57_57153

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l57_57153


namespace factorize_x4_plus_16_l57_57278

theorem factorize_x4_plus_16: ∀ (x : ℝ), x^4 + 16 = (x^2 + 2 * x + 2) * (x^2 - 2 * x + 2) :=
by
  intro x
  sorry

end factorize_x4_plus_16_l57_57278


namespace area_quotient_eq_correct_l57_57848

noncomputable def is_in_plane (x y z : ℝ) : Prop :=
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ x + y + z = 2

def supports (x y z a b c : ℝ) : Prop :=
  (x ≥ a ∧ y ≥ b ∧ z < c) ∨ (x ≥ a ∧ y < b ∧ z ≥ c) ∨ (x < a ∧ y ≥ b ∧ z ≥ c)

def in_S (x y z : ℝ) : Prop :=
  is_in_plane x y z ∧ supports x y z 1 (2/3) (1/3)

noncomputable def area_S : ℝ := 
  -- Placeholder for the computed area of S
  sorry

noncomputable def area_T : ℝ := 
  -- Placeholder for the computed area of T
  sorry

theorem area_quotient_eq_correct :
  (area_S / area_T) = (3 / (8 * Real.sqrt 3)) := 
  sorry

end area_quotient_eq_correct_l57_57848


namespace find_f_neg_one_l57_57815

theorem find_f_neg_one (f : ℝ → ℝ) (h1 : ∀ x, f (-x) + x^2 = - (f x + x^2)) (h2 : f 1 = 1) : f (-1) = -3 := by
  sorry

end find_f_neg_one_l57_57815


namespace employee_salary_l57_57700

theorem employee_salary (A B : ℝ) (h1 : A + B = 560) (h2 : A = 1.5 * B) : B = 224 :=
by
  sorry

end employee_salary_l57_57700


namespace largest_divisor_composite_difference_l57_57797

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_composite_difference_l57_57797


namespace find_circle_center_l57_57242

noncomputable def midpoint_line (a b : ℝ) : ℝ :=
  (a + b) / 2

noncomputable def circle_center (x y : ℝ) : Prop :=
  6 * x - 5 * y = midpoint_line 40 (-20) ∧ 3 * x + 2 * y = 0

theorem find_circle_center : circle_center (20 / 27) (-10 / 9) :=
by
  -- Here would go the proof steps, but we skip it
  sorry

end find_circle_center_l57_57242


namespace isabella_total_haircut_length_l57_57318

theorem isabella_total_haircut_length :
  (18 - 14) + (14 - 9) = 9 := 
sorry

end isabella_total_haircut_length_l57_57318


namespace part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l57_57295

open Set Real

def setA (a : ℝ) : Set ℝ := {x : ℝ | 3 ≤ x ∧ x ≤ a + 5}
def setB : Set ℝ := {x : ℝ | 2 < x ∧ x < 10}

theorem part_one_a_two_complement_union (a : ℝ) (h : a = 2) :
  compl (setA a ∪ setB) = Iic 2 ∪ Ici 10 := sorry

theorem part_one_a_two_complement_intersection (a : ℝ) (h : a = 2) :
  compl (setA a) ∩ setB = Ioo 2 3 ∪ Ioo 7 10 := sorry

theorem part_two_subset (a : ℝ) (h : setA a ⊆ setB) :
  a < 5 := sorry

end part_one_a_two_complement_union_part_one_a_two_complement_intersection_part_two_subset_l57_57295


namespace correct_statement_D_l57_57058

theorem correct_statement_D : (- 3 / 5 : ℚ) < (- 4 / 7 : ℚ) :=
  by
  -- The proof step is omitted as per the instruction
  sorry

end correct_statement_D_l57_57058


namespace telephone_charges_equal_l57_57898

theorem telephone_charges_equal (m : ℝ) :
  (9 + 0.25 * m = 12 + 0.20 * m) → m = 60 :=
by
  intro h
  sorry

end telephone_charges_equal_l57_57898


namespace N_subseteq_M_l57_57325

/--
Let M = { x | ∃ n ∈ ℤ, x = n / 2 + 1 } and
N = { y | ∃ m ∈ ℤ, y = m + 0.5 }.
Prove that N is a subset of M.
-/
theorem N_subseteq_M : 
  let M := { x : ℝ | ∃ n : ℤ, x = n / 2 + 1 }
  let N := { y : ℝ | ∃ m : ℤ, y = m + 0.5 }
  N ⊆ M := sorry

end N_subseteq_M_l57_57325


namespace problem_statement_l57_57648

variable {a b c d k : ℝ}

theorem problem_statement (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
    (h_pos : 0 < k)
    (h_sum_ab : a + b = k)
    (h_sum_cd : c + d = k^2)
    (h_roots1 : ∀ x, x^2 - 4*a*x - 5*b = 0 → x = c ∨ x = d)
    (h_roots2 : ∀ x, x^2 - 4*c*x - 5*d = 0 → x = a ∨ x = b) : 
    a + b + c + d = k + k^2 :=
sorry

end problem_statement_l57_57648


namespace necessary_but_not_sufficient_l57_57326

open Set

variable {α : Type*}

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient : 
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ b, b ∈ M ∧ b ∉ N) := 
by 
  sorry

end necessary_but_not_sufficient_l57_57326


namespace calculate_tax_l57_57422

noncomputable def cadastral_value : ℝ := 3000000 -- 3 million rubles
noncomputable def tax_rate : ℝ := 0.001        -- 0.1% converted to decimal
noncomputable def tax : ℝ := cadastral_value * tax_rate -- Tax formula

theorem calculate_tax : tax = 3000 := by
  sorry

end calculate_tax_l57_57422


namespace diagonal_length_of_rectangular_prism_l57_57406

-- Define the dimensions of the rectangular prism
variables (a b c : ℕ) (a_pos : a = 12) (b_pos : b = 15) (c_pos : c = 8)

-- Define the theorem statement
theorem diagonal_length_of_rectangular_prism : 
  ∃ d : ℝ, d = Real.sqrt (a ^ 2 + b ^ 2 + c ^ 2) ∧ d = Real.sqrt 433 := 
by
  -- Note that the proof is intentionally omitted
  sorry

end diagonal_length_of_rectangular_prism_l57_57406


namespace distance_3_units_l57_57686

theorem distance_3_units (x : ℤ) (h : |x + 2| = 3) : x = -5 ∨ x = 1 := by
  sorry

end distance_3_units_l57_57686


namespace when_was_p_turned_off_l57_57375

noncomputable def pipe_p_rate := (1/12 : ℚ)  -- Pipe p rate
noncomputable def pipe_q_rate := (1/15 : ℚ)  -- Pipe q rate
noncomputable def combined_rate := (3/20 : ℚ) -- Combined rate of p and q when both are open
noncomputable def time_after_p_off := (1.5 : ℚ)  -- Time for q to fill alone after p is off
noncomputable def fill_cistern (t : ℚ) := combined_rate * t + pipe_q_rate * time_after_p_off

theorem when_was_p_turned_off (t : ℚ) : fill_cistern t = 1 ↔ t = 6 := sorry

end when_was_p_turned_off_l57_57375


namespace convert_234_base5_to_binary_l57_57940

def base5_to_decimal (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 4 * 5^0

def decimal_to_binary (n : Nat) : List Nat :=
  let rec to_binary_aux (n : Nat) (accum : List Nat) : List Nat :=
    if n = 0 then accum
    else to_binary_aux (n / 2) ((n % 2) :: accum)
  to_binary_aux n []

theorem convert_234_base5_to_binary :
  (base5_to_decimal 234 = 69) ∧ (decimal_to_binary 69 = [1,0,0,0,1,0,1]) :=
by
  sorry

end convert_234_base5_to_binary_l57_57940


namespace greatest_possible_value_l57_57343

theorem greatest_possible_value (x y : ℝ) (h1 : -4 ≤ x) (h2 : x ≤ -2) (h3 : 2 ≤ y) (h4 : y ≤ 4) : 
  ∃ z: ℝ, z = (x + y) / x ∧ (∀ z', z' = (x' + y') / x' ∧ -4 ≤ x' ∧ x' ≤ -2 ∧ 2 ≤ y' ∧ y' ≤ 4 → z' ≤ z) ∧ z = 0 :=
by
  sorry

end greatest_possible_value_l57_57343


namespace option_D_is_negative_l57_57413

theorem option_D_is_negative :
  let A := abs (-4)
  let B := -(-4)
  let C := (-4) ^ 2
  let D := -(4 ^ 2)
  D < 0 := by
{
  -- Place sorry here since we are not required to provide the proof
  sorry
}

end option_D_is_negative_l57_57413


namespace saree_stripes_l57_57378

theorem saree_stripes
  (G : ℕ) (B : ℕ) (Br : ℕ) (total_stripes : ℕ) (total_patterns : ℕ)
  (h1 : G = 3 * Br)
  (h2 : B = 5 * G)
  (h3 : Br = 4)
  (h4 : B + G + Br = 100)
  (h5 : total_stripes = 100)
  (h6 : total_patterns = total_stripes / 3) :
  B = 84 ∧ total_patterns = 33 := 
  by {
    sorry
  }

end saree_stripes_l57_57378


namespace Ivan_expected_shots_l57_57154

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l57_57154


namespace number_of_groups_l57_57747

theorem number_of_groups (max min c : ℕ) (h_max : max = 140) (h_min : min = 50) (h_c : c = 10) : 
  (max - min) / c + 1 = 10 := 
by
  sorry

end number_of_groups_l57_57747


namespace intersection_A_B_l57_57111

/-- Definitions for the sets A and B --/
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {1, 2, 4, 5}

-- Theorem statement regarding the intersection of sets A and B
theorem intersection_A_B : A ∩ B = {1} :=
by sorry

end intersection_A_B_l57_57111


namespace intersection_empty_l57_57292

def A : Set ℝ := {x | x^2 + 2 * x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := 
by
  sorry

end intersection_empty_l57_57292


namespace pascal_triangle_fifth_number_l57_57582

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57582


namespace lower_limit_brother_l57_57141

variable (W B : Real)

-- Arun's opinion
def aruns_opinion := 66 < W ∧ W < 72

-- Brother's opinion
def brothers_opinion := B < W ∧ W < 70

-- Mother's opinion
def mothers_opinion := W ≤ 69

-- Given the average probable weight of Arun which is 68 kg
def average_weight := (69 + (max 66 B)) / 2 = 68

theorem lower_limit_brother (h₁ : aruns_opinion W) (h₂ : brothers_opinion W B) (h₃ : mothers_opinion W) (h₄ : average_weight B) :
  B = 67 := sorry

end lower_limit_brother_l57_57141


namespace gcd_of_B_is_2_l57_57641

-- Definitions based on conditions
def B : Set ℕ := { n | ∃ x : ℕ, x > 0 ∧ n = 4 * x + 2 }

-- Statement of the proof problem
theorem gcd_of_B_is_2 : Nat.gcd_set B = 2 :=
sorry

end gcd_of_B_is_2_l57_57641


namespace max_rectangle_area_l57_57030

-- Lean statement for the proof problem

theorem max_rectangle_area (x : ℝ) (y : ℝ) (h1 : 2 * x + 2 * y = 24) : ∃ A : ℝ, A = 36 :=
by
  -- Definitions for perimeter and area
  let P := 2 * x + 2 * y
  let A := x * y

  -- Conditions
  have h1 : P = 24 := h1

  -- Setting maximum area and completing the proof
  sorry

end max_rectangle_area_l57_57030


namespace rain_probability_tel_aviv_l57_57347

noncomputable theory
open Classical

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end rain_probability_tel_aviv_l57_57347


namespace total_time_spent_l57_57236

-- Definitions based on the conditions
def number_of_chairs := 2
def number_of_tables := 2
def minutes_per_piece := 8
def total_pieces := number_of_chairs + number_of_tables

-- The statement we want to prove
theorem total_time_spent : total_pieces * minutes_per_piece = 32 :=
by
  sorry

end total_time_spent_l57_57236


namespace pascal_fifteen_four_l57_57568

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57568


namespace gcd_factorial_8_and_6_squared_l57_57768

-- We will use nat.factorial and other utility functions from Mathlib

noncomputable def factorial_8 : ℕ := nat.factorial 8
noncomputable def factorial_6_squared : ℕ := (nat.factorial 6) ^ 2

theorem gcd_factorial_8_and_6_squared : nat.gcd factorial_8 factorial_6_squared = 1440 := by
  -- We'll need Lean to compute prime factorizations
  sorry

end gcd_factorial_8_and_6_squared_l57_57768


namespace union_of_M_and_N_l57_57238

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {0, 1, 2}

theorem union_of_M_and_N : M ∪ N = {-1, 0, 1, 2} :=
by
  sorry

end union_of_M_and_N_l57_57238


namespace incorrect_axis_symmetry_l57_57284

noncomputable def quadratic_function (x : ℝ) : ℝ := - (x + 2)^2 - 3

theorem incorrect_axis_symmetry :
  (∀ x : ℝ, quadratic_function x < 0) ∧
  (∀ x : ℝ, x > -1 → (quadratic_function x < quadratic_function (-2))) ∧
  (¬∃ x : ℝ, quadratic_function x = 0) ∧
  (¬ ∀ x : ℝ, x = 2) →
  false :=
by
  sorry

end incorrect_axis_symmetry_l57_57284


namespace solve_equation_l57_57425

theorem solve_equation (x : ℚ) :
  (1 / (x + 2) + 3 * x / (x + 2) + 4 / (x + 2) = 1) → x = -3 / 2 :=
by
  sorry

end solve_equation_l57_57425


namespace neg_p_l57_57653

variable {α : Type}
variable (x : α)

def p (x : Real) : Prop := ∀ x : Real, x > 1 → x^2 - 1 > 0

theorem neg_p : ¬( ∀ x : Real, x > 1 → x^2 - 1 > 0) ↔ ∃ x : Real, x > 1 ∧ x^2 - 1 ≤ 0 := 
by 
  sorry

end neg_p_l57_57653


namespace cheryl_gave_mms_to_sister_l57_57755

-- Definitions based on the problem conditions
def initial_mms := 25
def ate_after_lunch := 7
def ate_after_dinner := 5
def mms_left := initial_mms - ate_after_lunch - ate_after_dinner

-- Lean statement for the proof problem
theorem cheryl_gave_mms_to_sister : initial_mms - mms_left = 12 :=
by
  unfold initial_mms mms_left
  rw [sub_sub, sub_self, sub_add],
  sorry  -- proof omitted.

end cheryl_gave_mms_to_sister_l57_57755


namespace Ivan_expected_shots_l57_57152

noncomputable def expected_shots (n : ℕ) (p : ℝ) (gain : ℕ) : ℝ :=
  let a := 1 / (1 - p / gain)
  n * a

theorem Ivan_expected_shots
  (initial_arrows : ℕ)
  (hit_probability : ℝ)
  (arrows_per_hit : ℕ)
  (expected_shots_value : ℝ) :
  initial_arrows = 14 →
  hit_probability = 0.1 →
  arrows_per_hit = 3 →
  expected_shots_value = 20 →
  expected_shots initial_arrows hit_probability arrows_per_hit = expected_shots_value := by
  sorry

end Ivan_expected_shots_l57_57152


namespace min_sum_xy_l57_57972

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l57_57972


namespace quadratic_equation_value_l57_57114

theorem quadratic_equation_value (a : ℝ) (h₁ : a^2 - 2 = 2) (h₂ : a ≠ 2) : a = -2 :=
by
  sorry

end quadratic_equation_value_l57_57114


namespace product_of_digits_l57_57993

theorem product_of_digits (A B : ℕ) (h1 : A + B = 14) (h2 : (10 * A + B) % 4 = 0) : A * B = 48 :=
by
  sorry

end product_of_digits_l57_57993


namespace rate_per_sq_meter_l57_57210

def length : ℝ := 5.5
def width : ℝ := 3.75
def totalCost : ℝ := 14437.5

theorem rate_per_sq_meter : (totalCost / (length * width)) = 700 := 
by sorry

end rate_per_sq_meter_l57_57210


namespace intersection_points_l57_57896

def parabola1 (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
def parabola2 (x : ℝ) : ℝ := 9 * x^2 + 6 * x + 2

theorem intersection_points :
  {p : ℝ × ℝ | parabola1 p.1 = p.2 ∧ parabola2 p.1 = p.2} = 
  {(-5/3, 17), (0, 2)} :=
by
  sorry

end intersection_points_l57_57896


namespace least_subtraction_divisible_l57_57052

def least_subtrahend (n m : ℕ) : ℕ :=
n % m

theorem least_subtraction_divisible (n : ℕ) (m : ℕ) (sub : ℕ) :
  n = 13604 → m = 87 → sub = least_subtrahend n m → (n - sub) % m = 0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end least_subtraction_divisible_l57_57052


namespace count_ball_distributions_l57_57473

theorem count_ball_distributions : 
  ∃ (n : ℕ), n = 3 ∧
  (∀ (balls boxes : ℕ), balls = 5 → boxes = 3 → (∀ (dist : ℕ → ℕ), (sorry: Prop))) := sorry

end count_ball_distributions_l57_57473


namespace find_y_l57_57197

open Real

structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

def parallel (v₁ v₂ : Vec3) : Prop := ∃ s : ℝ, v₁ = ⟨s * v₂.x, s * v₂.y, s * v₂.z⟩

def orthogonal (v₁ v₂ : Vec3) : Prop := (v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z) = 0

noncomputable def correct_y (x y : Vec3) : Vec3 :=
  ⟨(8 : ℝ) - 2 * (2 : ℝ), (-4 : ℝ) - 2 * (2 : ℝ), (2 : ℝ) - 2 * (2 : ℝ)⟩

theorem find_y :
  ∀ (x y : Vec3),
    (x.x + y.x = 8) ∧ (x.y + y.y = -4) ∧ (x.z + y.z = 2) →
    (parallel x ⟨2, 2, 2⟩) →
    (orthogonal y ⟨1, -1, 0⟩) →
    y = ⟨4, -8, -2⟩ :=
by
  intros x y Hxy Hparallel Horthogonal
  sorry

end find_y_l57_57197


namespace number_of_girls_and_boys_l57_57361

-- Definitions for the conditions
def ratio_girls_to_boys (g b : ℕ) := g = 4 * (g + b) / 7 ∧ b = 3 * (g + b) / 7
def total_students (g b : ℕ) := g + b = 56

-- The main proof statement
theorem number_of_girls_and_boys (g b : ℕ) 
  (h_ratio : ratio_girls_to_boys g b)
  (h_total : total_students g b) : 
  g = 32 ∧ b = 24 :=
by {
  sorry
}

end number_of_girls_and_boys_l57_57361


namespace alcohol_to_water_ratio_l57_57895

variable {V p q : ℚ}

def alcohol_volume_jar1 (V p : ℚ) : ℚ := (2 * p) / (2 * p + 3) * V
def water_volume_jar1 (V p : ℚ) : ℚ := 3 / (2 * p + 3) * V
def alcohol_volume_jar2 (V q : ℚ) : ℚ := q / (q + 2) * 2 * V
def water_volume_jar2 (V q : ℚ) : ℚ := 2 / (q + 2) * 2 * V

def total_alcohol_volume (V p q : ℚ) : ℚ :=
  alcohol_volume_jar1 V p + alcohol_volume_jar2 V q

def total_water_volume (V p q : ℚ) : ℚ :=
  water_volume_jar1 V p + water_volume_jar2 V q

theorem alcohol_to_water_ratio (V p q : ℚ) :
  (total_alcohol_volume V p q) / (total_water_volume V p q) = (2 * p + 2 * q) / (3 * p + q + 10) :=
by
  sorry

end alcohol_to_water_ratio_l57_57895


namespace pascal_fifth_number_in_row_15_l57_57548

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57548


namespace car_capacities_rental_plans_l57_57308

-- Define the capacities for part 1
def capacity_A : ℕ := 3
def capacity_B : ℕ := 4

theorem car_capacities (x y : ℕ) (h₁ : 2 * x + y = 10) (h₂ : x + 2 * y = 11) : 
  x = capacity_A ∧ y = capacity_B := by
  sorry

-- Define the valid rental plans for part 2
def valid_rental_plan (a b : ℕ) : Prop :=
  3 * a + 4 * b = 31

theorem rental_plans (a b : ℕ) (h : valid_rental_plan a b) : 
  (a = 1 ∧ b = 7) ∨ (a = 5 ∧ b = 4) ∨ (a = 9 ∧ b = 1) := by
  sorry

end car_capacities_rental_plans_l57_57308


namespace min_value_max_value_l57_57189

theorem min_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 0) := sorry

theorem max_value (a b c : ℝ) (h1 : a^2 + a * b + b^2 = 11) (h2 : b^2 + b * c + c^2 = 11) : 
  (∃ v, v = c^2 + c * a + a^2 ∧ v = 44) := sorry

end min_value_max_value_l57_57189


namespace find_y_l57_57476

theorem find_y (y : ℝ) (hy : 0 < y) 
  (h : (Real.sqrt (12 * y)) * (Real.sqrt (6 * y)) * (Real.sqrt (18 * y)) * (Real.sqrt (9 * y)) = 27) : 
  y = 1 / 2 := 
sorry

end find_y_l57_57476


namespace five_a_plus_five_b_eq_neg_twenty_five_thirds_l57_57421

variable (g f : ℝ → ℝ)
variable (a b : ℝ)
axiom g_def : ∀ x, g x = 3 * x + 5
axiom g_inv_rel : ∀ x, g x = (f⁻¹ x) - 1
axiom f_def : ∀ x, f x = a * x + b
axiom f_inv_def : ∀ x, f⁻¹ (f x) = x

theorem five_a_plus_five_b_eq_neg_twenty_five_thirds :
    5 * a + 5 * b = -25 / 3 :=
sorry

end five_a_plus_five_b_eq_neg_twenty_five_thirds_l57_57421


namespace triangle_inequality_l57_57019

theorem triangle_inequality (a b c m_A : ℝ)
  (h1 : 2*m_A ≤ b + c)
  (h2 : a^2 + (2*m_A)^2 = (b^2) + (c^2)) :
  a^2 + 4*m_A^2 ≤ (b + c)^2 :=
by {
  sorry
}

end triangle_inequality_l57_57019


namespace smallest_base_l57_57713

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end smallest_base_l57_57713


namespace incorrect_statement_maximum_value_l57_57227

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem incorrect_statement_maximum_value :
  ∃ (a b c : ℝ), 
    (quadratic_function a b c 1 = -40) ∧
    (quadratic_function a b c (-1) = -8) ∧
    (quadratic_function a b c (-3) = 8) ∧
    (∀ (x_max : ℝ), (x_max = -b / (2 * a)) →
      (quadratic_function a b c x_max = 10) ∧
      (quadratic_function a b c x_max ≠ 8)) :=
by
  sorry

end incorrect_statement_maximum_value_l57_57227


namespace sqrt_of_sum_of_powers_l57_57043

theorem sqrt_of_sum_of_powers :
  sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 :=
sorry

end sqrt_of_sum_of_powers_l57_57043


namespace sum_of_coefficients_l57_57961

theorem sum_of_coefficients (a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ) (hx : (1 - 2 * x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  a_2 + a_3 + a_4 + a_5 + a_6 + a_7 = 12 :=
sorry

end sum_of_coefficients_l57_57961


namespace largest_integer_divides_difference_l57_57787

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end largest_integer_divides_difference_l57_57787


namespace range_g_l57_57951

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + 
  Real.pi * (Real.arcsin (x / 3)) - 
  (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_g : 
  set.range g = set.Icc (Real.pi^2 / 6) (25 * Real.pi^2 / 6) := 
sorry

end range_g_l57_57951


namespace abc_sum_seven_l57_57022

theorem abc_sum_seven (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_eq : (a + b + c)^3 - a^3 - b^3 - c^3 = 210) : a + b + c = 7 :=
sorry

end abc_sum_seven_l57_57022


namespace isosceles_triangle_side_length_l57_57137

theorem isosceles_triangle_side_length (n : ℕ) : 
  (∃ a b : ℕ, a ≠ 4 ∧ b ≠ 4 ∧ (a = b ∨ a = 4 ∨ b = 4) ∧ 
  a^2 - 6*a + n = 0 ∧ b^2 - 6*b + n = 0) → 
  (n = 8 ∨ n = 9) := 
by
  sorry

end isosceles_triangle_side_length_l57_57137


namespace find_b_l57_57118

theorem find_b (b : ℝ) (h : ∃ (f_inv : ℝ → ℝ), (∀ x y, f_inv (2^x + b) = y) ∧ f_inv 5 = 2) :
    b = 1 := by
  sorry

end find_b_l57_57118


namespace area_of_rectangle_l57_57146

theorem area_of_rectangle (s : ℝ) (h1 : 4 * s = 100) : 2 * s * 2 * s = 2500 := by
  sorry

end area_of_rectangle_l57_57146


namespace bench_allocation_l57_57748

theorem bench_allocation (M : ℕ) : (∃ M, M > 0 ∧ 5 * M = 13 * M) → M = 5 :=
by
  sorry

end bench_allocation_l57_57748


namespace largest_divisor_of_difference_between_n_and_n4_l57_57799

theorem largest_divisor_of_difference_between_n_and_n4 (n : ℤ) (h_composite : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ p * q = n) :
  ∃ k : ℤ, k = 6 ∧ k ∣ (n^4 - n) :=
begin
  sorry
end

end largest_divisor_of_difference_between_n_and_n4_l57_57799


namespace saturn_moon_approximation_l57_57015

theorem saturn_moon_approximation : (1.2 * 10^5) * 10 = 1.2 * 10^6 := 
by sorry

end saturn_moon_approximation_l57_57015


namespace b_completes_work_in_48_days_l57_57907

noncomputable def work_rate (days : ℕ) : ℚ := 1 / days

theorem b_completes_work_in_48_days (a b c : ℕ) 
  (h1 : work_rate (a + b) = work_rate 16)
  (h2 : work_rate a = work_rate 24)
  (h3 : work_rate c = work_rate 48) :
  work_rate b = work_rate 48 :=
by
  sorry

end b_completes_work_in_48_days_l57_57907


namespace harmonic_mean_pairs_l57_57784

open Nat

theorem harmonic_mean_pairs :
  ∃ n, n = 199 ∧ 
  (∀ (x y : ℕ), 0 < x → 0 < y → 
  x < y → (2 * x * y) / (x + y) = 6^10 → 
  x * y - (3^10 * 2^9) * (x - 1) - (3^10 * 2^9) * (y - 1) = 3^20 * 2^18) :=
sorry

end harmonic_mean_pairs_l57_57784


namespace heavy_tailed_permutations_count_l57_57437

/-- A permutation is heavy-tailed if the sum of the first three numbers is less than the sum of 
    the last three numbers and the third number is even. -/
def heavy_tailed (p : Perm (Fin 6)) : Prop :=
  p 0 + p 1 + p 2 < p 3 + p 4 + p 5 ∧ (p 2) % 2 = 0

open Finset

/-- The number of heavy-tailed permutations of the set {1, 2, 3, 4, 5, 6} -/
theorem heavy_tailed_permutations_count : 
  (Finset.univ.filter heavy_tailed).card = 140 := sorry

end heavy_tailed_permutations_count_l57_57437


namespace pascal_fifteen_four_l57_57566

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57566


namespace expected_shots_l57_57150

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l57_57150


namespace determine_ABCC_l57_57271

theorem determine_ABCC :
  ∃ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ 
    C ≠ D ∧ C ≠ E ∧ 
    D ≠ E ∧ 
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    1000 * A + 100 * B + 11 * C = (11 * D - E) * 100 + 11 * D * E ∧ 
    1000 * A + 100 * B + 11 * C = 1966 :=
sorry

end determine_ABCC_l57_57271


namespace triangle_interior_angles_l57_57899

theorem triangle_interior_angles (E1 E2 E3 : ℝ) (I1 I2 I3 : ℝ) (x : ℝ)
  (h1 : E1 = 12 * x) 
  (h2 : E2 = 13 * x) 
  (h3 : E3 = 15 * x)
  (h4 : E1 + E2 + E3 = 360) 
  (h5 : I1 = 180 - E1) 
  (h6 : I2 = 180 - E2) 
  (h7 : I3 = 180 - E3) :
  I1 = 72 ∧ I2 = 63 ∧ I3 = 45 :=
by
  sorry

end triangle_interior_angles_l57_57899


namespace monotonically_increasing_interval_l57_57935

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

theorem monotonically_increasing_interval :
  (∀ x y, 0 < x ∧ x < y ∧ y < Real.pi / 2 → f x < f y) :=
sorry

end monotonically_increasing_interval_l57_57935


namespace pyramid_properties_l57_57033

open Real EuclideanGeometry

-- Define the points O, A, B, C
def O : EuclideanSpace ℝ (fin 3) := ![0, 0, 0]
def A : EuclideanSpace ℝ (fin 3) := ![5, 2, 0]
def B : EuclideanSpace ℝ (fin 3) := ![2, 5, 0]
def C : EuclideanSpace ℝ (fin 3) := ![1, 2, 4]

-- Define vectors AB, AC, AO
def vector_AB : EuclideanSpace ℝ (fin 3) := B - A
def vector_AC : EuclideanSpace ℝ (fin 3) := C - A
def vector_AO : EuclideanSpace ℝ (fin 3) := O - A

-- Given calculated values
noncomputable def volume_OABC : ℝ := 14
noncomputable def area_ABC : ℝ := 6 * Real.sqrt 3
noncomputable def height_OD : ℝ := 7 * Real.sqrt 3 / 3

-- Proof statement
theorem pyramid_properties :
  let V := 1 / 6 * Real.abs (vector_AB.1 * (vector_AC.2 * vector_AO.3 - vector_AC.3 * vector_AO.2)
                         - vector_AB.2 * (vector_AC.1 * vector_AO.3 - vector_AC.3 * vector_AO.1)
                         + vector_AB.3 * (vector_AC.1 * vector_AO.2 - vector_AC.2 * vector_AO.1)) in
  let S := 1 / 2 * Real.sqrt ((vector_AB.2 * vector_AC.3 - vector_AB.3 * vector_AC.2)^2
                            + (vector_AB.3 * vector_AC.1 - vector_AB.1 * vector_AC.3)^2
                            + (vector_AB.1 * vector_AC.2 - vector_AB.2 * vector_AC.1)^2) in
  V = volume_OABC ∧ S = area_ABC ∧ (3 * V) / S = height_OD := by {
  -- The detailed proof will be provided here
  sorry
}

end pyramid_properties_l57_57033


namespace simplify_expression_l57_57737

theorem simplify_expression (a b : ℝ) (h1 : a ≠ b) (h2 : a ≠ 0) (h3 : b ≠ 0) :
  ( (1/(a-b) - 2 * a * b / (a^3 - a^2 * b + a * b^2 - b^3)) / 
    ((a^2 + a * b) / (a^3 + a^2 * b + a * b^2 + b^3) + 
    b / (a^2 + b^2)) ) = (a - b) / (a + b) :=
by
  sorry

end simplify_expression_l57_57737


namespace photo_arrangements_l57_57409

-- The description of the problem conditions translated into definitions
def num_positions := 6  -- Total positions (1 teacher + 5 students)

def teacher_positions := 4  -- Positions where teacher can stand (not at either end)

def student_permutations : ℕ := Nat.factorial 5  -- Number of ways to arrange 5 students

-- The total number of valid arrangements where the teacher does not stand at either end
def total_valid_arrangements : ℕ := teacher_positions * student_permutations

-- Statement to be proven
theorem photo_arrangements:
  total_valid_arrangements = 480 :=
by
  sorry

end photo_arrangements_l57_57409


namespace find_m_from_parallel_l57_57300

theorem find_m_from_parallel (m : ℝ) : 
  (∃ (A B : ℝ×ℝ), A = (-2, m) ∧ B = (m, 4) ∧
  (∃ (a b c : ℝ), a = 2 ∧ b = 1 ∧ c = -1 ∧
  (a * (B.1 - A.1) + b * (B.2 - A.2) = 0)) ) 
  → m = -8 :=
by
  sorry

end find_m_from_parallel_l57_57300


namespace total_rent_of_field_is_correct_l57_57068

namespace PastureRental

def cowMonths (cows : ℕ) (months : ℕ) : ℕ := cows * months

def aCowMonths : ℕ := cowMonths 24 3
def bCowMonths : ℕ := cowMonths 10 5
def cCowMonths : ℕ := cowMonths 35 4
def dCowMonths : ℕ := cowMonths 21 3

def totalCowMonths : ℕ := aCowMonths + bCowMonths + cCowMonths + dCowMonths

def rentPerCowMonth : ℕ := 1440 / aCowMonths

def totalRent : ℕ := rentPerCowMonth * totalCowMonths

theorem total_rent_of_field_is_correct :
  totalRent = 6500 :=
by
  sorry

end PastureRental

end total_rent_of_field_is_correct_l57_57068


namespace ratio_MN_l57_57475

variables (Q P R M N : ℝ)

def satisfies_conditions (Q P R M N : ℝ) : Prop :=
  M = 0.40 * Q ∧
  Q = 0.25 * P ∧
  R = 0.60 * P ∧
  N = 0.50 * R

theorem ratio_MN (Q P R M N : ℝ) (h : satisfies_conditions Q P R M N) : M / N = 1 / 3 :=
by {
  sorry
}

end ratio_MN_l57_57475


namespace pascal_triangle_row_fifth_number_l57_57542

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57542


namespace molecular_weight_is_44_02_l57_57704

-- Definition of atomic weights and the number of atoms
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00
def count_N : ℕ := 2
def count_O : ℕ := 1

-- The compound's molecular weight calculation
def molecular_weight : ℝ := (count_N * atomic_weight_N) + (count_O * atomic_weight_O)

-- The proof statement that the molecular weight of the compound is approximately 44.02 amu
theorem molecular_weight_is_44_02 : molecular_weight = 44.02 := 
by
  sorry

#eval molecular_weight  -- Should output 44.02 (not part of the theorem, just for checking)

end molecular_weight_is_44_02_l57_57704


namespace max_gcd_b_eq_1_l57_57850

-- Define bn as bn = 2^n - 1 for natural number n
def b (n : ℕ) : ℕ := 2^n - 1

-- Define en as the greatest common divisor of bn and bn+1
def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n + 1))

-- The theorem to prove:
theorem max_gcd_b_eq_1 (n : ℕ) : e n = 1 :=
  sorry

end max_gcd_b_eq_1_l57_57850


namespace smallest_base_l57_57714

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end smallest_base_l57_57714


namespace tricia_age_l57_57367

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l57_57367


namespace tangent_line_parabola_l57_57682

theorem tangent_line_parabola (d : ℝ) :
  (∃ (f g : ℝ → ℝ), (∀ x y, y = f x ↔ y = 3 * x + d) ∧ (∀ x y, y = g x ↔ y ^ 2 = 12 * x)
  ∧ (∀ x y, y = f x ∧ y = g x → y = 3 * x + d ∧ y ^ 2 = 12 * x )) →
  d = 1 :=
sorry

end tangent_line_parabola_l57_57682


namespace domain_of_g_l57_57380

theorem domain_of_g (x y : ℝ) : 
  (∃ g : ℝ, g = 1 / (x^2 + (x - y)^2 + y^2)) ↔ (x, y) ≠ (0, 0) :=
by sorry

end domain_of_g_l57_57380


namespace boat_speed_in_still_water_l57_57688

theorem boat_speed_in_still_water (V_b : ℝ) : 
  (∀ t : ℝ, t = 26 / (V_b + 6) → t = 14 / (V_b - 6)) → V_b = 20 :=
by
  sorry

end boat_speed_in_still_water_l57_57688


namespace gcd_factorial_8_6_squared_l57_57771

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end gcd_factorial_8_6_squared_l57_57771


namespace range_of_k_l57_57479

theorem range_of_k 
  (h : ∀ x : ℝ, (k^2 - 2*k + 3/2)^x < (k^2 - 2*k + 3/2)^(1 - x) ↔ x ∈ Ioi (1/2)) :
  1 - Real.sqrt 2 / 2 < k ∧ k < 1 + Real.sqrt 2 / 2 :=
by sorry

end range_of_k_l57_57479


namespace line_equation_passes_through_and_has_normal_l57_57992

theorem line_equation_passes_through_and_has_normal (x y : ℝ) 
    (H1 : ∃ l : ℝ → ℝ, l 3 = 4)
    (H2 : ∃ n : ℝ × ℝ, n = (1, 2)) : 
    x + 2 * y - 11 = 0 :=
sorry

end line_equation_passes_through_and_has_normal_l57_57992


namespace inequality_holds_l57_57664

theorem inequality_holds (a b : ℝ) : 
  a^2 + a * b + b^2 ≥ 3 * (a + b - 1) :=
sorry

end inequality_holds_l57_57664


namespace exam_correct_answers_count_l57_57232

theorem exam_correct_answers_count (x y : ℕ) (h1 : x + y = 80) (h2 : 4 * x - y = 130) : x = 42 :=
by {
  -- (proof to be completed later)
  sorry
}

end exam_correct_answers_count_l57_57232


namespace sum_at_simple_interest_l57_57083

theorem sum_at_simple_interest (P R : ℝ) (h1 : P * R * 3 / 100 - P * (R + 3) * 3 / 100 = -90) : P = 1000 :=
sorry

end sum_at_simple_interest_l57_57083


namespace range_of_a_for_quadratic_inequality_l57_57829

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ (x : ℝ), 1 ≤ x ∧ x < 5 → x^2 - (a + 1)*x + a ≤ 0) ↔ (4 ≤ a ∧ a < 5) :=
sorry

end range_of_a_for_quadratic_inequality_l57_57829


namespace find_y_l57_57976

theorem find_y (y : ℝ) (h : (8 + 15 + 22 + 5 + y) / 5 = 12) : y = 10 :=
by
  -- the proof is skipped
  sorry

end find_y_l57_57976


namespace expected_shots_ivan_l57_57155

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l57_57155


namespace mutually_exclusive_events_l57_57841

-- Definitions based on the given conditions
def sample_inspection (n : ℕ) := n = 10
def event_A (defective_products : ℕ) := defective_products ≥ 2
def event_B (defective_products : ℕ) := defective_products ≤ 1

-- The proof statement
theorem mutually_exclusive_events (n : ℕ) (defective_products : ℕ) 
  (h1 : sample_inspection n) (h2 : event_A defective_products) : 
  event_B defective_products = false :=
by
  sorry

end mutually_exclusive_events_l57_57841


namespace pascal_triangle_fifth_number_l57_57497

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57497


namespace inequality_solution_l57_57059

theorem inequality_solution (x : ℚ) (hx : x = 3 ∨ x = 2 ∨ x = 1 ∨ x = 0) : 
  (1 / 3) - (x / 3) < -(1 / 2) → x = 3 :=
by
  sorry

end inequality_solution_l57_57059


namespace GCD_of_set_B_is_2_l57_57632

/-- Auxiliary definition for the set B -/
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2}

/-- The greatest common divisor of all numbers in the set B is 2 -/
theorem GCD_of_set_B_is_2 : Nat.gcd_set B = 2 := 
sorry

end GCD_of_set_B_is_2_l57_57632


namespace solve_fractional_eq1_l57_57669

theorem solve_fractional_eq1 : ¬ ∃ (x : ℝ), 1 / (x - 2) = (1 - x) / (2 - x) - 3 :=
by sorry

end solve_fractional_eq1_l57_57669


namespace ratio_jacob_edward_l57_57319

-- Definitions and conditions
def brian_shoes : ℕ := 22
def edward_shoes : ℕ := 3 * brian_shoes
def total_shoes : ℕ := 121
def jacob_shoes : ℕ := total_shoes - brian_shoes - edward_shoes

-- Statement of the problem
theorem ratio_jacob_edward (h_brian : brian_shoes = 22)
                          (h_edward : edward_shoes = 3 * brian_shoes)
                          (h_total : total_shoes = 121)
                          (h_jacob : jacob_shoes = total_shoes - brian_shoes - edward_shoes) :
                          jacob_shoes / edward_shoes = 1 / 2 :=
by sorry

end ratio_jacob_edward_l57_57319


namespace cost_price_of_watch_l57_57086

/-
Let's state the problem conditions as functions
C represents the cost price
SP1 represents the selling price at 36% loss
SP2 represents the selling price at 4% gain
-/

def cost_price (C : ℝ) : ℝ := C

def selling_price_loss (C : ℝ) : ℝ := 0.64 * C

def selling_price_gain (C : ℝ) : ℝ := 1.04 * C

def price_difference (C : ℝ) : ℝ := (selling_price_gain C) - (selling_price_loss C)

theorem cost_price_of_watch : ∀ C : ℝ, price_difference C = 140 → C = 350 :=
by
   intro C H
   sorry

end cost_price_of_watch_l57_57086


namespace arithmetic_sequence_sum_l57_57049

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end arithmetic_sequence_sum_l57_57049


namespace find_xyz_squares_l57_57286

theorem find_xyz_squares (x y z : ℤ)
  (h1 : x + y + z = 3)
  (h2 : x^3 + y^3 + z^3 = 3) :
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 :=
sorry

end find_xyz_squares_l57_57286


namespace smallest_base_to_express_100_with_three_digits_l57_57720

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end smallest_base_to_express_100_with_three_digits_l57_57720


namespace inequality_not_true_l57_57436

variable {x y : ℝ}

theorem inequality_not_true (h : x > y) : ¬(-3 * x + 6 > -3 * y + 6) :=
by
  sorry

end inequality_not_true_l57_57436


namespace intersection_empty_l57_57294

open Set

def A : Set ℝ := {x | x^2 + 2*x - 3 < 0}
def B : Set ℝ := {-3, 1, 2}

theorem intersection_empty : A ∩ B = ∅ := by
  sorry

end intersection_empty_l57_57294


namespace range_of_a_l57_57139

theorem range_of_a (a : ℝ) (h : ∅ ⊂ {x : ℝ | x^2 ≤ a}) : 0 ≤ a :=
by
  sorry

end range_of_a_l57_57139


namespace pascal_fifth_number_l57_57487

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57487


namespace gcd_of_sum_of_four_consecutive_integers_l57_57638

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l57_57638


namespace no_roots_of_form_one_over_n_l57_57929

theorem no_roots_of_form_one_over_n (a b c : ℤ) (h_a : a % 2 = 1) (h_b : b % 2 = 1) (h_c : c % 2 = 1) :
  ∀ n : ℕ, ¬(a * (1 / (n:ℚ))^2 + b * (1 / (n:ℚ)) + c = 0) := by
  sorry

end no_roots_of_form_one_over_n_l57_57929


namespace pascal_triangle_fifth_number_l57_57513

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57513


namespace negation_of_universal_prop_l57_57213

theorem negation_of_universal_prop :
  (¬ (∀ x : ℝ, x^2 - 5 * x + 3 ≤ 0)) ↔ (∃ x : ℝ, x^2 - 5 * x + 3 > 0) :=
by sorry

end negation_of_universal_prop_l57_57213


namespace gcd_factorial_l57_57760

theorem gcd_factorial (a b : ℕ) (h : a = 8! ∧ b = (6!)^2) : Nat.gcd a b = 5760 :=
by sorry

end gcd_factorial_l57_57760


namespace pascal_triangle_fifth_number_l57_57495

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57495


namespace find_a_and_b_l57_57306

theorem find_a_and_b (a b : ℝ) (h1 : b ≠ 0) 
  (h2 : (ab = a + b ∨ ab = a - b ∨ ab = a / b) 
  ∧ (a + b = a - b ∨ a + b = a / b) 
  ∧ (a - b = a / b)) : 
  (a = 1 / 2 ∨ a = -1 / 2) ∧ b = -1 := by
  sorry

end find_a_and_b_l57_57306


namespace smallest_base_l57_57711

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end smallest_base_l57_57711


namespace probability_of_number_3_l57_57743

-- Definition: A fair six-sided die has outcomes from 1 to 6
def fair_six_sided_die : finset ℕ := {1, 2, 3, 4, 5, 6}

-- Definition: Event of interest: number facing up is 3
def event_number_3 : set ℕ := {3}

-- The probability of a specific event in a uniform probability space
-- is the ratio of the number of favorable outcomes 
-- to the number of possible outcomes
noncomputable def probability (event : set ℕ) (outcomes : finset ℕ) : ℝ :=
  (finset.card (outcomes ∩ event.to_finset)).to_real / (finset.card outcomes).to_real

-- The assertion to be proved
theorem probability_of_number_3 :
  probability event_number_3 fair_six_sided_die = (1 : ℝ) / (6 : ℝ) :=
sorry

end probability_of_number_3_l57_57743


namespace find_nat_numbers_l57_57432

theorem find_nat_numbers (a b : ℕ) (h : 1 / (a - b) = 3 * (1 / (a * b))) : a = 6 ∧ b = 2 :=
sorry

end find_nat_numbers_l57_57432


namespace pascal_triangle_fifth_number_l57_57500

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57500


namespace age_of_25th_student_l57_57345

-- Definitions derived from problem conditions
def averageAgeClass (totalAge : ℕ) (totalStudents : ℕ) : ℕ := totalAge / totalStudents
def totalAgeGivenAverage (numStudents : ℕ) (averageAge : ℕ) : ℕ := numStudents * averageAge

-- Given conditions
def totalAgeOfAllStudents := 25 * 24
def totalAgeOf8Students := totalAgeGivenAverage 8 22
def totalAgeOf10Students := totalAgeGivenAverage 10 20
def totalAgeOf6Students := totalAgeGivenAverage 6 28
def totalAgeOf24Students := totalAgeOf8Students + totalAgeOf10Students + totalAgeOf6Students

-- The proof that the age of the 25th student is 56 years
theorem age_of_25th_student : totalAgeOfAllStudents - totalAgeOf24Students = 56 := by
  sorry

end age_of_25th_student_l57_57345


namespace volume_of_union_of_regular_tetrahedrons_l57_57221

-- Definitions for the conditions set up
def unit_cube_vertices : Set (ℝ × ℝ × ℝ) := {
  (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
  (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
}

-- Given two specific regular tetrahedrons A and B given by vertices of the cube
def tet_A : Set (ℝ × ℝ × ℝ) := { (0, 0, 0), (1, 0, 1), (0, 1, 1), (1, 1, 0) }
def tet_B : Set (ℝ × ℝ × ℝ) := { (0, 0, 1), (1, 1, 1), (0, 1, 0), (1, 0, 0) }

-- Statement to prove the volume of the union of these tetrahedrons
theorem volume_of_union_of_regular_tetrahedrons (A B : Set (ℝ × ℝ × ℝ)) (hA : A = tet_A) (hB : B = tet_B) :
  volume (A ∪ B) = 1 / 2 :=
  sorry

end volume_of_union_of_regular_tetrahedrons_l57_57221


namespace yellow_highlighters_l57_57480

def highlighters (pink blue yellow total : Nat) : Prop :=
  (pink + blue + yellow = total)

theorem yellow_highlighters (h : highlighters 3 5 y 15) : y = 7 :=
by 
  sorry

end yellow_highlighters_l57_57480


namespace Irene_age_is_46_l57_57942

-- Definitions based on the given conditions
def Eddie_age : ℕ := 92
def Becky_age : ℕ := Eddie_age / 4
def Irene_age : ℕ := 2 * Becky_age

-- Theorem we aim to prove that Irene's age is 46
theorem Irene_age_is_46 : Irene_age = 46 := by
  sorry

end Irene_age_is_46_l57_57942


namespace circulation_ratio_l57_57735

theorem circulation_ratio (A C_1971 C_total : ℕ) 
(hC1971 : C_1971 = 4 * A) 
(hCtotal : C_total = C_1971 + 9 * A) : 
(C_1971 : ℚ) / (C_total : ℚ) = 4 / 13 := 
sorry

end circulation_ratio_l57_57735


namespace whisky_replacement_l57_57401

variable (V : ℝ) (x : ℝ)

theorem whisky_replacement (h_condition : 0.40 * V - 0.40 * x + 0.19 * x = 0.26 * V) : 
  x = (2 / 3) * V := 
sorry

end whisky_replacement_l57_57401


namespace tangent_intersects_x_axis_l57_57218

theorem tangent_intersects_x_axis (x0 x1 : ℝ) (hx : ∀ x : ℝ, x1 = x0 - 1) :
  x1 - x0 = -1 :=
by
  sorry

end tangent_intersects_x_axis_l57_57218


namespace suji_present_age_l57_57733

/-- Present ages of Abi and Suji are in the ratio of 5:4. --/
def abi_suji_ratio (abi_age suji_age : ℕ) : Prop := abi_age = 5 * (suji_age / 4)

/-- 3 years hence, the ratio of their ages will be 11:9. --/
def abi_suji_ratio_future (abi_age suji_age : ℕ) : Prop :=
  ((abi_age + 3).toFloat / (suji_age + 3).toFloat) = 11 / 9

theorem suji_present_age (suji_age : ℕ) (abi_age : ℕ) (x : ℕ) 
  (h1 : abi_age = 5 * x) (h2 : suji_age = 4 * x)
  (h3 : abi_suji_ratio_future abi_age suji_age) :
  suji_age = 24 := 
sorry

end suji_present_age_l57_57733


namespace pascal_row_fifth_number_l57_57538

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57538


namespace gcd_factorial_eight_six_sq_l57_57780

/-- Prove that the gcd of 8! and (6!)^2 is 11520 -/
theorem gcd_factorial_eight_six_sq : Int.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 11520 :=
by
  -- We'll put the proof here, but for now, we use sorry to omit it.
  sorry

end gcd_factorial_eight_six_sq_l57_57780


namespace pascal_fifth_element_15th_row_l57_57601

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57601


namespace largest_integer_divides_difference_l57_57788

theorem largest_integer_divides_difference (n : ℕ) 
  (h_composite : ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ m * k = n) :
  6 ∣ (n^4 - n) :=
sorry

end largest_integer_divides_difference_l57_57788


namespace negation_correct_l57_57729

-- Define the statement to be negated
def original_statement (x : ℕ) : Prop := ∀ x : ℕ, x^2 ≠ 4

-- Define the negation of the original statement
def negated_statement (x : ℕ) : Prop := ∃ x : ℕ, x^2 = 4

-- Prove that the negation of the original statement is the given negated statement
theorem negation_correct : (¬ (∀ x : ℕ, x^2 ≠ 4)) ↔ (∃ x : ℕ, x^2 = 4) :=
by sorry

end negation_correct_l57_57729


namespace smallest_base_for_100_l57_57718

theorem smallest_base_for_100 :
  ∃ (b : ℕ), (b^2 ≤ 100) ∧ (100 < b^3) ∧ ∀ (b' : ℕ), (b'^2 ≤ 100) ∧ (100 < b'^3) → b ≤ b' :=
by
  sorry

end smallest_base_for_100_l57_57718


namespace red_pencils_in_box_l57_57219

theorem red_pencils_in_box (B R G : ℕ) 
  (h1 : B + R + G = 20)
  (h2 : B = 6 * G)
  (h3 : R < B) : R = 6 := by
  sorry

end red_pencils_in_box_l57_57219


namespace Pascal_triangle_fifth_number_l57_57558

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57558


namespace irene_age_is_46_l57_57943

def eddie_age : ℕ := 92

def becky_age (e_age : ℕ) : ℕ := e_age / 4

def irene_age (b_age : ℕ) : ℕ := 2 * b_age

theorem irene_age_is_46 : irene_age (becky_age eddie_age) = 46 := 
  by
    sorry

end irene_age_is_46_l57_57943


namespace largest_divisor_composite_difference_l57_57795

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_composite_difference_l57_57795


namespace sandra_socks_l57_57193

variables (x y z : ℕ)

theorem sandra_socks :
  x + y + z = 15 →
  2 * x + 3 * y + 5 * z = 36 →
  x ≤ 6 →
  y ≤ 6 →
  z ≤ 6 →
  x = 11 :=
by
  sorry

end sandra_socks_l57_57193


namespace distance_between_opposite_vertices_l57_57818

noncomputable def calculate_d (a b c v k t : ℝ) : ℝ :=
  (1 / (2 * k)) * Real.sqrt (2 * (k^4 - 16 * t^2 - 8 * v * k))

theorem distance_between_opposite_vertices (a b c v k t d : ℝ)
  (h1 : v = a * b * c)
  (h2 : k = a + b + c)
  (h3 : 16 * t^2 = k * (k - 2 * a) * (k - 2 * b) * (k - 2 * c))
  : d = calculate_d a b c v k t := 
by {
    -- The proof is omitted based on the requirement.
    sorry
}

end distance_between_opposite_vertices_l57_57818


namespace fourth_ball_black_probability_l57_57394

-- Definitions from the conditions
def total_balls : ℕ := 6
def black_balls : ℕ := 3
def red_balls : ℕ := 3

theorem fourth_ball_black_probability :
  (4 : ℕ) ∈ Finset.range (total_balls + 1) →
  (black_balls + red_balls = total_balls) →
  (red_balls = 3) →
  (black_balls = 3) →
  sorry 
  -- The probability that the fourth ball selected is black is 1/2 
  -- Possible implementation could leverage definitions and requisite libraries.

end fourth_ball_black_probability_l57_57394


namespace valid_three_digit_numbers_l57_57723

   noncomputable def three_digit_num_correct (A : ℕ) : Prop :=
     (100 ≤ A ∧ A < 1000) ∧ (1000000 + A = A * A)

   theorem valid_three_digit_numbers (A : ℕ) :
     three_digit_num_correct A → (A = 625 ∨ A = 376) :=
   by
     sorry
   
end valid_three_digit_numbers_l57_57723


namespace eval_sum_and_subtract_l57_57757

theorem eval_sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by {
  -- The rest of the proof should go here, but we'll use sorry to skip it.
  sorry
}

end eval_sum_and_subtract_l57_57757


namespace min_positive_period_and_min_value_l57_57684

noncomputable def f (x : ℝ) : ℝ :=
  1 + (1/2) * Real.sin (2 * x)

theorem min_positive_period_and_min_value :
  (∀ (x : ℝ), f(x + π) = f(x)) ∧ (∃ (x : ℝ), f(x) = 1/2) :=
by
  sorry

end min_positive_period_and_min_value_l57_57684


namespace length_of_plot_correct_l57_57388

noncomputable def length_of_plot (b : ℕ) : ℕ := b + 30

theorem length_of_plot_correct (b : ℕ) (cost_per_meter total_cost : ℝ) 
    (h1 : length_of_plot b = b + 30)
    (h2 : cost_per_meter = 26.50)
    (h3 : total_cost = 5300)
    (h4 : 2 * (b + (b + 30)) * cost_per_meter = total_cost) :
    length_of_plot 35 = 65 :=
by
  sorry

end length_of_plot_correct_l57_57388


namespace Kolya_Homework_Problem_l57_57003

-- Given conditions as definitions
def squaresToDigits (x : ℕ) (a b : ℕ) : Prop := x^2 = 10 * a + b
def doubledToDigits (x : ℕ) (a b : ℕ) : Prop := 2 * x = 10 * b + a

-- The main theorem statement
theorem Kolya_Homework_Problem :
  ∃ (x a b : ℕ), squaresToDigits x a b ∧ doubledToDigits x a b ∧ x = 9 ∧ x^2 = 81 :=
by
  -- proof skipped
  sorry

end Kolya_Homework_Problem_l57_57003


namespace find_factorial_number_l57_57279

def is_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def is_factorial_number (n : ℕ) : Prop :=
  ∃ x y z : ℕ, (0 ≤ x ∧ x ≤ 5) ∧
               (0 ≤ y ∧ y ≤ 5) ∧
               (0 ≤ z ∧ z ≤ 5) ∧
               n = 100 * x + 10 * y + z ∧
               n = x.factorial + y.factorial + z.factorial

theorem find_factorial_number : ∃ n, is_three_digit_number n ∧ is_factorial_number n ∧ n = 145 :=
by {
  sorry
}

end find_factorial_number_l57_57279


namespace find_prob_Y_ge_1_l57_57115

variable {Ω : Type*} [MeasureSpace Ω]

-- Define binomial random variables X and Y
noncomputable def binomial (n : ℕ) (p : ℝ) : MeasureTheory.ProbabilityMassFunction (fin (n + 1)) :=
sorry

-- Define the random variables X and Y as binomial distributions
constant p : ℝ
constant X : MeasureTheory.ProbabilityMassFunction (fin 3)
constant Y : MeasureTheory.ProbabilityMassFunction (fin 4)

-- Conditions
axiom hX : X = binomial 2 p
axiom hY : Y = binomial 3 p
axiom hPX : X.prob (λ k, k ≥ 1) = 3 / 4

-- Theorem to prove
theorem find_prob_Y_ge_1 : Y.prob (λ k, k ≥ 1) = 7 / 8 :=
by {
  sorry
}

end find_prob_Y_ge_1_l57_57115


namespace statement_bug_travel_direction_l57_57186

/-
  Theorem statement: On a plane with a grid formed by regular hexagons of side length 1,
  if a bug traveled from node A to node B along the shortest path of 100 units,
  then the bug traveled exactly 50 units in one direction.
-/
theorem bug_travel_direction (side_length : ℝ) (total_distance : ℝ) 
  (hexagonal_grid : Π (x y : ℝ), Prop) (A B : ℝ × ℝ) 
  (shortest_path : ℝ) :
  side_length = 1 ∧ shortest_path = 100 →
  ∃ (directional_travel : ℝ), directional_travel = 50 :=
by
  sorry

end statement_bug_travel_direction_l57_57186


namespace fifth_number_in_pascal_row_l57_57616

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57616


namespace abs_sum_div_diff_sqrt_7_5_l57_57132

theorem abs_sum_div_diff_sqrt_7_5 (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (h : a^2 + b^2 = 12 * a * b) :
  abs ((a + b) / (a - b)) = Real.sqrt (7 / 5) :=
by
  sorry

end abs_sum_div_diff_sqrt_7_5_l57_57132


namespace exponent_multiplication_l57_57064

variable (x : ℤ)

theorem exponent_multiplication :
  (-x^2) * x^3 = -x^5 :=
sorry

end exponent_multiplication_l57_57064


namespace gcd_factorial_l57_57761

theorem gcd_factorial (a b : ℕ) (h : a = 8! ∧ b = (6!)^2) : Nat.gcd a b = 5760 :=
by sorry

end gcd_factorial_l57_57761


namespace avg_age_of_14_students_l57_57875

theorem avg_age_of_14_students (avg_age_25 : ℕ) (avg_age_10 : ℕ) (age_25th : ℕ) (total_students : ℕ) (remaining_students : ℕ) :
  avg_age_25 = 25 →
  avg_age_10 = 22 →
  age_25th = 13 →
  total_students = 25 →
  remaining_students = 14 →
  ( (total_students * avg_age_25) - (10 * avg_age_10) - age_25th ) / remaining_students = 28 :=
by
  intros
  sorry

end avg_age_of_14_students_l57_57875


namespace train_crossing_time_l57_57928

noncomputable def length_of_train : ℝ := 120 -- meters
noncomputable def speed_of_train_kmh : ℝ := 27 -- kilometers per hour
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmh * (1000 / 3600) -- converted to meters per second
noncomputable def time_to_cross : ℝ := length_of_train / speed_of_train_mps

theorem train_crossing_time :
  time_to_cross = 16 :=
by
  -- proof goes here
  sorry

end train_crossing_time_l57_57928


namespace arrange_natural_numbers_divisors_l57_57844

theorem arrange_natural_numbers_divisors :
  ∃ (seq : List ℕ), seq = [7, 1, 8, 4, 10, 6, 9, 3, 2, 5] ∧ 
  seq.length = 10 ∧
  ∀ n (h : n < seq.length), seq[n] ∣ (List.take n seq).sum := 
by
  sorry

end arrange_natural_numbers_divisors_l57_57844


namespace complex_multiplication_l57_57877

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i := by
  sorry

end complex_multiplication_l57_57877


namespace total_cost_for_trip_l57_57130

def cost_of_trip (students : ℕ) (teachers : ℕ) (seats_per_bus : ℕ) (cost_per_bus : ℕ) (toll_per_bus : ℕ) : ℕ :=
  let total_people := students + teachers
  let buses_required := (total_people + seats_per_bus - 1) / seats_per_bus -- ceiling division
  let total_rent_cost := buses_required * cost_per_bus
  let total_toll_cost := buses_required * toll_per_bus
  total_rent_cost + total_toll_cost

theorem total_cost_for_trip
  (students : ℕ := 252)
  (teachers : ℕ := 8)
  (seats_per_bus : ℕ := 41)
  (cost_per_bus : ℕ := 300000)
  (toll_per_bus : ℕ := 7500) :
  cost_of_trip students teachers seats_per_bus cost_per_bus toll_per_bus = 2152500 := by
  sorry -- Proof to be filled

end total_cost_for_trip_l57_57130


namespace eden_initial_bears_l57_57095

theorem eden_initial_bears (d_total : ℕ) (d_favorite : ℕ) (sisters : ℕ) (eden_after : ℕ) (each_share : ℕ)
  (h1 : d_total = 20)
  (h2 : d_favorite = 8)
  (h3 : sisters = 3)
  (h4 : eden_after = 14)
  (h5 : each_share = (d_total - d_favorite) / sisters)
  : (eden_after - each_share) = 10 :=
by
  sorry

end eden_initial_bears_l57_57095


namespace find_smaller_angle_l57_57680

theorem find_smaller_angle (x : ℝ) (h1 : (x + (x + 18) = 180)) : x = 81 := 
by 
  sorry

end find_smaller_angle_l57_57680


namespace largest_divisor_of_n_pow4_minus_n_l57_57803

theorem largest_divisor_of_n_pow4_minus_n (n : ℕ) (h : ¬ prime n ∧ n > 1) : ∃ d, d = 6 ∧ ∀ k, k ∣ n * (n - 1) * (n + 1) * (n^2 + n + 1) → k ≤ 6 := sorry

end largest_divisor_of_n_pow4_minus_n_l57_57803


namespace shifted_line_does_not_pass_through_third_quadrant_l57_57728

-- The condition: The original line is y = -2x - 1
def original_line (x : ℝ) : ℝ := -2 * x - 1

-- The condition: The line is shifted 3 units to the right
def shifted_line (x : ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_through_third_quadrant :
  ¬(∃ (x y : ℝ), y = shifted_line x ∧ x < 0 ∧ y < 0) :=
sorry

end shifted_line_does_not_pass_through_third_quadrant_l57_57728


namespace shaded_area_fraction_l57_57901

theorem shaded_area_fraction (ABCD_area : ℝ) (shaded_square1_area : ℝ) (shaded_rectangle_area : ℝ) (shaded_square2_area : ℝ) (total_shaded_area : ℝ)
  (h_ABCD : ABCD_area = 36) 
  (h_shaded_square1 : shaded_square1_area = 4)
  (h_shaded_rectangle : shaded_rectangle_area = 12)
  (h_shaded_square2 : shaded_square2_area = 36)
  (h_total_shaded : total_shaded_area = 16) :
  (total_shaded_area / ABCD_area) = 4 / 9 :=
by 
  simp [h_ABCD, h_total_shaded]
  sorry

end shaded_area_fraction_l57_57901


namespace largest_divisor_of_n_pow4_minus_n_l57_57806

theorem largest_divisor_of_n_pow4_minus_n (n : ℕ) (h : ¬ prime n ∧ n > 1) : ∃ d, d = 6 ∧ ∀ k, k ∣ n * (n - 1) * (n + 1) * (n^2 + n + 1) → k ≤ 6 := sorry

end largest_divisor_of_n_pow4_minus_n_l57_57806


namespace positive_number_percentage_of_itself_is_9_l57_57922

theorem positive_number_percentage_of_itself_is_9 (x : ℝ) (hx_pos : 0 < x) (h_condition : 0.01 * x^2 = 9) : x = 30 :=
by
  sorry

end positive_number_percentage_of_itself_is_9_l57_57922


namespace not_q_is_false_l57_57290

variable (n : ℤ)

-- Definition of the propositions
def p (n : ℤ) : Prop := 2 * n - 1 % 2 = 1 -- 2n - 1 is odd
def q (n : ℤ) : Prop := (2 * n + 1) % 2 = 0 -- 2n + 1 is even

-- Proof statement: Not q is false, meaning q is false
theorem not_q_is_false (n : ℤ) : ¬ q n = False := sorry

end not_q_is_false_l57_57290


namespace daily_wage_male_worker_l57_57740

variables
  (num_male : ℕ) (num_female : ℕ) (num_child : ℕ)
  (wage_female : ℝ) (wage_child : ℝ) (avg_wage : ℝ)
  (total_workers : ℕ := num_male + num_female + num_child)
  (total_wage_all : ℝ := avg_wage * total_workers)
  (total_wage_female : ℝ := num_female * wage_female)
  (total_wage_child : ℝ := num_child * wage_child)
  (total_wage_male : ℝ := total_wage_all - (total_wage_female + total_wage_child))
  (wage_per_male : ℝ := total_wage_male / num_male)

theorem daily_wage_male_worker :
  num_male = 20 →
  num_female = 15 →
  num_child = 5 →
  wage_female = 20 →
  wage_child = 8 →
  avg_wage = 21 →
  wage_per_male = 25 :=
by
  intros
  sorry

end daily_wage_male_worker_l57_57740


namespace younger_age_is_12_l57_57874

theorem younger_age_is_12 
  (y elder : ℕ)
  (h_diff : elder = y + 20)
  (h_past : elder - 7 = 5 * (y - 7)) :
  y = 12 :=
by
  sorry

end younger_age_is_12_l57_57874


namespace power_equation_l57_57310

theorem power_equation (x a b : ℝ) (ha : 3^x = a) (hb : 5^x = b) : 45^x = a^2 * b :=
sorry

end power_equation_l57_57310


namespace prob_purchase_either_is_correct_prob_purchase_at_least_one_is_correct_prob_dist_correct_l57_57243

noncomputable def prob_purchase_either : ℝ := 0.5
noncomputable def prob_purchase_A : ℝ := 0.5
noncomputable def prob_purchase_B : ℝ := 0.6
noncomputable def prob_purchase_neither_A_B : ℝ := 0.2

theorem prob_purchase_either_is_correct :
  prob_purchase_either = prob_purchase_A * (1 - prob_purchase_B) + (1 - prob_purchase_A) * prob_purchase_B := 
sorry

theorem prob_purchase_at_least_one_is_correct :
  1 - (1 - prob_purchase_A) * (1 - prob_purchase_B) = 0.8 :=
sorry

noncomputable def binom_dist : ℕ → ℝ
| 0 := 0.2^3
| 1 := 3 * 0.8 * 0.2^2
| 2 := 3 * 0.8^2 * 0.2
| 3 := 0.8^3
| _ := 0

theorem prob_dist_correct :
  binom_dist 0 = 0.008 ∧ binom_dist 1 = 0.096 ∧ binom_dist 2 = 0.384 ∧ binom_dist 3 = 0.512 :=
sorry

end prob_purchase_either_is_correct_prob_purchase_at_least_one_is_correct_prob_dist_correct_l57_57243


namespace log_graph_passes_fixed_point_l57_57879

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem log_graph_passes_fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) :
  log_a a (-1 + 2) = 0 :=
by
  sorry

end log_graph_passes_fixed_point_l57_57879


namespace machine_A_production_is_4_l57_57659

noncomputable def machine_production (A : ℝ) (B : ℝ) (T_A : ℝ) (T_B : ℝ) := 
  (440 / A = T_A) ∧
  (440 / B = T_B) ∧
  (T_A = T_B + 10) ∧
  (B = 1.10 * A)

theorem machine_A_production_is_4 {A B T_A T_B : ℝ}
  (h : machine_production A B T_A T_B) : 
  A = 4 :=
by
  sorry

end machine_A_production_is_4_l57_57659


namespace problem_statement_l57_57228

theorem problem_statement : 2017 - (1 / 2017) = (2018 * 2016) / 2017 :=
by
  sorry

end problem_statement_l57_57228


namespace multiples_count_l57_57456

theorem multiples_count :
  let n := 200,
      count_multiples (k : ℕ) := n / k,
      lcm := Nat.lcm 6 8
  in
  (count_multiples 6 - count_multiples lcm) + (count_multiples 8 - count_multiples lcm) = 42 :=
by
  sorry

end multiples_count_l57_57456


namespace trapezoid_diagonal_comparison_l57_57188

variable {A B C D: Type}
variable (α β : Real) -- Representing angles
variable (AB CD BD AC : Real) -- Representing lengths of sides and diagonals
variable (h : Real) -- Height
variable (A' B' : Real) -- Projections

noncomputable def trapezoid (AB CD: Real) := True -- Trapezoid definition placeholder
noncomputable def angle_relation (α β : Real) := α < β -- Angle relationship

theorem trapezoid_diagonal_comparison
  (trapezoid_ABCD: trapezoid AB CD)
  (angle_relation_ABC_DCB : angle_relation α β)
  : BD > AC :=
sorry

end trapezoid_diagonal_comparison_l57_57188


namespace ratio_monkeys_snakes_l57_57087

def parrots : ℕ := 8
def snakes : ℕ := 3 * parrots
def elephants : ℕ := (parrots + snakes) / 2
def zebras : ℕ := elephants - 3
def monkeys : ℕ := zebras + 35

theorem ratio_monkeys_snakes : (monkeys : ℕ) / (snakes : ℕ) = 2 / 1 :=
by
  sorry

end ratio_monkeys_snakes_l57_57087


namespace convex_polyhedron_triangular_face_or_three_edges_vertex_l57_57865

theorem convex_polyhedron_triangular_face_or_three_edges_vertex
  (M N K : ℕ) 
  (euler_formula : N - M + K = 2) :
  ∃ (f : ℕ), (f ≤ N ∧ f = 3) ∨ ∃ (v : ℕ), (v ≤ K ∧ v = 3) := 
sorry

end convex_polyhedron_triangular_face_or_three_edges_vertex_l57_57865


namespace greatest_common_divisor_of_B_l57_57625

def B : Set ℕ := { n | ∃ x : ℕ, n = 4 * x + 6 }

theorem greatest_common_divisor_of_B : ∃ d : ℕ, IsGreatestCommonDivisor B d ∧ d = 2 := by
  sorry

end greatest_common_divisor_of_B_l57_57625


namespace ratio_of_areas_l57_57687

theorem ratio_of_areas (len_rect width_rect area_tri : ℝ) (h1 : len_rect = 6) (h2 : width_rect = 4) (h3 : area_tri = 60) :
    (len_rect * width_rect) / area_tri = 2 / 5 :=
by
  rw [h1, h2, h3]
  norm_num

end ratio_of_areas_l57_57687


namespace find_k_l57_57307

-- Defining the vectors
def a (k : ℝ) : ℝ × ℝ := (k, -2)
def b : ℝ × ℝ := (2, 2)

-- Condition 1: a + b is not the zero vector
def non_zero_sum (k : ℝ) := (a k).1 + b.1 ≠ 0 ∨ (a k).2 + b.2 ≠ 0

-- Condition 2: a is perpendicular to a + b
def perpendicular (k : ℝ) := (a k).1 * ((a k).1 + b.1) + (a k).2 * ((a k).2 + b.2) = 0

-- The theorem to prove
theorem find_k (k : ℝ) (cond1 : non_zero_sum k) (cond2 : perpendicular k) : k = 0 := 
sorry

end find_k_l57_57307


namespace stan_weighs_5_more_than_steve_l57_57873

theorem stan_weighs_5_more_than_steve
(S V J : ℕ) 
(h1 : J = 110)
(h2 : V = J - 8)
(h3 : S + V + J = 319) : 
(S - V = 5) :=
by
  sorry

end stan_weighs_5_more_than_steve_l57_57873


namespace tel_aviv_rain_probability_l57_57353

def binom (n k : ℕ) : ℕ := Nat.choose n k

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binom n k : ℝ) * (p ^ k) * ((1 - p) ^ (n - k))

theorem tel_aviv_rain_probability :
  binomial_probability 6 4 0.5 = 0.234375 :=
by
  sorry

end tel_aviv_rain_probability_l57_57353


namespace infinite_impossible_values_of_d_l57_57359

theorem infinite_impossible_values_of_d 
  (pentagon_perimeter square_perimeter : ℕ) 
  (d : ℕ) 
  (h1 : pentagon_perimeter = 5 * (d + ((square_perimeter) / 4)) )
  (h2 : square_perimeter > 0)
  (h3 : pentagon_perimeter - square_perimeter = 2023) :
  ∀ n : ℕ, n > 404 → ¬∃ d : ℕ, d = n :=
by {
  sorry
}

end infinite_impossible_values_of_d_l57_57359


namespace arithmetic_sequence_sum_l57_57063

-- Define the arithmetic sequence properties
def seq : List ℕ := [81, 83, 85, 87, 89, 91, 93, 95, 97, 99]
def first := 81
def last := 99
def common_diff := 2
def n := 10

-- Main theorem statement proving the desired property
theorem arithmetic_sequence_sum :
  2 * (seq.sum) = 1800 := by
  sorry

end arithmetic_sequence_sum_l57_57063


namespace largest_divisor_of_n_pow4_minus_n_l57_57805

theorem largest_divisor_of_n_pow4_minus_n (n : ℕ) (h : ¬ prime n ∧ n > 1) : ∃ d, d = 6 ∧ ∀ k, k ∣ n * (n - 1) * (n + 1) * (n^2 + n + 1) → k ≤ 6 := sorry

end largest_divisor_of_n_pow4_minus_n_l57_57805


namespace positive_integer_representation_l57_57103

theorem positive_integer_representation (a b c n : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) 
  (h₄ : n = (abc + a * b + a) / (abc + c * b + c)) : n = 1 ∨ n = 2 := 
by
  sorry

end positive_integer_representation_l57_57103


namespace find_f_l57_57975

def f : ℝ → ℝ := sorry

theorem find_f (x : ℝ) : f (x + 2) = 2 * x + 3 → f x = 2 * x - 1 :=
by
  intro h
  -- Proof goes here 
  sorry

end find_f_l57_57975


namespace count_multiples_6_or_8_not_both_l57_57461

theorem count_multiples_6_or_8_not_both : 
  let count_multiples (n m : ℕ) (limit : ℕ) := limit / m
  let lcm := 24
  let limit := 200
  let multiples_6 := count_multiples limit 6
  let multiples_8 := count_multiples limit 8
  let multiples_both := count_multiples limit lcm
in
  (multiples_6 - multiples_both) + (multiples_8 - multiples_both) = 42 :=
by sorry

end count_multiples_6_or_8_not_both_l57_57461


namespace radius_calculation_l57_57387

noncomputable def radius_of_circle (n : ℕ) : ℝ :=
if 2 ≤ n ∧ n ≤ 11 then
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61
else
  0  -- Outside the specified range

theorem radius_calculation (n : ℕ) (hn : 2 ≤ n ∧ n ≤ 11) :
  radius_of_circle n =
  if n ≤ 7 then 1 else
  if n = 8 then 1.15 else
  if n = 9 then 1.30 else
  if n = 10 then 1.46 else
  1.61 :=
sorry

end radius_calculation_l57_57387


namespace rectangle_area_y_l57_57265

theorem rectangle_area_y (y : ℚ) (h_pos: y > 0) 
  (h_area: ((6 : ℚ) - (-2)) * (y - 2) = 64) : y = 10 :=
by
  sorry

end rectangle_area_y_l57_57265


namespace smallest_base_l57_57712

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end smallest_base_l57_57712


namespace fractional_eq_solution_l57_57338

theorem fractional_eq_solution : ∀ x : ℝ, (x ≠ 3) → ((2 - x) / (x - 3) + 1 / (3 - x) = 1) → (x = 2) :=
by
  intros x h_cond h_eq
  sorry

end fractional_eq_solution_l57_57338


namespace total_matches_round_robin_l57_57408

/-- A round-robin chess tournament is organized in two groups with different numbers of players. 
Group A consists of 6 players, and Group B consists of 5 players. 
Each player in each group plays every other player in the same group exactly once. 
Prove that the total number of matches is 25. -/
theorem total_matches_round_robin 
  (nA : ℕ) (nB : ℕ) 
  (hA : nA = 6) (hB : nB = 5) : 
  (nA * (nA - 1) / 2) + (nB * (nB - 1) / 2) = 25 := 
  by
    sorry

end total_matches_round_robin_l57_57408


namespace total_tweets_is_correct_l57_57862

-- Define the conditions of Polly's tweeting behavior and durations
def happy_tweets := 18
def hungry_tweets := 4
def mirror_tweets := 45
def duration := 20

-- Define the total tweets calculation
def total_tweets := duration * happy_tweets + duration * hungry_tweets + duration * mirror_tweets

-- Prove that the total number of tweets is 1340
theorem total_tweets_is_correct : total_tweets = 1340 := by
  sorry

end total_tweets_is_correct_l57_57862


namespace solve_fractional_equation_l57_57340

noncomputable def fractional_equation (x : ℝ) : Prop :=
  (2 - x) / (x - 3) + 1 / (3 - x) = 1

theorem solve_fractional_equation :
  ∀ x : ℝ, x ≠ 3 → fractional_equation x ↔ x = 2 :=
by
  intros x h
  unfold fractional_equation
  sorry

end solve_fractional_equation_l57_57340


namespace regular_hexagon_area_decrease_l57_57080

noncomputable def area_decrease (original_area : ℝ) (side_decrease : ℝ) : ℝ :=
  let s := (2 * original_area) / (3 * Real.sqrt 3)
  let new_side := s - side_decrease
  let new_area := (3 * Real.sqrt 3 / 2) * new_side ^ 2
  original_area - new_area

theorem regular_hexagon_area_decrease :
  area_decrease (150 * Real.sqrt 3) 3 = 76.5 * Real.sqrt 3 :=
by
  sorry

end regular_hexagon_area_decrease_l57_57080


namespace solve_for_multiplier_l57_57230

namespace SashaSoup
  
-- Variables representing the amounts of salt
variables (x y : ℝ)

-- Condition provided: amount of salt added today
def initial_salt := 2 * x
def additional_salt_today := 0.5 * y

-- Given relationship
axiom salt_relationship : x = 0.5 * y

-- The multiplier k to achieve the required amount of salt
def required_multiplier : ℝ := 1.5

-- Lean theorem statement
theorem solve_for_multiplier :
  (2 * x) * required_multiplier = x + y :=
by
  -- Mathematical proof goes here but since asked to skip proof we use sorry
  sorry

end SashaSoup

end solve_for_multiplier_l57_57230


namespace digit_1035_is_2_l57_57851

noncomputable def sequence_digits (n : ℕ) : ℕ :=
  -- Convert the sequence of numbers from 1 to n to digits and return a specific position.
  sorry

theorem digit_1035_is_2 : sequence_digits 500 = 2 :=
  sorry

end digit_1035_is_2_l57_57851


namespace pascal_fifth_element_15th_row_l57_57598

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57598


namespace f_is_increasing_l57_57934

noncomputable theory

def is_monotonically_increasing (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x < f y

def f (x : ℝ) : ℝ := 7 * Real.sin (x - Real.pi / 6)

open Set

theorem f_is_increasing (f : ℝ → ℝ) (I : set ℝ) (h : ∀ x ∈ I, ∃ y ∈ I, x < y) : is_monotonically_increasing f I := sorry

example : is_monotonically_increasing f (Icc 0 (Real.pi / 2)) :=
begin
  unfold is_monotonically_increasing,
  intros x y hx hy hxy,
  have : 7 * Real.cos (x - Real.pi / 6) > 0 := sorry,
  have : 7 * Real.cos (y - Real.pi / 6) > 0 := sorry,
  calc
    f x = 7 * Real.sin (x - Real.pi / 6) : rfl
    ... < 7 * Real.sin (y - Real.pi / 6) : sorry
    ... = f y : rfl,
end

end f_is_increasing_l57_57934


namespace line_equation_l57_57104

theorem line_equation (P A B : ℝ × ℝ) (h1 : P = (-1, 3)) (h2 : A = (1, 2)) (h3 : B = (3, 1)) :
  ∃ c : ℝ, (x - 2*y + c = 0) ∧ (4*x - 2*y - 5 = 0) :=
by
  sorry

end line_equation_l57_57104


namespace pascal_fifth_number_in_row_15_l57_57555

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57555


namespace total_songs_time_l57_57689

-- Definitions of durations for each radio show
def duration_show1 : ℕ := 180
def duration_show2 : ℕ := 240
def duration_show3 : ℕ := 120

-- Definitions of talking segments for each show
def talking_segments_show1 : ℕ := 3 * 10  -- 3 segments, 10 minutes each
def talking_segments_show2 : ℕ := 4 * 15  -- 4 segments, 15 minutes each
def talking_segments_show3 : ℕ := 2 * 8   -- 2 segments, 8 minutes each

-- Definitions of ad breaks for each show
def ad_breaks_show1 : ℕ := 5 * 5  -- 5 breaks, 5 minutes each
def ad_breaks_show2 : ℕ := 6 * 4  -- 6 breaks, 4 minutes each
def ad_breaks_show3 : ℕ := 3 * 6  -- 3 breaks, 6 minutes each

-- Function to calculate time spent on songs for a given show
def time_spent_on_songs (duration talking ad_breaks : ℕ) : ℕ :=
  duration - talking - ad_breaks

-- Total time spent on songs for all three shows
def total_time_spent_on_songs : ℕ :=
  time_spent_on_songs duration_show1 talking_segments_show1 ad_breaks_show1 +
  time_spent_on_songs duration_show2 talking_segments_show2 ad_breaks_show2 +
  time_spent_on_songs duration_show3 talking_segments_show3 ad_breaks_show3

-- The theorem we want to prove
theorem total_songs_time : total_time_spent_on_songs = 367 := 
  sorry

end total_songs_time_l57_57689


namespace sum_product_of_pairs_l57_57857

theorem sum_product_of_pairs (x y z : ℝ) 
  (h1 : x + y + z = 20) 
  (h2 : x^2 + y^2 + z^2 = 200) :
  x * y + x * z + y * z = 100 := 
by
  sorry

end sum_product_of_pairs_l57_57857


namespace largest_divisor_of_n_l57_57909

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 12 ∣ n :=
by sorry

end largest_divisor_of_n_l57_57909


namespace nicholas_paid_more_than_kenneth_l57_57013

def price_per_yard : ℝ := 40
def kenneth_yards : ℝ := 700
def nicholas_multiplier : ℝ := 6
def discount_rate : ℝ := 0.15

def kenneth_total_cost : ℝ := price_per_yard * kenneth_yards
def nicholas_yards : ℝ := nicholas_multiplier * kenneth_yards
def nicholas_original_cost : ℝ := price_per_yard * nicholas_yards
def discount_amount : ℝ := discount_rate * nicholas_original_cost
def nicholas_discounted_cost : ℝ := nicholas_original_cost - discount_amount
def difference_in_cost : ℝ := nicholas_discounted_cost - kenneth_total_cost

theorem nicholas_paid_more_than_kenneth :
  difference_in_cost = 114800 := by
  sorry

end nicholas_paid_more_than_kenneth_l57_57013


namespace cos_A_minus_B_minus_3pi_div_2_l57_57813

theorem cos_A_minus_B_minus_3pi_div_2 (A B : ℝ)
  (h1 : Real.tan B = 2 * Real.tan A)
  (h2 : Real.cos A * Real.sin B = 4 / 5) :
  Real.cos (A - B - 3 * Real.pi / 2) = 2 / 5 := 
sorry

end cos_A_minus_B_minus_3pi_div_2_l57_57813


namespace arithmetic_seq_40th_term_l57_57383

theorem arithmetic_seq_40th_term (a₁ d : ℕ) (n : ℕ) (h1 : a₁ = 3) (h2 : d = 4) (h3 : n = 40) : 
  a₁ + (n - 1) * d = 159 :=
by
  sorry

end arithmetic_seq_40th_term_l57_57383


namespace max_k_value_l57_57967

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def B (i : ℕ) := {b : Finset ℕ // b ⊆ A ∧ b ≠ ∅ ∧ ∀ j ≠ i, ∃ k : Finset ℕ, k ⊆ A ∧ k ≠ ∅ ∧ (b ∩ k).card ≤ 2}

theorem max_k_value : ∃ k, k = 175 :=
  by
    sorry

end max_k_value_l57_57967


namespace Pascal_triangle_fifth_number_l57_57560

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57560


namespace exists_unique_root_in_interval_l57_57363

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 2

theorem exists_unique_root_in_interval : 
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 :=
sorry

end exists_unique_root_in_interval_l57_57363


namespace dylan_speed_constant_l57_57273

theorem dylan_speed_constant (d t s : ℝ) (h1 : d = 1250) (h2 : t = 25) (h3 : s = d / t) : s = 50 := 
by 
  -- Proof steps will go here
  sorry

end dylan_speed_constant_l57_57273


namespace probability_shadedRegion_l57_57936

noncomputable def triangleVertices : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((0, 0), (0, 5), (5, 0))

noncomputable def totalArea : ℝ :=
  12.5

noncomputable def shadedArea : ℝ :=
  4.5

theorem probability_shadedRegion (x y : ℝ) :
  let p := (x, y)
  let condition := x + y <= 3
  let totalArea := 12.5
  let shadedArea := 4.5
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 5 ∧ 0 ≤ p.2 ∧ p.2 ≤ 5 ∧ p.1 + p.2 ≤ 5}) →
  (p ∈ {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3 ∧ p.1 + p.2 ≤ 3}) →
  (shadedArea / totalArea) = 9/25 :=
by
  sorry

end probability_shadedRegion_l57_57936


namespace pascal_fifth_element_15th_row_l57_57597

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57597


namespace tangent_line_with_min_slope_inclination_angle_range_l57_57299

noncomputable def cubic_curve (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x + 1

theorem tangent_line_with_min_slope :
  ∃ x y : ℝ, cubic_curve 2 = 5/3 ∧ (deriv cubic_curve 2) = -1 ∧ (3 * x + 3 * y - 11 = 0) :=
by
  sorry

theorem inclination_angle_range :
  ∀ α : ℝ, (tan α ≥ -1) → (α ∈ set.Ico 0 (π / 2) ∪ set.Ico (3 * π / 4) π) :=
by
  sorry

end tangent_line_with_min_slope_inclination_angle_range_l57_57299


namespace pascal_fifth_number_in_row_15_l57_57549

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57549


namespace tan_identity_equality_l57_57443

theorem tan_identity_equality
  (α β : ℝ)
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β - π / 4) = 1 / 4) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = 3 / 22 :=
by
  sorry

end tan_identity_equality_l57_57443


namespace rain_probability_tel_aviv_l57_57349

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l57_57349


namespace number_of_true_propositions_l57_57685

theorem number_of_true_propositions :
  let P1 := false -- Swinging on a swing can be regarded as a translation motion.
  let P2 := false -- Two lines intersected by a third line have equal corresponding angles.
  let P3 := true  -- There is one and only one line passing through a point parallel to a given line.
  let P4 := false -- Angles that are not vertical angles are not equal.
  (if P1 then 1 else 0) + (if P2 then 1 else 0) + (if P3 then 1 else 0) + (if P4 then 1 else 0) = 1 :=
by
  sorry

end number_of_true_propositions_l57_57685


namespace pascal_triangle_row_fifth_number_l57_57546

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57546


namespace world_cup_teams_count_l57_57148

/-- In the world cup inauguration event, captains and vice-captains of all the teams are invited and awarded welcome gifts. There are some teams participating in the world cup, and 14 gifts are needed for this event. If each team has a captain and a vice-captain, and thus receives 2 gifts, then the number of teams participating is 7. -/
theorem world_cup_teams_count (total_gifts : ℕ) (gifts_per_team : ℕ) (teams : ℕ) 
  (h1 : total_gifts = 14) 
  (h2 : gifts_per_team = 2) 
  (h3 : total_gifts = teams * gifts_per_team) 
: teams = 7 :=
by sorry

end world_cup_teams_count_l57_57148


namespace no_real_roots_range_l57_57478

theorem no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_range_l57_57478


namespace toby_total_time_l57_57890

theorem toby_total_time :
  let speed_loaded := 10
  let speed_unloaded := 20
  let distance_part1 := 180
  let distance_part2 := 120
  let distance_part3 := 80
  let distance_part4 := 140
  (distance_part1 / speed_loaded) +
  (distance_part2 / speed_unloaded) +
  (distance_part3 / speed_loaded) +
  (distance_part4 / speed_unloaded) = 39 :=
by
  let speed_loaded := 10
  let speed_unloaded := 20
  let distance_part1 := 180
  let distance_part2 := 120
  let distance_part3 := 80
  let distance_part4 := 140
  have t1 := distance_part1 / speed_loaded
  have t2 := distance_part2 / speed_unloaded
  have t3 := distance_part3 / speed_loaded
  have t4 := distance_part4 / speed_unloaded
  calc t1 + t2 + t3 + t4 = 18 + 6 + 8 + 7 : by
       unfold t1 t2 t3 t4;
       sorry
                         .= 39 : by sorry

end toby_total_time_l57_57890


namespace fifth_number_in_pascals_triangle_l57_57587

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57587


namespace parabola_constant_term_l57_57920

theorem parabola_constant_term :
  ∃ b c : ℝ, (∀ x : ℝ, (x = 2 → 3 = x^2 + b * x + c) ∧ (x = 4 → 3 = x^2 + b * x + c)) → c = 11 :=
by
  sorry

end parabola_constant_term_l57_57920


namespace find_extrema_A_l57_57255

def eight_digit_number(n : ℕ) : Prop := n ≥ 10^7 ∧ n < 10^8

def coprime_with_thirtysix(n : ℕ) : Prop := Nat.gcd n 36 = 1

def transform_last_to_first(n : ℕ) : ℕ := 
  let last := n % 10
  let rest := n / 10
  last * 10^7 + rest

theorem find_extrema_A :
  ∃ (A_max A_min : ℕ), 
    (∃ B_max B_min : ℕ, 
      eight_digit_number B_max ∧ 
      eight_digit_number B_min ∧ 
      coprime_with_thirtysix B_max ∧ 
      coprime_with_thirtysix B_min ∧ 
      B_max > 77777777 ∧ 
      B_min > 77777777 ∧ 
      transform_last_to_first B_max = A_max ∧ 
      transform_last_to_first B_min = A_min) ∧ 
    A_max = 99999998 ∧ 
    A_min = 17777779 := 
  sorry

end find_extrema_A_l57_57255


namespace area_of_region_l57_57098

theorem area_of_region : 
  (∃ A : ℝ, 
    (∀ x y : ℝ, 
      (|4 * x - 20| + |3 * y + 9| ≤ 4) → 
      A = (32 / 3))) :=
by 
  sorry

end area_of_region_l57_57098


namespace expected_shots_ivan_l57_57156

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l57_57156


namespace pascal_15_5th_number_l57_57524

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57524


namespace johnny_fishes_l57_57195

theorem johnny_fishes
  (total_fishes : ℕ)
  (sony_ratio : ℕ)
  (total_is_40 : total_fishes = 40)
  (sony_is_4x_johnny : sony_ratio = 4)
  : ∃ (johnny_fishes : ℕ), johnny_fishes + sony_ratio * johnny_fishes = total_fishes ∧ johnny_fishes = 8 :=
by
  sorry

end johnny_fishes_l57_57195


namespace altitude_from_A_to_BC_l57_57968

theorem altitude_from_A_to_BC (x y : ℝ) : 
  (3 * x + 4 * y + 12 = 0) ∧ 
  (4 * x - 3 * y + 16 = 0) ∧ 
  (2 * x + y - 2 = 0) → 
  (∃ (m b : ℝ), (y = m * x + b) ∧ (m = 1 / 2) ∧ (b = 2 - 8)) :=
by 
  sorry

end altitude_from_A_to_BC_l57_57968


namespace number_of_participants_l57_57477

-- Define the conditions and theorem
theorem number_of_participants (n : ℕ) (h : n * (n - 1) / 2 = 231) : n = 22 :=
  sorry

end number_of_participants_l57_57477


namespace pascal_triangle_row_fifth_number_l57_57541

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57541


namespace total_climbing_time_l57_57622

theorem total_climbing_time :
  let a := 30
  let d := 10
  let n := 8
  let S := (n / 2) * (2 * a + (n - 1) * d)
  S = 520 :=
by
  let a := 30
  let d := 10
  let n := 8
  let S := (n / 2) * (2 * a + (n - 1) * d)
  sorry

end total_climbing_time_l57_57622


namespace mcq_options_l57_57079

theorem mcq_options :
  ∃ n : ℕ, (1/n : ℝ) * (1/2) * (1/2) = (1/12) ∧ n = 3 :=
by
  sorry

end mcq_options_l57_57079


namespace sum_of_a_is_9_l57_57110

theorem sum_of_a_is_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 5 / 7 = (a2 / 2.factorial) + (a3 / 3.factorial) + (a4 / 4.factorial) + (a5 / 5.factorial) + (a6 / 6.factorial) + (a7 / 7.factorial))
  (h2 : 0 ≤ a2 ∧ a2 < 2)
  (h3 : 0 ≤ a3 ∧ a3 < 3)
  (h4 : 0 ≤ a4 ∧ a4 < 4)
  (h5 : 0 ≤ a5 ∧ a5 < 5)
  (h6 : 0 ≤ a6 ∧ a6 < 6)
  (h7 : 0 ≤ a7 ∧ a7 < 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 :=
sorry

end sum_of_a_is_9_l57_57110


namespace tricia_age_l57_57366

theorem tricia_age :
  ∀ (T A Y E K R V : ℕ),
    T = 1 / 3 * A →
    A = 1 / 4 * Y →
    Y = 2 * E →
    K = 1 / 3 * E →
    R = K + 10 →
    R = V - 2 →
    V = 22 →
    T = 5 :=
by sorry

end tricia_age_l57_57366


namespace proof_problem_l57_57671

axiom sqrt (x : ℝ) : ℝ
axiom cbrt (x : ℝ) : ℝ
noncomputable def sqrtValue : ℝ :=
  sqrt 81

theorem proof_problem (m n : ℝ) (hm : sqrt m = 3) (hn : cbrt n = -4) : sqrt (2 * m - n - 1) = 9 ∨ sqrt (2 * m - n - 1) = -9 :=
by
  sorry

end proof_problem_l57_57671


namespace number_of_minibuses_l57_57198

theorem number_of_minibuses (total_students : ℕ) (capacity : ℕ) (h : total_students = 48) (h_capacity : capacity = 8) : 
  ∃ minibuses, minibuses = (total_students + capacity - 1) / capacity ∧ minibuses = 7 :=
by
  have h1 : (48 + 8 - 1) = 55 := by simp [h, h_capacity]
  have h2 : 55 / 8 = 6 := by simp [h, h_capacity]
  use 7
  sorry

end number_of_minibuses_l57_57198


namespace penultimate_digit_even_l57_57866

theorem penultimate_digit_even (n : ℕ) (h : n > 2) : ∃ k : ℕ, ∃ d : ℕ, d % 2 = 0 ∧ 10 * d + k = (3 ^ n) % 100 :=
sorry

end penultimate_digit_even_l57_57866


namespace pascal_15_5th_number_l57_57527

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57527


namespace triangular_25_eq_325_l57_57225

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end triangular_25_eq_325_l57_57225


namespace original_wire_length_l57_57744

theorem original_wire_length (side_len total_area : ℕ) (h1 : side_len = 2) (h2 : total_area = 92) :
  (total_area / (side_len * side_len)) * (4 * side_len) = 184 := 
by
  sorry

end original_wire_length_l57_57744


namespace decagon_diagonal_relation_l57_57142

-- Define side length, shortest diagonal, and longest diagonal in a regular decagon
variable (a b d : ℝ)
variable (h1 : a > 0) -- Side length must be positive
variable (h2 : b > 0) -- Shortest diagonal length must be positive
variable (h3 : d > 0) -- Longest diagonal length must be positive

theorem decagon_diagonal_relation (ha : d^2 = 5 * a^2) (hb : b^2 = 3 * a^2) : b^2 = a * d :=
sorry

end decagon_diagonal_relation_l57_57142


namespace pascal_triangle_fifth_number_l57_57609

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57609


namespace set_B_roster_method_l57_57812

def A : Set ℤ := {-2, 2, 3, 4}
def B : Set ℤ := {x | ∃ t ∈ A, x = t^2}

theorem set_B_roster_method : B = {4, 9, 16} :=
by
  sorry

end set_B_roster_method_l57_57812


namespace total_tweets_is_correct_l57_57861

-- Define the conditions of Polly's tweeting behavior and durations
def happy_tweets := 18
def hungry_tweets := 4
def mirror_tweets := 45
def duration := 20

-- Define the total tweets calculation
def total_tweets := duration * happy_tweets + duration * hungry_tweets + duration * mirror_tweets

-- Prove that the total number of tweets is 1340
theorem total_tweets_is_correct : total_tweets = 1340 := by
  sorry

end total_tweets_is_correct_l57_57861


namespace no_positive_rational_solutions_l57_57008

theorem no_positive_rational_solutions (n : ℕ) (h_pos_n : 0 < n) : 
  ¬ ∃ (x y : ℚ) (h_x_pos : 0 < x) (h_y_pos : 0 < y), x + y + (1/x) + (1/y) = 3 * n :=
by
  sorry

end no_positive_rational_solutions_l57_57008


namespace foci_distance_l57_57679

open Real

-- Defining parameters and conditions
variables (a : ℝ) (b : ℝ) (c : ℝ)
  (F1 F2 A B : ℝ × ℝ) -- Foci and points A, B
  (hyp_cavity : c ^ 2 = a ^ 2 + b ^ 2)
  (perimeters_eq : dist A B = 3 * a ∧ dist A F1 + dist B F1 = dist B F1 + dist B F2 + dist F1 F2)
  (distance_property : dist A F2 - dist A F1 = 2 * a)
  (c_value : c = 2 * a) -- Derived from hyperbolic definition
  
-- Main theorem to prove the distance between foci
theorem foci_distance : dist F1 F2 = 4 * a :=
  sorry

end foci_distance_l57_57679


namespace ratio_equation_solution_l57_57301

variable (x y z : ℝ)
variables (hx : x > 0) (hy : y > 0) (hz : z > 0)
variables (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)

theorem ratio_equation_solution
  (h : y / (2 * x - z) = (x + y) / (2 * z) ∧ (x + y) / (2 * z) = x / y) :
  x / y = 3 :=
sorry

end ratio_equation_solution_l57_57301


namespace triangular_25_l57_57223

-- Defining the formula for the n-th triangular number.
def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Stating that the 25th triangular number is 325.
theorem triangular_25 : triangular 25 = 325 :=
  by
    -- We don't prove it here, so we simply state it requires a proof.
    sorry

end triangular_25_l57_57223


namespace fraction_cookies_blue_or_green_l57_57072

theorem fraction_cookies_blue_or_green (C : ℕ) (h1 : 1/C = 1/4) (h2 : 0.5555555555555556 = 5/9) :
  (1/4 + (5/9) * (3/4)) = (2/3) :=
by sorry

end fraction_cookies_blue_or_green_l57_57072


namespace pascal_fifteen_four_l57_57570

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57570


namespace triangle_BX_in_terms_of_sides_l57_57412

-- Define the triangle with angles and points
variables {A B C : ℝ}
variables {AB AC BC : ℝ}
variables (X Y : ℝ) (AZ : ℝ)

-- Add conditions as assumptions
variables (angle_A_bisector : 2 * A = (B + C)) -- AZ is the angle bisector of angle A
variables (angle_B_lt_C : B < C) -- angle B < angle C
variables (point_XY : X / AB = Y / AC ∧ X = Y) -- BX = CY and angles BZX = CZY

-- Define the statement to be proved
theorem triangle_BX_in_terms_of_sides :
    BX = CY →
    (AZ < 1 ∧ AZ > 0) →
    A + B + C = π → 
    BX = (BC * BC) / (AB + AC) :=
sorry

end triangle_BX_in_terms_of_sides_l57_57412


namespace fifth_number_in_pascals_triangle_l57_57588

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57588


namespace total_distance_traveled_l57_57125

/-- Defining the distance Greg travels in each leg of his trip -/
def distance_workplace_to_market : ℕ := 30

def distance_market_to_friend : ℕ := distance_workplace_to_market + 10

def distance_friend_to_aunt : ℕ := 5

def distance_aunt_to_grocery : ℕ := 7

def distance_grocery_to_home : ℕ := 18

/-- The total distance Greg traveled during his entire trip is the sum of all individual distances -/
theorem total_distance_traveled :
  distance_workplace_to_market + distance_market_to_friend + distance_friend_to_aunt + distance_aunt_to_grocery + distance_grocery_to_home = 100 :=
by
  sorry

end total_distance_traveled_l57_57125


namespace jessica_guess_l57_57620

-- Step a: Define the conditions
def bags : ℕ := 3
def red_jellybeans_bag : ℕ := 24
def white_jellybeans_bag : ℕ := 18

-- Step c: Define the mathematical equivalent problem
theorem jessica_guess :
  let total_jellybeans_bag := red_jellybeans_bag + white_jellybeans_bag in
  let total_jellybeans_fishbowl := total_jellybeans_bag * bags in
  total_jellybeans_fishbowl = 126 :=
by
  sorry

end jessica_guess_l57_57620


namespace pascal_triangle_row_fifth_number_l57_57543

theorem pascal_triangle_row_fifth_number : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_row_fifth_number_l57_57543


namespace linear_function_passing_quadrants_l57_57117

theorem linear_function_passing_quadrants (b : ℝ) :
  (∀ x : ℝ, (y = x + b) ∧ (y > 0 ↔ (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0))) →
  b > 0 :=
sorry

end linear_function_passing_quadrants_l57_57117


namespace proof_math_problem_lean_l57_57464

def upper_bound := 201
def multiple_of_6 (n : ℕ) := (n % 6) = 0
def multiple_of_8 (n : ℕ) := (n % 8) = 0
def lcm_6_8 := 24

def count_multiples_less_than (multiple : ℕ) (bound : ℕ) : ℕ :=
  (bound - 1) / multiple

def math_problem_lean : Prop :=
  let count6 := count_multiples_less_than 6 upper_bound in
  let count8 := count_multiples_less_than 8 upper_bound in
  let count24 := count_multiples_less_than lcm_6_8 upper_bound in
  let result := count6 + count8 - 2 * count24 in
  result = 42

theorem proof_math_problem_lean : math_problem_lean := 
  sorry

end proof_math_problem_lean_l57_57464


namespace gcd_factorials_l57_57765


noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorials :
  gcd (factorial 8) (factorial 6 * factorial 6) = 5760 := by
  sorry

end gcd_factorials_l57_57765


namespace cost_prices_sum_l57_57249

theorem cost_prices_sum
  (W B : ℝ)
  (h1 : 0.9 * W + 196 = 1.04 * W)
  (h2 : 1.08 * B - 150 = 1.02 * B) :
  W + B = 3900 := 
sorry

end cost_prices_sum_l57_57249


namespace total_players_on_team_l57_57887

theorem total_players_on_team (M W : ℕ) (h1 : W = M + 2) (h2 : (M : ℝ) / W = 0.7777777777777778) : M + W = 16 :=
by 
  sorry

end total_players_on_team_l57_57887


namespace parcels_division_l57_57953

theorem parcels_division (x y n : ℕ) (h : 5 + 2 * x + 3 * y = 4 * n) (hn : n = x + y) :
    n = 3 ∨ n = 4 ∨ n = 5 := 
sorry

end parcels_division_l57_57953


namespace min_sum_xy_l57_57971

theorem min_sum_xy (x y : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_ne_hy : x ≠ y)
  (h_eq : (1 / x : ℝ) + (1 / y) = 1 / 12) : x + y = 49 :=
sorry

end min_sum_xy_l57_57971


namespace pascal_triangle_fifth_number_l57_57605

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57605


namespace leif_apples_oranges_l57_57168

theorem leif_apples_oranges : 
  let apples := 14
  let dozens_of_oranges := 2 
  let oranges := dozens_of_oranges * 12
  in oranges - apples = 10 :=
by 
  let apples := 14
  let dozens_of_oranges := 2
  let oranges := dozens_of_oranges * 12
  show oranges - apples = 10
  sorry

end leif_apples_oranges_l57_57168


namespace pascal_fifteen_four_l57_57569

theorem pascal_fifteen_four : nat.choose 15 4 = 1365 := by
  -- Sorry, no proof required per instructions
  sorry

end pascal_fifteen_four_l57_57569


namespace multiples_6_8_not_both_l57_57472

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l57_57472


namespace rain_probability_tel_aviv_l57_57350

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l57_57350


namespace value_of_expression_eq_34_l57_57226

theorem value_of_expression_eq_34 : (2 - 6 + 10 - 14 + 18 - 22 + 26 - 30 + 34 - 38 + 42 - 46 + 50 - 54 + 58 - 62 + 66 - 70 + 70) = 34 :=
by
  sorry

end value_of_expression_eq_34_l57_57226


namespace factorize_a3_minus_4ab2_l57_57431

theorem factorize_a3_minus_4ab2 (a b : ℝ) : a^3 - 4 * a * b^2 = a * (a + 2 * b) * (a - 2 * b) :=
by
  -- Proof is omitted; write 'sorry' as a placeholder
  sorry

end factorize_a3_minus_4ab2_l57_57431


namespace min_value_expression_l57_57835

noncomputable 
def min_value_condition (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : ℝ :=
  (a + 1) * (b + 1) * (c + 1)

theorem min_value_expression (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) : 
  min_value_condition a b c h_pos h_abc = 8 :=
sorry

end min_value_expression_l57_57835


namespace sum_of_sides_eq_l57_57842

open Real

theorem sum_of_sides_eq (a h : ℝ) (α : ℝ) (ha : a > 0) (hh : h > 0) (hα : 0 < α ∧ α < π) :
  ∃ b c : ℝ, b + c = sqrt (a^2 + 2 * a * h * (cos (α / 2) / sin (α / 2))) :=
by
  sorry

end sum_of_sides_eq_l57_57842


namespace monotonic_interval_l57_57933

noncomputable def f (x : ℝ) : ℝ := 7 * Real.sin(x - Real.pi / 6)

def is_monotonically_increasing_interval (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y : ℝ, a < x ∧ x < y ∧ y < b → f x < f y

theorem monotonic_interval : is_monotonically_increasing_interval f 0 (Real.pi / 2) :=
sorry

end monotonic_interval_l57_57933


namespace min_value_f_l57_57106

noncomputable def f (x : ℝ) : ℝ := x + (1 / x) + (1 / (x^2 + (1 / x)))

theorem min_value_f : ∃ x > 0, ∀ y > 0, f y ≥ f x ∧ f x = 5 / 2 :=
by
  sorry

end min_value_f_l57_57106


namespace total_diagonals_in_rectangular_prism_l57_57746

-- We define the rectangular prism with its properties
structure RectangularPrism :=
  (vertices : ℕ)
  (edges : ℕ)
  (distinct_dimensions : ℕ)

-- We specify the conditions for the rectangular prism
def givenPrism : RectangularPrism :=
{
  vertices := 8,
  edges := 12,
  distinct_dimensions := 3
}

-- We assert the total number of diagonals in the rectangular prism
theorem total_diagonals_in_rectangular_prism (P : RectangularPrism) : P = givenPrism → ∃ diag, diag = 16 :=
by
  intro h
  have diag := 16
  use diag
  sorry

end total_diagonals_in_rectangular_prism_l57_57746


namespace monotonicity_of_g_l57_57449

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.logb a (|x + 1|)
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := Real.logb a (- (3 / 2) * x^2 + a * x)

theorem monotonicity_of_g (a : ℝ) (h : 0 < a ∧ a ≠ 1) (h0 : ∀ x : ℝ, 0 < x ∧ x < 1 → f x a < 0) :
  ∀ x : ℝ, 0 < x ∧ x ≤ a / 3 → (g x a) < (g (x + ε) a) := 
sorry


end monotonicity_of_g_l57_57449


namespace expected_shots_ivan_l57_57157

noncomputable def expected_shot_count : ℝ :=
  let n := 14
  let p_hit := 0.1
  let q_miss := 1 - p_hit
  let arrows_per_hit := 3
  let expected_shots_per_arrow := (q_miss + p_hit * (1 + 3 * expected_shots_per_arrow))
  n * expected_shots_per_arrow

theorem expected_shots_ivan : expected_shot_count = 20 :=
  sorry

end expected_shots_ivan_l57_57157


namespace minimum_routes_l57_57999

theorem minimum_routes (a b c : ℕ) (h1 : a + b ≥ 14) (h2 : b + c ≥ 14) (h3 : c + a ≥ 14) :
  a + b + c ≥ 21 :=
by sorry

end minimum_routes_l57_57999


namespace fifth_number_in_pascal_row_l57_57615

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57615


namespace tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l57_57834

/-- Given the trigonometric identity and the ratio, we want to prove the relationship between the tangents of the angles. -/
theorem tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n
  (α β m n : ℝ)
  (h : (Real.sin (α + β)) / (Real.sin (α - β)) = m / n) :
  (Real.tan β) / (Real.tan α) = (m - n) / (m + n) :=
  sorry

end tan_beta_tan_alpha_eq_m_minus_n_over_m_plus_n_l57_57834


namespace fifth_number_in_pascals_triangle_l57_57590

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57590


namespace apple_crisps_calculation_l57_57191

theorem apple_crisps_calculation (apples crisps : ℕ) (h : crisps = 3 ∧ apples = 12) : 
  (36 / apples) * crisps = 9 := by
  sorry

end apple_crisps_calculation_l57_57191


namespace find_x_plus_2y_sq_l57_57376

theorem find_x_plus_2y_sq (x y : ℝ) 
  (h : 8 * y^4 + 4 * x^2 * y^2 + 4 * x * y^2 + 2 * x^3 + 2 * y^2 + 2 * x = x^2 + 1) : 
  x + 2 * y^2 = 1 / 2 :=
sorry

end find_x_plus_2y_sq_l57_57376


namespace greta_hours_worked_l57_57126

-- Define the problem conditions
def greta_hourly_rate := 12
def lisa_hourly_rate := 15
def lisa_hours_to_equal_greta_earnings := 32
def greta_earnings (hours_worked : ℕ) := greta_hourly_rate * hours_worked
def lisa_earnings := lisa_hourly_rate * lisa_hours_to_equal_greta_earnings

-- Problem statement
theorem greta_hours_worked (G : ℕ) (H : greta_earnings G = lisa_earnings) : G = 40 := by
  sorry

end greta_hours_worked_l57_57126


namespace evaluate_expression_l57_57429

theorem evaluate_expression :
  let c := (-2 : ℚ)
  let x := (2 : ℚ) / 5
  let y := (3 : ℚ) / 5
  let z := (-3 : ℚ)
  c * x^3 * y^4 * z^2 = (-11664) / 78125 := by
  sorry

end evaluate_expression_l57_57429


namespace pascal_row_fifth_number_l57_57537

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57537


namespace positive_y_percentage_l57_57923

theorem positive_y_percentage (y : ℝ) (hy_pos : 0 < y) (h : 0.01 * y * y = 9) : y = 30 := by
  sorry

end positive_y_percentage_l57_57923


namespace peanut_count_l57_57995

-- Definitions
def initial_peanuts : Nat := 10
def added_peanuts : Nat := 8

-- Theorem to prove
theorem peanut_count : (initial_peanuts + added_peanuts) = 18 := 
by
  -- Proof placeholder
  sorry

end peanut_count_l57_57995


namespace integer_solutions_count_l57_57272

theorem integer_solutions_count (x : ℤ) :
  (75 ^ 60 * x ^ 60 > x ^ 120 ∧ x ^ 120 > 3 ^ 240) → ∃ n : ℕ, n = 65 :=
by
  sorry

end integer_solutions_count_l57_57272


namespace monotonic_increasing_l57_57932

noncomputable def f (x : ℝ) : ℝ := 7 * sin (x - π / 6)

theorem monotonic_increasing : ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 < π / 2) → f x1 < f x2 := sorry

end monotonic_increasing_l57_57932


namespace solve_for_m_l57_57984

-- Define the conditions for the lines being parallel
def condition_one (m : ℝ) : Prop :=
  ∃ x y : ℝ, x + m * y + 3 = 0

def condition_two (m : ℝ) : Prop :=
  ∃ x y : ℝ, (m - 1) * x + 2 * m * y + 2 * m = 0

def are_parallel (A B C D : ℝ) : Prop :=
  A * D = B * C

theorem solve_for_m :
  ∀ (m : ℝ),
    (condition_one m) → 
    (condition_two m) → 
    (are_parallel 1 m 3 (2 * m)) →
    (m = 0) :=
by
  intro m h1 h2 h_parallel
  sorry

end solve_for_m_l57_57984


namespace fifth_number_in_pascal_row_l57_57619

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57619


namespace find_minimum_value_l57_57955

theorem find_minimum_value (c : ℝ) : 
  (∀ c : ℝ, (c = -12) ↔ (∀ d : ℝ, (1 / 3) * d^2 + 8 * d - 7 ≥ (1 / 3) * (-12)^2 + 8 * (-12) - 7)) :=
sorry

end find_minimum_value_l57_57955


namespace abs_eq_five_l57_57836

theorem abs_eq_five (x : ℝ) : |x| = 5 → (x = 5 ∨ x = -5) :=
by
  intro h
  sorry

end abs_eq_five_l57_57836


namespace find_n_in_permutation_combination_equation_l57_57786

-- Lean statement for the proof problem

theorem find_n_in_permutation_combination_equation :
  ∃ (n : ℕ), (n > 0) ∧ (Nat.factorial 8 / Nat.factorial (8 - n) = 2 * (Nat.factorial 8 / (Nat.factorial 2 * Nat.factorial 6)))
  := sorry

end find_n_in_permutation_combination_equation_l57_57786


namespace cricket_team_rh_players_l57_57662

theorem cricket_team_rh_players (total_players throwers non_throwers lh_non_throwers rh_non_throwers rh_players : ℕ)
    (h1 : total_players = 58)
    (h2 : throwers = 37)
    (h3 : non_throwers = total_players - throwers)
    (h4 : lh_non_throwers = non_throwers / 3)
    (h5 : rh_non_throwers = non_throwers - lh_non_throwers)
    (h6 : rh_players = throwers + rh_non_throwers) :
  rh_players = 51 := by
  sorry

end cricket_team_rh_players_l57_57662


namespace intersection_of_A_and_B_l57_57440

noncomputable def A := {x : ℝ | Real.log x ≤ 0}
noncomputable def B := {x : ℝ | abs (x^2 - 1) ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = A :=
sorry

end intersection_of_A_and_B_l57_57440


namespace ferry_routes_ratio_l57_57959

theorem ferry_routes_ratio :
  ∀ (D_P D_Q : ℝ) (speed_P time_P speed_Q time_Q : ℝ),
  speed_P = 8 →
  time_P = 3 →
  speed_Q = speed_P + 4 →
  time_Q = time_P + 1 →
  D_P = speed_P * time_P →
  D_Q = speed_Q * time_Q →
  D_Q / D_P = 2 :=
by sorry

end ferry_routes_ratio_l57_57959


namespace carl_garden_area_l57_57753

theorem carl_garden_area 
  (total_posts : Nat)
  (length_post_distance : Nat)
  (corner_posts : Nat)
  (longer_side_multiplier : Nat)
  (posts_per_shorter_side : Nat)
  (posts_per_longer_side : Nat)
  (shorter_side_distance : Nat)
  (longer_side_distance : Nat) :
  total_posts = 24 →
  length_post_distance = 5 →
  corner_posts = 4 →
  longer_side_multiplier = 2 →
  posts_per_shorter_side = (24 + 4) / 6 →
  posts_per_longer_side = (24 + 4) / 6 * 2 →
  shorter_side_distance = (posts_per_shorter_side - 1) * length_post_distance →
  longer_side_distance = (posts_per_longer_side - 1) * length_post_distance →
  shorter_side_distance * longer_side_distance = 900 :=
by
  intros
  sorry

end carl_garden_area_l57_57753


namespace pascal_triangle_fifth_number_l57_57517

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57517


namespace math_problem_l57_57956

theorem math_problem (x : ℝ) :
  (x^3 - 8*x^2 + 16*x > 64) ∧ (x^2 - 4*x + 5 > 0) → x > 4 :=
by
  sorry

end math_problem_l57_57956


namespace largest_divisor_of_n_pow4_minus_n_for_composites_l57_57794

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end largest_divisor_of_n_pow4_minus_n_for_composites_l57_57794


namespace gcd_equation_solutions_l57_57097

theorem gcd_equation_solutions:
  ∀ (x y : ℕ), x > 0 ∧ y > 0 ∧ x + y^2 + Nat.gcd x y ^ 3 = x * y * Nat.gcd x y 
  → (x = 4 ∧ y = 2) ∨ (x = 4 ∧ y = 6) ∨ (x = 5 ∧ y = 2) ∨ (x = 5 ∧ y = 3) := 
by
  intros x y h
  sorry

end gcd_equation_solutions_l57_57097


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l57_57642

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l57_57642


namespace pascal_triangle_15_4_l57_57506

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57506


namespace gcd_of_sum_of_four_consecutive_integers_is_two_l57_57643

/-- Let B be the set of all numbers which can be represented as the sum of four consecutive positive integers.
    The greatest common divisor of all numbers in B is 2. -/
theorem gcd_of_sum_of_four_consecutive_integers_is_two (B : Set ℕ) 
  (hB : ∀ x : ℕ, x > 0 → (4 * x + 6) ∈ B) : gcd (B : Finset ℕ) = 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_is_two_l57_57643


namespace pascal_fifth_number_in_row_15_l57_57551

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57551


namespace TrigPowerEqualsOne_l57_57263

theorem TrigPowerEqualsOne : ((Real.cos (160 * Real.pi / 180) + Real.sin (160 * Real.pi / 180) * Complex.I)^36 = 1) :=
by
  sorry

end TrigPowerEqualsOne_l57_57263


namespace gcd_of_sum_of_four_consecutive_integers_l57_57639

theorem gcd_of_sum_of_four_consecutive_integers :
  let B := {n | ∃ x : ℕ, n = (x-1) + x + (x+1) + (x+2) ∧ 0 < x}
  gcd B 2 :=
by
  sorry

end gcd_of_sum_of_four_consecutive_integers_l57_57639


namespace min_max_expression_l57_57954

theorem min_max_expression (x : ℝ) (h : 2 ≤ x ∧ x ≤ 7) :
  ∃ (a : ℝ) (b : ℝ), a = 11 / 3 ∧ b = 87 / 16 ∧ 
  (∀ y, 2 ≤ y ∧ y ≤ 7 → 11 / 3 ≤ (y^2 + 4*y + 10) / (2*y + 2)) ∧
  (∀ y, 2 ≤ y ∧ y ≤ 7 → (y^2 + 4*y + 10) / (2*y + 2) ≤ 87 / 16) :=
sorry

end min_max_expression_l57_57954


namespace problem1_l57_57391

theorem problem1 (x : ℝ) (hx : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 := 
sorry

end problem1_l57_57391


namespace find_tricias_age_l57_57372

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l57_57372


namespace pascal_fifth_number_l57_57490

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57490


namespace complement_M_eq_45_l57_57654

open Set Nat

/-- Define the universal set U and the set M in Lean -/
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

def M : Set ℕ := {x | 6 % x = 0 ∧ x ∈ U}

/-- Lean theorem statement for the complement of M in U -/
theorem complement_M_eq_45 : (U \ M) = {4, 5} :=
by
  sorry

end complement_M_eq_45_l57_57654


namespace simplify_expression_l57_57668

theorem simplify_expression (x y : ℤ) (h₁ : x = 2) (h₂ : y = -3) :
  ((2 * x - y) ^ 2 - (x - y) * (x + y) - 2 * y ^ 2) / x = 18 :=
by
  sorry

end simplify_expression_l57_57668


namespace first_machine_defect_probability_l57_57997

/-- Probability that a randomly selected defective item was made by the first machine is 0.5 
given certain conditions. -/
theorem first_machine_defect_probability :
  let PFirstMachine := 0.4
  let PSecondMachine := 0.6
  let DefectRateFirstMachine := 0.03
  let DefectRateSecondMachine := 0.02
  let TotalDefectProbability := PFirstMachine * DefectRateFirstMachine + PSecondMachine * DefectRateSecondMachine
  let PDefectGivenFirstMachine := PFirstMachine * DefectRateFirstMachine / TotalDefectProbability
  PDefectGivenFirstMachine = 0.5 :=
by
  sorry

end first_machine_defect_probability_l57_57997


namespace convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l57_57419

theorem convert_deg_to_rad1 : 780 * (Real.pi / 180) = (13 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad2 : -1560 * (Real.pi / 180) = -(26 * Real.pi) / 3 := sorry
theorem convert_deg_to_rad3 : 67.5 * (Real.pi / 180) = (3 * Real.pi) / 8 := sorry
theorem convert_rad_to_deg1 : -(10 * Real.pi / 3) * (180 / Real.pi) = -600 := sorry
theorem convert_rad_to_deg2 : (Real.pi / 12) * (180 / Real.pi) = 15 := sorry
theorem convert_rad_to_deg3 : (7 * Real.pi / 4) * (180 / Real.pi) = 315 := sorry

end convert_deg_to_rad1_convert_deg_to_rad2_convert_deg_to_rad3_convert_rad_to_deg1_convert_rad_to_deg2_convert_rad_to_deg3_l57_57419


namespace proof_problem_l57_57435

theorem proof_problem (x : ℝ) : (0 < x ∧ x < 5) → (x^2 - 5 * x < 0) ∧ (|x - 2| < 3) :=
by
  sorry

end proof_problem_l57_57435


namespace monotonic_intervals_range_of_c_l57_57826

noncomputable def f (x : ℝ) (b c : ℝ) : ℝ := c * Real.log x + (1 / 2) * x ^ 2 + b * x

lemma extreme_point_condition {b c : ℝ} (h1 : c ≠ 0) (h2 : f 1 b c = 0) : b + c + 1 = 0 :=
sorry

theorem monotonic_intervals (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : c > 1) :
  (∀ x, 0 < x ∧ x < 1 → f 1 b c < f x b c) ∧ 
  (∀ x, 1 < x ∧ x < c → f 1 b c > f x b c) ∧ 
  (∀ x, x > c → f 1 b c < f x b c) :=
sorry

theorem range_of_c (b c : ℝ) (h1 : c ≠ 0) (h2 : f 1 b c = 0) (h3 : (f 1 b c < 0)) :
  -1 / 2 < c ∧ c < 0 :=
sorry

end monotonic_intervals_range_of_c_l57_57826


namespace cos_double_angle_l57_57285

theorem cos_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 5) = Real.sqrt 3 / 3) :
  Real.cos (2 * α + 2 * Real.pi / 5) = 1 / 3 :=
by
  sorry

end cos_double_angle_l57_57285


namespace total_history_and_maths_l57_57481

-- Defining the conditions
def total_students : ℕ := 25
def fraction_like_maths : ℚ := 2 / 5
def fraction_like_science : ℚ := 1 / 3

-- Theorem statement
theorem total_history_and_maths : (total_students * fraction_like_maths + (total_students * (1 - fraction_like_maths) * (1 - fraction_like_science))) = 20 := by
  sorry

end total_history_and_maths_l57_57481


namespace max_brownies_l57_57124

theorem max_brownies (m n : ℕ) (h1 : (m-2)*(n-2) = 2*(2*m + 2*n - 4)) : m * n ≤ 294 :=
by sorry

end max_brownies_l57_57124


namespace find_c_if_quadratic_lt_zero_l57_57900

theorem find_c_if_quadratic_lt_zero (c : ℝ) : 
  (∀ x : ℝ, -x^2 + c * x - 12 < 0 ↔ (x < 2 ∨ x > 7)) → c = 9 := 
by
  sorry

end find_c_if_quadratic_lt_zero_l57_57900


namespace solution_set_l57_57005

-- Define the conditions
variables {f : ℝ → ℝ}

-- Condition: f(x) is an odd function
axiom odd_function : ∀ x : ℝ, f (-x) = -f x

-- Condition: xf'(x) + f(x) < 0 for x in (-∞, 0)
axiom condition1 : ∀ x : ℝ, x < 0 → x * (deriv f x) + f x < 0

-- Condition: f(-2) = 0
axiom f_neg2_zero : f (-2) = 0

-- Goal: Prove the solution set of the inequality xf(x) < 0 is {x | -2 < x < 0 ∨ 0 < x < 2}
theorem solution_set : ∀ x : ℝ, (x * f x < 0) ↔ (-2 < x ∧ x < 0 ∨ 0 < x ∧ x < 2) := by
  sorry

end solution_set_l57_57005


namespace pascal_row_fifth_number_l57_57533

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57533


namespace pascal_triangle_fifth_number_l57_57580

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57580


namespace can_split_3x3x3_into_9_corners_l57_57912

-- Define the conditions
def number_of_cubes_in_3x3x3 : ℕ := 27
def number_of_units_in_corner : ℕ := 3
def number_of_corners : ℕ := 9

-- Prove the proposition
theorem can_split_3x3x3_into_9_corners :
  (number_of_corners * number_of_units_in_corner = number_of_cubes_in_3x3x3) :=
by
  sorry

end can_split_3x3x3_into_9_corners_l57_57912


namespace geom_prog_min_third_term_l57_57082

theorem geom_prog_min_third_term :
  ∃ (d : ℝ), (-4 + 10 * Real.sqrt 6 = d ∨ -4 - 10 * Real.sqrt 6 = d) ∧
  (∀ x, x = 37 + 2 * d → x ≤ 29 - 20 * Real.sqrt 6) := 
sorry

end geom_prog_min_third_term_l57_57082


namespace triangular_25_eq_325_l57_57224

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_25_eq_325 : triangular_number 25 = 325 :=
by
  -- proof would go here
  sorry

end triangular_25_eq_325_l57_57224


namespace sandy_age_l57_57233

theorem sandy_age (S M : ℕ) (h1 : M = S + 18) (h2 : S * 9 = M * 7) : S = 63 := by
  sorry

end sandy_age_l57_57233


namespace sqrt_7_estimate_l57_57428

theorem sqrt_7_estimate (h1 : 4 < 7) (h2 : 7 < 9) (h3 : Nat.sqrt 4 = 2) (h4 : Nat.sqrt 9 = 3) : 2 < Real.sqrt 7 ∧ Real.sqrt 7 < 3 :=
  by {
    -- the proof would go here, but use 'sorry' to omit it
    sorry
  }

end sqrt_7_estimate_l57_57428


namespace fifth_number_in_pascals_triangle_l57_57584

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57584


namespace fourth_guard_run_distance_l57_57217

-- Define the rectangle's dimensions
def length : ℝ := 300
def width : ℝ := 200

-- Define the perimeter of the rectangle
def perimeter : ℝ := 2 * (length + width)

-- Given the sum of the distances run by three guards
def sum_of_three_guards : ℝ := 850

-- The fourth guard's distance is what we need to prove
def fourth_guard_distance := perimeter - sum_of_three_guards

-- The proof goal: we need to show that the fourth guard's distance is 150 meters
theorem fourth_guard_run_distance : fourth_guard_distance = 150 := by
  sorry  -- This placeholder means that the proof is omitted

end fourth_guard_run_distance_l57_57217


namespace bond_interest_percentage_l57_57322

noncomputable def interest_percentage_of_selling_price (face_value interest_rate : ℝ) (selling_price : ℝ) : ℝ :=
  (face_value * interest_rate) / selling_price * 100

theorem bond_interest_percentage :
  let face_value : ℝ := 5000
  let interest_rate : ℝ := 0.07
  let selling_price : ℝ := 5384.615384615386
  interest_percentage_of_selling_price face_value interest_rate selling_price = 6.5 :=
by
  sorry

end bond_interest_percentage_l57_57322


namespace range_f_3_l57_57827

section

variables (a c : ℝ) (f : ℝ → ℝ)
def quadratic_function := ∀ x, f x = a * x^2 - c

-- Define the constraints given in the problem
axiom h1 : -4 ≤ f 1 ∧ f 1 ≤ -1
axiom h2 : -1 ≤ f 2 ∧ f 2 ≤ 5

-- Prove that the correct range for f(3) is -1 ≤ f(3) ≤ 20
theorem range_f_3 (a c : ℝ) (f : ℝ → ℝ) (h1 : -4 ≤ f 1 ∧ f 1 ≤ -1) (h2 : -1 ≤ f 2 ∧ f 2 ≤ 5):
  -1 ≤ f 3 ∧ f 3 ≤ 20 :=
sorry

end

end range_f_3_l57_57827


namespace greatest_integer_100y_l57_57651

noncomputable def y : ℝ := (∑ n in finset.range 1 31, real.cos (n * real.pi / 180)) / (∑ n in finset.range 1 31, real.sin (n * real.pi / 180))

theorem greatest_integer_100y : ⌊100 * y⌋ = 373 := sorry

end greatest_integer_100y_l57_57651


namespace man_speed_is_correct_l57_57085

noncomputable def train_length : ℝ := 275
noncomputable def train_speed_kmh : ℝ := 60
noncomputable def time_seconds : ℝ := 14.998800095992323

noncomputable def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
noncomputable def relative_speed_ms : ℝ := train_length / time_seconds
noncomputable def man_speed_ms : ℝ := relative_speed_ms - train_speed_ms
noncomputable def man_speed_kmh : ℝ := man_speed_ms * (3600 / 1000)
noncomputable def expected_man_speed_kmh : ℝ := 6.006

theorem man_speed_is_correct : abs (man_speed_kmh - expected_man_speed_kmh) < 0.001 :=
by
  -- proof goes here
  sorry

end man_speed_is_correct_l57_57085


namespace simplify_to_quadratic_form_l57_57335

noncomputable def simplify_expression (p : ℝ) : ℝ :=
  ((6 * p + 2) - 3 * p * 5) ^ 2 + (5 - 2 / 4) * (8 * p - 12)

theorem simplify_to_quadratic_form (p : ℝ) : simplify_expression p = 81 * p ^ 2 - 50 :=
sorry

end simplify_to_quadratic_form_l57_57335


namespace total_amount_shared_l57_57930

theorem total_amount_shared (a b c : ℕ) (h_ratio : a * 5 = b * 3) (h_ben : b = 25) (h_ratio_ben : b * 12 = c * 5) :
  a + b + c = 100 := by
  sorry

end total_amount_shared_l57_57930


namespace probability_letter_in_PROBABILITY_l57_57135

theorem probability_letter_in_PROBABILITY :
  let alphabet_size := 26
  let unique_letters_in_PROBABILITY := 9
  (unique_letters_in_PROBABILITY : ℝ) / (alphabet_size : ℝ) = 9 / 26 := by
    sorry

end probability_letter_in_PROBABILITY_l57_57135


namespace pascal_fifth_number_l57_57485

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57485


namespace smallest_base_for_100_l57_57708

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end smallest_base_for_100_l57_57708


namespace cost_of_shoes_is_150_l57_57327

def cost_sunglasses : ℕ := 50
def pairs_sunglasses : ℕ := 2
def cost_jeans : ℕ := 100

def cost_basketball_cards : ℕ := 25
def decks_basketball_cards : ℕ := 2

-- Define the total amount spent by Mary and Rose
def total_mary : ℕ := cost_sunglasses * pairs_sunglasses + cost_jeans
def cost_shoes (total_rose : ℕ) (cost_cards : ℕ) : ℕ := total_rose - cost_cards

theorem cost_of_shoes_is_150 (total_spent : ℕ) :
  total_spent = total_mary →
  cost_shoes total_spent (cost_basketball_cards * decks_basketball_cards) = 150 :=
by
  intro h
  sorry

end cost_of_shoes_is_150_l57_57327


namespace find_A_max_min_l57_57258

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end find_A_max_min_l57_57258


namespace solution_set_of_inequality_l57_57288

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x, x ≥ 0 → f x = 2^x - 4

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h1 : is_even_function f)
  (h2 : satisfies_condition f) :
  {x : ℝ | f (x - 2) > 0} = {x : ℝ | x < 0 ∨ x > 4} :=
sorry

end solution_set_of_inequality_l57_57288


namespace find_math_marks_l57_57096

theorem find_math_marks :
  ∀ (english marks physics chemistry biology : ℕ) (average : ℕ),
  average = 78 →
  english = 91 →
  physics = 82 →
  chemistry = 67 →
  biology = 85 →
  (english + marks + physics + chemistry + biology) / 5 = average →
  marks = 65 :=
by
  intros english marks physics chemistry biology average h_average h_english h_physics h_chemistry h_biology h_avg_eq
  sorry

end find_math_marks_l57_57096


namespace arithmetic_sequence_sum_l57_57050

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end arithmetic_sequence_sum_l57_57050


namespace ivan_expected_shots_l57_57159

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l57_57159


namespace find_d_minus_b_l57_57324

theorem find_d_minus_b (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a^5 = b^4) (h2 : c^3 = d^2) (h3 : c - a = 19) : d - b = 757 := 
by sorry

end find_d_minus_b_l57_57324


namespace possible_values_of_a_l57_57831

-- Define the sets P and Q under the conditions given
def P : Set ℝ := {x | x^2 + x - 6 = 0}
def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

-- Prove that if Q ⊆ P, then a ∈ {0, 1/3, -1/2}
theorem possible_values_of_a (a : ℝ) (h : Q a ⊆ P) : a = 0 ∨ a = 1/3 ∨ a = -1/2 :=
sorry

end possible_values_of_a_l57_57831


namespace gcd_factorial_8_6_squared_l57_57772

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end gcd_factorial_8_6_squared_l57_57772


namespace distinct_products_count_is_26_l57_57987

open Finset

def set_numbers : Finset ℕ := {2, 3, 5, 7, 11}

def distinct_products_count (s : Finset ℕ) : ℕ :=
  let pairs := s.powerset.filter (λ t => 2 ≤ t.card)
  pairs.card

theorem distinct_products_count_is_26 : distinct_products_count set_numbers = 26 := by
  sorry

end distinct_products_count_is_26_l57_57987


namespace geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l57_57819

def seq_an : ℕ → ℝ := sorry
def sum_Sn : ℕ → ℝ := sorry

axiom Sn_recurrence (n : ℕ) : sum_Sn (n + 1) = (1/2) * sum_Sn n + 2
axiom a1_def : seq_an 1 = 2
axiom a2_def : seq_an 2 = 1

theorem geometric_seq (n : ℕ) : ∃ r : ℝ, ∀ (m : ℕ), sum_Sn m - 4 = (sum_Sn 1 - 4) * r^(m-1) := 
sorry

theorem an_formula (n : ℕ) : seq_an n = (1/2)^(n-2) := 
sorry

theorem inequality_proof (t n : ℕ) (t_pos : 0 < t) : 
  (seq_an t * sum_Sn (n + 1) - 1) / (seq_an t * seq_an (n + 1) - 1) < 1/2 :=
sorry

theorem find_t : ∃ (t : ℕ), t = 3 ∨ t = 4 := 
sorry

theorem sum_not_in_seq (m n k : ℕ) (distinct : k ≠ m ∧ m ≠ n ∧ k ≠ n) : 
  (seq_an m + seq_an n ≠ seq_an k) :=
sorry

end geometric_seq_an_formula_inequality_proof_find_t_sum_not_in_seq_l57_57819


namespace liliane_has_44_44_more_cookies_l57_57656

variables (J : ℕ) (L O : ℕ) (totalCookies : ℕ)

def liliane_has_more_30_percent (J L : ℕ) : Prop :=
  L = J + (3 * J / 10)

def oliver_has_less_10_percent (J O : ℕ) : Prop :=
  O = J - (J / 10)

def total_cookies (J L O totalCookies : ℕ) : Prop :=
  J + L + O = totalCookies

theorem liliane_has_44_44_more_cookies
  (h1 : liliane_has_more_30_percent J L)
  (h2 : oliver_has_less_10_percent J O)
  (h3 : total_cookies J L O totalCookies)
  (h4 : totalCookies = 120) :
  (L - O) * 100 / O = 4444 / 100 := sorry

end liliane_has_44_44_more_cookies_l57_57656


namespace pascal_fifth_element_15th_row_l57_57596

theorem pascal_fifth_element_15th_row :
  (Nat.choose 15 4) = 1365 :=
by {
  sorry 
}

end pascal_fifth_element_15th_row_l57_57596


namespace find_A_max_min_l57_57257

def is_coprime_with_36 (n : ℕ) : Prop := Nat.gcd n 36 = 1

def move_last_digit_to_first (n : ℕ) : ℕ :=
  let d := n % 10
  let rest := n / 10
  d * 10^7 + rest

theorem find_A_max_min (B : ℕ) 
  (h1 : B > 77777777) 
  (h2 : is_coprime_with_36 B) : 
  move_last_digit_to_first B = 99999998 ∨ 
  move_last_digit_to_first B = 17777779 := 
by
  sorry

end find_A_max_min_l57_57257


namespace fifth_number_in_pascals_triangle_l57_57589

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57589


namespace gretchen_charge_per_drawing_l57_57985

-- Given conditions
def sold_on_Saturday : ℕ := 24
def sold_on_Sunday : ℕ := 16
def total_amount : ℝ := 800
def total_drawings := sold_on_Saturday + sold_on_Sunday

-- Assertion to prove
theorem gretchen_charge_per_drawing (x : ℝ) (h : total_drawings * x = total_amount) : x = 20 :=
by
  sorry

end gretchen_charge_per_drawing_l57_57985


namespace gcd_factorial_eight_six_sq_l57_57779

/-- Prove that the gcd of 8! and (6!)^2 is 11520 -/
theorem gcd_factorial_eight_six_sq : Int.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 11520 :=
by
  -- We'll put the proof here, but for now, we use sorry to omit it.
  sorry

end gcd_factorial_eight_six_sq_l57_57779


namespace translate_graph_downward_3_units_l57_57894

def f (x : ℝ) : ℝ := 3 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem translate_graph_downward_3_units :
  ∀ x : ℝ, g x = f x - 3 :=
by
  sorry

end translate_graph_downward_3_units_l57_57894


namespace gcd_factorial_eight_squared_six_factorial_squared_l57_57777

theorem gcd_factorial_eight_squared_six_factorial_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 2880 := by
  sorry

end gcd_factorial_eight_squared_six_factorial_squared_l57_57777


namespace smallest_base_to_express_100_with_three_digits_l57_57721

theorem smallest_base_to_express_100_with_three_digits : 
  ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ b' : ℕ, (b'^2 ≤ 100 ∧ 100 < b'^3) → b ≤ b' ∧ b = 5 :=
by
  sorry

end smallest_base_to_express_100_with_three_digits_l57_57721


namespace trapezoid_area_correct_l57_57109

-- Given sides of the trapezoid
def sides : List ℚ := [4, 6, 8, 10]

-- Definition of the function to calculate the sum of all possible areas.
noncomputable def sumOfAllPossibleAreas (sides : List ℚ) : ℚ :=
  -- Assuming configurations and calculations are correct by problem statement
  let r4 := 21
  let r5 := 7
  let r6 := 0
  let n4 := 3
  let n5 := 15
  r4 + r5 + r6 + n4 + n5

-- Check that the given sides lead to sum of areas equal to 46
theorem trapezoid_area_correct : sumOfAllPossibleAreas sides = 46 := by
  sorry

end trapezoid_area_correct_l57_57109


namespace solution_set_of_inequality_l57_57838

theorem solution_set_of_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 2| + |x + 3| ≥ a) ↔ a ≤ 5 :=
sorry

end solution_set_of_inequality_l57_57838


namespace pascal_15_5th_number_l57_57525

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57525


namespace perpendicular_distance_l57_57264

structure Vertex :=
  (x : ℝ) (y : ℝ) (z : ℝ)

def S : Vertex := ⟨6, 0, 0⟩
def P : Vertex := ⟨0, 0, 0⟩
def Q : Vertex := ⟨0, 5, 0⟩
def R : Vertex := ⟨0, 0, 4⟩

noncomputable def distance_from_point_to_plane (S P Q R : Vertex) : ℝ := sorry

theorem perpendicular_distance (S P Q R : Vertex) (hS : S = ⟨6, 0, 0⟩) (hP : P = ⟨0, 0, 0⟩) (hQ : Q = ⟨0, 5, 0⟩) (hR : R = ⟨0, 0, 4⟩) :
  distance_from_point_to_plane S P Q R = 6 :=
  sorry

end perpendicular_distance_l57_57264


namespace pascal_triangle_fifth_number_l57_57603

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57603


namespace rational_solutions_quad_eq_iff_k_eq_4_l57_57424

theorem rational_solutions_quad_eq_iff_k_eq_4 (k : ℕ) (hk : 0 < k) : 
  (∃ x : ℚ, x^2 + 24/k * x + 9 = 0) ↔ k = 4 :=
sorry

end rational_solutions_quad_eq_iff_k_eq_4_l57_57424


namespace arithmetic_sequence_sum_l57_57051

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end arithmetic_sequence_sum_l57_57051


namespace tan_20_plus_4sin_20_eq_sqrt3_l57_57280

theorem tan_20_plus_4sin_20_eq_sqrt3 :
  (Real.tan (20 * Real.pi / 180) + 4 * Real.sin (20 * Real.pi / 180)) = Real.sqrt 3 := by
  sorry

end tan_20_plus_4sin_20_eq_sqrt3_l57_57280


namespace fifth_number_in_pascals_triangle_l57_57591

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57591


namespace find_x_if_perpendicular_l57_57453

-- Define vectors a and b in the given conditions
def vector_a (x : ℝ) : ℝ × ℝ := (x - 5, 3)
def vector_b (x : ℝ) : ℝ × ℝ := (2, x)

-- Define the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- The Lean theorem statement equivalent to the math problem
theorem find_x_if_perpendicular (x : ℝ) (h : dot_product (vector_a x) (vector_b x) = 0) : x = 2 := by
  sorry

end find_x_if_perpendicular_l57_57453


namespace fifth_number_in_pascal_row_l57_57613

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57613


namespace inequality_xy_yz_zx_l57_57964

theorem inequality_xy_yz_zx {x y z : ℝ} (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x * y + 2 * y * z + 2 * z * x) / (x^2 + y^2 + z^2) <= 1 / 4 * (Real.sqrt 33 + 1) :=
sorry

end inequality_xy_yz_zx_l57_57964


namespace arithmetic_sequence_sum_l57_57048

theorem arithmetic_sequence_sum (d : ℕ) (y : ℕ) (x : ℕ) (h_y : y = 39) (h_d : d = 6) 
  (h_x : x = y - d) : 
  x + y = 72 := by 
  sorry

end arithmetic_sequence_sum_l57_57048


namespace value_of_expression_l57_57814

theorem value_of_expression (x y z : ℤ) (hx : x = 3) (hy : y = 2) (hz : z = 4) : 
  (3 * x - 4 * y) / z = 1 / 4 := 
by 
  sorry

end value_of_expression_l57_57814


namespace pascal_triangle_fifth_number_l57_57514

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57514


namespace smallest_base_for_100_l57_57710

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end smallest_base_for_100_l57_57710


namespace sqrt_of_sum_of_powers_l57_57042

theorem sqrt_of_sum_of_powers : Real.sqrt (4^3 + 4^3 + 4^3 + 4^3) = 16 := by
  sorry

end sqrt_of_sum_of_powers_l57_57042


namespace pascal_triangle_fifth_number_l57_57576

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57576


namespace find_age_of_30th_student_l57_57672

theorem find_age_of_30th_student :
  let avg1 := 23.5
  let n1 := 30
  let avg2 := 21.3
  let n2 := 9
  let avg3 := 19.7
  let n3 := 12
  let avg4 := 24.2
  let n4 := 7
  let avg5 := 35
  let n5 := 1
  let total_age_30 := n1 * avg1
  let total_age_9 := n2 * avg2
  let total_age_12 := n3 * avg3
  let total_age_7 := n4 * avg4
  let total_age_1 := n5 * avg5
  let total_age_29 := total_age_9 + total_age_12 + total_age_7 + total_age_1
  let age_30th := total_age_30 - total_age_29
  age_30th = 72.5 :=
by
  sorry

end find_age_of_30th_student_l57_57672


namespace length_of_bridge_l57_57248

noncomputable def L_train : ℝ := 110
noncomputable def v_train : ℝ := 72 * (1000 / 3600)
noncomputable def t : ℝ := 12.099

theorem length_of_bridge : (v_train * t - L_train) = 131.98 :=
by
  -- The proof should come here
  sorry

end length_of_bridge_l57_57248


namespace pascal_fifth_number_l57_57489

theorem pascal_fifth_number (n : ℕ) (hn : n = 15) : ∑ (k : ℕ) in (finset.range 5), (nat.choose n k) = 1365 := 
by 
  -- The statement asserts that the fifth number in the specific Pascal's triangle row is 1365. 
  -- Proof is to be completed.
  sorry

end pascal_fifth_number_l57_57489


namespace perimeter_of_triangle_l57_57444

noncomputable def ellipse_perimeter (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) : ℝ :=
  let a := 2
  let c := 1
  2 * a + 2 * c

theorem perimeter_of_triangle (x y : ℝ) (h : x^2 / 4 + y^2 / 3 = 1) :
  ellipse_perimeter x y h = 6 :=
by 
  sorry

end perimeter_of_triangle_l57_57444


namespace line_through_A_parallel_line_through_B_perpendicular_l57_57239

-- 1. Prove the equation of the line passing through point A(2, 1) and parallel to the line 2x + y - 10 = 0 is 2x + y - 5 = 0.
theorem line_through_A_parallel :
  ∃ (l : ℝ → ℝ), (∀ x, 2 * x + l x - 5 = 0) ∧ (l 2 = 1) ∧ (∃ k, ∀ x, l x = -2 * (x - 2) + k) :=
sorry

-- 2. Prove the equation of the line passing through point B(3, 2) and perpendicular to the line 4x + 5y - 8 = 0 is 5x - 4y - 7 = 0.
theorem line_through_B_perpendicular :
  ∃ (m : ℝ) (l : ℝ → ℝ), (∀ x, 5 * x - 4 * l x - 7 = 0) ∧ (l 3 = 2) ∧ (m = -7) :=
sorry

end line_through_A_parallel_line_through_B_perpendicular_l57_57239


namespace largest_divisor_of_n_pow4_minus_n_for_composites_l57_57791

def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬(∀ k : ℕ, 1 < k ∧ k < n → ¬(k ∣ n))

theorem largest_divisor_of_n_pow4_minus_n_for_composites : ∀ n : ℕ, is_composite n → 6 ∣ (n^4 - n) :=
by
  intro n
  intro hn
  -- we would add proof steps here if necessary
  sorry

end largest_divisor_of_n_pow4_minus_n_for_composites_l57_57791


namespace simplify_expression_l57_57872

theorem simplify_expression (x : ℝ) : 
  8 * x + 15 - 3 * x + 5 * 7 = 5 * x + 50 :=
by
  sorry

end simplify_expression_l57_57872


namespace gcd_factorial_l57_57762

theorem gcd_factorial (a b : ℕ) (h : a = 8! ∧ b = (6!)^2) : Nat.gcd a b = 5760 :=
by sorry

end gcd_factorial_l57_57762


namespace pascal_triangle_15_4_l57_57508

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57508


namespace john_pays_in_30_day_month_l57_57162

-- The cost of one pill
def cost_per_pill : ℝ := 1.5

-- The number of pills John takes per day
def pills_per_day : ℕ := 2

-- The number of days in a month
def days_in_month : ℕ := 30

-- The insurance coverage percentage
def insurance_coverage : ℝ := 0.40

-- Calculate the total cost John has to pay after insurance coverage in a 30-day month
theorem john_pays_in_30_day_month : (2 * 30) * 1.5 * 0.60 = 54 :=
by
  sorry

end john_pays_in_30_day_month_l57_57162


namespace pascal_15_5th_number_l57_57522

def pascal_fifth_number_of_row (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

theorem pascal_15_5th_number : pascal_fifth_number_of_row 15 4 = 1365 := by
  sorry

end pascal_15_5th_number_l57_57522


namespace find_x_l57_57474

theorem find_x (x : ℝ) : 
  16^(x + 2) = 300 + 12 * 16^x → 
  x = Real.log (150/122) / Real.log 16 :=
by
  intro h
  sorry

end find_x_l57_57474


namespace polynomial_lt_factorial_l57_57439

theorem polynomial_lt_factorial (A B C : ℝ) : ∃N : ℕ, ∀n : ℕ, n > N → An^2 + Bn + C < n! := 
by
  sorry

end polynomial_lt_factorial_l57_57439


namespace classroom_count_l57_57876

-- Definitions for conditions
def average_age_all (sum_ages : ℕ) (num_people : ℕ) : ℕ := sum_ages / num_people
def average_age_excluding_teacher (sum_ages : ℕ) (num_people : ℕ) (teacher_age : ℕ) : ℕ :=
  (sum_ages - teacher_age) / (num_people - 1)

-- Theorem statement using the provided conditions
theorem classroom_count (x : ℕ) (h1 : average_age_all (11 * x) x = 11)
  (h2 : average_age_excluding_teacher (11 * x) x 30 = 10) : x = 20 :=
  sorry

end classroom_count_l57_57876


namespace pascal_triangle_fifth_number_l57_57494

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57494


namespace points_3_units_away_from_origin_l57_57883

theorem points_3_units_away_from_origin (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end points_3_units_away_from_origin_l57_57883


namespace Tricia_is_five_years_old_l57_57371

noncomputable def Vincent_age : ℕ := 22
noncomputable def Rupert_age : ℕ := Vincent_age - 2
noncomputable def Khloe_age : ℕ := Rupert_age - 10
noncomputable def Eugene_age : ℕ := 3 * Khloe_age
noncomputable def Yorick_age : ℕ := 2 * Eugene_age
noncomputable def Amilia_age : ℕ := Yorick_age / 4
noncomputable def Tricia_age : ℕ := Amilia_age / 3

theorem Tricia_is_five_years_old : Tricia_age = 5 := by
  unfold Tricia_age Amilia_age Yorick_age Eugene_age Khloe_age Rupert_age Vincent_age
  sorry

end Tricia_is_five_years_old_l57_57371


namespace project_contribution_l57_57021

theorem project_contribution (total_cost : ℝ) (num_participants : ℝ) (expected_contribution : ℝ) 
  (h1 : total_cost = 25 * 10^9) 
  (h2 : num_participants = 300 * 10^6) 
  (h3 : expected_contribution = 83) : 
  total_cost / num_participants = expected_contribution := 
by 
  sorry

end project_contribution_l57_57021


namespace composite_divisible_by_six_l57_57808

theorem composite_divisible_by_six (n : ℤ) (h : ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ n = a * b) : 6 ∣ (n^4 - n) :=
sorry

end composite_divisible_by_six_l57_57808


namespace solution_l57_57020

def question (x : ℝ) : Prop := (x - 5) / ((x - 3) ^ 2) < 0

theorem solution :
  {x : ℝ | question x} = {x : ℝ | x < 3} ∪ {x : ℝ | 3 < x ∧ x < 5} :=
by {
  sorry
}

end solution_l57_57020


namespace min_value_inequality_l57_57852

theorem min_value_inequality (y1 y2 y3 : ℝ) (h_pos : 0 < y1 ∧ 0 < y2 ∧ 0 < y3) (h_sum : 2 * y1 + 3 * y2 + 4 * y3 = 120) :
  y1^2 + 4 * y2^2 + 9 * y3^2 ≥ 14400 / 29 :=
sorry

end min_value_inequality_l57_57852


namespace min_value_frac_2_over_a_plus_3_over_b_l57_57438

theorem min_value_frac_2_over_a_plus_3_over_b 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (hline : 2 * a + 3 * b = 1) :
  (2 / a + 3 / b) ≥ 25 :=
sorry

end min_value_frac_2_over_a_plus_3_over_b_l57_57438


namespace bananas_to_oranges_l57_57196

theorem bananas_to_oranges :
  (3 / 4 : ℝ) * 16 = 12 →
  (2 / 3 : ℝ) * 9 = 6 :=
by
  intro h
  sorry

end bananas_to_oranges_l57_57196


namespace percentage_difference_l57_57061

theorem percentage_difference (x y : ℝ) (h : x = 3 * y) : ((x - y) / x) * 100 = 66.67 :=
by
  sorry

end percentage_difference_l57_57061


namespace largest_divisor_composite_difference_l57_57796

-- Define when an integer is composite
def is_composite (n : ℕ) : Prop :=
  n > 1 ∧ ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0

theorem largest_divisor_composite_difference (n : ℕ) (h : is_composite n) : 6 ∣ (n^4 - n) :=
by sorry

end largest_divisor_composite_difference_l57_57796


namespace winnie_keeps_lollipops_l57_57385

theorem winnie_keeps_lollipops :
  let cherry := 36
  let wintergreen := 125
  let grape := 8
  let shrimp_cocktail := 241
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 13
  total_lollipops % friends = 7 :=
by
  sorry

end winnie_keeps_lollipops_l57_57385


namespace cheryl_gave_mms_to_sister_l57_57754

-- Definitions for given conditions in the problem
def ate_after_lunch : ℕ := 7
def ate_after_dinner : ℕ := 5
def initial_mms : ℕ := 25

-- The statement to be proved
theorem cheryl_gave_mms_to_sister : (initial_mms - (ate_after_lunch + ate_after_dinner)) = 13 := by
  sorry

end cheryl_gave_mms_to_sister_l57_57754


namespace gcd_of_B_l57_57634

def is_in_B (n : ℕ) := ∃ x : ℕ, x > 0 ∧ n = 4*x + 2

theorem gcd_of_B : ∃ d, (∀ n, is_in_B n → d ∣ n) ∧ (∀ d', (∀ n, is_in_B n → d' ∣ n) → d' ∣ d) ∧ d = 2 := 
by
  sorry

end gcd_of_B_l57_57634


namespace find_initial_number_l57_57783

theorem find_initial_number (N : ℕ) (k : ℤ) (h : N - 3 = 15 * k) : N = 18 := 
by
  sorry

end find_initial_number_l57_57783


namespace minimum_of_f_range_of_a_l57_57303

open Real

noncomputable def f (x : ℝ) : ℝ := x * log x

theorem minimum_of_f :
  let e_inv := (1 : ℝ) / Real.exp 1 in
  (∀ x > 0, f x ≥ -e_inv) ∧ (f e_inv = -e_inv) := 
by
  intro e_inv
  simp [f, e_inv]
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 1, f x ≥ a * x - 1) ↔ a ≤ 1 := 
by
  sorry

end minimum_of_f_range_of_a_l57_57303


namespace probability_of_rolling_3_or_5_is_1_over_4_l57_57054

def fair_8_sided_die := {outcome : Fin 8 // true}

theorem probability_of_rolling_3_or_5_is_1_over_4 :
  (1 / 4 : ℚ) = 2 / 8 :=
by sorry

end probability_of_rolling_3_or_5_is_1_over_4_l57_57054


namespace find_P_l57_57204

variable (a b c d P : ℝ)

theorem find_P 
  (h1 : (a + b + c + d) / 4 = 8) 
  (h2 : (a + b + c + d + P) / 5 = P) : 
  P = 8 := 
by 
  sorry

end find_P_l57_57204


namespace geometric_progression_theorem_l57_57864

variables {a b c : ℝ} {n : ℕ} {q : ℝ}

-- Define the terms in the geometric progression
def nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^n
def second_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(2 * n)
def fourth_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(4 * n)

-- Conditions
axiom nth_term_def : b = nth_term a q n
axiom second_nth_term_def : b = second_nth_term a q n
axiom fourth_nth_term_def : c = fourth_nth_term a q n

-- Statement to prove
theorem geometric_progression_theorem :
  b * (b^2 - a^2) = a^2 * (c - b) :=
sorry

end geometric_progression_theorem_l57_57864


namespace rain_probability_tel_aviv_l57_57351

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

noncomputable def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (binomial_coefficient n k) * (p^k) * ((1 - p)^(n - k))

theorem rain_probability_tel_aviv : binomial_probability 6 4 0.5 = 0.234375 :=
  by
    sorry

end rain_probability_tel_aviv_l57_57351


namespace square_side_length_l57_57190

theorem square_side_length 
  (A B C D E : Type) 
  (AB AC hypotenuse square_side_length : ℝ) 
  (h1: AB = 9) 
  (h2: AC = 12) 
  (h3: hypotenuse = Real.sqrt (9^2 + 12^2)) 
  (h4: square_side_length = 300 / 41) 
  : square_side_length = 300 / 41 := 
by 
  sorry

end square_side_length_l57_57190


namespace negative_number_among_options_l57_57414

theorem negative_number_among_options : 
  ∃ (x : ℤ), x < 0 ∧ 
    (x = |-4| ∨ x = -(-4) ∨ x = (-4)^2 ∨ x = -4^2)
:= by
  use -16
  split
  {
    -- prove that -16 is negative
    linarith
  }
  {
    -- prove that -16 is one of the options
    right; right; right
    norm_num
  }

end negative_number_among_options_l57_57414


namespace incorrect_statement_B_l57_57283

-- Defining the quadratic function
def quadratic_function (x : ℝ) : ℝ := -(x + 2)^2 - 3

-- Conditions derived from the problem
def statement_A : Prop := (∃ h k : ℝ, h < 0 ∧ k = 0)
def statement_B : Prop := (axis_of_symmetry (quadratic_function) = 2)
def statement_C : Prop := (¬ ∃ x : ℝ, quadratic_function x = 0)
def statement_D : Prop := (∀ x > -1, ∀ y > x, quadratic_function y < quadratic_function x)

-- The proof problem: show that statement B is incorrect
theorem incorrect_statement_B : statement_B = false :=
by sorry

end incorrect_statement_B_l57_57283


namespace washing_machine_cost_l57_57127

variable (W D : ℝ)
variable (h1 : D = W - 30)
variable (h2 : 0.90 * (W + D) = 153)

theorem washing_machine_cost :
  W = 100 := by
  sorry

end washing_machine_cost_l57_57127


namespace determine_c_l57_57937

theorem determine_c (c : ℝ) (r : ℝ) (h1 : 2 * r^2 - 8 * r - c = 0) (h2 : r ≠ 0) (h3 : 2 * (r + 5.5)^2 + 5 * (r + 5.5) = c) :
  c = 12 :=
sorry

end determine_c_l57_57937


namespace total_stocks_l57_57663

-- Define the conditions as given in the math problem
def closed_higher : ℕ := 1080
def ratio : ℝ := 1.20

-- Using ℕ for the number of stocks that closed lower
def closed_lower (x : ℕ) : Prop := 1080 = x * ratio ∧ closed_higher = x + x * (1 / 5)

-- Definition to compute the total number of stocks on the stock exchange
def total_number_of_stocks (x : ℕ) : ℕ := closed_higher + x

-- The main theorem to be proved
theorem total_stocks (x : ℕ) (h : closed_lower x) : total_number_of_stocks x = 1980 :=
sorry

end total_stocks_l57_57663


namespace find_base_number_l57_57705

-- Define the base number
def base_number (x : ℕ) (k : ℕ) : Prop := x ^ k > 4 ^ 22

-- State the theorem based on the problem conditions
theorem find_base_number : ∃ x : ℕ, ∀ k : ℕ, (k = 8) → (base_number x k) → (x = 64) :=
by sorry

end find_base_number_l57_57705


namespace number_of_students_at_table_l57_57667

theorem number_of_students_at_table :
  ∃ (n : ℕ), n ∣ 119 ∧ (n = 7 ∨ n = 17) :=
sorry

end number_of_students_at_table_l57_57667


namespace ivan_expected_shots_l57_57160

noncomputable def expected_shots (n : ℕ) (p_hit : ℝ) : ℝ :=
let a := (1 - p_hit) + p_hit * (1 + 3 * a) in
n * (1 / (1 - 0.3))

theorem ivan_expected_shots : expected_shots 14 0.1 = 20 := by
  sorry

end ivan_expected_shots_l57_57160


namespace g_five_eq_one_l57_57208

noncomputable def g : ℝ → ℝ := sorry

axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_nonzero : ∀ x : ℝ, g x ≠ 0

theorem g_five_eq_one : g 5 = 1 :=
by
  sorry

end g_five_eq_one_l57_57208


namespace mixed_doubles_selection_l57_57925

-- Given conditions
def num_male_players : ℕ := 5
def num_female_players : ℕ := 4

-- The statement to show the number of different ways to select two players is 20
theorem mixed_doubles_selection : (num_male_players * num_female_players) = 20 := by
  -- Proof to be filled in
  sorry

end mixed_doubles_selection_l57_57925


namespace calc_perimeter_l57_57389

noncomputable def width (w: ℝ) (h: ℝ) : Prop :=
  h = w + 10

noncomputable def cost (P: ℝ) (rate: ℝ) (total_cost: ℝ) : Prop :=
  total_cost = P * rate

noncomputable def perimeter (w: ℝ) (P: ℝ) : Prop :=
  P = 2 * (w + (w + 10))

theorem calc_perimeter {w P : ℝ} (h_rate : ℝ) (h_total_cost : ℝ)
  (h1 : width w (w + 10))
  (h2 : cost (2 * (w + (w + 10))) h_rate h_total_cost) :
  P = 2 * (w + (w + 10)) →
  h_total_cost = 910 →
  h_rate = 6.5 →
  w = 30 →
  P = 140 :=
sorry

end calc_perimeter_l57_57389


namespace police_station_distance_l57_57927

theorem police_station_distance (thief_speed police_speed: ℝ) (delay chase_time: ℝ) 
  (h_thief_speed: thief_speed = 20) 
  (h_police_speed: police_speed = 40) 
  (h_delay: delay = 1)
  (h_chase_time: chase_time = 4) : 
  ∃ D: ℝ, D = 60 :=
by
  sorry

end police_station_distance_l57_57927


namespace shifted_line_does_not_pass_through_third_quadrant_l57_57727

-- The condition: The original line is y = -2x - 1
def original_line (x : ℝ) : ℝ := -2 * x - 1

-- The condition: The line is shifted 3 units to the right
def shifted_line (x : ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_through_third_quadrant :
  ¬(∃ (x y : ℝ), y = shifted_line x ∧ x < 0 ∧ y < 0) :=
sorry

end shifted_line_does_not_pass_through_third_quadrant_l57_57727


namespace total_tickets_sold_l57_57926

-- Definitions and conditions
def orchestra_ticket_price : ℕ := 12
def balcony_ticket_price : ℕ := 8
def total_revenue : ℕ := 3320
def ticket_difference : ℕ := 190

-- Variables
variables (x y : ℕ) -- x is the number of orchestra tickets, y is the number of balcony tickets

-- Statements of conditions
def revenue_eq : Prop := orchestra_ticket_price * x + balcony_ticket_price * y = total_revenue
def tickets_relation : Prop := y = x + ticket_difference

-- The proof problem statement
theorem total_tickets_sold (h1 : revenue_eq x y) (h2 : tickets_relation x y) : x + y = 370 :=
by
  sorry

end total_tickets_sold_l57_57926


namespace larger_number_of_hcf_lcm_l57_57678

theorem larger_number_of_hcf_lcm (hcf : ℕ) (a b : ℕ) (f1 f2 : ℕ) 
  (hcf_condition : hcf = 20) 
  (factors_condition : f1 = 21 ∧ f2 = 23) 
  (lcm_condition : Nat.lcm a b = hcf * f1 * f2):
  max a b = 460 := 
  sorry

end larger_number_of_hcf_lcm_l57_57678


namespace fifth_number_in_pascal_row_l57_57618

-- Definition of the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Pascal's triangle row starting with 1 and then 15 corresponds to binomial coefficients with n = 15
def pascalRowStartsWith1And15 : Prop := 
  ∃ (k : ℕ), binomial 15 k = 15

-- Prove that the fifth number in this row is 1365
theorem fifth_number_in_pascal_row : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascal_row_l57_57618


namespace pascal_row_fifth_number_l57_57530

-- Define the given conditions
def row_corresponds_to_binomials : Prop :=
  ∀ n : ℕ, ∀ k : ℕ, (n = 15 ∧ k ≤ 4) -> 
  (∀ binomialTheorem : ℕ,  binomialTheorem = nat.choose n k)

-- State the problem and the proof target
theorem pascal_row_fifth_number :
  ∀ k : ℕ, k = 4 -> (15.choose k) = 1365 := 
by
  intros k hk
  rw hk
  have : row_corresponds_to_binomials, sorry
  exact this 15 4 (by simp)

end pascal_row_fifth_number_l57_57530


namespace original_number_of_movies_l57_57405

/-- Suppose a movie buff owns movies on DVD, Blu-ray, and digital copies in a ratio of 7:2:1.
    After purchasing 5 more Blu-ray movies and 3 more digital copies, the ratio changes to 13:4:2.
    She owns movies on no other medium.
    Prove that the original number of movies in her library before the extra purchase was 390. -/
theorem original_number_of_movies (x : ℕ) (h1 : 7 * x != 0) 
  (h2 : 2 * x != 0) (h3 : x != 0)
  (h4 : 7 * x / (2 * x + 5) = 13 / 4)
  (h5 : 7 * x / (x + 3) = 13 / 2) : 10 * x = 390 :=
by
  sorry

end original_number_of_movies_l57_57405


namespace range_of_m_in_first_quadrant_l57_57317

theorem range_of_m_in_first_quadrant (m : ℝ) : ((m - 1 > 0) ∧ (m + 2 > 0)) ↔ m > 1 :=
by sorry

end range_of_m_in_first_quadrant_l57_57317


namespace right_triangle_legs_l57_57840

theorem right_triangle_legs (a b : ℤ) (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a^2 + b^2 = 65^2) : 
  a = 16 ∧ b = 63 ∨ a = 63 ∧ b = 16 :=
sorry

end right_triangle_legs_l57_57840


namespace Tom_final_balance_l57_57893

theorem Tom_final_balance :
  let initial_allowance := 12
  let week1_spending := initial_allowance / 3
  let balance_after_week1 := initial_allowance - week1_spending
  let week2_spending := balance_after_week1 / 4
  let balance_after_week2 := balance_after_week1 - week2_spending
  let additional_earning := 5
  let balance_after_earning := balance_after_week2 + additional_earning
  let week3_spending := balance_after_earning / 2
  let balance_after_week3 := balance_after_earning - week3_spending
  let penultimate_day_spending := 3
  let final_balance := balance_after_week3 - penultimate_day_spending
  final_balance = 2.50 :=
by
  sorry

end Tom_final_balance_l57_57893


namespace pudding_cups_initial_l57_57888

theorem pudding_cups_initial (P : ℕ) (students : ℕ) (extra_cups : ℕ) 
  (h1 : students = 218) (h2 : extra_cups = 121) (h3 : P + extra_cups = students) : P = 97 := 
by
  sorry

end pudding_cups_initial_l57_57888


namespace robin_spent_on_leftover_drinks_l57_57666

-- Define the number of each type of drink bought and consumed
def sodas_bought : Nat := 30
def sodas_price : Nat := 2
def sodas_consumed : Nat := 10

def energy_drinks_bought : Nat := 20
def energy_drinks_price : Nat := 3
def energy_drinks_consumed : Nat := 14

def smoothies_bought : Nat := 15
def smoothies_price : Nat := 4
def smoothies_consumed : Nat := 5

-- Define the total cost calculation
def total_spent_on_leftover_drinks : Nat :=
  (sodas_bought * sodas_price - sodas_consumed * sodas_price) +
  (energy_drinks_bought * energy_drinks_price - energy_drinks_consumed * energy_drinks_price) +
  (smoothies_bought * smoothies_price - smoothies_consumed * smoothies_price)

theorem robin_spent_on_leftover_drinks : total_spent_on_leftover_drinks = 98 := by
  -- Provide the proof steps here (not required for this task)
  sorry

end robin_spent_on_leftover_drinks_l57_57666


namespace pascal_triangle_fifth_number_l57_57607

theorem pascal_triangle_fifth_number (n k r : ℕ) (h1 : n = 15) (h2 : k = 4) (h3 : r = Nat.choose n k) : 
  r = 1365 := 
by
  -- The proof goes here
  sorry

end pascal_triangle_fifth_number_l57_57607


namespace largest_divisor_of_difference_between_n_and_n4_l57_57800

theorem largest_divisor_of_difference_between_n_and_n4 (n : ℤ) (h_composite : ∃ p q : ℕ, 1 < p ∧ 1 < q ∧ p * q = n) :
  ∃ k : ℤ, k = 6 ∧ k ∣ (n^4 - n) :=
begin
  sorry
end

end largest_divisor_of_difference_between_n_and_n4_l57_57800


namespace pascal_triangle_fifth_number_l57_57519

/-- In the row of Pascal's triangle that starts with 1 and then 15, the fifth number is 1365. -/
theorem pascal_triangle_fifth_number :
  (Nat.choose 15 4) = 1365 :=
by
  sorry

end pascal_triangle_fifth_number_l57_57519


namespace find_x_l57_57910

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l57_57910


namespace A_alone_finishes_work_in_30_days_l57_57730

noncomputable def work_rate_A (B : ℝ) : ℝ := 2 * B

noncomputable def total_work (B : ℝ) : ℝ := 60 * B

theorem A_alone_finishes_work_in_30_days (B : ℝ) : (total_work B) / (work_rate_A B) = 30 := by
  sorry

end A_alone_finishes_work_in_30_days_l57_57730


namespace box_dimension_triples_l57_57246

theorem box_dimension_triples (N : ℕ) :
  ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (1 / a + 1 / b + 1 / c = 1 / 8) → ∃ k, k = N := sorry 

end box_dimension_triples_l57_57246


namespace complex_abs_sum_eq_1_or_3_l57_57179

open Complex

theorem complex_abs_sum_eq_1_or_3 (a b c : ℂ) (ha : abs a = 1) (hb : abs b = 1) (hc : abs c = 1) 
  (h : a^3/(b^2 * c) + b^3/(a^2 * c) + c^3/(a^2 * b) = 1) : abs (a + b + c) = 1 ∨ abs (a + b + c) = 3 := 
by
  sorry

end complex_abs_sum_eq_1_or_3_l57_57179


namespace sqrt_sum_of_cubes_l57_57046

theorem sqrt_sum_of_cubes :
  √(4^3 + 4^3 + 4^3 + 4^3) = 16 :=
by
  sorry

end sqrt_sum_of_cubes_l57_57046


namespace admission_price_for_children_l57_57365

theorem admission_price_for_children (people_at_play : ℕ) (admission_price_adult : ℕ) (total_receipts : ℕ) (adults_attended : ℕ) 
  (h1 : people_at_play = 610) (h2 : admission_price_adult = 2) (h3 : total_receipts = 960) (h4 : adults_attended = 350) : 
  ∃ (admission_price_child : ℕ), admission_price_child = 1 :=
by
  sorry

end admission_price_for_children_l57_57365


namespace mod_exp_sub_l57_57756

theorem mod_exp_sub (a b k : ℕ) (h₁ : a ≡ 6 [MOD 7]) (h₂ : b ≡ 4 [MOD 7]) :
  (a ^ k - b ^ k) % 7 = 2 :=
sorry

end mod_exp_sub_l57_57756


namespace oak_trees_problem_l57_57693

theorem oak_trees_problem (c t n : ℕ) 
  (h1 : c = 9) 
  (h2 : t = 11) 
  (h3 : t = c + n) 
  : n = 2 := 
by 
  sorry

end oak_trees_problem_l57_57693


namespace g_f_neg3_l57_57649

def f (x : ℤ) : ℤ := x^3 - 1
def g (x : ℤ) : ℤ := 3 * x^2 + 3 * x + 1

theorem g_f_neg3 : g (f (-3)) = 2285 :=
by
  -- provide the proof here
  sorry

end g_f_neg3_l57_57649


namespace new_person_weight_l57_57674

-- Define the conditions of the problem
variables (avg_weight : ℝ) (weight_replaced_person : ℝ) (num_persons : ℕ)
variable (weight_increase : ℝ)

-- Given conditions
def condition (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ) (weight_increase : ℝ) : Prop :=
  num_persons = 10 ∧ weight_replaced_person = 60 ∧ weight_increase = 5

-- The proof problem
theorem new_person_weight (avg_weight weight_replaced_person : ℝ) (num_persons : ℕ)
  (weight_increase : ℝ) (h : condition avg_weight weight_replaced_person num_persons weight_increase) :
  weight_replaced_person + num_persons * weight_increase = 110 :=
sorry

end new_person_weight_l57_57674


namespace distance_of_third_point_on_trip_l57_57867

theorem distance_of_third_point_on_trip (D : ℝ) (h1 : D + 2 * D + (1/2) * D + 7 * D = 560) :
  (1/2) * D = 27 :=
by
  sorry

end distance_of_third_point_on_trip_l57_57867


namespace cubic_roots_reciprocal_squares_sum_l57_57939

-- Define the roots a, b, and c
variables (a b c : ℝ)

-- Define the given cubic equation conditions
variables (h1 : a + b + c = 6) (h2 : a * b + b * c + c * a = 11) (h3 : a * b * c = 6)

-- Define the target statement
theorem cubic_roots_reciprocal_squares_sum :
  (1 / a^2) + (1 / b^2) + (1 / c^2) = 49 / 36 :=
by
  sorry

end cubic_roots_reciprocal_squares_sum_l57_57939


namespace kevin_speed_first_half_l57_57002

-- Let's define the conditions as variables and constants
variable (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ)
variable (time_20mph : ℝ) (time_8mph : ℝ) (distance_first_half : ℕ)
variable (speed_first_half : ℝ)

-- Conditions from the problem
def conditions (total_distance : ℕ) (distance_20mph : ℕ) (distance_8mph : ℕ) : Prop :=
  total_distance = 17 ∧ 
  distance_20mph = 20 * 1 / 2 ∧
  distance_8mph = 8 * 1 / 4

-- Proof objective based on conditions and correct answer
theorem kevin_speed_first_half (
  h : conditions total_distance distance_20mph distance_8mph
) : speed_first_half = 10 := by
  sorry

end kevin_speed_first_half_l57_57002


namespace bob_password_probability_l57_57089

def num_non_negative_single_digits : ℕ := 10
def num_odd_single_digits : ℕ := 5
def num_even_positive_single_digits : ℕ := 4
def probability_first_digit_odd : ℚ := num_odd_single_digits / num_non_negative_single_digits
def probability_middle_letter : ℚ := 1
def probability_last_digit_even_positive : ℚ := num_even_positive_single_digits / num_non_negative_single_digits

theorem bob_password_probability :
  probability_first_digit_odd * probability_middle_letter * probability_last_digit_even_positive = 1 / 5 :=
by
  sorry

end bob_password_probability_l57_57089


namespace area_of_triangle_is_24_l57_57758

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

end area_of_triangle_is_24_l57_57758


namespace jellybean_probability_l57_57241

/-- A bowl contains 15 jellybeans: five red, three blue, five white, and two green. If you pick four 
    jellybeans from the bowl at random and without replacement, the probability that exactly three will 
    be red is 20/273. -/
theorem jellybean_probability :
  let total_jellybeans := 15
  let red_jellybeans := 5
  let blue_jellybeans := 3
  let white_jellybeans := 5
  let green_jellybeans := 2
  let total_combinations := Nat.choose total_jellybeans 4
  let favorable_combinations := (Nat.choose red_jellybeans 3) * (Nat.choose (total_jellybeans - red_jellybeans) 1)
  let probability := favorable_combinations / total_combinations
  probability = 20 / 273 :=
by
  sorry

end jellybean_probability_l57_57241


namespace shifted_line_does_not_pass_third_quadrant_l57_57726

def line_eq (x: ℝ) : ℝ := -2 * x - 1
def shifted_line_eq (x: ℝ) : ℝ := -2 * (x - 3) - 1

theorem shifted_line_does_not_pass_third_quadrant :
  ¬∃ x y : ℝ, shifted_line_eq x = y ∧ x < 0 ∧ y < 0 :=
sorry

end shifted_line_does_not_pass_third_quadrant_l57_57726


namespace slower_train_speed_is_36_l57_57037

def speed_of_slower_train (v : ℕ) : Prop :=
  let length_of_each_train := 100
  let distance_covered := length_of_each_train * 2
  let time_taken := 72
  let faster_train_speed := 46
  let relative_speed := (faster_train_speed - v) * (1000 / 3600)
  distance_covered = relative_speed * time_taken

theorem slower_train_speed_is_36 : ∃ v, speed_of_slower_train v ∧ v = 36 :=
by
  use 36
  unfold speed_of_slower_train
  -- Prove that the equation holds when v = 36
  sorry

end slower_train_speed_is_36_l57_57037


namespace smallest_positive_integer_in_form_l57_57706

theorem smallest_positive_integer_in_form (m n : ℤ) : 
  ∃ m n : ℤ, 3001 * m + 24567 * n = 1 :=
by
  sorry

end smallest_positive_integer_in_form_l57_57706


namespace distinct_positive_and_conditions_l57_57060

theorem distinct_positive_and_conditions (a b : ℕ) (h_distinct: a ≠ b) (h_pos1: 0 < a) (h_pos2: 0 < b) (h_eq: a^3 - b^3 = a^2 - b^2) : 
  ∃ (c : ℕ), c = 9 * a * b ∧ (c = 1 ∨ c = 2 ∨ c = 3) :=
by
  sorry

end distinct_positive_and_conditions_l57_57060


namespace range_of_x_l57_57147

theorem range_of_x (x : ℝ) : x ≠ 3 ↔ ∃ y : ℝ, y = (x + 2) / (x - 3) :=
by {
  sorry
}

end range_of_x_l57_57147


namespace speed_in_still_water_l57_57244

theorem speed_in_still_water (upstream downstream : ℝ) (h_upstream : upstream = 37) (h_downstream : downstream = 53) : 
  (upstream + downstream) / 2 = 45 := 
by
  sorry

end speed_in_still_water_l57_57244


namespace study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l57_57749

theorem study_video_game_inversely_proportional_1 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : s = 6) : v = 2 :=
by
  sorry

theorem study_video_game_inversely_proportional_2 (k : ℕ) (s : ℕ) (v : ℕ) 
  (h1 : s * v = k) (h2 : k = 12) (h3 : v = 6) : s = 2 :=
by
  sorry

end study_video_game_inversely_proportional_1_study_video_game_inversely_proportional_2_l57_57749


namespace pascal_triangle_fifth_number_l57_57579

theorem pascal_triangle_fifth_number : 
  (Nat.choose 15 4) = 1365 :=
by 
  sorry

end pascal_triangle_fifth_number_l57_57579


namespace ratio_of_radii_of_cylinders_l57_57092

theorem ratio_of_radii_of_cylinders
  (r_V r_B h_V h_B : ℝ)
  (h1 : h_V = 1/2 * h_B)
  (h2 : π * r_B^2 * h_B / 2  = 4)
  (h3 : π * r_V^2 * h_V = 16) :
  r_V / r_B = 2 := 
by 
  sorry

end ratio_of_radii_of_cylinders_l57_57092


namespace river_depth_l57_57924

theorem river_depth (width depth : ℝ) (flow_rate_kmph : ℝ) (volume_m3_per_min : ℝ) 
  (h1 : width = 75) 
  (h2 : flow_rate_kmph = 4) 
  (h3 : volume_m3_per_min = 35000) : 
  depth = 7 := 
by
  sorry

end river_depth_l57_57924


namespace largest_value_satisfies_abs_equation_l57_57105

theorem largest_value_satisfies_abs_equation (x : ℝ) : |5 - x| = 15 + x → x = -5 := by
  intros h
  sorry

end largest_value_satisfies_abs_equation_l57_57105


namespace find_ages_l57_57017

theorem find_ages (P F M : ℕ) 
  (h1 : F - P = 31)
  (h2 : (F + 8) + (P + 8) = 69)
  (h3 : F - M = 4)
  (h4 : (P + 5) + (M + 5) = 65) :
  P = 11 ∧ F = 42 ∧ M = 38 :=
by
  sorry

end find_ages_l57_57017


namespace multiples_6_8_not_both_l57_57471

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l57_57471


namespace pascal_triangle_15_4_l57_57505

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57505


namespace red_apples_sold_l57_57745

-- Define the variables and constants
variables (R G : ℕ)

-- Conditions (Definitions)
def ratio_condition : Prop := R / G = 8 / 3
def combine_condition : Prop := R + G = 44

-- Theorem statement to show number of red apples sold is 32 under given conditions
theorem red_apples_sold : ratio_condition R G → combine_condition R G → R = 32 :=
by
sorry

end red_apples_sold_l57_57745


namespace intersection_is_empty_l57_57112

def A : Set ℝ := { α | ∃ k : ℤ, α = (5 * k * Real.pi) / 3 }
def B : Set ℝ := { β | ∃ k : ℤ, β = (3 * k * Real.pi) / 2 }

theorem intersection_is_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_is_empty_l57_57112


namespace probability_even_sum_rows_columns_l57_57214

open Probability

def has_even_sum_rows_and_columns (grid : Matrix (Fin 3) (Fin 3) ℕ) : Prop :=
  ∀ i, (∑ j, grid i j) % 2 = 0 ∧ (∑ j, grid j i) % 2 = 0

def all_1_to_9 : Fin 3 → Fin 3 → Finset ℕ :=
  {1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem probability_even_sum_rows_columns :
  let grids := { grid : Matrix (Fin 3) (Fin 3) ℕ // ∀ i j, grid i j ∈ all_1_to_9 ∧ ∀ n ∈ all_1_to_9, ∃ (i j : Fin 3), grid i j = n }
  ∃! p : ℚ, (∀ g ∈ grids, has_even_sum_rows_and_columns g → Probs g = p) ∧ p = 3 / 32 :=
sorry

end probability_even_sum_rows_columns_l57_57214


namespace gcd_factorial_eight_squared_six_factorial_squared_l57_57778

theorem gcd_factorial_eight_squared_six_factorial_squared :
  Nat.gcd (Nat.factorial 8) ((Nat.factorial 6) ^ 2) = 2880 := by
  sorry

end gcd_factorial_eight_squared_six_factorial_squared_l57_57778


namespace cost_of_acai_berry_juice_l57_57881

theorem cost_of_acai_berry_juice 
  (cost_per_litre_cocktail : ℝ) 
  (cost_per_litre_mixed_fruit : ℝ)
  (volume_mixed_fruit : ℝ)
  (volume_acai_berry : ℝ)
  (total_volume : ℝ) 
  (total_cost_of_mixed_fruit : ℝ)
  (total_cost_cocktail : ℝ)
  : cost_per_litre_cocktail = 1399.45 ∧ 
    cost_per_litre_mixed_fruit = 262.85 ∧ 
    volume_mixed_fruit = 37 ∧ 
    volume_acai_berry = 24.666666666666668 ∧ 
    total_volume = 61.666666666666668 ∧ 
    total_cost_of_mixed_fruit = volume_mixed_fruit * cost_per_litre_mixed_fruit ∧
    total_cost_of_mixed_fruit = 9725.45 ∧
    total_cost_cocktail = total_volume * cost_per_litre_cocktail ∧ 
    total_cost_cocktail = 86327.77 
    → 24.666666666666668 * 3105.99 + 9725.45 = 86327.77 :=
sorry

end cost_of_acai_berry_juice_l57_57881


namespace pascal_fifth_number_in_row_15_l57_57556

theorem pascal_fifth_number_in_row_15 : (Nat.choose 15 4) = 1365 := 
by
  sorry

end pascal_fifth_number_in_row_15_l57_57556


namespace multiples_6_8_not_both_l57_57470

theorem multiples_6_8_not_both (n : ℕ) (h : n < 201) : 
  ∃ k : ℕ, (∀ i : ℕ, (i < n → (i % 6 = 0 ∨ i % 8 = 0) ∧ ¬ (i % 24 = 0)) ↔ k = 42) :=
by {
  -- this theorem states that the number of positive integers less than 201 that are multiples 
  -- of either 6 or 8, but not both, is 42.
  sorry
}

end multiples_6_8_not_both_l57_57470


namespace cost_of_items_l57_57185

theorem cost_of_items (x : ℝ) (cost_caramel_apple cost_ice_cream_cone : ℝ) :
  3 * cost_caramel_apple + 4 * cost_ice_cream_cone = 2 ∧
  cost_caramel_apple = cost_ice_cream_cone + 0.25 →
  cost_ice_cream_cone = 0.17857 ∧ cost_caramel_apple = 0.42857 :=
sorry

end cost_of_items_l57_57185


namespace bacteria_growth_relation_l57_57259

variable (w1: ℝ := 10.0) (w2: ℝ := 16.0) (w3: ℝ := 25.6)

theorem bacteria_growth_relation :
  (w2 / w1) = (w3 / w2) :=
by
  sorry

end bacteria_growth_relation_l57_57259


namespace correct_option_d_l57_57990

theorem correct_option_d (a b c : ℝ) (h: a < b ∧ b < 0) : a^2 > ab ∧ ab > b^2 :=
by
  sorry

end correct_option_d_l57_57990


namespace quadrilateral_count_l57_57329

-- Define the number of points
def num_points := 9

-- Define the number of vertices in a quadrilateral
def vertices_in_quadrilateral := 4

-- Use a combination function to find the number of ways to choose 4 points out of 9
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The theorem that asserts the number of quadrilaterals that can be formed
theorem quadrilateral_count : combination num_points vertices_in_quadrilateral = 126 :=
by
  -- The proof would go here
  sorry

end quadrilateral_count_l57_57329


namespace line_slope_intercept_l57_57817

theorem line_slope_intercept (a b: ℝ) (h₁: ∀ x y, (x, y) = (2, 3) ∨ (x, y) = (10, 19) → y = a * x + b)
  (h₂: (a * 6 + b) = 11) : a - b = 3 :=
by
  sorry

end line_slope_intercept_l57_57817


namespace expected_shots_l57_57149

/-
Ivan's initial setup is described as:
- Initial arrows: 14
- Probability of hitting a cone: 0.1
- Number of additional arrows per hit: 3
- Goal: Expected number of shots until Ivan runs out of arrows is 20
-/
noncomputable def probability_hit := 0.1
noncomputable def initial_arrows := 14
noncomputable def additional_arrows_per_hit := 3

theorem expected_shots (n : ℕ) : n = initial_arrows → 
  (probability_hit = 0.1 ∧ additional_arrows_per_hit = 3) →
  E := 20 :=
by
  sorry

end expected_shots_l57_57149


namespace Pascal_triangle_fifth_number_l57_57564

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57564


namespace pascal_triangle_fifth_number_l57_57499

theorem pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  nat.choose n k = 1365 :=
by
  -- conditions
  rw [h_n, h_k]
  -- conclusion
  sorry

end pascal_triangle_fifth_number_l57_57499


namespace range_g_l57_57950

noncomputable def g (x : ℝ) : ℝ := 
  (Real.arccos (x / 3))^2 + Real.pi * Real.arcsin (x / 3) - (Real.arcsin (x / 3))^2 + 
  (Real.pi^2 / 18) * (x^2 + 9 * x + 27)

theorem range_g :
  set.range g = set.Icc (11 * Real.pi^2 / 24) (59 * Real.pi^2 / 24) := by
  sorry

end range_g_l57_57950


namespace A_inter_B_complement_l57_57113

def A : Set ℝ := {x : ℝ | -4 < x^2 - 5*x + 2 ∧ x^2 - 5*x + 2 < 26}
def B : Set ℝ := {x : ℝ | -x^2 + 4*x - 3 < 0}

theorem A_inter_B_complement :
  A ∩ B = {x : ℝ | (-3 < x ∧ x < 1) ∨ (3 < x ∧ x < 8)} ∧
  {x | x ∉ A ∩ B} = {x : ℝ | x ≤ -3 ∨ (1 ≤ x ∧ x ≤ 3) ∨ x ≥ 8 } :=
by
  sorry

end A_inter_B_complement_l57_57113


namespace Pascal_triangle_fifth_number_l57_57557

theorem Pascal_triangle_fifth_number (n k : ℕ) (h_n : n = 15) (h_k : k = 4) :
  Nat.binom 15 4 = 1365 := by
  rw [h_n, h_k]
  sorry

end Pascal_triangle_fifth_number_l57_57557


namespace train_cross_platform_time_l57_57084

noncomputable def kmph_to_mps (s : ℚ) : ℚ :=
  (s * 1000) / 3600

theorem train_cross_platform_time :
  let train_length := 110
  let speed_kmph := 52
  let platform_length := 323.36799999999994
  let speed_mps := kmph_to_mps 52
  let total_distance := train_length + platform_length
  let time := total_distance / speed_mps
  time = 30 := 
by
  sorry

end train_cross_platform_time_l57_57084


namespace pascal_triangle_15_4_l57_57503

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57503


namespace max_arith_seq_20_terms_l57_57849

noncomputable def max_arithmetic_sequences :
  Nat :=
  180

theorem max_arith_seq_20_terms (a : Nat → Nat) :
  (∀ (k : Nat), k ≥ 1 ∧ k ≤ 20 → ∃ d : Nat, a (k + 1) = a k + d) →
  (P : Nat) = max_arithmetic_sequences :=
  by
  -- here's where the proof would go
  sorry

end max_arith_seq_20_terms_l57_57849


namespace find_A_max_min_l57_57254

theorem find_A_max_min :
  ∃ (A_max A_min : ℕ), 
    (A_max = 99999998 ∧ A_min = 17777779) ∧
    (∀ B A, 
      (B > 77777777) ∧
      (Nat.coprime B 36) ∧
      (A = (B % 10) * 10000000 + B / 10) →
      (A ≤ 99999998 ∧ A ≥ 17777779)) :=
by 
  existsi 99999998
  existsi 17777779
  split
  { 
    split 
    { 
      refl 
    }
    refl 
  }
  intros B A h
  sorry

end find_A_max_min_l57_57254


namespace other_carton_racket_count_l57_57024

def num_total_cartons : Nat := 38
def num_total_rackets : Nat := 100
def num_specific_cartons : Nat := 24
def num_rackets_per_specific_carton : Nat := 3

def num_remaining_cartons := num_total_cartons - num_specific_cartons
def num_remaining_rackets := num_total_rackets - (num_specific_cartons * num_rackets_per_specific_carton)

theorem other_carton_racket_count :
  (num_remaining_rackets / num_remaining_cartons) = 2 :=
by
  sorry

end other_carton_racket_count_l57_57024


namespace successive_discounts_final_price_l57_57916

noncomputable def initial_price : ℝ := 10000
noncomputable def discount1 : ℝ := 0.20
noncomputable def discount2 : ℝ := 0.10
noncomputable def discount3 : ℝ := 0.05

theorem successive_discounts_final_price :
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let final_selling_price := price_after_second_discount * (1 - discount3)
  final_selling_price = 6840 := by
  sorry

end successive_discounts_final_price_l57_57916


namespace pascal_triangle_15_4_l57_57504

theorem pascal_triangle_15_4 : nat.choose 15 4 = 1365 :=
by
  sorry

end pascal_triangle_15_4_l57_57504


namespace f_f_five_eq_five_l57_57207

-- Define the function and its properties
noncomputable def f : ℝ → ℝ := sorry

-- Hypotheses
axiom h1 : ∀ x : ℝ, f (x + 2) = -f x
axiom h2 : f 1 = -5

-- Theorem to prove
theorem f_f_five_eq_five : f (f 5) = 5 :=
sorry

end f_f_five_eq_five_l57_57207


namespace infinitely_many_divisible_by_100_l57_57665

open Nat

theorem infinitely_many_divisible_by_100 : ∀ p : ℕ, ∃ n : ℕ, n = 100 * p + 6 ∧ 100 ∣ (2^n + n^2) := by
  sorry

end infinitely_many_divisible_by_100_l57_57665


namespace fifth_number_in_pascals_triangle_l57_57586

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l57_57586
