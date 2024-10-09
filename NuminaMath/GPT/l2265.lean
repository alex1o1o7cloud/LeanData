import Mathlib

namespace max_value_xyz_l2265_226501

theorem max_value_xyz (x y z : ℝ) (h : x + y + 2 * z = 5) : 
  (∃ x y z : ℝ, x + y + 2 * z = 5 ∧ xy + xz + yz = 25/6) :=
sorry

end max_value_xyz_l2265_226501


namespace tenth_pair_in_twentieth_row_l2265_226569

noncomputable def pair_in_row (n k : ℕ) : ℕ × ℕ :=
  if k = 0 ∨ k > n then (0, 0) else (k, n + 1 - k)

theorem tenth_pair_in_twentieth_row : pair_in_row 20 10 = (10, 11) := by
  sorry

end tenth_pair_in_twentieth_row_l2265_226569


namespace boys_playing_both_sports_l2265_226570

theorem boys_playing_both_sports : 
  ∀ (total boys basketball football neither both : ℕ), 
  total = 22 → boys = 22 → basketball = 13 → football = 15 → neither = 3 → 
  boys = basketball + football - both + neither → 
  both = 9 :=
by
  intros total boys basketball football neither both
  intros h_total h_boys h_basketball h_football h_neither h_formula
  sorry

end boys_playing_both_sports_l2265_226570


namespace right_triangle_consecutive_sides_l2265_226561

theorem right_triangle_consecutive_sides (n : ℕ) (n_pos : 0 < n) :
    (n+1)^2 + n^2 = (n+2)^2 ↔ (n = 3) :=
by
  sorry

end right_triangle_consecutive_sides_l2265_226561


namespace a_18_value_l2265_226584

variable (a : ℕ → ℚ)

axiom a1 : a 1 = 1
axiom a2 : a 2 = 2
axiom a_rec (n : ℕ) (hn : 2 ≤ n) : 2 * n * a n = (n - 1) * a (n - 1) + (n + 1) * a (n + 1)

theorem a_18_value : a 18 = 26 / 9 :=
sorry

end a_18_value_l2265_226584


namespace second_smallest_perimeter_l2265_226564

theorem second_smallest_perimeter (a b c : ℕ) (h1 : a + 1 = b) (h2 : b + 1 = c) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) → 
  (a + b + c = 12) :=
by
  sorry

end second_smallest_perimeter_l2265_226564


namespace least_gumballs_to_ensure_five_gumballs_of_same_color_l2265_226568

-- Define the number of gumballs for each color
def red_gumballs := 12
def white_gumballs := 10
def blue_gumballs := 11

-- Define the minimum number of gumballs required to ensure five of the same color
def min_gumballs_to_ensure_five_of_same_color := 13

-- Prove the question == answer given conditions
theorem least_gumballs_to_ensure_five_gumballs_of_same_color :
  (red_gumballs + white_gumballs + blue_gumballs) = 33 → min_gumballs_to_ensure_five_of_same_color = 13 :=
by {
  sorry
}

end least_gumballs_to_ensure_five_gumballs_of_same_color_l2265_226568


namespace geometric_series_common_ratio_l2265_226550

theorem geometric_series_common_ratio (a₁ q : ℝ) (S₃ : ℝ)
  (h1 : S₃ = 7 * a₁)
  (h2 : S₃ = a₁ + a₁ * q + a₁ * q^2) :
  q = 2 ∨ q = -3 :=
by
  sorry

end geometric_series_common_ratio_l2265_226550


namespace expected_value_of_win_is_2_5_l2265_226560

noncomputable def expected_value_of_win : ℚ := 
  (1/6) * (6 - 1) + (1/6) * (6 - 2) + (1/6) * (6 - 3) + 
  (1/6) * (6 - 4) + (1/6) * (6 - 5) + (1/6) * (6 - 6)

theorem expected_value_of_win_is_2_5 : expected_value_of_win = 5 / 2 := 
by
  -- Proof steps will go here
  sorry

end expected_value_of_win_is_2_5_l2265_226560


namespace rate_2nd_and_3rd_hours_equals_10_l2265_226503

-- Define the conditions as given in the problem
def total_gallons_after_5_hours := 34 
def rate_1st_hour := 8 
def rate_4th_hour := 14 
def water_lost_5th_hour := 8 

-- Problem statement: Prove the rate during 2nd and 3rd hours is 10 gallons/hour
theorem rate_2nd_and_3rd_hours_equals_10 (R : ℕ) :
  total_gallons_after_5_hours = rate_1st_hour + 2 * R + rate_4th_hour - water_lost_5th_hour →
  R = 10 :=
by sorry

end rate_2nd_and_3rd_hours_equals_10_l2265_226503


namespace negation_equivalence_l2265_226512

-- Define the original proposition P
def proposition_P : Prop := ∀ x : ℝ, 0 ≤ x → x^3 + 2 * x ≥ 0

-- Define the negation of the proposition P
def negation_P : Prop := ∃ x : ℝ, 0 ≤ x ∧ x^3 + 2 * x < 0

-- The statement to be proven
theorem negation_equivalence : ¬ proposition_P ↔ negation_P := 
by sorry

end negation_equivalence_l2265_226512


namespace problem1_problem2_l2265_226554

-- Problem 1
theorem problem1 : 2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / Real.sqrt 2 = (3 * Real.sqrt 2) / 2 :=
by sorry

-- Problem 2
theorem problem2 : (Real.sqrt 3 - Real.sqrt 2)^2 + (Real.sqrt 8 - Real.sqrt 3) * (2 * Real.sqrt 2 + Real.sqrt 3) = 10 - 2 * Real.sqrt 6 :=
by sorry

end problem1_problem2_l2265_226554


namespace intersection_of_complements_l2265_226580

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

def complement (U A : Set ℕ) : Set ℕ := U \ A

theorem intersection_of_complements :
  U = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} →
  A = {0, 1, 3, 5, 8} →
  B = {2, 4, 5, 6, 8} →
  (complement U A ∩ complement U B) = {7, 9} :=
by
  intros hU hA hB
  sorry

end intersection_of_complements_l2265_226580


namespace simplify_expression_l2265_226524

theorem simplify_expression (x : ℤ) : 
  (12*x^10 + 5*x^9 + 3*x^8) + (2*x^12 + 9*x^10 + 4*x^8 + 6*x^4 + 7*x^2 + 10)
  = 2*x^12 + 21*x^10 + 5*x^9 + 7*x^8 + 6*x^4 + 7*x^2 + 10 :=
by sorry

end simplify_expression_l2265_226524


namespace find_savings_l2265_226513

def income : ℕ := 15000
def expenditure (I : ℕ) : ℕ := 4 * I / 5
def savings (I E : ℕ) : ℕ := I - E

theorem find_savings : savings income (expenditure income) = 3000 := 
by
  sorry

end find_savings_l2265_226513


namespace find_A_l2265_226538

namespace PolynomialDecomposition

theorem find_A (x A B C : ℝ)
  (h : (x^3 + 2 * x^2 - 17 * x - 30)⁻¹ = A / (x - 5) + B / (x + 2) + C / ((x + 2)^2)) :
  A = 1 / 49 :=
by sorry

end PolynomialDecomposition

end find_A_l2265_226538


namespace incorrect_statement_l2265_226573

theorem incorrect_statement (p q : Prop) (hp : ¬ p) (hq : q) : ¬ (¬ q) :=
by
  sorry

end incorrect_statement_l2265_226573


namespace solve_equation_1_solve_equation_2_l2265_226571

theorem solve_equation_1 (x : ℝ) : x * (x + 2) = 2 * (x + 2) ↔ x = -2 ∨ x = 2 := 
by sorry

theorem solve_equation_2 (x : ℝ) : 3 * x^2 - x - 1 = 0 ↔ x = (1 + Real.sqrt 13) / 6 ∨ x = (1 - Real.sqrt 13) / 6 := 
by sorry

end solve_equation_1_solve_equation_2_l2265_226571


namespace outfit_choices_l2265_226511

/-- Given 8 shirts, 8 pairs of pants, and 8 hats, each in 8 colors,
only 6 colors have a matching shirt, pair of pants, and hat.
Each item in the outfit must be of a different color.
Prove that the number of valid outfits is 368. -/
theorem outfit_choices (shirts pants hats colors : ℕ)
  (matching_colors : ℕ)
  (h_shirts : shirts = 8)
  (h_pants : pants = 8)
  (h_hats : hats = 8)
  (h_colors : colors = 8)
  (h_matching_colors : matching_colors = 6) :
  (shirts * pants * hats) - 3 * (matching_colors * colors) = 368 := 
by {
  sorry
}

end outfit_choices_l2265_226511


namespace three_times_x_not_much_different_from_two_l2265_226551

theorem three_times_x_not_much_different_from_two (x : ℝ) :
  3 * x - 2 ≤ -1 := 
sorry

end three_times_x_not_much_different_from_two_l2265_226551


namespace min_distance_from_circle_to_line_l2265_226556

-- Define the circle and line conditions
def is_on_circle (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1
def line (x y : ℝ) : Prop := 4*x + 3*y + 5 = 0

-- The theorem to prove
theorem min_distance_from_circle_to_line (x y : ℝ) (h : is_on_circle x y) : 
  ∃ m_dist : ℝ, m_dist = 2 :=
by
  -- Place holder proof
  sorry

end min_distance_from_circle_to_line_l2265_226556


namespace parallel_lines_l2265_226521

-- Definitions of the lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℝ) (x y : ℝ) : Prop := 2 * x + (5 + m) * y = 8

-- Definition of parallel lines: slopes are equal and the lines are not identical
def slopes_equal (m : ℝ) : Prop := -(3 + m) / 4 = -2 / (5 + m)
def not_identical_lines (m : ℝ) : Prop := l1 m ≠ l2 m

-- Theorem stating the given conditions
theorem parallel_lines (m : ℝ) (x y : ℝ) : slopes_equal m → not_identical_lines m → m = -7 := by
  sorry

end parallel_lines_l2265_226521


namespace billiard_ball_returns_l2265_226527

theorem billiard_ball_returns
  (w h : ℕ)
  (launch_angle : ℝ)
  (reflect_angle : ℝ)
  (start_A : ℝ × ℝ)
  (h_w : w = 2021)
  (h_h : h = 4300)
  (h_launch : launch_angle = 45)
  (h_reflect : reflect_angle = 45)
  (h_in_rect : ∀ (x y : ℝ), 0 ≤ x ∧ x ≤ 2021 ∧ 0 ≤ y ∧ y ≤ 4300) :
  ∃ (bounces : ℕ), bounces = 294 :=
by
  sorry

end billiard_ball_returns_l2265_226527


namespace area_ratio_of_squares_l2265_226535

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * (4 * b) = 4 * a) : (a * a) / (b * b) = 16 :=
by
  sorry

end area_ratio_of_squares_l2265_226535


namespace projection_of_vectors_l2265_226599

variables {a b : ℝ}

noncomputable def vector_projection (a b : ℝ) : ℝ :=
  (a * b) / b^2 * b

theorem projection_of_vectors
  (ha : abs a = 6)
  (hb : abs b = 3)
  (hab : a * b = -12) : vector_projection a b = -4 :=
sorry

end projection_of_vectors_l2265_226599


namespace johns_balance_at_end_of_first_year_l2265_226587

theorem johns_balance_at_end_of_first_year (initial_deposit interest_first_year : ℝ) 
  (h1 : initial_deposit = 5000) 
  (h2 : interest_first_year = 500) :
  initial_deposit + interest_first_year = 5500 :=
by
  rw [h1, h2]
  norm_num

end johns_balance_at_end_of_first_year_l2265_226587


namespace remainder_is_zero_l2265_226591

theorem remainder_is_zero (D R r : ℕ) (h1 : D = 12 * 42 + R)
                           (h2 : D = 21 * 24 + r)
                           (h3 : r < 21) :
                           r = 0 :=
by 
  sorry

end remainder_is_zero_l2265_226591


namespace sequence_difference_constant_l2265_226532

theorem sequence_difference_constant :
  ∀ (x y : ℕ → ℕ), x 1 = 2 → y 1 = 1 →
  (∀ k, k > 1 → x k = 2 * x (k - 1) + 3 * y (k - 1)) →
  (∀ k, k > 1 → y k = x (k - 1) + 2 * y (k - 1)) →
  ∀ k, x k ^ 2 - 3 * y k ^ 2 = 1 :=
by
  -- Insert the proof steps here
  sorry

end sequence_difference_constant_l2265_226532


namespace least_number_l2265_226530

theorem least_number (n : ℕ) (h1 : n % 38 = 1) (h2 : n % 3 = 1) : n = 115 :=
sorry

end least_number_l2265_226530


namespace simplify_expression_l2265_226594

theorem simplify_expression :
  (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 :=
by
  sorry

end simplify_expression_l2265_226594


namespace largest_side_of_triangle_l2265_226577

theorem largest_side_of_triangle (x y Δ c : ℕ)
  (h1 : (x + 2 * Δ / x = y + 2 * Δ / y))
  (h2 : x = 60)
  (h3 : y = 63) :
  c = 87 :=
sorry

end largest_side_of_triangle_l2265_226577


namespace C_is_20_years_younger_l2265_226519

variable (A B C : ℕ)

-- Conditions from the problem
axiom age_condition : A + B = B + C + 20

-- Theorem representing the proof problem
theorem C_is_20_years_younger : A = C + 20 := sorry

end C_is_20_years_younger_l2265_226519


namespace square_in_S_l2265_226502

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

def S (n : ℕ) : Prop :=
  is_sum_of_two_squares (n - 1) ∧ is_sum_of_two_squares n ∧ is_sum_of_two_squares (n + 1)

theorem square_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end square_in_S_l2265_226502


namespace min_distance_PA_l2265_226563

theorem min_distance_PA :
  let A : ℝ × ℝ := (0, 1)
  ∀ (P : ℝ × ℝ), (∃ x : ℝ, x > 0 ∧ P = (x, (x + 2) / x)) →
  ∃ d : ℝ, d = 2 ∧ ∀ Q : ℝ × ℝ, (∃ x : ℝ, x > 0 ∧ Q = (x, (x + 2) / x)) → dist A Q ≥ d :=
by
  sorry

end min_distance_PA_l2265_226563


namespace less_money_than_Bob_l2265_226507

noncomputable def Jennas_money (P: ℝ) : ℝ := 2 * P
noncomputable def Phils_money (B: ℝ) : ℝ := B / 3
noncomputable def Bobs_money : ℝ := 60
noncomputable def Johns_money (P: ℝ) : ℝ := P + 0.35 * P
noncomputable def average (x y: ℝ) : ℝ := (x + y) / 2

theorem less_money_than_Bob :
  ∀ (P Q J B : ℝ),
    P = Phils_money B →
    J = Jennas_money P →
    Q = Johns_money P →
    B = Bobs_money →
    average J Q = B - 0.25 * B →
    B - J = 20
  :=
by
  intros P Q J B hP hJ hQ hB h_avg
  -- Proof goes here
  sorry

end less_money_than_Bob_l2265_226507


namespace annual_donation_amount_l2265_226595

-- Define the conditions
variables (age_start age_end : ℕ)
variables (total_donations : ℕ)

-- Define the question (prove the annual donation amount) given these conditions
theorem annual_donation_amount (h1 : age_start = 13) (h2 : age_end = 33) (h3 : total_donations = 105000) :
  total_donations / (age_end - age_start) = 5250 :=
by
   sorry

end annual_donation_amount_l2265_226595


namespace find_b6_l2265_226526

def fib (b : ℕ → ℕ) : Prop :=
  ∀ n, b (n + 2) = b (n + 1) + b n

theorem find_b6 (b : ℕ → ℕ) (b1 b2 : ℕ)
  (h1 : b 1 = b1) (h2 : b 2 = b2) (h3 : b 5 = 55)
  (hfib : fib b) : b 6 = 84 :=
  sorry

end find_b6_l2265_226526


namespace kiran_money_l2265_226559

theorem kiran_money (R G K : ℕ) (h1: R / G = 6 / 7) (h2: G / K = 6 / 15) (h3: R = 36) : K = 105 := by
  sorry

end kiran_money_l2265_226559


namespace no_number_exists_decreasing_by_removing_digit_l2265_226515

theorem no_number_exists_decreasing_by_removing_digit :
  ¬ ∃ (x y n : ℕ), x * 10^n + y = 58 * y :=
by
  sorry

end no_number_exists_decreasing_by_removing_digit_l2265_226515


namespace monthly_interest_payment_l2265_226593

theorem monthly_interest_payment (P : ℝ) (R : ℝ) (monthly_payment : ℝ)
  (hP : P = 28800) (hR : R = 0.09) : 
  monthly_payment = (P * R) / 12 :=
by
  sorry

end monthly_interest_payment_l2265_226593


namespace find_point_on_curve_l2265_226536

theorem find_point_on_curve :
  ∃ P : ℝ × ℝ, (P.1^3 - P.1 + 3 = P.2) ∧ (3 * P.1^2 - 1 = 2) ∧ (P = (1, 3) ∨ P = (-1, 3)) :=
sorry

end find_point_on_curve_l2265_226536


namespace hunter_movies_count_l2265_226514

theorem hunter_movies_count (H : ℕ) 
  (dalton_movies : ℕ := 7)
  (alex_movies : ℕ := 15)
  (together_movies : ℕ := 2)
  (total_movies : ℕ := 30)
  (all_different_movies : dalton_movies + alex_movies - together_movies + H = total_movies) :
  H = 8 :=
by
  -- The mathematical proof will go here
  sorry

end hunter_movies_count_l2265_226514


namespace find_unique_p_l2265_226585

theorem find_unique_p (p : ℝ) (h1 : p ≠ 0) : (∀ x : ℝ, p * x^2 - 10 * x + 2 = 0 → p = 12.5) :=
by sorry

end find_unique_p_l2265_226585


namespace general_formula_a_sum_bn_l2265_226518

noncomputable section

open Nat

-- Define the sequence Sn
def S (n : ℕ) : ℕ := 2^n + n - 1

-- Define the sequence an
def a (n : ℕ) : ℕ := 1 + 2^(n-1)

-- Define the sequence bn
def b (n : ℕ) : ℕ := 2 * n * (a n - 1)

-- Define the sum Tn
def T (n : ℕ) : ℕ := n * 2^n

-- Proposition 1: General formula for an
theorem general_formula_a (n : ℕ) : a n = 1 + 2^(n-1) :=
by
  sorry

-- Proposition 2: Sum of first n terms of bn
theorem sum_bn (n : ℕ) : T n = 2 + (n - 1) * 2^(n+1) :=
by
  sorry

end general_formula_a_sum_bn_l2265_226518


namespace purely_imaginary_z_point_on_line_z_l2265_226566

-- Proof problem for (I)
theorem purely_imaginary_z (a : ℝ) (z : ℂ) (h : z = Complex.mk 0 (a+2)) 
: a = 2 :=
sorry

-- Proof problem for (II)
theorem point_on_line_z (a : ℝ) (x y : ℝ) (h1 : x = a^2-4) (h2 : y = a+2) (h3 : x + 2*y + 1 = 0) 
: a = -1 :=
sorry

end purely_imaginary_z_point_on_line_z_l2265_226566


namespace miriam_cleaning_room_time_l2265_226537

theorem miriam_cleaning_room_time
  (laundry_time : Nat := 30)
  (bathroom_time : Nat := 15)
  (homework_time : Nat := 40)
  (total_time : Nat := 120) :
  ∃ room_time : Nat, laundry_time + bathroom_time + homework_time + room_time = total_time ∧
                  room_time = 35 := by
  sorry

end miriam_cleaning_room_time_l2265_226537


namespace work_completion_days_l2265_226582

theorem work_completion_days
  (A B : ℝ)
  (h1 : A + B = 1 / 12)
  (h2 : A = 1 / 20)
  : 1 / (A + B / 2) = 15 :=
by 
  sorry

end work_completion_days_l2265_226582


namespace sequence_a_11_l2265_226583

theorem sequence_a_11 (a : ℕ → ℚ) (arithmetic_seq : ℕ → ℚ)
  (h1 : a 3 = 2)
  (h2 : a 7 = 1)
  (h_arith : ∀ n, arithmetic_seq n = 1 / (a n + 1))
  (arith_property : ∀ n, arithmetic_seq (n + 1) - arithmetic_seq n = arithmetic_seq (n + 2) - arithmetic_seq (n + 1)) :
  a 11 = 1 / 2 :=
by
  sorry

end sequence_a_11_l2265_226583


namespace sequence_unbounded_l2265_226528

theorem sequence_unbounded 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a n = |a (n + 1) - a (n + 2)|)
  (h2 : 0 < a 0)
  (h3 : 0 < a 1)
  (h4 : a 0 ≠ a 1) :
  ¬ ∃ M : ℝ, ∀ n, |a n| ≤ M := 
sorry

end sequence_unbounded_l2265_226528


namespace profit_percentage_l2265_226576

theorem profit_percentage (purchase_price sell_price : ℝ) (h1 : purchase_price = 600) (h2 : sell_price = 624) :
  ((sell_price - purchase_price) / purchase_price) * 100 = 4 := by
  sorry

end profit_percentage_l2265_226576


namespace fraction_meaningful_l2265_226596

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ∃ y, y = 1 / (x - 1) :=
by
  sorry

end fraction_meaningful_l2265_226596


namespace average_stamps_per_day_l2265_226542

theorem average_stamps_per_day :
  let a1 := 8
  let d := 8
  let n := 6
  let stamps_collected : Fin n → ℕ := λ i => a1 + i * d
  -- sum the stamps collected over six days
  let S := List.sum (List.ofFn stamps_collected)
  -- calculate average
  let average := S / n
  average = 28 :=
by sorry

end average_stamps_per_day_l2265_226542


namespace sequence_first_term_eq_three_l2265_226565

theorem sequence_first_term_eq_three
  (a : ℕ → ℕ)
  (h_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_nz : ∀ n : ℕ, 0 < a n)
  (h_a11 : a 11 = 157) :
  a 1 = 3 :=
sorry

end sequence_first_term_eq_three_l2265_226565


namespace sky_color_changes_l2265_226590

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end sky_color_changes_l2265_226590


namespace smallest_k_satisfies_l2265_226589

noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

theorem smallest_k_satisfies (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (sqrt (x * y)) + (1 / 2) * (sqrt (abs (x - y))) ≥ (x + y) / 2 :=
by
  sorry

end smallest_k_satisfies_l2265_226589


namespace percentage_of_sikh_boys_l2265_226525

theorem percentage_of_sikh_boys (total_boys muslim_percentage hindu_percentage other_boys : ℕ) 
  (h₁ : total_boys = 300) 
  (h₂ : muslim_percentage = 44) 
  (h₃ : hindu_percentage = 28) 
  (h₄ : other_boys = 54) : 
  (10 : ℝ) = 
  (((total_boys - (muslim_percentage * total_boys / 100 + hindu_percentage * total_boys / 100 + other_boys)) * 100) / total_boys : ℝ) :=
by
  sorry

end percentage_of_sikh_boys_l2265_226525


namespace pyramid_volume_l2265_226581

-- Define the conditions
def height_vertex_to_center_base := 12 -- cm
def side_of_square_base := 10 -- cm
def base_area := side_of_square_base * side_of_square_base -- cm²
def volume := (1 / 3) * base_area * height_vertex_to_center_base -- cm³

-- State the theorem
theorem pyramid_volume : volume = 400 := 
by
  -- Placeholder for the proof
  sorry

end pyramid_volume_l2265_226581


namespace double_given_number_l2265_226540

def given_number : ℝ := 1.2 * 10^6

def double_number (x: ℝ) : ℝ := x * 2

theorem double_given_number : double_number given_number = 2.4 * 10^6 :=
by sorry

end double_given_number_l2265_226540


namespace Carolina_mailed_five_letters_l2265_226505

-- Definitions translating the given conditions into Lean
def cost_of_mail (cost_letters cost_packages : ℝ) (num_letters num_packages : ℕ) : ℝ :=
  cost_letters * num_letters + cost_packages * num_packages

-- The main theorem to prove the desired answer
theorem Carolina_mailed_five_letters (P L : ℕ)
  (h1 : L = P + 2)
  (h2 : cost_of_mail 0.37 0.88 L P = 4.49) :
  L = 5 := 
sorry

end Carolina_mailed_five_letters_l2265_226505


namespace problem_statement_l2265_226541

noncomputable def a : ℚ := 18 / 11
noncomputable def c : ℚ := -30 / 11

theorem problem_statement (a b c : ℚ) (h1 : b / a = 4)
    (h2 : b = 18 - 7 * a) (h3 : c = 2 * a - 6):
    a = 18 / 11 ∧ c = -30 / 11 :=
by
  sorry

end problem_statement_l2265_226541


namespace discount_percentage_l2265_226553

theorem discount_percentage (original_price sale_price : ℕ) (h₁ : original_price = 1200) (h₂ : sale_price = 1020) : 
  ((original_price - sale_price) * 100 / original_price : ℝ) = 15 :=
by
  sorry

end discount_percentage_l2265_226553


namespace abc_is_772_l2265_226509

noncomputable def find_abc (a b c : ℝ) : ℝ :=
if h₁ : 0 < a ∧ 0 < b ∧ 0 < c ∧ a * (b + c) = 160 ∧ b * (c + a) = 168 ∧ c * (a + b) = 180
then 772 else 0

theorem abc_is_772 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
(h₄ : a * (b + c) = 160) (h₅ : b * (c + a) = 168) (h₆ : c * (a + b) = 180) :
  find_abc a b c = 772 := by
  sorry

end abc_is_772_l2265_226509


namespace div_relation_l2265_226555

theorem div_relation (a b d : ℝ) (h1 : a / b = 3) (h2 : b / d = 2 / 5) : d / a = 5 / 6 := by
  sorry

end div_relation_l2265_226555


namespace weight_of_b_l2265_226586

theorem weight_of_b (a b c : ℝ) 
  (h1 : (a + b + c) / 3 = 45) 
  (h2 : (a + b) / 2 = 41) 
  (h3 : (b + c) / 2 = 43) 
  : b = 33 :=
by
  sorry

end weight_of_b_l2265_226586


namespace interpretation_of_k5_3_l2265_226533

theorem interpretation_of_k5_3 (k : ℕ) (hk : 0 < k) : (k^5)^3 = k^5 * k^5 * k^5 :=
by sorry

end interpretation_of_k5_3_l2265_226533


namespace faith_change_l2265_226523

def flour_cost : ℕ := 5
def cake_stand_cost : ℕ := 28
def given_amount : ℕ := 20 * 2 + 3
def total_cost := flour_cost + cake_stand_cost
def change := given_amount - total_cost

theorem faith_change : change = 10 :=
by
  -- the proof goes here
  sorry

end faith_change_l2265_226523


namespace first_bell_weight_l2265_226534

-- Given conditions from the problem
variable (x : ℕ) -- weight of the first bell in pounds
variable (total_weight : ℕ)

-- The condition as the sum of the weights
def bronze_weights (x total_weight : ℕ) : Prop :=
  x + 2 * x + 8 * 2 * x = total_weight

-- Prove that the weight of the first bell is 50 pounds given the total weight is 550 pounds
theorem first_bell_weight : bronze_weights x 550 → x = 50 := by
  intro h
  sorry

end first_bell_weight_l2265_226534


namespace fractional_expression_simplification_l2265_226520

theorem fractional_expression_simplification (x : ℕ) (h : x - 3 < 0) : 
  (2 * x) / (x^2 - 1) - 1 / (x - 1) = 1 / 3 :=
by {
  -- Typical proof steps would go here, adhering to the natural conditions.
  sorry
}

end fractional_expression_simplification_l2265_226520


namespace exactly_three_correct_is_impossible_l2265_226548

theorem exactly_three_correct_is_impossible (n : ℕ) (hn : n = 5) (f : Fin n → Fin n) :
  (∃ S : Finset (Fin n), S.card = 3 ∧ ∀ i ∈ S, f i = i) → False :=
by
  intros h
  sorry

end exactly_three_correct_is_impossible_l2265_226548


namespace factorize_expression_l2265_226557

theorem factorize_expression (a x y : ℤ) : a^2 * (x - y) + 4 * (y - x) = (x - y) * (a + 2) * (a - 2) :=
by
  sorry

end factorize_expression_l2265_226557


namespace evaluation_of_expression_l2265_226552

theorem evaluation_of_expression :
  (3 + 1) * (3^2 + 1^2) * (3^4 + 1^4) = 3280 := by
  sorry

end evaluation_of_expression_l2265_226552


namespace letters_into_mailboxes_l2265_226543

theorem letters_into_mailboxes (n m : ℕ) (h1 : n = 3) (h2 : m = 5) : m^n = 125 :=
by
  rw [h1, h2]
  exact rfl

end letters_into_mailboxes_l2265_226543


namespace find_z_l2265_226592

variable {x y z w : ℝ}

theorem find_z (h : (1/x) + (1/y) = (1/z) + w) : z = (x * y) / (x + y - w * x * y) :=
by sorry

end find_z_l2265_226592


namespace necessary_but_not_sufficient_condition_l2265_226546

variable {a : Nat → Real} -- Sequence a_n
variable {q : Real} -- Common ratio
variable (a1_pos : a 1 > 0) -- Condition a1 > 0

-- Definition of geometric sequence
def is_geometric_sequence (a : Nat → Real) (q : Real) : Prop :=
  ∀ n : Nat, a (n + 1) = a n * q

-- Definition of increasing sequence
def is_increasing_sequence (a : Nat → Real) : Prop :=
  ∀ n : Nat, a n < a (n + 1)

-- Theorem statement
theorem necessary_but_not_sufficient_condition (a : Nat → Real) (q : Real) (a1_pos : a 1 > 0) :
  is_geometric_sequence a q →
  is_increasing_sequence a →
  q > 0 ∧ ¬(q > 0 → is_increasing_sequence a) := by
  sorry

end necessary_but_not_sufficient_condition_l2265_226546


namespace find_number_l2265_226510

theorem find_number (n x : ℝ) (hx : x = 0.8999999999999999) (h : n / x = 0.01) : n = 0.008999999999999999 := by
  sorry

end find_number_l2265_226510


namespace flag_count_l2265_226522

-- Definitions based on the conditions
def colors : ℕ := 3
def stripes : ℕ := 3

-- The main statement
theorem flag_count : colors ^ stripes = 27 :=
by
  sorry

end flag_count_l2265_226522


namespace max_a_plus_b_min_a_squared_plus_b_squared_l2265_226506

theorem max_a_plus_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  a + b ≤ 2 := 
sorry

theorem min_a_squared_plus_b_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^2 + b^2 - a - b + a * b = 1) :
  2 ≤ a^2 + b^2 := 
sorry

end max_a_plus_b_min_a_squared_plus_b_squared_l2265_226506


namespace helen_made_56_pies_l2265_226508

theorem helen_made_56_pies (pinky_pies total_pies : ℕ) (h_pinky : pinky_pies = 147) (h_total : total_pies = 203) :
  (total_pies - pinky_pies) = 56 :=
by
  sorry

end helen_made_56_pies_l2265_226508


namespace is_rectangle_l2265_226544

-- Define the points A, B, C, and D.
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (1, 6)
def C : ℝ × ℝ := (5, 4)
def D : ℝ × ℝ := (2, -2)

-- Define the vectors AB, DC, AD.
def vec (P Q : ℝ × ℝ) : ℝ × ℝ := (Q.1 - P.1, Q.2 - P.2)
def AB := vec A B
def DC := vec D C
def AD := vec A D

-- Function to compute dot product of two vectors.
def dot (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that quadrilateral ABCD is a rectangle.
theorem is_rectangle : AB = DC ∧ dot AB AD = 0 := by
  sorry

end is_rectangle_l2265_226544


namespace tan_sum_pi_over_4_l2265_226579

open Real

theorem tan_sum_pi_over_4 {α : ℝ} (h₁ : cos (2 * α) + sin α * (2 * sin α - 1) = 2 / 5) (h₂ : π / 4 < α) (h₃ : α < π) : 
    tan (α + π / 4) = 1 / 7 := sorry

end tan_sum_pi_over_4_l2265_226579


namespace spell_AMCB_paths_equals_24_l2265_226545

def central_A_reachable_M : Nat := 4
def M_reachable_C : Nat := 2
def C_reachable_B : Nat := 3

theorem spell_AMCB_paths_equals_24 :
  central_A_reachable_M * M_reachable_C * C_reachable_B = 24 := by
  sorry

end spell_AMCB_paths_equals_24_l2265_226545


namespace simplify_complex_div_l2265_226531

theorem simplify_complex_div (a b c d : ℝ) (i : ℂ)
  (h1 : (a = 3) ∧ (b = 5) ∧ (c = -2) ∧ (d = 7) ∧ (i = Complex.I)) :
  ((Complex.mk a b) / (Complex.mk c d) = (Complex.mk (29/53) (-31/53))) :=
by
  sorry

end simplify_complex_div_l2265_226531


namespace tens_digit_of_6_pow_4_is_9_l2265_226575

theorem tens_digit_of_6_pow_4_is_9 : (6 ^ 4 / 10) % 10 = 9 :=
by
  sorry

end tens_digit_of_6_pow_4_is_9_l2265_226575


namespace lily_spent_amount_l2265_226578

def num_years (start_year end_year : ℕ) : ℕ :=
  end_year - start_year

def total_spent (cost_per_plant num_years : ℕ) : ℕ :=
  cost_per_plant * num_years

theorem lily_spent_amount :
  let start_year := 1989
  let end_year := 2021
  let cost_per_plant := 20
  num_years start_year end_year = 32 →
  total_spent cost_per_plant 32 = 640 :=
by
  intros
  sorry

end lily_spent_amount_l2265_226578


namespace find_xy_l2265_226547

theorem find_xy (x y : ℝ) :
  x^2 + y^2 = 2 ∧ (x^2 / (2 - y) + y^2 / (2 - x) = 2) → (x = 1 ∧ y = 1) :=
by
  sorry

end find_xy_l2265_226547


namespace radius_of_larger_ball_l2265_226517

theorem radius_of_larger_ball :
  let V_small : ℝ := (4/3) * Real.pi * (2^3)
  let V_total : ℝ := 6 * V_small
  let R : ℝ := (48:ℝ)^(1/3)
  V_total = (4/3) * Real.pi * (R^3) := 
  by
  sorry

end radius_of_larger_ball_l2265_226517


namespace adela_numbers_l2265_226549

theorem adela_numbers (a b : ℕ) (ha : a > 0) (hb : b > 0) (h : (a - b)^2 = a^2 - b^2 - 4038) :
  (a = 2020 ∧ b = 1) ∨ (a = 2020 ∧ b = 2019) ∨ (a = 676 ∧ b = 3) ∨ (a = 676 ∧ b = 673) :=
sorry

end adela_numbers_l2265_226549


namespace initial_birds_count_l2265_226574

variable (init_birds landed_birds total_birds : ℕ)

theorem initial_birds_count :
  (landed_birds = 8) →
  (total_birds = 20) →
  (init_birds + landed_birds = total_birds) →
  (init_birds = 12) :=
by
  intros h1 h2 h3
  sorry

end initial_birds_count_l2265_226574


namespace tan_inequality_l2265_226516

variable {x : ℝ}

noncomputable def f (x : ℝ) : ℝ := Real.tan x

theorem tan_inequality (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < π / 2) (h3 : 0 < x2) (h4 : x2 < π / 2) (h5 : x1 ≠ x2) :
  (1/2) * (f x1 + f x2) > f ((x1 + x2) / 2) :=
  sorry

end tan_inequality_l2265_226516


namespace total_flowers_l2265_226572

def number_of_flowers (F : ℝ) : Prop :=
  let vases := (F - 7.0) / 6.0
  vases = 6.666666667

theorem total_flowers : number_of_flowers 47.0 :=
by
  sorry

end total_flowers_l2265_226572


namespace area_enclosed_by_graph_l2265_226598

theorem area_enclosed_by_graph (x y : ℝ) (h : abs (5 * x) + abs (3 * y) = 15) : 
  ∃ (area : ℝ), area = 30 :=
sorry

end area_enclosed_by_graph_l2265_226598


namespace probability_of_winning_is_correct_l2265_226567

theorem probability_of_winning_is_correct :
  ∀ (PWin PLoss PTie : ℚ),
    PLoss = 5/12 →
    PTie = 1/6 →
    PWin + PLoss + PTie = 1 →
    PWin = 5/12 := 
by
  intros PWin PLoss PTie hLoss hTie hSum
  sorry

end probability_of_winning_is_correct_l2265_226567


namespace min_value_of_m_l2265_226588

theorem min_value_of_m (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : a + b + c = 3) :
  a^2 + b^2 + c^2 ≥ 3 :=
sorry

end min_value_of_m_l2265_226588


namespace find_difference_of_a_b_l2265_226529

noncomputable def a_b_are_relative_prime_and_positive (a b : ℕ) (hab_prime : Nat.gcd a b = 1) (ha_pos : a > 0) (hb_pos : b > 0) (h_gt : a > b) : Prop :=
  a ^ 3 - b ^ 3 = (131 / 5) * (a - b) ^ 3

theorem find_difference_of_a_b (a b : ℕ) 
  (hab_prime : Nat.gcd a b = 1) 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (h_gt : a > b) 
  (h_eq : (a ^ 3 - b ^ 3 : ℚ) / (a - b) ^ 3 = 131 / 5) : 
  a - b = 7 :=
  sorry

end find_difference_of_a_b_l2265_226529


namespace group_age_analysis_l2265_226562

theorem group_age_analysis (total_members : ℕ) (average_age : ℝ) (zero_age_members : ℕ) 
  (h1 : total_members = 50) (h2 : average_age = 5) (h3 : zero_age_members = 10) :
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  non_zero_members = 40 ∧ non_zero_average_age = 6.25 :=
by
  let total_age := total_members * average_age
  let non_zero_members := total_members - zero_age_members
  let non_zero_average_age := total_age / non_zero_members
  have h_non_zero_members : non_zero_members = 40 := by sorry
  have h_non_zero_average_age : non_zero_average_age = 6.25 := by sorry
  exact ⟨h_non_zero_members, h_non_zero_average_age⟩

end group_age_analysis_l2265_226562


namespace probability_collinear_dots_l2265_226558

theorem probability_collinear_dots 
  (rows : ℕ) (cols : ℕ) (total_dots : ℕ) (collinear_sets : ℕ) (total_ways : ℕ) : 
  rows = 5 → cols = 4 → total_dots = 20 → collinear_sets = 20 → total_ways = 4845 → 
  (collinear_sets : ℚ) / total_ways = 4 / 969 :=
by
  intros hrows hcols htotal_dots hcollinear_sets htotal_ways
  sorry

end probability_collinear_dots_l2265_226558


namespace average_snowfall_dec_1861_l2265_226504

theorem average_snowfall_dec_1861 (snowfall : ℕ) (days_in_dec : ℕ) (hours_in_day : ℕ) 
  (time_period : ℕ) (Avg_inch_per_hour : ℚ) : 
  snowfall = 492 ∧ days_in_dec = 31 ∧ hours_in_day = 24 ∧ time_period = days_in_dec * hours_in_day ∧ 
  Avg_inch_per_hour = snowfall / time_period → 
  Avg_inch_per_hour = 492 / (31 * 24) :=
by sorry

end average_snowfall_dec_1861_l2265_226504


namespace sachin_borrowed_amount_l2265_226597

variable (P : ℝ) (gain : ℝ)
variable (interest_rate_borrow : ℝ := 4 / 100)
variable (interest_rate_lend : ℝ := 25 / 4 / 100)
variable (time_period : ℝ := 2)
variable (gain_provided : ℝ := 112.5)

theorem sachin_borrowed_amount (h : gain = 0.0225 * P) : P = 5000 :=
by sorry

end sachin_borrowed_amount_l2265_226597


namespace wall_length_eq_800_l2265_226500

theorem wall_length_eq_800 
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_width : ℝ) (wall_height : ℝ)
  (num_bricks : ℝ) 
  (brick_volume : ℝ) 
  (total_brick_volume : ℝ)
  (wall_volume : ℝ) :
  brick_length = 25 → 
  brick_width = 11.25 → 
  brick_height = 6 → 
  wall_width = 600 → 
  wall_height = 22.5 → 
  num_bricks = 6400 → 
  brick_volume = brick_length * brick_width * brick_height → 
  total_brick_volume = brick_volume * num_bricks → 
  total_brick_volume = wall_volume →
  wall_volume = (800 : ℝ) * wall_width * wall_height :=
by
  sorry

end wall_length_eq_800_l2265_226500


namespace john_paid_more_l2265_226539

-- Define the required variables
def original_price : ℝ := 84.00000000000009
def discount_rate : ℝ := 0.10
def tip_rate : ℝ := 0.15

-- Define John and Jane's payments
def discounted_price : ℝ := original_price * (1 - discount_rate)
def johns_tip : ℝ := tip_rate * original_price
def johns_total_payment : ℝ := original_price + johns_tip
def janes_tip : ℝ := tip_rate * discounted_price
def janes_total_payment : ℝ := discounted_price + janes_tip

-- Calculate the difference
def payment_difference : ℝ := johns_total_payment - janes_total_payment

-- Statement to prove the payment difference equals $9.66
theorem john_paid_more : payment_difference = 9.66 := by
  sorry

end john_paid_more_l2265_226539
