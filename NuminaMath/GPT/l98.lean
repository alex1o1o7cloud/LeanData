import Mathlib

namespace cube_volume_l98_98087

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98087


namespace triangle_area_formula_l98_98231

theorem triangle_area_formula (a b c R : ℝ) (α β γ : ℝ) 
    (h1 : a / (Real.sin α) = 2 * R) 
    (h2 : b / (Real.sin β) = 2 * R) 
    (h3 : c / (Real.sin γ) = 2 * R) :
    let S := (1 / 2) * a * b * (Real.sin γ)
    S = a * b * c / (4 * R) := 
by 
  sorry

end triangle_area_formula_l98_98231


namespace correct_difference_is_nine_l98_98864

-- Define the conditions
def misunderstood_number : ℕ := 35
def actual_number : ℕ := 53
def incorrect_difference : ℕ := 27

-- Define the two-digit number based on Yoongi's incorrect calculation
def original_number : ℕ := misunderstood_number + incorrect_difference

-- State the theorem
theorem correct_difference_is_nine : (original_number - actual_number) = 9 :=
by
  -- Proof steps go here
  sorry

end correct_difference_is_nine_l98_98864


namespace rationalize_denominator_l98_98818

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98818


namespace greatest_possible_perimeter_l98_98362

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l98_98362


namespace greatest_4_digit_number_divisible_by_15_25_40_75_l98_98445

theorem greatest_4_digit_number_divisible_by_15_25_40_75 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999) ∧ (15 ∣ n) ∧ (25 ∣ n) ∧ (40 ∣ n) ∧ (75 ∣ n) ∧ n = 9600 :=
by
  -- Proof to be provided
  sorry

end greatest_4_digit_number_divisible_by_15_25_40_75_l98_98445


namespace arithmetic_sequence_nth_term_l98_98541

theorem arithmetic_sequence_nth_term (x n : ℕ) (a1 a2 a3 : ℚ) (a_n : ℕ) :
  a1 = 3 * x - 5 ∧ a2 = 7 * x - 17 ∧ a3 = 4 * x + 3 ∧ a_n = 4033 →
  n = 641 :=
by sorry

end arithmetic_sequence_nth_term_l98_98541


namespace minimum_cross_section_area_l98_98670

noncomputable theory

open Real

variable (h : ℝ) (α β : ℝ)
variable (TA TC : Triangle ℝ)
variable (BD : Line ℝ)
variable (base_plane : Plane ℝ)
variable (pyramid_plane : Plane ℝ)

axiom pyramid_definition :
  IsPyramidWithRectBase TA TC BD h α β base_plane pyramid_plane

theorem minimum_cross_section_area :
  α = π / 6 → β = π / 3 →
  MinimumAreaOfCrossSection BD TA TC h α β = h^2 * sqrt 3 / 8 :=
by
  sorry

end minimum_cross_section_area_l98_98670


namespace maximum_additional_voters_l98_98047

-- Define conditions
structure MovieRating (n : ℕ) (x : ℤ) where
  (sum_scores : ℤ) : sum_scores = n * x

-- Define a function to verify the rating decrease condition
def rating_decrease_condition (n : ℕ) (x y : ℤ) : Prop :=
  (n*x + y) / (n+1) = x - 1

-- Problem: To prove that the maximum number of additional voters after moment T is 5
theorem maximum_additional_voters (n additional_voters : ℕ) (x y : ℤ) (initial_condition : MovieRating n x) :
  initial_condition.sum_scores = n * x ∧
  (∀ k, 1 ≤ k → k ≤ additional_voters → 
    ∃ y, rating_decrease_condition (n + k - 1) (x - (k-1)) y ∧ y ≤ 0) →
  additional_voters ≤ 5 :=
by
  sorry

end maximum_additional_voters_l98_98047


namespace lift_ratio_l98_98511

theorem lift_ratio (total_weight first_lift second_lift : ℕ) (h1 : total_weight = 1500)
(h2 : first_lift = 600) (h3 : first_lift = 2 * (second_lift - 300)) : first_lift / second_lift = 1 := 
by
  sorry

end lift_ratio_l98_98511


namespace greatest_triangle_perimeter_l98_98375

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l98_98375


namespace min_value_expression_l98_98617

theorem min_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a * b = 3) :
  ∃ x : ℝ, (x = (a^2 + b^2 + 22) / (a + b)) ∧ (x = 8) :=
by
  sorry

end min_value_expression_l98_98617


namespace rationalize_sqrt_fraction_denom_l98_98798

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l98_98798


namespace sum_w_leq_one_equality_condition_l98_98928

open Finset

noncomputable def w (a b n : ℕ) : ℚ := 1 / Nat.choose n (a + b)

theorem sum_w_leq_one 
  {X : Finset ℕ} (n : ℕ) (A B : Fin n → Finset ℕ) (t : ℕ)
  (hA : ∀ i, i < t → A i ∩ B i = ∅) 
  (hB : ∀ i j, i < t → j < t → i ≠ j → ¬(A i ⊆ A j ∪ B j))
  (ha : ∀ i, i < t → (A i).card = a i) 
  (hb : ∀ i, i < t → (B i).card = b i) :
  ∑ i in range t, w (a i) (b i) n ≤ 1 := sorry

theorem equality_condition
  {X : Finset ℕ} (n : ℕ) (A B : Fin n → Finset ℕ) (t : ℕ) 
  (hA : ∀ i, i < t → A i ∩ B i = ∅)
  (hB : ∀ i j, i < t → j < t → i ≠ j → ¬(A i ⊆ A j ∪ B j)) 
  (hc : ∀ i, i < t → (A i).card = a) 
  (hb : ∀ i, i < t → (B i).card = b) 
  (hb_eq : ∀ i, i < t → B i = B 0) 
  (hA_eq : ∀ i, i < t → A i ⊆ (X \ B 0)) : 
  ∑ i in range t, w (a) (b) n = 1 := sorry

end sum_w_leq_one_equality_condition_l98_98928


namespace solve_for_b_l98_98962

def p (x : ℝ) : ℝ := 2 * x - 5
def q (x : ℝ) (b : ℝ) : ℝ := 3 * x - b

theorem solve_for_b (b : ℝ) : p (q 5 b) = 11 → b = 7 := by
  sorry

end solve_for_b_l98_98962


namespace sqrt_expression_l98_98731

theorem sqrt_expression (x : ℝ) (h : x < 0) : 
  Real.sqrt (x^2 / (1 + (x + 1) / x)) = Real.sqrt (x^3 / (2 * x + 1)) :=
by
  sorry

end sqrt_expression_l98_98731


namespace proportion_equivalence_l98_98922

variable {x y : ℝ}

theorem proportion_equivalence (h : 3 * x = 5 * y) (hy : y ≠ 0) : 
  x / 5 = y / 3 :=
by
  -- Proof goes here
  sorry

end proportion_equivalence_l98_98922


namespace andy_tomatoes_l98_98577

theorem andy_tomatoes (P : ℕ) (h1 : ∀ P, 7 * P / 3 = 42) : P = 18 := by
  sorry

end andy_tomatoes_l98_98577


namespace f_x_plus_f_neg_x_eq_seven_l98_98454

variable (f : ℝ → ℝ)

-- Given conditions: 
axiom cond1 : ∀ x : ℝ, f x + f (1 - x) = 10
axiom cond2 : ∀ x : ℝ, f (1 + x) = 3 + f x

-- Prove statement:
theorem f_x_plus_f_neg_x_eq_seven : ∀ x : ℝ, f x + f (-x) = 7 := 
by
  sorry

end f_x_plus_f_neg_x_eq_seven_l98_98454


namespace smallest_negative_integer_solution_l98_98289

theorem smallest_negative_integer_solution :
  ∃ x : ℤ, 45 * x + 8 ≡ 5 [ZMOD 24] ∧ x = -7 :=
sorry

end smallest_negative_integer_solution_l98_98289


namespace cube_difference_l98_98790

theorem cube_difference (n : ℕ) (h: 0 < n) : (n + 1)^3 - n^3 = 3 * n^2 + 3 * n + 1 := 
sorry

end cube_difference_l98_98790


namespace questions_ratio_l98_98530

theorem questions_ratio (R A : ℕ) (H₁ : R + 6 + A = 24) :
  (R, 6, A) = (R, 6, A) :=
sorry

end questions_ratio_l98_98530


namespace rationalize_sqrt_fraction_l98_98806

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l98_98806


namespace find_pumpkin_seed_packets_l98_98875

variable (P : ℕ)

-- Problem assumptions (conditions)
def pumpkin_seed_cost : ℝ := 2.50
def tomato_seed_cost_total : ℝ := 1.50 * 4
def chili_pepper_seed_cost_total : ℝ := 0.90 * 5
def total_spent : ℝ := 18.00

-- Main theorem to prove
theorem find_pumpkin_seed_packets (P : ℕ) (h : (pumpkin_seed_cost * P) + tomato_seed_cost_total + chili_pepper_seed_cost_total = total_spent) : P = 3 := by sorry

end find_pumpkin_seed_packets_l98_98875


namespace tammy_weekly_distance_l98_98425

-- Define the conditions.
def track_length : ℕ := 50
def loops_per_day : ℕ := 10
def days_in_week : ℕ := 7

-- Using the conditions, prove the total distance per week is 3500 meters.
theorem tammy_weekly_distance : (track_length * loops_per_day * days_in_week) = 3500 := by
  sorry

end tammy_weekly_distance_l98_98425


namespace solution_correct_l98_98536

noncomputable def a := 3 + 3 * Real.sqrt 2
noncomputable def b := 3 - 3 * Real.sqrt 2

theorem solution_correct (h : a ≥ b) : 3 * a + 2 * b = 15 + 3 * Real.sqrt 2 :=
by sorry

end solution_correct_l98_98536


namespace rationalize_sqrt_5_over_12_l98_98826

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l98_98826


namespace range_of_a_l98_98600

def is_ellipse (a : ℝ) : Prop :=
  2 * a > 0 ∧ 3 * a - 6 > 0 ∧ 2 * a < 3 * a - 6

def discriminant_neg (a : ℝ) : Prop :=
  a^2 + 8 * a - 48 < 0

def p (a : ℝ) : Prop := is_ellipse a
def q (a : ℝ) : Prop := discriminant_neg a

theorem range_of_a (a : ℝ) : p a ∧ q a → 2 < a ∧ a < 4 :=
by
  sorry

end range_of_a_l98_98600


namespace zero_of_f_l98_98785

noncomputable def f (x : ℝ) : ℝ := 2^x - 4

theorem zero_of_f : f 2 = 0 :=
by
  sorry

end zero_of_f_l98_98785


namespace number_2120_in_33rd_group_l98_98004

def last_number_in_group (n : ℕ) := 2 * n * (n + 1)

theorem number_2120_in_33rd_group :
  ∃ n, n = 33 ∧ (last_number_in_group (n - 1) < 2120) ∧ (2120 <= last_number_in_group n) :=
sorry

end number_2120_in_33rd_group_l98_98004


namespace events_A_B_equal_prob_l98_98766

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l98_98766


namespace time_to_fill_bottle_l98_98714

-- Definitions
def flow_rate := 500 / 6 -- mL per second
def volume := 250 -- mL

-- Target theorem
theorem time_to_fill_bottle (r : ℝ) (v : ℝ) (t : ℝ) (h : r = flow_rate) (h2 : v = volume) : t = 3 :=
by
  sorry

end time_to_fill_bottle_l98_98714


namespace midpoint_sum_is_correct_l98_98032

theorem midpoint_sum_is_correct:
  let A := (10, 8)
  let B := (-4, -6)
  let midpoint := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (midpoint.1 + midpoint.2) = 4 :=
by
  sorry

end midpoint_sum_is_correct_l98_98032


namespace gcd_79625_51575_l98_98734

theorem gcd_79625_51575 : Nat.gcd 79625 51575 = 25 :=
by
  sorry

end gcd_79625_51575_l98_98734


namespace line_exists_l98_98647

theorem line_exists (x y x' y' : ℝ)
  (h1 : x' = 3 * x + 2 * y + 1)
  (h2 : y' = x + 4 * y - 3) : 
  (∃ A B C : ℝ, A * x + B * y + C = 0 ∧ A * x' + B * y' + C = 0 ∧ 
  ((A = 1 ∧ B = -1 ∧ C = 4) ∨ (A = 4 ∧ B = -8 ∧ C = -5))) :=
sorry

end line_exists_l98_98647


namespace toy_value_l98_98884

theorem toy_value
  (t : ℕ)                 -- total number of toys
  (W : ℕ)                 -- total worth in dollars
  (v : ℕ)                 -- value of one specific toy
  (x : ℕ)                 -- value of one of the other toys
  (h1 : t = 9)            -- condition 1: total number of toys
  (h2 : W = 52)           -- condition 2: total worth
  (h3 : v = 12)           -- condition 3: value of one specific toy
  (h4 : (t - 1) * x + v = W) -- condition 4: equation based on the problem
  : x = 5 :=              -- theorem statement: other toy's value
by {
  -- proof goes here
  sorry
}

end toy_value_l98_98884


namespace greatest_possible_perimeter_l98_98328

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l98_98328


namespace inequality_proof_l98_98601

theorem inequality_proof (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) :
    a * (b^2 + c^2) + b * (c^2 + a^2) ≥ 4 * a * b * c :=
by
  sorry

end inequality_proof_l98_98601


namespace solve_rectangular_field_problem_l98_98264

-- Define the problem
def f (L W : ℝ) := L * W = 80 ∧ 2 * W + L = 28

-- Define the length of the uncovered side
def length_of_uncovered_side (L: ℝ) := L = 20

-- The statement we need to prove
theorem solve_rectangular_field_problem (L W : ℝ) (h : f L W) : length_of_uncovered_side L :=
by
  sorry

end solve_rectangular_field_problem_l98_98264


namespace petya_finishes_earlier_than_masha_l98_98229

variable (t_P t_M t_K : ℕ)

-- Given conditions
def condition1 := t_K = 2 * t_P
def condition2 := t_P + 12 = t_K
def condition3 := t_M = 3 * t_P

-- The proof goal: Petya finishes 24 seconds earlier than Masha
theorem petya_finishes_earlier_than_masha
    (h1 : condition1 t_P t_K)
    (h2 : condition2 t_P t_K)
    (h3 : condition3 t_P t_M) :
    t_M - t_P = 24 := by
  sorry

end petya_finishes_earlier_than_masha_l98_98229


namespace compute_fractions_product_l98_98469

theorem compute_fractions_product :
  (2 * (2^4 - 1) / (2 * (2^4 + 1))) *
  (2 * (3^4 - 1) / (2 * (3^4 + 1))) *
  (2 * (4^4 - 1) / (2 * (4^4 + 1))) *
  (2 * (5^4 - 1) / (2 * (5^4 + 1))) *
  (2 * (6^4 - 1) / (2 * (6^4 + 1))) *
  (2 * (7^4 - 1) / (2 * (7^4 + 1)))
  = 4400 / 135 := by
sorry

end compute_fractions_product_l98_98469


namespace trajectory_of_midpoint_l98_98607

theorem trajectory_of_midpoint (x y : ℝ) (A B : ℝ × ℝ) 
  (hB : B = (4, 0)) (hA_on_circle : (A.1)^2 + (A.2)^2 = 4)
  (hM : ((x, y) = ( (A.1 + B.1)/2, (A.2 + B.2)/2))) :
  (x - 2)^2 + y^2 = 1 :=
sorry

end trajectory_of_midpoint_l98_98607


namespace cube_volume_of_surface_area_l98_98066

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98066


namespace value_of_expression_l98_98450

-- defining the conditions
def in_interval (a : ℝ) : Prop := 1 < a ∧ a < 2

-- defining the algebraic expression
def algebraic_expression (a : ℝ) : ℝ := abs (a - 2) + abs (1 - a)

-- theorem to be proved
theorem value_of_expression (a : ℝ) (h : in_interval a) : algebraic_expression a = 1 :=
by
  -- proof will go here
  sorry

end value_of_expression_l98_98450


namespace no_real_solutions_l98_98283

theorem no_real_solutions :
  ∀ (x : ℝ), (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) ≠ 1 / 8) :=
by
  intro x
  sorry

end no_real_solutions_l98_98283


namespace wheat_pile_weight_l98_98257

noncomputable def weight_of_conical_pile
  (circumference : ℝ) (height : ℝ) (density : ℝ) : ℝ :=
  let r := circumference / (2 * 3.14)
  let volume := (1.0 / 3.0) * 3.14 * r^2 * height
  volume * density

theorem wheat_pile_weight :
  weight_of_conical_pile 12.56 1.2 30 = 150.72 :=
by
  sorry

end wheat_pile_weight_l98_98257


namespace tank_capacity_l98_98498

theorem tank_capacity : 
  ∀ (T : ℚ), (3 / 4) * T + 8 = (7 / 8) * T → T = 64 := by
  intros T h
  have h1 : (7 / 8) * T - (3 / 4) * T = 8 := by linarith
  have h2 : (1 / 8) * T = 8 := by linarith
  have h3 : T = 8 * 8 := by calc
    T = 8 * (8 : ℚ) : by rw [h2, rat.inv_mul_eq_iff (ne_of_eq_of_ne (by norm_num) one_ne_zero)]
  rw h3
  norm_num


end tank_capacity_l98_98498


namespace day_after_2_pow_20_is_friday_l98_98983

-- Define the given conditions
def today_is_monday : ℕ := 0 -- Assuming Monday is represented by 0

-- Define the number of days after \(2^{20}\) days
def days_after : ℕ := 2^20

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Define the function to find the day of the week after a given number of days
def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % days_in_week

-- The theorem to prove
theorem day_after_2_pow_20_is_friday :
  day_of_week today_is_monday days_after = 5 := -- Friday is represented by 5 here
sorry

end day_after_2_pow_20_is_friday_l98_98983


namespace product_of_g_at_roots_l98_98781

noncomputable def f (x : ℝ) : ℝ := x^5 + x^2 + 1
noncomputable def g (x : ℝ) : ℝ := x^2 - 2
noncomputable def roots : List ℝ := sorry -- To indicate the list of roots x_1, x_2, x_3, x_4, x_5 of the polynomial f(x)

theorem product_of_g_at_roots :
  (roots.map g).prod = -23 := sorry

end product_of_g_at_roots_l98_98781


namespace conditional_probability_heads_on_second_given_heads_on_first_l98_98291

open ProbabilityTheory

-- Definitions based on conditions in the problem
def A : Event := {ω | ω.headsOnFirstFlip}
def B : Event := {ω | ω.headsOnSecondFlip}
def P : Measure Space := { measureOfEvent ω | ω = 1 / 2 }

-- Theorem statement
theorem conditional_probability_heads_on_second_given_heads_on_first :
  P(B|A) = 1 / 2 :=
sorry

end conditional_probability_heads_on_second_given_heads_on_first_l98_98291


namespace max_rectangles_in_triangle_l98_98527

theorem max_rectangles_in_triangle : 
  (∃ (n : ℕ), n = 192 ∧ 
  ∀ (i j : ℕ), i + j < 7 → ∀ (a b : ℕ), a ≤ 6 - i ∧ b ≤ 6 - j → 
  ∃ (rectangles : ℕ), rectangles = (6 - i) * (6 - j)) :=
sorry

end max_rectangles_in_triangle_l98_98527


namespace petya_second_race_finishes_first_l98_98659

variable (t v_P v_V : ℝ)
variable (h1 : v_P * t = 100)
variable (h2 : v_V * t = 90)
variable (d : ℝ)

theorem petya_second_race_finishes_first :
  v_V = 0.9 * v_P ∧
  d * v_P = 10 + d * (0.9 * v_P) →
  ∃ t2 : ℝ, t2 = 100 / v_P ∧ (v_V * t2 = 90) →
  ∃ t3 : ℝ, t3 = t2 + d / 10 ∧ (d * v_P = 100) →
  v_P * d / 10 - v_V * d / 10 = 1 :=
by
  sorry

end petya_second_race_finishes_first_l98_98659


namespace frank_bags_on_saturday_l98_98920

def bags_filled_on_saturday (total_cans : Nat) (cans_per_bag : Nat) (bags_on_sunday : Nat) : Nat :=
  total_cans / cans_per_bag - bags_on_sunday

theorem frank_bags_on_saturday : 
  let total_cans := 40
  let cans_per_bag := 5
  let bags_on_sunday := 3
  bags_filled_on_saturday total_cans cans_per_bag bags_on_sunday = 5 :=
  by
  -- Proof to be provided
  sorry

end frank_bags_on_saturday_l98_98920


namespace power_function_analysis_l98_98930

theorem power_function_analysis (f : ℝ → ℝ) (α : ℝ) (h : ∀ x > 0, f x = x ^ α) (h_f : f 9 = 3) :
  (∀ x ≥ 0, f x = x ^ (1 / 2)) ∧
  (∀ x ≥ 4, f x ≥ 2) ∧
  (∀ x1 x2 : ℝ, x2 > x1 ∧ x1 > 0 → (f (x1) + f (x2)) / 2 < f ((x1 + x2) / 2)) :=
by
  -- Solution steps would go here
  sorry

end power_function_analysis_l98_98930


namespace cube_volume_from_surface_area_l98_98055

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98055


namespace problem_solution_l98_98413

-- Define the structure of the dartboard and scoring
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  regions : List (List ℤ) -- List of lists representing scores in the regions

-- Define the probability calculation function
noncomputable def probability_odd_score (d : Dartboard) : ℚ := sorry

-- Define the specific dartboard with given conditions
def revised_dartboard : Dartboard :=
  { inner_radius := 4.5,
    intermediate_radius := 6.75,
    outer_radius := 9,
    regions := [[3, 2, 2], [2, 1, 1], [1, 1, 3]] }

-- The theorem to prove the solution to the problem
theorem problem_solution : probability_odd_score revised_dartboard = 265 / 855 :=
  sorry

end problem_solution_l98_98413


namespace geometric_series_common_ratio_l98_98177

theorem geometric_series_common_ratio (a S r : ℝ)
  (h1 : a = 172)
  (h2 : S = 400)
  (h3 : S = a / (1 - r)) :
  r = 57 / 100 := 
sorry

end geometric_series_common_ratio_l98_98177


namespace total_team_formation_plans_l98_98738

def numberOfWaysToChooseDoctors (m f : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose m (k - 1) * Nat.choose f 1) +
  (Nat.choose m 1 * Nat.choose f (k - 1))

theorem total_team_formation_plans :
  let m := 5
  let f := 4
  let total := 3
  numberOfWaysToChooseDoctors m f total = 70 :=
by
  let m := 5
  let f := 4
  let total := 3
  unfold numberOfWaysToChooseDoctors
  sorry

end total_team_formation_plans_l98_98738


namespace trigonometric_identity_l98_98040

theorem trigonometric_identity (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin A * Real.cos B * Real.cos C + Real.cos A * Real.sin B * Real.cos C + 
  Real.cos A * Real.cos B * Real.sin C = Real.sin A * Real.sin B * Real.sin C :=
by 
  sorry

end trigonometric_identity_l98_98040


namespace prob_at_least_7_consecutive_heads_l98_98568

theorem prob_at_least_7_consecutive_heads : 
  let total_outcomes := 2^10,
      successful_outcomes := 49
  in successful_outcomes / total_outcomes = (49:ℚ) / 1024 :=
by sorry

end prob_at_least_7_consecutive_heads_l98_98568


namespace tan_theta_half_l98_98481

open Real

theorem tan_theta_half (θ : ℝ) 
  (h0 : 0 < θ) 
  (h1 : θ < π / 2) 
  (h2 : ∃ k : ℝ, (sin (2 * θ), cos θ) = k • (cos θ, 1)) : 
  tan θ = 1 / 2 := by 
sorry

end tan_theta_half_l98_98481


namespace correct_statements_l98_98862

-- Define the statements
def statement1 : Prop := ∀ (f : ℝ → ℝ), (det : ℝ → ℝ) → (∀ x, f x = det x)
def statement2 : Prop := ∀ (x y : ℝ), (residual_plot : ℝ × ℝ) → y = snd residual_plot
def statement3 : Prop := ¬(∀ (x y : ℝ), (corr : ℝ × ℝ) → (∀ x y, x = y))
def statement4 : Prop := complex.conj ⟨-1, 1⟩ = ⟨-1, -1⟩

-- Prove that statements 1, 2, and 4 are correct
theorem correct_statements : statement1 ∧ statement2 ∧ statement4 := by
  sorry

end correct_statements_l98_98862


namespace bridge_length_is_correct_l98_98867

noncomputable def length_of_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  let total_distance := speed_ms * time_sec
  total_distance - length_of_train

theorem bridge_length_is_correct :
  length_of_bridge 160 45 30 = 215 := by
  sorry

end bridge_length_is_correct_l98_98867


namespace max_xy_l98_98643

open Real

theorem max_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 6 * x + 8 * y = 72) (h4 : x = 2 * y) : 
  x * y = 25.92 := 
sorry

end max_xy_l98_98643


namespace find_initial_passengers_l98_98043

def initial_passengers_found (P : ℕ) : Prop :=
  let after_first_station := (2 / 3 : ℚ) * P + 280
  let after_second_station := (1 / 2 : ℚ) * after_first_station + 12
  after_second_station = 242

theorem find_initial_passengers :
  ∃ P : ℕ, initial_passengers_found P ∧ P = 270 :=
by
  sorry

end find_initial_passengers_l98_98043


namespace percent_students_own_only_cats_l98_98325

theorem percent_students_own_only_cats (total_students : ℕ) (students_owning_cats : ℕ) (students_owning_dogs : ℕ) (students_owning_both : ℕ) (h_total : total_students = 500) (h_cats : students_owning_cats = 80) (h_dogs : students_owning_dogs = 150) (h_both : students_owning_both = 40) : 
  (students_owning_cats - students_owning_both) * 100 / total_students = 8 := 
by
  sorry

end percent_students_own_only_cats_l98_98325


namespace simplify_fraction_l98_98533

theorem simplify_fraction (a b gcd : ℕ) (h1 : a = 72) (h2 : b = 108) (h3 : gcd = Nat.gcd a b) : (a / gcd) / (b / gcd) = 2 / 3 :=
by
  -- the proof is omitted here
  sorry

end simplify_fraction_l98_98533


namespace max_value_of_expression_l98_98570

theorem max_value_of_expression (x y z : ℝ) (h : x^2 + y^2 = z^2) :
  ∃ t, t = (3 * Real.sqrt 2) / 2 ∧ ∀ u, u = (x + 2 * y) / z → u ≤ t := by
  sorry

end max_value_of_expression_l98_98570


namespace sum_of_products_of_roots_l98_98522

noncomputable def poly : Polynomial ℝ := 5 * Polynomial.X^3 - 10 * Polynomial.X^2 + 17 * Polynomial.X - 7

theorem sum_of_products_of_roots :
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ poly.eval p = 0 ∧ poly.eval q = 0 ∧ poly.eval r = 0) →
  (∃ p q r : ℝ, p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ ((p * q + p * r + q * r) = 17 / 5)) :=
by
  sorry

end sum_of_products_of_roots_l98_98522


namespace probability_equality_l98_98763

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l98_98763


namespace complement_S_union_T_eq_l98_98000

noncomputable def S := {x : ℝ | x > -2}
noncomputable def T := {x : ℝ | x^2 + 3 * x - 4 ≤ 0}
noncomputable def complement_S := {x : ℝ | x ≤ -2}

theorem complement_S_union_T_eq : (complement_S ∪ T) = {x : ℝ | x ≤ 1} := by 
  sorry

end complement_S_union_T_eq_l98_98000


namespace interval_between_births_l98_98439

variables {A1 A2 A3 A4 A5 : ℝ}
variable {x : ℝ}

def ages (A1 A2 A3 A4 A5 : ℝ) := A1 + A2 + A3 + A4 + A5 = 50
def youngest (A1 : ℝ) := A1 = 4
def interval (x : ℝ) := x = 3.4

theorem interval_between_births
  (h_age_sum: ages A1 A2 A3 A4 A5)
  (h_youngest: youngest A1)
  (h_ages: A2 = A1 + x ∧ A3 = A1 + 2 * x ∧ A4 = A1 + 3 * x ∧ A5 = A1 + 4 * x) :
  interval x :=
by {
  sorry
}

end interval_between_births_l98_98439


namespace arithmetic_sequence_50th_term_l98_98575

-- Definitions as per the conditions
def a_1 : ℤ := 48
def d : ℤ := -2
def n : ℕ := 50

-- Statement to prove the 50th term in the series
theorem arithmetic_sequence_50th_term : a_1 + (n - 1) * d = -50 :=
by
  sorry

end arithmetic_sequence_50th_term_l98_98575


namespace cube_volume_of_surface_area_l98_98068

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98068


namespace country_of_second_se_asian_fields_medal_recipient_l98_98296

-- Given conditions as definitions
def is_highest_recognition (award : String) : Prop :=
  award = "Fields Medal"

def fields_medal_freq (years : Nat) : Prop :=
  years = 4 -- Fields Medal is awarded every four years

def second_se_asian_recipient (name : String) : Prop :=
  name = "Ngo Bao Chau"

-- The main theorem to prove
theorem country_of_second_se_asian_fields_medal_recipient :
  ∀ (award : String) (years : Nat) (name : String),
    is_highest_recognition award ∧ fields_medal_freq years ∧ second_se_asian_recipient name →
    (name = "Ngo Bao Chau" → ∃ (country : String), country = "Vietnam") :=
by
  intros award years name h
  sorry

end country_of_second_se_asian_fields_medal_recipient_l98_98296


namespace railway_original_stations_l98_98122

theorem railway_original_stations (m n : ℕ) (hn : n > 1) (h : n * (2 * m - 1 + n) = 58) : m = 14 :=
by
  sorry

end railway_original_stations_l98_98122


namespace evaluate_g_at_4_l98_98757

def g (x : ℕ) : ℕ := 5 * x - 2

theorem evaluate_g_at_4 : g 4 = 18 := by
  sorry

end evaluate_g_at_4_l98_98757


namespace equalize_expenses_l98_98856

variable {x y : ℝ} 

theorem equalize_expenses (h : x > y) : (x + y) / 2 - y = (x - y) / 2 :=
by sorry

end equalize_expenses_l98_98856


namespace greatest_possible_perimeter_l98_98366

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l98_98366


namespace power_function_value_at_9_l98_98199

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_value_at_9 (h : f 2 = Real.sqrt 2) : f 9 = 3 :=
by sorry

end power_function_value_at_9_l98_98199


namespace grade_representation_l98_98616

theorem grade_representation :
  (8, 1) = (8, 1) :=
by
  sorry

end grade_representation_l98_98616


namespace angle_same_terminal_side_l98_98701

theorem angle_same_terminal_side (k : ℤ) : 
  ∃ (θ : ℤ), θ = -324 ∧ 
    ∀ α : ℤ, α = 36 + k * 360 → 
            ( (α % 360 = θ % 360) ∨ (α % 360 + 360 = θ % 360) ∨ (θ % 360 + 360 = α % 360)) :=
by
  sorry

end angle_same_terminal_side_l98_98701


namespace complex_number_properties_l98_98602

theorem complex_number_properties (z : ℂ) (h : z^2 = 3 + 4 * Complex.I) : 
  (z.im = 1 ∨ z.im = -1) ∧ Complex.abs z = Real.sqrt 5 := 
by
  sorry

end complex_number_properties_l98_98602


namespace value_calculation_l98_98852

theorem value_calculation :
  6 * 100000 + 8 * 1000 + 6 * 100 + 7 * 1 = 608607 :=
by
  sorry

end value_calculation_l98_98852


namespace solve_cubic_equation_l98_98009

theorem solve_cubic_equation (x y z : ℤ) (h : x^3 - 3*y^3 - 9*z^3 = 0) : x = 0 ∧ y = 0 ∧ z = 0 :=
by
  sorry

end solve_cubic_equation_l98_98009


namespace vector_scalar_sub_l98_98724

def a : ℝ × ℝ := (3, -9)
def b : ℝ × ℝ := (2, -8)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_scalar_sub:
  scalar1 • a - scalar2 • b = (6, -12) := by
  sorry

end vector_scalar_sub_l98_98724


namespace percentage_men_science_majors_l98_98870

theorem percentage_men_science_majors (total_students : ℕ) (women_science_majors_ratio : ℚ) (nonscience_majors_ratio : ℚ) (men_class_ratio : ℚ) :
  women_science_majors_ratio = 0.2 → 
  nonscience_majors_ratio = 0.6 → 
  men_class_ratio = 0.4 → 
  ∃ men_science_majors_percent : ℚ, men_science_majors_percent = 0.7 :=
by
  intros h_women_science_majors h_nonscience_majors h_men_class
  sorry

end percentage_men_science_majors_l98_98870


namespace hotel_flat_fee_l98_98459

theorem hotel_flat_fee (f n : ℝ) (h1 : f + n = 120) (h2 : f + 6 * n = 330) : f = 78 :=
by
  sorry

end hotel_flat_fee_l98_98459


namespace abe_job_time_l98_98194

theorem abe_job_time (A G C: ℕ) : G = 70 → C = 21 → (1 / G + 1 / A = 1 / C) → A = 30 := by
sorry

end abe_job_time_l98_98194


namespace max_perimeter_l98_98388

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l98_98388


namespace curve_is_ellipse_with_foci_on_y_axis_l98_98921

theorem curve_is_ellipse_with_foci_on_y_axis (α : ℝ) (hα : 0 < α ∧ α < 90) :
  ∃ a b : ℝ, (0 < a) ∧ (0 < b) ∧ (a < b) ∧ 
  (∀ x y : ℝ, x^2 + y^2 * (Real.cos α) = 1 ↔ (x/a)^2 + (y/b)^2 = 1) :=
sorry

end curve_is_ellipse_with_foci_on_y_axis_l98_98921


namespace treasure_in_heaviest_bag_l98_98555

theorem treasure_in_heaviest_bag (A B C D : ℝ) (h1 : A + B < C)
                                        (h2 : A + C = D)
                                        (h3 : A + D > B + C) : D > A ∧ D > B ∧ D > C :=
by 
  sorry

end treasure_in_heaviest_bag_l98_98555


namespace event_probabilities_equal_l98_98761

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l98_98761


namespace find_n_solution_l98_98595

theorem find_n_solution (n : ℚ) (h : (2 / (n+2)) + (4 / (n+2)) + (n / (n+2)) = 4) : n = -2 / 3 := 
by 
  sorry

end find_n_solution_l98_98595


namespace average_marks_l98_98685

variable (M P C : ℤ)

-- Conditions
axiom h1 : M + P = 50
axiom h2 : C = P + 20

-- Theorem statement
theorem average_marks : (M + C) / 2 = 35 := by
  sorry

end average_marks_l98_98685


namespace rationalize_denominator_l98_98819

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98819


namespace tangent_line_curve_l98_98191

theorem tangent_line_curve (x₀ : ℝ) (a : ℝ) :
  (ax₀ + 2 = e^x₀ + 1) ∧ (a = e^x₀) → a = 1 := by
  sorry

end tangent_line_curve_l98_98191


namespace inner_cube_surface_area_l98_98150

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l98_98150


namespace greatest_possible_perimeter_l98_98373

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l98_98373


namespace range_of_a_l98_98677

noncomputable def cubic_function (x : ℝ) : ℝ := x^3 - 3 * x

theorem range_of_a (a : ℝ) : 
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ (cubic_function x1 = a) ∧ (cubic_function x2 = a) ∧ (cubic_function x3 = a)) ↔ (-2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l98_98677


namespace distance_from_point_to_line_l98_98310

-- Definition of the conditions
def point := (3, 0)
def line_y := 1

-- Problem statement: Prove that the distance between the point (3,0) and the line y=1 is 1.
theorem distance_from_point_to_line (point : ℝ × ℝ) (line_y : ℝ) : abs (point.snd - line_y) = 1 :=
by
  -- insert proof here
  sorry

end distance_from_point_to_line_l98_98310


namespace inner_cube_surface_area_l98_98154

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l98_98154


namespace rationalize_sqrt_fraction_l98_98837

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l98_98837


namespace trips_needed_to_fill_pool_l98_98898

def caleb_gallons_per_trip : ℕ := 7
def cynthia_gallons_per_trip : ℕ := 8
def pool_capacity : ℕ := 105

theorem trips_needed_to_fill_pool : (pool_capacity / (caleb_gallons_per_trip + cynthia_gallons_per_trip) = 7) :=
by
  sorry

end trips_needed_to_fill_pool_l98_98898


namespace inequality_bounds_l98_98927

noncomputable def f (a b A B : ℝ) (θ : ℝ) : ℝ :=
  1 - a * Real.cos θ - b * Real.sin θ - A * Real.cos (2 * θ) - B * Real.sin (2 * θ)

theorem inequality_bounds (a b A B : ℝ) (h : ∀ θ : ℝ, f a b A B θ ≥ 0) : 
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 :=
sorry

end inequality_bounds_l98_98927


namespace no_such_polynomial_exists_l98_98515

open Polynomial

noncomputable def example_polynomial : Type := {P : Polynomial ℤ // P.eval 2 = 4 ∧ P.eval (P.eval 2) = 7}

theorem no_such_polynomial_exists : ¬ ∃ P : example_polynomial, true :=
by
  sorry

end no_such_polynomial_exists_l98_98515


namespace george_speed_second_segment_l98_98431

theorem george_speed_second_segment 
  (distance_total : ℝ)
  (speed_normal : ℝ)
  (distance_first : ℝ)
  (speed_first : ℝ) : 
  distance_total = 1 ∧ 
  speed_normal = 3 ∧ 
  distance_first = 0.5 ∧ 
  speed_first = 2 →
  (distance_first / speed_first + 0.5 * speed_second = 1 / speed_normal → speed_second = 6) :=
sorry

end george_speed_second_segment_l98_98431


namespace max_triangle_perimeter_l98_98347

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l98_98347


namespace cube_volume_l98_98111

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98111


namespace second_remainder_l98_98190

theorem second_remainder (n : ℕ) : n = 210 ∧ n % 13 = 3 → n % 17 = 6 :=
by
  sorry

end second_remainder_l98_98190


namespace distance_to_focus_2_l98_98563

-- Definition of the ellipse and the given distance to one focus
def ellipse (P : ℝ × ℝ) : Prop := (P.1^2)/25 + (P.2^2)/16 = 1
def distance_to_focus_1 (P : ℝ × ℝ) : Prop := dist P (5, 0) = 3

-- Proof problem statement
theorem distance_to_focus_2 (P : ℝ × ℝ) (h₁ : ellipse P) (h₂ : distance_to_focus_1 P) :
  dist P (-5, 0) = 7 :=
sorry

end distance_to_focus_2_l98_98563


namespace find_a7_a8_l98_98394

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (r : ℝ) :=
∀ n, a (n + 1) = r * a n

theorem find_a7_a8
  (a : ℕ → ℝ)
  (r : ℝ)
  (hs : geometric_sequence_property a r)
  (h1 : a 1 + a 2 = 40)
  (h2 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end find_a7_a8_l98_98394


namespace distinct_bead_arrangements_l98_98635

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n-1)

theorem distinct_bead_arrangements : factorial 8 / (8 * 2) = 2520 := 
  by sorry

end distinct_bead_arrangements_l98_98635


namespace min_fraction_sum_is_15_l98_98649

theorem min_fraction_sum_is_15
  (A B C D : ℕ)
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_digits : A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10)
  (h_nonzero_int : ∃ k : ℤ, k ≠ 0 ∧ (A + B : ℤ) = k * (C + D))
  : C + D = 15 :=
sorry

end min_fraction_sum_is_15_l98_98649


namespace greatest_possible_perimeter_l98_98383

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l98_98383


namespace probability_of_rain_l98_98421

variable (P_R P_B0 : ℝ)
variable (H1 : 0 ≤ P_R ∧ P_R ≤ 1)
variable (H2 : 0 ≤ P_B0 ∧ P_B0 ≤ 1)
variable (H : P_R + P_B0 - P_R * P_B0 = 0.2)

theorem probability_of_rain : 
  P_R = 1/9 :=
by
  sorry

end probability_of_rain_l98_98421


namespace value_of_each_other_toy_l98_98885

-- Definitions for the conditions
def total_toys : ℕ := 9
def total_worth : ℕ := 52
def single_toy_value : ℕ := 12

-- Definition to represent the value of each of the other toys
def other_toys_value (same_value : ℕ) : Prop :=
  (total_worth - single_toy_value) / (total_toys - 1) = same_value

-- The theorem to be proven
theorem value_of_each_other_toy : other_toys_value 5 :=
  sorry

end value_of_each_other_toy_l98_98885


namespace inner_cube_surface_area_l98_98138

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l98_98138


namespace sum_of_three_positive_integers_l98_98772

theorem sum_of_three_positive_integers (n : ℕ) (h : n ≥ 3) :
  ∃ k : ℕ, k = (n - 1) * (n - 2) / 2 := 
sorry

end sum_of_three_positive_integers_l98_98772


namespace tetrahedron_face_inequality_l98_98014

theorem tetrahedron_face_inequality
    (A B C D : ℝ) :
    |A^2 + B^2 - C^2 - D^2| ≤ 2 * (A * B + C * D) := by
  sorry

end tetrahedron_face_inequality_l98_98014


namespace find_smaller_number_l98_98735

theorem find_smaller_number (x : ℕ) (hx : x + 4 * x = 45) : x = 9 :=
by
  sorry

end find_smaller_number_l98_98735


namespace smallest_positive_integer_l98_98262

theorem smallest_positive_integer (n : ℕ) (h₁ : n > 1) (h₂ : n % 2 = 1) (h₃ : n % 3 = 1) (h₄ : n % 4 = 1) (h₅ : n % 5 = 1) : n = 61 :=
by
  sorry

end smallest_positive_integer_l98_98262


namespace mike_laptop_row_division_impossible_l98_98946

theorem mike_laptop_row_division_impossible (total_laptops : ℕ) (num_rows : ℕ) 
(types_ratios : List ℕ)
(H_total : total_laptops = 44)
(H_rows : num_rows = 5) 
(H_ratio : types_ratios = [2, 3, 4]) :
  ¬ (∃ (n : ℕ), (total_laptops = n * num_rows) 
  ∧ (n % (types_ratios.sum) = 0)
  ∧ (∀ (t : ℕ), t ∈ types_ratios → t ≤ n)) := sorry

end mike_laptop_row_division_impossible_l98_98946


namespace second_die_sides_l98_98637

theorem second_die_sides (p : ℚ) (n : ℕ) (h1 : p = 0.023809523809523808) (h2 : n ≠ 0) :
  let first_die_sides := 6
  let probability := (1 : ℚ) / first_die_sides * (1 : ℚ) / n
  probability = p → n = 7 :=
by
  intro h
  sorry

end second_die_sides_l98_98637


namespace avg_age_adults_l98_98769

-- Given conditions
def num_members : ℕ := 50
def avg_age_members : ℕ := 20
def num_girls : ℕ := 25
def num_boys : ℕ := 20
def num_adults : ℕ := 5
def avg_age_girls : ℕ := 18
def avg_age_boys : ℕ := 22

-- Prove that the average age of the adults is 22 years
theorem avg_age_adults :
  (num_members * avg_age_members - num_girls * avg_age_girls - num_boys * avg_age_boys) / num_adults = 22 :=
by 
  sorry

end avg_age_adults_l98_98769


namespace greatest_k_for_quadratic_roots_diff_l98_98549

theorem greatest_k_for_quadratic_roots_diff (k : ℝ)
  (H : ∀ x: ℝ, (x^2 + k * x + 8 = 0) → (∃ a b : ℝ, a ≠ b ∧ (a - b)^2 = 84)) :
  k = 2 * Real.sqrt 29 :=
by
  sorry

end greatest_k_for_quadratic_roots_diff_l98_98549


namespace greatest_possible_perimeter_l98_98360

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l98_98360


namespace Elina_garden_area_l98_98943

theorem Elina_garden_area :
  ∀ (L W: ℝ),
    (30 * L = 1500) →
    (12 * (2 * (L + W)) = 1500) →
    (L * W = 625) :=
by
  intros L W h1 h2
  sorry

end Elina_garden_area_l98_98943


namespace rationalize_denominator_l98_98813

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98813


namespace cube_volume_l98_98105

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98105


namespace beaver_stores_60_carrots_l98_98736

theorem beaver_stores_60_carrots (b r : ℕ) (h1 : 4 * b = 5 * r) (h2 : b = r + 3) : 4 * b = 60 :=
by
  sorry

end beaver_stores_60_carrots_l98_98736


namespace domain_of_f_l98_98910

noncomputable def f (x : ℝ) : ℝ := (x + 3) / (x^2 - 5 * x + 6)

theorem domain_of_f :
  {x : ℝ | x^2 - 5 * x + 6 ≠ 0} = {x : ℝ | x < 2} ∪ {x : ℝ | 2 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l98_98910


namespace max_ab_value_l98_98209

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * a * x - b^2 + 12 ≤ 0 → x = a) : ab = 6 := by
  sorry

end max_ab_value_l98_98209


namespace problem1_problem2_l98_98780

def p (x a : ℝ) : Prop := x^2 + 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6*x - 72 <= 0) ∧ (x^2 + x - 6 > 0)

theorem problem1 (x : ℝ) (a : ℝ) (h : a = -1): (p x a ∨ q x) → (-6 ≤ x ∧ x < -3) ∨ (1 < x ∧ x ≤ 12) :=
sorry

theorem problem2 (a : ℝ): (¬ ∃ x : ℝ, p x a) → (¬ ∃ x : ℝ, q x) → (-4 ≤ a ∧ a ≤ -2) :=
sorry

end problem1_problem2_l98_98780


namespace rationalize_sqrt_fraction_denom_l98_98801

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l98_98801


namespace parallel_line_segment_length_l98_98676

theorem parallel_line_segment_length (AB : ℝ) (S : ℝ) (x : ℝ) 
  (h1 : AB = 36) 
  (h2 : S = (S / 2) * 2)
  (h3 : x / AB = (↑(1 : ℝ) / 2 * S / S) ^ (1 / 2)) : 
  x = 18 * Real.sqrt 2 :=
by 
    sorry 

end parallel_line_segment_length_l98_98676


namespace point_transformations_l98_98546

theorem point_transformations (a b : ℝ) (h : (a ≠ 2 ∨ b ≠ 3))
  (H1 : ∃ x y : ℝ, (x, y) = (2 - (b - 3), 3 + (a - 2)) ∧ (y, x) = (-4, 2)) :
  b - a = -6 :=
by
  sorry

end point_transformations_l98_98546


namespace inner_cube_surface_area_l98_98139

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l98_98139


namespace sunday_to_saturday_ratio_l98_98013

theorem sunday_to_saturday_ratio : 
  ∀ (sold_friday sold_saturday sold_sunday total_sold : ℕ),
  sold_friday = 40 →
  sold_saturday = (2 * sold_friday - 10) →
  total_sold = 145 →
  total_sold = sold_friday + sold_saturday + sold_sunday →
  (sold_sunday : ℚ) / (sold_saturday : ℚ) = 1 / 2 :=
by
  intro sold_friday sold_saturday sold_sunday total_sold
  intros h_friday h_saturday h_total h_sum
  sorry

end sunday_to_saturday_ratio_l98_98013


namespace pyramid_volume_l98_98428

theorem pyramid_volume
  (a b c : ℝ) (h1 : a = 6) (h2 : b = 5) (h3 : c = 5)
  (angle_lateral : ℝ) (h4 : angle_lateral = 45) :
  ∃ (V : ℝ), V = 6 :=
by
  -- the proof steps would be included here
  sorry

end pyramid_volume_l98_98428


namespace factory_produces_6400_toys_per_week_l98_98703

-- Definition of worker productivity per day
def toys_per_day : ℝ := 2133.3333333333335

-- Definition of workdays per week
def workdays_per_week : ℕ := 3

-- Definition of total toys produced per week
def toys_per_week : ℝ := toys_per_day * workdays_per_week

-- Theorem stating the total number of toys produced per week
theorem factory_produces_6400_toys_per_week : toys_per_week = 6400 :=
by
  sorry

end factory_produces_6400_toys_per_week_l98_98703


namespace greatest_possible_perimeter_l98_98358

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l98_98358


namespace frog_landing_safely_l98_98248

theorem frog_landing_safely :
  let m := 1
  let n := 8
  100 * m + n = 108 :=
begin
  sorry
end

end frog_landing_safely_l98_98248


namespace john_total_cost_l98_98566

-- Define the costs and usage details
def base_cost : ℕ := 25
def cost_per_text_cent : ℕ := 10
def cost_per_extra_minute_cent : ℕ := 15
def included_hours : ℕ := 20
def texts_sent : ℕ := 150
def hours_talked : ℕ := 22

-- Prove that the total cost John had to pay is $58
def total_cost_john : ℕ :=
  let base_cost_dollars := base_cost
  let text_cost_dollars := (texts_sent * cost_per_text_cent) / 100
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_minutes_cost_dollars := (extra_minutes * cost_per_extra_minute_cent) / 100
  base_cost_dollars + text_cost_dollars + extra_minutes_cost_dollars

theorem john_total_cost (h1 : base_cost = 25)
                        (h2 : cost_per_text_cent = 10)
                        (h3 : cost_per_extra_minute_cent = 15)
                        (h4 : included_hours = 20)
                        (h5 : texts_sent = 150)
                        (h6 : hours_talked = 22) : 
  total_cost_john = 58 := by
  sorry

end john_total_cost_l98_98566


namespace original_percentage_alcohol_l98_98256

-- Definitions of the conditions
def original_mixture_volume : ℝ := 15
def additional_water_volume : ℝ := 3
def final_percentage_alcohol : ℝ := 20.833333333333336
def final_mixture_volume : ℝ := original_mixture_volume + additional_water_volume

-- Lean statement to prove
theorem original_percentage_alcohol (A : ℝ) :
  (A / 100 * original_mixture_volume) = (final_percentage_alcohol / 100 * final_mixture_volume) →
  A = 25 :=
by
  sorry

end original_percentage_alcohol_l98_98256


namespace paula_travel_fraction_l98_98651

theorem paula_travel_fraction :
  ∀ (f : ℚ), 
    (∀ (L_time P_time travel_total : ℚ), 
      L_time = 70 →
      P_time = 70 * f →
      travel_total = 504 →
      (L_time + 5 * L_time + P_time + P_time = travel_total) →
      f = 3/5) :=
by
  sorry

end paula_travel_fraction_l98_98651


namespace cabinets_and_perimeter_l98_98777

theorem cabinets_and_perimeter :
  ∀ (original_cabinets : ℕ) (install_factor : ℕ) (num_counters : ℕ) 
    (cabinets_L_1 cabinets_L_2 cabinets_L_3 removed_cabinets cabinet_height total_cabinets perimeter : ℕ),
    original_cabinets = 3 →
    install_factor = 2 →
    num_counters = 4 →
    cabinets_L_1 = 3 →
    cabinets_L_2 = 5 →
    cabinets_L_3 = 7 →
    removed_cabinets = 2 →
    cabinet_height = 2 →
    total_cabinets = (original_cabinets * install_factor * num_counters) + 
                     (cabinets_L_1 + cabinets_L_2 + cabinets_L_3) - removed_cabinets →
    perimeter = (cabinets_L_1 * cabinet_height) +
                (cabinets_L_3 * cabinet_height) +
                2 * (cabinets_L_2 * cabinet_height) →
    total_cabinets = 37 ∧
    perimeter = 40 :=
by
  intros
  sorry

end cabinets_and_perimeter_l98_98777


namespace rationalize_sqrt_fraction_l98_98831

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l98_98831


namespace probability_all_white_l98_98871

-- Defined conditions
def total_balls := 15
def white_balls := 8
def black_balls := 7
def balls_drawn := 5

-- Lean theorem statement
theorem probability_all_white :
  (nat.choose white_balls balls_drawn) / (nat.choose total_balls balls_drawn) = (56 : ℚ) / 3003 :=
by
  sorry

end probability_all_white_l98_98871


namespace cube_volume_l98_98074

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98074


namespace sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l98_98261

theorem sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees :
  ∃ (n : ℕ), (n * (n - 3) / 2 = 14) → ((n - 2) * 180 = 900) :=
by
  sorry

end sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l98_98261


namespace greatest_possible_perimeter_l98_98365

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l98_98365


namespace infinitely_many_primes_congruent_3_mod_4_l98_98840

def is_congruent_3_mod_4 (p : ℕ) : Prop :=
  p % 4 = 3

def is_prime (p : ℕ) : Prop :=
  Nat.Prime p

def S (p : ℕ) : Prop :=
  is_prime p ∧ is_congruent_3_mod_4 p

theorem infinitely_many_primes_congruent_3_mod_4 :
  ∀ n : ℕ, ∃ p : ℕ, p > n ∧ S p :=
sorry

end infinitely_many_primes_congruent_3_mod_4_l98_98840


namespace soccer_tournament_eq_l98_98213

theorem soccer_tournament_eq (x : ℕ) (h : (x * (x - 1)) / 2 = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  sorry

end soccer_tournament_eq_l98_98213


namespace solution_pairs_l98_98187

theorem solution_pairs (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  x ^ 2 + y ^ 2 - 5 * x * y + 5 = 0 ↔ (x = 3 ∧ y = 1) ∨ (x = 2 ∧ y = 1) ∨ (x = 9 ∧ y = 2) ∨ (x = 1 ∧ y = 2) := by
  sorry

end solution_pairs_l98_98187


namespace remainder_p_q_add_42_l98_98225

def p (k : ℤ) : ℤ := 98 * k + 84
def q (m : ℤ) : ℤ := 126 * m + 117

theorem remainder_p_q_add_42 (k m : ℤ) : 
  (p k + q m) % 42 = 33 := by
  sorry

end remainder_p_q_add_42_l98_98225


namespace fifth_segment_student_l98_98692

variable (N : ℕ) (n : ℕ) (second_segment_student : ℕ)

def sampling_interval (N n : ℕ) : ℕ := N / n

def initial_student (second_segment_student interval : ℕ) : ℕ := second_segment_student - interval

def student_number (initial_student interval : ℕ) (segment : ℕ) : ℕ :=
  initial_student + (segment - 1) * interval

theorem fifth_segment_student (N n : ℕ) (second_segment_student : ℕ) (hN : N = 700) (hn : n = 50) (hsecond : second_segment_student = 20) :
  student_number (initial_student second_segment_student (sampling_interval N n)) (sampling_interval N n) 5 = 62 := by
  sorry

end fifth_segment_student_l98_98692


namespace inequality_solution_l98_98476

theorem inequality_solution (m : ℝ) (x : ℝ) (hm : 0 ≤ m ∧ m ≤ 1) (ineq : m * x^2 - 2 * x - m ≥ 2) : x ≤ -1 :=
sorry

end inequality_solution_l98_98476


namespace abcdefg_defghij_value_l98_98206

variable (a b c d e f g h i : ℚ)

theorem abcdefg_defghij_value :
  (a / b = -7 / 3) →
  (b / c = -5 / 2) →
  (c / d = 2) →
  (d / e = -3 / 2) →
  (e / f = 4 / 3) →
  (f / g = -1 / 4) →
  (g / h = 3 / -5) →
  (abcdefg / defghij = (-21 / 16) * (c / i)) :=
by
  sorry

end abcdefg_defghij_value_l98_98206


namespace encoded_base5_to_base10_l98_98054

-- Given definitions
def base5_to_int (d1 d2 d3 : ℕ) : ℕ := d1 * 25 + d2 * 5 + d3

def V := 2
def W := 0
def X := 4
def Y := 1
def Z := 3

-- Prove that the base-10 expression for the integer coded as XYZ is 108
theorem encoded_base5_to_base10 :
  base5_to_int X Y Z = 108 :=
sorry

end encoded_base5_to_base10_l98_98054


namespace cube_volume_l98_98075

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98075


namespace simplify_expression_l98_98668

variable (b : ℝ)

theorem simplify_expression (b : ℝ) : (3 * b - 3 - 5 * b) / 3 = - (2 / 3) * b - 1 :=
by
  sorry

end simplify_expression_l98_98668


namespace FC_value_l98_98482

theorem FC_value (DC CB AD FC : ℝ) (h1 : DC = 10) (h2 : CB = 9)
  (h3 : AB = (1 / 3) * AD) (h4 : ED = (3 / 4) * AD) : FC = 13.875 := by
  sorry

end FC_value_l98_98482


namespace maximum_value_2a_plus_b_l98_98926

variable (a b : ℝ)

theorem maximum_value_2a_plus_b (h : 4 * a^2 + b^2 + a * b = 1) : 2 * a + b ≤ 2 * Real.sqrt (10) / 5 :=
by sorry

end maximum_value_2a_plus_b_l98_98926


namespace remainder_369963_div_6_is_3_l98_98989

def is_divisible_by (a b : ℕ) : Prop := b ∣ a

def remainder_when_divided (a b : ℕ) (r : ℕ) : Prop := a % b = r

theorem remainder_369963_div_6_is_3 :
  remainder_when_divided 369963 6 3 :=
by
  have h₁ : 369963 % 2 = 1 := by
    sorry -- It is known that 369963 is not divisible by 2.
  have h₂ : 369963 % 3 = 0 := by
    sorry -- It is known that 369963 is divisible by 3.
  have h₃ : 369963 % 6 = 3 := by
    sorry -- From the above properties, derive that the remainder when 369963 is divided by 6 is 3.
  exact h₃

end remainder_369963_div_6_is_3_l98_98989


namespace sin_square_alpha_minus_pi_div_4_l98_98746

theorem sin_square_alpha_minus_pi_div_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α - Real.pi / 4) ^ 2 = 1 / 6 := 
sorry

end sin_square_alpha_minus_pi_div_4_l98_98746


namespace max_value_expression_l98_98286

theorem max_value_expression : ∃ (max_val : ℝ), max_val = (1 / 16) ∧ ∀ a b : ℝ, 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 → (a - b^2) * (b - a^2) ≤ max_val :=
by
  sorry

end max_value_expression_l98_98286


namespace inner_cube_surface_area_l98_98136

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l98_98136


namespace find_number_subtracted_l98_98455

-- Given a number x, where the ratio of the two natural numbers is 6:5,
-- and another number y is subtracted to both numbers such that the new ratio becomes 5:4,
-- and the larger number exceeds the smaller number by 5,
-- prove that y = 5.
theorem find_number_subtracted (x y : ℕ) (h1 : 6 * x - 5 * x = 5) (h2 : (30 - y) * 4 = (25 - y) * 5) : y = 5 :=
by sorry

end find_number_subtracted_l98_98455


namespace sum_of_digits_M_l98_98218

-- Definitions
def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Conditions
variables (M : ℕ)
  (h1 : M % 2 = 0)  -- M is even
  (h2 : ∀ d ∈ M.digits 10, d = 0 ∨ d = 2 ∨ d = 4 ∨ d = 5 ∨ d = 7 ∨ d = 9)  -- Digits of M
  (h3 : sum_of_digits (2 * M) = 31)  -- Sum of digits of 2M
  (h4 : sum_of_digits (M / 2) = 28)  -- Sum of digits of M/2

-- Goal
theorem sum_of_digits_M :
  sum_of_digits M = 29 :=
sorry

end sum_of_digits_M_l98_98218


namespace valve_XY_time_correct_l98_98474

-- Given conditions
def valve_rates (x y z : ℝ) := (x + y + z = 1/2 ∧ x + z = 1/4 ∧ y + z = 1/3)
def total_fill_time (t : ℝ) (x y : ℝ) := t = 1 / (x + y)

-- The proof problem
theorem valve_XY_time_correct (x y z : ℝ) (t : ℝ) 
  (h : valve_rates x y z) : total_fill_time t x y → t = 2.4 :=
by
  -- Assume h defines the rates
  have h1 : x + y + z = 1/2 := h.1
  have h2 : x + z = 1/4 := h.2.1
  have h3 : y + z = 1/3 := h.2.2
  
  sorry

end valve_XY_time_correct_l98_98474


namespace quadratic_roots_identity_l98_98520

variable (α β : ℝ)
variable (h1 : α^2 + 3*α - 7 = 0)
variable (h2 : β^2 + 3*β - 7 = 0)

-- The problem is to prove that α^2 + 4*α + β = 4
theorem quadratic_roots_identity :
  α^2 + 4*α + β = 4 :=
sorry

end quadratic_roots_identity_l98_98520


namespace A1_and_A2_complement_independent_l98_98324

open MeasureTheory

-- Hypothetical events A1 and A2
variable (Ω : Type) [MeasurableSpace Ω]
variable (P : ProbabilityMeasure Ω)

-- Event A1: drawing a black ball the first time
variable (A1 : Event Ω)

-- Event A2: drawing a black ball the second time
variable (A2 : Event Ω)

-- Event A2 complement: not drawing a black ball the second time
def A2_complement : Event Ω := A2ᶜ

-- Balls drawn with replacement implies independent events
variable (h_independent : P.IndepEvents A1 A2)

-- Prove that A1 and A2_complement are independent
theorem A1_and_A2_complement_independent : P.IndepEvents A1 A2_complement :=
by
  sorry

end A1_and_A2_complement_independent_l98_98324


namespace savings_calculation_l98_98453

theorem savings_calculation (income expenditure savings : ℕ) (ratio_income ratio_expenditure : ℕ)
  (h_ratio : ratio_income = 10) (h_ratio2 : ratio_expenditure = 7) (h_income : income = 10000)
  (h_expenditure : 10 * expenditure = 7 * income) :
  savings = income - expenditure :=
by
  sorry

end savings_calculation_l98_98453


namespace cube_volume_l98_98093

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98093


namespace hash_op_is_100_l98_98414

def hash_op (a b : ℕ) : ℕ := 4 * a^2 + 4 * b^2 + 8 * a * b

theorem hash_op_is_100 (a b : ℕ) (h1 : a + b = 5) : hash_op a b = 100 :=
sorry

end hash_op_is_100_l98_98414


namespace fill_time_with_conditions_l98_98708

-- Define rates as constants
def pipeA_rate := 1 / 10
def pipeB_rate := 1 / 6
def pipeC_rate := 1 / 5
def tarp_factor := 1 / 2
def leak_rate := 1 / 15

-- Define effective fill rate taking into account the tarp and leak
def effective_fill_rate := ((pipeA_rate + pipeB_rate + pipeC_rate) * tarp_factor) - leak_rate

-- Define the required time to fill the pool
def required_time := 1 / effective_fill_rate

theorem fill_time_with_conditions :
  required_time = 6 :=
by
  sorry

end fill_time_with_conditions_l98_98708


namespace lizette_quiz_average_l98_98963

theorem lizette_quiz_average
  (Q1 Q2 : ℝ)
  (Q3 : ℝ := 92)
  (h : (Q1 + Q2 + Q3) / 3 = 94) :
  (Q1 + Q2) / 2 = 95 := by
sorry

end lizette_quiz_average_l98_98963


namespace find_varphi_l98_98542

theorem find_varphi (ϕ : ℝ) (h1 : 0 < ϕ) (h2 : ϕ < π)
(h_symm : ∃ k : ℤ, ϕ = k * π + 2 * π / 3) :
ϕ = 2 * π / 3 :=
sorry

end find_varphi_l98_98542


namespace range_of_a_l98_98594

theorem range_of_a (a : ℝ) : (∀ (x : ℝ), (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ (1 < a ∧ a ≤ 2) :=
by sorry

end range_of_a_l98_98594


namespace cube_volume_l98_98086

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98086


namespace find_largest_x_and_compute_ratio_l98_98591

theorem find_largest_x_and_compute_ratio (a b c d : ℤ) (h : x = (a + b * Real.sqrt c) / d)
   (cond : (5 * x / 7) + 1 = 3 / x) : a * c * d / b = -70 :=
by
  sorry

end find_largest_x_and_compute_ratio_l98_98591


namespace inner_cube_surface_area_l98_98124

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l98_98124


namespace max_teams_4_weeks_l98_98247

noncomputable def max_teams_in_tournament (weeks number_teams : ℕ) : ℕ :=
  if h : weeks > 0 then (number_teams * (number_teams - 1)) / (2 * weeks) else 0

theorem max_teams_4_weeks : max_teams_in_tournament 4 7 = 6 := by
  -- Assumptions
  let n := 6
  let teams := 7 * n
  let weeks := 4
  
  -- Define the constraints and checks here
  sorry

end max_teams_4_weeks_l98_98247


namespace simplify_and_evaluate_l98_98007

theorem simplify_and_evaluate :
  ∀ (a b : ℤ), a = -1 → b = 4 →
  (a + b)^2 - 2 * a * (a - b) + (a + 2 * b) * (a - 2 * b) = -64 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end simplify_and_evaluate_l98_98007


namespace evaluate_f_at_3_l98_98933

-- Function definition
def f (x : ℚ) : ℚ := (x - 2) / (4 * x + 5)

-- Problem statement
theorem evaluate_f_at_3 : f 3 = 1 / 17 := by
  sorry

end evaluate_f_at_3_l98_98933


namespace greatest_triangle_perimeter_l98_98379

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l98_98379


namespace second_race_distance_remaining_l98_98457

theorem second_race_distance_remaining
  (race_distance : ℕ)
  (A_finish_time : ℕ)
  (B_remaining_distance : ℕ)
  (A_start_behind : ℕ)
  (A_speed : ℝ)
  (B_speed : ℝ)
  (A_distance_second_race : ℕ)
  (B_distance_second_race : ℝ)
  (v_ratio : ℝ)
  (B_remaining_second_race : ℝ) :
  race_distance = 10000 →
  A_finish_time = 50 →
  B_remaining_distance = 500 →
  A_start_behind = 500 →
  A_speed = race_distance / A_finish_time →
  B_speed = (race_distance - B_remaining_distance) / A_finish_time →
  v_ratio = A_speed / B_speed →
  v_ratio = 20 / 19 →
  A_distance_second_race = race_distance + A_start_behind →
  B_distance_second_race = B_speed * (A_distance_second_race / A_speed) →
  B_remaining_second_race = race_distance - B_distance_second_race →
  B_remaining_second_race = 25 := 
by
  sorry

end second_race_distance_remaining_l98_98457


namespace solve_for_a_l98_98293

theorem solve_for_a {a x : ℝ} (H : (x - 2) * (a * x^2 - x + 1) = a * x^3 + (-1 - 2 * a) * x^2 + 3 * x - 2 ∧ (-1 - 2 * a) = 0) : a = -1/2 := sorry

end solve_for_a_l98_98293


namespace gambler_received_max_2240_l98_98558

def largest_amount_received_back (x y l : ℕ) : ℕ :=
  if 2 * l + 2 = 14 ∨ 2 * l - 2 = 14 then 
    let lost_value_1 := (6 * 100 + 8 * 20)
    let lost_value_2 := (8 * 100 + 6 * 20)
    max (3000 - lost_value_1) (3000 - lost_value_2)
  else 0

theorem gambler_received_max_2240 {x y : ℕ} (hx : 20 * x + 100 * y = 3000)
  (hl : ∃ l : ℕ, (l + (l + 2) = 14 ∨ l + (l - 2) = 14)) :
  largest_amount_received_back x y 6 = 2240 ∧ largest_amount_received_back x y 8 = 2080 := by
  sorry

end gambler_received_max_2240_l98_98558


namespace three_irrational_numbers_l98_98398

theorem three_irrational_numbers (a b c d e : ℝ) 
  (ha : ¬ ∃ q1 q2 : ℚ, a = q1 + q2) 
  (hb : ¬ ∃ q1 q2 : ℚ, b = q1 + q2) 
  (hc : ¬ ∃ q1 q2 : ℚ, c = q1 + q2) 
  (hd : ¬ ∃ q1 q2 : ℚ, d = q1 + q2) 
  (he : ¬ ∃ q1 q2 : ℚ, e = q1 + q2) : 
  ∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) 
  ∧ (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) 
  ∧ (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e)
  ∧ (¬ ∃ q1 q2 : ℚ, x + y = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, y + z = q1 + q2) 
  ∧ (¬ ∃ q1 q2 : ℚ, z + x = q1 + q2) :=
sorry

end three_irrational_numbers_l98_98398


namespace line_eq_l98_98911

theorem line_eq (x y : ℝ) :
  (∃ (x1 y1 x2 y2 : ℝ), x1 = 5 ∧ y1 = 0 ∧ x2 = 2 ∧ y2 = -5 ∧
    (y - y1) / (x - x1) = (y2 - y1) / (x2 - x1)) →
  5 * x - 3 * y - 25 = 0 :=
sorry

end line_eq_l98_98911


namespace lizard_eye_difference_l98_98516

def jan_eye : ℕ := 3
def jan_wrinkle : ℕ := 3 * jan_eye
def jan_spot : ℕ := 7 * jan_wrinkle

def cousin_eye : ℕ := 3
def cousin_wrinkle : ℕ := 2 * cousin_eye
def cousin_spot : ℕ := 5 * cousin_wrinkle

def total_eyes : ℕ := jan_eye + cousin_eye
def total_wrinkles : ℕ := jan_wrinkle + cousin_wrinkle
def total_spots : ℕ := jan_spot + cousin_spot
def total_spots_and_wrinkles : ℕ := total_wrinkles + total_spots

theorem lizard_eye_difference : total_spots_and_wrinkles - total_eyes = 102 := by
  sorry

end lizard_eye_difference_l98_98516


namespace greatest_possible_perimeter_l98_98332

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l98_98332


namespace carla_water_requirement_l98_98582

theorem carla_water_requirement (h: ℕ) (p: ℕ) (c: ℕ) (gallons_per_pig: ℕ) (horse_factor: ℕ) 
  (num_pigs: ℕ) (num_horses: ℕ) (tank_water: ℕ): 
  num_pigs = 8 ∧ num_horses = 10 ∧ gallons_per_pig = 3 ∧ horse_factor = 2 ∧ tank_water = 30 →
  h = horse_factor * gallons_per_pig ∧ p = num_pigs * gallons_per_pig ∧ c = tank_water →
  h * num_horses + p + c = 114 :=
by
  intro h1 h2
  cases h1
  cases h2
  sorry

end carla_water_requirement_l98_98582


namespace greatest_possible_perimeter_l98_98336

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l98_98336


namespace goldie_earnings_l98_98200

theorem goldie_earnings (hourly_rate : ℝ) (hours_week1 : ℝ) (hours_week2 : ℝ) :
  hourly_rate = 5 → hours_week1 = 20 → hours_week2 = 30 → (hourly_rate * (hours_week1 + hours_week2) = 250) :=
by
  intro h_rate
  intro h_week1
  intro h_week2
  rw [h_rate, h_week1, h_week2]
  norm_num
  sorry

end goldie_earnings_l98_98200


namespace simplify_expression_l98_98535

variable {a b : ℝ}

theorem simplify_expression {a b : ℝ} (h : |2 - a + b| + (ab + 1)^2 = 0) :
  (4 * a - 5 * b - a * b) - (2 * a - 3 * b + 5 * a * b) = 10 := by
  sorry

end simplify_expression_l98_98535


namespace tylenol_mg_per_tablet_l98_98652

noncomputable def dose_intervals : ℕ := 3  -- Mark takes Tylenol 3 times
noncomputable def total_mg : ℕ := 3000     -- Total intake in milligrams
noncomputable def tablets_per_dose : ℕ := 2  -- Number of tablets per dose

noncomputable def tablet_mg : ℕ :=
  total_mg / dose_intervals / tablets_per_dose

theorem tylenol_mg_per_tablet : tablet_mg = 500 := by
  sorry

end tylenol_mg_per_tablet_l98_98652


namespace remainder_of_cake_l98_98223

theorem remainder_of_cake (John Emily : ℝ) (h1 : 0.60 ≤ John) (h2 : Emily = 0.50 * (1 - John)) :
  1 - John - Emily = 0.20 :=
by
  sorry

end remainder_of_cake_l98_98223


namespace calculate_glass_area_l98_98949

-- Given conditions as definitions
def long_wall_length : ℕ := 30
def long_wall_height : ℕ := 12
def short_wall_length : ℕ := 20

-- Total area of glass required (what we want to prove)
def total_glass_area : ℕ := 960

-- The theorem to prove
theorem calculate_glass_area
  (a1 : long_wall_length = 30)
  (a2 : long_wall_height = 12)
  (a3 : short_wall_length = 20) :
  2 * (long_wall_length * long_wall_height) + (short_wall_length * long_wall_height) = total_glass_area :=
by
  -- The proof is omitted
  sorry

end calculate_glass_area_l98_98949


namespace floor_sqrt_30_squared_eq_25_l98_98906

theorem floor_sqrt_30_squared_eq_25 (h1 : 5 < Real.sqrt 30) (h2 : Real.sqrt 30 < 6) : Int.floor (Real.sqrt 30) ^ 2 = 25 := 
by
  sorry

end floor_sqrt_30_squared_eq_25_l98_98906


namespace simplify_expr_l98_98585

theorem simplify_expr (x : ℝ) (hx : x ≠ 0) :
  (3/4) * (8/(x^2) + 12*x - 5) = 6/(x^2) + 9*x - 15/4 := by
  sorry

end simplify_expr_l98_98585


namespace greatest_possible_perimeter_l98_98372

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l98_98372


namespace digging_depth_l98_98052

theorem digging_depth :
  (∃ (D : ℝ), 750 * D = 75000) → D = 100 :=
by
  sorry

end digging_depth_l98_98052


namespace decrease_A_share_l98_98233

theorem decrease_A_share :
  ∃ (a b x : ℝ),
    a + b + 495 = 1010 ∧
    (a - x) / 3 = 96 ∧
    (b - 10) / 2 = 96 ∧
    x = 25 :=
by
  sorry

end decrease_A_share_l98_98233


namespace boys_on_soccer_team_l98_98573

theorem boys_on_soccer_team (B G : ℕ) (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 20) : B = 15 :=
sorry

end boys_on_soccer_team_l98_98573


namespace cube_volume_l98_98103

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98103


namespace remainder_of_p_div_x_minus_3_l98_98915

def p (x : ℝ) : ℝ := x^4 - x^3 - 4 * x + 7

theorem remainder_of_p_div_x_minus_3 : 
  let remainder := p 3 
  remainder = 49 := 
by
  sorry

end remainder_of_p_div_x_minus_3_l98_98915


namespace inner_cube_surface_area_l98_98148

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l98_98148


namespace solve_arithmetic_sequence_l98_98972

theorem solve_arithmetic_sequence (y : ℝ) (h1 : y ^ 2 = (4 + 25) / 2) (h2 : y > 0) :
  y = Real.sqrt 14.5 :=
sorry

end solve_arithmetic_sequence_l98_98972


namespace negation_equivalence_l98_98020

theorem negation_equivalence (x : ℝ) :
  (¬ (x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (x < 1 → x^2 - 4*x + 2 < -1) :=
by
  sorry

end negation_equivalence_l98_98020


namespace greatest_perimeter_of_triangle_l98_98338

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l98_98338


namespace intersection_points_l98_98471

noncomputable def even_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = f x

def monotonically_increasing (f : ℝ → ℝ) :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

theorem intersection_points (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_mono_inc : monotonically_increasing f)
  (h_sign_change : f 1 * f 2 < 0) :
  ∃! x1 x2 : ℝ, f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2 :=
sorry

end intersection_points_l98_98471


namespace inner_cube_surface_area_l98_98131

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l98_98131


namespace min_value_a_2b_l98_98925

theorem min_value_a_2b (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = a * b) :
  a + 2 * b ≥ 9 :=
sorry

end min_value_a_2b_l98_98925


namespace problem_statement_l98_98995

-- Define the function
def f (x : ℝ) := -2 * x^2

-- We need to show that f is monotonically decreasing and even on (0, +∞)
theorem problem_statement : (∀ x y : ℝ, 0 < x → 0 < y → x < y → f y < f x) ∧ (∀ x : ℝ, f (-x) = f x) := 
by {
  sorry -- proof goes here
}

end problem_statement_l98_98995


namespace union_sets_S_T_l98_98605

open Set Int

def S : Set Int := { s : Int | ∃ n : Int, s = 2 * n + 1 }
def T : Set Int := { t : Int | ∃ n : Int, t = 4 * n + 1 }

theorem union_sets_S_T : S ∪ T = S := 
by sorry

end union_sets_S_T_l98_98605


namespace find_pairs_l98_98561

noncomputable def pairs_of_real_numbers (α β : ℝ) := 
  ∀ x y z w : ℝ, 0 < x → 0 < y → 0 < z → 0 < w →
    (x + y^2 + z^3 + w^6 ≥ α * (x * y * z * w)^β)

theorem find_pairs (α β : ℝ) :
  (∃ x y z w : ℝ, 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w ∧
    (x + y^2 + z^3 + w^6 = α * (x * y * z * w)^β))
  →
  pairs_of_real_numbers α β :=
sorry

end find_pairs_l98_98561


namespace max_perimeter_l98_98389

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l98_98389


namespace inner_cube_surface_area_l98_98134

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l98_98134


namespace inner_cube_surface_area_l98_98144

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l98_98144


namespace universal_friendship_l98_98396

-- Define the inhabitants and their relationships
def inhabitants (n : ℕ) : Type := Fin n

-- Condition for friends and enemies
inductive Relationship (n : ℕ) : inhabitants n → inhabitants n → Prop
| friend (A B : inhabitants n) : Relationship n A B
| enemy (A B : inhabitants n) : Relationship n A B

-- Transitivity condition
axiom transitivity {n : ℕ} {A B C : inhabitants n} :
  Relationship n A B = Relationship n B C → Relationship n A C = Relationship n A B

-- At least two friends among any three inhabitants
axiom at_least_two_friends {n : ℕ} (A B C : inhabitants n) :
  ∃ X Y : inhabitants n, X ≠ Y ∧ Relationship n X Y = Relationship n X Y

-- Inhabitants can start a new life switching relationships
axiom start_new_life {n : ℕ} (A : inhabitants n) :
  ∀ B : inhabitants n, Relationship n A B = Relationship n B A

-- The main theorem we need to prove
theorem universal_friendship (n : ℕ) : 
  ∀ A B : inhabitants n, ∃ C : inhabitants n, Relationship n A C = Relationship n B C :=
sorry

end universal_friendship_l98_98396


namespace volume_of_cube_l98_98095

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98095


namespace right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l98_98879

-- Definitions for part (a)
def is_right_angled_triangle_a (a b c r r_a r_b r_c : ℝ) :=
  r + r_a + r_b + r_c = a + b + c

def right_angled_triangle_a (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_a (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_a a b c ↔ is_right_angled_triangle_a a b c r r_a r_b r_c := sorry

-- Definitions for part (b)
def is_right_angled_triangle_b (a b c r r_a r_b r_c : ℝ) :=
  r^2 + r_a^2 + r_b^2 + r_c^2 = a^2 + b^2 + c^2

def right_angled_triangle_b (a b c : ℝ) :=
  a^2 + b^2 = c^2 -- Assuming c is the hypotenuse

theorem right_triangle_iff_sum_excircles_b (a b c r r_a r_b r_c : ℝ) :
  right_angled_triangle_b a b c ↔ is_right_angled_triangle_b a b c r r_a r_b r_c := sorry

end right_triangle_iff_sum_excircles_a_right_triangle_iff_sum_excircles_b_l98_98879


namespace strictly_increasing_intervals_l98_98475

-- Define the function y = cos^2(x + π/2)
noncomputable def y (x : ℝ) : ℝ := (Real.cos (x + Real.pi / 2))^2

-- Define the assertion
theorem strictly_increasing_intervals (k : ℤ) : 
  StrictMonoOn y (Set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2)) :=
sorry

end strictly_increasing_intervals_l98_98475


namespace sequence_geq_four_l98_98923

theorem sequence_geq_four (a : ℕ → ℝ) (h0 : a 1 = 5) 
    (h1 : ∀ n ≥ 1, a (n+1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)) : 
    ∀ n ≥ 1, a n ≥ 4 := 
by
  sorry

end sequence_geq_four_l98_98923


namespace min_notebooks_needed_l98_98461

variable (cost_pen cost_notebook num_pens discount_threshold : ℕ)

theorem min_notebooks_needed (x : ℕ)
    (h1 : cost_pen = 10)
    (h2 : cost_notebook = 4)
    (h3 : num_pens = 3)
    (h4 : discount_threshold = 100)
    (h5 : num_pens * cost_pen + x * cost_notebook ≥ discount_threshold) :
    x ≥ 18 := 
sorry

end min_notebooks_needed_l98_98461


namespace greatest_possible_perimeter_l98_98327

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l98_98327


namespace wind_velocity_l98_98435

def pressure (P A V : ℝ) (k : ℝ) : Prop :=
  P = k * A * V^2

theorem wind_velocity (k : ℝ) (h_initial : pressure 4 4 8 k) (h_final : pressure 64 16 v k) : v = 16 := by
  sorry

end wind_velocity_l98_98435


namespace total_bathing_suits_l98_98051

def men_bathing_suits : ℕ := 14797
def women_bathing_suits : ℕ := 4969

theorem total_bathing_suits : men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l98_98051


namespace total_cost_l98_98578

-- Definitions based on the problem's conditions
def cost_hamburger : ℕ := 4
def cost_milkshake : ℕ := 3

def qty_hamburgers : ℕ := 7
def qty_milkshakes : ℕ := 6

-- The proof statement
theorem total_cost :
  (qty_hamburgers * cost_hamburger + qty_milkshakes * cost_milkshake) = 46 :=
by
  sorry

end total_cost_l98_98578


namespace limit_expression_l98_98929

variables {a b x₀ : ℝ}
variables {f : ℝ → ℝ}

-- Conditions
lemma f_diff_interval (h : Function.DifferentiableOn ℝ f (Ioo a b)) (hx₀ : x₀ ∈ Ioo a b) :
  (∃ f' : ℝ → ℝ, ∀ x ∈ Ioo a b, HasDerivAt f (f' x) x) :=
begin
  sorry,  -- Proof of differentiability leading to the lemma can be expanded here
end

-- Statement
theorem limit_expression (h : Function.DifferentiableOn ℝ f (Ioo a b)) (hx₀ : x₀ ∈ Ioo a b) :
  (∃ f' : ℝ, HasDerivAt f f' x₀) →
  (∃ L : ℝ, (L = 2 * (classical.some (h x₀ hx₀).Exists.some)) ∧
    Filter.Tendsto (λ h, (f (x₀ + h) - f (x₀ - h)) / h) (nhds_within 0 ℝ) (nhds L)) :=
begin
  sorry  -- Proof of the statement can be expanded here
end

end limit_expression_l98_98929


namespace cube_volume_l98_98080

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98080


namespace smallest_integer_condition_l98_98030

theorem smallest_integer_condition (x : ℝ) (hz : 9 = 9) (hineq : 27^9 > x^24) : x < 27 :=
  by {
    sorry
  }

end smallest_integer_condition_l98_98030


namespace part_a_part_b_l98_98423

-- Part (a): Number of ways to distribute 20 identical balls into 6 boxes so that no box is empty
theorem part_a:
  ∃ (n : ℕ), n = Nat.choose 19 5 :=
sorry

-- Part (b): Number of ways to distribute 20 identical balls into 6 boxes if some boxes can be empty
theorem part_b:
  ∃ (n : ℕ), n = Nat.choose 25 5 :=
sorry

end part_a_part_b_l98_98423


namespace minimum_notes_to_determine_prize_location_l98_98853

/--
There are 100 boxes, numbered from 1 to 100. A prize is hidden in one of the boxes, 
and the host knows its location. The viewer can send the host a batch of notes 
with questions that require a "yes" or "no" answer. The host shuffles the notes 
in the batch and, without announcing the questions aloud, honestly answers 
all of them. Prove that the minimum number of notes that need to be sent to 
definitely determine where the prize is located is 99.
-/
theorem minimum_notes_to_determine_prize_location : 
  ∀ (boxes : Fin 100 → Prop) (prize_location : ∃ i : Fin 100, boxes i) 
    (batch_size : Nat), 
  (batch_size + 1) ≥ 100 → batch_size = 99 :=
by
  sorry

end minimum_notes_to_determine_prize_location_l98_98853


namespace compare_real_numbers_l98_98720

theorem compare_real_numbers (a b : ℝ) : (a > b) ∨ (a = b) ∨ (a < b) :=
sorry

end compare_real_numbers_l98_98720


namespace sections_in_orchard_l98_98673

-- Conditions: Farmers harvest 45 sacks from each section daily, 360 sacks are harvested daily
def harvest_sacks_per_section : ℕ := 45
def total_sacks_harvested_daily : ℕ := 360

-- Statement: Prove that the number of sections is 8 given the conditions
theorem sections_in_orchard (h1 : harvest_sacks_per_section = 45) (h2 : total_sacks_harvested_daily = 360) :
  total_sacks_harvested_daily / harvest_sacks_per_section = 8 :=
sorry

end sections_in_orchard_l98_98673


namespace smallest_number_among_bases_l98_98176

noncomputable def convert_base_9 (n : ℕ) : ℕ :=
match n with
| 85 => 8 * 9 + 5
| _ => 0

noncomputable def convert_base_4 (n : ℕ) : ℕ :=
match n with
| 1000 => 1 * 4^3
| _ => 0

noncomputable def convert_base_2 (n : ℕ) : ℕ :=
match n with
| 111111 => 1 * 2^6 - 1
| _ => 0

theorem smallest_number_among_bases:
  min (min (convert_base_9 85) (convert_base_4 1000)) (convert_base_2 111111) = convert_base_2 111111 :=
by {
  sorry
}

end smallest_number_among_bases_l98_98176


namespace find_a_l98_98504

theorem find_a (a : ℝ) (h : ∀ B: ℝ × ℝ, (B = (a, 0)) → (2 - 0) * (0 - 2) = (4 - 2) * (2 - a)) : a = 4 :=
by
  sorry

end find_a_l98_98504


namespace my_problem_l98_98405

-- Definitions and conditions from the problem statement
variables (p q r u v w : ℝ)

-- Conditions
axiom h1 : 17 * u + q * v + r * w = 0
axiom h2 : p * u + 29 * v + r * w = 0
axiom h3 : p * u + q * v + 56 * w = 0
axiom h4 : p ≠ 17
axiom h5 : u ≠ 0

-- Problem statement to prove
theorem my_problem : (p / (p - 17)) + (q / (q - 29)) + (r / (r - 56)) = 0 :=
sorry

end my_problem_l98_98405


namespace can_pay_without_change_l98_98468

theorem can_pay_without_change (n : ℕ) (h : n > 7) :
  ∃ (a b : ℕ), 3 * a + 5 * b = n :=
sorry

end can_pay_without_change_l98_98468


namespace inner_cube_surface_area_l98_98155

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l98_98155


namespace greatest_possible_perimeter_l98_98329

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l98_98329


namespace greatest_possible_perimeter_l98_98335

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l98_98335


namespace value_of_f_is_29_l98_98961

noncomputable def f (x : ℕ) : ℕ := 3 * x - 4
noncomputable def g (x : ℕ) : ℕ := x^2 + 1

theorem value_of_f_is_29 :
  f (1 + g 3) = 29 := by
  sorry

end value_of_f_is_29_l98_98961


namespace usual_time_to_office_l98_98028

theorem usual_time_to_office (P : ℝ) (T : ℝ) (h1 : T = (3 / 4) * (T + 20)) : T = 60 :=
by
  sorry

end usual_time_to_office_l98_98028


namespace impossible_tiling_of_chessboard_l98_98467

theorem impossible_tiling_of_chessboard : 
  ¬ ∃ (tiling : Tiling (8 * 8 - 2) (1 * 2)), 
        ∀ (i j : ℕ) (hij : i < 8 ∧ j < 8 ∧ (i, j) ≠ (1, 1) ∧ (i, j) ≠ (8, 8)), 
          tiling.covers i j :=
sorry

end impossible_tiling_of_chessboard_l98_98467


namespace cube_volume_l98_98108

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98108


namespace exists_natural_number_plane_assignment_l98_98776

theorem exists_natural_number_plane_assignment :
  ∃ f : ℤ × ℤ → ℕ,
    (∀ n : ℕ, ∃ p : ℤ × ℤ, f p = n) ∧
    (∀ (a b c : ℤ) (h₁ : a ≠ 0 ∨ b ≠ 0) (h₂ : c ≠ 0),
       ∀ p₁ p₂ : ℤ × ℤ,
       a * p₁.1 + b * p₁.2 = c → a * p₂.1 + b * p₂.2 = c →
       ∃ d : ℤ, f p₁ = f (p₁.1 + d * b, p₁.2 - d * a) ∧ f p₂ = f (p₂.1 + d * b, p₂.2 - d * a)) :=
by {
  sorry
}

end exists_natural_number_plane_assignment_l98_98776


namespace remainder_when_divided_by_x_plus_2_l98_98556

def q (x D E F : ℝ) : ℝ := D*x^4 + E*x^2 + F*x - 2

theorem remainder_when_divided_by_x_plus_2 (D E F : ℝ) (h : q 2 D E F = 14) : q (-2) D E F = -18 := 
by 
     sorry

end remainder_when_divided_by_x_plus_2_l98_98556


namespace triangle_angle_contradiction_l98_98794

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α < 60) (h3 : β < 60) (h4 : γ < 60) : false := 
sorry

end triangle_angle_contradiction_l98_98794


namespace inner_cube_surface_area_l98_98156

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l98_98156


namespace exposed_circular_segment_sum_l98_98874

theorem exposed_circular_segment_sum (r h : ℕ) (angle : ℕ) (a b c : ℕ) :
    r = 8 ∧ h = 10 ∧ angle = 90 ∧ a = 16 ∧ b = 0 ∧ c = 0 → a + b + c = 16 :=
by
  intros
  sorry

end exposed_circular_segment_sum_l98_98874


namespace airplane_rows_l98_98279

theorem airplane_rows (r : ℕ) (h1 : ∀ (seats_per_row total_rows : ℕ), seats_per_row = 8 → total_rows = r →
  ∀ occupied_seats : ℕ, occupied_seats = (3 * seats_per_row) / 4 →
  ∀ unoccupied_seats : ℕ, unoccupied_seats = seats_per_row * total_rows - occupied_seats * total_rows →
  unoccupied_seats = 24): 
  r = 12 :=
by
  sorry

end airplane_rows_l98_98279


namespace heartsuit_ratio_l98_98502

def heartsuit (n m : ℕ) : ℕ := n^2 * m^3

theorem heartsuit_ratio :
  (heartsuit 3 5) / (heartsuit 5 3) = 5 / 3 :=
by
  sorry

end heartsuit_ratio_l98_98502


namespace passing_grade_fraction_l98_98560

theorem passing_grade_fraction (A B C D F : ℚ) (hA : A = 1/4) (hB : B = 1/2) (hC : C = 1/8) (hD : D = 1/12) (hF : F = 1/24) : 
  A + B + C = 7/8 :=
by
  sorry

end passing_grade_fraction_l98_98560


namespace solve_for_x_l98_98756

theorem solve_for_x (x y : ℝ) (h1 : 3 * x - 2 * y = 8) (h2 : 2 * x + 3 * y = 1) : x = 2 := 
by 
  sorry

end solve_for_x_l98_98756


namespace exists_four_scientists_l98_98892

theorem exists_four_scientists {n : ℕ} (h1 : n = 50)
  (knows : Fin n → Finset (Fin n))
  (h2 : ∀ x, (knows x).card ≥ 25) :
  ∃ a b c d : Fin n, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧
  a ≠ c ∧ b ≠ d ∧
  a ∈ knows b ∧ b ∈ knows c ∧ c ∈ knows d ∧ d ∈ knows a :=
by
  sorry

end exists_four_scientists_l98_98892


namespace rationalize_denominator_l98_98820

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98820


namespace cube_volume_l98_98104

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98104


namespace greatest_possible_perimeter_l98_98337

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l98_98337


namespace option_B_equals_six_l98_98861

theorem option_B_equals_six :
  (3 - (-3)) = 6 :=
by
  sorry

end option_B_equals_six_l98_98861


namespace power_function_m_eq_4_l98_98751

theorem power_function_m_eq_4 (m : ℝ) :
  (m^2 - 3*m - 3 = 1) → m = 4 :=
by
  sorry

end power_function_m_eq_4_l98_98751


namespace maximum_sum_of_squares_l98_98604

theorem maximum_sum_of_squares (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 5) :
  (a - b)^2 + (a - c)^2 + (a - d)^2 + (b - c)^2 + (b - d)^2 + (c - d)^2 ≤ 20 :=
sorry

end maximum_sum_of_squares_l98_98604


namespace graph_fixed_point_l98_98321

theorem graph_fixed_point (f : ℝ → ℝ) (h : f 1 = 1) : f 1 = 1 :=
by
  sorry

end graph_fixed_point_l98_98321


namespace Doug_lost_marbles_l98_98728

theorem Doug_lost_marbles (D E L : ℕ) 
    (h1 : E = D + 22) 
    (h2 : E = D - L + 30) 
    : L = 8 := by
  sorry

end Doug_lost_marbles_l98_98728


namespace preimage_of_4_neg_2_eq_1_3_l98_98749

def mapping (x y : ℝ) : ℝ × ℝ := (x + y, x - y)

theorem preimage_of_4_neg_2_eq_1_3 : ∃ x y : ℝ, mapping x y = (4, -2) ∧ (x = 1) ∧ (y = 3) :=
by 
  sorry

end preimage_of_4_neg_2_eq_1_3_l98_98749


namespace distribution_of_balls_l98_98204

theorem distribution_of_balls :
  ∃ (P : ℕ → ℕ → ℕ), P 6 4 = 9 := 
by
  sorry

end distribution_of_balls_l98_98204


namespace y_intercept_of_line_l98_98022

theorem y_intercept_of_line (m : ℝ) (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : b = 0) (h_slope : m = 3) (h_x_intercept : (a, b) = (4, 0)) :
  ∃ y : ℝ, (0, y) = (0, -12) :=
by 
  sorry

end y_intercept_of_line_l98_98022


namespace product_AM_CN_constant_l98_98215

variables {A B C M N : Point}
variables (tri : Triangle A B C)
variables (isosceles : is_isosceles tri)
variables (semi_circle : Semicircle (Segment AC))
variables (tangent_M : is_tangent semi_circle (Segment AB) M)
variables (tangent_N : is_tangent semi_circle (Segment BC) N)

theorem product_AM_CN_constant : 
  AM * CN = (AB / 2) ^ 2 :=
sorry

end product_AM_CN_constant_l98_98215


namespace rationalize_sqrt_fraction_l98_98804

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l98_98804


namespace find_m_and_n_l98_98627

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end find_m_and_n_l98_98627


namespace meena_work_days_l98_98418

theorem meena_work_days (M : ℝ) : 1/5 + 1/M = 3/10 → M = 10 :=
by
  sorry

end meena_work_days_l98_98418


namespace greatest_possible_perimeter_l98_98356

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l98_98356


namespace speed_upstream_l98_98850

-- Conditions definitions
def speed_of_boat_still_water : ℕ := 50
def speed_of_current : ℕ := 20

-- Theorem stating the problem
theorem speed_upstream : (speed_of_boat_still_water - speed_of_current = 30) :=
by
  -- Proof is omitted
  sorry

end speed_upstream_l98_98850


namespace right_triangle_angles_l98_98540

theorem right_triangle_angles (α β : ℝ) (h : α + β = 90) 
  (h_ratio : (180 - α) / (90 + α) = 9 / 11) : 
  (α = 58.5 ∧ β = 31.5) :=
by sorry

end right_triangle_angles_l98_98540


namespace packages_bought_l98_98039

theorem packages_bought (total_tshirts : ℕ) (tshirts_per_package : ℕ) 
  (h1 : total_tshirts = 426) (h2 : tshirts_per_package = 6) : 
  (total_tshirts / tshirts_per_package) = 71 :=
by 
  sorry

end packages_bought_l98_98039


namespace arc_length_correct_l98_98748

noncomputable def chord_length := 2
noncomputable def central_angle := 2
noncomputable def half_chord_length := 1
noncomputable def radius := 1 / Real.sin 1
noncomputable def arc_length := 2 * radius

theorem arc_length_correct :
  arc_length = 2 / Real.sin 1 := by
sorry

end arc_length_correct_l98_98748


namespace cos_half_angle_l98_98483

theorem cos_half_angle (α : ℝ) (h1 : Real.sin α = 4/5) (h2 : 0 < α ∧ α < Real.pi / 2) : 
    Real.cos (α / 2) = 2 * Real.sqrt 5 / 5 := 
by 
    sorry

end cos_half_angle_l98_98483


namespace max_triangle_perimeter_l98_98346

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l98_98346


namespace value_of_c_l98_98320

theorem value_of_c (b c : ℝ) (h : (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c * x + 12)) : c = 7 :=
sorry

end value_of_c_l98_98320


namespace calculate_expression_l98_98721

theorem calculate_expression : 
  let a := (-1 : Int) ^ 2023
  let b := (-8 : Int) / (-4)
  let c := abs (-5)
  a + b - c = -4 := 
by
  sorry

end calculate_expression_l98_98721


namespace matinee_ticket_price_l98_98245

theorem matinee_ticket_price
  (M : ℝ)  -- Denote M as the price of a matinee ticket
  (evening_ticket_price : ℝ := 12)  -- Price of an evening ticket
  (ticket_3D_price : ℝ := 20)  -- Price of a 3D ticket
  (matinee_tickets_sold : ℕ := 200)  -- Number of matinee tickets sold
  (evening_tickets_sold : ℕ := 300)  -- Number of evening tickets sold
  (tickets_3D_sold : ℕ := 100)  -- Number of 3D tickets sold
  (total_revenue : ℝ := 6600) -- Total revenue
  (h : matinee_tickets_sold * M + evening_tickets_sold * evening_ticket_price + tickets_3D_sold * ticket_3D_price = total_revenue) :
  M = 5 :=
by
  sorry

end matinee_ticket_price_l98_98245


namespace probability_interval_27_33_is_correct_sample_mean_is_correct_probability_X_lt_2737_is_correct_l98_98882

-- Conditions
def frequency_distribution : List (ℝ × ℕ) :=
  [ (14, 2), (17, 5), (20, 11), (23, 14), (26, 11), (29, 4), (32, 3) ]

def total_plants : ℕ := 50

def sample_mean : ℝ := 23.06

def sample_variance : ℝ := 18.5364

-- Questions

-- 1. Probability of the interval [27.5, 33.5]
def prob_interval_27_33 : ℝ :=
  (7 : ℝ) / (50 : ℝ)

-- 2. Sample mean calculation
def calc_sample_mean (data : List (ℝ × ℕ)) : ℝ :=
  (data.map (λ (x : ℝ × ℕ), x.1 * (x.2 : ℝ))).sum / (total_plants : ℝ)

-- 3. Probability P(X < 27.37) under normal distribution
def calc_prob_X_lt_2737 (μ σ: ℝ) : ℝ :=
  (probability (λ x, x < 27.37) (NormalDistribution PDF μ σ))

theorem probability_interval_27_33_is_correct :
  prob_interval_27_33 = 0.14 := sorry

theorem sample_mean_is_correct :
  calc_sample_mean frequency_distribution = sample_mean := sorry

theorem probability_X_lt_2737_is_correct :
  calc_prob_X_lt_2737 sample_mean (cmath.sqrt sample_variance) = 0.8413 := sorry

end probability_interval_27_33_is_correct_sample_mean_is_correct_probability_X_lt_2737_is_correct_l98_98882


namespace sin_neg_390_eq_neg_half_l98_98896

theorem sin_neg_390_eq_neg_half : Real.sin (-390 * Real.pi / 180) = -1 / 2 :=
  sorry

end sin_neg_390_eq_neg_half_l98_98896


namespace division_decomposition_l98_98035

theorem division_decomposition (a b : ℕ) (h₁ : a = 36) (h₂ : b = 3)
    (h₃ : 30 / b = 10) (h₄ : 6 / b = 2) (h₅ : 10 + 2 = 12) :
    a / b = (30 / b) + (6 / b) := 
sorry

end division_decomposition_l98_98035


namespace discount_amount_l98_98038

/-- Suppose Maria received a 25% discount on DVDs, and she paid $120.
    The discount she received is $40. -/
theorem discount_amount (P : ℝ) (h : 0.75 * P = 120) : P - 120 = 40 := 
sorry

end discount_amount_l98_98038


namespace cube_volume_l98_98073

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98073


namespace min_visible_sum_of_values_l98_98565

-- Definitions based on the problem conditions
def is_standard_die (die : ℕ → ℕ) : Prop :=
  ∀ (i j : ℕ), (i + j = 7) → (die j + die i = 7)

def corner_cubes (cubes : ℕ) : ℕ := 8
def edge_cubes (cubes : ℕ) : ℕ := 24
def face_center_cubes (cubes : ℕ) : ℕ := 24

-- The proof statement
theorem min_visible_sum_of_values
  (m : ℕ)
  (condition1 : is_standard_die m)
  (condition2 : corner_cubes 64 = 8)
  (condition3 : edge_cubes 64 = 24)
  (condition4 : face_center_cubes 64 = 24)
  (condition5 : 64 = 8 + 24 + 24 + 8): 
  m = 144 :=
sorry

end min_visible_sum_of_values_l98_98565


namespace inner_cube_surface_area_l98_98166

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l98_98166


namespace exactly_one_germinates_l98_98440

theorem exactly_one_germinates (pA pB : ℝ) (hA : pA = 0.8) (hB : pB = 0.9) : 
  (pA * (1 - pB) + (1 - pA) * pB) = 0.26 :=
by
  sorry

end exactly_one_germinates_l98_98440


namespace solve_for_y_l98_98971

theorem solve_for_y (y : ℝ) (h1 : y > 0) (h2 : y^2 = (4 + 25) / 2) : y = real.sqrt(14.5) :=
sorry

end solve_for_y_l98_98971


namespace inner_cube_surface_area_l98_98161

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l98_98161


namespace segment_MN_length_l98_98415

theorem segment_MN_length
  (A B C D M N : ℝ)
  (hA : A < B)
  (hB : B < C)
  (hC : C < D)
  (hM : M = (A + C) / 2)
  (hN : N = (B + D) / 2)
  (hAD : D - A = 68)
  (hBC : C - B = 26) :
  |M - N| = 21 :=
sorry

end segment_MN_length_l98_98415


namespace unique_root_range_l98_98016

theorem unique_root_range (a : ℝ) :
  (x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → (∃! x : ℝ, x^3 + (1 - 3 * a) * x^2 + 2 * a^2 * x - 2 * a * x + x + a^2 - a = 0) 
  → - (Real.sqrt 3) / 2 < a ∧ a < (Real.sqrt 3) / 2 :=
by
  sorry

end unique_root_range_l98_98016


namespace even_function_f_D_l98_98994

noncomputable def f_A (x : ℝ) : ℝ := 2 * |x| - 1
def D_f_A := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

def f_B (x : ℕ) : ℕ := x^2 + x

def f_C (x : ℝ) : ℝ := x ^ 3

noncomputable def f_D (x : ℝ) : ℝ := x^2
def D_f_D := {x : ℝ | (-1 ≤ x ∧ x < 0) ∨ (0 < x ∧ x ≤ 1)}

theorem even_function_f_D : 
  ∀ x ∈ D_f_D, f_D (-x) = f_D (x) :=
sorry

end even_function_f_D_l98_98994


namespace value_of_c_div_b_l98_98238

theorem value_of_c_div_b (a b c : ℕ) (h1 : a = 0) (h2 : a < b) (h3 : b < c) 
  (h4 : b ≠ a + 1) (h5 : (a + b + c) / 3 = 4 * b) : c / b = 11 :=
by
  sorry

end value_of_c_div_b_l98_98238


namespace evaluate_expression_l98_98730

theorem evaluate_expression : (-1 : ℤ)^(3^3) + (1 : ℤ)^(3^3) = 0 := 
by
  sorry

end evaluate_expression_l98_98730


namespace problem_1_problem_2_l98_98312

def setP (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def setS (x : ℝ) (m : ℝ) : Prop := |x - 1| ≤ m

theorem problem_1 (m : ℝ) : (m ∈ Set.Iic (3)) → ∀ x, (setP x ∨ setS x m) → setP x := sorry

theorem problem_2 : ¬ ∃ m : ℝ, ∀ x : ℝ, (setP x ↔ setS x m) := sorry

end problem_1_problem_2_l98_98312


namespace average_speed_stan_l98_98844

theorem average_speed_stan (d1 d2 : ℝ) (h1 h2 rest : ℝ) (total_distance total_time : ℝ) (avg_speed : ℝ) :
  d1 = 350 → 
  d2 = 400 → 
  h1 = 6 → 
  h2 = 7 → 
  rest = 0.5 → 
  total_distance = d1 + d2 → 
  total_time = h1 + h2 + rest → 
  avg_speed = total_distance / total_time → 
  avg_speed = 55.56 :=
by 
  intros h_d1 h_d2 h_h1 h_h2 h_rest h_total_distance h_total_time h_avg_speed
  sorry

end average_speed_stan_l98_98844


namespace rationalize_sqrt_fraction_l98_98829

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l98_98829


namespace common_difference_is_3_l98_98306

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop := 
  a 3 + a 11 = 24

def condition2 (a : ℕ → ℝ) : Prop := 
  a 4 = 3

theorem common_difference_is_3 (h_arith : is_arithmetic a d) (h1 : condition1 a) (h2 : condition2 a) : 
  d = 3 := 
sorry

end common_difference_is_3_l98_98306


namespace trim_area_dodecagon_pie_l98_98478

theorem trim_area_dodecagon_pie :
  let d := 8 -- diameter of the pie
  let r := d / 2 -- radius of the pie
  let A_circle := π * r^2 -- area of the circle
  let A_dodecagon := 3 * r^2 -- area of the dodecagon
  let A_trimmed := A_circle - A_dodecagon -- area to be trimmed
  let a := 16 -- coefficient of π in A_trimmed
  let b := 48 -- constant term in A_trimmed
  a + b = 64 := 
by 
  sorry

end trim_area_dodecagon_pie_l98_98478


namespace annual_interest_rate_of_second_investment_l98_98442

-- Definitions for the conditions
def total_income : ℝ := 575
def investment1 : ℝ := 3000
def rate1 : ℝ := 0.085
def income1 : ℝ := investment1 * rate1
def investment2 : ℝ := 5000
def target_income : ℝ := total_income - income1

-- Lean 4 statement to prove the annual simple interest rate of the second investment
theorem annual_interest_rate_of_second_investment : ∃ (r : ℝ), target_income = investment2 * (r / 100) ∧ r = 6.4 :=
by sorry

end annual_interest_rate_of_second_investment_l98_98442


namespace inner_cube_surface_area_l98_98141

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l98_98141


namespace cost_of_pen_l98_98441

theorem cost_of_pen :
  ∃ p q : ℚ, (3 * p + 4 * q = 264) ∧ (4 * p + 2 * q = 230) ∧ (p = 39.2) :=
by
  sorry

end cost_of_pen_l98_98441


namespace sum_of_midpoint_coords_l98_98448

theorem sum_of_midpoint_coords :
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  xm + ym = 11 :=
by
  let x1 := 10
  let y1 := 7
  let x2 := 4
  let y2 := 1
  let xm := (x1 + x2) / 2
  let ym := (y1 + y2) / 2
  sorry

end sum_of_midpoint_coords_l98_98448


namespace carla_gas_cost_l98_98722

theorem carla_gas_cost:
  let distance_grocery := 8
  let distance_school := 6
  let distance_bank := 12
  let distance_practice := 9
  let distance_dinner := 15
  let distance_home := 2 * distance_practice
  let total_distance := distance_grocery + distance_school + distance_bank + distance_practice + distance_dinner + distance_home
  let miles_per_gallon := 25
  let price_per_gallon_first := 2.35
  let price_per_gallon_second := 2.65
  let total_gallons := total_distance / miles_per_gallon
  let gallons_per_fill_up := total_gallons / 2
  let cost_first := gallons_per_fill_up * price_per_gallon_first
  let cost_second := gallons_per_fill_up * price_per_gallon_second
  let total_cost := cost_first + cost_second
  total_cost = 6.80 :=
by sorry

end carla_gas_cost_l98_98722


namespace rhombus_diagonal_length_l98_98976

theorem rhombus_diagonal_length
  (d2 : ℝ)
  (h1 : d2 = 20)
  (area : ℝ)
  (h2 : area = 150) :
  ∃ d1 : ℝ, d1 = 15 ∧ (area = (d1 * d2) / 2) := by
  sorry

end rhombus_diagonal_length_l98_98976


namespace volume_of_cube_l98_98101

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98101


namespace total_subjects_is_41_l98_98657

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end total_subjects_is_41_l98_98657


namespace cricket_team_members_l98_98854

theorem cricket_team_members (avg_whole_team: ℕ) (captain_age: ℕ) (wicket_keeper_age: ℕ) 
(remaining_avg_age: ℕ) (n: ℕ):
avg_whole_team = 23 →
captain_age = 25 →
wicket_keeper_age = 30 →
remaining_avg_age = 22 →
(n * avg_whole_team - captain_age - wicket_keeper_age = (n - 2) * remaining_avg_age) →
n = 11 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end cricket_team_members_l98_98854


namespace inequality_abc_l98_98778

theorem inequality_abc (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1/a + 1/(b * c)) * (1/b + 1/(c * a)) * (1/c + 1/(a * b)) ≥ 1728 :=
by sorry

end inequality_abc_l98_98778


namespace cube_volume_l98_98078

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98078


namespace find_length_of_CE_l98_98774

theorem find_length_of_CE
  (triangle_ABE_right : ∀ A B E : Type, ∃ (angle_AEB : Real), angle_AEB = 45)
  (triangle_BCE_right : ∀ B C E : Type, ∃ (angle_BEC : Real), angle_BEC = 45)
  (triangle_CDE_right : ∀ C D E : Type, ∃ (angle_CED : Real), angle_CED = 45)
  (AE_is_32 : 32 = 32) :
  ∃ (CE : ℝ), CE = 16 * Real.sqrt 2 :=
by
  sorry

end find_length_of_CE_l98_98774


namespace brooke_total_jumping_jacks_l98_98532

def sj1 : Nat := 20
def sj2 : Nat := 36
def sj3 : Nat := 40
def sj4 : Nat := 50
def Brooke_jumping_jacks : Nat := 3 * (sj1 + sj2 + sj3 + sj4)

theorem brooke_total_jumping_jacks : Brooke_jumping_jacks = 438 := by
  sorry

end brooke_total_jumping_jacks_l98_98532


namespace required_integer_l98_98846

def digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d1 := n / 1000
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  let d4 := n % 10
  d1 + d2 + d3 + d4 = sum

def middle_digits_sum_to (n : ℕ) (sum : ℕ) : Prop :=
  let d2 := (n / 100) % 10
  let d3 := (n / 10) % 10
  d2 + d3 = sum

def thousands_minus_units (n : ℕ) (diff : ℕ) : Prop :=
  let d1 := n / 1000
  let d4 := n % 10
  d1 - d4 = diff

def divisible_by (n : ℕ) (d : ℕ) : Prop :=
  n % d = 0

theorem required_integer : 
  ∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    digits_sum_to n 18 ∧ 
    middle_digits_sum_to n 9 ∧ 
    thousands_minus_units n 3 ∧ 
    divisible_by n 9 ∧ 
    n = 6453 :=
by
  sorry

end required_integer_l98_98846


namespace inner_cube_surface_area_l98_98165

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l98_98165


namespace greatest_possible_perimeter_l98_98371

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l98_98371


namespace roots_quadratic_sum_l98_98625

theorem roots_quadratic_sum (a b : ℝ) (h1 : (-2) + (-(1/4)) = -b/a)
  (h2 : -2 * (-(1/4)) = -2/a) : a + b = -13 := by
  sorry

end roots_quadratic_sum_l98_98625


namespace amusement_park_ticket_cost_l98_98704

theorem amusement_park_ticket_cost (T_adult T_child : ℕ) (num_children num_adults : ℕ) 
  (h1 : T_adult = 15) (h2 : T_child = 8) 
  (h3 : num_children = 15) (h4 : num_adults = 25 + num_children) :
  num_adults * T_adult + num_children * T_child = 720 :=
by
  sorry

end amusement_park_ticket_cost_l98_98704


namespace ratio_of_rectangle_to_triangle_l98_98992

variable (L W : ℝ)

theorem ratio_of_rectangle_to_triangle (hL : L > 0) (hW : W > 0) : 
    L * W / (1/2 * L * W) = 2 := 
by
  sorry

end ratio_of_rectangle_to_triangle_l98_98992


namespace complex_is_purely_imaginary_iff_a_eq_2_l98_98205

theorem complex_is_purely_imaginary_iff_a_eq_2 (a : ℝ) :
  (a = 2) ↔ ((a^2 - 4 = 0) ∧ (a + 2 ≠ 0)) :=
by sorry

end complex_is_purely_imaginary_iff_a_eq_2_l98_98205


namespace mean_of_eight_numbers_l98_98551

theorem mean_of_eight_numbers (sum_of_numbers : ℚ) (h : sum_of_numbers = 3/4) : 
  sum_of_numbers / 8 = 3/32 := by
  sorry

end mean_of_eight_numbers_l98_98551


namespace general_term_of_sequence_l98_98603

noncomputable def a (n : ℕ) : ℝ :=
  if n = 1 then 1 else
  if n = 2 then 2 else
  sorry -- the recurrence relation will go here, but we'll skip its implementation

theorem general_term_of_sequence :
  ∀ n : ℕ, n ≥ 1 → a n = 3 - (2 / n) :=
by sorry

end general_term_of_sequence_l98_98603


namespace lila_will_have_21_tulips_l98_98682

def tulip_orchid_ratio := 3 / 4

def initial_orchids := 16

def added_orchids := 12

def total_orchids : ℕ := initial_orchids + added_orchids

def groups_of_orchids : ℕ := total_orchids / 4

def total_tulips : ℕ := 3 * groups_of_orchids

theorem lila_will_have_21_tulips :
  total_tulips = 21 := by
  sorry

end lila_will_have_21_tulips_l98_98682


namespace mushroom_mass_decrease_l98_98443

theorem mushroom_mass_decrease :
  ∀ (initial_mass water_content_fresh water_content_dry : ℝ),
  water_content_fresh = 0.8 →
  water_content_dry = 0.2 →
  (initial_mass * (1 - water_content_fresh) / (1 - water_content_dry) = initial_mass * 0.25) →
  (initial_mass - initial_mass * 0.25) / initial_mass = 0.75 :=
by
  intros initial_mass water_content_fresh water_content_dry h_fresh h_dry h_dry_mass
  sorry

end mushroom_mass_decrease_l98_98443


namespace find_a_20_l98_98307

-- Arithmetic sequence definition and known conditions
def arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

variable (a : ℕ → ℤ)

-- Conditions
def condition1 : Prop := a 1 + a 2 + a 3 = 6
def condition2 : Prop := a 5 = 8

-- The main statement to prove
theorem find_a_20 (h_arith : arithmetic_sequence a) (h_cond1 : condition1 a) (h_cond2 : condition2 a) : 
  a 20 = 38 := by
  sorry

end find_a_20_l98_98307


namespace jacks_paycheck_l98_98947

theorem jacks_paycheck (P : ℝ) (h1 : 0.2 * 0.8 * P = 20) : P = 125 :=
sorry

end jacks_paycheck_l98_98947


namespace volume_of_cube_l98_98096

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98096


namespace family_members_count_l98_98639

variable (F : ℕ) -- Number of other family members

def annual_cost_per_person : ℕ := 4000 + 12 * 1000
def john_total_cost_for_family (F : ℕ) : ℕ := (F + 1) * annual_cost_per_person / 2

theorem family_members_count :
  john_total_cost_for_family F = 32000 → F = 3 := by
  sorry

end family_members_count_l98_98639


namespace determine_x_value_l98_98523

variable {a b x r : ℝ}
variable (b_nonzero : b ≠ 0)

theorem determine_x_value (h1 : r = (3 * a)^(3 * b)) (h2 : r = a^b * x^b) : x = 27 * a^2 :=
by
  sorry

end determine_x_value_l98_98523


namespace find_ABC_l98_98877

theorem find_ABC {A B C : ℕ} (h₀ : ∀ n : ℕ, n ≤ 9 → n ≤ 9) (h₁ : 0 ≤ A) (h₂ : A ≤ 9) 
  (h₃ : 0 ≤ B) (h₄ : B ≤ 9) (h₅ : 0 ≤ C) (h₆ : C ≤ 9) (h₇ : 100 * A + 10 * B + C = B^C - A) :
  100 * A + 10 * B + C = 127 := by {
  sorry
}

end find_ABC_l98_98877


namespace number_of_elements_in_A_l98_98422

theorem number_of_elements_in_A (a b : ℕ) (h1 : a = 3 * b)
  (h2 : a + b - 100 = 500) (h3 : 100 = 100) (h4 : a - 100 = b - 100 + 50) : a = 450 := by
  sorry

end number_of_elements_in_A_l98_98422


namespace shopkeeper_loss_percent_l98_98041

theorem shopkeeper_loss_percent
  (initial_value : ℝ)
  (profit_percent : ℝ)
  (loss_percent : ℝ)
  (remaining_value_percent : ℝ)
  (profit_percent_10 : profit_percent = 0.10)
  (loss_percent_70 : loss_percent = 0.70)
  (initial_value_100 : initial_value = 100)
  (remaining_value_percent_30 : remaining_value_percent = 0.30)
  (selling_price : ℝ := initial_value * (1 + profit_percent))
  (remaining_value : ℝ := initial_value * remaining_value_percent)
  (remaining_selling_price : ℝ := remaining_value * (1 + profit_percent))
  (loss_value : ℝ := initial_value - remaining_selling_price)
  (shopkeeper_loss_percent : ℝ := loss_value / initial_value * 100) : 
  shopkeeper_loss_percent = 67 :=
sorry

end shopkeeper_loss_percent_l98_98041


namespace tom_profit_calculation_l98_98688

theorem tom_profit_calculation :
  let flour_needed := 500
  let flour_per_bag := 50
  let flour_bag_cost := 20
  let salt_needed := 10
  let salt_cost_per_pound := 0.2
  let promotion_cost := 1000
  let tickets_sold := 500
  let ticket_price := 20

  let flour_bags := flour_needed / flour_per_bag
  let cost_flour := flour_bags * flour_bag_cost
  let cost_salt := salt_needed * salt_cost_per_pound
  let total_expenses := cost_flour + cost_salt + promotion_cost
  let total_revenue := tickets_sold * ticket_price

  let profit := total_revenue - total_expenses

  profit = 8798 := by
  sorry

end tom_profit_calculation_l98_98688


namespace common_tangents_l98_98679

def circle1_eqn (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y = 0
def circle2_eqn (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2*y + 4 = 0

theorem common_tangents :
  ∃ (n : ℕ), n = 4 ∧ 
    (∀ (L : ℝ → ℝ → Prop), 
      (∀ x y, L x y → circle1_eqn x y ∧ circle2_eqn x y) → n = 4) := 
sorry

end common_tangents_l98_98679


namespace hannahs_trip_cost_l98_98314

noncomputable def calculate_gas_cost (initial_odometer final_odometer : ℕ) (fuel_economy_mpg : ℚ) (cost_per_gallon : ℚ) : ℚ :=
  let distance := final_odometer - initial_odometer
  let fuel_used := distance / fuel_economy_mpg
  fuel_used * cost_per_gallon

theorem hannahs_trip_cost :
  calculate_gas_cost 36102 36131 32 (385 / 100) = 276 / 100 :=
by
  sorry

end hannahs_trip_cost_l98_98314


namespace rationalize_sqrt_5_over_12_l98_98827

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l98_98827


namespace sum_of_arithmetic_sequence_2008_terms_l98_98576

theorem sum_of_arithmetic_sequence_2008_terms :
  let a := -1776
  let d := 11
  let n := 2008
  let l := a + (n - 1) * d
  let S := (n / 2) * (a + l)
  S = 18599100 := by
  sorry

end sum_of_arithmetic_sequence_2008_terms_l98_98576


namespace number_of_bedrooms_l98_98239

-- Conditions
def battery_life : ℕ := 10
def vacuum_time_per_room : ℕ := 4
def num_initial_rooms : ℕ := 2 -- kitchen and living room
def num_charges : ℕ := 2

-- Computation of total vacuuming time
def total_vacuuming_time : ℕ := battery_life * (num_charges + 1)

-- Computation of remaining time for bedrooms
def time_for_bedrooms : ℕ := total_vacuuming_time - (vacuum_time_per_room * num_initial_rooms)

-- Proof problem: Prove number of bedrooms
theorem number_of_bedrooms (B : ℕ) (h : B = time_for_bedrooms / vacuum_time_per_room) : B = 5 := by 
  sorry

end number_of_bedrooms_l98_98239


namespace remainder_mod_1000_l98_98956

open Finset

noncomputable def T : Finset ℕ := (range 12).map ⟨λ x, x + 1, λ x y h, by linarith⟩

def m : ℕ := (3 ^ card T) / 2 - 2 * (2 ^ card T) / 2 + 1 / 2

theorem remainder_mod_1000 : m % 1000 = 625 := by
  -- m is defined considering the steps mentioned in the problem
  have hT: card T = 12 := by
    rw [T, card_map, card_range]
    simp
  -- calculations for m
  have h3pow : 3 ^ 12 = 531441 := by norm_num
  have h2pow : 2 ^ 12 = 4096 := by norm_num
  have h2powDoubled : 2 * 4096 = 8192 := by norm_num
  have hend: (531441 - 8192 + 1) / 2 = 261625 := by norm_num
  -- combining all
  rw [m, hT, h3pow, h2pow, h2powDoubled, hend]
  norm_num
  sorry

end remainder_mod_1000_l98_98956


namespace range_of_a_l98_98493

noncomputable def f (a x : ℝ) : ℝ := Real.log x + 2 * a * (1 - x)

theorem range_of_a (a : ℝ) :
  (∀ x, x > 2 → f a x > f a 2) ↔ (a ∈ Set.Iic 0 ∪ Set.Ici (1 / 4)) :=
by
  sorry

end range_of_a_l98_98493


namespace radian_measure_sector_l98_98620

theorem radian_measure_sector (r l : ℝ) (h1 : 2 * r + l = 12) (h2 : (1 / 2) * l * r = 8) :
  l / r = 1 ∨ l / r = 4 := by
  sorry

end radian_measure_sector_l98_98620


namespace cube_volume_of_surface_area_l98_98070

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98070


namespace remainder_sets_two_disjoint_subsets_l98_98954

noncomputable def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem remainder_sets_two_disjoint_subsets (m : ℕ)
  (h : m = (3^12 - 2 * 2^12 + 1) / 2) : m % 1000 = 625 := 
by {
  -- math proof is omitted
  sorry
}

end remainder_sets_two_disjoint_subsets_l98_98954


namespace no_adjacent_same_roll_proof_l98_98192

open ProbabilityTheory

-- Define the probability calculation in the Lean framework.
def no_adjacent_same_roll_prob : ℚ :=
  let pA := 1 / 8
  let p_diff_7 := (7 / 8) ^ 2
  let p_diff_6 := 6 / 8
  let p_diff_5 := 5 / 8
  let case1 := pA * p_diff_7 * p_diff_6
  let p_diff_7_rest := 7 / 8
  let p_diff_6_2 := (6 / 8) ^ 2
  let case2 := p_diff_7_rest * p_diff_6_2 * p_diff_5
  (case1 + case2) / 8

-- Statement of the proof problem, including all relevant conditions and the final proof goal.
theorem no_adjacent_same_roll_proof :
  no_adjacent_same_roll_prob = 777 / 2048 :=
sorry

end no_adjacent_same_roll_proof_l98_98192


namespace circle_center_eq_circle_center_is_1_3_2_l98_98727

-- Define the problem: Given the equation of the circle, prove the center is (1, 3/2)
theorem circle_center_eq (x y : ℝ) :
  16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0 ↔ (x - 1)^2 + (y - 3/2)^2 = 3 := sorry

-- Prove that the center of the circle from the given equation is (1, 3/2)
theorem circle_center_is_1_3_2 :
  ∃ x y : ℝ, (16 * x^2 - 32 * x + 16 * y^2 - 48 * y + 100 = 0) ∧ (x = 1) ∧ (y = 3 / 2) := sorry

end circle_center_eq_circle_center_is_1_3_2_l98_98727


namespace tourists_speeds_l98_98003

theorem tourists_speeds (x y : ℝ) :
  (20 / x + 2.5 = 20 / y) →
  (20 / (x - 2) = 20 / (1.5 * y)) →
  x = 8 ∧ y = 4 :=
by
  intros h1 h2
  -- The proof would go here
  sorry

end tourists_speeds_l98_98003


namespace problem_1_problem_2_l98_98492

noncomputable def f (x : ℝ) := Real.sin x + (x - 1) / Real.exp x

theorem problem_1 (x : ℝ) (h₀ : x ∈ Set.Icc (-Real.pi) (Real.pi / 2)) :
  MonotoneOn f (Set.Icc (-Real.pi) (Real.pi / 2)) :=
sorry

theorem problem_2 (k : ℝ) :
  ∀ x ∈ Set.Icc (-Real.pi) 0, ((f x - Real.sin x) * Real.exp x - Real.cos x) ≤ k * Real.sin x → 
  k ∈ Set.Iic (1 + Real.pi / 2) :=
sorry

end problem_1_problem_2_l98_98492


namespace gasoline_expense_l98_98005

-- Definitions for the conditions
def lunch_cost : ℝ := 15.65
def gift_cost_per_person : ℝ := 5
def number_of_people : ℕ := 2
def grandma_gift_per_person : ℝ := 10
def initial_amount : ℝ := 50
def amount_left_for_return_trip : ℝ := 36.35

-- Definition for the total gift cost
def total_gift_cost : ℝ := number_of_people * gift_cost_per_person

-- Definition for the total amount received from grandma
def total_grandma_gift : ℝ := number_of_people * grandma_gift_per_person

-- Definition for the total initial amount including the gift from grandma
def total_initial_amount_with_gift : ℝ := initial_amount + total_grandma_gift

-- Definition for remaining amount after spending on lunch and gifts
def remaining_after_known_expenses : ℝ := total_initial_amount_with_gift - lunch_cost - total_gift_cost

-- The Lean theorem to prove the gasoline expense
theorem gasoline_expense : remaining_after_known_expenses - amount_left_for_return_trip = 8 := by
  sorry

end gasoline_expense_l98_98005


namespace a_9_value_l98_98486

-- Define the sequence and its sum of the first n terms
def S (n : ℕ) : ℕ := n^2

-- Define the terms of the sequence
def a (n : ℕ) : ℕ := if n = 1 then S 1 else S n - S (n - 1)

-- The main statement to be proved
theorem a_9_value : a 9 = 17 :=
by
  sorry

end a_9_value_l98_98486


namespace ratio_part_to_whole_l98_98792

variable (N : ℝ)

theorem ratio_part_to_whole :
  (1 / 1) * (1 / 3) * (2 / 5) * N = 10 →
  0.4 * N = 120 →
  (10 / ((1 / 3) * (2 / 5) * N) = 1 / 4) :=
by
  intros h1 h2
  sorry

end ratio_part_to_whole_l98_98792


namespace claudia_coins_l98_98900

variable (x y : ℕ)

theorem claudia_coins :
  (x + y = 15 ∧ ((145 - 5 * x) / 5) + 1 = 23) → y = 9 :=
by
  intro h
  -- The proof steps would go here, but we'll leave it as sorry for now.
  sorry

end claudia_coins_l98_98900


namespace jen_ducks_l98_98952

theorem jen_ducks (c d : ℕ) (h1 : d = 4 * c + 10) (h2 : c + d = 185) : d = 150 := by
  sorry

end jen_ducks_l98_98952


namespace parrot_age_is_24_l98_98857

variable (cat_age : ℝ) (rabbit_age : ℝ) (dog_age : ℝ) (parrot_age : ℝ)

def ages (cat_age rabbit_age dog_age parrot_age : ℝ) : Prop :=
  cat_age = 8 ∧
  rabbit_age = cat_age / 2 ∧
  dog_age = rabbit_age * 3 ∧
  parrot_age = cat_age + rabbit_age + dog_age

theorem parrot_age_is_24 (cat_age rabbit_age dog_age parrot_age : ℝ) :
  ages cat_age rabbit_age dog_age parrot_age → parrot_age = 24 :=
by
  intro h
  sorry

end parrot_age_is_24_l98_98857


namespace tangent_expression_l98_98902

theorem tangent_expression :
  (Real.tan (10 * Real.pi / 180) + Real.tan (50 * Real.pi / 180) + Real.tan (120 * Real.pi / 180))
  / (Real.tan (10 * Real.pi / 180) * Real.tan (50 * Real.pi / 180)) = -Real.sqrt 3 := by
  sorry

end tangent_expression_l98_98902


namespace rationalize_sqrt_fraction_l98_98833

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l98_98833


namespace a_2011_value_l98_98311

noncomputable def sequence_a : ℕ → ℝ
| 0 => 6/7
| (n + 1) => if 0 ≤ sequence_a n ∧ sequence_a n < 1/2 then 2 * sequence_a n
              else 2 * sequence_a n - 1

theorem a_2011_value : sequence_a 2011 = 6/7 := sorry

end a_2011_value_l98_98311


namespace total_subjects_l98_98654

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end total_subjects_l98_98654


namespace value_of_t_l98_98599

theorem value_of_t (x y t : ℝ) (hx : 2^x = t) (hy : 7^y = t) (hxy : 1/x + 1/y = 2) : t = Real.sqrt 14 :=
by
  sorry

end value_of_t_l98_98599


namespace angle_BDC_is_15_degrees_l98_98011

theorem angle_BDC_is_15_degrees (A B C D : Type) (AB AC AD CD : ℝ) (angle_BAC : ℝ) :
  AB = AC → AC = AD → CD = 2 * AC → angle_BAC = 30 →
  ∃ angle_BDC, angle_BDC = 15 := 
by
  sorry

end angle_BDC_is_15_degrees_l98_98011


namespace cube_volume_of_surface_area_l98_98069

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98069


namespace go_stones_problem_l98_98473

theorem go_stones_problem
  (x : ℕ) 
  (h1 : x / 7 + 40 = 555 / 5) 
  (black_stones : ℕ) 
  (h2 : black_stones = 55) :
  (x - black_stones = 442) :=
sorry

end go_stones_problem_l98_98473


namespace rationalize_denominator_XYZ_sum_l98_98419

noncomputable def a := (5 : ℝ)^(1/3)
noncomputable def b := (4 : ℝ)^(1/3)

theorem rationalize_denominator_XYZ_sum : 
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  X + Y + Z + W = 62 :=
by 
  sorry

end rationalize_denominator_XYZ_sum_l98_98419


namespace max_projection_area_of_tetrahedron_l98_98025

/-- 
Two adjacent faces of a tetrahedron are isosceles right triangles with a hypotenuse of 2,
and they form a dihedral angle of 60 degrees. The tetrahedron rotates around the common edge
of these faces. The maximum area of the projection of the rotating tetrahedron onto 
the plane containing the given edge is 1.
-/
theorem max_projection_area_of_tetrahedron (S hypotenuse dihedral max_proj_area : ℝ)
  (is_isosceles_right_triangle : ∀ (a b : ℝ), a^2 + b^2 = hypotenuse^2)
  (hypotenuse_len : hypotenuse = 2)
  (dihedral_angle : dihedral = 60) :
  max_proj_area = 1 :=
  sorry

end max_projection_area_of_tetrahedron_l98_98025


namespace heaviest_person_is_42_27_l98_98889

-- Define the main parameters using the conditions
def heaviest_person_weight (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real) (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) : Real :=
  let h := P 2 + 7.7
  h

-- State the theorem
theorem heaviest_person_is_42_27 (M : ℕ → Real) (P : ℕ → Real) (Q : ℕ → Real)
  (H : P 2 = 7.7) (L : Q 3 = 4.8) (S : P 1 + P 2 + P 3 = 106.6) :
  heaviest_person_weight M P Q H L S = 42.27 :=
sorry

end heaviest_person_is_42_27_l98_98889


namespace tangent_line_coordinates_l98_98909

theorem tangent_line_coordinates :
  ∃ x₀ : ℝ, ∃ y₀ : ℝ, (x₀ = 1 ∧ y₀ = Real.exp 1) ∧
  (∀ x : ℝ, ∀ y : ℝ, y = Real.exp x → ∃ m : ℝ, 
    (m = Real.exp 1 ∧ (y - y₀ = m * (x - x₀))) ∧
    (0 - y₀ = m * (0 - x₀))) := sorry

end tangent_line_coordinates_l98_98909


namespace cube_volume_from_surface_area_l98_98056

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98056


namespace circle_radius_l98_98741

theorem circle_radius (A B C O : Type) (AB AC : ℝ) (OA : ℝ) (r : ℝ) 
  (h1 : AB * AC = 60)
  (h2 : OA = 8) 
  (h3 : (8 + r) * (8 - r) = 60) : r = 2 :=
sorry

end circle_radius_l98_98741


namespace inner_cube_surface_area_l98_98170

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l98_98170


namespace domain_of_f_l98_98276

theorem domain_of_f (x : ℝ) : (2*x - x^2 > 0 ∧ x ≠ 1) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) :=
by
  -- proof omitted
  sorry

end domain_of_f_l98_98276


namespace angle_AMC_165_degrees_l98_98782

theorem angle_AMC_165_degrees {O A B C M : Point}
  (hO_center : is_circumcenter O A B C)
  (hO_opposite_B : O ∈ line_through A C ∧ B ∉ line_through A C) 
  (h_angle_AOC : ∠AOC = 60) 
  (hM_incenter : is_incenter M A B C) :
  ∠AMC = 165 := 
by 
parity
/-
This means proving the angle ∠AMC equals 165 degrees given:
1. O is the center of the circumcircle of triangle ABC
2. O and B lie on opposite sides of the line through A and C
3. The angle AOC equals 60 degrees
4. M is the incenter of triangle ABC.
-/
sorry -- proof would go here

end angle_AMC_165_degrees_l98_98782


namespace find_other_number_l98_98669

-- Definitions for the given conditions
def A : ℕ := 500
def LCM : ℕ := 3000
def HCF : ℕ := 100

-- Theorem statement: If A = 500, LCM(A, B) = 3000, and HCF(A, B) = 100, then B = 600.
theorem find_other_number (B : ℕ) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 :=
by
  sorry

end find_other_number_l98_98669


namespace greatest_perimeter_l98_98353

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l98_98353


namespace monica_tiles_l98_98411

theorem monica_tiles (room_length : ℕ) (room_width : ℕ) (border_tile_size : ℕ) (inner_tile_size : ℕ) 
  (border_tiles : ℕ) (inner_tiles : ℕ) (total_tiles : ℕ) :
  room_length = 24 ∧ room_width = 18 ∧ border_tile_size = 2 ∧ inner_tile_size = 3 ∧ 
  border_tiles = 38 ∧ inner_tiles = 32 → total_tiles = 70 :=
by {
  sorry
}

end monica_tiles_l98_98411


namespace greatest_possible_perimeter_l98_98385

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l98_98385


namespace inner_cube_surface_area_l98_98149

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l98_98149


namespace range_of_m_l98_98547

open Real

theorem range_of_m (a b m : ℝ) (x : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 1 / a + 9 / b = 1) :
  a + b ≥ -x^2 + 4 * x + 18 - m ↔ m ≥ 6 :=
by sorry

end range_of_m_l98_98547


namespace inner_cube_surface_area_l98_98125

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l98_98125


namespace probability_prime_factor_of_120_l98_98119

open Nat

theorem probability_prime_factor_of_120 : 
  let s := Finset.range 61
  let primes := {2, 3, 5}
  let prime_factors_of_5_fact := primes ∩ s
  (prime_factors_of_5_fact.card : ℚ) / s.card = 1 / 20 :=
by
  sorry

end probability_prime_factor_of_120_l98_98119


namespace f_4_1981_l98_98242

-- Define the function f with its properties
axiom f : ℕ → ℕ → ℕ

axiom f_0_y (y : ℕ) : f 0 y = y + 1
axiom f_x1_0 (x : ℕ) : f (x + 1) 0 = f x 1
axiom f_x1_y1 (x y : ℕ) : f (x + 1) (y + 1) = f x (f (x + 1) y)

theorem f_4_1981 : f 4 1981 = 2 ^ 3964 - 3 :=
sorry

end f_4_1981_l98_98242


namespace rolls_for_mode_of_two_l98_98613

theorem rolls_for_mode_of_two (n : ℕ) (p : ℚ := 1/6) (m0 : ℕ := 32) : 
  (n : ℚ) * p - (1 - p) ≤ m0 ∧ m0 ≤ (n : ℚ) * p + p ↔ 191 ≤ n ∧ n ≤ 197 := 
by
  sorry

end rolls_for_mode_of_two_l98_98613


namespace smallest_integer_k_l98_98447

theorem smallest_integer_k :
  ∃ k : ℕ, 
    k > 1 ∧ 
    k % 19 = 1 ∧ 
    k % 14 = 1 ∧ 
    k % 9 = 1 ∧ 
    k = 2395 :=
by {
  sorry
}

end smallest_integer_k_l98_98447


namespace inner_cube_surface_area_l98_98140

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l98_98140


namespace find_circle_eq_find_range_of_dot_product_l98_98395

open Real
open Set

-- Define the problem conditions
def line_eq (x y : ℝ) : Prop := x - sqrt 3 * y = 4
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the point P inside the circle and condition that |PA|, |PO|, |PB| form a geometric sequence
def geometric_sequence_condition (x y : ℝ) : Prop :=
  sqrt ((x + 2)^2 + y^2) * sqrt ((x - 2)^2 + y^2) = x^2 + y^2

-- Prove the equation of the circle
theorem find_circle_eq :
  (∃ (r : ℝ), ∀ (x y : ℝ), line_eq x y → r = 2) → circle_eq x y :=
by
  -- skipping the proof
  sorry

-- Prove the range of values for the dot product
theorem find_range_of_dot_product :
  (∀ (x y : ℝ), circle_eq x y ∧ geometric_sequence_condition x y) →
  -2 < (x^2 - 1 * y^2 - 1) → (x^2 - 4 + y^2) < 0 :=
by
  -- skipping the proof
  sorry

end find_circle_eq_find_range_of_dot_product_l98_98395


namespace expected_amoebas_after_one_week_l98_98890

section AmoebaProblem

-- Definitions from conditions
def initial_amoebas : ℕ := 1
def split_probability : ℝ := 0.8
def days : ℕ := 7

-- Function to calculate expected amoebas
def expected_amoebas (n : ℕ) : ℝ :=
  initial_amoebas * ((2 : ℝ) ^ n) * (split_probability ^ n)

-- Theorem statement
theorem expected_amoebas_after_one_week :
  expected_amoebas days = 26.8435456 :=
by sorry

end AmoebaProblem

end expected_amoebas_after_one_week_l98_98890


namespace g_of_5_l98_98241

variable {g : ℝ → ℝ}
variable (h1 : ∀ x y : ℝ, 2 * x * g y = 3 * y * g x)
variable (h2 : g 10 = 15)

theorem g_of_5 : g 5 = 45 / 4 :=
  sorry

end g_of_5_l98_98241


namespace function_value_proof_l98_98648

theorem function_value_proof (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : ∀ x, f (x + 1) = -f (-x + 1))
    (h2 : ∀ x, f (x + 2) = f (-x + 2))
    (h3 : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b)
    (h4 : ∀ x y : ℝ, x - y - 3 = 0)
    : f (9/2) = 5/4 := by
  sorry

end function_value_proof_l98_98648


namespace students_number_l98_98686

theorem students_number (C P S : ℕ) : C = 315 ∧ 121 + C = P * S -> S = 4 := by
  sorry

end students_number_l98_98686


namespace line_equation_intersects_ellipse_l98_98198

theorem line_equation_intersects_ellipse :
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y : ℝ, l x y ↔ 5 * x + 4 * y - 9 = 0) ∧
    (∃ M N : ℝ × ℝ,
      (M.1^2 / 20 + M.2^2 / 16 = 1) ∧
      (N.1^2 / 20 + N.2^2 / 16 = 1) ∧
      ((M.1 + N.1) / 2 = 1) ∧
      ((M.2 + N.2) / 2 = 1)) :=
sorry

end line_equation_intersects_ellipse_l98_98198


namespace least_number_of_marbles_divisible_by_2_3_4_5_6_7_l98_98712

theorem least_number_of_marbles_divisible_by_2_3_4_5_6_7 : 
  ∃ n : ℕ, (∀ k ∈ [2, 3, 4, 5, 6, 7], k ∣ n) ∧ n = 420 :=
  by sorry

end least_number_of_marbles_divisible_by_2_3_4_5_6_7_l98_98712


namespace greatest_perimeter_of_triangle_l98_98342

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l98_98342


namespace inverse_of_exponential_l98_98904

theorem inverse_of_exponential (x : ℝ) (h : x > 0) : 2^(1 + log x) = x :=
sorry

end inverse_of_exponential_l98_98904


namespace percentage_of_students_owning_only_cats_is_10_percent_l98_98942

def total_students : ℕ := 500
def cat_owners : ℕ := 75
def dog_owners : ℕ := 150
def both_cat_and_dog_owners : ℕ := 25
def only_cat_owners : ℕ := cat_owners - both_cat_and_dog_owners
def percent_owning_only_cats : ℚ := (only_cat_owners * 100) / total_students

theorem percentage_of_students_owning_only_cats_is_10_percent : percent_owning_only_cats = 10 := by
  sorry

end percentage_of_students_owning_only_cats_is_10_percent_l98_98942


namespace line_passes_fixed_point_l98_98017

theorem line_passes_fixed_point (k : ℝ) :
    ((k + 1) * -1) - ((2 * k - 1) * 1) + 3 * k = 0 :=
by
    -- The proof is omitted as the primary aim is to ensure the correct Lean statement.
    sorry

end line_passes_fixed_point_l98_98017


namespace inequality_solution_set_inequality_proof_2_l98_98255

theorem inequality_solution_set : 
  { x : ℝ | |x + 1| + |x + 3| < 4 } = { x : ℝ | -4 < x ∧ x < 0 } :=
sorry

theorem inequality_proof_2 (a b : ℝ) (ha : -4 < a) (ha' : a < 0) (hb : -4 < b) (hb' : b < 0) : 
  2 * |a - b| < |a * b + 2 * a + 2 * b| :=
sorry

end inequality_solution_set_inequality_proof_2_l98_98255


namespace greatest_possible_perimeter_l98_98384

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l98_98384


namespace mysterious_neighbor_is_13_l98_98217

variable (x : ℕ) (h1 : x < 15) (h2 : 2 * x * 30 = 780)

theorem mysterious_neighbor_is_13 : x = 13 :=
by {
    sorry 
}

end mysterious_neighbor_is_13_l98_98217


namespace paper_string_area_l98_98687

theorem paper_string_area (side len overlap : ℝ) (n : ℕ) (h_side : side = 30) 
                          (h_len : len = 30) (h_overlap : overlap = 7) (h_n : n = 6) :
  let area_one_sheet := side * len
  let effective_len := side - overlap
  let total_length := len + effective_len * (n - 1)
  let width := side
  let area := total_length * width
  area = 4350 := 
by
  sorry

end paper_string_area_l98_98687


namespace positive_difference_is_zero_l98_98401

-- Definitions based on conditions
def jo_sum (n : ℕ) : ℕ := (n * (n + 1)) / 2

def rounded_to_nearest_5 (x : ℕ) : ℕ :=
  if x % 5 = 0 then x
  else (x / 5) * 5 + (if x % 5 >= 3 then 5 else 0)

def alan_sum (n : ℕ) : ℕ :=
  (List.range (n + 1)).map rounded_to_nearest_5 |>.sum

-- Theorem based on question and correct answer
theorem positive_difference_is_zero :
  jo_sum 120 - alan_sum 120 = 0 := sorry

end positive_difference_is_zero_l98_98401


namespace rationalize_sqrt_fraction_l98_98836

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l98_98836


namespace science_club_officers_l98_98001

-- Definitions of the problem conditions
def num_members : ℕ := 25
def num_officers : ℕ := 3
def alice : ℕ := 1 -- unique identifier for Alice
def bob : ℕ := 2 -- unique identifier for Bob

-- Main theorem statement
theorem science_club_officers :
  ∃ (ways_to_choose_officers : ℕ), ways_to_choose_officers = 10764 :=
  sorry

end science_club_officers_l98_98001


namespace inner_cube_surface_area_l98_98168

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l98_98168


namespace domain_of_f_l98_98733

noncomputable def f (x : ℝ) : ℝ := (5 * x + 2) / Real.sqrt (2 * x - 10)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f y = f x} = {x : ℝ | x > 5} :=
by
  sorry

end domain_of_f_l98_98733


namespace petya_wins_second_race_l98_98660

theorem petya_wins_second_race 
  (v_P v_V : ℝ) -- Petya's and Vasya's speeds
  (h1 : v_V = 0.9 * v_P) -- Condition from the first race
  (d_P d_V : ℝ) -- Distances covered by Petya and Vasya in the first race
  (h2 : d_P = 100) -- Petya covers 100 meters
  (h3 : d_V = 90) -- Vasya covers 90 meters
  (start_diff : ℝ) -- Initial distance difference in the second race
  (h4 : start_diff = 10) -- Petya starts 10 meters behind Vasya
  (race_length : ℝ) -- Total race length
  (h5 : race_length = 100) -- The race is 100 meters long
  : (v_P * (race_length / v_P) - v_V * (race_length / v_P)) = 1 :=
by
  sorry

end petya_wins_second_race_l98_98660


namespace conic_sections_l98_98583

theorem conic_sections (x y : ℝ) :
  y^4 - 9 * x^4 = 3 * y^2 - 4 →
  (∃ c : ℝ, (c = 5/2 ∨ c = 1) ∧ y^2 - 3 * x^2 = c) :=
by
  sorry

end conic_sections_l98_98583


namespace coprime_sum_product_l98_98646

theorem coprime_sum_product (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (a + b) (a * b) = 1 := by
  sorry

end coprime_sum_product_l98_98646


namespace inner_cube_surface_area_l98_98152

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l98_98152


namespace garment_industry_initial_men_l98_98212

theorem garment_industry_initial_men (M : ℕ) :
  (M * 8 * 10 = 6 * 20 * 8) → M = 12 :=
by
  sorry

end garment_industry_initial_men_l98_98212


namespace total_subjects_l98_98655

theorem total_subjects (m : ℕ) (k : ℕ) (j : ℕ) (h1 : m = 10) (h2 : k = m + 4) (h3 : j = k + 3) : m + k + j = 41 :=
by
  -- Ignoring proof as per instruction
  sorry

end total_subjects_l98_98655


namespace solve_n_m_l98_98745

noncomputable def exponents_of_linear_equation (n m : ℕ) (x y : ℝ) : Prop :=
2 * x ^ (n - 3) - (1 / 3) * y ^ (2 * m + 1) = 0

theorem solve_n_m (n m : ℕ) (x y : ℝ) (h_linear : exponents_of_linear_equation n m x y) :
  n ^ m = 1 :=
sorry

end solve_n_m_l98_98745


namespace unique_solution_to_equation_l98_98593

theorem unique_solution_to_equation (x y z t : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (ht : t > 0) 
  (h : 1 + 5^x = 2^y + 2^z * 5^t) : (x, y, z, t) = (2, 4, 1, 1) := 
sorry

end unique_solution_to_equation_l98_98593


namespace probabilities_equal_l98_98764

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l98_98764


namespace complex_number_solution_l98_98590

theorem complex_number_solution (a b : ℝ) (z : ℂ) :
  z = a + b * I →
  (a - 2) ^ 2 + b ^ 2 = 25 →
  (a + 4) ^ 2 + b ^ 2 = 25 →
  a ^ 2 + (b - 2) ^ 2 = 25 →
  z = -1 - 4 * I :=
sorry

end complex_number_solution_l98_98590


namespace rationalize_denominator_l98_98808

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l98_98808


namespace fewest_coach_handshakes_l98_98268

noncomputable def binom (n : ℕ) := n * (n - 1) / 2

theorem fewest_coach_handshakes : 
  ∃ (k1 k2 k3 : ℕ), binom 43 + k1 + k2 + k3 = 903 ∧ k1 + k2 + k3 = 0 := 
by
  use 0, 0, 0
  sorry

end fewest_coach_handshakes_l98_98268


namespace problem_1_problem_2_l98_98470

section proof_problem

variables (a b c d : ℤ)
variables (op : ℤ → ℤ → ℤ)
variables (add : ℤ → ℤ → ℤ)

-- Define the given conditions
axiom op_idem : ∀ (a : ℤ), op a a = a
axiom op_zero : ∀ (a : ℤ), op a 0 = 2 * a
axiom op_add : ∀ (a b c d : ℤ), add (op a b) (op c d) = op (a + c) (b + d)

-- Define the problems to prove
theorem problem_1 : add (op 2 3) (op 0 3) = -2 := sorry
theorem problem_2 : op 1024 48 = 2000 := sorry

end proof_problem

end problem_1_problem_2_l98_98470


namespace roof_shingle_width_l98_98265

theorem roof_shingle_width (L A W : ℕ) (hL : L = 10) (hA : A = 70) (hArea : A = L * W) : W = 7 :=
by
  sorry

end roof_shingle_width_l98_98265


namespace hexagon_pillar_height_l98_98216

noncomputable def height_of_pillar_at_vertex_F (s : ℝ) (hA hB hC : ℝ) (A : ℝ × ℝ) : ℝ :=
  10

theorem hexagon_pillar_height :
  ∀ (s hA hB hC : ℝ) (A : ℝ × ℝ),
  s = 8 ∧ hA = 15 ∧ hB = 10 ∧ hC = 12 ∧ A = (3, 3 * Real.sqrt 3) →
  height_of_pillar_at_vertex_F s hA hB hC A = 10 := by
  sorry

end hexagon_pillar_height_l98_98216


namespace trapezoid_height_ratios_l98_98675

theorem trapezoid_height_ratios (A B C D O M N K L : ℝ) (h : ℝ) (h_AD : D = 2 * B) 
  (h_OK : K = h / 3) (h_OL : L = (2 * h) / 3) :
  (K / h = 1 / 3) ∧ (L / h = 2 / 3) := by
  sorry

end trapezoid_height_ratios_l98_98675


namespace geometric_sequence_a8_l98_98219

theorem geometric_sequence_a8 (a : ℕ → ℝ) (q : ℝ) 
  (h₁ : a 3 = 3)
  (h₂ : a 6 = 24)
  (h₃ : ∀ n, a (n + 1) = a n * q) : 
  a 8 = 96 :=
by
  sorry

end geometric_sequence_a8_l98_98219


namespace rachel_age_when_emily_half_age_l98_98281

-- Conditions
def Emily_current_age : ℕ := 20
def Rachel_current_age : ℕ := 24

-- Proof statement
theorem rachel_age_when_emily_half_age :
  ∃ x : ℕ, (Emily_current_age - x = (Rachel_current_age - x) / 2) ∧ (Rachel_current_age - x = 8) := 
sorry

end rachel_age_when_emily_half_age_l98_98281


namespace min_value_x_squared_plus_y_squared_plus_z_squared_l98_98034

theorem min_value_x_squared_plus_y_squared_plus_z_squared (x y z : ℝ) (h : x + y + z = 1) : 
  x^2 + y^2 + z^2 ≥ 1/3 :=
by
  sorry

end min_value_x_squared_plus_y_squared_plus_z_squared_l98_98034


namespace shapes_fit_exactly_l98_98843

-- Conditions: Shapes are drawn on a piece of paper and folded along a central bold line
def shapes_drawn_on_paper := true
def paper_folded_along_central_line := true

-- Define the main proof problem
theorem shapes_fit_exactly : shapes_drawn_on_paper ∧ paper_folded_along_central_line → 
  number_of_shapes_fitting_exactly_on_top = 3 :=
by
  intros h
  sorry

end shapes_fit_exactly_l98_98843


namespace greatest_possible_perimeter_l98_98359

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l98_98359


namespace no_valid_n_lt_200_l98_98316

noncomputable def roots_are_consecutive (n m : ℕ) : Prop :=
  ∃ k : ℕ, m = k * (k + 1) ∧ n = 2 * k + 1

theorem no_valid_n_lt_200 :
  ¬∃ n m : ℕ, n < 200 ∧
              m % 4 = 0 ∧
              ∃ t : ℕ, t^2 = m ∧
              roots_are_consecutive n m := 
by
  sorry

end no_valid_n_lt_200_l98_98316


namespace simplify_expression_l98_98667

theorem simplify_expression (n : ℕ) : 
  (2^(n+5) - 3 * 2^n) / (3 * 2^(n+3)) = 29 / 24 :=
by sorry

end simplify_expression_l98_98667


namespace inequality_correct_l98_98758

-- Theorem: For all real numbers x and y, if x ≥ y, then x² + y² ≥ 2xy.
theorem inequality_correct (x y : ℝ) (h : x ≥ y) : x^2 + y^2 ≥ 2 * x * y := 
by {
  -- Placeholder for the proof
  sorry
}

end inequality_correct_l98_98758


namespace product_not_divisible_by_770_l98_98230

theorem product_not_divisible_by_770 (a b : ℕ) (h : a + b = 770) : ¬ (a * b) % 770 = 0 :=
sorry

end product_not_divisible_by_770_l98_98230


namespace greatest_possible_perimeter_l98_98363

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l98_98363


namespace cube_volume_l98_98117

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98117


namespace total_dresses_l98_98181

theorem total_dresses (E M D S: ℕ) 
  (h1 : D = M + 12)
  (h2 : M = E / 2)
  (h3 : E = 16)
  (h4 : S = D - 5) : 
  E + M + D + S = 59 :=
by
  sorry

end total_dresses_l98_98181


namespace greatest_perimeter_of_triangle_l98_98341

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l98_98341


namespace transmission_time_l98_98729

theorem transmission_time :
  let regular_blocks := 70
  let large_blocks := 30
  let chunks_per_regular_block := 800
  let chunks_per_large_block := 1600
  let channel_rate := 200
  let total_chunks := (regular_blocks * chunks_per_regular_block) + (large_blocks * chunks_per_large_block)
  let total_time_seconds := total_chunks / channel_rate
  let total_time_minutes := total_time_seconds / 60
  total_time_minutes = 8.67 := 
by 
  sorry

end transmission_time_l98_98729


namespace cube_volume_from_surface_area_l98_98057

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98057


namespace number_of_hours_sold_l98_98718

def packs_per_hour_peak := 6
def packs_per_hour_low := 4
def price_per_pack := 60
def extra_revenue := 1800

def revenue_per_hour_peak := packs_per_hour_peak * price_per_pack
def revenue_per_hour_low := packs_per_hour_low * price_per_pack
def revenue_diff_per_hour := revenue_per_hour_peak - revenue_per_hour_low

theorem number_of_hours_sold (h : ℕ) 
  (h_eq : revenue_diff_per_hour * h = extra_revenue) : 
  h = 15 :=
by
  -- skip proof
  sorry

end number_of_hours_sold_l98_98718


namespace roots_of_transformed_quadratic_l98_98210

variable {a b c : ℝ}

theorem roots_of_transformed_quadratic
    (h₁: a ≠ 0)
    (h₂: ∀ x, a * (x - 1)^2 - 1 = ax^2 + bx + c - 1)
    (h₃: ax^2 + bx + c = -1) :
    (x = 1) ∧ (x = 1) := 
  sorry

end roots_of_transformed_quadratic_l98_98210


namespace rectangle_breadth_approx_1_1_l98_98545

theorem rectangle_breadth_approx_1_1 (s b : ℝ) (h1 : 4 * s = 2 * (16 + b))
  (h2 : abs ((π * s / 2) + s - 21.99) < 0.01) : abs (b - 1.1) < 0.01 :=
sorry

end rectangle_breadth_approx_1_1_l98_98545


namespace taxes_taken_out_l98_98517

theorem taxes_taken_out
  (gross_pay : ℕ)
  (retirement_percentage : ℝ)
  (net_pay_after_taxes : ℕ)
  (tax_amount : ℕ) :
  gross_pay = 1120 →
  retirement_percentage = 0.25 →
  net_pay_after_taxes = 740 →
  tax_amount = gross_pay - (gross_pay * retirement_percentage) - net_pay_after_taxes :=
by
  sorry

end taxes_taken_out_l98_98517


namespace count_integers_with_factors_l98_98315

theorem count_integers_with_factors (x y z : ℕ) (h1 : y > x) (h2 : x > 0) :
  let lcm_val := lcm 16 9 in
  let min_val := Nat.ceil (z / lcm_val) * lcm_val in
  let max_val := Nat.floor (y / lcm_val) * lcm_val in
  let count := ((max_val / lcm_val) - (min_val / lcm_val) + 1) in
  (count = 2) :=
by
  sorry

end count_integers_with_factors_l98_98315


namespace find_smallest_n_l98_98430

-- Definitions of the condition that m and n are relatively prime and that the fraction includes the digits 4, 5, and 6 consecutively
def is_coprime (m n : ℕ) : Prop := Nat.gcd m n = 1

def has_digits_456 (m n : ℕ) : Prop := 
  ∃ k : ℕ, ∃ c : ℕ, 10^k * m % (10^k * n) = 456 * 10^c

-- The theorem to prove the smallest value of n
theorem find_smallest_n (m n : ℕ) (h1 : is_coprime m n) (h2 : m < n) (h3 : has_digits_456 m n) : n = 230 :=
sorry

end find_smallest_n_l98_98430


namespace initial_men_is_250_l98_98260

-- Define the given conditions
def provisions (initial_men remaining_men initial_days remaining_days : ℕ) : Prop :=
  initial_men * initial_days = remaining_men * remaining_days

-- Define the problem statement
theorem initial_men_is_250 (initial_days remaining_days : ℕ) (remaining_men_leaving : ℕ) :
  provisions initial_men (initial_men - remaining_men_leaving) initial_days remaining_days → initial_men = 250 :=
by
  intros h
  -- Requirement to solve the theorem.
  -- This is where the proof steps would go, but we put sorry to satisfy the statement requirement.
  sorry

end initial_men_is_250_l98_98260


namespace conical_tank_volume_l98_98271

theorem conical_tank_volume
  (diameter : ℝ) (height : ℝ) (depth_linear : ∀ x : ℝ, 0 ≤ x ∧ x ≤ diameter / 2 → height - (height / (diameter / 2)) * x = 0) :
  diameter = 20 → height = 6 → (1 / 3) * Real.pi * (10 ^ 2) * height = 200 * Real.pi :=
by
  sorry

end conical_tank_volume_l98_98271


namespace volume_of_cube_l98_98099

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98099


namespace largest_divisible_l98_98249

theorem largest_divisible (n : ℕ) (h1 : n > 0) (h2 : (n^3 + 200) % (n - 8) = 0) : n = 5376 :=
by
  sorry

end largest_divisible_l98_98249


namespace add_decimals_l98_98174

theorem add_decimals :
  0.0935 + 0.007 + 0.2 = 0.3005 :=
by sorry

end add_decimals_l98_98174


namespace remainder_of_nonempty_disjoint_subsets_l98_98958

theorem remainder_of_nonempty_disjoint_subsets (T : Set ℕ) (hT : T = {1, 2, 3, ..., 12}) :
  let m := (3 ^ 12 - 2 * 2 ^ 12 + 1) / 2 in
  m % 1000 = 125 := 
by
  sorry

end remainder_of_nonempty_disjoint_subsets_l98_98958


namespace tips_fraction_l98_98633

variable (S : ℚ)

def week1_tips : ℚ := 11 / 4 * S
def week2_tips : ℚ := 7 / 3 * S
def total_income_salary : ℚ := 2 * S
def total_tips : ℚ := week1_tips S + week2_tips S
def total_income : ℚ := total_income_salary S + total_tips S

theorem tips_fraction :
  total_tips S / total_income S = 61 / 85 :=
by
  sorry

end tips_fraction_l98_98633


namespace second_cube_surface_area_l98_98129

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l98_98129


namespace katie_pink_marbles_l98_98519

-- Define variables for the problem
variables (P O R : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  O = P - 9 ∧
  R = 4 * (P - 9) ∧
  P + O + R = 33

-- Desired result
def result : Prop :=
  P = 13

-- Proof statement
theorem katie_pink_marbles : conditions P O R → result P :=
by
  intros h
  sorry

end katie_pink_marbles_l98_98519


namespace minimum_value_a_plus_b_plus_c_l98_98743

theorem minimum_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 4 * b + 7 * c ≤ 2 * a * b * c) : a + b + c ≥ 15 / 2 :=
by
  sorry

end minimum_value_a_plus_b_plus_c_l98_98743


namespace translated_upwards_2_units_l98_98859

theorem translated_upwards_2_units (x : ℝ) : (x + 2 > 0) → (x > -2) :=
by 
  intros h
  exact sorry

end translated_upwards_2_units_l98_98859


namespace cube_volume_l98_98082

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98082


namespace students_arrangement_l98_98027

theorem students_arrangement (B1 B2 S1 S2 T1 T2 C1 C2 : ℕ) :
  (B1 = B2 ∧ S1 ≠ S2 ∧ T1 ≠ T2 ∧ C1 ≠ C2) →
  (C1 ≠ C2) →
  (arrangements = 7200) :=
by
  sorry

end students_arrangement_l98_98027


namespace minimum_value_l98_98642

theorem minimum_value {a b c : ℝ} (h_pos: 0 < a ∧ 0 < b ∧ 0 < c) (h_eq: a * b * c = 1 / 2) :
  ∃ x, x = a^2 + 4 * a * b + 9 * b^2 + 8 * b * c + 3 * c^2 ∧ x = 13.5 :=
sorry

end minimum_value_l98_98642


namespace max_perimeter_l98_98391

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l98_98391


namespace cosine_120_eq_neg_one_half_l98_98444

theorem cosine_120_eq_neg_one_half : Real.cos (120 * Real.pi / 180) = -1/2 :=
by
-- Proof omitted
sorry

end cosine_120_eq_neg_one_half_l98_98444


namespace significant_digits_of_side_length_l98_98182

noncomputable def num_significant_digits (n : Float) : Nat :=
  -- This is a placeholder function to determine the number of significant digits
  sorry

theorem significant_digits_of_side_length :
  ∀ (A : Float), A = 3.2400 → num_significant_digits (Float.sqrt A) = 5 :=
by
  intro A h
  -- Proof would go here
  sorry

end significant_digits_of_side_length_l98_98182


namespace next_perfect_square_l98_98851

theorem next_perfect_square (x : ℤ) (h : ∃ k : ℤ, x = k^2) : ∃ z : ℤ, z = x + 2 * Int.sqrt x + 1 :=
by
  sorry

end next_perfect_square_l98_98851


namespace inequality_holds_for_all_real_numbers_l98_98623

theorem inequality_holds_for_all_real_numbers (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + k * x - 3 / 4 < 0) ↔ (k ∈ Set.Icc (-3 : ℝ) 0) := 
sorry

end inequality_holds_for_all_real_numbers_l98_98623


namespace sin_alpha_plus_2beta_l98_98938

theorem sin_alpha_plus_2beta
  (α β : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hcosalpha_plus_beta : Real.cos (α + β) = -5 / 13)
  (h sinbeta : Real.sin β = 3 / 5) :
  Real.sin (α + 2 * β) = 33 / 65 :=
  sorry

end sin_alpha_plus_2beta_l98_98938


namespace molecular_weight_8_moles_Al2O3_l98_98029

noncomputable def molecular_weight_Al2O3 (atomic_weight_Al : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  2 * atomic_weight_Al + 3 * atomic_weight_O

theorem molecular_weight_8_moles_Al2O3
  (atomic_weight_Al : ℝ := 26.98)
  (atomic_weight_O : ℝ := 16.00)
  : molecular_weight_Al2O3 atomic_weight_Al atomic_weight_O * 8 = 815.68 := by
  sorry

end molecular_weight_8_moles_Al2O3_l98_98029


namespace no_solution_ineq_range_a_l98_98978

theorem no_solution_ineq_range_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + 4 < 0 → false) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end no_solution_ineq_range_a_l98_98978


namespace min_value_of_expression_l98_98596

variable (a b : ℝ)

theorem min_value_of_expression (h : b ≠ 0) : 
  ∃ (a b : ℝ), (a^2 + b^2 + a / b + 1 / b^2) = Real.sqrt 3 :=
sorry

end min_value_of_expression_l98_98596


namespace net_profit_calc_l98_98436

theorem net_profit_calc (purchase_price : ℕ) (overhead_percentage : ℝ) (markup : ℝ) 
  (h_pp : purchase_price = 48) (h_op : overhead_percentage = 0.10) (h_markup : markup = 35) :
  let overhead := overhead_percentage * purchase_price
  let net_profit := markup - overhead
  net_profit = 30.20 := by
    sorry

end net_profit_calc_l98_98436


namespace sum_of_first_five_terms_l98_98308

noncomputable def S₅ (a : ℕ → ℝ) := (a 1 + a 5) / 2 * 5

theorem sum_of_first_five_terms (a : ℕ → ℝ) (a_2 a_4 : ℝ)
  (h1 : a 2 = 4)
  (h2 : a 4 = 2)
  (h3 : ∀ n : ℕ, a n = a 1 + (n - 1) * (a 2 - a 1)) :
  S₅ a = 15 :=
sorry

end sum_of_first_five_terms_l98_98308


namespace greatest_perimeter_l98_98352

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l98_98352


namespace carpet_needed_for_room_l98_98709

theorem carpet_needed_for_room
  (length_feet : ℕ) (width_feet : ℕ)
  (area_conversion_factor : ℕ)
  (length_given : length_feet = 12)
  (width_given : width_feet = 6)
  (conversion_given : area_conversion_factor = 9) :
  (length_feet * width_feet) / area_conversion_factor = 8 := 
by
  sorry

end carpet_needed_for_room_l98_98709


namespace trips_needed_to_fill_pool_l98_98897

def caleb_gallons_per_trip : ℕ := 7
def cynthia_gallons_per_trip : ℕ := 8
def pool_capacity : ℕ := 105

theorem trips_needed_to_fill_pool : (pool_capacity / (caleb_gallons_per_trip + cynthia_gallons_per_trip) = 7) :=
by
  sorry

end trips_needed_to_fill_pool_l98_98897


namespace correct_calculation_l98_98702

theorem correct_calculation (x : ℕ) (h : 637 = x + 238) : x - 382 = 17 :=
by
  sorry

end correct_calculation_l98_98702


namespace emma_age_l98_98658

variables (O N L E : ℕ)

def oliver_eq : Prop := O = N - 5
def nancy_eq : Prop := N = L + 6
def emma_eq : Prop := E = L + 4
def oliver_age : Prop := O = 16

theorem emma_age :
  oliver_eq O N ∧ nancy_eq N L ∧ emma_eq E L ∧ oliver_age O → E = 19 :=
by
  sorry

end emma_age_l98_98658


namespace race_length_l98_98269

theorem race_length (covered_meters remaining_meters race_length : ℕ)
  (h_covered : covered_meters = 721)
  (h_remaining : remaining_meters = 279)
  (h_race_length : race_length = covered_meters + remaining_meters) :
  race_length = 1000 :=
by
  rw [h_covered, h_remaining] at h_race_length
  exact h_race_length

end race_length_l98_98269


namespace greatest_triangle_perimeter_l98_98377

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l98_98377


namespace simplify_fraction_l98_98666

-- Define the original expressions
def expr1 := 3 / (Real.sqrt 5 + 2)
def expr2 := 4 / (Real.sqrt 7 - 2)

-- State the mathematical problem.
theorem simplify_fraction :
  (1 / (expr1 + expr2)) =
  ((9 * Real.sqrt 5 + 4 * Real.sqrt 7 + 10) / 
  ((9 * Real.sqrt 5 + 4 * Real.sqrt 7) ^ 2 - 100)) :=
by sorry

end simplify_fraction_l98_98666


namespace greatest_possible_perimeter_l98_98333

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l98_98333


namespace find_g_2022_l98_98588

def g : ℝ → ℝ := sorry -- This is pre-defined to say there exists such a function

theorem find_g_2022 (g : ℝ → ℝ)
  (h : ∀ x y : ℝ, g (x - y) = g x + g y - 2021 * (x + y)) :
  g 2022 = 4086462 :=
sorry

end find_g_2022_l98_98588


namespace evaluate_expression_l98_98534

noncomputable def expression (a : ℚ) : ℚ := 
  (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2 * a)

theorem evaluate_expression (a : ℚ) (ha : a = -1/3) : expression a = -2 :=
by 
  rw [expression, ha]
  sorry

end evaluate_expression_l98_98534


namespace point_transformation_l98_98606

theorem point_transformation : ∀ (P : ℝ×ℝ), P = (1, -2) → P = (-1, 2) :=
by
  sorry

end point_transformation_l98_98606


namespace dodgeball_tournament_l98_98525

theorem dodgeball_tournament (N : ℕ) (points : ℕ) :
  points = 1151 →
  (∀ {G : ℕ}, G = N * (N - 1) / 2 →
    (∃ (win_points loss_points tie_points : ℕ), 
      win_points = 15 * (N * (N - 1) / 2 - tie_points) ∧ 
      tie_points = 11 * tie_points ∧ 
      points = win_points + tie_points + loss_points)) → 
  N = 12 :=
by
  intro h_points h_games
  sorry

end dodgeball_tournament_l98_98525


namespace events_A_B_equal_prob_l98_98767

variable {u j p b : ℝ}

-- Define the conditions
axiom u_gt_j : u > j
axiom b_gt_p : b > p

noncomputable def prob_event_A : ℝ :=
  (u / (u + p) * (b / (u + b))) * (j / (j + b) * (p / (j + p)))

noncomputable def prob_event_B : ℝ :=
  (u / (u + b) * (p / (u + p))) * (j / (j + p) * (b / (j + b)))

-- Statement of the problem
theorem events_A_B_equal_prob :
  prob_event_A = prob_event_B :=
  by
    sorry

end events_A_B_equal_prob_l98_98767


namespace area_difference_equal_28_5_l98_98487

noncomputable def square_side_length (d: ℝ) : ℝ := d / Real.sqrt 2
noncomputable def square_area (d: ℝ) : ℝ := (square_side_length d) ^ 2
noncomputable def circle_radius (D: ℝ) : ℝ := D / 2
noncomputable def circle_area (D: ℝ) : ℝ := Real.pi * (circle_radius D) ^ 2
noncomputable def area_difference (d D : ℝ) : ℝ := |circle_area D - square_area d|

theorem area_difference_equal_28_5 :
  ∀ (d D : ℝ), d = 10 → D = 10 → area_difference d D = 28.5 :=
by
  intros d D hd hD
  rw [hd, hD]
  -- Remaining steps involve computing the known areas and their differences
  sorry

end area_difference_equal_28_5_l98_98487


namespace total_amount_l98_98872

noncomputable def A : ℝ := 396.00000000000006
noncomputable def B : ℝ := A * (3 / 2)
noncomputable def C : ℝ := B * 4

theorem total_amount (A_eq : A = 396.00000000000006) (A_B_relation : A = (2 / 3) * B) (B_C_relation : B = (1 / 4) * C) :
  396.00000000000006 + B + C = 3366.000000000001 := by
  sorry

end total_amount_l98_98872


namespace tobias_downloads_l98_98855

theorem tobias_downloads : 
  ∀ (m : ℕ), (∀ (price_per_app total_spent : ℝ), 
  price_per_app = 2.00 + 2.00 * 0.10 ∧ 
  total_spent = 52.80 → 
  m = total_spent / price_per_app) → 
  m = 24 := 
  sorry

end tobias_downloads_l98_98855


namespace cards_exchanged_l98_98719

theorem cards_exchanged (x : ℕ) (h : x * (x - 1) = 1980) : x * (x - 1) = 1980 :=
by sorry

end cards_exchanged_l98_98719


namespace greatest_possible_perimeter_l98_98369

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l98_98369


namespace xyz_equality_l98_98965

theorem xyz_equality (x y z : ℝ) (h : x^2 + y^2 + z^2 = x * y + y * z + z * x) : x = y ∧ y = z :=
by
  sorry

end xyz_equality_l98_98965


namespace cube_volume_l98_98077

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98077


namespace find_a_l98_98754

-- Define the sets A and B and their union
variables (a : ℕ)
def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a^2}
def C : Set ℕ := {0, 1, 2, 3, 9}

-- Define the condition and prove that it implies a = 3
theorem find_a (h : A a ∪ B a = C) : a = 3 := 
by
  sorry

end find_a_l98_98754


namespace cube_volume_l98_98072

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98072


namespace parabola_focus_directrix_distance_l98_98705

theorem parabola_focus_directrix_distance {a : ℝ} (h₀ : a > 0):
  (∃ (b : ℝ), ∃ (x1 x2 : ℝ), (x1 + x2 = 1 / a) ∧ (1 / (2 * a) = 1)) → 
  (1 / (2 * a) / 2 = 1 / 4) :=
by
  sorry

end parabola_focus_directrix_distance_l98_98705


namespace div_sqrt3_mul_inv_sqrt3_eq_one_l98_98466

theorem div_sqrt3_mul_inv_sqrt3_eq_one :
  (3 / Real.sqrt 3) * (1 / Real.sqrt 3) = 1 :=
by
  sorry

end div_sqrt3_mul_inv_sqrt3_eq_one_l98_98466


namespace factor_squared_l98_98953

-- Define P(x, y) as a polynomial in two variables satisfying the given conditions
def polynomial_symmetric (P : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ) : Prop :=
  ∀ x y : ℝ, P x y = P y x

def is_factor (f g : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ) : Prop :=
  ∃ h : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ, g = λ x y, f x y * h x y

-- The theorem to prove that (x - y)^2 is a factor of P(x, y)
theorem factor_squared (P : Polynomial ℝ → Polynomial ℝ → Polynomial ℝ) :
  polynomial_symmetric P →
  is_factor (λ x y, Polynomial.C (x - y)) P →
  is_factor (λ x y, Polynomial.C (x - y)^2) P :=
by
  intro h_symm h_factor
  sorry

end factor_squared_l98_98953


namespace remainder_of_173_mod_13_l98_98993

theorem remainder_of_173_mod_13 : ∀ (m : ℤ), 173 = 8 * m + 5 → 173 < 180 → 173 % 13 = 4 :=
by
  intro m hm h
  sorry

end remainder_of_173_mod_13_l98_98993


namespace find_a_l98_98609

noncomputable def f (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + Real.log x

theorem find_a (h_max : ∃ (x : Set.Icc (0 : ℝ) 1), f (-Real.exp 1) x = -1) : 
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≤ -1) → a = -Real.exp 1 :=
sorry

end find_a_l98_98609


namespace greatest_possible_perimeter_l98_98330

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l98_98330


namespace trigonometric_identity_proof_l98_98981

theorem trigonometric_identity_proof
  (α : Real)
  (h1 : Real.sin (Real.pi + α) = -Real.sin α)
  (h2 : Real.cos (Real.pi + α) = -Real.cos α)
  (h3 : Real.cos (-α) = Real.cos α)
  (h4 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) :
  Real.sin (Real.pi + α) ^ 2 - Real.cos (Real.pi + α) * Real.cos (-α) + 1 = 2 := 
by
  sorry

end trigonometric_identity_proof_l98_98981


namespace maximum_value_of_transformed_function_l98_98543

theorem maximum_value_of_transformed_function (a b : ℝ) (h_max : ∀ x : ℝ, a * (Real.cos x) + b ≤ 1)
  (h_min : ∀ x : ℝ, a * (Real.cos x) + b ≥ -7) : 
  ∃ ab : ℝ, (ab = 3 + a * b * (Real.sin x)) ∧ (∀ x : ℝ, ab ≤ 15) :=
by
  sorry

end maximum_value_of_transformed_function_l98_98543


namespace zero_sum_of_squares_l98_98796

theorem zero_sum_of_squares {a b : ℝ} (h : a^2 + b^2 = 0) : a = 0 ∧ b = 0 :=
sorry

end zero_sum_of_squares_l98_98796


namespace sequence_is_increasing_l98_98484

-- Define the sequence recurrence property
def sequence_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) - a n = 3

-- The theorem statement
theorem sequence_is_increasing (a : ℕ → ℤ) (h : sequence_condition a) : 
  ∀ n : ℕ, a n < a (n + 1) :=
by
  unfold sequence_condition at h
  intro n
  specialize h n
  sorry

end sequence_is_increasing_l98_98484


namespace find_complex_number_z_l98_98488

-- Given the complex number z and the equation \(\frac{z}{1+i} = i^{2015} + i^{2016}\)
-- prove that z = -2i
theorem find_complex_number_z (z : ℂ) (h : z / (1 + (1 : ℂ) * I) = I ^ 2015 + I ^ 2016) : z = -2 * I := 
by
  sorry

end find_complex_number_z_l98_98488


namespace max_triangle_perimeter_l98_98345

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l98_98345


namespace Panthers_total_games_l98_98464

/-
Given:
1) The Panthers had won 60% of their basketball games before the district play.
2) During district play, they won four more games and lost four.
3) They finished the season having won half of their total games.
Prove that the total number of games they played in all is 48.
-/

theorem Panthers_total_games
  (y : ℕ) -- total games before district play
  (x : ℕ) -- games won before district play
  (h1 : x = 60 * y / 100) -- they won 60% of the games before district play
  (h2 : (x + 4) = 50 * (y + 8) / 100) -- they won half of the total games including district play
  : (y + 8) = 48 := -- total games they played in all
sorry

end Panthers_total_games_l98_98464


namespace shortest_altitude_triangle_l98_98880

/-- Given a triangle with sides 18, 24, and 30, prove that its shortest altitude is 18. -/
theorem shortest_altitude_triangle (a b c : ℝ) (h1 : a = 18) (h2 : b = 24) (h3 : c = 30) 
  (h_right : a ^ 2 + b ^ 2 = c ^ 2) : 
  exists h : ℝ, h = 18 :=
by
  sorry

end shortest_altitude_triangle_l98_98880


namespace constant_term_correct_l98_98564

theorem constant_term_correct:
    ∀ (a k n : ℤ), 
      (∀ x : ℤ, (3 * x + 2) * (2 * x - 3) = a * x^2 + k * x + n) 
      → a - n + k = 7 
      → n = -6 := 
by
    intros a k n h h2
    have h1 := h 0
    sorry

end constant_term_correct_l98_98564


namespace set_A_is_correct_l98_98185

open Complex

def A : Set ℤ := {x | ∃ n : ℕ, n > 0 ∧ x = (I ^ n + (-I) ^ n).re}

theorem set_A_is_correct : A = {-2, 0, 2} :=
sorry

end set_A_is_correct_l98_98185


namespace total_worth_of_stock_l98_98634

theorem total_worth_of_stock (x y : ℕ) (cheap_cost expensive_cost : ℝ) 
  (h1 : y = 21) (h2 : x + y = 22)
  (h3 : expensive_cost = 10) (h4 : cheap_cost = 2.5) :
  (x * expensive_cost + y * cheap_cost) = 62.5 :=
by
  sorry

end total_worth_of_stock_l98_98634


namespace second_cube_surface_area_l98_98130

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l98_98130


namespace lucy_final_balance_l98_98407

def initial_balance : ℝ := 65
def deposit : ℝ := 15
def withdrawal : ℝ := 4

theorem lucy_final_balance : initial_balance + deposit - withdrawal = 76 :=
by
  sorry

end lucy_final_balance_l98_98407


namespace inappropriate_expression_is_D_l98_98252

-- Definitions of each expression as constants
def expr_A : String := "Recently, I have had the honor to read your masterpiece, and I felt enlightened."
def expr_B : String := "Your visit has brought glory to my humble abode."
def expr_C : String := "It's the first time you honor my place with a visit, and I apologize for any lack of hospitality."
def expr_D : String := "My mother has been slightly unwell recently, I hope you won't bother her."

-- Definition of the problem context
def is_inappropriate (expr : String) : Prop := 
  expr = expr_D

-- The theorem statement
theorem inappropriate_expression_is_D : is_inappropriate expr_D := 
by
  sorry

end inappropriate_expression_is_D_l98_98252


namespace smallest_sector_angle_division_is_10_l98_98410

/-
  Prove that the smallest possible sector angle in a 15-sector division of a circle,
  where the central angles form an arithmetic sequence with integer values and the
  total sum of angles is 360 degrees, is 10 degrees.
-/
theorem smallest_sector_angle_division_is_10 :
  ∃ (a1 d : ℕ), (∀ i, i ∈ (List.range 15) → a1 + i * d > 0) ∧ (List.sum (List.map (fun i => a1 + i * d) (List.range 15)) = 360) ∧
  a1 = 10 := by
  sorry

end smallest_sector_angle_division_is_10_l98_98410


namespace sample_size_is_correct_l98_98711

-- Define the conditions
def num_classes := 40
def students_per_class := 50
def selected_students := 150

-- Define the statement to prove the sample size
theorem sample_size_is_correct : selected_students = 150 := by 
  -- Proof is skipped with sorry
  sorry

end sample_size_is_correct_l98_98711


namespace collinear_A₁_F_B_iff_q_eq_4_l98_98753

open Real

theorem collinear_A₁_F_B_iff_q_eq_4
  (m q : ℝ) (h_m : m ≠ 0)
  (A B : ℝ × ℝ)
  (h_A : 3 * (m * A.snd + q)^2 + 4 * A.snd^2 = 12)
  (h_B : 3 * (m * B.snd + q)^2 + 4 * B.snd^2 = 12)
  (A₁ : ℝ × ℝ := (A.fst, -A.snd))
  (F : ℝ × ℝ := (1, 0)) :
  ((q = 4) ↔ (∃ k : ℝ, k * (F.fst - A₁.fst) = F.snd - A₁.snd ∧ k * (B.fst - F.fst) = B.snd - F.snd)) :=
sorry

end collinear_A₁_F_B_iff_q_eq_4_l98_98753


namespace cube_volume_l98_98089

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98089


namespace bobby_initial_candy_l98_98180

theorem bobby_initial_candy (candy_ate_start candy_ate_more candy_left : ℕ)
  (h1 : candy_ate_start = 9) (h2 : candy_ate_more = 5) (h3 : candy_left = 8) :
  candy_ate_start + candy_ate_more + candy_left = 22 :=
by
  rw [h1, h2, h3]
  -- sorry


end bobby_initial_candy_l98_98180


namespace pentagram_coloring_equals_1020_l98_98985

-- Definitions and conditions for the problem
def colors := Finset.range 5
def vertices := Finset.fin 5

-- Prove that the number of valid colorings of the pentagram is 1020
theorem pentagram_coloring_equals_1020 :
  ∃ (f : Π v : vertices, colors),
    (∀ v₁ v₂ : vertices, (v₁ ≠ v₂ ∧ adjacent v₁ v₂) → f v₁ ≠ f v₂) ∧ -- adjacent vertices receive different colors
    finset.card {f : Π v : vertices, colors // -- the count of all such valid functions
    ∀ v₁ v₂ : vertices, (v₁ ≠ v₂ ∧ adjacent v₁ v₂) → f v₁ ≠ f v₂ }.to_finset = 1020 :=
sorry

end pentagram_coloring_equals_1020_l98_98985


namespace general_term_formula_l98_98945

noncomputable def Sn (n : ℕ) (a : ℝ) : ℝ := 3^n + a
noncomputable def an (n : ℕ) : ℝ := 2 * 3^(n-1)

theorem general_term_formula {a : ℝ} (n : ℕ) (h : Sn n a = 3^n + a) :
  Sn n a - Sn (n-1) a = an n :=
sorry

end general_term_formula_l98_98945


namespace cube_volume_l98_98113

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98113


namespace range_of_k_l98_98304

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, 1 < x → k * (Real.exp (k * x) + 1) - ((1 / x) + 1) * Real.log x > 0) ↔ k > 1 / Real.exp 1 := 
  sorry

end range_of_k_l98_98304


namespace cube_volume_l98_98106

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98106


namespace quadrilateral_angle_contradiction_l98_98557

theorem quadrilateral_angle_contradiction (a b c d : ℝ)
  (h : 0 < a ∧ a < 180 ∧ 0 < b ∧ b < 180 ∧ 0 < c ∧ c < 180 ∧ 0 < d ∧ d < 180)
  (sum_eq_360 : a + b + c + d = 360) :
  (¬ (a ≤ 90 ∨ b ≤ 90 ∨ c ≤ 90 ∨ d ≤ 90)) → (90 < a ∧ 90 < b ∧ 90 < c ∧ 90 < d) :=
sorry

end quadrilateral_angle_contradiction_l98_98557


namespace area_of_square_is_correct_l98_98294

-- Define the nature of the problem setup and parameters
def radius_of_circle : ℝ := 7
def diameter_of_circle : ℝ := 2 * radius_of_circle
def side_length_of_square : ℝ := 2 * diameter_of_circle
def area_of_square : ℝ := side_length_of_square ^ 2

-- Statement of the problem to prove
theorem area_of_square_is_correct : area_of_square = 784 := by
  sorry

end area_of_square_is_correct_l98_98294


namespace greatest_triangle_perimeter_l98_98378

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l98_98378


namespace toy_value_l98_98888

theorem toy_value (n : ℕ) (total_value special_toy_value : ℕ)
  (h₀ : n = 9) (h₁ : total_value = 52) (h₂ : special_toy_value = 12) :
  (total_value - special_toy_value) / (n - 1) = 5 :=
by
  have m : ℕ := n - 1
  have other_toys_value : ℕ := total_value - special_toy_value
  show other_toys_value / m = 5
  sorry

end toy_value_l98_98888


namespace boat_speed_still_water_l98_98046

theorem boat_speed_still_water : 
  ∀ (b s : ℝ), (b + s = 11) → (b - s = 5) → b = 8 := 
by 
  intros b s h1 h2
  sorry

end boat_speed_still_water_l98_98046


namespace total_clients_l98_98653

theorem total_clients (V K B N : Nat) (hV : V = 7) (hK : K = 8) (hB : B = 3) (hN : N = 18) :
    V + K - B + N = 30 := by
  sorry

end total_clients_l98_98653


namespace sqrt_of_4_eq_2_l98_98272

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_4_eq_2_l98_98272


namespace gumballs_per_box_l98_98202

-- Given conditions
def total_gumballs : ℕ := 20
def total_boxes : ℕ := 4

-- Mathematically equivalent proof problem
theorem gumballs_per_box:
  total_gumballs / total_boxes = 5 := by
  sorry

end gumballs_per_box_l98_98202


namespace find_w_l98_98538

variable (x y z w : ℝ)

theorem find_w (h : (x + y + z) / 3 = (y + z + w) / 3 + 10) : w = x - 30 := by 
  sorry

end find_w_l98_98538


namespace total_glass_area_l98_98950

theorem total_glass_area 
  (len₁ len₂ len₃ wid₁ wid₂ wid₃ : ℕ)
  (h₁ : len₁ = 30) (h₂ : wid₁ = 12)
  (h₃ : len₂ = 30) (h₄ : wid₂ = 12)
  (h₅ : len₃ = 20) (h₆ : wid₃ = 12) :
  (len₁ * wid₁ + len₂ * wid₂ + len₃ * wid₃) = 960 := 
by
  sorry

end total_glass_area_l98_98950


namespace simplify_expression_l98_98895

theorem simplify_expression :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2 / 3) = 7 + 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l98_98895


namespace rhombus_area_l98_98244

theorem rhombus_area : 
  ∃ (d1 d2 : ℝ), (∀ (x : ℝ), x^2 - 14 * x + 48 = 0 → x = d1 ∨ x = d2) ∧
  (∀ (A : ℝ), A = d1 * d2 / 2 → A = 24) :=
by 
sorry

end rhombus_area_l98_98244


namespace digit_agreement_l98_98849

theorem digit_agreement (N : ℕ) (abcd : ℕ) (h1 : N % 10000 = abcd) (h2 : N ^ 2 % 10000 = abcd) (h3 : ∃ a b c d, abcd = a * 1000 + b * 100 + c * 10 + d ∧ a ≠ 0) : abcd / 10 = 937 := sorry

end digit_agreement_l98_98849


namespace number_of_welders_left_l98_98050

-- Definitions for the given problem
def total_welders : ℕ := 36
def initial_days : ℝ := 1
def remaining_days : ℝ := 3.0000000000000004
def total_days : ℝ := 3

-- Condition equations
variable (r : ℝ) -- rate at which each welder works
variable (W : ℝ) -- total work

-- Equation representing initial total work
def initial_work : W = total_welders * r * total_days := by sorry

-- Welders who left for another project
variable (X : ℕ) -- number of welders who left

-- Equation representing remaining work
def remaining_work : (total_welders - X) * r * remaining_days = W - (total_welders * r * initial_days) := by sorry

-- Theorem to prove
theorem number_of_welders_left :
  (total_welders * total_days : ℝ) = W →
  (total_welders - X) * remaining_days = W - (total_welders * r * initial_days) →
  X = 12 :=
sorry

end number_of_welders_left_l98_98050


namespace arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l98_98301

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) :
  ∀ n : ℕ, a n = 5 - 2 * n :=
by
  sorry

theorem max_sum_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0)
  (h_a2 : a 2 = 1) (h_a5 : a 5 = -5) (h_sum : ∀ n : ℕ, S n = (n * (a 0 + a (n - 1))) / 2) :
  S 2 = 4 :=
by
  sorry

end arithmetic_sequence_general_formula_max_sum_arithmetic_sequence_l98_98301


namespace probability_two_doors_open_l98_98584

def binomial (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_two_doors_open :
  let total_doors := 5
  let total_combinations := 2 ^ total_doors
  let favorable_combinations := binomial total_doors 2
  let probability := favorable_combinations / total_combinations
  probability = 5 / 16 :=
by
  sorry

end probability_two_doors_open_l98_98584


namespace dave_pieces_l98_98726

theorem dave_pieces (boxes_bought : ℕ) (boxes_given : ℕ) (pieces_per_box : ℕ) 
  (h₁ : boxes_bought = 12) (h₂ : boxes_given = 5) (h₃ : pieces_per_box = 3) : 
  boxes_bought - boxes_given * pieces_per_box = 21 :=
by
  sorry

end dave_pieces_l98_98726


namespace quadratic_equation_m_value_l98_98786

-- Definition of the quadratic equation having exactly one solution with the given parameters
def quadratic_equation_has_one_solution (a b c : ℚ) : Prop :=
  b^2 - 4 * a * c = 0

-- Given constants in the problem
def a : ℚ := 3
def b : ℚ := -7

-- The value of m we aim to prove
def m_correct : ℚ := 49 / 12

-- The theorem stating the problem
theorem quadratic_equation_m_value (m : ℚ) (h : quadratic_equation_has_one_solution a b m) : m = m_correct :=
  sorry

end quadratic_equation_m_value_l98_98786


namespace prob_A_B_same_fee_expectation_η_l98_98392

noncomputable def rental_fee (hours: ℝ) : ℝ :=
if hours ≤ 2 then 0 else 2 * ⌈hours - 2⌉

def A_hours_distribution : ℕ → ℝ
| 2 := (1 / 4)
| 3 := (1 / 2)
| 4 := (1 / 4)
| _ := 0

def B_hours_distribution : ℕ → ℝ
| 2 := (1 / 2)
| 3 := (1 / 4)
| 4 := (1 / 4)
| _ := 0

def rental_fee_A (hours: ℕ) : ℝ :=
rental_fee hours

def rental_fee_B (hours: ℕ) : ℝ :=
rental_fee hours

def same_fee_prob : ℝ :=
(1 / 4) * (1 / 2) + 
(1 / 2) * (1 / 4) + 
(1 / 4) * (1 / 4)

theorem prob_A_B_same_fee : same_fee_prob = 5 / 16 := 
by sorry

def η_distribution : ℝ → ℝ
| 0 := (1 / 8)
| 2 := (5 / 16)
| 4 := (5 / 16)
| 6 := (3 / 16)
| 8 := (1 / 16)
| _ := 0

def E_η : ℝ :=
(5 / 16) * 2 + (5 / 16) * 4 + (3 / 16) * 6 + (1 / 16) * 8

theorem expectation_η : E_η = 7 / 2 :=
by sorry

end prob_A_B_same_fee_expectation_η_l98_98392


namespace second_cube_surface_area_l98_98127

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l98_98127


namespace num_ways_arrange_passengers_l98_98506

theorem num_ways_arrange_passengers 
  (seats : ℕ) (passengers : ℕ) (consecutive_empty : ℕ)
  (h1 : seats = 10) (h2 : passengers = 4) (h3 : consecutive_empty = 5) :
  ∃ ways, ways = 480 := by
  sorry

end num_ways_arrange_passengers_l98_98506


namespace volume_of_cube_l98_98102

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98102


namespace mrs_blue_expected_tomato_yield_l98_98226

-- Definitions for conditions
def steps_length := 3 -- each step measures 3 feet
def length_steps := 18 -- 18 steps in length
def width_steps := 25 -- 25 steps in width
def yield_per_sq_ft := 3 / 4 -- three-quarters of a pound per square foot

-- Define the total expected yield in pounds
def expected_yield : ℝ :=
  let length_ft := length_steps * steps_length
  let width_ft := width_steps * steps_length
  let area := length_ft * width_ft
  area * yield_per_sq_ft

-- The goal statement
theorem mrs_blue_expected_tomato_yield : expected_yield = 3037.5 := by
  sorry

end mrs_blue_expected_tomato_yield_l98_98226


namespace division_expression_result_l98_98977

theorem division_expression_result :
  -1 / (-5) / (-1 / 5) = -1 :=
by sorry

end division_expression_result_l98_98977


namespace real_roots_a_set_t_inequality_l98_98280

noncomputable def set_of_a : Set ℝ := {a | -1 ≤ a ∧ a ≤ 7}

theorem real_roots_a_set (x a : ℝ) :
  (∃ x, x^2 - 4 * x + abs (a - 3) = 0) ↔ a ∈ set_of_a := 
by
  sorry

theorem t_inequality (t a : ℝ) (h : ∀ a ∈ set_of_a, t^2 - 2 * a * t + 12 < 0) :
  3 < t ∧ t < 4 := 
by
  sorry

end real_roots_a_set_t_inequality_l98_98280


namespace max_triangle_perimeter_l98_98344

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l98_98344


namespace area_NPQ_l98_98631

noncomputable def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  (1 / 2) * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_NPQ :
  let X : ℝ × ℝ := (0, 0)
  let Y : ℝ × ℝ := (20, 0)
  let Z : ℝ × ℝ := (15.2, 13)
  let P : ℝ × ℝ := (10, 6.5) -- Circumcenter
  let Q : ℝ × ℝ := (7.7, 5.1) -- Incenter
  let N : ℝ × ℝ := (15.2, 9.5) -- Excircle center
  in triangle_area N P Q = 49.21 :=
by
  -- sorry is used to skip the proof.
  sorry

end area_NPQ_l98_98631


namespace cube_volume_l98_98118

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98118


namespace minimum_F_l98_98773

noncomputable def F (x : ℝ) : ℝ :=
  (1800 / (x + 5)) + (0.5 * x)

theorem minimum_F : ∃ x : ℝ, x ≥ 0 ∧ F x = 57.5 ∧ ∀ y ≥ 0, F y ≥ F x := by
  use 55
  sorry

end minimum_F_l98_98773


namespace find_other_endpoint_l98_98567

set_option pp.funBinderTypes true

def circle_center : (ℝ × ℝ) := (5, -2)
def diameter_endpoint1 : (ℝ × ℝ) := (1, 2)
def diameter_endpoint2 : (ℝ × ℝ) := (9, -6)

theorem find_other_endpoint (c : ℝ × ℝ) (e1 : ℝ × ℝ) (e2 : ℝ × ℝ) : 
  c = circle_center ∧ e1 = diameter_endpoint1 → e2 = diameter_endpoint2 := by
  sorry

end find_other_endpoint_l98_98567


namespace lionel_distance_walked_when_met_l98_98650

theorem lionel_distance_walked_when_met (distance_between : ℕ) (lionel_speed : ℕ) (walt_speed : ℕ) (advance_time : ℕ) 
(h1 : distance_between = 48) 
(h2 : lionel_speed = 2) 
(h3 : walt_speed = 6) 
(h4 : advance_time = 2) : 
  ∃ D : ℕ, D = 15 :=
by
  sorry

end lionel_distance_walked_when_met_l98_98650


namespace parabola_coordinates_l98_98208

theorem parabola_coordinates (x y : ℝ) (h_parabola : y^2 = 4 * x) (h_distance : (x - 1)^2 + y^2 = 100) :
  (x = 9 ∧ y = 6) ∨ (x = 9 ∧ y = -6) :=
by
  sorry

end parabola_coordinates_l98_98208


namespace func1_max_min_func2_max_min_l98_98592

noncomputable def func1 (x : ℝ) : ℝ := 2 * Real.sin x - 3
noncomputable def func2 (x : ℝ) : ℝ := (7/4 : ℝ) + Real.sin x - (Real.sin x) ^ 2

theorem func1_max_min : (∀ x : ℝ, func1 x ≤ -1) ∧ (∃ x : ℝ, func1 x = -1) ∧ (∀ x : ℝ, func1 x ≥ -5) ∧ (∃ x : ℝ, func1 x = -5)  :=
by
  sorry

theorem func2_max_min : (∀ x : ℝ, func2 x ≤ 2) ∧ (∃ x : ℝ, func2 x = 2) ∧ (∀ x : ℝ, func2 x ≥ 7 / 4) ∧ (∃ x : ℝ, func2 x = 7 / 4) :=
by
  sorry

end func1_max_min_func2_max_min_l98_98592


namespace cube_volume_l98_98076

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98076


namespace find_number_l98_98234

theorem find_number (x : ℤ) (h : 22 * (x - 36) = 748) : x = 70 :=
sorry

end find_number_l98_98234


namespace boat_speed_still_water_l98_98979

variable (V_b V_s : ℝ)

def upstream : Prop := V_b - V_s = 10
def downstream : Prop := V_b + V_s = 40

theorem boat_speed_still_water (h1 : upstream V_b V_s) (h2 : downstream V_b V_s) : V_b = 25 :=
by
  sorry

end boat_speed_still_water_l98_98979


namespace sequence_positive_and_divisible_l98_98683

theorem sequence_positive_and_divisible:
  ∃ (a : ℕ → ℕ), 
    (a 1 = 2) ∧ (a 2 = 500) ∧ (a 3 = 2000) ∧ 
    (∀ n ≥ 2, (a (n + 2) + a (n + 1)) * a (n - 1) = a (n + 1) * (a (n + 1) + a (n - 1))) ∧ 
    (∀ n, a n > 0) ∧ 
    (2 ^ 2000 ∣ a 2000) := 
sorry

end sequence_positive_and_divisible_l98_98683


namespace cube_volume_l98_98092

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98092


namespace min_value_of_a_l98_98784

variables (a b c d : ℕ)

-- Conditions
def conditions : Prop :=
  a > b ∧ b > c ∧ c > d ∧
  a + b + c + d = 2004 ∧
  a^2 - b^2 + c^2 - d^2 = 2004

-- Theorem: minimum value of a
theorem min_value_of_a (h : conditions a b c d) : a = 503 :=
sorry

end min_value_of_a_l98_98784


namespace max_triangle_perimeter_l98_98349

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l98_98349


namespace greatest_possible_perimeter_l98_98357

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l98_98357


namespace proof_problem_l98_98303

theorem proof_problem (a b c : ℝ) (h : a > b) (h1 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0) :=
sorry

end proof_problem_l98_98303


namespace rationalize_denominator_l98_98811

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l98_98811


namespace average_of_four_l98_98319

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end average_of_four_l98_98319


namespace average_age_of_town_population_l98_98770

theorem average_age_of_town_population
  (children adults : ℕ)
  (ratio_condition : 3 * adults = 2 * children)
  (avg_age_children : ℕ := 10)
  (avg_age_adults : ℕ := 40) :
  ((10 * children + 40 * adults) / (children + adults) = 22) :=
by
  sorry

end average_age_of_town_population_l98_98770


namespace good_goods_not_cheap_is_sufficient_condition_l98_98969

theorem good_goods_not_cheap_is_sufficient_condition
  (goods_good : Prop)
  (goods_not_cheap : Prop)
  (h : goods_good → goods_not_cheap) :
  (goods_good → goods_not_cheap) :=
by
  exact h

end good_goods_not_cheap_is_sufficient_condition_l98_98969


namespace sequence_problem_l98_98636

-- Given sequence
variable (P Q R S T U V : ℤ)

-- Given conditions
variable (hR : R = 7)
variable (hPQ : P + Q + R = 21)
variable (hQS : Q + R + S = 21)
variable (hST : R + S + T = 21)
variable (hTU : S + T + U = 21)
variable (hUV : T + U + V = 21)

theorem sequence_problem : P + V = 14 := by
  sorry

end sequence_problem_l98_98636


namespace temperature_difference_is_correct_l98_98002

def highest_temperature : ℤ := -9
def lowest_temperature : ℤ := -22
def temperature_difference : ℤ := highest_temperature - lowest_temperature

theorem temperature_difference_is_correct :
  temperature_difference = 13 := by
  -- We need to prove this statement is correct
  sorry

end temperature_difference_is_correct_l98_98002


namespace number_of_arrangements_l98_98841

theorem number_of_arrangements (A B : Type) (individuals : Fin 6 → Type)
  (adjacent_condition : ∃ (i : Fin 5), individuals i = B ∧ individuals (i + 1) = A) :
  ∃ (n : ℕ), n = 120 :=
by
  sorry

end number_of_arrangements_l98_98841


namespace series_sum_equals_1_over_400_l98_98901

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 3) / ((4 * n + 1)^2 * (4 * n + 5)^2)

theorem series_sum_equals_1_over_400 :
  ∑' n, series_term (n + 1) = 1 / 400 := by
  sorry

end series_sum_equals_1_over_400_l98_98901


namespace probability_abs_diff_gt_1_l98_98006

open probability_theory

noncomputable def coin_flip : ProbMeasure ℝ :=
if h : true then
  ⟨[0, 2], sorry⟩ -- This represents the distribution condition
else
  ⟨[], sorry⟩     -- Dummy value to satisfy the type

def chosen_distribution (s : set ℝ) : ProbMeasure ℝ :=
if h : true then 
  ⟨[0, 2], sorry⟩ -- This represents the uniform distribution condition
else 
  ⟨[], sorry⟩     -- Dummy value to satisfy the type

noncomputable def prob_event (s : set (ℝ × ℝ)) : ℝ :=
(coin_flip.prod coin_flip).val s

theorem probability_abs_diff_gt_1 :
  prob_event {p : ℝ × ℝ | abs (p.1 - p.2) > 1} = 7 / 8 :=
sorry

end probability_abs_diff_gt_1_l98_98006


namespace area_shaded_smaller_dodecagon_area_in_circle_l98_98698

-- Part (a) statement
theorem area_shaded_smaller (dodecagon_area : ℝ) (shaded_area : ℝ) 
  (h : shaded_area = (1 / 12) * dodecagon_area) :
  shaded_area = dodecagon_area / 12 :=
sorry

-- Part (b) statement
theorem dodecagon_area_in_circle (r : ℝ) (A : ℝ) 
  (h : r = 1) (h' : A = (1 / 2) * 12 * r ^ 2 * Real.sin (2 * Real.pi / 12)) :
  A = 3 :=
sorry

end area_shaded_smaller_dodecagon_area_in_circle_l98_98698


namespace Martha_improvement_in_lap_time_l98_98409

theorem Martha_improvement_in_lap_time 
  (initial_laps : ℕ) (initial_time : ℕ) 
  (first_month_laps : ℕ) (first_month_time : ℕ) 
  (second_month_laps : ℕ) (second_month_time : ℕ)
  (sec_per_min : ℕ)
  (conds : initial_laps = 15 ∧ initial_time = 30 ∧ first_month_laps = 18 ∧ first_month_time = 27 ∧ 
           second_month_laps = 20 ∧ second_month_time = 27 ∧ sec_per_min = 60)
  : ((initial_time / initial_laps : ℚ) - (second_month_time / second_month_laps)) * sec_per_min = 39 :=
by
  sorry

end Martha_improvement_in_lap_time_l98_98409


namespace rita_canoe_trip_distance_l98_98838

theorem rita_canoe_trip_distance 
  (D : ℝ)
  (h_upstream : ∃ t1, t1 = D / 3)
  (h_downstream : ∃ t2, t2 = D / 9)
  (h_total_time : ∃ t1 t2, t1 + t2 = 8) :
  D = 18 :=
by
  sorry

end rita_canoe_trip_distance_l98_98838


namespace find_function_f_l98_98912

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function_f (a : ℝ) (h1 : a > 0) :
  (∀ x y : ℝ, 0 < x ∧ 0 < y →
    f (f y / f x + 1) = f (x + y / x + 1) - f x) →
  ∀ x : ℝ, 0 < x → f x = a * x :=
  by sorry

end find_function_f_l98_98912


namespace inner_cube_surface_area_l98_98142

theorem inner_cube_surface_area (S : ℝ) (hS : S = 54) :
  let s := Real.sqrt (S / 6),
      sphere_diameter := s,
      l := Real.sqrt (sphere_diameter ^ 2 / 3)
  in 6 * l ^ 2 = 18 :=
by {
  let s := Real.sqrt (54 / 6),
  let sphere_diameter := s,
  let l := Real.sqrt (sphere_diameter ^ 2 / 3),
  have h1 : s = 3, by { norm_num1, },
  have h2 : sphere_diameter = 3, by { rw h1, },
  have h3 : l = Real.sqrt (3 ^ 2 / 3), by { rw h2, },
  have h4 : l = Real.sqrt 3, by { norm_num1, },
  have h5 : 6 * (Real.sqrt 3) ^ 2 = 18, by { norm_num1, },
  exact h5,
}

end inner_cube_surface_area_l98_98142


namespace solve_m_n_l98_98630

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end solve_m_n_l98_98630


namespace no_perfect_squares_exist_l98_98278

theorem no_perfect_squares_exist (x y : ℕ) :
  ¬(∃ k1 k2 : ℕ, x^2 + y = k1^2 ∧ y^2 + x = k2^2) :=
sorry

end no_perfect_squares_exist_l98_98278


namespace correct_transformation_l98_98173

theorem correct_transformation (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b :=
sorry

end correct_transformation_l98_98173


namespace average_output_assembly_line_l98_98044

theorem average_output_assembly_line (initial_cogs second_batch_cogs rate1 rate2 : ℕ) (time1 time2 : ℚ)
  (h1 : initial_cogs = 60)
  (h2 : second_batch_cogs = 60)
  (h3 : rate1 = 90)
  (h4 : rate2 = 60)
  (h5 : time1 = 60 / 90)
  (h6 : time2 = 60 / 60)
  (h7 : (120 : ℚ) / (time1 + time2) = (72 : ℚ)) :
  (120 : ℚ) / (time1 + time2) = 72 := by
  sorry

end average_output_assembly_line_l98_98044


namespace fourth_number_second_set_l98_98848

theorem fourth_number_second_set :
  (∃ (x y : ℕ), (28 + x + 42 + 78 + 104) / 5 = 90 ∧ (128 + 255 + 511 + y + x) / 5 = 423 ∧ x = 198) →
  (y = 1023) :=
by
  sorry

end fourth_number_second_set_l98_98848


namespace bobby_candy_left_l98_98465

def initial_candy := 22
def eaten_candy1 := 9
def eaten_candy2 := 5

theorem bobby_candy_left : initial_candy - eaten_candy1 - eaten_candy2 = 8 :=
by
  sorry

end bobby_candy_left_l98_98465


namespace fraction_speed_bus_train_l98_98982

theorem fraction_speed_bus_train :
  let speed_train := 16 * 5
  let speed_bus := 480 / 8
  let speed_train_prop := speed_train = 80
  let speed_bus_prop := speed_bus = 60
  speed_bus / speed_train = 3 / 4 :=
by
  sorry

end fraction_speed_bus_train_l98_98982


namespace find_e_value_l98_98739

-- Define constants a, b, c, d, and e
variables (a b c d e : ℝ)

-- Theorem statement
theorem find_e_value (h1 : (2 : ℝ)^7 * a + (2 : ℝ)^5 * b + (2 : ℝ)^3 * c + 2 * d + e = 23)
                     (h2 : ((-2) : ℝ)^7 * a + ((-2) : ℝ)^5 * b + ((-2) : ℝ)^3 * c + ((-2) : ℝ) * d + e = -35) :
  e = -6 :=
sorry

end find_e_value_l98_98739


namespace remainder_of_3_pow_108_plus_5_l98_98990

theorem remainder_of_3_pow_108_plus_5 :
  (3^108 + 5) % 10 = 6 := by
  sorry

end remainder_of_3_pow_108_plus_5_l98_98990


namespace correct_operation_l98_98037

theorem correct_operation (x : ℝ) (hx : x ≠ 0) : x^2 / x^8 = 1 / x^6 :=
by
  sorry

end correct_operation_l98_98037


namespace classroom_books_l98_98507

theorem classroom_books (students_group1 students_group2 books_per_student_group1 books_per_student_group2 books_brought books_lost : ℕ)
  (h1 : students_group1 = 20)
  (h2 : books_per_student_group1 = 15)
  (h3 : students_group2 = 25)
  (h4 : books_per_student_group2 = 10)
  (h5 : books_brought = 30)
  (h6 : books_lost = 7) :
  (students_group1 * books_per_student_group1 + students_group2 * books_per_student_group2 + books_brought - books_lost) = 573 := by
  sorry

end classroom_books_l98_98507


namespace glove_pair_probability_l98_98458

/-- 
A box contains 6 pairs of black gloves (i.e., 12 black gloves) and 4 pairs of beige gloves (i.e., 8 beige gloves).
We need to prove that the probability of drawing a matching pair of gloves is 47/95.
-/
theorem glove_pair_probability : 
  let total_gloves := 20
  let black_gloves := 12
  let beige_gloves := 8
  let P1_black := (black_gloves / total_gloves) * ((black_gloves - 1) / (total_gloves - 1))
  let P2_beige := (beige_gloves / total_gloves) * ((beige_gloves - 1) / (total_gloves - 1))
  let total_probability := P1_black + P2_beige
  total_probability = 47 / 95 :=
sorry

end glove_pair_probability_l98_98458


namespace cube_volume_l98_98071

-- Define the surface area condition
def surface_area := 150

-- Define the formula for the surface area in terms of the edge length
def edge_length (s : ℝ) : Prop := 6 * s^2 = surface_area

-- Define the formula for volume in terms of the edge length
def volume (s : ℝ) : ℝ := s^3

-- Define the statement we need to prove
theorem cube_volume : ∃ s : ℝ, edge_length s ∧ volume s = 125 :=
by sorry

end cube_volume_l98_98071


namespace cube_volume_l98_98109

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98109


namespace remainder_sets_two_disjoint_subsets_l98_98955

noncomputable def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem remainder_sets_two_disjoint_subsets (m : ℕ)
  (h : m = (3^12 - 2 * 2^12 + 1) / 2) : m % 1000 = 625 := 
by {
  -- math proof is omitted
  sorry
}

end remainder_sets_two_disjoint_subsets_l98_98955


namespace smallest_positive_integer_expr_2010m_44000n_l98_98277

theorem smallest_positive_integer_expr_2010m_44000n :
  ∃ (m n : ℤ), 10 = gcd 2010 44000 :=
by
  sorry

end smallest_positive_integer_expr_2010m_44000n_l98_98277


namespace part1_distance_part2_equation_l98_98752

noncomputable section

-- Define the conditions for Part 1
def hyperbola_C1 (x y : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = 1

-- Define the point M(3, t) existing on hyperbola C₁
def point_on_hyperbola_C1 (t : ℝ) : Prop := hyperbola_C1 3 t

-- Define the right focus of hyperbola C1
def right_focus_C1 : ℝ × ℝ := (4, 0)

-- Part 1: Distance from point M to the right focus
theorem part1_distance (t : ℝ) (h : point_on_hyperbola_C1 t) :  
  let distance := Real.sqrt ((3 - 4)^2 + (t - 0)^2)
  distance = 4 := sorry

-- Define the conditions for Part 2
def hyperbola_C2 (x y : ℝ) (m : ℝ) : Prop := (x^2 / 4) - (y^2 / 12) = m

-- Define the point (-3, 2√6) existing on hyperbola C₂
def point_on_hyperbola_C2 (m : ℝ) : Prop := hyperbola_C2 (-3) (2 * Real.sqrt 6) m

-- Part 2: The standard equation of hyperbola C₂
theorem part2_equation (h : point_on_hyperbola_C2 (1/4)) : 
  ∀ (x y : ℝ), hyperbola_C2 x y (1/4) ↔ (x^2 - (y^2 / 3) = 1) := sorry

end part1_distance_part2_equation_l98_98752


namespace train_length_l98_98574

theorem train_length (t : ℝ) (v : ℝ) (h1 : t = 13) (h2 : v = 58.15384615384615) : abs (v * t - 756) < 1 :=
by
  sorry

end train_length_l98_98574


namespace valid_numbers_eq_l98_98589

-- Definition of the number representation
def is_valid_number (x : ℕ) : Prop :=
  100 ≤ x ∧ x ≤ 999 ∧
  ∃ (a b c : ℕ), 
    1 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    x = 100 * a + 10 * b + c ∧
    x = a^3 + b^3 + c^3

-- The theorem to prove
theorem valid_numbers_eq : 
  {x : ℕ | is_valid_number x} = {153, 407} :=
by
  sorry

end valid_numbers_eq_l98_98589


namespace common_difference_arithmetic_sequence_l98_98980

-- Define the arithmetic sequence properties
variable (S : ℕ → ℕ) -- S represents the sum of the first n terms
variable (a : ℕ → ℕ) -- a represents the terms in the arithmetic sequence
variable (d : ℤ) -- common difference

-- Define the conditions
axiom S2_eq_6 : S 2 = 6
axiom a1_eq_4 : a 1 = 4

-- The problem: show that d = -2
theorem common_difference_arithmetic_sequence :
  (a 2 - a 1 = d) → d = -2 :=
by
  sorry

end common_difference_arithmetic_sequence_l98_98980


namespace power_mod_result_l98_98694

theorem power_mod_result :
  9^1002 % 50 = 1 := by
  sorry

end power_mod_result_l98_98694


namespace scoring_situations_4_students_l98_98508

noncomputable def number_of_scoring_situations (students : ℕ) (topicA_score : ℤ) (topicB_score : ℤ) : ℕ :=
  let combinations := Nat.choose 4 2
  let first_category := combinations * 2 * 2
  let second_category := 2 * combinations
  first_category + second_category

theorem scoring_situations_4_students : number_of_scoring_situations 4 100 90 = 36 :=
by
  -- The proof is omitted as per the instructions.
  sorry

end scoring_situations_4_students_l98_98508


namespace min_lamps_l98_98460

theorem min_lamps (n p : ℕ) (h1: p > 0) (h_total_profit : 3 * (3 * p / 4 / n) + (n - 3) * (p / n + 10) - p = 100) : n = 13 :=
by
  sorry

end min_lamps_l98_98460


namespace frank_reads_pages_per_day_l98_98295

-- Define the conditions and problem statement
def total_pages : ℕ := 450
def total_chapters : ℕ := 41
def total_days : ℕ := 30

-- The derived value we need to prove
def pages_per_day : ℕ := total_pages / total_days

-- The theorem to prove
theorem frank_reads_pages_per_day : pages_per_day = 15 :=
  by
  -- Proof goes here
  sorry

end frank_reads_pages_per_day_l98_98295


namespace cube_volume_from_surface_area_l98_98060

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98060


namespace rectangle_area_l98_98944

theorem rectangle_area {AB AC BC : ℕ} (hAB : AB = 15) (hAC : AC = 17)
  (hRightTriangle : AC * AC = AB * AB + BC * BC) : AB * BC = 120 := by
  sorry

end rectangle_area_l98_98944


namespace shanille_probability_l98_98968

-- Defining the probability function according to the problem's conditions.
def hit_probability (n k : ℕ) : ℚ :=
  if n = 100 ∧ k = 50 then 1 / 99 else 0

-- Prove that the probability Shanille hits exactly 50 of her first 100 shots is 1/99.
theorem shanille_probability :
  hit_probability 100 50 = 1 / 99 :=
by
  -- proof omitted
  sorry

end shanille_probability_l98_98968


namespace factor_example_solve_equation_example_l98_98700

-- Factorization proof problem
theorem factor_example (m a b : ℝ) : 
  (m * a ^ 2 - 4 * m * b ^ 2) = m * (a + 2 * b) * (a - 2 * b) :=
sorry

-- Solving the equation proof problem
theorem solve_equation_example (x : ℝ) (hx1: x ≠ 2) (hx2: x ≠ 0) : 
  (1 / (x - 2) = 3 / x) ↔ x = 3 :=
sorry

end factor_example_solve_equation_example_l98_98700


namespace greatest_perimeter_of_triangle_l98_98343

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l98_98343


namespace cost_price_for_fabrics_l98_98713

noncomputable def total_cost_price (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  selling_price - (meters_sold * profit_per_meter)

noncomputable def cost_price_per_meter (meters_sold: ℕ) (selling_price: ℚ) (profit_per_meter: ℚ): ℚ :=
  total_cost_price meters_sold selling_price profit_per_meter / meters_sold

theorem cost_price_for_fabrics :
  cost_price_per_meter 45 6000 12 = 121.33 ∧
  cost_price_per_meter 60 10800 15 = 165 ∧
  cost_price_per_meter 30 3900 10 = 120 :=
by
  sorry

end cost_price_for_fabrics_l98_98713


namespace platform_length_l98_98865

noncomputable def train_length := 420 -- length of the train in meters
noncomputable def time_to_cross_platform := 60 -- time to cross the platform in seconds
noncomputable def time_to_cross_pole := 30 -- time to cross the signal pole in seconds

theorem platform_length :
  ∃ L, L = 420 ∧ train_length / time_to_cross_pole = train_length / time_to_cross_platform * (train_length + L) / time_to_cross_platform :=
by
  use 420
  sorry

end platform_length_l98_98865


namespace min_value_expression_l98_98196

theorem min_value_expression :
  ∀ (x y : ℝ), x > 0 → y > 0 → x + y = 1 → (∃ (c : ℝ), c = 16 ∧ ∀ z, z = (1 / x + 9 / y) → z ≥ c) :=
by
  sorry

end min_value_expression_l98_98196


namespace greatest_possible_perimeter_l98_98326

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l98_98326


namespace rationalize_sqrt_fraction_l98_98835

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l98_98835


namespace find_x_y_l98_98197

theorem find_x_y (x y : ℝ) (h1 : x + Real.cos y = 2023) (h2 : x + 2023 * Real.sin y = 2022) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2022 :=
sorry

end find_x_y_l98_98197


namespace principal_amount_is_1200_l98_98672

-- Define the given conditions
def simple_interest (P : ℝ) : ℝ := 0.10 * P
def compound_interest (P : ℝ) : ℝ := 0.1025 * P

-- Define given difference
def interest_difference (P : ℝ) : ℝ := compound_interest P - simple_interest P

-- The main goal is to prove that the principal amount P that satisfies the difference condition is 1200
theorem principal_amount_is_1200 : ∃ P : ℝ, interest_difference P = 3 ∧ P = 1200 :=
by
  sorry -- Proof to be completed

end principal_amount_is_1200_l98_98672


namespace triangle_is_right_triangle_l98_98193

variable (A B C : ℝ) (a b c : ℝ)

-- Conditions definitions
def condition1 : Prop := A + B = C
def condition2 : Prop := a / b = 3 / 4 ∧ b / c = 4 / 5 ∧ a / c = 3 / 5
def condition3 : Prop := A = 90 - B

-- Proof problem
theorem triangle_is_right_triangle (h1 : condition1 A B C) (h2 : condition2 a b c) (h3 : condition3 A B) : C = 90 := 
sorry

end triangle_is_right_triangle_l98_98193


namespace riding_mower_speed_l98_98400

theorem riding_mower_speed :
  (∃ R : ℝ, 
     (8 * (3 / 4) = 6) ∧       -- Jerry mows 6 acres with the riding mower
     (8 * (1 / 4) = 2) ∧       -- Jerry mows 2 acres with the push mower
     (2 / 1 = 2) ∧             -- Push mower takes 2 hours to mow 2 acres
     (5 - 2 = 3) ∧             -- Time spent on the riding mower is 3 hours
     (6 / 3 = R) ∧             -- Riding mower cuts 6 acres in 3 hours
     R = 2) :=                 -- Therefore, R (speed of riding mower in acres per hour) is 2
sorry

end riding_mower_speed_l98_98400


namespace cube_volume_of_surface_area_l98_98064

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98064


namespace greatest_perimeter_of_triangle_l98_98340

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l98_98340


namespace total_increase_percentage_l98_98045

-- Define the conditions: original speed S, first increase by 30%, then another increase by 10%
def original_speed (S : ℝ) := S
def first_increase (S : ℝ) := S * 1.30
def second_increase (S : ℝ) := (S * 1.30) * 1.10

-- Prove that the total increase in speed is 43% of the original speed
theorem total_increase_percentage (S : ℝ) :
  (second_increase S - original_speed S) / original_speed S * 100 = 43 :=
by
  sorry

end total_increase_percentage_l98_98045


namespace hyperbola_asymptote_value_of_a_l98_98309

-- Define the hyperbola and the conditions given
variables {a : ℝ} (h1 : a > 0) (h2 : ∀ x y : ℝ, 3 * x + 2 * y = 0 ∧ 3 * x - 2 * y = 0)

theorem hyperbola_asymptote_value_of_a :
  a = 2 := by
  sorry

end hyperbola_asymptote_value_of_a_l98_98309


namespace greatest_possible_perimeter_l98_98361

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (1 ≤ x) ∧
             (x + 4 * x + 20 = 50 ∧
             (5 * x > 20) ∧
             (x + 20 > 4 * x) ∧
             (4 * x + 20 > x)) := sorry

end greatest_possible_perimeter_l98_98361


namespace remainder_division_l98_98240

theorem remainder_division (x r : ℕ) (h₁ : 1650 - x = 1390) (h₂ : 1650 = 6 * x + r) : r = 90 := by
  sorry

end remainder_division_l98_98240


namespace sum_of_first_5_terms_is_55_l98_98924

variable (a : ℕ → ℝ) -- the arithmetic sequence
variable (d : ℝ) -- the common difference
variable (a_2 : a 2 = 7)
variable (a_4 : a 4 = 15)
noncomputable def sum_of_first_5_terms : ℝ := (5 * (a 2 + a 4)) / 2

theorem sum_of_first_5_terms_is_55 :
  sum_of_first_5_terms a = 55 :=
by
  sorry

end sum_of_first_5_terms_is_55_l98_98924


namespace lcm_of_pack_sizes_l98_98907

theorem lcm_of_pack_sizes :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 19) 8) 11) 17) 23 = 772616 := by
  sorry

end lcm_of_pack_sizes_l98_98907


namespace rectangle_area_l98_98266

theorem rectangle_area
  (s : ℝ)
  (h_square_area : s^2 = 49)
  (rect_width : ℝ := s)
  (rect_length : ℝ := 3 * rect_width)
  (h_rect_width_eq_s : rect_width = s)
  (h_rect_length_eq_3w : rect_length = 3 * rect_width) :
  rect_width * rect_length = 147 :=
by 
  skip
  sorry

end rectangle_area_l98_98266


namespace groups_of_men_and_women_l98_98514

def problem_statement : Prop :=
  let men : Finset ℕ := {1, 2, 3, 4}
  let women : Finset ℕ := {1, 2, 3, 4, 5}
  let group_size := 3
  let total_groups := 3
  -- condition: each group must have at least one man and one woman
  ∃ (group1 group2 group3 : Finset ℕ),
    group1.card = group_size ∧ group2.card = group_size ∧ group3.card = group_size ∧
    group1 ≠ group2 ∧ group2 ≠ group3 ∧ group3 ≠ group1 ∧
    ((group1 ∩ men).nonempty ∧ (group1 ∩ women).nonempty) ∧
    ((group2 ∩ men).nonempty ∧ (group2 ∩ women).nonempty) ∧
    ((group3 ∩ men).nonempty ∧ (group3 ∩ women).nonempty) ∧
    (group1 ∪ group2 ∪ group3 = men ∪ women) ∧
    Finset.card (group1 ∪ group2 ∪ group3) = men.card + women.card

theorem groups_of_men_and_women :
  problem_statement → ∃ n : ℕ, n = 360 :=
by
  sorry

end groups_of_men_and_women_l98_98514


namespace sufficient_but_not_necessary_condition_l98_98438

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (0 ≤ x ∧ x ≤ 1) → |x| ≤ 1 :=
by sorry

end sufficient_but_not_necessary_condition_l98_98438


namespace problem_126_times_3_pow_6_l98_98497

theorem problem_126_times_3_pow_6 (p : ℝ) (h : 126 * 3^8 = p) : 
  126 * 3^6 = (1 / 9) * p := 
by {
  -- Placeholder for the proof
  sorry
}

end problem_126_times_3_pow_6_l98_98497


namespace volume_of_cube_l98_98098

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98098


namespace find_number_to_be_multiplied_l98_98998

-- Define the conditions of the problem
variable (x : ℕ)

-- Condition 1: The correct multiplication would have been 43x
-- Condition 2: The actual multiplication done was 34x
-- Condition 3: The difference between correct and actual result is 1242

theorem find_number_to_be_multiplied (h : 43 * x - 34 * x = 1242) : 
  x = 138 := by
  sorry

end find_number_to_be_multiplied_l98_98998


namespace volume_of_cube_l98_98097

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98097


namespace simplify_expression_l98_98008

theorem simplify_expression : 4 * (15 / 7) * (21 / -45) = -4 :=
by 
    -- Lean's type system will verify the correctness of arithmetic simplifications.
    sorry

end simplify_expression_l98_98008


namespace combined_molecular_weight_l98_98693

-- Define atomic masses of elements
def atomic_mass_Ca : Float := 40.08
def atomic_mass_Br : Float := 79.904
def atomic_mass_Sr : Float := 87.62
def atomic_mass_Cl : Float := 35.453

-- Define number of moles for each compound
def moles_CaBr2 : Float := 4
def moles_SrCl2 : Float := 3

-- Define molar masses of compounds
def molar_mass_CaBr2 : Float := atomic_mass_Ca + 2 * atomic_mass_Br
def molar_mass_SrCl2 : Float := atomic_mass_Sr + 2 * atomic_mass_Cl

-- Define total mass calculation for each compound
def total_mass_CaBr2 : Float := moles_CaBr2 * molar_mass_CaBr2
def total_mass_SrCl2 : Float := moles_SrCl2 * molar_mass_SrCl2

-- Prove the combined molecular weight
theorem combined_molecular_weight :
  total_mass_CaBr2 + total_mass_SrCl2 = 1275.13 :=
  by
    -- The proof will be here
    sorry

end combined_molecular_weight_l98_98693


namespace larger_integer_is_50_l98_98427

-- Definition of the problem conditions.
def is_two_digit (x : ℕ) : Prop := 10 ≤ x ∧ x ≤ 99

def problem_conditions (m n : ℕ) : Prop := 
  is_two_digit m ∧ is_two_digit n ∧
  (m + n) / 2 = m + n / 100

-- Statement of the proof problem.
theorem larger_integer_is_50 (m n : ℕ) (h : problem_conditions m n) : max m n = 50 :=
  sorry

end larger_integer_is_50_l98_98427


namespace inner_cube_surface_area_l98_98167

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l98_98167


namespace danny_age_l98_98274

theorem danny_age (D : ℕ) (h : D - 19 = 3 * (26 - 19)) : D = 40 := by
  sorry

end danny_age_l98_98274


namespace purple_chip_value_l98_98768

theorem purple_chip_value 
  (x : ℕ)
  (blue_chip_value : 1 = 1)
  (green_chip_value : 5 = 5)
  (red_chip_value : 11 = 11)
  (purple_chip_condition1 : x > 5)
  (purple_chip_condition2 : x < 11)
  (product_of_points : ∃ b g p r, (b = 1 ∨ b = 1) ∧ (g = 5 ∨ g = 5) ∧ (p = x ∨ p = x) ∧ (r = 11 ∨ r = 11) ∧ b * g * p * r = 28160) : 
  x = 7 :=
sorry

end purple_chip_value_l98_98768


namespace find_cows_l98_98632

-- Define the number of ducks (D) and cows (C)
variables (D C : ℕ)

-- Define the main condition given in the problem
def legs_eq_condition (D C : ℕ) : Prop :=
  2 * D + 4 * C = 2 * (D + C) + 36

-- State the theorem we wish to prove
theorem find_cows (D C : ℕ) (h : legs_eq_condition D C) : C = 18 :=
sorry

end find_cows_l98_98632


namespace find_b_l98_98120

def passesThrough (b c : ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = P.1^2 + b * P.1 + c

theorem find_b (b c : ℝ)
  (H1 : passesThrough b c (1, 2))
  (H2 : passesThrough b c (5, 2)) :
  b = -6 :=
by
  sorry

end find_b_l98_98120


namespace two_trains_clearing_time_l98_98554

noncomputable def length_train1 : ℝ := 100  -- Length of Train 1 in meters
noncomputable def length_train2 : ℝ := 160  -- Length of Train 2 in meters
noncomputable def speed_train1 : ℝ := 42 * 1000 / 3600  -- Speed of Train 1 in m/s
noncomputable def speed_train2 : ℝ := 30 * 1000 / 3600  -- Speed of Train 2 in m/s
noncomputable def total_distance : ℝ := length_train1 + length_train2  -- Total distance to be covered
noncomputable def relative_speed : ℝ := speed_train1 + speed_train2  -- Relative speed

theorem two_trains_clearing_time : total_distance / relative_speed = 13 := by
  sorry

end two_trains_clearing_time_l98_98554


namespace calculate_r_when_n_is_3_l98_98645

theorem calculate_r_when_n_is_3 : 
  ∀ (r s n : ℕ), r = 4^s - s → s = 3^n + 2 → n = 3 → r = 4^29 - 29 :=
by 
  intros r s n h1 h2 h3
  sorry

end calculate_r_when_n_is_3_l98_98645


namespace scientific_notation_five_hundred_billion_l98_98179

theorem scientific_notation_five_hundred_billion :
  500000000000 = 5 * 10^11 := by
  sorry

end scientific_notation_five_hundred_billion_l98_98179


namespace cube_volume_l98_98107

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98107


namespace actual_tax_equals_600_l98_98873

-- Definition for the first condition: initial tax amount
variable (a : ℝ)

-- Define the first reduction: 25% reduction
def first_reduction (a : ℝ) : ℝ := 0.75 * a

-- Define the second reduction: further 20% reduction
def second_reduction (tax_after_first_reduction : ℝ) : ℝ := 0.80 * tax_after_first_reduction

-- Define the final reduction: combination of both reductions
def final_tax (a : ℝ) : ℝ := second_reduction (first_reduction a)

-- Proof that with a = 1000, the actual tax is 600 million euros
theorem actual_tax_equals_600 (a : ℝ) (h₁ : a = 1000) : final_tax a = 600 := by
    rw [h₁]
    simp [final_tax, first_reduction, second_reduction]
    sorry

end actual_tax_equals_600_l98_98873


namespace second_cube_surface_area_l98_98128

theorem second_cube_surface_area (s : ℝ) (a : ℝ) (b : ℝ) (c : ℝ) (sa : ℝ) :
  6 * s^2 = 54 →
  a = s →
  b = a * (1 / 2) →
  c * Real.sqrt 3 = 2 * b →
  sa = 6 * c^2 →
  sa = 18 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end second_cube_surface_area_l98_98128


namespace function_continuous_at_point_continuous_at_f_l98_98417

noncomputable def delta (ε : ℝ) : ℝ := ε / 12

theorem function_continuous_at_point :
  ∀ (f : ℝ → ℝ) (x₀ : ℝ), (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε) :=
by
  let f := fun x => -2 * x^2 - 4
  let x₀ := 3
  have h1 : f x₀ = -22 := by linarith
  have h2 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - x₀) < δ → abs (f x - f x₀) < ε :=
    by
      intros ε ε_pos
      use (ε / 12)
      split
      { exact div_pos ε_pos twelve_pos }
      { intros x hx
        calc
        abs (f x - f x₀)
          = abs (-2 * x^2 - 4 - (-22)) : by simp [h1]
      ... = abs (-2 * (x^2 - 9)) : by ring 
      ... = 2 * abs (x^2 - 9) : by rw [abs_mul, abs_neg]; simp
      ... < ε : by 
        let δ := ε / 12
        have hx3 : abs (x - 3) < δ := hx
        have h2 : abs (x + 3) ≤ 6 :=
          calc 
            abs (x + 3) ≤ abs (x - 3) + 6 : by linarith
            ... ≤ δ + 6 : by linarith
            ... ≤ ε / 12 + 6 : by linarith
        exact mul_lt_of_lt_div ((div_pos ε_pos twelve_pos).le)

theorem continuous_at_f : ∀ (ε : ℝ), ε > 0 → ∃ δ > 0, ∀ x, |x - 3| < δ → |(-2 * x^2 - 4) - (-22)| < ε :=
by
  intros ε ε_pos
  unfold delta
  use δ ε
  split
  { exact div_pos ε_pos twelve_pos }
  { intros x h
    calc
    |(-2 * x^2 - 4) - (-22)|
      = 2 * |x^2 - 9| : by norm_num
  ... < ε : by
      let h' := abs_sub_lt_iff.mp h
      exact lt_of_le_of_lt (abs_mul _ _) (by linarith [div_pos ε_pos twelve_pos,
        le_abs_self, (mul_div_cancel' _ twelve_pos.ne.symm)]) }

end function_continuous_at_point_continuous_at_f_l98_98417


namespace cube_volume_l98_98088

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98088


namespace polynomial_roots_l98_98284

theorem polynomial_roots :
  ∀ (x : ℝ), (x^3 - x^2 - 6 * x + 8 = 0) ↔ (x = 2 ∨ x = (-1 + Real.sqrt 17) / 2 ∨ x = (-1 - Real.sqrt 17) / 2) :=
by
  sorry

end polynomial_roots_l98_98284


namespace smallest_n_division_l98_98446

-- Lean statement equivalent to the mathematical problem
theorem smallest_n_division (n : ℕ) (hn : n ≥ 3) : 
  (∃ (s : Finset ℕ), (∀ m ∈ s, 3 ≤ m ∧ m ≤ 2006) ∧ s.card = n - 2) ↔ n = 3 := 
sorry

end smallest_n_division_l98_98446


namespace average_hours_per_day_l98_98429

theorem average_hours_per_day (h : ℝ) :
  (3 * h * 12 + 2 * h * 9 = 108) → h = 2 :=
by 
  intro h_condition
  sorry

end average_hours_per_day_l98_98429


namespace ninth_term_arith_seq_l98_98684

theorem ninth_term_arith_seq (a d : ℤ) (h1 : a + 2 * d = 25) (h2 : a + 5 * d = 31) : a + 8 * d = 37 :=
sorry

end ninth_term_arith_seq_l98_98684


namespace vlecks_in_straight_angle_l98_98526

theorem vlecks_in_straight_angle (V : Type) [LinearOrderedField V] (full_circle_vlecks : V) (h1 : full_circle_vlecks = 600) :
  (full_circle_vlecks / 2) = 300 :=
by
  sorry

end vlecks_in_straight_angle_l98_98526


namespace x_cubed_plus_y_cubed_l98_98936

theorem x_cubed_plus_y_cubed:
  ∀ (x y : ℝ), (x * (x ^ 4 + y ^ 4) = y ^ 5) → (x ^ 2 * (x + y) ≠ y ^ 3) → (x ^ 3 + y ^ 3 = 1) :=
by
  intros x y h1 h2
  sorry

end x_cubed_plus_y_cubed_l98_98936


namespace friendly_sequences_exist_l98_98839

theorem friendly_sequences_exist :
  ∃ (a b : ℕ → ℕ), 
    (∀ n, a n = 2^(n-1)) ∧ 
    (∀ n, b n = 2*n - 1) ∧ 
    (∀ k : ℕ, ∃ (i j : ℕ), k = a i * b j) :=
by
  sorry

end friendly_sequences_exist_l98_98839


namespace number_of_nickels_l98_98313

def dimes : ℕ := 10
def pennies_per_dime : ℕ := 10
def pennies_per_nickel : ℕ := 5
def total_pennies : ℕ := 150

theorem number_of_nickels (total_value_dimes : ℕ := dimes * pennies_per_dime)
  (pennies_needed_from_nickels : ℕ := total_pennies - total_value_dimes)
  (n : ℕ) : n = pennies_needed_from_nickels / pennies_per_nickel → n = 10 := by
  sorry

end number_of_nickels_l98_98313


namespace rationalize_denominator_l98_98812

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l98_98812


namespace speed_of_faster_train_l98_98691

theorem speed_of_faster_train
  (length_each_train : ℕ)
  (length_in_meters : length_each_train = 50)
  (speed_slower_train_kmh : ℝ)
  (speed_slower : speed_slower_train_kmh = 36)
  (pass_time_seconds : ℕ)
  (pass_time : pass_time_seconds = 36) :
  ∃ speed_faster_train_kmh, speed_faster_train_kmh = 46 :=
by
  sorry

end speed_of_faster_train_l98_98691


namespace greatest_possible_perimeter_l98_98380

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l98_98380


namespace arithmetic_sequence_diff_l98_98393

-- Define an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Given condition for the arithmetic sequence
def condition (a : ℕ → ℝ) : Prop := 
  a 2 + a 4 + a 6 + a 8 + a 10 = 80

-- Definition of the common difference
def common_difference (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- The proof problem statement in Lean 4
theorem arithmetic_sequence_diff (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a → condition a → common_difference a d → a 7 - a 8 = -d :=
by
  intros _ _ _
  -- Proof will be conducted here
  sorry

end arithmetic_sequence_diff_l98_98393


namespace greatest_perimeter_l98_98354

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l98_98354


namespace estimated_total_fish_population_l98_98553

-- Definitions of the initial conditions
def tagged_fish_in_first_catch : ℕ := 100
def total_fish_in_second_catch : ℕ := 300
def tagged_fish_in_second_catch : ℕ := 15

-- The theorem to prove the estimated number of total fish in the pond
theorem estimated_total_fish_population (tagged_fish_in_first_catch : ℕ) (total_fish_in_second_catch : ℕ) (tagged_fish_in_second_catch : ℕ) : ℕ :=
  2000

-- Assertion of the theorem with actual numbers
example : estimated_total_fish_population tagged_fish_in_first_catch total_fish_in_second_catch tagged_fish_in_second_catch = 2000 := by
  sorry

end estimated_total_fish_population_l98_98553


namespace selling_price_of_cycle_l98_98258

theorem selling_price_of_cycle
  (cost_price : ℕ)
  (gain_percent_decimal : ℚ)
  (h_cp : cost_price = 850)
  (h_gpd : gain_percent_decimal = 27.058823529411764 / 100) :
  ∃ selling_price : ℚ, selling_price = cost_price * (1 + gain_percent_decimal) ∧ selling_price = 1080 := 
by
  use (cost_price * (1 + gain_percent_decimal))
  sorry

end selling_price_of_cycle_l98_98258


namespace rationalize_denominator_l98_98817

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98817


namespace cube_volume_of_surface_area_l98_98063

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98063


namespace degree_le_of_lt_eventually_l98_98783

open Polynomial

theorem degree_le_of_lt_eventually {P Q : Polynomial ℝ} (h_exists : ∃ N : ℝ, ∀ x : ℝ, x > N → P.eval x < Q.eval x) :
  P.degree ≤ Q.degree :=
sorry

end degree_le_of_lt_eventually_l98_98783


namespace greatest_triangle_perimeter_l98_98376

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l98_98376


namespace AngeliCandies_l98_98716

def CandyProblem : Prop :=
  ∃ (C B G : ℕ), 
    (1/3 : ℝ) * C = 3 * (B : ℝ) ∧
    (2/3 : ℝ) * C = 2 * (G : ℝ) ∧
    (B + G = 40) ∧ 
    C = 144

theorem AngeliCandies :
  CandyProblem :=
sorry

end AngeliCandies_l98_98716


namespace decrypt_encryption_l98_98183

-- Encryption function description
def encrypt_digit (d : ℕ) : ℕ := 10 - (d * 7 % 10)

def encrypt_number (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let encrypted_digits := digits.map encrypt_digit
  encrypted_digits.foldr (λ d acc => d + acc * 10) 0
  
noncomputable def digit_match (d: ℕ) : ℕ :=
  match d with
  | 0 => 0 | 1 => 3 | 2 => 8 | 3 => 1 | 4 => 6 | 5 => 5
  | 6 => 8 | 7 => 1 | 8 => 4 | 9 => 7 | _ => 0

theorem decrypt_encryption:
encrypt_number 891134 = 473392 :=
by
  sorry

end decrypt_encryption_l98_98183


namespace andrew_age_l98_98178

theorem andrew_age (a g : ℕ) (h1 : g = 10 * a) (h2 : g - a = 63) : a = 7 := by
  sorry

end andrew_age_l98_98178


namespace f_equals_n_l98_98917

-- Define the function and P(n)
def f (n : ℕ) : ℕ := sorry

def P (n : ℕ) : ℕ := (list.range n).map (fun i => f (i + 1)).prod

-- Problem statement
theorem f_equals_n (f : ℕ → ℕ)
  (h : ∀ a b : ℕ, (P f a + P f b) ∣ (a.factorial + b.factorial)) :
  ∀ n : ℕ, f n = n :=
sorry

end f_equals_n_l98_98917


namespace percentage_discount_l98_98572

theorem percentage_discount (C S S' : ℝ) (h1 : S = 1.14 * C) (h2 : S' = 2.20 * C) :
  (S' - S) / S' * 100 = 48.18 :=
by 
  sorry

end percentage_discount_l98_98572


namespace transform_equation_l98_98858

theorem transform_equation (x : ℝ) (h₁ : x ≠ 3 / 2) (h₂ : 5 - 3 * x = 1) :
  x = 4 / 3 :=
sorry

end transform_equation_l98_98858


namespace proof_problem_l98_98916

noncomputable def problem_statement (m : ℕ) : Prop :=
  ∀ pairs : List (ℕ × ℕ),
  (∀ (x y : ℕ), (x, y) ∈ pairs ↔ x^2 - 3 * y^2 + 2 = 16 * m ∧ 2 * y ≤ x - 1) →
  pairs.length % 2 = 0 ∨ pairs.length = 0

theorem proof_problem (m : ℕ) (hm : m > 0) : problem_statement m :=
by
  sorry

end proof_problem_l98_98916


namespace cube_volume_from_surface_area_l98_98059

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98059


namespace problem_statement_l98_98299

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry

axiom f_pos (x : ℝ) : x > 0 → f x > 0
axiom f'_less_f (x : ℝ) : f' x < f x
axiom f_has_deriv_at : ∀ x, HasDerivAt f (f' x) x

def a : ℝ := sorry
axiom a_in_range : 0 < a ∧ a < 1

theorem problem_statement : 3 * f 0 > f a ∧ f a > a * f 1 :=
  sorry

end problem_statement_l98_98299


namespace remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l98_98189

theorem remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0:
  (7 * 12^24 + 3^24) % 11 = 0 := sorry

end remainder_7_mul_12_pow_24_add_3_pow_24_mod_11_eq_0_l98_98189


namespace cubic_sum_l98_98937

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 :=
by
  sorry

end cubic_sum_l98_98937


namespace minimum_value_expression_l98_98960

theorem minimum_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  8 * a^3 + 12 * b^3 + 27 * c^3 + (3 / (27 * a * b * c)) ≥ 6 :=
by
  sorry

end minimum_value_expression_l98_98960


namespace factorization_correct_l98_98732

theorem factorization_correct {x : ℝ} : (x - 15)^2 = x^2 - 30*x + 225 :=
by
  sorry

end factorization_correct_l98_98732


namespace problem_statement_l98_98663

theorem problem_statement (x y : ℝ) (h1 : x + y = 5) (h2 : x * y = 4) :
  x + (x^3 / y^2) + (y^3 / x^2) + y = 74.0625 :=
sorry

end problem_statement_l98_98663


namespace player_matches_average_increase_l98_98876

theorem player_matches_average_increase 
  (n T : ℕ) 
  (h1 : T = 32 * n) 
  (h2 : (T + 76) / (n + 1) = 36) : 
  n = 10 := 
by 
  sorry

end player_matches_average_increase_l98_98876


namespace inner_cube_surface_area_l98_98164

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l98_98164


namespace cube_volume_l98_98085

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98085


namespace cube_volume_l98_98110

theorem cube_volume (s : ℝ) (V : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98110


namespace compute_result_l98_98725

theorem compute_result : (300000 * 200000) / 100000 = 600000 := by
  sorry

end compute_result_l98_98725


namespace number_of_true_propositions_l98_98624

variable (x : ℝ)

def original_proposition (x : ℝ) : Prop := (x = 5) → (x^2 - 8 * x + 15 = 0)
def converse_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 = 0) → (x = 5)
def inverse_proposition (x : ℝ) : Prop := (x ≠ 5) → (x^2 - 8 * x + 15 ≠ 0)
def contrapositive_proposition (x : ℝ) : Prop := (x^2 - 8 * x + 15 ≠ 0) → (x ≠ 5)

theorem number_of_true_propositions : 
  (original_proposition x ∧ contrapositive_proposition x) ∧
  ¬(converse_proposition x) ∧ ¬(inverse_proposition x) ↔ true := sorry

end number_of_true_propositions_l98_98624


namespace problem_statement_l98_98232

theorem problem_statement (a b : ℕ) (ha : a = 55555) (hb : b = 66666) :
  55554 * 55559 * 55552 - 55556 * 55551 * 55558 =
  66665 * 66670 * 66663 - 66667 * 66662 * 66669 := 
by
  sorry

end problem_statement_l98_98232


namespace rationalize_sqrt_5_over_12_l98_98824

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l98_98824


namespace number_of_first_year_students_to_be_sampled_l98_98172

-- Definitions based on the conditions
def total_students_in_each_grade (x : ℕ) : List ℕ := [4*x, 5*x, 5*x, 6*x]
def total_undergraduate_students (x : ℕ) : ℕ := 4*x + 5*x + 5*x + 6*x
def sample_size : ℕ := 300
def sampling_fraction (x : ℕ) : ℚ := sample_size / total_undergraduate_students x
def first_year_sampling (x : ℕ) : ℕ := (4*x) * sample_size / total_undergraduate_students x

-- Statement to prove
theorem number_of_first_year_students_to_be_sampled {x : ℕ} (hx_pos : x > 0) :
  first_year_sampling x = 60 := 
by
  -- skip the proof
  sorry

end number_of_first_year_students_to_be_sampled_l98_98172


namespace calc_expr_l98_98579

theorem calc_expr : 3000 * (3000 ^ 3000) + 3000 ^ 2 = 3000 ^ 3001 :=
by sorry

end calc_expr_l98_98579


namespace income_after_selling_more_l98_98426

theorem income_after_selling_more (x y : ℝ)
  (h1 : 26 * x + 14 * y = 264) 
  : 39 * x + 21 * y = 396 := 
by 
  sorry

end income_after_selling_more_l98_98426


namespace volume_of_cut_out_box_l98_98267

theorem volume_of_cut_out_box (x : ℝ) : 
  let l := 16
  let w := 12
  let new_l := l - 2 * x
  let new_w := w - 2 * x
  let height := x
  let V := new_l * new_w * height
  V = 4 * x^3 - 56 * x^2 + 192 * x :=
by
  sorry

end volume_of_cut_out_box_l98_98267


namespace min_value_expression_l98_98914

theorem min_value_expression (y : ℝ) (hy : y > 0) : 9 * y + 1 / y^6 ≥ 10 :=
by
  sorry

end min_value_expression_l98_98914


namespace no_solutions_for_inequalities_l98_98010

theorem no_solutions_for_inequalities (x y z t : ℝ) :
  |x| < |y - z + t| →
  |y| < |x - z + t| →
  |z| < |x - y + t| →
  |t| < |x - y + z| →
  False :=
by
  sorry

end no_solutions_for_inequalities_l98_98010


namespace calculate_glass_area_l98_98948

-- Given conditions as definitions
def long_wall_length : ℕ := 30
def long_wall_height : ℕ := 12
def short_wall_length : ℕ := 20

-- Total area of glass required (what we want to prove)
def total_glass_area : ℕ := 960

-- The theorem to prove
theorem calculate_glass_area
  (a1 : long_wall_length = 30)
  (a2 : long_wall_height = 12)
  (a3 : short_wall_length = 20) :
  2 * (long_wall_length * long_wall_height) + (short_wall_length * long_wall_height) = total_glass_area :=
by
  -- The proof is omitted
  sorry

end calculate_glass_area_l98_98948


namespace sin_omega_x_increasing_and_maximum_ω_l98_98489

noncomputable def function_range (ω : ℝ) : Prop :=
  (∀ x ∈ Icc (- (2/3) * real.pi / ω) (5/6 * real.pi / ω), 
    (ω * real.cos (ω * x)) > 0)
  ∧ (1/2 ≤ ω ∧ ω ≤ 3/5)

theorem sin_omega_x_increasing_and_maximum_ω (ω : ℝ) (h : ω > 0) :
  function_range ω ↔ (1/2 ≤ ω ∧ ω ≤ 3/5) := 
sorry

end sin_omega_x_increasing_and_maximum_ω_l98_98489


namespace option_A_option_C_l98_98919

/-- Definition of the set M such that M = {a | a = x^2 - y^2, x, y ∈ ℤ} -/
def M := {a : ℤ | ∃ x y : ℤ, a = x^2 - y^2}

/-- Definition of the set B such that B = {b | b = 2n + 1, n ∈ ℕ} -/
def B := {b : ℤ | ∃ n : ℕ, b = 2 * n + 1}

theorem option_A (a1 a2 : ℤ) (ha1 : a1 ∈ M) (ha2 : a2 ∈ M) : a1 * a2 ∈ M := sorry

theorem option_C : B ⊆ M := sorry

end option_A_option_C_l98_98919


namespace perimeter_one_face_of_cube_is_24_l98_98680

noncomputable def cube_volume : ℝ := 216
def perimeter_of_face_of_cube (V : ℝ) : ℝ := 4 * (V^(1/3) : ℝ)

theorem perimeter_one_face_of_cube_is_24 :
  perimeter_of_face_of_cube cube_volume = 24 := 
by
  -- This proof will invoke the calculation shown in the problem.
  sorry

end perimeter_one_face_of_cube_is_24_l98_98680


namespace probabilities_equal_l98_98765

noncomputable def probability (m1 m2 : ℕ) : ℚ := m1 / (m1 + m2 : ℚ)

theorem probabilities_equal 
  (u j p b : ℕ) 
  (huj : u > j) 
  (hbp : b > p) : 
  (probability u p) * (probability b u) * (probability j b) * (probability p j) = 
  (probability u b) * (probability p u) * (probability j p) * (probability b j) :=
by
  sorry

end probabilities_equal_l98_98765


namespace lilyPadsFullCoverage_l98_98509

def lilyPadDoubling (t: ℕ) : ℕ :=
  t + 1

theorem lilyPadsFullCoverage (t: ℕ) (h: t = 47) : lilyPadDoubling t = 48 :=
by
  rw [h]
  unfold lilyPadDoubling
  rfl

end lilyPadsFullCoverage_l98_98509


namespace rationalize_sqrt_fraction_l98_98807

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l98_98807


namespace compute_polynomial_at_3_l98_98406

noncomputable def polynomial_p (b : Fin 6 → ℕ) (x : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * x^2 + b 3 * x^3 + b 4 * x^4 + b 5 * x^5

theorem compute_polynomial_at_3
  (b : Fin 6 → ℕ)
  (hbi : ∀ i, b i < 5)
  (hP5 : polynomial_p b (Real.sqrt 5) = 40 + 31 * Real.sqrt 5) :
  polynomial_p b 3 = 381 :=
sorry

end compute_polynomial_at_3_l98_98406


namespace greatest_possible_perimeter_l98_98331

theorem greatest_possible_perimeter :
  ∀ (x : ℕ), (4 * x + x > 20) ∧ (x + 20 > 4 * x) → (∃ (x' ∈ {5, 6}), x = x' ∧ (perimeter (x, 4 * x, 20) = 50)) := 
by
  sorry

def perimeter (a b c : ℕ) : ℕ := a + b + c

end greatest_possible_perimeter_l98_98331


namespace average_of_four_l98_98318

-- Define the variables
variables {p q r s : ℝ}

-- Conditions as hypotheses
theorem average_of_four (h : (5 / 4) * (p + q + r + s) = 15) : (p + q + r + s) / 4 = 3 := 
by
  sorry

end average_of_four_l98_98318


namespace inner_cube_surface_area_l98_98153

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l98_98153


namespace johnny_practice_l98_98640

variable (P : ℕ) -- Current amount of practice in days
variable (h : P = 40) -- Given condition translating Johnny's practice amount
variable (d : ℕ) -- Additional days needed

theorem johnny_practice : d = 80 :=
by
  have goal : 3 * P = P + d := sorry
  have initial_condition : P = 40 := sorry
  have required : d = 3 * 40 - 40 := sorry
  sorry

end johnny_practice_l98_98640


namespace expansion_coeff_l98_98614

theorem expansion_coeff (a b : ℝ) (x : ℝ) (h : (1 + a * x) ^ 5 = 1 + 10 * x + b * x^2 + a^5 * x^5) :
  b = 40 :=
sorry

end expansion_coeff_l98_98614


namespace value_of_each_other_toy_l98_98886

-- Definitions for the conditions
def total_toys : ℕ := 9
def total_worth : ℕ := 52
def single_toy_value : ℕ := 12

-- Definition to represent the value of each of the other toys
def other_toys_value (same_value : ℕ) : Prop :=
  (total_worth - single_toy_value) / (total_toys - 1) = same_value

-- The theorem to be proven
theorem value_of_each_other_toy : other_toys_value 5 :=
  sorry

end value_of_each_other_toy_l98_98886


namespace only_prime_when_p_is_2_l98_98275

theorem only_prime_when_p_is_2 {p : ℕ} (hp : Prime p) :
  p = 2 ∨ ¬ Prime (1 + ∑ i in Finset.range (p+1), i^p) :=
begin
  sorry
end

end only_prime_when_p_is_2_l98_98275


namespace find_g_l98_98674

noncomputable def g : ℝ → ℝ := sorry

theorem find_g :
  (g 1 = 2) ∧ (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) ↔ (∀ x : ℝ, g x = 2 * (4^x - 3^x)) := 
by
  sorry

end find_g_l98_98674


namespace find_constant_k_l98_98253

theorem find_constant_k (k : ℝ) :
    -x^2 - (k + 9) * x - 8 = -(x - 2) * (x - 4) → k = -15 := by
  sorry

end find_constant_k_l98_98253


namespace smallest_positive_period_l98_98622

-- Define a predicate for a function to have a period
def is_periodic {α : Type*} [AddGroup α] (f : α → ℝ) (T : α) : Prop :=
  ∀ x, f (x) = f (x - T)

-- The actual problem statement
theorem smallest_positive_period {f : ℝ → ℝ} 
  (h : ∀ x : ℝ, f (3 * x) = f (3 * x - 3 / 2)) : 
  is_periodic f (1 / 2) ∧ 
  ¬ (∃ T : ℝ, 0 < T ∧ T < 1 / 2 ∧ is_periodic f T) :=
by
  sorry

end smallest_positive_period_l98_98622


namespace inequality_always_true_l98_98456

theorem inequality_always_true (a : ℝ) :
  (∀ x : ℝ, ax^2 - x + 1 > 0) ↔ a > 1/4 :=
sorry

end inequality_always_true_l98_98456


namespace rationalize_sqrt_fraction_l98_98803

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l98_98803


namespace inequality_of_ab_bc_ca_l98_98740

theorem inequality_of_ab_bc_ca (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c)
  (h₃ : a^4 + b^4 + c^4 = 3) : 
  (1 / (4 - a * b)) + (1 / (4 - b * c)) + (1 / (4 - c * a)) ≤ 1 :=
by
  sorry

end inequality_of_ab_bc_ca_l98_98740


namespace greatest_possible_value_of_n_l98_98207

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 6400) : n ≤ 7 :=
by
  sorry

end greatest_possible_value_of_n_l98_98207


namespace quadratic_inequality_solution_l98_98842

theorem quadratic_inequality_solution (x : ℝ) :
  3 * x^2 - 2 * x - 8 ≤ 0 ↔ -4/3 ≤ x ∧ x ≤ 2 :=
sorry

end quadratic_inequality_solution_l98_98842


namespace corrected_mean_l98_98868

theorem corrected_mean (n : ℕ) (mean incorrect_observation correct_observation : ℝ) (h_n : n = 50) (h_mean : mean = 32) (h_incorrect : incorrect_observation = 23) (h_correct : correct_observation = 48) : 
  (mean * n + (correct_observation - incorrect_observation)) / n = 32.5 := 
by 
  sorry

end corrected_mean_l98_98868


namespace find_fz_l98_98934

noncomputable def v (x y : ℝ) : ℝ :=
  3^x * Real.sin (y * Real.log 3)

theorem find_fz (x y : ℝ) (C : ℂ) (z : ℂ) (hz : z = x + y * Complex.I) :
  ∃ f : ℂ → ℂ, f z = 3^z + C :=
by
  sorry

end find_fz_l98_98934


namespace toy_cost_price_l98_98053

theorem toy_cost_price (x : ℝ) (h : 1.5 * x * 0.8 - x = 20) : x = 100 := 
sorry

end toy_cost_price_l98_98053


namespace line_through_nodes_l98_98988

def Point := (ℤ × ℤ)

structure Triangle :=
  (A B C : Point)

def is_node (p : Point) : Prop := 
  ∃ (x y : ℤ), p = (x, y)

def strictly_inside (p : Point) (t : Triangle) : Prop := 
  -- Assume we have a function that defines if a point is strictly inside a triangle
  sorry

def nodes_inside (t : Triangle) (nodes : List Point) : Prop := 
  nodes.length = 2 ∧ ∀ p, p ∈ nodes → strictly_inside p t

theorem line_through_nodes (t : Triangle) (node1 node2 : Point) (h_inside : nodes_inside t [node1, node2]) :
   ∃ (v : Point), v ∈ [t.A, t.B, t.C] ∨
   (∃ (s : Triangle -> Point -> Point -> Prop), s t node1 node2) := 
sorry

end line_through_nodes_l98_98988


namespace total_matches_equation_l98_98214

theorem total_matches_equation (x : ℕ) (h : ((x * (x - 1)) / 2) = 28) : (1 / 2 : ℚ) * x * (x - 1) = 28 := by
  have h1 : ((x * (x - 1)) / 2 : ℚ) = (1 / 2) * x * (x - 1),
    sorry
  rw ← h1 at h
  exact h

end total_matches_equation_l98_98214


namespace remainder_mod_1000_l98_98957

open Finset

noncomputable def T : Finset ℕ := (range 12).map ⟨λ x, x + 1, λ x y h, by linarith⟩

def m : ℕ := (3 ^ card T) / 2 - 2 * (2 ^ card T) / 2 + 1 / 2

theorem remainder_mod_1000 : m % 1000 = 625 := by
  -- m is defined considering the steps mentioned in the problem
  have hT: card T = 12 := by
    rw [T, card_map, card_range]
    simp
  -- calculations for m
  have h3pow : 3 ^ 12 = 531441 := by norm_num
  have h2pow : 2 ^ 12 = 4096 := by norm_num
  have h2powDoubled : 2 * 4096 = 8192 := by norm_num
  have hend: (531441 - 8192 + 1) / 2 = 261625 := by norm_num
  -- combining all
  rw [m, hT, h3pow, h2pow, h2powDoubled, hend]
  norm_num
  sorry

end remainder_mod_1000_l98_98957


namespace axis_of_symmetry_l98_98940

-- Given conditions
variables {b c : ℝ}
axiom eq_roots : ∃ (x1 x2 : ℝ), (x1 = -1 ∧ x2 = 2) ∧ (x1 + x2 = -b) ∧ (x1 * x2 = c)

-- Question translation to Lean statement
theorem axis_of_symmetry : 
  ∀ b c, 
  (∃ (x1 x2 : ℝ), x1 = -1 ∧ x2 = 2 ∧ x1 + x2 = -b ∧ x1 * x2 = c) 
  → -b / 2 = 1 / 2 := 
by 
  sorry

end axis_of_symmetry_l98_98940


namespace product_modulo_25_l98_98537

theorem product_modulo_25 : 
  (123 ≡ 3 [MOD 25]) → 
  (456 ≡ 6 [MOD 25]) → 
  (789 ≡ 14 [MOD 25]) → 
  (123 * 456 * 789 ≡ 2 [MOD 25]) := 
by 
  intros h1 h2 h3 
  sorry

end product_modulo_25_l98_98537


namespace number_of_sports_books_l98_98496

def total_books : ℕ := 58
def school_books : ℕ := 19
def sports_books (total_books school_books : ℕ) : ℕ := total_books - school_books

theorem number_of_sports_books : sports_books total_books school_books = 39 := by
  -- proof goes here
  sorry

end number_of_sports_books_l98_98496


namespace gas_volume_at_31_degrees_l98_98477

theorem gas_volume_at_31_degrees :
  (∀ T V : ℕ, (T = 45 → V = 30) ∧ (∀ k, T = 45 - 2 * k → V = 30 - 3 * k)) →
  ∃ V, (T = 31) ∧ (V = 9) :=
by
  -- The proof would go here
  sorry

end gas_volume_at_31_degrees_l98_98477


namespace paper_clips_distribution_l98_98787

theorem paper_clips_distribution (P c b : ℕ) (hP : P = 81) (hc : c = 9) (hb : b = P / c) : b = 9 :=
by
  rw [hP, hc] at hb
  simp at hb
  exact hb

end paper_clips_distribution_l98_98787


namespace missing_number_l98_98750

theorem missing_number (n : ℝ) (h : (0.0088 * 4.5) / (0.05 * n * 0.008) = 990) : n = 0.1 :=
sorry

end missing_number_l98_98750


namespace solve_quadratic_eq_l98_98021

theorem solve_quadratic_eq (a c : ℝ) (h1 : a + c = 31) (h2 : a < c) (h3 : (24:ℝ)^2 - 4 * a * c = 0) : a = 9 ∧ c = 22 :=
by {
  sorry
}

end solve_quadratic_eq_l98_98021


namespace sleeves_add_correct_weight_l98_98402

variable (R W_r W_s S : ℝ)

-- Conditions
def raw_squat : Prop := R = 600
def wraps_add_25_percent : Prop := W_r = R + 0.25 * R
def wraps_vs_sleeves_difference : Prop := W_r = W_s + 120

-- To Prove
theorem sleeves_add_correct_weight (h1 : raw_squat R) (h2 : wraps_add_25_percent R W_r) (h3 : wraps_vs_sleeves_difference W_r W_s) : S = 30 :=
by
  sorry

end sleeves_add_correct_weight_l98_98402


namespace solve_for_t_l98_98317

theorem solve_for_t (s t : ℚ) (h1 : 8 * s + 6 * t = 160) (h2 : s = t + 3) : t = 68 / 7 :=
by
  sorry

end solve_for_t_l98_98317


namespace total_subjects_is_41_l98_98656

-- Define the number of subjects taken by Monica, Marius, and Millie
def subjects_monica := 10
def subjects_marius := subjects_monica + 4
def subjects_millie := subjects_marius + 3

-- Define the total number of subjects taken by all three
def total_subjects := subjects_monica + subjects_marius + subjects_millie

theorem total_subjects_is_41 : total_subjects = 41 := by
  -- This is where the proof would be, but we only need the statement
  sorry

end total_subjects_is_41_l98_98656


namespace intersection_points_l98_98528

variables (a b c : ℝ)
variables (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)

theorem intersection_points :
  ∃ p q : ℝ × ℝ, 
    (p = (0, c) ∨ p = (-1, a - b + c)) ∧ 
    (q = (0, c) ∨ q = (-1, a - b + c)) ∧
    p ≠ q ∧
    (∃ x : ℝ, (x, ax^2 + bx + c) = p) ∧
    (∃ x : ℝ, (x, -ax^3 + bx + c) = q) :=
by
  sorry

end intersection_points_l98_98528


namespace range_of_f_l98_98195

def diamond (x y : ℝ) := (x + y) ^ 2 - x * y

def f (a x : ℝ) := diamond a x

theorem range_of_f (a : ℝ) (h : diamond 1 a = 3) :
  ∃ b : ℝ, ∀ x : ℝ, x > 0 → f a x > b :=
sorry

end range_of_f_l98_98195


namespace rem_product_eq_l98_98966

theorem rem_product_eq 
  (P Q R k : ℤ) 
  (hk : k > 0) 
  (hPQ : P * Q = R) : 
  ((P % k) * (Q % k)) % k = R % k :=
by
  sorry

end rem_product_eq_l98_98966


namespace box_dimensions_l98_98221

-- Given conditions
variables (a b c : ℕ)
axiom h1 : a + c = 17
axiom h2 : a + b = 13
axiom h3 : b + c = 20

theorem box_dimensions : a = 5 ∧ b = 8 ∧ c = 12 :=
by {
  -- These parts will contain the actual proof, which we omit for now
  sorry
}

end box_dimensions_l98_98221


namespace greatest_triangle_perimeter_l98_98374

theorem greatest_triangle_perimeter :
  ∃ x : ℕ, (x > 4) ∧ (x ≤ 6) ∧ (∀ (y : ℕ), (y > 4) ∧ (y ≤ 6) → 5 * y + 20 = 50) := sorry

end greatest_triangle_perimeter_l98_98374


namespace false_converse_of_vertical_angles_l98_98996

theorem false_converse_of_vertical_angles
  (P : Prop) (Q : Prop) (V : ∀ {A B C D : Type}, (A = B ∧ C = D) → P) (C1 : P → Q) :
  ¬ (Q → P) :=
sorry

end false_converse_of_vertical_angles_l98_98996


namespace inner_cube_surface_area_l98_98147

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h : surface_area_outer_cube = 54) : 
  ∃ (surface_area_inner_cube : ℝ), surface_area_inner_cube = 18 :=
by 
  let side_length_outer_cube := real.sqrt (surface_area_outer_cube / 6)
  let diameter_sphere := side_length_outer_cube
  let side_length_inner_cube := diameter_sphere / real.sqrt 3
  let surface_area_inner_cube := 6 * (side_length_inner_cube ^ 2)
  use surface_area_inner_cube
  have h1 : side_length_outer_cube = 3 := by 
    rw [real.sqrt_eq_rpow, div_eq_mul_inv, mul_comm, ← real.rpow_mul, real.rpow_nat_cast,
        ← pow_two, h, real.rpow_two]
  have h2 : diameter_sphere = 3 := by rw [← h1]
  have h3 : side_length_inner_cube = real.sqrt 3 := by rw [h2, div_mul, mul_inv_cancel (real.sqrt_ne_zero)]
  have h4 : surface_area_inner_cube = 6 * 3 := by rw [h3, real.sqrt_mul_self, mul_div_cancel' _ (real.sqrt_ne_zero)]
  exact eq.symm (eq.trans h4 h)

end inner_cube_surface_area_l98_98147


namespace min_neg_condition_l98_98908

theorem min_neg_condition (a : ℝ) (x : ℝ) :
  (∀ x : ℝ, min (2^(x-1) - 3^(4-x) + a) (a + 5 - x^3 - 2*x) < 0) → a < -7 :=
sorry

end min_neg_condition_l98_98908


namespace greatest_possible_perimeter_l98_98367

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l98_98367


namespace pool_water_amount_correct_l98_98723

noncomputable def water_in_pool_after_ten_hours : ℝ :=
  let h1 := 8
  let h2_3 := 10 * 2
  let h4_5 := 14 * 2
  let h6 := 12
  let h7 := 12 - 8
  let h8 := 12 - 18
  let h9 := 12 - 24
  let h10 := 6
  h1 + h2_3 + h4_5 + h6 + h7 + h8 + h9 + h10

theorem pool_water_amount_correct :
  water_in_pool_after_ten_hours = 60 := 
sorry

end pool_water_amount_correct_l98_98723


namespace inner_cube_surface_area_l98_98146

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l98_98146


namespace transformed_system_solution_l98_98626

theorem transformed_system_solution 
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h1 : a1 * 3 + b1 * 4 = c1)
  (h2 : a2 * 3 + b2 * 4 = c2) :
  (3 * a1 * 5 + 4 * b1 * 5 = 5 * c1) ∧ (3 * a2 * 5 + 4 * b2 * 5 = 5 * c2) :=
by 
  sorry

end transformed_system_solution_l98_98626


namespace inner_cube_surface_area_l98_98151

-- Define the side length of the outer cube from its surface area.
def side_length_of_cube (A : ℝ) : ℝ := real.sqrt (A / 6)

-- Define the diameter of the sphere inscribed in the outer cube.
def diameter_of_sphere (s : ℝ) : ℝ := s

-- Define the side length of the inner cube inscribed in the sphere.
def side_length_of_inner_cube (d : ℝ) : ℝ := d / real.sqrt 3

-- Define the surface area of a cube given its side length.
def surface_area_of_cube (l : ℝ) : ℝ := 6 * l^2

theorem inner_cube_surface_area (A : ℝ) (h1 : A = 54) :
  surface_area_of_cube (side_length_of_inner_cube (diameter_of_sphere (side_length_of_cube A))) = 18 :=
by
  -- leave the proof blank
  sorry

end inner_cube_surface_area_l98_98151


namespace solution_set_of_inequality_l98_98437

theorem solution_set_of_inequality (x : ℝ) :
  (|x| - 2) * (x - 1) ≥ 0 ↔ (-2 ≤ x ∧ x ≤ 1) ∨ (x ≥ 2) :=
by
  sorry

end solution_set_of_inequality_l98_98437


namespace greatest_possible_perimeter_l98_98382

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l98_98382


namespace event_probabilities_equal_l98_98760

variables (u j p b : ℝ)

-- Basic assumptions stated in the problem
axiom (hu_gt_hj : u > j)
axiom (hb_gt_hp : b > p)

-- Define the probabilities of events A and B
def prob_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def prob_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

-- The statement to be proved
theorem event_probabilities_equal : prob_A u j p b = prob_B u j p b :=
  sorry

end event_probabilities_equal_l98_98760


namespace correct_option_B_l98_98036

theorem correct_option_B (x : ℝ) : (1 - x)^2 = 1 - 2 * x + x^2 :=
sorry

end correct_option_B_l98_98036


namespace greatest_perimeter_of_triangle_l98_98339

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), 
    4 < x ∧ x < 20 / 3 ∧ 
    (x + 4 * x + 20 = 50) :=
by 
  sorry

end greatest_perimeter_of_triangle_l98_98339


namespace cube_volume_l98_98081

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98081


namespace quarterback_passes_left_l98_98121

noncomputable def number_of_passes (L : ℕ) : Prop :=
  let R := 2 * L
  let C := L + 2
  L + R + C = 50

theorem quarterback_passes_left : ∃ L, number_of_passes L ∧ L = 12 := by
  sorry

end quarterback_passes_left_l98_98121


namespace inner_cube_surface_area_l98_98163

theorem inner_cube_surface_area
  (S : ℝ) (hS : S = 54)
  (cube_side_length : ℝ) (h_cube_side_length : cube_side_length = sqrt (54 / 6))
  (sphere_diameter : ℝ) (h_sphere_diameter : sphere_diameter = cube_side_length)
  (inner_cube_diagonal : ℝ) (h_inner_cube_diagonal : inner_cube_diagonal = sphere_diameter)
  (inner_cube_side_length : ℝ) (h_inner_cube_side_length : inner_cube_side_length = sqrt (inner_cube_diagonal^2 / 3)) :
  6 * inner_cube_side_length^2 = 18 :=
by sorry

end inner_cube_surface_area_l98_98163


namespace speed_of_stream_l98_98499

theorem speed_of_stream
  (V S : ℝ)
  (h1 : 27 = 9 * (V - S))
  (h2 : 81 = 9 * (V + S)) :
  S = 3 :=
by
  sorry

end speed_of_stream_l98_98499


namespace inner_cube_surface_area_l98_98160

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l98_98160


namespace yellow_ball_impossible_at_least_one_red_ball_probability_of_red_ball_find_x_l98_98512

-- Given conditions
def initial_white_balls := 8
def initial_red_balls := 12
def total_balls := initial_white_balls + initial_red_balls

-- Question 1(a): Drawing a yellow ball is impossible
theorem yellow_ball_impossible (total_balls : ℕ) : false :=
by 
  sorry -- The proof would go here

-- Question 1(b): Probability of drawing at least one red ball
theorem at_least_one_red_ball (total_balls : ℕ) (drawn_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) (total_balls = white_balls + red_balls) : ℝ :=
by
  have h : white_balls < drawn_balls → 1 = 1 :=
  by
    sorry -- The proof would go here
  h

-- Question 2: Probability of drawing a red ball at random
theorem probability_of_red_ball (red_balls white_balls : ℕ) : ℝ :=
  red_balls / (red_balls + white_balls)

-- Question 3: Finding x given the probability of drawing a white ball is 4/5
theorem find_x (initial_white_balls initial_red_balls : ℕ) (draw_white_prob : ℝ) : ℕ :=
by
  let x := initial_white_balls + 8 -- filter x from the probability 4/5 assumption
  sorry -- The proof would go here

end yellow_ball_impossible_at_least_one_red_ball_probability_of_red_ball_find_x_l98_98512


namespace sequence_term_divisible_by_n_l98_98661

theorem sequence_term_divisible_by_n (n : ℕ) (hn1 : 1 < n) (hn_odd : n % 2 = 1) :
  ∃ k : ℕ, 1 ≤ k ∧ k < n ∧ n ∣ (2^k - 1) :=
by
  sorry

end sequence_term_divisible_by_n_l98_98661


namespace find_wind_speed_l98_98863

-- Definitions from conditions
def speed_with_wind (j w : ℝ) := (j + w) * 6 = 3000
def speed_against_wind (j w : ℝ) := (j - w) * 9 = 3000

-- Theorem to prove the wind speed is 83.335 mph
theorem find_wind_speed (j w : ℝ) (h1 : speed_with_wind j w) (h2 : speed_against_wind j w) : w = 83.335 :=
by 
  -- Here we would prove the theorem using the given conditions
  sorry

end find_wind_speed_l98_98863


namespace rationalize_denominator_l98_98815

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98815


namespace inequality_k_distance_comparison_l98_98562

theorem inequality_k (k : ℝ) (x : ℝ) : 
  -3 < k ∧ k ≤ 0 → 2 * k * x^2 + k * x - 3/8 < 0 := sorry

theorem distance_comparison (a b : ℝ) (hab : a ≠ b) : 
  (abs ((a^2 + b^2) / 2 - (a + b)^2 / 4) > abs (a * b - (a + b)^2 / 4)) := sorry

end inequality_k_distance_comparison_l98_98562


namespace range_of_a_l98_98494

theorem range_of_a (a : ℝ) : 
  ¬(∀ x : ℝ, x^2 - 5 * x + 15 / 2 * a <= 0) -> a > 5 / 6 :=
by
  sorry

end range_of_a_l98_98494


namespace cube_volume_l98_98091

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98091


namespace max_perimeter_l98_98387

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l98_98387


namespace rationalize_cube_root_identity_l98_98420

theorem rationalize_cube_root_identity :
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  a^3 = 5 ∧ b^3 = 4 ∧ a - b ≠ 0 ∧
  (X + Y + Z + W) = 62 :=
by
  -- Define a and b
  let a := (5 : ℝ)^(1/3)
  let b := (4 : ℝ)^(1/3)
  -- Rationalize using identity a^3 - b^3 = (a - b)(a^2 + ab + b^2)
  have h1 : a^3 = 5, by sorry
  have h2 : b^3 = 4, by sorry
  have h3 : a - b ≠ 0, by sorry
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  -- Conclude the sum X + Y + Z + W = 62
  have h4 : (X + Y + Z + W) = 62, by sorry
  -- Returning the combined statement
  exact ⟨h1, h2, h3, h4⟩

end rationalize_cube_root_identity_l98_98420


namespace inner_cube_surface_area_l98_98132

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l98_98132


namespace wang_hao_height_is_158_l98_98899

/-- Yao Ming's height in meters. -/
def yao_ming_height : ℝ := 2.29

/-- Wang Hao is 0.71 meters shorter than Yao Ming. -/
def height_difference : ℝ := 0.71

/-- Wang Hao's height in meters. -/
def wang_hao_height : ℝ := yao_ming_height - height_difference

theorem wang_hao_height_is_158 :
  wang_hao_height = 1.58 :=
by
  sorry

end wang_hao_height_is_158_l98_98899


namespace tablecloth_overhang_l98_98412

theorem tablecloth_overhang (d r l overhang1 overhang2 : ℝ) (h1 : d = 0.6) (h2 : r = d / 2) (h3 : l = 1) 
  (h4 : overhang1 = 0.5) (h5 : overhang2 = 0.3) :
  ∃ overhang3 overhang4 : ℝ, overhang3 = 0.33 ∧ overhang4 = 0.52 := 
sorry

end tablecloth_overhang_l98_98412


namespace parabola_line_intersect_solutions_count_l98_98598

theorem parabola_line_intersect_solutions_count :
  ∃ b1 b2 : ℝ, (b1 ≠ b2 ∧ (b1^2 - b1 - 3 = 0) ∧ (b2^2 - b2 - 3 = 0)) :=
by
  sorry

end parabola_line_intersect_solutions_count_l98_98598


namespace avg_calculation_l98_98235

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem avg_calculation :
  avg4 (avg2 1 2) (avg2 3 1) (avg2 2 0) (avg2 1 1) = 11 / 8 := by
  sorry

end avg_calculation_l98_98235


namespace compute_g3_l98_98932

def g (x : ℤ) : ℤ := 7 * x - 3

theorem compute_g3: g (g (g 3)) = 858 :=
by
  sorry

end compute_g3_l98_98932


namespace percentage_greater_l98_98505

theorem percentage_greater (x : ℝ) (h1 : x = 96) (h2 : x > 80) : ((x - 80) / 80) * 100 = 20 :=
by
  sorry

end percentage_greater_l98_98505


namespace operation_result_l98_98292

-- Define the operation
def operation (a b : ℝ) : ℝ := (a - b) ^ 3

theorem operation_result (x y : ℝ) : operation ((x - y) ^ 3) ((y - x) ^ 3) = -8 * (y - x) ^ 9 := 
  sorry

end operation_result_l98_98292


namespace recur_decimal_times_nine_l98_98288

theorem recur_decimal_times_nine : 
  (0.3333333333333333 : ℝ) * 9 = 3 :=
by
  -- Convert 0.\overline{3} to a fraction
  have h1 : (0.3333333333333333 : ℝ) = (1 / 3 : ℝ), by sorry
  -- Perform multiplication and simplification
  calc
    (0.3333333333333333 : ℝ) * 9 = (1 / 3 : ℝ) * 9 : by rw h1
                              ... = (1 * 9) / 3 : by sorry
                              ... = 9 / 3 : by sorry
                              ... = 3 : by sorry

end recur_decimal_times_nine_l98_98288


namespace cube_volume_from_surface_area_l98_98061

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98061


namespace inequality_inequality_triangle_abc_l98_98397

variables (A B C : ℝ)
variables (a b c : ℝ)
variables (R r : ℝ)
variables [Fact (a > 0)] [Fact (b > 0)] [Fact (c > 0)]
variables [Fact (R > 0)] [Fact (r > 0)]

noncomputable def cot (x : ℝ) := real.cos x / real.sin x

theorem inequality_inequality_triangle_abc
  (h₁: a > 0) (h₂: b > 0) (h₃: c > 0) (h₄: R > 0) (h₅: r > 0) :
  6*r ≤ a * cot A + b * cot B + c * cot C ∧ a * cot A + b * cot B + c * cot C ≤ 3*R :=
sorry

end inequality_inequality_triangle_abc_l98_98397


namespace points_three_units_away_from_neg3_l98_98246

theorem points_three_units_away_from_neg3 (x : ℝ) : (abs (x + 3) = 3) ↔ (x = 0 ∨ x = -6) :=
by
  sorry

end points_three_units_away_from_neg3_l98_98246


namespace division_value_l98_98322

theorem division_value (x y : ℝ) (h1 : (x - 5) / y = 7) (h2 : (x - 14) / 10 = 4) : y = 7 :=
sorry

end division_value_l98_98322


namespace toy_value_l98_98883

theorem toy_value
  (t : ℕ)                 -- total number of toys
  (W : ℕ)                 -- total worth in dollars
  (v : ℕ)                 -- value of one specific toy
  (x : ℕ)                 -- value of one of the other toys
  (h1 : t = 9)            -- condition 1: total number of toys
  (h2 : W = 52)           -- condition 2: total worth
  (h3 : v = 12)           -- condition 3: value of one specific toy
  (h4 : (t - 1) * x + v = W) -- condition 4: equation based on the problem
  : x = 5 :=              -- theorem statement: other toy's value
by {
  -- proof goes here
  sorry
}

end toy_value_l98_98883


namespace goldie_earnings_l98_98201

theorem goldie_earnings
  (hourly_wage : ℕ := 5)
  (hours_last_week : ℕ := 20)
  (hours_this_week : ℕ := 30) :
  hourly_wage * hours_last_week + hourly_wage * hours_this_week = 250 :=
by
  sorry

end goldie_earnings_l98_98201


namespace rationalize_sqrt_fraction_l98_98834

theorem rationalize_sqrt_fraction : sqrt (5 / 12) = sqrt 15 / 6 := 
  sorry

end rationalize_sqrt_fraction_l98_98834


namespace last_three_digits_of_2_pow_6000_l98_98997

theorem last_three_digits_of_2_pow_6000 (h : 2^200 ≡ 1 [MOD 800]) : (2^6000 ≡ 1 [MOD 800]) :=
sorry

end last_three_digits_of_2_pow_6000_l98_98997


namespace factorization_a4_plus_4_l98_98847

theorem factorization_a4_plus_4 (a : ℝ) : a^4 + 4 = (a^2 - 2*a + 2) * (a^2 + 2*a + 2) :=
by sorry

end factorization_a4_plus_4_l98_98847


namespace chemist_salt_solution_l98_98866

theorem chemist_salt_solution (x : ℝ) 
  (hx : 0.60 * x = 0.20 * (1 + x)) : x = 0.5 :=
sorry

end chemist_salt_solution_l98_98866


namespace solve_for_x_l98_98999

theorem solve_for_x (x : ℝ) : (3 / 2) * x - 3 = 15 → x = 12 := 
by
  sorry

end solve_for_x_l98_98999


namespace rationalize_sqrt_fraction_l98_98832

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l98_98832


namespace greatest_possible_perimeter_l98_98334

-- Define a predicate for the integer side lengths of the triangle
def triangle_sides (x : ℕ) : Prop :=
  (4 * x < x + 20) ∧
  (x + 4 * x > 20) ∧
  (x + 20 > 4 * x)

-- Define the proposition for the maximum perimeter
def max_perimeter_of_triangle_with_sides : ℕ :=
  if ∃ x : ℕ, triangle_sides x then
    50
  else
    0

-- Proving that the maximum possible perimeter is 50
theorem greatest_possible_perimeter :
  max_perimeter_of_triangle_with_sides = 50 :=
sorry

end greatest_possible_perimeter_l98_98334


namespace greatest_perimeter_l98_98350

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l98_98350


namespace initial_mixture_volume_l98_98619

theorem initial_mixture_volume (x : ℝ) (hx1 : 0.10 * x + 10 = 0.28 * (x + 10)) : x = 40 :=
by
  sorry

end initial_mixture_volume_l98_98619


namespace angle_between_planes_is_correct_l98_98188

open Real

noncomputable def normal_vector_plane1 : (ℝ × ℝ × ℝ) := (1, 1, 3)
noncomputable def normal_vector_plane2 : (ℝ × ℝ × ℝ) := (0, 1, 1)

noncomputable def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

noncomputable def magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

noncomputable def cos_angle : ℝ :=
  dot_product normal_vector_plane1 normal_vector_plane2 / (magnitude normal_vector_plane1 * magnitude normal_vector_plane2)

noncomputable def angle_between_planes : ℝ :=
  arccos cos_angle

theorem angle_between_planes_is_correct : 
  abs (angle_between_planes - (31 * (π / 180) + (28 / 60) * (π / 180) + (56 / 3600) * (π / 180))) < 1e-6 :=
sorry

end angle_between_planes_is_correct_l98_98188


namespace volume_to_surface_area_ratio_l98_98891

-- Define the structure of the object consisting of unit cubes
structure CubicObject where
  volume : ℕ
  surface_area : ℕ

-- Define a specific cubic object based on given conditions
def specialCubicObject : CubicObject := {
  volume := 8,
  surface_area := 29
}

-- Statement to prove the ratio of the volume to the surface area
theorem volume_to_surface_area_ratio :
  (specialCubicObject.volume : ℚ) / (specialCubicObject.surface_area : ℚ) = 8 / 29 := by
  sorry

end volume_to_surface_area_ratio_l98_98891


namespace inner_cube_surface_area_l98_98126

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l98_98126


namespace find_b_compare_f_l98_98610

-- Definition from conditions
def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := -x^2 + b*x + c

-- Part 1: Prove that b = 4
theorem find_b (b c : ℝ) (h : ∀ x : ℝ, f (2 + x) b c = f (2 - x) b c) : b = 4 :=
sorry

-- Part 2: Prove the comparison of f(\frac{5}{4}) and f(-a^2 - a + 1)
theorem compare_f (c : ℝ) (a : ℝ) (h₁ : ∀ x : ℝ, f (2 + x) 4 c = f (2 - x) 4 c) (h₂ : f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c) :
f (5/4) 4 c < f (-(a^2 + a - 1)) 4 c := 
sorry

end find_b_compare_f_l98_98610


namespace odd_function_symmetry_l98_98243

def f (x : ℝ) : ℝ := x^3 + x

-- Prove that f(-x) = -f(x)
theorem odd_function_symmetry : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end odd_function_symmetry_l98_98243


namespace expression_c_is_positive_l98_98228

def A : ℝ := 2.1
def B : ℝ := -0.5
def C : ℝ := -3.0
def D : ℝ := 4.2
def E : ℝ := 0.8

theorem expression_c_is_positive : |C| + |B| > 0 :=
by {
  sorry
}

end expression_c_is_positive_l98_98228


namespace cube_volume_l98_98084

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98084


namespace greatest_possible_perimeter_l98_98370

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l98_98370


namespace inner_cube_surface_area_l98_98158

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l98_98158


namespace rationalize_sqrt_fraction_denom_l98_98802

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l98_98802


namespace outer_circle_increase_l98_98023

theorem outer_circle_increase : 
  let R_o := 6
  let R_i := 4
  let R_i_new := (3 : ℝ)  -- 4 * (3/4)
  let A_original := 20 * Real.pi  -- π * (6^2 - 4^2)
  let A_new := 72 * Real.pi  -- 3.6 * A_original
  ∃ (x : ℝ), 
    let R_o_new := R_o * (1 + x / 100)
    π * R_o_new^2 - π * R_i_new^2 = A_new →
    x = 50 := 
sorry

end outer_circle_increase_l98_98023


namespace race_distance_l98_98510

theorem race_distance {d a b c : ℝ} 
    (h1 : d / a = (d - 25) / b)
    (h2 : d / b = (d - 15) / c)
    (h3 : d / a = (d - 35) / c) :
  d = 75 :=
by
  sorry

end race_distance_l98_98510


namespace problem_1_problem_2_problem_3_l98_98297

-- Condition: x1 and x2 are the roots of the quadratic equation x^2 - 2(m+2)x + m^2 = 0
variables {x1 x2 m : ℝ}
axiom roots_quadratic_equation : x1^2 - 2*(m+2) * x1 + m^2 = 0 ∧ x2^2 - 2*(m+2) * x2 + m^2 = 0

-- 1. When m = 0, the roots of the equation are 0 and 4
theorem problem_1 (h : m = 0) : x1 = 0 ∧ x2 = 4 :=
by 
  sorry

-- 2. If (x1 - 2)(x2 - 2) = 41, then m = 9
theorem problem_2 (h : (x1 - 2) * (x2 - 2) = 41) : m = 9 :=
by
  sorry

-- 3. Given an isosceles triangle ABC with one side length 9, if x1 and x2 are the lengths of the other two sides, 
--    prove that the perimeter is 19.
theorem problem_3 (h1 : x1 + x2 > 9) (h2 : 9 + x1 > x2) (h3 : 9 + x2 > x1) : x1 = 1 ∧ x2 = 9 ∧ (x1 + x2 + 9) = 19 :=
by 
  sorry

end problem_1_problem_2_problem_3_l98_98297


namespace intermediate_value_theorem_example_l98_98918

theorem intermediate_value_theorem_example (f : ℝ → ℝ) :
  f 2007 < 0 → f 2008 < 0 → f 2009 > 0 → ∃ x, 2007 < x ∧ x < 2008 ∧ f x = 0 :=
by
  sorry

end intermediate_value_theorem_example_l98_98918


namespace intersecting_lines_sum_constant_l98_98678

theorem intersecting_lines_sum_constant
  (c d : ℝ)
  (h1 : 3 = (1 / 3) * 3 + c)
  (h2 : 3 = (1 / 3) * 3 + d) :
  c + d = 4 :=
by
  sorry

end intersecting_lines_sum_constant_l98_98678


namespace rhombus_diagonal_l98_98671

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 20) (h2 : area = 170) :
  (area = (d1 * d2) / 2) → d2 = 17 :=
by
  sorry

end rhombus_diagonal_l98_98671


namespace intersection_is_correct_complement_is_correct_l98_98755

open Set

variable {U : Set ℝ} (A B : Set ℝ)

-- Define the universal set U
def U_def : Set ℝ := { x | 1 < x ∧ x < 7 }

-- Define set A
def A_def : Set ℝ := { x | 2 ≤ x ∧ x < 5 }

-- Define set B using the simplified condition from the inequality
def B_def : Set ℝ := { x | x ≥ 3 }

-- Proof statement that A ∩ B is as specified
theorem intersection_is_correct :
  (A_def ∩ B_def) = { x : ℝ | 3 ≤ x ∧ x < 5 } := by
  sorry

-- Proof statement for the complement of A relative to U
theorem complement_is_correct :
  (U_def \ A_def) = { x : ℝ | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 7) } := by
  sorry

end intersection_is_correct_complement_is_correct_l98_98755


namespace martha_knits_hat_in_2_hours_l98_98964

-- Definitions based on given conditions
variables (H : ℝ)
def knit_times (H : ℝ) : ℝ := H + 3 + 2 + 3 + 6

def total_knitting_time (H : ℝ) : ℝ := 3 * knit_times H

-- The main statement to be proven
theorem martha_knits_hat_in_2_hours (H : ℝ) (h : total_knitting_time H = 48) : H = 2 := 
by
  sorry

end martha_knits_hat_in_2_hours_l98_98964


namespace total_tiles_to_be_replaced_l98_98715

-- Define the given conditions
def horizontal_paths : List ℕ := [30, 50, 30, 20, 20, 50]
def vertical_paths : List ℕ := [20, 50, 20, 50, 50]
def intersections : ℕ := List.sum [2, 3, 3, 4, 4]

-- Problem statement: Prove that the total number of tiles to be replaced is 374
theorem total_tiles_to_be_replaced : List.sum horizontal_paths + List.sum vertical_paths - intersections = 374 := 
by sorry

end total_tiles_to_be_replaced_l98_98715


namespace solution_unique_for_alpha_neg_one_l98_98779

noncomputable def alpha : ℝ := sorry

axiom alpha_nonzero : alpha ≠ 0

def functional_eqn (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f (f (x + y)) = f (x + y) + f (x) * f (y) + alpha * x * y

theorem solution_unique_for_alpha_neg_one (f : ℝ → ℝ) :
  (alpha = -1 → (∀ x : ℝ, f x = x)) ∧ (alpha ≠ -1 → ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, functional_eqn f x y) :=
sorry

end solution_unique_for_alpha_neg_one_l98_98779


namespace rationalize_sqrt_fraction_denom_l98_98799

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l98_98799


namespace diet_soda_bottles_l98_98569

/-- Define variables for the number of bottles. -/
def total_bottles : ℕ := 38
def regular_soda : ℕ := 30

/-- Define the problem of finding the number of diet soda bottles -/
def diet_soda := total_bottles - regular_soda

/-- Claim that the number of diet soda bottles is 8 -/
theorem diet_soda_bottles : diet_soda = 8 :=
by
  sorry

end diet_soda_bottles_l98_98569


namespace product_of_repeating_decimal_l98_98287

theorem product_of_repeating_decimal (x : ℝ) (h : x = 1 / 3) : x * 9 = 3 :=
by
  rw [h]
  norm_num
  sorry

end product_of_repeating_decimal_l98_98287


namespace union_A_B_correct_l98_98300

def A : Set ℕ := {0, 1}
def B : Set ℕ := {x | 0 < x ∧ x < 3}

theorem union_A_B_correct : A ∪ B = {0, 1, 2} :=
by sorry

end union_A_B_correct_l98_98300


namespace sin_double_angle_identity_l98_98615

theorem sin_double_angle_identity (alpha : ℝ) (h : Real.cos (Real.pi / 4 - alpha) = -4 / 5) : 
  Real.sin (2 * alpha) = 7 / 25 :=
by
  sorry

end sin_double_angle_identity_l98_98615


namespace rationalize_denominator_l98_98822

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98822


namespace determine_placemat_length_l98_98571

theorem determine_placemat_length :
  ∃ (y : ℝ), ∀ (r : ℝ), r = 5 →
  (∀ (n : ℕ), n = 8 →
  (∀ (w : ℝ), w = 1 →
  y = 10 * Real.sin (5 * Real.pi / 16))) :=
by
  sorry

end determine_placemat_length_l98_98571


namespace arrange_books_correct_l98_98771

def math_books : Nat := 4
def history_books : Nat := 4

def arrangements (m h : Nat) : Nat := sorry

theorem arrange_books_correct :
  arrangements math_books history_books = 576 := sorry

end arrange_books_correct_l98_98771


namespace troy_initial_straws_l98_98984

theorem troy_initial_straws (total_piglets : ℕ) (straws_per_piglet : ℕ)
  (fraction_adult_pigs : ℚ) (fraction_piglets : ℚ) 
  (adult_pigs_straws : ℕ) (piglets_straws : ℕ) 
  (total_straws : ℕ) (initial_straws : ℚ) :
  total_piglets = 20 →
  straws_per_piglet = 6 →
  fraction_adult_pigs = 3 / 5 →
  fraction_piglets = 3 / 5 →
  piglets_straws = total_piglets * straws_per_piglet →
  adult_pigs_straws = piglets_straws →
  total_straws = piglets_straws + adult_pigs_straws →
  (fraction_adult_pigs + fraction_piglets) * initial_straws = total_straws →
  initial_straws = 200 := 
by 
  sorry

end troy_initial_straws_l98_98984


namespace rationalize_sqrt_fraction_l98_98828

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l98_98828


namespace express_nineteen_in_base_3_l98_98586

theorem express_nineteen_in_base_3 :
  nat.to_digits 3 19 = [2, 0, 1] :=
by
  sorry

end express_nineteen_in_base_3_l98_98586


namespace evaluate_expression_l98_98618

theorem evaluate_expression (x : ℕ) (h : x = 5) : 2 * x ^ 2 + 5 = 55 := by
  sorry

end evaluate_expression_l98_98618


namespace range_of_a_l98_98323

-- Define the problem statement in Lean 4
theorem range_of_a (a : ℝ) : (∀ x : ℝ, ((x^2 - (a-1)*x + 1) > 0)) → (-1 < a ∧ a < 3) :=
by
  intro h
  sorry -- Proof to be filled in

end range_of_a_l98_98323


namespace cube_volume_from_surface_area_l98_98062

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98062


namespace max_area_quadrilateral_l98_98913

open Real

theorem max_area_quadrilateral (a b c d α : ℝ) (h1 : a * b = 1) (h2 : b * c = 1) (h3 : c * d = 1) (h4 : d * a = 1) :
  ∃ S, S = 1 ∧ ∀ x, x ≤ S := by 
  sorry

end max_area_quadrilateral_l98_98913


namespace inner_cube_surface_area_l98_98137

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l98_98137


namespace sum_odd_product_even_l98_98795

theorem sum_odd_product_even (a b : ℤ) (h1 : ∃ k : ℤ, a = 2 * k) 
                             (h2 : ∃ m : ℤ, b = 2 * m + 1) 
                             (h3 : ∃ n : ℤ, a + b = 2 * n + 1) : 
  ∃ p : ℤ, a * b = 2 * p := 
  sorry

end sum_odd_product_even_l98_98795


namespace problem_l98_98521

-- Definitions
variables {a b : ℝ}
def is_root (p : ℝ → ℝ) (x : ℝ) : Prop := p x = 0

-- Root condition using the given equation
def quadratic_eq (x : ℝ) : ℝ := (x - 3) * (2 * x + 7) - (x^2 - 11 * x + 28)

-- Statement to prove
theorem problem (ha : is_root quadratic_eq a) (hb : is_root quadratic_eq b) (h_distinct : a ≠ b):
  (a + 2) * (b + 2) = -66 :=
sorry

end problem_l98_98521


namespace correct_sampling_methods_l98_98662

theorem correct_sampling_methods :
  (let num_balls := 1000
   let red_box := 500
   let blue_box := 200
   let yellow_box := 300
   let sample_balls := 100
   let num_students := 20
   let selected_students := 3
   let q1_method := "stratified"
   let q2_method := "simple_random"
   q1_method = "stratified" ∧ q2_method = "simple_random") := sorry

end correct_sampling_methods_l98_98662


namespace max_area_cross_section_of_prism_l98_98710

noncomputable def prism_vertex_A : ℝ × ℝ × ℝ := (3, 0, 0)
noncomputable def prism_vertex_B : ℝ × ℝ × ℝ := (-3, 0, 0)
noncomputable def prism_vertex_C : ℝ × ℝ × ℝ := (0, 3 * Real.sqrt 3, 0)
noncomputable def plane_eq (x y z : ℝ) : ℝ := 2 * x - 3 * y + 6 * z

-- Statement
theorem max_area_cross_section_of_prism (h : ℝ) (A B C : ℝ × ℝ × ℝ)
  (plane : ℝ → ℝ → ℝ → ℝ) (cond_h : h = 5)
  (cond_A : A = prism_vertex_A) (cond_B : B = prism_vertex_B) 
  (cond_C : C = prism_vertex_C) (cond_plane : ∀ x y z, plane x y z = 2 * x - 3 * y + 6 * z - 30) : 
  ∃ cross_section : ℝ, cross_section = 0 :=
by
  sorry

end max_area_cross_section_of_prism_l98_98710


namespace max_value_m_l98_98251

theorem max_value_m (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  (∀ (m : ℝ), (4 / (1 - x) ≥ m - 1 / x)) ↔ (∃ (m : ℝ), m ≤ 9) :=
by
  sorry

end max_value_m_l98_98251


namespace fair_dice_game_l98_98696

theorem fair_dice_game : 
  let outcomes := [(x, y) | x <- [1,2,3,4,5,6], y <- [1,2,3,4,5,6]] in
  let odd_sum := [(x, y) | (x, y) ∈ outcomes, (x + y) % 2 = 1] in
  let even_sum := [(x, y) | (x, y) ∈ outcomes, (x + y) % 2 = 0] in
  probability (odd_sum.length.to_real / outcomes.length.to_real) = probability (even_sum.length.to_real / outcomes.length.to_real) :=
sorry

end fair_dice_game_l98_98696


namespace probability_equality_l98_98762

variables (u j p b : ℝ)
variables (hu : u > j) (hb : b > p)

def probability_A : ℝ :=
  (u * b * j * p) / ((u + p) * (u + b) * (j + p) * (j + b))

def probability_B : ℝ :=
  (u * p * j * b) / ((u + b) * (u + p) * (j + p) * (j + b))

theorem probability_equality (hu : u > j) (hb : b > p) : probability_A u j p b = probability_B u j p b :=
by sorry

end probability_equality_l98_98762


namespace jane_evening_pages_l98_98222

theorem jane_evening_pages :
  ∀ (P : ℕ), (7 * (5 + P) = 105) → P = 10 :=
by
  intros P h
  sorry

end jane_evening_pages_l98_98222


namespace rationalize_denominator_l98_98814

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98814


namespace max_voters_after_T_l98_98048

theorem max_voters_after_T (x : ℕ) (n : ℕ) (y : ℕ) (T : ℕ)  
  (h1 : x <= 10)
  (h2 : x > 0)
  (h3 : (nx + y) ≤ (n + 1) * (x - 1))
  (h4 : ∀ k, (x - k ≥ 0) ↔ (n ≤ T + 5)) :
  ∃ (m : ℕ), m = 5 := 
sorry

end max_voters_after_T_l98_98048


namespace count_valid_orderings_l98_98987

-- Define the houses and conditions
inductive HouseColor where
  | Green
  | Purple
  | Blue
  | Pink
  | X -- Representing the fifth unspecified house

open HouseColor

def validOrderings : List (List HouseColor) :=
  [
    [Green, Blue, Purple, Pink, X], 
    [Green, Blue, X, Purple, Pink],
    [Green, X, Purple, Blue, Pink],
    [X, Pink, Purple, Blue, Green],
    [X, Purple, Pink, Blue, Green],
    [X, Pink, Blue, Purple, Green]
  ] 

-- Prove that there are exactly 6 valid orderings
theorem count_valid_orderings : (validOrderings.length = 6) :=
by
  -- Since we list all possible valid orderings above, just compute the length
  sorry

end count_valid_orderings_l98_98987


namespace tshirt_cost_correct_l98_98273

   -- Definitions of the conditions
   def initial_amount : ℕ := 91
   def cost_of_sweater : ℕ := 24
   def cost_of_shoes : ℕ := 11
   def remaining_amount : ℕ := 50

   -- Define the total cost of the T-shirt purchase
   noncomputable def cost_of_tshirt := 
     initial_amount - remaining_amount - cost_of_sweater - cost_of_shoes

   -- Proof statement that cost_of_tshirt = 6
   theorem tshirt_cost_correct : cost_of_tshirt = 6 := 
   by
     sorry
   
end tshirt_cost_correct_l98_98273


namespace geometric_sequence_fourth_term_l98_98759

theorem geometric_sequence_fourth_term
  (a₁ a₅ : ℕ)
  (r : ℕ)
  (h₁ : a₁ = 3)
  (h₂ : a₅ = 2187)
  (h₃ : a₅ = a₁ * r ^ 4) :
  a₁ * r ^ 3 = 2187 :=
by {
  sorry
}

end geometric_sequence_fourth_term_l98_98759


namespace student_weight_l98_98500

-- Definitions based on conditions
variables (S R : ℝ)

-- Conditions as assertions
def condition1 : Prop := S - 5 = 2 * R
def condition2 : Prop := S + R = 104

-- The statement we want to prove
theorem student_weight (h1 : condition1 S R) (h2 : condition2 S R) : S = 71 :=
by
  sorry

end student_weight_l98_98500


namespace all_statements_true_l98_98424

noncomputable def g : ℝ → ℝ := sorry

axiom g_defined (x : ℝ) : ∃ y, g x = y
axiom g_positive (x : ℝ) : g x > 0
axiom g_multiplicative (a b : ℝ) : g (a) * g (b) = g (a + b)
axiom g_div (a b : ℝ) (h : a > b) : g (a - b) = g (a) / g (b)

theorem all_statements_true :
  (g 0 = 1) ∧
  (∀ a, g (-a) = 1 / g (a)) ∧
  (∀ a, g (a) = (g (3 * a))^(1 / 3)) ∧
  (∀ a b, b > a → g (b - a) < g (b)) :=
by
  sorry

end all_statements_true_l98_98424


namespace temperature_difference_l98_98791

theorem temperature_difference (H L : ℝ) (hH : H = 8) (hL : L = -2) :
  H - L = 10 :=
by
  rw [hH, hL]
  norm_num

end temperature_difference_l98_98791


namespace number_of_adult_tickets_l98_98024

-- Define the parameters of the problem
def price_adult_ticket : ℝ := 5.50
def price_child_ticket : ℝ := 3.50
def total_tickets : ℕ := 21
def total_cost : ℝ := 83.50

-- Define the main theorem to be proven
theorem number_of_adult_tickets : 
  ∃ (A C : ℕ), A + C = total_tickets ∧ 
                (price_adult_ticket * A + price_child_ticket * C = total_cost) ∧ 
                 A = 5 :=
by
  -- The proof content will be filled in later
  sorry

end number_of_adult_tickets_l98_98024


namespace profit_percentage_l98_98559

theorem profit_percentage (C S : ℝ) (h : 17 * C = 16 * S) : (S - C) / C * 100 = 6.25 := by
  sorry

end profit_percentage_l98_98559


namespace jars_of_pickled_mangoes_l98_98518

def total_mangoes := 54
def ratio_ripe := 1/3
def ratio_unripe := 2/3
def kept_unripe_mangoes := 16
def mangoes_per_jar := 4

theorem jars_of_pickled_mangoes : 
  (total_mangoes * ratio_unripe - kept_unripe_mangoes) / mangoes_per_jar = 5 :=
by
  sorry

end jars_of_pickled_mangoes_l98_98518


namespace inner_cube_surface_area_l98_98159

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l98_98159


namespace greatest_possible_perimeter_l98_98368

noncomputable def max_triangle_perimeter (x : ℕ) : ℕ :=
if (x > 4 ∧ 3 * x < 20) then (x + 4 * x + 20) else 0

theorem greatest_possible_perimeter : max_triangle_perimeter 6 = 50 :=
by
  have h1 : 6 > 4 := by decide
  have h2 : 3 * 6 < 20 := by decide
  rw [max_triangle_perimeter]
  rw [if_pos]
  simp
  apply And.intro h1 h2
  sorry

end greatest_possible_perimeter_l98_98368


namespace father_age_when_sum_100_l98_98524

/-- Given the current ages of the mother and father, prove that the father's age will be 51 years old when the sum of their ages is 100. -/
theorem father_age_when_sum_100 (M F : ℕ) (hM : M = 42) (hF : F = 44) :
  ∃ X : ℕ, (M + X) + (F + X) = 100 ∧ F + X = 51 :=
by
  sorry

end father_age_when_sum_100_l98_98524


namespace matching_pair_probability_l98_98973

def total_pairs : ℕ := 17

def black_pairs : ℕ := 8
def brown_pairs : ℕ := 4
def gray_pairs : ℕ := 3
def red_pairs : ℕ := 2

def total_shoes : ℕ := 2 * (black_pairs + brown_pairs + gray_pairs + red_pairs)

def prob_match (n_pairs : ℕ) (total_shoes : ℕ) :=
  (2 * n_pairs / total_shoes) * (n_pairs / (total_shoes - 1))

noncomputable def probability_of_matching_pair :=
  (prob_match black_pairs total_shoes) +
  (prob_match brown_pairs total_shoes) +
  (prob_match gray_pairs total_shoes) +
  (prob_match red_pairs total_shoes)

theorem matching_pair_probability :
  probability_of_matching_pair = 93 / 551 :=
sorry

end matching_pair_probability_l98_98973


namespace remove_denominators_l98_98860

theorem remove_denominators (x : ℝ) : (1 / 2 - (x - 1) / 3 = 1) → (3 - 2 * (x - 1) = 6) :=
by
  intro h
  sorry

end remove_denominators_l98_98860


namespace cone_radius_l98_98608

theorem cone_radius (r l : ℝ) 
  (surface_area_eq : π * r^2 + π * r * l = 12 * π)
  (net_is_semicircle : π * l = 2 * π * r) : 
  r = 2 :=
by
  sorry

end cone_radius_l98_98608


namespace inner_cube_surface_area_l98_98157

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (h_outer_cube : surface_area_outer_cube = 54) :
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 :=
by
  sorry

end inner_cube_surface_area_l98_98157


namespace cube_volume_l98_98116

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98116


namespace purchase_price_of_radio_l98_98263

theorem purchase_price_of_radio 
  (selling_price : ℚ) (loss_percentage : ℚ) (purchase_price : ℚ) 
  (h1 : selling_price = 465.50)
  (h2 : loss_percentage = 0.05):
  purchase_price = 490 :=
by 
  sorry

end purchase_price_of_radio_l98_98263


namespace apples_for_juice_l98_98975

def totalApples : ℝ := 6
def exportPercentage : ℝ := 0.25
def juicePercentage : ℝ := 0.60

theorem apples_for_juice : 
  let remainingApples := totalApples * (1 - exportPercentage)
  let applesForJuice := remainingApples * juicePercentage
  applesForJuice = 2.7 :=
by
  sorry

end apples_for_juice_l98_98975


namespace coin_die_sum_probability_l98_98893

theorem coin_die_sum_probability : 
  let coin_sides := [5, 15]
  let die_sides := [1, 2, 3, 4, 5, 6]
  let ben_age := 18
  (1 / 2 : ℚ) * (1 / 6 : ℚ) = 1 / 12 :=
by
  sorry

end coin_die_sum_probability_l98_98893


namespace equal_intercepts_line_l98_98432

theorem equal_intercepts_line (x y : ℝ)
  (h1 : x + 2*y - 6 = 0) 
  (h2 : x - 2*y + 2 = 0) 
  (hx : x = 2) 
  (hy : y = 2) :
  (y = x) ∨ (x + y = 4) :=
sorry

end equal_intercepts_line_l98_98432


namespace max_value_l98_98695

-- Define the weights and values of gemstones
def weight_sapphire : ℕ := 6
def value_sapphire : ℕ := 15
def weight_ruby : ℕ := 3
def value_ruby : ℕ := 9
def weight_diamond : ℕ := 2
def value_diamond : ℕ := 5

-- Define the weight capacity
def max_weight : ℕ := 24

-- Define the availability constraint
def min_availability : ℕ := 10

-- The goal is to prove that the maximum value is 72
theorem max_value : ∃ (num_sapphire num_ruby num_diamond : ℕ),
  num_sapphire >= min_availability ∧
  num_ruby >= min_availability ∧
  num_diamond >= min_availability ∧
  num_sapphire * weight_sapphire + num_ruby * weight_ruby + num_diamond * weight_diamond ≤ max_weight ∧
  num_sapphire * value_sapphire + num_ruby * value_ruby + num_diamond * value_diamond = 72 :=
by sorry

end max_value_l98_98695


namespace sum_of_three_smallest_positive_solutions_equals_ten_and_half_l98_98290

noncomputable def sum_three_smallest_solutions : ℚ :=
    let x1 : ℚ := 2.75
    let x2 : ℚ := 3 + (4 / 9)
    let x3 : ℚ := 4 + (5 / 16)
    x1 + x2 + x3

theorem sum_of_three_smallest_positive_solutions_equals_ten_and_half :
  sum_three_smallest_solutions = 10.5 :=
by
  sorry

end sum_of_three_smallest_positive_solutions_equals_ten_and_half_l98_98290


namespace inner_cube_surface_area_l98_98133

-- Definitions for problem conditions
def original_cube_surface_area : ℝ := 54
def sphere_inscribed_in_cube (cube_side : ℝ) : Prop := 
  cube_side^2 * 6 = original_cube_surface_area
def second_cube_inscribed_in_sphere (sphere_diameter inner_cube_side : ℝ) : Prop :=
  sphere_diameter = inner_cube_side * real.sqrt 3 * 2

-- Main Theorem to Prove
theorem inner_cube_surface_area (original_cube_side inner_cube_side : ℝ) 
  (h_cube : sphere_inscribed_in_cube original_cube_side)
  (h_inner_cube : second_cube_inscribed_in_sphere original_cube_side inner_cube_side) :
  6 * inner_cube_side^2 = 18 :=
by 
  sorry

end inner_cube_surface_area_l98_98133


namespace one_equation_does_not_pass_origin_l98_98612

def passes_through_origin (eq : ℝ → ℝ) : Prop := eq 0 = 0

def equation1 (x : ℝ) : ℝ := x^4 + 1
def equation2 (x : ℝ) : ℝ := x^4 + x
def equation3 (x : ℝ) : ℝ := x^4 + x^2
def equation4 (x : ℝ) : ℝ := x^4 + x^3

theorem one_equation_does_not_pass_origin :
  (¬ passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  ¬ passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  ¬ passes_through_origin equation3 ∧ 
  passes_through_origin equation4) ∨
  (passes_through_origin equation1 ∧ 
  passes_through_origin equation2 ∧ 
  passes_through_origin equation3 ∧ 
  ¬ passes_through_origin equation4) :=
sorry

end one_equation_does_not_pass_origin_l98_98612


namespace perpendicular_length_GH_from_centroid_l98_98664

theorem perpendicular_length_GH_from_centroid
  (A B C D E F G : ℝ)
  -- Conditions for distances from vertices to the line RS
  (hAD : AD = 12)
  (hBE : BE = 12)
  (hCF : CF = 18)
  -- Define the coordinates based on the vertical distances to line RS
  (yA : A = 12)
  (yB : B = 12)
  (yC : C = 18)
  -- Define the centroid G of triangle ABC based on the average of the y-coordinates
  (yG : G = (A + B + C) / 3)
  : G = 14 :=
by
  sorry

end perpendicular_length_GH_from_centroid_l98_98664


namespace probability_of_at_least_one_l98_98026

theorem probability_of_at_least_one (P_1 P_2 : ℝ) (h1 : 0 ≤ P_1 ∧ P_1 ≤ 1) (h2 : 0 ≤ P_2 ∧ P_2 ≤ 1) :
  1 - (1 - P_1) * (1 - P_2) = P_1 + P_2 - P_1 * P_2 :=
by
  sorry

end probability_of_at_least_one_l98_98026


namespace area_of_rectangle_ABCD_l98_98220

-- Definitions for the conditions
def small_square_area := 4
def total_small_squares := 2
def large_square_area := (2 * (2 : ℝ)) * (2 * (2 : ℝ))
def total_squares_area := total_small_squares * small_square_area + large_square_area

-- The main proof statement
theorem area_of_rectangle_ABCD : total_squares_area = 24 := 
by
  sorry

end area_of_rectangle_ABCD_l98_98220


namespace interns_survival_probability_l98_98480

-- Defining the problem
noncomputable def probability_of_survival (n : ℕ) (k : ℕ) : ℝ :=
  let num_permutations := nat.factorial n
  let favorable_permutations := finset.sum (finset.range (k+1)) 
    (λ m, nat.choose n m * nat.factorial (n - m) * nat.factorial (m - 1))
  in (favorable_permutations : ℝ) / num_permutations

-- Required proof statement:
theorem interns_survival_probability (n : ℕ) (k : ℕ) (h₁ : n = 44) (h₂ : k = 21) :
  probability_of_survival 44 21 > 0.3 :=
sorry

end interns_survival_probability_l98_98480


namespace initial_toys_count_l98_98894

-- Definitions for the conditions
def initial_toys (X : ℕ) : ℕ := X
def lost_toys (X : ℕ) : ℕ := X - 6
def found_toys (X : ℕ) : ℕ := (lost_toys X) + 9
def borrowed_toys (X : ℕ) : ℕ := (found_toys X) + 5
def traded_toys (X : ℕ) : ℕ := (borrowed_toys X) - 3

-- Statement to prove
theorem initial_toys_count (X : ℕ) : traded_toys X = 43 → X = 38 :=
by
  -- Proof to be filled in
  sorry

end initial_toys_count_l98_98894


namespace non_zero_real_value_l98_98991

theorem non_zero_real_value (y : ℝ) (hy : y ≠ 0) (h : (3 * y)^5 = (9 * y)^4) : y = 27 :=
sorry

end non_zero_real_value_l98_98991


namespace total_glass_area_l98_98951

theorem total_glass_area 
  (len₁ len₂ len₃ wid₁ wid₂ wid₃ : ℕ)
  (h₁ : len₁ = 30) (h₂ : wid₁ = 12)
  (h₃ : len₂ = 30) (h₄ : wid₂ = 12)
  (h₅ : len₃ = 20) (h₆ : wid₃ = 12) :
  (len₁ * wid₁ + len₂ * wid₂ + len₃ * wid₃) = 960 := 
by
  sorry

end total_glass_area_l98_98951


namespace initial_men_invited_l98_98227

theorem initial_men_invited (M W C : ℕ) (h1 : W = M / 2) (h2 : C + 10 = 30) (h3 : M + W + C = 80) (h4 : C = 20) : M = 40 :=
sorry

end initial_men_invited_l98_98227


namespace tom_made_money_correct_l98_98689

-- Define constants for flour, salt, promotion cost, ticket price, and tickets sold
def flour_needed : ℕ := 500
def flour_bag_size : ℕ := 50
def flour_bag_cost : ℕ := 20
def salt_needed : ℕ := 10
def salt_cost_per_pound : ℚ := 0.2
def promotion_cost : ℕ := 1000
def ticket_price : ℕ := 20
def tickets_sold : ℕ := 500

-- Compute how much money Tom made
def money_made : ℤ :=
  let flour_bags := flour_needed / flour_bag_size
  let total_flour_cost := flour_bags * flour_bag_cost
  let total_salt_cost := salt_needed * salt_cost_per_pound
  let total_cost := total_flour_cost + total_salt_cost + promotion_cost
  let total_revenue := tickets_sold * ticket_price
  total_revenue - total_cost

-- The theorem statement
theorem tom_made_money_correct :
  money_made = 8798 := by
  sorry

end tom_made_money_correct_l98_98689


namespace no_p_safe_numbers_l98_98903

/-- A number n is p-safe if it differs in absolute value by more than 2 from all multiples of p. -/
def p_safe (n p : ℕ) : Prop := ∀ k : ℤ, abs (n - k * p) > 2 

/-- The main theorem stating that there are no numbers that are simultaneously 5-safe, 
    7-safe, and 9-safe from 1 to 15000. -/
theorem no_p_safe_numbers (n : ℕ) (hp : 1 ≤ n ∧ n ≤ 15000) : 
  ¬ (p_safe n 5 ∧ p_safe n 7 ∧ p_safe n 9) :=
sorry

end no_p_safe_numbers_l98_98903


namespace rationalize_sqrt_fraction_l98_98805

theorem rationalize_sqrt_fraction :
  (Real.sqrt (5 / 12) = (Real.sqrt 15) / 6) :=
by
  sorry

end rationalize_sqrt_fraction_l98_98805


namespace value_of_3_W_4_l98_98939

def W (a b : ℤ) : ℤ := b + 5 * a - 3 * a ^ 2

theorem value_of_3_W_4 : W 3 4 = -8 :=
by
  sorry

end value_of_3_W_4_l98_98939


namespace john_newspaper_percentage_less_l98_98638

theorem john_newspaper_percentage_less
  (total_newspapers : ℕ)
  (selling_price : ℝ)
  (percentage_sold : ℝ)
  (profit : ℝ)
  (total_cost : ℝ)
  (cost_per_newspaper : ℝ)
  (percentage_less : ℝ)
  (h1 : total_newspapers = 500)
  (h2 : selling_price = 2)
  (h3 : percentage_sold = 0.80)
  (h4 : profit = 550)
  (h5 : total_cost = 800 - profit)
  (h6 : cost_per_newspaper = total_cost / total_newspapers)
  (h7 : percentage_less = ((selling_price - cost_per_newspaper) / selling_price) * 100) :
  percentage_less = 75 :=
by
  sorry

end john_newspaper_percentage_less_l98_98638


namespace units_digit_product_odd_integers_10_to_110_l98_98033

-- Define the set of odd integer numbers between 10 and 110
def oddNumbersInRange : List ℕ := List.filter (fun n => n % 2 = 1) (List.range' 10 101)

-- Define the set of relevant odd multiples of 5 within the range
def oddMultiplesOfFive : List ℕ := List.filter (fun n => n % 5 = 0) oddNumbersInRange

-- Prove that the product of all odd positive integers between 10 and 110 has units digit 5
theorem units_digit_product_odd_integers_10_to_110 :
  let product : ℕ := List.foldl (· * ·) 1 oddNumbersInRange
  product % 10 = 5 :=
by
  sorry

end units_digit_product_odd_integers_10_to_110_l98_98033


namespace amy_owes_thirty_l98_98531

variable (A D : ℝ)

theorem amy_owes_thirty
  (total_pledged remaining_owed sally_carl_owe derek_half_amys_owes : ℝ)
  (h1 : total_pledged = 285)
  (h2 : remaining_owed = 400 - total_pledged)
  (h3 : sally_carl_owe = 35 + 35)
  (h4 : derek_half_amys_owes = A / 2)
  (h5 : remaining_owed - sally_carl_owe = 45)
  (h6 : 45 = A + (A / 2)) :
  A = 30 :=
by
  -- Proof steps skipped
  sorry

end amy_owes_thirty_l98_98531


namespace good_goods_sufficient_condition_l98_98970

-- Conditions
def good_goods (G: Type) (g: G) : Prop := (g = "good")
def not_cheap (G: Type) (g: G) : Prop := ¬(g = "cheap")

-- Statement
theorem good_goods_sufficient_condition (G: Type) (g: G) : 
  (good_goods G g) → (not_cheap G g) :=
sorry

end good_goods_sufficient_condition_l98_98970


namespace original_price_of_coffee_l98_98403

variable (P : ℝ)

theorem original_price_of_coffee :
  (4 * P - 2 * (1.5 * P) = 2) → P = 2 :=
by
  sorry

end original_price_of_coffee_l98_98403


namespace extremal_values_d_l98_98485

theorem extremal_values_d (P : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ → Prop)
  (hC : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1)
  (hA : A = (-1, 0)) (hB : B = (1, 0)) (hP : ∃ (x y : ℝ), C (x, y)) :
  ∃ (max_d min_d : ℝ), max_d = 14 ∧ min_d = 10 :=
by
  -- Necessary assumptions
  have h₁ : ∀ (x y : ℝ), C (x, y) ↔ (x - 3)^2 + (y - 4)^2 = 1 := hC
  have h₂ : A = (-1, 0) := hA
  have h₃ : B = (1, 0) := hB
  have h₄ : ∃ (x y : ℝ), C (x, y) := hP
  sorry

end extremal_values_d_l98_98485


namespace algebraic_expression_l98_98503

-- Given conditions in the problem.
variables (x y : ℝ)

-- The statement to be proved: If 2x - 3y = 1, then 6y - 4x + 8 = 6.
theorem algebraic_expression (h : 2 * x - 3 * y = 1) : 6 * y - 4 * x + 8 = 6 :=
by 
  sorry

end algebraic_expression_l98_98503


namespace extremum_condition_l98_98018

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + x + 1

def has_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, 3 * a * x^2 + 1 = 0

theorem extremum_condition (a : ℝ) : has_extremum a ↔ a < 0 := 
  sorry

end extremum_condition_l98_98018


namespace distance_CD_l98_98550

-- Conditions
variable (width_small : ℝ) 
variable (length_small : ℝ := 2 * width_small) 
variable (perimeter_small : ℝ := 2 * (width_small + length_small))
variable (width_large : ℝ := 3 * width_small)
variable (length_large : ℝ := 2 * length_small)
variable (area_large : ℝ := width_large * length_large)

-- Condition assertions
axiom smaller_rectangle_perimeter : perimeter_small = 6
axiom larger_rectangle_area : area_large = 12

-- Calculating distance hypothesis
theorem distance_CD (CD_x CD_y : ℝ) (width_small length_small width_large length_large : ℝ) 
  (smaller_rectangle_perimeter : 2 * (width_small + length_small) = 6)
  (larger_rectangle_area : (3 * width_small) * (2 * length_small) = 12)
  (CD_x_def : CD_x = 2 * length_small)
  (CD_y_def : CD_y = 2 * width_large - width_small)
  : Real.sqrt ((CD_x) ^ 2 + (CD_y) ^ 2) = Real.sqrt 45 := 
sorry

end distance_CD_l98_98550


namespace minimum_value_expression_l98_98597

theorem minimum_value_expression (a b : ℝ) (hne : b ≠ 0) :
    ∃ a b, a^2 + b^2 + a / b + 1 / b^2 = sqrt 3 :=
sorry

end minimum_value_expression_l98_98597


namespace prime_sum_probability_l98_98236

noncomputable def probability_prime_sum_two_8_sided_dice : ℚ :=
  let outcomes := (fin 8) × (fin 8)
  let possible_sums := { (i : ℕ, j : ℕ) // i < 8 ∧ j < 8 // (i + 1) + (j + 1) }
  let prime_sums := { s : ℕ // s ∈ [2, 3, 5, 7, 11, 13] }
  (23 : ℚ) / (64 : ℚ)

theorem prime_sum_probability : probability_prime_sum_two_8_sided_dice = 23 / 64 := 
by sorry

end prime_sum_probability_l98_98236


namespace min_value_of_expression_l98_98935

theorem min_value_of_expression
  (x y : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (h_eq : x * (x + y) = 5 * x + y) : 2 * x + y ≥ 9 :=
sorry

end min_value_of_expression_l98_98935


namespace rationalize_sqrt_fraction_denom_l98_98800

theorem rationalize_sqrt_fraction_denom : sqrt (5 / 12) = sqrt (15) / 6 := by
  sorry

end rationalize_sqrt_fraction_denom_l98_98800


namespace max_triangle_perimeter_l98_98348

theorem max_triangle_perimeter :
  ∃ (a : ℕ), 4 < a → a < 7 ∧ (a * 5 > 20) ∧ (a + 20 > a * 4) ∧ (a * 4 + 20 > a) ∧
  (if a = 5 then 5 + 20 + 20 = 45 else if a = 6 then 6 + 24 + 20 = 50 else false) :=
by
  -- Define the conditions
  let a := a in
  -- Given a ∈ ℕ, 4 < a < 7, and triangle inequalities hold
  assume (a : ℕ) (h1 : 4 < a) (h2 : a < 7) (h3 : 5 * a > 20) (h4 : a + 20 > 4 * a) (h5 : 4 * a + 20 > a),
  -- Check the perimeters for a = 5 and a = 6
  existsi a,
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  split,
  { assumption },
  {
    cases a,
    -- Case when a = 0
    { simp },
    cases a,
    -- Case when a = 1
    { simp },
    cases a,
    -- Case when a = 2
    { simp },
    cases a,
    -- Case when a = 3
    { simp },
    cases a,
    -- Case when a = 4
    { simp },
    cases a,
    { 
      -- Case when a = 5
      left,
      linarith,
    },
    cases a,
    {
      -- Case when a = 6
      right,
      left,
      linarith,
    },
    -- Remaining cases where a >= 7 are invalid due to earlier constraints
    simp,
    intros,
  },
  sorry

end max_triangle_perimeter_l98_98348


namespace solution_set_inequality_l98_98285

theorem solution_set_inequality :
  {x : ℝ | (x^2 + 4) / (x - 4)^2 ≥ 0} = {x | x < 4} ∪ {x | x > 4} :=
by
  sorry

end solution_set_inequality_l98_98285


namespace scientific_notation_l98_98237

theorem scientific_notation : (20160 : ℝ) = 2.016 * 10^4 := 
  sorry

end scientific_notation_l98_98237


namespace seq_value_is_minus_30_l98_98742

open Nat  -- Open the natural numbers namespace

noncomputable def seq_condition (a : ℕ → ℤ) :=
  ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q

theorem seq_value_is_minus_30 (a : ℕ → ℤ) (h_seq : seq_condition a) (h_a2 : a 2 = -6) :
  a 10 = -30 :=
by 
  sorry

end seq_value_is_minus_30_l98_98742


namespace x0_y0_sum_eq_31_l98_98699

theorem x0_y0_sum_eq_31 :
  ∃ x0 y0 : ℕ, (0 ≤ x0 ∧ x0 < 37) ∧ (0 ≤ y0 ∧ y0 < 37) ∧ 
  (2 * x0 ≡ 1 [MOD 37]) ∧ (3 * y0 ≡ 36 [MOD 37]) ∧ 
  (x0 + y0 = 31) :=
sorry

end x0_y0_sum_eq_31_l98_98699


namespace calculate_coeffs_l98_98580

noncomputable def quadratic_coeffs (p q : ℝ) : Prop :=
  if p = 1 then true else if p = -2 then q = -1 else false

theorem calculate_coeffs (p q : ℝ) :
    (∃ p q, (x^2 + p * x + q = 0) ∧ (x^2 - p^2 * x + p * q = 0)) →
    quadratic_coeffs p q :=
by sorry

end calculate_coeffs_l98_98580


namespace inner_cube_surface_area_l98_98123

theorem inner_cube_surface_area (surface_area_outer_cube : ℝ) (inscribed_sphere : ∃ radius : ℝ, radius = 3 / √3) 
  (surface_area_outer_cube = 54) : 
  ∃ surface_area_inner_cube : ℝ, surface_area_inner_cube = 18 := 
by
  sorry

end inner_cube_surface_area_l98_98123


namespace rationalize_denominator_l98_98810

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l98_98810


namespace falling_body_time_l98_98611

theorem falling_body_time (g : ℝ) (h_g : g = 9.808) (d : ℝ) (t1 : ℝ) (h_d : d = 49.34) (h_t1 : t1 = 1.3) : 
  ∃ t : ℝ, (1 / 2 * g * (t + t1)^2 - 1 / 2 * g * t^2 = d) → t = 7.088 :=
by 
  use 7.088
  intros h
  sorry

end falling_body_time_l98_98611


namespace find_y_of_arithmetic_mean_l98_98302

theorem find_y_of_arithmetic_mean (y : ℝ) (h : (8 + 16 + 12 + 24 + 7 + y) / 6 = 12) : y = 5 :=
by
  sorry

end find_y_of_arithmetic_mean_l98_98302


namespace divisible_by_3_l98_98529

theorem divisible_by_3 (x y : ℤ) (h : (x^2 + y^2) % 3 = 0) : x % 3 = 0 ∧ y % 3 = 0 :=
sorry

end divisible_by_3_l98_98529


namespace greatest_perimeter_l98_98355

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l98_98355


namespace find_m_and_n_l98_98628

theorem find_m_and_n (m n : ℝ) 
  (h1 : m + n = 6) 
  (h2 : 2 * m - n = 6) : 
  m = 4 ∧ n = 2 := 
by 
  sorry

end find_m_and_n_l98_98628


namespace find_x10_l98_98644

theorem find_x10 (x : ℕ → ℝ) :
  x 1 = 1 ∧ x 2 = 1 ∧ (∀ n ≥ 2, x (n + 1) = (x n * x (n - 1)) / (x n + x (n - 1))) →
  x 10 = 1 / 55 :=
by sorry

end find_x10_l98_98644


namespace cube_volume_l98_98079

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98079


namespace desired_annual_profit_is_30500000_l98_98881

noncomputable def annual_fixed_costs : ℝ := 50200000
noncomputable def average_cost_per_car : ℝ := 5000
noncomputable def number_of_cars : ℕ := 20000
noncomputable def selling_price_per_car : ℝ := 9035

noncomputable def total_revenue : ℝ :=
  selling_price_per_car * number_of_cars

noncomputable def total_variable_costs : ℝ :=
  average_cost_per_car * number_of_cars

noncomputable def total_costs : ℝ :=
  annual_fixed_costs + total_variable_costs

noncomputable def desired_annual_profit : ℝ :=
  total_revenue - total_costs

theorem desired_annual_profit_is_30500000:
  desired_annual_profit = 30500000 := by
  sorry

end desired_annual_profit_is_30500000_l98_98881


namespace sum_of_cubes_of_nonneg_rationals_l98_98641

theorem sum_of_cubes_of_nonneg_rationals (n : ℤ) (h1 : n > 1) (h2 : ∃ a b : ℚ, a^3 + b^3 = n) :
  ∃ c d : ℚ, c ≥ 0 ∧ d ≥ 0 ∧ c^3 + d^3 = n :=
sorry

end sum_of_cubes_of_nonneg_rationals_l98_98641


namespace zoe_pictures_l98_98451

theorem zoe_pictures (P : ℕ) (h1 : P + 16 = 44) : P = 28 :=
by sorry

end zoe_pictures_l98_98451


namespace rationalize_sqrt_5_over_12_l98_98825

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l98_98825


namespace rationalize_denominator_l98_98797

theorem rationalize_denominator :
  (3 : ℝ) / Real.sqrt 48 = Real.sqrt 3 / 4 :=
by
  sorry

end rationalize_denominator_l98_98797


namespace linear_independent_vectors_p_value_l98_98472

theorem linear_independent_vectors_p_value (p : ℝ) :
  (∃ (a b : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ a * (2 : ℝ) + b * (5 : ℝ) = 0 ∧ a * (4 : ℝ) + b * p = 0) ↔ p = 10 :=
by
  sorry

end linear_independent_vectors_p_value_l98_98472


namespace inner_cube_surface_area_l98_98143

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l98_98143


namespace isabel_homework_problems_l98_98399

theorem isabel_homework_problems (initial_problems finished_problems remaining_pages problems_per_page : ℕ) 
  (h1 : initial_problems = 72)
  (h2 : finished_problems = 32)
  (h3 : remaining_pages = 5)
  (h4 : initial_problems - finished_problems = 40)
  (h5 : 40 = remaining_pages * problems_per_page) : 
  problems_per_page = 8 := 
by sorry

end isabel_homework_problems_l98_98399


namespace apple_tree_yield_l98_98788

theorem apple_tree_yield (A : ℝ) 
    (h1 : Magdalena_picks_day1 = A / 5)
    (h2 : Magdalena_picks_day2 = 2 * (A / 5))
    (h3 : Magdalena_picks_day3 = (A / 5) + 20)
    (h4 : remaining_apples = 20)
    (total_picked : Magdalena_picks_day1 + Magdalena_picks_day2 + Magdalena_picks_day3 + remaining_apples = A)
    : A = 200 :=
by
    sorry

end apple_tree_yield_l98_98788


namespace imaginary_part_of_z_l98_98501

open Complex -- open complex number functions

theorem imaginary_part_of_z (z : ℂ) (h : (z + 1) * (2 - I) = 5 * I) :
  z.im = 2 :=
sorry

end imaginary_part_of_z_l98_98501


namespace number_of_zeros_of_f_l98_98544

noncomputable def f (x : ℝ) := (1 / 3) * x ^ 3 - x ^ 2 - 3 * x + 9

theorem number_of_zeros_of_f : ∃ (z : ℕ), z = 2 ∧ ∀ x : ℝ, (f x = 0 → x = -3 ∨ x = -2 / 3 ∨ x = 1 ∨ x = 3) := 
sorry

end number_of_zeros_of_f_l98_98544


namespace roof_length_width_difference_l98_98548

theorem roof_length_width_difference
  {w l : ℝ} 
  (h_area : l * w = 576) 
  (h_length : l = 4 * w) 
  (hw_pos : w > 0) :
  l - w = 36 :=
by 
  sorry

end roof_length_width_difference_l98_98548


namespace cube_volume_l98_98094

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98094


namespace probability_of_matching_pair_l98_98184

def total_socks : ℕ := 12 + 6 + 9
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

def black_pairs : ℕ := choose_two 12
def white_pairs : ℕ := choose_two 6
def blue_pairs : ℕ := choose_two 9

def total_pairs : ℕ := choose_two total_socks
def matching_pairs : ℕ := black_pairs + white_pairs + blue_pairs

def probability : ℚ := matching_pairs / total_pairs

theorem probability_of_matching_pair :
  probability = 1 / 3 :=
by
  -- The proof will go here
  sorry

end probability_of_matching_pair_l98_98184


namespace limit_proof_l98_98869

noncomputable def limit_function (x : ℝ) : ℝ :=
  (∛(x / 16) - 1 / 4) / (sqrt (1 / 4 + x) - sqrt (2 * x))

theorem limit_proof :
  tendsto limit_function (𝓝 (1 / 4)) (𝓝 (-2 * sqrt 2 / 6)) :=
sorry

end limit_proof_l98_98869


namespace find_larger_number_l98_98539

theorem find_larger_number (x y : ℝ) (h1 : x - y = 1860) (h2 : 0.075 * x = 0.125 * y) :
  x = 4650 :=
by
  sorry

end find_larger_number_l98_98539


namespace find_AC_l98_98270

theorem find_AC (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (max_val : A - C = 3) (min_val : -A - C = -1) : 
  A = 2 ∧ C = 1 :=
by
  sorry

end find_AC_l98_98270


namespace inner_cube_surface_area_l98_98169

theorem inner_cube_surface_area (S : ℝ) 
    (h1 : ∃ s, s^2 = 9 ∧ 6 * s^2 = S := by { use 3, split; norm_num }) :
  ∃ innerS, innerS = 18 :=
begin
  -- Assume the side length of the inner cube
  let l := sqrt 3,
  -- Calculate the surface area of the inner cube
  let innerS := 6 * l^2,
  -- Show that the calculated surface area is 18 square meters
  use innerS,
  norm_num,
  rw [innerS, mul_assoc, ←pow_two, pow_succ, pow_one],
  norm_num,
end

end inner_cube_surface_area_l98_98169


namespace share_of_A_l98_98967

-- Definitions corresponding to the conditions
variables (A B C : ℝ)
variable (total : ℝ := 578)
variable (share_ratio_B_C : ℝ := 1 / 4)
variable (share_ratio_A_B : ℝ := 2 / 3)

-- Conditions
def condition1 : B = share_ratio_B_C * C := by sorry
def condition2 : A = share_ratio_A_B * B := by sorry
def condition3 : A + B + C = total := by sorry

-- The equivalent math proof problem statement
theorem share_of_A :
  A = 68 :=
by sorry

end share_of_A_l98_98967


namespace rationalize_denominator_l98_98816

theorem rationalize_denominator :
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98816


namespace toy_value_l98_98887

theorem toy_value (n : ℕ) (total_value special_toy_value : ℕ)
  (h₀ : n = 9) (h₁ : total_value = 52) (h₂ : special_toy_value = 12) :
  (total_value - special_toy_value) / (n - 1) = 5 :=
by
  have m : ℕ := n - 1
  have other_toys_value : ℕ := total_value - special_toy_value
  show other_toys_value / m = 5
  sorry

end toy_value_l98_98887


namespace duration_of_investment_l98_98707

-- Define the constants as given in the conditions
def Principal : ℝ := 7200
def Rate : ℝ := 17.5
def SimpleInterest : ℝ := 3150

-- Define the time variable we want to prove
def Time : ℝ := 2.5

-- Prove that the calculated time matches the expected value
theorem duration_of_investment :
  SimpleInterest = (Principal * Rate * Time) / 100 :=
sorry

end duration_of_investment_l98_98707


namespace min_value_expression_l98_98747

theorem min_value_expression (a d b c : ℝ) (habd : a ≥ 0 ∧ d ≥ 0) (hbc : b > 0 ∧ c > 0) (h_cond : b + c ≥ a + d) :
  (b / (c + d) + c / (a + b)) ≥ (Real.sqrt 2 - 1 / 2) :=
sorry

end min_value_expression_l98_98747


namespace pieces_by_first_team_correct_l98_98737

-- Define the number of pieces required.
def total_pieces : ℕ := 500

-- Define the number of pieces made by the second team.
def pieces_by_second_team : ℕ := 131

-- Define the number of pieces made by the third team.
def pieces_by_third_team : ℕ := 180

-- Define the number of pieces made by the first team.
def pieces_by_first_team : ℕ := total_pieces - (pieces_by_second_team + pieces_by_third_team)

-- Statement to prove
theorem pieces_by_first_team_correct : pieces_by_first_team = 189 := 
by 
  -- Proof to be filled in
  sorry

end pieces_by_first_team_correct_l98_98737


namespace volume_of_cube_l98_98100

-- Definition of the surface area condition
def surface_area_condition (s : ℝ) : Prop :=
  6 * s^2 = 150

-- The main theorem to prove
theorem volume_of_cube (s : ℝ) (h : surface_area_condition s) : s^3 = 125 :=
by
  sorry

end volume_of_cube_l98_98100


namespace part_one_part_two_l98_98905

-- Part (1)
theorem part_one (a : ℝ) (h : a ≤ 2) (x : ℝ) :
  (|x - 1| + |x - a| ≥ 2 ↔ x ≤ 0.5 ∨ x ≥ 2.5) :=
sorry

-- Part (2)
theorem part_two (a : ℝ) (h1 : a > 1) (h2 : ∀ x : ℝ, |x - 1| + |x - a| + |x - 1| ≥ 1) :
  a ≥ 2 :=
sorry

end part_one_part_two_l98_98905


namespace prob_teamB_wins_first_game_l98_98974
-- Import the necessary library

-- Define the conditions and the question in a Lean theorem statement
theorem prob_teamB_wins_first_game :
  (∀ (win_A win_B : ℕ), win_A < 4 ∧ win_B = 4) →
  (∀ (team_wins_game : ℕ → Prop), (team_wins_game 2 = false) ∧ (team_wins_game 3 = true)) →
  (∀ (team_wins_series : Prop), team_wins_series = (win_B ≥ 4 ∧ win_A < 4)) →
  (∀ (game_outcome_distribution : ℕ → ℕ → ℕ → ℕ → ℚ), game_outcome_distribution 4 4 2 2 = 1 / 2) →
  (∀ (first_game_outcome : Prop), first_game_outcome = true) →
  true :=
sorry

end prob_teamB_wins_first_game_l98_98974


namespace probability_of_staying_in_dark_l98_98042

theorem probability_of_staying_in_dark (revolutions_per_minute : ℕ) (time_in_seconds : ℕ) (dark_time : ℕ) :
  revolutions_per_minute = 2 →
  time_in_seconds = 60 →
  dark_time = 5 →
  (5 / 6 : ℝ) = 5 / 6 :=
by
  intros
  sorry

end probability_of_staying_in_dark_l98_98042


namespace garden_remaining_area_is_250_l98_98259

open Nat

-- Define the dimensions of the rectangular garden
def garden_length : ℕ := 18
def garden_width : ℕ := 15
-- Define the dimensions of the square cutouts
def cutout1_side : ℕ := 4
def cutout2_side : ℕ := 2

-- Calculate areas based on the definitions
def garden_area : ℕ := garden_length * garden_width
def cutout1_area : ℕ := cutout1_side * cutout1_side
def cutout2_area : ℕ := cutout2_side * cutout2_side

-- Calculate total area excluding the cutouts
def remaining_area : ℕ := garden_area - cutout1_area - cutout2_area

-- Prove that the remaining area is 250 square feet
theorem garden_remaining_area_is_250 : remaining_area = 250 :=
by
  sorry

end garden_remaining_area_is_250_l98_98259


namespace intersection_expression_value_l98_98433

theorem intersection_expression_value
  (x₁ y₁ x₂ y₂ : ℝ)
  (h₁ : x₁ * y₁ = 1)
  (h₂ : x₂ * y₂ = 1)
  (h₃ : x₁ = -x₂)
  (h₄ : y₁ = -y₂) :
  x₁ * y₂ + x₂ * y₁ = -2 :=
by
  sorry

end intersection_expression_value_l98_98433


namespace find_a_l98_98305

theorem find_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x = 0 ∧ x = 1) → a = -1 := by
  intro h
  obtain ⟨x, hx, rfl⟩ := h
  have H : 1^2 + a * 1 = 0 := hx
  linarith

end find_a_l98_98305


namespace cally_pants_count_l98_98581

variable (cally_white_shirts : ℕ)
variable (cally_colored_shirts : ℕ)
variable (cally_shorts : ℕ)
variable (danny_white_shirts : ℕ)
variable (danny_colored_shirts : ℕ)
variable (danny_shorts : ℕ)
variable (danny_pants : ℕ)
variable (total_clothes_washed : ℕ)
variable (cally_pants : ℕ)

-- Given conditions
#check cally_white_shirts = 10
#check cally_colored_shirts = 5
#check cally_shorts = 7
#check danny_white_shirts = 6
#check danny_colored_shirts = 8
#check danny_shorts = 10
#check danny_pants = 6
#check total_clothes_washed = 58
#check cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed

-- Proof goal
theorem cally_pants_count (cally_white_shirts cally_colored_shirts cally_shorts danny_white_shirts danny_colored_shirts danny_shorts danny_pants cally_pants total_clothes_washed : ℕ) :
  cally_white_shirts = 10 →
  cally_colored_shirts = 5 →
  cally_shorts = 7 →
  danny_white_shirts = 6 →
  danny_colored_shirts = 8 →
  danny_shorts = 10 →
  danny_pants = 6 →
  total_clothes_washed = 58 →
  (cally_white_shirts + cally_colored_shirts + cally_shorts + danny_white_shirts + danny_colored_shirts + danny_shorts + cally_pants + danny_pants = total_clothes_washed) →
  cally_pants = 6 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end cally_pants_count_l98_98581


namespace max_perimeter_l98_98386

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l98_98386


namespace cube_volume_l98_98090

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 := by
  have side_area_eq : 25 = 150 / 6 := by linarith
  have edge_length_eq : 5 = Real.sqrt 25 := by rw [Real.sqrt_eq, mul_self_eq]; norm_num
  have volume_eq : 125 = 5 ^ 3 := by norm_num
  use 125
  sorry

end cube_volume_l98_98090


namespace train_average_speed_l98_98462

theorem train_average_speed :
  let start_time := 9.0 -- Start time in hours (9:00 am)
  let end_time := 13.75 -- End time in hours (1:45 pm)
  let total_distance := 348.0 -- Total distance in km
  let halt_time := 0.75 -- Halt time in hours (45 minutes)
  let scheduled_time := end_time - start_time -- Total scheduled time in hours
  let actual_travel_time := scheduled_time - halt_time -- Actual travel time in hours
  let average_speed := total_distance / actual_travel_time -- Average speed formula
  average_speed = 87.0 := sorry

end train_average_speed_l98_98462


namespace integral_solution_unique_l98_98186

theorem integral_solution_unique (a b c : ℤ) : a^2 + b^2 + c^2 = a^2 * b^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by
  sorry

end integral_solution_unique_l98_98186


namespace cube_volume_of_surface_area_l98_98065

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98065


namespace relative_frequency_defective_books_l98_98681

theorem relative_frequency_defective_books 
  (N_defective : ℤ) (N_total : ℤ)
  (h_defective : N_defective = 5)
  (h_total : N_total = 100) :
  (N_defective : ℚ) / N_total = 0.05 := by
  sorry

end relative_frequency_defective_books_l98_98681


namespace quadratic_no_real_roots_l98_98621

theorem quadratic_no_real_roots (m : ℝ) (h : ∀ x : ℝ, x^2 - m * x + 1 ≠ 0) : m = 0 :=
by
  sorry

end quadratic_no_real_roots_l98_98621


namespace total_tiles_l98_98171

/-- A square-shaped floor is covered with congruent square tiles. 
If the total number of tiles on the two diagonals is 88 and the floor 
forms a perfect square with an even side length, then the number of tiles 
covering the floor is 1936. -/
theorem total_tiles (n : ℕ) (hn_even : n % 2 = 0) (h_diag : 2 * n = 88) : n^2 = 1936 := 
by 
  sorry

end total_tiles_l98_98171


namespace restaurant_meal_cost_l98_98717

def cost_of_group_meal (total_people : Nat) (kids : Nat) (adult_meal_cost : Nat) : Nat :=
  let adults := total_people - kids
  adults * adult_meal_cost

theorem restaurant_meal_cost :
  cost_of_group_meal 9 2 2 = 14 := by
  sorry

end restaurant_meal_cost_l98_98717


namespace problem_solution_l98_98449

theorem problem_solution : (3106 - 2935 + 17)^2 / 121 = 292 := by
  sorry

end problem_solution_l98_98449


namespace rationalize_denominator_l98_98809

theorem rationalize_denominator :
  Real.sqrt (5 / 12) = Real.sqrt 15 / 6 := 
by
  sorry

end rationalize_denominator_l98_98809


namespace floor_cube_neg_seven_four_l98_98282

theorem floor_cube_neg_seven_four :
  (Int.floor ((-7 / 4 : ℚ) ^ 3) = -6) :=
by
  sorry

end floor_cube_neg_seven_four_l98_98282


namespace probability_problems_l98_98513

theorem probability_problems (x : ℕ) :
  (0 = (if 8 + 12 > 8 then 0 else 1)) ∧
  (1 = (1 - 0)) ∧
  (3 / 5 = 12 / 20) ∧
  (4 / 5 = (8 + x) / 20 → x = 8) := by sorry

end probability_problems_l98_98513


namespace simplify_fraction_l98_98665

theorem simplify_fraction :
  (1 / (3 / (Real.sqrt 5 + 2) + 4 / (Real.sqrt 7 - 2))) = (3 / (9 * Real.sqrt 5 + 4 * Real.sqrt 7 - 10)) :=
sorry

end simplify_fraction_l98_98665


namespace cube_volume_l98_98083

theorem cube_volume (s : ℝ) (h : 6 * s^2 = 150) : s^3 = 125 := by
  sorry

end cube_volume_l98_98083


namespace inequality_proof_l98_98479

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_geq : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
sorry

end inequality_proof_l98_98479


namespace least_value_of_x_l98_98941

theorem least_value_of_x 
  (x : ℕ) (p : ℕ) 
  (h1 : x > 0) 
  (h2 : Prime p) 
  (h3 : ∃ q, Prime q ∧ q % 2 = 1 ∧ x = 9 * p * q) : 
  x = 90 := 
sorry

end least_value_of_x_l98_98941


namespace overall_average_of_marks_l98_98845

theorem overall_average_of_marks (n total_boys passed_boys failed_boys avg_passed avg_failed : ℕ) 
  (h1 : total_boys = 120)
  (h2 : passed_boys = 105)
  (h3 : failed_boys = 15)
  (h4 : total_boys = passed_boys + failed_boys)
  (h5 : avg_passed = 39)
  (h6 : avg_failed = 15) :
  ((passed_boys * avg_passed + failed_boys * avg_failed) / total_boys = 36) :=
by
  sorry

end overall_average_of_marks_l98_98845


namespace inner_cube_surface_area_l98_98162

/-- Given a cube with surface area 54 square meters that contains an inscribed sphere,
and a second cube is inscribed within that sphere, prove that the surface area
of the inscribed inner cube is 18 square meters. -/
theorem inner_cube_surface_area (surface_area : ℝ) (h_sa : surface_area = 54) :
  ∃ inner_surface_area, inner_surface_area = 18 :=
by
  let side_length := real.sqrt (surface_area / 6)
  have h_side_length : side_length = 3 := 
    by sorry -- Calculation showing side_length derived from the given surface_area
  
  let sphere_diameter := side_length
  have h_sphere_diameter : sphere_diameter = 3 := by sorry -- Diameter is the same as side length
  
  let inner_cube_side := real.sqrt (sphere_diameter^2 / 3)
  have h_inner_cube_side : inner_cube_side = real.sqrt 3 :=
    by sorry -- Calculating the side length of the inner cube
  
  let inner_surface_area := 6 * (inner_cube_side ^ 2)
  have h_inner_surface_area : inner_surface_area = 18 :=
    by sorry -- Calculating the surface area of the inner cube
  
  use inner_surface_area
  exact h_inner_surface_area

end inner_cube_surface_area_l98_98162


namespace time_for_A_to_finish_race_l98_98211

-- Definitions based on the conditions
def race_distance : ℝ := 120
def B_time : ℝ := 45
def B_beaten_distance : ℝ := 24

-- Proof statement: We need to show that A's time is 56.25 seconds
theorem time_for_A_to_finish_race : ∃ (t : ℝ), t = 56.25 ∧ (120 / t = 96 / 45)
  := sorry

end time_for_A_to_finish_race_l98_98211


namespace pb_pc_lt_ad_l98_98775

-- Needed definitions for points, distances, and angles
variables {Point : Type*} [coordinate_space Point]

def distance (a b : Point) : ℝ := sorry
def angle (a b c : Point) : real.angle := sorry

-- Convex quadrilateral properties
variables (A B C D P : Point)
variables (h_convex : convex_quadrilateral A B C D)
variables (h_AB_eq_CD : distance A B = distance C D)
variables (h_angles_sum : angle P B A + angle P C D = 180)

-- Goal: Prove that PB + PC < AD
theorem pb_pc_lt_ad (h_convex : convex_quadrilateral A B C D) 
                     (h_AB_eq_CD : distance A B = distance C D)
                     (h_angles_sum : angle P B A + angle P C D = 180) :
  distance P B + distance P C < distance A D := 
sorry

end pb_pc_lt_ad_l98_98775


namespace soccer_ball_purchase_l98_98878

theorem soccer_ball_purchase (wholesale_price retail_price profit remaining_balls final_profit : ℕ)
  (h1 : wholesale_price = 30)
  (h2 : retail_price = 45)
  (h3 : profit = retail_price - wholesale_price)
  (h4 : remaining_balls = 30)
  (h5 : final_profit = 1500) :
  ∃ (initial_balls : ℕ), (initial_balls - remaining_balls) * profit = final_profit ∧ initial_balls = 130 :=
by
  sorry

end soccer_ball_purchase_l98_98878


namespace cube_volume_l98_98115

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98115


namespace constant_term_binomial_expansion_l98_98015

theorem constant_term_binomial_expansion :
  (∃ r : ℕ, 12 - 3 * r = 0 ∧ (¬r >= 7) ∧ (x : ℚ => (x^2 - 2/x) ^ 6) = (240 : ℕ) :=
begin
  sorry
end

end constant_term_binomial_expansion_l98_98015


namespace rationalize_denominator_l98_98821

theorem rationalize_denominator : Real.sqrt (5 / 12) = Real.sqrt 15 / 6 :=
by
  sorry

end rationalize_denominator_l98_98821


namespace least_integer_gt_square_l98_98434

theorem least_integer_gt_square (x : ℝ) (y : ℝ) (h1 : x = 2) (h2 : y = Real.sqrt 3) :
  ∃ (n : ℤ), n = 14 ∧ n > (x + y) ^ 2 := by
  sorry

end least_integer_gt_square_l98_98434


namespace max_perimeter_l98_98390

-- Define the three sides of the triangle
def valid_triangle (x : ℕ) : Prop :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  -- Triangle inequalities
  a + b > c ∧ a + c > b ∧ b + c > a

-- Define the perimeter calculation
def perimeter (x : ℕ) : ℕ :=
  let a : ℕ := x
  let b : ℕ := 4 * x
  let c : ℕ := 20 in
  a + b + c

-- Prove the maximum perimeter given the conditions
theorem max_perimeter : ∃ x : ℕ, 4 < x ∧ x < 7 ∧ valid_triangle x ∧ perimeter x = 50 :=
by
  use 6
  simp [valid_triangle, perimeter]
  sorry

end max_perimeter_l98_98390


namespace inner_cube_surface_area_l98_98135

theorem inner_cube_surface_area (S_outer : ℝ) (h_outer : S_outer = 54) : 
  ∃ S_inner : ℝ, S_inner = 27 := by
  -- The proof will go here
  sorry

end inner_cube_surface_area_l98_98135


namespace smallest_next_divisor_l98_98552

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor (m : ℕ) (h_even : is_even m)
  (h_four_digit : is_four_digit m)
  (h_div_437 : is_divisor 437 m) :
  ∃ next_div : ℕ, next_div > 437 ∧ is_divisor next_div m ∧ 
  ∀ d, d > 437 ∧ is_divisor d m → next_div ≤ d :=
sorry

end smallest_next_divisor_l98_98552


namespace sum_of_coordinates_of_D_is_12_l98_98793

theorem sum_of_coordinates_of_D_is_12 :
  (exists (x y : ℝ), (5 = (11 + x) / 2) ∧ (9 = (5 + y) / 2) ∧ (x + y = 12)) :=
by
  sorry

end sum_of_coordinates_of_D_is_12_l98_98793


namespace min_shift_odd_func_l98_98491

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem min_shift_odd_func (hφ : ∀ x : ℝ, f (x) = -f (-x + 2 * φ + (Real.pi / 3))) (hφ_positive : φ > 0) :
  φ = Real.pi / 6 :=
sorry

end min_shift_odd_func_l98_98491


namespace inner_cube_surface_area_l98_98145

theorem inner_cube_surface_area (A B : Type) [MetricSpace A] [MetricSpace B] (cube : B) (surface_area_cube : ℝ) (surface_area_cube = 54) 
(inner_cube_inscribed : B → A) : 
surface_area (inner_cube_inscribed cube) = 18 :=
by sorry

end inner_cube_surface_area_l98_98145


namespace proof_problem_l98_98298

theorem proof_problem
  (x y a b c d : ℝ)
  (h1 : |x - 1| + (y + 2)^2 = 0)
  (h2 : a * b = 1)
  (h3 : c + d = 0) :
  (x + y)^3 - (-a * b)^2 + 3 * c + 3 * d = -2 :=
by
  -- The proof steps go here.
  sorry

end proof_problem_l98_98298


namespace cube_volume_of_surface_area_l98_98067

theorem cube_volume_of_surface_area (s : ℝ) (V : ℝ) 
  (h₁ : 6 * s^2 = 150) :
  V = s^3 → V = 125 := by
  -- proof part, to be filled in
  sorry

end cube_volume_of_surface_area_l98_98067


namespace rationalize_sqrt_fraction_l98_98830

theorem rationalize_sqrt_fraction : 
  (sqrt (5 / 12) = sqrt 5 / sqrt 12) → 
  (sqrt 12 = 2 * sqrt 3) → 
  sqrt (5 / 12) = sqrt 15 / 6 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end rationalize_sqrt_fraction_l98_98830


namespace smallest_positive_multiple_of_6_and_5_l98_98031

theorem smallest_positive_multiple_of_6_and_5 : ∃ (n : ℕ), (n > 0) ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 6 = 0) ∧ (m % 5 = 0) → n ≤ m) :=
  sorry

end smallest_positive_multiple_of_6_and_5_l98_98031


namespace cube_volume_from_surface_area_l98_98058

-- Define the condition: a cube has a surface area of 150 square centimeters
def surface_area (s : ℝ) : ℝ := 6 * s^2

-- Define the volume of the cube
def volume (s : ℝ) : ℝ := s^3

-- Define the main theorem to prove the volume given the surface area condition
theorem cube_volume_from_surface_area (s : ℝ) (h : surface_area s = 150) : volume s = 125 :=
by
  sorry

end cube_volume_from_surface_area_l98_98058


namespace solve_m_n_l98_98629

theorem solve_m_n (m n : ℤ) :
  (m * 1 + n * 1 = 6) ∧ (m * 2 + n * -1 = 6) → (m = 4) ∧ (n = 2) := by
  sorry

end solve_m_n_l98_98629


namespace unique_reconstruction_l98_98986

theorem unique_reconstruction (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (a b c d : ℝ) (Ha : x + y = a) (Hb : x - y = b) (Hc : x * y = c) (Hd : x / y = d) :
  ∃! (x' y' : ℝ), x' > 0 ∧ y' > 0 ∧ x' + y' = a ∧ x' - y' = b ∧ x' * y' = c ∧ x' / y' = d := 
sorry

end unique_reconstruction_l98_98986


namespace odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l98_98490

noncomputable def f (x : ℝ) (k : ℝ) := 2^x + k * 2^(-x)

-- Prove that if f(x) is an odd function, then k = -1.
theorem odd_function_k_eq_neg_one {k : ℝ} (h : ∀ x, f x k = -f (-x) k) : k = -1 :=
by sorry

-- Prove that if for all x in [0, +∞), f(x) > 2^(-x), then k > 0.
theorem f_x_greater_2_neg_x_k_gt_zero {k : ℝ} (h : ∀ x, 0 ≤ x → f x k > 2^(-x)) : k > 0 :=
by sorry

end odd_function_k_eq_neg_one_f_x_greater_2_neg_x_k_gt_zero_l98_98490


namespace remainder_of_nonempty_disjoint_subsets_l98_98959

theorem remainder_of_nonempty_disjoint_subsets (T : Set ℕ) (hT : T = {1, 2, 3, ..., 12}) :
  let m := (3 ^ 12 - 2 * 2 ^ 12 + 1) / 2 in
  m % 1000 = 125 := 
by
  sorry

end remainder_of_nonempty_disjoint_subsets_l98_98959


namespace sugar_cubes_left_l98_98789

theorem sugar_cubes_left (h w d : ℕ) (hd1 : w * d = 77) (hd2 : h * d = 55) :
  (h - 1) * w * (d - 1) = 300 ∨ (h - 1) * w * (d - 1) = 0 :=
by
  sorry

end sugar_cubes_left_l98_98789


namespace find_a_l98_98931

noncomputable def f (x : ℝ) (a : ℝ) := 3^x + a / (3^x + 1)

theorem find_a (a : ℝ) (h₁ : 0 < a) (h₂ : ∀ x, f x a ≥ 5) (h₃ : ∃ x, f x a = 5) : a = 9 := by
  sorry

end find_a_l98_98931


namespace greatest_perimeter_l98_98351

theorem greatest_perimeter :
  ∀ (y : ℤ), (y > 4 ∧ y < 20 / 3) → (∃ p : ℤ, p = y + 4 * y + 20 ∧ p = 50) :=
by
  sorry

end greatest_perimeter_l98_98351


namespace greatest_possible_perimeter_l98_98381

theorem greatest_possible_perimeter :
  ∃ (x : ℕ), (4 * x < x + 20) ∧ (x + 20 > 4 * x) ∧ (x + 4 * x > 20) ∧ (4 < x ∧ x ≤ 6) ∧
  (∀ y, (4 * y < y + 20) ∧ (y + 20 > 4 * y) ∧ (y + 4 * y > 20) ∧ (4 < y ∧ y ≤ 6) → 
    (x + 4 * x + 20 ≥ y + 4 * y + 20)) :=
by sorry

end greatest_possible_perimeter_l98_98381


namespace anne_already_made_8_drawings_l98_98463

-- Define the conditions as Lean definitions
def num_markers : ℕ := 12
def drawings_per_marker : ℚ := 3 / 2 -- Equivalent to 1.5
def remaining_drawings : ℕ := 10

-- Calculate the total number of drawings Anne can make with her markers
def total_drawings : ℚ := num_markers * drawings_per_marker

-- Calculate the already made drawings
def already_made_drawings : ℚ := total_drawings - remaining_drawings

-- The theorem to prove
theorem anne_already_made_8_drawings : already_made_drawings = 8 := 
by 
  have h1 : total_drawings = 18 := by sorry -- Calculating total drawings as 18
  have h2 : already_made_drawings = 8 := by sorry -- Calculating already made drawings as total drawings minus remaining drawings
  exact h2

end anne_already_made_8_drawings_l98_98463


namespace factorize_expression_l98_98587

-- Define the variables a and b
variables (a b : ℝ)

-- State the theorem
theorem factorize_expression : 5*a^2*b - 20*b^3 = 5*b*(a + 2*b)*(a - 2*b) :=
by sorry

end factorize_expression_l98_98587


namespace pure_milk_in_final_solution_l98_98452

noncomputable def final_quantity_of_milk (initial_milk : ℕ) (milk_removed_each_step : ℕ) (steps : ℕ) : ℝ :=
  let remaining_milk_step1 := initial_milk - milk_removed_each_step
  let proportion := (milk_removed_each_step : ℝ) / (initial_milk : ℝ)
  let milk_removed_step2 := proportion * remaining_milk_step1
  remaining_milk_step1 - milk_removed_step2

theorem pure_milk_in_final_solution :
  final_quantity_of_milk 30 9 2 = 14.7 :=
by
  sorry

end pure_milk_in_final_solution_l98_98452


namespace perfect_square_expression_l98_98254

theorem perfect_square_expression (p : ℝ) (h : p = 0.28) : 
  (12.86 * 12.86 + 12.86 * p + 0.14 * 0.14) = (12.86 + 0.14) * (12.86 + 0.14) :=
by 
  -- proof goes here
  sorry

end perfect_square_expression_l98_98254


namespace mother_kept_one_third_l98_98706

-- Define the problem conditions
def total_sweets : ℕ := 27
def eldest_sweets : ℕ := 8
def youngest_sweets : ℕ := eldest_sweets / 2
def second_sweets : ℕ := 6
def total_children_sweets : ℕ := eldest_sweets + youngest_sweets + second_sweets
def sweets_mother_kept : ℕ := total_sweets - total_children_sweets
def fraction_mother_kept : ℚ := sweets_mother_kept / total_sweets

-- Prove the fraction of sweets the mother kept
theorem mother_kept_one_third : fraction_mother_kept = 1 / 3 := 
  by
    sorry

end mother_kept_one_third_l98_98706


namespace tile_ratio_l98_98012

/-- Given the initial configuration and extension method, the ratio of black tiles to white tiles in the new design is 22/27. -/
theorem tile_ratio (initial_black : ℕ) (initial_white : ℕ) (border_black : ℕ) (border_white : ℕ) (total_tiles : ℕ)
  (h1 : initial_black = 10)
  (h2 : initial_white = 15)
  (h3 : border_black = 12)
  (h4 : border_white = 12)
  (h5 : total_tiles = 49) :
  (initial_black + border_black) / (initial_white + border_white) = 22 / 27 := 
by {
  /- 
     Here we would provide the proof steps if needed.
     This is a theorem stating that the ratio of black to white tiles 
     in the new design is 22 / 27 given the initial conditions.
  -/
  sorry 
}

end tile_ratio_l98_98012


namespace quadratic_equation_in_x_l98_98175

theorem quadratic_equation_in_x (k x : ℝ) : 
  (k^2 + 1) * x^2 - (k * x - 8) - 1 = 0 := 
sorry

end quadratic_equation_in_x_l98_98175


namespace part1_eq_part2_if_empty_intersection_then_a_geq_3_l98_98495

open Set

variable {U : Type} {a : ℝ}

def universal_set : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}
def B1 (a : ℝ) : Set ℝ := {x : ℝ | x > a}
def complement_B1 (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def intersection_with_complement (a : ℝ) : Set ℝ := A ∩ complement_B1 a

-- Statement for part (1)
theorem part1_eq {a : ℝ} (h : a = 2) : intersection_with_complement a = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by sorry

-- Statement for part (2)
theorem part2_if_empty_intersection_then_a_geq_3 
(h : A ∩ B1 a = ∅) : a ≥ 3 :=
by sorry

end part1_eq_part2_if_empty_intersection_then_a_geq_3_l98_98495


namespace unique_B_squared_l98_98224

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

theorem unique_B_squared (h : B ^ 4 = 0) :
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B ^ 2 = B2 :=
by sorry

end unique_B_squared_l98_98224


namespace negation_proposition_l98_98019

theorem negation_proposition :
  (¬ ∃ x : ℝ, (x > -1 ∧ x < 3) ∧ (x^2 - 1 ≤ 2 * x)) ↔ 
  (∀ x : ℝ, (x > -1 ∧ x < 3) → (x^2 - 1 > 2 * x)) :=
by {
  sorry
}

end negation_proposition_l98_98019


namespace max_additional_voters_l98_98049

theorem max_additional_voters (x n y : ℕ) (hx : 0 ≤ x ∧ x ≤ 10) (hy : y = x - n - 1)
  (hT : (nx / n).is_integer) (h_decrease : ∀ v, (nx + v) / (n + 1) = x - 1 → ∀ m, x - m ≤ 0 → m ≤ 5) :
  ∃ y, y ≥ 0 ∧ y ≤ 5 := sorry

end max_additional_voters_l98_98049


namespace cube_volume_l98_98114

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98114


namespace max_fraction_l98_98744

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  1 + y / x ≤ -2 :=
sorry

end max_fraction_l98_98744


namespace continuity_at_three_l98_98416

noncomputable def f (x : ℝ) : ℝ := -2 * x ^ 2 - 4

theorem continuity_at_three (ε : ℝ) (hε : 0 < ε) :
  ∃ δ > 0, ∀ x : ℝ, |x - 3| < δ → |f x - f 3| < ε :=
sorry

end continuity_at_three_l98_98416


namespace cube_volume_l98_98112

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V : ℝ, V = 125 :=
by {
  let area_of_one_face := S / 6,
  let edge_length := real.sqrt area_of_one_face,
  let volume := edge_length ^ 3,
  use volume,
  have h_area_of_one_face : area_of_one_face = 25 := by {
    calc area_of_one_face = S / 6 : rfl
                     ... = 150 / 6 : by rw h
                     ... = 25 : by norm_num,
  },
  have h_edge_length : edge_length = 5 := by {
    calc edge_length = real.sqrt 25 : by rw h_area_of_one_face
                 ... = 5 : by norm_num,
  },
  show volume = 125, from by {
    calc volume = 5 ^ 3 : by rw h_edge_length
           ... = 125 : by norm_num,
    },
}

end cube_volume_l98_98112


namespace petrol_price_l98_98697

theorem petrol_price (P : ℝ) (h : 0.9 * P = 0.9 * P) : (250 / (0.9 * P) - 250 / P = 5) → P = 5.56 :=
by
  sorry

end petrol_price_l98_98697


namespace rationalize_sqrt_5_over_12_l98_98823

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end rationalize_sqrt_5_over_12_l98_98823


namespace elodie_rats_l98_98404

-- Define the problem conditions as hypotheses
def E (H : ℕ) : ℕ := H + 10
def K (H : ℕ) : ℕ := 3 * (E H + H)

-- The goal is to prove E = 30 given the conditions
theorem elodie_rats (H : ℕ) (h1 : E (H := H) + H + K (H := H) = 200) : E H = 30 :=
by
  sorry

end elodie_rats_l98_98404


namespace sum_of_prime_factors_2310_l98_98250

def is_prime (n : Nat) : Prop :=
  2 ≤ n ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def prime_factors_sum (n : Nat) : Nat :=
  (List.filter Nat.Prime (Nat.factors n)).sum

theorem sum_of_prime_factors_2310 :
  prime_factors_sum 2310 = 28 :=
by
  sorry

end sum_of_prime_factors_2310_l98_98250


namespace balls_into_boxes_l98_98203

theorem balls_into_boxes {balls boxes : ℕ} (h_balls : balls = 6) (h_boxes : boxes = 4) : 
  (indistinguishable_partitions balls boxes).count = 9 := 
by
  sorry

end balls_into_boxes_l98_98203


namespace greatest_possible_perimeter_l98_98364

noncomputable def max_perimeter (x : ℕ) : ℕ := if 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x then x + 4 * x + 20 else 0

theorem greatest_possible_perimeter : 
  (∀ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x → max_perimeter x ≤ 50) ∧ (∃ x : ℕ, 5 * x > 20 ∧ x + 20 > 4 * x ∧ 4 * x + 20 > x ∧ max_perimeter x = 50) :=
by
  sorry

end greatest_possible_perimeter_l98_98364


namespace lucy_bank_balance_l98_98408

def initial_balance : ℕ := 65
def deposit : ℕ := 15
def withdrawal : ℕ := 4

theorem lucy_bank_balance : initial_balance + deposit - withdrawal = 76 := by
  rw [← Nat.add_sub_assoc, Nat.add_sub_self, Nat.add_zero]
  exact rfl

end lucy_bank_balance_l98_98408


namespace tracy_initial_candies_l98_98690

theorem tracy_initial_candies (y : ℕ) 
  (condition1 : y - y / 4 = y * 3 / 4)
  (condition2 : y * 3 / 4 - (y * 3 / 4) / 3 = y / 2)
  (condition3 : y / 2 - 24 = y / 2 - 12 - 12)
  (condition4 : y / 2 - 24 - 4 = 2) : 
  y = 60 :=
by sorry

end tracy_initial_candies_l98_98690
