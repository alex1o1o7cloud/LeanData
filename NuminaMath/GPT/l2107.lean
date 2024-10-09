import Mathlib

namespace alexa_emily_profit_l2107_210729

def lemonade_stand_profit : ℕ :=
  let total_expenses := 10 + 5 + 3
  let price_per_cup := 4
  let cups_sold := 21
  let total_revenue := price_per_cup * cups_sold
  total_revenue - total_expenses

theorem alexa_emily_profit : lemonade_stand_profit = 66 :=
  by
  sorry

end alexa_emily_profit_l2107_210729


namespace mean_of_set_l2107_210764

theorem mean_of_set {m : ℝ} 
  (median_condition : (m + 8 + m + 11) / 2 = 19) : 
  (m + (m + 6) + (m + 8) + (m + 11) + (m + 18) + (m + 20)) / 6 = 20 := 
by 
  sorry

end mean_of_set_l2107_210764


namespace part1_part2_l2107_210781

def partsProcessedA : ℕ → ℕ
| 0 => 10
| (n + 1) => if n = 0 then 8 else partsProcessedA n - 2

def partsProcessedB : ℕ → ℕ
| 0 => 8
| (n + 1) => if n = 0 then 7 else partsProcessedB n - 1

def partsProcessedLineB_A (n : ℕ) := 7 * n
def partsProcessedLineB_B (n : ℕ) := 8 * n

def maxSetsIn14Days : ℕ := 
  let aLineA := 2 * (10 + 8 + 6) + (10 + 8)
  let aLineB := 2 * (8 + 7 + 6) + (8 + 8)
  min aLineA aLineB

theorem part1 :
  partsProcessedA 0 + partsProcessedA 1 + partsProcessedA 2 = 24 := 
by sorry

theorem part2 :
  maxSetsIn14Days = 106 :=
by sorry

end part1_part2_l2107_210781


namespace range_of_m_l2107_210765

noncomputable def set_M (m : ℝ) : Set ℝ := {x | x < m}
noncomputable def set_N : Set ℝ := {y | ∃ (x : ℝ), y = Real.log x / Real.log 2 - 1 ∧ 4 ≤ x}

theorem range_of_m (m : ℝ) : set_M m ∩ set_N = ∅ → m < 1 
:= by
  sorry

end range_of_m_l2107_210765


namespace f_100_eq_11_l2107_210790
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def f (n : ℕ) : ℕ := sum_of_digits (n^2 + 1)

def f_iter : ℕ → ℕ → ℕ
| 0,    n => f n
| k+1,  n => f (f_iter k n)

theorem f_100_eq_11 (n : ℕ) (h : n = 1990) : f_iter 100 n = 11 := by
  sorry

end f_100_eq_11_l2107_210790


namespace isosceles_obtuse_triangle_smallest_angle_l2107_210774

theorem isosceles_obtuse_triangle_smallest_angle :
  ∀ (α β : ℝ), 0 < α ∧ α = 1.5 * 90 ∧ α + 2 * β = 180 ∧ β = 22.5 := by
  sorry

end isosceles_obtuse_triangle_smallest_angle_l2107_210774


namespace simplified_expression_evaluation_l2107_210726

theorem simplified_expression_evaluation (x : ℝ) (hx : x = Real.sqrt 7) :
    (2 * x + 3) * (2 * x - 3) - (x + 2)^2 + 4 * (x + 3) = 20 :=
by
  sorry

end simplified_expression_evaluation_l2107_210726


namespace base_subtraction_l2107_210703

def base8_to_base10 (n : Nat) : Nat :=
  -- base 8 number 54321 (in decimal representation)
  5 * 4096 + 4 * 512 + 3 * 64 + 2 * 8 + 1

def base5_to_base10 (n : Nat) : Nat :=
  -- base 5 number 4321 (in decimal representation)
  4 * 125 + 3 * 25 + 2 * 5 + 1

theorem base_subtraction :
  base8_to_base10 54321 - base5_to_base10 4321 = 22151 := by
  sorry

end base_subtraction_l2107_210703


namespace sum_sequence_a_b_eq_1033_l2107_210747

def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

theorem sum_sequence_a_b_eq_1033 : 
  (a (b 1)) + (a (b 2)) + (a (b 3)) + (a (b 4)) + (a (b 5)) + 
  (a (b 6)) + (a (b 7)) + (a (b 8)) + (a (b 9)) + (a (b 10)) = 1033 := by
  sorry

end sum_sequence_a_b_eq_1033_l2107_210747


namespace calc_root_diff_l2107_210767

theorem calc_root_diff : 81^(1/4) - 16^(1/2) = -1 := by
  sorry

end calc_root_diff_l2107_210767


namespace find_y_l2107_210751

theorem find_y (a b c x : ℝ) (p q r y: ℝ) (hx : x ≠ 1) 
  (h₁ : (Real.log a) / p = Real.log x) 
  (h₂ : (Real.log b) / q = Real.log x) 
  (h₃ : (Real.log c) / r = Real.log x)
  (h₄ : (b^3) / (a^2 * c) = x^y) : 
  y = 3 * q - 2 * p - r := 
by {
  sorry
}

end find_y_l2107_210751


namespace lines_with_equal_intercepts_l2107_210746

theorem lines_with_equal_intercepts (A : ℝ × ℝ) (hA : A = (1, 2)) :
  ∃ (n : ℕ), n = 3 ∧ (∀ l : ℝ → ℝ, (l 1 = 2) → ((l 0 = l (-0)) ∨ (l (-0) = l 0))) :=
by
  sorry

end lines_with_equal_intercepts_l2107_210746


namespace find_incorrect_value_l2107_210787

theorem find_incorrect_value (n : ℕ) (mean_initial mean_correct : ℕ) (wrongly_copied correct_value incorrect_value : ℕ) 
  (h1 : n = 30) 
  (h2 : mean_initial = 150) 
  (h3 : mean_correct = 151) 
  (h4 : correct_value = 165) 
  (h5 : n * mean_initial = 4500) 
  (h6 : n * mean_correct = 4530) 
  (h7 : n * mean_correct - n * mean_initial = 30) 
  (h8 : correct_value - (n * mean_correct - n * mean_initial) = incorrect_value) : 
  incorrect_value = 135 :=
by
  sorry

end find_incorrect_value_l2107_210787


namespace cos_diff_half_l2107_210794

theorem cos_diff_half (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1 / 2)
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) :
  Real.cos (α - β) = -1 / 2 :=
by
  sorry

end cos_diff_half_l2107_210794


namespace angle_between_hands_at_3_40_l2107_210731

def degrees_per_minute_minute_hand := 360 / 60
def minutes_passed := 40
def degrees_minute_hand := degrees_per_minute_minute_hand * minutes_passed -- 240 degrees

def degrees_per_hour_hour_hand := 360 / 12
def hours_passed := 3
def degrees_hour_hand_at_hour := degrees_per_hour_hour_hand * hours_passed -- 90 degrees

def degrees_per_minute_hour_hand := degrees_per_hour_hour_hand / 60
def degrees_hour_hand_additional := degrees_per_minute_hour_hand * minutes_passed -- 20 degrees

def total_degrees_hour_hand := degrees_hour_hand_at_hour + degrees_hour_hand_additional -- 110 degrees

def expected_angle_between_hands := 130

theorem angle_between_hands_at_3_40
  (h1: degrees_minute_hand = 240)
  (h2: total_degrees_hour_hand = 110):
  (degrees_minute_hand - total_degrees_hour_hand = expected_angle_between_hands) :=
by
  sorry

end angle_between_hands_at_3_40_l2107_210731


namespace average_age_of_second_group_is_16_l2107_210788

theorem average_age_of_second_group_is_16
  (total_age_15_students : ℕ := 225)
  (total_age_first_group_7_students : ℕ := 98)
  (age_15th_student : ℕ := 15) :
  (total_age_15_students - total_age_first_group_7_students - age_15th_student) / 7 = 16 := 
by
  sorry

end average_age_of_second_group_is_16_l2107_210788


namespace count_special_integers_l2107_210760

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def base7 (n : ℕ) : ℕ := 
  let c := n / 343
  let rem1 := n % 343
  let d := rem1 / 49
  let rem2 := rem1 % 49
  let e := rem2 / 7
  let f := rem2 % 7
  343 * c + 49 * d + 7 * e + f

def base8 (n : ℕ) : ℕ := 
  let g := n / 512
  let rem1 := n % 512
  let h := rem1 / 64
  let rem2 := rem1 % 64
  let i := rem2 / 8
  let j := rem2 % 8
  512 * g + 64 * h + 8 * i + j

def matches_last_two_digits (n t : ℕ) : Prop := (t % 100) = (3 * (n % 100))

theorem count_special_integers : 
  ∃! (N : ℕ), is_three_digit N ∧ 
    matches_last_two_digits N (base7 N + base8 N) :=
sorry

end count_special_integers_l2107_210760


namespace slope_of_line_l2107_210718

theorem slope_of_line (x y : ℝ) : (∃ (m b : ℝ), (3 * y + 2 * x = 12) ∧ (m = -2 / 3) ∧ (y = m * x + b)) :=
sorry

end slope_of_line_l2107_210718


namespace g_of_f_of_3_eq_1902_l2107_210783

def f (x : ℕ) := x^3 - 2
def g (x : ℕ) := 3 * x^2 + x + 2

theorem g_of_f_of_3_eq_1902 : g (f 3) = 1902 := by
  sorry

end g_of_f_of_3_eq_1902_l2107_210783


namespace distinct_points_count_l2107_210773

theorem distinct_points_count :
  ∃ (P : Finset (ℝ × ℝ)), 
    (∀ p ∈ P, p.1^2 + p.2^2 = 1 ∧ p.1^2 + 9 * p.2^2 = 9) ∧ P.card = 2 :=
by
  sorry

end distinct_points_count_l2107_210773


namespace largest_n_divides_1005_fact_l2107_210748

theorem largest_n_divides_1005_fact (n : ℕ) : (∃ n, 10^n ∣ (Nat.factorial 1005)) ↔ n = 250 :=
by
  sorry

end largest_n_divides_1005_fact_l2107_210748


namespace divisible_by_6_l2107_210782

theorem divisible_by_6 (n : ℕ) : 6 ∣ ((n - 1) * n * (n^3 + 1)) := sorry

end divisible_by_6_l2107_210782


namespace quadratic_inequality_condition_l2107_210740

theorem quadratic_inequality_condition (x : ℝ) : x^2 - 2*x - 3 < 0 ↔ x ∈ Set.Ioo (-1) 3 := 
sorry

end quadratic_inequality_condition_l2107_210740


namespace family_eggs_count_l2107_210733

theorem family_eggs_count : 
  ∀ (initial_eggs parent_use child_use : ℝ) (chicken1 chicken2 chicken3 chicken4 : ℝ), 
    initial_eggs = 25 →
    parent_use = 7.5 + 2.5 →
    chicken1 = 2.5 →
    chicken2 = 3 →
    chicken3 = 4.5 →
    chicken4 = 1 →
    child_use = 1.5 + 0.5 →
    (initial_eggs - parent_use + (chicken1 + chicken2 + chicken3 + chicken4) - child_use) = 24 :=
by
  intros initial_eggs parent_use child_use chicken1 chicken2 chicken3 chicken4 
         h_initial_eggs h_parent_use h_chicken1 h_chicken2 h_chicken3 h_chicken4 h_child_use
  -- Proof goes here
  sorry

end family_eggs_count_l2107_210733


namespace maximum_value_l2107_210789

noncomputable def maxValue (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y^2 + x^3 * y^3 + x^2 * y^4 + x * y^5

theorem maximum_value (x y : ℝ) (h : x + y = 5) : maxValue x y h ≤ 625 / 4 :=
sorry

end maximum_value_l2107_210789


namespace slope_of_line_through_points_l2107_210786

theorem slope_of_line_through_points :
  let x1 := 1
  let y1 := 3
  let x2 := 5
  let y2 := 7
  let m := (y2 - y1) / (x2 - x1)
  m = 1 := by
  sorry

end slope_of_line_through_points_l2107_210786


namespace dave_apps_left_l2107_210736

theorem dave_apps_left (A : ℕ) 
  (h1 : 24 = A + 22) : A = 2 :=
by
  sorry

end dave_apps_left_l2107_210736


namespace a_lt_1_sufficient_but_not_necessary_l2107_210762

noncomputable def represents_circle (a : ℝ) : Prop :=
  a^2 - 10 * a + 9 > 0

theorem a_lt_1_sufficient_but_not_necessary (a : ℝ) :
  represents_circle a → ((a < 1) ∨ (a > 9)) :=
sorry

end a_lt_1_sufficient_but_not_necessary_l2107_210762


namespace polynomial_divisibility_l2107_210711

theorem polynomial_divisibility (p q : ℝ) :
    (∀ x, x = -2 ∨ x = 3 → (x^6 - x^5 + x^4 - p*x^3 + q*x^2 - 7*x - 35) = 0) →
    (p, q) = (6.86, -36.21) :=
by
  sorry

end polynomial_divisibility_l2107_210711


namespace sin_15_add_sin_75_l2107_210793

theorem sin_15_add_sin_75 : 
  Real.sin (15 * Real.pi / 180) + Real.sin (75 * Real.pi / 180) = Real.sqrt 6 / 2 :=
by
  sorry

end sin_15_add_sin_75_l2107_210793


namespace regression_decrease_by_5_l2107_210757

theorem regression_decrease_by_5 (x y : ℝ) (h : y = 2 - 2.5 * x) :
  y = 2 - 2.5 * (x + 2) → y ≠ 2 - 2.5 * x - 5 :=
by sorry

end regression_decrease_by_5_l2107_210757


namespace tan_equals_three_l2107_210744

variable (α : ℝ)

theorem tan_equals_three : 
  (Real.tan α = 3) → (1 / (Real.sin α * Real.sin α + 2 * Real.sin α * Real.cos α) = 2 / 3) :=
by
  intro h
  sorry

end tan_equals_three_l2107_210744


namespace triangle_inequality_equality_condition_l2107_210778

variables {A B C a b c : ℝ}

theorem triangle_inequality (A a B b C c : ℝ) :
  A * a + B * b + C * c ≥ 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b) :=
sorry

theorem equality_condition (A B C a b c : ℝ) :
  (A * a + B * b + C * c = 1 / 2 * (A * b + B * a + A * c + C * a + B * c + C * b)) ↔ (a = b ∧ b = c ∧ A = B ∧ B = C) :=
sorry

end triangle_inequality_equality_condition_l2107_210778


namespace min_sum_reciprocal_l2107_210708

theorem min_sum_reciprocal (a b c : ℝ) (hp0 : 0 < a) (hp1 : 0 < b) (hp2 : 0 < c) (h : a + b + c = 1) : 
  (1 / a) + (1 / b) + (1 / c) ≥ 9 :=
by
  sorry

end min_sum_reciprocal_l2107_210708


namespace units_digit_7_pow_451_l2107_210721

theorem units_digit_7_pow_451 : (7^451 % 10) = 3 := by
  sorry

end units_digit_7_pow_451_l2107_210721


namespace denver_wood_used_per_birdhouse_l2107_210702

-- Definitions used in the problem
def cost_per_piee_of_wood : ℝ := 1.50
def profit_per_birdhouse : ℝ := 5.50
def price_for_two_birdhouses : ℝ := 32
def num_birdhouses_purchased : ℝ := 2

-- Property to prove
theorem denver_wood_used_per_birdhouse (W : ℝ) 
  (h : num_birdhouses_purchased * (cost_per_piee_of_wood * W + profit_per_birdhouse) = price_for_two_birdhouses) : 
  W = 7 :=
sorry

end denver_wood_used_per_birdhouse_l2107_210702


namespace smallest_sum_of_digits_l2107_210719

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem smallest_sum_of_digits (N : ℕ) (hN_pos : 0 < N) 
  (h : sum_of_digits N = 3 * sum_of_digits (N + 1)) :
  sum_of_digits N = 12 :=
by {
  sorry
}

end smallest_sum_of_digits_l2107_210719


namespace math_problem_l2107_210761

theorem math_problem :
  18 * 35 + 45 * 18 - 18 * 10 = 1260 :=
by
  sorry

end math_problem_l2107_210761


namespace sum_of_GCF_and_LCM_l2107_210707

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l2107_210707


namespace sophie_hours_needed_l2107_210795

-- Sophie needs 206 hours to finish the analysis of all bones.
theorem sophie_hours_needed (num_bones : ℕ) (time_per_bone : ℕ) (total_hours : ℕ) (h1 : num_bones = 206) (h2 : time_per_bone = 1) : 
  total_hours = num_bones * time_per_bone :=
by
  rw [h1, h2]
  norm_num
  sorry

end sophie_hours_needed_l2107_210795


namespace arithmetic_sequence_number_of_terms_l2107_210709

theorem arithmetic_sequence_number_of_terms 
  (a d : ℝ) (n : ℕ) 
  (h1 : a + (a + d) + (a + 2 * d) = 34) 
  (h2 : (a + (n-3) * d) + (a + (n-2) * d) + (a + (n-1) * d) = 146) 
  (h3 : (n : ℝ) / 2 * (2 * a + (n - 1) * d) = 390) : 
  n = 13 :=
by 
  sorry

end arithmetic_sequence_number_of_terms_l2107_210709


namespace expression_divisible_by_x_minus_1_squared_l2107_210763

theorem expression_divisible_by_x_minus_1_squared :
  ∀ (n : ℕ) (x : ℝ), x ≠ 1 →
  (n * x^(n + 1) * (1 - 1 / x) - x^n * (1 - 1 / x^n)) / (x - 1)^2 = 
  (n * x^(n + 1) - n * x^n - x^n + 1) / (x - 1)^2 :=
by
  intro n x hx_ne_1
  sorry

end expression_divisible_by_x_minus_1_squared_l2107_210763


namespace ratio_of_speeds_l2107_210715

theorem ratio_of_speeds (k r t V1 V2 : ℝ) (hk : 0 < k) (hr : 0 < r) (ht : 0 < t)
    (h1 : r * (V1 - V2) = k) (h2 : t * (V1 + V2) = k) :
    |r + t| / |r - t| = V1 / V2 :=
by
  sorry

end ratio_of_speeds_l2107_210715


namespace curve_intersection_three_points_l2107_210714

theorem curve_intersection_three_points (a : ℝ) :
  (∀ x y : ℝ, ((x^2 - y^2 = a^2) ∧ ((x-1)^2 + y^2 = 1)) → (a = 0)) :=
by
  sorry

end curve_intersection_three_points_l2107_210714


namespace subset_of_primes_is_all_primes_l2107_210754

theorem subset_of_primes_is_all_primes
  (P : Set ℕ)
  (M : Set ℕ)
  (hP : ∀ n, n ∈ P ↔ Nat.Prime n)
  (hM : ∀ S : Finset ℕ, (∀ p ∈ S, p ∈ M) → ∀ p, p ∣ (Finset.prod S id + 1) → p ∈ M) :
  M = P :=
sorry

end subset_of_primes_is_all_primes_l2107_210754


namespace original_daily_production_l2107_210713

theorem original_daily_production (x N : ℕ) (h1 : N = (x - 3) * 31 + 60) (h2 : N = (x + 3) * 25 - 60) : x = 8 :=
sorry

end original_daily_production_l2107_210713


namespace major_premise_wrong_l2107_210739

-- Definition of the problem conditions and the proof goal
theorem major_premise_wrong :
  (∀ a : ℝ, |a| > 0) ↔ false :=
by {
  sorry  -- the proof goes here but is omitted as per the instructions
}

end major_premise_wrong_l2107_210739


namespace derivative_at_neg_one_l2107_210732

def f (x : ℝ) : ℝ := List.prod (List.map (λ k => (x^3 + k)) (List.range' 1 100))

theorem derivative_at_neg_one : deriv f (-1) = 3 * Nat.factorial 99 := by
  sorry

end derivative_at_neg_one_l2107_210732


namespace zero_count_at_end_of_45_320_125_product_l2107_210724

theorem zero_count_at_end_of_45_320_125_product :
  let p := 45 * 320 * 125
  45 = 5 * 3^2 ∧ 320 = 2^6 * 5 ∧ 125 = 5^3 →
  p = 2^6 * 3^2 * 5^5 →
  p % 10^5 = 0 ∧ p % 10^6 ≠ 0 :=
by
  sorry

end zero_count_at_end_of_45_320_125_product_l2107_210724


namespace f_11_5_equals_neg_1_l2107_210756

-- Define the function f with the given properties
axiom odd_function (f : ℝ → ℝ) : ∀ x, f (-x) = -f x
axiom periodic_function (f : ℝ → ℝ) : ∀ x, f (x + 2) = f x
axiom f_interval (f : ℝ → ℝ) : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x

-- State the theorem to be proved
theorem f_11_5_equals_neg_1 (f : ℝ → ℝ) 
  (odd_f : ∀ x, f (-x) = -f x)
  (periodic_f : ∀ x, f (x + 2) = f x)
  (f_int : ∀ x, 0 ≤ x ∧ x ≤ 1 → f x = 2 * x) :
  f (11.5) = -1 :=
sorry

end f_11_5_equals_neg_1_l2107_210756


namespace min_value_expression_l2107_210717

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2 / b) * (a + 2 / b - 1010) + (b + 2 / a) * (b + 2 / a - 1010) + 101010 = -404040 :=
sorry

end min_value_expression_l2107_210717


namespace remainder_24_2377_mod_15_l2107_210741

theorem remainder_24_2377_mod_15 :
  24^2377 % 15 = 9 :=
sorry

end remainder_24_2377_mod_15_l2107_210741


namespace red_balls_in_bag_l2107_210701

theorem red_balls_in_bag : ∃ x : ℕ, (3 : ℚ) / (4 + (x : ℕ)) = 1 / 2 ∧ x = 2 := sorry

end red_balls_in_bag_l2107_210701


namespace gcd_factorial_8_6_squared_l2107_210730

theorem gcd_factorial_8_6_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_8_6_squared_l2107_210730


namespace fraction_filled_in_5_minutes_l2107_210710

-- Conditions
def fill_time : ℕ := 55 -- Total minutes to fill the cistern
def duration : ℕ := 5  -- Minutes we are examining

-- The theorem to prove that the fraction filled in 'duration' minutes is 1/11
theorem fraction_filled_in_5_minutes : (duration : ℚ) / (fill_time : ℚ) = 1 / 11 :=
by
  have fraction_per_minute : ℚ := 1 / fill_time
  have fraction_in_5_minutes : ℚ := duration * fraction_per_minute
  sorry -- Proof steps would go here, if needed.

end fraction_filled_in_5_minutes_l2107_210710


namespace extreme_point_of_f_l2107_210723

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - Real.log x

theorem extreme_point_of_f : 
  ∃ c : ℝ, c = Real.sqrt 3 / 3 ∧ (∀ x: ℝ, x > 0 → (f x > f c → x > c) ∧ (f x < f c → x < c)) := 
sorry

end extreme_point_of_f_l2107_210723


namespace evaluations_total_l2107_210768

theorem evaluations_total :
    let class_A_students := 30
    let class_A_mc := 12
    let class_A_essay := 3
    let class_A_presentation := 1

    let class_B_students := 25
    let class_B_mc := 15
    let class_B_short_answer := 5
    let class_B_essay := 2

    let class_C_students := 35
    let class_C_mc := 10
    let class_C_essay := 3
    let class_C_presentation_groups := class_C_students / 5 -- groups of 5

    let class_D_students := 40
    let class_D_mc := 11
    let class_D_short_answer := 4
    let class_D_essay := 3

    let class_E_students := 20
    let class_E_mc := 14
    let class_E_short_answer := 5
    let class_E_essay := 2

    let total_mc := (class_A_students * class_A_mc) +
                    (class_B_students * class_B_mc) +
                    (class_C_students * class_C_mc) +
                    (class_D_students * class_D_mc) +
                    (class_E_students * class_E_mc)

    let total_short_answer := (class_B_students * class_B_short_answer) +
                              (class_D_students * class_D_short_answer) +
                              (class_E_students * class_E_short_answer)

    let total_essay := (class_A_students * class_A_essay) +
                       (class_B_students * class_B_essay) +
                       (class_C_students * class_C_essay) +
                       (class_D_students * class_D_essay) +
                       (class_E_students * class_E_essay)

    let total_presentation := (class_A_students * class_A_presentation) +
                              class_C_presentation_groups

    total_mc + total_short_answer + total_essay + total_presentation = 2632 := by
    sorry

end evaluations_total_l2107_210768


namespace proof_find_C_proof_find_cos_A_l2107_210728

noncomputable def find_C {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : Prop :=
  ∃ (C : ℝ), 0 < C ∧ C < Real.pi ∧ C = Real.pi / 6

noncomputable def find_cos_A {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : Prop :=
  ∃ (A : ℝ), Real.cos A = (Real.sqrt 5 - 2 * Real.sqrt 3) / 6

theorem proof_find_C (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : find_C h1 :=
  sorry

theorem proof_find_cos_A (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : find_cos_A h1 h2 :=
  sorry

end proof_find_C_proof_find_cos_A_l2107_210728


namespace positive_value_of_X_l2107_210720

def hash_relation (X Y : ℕ) : ℕ := X^2 + Y^2

theorem positive_value_of_X (X : ℕ) (h : hash_relation X 7 = 290) : X = 17 :=
by sorry

end positive_value_of_X_l2107_210720


namespace unique_solution_of_inequality_l2107_210752

open Real

theorem unique_solution_of_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 + 2 * b * x + 2 * b| ≤ 1) ↔ b = 1 := 
by exact sorry

end unique_solution_of_inequality_l2107_210752


namespace right_triangle_set_C_l2107_210737

theorem right_triangle_set_C :
  ∃ (a b c : ℕ), a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2 :=
by
  sorry

end right_triangle_set_C_l2107_210737


namespace units_digit_fraction_l2107_210770

-- Given conditions
def numerator : ℕ := 30 * 31 * 32 * 33 * 34 * 35
def denominator : ℕ := 1500
def simplified_fraction : ℕ := 2^5 * 3 * 31 * 33 * 17 * 7

-- Statement of the proof goal
theorem units_digit_fraction :
  (simplified_fraction) % 10 = 2 := by
  sorry

end units_digit_fraction_l2107_210770


namespace john_pays_12_dollars_l2107_210755

/-- Define the conditions -/
def number_of_toys : ℕ := 5
def cost_per_toy : ℝ := 3
def discount_rate : ℝ := 0.2

/-- Define the total cost before discount -/
def total_cost_before_discount := number_of_toys * cost_per_toy

/-- Define the discount amount -/
def discount_amount := total_cost_before_discount * discount_rate

/-- Define the final amount John pays -/
def final_amount := total_cost_before_discount - discount_amount

/-- The theorem to be proven -/
theorem john_pays_12_dollars : final_amount = 12 := by
  -- Proof goes here
  sorry

end john_pays_12_dollars_l2107_210755


namespace core_temperature_calculation_l2107_210742

-- Define the core temperature of the Sun, given in degrees Celsius
def T_Sun : ℝ := 19200000

-- Define the multiple factor
def factor : ℝ := 312.5

-- The expected result in scientific notation
def expected_temperature : ℝ := 6.0 * (10 ^ 9)

-- Prove that the calculated temperature is equal to the expected temperature
theorem core_temperature_calculation : (factor * T_Sun) = expected_temperature := by
  sorry

end core_temperature_calculation_l2107_210742


namespace relationship_of_AT_l2107_210799

def S : ℝ := 300
def PC : ℝ := S + 500
def total_cost : ℝ := 2200

theorem relationship_of_AT (AT : ℝ) 
  (h1: S + PC + AT = total_cost) : 
  AT = S + PC - 400 :=
by
  sorry

end relationship_of_AT_l2107_210799


namespace sum_of_digits_of_a_l2107_210725

-- Define a as 10^10 - 47
def a : ℕ := (10 ^ 10) - 47

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The theorem to prove that the sum of all the digits of a is 81
theorem sum_of_digits_of_a : sum_of_digits a = 81 := by
  sorry

end sum_of_digits_of_a_l2107_210725


namespace infinite_primes_l2107_210775

theorem infinite_primes : ∀ (p : ℕ), Prime p → ¬ (∃ q : ℕ, Prime q ∧ q > p) := sorry

end infinite_primes_l2107_210775


namespace Nedy_crackers_total_l2107_210722

theorem Nedy_crackers_total :
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  (packs_from_Mon_to_Thu + packs_on_Fri) = 24 :=
by
  let packs_from_Mon_to_Thu := 8
  let packs_on_Fri := 2 * packs_from_Mon_to_Thu
  show packs_from_Mon_to_Thu + packs_on_Fri = 24
  sorry

end Nedy_crackers_total_l2107_210722


namespace distance_from_point_to_y_axis_l2107_210792

/-- Proof that the distance from point P(-4, 3) to the y-axis is 4. -/
theorem distance_from_point_to_y_axis {P : ℝ × ℝ} (hP : P = (-4, 3)) : |P.1| = 4 :=
by {
   -- The proof will depend on the properties of absolute value
   -- and the given condition about the coordinates of P.
   sorry
}

end distance_from_point_to_y_axis_l2107_210792


namespace car_speed_l2107_210759

theorem car_speed (distance : ℝ) (time : ℝ) (h_distance : distance = 495) (h_time : time = 5) : 
  distance / time = 99 :=
by
  rw [h_distance, h_time]
  norm_num

end car_speed_l2107_210759


namespace Amy_work_hours_l2107_210772

theorem Amy_work_hours (summer_weeks: ℕ) (summer_hours_per_week: ℕ) (summer_total_earnings: ℕ)
                       (school_weeks: ℕ) (school_total_earnings: ℕ) (hourly_wage: ℕ) 
                       (school_hours_per_week: ℕ):
    summer_weeks = 8 →
    summer_hours_per_week = 40 →
    summer_total_earnings = 3200 →
    school_weeks = 32 →
    school_total_earnings = 4800 →
    hourly_wage = summer_total_earnings / (summer_weeks * summer_hours_per_week) →
    school_hours_per_week = school_total_earnings / (hourly_wage * school_weeks) →
    school_hours_per_week = 15 :=
by
  intros
  sorry

end Amy_work_hours_l2107_210772


namespace tylenol_mg_per_tablet_l2107_210798

noncomputable def dose_intervals : ℕ := 3  -- Mark takes Tylenol 3 times
noncomputable def total_mg : ℕ := 3000     -- Total intake in milligrams
noncomputable def tablets_per_dose : ℕ := 2  -- Number of tablets per dose

noncomputable def tablet_mg : ℕ :=
  total_mg / dose_intervals / tablets_per_dose

theorem tylenol_mg_per_tablet : tablet_mg = 500 := by
  sorry

end tylenol_mg_per_tablet_l2107_210798


namespace jackson_pays_2100_l2107_210716

def tile_cost (length : ℝ) (width : ℝ) (tiles_per_sqft : ℝ) (percent_green : ℝ) (cost_green : ℝ) (cost_red : ℝ) : ℝ :=
  let area := length * width
  let total_tiles := area * tiles_per_sqft
  let green_tiles := total_tiles * percent_green
  let red_tiles := total_tiles - green_tiles
  let cost_green_total := green_tiles * cost_green
  let cost_red_total := red_tiles * cost_red
  cost_green_total + cost_red_total

theorem jackson_pays_2100 :
  tile_cost 10 25 4 0.4 3 1.5 = 2100 :=
by
  sorry

end jackson_pays_2100_l2107_210716


namespace maximum_m2_n2_l2107_210727

theorem maximum_m2_n2 
  (m n : ℤ)
  (hm : 1 ≤ m ∧ m ≤ 1981) 
  (hn : 1 ≤ n ∧ n ≤ 1981)
  (h : (n^2 - m*n - m^2)^2 = 1) :
  m^2 + n^2 ≤ 3524578 :=
sorry

end maximum_m2_n2_l2107_210727


namespace brand_tangyuan_purchase_l2107_210735

theorem brand_tangyuan_purchase (x y : ℕ) 
  (h1 : x + y = 1000) 
  (h2 : x = 2 * y + 20) : 
  x = 670 ∧ y = 330 := 
sorry

end brand_tangyuan_purchase_l2107_210735


namespace parallel_lines_equal_slopes_l2107_210771

theorem parallel_lines_equal_slopes (a : ℝ) :
  (∀ x y, ax + 2 * y + 3 * a = 0 → 3 * x + (a - 1) * y = -7 + a) →
  a = 3 := sorry

end parallel_lines_equal_slopes_l2107_210771


namespace derivative_of_odd_function_is_even_l2107_210785

-- Define an odd function f
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

-- Define the main theorem
theorem derivative_of_odd_function_is_even (f g : ℝ → ℝ) 
  (h1 : is_odd_function f) 
  (h2 : ∀ x, g x = deriv f x) :
  ∀ x, g (-x) = g x :=
by
  sorry

end derivative_of_odd_function_is_even_l2107_210785


namespace find_m_of_quadratic_fn_l2107_210750

theorem find_m_of_quadratic_fn (m : ℚ) (h : 2 * m - 1 = 2) : m = 3 / 2 :=
by
  sorry

end find_m_of_quadratic_fn_l2107_210750


namespace ratio_equation_solution_l2107_210776

theorem ratio_equation_solution (x : ℝ) :
  (4 + 2 * x) / (6 + 3 * x) = (2 + x) / (3 + 2 * x) → (x = 0 ∨ x = 4) :=
by
  -- the proof steps would go here
  sorry

end ratio_equation_solution_l2107_210776


namespace quadrilateral_angle_B_l2107_210734

/-- In quadrilateral ABCD,
given that angle A + angle C = 150 degrees,
prove that angle B = 105 degrees. -/
theorem quadrilateral_angle_B (A C : ℝ) (B : ℝ) (h1 : A + C = 150) (h2 : A + B = 180) : B = 105 :=
by
  sorry

end quadrilateral_angle_B_l2107_210734


namespace molecular_weight_of_ammonium_bromide_l2107_210712

-- Define the atomic weights for the elements.
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90

-- Define the molecular weight of ammonium bromide based on its formula NH4Br.
def molecular_weight_NH4Br : ℝ := (1 * atomic_weight_N) + (4 * atomic_weight_H) + (1 * atomic_weight_Br)

-- State the theorem that the molecular weight of ammonium bromide is approximately 97.95 g/mol.
theorem molecular_weight_of_ammonium_bromide :
  molecular_weight_NH4Br = 97.95 :=
by
  -- The following line is a placeholder to mark the theorem as correct without proving it.
  sorry

end molecular_weight_of_ammonium_bromide_l2107_210712


namespace triangle_angle_sum_33_75_l2107_210796

theorem triangle_angle_sum_33_75 (x : ℝ) 
  (h₁ : 45 + 3 * x + x = 180) : 
  x = 33.75 :=
  sorry

end triangle_angle_sum_33_75_l2107_210796


namespace remainder_equivalence_l2107_210704

theorem remainder_equivalence (x : ℕ) (r : ℕ) (hx_pos : 0 < x) 
  (h1 : ∃ q1, 100 = q1 * x + r) (h2 : ∃ q2, 197 = q2 * x + r) : 
  r = 3 :=
by
  sorry

end remainder_equivalence_l2107_210704


namespace colin_speed_l2107_210780

variable (B T Br C : ℝ)

def Bruce := B = 1
def Tony := T = 2 * B
def Brandon := Br = T / 3
def Colin := C = 6 * Br

theorem colin_speed : Bruce B → Tony B T → Brandon T Br → Colin Br C → C = 4 := by
  sorry

end colin_speed_l2107_210780


namespace g_value_at_2_l2107_210758

theorem g_value_at_2 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = x^2 - 2) : g 2 = 11 / 28 :=
sorry

end g_value_at_2_l2107_210758


namespace num_digits_divisible_l2107_210705

theorem num_digits_divisible (h : Nat) :
  (∃ n : Fin 10, (10 * 24 + n) % n = 0) -> h = 7 :=
by sorry

end num_digits_divisible_l2107_210705


namespace complete_square_solution_l2107_210766

theorem complete_square_solution (a b c : ℤ) (h1 : a^2 = 25) (h2 : 10 * b = 30) (h3 : (a * x + b)^2 = 25 * x^2 + 30 * x + c) :
  a + b + c = -58 :=
by
  sorry

end complete_square_solution_l2107_210766


namespace max_balls_drawn_l2107_210706

structure DrawingConditions where
  yellow_items : ℕ
  round_items : ℕ
  edible_items : ℕ
  tomato_properties : Prop
  ball_properties : Prop
  banana_properties : Prop

axiom petya_conditions : DrawingConditions → Prop 

theorem max_balls_drawn (cond : DrawingConditions)
  (h1 : cond.yellow_items = 15)
  (h2 : cond.round_items = 18)
  (h3 : cond.edible_items = 13)
  (h4 : cond.tomato_properties)
  (h5 : cond.ball_properties)
  (h6 : cond.banana_properties) :
  ∀ balls : ℕ, balls ≤ 18 :=
by
  sorry

end max_balls_drawn_l2107_210706


namespace total_count_not_47_l2107_210743

theorem total_count_not_47 (h c : ℕ) : 11 * h + 6 * c ≠ 47 := by
  sorry

end total_count_not_47_l2107_210743


namespace tom_purchases_mangoes_l2107_210797

theorem tom_purchases_mangoes (m : ℕ) (h1 : 8 * 70 + m * 65 = 1145) : m = 9 :=
by
  sorry

end tom_purchases_mangoes_l2107_210797


namespace find_OH_squared_l2107_210745

variables {O H : Type} {a b c R : ℝ}

-- Given conditions
def is_circumcenter (O : Type) (ABC : Type) := true -- Placeholder definition
def is_orthocenter (H : Type) (ABC : Type) := true -- Placeholder definition
def circumradius (O : Type) (R : ℝ) := true -- Placeholder definition
def sides_squared_sum (a b c : ℝ) := a^2 + b^2 + c^2

-- The theorem to be proven
theorem find_OH_squared (O H : Type) (a b c : ℝ) (R : ℝ) 
  (circ : is_circumcenter O ABC) 
  (orth: is_orthocenter H ABC) 
  (radius : circumradius O R) 
  (terms_sum : sides_squared_sum a b c = 50)
  (R_val : R = 10) 
  : OH^2 = 850 := 
sorry

end find_OH_squared_l2107_210745


namespace total_kids_attended_camp_l2107_210753

theorem total_kids_attended_camp :
  let n1 := 34044
  let n2 := 424944
  n1 + n2 = 458988 := 
by {
  sorry
}

end total_kids_attended_camp_l2107_210753


namespace sasha_took_right_triangle_l2107_210738

-- Define types of triangles
inductive Triangle
| acute
| right
| obtuse

open Triangle

-- Define the function that determines if Borya can form a triangle identical to Sasha's
def can_form_identical_triangle (t1 t2 t3: Triangle) : Bool :=
match t1, t2, t3 with
| right, acute, obtuse => true
| _ , _ , _ => false

-- Define the main theorem
theorem sasha_took_right_triangle : 
  ∀ (sasha_takes borya_takes1 borya_takes2 : Triangle),
  (sasha_takes ≠ borya_takes1 ∧ sasha_takes ≠ borya_takes2 ∧ borya_takes1 ≠ borya_takes2) →
  can_form_identical_triangle sasha_takes borya_takes1 borya_takes2 →
  sasha_takes = right :=
by sorry

end sasha_took_right_triangle_l2107_210738


namespace ratio_proof_l2107_210749

variable (a b c d : ℚ)

theorem ratio_proof 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 2)
  (h3 : c / d = 7) :
  d / a = 2 / 35 := by
  sorry

end ratio_proof_l2107_210749


namespace triangle_proof_l2107_210779

noncomputable def length_DC (AB DA BC DB : ℝ) : ℝ :=
  Real.sqrt (BC^2 - DB^2)

theorem triangle_proof :
  ∀ (AB DA BC DB : ℝ), AB = 30 → DA = 24 → BC = 22.5 → DB = 18 →
  length_DC AB DA BC DB = 13.5 :=
by
  intros AB DA BC DB hAB hDA hBC hDB
  rw [length_DC]
  sorry

end triangle_proof_l2107_210779


namespace martina_success_rate_l2107_210700

theorem martina_success_rate
  (games_played : ℕ) (games_won : ℕ) (games_remaining : ℕ)
  (games_won_remaining : ℕ) :
  games_played = 15 → 
  games_won = 9 → 
  games_remaining = 5 → 
  games_won_remaining = 5 → 
  ((games_won + games_won_remaining) / (games_played + games_remaining) : ℚ) * 100 = 70 := 
by
  intros h1 h2 h3 h4
  sorry

end martina_success_rate_l2107_210700


namespace fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l2107_210777

theorem fourth_vertex_of_regular_tetrahedron_exists_and_is_unique :
  ∃ (x y z : ℤ),
    (x, y, z) ≠ (1, 2, 3) ∧ (x, y, z) ≠ (5, 3, 2) ∧ (x, y, z) ≠ (4, 2, 6) ∧
    (x - 1)^2 + (y - 2)^2 + (z - 3)^2 = 18 ∧
    (x - 5)^2 + (y - 3)^2 + (z - 2)^2 = 18 ∧
    (x - 4)^2 + (y - 2)^2 + (z - 6)^2 = 18 ∧
    (x, y, z) = (2, 3, 5) :=
by
  -- Proof goes here
  sorry

end fourth_vertex_of_regular_tetrahedron_exists_and_is_unique_l2107_210777


namespace sum_of_coordinates_l2107_210769

theorem sum_of_coordinates (x y : ℝ) (h : x^2 + y^2 = 16 * x - 12 * y + 20) : x + y = 2 :=
sorry

end sum_of_coordinates_l2107_210769


namespace relationship_between_a_b_c_l2107_210791

noncomputable def a := 33
noncomputable def b := 5 * 6^1 + 2 * 6^0
noncomputable def c := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l2107_210791


namespace percentage_small_bottles_sold_l2107_210784

theorem percentage_small_bottles_sold :
  ∀ (x : ℕ), (6000 - (x * 60)) + 8500 = 13780 → x = 12 :=
by
  intro x h
  sorry

end percentage_small_bottles_sold_l2107_210784
