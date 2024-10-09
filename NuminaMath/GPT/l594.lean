import Mathlib

namespace solve_quadratic1_solve_quadratic2_l594_59414

-- For the first quadratic equation: 3x^2 = 6x
theorem solve_quadratic1 (x : ℝ) (h : 3 * x^2 = 6 * x) : x = 0 ∨ x = 2 :=
sorry

-- For the second quadratic equation: x^2 - 6x + 5 = 0
theorem solve_quadratic2 (x : ℝ) (h : x^2 - 6 * x + 5 = 0) : x = 5 ∨ x = 1 :=
sorry

end solve_quadratic1_solve_quadratic2_l594_59414


namespace problem_l594_59482

theorem problem (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 :=
by
  sorry

end problem_l594_59482


namespace geometric_sequence_properties_l594_59488

noncomputable def a_n (n : ℕ) : ℚ :=
if n = 0 then 1 / 4 else (1 / 4) * 2^(n-1)

def S_n (n : ℕ) : ℚ :=
(1/4) * (1 - 2^n) / (1 - 2)

theorem geometric_sequence_properties :
  (a_n 2 = 1 / 2) ∧ (∀ n : ℕ, 1 ≤ n → a_n n = 2^(n-3)) ∧ S_n 5 = 31 / 16 :=
by {
  sorry
}

end geometric_sequence_properties_l594_59488


namespace custom_op_two_neg_four_l594_59437

-- Define the binary operation *
def custom_op (x y : ℚ) : ℚ := (x * y) / (x + y)

-- Proposition stating 2 * (-4) = 4 using the custom operation
theorem custom_op_two_neg_four : custom_op 2 (-4) = 4 :=
by
  sorry

end custom_op_two_neg_four_l594_59437


namespace number_of_ways_to_place_rooks_l594_59474

theorem number_of_ways_to_place_rooks :
  let columns := 6
  let rows := 2006
  let rooks := 3
  ((Nat.choose columns rooks) * (rows * (rows - 1) * (rows - 2))) = 20 * 2006 * 2005 * 2004 :=
by {
  sorry
}

end number_of_ways_to_place_rooks_l594_59474


namespace find_g_5_l594_59432

-- Define the function g and the condition it satisfies
variable {g : ℝ → ℝ}
variable (hg : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x)

-- The proof goal
theorem find_g_5 : g 5 = 206 / 35 :=
by
  -- To be proven using the given condition hg
  sorry

end find_g_5_l594_59432


namespace product_of_roots_l594_59401

noncomputable def quadratic_has_product_of_roots (A B C : ℤ) : ℚ :=
  C / A

theorem product_of_roots (α β : ℚ) (h : 12 * α^2 + 28 * α - 320 = 0) (h2 : 12 * β^2 + 28 * β - 320 = 0) :
  quadratic_has_product_of_roots 12 28 (-320) = -80 / 3 :=
by
  -- Insert proof here
  sorry

end product_of_roots_l594_59401


namespace integer_type_l594_59472

theorem integer_type (f : ℕ) (h : f = 14) (x : ℕ) (hx : 3150 * f = x * x) : f > 0 :=
by
  sorry

end integer_type_l594_59472


namespace symmetric_with_origin_l594_59495

-- Define the original point P
def P : ℝ × ℝ := (2, -3)

-- Define the function for finding the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- Prove that the symmetric point of P with respect to the origin is (-2, 3)
theorem symmetric_with_origin :
  symmetric_point P = (-2, 3) :=
by
  -- Placeholders for proof
  sorry

end symmetric_with_origin_l594_59495


namespace rectangle_area_l594_59468

theorem rectangle_area (w l : ℝ) (h1 : l = 2 * w) (h2 : 2 * l + 2 * w = 4) :
  l * w = 8 / 9 :=
by
  sorry

end rectangle_area_l594_59468


namespace max_knights_l594_59434

/-- 
On an island with knights who always tell the truth and liars who always lie,
100 islanders seated around a round table where:
  - 50 of them say "both my neighbors are liars,"
  - The other 50 say "among my neighbors, there is exactly one liar."
Prove that the maximum number of knights at the table is 67.
-/
theorem max_knights (K L : ℕ) (h1 : K + L = 100) (h2 : ∃ k, k ≤ 25 ∧ K = 2 * k + (100 - 3 * k) / 2) : K = 67 :=
sorry

end max_knights_l594_59434


namespace find_n_divides_2_pow_2000_l594_59425

theorem find_n_divides_2_pow_2000 (n : ℕ) (h₁ : n > 2) :
  (1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ (2 ^ 2000) →
  n = 3 ∨ n = 7 ∨ n = 23 :=
sorry

end find_n_divides_2_pow_2000_l594_59425


namespace find_g_three_l594_59477

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_three (h : ∀ x : ℝ, g (3^x) + (x + 1) * g (3^(-x)) = 3) : g 3 = -3 :=
sorry

end find_g_three_l594_59477


namespace empty_subset_singleton_zero_l594_59492

theorem empty_subset_singleton_zero : ∅ ⊆ ({0} : Set ℕ) :=
by
  sorry

end empty_subset_singleton_zero_l594_59492


namespace case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l594_59443

theorem case_one_ellipses_foci_xaxis :
  ∀ (a : ℝ) (e : ℝ), a = 6 ∧ e = 2 / 3 → (∃ (b : ℝ), (b^2 = (a^2 - (e * a)^2) ∧ (a > 0) → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1)) ∨ (y^2 / a^2 + x^2 / b^2 = 1)))) :=
by
  sorry

theorem case_two_ellipses_foci_exact :
  ∀ (F1 F2 : ℝ × ℝ), F1 = (-4,0) ∧ F2 = (4,0) ∧ ∀ P : ℝ × ℝ, ((dist P F1) + (dist P F2) = 10) →
  ∃ (a : ℝ) (b : ℝ), a = 5 ∧ b^2 = a^2 - 4^2 → ((∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1))) :=
by
  sorry

end case_one_ellipses_foci_xaxis_case_two_ellipses_foci_exact_l594_59443


namespace inverse_of_composed_function_l594_59416

theorem inverse_of_composed_function :
  let f (x : ℝ) := 4 * x + 5
  let g (x : ℝ) := 3 * x - 4
  let k (x : ℝ) := f (g x)
  ∀ y : ℝ, k ( (y + 11) / 12 ) = y :=
by
  sorry

end inverse_of_composed_function_l594_59416


namespace second_divisor_is_24_l594_59408

theorem second_divisor_is_24 (m n k l : ℤ) (hm : m = 288 * k + 47) (hn : m = n * l + 23) : n = 24 :=
by
  sorry

end second_divisor_is_24_l594_59408


namespace min_value_f_l594_59493

noncomputable def f (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x - 1

theorem min_value_f (a : ℝ) : 
  (∀ x ∈ (Set.Icc (-1 : ℝ) 1), f a x ≥ 
    if a < -1 then 2 * a 
    else if -1 ≤ a ∧ a ≤ 1 then -1 - a ^ 2 
    else -2 * a) := 
by
  sorry

end min_value_f_l594_59493


namespace range_of_k_l594_59451

variable (k : ℝ)
def f (x : ℝ) : ℝ := k * x + 1
def g (x : ℝ) : ℝ := x^2 - 1

theorem range_of_k (h : ∀ x : ℝ, f k x > 0 ∨ g x > 0) : k ∈ Set.Ioo (-1 : ℝ) (1 : ℝ) := 
sorry

end range_of_k_l594_59451


namespace cos_theta_when_f_maximizes_l594_59496

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x)

theorem cos_theta_when_f_maximizes (θ : ℝ) (h : ∀ x, f x ≤ f θ) : Real.cos θ = Real.sqrt 3 / 2 := by
  sorry

end cos_theta_when_f_maximizes_l594_59496


namespace find_point_P_l594_59481

/-- 
Given two points A and B, find the coordinates of point P that lies on the line AB
and satisfies that the distance from A to P is half the vector from A to B.
-/
theorem find_point_P 
  (A B : ℝ × ℝ) 
  (hA : A = (3, -4)) 
  (hB : B = (-9, 2)) 
  (P : ℝ × ℝ) 
  (hP : P.1 - A.1 = (1/2) * (B.1 - A.1) ∧ P.2 - A.2 = (1/2) * (B.2 - A.2)) : 
  P = (-3, -1) := 
sorry

end find_point_P_l594_59481


namespace find_f_expression_find_f_range_l594_59454

noncomputable def y (t x : ℝ) : ℝ := 1 - 2 * t - 2 * t * x + 2 * x ^ 2

noncomputable def f (t : ℝ) : ℝ := 
  if t < -2 then 3 
  else if t > 2 then -4 * t + 3 
  else -t ^ 2 / 2 - 2 * t + 1

theorem find_f_expression (t : ℝ) : 
  f t = if t < -2 then 3 else 
          if t > 2 then -4 * t + 3 
          else - t ^ 2 / 2 - 2 * t + 1 :=
sorry

theorem find_f_range (t : ℝ) (ht : -2 ≤ t ∧ t ≤ 0) : 
  1 ≤ f t ∧ f t ≤ 3 := 
sorry

end find_f_expression_find_f_range_l594_59454


namespace dice_probability_l594_59446

theorem dice_probability :
  let one_digit_prob := 9 / 20
  let two_digit_prob := 11 / 20
  let number_of_dice := 5
  ∃ p : ℚ,
    (number_of_dice.choose 2) * (one_digit_prob ^ 2) * (two_digit_prob ^ 3) = p ∧
    p = 107811 / 320000 :=
by
  sorry

end dice_probability_l594_59446


namespace relationship_not_true_l594_59464

theorem relationship_not_true (a b : ℕ) :
  (b = a + 5 ∨ b = a + 15 ∨ b = a + 29) → ¬(a = b - 9) :=
by
  sorry

end relationship_not_true_l594_59464


namespace deans_height_l594_59420

theorem deans_height
  (D : ℕ) 
  (h1 : 10 * D = D + 81) : 
  D = 9 := sorry

end deans_height_l594_59420


namespace era_slices_burger_l594_59435

theorem era_slices_burger (slices_per_burger : ℕ) (h : 5 * slices_per_burger = 10) : slices_per_burger = 2 :=
by 
  sorry

end era_slices_burger_l594_59435


namespace opposite_neg_two_l594_59471

def opposite (x : Int) : Int := -x

theorem opposite_neg_two : opposite (-2) = 2 := by
  sorry

end opposite_neg_two_l594_59471


namespace find_rs_l594_59404

theorem find_rs :
  ∃ r s : ℝ, ∀ x : ℝ, 8 * x^4 - 4 * x^3 - 42 * x^2 + 45 * x - 10 = 8 * (x - r) ^ 2 * (x - s) * (x - 1) :=
sorry

end find_rs_l594_59404


namespace cattle_transport_problem_l594_59403

noncomputable def truck_capacity 
    (total_cattle : ℕ)
    (distance_one_way : ℕ)
    (speed : ℕ)
    (total_time : ℕ) : ℕ :=
  total_cattle / (total_time / ((distance_one_way * 2) / speed))

theorem cattle_transport_problem :
  truck_capacity 400 60 60 40 = 20 := by
  -- The theorem statement follows the structure from the conditions and question
  sorry

end cattle_transport_problem_l594_59403


namespace smallest_c_for_defined_expression_l594_59448

theorem smallest_c_for_defined_expression :
  ∃ (c : ℤ), (∀ x : ℝ, x^2 + (c : ℝ) * x + 15 ≠ 0) ∧
             (∀ k : ℤ, (∀ x : ℝ, x^2 + (k : ℝ) * x + 15 ≠ 0) → c ≤ k) ∧
             c = -7 :=
by 
  sorry

end smallest_c_for_defined_expression_l594_59448


namespace simplify_and_evaluate_l594_59407

noncomputable def a := 2 * Real.sqrt 3 + 3
noncomputable def expr := (1 - 1 / (a - 2)) / ((a ^ 2 - 6 * a + 9) / (2 * a - 4))

theorem simplify_and_evaluate : expr = Real.sqrt 3 / 3 := by
  sorry

end simplify_and_evaluate_l594_59407


namespace expression_value_l594_59406

theorem expression_value : (19 + 12) ^ 2 - (12 ^ 2 + 19 ^ 2) = 456 := 
by sorry

end expression_value_l594_59406


namespace smallest_next_divisor_of_m_l594_59473

theorem smallest_next_divisor_of_m (m : ℕ) (h1 : m % 2 = 0) (h2 : 10000 ≤ m ∧ m < 100000) (h3 : 523 ∣ m) : 
  ∃ d : ℕ, 523 < d ∧ d ∣ m ∧ ∀ e : ℕ, 523 < e ∧ e ∣ m → d ≤ e :=
by
  sorry

end smallest_next_divisor_of_m_l594_59473


namespace square_completion_l594_59438

theorem square_completion (a : ℝ) (h : a^2 + 2 * a - 2 = 0) : (a + 1)^2 = 3 := 
by 
  sorry

end square_completion_l594_59438


namespace total_ingredients_used_l594_59491

theorem total_ingredients_used (water oliveOil salt : ℕ) 
  (h_ratio : water / oliveOil = 3 / 2) 
  (h_salt : water / salt = 3 / 1)
  (h_water_cups : water = 15) : 
  water + oliveOil + salt = 30 :=
sorry

end total_ingredients_used_l594_59491


namespace root_exists_in_interval_l594_59445

def f (x : ℝ) : ℝ := 2 * x + x - 2

theorem root_exists_in_interval :
  (∃ x ∈ (Set.Ioo 0 1), f x = 0) :=
by
  sorry

end root_exists_in_interval_l594_59445


namespace minimum_value_of_f_l594_59483

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem minimum_value_of_f :
  ∃ x : ℝ, f x = -(4 / 3) :=
by
  use 2
  have hf : f 2 = -(4 / 3) := by
    sorry
  exact hf

end minimum_value_of_f_l594_59483


namespace statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l594_59415

-- Statement A
theorem statement_A_incorrect (a b c d : ℝ) (ha : a < b) (hc : c < d) : ¬ (a * c < b * d) := by
  sorry

-- Statement B
theorem statement_B_correct (a b : ℝ) (ha : -2 < a ∧ a < 3) (hb : 1 < b ∧ b < 2) : -1 < a / b ∧ a / b < 3 := by
  sorry

-- Statement C
theorem statement_C_incorrect (m : ℝ) : ¬ (∀ x > 0, x / 2 + 2 / x ≥ m) ∧ (m ≤ 1) := by
  sorry

-- Statement D
theorem statement_D_incorrect : ∃ x : ℝ, (x^2 + 2) + 1 / (x^2 + 2) ≠ 2 := by
  sorry

end statement_A_incorrect_statement_B_correct_statement_C_incorrect_statement_D_incorrect_l594_59415


namespace rationalize_denominator_and_product_l594_59498

theorem rationalize_denominator_and_product :
  let A := -11
  let B := -5
  let C := 5
  let expr := (3 + Real.sqrt 5) / (2 - Real.sqrt 5)
  (expr * (2 + Real.sqrt 5) / (2 + Real.sqrt 5) = A + B * Real.sqrt C) ∧ (A * B * C = 275) :=
by
  sorry

end rationalize_denominator_and_product_l594_59498


namespace greater_number_l594_59440

theorem greater_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 12) : a = 26 :=
by
  have h3 : 2 * a = 52 := by linarith
  have h4 : a = 26 := by linarith
  exact h4

end greater_number_l594_59440


namespace find_number_l594_59449

noncomputable def N := 953.87

theorem find_number (h : (0.47 * N - 0.36 * 1412) + 65 = 5) : N = 953.87 := sorry

end find_number_l594_59449


namespace simplify_fraction_l594_59465

theorem simplify_fraction (x : ℚ) : (x + 2) / 4 + (3 - 4 * x) / 3 = (-13 * x + 18) / 12 :=
by sorry

end simplify_fraction_l594_59465


namespace matrix_power_2023_correct_l594_59467

noncomputable def matrix_power_2023 : Matrix (Fin 2) (Fin 2) ℤ :=
  let A := !![1, 0; 2, 1]  -- Define the matrix
  A^2023

theorem matrix_power_2023_correct :
  matrix_power_2023 = !![1, 0; 4046, 1] := by
  sorry

end matrix_power_2023_correct_l594_59467


namespace distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l594_59486

variable (m : ℝ)

-- Part 1: Prove that if the quadratic equation has two distinct real roots, then m < 13/4.
theorem distinct_real_roots_iff_m_lt_13_over_4 (h : (3 * 3 - 4 * (m - 1)) > 0) : m < 13 / 4 := 
by
  sorry

-- Part 2: Prove that if the quadratic equation has two equal real roots, then the root is 3/2.
theorem equal_real_roots_root_eq_3_over_2 (h : (3 * 3 - 4 * (m - 1)) = 0) : m = 13 / 4 ∧ ∀ x, (x^2 + 3 * x + (13/4 - 1) = 0) → x = 3 / 2 :=
by
  sorry

end distinct_real_roots_iff_m_lt_13_over_4_equal_real_roots_root_eq_3_over_2_l594_59486


namespace scientific_notation_of_384000_l594_59455

theorem scientific_notation_of_384000 :
  (384000 : ℝ) = 3.84 * 10^5 :=
by
  sorry

end scientific_notation_of_384000_l594_59455


namespace polynomial_positive_values_l594_59405

noncomputable def P (x y : ℝ) : ℝ := x^2 + (x*y + 1)^2

theorem polynomial_positive_values :
  ∀ (z : ℝ), (∃ (x y : ℝ), P x y = z) ↔ z > 0 :=
by
  sorry

end polynomial_positive_values_l594_59405


namespace count_households_in_apartment_l594_59419

noncomputable def total_households 
  (houses_left : ℕ)
  (houses_right : ℕ)
  (floors_above : ℕ)
  (floors_below : ℕ) 
  (households_per_house : ℕ) : ℕ :=
(houses_left + houses_right) * (floors_above + floors_below) * households_per_house

theorem count_households_in_apartment : 
  ∀ (houses_left houses_right floors_above floors_below households_per_house : ℕ),
  houses_left = 1 →
  houses_right = 6 →
  floors_above = 1 →
  floors_below = 3 →
  households_per_house = 3 →
  total_households houses_left houses_right floors_above floors_below households_per_house = 105 :=
by
  intros houses_left houses_right floors_above floors_below households_per_house hl hr fa fb hh
  rw [hl, hr, fa, fb, hh]
  unfold total_households
  norm_num
  sorry

end count_households_in_apartment_l594_59419


namespace problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l594_59422

noncomputable def f (a x : ℝ) := a^(3 * x + 1)
noncomputable def g (a x : ℝ) := (1 / a)^(5 * x - 2)

variables {a x : ℝ}

theorem problem_1 (h : 0 < a ∧ a < 1) : f a x < 1 ↔ x > -1/3 :=
sorry

theorem problem_2_0_lt_a_lt_1 (h : 0 < a ∧ a < 1) : f a x ≥ g a x ↔ x ≤ 1 / 8 :=
sorry

theorem problem_2_a_gt_1 (h : a > 1) : f a x ≥ g a x ↔ x ≥ 1 / 8 :=
sorry

end problem_1_problem_2_0_lt_a_lt_1_problem_2_a_gt_1_l594_59422


namespace nth_equation_l594_59466

theorem nth_equation (n : ℕ) (h : n > 0) : (1 / n) * ((n^2 + 2 * n) / (n + 1)) - (1 / (n + 1)) = 1 :=
by
  sorry

end nth_equation_l594_59466


namespace trig_expression_value_l594_59409

theorem trig_expression_value
  (x : ℝ)
  (h : Real.tan (x + Real.pi / 4) = -3) :
  (Real.sin x + 2 * Real.cos x) / (3 * Real.sin x + 4 * Real.cos x) = 2 / 5 :=
by
  sorry

end trig_expression_value_l594_59409


namespace flowers_per_basket_l594_59453

-- Definitions derived from the conditions
def initial_flowers : ℕ := 10
def grown_flowers : ℕ := 20
def dead_flowers : ℕ := 10
def baskets : ℕ := 5

-- Theorem stating the equivalence of the problem to its solution
theorem flowers_per_basket :
  (initial_flowers + grown_flowers - dead_flowers) / baskets = 4 :=
by
  sorry

end flowers_per_basket_l594_59453


namespace carolyn_practice_time_l594_59427

theorem carolyn_practice_time :
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  monthly_minutes_total = 1920 :=
by
  let minutes_piano := 20
  let days_per_week := 6
  let weeks_per_month := 4
  let multiplier_violin := 3
  let daily_minutes_piano := minutes_piano
  let daily_minutes_violin := multiplier_violin * minutes_piano
  let daily_minutes_total := daily_minutes_piano + daily_minutes_violin
  let weekly_minutes_total := daily_minutes_total * days_per_week
  let monthly_minutes_total := weekly_minutes_total * weeks_per_month
  sorry

end carolyn_practice_time_l594_59427


namespace standing_arrangements_l594_59412

theorem standing_arrangements : ∃ (arrangements : ℕ), arrangements = 2 :=
by
  -- Given that Jia, Yi, Bing, and Ding are four distinct people standing in a row
  -- We need to prove that there are exactly 2 different ways for them to stand such that Jia is not at the far left and Yi is not at the far right
  sorry

end standing_arrangements_l594_59412


namespace integer_solutions_l594_59479

theorem integer_solutions (n : ℕ) :
  n = 7 ↔ ∃ (x : ℤ), ∀ (x : ℤ), (3 * x^2 + 17 * x + 14 ≤ 20)  :=
by
  sorry

end integer_solutions_l594_59479


namespace concentric_circles_circumference_difference_l594_59494

theorem concentric_circles_circumference_difference :
  ∀ (radius_diff inner_diameter : ℝ),
  radius_diff = 15 →
  inner_diameter = 50 →
  ((π * (inner_diameter + 2 * radius_diff)) - (π * inner_diameter)) = 30 * π :=
by
  sorry

end concentric_circles_circumference_difference_l594_59494


namespace asymptotes_and_eccentricity_of_hyperbola_l594_59497

noncomputable def hyperbola_asymptotes_and_eccentricity : Prop :=
  let a := 1
  let b := Real.sqrt 2
  let c := Real.sqrt 3
  ∀ (x y : ℝ), x^2 - (y^2 / 2) = 1 →
    ((y = 2 * x ∨ y = -2 * x) ∧ Real.sqrt (1 + (b^2 / a^2)) = c)

theorem asymptotes_and_eccentricity_of_hyperbola :
  hyperbola_asymptotes_and_eccentricity :=
by
  sorry

end asymptotes_and_eccentricity_of_hyperbola_l594_59497


namespace number_of_squares_in_figure_100_l594_59447

theorem number_of_squares_in_figure_100 :
  ∃ (a b c : ℤ), (c = 1) ∧ (a + b + c = 7) ∧ (4 * a + 2 * b + c = 19) ∧ (3 * 100^2 + 3 * 100 + 1 = 30301) :=
sorry

end number_of_squares_in_figure_100_l594_59447


namespace age_of_B_l594_59499

variables (A B C : ℕ)

theorem age_of_B (h1 : (A + B + C) / 3 = 25) (h2 : (A + C) / 2 = 29) : B = 17 := 
by
  -- Skipping the proof steps
  sorry

end age_of_B_l594_59499


namespace neg_triangle_obtuse_angle_l594_59470

theorem neg_triangle_obtuse_angle : 
  (¬ ∀ (A B C : ℝ), A + B + C = π → max (max A B) C < π/2) ↔ (∃ (A B C : ℝ), A + B + C = π ∧ min (min A B) C > π/2) :=
by
  sorry

end neg_triangle_obtuse_angle_l594_59470


namespace problem_equivalence_of_angles_l594_59444

noncomputable def ctg (x : ℝ) : ℝ := 1 / (Real.tan x)

theorem problem_equivalence_of_angles
  (a b c t S ω : ℝ)
  (hS : S = Real.sqrt ((a^2 + b^2 + c^2)^2 + (4 * t)^2))
  (h1 : ctg ω = (a^2 + b^2 + c^2) / (4 * t))
  (h2 : Real.cos ω = (a^2 + b^2 + c^2) / S)
  (h3 : Real.sin ω = (4 * t) / S) :
  True :=
sorry

end problem_equivalence_of_angles_l594_59444


namespace sin_16_over_3_pi_l594_59480

theorem sin_16_over_3_pi : Real.sin (16 / 3 * Real.pi) = -Real.sqrt 3 / 2 := 
sorry

end sin_16_over_3_pi_l594_59480


namespace value_of_x_l594_59463

theorem value_of_x (x : ℕ) (h : x + (10 * x + x) = 12) : x = 1 := by
  sorry

end value_of_x_l594_59463


namespace measure_4_minutes_with_hourglasses_l594_59429

/-- Prove that it is possible to measure exactly 4 minutes using hourglasses of 9 minutes and 7 minutes and the minimum total time required is 18 minutes -/
theorem measure_4_minutes_with_hourglasses : 
  ∃ (a b : ℕ), (9 * a - 7 * b = 4) ∧ (a + b) * 1 ≤ 2 ∧ (a * 9 ≤ 18 ∧ b * 7 <= 18) :=
by {
  sorry
}

end measure_4_minutes_with_hourglasses_l594_59429


namespace number_of_factors_multiples_of_360_l594_59476

def n : ℕ := 2^10 * 3^14 * 5^8

theorem number_of_factors_multiples_of_360 (n : ℕ) (hn : n = 2^10 * 3^14 * 5^8) : 
  ∃ (k : ℕ), k = 832 ∧ 
  (∀ m : ℕ, m ∣ n → 360 ∣ m → k = 8 * 13 * 8) := 
sorry

end number_of_factors_multiples_of_360_l594_59476


namespace width_of_rectangle_l594_59439

-- Define the side length of the square and the length of the rectangle.
def side_length_square : ℝ := 12
def length_rectangle : ℝ := 18

-- Calculate the perimeter of the square.
def perimeter_square : ℝ := 4 * side_length_square

-- This definition represents the perimeter of the rectangle made from the same wire.
def perimeter_rectangle : ℝ := perimeter_square

-- Show that the width of the rectangle is 6 cm.
theorem width_of_rectangle : ∃ W : ℝ, 2 * (length_rectangle + W) = perimeter_rectangle ∧ W = 6 :=
by
  use 6
  simp [length_rectangle, perimeter_rectangle, side_length_square]
  norm_num
  sorry

end width_of_rectangle_l594_59439


namespace calculate_shot_cost_l594_59441

theorem calculate_shot_cost :
  let num_pregnant_dogs := 3
  let puppies_per_dog := 4
  let shots_per_puppy := 2
  let cost_per_shot := 5
  let total_puppies := num_pregnant_dogs * puppies_per_dog
  let total_shots := total_puppies * shots_per_puppy
  let total_cost := total_shots * cost_per_shot
  total_cost = 120 :=
by
  sorry

end calculate_shot_cost_l594_59441


namespace sandy_savings_percentage_l594_59428

theorem sandy_savings_percentage
  (S : ℝ) -- Sandy's salary last year
  (H1 : 0.10 * S = saved_last_year) -- Last year, Sandy saved 10% of her salary.
  (H2 : 1.10 * S = salary_this_year) -- This year, Sandy made 10% more than last year.
  (H3 : 0.15 * salary_this_year = saved_this_year) -- This year, Sandy saved 15% of her salary.
  : (saved_this_year / saved_last_year) * 100 = 165 := 
by 
  sorry

end sandy_savings_percentage_l594_59428


namespace meiosis_fertilization_stability_l594_59436

def maintains_chromosome_stability (x : String) : Prop :=
  x = "Meiosis and Fertilization"

theorem meiosis_fertilization_stability :
  maintains_chromosome_stability "Meiosis and Fertilization" :=
by
  sorry

end meiosis_fertilization_stability_l594_59436


namespace greatest_c_for_expression_domain_all_real_l594_59485

theorem greatest_c_for_expression_domain_all_real :
  ∃ c : ℤ, c ≤ 7 ∧ c ^ 2 < 60 ∧ ∀ d : ℤ, d > 7 → ¬ (d ^ 2 < 60) := sorry

end greatest_c_for_expression_domain_all_real_l594_59485


namespace f_sub_f_neg_l594_59458

-- Define the function f
def f (x : ℝ) : ℝ := x^5 + x^3 + 7 * x

-- State the theorem
theorem f_sub_f_neg : f 3 - f (-3) = 582 :=
by
  -- Definitions and calculations for the proof
  -- (You can complete this part in later proof development)
  sorry

end f_sub_f_neg_l594_59458


namespace product_of_three_greater_than_two_or_four_of_others_l594_59462

theorem product_of_three_greater_than_two_or_four_of_others 
  (x : Fin 10 → ℕ) 
  (h_unique : ∀ i j : Fin 10, i ≠ j → x i ≠ x j) 
  (h_positive : ∀ i : Fin 10, 0 < x i) : 
  ∃ i j k : Fin 10, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ 
    (∀ a b : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ b ≠ i ∧ b ≠ j ∧ b ≠ k → 
      x i * x j * x k > x a * x b) ∨ 
    (∀ a b c d : Fin 10, a ≠ i ∧ a ≠ j ∧ a ≠ k ∧ 
      b ≠ i ∧ b ≠ j ∧ b ≠ k ∧ 
      c ≠ i ∧ c ≠ j ∧ c ≠ k ∧ 
      d ≠ i ∧ d ≠ j ∧ d ≠ k → 
      x i * x j * x k > x a * x b * x c * x d) := sorry

end product_of_three_greater_than_two_or_four_of_others_l594_59462


namespace max_a4_l594_59402

theorem max_a4 (a1 d a4 : ℝ) 
  (h1 : 2 * a1 + 3 * d ≥ 5) 
  (h2 : a1 + 2 * d ≤ 3) 
  (ha4 : a4 = a1 + 3 * d) : 
  a4 ≤ 4 := 
by 
  sorry

end max_a4_l594_59402


namespace find_S5_l594_59484

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

axiom arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : ∀ n : ℕ, a (n + 1) = a 1 + n * d
axiom sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_S5 (h : a 1 + a 3 + a 5 = 3) : S 5 = 5 :=
by
  sorry

end find_S5_l594_59484


namespace union_of_A_and_B_l594_59489

noncomputable def A : Set ℝ := {x : ℝ | x^2 - 2 * x < 0}
noncomputable def B : Set ℝ := {x : ℝ | 1 < x }

theorem union_of_A_and_B : A ∪ B = {x : ℝ | 0 < x} :=
by
  sorry

end union_of_A_and_B_l594_59489


namespace sum_of_number_and_reverse_l594_59431

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) : (10 * a + b) + (10 * b + a) = 99 := by
  sorry

end sum_of_number_and_reverse_l594_59431


namespace cooling_constant_l594_59475

theorem cooling_constant (θ0 θ1 θ t k : ℝ) (h1 : θ1 = 60) (h0 : θ0 = 15) (ht : t = 3) (hθ : θ = 42)
  (h_temp_formula : θ = θ0 + (θ1 - θ0) * Real.exp (-k * t)) :
  k = 0.17 :=
by sorry

end cooling_constant_l594_59475


namespace value_of_a_plus_b_l594_59430

theorem value_of_a_plus_b (a b : ℝ) : (|a - 1| + (b + 3)^2 = 0) → (a + b = -2) :=
by
  sorry

end value_of_a_plus_b_l594_59430


namespace total_cost_of_new_movie_l594_59487

noncomputable def previous_movie_length_hours : ℕ := 2
noncomputable def new_movie_length_increase_percent : ℕ := 60
noncomputable def previous_movie_cost_per_minute : ℕ := 50
noncomputable def new_movie_cost_per_minute_factor : ℕ := 2 

theorem total_cost_of_new_movie : 
  let new_movie_length_hours := previous_movie_length_hours + (previous_movie_length_hours * new_movie_length_increase_percent / 100)
  let new_movie_length_minutes := new_movie_length_hours * 60
  let new_movie_cost_per_minute := previous_movie_cost_per_minute * new_movie_cost_per_minute_factor
  let total_cost := new_movie_length_minutes * new_movie_cost_per_minute
  total_cost = 19200 := 
by
  sorry

end total_cost_of_new_movie_l594_59487


namespace fraction_equality_l594_59426

theorem fraction_equality :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 3) = 59 / 61 :=
by
  sorry

end fraction_equality_l594_59426


namespace price_of_two_identical_filters_l594_59421

def price_of_individual_filters (x : ℝ) : Prop :=
  let total_individual := 2 * 14.05 + 19.50 + 2 * x
  total_individual = 87.50 / 0.92

theorem price_of_two_identical_filters
  (h1 : price_of_individual_filters 23.76) :
  23.76 * 2 + 28.10 + 19.50 = 87.50 / 0.92 :=
by sorry

end price_of_two_identical_filters_l594_59421


namespace percentage_loss_is_correct_l594_59413

-- Define the cost price and selling price
def cost_price : ℕ := 2000
def selling_price : ℕ := 1800

-- Define the calculation of loss and percentage loss
def loss (cp sp : ℕ) := cp - sp
def percentage_loss (loss cp : ℕ) := (loss * 100) / cp

-- The goal is to prove that the percentage loss is 10%
theorem percentage_loss_is_correct : percentage_loss (loss cost_price selling_price) cost_price = 10 := by
  sorry

end percentage_loss_is_correct_l594_59413


namespace train_pass_time_l594_59424

noncomputable def train_length : ℕ := 360
noncomputable def platform_length : ℕ := 140
noncomputable def train_speed_kmh : ℕ := 45

noncomputable def convert_speed_to_mps (speed_kmh : ℕ) : ℚ := 
  (speed_kmh * 1000) / 3600

noncomputable def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

noncomputable def time_to_pass (distance : ℕ) (speed_mps : ℚ) : ℚ :=
  distance / speed_mps

theorem train_pass_time 
  (train_len : ℕ) 
  (platform_len : ℕ) 
  (speed_kmh : ℕ) : 
  time_to_pass (total_distance train_len platform_len) (convert_speed_to_mps speed_kmh) = 40 := 
by 
  sorry

end train_pass_time_l594_59424


namespace max_ratio_of_three_digit_to_sum_l594_59433

theorem max_ratio_of_three_digit_to_sum (a b c : ℕ) 
  (ha : 1 ≤ a ∧ a ≤ 9)
  (hb : 0 ≤ b ∧ b ≤ 9)
  (hc : 0 ≤ c ∧ c ≤ 9) :
  (100 * a + 10 * b + c) / (a + b + c) ≤ 100 :=
by sorry

end max_ratio_of_three_digit_to_sum_l594_59433


namespace ratio_of_ages_l594_59456

theorem ratio_of_ages (age_saras age_kul : ℕ) (h_saras : age_saras = 33) (h_kul : age_kul = 22) : 
  age_saras / Nat.gcd age_saras age_kul = 3 ∧ age_kul / Nat.gcd age_saras age_kul = 2 :=
by
  sorry

end ratio_of_ages_l594_59456


namespace andrew_age_l594_59417

variable (a g s : ℝ)

theorem andrew_age :
  g = 10 * a ∧ g - s = a + 45 ∧ s = 5 → a = 50 / 9 := by
  sorry

end andrew_age_l594_59417


namespace arithmetic_sequence_sum_l594_59423

theorem arithmetic_sequence_sum :
  ∃ x y d : ℕ,
    d = 6
    ∧ x = 3 + d * (3 - 1)
    ∧ y = x + d
    ∧ y + d = 39
    ∧ x + y = 60 :=
by
  sorry

end arithmetic_sequence_sum_l594_59423


namespace intersecting_to_quadrilateral_l594_59450

-- Define the geometric solids
inductive GeometricSolid
| cone : GeometricSolid
| sphere : GeometricSolid
| cylinder : GeometricSolid

-- Define a function that checks if intersecting a given solid with a plane can produce a quadrilateral
def can_intersect_to_quadrilateral (solid : GeometricSolid) : Prop :=
  match solid with
  | GeometricSolid.cone => false
  | GeometricSolid.sphere => false
  | GeometricSolid.cylinder => true

-- State the theorem
theorem intersecting_to_quadrilateral (solid : GeometricSolid) :
  can_intersect_to_quadrilateral solid ↔ solid = GeometricSolid.cylinder :=
sorry

end intersecting_to_quadrilateral_l594_59450


namespace cyclists_meet_at_start_point_l594_59442

-- Conditions from the problem
def cyclist1_speed : ℝ := 7 -- speed of the first cyclist in m/s
def cyclist2_speed : ℝ := 8 -- speed of the second cyclist in m/s
def circumference : ℝ := 600 -- circumference of the circular track in meters

-- Relative speed when cyclists move in opposite directions
def relative_speed := cyclist1_speed + cyclist2_speed

-- Prove that they meet at the starting point after 40 seconds
theorem cyclists_meet_at_start_point :
  (circumference / relative_speed) = 40 := by
  -- the proof would go here
  sorry

end cyclists_meet_at_start_point_l594_59442


namespace distance_covered_l594_59469

-- Define the rate and time as constants
def rate : ℝ := 4 -- 4 miles per hour
def time : ℝ := 2 -- 2 hours

-- Theorem statement: Verify the distance covered
theorem distance_covered : rate * time = 8 := 
by
  sorry

end distance_covered_l594_59469


namespace sufficiency_not_necessity_l594_59457

theorem sufficiency_not_necessity (x y : ℝ) :
  (x > 3 ∧ y > 3) → (x + y > 6 ∧ x * y > 9) ∧ (¬ (x + y > 6 ∧ x * y > 9 → x > 3 ∧ y > 3)) :=
by
  sorry

end sufficiency_not_necessity_l594_59457


namespace cost_price_equal_l594_59478

theorem cost_price_equal (total_selling_price : ℝ) (profit_percent_first profit_percent_second : ℝ) (length_first_segment length_second_segment : ℝ) (C : ℝ) :
  total_selling_price = length_first_segment * (1 + profit_percent_first / 100) * C + length_second_segment * (1 + profit_percent_second / 100) * C →
  C = 15360 / (66 + 72) :=
by {
  sorry
}

end cost_price_equal_l594_59478


namespace boy_age_is_10_l594_59411

-- Define the boy's current age as a variable
def boy_current_age := 10

-- Define a condition based on the boy's statement
def boy_statement_condition (x : ℕ) : Prop :=
  x = 2 * (x - 5)

-- The main theorem stating equivalence of the boy's current age to 10 given the condition
theorem boy_age_is_10 (x : ℕ) (h : boy_statement_condition x) : x = boy_current_age := by
  sorry

end boy_age_is_10_l594_59411


namespace lifespan_of_bat_l594_59400

theorem lifespan_of_bat (B : ℕ) (h₁ : ∀ B, B - 6 < B)
    (h₂ : ∀ B, 4 * (B - 6) < 4 * B)
    (h₃ : B + (B - 6) + 4 * (B - 6) = 30) :
    B = 10 := by
  sorry

end lifespan_of_bat_l594_59400


namespace solution_set_of_xf_x_gt_0_l594_59410

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, f (-x) = - f x
axiom h2 : f 2 = 0
axiom h3 : ∀ x : ℝ, 0 < x → x * (deriv f x) + f x < 0

theorem solution_set_of_xf_x_gt_0 :
  {x : ℝ | x * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
by {
  sorry
}

end solution_set_of_xf_x_gt_0_l594_59410


namespace taxi_fare_miles_l594_59459

theorem taxi_fare_miles (total_spent : ℝ) (tip : ℝ) (base_fare : ℝ) (additional_fare_rate : ℝ) (base_mile : ℝ) (additional_mile_unit : ℝ) (x : ℝ) :
  (total_spent = 15) →
  (tip = 3) →
  (base_fare = 3) →
  (additional_fare_rate = 0.25) →
  (base_mile = 0.5) →
  (additional_mile_unit = 0.1) →
  (x = base_mile + (total_spent - tip - base_fare) / (additional_fare_rate / additional_mile_unit)) →
  x = 4.1 :=
by
  intros
  sorry

end taxi_fare_miles_l594_59459


namespace dwayneA_students_l594_59490

-- Define the number of students who received an 'A' in Mrs. Carter's class
def mrsCarterA := 8
-- Define the total number of students in Mrs. Carter's class
def mrsCarterTotal := 20
-- Define the total number of students in Mr. Dwayne's class
def mrDwayneTotal := 30
-- Calculate the ratio of students who received an 'A' in Mrs. Carter's class
def carterRatio := mrsCarterA / mrsCarterTotal
-- Calculate the number of students who received an 'A' in Mr. Dwayne's class based on the same ratio
def mrDwayneA := (carterRatio * mrDwayneTotal)

-- Prove that the number of students who received an 'A' in Mr. Dwayne's class is 12
theorem dwayneA_students :
  mrDwayneA = 12 := 
by
  -- Since def calculation does not automatically prove equality, we will need to use sorry to skip the proof for now.
  sorry

end dwayneA_students_l594_59490


namespace living_room_area_l594_59418

theorem living_room_area (L W : ℝ) (percent_covered : ℝ) (expected_area : ℝ) 
  (hL : L = 6.5) (hW : W = 12) (hpercent : percent_covered = 0.85) 
  (hexpected_area : expected_area = 91.76) : 
  (L * W / percent_covered = expected_area) :=
by
  sorry  -- The proof is omitted.

end living_room_area_l594_59418


namespace fibonacci_polynomial_property_l594_59461

-- Define the Fibonacci sequence
def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n+2) => fibonacci n + fibonacci (n+1)

-- Define the polynomial P(x) of degree 990
noncomputable def P : ℕ → ℕ :=
  sorry  -- To be defined as a polynomial with specified properties

-- Statement of the problem (theorem)
theorem fibonacci_polynomial_property (P : ℕ → ℕ) (hP : ∀ k, 992 ≤ k → k ≤ 1982 → P k = fibonacci k) :
  P 1983 = fibonacci 1983 - 1 :=
sorry  -- Proof omitted

end fibonacci_polynomial_property_l594_59461


namespace henry_money_l594_59460

-- Define the conditions
def initial : ℕ := 11
def birthday : ℕ := 18
def spent : ℕ := 10

-- Define the final amount
def final_amount : ℕ := initial + birthday - spent

-- State the theorem
theorem henry_money : final_amount = 19 := by
  -- Skipping the proof
  sorry

end henry_money_l594_59460


namespace gcd_expression_infinite_composite_pairs_exists_l594_59452

-- Part (a)
theorem gcd_expression (n : ℕ) (a : ℕ) (b : ℕ) (hn : n > 0) (ha : a > 0) (hb : b > 0) :
  Nat.gcd (n^a + 1) (n^b + 1) ≤ n^(Nat.gcd a b) + 1 :=
by
  sorry

-- Part (b)
theorem infinite_composite_pairs_exists (n : ℕ) (hn : n > 0) :
  ∃ (pairs : ℕ × ℕ → Prop), (∀ a b, pairs (a, b) → a > 1 ∧ b > 1 ∧ ∃ d, d > 1 ∧ a = d ∧ b = dn) ∧
  (∀ a b, pairs (a, b) → Nat.gcd (n^a + 1) (n^b + 1) = n^(Nat.gcd a b) + 1) ∧
  (∀ x y, x > 1 → y > 1 → x ∣ y ∨ y ∣ x → ¬pairs (x, y)) :=
by
  sorry

end gcd_expression_infinite_composite_pairs_exists_l594_59452
