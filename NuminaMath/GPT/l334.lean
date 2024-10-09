import Mathlib

namespace max_nine_multiple_l334_33430

theorem max_nine_multiple {a b c n : ℕ} (h1 : Prime a) (h2 : Prime b) (h3 : Prime c) (h4 : 3 < a) (h5 : 3 < b) (h6 : 3 < c) (h7 : 2 * a + 5 * b = c) : 9 ∣ (a + b + c) :=
sorry

end max_nine_multiple_l334_33430


namespace initial_ducks_count_l334_33490

theorem initial_ducks_count (D : ℕ) 
  (h1 : ∃ (G : ℕ), G = 2 * D - 10) 
  (h2 : ∃ (D_new : ℕ), D_new = D + 4) 
  (h3 : ∃ (G_new : ℕ), G_new = 2 * D - 20) 
  (h4 : ∀ (D_new G_new : ℕ), G_new = D_new + 1) : 
  D = 25 := by
  sorry

end initial_ducks_count_l334_33490


namespace find_n_value_l334_33412

theorem find_n_value (n : ℕ) (h : 2^6 * 3^3 * n = Nat.factorial 9) : n = 210 := sorry

end find_n_value_l334_33412


namespace harrison_grade_levels_l334_33405

theorem harrison_grade_levels
  (total_students : ℕ)
  (percent_moving : ℚ)
  (advanced_class_size : ℕ)
  (num_normal_classes : ℕ)
  (normal_class_size : ℕ)
  (students_moving : ℕ)
  (students_per_grade_level : ℕ)
  (grade_levels : ℕ) :
  total_students = 1590 →
  percent_moving = 40 / 100 →
  advanced_class_size = 20 →
  num_normal_classes = 6 →
  normal_class_size = 32 →
  students_moving = total_students * percent_moving →
  students_per_grade_level = advanced_class_size + num_normal_classes * normal_class_size →
  grade_levels = students_moving / students_per_grade_level →
  grade_levels = 3 :=
by
  intros
  sorry

end harrison_grade_levels_l334_33405


namespace find_m_l334_33499

theorem find_m (m : ℝ) : (∀ x : ℝ, 0 < x → x < 2 → - (1/2)*x^2 + 2*x > -m*x) ↔ m = -1 := 
sorry

end find_m_l334_33499


namespace sin_C_value_l334_33423

theorem sin_C_value (A B C : Real) (AC BC : Real) (h_AC : AC = 3) (h_BC : BC = 2 * Real.sqrt 3) (h_A : A = 2 * B) :
    let C : Real := Real.pi - A - B
    Real.sin C = Real.sqrt 6 / 9 :=
  sorry

end sin_C_value_l334_33423


namespace triangle_angle_contradiction_l334_33457

theorem triangle_angle_contradiction (A B C : ℝ) (hA : 60 < A) (hB : 60 < B) (hC : 60 < C) (h_sum : A + B + C = 180) : false :=
by {
  -- This would be the proof part, which we don't need to detail according to the instructions.
  sorry
}

end triangle_angle_contradiction_l334_33457


namespace find_ABC_l334_33486

noncomputable def problem (A B C : ℕ) : Prop :=
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
  A < 8 ∧ B < 8 ∧ C < 6 ∧
  (A * 8 + B + C = 8 * 2 + C) ∧
  (A * 8 + B + B * 8 + A = C * 8 + C) ∧
  (100 * A + 10 * B + C = 246)

theorem find_ABC : ∃ A B C : ℕ, problem A B C := sorry

end find_ABC_l334_33486


namespace fraction_complex_z_l334_33428

theorem fraction_complex_z (z : ℂ) (hz : z = 1 - I) : 2 / z = 1 + I := by
    sorry

end fraction_complex_z_l334_33428


namespace sum_of_digits_l334_33482

theorem sum_of_digits (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                      (h_range : 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9)
                      (h_product : a * b * c * d = 810) :
  a + b + c + d = 23 := sorry

end sum_of_digits_l334_33482


namespace smallest_n_in_range_l334_33410

theorem smallest_n_in_range : ∃ n : ℕ, n > 1 ∧ (n % 3 = 2) ∧ (n % 7 = 2) ∧ (n % 8 = 2) ∧ 120 ≤ n ∧ n ≤ 149 :=
by
  sorry

end smallest_n_in_range_l334_33410


namespace range_x_sub_cos_y_l334_33444

theorem range_x_sub_cos_y (x y : ℝ) (h : x^2 + 2 * Real.cos y = 1) : 
  -1 ≤ x - Real.cos y ∧ x - Real.cos y ≤ Real.sqrt 3 + 1 :=
sorry

end range_x_sub_cos_y_l334_33444


namespace total_feed_per_week_l334_33492

-- Define the conditions
def daily_feed_per_pig : ℕ := 10
def number_of_pigs : ℕ := 2
def days_per_week : ℕ := 7

-- Theorem statement
theorem total_feed_per_week : daily_feed_per_pig * number_of_pigs * days_per_week = 140 := 
  sorry

end total_feed_per_week_l334_33492


namespace city_renumbering_not_possible_l334_33437

-- Defining the problem conditions
def city_renumbering_invalid (city_graph : Type) (connected : city_graph → city_graph → Prop) : Prop :=
  ∃ (M N : city_graph), ∀ (renumber : city_graph → city_graph),
  (renumber M = N ∧ renumber N = M) → ¬(
    ∀ x y : city_graph,
    connected x y ↔ connected (renumber x) (renumber y)
  )

-- Statement of the problem
theorem city_renumbering_not_possible (city_graph : Type) (connected : city_graph → city_graph → Prop) :
  city_renumbering_invalid city_graph connected :=
sorry

end city_renumbering_not_possible_l334_33437


namespace decrease_in_profit_due_to_looms_breakdown_l334_33442

theorem decrease_in_profit_due_to_looms_breakdown :
  let num_looms := 70
  let month_days := 30
  let total_sales := 1000000
  let total_expenses := 150000
  let daily_sales_per_loom := total_sales / (num_looms * month_days)
  let daily_expenses_per_loom := total_expenses / (num_looms * month_days)
  let loom1_days := 10
  let loom2_days := 5
  let loom3_days := 15
  let loom_repair_cost := 2000
  let loom1_loss := daily_sales_per_loom * loom1_days
  let loom2_loss := daily_sales_per_loom * loom2_days
  let loom3_loss := daily_sales_per_loom * loom3_days
  let total_loss_sales := loom1_loss + loom2_loss + loom3_loss
  let total_repair_cost := loom_repair_cost * 3
  let decrease_in_profit := total_loss_sales + total_repair_cost
  decrease_in_profit = 20285.70 := by
  sorry

end decrease_in_profit_due_to_looms_breakdown_l334_33442


namespace keith_apples_correct_l334_33440

def mike_apples : ℕ := 7
def nancy_apples : ℕ := 3
def total_apples : ℕ := 16
def keith_apples : ℕ := total_apples - (mike_apples + nancy_apples)

theorem keith_apples_correct : keith_apples = 6 := by
  -- the actual proof would go here
  sorry

end keith_apples_correct_l334_33440


namespace jillian_largest_apartment_l334_33467

noncomputable def largest_apartment_size (budget : ℝ) (rate : ℝ) : ℝ :=
  budget / rate

theorem jillian_largest_apartment : largest_apartment_size 720 1.20 = 600 := by
  sorry

end jillian_largest_apartment_l334_33467


namespace interest_rate_is_five_percent_l334_33461

-- Define the problem parameters
def principal : ℝ := 1200
def amount_after_period : ℝ := 1344
def time_period : ℝ := 2.4

-- Define the simple interest formula
def interest (P R T : ℝ) : ℝ := P * R * T

-- The goal is to prove that the rate of interest is 5% per year
theorem interest_rate_is_five_percent :
  ∃ R, interest principal R time_period = amount_after_period - principal ∧ R = 0.05 :=
by
  sorry

end interest_rate_is_five_percent_l334_33461


namespace max_median_value_l334_33447

theorem max_median_value (x : ℕ) (h : 198 + x ≤ 392) : x ≤ 194 :=
by {
  sorry
}

end max_median_value_l334_33447


namespace MaryAddedCandy_l334_33411

-- Definitions based on the conditions
def MaryInitialCandyCount (MeganCandyCount : ℕ) : ℕ :=
  3 * MeganCandyCount

-- Given conditions
def MeganCandyCount : ℕ := 5
def MaryTotalCandyCount : ℕ := 25

-- Proof statement
theorem MaryAddedCandy : 
  let MaryInitialCandy := MaryInitialCandyCount MeganCandyCount
  MaryTotalCandyCount - MaryInitialCandy = 10 :=
by 
  sorry

end MaryAddedCandy_l334_33411


namespace problem_lean_l334_33483

theorem problem_lean (x y : ℝ) (h₁ : (|x + 2| ≥ 0) ∧ (|y - 4| ≥ 0)) : 
  (|x + 2| = 0 ∧ |y - 4| = 0) → x + y - 3 = -1 :=
by sorry

end problem_lean_l334_33483


namespace inequality_proof_equality_case_l334_33414

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) ≥ c / b - (c^2) / a) :=
sorry

theorem equality_case (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) : 
  (a^2 / (b^3 * c) - a / (b^2) = c / b - c^2 / a) ↔ (a = b * c) :=
sorry

end inequality_proof_equality_case_l334_33414


namespace cube_volume_increase_l334_33481

variable (a : ℝ) (h : a ≥ 0)

theorem cube_volume_increase :
  ((2 * a) ^ 3) = 8 * (a ^ 3) :=
by sorry

end cube_volume_increase_l334_33481


namespace polynomial_evaluation_l334_33498

theorem polynomial_evaluation (x : ℝ) (h₁ : 0 < x) (h₂ : x^2 - 2 * x - 15 = 0) :
  x^3 - 2 * x^2 - 8 * x + 16 = 51 :=
sorry

end polynomial_evaluation_l334_33498


namespace irreducible_fraction_l334_33429

theorem irreducible_fraction (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l334_33429


namespace shorter_side_length_l334_33400

theorem shorter_side_length 
  (L W : ℝ) 
  (h1 : L * W = 117) 
  (h2 : 2 * L + 2 * W = 44) :
  L = 9 ∨ W = 9 :=
by
  sorry

end shorter_side_length_l334_33400


namespace two_p_plus_q_l334_33471

variable {p q : ℚ}  -- Variables are rationals

theorem two_p_plus_q (h : p / q = 5 / 4) : 2 * p + q = 7 * q / 2 :=
by sorry

end two_p_plus_q_l334_33471


namespace range_of_m_l334_33445

-- Definitions and conditions
def ellipse (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

def is_eccentricity (e a b : ℝ) : Prop :=
  e = Real.sqrt (1 - (b^2 / a^2))

def is_semi_latus_rectum (d a b : ℝ) : Prop :=
  d = 2 * b^2 / a

-- Main theorem statement
theorem range_of_m (a b m : ℝ) (x y : ℝ) 
  (h1 : a > b) 
  (h2 : b > 0)
  (h3 : is_eccentricity (Real.sqrt (3) / 2) a b)
  (h4 : is_semi_latus_rectum 1 a b)
  (h_ellipse : ellipse a b x y) : 
  m ∈ Set.Ioo (-3 / 2 : ℝ) (3 / 2 : ℝ) := 
sorry

end range_of_m_l334_33445


namespace problem_A_problem_B_problem_C_problem_D_l334_33422

theorem problem_A (a b: ℝ) (h : b > 0 ∧ a > b) : ¬(1/a > 1/b) := 
by {
  sorry
}

theorem problem_B (a b: ℝ) (h : a < b ∧ b < 0): (a^2 > a*b) := 
by {
  sorry
}

theorem problem_C (a b: ℝ) (h : a > b): ¬(|a| > |b|) := 
by {
  sorry
}

theorem problem_D (a: ℝ) (h : a > 2): (a + 4/(a-2) ≥ 6) := 
by {
  sorry
}

end problem_A_problem_B_problem_C_problem_D_l334_33422


namespace area_regular_octagon_l334_33495

theorem area_regular_octagon (AB BC: ℝ) (hAB: AB = 2) (hBC: BC = 2) :
  let side_length := 2 * Real.sqrt 2
  let triangle_area := (AB * AB) / 2
  let total_triangle_area := 4 * triangle_area
  let side_length_rect := 4 + 2 * Real.sqrt 2
  let rect_area := side_length_rect * side_length_rect
  let octagon_area := rect_area - total_triangle_area
  octagon_area = 16 + 8 * Real.sqrt 2 :=
by sorry

end area_regular_octagon_l334_33495


namespace abs_condition_l334_33409

theorem abs_condition (x : ℝ) : |2 * x - 7| ≤ 0 ↔ x = 7 / 2 := 
by
  sorry

end abs_condition_l334_33409


namespace total_investment_amount_l334_33485

theorem total_investment_amount 
    (x : ℝ) 
    (h1 : 6258.0 * 0.08 + x * 0.065 = 678.87) : 
    x + 6258.0 = 9000.0 :=
sorry

end total_investment_amount_l334_33485


namespace polygon_area_is_14_l334_33453

def vertices : List (ℕ × ℕ) :=
  [(1, 2), (2, 2), (3, 3), (3, 4), (4, 5), (5, 5), (6, 5), (6, 4), (5, 3),
   (4, 3), (4, 2), (3, 1), (2, 1), (1, 1)]

noncomputable def area_of_polygon (vs : List (ℕ × ℕ)) : ℝ := sorry

theorem polygon_area_is_14 :
  area_of_polygon vertices = 14 := sorry

end polygon_area_is_14_l334_33453


namespace maggie_remaining_goldfish_l334_33401

theorem maggie_remaining_goldfish
  (total_goldfish : ℕ)
  (allowed_fraction : ℕ → ℕ)
  (caught_fraction : ℕ → ℕ)
  (halfsies : ℕ)
  (remaining_goldfish : ℕ)
  (h1 : total_goldfish = 100)
  (h2 : allowed_fraction total_goldfish = total_goldfish / 2)
  (h3 : caught_fraction (allowed_fraction total_goldfish) = (3 * allowed_fraction total_goldfish) / 5)
  (h4 : halfsies = allowed_fraction total_goldfish)
  (h5 : remaining_goldfish = halfsies - caught_fraction halfsies) :
  remaining_goldfish = 20 :=
sorry

end maggie_remaining_goldfish_l334_33401


namespace average_and_variance_of_new_data_set_l334_33449

theorem average_and_variance_of_new_data_set
  (avg : ℝ) (var : ℝ) (constant : ℝ)
  (h_avg : avg = 2.8)
  (h_var : var = 3.6)
  (h_const : constant = 60) :
  (avg + constant = 62.8) ∧ (var = 3.6) :=
sorry

end average_and_variance_of_new_data_set_l334_33449


namespace carl_weight_l334_33413

variable (Al Ben Carl Ed : ℝ)

axiom h1 : Ed = 146
axiom h2 : Ed + 38 = Al
axiom h3 : Al = Ben + 25
axiom h4 : Ben = Carl - 16

theorem carl_weight : Carl = 175 :=
by
  sorry

end carl_weight_l334_33413


namespace transformed_function_equivalence_l334_33403

-- Define the original function
def original_function (x : ℝ) : ℝ := 2 * x + 1

-- Define the transformation involving shifting 2 units to the right
def transformed_function (x : ℝ) : ℝ := original_function (x - 2)

-- The theorem we want to prove
theorem transformed_function_equivalence : 
  ∀ x : ℝ, transformed_function x = 2 * x - 3 :=
by
  sorry

end transformed_function_equivalence_l334_33403


namespace converse_and_inverse_l334_33427

-- Definitions
def is_circle (s : Type) : Prop := sorry
def has_no_corners (s : Type) : Prop := sorry

-- Converse Statement
def converse_false (s : Type) : Prop :=
  has_no_corners s → is_circle s → False

-- Inverse Statement
def inverse_true (s : Type) : Prop :=
  ¬ is_circle s → ¬ has_no_corners s

-- Main Proof Problem
theorem converse_and_inverse (s : Type) :
  (converse_false s) ∧ (inverse_true s) := sorry

end converse_and_inverse_l334_33427


namespace angle_F_calculation_l334_33416

theorem angle_F_calculation (D E F : ℝ) :
  D = 80 ∧ E = 2 * F + 30 ∧ D + E + F = 180 → F = 70 / 3 :=
by
  intro h
  cases' h with hD h_remaining
  cases' h_remaining with hE h_sum
  sorry

end angle_F_calculation_l334_33416


namespace ScarlettsDishCost_l334_33408

theorem ScarlettsDishCost (L P : ℝ) (tip_rate tip_amount : ℝ) (x : ℝ) 
  (hL : L = 10) (hP : P = 17) (htip_rate : tip_rate = 0.10) (htip_amount : tip_amount = 4) 
  (h : tip_rate * (L + P + x) = tip_amount) : x = 13 :=
by
  sorry

end ScarlettsDishCost_l334_33408


namespace jed_correct_speed_l334_33433

def fine_per_mph := 16
def jed_fine := 256
def speed_limit := 50

def jed_speed : Nat := speed_limit + jed_fine / fine_per_mph

theorem jed_correct_speed : jed_speed = 66 := by
  sorry

end jed_correct_speed_l334_33433


namespace compound_interest_l334_33476

theorem compound_interest 
  (P : ℝ) (r : ℝ) (t : ℕ) : P = 500 → r = 0.02 → t = 3 → (P * (1 + r)^t) - P = 30.60 :=
by
  intros P_invest rate years
  simp [P_invest, rate, years]
  sorry

end compound_interest_l334_33476


namespace gcd_a_b_eq_one_l334_33452

def a : ℕ := 130^2 + 240^2 + 350^2
def b : ℕ := 131^2 + 241^2 + 349^2

theorem gcd_a_b_eq_one : Nat.gcd a b = 1 := by
  sorry

end gcd_a_b_eq_one_l334_33452


namespace total_distance_is_correct_l334_33424

noncomputable def magic_ball_total_distance : ℕ := sorry

theorem total_distance_is_correct : magic_ball_total_distance = 80 := sorry

end total_distance_is_correct_l334_33424


namespace base4_to_base10_conversion_l334_33478

theorem base4_to_base10_conversion : (2 * 4^4 + 1 * 4^3 + 0 * 4^2 + 1 * 4^1 + 2 * 4^0 = 582) :=
by {
  -- The proof is omitted
  sorry
}

end base4_to_base10_conversion_l334_33478


namespace jesse_stamps_l334_33417

variable (A E : Nat)

theorem jesse_stamps :
  E = 3 * A ∧ E + A = 444 → E = 333 :=
by
  sorry

end jesse_stamps_l334_33417


namespace smallest_sum_of_squares_l334_33446

theorem smallest_sum_of_squares (x y : ℤ) (h : x^2 - y^2 = 175) : 
  ∃ (x y : ℤ), x^2 - y^2 = 175 ∧ x^2 + y^2 = 625 :=
sorry

end smallest_sum_of_squares_l334_33446


namespace benjamin_billboards_l334_33466

theorem benjamin_billboards (B : ℕ) (h1 : 20 + 23 + B = 60) : B = 17 :=
by
  sorry

end benjamin_billboards_l334_33466


namespace value_of_r_l334_33450

theorem value_of_r (n : ℕ) (h : n = 3) : 
  let s := 2^n - 1
  let r := 4^s - s
  r = 16377 := by
  let s := 2^3 - 1
  let r := 4^s - s
  sorry

end value_of_r_l334_33450


namespace product_of_possible_values_l334_33494

theorem product_of_possible_values (N : ℤ) (M L : ℤ) 
(h1 : M = L + N)
(h2 : M - 3 = L + N - 3)
(h3 : L + 5 = L + 5)
(h4 : |(L + N - 3) - (L + 5)| = 4) :
N = 12 ∨ N = 4 → (12 * 4 = 48) :=
by sorry

end product_of_possible_values_l334_33494


namespace min_sum_of_factors_l334_33470

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 2310) : 
  a + b + c ≥ 42 :=
sorry

end min_sum_of_factors_l334_33470


namespace water_added_l334_33493

theorem water_added (initial_fullness : ℝ) (fullness_after : ℝ) (capacity : ℝ) 
  (h_initial : initial_fullness = 0.30) (h_after : fullness_after = 3/4) (h_capacity : capacity = 100) : 
  fullness_after * capacity - initial_fullness * capacity = 45 := 
by 
  sorry

end water_added_l334_33493


namespace no_sum_of_consecutive_integers_to_420_l334_33473

noncomputable def perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def sum_sequence (n a : ℕ) : ℕ :=
n * a + n * (n - 1) / 2

theorem no_sum_of_consecutive_integers_to_420 
  (h1 : 420 > 0)
  (h2 : ∀ (n a : ℕ), n ≥ 2 → sum_sequence n a = 420 → perfect_square a)
  (h3 : ∃ n a, n ≥ 2 ∧ sum_sequence n a = 420 ∧ perfect_square a) :
  false :=
by
  sorry

end no_sum_of_consecutive_integers_to_420_l334_33473


namespace sum_of_a_b_l334_33419

def symmetric_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem sum_of_a_b (a b : ℝ) (h : symmetric_x_axis (3, a) (b, 4)) : a + b = -1 :=
by
  sorry

end sum_of_a_b_l334_33419


namespace largest_r_l334_33431

theorem largest_r (a : ℕ → ℕ) (h : ∀ n, 0 < a n ∧ a n ≤ a (n + 2) ∧ a (n + 2) ≤ Int.sqrt (a n ^ 2 + 2 * a (n + 1))) :
  ∃ M, ∀ n ≥ M, a (n + 2) = a n :=
sorry

end largest_r_l334_33431


namespace evaluate_expression_l334_33448

theorem evaluate_expression 
    (a b c : ℕ) 
    (ha : a = 7)
    (hb : b = 11)
    (hc : c = 13) :
  let numerator := a^3 * (1 / b - 1 / c) + b^3 * (1 / c - 1 / a) + c^3 * (1 / a - 1 / b)
  let denominator := a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)
  numerator / denominator = 31 := 
by {
  sorry
}

end evaluate_expression_l334_33448


namespace solve_for_k_l334_33491

theorem solve_for_k (x y : ℤ) (h₁ : x = 1) (h₂ : y = k) (h₃ : 2 * x + y = 6) : k = 4 :=
by 
  sorry

end solve_for_k_l334_33491


namespace word_count_proof_l334_33434

def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def consonants : List Char := ['B', 'C', 'D', 'F']
def vowels : List Char := ['A', 'E']

def count_unrestricted_words : ℕ := 6 ^ 5
def count_all_vowel_words : ℕ := 2 ^ 5
def count_one_consonant_words : ℕ := 5 * 4 * (2 ^ 4)
def count_fewer_than_two_consonant_words : ℕ := count_all_vowel_words + count_one_consonant_words

def count_words_with_at_least_two_consonants : ℕ :=
  count_unrestricted_words - count_fewer_than_two_consonant_words

theorem word_count_proof :
  count_words_with_at_least_two_consonants = 7424 := 
by
  -- Proof will be provided here. For now we skip it.
  sorry

end word_count_proof_l334_33434


namespace smallest_Y_74_l334_33407

def isDigitBin (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), d = 0 ∨ d = 1

def smallest_Y (Y : ℕ) : Prop :=
  ∃ T : ℕ, T > 0 ∧ isDigitBin T ∧ T % 15 = 0 ∧ Y = T / 15

theorem smallest_Y_74 : smallest_Y 74 := by
  sorry

end smallest_Y_74_l334_33407


namespace probability_all_digits_distinct_probability_all_digits_odd_l334_33443

-- Definitions to be used in the proof
def total_possibilities : ℕ := 10^5
def all_distinct_possibilities : ℕ := 10 * 9 * 8 * 7 * 6
def all_odd_possibilities : ℕ := 5^5

-- Probabilities
def prob_all_distinct : ℚ := all_distinct_possibilities / total_possibilities
def prob_all_odd : ℚ := all_odd_possibilities / total_possibilities

-- Lean 4 Statements to Prove
theorem probability_all_digits_distinct :
  prob_all_distinct = 30240 / 100000 := by
  sorry

theorem probability_all_digits_odd :
  prob_all_odd = 3125 / 100000 := by
  sorry

end probability_all_digits_distinct_probability_all_digits_odd_l334_33443


namespace halfway_between_one_eighth_and_one_tenth_l334_33496

theorem halfway_between_one_eighth_and_one_tenth :
  (1 / 8 + 1 / 10) / 2 = 9 / 80 :=
by
  sorry

end halfway_between_one_eighth_and_one_tenth_l334_33496


namespace sport_formulation_water_quantity_l334_33460

theorem sport_formulation_water_quantity (flavoring : ℝ) (corn_syrup : ℝ) (water : ℝ)
    (hs : flavoring / corn_syrup = 1 / 12) 
    (hw : flavoring / water = 1 / 30) 
    (sport_fs_ratio : flavoring / corn_syrup = 3 * (1 / 12)) 
    (sport_fw_ratio : flavoring / water = (1 / 2) * (1 / 30)) 
    (cs_sport : corn_syrup = 1) : 
    water = 15 :=
by
  sorry

end sport_formulation_water_quantity_l334_33460


namespace alcohol_percentage_new_mixture_l334_33455

theorem alcohol_percentage_new_mixture (initial_volume new_volume alcohol_initial : ℝ)
  (h1 : initial_volume = 15)
  (h2 : alcohol_initial = 0.20 * initial_volume)
  (h3 : new_volume = initial_volume + 5) :
  (alcohol_initial / new_volume) * 100 = 15 := by
  sorry

end alcohol_percentage_new_mixture_l334_33455


namespace exponentiation_power_rule_l334_33487

theorem exponentiation_power_rule (a : ℝ) : (a^2)^3 = a^6 :=
by sorry

end exponentiation_power_rule_l334_33487


namespace hare_height_l334_33488

theorem hare_height (camel_height_ft : ℕ) (hare_height_in_inches : ℕ) :
  (camel_height_ft = 28) ∧ (hare_height_in_inches * 24 = camel_height_ft * 12) → hare_height_in_inches = 14 :=
by
  sorry

end hare_height_l334_33488


namespace least_number_to_subtract_l334_33489

theorem least_number_to_subtract (n m : ℕ) (h : n = 56783421) (d : m = 569) : (n % m) = 56783421 % 569 := 
by sorry

end least_number_to_subtract_l334_33489


namespace fraction_to_decimal_l334_33421

theorem fraction_to_decimal : (7 / 50 : ℝ) = 0.14 := by
  sorry

end fraction_to_decimal_l334_33421


namespace sugar_per_chocolate_bar_l334_33402

-- Definitions from conditions
def total_sugar : ℕ := 177
def lollipop_sugar : ℕ := 37
def chocolate_bar_count : ℕ := 14

-- Proof problem statement
theorem sugar_per_chocolate_bar : 
  (total_sugar - lollipop_sugar) / chocolate_bar_count = 10 := 
by 
  sorry

end sugar_per_chocolate_bar_l334_33402


namespace find_r_l334_33463

theorem find_r 
  (r RB QC : ℝ)
  (angleA : ℝ)
  (h0 : RB = 6)
  (h1 : QC = 4)
  (h2 : angleA = 90) :
  (r + 6) ^ 2 + (r + 4) ^ 2 = 10 ^ 2 → r = 2 := 
by 
  sorry

end find_r_l334_33463


namespace eva_total_marks_correct_l334_33454

-- Definitions based on conditions
def math_marks_second_sem : ℕ := 80
def arts_marks_second_sem : ℕ := 90
def science_marks_second_sem : ℕ := 90

def math_marks_first_sem : ℕ := math_marks_second_sem + 10
def arts_marks_first_sem : ℕ := arts_marks_second_sem - 15
def science_marks_first_sem : ℕ := science_marks_second_sem - (science_marks_second_sem / 3)

def total_marks_first_sem : ℕ := math_marks_first_sem + arts_marks_first_sem + science_marks_first_sem
def total_marks_second_sem : ℕ := math_marks_second_sem + arts_marks_second_sem + science_marks_second_sem

def total_marks_both_sems : ℕ := total_marks_first_sem + total_marks_second_sem

-- Theorem to be proved
theorem eva_total_marks_correct : total_marks_both_sems = 485 := by
  -- Here, we state that we need to prove the total marks sum up to 485
  sorry

end eva_total_marks_correct_l334_33454


namespace projection_of_a_onto_b_l334_33459

namespace VectorProjection

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2)

noncomputable def scalar_projection (a b : ℝ × ℝ) : ℝ :=
  (dot_product a b) / (magnitude b)

theorem projection_of_a_onto_b :
  scalar_projection (1, -2) (3, 4) = -1 := by
    sorry

end VectorProjection

end projection_of_a_onto_b_l334_33459


namespace big_cows_fewer_than_small_cows_l334_33465

theorem big_cows_fewer_than_small_cows (b s : ℕ) (h1 : b = 6) (h2 : s = 7) : 
  (s - b) / s = 1 / 7 :=
by
  sorry

end big_cows_fewer_than_small_cows_l334_33465


namespace trapezoid_perimeter_l334_33404

-- Define the properties of the isosceles trapezoid
structure IsoscelesTrapezoid (A B C D : Type) :=
  (AB CD BC DA : ℝ)
  (AB_parallel_CD : AB = CD)
  (BC_eq_DA : BC = 13)
  (DA_eq_BC : DA = 13)
  (sum_AB_CD : AB + CD = 24)

-- Define the problem's conditions as Lean definitions
def trapezoidABCD : IsoscelesTrapezoid ℝ ℝ ℝ ℝ :=
{
  AB := 12,
  CD := 12,
  BC := 13,
  DA := 13,
  AB_parallel_CD := by sorry,
  BC_eq_DA := by sorry,
  DA_eq_BC := by sorry,
  sum_AB_CD := by sorry,
}

-- State the theorem we want to prove
theorem trapezoid_perimeter (trapezoid : IsoscelesTrapezoid ℝ ℝ ℝ ℝ) : 
  trapezoid.AB + trapezoid.BC + trapezoid.CD + trapezoid.DA = 50 :=
by sorry

end trapezoid_perimeter_l334_33404


namespace exists_sequence_l334_33458

theorem exists_sequence (n : ℕ) : ∃ (a : ℕ → ℕ), 
  (∀ i, 1 ≤ i → i < n → (a i > a (i + 1))) ∧
  (∀ i, 1 ≤ i → i < n → (a i ∣ a (i + 1)^2)) ∧
  (∀ i j, 1 ≤ i → 1 ≤ j → i < n → j < n → (i ≠ j → ¬(a i ∣ a j))) :=
sorry

end exists_sequence_l334_33458


namespace max_sinA_cosB_cosC_l334_33415

theorem max_sinA_cosB_cosC (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 0 < A ∧ A < 180) (h3 : 0 < B ∧ B < 180) (h4 : 0 < C ∧ C < 180) : 
  ∃ M : ℝ, M = (1 + Real.sqrt 5) / 2 ∧ ∀ a b c : ℝ, a + b + c = 180 → 0 < a ∧ a < 180 → 0 < b ∧ b < 180 → 0 < c ∧ c < 180 → (Real.sin a + Real.cos b * Real.cos c) ≤ M :=
by sorry

end max_sinA_cosB_cosC_l334_33415


namespace width_of_cistern_is_6_l334_33418

-- Length of the cistern
def length : ℝ := 8

-- Breadth of the water surface
def breadth : ℝ := 1.85

-- Total wet surface area
def total_wet_surface_area : ℝ := 99.8

-- Let w be the width of the cistern
def width (w : ℝ) : Prop :=
  total_wet_surface_area = (length * w) + 2 * (length * breadth) + 2 * (w * breadth)

theorem width_of_cistern_is_6 : width 6 :=
  by
    -- This proof is omitted. The statement asserts that the width is 6 meters.
    sorry

end width_of_cistern_is_6_l334_33418


namespace center_of_symmetry_l334_33468

-- Define the given conditions
def has_axis_symmetry_x (F : Set (ℝ × ℝ)) : Prop := 
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, y) ∈ F

def has_axis_symmetry_y (F : Set (ℝ × ℝ)) : Prop :=
  ∀ (x y : ℝ), (x, y) ∈ F → (x, -y) ∈ F
  
-- Define the central proof goal
theorem center_of_symmetry (F : Set (ℝ × ℝ)) (H1: has_axis_symmetry_x F) (H2: has_axis_symmetry_y F) :
  ∀ (x y : ℝ), (x, y) ∈ F → (-x, -y) ∈ F :=
sorry

end center_of_symmetry_l334_33468


namespace arrange_f_values_l334_33477

noncomputable def f : ℝ → ℝ := sorry -- Assuming the actual definition is not necessary

-- The function f is even
def even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

-- The function f is strictly decreasing on (-∞, 0)
def strictly_decreasing_on_negative (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, x1 < 0 ∧ x2 < 0 ∧ x1 ≠ x2 → (x1 < x2 ↔ f x1 > f x2)

theorem arrange_f_values (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_decreasing : strictly_decreasing_on_negative f) :
  f 3 < f (-2) ∧ f (-2) < f 1 :=
by
  -- The actual proof would go here.
  sorry

end arrange_f_values_l334_33477


namespace line_x_intercept_l334_33484

-- Define the given points
def Point1 : ℝ × ℝ := (10, 3)
def Point2 : ℝ × ℝ := (-10, -7)

-- Define the x-intercept problem
theorem line_x_intercept (x : ℝ) : 
  ∃ m b : ℝ, (Point1.2 = m * Point1.1 + b) ∧ (Point2.2 = m * Point2.1 + b) ∧ (0 = m * x + b) → x = 4 :=
by
  sorry

end line_x_intercept_l334_33484


namespace floor_neg_sqrt_eval_l334_33441

theorem floor_neg_sqrt_eval :
  ⌊-(Real.sqrt (64 / 9))⌋ = -3 :=
by
  sorry

end floor_neg_sqrt_eval_l334_33441


namespace max_cos_a_correct_l334_33472

noncomputable def max_cos_a (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) : ℝ :=
  Real.sqrt 3 - 1

theorem max_cos_a_correct (a b : ℝ) (h : Real.cos (a + b) = Real.cos a + Real.cos b) :
  max_cos_a a b h = Real.sqrt 3 - 1 :=
sorry

end max_cos_a_correct_l334_33472


namespace range_of_a_l334_33432

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (x > 3 → x > a)) ↔ (a ≤ 3) :=
sorry

end range_of_a_l334_33432


namespace soccer_field_kids_l334_33406

def a := 14
def b := 22
def c := a + b

theorem soccer_field_kids : c = 36 :=
by
    sorry

end soccer_field_kids_l334_33406


namespace parabola_circle_intersection_radius_squared_l334_33425

theorem parabola_circle_intersection_radius_squared :
  (∀ x y, y = (x - 2)^2 → x + 1 = (y + 2)^2 → (x - 1)^2 + (y + 1)^2 = 1) :=
sorry

end parabola_circle_intersection_radius_squared_l334_33425


namespace height_of_water_a_height_of_water_b_height_of_water_c_l334_33474

noncomputable def edge_length : ℝ := 10  -- Edge length of the cube in cm.
noncomputable def angle_deg : ℝ := 20   -- Angle in degrees.

noncomputable def volume_a : ℝ := 100  -- Volume in cm^3 for case a)
noncomputable def height_a : ℝ := 2.53  -- Height in cm for case a)

noncomputable def volume_b : ℝ := 450  -- Volume in cm^3 for case b)
noncomputable def height_b : ℝ := 5.94  -- Height in cm for case b)

noncomputable def volume_c : ℝ := 900  -- Volume in cm^3 for case c)
noncomputable def height_c : ℝ := 10.29  -- Height in cm for case c)

theorem height_of_water_a :
  ∀ (edge_length angle_deg volume_a : ℝ), volume_a = 100 → height_a = 2.53 := by 
  sorry

theorem height_of_water_b :
  ∀ (edge_length angle_deg volume_b : ℝ), volume_b = 450 → height_b = 5.94 := by 
  sorry

theorem height_of_water_c :
  ∀ (edge_length angle_deg volume_c : ℝ), volume_c = 900 → height_c = 10.29 := by 
  sorry

end height_of_water_a_height_of_water_b_height_of_water_c_l334_33474


namespace find_x_l334_33436

/-- Let x be a real number such that the square roots of a positive number are given by x - 4 and 3. 
    Prove that x equals 1. -/
theorem find_x (x : ℝ) 
  (h₁ : ∃ n : ℝ, n > 0 ∧ n.sqrt = x - 4 ∧ n.sqrt = 3) : 
  x = 1 :=
by
  sorry

end find_x_l334_33436


namespace sum_of_remainders_l334_33464

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 9) : (n % 4 + n % 5) = 5 :=
by
  sorry

end sum_of_remainders_l334_33464


namespace circle_tangent_values_l334_33497

theorem circle_tangent_values (m : ℝ) :
  (∀ x y : ℝ, ((x - m)^2 + (y + 2)^2 = 9) → ((x + 1)^2 + (y - m)^2 = 4)) → 
  m = 2 ∨ m = -5 :=
by
  sorry

end circle_tangent_values_l334_33497


namespace value_of_f_2012_1_l334_33438

noncomputable def f : ℝ → ℝ :=
sorry

-- Condition 1: f is even
axiom even_f : ∀ x : ℝ, f x = f (-x)

-- Condition 2: f(x + 3) = -f(x)
axiom periodicity_f : ∀ x : ℝ, f (x + 3) = -f x

-- Condition 3: f(x) = 2x + 3 for -3 ≤ x ≤ 0
axiom defined_f_on_interval : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 0 → f x = 2 * x + 3

-- Assertion to prove
theorem value_of_f_2012_1 : f 2012.1 = -1.2 :=
by sorry

end value_of_f_2012_1_l334_33438


namespace fewerCansCollected_l334_33456

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l334_33456


namespace amount_saved_l334_33462

theorem amount_saved (list_price : ℝ) (tech_deals_discount : ℝ) (electro_bargains_discount : ℝ)
    (tech_deals_price : ℝ) (electro_bargains_price : ℝ) (amount_saved : ℝ) :
  tech_deals_discount = 0.15 →
  list_price = 120 →
  tech_deals_price = list_price * (1 - tech_deals_discount) →
  electro_bargains_discount = 20 →
  electro_bargains_price = list_price - electro_bargains_discount →
  amount_saved = tech_deals_price - electro_bargains_price →
  amount_saved = 2 :=
by
  -- proof steps would go here
  sorry

end amount_saved_l334_33462


namespace shortest_side_of_similar_triangle_l334_33469

theorem shortest_side_of_similar_triangle (h1 : ∀ (a b c : ℝ), a^2 + b^2 = c^2)
  (h2 : 15^2 + b^2 = 34^2) (h3 : ∃ (k : ℝ), k = 68 / 34) :
  ∃ s : ℝ, s = 2 * Real.sqrt 931 :=
by
  sorry

end shortest_side_of_similar_triangle_l334_33469


namespace cuboid_distance_properties_l334_33479

theorem cuboid_distance_properties (cuboid : Type) :
  (∃ P : cuboid → ℝ, ∀ V1 V2 : cuboid, P V1 = P V2) ∧
  ¬ (∃ Q : cuboid → ℝ, ∀ E1 E2 : cuboid, Q E1 = Q E2) ∧
  ¬ (∃ R : cuboid → ℝ, ∀ F1 F2 : cuboid, R F1 = R F2) := 
sorry

end cuboid_distance_properties_l334_33479


namespace difference_of_triangular_2010_2009_l334_33420

def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

theorem difference_of_triangular_2010_2009 :
  triangular 2010 - triangular 2009 = 2010 :=
by
  sorry

end difference_of_triangular_2010_2009_l334_33420


namespace a_range_l334_33475

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2
def g (x a : ℝ) : ℝ := 2 * x + a

theorem a_range :
  (∃ (x1 x2 : ℝ), x1 ∈ Set.Icc (1 / 2) 2 ∧ x2 ∈ Set.Icc (1 / 2) 2 ∧ f x1 = g x2 a) ↔ -5 ≤ a ∧ a ≤ 0 := 
by 
  sorry

end a_range_l334_33475


namespace number_of_stickers_after_losing_page_l334_33451

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l334_33451


namespace correct_calculation_l334_33435

theorem correct_calculation (m n : ℝ) :
  3 * m^2 * n - 3 * m^2 * n = 0 ∧
  ¬ (3 * m^2 - 2 * m^2 = 1) ∧
  ¬ (3 * m^2 + 2 * m^2 = 5 * m^4) ∧
  ¬ (3 * m + 2 * n = 5 * m * n) := by
  sorry

end correct_calculation_l334_33435


namespace area_of_isosceles_trapezoid_l334_33439

theorem area_of_isosceles_trapezoid (R α : ℝ) (hR : R > 0) (hα1 : 0 < α) (hα2 : α < π) :
  let a := 2 * R
  let b := 2 * R * Real.sin (α / 2)
  let h := R * Real.cos (α / 2)
  (1 / 2) * (a + b) * h = R^2 * (1 + Real.sin (α / 2)) * Real.cos (α / 2) :=
by
  sorry

end area_of_isosceles_trapezoid_l334_33439


namespace fans_with_all_vouchers_l334_33480

theorem fans_with_all_vouchers (total_fans : ℕ) 
    (soda_interval : ℕ) (popcorn_interval : ℕ) (hotdog_interval : ℕ)
    (h1 : soda_interval = 60) (h2 : popcorn_interval = 80) (h3 : hotdog_interval = 100)
    (h4 : total_fans = 4500)
    (h5 : Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval) = 1200) :
    (total_fans / Nat.lcm soda_interval (Nat.lcm popcorn_interval hotdog_interval)) = 3 := 
by
    sorry

end fans_with_all_vouchers_l334_33480


namespace condition_sufficiency_l334_33426

theorem condition_sufficiency (x₁ x₂ : ℝ) :
  (x₁ > 4 ∧ x₂ > 4) → (x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) ∧ ¬ ((x₁ + x₂ > 8 ∧ x₁ * x₂ > 16) → (x₁ > 4 ∧ x₂ > 4)) :=
by 
  sorry

end condition_sufficiency_l334_33426
