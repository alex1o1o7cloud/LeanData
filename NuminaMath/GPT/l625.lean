import Mathlib

namespace NUMINAMATH_GPT_check_true_propositions_l625_62545

open Set

theorem check_true_propositions : 
  ∀ (Prop1 Prop2 Prop3 : Prop),
    (Prop1 ↔ (∀ x : ℝ, x^2 > 0)) →
    (Prop2 ↔ ∃ x : ℝ, x^2 ≤ x) →
    (Prop3 ↔ ∀ (M N : Set ℝ) (x : ℝ), x ∈ (M ∩ N) → x ∈ M ∧ x ∈ N) →
    (¬Prop1 ∧ Prop2 ∧ Prop3) →
    (2 = 2) := sorry

end NUMINAMATH_GPT_check_true_propositions_l625_62545


namespace NUMINAMATH_GPT_exponent_multiplication_l625_62564

theorem exponent_multiplication (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (3^a)^b = 3^3) : 3^a * 3^b = 3^4 :=
by
  sorry

end NUMINAMATH_GPT_exponent_multiplication_l625_62564


namespace NUMINAMATH_GPT_old_fridge_cost_l625_62588

-- Define the daily cost of Kurt's old refrigerator
variable (x : ℝ)

-- Define the conditions given in the problem
def new_fridge_cost_per_day : ℝ := 0.45
def savings_per_month : ℝ := 12
def days_in_month : ℝ := 30

-- State the theorem to prove
theorem old_fridge_cost :
  30 * x - 30 * new_fridge_cost_per_day = savings_per_month → x = 0.85 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_old_fridge_cost_l625_62588


namespace NUMINAMATH_GPT_five_digit_number_with_integer_cube_root_l625_62566

theorem five_digit_number_with_integer_cube_root (n : ℕ) 
  (h1 : n ≥ 10000 ∧ n < 100000) 
  (h2 : n % 10 = 3) 
  (h3 : ∃ k : ℕ, k^3 = n) : 
  n = 19683 ∨ n = 50653 :=
sorry

end NUMINAMATH_GPT_five_digit_number_with_integer_cube_root_l625_62566


namespace NUMINAMATH_GPT_sale_coupon_discount_l625_62528

theorem sale_coupon_discount
  (original_price : ℝ)
  (sale_price : ℝ)
  (price_after_coupon : ℝ)
  (h1 : sale_price = 0.5 * original_price)
  (h2 : price_after_coupon = 0.8 * sale_price) :
  (original_price - price_after_coupon) / original_price * 100 = 60 := by
sorry

end NUMINAMATH_GPT_sale_coupon_discount_l625_62528


namespace NUMINAMATH_GPT_prove_value_range_for_a_l625_62591

noncomputable def f (x a : ℝ) : ℝ :=
  (x^2 + a*x + 7 + a) / (x + 1)

noncomputable def g (x : ℝ) : ℝ := 
  - ((x + 1) + (8 / (x + 1))) + 6

theorem prove_value_range_for_a (a : ℝ) :
  (∀ x : ℕ, x > 0 → f x a ≥ 4) ↔ (a ≥ 1 / 3) :=
sorry

end NUMINAMATH_GPT_prove_value_range_for_a_l625_62591


namespace NUMINAMATH_GPT_sixth_edge_length_l625_62568

theorem sixth_edge_length (a b c d o : Type) (distance : a -> a -> ℝ) (circumradius : ℝ) 
  (edge_length : ℝ) (h : ∀ (x y : a), x ≠ y → distance x y = edge_length ∨ distance x y = circumradius)
  (eq_edge_length : edge_length = 3) (eq_circumradius : circumradius = 2) : 
  ∃ ad : ℝ, ad = 6 * Real.sqrt (3 / 7) := 
by
  sorry

end NUMINAMATH_GPT_sixth_edge_length_l625_62568


namespace NUMINAMATH_GPT_simple_interest_rate_l625_62585

theorem simple_interest_rate (P A T : ℕ) (P_val : P = 750) (A_val : A = 900) (T_val : T = 8) : 
  ∃ (R : ℚ), R = 2.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_simple_interest_rate_l625_62585


namespace NUMINAMATH_GPT_sum_first_n_natural_numbers_l625_62510

theorem sum_first_n_natural_numbers (n : ℕ) (h : (n * (n + 1)) / 2 = 1035) : n = 46 :=
sorry

end NUMINAMATH_GPT_sum_first_n_natural_numbers_l625_62510


namespace NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l625_62595

section Problem1
variables (a b c : ℝ) (h1 : c ≠ 0) (h2 : a ≠ 0) (h3 : b ≠ 0)

theorem simplify_expression_1 :
  ((a^2 * b / (-c))^3 * (c^2 / (- (a * b)))^2 / (b * c / a)^4)
  = - (a^10 / (b^3 * c^7)) :=
by sorry
end Problem1

section Problem2
variables (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ a) (h3 : b ≠ 0)

theorem simplify_expression_2 :
  ((2 / (a^2 - b^2) - 1 / (a^2 - a * b)) / (a / (a + b))) = 1 / a^2 :=
by sorry
end Problem2

end NUMINAMATH_GPT_simplify_expression_1_simplify_expression_2_l625_62595


namespace NUMINAMATH_GPT_trajectory_moving_circle_l625_62563

theorem trajectory_moving_circle : 
  (∃ P : ℝ × ℝ, (∃ r : ℝ, (P.1 + 1)^2 = r^2 ∧ (P.1 - 2)^2 + P.2^2 = (r + 1)^2) ∧
  P.2^2 = 8 * P.1) :=
sorry

end NUMINAMATH_GPT_trajectory_moving_circle_l625_62563


namespace NUMINAMATH_GPT_inequality_proof_l625_62594

theorem inequality_proof (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) : 
  a^4 + b^4 + c^4 ≥ a * b * c * (a + b + c) := 
by 
  sorry

end NUMINAMATH_GPT_inequality_proof_l625_62594


namespace NUMINAMATH_GPT_alcohol_water_ratio_l625_62538

theorem alcohol_water_ratio (A W A_new W_new : ℝ) (ha1 : A / W = 4 / 3) (ha2: A = 5) (ha3: W_new = W + 7) : A / W_new = 1 / 2.15 :=
by
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l625_62538


namespace NUMINAMATH_GPT_sample_size_correct_l625_62541

def sample_size (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ) (S : ℕ) : Prop :=
  sum_frequencies = 20 ∧ frequency_sum_ratio = 0.4 → S = 50

theorem sample_size_correct :
  ∀ (sum_frequencies : ℕ) (frequency_sum_ratio : ℚ),
    sample_size sum_frequencies frequency_sum_ratio 50 :=
by
  intros sum_frequencies frequency_sum_ratio
  sorry

end NUMINAMATH_GPT_sample_size_correct_l625_62541


namespace NUMINAMATH_GPT_find_x_l625_62555

variables {a b : EuclideanSpace ℝ (Fin 2)} {x : ℝ}

theorem find_x (h1 : ‖a + b‖ = 1) (h2 : ‖a - b‖ = x) (h3 : inner a b = -(3 / 8) * x) : x = 2 ∨ x = -(1 / 2) :=
sorry

end NUMINAMATH_GPT_find_x_l625_62555


namespace NUMINAMATH_GPT_distinct_values_of_expr_l625_62577

theorem distinct_values_of_expr : 
  let a := 3^(3^(3^3));
  let b := 3^((3^3)^3);
  let c := ((3^3)^3)^3;
  let d := (3^(3^3))^3;
  let e := (3^3)^(3^3);
  (a ≠ b) ∧ (c ≠ b) ∧ (d ≠ b) ∧ (d ≠ a) ∧ (e ≠ a) ∧ (e ≠ b) ∧ (e ≠ d) := sorry

end NUMINAMATH_GPT_distinct_values_of_expr_l625_62577


namespace NUMINAMATH_GPT_map_distance_to_actual_distance_l625_62524

theorem map_distance_to_actual_distance
  (map_distance : ℝ)
  (scale_inches : ℝ)
  (scale_miles : ℝ)
  (actual_distance : ℝ)
  (h_scale : scale_inches = 0.5)
  (h_scale_miles : scale_miles = 10)
  (h_map_distance : map_distance = 20) :
  actual_distance = 400 :=
by
  sorry

end NUMINAMATH_GPT_map_distance_to_actual_distance_l625_62524


namespace NUMINAMATH_GPT_solution_set_lg2_l625_62509

noncomputable def f : ℝ → ℝ := sorry

axiom f_1 : f 1 = 1
axiom f_deriv_lt : ∀ x : ℝ, deriv f x < 1

theorem solution_set_lg2 : { x : ℝ | f (Real.log x ^ 2) < Real.log x ^ 2 } = { x : ℝ | (1/10 : ℝ) < x ∧ x < 10 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_lg2_l625_62509


namespace NUMINAMATH_GPT_ratio_final_to_initial_l625_62599

def initial_amount (P : ℝ) := P
def interest_rate := 4 / 100
def time_period := 25

def simple_interest (P : ℝ) := P * interest_rate * time_period

def final_amount (P : ℝ) := P + simple_interest P

theorem ratio_final_to_initial (P : ℝ) (hP : P > 0) :
  final_amount P / initial_amount P = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_final_to_initial_l625_62599


namespace NUMINAMATH_GPT_find_b_eq_five_l625_62537

/--
Given points A(4, 2) and B(0, b) in the Cartesian coordinate system,
and the condition that the distances from O (the origin) to B and from B to A are equal,
prove that b = 5.
-/
theorem find_b_eq_five : ∃ b : ℝ, (dist (0, 0) (0, b) = dist (0, b) (4, 2)) ∧ b = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_b_eq_five_l625_62537


namespace NUMINAMATH_GPT_rational_solutions_count_l625_62508

theorem rational_solutions_count :
  ∃ (sols : Finset (ℚ × ℚ × ℚ)), 
    (∀ (x y z : ℚ), (x + y + z = 0) ∧ (x * y * z + z = 0) ∧ (x * y + y * z + x * z + y = 0) ↔ (x, y, z) ∈ sols) ∧
    sols.card = 3 :=
by
  sorry

end NUMINAMATH_GPT_rational_solutions_count_l625_62508


namespace NUMINAMATH_GPT_largest_divisor_of_five_consecutive_integers_product_correct_l625_62532

noncomputable def largest_divisor_of_five_consecutive_integers_product : ℕ :=
  120

theorem largest_divisor_of_five_consecutive_integers_product_correct :
  ∀ (n : ℕ), (∃ k : ℕ, k = n * (n + 1) * (n + 2) * (n + 3) * (n + 4) ∧ 120 ∣ k) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_five_consecutive_integers_product_correct_l625_62532


namespace NUMINAMATH_GPT_ratio_of_female_to_male_members_l625_62536

theorem ratio_of_female_to_male_members 
  (f m : ℕ) 
  (avg_age_female : ℕ) 
  (avg_age_male : ℕ)
  (avg_age_all : ℕ) 
  (H1 : avg_age_female = 45)
  (H2 : avg_age_male = 25)
  (H3 : avg_age_all = 35)
  (H4 : (f + m) ≠ 0) :
  (45 * f + 25 * m) / (f + m) = 35 → f = m :=
by sorry

end NUMINAMATH_GPT_ratio_of_female_to_male_members_l625_62536


namespace NUMINAMATH_GPT_intersection_A_B_union_A_complement_B_subset_C_B_range_l625_62516

def set_A : Set ℝ := { x | 1 ≤ x ∧ x < 6 }
def set_B : Set ℝ := { x | 2 < x ∧ x < 9 }
def set_C (a : ℝ) : Set ℝ := { x | a < x ∧ x < a + 1 }

theorem intersection_A_B :
  set_A ∩ set_B = { x | 2 < x ∧ x < 6 } :=
sorry

theorem union_A_complement_B :
  set_A ∪ (compl set_B) = { x | x < 6 } ∪ { x | x ≥ 9 } :=
sorry

theorem subset_C_B_range (a : ℝ) :
  (set_C a ⊆ set_B) → (2 ≤ a ∧ a ≤ 8) :=
sorry

end NUMINAMATH_GPT_intersection_A_B_union_A_complement_B_subset_C_B_range_l625_62516


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l625_62580

-- Define the context/conditions
noncomputable def hyperbola_vertex_to_asymptote_distance (a b e : ℝ) : Prop :=
  (2 = b / e)

noncomputable def hyperbola_focus_to_asymptote_distance (a b e : ℝ) : Prop :=
  (6 = b)

-- Define the main theorem to prove the eccentricity
theorem hyperbola_eccentricity (a b e : ℝ) (h1 : hyperbola_vertex_to_asymptote_distance a b e) (h2 : hyperbola_focus_to_asymptote_distance a b e) : 
  e = 3 := 
sorry 

end NUMINAMATH_GPT_hyperbola_eccentricity_l625_62580


namespace NUMINAMATH_GPT_Lisa_goal_achievable_l625_62533

open Nat

theorem Lisa_goal_achievable :
  ∀ (total_quizzes quizzes_with_A goal_percentage : ℕ),
  total_quizzes = 60 →
  quizzes_with_A = 25 →
  goal_percentage = 85 →
  (quizzes_with_A < goal_percentage * total_quizzes / 100) →
  (∃ remaining_quizzes, goal_percentage * total_quizzes / 100 - quizzes_with_A > remaining_quizzes) :=
by
  intros total_quizzes quizzes_with_A goal_percentage h_total h_A h_goal h_lack
  let needed_quizzes := goal_percentage * total_quizzes / 100
  let remaining_quizzes := total_quizzes - 35
  have h_needed := needed_quizzes - quizzes_with_A
  use remaining_quizzes
  sorry

end NUMINAMATH_GPT_Lisa_goal_achievable_l625_62533


namespace NUMINAMATH_GPT_exists_positive_integers_abc_l625_62547

theorem exists_positive_integers_abc (m n : ℕ) (h_coprime : Nat.gcd m n = 1) (h_m_gt_one : 1 < m) (h_n_gt_one : 1 < n) :
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ m^a = 1 + n^b * c ∧ Nat.gcd c n = 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integers_abc_l625_62547


namespace NUMINAMATH_GPT_theta_range_l625_62550

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem theta_range (k : ℤ) (θ : ℝ) : 
  (2 * ↑k * π - 5 * π / 6 < θ ∧ θ < 2 * ↑k * π - π / 6) →
  (f (1 / (Real.sin θ)) + f (Real.cos (2 * θ)) < f π - f (1 / π)) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_theta_range_l625_62550


namespace NUMINAMATH_GPT_problem_condition_l625_62597

noncomputable def f : ℝ → ℝ := sorry

theorem problem_condition (h: ∀ x : ℝ, f x > (deriv f) x) : 3 * f (Real.log 2) > 2 * f (Real.log 3) :=
sorry

end NUMINAMATH_GPT_problem_condition_l625_62597


namespace NUMINAMATH_GPT_angle_F_measure_l625_62572

-- Given conditions
def D := 74
def sum_of_angles (x E D : ℝ) := x + E + D = 180
def E_formula (x : ℝ) := 2 * x - 10

-- Proof problem statement in Lean 4
theorem angle_F_measure :
  ∃ x : ℝ, x = (116 / 3) ∧
    sum_of_angles x (E_formula x) D :=
sorry

end NUMINAMATH_GPT_angle_F_measure_l625_62572


namespace NUMINAMATH_GPT_perimeter_shaded_region_l625_62549

-- Definitions based on conditions
def circle_radius : ℝ := 10
def central_angle : ℝ := 300

-- Statement: Perimeter of the shaded region
theorem perimeter_shaded_region 
  : (10 : ℝ) + (10 : ℝ) + ((5 / 6) * (2 * Real.pi * 10)) = (20 : ℝ) + (50 / 3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_perimeter_shaded_region_l625_62549


namespace NUMINAMATH_GPT_total_members_in_sports_club_l625_62539

-- Definitions as per the conditions
def B : ℕ := 20 -- number of members who play badminton
def T : ℕ := 23 -- number of members who play tennis
def Both : ℕ := 7 -- number of members who play both badminton and tennis
def Neither : ℕ := 6 -- number of members who do not play either sport

-- Theorem statement to prove the correct answer
theorem total_members_in_sports_club : B + T - Both + Neither = 42 :=
by
  sorry

end NUMINAMATH_GPT_total_members_in_sports_club_l625_62539


namespace NUMINAMATH_GPT_expected_value_a_squared_is_correct_l625_62578

variables (n : ℕ)
noncomputable def expected_value_a_squared := ((2 * n) + (n^2)) / 3

theorem expected_value_a_squared_is_correct : 
  expected_value_a_squared n = ((2 * n) + (n^2)) / 3 := 
by 
  sorry

end NUMINAMATH_GPT_expected_value_a_squared_is_correct_l625_62578


namespace NUMINAMATH_GPT_pyramid_volume_l625_62583

-- Definitions based on the given conditions
def AB : ℝ := 15
def AD : ℝ := 8
def Area_Δ_ABE : ℝ := 120
def Area_Δ_CDE : ℝ := 64
def h : ℝ := 16
def Base_Area : ℝ := AB * AD

-- Statement to prove the volume of the pyramid is 640
theorem pyramid_volume : (1 / 3) * Base_Area * h = 640 :=
sorry

end NUMINAMATH_GPT_pyramid_volume_l625_62583


namespace NUMINAMATH_GPT_parallel_vectors_implies_value_of_t_l625_62527

theorem parallel_vectors_implies_value_of_t (t : ℝ) :
  let a := (1, t)
  let b := (t, 9)
  (1 * 9 - t^2 = 0) → (t = 3 ∨ t = -3) := 
by sorry

end NUMINAMATH_GPT_parallel_vectors_implies_value_of_t_l625_62527


namespace NUMINAMATH_GPT_parabola_distance_relation_l625_62500

theorem parabola_distance_relation {n : ℝ} {x₁ x₂ y₁ y₂ : ℝ}
  (h₁ : y₁ = x₁^2 - 4 * x₁ + n)
  (h₂ : y₂ = x₂^2 - 4 * x₂ + n)
  (h : y₁ > y₂) :
  |x₁ - 2| > |x₂ - 2| := 
sorry

end NUMINAMATH_GPT_parabola_distance_relation_l625_62500


namespace NUMINAMATH_GPT_exponent_form_l625_62504

theorem exponent_form (y : ℕ) (w : ℕ) (k : ℕ) : w = 3 ^ y → w % 10 = 7 → ∃ (k : ℕ), y = 4 * k + 3 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_exponent_form_l625_62504


namespace NUMINAMATH_GPT_marta_total_spent_l625_62596

theorem marta_total_spent :
  let sale_book_cost := 5 * 10
  let online_book_cost := 40
  let bookstore_book_cost := 3 * online_book_cost
  let total_spent := sale_book_cost + online_book_cost + bookstore_book_cost
  total_spent = 210 := sorry

end NUMINAMATH_GPT_marta_total_spent_l625_62596


namespace NUMINAMATH_GPT_simplify_expression_l625_62574

variable (x : ℝ)

theorem simplify_expression :
  (25 * x^3) * (8 * x^2) * (1 / (4 * x) ^ 3) = (25 / 8) * x^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l625_62574


namespace NUMINAMATH_GPT_gcf_45_75_90_l625_62540

-- Definitions as conditions
def number1 : Nat := 45
def number2 : Nat := 75
def number3 : Nat := 90

def factors_45 : Nat × Nat := (3, 2) -- represents 3^2 * 5^1 {prime factor 3, prime factor 5}
def factors_75 : Nat × Nat := (5, 1) -- represents 3^1 * 5^2 {prime factor 3, prime factor 5}
def factors_90 : Nat × Nat := (3, 2) -- represents 2^1 * 3^2 * 5^1 {prime factor 3, prime factor 5}

-- Theorems to be proved
theorem gcf_45_75_90 : Nat.gcd (Nat.gcd number1 number2) number3 = 15 :=
by {
  -- This is here as placeholder for actual proof
  sorry
}

end NUMINAMATH_GPT_gcf_45_75_90_l625_62540


namespace NUMINAMATH_GPT_students_wearing_other_colors_l625_62570

-- Definitions based on conditions
def total_students := 700
def percentage_blue := 45 / 100
def percentage_red := 23 / 100
def percentage_green := 15 / 100

-- The proof problem statement
theorem students_wearing_other_colors :
  (total_students - total_students * (percentage_blue + percentage_red + percentage_green)) = 119 :=
by
  sorry

end NUMINAMATH_GPT_students_wearing_other_colors_l625_62570


namespace NUMINAMATH_GPT_min_value_of_x2_add_y2_l625_62579

theorem min_value_of_x2_add_y2 (x y : ℝ) (h : (x + 5)^2 + (y - 12)^2 = 14^2) : x^2 + y^2 ≥ 1 :=
sorry

end NUMINAMATH_GPT_min_value_of_x2_add_y2_l625_62579


namespace NUMINAMATH_GPT_trajectory_of_M_l625_62515

theorem trajectory_of_M (x y t : ℝ) (M P F : ℝ × ℝ)
    (hF : F = (1, 0))
    (hP : P = (1/4 * t^2, t))
    (hFP : (P.1 - F.1, P.2 - F.2) = (1/4 * t^2 - 1, t))
    (hFM : (M.1 - F.1, M.2 - F.2) = (x - 1, y))
    (hFP_FM : (P.1 - F.1, P.2 - F.2) = (2 * (M.1 - F.1), 2 * (M.2 - F.2))) :
  y^2 = 2 * x - 1 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_M_l625_62515


namespace NUMINAMATH_GPT_difference_two_digit_interchanged_l625_62576

theorem difference_two_digit_interchanged
  (x y : ℕ)
  (h1 : y = 2 * x)
  (h2 : (10 * x + y) - (x + y) = 8) :
  (10 * y + x) - (10 * x + y) = 9 := by
sorry

end NUMINAMATH_GPT_difference_two_digit_interchanged_l625_62576


namespace NUMINAMATH_GPT_find_range_of_a_l625_62521

def p (a : ℝ) : Prop := 
  a = 0 ∨ (a > 0 ∧ a^2 - 4 * a < 0)

def q (a : ℝ) : Prop := 
  a^2 - 2 * a - 3 < 0

theorem find_range_of_a (a : ℝ) 
  (h1 : p a ∨ q a) 
  (h2 : ¬(p a ∧ q a)) : 
  (-1 < a ∧ a < 0) ∨ (3 ≤ a ∧ a < 4) := 
sorry

end NUMINAMATH_GPT_find_range_of_a_l625_62521


namespace NUMINAMATH_GPT_Jules_height_l625_62575

theorem Jules_height (Ben_initial_height Jules_initial_height Ben_current_height Jules_current_height : ℝ) 
  (h_initial : Ben_initial_height = Jules_initial_height)
  (h_Ben_growth : Ben_current_height = 1.25 * Ben_initial_height)
  (h_Jules_growth : Jules_current_height = Jules_initial_height + (Ben_current_height - Ben_initial_height) / 3)
  (h_Ben_current : Ben_current_height = 75) 
  : Jules_current_height = 65 := 
by
  -- Use the conditions to prove that Jules is now 65 inches tall
  sorry

end NUMINAMATH_GPT_Jules_height_l625_62575


namespace NUMINAMATH_GPT_Jims_apples_fits_into_average_l625_62552

def Jim_apples : Nat := 20
def Jane_apples : Nat := 60
def Jerry_apples : Nat := 40

def total_apples : Nat := Jim_apples + Jane_apples + Jerry_apples
def number_of_people : Nat := 3
def average_apples_per_person : Nat := total_apples / number_of_people

theorem Jims_apples_fits_into_average :
  average_apples_per_person / Jim_apples = 2 := by
  sorry

end NUMINAMATH_GPT_Jims_apples_fits_into_average_l625_62552


namespace NUMINAMATH_GPT_clinton_earnings_correct_l625_62560

-- Define the conditions as variables/constants
def num_students_Arlington : ℕ := 8
def days_Arlington : ℕ := 4

def num_students_Bradford : ℕ := 6
def days_Bradford : ℕ := 7

def num_students_Clinton : ℕ := 7
def days_Clinton : ℕ := 8

def total_compensation : ℝ := 1456

noncomputable def total_student_days : ℕ :=
  num_students_Arlington * days_Arlington + num_students_Bradford * days_Bradford + num_students_Clinton * days_Clinton

noncomputable def daily_wage : ℝ :=
  total_compensation / total_student_days

noncomputable def earnings_Clinton : ℝ :=
  daily_wage * (num_students_Clinton * days_Clinton)

theorem clinton_earnings_correct : earnings_Clinton = 627.2 := by 
  sorry

end NUMINAMATH_GPT_clinton_earnings_correct_l625_62560


namespace NUMINAMATH_GPT_price_of_basic_computer_l625_62529

theorem price_of_basic_computer (C P : ℝ) 
    (h1 : C + P = 2500) 
    (h2 : P = (1/8) * (C + 500 + P)) :
    C = 2125 :=
by
  sorry

end NUMINAMATH_GPT_price_of_basic_computer_l625_62529


namespace NUMINAMATH_GPT_divide_circle_three_equal_areas_l625_62517

theorem divide_circle_three_equal_areas (OA : ℝ) (r1 r2 : ℝ) 
  (hr1 : r1 = (OA * Real.sqrt 3) / 3) 
  (hr2 : r2 = (OA * Real.sqrt 6) / 3) : 
  ∀ (r : ℝ), r = OA → 
  (∀ (A1 A2 A3 : ℝ), A1 = π * r1 ^ 2 ∧ A2 = π * (r2 ^ 2 - r1 ^ 2) ∧ A3 = π * (r ^ 2 - r2 ^ 2) →
  A1 = A2 ∧ A2 = A3) :=
by
  sorry

end NUMINAMATH_GPT_divide_circle_three_equal_areas_l625_62517


namespace NUMINAMATH_GPT_final_amount_in_account_l625_62561

noncomputable def initial_deposit : ℝ := 1000
noncomputable def first_year_interest_rate : ℝ := 0.2
noncomputable def first_year_balance : ℝ := initial_deposit * (1 + first_year_interest_rate)
noncomputable def withdrawal_amount : ℝ := first_year_balance / 2
noncomputable def after_withdrawal_balance : ℝ := first_year_balance - withdrawal_amount
noncomputable def second_year_interest_rate : ℝ := 0.15
noncomputable def final_balance : ℝ := after_withdrawal_balance * (1 + second_year_interest_rate)

theorem final_amount_in_account : final_balance = 690 := by
  sorry

end NUMINAMATH_GPT_final_amount_in_account_l625_62561


namespace NUMINAMATH_GPT_smallest_x_l625_62569

theorem smallest_x (x : ℕ) : 
  (x % 5 = 4) ∧ (x % 7 = 6) ∧ (x % 9 = 8) ↔ x = 314 := 
by
  sorry

end NUMINAMATH_GPT_smallest_x_l625_62569


namespace NUMINAMATH_GPT_mean_of_set_l625_62554

theorem mean_of_set (m : ℝ) (h : m + 7 = 12) :
  (m + (m + 6) + (m + 7) + (m + 11) + (m + 18)) / 5 = 13.4 :=
by sorry

end NUMINAMATH_GPT_mean_of_set_l625_62554


namespace NUMINAMATH_GPT_cyclist_speed_ratio_l625_62512

variables (k r t v1 v2 : ℝ)
variable (h1 : v1 = 2 * v2) -- Condition 5

-- When traveling in the same direction, relative speed is v1 - v2 and they cover 2k miles in 3r hours
variable (h2 : 2 * k = (v1 - v2) * 3 * r)

-- When traveling in opposite directions, relative speed is v1 + v2 and they pass each other in 2t hours
variable (h3 : 2 * k = (v1 + v2) * 2 * t)

theorem cyclist_speed_ratio (h1 : v1 = 2 * v2) (h2 : 2 * k = (v1 - v2) * 3 * r) (h3 : 2 * k = (v1 + v2) * 2 * t) :
  v1 / v2 = 2 :=
sorry

end NUMINAMATH_GPT_cyclist_speed_ratio_l625_62512


namespace NUMINAMATH_GPT_ratio_monkeys_snakes_l625_62514

def parrots : ℕ := 8
def snakes : ℕ := 3 * parrots
def elephants : ℕ := (parrots + snakes) / 2
def zebras : ℕ := elephants - 3
def monkeys : ℕ := zebras + 35

theorem ratio_monkeys_snakes : (monkeys : ℕ) / (snakes : ℕ) = 2 / 1 :=
by
  sorry

end NUMINAMATH_GPT_ratio_monkeys_snakes_l625_62514


namespace NUMINAMATH_GPT_erasers_pens_markers_cost_l625_62506

theorem erasers_pens_markers_cost 
  (E P M : ℝ)
  (h₁ : E + 3 * P + 2 * M = 240)
  (h₂ : 2 * E + 4 * M + 5 * P = 440) :
  3 * E + 4 * P + 6 * M = 520 :=
sorry

end NUMINAMATH_GPT_erasers_pens_markers_cost_l625_62506


namespace NUMINAMATH_GPT_initial_balloons_correct_l625_62531

-- Define the variables corresponding to the conditions given in the problem
def boy_balloon_count := 3
def girl_balloon_count := 12
def balloons_sold := boy_balloon_count + girl_balloon_count
def balloons_remaining := 21

-- State the theorem asserting the initial number of balloons
theorem initial_balloons_correct :
  balloons_sold + balloons_remaining = 36 := sorry

end NUMINAMATH_GPT_initial_balloons_correct_l625_62531


namespace NUMINAMATH_GPT_has_exactly_one_solution_l625_62501

theorem has_exactly_one_solution (a : ℝ) : 
  (∀ x : ℝ, 5^(x^2 + 2 * a * x + a^2) = a * x^2 + 2 * a^2 * x + a^3 + a^2 - 6 * a + 6) ↔ (a = 1) :=
sorry

end NUMINAMATH_GPT_has_exactly_one_solution_l625_62501


namespace NUMINAMATH_GPT_math_problem_l625_62556

theorem math_problem
  (x y : ℝ)
  (h1 : x + y = 5)
  (h2 : x * y = -3)
  : x + (x^3 / y^2) + (y^3 / x^2) + y = 590.5 :=
sorry

end NUMINAMATH_GPT_math_problem_l625_62556


namespace NUMINAMATH_GPT_fry_sausage_time_l625_62559

variable (time_per_sausage : ℕ)

noncomputable def time_for_sausages (sausages : ℕ) (tps : ℕ) : ℕ :=
  sausages * tps

noncomputable def time_for_eggs (eggs : ℕ) (minutes_per_egg : ℕ) : ℕ :=
  eggs * minutes_per_egg

noncomputable def total_time (time_sausages : ℕ) (time_eggs : ℕ) : ℕ :=
  time_sausages + time_eggs

theorem fry_sausage_time :
  let sausages := 3
  let eggs := 6
  let minutes_per_egg := 4
  let total_time_taken := 39
  total_time (time_for_sausages sausages time_per_sausage) (time_for_eggs eggs minutes_per_egg) = total_time_taken
  → time_per_sausage = 5 := by
  sorry

end NUMINAMATH_GPT_fry_sausage_time_l625_62559


namespace NUMINAMATH_GPT_fg_minus_gf_l625_62567

noncomputable def f (x : ℝ) : ℝ := 8 * x - 12
noncomputable def g (x : ℝ) : ℝ := x / 4 - 1

theorem fg_minus_gf (x : ℝ) : f (g x) - g (f x) = -16 :=
by
  -- We skip the proof.
  sorry

end NUMINAMATH_GPT_fg_minus_gf_l625_62567


namespace NUMINAMATH_GPT_max_ab_l625_62553

noncomputable def f (a x : ℝ) : ℝ := -a * Real.log x + (a + 1) * x - (1/2) * x^2

theorem max_ab (a b : ℝ) (h₁ : 0 < a)
  (h₂ : ∀ x, f a x ≥ - (1/2) * x^2 + a * x + b) : 
  ab ≤ ((Real.exp 1) / 2) :=
sorry

end NUMINAMATH_GPT_max_ab_l625_62553


namespace NUMINAMATH_GPT_correct_observation_value_l625_62571

theorem correct_observation_value (mean : ℕ) (n : ℕ) (incorrect_obs : ℕ) (corrected_mean : ℚ) (original_sum : ℚ) (remaining_sum : ℚ) (corrected_sum : ℚ) :
  mean = 30 →
  n = 50 →
  incorrect_obs = 23 →
  corrected_mean = 30.5 →
  original_sum = (n * mean) →
  remaining_sum = (original_sum - incorrect_obs) →
  corrected_sum = (n * corrected_mean) →
  ∃ x : ℕ, remaining_sum + x = corrected_sum → x = 48 :=
by
  intros h_mean h_n h_incorrect_obs h_corrected_mean h_original_sum h_remaining_sum h_corrected_sum
  have original_mean := h_mean
  have observations := h_n
  have incorrect_observation := h_incorrect_obs
  have new_mean := h_corrected_mean
  have original_sum_calc := h_original_sum
  have remaining_sum_calc := h_remaining_sum
  have corrected_sum_calc := h_corrected_sum
  use 48
  sorry

end NUMINAMATH_GPT_correct_observation_value_l625_62571


namespace NUMINAMATH_GPT_x_equals_y_l625_62505

-- Conditions
def x := 2 * 20212021 * 1011 * 202320232023
def y := 43 * 47 * 20232023 * 202220222022

-- Proof statement
theorem x_equals_y : x = y := sorry

end NUMINAMATH_GPT_x_equals_y_l625_62505


namespace NUMINAMATH_GPT_distinct_terms_in_expansion_l625_62526

theorem distinct_terms_in_expansion :
  let n1 := 2 -- number of terms in (x + y)
  let n2 := 3 -- number of terms in (a + b + c)
  let n3 := 3 -- number of terms in (d + e + f)
  (n1 * n2 * n3) = 18 :=
by
  sorry

end NUMINAMATH_GPT_distinct_terms_in_expansion_l625_62526


namespace NUMINAMATH_GPT_unique_root_of_quadratic_eq_l625_62544

theorem unique_root_of_quadratic_eq (a b c : ℝ) (d : ℝ) 
  (h_seq : b = a - d ∧ c = a - 2 * d) 
  (h_nonneg : a ≥ b ∧ b ≥ c ∧ c ≥ 0) 
  (h_discriminant : (-(a - d))^2 - 4 * a * (a - 2 * d) = 0) :
  ∃ x : ℝ, (ax^2 - bx + c = 0) ∧ x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_unique_root_of_quadratic_eq_l625_62544


namespace NUMINAMATH_GPT_equality_of_arithmetic_sums_l625_62542

def sum_arithmetic_sequence (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem equality_of_arithmetic_sums (n : ℕ) (h : n ≠ 0) :
  sum_arithmetic_sequence 8 4 n = sum_arithmetic_sequence 17 2 n ↔ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_equality_of_arithmetic_sums_l625_62542


namespace NUMINAMATH_GPT_rich_avg_time_per_mile_l625_62503

-- Define the total time in minutes and the total distance
def total_minutes : ℕ := 517
def total_miles : ℕ := 50

-- Define a function to calculate the average time per mile
def avg_time_per_mile (total_time : ℕ) (distance : ℕ) : ℚ :=
  total_time / distance

-- Theorem statement
theorem rich_avg_time_per_mile :
  avg_time_per_mile total_minutes total_miles = 10.34 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_rich_avg_time_per_mile_l625_62503


namespace NUMINAMATH_GPT_total_chickens_l625_62590

theorem total_chickens (hens : ℕ) (roosters : ℕ) (h1 : hens = 52) (h2 : roosters = hens + 16) : hens + roosters = 120 :=
by
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_total_chickens_l625_62590


namespace NUMINAMATH_GPT_distinct_exponentiations_are_four_l625_62582

def power (a b : ℕ) : ℕ := a^b

def expr1 := power 3 (power 3 (power 3 3))
def expr2 := power 3 (power (power 3 3) 3)
def expr3 := power (power (power 3 3) 3) 3
def expr4 := power (power 3 (power 3 3)) 3
def expr5 := power (power 3 3) (power 3 3)

theorem distinct_exponentiations_are_four : 
  (expr1 ≠ expr2 ∧ expr1 ≠ expr3 ∧ expr1 ≠ expr4 ∧ expr1 ≠ expr5 ∧
   expr2 ≠ expr3 ∧ expr2 ≠ expr4 ∧ expr2 ≠ expr5 ∧
   expr3 ≠ expr4 ∧ expr3 ≠ expr5 ∧
   expr4 ≠ expr5) :=
sorry

end NUMINAMATH_GPT_distinct_exponentiations_are_four_l625_62582


namespace NUMINAMATH_GPT_mabel_total_tomatoes_l625_62543

theorem mabel_total_tomatoes (n1 n2 n3 n4 : ℕ)
  (h1 : n1 = 8)
  (h2 : n2 = n1 + 4)
  (h3 : n3 = 3 * (n1 + n2))
  (h4 : n4 = 3 * (n1 + n2)) :
  n1 + n2 + n3 + n4 = 140 :=
by
  sorry

end NUMINAMATH_GPT_mabel_total_tomatoes_l625_62543


namespace NUMINAMATH_GPT_tree_break_height_l625_62565

-- Define the problem conditions and prove the required height h
theorem tree_break_height (height_tree : ℝ) (distance_shore : ℝ) (height_break : ℝ) : 
  height_tree = 20 → distance_shore = 6 → 
  (distance_shore ^ 2 + height_break ^ 2 = (height_tree - height_break) ^ 2) →
  height_break = 9.1 :=
by
  intros h_tree_eq h_shore_eq hyp_eq
  have h_tree_20 := h_tree_eq
  have h_shore_6 := h_shore_eq
  have hyp := hyp_eq
  sorry -- Proof of the theorem is omitted

end NUMINAMATH_GPT_tree_break_height_l625_62565


namespace NUMINAMATH_GPT_coconut_grove_l625_62502

theorem coconut_grove (x : ℕ) :
  (40 * (x + 2) + 120 * x + 180 * (x - 2) = 100 * 3 * x) → 
  x = 7 := by
  sorry

end NUMINAMATH_GPT_coconut_grove_l625_62502


namespace NUMINAMATH_GPT_find_a_plus_k_l625_62573

variable (a k : ℝ)

noncomputable def f (x : ℝ) : ℝ := (a - 1) * x^k

theorem find_a_plus_k
  (h1 : f a k (Real.sqrt 2) = 2)
  (h2 : (Real.sqrt 2)^2 = 2) : a + k = 4 := 
sorry

end NUMINAMATH_GPT_find_a_plus_k_l625_62573


namespace NUMINAMATH_GPT_students_not_in_same_column_or_row_l625_62551

-- Define the positions of student A and student B as conditions
structure Position where
  row : Nat
  col : Nat

-- Student A's position is in the 3rd row and 6th column
def StudentA : Position := {row := 3, col := 6}

-- Student B's position is described in a relative manner in terms of columns and rows
def StudentB : Position := {row := 6, col := 3}

-- Formalize the proof statement
theorem students_not_in_same_column_or_row :
  StudentA.row ≠ StudentB.row ∧ StudentA.col ≠ StudentB.col :=
by {
  sorry
}

end NUMINAMATH_GPT_students_not_in_same_column_or_row_l625_62551


namespace NUMINAMATH_GPT_erdos_ginzburg_ziv_2047_l625_62523

open Finset

theorem erdos_ginzburg_ziv_2047 (s : Finset ℕ) (h : s.card = 2047) : 
  ∃ t ⊆ s, t.card = 1024 ∧ (t.sum id) % 1024 = 0 :=
sorry

end NUMINAMATH_GPT_erdos_ginzburg_ziv_2047_l625_62523


namespace NUMINAMATH_GPT_range_of_a_l625_62513

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x → x - Real.log x - a > 0) → a < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a_l625_62513


namespace NUMINAMATH_GPT_solve_inequalities_l625_62598

theorem solve_inequalities (x : ℝ) :
  (3 * x^2 - x > 4) ∧ (x < 3) ↔ (1 < x ∧ x < 3) := 
by 
  sorry

end NUMINAMATH_GPT_solve_inequalities_l625_62598


namespace NUMINAMATH_GPT_water_breaks_vs_sitting_breaks_l625_62522

theorem water_breaks_vs_sitting_breaks :
  (240 / 20) - (240 / 120) = 10 := by
  sorry

end NUMINAMATH_GPT_water_breaks_vs_sitting_breaks_l625_62522


namespace NUMINAMATH_GPT_volume_after_increase_l625_62581

variable (l w h : ℕ)
variable (V S E : ℕ)

noncomputable def original_volume : ℕ := l * w * h
noncomputable def surface_sum : ℕ := (l * w) + (w * h) + (h * l)
noncomputable def edge_sum : ℕ := l + w + h

theorem volume_after_increase (h_volume : original_volume l w h = 5400)
  (h_surface : surface_sum l w h = 1176)
  (h_edge : edge_sum l w h = 60) : 
  (l + 1) * (w + 1) * (h + 1) = 6637 := sorry

end NUMINAMATH_GPT_volume_after_increase_l625_62581


namespace NUMINAMATH_GPT_mean_age_of_seven_friends_l625_62525

theorem mean_age_of_seven_friends 
  (mean_age_group1: ℕ)
  (mean_age_group2: ℕ)
  (n1: ℕ)
  (n2: ℕ)
  (total_friends: ℕ) :
  mean_age_group1 = 147 → 
  mean_age_group2 = 161 →
  n1 = 3 → 
  n2 = 4 →
  total_friends = 7 →
  (mean_age_group1 * n1 + mean_age_group2 * n2) / total_friends = 155 := by
  sorry

end NUMINAMATH_GPT_mean_age_of_seven_friends_l625_62525


namespace NUMINAMATH_GPT_domain_of_f_monotonicity_of_f_inequality_solution_l625_62557

open Real

noncomputable def f (x : ℝ) : ℝ := log ((1 - x) / (1 + x))

theorem domain_of_f :
  ∀ x, -1 < x ∧ x < 1 → ∃ y, y = f x :=
by
  intro x h
  use log ((1 - x) / (1 + x))
  simp [f]

theorem monotonicity_of_f :
  ∀ x y, -1 < x ∧ x < 1 → -1 < y ∧ y < 1 → x < y → f x > f y :=
sorry

theorem inequality_solution :
  ∀ x, f (2 * x - 1) < 0 ↔ (1 / 2 < x ∧ x < 1) :=
sorry

end NUMINAMATH_GPT_domain_of_f_monotonicity_of_f_inequality_solution_l625_62557


namespace NUMINAMATH_GPT_refills_needed_l625_62593

theorem refills_needed 
  (cups_per_day : ℕ)
  (bottle_capacity_oz : ℕ)
  (oz_per_cup : ℕ)
  (total_oz : ℕ)
  (refills : ℕ)
  (h1 : cups_per_day = 12)
  (h2 : bottle_capacity_oz = 16)
  (h3 : oz_per_cup = 8)
  (h4 : total_oz = cups_per_day * oz_per_cup)
  (h5 : refills = total_oz / bottle_capacity_oz) :
  refills = 6 :=
by
  sorry

end NUMINAMATH_GPT_refills_needed_l625_62593


namespace NUMINAMATH_GPT_compare_neg_two_cubed_l625_62511

-- Define the expressions
def neg_two_cubed : ℤ := (-2) ^ 3
def neg_two_cubed_alt : ℤ := -(2 ^ 3)

-- Statement of the problem
theorem compare_neg_two_cubed : neg_two_cubed = neg_two_cubed_alt :=
by
  sorry

end NUMINAMATH_GPT_compare_neg_two_cubed_l625_62511


namespace NUMINAMATH_GPT_overall_class_average_proof_l625_62535

noncomputable def group_1_weighted_average := (0.40 * 80) + (0.60 * 80)
noncomputable def group_2_weighted_average := (0.30 * 60) + (0.70 * 60)
noncomputable def group_3_weighted_average := (0.50 * 40) + (0.50 * 40)
noncomputable def group_4_weighted_average := (0.20 * 50) + (0.80 * 50)

noncomputable def overall_class_average := (0.20 * group_1_weighted_average) + 
                                           (0.50 * group_2_weighted_average) + 
                                           (0.25 * group_3_weighted_average) + 
                                           (0.05 * group_4_weighted_average)

theorem overall_class_average_proof : overall_class_average = 58.5 :=
by 
  unfold overall_class_average
  unfold group_1_weighted_average
  unfold group_2_weighted_average
  unfold group_3_weighted_average
  unfold group_4_weighted_average
  -- now perform the arithmetic calculations
  sorry

end NUMINAMATH_GPT_overall_class_average_proof_l625_62535


namespace NUMINAMATH_GPT_minimum_distance_between_tracks_l625_62507

-- Problem statement as Lean definitions and theorem to prove
noncomputable def rational_man_track (t : ℝ) : ℝ × ℝ :=
  (Real.cos t, Real.sin t)

noncomputable def hyperbolic_man_track (t : ℝ) : ℝ × ℝ :=
  (-1 + 3 * Real.cos (t / 2), 5 * Real.sin (t / 2))

noncomputable def circle_eq := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}

noncomputable def ellipse_eq := {p : ℝ × ℝ | (p.1 + 1)^2 / 9 + p.2^2 / 25 = 1}

theorem minimum_distance_between_tracks : 
  ∃ A ∈ circle_eq, ∃ B ∈ ellipse_eq, dist A B = Real.sqrt 14 - 1 := 
sorry

end NUMINAMATH_GPT_minimum_distance_between_tracks_l625_62507


namespace NUMINAMATH_GPT_total_students_class_l625_62548

theorem total_students_class (S R : ℕ) 
  (h1 : 2 + 12 + 10 + R = S)
  (h2 : (0 * 2) + (1 * 12) + (2 * 10) + (3 * R) = 2 * S) :
  S = 40 := by
  sorry

end NUMINAMATH_GPT_total_students_class_l625_62548


namespace NUMINAMATH_GPT_probability_not_yellow_l625_62519

-- Define the conditions
def red_jelly_beans : Nat := 4
def green_jelly_beans : Nat := 7
def yellow_jelly_beans : Nat := 9
def blue_jelly_beans : Nat := 10

-- Definitions used in the proof problem
def total_jelly_beans : Nat := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans
def non_yellow_jelly_beans : Nat := total_jelly_beans - yellow_jelly_beans

-- Lean statement of the probability problem
theorem probability_not_yellow : 
  (non_yellow_jelly_beans : ℚ) / (total_jelly_beans : ℚ) = 7 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_probability_not_yellow_l625_62519


namespace NUMINAMATH_GPT_first_term_geometric_sequence_l625_62589

theorem first_term_geometric_sequence (a5 a6 : ℚ) (h1 : a5 = 48) (h2 : a6 = 64) : 
  ∃ a : ℚ, a = 243 / 16 :=
by
  sorry

end NUMINAMATH_GPT_first_term_geometric_sequence_l625_62589


namespace NUMINAMATH_GPT_neither_coffee_nor_tea_l625_62562

theorem neither_coffee_nor_tea (total_businesspeople coffee_drinkers tea_drinkers both_drinkers : ℕ) 
    (h_total : total_businesspeople = 35)
    (h_coffee : coffee_drinkers = 18)
    (h_tea : tea_drinkers = 15)
    (h_both : both_drinkers = 6) :
    (total_businesspeople - (coffee_drinkers + tea_drinkers - both_drinkers)) = 8 := 
by
  sorry

end NUMINAMATH_GPT_neither_coffee_nor_tea_l625_62562


namespace NUMINAMATH_GPT_lakeside_fitness_center_ratio_l625_62530

theorem lakeside_fitness_center_ratio (f m c : ℕ)
  (h_avg_age : (35 * f + 30 * m + 10 * c) / (f + m + c) = 25) :
  f = 3 * (m / 6) ∧ f = 3 * (c / 2) :=
by
  sorry

end NUMINAMATH_GPT_lakeside_fitness_center_ratio_l625_62530


namespace NUMINAMATH_GPT_other_root_of_quadratic_l625_62592

variable (p : ℝ)

theorem other_root_of_quadratic (h1: 3 * (-2) * r_2 = -6) : r_2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l625_62592


namespace NUMINAMATH_GPT_additional_amount_deductibles_next_year_l625_62584

theorem additional_amount_deductibles_next_year :
  let avg_deductible : ℝ := 3000
  let inflation_rate : ℝ := 0.03
  let plan_a_rate : ℝ := 2 / 3
  let plan_b_rate : ℝ := 1 / 2
  let plan_c_rate : ℝ := 3 / 5
  let plan_a_percent : ℝ := 0.40
  let plan_b_percent : ℝ := 0.30
  let plan_c_percent : ℝ := 0.30
  let additional_a : ℝ := avg_deductible * plan_a_rate
  let additional_b : ℝ := avg_deductible * plan_b_rate
  let additional_c : ℝ := avg_deductible * plan_c_rate
  let weighted_additional : ℝ := (additional_a * plan_a_percent) + (additional_b * plan_b_percent) + (additional_c * plan_c_percent)
  let inflation_increase : ℝ := weighted_additional * inflation_rate
  let total_additional_amount : ℝ := weighted_additional + inflation_increase
  total_additional_amount = 1843.70 :=
sorry

end NUMINAMATH_GPT_additional_amount_deductibles_next_year_l625_62584


namespace NUMINAMATH_GPT_problem_statement_l625_62546

-- Define that f is an even function and decreasing on (0, +∞)
variables {f : ℝ → ℝ}

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f (x)

def is_decreasing_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → x < y → f y < f x

-- Main statement: Prove the specific inequality under the given conditions
theorem problem_statement (f_even : is_even_function f) (f_decreasing : is_decreasing_on_pos f) :
  f (1/2) > f (-2/3) ∧ f (-2/3) > f (3/4) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l625_62546


namespace NUMINAMATH_GPT_alloy_chromium_amount_l625_62587

theorem alloy_chromium_amount
  (x : ℝ) -- The amount of the first alloy used (in kg)
  (h1 : 0.10 * x + 0.08 * 35 = 0.086 * (x + 35)) -- Condition based on percentages of chromium
  : x = 15 := 
by
  sorry

end NUMINAMATH_GPT_alloy_chromium_amount_l625_62587


namespace NUMINAMATH_GPT_xy_equation_solution_l625_62518

theorem xy_equation_solution (x y : ℝ) (h1 : x * y = 10) (h2 : x^2 * y + x * y^2 + x + y = 120) :
  x^2 + y^2 = 11980 / 121 :=
by
  sorry

end NUMINAMATH_GPT_xy_equation_solution_l625_62518


namespace NUMINAMATH_GPT_length_of_BC_is_7_l625_62520

noncomputable def triangle_length_BC (a b c : ℝ) (A : ℝ) (S : ℝ) (P : ℝ) : Prop :=
  (P = a + b + c) ∧ (P = 20) ∧ (S = 1 / 2 * b * c * Real.sin A) ∧ (S = 10 * Real.sqrt 3) ∧ (A = Real.pi / 3) ∧ (b * c = 20)

theorem length_of_BC_is_7 : ∃ a b c, triangle_length_BC a b c (Real.pi / 3) (10 * Real.sqrt 3) 20 ∧ a = 7 := 
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_length_of_BC_is_7_l625_62520


namespace NUMINAMATH_GPT_arithmetic_mean_of_q_and_r_l625_62534

theorem arithmetic_mean_of_q_and_r (p q r : ℝ) 
  (h₁: (p + q) / 2 = 10) 
  (h₂: r - p = 20) : 
  (q + r) / 2 = 20 :=
sorry

end NUMINAMATH_GPT_arithmetic_mean_of_q_and_r_l625_62534


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_of_sin_l625_62586

open Real

theorem sufficient_not_necessary_condition_of_sin (θ : ℝ) :
  (abs (θ - π / 12) < π / 12) → (sin θ < 1 / 2) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_of_sin_l625_62586


namespace NUMINAMATH_GPT_functional_equation_solution_l625_62558

theorem functional_equation_solution :
  ∀ (f : ℝ → ℝ), (∀ x y, f (f (f x)) + f (f y) = f y + x) → (∀ x, f x = x) :=
by
  intros f h x
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l625_62558
