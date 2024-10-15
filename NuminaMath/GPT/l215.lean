import Mathlib

namespace NUMINAMATH_GPT_even_x_satisfies_remainder_l215_21556

theorem even_x_satisfies_remainder 
  (z : ℕ) 
  (hz : z % 4 = 0) : 
  ∃ (x : ℕ), x % 2 = 0 ∧ (z * (2 + x + z) + 3) % 2 = 1 := 
by
  sorry

end NUMINAMATH_GPT_even_x_satisfies_remainder_l215_21556


namespace NUMINAMATH_GPT_more_stickers_correct_l215_21501

def total_stickers : ℕ := 58
def first_box_stickers : ℕ := 23
def second_box_stickers : ℕ := total_stickers - first_box_stickers
def more_stickers_in_second_box : ℕ := second_box_stickers - first_box_stickers

theorem more_stickers_correct : more_stickers_in_second_box = 12 := by
  sorry

end NUMINAMATH_GPT_more_stickers_correct_l215_21501


namespace NUMINAMATH_GPT_combined_forgotten_angles_l215_21528

-- Define primary conditions
def initial_angle_sum : ℝ := 2873
def correct_angle_sum : ℝ := 16 * 180

-- The theorem to prove
theorem combined_forgotten_angles : correct_angle_sum - initial_angle_sum = 7 :=
by sorry

end NUMINAMATH_GPT_combined_forgotten_angles_l215_21528


namespace NUMINAMATH_GPT_kamala_overestimation_l215_21554

theorem kamala_overestimation : 
  let p := 150
  let q := 50
  let k := 2
  let d := 3
  let p_approx := 160
  let q_approx := 45
  let k_approx := 1
  let d_approx := 4
  let true_value := (p / q) - k + d
  let approx_value := (p_approx / q_approx) - k_approx + d_approx
  approx_value > true_value := 
  by 
  -- Skipping the detailed proof steps.
  sorry

end NUMINAMATH_GPT_kamala_overestimation_l215_21554


namespace NUMINAMATH_GPT_angle_triple_of_supplement_l215_21511

theorem angle_triple_of_supplement (x : ℝ) (h : x = 3 * (180 - x)) : x = 135 :=
by
  sorry

end NUMINAMATH_GPT_angle_triple_of_supplement_l215_21511


namespace NUMINAMATH_GPT_simplify_fraction_l215_21560

theorem simplify_fraction : (5^3 + 5^5) / (5^4 - 5^2) = 65 / 12 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_fraction_l215_21560


namespace NUMINAMATH_GPT_fractional_eq_solution_l215_21584

theorem fractional_eq_solution (x : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ 2) :
  (1 / (x - 1) = 2 / (x - 2)) → (x = 2) :=
by
  sorry

end NUMINAMATH_GPT_fractional_eq_solution_l215_21584


namespace NUMINAMATH_GPT_no_solutions_for_divisibility_by_3_l215_21512

theorem no_solutions_for_divisibility_by_3 (x y : ℤ) : ¬ (x^2 + y^2 + x + y ∣ 3) :=
sorry

end NUMINAMATH_GPT_no_solutions_for_divisibility_by_3_l215_21512


namespace NUMINAMATH_GPT_sum_of_consecutive_numbers_LCM_168_l215_21557

theorem sum_of_consecutive_numbers_LCM_168
  (x y z : ℕ)
  (h1 : y = x + 1)
  (h2 : z = y + 1)
  (h3 : Nat.lcm (Nat.lcm x y) z = 168) :
  x + y + z = 21 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_numbers_LCM_168_l215_21557


namespace NUMINAMATH_GPT_savings_fraction_l215_21595

theorem savings_fraction 
(P : ℝ) 
(f : ℝ) 
(h1 : P > 0) 
(h2 : 12 * f * P = 5 * (1 - f) * P) : 
    f = 5 / 17 :=
by
  sorry

end NUMINAMATH_GPT_savings_fraction_l215_21595


namespace NUMINAMATH_GPT_cos_equation_solution_l215_21534

open Real

theorem cos_equation_solution (m : ℝ) :
  (∀ x : ℝ, 4 * cos x - cos x^2 + m - 3 = 0) ↔ (0 ≤ m ∧ m ≤ 8) := by
  sorry

end NUMINAMATH_GPT_cos_equation_solution_l215_21534


namespace NUMINAMATH_GPT_arithmetic_and_geometric_mean_l215_21552

theorem arithmetic_and_geometric_mean (x y : ℝ) (h₁ : (x + y) / 2 = 20) (h₂ : Real.sqrt (x * y) = Real.sqrt 150) : x^2 + y^2 = 1300 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_and_geometric_mean_l215_21552


namespace NUMINAMATH_GPT_investment_duration_l215_21523

noncomputable def log (x : ℝ) := Real.log x

theorem investment_duration 
  (P A : ℝ) 
  (r : ℝ) 
  (n : ℕ) 
  (t : ℝ) 
  (hP : P = 3000) 
  (hA : A = 3630) 
  (hr : r = 0.10) 
  (hn : n = 1) 
  (ht : A = P * (1 + r / n) ^ (n * t)) :
  t = 2 :=
by
  sorry

end NUMINAMATH_GPT_investment_duration_l215_21523


namespace NUMINAMATH_GPT_rectangle_area_l215_21513

-- Conditions: 
-- 1. The length of the rectangle is three times its width.
-- 2. The diagonal length of the rectangle is x.

theorem rectangle_area (x : ℝ) (w l : ℝ) (h1 : w * 3 = l) (h2 : w^2 + l^2 = x^2) :
  l * w = (3 / 10) * x^2 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l215_21513


namespace NUMINAMATH_GPT_april_roses_l215_21508

theorem april_roses (price_per_rose earnings roses_left : ℤ) 
  (h1 : price_per_rose = 4)
  (h2 : earnings = 36)
  (h3 : roses_left = 4) :
  4 + (earnings / price_per_rose) = 13 :=
by
  sorry

end NUMINAMATH_GPT_april_roses_l215_21508


namespace NUMINAMATH_GPT_geometric_sequence_first_term_l215_21570

-- Define factorial values for convenience
def fact (n : ℕ) : ℕ := Nat.factorial n
#eval fact 6 -- This should give us 720
#eval fact 7 -- This should give us 5040

-- State the hypotheses and the goal
theorem geometric_sequence_first_term (a r : ℝ)
  (h1 : a * r^2 = 720)
  (h2 : a * r^5 = 5040) :
  a = 720 / (7^(2/3 : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_l215_21570


namespace NUMINAMATH_GPT_marbles_total_l215_21516

-- Conditions
variables (T : ℕ) -- Total number of marbles
variables (h_red : T ≥ 12) -- At least 12 red marbles
variables (h_blue : T ≥ 8) -- At least 8 blue marbles
variables (h_prob : (T - 12 : ℚ) / T = (3 / 4 : ℚ)) -- Probability condition

-- Proof statement
theorem marbles_total : T = 48 :=
by
  -- Proof here
  sorry

end NUMINAMATH_GPT_marbles_total_l215_21516


namespace NUMINAMATH_GPT_taehyung_collected_most_points_l215_21504

def largest_collector : Prop :=
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  taehyung_points > yoongi_points ∧ 
  taehyung_points > jungkook_points ∧ 
  taehyung_points > yuna_points ∧ 
  taehyung_points > yoojung_points

theorem taehyung_collected_most_points : largest_collector :=
by
  let yoongi_points := 7
  let jungkook_points := 6
  let yuna_points := 9
  let yoojung_points := 8
  let taehyung_points := 10
  sorry

end NUMINAMATH_GPT_taehyung_collected_most_points_l215_21504


namespace NUMINAMATH_GPT_graph_symmetric_about_x_eq_pi_div_8_l215_21578

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos x * (Real.sin x + Real.cos x)

theorem graph_symmetric_about_x_eq_pi_div_8 :
  ∀ x, f (π / 8 - x) = f (π / 8 + x) :=
sorry

end NUMINAMATH_GPT_graph_symmetric_about_x_eq_pi_div_8_l215_21578


namespace NUMINAMATH_GPT_range_of_b_l215_21562

noncomputable def a_n (n : ℕ) (b : ℝ) : ℝ := n^2 + b * n

theorem range_of_b (b : ℝ) : (∀ n : ℕ, 0 < n → a_n (n+1) b > a_n n b) ↔ (-3 < b) :=
by
    sorry

end NUMINAMATH_GPT_range_of_b_l215_21562


namespace NUMINAMATH_GPT_greatest_possible_value_of_y_l215_21536

-- Definitions according to problem conditions
variables {x y : ℤ}

-- The theorem statement to prove
theorem greatest_possible_value_of_y (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_y_l215_21536


namespace NUMINAMATH_GPT_right_triangle_hypotenuse_l215_21551

theorem right_triangle_hypotenuse (a b : ℝ) (h : a^2 + b^2 = 39^2) : a = 15 ∧ b = 36 := by
  sorry

end NUMINAMATH_GPT_right_triangle_hypotenuse_l215_21551


namespace NUMINAMATH_GPT_work_done_time_l215_21538

/-
  Question: How many days does it take for \(a\) to do the work alone?

  Conditions:
  - \(b\) can do the work in 20 days.
  - \(c\) can do the work in 55 days.
  - \(a\) is assisted by \(b\) and \(c\) on alternate days, and the work can be done in 8 days.
  
  Correct Answer:
  - \(x = 8.8\)
-/

theorem work_done_time (x : ℝ) (h : 8 * x⁻¹ + 1 /  5 + 4 / 55 = 1): x = 8.8 :=
by sorry

end NUMINAMATH_GPT_work_done_time_l215_21538


namespace NUMINAMATH_GPT_intersection_M_N_l215_21559

def U : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def complement_U_N : Set ℝ := {x | 0 < x ∧ x < 2}
def N : Set ℝ := {x | (x ∈ U) ∧ ¬(x ∈ complement_U_N)}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x ≤ 0} :=
sorry

end NUMINAMATH_GPT_intersection_M_N_l215_21559


namespace NUMINAMATH_GPT_solve_for_a_l215_21518

theorem solve_for_a {f : ℝ → ℝ} (h1 : ∀ x : ℝ, f (2 * x + 1) = 3 * x - 2) (h2 : f a = 7) : a = 7 :=
sorry

end NUMINAMATH_GPT_solve_for_a_l215_21518


namespace NUMINAMATH_GPT_vec_addition_l215_21561

namespace VectorCalculation

open Real

def v1 : ℤ × ℤ := (3, -8)
def v2 : ℤ × ℤ := (2, -6)
def scalar : ℤ := 5

def scaled_v2 : ℤ × ℤ := (scalar * v2.1, scalar * v2.2)
def result : ℤ × ℤ := (v1.1 + scaled_v2.1, v1.2 + scaled_v2.2)

theorem vec_addition : result = (13, -38) := by
  sorry

end VectorCalculation

end NUMINAMATH_GPT_vec_addition_l215_21561


namespace NUMINAMATH_GPT_find_weight_A_l215_21592

noncomputable def weight_of_A (a b c d e : ℕ) : Prop :=
  (a + b + c) / 3 = 84 ∧
  (a + b + c + d) / 4 = 80 ∧
  e = d + 5 ∧
  (b + c + d + e) / 4 = 79 →
  a = 77

theorem find_weight_A (a b c d e : ℕ) : weight_of_A a b c d e :=
by
  sorry

end NUMINAMATH_GPT_find_weight_A_l215_21592


namespace NUMINAMATH_GPT_wilted_flowers_are_18_l215_21585

def picked_flowers := 53
def flowers_per_bouquet := 7
def bouquets_after_wilted := 5

def flowers_left := bouquets_after_wilted * flowers_per_bouquet
def flowers_wilted : ℕ := picked_flowers - flowers_left

theorem wilted_flowers_are_18 : flowers_wilted = 18 := by
  sorry

end NUMINAMATH_GPT_wilted_flowers_are_18_l215_21585


namespace NUMINAMATH_GPT_simplify_expression_l215_21521

theorem simplify_expression (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  ((x^3 + 2) / x * (y^3 + 2) / y) - ((x^3 - 2) / y * (y^3 - 2) / x) = 4 * (x^2 / y + y^2 / x) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l215_21521


namespace NUMINAMATH_GPT_necessary_not_sufficient_condition_l215_21544

theorem necessary_not_sufficient_condition (m : ℝ) 
  (h : 2 < m ∧ m < 6) :
  (∃ (x y : ℝ), (x^2 / (m - 2) + y^2 / (6 - m) = 1)) ∧ (∀ m', 2 < m' ∧ m' < 6 → ∃ (x' y' : ℝ), (x'^2 / (m' - 2) + y'^2 / (6 - m') = 1) ∧ m' ≠ 4) :=
by
  sorry

end NUMINAMATH_GPT_necessary_not_sufficient_condition_l215_21544


namespace NUMINAMATH_GPT_average_percentage_25_students_l215_21574

theorem average_percentage_25_students (s1 s2 : ℕ) (p1 p2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : p1 = 75) (h3 : s2 = 10) (h4 : p2 = 95) (h5 : n = 25) :
  ((s1 * p1 + s2 * p2) / n) = 83 := 
by
  sorry

end NUMINAMATH_GPT_average_percentage_25_students_l215_21574


namespace NUMINAMATH_GPT_min_value_m_n_l215_21580

noncomputable def log_a (a x : ℝ) := Real.log x / Real.log a

theorem min_value_m_n 
  (a : ℝ) (m n : ℝ)
  (h_a_pos : a > 0) (h_a_ne1 : a ≠ 1)
  (h_mn_pos : m > 0 ∧ n > 0)
  (h_line_eq : 2 * m + n = 1) :
  m + n = 3 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_m_n_l215_21580


namespace NUMINAMATH_GPT_initial_value_subtract_perfect_square_l215_21581

theorem initial_value_subtract_perfect_square :
  ∃ n : ℕ, n^2 = 308 - 139 :=
by
  sorry

end NUMINAMATH_GPT_initial_value_subtract_perfect_square_l215_21581


namespace NUMINAMATH_GPT_minimum_value_a_plus_b_plus_c_l215_21541

theorem minimum_value_a_plus_b_plus_c (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 2 * a + 4 * b + 7 * c ≤ 2 * a * b * c) : a + b + c ≥ 15 / 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_a_plus_b_plus_c_l215_21541


namespace NUMINAMATH_GPT_train_speed_l215_21582

noncomputable def original_speed_of_train (v d : ℝ) : Prop :=
  (120 ≤ v / (5/7)) ∧
  (2 * d) / (5 * v) = 65 / 60 ∧
  (2 * (d - 42)) / (5 * v) = 45 / 60

theorem train_speed (v d : ℝ) (h : original_speed_of_train v d) : v = 50.4 :=
by sorry

end NUMINAMATH_GPT_train_speed_l215_21582


namespace NUMINAMATH_GPT_log_base_30_of_8_l215_21543

theorem log_base_30_of_8 (a b : ℝ) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
  Real.logb 30 8 = (3 * (1 - a)) / (1 + b) :=
by
  sorry

end NUMINAMATH_GPT_log_base_30_of_8_l215_21543


namespace NUMINAMATH_GPT_meal_cost_l215_21553

theorem meal_cost (x : ℝ) (h1 : ∀ (x : ℝ), (x / 4) - 6 = x / 9) : 
  x = 43.2 :=
by
  have h : (∀ (x : ℝ), (x / 4) - (x / 9) = 6) := sorry
  exact sorry

end NUMINAMATH_GPT_meal_cost_l215_21553


namespace NUMINAMATH_GPT_sequence_to_geometric_l215_21590

variable (a : ℕ → ℝ)

def seq_geom (a : ℕ → ℝ) : Prop :=
∀ m n, a (m + n) = a m * a n

def condition (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 2) = a n * a (n + 1)

theorem sequence_to_geometric (a1 a2 : ℝ) (h1 : a 1 = a1) (h2 : a 2 = a2) (h : ∀ n, a (n + 2) = a n * a (n + 1)) :
  a1 = 1 → a2 = 1 → seq_geom a :=
by
  intros ha1 ha2
  have h_seq : ∀ n, a n = 1 := sorry
  intros m n
  sorry

end NUMINAMATH_GPT_sequence_to_geometric_l215_21590


namespace NUMINAMATH_GPT_Mark_has_23_kangaroos_l215_21547

theorem Mark_has_23_kangaroos :
  ∃ K G : ℕ, G = 3 * K ∧ 2 * K + 4 * G = 322 ∧ K = 23 :=
by
  sorry

end NUMINAMATH_GPT_Mark_has_23_kangaroos_l215_21547


namespace NUMINAMATH_GPT_lisa_earns_more_than_tommy_l215_21526

theorem lisa_earns_more_than_tommy {total_earnings : ℤ} (h1 : total_earnings = 60) :
  let lisa_earnings := total_earnings / 2
  let tommy_earnings := lisa_earnings / 2
  lisa_earnings - tommy_earnings = 15 :=
by
  sorry

end NUMINAMATH_GPT_lisa_earns_more_than_tommy_l215_21526


namespace NUMINAMATH_GPT_train_length_is_250_l215_21545

noncomputable def train_length (V₁ V₂ V₃ : ℕ) (T₁ T₂ T₃ : ℕ) : ℕ :=
  let S₁ := (V₁ * (5/18) * T₁)
  let S₂ := (V₂ * (5/18)* T₂)
  let S₃ := (V₃ * (5/18) * T₃)
  if S₁ = S₂ ∧ S₂ = S₃ then S₁ else 0

theorem train_length_is_250 :
  train_length 50 60 70 18 20 22 = 250 := by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_train_length_is_250_l215_21545


namespace NUMINAMATH_GPT_scientific_notation_of_3395000_l215_21597

theorem scientific_notation_of_3395000 :
  3395000 = 3.395 * 10^6 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_3395000_l215_21597


namespace NUMINAMATH_GPT_exists_primes_sum_2024_with_one_gt_1000_l215_21591

open Nat

-- Definition of primality
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Conditions given in the problem
def sum_primes_eq_2024 (p q : ℕ) : Prop :=
  p + q = 2024 ∧ is_prime p ∧ is_prime q

def at_least_one_gt_1000 (p q : ℕ) : Prop :=
  p > 1000 ∨ q > 1000

-- The theorem to be proved
theorem exists_primes_sum_2024_with_one_gt_1000 :
  ∃ (p q : ℕ), sum_primes_eq_2024 p q ∧ at_least_one_gt_1000 p q :=
sorry

end NUMINAMATH_GPT_exists_primes_sum_2024_with_one_gt_1000_l215_21591


namespace NUMINAMATH_GPT_lcm_12_18_l215_21546

theorem lcm_12_18 : Nat.lcm 12 18 = 36 := by
  sorry

end NUMINAMATH_GPT_lcm_12_18_l215_21546


namespace NUMINAMATH_GPT_seventeen_divides_9x_plus_5y_l215_21540

theorem seventeen_divides_9x_plus_5y (x y : ℤ) (h : 17 ∣ (2 * x + 3 * y)) : 17 ∣ (9 * x + 5 * y) :=
sorry

end NUMINAMATH_GPT_seventeen_divides_9x_plus_5y_l215_21540


namespace NUMINAMATH_GPT_correct_statement_c_l215_21586

theorem correct_statement_c (five_boys_two_girls : Nat := 7) (select_three : Nat := 3) :
  (∃ boys girls : Nat, boys + girls = five_boys_two_girls ∧ boys = 5 ∧ girls = 2) →
  (∃ selected_boys selected_girls : Nat, selected_boys + selected_girls = select_three ∧ selected_boys > 0) :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_c_l215_21586


namespace NUMINAMATH_GPT_mirasol_initial_amount_l215_21555

/-- 
Mirasol had some money in her account. She spent $10 on coffee beans and $30 on a tumbler. She has $10 left in her account.
Prove that the initial amount of money Mirasol had in her account is $50.
-/
theorem mirasol_initial_amount (spent_coffee : ℕ) (spent_tumbler : ℕ) (left_in_account : ℕ) :
  spent_coffee = 10 → spent_tumbler = 30 → left_in_account = 10 → 
  spent_coffee + spent_tumbler + left_in_account = 50 := 
by
  sorry

end NUMINAMATH_GPT_mirasol_initial_amount_l215_21555


namespace NUMINAMATH_GPT_greatest_value_NNM_l215_21542

theorem greatest_value_NNM :
  ∃ (M : ℕ), (M * M % 10 = M) ∧ (∃ (MM : ℕ), MM = 11 * M ∧ (MM * M = 396)) :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_NNM_l215_21542


namespace NUMINAMATH_GPT_q_simplified_l215_21575

noncomputable def q (a b c x : ℝ) : ℝ :=
  (x + a)^4 / ((a - b) * (a - c)) +
  (x + b)^4 / ((b - a) * (b - c)) +
  (x + c)^4 / ((c - a) * (c - b)) - 3 * x * (
      1 / ((a - b) * (a - c)) + 
      1 / ((b - a) * (b - c)) +
      1 / ((c - a) * (c - b))
  )

theorem q_simplified (a b c x : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) :
  q a b c x = a^2 + b^2 + c^2 + 4*x^2 - 4*(a + b + c)*x + 12*x :=
sorry

end NUMINAMATH_GPT_q_simplified_l215_21575


namespace NUMINAMATH_GPT_dave_total_earnings_l215_21503

def hourly_wage (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 7 else
  if day = 2 then 9 else
  if day = 3 then 8 else 
  0

def hours_worked (day : ℕ) : ℝ :=
  if day = 0 then 6 else
  if day = 1 then 2 else
  if day = 2 then 3 else
  if day = 3 then 5 else 
  0

def unpaid_break (day : ℕ) : ℝ :=
  if day = 0 then 0.5 else
  if day = 1 then 0.25 else
  if day = 2 then 0 else
  if day = 3 then 0.5 else 
  0

def daily_earnings (day : ℕ) : ℝ :=
  (hours_worked day - unpaid_break day) * hourly_wage day

def net_earnings (day : ℕ) : ℝ :=
  daily_earnings day - (daily_earnings day * 0.1)

def total_net_earnings : ℝ :=
  net_earnings 0 + net_earnings 1 + net_earnings 2 + net_earnings 3

theorem dave_total_earnings : total_net_earnings = 97.43 := by
  sorry

end NUMINAMATH_GPT_dave_total_earnings_l215_21503


namespace NUMINAMATH_GPT_youseff_time_difference_l215_21579

noncomputable def walking_time (blocks : ℕ) (time_per_block : ℕ) : ℕ := blocks * time_per_block
noncomputable def biking_time (blocks : ℕ) (time_per_block_seconds : ℕ) : ℕ := (blocks * time_per_block_seconds) / 60

theorem youseff_time_difference : walking_time 6 1 - biking_time 6 20 = 4 := by
  sorry

end NUMINAMATH_GPT_youseff_time_difference_l215_21579


namespace NUMINAMATH_GPT_value_of_a4_l215_21509

theorem value_of_a4 (a : ℕ → ℕ) (r : ℕ) (h1 : ∀ n, a (n+1) = r * a n) (h2 : a 4 / a 2 - a 3 = 0) (h3 : r = 2) :
  a 4 = 8 :=
sorry

end NUMINAMATH_GPT_value_of_a4_l215_21509


namespace NUMINAMATH_GPT_sum_of_three_numbers_l215_21500

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) 
 (h_median : b = 10) 
 (h_mean_least : (a + b + c) / 3 = a + 8)
 (h_mean_greatest : (a + b + c) / 3 = c - 20) : 
 a + b + c = 66 :=
by 
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l215_21500


namespace NUMINAMATH_GPT_player_A_wins_l215_21524

theorem player_A_wins (n : ℕ) : ∃ m, (m > 2 * n^2) ∧ (∀ S : Finset (ℕ × ℕ), S.card = m → ∃ (r c : Finset ℕ), r.card = n ∧ c.card = n ∧ ∀ rc ∈ r.product c, rc ∈ S → false) :=
by sorry

end NUMINAMATH_GPT_player_A_wins_l215_21524


namespace NUMINAMATH_GPT_total_profit_l215_21529

theorem total_profit (a_cap b_cap : ℝ) (a_profit : ℝ) (a_share b_share : ℝ) (P : ℝ) :
  a_cap = 15000 ∧ b_cap = 25000 ∧ a_share = 0.10 ∧ a_profit = 4200 →
  a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit →
  P = 9600 :=
by
  intros h1 h2
  have h3 : a_share * P + (a_cap / (a_cap + b_cap)) * (1 - a_share) * P = a_profit := h2
  sorry

end NUMINAMATH_GPT_total_profit_l215_21529


namespace NUMINAMATH_GPT_maximal_subset_with_property_A_l215_21576

-- Define property A for a subset S ⊆ {0, 1, 2, ..., 99}
def has_property_A (S : Finset ℕ) : Prop := 
  ∀ a b c : ℕ, (a * 10 + b ∈ S) → (b * 10 + c ∈ S) → False

-- Define the set of integers {0, 1, 2, ..., 99}
def numbers_set := Finset.range 100

-- The main statement to be proven
theorem maximal_subset_with_property_A :
  ∃ S : Finset ℕ, S ⊆ numbers_set ∧ has_property_A S ∧ S.card = 25 := 
sorry

end NUMINAMATH_GPT_maximal_subset_with_property_A_l215_21576


namespace NUMINAMATH_GPT_bag_ratio_l215_21535

noncomputable def ratio_of_costs : ℚ := 1 / 2

theorem bag_ratio :
  ∃ (shirt_cost shoes_cost total_cost bag_cost : ℚ),
    shirt_cost = 7 ∧
    shoes_cost = shirt_cost + 3 ∧
    total_cost = 2 * shirt_cost + shoes_cost ∧
    bag_cost = 36 - total_cost ∧
    bag_cost / total_cost = ratio_of_costs :=
sorry

end NUMINAMATH_GPT_bag_ratio_l215_21535


namespace NUMINAMATH_GPT_equivalence_of_statements_l215_21593

-- Variables used in the statements
variable (P Q : Prop)

-- Proof problem statement
theorem equivalence_of_statements : (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by sorry

end NUMINAMATH_GPT_equivalence_of_statements_l215_21593


namespace NUMINAMATH_GPT_total_selling_price_l215_21564

theorem total_selling_price (cost_price_per_metre profit_per_metre : ℝ)
  (total_metres_sold : ℕ) :
  cost_price_per_metre = 58.02564102564102 → 
  profit_per_metre = 29 → 
  total_metres_sold = 78 →
  (cost_price_per_metre + profit_per_metre) * total_metres_sold = 6788 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  -- backend calculation, checking computation level;
  sorry

end NUMINAMATH_GPT_total_selling_price_l215_21564


namespace NUMINAMATH_GPT_craig_total_commission_correct_l215_21565

-- Define the commission structures
def refrigerator_commission (price : ℝ) : ℝ := 75 + 0.08 * price
def washing_machine_commission (price : ℝ) : ℝ := 50 + 0.10 * price
def oven_commission (price : ℝ) : ℝ := 60 + 0.12 * price

-- Define total sales
def total_refrigerator_sales : ℝ := 5280
def total_washing_machine_sales : ℝ := 2140
def total_oven_sales : ℝ := 4620

-- Define number of appliances sold
def number_of_refrigerators : ℝ := 3
def number_of_washing_machines : ℝ := 4
def number_of_ovens : ℝ := 5

-- Calculate total commissions for each appliance category
def total_refrigerator_commission : ℝ := number_of_refrigerators * refrigerator_commission total_refrigerator_sales
def total_washing_machine_commission : ℝ := number_of_washing_machines * washing_machine_commission total_washing_machine_sales
def total_oven_commission : ℝ := number_of_ovens * oven_commission total_oven_sales

-- Calculate total commission for the week
def total_commission : ℝ := total_refrigerator_commission + total_washing_machine_commission + total_oven_commission

-- Prove that the total commission is as expected
theorem craig_total_commission_correct : total_commission = 5620.20 := 
by
  sorry

end NUMINAMATH_GPT_craig_total_commission_correct_l215_21565


namespace NUMINAMATH_GPT_connie_s_problem_l215_21571

theorem connie_s_problem (y : ℕ) (h : 3 * y = 90) : y / 3 = 10 :=
by
  sorry

end NUMINAMATH_GPT_connie_s_problem_l215_21571


namespace NUMINAMATH_GPT_num_squares_in_6_by_6_grid_l215_21530

def squares_in_grid (m n : ℕ) : ℕ :=
  (m - 1) * (m - 1) + (m - 2) * (m - 2) + 
  (m - 3) * (m - 3) + (m - 4) * (m - 4) + 
  (m - 5) * (m - 5)

theorem num_squares_in_6_by_6_grid : squares_in_grid 6 6 = 55 := 
by 
  sorry

end NUMINAMATH_GPT_num_squares_in_6_by_6_grid_l215_21530


namespace NUMINAMATH_GPT_find_ordered_pair_l215_21506

theorem find_ordered_pair : ∃ k a : ℤ, 
  (∀ x : ℝ, (x^3 - 4*x^2 + 9*x - 6) % (x^2 - x + k) = 2*x + a) ∧ k = 4 ∧ a = 6 :=
sorry

end NUMINAMATH_GPT_find_ordered_pair_l215_21506


namespace NUMINAMATH_GPT_yellow_balls_count_l215_21517

theorem yellow_balls_count (purple blue total_needed : ℕ) 
  (h_purple : purple = 7) 
  (h_blue : blue = 5) 
  (h_total : total_needed = 19) : 
  ∃ (yellow : ℕ), yellow = 6 :=
by
  sorry

end NUMINAMATH_GPT_yellow_balls_count_l215_21517


namespace NUMINAMATH_GPT_minimum_degree_g_l215_21589

-- Define the degree function for polynomials
noncomputable def degree (p : Polynomial ℤ) : ℕ := p.natDegree

-- Declare the variables and conditions for the proof
variables (f g h : Polynomial ℤ)
variables (deg_f : degree f = 10) (deg_h : degree h = 12)
variable (eqn : 2 * f + 5 * g = h)

-- State the main theorem for the problem
theorem minimum_degree_g : degree g ≥ 12 :=
    by sorry -- Proof to be provided

end NUMINAMATH_GPT_minimum_degree_g_l215_21589


namespace NUMINAMATH_GPT_nell_initial_cards_l215_21587

theorem nell_initial_cards (cards_given cards_left total_cards : ℕ)
  (h1 : cards_given = 301)
  (h2 : cards_left = 154)
  (h3 : total_cards = cards_given + cards_left) :
  total_cards = 455 := by
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_nell_initial_cards_l215_21587


namespace NUMINAMATH_GPT_spears_per_sapling_l215_21550

/-- Given that a log can produce 9 spears and 6 saplings plus a log produce 27 spears,
prove that a single sapling can produce 3 spears (S = 3). -/
theorem spears_per_sapling (L S : ℕ) (hL : L = 9) (h: 6 * S + L = 27) : S = 3 :=
by
  sorry

end NUMINAMATH_GPT_spears_per_sapling_l215_21550


namespace NUMINAMATH_GPT_geometric_progression_x_value_l215_21577

noncomputable def geometric_progression_solution (x : ℝ) : Prop :=
  let a := -30 + x
  let b := -10 + x
  let c := 40 + x
  b^2 = a * c

theorem geometric_progression_x_value :
  ∃ x : ℝ, geometric_progression_solution x ∧ x = 130 / 3 :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_x_value_l215_21577


namespace NUMINAMATH_GPT_train_speed_length_l215_21532

theorem train_speed_length (t1 t2 s : ℕ) (p : ℕ)
  (h1 : t1 = 7) 
  (h2 : t2 = 25) 
  (h3 : p = 378)
  (h4 : t2 - t1 = 18)
  (h5 : p / (t2 - t1) = 21) 
  (h6 : (p / (t2 - t1)) * t1 = 147) :
  (21, 147) = (21, 147) :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_length_l215_21532


namespace NUMINAMATH_GPT_jacoby_lottery_expense_l215_21573

-- Definitions based on the conditions:
def jacoby_trip_fund_needed : ℕ := 5000
def jacoby_hourly_wage : ℕ := 20
def jacoby_work_hours : ℕ := 10
def cookies_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2
def money_still_needed : ℕ := 3214

-- The statement to prove:
theorem jacoby_lottery_expense : 
  (jacoby_hourly_wage * jacoby_work_hours) + (cookies_price * cookies_sold) +
  lottery_winnings + (sister_gift * num_sisters) 
  - (jacoby_trip_fund_needed - money_still_needed) = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_jacoby_lottery_expense_l215_21573


namespace NUMINAMATH_GPT_squared_sum_inverse_l215_21594

theorem squared_sum_inverse (x : ℝ) (h : x + 1/x = 2) : x^2 + 1/x^2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_squared_sum_inverse_l215_21594


namespace NUMINAMATH_GPT_smallest_n_condition_l215_21537

open Nat

-- Define the sum of squares formula
noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- Define the condition for being a square number
def is_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The proof problem statement
theorem smallest_n_condition : 
  ∃ n : ℕ, n > 1 ∧ is_square (sum_of_squares n / n) ∧ (∀ m : ℕ, m > 1 ∧ is_square (sum_of_squares m / m) → n ≤ m) :=
sorry

end NUMINAMATH_GPT_smallest_n_condition_l215_21537


namespace NUMINAMATH_GPT_find_a2023_l215_21510

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∃ a1 : ℤ, ∀ n : ℕ, a n = a1 + n * d

theorem find_a2023 (a : ℕ → ℤ) (h_arith : arithmetic_sequence a)
  (h_cond1 : a 2 + a 7 = a 8 + 1)
  (h_cond2 : (a 4)^2 = a 2 * a 8) :
  a 2023 = 2023 := 
sorry

end NUMINAMATH_GPT_find_a2023_l215_21510


namespace NUMINAMATH_GPT_Christine_picked_10_pounds_l215_21505

-- Variable declarations for the quantities involved
variable (C : ℝ) -- Pounds of strawberries Christine picked
variable (pieStrawberries : ℝ := 3) -- Pounds of strawberries per pie
variable (pies : ℝ := 10) -- Number of pies
variable (totalStrawberries : ℝ := 30) -- Total pounds of strawberries for pies

-- The condition that Rachel picked twice as many strawberries as Christine
variable (R : ℝ := 2 * C)

-- The condition for the total pounds of strawberries picked by Christine and Rachel
axiom strawberries_eq : C + R = totalStrawberries

-- The goal is to prove that Christine picked 10 pounds of strawberries
theorem Christine_picked_10_pounds : C = 10 := by
  sorry

end NUMINAMATH_GPT_Christine_picked_10_pounds_l215_21505


namespace NUMINAMATH_GPT_find_A_l215_21507

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) 
(h_div9 : (A + 1 + 5 + B + 9 + 4) % 9 = 0) 
(h_div11 : (A + 5 + 9 - (1 + B + 4)) % 11 = 0) : A = 5 :=
by sorry

end NUMINAMATH_GPT_find_A_l215_21507


namespace NUMINAMATH_GPT_largest_y_coordinate_ellipse_l215_21599

theorem largest_y_coordinate_ellipse:
  (∀ x y : ℝ, (x^2 / 49) + ((y + 3)^2 / 25) = 1 → y ≤ 2)  ∧ 
  (∃ x : ℝ, (x^2 / 49) + ((2 + 3)^2 / 25) = 1) := sorry

end NUMINAMATH_GPT_largest_y_coordinate_ellipse_l215_21599


namespace NUMINAMATH_GPT_remainder_86592_8_remainder_8741_13_l215_21520

theorem remainder_86592_8 :
  86592 % 8 = 0 :=
by
  sorry

theorem remainder_8741_13 :
  8741 % 13 = 5 :=
by
  sorry

end NUMINAMATH_GPT_remainder_86592_8_remainder_8741_13_l215_21520


namespace NUMINAMATH_GPT_highest_score_batsman_l215_21549

variable (H L : ℕ)

theorem highest_score_batsman :
  (60 * 46) = (58 * 44 + H + L) ∧ (H - L = 190) → H = 199 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_highest_score_batsman_l215_21549


namespace NUMINAMATH_GPT_rectangle_area_is_30_l215_21572

def Point := (ℤ × ℤ)

def vertices : List Point := [(-5, 1), (1, 1), (1, -4), (-5, -4)]

theorem rectangle_area_is_30 :
  let length := (vertices[1].1 - vertices[0].1).natAbs
  let width := (vertices[0].2 - vertices[2].2).natAbs
  length * width = 30 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_is_30_l215_21572


namespace NUMINAMATH_GPT_relationship_of_a_b_c_l215_21539

noncomputable def a : ℝ := Real.log 3 / Real.log 2  -- a = log2(1/3)
noncomputable def b : ℝ := Real.exp (1 / 3)  -- b = e^(1/3)
noncomputable def c : ℝ := 1 / 3  -- c = e^ln(1/3) = 1/3

theorem relationship_of_a_b_c : b > c ∧ c > a :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_relationship_of_a_b_c_l215_21539


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l215_21596

theorem geometric_sequence_common_ratio (a : ℕ → ℚ) (q : ℚ) :
  (∀ n, a n = a 2 * q ^ (n - 2)) ∧ a 2 = 2 ∧ a 6 = 1 / 8 →
  (q = 1 / 2 ∨ q = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l215_21596


namespace NUMINAMATH_GPT_triangle_shape_l215_21567

theorem triangle_shape (a b : ℝ) (A B : ℝ)
  (h1 : a ≠ 0) (h2 : b ≠ 0) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π)
  (hTriangle : A + B + (π - A - B) = π)
  (h : a * Real.cos A = b * Real.cos B) : 
  (A = B ∨ A + B = π / 2) := sorry

end NUMINAMATH_GPT_triangle_shape_l215_21567


namespace NUMINAMATH_GPT_units_digit_6_power_l215_21533

theorem units_digit_6_power (n : ℕ) : (6^n % 10) = 6 :=
sorry

end NUMINAMATH_GPT_units_digit_6_power_l215_21533


namespace NUMINAMATH_GPT_coastal_village_population_l215_21569

variable (N : ℕ) (k : ℕ) (parts_for_males : ℕ) (total_males : ℕ)

theorem coastal_village_population 
  (h_total_population : N = 540)
  (h_division : k = 4)
  (h_parts_for_males : parts_for_males = 2)
  (h_total_males : total_males = (N / k) * parts_for_males) :
  total_males = 270 := 
by
  sorry

end NUMINAMATH_GPT_coastal_village_population_l215_21569


namespace NUMINAMATH_GPT_ratio_S3_S9_l215_21527

noncomputable def Sn (a r : ℝ) (n : ℕ) : ℝ := (a * (1 - r ^ n)) / (1 - r)

theorem ratio_S3_S9 (a r : ℝ) (h1 : r ≠ 1) (h2 : Sn a r 6 = 3 * Sn a r 3) :
  Sn a r 3 / Sn a r 9 = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_ratio_S3_S9_l215_21527


namespace NUMINAMATH_GPT_interest_difference_correct_l215_21598

-- Define the basic parameters and constants
def principal : ℝ := 147.69
def rate : ℝ := 0.15
def time1 : ℝ := 3.5
def time2 : ℝ := 10
def interest1 : ℝ := principal * rate * time1
def interest2 : ℝ := principal * rate * time2
def difference : ℝ := 143.998

-- Theorem statement: The difference between the interests is approximately Rs. 143.998
theorem interest_difference_correct :
  interest2 - interest1 = difference := sorry

end NUMINAMATH_GPT_interest_difference_correct_l215_21598


namespace NUMINAMATH_GPT_find_value_of_expression_l215_21515

theorem find_value_of_expression
  (a b c m : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = m)
  (h5 : a^2 + b^2 + c^2 = m^2 / 2) :
  (a * (m - 2 * a)^2 + b * (m - 2 * b)^2 + c * (m - 2 * c)^2) / (a * b * c) = 12 := 
sorry

end NUMINAMATH_GPT_find_value_of_expression_l215_21515


namespace NUMINAMATH_GPT_tv_weight_difference_l215_21522

-- Definitions for the given conditions
def bill_tv_length : ℕ := 48
def bill_tv_width : ℕ := 100
def bob_tv_length : ℕ := 70
def bob_tv_width : ℕ := 60
def weight_per_square_inch : ℕ := 4
def ounces_per_pound : ℕ := 16

-- The statement to prove
theorem tv_weight_difference : (bill_tv_length * bill_tv_width * weight_per_square_inch)
                               - (bob_tv_length * bob_tv_width * weight_per_square_inch)
                               = 150 * ounces_per_pound := by
  sorry

end NUMINAMATH_GPT_tv_weight_difference_l215_21522


namespace NUMINAMATH_GPT_sum_of_areas_of_squares_l215_21548

theorem sum_of_areas_of_squares (a b x : ℕ) 
  (h_overlapping_min : 9 ≤ (min a b) ^ 2)
  (h_overlapping_max : (min a b) ^ 2 ≤ 25)
  (h_sum_of_sides : a + b + x = 23) :
  a^2 + b^2 + x^2 = 189 := 
sorry

end NUMINAMATH_GPT_sum_of_areas_of_squares_l215_21548


namespace NUMINAMATH_GPT_part1_part2_l215_21519

noncomputable def f (a : ℝ) (a_pos : a > 1) (x : ℝ) : ℝ :=
  a^x + (x - 2) / (x + 1)

-- Statement for part 1
theorem part1 (a : ℝ) (a_pos : a > 1) : ∀ x : ℝ, -1 < x → f a a_pos x ≤ f a a_pos (x + ε) → 0 < ε := sorry

-- Statement for part 2
theorem part2 (a : ℝ) (a_pos : a > 1) : ¬ ∃ x : ℝ, x < 0 ∧ f a a_pos x = 0 := sorry

end NUMINAMATH_GPT_part1_part2_l215_21519


namespace NUMINAMATH_GPT_find_side_length_l215_21583

theorem find_side_length
  (a b : ℝ)
  (S : ℝ)
  (h1 : a = 4)
  (h2 : b = 5)
  (h3 : S = 5 * Real.sqrt 3) :
  ∃ c : ℝ, c = Real.sqrt 21 ∨ c = Real.sqrt 61 :=
by
  sorry

end NUMINAMATH_GPT_find_side_length_l215_21583


namespace NUMINAMATH_GPT_problem_statement_l215_21531

-- Defining the propositions p and q as Boolean variables
variables (p q : Prop)

-- Assume the given conditions
theorem problem_statement (hnp : ¬¬p) (hnpq : ¬(p ∧ q)) : p ∧ ¬q :=
by {
  -- Derived steps to satisfy the conditions are implicit within this scope
  sorry
}

end NUMINAMATH_GPT_problem_statement_l215_21531


namespace NUMINAMATH_GPT_gcd_factorial_l215_21558

theorem gcd_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 10 / Nat.factorial 4) = 5040 :=
by
  sorry

end NUMINAMATH_GPT_gcd_factorial_l215_21558


namespace NUMINAMATH_GPT_bob_paid_correctly_l215_21588

-- Define the variables involved
def alice_acorns : ℕ := 3600
def price_per_acorn : ℕ := 15
def multiplier : ℕ := 9
def total_amount_alice_paid : ℕ := alice_acorns * price_per_acorn

-- Define Bob's payment amount
def bob_payment : ℕ := total_amount_alice_paid / multiplier

-- The main theorem
theorem bob_paid_correctly : bob_payment = 6000 := by
  sorry

end NUMINAMATH_GPT_bob_paid_correctly_l215_21588


namespace NUMINAMATH_GPT_chickens_count_l215_21502

theorem chickens_count (rabbits frogs : ℕ) (h_rabbits : rabbits = 49) (h_frogs : frogs = 37) :
  ∃ (C : ℕ), frogs + C = rabbits + 9 ∧ C = 21 :=
by
  sorry

end NUMINAMATH_GPT_chickens_count_l215_21502


namespace NUMINAMATH_GPT_journey_length_l215_21563

theorem journey_length (speed time : ℝ) (portions_covered total_portions : ℕ)
  (h_speed : speed = 40) (h_time : time = 0.7) (h_portions_covered : portions_covered = 4) (h_total_portions : total_portions = 5) :
  (speed * time / portions_covered) * total_portions = 35 :=
by
  sorry

end NUMINAMATH_GPT_journey_length_l215_21563


namespace NUMINAMATH_GPT_sum_non_solutions_eq_neg21_l215_21514

theorem sum_non_solutions_eq_neg21
  (A B C : ℝ)
  (h1 : ∀ x, ∃ k : ℝ, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h2 : ∃ A B C, ∀ x, (x ≠ -C) ∧ (x ≠ -9) → (x + B) * (A * x + 36) = 3 * (x + C) * (x + 9))
  (h3 : ∃! x, (x + C) * (x + 9) = 0)
   :
  -9 + -12 = -21 := by sorry

end NUMINAMATH_GPT_sum_non_solutions_eq_neg21_l215_21514


namespace NUMINAMATH_GPT_geometric_sequence_a6_l215_21566

variable {a : ℕ → ℝ} (h_geo : ∀ n, a (n+1) / a n = a (n+2) / a (n+1))

theorem geometric_sequence_a6 (h5 : a 5 = 2) (h7 : a 7 = 8) : a 6 = 4 ∨ a 6 = -4 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a6_l215_21566


namespace NUMINAMATH_GPT_jim_gave_away_675_cards_l215_21525

def total_cards_gave_away
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  : ℕ :=
  (sets_to_brother + sets_to_sister + sets_to_friend) * cards_per_set

theorem jim_gave_away_675_cards
  (cards_per_set : ℕ)
  (sets_to_brother sets_to_sister sets_to_friend : ℕ)
  (h_brother : sets_to_brother = 15)
  (h_sister : sets_to_sister = 8)
  (h_friend : sets_to_friend = 4)
  (h_cards_per_set : cards_per_set = 25)
  : total_cards_gave_away cards_per_set sets_to_brother sets_to_sister sets_to_friend = 675 :=
by
  sorry

end NUMINAMATH_GPT_jim_gave_away_675_cards_l215_21525


namespace NUMINAMATH_GPT_can_be_divided_into_6_triangles_l215_21568

-- Define the initial rectangle dimensions
def initial_rectangle_length := 6
def initial_rectangle_width := 5

-- Define the cut-out rectangle dimensions
def cutout_rectangle_length := 2
def cutout_rectangle_width := 1

-- Total area before the cut-out
def total_area : Nat := initial_rectangle_length * initial_rectangle_width

-- Cut-out area
def cutout_area : Nat := cutout_rectangle_length * cutout_rectangle_width

-- Remaining area after the cut-out
def remaining_area : Nat := total_area - cutout_area

-- The statement to be proved
theorem can_be_divided_into_6_triangles :
  remaining_area = 28 → (∃ (triangles : List (Nat × Nat × Nat)), triangles.length = 6) :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_can_be_divided_into_6_triangles_l215_21568
