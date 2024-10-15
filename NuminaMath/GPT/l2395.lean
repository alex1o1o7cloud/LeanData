import Mathlib

namespace NUMINAMATH_GPT_cabbage_count_l2395_239563

theorem cabbage_count 
  (length : ℝ)
  (width : ℝ)
  (density : ℝ)
  (h_length : length = 16)
  (h_width : width = 12)
  (h_density : density = 9) : 
  length * width * density = 1728 := 
by
  rw [h_length, h_width, h_density]
  norm_num
  done

end NUMINAMATH_GPT_cabbage_count_l2395_239563


namespace NUMINAMATH_GPT_find_value_of_a_3m_2n_l2395_239582

variable {a : ℝ} {m n : ℕ}
axiom h1 : a ^ m = 2
axiom h2 : a ^ n = 5

theorem find_value_of_a_3m_2n : a ^ (3 * m - 2 * n) = 8 / 25 := by
  sorry

end NUMINAMATH_GPT_find_value_of_a_3m_2n_l2395_239582


namespace NUMINAMATH_GPT_words_memorized_l2395_239552

theorem words_memorized (x y z : ℕ) (h1 : x = 4 * (y + z) / 5) (h2 : x + y = 6 * z / 5) (h3 : 100 < x + y + z ∧ x + y + z < 200) : 
  x + y + z = 198 :=
by
  sorry

end NUMINAMATH_GPT_words_memorized_l2395_239552


namespace NUMINAMATH_GPT_cheese_pops_count_l2395_239541

-- Define the number of hotdogs, chicken nuggets, and total portions
def hotdogs : ℕ := 30
def chicken_nuggets : ℕ := 40
def total_portions : ℕ := 90

-- Define the number of bite-sized cheese pops
def cheese_pops : ℕ := total_portions - hotdogs - chicken_nuggets

-- Theorem to prove that the number of bite-sized cheese pops Andrew brought is 20
theorem cheese_pops_count :
  cheese_pops = 20 :=
by
  -- The following proof is omitted
  sorry

end NUMINAMATH_GPT_cheese_pops_count_l2395_239541


namespace NUMINAMATH_GPT_value_at_7_5_l2395_239574

def f : ℝ → ℝ := sorry

axiom odd_function (x : ℝ) : f (-x) = -f x
axiom periodic_function (x : ℝ) : f (x + 2) = -f x
axiom interval_condition (x : ℝ) (h : 0 ≤ x ∧ x ≤ 1) : f x = x

theorem value_at_7_5 : f 7.5 = -0.5 := by
  sorry

end NUMINAMATH_GPT_value_at_7_5_l2395_239574


namespace NUMINAMATH_GPT_order_of_a_b_c_l2395_239514

noncomputable def a : ℝ := Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.log 2 / Real.log 3
noncomputable def c : ℝ := (1 / 2) * (Real.log 5 / Real.log 2)

theorem order_of_a_b_c : a > c ∧ c > b :=
by
  -- proof here
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l2395_239514


namespace NUMINAMATH_GPT_weight_ratio_l2395_239598

noncomputable def students_weight : ℕ := 79
noncomputable def siblings_total_weight : ℕ := 116

theorem weight_ratio (S W : ℕ) (h1 : siblings_total_weight = S + W) (h2 : students_weight = S):
  (S - 5) / (siblings_total_weight - S) = 2 :=
by
  sorry

end NUMINAMATH_GPT_weight_ratio_l2395_239598


namespace NUMINAMATH_GPT_C_share_correct_l2395_239590

def investment_A := 27000
def investment_B := 72000
def investment_C := 81000
def total_profit := 80000

def gcd_investment : ℕ := Nat.gcd investment_A (Nat.gcd investment_B investment_C)
def ratio_A : ℕ := investment_A / gcd_investment
def ratio_B : ℕ := investment_B / gcd_investment
def ratio_C : ℕ := investment_C / gcd_investment
def total_parts : ℕ := ratio_A + ratio_B + ratio_C

def C_share : ℕ := (ratio_C / total_parts) * total_profit

theorem C_share_correct : C_share = 36000 := 
by sorry

end NUMINAMATH_GPT_C_share_correct_l2395_239590


namespace NUMINAMATH_GPT_largest_n_rational_sqrt_l2395_239575

theorem largest_n_rational_sqrt : ∃ n : ℕ, 
  (∀ k l : ℤ, k = Int.natAbs (Int.sqrt (n - 100)) ∧ l = Int.natAbs (Int.sqrt (n + 100)) → 
  k + l = 100) ∧ 
  (n = 2501) :=
by
  sorry

end NUMINAMATH_GPT_largest_n_rational_sqrt_l2395_239575


namespace NUMINAMATH_GPT_least_additional_squares_needed_for_symmetry_l2395_239509

-- Conditions
def grid_size : ℕ := 5
def initial_shaded_squares : List (ℕ × ℕ) := [(1, 5), (3, 3), (5, 1)]

-- Goal statement
theorem least_additional_squares_needed_for_symmetry
  (grid_size : ℕ)
  (initial_shaded_squares : List (ℕ × ℕ)) : 
  ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℕ), (x, y) ∈ initial_shaded_squares ∨ (grid_size - x + 1, y) ∈ initial_shaded_squares ∨ (x, grid_size - y + 1) ∈ initial_shaded_squares ∨ (grid_size - x + 1, grid_size - y + 1) ∈ initial_shaded_squares) :=
sorry

end NUMINAMATH_GPT_least_additional_squares_needed_for_symmetry_l2395_239509


namespace NUMINAMATH_GPT_box_volume_l2395_239530

theorem box_volume (L W H : ℝ) (h1 : L * W = 120) (h2 : W * H = 72) (h3 : L * H = 60) : L * W * H = 720 := 
by sorry

end NUMINAMATH_GPT_box_volume_l2395_239530


namespace NUMINAMATH_GPT_log_ab_eq_l2395_239562

-- Definition and conditions
variables (a b x : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hx : 0 < x)

-- The theorem to prove
theorem log_ab_eq (a b x : ℝ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) :
  Real.log (x) / Real.log (a * b) = (Real.log (x) / Real.log (a)) * (Real.log (x) / Real.log (b)) / ((Real.log (x) / Real.log (a)) + (Real.log (x) / Real.log (b))) :=
sorry

end NUMINAMATH_GPT_log_ab_eq_l2395_239562


namespace NUMINAMATH_GPT_students_with_grade_B_and_above_l2395_239578

theorem students_with_grade_B_and_above (total_students : ℕ) (percent_below_B : ℕ) 
(h1 : total_students = 60) (h2 : percent_below_B = 40) : 
(total_students * (100 - percent_below_B) / 100) = 36 := by
  sorry

end NUMINAMATH_GPT_students_with_grade_B_and_above_l2395_239578


namespace NUMINAMATH_GPT_circle_radius_6_l2395_239568

theorem circle_radius_6 (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 10*x + y^2 + 6*y - k = 0 ↔ (x + 5)^2 + (y + 3)^2 = 36) → k = 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_6_l2395_239568


namespace NUMINAMATH_GPT_no_55_rooms_l2395_239558

theorem no_55_rooms 
  (count_roses count_carnations count_chrysanthemums : ℕ)
  (rooms_with_CC rooms_with_CR rooms_with_HR : ℕ)
  (at_least_one_bouquet_in_each_room: ∀ (room: ℕ), room > 0)
  (total_rooms : ℕ)
  (h_bouquets : count_roses = 30 ∧ count_carnations = 20 ∧ count_chrysanthemums = 10)
  (h_overlap_conditions: rooms_with_CC = 2 ∧ rooms_with_CR = 3 ∧ rooms_with_HR = 4):
  (total_rooms != 55) :=
sorry

end NUMINAMATH_GPT_no_55_rooms_l2395_239558


namespace NUMINAMATH_GPT_factorize_quadratic_l2395_239554

theorem factorize_quadratic : ∀ x : ℝ, x^2 - 7*x + 10 = (x - 2)*(x - 5) :=
by
  sorry

end NUMINAMATH_GPT_factorize_quadratic_l2395_239554


namespace NUMINAMATH_GPT_gcd_888_1147_l2395_239588

theorem gcd_888_1147 : Nat.gcd 888 1147 = 37 := by
  sorry

end NUMINAMATH_GPT_gcd_888_1147_l2395_239588


namespace NUMINAMATH_GPT_minimum_value_l2395_239571

theorem minimum_value (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
    (h_condition : (1 / a) + (1 / b) + (1 / c) = 9) : 
    a^3 * b^2 * c ≥ 64 / 729 :=
sorry

end NUMINAMATH_GPT_minimum_value_l2395_239571


namespace NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2395_239545

theorem simplify_expr1 (a b : ℝ) : 6 * a^2 - 2 * a * b - 2 * (3 * a^2 - (1 / 2) * a * b) = -a * b :=
by
  sorry

theorem simplify_expr2 (t : ℝ) : -(t^2 - t - 1) + (2 * t^2 - 3 * t + 1) = t^2 - 2 * t + 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expr1_simplify_expr2_l2395_239545


namespace NUMINAMATH_GPT_max_value_of_E_l2395_239501

variable (a b c d : ℝ)

def E (a b c d : ℝ) : ℝ := a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_of_E :
  -5.5 ≤ a ∧ a ≤ 5.5 →
  -5.5 ≤ b ∧ b ≤ 5.5 →
  -5.5 ≤ c ∧ c ≤ 5.5 →
  -5.5 ≤ d ∧ d ≤ 5.5 →
  E a b c d ≤ 132 := by
  sorry

end NUMINAMATH_GPT_max_value_of_E_l2395_239501


namespace NUMINAMATH_GPT_proof_system_solution_l2395_239518

noncomputable def solve_system : Prop :=
  ∃ x y : ℚ, x + 4 * y = 14 ∧ (x - 3) / 4 - (y - 3) / 3 = 1 / 12 ∧ x = 3 ∧ y = 11 / 4

theorem proof_system_solution : solve_system :=
sorry

end NUMINAMATH_GPT_proof_system_solution_l2395_239518


namespace NUMINAMATH_GPT_common_difference_is_3_l2395_239560

variables {a : ℕ → ℝ} {d a1 : ℝ}

-- Define the arithmetic sequence
def arithmetic_sequence (a_n : ℕ → ℝ) (a1 d : ℝ) : Prop := 
  ∀ n, a_n n = a1 + (n - 1) * d

-- Conditions
def a2_eq : a 2 = 3 := sorry
def a5_eq : a 5 = 12 := sorry

-- Theorem to prove the common difference is 3
theorem common_difference_is_3 :
  ∀ {a : ℕ → ℝ} {a1 d : ℝ},
  (arithmetic_sequence a a1 d)
  → a 2 = 3 
  → a 5 = 12 
  → d = 3 :=
  by
  intros a a1 d h_seq h_a2 h_a5
  sorry

end NUMINAMATH_GPT_common_difference_is_3_l2395_239560


namespace NUMINAMATH_GPT_range_of_x_l2395_239580

theorem range_of_x (x : ℝ) : (4 : ℝ)^(2 * x - 1) > (1 / 2) ^ (-x - 4) → x > 2 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l2395_239580


namespace NUMINAMATH_GPT_last_digit_of_exponents_l2395_239521

theorem last_digit_of_exponents : 
  (∃k, 2011 = 4 * k + 3 ∧ 
         (2^2011 % 10 = 8) ∧ 
         (3^2011 % 10 = 7)) → 
  ((2^2011 + 3^2011) % 10 = 5) := 
by 
  sorry

end NUMINAMATH_GPT_last_digit_of_exponents_l2395_239521


namespace NUMINAMATH_GPT_min_value_geq_four_l2395_239565

noncomputable def min_value_expression (x y z : ℝ) : ℝ :=
  (x + y) / (x * y * z)

theorem min_value_geq_four (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) :
  4 ≤ min_value_expression x y z :=
sorry

end NUMINAMATH_GPT_min_value_geq_four_l2395_239565


namespace NUMINAMATH_GPT_expression_value_l2395_239535

theorem expression_value :
  (2^1006 + 5^1007)^2 - (2^1006 - 5^1007)^2 = 40 * 10^1006 :=
by sorry

end NUMINAMATH_GPT_expression_value_l2395_239535


namespace NUMINAMATH_GPT_general_formula_a_sum_bn_l2395_239502

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

end NUMINAMATH_GPT_general_formula_a_sum_bn_l2395_239502


namespace NUMINAMATH_GPT_tangent_and_normal_lines_l2395_239520

noncomputable def x (t : ℝ) := 2 * Real.exp t
noncomputable def y (t : ℝ) := Real.exp (-t)

theorem tangent_and_normal_lines (t0 : ℝ) (x0 y0 : ℝ) (m_tangent m_normal : ℝ)
  (hx0 : x0 = x t0)
  (hy0 : y0 = y t0)
  (hm_tangent : m_tangent = -(1 / 2))
  (hm_normal : m_normal = 2) :
  (∀ x y : ℝ, y = m_tangent * x + 2) ∧ (∀ x y : ℝ, y = m_normal * x - 3) :=
by
  sorry

end NUMINAMATH_GPT_tangent_and_normal_lines_l2395_239520


namespace NUMINAMATH_GPT_time_per_step_l2395_239546

def apply_and_dry_time (total_time steps : ℕ) : ℕ :=
  total_time / steps

theorem time_per_step : apply_and_dry_time 120 6 = 20 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_time_per_step_l2395_239546


namespace NUMINAMATH_GPT_max_value_xyz_l2395_239511

theorem max_value_xyz (x y z : ℝ) (h : x + y + 2 * z = 5) : 
  (∃ x y z : ℝ, x + y + 2 * z = 5 ∧ xy + xz + yz = 25/6) :=
sorry

end NUMINAMATH_GPT_max_value_xyz_l2395_239511


namespace NUMINAMATH_GPT_cid_earnings_l2395_239595

variable (x : ℕ)
variable (oil_change_price repair_price car_wash_price : ℕ)
variable (cars_repaired cars_washed total_earnings : ℕ)

theorem cid_earnings :
  (oil_change_price = 20) →
  (repair_price = 30) →
  (car_wash_price = 5) →
  (cars_repaired = 10) →
  (cars_washed = 15) →
  (total_earnings = 475) →
  (oil_change_price * x + repair_price * cars_repaired + car_wash_price * cars_washed = total_earnings) →
  x = 5 := by sorry

end NUMINAMATH_GPT_cid_earnings_l2395_239595


namespace NUMINAMATH_GPT_books_combination_l2395_239561

theorem books_combination :
  (Nat.choose 15 3) = 455 := 
sorry

end NUMINAMATH_GPT_books_combination_l2395_239561


namespace NUMINAMATH_GPT_focal_length_is_correct_l2395_239534

def hyperbola_eqn : Prop := (∀ x y : ℝ, (x^2 / 4) - (y^2 / 9) = 1 → True)

noncomputable def focal_length_of_hyperbola : ℝ :=
  2 * Real.sqrt (4 + 9)

theorem focal_length_is_correct : hyperbola_eqn → focal_length_of_hyperbola = 2 * Real.sqrt 13 := by
  intro h
  sorry

end NUMINAMATH_GPT_focal_length_is_correct_l2395_239534


namespace NUMINAMATH_GPT_vishal_investment_more_than_trishul_l2395_239519

theorem vishal_investment_more_than_trishul :
  ∀ (V T R : ℝ), R = 2000 → T = R - 0.10 * R → V + T + R = 5780 → (V - T) / T * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end NUMINAMATH_GPT_vishal_investment_more_than_trishul_l2395_239519


namespace NUMINAMATH_GPT_muffin_banana_cost_ratio_l2395_239540

variables (m b c : ℕ) -- costs of muffin, banana, and cookie respectively
variables (susie_cost calvin_cost : ℕ)

-- Conditions
def susie_cost_eq : Prop := susie_cost = 5 * m + 4 * b + 2 * c
def calvin_cost_eq : Prop := calvin_cost = 3 * (5 * m + 4 * b + 2 * c)
def calvin_cost_eq_reduced : Prop := calvin_cost = 3 * m + 20 * b + 6 * c
def cookie_cost_eq : Prop := c = 2 * b

-- Question and Answer
theorem muffin_banana_cost_ratio
  (h1 : susie_cost_eq m b c susie_cost)
  (h2 : calvin_cost_eq m b c calvin_cost)
  (h3 : calvin_cost_eq_reduced m b c calvin_cost)
  (h4 : cookie_cost_eq b c)
  : m = 4 * b / 3 :=
sorry

end NUMINAMATH_GPT_muffin_banana_cost_ratio_l2395_239540


namespace NUMINAMATH_GPT_initial_albums_in_cart_l2395_239544

theorem initial_albums_in_cart (total_songs : ℕ) (songs_per_album : ℕ) (removed_albums : ℕ) 
  (h_total: total_songs = 42) 
  (h_songs_per_album: songs_per_album = 7)
  (h_removed: removed_albums = 2): 
  (total_songs / songs_per_album) + removed_albums = 8 := 
by
  sorry

end NUMINAMATH_GPT_initial_albums_in_cart_l2395_239544


namespace NUMINAMATH_GPT_units_digit_is_six_l2395_239539

theorem units_digit_is_six (n : ℤ) (h : (n^2 / 10 % 10) = 7) : (n^2 % 10) = 6 :=
by sorry

end NUMINAMATH_GPT_units_digit_is_six_l2395_239539


namespace NUMINAMATH_GPT_maximize_z_l2395_239599

open Real

theorem maximize_z (x y : ℝ) (h1 : x + y ≤ 10) (h2 : 3 * x + y ≤ 18) (h3 : 0 ≤ x) (h4 : 0 ≤ y) :
  (∀ x y, x + y ≤ 10 ∧ 3 * x + y ≤ 18 ∧ 0 ≤ x ∧ 0 ≤ y → x + y / 2 ≤ 7) :=
by
  sorry

end NUMINAMATH_GPT_maximize_z_l2395_239599


namespace NUMINAMATH_GPT_average_difference_correct_l2395_239573

def daily_diff : List ℤ := [15, 0, -15, 25, 5, -5, 10]
def number_of_days : ℤ := 7

theorem average_difference_correct :
  (daily_diff.sum : ℤ) / number_of_days = 5 := by
  sorry

end NUMINAMATH_GPT_average_difference_correct_l2395_239573


namespace NUMINAMATH_GPT_words_to_numbers_l2395_239556

def word_to_num (w : String) : Float := sorry

theorem words_to_numbers :
  word_to_num "fifty point zero zero one" = 50.001 ∧
  word_to_num "seventy-five point zero six" = 75.06 :=
by
  sorry

end NUMINAMATH_GPT_words_to_numbers_l2395_239556


namespace NUMINAMATH_GPT_sqrt_sum_eq_pow_l2395_239597

/-- 
For the value \( k = 3/2 \), the expression \( \sqrt{2016} + \sqrt{56} \) equals \( 14^k \)
-/
theorem sqrt_sum_eq_pow (k : ℝ) (h : k = 3 / 2) : 
  (Real.sqrt 2016 + Real.sqrt 56) = 14 ^ k := 
by 
  sorry

end NUMINAMATH_GPT_sqrt_sum_eq_pow_l2395_239597


namespace NUMINAMATH_GPT_fraction_product_l2395_239594

theorem fraction_product :
  (7 / 4 : ℚ) * (14 / 35) * (21 / 12) * (28 / 56) * (49 / 28) * (42 / 84) * (63 / 36) * (56 / 112) = (1201 / 12800) := 
by
  sorry

end NUMINAMATH_GPT_fraction_product_l2395_239594


namespace NUMINAMATH_GPT_ratio_tina_betsy_l2395_239508

theorem ratio_tina_betsy :
  ∀ (t_cindy t_betsy t_tina : ℕ),
  t_cindy = 12 →
  t_betsy = t_cindy / 2 →
  t_tina = t_cindy + 6 →
  t_tina / t_betsy = 3 :=
by
  intros t_cindy t_betsy t_tina h_cindy h_betsy h_tina
  sorry

end NUMINAMATH_GPT_ratio_tina_betsy_l2395_239508


namespace NUMINAMATH_GPT_unique_positive_real_solution_l2395_239529

theorem unique_positive_real_solution (x : ℝ) (hx_pos : x > 0) (h_eq : (x - 5) / 10 = 5 / (x - 10)) : 
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_unique_positive_real_solution_l2395_239529


namespace NUMINAMATH_GPT_mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l2395_239576

-- Definitions
def original_price : ℕ := 80
def discount_mallA (n : ℕ) : ℕ := min ((4 * n) * n) (80 * n / 2)
def discount_mallB (n : ℕ) : ℕ := (80 * n * 3) / 10

def total_cost_mallA (n : ℕ) : ℕ := (original_price * n) - discount_mallA n
def total_cost_mallB (n : ℕ) : ℕ := (original_price * n) - discount_mallB n

-- Theorem statements
theorem mall_b_better_for_fewer_than_6 (n : ℕ) (h : n < 6) : total_cost_mallA n > total_cost_mallB n := sorry
theorem mall_equal_for_6 (n : ℕ) (h : n = 6) : total_cost_mallA n = total_cost_mallB n := sorry
theorem mall_a_better_for_more_than_6 (n : ℕ) (h : n > 6) : total_cost_mallA n < total_cost_mallB n := sorry

end NUMINAMATH_GPT_mall_b_better_for_fewer_than_6_mall_equal_for_6_mall_a_better_for_more_than_6_l2395_239576


namespace NUMINAMATH_GPT_downstream_speed_l2395_239533

-- Define constants based on conditions given 
def V_upstream : ℝ := 30
def V_m : ℝ := 35

-- Define the speed of the stream based on the given conditions and upstream speed
def V_s : ℝ := V_m - V_upstream

-- The downstream speed is the man's speed in still water plus the stream speed
def V_downstream : ℝ := V_m + V_s

-- Theorem to be proved
theorem downstream_speed : V_downstream = 40 :=
by
  -- The actual proof steps are omitted
  sorry

end NUMINAMATH_GPT_downstream_speed_l2395_239533


namespace NUMINAMATH_GPT_sample_size_product_A_l2395_239570

theorem sample_size_product_A 
  (ratio_A : ℕ)
  (ratio_B : ℕ)
  (ratio_C : ℕ)
  (total_ratio : ℕ)
  (sample_size : ℕ) 
  (h_ratio : ratio_A = 2 ∧ ratio_B = 3 ∧ ratio_C = 5)
  (h_total_ratio : total_ratio = ratio_A + ratio_B + ratio_C)
  (h_sample_size : sample_size = 80) :
  (80 * (ratio_A : ℚ) / total_ratio) = 16 :=
by
  sorry

end NUMINAMATH_GPT_sample_size_product_A_l2395_239570


namespace NUMINAMATH_GPT_C_is_20_years_younger_l2395_239503

variable (A B C : ℕ)

-- Conditions from the problem
axiom age_condition : A + B = B + C + 20

-- Theorem representing the proof problem
theorem C_is_20_years_younger : A = C + 20 := sorry

end NUMINAMATH_GPT_C_is_20_years_younger_l2395_239503


namespace NUMINAMATH_GPT_semi_circle_radius_l2395_239559

theorem semi_circle_radius (π : ℝ) (hπ : Real.pi = π) (P : ℝ) (hP : P = 180) : 
  ∃ r : ℝ, r = 180 / (π + 2) :=
by
  sorry

end NUMINAMATH_GPT_semi_circle_radius_l2395_239559


namespace NUMINAMATH_GPT_find_c_value_l2395_239500

def projection_condition (v u : ℝ × ℝ) (c : ℝ) : Prop :=
  let v := (5, c)
  let u := (3, 2)
  let dot_product := (v.fst * u.fst + v.snd * u.snd)
  let norm_u_sq := (u.fst^2 + u.snd^2)
  (dot_product / norm_u_sq) * u.fst = -28 / 13 * u.fst

theorem find_c_value : ∃ c : ℝ, projection_condition (5, c) (3, 2) c :=
by
  use -43 / 2
  unfold projection_condition
  sorry

end NUMINAMATH_GPT_find_c_value_l2395_239500


namespace NUMINAMATH_GPT_find_y_l2395_239579

theorem find_y (y : ℕ) : (8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10 + 8 ^ 10) = 2 ^ y → y = 33 := 
by 
  sorry

end NUMINAMATH_GPT_find_y_l2395_239579


namespace NUMINAMATH_GPT_final_value_of_S_is_10_l2395_239536

-- Define the initial value of S
def initial_S : ℕ := 1

-- Define the sequence of I values
def I_values : List ℕ := [1, 3, 5]

-- Define the update operation on S
def update_S (S : ℕ) (I : ℕ) : ℕ := S + I

-- Final value of S after all updates
def final_S : ℕ := (I_values.foldl update_S initial_S)

-- The theorem stating that the final value of S is 10
theorem final_value_of_S_is_10 : final_S = 10 :=
by
  sorry

end NUMINAMATH_GPT_final_value_of_S_is_10_l2395_239536


namespace NUMINAMATH_GPT_lex_reads_in_12_days_l2395_239550

theorem lex_reads_in_12_days
  (total_pages : ℕ)
  (pages_per_day : ℕ)
  (h1 : total_pages = 240)
  (h2 : pages_per_day = 20) :
  total_pages / pages_per_day = 12 :=
by
  sorry

end NUMINAMATH_GPT_lex_reads_in_12_days_l2395_239550


namespace NUMINAMATH_GPT_amount_a_put_in_correct_l2395_239542

noncomputable def amount_a_put_in (total_profit managing_fee total_received_by_a profit_remaining: ℝ) : ℝ :=
  let capital_b := 2500
  let a_receives_from_investment := total_received_by_a - managing_fee
  let profit_ratio := a_receives_from_investment / profit_remaining
  profit_ratio * capital_b

theorem amount_a_put_in_correct :
  amount_a_put_in 9600 960 6000 8640 = 3500 :=
by
  dsimp [amount_a_put_in]
  sorry

end NUMINAMATH_GPT_amount_a_put_in_correct_l2395_239542


namespace NUMINAMATH_GPT_square_in_S_l2395_239512

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ a b : ℕ, a^2 + b^2 = n

def S (n : ℕ) : Prop :=
  is_sum_of_two_squares (n - 1) ∧ is_sum_of_two_squares n ∧ is_sum_of_two_squares (n + 1)

theorem square_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end NUMINAMATH_GPT_square_in_S_l2395_239512


namespace NUMINAMATH_GPT_quadratic_equation_roots_l2395_239583

-- Define the two numbers α and β such that their arithmetic and geometric means are given.
variables (α β : ℝ)

-- Arithmetic mean condition
def arithmetic_mean_condition : Prop := (α + β = 16)

-- Geometric mean condition
def geometric_mean_condition : Prop := (α * β = 225)

-- The quadratic equation with roots α and β
def quadratic_equation (x : ℝ) : ℝ := x^2 - 16 * x + 225

-- The proof statement
theorem quadratic_equation_roots (α β : ℝ) (h1 : arithmetic_mean_condition α β) (h2 : geometric_mean_condition α β) :
  ∃ x : ℝ, quadratic_equation x = 0 :=
sorry

end NUMINAMATH_GPT_quadratic_equation_roots_l2395_239583


namespace NUMINAMATH_GPT_extra_charge_per_wand_l2395_239555

theorem extra_charge_per_wand
  (cost_per_wand : ℕ)
  (num_wands : ℕ)
  (total_collected : ℕ)
  (num_wands_sold : ℕ)
  (h_cost : cost_per_wand = 60)
  (h_num_wands : num_wands = 3)
  (h_total_collected : total_collected = 130)
  (h_num_wands_sold : num_wands_sold = 2) :
  ((total_collected / num_wands_sold) - cost_per_wand) = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_extra_charge_per_wand_l2395_239555


namespace NUMINAMATH_GPT_cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l2395_239538

-- Defining the conditions
def cos_x_eq_one (x : ℝ) : Prop := Real.cos x = 1
def sin_x_eq_zero (x : ℝ) : Prop := Real.sin x = 0

-- Main theorem statement
theorem cos_x_is_necessary_but_not_sufficient_for_sin_x_zero (x : ℝ) : 
  (∀ x, cos_x_eq_one x → sin_x_eq_zero x) ∧ (∃ x, sin_x_eq_zero x ∧ ¬ cos_x_eq_one x) :=
by 
  sorry

end NUMINAMATH_GPT_cos_x_is_necessary_but_not_sufficient_for_sin_x_zero_l2395_239538


namespace NUMINAMATH_GPT_susan_mean_l2395_239517

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_mean :
  (susan_scores.sum) / (susan_scores.length) = 94 := by
  sorry

end NUMINAMATH_GPT_susan_mean_l2395_239517


namespace NUMINAMATH_GPT_circular_paper_pieces_needed_l2395_239577

-- Definition of the problem conditions
def side_length_dm := 10
def side_length_cm := side_length_dm * 10
def perimeter_cm := 4 * side_length_cm
def number_of_sides := 4
def semicircles_per_side := 1
def total_semicircles := number_of_sides * semicircles_per_side
def semicircles_to_circles := 2
def total_circles := total_semicircles / semicircles_to_circles
def paper_pieces_per_circle := 20

-- Main theorem stating the problem and the answer.
theorem circular_paper_pieces_needed : (total_circles * paper_pieces_per_circle) = 40 :=
by sorry

end NUMINAMATH_GPT_circular_paper_pieces_needed_l2395_239577


namespace NUMINAMATH_GPT_num_cows_on_farm_l2395_239515

variables (D C S : ℕ)

def total_legs : ℕ := 8 * S + 2 * D + 4 * C
def total_heads : ℕ := D + C + S

theorem num_cows_on_farm
  (h1 : S = 2 * D)
  (h2 : total_legs D C S = 2 * total_heads D C S + 72)
  (h3 : D + C + S ≤ 40) :
  C = 30 :=
sorry

end NUMINAMATH_GPT_num_cows_on_farm_l2395_239515


namespace NUMINAMATH_GPT_multiple_of_B_share_l2395_239524

theorem multiple_of_B_share (A B C : ℝ) (k : ℝ) 
    (h1 : 3 * A = k * B) 
    (h2 : k * B = 7 * 84) 
    (h3 : C = 84)
    (h4 : A + B + C = 427) :
    k = 4 :=
by
  -- We do not need the detailed proof steps here.
  sorry

end NUMINAMATH_GPT_multiple_of_B_share_l2395_239524


namespace NUMINAMATH_GPT_simplify_expression_l2395_239505

theorem simplify_expression (x : ℤ) : 
  (12*x^10 + 5*x^9 + 3*x^8) + (2*x^12 + 9*x^10 + 4*x^8 + 6*x^4 + 7*x^2 + 10)
  = 2*x^12 + 21*x^10 + 5*x^9 + 7*x^8 + 6*x^4 + 7*x^2 + 10 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l2395_239505


namespace NUMINAMATH_GPT_point_B_in_fourth_quadrant_l2395_239581

theorem point_B_in_fourth_quadrant (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : (b > 0 ∧ a < 0) :=
by {
    sorry
}

end NUMINAMATH_GPT_point_B_in_fourth_quadrant_l2395_239581


namespace NUMINAMATH_GPT_operation_result_l2395_239537

def operation (a b : Int) : Int :=
  (a + b) * (a - b)

theorem operation_result :
  operation 4 (operation 2 (-1)) = 7 :=
by
  sorry

end NUMINAMATH_GPT_operation_result_l2395_239537


namespace NUMINAMATH_GPT_cube_painting_equiv_1260_l2395_239564

def num_distinguishable_paintings_of_cube : Nat :=
  1260

theorem cube_painting_equiv_1260 :
  ∀ (colors : Fin 8 → Color), -- assuming we have a type Color representing colors
    (∀ i j : Fin 6, i ≠ j → colors i ≠ colors j) →  -- each face has a different color
    ∃ f : Cube × Fin 8 → Cube × Fin 8, -- considering symmetry transformations (rotations)
      num_distinguishable_paintings_of_cube = 1260 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_cube_painting_equiv_1260_l2395_239564


namespace NUMINAMATH_GPT_train_length_is_100_meters_l2395_239548

-- Definitions of conditions
def speed_kmh := 40  -- speed in km/hr
def time_s := 9  -- time in seconds

-- Conversion factors
def km_to_m := 1000  -- 1 km = 1000 meters
def hr_to_s := 3600  -- 1 hour = 3600 seconds

-- Converting speed from km/hr to m/s
def speed_ms := (speed_kmh * km_to_m) / hr_to_s

-- The proof that the length of the train is 100 meters
theorem train_length_is_100_meters :
  (speed_ms * time_s) = 100 :=
by
  sorry

-- The Lean statement merely sets up the problem as asked.

end NUMINAMATH_GPT_train_length_is_100_meters_l2395_239548


namespace NUMINAMATH_GPT_orchestra_ticket_cost_l2395_239526

noncomputable def cost_balcony : ℝ := 8  -- cost of balcony tickets
noncomputable def total_sold : ℝ := 340  -- total tickets sold
noncomputable def total_revenue : ℝ := 3320  -- total revenue
noncomputable def extra_balcony : ℝ := 40  -- extra tickets sold for balcony than orchestra

theorem orchestra_ticket_cost (x y : ℝ) (h1 : x + extra_balcony = total_sold)
    (h2 : y = x + extra_balcony) (h3 : x + y = total_sold)
    (h4 : x + cost_balcony * y = total_revenue) : 
    cost_balcony = 8 → x = 12 :=
by
  sorry

end NUMINAMATH_GPT_orchestra_ticket_cost_l2395_239526


namespace NUMINAMATH_GPT_largest_increase_between_2006_and_2007_l2395_239585

-- Define the number of students taking the AMC in each year
def students_2002 := 50
def students_2003 := 55
def students_2004 := 63
def students_2005 := 70
def students_2006 := 75
def students_2007_AMC10 := 90
def students_2007_AMC12 := 15

-- Define the total number of students participating in any AMC contest each year
def total_students_2002 := students_2002
def total_students_2003 := students_2003
def total_students_2004 := students_2004
def total_students_2005 := students_2005
def total_students_2006 := students_2006
def total_students_2007 := students_2007_AMC10 + students_2007_AMC12

-- Function to calculate percentage increase
def percentage_increase (old new : ℕ) : ℚ :=
  ((new - old : ℕ) : ℚ) / old * 100

-- Calculate percentage increases between the years
def inc_2002_2003 := percentage_increase total_students_2002 total_students_2003
def inc_2003_2004 := percentage_increase total_students_2003 total_students_2004
def inc_2004_2005 := percentage_increase total_students_2004 total_students_2005
def inc_2005_2006 := percentage_increase total_students_2005 total_students_2006
def inc_2006_2007 := percentage_increase total_students_2006 total_students_2007

-- Prove that the largest percentage increase is between 2006 and 2007
theorem largest_increase_between_2006_and_2007 :
  inc_2006_2007 > inc_2005_2006 ∧
  inc_2006_2007 > inc_2004_2005 ∧
  inc_2006_2007 > inc_2003_2004 ∧
  inc_2006_2007 > inc_2002_2003 := 
by {
  sorry
}

end NUMINAMATH_GPT_largest_increase_between_2006_and_2007_l2395_239585


namespace NUMINAMATH_GPT_g_increasing_in_interval_l2395_239587

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 - a * x^2 + a * x + 2
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * x + a
noncomputable def f'' (a : ℝ) (x : ℝ) : ℝ := 2 * x - 2 * a

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f'' a x / x
noncomputable def g' (a : ℝ) (x : ℝ) : ℝ := 1 - a / (x^2)

theorem g_increasing_in_interval (a : ℝ) (h : a < 1) :
  ∀ x : ℝ, 1 < x → 0 < g' a x := by
  sorry

end NUMINAMATH_GPT_g_increasing_in_interval_l2395_239587


namespace NUMINAMATH_GPT_third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l2395_239591

-- Define the first finite difference function
def delta (f : ℕ → ℤ) (n : ℕ) : ℤ := f (n + 1) - f n

-- Define the second finite difference using the first
def delta2 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta f) n

-- Define the third finite difference using the second
def delta3 (f : ℕ → ℤ) (n : ℕ) : ℤ := delta (delta2 f) n

-- Prove the third finite difference of n^3 is 6
theorem third_diff_n_cube_is_const_6 :
  delta3 (fun (n : ℕ) => (n : ℤ)^3) = fun _ => 6 := 
by
  sorry

-- Prove the third finite difference of the general form function is 6
theorem third_diff_general_form_is_6 (a b c : ℤ) :
  delta3 (fun (n : ℕ) => (n : ℤ)^3 + a * (n : ℤ)^2 + b * (n : ℤ) + c) = fun _ => 6 := 
by
  sorry

end NUMINAMATH_GPT_third_diff_n_cube_is_const_6_third_diff_general_form_is_6_l2395_239591


namespace NUMINAMATH_GPT_find_k_l2395_239569

theorem find_k (k : ℝ) (h : ∃ (k : ℝ), 3 = k * (-1) - 2) : k = -5 :=
by
  rcases h with ⟨k, hk⟩
  sorry

end NUMINAMATH_GPT_find_k_l2395_239569


namespace NUMINAMATH_GPT_problem_decimal_parts_l2395_239557

theorem problem_decimal_parts :
  let a := 5 + Real.sqrt 7 - 7
  let b := 5 - Real.sqrt 7 - 2
  (a + b) ^ 2023 = 1 :=
by
  sorry

end NUMINAMATH_GPT_problem_decimal_parts_l2395_239557


namespace NUMINAMATH_GPT_value_of_y_l2395_239516

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end NUMINAMATH_GPT_value_of_y_l2395_239516


namespace NUMINAMATH_GPT_complete_square_l2395_239528

theorem complete_square (x : ℝ) : (x^2 - 4*x + 2 = 0) → ((x - 2)^2 = 2) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_complete_square_l2395_239528


namespace NUMINAMATH_GPT_find_sample_size_l2395_239584

def sample_size (sample : List ℕ) : ℕ :=
  sample.length

theorem find_sample_size :
  sample_size (List.replicate 500 0) = 500 :=
by
  sorry

end NUMINAMATH_GPT_find_sample_size_l2395_239584


namespace NUMINAMATH_GPT_find_a_l2395_239525

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(-x) * (1 - a^x)

theorem find_a (a : ℝ) : 
  (∀ x : ℝ, f a (-x) = -f a x) ∧ a > 0 ∧ a ≠ 1 → a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l2395_239525


namespace NUMINAMATH_GPT_circle_incircle_tangent_radius_l2395_239593

theorem circle_incircle_tangent_radius (r1 r2 r3 : ℕ) (k : ℕ) (h1 : r1 = 1) (h2 : r2 = 4) (h3 : r3 = 9) : 
  k = 11 :=
by
  -- Definitions according to the problem
  let k₁ := r1
  let k₂ := r2
  let k₃ := r3
  -- Hypotheses given by the problem
  have h₁ : k₁ = 1 := h1
  have h₂ : k₂ = 4 := h2
  have h₃ : k₃ = 9 := h3
  -- Prove the radius of the incircle k
  sorry

end NUMINAMATH_GPT_circle_incircle_tangent_radius_l2395_239593


namespace NUMINAMATH_GPT_speed_conversion_l2395_239504

noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * (1000 / 3600)

theorem speed_conversion (h : kmh_to_ms 1 = 1000 / 3600) :
  kmh_to_ms 1.7 = 0.4722 :=
by sorry

end NUMINAMATH_GPT_speed_conversion_l2395_239504


namespace NUMINAMATH_GPT_focus_of_parabola_l2395_239543

theorem focus_of_parabola (y : ℝ → ℝ) (h : ∀ x, y x = 16 * x^2) : 
    ∃ p, p = (0, 1/64) := 
by
    existsi (0, 1/64)
    -- The proof would go here, but we are adding sorry to skip it 
    sorry

end NUMINAMATH_GPT_focus_of_parabola_l2395_239543


namespace NUMINAMATH_GPT_unknown_subtraction_problem_l2395_239586

theorem unknown_subtraction_problem (x y : ℝ) (h1 : x = 40) (h2 : x / 4 * 5 + 10 - y = 48) : y = 12 :=
by
  sorry

end NUMINAMATH_GPT_unknown_subtraction_problem_l2395_239586


namespace NUMINAMATH_GPT_one_inch_cubes_with_two_or_more_painted_faces_l2395_239567

def original_cube_length : ℕ := 4

def total_one_inch_cubes : ℕ := original_cube_length ^ 3

def corners_count : ℕ := 8

def edges_minus_corners_count : ℕ := 12 * 2

theorem one_inch_cubes_with_two_or_more_painted_faces
  (painted_faces_on_each_face : ∀ i : ℕ, i < total_one_inch_cubes → ℕ) : 
  ∃ n : ℕ, n = corners_count + edges_minus_corners_count ∧ n = 32 := 
by
  simp only [corners_count, edges_minus_corners_count, total_one_inch_cubes]
  sorry

end NUMINAMATH_GPT_one_inch_cubes_with_two_or_more_painted_faces_l2395_239567


namespace NUMINAMATH_GPT_wall_length_eq_800_l2395_239510

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

end NUMINAMATH_GPT_wall_length_eq_800_l2395_239510


namespace NUMINAMATH_GPT_average_speed_l2395_239527

theorem average_speed (x y : ℝ) (h₁ : 0 ≤ x) (h₂ : 0 ≤ y)
  (total_time : x / 4 + y / 3 + y / 6 + x / 4 = 5) :
  (2 * (x + y)) / 5 = 4 :=
by
  sorry

end NUMINAMATH_GPT_average_speed_l2395_239527


namespace NUMINAMATH_GPT_least_n_froods_l2395_239522

def froods_score (n : ℕ) : ℕ := n * (n + 1) / 2
def eating_score (n : ℕ) : ℕ := n ^ 2

theorem least_n_froods :
    ∃ n : ℕ, 0 < n ∧ (froods_score n > eating_score n) ∧ (∀ m : ℕ, 0 < m ∧ m < n → froods_score m ≤ eating_score m) :=
  sorry

end NUMINAMATH_GPT_least_n_froods_l2395_239522


namespace NUMINAMATH_GPT_average_snowfall_dec_1861_l2395_239507

theorem average_snowfall_dec_1861 (snowfall : ℕ) (days_in_dec : ℕ) (hours_in_day : ℕ) 
  (time_period : ℕ) (Avg_inch_per_hour : ℚ) : 
  snowfall = 492 ∧ days_in_dec = 31 ∧ hours_in_day = 24 ∧ time_period = days_in_dec * hours_in_day ∧ 
  Avg_inch_per_hour = snowfall / time_period → 
  Avg_inch_per_hour = 492 / (31 * 24) :=
by sorry

end NUMINAMATH_GPT_average_snowfall_dec_1861_l2395_239507


namespace NUMINAMATH_GPT_root_equality_l2395_239547

theorem root_equality (p q : ℝ) (h1 : 1 + p + q = (2 - 2 * q) / p) (h2 : 1 + p + q = (1 - p + q) / q) :
  p + q = 1 :=
sorry

end NUMINAMATH_GPT_root_equality_l2395_239547


namespace NUMINAMATH_GPT_most_people_can_attend_on_most_days_l2395_239531

-- Define the days of the week as a type
inductive Day
| Mon | Tues | Wed | Thurs | Fri

open Day

-- Define the availability of each person
def is_available (person : String) (day : Day) : Prop :=
  match person, day with
  | "Anna", Mon => False
  | "Anna", Wed => False
  | "Anna", Fri => False
  | "Bill", Tues => False
  | "Bill", Thurs => False
  | "Bill", Fri => False
  | "Carl", Mon => False
  | "Carl", Tues => False
  | "Carl", Thurs => False
  | "Diana", Wed => False
  | "Diana", Fri => False
  | _, _ => True

-- Prove the result
theorem most_people_can_attend_on_most_days :
  {d : Day | d ∈ [Mon, Tues, Wed]} = {d : Day | ∀p : String, is_available p d → p ∈ ["Bill", "Carl", "Diana"] ∨ p ∉ ["Anna", "Bill"]} :=
sorry

end NUMINAMATH_GPT_most_people_can_attend_on_most_days_l2395_239531


namespace NUMINAMATH_GPT_outfit_choices_l2395_239506

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

end NUMINAMATH_GPT_outfit_choices_l2395_239506


namespace NUMINAMATH_GPT_choose_5_starters_including_twins_l2395_239572

def number_of_ways_choose_starters (total_players : ℕ) (members_in_lineup : ℕ) (twins1 twins2 : (ℕ × ℕ)) : ℕ :=
1834

theorem choose_5_starters_including_twins :
  number_of_ways_choose_starters 18 5 (1, 2) (3, 4) = 1834 :=
sorry

end NUMINAMATH_GPT_choose_5_starters_including_twins_l2395_239572


namespace NUMINAMATH_GPT_a_gt_abs_b_suff_not_necc_l2395_239589

theorem a_gt_abs_b_suff_not_necc (a b : ℝ) (h : a > |b|) : 
  a^2 > b^2 ∧ ∀ a b : ℝ, (a^2 > b^2 → |a| > |b|) → ¬ (a < -|b|) := 
by
  sorry

end NUMINAMATH_GPT_a_gt_abs_b_suff_not_necc_l2395_239589


namespace NUMINAMATH_GPT_diff_of_cubes_divisible_by_9_l2395_239513

theorem diff_of_cubes_divisible_by_9 (a b : ℤ) : 9 ∣ ((2 * a + 1)^3 - (2 * b + 1)^3) := 
sorry

end NUMINAMATH_GPT_diff_of_cubes_divisible_by_9_l2395_239513


namespace NUMINAMATH_GPT_yolk_count_proof_l2395_239596

-- Define the conditions of the problem
def eggs_in_carton : ℕ := 12
def double_yolk_eggs : ℕ := 5
def single_yolk_eggs : ℕ := eggs_in_carton - double_yolk_eggs
def yolks_in_double_yolk_eggs : ℕ := double_yolk_eggs * 2
def yolks_in_single_yolk_eggs : ℕ := single_yolk_eggs
def total_yolks : ℕ := yolks_in_single_yolk_eggs + yolks_in_double_yolk_eggs

-- Stating the theorem to prove the total number of yolks is 17
theorem yolk_count_proof : total_yolks = 17 := 
by
  sorry

end NUMINAMATH_GPT_yolk_count_proof_l2395_239596


namespace NUMINAMATH_GPT_find_a_value_l2395_239532

theorem find_a_value :
  (∀ y : ℝ, y ∈ Set.Ioo (-3/2 : ℝ) 4 → y * (2 * y - 3) < (12 : ℝ)) ↔ (12 = 12) := 
by 
  sorry

end NUMINAMATH_GPT_find_a_value_l2395_239532


namespace NUMINAMATH_GPT_select_representatives_l2395_239592

theorem select_representatives
  (female_count : ℕ) (male_count : ℕ)
  (female_count_eq : female_count = 4)
  (male_count_eq : male_count = 6) :
  female_count * male_count = 24 := by
  sorry

end NUMINAMATH_GPT_select_representatives_l2395_239592


namespace NUMINAMATH_GPT_problem_statement_l2395_239553

theorem problem_statement (x y z : ℝ) (hx : x + y + z = 2) (hxy : xy + xz + yz = -9) (hxyz : xyz = 1) :
  (yz / x) + (xz / y) + (xy / z) = 77 := sorry

end NUMINAMATH_GPT_problem_statement_l2395_239553


namespace NUMINAMATH_GPT_minjeong_walk_distance_l2395_239551

noncomputable def park_side_length : ℕ := 40
noncomputable def square_sides : ℕ := 4

theorem minjeong_walk_distance (side_length : ℕ) (sides : ℕ) (h : side_length = park_side_length) (h2 : sides = square_sides) : 
  side_length * sides = 160 := by
  sorry

end NUMINAMATH_GPT_minjeong_walk_distance_l2395_239551


namespace NUMINAMATH_GPT_matthew_hotdogs_l2395_239566

-- Definitions based on conditions
def hotdogs_ella_emma : ℕ := 2 + 2
def hotdogs_luke : ℕ := 2 * hotdogs_ella_emma
def hotdogs_hunter : ℕ := (3 * hotdogs_ella_emma) / 2  -- Multiplying by 1.5 

-- Theorem statement to prove the total number of hotdogs
theorem matthew_hotdogs : hotdogs_ella_emma + hotdogs_luke + hotdogs_hunter = 18 := by
  sorry

end NUMINAMATH_GPT_matthew_hotdogs_l2395_239566


namespace NUMINAMATH_GPT_find_number_of_dogs_l2395_239549

variables (D P S : ℕ)
theorem find_number_of_dogs (h1 : D = 2 * P) (h2 : P = 2 * S) (h3 : 4 * D + 4 * P + 2 * S = 510) :
  D = 60 := 
sorry

end NUMINAMATH_GPT_find_number_of_dogs_l2395_239549


namespace NUMINAMATH_GPT_find_inverse_of_f_at_4_l2395_239523

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Statement of the problem
theorem find_inverse_of_f_at_4 : ∃ t : ℝ, f t = 4 ∧ t ≤ 1 ∧ t = -1 := by
  sorry

end NUMINAMATH_GPT_find_inverse_of_f_at_4_l2395_239523
