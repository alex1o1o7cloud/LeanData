import Mathlib

namespace NUMINAMATH_GPT_problem_divisible_by_900_l554_55423

theorem problem_divisible_by_900 (X : ℕ) (a b c d : ℕ) 
  (h1 : 1000 <= X)
  (h2 : X < 10000)
  (h3 : X = 1000 * a + 100 * b + 10 * c + d)
  (h4 : d ≠ 0)
  (h5 : (X + (1000 * a + 100 * c + 10 * b + d)) % 900 = 0)
  : X % 90 = 45 := 
sorry

end NUMINAMATH_GPT_problem_divisible_by_900_l554_55423


namespace NUMINAMATH_GPT_cos_phi_expression_l554_55417

theorem cos_phi_expression (a b c : ℝ) (φ R : ℝ)
  (habc : a > 0 ∧ b > 0 ∧ c > 0)
  (angles : 2 * φ + 3 * φ + 4 * φ = π)
  (law_of_sines : a / Real.sin (2 * φ) = 2 * R ∧ b / Real.sin (3 * φ) = 2 * R ∧ c / Real.sin (4 * φ) = 2 * R) :
  Real.cos φ = (a + c) / (2 * b) := 
by 
  sorry

end NUMINAMATH_GPT_cos_phi_expression_l554_55417


namespace NUMINAMATH_GPT_trig_identity_l554_55445

open Real

theorem trig_identity (α : ℝ) (h_tan : tan α = 2) (h_quad : 0 < α ∧ α < π / 2) :
  sin (2 * α) + cos α = (4 + sqrt 5) / 5 :=
sorry

end NUMINAMATH_GPT_trig_identity_l554_55445


namespace NUMINAMATH_GPT_exactly_two_succeed_probability_l554_55406

-- Define the probabilities of events A, B, and C decrypting the code
def P_A_decrypts : ℚ := 1/5
def P_B_decrypts : ℚ := 1/4
def P_C_decrypts : ℚ := 1/3

-- Define the probabilities of events A, B, and C not decrypting the code
def P_A_not_decrypts : ℚ := 1 - P_A_decrypts
def P_B_not_decrypts : ℚ := 1 - P_B_decrypts
def P_C_not_decrypts : ℚ := 1 - P_C_decrypts

-- Define the probability that exactly two out of A, B, and C decrypt the code
def P_exactly_two_succeed : ℚ :=
  (P_A_decrypts * P_B_decrypts * P_C_not_decrypts) +
  (P_A_decrypts * P_B_not_decrypts * P_C_decrypts) +
  (P_A_not_decrypts * P_B_decrypts * P_C_decrypts)

-- Prove that this probability is equal to 3/20
theorem exactly_two_succeed_probability : P_exactly_two_succeed = 3 / 20 := by
  sorry

end NUMINAMATH_GPT_exactly_two_succeed_probability_l554_55406


namespace NUMINAMATH_GPT_triangular_weight_is_60_l554_55455

variable (w_round w_triangular w_rectangular : ℝ)

axiom rectangular_weight : w_rectangular = 90
axiom balance1 : w_round + w_triangular = 3 * w_round
axiom balance2 : 4 * w_round + w_triangular = w_triangular + w_round + w_rectangular

theorem triangular_weight_is_60 :
  w_triangular = 60 :=
by
  sorry

end NUMINAMATH_GPT_triangular_weight_is_60_l554_55455


namespace NUMINAMATH_GPT_find_theta_l554_55425

theorem find_theta (θ : Real) (h : abs θ < π / 2) (h_eq : Real.sin (π + θ) = -Real.sqrt 3 * Real.cos (2 * π - θ)) :
  θ = π / 3 :=
sorry

end NUMINAMATH_GPT_find_theta_l554_55425


namespace NUMINAMATH_GPT_linear_function_quadrants_l554_55409

theorem linear_function_quadrants (m : ℝ) (h1 : m - 2 < 0) (h2 : m + 1 > 0) : -1 < m ∧ m < 2 := 
by 
  sorry

end NUMINAMATH_GPT_linear_function_quadrants_l554_55409


namespace NUMINAMATH_GPT_expression_simplification_l554_55458

theorem expression_simplification :
  (4 * 6 / (12 * 8)) * ((5 * 12 * 8) / (4 * 5 * 5)) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l554_55458


namespace NUMINAMATH_GPT_can_transfer_increase_average_l554_55448

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_can_transfer_increase_average_l554_55448


namespace NUMINAMATH_GPT_fred_likes_12_pairs_of_digits_l554_55483

theorem fred_likes_12_pairs_of_digits :
  (∃ (pairs : Finset (ℕ × ℕ)), 
    (∀ (a b : ℕ), (a, b) ∈ pairs ↔ ∃ (n : ℕ), n < 100 ∧ n % 8 = 0 ∧ n = 10 * a + b) ∧
    pairs.card = 12) :=
by
  sorry

end NUMINAMATH_GPT_fred_likes_12_pairs_of_digits_l554_55483


namespace NUMINAMATH_GPT_angle_division_quadrant_l554_55479

variable (k : ℤ)
variable (α : ℝ)
variable (h : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi)

theorem angle_division_quadrant 
  (hα_sec_quadrant : 2 * k * Real.pi + Real.pi / 2 < α ∧ α < 2 * k * Real.pi + Real.pi) : 
  (∃ m : ℤ, (m = 0 ∧ Real.pi / 4 < α / 2 ∧ α / 2 < Real.pi / 2) ∨ 
             (m = 1 ∧ Real.pi * (2 * m + 1) / 4 < α / 2 ∧ α / 2 < Real.pi * (2 * m + 1) / 2)) :=
sorry

end NUMINAMATH_GPT_angle_division_quadrant_l554_55479


namespace NUMINAMATH_GPT_range_of_b_l554_55478

noncomputable def f (x a b : ℝ) := (x - a)^2 * (x + b) * Real.exp x

theorem range_of_b (a b : ℝ) (h_max : ∃ δ > 0, ∀ x, |x - a| < δ → f x a b ≤ f a a b) : b < -a := sorry

end NUMINAMATH_GPT_range_of_b_l554_55478


namespace NUMINAMATH_GPT_regular_21_gon_symmetries_and_angle_sum_l554_55418

theorem regular_21_gon_symmetries_and_angle_sum :
  let L' := 21
  let R' := 360 / 21
  L' + R' = 38.142857 := by
    sorry

end NUMINAMATH_GPT_regular_21_gon_symmetries_and_angle_sum_l554_55418


namespace NUMINAMATH_GPT_simplified_expression_result_l554_55461

theorem simplified_expression_result :
  ((2 + 3 + 6 + 7) / 3) + ((3 * 6 + 9) / 4) = 12.75 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplified_expression_result_l554_55461


namespace NUMINAMATH_GPT_cost_per_day_is_18_l554_55402

def cost_per_day_first_week (x : ℕ) : Prop :=
  let cost_per_day_rest_week := 12
  let total_days := 23
  let total_cost := 318
  let first_week_days := 7
  let remaining_days := total_days - first_week_days
  (first_week_days * x) + (remaining_days * cost_per_day_rest_week) = total_cost

theorem cost_per_day_is_18 : cost_per_day_first_week 18 :=
  sorry

end NUMINAMATH_GPT_cost_per_day_is_18_l554_55402


namespace NUMINAMATH_GPT_nonneg_triple_inequality_l554_55428

theorem nonneg_triple_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  (1/3) * (a + b + c)^2 ≥ a * Real.sqrt (b * c) + b * Real.sqrt (c * a) + c * Real.sqrt (a * b) :=
by
  sorry

end NUMINAMATH_GPT_nonneg_triple_inequality_l554_55428


namespace NUMINAMATH_GPT_solve_fraction_equation_l554_55433

theorem solve_fraction_equation (x : ℚ) (h : x ≠ -1) : 
  (x / (x + 1) = 2 * x / (3 * x + 3) - 1) → x = -3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_fraction_equation_l554_55433


namespace NUMINAMATH_GPT_digit_B_for_divisibility_l554_55463

theorem digit_B_for_divisibility (B : ℕ) (h : (40000 + 1000 * B + 100 * B + 20 + 6) % 7 = 0) : B = 1 :=
sorry

end NUMINAMATH_GPT_digit_B_for_divisibility_l554_55463


namespace NUMINAMATH_GPT_total_chocolate_bars_l554_55467

theorem total_chocolate_bars (n_small_boxes : ℕ) (bars_per_box : ℕ) (total_bars : ℕ) :
  n_small_boxes = 16 → bars_per_box = 25 → total_bars = 16 * 25 → total_bars = 400 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_chocolate_bars_l554_55467


namespace NUMINAMATH_GPT_program_exists_l554_55485
open Function

-- Define the chessboard and labyrinth
namespace ChessMaze

structure Position :=
  (row : Nat)
  (col : Nat)
  (h_row : row < 8)
  (h_col : col < 8)

inductive Command
| RIGHT | LEFT | UP | DOWN

structure Labyrinth :=
  (barriers : Position → Position → Bool) -- True if there's a barrier between the two positions

def accessible (L : Labyrinth) (start : Position) (cmd : List Command) : Set Position :=
  -- The set of positions accessible after applying the commands from start in labyrinth L
  sorry

-- The main theorem we want to prove
theorem program_exists : 
  ∃ (cmd : List Command), ∀ (L : Labyrinth) (start : Position), ∀ pos ∈ accessible L start cmd, ∃ p : Position, p = pos :=
  sorry

end ChessMaze

end NUMINAMATH_GPT_program_exists_l554_55485


namespace NUMINAMATH_GPT_math_books_count_l554_55422

theorem math_books_count (M H : ℤ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 397) : M = 53 :=
by
  sorry

end NUMINAMATH_GPT_math_books_count_l554_55422


namespace NUMINAMATH_GPT_max_sum_terms_arithmetic_seq_l554_55424

theorem max_sum_terms_arithmetic_seq (a1 d : ℝ) (h1 : a1 > 0) 
  (h2 : 3 * (2 * a1 + 2 * d) = 11 * (2 * a1 + 10 * d)) :
  ∃ (n : ℕ),  (∀ k, 1 ≤ k ∧ k ≤ n → a1 + (k - 1) * d > 0) ∧  a1 + n * d ≤ 0 ∧ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_max_sum_terms_arithmetic_seq_l554_55424


namespace NUMINAMATH_GPT_find_first_term_geometric_sequence_l554_55496

theorem find_first_term_geometric_sequence 
  (a b c : ℚ) 
  (h₁ : b = a * 4) 
  (h₂ : 36 = a * 4^2) 
  (h₃ : c = a * 4^3) 
  (h₄ : 144 = a * 4^4) : 
  a = 9 / 4 :=
sorry

end NUMINAMATH_GPT_find_first_term_geometric_sequence_l554_55496


namespace NUMINAMATH_GPT_average_speed_of_train_b_l554_55488

-- Given conditions
def distance_between_trains_initially := 13
def speed_of_train_a := 37
def time_to_overtake := 5
def distance_a_in_5_hours := speed_of_train_a * time_to_overtake
def distance_b_to_overtake := distance_between_trains_initially + distance_a_in_5_hours + 17

-- Prove: The average speed of Train B
theorem average_speed_of_train_b : 
  ∃ v_B, v_B = distance_b_to_overtake / time_to_overtake ∧ v_B = 43 :=
by
  -- The proof should go here, but we use sorry to skip it.
  sorry

end NUMINAMATH_GPT_average_speed_of_train_b_l554_55488


namespace NUMINAMATH_GPT_common_root_l554_55407

def f (x : ℝ) : ℝ := x^4 - x^3 - 22 * x^2 + 16 * x + 96
def g (x : ℝ) : ℝ := x^3 - 2 * x^2 - 3 * x + 10

theorem common_root :
  f (-2) = 0 ∧ g (-2) = 0 := by
  sorry

end NUMINAMATH_GPT_common_root_l554_55407


namespace NUMINAMATH_GPT_find_value_of_fraction_l554_55498

theorem find_value_of_fraction (x y z : ℝ)
  (h1 : 3 * x - 4 * y - z = 0)
  (h2 : x + 4 * y - 15 * z = 0)
  (h3 : z ≠ 0) :
  (x^2 + 3 * x * y - y * z) / (y^2 + z^2) = 2.4 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_fraction_l554_55498


namespace NUMINAMATH_GPT_exchange_positions_l554_55499

theorem exchange_positions : ∀ (people : ℕ), people = 8 → (∃ (ways : ℕ), ways = 336) :=
by sorry

end NUMINAMATH_GPT_exchange_positions_l554_55499


namespace NUMINAMATH_GPT_sales_second_month_l554_55481

theorem sales_second_month 
  (sale_1 : ℕ) (sale_2 : ℕ) (sale_3 : ℕ) (sale_4 : ℕ) (sale_5 : ℕ) (sale_6 : ℕ)
  (avg_sale : ℕ)
  (h1 : sale_1 = 5400)
  (h2 : sale_3 = 6300)
  (h3 : sale_4 = 7200)
  (h4 : sale_5 = 4500)
  (h5 : sale_6 = 1200)
  (h_avg : avg_sale = 5600) :
  sale_2 = 9000 := 
by sorry

end NUMINAMATH_GPT_sales_second_month_l554_55481


namespace NUMINAMATH_GPT_avg_age_grandparents_is_64_l554_55486

-- Definitions of conditions
def num_grandparents : ℕ := 2
def num_parents : ℕ := 2
def num_grandchildren : ℕ := 3
def num_family_members : ℕ := num_grandparents + num_parents + num_grandchildren

def avg_age_parents : ℕ := 39
def avg_age_grandchildren : ℕ := 6
def avg_age_family : ℕ := 32

-- Total number of family members
theorem avg_age_grandparents_is_64 (G : ℕ) :
  (num_grandparents * G) + (num_parents * avg_age_parents) + (num_grandchildren * avg_age_grandchildren) = (num_family_members * avg_age_family) →
  G = 64 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_avg_age_grandparents_is_64_l554_55486


namespace NUMINAMATH_GPT_find_value_of_a_l554_55419

theorem find_value_of_a (a : ℚ) (h : a + a / 4 - 1 / 2 = 2) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a_l554_55419


namespace NUMINAMATH_GPT_andy_questions_wrong_l554_55497

variables (a b c d : ℕ)

-- Given conditions
def condition1 : Prop := a + b = c + d
def condition2 : Prop := a + d = b + c + 6
def condition3 : Prop := c = 7

-- The theorem to prove
theorem andy_questions_wrong (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 c) : a = 10 :=
by
  sorry

end NUMINAMATH_GPT_andy_questions_wrong_l554_55497


namespace NUMINAMATH_GPT_logan_average_speed_l554_55472

theorem logan_average_speed 
  (tamika_hours : ℕ)
  (tamika_speed : ℕ)
  (logan_hours : ℕ)
  (tamika_distance : ℕ)
  (logan_distance : ℕ)
  (distance_diff : ℕ)
  (diff_condition : tamika_distance = logan_distance + distance_diff) :
  tamika_hours = 8 →
  tamika_speed = 45 →
  logan_hours = 5 →
  tamika_distance = tamika_speed * tamika_hours →
  distance_diff = 85 →
  logan_distance / logan_hours = 55 :=
by
  sorry

end NUMINAMATH_GPT_logan_average_speed_l554_55472


namespace NUMINAMATH_GPT_product_divisible_by_sum_l554_55484

theorem product_divisible_by_sum (m n : ℕ) (h : ∃ k : ℕ, m * n = k * (m + n)) : m + n ≤ Nat.gcd m n * Nat.gcd m n := by
  sorry

end NUMINAMATH_GPT_product_divisible_by_sum_l554_55484


namespace NUMINAMATH_GPT_box_width_l554_55429

theorem box_width (h : ℝ) (d : ℝ) (l : ℝ) (w : ℝ) 
  (h_eq_8 : h = 8)
  (l_eq_2h : l = 2 * h)
  (d_eq_20 : d = 20) :
  w = 4 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_box_width_l554_55429


namespace NUMINAMATH_GPT_proof_problem_l554_55477

universe u

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 3, 4}
def B : Set ℕ := {1, 3}
def complement (s : Set ℕ) : Set ℕ := {x | x ∉ s}

theorem proof_problem : ((complement A ∪ A) ∪ B) = U :=
by sorry

end NUMINAMATH_GPT_proof_problem_l554_55477


namespace NUMINAMATH_GPT_maximum_value_l554_55414

variable {a b c : ℝ}

-- Conditions
variable (h : a^2 + b^2 = c^2)

theorem maximum_value (h : a^2 + b^2 = c^2) : 
  (∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ 
   (∀ x y z : ℝ, x^2 + y^2 = z^2 → (x^2 + y^2 + x*y) / z^2 ≤ 1.5)) := 
sorry

end NUMINAMATH_GPT_maximum_value_l554_55414


namespace NUMINAMATH_GPT_expression_value_l554_55440

-- Define the difference of squares identity
lemma diff_of_squares (x y : ℤ) : x^2 - y^2 = (x + y) * (x - y) :=
by sorry

-- Define the specific values for x and y
def x := 7
def y := 3

-- State the theorem to be proven
theorem expression_value : ((x^2 - y^2)^2) = 1600 :=
by sorry

end NUMINAMATH_GPT_expression_value_l554_55440


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l554_55416

theorem smallest_four_digit_multiple_of_18 : ∃ n : ℕ, n = 1008 ∧ (1000 ≤ n) ∧ (n < 10000) ∧ (n % 18 = 0) ∧ 
                                ∀ m : ℕ, ((1000 ≤ m) ∧ (m < 10000) ∧ (m % 18 = 0)) → 1008 ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_18_l554_55416


namespace NUMINAMATH_GPT_line_parallel_condition_l554_55415

theorem line_parallel_condition (a : ℝ) : (a = 2) ↔ (∀ x y : ℝ, (ax + 2 * y = 0 → x + y ≠ 1)) :=
by
  sorry

end NUMINAMATH_GPT_line_parallel_condition_l554_55415


namespace NUMINAMATH_GPT_determine_values_of_x_l554_55446

variable (x : ℝ)

theorem determine_values_of_x (h1 : 1/x < 3) (h2 : 1/x > -4) : x > 1/3 ∨ x < -1/4 := 
  sorry


end NUMINAMATH_GPT_determine_values_of_x_l554_55446


namespace NUMINAMATH_GPT_add_fractions_add_fractions_as_mixed_l554_55459

theorem add_fractions : (3 / 4) + (5 / 6) + (4 / 3) = (35 / 12) := sorry

theorem add_fractions_as_mixed : (3 / 4) + (5 / 6) + (4 / 3) = 2 + 11 / 12 := sorry

end NUMINAMATH_GPT_add_fractions_add_fractions_as_mixed_l554_55459


namespace NUMINAMATH_GPT_min_segments_of_polyline_l554_55439

theorem min_segments_of_polyline (n : ℕ) (h : n ≥ 2) : 
  ∃ s : ℕ, s = 2 * n - 2 := sorry

end NUMINAMATH_GPT_min_segments_of_polyline_l554_55439


namespace NUMINAMATH_GPT_unique_real_solution_l554_55404

theorem unique_real_solution (a : ℝ) : 
  (∀ x : ℝ, (x^3 - a * x^2 - (a + 1) * x + (a^2 - 2) = 0)) ↔ (a < 7 / 4) := 
sorry

end NUMINAMATH_GPT_unique_real_solution_l554_55404


namespace NUMINAMATH_GPT_nonagon_perimeter_l554_55480

theorem nonagon_perimeter :
  (2 + 2 + 3 + 3 + 1 + 3 + 2 + 2 + 2 = 20) := by
  sorry

end NUMINAMATH_GPT_nonagon_perimeter_l554_55480


namespace NUMINAMATH_GPT_probability_heads_exactly_2_times_three_tosses_uniform_coin_l554_55493

noncomputable def probability_heads_exactly_2_times (n k : ℕ) (p : ℚ) : ℚ :=
(n.choose k : ℚ) * p^k * (1 - p)^(n - k)

theorem probability_heads_exactly_2_times_three_tosses_uniform_coin :
  probability_heads_exactly_2_times 3 2 (1/2) = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_probability_heads_exactly_2_times_three_tosses_uniform_coin_l554_55493


namespace NUMINAMATH_GPT_project_profit_starts_from_4th_year_l554_55456

def initial_investment : ℝ := 144
def maintenance_cost (n : ℕ) : ℝ := 4 * n^2 + 40 * n
def annual_income : ℝ := 100

def net_profit (n : ℕ) : ℝ := 
  annual_income * n - maintenance_cost n - initial_investment

theorem project_profit_starts_from_4th_year :
  ∀ n : ℕ, 3 < n ∧ n < 12 → net_profit n > 0 :=
by
  intros n hn
  sorry

end NUMINAMATH_GPT_project_profit_starts_from_4th_year_l554_55456


namespace NUMINAMATH_GPT_expand_polynomial_l554_55487

theorem expand_polynomial (z : ℂ) :
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 :=
by sorry

end NUMINAMATH_GPT_expand_polynomial_l554_55487


namespace NUMINAMATH_GPT_friend_charge_per_animal_l554_55494

-- Define the conditions.
def num_cats := 2
def num_dogs := 3
def total_payment := 65

-- Define the total number of animals.
def total_animals := num_cats + num_dogs

-- Define the charge per animal per night.
def charge_per_animal := total_payment / total_animals

-- State the theorem.
theorem friend_charge_per_animal : charge_per_animal = 13 := by
  -- Proof goes here.
  sorry

end NUMINAMATH_GPT_friend_charge_per_animal_l554_55494


namespace NUMINAMATH_GPT_quadratic_distinct_roots_k_range_l554_55451

theorem quadratic_distinct_roots_k_range (k : ℝ) :
  (k - 1) * x^2 + 2 * x - 2 = 0 ∧ 
  ∀ Δ, Δ = 2^2 - 4*(k-1)*(-2) ∧ Δ > 0 ∧ (k ≠ 1) ↔ k > 1/2 ∧ k ≠ 1 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_distinct_roots_k_range_l554_55451


namespace NUMINAMATH_GPT_min_value_3_div_a_add_2_div_b_l554_55400

/-- Given positive real numbers a and b, and the condition that the lines
(a + 1)x + 2y - 1 = 0 and 3x + (b - 2)y + 2 = 0 are perpendicular,
prove that the minimum value of 3/a + 2/b is 25, given the condition 3a + 2b = 1. -/
theorem min_value_3_div_a_add_2_div_b (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (h : 3 * a + 2 * b = 1) : 3 / a + 2 / b ≥ 25 :=
sorry

end NUMINAMATH_GPT_min_value_3_div_a_add_2_div_b_l554_55400


namespace NUMINAMATH_GPT_smallest_non_representable_l554_55464

def isRepresentable (n : ℕ) : Prop :=
  ∃ a b c d : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ n = (2^a - 2^b) / (2^c - 2^d)

theorem smallest_non_representable : ∀ n : ℕ, 0 < n → ¬ isRepresentable 11 ∧ ∀ k : ℕ, 0 < k ∧ k < 11 → isRepresentable k :=
by sorry

end NUMINAMATH_GPT_smallest_non_representable_l554_55464


namespace NUMINAMATH_GPT_abc_system_proof_l554_55420

theorem abc_system_proof (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a^2 + a = b^2) (h5 : b^2 + b = c^2) (h6 : c^2 + c = a^2) :
  (a - b) * (b - c) * (c - a) = 1 :=
by
  sorry

end NUMINAMATH_GPT_abc_system_proof_l554_55420


namespace NUMINAMATH_GPT_minimum_value_l554_55403

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2*y = 1) :
  ∀ (z : ℝ), z = (1/x + 1/y) → z ≥ 3 + 2*Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_l554_55403


namespace NUMINAMATH_GPT_part1_part2_l554_55447

-- Part (1)  
theorem part1 (m : ℝ) : (∀ x : ℝ, 1 < x ∧ x < 3 → 2 * m < x ∧ x < 1 - m) ↔ (m ≤ -2) :=
sorry

-- Part (2)
theorem part2 (m : ℝ) : (∀ x : ℝ, (1 < x ∧ x < 3) → ¬ (2 * m < x ∧ x < 1 - m)) ↔ (0 ≤ m) :=
sorry

end NUMINAMATH_GPT_part1_part2_l554_55447


namespace NUMINAMATH_GPT_power_binary_representation_zero_digit_l554_55468

theorem power_binary_representation_zero_digit
  (a n s : ℕ) (ha : a > 1) (hn : n > 1) (hs : s > 0) :
  a ^ n ≠ 2 ^ s - 1 :=
by
  sorry

end NUMINAMATH_GPT_power_binary_representation_zero_digit_l554_55468


namespace NUMINAMATH_GPT_initial_volume_of_mixture_l554_55443

theorem initial_volume_of_mixture (V : ℝ) :
  let V_new := V + 8
  let initial_water := 0.20 * V
  let new_water := initial_water + 8
  let new_mixture := V_new
  new_water = 0.25 * new_mixture →
  V = 120 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_initial_volume_of_mixture_l554_55443


namespace NUMINAMATH_GPT_fraction_to_decimal_l554_55470

theorem fraction_to_decimal : (5 : ℚ) / 16 = 0.3125 := 
by
  have h1 : (5 / 16 : ℚ) = (3125 / 10000) := by sorry
  have h2 : (3125 / 10000 : ℚ) = 0.3125 := by sorry
  rw [h1, h2]

end NUMINAMATH_GPT_fraction_to_decimal_l554_55470


namespace NUMINAMATH_GPT_men_became_absent_l554_55453

theorem men_became_absent (num_men absent : ℤ) 
  (num_men_eq : num_men = 180) 
  (days_planned : ℤ) (days_planned_eq : days_planned = 55)
  (days_taken : ℤ) (days_taken_eq : days_taken = 60)
  (work_planned : ℤ) (work_planned_eq : work_planned = num_men * days_planned)
  (work_taken : ℤ) (work_taken_eq : work_taken = (num_men - absent) * days_taken)
  (work_eq : work_planned = work_taken) :
  absent = 15 :=
  by sorry

end NUMINAMATH_GPT_men_became_absent_l554_55453


namespace NUMINAMATH_GPT_value_of_f_is_29_l554_55489

noncomputable def f (x : ℕ) : ℕ := 3 * x - 4
noncomputable def g (x : ℕ) : ℕ := x^2 + 1

theorem value_of_f_is_29 :
  f (1 + g 3) = 29 := by
  sorry

end NUMINAMATH_GPT_value_of_f_is_29_l554_55489


namespace NUMINAMATH_GPT_determine_m_l554_55476

noncomputable def f (m x : ℝ) := (m^2 - m - 1) * x^(-5 * m - 3)

theorem determine_m : ∃ m : ℝ, (∀ x > 0, f m x = (m^2 - m - 1) * x^(-5 * m - 3)) ∧ (∀ x > 0, (m^2 - m - 1) * x^(-(5 * m + 3)) = (m^2 - m - 1) * x^(-5 * m - 3) → -5 * m - 3 > 0) ∧ m = -1 :=
by
  sorry

end NUMINAMATH_GPT_determine_m_l554_55476


namespace NUMINAMATH_GPT_sequence_general_term_l554_55421

/-- 
  Define the sequence a_n recursively as:
  a_1 = 2
  a_n = 2 * a_(n-1) - 1

  Prove that the general term of the sequence is:
  a_n = 2^(n-1) + 1
-/
theorem sequence_general_term {a : ℕ → ℕ} 
  (h₁ : a 1 = 2) 
  (h₂ : ∀ n, a (n + 1) = 2 * a n - 1) 
  (n : ℕ) : 
  a n = 2^(n-1) + 1 := by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l554_55421


namespace NUMINAMATH_GPT_berry_saturday_reading_l554_55457

-- Given data
def sunday_pages := 43
def monday_pages := 65
def tuesday_pages := 28
def wednesday_pages := 0
def thursday_pages := 70
def friday_pages := 56
def average_goal := 50
def days_in_week := 7

-- Calculate total pages to meet the weekly goal
def weekly_goal := days_in_week * average_goal

-- Calculate pages read so far from Sunday to Friday
def pages_read := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

-- Calculate required pages to read on Saturday
def saturday_pages_required := weekly_goal - pages_read

-- The theorem statement: Berry needs to read 88 pages on Saturday.
theorem berry_saturday_reading : saturday_pages_required = 88 := 
by {
  -- The proof is omitted as per the instructions
  sorry
}

end NUMINAMATH_GPT_berry_saturday_reading_l554_55457


namespace NUMINAMATH_GPT_two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l554_55495

theorem two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n
  (n : ℕ) (h : 2 < n) : (2 * n - 1) ^ n + (2 * n) ^ n < (2 * n + 1) ^ n :=
sorry

end NUMINAMATH_GPT_two_to_n_minus_one_pow_n_plus_two_n_pow_n_lt_two_n_plus_one_pow_n_l554_55495


namespace NUMINAMATH_GPT_arrange_desc_l554_55427

noncomputable def a : ℝ := Real.sin (33 * Real.pi / 180)
noncomputable def b : ℝ := Real.sin (35 * Real.pi / 180)
noncomputable def c : ℝ := Real.tan (35 * Real.pi / 180)
noncomputable def d : ℝ := Real.log 5

theorem arrange_desc : d > c ∧ c > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_arrange_desc_l554_55427


namespace NUMINAMATH_GPT_solution_is_option_C_l554_55462

-- Define the equation.
def equation (x y : ℤ) : Prop := x - 2 * y = 3

-- Define the given conditions as terms in Lean.
def option_A := (1, 1)   -- (x = 1, y = 1)
def option_B := (-1, 1)  -- (x = -1, y = 1)
def option_C := (1, -1)  -- (x = 1, y = -1)
def option_D := (-1, -1) -- (x = -1, y = -1)

-- The goal is to prove that option C is a solution to the equation.
theorem solution_is_option_C : equation 1 (-1) :=
by {
  -- Proof will go here
  sorry
}

end NUMINAMATH_GPT_solution_is_option_C_l554_55462


namespace NUMINAMATH_GPT_value_of_x_l554_55482

theorem value_of_x (a b x : ℝ) (h : x^2 + 4 * b^2 = (2 * a - x)^2) : 
  x = (a^2 - b^2) / a :=
by
  sorry

end NUMINAMATH_GPT_value_of_x_l554_55482


namespace NUMINAMATH_GPT_area_R_l554_55431

-- Define the given matrix as a 2x2 real matrix
def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 6, -5]

-- Define the original area of region R
def area_R : ℝ := 15

-- Define the area scaling factor as the absolute value of the determinant of A
def scaling_factor : ℝ := |Matrix.det A|

-- Prove that the area of the region R' is 585
theorem area_R' : scaling_factor * area_R = 585 := by
  sorry

end NUMINAMATH_GPT_area_R_l554_55431


namespace NUMINAMATH_GPT_percentage_discount_l554_55444

theorem percentage_discount (individual_payment_without_discount final_payment discount_per_person : ℝ)
  (h1 : 3 * individual_payment_without_discount = final_payment + 3 * discount_per_person)
  (h2 : discount_per_person = 4)
  (h3 : final_payment = 48) :
  discount_per_person / (individual_payment_without_discount * 3) * 100 = 20 :=
by
  -- Proof to be provided here
  sorry

end NUMINAMATH_GPT_percentage_discount_l554_55444


namespace NUMINAMATH_GPT_cyclist_time_no_wind_l554_55408

theorem cyclist_time_no_wind (v w : ℝ) 
    (h1 : v + w = 1 / 3) 
    (h2 : v - w = 1 / 4) : 
    1 / v = 24 / 7 := 
by
  sorry

end NUMINAMATH_GPT_cyclist_time_no_wind_l554_55408


namespace NUMINAMATH_GPT_find_f_zero_l554_55475

theorem find_f_zero (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y - x * y) 
  (h1 : f 1 = 1) : 
  f 0 = 0 := 
sorry

end NUMINAMATH_GPT_find_f_zero_l554_55475


namespace NUMINAMATH_GPT_find_a_to_satisfy_divisibility_l554_55491

theorem find_a_to_satisfy_divisibility (a : ℕ) (h₀ : 0 ≤ a) (h₁ : a < 11) (h₂ : (2 * 10^10 + a) % 11 = 0) : a = 9 :=
sorry

end NUMINAMATH_GPT_find_a_to_satisfy_divisibility_l554_55491


namespace NUMINAMATH_GPT_district_B_high_schools_l554_55492

theorem district_B_high_schools :
  ∀ (total_schools public_schools parochial_schools private_schools districtA_schools districtB_private_schools: ℕ),
  total_schools = 50 ∧ 
  public_schools = 25 ∧ 
  parochial_schools = 16 ∧ 
  private_schools = 9 ∧ 
  districtA_schools = 18 ∧ 
  districtB_private_schools = 2 ∧ 
  (∃ districtC_schools, 
     districtC_schools = public_schools / 3 + parochial_schools / 3 + private_schools / 3) →
  ∃ districtB_schools, 
    districtB_schools = total_schools - districtA_schools - (public_schools / 3 + parochial_schools / 3 + private_schools / 3) ∧ 
    districtB_schools = 5 := by
  sorry

end NUMINAMATH_GPT_district_B_high_schools_l554_55492


namespace NUMINAMATH_GPT_original_salary_l554_55466

theorem original_salary (x : ℝ)
  (h1 : x * 1.10 * 0.95 = 3135) : x = 3000 :=
by
  sorry

end NUMINAMATH_GPT_original_salary_l554_55466


namespace NUMINAMATH_GPT_lasagna_package_weight_l554_55490

theorem lasagna_package_weight 
  (beef : ℕ) 
  (noodles_needed_per_beef : ℕ) 
  (current_noodles : ℕ) 
  (packages_needed : ℕ) 
  (noodles_per_package : ℕ) 
  (H1 : beef = 10)
  (H2 : noodles_needed_per_beef = 2)
  (H3 : current_noodles = 4)
  (H4 : packages_needed = 8)
  (H5 : noodles_per_package = (2 * beef - current_noodles) / packages_needed) :
  noodles_per_package = 2 := 
by
  sorry

end NUMINAMATH_GPT_lasagna_package_weight_l554_55490


namespace NUMINAMATH_GPT_max_ab_l554_55438

theorem max_ab (a b : ℝ) (h : a + b = 1) : ab ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_ab_l554_55438


namespace NUMINAMATH_GPT_number_of_girls_l554_55412

-- Define the number of girls and boys
variables (G B : ℕ)

-- Define the conditions
def condition1 : Prop := B = 2 * G - 16
def condition2 : Prop := G + B = 68

-- The theorem we want to prove
theorem number_of_girls (h1 : condition1 G B) (h2 : condition2 G B) : G = 28 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_l554_55412


namespace NUMINAMATH_GPT_find_cosine_of_dihedral_angle_l554_55430

def dihedral_cosine (R r : ℝ) (α β : ℝ) : Prop :=
  R = 2 * r ∧ β = Real.pi / 4 → Real.cos α = 8 / 9

theorem find_cosine_of_dihedral_angle : ∃ α, ∀ R r : ℝ, dihedral_cosine R r α (Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_find_cosine_of_dihedral_angle_l554_55430


namespace NUMINAMATH_GPT_stations_visited_l554_55436

-- Define the total number of nails
def total_nails : ℕ := 560

-- Define the number of nails left at each station
def nails_per_station : ℕ := 14

-- Main theorem statement
theorem stations_visited : total_nails / nails_per_station = 40 := by
  sorry

end NUMINAMATH_GPT_stations_visited_l554_55436


namespace NUMINAMATH_GPT_papers_left_l554_55437

def total_papers_bought : ℕ := 20
def pictures_drawn_today : ℕ := 6
def pictures_drawn_yesterday_before_work : ℕ := 6
def pictures_drawn_yesterday_after_work : ℕ := 6

theorem papers_left :
  total_papers_bought - (pictures_drawn_today + pictures_drawn_yesterday_before_work + pictures_drawn_yesterday_after_work) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_papers_left_l554_55437


namespace NUMINAMATH_GPT_total_hours_watching_tv_and_playing_games_l554_55441

-- Defining the conditions provided in the problem
def hours_watching_tv_saturday : ℕ := 6
def hours_watching_tv_sunday : ℕ := 3
def hours_watching_tv_tuesday : ℕ := 2
def hours_watching_tv_thursday : ℕ := 4

def hours_playing_games_monday : ℕ := 3
def hours_playing_games_wednesday : ℕ := 5
def hours_playing_games_friday : ℕ := 1

-- The proof statement
theorem total_hours_watching_tv_and_playing_games :
  hours_watching_tv_saturday + hours_watching_tv_sunday + hours_watching_tv_tuesday + hours_watching_tv_thursday
  + hours_playing_games_monday + hours_playing_games_wednesday + hours_playing_games_friday = 24 := 
by
  sorry

end NUMINAMATH_GPT_total_hours_watching_tv_and_playing_games_l554_55441


namespace NUMINAMATH_GPT_abc_min_value_l554_55432

open Real

theorem abc_min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_sum : a + b + c = 1) (h_bound : a ≤ b ∧ b ≤ c ∧ c ≤ 3 * a) :
  3 * a * a * (1 - 4 * a) = (9/343) := 
sorry

end NUMINAMATH_GPT_abc_min_value_l554_55432


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l554_55449

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ -8/3 < k ∧ k < 6 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l554_55449


namespace NUMINAMATH_GPT_part_1_a_part_1_b_part_2_l554_55471

open Set

variable (a : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def compl_U_A : Set ℝ := compl A

theorem part_1_a :
  A ∩ B 1 = {x : ℝ | x < -2} :=
by
  sorry

theorem part_1_b :
  A ∪ B 1 = {x : ℝ | x > 2 ∨ x ≤ 1} :=
by
  sorry

theorem part_2 :
  compl_U_A ⊆ B a → a ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_part_1_a_part_1_b_part_2_l554_55471


namespace NUMINAMATH_GPT_fraction_c_over_d_l554_55435

-- Assume that we have a polynomial equation ax^3 + bx^2 + cx + d = 0 with roots 1, 2, 3
def polynomial (a b c d x : ℝ) : Prop := a * x^3 + b * x^2 + c * x + d = 0

-- The roots of the polynomial are 1, 2, 3
def roots (a b c d : ℝ) : Prop := polynomial a b c d 1 ∧ polynomial a b c d 2 ∧ polynomial a b c d 3

-- Vieta's formulas give us the relation for c and d in terms of the roots
theorem fraction_c_over_d (a b c d : ℝ) (h : roots a b c d) : c / d = -11 / 6 :=
sorry

end NUMINAMATH_GPT_fraction_c_over_d_l554_55435


namespace NUMINAMATH_GPT_marbles_leftover_l554_55410

theorem marbles_leftover (r p : ℕ) (hr : r % 8 = 5) (hp : p % 8 = 7) : (r + p) % 8 = 4 :=
by
  sorry

end NUMINAMATH_GPT_marbles_leftover_l554_55410


namespace NUMINAMATH_GPT_P_subset_Q_l554_55442

def P : Set ℝ := {x | x > 1}
def Q : Set ℝ := {x | x > 0}

theorem P_subset_Q : P ⊂ Q :=
by
  sorry

end NUMINAMATH_GPT_P_subset_Q_l554_55442


namespace NUMINAMATH_GPT_stone_145_is_5_l554_55411

theorem stone_145_is_5 :
  ∀ (n : ℕ), (1 ≤ n ∧ n ≤ 15) → (145 % 28) = 5 → n = 5 :=
by
  intros n h h145
  sorry

end NUMINAMATH_GPT_stone_145_is_5_l554_55411


namespace NUMINAMATH_GPT_equivalent_statements_l554_55434

-- Definitions
variables (P Q : Prop)

-- Original statement
def original_statement := P → Q

-- Statements
def statement_I := P → Q
def statement_II := Q → P
def statement_III := ¬ Q → ¬ P
def statement_IV := ¬ P ∨ Q

-- Proof problem
theorem equivalent_statements : 
  (statement_III P Q ∧ statement_IV P Q) ↔ original_statement P Q :=
sorry

end NUMINAMATH_GPT_equivalent_statements_l554_55434


namespace NUMINAMATH_GPT_clean_time_per_room_l554_55474

variable (h : ℕ)

-- Conditions
def floors := 4
def rooms_per_floor := 10
def total_rooms := floors * rooms_per_floor
def hourly_wage := 15
def total_earnings := 3600

-- Question and condition mapping to conclusion
theorem clean_time_per_room (H1 : total_rooms = 40) 
                            (H2 : total_earnings = 240 * hourly_wage) 
                            (H3 : 240 = 40 * h) :
                            h = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_clean_time_per_room_l554_55474


namespace NUMINAMATH_GPT_regular_price_of_Pony_jeans_l554_55426

theorem regular_price_of_Pony_jeans 
(Fox_price : ℝ) 
(Pony_price : ℝ) 
(savings : ℝ) 
(Fox_discount_rate : ℝ) 
(Pony_discount_rate : ℝ)
(h1 : Fox_price = 15)
(h2 : savings = 8.91)
(h3 : Fox_discount_rate + Pony_discount_rate = 0.22)
(h4 : Pony_discount_rate = 0.10999999999999996) : Pony_price = 18 := 
sorry

end NUMINAMATH_GPT_regular_price_of_Pony_jeans_l554_55426


namespace NUMINAMATH_GPT_original_equation_solution_l554_55401

noncomputable def original_equation : Prop :=
  ∃ Y P A K P O C : ℕ,
  (Y = 5) ∧ (P = 2) ∧ (A = 0) ∧ (K = 2) ∧ (P = 4) ∧ (O = 0) ∧ (C = 0) ∧
  (Y.factorial * P.factorial * A.factorial = K * 10000 + P * 1000 + O * 100 + C * 10 + C)

theorem original_equation_solution : original_equation :=
  sorry

end NUMINAMATH_GPT_original_equation_solution_l554_55401


namespace NUMINAMATH_GPT_kendra_shirts_needed_l554_55450

def school_shirts_per_week : Nat := 5
def club_shirts_per_week : Nat := 3
def spirit_day_shirt_per_week : Nat := 1
def saturday_shirts_per_week : Nat := 3
def sunday_shirts_per_week : Nat := 3
def family_reunion_shirt_per_month : Nat := 1

def total_shirts_needed_per_week : Nat :=
  school_shirts_per_week + club_shirts_per_week + spirit_day_shirt_per_week +
  saturday_shirts_per_week + sunday_shirts_per_week

def total_shirts_needed_per_four_weeks : Nat :=
  total_shirts_needed_per_week * 4 + family_reunion_shirt_per_month

theorem kendra_shirts_needed : total_shirts_needed_per_four_weeks = 61 := by
  sorry

end NUMINAMATH_GPT_kendra_shirts_needed_l554_55450


namespace NUMINAMATH_GPT_sum_of_first_11_terms_of_arithmetic_seq_l554_55405

noncomputable def arithmetic_sequence_SUM (a d : ℚ) : ℚ :=  
  11 / 2 * (2 * a + 10 * d)

theorem sum_of_first_11_terms_of_arithmetic_seq
  (a d : ℚ)
  (h : a + 2 * d + a + 6 * d = 16) :
  arithmetic_sequence_SUM a d = 88 := 
  sorry

end NUMINAMATH_GPT_sum_of_first_11_terms_of_arithmetic_seq_l554_55405


namespace NUMINAMATH_GPT_obtuse_angle_condition_l554_55413

def dot_product (a b : (ℝ × ℝ)) : ℝ := a.1 * b.1 + a.2 * b.2

def is_obtuse_angle (a b : (ℝ × ℝ)) : Prop := dot_product a b < 0

def is_parallel (a b : (ℝ × ℝ)) : Prop := ∃ k : ℝ, b = (k * a.1, k * a.2)

theorem obtuse_angle_condition :
  (∀ (x : ℝ), x > 0 → is_obtuse_angle (-1, 0) (x, 1 - x) ∧ ¬is_parallel (-1, 0) (x, 1 - x)) ∧ 
  (∀ (x : ℝ), is_obtuse_angle (-1, 0) (x, 1 - x) → x > 0) :=
sorry

end NUMINAMATH_GPT_obtuse_angle_condition_l554_55413


namespace NUMINAMATH_GPT_trig_identity_l554_55473

theorem trig_identity (α : ℝ) (h : Real.tan α = 4) : (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = 9 := by
  sorry

end NUMINAMATH_GPT_trig_identity_l554_55473


namespace NUMINAMATH_GPT_distinct_ordered_pairs_l554_55460

theorem distinct_ordered_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h : 1/m + 1/n = 1/5) : 
  ∃! (m n : ℕ), m > 0 ∧ n > 0 ∧ (1 / m + 1 / n = 1 / 5) :=
sorry

end NUMINAMATH_GPT_distinct_ordered_pairs_l554_55460


namespace NUMINAMATH_GPT_find_slope_l554_55469

theorem find_slope (k : ℝ) : (∃ x : ℝ, (y = k * x + 2) ∧ (y = 0) ∧ (abs x = 4)) ↔ (k = 1/2 ∨ k = -1/2) := by
  sorry

end NUMINAMATH_GPT_find_slope_l554_55469


namespace NUMINAMATH_GPT_percent_increase_in_sales_l554_55465

theorem percent_increase_in_sales :
  let new := 416
  let old := 320
  (new - old) / old * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_percent_increase_in_sales_l554_55465


namespace NUMINAMATH_GPT_part_I_part_II_l554_55452

-- Definitions of the sets A, B, and C
def A : Set ℝ := { x | x ≤ -1 ∨ x ≥ 3 }
def B : Set ℝ := { x | 1 ≤ x ∧ x ≤ 6 }
def C (m : ℝ) : Set ℝ := { x | m + 1 ≤ x ∧ x ≤ 2 * m }

-- Proof statements
theorem part_I : A ∩ B = { x | 3 ≤ x ∧ x ≤ 6 } :=
by sorry

theorem part_II (m : ℝ) : (B ∪ C m = B) → (m ≤ 3) :=
by sorry

end NUMINAMATH_GPT_part_I_part_II_l554_55452


namespace NUMINAMATH_GPT_angle_relation_l554_55454

theorem angle_relation (R : ℝ) (hR : R > 0) (d : ℝ) (hd : d > R) 
  (α β : ℝ) : β = 3 * α :=
sorry

end NUMINAMATH_GPT_angle_relation_l554_55454
