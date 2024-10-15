import Mathlib

namespace NUMINAMATH_GPT_min_value_fraction_l1813_181387

theorem min_value_fraction (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x * y = x + 2 * y + 6) : 
  (∃ (z : ℝ), z = 1 / x + 1 / (2 * y) ∧ z ≥ 1 / 3) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l1813_181387


namespace NUMINAMATH_GPT_tangent_line_to_curve_at_point_l1813_181322

theorem tangent_line_to_curve_at_point :
  ∀ (x y : ℝ),
  (y = 2 * Real.log x) →
  (x = 2) →
  (y = 2 * Real.log 2) →
  (x - y + 2 * Real.log 2 - 2 = 0) := by
  sorry

end NUMINAMATH_GPT_tangent_line_to_curve_at_point_l1813_181322


namespace NUMINAMATH_GPT_factorize_expression_l1813_181343

theorem factorize_expression (m x : ℝ) : m * x^2 - 6 * m * x + 9 * m = m * (x - 3)^2 :=
by
  sorry

end NUMINAMATH_GPT_factorize_expression_l1813_181343


namespace NUMINAMATH_GPT_total_pairs_of_shoes_tried_l1813_181354

theorem total_pairs_of_shoes_tried (first_store_pairs second_store_additional third_store_pairs fourth_store_factor : ℕ) 
  (h_first : first_store_pairs = 7)
  (h_second : second_store_additional = 2)
  (h_third : third_store_pairs = 0)
  (h_fourth : fourth_store_factor = 2) :
  first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs + 
    (fourth_store_factor * (first_store_pairs + (first_store_pairs + second_store_additional) + third_store_pairs)) = 48 := 
  by 
    sorry

end NUMINAMATH_GPT_total_pairs_of_shoes_tried_l1813_181354


namespace NUMINAMATH_GPT_pure_imaginary_denom_rationalization_l1813_181383

theorem pure_imaginary_denom_rationalization (a : ℝ) : 
  (∃ b : ℝ, 1 - a * Complex.I * Complex.I = b * Complex.I) → a = 0 :=
by
  sorry

end NUMINAMATH_GPT_pure_imaginary_denom_rationalization_l1813_181383


namespace NUMINAMATH_GPT_black_lambs_count_l1813_181385

/-- Definition of the total number of lambs. -/
def total_lambs : Nat := 6048

/-- Definition of the number of white lambs. -/
def white_lambs : Nat := 193

/-- Prove that the number of black lambs is 5855. -/
theorem black_lambs_count : total_lambs - white_lambs = 5855 := by
  sorry

end NUMINAMATH_GPT_black_lambs_count_l1813_181385


namespace NUMINAMATH_GPT_find_second_number_l1813_181388

theorem find_second_number (x : ℕ) : 9548 + x = 3362 + 13500 → x = 7314 := by
  sorry

end NUMINAMATH_GPT_find_second_number_l1813_181388


namespace NUMINAMATH_GPT_track_length_proof_l1813_181371

noncomputable def track_length : ℝ :=
  let x := 541.67
  x

theorem track_length_proof
  (p : ℝ)
  (q : ℝ)
  (h1 : p = 1 / 4)
  (h2 : q = 120)
  (h3 : ¬(p = q))
  (h4 : ∃ r : ℝ, r = 180)
  (speed_constant : ∃ b_speed, ∃ s_speed, b_speed * t = q ∧ s_speed * t = r) :
  track_length = 541.67 :=
sorry

end NUMINAMATH_GPT_track_length_proof_l1813_181371


namespace NUMINAMATH_GPT_smallest_c_in_range_l1813_181359

-- Define the quadratic function g(x)
def g (x c : ℝ) : ℝ := 2 * x ^ 2 - 4 * x + c

-- Define the condition for c
def in_range_5 (c : ℝ) : Prop :=
  ∃ x : ℝ, g x c = 5

-- The theorem stating that the smallest value of c for which 5 is in the range of g is 7
theorem smallest_c_in_range : ∃ c : ℝ, c = 7 ∧ ∀ c' : ℝ, (in_range_5 c' → 7 ≤ c') :=
sorry

end NUMINAMATH_GPT_smallest_c_in_range_l1813_181359


namespace NUMINAMATH_GPT_gcd_84_120_eq_12_l1813_181393

theorem gcd_84_120_eq_12 : Int.gcd 84 120 = 12 := by
  sorry

end NUMINAMATH_GPT_gcd_84_120_eq_12_l1813_181393


namespace NUMINAMATH_GPT_max_value_of_f_l1813_181375

variable (n : ℕ)

-- Define the quadratic function with coefficients a, b, and c.
noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^2 + b * x + c

-- Given conditions
axiom f_n : ∃ a b c, f n a b c = 6
axiom f_n1 : ∃ a b c, f (n + 1) a b c = 14
axiom f_n2 : ∃ a b c, f (n + 2) a b c = 14

-- The main goal is to prove the maximum value of f(x) is 15.
theorem max_value_of_f : ∃ a b c, (∀ x : ℝ, f x a b c ≤ 15) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1813_181375


namespace NUMINAMATH_GPT_simplify_fraction_l1813_181386

theorem simplify_fraction : 
  (1722^2 - 1715^2) / (1731^2 - 1708^2) = (7 * 3437) / (23 * 3439) :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1813_181386


namespace NUMINAMATH_GPT_find_n_l1813_181312

theorem find_n (n : ℕ) (h : n ≥ 2) : 
  (∀ (i j : ℕ), 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔ ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1813_181312


namespace NUMINAMATH_GPT_problem1_problem2_l1813_181314

-- For problem (1)
theorem problem1 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + c = 1) :
  (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c := sorry

-- For problem (2)
theorem problem2 (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : b^2 = a * c) :
  a^2 + b^2 + c^2 > (a - b + c)^2 := sorry

end NUMINAMATH_GPT_problem1_problem2_l1813_181314


namespace NUMINAMATH_GPT_rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l1813_181320

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

noncomputable def transition_rule (n : ℕ) : ℕ :=
  2 * (sum_of_digits n)

theorem rule_for_sequence :
  transition_rule 3 = 6 ∧ transition_rule 6 = 12 :=
by
  sorry

theorem natural_number_self_map :
  ∀ n : ℕ, transition_rule n = n ↔ n = 18 :=
by
  sorry

theorem power_of_2_to_single_digit :
  ∃ x : ℕ, transition_rule (2^1991) = x ∧ x < 10 :=
by
  sorry

end NUMINAMATH_GPT_rule_for_sequence_natural_number_self_map_power_of_2_to_single_digit_l1813_181320


namespace NUMINAMATH_GPT_number_of_devices_bought_l1813_181364

-- Define the essential parameters
def original_price : Int := 800000
def discounted_price : Int := 450000
def total_discount : Int := 16450000

-- Define the main statement to prove
theorem number_of_devices_bought : (total_discount / (original_price - discounted_price) = 47) :=
by
  -- The essential proof is skipped here with sorry
  sorry

end NUMINAMATH_GPT_number_of_devices_bought_l1813_181364


namespace NUMINAMATH_GPT_ellipse_focus_eccentricity_l1813_181311

theorem ellipse_focus_eccentricity (m : ℝ) :
  (∀ x y : ℝ, (x^2 / 2) + (y^2 / m) = 1 → y = 0 ∨ x = 0) ∧
  (∀ e : ℝ, e = 1 / 2) →
  m = 3 / 2 :=
sorry

end NUMINAMATH_GPT_ellipse_focus_eccentricity_l1813_181311


namespace NUMINAMATH_GPT_length_of_each_lateral_edge_l1813_181335

-- Define the concept of a prism with a certain number of vertices and lateral edges
structure Prism where
  vertices : ℕ
  lateral_edges : ℕ

-- Example specific to the problem: Define the conditions given in the problem statement
def given_prism : Prism := { vertices := 12, lateral_edges := 6 }
def sum_lateral_edges : ℕ := 30

-- The main proof statement: Prove the length of each lateral edge
theorem length_of_each_lateral_edge (p : Prism) (h : p = given_prism) :
  (sum_lateral_edges / p.lateral_edges) = 5 :=
by 
  -- The details of the proof will replace 'sorry'
  sorry

end NUMINAMATH_GPT_length_of_each_lateral_edge_l1813_181335


namespace NUMINAMATH_GPT_range_of_x_l1813_181307

theorem range_of_x (x a1 a2 y : ℝ) (d r : ℝ) (hx : x ≠ 0) 
  (h_arith : a1 = x + d ∧ a2 = x + 2 * d ∧ y = x + 3 * d)
  (h_geom : b1 = x * r ∧ b2 = x * r^2 ∧ y = x * r^3) : 4 ≤ x :=
by
  -- Assume x ≠ 0 as given and the sequences are arithmetic and geometric
  have hx3d := h_arith.2.2
  have hx3r := h_geom.2.2
  -- Substituting y in both sequences
  simp only [hx3d, hx3r] at *
  -- Solving for d and determining constraints
  sorry

end NUMINAMATH_GPT_range_of_x_l1813_181307


namespace NUMINAMATH_GPT_village_transportation_problem_l1813_181310

noncomputable def comb (n k : ℕ) : ℕ := Nat.choose n k

variable (total odd : ℕ) (a : ℕ)

theorem village_transportation_problem 
  (h_total : total = 15)
  (h_odd : odd = 7)
  (h_selected : 10 = 10)
  (h_eq : (comb 7 4) * (comb 8 6) / (comb 15 10) = (comb 7 (10 - a)) * (comb 8 a) / (comb 15 10)) :
  a = 6 := 
sorry

end NUMINAMATH_GPT_village_transportation_problem_l1813_181310


namespace NUMINAMATH_GPT_parametric_line_l1813_181319

theorem parametric_line (s m : ℤ) :
  (∀ t : ℤ, ∃ x y : ℤ, 
    y = 5 * x - 7 ∧
    x = s + 6 * t ∧ y = 3 + m * t ) → 
  (s = 2 ∧ m = 30) :=
by
  sorry

end NUMINAMATH_GPT_parametric_line_l1813_181319


namespace NUMINAMATH_GPT_amount_spent_on_tumbler_l1813_181321

def initial_amount : ℕ := 50
def spent_on_coffee : ℕ := 10
def amount_left : ℕ := 10
def total_spent : ℕ := initial_amount - amount_left

theorem amount_spent_on_tumbler : total_spent - spent_on_coffee = 30 := by
  sorry

end NUMINAMATH_GPT_amount_spent_on_tumbler_l1813_181321


namespace NUMINAMATH_GPT_correct_statements_l1813_181337

variables (a : Nat → ℤ) (d : ℤ)

-- Suppose {a_n} is an arithmetic sequence with common difference d
def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * d)) / 2

-- Conditions: S_11 > 0 and S_12 < 0
axiom S11_pos : S a d 11 > 0
axiom S12_neg : S a d 12 < 0

-- The goal is to determine which statements are correct
theorem correct_statements : (d < 0) ∧ (∀ n, 1 ≤ n → n ≤ 12 → S a d 6 ≥ S a d n ∧ S a d 6 ≠ S a d 11 ) := 
sorry

end NUMINAMATH_GPT_correct_statements_l1813_181337


namespace NUMINAMATH_GPT_coins_problem_l1813_181327

theorem coins_problem (x y : ℕ) (h1 : x + y = 20) (h2 : x + 5 * y = 80) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_coins_problem_l1813_181327


namespace NUMINAMATH_GPT_correct_system_of_equations_l1813_181334

theorem correct_system_of_equations (x y : ℝ) 
  (h1 : x - y = 5) (h2 : y - (1/2) * x = 5) : 
  (x - y = 5) ∧ (y - (1/2) * x = 5) :=
by { sorry }

end NUMINAMATH_GPT_correct_system_of_equations_l1813_181334


namespace NUMINAMATH_GPT_second_player_wins_l1813_181369

-- Piles of balls and game conditions
def two_pile_game (pile1 pile2 : ℕ) : Prop :=
  ∀ (player1_turn : ℕ → Prop) (player2_turn : ℕ → Prop),
    (∀ n : ℕ, player1_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player1's move
    (∀ n : ℕ, player2_turn n → (pile1 ≥ n ∧ pile2 ≥ n) ∨ pile1 ≥ n ∨ pile2 ≥ n) → -- player2's move
    -- - Second player has a winning strategy
    ∃ (win_strategy : ℕ → ℕ), ∀ k : ℕ, player1_turn k → player2_turn (win_strategy k) 

-- Lean statement of the problem
theorem second_player_wins : ∀ (pile1 pile2 : ℕ), pile1 = 30 ∧ pile2 = 30 → two_pile_game pile1 pile2 :=
  by
    intros pile1 pile2 h
    sorry  -- Placeholder for the proof


end NUMINAMATH_GPT_second_player_wins_l1813_181369


namespace NUMINAMATH_GPT_b_range_condition_l1813_181370

theorem b_range_condition (b : ℝ) : 
  -2 * Real.sqrt 6 < b ∧ b < 2 * Real.sqrt 6 ↔ (b^2 - 24) < 0 :=
by
  sorry

end NUMINAMATH_GPT_b_range_condition_l1813_181370


namespace NUMINAMATH_GPT_relationship_between_abc_l1813_181376

noncomputable def a : ℝ := (0.6 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (1.5 : ℝ)
noncomputable def c : ℝ := (1.5 : ℝ) ^ (0.6 : ℝ)

theorem relationship_between_abc : c > a ∧ a > b := sorry

end NUMINAMATH_GPT_relationship_between_abc_l1813_181376


namespace NUMINAMATH_GPT_slips_with_3_l1813_181367

variable (total_slips : ℕ) (expected_value : ℚ) (num_slips_with_3 : ℕ)

def num_slips_with_9 := total_slips - num_slips_with_3

def expected_value_calc (total_slips expected_value : ℚ) (num_slips_with_3 num_slips_with_9 : ℕ) : ℚ :=
  (num_slips_with_3 / total_slips) * 3 + (num_slips_with_9 / total_slips) * 9

theorem slips_with_3 (h1 : total_slips = 15) (h2 : expected_value = 5.4)
  (h3 : expected_value_calc total_slips expected_value num_slips_with_3 (num_slips_with_9 total_slips num_slips_with_3) = expected_value) :
  num_slips_with_3 = 9 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_slips_with_3_l1813_181367


namespace NUMINAMATH_GPT_maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l1813_181392

noncomputable def maxCubeSideLength (a b c : ℝ) : ℝ :=
  a * b * c / (a * b + b * c + a * c)

noncomputable def maxRectParallelepipedDims (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a / 3, b / 3, c / 3)

theorem maxCubeSideLength_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxCubeSideLength a b c = a * b * c / (a * b + b * c + a * c) :=
sorry

theorem maxRectParallelepipedDims_correct (a b c : ℝ) :
  a > 0 → b > 0 → c > 0 →
  maxRectParallelepipedDims a b c = (a / 3, b / 3, c / 3) :=
sorry

end NUMINAMATH_GPT_maxCubeSideLength_correct_maxRectParallelepipedDims_correct_l1813_181392


namespace NUMINAMATH_GPT_base7_addition_sum_l1813_181326

theorem base7_addition_sum :
  let n1 := 256
  let n2 := 463
  let n3 := 132
  n1 + n2 + n3 = 1214 := sorry

end NUMINAMATH_GPT_base7_addition_sum_l1813_181326


namespace NUMINAMATH_GPT_right_triangle_legs_l1813_181309

theorem right_triangle_legs (a b : ℤ) (ha : 0 ≤ a) (hb : 0 ≤ b) (h : a^2 + b^2 = 65^2) : 
  a = 16 ∧ b = 63 ∨ a = 63 ∧ b = 16 :=
sorry

end NUMINAMATH_GPT_right_triangle_legs_l1813_181309


namespace NUMINAMATH_GPT_probability_of_receiving_1_l1813_181347

-- Define the probabilities and events
def P_A : ℝ := 0.5
def P_not_A : ℝ := 0.5
def P_B_given_A : ℝ := 0.9
def P_not_B_given_A : ℝ := 0.1
def P_B_given_not_A : ℝ := 0.05
def P_not_B_given_not_A : ℝ := 0.95

-- The main theorem that needs to be proved
theorem probability_of_receiving_1 : 
  (P_A * P_not_B_given_A + P_not_A * P_not_B_given_not_A) = 0.525 := by
  sorry

end NUMINAMATH_GPT_probability_of_receiving_1_l1813_181347


namespace NUMINAMATH_GPT_geometric_series_terms_l1813_181396

theorem geometric_series_terms 
    (b1 q : ℝ)
    (h₁ : (b1^2 / (1 + q + q^2)) = 12)
    (h₂ : (b1^2 / (1 + q^2)) = (36 / 5)) :
    (b1 = 3 ∨ b1 = -3) ∧ q = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_terms_l1813_181396


namespace NUMINAMATH_GPT_odd_number_as_difference_of_squares_l1813_181372

theorem odd_number_as_difference_of_squares (n : ℤ) (h : ∃ k : ℤ, n = 2 * k + 1) :
  ∃ a b : ℤ, n = a^2 - b^2 :=
by
  sorry

end NUMINAMATH_GPT_odd_number_as_difference_of_squares_l1813_181372


namespace NUMINAMATH_GPT_max_sum_hex_digits_l1813_181340

theorem max_sum_hex_digits 
  (a b c : ℕ) (y : ℕ) 
  (h_a : 0 ≤ a ∧ a < 16)
  (h_b : 0 ≤ b ∧ b < 16)
  (h_c : 0 ≤ c ∧ c < 16)
  (h_y : 0 < y ∧ y ≤ 16)
  (h_fraction : (a * 256 + b * 16 + c) * y = 4096) : 
  a + b + c ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_sum_hex_digits_l1813_181340


namespace NUMINAMATH_GPT_least_value_of_a_l1813_181361

theorem least_value_of_a (a : ℝ) (h : a^2 - 12 * a + 35 ≤ 0) : 5 ≤ a :=
by {
  sorry
}

end NUMINAMATH_GPT_least_value_of_a_l1813_181361


namespace NUMINAMATH_GPT_worker_C_work_rate_worker_C_days_l1813_181342

theorem worker_C_work_rate (A B C: ℚ) (hA: A = 1/10) (hB: B = 1/15) (hABC: A + B + C = 1/4) : C = 1/12 := 
by
  sorry

theorem worker_C_days (C: ℚ) (hC: C = 1/12) : 1 / C = 12 :=
by
  sorry

end NUMINAMATH_GPT_worker_C_work_rate_worker_C_days_l1813_181342


namespace NUMINAMATH_GPT_g_at_4_l1813_181330

noncomputable def f (x : ℝ) : ℝ := 4 / (3 - x)

noncomputable def f_inv (y : ℝ) : ℝ := (3 * y - 4) / y

noncomputable def g (x : ℝ) : ℝ := 1 / (f_inv x) + 5

theorem g_at_4 : g 4 = 11 / 2 :=
by
  sorry

end NUMINAMATH_GPT_g_at_4_l1813_181330


namespace NUMINAMATH_GPT_range_of_a_l1813_181328

noncomputable def f (x a : ℝ) := x^2 - a * x
noncomputable def g (x : ℝ) := Real.exp x
noncomputable def h (x : ℝ) := x - (Real.log x / x)

theorem range_of_a :
  ∀ a : ℝ, (∃ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 ∧ (f x a = Real.log x)) ↔ (1 ≤ a ∧ a ≤ Real.exp 1 + 1 / Real.exp 1) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1813_181328


namespace NUMINAMATH_GPT_bacteria_population_at_15_l1813_181353

noncomputable def bacteria_population (t : ℕ) : ℕ := 
  20 * 2 ^ (t / 3)

theorem bacteria_population_at_15 : bacteria_population 15 = 640 := by
  sorry

end NUMINAMATH_GPT_bacteria_population_at_15_l1813_181353


namespace NUMINAMATH_GPT_coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l1813_181373

def coprime_distinct_remainders (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : Prop :=
  ∃ (a : Fin m → ℤ) (b : Fin k → ℤ),
    (∀ (i : Fin m) (j : Fin k), ∀ (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k))

def not_coprime_congruent_product (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : Prop :=
  ∀ (a : Fin m → ℤ) (b : Fin k → ℤ),
    ∃ (i : Fin m) (j : Fin k) (s : Fin m) (t : Fin k),
      (i ≠ s ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a s * b t) % (m * k)

-- Example statement to assert the existence of the above properties
theorem coprime_mk_has_distinct_products 
  (m k : ℕ) (coprime_mk : Nat.gcd m k = 1) : coprime_distinct_remainders m k coprime_mk :=
sorry

theorem not_coprime_mk_has_congruent_products 
  (m k : ℕ) (not_coprime_mk : Nat.gcd m k > 1) : not_coprime_congruent_product m k not_coprime_mk :=
sorry

end NUMINAMATH_GPT_coprime_mk_has_distinct_products_not_coprime_mk_has_congruent_products_l1813_181373


namespace NUMINAMATH_GPT_symmetric_line_eq_l1813_181315

-- Given lines
def line₁ (x y : ℝ) : Prop := 2 * x - y + 1 = 0
def mirror_line (x y : ℝ) : Prop := y = -x

-- Definition of symmetry about the line y = -x
def symmetric_about (l₁ l₂: ℝ → ℝ → Prop) : Prop :=
∀ x y, l₁ x y ↔ l₂ y (-x)

-- Definition of line l₂ that is symmetric to line₁ about the mirror_line
def line₂ (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Theorem stating that the symmetric line to line₁ about y = -x is line₂
theorem symmetric_line_eq :
  symmetric_about line₁ line₂ :=
sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1813_181315


namespace NUMINAMATH_GPT_find_x_l1813_181352

theorem find_x (x : ℕ) (h : 2^10 = 32^x) (h32 : 32 = 2^5) : x = 2 :=
sorry

end NUMINAMATH_GPT_find_x_l1813_181352


namespace NUMINAMATH_GPT_sum_diff_reciprocals_equals_zero_l1813_181360

theorem sum_diff_reciprocals_equals_zero
  (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : (1 / (a + 1)) + (1 / (a - 1)) + (1 / (b + 1)) + (1 / (b - 1)) = 0) :
  (a + b) - (1 / a + 1 / b) = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_diff_reciprocals_equals_zero_l1813_181360


namespace NUMINAMATH_GPT_remainder_101_mul_103_mod_11_l1813_181356

theorem remainder_101_mul_103_mod_11 : (101 * 103) % 11 = 8 :=
by
  sorry

end NUMINAMATH_GPT_remainder_101_mul_103_mod_11_l1813_181356


namespace NUMINAMATH_GPT_smallest_positive_value_l1813_181316

theorem smallest_positive_value (a b c d e : ℝ) (h1 : a = 8 - 2 * Real.sqrt 14) 
  (h2 : b = 2 * Real.sqrt 14 - 8) 
  (h3 : c = 20 - 6 * Real.sqrt 10) 
  (h4 : d = 64 - 16 * Real.sqrt 4) 
  (h5 : e = 16 * Real.sqrt 4 - 64) :
  a = 8 - 2 * Real.sqrt 14 ∧ 0 < a ∧ a < c ∧ a < d :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_value_l1813_181316


namespace NUMINAMATH_GPT_a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l1813_181394

-- Definitions for the point P and movements
def move (P : ℤ) (flip : Bool) : ℤ :=
  if flip then P + 1 else -P

-- Definitions for probabilities
def probability_of_event (events : ℕ) (successful : ℕ) : ℚ :=
  successful / events

def probability_a3_zero : ℚ :=
  probability_of_event 8 2  -- 2 out of 8 sequences lead to a3 = 0

def probability_a4_one : ℚ :=
  probability_of_event 16 2  -- 2 out of 16 sequences lead to a4 = 1

noncomputable def probability_an_n_minus_3 (n : ℕ) : ℚ :=
  if n < 3 then 0 else (n - 1) / (2 ^ n)

-- Statements to prove
theorem a3_probability_is_one_fourth : probability_a3_zero = 1/4 := by
  sorry

theorem a4_probability_is_one_eighth : probability_a4_one = 1/8 := by
  sorry

theorem an_n_minus_3_probability (n : ℕ) (hn : n ≥ 3) : probability_an_n_minus_3 n = (n - 1) / (2^n) := by
  sorry

end NUMINAMATH_GPT_a3_probability_is_one_fourth_a4_probability_is_one_eighth_an_n_minus_3_probability_l1813_181394


namespace NUMINAMATH_GPT_sum_of_vertices_l1813_181317

theorem sum_of_vertices (pentagon_vertices : Nat := 5) (hexagon_vertices : Nat := 6) :
  (2 * pentagon_vertices) + (2 * hexagon_vertices) = 22 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_vertices_l1813_181317


namespace NUMINAMATH_GPT_quadrilateral_area_l1813_181308

theorem quadrilateral_area {AB BC : ℝ} (hAB : AB = 4) (hBC : BC = 8) :
  ∃ area : ℝ, area = 16 := by
  sorry

end NUMINAMATH_GPT_quadrilateral_area_l1813_181308


namespace NUMINAMATH_GPT_perpendicular_lines_a_eq_neg6_l1813_181379

theorem perpendicular_lines_a_eq_neg6 
  (a : ℝ) 
  (h1 : ∀ x y : ℝ, ax + 2*y + 1 = 0) 
  (h2 : ∀ x y : ℝ, x + 3*y - 2 = 0) 
  (h_perpendicular : ∀ m1 m2 : ℝ, m1 * m2 = -1) : 
  a = -6 := 
by 
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_eq_neg6_l1813_181379


namespace NUMINAMATH_GPT_percent_equivalence_l1813_181302

theorem percent_equivalence (y : ℝ) (h : y ≠ 0) : 0.21 * y = 0.21 * y :=
by sorry

end NUMINAMATH_GPT_percent_equivalence_l1813_181302


namespace NUMINAMATH_GPT_slope_of_line_l1813_181339

def point1 : ℝ × ℝ := (2, 3)
def point2 : ℝ × ℝ := (4, 5)

theorem slope_of_line : 
  let (x1, y1) := point1
  let (x2, y2) := point2
  (x2 - x1) ≠ 0 → (y2 - y1) / (x2 - x1) = 1 := by
  sorry

end NUMINAMATH_GPT_slope_of_line_l1813_181339


namespace NUMINAMATH_GPT_curling_teams_l1813_181357

-- Define the problem conditions and state the theorem
theorem curling_teams (x : ℕ) (h : x * (x - 1) / 2 = 45) : x = 10 :=
sorry

end NUMINAMATH_GPT_curling_teams_l1813_181357


namespace NUMINAMATH_GPT_largest_possible_s_l1813_181350

theorem largest_possible_s (r s : ℕ) (h1 : r ≥ s) (h2 : s ≥ 3) 
  (h3 : ((r - 2) * 180 : ℚ) / r = (29 / 28) * ((s - 2) * 180 / s)) :
    s = 114 := by sorry

end NUMINAMATH_GPT_largest_possible_s_l1813_181350


namespace NUMINAMATH_GPT_zero_exponent_rule_proof_l1813_181355

-- Defining the condition for 818 being non-zero
def eight_hundred_eighteen_nonzero : Prop := 818 ≠ 0

-- Theorem statement
theorem zero_exponent_rule_proof (h : eight_hundred_eighteen_nonzero) : 818 ^ 0 = 1 := by
  sorry

end NUMINAMATH_GPT_zero_exponent_rule_proof_l1813_181355


namespace NUMINAMATH_GPT_mirka_number_l1813_181348

noncomputable def original_number (a b : ℕ) : ℕ := 10 * a + b
noncomputable def reversed_number (a b : ℕ) : ℕ := 10 * b + a

theorem mirka_number (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 4) (h2 : b = 2 * a) :
  original_number a b = 12 ∨ original_number a b = 24 ∨ original_number a b = 36 ∨ original_number a b = 48 :=
by
  sorry

end NUMINAMATH_GPT_mirka_number_l1813_181348


namespace NUMINAMATH_GPT_triangle_region_areas_l1813_181380

open Real

theorem triangle_region_areas (A B C : ℝ) 
  (h1 : 20^2 + 21^2 = 29^2)
  (h2 : ∃ (triangle_area : ℝ), triangle_area = 210)
  (h3 : C > A)
  (h4 : C > B)
  : A + B + 210 = C := 
sorry

end NUMINAMATH_GPT_triangle_region_areas_l1813_181380


namespace NUMINAMATH_GPT_ratio_smaller_to_larger_dimension_of_framed_painting_l1813_181318

-- Definitions
def painting_width : ℕ := 16
def painting_height : ℕ := 20
def side_frame_width (x : ℝ) : ℝ := x
def top_frame_width (x : ℝ) : ℝ := 1.5 * x
def total_frame_area (x : ℝ) : ℝ := (painting_width + 2 * side_frame_width x) * (painting_height + 2 * top_frame_width x) - painting_width * painting_height
def frame_area_eq_painting_area (x : ℝ) : Prop := total_frame_area x = painting_width * painting_height

-- Lean statement
theorem ratio_smaller_to_larger_dimension_of_framed_painting :
  ∃ x : ℝ, frame_area_eq_painting_area x → 
  ((painting_width + 2 * side_frame_width x) / (painting_height + 2 * top_frame_width x)) = (3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_ratio_smaller_to_larger_dimension_of_framed_painting_l1813_181318


namespace NUMINAMATH_GPT_dollar_triple_60_l1813_181303

-- Define the function $N
def dollar (N : Real) : Real :=
  0.4 * N + 2

-- Proposition proving that $$(($60)) = 6.96
theorem dollar_triple_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end NUMINAMATH_GPT_dollar_triple_60_l1813_181303


namespace NUMINAMATH_GPT_expression_value_l1813_181365

theorem expression_value (x y : ℝ) (h : x - y = 1) : 
  x^4 - x * y^3 - x^3 * y - 3 * x^2 * y + 3 * x * y^2 + y^4 = 1 := 
by
  sorry

end NUMINAMATH_GPT_expression_value_l1813_181365


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l1813_181336

noncomputable def condition_to_bool (a b : ℝ) : Bool :=
a > b ∧ b > 0

theorem sufficient_but_not_necessary (a b : ℝ) (h : condition_to_bool a b) :
  (a > b ∧ b > 0) → (a^2 > b^2) ∧ (∃ a' b' : ℝ, a'^2 > b'^2 ∧ ¬ (a' > b' ∧ b' > 0)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l1813_181336


namespace NUMINAMATH_GPT_repeating_decimal_subtraction_simplified_l1813_181363

theorem repeating_decimal_subtraction_simplified :
  let x := (567 / 999 : ℚ)
  let y := (234 / 999 : ℚ)
  let z := (891 / 999 : ℚ)
  x - y - z = -186 / 333 :=
by
  sorry

end NUMINAMATH_GPT_repeating_decimal_subtraction_simplified_l1813_181363


namespace NUMINAMATH_GPT_area_union_of_rectangle_and_circle_l1813_181344

theorem area_union_of_rectangle_and_circle :
  let length := 12
  let width := 15
  let r := 15
  let area_rectangle := length * width
  let area_circle := Real.pi * r^2
  let area_overlap := (1/4) * area_circle
  let area_union := area_rectangle + area_circle - area_overlap
  area_union = 180 + 168.75 * Real.pi := by
    sorry

end NUMINAMATH_GPT_area_union_of_rectangle_and_circle_l1813_181344


namespace NUMINAMATH_GPT_robot_steps_difference_zero_l1813_181300

/-- Define the robot's position at second n --/
def robot_position (n : ℕ) : ℤ :=
  let cycle_length := 7
  let cycle_steps := 4 - 3
  let full_cycles := n / cycle_length
  let remainder := n % cycle_length
  full_cycles + if remainder = 0 then 0 else
    if remainder ≤ 4 then remainder else 4 - (remainder - 4)

/-- The main theorem to prove x_2007 - x_2011 = 0 --/
theorem robot_steps_difference_zero : 
  robot_position 2007 - robot_position 2011 = 0 :=
by sorry

end NUMINAMATH_GPT_robot_steps_difference_zero_l1813_181300


namespace NUMINAMATH_GPT_find_a_plus_b_l1813_181325

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * Real.log x

theorem find_a_plus_b (a b : ℝ) :
  (∃ x : ℝ, x = 1 ∧ f a b x = 1 / 2 ∧ (deriv (f a b)) 1 = 0) →
  a + b = -1/2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1813_181325


namespace NUMINAMATH_GPT_minimize_product_of_roots_of_quadratic_eq_l1813_181374

theorem minimize_product_of_roots_of_quadratic_eq (k : ℝ) :
  (∃ x y : ℝ, 2 * x^2 + 5 * x + k = 0 ∧ 2 * y^2 + 5 * y + k = 0) 
  → k = 25 / 8 :=
sorry

end NUMINAMATH_GPT_minimize_product_of_roots_of_quadratic_eq_l1813_181374


namespace NUMINAMATH_GPT_lcm_inequality_l1813_181381

open Nat

-- Assume positive integers n and m, with n > m
theorem lcm_inequality (n m : ℕ) (h1 : 0 < n) (h2 : 0 < m) (h3 : n > m) :
  Nat.lcm m n + Nat.lcm (m+1) (n+1) ≥ 2 * m * Real.sqrt n := 
  sorry

end NUMINAMATH_GPT_lcm_inequality_l1813_181381


namespace NUMINAMATH_GPT_remainder_when_divided_by_6_l1813_181358

theorem remainder_when_divided_by_6 (n : ℤ) (h_pos : 0 < n) (h_mod12 : n % 12 = 8) : n % 6 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_6_l1813_181358


namespace NUMINAMATH_GPT_cara_total_bread_l1813_181306

theorem cara_total_bread 
  (d : ℕ) (L : ℕ) (B : ℕ) (S : ℕ) 
  (h_dinner : d = 240) 
  (h_lunch : d = 8 * L) 
  (h_breakfast : d = 6 * B) 
  (h_snack : d = 4 * S) : 
  d + L + B + S = 370 := 
sorry

end NUMINAMATH_GPT_cara_total_bread_l1813_181306


namespace NUMINAMATH_GPT_evaluate_expression_when_c_is_4_l1813_181324

variable (c : ℕ)

theorem evaluate_expression_when_c_is_4 : (c = 4) → ((c^2 - c! * (c - 1)^c)^2 = 3715584) :=
by
  -- This is where the proof would go, but we only need to set up the statement.
  sorry

end NUMINAMATH_GPT_evaluate_expression_when_c_is_4_l1813_181324


namespace NUMINAMATH_GPT_find_n_from_equation_l1813_181338

theorem find_n_from_equation : ∃ n : ℤ, n + (n + 1) + (n + 2) + (n + 3) = 22 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_n_from_equation_l1813_181338


namespace NUMINAMATH_GPT_make_polynomial_perfect_square_l1813_181378

theorem make_polynomial_perfect_square (m : ℝ) :
  m = 196 → ∃ (f : ℝ → ℝ), ∀ x : ℝ, (x - 1) * (x + 3) * (x - 4) * (x - 8) + m = (f x) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_make_polynomial_perfect_square_l1813_181378


namespace NUMINAMATH_GPT_find_m_l1813_181368

theorem find_m (m x_1 x_2 : ℝ) 
  (h1 : x_1^2 + m * x_1 - 3 = 0) 
  (h2 : x_2^2 + m * x_2 - 3 = 0) 
  (h3 : x_1 + x_2 - x_1 * x_2 = 5) : 
  m = -2 :=
sorry

end NUMINAMATH_GPT_find_m_l1813_181368


namespace NUMINAMATH_GPT_building_time_l1813_181362

theorem building_time (b p : ℕ) 
  (h1 : b = 3 * p - 5) 
  (h2 : b + p = 67) 
  : b = 49 := 
by 
  sorry

end NUMINAMATH_GPT_building_time_l1813_181362


namespace NUMINAMATH_GPT_point_in_second_quadrant_l1813_181323

theorem point_in_second_quadrant (m : ℝ) (h1 : 3 - m < 0) (h2 : m - 1 > 0) : m > 3 :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l1813_181323


namespace NUMINAMATH_GPT_strawberry_picking_l1813_181390

theorem strawberry_picking 
  (e : ℕ) (n : ℕ) (p : ℕ) (A : ℕ) (w : ℕ) 
  (h1 : e = 4) 
  (h2 : n = 3) 
  (h3 : p = 20) 
  (h4 : A = 128) 
  : w = 7 :=
by 
  -- proof steps to be filled in
  sorry

end NUMINAMATH_GPT_strawberry_picking_l1813_181390


namespace NUMINAMATH_GPT_inequality_proof_l1813_181397

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 0) :
  (33 * a^2 - a) / (33 * a^2 + 1) + (33 * b^2 - b) / (33 * b^2 + 1) + (33 * c^2 - c) / (33 * c^2 + 1) ≥ 0 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1813_181397


namespace NUMINAMATH_GPT_rationalize_denominator_l1813_181329

theorem rationalize_denominator : (3 : ℝ) / Real.sqrt 75 = (Real.sqrt 3) / 5 :=
by
  sorry

end NUMINAMATH_GPT_rationalize_denominator_l1813_181329


namespace NUMINAMATH_GPT_ratio_of_x_to_y_l1813_181377

theorem ratio_of_x_to_y (x y : ℤ) (h : (12 * x - 5 * y) / (17 * x - 3 * y) = 5 / 7) : x / y = -20 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_x_to_y_l1813_181377


namespace NUMINAMATH_GPT_negation_exists_ltx2_plus_x_plus_1_lt_0_l1813_181349

theorem negation_exists_ltx2_plus_x_plus_1_lt_0 :
  ¬ (∃ x : ℝ, x^2 + x + 1 < 0) ↔ ∀ x : ℝ, x^2 + x + 1 ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_ltx2_plus_x_plus_1_lt_0_l1813_181349


namespace NUMINAMATH_GPT_min_platforms_needed_l1813_181305

theorem min_platforms_needed :
  let slabs_7_tons := 120
  let slabs_9_tons := 80
  let weight_7_tons := 7
  let weight_9_tons := 9
  let max_weight_per_platform := 40
  let total_weight := slabs_7_tons * weight_7_tons + slabs_9_tons * weight_9_tons
  let platforms_needed_per_7_tons := slabs_7_tons / 3
  let platforms_needed_per_9_tons := slabs_9_tons / 2
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 ∧ 3 * platforms_needed_per_7_tons = slabs_7_tons ∧ 2 * platforms_needed_per_9_tons = slabs_9_tons →
  platforms_needed_per_7_tons = 40 ∧ platforms_needed_per_9_tons = 40 :=
by
  sorry

end NUMINAMATH_GPT_min_platforms_needed_l1813_181305


namespace NUMINAMATH_GPT_impossible_load_two_coins_l1813_181366

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end NUMINAMATH_GPT_impossible_load_two_coins_l1813_181366


namespace NUMINAMATH_GPT_hyperbola_eccentricity_is_2_l1813_181332

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) : ℝ :=
c / a

theorem hyperbola_eccentricity_is_2 (a b c : ℝ)
  (h : a > 0 ∧ b > 0 ∧ c > 0)
  (H1 : b^2 = c^2 - a^2)
  (H2 : 3 * c^2 = 4 * b^2) :
  hyperbola_eccentricity a b c h H1 H2 = 2 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_is_2_l1813_181332


namespace NUMINAMATH_GPT_c_completion_days_l1813_181399

noncomputable def work_rate (days: ℕ) := (1 : ℝ) / days

theorem c_completion_days : 
  ∀ (W : ℝ) (Ra Rb Rc : ℝ) (Dc : ℕ),
  Ra = work_rate 30 → Rb = work_rate 30 → Rc = work_rate Dc →
  (Ra + Rb + Rc) * 8 + (Ra + Rb) * 4 = W → 
  Dc = 40 :=
by
  intros W Ra Rb Rc Dc hRa hRb hRc hW
  sorry

end NUMINAMATH_GPT_c_completion_days_l1813_181399


namespace NUMINAMATH_GPT_find_a_plus_2b_l1813_181391

open Real

theorem find_a_plus_2b 
  (a b : ℝ) 
  (ha : 0 < a ∧ a < π / 2) 
  (hb : 0 < b ∧ b < π / 2) 
  (h1 : 4 * (sin a)^2 + 3 * (sin b)^2 = 1) 
  (h2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0) :
  a + 2 * b = π / 2 :=
sorry

end NUMINAMATH_GPT_find_a_plus_2b_l1813_181391


namespace NUMINAMATH_GPT_teacher_already_graded_worksheets_l1813_181389

-- Define the conditions
def num_worksheets : ℕ := 9
def problems_per_worksheet : ℕ := 4
def remaining_problems : ℕ := 16
def total_problems := num_worksheets * problems_per_worksheet

-- Define the required proof
theorem teacher_already_graded_worksheets :
  (total_problems - remaining_problems) / problems_per_worksheet = 5 :=
by sorry

end NUMINAMATH_GPT_teacher_already_graded_worksheets_l1813_181389


namespace NUMINAMATH_GPT_square_side_length_l1813_181384

theorem square_side_length :
  ∀ (s : ℝ), (∃ w l : ℝ, w = 6 ∧ l = 24 ∧ s^2 = w * l) → s = 12 := by 
  sorry

end NUMINAMATH_GPT_square_side_length_l1813_181384


namespace NUMINAMATH_GPT_range_of_a_l1813_181304

def A (a : ℝ) : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ x^2 + a * x - y + 2 = 0}
def B : Set (ℝ × ℝ) := {p | ∃ (x y : ℝ), p = (x, y) ∧ 2 * x - y + 1 = 0 ∧ x > 0}

theorem range_of_a (a : ℝ) : (∃ p, p ∈ A a ∧ p ∈ B) ↔ a ∈ Set.Iic 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1813_181304


namespace NUMINAMATH_GPT_find_angle_between_vectors_l1813_181331

noncomputable def angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : ℝ :=
  60

theorem find_angle_between_vectors 
  (a b : ℝ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) 
  (perp1 : (a + 3*b) * (7*a - 5*b) = 0) 
  (perp2 : (a - 4*b) * (7*a - 2*b) = 0) : angle_between_vectors a b a_nonzero b_nonzero perp1 perp2 = 60 :=
  by 
  sorry

end NUMINAMATH_GPT_find_angle_between_vectors_l1813_181331


namespace NUMINAMATH_GPT_fair_hair_percentage_l1813_181301

-- Define the main entities
variables (E F W : ℝ)

-- Define the conditions given in the problem
def women_with_fair_hair : Prop := W = 0.32 * E
def fair_hair_women_ratio : Prop := W = 0.40 * F

-- Define the theorem to prove
theorem fair_hair_percentage
  (hwf: women_with_fair_hair E W)
  (fhr: fair_hair_women_ratio W F) :
  (F / E) * 100 = 80 :=
by
  sorry

end NUMINAMATH_GPT_fair_hair_percentage_l1813_181301


namespace NUMINAMATH_GPT_joan_final_oranges_l1813_181398

def joan_oranges_initial := 75
def tom_oranges := 42
def sara_sold := 40
def christine_added := 15

theorem joan_final_oranges : joan_oranges_initial + tom_oranges - sara_sold + christine_added = 92 :=
by 
  sorry

end NUMINAMATH_GPT_joan_final_oranges_l1813_181398


namespace NUMINAMATH_GPT_solve_for_x_minus_y_l1813_181345

theorem solve_for_x_minus_y (x y : ℝ) (h1 : 4 = 0.25 * x) (h2 : 4 = 0.50 * y) : x - y = 8 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_minus_y_l1813_181345


namespace NUMINAMATH_GPT_general_term_formula_l1813_181333

def Sn (a_n : ℕ → ℕ) (n : ℕ) : ℕ := 2 * a_n n - 2^(n + 1)

theorem general_term_formula (a_n : ℕ → ℕ) (h : ∀ n : ℕ, n > 0 → Sn a_n n = (2 * a_n n - 2^(n + 1))) :
  ∀ n : ℕ, n > 0 → a_n n = (n + 1) * 2^n :=
sorry

end NUMINAMATH_GPT_general_term_formula_l1813_181333


namespace NUMINAMATH_GPT_gnuff_tutoring_rate_l1813_181382

theorem gnuff_tutoring_rate (flat_rate : ℕ) (total_paid : ℕ) (minutes : ℕ) :
  flat_rate = 20 → total_paid = 146 → minutes = 18 → (total_paid - flat_rate) / minutes = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_gnuff_tutoring_rate_l1813_181382


namespace NUMINAMATH_GPT_club_members_problem_l1813_181313

theorem club_members_problem 
    (T : ℕ) (C : ℕ) (D : ℕ) (B : ℕ) 
    (h_T : T = 85) (h_C : C = 45) (h_D : D = 32) (h_B : B = 18) :
    let Cₒ := C - B
    let Dₒ := D - B
    let N := T - (Cₒ + Dₒ + B)
    N = 26 :=
by
  sorry

end NUMINAMATH_GPT_club_members_problem_l1813_181313


namespace NUMINAMATH_GPT_red_ball_count_l1813_181395

theorem red_ball_count (w : ℕ) (f : ℝ) (total : ℕ) (r : ℕ) 
  (hw : w = 60)
  (hf : f = 0.25)
  (ht : total = w / (1 - f))
  (hr : r = total * f) : 
  r = 20 :=
by 
  -- Lean doesn't require a proof for the problem statement
  sorry

end NUMINAMATH_GPT_red_ball_count_l1813_181395


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l1813_181341

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 4 * x^2) : (0, 1/16) = (0, 1/16) :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l1813_181341


namespace NUMINAMATH_GPT_gcd_expression_l1813_181346

noncomputable def odd_multiple_of_7771 (a : ℕ) : Prop := 
  ∃ k : ℕ, k % 2 = 1 ∧ a = 7771 * k

theorem gcd_expression (a : ℕ) (h : odd_multiple_of_7771 a) : 
  Int.gcd (8 * a^2 + 57 * a + 132) (2 * a + 9) = 9 :=
  sorry

end NUMINAMATH_GPT_gcd_expression_l1813_181346


namespace NUMINAMATH_GPT_correct_operation_l1813_181351

theorem correct_operation (x : ℝ) : (x^2) * (x^4) = x^6 :=
  sorry

end NUMINAMATH_GPT_correct_operation_l1813_181351
