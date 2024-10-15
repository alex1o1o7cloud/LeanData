import Mathlib

namespace NUMINAMATH_GPT_integral_cos_neg_one_l2314_231417

theorem integral_cos_neg_one: 
  ∫ x in (Set.Icc (Real.pi / 2) Real.pi), Real.cos x = -1 :=
by
  sorry

end NUMINAMATH_GPT_integral_cos_neg_one_l2314_231417


namespace NUMINAMATH_GPT_neon_signs_blink_together_l2314_231445

theorem neon_signs_blink_together :
  Nat.lcm (Nat.lcm (Nat.lcm 7 11) 13) 17 = 17017 :=
by
  sorry

end NUMINAMATH_GPT_neon_signs_blink_together_l2314_231445


namespace NUMINAMATH_GPT_triangle_lines_l2314_231496

/-- Given a triangle with vertices A(1, 2), B(-1, 4), and C(4, 5):
  1. The equation of the line l₁ containing the altitude from A to side BC is 5x + y - 7 = 0.
  2. The equation of the line l₂ passing through C such that the distances from A and B to l₂ are equal
     is either x + y - 9 = 0 or x - 2y + 6 = 0. -/
theorem triangle_lines (A B C : ℝ × ℝ)
  (hA : A = (1, 2))
  (hB : B = (-1, 4))
  (hC : C = (4, 5)) :
  ∃ l₁ l₂ : ℝ × ℝ × ℝ,
  (l₁ = (5, 1, -7)) ∧
  ((l₂ = (1, 1, -9)) ∨ (l₂ = (1, -2, 6))) := by
  sorry

end NUMINAMATH_GPT_triangle_lines_l2314_231496


namespace NUMINAMATH_GPT_max_value_expr_l2314_231414

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end NUMINAMATH_GPT_max_value_expr_l2314_231414


namespace NUMINAMATH_GPT_bridge_length_l2314_231488

noncomputable def train_length : ℝ := 250 -- in meters
noncomputable def train_speed_kmh : ℝ := 60 -- in km/hr
noncomputable def crossing_time : ℝ := 20 -- in seconds

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600 -- converting to m/s

noncomputable def total_distance_covered : ℝ := train_speed_ms * crossing_time -- distance covered in 20 seconds

theorem bridge_length : total_distance_covered - train_length = 83.4 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_bridge_length_l2314_231488


namespace NUMINAMATH_GPT_puppies_given_to_friends_l2314_231418

def original_puppies : ℕ := 8
def current_puppies : ℕ := 4

theorem puppies_given_to_friends : original_puppies - current_puppies = 4 :=
by
  sorry

end NUMINAMATH_GPT_puppies_given_to_friends_l2314_231418


namespace NUMINAMATH_GPT_find_g3_l2314_231485

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition : ∀ x : ℝ, g (3^x) + x * g (3^(-x)) = 1

theorem find_g3 : g 3 = 0 := by
  sorry

end NUMINAMATH_GPT_find_g3_l2314_231485


namespace NUMINAMATH_GPT_dividend_calculation_l2314_231451

theorem dividend_calculation (D : ℝ) (Q : ℝ) (R : ℝ) (Dividend : ℝ) (h1 : D = 47.5) (h2 : Q = 24.3) (h3 : R = 32.4)  :
  Dividend = D * Q + R := by
  rw [h1, h2, h3]
  sorry -- This skips the actual computation proof

end NUMINAMATH_GPT_dividend_calculation_l2314_231451


namespace NUMINAMATH_GPT_find_c_of_parabola_l2314_231415

theorem find_c_of_parabola (a b c : ℚ) (h_vertex : (5 : ℚ) = a * (3 : ℚ)^2 + b * (3 : ℚ) + c)
    (h_point : (7 : ℚ) = a * (1 : ℚ)^2 + b * (1 : ℚ) + c) :
  c = 19 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_c_of_parabola_l2314_231415


namespace NUMINAMATH_GPT_minimum_cost_peking_opera_l2314_231441

theorem minimum_cost_peking_opera (T p₆ p₁₀ : ℕ) (xₛ yₛ : ℕ) :
  T = 140 ∧ p₆ = 6 ∧ p₁₀ = 10 ∧ xₛ + yₛ = T ∧ yₛ ≥ 2 * xₛ →
  6 * xₛ + 10 * yₛ = 1216 ∧ xₛ = 46 ∧ yₛ = 94 :=
by
   -- Proving this is skipped (left as a sorry)
  sorry

end NUMINAMATH_GPT_minimum_cost_peking_opera_l2314_231441


namespace NUMINAMATH_GPT_alyona_final_balances_l2314_231498

noncomputable def finalBalances (initialEuros initialDollars initialRubles : ℕ)
                                (interestRateEuroDollar interestRateRuble : ℚ)
                                (conversionRateEuroToRubles1 conversionRateRublesToDollars1 : ℚ)
                                (conversionRateDollarsToRubles2 conversionRateRublesToEuros2 : ℚ) :
                                ℕ × ℕ × ℕ :=
  sorry

theorem alyona_final_balances :
  finalBalances 3000 4000 240000
                (2.1 / 100) (7.9 / 100)
                60.10 58.90
                58.50 63.20 = (4040, 3286, 301504) :=
  sorry

end NUMINAMATH_GPT_alyona_final_balances_l2314_231498


namespace NUMINAMATH_GPT_sequence_arithmetic_l2314_231481

variable (a b : ℕ → ℤ)

theorem sequence_arithmetic :
  a 0 = 3 →
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →
  b 3 = -2 →
  b 10 = 12 →
  a 8 = 3 :=
by
  intros h1 ha hb3 hb10
  sorry

end NUMINAMATH_GPT_sequence_arithmetic_l2314_231481


namespace NUMINAMATH_GPT_student_solved_18_correctly_l2314_231490

theorem student_solved_18_correctly (total_problems : ℕ) (correct : ℕ) (wrong : ℕ) 
  (h1 : total_problems = 54) (h2 : wrong = 2 * correct) (h3 : total_problems = correct + wrong) :
  correct = 18 :=
by
  sorry

end NUMINAMATH_GPT_student_solved_18_correctly_l2314_231490


namespace NUMINAMATH_GPT_upper_limit_of_people_l2314_231420

theorem upper_limit_of_people (P : ℕ) (h1 : 36 = (3 / 8) * P) (h2 : P > 50) (h3 : (5 / 12) * P = 40) : P = 96 :=
by
  sorry

end NUMINAMATH_GPT_upper_limit_of_people_l2314_231420


namespace NUMINAMATH_GPT_find_t_l2314_231436

theorem find_t : ∃ t, ∀ (x y : ℝ), (x, y) = (0, 1) ∨ (x, y) = (-6, -3) → (t, 7) ∈ {p : ℝ × ℝ | ∃ m b, p.2 = m * p.1 + b ∧ ((0, 1) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) ∧ ((-6, -3) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) } → t = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_t_l2314_231436


namespace NUMINAMATH_GPT_triangle_angles_and_type_l2314_231411

theorem triangle_angles_and_type
  (largest_angle : ℝ)
  (smallest_angle : ℝ)
  (middle_angle : ℝ)
  (h1 : largest_angle = 90)
  (h2 : largest_angle = 3 * smallest_angle)
  (h3 : largest_angle + smallest_angle + middle_angle = 180) :
  (largest_angle = 90 ∧ middle_angle = 60 ∧ smallest_angle = 30 ∧ largest_angle = 90) := by
  sorry

end NUMINAMATH_GPT_triangle_angles_and_type_l2314_231411


namespace NUMINAMATH_GPT_real_root_in_interval_l2314_231492

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 3

theorem real_root_in_interval : ∃ α : ℝ, f α = 0 ∧ 1 < α ∧ α < 2 :=
sorry

end NUMINAMATH_GPT_real_root_in_interval_l2314_231492


namespace NUMINAMATH_GPT_impossible_event_abs_lt_zero_l2314_231465

theorem impossible_event_abs_lt_zero (a : ℝ) : ¬ (|a| < 0) :=
sorry

end NUMINAMATH_GPT_impossible_event_abs_lt_zero_l2314_231465


namespace NUMINAMATH_GPT_solve_for_xy_l2314_231477

theorem solve_for_xy (x y : ℝ) (h : 2 * x - 3 ≤ Real.log (x + y + 1) + Real.log (x - y - 2)) : x * y = -9 / 4 :=
by sorry

end NUMINAMATH_GPT_solve_for_xy_l2314_231477


namespace NUMINAMATH_GPT_range_of_a_for_quadratic_inequality_l2314_231460

theorem range_of_a_for_quadratic_inequality :
  ∀ a : ℝ, (∀ (x : ℝ), 1 ≤ x ∧ x < 5 → x^2 - (a + 1)*x + a ≤ 0) ↔ (4 ≤ a ∧ a < 5) :=
sorry

end NUMINAMATH_GPT_range_of_a_for_quadratic_inequality_l2314_231460


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2314_231423

noncomputable def a_n (n : ℕ) : ℕ :=
  n

def b_n (n : ℕ) : ℕ :=
  n * 2^n

def S_n (n : ℕ) : ℕ :=
  (n - 1) * 2^(n + 1) + 2

theorem arithmetic_sequence_sum
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (h1 : a_n 1 + a_n 2 + a_n 3 = 6)
  (h2 : a_n 5 = 5)
  (h3 : ∀ n, b_n n = a_n n * 2^(a_n n)) :
  (∀ n, a_n n = n) ∧ (∀ n, S_n n = (n - 1) * 2^(n + 1) + 2) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2314_231423


namespace NUMINAMATH_GPT_flour_more_than_salt_l2314_231428

open Function

-- Definitions based on conditions
def flour_needed : ℕ := 12
def flour_added : ℕ := 2
def salt_needed : ℕ := 7
def salt_added : ℕ := 0

-- Given that these definitions hold, prove the following theorem
theorem flour_more_than_salt : (flour_needed - flour_added) - (salt_needed - salt_added) = 3 :=
by
  -- Here you would include the proof, but as instructed, we skip it with "sorry".
  sorry

end NUMINAMATH_GPT_flour_more_than_salt_l2314_231428


namespace NUMINAMATH_GPT_proof_problem_l2314_231464

noncomputable def sqrt_repeated (x : ℕ) (y : ℕ) : ℕ :=
Nat.sqrt x ^ y

theorem proof_problem (x y z : ℕ) :
  (sqrt_repeated x y = z) ↔ 
  ((∃ t : ℕ, x = t^2 ∧ y = 1 ∧ z = t) ∨ (x = 0 ∧ z = 0 ∧ y ≠ 0)) :=
sorry

end NUMINAMATH_GPT_proof_problem_l2314_231464


namespace NUMINAMATH_GPT_color_blocks_probability_at_least_one_box_match_l2314_231454

/-- Given Ang, Ben, and Jasmin each having 6 blocks of different colors (red, blue, yellow, white, green, and orange) 
    and they independently place one of their blocks into each of 6 empty boxes, 
    the proof shows that the probability that at least one box receives 3 blocks all of the same color is 1/6. 
    Since 1/6 is equal to the fraction m/n where m=1 and n=6 are relatively prime, thus m+n=7. -/
theorem color_blocks_probability_at_least_one_box_match (p : ℕ × ℕ) (h : p = (1, 6)) : p.1 + p.2 = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_color_blocks_probability_at_least_one_box_match_l2314_231454


namespace NUMINAMATH_GPT_solve_system_of_equations_l2314_231432

theorem solve_system_of_equations :
  ∃ x y z : ℚ, 
    (y * z = 3 * y + 2 * z - 8) ∧
    (z * x = 4 * z + 3 * x - 8) ∧
    (x * y = 2 * x + y - 1) ∧
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ 
     (x = 3 ∧ y = 5 / 2 ∧ z = -1)) := 
by
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l2314_231432


namespace NUMINAMATH_GPT_marcella_shoes_lost_l2314_231448

theorem marcella_shoes_lost (pairs_initial : ℕ) (pairs_left_max : ℕ) (individuals_initial : ℕ) (individuals_left_max : ℕ) (pairs_initial_eq : pairs_initial = 25) (pairs_left_max_eq : pairs_left_max = 20) (individuals_initial_eq : individuals_initial = pairs_initial * 2) (individuals_left_max_eq : individuals_left_max = pairs_left_max * 2) : (individuals_initial - individuals_left_max) = 10 := 
by
  sorry

end NUMINAMATH_GPT_marcella_shoes_lost_l2314_231448


namespace NUMINAMATH_GPT_captain_and_vicecaptain_pair_boys_and_girls_l2314_231472

-- Problem A
theorem captain_and_vicecaptain (n : ℕ) (h : n = 11) : ∃ ways : ℕ, ways = 110 :=
by
  sorry

-- Problem B
theorem pair_boys_and_girls (N : ℕ) : ∃ ways : ℕ, ways = Nat.factorial N :=
by
  sorry

end NUMINAMATH_GPT_captain_and_vicecaptain_pair_boys_and_girls_l2314_231472


namespace NUMINAMATH_GPT_express_x_in_terms_of_y_l2314_231434

variable {x y : ℝ}

theorem express_x_in_terms_of_y (h : 3 * x - 4 * y = 6) : x = (6 + 4 * y) / 3 := 
sorry

end NUMINAMATH_GPT_express_x_in_terms_of_y_l2314_231434


namespace NUMINAMATH_GPT_profit_percentage_is_30_percent_l2314_231402

theorem profit_percentage_is_30_percent (CP SP : ℕ) (h1 : CP = 280) (h2 : SP = 364) :
  ((SP - CP : ℤ) / (CP : ℤ) : ℚ) * 100 = 30 :=
by sorry

end NUMINAMATH_GPT_profit_percentage_is_30_percent_l2314_231402


namespace NUMINAMATH_GPT_min_rings_to_connect_all_segments_l2314_231413

-- Define the problem setup
structure ChainSegment where
  rings : Fin 3 → Type

-- Define the number of segments
def num_segments : ℕ := 5

-- Define the minimum number of rings to be opened and rejoined
def min_rings_to_connect (seg : Fin num_segments) : ℕ :=
  3

theorem min_rings_to_connect_all_segments :
  ∀ segs : Fin num_segments,
  (∃ n, n = min_rings_to_connect segs) :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_min_rings_to_connect_all_segments_l2314_231413


namespace NUMINAMATH_GPT_maximal_partition_sets_l2314_231475

theorem maximal_partition_sets : 
  ∃(n : ℕ), (∀(a : ℕ), a * n = 16657706 → (a = 5771 ∧ n = 2886)) := 
by
  sorry

end NUMINAMATH_GPT_maximal_partition_sets_l2314_231475


namespace NUMINAMATH_GPT_total_ticket_cost_l2314_231430

theorem total_ticket_cost (V G : ℕ) 
  (h1 : V + G = 320) 
  (h2 : V = G - 276) 
  (price_vip : ℕ := 45) 
  (price_regular : ℕ := 20) : 
  (price_vip * V + price_regular * G = 6950) :=
by sorry

end NUMINAMATH_GPT_total_ticket_cost_l2314_231430


namespace NUMINAMATH_GPT_pq_sufficient_but_not_necessary_condition_l2314_231495

theorem pq_sufficient_but_not_necessary_condition (p q : Prop) (hpq : p ∧ q) :
  ¬¬p = p :=
by
  sorry

end NUMINAMATH_GPT_pq_sufficient_but_not_necessary_condition_l2314_231495


namespace NUMINAMATH_GPT_birth_date_of_older_friend_l2314_231401

/-- Lean 4 statement for the proof problem --/
theorem birth_date_of_older_friend
  (d m y : ℕ)
  (h1 : y ≥ 1900 ∧ y < 2000)
  (h2 : d + 7 < 32) -- Assuming the month has at most 31 days
  (h3 : ((d+7) * 10^4 + m * 10^2 + y % 100) = 6 * (d * 10^4 + m * 10^2 + y % 100))
  (h4 : m > 0 ∧ m < 13)  -- Months are between 1 and 12
  (h5 : (d * 10^4 + m * 10^2 + y % 100) < (d+7) * 10^4 + m * 10^2 + y % 100) -- d < d+7 so older means smaller number
  : d = 1 ∧ m = 4 ∧ y = 1900 :=
by
  sorry -- Proof omitted

end NUMINAMATH_GPT_birth_date_of_older_friend_l2314_231401


namespace NUMINAMATH_GPT_length_of_other_diagonal_l2314_231466

theorem length_of_other_diagonal (d1 d2 : ℝ) (A : ℝ) (h1 : d1 = 15) (h2 : A = 150) : d2 = 20 :=
by
  sorry

end NUMINAMATH_GPT_length_of_other_diagonal_l2314_231466


namespace NUMINAMATH_GPT_find_third_vertex_l2314_231459

open Real

-- Define the vertices of the triangle
def vertex1 : ℝ × ℝ := (9, 3)
def vertex2 : ℝ × ℝ := (0, 0)

-- Define the conditions
def on_negative_x_axis (p : ℝ × ℝ) : Prop :=
  p.2 = 0 ∧ p.1 < 0

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * abs ((b.1 - a.1) * (c.2 - a.2) - (c.1 - a.1) * (b.2 - a.2))

-- Statement of the problem in Lean
theorem find_third_vertex :
  ∃ (vertex3 : ℝ × ℝ), 
    on_negative_x_axis vertex3 ∧ 
    area_of_triangle vertex1 vertex2 vertex3 = 45 ∧
    vertex3 = (-30, 0) :=
sorry

end NUMINAMATH_GPT_find_third_vertex_l2314_231459


namespace NUMINAMATH_GPT_not_perfect_square_l2314_231435

theorem not_perfect_square (n : ℕ) (h : 0 < n) : ¬ ∃ k : ℕ, k * k = 2551 * 543^n - 2008 * 7^n :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_square_l2314_231435


namespace NUMINAMATH_GPT_tan_gt_neg_one_solution_set_l2314_231437

def tangent_periodic_solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, k * Real.pi - Real.pi / 4 < x ∧ x < k * Real.pi + Real.pi / 2

theorem tan_gt_neg_one_solution_set (x : ℝ) :
  tangent_periodic_solution_set x ↔ Real.tan x > -1 :=
by
  sorry

end NUMINAMATH_GPT_tan_gt_neg_one_solution_set_l2314_231437


namespace NUMINAMATH_GPT_value_range_of_function_l2314_231467

theorem value_range_of_function :
  ∀ (x : ℝ), -1 ≤ Real.sin x ∧ Real.sin x ≤ 1 → -1 ≤ Real.sin x * Real.sin x - 2 * Real.sin x ∧ Real.sin x * Real.sin x - 2 * Real.sin x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_value_range_of_function_l2314_231467


namespace NUMINAMATH_GPT_inequality_transformation_l2314_231440

variable {x y : ℝ}

theorem inequality_transformation (h : x > y) : x + 5 > y + 5 :=
by
  sorry

end NUMINAMATH_GPT_inequality_transformation_l2314_231440


namespace NUMINAMATH_GPT_inequality_proof_l2314_231424

theorem inequality_proof
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (ha1 : 0 < a1) (ha2 : 0 < a2) (ha3 : 0 < a3)
  (hb1 : 0 < b1) (hb2 : 0 < b2) (hb3 : 0 < b3) :
  (a1 * b2 + a2 * b1 + a2 * b3 + a3 * b2 + a3 * b1 + a1 * b3)^2 ≥
    4 * (a1 * a2 + a2 * a3 + a3 * a1) * (b1 * b2 + b2 * b3 + b3 * b1) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l2314_231424


namespace NUMINAMATH_GPT_smallest_four_digit_int_equiv_8_mod_9_l2314_231450

theorem smallest_four_digit_int_equiv_8_mod_9 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 8 ∧ n = 1007 := 
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_int_equiv_8_mod_9_l2314_231450


namespace NUMINAMATH_GPT_maximal_k_value_l2314_231412

noncomputable def max_edges (n : ℕ) : ℕ :=
  2 * n - 4
   
theorem maximal_k_value (k n : ℕ) (h1 : n = 2016) (h2 : k ≤ max_edges n) :
  k = 4028 :=
by sorry

end NUMINAMATH_GPT_maximal_k_value_l2314_231412


namespace NUMINAMATH_GPT_proper_polygons_m_lines_l2314_231421

noncomputable def smallest_m := 2

theorem proper_polygons_m_lines (P : Finset (Set (ℝ × ℝ)))
  (properly_placed : ∀ (p1 p2 : Set (ℝ × ℝ)), p1 ∈ P → p2 ∈ P → ∃ l : Set (ℝ × ℝ), (0, 0) ∈ l ∧ ∀ (p : Set (ℝ × ℝ)), p ∈ P → ¬Disjoint l p) :
  ∃ (m : ℕ), m = smallest_m ∧ ∀ (lines : Finset (Set (ℝ × ℝ))), 
    (∀ l ∈ lines, (0, 0) ∈ l) → lines.card = m → ∀ p ∈ P, ∃ l ∈ lines, ¬Disjoint l p := sorry

end NUMINAMATH_GPT_proper_polygons_m_lines_l2314_231421


namespace NUMINAMATH_GPT_find_price_per_package_l2314_231427

theorem find_price_per_package (P : ℝ) :
  (10 * P + 50 * (4/5 * P) = 1340) → (P = 26.80) := by
  intros h
  sorry

end NUMINAMATH_GPT_find_price_per_package_l2314_231427


namespace NUMINAMATH_GPT_find_n_expansion_l2314_231486

theorem find_n_expansion : 
  (∃ n : ℕ, 4^n + 2^n = 1056) → n = 5 :=
by sorry

end NUMINAMATH_GPT_find_n_expansion_l2314_231486


namespace NUMINAMATH_GPT_FindAngleB_FindIncircleRadius_l2314_231442

-- Define the problem setting
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Condition 1: a + c = 2b * sin (C + π / 6)
def Condition1 (T : Triangle) : Prop :=
  T.a + T.c = 2 * T.b * Real.sin (T.C + Real.pi / 6)

-- Condition 2: (b + c) (sin B - sin C) = (a - c) sin A
def Condition2 (T : Triangle) : Prop :=
  (T.b + T.c) * (Real.sin T.B - Real.sin T.C) = (T.a - T.c) * Real.sin T.A

-- Condition 3: (2a - c) cos B = b cos C
def Condition3 (T : Triangle) : Prop :=
  (2 * T.a - T.c) * Real.cos T.B = T.b * Real.cos T.C

-- Given: radius of incircle and dot product of vectors condition
def Given (T : Triangle) (r : ℝ) : Prop :=
  (T.a + T.c = 4 * Real.sqrt 3) ∧
  (2 * T.b * (T.a * T.c * Real.cos T.B - 3 * Real.sqrt 3 / 2) = 6)

-- Proof of B = π / 3
theorem FindAngleB (T : Triangle) :
  (Condition1 T ∨ Condition2 T ∨ Condition3 T) → T.B = Real.pi / 3 := 
sorry

-- Proof for the radius of the incircle
theorem FindIncircleRadius (T : Triangle) (r : ℝ) :
  Given T r → T.B = Real.pi / 3 → r = 1 := 
sorry


end NUMINAMATH_GPT_FindAngleB_FindIncircleRadius_l2314_231442


namespace NUMINAMATH_GPT_solve_for_x_l2314_231478

theorem solve_for_x (x : ℕ) (h : x + 1 = 4) : x = 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2314_231478


namespace NUMINAMATH_GPT_minimum_red_chips_l2314_231469

variable (w b r : ℕ)

axiom C1 : b ≥ (1 / 3 : ℚ) * w
axiom C2 : b ≤ (1 / 4 : ℚ) * r
axiom C3 : w + b ≥ 75

theorem minimum_red_chips : r = 76 := by sorry

end NUMINAMATH_GPT_minimum_red_chips_l2314_231469


namespace NUMINAMATH_GPT_no_solution_to_system_l2314_231408

theorem no_solution_to_system : ∀ (x y : ℝ), ¬ (y^2 - (⌊x⌋ : ℝ)^2 = 2001 ∧ x^2 + (⌊y⌋ : ℝ)^2 = 2001) :=
by sorry

end NUMINAMATH_GPT_no_solution_to_system_l2314_231408


namespace NUMINAMATH_GPT_ads_not_blocked_not_interesting_l2314_231410

theorem ads_not_blocked_not_interesting:
  (let A_blocks := 0.75
   let B_blocks := 0.85
   let C_blocks := 0.95
   let A_let_through := 1 - A_blocks
   let B_let_through := 1 - B_blocks
   let C_let_through := 1 - C_blocks
   let all_let_through := A_let_through * B_let_through * C_let_through
   let interesting := 0.15
   let not_interesting := 1 - interesting
   (all_let_through * not_interesting) = 0.00159375) :=
  sorry

end NUMINAMATH_GPT_ads_not_blocked_not_interesting_l2314_231410


namespace NUMINAMATH_GPT_Allyson_age_is_28_l2314_231422

-- Define the conditions of the problem
def Hiram_age : ℕ := 40
def add_12_to_Hiram_age (h_age : ℕ) : ℕ := h_age + 12
def twice_Allyson_age (a_age : ℕ) : ℕ := 2 * a_age
def condition (h_age : ℕ) (a_age : ℕ) : Prop := add_12_to_Hiram_age h_age = twice_Allyson_age a_age - 4

-- Define the theorem to be proven
theorem Allyson_age_is_28 (a_age : ℕ) (h_age : ℕ) (h_condition : condition h_age a_age) (h_hiram : h_age = Hiram_age) : a_age = 28 :=
by sorry

end NUMINAMATH_GPT_Allyson_age_is_28_l2314_231422


namespace NUMINAMATH_GPT_remainder_calculation_l2314_231444

theorem remainder_calculation 
  (x : ℤ) (y : ℝ)
  (hx : 0 < x)
  (hy : y = 70.00000000000398)
  (hx_div_y : (x : ℝ) / y = 86.1) :
  x % y = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_calculation_l2314_231444


namespace NUMINAMATH_GPT_sin_arithmetic_sequence_l2314_231455

noncomputable def sin_value (a : ℝ) := Real.sin (a * (Real.pi / 180))

theorem sin_arithmetic_sequence (a : ℝ) : 
  (0 < a) ∧ (a < 360) ∧ (sin_value a + sin_value (3 * a) = 2 * sin_value (2 * a)) ↔ a = 90 ∨ a = 270 :=
by 
  sorry

end NUMINAMATH_GPT_sin_arithmetic_sequence_l2314_231455


namespace NUMINAMATH_GPT_additional_fee_per_minute_for_second_plan_l2314_231438

theorem additional_fee_per_minute_for_second_plan :
  (∃ x : ℝ, (22 + 0.13 * 280 = 8 + x * 280) ∧ x = 0.18) :=
sorry

end NUMINAMATH_GPT_additional_fee_per_minute_for_second_plan_l2314_231438


namespace NUMINAMATH_GPT_tan_alpha_plus_pi_l2314_231409

-- Define the given conditions and prove the desired equality.
theorem tan_alpha_plus_pi 
  (α : ℝ) 
  (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (π - α) = 1 / 3) : 
  Real.tan (α + π) = -2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_alpha_plus_pi_l2314_231409


namespace NUMINAMATH_GPT_problem_l2314_231499

open Function

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + m) + a (n - m) = 2 * a n

theorem problem (h_arith : arithmetic_sequence a) (h_eq : a 1 + 3 * a 6 + a 11 = 10) :
  a 5 + a 7 = 4 := 
sorry

end NUMINAMATH_GPT_problem_l2314_231499


namespace NUMINAMATH_GPT_distance_from_C_to_B_is_80_l2314_231426

theorem distance_from_C_to_B_is_80
  (x : ℕ)
  (h1 : x = 60)
  (h2 : ∀ (ab cb : ℕ), ab = x → cb = x + 20  → (cb = 80))
  : x + 20 = 80 := by
  sorry

end NUMINAMATH_GPT_distance_from_C_to_B_is_80_l2314_231426


namespace NUMINAMATH_GPT_rotated_triangle_forms_two_cones_l2314_231400

/-- Prove that the spatial geometric body formed when a right-angled triangle 
is rotated 360° around its hypotenuse is two cones. -/
theorem rotated_triangle_forms_two_cones (a b c : ℝ) (h1 : a^2 + b^2 = c^2) : 
  ∃ (cones : ℕ), cones = 2 :=
by
  sorry

end NUMINAMATH_GPT_rotated_triangle_forms_two_cones_l2314_231400


namespace NUMINAMATH_GPT_sum_a2000_inv_a2000_l2314_231449

theorem sum_a2000_inv_a2000 (a : ℂ) (h : a^2 - a + 1 = 0) : a^2000 + 1/(a^2000) = -1 :=
by
    sorry

end NUMINAMATH_GPT_sum_a2000_inv_a2000_l2314_231449


namespace NUMINAMATH_GPT_derivative_at_one_l2314_231463

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.log x

theorem derivative_at_one : (deriv f 1) = 4 := by
  sorry

end NUMINAMATH_GPT_derivative_at_one_l2314_231463


namespace NUMINAMATH_GPT_perpendicular_lines_l2314_231453

theorem perpendicular_lines (m : ℝ) :
  (m+2)*(m-1) + m*(m-4) = 0 ↔ m = 2 ∨ m = -1/2 :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l2314_231453


namespace NUMINAMATH_GPT_arithmetic_sequence_term_number_l2314_231468

theorem arithmetic_sequence_term_number :
  ∀ (a : ℕ → ℤ) (n : ℕ),
    (a 1 = 1) →
    (∀ m, a (m + 1) = a m + 3) →
    (a n = 2014) →
    n = 672 :=
by
  -- conditions
  intro a n h1 h2 h3
  -- proof skipped
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_number_l2314_231468


namespace NUMINAMATH_GPT_total_cable_cost_l2314_231404

theorem total_cable_cost 
    (num_east_west_streets : ℕ)
    (length_east_west_street : ℕ)
    (num_north_south_streets : ℕ)
    (length_north_south_street : ℕ)
    (cable_multiplier : ℕ)
    (cable_cost_per_mile : ℕ)
    (h1 : num_east_west_streets = 18)
    (h2 : length_east_west_street = 2)
    (h3 : num_north_south_streets = 10)
    (h4 : length_north_south_street = 4)
    (h5 : cable_multiplier = 5)
    (h6 : cable_cost_per_mile = 2000) :
    (num_east_west_streets * length_east_west_street + num_north_south_streets * length_north_south_street) * cable_multiplier * cable_cost_per_mile = 760000 := 
by
    sorry

end NUMINAMATH_GPT_total_cable_cost_l2314_231404


namespace NUMINAMATH_GPT_line_equation_l2314_231461

theorem line_equation (P : ℝ × ℝ) (hP : P = (1, 5)) (h1 : ∃ a, a ≠ 0 ∧ (P.1 + P.2 = a)) (h2 : x_intercept = y_intercept) : 
  (∃ a, a ≠ 0 ∧ P = (a, 0) ∧ P = (0, a) → x + y - 6 = 0) ∨ (5*P.1 - P.2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_line_equation_l2314_231461


namespace NUMINAMATH_GPT_num_perfect_squares_mul_36_lt_10pow8_l2314_231483

theorem num_perfect_squares_mul_36_lt_10pow8 : 
  ∃(n : ℕ), n = 1666 ∧ 
  ∀ (N : ℕ), (1 ≤ N) → (N^2 < 10^8) → (N^2 % 36 = 0) → 
  (N ≤ 9996 ∧ N % 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_num_perfect_squares_mul_36_lt_10pow8_l2314_231483


namespace NUMINAMATH_GPT_total_valid_votes_l2314_231479

theorem total_valid_votes (V : ℝ)
  (h1 : ∃ c1 c2 : ℝ, c1 = 0.70 * V ∧ c2 = 0.30 * V)
  (h2 : ∀ c1 c2, c1 - c2 = 182) : V = 455 :=
sorry

end NUMINAMATH_GPT_total_valid_votes_l2314_231479


namespace NUMINAMATH_GPT_sequence_non_existence_l2314_231480

variable (α β : ℝ)
variable (r : ℝ)

theorem sequence_non_existence 
  (hαβ : α * β > 0) :  
  (∃ (x : ℕ → ℝ), x 0 = r ∧ ∀ n, x (n + 1) = (x n + α) / (β * (x n) + 1) → false) ↔ 
  r = - (1 / β) :=
sorry

end NUMINAMATH_GPT_sequence_non_existence_l2314_231480


namespace NUMINAMATH_GPT_proof_problem_l2314_231456

-- Conditions
def in_fourth_quadrant (α : ℝ) : Prop := (α > 3 * Real.pi / 2) ∧ (α < 2 * Real.pi)
def x_coordinate_unit_circle (α : ℝ) : Prop := Real.cos α = 1/3

-- Proof statement
theorem proof_problem (α : ℝ) (h1 : in_fourth_quadrant α) (h2 : x_coordinate_unit_circle α) :
  Real.tan α = -2 * Real.sqrt 2 ∧
  ((Real.sin α)^2 - Real.sqrt 2 * (Real.sin α) * (Real.cos α)) / (1 + (Real.cos α)^2) = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l2314_231456


namespace NUMINAMATH_GPT_evaluate_expression_l2314_231471

-- Define the conditions
def two_pow_nine : ℕ := 2 ^ 9
def neg_one_pow_eight : ℤ := (-1) ^ 8

-- Define the proof statement
theorem evaluate_expression : two_pow_nine + neg_one_pow_eight = 513 := 
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2314_231471


namespace NUMINAMATH_GPT_nth_term_l2314_231452

theorem nth_term (b : ℕ → ℝ) (h₀ : b 1 = 1)
  (h_rec : ∀ n ≥ 1, (b (n + 1))^2 = 36 * (b n)^2) : 
  b 50 = 6^49 :=
by
  sorry

end NUMINAMATH_GPT_nth_term_l2314_231452


namespace NUMINAMATH_GPT_parallelogram_circumference_l2314_231419

-- Define the lengths of the sides of the parallelogram.
def side1 : ℝ := 18
def side2 : ℝ := 12

-- Define the formula for the circumference (or perimeter) of the parallelogram.
def circumference (a b : ℝ) : ℝ :=
  2 * (a + b)

-- Statement of the proof problem:
theorem parallelogram_circumference : circumference side1 side2 = 60 := 
  by
    sorry

end NUMINAMATH_GPT_parallelogram_circumference_l2314_231419


namespace NUMINAMATH_GPT_vector_addition_l2314_231484

variable {𝕍 : Type} [AddCommGroup 𝕍] [Module ℝ 𝕍]
variable (a b : 𝕍)

theorem vector_addition : 
  (1 / 2 : ℝ) • (2 • a - 4 • b) + 2 • b = a :=
by
  sorry

end NUMINAMATH_GPT_vector_addition_l2314_231484


namespace NUMINAMATH_GPT_Arthur_total_distance_l2314_231431

/-- Arthur walks 8 blocks south and then 10 blocks west. Each block is one-fourth of a mile.
How many miles did Arthur walk in total? -/
theorem Arthur_total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length_miles : ℝ) :
  blocks_south = 8 ∧ blocks_west = 10 ∧ block_length_miles = 1/4 →
  (blocks_south + blocks_west) * block_length_miles = 4.5 :=
by
  intro h
  have h1 : blocks_south = 8 := h.1
  have h2 : blocks_west = 10 := h.2.1
  have h3 : block_length_miles = 1 / 4 := h.2.2
  sorry

end NUMINAMATH_GPT_Arthur_total_distance_l2314_231431


namespace NUMINAMATH_GPT_arithmetic_expression_evaluation_l2314_231476

theorem arithmetic_expression_evaluation : 5 * (9 / 3) + 7 * 4 - 36 / 4 = 34 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_evaluation_l2314_231476


namespace NUMINAMATH_GPT_oranges_packed_in_a_week_l2314_231482

open Nat

def oranges_per_box : Nat := 15
def boxes_per_day : Nat := 2150
def days_per_week : Nat := 7

theorem oranges_packed_in_a_week : oranges_per_box * boxes_per_day * days_per_week = 225750 :=
  sorry

end NUMINAMATH_GPT_oranges_packed_in_a_week_l2314_231482


namespace NUMINAMATH_GPT_cost_price_of_pots_l2314_231443

variable (C : ℝ)

-- Define the conditions
def selling_price (C : ℝ) := 1.25 * C
def total_revenue (selling_price : ℝ) := 150 * selling_price

-- State the main proof goal
theorem cost_price_of_pots (h : total_revenue (selling_price C) = 450) : C = 2.4 := by
  sorry

end NUMINAMATH_GPT_cost_price_of_pots_l2314_231443


namespace NUMINAMATH_GPT_beetles_consumed_per_day_l2314_231446

-- Definitions
def bird_eats_beetles (n : Nat) : Nat := 12 * n
def snake_eats_birds (n : Nat) : Nat := 3 * n
def jaguar_eats_snakes (n : Nat) : Nat := 5 * n
def crocodile_eats_jaguars (n : Nat) : Nat := 2 * n

-- Initial values
def initial_jaguars : Nat := 6
def initial_crocodiles : Nat := 30
def net_increase_birds : Nat := 4
def net_increase_snakes : Nat := 2
def net_increase_jaguars : Nat := 1

-- Proof statement
theorem beetles_consumed_per_day : 
  bird_eats_beetles (snake_eats_birds (jaguar_eats_snakes initial_jaguars)) = 1080 := 
by 
  sorry

end NUMINAMATH_GPT_beetles_consumed_per_day_l2314_231446


namespace NUMINAMATH_GPT_counterexample_disproving_proposition_l2314_231429

theorem counterexample_disproving_proposition (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = angle2) : False :=
by
  have h_contradiction : angle1 ≠ angle2 := sorry
  exact h_contradiction h2

end NUMINAMATH_GPT_counterexample_disproving_proposition_l2314_231429


namespace NUMINAMATH_GPT_michael_can_cover_both_classes_l2314_231457

open Nat

def total_students : ℕ := 30
def german_students : ℕ := 20
def japanese_students : ℕ := 24

-- Calculate the number of students taking both German and Japanese using inclusion-exclusion principle.
def both_students : ℕ := german_students + japanese_students - total_students

-- Calculate the number of students only taking German.
def only_german_students : ℕ := german_students - both_students

-- Calculate the number of students only taking Japanese.
def only_japanese_students : ℕ := japanese_students - both_students

-- Calculate the total number of ways to choose 2 students out of 30.
def total_ways_to_choose_2 : ℕ := (total_students * (total_students - 1)) / 2

-- Calculate the number of ways to choose 2 students only taking German or only taking Japanese.
def undesirable_outcomes : ℕ := (only_german_students * (only_german_students - 1)) / 2 + (only_japanese_students * (only_japanese_students - 1)) / 2

-- Calculate the probability of undesirable outcomes.
def undesirable_probability : ℚ := undesirable_outcomes / total_ways_to_choose_2

-- Calculate the probability Michael can cover both German and Japanese classes.
def desired_probability : ℚ := 1 - undesirable_probability

theorem michael_can_cover_both_classes : desired_probability = 25 / 29 := by sorry

end NUMINAMATH_GPT_michael_can_cover_both_classes_l2314_231457


namespace NUMINAMATH_GPT_scientific_notation_of_8450_l2314_231462

theorem scientific_notation_of_8450 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ (8450 : ℝ) = a * 10^n ∧ (a = 8.45) ∧ (n = 3) :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_8450_l2314_231462


namespace NUMINAMATH_GPT_revenue_function_correct_strategy_not_profitable_l2314_231433

-- Given conditions 
def purchase_price : ℝ := 1
def last_year_price : ℝ := 2
def last_year_sales_volume : ℕ := 10000
def last_year_revenue : ℝ := 20000
def proportionality_constant : ℝ := 4
def increased_sales_volume (x : ℝ) : ℝ := proportionality_constant * (2 - x) ^ 2

-- Questions translated to Lean statements
def revenue_this_year (x : ℝ) : ℝ := 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17

theorem revenue_function_correct (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x = 4 * x ^ 3 - 20 * x ^ 2 + 33 * x - 17 :=
by
  sorry

theorem strategy_not_profitable (x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) : 
    revenue_this_year x ≤ last_year_revenue :=
by
  sorry

end NUMINAMATH_GPT_revenue_function_correct_strategy_not_profitable_l2314_231433


namespace NUMINAMATH_GPT_probability_greater_than_n_l2314_231447

theorem probability_greater_than_n (n : ℕ) : 
  (1 ≤ n ∧ n ≤ 5) → (∃ k, k = 6 - n - 1 ∧ k / 6 = 1 / 2) → n = 3 := 
by sorry

end NUMINAMATH_GPT_probability_greater_than_n_l2314_231447


namespace NUMINAMATH_GPT_ezekiel_first_day_distance_l2314_231403

noncomputable def distance_first_day (total_distance second_day_distance third_day_distance : ℕ) :=
  total_distance - (second_day_distance + third_day_distance)

theorem ezekiel_first_day_distance:
  ∀ (total_distance second_day_distance third_day_distance : ℕ),
  total_distance = 50 →
  second_day_distance = 25 →
  third_day_distance = 15 →
  distance_first_day total_distance second_day_distance third_day_distance = 10 :=
by
  intros total_distance second_day_distance third_day_distance h1 h2 h3
  sorry

end NUMINAMATH_GPT_ezekiel_first_day_distance_l2314_231403


namespace NUMINAMATH_GPT_polar_line_equation_l2314_231473

/-- A line that passes through a given point in polar coordinates and is parallel to the polar axis
    has a specific polar coordinate equation. -/
theorem polar_line_equation (r : ℝ) (θ : ℝ) (h : r = 6 ∧ θ = π / 6) : θ = π / 6 :=
by
  /- We are given that the line passes through the point \(C(6, \frac{\pi}{6})\) which means
     \(r = 6\) and \(θ = \frac{\pi}{6}\). Since the line is parallel to the polar axis, 
     the angle \(θ\) remains the same. Therefore, the polar coordinate equation of the line 
     is simply \(θ = \frac{\pi}{6}\). -/
  sorry

end NUMINAMATH_GPT_polar_line_equation_l2314_231473


namespace NUMINAMATH_GPT_roots_sum_reciprocal_squares_l2314_231406

theorem roots_sum_reciprocal_squares (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + bc + ca = 20) (h3 : abc = 3) :
  (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2) = 328 / 9 := 
by
  sorry

end NUMINAMATH_GPT_roots_sum_reciprocal_squares_l2314_231406


namespace NUMINAMATH_GPT_coin_value_difference_l2314_231474

theorem coin_value_difference (p n d : ℕ) (h : p + n + d = 3000) (hp : p ≥ 1) (hn : n ≥ 1) (hd : d ≥ 1) : 
  (p + 5 * n + 10 * d).max - (p + 5 * n + 10 * d).min = 26973 := 
sorry

end NUMINAMATH_GPT_coin_value_difference_l2314_231474


namespace NUMINAMATH_GPT_UncleVanya_travel_time_l2314_231487

-- Define the conditions
variables (x y z : ℝ)
variables (h1 : 2 * x + 3 * y + 20 * z = 66)
variables (h2 : 5 * x + 8 * y + 30 * z = 144)

-- Question: how long will it take to walk 4 km, cycle 5 km, and drive 80 km
theorem UncleVanya_travel_time : 4 * x + 5 * y + 80 * z = 174 :=
sorry

end NUMINAMATH_GPT_UncleVanya_travel_time_l2314_231487


namespace NUMINAMATH_GPT_cost_of_individual_roll_l2314_231458

theorem cost_of_individual_roll
  (p : ℕ) (c : ℝ) (s : ℝ) (x : ℝ)
  (hc : c = 9)
  (hp : p = 12)
  (hs : s = 0.25)
  (h : 12 * x = 9 * (1 + s)) :
  x = 0.9375 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_individual_roll_l2314_231458


namespace NUMINAMATH_GPT_range_of_x_l2314_231489

variable {f : ℝ → ℝ}
variable (hf1 : ∀ x : ℝ, has_deriv_at f (derivative f x) x)
variable (hf2 : ∀ x : ℝ, derivative f x > - f x)

theorem range_of_x (h : f (Real.log 3) = 1/3) : 
  {x : ℝ | f x > 1 / Real.exp x} = Set.Ioi (Real.log 3) := 
by 
  sorry

end NUMINAMATH_GPT_range_of_x_l2314_231489


namespace NUMINAMATH_GPT_distinct_four_digit_integers_with_digit_product_eight_l2314_231407

theorem distinct_four_digit_integers_with_digit_product_eight : 
  ∃ (n : ℕ), 1000 ≤ n ∧ n < 10000 ∧ (∀ (a b c d : ℕ), 10 > a ∧ 10 > b ∧ 10 > c ∧ 10 > d ∧ n = 1000 * a + 100 * b + 10 * c + d ∧ a * b * c * d = 8) ∧ (∃ (count : ℕ), count = 20 ) :=
sorry

end NUMINAMATH_GPT_distinct_four_digit_integers_with_digit_product_eight_l2314_231407


namespace NUMINAMATH_GPT_correct_completion_at_crossroads_l2314_231497

theorem correct_completion_at_crossroads :
  (∀ (s : String), 
    s = "An accident happened at a crossroads a few meters away from a bank" → 
    (∃ (general_sense : Bool), general_sense = tt)) :=
by
  sorry

end NUMINAMATH_GPT_correct_completion_at_crossroads_l2314_231497


namespace NUMINAMATH_GPT_equivalent_operation_l2314_231405

theorem equivalent_operation : 
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  (initial_op * multiply_3_2) = (5 / 4 : ℝ) :=
by
  -- setup operations
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  -- state the goal
  have h : (initial_op * multiply_3_2) = (5 / 4 : ℝ) := sorry
  exact h

end NUMINAMATH_GPT_equivalent_operation_l2314_231405


namespace NUMINAMATH_GPT_water_percentage_in_dried_grapes_l2314_231494

noncomputable def fresh_grape_weight : ℝ := 40  -- weight of fresh grapes in kg
noncomputable def dried_grape_weight : ℝ := 5  -- weight of dried grapes in kg
noncomputable def water_percentage_fresh : ℝ := 0.90  -- percentage of water in fresh grapes

noncomputable def water_weight_fresh : ℝ := fresh_grape_weight * water_percentage_fresh
noncomputable def solid_weight_fresh : ℝ := fresh_grape_weight * (1 - water_percentage_fresh)
noncomputable def water_weight_dried : ℝ := dried_grape_weight - solid_weight_fresh
noncomputable def water_percentage_dried : ℝ := (water_weight_dried / dried_grape_weight) * 100

theorem water_percentage_in_dried_grapes : water_percentage_dried = 20 := by
  sorry

end NUMINAMATH_GPT_water_percentage_in_dried_grapes_l2314_231494


namespace NUMINAMATH_GPT_average_brown_mms_per_bag_l2314_231425

-- Definitions based on the conditions
def bag1_brown_mm : ℕ := 9
def bag2_brown_mm : ℕ := 12
def bag3_brown_mm : ℕ := 8
def bag4_brown_mm : ℕ := 8
def bag5_brown_mm : ℕ := 3
def number_of_bags : ℕ := 5

-- The proof problem statement
theorem average_brown_mms_per_bag : 
  (bag1_brown_mm + bag2_brown_mm + bag3_brown_mm + bag4_brown_mm + bag5_brown_mm) / number_of_bags = 8 := 
by
  sorry

end NUMINAMATH_GPT_average_brown_mms_per_bag_l2314_231425


namespace NUMINAMATH_GPT_probability_p_s_multiple_of_7_l2314_231493

section
variables (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 60) (h2 : 1 ≤ b ∧ b ≤ 60) (h3 : a ≠ b)

theorem probability_p_s_multiple_of_7 :
  (∃ k : ℕ, a * b + a + b = 7 * k) → (64 / 1770 : ℚ) = 32 / 885 :=
sorry
end

end NUMINAMATH_GPT_probability_p_s_multiple_of_7_l2314_231493


namespace NUMINAMATH_GPT_ratio_of_areas_ACP_BQA_l2314_231470

open EuclideanGeometry

-- Define the geometric configuration
variables (A B C D P Q : Point)
  (is_square : square A B C D)
  (is_bisector_CAD : is_angle_bisector A C D P)
  (is_bisector_ABD : is_angle_bisector B A D Q)

-- Define the areas of triangles
def area_triangle (X Y Z : Point) : Real := sorry -- Placeholder for the area function

-- Lean statement for the proof problem
theorem ratio_of_areas_ACP_BQA 
  (h_square : is_square) 
  (h_bisector_CAD : is_bisector_CAD) 
  (h_bisector_ABD : is_bisector_ABD) :
  (area_triangle A C P) / (area_triangle B Q A) = 2 :=
sorry

end NUMINAMATH_GPT_ratio_of_areas_ACP_BQA_l2314_231470


namespace NUMINAMATH_GPT_exists_segment_with_points_l2314_231416

theorem exists_segment_with_points (S : Finset ℕ) (n : ℕ) (hS : S.card = 6 * n)
  (hB : ∃ B : Finset ℕ, B ⊆ S ∧ B.card = 4 * n) (hG : ∃ G : Finset ℕ, G ⊆ S ∧ G.card = 2 * n) :
  ∃ t : Finset ℕ, t ⊆ S ∧ t.card = 3 * n ∧ (∃ B' : Finset ℕ, B' ⊆ t ∧ B'.card = 2 * n) ∧ (∃ G' : Finset ℕ, G' ⊆ t ∧ G'.card = n) :=
  sorry

end NUMINAMATH_GPT_exists_segment_with_points_l2314_231416


namespace NUMINAMATH_GPT_average_discount_rate_l2314_231491

theorem average_discount_rate :
  ∃ x : ℝ, (7200 * (1 - x)^2 = 3528) ∧ x = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_average_discount_rate_l2314_231491


namespace NUMINAMATH_GPT_ratio_father_to_children_after_5_years_l2314_231439

def father's_age := 15
def sum_children_ages := father's_age / 3

def father's_age_after_5_years := father's_age + 5
def sum_children_ages_after_5_years := sum_children_ages + 10

theorem ratio_father_to_children_after_5_years :
  father's_age_after_5_years / sum_children_ages_after_5_years = 4 / 3 := by
  sorry

end NUMINAMATH_GPT_ratio_father_to_children_after_5_years_l2314_231439
