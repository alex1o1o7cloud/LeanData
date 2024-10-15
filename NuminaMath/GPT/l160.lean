import Mathlib

namespace NUMINAMATH_GPT_trapezoid_diagonals_perpendicular_iff_geometric_mean_l160_16016

structure Trapezoid :=
(a b c d e f : ℝ) -- lengths of sides a, b, c, d, and diagonals e, f.
(right_angle : d^2 = a^2 + c^2) -- Condition that makes it a right-angled trapezoid.

theorem trapezoid_diagonals_perpendicular_iff_geometric_mean (T : Trapezoid) :
  (T.e * T.e + T.f * T.f = T.a * T.a + T.b * T.b + T.c * T.c + T.d * T.d) ↔ 
  (T.d * T.d = T.a * T.c) := 
sorry

end NUMINAMATH_GPT_trapezoid_diagonals_perpendicular_iff_geometric_mean_l160_16016


namespace NUMINAMATH_GPT_three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l160_16008

theorem three_hundred_thousand_times_three_hundred_thousand_minus_one_million :
  (300000 * 300000) - 1000000 = 89990000000 := by
  sorry 

end NUMINAMATH_GPT_three_hundred_thousand_times_three_hundred_thousand_minus_one_million_l160_16008


namespace NUMINAMATH_GPT_range_of_expression_l160_16070

theorem range_of_expression (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 ≤ β ∧ β ≤ π / 2) :
    -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π :=
by
  sorry

end NUMINAMATH_GPT_range_of_expression_l160_16070


namespace NUMINAMATH_GPT_percentage_y_more_than_z_l160_16037

theorem percentage_y_more_than_z :
  ∀ (P y x k : ℕ),
    P = 200 →
    740 = x + y + P →
    x = (5 / 4) * y →
    y = P * (1 + k / 100) →
    k = 20 :=
by
  sorry

end NUMINAMATH_GPT_percentage_y_more_than_z_l160_16037


namespace NUMINAMATH_GPT_initial_green_hard_hats_l160_16000

noncomputable def initial_pink_hard_hats : ℕ := 26
noncomputable def initial_yellow_hard_hats : ℕ := 24
noncomputable def carl_taken_pink_hard_hats : ℕ := 4
noncomputable def john_taken_pink_hard_hats : ℕ := 6
noncomputable def john_taken_green_hard_hats (G : ℕ) : ℕ := 2 * john_taken_pink_hard_hats
noncomputable def remaining_pink_hard_hats : ℕ := initial_pink_hard_hats - carl_taken_pink_hard_hats - john_taken_pink_hard_hats
noncomputable def total_remaining_hard_hats (G : ℕ) : ℕ := remaining_pink_hard_hats + (G - john_taken_green_hard_hats G) + initial_yellow_hard_hats

theorem initial_green_hard_hats (G : ℕ) :
  total_remaining_hard_hats G = 43 ↔ G = 15 := by
  sorry

end NUMINAMATH_GPT_initial_green_hard_hats_l160_16000


namespace NUMINAMATH_GPT_num_workers_in_factory_l160_16082

theorem num_workers_in_factory 
  (average_salary_total : ℕ → ℕ → ℕ)
  (old_supervisor_salary : ℕ)
  (average_salary_9_new : ℕ)
  (new_supervisor_salary : ℕ) :
  ∃ (W : ℕ), 
  average_salary_total (W + 1) 430 = W * 430 + 870 ∧ 
  average_salary_9_new = 9 * 390 ∧ 
  W + 1 = (9 * 390 - 510 + 870) / 430 := 
by {
  sorry
}

end NUMINAMATH_GPT_num_workers_in_factory_l160_16082


namespace NUMINAMATH_GPT_complement_M_in_U_l160_16068

def M (x : ℝ) : Prop := 0 < x ∧ x < 2

def complement_M (x : ℝ) : Prop := x ≤ 0 ∨ x ≥ 2

theorem complement_M_in_U (x : ℝ) : ¬ M x ↔ complement_M x :=
by sorry

end NUMINAMATH_GPT_complement_M_in_U_l160_16068


namespace NUMINAMATH_GPT_exists_x_y_with_specific_difference_l160_16026

theorem exists_x_y_with_specific_difference :
  ∃ x y : ℤ, (2 * x^2 + 8 * y = 26) ∧ (x - y = 26) := 
sorry

end NUMINAMATH_GPT_exists_x_y_with_specific_difference_l160_16026


namespace NUMINAMATH_GPT_cannot_determine_orange_groups_l160_16069

-- Definitions of the conditions
def oranges := 87
def bananas := 290
def bananaGroups := 2
def bananasPerGroup := 145

-- Lean statement asserting that the number of groups of oranges 
-- cannot be determined from the given conditions
theorem cannot_determine_orange_groups:
  ∀ (number_of_oranges_per_group : ℕ), 
  (bananasPerGroup * bananaGroups = bananas) ∧ (oranges = 87) → 
  ¬(∃ (number_of_orange_groups : ℕ), oranges = number_of_oranges_per_group * number_of_orange_groups) :=
by
  sorry -- Since we are not required to provide the proof here

end NUMINAMATH_GPT_cannot_determine_orange_groups_l160_16069


namespace NUMINAMATH_GPT_triangle_solid_revolution_correct_l160_16031

noncomputable def triangle_solid_revolution (t : ℝ) (alpha beta gamma : ℝ) (longest_side : string) : ℝ × ℝ :=
  let pi := Real.pi;
  let sin := Real.sin;
  let cos := Real.cos;
  let sqrt := Real.sqrt;
  let to_rad (x : ℝ) : ℝ := x * pi / 180;
  let alpha_rad := to_rad alpha;
  let beta_rad := to_rad beta;
  let gamma_rad := to_rad gamma;
  let a := sqrt (2 * t * sin alpha_rad / (sin beta_rad * sin gamma_rad));
  let b := sqrt (2 * t * sin beta_rad / (sin gamma_rad * sin alpha_rad));
  let m_c := sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  let F := 2 * pi * t * cos ((alpha_rad - beta_rad) / 2) / sin (gamma_rad / 2);
  let K := 2 * pi / 3 * t * sqrt (2 * t * sin alpha_rad * sin beta_rad / sin gamma_rad);
  (F, K)

theorem triangle_solid_revolution_correct :
  triangle_solid_revolution 80.362 (39 + 34/60 + 30/3600) (60 : ℝ) (80 + 25/60 + 30/3600) "c" = (769.3, 1595.3) :=
sorry

end NUMINAMATH_GPT_triangle_solid_revolution_correct_l160_16031


namespace NUMINAMATH_GPT_proof_problem_l160_16049

-- Given condition
variable (a b : ℝ)
variable (h1 : 0 < a) (h2 : 0 < b)
variable (h3 : Real.log a + Real.log (b ^ 2) ≥ 2 * a + (b ^ 2) / 2 - 2)

-- Proof statement
theorem proof_problem : a - 2 * b = 1/2 - 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_proof_problem_l160_16049


namespace NUMINAMATH_GPT_max_log2_x_2log2_y_l160_16028

noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem max_log2_x_2log2_y {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : x + y^2 = 2) :
  log2 x + 2 * log2 y ≤ 0 :=
sorry

end NUMINAMATH_GPT_max_log2_x_2log2_y_l160_16028


namespace NUMINAMATH_GPT_value_of_k_l160_16063

theorem value_of_k : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := 
by 
  sorry

end NUMINAMATH_GPT_value_of_k_l160_16063


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l160_16050

theorem sum_of_squares_of_roots (α β : ℝ)
  (h_root1 : 10 * α^2 - 14 * α - 24 = 0)
  (h_root2 : 10 * β^2 - 14 * β - 24 = 0)
  (h_distinct : α ≠ β) :
  α^2 + β^2 = 169 / 25 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l160_16050


namespace NUMINAMATH_GPT_number_of_donut_selections_l160_16038

-- Definitions for the problem
def g : ℕ := sorry
def c : ℕ := sorry
def p : ℕ := sorry

-- Condition: Pat wants to buy four donuts from three types
def equation : Prop := g + c + p = 4

-- Question: Prove the number of different selections possible
theorem number_of_donut_selections : (∃ n, n = 15) := 
by 
  -- Use combinatorial method to establish this
  sorry

end NUMINAMATH_GPT_number_of_donut_selections_l160_16038


namespace NUMINAMATH_GPT_original_cost_prices_l160_16089

variable (COST_A COST_B COST_C : ℝ)

theorem original_cost_prices :
  (COST_A * 0.8 + 100 = COST_A * 1.05) →
  (COST_B * 1.1 - 80 = COST_B * 0.92) →
  (COST_C * 0.85 + 120 = COST_C * 1.07) →
  COST_A = 400 ∧
  COST_B = 4000 / 9 ∧
  COST_C = 6000 / 11 := by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_original_cost_prices_l160_16089


namespace NUMINAMATH_GPT_graph_of_direct_proportion_is_line_l160_16013

-- Define the direct proportion function
def direct_proportion (k : ℝ) (x : ℝ) : ℝ :=
  k * x

-- State the theorem to prove the graph of this function is a straight line
theorem graph_of_direct_proportion_is_line (k : ℝ) :
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x : ℝ, direct_proportion k x = a * x + b ∧ b = 0 := 
by 
  sorry

end NUMINAMATH_GPT_graph_of_direct_proportion_is_line_l160_16013


namespace NUMINAMATH_GPT_smallest_product_not_factor_of_48_l160_16002

theorem smallest_product_not_factor_of_48 (a b : ℕ) (h1 : a ≠ b) (h2 : a ∣ 48) (h3 : b ∣ 48) (h4 : ¬ (a * b ∣ 48)) : a * b = 32 :=
sorry

end NUMINAMATH_GPT_smallest_product_not_factor_of_48_l160_16002


namespace NUMINAMATH_GPT_symmetry_about_origin_l160_16056

-- Define the conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x : ℝ, g (-x) = g x

-- Define the function v based on f and g
def v (f g : ℝ → ℝ) (x : ℝ) : ℝ := f x * |g x|

-- The theorem statement
theorem symmetry_about_origin (f g : ℝ → ℝ) (h_odd : is_odd f) (h_even : is_even g) : 
  ∀ x : ℝ, v f g (-x) = -v f g x := 
by
  sorry

end NUMINAMATH_GPT_symmetry_about_origin_l160_16056


namespace NUMINAMATH_GPT_hexagon_monochromatic_triangle_probability_l160_16085

open Classical

-- Define the total number of edges in the hexagon
def total_edges : ℕ := 15

-- Define the number of triangles from 6 vertices
def total_triangles : ℕ := Nat.choose 6 3

-- Define the probability that a given triangle is not monochromatic
def prob_not_monochromatic_triangle : ℚ := 3 / 4

-- Calculate the probability of having at least one monochromatic triangle
def prob_at_least_one_monochromatic_triangle : ℚ := 
  1 - (prob_not_monochromatic_triangle ^ total_triangles)

theorem hexagon_monochromatic_triangle_probability :
  abs ((prob_at_least_one_monochromatic_triangle : ℝ) - 0.9968) < 0.0001 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_monochromatic_triangle_probability_l160_16085


namespace NUMINAMATH_GPT_max_min_f_l160_16072

noncomputable def f (x : ℝ) : ℝ := 
  5 * Real.cos x ^ 2 - 6 * Real.sin (2 * x) + 20 * Real.sin x - 30 * Real.cos x + 7

theorem max_min_f :
  (∃ x : ℝ, f x = 16 + 10 * Real.sqrt 13) ∧
  (∃ x : ℝ, f x = 16 - 10 * Real.sqrt 13) :=
sorry

end NUMINAMATH_GPT_max_min_f_l160_16072


namespace NUMINAMATH_GPT_minutes_before_noon_l160_16019

theorem minutes_before_noon
    (x : ℕ)
    (h1 : 20 <= x)
    (h2 : 180 - (x - 20) = 3 * (x - 20)) :
    x = 65 := by
  sorry

end NUMINAMATH_GPT_minutes_before_noon_l160_16019


namespace NUMINAMATH_GPT_guiding_normal_vector_l160_16065

noncomputable def ellipsoid (x y z : ℝ) : ℝ := x^2 + 2 * y^2 + 3 * z^2 - 6

def point_M0 : ℝ × ℝ × ℝ := (1, -1, 1)

def normal_vector (x y z : ℝ) : ℝ × ℝ × ℝ := (
  2 * x,
  4 * y,
  6 * z
)

theorem guiding_normal_vector : normal_vector 1 (-1) 1 = (2, -4, 6) :=
by
  sorry

end NUMINAMATH_GPT_guiding_normal_vector_l160_16065


namespace NUMINAMATH_GPT_total_handshakes_tournament_l160_16057

/-- 
In a women's doubles tennis tournament, four teams of two women competed. After the tournament, 
each woman shook hands only once with each of the other players, except with her own partner.
Prove that the total number of unique handshakes is 24.
-/
theorem total_handshakes_tournament : 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  total_handshakes = 24 :=
by 
  let num_teams := 4
  let team_size := 2
  let total_women := num_teams * team_size
  let handshake_per_woman := total_women - team_size
  let total_handshakes := (total_women * handshake_per_woman) / 2
  have : total_handshakes = 24 := sorry
  exact this

end NUMINAMATH_GPT_total_handshakes_tournament_l160_16057


namespace NUMINAMATH_GPT_impossible_to_achieve_desired_piles_l160_16022

def initial_piles : List ℕ := [51, 49, 5]

def desired_piles : List ℕ := [52, 48, 5]

def combine_piles (x y : ℕ) : ℕ := x + y

def divide_pile (x : ℕ) (h : x % 2 = 0) : List ℕ := [x / 2, x / 2]

theorem impossible_to_achieve_desired_piles :
  ∀ (piles : List ℕ), 
    (piles = initial_piles) →
    (∀ (p : List ℕ), 
      (p = desired_piles) → 
      False) :=
sorry

end NUMINAMATH_GPT_impossible_to_achieve_desired_piles_l160_16022


namespace NUMINAMATH_GPT_positive_integer_conditions_l160_16039

theorem positive_integer_conditions (p : ℕ) (hp : p > 0) :
  (∃ q : ℕ, q > 0 ∧ (5 * p + 36) = q * (2 * p - 9)) ↔ (p = 5 ∨ p = 6 ∨ p = 9 ∨ p = 18) :=
by sorry

end NUMINAMATH_GPT_positive_integer_conditions_l160_16039


namespace NUMINAMATH_GPT_Peggy_needs_to_add_stamps_l160_16040

theorem Peggy_needs_to_add_stamps :
  ∀ (Peggy_stamps Bert_stamps Ernie_stamps : ℕ),
  Peggy_stamps = 75 →
  Ernie_stamps = 3 * Peggy_stamps →
  Bert_stamps = 4 * Ernie_stamps →
  Bert_stamps - Peggy_stamps = 825 :=
by
  intros Peggy_stamps Bert_stamps Ernie_stamps hPeggy hErnie hBert
  sorry

end NUMINAMATH_GPT_Peggy_needs_to_add_stamps_l160_16040


namespace NUMINAMATH_GPT_solve_quadratic_eq_l160_16006

theorem solve_quadratic_eq (x : ℝ) (h : x^2 + 2 * x - 15 = 0) : x = 3 ∨ x = -5 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_quadratic_eq_l160_16006


namespace NUMINAMATH_GPT_min_value_of_a_l160_16090

theorem min_value_of_a (a b c d : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > d) (h4 : a + b + c + d = 2004) (h5 : a^2 - b^2 + c^2 - d^2 = 2004) : a = 503 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l160_16090


namespace NUMINAMATH_GPT_average_salary_l160_16096

def A_salary : ℝ := 9000
def B_salary : ℝ := 5000
def C_salary : ℝ := 11000
def D_salary : ℝ := 7000
def E_salary : ℝ := 9000
def number_of_people : ℝ := 5
def total_salary : ℝ := A_salary + B_salary + C_salary + D_salary + E_salary

theorem average_salary : (total_salary / number_of_people) = 8200 := by
  sorry

end NUMINAMATH_GPT_average_salary_l160_16096


namespace NUMINAMATH_GPT_goods_train_length_l160_16053

noncomputable def speed_kmh : ℕ := 72  -- Speed of the goods train in km/hr
noncomputable def platform_length : ℕ := 280  -- Length of the platform in meters
noncomputable def time_seconds : ℕ := 26  -- Time taken to cross the platform in seconds
noncomputable def speed_mps : ℤ := speed_kmh * 1000 / 3600 -- Speed of the goods train in meters/second

theorem goods_train_length : 20 * time_seconds = 280 + 240 :=
by
  sorry

end NUMINAMATH_GPT_goods_train_length_l160_16053


namespace NUMINAMATH_GPT_find_sum_on_si_l160_16011

noncomputable def sum_invested_on_si (r1 r2 r3 : ℝ) (years_si: ℕ) (ci_rate: ℝ) (principal_ci: ℝ) (years_ci: ℕ) (times_compounded: ℕ) :=
  let ci_rate_period := ci_rate / times_compounded
  let amount_ci := principal_ci * (1 + ci_rate_period / 1)^(years_ci * times_compounded)
  let ci := amount_ci - principal_ci
  let si := ci / 2
  let total_si_rate := r1 / 100 + r2 / 100 + r3 / 100
  let principle_si := si / total_si_rate
  principle_si

theorem find_sum_on_si :
  sum_invested_on_si 0.05 0.06 0.07 3 0.10 4000 2 2 = 2394.51 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_on_si_l160_16011


namespace NUMINAMATH_GPT_problem1_problem2_l160_16042

-- Problem 1: Prove that (-11) + 8 + (-4) = -7
theorem problem1 : (-11) + 8 + (-4) = -7 := by
  sorry

-- Problem 2: Prove that -1^2023 - |1 - 1/3| * (-3/2)^2 = -(5/2)
theorem problem2 : (-1 : ℚ)^2023 - abs (1 - 1/3) * (-3/2)^2 = -(5/2) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l160_16042


namespace NUMINAMATH_GPT_bricks_needed_for_courtyard_l160_16033

noncomputable def total_bricks_required (courtyard_length courtyard_width : ℝ)
  (brick_length_cm brick_width_cm : ℝ) : ℝ :=
  let courtyard_area := courtyard_length * courtyard_width
  let brick_length := brick_length_cm / 100
  let brick_width := brick_width_cm / 100
  let brick_area := brick_length * brick_width
  courtyard_area / brick_area

theorem bricks_needed_for_courtyard :
  total_bricks_required 35 24 15 8 = 70000 := by
  sorry

end NUMINAMATH_GPT_bricks_needed_for_courtyard_l160_16033


namespace NUMINAMATH_GPT_greatest_common_divisor_l160_16091

theorem greatest_common_divisor (n : ℕ) (h1 : ∃ d : ℕ, d = gcd 180 n ∧ (∃ (l : List ℕ), l.length = 5 ∧ ∀ x : ℕ, x ∈ l → x ∣ d)) :
  ∃ x : ℕ, x = 27 :=
by
  sorry

end NUMINAMATH_GPT_greatest_common_divisor_l160_16091


namespace NUMINAMATH_GPT_max_distinct_dance_counts_l160_16048

theorem max_distinct_dance_counts (B G : ℕ) (hB : B = 29) (hG : G = 15) 
  (dance_with : ℕ → ℕ → Prop)
  (h_dance_limit : ∀ b g, dance_with b g → b ≤ B ∧ g ≤ G) :
  ∃ max_counts : ℕ, max_counts = 29 :=
by
  -- The statement of the theorem. Proof is omitted.
  sorry

end NUMINAMATH_GPT_max_distinct_dance_counts_l160_16048


namespace NUMINAMATH_GPT_coordinates_of_P_l160_16064

def P : Prod Int Int := (-1, 2)

theorem coordinates_of_P :
  P = (-1, 2) := 
  by
    -- The proof is omitted as per instructions
    sorry

end NUMINAMATH_GPT_coordinates_of_P_l160_16064


namespace NUMINAMATH_GPT_problem_l160_16034

variables {b1 b2 b3 a1 a2 : ℤ}

-- Condition: five numbers -9, b1, b2, b3, -1 form a geometric sequence.
def is_geometric_seq (b1 b2 b3 : ℤ) : Prop :=
b1^2 = -9 * b2 ∧ b2^2 = b1 * b3 ∧ b1 * b3 = 9

-- Condition: four numbers -9, a1, a2, -3 form an arithmetic sequence.
def is_arithmetic_seq (a1 a2 : ℤ) : Prop :=
2 * a1 = -9 + a2 ∧ 2 * a2 = a1 - 3

-- Proof problem: prove that b2(a2 - a1) = -6
theorem problem (h_geom : is_geometric_seq b1 b2 b3) (h_arith : is_arithmetic_seq a1 a2) : 
  b2 * (a2 - a1) = -6 :=
by sorry

end NUMINAMATH_GPT_problem_l160_16034


namespace NUMINAMATH_GPT_mrs_hilt_candy_l160_16055

theorem mrs_hilt_candy : 2 * 9 + 3 * 9 + 1 * 9 = 54 :=
by
  sorry

end NUMINAMATH_GPT_mrs_hilt_candy_l160_16055


namespace NUMINAMATH_GPT_largest_gcd_l160_16010

theorem largest_gcd (a b : ℕ) (h : a + b = 1008) : ∃ d, d = gcd a b ∧ (∀ d', d' = gcd a b → d' ≤ d) ∧ d = 504 :=
by
  sorry

end NUMINAMATH_GPT_largest_gcd_l160_16010


namespace NUMINAMATH_GPT_cos_135_eq_neg_inv_sqrt2_l160_16077

theorem cos_135_eq_neg_inv_sqrt2 :
  Real.cos (135 * Real.pi / 180) = -1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_cos_135_eq_neg_inv_sqrt2_l160_16077


namespace NUMINAMATH_GPT_find_area_MOI_l160_16007

noncomputable def incenter_coords (a b c : ℝ) (A B C : (ℝ × ℝ)) : (ℝ × ℝ) :=
  ((a * A.1 + b * B.1 + c * C.1) / (a + b + c), (a * A.2 + b * B.2 + c * C.2) / (a + b + c))

noncomputable def shoelace_area (P Q R : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

theorem find_area_MOI :
  let A := (0, 0)
  let B := (8, 0)
  let C := (0, 17)
  let O := (4, 8.5)
  let I := incenter_coords 8 15 17 A B C
  let M := (6.25, 6.25)
  shoelace_area M O I = 25.78125 :=
by
  sorry

end NUMINAMATH_GPT_find_area_MOI_l160_16007


namespace NUMINAMATH_GPT_max_yellow_apples_max_total_apples_l160_16043

-- Definitions for the conditions
def num_green_apples : Nat := 10
def num_yellow_apples : Nat := 13
def num_red_apples : Nat := 18

-- Predicate for the stopping condition
def stop_condition (green yellow red : Nat) : Prop :=
  green < yellow ∧ yellow < red

-- Proof problem for maximum number of yellow apples
theorem max_yellow_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → y ≤ 13) →
  yellow ≤ 13 :=
sorry

-- Proof problem for maximum total number of apples
theorem max_total_apples (green yellow red : Nat) :
  num_green_apples = 10 →
  num_yellow_apples = 13 →
  num_red_apples = 18 →
  (∀ g y r, stop_condition g y r → g + y + r ≤ 39) →
  green + yellow + red ≤ 39 :=
sorry

end NUMINAMATH_GPT_max_yellow_apples_max_total_apples_l160_16043


namespace NUMINAMATH_GPT_problem_statement_l160_16092

theorem problem_statement (a b c : ℝ) 
  (h1 : 2011 * (a + b + c) = 1)
  (h2 : a * b + a * c + b * c = 2011 * a * b * c) :
  a ^ 2011 * b ^ 2011 + c ^ 2011 = 1 / 2011^2011 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l160_16092


namespace NUMINAMATH_GPT_ratio_alan_to_ben_l160_16012

theorem ratio_alan_to_ben (A B L : ℕ) (hA : A = 48) (hL : L = 36) (hB : B = L / 3) : A / B = 4 := by
  sorry

end NUMINAMATH_GPT_ratio_alan_to_ben_l160_16012


namespace NUMINAMATH_GPT_age_of_other_man_l160_16066

-- Definitions of the given conditions
def average_age_increase (avg_men : ℕ → ℝ) (men_removed women_avg : ℕ) : Prop :=
  avg_men 8 + 2 = avg_men 6 + women_avg / 2

def one_man_age : ℕ := 24
def women_avg : ℕ := 30

-- Statement of the problem to prove
theorem age_of_other_man (avg_men : ℕ → ℝ) (other_man : ℕ) :
  average_age_increase avg_men 24 women_avg →
  other_man = 20 :=
sorry

end NUMINAMATH_GPT_age_of_other_man_l160_16066


namespace NUMINAMATH_GPT_tangent_line_slope_l160_16083

theorem tangent_line_slope (m : ℝ) :
  (∀ x y, (x^2 + y^2 - 4*x + 2 = 0) → (y = m * x)) → (m = 1 ∨ m = -1) := 
by
  intro h
  sorry

end NUMINAMATH_GPT_tangent_line_slope_l160_16083


namespace NUMINAMATH_GPT_initial_maple_trees_l160_16005

theorem initial_maple_trees
  (initial_maple_trees : ℕ)
  (to_be_planted : ℕ)
  (final_maple_trees : ℕ)
  (h1 : to_be_planted = 9)
  (h2 : final_maple_trees = 11) :
  initial_maple_trees + to_be_planted = final_maple_trees → initial_maple_trees = 2 := 
by 
  sorry

end NUMINAMATH_GPT_initial_maple_trees_l160_16005


namespace NUMINAMATH_GPT_find_slope_l160_16094

theorem find_slope 
  (k : ℝ)
  (y : ℝ -> ℝ)
  (P : ℝ × ℝ)
  (l : ℝ -> ℝ -> Prop)
  (A B F : ℝ × ℝ)
  (C : ℝ × ℝ -> Prop)
  (d : ℝ × ℝ -> ℝ × ℝ -> ℝ)
  (k_pos : P = (3, 0))
  (k_slope : ∀ x, y x = k * (x - 3))
  (k_int_hyperbola_A : C A)
  (k_int_hyperbola_B : C B)
  (k_focus : F = (2, 0))
  (k_sum_dist : d A F + d B F = 16) :
  k = 1 ∨ k = -1 :=
sorry

end NUMINAMATH_GPT_find_slope_l160_16094


namespace NUMINAMATH_GPT_house_selling_price_l160_16060

theorem house_selling_price
  (original_price : ℝ := 80000)
  (profit_rate : ℝ := 0.20)
  (commission_rate : ℝ := 0.05):
  original_price + (original_price * profit_rate) + (original_price * commission_rate) = 100000 := by
  sorry

end NUMINAMATH_GPT_house_selling_price_l160_16060


namespace NUMINAMATH_GPT_marks_in_physics_l160_16003

section
variables (P C M B CS : ℕ)

-- Given conditions
def condition_1 : Prop := P + C + M + B + CS = 375
def condition_2 : Prop := P + M + B = 255
def condition_3 : Prop := P + C + CS = 210

-- Prove that P = 90
theorem marks_in_physics : condition_1 P C M B CS → condition_2 P M B → condition_3 P C CS → P = 90 :=
by sorry
end

end NUMINAMATH_GPT_marks_in_physics_l160_16003


namespace NUMINAMATH_GPT_winner_percentage_l160_16054

theorem winner_percentage (V_winner V_margin V_total : ℕ) (h_winner: V_winner = 806) (h_margin: V_margin = 312) (h_total: V_total = V_winner + (V_winner - V_margin)) :
  ((V_winner: ℚ) / V_total) * 100 = 62 := by
  sorry

end NUMINAMATH_GPT_winner_percentage_l160_16054


namespace NUMINAMATH_GPT_white_truck_chance_l160_16015

-- Definitions from conditions
def trucks : ℕ := 50
def cars : ℕ := 40
def vans : ℕ := 30

def red_trucks : ℕ := 50 / 2
def black_trucks : ℕ := (20 * 50) / 100

-- The remaining percentage (30%) of trucks is assumed to be white.
def white_trucks : ℕ := (30 * 50) / 100

def total_vehicles : ℕ := trucks + cars + vans

-- Given
def percentage_white_truck : ℕ := (white_trucks * 100) / total_vehicles

-- Theorem that proves the problem statement
theorem white_truck_chance : percentage_white_truck = 13 := 
by
  -- Proof will be written here (currently stubbed)
  sorry

end NUMINAMATH_GPT_white_truck_chance_l160_16015


namespace NUMINAMATH_GPT_number_of_true_propositions_l160_16081

theorem number_of_true_propositions :
  let P1 := false -- Swinging on a swing can be regarded as a translation motion.
  let P2 := false -- Two lines intersected by a third line have equal corresponding angles.
  let P3 := true  -- There is one and only one line passing through a point parallel to a given line.
  let P4 := false -- Angles that are not vertical angles are not equal.
  (if P1 then 1 else 0) + (if P2 then 1 else 0) + (if P3 then 1 else 0) + (if P4 then 1 else 0) = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_true_propositions_l160_16081


namespace NUMINAMATH_GPT_primes_up_to_floor_implies_all_primes_l160_16014

/-- Define the function f. -/
def f (x p : ℕ) : ℕ := x^2 + x + p

/-- Define the initial prime condition. -/
def primes_up_to_floor_sqrt_p_over_3 (p : ℕ) : Prop :=
  ∀ x, x ≤ Nat.floor (Nat.sqrt (p / 3)) → Nat.Prime (f x p)

/-- Define the property we want to prove. -/
def all_primes_up_to_p_minus_2 (p : ℕ) : Prop :=
  ∀ x, x ≤ p - 2 → Nat.Prime (f x p)

/-- The main theorem statement. -/
theorem primes_up_to_floor_implies_all_primes
  (p : ℕ) (h : primes_up_to_floor_sqrt_p_over_3 p) : all_primes_up_to_p_minus_2 p :=
sorry

end NUMINAMATH_GPT_primes_up_to_floor_implies_all_primes_l160_16014


namespace NUMINAMATH_GPT_cannot_be_sum_of_two_or_more_consecutive_integers_l160_16062

def is_power_of_two (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2^k

theorem cannot_be_sum_of_two_or_more_consecutive_integers (n : ℕ) :
  (¬∃ k m : ℕ, k ≥ 2 ∧ n = (k * (2 * m + k + 1)) / 2) ↔ is_power_of_two n :=
by
  sorry

end NUMINAMATH_GPT_cannot_be_sum_of_two_or_more_consecutive_integers_l160_16062


namespace NUMINAMATH_GPT_correct_equation_l160_16080

namespace MathProblem

def is_two_digit_positive_integer (P : ℤ) : Prop :=
  10 ≤ P ∧ P < 100

def equation_A : Prop :=
  ∀ x : ℤ, x^2 + (-98)*x + 2001 = (x - 29) * (x - 69)

def equation_B : Prop :=
  ∀ x : ℤ, x^2 + (-110)*x + 2001 = (x - 23) * (x - 87)

def equation_C : Prop :=
  ∀ x : ℤ, x^2 + 110*x + 2001 = (x + 23) * (x + 87)

def equation_D : Prop :=
  ∀ x : ℤ, x^2 + 98*x + 2001 = (x + 29) * (x + 69)

theorem correct_equation :
  is_two_digit_positive_integer 98 ∧ equation_D :=
  sorry

end MathProblem

end NUMINAMATH_GPT_correct_equation_l160_16080


namespace NUMINAMATH_GPT_gcd_2000_7700_l160_16047

theorem gcd_2000_7700 : Nat.gcd 2000 7700 = 100 := by
  -- Prime factorizations of 2000 and 7700
  have fact_2000 : 2000 = 2^4 * 5^3 := sorry
  have fact_7700 : 7700 = 2^2 * 5^2 * 7 * 11 := sorry
  -- Proof of gcd
  sorry

end NUMINAMATH_GPT_gcd_2000_7700_l160_16047


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l160_16020

def setA (x : ℝ) : Prop := -1 < x ∧ x < 1
def setB (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def setC (x : ℝ) : Prop := 0 ≤ x ∧ x < 1

theorem intersection_of_A_and_B : {x : ℝ | setA x} ∩ {x | setB x} = {x | setC x} := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l160_16020


namespace NUMINAMATH_GPT_ratio_A_B_l160_16097

variable (A B C : ℕ)

theorem ratio_A_B 
  (h1: A + B + C = 98) 
  (h2: B = 30) 
  (h3: (B : ℚ) / C = 5 / 8) 
  : (A : ℚ) / B = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_A_B_l160_16097


namespace NUMINAMATH_GPT_solve_diophantine_equation_l160_16032

def is_solution (m n : ℕ) : Prop := 2^m - 3^n = 1

theorem solve_diophantine_equation : 
  { (m, n) : ℕ × ℕ | is_solution m n } = { (1, 0), (2, 1) } :=
by
  sorry

end NUMINAMATH_GPT_solve_diophantine_equation_l160_16032


namespace NUMINAMATH_GPT_smallest_b_l160_16061

theorem smallest_b (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : a * b * c = 360) : b = 3 :=
sorry

end NUMINAMATH_GPT_smallest_b_l160_16061


namespace NUMINAMATH_GPT_gcd_1029_1437_5649_l160_16001

theorem gcd_1029_1437_5649 : Nat.gcd (Nat.gcd 1029 1437) 5649 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_1029_1437_5649_l160_16001


namespace NUMINAMATH_GPT_original_cost_before_changes_l160_16021

variable (C : ℝ)

theorem original_cost_before_changes (h : 2 * C * 1.20 = 480) : C = 200 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_original_cost_before_changes_l160_16021


namespace NUMINAMATH_GPT_teachers_students_relationship_l160_16025

variables (m n k l : ℕ)

theorem teachers_students_relationship
  (teachers_count : m > 0)
  (students_count : n > 0)
  (students_per_teacher : k > 0)
  (teachers_per_student : l > 0)
  (h1 : ∀ p ∈ (Finset.range m), (Finset.card (Finset.range k)) = k)
  (h2 : ∀ s ∈ (Finset.range n), (Finset.card (Finset.range l)) = l) :
  m * k = n * l :=
sorry

end NUMINAMATH_GPT_teachers_students_relationship_l160_16025


namespace NUMINAMATH_GPT_order_of_means_l160_16036

variables (a b : ℝ)
-- a and b are positive and unequal
axiom h1 : 0 < a
axiom h2 : 0 < b
axiom h3 : a ≠ b

-- Definitions of the means
noncomputable def AM : ℝ := (a + b) / 2
noncomputable def GM : ℝ := Real.sqrt (a * b)
noncomputable def HM : ℝ := (2 * a * b) / (a + b)
noncomputable def QM : ℝ := Real.sqrt ((a^2 + b^2) / 2)

-- The theorem to prove the order of the means
theorem order_of_means (h1 : 0 < a) (h2 : 0 < b) (h3 : a ≠ b) :
  QM a b > AM a b ∧ AM a b > GM a b ∧ GM a b > HM a b :=
sorry

end NUMINAMATH_GPT_order_of_means_l160_16036


namespace NUMINAMATH_GPT_evaluate_expression_l160_16073

theorem evaluate_expression : -20 + 8 * (10 / 2) - 4 = 16 :=
by
  sorry -- Proof to be completed

end NUMINAMATH_GPT_evaluate_expression_l160_16073


namespace NUMINAMATH_GPT_question1_question2_l160_16098

-- Define the sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := { x | 2 * x + a > 0 }
def setB : Set ℝ := { x | x^2 - 2 * x - 3 > 0 }

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Question 1: When a = 2, find the set A ∩ B
theorem question1 : A ∩ B = { x | x > 3 } :=
  sorry

-- Question 2: If A ∩ (complement of B) = ∅, find the range of a
theorem question2 : A ∩ (U \ B) = ∅ → a ≤ -6 :=
  sorry

end NUMINAMATH_GPT_question1_question2_l160_16098


namespace NUMINAMATH_GPT_christen_peeled_potatoes_l160_16087

open Nat

theorem christen_peeled_potatoes :
  ∀ (total_potatoes homer_rate homer_time christen_rate : ℕ) (combined_rate : ℕ),
    total_potatoes = 60 →
    homer_rate = 4 →
    homer_time = 6 →
    christen_rate = 6 →
    combined_rate = homer_rate + christen_rate →
    Nat.ceil ((total_potatoes - (homer_rate * homer_time)) / combined_rate * christen_rate) = 21 :=
by
  intros total_potatoes homer_rate homer_time christen_rate combined_rate
  intros htp hr ht cr cr_def
  rw [htp, hr, ht, cr, cr_def]
  sorry

end NUMINAMATH_GPT_christen_peeled_potatoes_l160_16087


namespace NUMINAMATH_GPT_product_of_solutions_l160_16024

theorem product_of_solutions (x : ℝ) (h : |(18 / x) - 6| = 3) : 2 * 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_product_of_solutions_l160_16024


namespace NUMINAMATH_GPT_share_of_B_l160_16004

noncomputable def problem_statement (A B C : ℝ) : Prop :=
  A + B + C = 595 ∧ A = (2/3) * B ∧ B = (1/4) * C

theorem share_of_B (A B C : ℝ) (h : problem_statement A B C) : B = 105 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_share_of_B_l160_16004


namespace NUMINAMATH_GPT_max_xy_value_l160_16095

theorem max_xy_value {x y : ℝ} (h : 2 * x + y = 1) : ∃ z, z = x * y ∧ z = 1 / 8 :=
by sorry

end NUMINAMATH_GPT_max_xy_value_l160_16095


namespace NUMINAMATH_GPT_speed_of_man_rowing_upstream_l160_16009

-- Define conditions
def V_m : ℝ := 20 -- speed of the man in still water (kmph)
def V_downstream : ℝ := 25 -- speed of the man rowing downstream (kmph)
def V_s : ℝ := V_downstream - V_m -- calculate the speed of the stream

-- Define the theorem to prove the speed of the man rowing upstream
theorem speed_of_man_rowing_upstream 
  (V_m : ℝ) (V_downstream : ℝ) (V_s : ℝ := V_downstream - V_m) : 
  V_upstream = V_m - V_s :=
by
  sorry

end NUMINAMATH_GPT_speed_of_man_rowing_upstream_l160_16009


namespace NUMINAMATH_GPT_problem_l160_16084

theorem problem
  (a b c d e : ℝ)
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : e < 0)
  (h4 : |e| = 1) :
  (- (a * b))^2009 - (c + d)^2010 - e^2011 = 0 := 
by
  sorry

end NUMINAMATH_GPT_problem_l160_16084


namespace NUMINAMATH_GPT_find_some_number_l160_16041

theorem find_some_number (some_number : ℕ) : 
  ( ∃ n:ℕ, n = 54 ∧ (n / 18) * (n / some_number) = 1 ) ∧ some_number = 162 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_some_number_l160_16041


namespace NUMINAMATH_GPT_cone_curved_surface_area_at_5_seconds_l160_16059

theorem cone_curved_surface_area_at_5_seconds :
  let l := λ t : ℝ => 10 + 2 * t
  let r := λ t : ℝ => 5 + 1 * t
  let CSA := λ t : ℝ => Real.pi * r t * l t
  CSA 5 = 160 * Real.pi :=
by
  -- Definitions and calculations in the problem ensure this statement
  sorry

end NUMINAMATH_GPT_cone_curved_surface_area_at_5_seconds_l160_16059


namespace NUMINAMATH_GPT_hyperbola_asymptote_slope_l160_16044

theorem hyperbola_asymptote_slope :
  (∃ m : ℚ, m > 0 ∧ ∀ x : ℚ, ∀ y : ℚ, ((x*x/16 - y*y/25 = 1) → (y = m * x ∨ y = -m * x))) → m = 5/4 :=
sorry

end NUMINAMATH_GPT_hyperbola_asymptote_slope_l160_16044


namespace NUMINAMATH_GPT_value_of_a_plus_b_is_zero_l160_16017

noncomputable def sum_geometric_sequence (a b : ℝ) (n : ℕ) : ℝ :=
  a * 2^n + b

theorem value_of_a_plus_b_is_zero (a b : ℝ) (S : ℕ → ℝ) (hS : ∀ n, S n = sum_geometric_sequence a b n) :
  a + b = 0 := 
sorry

end NUMINAMATH_GPT_value_of_a_plus_b_is_zero_l160_16017


namespace NUMINAMATH_GPT_speed_of_train_l160_16027

-- Conditions
def train_length : ℝ := 180
def total_length : ℝ := 195
def time_cross_bridge : ℝ := 30

-- Conversion factor for units (1 m/s = 3.6 km/hr)
def conversion_factor : ℝ := 3.6

-- Theorem statement
theorem speed_of_train : 
  (total_length - train_length) / time_cross_bridge * conversion_factor = 23.4 :=
sorry

end NUMINAMATH_GPT_speed_of_train_l160_16027


namespace NUMINAMATH_GPT_simplified_value_of_expression_l160_16076

theorem simplified_value_of_expression :
  (12 ^ 0.6) * (12 ^ 0.4) * (8 ^ 0.2) * (8 ^ 0.8) = 96 := 
by
  sorry

end NUMINAMATH_GPT_simplified_value_of_expression_l160_16076


namespace NUMINAMATH_GPT_annual_income_A_l160_16058

variable (A B C : ℝ)
variable (monthly_income_C : C = 17000)
variable (monthly_income_B : B = C + 0.12 * C)
variable (ratio_A_to_B : A / B = 5 / 2)

theorem annual_income_A (A B C : ℝ) 
    (hC : C = 17000) 
    (hB : B = C + 0.12 * C) 
    (hR : A / B = 5 / 2) : 
    A * 12 = 571200 :=
by
  sorry

end NUMINAMATH_GPT_annual_income_A_l160_16058


namespace NUMINAMATH_GPT_age_difference_constant_l160_16079

theorem age_difference_constant (seokjin_age_mother_age_diff : ∀ (t : ℕ), 33 - 7 = 26) : 
  ∀ (n : ℕ), 33 + n - (7 + n) = 26 := 
by
  sorry

end NUMINAMATH_GPT_age_difference_constant_l160_16079


namespace NUMINAMATH_GPT_rectangle_length_width_difference_l160_16078

theorem rectangle_length_width_difference
  (x y : ℝ)
  (h1 : y = 1 / 3 * x)
  (h2 : 2 * x + 2 * y = 32)
  (h3 : Real.sqrt (x^2 + y^2) = 17) :
  abs (x - y) = 8 :=
sorry

end NUMINAMATH_GPT_rectangle_length_width_difference_l160_16078


namespace NUMINAMATH_GPT_angle_C_in_triangle_l160_16052

theorem angle_C_in_triangle (A B C : ℝ) (h : A + B = 80) : C = 100 :=
by
  -- The measure of angle C follows from the condition and the properties of triangles.
  sorry

end NUMINAMATH_GPT_angle_C_in_triangle_l160_16052


namespace NUMINAMATH_GPT_sets_satisfying_union_l160_16018

open Set

theorem sets_satisfying_union :
  {B : Set ℕ | {1, 2} ∪ B = {1, 2, 3}} = { {3}, {1, 3}, {2, 3}, {1, 2, 3} } :=
by
  sorry

end NUMINAMATH_GPT_sets_satisfying_union_l160_16018


namespace NUMINAMATH_GPT_chord_ratio_l160_16029

theorem chord_ratio {FQ HQ : ℝ} (h : EQ * FQ = GQ * HQ) (h_eq : EQ = 5) (h_gq : GQ = 12) : 
  FQ / HQ = 12 / 5 :=
by
  rw [h_eq, h_gq] at h
  sorry

end NUMINAMATH_GPT_chord_ratio_l160_16029


namespace NUMINAMATH_GPT_addition_in_sets_l160_16071

theorem addition_in_sets (a b : ℤ) (hA : ∃ k : ℤ, a = 2 * k) (hB : ∃ k : ℤ, b = 2 * k + 1) : ∃ k : ℤ, a + b = 2 * k + 1 :=
by
  sorry

end NUMINAMATH_GPT_addition_in_sets_l160_16071


namespace NUMINAMATH_GPT_find_d_l160_16075

-- Given conditions
def line_eq (x y : ℚ) : Prop := y = (3 * x - 4) / 4

def parametrized_eq (v d : ℚ × ℚ) (t x y : ℚ) : Prop :=
  (x, y) = (v.1 + t * d.1, v.2 + t * d.2)

def distance_eq (x y : ℚ) (t : ℚ) : Prop :=
  (x - 3) * (x - 3) + (y - 1) * (y - 1) = t * t

-- The proof problem statement
theorem find_d (d : ℚ × ℚ) 
  (h_d : d = (7/2, 5/2)) :
  ∀ (x y t : ℚ) (v : ℚ × ℚ) (h_v : v = (3, 1)),
    (x ≥ 3) → 
    line_eq x y → 
    parametrized_eq v d t x y → 
    distance_eq x y t → 
    d = (7/2, 5/2) := 
by 
  intros;
  sorry


end NUMINAMATH_GPT_find_d_l160_16075


namespace NUMINAMATH_GPT_value_of_B_minus_3_plus_A_l160_16035

theorem value_of_B_minus_3_plus_A (A B : ℝ) (h : A + B = 5) : B - 3 + A = 2 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_B_minus_3_plus_A_l160_16035


namespace NUMINAMATH_GPT_birdseed_needed_weekly_birdseed_needed_l160_16088

def parakeet_daily_consumption := 2
def parrot_daily_consumption := 14
def finch_daily_consumption := parakeet_daily_consumption / 2
def num_parakeets := 3
def num_parrots := 2
def num_finches := 4
def days_in_week := 7

theorem birdseed_needed :
  num_parakeets * parakeet_daily_consumption +
  num_parrots * parrot_daily_consumption +
  num_finches * finch_daily_consumption = 38 :=
by
  sorry

theorem weekly_birdseed_needed :
  38 * days_in_week = 266 :=
by
  sorry

end NUMINAMATH_GPT_birdseed_needed_weekly_birdseed_needed_l160_16088


namespace NUMINAMATH_GPT_percentage_increase_l160_16099

variables (P : ℝ) (buy_price : ℝ := 0.60 * P) (sell_price : ℝ := 1.08000000000000007 * P)

theorem percentage_increase (h: (0.60 : ℝ) * P = buy_price) (h1: (1.08000000000000007 : ℝ) * P = sell_price) :
  ((sell_price - buy_price) / buy_price) * 100 = 80.00000000000001 :=
  sorry

end NUMINAMATH_GPT_percentage_increase_l160_16099


namespace NUMINAMATH_GPT_complement_intersection_range_of_a_l160_16067

open Set

variable {α : Type*} [TopologicalSpace α]

def U : Set ℝ := univ

def A : Set ℝ := { x | -1 < x ∧ x < 1 }

def B : Set ℝ := { x | 1/2 ≤ x ∧ x ≤ 3/2 }

def C (a : ℝ) : Set ℝ := { x | a - 4 < x ∧ x ≤ 2 * a - 7 }

-- Question 1
theorem complement_intersection (x : ℝ) :
  x ∈ (U \ A) ∩ B ↔ 1 ≤ x ∧ x ≤ 3 / 2 := sorry

-- Question 2
theorem range_of_a {a : ℝ} (h : A ∩ C a = C a) : a < 4 := sorry

end NUMINAMATH_GPT_complement_intersection_range_of_a_l160_16067


namespace NUMINAMATH_GPT_circle_equations_l160_16030

-- Given conditions: the circle passes through points O(0,0), A(1,1), B(4,2)
-- Prove the general equation of the circle and the standard equation 

theorem circle_equations : 
  ∃ (D E F : ℝ), (∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ 
                      (x, y) = (0, 0) ∨ (x, y) = (1, 1) ∨ (x, y) = (4, 2)) ∧
  (D = -8) ∧ (E = 6) ∧ (F = 0) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 8 * x + 6 * y = 0 ↔ (x - 4)^2 + (y + 3)^2 = 25) :=
sorry

end NUMINAMATH_GPT_circle_equations_l160_16030


namespace NUMINAMATH_GPT_range_of_a_l160_16046

open Real

noncomputable def doesNotPassThroughSecondQuadrant (a : ℝ) : Prop :=
  ∀ x y : ℝ, (3 * a - 1) * x + (2 - a) * y - 1 ≠ 0

theorem range_of_a : {a : ℝ | doesNotPassThroughSecondQuadrant a} = {a : ℝ | 2 ≤ a } :=
by
  ext
  sorry

end NUMINAMATH_GPT_range_of_a_l160_16046


namespace NUMINAMATH_GPT_number_of_correct_answers_l160_16086

-- We define variables C (number of correct answers) and W (number of wrong answers).
variables (C W : ℕ)

-- Define the conditions given in the problem.
def conditions :=
  C + W = 75 ∧ 4 * C - W = 125

-- Define the theorem which states that the number of correct answers is 40.
theorem number_of_correct_answers
  (h : conditions C W) :
  C = 40 :=
sorry

end NUMINAMATH_GPT_number_of_correct_answers_l160_16086


namespace NUMINAMATH_GPT_cubes_painted_on_one_side_l160_16051

def is_cube_painted_on_one_side (l w h : ℕ) (cube_size : ℕ) : ℕ :=
  let top_bottom := (l - 2) * (w - 2) * 2
  let front_back := (l - 2) * (h - 2) * 2
  let left_right := (w - 2) * (h - 2) * 2
  top_bottom + front_back + left_right

theorem cubes_painted_on_one_side (l w h cube_size : ℕ) (h_l : l = 5) (h_w : w = 4) (h_h : h = 3) (h_cube_size : cube_size = 1) :
  is_cube_painted_on_one_side l w h cube_size = 22 :=
by
  sorry

end NUMINAMATH_GPT_cubes_painted_on_one_side_l160_16051


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l160_16045

section geometric_progression

variables {a b c : ℝ}

def geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, a = b / r ∧ c = b * r

def necessary_condition (a b c : ℝ) : Prop :=
  a * c = b^2

theorem necessary_but_not_sufficient :
  (geometric_progression a b c → necessary_condition a b c) ∧
  (¬ (necessary_condition a b c → geometric_progression a b c)) :=
by sorry

end geometric_progression

end NUMINAMATH_GPT_necessary_but_not_sufficient_l160_16045


namespace NUMINAMATH_GPT_find_original_number_l160_16074

theorem find_original_number (x : ℚ) (h : 1 + 1 / x = 8 / 3) : x = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_find_original_number_l160_16074


namespace NUMINAMATH_GPT_sum_of_vertices_l160_16093

theorem sum_of_vertices (vertices_rectangle : ℕ) (vertices_pentagon : ℕ) 
  (h_rect : vertices_rectangle = 4) (h_pent : vertices_pentagon = 5) : 
  vertices_rectangle + vertices_pentagon = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_vertices_l160_16093


namespace NUMINAMATH_GPT_bound_on_k_l160_16023

variables {n k : ℕ}
variables (a : ℕ → ℕ) (h1 : 1 ≤ k) (h2 : ∀ i j, 1 ≤ i → j ≤ k → i < j → a i < a j)
variables (h3 : ∀ i, a i ≤ n) (h4 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → a i ≠ a j))
variables (h5 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → ∀ m p, m ≤ p → m ≤ k → p ≤ k → a i + a j ≠ a m + a p))

theorem bound_on_k : k ≤ Nat.floor (Real.sqrt (2 * n) + 1) :=
sorry

end NUMINAMATH_GPT_bound_on_k_l160_16023
