import Mathlib

namespace NUMINAMATH_GPT_prime_sum_product_l2001_200181

theorem prime_sum_product (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hsum : p + q = 102) (hgt : p > 30 ∨ q > 30) :
  p * q = 2201 := 
sorry

end NUMINAMATH_GPT_prime_sum_product_l2001_200181


namespace NUMINAMATH_GPT_frog_arrangements_l2001_200112

theorem frog_arrangements :
  let total_frogs := 7
  let green_frogs := 2
  let red_frogs := 3
  let blue_frogs := 2
  let valid_sequences := 4
  let green_permutations := Nat.factorial green_frogs
  let red_permutations := Nat.factorial red_frogs
  let blue_permutations := Nat.factorial blue_frogs
  let total_permutations := valid_sequences * (green_permutations * red_permutations * blue_permutations)
  total_frogs = green_frogs + red_frogs + blue_frogs → 
  green_frogs = 2 ∧ red_frogs = 3 ∧ blue_frogs = 2 →
  valid_sequences = 4 →
  total_permutations = 96 := 
by
  -- Given conditions lead to the calculation of total permutations 
  sorry

end NUMINAMATH_GPT_frog_arrangements_l2001_200112


namespace NUMINAMATH_GPT_geometric_sequence_general_formula_arithmetic_sequence_sum_l2001_200153

-- Problem (I)
theorem geometric_sequence_general_formula (a : ℕ → ℝ) (q a1 : ℝ)
  (h1 : ∀ n, a (n + 1) = q * a n)
  (h2 : a 1 + a 2 = 6)
  (h3 : a 1 * a 2 = a 3) :
  a n = 2 ^ n :=
sorry

-- Problem (II)
theorem arithmetic_sequence_sum (a b : ℕ → ℝ) (S T : ℕ → ℝ)
  (h1 : ∀ n, a n = 2 ^ n)
  (h2 : ∀ n, S n = (n * (b 1 + b n)) / 2)
  (h3 : ∀ n, S (2 * n + 1) = b n * b (n + 1))
  (h4 : ∀ n, b n = 2 * n + 1) :
  T n = 5 - (2 * n + 5) / 2 ^ n :=
sorry

end NUMINAMATH_GPT_geometric_sequence_general_formula_arithmetic_sequence_sum_l2001_200153


namespace NUMINAMATH_GPT_evaluate_expression_l2001_200187

noncomputable def ln (x : ℝ) : ℝ := Real.log x

theorem evaluate_expression : 
  2017 ^ ln (ln 2017) - (ln 2017) ^ ln 2017 = 0 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l2001_200187


namespace NUMINAMATH_GPT_recurring_subtraction_l2001_200102

theorem recurring_subtraction (x y : ℚ) (h1 : x = 35 / 99) (h2 : y = 7 / 9) : x - y = -14 / 33 := by
  sorry

end NUMINAMATH_GPT_recurring_subtraction_l2001_200102


namespace NUMINAMATH_GPT_smallest_number_of_students_l2001_200157

/--
At a school, the ratio of 10th-graders to 8th-graders is 3:2, 
and the ratio of 10th-graders to 9th-graders is 5:3. 
Prove that the smallest number of students from these grades is 34.
-/
theorem smallest_number_of_students {G8 G9 G10 : ℕ} 
  (h1 : 3 * G8 = 2 * G10) (h2 : 5 * G9 = 3 * G10) : 
  G10 + G8 + G9 = 34 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_students_l2001_200157


namespace NUMINAMATH_GPT_solve_abs_equation_l2001_200143

theorem solve_abs_equation (x : ℝ) :
  |2 * x - 1| + |x - 2| = |x + 1| ↔ 1 / 2 ≤ x ∧ x ≤ 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_abs_equation_l2001_200143


namespace NUMINAMATH_GPT_molecular_weight_of_10_moles_l2001_200158

-- Define the molecular weight of a compound as a constant
def molecular_weight (compound : Type) : ℝ := 840

-- Prove that the molecular weight of 10 moles of the compound is the same as the molecular weight of 1 mole of the compound
theorem molecular_weight_of_10_moles (compound : Type) :
  molecular_weight compound = 840 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_molecular_weight_of_10_moles_l2001_200158


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2001_200182

theorem quadratic_inequality_solution (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 + m * x - 1 < 0) ↔ -4 < m ∧ m ≤ 0 :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2001_200182


namespace NUMINAMATH_GPT_Hazel_shirts_proof_l2001_200108

variable (H : ℕ)

def shirts_received_by_Razel (h_shirts : ℕ) : ℕ :=
  2 * h_shirts

def total_shirts (h_shirts : ℕ) (r_shirts : ℕ) : ℕ :=
  h_shirts + r_shirts

theorem Hazel_shirts_proof
  (h_shirts : ℕ)
  (r_shirts : ℕ)
  (total : ℕ)
  (H_nonneg : 0 ≤ h_shirts)
  (R_twice_H : r_shirts = shirts_received_by_Razel h_shirts)
  (T_total : total = total_shirts h_shirts r_shirts)
  (total_is_18 : total = 18) :
  h_shirts = 6 :=
by
  sorry

end NUMINAMATH_GPT_Hazel_shirts_proof_l2001_200108


namespace NUMINAMATH_GPT_team_total_games_123_l2001_200199

theorem team_total_games_123 {G : ℕ} 
  (h1 : (55 / 100) * 35 + (90 / 100) * (G - 35) = (80 / 100) * G) : 
  G = 123 :=
sorry

end NUMINAMATH_GPT_team_total_games_123_l2001_200199


namespace NUMINAMATH_GPT_circle_tangent_problem_solution_l2001_200105

noncomputable def circle_tangent_problem
(radius : ℝ)
(center : ℝ × ℝ)
(point_A : ℝ × ℝ)
(distance_OA : ℝ)
(segment_BC : ℝ) : ℝ :=
  let r := radius
  let O := center
  let A := point_A
  let OA := distance_OA
  let BC := segment_BC
  let AT := Real.sqrt (OA^2 - r^2)
  2 * AT - BC

-- Definitions for the conditions
def radius : ℝ := 8
def center : ℝ × ℝ := (0, 0)
def point_A : ℝ × ℝ := (17, 0)
def distance_OA : ℝ := 17
def segment_BC : ℝ := 12

-- Statement of the problem as an example theorem
theorem circle_tangent_problem_solution :
  circle_tangent_problem radius center point_A distance_OA segment_BC = 18 :=
by
  -- We would provide the proof here. The proof steps are not required as per the instructions.
  sorry

end NUMINAMATH_GPT_circle_tangent_problem_solution_l2001_200105


namespace NUMINAMATH_GPT_find_triple_sum_l2001_200144

theorem find_triple_sum (x y z : ℝ) 
  (h1 : y + z = 20 - 4 * x)
  (h2 : x + z = 1 - 4 * y)
  (h3 : x + y = -12 - 4 * z) :
  3 * x + 3 * y + 3 * z = 9 / 2 := 
sorry

end NUMINAMATH_GPT_find_triple_sum_l2001_200144


namespace NUMINAMATH_GPT_tan_theta_equation_l2001_200189

theorem tan_theta_equation (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 6) :
  Real.tan θ + Real.tan (4 * θ) + Real.tan (6 * θ) = 0 → Real.tan θ = 1 / Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_theta_equation_l2001_200189


namespace NUMINAMATH_GPT_odd_expression_l2001_200124

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1
def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem odd_expression (k m : ℤ) (o := 2 * k + 3) (n := 2 * m) :
  is_odd (o^2 + n * o) :=
by sorry

end NUMINAMATH_GPT_odd_expression_l2001_200124


namespace NUMINAMATH_GPT_compute_expression_l2001_200163

theorem compute_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (h3 : x = 1 / z^2) : 
  (x - 1 / x) * (z^2 + 1 / z^2) = x^2 - z^4 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l2001_200163


namespace NUMINAMATH_GPT_sum_gcd_lcm_is_39_l2001_200168

theorem sum_gcd_lcm_is_39 : Nat.gcd 30 81 + Nat.lcm 36 12 = 39 := by 
  sorry

end NUMINAMATH_GPT_sum_gcd_lcm_is_39_l2001_200168


namespace NUMINAMATH_GPT_line_transformation_equiv_l2001_200173

theorem line_transformation_equiv :
  (∀ x y: ℝ, (2 * x - y - 3 = 0) ↔
    (7 * (x + 2 * y) - 5 * (-x + 4 * y) - 18 = 0)) :=
sorry

end NUMINAMATH_GPT_line_transformation_equiv_l2001_200173


namespace NUMINAMATH_GPT_hamburger_per_meatball_l2001_200192

theorem hamburger_per_meatball (family_members : ℕ) (total_hamburger : ℕ) (antonio_meatballs : ℕ) 
    (hmembers : family_members = 8)
    (hhamburger : total_hamburger = 4)
    (hantonio : antonio_meatballs = 4) : 
    (total_hamburger : ℝ) / (family_members * antonio_meatballs) = 0.125 := 
by
  sorry

end NUMINAMATH_GPT_hamburger_per_meatball_l2001_200192


namespace NUMINAMATH_GPT_taxi_fare_l2001_200165

theorem taxi_fare :
  ∀ (initial_fee rate_per_increment increment_distance total_distance : ℝ),
    initial_fee = 2.35 →
    rate_per_increment = 0.35 →
    increment_distance = (2 / 5) →
    total_distance = 3.6 →
    (initial_fee + rate_per_increment * (total_distance / increment_distance)) = 5.50 :=
by
  intros initial_fee rate_per_increment increment_distance total_distance
  intro h1 h2 h3 h4
  sorry -- Proof is not required.

end NUMINAMATH_GPT_taxi_fare_l2001_200165


namespace NUMINAMATH_GPT_noodles_given_to_William_l2001_200126

def initial_noodles : ℝ := 54.0
def noodles_left : ℝ := 42.0
def noodles_given : ℝ := initial_noodles - noodles_left

theorem noodles_given_to_William : noodles_given = 12.0 := 
by
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_noodles_given_to_William_l2001_200126


namespace NUMINAMATH_GPT_cube_inequality_l2001_200150

theorem cube_inequality (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : a^3 + b^3 > a^2 * b + a * b^2 := 
sorry

end NUMINAMATH_GPT_cube_inequality_l2001_200150


namespace NUMINAMATH_GPT_probability_drawing_balls_l2001_200170

theorem probability_drawing_balls :
  let total_balls := 15
  let red_balls := 10
  let blue_balls := 5
  let drawn_balls := 4
  let num_ways_to_draw_4_balls := Nat.choose total_balls drawn_balls
  let num_ways_to_draw_3_red_1_blue := (Nat.choose red_balls 3) * (Nat.choose blue_balls 1)
  let num_ways_to_draw_1_red_3_blue := (Nat.choose red_balls 1) * (Nat.choose blue_balls 3)
  let total_favorable_outcomes := num_ways_to_draw_3_red_1_blue + num_ways_to_draw_1_red_3_blue
  let probability := total_favorable_outcomes / num_ways_to_draw_4_balls
  probability = (140 : ℚ) / 273 :=
sorry

end NUMINAMATH_GPT_probability_drawing_balls_l2001_200170


namespace NUMINAMATH_GPT_animals_consuming_hay_l2001_200198

-- Define the rate of consumption for each animal
def rate_goat : ℚ := 1 / 6 -- goat consumes 1 cartload per 6 weeks
def rate_sheep : ℚ := 1 / 8 -- sheep consumes 1 cartload per 8 weeks
def rate_cow : ℚ := 1 / 3 -- cow consumes 1 cartload per 3 weeks

-- Define the number of animals
def num_goats : ℚ := 5
def num_sheep : ℚ := 3
def num_cows : ℚ := 2

-- Define the total rate of consumption
def total_rate : ℚ := (num_goats * rate_goat) + (num_sheep * rate_sheep) + (num_cows * rate_cow)

-- Define the total amount of hay to be consumed
def total_hay : ℚ := 30

-- Define the time required to consume the total hay at the calculated rate
def time_required : ℚ := total_hay / total_rate

-- Theorem stating the time required to consume 30 cartloads of hay is 16 weeks.
theorem animals_consuming_hay : time_required = 16 := by
  sorry

end NUMINAMATH_GPT_animals_consuming_hay_l2001_200198


namespace NUMINAMATH_GPT_third_derivative_correct_l2001_200147

noncomputable def func (x : ℝ) : ℝ := (1 + x^2) * Real.arctan x

theorem third_derivative_correct :
  (deriv^[3] func) x = (4 / (1 + x^2)^2) :=
sorry

end NUMINAMATH_GPT_third_derivative_correct_l2001_200147


namespace NUMINAMATH_GPT_simplify_fraction_l2001_200141

theorem simplify_fraction :
  (2 / (3 + Real.sqrt 5)) * (2 / (3 - Real.sqrt 5)) = 1 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2001_200141


namespace NUMINAMATH_GPT_ordered_pairs_unique_solution_l2001_200177

theorem ordered_pairs_unique_solution :
  ∃! (b c : ℕ), (b > 0) ∧ (c > 0) ∧ (b^2 - 4 * c = 0) ∧ (c^2 - 4 * b = 0) :=
sorry

end NUMINAMATH_GPT_ordered_pairs_unique_solution_l2001_200177


namespace NUMINAMATH_GPT_tangent_line_parabola_l2001_200193

theorem tangent_line_parabola (d : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + d → y^2 = 12 * x) → d = 1 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_parabola_l2001_200193


namespace NUMINAMATH_GPT_angle_C_modified_l2001_200121

theorem angle_C_modified (A B C : ℝ) (h_eq_triangle: A = B) (h_C_modified: C = A + 40) (h_sum_angles: A + B + C = 180) : 
  C = 86.67 := 
by 
  sorry

end NUMINAMATH_GPT_angle_C_modified_l2001_200121


namespace NUMINAMATH_GPT_find_x6_l2001_200109

-- Definition of the variables xi for i = 1, ..., 10.
variables {x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ}

-- Given conditions as equations.
axiom eq1 : (x2 + x4) / 2 = 3
axiom eq2 : (x4 + x6) / 2 = 5
axiom eq3 : (x6 + x8) / 2 = 7
axiom eq4 : (x8 + x10) / 2 = 9
axiom eq5 : (x10 + x2) / 2 = 1

axiom eq6 : (x1 + x3) / 2 = 2
axiom eq7 : (x3 + x5) / 2 = 4
axiom eq8 : (x5 + x7) / 2 = 6
axiom eq9 : (x7 + x9) / 2 = 8
axiom eq10 : (x9 + x1) / 2 = 10

-- The theorem to prove.
theorem find_x6 : x6 = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_x6_l2001_200109


namespace NUMINAMATH_GPT_shaded_region_perimeter_l2001_200136

theorem shaded_region_perimeter (r : ℝ) (θ : ℝ) (h₁ : r = 2) (h₂ : θ = 90) : 
  (2 * r + (2 * π * r * (1 - θ / 180))) = π + 4 := 
by sorry

end NUMINAMATH_GPT_shaded_region_perimeter_l2001_200136


namespace NUMINAMATH_GPT_strictly_increasing_interval_l2001_200188

def f (x : ℝ) : ℝ := -x^3 + 3*x + 2

theorem strictly_increasing_interval : { x : ℝ | -1 < x ∧ x < 1 } = { x : ℝ | -3 * (x + 1) * (x - 1) > 0 } :=
sorry

end NUMINAMATH_GPT_strictly_increasing_interval_l2001_200188


namespace NUMINAMATH_GPT_highest_qualification_number_possible_l2001_200119

theorem highest_qualification_number_possible (n : ℕ) (qualifies : ℕ → ℕ → Prop)
    (h512 : n = 512)
    (hqualifies : ∀ a b, qualifies a b ↔ (a < b ∧ b - a ≤ 2)): 
    ∃ k, k = 18 ∧ (∀ m, qualifies m k → m < k) :=
by
  sorry

end NUMINAMATH_GPT_highest_qualification_number_possible_l2001_200119


namespace NUMINAMATH_GPT_tree_graph_probability_127_l2001_200116

theorem tree_graph_probability_127 :
  let n := 5
  let p := 125
  let q := 1024
  q ^ (1/10) + p = 127 :=
by
  sorry

end NUMINAMATH_GPT_tree_graph_probability_127_l2001_200116


namespace NUMINAMATH_GPT_vertex_x_coord_l2001_200155

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

-- Conditions based on given points
def conditions (a b c : ℝ) : Prop :=
  quadratic a b c 2 = 4 ∧
  quadratic a b c 8 =4 ∧
  quadratic a b c 10 = 13

-- Statement to prove the x-coordinate of the vertex is 5
theorem vertex_x_coord (a b c : ℝ) (h : conditions a b c) : 
  (-(b) / (2 * a)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_vertex_x_coord_l2001_200155


namespace NUMINAMATH_GPT_solve_bx2_ax_1_lt_0_l2001_200111

noncomputable def quadratic_inequality_solution (a b : ℝ) (x : ℝ) : Prop :=
  x^2 + a * x + b > 0

theorem solve_bx2_ax_1_lt_0 (a b : ℝ) :
  (∀ x : ℝ, quadratic_inequality_solution a b x ↔ (x < -2 ∨ x > -1/2)) →
  (∀ x : ℝ, (x = -2 ∨ x = -1/2) → x^2 + a * x + b = 0) →
  (b * x^2 + a * x + 1 < 0) ↔ (-2 < x ∧ x < -1/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_bx2_ax_1_lt_0_l2001_200111


namespace NUMINAMATH_GPT_campaign_fliers_l2001_200122

theorem campaign_fliers (total_fliers : ℕ) (fraction_morning : ℚ) (fraction_afternoon : ℚ) 
  (remaining_fliers_after_morning : ℕ) (remaining_fliers_after_afternoon : ℕ) :
  total_fliers = 1000 → fraction_morning = 1/5 → fraction_afternoon = 1/4 → 
  remaining_fliers_after_morning = total_fliers - total_fliers * fraction_morning → 
  remaining_fliers_after_afternoon = remaining_fliers_after_morning - remaining_fliers_after_morning * fraction_afternoon → 
  remaining_fliers_after_afternoon = 600 := 
by
  sorry

end NUMINAMATH_GPT_campaign_fliers_l2001_200122


namespace NUMINAMATH_GPT_lcm_quadruples_count_l2001_200169

-- Define the problem conditions
variables (r s : ℕ) (hr : r > 0) (hs : s > 0)

-- Define the mathematical problem statement
theorem lcm_quadruples_count :
  ( ∀ (a b c d : ℕ),
    lcm (lcm a b) c = lcm (lcm a b) d ∧
    lcm (lcm a b) c = lcm (lcm a c) d ∧
    lcm (lcm a b) c = lcm (lcm b c) d ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    a = 3 ^ r * 7 ^ s ∧
    b = 3 ^ r * 7 ^ s ∧
    c = 3 ^ r * 7 ^ s ∧
    d = 3 ^ r * 7 ^ s 
  → ∃ n, n = (1 + 4 * r + 6 * r^2) * (1 + 4 * s + 6 * s^2)) :=
sorry

end NUMINAMATH_GPT_lcm_quadruples_count_l2001_200169


namespace NUMINAMATH_GPT_find_a13_l2001_200110

variable (a_n : ℕ → ℝ)
variable (d : ℝ)
variable (h_arith : ∀ n, a_n (n + 1) = a_n n + d)
variable (h_geo : a_n 9 ^ 2 = a_n 1 * a_n 5)
variable (h_sum : a_n 1 + 3 * a_n 5 + a_n 9 = 20)

theorem find_a13 (h_non_zero_d : d ≠ 0):
  a_n 13 = 28 :=
sorry

end NUMINAMATH_GPT_find_a13_l2001_200110


namespace NUMINAMATH_GPT_y_intercept_of_line_l2001_200166

def equation (x y : ℝ) : Prop := 3 * x - 5 * y = 10

theorem y_intercept_of_line : equation 0 (-2) :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l2001_200166


namespace NUMINAMATH_GPT_pencil_length_eq_eight_l2001_200191

theorem pencil_length_eq_eight (L : ℝ) 
  (h1 : (1/8) * L + (1/2) * ((7/8) * L) + (7/2) = L) : 
  L = 8 :=
by
  sorry

end NUMINAMATH_GPT_pencil_length_eq_eight_l2001_200191


namespace NUMINAMATH_GPT_sum_of_coefficients_l2001_200127

theorem sum_of_coefficients:
  (∀ x : ℝ, (2*x - 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) →
  a_1 + a_3 + a_5 = -364 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2001_200127


namespace NUMINAMATH_GPT_milk_transfer_proof_l2001_200128

theorem milk_transfer_proof :
  ∀ (A B C x : ℝ), 
  A = 1232 →
  B = A - 0.625 * A → 
  C = A - B → 
  B + x = C - x → 
  x = 154 :=
by
  intros A B C x hA hB hC hEqual
  sorry

end NUMINAMATH_GPT_milk_transfer_proof_l2001_200128


namespace NUMINAMATH_GPT_mixed_number_calculation_l2001_200101

/-
  We need to define a proof that shows:
  75 * (2 + 3/7 - 5 * (1/3)) / (3 + 1/5 + 2 + 1/6) = -208 + 7/9
-/
theorem mixed_number_calculation :
  75 * ((17 / 7) - (16 / 3)) / ((16 / 5) + (13 / 6)) = -208 + 7 / 9 := by
  sorry

end NUMINAMATH_GPT_mixed_number_calculation_l2001_200101


namespace NUMINAMATH_GPT_sum_of_ages_l2001_200133

/-
Juliet is 3 years older than her sister Maggie but 2 years younger than her elder brother Ralph.
If Juliet is 10 years old, the sum of Maggie's and Ralph's ages is 19 years.
-/
theorem sum_of_ages (juliet_age maggie_age ralph_age : ℕ) :
  juliet_age = 10 →
  juliet_age = maggie_age + 3 →
  ralph_age = juliet_age + 2 →
  maggie_age + ralph_age = 19 := by
  sorry

end NUMINAMATH_GPT_sum_of_ages_l2001_200133


namespace NUMINAMATH_GPT_thought_number_is_24_l2001_200146

variable (x : ℝ)

theorem thought_number_is_24 (h : x / 4 + 9 = 15) : x = 24 := by
  sorry

end NUMINAMATH_GPT_thought_number_is_24_l2001_200146


namespace NUMINAMATH_GPT_nonagon_diagonals_l2001_200172

-- Define the number of sides for a nonagon.
def n : ℕ := 9

-- Define the formula for the number of diagonals in a polygon.
def D (n : ℕ) : ℕ := n * (n - 3) / 2

-- State the theorem to prove that the number of diagonals in a nonagon is 27.
theorem nonagon_diagonals : D n = 27 := by
  sorry

end NUMINAMATH_GPT_nonagon_diagonals_l2001_200172


namespace NUMINAMATH_GPT_bus_travel_time_kimovsk_moscow_l2001_200154

noncomputable def travel_time_kimovsk_moscow (d1 d2 d3: ℝ) (max_speed: ℝ) (t_kt: ℝ) (t_nm: ℝ) : Prop :=
  35 ≤ d1 ∧ d1 ≤ 35 ∧
  60 ≤ d2 ∧ d2 ≤ 60 ∧
  200 ≤ d3 ∧ d3 ≤ 200 ∧
  max_speed <= 60 ∧
  2 ≤ t_kt ∧ t_kt ≤ 2 ∧
  5 ≤ t_nm ∧ t_nm ≤ 5 ∧
  (5 + 7/12 : ℝ) ≤ t_kt + t_nm ∧ t_kt + t_nm ≤ 6

theorem bus_travel_time_kimovsk_moscow
  (d1 d2 d3 : ℝ) (max_speed : ℝ) (t_kt : ℝ) (t_nm : ℝ) :
  travel_time_kimovsk_moscow d1 d2 d3 max_speed t_kt t_nm := 
by
  sorry

end NUMINAMATH_GPT_bus_travel_time_kimovsk_moscow_l2001_200154


namespace NUMINAMATH_GPT_smallest_m_l2001_200148

theorem smallest_m (m : ℕ) (h1 : 7 ≡ 2 [MOD 5]) : 
  (7^m ≡ m^7 [MOD 5]) ↔ (m = 7) :=
by sorry

end NUMINAMATH_GPT_smallest_m_l2001_200148


namespace NUMINAMATH_GPT_copies_per_person_l2001_200118

-- Definitions derived from the conditions
def pages_per_contract : ℕ := 20
def total_pages_copied : ℕ := 360
def number_of_people : ℕ := 9

-- Theorem stating the result based on the conditions
theorem copies_per_person : (total_pages_copied / pages_per_contract) / number_of_people = 2 := by
  sorry

end NUMINAMATH_GPT_copies_per_person_l2001_200118


namespace NUMINAMATH_GPT_socks_selection_l2001_200107

/-!
  # Socks Selection Problem
  Prove the total number of ways to choose a pair of socks of different colors
  given:
  1. there are 5 white socks,
  2. there are 4 brown socks,
  3. there are 3 blue socks,
  is equal to 47.
-/

theorem socks_selection : 
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  5 * 4 + 4 * 3 + 5 * 3 = 47 :=
by
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  sorry

end NUMINAMATH_GPT_socks_selection_l2001_200107


namespace NUMINAMATH_GPT_train_speed_approx_l2001_200123

noncomputable def distance_in_kilometers (d : ℝ) : ℝ :=
d / 1000

noncomputable def time_in_hours (t : ℝ) : ℝ :=
t / 3600

noncomputable def speed_in_kmh (d : ℝ) (t : ℝ) : ℝ :=
distance_in_kilometers d / time_in_hours t

theorem train_speed_approx (d t : ℝ) (h_d : d = 200) (h_t : t = 5.80598713393251) :
  abs (speed_in_kmh d t - 124.019) < 1e-3 :=
by
  rw [h_d, h_t]
  simp only [distance_in_kilometers, time_in_hours, speed_in_kmh]
  norm_num
  -- We're using norm_num to deal with numerical approximations and constants
  -- The actual calculations can be verified through manual checks or external tools but in Lean we skip this step.
  sorry

end NUMINAMATH_GPT_train_speed_approx_l2001_200123


namespace NUMINAMATH_GPT_line_intersects_hyperbola_l2001_200162

theorem line_intersects_hyperbola 
  (k : ℝ)
  (hyp : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) :
  -Real.sqrt 15 / 3 < k ∧ k < -1 := 
sorry


end NUMINAMATH_GPT_line_intersects_hyperbola_l2001_200162


namespace NUMINAMATH_GPT_max_metro_lines_l2001_200113

theorem max_metro_lines (lines : ℕ) 
  (stations_per_line : ℕ) 
  (max_interchange : ℕ) 
  (max_lines_per_interchange : ℕ) :
  (stations_per_line >= 4) → 
  (max_interchange <= 3) → 
  (max_lines_per_interchange <= 2) → 
  (∀ s_1 s_2, ∃ t_1 t_2, t_1 ≤ max_interchange ∧ t_2 ≤ max_interchange ∧
     (s_1 = t_1 ∨ s_2 = t_1 ∨ s_1 = t_2 ∨ s_2 = t_2)) → 
  lines ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_max_metro_lines_l2001_200113


namespace NUMINAMATH_GPT_find_b_l2001_200160

theorem find_b (a c S : ℝ) (h₁ : a = 5) (h₂ : c = 2) (h₃ : S = 4) : 
  b = Real.sqrt 17 ∨ b = Real.sqrt 41 := by
  sorry

end NUMINAMATH_GPT_find_b_l2001_200160


namespace NUMINAMATH_GPT_find_a_geometric_sequence_l2001_200135

theorem find_a_geometric_sequence (a : ℤ) (T : ℕ → ℤ) (b : ℕ → ℤ) :
  (∀ n, T n = 3 ^ n + a) →
  b 1 = T 1 →
  (∀ n, n ≥ 2 → b n = T n - T (n - 1)) →
  (∀ n, n ≥ 2 → (∃ r, r * b n = b (n - 1))) →
  a = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_a_geometric_sequence_l2001_200135


namespace NUMINAMATH_GPT_euclidean_steps_arbitrarily_large_l2001_200156

def fib : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fib (n + 1) + fib n

theorem euclidean_steps_arbitrarily_large (n : ℕ) (h : n ≥ 2) :
  gcd (fib (n+1)) (fib n) = gcd (fib 1) (fib 0) := 
sorry

end NUMINAMATH_GPT_euclidean_steps_arbitrarily_large_l2001_200156


namespace NUMINAMATH_GPT_F_2021_F_integer_F_divisibility_l2001_200151

/- Part 1 -/
def F (n : ℕ) : ℕ := 
  let a := n / 1000
  let b := (n % 1000) / 100
  let c := (n % 100) / 10
  let d := n % 10
  let n' := 1000 * c + 100 * d + 10 * a + b
  (n + n') / 101

theorem F_2021 : F 2021 = 41 :=
  sorry

/- Part 2 -/
theorem F_integer (a b c d : ℕ) (ha : 1 ≤ a) (hb : a ≤ 9) (hc : 0 ≤ b) (hd : b ≤ 9)
(hc' : 0 ≤ c) (hd' : c ≤ 9) (hc'' : 0 ≤ d) (hd'' : d ≤ 9) :
  let n := 1000 * a + 100 * b + 10 * c + d
  let n' := 1000 * c + 100 * d + 10 * a + b
  F n = (101 * (10 * a + b + 10 * c + d)) / 101 :=
  sorry

/- Part 3 -/
theorem F_divisibility (a b : ℕ) (ha : 1 ≤ a ∧ a ≤ 5) (hb : 5 ≤ b ∧ b ≤ 9) :
  let s := 3800 + 10 * a + b
  let t := 1000 * b + 100 * a + 13
  (3 * F t - F s) % 8 = 0 ↔ s = 3816 ∨ s = 3847 ∨ s = 3829 :=
  sorry

end NUMINAMATH_GPT_F_2021_F_integer_F_divisibility_l2001_200151


namespace NUMINAMATH_GPT_hyperbola_focus_l2001_200142

noncomputable def c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem hyperbola_focus
  (a b : ℝ)
  (hEq : ∀ x y : ℝ, ((x - 1)^2 / a^2) - ((y - 10)^2 / b^2) = 1):
  (1 + c 7 3, 10) = (1 + Real.sqrt (7^2 + 3^2), 10) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_focus_l2001_200142


namespace NUMINAMATH_GPT_percentage_increase_decrease_exceeds_original_l2001_200185

open Real

theorem percentage_increase_decrease_exceeds_original (p q M : ℝ) (hp : 0 < p) (hq1 : 0 < q) (hq2 : q < 100) (hM : 0 < M) :
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > (100 * q) / (100 - q)) :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_decrease_exceeds_original_l2001_200185


namespace NUMINAMATH_GPT_solve_for_x_l2001_200100

theorem solve_for_x (x : ℤ) (h : (-1) * 2 * x * 4 = 24) : x = -3 := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l2001_200100


namespace NUMINAMATH_GPT_scientific_notation_l2001_200130

def significant_digits : ℝ := 4.032
def exponent : ℤ := 11
def original_number : ℝ := 403200000000

theorem scientific_notation : original_number = significant_digits * 10 ^ exponent := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_l2001_200130


namespace NUMINAMATH_GPT_years_passed_l2001_200134

def initial_ages : List ℕ := [19, 34, 37, 42, 48]

def new_ages (x : ℕ) : List ℕ :=
  initial_ages.map (λ age => age + x)

-- Hypothesis: The new ages fit the following stem-and-leaf plot structure
def valid_stem_and_leaf (ages : List ℕ) : Bool :=
  ages = [25, 31, 34, 37, 43, 48]

theorem years_passed : ∃ x : ℕ, valid_stem_and_leaf (new_ages x) := by
  sorry

end NUMINAMATH_GPT_years_passed_l2001_200134


namespace NUMINAMATH_GPT_find_a_plus_b_l2001_200178

theorem find_a_plus_b (a b : ℝ) (h_sum : 2 * a = -6) (h_prod : a^2 - b = 1) : a + b = 5 :=
by {
  -- Proof would go here; we assume the theorem holds true.
  sorry
}

end NUMINAMATH_GPT_find_a_plus_b_l2001_200178


namespace NUMINAMATH_GPT_line_passes_fixed_point_l2001_200140

theorem line_passes_fixed_point (k b : ℝ) (h : -1 = (k + b) / 2) :
  ∃ (x y : ℝ), x = 1 ∧ y = -2 ∧ y = k * x + b :=
by
  sorry

end NUMINAMATH_GPT_line_passes_fixed_point_l2001_200140


namespace NUMINAMATH_GPT_garden_area_l2001_200103

-- Definitions for the conditions
def perimeter : ℕ := 36
def width : ℕ := 10

-- Define the length using the perimeter and width
def length : ℕ := (perimeter - 2 * width) / 2

-- Define the area using the length and width
def area : ℕ := length * width

-- The theorem to prove the area is 80 square feet given the conditions
theorem garden_area : area = 80 :=
by 
  -- Here we use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_garden_area_l2001_200103


namespace NUMINAMATH_GPT_amount_saved_per_person_l2001_200149

-- Definitions based on the conditions
def original_price := 60
def discounted_price := 48
def number_of_people := 3
def discount := original_price - discounted_price

-- Proving that each person paid 4 dollars less.
theorem amount_saved_per_person : discount / number_of_people = 4 :=
by
  sorry

end NUMINAMATH_GPT_amount_saved_per_person_l2001_200149


namespace NUMINAMATH_GPT_point_in_fourth_quadrant_l2001_200139

def point : ℝ × ℝ := (3, -2)

def is_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant : is_fourth_quadrant point :=
by
  sorry

end NUMINAMATH_GPT_point_in_fourth_quadrant_l2001_200139


namespace NUMINAMATH_GPT_parabola_through_points_with_h_l2001_200175

noncomputable def quadratic_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_through_points_with_h (
    a h k : ℝ) 
    (H0 : quadratic_parabola a h k 0 = 4)
    (H1 : quadratic_parabola a h k 6 = 5)
    (H2 : a < 0)
    (H3 : 0 < h)
    (H4 : h < 6) : 
    h = 4 := 
sorry

end NUMINAMATH_GPT_parabola_through_points_with_h_l2001_200175


namespace NUMINAMATH_GPT_fisherman_total_fish_l2001_200174

theorem fisherman_total_fish :
  let bass := 32
  let trout := bass / 4
  let blue_gill := 2 * bass
  bass + trout + blue_gill = 104 :=
by
  sorry

end NUMINAMATH_GPT_fisherman_total_fish_l2001_200174


namespace NUMINAMATH_GPT_actual_area_l2001_200179

open Real

theorem actual_area
  (scale : ℝ)
  (mapped_area_cm2 : ℝ)
  (actual_area_cm2 : ℝ)
  (actual_area_m2 : ℝ)
  (h_scale : scale = 1 / 50000)
  (h_mapped_area : mapped_area_cm2 = 100)
  (h_proportion : mapped_area_cm2 / actual_area_cm2 = scale ^ 2)
  : actual_area_m2 = 2.5 * 10^7 :=
by
  sorry

end NUMINAMATH_GPT_actual_area_l2001_200179


namespace NUMINAMATH_GPT_value_of_y_l2001_200125

theorem value_of_y (y : ℝ) (h : |y| = |y - 3|) : y = 3 / 2 :=
sorry

end NUMINAMATH_GPT_value_of_y_l2001_200125


namespace NUMINAMATH_GPT_smallest_possible_value_of_other_integer_l2001_200180

theorem smallest_possible_value_of_other_integer (x b : ℕ) (h_gcd_lcm : ∀ m n : ℕ, m = 36 → gcd m n = x + 5 → lcm m n = x * (x + 5)) : 
  b > 0 → ∃ b, b = 1 ∧ gcd 36 b = x + 5 ∧ lcm 36 b = x * (x + 5) := 
by {
   sorry 
}

end NUMINAMATH_GPT_smallest_possible_value_of_other_integer_l2001_200180


namespace NUMINAMATH_GPT_equal_expressions_l2001_200164

theorem equal_expressions : (-2)^3 = -(2^3) :=
by sorry

end NUMINAMATH_GPT_equal_expressions_l2001_200164


namespace NUMINAMATH_GPT_polygon_sides_l2001_200190

open Real

theorem polygon_sides (n : ℕ) : 
  (∀ (angle : ℝ), angle = 40 → n * angle = 360) → n = 9 := by
  intro h
  have h₁ := h 40 rfl
  sorry

end NUMINAMATH_GPT_polygon_sides_l2001_200190


namespace NUMINAMATH_GPT_fraction_zero_implies_x_is_minus_one_l2001_200129

variable (x : ℝ)

theorem fraction_zero_implies_x_is_minus_one (h : (x^2 - 1) / (1 - x) = 0) : x = -1 :=
sorry

end NUMINAMATH_GPT_fraction_zero_implies_x_is_minus_one_l2001_200129


namespace NUMINAMATH_GPT_handshakes_at_networking_event_l2001_200183

noncomputable def total_handshakes (n : ℕ) (exclude : ℕ) : ℕ :=
  (n * (n - 1 - exclude)) / 2

theorem handshakes_at_networking_event : total_handshakes 12 1 = 60 := by
  sorry

end NUMINAMATH_GPT_handshakes_at_networking_event_l2001_200183


namespace NUMINAMATH_GPT_solution_l2001_200104

variable (x y z : ℝ)
variable (hx : x > 0)
variable (hy : y > 0)
variable (hz : z > 0)

-- Condition 1: 20/x + 6/y = 1
axiom eq1 : 20 / x + 6 / y = 1

-- Condition 2: 4/x + 2/y = 2/9
axiom eq2 : 4 / x + 2 / y = 2 / 9

-- What we need to prove: 1/z = 1/x + 1/y
axiom eq3 : 1 / x + 1 / y = 1 / z

theorem solution : z = 14.4 := by
  -- Omitted proof, just the statement
  sorry

end NUMINAMATH_GPT_solution_l2001_200104


namespace NUMINAMATH_GPT_box_volume_l2001_200186

variable (l w h : ℝ)
variable (lw_eq : l * w = 30)
variable (wh_eq : w * h = 40)
variable (lh_eq : l * h = 12)

theorem box_volume : l * w * h = 120 := by
  sorry

end NUMINAMATH_GPT_box_volume_l2001_200186


namespace NUMINAMATH_GPT_find_original_b_l2001_200137

variable {a b c : ℝ}
variable (H_inv_prop : a * b = c) (H_a_increase : 1.20 * a * 80 = c)

theorem find_original_b : b = 96 :=
  by
  sorry

end NUMINAMATH_GPT_find_original_b_l2001_200137


namespace NUMINAMATH_GPT_divisor_of_5025_is_5_l2001_200106

/--
  Given an original number n which is 5026,
  and a resulting number after subtracting 1 from n,
  prove that the divisor of the resulting number is 5.
-/
theorem divisor_of_5025_is_5 (n : ℕ) (h₁ : n = 5026) (d : ℕ) (h₂ : (n - 1) % d = 0) : d = 5 :=
sorry

end NUMINAMATH_GPT_divisor_of_5025_is_5_l2001_200106


namespace NUMINAMATH_GPT_sequence_a10_l2001_200184

theorem sequence_a10 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n+1) - a n = 1 / (4 * ↑n^2 - 1)) :
  a 10 = 28 / 19 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a10_l2001_200184


namespace NUMINAMATH_GPT_min_xy_min_x_plus_y_l2001_200131

theorem min_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : xy ≥ 36 :=
sorry  

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 9 / y = 1) : x + y ≥ 16 :=
sorry

end NUMINAMATH_GPT_min_xy_min_x_plus_y_l2001_200131


namespace NUMINAMATH_GPT_pascal_fifth_number_l2001_200196

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end NUMINAMATH_GPT_pascal_fifth_number_l2001_200196


namespace NUMINAMATH_GPT_calculate_expression_l2001_200138

theorem calculate_expression (x : ℝ) (h₁ : x ≠ 5) (h₂ : x = 4) : (x^2 - 3 * x - 10) / (x - 5) = 6 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2001_200138


namespace NUMINAMATH_GPT_negation_proposition_l2001_200114

theorem negation_proposition (p : Prop) (h : ∀ x : ℝ, 2 * x^2 + 1 > 0) : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_proposition_l2001_200114


namespace NUMINAMATH_GPT_average_difference_l2001_200152

theorem average_difference :
  let avg1 := (10 + 30 + 50) / 3
  let avg2 := (20 + 40 + 6) / 3
  avg1 - avg2 = 8 := by
  sorry

end NUMINAMATH_GPT_average_difference_l2001_200152


namespace NUMINAMATH_GPT_jericho_altitude_300_l2001_200115

def jericho_altitude (below_sea_level : Int) : Prop :=
  below_sea_level = -300

theorem jericho_altitude_300 (below_sea_level : Int)
  (h1 : below_sea_level = -300) : jericho_altitude below_sea_level :=
by
  sorry

end NUMINAMATH_GPT_jericho_altitude_300_l2001_200115


namespace NUMINAMATH_GPT_equivalent_annual_rate_correct_l2001_200132

noncomputable def quarterly_rate (annual_rate : ℝ) : ℝ :=
  annual_rate / 4

noncomputable def effective_annual_rate (quarterly_rate : ℝ) : ℝ :=
  (1 + quarterly_rate / 100)^4

noncomputable def equivalent_annual_rate (annual_rate : ℝ) : ℝ :=
  (effective_annual_rate (quarterly_rate annual_rate) - 1) * 100

theorem equivalent_annual_rate_correct :
  equivalent_annual_rate 8 = 8.24 := 
by
  sorry

end NUMINAMATH_GPT_equivalent_annual_rate_correct_l2001_200132


namespace NUMINAMATH_GPT_simplify_expression_l2001_200194

variable (y : ℤ)

theorem simplify_expression : 5 * y + 7 * y - 3 * y = 9 * y := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2001_200194


namespace NUMINAMATH_GPT_woman_born_second_half_20th_century_l2001_200161

theorem woman_born_second_half_20th_century (x : ℕ) (hx : 45 < x ∧ x < 50) (h_year : x * x = 2025) :
  x * x - x = 1980 :=
by {
  -- Add the crux of the problem here.
  sorry
}

end NUMINAMATH_GPT_woman_born_second_half_20th_century_l2001_200161


namespace NUMINAMATH_GPT_mix_alcohol_solutions_l2001_200176

-- Definitions capturing the conditions from part (a)
def volume_solution_y : ℝ := 600
def percent_alcohol_x : ℝ := 0.1
def percent_alcohol_y : ℝ := 0.3
def desired_percent_alcohol : ℝ := 0.25

-- The resulting Lean statement to prove question == answer given conditions
theorem mix_alcohol_solutions (Vx : ℝ) (h : (percent_alcohol_x * Vx + percent_alcohol_y * volume_solution_y) / (Vx + volume_solution_y) = desired_percent_alcohol) : Vx = 200 :=
sorry

end NUMINAMATH_GPT_mix_alcohol_solutions_l2001_200176


namespace NUMINAMATH_GPT_citizen_income_l2001_200159

theorem citizen_income (tax_paid : ℝ) (base_income : ℝ) (base_rate excess_rate : ℝ) (income : ℝ) 
  (h1 : 0 < base_income) (h2 : base_rate * base_income = 4400) (h3 : tax_paid = 8000)
  (h4 : excess_rate = 0.20) (h5 : base_rate = 0.11)
  (h6 : tax_paid = base_rate * base_income + excess_rate * (income - base_income)) :
  income = 58000 :=
sorry

end NUMINAMATH_GPT_citizen_income_l2001_200159


namespace NUMINAMATH_GPT_average_speed_with_stoppages_l2001_200117

/--The average speed of the bus including stoppages is 20 km/hr, 
  given that the bus stops for 40 minutes per hour and 
  has an average speed of 60 km/hr excluding stoppages.--/
theorem average_speed_with_stoppages 
  (avg_speed_without_stoppages : ℝ)
  (stoppage_time_per_hour : ℕ) 
  (running_time_per_hour : ℕ) 
  (avg_speed_with_stoppages : ℝ) 
  (h1 : avg_speed_without_stoppages = 60) 
  (h2 : stoppage_time_per_hour = 40) 
  (h3 : running_time_per_hour = 20) 
  (h4 : running_time_per_hour + stoppage_time_per_hour = 60):
  avg_speed_with_stoppages = 20 := 
sorry

end NUMINAMATH_GPT_average_speed_with_stoppages_l2001_200117


namespace NUMINAMATH_GPT_ratio_of_smaller_to_bigger_l2001_200195

theorem ratio_of_smaller_to_bigger (S B : ℕ) (h_bigger: B = 104) (h_sum: S + B = 143) :
  S / B = 39 / 104 := sorry

end NUMINAMATH_GPT_ratio_of_smaller_to_bigger_l2001_200195


namespace NUMINAMATH_GPT_find_angle_A_l2001_200167

theorem find_angle_A (a b : ℝ) (B A : ℝ) (h1 : a = Real.sqrt 2) (h2 : b = 2) (h3 : B = Real.pi / 4) : A = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l2001_200167


namespace NUMINAMATH_GPT_correct_option_C_l2001_200145

variable (a : ℝ)

theorem correct_option_C : (a^2 * a = a^3) :=
by sorry

end NUMINAMATH_GPT_correct_option_C_l2001_200145


namespace NUMINAMATH_GPT_meals_for_children_l2001_200120

theorem meals_for_children (C : ℕ)
  (H1 : 70 * C = 70 * 45)
  (H2 : 70 * 45 = 2 * 45 * 35) :
  C = 90 :=
by
  sorry

end NUMINAMATH_GPT_meals_for_children_l2001_200120


namespace NUMINAMATH_GPT_pats_stick_covered_l2001_200197

/-
Assumptions:
1. Pat's stick is 30 inches long.
2. Jane's stick is 22 inches long.
3. Jane’s stick is two feet (24 inches) shorter than Sarah’s stick.
4. The portion of Pat's stick not covered in dirt is half as long as Sarah’s stick.

Prove that the length of Pat's stick covered in dirt is 7 inches.
-/

theorem pats_stick_covered  (pat_stick_len : ℕ) (jane_stick_len : ℕ) (jane_sarah_diff : ℕ) (pat_not_covered_by_dirt : ℕ) :
  pat_stick_len = 30 → jane_stick_len = 22 → jane_sarah_diff = 24 → pat_not_covered_by_dirt * 2 = jane_stick_len + jane_sarah_diff → 
    (pat_stick_len - pat_not_covered_by_dirt) = 7 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_pats_stick_covered_l2001_200197


namespace NUMINAMATH_GPT_tan_theta_eq_neg3_then_expr_eq_5_div_2_l2001_200171

theorem tan_theta_eq_neg3_then_expr_eq_5_div_2
  (θ : ℝ) (h : Real.tan θ = -3) :
  (Real.sin θ - 2 * Real.cos θ) / (Real.cos θ + Real.sin θ) = 5 / 2 := 
sorry

end NUMINAMATH_GPT_tan_theta_eq_neg3_then_expr_eq_5_div_2_l2001_200171
