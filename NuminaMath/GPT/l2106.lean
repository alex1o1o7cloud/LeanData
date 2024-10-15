import Mathlib

namespace NUMINAMATH_GPT_inequality_square_l2106_210614

theorem inequality_square (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a^2 > b^2 :=
sorry

end NUMINAMATH_GPT_inequality_square_l2106_210614


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_4_l2106_210651

noncomputable def f (x : ℝ) : ℝ := x^4 - 9 * x^3 + 21 * x^2 + x - 18

theorem remainder_when_divided_by_x_minus_4 : f 4 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_4_l2106_210651


namespace NUMINAMATH_GPT_purple_tile_cost_correct_l2106_210627

-- Definitions of given conditions
def turquoise_cost_per_tile : ℕ := 13
def wall1_area : ℕ := 5 * 8
def wall2_area : ℕ := 7 * 8
def total_area : ℕ := wall1_area + wall2_area
def tiles_per_square_foot : ℕ := 4
def total_tiles_needed : ℕ := total_area * tiles_per_square_foot
def turquoise_total_cost : ℕ := total_tiles_needed * turquoise_cost_per_tile
def savings : ℕ := 768
def purple_total_cost : ℕ := turquoise_total_cost - savings
def purple_cost_per_tile : ℕ := 11

-- Theorem stating the problem
theorem purple_tile_cost_correct :
  purple_total_cost / total_tiles_needed = purple_cost_per_tile :=
sorry

end NUMINAMATH_GPT_purple_tile_cost_correct_l2106_210627


namespace NUMINAMATH_GPT_median_computation_l2106_210673

noncomputable def length_of_median (A B C A1 P Q R : ℝ) : Prop :=
  let AB := 10
  let AC := 6
  let BC := Real.sqrt (AB^2 - AC^2)
  let A1C := 24 / 7
  let A1B := 32 / 7
  let QR := Real.sqrt (A1B^2 - A1C^2)
  let median_length := QR / 2
  median_length = 4 * Real.sqrt 7 / 7

theorem median_computation (A B C A1 P Q R : ℝ) :
  length_of_median A B C A1 P Q R := by
  sorry

end NUMINAMATH_GPT_median_computation_l2106_210673


namespace NUMINAMATH_GPT_largest_area_of_triangle_DEF_l2106_210617

noncomputable def maxAreaTriangleDEF : Real :=
  let DE := 16.0
  let EF_to_FD := 25.0 / 24.0
  let max_area := 446.25
  max_area

theorem largest_area_of_triangle_DEF :
  ∀ (DE : Real) (EF FD : Real),
    DE = 16 ∧ EF / FD = 25 / 24 → 
    (∃ (area : Real), area ≤ maxAreaTriangleDEF) :=
by 
  sorry

end NUMINAMATH_GPT_largest_area_of_triangle_DEF_l2106_210617


namespace NUMINAMATH_GPT_round_robin_chess_l2106_210628

/-- 
In a round-robin chess tournament, two boys and several girls participated. 
The boys together scored 8 points, while all the girls scored an equal number of points.
We are to prove that the number of girls could have participated in the tournament is 7 or 14,
given that a win is 1 point, a draw is 0.5 points, and a loss is 0 points.
-/
theorem round_robin_chess (n : ℕ) (x : ℚ) (h : 2 * n * x + 16 = n ^ 2 + 3 * n + 2) : n = 7 ∨ n = 14 :=
sorry

end NUMINAMATH_GPT_round_robin_chess_l2106_210628


namespace NUMINAMATH_GPT_tan_seven_pi_over_four_l2106_210636

theorem tan_seven_pi_over_four : Real.tan (7 * Real.pi / 4) = -1 :=
by sorry

end NUMINAMATH_GPT_tan_seven_pi_over_four_l2106_210636


namespace NUMINAMATH_GPT_triangle_inequality_difference_l2106_210667

theorem triangle_inequality_difference :
  ∀ (x : ℤ), (x + 8 > 3) → (x + 3 > 8) → (8 + 3 > x) →
  ( 10 - 6 = 4 ) :=
by sorry

end NUMINAMATH_GPT_triangle_inequality_difference_l2106_210667


namespace NUMINAMATH_GPT_average_bowling_score_l2106_210640

-- Definitions of the scores
def g : ℕ := 120
def m : ℕ := 113
def b : ℕ := 85

-- Theorem statement: The average score is 106
theorem average_bowling_score : (g + m + b) / 3 = 106 := by
  sorry

end NUMINAMATH_GPT_average_bowling_score_l2106_210640


namespace NUMINAMATH_GPT_triangle_cosine_l2106_210638

theorem triangle_cosine (LM : ℝ) (cos_N : ℝ) (LN : ℝ) (h1 : LM = 20) (h2 : cos_N = 3/5) :
  LM / LN = cos_N → LN = 100 / 3 :=
by
  intro h3
  sorry

end NUMINAMATH_GPT_triangle_cosine_l2106_210638


namespace NUMINAMATH_GPT_shopping_money_l2106_210641

theorem shopping_money (X : ℝ) (h : 0.70 * X = 840) : X = 1200 :=
sorry

end NUMINAMATH_GPT_shopping_money_l2106_210641


namespace NUMINAMATH_GPT_smallest_even_number_of_sum_1194_l2106_210634

-- Defining the given condition
def sum_of_three_consecutive_even_numbers (x : ℕ) : Prop :=
  x + (x + 2) + (x + 4) = 1194

-- Stating the theorem to prove the smallest even number
theorem smallest_even_number_of_sum_1194 :
  ∃ x : ℕ, sum_of_three_consecutive_even_numbers x ∧ x = 396 :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_number_of_sum_1194_l2106_210634


namespace NUMINAMATH_GPT_polynomial_divisibility_l2106_210683

-- Define the polynomial f(x)
def f (x : ℝ) (m : ℝ) : ℝ := 4 * x^3 - 8 * x^2 + m * x - 16

-- Prove that f(x) is divisible by x-2 if and only if m=8
theorem polynomial_divisibility (m : ℝ) :
  (∀ (x : ℝ), (x - 2) ∣ f x m) ↔ m = 8 := 
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l2106_210683


namespace NUMINAMATH_GPT_total_respondents_l2106_210626

theorem total_respondents (x_preference resp_y : ℕ) (h1 : x_preference = 360) (h2 : 9 * resp_y = x_preference) : 
  resp_y + x_preference = 400 :=
by 
  sorry

end NUMINAMATH_GPT_total_respondents_l2106_210626


namespace NUMINAMATH_GPT_alpha_plus_2beta_l2106_210664

noncomputable def sin_square (θ : ℝ) := (Real.sin θ)^2
noncomputable def sin_double (θ : ℝ) := Real.sin (2 * θ)

theorem alpha_plus_2beta (α β : ℝ) (hα : 0 < α ∧ α < Real.pi / 2) 
(hβ : 0 < β ∧ β < Real.pi / 2) 
(h1 : 3 * sin_square α + 2 * sin_square β = 1)
(h2 : 3 * sin_double α - 2 * sin_double β = 0) : 
α + 2 * β = 5 * Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_alpha_plus_2beta_l2106_210664


namespace NUMINAMATH_GPT_minimal_rope_cost_l2106_210671

theorem minimal_rope_cost :
  let pieces_needed := 10
  let length_per_piece := 6 -- inches
  let total_length_needed := pieces_needed * length_per_piece -- inches
  let one_foot_length := 12 -- inches
  let cost_six_foot_rope := 5 -- dollars
  let cost_one_foot_rope := 1.25 -- dollars
  let six_foot_length := 6 * one_foot_length -- inches
  let one_foot_total_cost := (total_length_needed / one_foot_length) * cost_one_foot_rope
  let six_foot_total_cost := cost_six_foot_rope
  total_length_needed <= six_foot_length ∧ six_foot_total_cost < one_foot_total_cost →
  six_foot_total_cost = 5 := sorry

end NUMINAMATH_GPT_minimal_rope_cost_l2106_210671


namespace NUMINAMATH_GPT_factorial_div_add_two_l2106_210609

def factorial (n : ℕ) : ℕ :=
match n with
| 0 => 1
| n + 1 => (n + 1) * factorial n

theorem factorial_div_add_two :
  (factorial 50) / (factorial 48) + 2 = 2452 :=
by
  sorry

end NUMINAMATH_GPT_factorial_div_add_two_l2106_210609


namespace NUMINAMATH_GPT_intersection_nonempty_implies_t_lt_1_l2106_210600

def M (x : ℝ) := x ≤ 1
def P (t : ℝ) (x : ℝ) := x > t

theorem intersection_nonempty_implies_t_lt_1 {t : ℝ} (h : ∃ x, M x ∧ P t x) : t < 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_nonempty_implies_t_lt_1_l2106_210600


namespace NUMINAMATH_GPT_paint_area_is_correct_l2106_210666

-- Define the dimensions of the wall, window, and door
def wall_height : ℕ := 10
def wall_length : ℕ := 15
def window_height : ℕ := 3
def window_length : ℕ := 5
def door_height : ℕ := 1
def door_length : ℕ := 7

-- Calculate area
def wall_area : ℕ := wall_height * wall_length
def window_area : ℕ := window_height * window_length
def door_area : ℕ := door_height * door_length

-- Calculate area to be painted
def area_to_be_painted : ℕ := wall_area - window_area - door_area

-- The theorem statement
theorem paint_area_is_correct : area_to_be_painted = 128 := 
by
  -- The proof would go here (omitted)
  sorry

end NUMINAMATH_GPT_paint_area_is_correct_l2106_210666


namespace NUMINAMATH_GPT_jose_peanuts_l2106_210601

def kenya_peanuts : Nat := 133
def difference_peanuts : Nat := 48

theorem jose_peanuts : (kenya_peanuts - difference_peanuts) = 85 := by
  sorry

end NUMINAMATH_GPT_jose_peanuts_l2106_210601


namespace NUMINAMATH_GPT_p_minus_q_eq_16_sqrt_2_l2106_210643

theorem p_minus_q_eq_16_sqrt_2 (p q : ℝ) (h_eq : ∀ x : ℝ, (x - 4) * (x + 4) = 28 * x - 84 → x = p ∨ x = q)
  (h_distinct : p ≠ q) (h_p_gt_q : p > q) : p - q = 16 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_p_minus_q_eq_16_sqrt_2_l2106_210643


namespace NUMINAMATH_GPT_lee_charged_per_action_figure_l2106_210668

theorem lee_charged_per_action_figure :
  ∀ (sneakers_cost savings action_figures leftovers price_per_fig),
    sneakers_cost = 90 →
    savings = 15 →
    action_figures = 10 →
    leftovers = 25 →
    price_per_fig = 10 →
    (savings + action_figures * price_per_fig) - sneakers_cost = leftovers → price_per_fig = 10 :=
by
  intros sneakers_cost savings action_figures leftovers price_per_fig
  intros h_sneakers_cost h_savings h_action_figures h_leftovers h_price_per_fig
  intros h_total
  sorry

end NUMINAMATH_GPT_lee_charged_per_action_figure_l2106_210668


namespace NUMINAMATH_GPT_sin_of_alpha_l2106_210632

theorem sin_of_alpha 
  (α : ℝ) 
  (h : Real.cos (α - Real.pi / 2) = 1 / 3) : 
  Real.sin α = 1 / 3 := 
by 
  sorry

end NUMINAMATH_GPT_sin_of_alpha_l2106_210632


namespace NUMINAMATH_GPT_range_of_a_max_area_of_triangle_l2106_210670

variable (p a : ℝ) (h : p > 0)

def parabola_eq (x y : ℝ) := y ^ 2 = 2 * p * x
def line_eq (x y : ℝ) := y = x - a
def intersects_parabola (A B : ℝ × ℝ) := parabola_eq p A.fst A.snd ∧ line_eq a A.fst A.snd ∧ parabola_eq p B.fst B.snd ∧ line_eq a B.fst B.snd
def ab_length_le_2p (A B : ℝ × ℝ) := (Real.sqrt ((A.fst - B.fst)^2 + (A.snd - B.snd)^2) ≤ 2 * p)

theorem range_of_a
  (A B : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B) :
  - p / 2 < a ∧ a ≤ - p / 4 := sorry

theorem max_area_of_triangle
  (A B : ℝ × ℝ) (N : ℝ × ℝ)
  (h_intersects : intersects_parabola a p A B)
  (h_ab_length : ab_length_le_2p p A B)
  (h_N : N.snd = 0) :
  ∃ (S : ℝ), S = Real.sqrt 2 * p^2 := sorry

end NUMINAMATH_GPT_range_of_a_max_area_of_triangle_l2106_210670


namespace NUMINAMATH_GPT_unpainted_cubes_l2106_210688

theorem unpainted_cubes (n : ℕ) (cubes_per_face : ℕ) (faces : ℕ) (total_cubes : ℕ) (painted_cubes : ℕ) :
  n = 6 → cubes_per_face = 4 → faces = 6 → total_cubes = 216 → painted_cubes = 24 → 
  total_cubes - painted_cubes = 192 := by
  intros
  sorry

end NUMINAMATH_GPT_unpainted_cubes_l2106_210688


namespace NUMINAMATH_GPT_range_of_2a_plus_3b_l2106_210637

theorem range_of_2a_plus_3b (a b : ℝ) (h1 : -1 < a + b) (h2 : a + b < 3) (h3 : 2 < a - b) (h4 : a - b < 4) :
  -9 / 2 < 2 * a + 3 * b ∧ 2 * a + 3 * b < 13 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_2a_plus_3b_l2106_210637


namespace NUMINAMATH_GPT_calculate_expression_l2106_210686

theorem calculate_expression (h₁ : x = 7 / 8) (h₂ : y = 5 / 6) (hx : x ≠ 0) (hy : y ≠ 0) :
  (4 * x - 6 * y) / (60 * x * y) = -6 / 175 := 
sorry

end NUMINAMATH_GPT_calculate_expression_l2106_210686


namespace NUMINAMATH_GPT_metal_relative_atomic_mass_is_24_l2106_210657

noncomputable def relative_atomic_mass (metal_mass : ℝ) (hcl_mass_percent : ℝ) (hcl_total_mass : ℝ) (mol_mass_hcl : ℝ) : ℝ :=
  let moles_hcl := (hcl_total_mass * hcl_mass_percent / 100) / mol_mass_hcl
  let maximum_molar_mass := metal_mass / (moles_hcl / 2)
  let minimum_molar_mass := metal_mass / (moles_hcl / 2)
  if 20 < maximum_molar_mass ∧ maximum_molar_mass < 28 then
    24
  else
    0

theorem metal_relative_atomic_mass_is_24
  (metal_mass_1 : ℝ)
  (metal_mass_2 : ℝ)
  (hcl_mass_percent : ℝ)
  (hcl_total_mass : ℝ)
  (mol_mass_hcl : ℝ)
  (moles_used_1 : ℝ)
  (moles_used_2 : ℝ)
  (excess : Bool)
  (complete : Bool) :
  relative_atomic_mass 3.5 18.25 50 36.5 = 24 :=
by
  sorry

end NUMINAMATH_GPT_metal_relative_atomic_mass_is_24_l2106_210657


namespace NUMINAMATH_GPT_find_t_l2106_210621

variable (a b c : ℝ × ℝ)
variable (t : ℝ)

-- Definitions based on given conditions
def vec_a : ℝ × ℝ := (3, 1)
def vec_b : ℝ × ℝ := (1, 3)
def vec_c (t : ℝ) : ℝ × ℝ := (t, 2)

-- Dot product definition
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Condition that (vec_a - vec_c) is perpendicular to vec_b
def perpendicular_condition (t : ℝ) : Prop :=
  dot_product (vec_a - vec_c t) vec_b = 0

-- Proof statement
theorem find_t : ∃ t : ℝ, perpendicular_condition t ∧ t = 0 := 
by
  sorry

end NUMINAMATH_GPT_find_t_l2106_210621


namespace NUMINAMATH_GPT_triangle_DEF_angle_l2106_210675

noncomputable def one_angle_of_triangle_DEF (x : ℝ) : ℝ :=
  let arc_DE := 2 * x + 40
  let arc_EF := 3 * x + 50
  let arc_FD := 4 * x - 30
  if (arc_DE + arc_EF + arc_FD = 360)
  then (1 / 2) * arc_EF
  else 0

theorem triangle_DEF_angle (x : ℝ) (h : 2 * x + 40 + 3 * x + 50 + 4 * x - 30 = 360) :
  one_angle_of_triangle_DEF x = 75 :=
by sorry

end NUMINAMATH_GPT_triangle_DEF_angle_l2106_210675


namespace NUMINAMATH_GPT_probability_correct_l2106_210660

-- Define the set of segment lengths
def segment_lengths : List ℕ := [1, 3, 5, 7, 9]

-- Define the triangle inequality condition
def forms_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Calculate the number of favorable outcomes, i.e., sets that can form a triangle
def favorable_sets : List (ℕ × ℕ × ℕ) :=
  [(3, 5, 7), (3, 7, 9), (5, 7, 9)]

-- Define the total number of ways to select three segments out of five
def total_combinations : ℕ :=
  10

-- Define the number of favorable sets
def number_of_favorable_sets : ℕ :=
  favorable_sets.length

-- Calculate the probability of selecting three segments that form a triangle
def probability_of_triangle : ℚ :=
  number_of_favorable_sets / total_combinations

-- The theorem to prove
theorem probability_correct : probability_of_triangle = 3 / 10 :=
  by {
    -- Placeholder for the proof
    sorry
  }

end NUMINAMATH_GPT_probability_correct_l2106_210660


namespace NUMINAMATH_GPT_parallel_lines_solution_l2106_210639

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, (1 + a) * x + y + 1 = 0 → 2 * x + a * y + 2 = 0 → (a = 1 ∨ a = -2)) :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_solution_l2106_210639


namespace NUMINAMATH_GPT_car_speed_is_80_l2106_210620

theorem car_speed_is_80 
  (d : ℝ) (t_delay : ℝ) (v_train_factor : ℝ)
  (t_car t_train : ℝ) (v : ℝ) :
  ((d = 75) ∧ (t_delay = 12.5 / 60) ∧ (v_train_factor = 1.5) ∧ 
   (d = v * t_car) ∧ (d = v_train_factor * v * (t_car - t_delay))) →
  v = 80 := 
sorry

end NUMINAMATH_GPT_car_speed_is_80_l2106_210620


namespace NUMINAMATH_GPT_jessica_cut_r_l2106_210692

variable (r_i r_t r_c : ℕ)

theorem jessica_cut_r : r_i = 7 → r_g = 59 → r_t = 20 → r_c = r_t - r_i → r_c = 13 :=
by
  intros h_i h_g h_t h_c
  have h1 : r_i = 7 := h_i
  have h2 : r_t = 20 := h_t
  have h3 : r_c = r_t - r_i := h_c
  have h_correct : r_c = 13
  · sorry
  exact h_correct

end NUMINAMATH_GPT_jessica_cut_r_l2106_210692


namespace NUMINAMATH_GPT_determine_y_l2106_210606

theorem determine_y (x y : ℝ) (h₁ : x^2 = y - 7) (h₂ : x = 7) : y = 56 :=
sorry

end NUMINAMATH_GPT_determine_y_l2106_210606


namespace NUMINAMATH_GPT_traffic_accident_emergency_number_l2106_210624

theorem traffic_accident_emergency_number (A B C D : ℕ) (h1 : A = 122) (h2 : B = 110) (h3 : C = 120) (h4 : D = 114) : 
  A = 122 := 
by
  exact h1

end NUMINAMATH_GPT_traffic_accident_emergency_number_l2106_210624


namespace NUMINAMATH_GPT_fractional_units_l2106_210618

-- Define the mixed number and the smallest composite number
def mixed_number := 3 + 2/7
def smallest_composite := 4

-- To_struct fractional units of 3 2/7
theorem fractional_units (u : ℚ) (n : ℕ) (m : ℕ):
  u = 1/7 ∧ n = 23 ∧ m = 5 :=
by
  have h1 : u = 1 / 7 := sorry
  have h2 : mixed_number = 23 * u := sorry
  have h3 : smallest_composite - mixed_number = 5 * u := sorry
  have h4 : n = 23 := sorry
  have h5 : m = 5 := sorry
  exact ⟨h1, h4, h5⟩

end NUMINAMATH_GPT_fractional_units_l2106_210618


namespace NUMINAMATH_GPT_multiply_polynomials_l2106_210630

theorem multiply_polynomials (x y : ℝ) : 
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by 
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l2106_210630


namespace NUMINAMATH_GPT_sum_of_fractions_l2106_210644

theorem sum_of_fractions :
  (2 / 8) + (4 / 8) + (6 / 8) + (8 / 8) + (10 / 8) + 
  (12 / 8) + (14 / 8) + (16 / 8) + (18 / 8) + (20 / 8) = 13.75 :=
by sorry

end NUMINAMATH_GPT_sum_of_fractions_l2106_210644


namespace NUMINAMATH_GPT_value_of_u_when_m_is_3_l2106_210674

theorem value_of_u_when_m_is_3 :
  ∀ (u t m : ℕ), (t = 3^m + m) → (u = 4^t - 3 * t) → m = 3 → u = 4^30 - 90 :=
by
  intros u t m ht hu hm
  sorry

end NUMINAMATH_GPT_value_of_u_when_m_is_3_l2106_210674


namespace NUMINAMATH_GPT_age_of_teacher_l2106_210615

theorem age_of_teacher
    (n_students : ℕ)
    (avg_age_students : ℕ)
    (new_avg_age : ℕ)
    (n_total : ℕ)
    (H1 : n_students = 22)
    (H2 : avg_age_students = 21)
    (H3 : new_avg_age = avg_age_students + 1)
    (H4 : n_total = n_students + 1) :
    ((new_avg_age * n_total) - (avg_age_students * n_students) = 44) :=
by
    sorry

end NUMINAMATH_GPT_age_of_teacher_l2106_210615


namespace NUMINAMATH_GPT_cubic_identity_l2106_210633

theorem cubic_identity (a b c : ℝ) 
  (h1 : a + b + c = 12)
  (h2 : ab + ac + bc = 30)
  : a^3 + b^3 + c^3 - 3 * a * b * c = 648 := by
  sorry

end NUMINAMATH_GPT_cubic_identity_l2106_210633


namespace NUMINAMATH_GPT_largest_k_for_sum_of_integers_l2106_210676

theorem largest_k_for_sum_of_integers (k : ℕ) (n : ℕ) (h1 : 3^12 = k * n + k * (k + 1) / 2) 
  (h2 : k ∣ 2 * 3^12) (h3 : k < 1031) : k ≤ 486 :=
by 
  sorry -- The proof is skipped here, only the statement is required 

end NUMINAMATH_GPT_largest_k_for_sum_of_integers_l2106_210676


namespace NUMINAMATH_GPT_remainder_when_divided_by_20_l2106_210635

theorem remainder_when_divided_by_20 
  (n r : ℤ) 
  (k : ℤ)
  (h1 : n % 20 = r) 
  (h2 : 2 * n % 10 = 2)
  (h3 : 0 ≤ r ∧ r < 20)
  : r = 1 := 
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_20_l2106_210635


namespace NUMINAMATH_GPT_negation_of_proposition_l2106_210680

variable (a b : ℝ)

theorem negation_of_proposition :
  (¬ (a * b = 0 → a = 0 ∨ b = 0)) ↔ (a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l2106_210680


namespace NUMINAMATH_GPT_tiles_per_row_24_l2106_210629

noncomputable def num_tiles_per_row (area : ℝ) (tile_size : ℝ) : ℝ :=
  let side_length_ft := Real.sqrt area
  let side_length_in := side_length_ft * 12
  side_length_in / tile_size

theorem tiles_per_row_24 :
  num_tiles_per_row 324 9 = 24 :=
by
  sorry

end NUMINAMATH_GPT_tiles_per_row_24_l2106_210629


namespace NUMINAMATH_GPT_solve_equation_nat_numbers_l2106_210656

theorem solve_equation_nat_numbers (a b : ℕ) (h : (a, b) = (11, 170) ∨ (a, b) = (22, 158) ∨ (a, b) = (33, 146) ∨
                                    (a, b) = (44, 134) ∨ (a, b) = (55, 122) ∨ (a, b) = (66, 110) ∨
                                    (a, b) = (77, 98) ∨ (a, b) = (88, 86) ∨ (a, b) = (99, 74) ∨
                                    (a, b) = (110, 62) ∨ (a, b) = (121, 50) ∨ (a, b) = (132, 38) ∨
                                    (a, b) = (143, 26) ∨ (a, b) = (154, 14) ∨ (a, b) = (165, 2)) :
  12 * a + 11 * b = 2002 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_nat_numbers_l2106_210656


namespace NUMINAMATH_GPT_tangent_lines_create_regions_l2106_210663

theorem tangent_lines_create_regions (n : ℕ) (h : n = 26) : ∃ k, k = 68 :=
by
  have h1 : ∃ k, k = 68 := ⟨68, rfl⟩
  exact h1

end NUMINAMATH_GPT_tangent_lines_create_regions_l2106_210663


namespace NUMINAMATH_GPT_max_regions_7_dots_l2106_210665

-- Definitions based on conditions provided.
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def R (n : ℕ) : ℕ := 1 + binom n 2 + binom n 4

-- The goal is to state the proposition that the maximum number of regions created by joining 7 dots on a circle is 57.
theorem max_regions_7_dots : R 7 = 57 :=
by
  -- The proof is to be filled in here
  sorry

end NUMINAMATH_GPT_max_regions_7_dots_l2106_210665


namespace NUMINAMATH_GPT_interest_rate_l2106_210616

theorem interest_rate (P : ℝ) (t : ℝ) (d : ℝ) (r : ℝ) : 
  P = 8000.000000000171 → t = 2 → d = 20 →
  (P * (1 + r/100)^2 - P - (P * r * t / 100) = d) → r = 5 :=
by
  intros hP ht hd heq
  sorry

end NUMINAMATH_GPT_interest_rate_l2106_210616


namespace NUMINAMATH_GPT_point_P_coordinates_l2106_210691

theorem point_P_coordinates :
  ∃ (x y : ℝ), (y = (x^3 - 10 * x + 3)) ∧ (x < 0) ∧ (3 * x^2 - 10 = 2) ∧ (x = -2 ∧ y = 15) := by
sorry

end NUMINAMATH_GPT_point_P_coordinates_l2106_210691


namespace NUMINAMATH_GPT_gwen_spent_money_l2106_210611

theorem gwen_spent_money (initial : ℕ) (remaining : ℕ) (spent : ℕ) 
  (h_initial : initial = 7) 
  (h_remaining : remaining = 5) 
  (h_spent : spent = initial - remaining) : 
  spent = 2 := 
sorry

end NUMINAMATH_GPT_gwen_spent_money_l2106_210611


namespace NUMINAMATH_GPT_woman_speed_still_water_l2106_210681

theorem woman_speed_still_water (v_w v_c : ℝ) 
    (h1 : 120 = (v_w + v_c) * 10)
    (h2 : 24 = (v_w - v_c) * 14) : 
    v_w = 48 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_woman_speed_still_water_l2106_210681


namespace NUMINAMATH_GPT_initial_workers_count_l2106_210649

theorem initial_workers_count (W : ℕ) 
  (h1 : W * 30 = W * 30) 
  (h2 : W * 15 = (W - 5) * 20)
  (h3 : W > 5) 
  : W = 20 :=
by {
  sorry
}

end NUMINAMATH_GPT_initial_workers_count_l2106_210649


namespace NUMINAMATH_GPT_group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l2106_210610

def mats_weaved (weavers mats days : ℕ) : ℕ :=
  (mats / days) * weavers

theorem group_a_mats_in_12_days (mats_req : ℕ) :
  let weavers := 4
  let mats_per_period := 4
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_b_mats_in_12_days (mats_req : ℕ) :
  let weavers := 6
  let mats_per_period := 9
  let period_days := 3
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

theorem group_c_mats_in_12_days (mats_req : ℕ) :
  let weavers := 8
  let mats_per_period := 16
  let period_days := 4
  let target_days := 12
  mats_req = (mats_weaved weavers mats_per_period period_days) * (target_days / period_days) :=
sorry

end NUMINAMATH_GPT_group_a_mats_in_12_days_group_b_mats_in_12_days_group_c_mats_in_12_days_l2106_210610


namespace NUMINAMATH_GPT_numbers_left_on_blackboard_l2106_210625

theorem numbers_left_on_blackboard (n11 n12 n13 n14 n15 : ℕ)
    (h_n11 : n11 = 11) (h_n12 : n12 = 12) (h_n13 : n13 = 13) (h_n14 : n14 = 14) (h_n15 : n15 = 15)
    (total_numbers : n11 + n12 + n13 + n14 + n15 = 65) :
  ∃ (remaining1 remaining2 : ℕ), remaining1 = 12 ∧ remaining2 = 14 := 
sorry

end NUMINAMATH_GPT_numbers_left_on_blackboard_l2106_210625


namespace NUMINAMATH_GPT_chimney_problem_l2106_210687

variable (x : ℕ) -- number of bricks in the chimney
variable (t : ℕ)
variables (brenda_hours brandon_hours : ℕ)

def brenda_rate := x / brenda_hours
def brandon_rate := x / brandon_hours
def combined_rate := (brenda_rate + brandon_rate - 15) * t

theorem chimney_problem (h1 : brenda_hours = 9)
    (h2 : brandon_hours = 12)
    (h3 : t = 6)
    (h4 : combined_rate = x) : x = 540 := sorry

end NUMINAMATH_GPT_chimney_problem_l2106_210687


namespace NUMINAMATH_GPT_age_difference_l2106_210694

variable {A B C : ℕ}

-- Definition of conditions
def condition1 (A B C : ℕ) : Prop := A + B > B + C
def condition2 (A C : ℕ) : Prop := C = A - 16

-- The theorem stating the math problem
theorem age_difference (h1 : condition1 A B C) (h2 : condition2 A C) :
  (A + B) - (B + C) = 16 := by
  sorry

end NUMINAMATH_GPT_age_difference_l2106_210694


namespace NUMINAMATH_GPT_number_of_racks_l2106_210679

theorem number_of_racks (cds_per_rack total_cds : ℕ) (h1 : cds_per_rack = 8) (h2 : total_cds = 32) :
  total_cds / cds_per_rack = 4 :=
by
  -- actual proof goes here
  sorry

end NUMINAMATH_GPT_number_of_racks_l2106_210679


namespace NUMINAMATH_GPT_sqrt_224_between_14_and_15_l2106_210693

theorem sqrt_224_between_14_and_15 : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end NUMINAMATH_GPT_sqrt_224_between_14_and_15_l2106_210693


namespace NUMINAMATH_GPT_december_19th_day_l2106_210652

theorem december_19th_day (december_has_31_days : true)
  (december_1st_is_monday : true)
  (day_of_week : ℕ → ℕ) :
  day_of_week 19 = 5 :=
sorry

end NUMINAMATH_GPT_december_19th_day_l2106_210652


namespace NUMINAMATH_GPT_average_speed_round_trip_l2106_210696

noncomputable def distance_AB : ℝ := 120
noncomputable def speed_AB : ℝ := 30
noncomputable def speed_BA : ℝ := 40

theorem average_speed_round_trip :
  (2 * distance_AB * speed_AB * speed_BA) / (distance_AB * (speed_AB + speed_BA)) = 34 := 
  by 
    sorry

end NUMINAMATH_GPT_average_speed_round_trip_l2106_210696


namespace NUMINAMATH_GPT_ratio_frogs_to_dogs_l2106_210653

variable (D C F : ℕ)

-- Define the conditions as given in the problem statement
def cats_eq_dogs_implied : Prop := C = Nat.div (4 * D) 5
def frogs : Prop := F = 160
def total_animals : Prop := D + C + F = 304

-- Define the statement to be proved
theorem ratio_frogs_to_dogs (h1 : cats_eq_dogs_implied D C) (h2 : frogs F) (h3 : total_animals D C F) : F / D = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_frogs_to_dogs_l2106_210653


namespace NUMINAMATH_GPT_andrea_needs_to_buy_sod_squares_l2106_210698

theorem andrea_needs_to_buy_sod_squares :
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  1500 = total_area / area_of_sod_square :=
by
  let area_section1 := 30 * 40
  let area_section2 := 60 * 80
  let total_area := area_section1 + area_section2
  let area_of_sod_square := 2 * 2
  sorry

end NUMINAMATH_GPT_andrea_needs_to_buy_sod_squares_l2106_210698


namespace NUMINAMATH_GPT_oblique_projection_correctness_l2106_210631

structure ProjectionConditions where
  intuitive_diagram_of_triangle_is_triangle : Prop
  intuitive_diagram_of_parallelogram_is_parallelogram : Prop

theorem oblique_projection_correctness (c : ProjectionConditions)
  (h1 : c.intuitive_diagram_of_triangle_is_triangle)
  (h2 : c.intuitive_diagram_of_parallelogram_is_parallelogram) :
  c.intuitive_diagram_of_triangle_is_triangle ∧ c.intuitive_diagram_of_parallelogram_is_parallelogram :=
by
  sorry

end NUMINAMATH_GPT_oblique_projection_correctness_l2106_210631


namespace NUMINAMATH_GPT_calc_fraction_l2106_210654
-- Import necessary libraries

-- Define the necessary fractions and the given expression
def expr := (5 / 6) * (1 / (7 / 8 - 3 / 4))

-- State the theorem
theorem calc_fraction : expr = 20 / 3 := 
by
  sorry

end NUMINAMATH_GPT_calc_fraction_l2106_210654


namespace NUMINAMATH_GPT_basic_spatial_data_source_l2106_210647

def source_of_basic_spatial_data (s : String) : Prop :=
  s = "Detailed data provided by high-resolution satellite remote sensing technology" ∨
  s = "Data from various databases provided by high-speed networks" ∨
  s = "Various data collected and organized through the information highway" ∨
  s = "Various spatial exchange data provided by GIS"

theorem basic_spatial_data_source :
  source_of_basic_spatial_data "Data from various databases provided by high-speed networks" :=
sorry

end NUMINAMATH_GPT_basic_spatial_data_source_l2106_210647


namespace NUMINAMATH_GPT_chloe_profit_l2106_210689

theorem chloe_profit 
  (cost_per_dozen : ℕ)
  (selling_price_per_half_dozen : ℕ)
  (dozens_sold : ℕ)
  (h1 : cost_per_dozen = 50)
  (h2 : selling_price_per_half_dozen = 30)
  (h3 : dozens_sold = 50) : 
  (selling_price_per_half_dozen - cost_per_dozen / 2) * (dozens_sold * 2) = 500 :=
by 
  sorry

end NUMINAMATH_GPT_chloe_profit_l2106_210689


namespace NUMINAMATH_GPT_closest_time_to_1600_mirror_l2106_210699

noncomputable def clock_in_mirror_time (hour_hand_minute: ℕ) (minute_hand_minute: ℕ) : (ℕ × ℕ) :=
  let hour_in_mirror := (12 - hour_hand_minute) % 12
  let minute_in_mirror := minute_hand_minute
  (hour_in_mirror, minute_in_mirror)

theorem closest_time_to_1600_mirror (A B C D : (ℕ × ℕ)) :
  clock_in_mirror_time 4 0 = D → D = (8, 0) :=
by
  -- Introduction of hypothesis that clock closest to 16:00 (4:00) is represented by D
  intro h
  -- State the conclusion based on the given hypothesis
  sorry

end NUMINAMATH_GPT_closest_time_to_1600_mirror_l2106_210699


namespace NUMINAMATH_GPT_wall_print_costs_are_15_l2106_210608

-- Define the cost of curtains, installation, total cost, and number of wall prints.
variable (cost_curtain : ℕ := 30)
variable (num_curtains : ℕ := 2)
variable (cost_installation : ℕ := 50)
variable (num_wall_prints : ℕ := 9)
variable (total_cost : ℕ := 245)

-- Define the total cost of curtains
def total_cost_curtains : ℕ := num_curtains * cost_curtain

-- Define the total fixed costs
def total_fixed_costs : ℕ := total_cost_curtains + cost_installation

-- Define the total cost of wall prints
def total_cost_wall_prints : ℕ := total_cost - total_fixed_costs

-- Define the cost per wall print
def cost_per_wall_print : ℕ := total_cost_wall_prints / num_wall_prints

-- Prove the cost per wall print is $15.00
theorem wall_print_costs_are_15 : cost_per_wall_print = 15 := by
  -- This is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_wall_print_costs_are_15_l2106_210608


namespace NUMINAMATH_GPT_probability_of_both_selected_l2106_210672

theorem probability_of_both_selected :
  let pX := 1 / 5
  let pY := 2 / 7
  (pX * pY) = 2 / 35 :=
by
  let pX := 1 / 5
  let pY := 2 / 7
  show (pX * pY) = 2 / 35
  sorry

end NUMINAMATH_GPT_probability_of_both_selected_l2106_210672


namespace NUMINAMATH_GPT_earthquake_energy_multiple_l2106_210645

theorem earthquake_energy_multiple (E : ℕ → ℝ) (n9 n7 : ℕ)
  (h1 : E n9 = 10 ^ n9) 
  (h2 : E n7 = 10 ^ n7) 
  (hn9 : n9 = 9) 
  (hn7 : n7 = 7) : 
  E n9 / E n7 = 100 := 
by 
  sorry

end NUMINAMATH_GPT_earthquake_energy_multiple_l2106_210645


namespace NUMINAMATH_GPT_find_h_l2106_210642

noncomputable def y1 (x h j : ℝ) := 4 * (x - h) ^ 2 + j
noncomputable def y2 (x h k : ℝ) := 3 * (x - h) ^ 2 + k

theorem find_h (h j k : ℝ)
  (C1 : y1 0 h j = 2024)
  (C2 : y2 0 h k = 2025)
  (H1 : y1 x h j = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 506)
  (H2 : y2 x h k = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 675) :
  h = 22.5 :=
sorry

end NUMINAMATH_GPT_find_h_l2106_210642


namespace NUMINAMATH_GPT_race_distance_l2106_210612

theorem race_distance {a b c : ℝ} (h1 : b = 0.9 * a) (h2 : c = 0.95 * b) :
  let andrei_distance := 1000
  let boris_distance := andrei_distance - 100
  let valentin_distance := boris_distance - 50
  let valentin_actual_distance := (c / a) * andrei_distance
  andrei_distance - valentin_actual_distance = 145 :=
by
  sorry

end NUMINAMATH_GPT_race_distance_l2106_210612


namespace NUMINAMATH_GPT_chocolate_squares_remaining_l2106_210669

theorem chocolate_squares_remaining (m : ℕ) : m * 6 - 21 = 45 :=
by
  sorry

end NUMINAMATH_GPT_chocolate_squares_remaining_l2106_210669


namespace NUMINAMATH_GPT_vector_scalar_sub_l2106_210655

def a : ℝ × ℝ := (3, -9)
def b : ℝ × ℝ := (2, -8)
def scalar1 : ℝ := 4
def scalar2 : ℝ := 3

theorem vector_scalar_sub:
  scalar1 • a - scalar2 • b = (6, -12) := by
  sorry

end NUMINAMATH_GPT_vector_scalar_sub_l2106_210655


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l2106_210619

-- Problem 1: f(x-2) = 3x - 5 implies f(x) = 3x + 1
def problem1 (x : ℝ) (f : ℝ → ℝ) : Prop := 
  ∀ x : ℝ, f (x - 2) = 3 * x - 5 → f x = 3 * x + 1

-- Problem 2: Quadratic function satisfying specific conditions
def is_quadratic (f : ℝ → ℝ) : Prop := 
  ∃ a b c : ℝ, ∀ x : ℝ, f x = a*x^2 + b*x + c

def problem2 (f : ℝ → ℝ) : Prop :=
  is_quadratic f ∧
  (f 0 = 4) ∧
  (∀ x : ℝ, f (3 - x) = f x) ∧
  (∀ x : ℝ, f x ≥ 7/4) →
  (∀ x : ℝ, f x = x^2 - 3*x + 4)

-- Statements to be proved
theorem problem1_solution : ∀ f : ℝ → ℝ, problem1 x f := sorry
theorem problem2_solution : ∀ f : ℝ → ℝ, problem2 f := sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l2106_210619


namespace NUMINAMATH_GPT_percentage_scientists_born_in_june_l2106_210658

theorem percentage_scientists_born_in_june :
  (18 / 200 * 100) = 9 :=
by sorry

end NUMINAMATH_GPT_percentage_scientists_born_in_june_l2106_210658


namespace NUMINAMATH_GPT_units_digit_problem_l2106_210682

open BigOperators

-- Define relevant constants
def A : ℤ := 21
noncomputable def B : ℤ := 14 -- since B = sqrt(196) = 14

-- Define the terms
noncomputable def term1 : ℤ := (A + B) ^ 20
noncomputable def term2 : ℤ := (A - B) ^ 20

-- Statement of the theorem
theorem units_digit_problem :
  ((term1 - term2) % 10) = 4 := 
sorry

end NUMINAMATH_GPT_units_digit_problem_l2106_210682


namespace NUMINAMATH_GPT_find_g_neg2_l2106_210690

-- Definitions of the conditions
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x 

variables (f : ℝ → ℝ) (g : ℝ → ℝ)
variables (h_even_f : even_function f)
variables (h_g_def : ∀ x, g x = f x + x^3)
variables (h_g_2 : g 2 = 10)

-- Statement to prove
theorem find_g_neg2 : g (-2) = -6 :=
sorry

end NUMINAMATH_GPT_find_g_neg2_l2106_210690


namespace NUMINAMATH_GPT_min_value_of_squares_l2106_210603

variable (a b t : ℝ)

theorem min_value_of_squares (ht : 0 < t) (habt : a + b = t) : 
  a^2 + b^2 ≥ t^2 / 2 := 
by
  sorry

end NUMINAMATH_GPT_min_value_of_squares_l2106_210603


namespace NUMINAMATH_GPT_m_range_decrease_y_l2106_210602

theorem m_range_decrease_y {m : ℝ} : (∀ x1 x2 : ℝ, x1 < x2 → (2 * m + 2) * x1 + 5 > (2 * m + 2) * x2 + 5) ↔ m < -1 :=
by
  sorry

end NUMINAMATH_GPT_m_range_decrease_y_l2106_210602


namespace NUMINAMATH_GPT_count_integers_in_interval_l2106_210662

theorem count_integers_in_interval : 
  ∃ (k : ℤ), k = 46 ∧ 
  (∀ n : ℤ, -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ) → (-13 ≤ n ∧ n ≤ 32)) ∧ 
  (∀ n : ℤ, -13 ≤ n ∧ n ≤ 32 → -5 * (2.718 : ℝ) ≤ (n : ℝ) ∧ (n : ℝ) ≤ 12 * (2.718 : ℝ)) :=
sorry

end NUMINAMATH_GPT_count_integers_in_interval_l2106_210662


namespace NUMINAMATH_GPT_acute_triangle_angles_l2106_210697

theorem acute_triangle_angles (α β γ : ℕ) (h1 : α ≥ β) (h2 : β ≥ γ) (h3 : α = 5 * γ) (h4 : α + β + γ = 180) :
  (α = 85 ∧ β = 78 ∧ γ = 17) :=
sorry

end NUMINAMATH_GPT_acute_triangle_angles_l2106_210697


namespace NUMINAMATH_GPT_cost_of_one_book_l2106_210613

theorem cost_of_one_book (m : ℕ) (H1: 1100 < 900 + 9 * m ∧ 900 + 9 * m < 1200)
                                (H2: 1500 < 1300 + 13 * m ∧ 1300 + 13 * m < 1600) : 
                                m = 23 :=
by {
  sorry
}

end NUMINAMATH_GPT_cost_of_one_book_l2106_210613


namespace NUMINAMATH_GPT_angle_sum_around_point_l2106_210695

theorem angle_sum_around_point (y : ℝ) (h : 170 + y + y = 360) : y = 95 := 
sorry

end NUMINAMATH_GPT_angle_sum_around_point_l2106_210695


namespace NUMINAMATH_GPT_min_a2_b2_l2106_210684

theorem min_a2_b2 (a b : ℝ) (h : ∃ x : ℝ, x^4 + a * x^3 + b * x^2 + a * x + 1 = 0) :
  a^2 + b^2 ≥ 4 / 5 :=
sorry

end NUMINAMATH_GPT_min_a2_b2_l2106_210684


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l2106_210605

-- Define the conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ b = c ∨ c = a) ∧ (a + b > c) ∧ (b + c > a) ∧ (c + a > b)

-- Define the side lengths
def side1 := 2
def side2 := 2
def base := 5

-- Define the perimeter
def perimeter (a b c : ℝ) := a + b + c

-- State the theorem
theorem isosceles_triangle_perimeter : isosceles_triangle side1 side2 base → perimeter side1 side2 base = 9 :=
  by sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l2106_210605


namespace NUMINAMATH_GPT_part1_part2_l2106_210659

-- Define the function y in Lean
def y (m x : ℝ) : ℝ := m * x^2 - m * x - 1

-- Part (1)
theorem part1 (x : ℝ) : y (1/2) x < 0 ↔ -1 < x ∧ x < 2 :=
  sorry

-- Part (2)
theorem part2 (x m : ℝ) : y m x < (1 - m) * x - 1 ↔ 
  (m = 0 → x > 0) ∧ 
  (m > 0 → 0 < x ∧ x < 1 / m) ∧ 
  (m < 0 → x < 1 / m ∨ x > 0) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l2106_210659


namespace NUMINAMATH_GPT_find_triangle_altitude_l2106_210646

variable (A b h : ℝ)

theorem find_triangle_altitude (h_eq_40 :  A = 800 ∧ b = 40) : h = 40 :=
sorry

end NUMINAMATH_GPT_find_triangle_altitude_l2106_210646


namespace NUMINAMATH_GPT_fraction_identity_l2106_210622

theorem fraction_identity (f : ℚ) (h : 32 * f^2 = 2^3) : f = 1 / 2 :=
sorry

end NUMINAMATH_GPT_fraction_identity_l2106_210622


namespace NUMINAMATH_GPT_least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l2106_210623

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 3) + 2

theorem least_positive_period_of_f :
  ∃ T > 0, ∀ x, f (x + T) = f x :=
sorry

theorem maximum_value_of_f :
  ∃ x, f x = 3 :=
sorry

theorem monotonically_increasing_intervals_of_f :
  ∀ k : ℤ, ∃ a b : ℝ, a = -Real.pi / 12 + k * Real.pi ∧ b = 5 * Real.pi / 12 + k * Real.pi ∧ ∀ x, a < x ∧ x < b → ∀ x', a ≤ x' ∧ x' ≤ x → f x' < f x :=
sorry

end NUMINAMATH_GPT_least_positive_period_of_f_maximum_value_of_f_monotonically_increasing_intervals_of_f_l2106_210623


namespace NUMINAMATH_GPT_matrix_A_to_power_4_l2106_210604

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end NUMINAMATH_GPT_matrix_A_to_power_4_l2106_210604


namespace NUMINAMATH_GPT_temperature_on_tuesday_l2106_210685

variable (T W Th F : ℝ)

-- Conditions
axiom H1 : (T + W + Th) / 3 = 42
axiom H2 : (W + Th + F) / 3 = 44
axiom H3 : F = 43

-- Proof statement
theorem temperature_on_tuesday : T = 37 :=
by
  -- This would be the place to fill in the proof using H1, H2, and H3
  sorry

end NUMINAMATH_GPT_temperature_on_tuesday_l2106_210685


namespace NUMINAMATH_GPT_unique_solution_a_l2106_210678

theorem unique_solution_a (a : ℚ) : 
  (∃ x : ℚ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0 ∧ 
  ∀ y : ℚ, (y ≠ x → (a^2 - 1) * y^2 + (a + 1) * y + 1 ≠ 0)) ↔ a = 1 ∨ a = 5/3 := 
sorry

end NUMINAMATH_GPT_unique_solution_a_l2106_210678


namespace NUMINAMATH_GPT_increasing_function_l2106_210677

theorem increasing_function (k b : ℝ) :
  (∀ x1 x2 : ℝ, x1 < x2 → (2 * k + 1) * x1 + b < (2 * k + 1) * x2 + b) ↔ k > -1/2 := 
by
  sorry

end NUMINAMATH_GPT_increasing_function_l2106_210677


namespace NUMINAMATH_GPT_sum_of_coeffs_l2106_210607

theorem sum_of_coeffs (x y : ℤ) : (x - 3 * y) ^ 20 = 2 ^ 20 := by
  sorry

end NUMINAMATH_GPT_sum_of_coeffs_l2106_210607


namespace NUMINAMATH_GPT_max_area_inscribed_octagon_l2106_210650

theorem max_area_inscribed_octagon
  (R : ℝ)
  (s : ℝ)
  (a b : ℝ)
  (h1 : s^2 = 5)
  (h2 : (a * b) = 4)
  (h3 : (s * Real.sqrt 2) = (2*R))
  (h4 : (Real.sqrt (a^2 + b^2)) = 2 * R) :
  ∃ A : ℝ, A = 3 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_max_area_inscribed_octagon_l2106_210650


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2106_210661

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (h : a 2 + a 10 = 16) : a 4 + a 6 + a 8 = 24 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2106_210661


namespace NUMINAMATH_GPT_tournament_total_players_l2106_210648

theorem tournament_total_players (n : ℕ) (total_points : ℕ) (total_games : ℕ) (half_points : ℕ → ℕ) :
  (∀ k, half_points k * 2 = total_points) ∧ total_points = total_games ∧
  total_points = n * (n + 11) + 132 ∧
  total_games = (n + 12) * (n + 11) / 2 →
  n + 12 = 24 :=
by
  sorry

end NUMINAMATH_GPT_tournament_total_players_l2106_210648
