import Mathlib

namespace NUMINAMATH_GPT_find_common_difference_l1573_157331

variable {a : ℕ → ℝ} (d : ℝ) (a₁ : ℝ)

-- defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a n = a₁ + n * d

-- condition for the sum of even indexed terms
def sum_even_terms (a : ℕ → ℝ) : ℝ := a 2 + a 4 + a 6 + a 8 + a 10

-- condition for the sum of odd indexed terms
def sum_odd_terms (a : ℕ → ℝ) : ℝ := a 1 + a 3 + a 5 + a 7 + a 9

-- main theorem to prove
theorem find_common_difference
  (a : ℕ → ℝ) (a₁ : ℝ) (d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h_even_sum : sum_even_terms a = 30)
  (h_odd_sum : sum_odd_terms a = 25) :
  d = 1 := by
  sorry

end NUMINAMATH_GPT_find_common_difference_l1573_157331


namespace NUMINAMATH_GPT_inradius_of_triangle_l1573_157315

theorem inradius_of_triangle (A p s r : ℝ) 
  (h1 : A = (1/2) * p) 
  (h2 : p = 2 * s) 
  (h3 : A = r * s) : 
  r = 1 :=
by
  sorry

end NUMINAMATH_GPT_inradius_of_triangle_l1573_157315


namespace NUMINAMATH_GPT_min_value_of_function_l1573_157314

noncomputable def y (θ : ℝ) : ℝ := (2 - Real.sin θ) / (1 - Real.cos θ)

theorem min_value_of_function : ∃ θ : ℝ, y θ = 3 / 4 :=
sorry

end NUMINAMATH_GPT_min_value_of_function_l1573_157314


namespace NUMINAMATH_GPT_g_difference_l1573_157368

-- Define the function g(n)
def g (n : ℤ) : ℚ := (1/2 : ℚ) * n^2 * (n + 3)

-- State the theorem
theorem g_difference (s : ℤ) : g s - g (s - 1) = (1/2 : ℚ) * (3 * s - 2) := by
  sorry

end NUMINAMATH_GPT_g_difference_l1573_157368


namespace NUMINAMATH_GPT_avg_annual_growth_rate_equation_l1573_157385

variable (x : ℝ)
def foreign_trade_income_2007 : ℝ := 250 -- million yuan
def foreign_trade_income_2009 : ℝ := 360 -- million yuan

theorem avg_annual_growth_rate_equation :
  2.5 * (1 + x) ^ 2 = 3.6 := sorry

end NUMINAMATH_GPT_avg_annual_growth_rate_equation_l1573_157385


namespace NUMINAMATH_GPT_right_triangle_sides_l1573_157378

theorem right_triangle_sides (a b c : ℕ) (h1 : a < b) 
  (h2 : 2 * c / 2 = c) 
  (h3 : exists x y, (x + y = 8 ∧ a < b) ∨ (x + y = 9 ∧ a < b)) 
  (h4 : a^2 + b^2 = c^2) : 
  a = 3 ∧ b = 4 ∧ c = 5 := 
by
  sorry

end NUMINAMATH_GPT_right_triangle_sides_l1573_157378


namespace NUMINAMATH_GPT_solve_for_x_l1573_157381

theorem solve_for_x :
  ∃ (x : ℝ), x ≠ 0 ∧ (5 * x)^10 = (10 * x)^5 ∧ x = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1573_157381


namespace NUMINAMATH_GPT_part1_part2_part3_l1573_157341

def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)*x^2 - (m - 1)*x + (m - 1)

theorem part1 (m : ℝ) : (∀ x : ℝ, f m x < 1) ↔ m < (1 - 2 * Real.sqrt 7) / 3 := 
sorry

theorem part2 (m : ℝ) (x : ℝ) : (f m x ≥ (m + 1) * x) ↔ 
  (m = -1 ∧ x ≥ 1) ∨ 
  (m > -1 ∧ (x ≤ (m - 1) / (m + 1) ∨ x ≥ 1)) ∨ 
  (m < -1 ∧ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1)) := 
sorry

theorem part3 (m : ℝ) : (∀ x : ℝ, -1/2 ≤ x ∧ x ≤ 1/2 → f m x ≥ 0) ↔
  m ≥ 1 := 
sorry

end NUMINAMATH_GPT_part1_part2_part3_l1573_157341


namespace NUMINAMATH_GPT_find_constant_l1573_157329

-- Given function f satisfying the conditions
variable (f : ℝ → ℝ)

-- Define the given conditions
variable (h1 : ∀ x : ℝ, f x + 3 * f (c - x) = x)
variable (h2 : f 2 = 2)

-- Statement to prove the constant c
theorem find_constant (c : ℝ) : (f x + 3 * f (c - x) = x) → (f 2 = 2) → c = 8 :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_find_constant_l1573_157329


namespace NUMINAMATH_GPT_blake_change_l1573_157387

-- Definitions based on conditions
def n_l : ℕ := 4
def n_c : ℕ := 6
def p_l : ℕ := 2
def p_c : ℕ := 4 * p_l
def amount_given : ℕ := 6 * 10

-- Total cost calculations derived from the conditions
def total_cost_lollipops : ℕ := n_l * p_l
def total_cost_chocolates : ℕ := n_c * p_c
def total_cost : ℕ := total_cost_lollipops + total_cost_chocolates

-- Calculating the change
def change : ℕ := amount_given - total_cost

-- Theorem stating the final answer
theorem blake_change : change = 4 := sorry

end NUMINAMATH_GPT_blake_change_l1573_157387


namespace NUMINAMATH_GPT_trivia_team_total_score_l1573_157353

theorem trivia_team_total_score 
  (scores : List ℕ)
  (present_members : List ℕ)
  (H_score : scores = [4, 6, 2, 8, 3, 5, 10, 3, 7])
  (H_present : present_members = scores) :
  List.sum present_members = 48 := 
by
  sorry

end NUMINAMATH_GPT_trivia_team_total_score_l1573_157353


namespace NUMINAMATH_GPT_ellipse_symmetry_range_l1573_157351

theorem ellipse_symmetry_range :
  ∀ (x₀ y₀ : ℝ), (x₀^2 / 4 + y₀^2 / 2 = 1) →
  ∃ (x₁ y₁ : ℝ), (x₁ = (4 * y₀ - 3 * x₀) / 5) ∧ (y₁ = (3 * y₀ + 4 * x₀) / 5) →
  -10 ≤ 3 * x₁ - 4 * y₁ ∧ 3 * x₁ - 4 * y₁ ≤ 10 :=
by intros x₀ y₀ h_linearity; sorry

end NUMINAMATH_GPT_ellipse_symmetry_range_l1573_157351


namespace NUMINAMATH_GPT_second_cannibal_wins_l1573_157309

/-- Define a data structure for the position on the chessboard -/
structure Position where
  x : Nat
  y : Nat
  deriving Inhabited, DecidableEq

/-- Check if two positions are adjacent in a legal move (vertical or horizontal) -/
def isAdjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y = p2.y + 1 ∨ p1.y = p2.y - 1)) ∨
  (p1.y = p2.y ∧ (p1.x = p2.x + 1 ∨ p1.x = p2.x - 1))

/-- Define the initial positions of the cannibals -/
def initialPositionFirstCannibal : Position := ⟨1, 1⟩
def initialPositionSecondCannibal : Position := ⟨8, 8⟩

/-- Define a move function for a cannibal (a valid move should keep it on the board) -/
def move (p : Position) (direction : String) : Position :=
  match direction with
  | "up"     => if p.y < 8 then ⟨p.x, p.y + 1⟩ else p
  | "down"   => if p.y > 1 then ⟨p.x, p.y - 1⟩ else p
  | "left"   => if p.x > 1 then ⟨p.x - 1, p.y⟩ else p
  | "right"  => if p.x < 8 then ⟨p.x + 1, p.y⟩ else p
  | _        => p

/-- Predicate determining if a cannibal can eat the other by moving to its position -/
def canEat (p1 p2 : Position) : Bool :=
  p1 = p2

/-- 
  Prove that the second cannibal will eat the first cannibal with the correct strategy. 
  We formalize the fact that with correct play, starting from the initial positions, 
  the second cannibal (initially at ⟨8, 8⟩) can always force a win.
-/
theorem second_cannibal_wins :
  ∀ (p1 p2 : Position), 
  p1 = initialPositionFirstCannibal →
  p2 = initialPositionSecondCannibal →
  (∃ strategy : (Position → String), ∀ positionFirstCannibal : Position, canEat (move p2 (strategy p2)) positionFirstCannibal) :=
by
  sorry

end NUMINAMATH_GPT_second_cannibal_wins_l1573_157309


namespace NUMINAMATH_GPT_part1_proof_l1573_157320

variable (α β t x1 x2 : ℝ)

-- Conditions
def quadratic_roots := 2 * α ^ 2 - t * α - 2 = 0 ∧ 2 * β ^ 2 - t * β - 2 = 0
def roots_relation := α + β = t / 2 ∧ α * β = -1
def points_in_interval := α < β ∧ α ≤ x1 ∧ x1 ≤ β ∧ α ≤ x2 ∧ x2 ≤ β ∧ x1 ≠ x2

-- Proof of Part 1
theorem part1_proof (h1 : quadratic_roots α β t) (h2 : roots_relation α β t)
                    (h3 : points_in_interval α β x1 x2) : 
                    4 * x1 * x2 - t * (x1 + x2) - 4 < 0 := 
sorry

end NUMINAMATH_GPT_part1_proof_l1573_157320


namespace NUMINAMATH_GPT_gross_profit_value_l1573_157311

theorem gross_profit_value (C GP : ℝ) (h1 : GP = 1.6 * C) (h2 : 91 = C + GP) : GP = 56 :=
by
  sorry

end NUMINAMATH_GPT_gross_profit_value_l1573_157311


namespace NUMINAMATH_GPT_product_cos_angles_l1573_157356

theorem product_cos_angles :
  (Real.cos (π / 15) * Real.cos (2 * π / 15) * Real.cos (3 * π / 15) * Real.cos (4 * π / 15) * Real.cos (5 * π / 15) * Real.cos (6 * π / 15) * Real.cos (7 * π / 15) = 1 / 128) :=
sorry

end NUMINAMATH_GPT_product_cos_angles_l1573_157356


namespace NUMINAMATH_GPT_find_unknown_number_l1573_157374

def op (a b : ℝ) := a * (b ^ (1 / 2))

theorem find_unknown_number (x : ℝ) (h : op 4 x = 12) : x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_unknown_number_l1573_157374


namespace NUMINAMATH_GPT_abs_value_solution_l1573_157349

theorem abs_value_solution (a : ℝ) : |-a| = |-5.333| → (a = 5.333 ∨ a = -5.333) :=
by
  sorry

end NUMINAMATH_GPT_abs_value_solution_l1573_157349


namespace NUMINAMATH_GPT_trigonometric_identity_l1573_157305

theorem trigonometric_identity :
  (Real.sin (20 * Real.pi / 180) * Real.cos (70 * Real.pi / 180) +
   Real.sin (10 * Real.pi / 180) * Real.sin (50 * Real.pi / 180)) = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_trigonometric_identity_l1573_157305


namespace NUMINAMATH_GPT_inscribed_circle_radius_eq_l1573_157304

noncomputable def inscribedCircleRadius :=
  let AB := 6
  let AC := 7
  let BC := 8
  let s := (AB + AC + BC) / 2
  let K := Real.sqrt (s * (s - AB) * (s - AC) * (s - BC))
  let r := K / s
  r

theorem inscribed_circle_radius_eq :
  inscribedCircleRadius = Real.sqrt 413.4375 / 10.5 := by
  sorry

end NUMINAMATH_GPT_inscribed_circle_radius_eq_l1573_157304


namespace NUMINAMATH_GPT_find_omega_value_l1573_157323

theorem find_omega_value (ω : ℝ) (h : ω > 0) (h_dist : (1/2) * (2 * π / ω) = π / 6) : ω = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_omega_value_l1573_157323


namespace NUMINAMATH_GPT_parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l1573_157302

noncomputable def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 2 * x

theorem parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2 :
  parabola_equation 1 (-Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_origin_directrix_xaxis_point_1_neg_sqrt2_l1573_157302


namespace NUMINAMATH_GPT_team_OT_matches_l1573_157352

variable (T x M: Nat)

-- Condition: Team C played T matches in the first week.
def team_C_matches_T : Nat := T

-- Condition: Team C played x matches in the first week.
def team_C_matches_x : Nat := x

-- Condition: Team O played M matches in the first week.
def team_O_matches_M : Nat := M

-- Condition: Team C has not played against Team A.
axiom C_not_played_A : ¬ (team_C_matches_T = team_C_matches_x)

-- Condition: Team B has not played against a specified team (interpreted).
axiom B_not_played_specified : ∀ x, ¬ (team_C_matches_x = x)

-- The proof for the number of matches played by team \(\overrightarrow{OT}\).
theorem team_OT_matches : T = 4 := 
    sorry

end NUMINAMATH_GPT_team_OT_matches_l1573_157352


namespace NUMINAMATH_GPT_union_complement_eq_set_l1573_157327

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {3, 5, 6}
def comp_U_N : Set ℕ := U \ N  -- complement of N with respect to U

theorem union_complement_eq_set :
  M ∪ comp_U_N = {1, 2, 3, 4} :=
by
  simp [U, M, N, comp_U_N]
  sorry

end NUMINAMATH_GPT_union_complement_eq_set_l1573_157327


namespace NUMINAMATH_GPT_sqrt_meaningful_range_l1573_157390

-- Define the condition
def sqrt_condition (x : ℝ) : Prop := 1 - 3 * x ≥ 0

-- State the theorem
theorem sqrt_meaningful_range (x : ℝ) (h : sqrt_condition x) : x ≤ 1 / 3 :=
sorry

end NUMINAMATH_GPT_sqrt_meaningful_range_l1573_157390


namespace NUMINAMATH_GPT_oil_cylinder_capacity_l1573_157361

theorem oil_cylinder_capacity
  (C : ℚ) -- total capacity of the cylinder, given as a rational number
  (h1 : 3 / 4 * C + 4 = 4 / 5 * C) -- equation representing the condition of initial and final amounts of oil in the cylinder
  : C = 80 := -- desired result showing the total capacity

sorry

end NUMINAMATH_GPT_oil_cylinder_capacity_l1573_157361


namespace NUMINAMATH_GPT_vector_subtraction_correct_l1573_157383

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_subtraction_correct : (a - b) = (5, -3) :=
by 
  have h1 : a = (2, 1) := by rfl
  have h2 : b = (-3, 4) := by rfl
  sorry

end NUMINAMATH_GPT_vector_subtraction_correct_l1573_157383


namespace NUMINAMATH_GPT_hancho_milk_l1573_157330

def initial_milk : ℝ := 1
def ye_seul_milk : ℝ := 0.1
def ga_young_milk : ℝ := ye_seul_milk + 0.2
def remaining_milk : ℝ := 0.3

theorem hancho_milk : (initial_milk - (ye_seul_milk + ga_young_milk + remaining_milk)) = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_hancho_milk_l1573_157330


namespace NUMINAMATH_GPT_problem_l1573_157348

open Real

noncomputable def f (ω a x : ℝ) := (1 / 2) * (sin (ω * x) + a * cos (ω * x))

theorem problem (a : ℝ) 
  (hω_range : 0 < ω ∧ ω ≤ 1)
  (h_f_sym1 : ∀ x, f ω a x = f ω a (π/3 - x))
  (h_f_sym2 : ∀ x, f ω a (x - π) = f ω a (x + π))
  (x1 x2 : ℝ) 
  (h_x_in_interval1 : -π/3 < x1 ∧ x1 < 5*π/3)
  (h_x_in_interval2 : -π/3 < x2 ∧ x2 < 5*π/3)
  (h_distinct : x1 ≠ x2)
  (h_f_neg_half1 : f ω a x1 = -1/2)
  (h_f_neg_half2 : f ω a x2 = -1/2) :
  (f 1 (sqrt 3) x = sin (x + π/3)) ∧ (x1 + x2 = 7*π/3) :=
by
  sorry

end NUMINAMATH_GPT_problem_l1573_157348


namespace NUMINAMATH_GPT_multiple_of_n_eventually_written_l1573_157335

theorem multiple_of_n_eventually_written (a b n : ℕ) (h_a_pos: 0 < a) (h_b_pos: 0 < b)  (h_ab_neq: a ≠ b) (h_n_pos: 0 < n) :
  ∃ m : ℕ, m % n = 0 :=
sorry

end NUMINAMATH_GPT_multiple_of_n_eventually_written_l1573_157335


namespace NUMINAMATH_GPT_new_students_admitted_l1573_157395

theorem new_students_admitted (orig_students : ℕ := 35) (increase_cost : ℕ := 42) (orig_expense : ℕ := 400) (dim_avg_expense : ℤ := 1) :
  ∃ (x : ℕ), x = 7 :=
by
  sorry

end NUMINAMATH_GPT_new_students_admitted_l1573_157395


namespace NUMINAMATH_GPT_trigonometric_expression_value_l1573_157389

theorem trigonometric_expression_value (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α ^ 2 - Real.cos α ^ 2) / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 3 / 5 := 
sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l1573_157389


namespace NUMINAMATH_GPT_sequence_general_term_l1573_157392

theorem sequence_general_term (a : ℕ+ → ℤ) (h₁ : a 1 = 2) (h₂ : ∀ n : ℕ+, a (n + 1) = a n - 1) :
  ∀ n : ℕ+, a n = 3 - n := 
sorry

end NUMINAMATH_GPT_sequence_general_term_l1573_157392


namespace NUMINAMATH_GPT_postage_unformable_l1573_157350

theorem postage_unformable (n : ℕ) (h₁ : n > 0) (h₂ : 110 = 7 * n - 7 - n) :
  n = 19 := 
sorry

end NUMINAMATH_GPT_postage_unformable_l1573_157350


namespace NUMINAMATH_GPT_max_quarters_is_13_l1573_157372

noncomputable def number_of_quarters (total_value : ℝ) (quarters nickels dimes : ℝ) : Prop :=
  total_value = 4.55 ∧
  quarters = nickels ∧
  dimes = quarters / 2 ∧
  (0.25 * quarters + 0.05 * nickels + 0.05 * quarters / 2 = 4.55)

theorem max_quarters_is_13 : ∃ q : ℝ, number_of_quarters 4.55 q q (q / 2) ∧ q = 13 :=
by
  sorry

end NUMINAMATH_GPT_max_quarters_is_13_l1573_157372


namespace NUMINAMATH_GPT_charles_whistles_l1573_157375

theorem charles_whistles (S C : ℕ) (h1 : S = 45) (h2 : S = C + 32) : C = 13 := 
by
  sorry

end NUMINAMATH_GPT_charles_whistles_l1573_157375


namespace NUMINAMATH_GPT_find_certain_number_l1573_157355

theorem find_certain_number (x : ℕ) 
  (h1 : (28 + x + 42 + 78 + 104) / 5 = 62) 
  (h2 : (48 + 62 + 98 + 124 + x) / 5 = 78) : 
  x = 58 := by
  sorry

end NUMINAMATH_GPT_find_certain_number_l1573_157355


namespace NUMINAMATH_GPT_john_weight_end_l1573_157324

def initial_weight : ℝ := 220
def loss_percentage : ℝ := 0.1
def weight_loss : ℝ := loss_percentage * initial_weight
def weight_gain_back : ℝ := 2
def net_weight_loss : ℝ := weight_loss - weight_gain_back
def final_weight : ℝ := initial_weight - net_weight_loss

theorem john_weight_end :
  final_weight = 200 := 
by 
  sorry

end NUMINAMATH_GPT_john_weight_end_l1573_157324


namespace NUMINAMATH_GPT_find_whole_wheat_pastry_flour_l1573_157364

variable (x : ℕ) -- where x is the pounds of whole-wheat pastry flour Sarah already had

-- Conditions
def rye_flour := 5
def whole_wheat_bread_flour := 10
def chickpea_flour := 3
def total_flour := 20

-- Total flour bought
def total_flour_bought := rye_flour + whole_wheat_bread_flour + chickpea_flour

-- Proof statement
theorem find_whole_wheat_pastry_flour (h : total_flour = total_flour_bought + x) : x = 2 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_find_whole_wheat_pastry_flour_l1573_157364


namespace NUMINAMATH_GPT_count_two_digit_numbers_l1573_157384

theorem count_two_digit_numbers : (99 - 10 + 1) = 90 := by
  sorry

end NUMINAMATH_GPT_count_two_digit_numbers_l1573_157384


namespace NUMINAMATH_GPT_equation_of_line_through_A_parallel_to_given_line_l1573_157367

theorem equation_of_line_through_A_parallel_to_given_line :
  ∃ c : ℝ, 
    (∀ x y : ℝ, 2 * x - y + c = 0 ↔ ∃ a b : ℝ, a = -1 ∧ b = 0 ∧ 2 * a - b + 1 = 0) :=
sorry

end NUMINAMATH_GPT_equation_of_line_through_A_parallel_to_given_line_l1573_157367


namespace NUMINAMATH_GPT_wall_length_l1573_157376

theorem wall_length (side_mirror : ℝ) (width_wall : ℝ) (length_wall : ℝ) 
  (h_mirror: side_mirror = 18) 
  (h_width: width_wall = 32)
  (h_area: (side_mirror ^ 2) * 2 = width_wall * length_wall):
  length_wall = 20.25 := 
by 
  -- The following 'sorry' is a placeholder for the proof
  sorry

end NUMINAMATH_GPT_wall_length_l1573_157376


namespace NUMINAMATH_GPT_range_of_m_l1573_157307

theorem range_of_m (m : ℝ) (h₁ : ∀ x : ℝ, -x^2 + 7*x + 8 ≥ 0 → x^2 - 7*x - 8 ≤ 0)
  (h₂ : ∀ x : ℝ, x^2 - 2*x + 1 - 4*m^2 ≤ 0 → 1 - 2*m ≤ x ∧ x ≤ 1 + 2*m)
  (not_p_sufficient_for_not_q : ∀ x : ℝ, ¬(-x^2 + 7*x + 8 ≥ 0) → ¬(x^2 - 2*x + 1 - 4*m^2 ≤ 0))
  (suff_non_necess : ∀ x : ℝ, (x^2 - 2*x + 1 - 4*m^2 ≤ 0) → ¬(x^2 - 7*x - 8 ≤ 0))
  : 0 < m ∧ m ≤ 1 := sorry

end NUMINAMATH_GPT_range_of_m_l1573_157307


namespace NUMINAMATH_GPT_find_k_l1573_157333

theorem find_k (k : ℝ) (hk : 0 < k) (slope_eq : (2 - k) / (k - 1) = k^2) : k = 1 :=
by sorry

end NUMINAMATH_GPT_find_k_l1573_157333


namespace NUMINAMATH_GPT_problem_inequality_l1573_157358

theorem problem_inequality 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : m + n = 1) : 
  (m + 1 / m) * (n + 1 / n) ≥ 25 / 4 := 
sorry

end NUMINAMATH_GPT_problem_inequality_l1573_157358


namespace NUMINAMATH_GPT_perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l1573_157371

section Problem

-- Definitions based on the problem conditions

-- Condition: Side length of each square is 1 cm
def side_length : ℝ := 1

-- Condition: Thickness of the nail for parts a) and b)
def nail_thickness_a := 0.1
def nail_thickness_b := 0

-- Given a perimeter P and area S, the perimeter cannot exceed certain thresholds based on problem analysis

theorem perimeter_less_than_1_km (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0.1) : P < 1000 * 100 :=
  sorry

theorem perimeter_less_than_1_km_zero_thickness (P : ℝ) (S : ℝ) (r : ℝ) (h1 : 2 * S ≥ r * P) (h2 : r = 0) : P < 1000 * 100 :=
  sorry

theorem perimeter_to_area_ratio (P : ℝ) (S : ℝ) (h : P / S ≤ 700) : P / S < 100000 :=
  sorry

end Problem

end NUMINAMATH_GPT_perimeter_less_than_1_km_perimeter_less_than_1_km_zero_thickness_perimeter_to_area_ratio_l1573_157371


namespace NUMINAMATH_GPT_find_interest_rate_l1573_157347

-- Defining the conditions
def P : ℝ := 5000
def A : ℝ := 5302.98
def t : ℝ := 1.5
def n : ℕ := 2

-- Statement of the problem in Lean 4
theorem find_interest_rate (P A t : ℝ) (n : ℕ) (hP : P = 5000) (hA : A = 5302.98) (ht : t = 1.5) (hn : n = 2) : 
  ∃ r : ℝ, r * 100 = 3.96 :=
sorry

end NUMINAMATH_GPT_find_interest_rate_l1573_157347


namespace NUMINAMATH_GPT_minimize_a2_b2_l1573_157370

theorem minimize_a2_b2 (a b t : ℝ) (h : 2 * a + b = 2 * t) : ∃ a b, (2 * a + b = 2 * t) ∧ (a^2 + b^2 = 4 * t^2 / 5) :=
by
  sorry

end NUMINAMATH_GPT_minimize_a2_b2_l1573_157370


namespace NUMINAMATH_GPT_value_of_a_l1573_157360

theorem value_of_a {a : ℝ} : 
  (∃ x : ℝ, (a - 1) * x^2 + 4 * x - 2 = 0 ∧ ∀ y : ℝ, (a - 1) * y^2 + 4 * y - 2 ≠ 0 → y = x) → 
  (a = 1 ∨ a = -1) :=
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l1573_157360


namespace NUMINAMATH_GPT_not_perfect_squares_l1573_157342

theorem not_perfect_squares :
  (∀ x : ℝ, x * x ≠ 8 ^ 2041) ∧ (∀ y : ℝ, y * y ≠ 10 ^ 2043) :=
by
  sorry

end NUMINAMATH_GPT_not_perfect_squares_l1573_157342


namespace NUMINAMATH_GPT_chuck_team_score_proof_chuck_team_score_l1573_157301

-- Define the conditions
def yellow_team_score : ℕ := 55
def lead : ℕ := 17

-- State the main proposition
theorem chuck_team_score (yellow_team_score : ℕ) (lead : ℕ) : ℕ :=
yellow_team_score + lead

-- Formulate the final proof goal
theorem proof_chuck_team_score : chuck_team_score yellow_team_score lead = 72 :=
by {
  -- This is the place where the proof should go
  sorry
}

end NUMINAMATH_GPT_chuck_team_score_proof_chuck_team_score_l1573_157301


namespace NUMINAMATH_GPT_baseball_football_difference_is_five_l1573_157373

-- Define the conditions
def total_cards : ℕ := 125
def baseball_cards : ℕ := 95
def some_more : ℕ := baseball_cards - 3 * (total_cards - baseball_cards)

-- Define the number of football cards
def football_cards : ℕ := total_cards - baseball_cards

-- Define the difference between the number of baseball cards and three times the number of football cards
def difference : ℕ := baseball_cards - 3 * football_cards

-- Statement of the proof
theorem baseball_football_difference_is_five : difference = 5 := 
by
  sorry

end NUMINAMATH_GPT_baseball_football_difference_is_five_l1573_157373


namespace NUMINAMATH_GPT_find_line_eq_l1573_157336

theorem find_line_eq (m b k : ℝ) (h1 : (2, 7) ∈ ⋃ x, {(x, m * x + b)}) (h2 : ∀ k, abs ((k^2 + 4 * k + 3) - (m * k + b)) = 4) (h3 : b ≠ 0) : (m = 10) ∧ (b = -13) := by
  sorry

end NUMINAMATH_GPT_find_line_eq_l1573_157336


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1573_157346

noncomputable def are_parallel (a : ℝ) : Prop :=
  (2 + a) * a * 3 * a = 3 * a * (a - 2)

theorem sufficient_but_not_necessary_condition :
  (are_parallel 4) ∧ (∃ a ≠ 4, are_parallel a) :=
by {
  sorry
}

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1573_157346


namespace NUMINAMATH_GPT_total_meals_sold_l1573_157345

-- Definitions based on the conditions
def ratio_kids_adult := 2 / 1
def kids_meals := 8

-- The proof problem statement
theorem total_meals_sold : (∃ adults_meals : ℕ, 2 * adults_meals = kids_meals) → (kids_meals + 4 = 12) := 
by 
  sorry

end NUMINAMATH_GPT_total_meals_sold_l1573_157345


namespace NUMINAMATH_GPT_largest_prime_divisor_l1573_157310

-- Let n be a positive integer
def is_positive_integer (n : ℕ) : Prop :=
  n > 0

-- Define that n equals the sum of the squares of its four smallest positive divisors
def is_sum_of_squares_of_smallest_divisors (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a = 2 ∧ b = 5 ∧ c = 10 ∧ n = 1 + a^2 + b^2 + c^2

-- Prove that the largest prime divisor of n is 13
theorem largest_prime_divisor (n : ℕ) (h1 : is_positive_integer n) (h2 : is_sum_of_squares_of_smallest_divisors n) :
  ∃ p : ℕ, Prime p ∧ p ∣ n ∧ ∀ q : ℕ, Prime q ∧ q ∣ n → q ≤ p ∧ p = 13 :=
by
  sorry

end NUMINAMATH_GPT_largest_prime_divisor_l1573_157310


namespace NUMINAMATH_GPT_soccer_team_points_l1573_157321

theorem soccer_team_points 
  (total_games : ℕ) 
  (wins : ℕ) 
  (losses : ℕ) 
  (points_per_win : ℕ) 
  (points_per_draw : ℕ) 
  (points_per_loss : ℕ) 
  (draws : ℕ := total_games - (wins + losses)) : 
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  46 = (wins * points_per_win) + (draws * points_per_draw) + (losses * points_per_loss) :=
by sorry

end NUMINAMATH_GPT_soccer_team_points_l1573_157321


namespace NUMINAMATH_GPT_integer_solution_unique_l1573_157300

theorem integer_solution_unique (x y : ℝ) (h : -1 < (y - x) / (x + y) ∧ (y - x) / (x + y) < 2) (hyx : ∃ n : ℤ, y = n * x) : y = x :=
by
  sorry

end NUMINAMATH_GPT_integer_solution_unique_l1573_157300


namespace NUMINAMATH_GPT_boat_travel_distance_downstream_l1573_157380

-- Define the given conditions
def speed_boat_still : ℝ := 22
def speed_stream : ℝ := 5
def time_downstream : ℝ := 5

-- Define the effective speed and the computed distance
def effective_speed_downstream : ℝ := speed_boat_still + speed_stream
def distance_traveled_downstream : ℝ := effective_speed_downstream * time_downstream

-- State the proof problem that distance_traveled_downstream is 135 km
theorem boat_travel_distance_downstream :
  distance_traveled_downstream = 135 :=
by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_boat_travel_distance_downstream_l1573_157380


namespace NUMINAMATH_GPT_percentage_more_l1573_157396

variable (J : ℝ) -- Juan's income
noncomputable def Tim_income := 0.60 * J -- T = 0.60J
noncomputable def Mart_income := 0.84 * J -- M = 0.84J

theorem percentage_more {J : ℝ} (T := Tim_income J) (M := Mart_income J) :
  ((M - T) / T) * 100 = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_more_l1573_157396


namespace NUMINAMATH_GPT_johns_number_is_1500_l1573_157325

def is_multiple_of (a b : Nat) : Prop := ∃ k, a = k * b

theorem johns_number_is_1500 (n : ℕ) (h1 : is_multiple_of n 125) (h2 : is_multiple_of n 30) (h3 : 1000 ≤ n ∧ n ≤ 3000) : n = 1500 :=
by
  -- proof structure goes here
  sorry

end NUMINAMATH_GPT_johns_number_is_1500_l1573_157325


namespace NUMINAMATH_GPT_find_ff_half_l1573_157340

noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then x + 1 else -x + 3

theorem find_ff_half : f (f (1 / 2)) = 3 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_ff_half_l1573_157340


namespace NUMINAMATH_GPT_geometric_sequence_a7_eq_64_l1573_157328

open Nat

theorem geometric_sequence_a7_eq_64 (a : ℕ → ℕ) (h1 : a 1 = 1) (hrec : ∀ n : ℕ, a (n + 1) = 2 * a n) : a 7 = 64 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_eq_64_l1573_157328


namespace NUMINAMATH_GPT_non_mobile_payment_probability_40_60_l1573_157312

variable (total_customers : ℕ)
variable (num_non_mobile_40_50 : ℕ)
variable (num_non_mobile_50_60 : ℕ)

theorem non_mobile_payment_probability_40_60 
  (h_total_customers: total_customers = 100)
  (h_num_non_mobile_40_50: num_non_mobile_40_50 = 9)
  (h_num_non_mobile_50_60: num_non_mobile_50_60 = 5) : 
  (num_non_mobile_40_50 + num_non_mobile_50_60 : ℚ) / total_customers = 7 / 50 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_non_mobile_payment_probability_40_60_l1573_157312


namespace NUMINAMATH_GPT_circumference_of_back_wheel_l1573_157382

theorem circumference_of_back_wheel
  (C_f : ℝ) (C_b : ℝ) (D : ℝ) (N_b : ℝ)
  (h1 : C_f = 30)
  (h2 : D = 1650)
  (h3 : (N_b + 5) * C_f = D)
  (h4 : N_b * C_b = D) :
  C_b = 33 :=
sorry

end NUMINAMATH_GPT_circumference_of_back_wheel_l1573_157382


namespace NUMINAMATH_GPT_find_A_from_AB9_l1573_157386

theorem find_A_from_AB9 (A B : ℕ) (h1 : 0 ≤ A ∧ A ≤ 9) (h2 : 0 ≤ B ∧ B ≤ 9) (h3 : 100 * A + 10 * B + 9 = 459) : A = 4 :=
sorry

end NUMINAMATH_GPT_find_A_from_AB9_l1573_157386


namespace NUMINAMATH_GPT_find_number_l1573_157326

theorem find_number (x : ℝ) (h : 45 - 3 * x = 12) : x = 11 :=
sorry

end NUMINAMATH_GPT_find_number_l1573_157326


namespace NUMINAMATH_GPT_steaks_from_15_pounds_of_beef_l1573_157359

-- Definitions for conditions
def pounds_to_ounces (pounds : ℕ) : ℕ := pounds * 16

def steaks_count (total_ounces : ℕ) (ounces_per_steak : ℕ) : ℕ := total_ounces / ounces_per_steak

-- Translate the problem to Lean statement
theorem steaks_from_15_pounds_of_beef : 
  steaks_count (pounds_to_ounces 15) 12 = 20 :=
by
  sorry

end NUMINAMATH_GPT_steaks_from_15_pounds_of_beef_l1573_157359


namespace NUMINAMATH_GPT_percentage_increase_l1573_157379

-- Define the initial and final prices as constants
def P_inicial : ℝ := 5.00
def P_final : ℝ := 5.55

-- Define the percentage increase proof
theorem percentage_increase : ((P_final - P_inicial) / P_inicial) * 100 = 11 := 
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1573_157379


namespace NUMINAMATH_GPT_maximum_number_of_buses_l1573_157388

-- Definitions of the conditions
def bus_stops : ℕ := 9 -- There are 9 bus stops in the city.
def max_common_stops : ℕ := 1 -- Any two buses have at most one common stop.
def stops_per_bus : ℕ := 3 -- Each bus stops at exactly three stops.

-- The main theorem statement
theorem maximum_number_of_buses (n : ℕ) :
  (bus_stops = 9) →
  (max_common_stops = 1) →
  (stops_per_bus = 3) →
  n ≤ 12 :=
sorry

end NUMINAMATH_GPT_maximum_number_of_buses_l1573_157388


namespace NUMINAMATH_GPT_fraction_of_crop_to_CD_is_correct_l1573_157397

-- Define the trapezoid with given conditions
structure Trapezoid :=
  (AB CD AD BC : ℝ)
  (angleA angleD : ℝ)
  (h: ℝ) -- height
  (Area Trapezoid total_area close_area_to_CD: ℝ) 

-- Assumptions
axiom AB_eq_CD (T : Trapezoid) : T.AB = 150 
axiom CD_eq_CD (T : Trapezoid) : T.CD = 200
axiom AD_eq_CD (T : Trapezoid) : T.AD = 130
axiom BC_eq_CD (T : Trapezoid) : T.BC = 130
axiom angleA_eq_75 (T : Trapezoid) : T.angleA = 75
axiom angleD_eq_75 (T : Trapezoid) : T.angleD = 75

-- The fraction calculation
noncomputable def fraction_to_CD (T : Trapezoid) : ℝ :=
  T.close_area_to_CD / T.total_area

-- Theorem stating the fraction of the crop that is brought to the longer base CD is 15/28
theorem fraction_of_crop_to_CD_is_correct (T : Trapezoid) 
  (h_pos : 0 < T.h)
  (total_area_def : T.total_area = (T.AB + T.CD) * T.h / 2)
  (close_area_def : T.close_area_to_CD = ((T.h / 4) * (T.AB + T.CD))) : 
  fraction_to_CD T = 15 / 28 :=
  sorry

end NUMINAMATH_GPT_fraction_of_crop_to_CD_is_correct_l1573_157397


namespace NUMINAMATH_GPT_total_weight_of_courtney_marble_collection_l1573_157322

def marble_weight_first_jar : ℝ := 80 * 0.35
def marble_weight_second_jar : ℝ := 160 * 0.45
def marble_weight_third_jar : ℝ := 20 * 0.25

/-- The total weight of Courtney's marble collection -/
theorem total_weight_of_courtney_marble_collection :
    marble_weight_first_jar + marble_weight_second_jar + marble_weight_third_jar = 105 := by
  sorry

end NUMINAMATH_GPT_total_weight_of_courtney_marble_collection_l1573_157322


namespace NUMINAMATH_GPT_find_x_l1573_157377

open Real

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_x (x : ℝ) : 
  let ab := (a.1 + x * b.1, a.2 + x * b.2)
  let minus_b := (-b.1, -b.2)
  dot_product ab minus_b = 0 
  → x = -2 / 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_x_l1573_157377


namespace NUMINAMATH_GPT_distance_to_city_hall_l1573_157306

variable (d : ℝ) (t : ℝ)

-- Conditions
def condition1 : Prop := d = 45 * (t + 1.5)
def condition2 : Prop := d - 45 = 65 * (t - 1.25)
def condition3 : Prop := t > 0

theorem distance_to_city_hall
  (h1 : condition1 d t)
  (h2 : condition2 d t)
  (h3 : condition3 t)
  : d = 300 := by
  sorry

end NUMINAMATH_GPT_distance_to_city_hall_l1573_157306


namespace NUMINAMATH_GPT_problem_b_lt_a_lt_c_l1573_157354

theorem problem_b_lt_a_lt_c (a b c : ℝ)
  (h1 : 1.001 * Real.exp a = Real.exp 1.001)
  (h2 : b - Real.sqrt (1000 / 1001) = 1.001 - Real.sqrt 1.001)
  (h3 : c = 1.001) : b < a ∧ a < c := by
  sorry

end NUMINAMATH_GPT_problem_b_lt_a_lt_c_l1573_157354


namespace NUMINAMATH_GPT_sum_of_three_consecutive_even_numbers_l1573_157399

theorem sum_of_three_consecutive_even_numbers (m : ℤ) (h : ∃ k, m = 2 * k) : 
  m + (m + 2) + (m + 4) = 3 * m + 6 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_consecutive_even_numbers_l1573_157399


namespace NUMINAMATH_GPT_sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l1573_157338

variable {a b : ℝ}

theorem sufficient_condition_for_reciprocal_square :
  (b > a ∧ a > 0) → (1 / a^2 > 1 / b^2) :=
sorry

theorem not_necessary_condition_for_reciprocal_square :
  ¬((1 / a^2 > 1 / b^2) → (b > a ∧ a > 0)) :=
sorry

end NUMINAMATH_GPT_sufficient_condition_for_reciprocal_square_not_necessary_condition_for_reciprocal_square_l1573_157338


namespace NUMINAMATH_GPT_right_triangle_third_side_square_l1573_157319

theorem right_triangle_third_side_square (a b : ℕ) (c : ℕ) 
  (h₁ : a = 3) (h₂ : b = 4) (h₃ : a^2 + b^2 = c^2) :
  c^2 = 25 ∨ a^2 + c^2 = b^2 ∨ a^2 + b^2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_square_l1573_157319


namespace NUMINAMATH_GPT_gcd_6Tn_nplus1_l1573_157366

theorem gcd_6Tn_nplus1 (n : ℕ) (h : 0 < n) : gcd (3 * n * n + 3 * n) (n + 1) = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_6Tn_nplus1_l1573_157366


namespace NUMINAMATH_GPT_multiplication_of_powers_of_10_l1573_157318

theorem multiplication_of_powers_of_10 : (10 : ℝ) ^ 65 * (10 : ℝ) ^ 64 = (10 : ℝ) ^ 129 := by
  sorry

end NUMINAMATH_GPT_multiplication_of_powers_of_10_l1573_157318


namespace NUMINAMATH_GPT_tailor_cut_skirt_l1573_157334

theorem tailor_cut_skirt (cut_pants cut_skirt : ℝ) (h1 : cut_pants = 0.5) (h2 : cut_skirt = cut_pants + 0.25) : cut_skirt = 0.75 :=
by
  sorry

end NUMINAMATH_GPT_tailor_cut_skirt_l1573_157334


namespace NUMINAMATH_GPT_number_of_perpendicular_points_on_ellipse_l1573_157369

theorem number_of_perpendicular_points_on_ellipse :
  ∃ (P : ℝ × ℝ), (P ∈ {P : ℝ × ℝ | (P.1^2 / 8) + (P.2^2 / 4) = 1})
  ∧ (∀ (F1 F2 : ℝ × ℝ), F1 ≠ F2 → ∀ (P : ℝ × ℝ), ((P.1 - F1.1) * (P.1 - F2.1) + (P.2 - F1.2) * (P.2 - F2.2)) = 0) :=
sorry

end NUMINAMATH_GPT_number_of_perpendicular_points_on_ellipse_l1573_157369


namespace NUMINAMATH_GPT_total_daisies_sold_l1573_157343

-- Conditions Definitions
def first_day_sales : ℕ := 45
def second_day_sales : ℕ := first_day_sales + 20
def third_day_sales : ℕ := 2 * second_day_sales - 10
def fourth_day_sales : ℕ := 120

-- Question: Prove that the total sales over the four days is 350.
theorem total_daisies_sold :
  first_day_sales + second_day_sales + third_day_sales + fourth_day_sales = 350 := by
  sorry

end NUMINAMATH_GPT_total_daisies_sold_l1573_157343


namespace NUMINAMATH_GPT_paint_intensity_change_l1573_157357

theorem paint_intensity_change (intensity_original : ℝ) (intensity_new : ℝ) (fraction_replaced : ℝ) 
  (h1 : intensity_original = 0.40) (h2 : intensity_new = 0.20) (h3 : fraction_replaced = 1) :
  intensity_new = 0.20 :=
by
  sorry

end NUMINAMATH_GPT_paint_intensity_change_l1573_157357


namespace NUMINAMATH_GPT_find_a_l1573_157391

theorem find_a (a : ℝ) (ha : a ≠ 0)
  (h_area : (1/2) * (a/2) * a^2 = 2) :
  a = 2 ∨ a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_l1573_157391


namespace NUMINAMATH_GPT_no_rational_roots_of_odd_coefficient_quadratic_l1573_157363

theorem no_rational_roots_of_odd_coefficient_quadratic 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ¬ ∃ r : ℚ, r * r * a + r * b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_rational_roots_of_odd_coefficient_quadratic_l1573_157363


namespace NUMINAMATH_GPT_initial_population_of_first_village_l1573_157316

theorem initial_population_of_first_village (P : ℕ) :
  (P - 1200 * 18) = (42000 + 800 * 18) → P = 78000 :=
by
  sorry

end NUMINAMATH_GPT_initial_population_of_first_village_l1573_157316


namespace NUMINAMATH_GPT_area_of_remaining_shape_l1573_157317

/-- Define the initial 6x6 square grid with each cell of size 1 cm. -/
def initial_square_area : ℝ := 6 * 6

/-- Define the area of the combined dark gray triangles forming a 1x3 rectangle. -/
def dark_gray_area : ℝ := 1 * 3

/-- Define the area of the combined light gray triangles forming a 2x3 rectangle. -/
def light_gray_area : ℝ := 2 * 3

/-- Define the total area of the gray triangles cut out. -/
def total_gray_area : ℝ := dark_gray_area + light_gray_area

/-- Calculate the area of the remaining figure after cutting out the gray triangles. -/
def remaining_area : ℝ := initial_square_area - total_gray_area

/-- Proof that the area of the remaining shape is 27 square centimeters. -/
theorem area_of_remaining_shape : remaining_area = 27 := by
  sorry

end NUMINAMATH_GPT_area_of_remaining_shape_l1573_157317


namespace NUMINAMATH_GPT_max_points_of_intersection_l1573_157344

-- Define the lines and their properties
variable (L : Fin 150 → Prop)

-- Condition: L_5n are parallel to each other
def parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k

-- Condition: L_{5n-1} pass through a given point B
def passing_through_B (n : ℕ) :=
  ∃ k, n = 5 * k + 1

-- Condition: L_{5n-2} are parallel to another line not parallel to those in parallel_group
def other_parallel_group (n : ℕ) :=
  ∃ k, n = 5 * k + 3

-- Total number of points of intersection of pairs of lines from the complete set
theorem max_points_of_intersection (L : Fin 150 → Prop)
  (h_distinct : ∀ i j : Fin 150, i ≠ j → L i ≠ L j)
  (h_parallel_group : ∀ i j : Fin 150, parallel_group i → parallel_group j → L i = L j)
  (h_through_B : ∀ i j : Fin 150, passing_through_B i → passing_through_B j → L i = L j)
  (h_other_parallel_group : ∀ i j : Fin 150, other_parallel_group i → other_parallel_group j → L i = L j)
  : ∃ P, P = 8071 := 
sorry

end NUMINAMATH_GPT_max_points_of_intersection_l1573_157344


namespace NUMINAMATH_GPT_golu_distance_after_turning_left_l1573_157362

theorem golu_distance_after_turning_left :
  ∀ (a c b : ℝ), a = 8 → c = 10 → (c ^ 2 = a ^ 2 + b ^ 2) → b = 6 :=
by
  intros a c b ha hc hpyth
  rw [ha, hc] at hpyth
  sorry

end NUMINAMATH_GPT_golu_distance_after_turning_left_l1573_157362


namespace NUMINAMATH_GPT_mike_disk_space_l1573_157393

theorem mike_disk_space (F L T : ℕ) (hF : F = 26) (hL : L = 2) : T = 28 :=
by
  have h : T = F + L := by sorry
  rw [hF, hL] at h
  assumption

end NUMINAMATH_GPT_mike_disk_space_l1573_157393


namespace NUMINAMATH_GPT_eduardo_ate_fraction_of_remaining_l1573_157394

theorem eduardo_ate_fraction_of_remaining (init_cookies : ℕ) (nicole_fraction : ℚ) (remaining_percent : ℚ) :
  init_cookies = 600 →
  nicole_fraction = 2 / 5 →
  remaining_percent = 24 / 100 →
  (360 - (600 * 24 / 100)) / 360 = 3 / 5 := by
  sorry

end NUMINAMATH_GPT_eduardo_ate_fraction_of_remaining_l1573_157394


namespace NUMINAMATH_GPT_quadratic_term_free_solution_l1573_157398

theorem quadratic_term_free_solution (m : ℝ) : 
  (∀ x : ℝ, ∃ (p : ℝ → ℝ), (x + m) * (x^2 + 2*x - 1) = p x + (2 + m) * x^2) → m = -2 :=
by
  intro H
  sorry

end NUMINAMATH_GPT_quadratic_term_free_solution_l1573_157398


namespace NUMINAMATH_GPT_problem_proof_l1573_157303

variables {m n : ℝ}

-- Line definitions
def l1 (m n x y : ℝ) : Prop := m * x + 8 * y + n = 0
def l2 (m x y : ℝ) : Prop := 2 * x + m * y - 1 = 0

-- Conditions
def intersects_at (m n : ℝ) : Prop :=
  l1 m n m (-1) ∧ l2 m m (-1)

def parallel (m n : ℝ) : Prop :=
  (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2)

def perpendicular (m n : ℝ) : Prop :=
  m = 0 ∧ n = 8

theorem problem_proof :
  intersects_at m n → (m = 1 ∧ n = 7) ∧
  parallel m n → (m = 4 ∧ n ≠ -2) ∨ (m = -4 ∧ n ≠ 2) ∧
  perpendicular m n → (m = 0 ∧ n = 8) :=
by
  sorry

end NUMINAMATH_GPT_problem_proof_l1573_157303


namespace NUMINAMATH_GPT_solution_set_inequality_l1573_157313

theorem solution_set_inequality (x : ℝ) : (x^2 - 2*x - 8 ≥ 0) ↔ (x ≤ -2 ∨ x ≥ 4) := 
sorry

end NUMINAMATH_GPT_solution_set_inequality_l1573_157313


namespace NUMINAMATH_GPT_least_sum_of_exponents_l1573_157308

theorem least_sum_of_exponents {n : ℕ} (h : n = 520) (h_exp : ∃ (a b : ℕ), 2^a + 2^b = n ∧ a ≠ b ∧ a = 9 ∧ b = 3) : 
    (∃ (s : ℕ), s = 9 + 3) :=
by
  sorry

end NUMINAMATH_GPT_least_sum_of_exponents_l1573_157308


namespace NUMINAMATH_GPT_correct_product_l1573_157365

theorem correct_product (a b : ℕ) (a' : ℕ) (h1 : a' = (a % 10) * 10 + (a / 10)) 
  (h2 : a' * b = 143) (h3 : 10 ≤ a ∧ a < 100):
  a * b = 341 :=
sorry

end NUMINAMATH_GPT_correct_product_l1573_157365


namespace NUMINAMATH_GPT_abs_inequality_solution_l1573_157337

theorem abs_inequality_solution (x : ℝ) :
  (abs (x - 2) + abs (x + 3) < 8) ↔ (-4.5 < x ∧ x < 3.5) :=
by sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1573_157337


namespace NUMINAMATH_GPT_find_value_l1573_157339

theorem find_value (x : ℝ) (f₁ f₂ : ℝ) (p : ℝ) (y₁ y₂ : ℝ) 
  (h1 : x * f₁ = (p * x) * y₁)
  (h2 : x * f₂ = (p * x) * y₂)
  (hf₁ : f₁ = 1 / 3)
  (hx : x = 4)
  (hy₁ : y₁ = 8)
  (hf₂ : f₂ = 1 / 8):
  y₂ = 3 := by
sorry

end NUMINAMATH_GPT_find_value_l1573_157339


namespace NUMINAMATH_GPT_inequality_proof_l1573_157332

theorem inequality_proof
  (x y : ℝ)
  (h : x^4 + y^4 ≥ 2) :
  |x^16 - y^16| + 4 * x^8 * y^8 ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1573_157332
