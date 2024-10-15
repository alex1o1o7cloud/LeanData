import Mathlib

namespace NUMINAMATH_GPT_tiling_tromino_l804_80471

theorem tiling_tromino (m n : ℕ) : (∀ t : ℕ, (t = 3) → (3 ∣ m * n)) → (m * n % 6 = 0) → (m * n % 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tiling_tromino_l804_80471


namespace NUMINAMATH_GPT_angle_terminal_side_eq_l804_80430

theorem angle_terminal_side_eq (α : ℝ) : 
  (α = -4 * Real.pi / 3 + 2 * Real.pi) → (0 ≤ α ∧ α < 2 * Real.pi) → α = 2 * Real.pi / 3 := 
by 
  sorry

end NUMINAMATH_GPT_angle_terminal_side_eq_l804_80430


namespace NUMINAMATH_GPT_minimum_value_inequality_maximum_value_inequality_l804_80409

noncomputable def minimum_value (x1 x2 x3 : ℝ) : ℝ :=
  (x1 + 3 * x2 + 5 * x3) * (x1 + x2 / 3 + x3 / 5)

theorem minimum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  1 ≤ minimum_value x1 x2 x3 :=
sorry

theorem maximum_value_inequality (x1 x2 x3 : ℝ) (h : 0 ≤ x1) (h : 0 ≤ x2) (h : 0 ≤ x3) (sum_eq : x1 + x2 + x3 = 1) :
  minimum_value x1 x2 x3 ≤ 9/5 :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_maximum_value_inequality_l804_80409


namespace NUMINAMATH_GPT_price_of_each_orange_l804_80449

theorem price_of_each_orange 
  (x : ℕ)
  (a o : ℕ)
  (h1 : a + o = 20)
  (h2 : 40 * a + x * o = 1120)
  (h3 : (a + o - 10) * 52 = 1120 - 10 * x) :
  x = 60 :=
sorry

end NUMINAMATH_GPT_price_of_each_orange_l804_80449


namespace NUMINAMATH_GPT_contrapositive_of_inequality_l804_80498

theorem contrapositive_of_inequality (a b c : ℝ) (h : a > b → a + c > b + c) : a + c ≤ b + c → a ≤ b :=
by
  intro h_le
  apply not_lt.mp
  intro h_gt
  have h2 := h h_gt
  linarith

end NUMINAMATH_GPT_contrapositive_of_inequality_l804_80498


namespace NUMINAMATH_GPT_olympiad2024_sum_l804_80497

theorem olympiad2024_sum (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_product : A * B * C = 2310) : 
  A + B + C ≤ 390 :=
sorry

end NUMINAMATH_GPT_olympiad2024_sum_l804_80497


namespace NUMINAMATH_GPT_possible_values_of_m_l804_80427

def F1 := (-3, 0)
def F2 := (3, 0)
def possible_vals := [2, -1, 4, -3, 1/2]

noncomputable def is_valid_m (m : ℝ) : Prop :=
  abs (2 * m - 1) < 6 ∧ m ≠ 1/2

theorem possible_values_of_m : {m ∈ possible_vals | is_valid_m m} = {2, -1} := by
  sorry

end NUMINAMATH_GPT_possible_values_of_m_l804_80427


namespace NUMINAMATH_GPT_usual_time_is_24_l804_80479

variable (R T : ℝ)
variable (usual_rate fraction_of_rate early_min : ℝ)
variable (h1 : fraction_of_rate = 6 / 7)
variable (h2 : early_min = 4)
variable (h3 : (R / (fraction_of_rate * R)) = 7 / 6)
variable (h4 : ((T - early_min) / T) = fraction_of_rate)

theorem usual_time_is_24 {R T : ℝ} (fraction_of_rate := 6/7) (early_min := 4) :
  fraction_of_rate = 6 / 7 ∧ early_min = 4 → 
  (T - early_min) / T = fraction_of_rate → 
  T = 24 :=
by
  intros hfraction_hearly htime_eq_fraction
  sorry

end NUMINAMATH_GPT_usual_time_is_24_l804_80479


namespace NUMINAMATH_GPT_color_coat_drying_time_l804_80414

theorem color_coat_drying_time : ∀ (x : ℕ), 2 + 2 * x + 5 = 13 → x = 3 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_color_coat_drying_time_l804_80414


namespace NUMINAMATH_GPT_probability_intersecting_diagonals_l804_80448

def number_of_vertices := 10

def number_of_diagonals : ℕ := Nat.choose number_of_vertices 2 - number_of_vertices

def number_of_ways_choose_two_diagonals := Nat.choose number_of_diagonals 2

def number_of_sets_of_intersecting_diagonals : ℕ := Nat.choose number_of_vertices 4

def intersection_probability : ℚ :=
  (number_of_sets_of_intersecting_diagonals : ℚ) / (number_of_ways_choose_two_diagonals : ℚ)

theorem probability_intersecting_diagonals :
  intersection_probability = 42 / 119 :=
by
  sorry

end NUMINAMATH_GPT_probability_intersecting_diagonals_l804_80448


namespace NUMINAMATH_GPT_find_initial_children_l804_80422

-- Definition of conditions
def initial_children_on_bus (X : ℕ) := 
  let final_children := (X + 40) - 60 
  final_children = 2

-- Theorem statement
theorem find_initial_children : 
  ∃ X : ℕ, initial_children_on_bus X ∧ X = 22 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_children_l804_80422


namespace NUMINAMATH_GPT_time_after_2004_hours_l804_80461

variable (h : ℕ) 

-- Current time is represented as an integer from 0 to 11 (9 o'clock).
def current_time : ℕ := 9

-- 12-hour clock cycles every 12 hours.
def cycle : ℕ := 12

-- Time after 2004 hours.
def hours_after : ℕ := 2004

-- Proof statement
theorem time_after_2004_hours (h : ℕ) :
  (current_time + hours_after) % cycle = current_time := 
sorry

end NUMINAMATH_GPT_time_after_2004_hours_l804_80461


namespace NUMINAMATH_GPT_jackie_apples_l804_80425

variable (A J : ℕ)

-- Condition: Adam has 3 more apples than Jackie.
axiom h1 : A = J + 3

-- Condition: Adam has 9 apples.
axiom h2 : A = 9

-- Question: How many apples does Jackie have?
theorem jackie_apples : J = 6 :=
by
  -- We would normally the proof steps here, but we'll skip to the answer
  sorry

end NUMINAMATH_GPT_jackie_apples_l804_80425


namespace NUMINAMATH_GPT_expr_1989_eval_expr_1990_eval_l804_80477

def nestedExpr : ℕ → ℤ
| 0     => 0
| (n+1) => -1 - (nestedExpr n)

-- Conditions translated into Lean definitions:
def expr_1989 := nestedExpr 1989
def expr_1990 := nestedExpr 1990

-- The proof statements:
theorem expr_1989_eval : expr_1989 = -1 := sorry
theorem expr_1990_eval : expr_1990 = 0 := sorry

end NUMINAMATH_GPT_expr_1989_eval_expr_1990_eval_l804_80477


namespace NUMINAMATH_GPT_G_greater_F_l804_80408

theorem G_greater_F (x : ℝ) : 
  let F := 2*x^2 - 3*x - 2
  let G := 3*x^2 - 7*x + 5
  G > F := 
sorry

end NUMINAMATH_GPT_G_greater_F_l804_80408


namespace NUMINAMATH_GPT_contracting_schemes_l804_80421

theorem contracting_schemes :
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  (Nat.choose total_projects a_contracts) *
  (Nat.choose (total_projects - a_contracts) b_contracts) *
  (Nat.choose ((total_projects - a_contracts) - b_contracts) c_contracts) = 60 :=
by
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  sorry

end NUMINAMATH_GPT_contracting_schemes_l804_80421


namespace NUMINAMATH_GPT_ab_square_value_l804_80476

noncomputable def cyclic_quadrilateral (AX AY BX BY CX CY AB2 : ℝ) : Prop :=
  AX * AY = 6 ∧
  BX * BY = 5 ∧
  CX * CY = 4 ∧
  AB2 = 122 / 15

theorem ab_square_value :
  ∃ (AX AY BX BY CX CY : ℝ), cyclic_quadrilateral AX AY BX BY CX CY (122 / 15) :=
by
  sorry

end NUMINAMATH_GPT_ab_square_value_l804_80476


namespace NUMINAMATH_GPT_quadrant_of_angle_l804_80453

-- Definitions for conditions
def sin_pos_cos_pos (α : ℝ) : Prop := (Real.sin α) * (Real.cos α) > 0

-- The theorem to prove
theorem quadrant_of_angle (α : ℝ) (h : sin_pos_cos_pos α) : 
  (0 < α ∧ α < π / 2) ∨ (π < α ∧ α < 3 * π / 2) :=
sorry

end NUMINAMATH_GPT_quadrant_of_angle_l804_80453


namespace NUMINAMATH_GPT_daily_wage_of_c_l804_80424

theorem daily_wage_of_c 
  (a_days : ℕ) (b_days : ℕ) (c_days : ℕ) 
  (wage_ratio_a_b : ℚ) (wage_ratio_b_c : ℚ) 
  (total_earnings : ℚ) 
  (A : ℚ) (C : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  wage_ratio_a_b = 3 / 4 →
  wage_ratio_b_c = 4 / 5 →
  total_earnings = 1850 →
  A = 75 →
  C = 208.33 := 
sorry

end NUMINAMATH_GPT_daily_wage_of_c_l804_80424


namespace NUMINAMATH_GPT_cos_B_in_third_quadrant_l804_80456

theorem cos_B_in_third_quadrant (B : ℝ) (hB: π < B ∧ B < 3 * π / 2) (hSinB: Real.sin B = 5 / 13) : Real.cos B = - 12 / 13 := by
  sorry

end NUMINAMATH_GPT_cos_B_in_third_quadrant_l804_80456


namespace NUMINAMATH_GPT_symmetric_point_of_M_origin_l804_80444

-- Define the point M with given coordinates
def M : (ℤ × ℤ) := (-3, -5)

-- The theorem stating that the symmetric point of M about the origin is (3, 5)
theorem symmetric_point_of_M_origin :
  let symmetric_point : (ℤ × ℤ) := (-M.1, -M.2)
  symmetric_point = (3, 5) :=
by
  -- (Proof should be filled)
  sorry

end NUMINAMATH_GPT_symmetric_point_of_M_origin_l804_80444


namespace NUMINAMATH_GPT_ab_value_l804_80400

theorem ab_value (a b : ℝ)
  (h1 : a + b = 4)
  (h2 : a^2 + b^2 = 18) : a * b = -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_ab_value_l804_80400


namespace NUMINAMATH_GPT_angle_PTV_60_l804_80482

variables (m n TV TPV PTV : ℝ)

-- We state the conditions
axiom parallel_lines : m = n
axiom angle_TPV : TPV = 150
axiom angle_TVP_perpendicular : TV = 90

-- The goal statement to prove
theorem angle_PTV_60 : PTV = 60 :=
by
  sorry

end NUMINAMATH_GPT_angle_PTV_60_l804_80482


namespace NUMINAMATH_GPT_find_other_number_l804_80463

theorem find_other_number (b : ℕ) (lcm_val gcd_val : ℕ)
  (h_lcm : Nat.lcm 240 b = 2520)
  (h_gcd : Nat.gcd 240 b = 24) :
  b = 252 :=
sorry

end NUMINAMATH_GPT_find_other_number_l804_80463


namespace NUMINAMATH_GPT_neg_of_exists_lt_is_forall_ge_l804_80499

theorem neg_of_exists_lt_is_forall_ge :
  (¬ (∃ x : ℝ, x^2 - 2 * x + 1 < 0)) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_of_exists_lt_is_forall_ge_l804_80499


namespace NUMINAMATH_GPT_alex_average_speed_l804_80417

def total_distance : ℕ := 48
def biking_time : ℕ := 6

theorem alex_average_speed : (total_distance / biking_time) = 8 := 
by
  sorry

end NUMINAMATH_GPT_alex_average_speed_l804_80417


namespace NUMINAMATH_GPT_allen_reading_days_l804_80405

theorem allen_reading_days (pages_per_day : ℕ) (total_pages : ℕ) 
  (h1 : pages_per_day = 10) (h2 : total_pages = 120) : 
  (total_pages / pages_per_day) = 12 := by
  sorry

end NUMINAMATH_GPT_allen_reading_days_l804_80405


namespace NUMINAMATH_GPT_reservoir_percentage_before_storm_l804_80487

variable (total_capacity : ℝ)
variable (water_after_storm : ℝ := 220 + 110)
variable (percentage_after_storm : ℝ := 0.60)
variable (original_contents : ℝ := 220)

theorem reservoir_percentage_before_storm :
  total_capacity = water_after_storm / percentage_after_storm →
  (original_contents / total_capacity) * 100 = 40 :=
by
  sorry

end NUMINAMATH_GPT_reservoir_percentage_before_storm_l804_80487


namespace NUMINAMATH_GPT_linear_price_item_func_l804_80445

noncomputable def price_item_func (x : ℝ) : Prop :=
  ∃ (y : ℝ), y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200

theorem linear_price_item_func : ∀ x, price_item_func x ↔ (∃ y, y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200) :=
by
  sorry

end NUMINAMATH_GPT_linear_price_item_func_l804_80445


namespace NUMINAMATH_GPT_point_A_2019_pos_l804_80459

noncomputable def A : ℕ → ℤ
| 0       => 2
| (n + 1) =>
    if (n + 1) % 2 = 1 then A n - (n + 1)
    else A n + (n + 1)

theorem point_A_2019_pos : A 2019 = -1008 := by
  sorry

end NUMINAMATH_GPT_point_A_2019_pos_l804_80459


namespace NUMINAMATH_GPT_vanessa_score_record_l804_80484

theorem vanessa_score_record 
  (team_total_points : ℕ) 
  (other_players_average : ℕ) 
  (num_other_players : ℕ) 
  (total_game_points : team_total_points = 55) 
  (average_points_per_player : other_players_average = 4) 
  (number_of_other_players : num_other_players = 7) 
  : 
  ∃ vanessa_points : ℕ, vanessa_points = 27 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_score_record_l804_80484


namespace NUMINAMATH_GPT_books_per_shelf_l804_80470

theorem books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves : ℕ) (books_left : ℕ) (books_per_shelf : ℕ) :
  total_books = 46 →
  books_taken = 10 →
  shelves = 9 →
  books_left = total_books - books_taken →
  books_per_shelf = books_left / shelves →
  books_per_shelf = 4 :=
by
  sorry

end NUMINAMATH_GPT_books_per_shelf_l804_80470


namespace NUMINAMATH_GPT_binom_solution_l804_80434

theorem binom_solution (x y : ℕ) (hxy : x > 0 ∧ y > 0) (bin_eq : Nat.choose x y = 1999000) : x = 1999000 ∨ x = 2000 := 
by
  sorry

end NUMINAMATH_GPT_binom_solution_l804_80434


namespace NUMINAMATH_GPT_expand_expression_l804_80488

theorem expand_expression (x : ℝ) : (2 * x - 3) * (2 * x + 3) * (4 * x ^ 2 + 9) = 4 * x ^ 4 - 81 := by
  sorry

end NUMINAMATH_GPT_expand_expression_l804_80488


namespace NUMINAMATH_GPT_part_I_part_II_l804_80475

noncomputable def f (x : ℝ) := x * (Real.log x - 1) + Real.log x + 1

theorem part_I :
  let f_tangent (x y : ℝ) := x - y - 1
  (∀ x y, f_tangent x y = 0 ↔ y = x - 1) ∧ f_tangent 1 (f 1) = 0 :=
by
  sorry

theorem part_II (m : ℝ) :
  (∀ x : ℝ, x > 0 → x^2 + x * (m - (Real.log x + 1 / x)) + 1 ≥ 0) → m ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l804_80475


namespace NUMINAMATH_GPT_circle_radius_tangent_to_circumcircles_l804_80429

noncomputable def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * (Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))))

theorem circle_radius_tangent_to_circumcircles (AB BC CA : ℝ) (H : Point) 
  (h_AB : AB = 13) (h_BC : BC = 14) (h_CA : CA = 15) : 
  (radius : ℝ) = 65 / 16 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_tangent_to_circumcircles_l804_80429


namespace NUMINAMATH_GPT_interval_x_2x_3x_l804_80460

theorem interval_x_2x_3x (x : ℝ) :
  (2 * x > 1) ∧ (2 * x < 2) ∧ (3 * x > 1) ∧ (3 * x < 2) ↔ (x > 1 / 2) ∧ (x < 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_interval_x_2x_3x_l804_80460


namespace NUMINAMATH_GPT_unique_real_solution_between_consecutive_integers_l804_80426

theorem unique_real_solution_between_consecutive_integers (k : ℕ) (h : k > 0) :
  ∃! x : ℝ, k < x ∧ x < k + 1 ∧ (⌊x⌋ : ℝ) * (x^2 + 1) = x^3 := sorry

end NUMINAMATH_GPT_unique_real_solution_between_consecutive_integers_l804_80426


namespace NUMINAMATH_GPT_find_y_for_slope_l804_80478

theorem find_y_for_slope (y : ℝ) :
  let R := (-3, 9)
  let S := (3, y)
  let slope := (S.2 - R.2) / (S.1 - R.1)
  slope = -2 ↔ y = -3 :=
by
  simp [slope]
  sorry

end NUMINAMATH_GPT_find_y_for_slope_l804_80478


namespace NUMINAMATH_GPT_cost_of_article_l804_80418

-- Conditions as Lean definitions
def price_1 : ℝ := 340
def price_2 : ℝ := 350
def price_diff : ℝ := price_2 - price_1 -- Rs. 10
def gain_percent_increase : ℝ := 0.04

-- Question: What is the cost of the article?
-- Answer: Rs. 90

theorem cost_of_article : ∃ C : ℝ, 
  price_diff = gain_percent_increase * (price_1 - C) ∧ C = 90 := 
sorry

end NUMINAMATH_GPT_cost_of_article_l804_80418


namespace NUMINAMATH_GPT_max_k_guarded_l804_80480

-- Define the size of the board
def board_size : ℕ := 8

-- Define the directions a guard can look
inductive Direction
| up | down | left | right

-- Define a guard's position on the board as a pair of Fin 8
def Position := Fin board_size × Fin board_size

-- Guard record that contains its position and direction
structure Guard where
  pos : Position
  dir : Direction

-- Function to determine if guard A is guarding guard B
def is_guarding (a b : Guard) : Bool :=
  match a.dir with
  | Direction.up    => a.pos.1 < b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.down  => a.pos.1 > b.pos.1 ∧ a.pos.2 = b.pos.2
  | Direction.left  => a.pos.1 = b.pos.1 ∧ a.pos.2 > b.pos.2
  | Direction.right => a.pos.1 = b.pos.1 ∧ a.pos.2 < b.pos.2

-- The main theorem states that the maximum k is 5
theorem max_k_guarded : ∃ k : ℕ, (∀ g : Guard, ∃ S : Finset Guard, (S.card ≥ k) ∧ (∀ s ∈ S, is_guarding s g)) ∧ k = 5 :=
by
  sorry

end NUMINAMATH_GPT_max_k_guarded_l804_80480


namespace NUMINAMATH_GPT_euler_polyhedron_problem_l804_80401

theorem euler_polyhedron_problem
  (V E F : ℕ)
  (t h T H : ℕ)
  (euler_formula : V - E + F = 2)
  (faces_count : F = 30)
  (tri_hex_faces : t + h = 30)
  (edges_equation : E = (3 * t + 6 * h) / 2)
  (vertices_equation1 : V = (3 * t) / T)
  (vertices_equation2 : V = (6 * h) / H)
  (T_val : T = 1)
  (H_val : H = 2)
  (t_val : t = 10)
  (h_val : h = 20)
  (edges_val : E = 75)
  (vertices_val : V = 60) :
  100 * H + 10 * T + V = 270 :=
by
  sorry

end NUMINAMATH_GPT_euler_polyhedron_problem_l804_80401


namespace NUMINAMATH_GPT_problem_solution_l804_80469

noncomputable def f1 (x : ℝ) : ℝ := -2 * x + 2 * Real.sqrt 2
noncomputable def f2 (x : ℝ) : ℝ := Real.sin x
noncomputable def f3 (x : ℝ) : ℝ := x + (1 / x)
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x
noncomputable def f5 (x : ℝ) : ℝ := -2 * Real.log x

def has_inverse_proportion_point (f : ℝ → ℝ) (domain : Set ℝ) : Prop :=
  ∃ x ∈ domain, x * f x = 1

theorem problem_solution :
  (has_inverse_proportion_point f1 univ) ∧
  (has_inverse_proportion_point f2 (Set.Icc 0 (2 * Real.pi))) ∧
  ¬ (has_inverse_proportion_point f3 (Set.Ioi 0)) ∧
  (has_inverse_proportion_point f4 univ) ∧
  ¬ (has_inverse_proportion_point f5 (Set.Ioi 0)) :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l804_80469


namespace NUMINAMATH_GPT_proof_problem_l804_80455

theorem proof_problem (a b c : ℝ) (h : a > b) (h1 : b > c) :
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0) :=
sorry

end NUMINAMATH_GPT_proof_problem_l804_80455


namespace NUMINAMATH_GPT_vectors_coplanar_l804_80450

/-- Vectors defined as 3-dimensional Euclidean space vectors. --/
def vector3 := (ℝ × ℝ × ℝ)

/-- Definitions for vectors a, b, c as given in the problem conditions. --/
def a : vector3 := (3, 1, -1)
def b : vector3 := (1, 0, -1)
def c : vector3 := (8, 3, -2)

/-- The scalar triple product of vectors a, b, c is the determinant of the matrix formed. --/
noncomputable def scalarTripleProduct (u v w : vector3) : ℝ :=
  let (u1, u2, u3) := u
  let (v1, v2, v3) := v
  let (w1, w2, w3) := w
  u1 * (v2 * w3 - v3 * w2) - u2 * (v1 * w3 - v3 * w1) + u3 * (v1 * w2 - v2 * w1)

/-- Statement to prove that vectors a, b, c are coplanar (i.e., their scalar triple product is zero). --/
theorem vectors_coplanar : scalarTripleProduct a b c = 0 :=
  by sorry

end NUMINAMATH_GPT_vectors_coplanar_l804_80450


namespace NUMINAMATH_GPT_find_b_l804_80428

theorem find_b (a b c : ℝ) (h₁ : c = 3)
  (h₂ : -a / 3 = c)
  (h₃ : -a / 3 = 1 + a + b + c) :
  b = -16 :=
by
  -- The solution steps are not necessary to include here.
  sorry

end NUMINAMATH_GPT_find_b_l804_80428


namespace NUMINAMATH_GPT_m_squared_divisible_by_64_l804_80442

theorem m_squared_divisible_by_64 (m : ℕ) (h : 8 ∣ m) : 64 ∣ m * m :=
sorry

end NUMINAMATH_GPT_m_squared_divisible_by_64_l804_80442


namespace NUMINAMATH_GPT_gcd_subtraction_method_gcd_euclidean_algorithm_l804_80474

theorem gcd_subtraction_method (a b : ℕ) (h₁ : a = 72) (h₂ : b = 168) : Int.gcd a b = 24 := by
  sorry

theorem gcd_euclidean_algorithm (a b : ℕ) (h₁ : a = 98) (h₂ : b = 280) : Int.gcd a b = 14 := by
  sorry

end NUMINAMATH_GPT_gcd_subtraction_method_gcd_euclidean_algorithm_l804_80474


namespace NUMINAMATH_GPT_young_member_age_diff_l804_80481

-- Definitions
def A : ℝ := sorry    -- Average age of committee members 4 years ago
def O : ℝ := sorry    -- Age of the old member
def N : ℝ := sorry    -- Age of the new member

-- Hypotheses
axiom avg_same : ∀ (t : ℝ), t = t
axiom replacement : 10 * A + 4 * 10 - 40 = 10 * A

-- Theorem
theorem young_member_age_diff : O - N = 40 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_young_member_age_diff_l804_80481


namespace NUMINAMATH_GPT_polygon_interior_angles_sum_l804_80407

theorem polygon_interior_angles_sum (n : ℕ) (hn : 180 * (n - 2) = 1980) : 180 * (n + 4 - 2) = 2700 :=
by
  sorry

end NUMINAMATH_GPT_polygon_interior_angles_sum_l804_80407


namespace NUMINAMATH_GPT_min_surface_area_of_stacked_solids_l804_80493

theorem min_surface_area_of_stacked_solids :
  ∀ (l w h : ℕ), l = 3 → w = 2 → h = 1 → 
  (2 * (l * w + l * h + w * h) - 2 * l * w = 32) :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  sorry

end NUMINAMATH_GPT_min_surface_area_of_stacked_solids_l804_80493


namespace NUMINAMATH_GPT_students_attended_game_l804_80458

variable (s n : ℕ)

theorem students_attended_game (h1 : s + n = 3000) (h2 : 10 * s + 15 * n = 36250) : s = 1750 := by
  sorry

end NUMINAMATH_GPT_students_attended_game_l804_80458


namespace NUMINAMATH_GPT_max_product_of_two_positive_numbers_l804_80495

theorem max_product_of_two_positive_numbers (x y s : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = s) : 
  x * y ≤ (s ^ 2) / 4 :=
sorry

end NUMINAMATH_GPT_max_product_of_two_positive_numbers_l804_80495


namespace NUMINAMATH_GPT_D_won_zero_matches_l804_80416

-- Define the players
inductive Player
| A | B | C | D deriving DecidableEq

-- Function to determine the winner of a match
def match_winner (p1 p2 : Player) : Option Player :=
  if p1 = Player.A ∧ p2 = Player.D then 
    some Player.A
  else if p2 = Player.A ∧ p1 = Player.D then 
    some Player.A
  else 
    none -- This represents that we do not know the outcome for matches not given

-- Assuming A, B, and C have won the same number of matches
def same_wins (w_A w_B w_C : Nat) : Prop := 
  w_A = w_B ∧ w_B = w_C

-- Define the problem statement
theorem D_won_zero_matches (w_D : Nat) (h_winner_AD: match_winner Player.A Player.D = some Player.A)
  (h_same_wins : ∃ w_A w_B w_C : Nat, same_wins w_A w_B w_C) : w_D = 0 :=
sorry

end NUMINAMATH_GPT_D_won_zero_matches_l804_80416


namespace NUMINAMATH_GPT_find_angle_A_l804_80411

variable {A B C a b c : ℝ}
variable {triangle_ABC : Prop}

theorem find_angle_A
  (h1 : a^2 + c^2 = b^2 + 2 * a * c * Real.cos C)
  (h2 : a = 2 * b * Real.sin A)
  (h3 : Real.cos B = Real.cos C)
  (h_triangle_angles : triangle_ABC) : A = 2 * Real.pi / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_angle_A_l804_80411


namespace NUMINAMATH_GPT_value_of_y_minus_x_l804_80403

theorem value_of_y_minus_x (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
sorry

end NUMINAMATH_GPT_value_of_y_minus_x_l804_80403


namespace NUMINAMATH_GPT_time_to_drain_l804_80467

theorem time_to_drain (V R C : ℝ) (hV : V = 75000) (hR : R = 60) (hC : C = 0.80) : 
  (V * C) / R = 1000 := by
  sorry

end NUMINAMATH_GPT_time_to_drain_l804_80467


namespace NUMINAMATH_GPT_disease_cases_1975_l804_80486

theorem disease_cases_1975 (cases_1950 cases_2000 : ℕ) (cases_1950_eq : cases_1950 = 500000)
  (cases_2000_eq : cases_2000 = 1000) (linear_decrease : ∀ t : ℕ, 1950 ≤ t ∧ t ≤ 2000 →
  ∃ k : ℕ, cases_1950 - (k * (t - 1950)) = cases_2000) : 
  ∃ cases_1975 : ℕ, cases_1975 = 250500 := 
by
  -- Setting up known values
  let decrease_duration := 2000 - 1950
  let total_decrease := cases_1950 - cases_2000
  let annual_decrease := total_decrease / decrease_duration
  let years_from_1950_to_1975 := 1975 - 1950
  let decline_by_1975 := annual_decrease * years_from_1950_to_1975
  let cases_1975 := cases_1950 - decline_by_1975
  -- Returning the desired value
  use cases_1975
  sorry

end NUMINAMATH_GPT_disease_cases_1975_l804_80486


namespace NUMINAMATH_GPT_relationship_between_sums_l804_80423

-- Conditions: four distinct positive integers
variables {a b c d : ℕ}
-- additional conditions: positive integers
variables (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0) (d_pos : d > 0)

-- Condition: a is the largest and d is the smallest
variables (a_largest : a > b ∧ a > c ∧ a > d)
variables (d_smallest : d < b ∧ d < c ∧ d < a)

-- Condition: a / b = c / d
variables (ratio_condition : a * d = b * c)

theorem relationship_between_sums :
  a + d > b + c :=
sorry

end NUMINAMATH_GPT_relationship_between_sums_l804_80423


namespace NUMINAMATH_GPT_find_largest_number_l804_80489

theorem find_largest_number (a b c d e : ℕ)
    (h1 : a + b + c + d = 240)
    (h2 : a + b + c + e = 260)
    (h3 : a + b + d + e = 280)
    (h4 : a + c + d + e = 300)
    (h5 : b + c + d + e = 320)
    (h6 : a + b = 40) :
    max a (max b (max c (max d e))) = 160 := by
  sorry

end NUMINAMATH_GPT_find_largest_number_l804_80489


namespace NUMINAMATH_GPT_problem1_problem2_l804_80439

namespace MathProofs

theorem problem1 : (0.25 * 4 - ((5 / 6) + (1 / 12)) * (6 / 5)) = (1 / 10) := by
  sorry

theorem problem2 : ((5 / 12) - (5 / 16)) * (4 / 5) + (2 / 3) - (3 / 4) = 0 := by
  sorry

end MathProofs

end NUMINAMATH_GPT_problem1_problem2_l804_80439


namespace NUMINAMATH_GPT_perfect_square_polynomial_l804_80436

-- Define the polynomial and the conditions
def polynomial (a b : ℚ) := fun x : ℚ => x^4 + x^3 + 2 * x^2 + a * x + b

-- The expanded form of a quadratic trinomial squared
def quadratic_square (p q : ℚ) := fun x : ℚ =>
  x^4 + 2 * p * x^3 + (p^2 + 2 * q) * x^2 + 2 * p * q * x + q^2

-- Main theorem statement
theorem perfect_square_polynomial :
  ∃ (a b : ℚ), 
  (∀ x : ℚ, polynomial a b x = (quadratic_square (1/2 : ℚ) (7/8 : ℚ) x)) ↔ 
  a = 7/8 ∧ b = 49/64 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_polynomial_l804_80436


namespace NUMINAMATH_GPT_salary_increase_l804_80462

variable (S : ℝ) -- Robert's original salary
variable (P : ℝ) -- Percentage increase after decrease in decimal form

theorem salary_increase (h1 : 0.5 * S * (1 + P) = 0.75 * S) : P = 0.5 := 
by 
  sorry

end NUMINAMATH_GPT_salary_increase_l804_80462


namespace NUMINAMATH_GPT_line_parameterization_l804_80413

theorem line_parameterization (s m : ℝ) :
  (∃ t : ℝ, ∀ x y : ℝ, (x = s + 2 * t ∧ y = 3 + m * t) ↔ y = 5 * x - 7) →
  s = 2 ∧ m = 10 :=
by
  intro h_conditions
  sorry

end NUMINAMATH_GPT_line_parameterization_l804_80413


namespace NUMINAMATH_GPT_twelve_women_reseated_l804_80402

def S (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 3
  else S (n - 1) + S (n - 2) + S (n - 3)

theorem twelve_women_reseated : S 12 = 1201 :=
by
  sorry

end NUMINAMATH_GPT_twelve_women_reseated_l804_80402


namespace NUMINAMATH_GPT_max_marks_l804_80419

variable (M : ℝ)

theorem max_marks (h1 : 0.35 * M = 175) : M = 500 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_max_marks_l804_80419


namespace NUMINAMATH_GPT_scaling_factor_is_2_l804_80457

-- Define the volumes of the original and scaled cubes
def V1 : ℕ := 343
def V2 : ℕ := 2744

-- Assume s1 cubed equals V1 and s2 cubed equals V2
def s1 : ℕ := 7  -- because 7^3 = 343
def s2 : ℕ := 14 -- because 14^3 = 2744

-- Scaling factor between the cubes
def scaling_factor : ℕ := s2 / s1 

-- The theorem stating the scaling factor is 2 given the volumes
theorem scaling_factor_is_2 (h1 : s1 ^ 3 = V1) (h2 : s2 ^ 3 = V2) : scaling_factor = 2 := by
  sorry

end NUMINAMATH_GPT_scaling_factor_is_2_l804_80457


namespace NUMINAMATH_GPT_handed_out_apples_l804_80412

def total_apples : ℤ := 96
def pies : ℤ := 9
def apples_per_pie : ℤ := 6
def apples_for_pies : ℤ := pies * apples_per_pie
def apples_handed_out : ℤ := total_apples - apples_for_pies

theorem handed_out_apples : apples_handed_out = 42 := by
  sorry

end NUMINAMATH_GPT_handed_out_apples_l804_80412


namespace NUMINAMATH_GPT_problem_statement_l804_80485

theorem problem_statement : (-0.125 ^ 2006) * (8 ^ 2005) = -0.125 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l804_80485


namespace NUMINAMATH_GPT_xiaoning_pe_comprehensive_score_l804_80451

def midterm_weight : ℝ := 0.3
def final_weight : ℝ := 0.7
def midterm_score : ℝ := 80
def final_score : ℝ := 90

theorem xiaoning_pe_comprehensive_score : midterm_score * midterm_weight + final_score * final_weight = 87 :=
by
  sorry

end NUMINAMATH_GPT_xiaoning_pe_comprehensive_score_l804_80451


namespace NUMINAMATH_GPT_six_letter_vowel_words_count_l804_80464

noncomputable def vowel_count_six_letter_words : Nat := 27^6

theorem six_letter_vowel_words_count :
  vowel_count_six_letter_words = 531441 :=
  by
    sorry

end NUMINAMATH_GPT_six_letter_vowel_words_count_l804_80464


namespace NUMINAMATH_GPT_smallest_number_l804_80492

-- Definitions of conditions for H, P, and S
def is_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k^3
def is_fifth_power (n : ℕ) : Prop := ∃ k : ℕ, n = k^5
def is_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def satisfies_conditions_H (H : ℕ) : Prop :=
  is_cube (H / 2) ∧ is_fifth_power (H / 3) ∧ is_square (H / 5)

def satisfies_conditions_P (P A B C : ℕ) : Prop :=
  P / 2 = A^2 ∧ P / 3 = B^3 ∧ P / 5 = C^5

def satisfies_conditions_S (S D E F : ℕ) : Prop :=
  S / 2 = D^5 ∧ S / 3 = E^2 ∧ S / 5 = F^3

-- Main statement: Prove that P is the smallest number satisfying the conditions
theorem smallest_number (H P S A B C D E F : ℕ)
  (hH : satisfies_conditions_H H)
  (hP : satisfies_conditions_P P A B C)
  (hS : satisfies_conditions_S S D E F) :
  P ≤ H ∧ P ≤ S :=
  sorry

end NUMINAMATH_GPT_smallest_number_l804_80492


namespace NUMINAMATH_GPT_maximum_profit_l804_80466

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  10.8 - (1/30) * x^2
else
  108 / x - 1000 / (3 * x^2)

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  x * R x - (10 + 2.7 * x)
else
  x * R x - (10 + 2.7 * x)

theorem maximum_profit : 
  ∃ x : ℝ, (0 < x ∧ x ≤ 10 → W x = 8.1 * x - (x^3 / 30) - 10) ∧ 
           (x > 10 → W x = 98 - 1000 / (3 * x) - 2.7 * x) ∧ 
           (∃ xmax : ℝ, xmax = 9 ∧ W 9 = 38.6) := 
sorry

end NUMINAMATH_GPT_maximum_profit_l804_80466


namespace NUMINAMATH_GPT_tan_problem_l804_80452

theorem tan_problem (m : ℝ) (α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end NUMINAMATH_GPT_tan_problem_l804_80452


namespace NUMINAMATH_GPT_point_in_second_quadrant_l804_80443

-- Define the point coordinates in the Cartesian plane
def x_coord : ℤ := -8
def y_coord : ℤ := 2

-- Define the quadrants based on coordinate conditions
def first_quadrant : Prop := x_coord > 0 ∧ y_coord > 0
def second_quadrant : Prop := x_coord < 0 ∧ y_coord > 0
def third_quadrant : Prop := x_coord < 0 ∧ y_coord < 0
def fourth_quadrant : Prop := x_coord > 0 ∧ y_coord < 0

-- Proof statement: The point (-8, 2) lies in the second quadrant
theorem point_in_second_quadrant : second_quadrant :=
by
  sorry

end NUMINAMATH_GPT_point_in_second_quadrant_l804_80443


namespace NUMINAMATH_GPT_seating_solution_l804_80440

/-- 
Imagine Abby, Bret, Carl, and Dana are seated in a row of four seats numbered from 1 to 4.
Joe observes them and declares:

- "Bret is sitting next to Dana" (False)
- "Carl is between Abby and Dana" (False)

Further, it is known that Abby is in seat #2.

Who is seated in seat #3? 
-/

def seating_problem : Prop :=
  ∃ (seats : ℕ → ℕ),
  (¬ (seats 1 = 1 ∧ seats 1 = 4 ∨ seats 4 = 1 ∧ seats 4 = 4)) ∧
  (¬ (seats 3 > seats 1 ∧ seats 3 < seats 2 ∨ seats 3 > seats 2 ∧ seats 3 < seats 1)) ∧
  (seats 2 = 2) →
  (seats 3 = 3)

theorem seating_solution : seating_problem :=
sorry

end NUMINAMATH_GPT_seating_solution_l804_80440


namespace NUMINAMATH_GPT_train_length_is_350_meters_l804_80447

noncomputable def length_of_train (speed_kmh : ℝ) (time_sec : ℝ) : ℝ :=
  let time_hr := time_sec / 3600
  speed_kmh * time_hr * 1000

theorem train_length_is_350_meters :
  length_of_train 60 21 = 350 :=
by
  sorry

end NUMINAMATH_GPT_train_length_is_350_meters_l804_80447


namespace NUMINAMATH_GPT_julie_monthly_salary_l804_80454

def hourly_rate : ℕ := 5
def hours_per_day : ℕ := 8
def days_per_week : ℕ := 6
def missed_days : ℕ := 1
def weeks_per_month : ℕ := 4

theorem julie_monthly_salary :
  (hourly_rate * hours_per_day * (days_per_week - missed_days) * weeks_per_month) = 920 :=
by
  sorry

end NUMINAMATH_GPT_julie_monthly_salary_l804_80454


namespace NUMINAMATH_GPT_algorithm_correct_l804_80494

def algorithm_output (x : Int) : Int :=
  let y := Int.natAbs x
  (2 ^ y) - y

theorem algorithm_correct : 
  algorithm_output (-3) = 5 :=
  by sorry

end NUMINAMATH_GPT_algorithm_correct_l804_80494


namespace NUMINAMATH_GPT_number_of_possible_n_l804_80468

theorem number_of_possible_n :
  ∃ (a : ℕ), (∀ n, (n = a^3) ∧ 
  ((∃ b c : ℕ, b ≠ c ∧ b ≠ a ∧ c ≠ a ∧ a = b * c)) ∧ 
  (a + b + c = 2010) ∧ 
  (a > 0) ∧
  (b > 0) ∧
  (c > 0)) → 
  ∃ (num_n : ℕ), num_n = 2009 :=
  sorry

end NUMINAMATH_GPT_number_of_possible_n_l804_80468


namespace NUMINAMATH_GPT_switches_connections_l804_80420

theorem switches_connections (n k : ℕ) (h1 : n = 30) (h2 : k = 4) : (n * k) / 2 = 60 :=
by
  sorry

end NUMINAMATH_GPT_switches_connections_l804_80420


namespace NUMINAMATH_GPT_solve_x_for_fraction_l804_80415

theorem solve_x_for_fraction :
  ∃ x : ℝ, (3 * x - 15) / 4 = (x + 7) / 3 ∧ x = 14.6 :=
by
  sorry

end NUMINAMATH_GPT_solve_x_for_fraction_l804_80415


namespace NUMINAMATH_GPT_quadratic_conversion_l804_80406

def quadratic_to_vertex_form (x : ℝ) : ℝ := 2 * x^2 - 8 * x - 1

theorem quadratic_conversion :
  (∀ x : ℝ, quadratic_to_vertex_form x = 2 * (x - 2)^2 - 9) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_conversion_l804_80406


namespace NUMINAMATH_GPT_divides_by_3_l804_80441

theorem divides_by_3 (a b c : ℕ) (h : 9 ∣ a ^ 3 + b ^ 3 + c ^ 3) : 3 ∣ a ∨ 3 ∣ b ∨ 3 ∣ c :=
sorry

end NUMINAMATH_GPT_divides_by_3_l804_80441


namespace NUMINAMATH_GPT_cubic_roots_proof_l804_80446

noncomputable def cubic_roots_reciprocal (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : ℝ :=
  (1 / a^2) + (1 / b^2) + (1 / c^2)

theorem cubic_roots_proof (a b c : ℝ) (h1 : a + b + c = 7) 
  (h2 : a * b + b * c + c * a = 3) (h3 : a * b * c = -4) : 
  cubic_roots_reciprocal a b c h1 h2 h3 = 65 / 16 :=
sorry

end NUMINAMATH_GPT_cubic_roots_proof_l804_80446


namespace NUMINAMATH_GPT_solve_eqn_l804_80410

theorem solve_eqn (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  3 ^ x = 2 ^ x * y + 1 ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) := by
  sorry

end NUMINAMATH_GPT_solve_eqn_l804_80410


namespace NUMINAMATH_GPT_B_plus_C_is_330_l804_80432

-- Definitions
def A : ℕ := 170
def B : ℕ := 300
def C : ℕ := 30

axiom h1 : A + B + C = 500
axiom h2 : A + C = 200
axiom h3 : C = 30

-- Theorem statement
theorem B_plus_C_is_330 : B + C = 330 :=
by
  sorry

end NUMINAMATH_GPT_B_plus_C_is_330_l804_80432


namespace NUMINAMATH_GPT_faye_age_l804_80433

def ages (C D E F : ℕ) :=
  D = E - 2 ∧
  E = C + 3 ∧
  F = C + 4 ∧
  D = 15

theorem faye_age (C D E F : ℕ) (h : ages C D E F) : F = 18 :=
by
  unfold ages at h
  sorry

end NUMINAMATH_GPT_faye_age_l804_80433


namespace NUMINAMATH_GPT_part_a_l804_80491

theorem part_a (α β : ℝ) (h₁ : α = 1.0000000004) (h₂ : β = 1.00000000002) (h₃ : α > β) :
  2.00000000002 / (β * β + 2.00000000002) > 2.00000000004 / α := 
sorry

end NUMINAMATH_GPT_part_a_l804_80491


namespace NUMINAMATH_GPT_lowest_possible_price_l804_80490

theorem lowest_possible_price
  (MSRP : ℕ) (max_initial_discount_percent : ℕ) (platinum_discount_percent : ℕ)
  (h1 : MSRP = 35) (h2 : max_initial_discount_percent = 40) (h3 : platinum_discount_percent = 30) :
  let initial_discount := max_initial_discount_percent * MSRP / 100
  let price_after_initial_discount := MSRP - initial_discount
  let platinum_discount := platinum_discount_percent * price_after_initial_discount / 100
  let lowest_price := price_after_initial_discount - platinum_discount
  lowest_price = 147 / 10 :=
by
  sorry

end NUMINAMATH_GPT_lowest_possible_price_l804_80490


namespace NUMINAMATH_GPT_calculation_correct_l804_80438

def f (x : ℚ) := (2 * x^2 + 6 * x + 9) / (x^2 + 3 * x + 5)
def g (x : ℚ) := 2 * x + 1

theorem calculation_correct : f (g 2) + g (f 2) = 308 / 45 := by
  sorry

end NUMINAMATH_GPT_calculation_correct_l804_80438


namespace NUMINAMATH_GPT_fraction_area_below_diagonal_is_one_l804_80404

noncomputable def fraction_below_diagonal (s : ℝ) : ℝ := 1

theorem fraction_area_below_diagonal_is_one (s : ℝ) :
  let long_side := 2 * s
  let P := (2 * s / 3, 0)
  let Q := (s, s / 2)
  -- Total area of the rectangle
  let total_area := s * 2 * s -- 2s^2
  -- Total area below the diagonal
  let area_below_diagonal := 2 * s * s  -- 2s^2
  -- Fraction of the area below diagonal
  fraction_below_diagonal s = area_below_diagonal / total_area := 
by 
  sorry

end NUMINAMATH_GPT_fraction_area_below_diagonal_is_one_l804_80404


namespace NUMINAMATH_GPT_pins_after_one_month_l804_80465

def avg_pins_per_day : ℕ := 10
def delete_pins_per_week_per_person : ℕ := 5
def group_size : ℕ := 20
def initial_pins : ℕ := 1000

theorem pins_after_one_month
  (avg_pins_per_day_pos : avg_pins_per_day = 10)
  (delete_pins_per_week_per_person_pos : delete_pins_per_week_per_person = 5)
  (group_size_pos : group_size = 20)
  (initial_pins_pos : initial_pins = 1000) : 
  1000 + (avg_pins_per_day * group_size * 30) - (delete_pins_per_week_per_person * group_size * 4) = 6600 :=
by
  sorry

end NUMINAMATH_GPT_pins_after_one_month_l804_80465


namespace NUMINAMATH_GPT_machine_tasks_l804_80483

theorem machine_tasks (y : ℕ) 
  (h1 : (1 : ℚ)/(y + 4) + (1 : ℚ)/(y + 3) + (1 : ℚ)/(4 * y) = (1 : ℚ)/y) : y = 1 :=
sorry

end NUMINAMATH_GPT_machine_tasks_l804_80483


namespace NUMINAMATH_GPT_increasing_sequence_a1_range_l804_80431

theorem increasing_sequence_a1_range
  (a : ℕ → ℝ) 
  (h : ∀ n, a (n + 1) = (4 * a n - 2) / (a n + 1))
  (strictly_increasing : ∀ n, a (n + 1) > a n) :
  1 < a 1 ∧ a 1 < 2 :=
sorry

end NUMINAMATH_GPT_increasing_sequence_a1_range_l804_80431


namespace NUMINAMATH_GPT_part1_part2_l804_80472

open Set Real

def A := {x : ℝ | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem part1 (m : ℝ) (h : Disjoint A (B m)) : m ∈ Iio 2 ∪ Ioi 4 := 
sorry

theorem part2 (m : ℝ) (h : A ∪ (univ \ (B m)) = univ) : m ∈ Iic 3 := 
sorry

end NUMINAMATH_GPT_part1_part2_l804_80472


namespace NUMINAMATH_GPT_not_equiv_2_pi_six_and_11_pi_six_l804_80437

def polar_equiv (r θ₁ θ₂ : ℝ) : Prop :=
  ∃ k : ℤ, θ₂ = θ₁ + 2 * ↑k * Real.pi

theorem not_equiv_2_pi_six_and_11_pi_six :
  ¬ polar_equiv 2 (Real.pi / 6) (11 * Real.pi / 6) := 
sorry

end NUMINAMATH_GPT_not_equiv_2_pi_six_and_11_pi_six_l804_80437


namespace NUMINAMATH_GPT_prob_all_pass_prob_at_least_one_pass_most_likely_event_l804_80435

noncomputable def probability_A := 2 / 5
noncomputable def probability_B := 3 / 4
noncomputable def probability_C := 1 / 3
noncomputable def prob_none_pass := (1 - probability_A) * (1 - probability_B) * (1 - probability_C)
noncomputable def prob_one_pass := 
  (probability_A * (1 - probability_B) * (1 - probability_C)) +
  ((1 - probability_A) * probability_B * (1 - probability_C)) +
  ((1 - probability_A) * (1 - probability_B) * probability_C)
noncomputable def prob_two_pass := 
  (probability_A * probability_B * (1 - probability_C)) +
  (probability_A * (1 - probability_B) * probability_C) +
  ((1 - probability_A) * probability_B * probability_C)

-- Prove that the probability that all three candidates pass is 1/10
theorem prob_all_pass : probability_A * probability_B * probability_C = 1 / 10 := by
  sorry

-- Prove that the probability that at least one candidate passes is 9/10
theorem prob_at_least_one_pass : 1 - prob_none_pass = 9 / 10 := by
  sorry

-- Prove that the most likely event of passing is exactly one candidate passing with probability 5/12
theorem most_likely_event : prob_one_pass > prob_two_pass ∧ prob_one_pass > probability_A * probability_B * probability_C ∧ prob_one_pass > prob_none_pass ∧ prob_one_pass = 5 / 12 := by
  sorry

end NUMINAMATH_GPT_prob_all_pass_prob_at_least_one_pass_most_likely_event_l804_80435


namespace NUMINAMATH_GPT_box_volume_increase_l804_80496

-- Conditions
def volume (l w h : ℝ) : ℝ := l * w * h
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + w * h + h * l)
def sum_of_edges (l w h : ℝ) : ℝ := 4 * (l + w + h)

-- The main theorem we want to state
theorem box_volume_increase
  (l w h : ℝ)
  (h_volume : volume l w h = 5000)
  (h_surface_area : surface_area l w h = 1800)
  (h_sum_of_edges : sum_of_edges l w h = 210) :
  volume (l + 2) (w + 2) (h + 2) = 7018 := 
by sorry

end NUMINAMATH_GPT_box_volume_increase_l804_80496


namespace NUMINAMATH_GPT_inequality_proof_l804_80473

theorem inequality_proof (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) : 
  (a / b + b / c + c / a) ^ 2 ≥ 3 * (a / c + c / b + b / a) :=
  sorry

end NUMINAMATH_GPT_inequality_proof_l804_80473
