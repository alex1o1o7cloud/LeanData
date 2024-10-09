import Mathlib

namespace quadratic_range_m_l1828_182828

theorem quadratic_range_m (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m * x + 1 = 0 ∧ y^2 + m * y + 1 = 0) ↔ (m < -2 ∨ m > 2) :=
by 
  sorry

end quadratic_range_m_l1828_182828


namespace vertices_of_square_l1828_182886

-- Define lattice points as points with integer coordinates
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Define the distance between two lattice points
def distance (P Q : LatticePoint) : ℤ :=
  (P.x - Q.x) * (P.x - Q.x) + (P.y - Q.y) * (P.y - Q.y)

-- Define the area of a triangle formed by three lattice points using the determinant method
def area (P Q R : LatticePoint) : ℤ :=
  (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)

-- Prove that three distinct lattice points form the vertices of a square given the condition
theorem vertices_of_square (P Q R : LatticePoint) (h₀ : P ≠ Q) (h₁ : Q ≠ R) (h₂ : P ≠ R)
    (h₃ : (distance P Q + distance Q R) < 8 * (area P Q R) + 1) :
    ∃ S : LatticePoint, S ≠ P ∧ S ≠ Q ∧ S ≠ R ∧
    (distance P Q = distance Q R ∧ distance Q R = distance R S ∧ distance R S = distance S P) := 
by sorry

end vertices_of_square_l1828_182886


namespace f_expression_when_x_gt_1_l1828_182859

variable (f : ℝ → ℝ)

-- conditions
def f_even : Prop := ∀ x, f (x + 1) = f (-x + 1)
def f_defn_when_x_lt_1 : Prop := ∀ x, x < 1 → f x = x ^ 2 + 1

-- theorem to prove
theorem f_expression_when_x_gt_1 (h_even : f_even f) (h_defn : f_defn_when_x_lt_1 f) : 
  ∀ x, x > 1 → f x = x ^ 2 - 4 * x + 5 := 
by
  sorry

end f_expression_when_x_gt_1_l1828_182859


namespace stamps_problem_l1828_182857

def largest_common_divisor (a b c : ℕ) : ℕ :=
  gcd (gcd a b) c

theorem stamps_problem :
  largest_common_divisor 1020 1275 1350 = 15 :=
by
  sorry

end stamps_problem_l1828_182857


namespace geometric_series_sum_l1828_182810

theorem geometric_series_sum :
  let a := (1 / 2 : ℝ)
  let r := (1 / 2 : ℝ)
  let n := 6
  (a * (1 - r^n) / (1 - r)) = (63 / 64 : ℝ) := 
by 
  sorry

end geometric_series_sum_l1828_182810


namespace min_value_64_l1828_182801

noncomputable def min_value_expr (a b c d e f g h : ℝ) : ℝ :=
  (a * e) ^ 2 + (b * f) ^ 2 + (c * g) ^ 2 + (d * h) ^ 2

theorem min_value_64 
  (a b c d e f g h : ℝ) 
  (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16)
  (h3 : a + b + c + d = e * f * g) :
  min_value_expr a b c d e f g h = 64 := 
sorry

end min_value_64_l1828_182801


namespace find_constant_k_l1828_182885

theorem find_constant_k 
  (k : ℝ)
  (h : ∀ x : ℝ, -x^2 - (k + 10)*x - 8 = -(x - 2)*(x - 4)) : 
  k = -16 :=
sorry

end find_constant_k_l1828_182885


namespace find_value_of_squares_l1828_182864

-- Defining the conditions
variable (a b c : ℝ)
variable (h1 : a^2 + 3 * b = 10)
variable (h2 : b^2 + 5 * c = 0)
variable (h3 : c^2 + 7 * a = -21)

-- Stating the theorem to prove the desired result
theorem find_value_of_squares : a^2 + b^2 + c^2 = 83 / 4 :=
   sorry

end find_value_of_squares_l1828_182864


namespace volume_of_orange_concentrate_l1828_182809

theorem volume_of_orange_concentrate
  (h_jug : ℝ := 8) -- height of the jug in inches
  (d_jug : ℝ := 3) -- diameter of the jug in inches
  (fraction_full : ℝ := 3 / 4) -- jug is three-quarters full
  (ratio_concentrate_to_water : ℝ := 1 / 5) -- ratio of concentrate to water
  : abs ((fraction_full * π * ((d_jug / 2)^2) * h_jug * (1 / (1 + ratio_concentrate_to_water))) - 2.25) < 0.01 :=
by
  sorry

end volume_of_orange_concentrate_l1828_182809


namespace son_age_l1828_182862

theorem son_age {S M : ℕ} 
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2)) : 
  S = 35 :=
by sorry

end son_age_l1828_182862


namespace problem1_solutions_problem2_solutions_l1828_182876

-- Problem 1: Solve x² - 7x + 6 = 0

theorem problem1_solutions (x : ℝ) : 
  x^2 - 7 * x + 6 = 0 ↔ (x = 1 ∨ x = 6) := by
  sorry

-- Problem 2: Solve (2x + 3)² = (x - 3)² 

theorem problem2_solutions (x : ℝ) : 
  (2 * x + 3)^2 = (x - 3)^2 ↔ (x = 0 ∨ x = -6) := by
  sorry

end problem1_solutions_problem2_solutions_l1828_182876


namespace inequality_proof_l1828_182882

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_geq : a ≥ b ∧ b ≥ c)
  (h_sum : a + b + c ≤ 1) :
  a^2 + 3 * b^2 + 5 * c^2 ≤ 1 := 
sorry

end inequality_proof_l1828_182882


namespace product_of_roots_of_quadratics_l1828_182871

noncomputable def product_of_roots : ℝ :=
  let r1 := 2021 / 2020
  let r2 := 2020 / 2019
  let r3 := 2019
  r1 * r2 * r3

theorem product_of_roots_of_quadratics (b : ℝ) 
  (h1 : ∃ x1 x2 : ℝ, 2020 * x1 * x1 + b * x1 + 2021 = 0 ∧ 2020 * x2 * x2 + b * x2 + 2021 = 0) 
  (h2 : ∃ y1 y2 : ℝ, 2019 * y1 * y1 + b * y1 + 2020 = 0 ∧ 2019 * y2 * y2 + b * y2 + 2020 = 0) 
  (h3 : ∃ z1 z2 : ℝ, z1 * z1 + b * z1 + 2019 = 0 ∧ z1 * z1 + b * z2 + 2019 = 0) :
  product_of_roots = 2021 :=
by
  sorry

end product_of_roots_of_quadratics_l1828_182871


namespace domain_of_function_l1828_182858

/-- The domain of the function \( y = \lg (12 + x - x^2) \) is the interval \(-3 < x < 4\). -/
theorem domain_of_function :
  {x : ℝ | 12 + x - x^2 > 0} = {x : ℝ | -3 < x ∧ x < 4} :=
sorry

end domain_of_function_l1828_182858


namespace reciprocal_inverse_proportional_l1828_182815

variable {x y k c : ℝ}

-- Given condition: x * y = k
axiom inverse_proportional (h : x * y = k) : ∃ c, (1/x) * (1/y) = c

theorem reciprocal_inverse_proportional (h : x * y = k) :
  ∃ c, (1/x) * (1/y) = c :=
inverse_proportional h

end reciprocal_inverse_proportional_l1828_182815


namespace find_x_to_print_800_leaflets_in_3_minutes_l1828_182881

theorem find_x_to_print_800_leaflets_in_3_minutes (x : ℝ) :
  (800 / 12 + 800 / x = 800 / 3) → (1 / 12 + 1 / x = 1 / 3) :=
by
  intro h
  have h1 : 800 / 12 = 200 / 3 := by norm_num
  have h2 : 800 / 3 = 800 / 3 := by norm_num
  sorry

end find_x_to_print_800_leaflets_in_3_minutes_l1828_182881


namespace solve_for_y_l1828_182819

theorem solve_for_y (y : ℝ) (h : 3 * y ^ (1 / 4) - 3 * y ^ (1 / 2) / y ^ (1 / 4) = 13 - 2 * y ^ (1 / 4)) :
  y = (13 / 2) ^ 4 :=
by sorry

end solve_for_y_l1828_182819


namespace diagram_is_knowledge_structure_l1828_182896

inductive DiagramType
| ProgramFlowchart
| ProcessFlowchart
| KnowledgeStructureDiagram
| OrganizationalStructureDiagram

axiom given_diagram : DiagramType
axiom diagram_is_one_of_them : 
  given_diagram = DiagramType.ProgramFlowchart ∨ 
  given_diagram = DiagramType.ProcessFlowchart ∨ 
  given_diagram = DiagramType.KnowledgeStructureDiagram ∨ 
  given_diagram = DiagramType.OrganizationalStructureDiagram

theorem diagram_is_knowledge_structure :
  given_diagram = DiagramType.KnowledgeStructureDiagram :=
sorry

end diagram_is_knowledge_structure_l1828_182896


namespace polyhedron_edges_vertices_l1828_182895

theorem polyhedron_edges_vertices (F : ℕ) (triangular_faces : Prop) (hF : F = 20) : ∃ S A : ℕ, S = 12 ∧ A = 30 :=
by
  -- stating the problem conditions and desired conclusion
  sorry

end polyhedron_edges_vertices_l1828_182895


namespace min_value_18_solve_inequality_l1828_182807

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (1/a^3) + (1/b^3) + (1/c^3) + 27 * a * b * c

theorem min_value_18 (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) :
  min_value a b c ≥ 18 :=
by sorry

theorem solve_inequality (x : ℝ) :
  abs (x + 1) - 2 * x < 18 ↔ x > -(19/3) :=
by sorry

end min_value_18_solve_inequality_l1828_182807


namespace windmere_zoo_two_legged_birds_l1828_182813

theorem windmere_zoo_two_legged_birds (b m u : ℕ) (head_count : b + m + u = 300) (leg_count : 2 * b + 4 * m + 3 * u = 710) : b = 230 :=
sorry

end windmere_zoo_two_legged_birds_l1828_182813


namespace max_sides_of_convex_polygon_with_4_obtuse_l1828_182861

theorem max_sides_of_convex_polygon_with_4_obtuse (n : ℕ) (hn : n ≥ 3) :
  (∃ k : ℕ, k = 4 ∧
    ∀ θ : Fin n → ℝ, 
      (∀ p, θ p > 90 ∧ ∃ t, θ t = 180 ∨ θ t < 90 ∨ θ t = 90) →
      4 = k →
      n ≤ 7
  ) :=
sorry

end max_sides_of_convex_polygon_with_4_obtuse_l1828_182861


namespace triangle_angles_l1828_182835

theorem triangle_angles (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 45) : B = 90 ∧ C = 45 :=
sorry

end triangle_angles_l1828_182835


namespace trucks_needed_l1828_182889

-- Definitions of the conditions
def total_apples : ℕ := 80
def apples_transported : ℕ := 56
def truck_capacity : ℕ := 4

-- Definition to calculate the remaining apples
def remaining_apples : ℕ := total_apples - apples_transported

-- The theorem statement
theorem trucks_needed : remaining_apples / truck_capacity = 6 := by
  sorry

end trucks_needed_l1828_182889


namespace find_n_between_50_and_150_l1828_182845

theorem find_n_between_50_and_150 :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧ 
  n % 9 = 3 ∧ 
  n % 6 = 3 ∧ 
  n % 4 = 1 ∧
  n = 105 :=
by
  sorry

end find_n_between_50_and_150_l1828_182845


namespace smaller_number_is_25_l1828_182827

theorem smaller_number_is_25 (x y : ℕ) (h1 : x + y = 62) (h2 : y = x + 12) : x = 25 :=
by sorry

end smaller_number_is_25_l1828_182827


namespace reassemble_into_square_conditions_l1828_182854

noncomputable def graph_paper_figure : Type := sorry
noncomputable def is_cuttable_into_parts (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def all_parts_are_triangles (figure : graph_paper_figure) (parts : ℕ) : Prop := sorry
noncomputable def can_reassemble_to_square (figure : graph_paper_figure) : Prop := sorry

theorem reassemble_into_square_conditions :
  ∀ (figure : graph_paper_figure), 
  (is_cuttable_into_parts figure 4 ∧ can_reassemble_to_square figure) ∧ 
  (is_cuttable_into_parts figure 5 ∧ all_parts_are_triangles figure 5 ∧ can_reassemble_to_square figure) :=
sorry

end reassemble_into_square_conditions_l1828_182854


namespace product_of_two_numbers_l1828_182833

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 6) (h2 : x^2 + y^2 = 170) : x * y = -67 := 
by 
  sorry

end product_of_two_numbers_l1828_182833


namespace geometric_sequence_sum_x_l1828_182825

variable {α : Type*} [Field α]

theorem geometric_sequence_sum_x (a : ℕ → α) (S : ℕ → α) (x : α) 
  (h₁ : ∀ n, S n = x * (3:α)^n + 1)
  (h₂ : ∀ n, a n = S n - S (n - 1)) :
  ∃ x, x = -1 :=
by
  let a1 := S 1
  let a2 := S 2 - S 1
  let a3 := S 3 - S 2
  have ha1 : a1 = 3 * x + 1 := sorry
  have ha2 : a2 = 6 * x := sorry
  have ha3 : a3 = 18 * x := sorry
  have h_geom : (6 * x)^2 = (3 * x + 1) * 18 * x := sorry
  have h_solve : 18 * x * (x + 1) = 0 := sorry
  have h_x_neg1 : x = 0 ∨ x = -1 := sorry
  exact ⟨-1, sorry⟩

end geometric_sequence_sum_x_l1828_182825


namespace avg_scores_relation_l1828_182817

variables (class_avg top8_avg other32_avg : ℝ)

theorem avg_scores_relation (h1 : 40 = 40) 
  (h2 : top8_avg = class_avg + 3) :
  other32_avg = top8_avg - 3.75 :=
sorry

end avg_scores_relation_l1828_182817


namespace range_of_a_l1828_182851

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a - 1) * x < a - 1 ↔ x > 1) : a < 1 := 
sorry

end range_of_a_l1828_182851


namespace shoveling_problem_l1828_182841

variable (S : ℝ) -- Wayne's son's shoveling rate (driveways per hour)
variable (W : ℝ) -- Wayne's shoveling rate (driveways per hour)
variable (T : ℝ) -- Time it takes for Wayne's son to shovel the driveway alone (hours)

theorem shoveling_problem 
  (h1 : W = 6 * S)
  (h2 : (S + W) * 3 = 1) : T = 21 := 
by
  sorry

end shoveling_problem_l1828_182841


namespace Sally_next_birthday_age_l1828_182878

variables (a m s d : ℝ)

def Adam_older_than_Mary := a = 1.3 * m
def Mary_younger_than_Sally := m = 0.75 * s
def Sally_younger_than_Danielle := s = 0.8 * d
def Sum_ages := a + m + s + d = 60

theorem Sally_next_birthday_age (a m s d : ℝ) 
  (H1 : Adam_older_than_Mary a m)
  (H2 : Mary_younger_than_Sally m s)
  (H3 : Sally_younger_than_Danielle s d)
  (H4 : Sum_ages a m s d) : 
  s + 1 = 16 := 
by sorry

end Sally_next_birthday_age_l1828_182878


namespace upper_bound_of_third_inequality_l1828_182816

variable (x : ℤ)

theorem upper_bound_of_third_inequality : (3 < x ∧ x < 10) →
                                          (5 < x ∧ x < 18) →
                                          (∃ n, n > x ∧ x > -2) →
                                          (0 < x ∧ x < 8) →
                                          (x + 1 < 9) →
                                          x < 8 :=
by { sorry }

end upper_bound_of_third_inequality_l1828_182816


namespace tan_of_obtuse_angle_l1828_182894

theorem tan_of_obtuse_angle (α : ℝ) (h_cos : Real.cos α = -1/2) (h_obtuse : π/2 < α ∧ α < π) :
  Real.tan α = -Real.sqrt 3 :=
sorry

end tan_of_obtuse_angle_l1828_182894


namespace cost_of_article_l1828_182856

theorem cost_of_article (C G1 G2 : ℝ) (h1 : G1 = 380 - C) (h2 : G2 = 450 - C) (h3 : G2 = 1.10 * G1) : 
  C = 320 :=
by
  sorry

end cost_of_article_l1828_182856


namespace sum_last_two_digits_pow_mod_eq_zero_l1828_182842

/-
Given condition: 
Sum of the last two digits of \( 9^{25} + 11^{25} \)
-/
theorem sum_last_two_digits_pow_mod_eq_zero : 
  let a := 9
  let b := 11
  let n := 25 
  (a ^ n + b ^ n) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_pow_mod_eq_zero_l1828_182842


namespace circle_diameter_eq_l1828_182869

-- Definitions
def line (x y : ℝ) : Prop := 3 * x - 4 * y + 12 = 0
def point_A (x y : ℝ) : Prop := x = 0 ∧ y = 3
def point_B (x y : ℝ) : Prop := x = -4 ∧ y = 0
def midpoint_AB (x y : ℝ) : Prop := x = -2 ∧ y = 3 / 2 -- Midpoint of A(0,3) and B(-4,0)
def diameter_AB : ℝ := 5

-- The equation of the circle with diameter AB
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 3 * y = 0

-- The proof statement
theorem circle_diameter_eq :
  (∃ A B : ℝ × ℝ, point_A A.1 A.2 ∧ point_B B.1 B.2 ∧ 
                   midpoint_AB ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧ diameter_AB = 5) →
  (∀ x y : ℝ, circle_eq x y) :=
sorry

end circle_diameter_eq_l1828_182869


namespace function_bounded_in_interval_l1828_182866

variables {f : ℝ → ℝ}

theorem function_bounded_in_interval (h : ∀ x y : ℝ, x > y → f x ^ 2 ≤ f y) : ∀ x : ℝ, 0 ≤ f x ∧ f x ≤ 1 :=
by
  sorry

end function_bounded_in_interval_l1828_182866


namespace probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l1828_182837

def num_faces : ℕ := 6
def possible_outcomes : ℕ := num_faces * num_faces

def count_odd_sum_outcomes : ℕ := 18 -- From solution steps
def probability_odd_sum : ℚ := count_odd_sum_outcomes / possible_outcomes

def count_2x_plus_y_less_than_10 : ℕ := 14 -- From solution steps
def probability_2x_plus_y_less_than_10 : ℚ := count_2x_plus_y_less_than_10 / possible_outcomes

theorem probability_odd_sum_is_one_half :
  probability_odd_sum = 1 / 2 :=
sorry

theorem probability_2x_plus_y_less_than_10_is_seven_eighteenths :
  probability_2x_plus_y_less_than_10 = 7 / 18 :=
sorry

end probability_odd_sum_is_one_half_probability_2x_plus_y_less_than_10_is_seven_eighteenths_l1828_182837


namespace N_vector_3_eq_result_vector_l1828_182872

noncomputable def matrix_N : Matrix (Fin 2) (Fin 2) ℝ :=
-- The matrix N is defined such that:
-- N * (vector 3 -2) = (vector 4 1)
-- N * (vector -2 3) = (vector 1 2)
sorry

def vector_1 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 3 | ⟨1,_⟩ => -2
def vector_2 : Fin 2 → ℝ := fun | ⟨0,_⟩ => -2 | ⟨1,_⟩ => 3
def vector_3 : Fin 2 → ℝ := fun | ⟨0,_⟩ => 7 | ⟨1,_⟩ => 0
def result_vector : Fin 2 → ℝ := fun | ⟨0,_⟩ => 14 | ⟨1,_⟩ => 7

theorem N_vector_3_eq_result_vector :
  matrix_N.mulVec vector_3 = result_vector := by
  -- Given conditions:
  -- matrix_N.mulVec vector_1 = vector_4
  -- and matrix_N.mulVec vector_2 = vector_5
  sorry

end N_vector_3_eq_result_vector_l1828_182872


namespace problem_ineq_l1828_182831

theorem problem_ineq (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
(h4 : x * y * z = 1) :
    (x^3 / ((1 + y)*(1 + z)) + y^3 / ((1 + z)*(1 + x)) + z^3 / ((1 + x)*(1 + y))) ≥ 3 / 4 := 
sorry

end problem_ineq_l1828_182831


namespace sum_of_first_and_fourth_l1828_182849

theorem sum_of_first_and_fourth (x : ℤ) (h : x + (x + 6) = 156) : (x + 2) = 77 :=
by {
  -- This block represents the assumptions and goal as expressed above,
  -- but the proof steps are omitted.
  sorry
}

end sum_of_first_and_fourth_l1828_182849


namespace line_parallel_to_plane_l1828_182848

-- Defining conditions
def vector_a : ℝ × ℝ × ℝ := (1, -1, 3)
def vector_n : ℝ × ℝ × ℝ := (0, 3, 1)

-- Lean theorem statement
theorem line_parallel_to_plane : 
  let ⟨a1, a2, a3⟩ := vector_a;
  let ⟨n1, n2, n3⟩ := vector_n;
  a1 * n1 + a2 * n2 + a3 * n3 = 0 :=
by 
  -- Proof omitted
  sorry

end line_parallel_to_plane_l1828_182848


namespace find_angle_C_find_area_triangle_l1828_182800

open Real

-- Let the angles and sides of the triangle be defined as follows
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
axiom condition1 : (a^2 + b^2 - c^2) * (tan C) = sqrt 2 * a * b
axiom condition2 : c = 2
axiom condition3 : b = 2 * sqrt 2

-- Proof statements
theorem find_angle_C :
  C = pi / 4 ∨ C = 3 * pi / 4 :=
sorry

theorem find_area_triangle :
  C = pi / 4 → a = 2 → (1 / 2) * a * b * sin C = 2 :=
sorry

end find_angle_C_find_area_triangle_l1828_182800


namespace a4_value_l1828_182820

axiom a_n : ℕ → ℝ
axiom S_n : ℕ → ℝ
axiom q : ℝ

-- Conditions
axiom a1_eq_1 : a_n 1 = 1
axiom S6_eq_4S3 : S_n 6 = 4 * S_n 3
axiom q_ne_1 : q ≠ 1

-- Arithmetic Sequence Sum Formula
axiom sum_formula : ∀ n, S_n n = (1 - q^n) / (1 - q)

-- nth-term Formula
axiom nth_term_formula : ∀ n, a_n n = a_n 1 * q^(n - 1)

-- Prove the value of the 4th term
theorem a4_value : a_n 4 = 3 := sorry

end a4_value_l1828_182820


namespace quadratic_eq_real_roots_l1828_182852

theorem quadratic_eq_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4 * x + 2 = 0) →
  (∃ y : ℝ, a * y^2 - 4 * y + 2 = 0) →
  a ≤ 2 ∧ a ≠ 0 :=
by sorry

end quadratic_eq_real_roots_l1828_182852


namespace middle_digit_zero_l1828_182855

theorem middle_digit_zero (a b c M : ℕ) (h1 : M = 36 * a + 6 * b + c) (h2 : M = 64 * a + 8 * b + c) (ha : 0 ≤ a ∧ a < 6) (hb : 0 ≤ b ∧ b < 6) (hc : 0 ≤ c ∧ c < 6) : 
  b = 0 := 
  by sorry

end middle_digit_zero_l1828_182855


namespace pancakes_needed_l1828_182836

theorem pancakes_needed (initial_pancakes : ℕ) (num_people : ℕ) (pancakes_left : ℕ) :
  initial_pancakes = 12 → num_people = 8 → pancakes_left = initial_pancakes - num_people →
  (num_people - pancakes_left) = 4 :=
by
  intros initial_pancakes_eq num_people_eq pancakes_left_eq
  sorry

end pancakes_needed_l1828_182836


namespace correct_weight_of_misread_boy_l1828_182821

variable (num_boys : ℕ) (avg_weight_incorrect : ℝ) (misread_weight : ℝ) (avg_weight_correct : ℝ)

theorem correct_weight_of_misread_boy
  (h1 : num_boys = 20)
  (h2 : avg_weight_incorrect = 58.4)
  (h3 : misread_weight = 56)
  (h4 : avg_weight_correct = 58.6) : 
  misread_weight + (num_boys * avg_weight_correct - num_boys * avg_weight_incorrect) / num_boys = 60 := 
by 
  -- skipping proof
  sorry

end correct_weight_of_misread_boy_l1828_182821


namespace sum_of_n_with_unformable_postage_120_equals_43_l1828_182860

theorem sum_of_n_with_unformable_postage_120_equals_43 :
  ∃ n1 n2 : ℕ, n1 = 21 ∧ n2 = 22 ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n1 * b + (n1 + 1) * c) ∧ 
  (∀ k : ℕ, k > 120 → ∃ a b c : ℕ, k = 7 * a + n2 * b + (n2 + 1) * c) ∧ 
  (120 = 7 * a + n1 * b + (n1 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (120 = 7 * a + n2 * b + (n2 + 1) * c → a = 0 ∧ b = 0 ∧ c = 0) ∧
  (n1 + n2 = 43) :=
by
  sorry

end sum_of_n_with_unformable_postage_120_equals_43_l1828_182860


namespace xy_sum_l1828_182802

theorem xy_sum (x y : ℝ) (h1 : 2 / x + 3 / y = 4) (h2 : 2 / x - 3 / y = -2) : x + y = 3 := by
  sorry

end xy_sum_l1828_182802


namespace quadratic_function_proof_l1828_182850

theorem quadratic_function_proof (a c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = a * x^2 - 4 * x + c)
  (h_sol_set : ∀ x, f x < 0 → (-1 < x ∧ x < 5)) :
  (a = 1 ∧ c = -5) ∧ (∀ x, 0 ≤ x ∧ x ≤ 3 → -9 ≤ f x ∧ f x ≤ -5) :=
by
  sorry

end quadratic_function_proof_l1828_182850


namespace solution_set_of_inequality_l1828_182824

theorem solution_set_of_inequality (x : ℝ) (h : (2 * x - 1) / x < 0) : 0 < x ∧ x < 1 / 2 :=
by
  sorry

end solution_set_of_inequality_l1828_182824


namespace infinitely_many_positive_integers_l1828_182879

theorem infinitely_many_positive_integers (k : ℕ) (m := 13 * k + 1) (h : m ≠ 8191) :
  8191 = 2 ^ 13 - 1 → ∃ (m : ℕ), ∀ k : ℕ, (13 * k + 1) ≠ 8191 ∧ ∃ (t : ℕ), (2 ^ (13 * k) - 1) = 8191 * m * t := by
  intros
  sorry

end infinitely_many_positive_integers_l1828_182879


namespace orange_slices_needed_l1828_182814

theorem orange_slices_needed (total_slices containers_capacity leftover_slices: ℕ) 
(h1 : containers_capacity = 4) 
(h2 : total_slices = 329) 
(h3 : leftover_slices = 1) :
    containers_capacity - leftover_slices = 3 :=
by
  sorry

end orange_slices_needed_l1828_182814


namespace prob_geometry_given_algebra_l1828_182880

variable (algebra geometry : ℕ) (total : ℕ)

/-- Proof of the probability of selecting a geometry question on the second draw,
    given that an algebra question is selected on the first draw. -/
theorem prob_geometry_given_algebra : 
  algebra = 3 ∧ geometry = 2 ∧ total = 5 →
  (algebra / (total : ℚ)) * (geometry / (total - 1 : ℚ)) = 1 / 2 :=
by
  intro h
  sorry

end prob_geometry_given_algebra_l1828_182880


namespace cans_of_beans_is_two_l1828_182803

-- Define the problem parameters
variable (C B T : ℕ)

-- Conditions based on the problem statement
axiom chili_can : C = 1
axiom tomato_to_bean_ratio : T = 3 * B / 2
axiom quadruple_batch_cans : 4 * (C + B + T) = 24

-- Prove the number of cans of beans is 2
theorem cans_of_beans_is_two : B = 2 :=
by
  -- Include conditions
  have h1 : C = 1 := by sorry
  have h2 : T = 3 * B / 2 := by sorry
  have h3 : 4 * (C + B + T) = 24 := by sorry
  -- Derive the answer (Proof omitted)
  sorry

end cans_of_beans_is_two_l1828_182803


namespace decreasing_interval_implies_a_ge_two_l1828_182818

-- The function f is given
def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2 * a * x - 3

-- Defining the condition for f(x) being decreasing in the interval (-8, 2)
def is_decreasing_in_interval (a : ℝ) : Prop :=
  ∀ x y : ℝ, (-8 < x ∧ x < y ∧ y < 2) → f x a > f y a

-- The proof statement
theorem decreasing_interval_implies_a_ge_two (a : ℝ) (h : is_decreasing_in_interval a) : a ≥ 2 :=
sorry

end decreasing_interval_implies_a_ge_two_l1828_182818


namespace sufficient_not_necessary_condition_l1828_182844

theorem sufficient_not_necessary_condition (a : ℝ) : (a = 2 → (a^2 - a) * 1 + 1 = 0) ∧ (¬ ((a^2 - a) * 1 + 1 = 0 → a = 2)) :=
by sorry

end sufficient_not_necessary_condition_l1828_182844


namespace fraction_simplification_l1828_182888

theorem fraction_simplification :
  (3 / 7 + 4 / 5) / (5 / 12 + 2 / 3) = 516 / 455 := by
  sorry

end fraction_simplification_l1828_182888


namespace initial_volume_of_mixture_l1828_182877

-- Define the initial condition volumes for p and q
def initial_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x)

-- Define the final condition volumes for p and q after adding 2 liters of q
def final_volumes (x : ℕ) : ℕ × ℕ := (3 * x, 2 * x + 2)

-- Define the initial total volume of the mixture
def initial_volume (x : ℕ) : ℕ := 5 * x

-- The theorem stating the solution
theorem initial_volume_of_mixture (x : ℕ) (h : 3 * x / (2 * x + 2) = 5 / 4) : 5 * x = 25 := 
by sorry

end initial_volume_of_mixture_l1828_182877


namespace no_real_roots_contradiction_l1828_182804

open Real

variables (a b : ℝ)

theorem no_real_roots_contradiction (h : ∀ x : ℝ, a * x^3 + a * x + b ≠ 0) : false :=
by
  sorry

end no_real_roots_contradiction_l1828_182804


namespace sales_on_second_day_l1828_182863

variable (m : ℕ)

-- Define the condition for sales on the first day
def first_day_sales : ℕ := m

-- Define the condition for sales on the second day
def second_day_sales : ℕ := 2 * first_day_sales m - 3

-- The proof statement
theorem sales_on_second_day (m : ℕ) : second_day_sales m = 2 * m - 3 := by
  -- provide the actual proof here
  sorry

end sales_on_second_day_l1828_182863


namespace root_expr_value_eq_175_div_11_l1828_182843

noncomputable def root_expr_value (a b c : ℝ) (h1 : a + b + c = 15) (h2 : ab + bc + ca = 25) (h3 : abc = 10) : ℝ :=
  (a / (1 / a + b * c)) + (b / (1 / b + c * a)) + (c / (1 / c + a * b))

theorem root_expr_value_eq_175_div_11 (a b c : ℝ) 
  (h1 : a + b + c = 15) 
  (h2 : ab + bc + ca = 25) 
  (h3 : abc = 10) : 
  root_expr_value a b c h1 h2 h3 = 175 / 11 := 
sorry

end root_expr_value_eq_175_div_11_l1828_182843


namespace binary_to_octal_101110_l1828_182826

theorem binary_to_octal_101110 : 
  ∀ (binary_to_octal : ℕ → ℕ), 
  binary_to_octal 0b101110 = 0o56 :=
by
  sorry

end binary_to_octal_101110_l1828_182826


namespace tickets_to_be_sold_l1828_182868

theorem tickets_to_be_sold : 
  let total_tickets := 200
  let jude_tickets := 16
  let andrea_tickets := 4 * jude_tickets
  let sandra_tickets := 2 * jude_tickets + 8
  total_tickets - (jude_tickets + andrea_tickets + sandra_tickets) = 80 := by
  sorry

end tickets_to_be_sold_l1828_182868


namespace fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l1828_182867

def positive_integers_up_to (n : ℕ) : List ℕ :=
  List.range' 1 n

def divisible_by_lcm (lcm : ℕ) (lst : List ℕ) : List ℕ :=
  lst.filter (λ x => x % lcm = 0)

noncomputable def fraction_divisible_by_both (n a b : ℕ) : ℚ :=
  let lcm_ab := Nat.lcm a b
  let elems := positive_integers_up_to n
  let divisible_elems := divisible_by_lcm lcm_ab elems
  divisible_elems.length / n

theorem fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25 :
  fraction_divisible_by_both 100 3 4 = (2 : ℚ) / 25 :=
by
  sorry

end fraction_of_integers_divisible_by_3_and_4_up_to_100_eq_2_over_25_l1828_182867


namespace algebra_expression_value_l1828_182829

theorem algebra_expression_value
  (x y : ℝ)
  (h : x - 2 * y + 2 = 5) : 4 * y - 2 * x + 1 = -5 :=
by sorry

end algebra_expression_value_l1828_182829


namespace difference_of_interests_l1828_182823

def investment_in_funds (X Y : ℝ) (total_investment : ℝ) : ℝ := X + Y
def interest_earned (investment_rate : ℝ) (amount : ℝ) : ℝ := investment_rate * amount

variable (X : ℝ) (Y : ℝ)
variable (total_investment : ℝ) (rate_X : ℝ) (rate_Y : ℝ)
variable (investment_X : ℝ) 

axiom h1 : total_investment = 100000
axiom h2 : rate_X = 0.23
axiom h3 : rate_Y = 0.17
axiom h4 : investment_X = 42000
axiom h5 : investment_in_funds X Y total_investment = total_investment - investment_X

-- We need to show the difference in interest is 200
theorem difference_of_interests : 
  let interest_X := interest_earned rate_X investment_X
  let investment_Y := total_investment - investment_X
  let interest_Y := interest_earned rate_Y investment_Y
  interest_Y - interest_X = 200 :=
by
  sorry

end difference_of_interests_l1828_182823


namespace time_to_pass_trolley_l1828_182892

/--
Conditions:
- Length of the train = 110 m
- Speed of the train = 60 km/hr
- Speed of the trolley = 12 km/hr

Prove that the time it takes for the train to pass the trolley completely is 5.5 seconds.
-/
theorem time_to_pass_trolley :
  ∀ (train_length : ℝ) (train_speed_kmh : ℝ) (trolley_speed_kmh : ℝ),
    train_length = 110 →
    train_speed_kmh = 60 →
    trolley_speed_kmh = 12 →
  train_length / ((train_speed_kmh + trolley_speed_kmh) * (1000 / 3600)) = 5.5 :=
by
  intros
  sorry

end time_to_pass_trolley_l1828_182892


namespace tan_alpha_is_neg_5_over_12_l1828_182887

variables (α : ℝ) (h1 : Real.sin α = 5/13) (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_is_neg_5_over_12 : Real.tan α = -5/12 :=
by
  sorry

end tan_alpha_is_neg_5_over_12_l1828_182887


namespace container_dimensions_l1828_182890

theorem container_dimensions (a b c : ℝ) 
  (h1 : a * b * 16 = 2400)
  (h2 : a * c * 10 = 2400)
  (h3 : b * c * 9.6 = 2400) :
  a = 12 ∧ b = 12.5 ∧ c = 20 :=
by
  sorry

end container_dimensions_l1828_182890


namespace tan_shift_symmetric_l1828_182893

theorem tan_shift_symmetric :
  let f (x : ℝ) := Real.tan (2 * x + Real.pi / 6)
  let g (x : ℝ) := f (x + Real.pi / 6)
  g (Real.pi / 4) = 0 ∧ ∀ x, g (Real.pi / 2 - x) = -g (Real.pi / 2 + x) :=
by
  sorry

end tan_shift_symmetric_l1828_182893


namespace students_spring_outing_l1828_182875

theorem students_spring_outing (n : ℕ) (h1 : n = 5) : 2^n = 32 :=
  by {
    sorry
  }

end students_spring_outing_l1828_182875


namespace milan_billed_minutes_l1828_182891

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l1828_182891


namespace green_balls_count_l1828_182874

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def yellow_balls : ℕ := 2
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def probability_neither_red_nor_purple : ℝ := 0.7

theorem green_balls_count (G : ℕ) :
  (white_balls + G + yellow_balls) / total_balls = probability_neither_red_nor_purple →
  G = 18 := 
by
  sorry

end green_balls_count_l1828_182874


namespace find_room_length_l1828_182808

theorem find_room_length (w : ℝ) (A : ℝ) (h_w : w = 8) (h_A : A = 96) : (A / w = 12) :=
by
  rw [h_w, h_A]
  norm_num

end find_room_length_l1828_182808


namespace variance_scaled_data_l1828_182834

noncomputable def variance (data : List ℝ) : ℝ :=
  let n := data.length
  let mean := data.sum / n
  (data.map (λ x => (x - mean) ^ 2)).sum / n

theorem variance_scaled_data (data : List ℝ) (h_len : data.length > 0) (h_var : variance data = 4) :
  variance (data.map (λ x => 2 * x)) = 16 :=
by
  sorry

end variance_scaled_data_l1828_182834


namespace guesthouse_rolls_probability_l1828_182830

theorem guesthouse_rolls_probability :
  let rolls := 12
  let guests := 3
  let types := 4
  let rolls_per_guest := 3
  let total_probability : ℚ := (12 / 12) * (9 / 11) * (6 / 10) * (3 / 9) *
                               (8 / 8) * (6 / 7) * (4 / 6) * (2 / 5) *
                               1
  let simplified_probability : ℚ := 24 / 1925
  total_probability = simplified_probability := sorry

end guesthouse_rolls_probability_l1828_182830


namespace total_sign_up_methods_l1828_182870

theorem total_sign_up_methods (n : ℕ) (k : ℕ) (h1 : n = 4) (h2 : k = 2) :
  k ^ n = 16 :=
by
  rw [h1, h2]
  norm_num

end total_sign_up_methods_l1828_182870


namespace find_a_l1828_182832

variable (a : ℝ)

def average_condition (a : ℝ) : Prop :=
  ((2 * a + 16) + (3 * a - 8)) / 2 = 74

theorem find_a (h: average_condition a) : a = 28 :=
  sorry

end find_a_l1828_182832


namespace integer_roots_of_quadratic_l1828_182806

theorem integer_roots_of_quadratic (a : ℤ) : 
  (∃ x : ℤ , x^2 + a * x + a = 0) ↔ (a = 0 ∨ a = 4) := 
sorry

end integer_roots_of_quadratic_l1828_182806


namespace ratio_of_areas_of_concentric_circles_eq_9_over_4_l1828_182898

theorem ratio_of_areas_of_concentric_circles_eq_9_over_4
  (C1 C2 : ℝ)
  (h1 : ∃ Q : ℝ, true) -- Existence of point Q
  (h2 : (30 / 360) * C1 = (45 / 360) * C2) -- Arcs formed by 30-degree and 45-degree angles are equal in length
  : (π * (C1 / (2 * π))^2) / (π * (C2 / (2 * π))^2) = 9 / 4 :=
by
  sorry

end ratio_of_areas_of_concentric_circles_eq_9_over_4_l1828_182898


namespace arabella_total_learning_time_l1828_182838

-- Define the conditions
def arabella_first_step_time := 30 -- in minutes
def arabella_second_step_time := arabella_first_step_time / 2 -- half the time of the first step
def arabella_third_step_time := arabella_first_step_time + arabella_second_step_time -- sum of the first and second steps

-- Define the total time spent
def arabella_total_time := arabella_first_step_time + arabella_second_step_time + arabella_third_step_time

-- The theorem to prove
theorem arabella_total_learning_time : arabella_total_time = 90 := 
  sorry

end arabella_total_learning_time_l1828_182838


namespace find_a_in_terms_of_x_l1828_182840

variable (a b x : ℝ)

theorem find_a_in_terms_of_x (h1 : a ≠ b) (h2 : a^3 - b^3 = 27 * x^3) (h3 : a - b = 3 * x) : a = 3 * x :=
sorry

end find_a_in_terms_of_x_l1828_182840


namespace number_of_triangles_fitting_in_square_l1828_182812

-- Define the conditions for the right triangle and the square
def right_triangle_height := 2
def right_triangle_width := 2
def square_side := 2

-- Define the areas
def area_triangle := (1 / 2) * right_triangle_height * right_triangle_width
def area_square := square_side * square_side

-- Define the proof statement to show the number of right triangles fitting in the square is 2
theorem number_of_triangles_fitting_in_square : (area_square / area_triangle) = 2 := by
  sorry

end number_of_triangles_fitting_in_square_l1828_182812


namespace train_speed_l1828_182899

theorem train_speed 
  (t1 : ℝ) (t2 : ℝ) (L : ℝ) (v : ℝ) 
  (h1 : t1 = 12) 
  (h2 : t2 = 44) 
  (h3 : L = v * 12)
  (h4 : L + 320 = v * 44) : 
  (v * 3.6 = 36) :=
by
  sorry

end train_speed_l1828_182899


namespace sunset_duration_l1828_182873

theorem sunset_duration (changes : ℕ) (interval : ℕ) (total_changes : ℕ) (h1 : total_changes = 12) (h2 : interval = 10) : ∃ hours : ℕ, hours = 2 :=
by
  sorry

end sunset_duration_l1828_182873


namespace prime_consecutive_fraction_equivalence_l1828_182846

theorem prime_consecutive_fraction_equivalence (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hq_p_consec : p + 1 ≤ q ∧ Nat.Prime (p + 1) -> p + 1 = q) (hpq : p < q) (frac_eq : p / q = 4 / 5) :
  25 / 7 + (2 * q - p) / (2 * q + p) = 4 := sorry

end prime_consecutive_fraction_equivalence_l1828_182846


namespace people_left_line_l1828_182805

-- Definitions based on the conditions given in the problem
def initial_people := 7
def new_people := 8
def final_people := 11

-- Proof statement
theorem people_left_line (L : ℕ) (h : 7 - L + 8 = 11) : L = 4 :=
by
  -- Adding the proof steps directly skips to the required proof
  sorry

end people_left_line_l1828_182805


namespace shaina_keeps_chocolate_l1828_182865

theorem shaina_keeps_chocolate :
  let total_chocolate := (60 : ℚ) / 7
  let number_of_piles := 5
  let weight_per_pile := total_chocolate / number_of_piles
  let given_weight_back := (1 / 2) * weight_per_pile
  let kept_weight := weight_per_pile - given_weight_back
  kept_weight = 6 / 7 :=
by
  sorry

end shaina_keeps_chocolate_l1828_182865


namespace fill_tank_time_l1828_182822

theorem fill_tank_time (hA : ∀ t : Real, t > 0 → (t / 10) = 1) 
                       (hB : ∀ t : Real, t > 0 → (t / 20) = 1) 
                       (hC : ∀ t : Real, t > 0 → (t / 30) = 1) : 
                       (60 / 7 : Real) = 60 / 7 :=
by
    sorry

end fill_tank_time_l1828_182822


namespace feet_per_inch_of_model_l1828_182897

def height_of_statue := 75 -- in feet
def height_of_model := 5 -- in inches

theorem feet_per_inch_of_model : (height_of_statue / height_of_model) = 15 :=
by
  sorry

end feet_per_inch_of_model_l1828_182897


namespace multiplicative_inverse_CD_mod_1000000_l1828_182883

theorem multiplicative_inverse_CD_mod_1000000 :
  let C := 123456
  let D := 166666
  let M := 48
  M * (C * D) % 1000000 = 1 := by
  sorry

end multiplicative_inverse_CD_mod_1000000_l1828_182883


namespace solution_inequality_1_solution_inequality_2_l1828_182847

theorem solution_inequality_1 (x : ℝ) : -x^2 + 4*x + 5 < 0 ↔ (x < -1 ∨ x > 5) :=
by sorry

theorem solution_inequality_2 (x : ℝ) : 2*x^2 - 5*x + 2 ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by sorry

end solution_inequality_1_solution_inequality_2_l1828_182847


namespace exists_pos_integers_l1828_182811

theorem exists_pos_integers (r : ℚ) (hr : r > 0) : 
  ∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ r = (a^3 + b^3) / (c^3 + d^3) :=
by sorry

end exists_pos_integers_l1828_182811


namespace wizard_viable_combinations_l1828_182884

def wizard_combination_problem : Prop :=
  let total_combinations := 4 * 6
  let incompatible_combinations := 3
  let viable_combinations := total_combinations - incompatible_combinations
  viable_combinations = 21

theorem wizard_viable_combinations : wizard_combination_problem :=
by
  sorry

end wizard_viable_combinations_l1828_182884


namespace number_of_cartons_of_pencils_l1828_182853

theorem number_of_cartons_of_pencils (P E : ℕ) 
  (h1 : P + E = 100) 
  (h2 : 6 * P + 3 * E = 360) : 
  P = 20 := 
by
  sorry

end number_of_cartons_of_pencils_l1828_182853


namespace find_a_tangent_to_curve_l1828_182839

theorem find_a_tangent_to_curve (a : ℝ) :
  (∃ (x₀ : ℝ), y = x - 1 ∧ y = e^(x + a) ∧ (e^(x₀ + a) = 1)) → a = -2 :=
by
  sorry

end find_a_tangent_to_curve_l1828_182839
