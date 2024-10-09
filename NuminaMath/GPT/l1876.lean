import Mathlib

namespace arithmetic_sequence_common_difference_l1876_187662

theorem arithmetic_sequence_common_difference 
  (a1 a2 a3 a4 d : ℕ)
  (S : ℕ → ℕ)
  (h1 : S 2 = a1 + a2)
  (h2 : S 4 = a1 + a2 + a3 + a4)
  (h3 : S 2 = 4)
  (h4 : S 4 = 20)
  (h5 : a2 = a1 + d)
  (h6 : a3 = a2 + d)
  (h7 : a4 = a3 + d) :
  d = 3 :=
by
  sorry

end arithmetic_sequence_common_difference_l1876_187662


namespace customer_payment_eq_3000_l1876_187640

theorem customer_payment_eq_3000 (cost_price : ℕ) (markup_percentage : ℕ) (payment : ℕ)
  (h1 : cost_price = 2500)
  (h2 : markup_percentage = 20)
  (h3 : payment = cost_price + (markup_percentage * cost_price / 100)) :
  payment = 3000 :=
by
  sorry

end customer_payment_eq_3000_l1876_187640


namespace cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l1876_187664

theorem cosine_theorem_a (a b c A : ℝ) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A := sorry

theorem cosine_theorem_b (a b c B : ℝ) :
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B := sorry

theorem cosine_theorem_c (a b c C : ℝ) :
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C := sorry

end cosine_theorem_a_cosine_theorem_b_cosine_theorem_c_l1876_187664


namespace diff_12_358_7_2943_l1876_187630

theorem diff_12_358_7_2943 : 12.358 - 7.2943 = 5.0637 :=
by
  -- Proof is not required, so we put sorry
  sorry

end diff_12_358_7_2943_l1876_187630


namespace drop_volume_l1876_187628

theorem drop_volume :
  let leak_rate := 3 -- drops per minute
  let pot_volume := 3 * 1000 -- volume in milliliters
  let time := 50 -- minutes
  let total_drops := leak_rate * time -- total number of drops
  (pot_volume / total_drops) = 20 := 
by
  let leak_rate : ℕ := 3
  let pot_volume : ℕ := 3 * 1000
  let time : ℕ := 50
  let total_drops := leak_rate * time
  have h : (pot_volume / total_drops) = 20 := by sorry
  exact h

end drop_volume_l1876_187628


namespace line_a_minus_b_l1876_187602

theorem line_a_minus_b (a b : ℝ)
  (h1 : (2 : ℝ) = a * (3 : ℝ) + b)
  (h2 : (26 : ℝ) = a * (7 : ℝ) + b) :
  a - b = 22 :=
by
  sorry

end line_a_minus_b_l1876_187602


namespace find_integer_x_l1876_187698

theorem find_integer_x (x : ℤ) :
  1 < x ∧ x < 9 ∧ 
  2 < x ∧ x < 15 ∧ 
  0 < x ∧ x < 7 ∧ 
  0 < x ∧ x < 4 ∧ 
  x + 1 < 5 
  → x = 3 :=
by
  intros h
  sorry

end find_integer_x_l1876_187698


namespace solution_correct_l1876_187629

noncomputable def a := 3 + 3 * Real.sqrt 2
noncomputable def b := 3 - 3 * Real.sqrt 2

theorem solution_correct (h : a ≥ b) : 3 * a + 2 * b = 15 + 3 * Real.sqrt 2 :=
by sorry

end solution_correct_l1876_187629


namespace sum_of_sides_l1876_187675

-- Definitions: Given conditions
def ratio (a b c : ℕ) : Prop := 
a * 5 = b * 3 ∧ b * 7 = c * 5

-- Given that the longest side is 21 cm and the ratio of the sides is 3:5:7
def similar_triangle (x y : ℕ) : Prop :=
ratio x y 21

-- Proof statement: The sum of the lengths of the other two sides is 24 cm
theorem sum_of_sides (x y : ℕ) (h : similar_triangle x y) : x + y = 24 :=
sorry

end sum_of_sides_l1876_187675


namespace factor_polynomial_l1876_187624

theorem factor_polynomial (x y z : ℤ) :
  x * (y - z) ^ 3 + y * (z - x) ^ 3 + z * (x - y) ^ 3 = (x - y) * (y - z) * (z - x) * (x + y + z) := 
by
  sorry

end factor_polynomial_l1876_187624


namespace symmetric_y_axis_l1876_187685

-- Definition of a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Definition of point symmetry with respect to the y-axis
def symmetric_about_y_axis (M : Point3D) : Point3D := 
  { x := -M.x, y := M.y, z := -M.z }

-- Theorem statement: proving the symmetry
theorem symmetric_y_axis (M : Point3D) : 
  symmetric_about_y_axis M = { x := -M.x, y := M.y, z := -M.z } := by
  sorry  -- Proof is left out as per instruction.

end symmetric_y_axis_l1876_187685


namespace collinear_points_sum_l1876_187690

theorem collinear_points_sum (a b : ℝ) 
  (h_collin: ∃ k : ℝ, 
    (1 - a) / (a - a) = k * (a - b) / (b - b) ∧
    (a - a) / (2 - b) = k * (2 - 3) / (3 - 3) ∧
    (a - b) / (3 - 3) = k * (a - a) / (3 - b) ) : 
  a + b = 4 :=
by
  sorry

end collinear_points_sum_l1876_187690


namespace subtract_and_convert_l1876_187668

theorem subtract_and_convert : (3/4 - 1/16 : ℚ) = 0.6875 :=
by
  sorry

end subtract_and_convert_l1876_187668


namespace parallel_vectors_x_value_l1876_187643

def vec (a b : ℝ) : ℝ × ℝ := (a, b)

theorem parallel_vectors_x_value (x : ℝ) :
  ∀ k : ℝ,
  k ≠ 0 ∧ k * 1 = -2 ∧ k * -2 = x →
  x = 4 :=
by
  intros k hk
  have hk1 : k * 1 = -2 := hk.2.1
  have hk2 : k * -2 = x := hk.2.2
  -- Proceed from here to the calculations according to the steps in b):
  sorry

end parallel_vectors_x_value_l1876_187643


namespace distance_from_mo_l1876_187636

-- Definitions based on conditions
-- 1. Grid squares have side length 1 cm.
-- 2. Shape shaded gray on the grid.
-- 3. The total shaded area needs to be divided into two equal parts.
-- 4. The line to be drawn is parallel to line MO.

noncomputable def grid_side_length : ℝ := 1.0
noncomputable def shaded_area : ℝ := 10.0
noncomputable def line_mo_distance (d : ℝ) : Prop := 
  ∃ parallel_line_distance, parallel_line_distance = d ∧ 
    ∃ equal_area, 2 * equal_area = shaded_area ∧ equal_area = 5.0

-- Theorem: The parallel line should be drawn at 2.6 cm 
theorem distance_from_mo (d : ℝ) : 
  d = 2.6 ↔ line_mo_distance d := 
by
  sorry

end distance_from_mo_l1876_187636


namespace minimum_value_of_f_ge_7_l1876_187614

noncomputable def f (x : ℝ) : ℝ :=
  x + (2 * x) / (x^2 + 1) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem minimum_value_of_f_ge_7 {x : ℝ} (hx : x > 0) : f x ≥ 7 := 
by
  sorry

end minimum_value_of_f_ge_7_l1876_187614


namespace min_value_fraction_l1876_187676

variable (a b : ℝ)
variable (h1 : 2 * a - 2 * b + 2 = 0) -- This corresponds to a + b = 1 based on the given center (-1, 2)
variable (ha : a > 0)
variable (hb : b > 0)

theorem min_value_fraction (h1 : a + b = 1) (ha : a > 0) (hb : b > 0) : 
  (4 / a) + (1 / b) ≥ 9 :=
  sorry

end min_value_fraction_l1876_187676


namespace last_person_is_knight_l1876_187603

-- Definitions for the conditions:
def first_whispered_number := 7
def last_announced_number_first_game := 3
def last_whispered_number_second_game := 5
def first_announced_number_second_game := 2

-- Definitions to represent the roles:
inductive Role
| knight
| liar

-- Definition of the last person in the first game being a knight:
def last_person_first_game_role := Role.knight

theorem last_person_is_knight 
  (h1 : Role.liar = Role.liar)
  (h2 : last_announced_number_first_game = 3)
  (h3 : first_whispered_number = 7)
  (h4 : first_announced_number_second_game = 2)
  (h5 : last_whispered_number_second_game = 5) :
  last_person_first_game_role = Role.knight :=
sorry

end last_person_is_knight_l1876_187603


namespace frog_return_prob_A_after_2022_l1876_187658

def initial_prob_A : ℚ := 1
def transition_prob_A_to_adj : ℚ := 1/3
def transition_prob_adj_to_A : ℚ := 1/3
def transition_prob_adj_to_adj : ℚ := 2/3

noncomputable def prob_A_return (n : ℕ) : ℚ :=
if (n % 2 = 0) then
  (2/9) * (1/2^(n/2)) + (1/9)
else
  0

theorem frog_return_prob_A_after_2022 : prob_A_return 2022 = (2/9) * (1/2^1010) + (1/9) :=
by
  sorry

end frog_return_prob_A_after_2022_l1876_187658


namespace shifted_graph_sum_l1876_187607

theorem shifted_graph_sum :
  let f (x : ℝ) := 3*x^2 - 2*x + 8
  let g (x : ℝ) := f (x - 6)
  let a := 3
  let b := -38
  let c := 128
  a + b + c = 93 :=
by
  sorry

end shifted_graph_sum_l1876_187607


namespace car_production_total_l1876_187677

theorem car_production_total (northAmericaCars europeCars : ℕ) (h1 : northAmericaCars = 3884) (h2 : europeCars = 2871) : northAmericaCars + europeCars = 6755 := by
  sorry

end car_production_total_l1876_187677


namespace tetrahedron_edge_length_l1876_187638

-- Definitions corresponding to the conditions of the problem.
def radius : ℝ := 2

def diameter : ℝ := 2 * radius

/-- Centers of four mutually tangent balls -/
def center_distance : ℝ := diameter

/-- The side length of the square formed by the centers of four balls on the floor. -/
def side_length_of_square : ℝ := center_distance

/-- The edge length of the tetrahedron circumscribed around the four balls. -/
def edge_length_tetrahedron : ℝ := side_length_of_square

-- The statement to be proved.
theorem tetrahedron_edge_length :
  edge_length_tetrahedron = 4 :=
by
  sorry  -- Proof to be constructed

end tetrahedron_edge_length_l1876_187638


namespace multiplication_sequence_result_l1876_187625

theorem multiplication_sequence_result : (1 * 3 * 5 * 7 * 9 * 11 = 10395) :=
by
  sorry

end multiplication_sequence_result_l1876_187625


namespace geometric_sum_proof_l1876_187645

theorem geometric_sum_proof (S : ℕ → ℝ) (a : ℕ → ℝ) (r : ℝ) (n : ℕ)
    (hS3 : S 3 = 8) (hS6 : S 6 = 7)
    (Sn_def : ∀ n, S n = a 0 * (1 - r ^ n) / (1 - r)) :
  a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = -7 / 8 :=
by
  sorry

end geometric_sum_proof_l1876_187645


namespace find_angle_B_l1876_187697

theorem find_angle_B (a b c A B C : ℝ)
  (h1 : a * Real.cos B - b * Real.cos A = c)
  (h2 : C = π / 5)
  (h3 : 0 < A) (h4 : A < π)
  (h5 : 0 < B) (h6 : B < π)
  (h7 : 0 < C) (h8 : C < π)
  (h_triangle : A + B + C = π) :
  B = 3 * π / 10 :=
sorry

end find_angle_B_l1876_187697


namespace derivative_log_base2_l1876_187631

noncomputable def log_base2 (x : ℝ) := Real.log x / Real.log 2

theorem derivative_log_base2 (x : ℝ) (h : x > 0) : 
  deriv (fun x => log_base2 x) x = 1 / (x * Real.log 2) :=
by
  sorry

end derivative_log_base2_l1876_187631


namespace intersection_y_condition_l1876_187655

theorem intersection_y_condition (a : ℝ) :
  (∃ x y : ℝ, 2 * x - a * y + 2 = 0 ∧ x + y = 0 ∧ y < 0) → a < -2 :=
by
  sorry

end intersection_y_condition_l1876_187655


namespace science_books_initially_l1876_187684

def initial_number_of_books (borrowed left : ℕ) : ℕ := 
borrowed + left

theorem science_books_initially (borrowed left : ℕ) (h1 : borrowed = 18) (h2 : left = 57) :
initial_number_of_books borrowed left = 75 := by
sorry

end science_books_initially_l1876_187684


namespace number_of_groups_l1876_187695

noncomputable def original_students : ℕ := 22 + 2

def students_per_group : ℕ := 8

theorem number_of_groups : original_students / students_per_group = 3 :=
by
  sorry

end number_of_groups_l1876_187695


namespace heidi_more_nail_polishes_l1876_187683

theorem heidi_more_nail_polishes :
  ∀ (k h r : ℕ), 
    k = 12 ->
    r = k - 4 ->
    h + r = 25 ->
    h - k = 5 :=
by
  intros k h r hk hr hr_sum
  sorry

end heidi_more_nail_polishes_l1876_187683


namespace union_of_A_and_B_l1876_187679

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l1876_187679


namespace equation_of_line_l1876_187615

-- Define the points P and Q
def P : (ℝ × ℝ) := (3, 2)
def Q : (ℝ × ℝ) := (4, 7)

-- Prove that the equation of the line passing through points P and Q is 5x - y - 13 = 0
theorem equation_of_line : ∃ (A B C : ℝ), A = 5 ∧ B = -1 ∧ C = -13 ∧
  ∀ x y : ℝ, (y - 2) / (7 - 2) = (x - 3) / (4 - 3) → 5 * x - y - 13 = 0 :=
by
  sorry

end equation_of_line_l1876_187615


namespace symmetric_point_correct_l1876_187604

-- Define the point and the symmetry operation
structure Point :=
  (x : ℝ)
  (y : ℝ)

def symmetric_with_respect_to_x_axis (p : Point) : Point :=
  {x := p.x, y := -p.y}

-- Define the specific point M
def M : Point := {x := 1, y := 2}

-- Define the expected answer point M'
def M' : Point := {x := 1, y := -2}

-- Prove that the symmetric point with respect to the x-axis is as expected
theorem symmetric_point_correct :
  symmetric_with_respect_to_x_axis M = M' :=
by sorry

end symmetric_point_correct_l1876_187604


namespace triangle_area_is_96_l1876_187652

-- Definitions of radii and sides being congruent
def tangent_circles (radius1 radius2 : ℝ) : Prop :=
  ∃ (O O' : ℝ × ℝ), dist O O' = radius1 + radius2

-- Given conditions
def radius_small : ℝ := 2
def radius_large : ℝ := 4
def sides_congruent (AB AC : ℝ) : Prop :=
  AB = AC

-- Theorem stating the goal
theorem triangle_area_is_96 
  (O O' : ℝ × ℝ)
  (AB AC : ℝ)
  (circ_tangent : tangent_circles radius_small radius_large)
  (sides_tangent : sides_congruent AB AC) :
  ∃ (BC : ℝ), ∃ (AF : ℝ), (1/2) * BC * AF = 96 := 
by
  sorry

end triangle_area_is_96_l1876_187652


namespace boat_license_combinations_l1876_187611

theorem boat_license_combinations :
  let letters := ['A', 'M', 'S']
  let non_zero_digits := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let any_digit := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  3 * 9 * 10^4 = 270000 := 
by 
  sorry

end boat_license_combinations_l1876_187611


namespace det_A_eq_6_l1876_187610

open Matrix

variables {R : Type*} [Field R]

def A (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![a, 2], ![-3, d]]

def B (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  ![![2 * a, 1], ![-1, d]]

noncomputable def B_inv (a d : R) : Matrix (Fin 2) (Fin 2) R :=
  let detB := (2 * a * d + 1)
  ![![d / detB, -1 / detB], ![1 / detB, (2 * a) / detB]]

theorem det_A_eq_6 (a d : R) (hB_inv : (A a d) + (B_inv a d) = 0) : det (A a d) = 6 :=
  sorry

end det_A_eq_6_l1876_187610


namespace find_m_value_l1876_187617

def f (x : ℝ) : ℝ := |x + 1| - |x - 1|

noncomputable def find_m (m : ℝ) : Prop :=
  f (f m) = f 2002 - 7 / 2

theorem find_m_value : find_m (-3 / 8) :=
by
  unfold find_m
  sorry

end find_m_value_l1876_187617


namespace sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l1876_187696

-- Problem 1
theorem sqrt_18_mul_sqrt_6 : (Real.sqrt 18 * Real.sqrt 6 = 6 * Real.sqrt 3) :=
sorry

-- Problem 2
theorem sqrt_8_sub_sqrt_2_add_2_sqrt_half : (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1 / 2) = 3 * Real.sqrt 2) :=
sorry

-- Problem 3
theorem sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3 : (Real.sqrt 12 * (Real.sqrt 9 / 3) / (Real.sqrt 3 / 3) = 6) :=
sorry

-- Problem 4
theorem sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5 : ((Real.sqrt 7 + Real.sqrt 5) * (Real.sqrt 7 - Real.sqrt 5) = 2) :=
sorry

end sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l1876_187696


namespace coefficient_A_l1876_187665

-- Definitions from the conditions
variable (A c₀ d : ℝ)
variable (h₁ : c₀ = 47)
variable (h₂ : A * c₀ + (d - 12) ^ 2 = 235)

-- The theorem to prove
theorem coefficient_A (h₁ : c₀ = 47) (h₂ : A * c₀ + (d - 12) ^ 2 = 235) : A = 5 :=
by sorry

end coefficient_A_l1876_187665


namespace cos_A_eq_neg_quarter_l1876_187674

-- Definitions of angles and sides in the triangle
variables (A B C : ℝ)
variables (a b c : ℝ)

-- Conditions from the math problem
axiom sin_arithmetic_sequence : 2 * Real.sin B = Real.sin A + Real.sin C
axiom side_relation : a = 2 * c

-- Question to be proved as Lean 4 statement
theorem cos_A_eq_neg_quarter (h1 : ∀ {x y z : ℝ}, 2 * y = x + z) 
                              (h2 : ∀ {a b c : ℝ}, a = 2 * c) : 
                              Real.cos A = -1/4 := 
sorry

end cos_A_eq_neg_quarter_l1876_187674


namespace fox_initial_coins_l1876_187621

theorem fox_initial_coins :
  ∃ (x : ℕ), ∀ (c1 c2 c3 : ℕ),
    c1 = 3 * x - 50 ∧
    c2 = 3 * c1 - 50 ∧
    c3 = 3 * c2 - 50 ∧
    3 * c3 - 50 = 20 →
    x = 25 :=
by
  sorry

end fox_initial_coins_l1876_187621


namespace students_selected_from_grade_10_l1876_187626

theorem students_selected_from_grade_10 (students_grade10 students_grade11 students_grade12 total_selected : ℕ)
  (h_grade10 : students_grade10 = 1200)
  (h_grade11 : students_grade11 = 1000)
  (h_grade12 : students_grade12 = 800)
  (h_total_selected : total_selected = 100) :
  students_grade10 * total_selected = 40 * (students_grade10 + students_grade11 + students_grade12) :=
by
  sorry

end students_selected_from_grade_10_l1876_187626


namespace original_profit_margin_theorem_l1876_187693

noncomputable def original_profit_margin (a : ℝ) (x : ℝ) (h : a > 0) : Prop := 
  (a * (1 + x) - a * (1 - 0.064)) / (a * (1 - 0.064)) = x + 0.08

theorem original_profit_margin_theorem (a : ℝ) (x : ℝ) (h : a > 0) :
  original_profit_margin a x h → x = 0.17 :=
sorry

end original_profit_margin_theorem_l1876_187693


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l1876_187654

-- Definitions based on given conditions
variables (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ)

-- Conditions
axiom a_4 : a 4 = 6
axiom a_6 : a 6 = 10
axiom all_positive_b : ∀ n, 0 < b n
axiom b_3 : b 3 = a 3
axiom T_2 : T 2 = 3

-- Required to prove
theorem arithmetic_sequence_general_formula : ∀ n, a n = 2 * n - 2 :=
sorry

theorem geometric_sequence_sum_formula : ∀ n, T n = 2^n - 1 :=
sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_formula_l1876_187654


namespace election_votes_l1876_187660

theorem election_votes
  (V : ℕ)  -- total number of votes
  (candidate1_votes_percent : ℕ := 80)  -- first candidate percentage
  (second_candidate_votes : ℕ := 480)  -- votes for second candidate
  (second_candidate_percent : ℕ := 20)  -- second candidate percentage
  (h : second_candidate_votes = (second_candidate_percent * V) / 100) :
  V = 2400 :=
sorry

end election_votes_l1876_187660


namespace prob_two_segments_same_length_l1876_187672

namespace hexagon_prob

noncomputable def prob_same_length : ℚ :=
  let total_elements : ℕ := 15
  let sides : ℕ := 6
  let diagonals : ℕ := 9
  (sides / total_elements) * ((sides - 1) / (total_elements - 1)) + (diagonals / total_elements) * ((diagonals - 1) / (total_elements - 1))

theorem prob_two_segments_same_length : prob_same_length = 17 / 35 :=
by
  sorry

end hexagon_prob

end prob_two_segments_same_length_l1876_187672


namespace cost_price_l1876_187678

theorem cost_price (SP : ℝ) (profit_percentage : ℝ) : SP = 600 ∧ profit_percentage = 60 → ∃ CP : ℝ, CP = 375 :=
by
  intro h
  sorry

end cost_price_l1876_187678


namespace intersection_empty_l1876_187634

-- Define the set M
def M : Set ℝ := { x | ∃ y, y = Real.log (1 - x)}

-- Define the set N
def N : Set (ℝ × ℝ) := { p | ∃ x, ∃ y, (p = (x, y)) ∧ (y = Real.exp x) ∧ (x ∈ Set.univ)}

-- Prove that M ∩ N = ∅
theorem intersection_empty : M ∩ (Prod.fst '' N) = ∅ :=
by
  sorry

end intersection_empty_l1876_187634


namespace fraction_of_original_price_l1876_187653

theorem fraction_of_original_price
  (CP SP : ℝ)
  (h1 : SP = 1.275 * CP)
  (f: ℝ)
  (h2 : f * SP = 0.85 * CP)
  : f = 17 / 25 :=
by
  sorry

end fraction_of_original_price_l1876_187653


namespace sum_geometric_arithmetic_progression_l1876_187618

theorem sum_geometric_arithmetic_progression :
  ∃ (a b r d : ℝ), a = 1 * r ∧ b = 1 * r^2 ∧ b = a + d ∧ 16 = b + d ∧ (a + b = 12.64) :=
by
  sorry

end sum_geometric_arithmetic_progression_l1876_187618


namespace find_a_value_l1876_187605

theorem find_a_value
  (a : ℕ)
  (x y : ℝ)
  (h1 : a * x + y = -4)
  (h2 : 2 * x + y = -2)
  (hx_neg : x < 0)
  (hy_pos : y > 0) :
  a = 3 :=
by
  sorry

end find_a_value_l1876_187605


namespace product_of_integers_l1876_187659

theorem product_of_integers (x y : ℤ) (h1 : Int.gcd x y = 5) (h2 : Int.lcm x y = 60) : x * y = 300 :=
by
  sorry

end product_of_integers_l1876_187659


namespace find_avg_mpg_first_car_l1876_187692

def avg_mpg_first_car (x : ℝ) : Prop :=
  let miles_per_month := 450 / 3
  let gallons_first_car := miles_per_month / x
  let gallons_second_car := miles_per_month / 10
  let gallons_third_car := miles_per_month / 15
  let total_gallons := 56 / 2
  gallons_first_car + gallons_second_car + gallons_third_car = total_gallons

theorem find_avg_mpg_first_car : avg_mpg_first_car 50 :=
  sorry

end find_avg_mpg_first_car_l1876_187692


namespace inequality_holds_l1876_187642

theorem inequality_holds (x a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) :
  (1 / (x - a)) + (1 / (x - b)) > 1 / (x - c) := 
by sorry

end inequality_holds_l1876_187642


namespace find_largest_divisor_l1876_187681

def f (n : ℕ) : ℕ := (2 * n + 7) * 3 ^ n + 9

theorem find_largest_divisor :
  ∃ m : ℕ, (∀ n : ℕ, f n % m = 0) ∧ m = 36 :=
sorry

end find_largest_divisor_l1876_187681


namespace min_shirts_to_save_money_l1876_187627

theorem min_shirts_to_save_money :
  ∃ (x : ℕ), 75 + 8 * x < 12 * x ∧ x = 19 :=
sorry

end min_shirts_to_save_money_l1876_187627


namespace time_for_B_alone_l1876_187689

theorem time_for_B_alone (h1 : 4 * (1/15 + 1/x) = 7/15) : x = 20 :=
sorry

end time_for_B_alone_l1876_187689


namespace condo_cats_l1876_187673

theorem condo_cats (x y : ℕ) (h1 : 2 * x + y = 29) : 6 * x + 3 * y = 87 := by
  sorry

end condo_cats_l1876_187673


namespace find_other_number_l1876_187635

theorem find_other_number (HCF LCM one_number other_number : ℤ)
  (hHCF : HCF = 12)
  (hLCM : LCM = 396)
  (hone_number : one_number = 48)
  (hrelation : HCF * LCM = one_number * other_number) :
  other_number = 99 :=
by
  sorry

end find_other_number_l1876_187635


namespace problem_solution_l1876_187649

-- Declare the proof problem in Lean 4

theorem problem_solution (x y : ℝ) 
  (h1 : (y + 1) ^ 2 + (x - 2) ^ (1/2) = 0) : 
  y ^ x = 1 :=
sorry

end problem_solution_l1876_187649


namespace sqrt_20_minus_1_range_l1876_187680

theorem sqrt_20_minus_1_range : 
  16 < 20 ∧ 20 < 25 ∧ Real.sqrt 16 = 4 ∧ Real.sqrt 25 = 5 → (3 < Real.sqrt 20 - 1 ∧ Real.sqrt 20 - 1 < 4) :=
by
  intro h
  sorry

end sqrt_20_minus_1_range_l1876_187680


namespace knights_count_l1876_187666

theorem knights_count (T F : ℕ) (h1 : T + F = 65) (h2 : ∀ n < 21, ¬(T = F - 20)) 
  (h3 : ∀ n ≥ 21, if n % 2 = 1 then T = (n - 1) / 2 + 1 else T = (n - 1) / 2):
  T = 23 :=
by
      -- Here the specific steps of the proof will go
      sorry

end knights_count_l1876_187666


namespace probability_no_adjacent_standing_l1876_187663

-- Define the problem conditions in Lean 4.
def total_outcomes := 2^10
def favorable_outcomes := 123

-- The probability is given by favorable outcomes over total outcomes.
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now state the theorem regarding the probability.
theorem probability_no_adjacent_standing : 
  probability = 123 / 1024 :=
by {
  sorry
}

end probability_no_adjacent_standing_l1876_187663


namespace simplify_expression_l1876_187686

theorem simplify_expression : 
  (1 / ((1 / (1 / 3)^1) + (1 / (1 / 3)^2) + (1 / (1 / 3)^3))) = 1 / 39 :=
by
  sorry

end simplify_expression_l1876_187686


namespace charlie_collected_15_seashells_l1876_187622

variables (c e : ℝ)

-- Charlie collected 10 more seashells than Emily
def charlie_more_seashells := c = e + 10

-- Emily collected one-third the number of seashells Charlie collected
def emily_seashells := e = c / 3

theorem charlie_collected_15_seashells (hc: charlie_more_seashells c e) (he: emily_seashells c e) : c = 15 := 
by sorry

end charlie_collected_15_seashells_l1876_187622


namespace calculate_weight_of_6_moles_HClO2_l1876_187694

noncomputable def weight_of_6_moles_HClO2 := 
  let molar_mass_H := 1.01
  let molar_mass_Cl := 35.45
  let molar_mass_O := 16.00
  let molar_mass_HClO2 := molar_mass_H + molar_mass_Cl + 2 * molar_mass_O
  let moles_HClO2 := 6
  moles_HClO2 * molar_mass_HClO2

theorem calculate_weight_of_6_moles_HClO2 : weight_of_6_moles_HClO2 = 410.76 :=
by
  sorry

end calculate_weight_of_6_moles_HClO2_l1876_187694


namespace sin_cos_difference_l1876_187600

theorem sin_cos_difference
  (θ : ℝ)
  (h1 : θ ∈ Set.Ioo 0 Real.pi)
  (h2 : Real.sin θ + Real.cos θ = 1 / 5) :
  Real.sin θ - Real.cos θ = 7 / 5 :=
sorry

end sin_cos_difference_l1876_187600


namespace students_exceed_goldfish_l1876_187657

theorem students_exceed_goldfish 
    (num_classrooms : ℕ) 
    (students_per_classroom : ℕ) 
    (goldfish_per_classroom : ℕ) 
    (h1 : num_classrooms = 5) 
    (h2 : students_per_classroom = 20) 
    (h3 : goldfish_per_classroom = 3) 
    : (students_per_classroom * num_classrooms) - (goldfish_per_classroom * num_classrooms) = 85 := by
  sorry

end students_exceed_goldfish_l1876_187657


namespace train_cross_time_approx_l1876_187633
noncomputable def time_to_cross_bridge
  (train_length : ℝ) (bridge_length : ℝ) (speed_kmh : ℝ) : ℝ :=
  ((train_length + bridge_length) / (speed_kmh * 1000 / 3600))

theorem train_cross_time_approx (train_length bridge_length speed_kmh : ℝ)
  (h_train_length : train_length = 250)
  (h_bridge_length : bridge_length = 300)
  (h_speed_kmh : speed_kmh = 44) :
  abs (time_to_cross_bridge train_length bridge_length speed_kmh - 45) < 1 :=
by
  sorry

end train_cross_time_approx_l1876_187633


namespace basketball_free_throws_l1876_187699

theorem basketball_free_throws (a b x : ℕ) 
  (h1 : 3 * b = 4 * a) 
  (h2 : x = 2 * a) 
  (h3 : 2 * a + 3 * b + x = 72) : 
  x = 18 := 
sorry

end basketball_free_throws_l1876_187699


namespace original_number_solution_l1876_187688

theorem original_number_solution (x : ℝ) (h : x^2 + 45 = 100) : x = Real.sqrt 55 ∨ x = -Real.sqrt 55 :=
by
  sorry

end original_number_solution_l1876_187688


namespace necessary_but_not_sufficient_l1876_187609

theorem necessary_but_not_sufficient :
    (∀ (x y : ℝ), x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
    ¬(∀ (x y : ℝ), x + y > 5 ∧ x * y > 6 → x > 2 ∧ y > 3) := by
  sorry

end necessary_but_not_sufficient_l1876_187609


namespace system1_solution_system2_solution_l1876_187623

theorem system1_solution (p q : ℝ) 
  (h1 : p + q = 4)
  (h2 : 2 * p - q = 5) : 
  p = 3 ∧ q = 1 := 
sorry

theorem system2_solution (v t : ℝ)
  (h3 : 2 * v + t = 3)
  (h4 : 3 * v - 2 * t = 3) :
  v = 9 / 7 ∧ t = 3 / 7 :=
sorry

end system1_solution_system2_solution_l1876_187623


namespace max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l1876_187620

-- Problem (1)
theorem max_xy (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x*y ≤ 4 :=
sorry

-- Additional statement to show when the maximum is achieved
theorem max_xy_is_4 (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y + x*y = 12) : x = 4 ∧ y = 1 ↔ x*y = 4 :=
sorry

-- Problem (2)
theorem min_x_plus_y (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x + y ≥ 9 :=
sorry

-- Additional statement to show when the minimum is achieved
theorem min_x_plus_y_is_9 (x y : ℝ) (h_pos_x : 4 < x) (h_pos_y : 0 < y) (h_constraint : x + 4*y = x*y) : x = 6 ∧ y = 3 ↔ x + y = 9 :=
sorry

end max_xy_max_xy_is_4_min_x_plus_y_min_x_plus_y_is_9_l1876_187620


namespace freezer_temp_calculation_l1876_187647

def refrigerator_temp : ℝ := 4
def freezer_temp (rt : ℝ) (d : ℝ) : ℝ := rt - d

theorem freezer_temp_calculation :
  (freezer_temp refrigerator_temp 22) = -18 :=
by
  sorry

end freezer_temp_calculation_l1876_187647


namespace highest_score_l1876_187669

-- Definitions based on conditions
variable (H L : ℕ)

-- Condition (1): H - L = 150
def condition1 : Prop := H - L = 150

-- Condition (2): H + L = 208
def condition2 : Prop := H + L = 208

-- Condition (3): Total runs in 46 innings at an average of 60, excluding two innings averages to 58
def total_runs := 60 * 46
def excluded_runs := total_runs - 2552

theorem highest_score
  (cond1 : condition1 H L)
  (cond2 : condition2 H L)
  : H = 179 :=
by sorry

end highest_score_l1876_187669


namespace limit_proof_l1876_187644

open Real

-- Define the conditions
axiom sin_6x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |sin (6 * x) / (6 * x) - 1| < ε
axiom arctg_2x_approx (x : ℝ) : ∀ ε > 0, x ≠ 0 → |arctan (2 * x) / (2 * x) - 1| < ε

-- State the limit proof problem
theorem limit_proof :
  (∃ ε > 0, ∀ x : ℝ, |x| < ε → x ≠ 0 →
  |(x * sin (6 * x)) / (arctan (2 * x)) ^ 2 - (3 / 2)| < ε) :=
sorry

end limit_proof_l1876_187644


namespace composite_quotient_l1876_187656

def first_eight_composites := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites := [16, 18, 20, 21, 22, 24, 25, 26]

def product (l : List ℕ) := l.foldl (· * ·) 1

theorem composite_quotient :
  let numerator := product first_eight_composites
  let denominator := product next_eight_composites
  numerator / denominator = (1 : ℚ)/(1430 : ℚ) :=
by
  sorry

end composite_quotient_l1876_187656


namespace y_minus_x_value_l1876_187613

theorem y_minus_x_value (x y : ℝ) (h1 : x + y = 500) (h2 : x / y = 0.8) : y - x = 55.56 :=
sorry

end y_minus_x_value_l1876_187613


namespace positive_value_of_n_l1876_187612

theorem positive_value_of_n (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 25 = 0 ∧ ∃! x : ℝ, 4 * x^2 + n * x + 25 = 0) →
  n = 20 :=
by
  sorry

end positive_value_of_n_l1876_187612


namespace percentage_increase_l1876_187671

theorem percentage_increase (lowest_price highest_price : ℝ) (h_low : lowest_price = 15) (h_high : highest_price = 25) :
  ((highest_price - lowest_price) / lowest_price) * 100 = 66.67 :=
by
  sorry

end percentage_increase_l1876_187671


namespace total_apples_picked_l1876_187641

def apples_picked : ℕ :=
  let mike := 7
  let nancy := 3
  let keith := 6
  let olivia := 12
  let thomas := 8
  mike + nancy + keith + olivia + thomas

theorem total_apples_picked :
  apples_picked = 36 :=
by
  -- Proof would go here; 'sorry' is used to skip the proof.
  sorry

end total_apples_picked_l1876_187641


namespace g_f2_minus_f_g2_eq_zero_l1876_187601

def f (x : ℝ) : ℝ := x^2 + 3 * x + 1

def g (x : ℝ) : ℝ := x^2 + 2 * x + 3

theorem g_f2_minus_f_g2_eq_zero : g (f 2) - f (g 2) = 0 := by
  sorry

end g_f2_minus_f_g2_eq_zero_l1876_187601


namespace subset_A_if_inter_eq_l1876_187650

variable {B : Set ℝ}

def A : Set ℝ := { x | 0 < x ∧ x < 2 }

theorem subset_A_if_inter_eq:
  A ∩ B = B ↔ B = ∅ ∨ B = {1} ∨ B = { x | 0 < x ∧ x < 2 } :=
by
  sorry

end subset_A_if_inter_eq_l1876_187650


namespace students_per_table_correct_l1876_187639

-- Define the number of tables and students
def num_tables := 34
def num_students := 204

-- Define x as the number of students per table
def students_per_table := 6

-- State the theorem
theorem students_per_table_correct : num_students / num_tables = students_per_table :=
by
  sorry

end students_per_table_correct_l1876_187639


namespace sum_first_20_integers_l1876_187682

def sum_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_first_20_integers : sum_first_n_integers 20 = 210 :=
by
  -- Provided proof omitted
  sorry

end sum_first_20_integers_l1876_187682


namespace number_of_parallel_lines_l1876_187691

/-- 
Given 10 parallel lines in the first set and the fact that the intersection 
of two sets of parallel lines forms 1260 parallelograms, 
prove that the second set contains 141 parallel lines.
-/
theorem number_of_parallel_lines (n : ℕ) (h₁ : 10 - 1 = 9) (h₂ : 9 * (n - 1) = 1260) : n = 141 :=
sorry

end number_of_parallel_lines_l1876_187691


namespace variance_le_second_moment_l1876_187616

noncomputable def variance (X : ℝ → ℝ) (MX : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - MX)^2]

noncomputable def second_moment (X : ℝ → ℝ) (C : ℝ) : ℝ :=
  sorry -- Assume defined as M[(X - C)^2]

theorem variance_le_second_moment (X : ℝ → ℝ) :
  ∀ C : ℝ, C ≠ MX → variance X MX ≤ second_moment X C := 
by
  sorry

end variance_le_second_moment_l1876_187616


namespace tomatoes_for_5_liters_l1876_187608

theorem tomatoes_for_5_liters (kg_per_3_liters : ℝ) (liters_needed : ℝ) :
  (kg_per_3_liters = 69 / 3) → (liters_needed = 5) → (kg_per_3_liters * liters_needed = 115) := 
by
  intros h1 h2
  sorry

end tomatoes_for_5_liters_l1876_187608


namespace profit_percent_is_20_l1876_187637

variable (C S : ℝ)

-- Definition from condition: The cost price of 60 articles is equal to the selling price of 50 articles
def condition : Prop := 60 * C = 50 * S

-- Definition of profit percent to be proven as 20%
def profit_percent_correct : Prop := ((S - C) / C) * 100 = 20

theorem profit_percent_is_20 (h : condition C S) : profit_percent_correct C S :=
sorry

end profit_percent_is_20_l1876_187637


namespace backpack_original_price_l1876_187648

-- Define original price of a ring-binder
def original_ring_binder_price : ℕ := 20

-- Define the number of ring-binders bought
def number_of_ring_binders : ℕ := 3

-- Define the new price increase for the backpack
def backpack_price_increase : ℕ := 5

-- Define the new price decrease for the ring-binder
def ring_binder_price_decrease : ℕ := 2

-- Define the total amount spent
def total_amount_spent : ℕ := 109

-- Define the original price of the backpack variable
variable (B : ℕ)

-- Theorem statement: under these conditions, the original price of the backpack must be 50
theorem backpack_original_price :
  (B + backpack_price_increase) + ((original_ring_binder_price - ring_binder_price_decrease) * number_of_ring_binders) = total_amount_spent ↔ B = 50 :=
by 
  sorry

end backpack_original_price_l1876_187648


namespace fermat_1000_units_digit_l1876_187687

-- Define Fermat numbers
def FermatNumber (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 1

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := n % 10

-- The theorem to be proven
theorem fermat_1000_units_digit : units_digit (FermatNumber 1000) = 7 := 
by sorry

end fermat_1000_units_digit_l1876_187687


namespace hash_7_2_eq_24_l1876_187606

def hash_op (a b : ℕ) : ℕ := 4 * a - 2 * b

theorem hash_7_2_eq_24 : hash_op 7 2 = 24 := by
  sorry

end hash_7_2_eq_24_l1876_187606


namespace points_on_hyperbola_order_l1876_187670

theorem points_on_hyperbola_order (k a b c : ℝ) (hk : k > 0)
  (h₁ : a = k / -2)
  (h₂ : b = k / 2)
  (h₃ : c = k / 3) :
  a < c ∧ c < b := 
sorry

end points_on_hyperbola_order_l1876_187670


namespace students_at_start_of_year_l1876_187646

variable (S : ℕ)

def initial_students := S
def students_left := 6
def students_new := 42
def end_year_students := 47

theorem students_at_start_of_year :
  initial_students + (students_new - students_left) = end_year_students → initial_students = 11 :=
by
  sorry

end students_at_start_of_year_l1876_187646


namespace arrange_numbers_l1876_187619

theorem arrange_numbers (x y z : ℝ) (h1 : x = 20.8) (h2 : y = 0.82) (h3 : z = Real.log 20.8) : z < y ∧ y < x :=
by
  sorry

end arrange_numbers_l1876_187619


namespace boards_nailing_l1876_187632

variables {x y a b : ℕ} 

theorem boards_nailing :
  (2 * x + 3 * y = 87) ∧
  (3 * a + 5 * b = 94) →
  (x + y = 30) ∧ (a + b = 30) :=
by
  sorry

end boards_nailing_l1876_187632


namespace determine_a_l1876_187651

open Complex

noncomputable def complex_eq_real_im_part (a : ℝ) : Prop :=
  let z := (a - I) * (1 + I) / I
  (z.re, z.im) = ((a - 1 : ℝ), -(a + 1 : ℝ))

theorem determine_a (a : ℝ) (h : complex_eq_real_im_part a) : a = -1 :=
sorry

end determine_a_l1876_187651


namespace boat_equation_l1876_187667

-- Define the conditions given in the problem
def total_boats : ℕ := 8
def large_boat_capacity : ℕ := 6
def small_boat_capacity : ℕ := 4
def total_students : ℕ := 38

-- Define the theorem to be proven
theorem boat_equation (x : ℕ) (h0 : x ≤ total_boats) : 
  large_boat_capacity * (total_boats - x) + small_boat_capacity * x = total_students := by
  sorry

end boat_equation_l1876_187667


namespace sum_of_cubes_is_24680_l1876_187661

noncomputable def jake_age := 10
noncomputable def amy_age := 12
noncomputable def ryan_age := 28

theorem sum_of_cubes_is_24680 (j a r : ℕ) (h1 : 2 * j + 3 * a = 4 * r)
  (h2 : j^3 + a^3 = 1 / 2 * r^3) (h3 : j + a + r = 50) : j^3 + a^3 + r^3 = 24680 :=
by
  sorry

end sum_of_cubes_is_24680_l1876_187661
