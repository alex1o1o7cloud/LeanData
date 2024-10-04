import Mathlib

namespace integer_solution_exists_l244_244451

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (a % 7 = 1 ∨ a % 7 = 6) :=
by sorry

end integer_solution_exists_l244_244451


namespace jason_pears_count_l244_244591

theorem jason_pears_count 
  (initial_pears : ℕ)
  (given_to_keith : ℕ)
  (received_from_mike : ℕ)
  (final_pears : ℕ)
  (h_initial : initial_pears = 46)
  (h_given : given_to_keith = 47)
  (h_received : received_from_mike = 12)
  (h_final : final_pears = 12) :
  initial_pears - given_to_keith + received_from_mike = final_pears :=
sorry

end jason_pears_count_l244_244591


namespace polyhedron_has_triangular_face_l244_244964

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end polyhedron_has_triangular_face_l244_244964


namespace flour_for_recipe_l244_244598

theorem flour_for_recipe (flour_needed shortening_have : ℚ)
  (flour_ratio shortening_ratio : ℚ) 
  (ratio : flour_ratio / shortening_ratio = 5)
  (shortening_used : shortening_ratio = 2 / 3) :
  flour_needed = 10 / 3 := 
by 
  sorry

end flour_for_recipe_l244_244598


namespace find_u_function_l244_244870

theorem find_u_function (u f : ℝ → ℝ) (h : ∃ f : ℝ → ℝ, (strictly_monotonic f) ∧ (∀ x y : ℝ, f(x + y) = f(x) * u(y) + f(y))) : 
  ∃ a : ℝ, ∀ x : ℝ, u(x) = Real.exp (a * x) :=
sorry

end find_u_function_l244_244870


namespace solve_quadratic1_solve_quadratic2_solve_quadratic3_solve_quadratic4_l244_244286

-- Problem 1
theorem solve_quadratic1 : ∃ (x1 x2 : ℂ), (x1 = 2 + complex.sqrt(6) ∧ x2 = 2 - complex.sqrt(6)) ∧ (∀ x, x^2 - 4 * x - 2 = 0 ↔ x = x1 ∨ x = x2) :=
sorry

-- Problem 2
theorem solve_quadratic2 : ∃ (y1 y2 : ℂ), (y1 = (3 + complex.sqrt(17)) / 4 ∧ y2 = (3 - complex.sqrt(17)) / 4) ∧ (∀ y, 2 * y^2 - 3 * y - 1 = 0 ↔ y = y1 ∨ y = y2) :=
sorry

-- Problem 3
theorem solve_quadratic3 : ∃ (x1 x2 : ℂ), (x1 = 1 ∧ x2 = -2/3) ∧ (∀ x, 3 * x * (x - 1) = 2 - 2 * x ↔ x = x1 ∨ x = x2) :=
sorry

-- Problem 4
theorem solve_quadratic4 : ∃ (x1 x2 : ℂ), (x1 = 1 ∧ x2 = -1/2) ∧ (∀ x, 2 * x^2 - x - 1 = 0 ↔ x = x1 ∨ x = x2) :=
sorry

end solve_quadratic1_solve_quadratic2_solve_quadratic3_solve_quadratic4_l244_244286


namespace permutation_two_books_from_five_l244_244720

def num_books : ℕ := 5
def num_books_to_select : ℕ := 2

theorem permutation_two_books_from_five :
  (∑  S in Finset.perm {1, 2, 3, 4, 5}, S.card = 2 ∧ S ≠ ∅) = 20 := 
sorry

end permutation_two_books_from_five_l244_244720


namespace nth_element_of_sequence_is_100_over_201_l244_244646
open Nat

-- Define the sequence according to the given conditions
def sequence (n : Nat) : ℚ :=
  (n : ℚ) / (1 + 2 * n : ℚ)

-- Define the statement to prove
theorem nth_element_of_sequence_is_100_over_201 :
  sequence 100 = 100 / 201 :=
by
  sorry

end nth_element_of_sequence_is_100_over_201_l244_244646


namespace surface_area_inequality_l244_244013

theorem surface_area_inequality
  (a b c d e f S : ℝ) :
  S ≤ (Real.sqrt 3 / 6) * (a^2 + b^2 + c^2 + d^2 + e^2 + f^2) :=
sorry

end surface_area_inequality_l244_244013


namespace part1_part2_l244_244615

variables (a b c : ℝ)

theorem part1 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  ab + bc + ac ≤ 1 / 3 := sorry

theorem part2 (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 1) :
  1 / a + 1 / b + 1 / c ≥ 9 := sorry

end part1_part2_l244_244615


namespace circles_are_separate_l244_244942

def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 1

def circle2_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 9

def center1 : (ℝ × ℝ) := (0, 0)
def center2 : (ℝ × ℝ) := (3, -4)

def radius1 : ℝ := 1
def radius2 : ℝ := 3

def dist_centers : ℝ := real.sqrt ((3 - 0)^2 + (-4 - 0)^2) -- distance between (0, 0) and (3, -4)

theorem circles_are_separate : dist_centers > radius1 + radius2 :=
by
  sorry

end circles_are_separate_l244_244942


namespace max_remaining_cookies_l244_244784

theorem max_remaining_cookies (total_cookies children : ℕ) 
  (h_cookies : total_cookies = 28) (h_children : children = 6) :
  let cookies_per_child := total_cookies / children,
      cookies_distributed := cookies_per_child * children,
      remaining_cookies := total_cookies - cookies_distributed
  in remaining_cookies = 4 :=
by
  sorry

end max_remaining_cookies_l244_244784


namespace hawaii_normal_avg_rain_l244_244977

variable x : ℝ -- normal average of rain per day
variable already_received : ℝ := 430 -- inches of rain already received
variable remaining_days : ℝ := 100 -- number of days left in the year
variable additional_average : ℝ := 3 -- average inches of rain required for the remaining days
variable total_days : ℝ := 365 -- total number of days in a year

theorem hawaii_normal_avg_rain : x = 2 :=
by
  let additional_rain := additional_average * remaining_days
  let total_rain := already_received + additional_rain
  let normal_average := total_rain / total_days
  have h : normal_average = x := by sorry
  -- need to show that x is indeed 2 using the given conditions
  sorry

end hawaii_normal_avg_rain_l244_244977


namespace triangle_bc_range_l244_244927

open Real

theorem triangle_bc_range (AB : ℝ) (C : ℝ) (BC : ℝ) (A : ℝ) 
  (h1 : AB = sqrt 3) 
  (h2 : C = π / 3)
  (h3 : A ∈ (π / 3, 2 * π / 3)) : 
  sqrt 3 < BC ∧ BC < 2 :=
sorry

end triangle_bc_range_l244_244927


namespace calculate_sum_of_triangles_l244_244670

def operation_triangle (a b c : Int) : Int :=
  a * b - c 

theorem calculate_sum_of_triangles :
  operation_triangle 3 4 5 + operation_triangle 1 2 4 + operation_triangle 2 5 6 = 9 :=
by 
  sorry

end calculate_sum_of_triangles_l244_244670


namespace range_of_x_l244_244033

def f (x : ℝ) : ℝ := 3^x - 3^(-x) - 2 * x

theorem range_of_x (x : ℝ) : 
  (2 < x ∧ x < real.sqrt 3) ↔ (x - 2) * f(real.log (x) / real.log (1 / 2)) < 0 :=
sorry

end range_of_x_l244_244033


namespace geom_seq_b_arith_seq_max_n_geom_seq_log_sum_investment_plan_years_l244_244364

-- Problem 1
theorem geom_seq_b (a b c : ℝ) (q : ℝ) (h1 : q ≠ 0) (h2 : 1 ≠ 0) (h_geom : [1, a, b, c, 4] = [1, 1 * q, 1 * q^2, 1 * q^3, 1 * q^4]) :
  b = 2 :=
by
  sorry

-- Problem 2
theorem arith_seq_max_n (a₁ d : ℝ) (h_d : d < 0) (S : ℕ → ℝ)
  (h_S16 : S 16 > 0) (h_S17 : S 17 < 0) (h_arith : ∀ n, S n = n * (a₁ + (n - 1) * d / 2)) :
  ∃ n, n = 8 :=
by
  sorry

-- Problem 3
theorem geom_seq_log_sum (a₁ a₂ a₃ a₄ : ℝ) (h_pos : ∀ n, a₁ > 0) (h_geom : a₁ * a₂ = a₂ * a₃) (h_a2a3 : a₂ * a₃ = 16) :
  ∑ i in [a₁, a₂, a₃, a₄], log 2 (i) = 8 :=
by
  sorry

-- Problem 4
theorem investment_plan_years :
  ∃ n : ℕ, (5 * n + 10 * (n * (n - 1) / 2)) ≥ 50 :=
by
  sorry

end geom_seq_b_arith_seq_max_n_geom_seq_log_sum_investment_plan_years_l244_244364


namespace triangle_inequality_property_l244_244763

noncomputable def perimeter (a b c : ℝ) : ℝ := a + b + c

noncomputable def circumradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  (a * b * c) / (4 * Real.sqrt (A * B * C))

noncomputable def inradius (a b c : ℝ) (A B C: ℝ) : ℝ := 
  Real.sqrt (A * B * C) * perimeter a b c

theorem triangle_inequality_property (a b c A B C : ℝ)
  (h₁ : ∀ {x}, x > 0)
  (h₂ : A ≠ B)
  (h₃ : B ≠ C)
  (h₄ : C ≠ A) :
  ¬ (perimeter a b c ≤ circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c > circumradius a b c A B C + inradius a b c A B C) ∧
  ¬ (perimeter a b c / 6 < circumradius a b c A B C + inradius a b c A B C ∨ 
  circumradius a b c A B C + inradius a b c A B C < 6 * perimeter a b c) :=
sorry

end triangle_inequality_property_l244_244763


namespace sum_of_squares_inequality_l244_244018

theorem sum_of_squares_inequality (a b c : ℝ) (h : a + 2 * b + 3 * c = 4) : a^2 + b^2 + c^2 ≥ 8 / 7 := by
  sorry

end sum_of_squares_inequality_l244_244018


namespace find_natural_numbers_l244_244061

theorem find_natural_numbers (a : ℕ) :
  (10 ≤ a + 50 ∧ a + 50 ≤ 99) ∧
  (10 ≤ a - 32 ∧ a - 32 ≤ 99) ∧
  (∀ m : ℕ, 10 ≤ m ∧ m ≤ 99 → gcd a m = 1) →
  a ∈ {42, 43, 44, 45, 46, 47, 48, 49} :=
  by 
    sorry

end find_natural_numbers_l244_244061


namespace volume_ratio_l244_244595

noncomputable def ratio_of_volumes (d1 h1 d2 h2 : ℝ) : ℝ :=
  let r1 := d1 / 2
  let V1 := π * r1^2 * h1
  let r2 := d2 / 2
  let V2 := π * r2^2 * h2
  V1 / V2

theorem volume_ratio (d1 d2 h1 h2 : ℝ) (h_d1 : d1 = 10) (h_h1 : h1 = 15)
                     (h_d2 : d2 = 15) (h_h2 : h2 = 10) :
  ratio_of_volumes d1 h1 d2 h2 = 2 / 3 :=
by
  simp [ratio_of_volumes, h_d1, h_h1, h_d2, h_h2]
  sorry

end volume_ratio_l244_244595


namespace total_stuffed_animals_l244_244637

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l244_244637


namespace devin_teaching_years_l244_244859

theorem devin_teaching_years :
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  calculus_years + algebra_years + statistics_years = 52 :=
by
  let calculus_years := 4
  let algebra_years := 2 * calculus_years
  let statistics_years := 5 * algebra_years
  show calculus_years + algebra_years + statistics_years = 52
  sorry

end devin_teaching_years_l244_244859


namespace find_angle_A_find_sum_b_c_l244_244975

variables (a b c : ℝ) (A B C : ℝ)

/-- Given conditions -/
def given1 := 2 * b * (2 * b - c) * Real.cos A = a^2 + b^2 - c^2
def given2 := S = (25 * Real.sqrt 3) / 4
def given3 := a = 5

/- Part (I): Prove the measure of angle A -/
theorem find_angle_A (h : given1) : A = Real.pi / 3 :=
sorry

/- Part (II): Prove b + c given the area -/
theorem find_sum_b_c (h1 : given2) (h2 : given3) (h3 : given1) : b + c = 10 :=
sorry

end find_angle_A_find_sum_b_c_l244_244975


namespace shortest_distance_reflection_l244_244897

-- Definitions from the conditions of the problem
def point_P : ℝ × ℝ := (1, 0)
def point_Q : ℝ × ℝ := (2, 1)
def line : ℝ × ℝ → Prop := λ P, P.1 - P.2 + 1 = 0

-- Statement that we need to prove
theorem shortest_distance_reflection :
  ∃ (B : ℝ × ℝ), (B.1 + 1) * (B.1 - 1) + (B.2 - 2) * (B.2 + 2) = 0 ∧
  dist B point_Q = sqrt 10 :=
sorry

end shortest_distance_reflection_l244_244897


namespace quadratic_has_two_real_roots_l244_244082

theorem quadratic_has_two_real_roots (k : ℝ) (h1 : k ≠ 0) (h2 : 4 - 12 * k ≥ 0) : 0 < k ∧ k ≤ 1 / 3 :=
sorry

end quadratic_has_two_real_roots_l244_244082


namespace ratio_of_AB_to_AC_l244_244406

-- Definitions

variables {ABC : Type} [triangle : Triangle ABC]
variables {A B C D : Point} (h_bisects : AngleBisector AD A B C)
variables (h_ratios : (BD / DC) = 3)

-- Theorem to prove
theorem ratio_of_AB_to_AC : 
  ∀ {A B C D : Point}, 
  Triangle ABC → 
  AngleBisector AD A B C → 
  (BD / DC = 3) → 
  (AB / AC = 3) :=
by sorry

end ratio_of_AB_to_AC_l244_244406


namespace circle_polar_eq_l244_244776

theorem circle_polar_eq (ρ θ : ℝ) :
  (exists radius center, radius = 1 ∧ center = (1, 0) ∧ ρ = sqrt ((1 - cos θ)^2 + (sin θ)^2)) → ρ = 2 * cos θ := 
by
  sorry

end circle_polar_eq_l244_244776


namespace find_ratio_l244_244586

variables (A B C M E F G : Type)
variables (P : Triangle A B C)
variables (hM : midpoint M B C)
variables (hAB : dist A B = 15)
variables (hAC : dist A C = 18)
variables (hE_on_AC : collinear A C E)
variables (hF_on_AB : collinear A B F)
variables (G_intersect : intersection G (line_through E F) (line_through A M))
variables (hAE_3AF : dist A E = 3 * dist A F)

theorem find_ratio (h_midpoint : midpoint M B C) :
  ratio_segment (segment_intersection G E F) E F = 5 / 6 :=
sorry

end find_ratio_l244_244586


namespace triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l244_244557

-- Given the conditions: two sides of one triangle are equal to two sides of another triangle.
-- And an angle opposite to one of these sides is equal to the angle opposite to the corresponding side.
variables {A B C D E F : Type}
variables {AB DE BC EF : ℝ} (h_AB_DE : AB = DE) (h_BC_EF : BC = EF)
variables {angle_A angle_D : ℝ} (h_angle_A_D : angle_A = angle_D)

-- Prove that the triangles may or may not be congruent
theorem triangles_may_or_may_not_be_congruent :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_be_congruent_or_not : Prop) :=
sorry

-- Prove that the triangles may have equal areas
theorem triangles_may_have_equal_areas :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_have_equal_areas : Prop) :=
sorry

end triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l244_244557


namespace inequalities_not_hold_range_a_l244_244921

theorem inequalities_not_hold_range_a (a : ℝ) :
  (¬ ∀ x : ℝ, x^2 - a * x + 1 ≤ 0) ∧ (¬ ∀ x : ℝ, a * x^2 + x - 1 > 0) ↔ (-2 < a ∧ a ≤ -1 / 4) :=
by
  sorry

end inequalities_not_hold_range_a_l244_244921


namespace merchant_profit_after_discount_l244_244806

/-- A merchant marks his goods up by 40% and then offers a discount of 20% 
on the marked price. Prove that the merchant makes a profit of 12%. -/
theorem merchant_profit_after_discount :
  ∀ (CP MP SP : ℝ),
    CP > 0 →
    MP = CP * 1.4 →
    SP = MP * 0.8 →
    ((SP - CP) / CP) * 100 = 12 :=
by
  intros CP MP SP hCP hMP hSP
  sorry

end merchant_profit_after_discount_l244_244806


namespace slope_of_tangent_at_A_l244_244714

theorem slope_of_tangent_at_A (f : ℝ → ℝ) (x : ℝ) (y : ℝ)
    (h1 : ∀ x, f x = x^2 + 3*x)
    (h2 : x = 2)
    (h3 : y = 10)
    (hA : f x = y) : (deriv f x) = 7 :=
by
  have h4 : f = fun x => x^2 + 3*x := funext h1
  rw [h4] at hA
  simp [h2] at hA
  exact deriv_const _ (deriv_x_pow 2 2).symm


end slope_of_tangent_at_A_l244_244714


namespace division_value_l244_244740

theorem division_value (n x : ℝ) (h₀ : n = 4.5) (h₁ : (n / x) * 12 = 9) : x = 6 :=
by
  sorry

end division_value_l244_244740


namespace reflect_across_x_axis_l244_244579

theorem reflect_across_x_axis (A : ℝ × ℝ) (hA : A = (1, -2)) : A.reflect_across_x_axis = (1, 2) :=
by
  sorry

end reflect_across_x_axis_l244_244579


namespace parabola_focus_line_l244_244163

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244163


namespace parabola_properties_l244_244190

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244190


namespace distinct_arrangements_appear_l244_244055

-- Given definitions
def word : list char := ['a', 'p', 'p', 'e', 'a', 'r']
def occurrences (l : list char) (c : char) : ℕ := list.count c l

-- The distinct arrangements problem statement
theorem distinct_arrangements_appear : 
  let n := word.length in
  let k1 := occurrences word 'p' in
  n = 6 ∧ k1 = 2 →
  (n.factorial / k1.factorial) = 360 :=
by
  intro h
  cases h with h1 h2
  rw [h1, h2]
  norm_num
  sorry

end distinct_arrangements_appear_l244_244055


namespace find_x_l244_244578

-- Let a be the list of scores from the 9 judges including an unknown score x
variables (a : list ℝ) 
-- Let x be the unclear score in the stem-and-leaf plot
variable (x : ℝ)
-- Given conditions from the problem, we assume there are 9 scores
-- and the total sum without the highest and lowest scores is 637 (average of the 7 middle scores is 91)
hypothesis h_length : a.length = 9
hypothesis h_average : (a.sum - a.max - a.min) / 7 = 91
hypothesis h_sum : (a.sum - a.max - a.min) = 637
-- We need to find the value of x that satisfies these conditions
definition find_score (a : list ℝ) (x : ℝ) : Prop :=
  x ∈ a ∧ a.length = 9 ∧ ((a.sum - a.max - a.min) / 7 = 91) ∧ ((a.sum - a.max - a.min) = 637) ∧ x = 1

-- We state the theorem to find x given the previously mentioned conditions
theorem find_x : ∃ x, find_score a x := sorry

end find_x_l244_244578


namespace max_area_of_triangle_proof_l244_244019

noncomputable def max_area_of_triangle 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h₁ : a = 2)
  (h₂ : (sin A - sin B) / sin C = (c - b) / (2 + b))
  : ℝ :=
  max (1 / 2 * b * c * sin A) := 
  sqrt 3

theorem max_area_of_triangle_proof 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h₁ : a = 2)
  (h₂ : (sin A - sin B) / sin C = (c - b) / (2 + b))
  : max_area_of_triangle a b c A B C h₁ h₂ = sqrt 3 :=
  sorry

end max_area_of_triangle_proof_l244_244019


namespace value_of_p_circle_tangent_to_directrix_l244_244142

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244142


namespace fraction_meaningful_l244_244552

theorem fraction_meaningful (x : ℝ) : x - 3 ≠ 0 ↔ x ≠ 3 :=
by sorry

end fraction_meaningful_l244_244552


namespace namjoon_used_pencils_l244_244293

variable (taehyungUsed : ℕ) (namjoonUsed : ℕ)

/-- 
Statement:
Taehyung and Namjoon each initially have 10 pencils.
Taehyung gives 3 of his remaining pencils to Namjoon.
After this, Taehyung ends up with 6 pencils and Namjoon ends up with 6 pencils.
We need to prove that Namjoon used 7 pencils.
-/
theorem namjoon_used_pencils (H1 : 10 - taehyungUsed = 9 - 3)
  (H2 : 13 - namjoonUsed = 6) : namjoonUsed = 7 :=
sorry

end namjoon_used_pencils_l244_244293


namespace primes_product_less_than_20_l244_244332

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end primes_product_less_than_20_l244_244332


namespace total_sheets_l244_244785

-- Define the conditions
def sheets_in_bundle : ℕ := 10
def bundles : ℕ := 3
def additional_sheets : ℕ := 8

-- Theorem to prove the total number of sheets Jungkook has
theorem total_sheets : bundles * sheets_in_bundle + additional_sheets = 38 := by
  sorry

end total_sheets_l244_244785


namespace find_other_sides_l244_244907

-- Definitions based on the problem statement
variables (a b c R r : ℝ)

-- Given conditions
def condition1 := (a = 79)
def condition2 := (R = 65)
def condition3 := (r = 28)

-- The Lean statement of the problem
theorem find_other_sides :
  a = 79 → R = 65 → r = 28 → b = 126 ∧ c = 120 :=
by
  intros h_a h_R h_r,
  -- Define and assert game conditions
  have h_cond1 := condition1 a,
  have h_cond2 := condition2 R,
  have h_cond3 := condition3 r,
  rw h_cond1 at h_a,
  rw h_cond2 at h_R,
  rw h_cond3 at h_r,
  -- Sorry to skip the proof
  sorry

end find_other_sides_l244_244907


namespace relationship_among_a_b_c_l244_244463

theorem relationship_among_a_b_c :
  let a := Real.logBase 0.2 0.3
  let b := Real.logBase 1.2 0.8
  let c := 1.5 ^ 0.5
  c > a ∧ a > b :=
by
  let a := Real.logBase 0.2 0.3
  let b := Real.logBase 1.2 0.8
  let c := 1.5 ^ 0.5
  sorry

end relationship_among_a_b_c_l244_244463


namespace nell_baseball_cards_l244_244645

theorem nell_baseball_cards 
  (ace_cards_now : ℕ) 
  (extra_baseball_cards : ℕ) 
  (B : ℕ) : 
  ace_cards_now = 55 →
  extra_baseball_cards = 123 →
  B = ace_cards_now + extra_baseball_cards →
  B = 178 :=
by
  intros h₁ h₂ h₃
  rw [h₁, h₂] at h₃
  exact h₃

end nell_baseball_cards_l244_244645


namespace decimal_111_to_base_5_l244_244431

def decimal_to_base_5 (n : ℕ) : ℕ :=
  let rec loop (n : ℕ) (acc : ℕ) (place : ℕ) :=
    if n = 0 then acc
    else 
      let rem := n % 5
      let q := n / 5
      loop q (acc + rem * place) (place * 10)
  loop n 0 1

theorem decimal_111_to_base_5 : decimal_to_base_5 111 = 421 :=
  sorry

end decimal_111_to_base_5_l244_244431


namespace parabola_p_and_circle_tangent_directrix_l244_244231

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244231


namespace starting_lineup_ways_l244_244649

-- Defining the number of players and the special designations
def num_players := 12
def num_offensive_linemen := 3
def num_running_backs := 4

-- The function to calculate the number of ways to choose the starting lineup
def ways_to_choose_lineup (n m ol rb : Nat) : Nat :=
  ol * rb * (n - 2) * (n - 3)

-- The theorem stating the result we want to prove
theorem starting_lineup_ways : 
  ways_to_choose_lineup num_players num_players num_offensive_linemen num_running_backs = 1080 := 
by
  simp [num_players, num_offensive_linemen, num_running_backs, ways_to_choose_lineup]
  sorry

end starting_lineup_ways_l244_244649


namespace parabola_properties_l244_244191

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244191


namespace arrangement_condition_1_arrangement_condition_2_arrangement_constellation_exactly_one_between_l244_244408

theorem arrangement_condition_1 (A B P Q R S T: Type) (arrangements: List [A, B, P, Q, R, S, T]) : 
  arrangements.head = A ∧ arrangements.last = B → arrangements.length = 7 := sorry

theorem arrangement_condition_2 (boys girls: Type) (arrangements: List [boys, girls]): 
  arrangements.length = 7 ∧ boys.length = 3 ∧ girls.length = 4 → (∃ sublist, sublist.length = 3 ∧ (∀ x ∈ sublist, x ∈ boys)) := sorry

theorem arrangement_constellation (n: ℕ): 
  (∃ arrangements: Set (List Type), arrangements.size = 1440 ∧ ∀ (boys: List Type), boys.length = 3 ∧ ∀ (x y: List Type), x y ∈ boys → x ≠ y ) := sorry

theorem exactly_one_between (A B: Type) (arrangements: List [ A, P, B] ) (P Q R S T: Type):
  arrangements.length = 7 ∧ arrangements.head = A ∧ arrangements.last = B ∧ ∃ single: [P] → single ∈ arrangements := sorry

end arrangement_condition_1_arrangement_condition_2_arrangement_constellation_exactly_one_between_l244_244408


namespace prob_less_than_or_equal_15_l244_244978

noncomputable def prob_between_1_and_15 : ℝ := 1 / 3
noncomputable def prob_at_least_1 : ℝ := 2 / 3

theorem prob_less_than_or_equal_15 : prob_at_least_1 = 2 / 3 → prob_between_1_and_15 = 1 / 3 → prob_at_least_1 = prob_less_than_or_equal_15 :=
by intros h1 h2; exact h1

end prob_less_than_or_equal_15_l244_244978


namespace magic_king_episodes_proof_l244_244710

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l244_244710


namespace simplify_fraction_l244_244666

variable {x y : ℝ}

theorem simplify_fraction (hx : x ≠ 0) : (x * y) / (3 * x) = y / 3 := by
  sorry

end simplify_fraction_l244_244666


namespace smallest_possible_sum_l244_244063

theorem smallest_possible_sum :
  ∃ (B : ℕ) (c : ℕ), B + c = 34 ∧ 
    (B ≥ 0 ∧ B < 5) ∧ 
    (c > 7) ∧ 
    (31 * B = 4 * c + 4) := 
by
  sorry

end smallest_possible_sum_l244_244063


namespace parabola_conditions_l244_244240

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244240


namespace sum_f_a_eq_4046_l244_244889

noncomputable def a : ℕ → ℝ := sorry

def f (x : ℝ) : ℝ := 4 / (1 + x^2)

axiom a1 : a 1 * a 2023 = 1
axiom geometric_seq (n : ℕ) (h : n ≤ 2023) : a n * a (2024 - n) = 1

theorem sum_f_a_eq_4046 : (∑ i in Finset.range 2023, f (a (i + 1))) = 4046 :=
sorry

end sum_f_a_eq_4046_l244_244889


namespace sum_of_cubes_of_ages_l244_244886

noncomputable def dick_age : ℕ := 2
noncomputable def tom_age : ℕ := 5
noncomputable def harry_age : ℕ := 6

theorem sum_of_cubes_of_ages :
  4 * dick_age + 2 * tom_age = 3 * harry_age ∧ 
  3 * harry_age^2 = 2 * dick_age^2 + 4 * tom_age^2 ∧ 
  Nat.gcd (Nat.gcd dick_age tom_age) harry_age = 1 → 
  dick_age^3 + tom_age^3 + harry_age^3 = 349 :=
by
  intros h
  sorry

end sum_of_cubes_of_ages_l244_244886


namespace megan_eggs_per_meal_l244_244266

-- Define the initial conditions
def initial_eggs_from_store : Nat := 12
def initial_eggs_from_neighbor : Nat := 12
def eggs_used_for_omelet : Nat := 2
def eggs_used_for_cake : Nat := 4
def meals_to_divide : Nat := 3

-- Calculate various steps
def total_initial_eggs : Nat := initial_eggs_from_store + initial_eggs_from_neighbor
def eggs_after_cooking : Nat := total_initial_eggs - eggs_used_for_omelet - eggs_used_for_cake
def eggs_after_giving_away : Nat := eggs_after_cooking / 2
def eggs_per_meal : Nat := eggs_after_giving_away / meals_to_divide

-- State the theorem to prove the value of eggs_per_meal
theorem megan_eggs_per_meal : eggs_per_meal = 3 := by
  sorry

end megan_eggs_per_meal_l244_244266


namespace log_prod_arithmetic_sequence_l244_244036

theorem log_prod_arithmetic_sequence :
  ∃ (a : ℕ → ℝ), (f : ℝ → ℝ) (f = λ x, 3^x) ∧ (∀ n, a (n + 1) = a n + 2) ∧
  (f (a 2 + a 4 + a 6 + a 8 + a 10) = 9) →
  (log 3 (f (a 1) * f (a 2) * f (a 3) * f (a 4) * f (a 5) * f (a 6) * f (a 7) * f (a 8) * f (a 9) * f (a 10)) = -6) := 
sorry

end log_prod_arithmetic_sequence_l244_244036


namespace volume_of_given_tetrahedron_is_zero_l244_244878

def point := (ℝ × ℝ × ℝ)

def vector (P Q : point) : point :=
  (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

def determinant (v1 v2 v3 : point) : ℝ :=
  v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) -
  v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) +
  v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)

def volume_of_tetrahedron (A B C D : point) : ℝ :=
  let v1 := vector A B in
  let v2 := vector A C in
  let v3 := vector A D in
  (1 / 6) * (determinant v1 v2 v3)

theorem volume_of_given_tetrahedron_is_zero :
  volume_of_tetrahedron (5, 8, 10) (10, 10, 17) (4, 45, 46) (2, 5, 4) = 0 :=
by
  sorry

end volume_of_given_tetrahedron_is_zero_l244_244878


namespace correct_answer_l244_244283

def f (x : ℝ) : ℝ := 2^x * x - x / 2^x

theorem correct_answer (m n : ℝ) (h : f m < f n) : m^2 < n^2 := 
sorry

end correct_answer_l244_244283


namespace two_digit_primes_count_l244_244517

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits := {3, 5, 7, 9}

def is_valid_two_digit_prime (n : ℕ) : Prop :=
  is_two_digit_number n ∧ is_prime n ∧ 
  ∃ t u : ℕ, t ∈ digits ∧ u ∈ digits ∧ t ≠ u ∧ n = t * 10 + u

theorem two_digit_primes_count : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, is_valid_two_digit_prime k → k < 100 := 
sorry

end two_digit_primes_count_l244_244517


namespace parabola_focus_line_l244_244161

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244161


namespace parallelogram_area_correct_l244_244379

noncomputable def parallelogram_area (x : ℝ) : ℝ :=
  let a := x / Real.sqrt 3 in
  let area := a * (2 * a) * (Real.sqrt 3 / 2) in
  area

theorem parallelogram_area_correct (x : ℝ) (h : 0 < x) :
  parallelogram_area x = x^2 / 3 := by
  have a := x / Real.sqrt 3
  have area := a * (2 * a) * (Real.sqrt 3 / 2)
  have calculation :=
    calc 
      a * (2 * a) * (Real.sqrt 3 / 2)
      = (x / Real.sqrt 3) * (2 * (x / Real.sqrt 3)) * (Real.sqrt 3 / 2) : by sorry
  have expected := x^2 / 3
  rw [calculation] at expected
  exact expected

end parallelogram_area_correct_l244_244379


namespace find_AD_find_a_rhombus_l244_244511

variable (a : ℝ) (AB AD : ℝ)

-- Problem 1: Given AB = 2, find AD
theorem find_AD (h1 : AB = 2)
    (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = AB ∨ x = AD) : AD = 5 := sorry

-- Problem 2: Find the value of a such that ABCD is a rhombus
theorem find_a_rhombus (h_quad : ∀ x, x^2 - (a-4)*x + (a-1) = 0 → x = 2 → AB = AD → x = a ∨ AB = AD → x = 10) :
    a = 10 := sorry

end find_AD_find_a_rhombus_l244_244511


namespace parabola_focus_line_l244_244167

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244167


namespace roots_difference_of_quadratic_l244_244871

theorem roots_difference_of_quadratic :
  ∀ (a b c : ℝ), a = 1 → b = -8 → c = 15 → (∀ x : ℝ, x^2 - 8 * x + 15 = 0) →
    let r1 := (-b + sqrt (b^2 - 4 * a * c)) / (2 * a) in
    let r2 := (-b - sqrt (b^2 - 4 * a * c)) / (2 * a) in
    (r1 * r2 < 20) → (r1 - r2 = 2) :=
by
  intros a b c ha hb hc hq
  have : r1 * r2 = 15 := sorry
  have : (r1 * r2 < 20) := by linarith
  have : (r1 - r2)^2 = 4 := by calc
    (r1 - r2)^2 = (r1 + r2)^2 - 4 * (r1 * r2) := sorry
    ... = 64 - 60 := sorry
    ... = 4 := sorry
  have : |r1 - r2| = sqrt 4 := by linarith
  have : r1 - r2 = 2 := sorry
  exact this

end roots_difference_of_quadratic_l244_244871


namespace sum_of_numbers_l244_244359

theorem sum_of_numbers : 4.75 + 0.303 + 0.432 = 5.485 :=
by
  -- The proof will be filled here
  sorry

end sum_of_numbers_l244_244359


namespace at_least_6_heads_probability_l244_244796

open_locale big_operators

theorem at_least_6_heads_probability : 
  let outcomes := 2 ^ 9 in
  let total_ways := (Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9) in
  total_ways / outcomes = 130 / 512 :=
by
  sorry

end at_least_6_heads_probability_l244_244796


namespace series_identity_l244_244281

theorem series_identity :
  (∑ k in (Finset.range 480).filter (λ k, ¬((k + 1) % 3 = 0)), (1 / (k + 1)) -
   2 * ∑ k in (Finset.range 481), if (k % 3 = 0) then (1 / (k + 1)) else 0) 
  = 2 * ∑ k in (Finset.range 160), (641 / ((161 + k) * (480 - k))) :=
by sorry

end series_identity_l244_244281


namespace satisfactory_fraction_l244_244809

theorem satisfactory_fraction :
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  satisfactory_grades / total_students = 7 / 10 :=
by
  let num_students_A := 8
  let num_students_B := 7
  let num_students_C := 6
  let num_students_D := 5
  let num_students_F := 4
  let satisfactory_grades := num_students_A + num_students_B + num_students_C
  let total_students := num_students_A + num_students_B + num_students_C + num_students_D + num_students_F
  have h1: satisfactory_grades = 21 := by sorry
  have h2: total_students = 30 := by sorry
  have fraction := (satisfactory_grades: ℚ) / total_students
  have simplified_fraction := fraction = 7 / 10
  exact sorry

end satisfactory_fraction_l244_244809


namespace theater_seats_l244_244816

theorem theater_seats
  (A : ℕ) -- Number of adult tickets
  (C : ℕ) -- Number of child tickets
  (hC : C = 63) -- 63 child tickets sold
  (total_revenue : ℕ) -- Total Revenue
  (hRev : total_revenue = 519) -- Total revenue is 519
  (adult_ticket_price : ℕ := 12) -- Price per adult ticket
  (child_ticket_price : ℕ := 5) -- Price per child ticket
  (hRevEq : adult_ticket_price * A + child_ticket_price * C = total_revenue) -- Revenue equation
  : A + C = 80 := sorry

end theater_seats_l244_244816


namespace compare_magnitudes_l244_244775

theorem compare_magnitudes (a b c d e : ℝ) (h₁ : a > b) (h₂ : b > 0) (h₃ : c < d) (h₄ : d < 0) (h₅ : e < 0) :
  (e / (a - c)) > (e / (b - d)) :=
  sorry

end compare_magnitudes_l244_244775


namespace problem_solution_l244_244669

open Function

-- Definition of g
def g : ℝ → ℝ
| -1 := 0
| 0 := 1
| 1 := 3
| 2 := 4
| 3 := 6
| x := sorry  -- For values not in the table

-- Definition of g inverse
def g_inv : ℝ → ℝ
| 0 := -1
| 1 := 0
| 3 := 1
| 4 := 2
| 6 := 3
| x := sorry  -- For values not in the table

-- Statement of the theorem
theorem problem_solution :
  g (g 1) + g (g_inv 2) + g_inv (g_inv 3) = 8 :=
by
  -- Just to skip the proof
  sorry

end problem_solution_l244_244669


namespace acute_triangle_tan_squared_geq_9_obtuse_triangle_tan_cot_squared_geq_9_l244_244366

-- Problem 1: Prove that for an acute triangle \( \triangle ABC \): \[ \tan^2 A + \tan^2 B + \tan^2 C \geq 9 \]
theorem acute_triangle_tan_squared_geq_9 {A B C : ℝ} (hA : 0 < A ∧ A < π/2) (hB : 0 < B ∧ B < π/2) (hC : 0 < C ∧ C < π/2) (hSum : A + B + C = π) : 
tan A ^ 2 + tan B ^ 2 + tan C ^ 2 ≥ 9 := sorry

-- Problem 2: Prove that for an obtuse triangle \( \triangle ABC \): \[ \tan^2 A + \cot^2 B + \cot^2 C \geq 9 \]
theorem obtuse_triangle_tan_cot_squared_geq_9 {A B C : ℝ} (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (hObtuse : A > π/2 ∨ B > π/2 ∨ C > π/2) (hSum : A + B + C = π) : 
tan A ^ 2 + cot B ^ 2 + cot C ^ 2 ≥ 9 := sorry

end acute_triangle_tan_squared_geq_9_obtuse_triangle_tan_cot_squared_geq_9_l244_244366


namespace distance_of_parallel_lines_l244_244969

noncomputable def line1_distance_from_line2 (a b : ℝ) : ℝ :=
  let l1 := a * x + (1 - b) * y + 5
  let l2 := (1 + a) * x - y - b
  let l3 := x - 2 * y + 3
  let parallel_cond := (-2) * (a * x + (1 - b) * y + 5) = 
                       (x - 2 * y + 3) ∧ 
                       (-2) * ((1 + a) * x - y - b) = 
                       (x - 2 * y + 3)
  let a_value := -1/2
  let b_value := 0
  d / (real.sqrt (1^2 + (-2 : ℝ)^2)) = 2 * real.sqrt 5

theorem distance_of_parallel_lines (a b : ℝ) (l1 l2 l3 : ℝ) :
  l1 ∥ l3 ∧ l2 ∥ l3 → line1_distance_from_line2 a b = 2 * real.sqrt 5 :=
  sorry

end distance_of_parallel_lines_l244_244969


namespace sequence_is_decreasing_l244_244259

noncomputable def f₁ (x : ℝ) : ℝ := x * Real.exp x
noncomputable def fₙ (n : ℕ) (x : ℝ) : ℝ :=
  (Nat.rec (λ α x, f₁ x) (λ n fn x, (fun f => (fun x => (f (x+1))) (f x)) x) n) x

def lowest_point (n : ℕ) : ℝ × ℝ :=
  (-↑n, -1 / Real.exp n)

def area (n : ℕ) : ℝ :=
  let Pₙ := lowest_point n
  let Pₙ₊₁ := lowest_point (n + 1)
  let Pₙ₊₂ := lowest_point (n + 2)
  let distance_pn_pn1 := Real.sqrt (1 + 1 / (Real.exp (2 * n)) * (1 - 1 / Real.exp 1) ^ 2)
  let d := (λ Pₙ Pₙ₊₁ Pₙ₊₂,
    (n * Real.exp 2 + Real.exp 2 - 2 * Real.exp (n + 1) + 1) / 
    Real.sqrt ((Real.exp 1 - 1) ^ 2 + Real.exp (2 * (n + 1)))
  ) Pₙ Pₙ₊₁ Pₙ₊₂
  (1 / 2) * distance_pn_pn1 * d

theorem sequence_is_decreasing : ∀ n : ℕ, area (n+1) < area n :=
sorry

end sequence_is_decreasing_l244_244259


namespace intersect_complement_l244_244941

-- Definition of the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Definition of set A
def A : Set ℕ := {1, 2, 3}

-- Definition of set B
def B : Set ℕ := {3, 4}

-- Definition of the complement of B in U
def CU (U : Set ℕ) (B : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ B}

-- Expected result of the intersection
def result : Set ℕ := {1, 2}

-- The proof statement
theorem intersect_complement :
  A ∩ CU U B = result :=
sorry

end intersect_complement_l244_244941


namespace max_value_x_sqrt_1_minus_x_sq_l244_244956

theorem max_value_x_sqrt_1_minus_x_sq (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : ∃ y, y = x + real.sqrt (1 - x^2) ∧ y ≤ real.sqrt 2 
:= sorry

end max_value_x_sqrt_1_minus_x_sq_l244_244956


namespace functions_equivalence_l244_244402

theorem functions_equivalence :
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = log 3 (log 3 x) ↔ x ≠ 0) ∧ 
    (∀ x, g x = x ↔ x ≠ 0) → 
    (∀ x, f x = g x) := by
  -- the proof goes here
  sorry

end functions_equivalence_l244_244402


namespace number_of_multiples_l244_244057

theorem number_of_multiples (n : ℕ) (h : n = 4050) : 
  let multiples_of_5 := (n / 5),
      multiples_of_7 := (n / 7),
      multiples_of_35 := (n / 35)
  in multiples_of_5 + multiples_of_7 - multiples_of_35 = 1273 := 
by {
  let multiples_of_5 := (n / 5);
  let multiples_of_7 := (n / 7);
  let multiples_of_35 := (n / 35);
  have : multiples_of_5 + multiples_of_7 - multiples_of_35 = 1273,
  sorry
}

end number_of_multiples_l244_244057


namespace expression_evaluation_l244_244864

theorem expression_evaluation (a b c : ℤ) 
  (h1 : c = a + 8) 
  (h2 : b = a + 4) 
  (h3 : a = 5) 
  (h4 : a + 2 ≠ 0) 
  (h5 : b - 3 ≠ 0) 
  (h6 : c + 7 ≠ 0) : 
  (a + 3) / (a + 2) * (b - 2) / (b - 3) * (c + 10) / (c + 7) = 23/15 :=
by
  sorry

end expression_evaluation_l244_244864


namespace hyperbola_equation_l244_244489

-- Definitions based on conditions
def parabola_focus (p : ℝ) : ℝ × ℝ := (0, p / 2)

def hyperbola_asymptote_ratio (a b : ℝ) : Prop :=
  a / b = 3 / 4

def hyperbola_focus_distance (a b c : ℝ) : Prop :=
  real.sqrt (a^2 + b^2) = c

-- The proof problem
theorem hyperbola_equation
  (p : ℝ)
  (focus : ℝ × ℝ)
  (a b c : ℝ)
  (h_parabola_focus : focus = parabola_focus p)
  (h_asymptote : hyperbola_asymptote_ratio a b)
  (h_focus_distance : hyperbola_focus_distance a b c)
  (h_c_value : c = 5) :
  a = 3 ∧ b = 4 ∧ (a = 3 → b = 4 → (∀ x y : ℝ, (y ^ 2 / a ^ 2 - x ^ 2 / b ^ 2 = 1) ↔ (y ^ 2 / 9 - x ^ 2 / 16 = 1))) :=
by
  sorry

end hyperbola_equation_l244_244489


namespace magic_king_total_episodes_l244_244708

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l244_244708


namespace solve_for_a_l244_244549

theorem solve_for_a (a : ℝ) : 
  (2 * a + 16 + 3 * a - 8) / 2 = 69 → a = 26 :=
by
  sorry

end solve_for_a_l244_244549


namespace prob_of_interval_l244_244030

open ProbabilityTheory

noncomputable def normal_ξ := ⁇ -- some placeholder for ξ, as Lean doesn't have direct access to named random variables from a distribution

theorem prob_of_interval 
  (σ : ℝ)
  (h₁ : normalDistribution 1 σ)
  (h₂ : P (λ x, x < 2) = 0.6) :
  P (λ x, 0 < x ∧ x < 1) = 0.1 := 
sorry

end prob_of_interval_l244_244030


namespace solve_for_b_l244_244304

theorem solve_for_b {b : ℝ} :
  let line1 := 3 * y - 2 * x + 1 = 0,
      line2 := 4 * y + b * x - 8 = 0,
      slope1 := 2 / 3,
      slope2 := -b / 4
  in (slope1 * slope2 = -1) → b = 6 :=
by
  intro h,
  simp at *,
  sorry

end solve_for_b_l244_244304


namespace abs_eq_5_necessity_l244_244336

variable {x : ℝ}

theorem abs_eq_5_necessity (h : x = 5) : |x| = 5 :=
by {
  rw h,
  simp
}

end abs_eq_5_necessity_l244_244336


namespace stuffed_animals_total_l244_244640

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l244_244640


namespace parabola_conditions_l244_244239

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244239


namespace problem_l244_244126

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244126


namespace p_eq_two_circle_tangent_proof_l244_244221

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244221


namespace three_digit_powers_of_3_l244_244434

theorem three_digit_powers_of_3 : 
  {n : ℤ // 100 ≤ 3^n ∧ 3^n ≤ 999}.finite ∧
  {n : ℤ // 100 ≤ 3^n ∧ 3^n ≤ 999}.to_finset.card = 2 := 
by 
  sorry

end three_digit_powers_of_3_l244_244434


namespace percentage_increase_visitors_l244_244827

theorem percentage_increase_visitors 
  (V_Oct : ℕ)
  (V_Nov V_Dec : ℕ)
  (h1 : V_Oct = 100)
  (h2 : V_Dec = V_Nov + 15)
  (h3 : V_Oct + V_Nov + V_Dec = 345) : 
  (V_Nov - V_Oct) * 100 / V_Oct = 15 := 
by 
  sorry

end percentage_increase_visitors_l244_244827


namespace f1_satisfies_f2_satisfies_f4_satisfies_l244_244823

def f1 (x : ℝ) : ℝ := |2 * x|
def f2 (x : ℝ) : ℝ := x
def f3 (x : ℝ) : ℝ := sqrt x
def f4 (x : ℝ) : ℝ := x - |x|

theorem f1_satisfies : ∀ x : ℝ, f1 (2 * x) = 2 * f1 x := by sorry
theorem f2_satisfies : ∀ x : ℝ, f2 (2 * x) = 2 * f2 x := by sorry
theorem f4_satisfies : ∀ x : ℝ, f4 (2 * x) = 2 * f4 x := by sorry

end f1_satisfies_f2_satisfies_f4_satisfies_l244_244823


namespace extremum_value_monotonicity_l244_244499

noncomputable def f (a x : ℝ) : ℝ := (x^2 + 1) * Real.exp (a * x)

theorem extremum_value (a : ℝ) (h : (f a 0.5).derivative = 0) :
  f a 0.5 = (5 / 4) * Real.exp (-2 / 5) := sorry

theorem monotonicity (a : ℝ) :
  (if a = 0 then ∀ x, (f a x).derivative ≥ 0 else
  if a > 1 then ∀ x, (f a x).derivative > 0 else
  if 0 < a ∧ a < 1 then ∃ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧
  ((f a x1).derivative < 0 ∧ (f a x2).derivative > 0) else
  if -1 < a ∧ a < 0 then ∃ x1 x2 : ℝ, -1 > x1 ∧ x2 ∈ Ioo x1 1 ∧ x2 > 1 ∧
  ((f a x1).derivative < 0 ∧ (f a x2).derivative > 0)) := sorry

end extremum_value_monotonicity_l244_244499


namespace triangle_arithmetic_sequence_b_value_l244_244584

theorem triangle_arithmetic_sequence_b_value
  (a b c : ℝ)
  (h_seq : 2 * b = a + c)
  (h_angle : ∠ABC = (30 : ℝ) * (Real.pi / 180))
  (h_area : (1 / 2) * a * c * (Real.sin (Real.pi / 6)) = 3 / 2) :
  b = 1 + Real.sqrt 3 :=
by
  sorry

end triangle_arithmetic_sequence_b_value_l244_244584


namespace probability_one_pair_l244_244887

theorem probability_one_pair (total_pairs : ℕ) (total_gloves : ℕ) (select_gloves : ℕ) :
  total_pairs = 6 → total_gloves = 12 → select_gloves = 4 →
  let total_outcomes := Nat.choose total_gloves select_gloves in
  let favorable_outcomes := Nat.choose total_pairs 1 * Nat.choose (total_pairs - 1) 2 * Nat.choose 2 1 * Nat.choose 2 1 in
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 16 / 33 := 
by
  intros h_pairs h_gloves h_select
  rw [h_pairs, h_gloves, h_select]
  have total_outcomes : ℚ := Nat.choose 12 4
  have favorable_outcomes : ℚ := Nat.choose 6 1 * Nat.choose 5 2 * Nat.choose 2 1 * Nat.choose 2 1
  have prob : ℚ := favorable_outcomes / total_outcomes
  exact sorry

end probability_one_pair_l244_244887


namespace price_increase_percentage_l244_244568

theorem price_increase_percentage (x : ℝ) :
  (0.9 * (1 + x / 100) * 0.9259259259259259 = 1) → x = 20 :=
by
  intros
  sorry

end price_increase_percentage_l244_244568


namespace midpoint_to_plane_distance_l244_244920

noncomputable def distance_to_plane (A B P: ℝ) (dA dB: ℝ) : ℝ :=
if h : A = B then |dA|
else if h1 : dA + dB = (2 : ℝ) * (dA + dB) / 2 then (dA + dB) / 2
else if h2 : |dB - dA| = (2 : ℝ) * |dB - dA| / 2 then |dB - dA| / 2
else 0

theorem midpoint_to_plane_distance
  (α : Type*)
  (A B P: ℝ)
  {dA dB : ℝ}
  (h_dA : dA = 3)
  (h_dB : dB = 5) :
  distance_to_plane A B P dA dB = 4 ∨ distance_to_plane A B P dA dB = 1 :=
by sorry

end midpoint_to_plane_distance_l244_244920


namespace parabola_focus_line_l244_244162

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244162


namespace cube_odd_minus_itself_div_by_24_l244_244655

theorem cube_odd_minus_itself_div_by_24 (n : ℤ) : 
  (2 * n + 1)^3 - (2 * n + 1) ≡ 0 [MOD 24] := 
by 
  sorry

end cube_odd_minus_itself_div_by_24_l244_244655


namespace distinct_three_digit_numbers_with_average_and_digit_5_l244_244516

theorem distinct_three_digit_numbers_with_average_and_digit_5 :
  ∃ n : ℕ, n = 18 ∧
  ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b + c) % 3 = 0 ∧
  (a ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (b ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (c ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (5 ∈ [a, b, c]) →
  ∃ l : List ℕ, l.length = n ∧
  ∀ (x : ℕ), x ∈ l →
  let digits := [x / 100, (x / 10) % 10, x % 10] in
  (digits.nodup) ∧
  ∃ m : ℕ, 2 * m = (digits.head + digits.nth 2).sum ∧
  5 ∈ digits.toList :=
sorry

end distinct_three_digit_numbers_with_average_and_digit_5_l244_244516


namespace parabola_properties_l244_244185

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244185


namespace steve_speed_on_way_back_l244_244686

def steve_speed_to_work (v : ℝ) : ℝ := 30 / v 
def steve_speed_back_home (v : ℝ) : ℝ := 30 / (2 * v)

theorem steve_speed_on_way_back :
  ∃ v : ℝ, (steve_speed_to_work v + steve_speed_back_home v = 6) ∧ (2 * v = 15) :=
begin
  sorry
end

end steve_speed_on_way_back_l244_244686


namespace ring_sector_area_l244_244725

theorem ring_sector_area (θ : ℝ) : 
  let r1 := 13
      r2 := 7
      area (r : ℝ) := 1/2 * r^2 * θ
  in area r1 - area r2 = 60 * θ :=
by
  sorry

end ring_sector_area_l244_244725


namespace range_of_d_l244_244902

noncomputable def sn (n a1 d : ℝ) := (n / 2) * (2 * a1 + (n - 1) * d)

theorem range_of_d (a1 d : ℝ) (h_eq : (sn 2 a1 d) * (sn 4 a1 d) / 2 + (sn 3 a1 d) ^ 2 / 9 + 2 = 0) :
  d ∈ Set.Iic (-Real.sqrt 2) ∪ Set.Ici (Real.sqrt 2) :=
sorry

end range_of_d_l244_244902


namespace p_eq_two_circle_tangent_proof_l244_244219

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244219


namespace sum_of_coordinates_of_D_l244_244459

-- Definition of points M, C and D
structure Point where
  x : ℝ
  y : ℝ

def M : Point := ⟨4, 7⟩
def C : Point := ⟨6, 2⟩

-- Conditions that M is the midpoint of segment CD
def isMidpoint (M C D : Point) : Prop :=
  ((C.x + D.x) / 2 = M.x) ∧
  ((C.y + D.y) / 2 = M.y)

-- Definition for the sum of the coordinates of a point
def sumOfCoordinates (P : Point) : ℝ :=
  P.x + P.y

-- The main theorem stating the sum of the coordinates of D is 14 given the conditions
theorem sum_of_coordinates_of_D :
  ∃ D : Point, isMidpoint M C D ∧ sumOfCoordinates D = 14 := 
sorry

end sum_of_coordinates_of_D_l244_244459


namespace problem1_problem2_l244_244052

noncomputable theory

variables {α x : ℝ}
variables (a b c : ℝ × ℝ)

def vector_a (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def vector_c (α x : ℝ) : ℝ × ℝ := (Real.sin x + 2 * Real.sin α, Real.cos x + 2 * Real.cos α)

def f (x : ℝ) (α : ℝ) : ℝ :=
let b := vector_b x in let c := vector_c α x in
b.1 * c.1 + b.2 * c.2

-- Problem 1
theorem problem1 (h1 : α = Real.pi / 4) :
  ∃ x, (Real.pi / 4 < x ∧ x < Real.pi) ∧ f x α = -3 / 2 ∧ x = 11 * Real.pi / 12 :=
by sorry

-- Problem 2
theorem problem2 (h2 : ∀ (a b : ℝ × ℝ), (a = vector_a α) → (b = vector_b x) → 
  0 < α → α < x → x < Real.pi → (a.1 * b.1 + a.2 * b.2 = Real.cos (x - α)) ∧ 
  (a.1 * (vector_c α x).1 + a.2 * (vector_c α x).2 = 0) ∧ (x - α = Real.pi / 3)) :
  x - α = Real.pi / 3 → tan (2 * α) = -Real.sqrt 3 / 5 :=
by sorry

end problem1_problem2_l244_244052


namespace minimal_board_size_for_dominoes_l244_244288

def board_size_is_minimal (n: ℕ) (total_area: ℕ) (domino_size: ℕ) (num_dominoes: ℕ) : Prop :=
  ∀ m: ℕ, m < n → ¬ (total_area ≥ m * m ∧ m * m = num_dominoes * domino_size)

theorem minimal_board_size_for_dominoes (n: ℕ) :
  board_size_is_minimal 77 2008 2 1004 :=
by
  sorry

end minimal_board_size_for_dominoes_l244_244288


namespace total_stuffed_animals_l244_244636

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l244_244636


namespace lillys_fish_l244_244261

variable (Lilly Rosy Total : ℕ)
variable (cond1 : Rosy = 14)
variable (cond2 : Total = 24)

theorem lillys_fish : Lilly + Rosy = Total → Lilly = 10 :=
by
  intros h
  rw [cond1, cond2] at h
  have Lilly_eq := Nat.sub_eq_of_eq_add h
  exact Lilly_eq

end lillys_fish_l244_244261


namespace factorial_expression_simplifies_l244_244416

theorem factorial_expression_simplifies :
    8! - 7 * 7! - 2 * 7! = -5040 := 
by
  sorry

end factorial_expression_simplifies_l244_244416


namespace two_digit_primes_count_l244_244519

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits := {3, 5, 7, 9}

def is_valid_two_digit_prime (n : ℕ) : Prop :=
  is_two_digit_number n ∧ is_prime n ∧ 
  ∃ t u : ℕ, t ∈ digits ∧ u ∈ digits ∧ t ≠ u ∧ n = t * 10 + u

theorem two_digit_primes_count : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, is_valid_two_digit_prime k → k < 100 := 
sorry

end two_digit_primes_count_l244_244519


namespace find_t_l244_244065

open Real

theorem find_t (t : ℝ) (h₀ : t ∈ Ioo 0 π) (h₁ : sin (2 * t) = -∫ x in 0..π, cos x) : t = π / 2 :=
by
  sorry

end find_t_l244_244065


namespace ff_2_eq_neg_4_over_3_l244_244932

noncomputable def f : ℝ → ℝ :=
λ x, if x ≥ 1 then (1 / 2) ^ x else 1 / (x - 1)

theorem ff_2_eq_neg_4_over_3 : f (f 2) = -4 / 3 :=
by
  sorry

end ff_2_eq_neg_4_over_3_l244_244932


namespace power_function_f_at_3_is_1_over_9_l244_244967

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - 2*m - 2) * x^(m-1)

-- Conditions
axiom exists_m : ∃ m : ℝ, (m^2 - 2*m - 2 > 0) ∧ (m - 1 < 0)

-- Theorem statement
theorem power_function_f_at_3_is_1_over_9 : ∀ (x : ℝ), 0 < x → (∃ m : ℝ, (m^2 - 2*m - 2 > 0) ∧ 
                                                              (m - 1 < 0)) → 
                                                              f (-1) 3 = 1 / 9 :=
by
  intro x hx
  intro ⟨m, h₁, h₂⟩
  sorry

end power_function_f_at_3_is_1_over_9_l244_244967


namespace lattice_points_count_l244_244944

def is_lattice_point (x y : ℤ) : Prop := x^2 + y^2 = 72

theorem lattice_points_count : 
  { p : ℤ × ℤ // is_lattice_point p.1 p.2 }.card = 12 :=
sorry

end lattice_points_count_l244_244944


namespace cube_root_inequality_l244_244545

theorem cube_root_inequality (a b : ℝ) (h : a > b) : (a ^ (1/3)) > (b ^ (1/3)) :=
sorry

end cube_root_inequality_l244_244545


namespace eight_child_cotton_l244_244989

theorem eight_child_cotton {a_1 a_8 d S_8 : ℕ} 
  (h1 : d = 17)
  (h2 : S_8 = 996)
  (h3 : 8 * a_1 + 28 * d = S_8) :
  a_8 = a_1 + 7 * d → a_8 = 184 := by
  intro h4
  subst_vars
  sorry

end eight_child_cotton_l244_244989


namespace cookies_sum_l244_244829

theorem cookies_sum (C : ℕ) (h1 : C % 6 = 5) (h2 : C % 9 = 7) (h3 : C < 80) :
  C = 29 :=
by sorry

end cookies_sum_l244_244829


namespace parabola_conditions_l244_244237

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244237


namespace prove_p_equals_2_l244_244199

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244199


namespace sin_minus_cos_eq_neg_sqrt_14_over_3_l244_244462

theorem sin_minus_cos_eq_neg_sqrt_14_over_3 (θ : ℝ) (h1 : θ ∈ set.Ioo (-real.pi / 2) (real.pi / 2)) (h2 : real.sin θ + real.cos θ = 2/3) :
  real.sin θ - real.cos θ = -real.sqrt 14 / 3 :=
by
  sorry

end sin_minus_cos_eq_neg_sqrt_14_over_3_l244_244462


namespace max_value_of_f_l244_244695

noncomputable def f (x : ℝ) : ℝ := x * (4 - x)

theorem max_value_of_f : ∃ y, ∀ x ∈ Set.Ioo 0 4, f x ≤ y ∧ y = 4 :=
by
  sorry

end max_value_of_f_l244_244695


namespace problem_statement_l244_244071

-- Definitions of A and B based on the given conditions
def A : ℤ := -5 * -3
def B : ℤ := 2 - 2

-- The theorem stating that A + B = 15
theorem problem_statement : A + B = 15 := 
by 
  sorry

end problem_statement_l244_244071


namespace lamps_turn_on_l244_244647

theorem lamps_turn_on (n : ℕ) :
  (∃ k : ℕ, (∀ m : ℕ, m < n → (initial_lit m = false))) ↔ (n % 3 = 1 ∨ n % 3 = 2) :=
sorry

end lamps_turn_on_l244_244647


namespace sqrt_square_result_l244_244422

theorem sqrt_square_result : 
  (let x := Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 3)) in x * x = 3.932) :=
sorry

end sqrt_square_result_l244_244422


namespace tank_volume_l244_244343

theorem tank_volume
  (pump_capacity : ℝ)
  (pump_efficiency : ℝ)
  (num_pumps : ℕ)
  (fill_percentage : ℝ)
  (total_volume : ℝ) :
  pump_capacity = 150 →
  pump_efficiency = 0.75 →
  num_pumps = 8 →
  fill_percentage = 0.85 →
  0.85 * total_volume = 8 * (150 * 0.75) →
  total_volume = 900 / 0.85 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4] at h5
  ring at h5
  exact h5

end tank_volume_l244_244343


namespace parabola_focus_line_l244_244168

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244168


namespace range_of_m_l244_244466

def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*x + m ≠ 0
def q (m : ℝ) : Prop := ∃ y : ℝ, ∀ x : ℝ, (x^2)/(m-1) + y^2 = 1
def not_p (m : ℝ) : Prop := ¬ (p m)
def p_and_q (m : ℝ) : Prop := (p m) ∧ (q m)

theorem range_of_m (m : ℝ) : (¬ (not_p m) ∧ ¬ (p_and_q m)) → 1 < m ∧ m ≤ 2 :=
sorry

end range_of_m_l244_244466


namespace percentage_of_full_marks_D_l244_244356

theorem percentage_of_full_marks_D (full_marks a b c d : ℝ)
  (h_full_marks : full_marks = 500)
  (h_a : a = 360)
  (h_a_b : a = b - 0.10 * b)
  (h_b_c : b = c + 0.25 * c)
  (h_c_d : c = d - 0.20 * d) :
  d / full_marks * 100 = 80 :=
by
  sorry

end percentage_of_full_marks_D_l244_244356


namespace tan_alpha_plus_2beta_l244_244026

theorem tan_alpha_plus_2beta 
  (α β : ℝ) 
  (h_acα : 0 < α ∧ α < π / 2) 
  (h_acβ : 0 < β ∧ β < π / 2) 
  (h_tan_α : Real.tan α = 1 / 7) 
  (h_sin_β : Real.sin β = sqrt 10 / 10) : 
  Real.tan (α + 2 * β) = 1 := 
sorry

end tan_alpha_plus_2beta_l244_244026


namespace corrected_mean_is_36_74_l244_244310

noncomputable def corrected_mean (incorrect_mean : ℝ) 
(number_of_observations : ℕ) 
(correct_value wrong_value : ℝ) : ℝ :=
(incorrect_mean * number_of_observations - wrong_value + correct_value) / number_of_observations

theorem corrected_mean_is_36_74 :
  corrected_mean 36 50 60 23 = 36.74 :=
by
  sorry

end corrected_mean_is_36_74_l244_244310


namespace prove_p_equals_2_l244_244193

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244193


namespace evaluate_expression_l244_244419

theorem evaluate_expression : 1 - (-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := 
by
  sorry

end evaluate_expression_l244_244419


namespace digit_difference_2500_in_bases_l244_244945

theorem digit_difference_2500_in_bases : 
  let n := 2500 in
  (nat.log n 2).nat_abs + 1 = 12 ∧ (nat.log n 7).nat_abs + 1 = 5 
  → 12 - 5 = 7 :=
by {
  intros h,
  cases h with h2 h7,
  rw [h2, h7],
  sorry
}

end digit_difference_2500_in_bases_l244_244945


namespace circle_equation_l244_244300

noncomputable def center_on_y_axis_and_radius := ∀ (a : ℝ), let center := (0, a) in abs (1 - center.1) = 1
noncomputable def circle_passing_through_point := ∀ (a : ℝ), let center := (0, a) in
  (1 - center.1)^2 + (3 - center.2)^2 = 1

theorem circle_equation
  (a : ℝ)
  (h_center : a = 3)
  (h_radius : center_on_y_axis_and_radius a)
  (h_passing : circle_passing_through_point a)
  : ∀ (x y : ℝ), x^2 + (y - a)^2 = 1 := by sorry

end circle_equation_l244_244300


namespace distance_C_to_line_AK_l244_244386

-- Defining the basic setup of the problem
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (1, 1)
def C : ℝ × ℝ := (1, 0)
def D : ℝ × ℝ := (0, 0)

def K : ℝ × ℝ := (2 / 3, 0)

-- The equation of the line AK
def line_AK (x : ℝ) : ℝ := (-3 / 2) * x + 1

-- Using the distance formula from a point to a line
def distance_point_to_line (x₁ y₁ : ℝ) (A B C : ℝ) : ℝ :=
  abs (A * x₁ + B * y₁ + C) / (real.sqrt (A * A + B * B))

-- The specific distance from vertex C to line AK
theorem distance_C_to_line_AK : distance_point_to_line 1 0 3 (-2) (-2) = 1 / real.sqrt 13 :=
by sorry

end distance_C_to_line_AK_l244_244386


namespace find_omega_intervals_of_monotonicity_l244_244461

noncomputable def a (ω x : ℝ) : ℝ × ℝ :=
  (-Real.sqrt 3 * Real.sin (ω * x), Real.cos (ω * x))

noncomputable def b (ω x : ℝ) : ℝ × ℝ :=
  (Real.cos (ω * x), Real.cos (ω * x))

noncomputable def f (ω x : ℝ) : ℝ :=
  let ⟨ax, ay⟩ := a ω x
  let ⟨bx, by⟩ := b ω x
  ax * bx + ay * by

theorem find_omega (ω : ℝ) (hω : ω > 0) (hx : ∃ a b, a ∈ ℝ ∧ b ∈ ℝ ∧ f ω a = f ω (a + π)) : ω = 1 :=
  sorry

theorem intervals_of_monotonicity (k : ℤ) : 
  (∀ x, x ∈ Set.Icc (k * Real.pi - Real.pi/3) (k * Real.pi + 2 * Real.pi/3) -> f 1 x = -Real.sin (2*x - Real.pi/6) + 1/2 -> strict_antimono f)
  ∧ 
  (∀ x, x ∈ Set.Icc (k * Real.pi + 2 * Real.pi/3) (k * Real.pi + 5 * Real.pi/3) -> f 1 x = -Real.sin (2*x - Real.pi/6) + 1/2 -> strict_mono f) :=
  sorry

end find_omega_intervals_of_monotonicity_l244_244461


namespace total_sum_of_money_l244_244758

theorem total_sum_of_money (x : ℝ) (A B C : ℝ) 
  (hA : A = x) 
  (hB : B = 0.65 * x) 
  (hC : C = 0.40 * x) 
  (hC_share : C = 32) :
  A + B + C = 164 := 
  sorry

end total_sum_of_money_l244_244758


namespace parabola_shift_right_l244_244676

theorem parabola_shift_right (x : ℝ) :
  let original_parabola := - (1 / 2) * x^2
  let shifted_parabola := - (1 / 2) * (x - 1)^2
  original_parabola = shifted_parabola :=
sorry

end parabola_shift_right_l244_244676


namespace domain_part1_domain_part2_l244_244446

noncomputable theory

-- Definition for part 1
def domain_of_ln_sqrt (x : ℝ) : Prop :=
  (1 + 1 / x > 0) ∧ (1 - x^2 ≥ 0)

-- Proof statement for part 1
theorem domain_part1 (x : ℝ) : 
  domain_of_ln_sqrt x ↔ (0 < x ∧ x ≤ 1) := sorry

-- Definition for part 2
def domain_of_ln_div (x : ℝ) : Prop :=
  (x + 1 > 0) ∧ (-x^2 - 3*x + 4 > 0)

-- Proof statement for part 2
theorem domain_part2 (x : ℝ) : 
  domain_of_ln_div x ↔ (-1 < x ∧ x < 1) := sorry

end domain_part1_domain_part2_l244_244446


namespace unit_vector_collinear_with_a_l244_244745

-- Given vector a
def a : ℝ × ℝ × ℝ := (3, 0, -4)

-- Define vector option D
def option_d : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

-- Define the condition for collinearity
def collinear (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * (v.1), k * (v.2), k * (v.3))

-- Define the condition for unit vector
def is_unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 + v.3^2 = 1

-- The main theorem statement
theorem unit_vector_collinear_with_a : 
  is_unit_vector option_d ∧ collinear option_d a :=
sorry

end unit_vector_collinear_with_a_l244_244745


namespace sales_worth_l244_244353

variable (S : ℝ)
def old_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)

theorem sales_worth :
  new_remuneration S = old_remuneration S + 600 → S = 24000 :=
by
  intro h
  sorry

end sales_worth_l244_244353


namespace probability_of_high_quality_l244_244724

def P_H1 : ℝ := 2 / 3
def P_H2 : ℝ := 1 / 3
def P_A_given_H1 : ℝ := 0.9
def P_A_given_H2 : ℝ := 0.81

theorem probability_of_high_quality : (P_H1 * P_A_given_H1 + P_H2 * P_A_given_H2) = 0.87 :=
by {
  -- Compute the total probability P(A)
  -- P(A) = P(H1) * P(A | H1) + P(H2) * P(A | H2)
  sorry
}

end probability_of_high_quality_l244_244724


namespace parabola_focus_line_tangent_circle_l244_244211

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244211


namespace volume_of_given_tetrahedron_is_zero_l244_244879

def point := (ℝ × ℝ × ℝ)

def vector (P Q : point) : point :=
  (Q.1 - P.1, Q.2 - P.2, Q.3 - P.3)

def determinant (v1 v2 v3 : point) : ℝ :=
  v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) -
  v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) +
  v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)

def volume_of_tetrahedron (A B C D : point) : ℝ :=
  let v1 := vector A B in
  let v2 := vector A C in
  let v3 := vector A D in
  (1 / 6) * (determinant v1 v2 v3)

theorem volume_of_given_tetrahedron_is_zero :
  volume_of_tetrahedron (5, 8, 10) (10, 10, 17) (4, 45, 46) (2, 5, 4) = 0 :=
by
  sorry

end volume_of_given_tetrahedron_is_zero_l244_244879


namespace equation_of_chord_line_l244_244916

theorem equation_of_chord_line (m n s t : ℝ)
  (h₀ : m > 0) (h₁ : n > 0) (h₂ : s > 0) (h₃ : t > 0)
  (h₄ : m + n = 3)
  (h₅ : m / s + n / t = 1)
  (h₆ : m < n)
  (h₇ : s + t = 3 + 2 * Real.sqrt 2)
  (h₈ : ∃ x1 x2 y1 y2 : ℝ, 
        (x1 + x2) / 2 = m ∧ (y1 + y2) / 2 = n ∧
        x1 ^ 2 / 4 + y1 ^ 2 / 16 = 1 ∧
        x2 ^ 2 / 4 + y2 ^ 2 / 16 = 1) 
  : 2 * m + n - 4 = 0 := sorry

end equation_of_chord_line_l244_244916


namespace parabola_focus_line_tangent_circle_l244_244207

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244207


namespace smallest_x_for_gg_defined_l244_244075

def g (x : ℝ) : ℝ := sqrt (x - 5)

theorem smallest_x_for_gg_defined :
  ∃ x : ℝ, (x ≥ 30) ∧ (∀ y : ℝ, (g(g(y)) = g(g(x)) → y = x)) :=
begin
  sorry
end

end smallest_x_for_gg_defined_l244_244075


namespace line_does_not_pass_second_quadrant_l244_244308

-- Definitions of conditions
variables (k b x y : ℝ)
variable  (h₁ : k > 0) -- condition k > 0
variable  (h₂ : b < 0) -- condition b < 0


theorem line_does_not_pass_second_quadrant : 
  ¬∃ (x y : ℝ), (x < 0 ∧ y > 0) ∧ (y = k * x + b) :=
sorry

end line_does_not_pass_second_quadrant_l244_244308


namespace peg_arrangement_count_l244_244755

theorem peg_arrangement_count :
  let yellow_pegs := 6
  let red_pegs := 5
  let green_pegs := 4
  let blue_pegs := 3
  let orange_pegs := 2
  let rows := 6
  let columns := 5
  let factorial (n : ℕ) : ℕ := (list.range (n + 1)).foldl (*) 1 in
  (factorial yellow_pegs) * (factorial red_pegs) * (factorial green_pegs) *
  (factorial blue_pegs) * (factorial orange_pegs) = 86400 :=
by
  rw [factorial, list.range, list.foldl, ← nat.add_sub_of_le, ← list.range_succ, add_comm] 
  sorry

end peg_arrangement_count_l244_244755


namespace sum_of_interior_angles_l244_244661

theorem sum_of_interior_angles (n : ℕ) (h : n ≥ 3) (polygon : Π (k : ℕ), k = n → Type) 
  (non_self_intersecting : ∀ (k : ℕ), k = n → Prop) : 
  ∑ (i : ℕ) in finset.range n, 
    (λ i, angle_sum polygon (non_self_intersecting n rfl) i) = (n - 2) * real.pi :=
sorry

end sum_of_interior_angles_l244_244661


namespace problem_l244_244131

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244131


namespace parabola_properties_l244_244184

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244184


namespace balls_into_boxes_l244_244539

theorem balls_into_boxes : (3^6 = 729) :=
by
  calc
    3^6 = 729 : sorry

end balls_into_boxes_l244_244539


namespace johns_gym_costs_l244_244593

theorem johns_gym_costs : 
  let cheap_gym_monthly := 10 in
  let cheap_gym_signup := 50 * (1 - 0.1) in
  let cheap_gym_maintenance := 30 in
  let cheap_gym_first_10_sessions := 25 in
  let cheap_gym_discount := 0.2 in
  let cheap_gym_next_10_sessions := cheap_gym_first_10_sessions * (1 - cheap_gym_discount) in
  let cheap_gym_annual := cheap_gym_monthly * 12 + cheap_gym_signup + cheap_gym_maintenance in
  let cheap_gym_training := cheap_gym_first_10_sessions * 10 + cheap_gym_next_10_sessions * 10 in
  let cheap_gym_total := cheap_gym_annual + cheap_gym_training in
  let expensive_gym_monthly := 30 in
  let expensive_gym_signup := 4 * expensive_gym_monthly * (1 - 0.1) in
  let expensive_gym_maintenance := 60 in
  let expensive_gym_first_5_sessions := 45 in
  let expensive_gym_discount := 0.15 in
  let expensive_gym_next_10_sessions := expensive_gym_first_5_sessions * (1 - expensive_gym_discount) in
  let expensive_gym_annual := expensive_gym_monthly * 12 + expensive_gym_signup + expensive_gym_maintenance in
  let expensive_gym_training := expensive_gym_first_5_sessions * 5 + expensive_gym_next_10_sessions * 10 in
  let expensive_gym_total := expensive_gym_annual + expensive_gym_training in
  let johns_total_cost := cheap_gym_total + expensive_gym_total in
  johns_total_cost = 1780.50 :=
by
  sorry

end johns_gym_costs_l244_244593


namespace prime_count_l244_244522

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def from_digits (tens units : ℕ) : ℕ :=
  10 * tens + units

def is_valid_prime (tens units : ℕ) : Prop :=
  {3, 5, 7, 9}.contains tens ∧ 
  {3, 5, 7, 9}.contains units ∧ 
  tens ≠ units ∧ 
  is_prime (from_digits tens units)

theorem prime_count : 
  (finset.univ.filter (λ p, ∃ tens ∈ {3, 5, 7, 9}, ∃ units ∈ {3, 5, 7, 9}, tens ≠ units ∧ is_prime (from_digits tens units))).card = 6 :=
by
  sorry

end prime_count_l244_244522


namespace first_number_less_than_zero_l244_244835

def count_down_sequence (start step: Int) : List Int :=
  List.range (start / step).length |>.map (λ n => start - step * n)

theorem first_number_less_than_zero (start step: Int) (h_start: start = 100) (h_step: step = 11) : 
  ∃ n, (count_down_sequence start step)[n] < 0 ∧ (∀ m, m < n → (count_down_sequence start step)[m] >= 0) := 
by
  sorry

end first_number_less_than_zero_l244_244835


namespace min_value_f_l244_244689

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 1/(x^2 - 2*x + 1)

theorem min_value_f : ∃ m ∈ set.Ioo (0:ℝ) 3, ∀ x ∈ set.Ioo (0:ℝ) 3, f x ≥ f m ∧ f m = 1 :=
by
  sorry

end min_value_f_l244_244689


namespace max_chips_l244_244577

def chip_color := {red, blue}

structure Board (n : ℕ) :=
  (grid : Fin n × Fin n → Option chip_color)

def sees_five_opposite_chips (board : Board 200) (pos : Fin 200 × Fin 200) (color : chip_color) : Prop :=
  let opposite_color := if color = chip_color.red then chip_color.blue else chip_color.red
  let same_row_chips := (fun i => board.grid (pos.1, i)) |>.filter (fun opt_chip => opt_chip = opposite_color) 
  let same_col_chips := (fun i => board.grid (i, pos.2)) |>.filter (fun opt_chip => opt_chip = opposite_color)
  (same_row_chips.length + same_col_chips.length) = 5

def satisfies_condition (board : Board 200) : Prop :=
  ∀ pos, ∀ color, (board.grid pos = some color) → sees_five_opposite_chips board pos color

theorem max_chips (board : Board 200) :
  satisfies_condition board →
  (∑ i in Finset.finRange 200, ∑ j in Finset.finRange 200, if board.grid (i, j) = none then 0 else 1) ≤ 3800 :=
sorry

end max_chips_l244_244577


namespace value_of_p_circle_tangent_to_directrix_l244_244141

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244141


namespace parabola_properties_l244_244187

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244187


namespace value_of_p_circle_tangent_to_directrix_l244_244139

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244139


namespace prob_diff_tens_digit_l244_244668

theorem prob_diff_tens_digit (s : Finset ℕ) (h1 : s.card = 5)
  (h2 : ∀ n ∈ s, 10 ≤ n ∧ n ≤ 59) :
  let p := (5.choose 5 * 10^5 : ℚ) / (50.choose 5 : ℚ) in 
  p = 2500 / 52969 := 
by 
  -- the proof is omitted as per the instructions
  sorry

end prob_diff_tens_digit_l244_244668


namespace part1_part2_l244_244501

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.exp (2 * x) - a * Real.exp x - x * Real.exp x

theorem part1 :
  (∀ x : ℝ, f a x ≥ 0) → a = 1 := sorry

theorem part2 (h : a = 1) :
  ∃ x₀ : ℝ, (∀ x : ℝ, f a x ≤ f a x₀) ∧
    (Real.log 2 / (2 * Real.exp 1) + 1 / (4 * Real.exp (2 * 1)) ≤ f a x₀ ∧
    f a x₀ < 1 / 4) := sorry

end part1_part2_l244_244501


namespace symmetry_of_graphs_l244_244041

noncomputable def f (a x : ℝ) : ℝ := log a (2 + a * x)
noncomputable def g (a x : ℝ) : ℝ := log (1 / a) (a + 2 * x)

theorem symmetry_of_graphs (a b : ℝ) 
    (h_a_pos : a > 0) 
    (h_a_ne_one : a ≠ 1)
    (symm_condition : ∀ x : ℝ, f a x + g a x = 2 * b) : 
    a + b = 2 := 
begin
  sorry
end

end symmetry_of_graphs_l244_244041


namespace symmetric_coordinates_l244_244583

-- Given point P with coordinates (3, -2, 1)
def P := (3, -2, 1 : ℤ × ℤ × ℤ)

-- Define the symmetric point Q with respect to the xOz plane
def symmetric_point_xOz (P : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ :=
  (P.1, -P.2, P.3)

-- Prove that the coordinates of Q are (3, 2, 1)
theorem symmetric_coordinates :
  symmetric_point_xOz P = (3, 2, 1) :=
  by
    simp [P, symmetric_point_xOz]
    rfl

end symmetric_coordinates_l244_244583


namespace radius_of_inscribed_circle_l244_244323

def is_tangent (P C : Point) (r R : ℝ) : Prop :=
  dist P C = R - r

def divides_equally (A B C : Point) (d : ℝ) : Prop :=
  dist A B = d ∧ dist B C = d

theorem radius_of_inscribed_circle {O₁ O₂ A B C O : Point}
  (hO₁O₂ : dist O₁ O₂ = 48)
  (hradii : ∀ P, P = O₁ ∨ P = O₂ → dist P C = 32)
  (hdivides : divides_equally O₁ A B 16 ∧ divides_equally B C O₂ 16)
  (htangent : is_tangent O O₁ r ∧ is_tangent O O₂ r ∧ dist O C = r) :
  r = 7 :=
sorry

end radius_of_inscribed_circle_l244_244323


namespace bracelet_ratio_l244_244837

-- Definition of the conditions
def initial_bingley_bracelets : ℕ := 5
def kelly_bracelets_given : ℕ := 16 / 4
def total_bracelets_after_receiving := initial_bingley_bracelets + kelly_bracelets_given
def bingley_remaining_bracelets : ℕ := 6
def bingley_bracelets_given := total_bracelets_after_receiving - bingley_remaining_bracelets

-- Lean 4 Statement
theorem bracelet_ratio : bingley_bracelets_given * 3 = total_bracelets_after_receiving := by
  sorry

end bracelet_ratio_l244_244837


namespace range_of_a_l244_244498

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) ↔ (0 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l244_244498


namespace find_a_l244_244251

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^4 + a * x else sorry

theorem find_a (a : ℝ) :
  odd_function f →
  (∀ x < 0, f x = x^4 + a * x) →
  f 2 = 6 →
  a = 11 :=
by
  intros h_odd h_neg h_f2
  sorry

end find_a_l244_244251


namespace parabola_focus_line_tangent_circle_l244_244209

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244209


namespace correct_propositions_l244_244301

theorem correct_propositions :
  (("if x = 1 → x^2 - 3 * x + 2 = 0" and not ("x^2 - 3 * x + 2 = 0 → x = 1")) ∈ {2, 3, 4}) ∧ 
  (("if x^2 - 3 * x + 2 → x = 1" then "not (x ≠ 1) → not (x^2 - 3 * x + 2 = 0) ") ∈ {2, 3, 4}) ∧ 
  (("if exists x > 0, x^2 + x + 1 < 0" then "for all x ≤ 0, x^2 + x + 1 ≥ 0") ∈ {2, 3, 4}) ∧ 
  (("if not (p ∨ q)" then "p = False" and "q = False") ∈ {2, 3, 4}) 
  :=
  sorry

end correct_propositions_l244_244301


namespace greatest_five_digit_number_l244_244373

def is_divisible_by (n d : ℕ) : Prop :=
  d ∣ n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let reversed := digits.reverse
  reversed.foldl (λ acc d, acc * 10 + d) 0

def alternating_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  List.sum (digits.enum.map (λ ⟨i, d⟩, if i % 2 = 0 then d else -d))

theorem greatest_five_digit_number :
  ∃ p : ℕ, 
    10000 ≤ p ∧ p < 100000 ∧ 
    is_divisible_by p 63 ∧ 
    is_divisible_by p 11 ∧ 
    let q := reverse_digits p in 
    is_divisible_by q 63 ∧
    ∀ p' : ℕ, 
      10000 ≤ p' ∧ p' < 100000 ∧ 
      is_divisible_by p' 63 ∧ 
      is_divisible_by p' 11 ∧ 
      let q' := reverse_digits p' in 
      is_divisible_by q' 63 → 
      p' ≤ p :=
  ⟨99729, by {
    sorry
  }⟩

end greatest_five_digit_number_l244_244373


namespace length_ratio_l244_244958

variable (d : ℝ) (s : ℝ) (S : ℝ)
-- 10 points form a segment of length s
axiom h1 : s = 9 * d
-- 100 points form a segment of length S
axiom h2 : S = 99 * d

theorem length_ratio : S = 11 * s := by
  rw [h1, h2]
  sorry

end length_ratio_l244_244958


namespace washing_machine_last_load_l244_244819

theorem washing_machine_last_load (W : ℕ) (T : ℕ) (loads : ℕ) :
  W = 28 → T = 200 → T % W = loads → loads = 4 :=
by
  intros hW hT hLoads
  rw [hW, hT] at hLoads
  have h: 200 % 28 = 4 := rfl
  exact h.symm

end washing_machine_last_load_l244_244819


namespace function_is_constant_l244_244122

variable (a : ℝ)
variable (f : ℝ → ℝ)
hypothesis (h1 : f 0 = 1/2)
hypothesis (h2 : ∀ x y : ℝ, f (x + y) = f x * f (a - y) + f y * f (a - x))

theorem function_is_constant : ∀ x : ℝ, f x = 1/2 := 
sorry

end function_is_constant_l244_244122


namespace general_formula_for_a_n_sum_of_first_n_terms_of_bn_l244_244924

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℤ := 11 - 2 * n

-- Define the sequence {b_n} as the absolute value of {a_n}
def b_n (n : ℕ) : ℕ := (11 - 2 * n).natAbs

-- Define the sum of first n terms of {a_n}, S_n
def S_n (n : ℕ) : ℕ := (n * (2 * 9 + (n - 1) * -2)) / 2

-- Define the sum of first n terms of {b_n}, T_n
def T_n (n : ℕ) : ℤ :=
  if n ≤ 5 then 10 * n - n * n
  else n * n - 10 * n + 50

-- Proving the general formula for {a_n}
theorem general_formula_for_a_n (n : ℕ) : a_n n = 11 - 2 * n :=
by
  sorry

-- Proving the sum of the first n terms of {b_n}
theorem sum_of_first_n_terms_of_bn (n : ℕ) : 
  T_n n = if n ≤ 5 then 10 * n - n * n else n * n - 10 * n + 50 :=
by
  sorry

end general_formula_for_a_n_sum_of_first_n_terms_of_bn_l244_244924


namespace possible_rectangle_configurations_l244_244296

-- Define the conditions as variables
variables (m n : ℕ)
-- Define the number of segments
def segments (m n : ℕ) : ℕ := 2 * m * n + m + n

theorem possible_rectangle_configurations : 
  (segments m n = 1997) → (m = 2 ∧ n = 399) ∨ (m = 8 ∧ n = 117) ∨ (m = 23 ∧ n = 42) :=
by
  sorry

end possible_rectangle_configurations_l244_244296


namespace max_distance_between_vertices_l244_244387

theorem max_distance_between_vertices (inner_perimeter outer_perimeter : ℕ) 
  (inner_perimeter_eq : inner_perimeter = 20) 
  (outer_perimeter_eq : outer_perimeter = 28) : 
  ∃ x y, x + y = 7 ∧ x^2 + y^2 = 25 ∧ (x^2 + (x + y)^2 = 65) :=
by
  sorry

end max_distance_between_vertices_l244_244387


namespace parabola_p_and_circle_tangent_directrix_l244_244225

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244225


namespace monotonically_increasing_intervals_range_of_a_for_exactly_three_zeros_l244_244935

section ProofProblem

open Real Nat

def f (x : ℝ) : ℝ := 2 * sin x * cos x - sin x ^ 2 - 3 * cos x ^ 2 + 1

theorem monotonically_increasing_intervals :
  ∀ k : ℤ, 
    ∀ x : ℝ, 
      (∃ y : ℝ, (x = -π/8 + k * π ∧ y = 3π/8 + k * π) ∧ 
        ∀ (x1 : ℝ), x1 ∈ Ioc x y → f' x1 > 0) :=
sorry

theorem range_of_a_for_exactly_three_zeros :
  ∀ a : ℝ, 
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ a → f x = 0 → 
      (∃ z1 z2 z3 : ℝ, (f z1 = 0 ∧ f z2 = 0 ∧ f z3 = 0 ∧ z1 = π/4 ∧ z2 = π/2 ∧ z3 = 5π/4)) 
      → 5 * π / 4 ≤ a ∧ a ≤ 3 * π / 2) :=
sorry

end ProofProblem

end monotonically_increasing_intervals_range_of_a_for_exactly_three_zeros_l244_244935


namespace solve_for_a_l244_244850

open Real

-- Define the curve and the line
def C1 (a : ℝ) : ℝ → ℝ := λ x, sqrt x + a
def l (x y : ℝ) : Prop := x - 2 * y = 0

-- Define the distance from a point to a line
def distance_point_line (x y : ℝ) : ℝ := abs (x - 2 * y) / sqrt 5

-- Define the condition that the minimum distance from C1 to l equals sqrt 5
def min_distance_condition (a : ℝ) : Prop :=
  ∃ (x : ℝ), distance_point_line x (C1 a x) = sqrt 5

-- The theorem to prove
theorem solve_for_a (a : ℝ) (h : min_distance_condition a) : a = -3 :=
sorry

end solve_for_a_l244_244850


namespace bolzano_weierstrass_theorem_l244_244653

theorem bolzano_weierstrass_theorem (a : ℕ → ℝ) (h_bounded: ∃ (M > 0), ∀ n, |a n| ≤ M ):
  ∃ (ℓ : ℝ), ∃ (ϵ > 0), ∃ (N : ℕ), ∀ n > N, ∃ (m > n), |a m - ℓ| < ϵ :=
sorry

end bolzano_weierstrass_theorem_l244_244653


namespace width_of_alley_l244_244571
-- Import the full Mathlib library for necessary mathematical functions and definitions.

-- Define the conditions from the problem.
variable (ladder_length : Real := 10)
variable (angle_Q : Real := 60)
variable (angle_R : Real := 70)
variable (cos60 : Real := Real.cos (60 * Real.pi / 180))
variable (cos70 : Real := Real.cos (70 * Real.pi / 180))

-- Define the expected width of the alley.
noncomputable def expected_width : Real := 8.42

-- Calculate horizontal distances based on the angles
noncomputable def distance_Q : Real := ladder_length * cos60
noncomputable def distance_R : Real := ladder_length * cos70

-- Define the statement to prove the width of the alley is 8.42 meters.
theorem width_of_alley (w : Real) : 
  w = distance_Q + distance_R :=
sorry

-- Assert that the calculated width matches the expected width.
#eval (distance_Q + distance_R = expected_width)

end width_of_alley_l244_244571


namespace p_eq_two_circle_tangent_proof_l244_244214

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244214


namespace tenth_term_arithmetic_sequence_l244_244337

def a : ℚ := 2 / 3
def d : ℚ := 2 / 3

theorem tenth_term_arithmetic_sequence : 
  let a := 2 / 3
  let d := 2 / 3
  let n := 10
  a + (n - 1) * d = 20 / 3 := by
  sorry

end tenth_term_arithmetic_sequence_l244_244337


namespace at_least_6_heads_probability_l244_244798

open_locale big_operators

theorem at_least_6_heads_probability : 
  let outcomes := 2 ^ 9 in
  let total_ways := (Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9) in
  total_ways / outcomes = 130 / 512 :=
by
  sorry

end at_least_6_heads_probability_l244_244798


namespace term_of_arithmetic_sequence_l244_244732

variable (a₁ : ℕ) (d : ℕ) (n : ℕ)

theorem term_of_arithmetic_sequence (h₁: a₁ = 2) (h₂: d = 5) (h₃: n = 50) :
    a₁ + (n - 1) * d = 247 := by
  sorry

end term_of_arithmetic_sequence_l244_244732


namespace sequence_a_n_derived_conditions_derived_sequence_is_even_l244_244901

-- Statement of the first problem
theorem sequence_a_n_derived_conditions (a : ℕ → ℝ) (b : ℕ → ℝ) (n : ℕ)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : b 1 = 5 ∧ b 2 = -2 ∧ b 3 = 7 ∧ b 4 = 2):
  a 1 = 2 ∧ a 2 = 1 ∧ a 3 = 4 ∧ a 4 = 5 :=
sorry

-- Statement of the second problem
theorem derived_sequence_is_even (a : ℕ → ℝ) (b : ℕ → ℝ) (c : ℕ → ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h1 : b 1 = a n)
  (h2 : ∀ k, 2 ≤ k ∧ k ≤ n → b k = a (k - 1) + a k - b (k - 1))
  (h3 : c 1 = b n)
  (h4 : ∀ k, 2 ≤ k ∧ k ≤ n → c k = b (k - 1) + b k - c (k - 1)):
  ∀ i, 1 ≤ i ∧ i ≤ n → c i = a i :=
sorry

end sequence_a_n_derived_conditions_derived_sequence_is_even_l244_244901


namespace students_enjoy_soccer_fraction_l244_244832

theorem students_enjoy_soccer_fraction :
  (∀ (total_students : ℕ),
    let enjoy_soccer := 0.7 * total_students
    let dont_enjoy_soccer := 0.3 * total_students
    let express_enjoyment := 0.75 * enjoy_soccer
    let not_express_enjoyment := 0.25 * enjoy_soccer
    let express_disinterest := 0.85 * dont_enjoy_soccer
    let incorrectly_say_enjoy := 0.15 * dont_enjoy_soccer
    let say_dont_enjoy := not_express_enjoyment + express_disinterest
    not_express_enjoyment / say_dont_enjoy = 13 / 32)
:= sorry

end students_enjoy_soccer_fraction_l244_244832


namespace vaccine_efficacy_prob_l244_244805

noncomputable theory

open ProbabilityTheory

/-- Let n be the number of vaccinated individuals and m be the number of unvaccinated individuals.
    P_chi2 is the probability Chi-squared test statistic.
    Null hypothesis H0: "the vaccine does not prevent Type A H1N1 influenza".
    If P_chi2 ≥ 6.635 is approximately 0.01, then the confidence level that the vaccine is effective is 99%. -/
theorem vaccine_efficacy_prob (n m : ℕ) (P_chi2 : ℝ) (H0 : Prop)
  (h_n : n = 1000) (h_m : m = 1000)
  (h_H0 : H0 = False) -- Null hypothesis formulated as a statement
  (h_P : P_chi2 = 0.01) :
  (confidence_level : ℝ) := 0.99 :=
by sorry

end vaccine_efficacy_prob_l244_244805


namespace parallel_edges_count_l244_244058

theorem parallel_edges_count (length width height : ℕ) (h_length : length = 8) (h_width : width = 4) (h_height : height = 2) : 
  num_parallel_edges length width height = 6 :=
sorry

noncomputable def num_parallel_edges (length width height : ℕ) : ℕ :=
-- implementation of counting pairs of parallel edges, assuming correctness
if length = 8 ∧ width = 4 ∧ height = 2 then 6 else 0

end parallel_edges_count_l244_244058


namespace number_of_possible_plans_most_cost_effective_plan_l244_244728

-- Defining the conditions of the problem
def price_A := 12 -- Price of model A in million yuan
def price_B := 10 -- Price of model B in million yuan
def capacity_A := 240 -- Treatment capacity of model A in tons/month
def capacity_B := 200 -- Treatment capacity of model B in tons/month
def total_budget := 105 -- Total budget in million yuan
def min_treatment_volume := 2040 -- Minimum required treatment volume in tons/month
def total_units := 10 -- Total number of units to be purchased

def valid_purchase_plan (x y : ℕ) :=
  x + y = total_units ∧
  price_A * x + price_B * y ≤ total_budget ∧
  capacity_A * x + capacity_B * y ≥ min_treatment_volume

-- Stating the theorem for how many possible purchase plans exist
theorem number_of_possible_plans : 
  ∃ k : ℕ, k = 3 ∧
    (∀ (x y : ℕ), 
      valid_purchase_plan x y →
      x ∈ {0, 1, 2} ∧ y = total_units - x) :=
sorry

-- Stating the theorem for the most cost-effective plan
theorem most_cost_effective_plan :
  ∃ (x y : ℕ),
    valid_purchase_plan x y ∧
    price_A * x + price_B * y = 102 ∧
    x = 1 ∧ y = 9 :=
sorry

end number_of_possible_plans_most_cost_effective_plan_l244_244728


namespace prove_no_intersection_l244_244976

structure Parallelepiped (α : Type*) :=
(A B C D A1 B1 C1 D1 : α)

structure Slope (α : Type*) :=
(direction : α)

noncomputable def project_and_intersect {α : Type*} 
  (p : Parallelepiped α) 
  (AP C1D : α) 
  (l : Slope α) 
  (skew : AP ≠ C1D) 
  (parallel_to_projection : ∀ dp : α, l.direction = dp) : Prop :=
  ¬ (∃ x : α, (x ∈ AP) ∧ (x ∈ C1D))

theorem prove_no_intersection 
  {α : Type*} 
  (p : Parallelepiped α) 
  (AP C1D : α) 
  (l : Slope α) 
  (skew : AP ≠ C1D) 
  (parallel_to_projection : ∀ dp : α, l.direction = dp) : 
  project_and_intersect p AP C1D l skew parallel_to_projection :=
begin
  sorry,
end

end prove_no_intersection_l244_244976


namespace sufficient_condition_not_necessary_condition_l244_244465

theorem sufficient_condition (a : ℝ) : (∀ x : ℝ, a < x^2 + 1) → (∃ x_0 : ℝ, a < 3 - x_0^2) :=
by
  intros h1
  use 0
  have h := h1 0
  exact h

theorem not_necessary_condition (a : ℝ) : (∃ x_0 : ℝ, a < 3 - x_0^2) → (∀ x : ℝ, a < x^2 + 1) → a < 1 :=
sorry

end sufficient_condition_not_necessary_condition_l244_244465


namespace largest_possible_three_day_success_ratio_l244_244398

noncomputable def beta_max_success_ratio : ℝ :=
  let (a : ℕ) := 33
  let (b : ℕ) := 50
  let (c : ℕ) := 225
  let (d : ℕ) := 300
  let (e : ℕ) := 100
  let (f : ℕ) := 200
  a / b + c / d + e / f

theorem largest_possible_three_day_success_ratio :
  beta_max_success_ratio = (358 / 600 : ℝ) :=
by
  sorry

end largest_possible_three_day_success_ratio_l244_244398


namespace largest_k_dividing_polynomial_l244_244362

def P (k : ℤ) : ℤ :=
  k^2020 + 2*k^2019 + 3*k^2018 + ⬝⬝⬝ + 2020*k + 2021

theorem largest_k_dividing_polynomial :
  ∀ k : ℤ, (k + 1) ∣ P k → k ≤ 1010 :=
  sorry

end largest_k_dividing_polynomial_l244_244362


namespace magic_king_total_episodes_l244_244705

theorem magic_king_total_episodes :
  (∑ i in finset.range 5, 20) + (∑ j in finset.range 5, 25) = 225 :=
by sorry

end magic_king_total_episodes_l244_244705


namespace possible_values_of_p_l244_244542

theorem possible_values_of_p (p : ℕ) (a b : ℕ) (h_fact : (x : ℤ) → x^2 - 5 * x + p = (x - a) * (x - b))
  (h1 : a + b = 5) (h2 : 1 ≤ a ∧ a ≤ 4) (h3 : 1 ≤ b ∧ b ≤ 4) : 
  p = 4 ∨ p = 6 :=
sorry

end possible_values_of_p_l244_244542


namespace range_of_ratios_l244_244585

noncomputable def triangle_relation (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) 
: interval ℝ 
:= ⟨2, sqrt 5⟩

theorem range_of_ratios (a b c : ℝ) 
  (hABC : a^2 + b^2 = c^2) -- assuming some relationship here for simplicity
  (h_height : (a^2 * c = b^2 + c^2)) 
  : 2 ≤ (b/c) + (c/b) ∧ (b/c) + (c/b) ≤ sqrt 5 :=
begin
  sorry
end

end range_of_ratios_l244_244585


namespace exists_x0_lt_g_l244_244893

def f (x : ℝ) : ℝ := x^3
def g (x a : ℝ) : ℝ := -x^2 + x - a

theorem exists_x0_lt_g (a : ℝ) (h : a > 5 / 27) : 
  ∃ x0 : ℝ, x0 ∈ set.Icc (-1 : ℝ) (1 : ℝ) ∧ f x0 < g x0 a :=
begin
  sorry
end

end exists_x0_lt_g_l244_244893


namespace problem1_problem2_l244_244054

-- Problem (I)
theorem problem1 (x : ℝ) (t : ℝ) (H1 : x ∈ Icc (-π/2) (π/2))
  (H2 : (λ x : ℝ, (2 * sin x, -1) - ((sin x - (sqrt 3) * cos x), -2)) = (λ x, (sin x + (sqrt 3) * cos x, 1)))
  (H3 : f x = ((sin x + (sqrt 3) * cos x, 1) · (2 * sin x, -1)) + t) 
  : t = -1 :=
sorry

-- Problem (II)
theorem problem2 (a b c S : ℝ) (H1 : a = 4) (H2 : S = sqrt 3) (H3 : (λ A : ℝ, 2 * sin A, -1) - ((sin A - (sqrt 3) * cos A), -2) = (λ A, (sin A + (sqrt 3) * cos A, 1))) 
  (H4 : f A = 2) (H5 : t = 0) 
  : b + c = 2 * sqrt 7 :=
sorry

end problem1_problem2_l244_244054


namespace parabola_focus_line_l244_244159

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244159


namespace proof_min_expression_value_l244_244485

noncomputable def min_expression_value (a b c : ℝ) :=
  (a > 0) ∧ (b > 0) ∧ (c > 2) ∧ (a + b = 2) →
  (∃ x : ℝ, x = sqrt 10 + sqrt 5 ∧
  x ≤ (ac / b + c / (ab) - c / 2 + sqrt 5 / (c - 2)))

theorem proof_min_expression_value
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : c > 2)
  (h₄ : a + b = 2) :
  ∃ x : ℝ, x = sqrt 10 + sqrt 5 ∧
  x ≤ (a * c / b + c / (a * b) - c / 2 + sqrt 5 / (c - 2)) := 
sorry

end proof_min_expression_value_l244_244485


namespace unit_circle_solution_l244_244986

noncomputable def unit_circle_point_x (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) : ℝ :=
  1 / 26

theorem unit_circle_solution (α : ℝ) (hα : α ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2)) 
  (hcos : Real.cos (α + Real.pi / 3) = -11 / 13) :
  unit_circle_point_x α hα hcos = 1 / 26 :=
by
  sorry

end unit_circle_solution_l244_244986


namespace hyperbola_product_of_distances_l244_244004

theorem hyperbola_product_of_distances (F1 F2 P : ℝ × ℝ) (a b c : ℝ) (θ : ℝ) 
  (hf1 : F1 = (-c, 0)) (hf2 : F2 = (c, 0)) (hc : c = real.sqrt 2) 
  (ha : a = 1) (hb : b = 1) 
  (hP_on_hyperbola : P.1 ^ 2 - P.2 ^ 2 = 1) 
  (h_angle : θ = 60) 
  (cosine_theta : real.cos (θ * real.pi / 180) = 1 / 2) :
  let PF1 := real.sqrt ((P.1 + c)^2 + P.2^2)
      PF2 := real.sqrt ((c - P.1)^2 + P.2^2) in
  PF1 * PF2 = 4 := 
by
  sorry

end hyperbola_product_of_distances_l244_244004


namespace vanessa_score_l244_244567

theorem vanessa_score (total_points team_score other_players_avg_score: ℝ) : 
  total_points = 72 ∧ team_score = 7 ∧ other_players_avg_score = 4.5 → 
  ∃ vanessa_points: ℝ, vanessa_points = 40.5 :=
by
  sorry

end vanessa_score_l244_244567


namespace sum_of_PQ_and_PR_in_equilateral_triangle_l244_244089

theorem sum_of_PQ_and_PR_in_equilateral_triangle :
  ∀ (A B C P Q R : Point) (AP AQ PR PQ BC AB AC : ℝ),
  AP + AQ = PR + PQ ∧
  PQ || BC ∧ PR || BC ∧
  is_equilateral ⟨A, B, C⟩ (2) ∧ 
  (perimeter (rectangle A P Q R)) = (perimeter (rectangle P B R C)) →
  AP + PR = 4 / 3 :=
begin
  intros A B C P Q R AP AQ PR PQ BC AB AC,
  assume h_cond,
  sorry
end

end sum_of_PQ_and_PR_in_equilateral_triangle_l244_244089


namespace car_race_probability_l244_244979

noncomputable def P (n : ℕ) : ℚ := 1 / n

theorem car_race_probability :
  let P_sunny := P 8 + P 12 + P 6 + P 10 + P 20,
      P_rainy := P 8 + P 12 + P 6 + P 10 + P 15 in
  P_sunny = 21 / 40 ∧ P_rainy = 13 / 24 :=
by
  -- Prove P_sunny = 21 / 40
  have : P 8 = 15 / 120 := by sorry
  have : P 12 = 10 / 120 := by sorry
  have : P 6 = 20 / 120 := by sorry
  have : P 10 = 12 / 120 := by sorry
  have : P 20 = 6 / 120 := by sorry
  calc
    P_sunny = (15 + 10 + 20 + 12 + 6) / 120 := by rw [P_sunny, this, this, this, this, this]
    ... = 63 / 120 := by norm_num
    ... = 21 / 40 := by norm_num,
  -- Prove P_rainy = 13 / 24
  have : P 15 = 8 / 120 := by sorry
  calc
    P_rainy = (15 + 10 + 20 + 12 + 8) / 120 := by rw [P_rainy, this, this, this, this, this]
    ... = 65 / 120 := by norm_num
    ... = 13 / 24 := by norm_num

end car_race_probability_l244_244979


namespace general_formula_arith_seq_sum_first_n_terms_seq_b_l244_244923

def arith_seq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Define the given conditions
def a3 := 5
def S9 := 9

-- Definition of the arithmetic sequence and its sum
noncomputable def a_n (n : ℕ) : ℕ := arith_seq 11 (-2) n -- this matches 11 - 2n

-- Proof of the general formula for an arithmetic sequence
theorem general_formula_arith_seq :
  ∀ (a1 d : ℕ),
  (arith_seq 11 (-2) 3 = 5) ∧
  (9 * a1 + 36 * d = 9) →
  ∀ n, a_n n = 11 - 2 * n := 
sorry

-- Definition of the sequence b_n and its sum T_n
def b_n (n : ℕ) : ℕ := abs (a_n n)

def T_n (n : ℕ) : ℕ :=
if n ≤ 5 then
  10 * n - n ^ 2
else
  n ^ 2 - 10 * n + 50

-- Proof of the sum of the first n terms of the sequence b_n
theorem sum_first_n_terms_seq_b :
  ∀ (n : ℕ),
  (T_n n) =
  (if n ≤ 5 then
    10 * n - n ^ 2
  else
    n ^ 2 - 10 * n + 50) := 
sorry

end general_formula_arith_seq_sum_first_n_terms_seq_b_l244_244923


namespace books_on_bottom_shelf_bottom_shelf_books_l244_244396

theorem books_on_bottom_shelf (x : ℕ) (hx1 : x / 2 of Early Universe + d songs)
  (r1 books to be Needed : ℕ) (hx2 : r1_1 = x / despite 
  ( hx                           
   zero                      and the                             
                                                                   
                       
                                      when tv_chunk_percent_1 == of Annulment 7
                                                                            
3₂)                                                                                
  (hx : r_remaining_ vor_ loyalty attended ( follow third_powers*==3Results) - at half_x 
                                 the number_gender 3→he Randomly sh8往
" ( to determine r8 receivedsh: 7 indivisible this)____:
    conclusion hence3 divide 
as
above the shelf : since before4 also; /in we ; Div ided

                                                                                
 7, there207af i4
            
8 6 = Int fan_b1cubed_3)

hx)              = ul Brackets-7_one ( prover needed)= books placed_done computationally ) z
book bottom shelf_eq
of_we) side 21 kh_note in thus4 large_w
      
2 in .bs33                                       unithese_go=> :
    nsbooks on #leftr>>> bottom shelf=last Indict f   

2of = statement3)

Lean statement thus follows steps:


theorem bottom_shelf_books {x : ℕ} (hx1 : x / 2 = 3 = yx_of by remaining hence 74Div)} imposes:
                                                  Two   
also from 10 3_Type_t removes_not accounts                 • :
    :   x_type3_∴ 
   
a: r =>3                                                                                                    is problém_num Direct=3en ysolution_right\n x 类型_type 20 ∴ =2 . thus_half 3f as_hence occurrence enumerated, total)}
                                                                                                                                       
    allocated_2_layers_ coverage4 ordering                      |

end books_on_bottom_shelf_bottom_shelf_books_l244_244396


namespace find_sets_and_range_of_a_l244_244027

noncomputable def domain_g (a : ℝ) : Set ℝ := { x | a < x ∧ x < 2 * a }

noncomputable def domain_f : Set ℝ := { x | x ≤ 2 }

noncomputable def range_f : Set ℝ := { y | 0 ≤ y ∧ y < 3 }

theorem find_sets_and_range_of_a (a : ℝ) (A B C : Set ℝ)
  (h1 : 0 < a) (h2 : a ≠ 1) (h3 : A = domain_g a) (h4 : B = domain_f) (h5 : C = range_f) :
  A = (a, 2 * a) ∧ B = [2, +∞) ∧ C = [0, 3) ∧ (A ∪ C = C → 0 < a ∧ a ≤ 3 / 2 ∧ a ≠ 1) :=
by
  sorry

end find_sets_and_range_of_a_l244_244027


namespace curve_properties_l244_244504

/-- Given the parametric equations for the curve C:
    C: x = 1 + cos(θ), y = sqrt(3) + sin(θ), θ is the parameter.

    Prove:
    1. The general equation of the curve C is (x - 1)^2 + (y - sqrt(3))^2 = 1.
    2. The minimum distance from any point on the curve to the origin is 1.
-/
theorem curve_properties :
  (∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → (∃ x y : ℝ, x = 1 + Real.cos θ ∧ y = sqrt 3 + Real.sin θ)) →
  (∀ (x y : ℝ), (x - 1)^2 + (y - sqrt 3)^2 = 1 → ∃ d : ℝ, d = 1 ∧ ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → Real.sqrt ((1 + Real.cos θ)^2 + (sqrt 3 + Real.sin θ)^2) ≥ d) :=
by
  intro h₁
  have h_gen_eqn : ∀ θ : ℝ, (1 + Real.cos θ)^2 + (sqrt 3 + Real.sin θ)^2 = 5 + 4 * Real.cos (θ - Real.pi / 3) := sorry
  have h_min_dist : ∀ θ : ℝ, Real.sqrt ((1 + Real.cos θ)^2 + (sqrt 3 + Real.sin θ)^2) = Real.sqrt (5 + 4 * Real.cos (θ - Real.pi / 3)) := sorry
  existsi (1 : ℝ)
  have h_min : ∀ θ : ℝ, 0 ≤ θ ∧ θ < 2 * Real.pi → Real.sqrt (5 + 4 * Real.cos (θ - Real.pi / 3)) ≥ 1 := sorry
  tauto

end curve_properties_l244_244504


namespace sum_of_tesseract_elements_l244_244590

noncomputable def tesseract_edges : ℕ := 32
noncomputable def tesseract_vertices : ℕ := 16
noncomputable def tesseract_faces : ℕ := 24

theorem sum_of_tesseract_elements : tesseract_edges + tesseract_vertices + tesseract_faces = 72 := by
  -- proof here
  sorry

end sum_of_tesseract_elements_l244_244590


namespace parabola_conditions_l244_244241

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244241


namespace find_a_l244_244865

theorem find_a (a : ℝ) (h : ∫ x in -a..a, (2 * x - 1) = -8) : a = 4 :=
sorry

end find_a_l244_244865


namespace polyhedron_has_triangular_face_l244_244963

-- Let's define the structure of a polyhedron, its vertices, edges, and faces.
structure Polyhedron :=
(vertices : ℕ)
(edges : ℕ)
(faces : ℕ)

-- Let's assume a function that indicates if a polyhedron is convex.
def is_convex (P : Polyhedron) : Prop := sorry  -- Convexity needs a rigorous formal definition.

-- Define a face of a polyhedron as an n-sided polygon.
structure Face :=
(sides : ℕ)

-- Predicate to check if a face is triangular.
def is_triangle (F : Face) : Prop := F.sides = 3

-- Predicate to check if each vertex has at least four edges meeting at it.
def each_vertex_has_at_least_four_edges (P : Polyhedron) : Prop := 
  sorry  -- This would need a more intricate definition involving the degrees of vertices.

-- We state the theorem using the defined concepts.
theorem polyhedron_has_triangular_face 
(P : Polyhedron) 
(h1 : is_convex P) 
(h2 : each_vertex_has_at_least_four_edges P) :
∃ (F : Face), is_triangle F :=
sorry

end polyhedron_has_triangular_face_l244_244963


namespace radians_to_degrees_l244_244442

theorem radians_to_degrees (x : ℝ) (h1 : x = π / 6 ∨ x = π / 8 ∨ x = 3 * π / 4) :
  (x = π / 6 → (180 * x / π) = 30) ∧
  (x = π / 8 → (180 * x / π) = 22.5) ∧
  (x = 3 * π / 4 → (180 * x / π) = 135) :=
by
  split
  . intro h
    rw [h, mul_div_cancel_left (180 : ℝ) (ne_of_gt π_gt_0)]
    norm_num
  split
  . intro h
    rw [h, mul_div_cancel_left (180 : ℝ) (ne_of_gt π_gt_0)]
    norm_num
  . intro h
    rw [h, mul_div_cancel_left (180 : ℝ) (ne_of_gt π_gt_0)]
    norm_num

end radians_to_degrees_l244_244442


namespace max_band_members_l244_244299

theorem max_band_members (x n N : ℕ) (h1 : N = x^2 + 5) (h2 : N = n * (n + 7)) : N ≤ 294 :=
sorry

example : ∃ N x n, N = 294 ∧ N = x^2 + 5 ∧ N = n * (n + 7) :=
begin
  use [294, 17, 14],
  split,
  refl,
  split;
  norm_num,
end

end max_band_members_l244_244299


namespace sum_of_intercepts_of_parabola_l244_244693

theorem sum_of_intercepts_of_parabola : 
  let parabola (y : ℝ) := 3 * y^2 - 9 * y + 5 in
  let a := parabola 0 in
  let b := (9 + Real.sqrt 21) / 6 in
  let c := (9 - Real.sqrt 21) / 6 in
  a + b + c = 8 := 
by
  let parabola : ℝ → ℝ := fun y => 3 * y^2 - 9 * y + 5
  have a : ℝ := parabola 0
  have b : ℝ := (9 + Real.sqrt 21) / 6
  have c : ℝ := (9 - Real.sqrt 21) / 6
  have sum : ℝ := a + b + c
  sorry

end sum_of_intercepts_of_parabola_l244_244693


namespace two_digit_primes_count_l244_244520

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits := {3, 5, 7, 9}

def is_valid_two_digit_prime (n : ℕ) : Prop :=
  is_two_digit_number n ∧ is_prime n ∧ 
  ∃ t u : ℕ, t ∈ digits ∧ u ∈ digits ∧ t ≠ u ∧ n = t * 10 + u

theorem two_digit_primes_count : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, is_valid_two_digit_prime k → k < 100 := 
sorry

end two_digit_primes_count_l244_244520


namespace floor_sqrt_sum_l244_244609

theorem floor_sqrt_sum : 
  let floor (x : ℝ) := ⌊x⌋ in
  let S := ∑ i in Finset.range 100 \Finset.range 1, floor (real.sqrt (i : ℝ)) in
  floor (real.sqrt S) = 18 :=
by
  let floor (x : ℝ) := ⌊x⌋
  let S := ∑ i in Finset.range 100 \Finset.range 1, floor (real.sqrt (i : ℝ))
  sorry

end floor_sqrt_sum_l244_244609


namespace symmetry_of_g_about_2_5_l244_244428

def g (x : ℝ) : ℝ := |⌊x + 2⌋| - |⌊3 - x⌋|

theorem symmetry_of_g_about_2_5 : ∀ x : ℝ, g (2.5 - x) = g x :=
begin
  sorry
end

end symmetry_of_g_about_2_5_l244_244428


namespace integer_part_mod_8_l244_244046

theorem integer_part_mod_8 (n : ℕ) (h : n ≥ 2009) :
  ∃ x : ℝ, x = (3 + Real.sqrt 8)^(2 * n) ∧ Int.floor (x) % 8 = 1 := 
sorry

end integer_part_mod_8_l244_244046


namespace definite_integral_eval_l244_244439

theorem definite_integral_eval :
  ∫ x in (1:ℝ)..(3:ℝ), (2 * x - 1 / x ^ 2) = 22 / 3 :=
by
  sorry

end definite_integral_eval_l244_244439


namespace remainder_of_series_sum_l244_244736

theorem remainder_of_series_sum :
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 51 → 3 + 6 * (n-1)) →
  (∑ i in Finset.range 51, 3 + 6 * i) % 6 = 3 :=
by
  sorry

end remainder_of_series_sum_l244_244736


namespace simplify_expression_l244_244665

theorem simplify_expression :
  (2 + 3) * (2^2 + 3^2) * (2^4 + 3^4) * (2^8 + 3^8) *
  (2^16 + 3^16) * (2^32 + 3^32) * (2^64 + 3^64) = 3^128 - 2^128 := 
by sorry

end simplify_expression_l244_244665


namespace triangle_side_value_l244_244104

noncomputable theory
open Real

variables {a b c : ℝ}
variables {A B C : ℝ}

theorem triangle_side_value (h1 : a + b + 10 * c = 2 * (sin A + sin B + 10 * sin C))
                           (h2 : A = π / 3) :
  a = sqrt 3 :=
by {
  sorry
}

end triangle_side_value_l244_244104


namespace expected_value_transformation_l244_244928

variable (X : MassFunction ℚ)

-- Conditions
def valid_distribution (X : MassFunction ℚ) : Prop :=
  X 0 = 0.3 ∧ X 2 = 0.2 ∧ X 4 = 0.5

-- Question
theorem expected_value_transformation :
  valid_distribution X →
  E (5 • X + 4) = 16 :=
by
  sorry

end expected_value_transformation_l244_244928


namespace slope_of_line_l244_244857

-- Define the equation condition
def equation (x y : ℝ) : Prop := 4 / x + 5 / y = 0

-- Define the slope calculation from the equation condition
theorem slope_of_line (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : equation x y) :
  (let m := - 5 / 4 in 
  ∀ x1 x2 y1 y2 : ℝ, equation x1 y1 → equation x2 y2 → 
  (y2 - y1) / (x2 - x1) = m) :=
by
  sorry

end slope_of_line_l244_244857


namespace intersect_sphere_with_triangle_area_l244_244010

theorem intersect_sphere_with_triangle_area
  (A B C P : Type)
  (dist_AB : dist A B = 1) (dist_BC : dist B C = 1) (dist_CA : dist C A = 1)
  (height_P_ABC : height P (triangle A B C) = sqrt 2)
  (r : ℝ) (r_eq : r = sqrt 2 / 6) :
  let sphere_radius := 2 * r in
  let incenter := incenter (triangle A B C) in
  intersect_area sphere_radius incenter (triangle A B C) = 1 / 4 + π / 24 :=
sorry

end intersect_sphere_with_triangle_area_l244_244010


namespace union_A_B_eq_l244_244481

def A := { x : ℝ | real.log (x - 1) < 0 }
def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = 2^x - 1 }

theorem union_A_B_eq : A ∪ B = {y : ℝ | 1 < y ∧ y < 3} :=
by sorry

end union_A_B_eq_l244_244481


namespace magic_king_total_episodes_l244_244707

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l244_244707


namespace p_eq_two_circle_tangent_proof_l244_244222

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244222


namespace day_of_week_50th_day_of_year_N_minus_1_l244_244108

variable (N : ℕ)

theorem day_of_week_50th_day_of_year_N_minus_1 :
  (∀ x : ℕ, x % 7 = 0 → (x + 250) % 7 = 3) → -- 250th day of year N is a Wednesday
  (∀ y : ℕ, y % 7 = 0 → (y + 150 + 365 + 1) % 7 = 3) → -- 150th day of year N+1 is a Wednesday
  (50 + 365 + 365) % 7 = 6 := -- 50th day of year N-1 is a Saturday
begin
  intros h1 h2,
  sorry
end

end day_of_week_50th_day_of_year_N_minus_1_l244_244108


namespace balls_in_boxes_l244_244537

theorem balls_in_boxes (b : ℕ) (k : ℕ) : (b = 6) → (k = 3) → (k^6 = 729) :=
begin
  intros hb hk,
  rw [hb, hk],
  norm_num,
end

end balls_in_boxes_l244_244537


namespace parabola_focus_line_l244_244164

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244164


namespace area_ABQCDP_l244_244606

/-- Let ABCD be a trapezoid with AB parallel to CD, AB = 15, BC = 8, CD = 25, and DA = 10. 
The bisectors of ∠A and ∠D meet at point P, and the bisectors of ∠B and ∠C meet at point Q. 
Prove that the area of hexagon ABQCDP is 22√39. --/

noncomputable def area_of_hexagon (A B C D P Q : Point) : ℝ := 
  if h : trapezoid A B C D ∧
            parallel (line_through A B) (line_through C D) ∧
            distance A B = 15 ∧
            distance B C = 8 ∧
            distance C D = 25 ∧
            distance D A = 10 ∧
            angle_bisectors_meet_at A D P ∧
            angle_bisectors_meet_at B C Q
  then 22 * real.sqrt 39
  else 0

theorem area_ABQCDP : ∀ (A B C D P Q : Point),
  trapezoid A B C D →
  parallel (line_through A B) (line_through C D) →
  distance A B = 15 →
  distance B C = 8 →
  distance C D = 25 →
  distance D A = 10 →
  angle_bisectors_meet_at A D P →
  angle_bisectors_meet_at B C Q →
  area_of_hexagon A B C D P Q = 22 * real.sqrt 39 :=
by { intros A B C D P Q h1 h2 h3 h4 h5 h6 h7 h8, sorry }

end area_ABQCDP_l244_244606


namespace two_digit_numbers_condition_l244_244760

theorem two_digit_numbers_condition (a b : ℕ) (h1 : a ≠ 0) (h2 : 1 ≤ a ∧ a ≤ 9) (h3 : 0 ≤ b ∧ b ≤ 9) :
  (a + 1) * (b + 1) = 10 * a + b + 1 ↔ b = 9 := 
sorry

end two_digit_numbers_condition_l244_244760


namespace true_proposition_is_D_l244_244939

variable (x_0 : ℝ)

def p : Prop := x_0^2 - x_0 + 1 < 0

def q : Prop := ∀ a b : ℝ, a^2 < b^2 → a < b

theorem true_proposition_is_D : ¬p ∧ ¬q :=
by 
  sorry

end true_proposition_is_D_l244_244939


namespace sum_of_integer_solutions_l244_244287

theorem sum_of_integer_solutions :
  ∑ k in Finset.filter (λ x, -1 ≤ x ∧ x < 2) (Finset.Icc (-1 : ℤ) 2), (k : ℤ) = 0 :=
by
  sorry

end sum_of_integer_solutions_l244_244287


namespace max_sum_expr_l244_244890

variable (a : ℕ → ℝ)

-- Defining the notation for the sum
noncomputable def sum_expr (a : ℕ → ℝ) : ℝ := 
  ∑ k in Finset.range 2020, (a k - a (k + 1) * a (k + 2))

-- Conditions
axiom a_nonneg : ∀ k, 0 ≤ a k
axiom a_le_one : ∀ k, a k ≤ 1
axiom a_2021_periodic : a 2021 = a 1
axiom a_2022_periodic : a 2022 = a 2

-- Theorem statement
theorem max_sum_expr : sum_expr a ≤ 1010 :=
begin
  sorry
end

end max_sum_expr_l244_244890


namespace probability_correct_l244_244564

noncomputable def probability_one_white_one_black
    (total_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (draw_balls : ℕ) :=
if (total_balls = 4) ∧ (white_balls = 2) ∧ (black_balls = 2) ∧ (draw_balls = 2) then
  (2 * 2) / (Nat.choose total_balls draw_balls : ℚ)
else
  0

theorem probability_correct:
  probability_one_white_one_black 4 2 2 2 = 2 / 3 :=
by
  sorry

end probability_correct_l244_244564


namespace num_multisets_l244_244896

open Polynomial

def polynomial1 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ) (x : ℤ) : Prop :=
  a₈ * x ^ 8 + a₇ * x ^ 7 + a₆ * x ^ 6 + a₅ * x ^ 5 + a₄ * x ^ 4 + a₃ * x ^ 3 + a₂ * x ^ 2 + a₁ * x + a₀ = 0

def polynomial2 (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ : ℤ) (x : ℤ) : Prop :=
  a₀ * x ^ 8 + a₁ * x ^ 7 + a₂ * x ^ 6 + a₃ * x ^ 5 + a₄ * x ^ 4 + a₅ * x ^ 3 + a₆ * x ^ 2 + a₇ * x + a₈ = 0

theorem num_multisets (a₀ a₈ : ℤ) (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℤ) (h₀ : a₀ ≠ 0) (h₈ : a₈ ≠ 0)
  (roots : Multiset ℤ) (hroots : ∀ r ∈ roots, polynomial1 a₀ a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ r)
  (hroots' : ∀ r ∈ roots, polynomial2 a₀ a₈ a₇ a₆ a₅ a₄ a₃ a₂ a₁ r) :
  roots.card = 8 ∧ roots.toFinset = {1, -1} → multiset.card roots = 9 :=
sorry

end num_multisets_l244_244896


namespace elise_spent_on_comic_book_l244_244862

-- Define the initial amount of money Elise had
def initial_amount : ℤ := 8

-- Define the amount saved from allowance
def saved_amount : ℤ := 13

-- Define the amount spent on puzzle
def spent_on_puzzle : ℤ := 18

-- Define the amount left after all expenditures
def amount_left : ℤ := 1

-- Define the total amount of money Elise had after saving
def total_amount : ℤ := initial_amount + saved_amount

-- Define the total amount spent which equals
-- the sum of amount spent on the comic book and the puzzle
def total_spent : ℤ := total_amount - amount_left

-- Define the amount spent on the comic book as the proposition to be proved
def spent_on_comic_book : ℤ := total_spent - spent_on_puzzle

-- State the theorem to prove how much Elise spent on the comic book
theorem elise_spent_on_comic_book : spent_on_comic_book = 2 :=
by
  sorry

end elise_spent_on_comic_book_l244_244862


namespace magic_8_ball_probability_l244_244114

noncomputable def probability_exactly_three_positive_answers :=
  let prob_positive := 1 / 4 in
  let prob_negative := 3 / 4 in
  let num_combinations := Nat.choose 6 3 in
  num_combinations * (prob_positive ^ 3) * (prob_negative ^ 3)

theorem magic_8_ball_probability :
  probability_exactly_three_positive_answers = 135 / 1024 :=
by
  sorry

end magic_8_ball_probability_l244_244114


namespace otimes_eq_abs_m_leq_m_l244_244433

noncomputable def otimes (x y : ℝ) : ℝ :=
if x ≤ y then x else y

theorem otimes_eq_abs_m_leq_m' :
  ∀ (m : ℝ), otimes (abs (m - 1)) m = abs (m - 1) → m ∈ Set.Ici (1 / 2) := 
by
  sorry

end otimes_eq_abs_m_leq_m_l244_244433


namespace atomic_weight_chlorine_l244_244312

-- Define the given conditions and constants
def molecular_weight_compound : ℝ := 53
def atomic_weight_nitrogen : ℝ := 14.01
def atomic_weight_hydrogen : ℝ := 1.01
def number_of_hydrogen_atoms : ℝ := 4
def number_of_nitrogen_atoms : ℝ := 1

-- Define the total weight of nitrogen and hydrogen in the compound
def total_weight_nh : ℝ := (number_of_nitrogen_atoms * atomic_weight_nitrogen) + (number_of_hydrogen_atoms * atomic_weight_hydrogen)

-- Define the statement to be proved: the atomic weight of chlorine
theorem atomic_weight_chlorine : (molecular_weight_compound - total_weight_nh) = 34.95 := by
  sorry

end atomic_weight_chlorine_l244_244312


namespace total_scientists_l244_244778

theorem total_scientists (WP NP WP_cap_NP Nobel_only three_greater : ℕ) 
  (h1 : WP = 31) 
  (h2 : WP_cap_NP = 18) 
  (h3 : NP = 29) 
  (h4 : Nobel_only = NP - WP_cap_NP)
  (h5 : Nobel_only = three_greater + (NP - WP)) 
  : 
  let not_NP := Nobel_only - 3 in
  let total := WP + (Nobel_only - 3) + Nobel_only in
  total = 50 :=
by {
  sorry
}

end total_scientists_l244_244778


namespace collinear_unit_vector_l244_244750

def vector3 := ℝ × ℝ × ℝ

def is_unit_vector (v : vector3) : Prop :=
  let (x, y, z) := v in x^2 + y^2 + z^2 = 1

def are_collinear (v₁ v₂ : vector3) : Prop :=
  ∃ k : ℝ, v₂ = (k * v₁.1, k * v₁.2, k * v₁.3)

def vec_a : vector3 := (3, 0, -4)

def magnitude (v : vector3) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem collinear_unit_vector :
  magnitude vec_a = 5 →
  is_unit_vector (-3/5, 0, 4/5) →
  are_collinear vec_a (-3/5, 0, 4/5) :=
 by
  sorry

end collinear_unit_vector_l244_244750


namespace officers_on_duty_l244_244274

theorem officers_on_duty
  (F : ℕ)                             -- Total female officers on the police force
  (on_duty_percentage : ℕ)            -- On duty percentage of female officers
  (H1 : on_duty_percentage = 18)      -- 18% of the female officers were on duty
  (H2 : F = 500)                      -- There were 500 female officers on the police force
  : ∃ T : ℕ, T = 2 * (on_duty_percentage * F) / 100 ∧ T = 180 :=
by
  sorry

end officers_on_duty_l244_244274


namespace parabola_conditions_l244_244238

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244238


namespace countLatticePoints_l244_244093

def isLatticePoint (x y : Int) : Prop :=
  2 * x ^ 2 + y ^ 2 = 18

def latticePoints := {(x, y) : Int × Int // isLatticePoint x y}

theorem countLatticePoints : latticePoints.card = 6 :=
by {
  sorry
}

end countLatticePoints_l244_244093


namespace meteorological_forecast_l244_244673

theorem meteorological_forecast (prob_rain : ℝ) (h1 : prob_rain = 0.7) :
  (prob_rain = 0.7) → "There is a high probability of needing to carry rain gear when going out tomorrow." = "Correct" :=
by
  intro h
  sorry

end meteorological_forecast_l244_244673


namespace number_of_possible_plans_most_cost_effective_plan_l244_244729

-- Defining the conditions of the problem
def price_A := 12 -- Price of model A in million yuan
def price_B := 10 -- Price of model B in million yuan
def capacity_A := 240 -- Treatment capacity of model A in tons/month
def capacity_B := 200 -- Treatment capacity of model B in tons/month
def total_budget := 105 -- Total budget in million yuan
def min_treatment_volume := 2040 -- Minimum required treatment volume in tons/month
def total_units := 10 -- Total number of units to be purchased

def valid_purchase_plan (x y : ℕ) :=
  x + y = total_units ∧
  price_A * x + price_B * y ≤ total_budget ∧
  capacity_A * x + capacity_B * y ≥ min_treatment_volume

-- Stating the theorem for how many possible purchase plans exist
theorem number_of_possible_plans : 
  ∃ k : ℕ, k = 3 ∧
    (∀ (x y : ℕ), 
      valid_purchase_plan x y →
      x ∈ {0, 1, 2} ∧ y = total_units - x) :=
sorry

-- Stating the theorem for the most cost-effective plan
theorem most_cost_effective_plan :
  ∃ (x y : ℕ),
    valid_purchase_plan x y ∧
    price_A * x + price_B * y = 102 ∧
    x = 1 ∧ y = 9 :=
sorry

end number_of_possible_plans_most_cost_effective_plan_l244_244729


namespace monthly_salary_l244_244759

theorem monthly_salary (S : ℝ) (h1 : 0.20 * S + 1.20 * 0.80 * S = S) (h2 : S - 1.20 * 0.80 * S = 260) : S = 6500 :=
by
  sorry

end monthly_salary_l244_244759


namespace imaginary_part_of_z_l244_244926

noncomputable def z : ℂ := (1 - 2 * Complex.i) / (2 + Complex.i)

theorem imaginary_part_of_z : z.im = -1 := by
  sorry

end imaginary_part_of_z_l244_244926


namespace flight_time_sum_l244_244594

theorem flight_time_sum (h m : ℕ)
  (Hdep : true)   -- Placeholder condition for the departure time being 3:45 PM
  (Hlay : 25 = 25)   -- Placeholder condition for the layover being 25 minutes
  (Harr : true)   -- Placeholder condition for the arrival time being 8:02 PM
  (HsameTZ : true)   -- Placeholder condition for the same time zone
  (H0m : 0 < m) 
  (Hm60 : m < 60)
  (Hfinal_time : (h, m) = (3, 52)) : 
  h + m = 55 := 
by {
  sorry
}

end flight_time_sum_l244_244594


namespace tangent_line_circle_sol_l244_244068

theorem tangent_line_circle_sol (r : ℝ) (h_pos : r > 0)
  (h_tangent : ∀ x y : ℝ, x^2 + y^2 = 2 * r → x + 2 * y = r) : r = 10 := 
sorry

end tangent_line_circle_sol_l244_244068


namespace exists_real_x_l244_244612

noncomputable def distance_to_nearest_integer (x : ℝ) : ℝ :=
  abs (x - round x)

theorem exists_real_x (n : ℕ) (a : Fin n → ℕ) (ha : ∀ i, 0 < a i) :
  ∃ x ∈ Ioo 0 1, ∀ i, distance_to_nearest_integer (a i * x) ≥ 1 / (2 * n) :=
by
  sorry

end exists_real_x_l244_244612


namespace GAUSS_1998_LCM_l244_244307

/-- The periodicity of cycling the word 'GAUSS' -/
def period_GAUSS : ℕ := 5

/-- The periodicity of cycling the number '1998' -/
def period_1998 : ℕ := 4

/-- The least common multiple (LCM) of the periodicities of 'GAUSS' and '1998' is 20 -/
theorem GAUSS_1998_LCM : Nat.lcm period_GAUSS period_1998 = 20 :=
by
  sorry

end GAUSS_1998_LCM_l244_244307


namespace max_cardinality_32_l244_244247

noncomputable def max_set_cardinality (S : Set ℕ) : Prop :=
  ∀ x ∈ S, ∃ S' : Set ℕ, S' = S \ {x} ∧ (S.card > 0 → (∑ y in S', y) % S.card = 0)

theorem max_cardinality_32 :
  ∀ S : Set ℕ, (1 ∈ S ∧ 2016 ∈ S ∧ (∀ x ∈ S, 1 ≤ x) ∧ max S = 2016 ∧ max_set_cardinality S) →
  S.card ≤ 32 :=
sorry

end max_cardinality_32_l244_244247


namespace angle_BAF_eq_angle_CAG_l244_244581

open EuclideanGeometry 

variables {A B C D E F G : Point}
variables {O P : Circle}

-- Given conditions
axiom triangle_ABC : IsTriangle A B C
axiom D_on_AB : OnLine D (LineThrough A B)
axiom E_on_AC : OnLine E (LineThrough A C)
axiom DE_parallel_BC : Parallel (LineThrough D E) (LineThrough B C)
axiom F_intersection_BE_CD : OnLine F (LineThrough B E) ∧ OnLine F (LineThrough C D)
axiom O_circumcircle_BDF : Circumcircle O B D F
axiom P_circumcircle_CEF : Circumcircle P C E F
axiom G_intersection_O_P : OnCircle G O ∧ OnCircle G P

-- Proof goal
theorem angle_BAF_eq_angle_CAG : ∠ B A F = ∠ C A G :=
  sorry

end angle_BAF_eq_angle_CAG_l244_244581


namespace intersection_A_B_l244_244909

def A : Set ℝ := {x | 2*x - 1 ≤ 0}
def B : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_A_B : A ∩ B = {x | 0 < x ∧ x ≤ 1/2} := 
by 
  sorry

end intersection_A_B_l244_244909


namespace area_R2_l244_244009

-- Definitions from conditions
def side_R1 : ℕ := 3
def area_R1 : ℕ := 24
def diagonal_ratio : ℤ := 2

-- Introduction of the theorem
theorem area_R2 (similar: ℤ) (a b: ℕ) :
  a * b = area_R1 ∧
  a = 3 ∧
  b * 3 = 8 * a ∧
  (a^2 + b^2 = 292) ∧
  similar * (a^2 + b^2) = 2 * 2 * 73 →
  (6 * 16 = 96) := by
sorry

end area_R2_l244_244009


namespace eggs_per_meal_l244_244264

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end eggs_per_meal_l244_244264


namespace cosine_of_angle_l244_244322

variable {A B C G H : Point}
variable {AB AC BC CA GH : ℝ}
variable {θ : ℝ}

-- Conditions
axiom midpoint_B : midpoint B G H
axiom AB_eq : AB = 1
axiom GH_eq : GH = 1
axiom BC_eq : BC = 8
axiom CA_eq : CA = sqrt 41
axiom dot_product_condition : (AB • AG) + (AC • AH) = 0

noncomputable def cos_angle_GH_BC : ℝ :=
  by
    sorry

-- Theorem
theorem cosine_of_angle : cos_angle_GH_BC = -1 / 4 :=
  by
    sorry

end cosine_of_angle_l244_244322


namespace relationship_among_a_b_c_l244_244022

theorem relationship_among_a_b_c (f : ℝ → ℝ)
  (hf_even : ∀ x, f x = f (-x))
  (hf_inc : ∀ x y, x < y ∧ y ≤ 0 → f x < f y)
  (a := f (Real.log 7 / Real.log 4))
  (b := f (Real.log 3 / Real.log (1/2)))
  (c := f (0.2 ^ 0.6)) :
  b < a ∧ a < c :=
by sorry

end relationship_among_a_b_c_l244_244022


namespace number_of_solutions_f_eq_x_l244_244067

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 4 * x + 2 else 2

theorem number_of_solutions_f_eq_x : set_of (λ x, f x = x).finite.card = 3 := by
  sorry

end number_of_solutions_f_eq_x_l244_244067


namespace cakes_initially_made_l244_244834

variables (sold bought total initial_cakes : ℕ)

theorem cakes_initially_made (h1 : sold = 105) (h2 : bought = 170) (h3 : total = 186) :
  initial_cakes = total - (sold - bought) :=
by
  rw [h1, h2, h3]
  sorry

end cakes_initially_made_l244_244834


namespace M_gt_N_l244_244069

variable {x y : ℝ}

theorem M_gt_N (h1 : x ≠ 2 ∨ y ≠ -1) (h2 : M = x^2 + y^2 - 4x + 2y) (h3 : N = -5) : M > N :=
by
  -- Proof will be inserted here
  sorry

end M_gt_N_l244_244069


namespace probability_at_least_6_heads_l244_244794

open Finset

noncomputable def binom (n k : ℕ) : ℕ := (finset.range (k + 1)).sum (λ i, if i.choose k = 0 then 0 else n.choose k)

theorem probability_at_least_6_heads : 
  (finset.sum (finset.range 10) (λ k, if k >= 6 then (nat.choose 9 k : ℚ) else 0)) / 2^9 = (130 : ℚ) / 512 :=
by sorry

end probability_at_least_6_heads_l244_244794


namespace sequence_periodicity_a5_a2019_l244_244383

theorem sequence_periodicity_a5_a2019 (a : ℕ → ℝ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, 0 < n → a n * a (n + 2) = 3 * a (n + 1)) :
  a 5 * a 2019 = 27 :=
sorry

end sequence_periodicity_a5_a2019_l244_244383


namespace arithmetic_seq_third_term_l244_244094

variable {a_n : ℕ → ℕ --Define arithmetic sequence a_n }

def is_arithmetic_sequence (seq : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, seq (n+1) - seq n = d

def sum_first_n (seq : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (seq 1 + seq n))/2

theorem arithmetic_seq_third_term (seq : ℕ → ℕ) (h1 : is_arithmetic_sequence seq) (h2 : sum_first_n seq 5 = 20) : seq 3 = 4 :=
sorry

end arithmetic_seq_third_term_l244_244094


namespace weight_of_replaced_student_l244_244680

-- Define the conditions as hypotheses
variable (W : ℝ)
variable (h : W - 46 = 40)

-- Prove that W = 86
theorem weight_of_replaced_student : W = 86 :=
by
  -- We should conclude the proof; for now, we leave a placeholder
  sorry

end weight_of_replaced_student_l244_244680


namespace admission_fee_for_adults_l244_244833

-- Define the given conditions
def total_attendees := 578
def num_adults := 342
def total_receipts := 985.00

-- Define the price paid by each adult
def price_per_adult := total_receipts / num_adults

-- State the theorem
theorem admission_fee_for_adults : price_per_adult = 985.00 / 342 := sorry

end admission_fee_for_adults_l244_244833


namespace bags_production_l244_244358

def machines_bags_per_minute (n : ℕ) : ℕ :=
  if n = 15 then 45 else 0 -- this definition is constrained by given condition

def bags_produced (machines : ℕ) (minutes : ℕ) : ℕ :=
  machines * (machines_bags_per_minute 15 / 15) * minutes

theorem bags_production (machines minutes : ℕ) (h : machines = 150 ∧ minutes = 8):
  bags_produced machines minutes = 3600 :=
by
  cases h with
  | intro h_machines h_minutes =>
    sorry

end bags_production_l244_244358


namespace find_value_of_cos_diff_plus_c_l244_244020

-- Definition for the given problem
def A (a b c : ℝ) : Set ℝ :=
  {sin a, cos b, 0, 1, -2, (Real.sqrt 2) / 2, Real.log c}

def B (a b c : ℝ) : Set ℝ :=
  {x | ∃ (a' ∈ A a b c), x = a' ^ 2022 + a' ^ 2}

axiom h_card_B (a b c : ℝ) : 
  Set.card (B a b c) = 4

noncomputable 
def cos_diff_plus_c (a b c : ℝ) : ℝ :=
  Real.cos (2 * a) - Real.cos (2 * b) + c 

theorem find_value_of_cos_diff_plus_c (a b c : ℝ) (h : c = Real.exp 1)  :
  cos_diff_plus_c a b c = 1 + Real.exp 1 :=
by
  sorry

end find_value_of_cos_diff_plus_c_l244_244020


namespace prime_count_l244_244525

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def from_digits (tens units : ℕ) : ℕ :=
  10 * tens + units

def is_valid_prime (tens units : ℕ) : Prop :=
  {3, 5, 7, 9}.contains tens ∧ 
  {3, 5, 7, 9}.contains units ∧ 
  tens ≠ units ∧ 
  is_prime (from_digits tens units)

theorem prime_count : 
  (finset.univ.filter (λ p, ∃ tens ∈ {3, 5, 7, 9}, ∃ units ∈ {3, 5, 7, 9}, tens ≠ units ∧ is_prime (from_digits tens units))).card = 6 :=
by
  sorry

end prime_count_l244_244525


namespace Natasha_speed_over_limit_l244_244644

theorem Natasha_speed_over_limit (d : ℕ) (t : ℕ) (speed_limit : ℕ) 
    (h1 : d = 60) 
    (h2 : t = 1) 
    (h3 : speed_limit = 50) : (d / t - speed_limit = 10) :=
by
  -- Because d = 60, t = 1, and speed_limit = 50, we need to prove (60 / 1 - 50) = 10
  sorry

end Natasha_speed_over_limit_l244_244644


namespace enhanced_computer_more_expensive_l244_244716

def basic_computer_price := 2000
def total_price := 2500
def printer_price := total_price - basic_computer_price
def enhanced_computer_price := (6 * (printer_price * 1/6)) - printer_price

theorem enhanced_computer_more_expensive :
  enhanced_computer_price - basic_computer_price = 500 :=
by
  have printer_price_eq : printer_price = 500 := by 
    simp [printer_price, total_price, basic_computer_price]
  have enhanced_price_eq : enhanced_computer_price = 2500 := by
    rw printer_price_eq
    simp [enhanced_computer_price, printer_price]
  rw [enhanced_price_eq, printer_price_eq]
  simp [basic_computer_price]
  exact rfl

end enhanced_computer_more_expensive_l244_244716


namespace find_pair_l244_244437

theorem find_pair (a b : ℤ) :
  (∀ x : ℝ, (a * x^4 + b * x^3 + 20 * x^2 - 12 * x + 10) = (2 * x^2 + 3 * x - 4) * (c * x^2 + d * x + e)) → 
  (a = 2) ∧ (b = 27) :=
sorry

end find_pair_l244_244437


namespace sum_of_specific_four_digit_perfect_squares_l244_244718

theorem sum_of_specific_four_digit_perfect_squares :
  ∃ (n1 n17r : ℕ), 
    (∀ (d1 d2 d3 d4 : ℕ), {d1, d2, d3, d4} ⊆ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧
    (n1 = 1089) ∧ (n17r = 9801) ∧
    nat.sqrt n1 * nat.sqrt n1 = n1 ∧ 
    nat.sqrt n17r = ((nat.sqrt n17r).digits 10).reverse.foldl (λ acc x, acc * 10 + x) 0 ∧
    n1 + n17r = 10890 :=
begin
  sorry
end

end sum_of_specific_four_digit_perfect_squares_l244_244718


namespace find_a_l244_244035

noncomputable def f (a x : ℝ) : ℝ := a^x - 4 * a + 3

theorem find_a (H : ∃ (a : ℝ), ∃ (x y : ℝ), f a x = y ∧ f y x = a ∧ x = 2 ∧ y = -1): ∃ a : ℝ, a = 2 :=
by
  obtain ⟨a, x, y, hx, hy, hx2, hy1⟩ := H
  --skipped proof
  sorry

end find_a_l244_244035


namespace find_digits_of_abc_l244_244282

theorem find_digits_of_abc (a b c : ℕ) (h1 : a ≠ c) (h2 : c - a = 3) (h3 : (100 * a + 10 * b + c) - (100 * c + 10 * a + b) = 100 * (a - (c - 1)) + 0 + (b - b)) : 
  100 * a + 10 * b + c = 619 :=
by
  sorry

end find_digits_of_abc_l244_244282


namespace shaded_triangle_area_l244_244685

theorem shaded_triangle_area 
(perimeter_large_square : ℕ)
(perimeter_small_square : ℕ)
(h_large_sq : perimeter_large_square = 28)
(h_small_sq : perimeter_small_square = 20) :
  let side_large_square := perimeter_large_square / 4,
      side_small_square := perimeter_small_square / 4,
      area_large_square := side_large_square * side_large_square,
      area_small_square := side_small_square * side_small_square,
      area_difference := area_large_square - area_small_square,
      num_triangles := 4,
      area_one_triangle := area_difference / num_triangles
  in area_one_triangle = 6 :=
sorry

end shaded_triangle_area_l244_244685


namespace parabola_focus_line_tangent_circle_l244_244208

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244208


namespace increase_in_average_l244_244851

theorem increase_in_average (s1 s2 s3 s4 s5: ℝ)
  (h1: s1 = 92) (h2: s2 = 86) (h3: s3 = 89) (h4: s4 = 94) (h5: s5 = 91):
  ( ((s1 + s2 + s3 + s4 + s5) / 5) - ((s1 + s2 + s3) / 3) ) = 1.4 :=
by
  sorry

end increase_in_average_l244_244851


namespace count_valid_temperatures_l244_244302

open BigOperators

noncomputable def convert_to_celsius (T : ℤ) : ℤ :=
Int.floor ((5 : ℚ)/(9 : ℚ) * (T - 32) + 0.5)

noncomputable def convert_to_fahrenheit (C : ℤ) : ℤ :=
Int.floor ((9 : ℚ)/(5 : ℚ) * ↑C + 32 + 0.5)

theorem count_valid_temperatures :
  {T : ℤ | 32 ≤ T ∧ T ≤ 1000 ∧ T = convert_to_fahrenheit (convert_to_celsius T)}.to_finset.card = 539 := 
sorry

end count_valid_temperatures_l244_244302


namespace max_peripheral_cities_l244_244765

-- Define the context: 100 cities, unique paths, and transfer conditions
def number_of_cities := 100
def max_transfers := 11
def max_peripheral_transfers := 10

-- Statement: Prove the maximum number of peripheral cities
theorem max_peripheral_cities (cities : Finset (Fin number_of_cities)) 
  (h_unique_paths : ∀ (A B : Fin number_of_cities), ∃! (path : Finset (Fin number_of_cities)), 
    path.card ≤ max_transfers + 1) 
  (h_reachable : ∀ (A B : Fin number_of_cities), 
    ∃ (path : Finset (Fin number_of_cities)), path.card ≤ max_transfers + 1) 
  (h_peripheral : ∀ (A B : Fin number_of_cities), 
    ¬(A ≠ B ∧ path.card ≤ max_peripheral_transfers + 1)) : 
  ∃ (peripheral : Finset (Fin number_of_cities)), peripheral.card = 89 := 
sorry

end max_peripheral_cities_l244_244765


namespace cost_of_painting_new_room_l244_244684

theorem cost_of_painting_new_room
  (L B H : ℝ)    -- Dimensions of the original room
  (c : ℝ)        -- Cost to paint the original room
  (h₁ : c = 350) -- Given that the cost of painting the original room is Rs. 350
  (A : ℝ)        -- Area of the walls of the original room
  (h₂ : A = 2 * (L + B) * H) -- Given the area calculation for the original room
  (newA : ℝ)     -- Area of the walls of the new room
  (h₃ : newA = 18 * (L + B) * H) -- Given the area calculation for the new room
  : (350 / (2 * (L + B) * H)) * (18 * (L + B) * H) = 3150 :=
by
  sorry

end cost_of_painting_new_room_l244_244684


namespace probability_sum_greater_than_9_l244_244984

def num_faces := 6
def total_outcomes := num_faces * num_faces
def favorable_outcomes := 6
def probability := favorable_outcomes / total_outcomes

theorem probability_sum_greater_than_9 (h : total_outcomes = 36) :
  probability = 1 / 6 :=
by
  sorry

end probability_sum_greater_than_9_l244_244984


namespace cows_in_group_l244_244980

variable (c h : ℕ)

/--
In a group of cows and chickens, the number of legs was 20 more than twice the number of heads.
Cows have 4 legs each and chickens have 2 legs each.
Each animal has one head.
-/
theorem cows_in_group (h : ℕ) (hc : 4 * c + 2 * h = 2 * (c + h) + 20) : c = 10 :=
by
  sorry

end cows_in_group_l244_244980


namespace parabola_condition_l244_244174

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244174


namespace convex_polyhedron_has_triangular_face_l244_244961

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end convex_polyhedron_has_triangular_face_l244_244961


namespace range_of_a_l244_244084

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ 2 * x^2 - a * x + 2 > 0) ↔ a < 5 := sorry

end range_of_a_l244_244084


namespace parabola_conditions_l244_244236

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244236


namespace f_2010_2013_l244_244029

noncomputable theory

open Function

def f : ℝ → ℝ := sorry

axiom f_defined : ∀ x : ℝ, f x ≠ sorry
axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_2x_eq_x : ∀ x : ℝ, f (2 - x) = f x
axiom f_1_eq_1 : f 1 = 1

theorem f_2010_2013 : f 2010 + f 2013 = 1 := sorry

end f_2010_2013_l244_244029


namespace smallest_x_for_g_g_l244_244074

def g (x : ℝ) := real.sqrt (x - 5)

theorem smallest_x_for_g_g (x : ℝ) : (∀ x, (g (g x)).is_defined → x ≥ 30) :=
by
  intros x hx
  sorry

end smallest_x_for_g_g_l244_244074


namespace sum_fractional_parts_zeta_l244_244423

noncomputable def fractional_part (x : ℝ) : ℝ := x - x.floor

theorem sum_fractional_parts_zeta :
  (∑ k in Finset.range 5, fractional_part (Real.zeta (2 * k + 1))) = 0.75 :=
by
  sorry

end sum_fractional_parts_zeta_l244_244423


namespace general_formula_for_a_n_sum_of_first_n_terms_of_bn_l244_244925

-- Define the arithmetic sequence {a_n}
def a_n (n : ℕ) : ℤ := 11 - 2 * n

-- Define the sequence {b_n} as the absolute value of {a_n}
def b_n (n : ℕ) : ℕ := (11 - 2 * n).natAbs

-- Define the sum of first n terms of {a_n}, S_n
def S_n (n : ℕ) : ℕ := (n * (2 * 9 + (n - 1) * -2)) / 2

-- Define the sum of first n terms of {b_n}, T_n
def T_n (n : ℕ) : ℤ :=
  if n ≤ 5 then 10 * n - n * n
  else n * n - 10 * n + 50

-- Proving the general formula for {a_n}
theorem general_formula_for_a_n (n : ℕ) : a_n n = 11 - 2 * n :=
by
  sorry

-- Proving the sum of the first n terms of {b_n}
theorem sum_of_first_n_terms_of_bn (n : ℕ) : 
  T_n n = if n ≤ 5 then 10 * n - n * n else n * n - 10 * n + 50 :=
by
  sorry

end general_formula_for_a_n_sum_of_first_n_terms_of_bn_l244_244925


namespace parabola_focus_line_tangent_circle_l244_244206

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244206


namespace certain_number_is_25_l244_244367

theorem certain_number_is_25 :
  ∃ x : ℚ, 22 = (4/5 : ℚ) * x + 2 ∧ x = 25 := 
by
  use 25
  split
  . calc
    22 = (0.55 : ℚ) * 40 := by norm_num
    ... = (22 : ℚ) : by rw [mul_comm, mul_div_cancel_left 40 (show 55 / 100 ≠ 0, by norm_num)]
    ... = (4/5 : ℚ) * 25 + 2 : by norm_num
    ... = 22 : by norm_num
  . refl

end certain_number_is_25_l244_244367


namespace find_judes_age_l244_244559

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end find_judes_age_l244_244559


namespace male_students_count_l244_244295

theorem male_students_count
  (average_all_students : ℕ → ℕ → ℚ → Prop)
  (average_male_students : ℕ → ℚ → Prop)
  (average_female_students : ℕ → ℚ → Prop)
  (F : ℕ)
  (total_average : average_all_students (F + M) (83 * M + 92 * F) 90)
  (male_average : average_male_students M 83)
  (female_average : average_female_students 28 92) :
  ∃ (M : ℕ), M = 8 :=
by {
  sorry
}

end male_students_count_l244_244295


namespace proof_problem_l244_244156

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244156


namespace max_MN_distance_l244_244960

noncomputable def f (x : ℝ) : ℝ := 2 * real.cos(π / 4 + x) ^ 2

noncomputable def g (x : ℝ) : ℝ := real.sqrt 3 * real.cos (2 * x)

def MN_distance (a : ℝ) : ℝ := |f a - g a|

theorem max_MN_distance : ∃ a : ℝ, MN_distance a = 3 :=
begin
  sorry
end

end max_MN_distance_l244_244960


namespace a_2016_eq_0_l244_244012

-- Define the sequence {a_n}
def a : ℕ → ℝ
| 0     := 1  -- Conventionally, Lean sequences are 0-indexed. a_0 refers to a_1 in the problem.
| (n+1) := a n + cos (n * Real.pi / 3)

-- The theorem to prove
theorem a_2016_eq_0 : a 2015 = 0 := sorry

end a_2016_eq_0_l244_244012


namespace k_is_odd_l244_244292

theorem k_is_odd (m n k : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_pos_k : 0 < k) (h : 3 * m * k = (m + 3)^n + 1) : Odd k :=
by {
  sorry
}

end k_is_odd_l244_244292


namespace distance_EC_l244_244687

/-
We are given:
  AB = 30
  BC = 80
  CD = 236
  DE = 86
  EA = 40

We need to prove:
  EC = 150
-/

theorem distance_EC : ∀ (A B C D E : Type) 
  [mat : MetricSpace A] [matB : MetricSpace B] [matC : MetricSpace C] [matD : MetricSpace D] [matE : MetricSpace E],
  dist A B = 30 ∧ dist B C = 80 ∧ dist C D = 236 ∧ dist D E = 86 ∧ dist E A = 40 →
  dist A E + dist C B + dist B A = 150 :=
  sorry

end distance_EC_l244_244687


namespace eval_f_f3_l244_244021

-- Define the function f
def f (x : ℚ) : ℚ :=
  x⁻² + x⁻² / (1 + x⁻²)

-- Define the specific problem of evaluating f(f(3))
theorem eval_f_f3 : f (f 3) = 10198200 / 3094021 := 
by {
  sorry -- Proof goes here
}

end eval_f_f3_l244_244021


namespace smallest_b_l244_244291

open Nat

-- Define the conditions
def cond1 (a b : ℕ) : Prop := a - b = 4
def cond2 (a b : ℕ) : Prop := gcd ((a^3 - b^3) / (a - b)) (a * b) = 4

-- Proof problem
theorem smallest_b (b : ℕ) (h1 : ∃ (a : ℕ), a - b = 4 ∧ gcd ((a^3 - b^3) / (a - b)) (a * b) = 4) : b = 2 :=
by sorry

end smallest_b_l244_244291


namespace legendre_transform_convex_double_legendre_transform_f_as_integral_expectation_legendre_l244_244255

noncomputable section

-- Part (a)
theorem legendre_transform_convex {f : ℝ → ℝ} (hf : convex_on ℝ f) :
  convex_on ℝ (λ y, supr (λ x, x * y - f x)) :=
sorry

theorem double_legendre_transform {f : ℝ → ℝ} (hf : convex_on ℝ f) :
  ∀ x, f x = supr (λ y, x * y - supr (λ x, x * y - f x)) :=
sorry

-- Part (b)
theorem f_as_integral {f : ℝ → ℝ} (hf : ∀ x, f x = ∫⁻ u in -∞..x, h u) (h : ℝ → ℝ) (hf : non_decreasing h) :
  ∀ p x, p ≥ 1 → f x = (x ^ p) / p → ∀ q, q = p / (p - 1) → supr (λ x, x * y - f x) = y ^ q / q :=
sorry

-- Part (c)
theorem expectation_legendre {X : ℝ → ℝ} (hX : E (λ x, max 0 (-X x)) < ∞) :
  ∀ (f := λ x, E (λ x, max 0 (x - X x))),
  supr (λ y, y * x - f x) = ∫⁻ u, (P (X ≤ u)) :=
sorry

end legendre_transform_convex_double_legendre_transform_f_as_integral_expectation_legendre_l244_244255


namespace price_per_chocolate_cookie_l244_244350

-- Define the known quantities
def num_chocolate_cookies : ℕ := 220
def num_vanilla_cookies : ℕ := 70
def price_per_vanilla_cookie : ℝ := 2
def total_revenue : ℝ := 360

-- Define the unknown variable
variable (x : ℝ)

-- Define the total revenue from chocolate cookies
def revenue_from_chocolate_cookies := num_chocolate_cookies * x

-- Define the total revenue from vanilla cookies
def revenue_from_vanilla_cookies := num_vanilla_cookies * price_per_vanilla_cookie

-- State the proof goal
theorem price_per_chocolate_cookie :
  revenue_from_chocolate_cookies + revenue_from_vanilla_cookies = total_revenue → x = 1 := by
  intros h
  sorry

end price_per_chocolate_cookie_l244_244350


namespace range_of_k_l244_244553

theorem range_of_k (k : ℝ) : 
  (∀ x y : ℝ, 0 < x → 0 < y → sqrt x + sqrt y ≤ k * sqrt (2 * x + y)) ↔ k ≥ sqrt 6 / 2 :=
by
  split
  · intro h
    -- Proof omitted
    sorry
  · intros hk x y hx hy
    -- Proof omitted
    sorry

end range_of_k_l244_244553


namespace balance_cliques_l244_244981

namespace competition

-- Definitions for mutual friendship and cliques
def mutual_friendship (competitors : Type) (friend : competitors → competitors → Prop) : Prop :=
  ∀ (x y : competitors), friend x y → friend y x

def clique (competitors : Type) (friend : competitors → competitors → Prop) (group : set competitors) : Prop :=
  ∀ x y ∈ group, friend x y

def largest_clique_even (competitors : Type) (friend : competitors → competitors → Prop) (size : ℕ) : Prop :=
  ∀ (group : set competitors), (clique competitors friend group) → (size ≤ group.card) → (size % 2 = 0)

-- Problem statement
theorem balance_cliques (competitors : Type) (friend : competitors → competitors → Prop) (size : ℕ)
  (mf : mutual_friendship competitors friend)
  (even_size : largest_clique_even competitors friend size) :
  ∃ (room1 room2 : set competitors), 
    (∀ x y ∈ room1, friend x y) ∧ 
    (∀ x y ∈ room2, friend x y) ∧ 
    (¬∃ z, z ∈ room1 ∧ z ∈ room2) ∧ 
    (size = room1.card ∧ size = room2.card) :=
sorry

end competition

end balance_cliques_l244_244981


namespace power_function_evaluation_l244_244494

noncomputable def f (α : ℝ) (x : ℝ) := x ^ α

theorem power_function_evaluation (α : ℝ) (h : f α 8 = 2) : f α (-1/8) = -1/2 :=
by
  sorry

end power_function_evaluation_l244_244494


namespace proof_problem_l244_244148

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244148


namespace jude_age_today_l244_244561
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end jude_age_today_l244_244561


namespace saved_percentage_is_correct_l244_244395

def rent : ℝ := 5000
def milk : ℝ := 1500
def groceries : ℝ := 4500
def education : ℝ := 2500
def petrol : ℝ := 2000
def miscellaneous : ℝ := 5200
def amount_saved : ℝ := 2300

noncomputable def total_expenses : ℝ :=
  rent + milk + groceries + education + petrol + miscellaneous

noncomputable def total_salary : ℝ :=
  total_expenses + amount_saved

noncomputable def percentage_saved : ℝ :=
  (amount_saved / total_salary) * 100

theorem saved_percentage_is_correct :
  percentage_saved = 8.846 := by
  sorry

end saved_percentage_is_correct_l244_244395


namespace fraction_order_l244_244347

theorem fraction_order :
  (19 / 15 < 17 / 13) ∧ (17 / 13 < 15 / 11) :=
by
  sorry

end fraction_order_l244_244347


namespace find_f_function_l244_244869

noncomputable def f : ℕ → ℕ := sorry -- The function we need to determine

-- Define the nth iteration of f
def iter (f : ℕ → ℕ) : ℕ → ℕ → ℕ
| 0, x := x
| (n + 1), x := f (iter f n x)

-- The theorem to prove
theorem find_f_function :
  (∀ a b : ℕ, 0 < a → 0 < b → iter f a b + iter f b a ∣ 2 * (f (a * b) + b ^ 2 - 1)) →
  (f = λ x, x + 1 ∨ ∃ k : ℕ, k ∣ 4 ∧ f 1 = k) :=
begin
  sorry
end

end find_f_function_l244_244869


namespace parabola_focus_line_tangent_circle_l244_244213

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244213


namespace multiples_7_not_14_less_350_l244_244946

theorem multiples_7_not_14_less_350 : 
  ∃ n : ℕ, n = 25 ∧ (∀ k : ℕ, k < 350 → (k % 7 = 0 ∧ k % 14 ≠ 0 → k ∈ {7 * m | m : ℕ}) ∨ (k % 14 = 0 → k ∉ {7 * m | m : ℕ})) := 
sorry

end multiples_7_not_14_less_350_l244_244946


namespace find_m_l244_244912

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
variables (e1 e2 : V) (a : V) (b : V) (m : ℝ)

-- Conditions
def non_collinear (e1 e2 : V) := ¬collinear ℝ ({(0:V), e1, e2} : set V)
def a_def := a = 2 • e1 - e2
def b_def := b = m • e1 + 3 • e2
def parallel (a b : V) := ∃ μ : ℝ, a = μ • b

-- Goal
theorem find_m (hne : non_collinear e1 e2)
(a_def : a = 2 • e1 - e2)
(b_def : b = m • e1 + 3 • e2)
(parallel_ab : parallel a b) : m = -6 := 
begin
  sorry
end

end find_m_l244_244912


namespace find_D_l244_244047

def Point3D := (ℝ × ℝ × ℝ)

def A : Point3D := (2, 0, 3)
def B : Point3D := (0, 3, -5)
def C : Point3D := (0, 0, 3)
def D : Point3D := (2, -3, 11)

def midpoint (p1 p2 : Point3D) : Point3D :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

theorem find_D (D : Point3D) :
  let M_AC := midpoint A C
  let M_BD := midpoint B D
  M_AC = M_BD →
  D = (2, -3, 11) :=
by
  intros
  simp [midpoint, A, B, C, D] at *
  sorry

end find_D_l244_244047


namespace base6_addition_l244_244731

/-- Adding two numbers in base 6 -/
theorem base6_addition : (3454 : ℕ) + (12345 : ℕ) = (142042 : ℕ) := by
  sorry

end base6_addition_l244_244731


namespace total_dogs_on_farm_l244_244722

-- Definitions based on conditions from part a)
def num_dog_houses : ℕ := 5
def num_dogs_per_house : ℕ := 4

-- Statement to prove
theorem total_dogs_on_farm : num_dog_houses * num_dogs_per_house = 20 :=
by
  sorry

end total_dogs_on_farm_l244_244722


namespace vector_ab_is_correct_circle_symmetry_l244_244085

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def reflect_point (p : ℝ × ℝ) (line_slope : ℝ) : ℝ × ℝ :=
  let denom := 1 + line_slope^2 in
  ((p.1 + line_slope * (p.2 - line_slope * p.1)) / denom,
   (line_slope * (p.1 + line_slope * p.2) - line_slope^2 * p.1 + p.2) / denom)

theorem vector_ab_is_correct :
  let A := (4, -3)
  let u := 6
  let v := 8
  distance A (A.1 + u, A.2 + v) = 10 ∧
  dot_product (u, v) (A.1, A.2) = 0 := by
sorry

theorem circle_symmetry :
  let center := reflect_point (3, -1) (1 / 2)
  let radius := real.sqrt 10
  ∃ c : ℝ × ℝ, c = center ∧
    (∀ x y, ((x - c.1)^2 + (y - c.2)^2 = radius^2)) := by
sorry

end vector_ab_is_correct_circle_symmetry_l244_244085


namespace shaded_region_probability_l244_244374

def regular_hexagon := Type
def equilateral_triangle := Type
def game_board := regular_hexagon × fin 6 equilateral_triangle

def shaded (t : equilateral_triangle) : Prop := sorry
def is_shaded_region (b : game_board) : bool :=
  match b with
  | (_, ⟨i, _⟩) => i < 2 -- assuming the first two are shaded

theorem shaded_region_probability (b : game_board) : 
  (∃ i, i < 2 → is_shaded_region b = true) → 
  (prob_shaded : ℚ := 2 / 7) :=
sorry

end shaded_region_probability_l244_244374


namespace train_cross_time_l244_244818

noncomputable def speed_kmh := 72
noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def length_train := 180
noncomputable def length_bridge := 270
noncomputable def total_distance := length_train + length_bridge
noncomputable def time_to_cross := total_distance / speed_mps

theorem train_cross_time :
  time_to_cross = 22.5 := 
sorry

end train_cross_time_l244_244818


namespace total_stuffed_animals_l244_244638

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l244_244638


namespace solve_inequality_l244_244616

variable {f : ℝ → ℝ}
variable {x : ℝ}

-- Condition 1: f is an odd function on [-1,1]
def is_odd_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x ∈ Icc a b, f (-x) = -f(x)

-- Condition 2: f is increasing on [-1,1]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop := 
  ∀ x y ∈ Icc a b, x ≤ y → f x ≤ f y

theorem solve_inequality 
  (h_odd : is_odd_on_interval f (-1) 1) 
  (h_increasing : is_increasing_on_interval f (-1) 1) : 
  (f (x - 1 / 2) + f (1 / 4 - x) < 0) ↔ (-1 / 2 ≤ x ∧ x ≤ 1) :=
by
  sorry

end solve_inequality_l244_244616


namespace calculate_otimes_l244_244848

def otimes (x y : ℝ) : ℝ := x^3 - y^2 + x

theorem calculate_otimes (k : ℝ) : 
  otimes k (otimes k k) = -k^6 + 2*k^5 - 3*k^4 + 3*k^3 - k^2 + 2*k := by
  sorry

end calculate_otimes_l244_244848


namespace reconstruct_right_triangle_l244_244042

theorem reconstruct_right_triangle (c d : ℝ) (hc : c > 0) (hd : d > 0) :
  ∃ A X Y: ℝ, (A ≠ X ∧ A ≠ Y ∧ X ≠ Y) ∧ 
  -- Right triangle with hypotenuse c
  (A - X) ^ 2 + (Y - X) ^ 2 = c ^ 2 ∧ 
  -- Difference of legs is d
  ∃ AY XY: ℝ, ((AY = abs (A - Y)) ∧ (XY = abs (Y - X)) ∧ (abs (AY - XY) = d)) := 
by
  sorry

end reconstruct_right_triangle_l244_244042


namespace chord_midpoint_line_eqn_l244_244588

-- Definitions of points and the ellipse condition
def P : ℝ × ℝ := (3, 2)

def is_midpoint (P E F : ℝ × ℝ) := 
  P.1 = (E.1 + F.1) / 2 ∧ P.2 = (E.2 + F.2) / 2

def ellipse (x y : ℝ) := 
  4 * x^2 + 9 * y^2 = 144

theorem chord_midpoint_line_eqn
  (E F : ℝ × ℝ) 
  (h1 : is_midpoint P E F)
  (h2 : ellipse E.1 E.2)
  (h3 : ellipse F.1 F.2):
  ∃ (m b : ℝ), (P.2 = m * P.1 + b) ∧ (2 * P.1 + 3 * P.2 - 12 = 0) :=
by 
  sorry

end chord_midpoint_line_eqn_l244_244588


namespace major_airlines_free_snacks_l244_244780

variable (S : ℝ)

theorem major_airlines_free_snacks (h1 : 0.5 ≤ 1) (h2 : 0.5 = 1) :
  0.5 ≤ S :=
sorry

end major_airlines_free_snacks_l244_244780


namespace cups_of_flour_already_put_in_l244_244262

-- Define the relevant constants for the problem
def total_flour_needed : ℕ := 10
def additional_flour_needed := (sugar_needed : ℕ := 3) + 5

-- Define a proof statement for how many cups of flour Mary has already put in
theorem cups_of_flour_already_put_in :
  total_flour_needed - additional_flour_needed = 2 := by
sorry

end cups_of_flour_already_put_in_l244_244262


namespace garden_perimeter_l244_244319

theorem garden_perimeter :
  ∀ (posts : ℕ) (post_width_inch : ℕ) (gap_feet : ℝ),
    posts = 36 →
    post_width_inch = 3 →
    gap_feet = 4 →
    (let post_width_feet := post_width_inch / 12 in
     let num_posts_per_side := posts / 4 in
     let total_post_width_per_side := num_posts_per_side * post_width_feet in
     let total_gaps_per_side := (num_posts_per_side - 1) * gap_feet in
     let side_length := total_post_width_per_side + total_gaps_per_side in
     let perimeter := 4 * side_length in
     perimeter = 137) :=
by { sorry }

end garden_perimeter_l244_244319


namespace parallel_vectors_l244_244051

theorem parallel_vectors (m : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2, 1)) (h₂ : b = (1, m))
  (h₃ : ∃ k : ℝ, b = k • a) : m = 1 / 2 :=
by 
  sorry

end parallel_vectors_l244_244051


namespace NaturalNumber_with_10_Divisors_Prime_2_3_l244_244378

theorem NaturalNumber_with_10_Divisors_Prime_2_3 (a b : ℕ) (N : ℕ) (h : (a + 1) * (b + 1) = 10) (h1 : N = 2^a * 3^b) :
  N = 162 ∨ N = 48 :=
begin
  sorry
end

end NaturalNumber_with_10_Divisors_Prime_2_3_l244_244378


namespace find_second_remainder_l244_244739

theorem find_second_remainder (k m n r : ℕ) 
  (h1 : n = 12 * k + 56) 
  (h2 : n = 34 * m + r) 
  (h3 : (22 + r) % 12 = 10) : 
  r = 10 :=
sorry

end find_second_remainder_l244_244739


namespace grid_midpoint_exists_l244_244278

theorem grid_midpoint_exists (points : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ (points i).fst % 2 = (points j).fst % 2 ∧ (points i).snd % 2 = (points j).snd % 2 :=
by 
  sorry

end grid_midpoint_exists_l244_244278


namespace figure_is_regular_polygon_l244_244802

-- Defining the conditions
def equiangular (P : Type) [has_angles P] : Prop :=
∀ (a1 a2 : angle P), a1 = a2

def equilateral (P : Type) [has_sides P] : Prop :=
∀ (s1 s2 : side P), s1 = s2

-- Defining a regular polygon
structure is_regular_polygon (P : Type) [has_angles P] [has_sides P] : Prop :=
(angles_eq : equiangular P)
(sides_eq : equilateral P)

-- The theorem statement
theorem figure_is_regular_polygon (P : Type) [has_angles P] [has_sides P] 
  (h1 : equiangular P) (h2 : equilateral P) : is_regular_polygon P :=
by {
  sorry
}

end figure_is_regular_polygon_l244_244802


namespace length_relation_concyclic_points_l244_244831

variables {m n b : ℝ}
variables {A B C D E F : ℝ × ℝ}
variables {AB CD EF : ℝ}

-- Assumptions
-- m and n are positive, and m ≠ n
axiom m_pos : m > 0
axiom n_pos : n > 0
axiom m_ne_n : m ≠ n
-- AB is a chord with slope 1
axiom AB_is_chord : B.2 - A.2 = B.1 - A.1 
-- E and F are midpoints of AB and CD respectively
axiom E_mid_AB : E = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
axiom F_mid_CD : F = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) 

noncomputable def AB_length : ℝ := real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def CD_length : ℝ := real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)

noncomputable def EF_length : ℝ := real.sqrt ((F.1 - E.1)^2 + (F.2 - E.2)^2)

theorem length_relation :
  (CD_length) ^ 2 - (AB_length) ^ 2 = 4 * (EF_length) ^ 2 := 
by { sorry }

theorem concyclic_points :
  ∃ (O : ℝ × ℝ) (r : ℝ), ∀ (X : ℝ × ℝ), X = A ∨ X = B ∨ X = C ∨ X = D → (X.1 - O.1)^2 + (X.2 - O.2)^2 = r^2 :=
by { sorry }

end length_relation_concyclic_points_l244_244831


namespace binomial_coefficient_middle_term_l244_244580

theorem binomial_coefficient_middle_term :
  let n := 11
  let sum_odd := 1024
  sum_odd = 2^(n-1) →
  let binom_coef := Nat.choose n (n / 2 - 1)
  binom_coef = 462 :=
by
  intro n
  let n := 11
  intro sum_odd
  let sum_odd := 1024
  intro h
  let binom_coef := Nat.choose n (n / 2 - 1)
  have : binom_coef = 462 := sorry
  exact this

end binomial_coefficient_middle_term_l244_244580


namespace snail_paths_count_l244_244814

theorem snail_paths_count (n : ℕ) : 
  let C (a b : ℕ) : ℕ := Nat.choose a b in
  (C (2 * n) n) ^ 2 = (Nat.choose (2 * n) n) ^ 2 := 
by
  sorry

end snail_paths_count_l244_244814


namespace total_money_received_a_l244_244352

-- Define the partners and their capitals
structure Partner :=
  (name : String)
  (capital : ℕ)
  (isWorking : Bool)

def a : Partner := { name := "a", capital := 3500, isWorking := true }
def b : Partner := { name := "b", capital := 2500, isWorking := false }

-- Define the total profit
def totalProfit : ℕ := 9600

-- Define the managing fee as 10% of total profit
def managingFee (total : ℕ) : ℕ := (10 * total) / 100

-- Define the remaining profit after deducting the managing fee
def remainingProfit (total : ℕ) (fee : ℕ) : ℕ := total - fee

-- Calculate the share of remaining profit based on capital contribution
def share (capital totalCapital remaining : ℕ) : ℕ := (capital * remaining) / totalCapital

-- Theorem to prove the total money received by partner a
theorem total_money_received_a :
  let totalCapitals := a.capital + b.capital
  let fee := managingFee totalProfit
  let remaining := remainingProfit totalProfit fee
  let aShare := share a.capital totalCapitals remaining
  (fee + aShare) = 6000 :=
by
  sorry

end total_money_received_a_l244_244352


namespace parabola_conditions_l244_244242

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244242


namespace point_opposite_sides_l244_244918

theorem point_opposite_sides (x_0 y_0 : ℝ) : 
  (3 * x_0 + 2 * y_0 - 8) * (3 * 1 + 2 * 2 - 8) < 0 → 3 * x_0 + 2 * y_0 > 0 :=
by 
  intro h,
  have hA : 3 * 1 + 2 * 2 - 8 = -1, by norm_num,
  rw hA at h,
  exact neg_lt_zero.1 ((mul_neg_iff.mp h).resolve_right zero_ne_neg_one.symm),
  sorry

end point_opposite_sides_l244_244918


namespace jude_age_today_l244_244562
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end jude_age_today_l244_244562


namespace player_b_championship_l244_244103

/-- Probability of Player B winning the championship in a best-of-five game system -/
theorem player_b_championship:
  let pB := (1 / 3 : ℝ) in
  let pA := (2 / 3 : ℝ) in
  let prob_3_wins := (5.choose 3) * (pB^3) * (pA^2) in
  let prob_4_wins := (5.choose 4) * (pB^4) * (pA) in
  let prob_5_wins := (pB^5) in
  prob_3_wins + prob_4_wins + prob_5_wins = 17 / 81 :=
sorry

end player_b_championship_l244_244103


namespace ellipse_minor_axis_length_l244_244817

open Real

theorem ellipse_minor_axis_length :
  ∃ (c : ℝ × ℝ) (a b : ℝ), 
    a > 0 ∧ b > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ {(0,0), (0,4), (2,0), (2,4), (-1,2)} → 
     ((x - c.1)^2 / a^2 + (y - c.2)^2 / b^2 = 1)) ∧
    b = 4/sqrt(3) ∧ 2 * b = 8 * sqrt(3) / 3
    :=
sorry

end ellipse_minor_axis_length_l244_244817


namespace ratio_squared_cross_section_face_l244_244096

theorem ratio_squared_cross_section_face (a : ℝ) :
  let A := (0, 0, 0)
  let B := (a, 0, 0)
  let G := (a, a, a)
  let H := (0, a, a)
  let K := (a / 2, 0, 0)
  let L := (a, a, a / 2)
  let face_area := a ^ 2
  let AG := sqrt ((a - 0) ^ 2 + (a - 0) ^ 2 + (a - 0) ^ 2)
  let KL := sqrt ((a - a / 2) ^ 2 + a ^ 2 + (a / 2 - 0) ^ 2)
  let cross_section_area := (1 / 2) * AG * KL
  let R := cross_section_area / face_area
  in R ^ 2 = 9 / 8 := 
by 
  sorry

end ratio_squared_cross_section_face_l244_244096


namespace sequence_formula_sum_formula_l244_244899

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 1 else 2 * n - 1

noncomputable def b (n : ℕ) : ℕ :=
  (2 * n - 1) * 2^n

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  finset.sum (finset.range (n+1)) a

def T (b : ℕ → ℕ) (n : ℕ) : ℕ :=
  finset.sum (finset.range n) b

theorem sequence_formula (n : ℕ) (h : n = 1 ∨ n > 1) : a n = 2 * n - 1 := 
  sorry 

theorem sum_formula (n : ℕ) : 
  T b n = (2 * n - 3) * 2^(n + 1) + 6 := 
  sorry

end sequence_formula_sum_formula_l244_244899


namespace proof_methods_l244_244120

-- Definitions based on the conditions given:
def synthetic_method : Prop := λ known conclusion, conclusion is derived from known
def analytic_method : Prop := λ conclusion known, known is derived from conclusion

-- Statements based on the problem statements:
def statement_1 : Prop := synthetic_method -> reverse reasoning method of seeking cause from result
def statement_2 : Prop := analytic_method -> reason from conclusion to the known
def statement_3 : Prop := ∀ problem, when proving problem, analytic_method and synthetic_method have exactly opposite thought processes and inverse procedures

-- The theorem we intend to prove:
theorem proof_methods :
  ¬ statement_1 ∧ ¬ statement_2 ∧ statement_3 :=
by
  -- Statements are assumed as incorrect or correct based on the solution; proof is a placeholder.
  sorry

end proof_methods_l244_244120


namespace tangent_line_equation_decreasing_intervals_sin_lt_polynomial_l244_244502

-- Definitions based on conditions
def f (x : ℝ) := Real.sin x
def g (x : ℝ) (m : ℝ) := m * x - x ^ 3 / 6

-- Statement for part (1)
theorem tangent_line_equation (x : ℝ) :
  x = π / 4 → f x = Real.sin (π / 4) → 
  (y = f x + (Real.cos (π / 4)) * (x - π / 4) → y = ( √2 / 2 ) * x + (1 / 2) - ( √2 * π / 8)) :=
sorry

-- Statement for part (2)
theorem decreasing_intervals (m : ℝ) :
  monotone_decreasing ((λ x, g x m) : ℝ → ℝ) ((λ x, (-∞, -sqrt (2 * m)] ∪ [sqrt (2 * m), ∞))) := 
sorry

-- Statement for part (3)
theorem sin_lt_polynomial (x : ℝ) (h : x > 0) :
  f x < x - x ^ 3 / 6 + x ^ 3 / 6 :=
sorry

end tangent_line_equation_decreasing_intervals_sin_lt_polynomial_l244_244502


namespace xsquared_plus_5x_minus_6_condition_l244_244297

theorem xsquared_plus_5x_minus_6_condition (x : ℝ) : 
  (x^2 + 5 * x - 6 > 0) → (x > 2) ∨ (((x > 1) ∨ (x < -6)) ∧ ¬(x > 2)) := 
sorry

end xsquared_plus_5x_minus_6_condition_l244_244297


namespace apollonius_circle_center_l244_244407

theorem apollonius_circle_center :
  ∃ (x y : ℝ), (∃ (M : ℝ × ℝ),
    let O := (0, 0)
    let A := (3, 0)
    ∀ (λ : ℝ), λ = 1/2 → 
      (dist M O / dist M A = λ) ∧ 
      M = (x, y)) ∧ (x = -1 ∧ y = 0) :=
sorry

end apollonius_circle_center_l244_244407


namespace cube_odd_minus_itself_div_by_24_l244_244656

theorem cube_odd_minus_itself_div_by_24 (n : ℤ) : 
  (2 * n + 1)^3 - (2 * n + 1) ≡ 0 [MOD 24] := 
by 
  sorry

end cube_odd_minus_itself_div_by_24_l244_244656


namespace parabola_p_and_circle_tangent_directrix_l244_244232

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244232


namespace fourth_power_sum_l244_244062

variable (a b c : ℝ)

theorem fourth_power_sum (h1 : a + b + c = 2) 
                         (h2 : a^2 + b^2 + c^2 = 3) 
                         (h3 : a^3 + b^3 + c^3 = 4) : 
                         a^4 + b^4 + c^4 = 41 / 6 := 
by 
  sorry

end fourth_power_sum_l244_244062


namespace relationship_between_abc_l244_244955

variable (a b c : ℝ)
variable (e : ℝ) (ln : ℝ → ℝ)
variable [RealExp e ln]

theorem relationship_between_abc (e_pos : e > 2.72)
  (h_increase : ∀ x > 0, e ^ x > x + 1)
  (a_def : a = 0.6 * e ^ 0.4)
  (b_def : b = 2 - ln 4)
  (c_def : c = e - 2) : a > c ∧ c > b :=
by
  sorry

end relationship_between_abc_l244_244955


namespace find_lambda_l244_244492

variables {V : Type*} [inner_product_space ℝ V]

def angle (u v : V) : ℝ := real.arccos ((⟪u, v⟫) / ((∥u∥) * (∥v∥)))

theorem find_lambda
  (A B C P : V)
  (h_angle: angle (B - A) (C - A) = real.pi * 2 / 3)
  (h_AB_norm: ∥B - A∥ = 2)
  (h_AC_norm: ∥C - A∥ = 2)
  (h_AP: ∃ λ : ℝ, P - A = λ • (B - A) + (C - A))
  (h_perp: ⟪P - A, C - B⟫ = 0) :
  ∃ λ : ℝ, λ = 1 :=
sorry

end find_lambda_l244_244492


namespace angle_degree_measure_l244_244733

theorem angle_degree_measure (x : ℝ) (h1 : (x + (90 - x) = 90)) (h2 : (x = 3 * (90 - x))) : x = 67.5 := by
  sorry

end angle_degree_measure_l244_244733


namespace nodes_inside_triangle_l244_244260

theorem nodes_inside_triangle (A B C : ℤ × ℤ) (h_vert : ∀ P, P = A ∨ P = B ∨ P = C)
  (h_nodes : ∃ P Q : ℤ × ℤ, P ≠ Q ∧ inside_triangle A B C P ∧ inside_triangle A B C Q) :
  ∃ P Q : ℤ × ℤ, (P ≠ Q ∧ inside_triangle A B C P ∧ inside_triangle A B C Q) ∧ 
  ((line_through P Q ∩ {A, B, C}).nonempty ∨ 
  (parallel_to_one_side A B C P Q)) := sorry

def inside_triangle (A B C P : ℤ × ℤ) : Prop := sorry
def line_through (P Q : ℤ × ℤ) : (ℤ × ℤ) → Prop := sorry
def parallel_to_one_side (A B C P Q : ℤ × ℤ) : Prop := sorry

end nodes_inside_triangle_l244_244260


namespace age_ratio_l244_244599

/-- 
Axiom: Kareem's age is 42 and his son's age is 14. 
-/
axiom Kareem_age : ℕ
axiom Son_age : ℕ

/-- 
Conditions: 
  - Kareem's age after 10 years plus his son's age after 10 years equals 76.
  - Kareem's current age is 42.
  - His son's current age is 14.
-/
axiom age_condition : Kareem_age + 10 + Son_age + 10 = 76
axiom Kareem_current_age : Kareem_age = 42
axiom Son_current_age : Son_age = 14

/-- 
Theorem: The ratio of Kareem's age to his son's age is 3:1.
-/
theorem age_ratio : Kareem_age / Son_age = 3 / 1 := by {
  -- Proof skipped
  sorry 
}

end age_ratio_l244_244599


namespace tetrahedron_volume_0_l244_244880

def point := ℝ × ℝ × ℝ

def coplanar (p1 p2 p3 p4: point): Prop :=
  ∃ (a b c d: ℝ), ∀ (p: point), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 → (a * p.1 + b * p.2 + c * p.3 = d)

def x_plus_y_eq_z_plus_3_plane (p: point): Prop := 
  p.1 + p.2 = p.3 + 3

def volume_of_tetrahedron_is_zero (p1 p2 p3 p4: point): Prop :=
  ∀ {a b c d: ℝ}, (∀ (p: point), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 → (a * p.1 + b * p.2 + c * p.3 = d)) → 0 = 0

theorem tetrahedron_volume_0 : 
  volume_of_tetrahedron_is_zero (5, 8, 10) (10, 10, 17) (4, 45, 46) (2, 5, 4) :=
begin
  sorry
end

end tetrahedron_volume_0_l244_244880


namespace cannot_reduce_box_dimension_l244_244789

theorem cannot_reduce_box_dimension :
  ∀ (box dimension1 dim2 dim3 : ℝ) (p1 p2 : ℝ×ℝ×ℝ),
    (box = (2, 2, 3)) → 
    (p1 = (1, 2, 3) ∧ p2 = (1, 2, 3)) →
    (p1.2.2 < 3 ∧ p2.2.1 < 2) →
    ¬(∃ reduced_dimension : ℝ, reduced_dimension < 2 ∨ reduced_dimension < 3) :=
by
  sorry

end cannot_reduce_box_dimension_l244_244789


namespace intersection_x_coords_equal_l244_244014

noncomputable def ellipse := 
  {a b c : ℝ // a > 0 ∧ b > 0 ∧ c > 0 ∧ (x, y : ℝ) // 
    \frac{x^2}{a^2} + \frac{y^2}{b^2} = 1}

theorem intersection_x_coords_equal (a b c : ℝ) 
(h1 : a > 0)
(h2 : b > 0)
(h3 : ∀ x y, x = 1 → y = (sqrt 3) / 2 → (c, 0) ∈ (1, (sqrt 3) / 2))
(h4 : a = 1) 
(h5 : c = 2) 
(m x1 y1 x2 y2: ℝ) :
(∃ A B : ellipse a b c,
 (A = (1, 0) → 
  B = (4my1y2 - 2y1 + 6y2) / (y1 + 3y2) →
  (∃ M N : ellipse a b c,
    M ≠ A ∧ 
    N ≠ B →
    let m := a^2 * (x1 + c) in
    let n := a * ((x2 - c) )  in
    A = M ∧ B = N)) :=
sorry

end intersection_x_coords_equal_l244_244014


namespace prove_p_equals_2_l244_244202

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244202


namespace smallest_value_t_for_circle_encompassment_l244_244692

theorem smallest_value_t_for_circle_encompassment :
  ∀ t : ℝ, (∀ θ : ℝ, (0 ≤ θ ∧ θ ≤ t → exists r : ℝ, r = sin θ ∧ (r * cos θ, r * sin θ) = (0, 0))) ↔ t = π := sorry

end smallest_value_t_for_circle_encompassment_l244_244692


namespace p_eq_two_circle_tangent_proof_l244_244215

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244215


namespace g_4_minus_g_7_l244_244691

theorem g_4_minus_g_7 (g : ℝ → ℝ) (h_linear : ∀ x y : ℝ, g (x + y) = g x + g y)
  (h_diff : ∀ k : ℝ, g (k + 1) - g k = 5) : g 4 - g 7 = -15 :=
by
  sorry

end g_4_minus_g_7_l244_244691


namespace circumscribed_circles_tangent_l244_244314

-- Given: BL is the angle bisector of triangle ABC
-- and the perpendicular bisector of BL intersects the external angle bisectors of ∠A and ∠C at points P and Q respectively
-- Prove: The circle circumscribed around the triangle PBQ is tangent to the circle circumscribed around the triangle ABC

theorem circumscribed_circles_tangent
  (A B C L P Q : Point)
  (h1 : isAngleBisector B L A C)
  (h2 : perpendicularBisector B L intersection (externalAngleBisector A) = P)
  (h3 : perpendicularBisector B L intersection (externalAngleBisector C) = Q)
  : tangent (circumscribed_circle ABC) (circumscribed_circle PBQ) :=
sorry

end circumscribed_circles_tangent_l244_244314


namespace second_derivative_equality_l244_244762

variable (t : ℝ)
variable (x : ℝ) (y : ℝ)

-- Given parametric equations
def parametric_x (t : ℝ) : ℝ := real.sqrt (t^3 - 1)
def parametric_y (t : ℝ) : ℝ := real.log t

-- Second derivative y'' with respect to x should be:
def second_derivative_y_x (t : ℝ) : ℝ := (2 * (2 - t^3)) / (3 * t^6)

theorem second_derivative_equality :
  ∀ (t : ℝ), second_derivative_y_x t = ((2 * (2 - t^3)) / (3 * t^6)) :=
by
  intro t
  sorry

end second_derivative_equality_l244_244762


namespace chord_length_intersection_l244_244550

theorem chord_length_intersection (m : ℝ) (h₀ : m > 0) :
  (∀ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 3 → x - y + m = 0) →
  m = 2 :=
begin
  sorry
end

end chord_length_intersection_l244_244550


namespace parabola_focus_line_tangent_circle_l244_244205

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244205


namespace sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l244_244587

open Real

namespace TriangleProofs

variables 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (BA BC : ℝ) 
  (h1 : sin B = sqrt 7 / 4) 
  (h2 : (cos A / sin A + cos C / sin C = 4 * sqrt 7 / 7)) 
  (h3 : BA * BC = 3 / 2)
  (h4 : a = b ∧ c = b)

-- 1. Prove that sin A * sin C = sin^2 B
theorem sin_a_mul_sin_c_eq_sin_sq_b : sin A * sin C = sin B ^ 2 := 
by sorry

-- 2. Prove that 0 < B ≤ π / 3
theorem zero_lt_B_le_pi_div_3 : 0 < B ∧ B ≤ π / 3 := 
by sorry

-- 3. Find the magnitude of the vector sum.
theorem magnitude_BC_add_BA : abs (BC + BA) = 2 * sqrt 2 := 
by sorry

end TriangleProofs

end sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l244_244587


namespace ab_plus_cd_111_333_l244_244614

theorem ab_plus_cd_111_333 (a b c d : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : a + b + d = 5) 
  (h3 : a + c + d = 20) 
  (h4 : b + c + d = 15) : 
  a * b + c * d = 111.333 := 
by
  sorry

end ab_plus_cd_111_333_l244_244614


namespace count_two_digit_primes_l244_244527

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_digits : set ℕ := {3, 5, 7, 9}

def two_digit_primes := { n | ∃ (a b : ℕ), a ∈ valid_digits ∧ b ∈ valid_digits ∧ a ≠ b ∧ n = 10 * a + b ∧ is_prime n }

theorem count_two_digit_primes : (two_digit_primes : set ℕ).card = 7 := 
  sorry

end count_two_digit_primes_l244_244527


namespace average_age_of_team_l244_244681

theorem average_age_of_team
    (A : ℝ)
    (captain_age : ℝ)
    (wicket_keeper_age : ℝ)
    (bowlers_count : ℝ)
    (batsmen_count : ℝ)
    (team_members_count : ℝ)
    (avg_bowlers_age : ℝ)
    (avg_batsmen_age : ℝ)
    (total_age_team : ℝ) :
    captain_age = 28 →
    wicket_keeper_age = 31 →
    bowlers_count = 5 →
    batsmen_count = 4 →
    avg_bowlers_age = A - 2 →
    avg_batsmen_age = A + 3 →
    total_age_team = 28 + 31 + 5 * (A - 2) + 4 * (A + 3) →
    team_members_count * A = total_age_team →
    team_members_count = 11 →
    A = 30.5 :=
by
  intros
  sorry

end average_age_of_team_l244_244681


namespace triangle_height_l244_244435

theorem triangle_height {a b c : ℕ} (ha : a = 15) (hb : b = 14) (hc : c = 13) :
  height_of_triangle_approx (a b c) ≈ 11.2 :=
sorry

end triangle_height_l244_244435


namespace radius_of_shorter_container_l244_244324

variable (V h r : ℝ)
variable (volume_eq : π * 8^2 * 4 * h = π * r^2 * h)
variable (h_pos : h ≠ 0)

theorem radius_of_shorter_container : r = 16 := by
  have eq1 : 256 * h = r^2 * h := by
    -- Given condition
    exact volume_eq
  have eq2 : 256 = r^2 := by
    -- Divide both sides by h (h is not 0)
    rwa [mul_right_inj' h_pos] at eq1
  have eq3 : r = real.sqrt 256 := by
    -- sqrt both sides
    rwa [real.sqrt_eq_iff_sq_eq, real.lt_of_le_of_ne (le_of_lt (mul_pos (by norm_num) h_pos))] at eq2
  -- Simplifying the square root
  rwa [real.sqrt_eq_iff_sq_eq, real.sqrt_eq_rfl] at eq3
  sorry

end radius_of_shorter_container_l244_244324


namespace inverse_value_l244_244629

-- Declare the function f and its properties
variables (f : ℝ → ℝ)

-- Assume the conditions
axiom odd : ∀ x : ℝ, f (-x) = -f x
axiom has_inverse : Function.Bijective f
axiom value_at_4 : f 4 = 2

-- Prove that f⁻¹(-2) = -4
theorem inverse_value : Function.LeftInverse f.symm f → f⁻¹ (-2) = -4 :=
by
  intro h
  have h1 : f (-4) = -2 := calc
    f (-4) = -f 4 : odd 4
        ... = -2   : value_at_4
  have h2 : f.symm (-2) = -4 := Function.LeftInverse.symm_apply_apply (Function.LeftInverse.symm h) h1
  exact h2

end inverse_value_l244_244629


namespace total_female_officers_on_police_force_l244_244648

theorem total_female_officers_on_police_force (total_officers_on_duty half_female: ℕ) (percentage_on_duty: ℝ) (h1: total_officers_on_duty = 100) (h2: half_female = total_officers_on_duty / 2) (h3: percentage_on_duty = 0.20): 
  (total_female_officers: ℕ) (h4: half_female = (percentage_on_duty * total_female_officers)) :
  total_female_officers = 250 :=
by
  sorry

end total_female_officers_on_police_force_l244_244648


namespace geometric_sequence_q_cubed_l244_244683

theorem geometric_sequence_q_cubed (q a_1 : ℝ) (h1 : q ≠ 0) (h2 : q ≠ 1) 
(h3 : 2 * (a_1 * (1 - q^9) / (1 - q)) = (a_1 * (1 - q^3) / (1 - q)) + (a_1 * (1 - q^6) / (1 - q))) : 
  q^3 = -1/2 := by
  sorry

end geometric_sequence_q_cubed_l244_244683


namespace impossible_closed_chain_1997_tiles_l244_244841

/- 
Problem Statement:
Given 1997 square tiles placed sequentially on an infinite checkerboard grid such that:
1. Each tile covers one cell,
2. Tiles are numbered from 1 to 1997,
3. Adjacent cells on the checkerboard are of different colors,
4. Tiles with odd numbers land on cells of one color, and tiles with even numbers land on cells of the opposite color,
prove that forming a closed chain with these 1997 tiles is impossible.
-/

theorem impossible_closed_chain_1997_tiles :
  ∀ (n : ℕ) (chain : Fin n → ℕ), n = 1997 →
  (∀ i : Fin (n - 1), adjacent_tiles (chain i) (chain (i + 1))) →
  adjacent_tiles (chain 0) (chain (Fin.ofNat (n - 1))) →
  (∀ i : Fin n, chain i % 2 = i % 2) →
  false :=
by
  intros n chain h1 h2 h3 h4
  sorry


end impossible_closed_chain_1997_tiles_l244_244841


namespace standard_equation_of_ellipse_value_of_k_l244_244482

theorem standard_equation_of_ellipse :
  ∃ a b: ℝ, a > 0 ∧ b > 0 ∧ ∃ x y : ℝ, x = 1 ∧ y = 3/2 ∧ 
    (\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1) ∧ 
    (PF_1 + PF_2 = 4) ∧ 
    (a = 2 ∧ b = sqrt 3 ∧ \frac{x^2}{4} + \frac{y^2}{3} = 1) := 
sorry

theorem value_of_k :
  ∃ k: ℝ, ∃ m: ℝ, ∃ A B: (ℝ × ℝ), k ≠ 0 ∧ m^2 = 1 + k^2 ∧ 
    A ≠ B ∧ tangent A B ∧ (\overrightarrow{OA} · \overrightarrow{OB} = -3/2) ∧ 
    (k = (sqrt 2) / 2 ∨ k = -(sqrt 2) / 2) := 
sorry

end standard_equation_of_ellipse_value_of_k_l244_244482


namespace sequence_2011_l244_244995

theorem sequence_2011 :
  ∀ (a : ℕ → ℤ), (a 1 = 1) →
                  (a 2 = 2) →
                  (∀ n : ℕ, a (n + 2) = a (n + 1) - a n) →
                  a 2011 = 1 :=
by {
  -- Insert proof here
  sorry
}

end sequence_2011_l244_244995


namespace proof_problem_l244_244149

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244149


namespace exist_odd_a_b_k_l244_244905

theorem exist_odd_a_b_k (m : ℤ) : 
  ∃ (a b k : ℤ), (a % 2 = 1) ∧ (b % 2 = 1) ∧ (k ≥ 0) ∧ (2 * m = a^19 + b^99 + k * 2^1999) :=
by {
  sorry
}

end exist_odd_a_b_k_l244_244905


namespace max_distance_travel_l244_244001

theorem max_distance_travel (front_tire_lifespan rear_tire_lifespan : ℕ) (h_front : front_tire_lifespan = 24000) (h_rear : rear_tire_lifespan = 36000) :
  ∃ max_distance : ℕ, max_distance = 28800 :=
begin
  use 28800,
  sorry
end

end max_distance_travel_l244_244001


namespace domain_of_sqrt_log10_theorem_l244_244447

def domain_of_sqrt_log10 (x : ℝ) : Prop :=
  x + 2 > 0 ∧ log 10 (x + 2) ≥ 0

theorem domain_of_sqrt_log10_theorem : 
  ∀ x : ℝ, (domain_of_sqrt_log10 x) ↔ x > -1 :=
by
  sorry

end domain_of_sqrt_log10_theorem_l244_244447


namespace present_ages_l244_244698

theorem present_ages
  (R D K : ℕ) (x : ℕ)
  (H1 : R = 4 * x)
  (H2 : D = 3 * x)
  (H3 : K = 5 * x)
  (H4 : R + 6 = 26)
  (H5 : (R + 8) + (D + 8) = K) :
  D = 15 ∧ K = 51 :=
sorry

end present_ages_l244_244698


namespace f_increasing_on_0_to_5pi_over_6_sin_theta_value_l244_244038

-- Define the function f(x)
def f (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

-- Question 1: Prove that f(x) is increasing on (0, 5π/6)
theorem f_increasing_on_0_to_5pi_over_6 : 
  ∃ I : set ℝ, I = set.Ioo 0 (5 * Real.pi / 6) ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f(x) < f(y) :=
sorry

-- Question 2: Prove the value of sin θ given f(θ) = -6/5 and 0 < θ < π
theorem sin_theta_value :
  ∀ θ : ℝ, 0 < θ ∧ θ < Real.pi ∧ f(θ) = -6/5 → sin θ = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end f_increasing_on_0_to_5pi_over_6_sin_theta_value_l244_244038


namespace p_eq_two_circle_tangent_proof_l244_244220

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244220


namespace probability_of_multiples_of_3_or_5_l244_244325

open Finset

def is_multiple_of (a b: ℕ) : Prop := b % a == 0

def multiples_of_3_or_5 : Finset ℕ := (range 31).filter (λ n, is_multiple_of 3 n ∨ is_multiple_of 5 n)

theorem probability_of_multiples_of_3_or_5 : 
  (Finset.card (multiples_of_3_or_5).choose 2) / (Finset.card (range 31).choose 2) = (13 : ℚ) / 63 := 
by
  sorry

end probability_of_multiples_of_3_or_5_l244_244325


namespace prove_p_equals_2_l244_244197

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244197


namespace parabola_properties_l244_244189

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244189


namespace find_least_number_subtracted_l244_244874

theorem find_least_number_subtracted (n m : ℕ) (h : n = 78721) (h1 : m = 23) : (n % m) = 15 := by
  sorry

end find_least_number_subtracted_l244_244874


namespace cos_30_deg_l244_244341

-- The condition implicitly includes the definition of cosine and the specific angle value

theorem cos_30_deg : cos (Real.pi / 6) = Real.sqrt 3 / 2 :=
by sorry

end cos_30_deg_l244_244341


namespace correct_conclusions_l244_244106

-- Definition of conditions and conclusions

def condition1 (a b c : ℝ) := a + c = 2 * b

def condition2 (c : ℝ) (b : ℝ) (B : ℝ) := c = 4 ∧ b = 2 * Real.sqrt 3 ∧ B = Real.pi / 6

def condition3 (B : ℝ) (b : ℝ) (ac : ℝ) := B = Real.pi / 6 ∧ b = 1 ∧ ac = 2 * Real.sqrt 3

def condition4 (a b c A B : ℝ) := (2 * c - b) * Real.cos A = a * Real.cos B

-- Correct conclusions
def conclusion2 (c : ℝ) (b : ℝ) (B : ℝ) := c = 4 ∧ b = 2 * Real.sqrt 3 ∧ B = Real.pi / 6 → some_solution_has_two_solutions

def conclusion3 (B : ℝ) (b : ℝ) (ac : ℝ) (a c : ℝ) := B = Real.pi / 6 ∧ b = 1 ∧ ac = 2 * Real.sqrt 3 ∧ a + c = 2 + Real.sqrt 3

-- The main theorem statement combining conditions and conclusions
theorem correct_conclusions
(a b c A B : ℝ)
(ac : ℝ) : 
  (condition1 a b c ∧ condition2 c b B ∧ condition3 B b ac ∧ condition4 a b c A B) →
  (conclusion2 c b B ∧ conclusion3 B b ac a c) :=
by sorry

end correct_conclusions_l244_244106


namespace product_primes_less_than_20_l244_244335

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end product_primes_less_than_20_l244_244335


namespace arun_cross_train_b_time_l244_244326

theorem arun_cross_train_b_time :
  ∀ (len_A len_B : ℕ) (speed_A speed_B : ℝ),
  len_A = 250 → len_B = 300 →
  speed_A = 20 → speed_B = 13.33 →
  (len_A + len_B) / (speed_A + speed_B) = 16.5 :=
by intros len_A len_B speed_A speed_B h1 h2 h3 h4
suffices H : (550 : ℝ) / 33.33 = 16.5 by assumption
sorry

end arun_cross_train_b_time_l244_244326


namespace chromatic_number_decrease_by_removal_l244_244662

theorem chromatic_number_decrease_by_removal 
  (n r : ℕ) (hn_pos : 0 < n) (hr_pos : 0 < r) :
  ∃ (N : ℕ), ∀ (G : SimpleGraph V) [fintype V] [decidable_rel G.adj]
    (hV_size : fintype.card V ≥ N)
    (hχG : χ G = n),
  ∃ (S : Finset V), S.card = r ∧ χ (G.delete_vertices S) ≥ n - 1 :=
by
  sorry

end chromatic_number_decrease_by_removal_l244_244662


namespace cofactor_A_23_l244_244453

-- Define the matrix A
def A : matrix (fin 3) (fin 3) ℤ :=
  ![
    [2, -4, 0],
    [-1, 3, 5],
    [1, -4, -3]
  ]

-- Define the submatrix by removing the 2nd row and 3rd column
def submatrix_23 (A : matrix (fin 3) (fin 3) ℤ) : matrix (fin 2) (fin 2) ℤ :=
  ![
    [A 0 0, A 0 1],
    [A 2 0, A 2 1]
  ]

-- Define the determinant of the submatrix
def submatrix_23_det (A : matrix (fin 3) (fin 3) ℤ) : ℤ :=
  A 0 0 * A 2 1 - A 0 1 * A 2 0

-- State the proof problem
theorem cofactor_A_23 : submatrix_23_det A = -4 := 
by
  sorry

end cofactor_A_23_l244_244453


namespace largest_possible_median_l244_244330

theorem largest_possible_median (x y : ℤ) :
  ∃ l : list ℤ, (l = [x, 2 * x, y, 3, 2, 5, 7] ∧
  (l.sorted! (fun a b => a < b)).nth (3) = some (7)) :=
begin
  sorry
end

end largest_possible_median_l244_244330


namespace circulation_ratio_l244_244682

theorem circulation_ratio (A : ℕ) :
  let circulation_1961 := 4 * A,
      total_circulation_1962_1970 := 9 * A,
      total_circulation_1961_1970 := circulation_1961 + total_circulation_1962_1970
  in circulation_1961 / total_circulation_1961_1970 = 4 / 13 :=
by
  sorry

end circulation_ratio_l244_244682


namespace sum_of_undefined_values_l244_244738

theorem sum_of_undefined_values (y : ℝ) :
  (y^2 - 7 * y + 12 = 0) → y = 3 ∨ y = 4 → (3 + 4 = 7) :=
by
  intro hy
  intro hy'
  sorry

end sum_of_undefined_values_l244_244738


namespace dot_product_a_b_l244_244508

def a : ℝ × ℝ × ℝ := (-1, -3, 2)
def b : ℝ × ℝ × ℝ := (1, 2, 0)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem dot_product_a_b : dot_product a b = -7 := by
  sorry

end dot_product_a_b_l244_244508


namespace find_judes_age_l244_244560

def jude_age (H : ℕ) (J : ℕ) : Prop :=
  H + 5 = 3 * (J + 5)

theorem find_judes_age : ∃ J : ℕ, jude_age 16 J ∧ J = 2 :=
by
  sorry

end find_judes_age_l244_244560


namespace probability_at_least_6_heads_in_9_flips_l244_244800

theorem probability_at_least_6_heads_in_9_flips : 
  let total_outcomes := 2 ^ 9 in
  let successful_outcomes := Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9 in
  successful_outcomes.toRational / total_outcomes.toRational = (130 : ℚ) / 512 :=
by
  sorry

end probability_at_least_6_heads_in_9_flips_l244_244800


namespace sum_of_specific_primes_l244_244438

theorem sum_of_specific_primes : 
  let primes := { p : ℕ | nat.prime p ∧ (∀ x : ℤ, ¬(5 * (10 * x + 2) ≡ 3 [MOD p])) } in 
  ∑ p in primes, p = 7 :=
by
  sorry

end sum_of_specific_primes_l244_244438


namespace integral_value_l244_244008

noncomputable def a : ℝ := -7 / 32

theorem integral_value :
  (∫ x in 1..(-32 * a), (Real.exp x - 1 / x)) = Real.exp 7 - Real.log 7 - Real.exp 1 :=
by
  sorry

end integral_value_l244_244008


namespace area_of_triangle_CDE_l244_244098

theorem area_of_triangle_CDE
  (DE : ℝ) (h : ℝ)
  (hDE : DE = 12) (hh : h = 15) :
  1/2 * DE * h = 90 := by
  sorry

end area_of_triangle_CDE_l244_244098


namespace proof_problem_l244_244150

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244150


namespace school_girls_more_than_boys_l244_244572

def num_initial_girls := 632
def num_initial_boys := 410
def num_new_girls := 465
def num_total_girls := num_initial_girls + num_new_girls
def num_difference_girls_boys := num_total_girls - num_initial_boys

theorem school_girls_more_than_boys :
  num_difference_girls_boys = 687 :=
by
  sorry

end school_girls_more_than_boys_l244_244572


namespace count_two_digit_primes_l244_244530

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_digits : set ℕ := {3, 5, 7, 9}

def two_digit_primes := { n | ∃ (a b : ℕ), a ∈ valid_digits ∧ b ∈ valid_digits ∧ a ≠ b ∧ n = 10 * a + b ∧ is_prime n }

theorem count_two_digit_primes : (two_digit_primes : set ℕ).card = 7 := 
  sorry

end count_two_digit_primes_l244_244530


namespace min_m_plus_n_l244_244474

variable {α : Type*}

def geom_seq (a : ℕ → α) (r : α) := ∀ n, a (n + 1) = a n * r

theorem min_m_plus_n (a : ℕ → ℝ) (r : ℝ) (h1 : geom_seq a r)
    (h2 : a 7 = a 6 + 2 * a 5)
    (h3 : ∃ m n, a m + a n = 4 * a 1) :
    ∃ m n, a m + a n = 4 * a 1 ∧ m + n = 4 :=
begin
  sorry
end

end min_m_plus_n_l244_244474


namespace g_function_property_l244_244252

theorem g_function_property (g : ℝ → ℝ) (h : ∀ x y : ℝ, g ((x - y) ^ 2) = g x ^ 2 - 4 * x * g y + 2 * y ^ 2) :
  let m := {y | ∃ x : ℝ, g x = y}.to_finset.card,
      t := {y | ∃ x : ℝ, g x = y}.to_finset.sum id
  in m * t = 8 :=
by
  sorry

end g_function_property_l244_244252


namespace smallest_portion_is_2_l244_244674

theorem smallest_portion_is_2 (a d : ℝ) (h1 : 5 * a = 120) (h2 : 3 * a + 3 * d = 7 * (2 * a - 3 * d)) : a - 2 * d = 2 :=
by sorry

end smallest_portion_is_2_l244_244674


namespace probability_at_least_3_out_of_6_babies_speak_l244_244970

noncomputable def binomial_prob (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  Nat.choose n k * (p^k) * ((1 - p)^(n - k))

noncomputable def prob_at_least_k (total : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  1 - (Finset.range k).sum (λ i => binomial_prob total i p)

theorem probability_at_least_3_out_of_6_babies_speak :
  prob_at_least_k 6 3 (2/5) = 7120/15625 :=
by
  sorry

end probability_at_least_3_out_of_6_babies_speak_l244_244970


namespace num_valid_two_digit_primes_l244_244532

-- Define the set from which the digits are chosen
def digit_set := {3, 5, 7, 9}

-- Define a function to check if a number is a two-digit prime formed by different tens and units digits from digit_set
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ {37, 53, 59, 73, 79, 97} -- Set of prime numbers obtained in the solution

-- Define the main theorem
theorem num_valid_two_digit_primes : (set.filter is_valid_prime { n | ∃ t u, t ≠ u ∧ t ∈ digit_set ∧ u ∈ digit_set ∧ n = 10 * t + u }).card = 7 := 
by
  sorry

end num_valid_two_digit_primes_l244_244532


namespace second_ace_position_most_likely_l244_244327

theorem second_ace_position_most_likely (deck : list ℕ) (h_len : deck.length = 52) (h_unique : deck.nodup) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 52 ∧ is_most_likely_second_ace_position(deck, n) = 18 :=
sorry

def is_most_likely_second_ace_position(deck : list ℕ, n : ℕ) : ℕ :=
sorry

end second_ace_position_most_likely_l244_244327


namespace number_of_valid_rook_placements_l244_244883

-- Define the problem conditions:
def is_checkerboard_pattern (board : matrix (fin 5) (fin 5) bool) : Prop :=
  ∀ i j, board i j = (i + j) % 2 = 1  -- Black if sum of i and j is odd

def is_black_square (board : matrix (fin 5) (fin 5) bool) (i j : fin 5) : Prop :=
  board i j

def is_valid_rook_placement (rooks : list (fin 5 × fin 5)) : Prop :=
  rooks.length = 5 ∧
  (∀ r ∈ rooks, is_black_square board r.1 r.2) ∧
  (function.injective (prod.fst ∘ function.uncurry id) rooks) ∧
  (function.injective (prod.snd ∘ function.uncurry id) rooks)

-- Main theorem statement:
theorem number_of_valid_rook_placements :
  ∃ rooks : list (fin 5 × fin 5), is_valid_rook_placement rooks ∧
  (card (set.univ {rooks : list (fin 5 × fin 5) | is_valid_rook_placement rooks}) = 1440) :=
begin
  sorry
end

end number_of_valid_rook_placements_l244_244883


namespace positive_integer_solutions_count_l244_244448

theorem positive_integer_solutions_count :
  (∃ (m n : ℕ+), 20 * m + 12 * n = 2012) ∧ (∀ (m n : ℕ+), 20 * m + 12 * n = 2012) -> (finset.card {p : ℕ+ × ℕ+ | 20 * p.1 + 12 * p.2 = 2012} = 34) :=
by
  sorry

end positive_integer_solutions_count_l244_244448


namespace highest_probability_highspeed_rail_l244_244983

def total_balls : ℕ := 10
def beidou_balls : ℕ := 3
def tianyan_balls : ℕ := 2
def highspeed_rail_balls : ℕ := 5

theorem highest_probability_highspeed_rail :
  (highspeed_rail_balls : ℚ) / total_balls > (beidou_balls : ℚ) / total_balls ∧
  (highspeed_rail_balls : ℚ) / total_balls > (tianyan_balls : ℚ) / total_balls :=
by {
  -- Proof skipped
  sorry
}

end highest_probability_highspeed_rail_l244_244983


namespace y_value_on_line_l244_244095

theorem y_value_on_line (x y : ℝ) (k : ℝ → ℝ)
  (h1 : k 0 = 0)
  (h2 : ∀ x, k x = (1/5) * x)
  (hx1 : k x = 1)
  (hx2 : k 5 = y) :
  y = 1 :=
sorry

end y_value_on_line_l244_244095


namespace origin_to_line_distance_l244_244994

theorem origin_to_line_distance (O : euclidean_space ℝ (fin 2)) (A B C : ℝ)
  (hO : O = ![0, 0])
  (hLine : A = 1 ∧ B = -2 ∧ C = 5) :
  euclidean_geometry.dist O (euclidean_geometry.affine_span ℝ {x : euclidean_space ℝ (fin 2) | A * (x 0) + B * (x 1) + C = 0}) = √5 :=
by
  sorry

end origin_to_line_distance_l244_244994


namespace find_a_l244_244968

theorem find_a (a : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x ∈ set.Icc (0 : ℝ) 3, ax^2 - 2 * ax ≤ 3) :
  a = 1 ∨ a = -3 := 
sorry

end find_a_l244_244968


namespace range_of_a_for_max_value_l244_244957

theorem range_of_a_for_max_value 
  (f : ℝ → ℝ) (a : ℝ)
  (h : f = λ x, (x - a) ^ 2 * (x - 1))
  (h_max : ∀ x, f x ≤ f a) :
  a < 1 := 
sorry

end range_of_a_for_max_value_l244_244957


namespace necessary_but_not_sufficient_l244_244479

noncomputable def not_coplanar (E F G H : Type) [affine_space ℝ E] (e f g h : E) : Prop :=
  ¬ exists (plane : set E), e ∈ plane ∧ f ∈ plane ∧ g ∈ plane ∧ h ∈ plane

noncomputable def do_not_intersect (E : Type) [affine_space ℝ E] (e f g h : E) : Prop :=
  ¬ exists (p : E), ∃ (t s : ℝ), p = t • e + (1 - t) • f ∧ p = s • g + (1 - s) • h

theorem necessary_but_not_sufficient (E F G H : Type) [affine_space ℝ E] (e f g h : E) :
  not_coplanar E F G H e f g h → do_not_intersect E e f g h :=
by sorry

end necessary_but_not_sufficient_l244_244479


namespace planes_parallel_l244_244556

-- Define the vectors u and v
def u : ℝ × ℝ × ℝ := (1, 2, -1)
def v : ℝ × ℝ × ℝ := (-3, -6, 3)

-- Define the relationship between u and v indicating parallelism
axiom parallel_vectors : ∃ k : ℝ, v = (k * u.1, k * u.2, k * u.3)

-- Define planes α and β being parallel if their normal vectors u and v are parallel
theorem planes_parallel (α β : Type) (nα nβ : ℝ × ℝ × ℝ)
  (hα : nα = u) (hβ : nβ = v) : 
  nβ = -3 * nα → α ∥ β :=
by
  intro h
  sorry

end planes_parallel_l244_244556


namespace line_intersects_x_axis_at_2_l244_244804

variables {R : Type*} [LinearOrderedField R]

-- Define points and line equation
def point1 : R × R := (3, -2)
def point2 : R × R := (-1, 6)

noncomputable def slope : R := ((point2.2 - point1.2) / (point2.1 - point1.1))

noncomputable def line_equation (x : R) : R := point1.2 + slope * (x - point1.1)

theorem line_intersects_x_axis_at_2 :
  ∃ x : R, line_equation x = 0 ∧ x = 2 :=
by {
  sorry -- Proof is not required.
}

end line_intersects_x_axis_at_2_l244_244804


namespace compute_PQ_square_l244_244575

variables (A B C D P Q : Type)
variables [add_comm_group A] [module ℝ A]
variables [add_comm_group B] [module ℝ B]
variables [add_comm_group C] [module ℝ C]
variables [add_comm_group D] [module ℝ D]
variables [add_comm_group P] [module ℝ P]
variables [add_comm_group Q] [module ℝ Q]
variables (AB DA : ℝ) (BC CD : ℝ)
variables (angle_A : ℝ)
variables (midpoint_P : (B + C)/2 = P)
variables (midpoint_Q : (D + A)/2 = Q)

theorem compute_PQ_square :
  AB = 16 → DA = 16 → BC = 20 → CD = 20 → angle_A = π/2 → midpoint_P → midpoint_Q →
  dist P Q = 20 :=
begin
  sorry,
end

end compute_PQ_square_l244_244575


namespace parabola_focus_line_l244_244169

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244169


namespace largest_number_of_digits_erased_l244_244734

theorem largest_number_of_digits_erased (num_repetitions : ℕ) (d1 d2 d3 d4 : ℤ) (total_digits : ℕ) (required_sum : ℤ) (total_sum : ℤ)
  (h1 : num_repetitions = 250)
  (h2 : d1 = 2 ∧ d2 = 0 ∧ d3 = 1 ∧ d4 = 8)
  (h3 : total_digits = 1000)
  (h4 : required_sum = 2018)
  (h5 : total_sum = 2750) :
  let sum_to_erase := total_sum - required_sum,
      total_erased := 741 in
  (∀ erased_digits : ℕ, erased_digits ≤ 1000 → (total_digits - erased_digits) = total_digits - total_erased) := sorry

end largest_number_of_digits_erased_l244_244734


namespace problem_proof_l244_244048

theorem problem_proof (a : ℝ)
  (statement1 : ∀ x : ℝ, x^2 - a*x + 1 ≠ 0)
  (statement2 : |a - 2| = 2 - a)
  (statement3 : ∀ x y : ℝ, (x + y^2 = a ∧ x - sin(y)^2 = -3) → ∃! (x y : ℝ), x + y^2 = a ∧ x - sin(y)^2 = -3) :
  a ∈ (-2 : ℝ, 2) ∨ a = -3 := by
sorry

end problem_proof_l244_244048


namespace conjugate_of_complex_number_in_third_quadrant_l244_244875

def quadrant_of_conjugate (z : ℂ) : String :=
  if (z.re < 0) ∧ (z.im < 0) then "third quadrant" else "not in third quadrant"

theorem conjugate_of_complex_number_in_third_quadrant :
  let z := (1 + Complex.i)^2 / (1 - Complex.i)
  quadrant_of_conjugate (conj z) = "third quadrant" :=
by
  let z := (1 + Complex.i)^2 / (1 - Complex.i)
  have h : conj z = -1 - Complex.i := sorry
  show quadrant_of_conjugate (conj z) = "third quadrant" from sorry

end conjugate_of_complex_number_in_third_quadrant_l244_244875


namespace range_of_a_valid_a_for_circle_passing_focus_l244_244503
-- Import necessary libraries

-- Problem definition and conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def line (a x y : ℝ) : Prop := y = a * x + 1
def focus : (ℝ × ℝ) := (1, 0)

-- Proof statements
theorem range_of_a (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ line a x1 y1 ∧ line a x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2))
  → a ∈ set.Ioo (-1 : ℝ) 0 ∪ set.Ioo 0 1 :=
sorry

theorem valid_a_for_circle_passing_focus (a : ℝ) :
  (∃ x1 y1 x2 y2 : ℝ, parabola x1 y1 ∧ parabola x2 y2 ∧ line a x1 y1 ∧ line a x2 y2 ∧ (x1 ≠ x2 ∨ y1 ≠ y2)
    ∧ let A := (x1, y1) in let B := (x2, y2) in (∃ AB_circle : ∀ x y : ℝ, x^2 + y^2 = 2 * (1 + a) * x + 2 * (1 + 0) * y - 2 * (1 + 1) in AB_circle 1 0))
  → a = -3 - 2 * real.sqrt 3 ∨ a = -3 + 2 * real.sqrt 3 :=
sorry

end range_of_a_valid_a_for_circle_passing_focus_l244_244503


namespace fractional_inequality_solution_set_l244_244715

theorem fractional_inequality_solution_set (x : ℝ) :
  (x / (x + 1) < 0) ↔ (-1 < x) ∧ (x < 0) :=
sorry

end fractional_inequality_solution_set_l244_244715


namespace proof_problem_l244_244155

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244155


namespace countSpacySets_15_l244_244950

inductive SpacySubset : ℕ → Set (Set ℕ) where
| baseCases : SpacySubset 1 := { ∅, {1} }
| baseCases' : SpacySubset 2 := { ∅, {1}, {2} }
| baseCases'' : SpacySubset 3 := { ∅, {1}, {2}, {3} }
-- For n ≥ 4: spacy sets according to the recurrence relation
| recur (n : ℕ) (h1 : n ≥ 4) : 
    SpacySubset n = { X ∈ SpacySubset (n-1) | X ∪ {n} ∉ SpacySubset (n-3) }

def countSpacySets (n : ℕ) : ℕ := 
    if n = 1 then 2 else
    if n = 2 then 3 else
    if n = 3 then 4 else
    countSpacySets (n - 1) + countSpacySets (n - 3)

theorem countSpacySets_15 : countSpacySets 15 = 406 :=
by
  -- Expanded details of steps to prove the theorem
  sorry

end countSpacySets_15_l244_244950


namespace stuffed_animals_total_l244_244639

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l244_244639


namespace area_is_integer_l244_244099

-- Define the geometrical relationships and constraints
def quadrilateral_integers (AB CD: ℕ) : Prop :=
  let area := ((AB + CD) * (Math.sqrt (AB * CD))) / 2
  (AB * CD = 100) ∧ int.is_int area

-- Using the conditions
def proof_quadrilateral : Prop :=
  quadrilateral_integers 10 10

theorem area_is_integer 
  (AB CD: ℕ) 
  (h1: AB = 10) 
  (h2: CD = 10) :
  proof_quadrilateral :=
by {
  rw [h1, h2],
  unfold proof_quadrilateral,
  unfold quadrilateral_integers,
  rw Nat.sqrt_eq' 100 10,
  apply sorry,
}

end area_is_integer_l244_244099


namespace jo_climbs_8_stairs_l244_244441

def f : ℕ → ℕ 
| 0     := 1
| 1     := 1
| 2     := 2
| 3     := 4
| (n+4) := f n + f (n+1) + f (n+2) + f (n+3)

theorem jo_climbs_8_stairs : f 8 = 108 := 
sorry

end jo_climbs_8_stairs_l244_244441


namespace coloring_numbers_l244_244277

theorem coloring_numbers (f : ℕ → ℕ) (h : ∀ n, f n < 2017) :
  ∃ x y, x ≠ y ∧ f x = f y ∧ (y / x) ∈ (set_of (λ z, z % 2016 = 0)) :=
by
  have axial_2016 := set.range (λ i : fin 2018, 2016 ^ (i : ℕ))
  have pigeonhole := pigeonhole_of_finite_bounded_range f 2017 2018 h
  obtain ⟨x, ⟨hx, ⟨y, ⟨hy, ⟨x_ne_y, ⟨fx_eq_fy, hrat⟩⟩⟩⟩⟩⟩ := pigeonhole 
  use x, y
  split
  exact x_ne_y
  split
  exact fx_eq_fy
  cases hrat
  use _,
  -- proving (y / x) ∈ set_of (λ z, z % 2016 = 0)
  sorry

end coloring_numbers_l244_244277


namespace sin_plus_cos_eq_3sqrt5_div_5_l244_244053

open Real

theorem sin_plus_cos_eq_3sqrt5_div_5 (θ : ℝ) (hθ : 0 < θ ∧ θ < π / 2) (h : sin θ - 2 * cos θ = 0) :
  sin θ + cos θ = 3 * sqrt 5 / 5 :=
begin
  -- Proof to be filled in
  sorry
end

end sin_plus_cos_eq_3sqrt5_div_5_l244_244053


namespace general_formula_arith_seq_sum_first_n_terms_seq_b_l244_244922

def arith_seq (a1 d : ℕ) (n : ℕ) : ℕ :=
  a1 + (n - 1) * d

-- Define the given conditions
def a3 := 5
def S9 := 9

-- Definition of the arithmetic sequence and its sum
noncomputable def a_n (n : ℕ) : ℕ := arith_seq 11 (-2) n -- this matches 11 - 2n

-- Proof of the general formula for an arithmetic sequence
theorem general_formula_arith_seq :
  ∀ (a1 d : ℕ),
  (arith_seq 11 (-2) 3 = 5) ∧
  (9 * a1 + 36 * d = 9) →
  ∀ n, a_n n = 11 - 2 * n := 
sorry

-- Definition of the sequence b_n and its sum T_n
def b_n (n : ℕ) : ℕ := abs (a_n n)

def T_n (n : ℕ) : ℕ :=
if n ≤ 5 then
  10 * n - n ^ 2
else
  n ^ 2 - 10 * n + 50

-- Proof of the sum of the first n terms of the sequence b_n
theorem sum_first_n_terms_seq_b :
  ∀ (n : ℕ),
  (T_n n) =
  (if n ≤ 5 then
    10 * n - n ^ 2
  else
    n ^ 2 - 10 * n + 50) := 
sorry

end general_formula_arith_seq_sum_first_n_terms_seq_b_l244_244922


namespace log_geometric_sequence_l244_244544

/-- If sqrt(2), sqrt(3), and sqrt(x) form a geometric sequence, then log_{(3 / sqrt(2))}(x) = 2. -/
theorem log_geometric_sequence (x : ℝ) (hx : 0 < x) (hgeom : ∃ r : ℝ, r ≠ 0 ∧ r * (sqrt 2) = sqrt 3 ∧ r * (sqrt 3) = sqrt x) :
    Real.logb (3 / sqrt 2) x = 2 :=
by
  sorry

end log_geometric_sequence_l244_244544


namespace average_speed_palindrome_l244_244861

open Nat

theorem average_speed_palindrome :
  ∀ (initial final : ℕ) (time : ℕ), (initial = 12321) →
    (final = 12421) →
    (time = 3) →
    (∃ speed : ℚ, speed = (final - initial) / time ∧ speed = 33.33) :=
by
  intros initial final time h_initial h_final h_time
  sorry

end average_speed_palindrome_l244_244861


namespace integral_x_squared_l244_244440

theorem integral_x_squared : ∫ x in 1..2, x^2 = 7 / 3 :=
by
  sorry

end integral_x_squared_l244_244440


namespace volume_of_regular_hexagonal_pyramid_l244_244898

-- Definitions of our conditions
def base_edge_length := 1
def lateral_edge_length := Real.sqrt 5

-- Statement of the proof problem
theorem volume_of_regular_hexagonal_pyramid :
  ∀ (a l : ℝ), a = base_edge_length ∧ l = lateral_edge_length → 
  let S := 6 * (1 / 2 * a * (Real.sqrt 3 / 2)) in
  let h := Real.sqrt(l^2 - a^2) in
  let V := 1 / 3 * S * h in
  V = Real.sqrt 3 := 
by 
  intros a l hl
  rw hl.1
  rw hl.2
  -- Sorry to skip proof steps
  sorry

end volume_of_regular_hexagonal_pyramid_l244_244898


namespace root_in_interval_one_two_l244_244305

noncomputable def f (x : ℝ) : ℝ := Real.exp x - x - 2

theorem root_in_interval_one_two :
  (f 1 < 0) ∧ (f 2 > 0) ∧ (∀ x, 0 < x → f' x > 0) → ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 := 
begin
  sorry
end

end root_in_interval_one_two_l244_244305


namespace horner_operations_l244_244476

def horner_eval (n : ℕ) (coeffs : Fin n → ℝ) (x : ℝ) : ℝ :=
  let coeffs_list := List.ofFn coeffs
  coeffs_list.foldr (λ a acc, a + x * acc) 0

def num_operations (n : ℕ) : ℕ × ℕ :=
  (n, n)

theorem horner_operations (n : ℕ) (coeffs : Fin n → ℝ) (x : ℝ) :
  num_operations n = (n, n) := by
  sorry

end horner_operations_l244_244476


namespace divisor_count_l244_244885

noncomputable def number_of_valid_divisors (x : ℕ) :=
  let div2016 := list.factors (2016) |> list.to_finset
  let div2015 := list.factors (2015) |> list.to_finset
  (div2016 ∪ div2015).card - (div2016 ∩ div2015).card + 1

theorem divisor_count :
  number_of_valid_divisors 2017 = 43
:= 
sorry

end divisor_count_l244_244885


namespace minimum_value_y_l244_244913

theorem minimum_value_y (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 2) :
  (∀ x : ℝ, x = (1 / a + 4 / b) → x ≥ 9 / 2) :=
sorry

end minimum_value_y_l244_244913


namespace probability_at_least_6_heads_in_9_flips_l244_244801

theorem probability_at_least_6_heads_in_9_flips : 
  let total_outcomes := 2 ^ 9 in
  let successful_outcomes := Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9 in
  successful_outcomes.toRational / total_outcomes.toRational = (130 : ℚ) / 512 :=
by
  sorry

end probability_at_least_6_heads_in_9_flips_l244_244801


namespace parabola_properties_l244_244188

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244188


namespace B_and_D_know_their_own_grades_l244_244405

-- Define the grades and allocation
inductive Grade
| excellent
| good

-- Define students A, B, C, and D
inductive Student
| A
| B
| C
| D

open Student Grade

-- Contextual conditions
axiom condition1 : ∃ (grades : Student → Grade), 
  (grades A = excellent ∨ grades A = good) ∧
  (grades B = excellent ∨ grades B = good) ∧
  (grades C = excellent ∨ grades C = good) ∧
  (grades D = excellent ∨ grades D = good) ∧ 
  (∀ g : Student → Grade, 
    (g A = excellent ∨ g A = good) → 
    (g B = excellent ∨ g B = good) → 
    (g C = excellent ∨ g C = good) → 
    (g D = excellent ∨ g D = good) →
    (g A ≠ g B ∨ g A ≠ g C ∨ g A ≠ g D ∨ g B ≠ g C ∨ g B ≠ g D ∨ g C ≠ g D))

axiom condition2 : ∀ (grades : Student → Grade), 
  (grades A = B ∘ grades ∧ grades A = C ∘ grades)

axiom condition3 : ∀ (grades : Student → Grade), 
  (grades B = grades C)

axiom condition4 : ∀ (grades : Student → Grade), 
  (grades D = grades A)

axiom condition5 : ∀ (grades : Student → Grade),
  (grades A ≠ grades B ∧ grades A ≠ grades C)

-- Define the proposition to be proved
theorem B_and_D_know_their_own_grades :
  ∀ (grades : Student → Grade), True := sorry

end B_and_D_know_their_own_grades_l244_244405


namespace perimeter_of_ABCDEFG_l244_244320

-- Definition of distance function (not importing since no specific knowledge of solution steps)
def distance (x y : ℝ) : ℝ := sorry

-- The problem statement in Lean 4
theorem perimeter_of_ABCDEFG
  (A B C D E F G : ℝ × ℝ)
  (hABC_eq : distance A B = 5 ∧ distance B C = 5 ∧ distance C A = 5)
  (hD_midpoint : D = (A + C) / 2)
  (hG_midpoint : G = (A + E) / 2)
  (hADE_eq : distance A D = distance D E ∧ distance D E = distance E A)
  (h_isosceles_EFG : distance E F = distance F G)
  (hEG_twice_EF : distance E G = 2 * distance E F) :
  distance A B + distance B C + distance C D + distance D E + distance E F + distance F G + distance G A = 18.75 :=
by sorry

end perimeter_of_ABCDEFG_l244_244320


namespace inequality_solution_set_l244_244452

theorem inequality_solution_set :
  {x : ℝ | (x^2 + 2*x + 2) / (x + 2) > 1} = {x : ℝ | (-2 < x ∧ x < -1) ∨ (0 < x)} :=
sorry

end inequality_solution_set_l244_244452


namespace find_d_l244_244426

noncomputable def f (z : ℂ) : ℂ := ((-2 + 2 * Complex.I * Real.sqrt 3) * z + (5 * Real.sqrt 3 - 9 * Complex.I)) / 3

theorem find_d : 
   let d := (7 * Real.sqrt 3) / 37 - (35 * Complex.I) / 37 
   in f d = d :=
by
    sorry

end find_d_l244_244426


namespace cube_of_odd_number_minus_itself_divisible_by_24_l244_244657

theorem cube_of_odd_number_minus_itself_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1) ^ 3 - (2 * n + 1)) :=
by
  sorry

end cube_of_odd_number_minus_itself_divisible_by_24_l244_244657


namespace inequality_system_range_l244_244044

theorem inequality_system_range (a : ℝ) :
  (∃ (x : ℤ), (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0)) ∧
  (∀ x : ℤ, (6 * (x : ℝ) + 2 > 3 * (x : ℝ) + 5) ∧ (2 * (x : ℝ) - a ≤ 0) → (x = 2 ∨ x = 3)) →
  6 ≤ a ∧ a < 8 :=
by
  sorry

end inequality_system_range_l244_244044


namespace prime_count_l244_244524

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def from_digits (tens units : ℕ) : ℕ :=
  10 * tens + units

def is_valid_prime (tens units : ℕ) : Prop :=
  {3, 5, 7, 9}.contains tens ∧ 
  {3, 5, 7, 9}.contains units ∧ 
  tens ≠ units ∧ 
  is_prime (from_digits tens units)

theorem prime_count : 
  (finset.univ.filter (λ p, ∃ tens ∈ {3, 5, 7, 9}, ∃ units ∈ {3, 5, 7, 9}, tens ≠ units ∧ is_prime (from_digits tens units))).card = 6 :=
by
  sorry

end prime_count_l244_244524


namespace positional_relationship_theorem_l244_244917

universe u

noncomputable def positional_relationship (a b : α) [line α] [plane α] : Prop :=
  parallel a α ∧ (b ∈ α) → (parallel a b ∨ skew a b)

-- In this problem, we assert the positional relationship given specific assumptions.
theorem positional_relationship_theorem {α : Type u} [line α] [plane α] 
  (a b : α) (α α' : plane α) : 
  parallel a α ∧ (b ∈ α) → (parallel a b ∨ skew a b) :=
sorry

end positional_relationship_theorem_l244_244917


namespace library_books_loan_l244_244971

theorem library_books_loan (h : ¬ ∀ (b : Book), b.available_for_loan) :
  (∃ (b : Book), ¬ b.available_for_loan) ∧ (¬ ∀ (b : Book), b.available_for_loan) :=
by 
  -- statement II: There is at least one book in this library that is not available for loan.
  have h2 : ∃ (b : Book), ¬ b.available_for_loan,
  from by_contradiction (λ h', h (λ b, classical.by_contradiction (λ hb, h' ⟨b, hb⟩))),
  -- statement IV: Not all books in this library are available for loan.
  suffices h4 : ¬ ∀ (b : Book), b.available_for_loan, from ⟨h2, h4⟩,
  exact h

end library_books_loan_l244_244971


namespace find_magnitude_of_diff_vector_l244_244484

def vec_sub (a b : ℝ × ℝ × ℝ) (c : ℝ) : ℝ × ℝ × ℝ :=
  (a.1 - c * b.1, a.2 - c * b.2, a.3 - c * b.3)

def vec_norm (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2 + v.3 ^ 2)

theorem find_magnitude_of_diff_vector :
  let a : ℝ × ℝ × ℝ := (1, 0, 2)
  let b : ℝ × ℝ × ℝ := (0, 1, 2)
  vec_norm (vec_sub a b 2) = 3 := by
  sorry

end find_magnitude_of_diff_vector_l244_244484


namespace Bernardo_wins_probability_l244_244414

/-- Define the set from which Bernardo and Silvia pick numbers -/
def BernardoSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def SilviaSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- Define the number of ways to pick 3 numbers from each set -/
def choose_Bernardo := (BernardoSet.card.choose 3)
def choose_Silvia := (SilviaSet.card.choose 3)

/-- Probability calculation -/
noncomputable def probability_Bernardo_wins : ℚ := (choose_Bernardo : ℚ) / (choose_Bernardo + choose_Silvia)

theorem Bernardo_wins_probability :
    probability_Bernardo_wins = 7 / 9 :=
sorry

end Bernardo_wins_probability_l244_244414


namespace probability_at_least_6_heads_l244_244795

open Finset

noncomputable def binom (n k : ℕ) : ℕ := (finset.range (k + 1)).sum (λ i, if i.choose k = 0 then 0 else n.choose k)

theorem probability_at_least_6_heads : 
  (finset.sum (finset.range 10) (λ k, if k >= 6 then (nat.choose 9 k : ℚ) else 0)) / 2^9 = (130 : ℚ) / 512 :=
by sorry

end probability_at_least_6_heads_l244_244795


namespace at_least_6_heads_probability_l244_244797

open_locale big_operators

theorem at_least_6_heads_probability : 
  let outcomes := 2 ^ 9 in
  let total_ways := (Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9) in
  total_ways / outcomes = 130 / 512 :=
by
  sorry

end at_least_6_heads_probability_l244_244797


namespace sqrt_square_eq_zero_l244_244066

theorem sqrt_square_eq_zero (x y : ℝ) (h : sqrt (x - 1) + (y + 2) ^ 2 = 0) : (x + y) ^ 2023 = -1 :=
sorry

end sqrt_square_eq_zero_l244_244066


namespace stuffed_animal_total_l244_244633

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l244_244633


namespace correct_answer_l244_244744

-- Definitions for the statements
variable (P1 P2 P3 P4 : Prop)

-- Definitions according to their correctness
def statement1 : Prop := P1  -- "The range of sample values will affect the applicable range of the regression equation."
def statement2 : Prop := P2  -- "The smaller the residual sum of squares, the better the fit of the model."
def statement3 : Prop := ¬P3 -- "Using the correlation coefficient R² to characterize the regression effect, the smaller the R², the better the fit of the model."
def statement4 : Prop := ¬P4 -- "Random error e is the unique quantity for measuring the predicted variable."

-- Theorem stating that the correct answer is option A: {1, 2}
theorem correct_answer :
  statement1 P1 P2 P3 P4 ∧ statement2 P1 P2 P3 P4 ∧ statement3 P1 P2 P3 P4 ∧ statement4 P1 P2 P3 P4 → 
  (P1 ∧ P2) ∧ (¬P3 ∧ ¬P4) :=
sorry

end correct_answer_l244_244744


namespace smallest_positive_whole_number_divisible_by_first_five_primes_l244_244737

def is_prime (n : Nat) : Prop := Nat.Prime n

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

def smallest_positive_divisible (lst : List Nat) : Nat :=
  List.foldl (· * ·) 1 lst

theorem smallest_positive_whole_number_divisible_by_first_five_primes :
  smallest_positive_divisible first_five_primes = 2310 := by
  sorry

end smallest_positive_whole_number_divisible_by_first_five_primes_l244_244737


namespace subgroup_gcd_l244_244622

namespace ProofProblem

-- Given conditions
variable {S : Set ℤ}

-- Definition of the gcd of the set S
noncomputable def gcd_set (S : Set ℤ) : ℤ :=
  multiset.gcd (multiset.map (↑) (S.to_finset.val))

-- The math proof problem rewritten in Lean 4 statement
theorem subgroup_gcd (d : ℤ) (h : d = gcd_set S) : 
  Subgroup.gmultiples d = Subgroup.closure S :=
sorry

end ProofProblem

end subgroup_gcd_l244_244622


namespace simplify_permutations_sum_l244_244664

-- Definition of factorial
def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

-- Definition of P_k (the number of permutations of k elements)
def P (k : ℕ) : ℕ := factorial k

-- Statement of the problem in Lean 4
theorem simplify_permutations_sum (n : ℕ) : (∑ k in (finset.range n).map nat.succ, k * P k) = factorial (n + 1) - 1 :=
by
  -- Sorry is placed here to avoid writing the proof
  sorry

end simplify_permutations_sum_l244_244664


namespace minimum_value_of_f_l244_244696

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3 * x + 9

-- State the theorem about the minimum value of the function
theorem minimum_value_of_f : ∃ x : ℝ, f x = 7 ∧ ∀ y : ℝ, f y ≥ 7 := sorry

end minimum_value_of_f_l244_244696


namespace min_value_expression_l244_244613

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 2 / b) * (a + 2 / b - 1010) + (b + 2 / a) * (b + 2 / a - 1010) + 101010 = -404040 :=
sorry

end min_value_expression_l244_244613


namespace inclination_angle_of_line_l244_244919

-- Definitions of the conditions
def direction_vector : ℝ × ℝ := (2, 2 * Real.sqrt 3)

-- The proof statement
theorem inclination_angle_of_line : 
  let θ := Real.atan (direction_vector.2 / direction_vector.1) in
  θ = Real.pi / 3 :=
by
  -- We know the direction vector is (2, 2 * sqrt(3))
  unfold direction_vector
  -- Calculate the inclination angle θ
  let θ := Real.atan (2 * Real.sqrt 3 / 2)
  -- Conclude θ = π/3
  have h : θ = Real.pi / 3 := sorry
  exact h

end inclination_angle_of_line_l244_244919


namespace solve_for_x_l244_244555

def angle_deg_min (deg min : ℕ) : ℝ := deg + min / 60.0

def alpha : ℝ := angle_deg_min 13 14

theorem solve_for_x :
  ∀ x : ℝ, (2 * alpha - x = 180 - 3 * x) → x = angle_deg_min 76 46 :=
by
  sorry

end solve_for_x_l244_244555


namespace parabola_focus_line_l244_244166

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244166


namespace problem_value_l244_244256

theorem problem_value (x : ℤ) (h : x = -2023) : 
  abs (abs (abs x - x) - abs x) - x = 4046 :=
by
  sorry

end problem_value_l244_244256


namespace problem_statement_l244_244257

theorem problem_statement (a b c x y z : ℂ)
  (h1 : a = (b + c) / (x - 2))
  (h2 : b = (c + a) / (y - 2))
  (h3 : c = (a + b) / (z - 2))
  (h4 : x * y + y * z + z * x = 67)
  (h5 : x + y + z = 2010) :
  x * y * z = -5892 :=
by {
  sorry
}

end problem_statement_l244_244257


namespace initialMachinesCount_l244_244289

def machinesWorkInDays (M : ℕ) (days : ℕ) (R : ℝ) : Prop :=
  M * R * days = 1

theorem initialMachinesCount :
  ∃ M R : ℝ, machinesWorkInDays M 12 R ∧ machinesWorkInDays (M + 6) 8 R ∧ M = 12 :=
by
  sorry

end initialMachinesCount_l244_244289


namespace find_angle_QOR_l244_244321

variables (P Q R O : Type) [circle O]
variable [is_tangent P O]
variable [is_tangent Q O]
variable [is_tangent R O]

variables (angle_PQR : angle QPR = 50)

theorem find_angle_QOR (h_tangent : ∀ (P Q R O : Type) [circle O], 
                        is_tangent P O → is_tangent Q O → is_tangent R O → 
                        angle QPR = 50) : 
  angle QOR = 65 :=
by
  sorry

end find_angle_QOR_l244_244321


namespace area_of_isosceles_triangle_with_conditions_l244_244675

theorem area_of_isosceles_triangle_with_conditions :
  ∃ (A : ℝ),
    let h := 10,
        perimeter := 40,
        angle_60 := 60
    in 
    A = (100 * Real.sqrt 3) / 3 :=
by
  sorry

end area_of_isosceles_triangle_with_conditions_l244_244675


namespace number_of_distinct_prime_factors_of_sum_of_divisors_400_l244_244514

-- Define the prime factorization of 400
def prime_factorization_400 : Multiset ℕ := {2, 2, 2, 2, 5, 5}

-- Define the function for sum of divisors
def sum_of_divisors (n : ℕ) : ℕ :=
  let factors := Multiset.to_finset (prime_factorization_400 : Multiset ℕ)
  in factors.sum (λ p, (List.range (Multiset.count p prime_factorization_400 + 1)).sum (λ i, p ^ i))

-- Define the function that counts distinct prime factors
def distinct_prime_factors_count (n : ℕ) : ℕ :=
  Multiset.to_finset (prime_factorization_400 : Multiset ℕ).card

-- State the theorem
theorem number_of_distinct_prime_factors_of_sum_of_divisors_400 : distinct_prime_factors_count (sum_of_divisors 400) = 1 :=
by
  sorry

end number_of_distinct_prime_factors_of_sum_of_divisors_400_l244_244514


namespace simplify_expression_l244_244285

variable (x : ℝ)

theorem simplify_expression :
  2 * x - 3 * (2 - x) + 4 * (1 + 3 * x) - 5 * (1 - x^2) = -5 * x^2 + 17 * x - 7 :=
by
  sorry

end simplify_expression_l244_244285


namespace exists_multiple_01_digits_l244_244279

theorem exists_multiple_01_digits {n : ℕ} (hn : 0 < n) : ∃ m : ℕ, (∀ d : ℕ, d ∈ (nat.digits 10 m) → d = 0 ∨ d = 1) ∧ n ∣ m :=
by
  sorry

end exists_multiple_01_digits_l244_244279


namespace factorization_l244_244443

def expression_to_factor (c : ℝ) : ℝ := 189 * c^2 + 27 * c - 36

def factored_expression (c : ℝ) : ℝ := 9 * (3 * c - 1) * (7 * c + 4)

theorem factorization (c : ℝ) : expression_to_factor c = factored_expression c := by
  sorry

end factorization_l244_244443


namespace parabola_condition_l244_244173

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244173


namespace find_length_BD_l244_244272

theorem find_length_BD (c : ℝ) (h : c ≥ Real.sqrt 7) :
  ∃BD, BD = Real.sqrt (c^2 - 7) :=
sorry

end find_length_BD_l244_244272


namespace cos_30_eq_sqrt3_div_2_l244_244339

theorem cos_30_eq_sqrt3_div_2 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l244_244339


namespace rhombus_concurrency_ratio_l244_244475

noncomputable def rhombus (A B C D : Point) : Prop :=
  parallelogram A B C D ∧ length A B = length B C

noncomputable def perpendicular (M N P : Point) : Prop :=
  angle M N P = π / 2

noncomputable def concurrent (P1 P2 P3 : Line) : Prop :=
  ∃ (R : Point), R ∈ P1 ∧ R ∈ P2 ∧ R ∈ P3

theorem rhombus_concurrency_ratio
  (A B C D M P Q : Point)
  (h_rhombus : rhombus A B C D)
  (hM_on_BC : segment M B ∈ segment B C)
  (hP_on_AD : perpendicular M P D ∧ point_on_line P (line A D))
  (hQ_on_AD : perpendicular M Q C ∧ point_on_line Q (line A D))
  (h_concurrent : concurrent (line P B) (line Q C) (line A M)) :
  length B M / length M C = 1 / 2 :=
sorry

end rhombus_concurrency_ratio_l244_244475


namespace inequality_proof_l244_244006

variable {n : ℕ}
variable {a : Fin n → ℝ}
variable {m : ℝ}

theorem inequality_proof (hpos : ∀ i, a i > 0) (hsum : ∑ i, a i = 1) (hm : m > 1) :
    ∑ i, (a i + (1 / a i)) ^ m ≥ n * (↑n + (1 / (↑n : ℝ))) ^ m := 
by
  sorry

end inequality_proof_l244_244006


namespace value_of_p_circle_tangent_to_directrix_l244_244146

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244146


namespace calculate_enclosed_area_l244_244838

open Real

noncomputable def enclosed_area_parametric_curve_line : ℝ :=
2 * sqrt 3

theorem calculate_enclosed_area : 
  let parametric_x := λ t : ℝ, 2 * (t - sin t),
      parametric_y := λ t : ℝ, 2 * (1 - cos t),
      line_y := 3 in
      (∫ t in (2 * π / 3)..(5 * π / 3), 
          ((parametric_x t * parametric_y' t) - 0) + 
          ((line_y - 0) * ((parametric_x (t + π) - parametric_x t)))
      ) = 2 * sqrt 3 :=
by
  sorry

end calculate_enclosed_area_l244_244838


namespace vector_D_collinear_with_a_l244_244753

def is_collinear (a b : ℝ × ℝ × ℝ) : Prop :=
∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)

def vector_a : ℝ × ℝ × ℝ := (3, 0, -4)

def vector_D : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

theorem vector_D_collinear_with_a : is_collinear vector_a vector_D :=
sorry

end vector_D_collinear_with_a_l244_244753


namespace num_valid_two_digit_primes_l244_244535

-- Define the set from which the digits are chosen
def digit_set := {3, 5, 7, 9}

-- Define a function to check if a number is a two-digit prime formed by different tens and units digits from digit_set
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ {37, 53, 59, 73, 79, 97} -- Set of prime numbers obtained in the solution

-- Define the main theorem
theorem num_valid_two_digit_primes : (set.filter is_valid_prime { n | ∃ t u, t ≠ u ∧ t ∈ digit_set ∧ u ∈ digit_set ∧ n = 10 * t + u }).card = 7 := 
by
  sorry

end num_valid_two_digit_primes_l244_244535


namespace prove_p_equals_2_l244_244192

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244192


namespace p_eq_two_circle_tangent_proof_l244_244224

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244224


namespace math_problem_l244_244418

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- State the theorem
theorem math_problem : fact 8 - 7 * fact 7 - 2 * fact 7 = -5040 := by
  sorry

end math_problem_l244_244418


namespace sum_D_r_leq_l244_244607

-- Let S be a finite set of points on a plane, where no three points are collinear.
-- Define the set
-- D(S, r) = { {X, Y} | X, Y in S, d(X, Y) = r }

noncomputable def is_finite_set_of_points (S : Set point) := S.finite
def no_three_points_collinear (S : Set point) := ∀ (X Y Z : point), X ≠ Y → Y ≠ Z → X ≠ Z → X ∈ S → Y ∈ S → Z ∈ S → ¬ collinear X Y Z
def dist (X Y : point) := real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2)
def D (S : Set point) (r : ℝ) := { p | ∃ X Y, p = {X, Y} ∧ X ∈ S ∧ Y ∈ S ∧ dist X Y = r }

-- Proof that
theorem sum_D_r_leq (S : Set point) (h1 : is_finite_set_of_points S) (h2 : no_three_points_collinear S) : 
  ∑ r > 0, |D S r|^2 ≤ (3 * |S|^2 * (|S| - 1)) / 4 := 
sorry

end sum_D_r_leq_l244_244607


namespace max_triangle_area_l244_244754

theorem max_triangle_area
  {A B M : ℝ × ℝ}
  (Curve_Condition : ∀ (P : ℝ × ℝ), P ∈ [A, B, M] → P.1 ^ 2 + P.2 ^ 2 = 1)
  (Line_Intersect : ∀ {P : ℝ × ℝ}, P ∈ [A, B] → P.1 = P.2 ) :
  ∃ (Mmax : ℝ × ℝ), 
    (A.1 * B.2 - A.2 * B.1) * (Mmax.1) = (sqrt 2 + 1) / 2 := 
sorry

end max_triangle_area_l244_244754


namespace proof_problem_l244_244157

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244157


namespace parabola_properties_l244_244186

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244186


namespace math_problem_l244_244070

noncomputable def calculate_pqs (p q s : ℝ) : ℝ := (p + q) * s

theorem math_problem
    (x : ℝ)
    (p q s : ℝ)
    (f : Polynomial ℝ)
    (g : Polynomial ℝ)
    (roots_f : List ℝ)
    (roots_g : List ℝ)
    (h_f_def : f = Polynomial.Coeff 4 * (X - C (roots_f.head)) * (X - C (roots_f.tail.head)) * (X - C (roots_f.tail.tail.head))
     = ([1, 4, 12, 4] : List ℝ))
    (h_g_def : g = Polynomial.Coeff 5 * (X - C (roots_g.head)) * (X - C (roots_g.tail.head)) * (X - C (roots_g.tail.tail.head)) * (X - C (roots_g.tail.tail.tail.head))
     = ([1, 6, 8 * p, 6 * q, s] : List ℝ))
    (h_roots_f : roots_f.sum = -4 ∧ roots_f.prod = -4)
    (h_roots_g : roots_g.sum = -6 ∧ (roots_g.prod - roots_f.prod) = roots_g.prod)
    (a b c d : ℝ)
    (h_a : a ∈ roots_f)
    (h_b : b ∈ roots_f)
    (h_c : c ∈ roots_f)
    (h_d : d ∉ roots_f ∧ d = -2)
    (h_d_sum : d = -2)
    (h_p : p = 2.5)
    (h_q : q = 4.67)
    (h_s : s = 8) :
    calculate_pqs p q s = 57.36 :=
by sorry

end math_problem_l244_244070


namespace total_amount_is_152_l244_244389

noncomputable def total_amount (p q r s t : ℝ) : ℝ := p + q + r + s + t

noncomputable def p_share (x : ℝ) : ℝ := 2 * x
noncomputable def q_share (x : ℝ) : ℝ := 1.75 * x
noncomputable def r_share (x : ℝ) : ℝ := 1.5 * x
noncomputable def s_share (x : ℝ) : ℝ := 1.25 * x
noncomputable def t_share (x : ℝ) : ℝ := 1.1 * x

theorem total_amount_is_152 (x : ℝ) (h1 : q_share x = 35) :
  total_amount (p_share x) (q_share x) (r_share x) (s_share x) (t_share x) = 152 := by
  sorry

end total_amount_is_152_l244_244389


namespace min_unsuccessful_placements_l244_244088

def grid := Array (Array ℤ)
def t_shape (g : grid) (i j : ℕ) (orientation : ℕ) : ℤ :=
  if orientation = 0 then 
    g.get! i.get! j + g.get! (i+1).get! j + g.get! (i+1).get! (j-1) + g.get! (i+1).get! (j+1)
  else 
    g.get! i.get! j + g.get! (i-1).get! j + g.get! (i+1).get! j + g.get! (i).get! (j+1)

-- Given a grid of +1 and -1
axiom grid_filled : ∀ (i j : ℕ), 0 ≤ i < 8 → 0 ≤ j < 8 → (item.get! (item.get! grid i) j = 1 ∨ item.get! (item.get! grid i) j = -1)

-- Proving the minimum number of unsuccessful T-shaped figure placements is 132
theorem min_unsuccessful_placements : 
  ∃ (g : grid), (∑ i in range 8, ∑ j in range 8, if (∃ o, t_shape g i j o ≠ 0) then 1 else 0) = 132 :=
sorry

end min_unsuccessful_placements_l244_244088


namespace general_term_sequence_l244_244940

-- Define the sequence according to the given conditions
def sequence (a : ℕ → ℝ) :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = (3^(n + 1) * a n) / (a n + 3^(n + 1))

-- The main theorem stating the general term of the sequence
theorem general_term_sequence {a : ℕ → ℝ}
  (h : sequence a) : ∀ n, a n = (2 * 3^n) / (3^n - 1) :=
by
  -- Skipping the proof
  sorry

end general_term_sequence_l244_244940


namespace ants_meeting_distance_l244_244677

-- Given conditions
def tile_width : ℝ := 4
def tile_length : ℝ := 6
def total_tiles_length : ℕ := 14
def total_tiles_width : ℕ := 12

-- Given
def total_distance := (total_tiles_length * tile_length) + (total_tiles_width * tile_width)
def half_distance := total_distance / 2

theorem ants_meeting_distance :
  half_distance = 66 := 
by
  sorry

-- The next part can be the meeting point, but it requires the figure to precisely define the point

end ants_meeting_distance_l244_244677


namespace option_A_l244_244490

variable (α β : Type) [plane α] [plane β]
variables (m n : Type) [line m] [line n]

-- Definitions of perpendicular and parallel relationships between lines and planes
noncomputable def perp_line_plane (l : Type) [line l] (p : Type) [plane p] : Prop := sorry
noncomputable def perp_planes (p1 p2 : Type) [plane p1] [plane p2] : Prop := sorry
noncomputable def parallel_line_plane (l : Type) [line l] (p : Type) [plane p] : Prop := sorry
noncomputable def perp_lines (l1 l2 : Type) [line l1] [line l2] : Prop := sorry

-- Given conditions
axiom A1 : perp_planes α β
axiom A2 : ¬(m = α ∨ m = β)
axiom A3 : ¬(n = α ∨ n = β)
axiom A4 : perp_lines m n

-- To Prove
theorem option_A : perp_line_plane m β → parallel_line_plane n β :=
by
  assume h : perp_line_plane m β
  sorry

end option_A_l244_244490


namespace min_two_way_airlines_l244_244566

theorem min_two_way_airlines 
  (V : Finset ℕ)
  (E : Finset (ℕ × ℕ))
  (companies : Finset (Finset (ℕ × ℕ)))
  (hV : V.card = 15)
  (hE : ∀ e ∈ E, (e.1 ∈ V ∧ e.2 ∈ V ∧ e.1 ≠ e.2))
  (hComp : companies.card = 3)
  (hEdges : ∀ edges ∈ companies, edges ⊆ E)
  (hConnectivity : ∀ (c ∈ companies), ∃ (G : Finset (ℕ × ℕ)), (G ⊆ E \ c) ∧ (V.card = 15) ∧ ( ∀ u v ∈ V, u ≠ v → ∃ p : G.Path u v, p.length > 0) )
  : E.card ≥ 21 :=
sorry

end min_two_way_airlines_l244_244566


namespace internal_angles_and_area_of_grey_triangle_l244_244730

/-- Given three identical grey triangles, 
    three identical squares, and an equilateral 
    center triangle with area 2 cm^2,
    the internal angles of the grey triangles 
    are 120 degrees and 30 degrees, and the 
    total grey area is 6 cm^2. -/
theorem internal_angles_and_area_of_grey_triangle 
  (triangle_area : ℝ)
  (α β : ℝ)
  (grey_area : ℝ) :
  triangle_area = 2 →  
  α = 120 ∧ β = 30 ∧ grey_area = 6 :=
by
  sorry

end internal_angles_and_area_of_grey_triangle_l244_244730


namespace solve_problem_l244_244713

open Real

noncomputable def problem_proof : Prop :=
  let a := 0.99 ^ 3.3
  let b := log 3 π
  let c := log 2 0.8
  0 < a ∧ a < 1 ∧ 1 < b ∧ c < 0 → c < a ∧ a < b

theorem solve_problem : problem_proof :=
  sorry

end solve_problem_l244_244713


namespace parabola_properties_l244_244182

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244182


namespace range_of_a_l244_244487

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ∈ set.Iio 1 ∨ x ∈ set.Ioi 5 → x^2 - 2 * (a - 2) * x + a > 0) →
  1 < a ∧ a ≤ 5 :=
sorry

end range_of_a_l244_244487


namespace proof_problem_l244_244872

noncomputable def line1_eq : Prop :=
  ∀ (x y : ℝ), y = 1 + √3 * (x - 2) ↔ √3 * x - y - 2 * √3 + 1 = 0

noncomputable def line2_eq1 : Prop :=
  ∀ (x y : ℝ), y = (-2 / 3) * x ↔ 2 * x + 3 * y = 0

noncomputable def line2_eq2 : Prop :=
  ∀ (x y : ℝ), x / (-1) + y / (-1) = 1 ↔ x + y + 1 = 0

theorem proof_problem :
  line1_eq ∧ (line2_eq1 ∨ line2_eq2) :=
by
  finish [line1_eq, line2_eq1, line2_eq2]

end proof_problem_l244_244872


namespace length_of_median_of_right_triangle_l244_244576

theorem length_of_median_of_right_triangle (A B C M : ℝ) (hAB : dist A B = 4) (hAC : dist A C = 3)
  (right_angle_CAB : ∠ A B C = 90) (midpoint_M : M = (B + C) / 2) : dist A M = 2.5 := 
sorry

end length_of_median_of_right_triangle_l244_244576


namespace problem_l244_244133

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244133


namespace gathering_knows_same_number_l244_244803

theorem gathering_knows_same_number (n : ℕ) (G : Finset (Finset ℕ)) (cond1 : ∀ A B ∈ G, (∀ x ∈ A, ∀ y ∈ B, (¬x = y)) → |A ∩ B| = 2) (cond2 : ∀ A B ∈ G, (∃ x ∈ A, ∃ y ∈ B, x = y) → |A ∩ B| = 0) :
  ∃ m : ℕ, ∀ A ∈ G, |A| = m :=
sorry

end gathering_knows_same_number_l244_244803


namespace johns_original_salary_l244_244119

def percentage_increase := 9.090909090909092 / 100
def johns_new_salary := 60
def expected_original_salary := 55

theorem johns_original_salary :
  ∃ x : ℝ, x + (percentage_increase * x) = johns_new_salary ∧ x = expected_original_salary :=
by
  sorry

end johns_original_salary_l244_244119


namespace surface_area_of_z_eq_xy_over_a_l244_244840

noncomputable def surface_area (a : ℝ) (h : a > 0) : ℝ :=
  (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1)

theorem surface_area_of_z_eq_xy_over_a (a : ℝ) (h : a > 0) :
  surface_area a h = (2 * Real.pi / 3) * a^2 * (2 * Real.sqrt 2 - 1) := 
sorry

end surface_area_of_z_eq_xy_over_a_l244_244840


namespace number_of_correct_conclusions_l244_244688

-- Define the conditions given in the problem
def conclusion1 (x : ℝ) : Prop := x > 0 → x > Real.sin x
def conclusion2 (x : ℝ) : Prop := (x - Real.sin x = 0 → x = 0) → (x ≠ 0 → x - Real.sin x ≠ 0)
def conclusion3 (p q : Prop) : Prop := (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)
def conclusion4 : Prop := ¬(∀ x : ℝ, x - Real.log x > 0) = ∃ x : ℝ, x - Real.log x ≤ 0

-- Prove the number of correct conclusions is 3
theorem number_of_correct_conclusions : 
  (∃ x1 : ℝ, conclusion1 x1) ∧
  (∃ x1 : ℝ, conclusion2 x1) ∧
  (∃ p q : Prop, conclusion3 p q) ∧
  ¬conclusion4 →
  3 = 3 :=
by
  intros
  sorry

end number_of_correct_conclusions_l244_244688


namespace problem_l244_244136

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244136


namespace inequality_amgm_l244_244624

variable {a b c : ℝ}

theorem inequality_amgm (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
    (a / Real.sqrt(a^2 + 9 * b * c)) + (b / Real.sqrt(b^2 + 9 * c * a)) + (c / Real.sqrt(c^2 + 9 * a * b)) 
    ≥ 3 / Real.sqrt(10) :=
by
  sorry

end inequality_amgm_l244_244624


namespace parabola_p_and_circle_tangent_directrix_l244_244228

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244228


namespace recommended_sleep_hours_l244_244116

theorem recommended_sleep_hours
  (R : ℝ)   -- The recommended number of hours of sleep per day
  (h1 : 2 * 3 + 5 * (0.60 * R) = 30) : R = 8 :=
sorry

end recommended_sleep_hours_l244_244116


namespace find_S11_l244_244495

variable (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ)

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d a1, ∀ n, a n = a1 + (n - 1) * d

axiom sum_of_first_n_terms (a : ℕ → ℕ) (S : ℕ → ℕ) : ∀ n, S n = n * (a 1 + a n) / 2
axiom condition1 : is_arithmetic_sequence a
axiom condition2 : a 5 + a 7 = (a 6)^2

-- Proof (statement) that the sum of the first 11 terms is 22
theorem find_S11 : S 11 = 22 :=
  sorry

end find_S11_l244_244495


namespace isosceles_triangle_circum_incenter_distance_l244_244845

variable {R r d : ℝ}

/-- The distance \(d\) between the centers of the circumscribed circle and the inscribed circle of an isosceles triangle satisfies \(d = \sqrt{R(R - 2r)}\) --/
theorem isosceles_triangle_circum_incenter_distance (hR : 0 < R) (hr : 0 < r) 
  (hIso : ∃ (A B C : ℝ × ℝ), (A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C) ∧ (dist A B = dist A C)) 
  : d = Real.sqrt (R * (R - 2 * r)) :=
sorry

end isosceles_triangle_circum_incenter_distance_l244_244845


namespace new_year_markup_l244_244380

variable (C : ℝ) -- original cost of the turtleneck sweater
variable (N : ℝ) -- New Year season markup in decimal form
variable (final_price : ℝ) -- final price in February

-- Conditions
def initial_markup (C : ℝ) := 1.20 * C
def after_new_year_markup (C : ℝ) (N : ℝ) := (1 + N) * initial_markup C
def discount_in_february (C : ℝ) (N : ℝ) := 0.94 * after_new_year_markup C N
def profit_in_february (C : ℝ) := 1.41 * C

-- Mathematically equivalent proof problem (statement only)
theorem new_year_markup :
  ∀ C : ℝ, ∀ N : ℝ,
    discount_in_february C N = profit_in_february C →
    N = 0.5 :=
by
  sorry

end new_year_markup_l244_244380


namespace parabola_condition_l244_244176

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244176


namespace tetrahedron_volume_l244_244654

theorem tetrahedron_volume (h_1 h_2 h_3 : ℝ) (V : ℝ)
  (h1_pos : 0 < h_1) (h2_pos : 0 < h_2) (h3_pos : 0 < h_3)
  (V_nonneg : 0 ≤ V) : 
  V ≥ (1 / 3) * h_1 * h_2 * h_3 := sorry

end tetrahedron_volume_l244_244654


namespace max_peripheral_cities_l244_244764

-- Define the context: 100 cities, unique paths, and transfer conditions
def number_of_cities := 100
def max_transfers := 11
def max_peripheral_transfers := 10

-- Statement: Prove the maximum number of peripheral cities
theorem max_peripheral_cities (cities : Finset (Fin number_of_cities)) 
  (h_unique_paths : ∀ (A B : Fin number_of_cities), ∃! (path : Finset (Fin number_of_cities)), 
    path.card ≤ max_transfers + 1) 
  (h_reachable : ∀ (A B : Fin number_of_cities), 
    ∃ (path : Finset (Fin number_of_cities)), path.card ≤ max_transfers + 1) 
  (h_peripheral : ∀ (A B : Fin number_of_cities), 
    ¬(A ≠ B ∧ path.card ≤ max_peripheral_transfers + 1)) : 
  ∃ (peripheral : Finset (Fin number_of_cities)), peripheral.card = 89 := 
sorry

end max_peripheral_cities_l244_244764


namespace two_digit_primes_count_l244_244521

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits := {3, 5, 7, 9}

def is_valid_two_digit_prime (n : ℕ) : Prop :=
  is_two_digit_number n ∧ is_prime n ∧ 
  ∃ t u : ℕ, t ∈ digits ∧ u ∈ digits ∧ t ≠ u ∧ n = t * 10 + u

theorem two_digit_primes_count : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, is_valid_two_digit_prime k → k < 100 := 
sorry

end two_digit_primes_count_l244_244521


namespace smallest_x_for_gg_defined_l244_244076

def g (x : ℝ) : ℝ := sqrt (x - 5)

theorem smallest_x_for_gg_defined :
  ∃ x : ℝ, (x ≥ 30) ∧ (∀ y : ℝ, (g(g(y)) = g(g(x)) → y = x)) :=
begin
  sorry
end

end smallest_x_for_gg_defined_l244_244076


namespace locus_of_midpoint_l244_244473

noncomputable def omega : Type := { point : Point // point ∈ circle O 6 }

theorem locus_of_midpoint
    (O : Point) (P : Point) (Q : Point)
    (hPQ : PQ = segment P Q)
    (dist_OP : dist O P = 15)
    (mem_circle : ∀ (Q : omega), dist O Q = 6)
    (M : Point)
    (hM : homothety P (1/3) Q = M) :
    ∀ (Q : omega), dist P M = 5 ∧ circle_radius M 2 := sorry

end locus_of_midpoint_l244_244473


namespace point_c_third_quadrant_l244_244092

variable (a b : ℝ)

-- Definition of the conditions
def condition_1 : Prop := b = -1
def condition_2 : Prop := a = -3

-- Definition to check if a point is in the third quadrant
def is_third_quadrant (a b : ℝ) : Prop := a < 0 ∧ b < 0

-- The main statement to be proven
theorem point_c_third_quadrant (h1 : condition_1 b) (h2 : condition_2 a) :
  is_third_quadrant a b :=
by
  -- Proof of the theorem (to be completed)
  sorry

end point_c_third_quadrant_l244_244092


namespace maximum_y_l244_244043

open Real

noncomputable def L (x y : ℝ) : ℝ := log (x^2 + y^2) (x + y)
def M (x y : ℝ) : Prop := x + y ≥ x^2 + y^2 ∧ x^2 + y^2 > 1
def N (x y : ℝ) : Prop := 0 < x + y ∧ x + y ≤ x^2 + y^2 ∧ x^2 + y^2 < 1
def P (x y : ℝ) : Prop := M x y ∨ N x y

theorem maximum_y (x y : ℝ) (h : L x y ≥ 1) (h' : P x y) : y ≤ 1/2 + sqrt 2/2 :=
sorry

end maximum_y_l244_244043


namespace multiples_7_not_14_less_350_l244_244947

theorem multiples_7_not_14_less_350 : 
  ∃ n : ℕ, n = 25 ∧ (∀ k : ℕ, k < 350 → (k % 7 = 0 ∧ k % 14 ≠ 0 → k ∈ {7 * m | m : ℕ}) ∨ (k % 14 = 0 → k ∉ {7 * m | m : ℕ})) := 
sorry

end multiples_7_not_14_less_350_l244_244947


namespace parallel_vectors_condition_l244_244509

def vectors_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b = (k * a.1, k * a.2)

theorem parallel_vectors_condition (m : ℝ) :
  vectors_parallel (1, m + 1) (m, 2) ↔ m = -2 ∨ m = 1 := by
  sorry

end parallel_vectors_condition_l244_244509


namespace parabola_p_and_circle_tangent_directrix_l244_244233

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244233


namespace seventeen_number_selection_l244_244458

theorem seventeen_number_selection : ∃ (n : ℕ), (∀ s : Finset ℕ, (s ⊆ Finset.range 17) → (Finset.card s = n) → ∃ x y : ℕ, (x ∈ s) ∧ (y ∈ s) ∧ (x ≠ y) ∧ (x = 3 * y ∨ y = 3 * x)) ∧ (n = 13) :=
by
  sorry

end seventeen_number_selection_l244_244458


namespace range_of_x_add_y_l244_244611

noncomputable def floor_not_exceeding (z : ℝ) : ℤ := ⌊z⌋

theorem range_of_x_add_y (x y : ℝ) (h1 : y = 3 * floor_not_exceeding x + 4) 
    (h2 : y = 4 * floor_not_exceeding (x - 3) + 7) (h3 : ¬ ∃ n : ℤ, x = n) : 
    40 < x + y ∧ x + y < 41 :=
by 
  sorry 

end range_of_x_add_y_l244_244611


namespace range_of_m_for_inversely_proportional_function_l244_244045

theorem range_of_m_for_inversely_proportional_function 
  (m : ℝ)
  (h : ∀ x₁ x₂ : ℝ, x₁ > 0 → x₂ > x₁ → (m - 1) / x₂ < (m - 1) / x₁) : 
  m > 1 :=
sorry

end range_of_m_for_inversely_proportional_function_l244_244045


namespace parabola_properties_l244_244181

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244181


namespace length_of_BC_l244_244565

open EuclideanGeometry

theorem length_of_BC
  (O A B C D : Point)
  (h1 : Collinear [O, A, D])
  (h2 : Circle O (dist O A))
  (h3 : dist O B = 7)
  (h4 : Angle A B O = 45) :
  dist B C = 7 := 
  sorry

end length_of_BC_l244_244565


namespace circumradius_of_triangle_ABC_l244_244558

noncomputable def triangleABC := 
{ A := ℝ,
  B := ℝ,
  C := ℝ,
  AB := 4,
  area := 2 * Real.sqrt 3,
  angle := Real.pi / 3 }

theorem circumradius_of_triangle_ABC : triangleABC.R = 2 := by
  sorry

end circumradius_of_triangle_ABC_l244_244558


namespace simplify_fraction_expression_l244_244663

theorem simplify_fraction_expression : (4 + 7 * Complex.i) / (4 - 7 * Complex.i) + (4 - 7 * Complex.i) / (4 + 7 * Complex.i) = -66 / 65 := by
  sorry

end simplify_fraction_expression_l244_244663


namespace smallest_prime_factor_in_C_l244_244284

def smallestPrimeFactor (n : ℕ) : ℕ :=
  Nat.minFactor n

theorem smallest_prime_factor_in_C :
  let C := {65, 67, 68, 71, 74}
  ∃ n ∈ C, smallestPrimeFactor n = 2 ∧
  ∀ m ∈ C, smallestPrimeFactor m ≥ 2 := by
  sorry

end smallest_prime_factor_in_C_l244_244284


namespace range_m_l244_244966

variable {f : ℝ → ℝ}

-- Conditions
def is_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = f (1 - x)

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 1 ≤ x1 → 1 ≤ x2 → x1 ≠ x2 → (f (x1) - f (x2)) / (x1 - x2) > 0

-- Proof problem statement
theorem range_m (h1 : is_symmetric f) (h2 : is_monotonic_increasing f) (m : ℝ) : f(m) > f (-1) → m ∈ Set.Ioo (-∞) (-1) ∪ Set.Ioo 3 ∞ :=
by
  -- Since it's just the statement, adding sorry to skip the proof detail
  sorry

end range_m_l244_244966


namespace same_number_of_real_roots_P_Q_P1_ne_Q1_l244_244828

namespace PolynomialProof

def P(x : ℝ) : ℝ := 3 * x^3 - 5 * x + 4
def Q(x : ℝ) : ℝ := 2 / 3 * x^3 - 2 / 3 * x + 2 / 3

theorem same_number_of_real_roots_P_Q : 
  ∃ r : ℝ, P r = 0 ↔ ∃ r : ℝ, Q r = 0 := sorry

theorem P1_ne_Q1 : P 1 ≠ Q 1 := 
  by {
    calc
    P 1 = 2 : by norm_num
    Q 1 = 2 / 3 : by norm_num
    2 ≠ 2 / 3 : by norm_num 
  }

end PolynomialProof

end same_number_of_real_roots_P_Q_P1_ne_Q1_l244_244828


namespace parabola_p_and_circle_tangent_directrix_l244_244227

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244227


namespace range_of_function_l244_244854

def f (x : ℝ) : ℝ := (sqrt (1 - (Real.sin x)^2)) / (Real.cos x) + (sqrt (1 - (Real.cos x)^2)) / (Real.sin x)

theorem range_of_function : 
  ∀ y, (∃ x, x ∈ ℝ ∧ sin x ≠ 0 ∧ cos x ≠ 0 ∧ f x = y) ↔ y ∈ {-2, 0, 2} :=
by
  sorry

end range_of_function_l244_244854


namespace sin_A_value_l244_244105

-- Definitions of the conditions for the triangle
variables {A B C : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables (angleB : Real.angle) (sideBC sideAB : ℝ)

-- Setting the given conditions
axiom h_angleB : angleB = Real.angle.pi / 2
axiom h_sideBC : sideBC = 4
axiom h_sideAB : sideAB = 10

-- Definition of the sine function for angle A
noncomputable def sin_A (BC AB : ℝ) : ℝ := BC / AB

-- Main theorem stating that given the conditions, sin A = 2 / 5
theorem sin_A_value : sin_A sideBC sideAB = 2 / 5 := sorry

end sin_A_value_l244_244105


namespace max_distance_travel_l244_244002

theorem max_distance_travel (front_tire_lifespan rear_tire_lifespan : ℕ) (h_front : front_tire_lifespan = 24000) (h_rear : rear_tire_lifespan = 36000) :
  ∃ max_distance : ℕ, max_distance = 28800 :=
begin
  use 28800,
  sorry
end

end max_distance_travel_l244_244002


namespace solution_l244_244273

def V := {v : ℤ × ℤ | true}  -- V is the set of all vectors with integer coordinates

def unit_vectors : set (ℤ × ℤ) := {⟨1,0⟩, ⟨0,1⟩, ⟨-1,0⟩, ⟨0,-1⟩}

def is_perpendicular (v w : ℤ × ℤ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def f (v : V) : ℝ

axiom unit_vector_condition :
  ∀ v ∈ unit_vectors, f v = 1

axiom additivity_perpendicular :
  ∀ v w : V, is_perpendicular v w → (f (v.1 + w.1, v.2 + w.2) = f v + f w)

theorem solution :
  ∀ (m n : ℤ), f (m, n) = m^2 + n^2 := sorry

end solution_l244_244273


namespace bees_count_on_fifth_day_l244_244783

theorem bees_count_on_fifth_day
  (initial_count : ℕ) (h_initial : initial_count = 1)
  (growth_factor : ℕ) (h_growth : growth_factor = 3) :
  let bees_at_day (n : ℕ) : ℕ := initial_count * (growth_factor + 1) ^ n
  bees_at_day 5 = 1024 := 
by {
  sorry
}

end bees_count_on_fifth_day_l244_244783


namespace johns_leftover_money_is_correct_l244_244117

def base8_to_base10 (n : Nat) : Nat :=
  n.digits 8 |> Nat.ofDigits 10

def johns_savings : Nat := base8_to_base10 5555

def ticket_cost : Nat := 1200

def leftover_savings : Nat := johns_savings - ticket_cost

theorem johns_leftover_money_is_correct :
  leftover_savings = 1725 :=
by
  have hs : johns_savings = 2925 := by sorry -- convert 5555_8 to base 10 manually
  rw [←hs]
  have ht := subtract_equation : leftover_savings = 2925 - ticket_cost := by sorry
  rw [ht, show 2925 - 1200 = 1725 by sorry]

end johns_leftover_money_is_correct_l244_244117


namespace angle_between_vectors_l244_244469

open real

variables {a b : ℝ^3}
variable angle : ℝ

-- Given conditions
def magnitude_a := real.sqrt 2
def magnitude_b := 2
def perpendicular := (a - b) ∈ realVector dot_productSpace ∧ (a - b) • a = 0

-- Prove that the angle between vectors a and b is π/4
theorem angle_between_vectors : 
  ∥a∥ = magnitude_a →
  ∥b∥ = magnitude_b →
  perpendicular →
  angle = π / 4 := 
sorry

end angle_between_vectors_l244_244469


namespace stream_rate_l244_244382

variable (r w : ℝ)

-- Condition 1: The rower travels 18 miles downstream in 4 hours less than it takes him to return upstream
def cond1 : Prop := 18 / (r + w) + 4 = 18 / (r - w)

-- Condition 2: If he triples his normal rowing speed, the time to travel downstream is only 2 hours less than the time to travel upstream
def cond2 : Prop := 18 / (3 * r + w) + 2 = 18 / (3 * r - w)

-- The goal is to prove that the rate of the stream's current is 9/8
theorem stream_rate : cond1 ∧ cond2 → w = 9 / 8 :=
by
  sorry

end stream_rate_l244_244382


namespace find_a_l244_244471

open Complex

theorem find_a (a : ℝ) (i : ℂ := Complex.I) (h : (a - i) ^ 2 = 2 * i) : a = -1 :=
sorry

end find_a_l244_244471


namespace intersection_point_l244_244873

variable (x y z t : ℝ)

-- Conditions
def line_parametric : Prop := 
  (x = 1 + 2 * t) ∧ 
  (y = 2) ∧ 
  (z = 4 + t)

def plane_equation : Prop :=
  x - 2 * y + 4 * z - 19 = 0

-- Problem statement
theorem intersection_point (h_line: line_parametric x y z t) (h_plane: plane_equation x y z):
  x = 3 ∧ y = 2 ∧ z = 5 :=
by
  sorry

end intersection_point_l244_244873


namespace inequality_proof_l244_244015

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + c^2 = 1) : 
  a + b + Real.sqrt 2 * c ≤ 2 := 
by 
  sorry

end inequality_proof_l244_244015


namespace equation_of_AB_right_angled_triangle_ABC_circumcircle_of_ABC_l244_244507

-- Given vertices of the triangle
def A : (ℝ × ℝ) := (4, 1)
def B : (ℝ × ℝ) := (1, 5)
def C : (ℝ × ℝ) := (-3, 2)

-- Prove the equation of line AB
theorem equation_of_AB :
  ∀ x y : ℝ, 4 * x + 3 * y - 19 = 0 ↔ (y - 1) * (1 - 4) = (x - 4) * (5 - 1) :=
begin
  sorry
end

-- Prove that triangle ABC is right-angled
theorem right_angled_triangle_ABC :
  ∃ k_AB k_BC : ℝ, 
    k_AB = (5 - 1) / (1 - 4) ∧ k_BC = (2 - 5) / (-3 - 1) ∧ k_AB * k_BC = -1 :=
begin
  sorry
end

-- Prove the equation of the circumcircle of triangle ABC
theorem circumcircle_of_ABC :
  (∀ x y : ℝ, (x - 1/2)^2 + (y - 3/2)^2 = 25/2) ↔ 
    (x - A.1)^2 + (y - A.2)^2 = ((B.1 - C.1)^2 + (B.2 - C.2)^2) / 2 :=
begin
  sorry
end

end equation_of_AB_right_angled_triangle_ABC_circumcircle_of_ABC_l244_244507


namespace volume_of_right_triangular_prism_is_correct_l244_244409

noncomputable def volume_of_right_triangular_prism (ABCD A1B1C1D1 : Cube) (P Q R : Point)
  (hPQ : midpoint AB P) (hQQ : midpoint AD Q) (hRR : midpoint AA1 R) : ℚ :=
  (1/2) * sorry  -- complete the proof here with the provided steps

theorem volume_of_right_triangular_prism_is_correct (ABCD A1B1C1D1 : Cube) (P Q R : Point)
  (hPQ : midpoint AB P) (hQQ : midpoint AD Q) (hRR : midpoint AA1 R)
  : volume_of_right_triangular_prism ABCD A1B1C1D1 P Q R hPQ hQQ hRR = 3/16 :=
begin
  -- the complete proof is expected here
  sorry,
end

end volume_of_right_triangular_prism_is_correct_l244_244409


namespace expression_eq_l244_244608

theorem expression_eq (x : ℝ) : 
    (x + 1)^4 + 4 * (x + 1)^3 + 6 * (x + 1)^2 + 4 * (x + 1) + 1 = (x + 2)^4 := 
  sorry

end expression_eq_l244_244608


namespace triangle_area_l244_244049

noncomputable def area_ABC (AB BC : ℝ) (angle_B : ℝ) : ℝ :=
  1/2 * AB * BC * Real.sin angle_B

theorem triangle_area
  (A B C : Type)
  (AB : ℝ) (A_eq : ℝ) (B_eq : ℝ)
  (h_AB : AB = 6)
  (h_A : A_eq = Real.pi / 6)
  (h_B : B_eq = 2 * Real.pi / 3) :
  area_ABC AB AB (2 * Real.pi / 3) = 9 * Real.sqrt 3 :=
by
  simp [area_ABC, h_AB, h_A, h_B]
  sorry

end triangle_area_l244_244049


namespace num_ways_to_write_2024_as_sum_of_twos_and_threes_l244_244541

theorem num_ways_to_write_2024_as_sum_of_twos_and_threes :
  let num_ways := {y // 0 ≤ y ∧ y ≤ 300 ∧ ∃ x, 0 ≤ x ∧ 2 * x + 3 * y = 2024}.to_finset.card in
  num_ways = 151 :=
by
  sorry

end num_ways_to_write_2024_as_sum_of_twos_and_threes_l244_244541


namespace incorrect_statement_l244_244005

-- Conditions
variable (A B C D E F : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] [Inhabited F]
variables (triangleABC : Triangle A B C) (triangleDEF : Triangle D E F)

-- Congruence of triangles
axiom congruent_triangles : triangleABC ≌ triangleDEF

-- Proving incorrect statement
theorem incorrect_statement : ¬ (AB = EF) := by
  sorry

end incorrect_statement_l244_244005


namespace problem_l244_244135

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244135


namespace number_of_red_balls_l244_244697

-- Conditions
variables (w r : ℕ)
variable (ratio_condition : 4 * r = 3 * w)
variable (white_balls : w = 8)

-- Prove the number of red balls
theorem number_of_red_balls : r = 6 :=
by
  sorry

end number_of_red_balls_l244_244697


namespace problem_l244_244134

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244134


namespace solution_l244_244468

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ),
    x - y = 1 ∧
    x^3 - y^3 = 2 ∧
    x^4 + y^4 = 23 / 9 ∧
    x^5 - y^5 = 29 / 9

theorem solution : problem_statement := sorry

end solution_l244_244468


namespace min_triangular_faces_l244_244790

theorem min_triangular_faces (l c e m n k : ℕ) (h1 : l > c) (h2 : l + c = e + 2) (h3 : l = c + k) (h4 : e ≥ (3 * m + 4 * n) / 2) :
  m ≥ 6 := sorry

end min_triangular_faces_l244_244790


namespace sin_intersection_ratios_eq_l244_244436

theorem sin_intersection_ratios_eq : ∃ (p q : ℕ), p.coprime q ∧ y = sin x ∧ y = sin 60 ∧ p = 1 ∧ q = 2 := 
by
  sorry

end sin_intersection_ratios_eq_l244_244436


namespace distance_from_point_on_ellipse_to_foci_l244_244895

noncomputable def a : ℝ := 5
noncomputable def sum_dist_to_foci : ℝ := 2 * a

theorem distance_from_point_on_ellipse_to_foci
  (P : ℝ × ℝ)
  (on_ellipse : (P.1^2) / 25 + (P.2^2) / 16 = 1)
  (dist_to_one_focus : ℝ)
  (dist_to_one_focus_value : dist_to_one_focus = 3) :
  let dist_to_other_focus := sum_dist_to_foci - dist_to_one_focus in
  dist_to_other_focus = 7 :=
sorry

end distance_from_point_on_ellipse_to_foci_l244_244895


namespace total_books_is_10_l244_244721

def total_books (B : ℕ) : Prop :=
  (2 / 5 : ℚ) * B + (3 / 10 : ℚ) * B + ((3 / 10 : ℚ) * B - 1) + 1 = B

theorem total_books_is_10 : total_books 10 := by
  sorry

end total_books_is_10_l244_244721


namespace rhombic_dodecahedron_no_Hamiltonian_path_l244_244660

-- Define the structure and properties of the rhombic dodecahedron vertices and edges
structure RhombicDodecahedron :=
  (T_vertices : Finset ℕ)
  (F_vertices : Finset ℕ)
  (edges : ℕ → ℕ → Prop)
  (num_T : T_vertices.card = 8)
  (num_F : F_vertices.card = 6)
  (T_neighbors_F : ∀ t ∈ T_vertices, ∀ f ∈ F_vertices, edges t f)
  (F_neighbors_T : ∀ f ∈ F_vertices, ∀ t ∈ T_vertices, edges f t)
  (T_deg_3 : ∀ t ∈ T_vertices, (T_vertices.filter (edges t)).card = 3)
  (F_deg_4 : ∀ f ∈ F_vertices, (F_vertices.filter (edges f)).card = 4)

-- Define Hamiltonian path conditions
def Hamiltonian_path (G : RhombicDodecahedron) (path : List ℕ) : Prop :=
  (path.nodup) ∧ (path.length = G.T_vertices.card + G.F_vertices.card) ∧
  (∀ i < path.length - 1, G.edges (path.nth_le i sorry) (path.nth_le (i+1) sorry))

-- The final theorem to prove the problem is unsolvable
theorem rhombic_dodecahedron_no_Hamiltonian_path (G : RhombicDodecahedron) :
  ¬ ∃ (path : List ℕ), Hamiltonian_path G path :=
by
  sorry

end rhombic_dodecahedron_no_Hamiltonian_path_l244_244660


namespace distinct_values_of_f_l244_244040

def f (x : ℝ) : ℤ := 
  (Real.floor x) + (Real.floor (2 * x)) + (Real.floor (5 * x / 3)) + (Real.floor (3 * x)) + (Real.floor (4 * x))

theorem distinct_values_of_f :
  let distinct_count : ℤ := 734
  ∃ count : ℤ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 100 → 
    f (x) ∈ (int.range distinct_count + 7) ∪ (int.range distinct_count + 8)) ∧ count = distinct_count :=
sorry

end distinct_values_of_f_l244_244040


namespace parabola_properties_l244_244183

-- Define the conditions
def O : Point := ⟨0, 0⟩
def parabola (p : ℝ) : (ℝ × ℝ) := { (x, y) | y^2 = 2 * p * x }
def line : (ℝ × ℝ) := { (x, y) | y = -√3 * (x - 1) }
def directrix (p : ℝ) : (ℝ × ℝ) := { (x, y) | x = -p / 2 }

-- Define the intersections M and N
def is_intersection (p : ℝ) (x : ℝ) (y : ℝ) : Prop :=
  y^2 = 2 * p * x ∧ y = -√3 * (x - 1)

-- Define the proof statement
theorem parabola_properties (p : ℝ) (M N : ℝ × ℝ)
  (h_focus : (p / 2, 0) ∈ parabola p)
  (h_line_focus : (p / 2, 0) ∈ line)
  (h_intersection_M : is_intersection p M.1 M.2)
  (h_intersection_N : is_intersection p N.1 N.2)
  (p_pos : p > 0) :
  p = 2 ∧ tangent_to_directrix (M, N) (directrix p) :=
sorry

end parabola_properties_l244_244183


namespace fixed_point_for_function_l244_244936

def passes_through_fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : Prop :=
  ∃ (x y : ℝ), (x = -2) ∧ (y = -1) ∧ (y = a^(x+2) - 2)

theorem fixed_point_for_function (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) :
  passes_through_fixed_point a h_pos h_ne_one :=
by
  use [-2, -1]
  constructor
  { refl }
  constructor
  { refl }
  sorry

end fixed_point_for_function_l244_244936


namespace common_tangents_of_two_circles_l244_244943

noncomputable def circle_equation_1 (x y : ℝ) : Prop := 
  x ^ 2 + y ^ 2 - 4 * x + 2 * y + 1 = 0

noncomputable def circle_equation_2 (x y : ℝ) : Prop := 
  x ^ 2 + y ^ 2 + 4 * x - 4 * y - 1 = 0

theorem common_tangents_of_two_circles : 
  (∀ x y : ℝ, circle_equation_1 x y ↔ (x - 2) ^ 2 + (y + 1) ^ 2 = 4) → 
  (∀ x y : ℝ, circle_equation_2 x y ↔ (x + 2) ^ 2 + (y - 2) ^ 2 = 9) → 
  3 := 
by 
  sorry

end common_tangents_of_two_circles_l244_244943


namespace count_two_digit_primes_l244_244528

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_digits : set ℕ := {3, 5, 7, 9}

def two_digit_primes := { n | ∃ (a b : ℕ), a ∈ valid_digits ∧ b ∈ valid_digits ∧ a ≠ b ∧ n = 10 * a + b ∧ is_prime n }

theorem count_two_digit_primes : (two_digit_primes : set ℕ).card = 7 := 
  sorry

end count_two_digit_primes_l244_244528


namespace probability_of_rhombus_is_half_l244_244985

open Set

def condition1 (AB BC : ℝ) : Prop :=
  AB = BC

def condition2 (AC BD : ℝ) : Prop :=
  AC = BD

def condition3 (A B D C : Point) : Prop :=
  is_perpendicular (line A C) (line B D)

def condition4 (A B C : Point) : Prop :=
  is_perpendicular (line A B) (line B C)

noncomputable def probability_of_rhombus
  (AB BC AC BD : ℝ)
  (A B C D : Point) :
  ℚ :=
  let conditions := [condition1 AB BC, condition2 AC BD, condition3 A B D C, condition4 A B C]
  let rhombus_conditions := [condition1 AB BC, condition3 A B D C]
  (card (filter id rhombus_conditions).to_finset).card.to_rat /
  (card (filter id conditions).to_finset).card.to_rat

theorem probability_of_rhombus_is_half
  (AB BC AC BD : ℝ)
  (A B C D : Point) :
  probability_of_rhombus AB BC AC BD A B C D = 1 / 2 :=
sorry

end probability_of_rhombus_is_half_l244_244985


namespace problem_l244_244127

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244127


namespace correct_product_l244_244570

theorem correct_product (a b b' : ℕ) (h1 : 0 < a) (h2 : 10 ≤ b ∧ b < 100) (h3 : b' = (b % 10) * 10 + b / 10) (h4 : a * b' = 180) : a * b = 180 :=
begin
  sorry
end

end correct_product_l244_244570


namespace max_profit_at_150_l244_244388

-- Define the conditions
def purchase_price : ℕ := 80
def total_items : ℕ := 1000
def selling_price_initial : ℕ := 100
def sales_volume_decrease : ℕ := 5

-- The profit function
def profit (x : ℕ) : ℤ :=
  (selling_price_initial + x) * (total_items - sales_volume_decrease * x) - purchase_price * total_items

-- The statement to prove: the selling price of 150 yuan/item maximizes the profit at 32500 yuan.
theorem max_profit_at_150 : profit 50 = 32500 := by
  sorry

end max_profit_at_150_l244_244388


namespace find_fourth_number_l244_244717

theorem find_fourth_number (x : ℝ) (h : (3.6 * 0.48 * 2.50) / (x * 0.09 * 0.5) = 800.0000000000001) : x = 0.3 :=
by
  sorry

end find_fourth_number_l244_244717


namespace probability_at_least_6_heads_l244_244793

open Finset

noncomputable def binom (n k : ℕ) : ℕ := (finset.range (k + 1)).sum (λ i, if i.choose k = 0 then 0 else n.choose k)

theorem probability_at_least_6_heads : 
  (finset.sum (finset.range 10) (λ k, if k >= 6 then (nat.choose 9 k : ℚ) else 0)) / 2^9 = (130 : ℚ) / 512 :=
by sorry

end probability_at_least_6_heads_l244_244793


namespace area_inequality_l244_244107

theorem area_inequality 
  (α β γ : ℝ) 
  (P Q S : ℝ) 
  (h1 : P / Q = α * β * γ) 
  (h2 : S = Q * (α + 1) * (β + 1) * (γ + 1)) : 
  (S ^ (1 / 3)) ≥ (P ^ (1 / 3)) + (Q ^ (1 / 3)) :=
by
  sorry

end area_inequality_l244_244107


namespace lcm_of_two_numbers_l244_244357

theorem lcm_of_two_numbers (A B : ℕ) 
  (h_prod : A * B = 987153000) 
  (h_hcf : Int.gcd A B = 440) : 
  Nat.lcm A B = 2243525 :=
by
  sorry

end lcm_of_two_numbers_l244_244357


namespace ellipse_equation_l244_244825

def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-4, 0), (4, 0))

def point_on_ellipse (P : ℝ × ℝ) : Prop := ∃ x y, P = (x, y)

def perpendicular_vectors (P : ℝ × ℝ) : Prop :=
  let (x1, y1) := foci.1
  let (x2, y2) := foci.2
  let (px, py) := P
  ((px - x1) * (px - x2) + (py - y1) * (py - y2)) = 0

def triangle_area (P : ℝ × ℝ) : ℝ :=
  let (x1, y1) := foci.1
  let (x2, y2) := foci.2
  let (px, py) := P
  (1 / 2) * abs (x1 * (y2 - py) + x2 * (py - y1) + px * (y1 - y2))

theorem ellipse_equation :
  ∀ P : ℝ × ℝ,
    point_on_ellipse P →
    perpendicular_vectors P →
    triangle_area P = 9 →
      ∀ x y, (x, y) = P → (x ^ 2 / 25 + y ^ 2 / 9 = 1) :=
by
  intros
  sorry

end ellipse_equation_l244_244825


namespace LittleJohnnyAnnualIncome_l244_244632

theorem LittleJohnnyAnnualIncome :
  ∀ (total_amount bank_amount bond_amount : ℝ) 
    (bank_interest bond_interest annual_income : ℝ),
    total_amount = 10000 →
    bank_amount = 6000 →
    bond_amount = 4000 →
    bank_interest = 0.05 →
    bond_interest = 0.09 →
    annual_income = bank_amount * bank_interest + bond_amount * bond_interest →
    annual_income = 660 :=
by
  intros total_amount bank_amount bond_amount bank_interest bond_interest annual_income 
  intros h_total_amount h_bank_amount h_bond_amount h_bank_interest h_bond_interest h_annual_income
  -- Proof is not required
  sorry

end LittleJohnnyAnnualIncome_l244_244632


namespace angle_EHC_is_45_l244_244996

open EuclideanGeometry

theorem angle_EHC_is_45
  (ABC : Triangle)
  (A B C H E : Point)
  (altitude_from_A : altitude A B C H)
  (angle_bisector_B : angle_bisector B A C E)
  (angle_BEA : ∠ B E A = 45) :
  ∠ E H C = 45 :=
by
  sorry

end angle_EHC_is_45_l244_244996


namespace max_peripheral_cities_l244_244772

-- Defining the conditions
def num_cities := 100
def max_transfers := 11
def unique_paths := true
def is_peripheral (A B : ℕ) (f : ℕ → ℕ → bool) : Prop := (f A B = false ∧ f B A = false ∧ 
                                                            A ≠ B ∧ ∀ k < max_transfers, f A B = false)

-- Mathematical proof problem statement
theorem max_peripheral_cities (f : ℕ → ℕ → bool) 
  (H1 : ∀ A B, A ≠ B → ∃ p, length p ≤ max_transfers ∧ unique_paths ∧ f A B = true) :
  ∃ x, num_cities - x = 89 ∧ (∀ A B, is_peripheral A B f → x < num_cities) :=
sorry

end max_peripheral_cities_l244_244772


namespace sister_granola_bars_l244_244512

-- Definitions based on conditions
def total_bars := 20
def chocolate_chip_bars := 8
def oat_honey_bars := 6
def peanut_butter_bars := 6

def greg_set_aside_chocolate := 3
def greg_set_aside_oat_honey := 2
def greg_set_aside_peanut_butter := 2

def final_chocolate_chip := chocolate_chip_bars - greg_set_aside_chocolate - 2  -- 2 traded away
def final_oat_honey := oat_honey_bars - greg_set_aside_oat_honey - 4           -- 4 traded away
def final_peanut_butter := peanut_butter_bars - greg_set_aside_peanut_butter

-- Final distribution to sisters
def older_sister_chocolate := 2.5 -- 2 whole bars + 1/2 bar
def younger_sister_peanut := 2.5  -- 2 whole bars + 1/2 bar

theorem sister_granola_bars :
  older_sister_chocolate = 2.5 ∧ younger_sister_peanut = 2.5 :=
by
  sorry

end sister_granola_bars_l244_244512


namespace problem_l244_244132

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244132


namespace A_beats_B_by_seconds_l244_244569

theorem A_beats_B_by_seconds :
  ∀ (T_B : ℕ),
  let S_A := 1000 / 119 in
  let S_B := 952 / 119 in
  let T_B := 1000 / S_B in
  119 < T_B →
  T_B - 119 = 6 :=
by 
  intros T_B
  let S_A := 1000 / 119
  let S_B := 952 / 119
  have h1 : S_B = 952 / 119 := rfl
  let T_B' : ℚ := 1000 / S_B
  have h2 : T_B' = 125 := sorry
  have h3 : T_B' > 119 := sorry
  have h4 : T_B' - 119 = 6 := sorry
  exact h4

end A_beats_B_by_seconds_l244_244569


namespace parabola_focus_line_tangent_circle_l244_244210

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244210


namespace part_a_part_b_part_c_part_d_part_e_l244_244280

open EuclideanGeometry

-- Noncomputable section is necessary if any definition relies on nonconstructive methods
noncomputable theory

-- Define the plane, circle S, and its center O
variable (plane : Type) [EuclideanPlane plane]
variable (S : Circle plane) (O : Point plane) (hO : O ∈ center S)

-- Given segment lengths
variables (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- Given line l, points A, B, C, and segments
variables (l : Line plane) (A B C : Point plane)

-- Areas to be proven
theorem part_a (P : Point plane) :
  ∃ (line_parallel line_perpendicular : Line plane),
    parallel l line_parallel ∧ perpendicular l line_perpendicular ∧ P ∈ line_parallel ∧ P ∈ line_perpendicular :=
sorry

theorem part_b (D E F G H : Point plane) :
  ∃ (line_segment : Line plane), D ∈ line_segment ∧ (distance D H = distance B C) :=
sorry

theorem part_c (P X : Point plane) :
  ∃ (line_segment : Line plane), P ∈ line_segment ∧ X ∈ line_segment ∧ (distance P X = (a * b) / c) :=
sorry

theorem part_d (l : Line plane) (A : Point plane) (r : ℝ) (hr : 0 < r) :
  ∃ (P Q : Point plane), P ∈ l ∧ Q ∈ l ∧ P ∈ circle A r ∧ Q ∈ circle A r :=
sorry

theorem part_e (A B : Point plane) (r1 r2 : ℝ) (hr1 : 0 < r1) (hr2 : 0 < r2) :
  ∃ (P Q : Point plane), P ∈ circle A r1 ∧ P ∈ circle B r2 ∧ Q ∈ circle A r1 ∧ Q ∈ circle B r2 :=
sorry

end part_a_part_b_part_c_part_d_part_e_l244_244280


namespace triangle_inequality_third_side_l244_244401

-- Conditions
def a : ℝ := 4
def b : ℝ := 9
def lengths : set ℝ := {4, 5, 9, 13}

-- Theorem: There exists a length l in lengths such that it forms a triangle with a and b
theorem triangle_inequality_third_side : 
  ∃ (l : ℝ), l ∈ lengths ∧ (b - a < l ∧ l < b + a) :=
by
  sorry

end triangle_inequality_third_side_l244_244401


namespace range_of_n_minus_m_l244_244037

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then (exp x - 1) else (3/2 * x + 1)

-- Statement of the theorem
theorem range_of_n_minus_m (m n : ℝ) (hmn : m < n) (hfmn : f m = f n) : 
  n - m ∈ Set.Ioc (2/3) (log (3/2) + 1/3) := 
sorry

end range_of_n_minus_m_l244_244037


namespace multiples_of_7_not_14_l244_244949

theorem multiples_of_7_not_14 (n : ℕ) : 
  ∃ count : ℕ, count = 25 ∧ 
    (∀ x, x < 350 → x % 7 = 0 → x % 14 ≠ 0 ↔ ∃ k, x = 7 * k ∧ k % 2 = 1) := 
by
  have count := (finset.range' 7 350).countp (λ x, x % 7 = 0 ∧ x % 14 ≠ 0)
  have h_count : count = 25 := sorry 
  exact ⟨25, h_count, sorry⟩

end multiples_of_7_not_14_l244_244949


namespace minimum_b_n_S_n_l244_244486

open Real

noncomputable def a_n (n : ℕ) : ℝ := ∫ x in 0..n, 2*x + 1

noncomputable def S_n (n : ℕ) : ℝ := ∑ k in Finset.range n, 1 / a_n (k + 1)

def b_n (n : ℕ) : ℝ := n - 35

noncomputable def b_n_S_n (n : ℕ) : ℝ := b_n n * S_n n

theorem minimum_b_n_S_n : ∃ (n : ℕ+), b_n_S_n n = -25 :=
by
  -- Proof will go here
  sorry

end minimum_b_n_S_n_l244_244486


namespace angle_between_slant_height_and_axis_l244_244080

theorem angle_between_slant_height_and_axis
  (L R : ℝ) (theta : ℝ)
  (h1 : 2 * ℝ.pi * R = (1 / 2) * ℝ.pi * L ^ 2)
  (h2 : L = 2 * R)
  (sin_theta : Real.sin theta = R / L)
  : theta = Real.arcsin (1 / 2) :=
begin
  sorry,
end

end angle_between_slant_height_and_axis_l244_244080


namespace part1_part2_l244_244934

def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (λ : ℝ) (h : ∀ x, f x ≥ λ) : λ ≤ 1 :=
sorry

theorem part2 :
  (∃ m : ℝ, ∃ t : ℝ, m^2 + 2*m + f t = 0) →
  (∀ t, f(t) ≤ 1 → -1 ≤ t ∧ t ≤ 0) :=
sorry

end part1_part2_l244_244934


namespace count_two_digit_primes_l244_244529

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_digits : set ℕ := {3, 5, 7, 9}

def two_digit_primes := { n | ∃ (a b : ℕ), a ∈ valid_digits ∧ b ∈ valid_digits ∧ a ≠ b ∧ n = 10 * a + b ∧ is_prime n }

theorem count_two_digit_primes : (two_digit_primes : set ℕ).card = 7 := 
  sorry

end count_two_digit_primes_l244_244529


namespace count_three_digit_multiples_of_25_not_60_l244_244059

theorem count_three_digit_multiples_of_25_not_60 : 
  ∃ n : ℕ, n = 33 ∧
    (∀ x, 100 ≤ x ∧ x < 1000 ∧ x % 25 = 0 ∧ x % 60 ≠ 0 ↔ x ∈ finset.range 1000) ∧ 
    ((finset.range 1000).filter (λ x, 100 ≤ x ∧ x < 1000 ∧ x % 25 = 0 ∧ x % 60 ≠ 0)).card = n :=
by
  sorry

end count_three_digit_multiples_of_25_not_60_l244_244059


namespace main_theorem_l244_244123

def is_beautiful_labeling (n : ℕ) (labeling : Fin (n + 1) → Fin (n + 1)) : Prop :=
  ∀ a b c d : Fin (n + 1), a < b → b < c → c < d → a + d = b + c → 
  let ad_chord := (a, d) in
  let bc_chord := (b, c) in
  ad_chord ≠ bc_chord -- You might need to adapt for actual chord representation

def beautiful_labelings_count (n : ℕ) : ℕ :=
  (labelings : Fin (n + 1) → Fin (n + 1)) | 
  is_beautiful_labeling n labeling ≃ Fin (n + 1)

def gcd (a b : ℕ) : ℕ := sorry -- Assume you have a gcd function defined

def ordered_pair_count (n : ℕ) : ℕ :=
  (pairs : ℕ × ℕ) | (fst pairs) + (snd pairs) ≤ n ∧ gcd (fst pairs) (snd pairs) = 1 ≃ Fin (n + 1)

theorem main_theorem (n : ℕ) (h : n ≥ 3) : 
  beautiful_labelings_count n = ordered_pair_count n + 1 :=
sorry

end main_theorem_l244_244123


namespace value_of_p_circle_tangent_to_directrix_l244_244140

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244140


namespace minimize_at_x_eq_4_l244_244457

def f (x : ℝ) : ℝ := (3/4) * x^2 - 6 * x + 8

theorem minimize_at_x_eq_4 : ∃ x, f x = f 4 :=
by
  sorry

end minimize_at_x_eq_4_l244_244457


namespace fraction_transform_l244_244077

theorem fraction_transform (x : ℝ) (h : (1/3) * x = 12) : (1/4) * x = 9 :=
by 
  sorry

end fraction_transform_l244_244077


namespace max_peripheral_cities_l244_244766

-- Define the context: 100 cities, unique paths, and transfer conditions
def number_of_cities := 100
def max_transfers := 11
def max_peripheral_transfers := 10

-- Statement: Prove the maximum number of peripheral cities
theorem max_peripheral_cities (cities : Finset (Fin number_of_cities)) 
  (h_unique_paths : ∀ (A B : Fin number_of_cities), ∃! (path : Finset (Fin number_of_cities)), 
    path.card ≤ max_transfers + 1) 
  (h_reachable : ∀ (A B : Fin number_of_cities), 
    ∃ (path : Finset (Fin number_of_cities)), path.card ≤ max_transfers + 1) 
  (h_peripheral : ∀ (A B : Fin number_of_cities), 
    ¬(A ≠ B ∧ path.card ≤ max_peripheral_transfers + 1)) : 
  ∃ (peripheral : Finset (Fin number_of_cities)), peripheral.card = 89 := 
sorry

end max_peripheral_cities_l244_244766


namespace distinct_arrangements_of_pebbles_in_octagon_l244_244592

noncomputable def number_of_distinct_arrangements : ℕ :=
  (Nat.factorial 8) / 16

theorem distinct_arrangements_of_pebbles_in_octagon : 
  number_of_distinct_arrangements = 2520 :=
by
  sorry

end distinct_arrangements_of_pebbles_in_octagon_l244_244592


namespace megan_eggs_per_meal_l244_244265

-- Define the initial conditions
def initial_eggs_from_store : Nat := 12
def initial_eggs_from_neighbor : Nat := 12
def eggs_used_for_omelet : Nat := 2
def eggs_used_for_cake : Nat := 4
def meals_to_divide : Nat := 3

-- Calculate various steps
def total_initial_eggs : Nat := initial_eggs_from_store + initial_eggs_from_neighbor
def eggs_after_cooking : Nat := total_initial_eggs - eggs_used_for_omelet - eggs_used_for_cake
def eggs_after_giving_away : Nat := eggs_after_cooking / 2
def eggs_per_meal : Nat := eggs_after_giving_away / meals_to_divide

-- State the theorem to prove the value of eggs_per_meal
theorem megan_eggs_per_meal : eggs_per_meal = 3 := by
  sorry

end megan_eggs_per_meal_l244_244265


namespace walking_speed_l244_244377

theorem walking_speed 
  (D : ℝ) 
  (V_w : ℝ) 
  (h1 : D = V_w * 8) 
  (h2 : D = 36 * 2) : 
  V_w = 9 :=
by
  sorry

end walking_speed_l244_244377


namespace cube_of_odd_number_minus_itself_divisible_by_24_l244_244658

theorem cube_of_odd_number_minus_itself_divisible_by_24 (n : ℤ) : 
  24 ∣ ((2 * n + 1) ^ 3 - (2 * n + 1)) :=
by
  sorry

end cube_of_odd_number_minus_itself_divisible_by_24_l244_244658


namespace acute_angle_parallel_vectors_l244_244510

variable {θ : ℝ}

def vector_a (θ : ℝ) : ℝ × ℝ := (1 - sin θ, 1)
def vector_b (θ : ℝ) : ℝ × ℝ := (1 / 2, 1 + sin θ)

theorem acute_angle_parallel_vectors (h : vector_a θ = (1 - sin θ, 1) ∧ vector_b θ = (1 / 2, 1 + sin θ) ∧ 
  ((1 - sin θ) * (1 + sin θ) = 1 / 2)) : θ = π / 4 :=
sorry

end acute_angle_parallel_vectors_l244_244510


namespace plane_angles_of_triang_pyramid_l244_244306

theorem plane_angles_of_triang_pyramid
  (S l : ℝ)
  (h1 : l > 0)
  (h2 : S > 0)
  (α : ℝ)
  (h3 : α = Real.arcsin (S * (Real.sqrt 3 - 1) / l ^ 2)) :
  ∃ (β γ : ℝ), (β = α - π/6 ∧ γ = α + π/6 ∧ S = (l ^ 2 / 2) * (Real.sin (α - π/6) + Real.sin α + Real.sin (α + π/6))) :=
by
  sorry

end plane_angles_of_triang_pyramid_l244_244306


namespace a_in_range_of_C_subset_B_l244_244505

open Set 

variable (a : ℝ)

noncomputable def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ a}
noncomputable def B : Set ℝ := {y | ∃ x ∈ A, y = 2 * x + 3}
noncomputable def C : Set ℝ := {z | ∃ x ∈ A, z = x^2}

theorem a_in_range_of_C_subset_B (h : C ⊆ B) : 1 / 2 ≤ a ∧ a ≤ 3 :=
begin
  sorry
end

end a_in_range_of_C_subset_B_l244_244505


namespace largest_n_not_sum_of_30_multiple_and_composite_l244_244329

theorem largest_n_not_sum_of_30_multiple_and_composite :
  ∃ (n : ℕ), n = 211 ∧ ∀ a b : ℕ, (n ≠ 30 * a + b) ∨ ¬ (0 ≤ b ∧ b < 30 ∧ prime b ∧ ∀ k < a, prime (b + 30 * k)) :=
by
  sorry

end largest_n_not_sum_of_30_multiple_and_composite_l244_244329


namespace possible_values_f5_l244_244249

theorem possible_values_f5 (f : ℝ → ℝ) (h : ∀ x y : ℝ, f(f(x) + y^2) = f(x^2 - y) + 4 * f(x) * y^2) : f 5 = 0 ∨ f 5 = 25 :=
sorry

end possible_values_f5_l244_244249


namespace num_valid_two_digit_primes_l244_244533

-- Define the set from which the digits are chosen
def digit_set := {3, 5, 7, 9}

-- Define a function to check if a number is a two-digit prime formed by different tens and units digits from digit_set
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ {37, 53, 59, 73, 79, 97} -- Set of prime numbers obtained in the solution

-- Define the main theorem
theorem num_valid_two_digit_primes : (set.filter is_valid_prime { n | ∃ t u, t ≠ u ∧ t ∈ digit_set ∧ u ∈ digit_set ∧ n = 10 * t + u }).card = 7 := 
by
  sorry

end num_valid_two_digit_primes_l244_244533


namespace exists_integer_a_l244_244603

theorem exists_integer_a (p : ℤ) (hp : Nat.Prime p) (hp2 : 2 ∣ (p - 1)) :
  ∀ (c : ℤ), ∃ (a : ℤ), (a ^ ((p + 1) / 2) + (a + c) ^ ((p + 1) / 2)) % p = c % p :=
by
  sorry

end exists_integer_a_l244_244603


namespace percentage_of_brand_z_l244_244355

/-- Define the initial and subsequent conditions for the fuel tank -/
def initial_fuel_tank : ℕ := 1
def first_stage_z_gasoline : ℚ := 1 / 4
def first_stage_y_gasoline : ℚ := 3 / 4
def second_stage_z_gasoline : ℚ := first_stage_z_gasoline / 2 + 1 / 2
def second_stage_y_gasoline : ℚ := first_stage_y_gasoline / 2
def final_stage_z_gasoline : ℚ := second_stage_z_gasoline / 2
def final_stage_y_gasoline : ℚ := second_stage_y_gasoline / 2 + 1 / 2

/-- Formal statement of the problem: Prove the percentage of Brand Z gasoline -/
theorem percentage_of_brand_z :
  ∃ (percentage : ℚ), percentage = (final_stage_z_gasoline / (final_stage_z_gasoline + final_stage_y_gasoline)) * 100 ∧ percentage = 31.25 :=
by {
  sorry
}

end percentage_of_brand_z_l244_244355


namespace parabola_conditions_l244_244244

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244244


namespace largest_number_peripheral_cities_l244_244768

-- Define the conditions in Lean
def is_peripheral (city : ℕ) (neighbors : ℕ → set ℕ) (c1 c2 : ℕ) : Prop :=
  (c1 ≠ c2) ∧ (∀ p, p ∈ neighbors c1 -> p ≠ c2) ∧ (∀ n, n < 10 → ¬ connected_within_n_steps neighbors c1 c2 n)

def connected_within_n_steps (neighbors : ℕ → set ℕ) (c1 c2 : ℕ) (n : ℕ) : Prop :=
  ∃ (path : list ℕ), (length path) ≤ n ∧ path.head = c1 ∧ path.last = some c2 ∧ ∀ i ∈ path, i ∈ neighbors (path.nth_le i sorry)

def max_peripheral_cities (total_cities : ℕ) (neighbors : ℕ → set ℕ) : ℕ :=
  if total_cities ≠ 100 ∨ ∃ c1 c2, c1 ≠ c2 ∧ ¬ connected_within_n_steps neighbors c1 c2 11 then 0 else
    let peripheral_count := (total_cities - (11 : ℕ)) in
    peripheral_count

theorem largest_number_peripheral_cities (neighbors : ℕ → set ℕ) :
  max_peripheral_cities 100 neighbors = 89 := sorry

end largest_number_peripheral_cities_l244_244768


namespace proof_problem_l244_244158

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244158


namespace find_value_l244_244064

theorem find_value (a b : ℝ) (h1 : 2 * a - 3 * b = 1) : 5 - 4 * a + 6 * b = 3 := 
by
  sorry

end find_value_l244_244064


namespace sugar_stored_in_room_l244_244582

-- Define constants
variables (S F B E C : ℝ)

-- Define the conditions
def condition1 : Prop := S / F = 5 / 2
def condition2 : Prop := F / B = 10 / 1
def condition3 : Prop := E / S = 3 / 4
def condition4 : Prop := C / F = 3 / 5
def condition5 : Prop := F / (B + 60) = 8 / 1
def condition6 : Prop := E / S = 5 / 6

-- The proof problem statement
theorem sugar_stored_in_room (h1 : condition1) 
                            (h2 : condition2) 
                            (h3 : condition3) 
                            (h4 : condition4) 
                            (h5 : condition5) 
                            (h6 : condition6) : 
                            S = 6000 :=
by
  sorry

end sugar_stored_in_room_l244_244582


namespace total_students_in_school_l244_244385

-- Definitions and conditions
def number_of_blind_students (B : ℕ) : Prop := ∃ B, 3 * B = 180
def number_of_other_disabilities (O : ℕ) (B : ℕ) : Prop := O = 2 * B
def total_students (T : ℕ) (D : ℕ) (B : ℕ) (O : ℕ) : Prop := T = D + B + O

theorem total_students_in_school : 
  ∃ (T B O : ℕ), number_of_blind_students B ∧ 
                 number_of_other_disabilities O B ∧ 
                 total_students T 180 B O ∧ 
                 T = 360 :=
by
  sorry

end total_students_in_school_l244_244385


namespace value_of_p_circle_tangent_to_directrix_l244_244137

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244137


namespace largest_possible_value_a_b_c_l244_244546

theorem largest_possible_value_a_b_c :
  ∃ (a b c : ℕ), 
  (0 ≤ a ∧ a ≤ 9) ∧ 
  (0 ≤ b ∧ b ≤ 9) ∧ 
  (0 ≤ c ∧ c ≤ 9) ∧ 
  ∃ (y : ℕ), 
    (0 < y ∧ y ≤ 50) ∧ 
    (0.abc = 1 / y) ∧
    ∃ (x : ℕ), 
      (0 < x ∧ x ≤ 9) ∧ 
      (0.yyy = 1 / x) ∧
      (a + b + c = 8) 
      sorry

end largest_possible_value_a_b_c_l244_244546


namespace investment_initial_amount_l244_244786

theorem investment_initial_amount (x : ℝ) (P : ℝ) (r : ℝ) (A : ℝ) (n t : ℕ) :
  r = 0.08 ∧ n = 1 ∧ t = 28 ∧ A = 16200 ∧ (112 / x) = 7.735 ∧ A = P * (1 + r / n) ^ (n * t) →
  P ≈ 1629.89 :=
by sorry

end investment_initial_amount_l244_244786


namespace quadratic_rewrite_l244_244701

noncomputable def a : ℕ := 6
noncomputable def b : ℕ := 6
noncomputable def c : ℕ := 284
noncomputable def quadratic_coeffs_sum : ℕ := a + b + c

theorem quadratic_rewrite :
  (∃ a b c : ℕ, 6 * (x : ℕ) ^ 2 + 72 * x + 500 = a * (x + b) ^ 2 + c) →
  quadratic_coeffs_sum = 296 := by sorry

end quadratic_rewrite_l244_244701


namespace keiko_ephraim_same_heads_l244_244600

def keiko_outcomes := {HH, HT, TH, TT}
def ephraim_outcomes := {HH, HT, TH, TT}

def prob_outcome (outcome : ℕ) : ℚ := 1 / 4
def prob_same_heads := 3 / 8

theorem keiko_ephraim_same_heads :
  let outcomes := keiko_outcomes × ephraim_outcomes
  let favorable_outcomes := {(TT, TT), (HH, HH)} ∪ ({(HT, HT), (HT, TH), (TH, HT), (TH, TH)} ∩ {(HT, HT), (TH, HT)})
  let p := ∑ x in favorable_outcomes, prob_outcome x.1 * prob_outcome x.2 in
  p = prob_same_heads :=
by sorry

end keiko_ephraim_same_heads_l244_244600


namespace find_f_neg1_l244_244618

-- Definitions based on conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable {b : ℝ} (f : ℝ → ℝ)

axiom odd_f : odd_function f
axiom f_form : ∀ x, 0 ≤ x → f x = 2^(x + 1) + 2 * x + b
axiom b_value : b = -2

theorem find_f_neg1 : f (-1) = -4 :=
sorry

end find_f_neg1_l244_244618


namespace find_a_and_b_l244_244430

theorem find_a_and_b (a b : ℝ) :
  let u := ![3, a, -10]
  let v := ![4, 6, b]
  vector.cross_product u v = ![0, 0, 12] →
  a = -15 ∧ b = -6 / 5 :=
by
  -- Definitions
  let u := ![3, a, -10]
  let v := ![4, 6, b]
  -- Given cross product definition
  have h : vector.cross_product u v = ![0, 0, 12] := sorry
  -- Calculations
  sorry


end find_a_and_b_l244_244430


namespace stuffed_animal_total_l244_244635

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l244_244635


namespace smallest_positive_period_minimum_value_symmetry_center_intervals_monotonically_increasing_l244_244497

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x - Real.pi / 6)

-- 1. Smallest positive period is π
theorem smallest_positive_period : ∃ T > 0, ∀ x, f(x + T) = f(x) ∧ T = Real.pi := sorry

-- 2. Minimum value of the function is -3
theorem minimum_value : ∃ x, f(x) = -3 := sorry

-- 3. Symmetry center of the graph of the function
theorem symmetry_center : ∃ k : ℤ, ∀ x, f(x) = f(2 * (x - (Real.pi / 12 + k * Real.pi / 2))) := sorry

-- 4. Intervals of monotonically increasing function
theorem intervals_monotonically_increasing : ∀ k : ℤ, ∀ x, x ∈ Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) → (∀ y ∈ Set.Icc x (Real.pi / 3 + k * Real.pi), f x ≤ f y) := sorry

end smallest_positive_period_minimum_value_symmetry_center_intervals_monotonically_increasing_l244_244497


namespace range_of_independent_variable_l244_244316

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y, y = x / (Real.sqrt (x + 4)) + 1 / (x - 1)) ↔ x > -4 ∧ x ≠ 1 := 
by
  sorry

end range_of_independent_variable_l244_244316


namespace points_of_tangency_collinear_l244_244248

variables {ℝ : Type} [linear_ordered_field ℝ] 
  (d1 d2 : line ℝ) (ω1 ω2 : circle ℝ) (T1 T2 T3 : point ℝ)

-- Defining the conditions
def parallel (l1 l2 : line ℝ) : Prop := 
  ∀ p q : point ℝ, p ∈ l1 → q ∈ l2 → collinear p q

def tangent (c : circle ℝ) (l : line ℝ) (p : point ℝ) : Prop :=
  p ∈ c ∧ p ∈ l ∧ ∀ (q : point ℝ), q ≠ p → q ∉ l

def tangent_to_each_other (c1 c2 : circle ℝ) (p : point ℝ) : Prop :=
  p ∈ c1 ∧ p ∈ c2 ∧ ∀ (r : point ℝ), r ≠ p → r ∉ c1 ∨ r ∉ c2

-- Statement of the problem
theorem points_of_tangency_collinear
  (h1 : parallel d1 d2)
  (h2 : ω1.tangent d1 T1)
  (h3 : ω2.tangent d2 T2)
  (h4 : tangent_to_each_other ω1 ω2 T3) :
  collinear T1 T2 T3 :=
sorry

end points_of_tangency_collinear_l244_244248


namespace sine_triangle_sides_l244_244602

variable {α β γ : ℝ}

-- Given conditions: α, β, γ are angles of a triangle.
def is_triangle_angles (α β γ : ℝ) : Prop :=
  α + β + γ = Real.pi ∧ 0 < α ∧ α < Real.pi ∧
  0 < β ∧ β < Real.pi ∧ 0 < γ ∧ γ < Real.pi

-- The proof statement: Prove that there exists a triangle with sides sin α, sin β, sin γ
theorem sine_triangle_sides (h : is_triangle_angles α β γ) :
  ∃ (x y z : ℝ), x = Real.sin α ∧ y = Real.sin β ∧ z = Real.sin γ ∧
  (x + y > z) ∧ (x + z > y) ∧ (y + z > x) := sorry

end sine_triangle_sides_l244_244602


namespace product_of_B_coords_l244_244911

structure Point where
  x : ℝ
  y : ℝ

def isMidpoint (M A B : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

theorem product_of_B_coords :
  ∀ (M A B : Point), 
  isMidpoint M A B →
  M = ⟨3, 7⟩ →
  A = ⟨5, 3⟩ →
  (B.x * B.y) = 11 :=
by intro M A B hM hM_def hA_def; sorry

end product_of_B_coords_l244_244911


namespace value_of_a2018_l244_244884

noncomputable def sequence : ℕ → ℝ := sorry

def delta (A : ℕ → ℝ) (n : ℕ) : ℝ := A (n + 1) - A n

def double_delta (A : ℕ → ℝ) (n : ℕ) : ℝ := delta (delta A) n

theorem value_of_a2018 (A : ℕ → ℝ) (h1 : ∀ n, double_delta A n = 1) 
                        (h2 : A 17 = 0) (h3 : A 2016 = 0) : A 2017 = 1000 := 
sorry

end value_of_a2018_l244_244884


namespace enrique_shreds_pages_l244_244863

theorem enrique_shreds_pages :
  ∃ n : ℕ, (n = 6 * 44) ∧ n = 264 :=
begin
  use 264,
  split,
  { simp },
  { refl },
end

end enrique_shreds_pages_l244_244863


namespace expansion_term_term_containing_inverse_square_sum_binomial_coefficient_and_n_l244_244929

theorem expansion_term (x : ℚ) :
  (2 * x^2 + 1 / x)^5 = ∑ i in (finset.range 6), i.choose 5 * (2 * x^2)^(5 - i) * (1 / x)^i :=
sorry

theorem term_containing_inverse_square (x : ℚ) :
  (2 * x^2 + 1 / x)^5 = ∑ i in (finset.range 6), i.choose 5 * (2 * x^2)^(5 - i) * (1 / x)^i 
  → exists j, (i.choose 5 * (2 * x^2)^(5 - j) * (1 / x)^j) = 10 / x^2 :=
sorry

theorem sum_binomial_coefficient_and_n :
  2 ^ 5 = 32
  ∧ (∃ n : ℕ, 32 = 4 * (n.choose 2) - 28)
  ∧ (∃ n : ℕ, n.choose 2 = 15 → n = 6) :=
sorry

end expansion_term_term_containing_inverse_square_sum_binomial_coefficient_and_n_l244_244929


namespace pentagon_centroid_area_ratio_l244_244125

noncomputable def centroid (A B C D : Point) : Point := (A + B + C + D) / 4

theorem pentagon_centroid_area_ratio (A B C D E : Point) :
  let G_A := centroid B C D E,
      G_B := centroid A C D E,
      G_C := centroid A B D E,
      G_D := centroid A B C E,
      G_E := centroid A B C D in
  area (Polygon.mk [G_A, G_B, G_C, G_D, G_E]) / area (Polygon.mk [A, B, C, D, E]) = 1 / 16 :=
by sorry

end pentagon_centroid_area_ratio_l244_244125


namespace original_price_of_cycle_l244_244376

theorem original_price_of_cycle (SP : ℝ) (P : ℝ) (loss_percent : ℝ) 
  (h_loss : loss_percent = 18) 
  (h_SP : SP = 1148) 
  (h_eq : SP = (1 - loss_percent / 100) * P) : 
  P = 1400 := 
by 
  sorry

end original_price_of_cycle_l244_244376


namespace two_digit_primes_count_l244_244518

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_two_digit_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

def digits := {3, 5, 7, 9}

def is_valid_two_digit_prime (n : ℕ) : Prop :=
  is_two_digit_number n ∧ is_prime n ∧ 
  ∃ t u : ℕ, t ∈ digits ∧ u ∈ digits ∧ t ≠ u ∧ n = t * 10 + u

theorem two_digit_primes_count : ∃ n : ℕ, n = 6 ∧ ∀ k : ℕ, is_valid_two_digit_prime k → k < 100 := 
sorry

end two_digit_primes_count_l244_244518


namespace exists_zero_point_in_interval_l244_244849

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x - 2 * x

theorem exists_zero_point_in_interval :
  ∃ c ∈ Set.Ioo 1 (Real.pi / 2), f c = 0 := 
sorry

end exists_zero_point_in_interval_l244_244849


namespace no_preimage_for_p_gt_1_l244_244694

def f (x : ℝ) : ℝ := -x^2 + 2 * x

theorem no_preimage_for_p_gt_1 (P : ℝ) (hP : P > 1) : ¬ ∃ x : ℝ, f x = P :=
sorry

end no_preimage_for_p_gt_1_l244_244694


namespace triangle_angles_in_given_ratio_l244_244311

theorem triangle_angles_in_given_ratio (x : ℝ) (y : ℝ) (z : ℝ) (h : x + y + z = 180) (r : x / 1 = y / 4 ∧ y / 4 = z / 7) : 
  x = 15 ∧ y = 60 ∧ z = 105 :=
by
  sorry

end triangle_angles_in_given_ratio_l244_244311


namespace problem_l244_244129

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244129


namespace ratio_area_square_EFGH_ABCD_l244_244124

theorem ratio_area_square_EFGH_ABCD (s : ℝ) (E F G H : ℝ × ℝ)
  (hE : E = (s / 2, -s * (sqrt 3 / 2))) (hF : F = ((3 / 2) * s, s / 2))
  (hG : G = (s / 2, (3 / 2) * s)) (hH : H = (-s * (sqrt 3 / 2), s / 2)) :
  let area_ABCD := s^2,
      side_EFGH := sqrt ((3 / 2 * s) ^ 2 + (s * (sqrt 3 / 2)) ^ 2),
      area_EFGH := side_EFGH^2 in
  (area_EFGH / area_ABCD) = 4.5 := 
by {
  sorry
}

end ratio_area_square_EFGH_ABCD_l244_244124


namespace new_average_doubled_marks_l244_244679

theorem new_average_doubled_marks (n : ℕ) (avg : ℕ) (h_n : n = 11) (h_avg : avg = 36) :
  (2 * avg * n) / n = 72 :=
by
  sorry

end new_average_doubled_marks_l244_244679


namespace magic_king_episodes_proof_l244_244711

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l244_244711


namespace jed_speed_l244_244563

theorem jed_speed
  (posted_speed_limit : ℕ := 50)
  (fine_per_mph_over_limit : ℕ := 16)
  (red_light_fine : ℕ := 75)
  (cellphone_fine : ℕ := 120)
  (parking_fine : ℕ := 50)
  (total_red_light_fines : ℕ := 2 * red_light_fine)
  (total_parking_fines : ℕ := 3 * parking_fine)
  (total_fine : ℕ := 1046)
  (non_speeding_fines : ℕ := total_red_light_fines + cellphone_fine + total_parking_fines)
  (speeding_fine : ℕ := total_fine - non_speeding_fines)
  (mph_over_limit : ℕ := speeding_fine / fine_per_mph_over_limit):
  (posted_speed_limit + mph_over_limit) = 89 :=
by
  sorry

end jed_speed_l244_244563


namespace nth_term_sequence_sum_sequence_l244_244900

open Nat

-- Define the sequence of sums S_n
def S (n : ℕ) : ℕ := n^2

-- Define the nth term of the sequence a_n
def a (n : ℕ) : ℕ := S n - S (n - 1)

-- Lemma to prove that a_n is 2n - 1
theorem nth_term_sequence (n : ℕ) : a n = 2 * n - 1 := by
  sorry

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 1 / ((a n) * (a (n + 1)))

-- Define the sum T_n of the first n terms of sequence b_n
def T (n : ℕ) : ℝ := ∑ i in range n, b i

-- Lemma to prove that T_n is n / (2n + 1)
theorem sum_sequence (n : ℕ) : T n = n / (2 * n + 1) := by
  sorry

end nth_term_sequence_sum_sequence_l244_244900


namespace product_primes_less_than_20_l244_244334

theorem product_primes_less_than_20 :
  (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 = 9699690) :=
by
  sorry

end product_primes_less_than_20_l244_244334


namespace parabola_conditions_l244_244246

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244246


namespace area_triangle_l244_244652

-- Define the ellipse and properties
variable (P : ℝ × ℝ)
variable (F1 F2 : ℝ × ℝ)
variable (a b c : ℝ)

-- Define the conditions
def on_ellipse : Prop := (P.2 ^ 2) / 5 + (P.1 ^ 2) / 4 = 1

def is_foci : Prop := (F1 = (0, c) ∧ F2 = (0, -c)) ∧ (c = 1) ∧ (a = Real.sqrt 5) ∧ (b = 2)

def angle_condition : Prop := ∠(F1, P, F2) = Real.pi / 6  -- 30 degrees in radians

-- Statement to prove the area
theorem area_triangle :
  on_ellipse P ∧ is_foci F1 F2 a b c ∧ angle_condition F1 P F2 →
  Real.abs (1/2 * Real.norm (P - F1) * Real.norm (P - F2) * Real.sin (Real.pi / 6)) = 8 - 4 * Real.sqrt 3 :=
by
  sorry

end area_triangle_l244_244652


namespace one_black_and_two_black_mutually_exclusive_but_not_complementary_l244_244888

def bag := {red, black} -- Define the colors available
def draw_balls (num : Nat) : Set (List bag) := 
  { list | list.length = num ∧ all elements of list ∈ bag }

def exactly_one_black (l : List bag) : Prop :=
  l.count black = 1

def exactly_two_black (l : List bag) : Prop :=
  l.count black = 2

def mutually_exclusive (P Q : List bag → Prop) : Prop :=
  ∀ (l : List bag), ¬ (P l ∧ Q l)

def complementary (P Q : List bag → Prop) : Prop :=
  ∀ (l : List bag), P l ∨ Q l

theorem one_black_and_two_black_mutually_exclusive_but_not_complementary:
  ∀ (bags : List bag), 
    bags ∈ draw_balls 2 →
    mutually_exclusive exactly_one_black exactly_two_black ∧ ¬ complementary exactly_one_black exactly_two_black :=
by
  intros bags h
  sorry

end one_black_and_two_black_mutually_exclusive_but_not_complementary_l244_244888


namespace right_triangle_squares_rectangles_l244_244610

theorem right_triangle_squares_rectangles
  (A B C : Type)
  [inner_product_space ℝ A]
  (triangle : affine.simplex ℝ A)
  (h_tri : triangle.vertices = ![B, C])
  (h_angle : ∠(triangle.vertex 1, triangle.vertex 0) = real.pi / 2) :
  ∃ n : ℕ, n = 9 :=
by
  sorry

end right_triangle_squares_rectangles_l244_244610


namespace root_exists_in_interval_l244_244852

noncomputable def f (x : ℝ) : ℝ := x^3 - x - 1

theorem root_exists_in_interval :
  ∃ c : ℝ, 1 < c ∧ c < 2 ∧ f c = 0 := 
sorry

end root_exists_in_interval_l244_244852


namespace complex_number_in_first_quadrant_l244_244003

noncomputable def z : ℂ := Complex.ofReal 1 + Complex.I

theorem complex_number_in_first_quadrant 
  (h : Complex.ofReal 1 + Complex.I = Complex.I / z) : 
  (0 < z.re ∧ 0 < z.im) :=
  sorry

end complex_number_in_first_quadrant_l244_244003


namespace value_of_p_circle_tangent_to_directrix_l244_244147

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244147


namespace parabola_focus_line_l244_244160

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244160


namespace parabola_condition_l244_244171

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244171


namespace perpendicular_lines_have_given_slope_l244_244081

theorem perpendicular_lines_have_given_slope (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 3 = 0 → x + (2 * k + 3) * y - 2 = 0) →
  k = -3 :=
by
  sorry

end perpendicular_lines_have_given_slope_l244_244081


namespace sum_of_first_2m_terms_l244_244317

variable {α : Type*} [LinearOrderedField α]

noncomputable def arithmeticSum (a d : α) (n : ℕ) : α :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_of_first_2m_terms 
  (a d : α) (m : ℕ) 
  (h₁ : arithmeticSum a d m = 30)
  (h₂ : arithmeticSum a d (3 * m) = 90)
  : arithmeticSum a d (2 * m) = 60 :=
  sorry

end sum_of_first_2m_terms_l244_244317


namespace largest_number_peripheral_cities_l244_244767

-- Define the conditions in Lean
def is_peripheral (city : ℕ) (neighbors : ℕ → set ℕ) (c1 c2 : ℕ) : Prop :=
  (c1 ≠ c2) ∧ (∀ p, p ∈ neighbors c1 -> p ≠ c2) ∧ (∀ n, n < 10 → ¬ connected_within_n_steps neighbors c1 c2 n)

def connected_within_n_steps (neighbors : ℕ → set ℕ) (c1 c2 : ℕ) (n : ℕ) : Prop :=
  ∃ (path : list ℕ), (length path) ≤ n ∧ path.head = c1 ∧ path.last = some c2 ∧ ∀ i ∈ path, i ∈ neighbors (path.nth_le i sorry)

def max_peripheral_cities (total_cities : ℕ) (neighbors : ℕ → set ℕ) : ℕ :=
  if total_cities ≠ 100 ∨ ∃ c1 c2, c1 ≠ c2 ∧ ¬ connected_within_n_steps neighbors c1 c2 11 then 0 else
    let peripheral_count := (total_cities - (11 : ℕ)) in
    peripheral_count

theorem largest_number_peripheral_cities (neighbors : ℕ → set ℕ) :
  max_peripheral_cities 100 neighbors = 89 := sorry

end largest_number_peripheral_cities_l244_244767


namespace number_of_integer_solutions_l244_244456

theorem number_of_integer_solutions :
  ∃ (n : ℕ), 
  (∀ (x y : ℤ), 2 * x + 3 * y = 7 ∧ 5 * x + n * y = n ^ 2) ∧
  (n = 8) := 
sorry

end number_of_integer_solutions_l244_244456


namespace stuffed_animals_total_l244_244641

theorem stuffed_animals_total :
  let McKenna := 34
  let Kenley := 2 * McKenna
  let Tenly := Kenley + 5
  McKenna + Kenley + Tenly = 175 :=
by
  sorry

end stuffed_animals_total_l244_244641


namespace color_of_last_bead_is_white_l244_244111

-- Defining the pattern of the beads
inductive BeadColor
| White
| Black
| Red

open BeadColor

-- Define the repeating pattern of the beads
def beadPattern : ℕ → BeadColor
| 0 => White
| 1 => Black
| 2 => Black
| 3 => Red
| 4 => Red
| 5 => Red
| (n + 6) => beadPattern n

-- Define the total number of beads
def totalBeads : ℕ := 85

-- Define the position of the last bead
def lastBead : ℕ := totalBeads - 1

-- Proving the color of the last bead
theorem color_of_last_bead_is_white : beadPattern lastBead = White :=
by
  sorry

end color_of_last_bead_is_white_l244_244111


namespace part1_part2_l244_244023

noncomputable theory

def f (x a: ℝ) := x^3 - a*x^2 - 9*x + 1

theorem part1 (a : ℝ) (ha : deriv (λ x : ℝ, f x a) 3 = 0) : a = 3 :=
by
  have hf_deriv : deriv (λ x, f x a) x = 3*x^2 - 2*a*x - 9 := sorry,
  -- Use the given extremum condition to solve for a
  sorry

theorem part2 :
  ∃ x_max x_min : ℝ, (x_max ∈ Set.Icc (-2 : ℝ) 0 ∧ x_min ∈ Set.Icc (-2 : ℝ) 0 ∧ 
    f x_max 3 = 6 ∧ f x_min 3 = -1) :=
by
  let f3 := f 3,
  have hf3 : deriv f3 = λ x, 3 * x^2 - 6 * x - 9 := sorry,
  -- Evaluation of critical points and endpoints within the interval [-2, 0]
  sorry

end part1_part2_l244_244023


namespace unit_vector_collinear_with_a_l244_244746

-- Given vector a
def a : ℝ × ℝ × ℝ := (3, 0, -4)

-- Define vector option D
def option_d : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

-- Define the condition for collinearity
def collinear (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * (v.1), k * (v.2), k * (v.3))

-- Define the condition for unit vector
def is_unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 + v.3^2 = 1

-- The main theorem statement
theorem unit_vector_collinear_with_a : 
  is_unit_vector option_d ∧ collinear option_d a :=
sorry

end unit_vector_collinear_with_a_l244_244746


namespace parabola_focus_line_tangent_circle_l244_244204

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244204


namespace prove_p_equals_2_l244_244194

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244194


namespace pyramid_volume_l244_244810

noncomputable def volume_of_pyramid (length width alt corner_edge : ℝ) :=
  (1 / 3) * (length * width) * (Real.sqrt (corner_edge ^ 2 - (Real.sqrt (length ^ 2 + width ^ 2) / 2) ^ 2))

theorem pyramid_volume :
  volume_of_pyramid 7 9  13.87 15 ≈ 291 := 
begin
  have result : volume_of_pyramid 7 9 13.87 15 = 
    (1 / 3) * (7 * 9) * (Real.sqrt (15 ^ 2 - (Real.sqrt (7 ^ 2 + 9 ^ 2) / 2) ^ 2)),
  exact volume_of_pyramid 7 9 13.87 15,
  linarith
end

end pyramid_volume_l244_244810


namespace cos_30_deg_l244_244340

-- The condition implicitly includes the definition of cosine and the specific angle value

theorem cos_30_deg : cos (Real.pi / 6) = Real.sqrt 3 / 2 :=
by sorry

end cos_30_deg_l244_244340


namespace find_natrual_numbers_l244_244444

theorem find_natrual_numbers (k n : ℕ) (A B : Matrix (Fin n) (Fin n) ℤ) 
  (h1 : k ≥ 1) 
  (h2 : n ≥ 2) 
  (h3 : A ^ 3 = 0) 
  (h4 : A ^ k * B + B * A = 1) : 
  k = 1 ∧ Even n := 
sorry

end find_natrual_numbers_l244_244444


namespace NoSuchMatrices_l244_244777

open Matrix

def SL2Z : set (Matrix (Fin 2) (Fin 2) ℤ) := {A | det A = 1}

noncomputable def ProblemPartA : Prop :=
  ∀ (A B C : Matrix (Fin 2) (Fin 2) ℤ), A ∈ SL2Z → B ∈ SL2Z → C ∈ SL2Z → A^2 + B^2 = C^2 → False

noncomputable def ProblemPartB : Prop :=
  ∀ (A B C : Matrix (Fin 2) (Fin 2) ℤ), A ∈ SL2Z → B ∈ SL2Z → C ∈ SL2Z → A^4 + B^4 = C^4 → False

-- A Lean statement that encapsulates the equivalence proofs for both parts (a) and (b)
theorem NoSuchMatrices : ProblemPartA ∧ ProblemPartB := by
  sorry

end NoSuchMatrices_l244_244777


namespace x0_eq_x6_count_l244_244472

def x_next (x : ℝ) : ℝ :=
if 3 * x < 1 then 3 * x
else if 3 * x < 2 then 3 * x - 1
else 3 * x - 2

def x_seq (x0 : ℝ) (n : ℕ) : ℝ :=
nat.iterate x_next n x0

noncomputable def ternary_representations_satisfying_condition : ℕ :=
@Fintype.card {x0 : ℝ // 0 ≤ x0 ∧ x0 < 1 ∧ x_seq x0 6 = x0} (by sorry)

theorem x0_eq_x6_count : ternary_representations_satisfying_condition = 729 :=
by sorry

end x0_eq_x6_count_l244_244472


namespace parabola_focus_line_tangent_circle_l244_244203

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244203


namespace f_2202_minus_f_2022_l244_244625

-- Definitions and conditions
def f : ℕ+ → ℕ+ := sorry -- The exact function is provided through conditions and will be proven property-wise.

axiom f_increasing {a b : ℕ+} : a < b → f a < f b
axiom f_range (n : ℕ+) : ∃ m : ℕ+, f n = ⟨m, sorry⟩ -- ensuring f maps to ℕ+
axiom f_property (n : ℕ+) : f (f n) = 3 * n

-- Prove the statement
theorem f_2202_minus_f_2022 : f 2202 - f 2022 = 1638 :=
by sorry

end f_2202_minus_f_2022_l244_244625


namespace proof_problem_l244_244152

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244152


namespace arithmetic_seq_inv_arithmetic_a_general_term_a_l244_244102

noncomputable def sequence_a (n : ℕ) : ℕ → ℝ
| 0       := 1
| (n + 1) := let an := sequence_a n in
              (aₙ : ℝ) / 3 * 2 -- using the recurrence relation

theorem arithmetic_seq (n : ℕ) (h : n ≥ 1) :
  3 * sequence_a(n) * sequence_a(n-1) + sequence_a(n) - sequence_a(n-1) = 0 :=
sorry

theorem inv_arithmetic_a : ∀ n ≥ 1, (1 : ℝ) / sequence_a n = 3 * n - 2 :=
sorry

theorem general_term_a (n : ℕ) (h : n ≥ 1) : sequence_a(n) = 1 / (3 * n - 2) :=
sorry

end arithmetic_seq_inv_arithmetic_a_general_term_a_l244_244102


namespace prove_p_equals_2_l244_244196

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244196


namespace sum_remainder_zero_l244_244742

theorem sum_remainder_zero
  (a b c : ℕ)
  (h₁ : a % 53 = 31)
  (h₂ : b % 53 = 15)
  (h₃ : c % 53 = 7) :
  (a + b + c) % 53 = 0 :=
by
  sorry

end sum_remainder_zero_l244_244742


namespace threeS_technology_correct_usage_l244_244399

-- Definitions based on the conditions provided
def RS_dynamic_population_info : Prop := "RS can obtain dynamic information about population growth"
def RS_dynamic_flood_info : Prop := "RS can obtain dynamic information about flooded areas"
def GPS_location_cash : Prop := "GPS can determine the location of cash transport vehicles"
def GIS_remote_sensing_fires : Prop := "GIS can obtain remote sensing information of forest fires"

-- The theorem to prove
theorem threeS_technology_correct_usage (COND1 : RS_dynamic_population_info) 
                                        (COND2 : RS_dynamic_flood_info)
                                        (COND3 : GPS_location_cash)
                                        (COND4 : GIS_remote_sensing_fires) :
  COND2 ∧ COND3 :=
by sorry

end threeS_technology_correct_usage_l244_244399


namespace new_boarders_joined_l244_244702

theorem new_boarders_joined (boarders_initial day_students_initial boarders_final x : ℕ)
  (h1 : boarders_initial = 220)
  (h2 : (5:ℕ) * day_students_initial = (12:ℕ) * boarders_initial)
  (h3 : day_students_initial = 528)
  (h4 : (1:ℕ) * day_students_initial = (2:ℕ) * (boarders_initial + x)) :
  x = 44 := by
  sorry

end new_boarders_joined_l244_244702


namespace proof_problem_l244_244153

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244153


namespace extremum_points_l244_244109

noncomputable def f (x1 x2 : ℝ) : ℝ := x1 * x2 / (1 + x1^2 * x2^2)

theorem extremum_points :
  (f 0 0 = 0) ∧
  (∀ x1 : ℝ, f x1 (-1 / x1) = -1 / 2) ∧
  (∀ x1 : ℝ, f x1 (1 / x1) = 1 / 2) ∧
  ∀ y1 y2 : ℝ, (f 0 0 < f y1 y2 → (0 < y1 ∧ 0 < y2)) ∧ 
             (f 0 0 > f y1 y2 → (0 > y1 ∧ 0 > y2)) :=
by
  sorry

end extremum_points_l244_244109


namespace parabola_condition_l244_244180

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244180


namespace initial_machines_l244_244787

theorem initial_machines (x : ℝ) : ∃ N : ℝ, (∀ r : ℝ, N * r = x / 8 ∧ 30 * r = 3 * x / 4) ∧ N = 5 :=
by
  use 5
  intro r
  split
  sorry
  sorry

end initial_machines_l244_244787


namespace students_move_bricks_l244_244954

variable (a b c : ℕ)

theorem students_move_bricks (h : a * b * c ≠ 0) : 
  (by let efficiency := (c : ℚ) / (a * b);
      let total_work := (a : ℚ);
      let required_time := total_work / efficiency;
      exact required_time = (a^2 * b) / (c^2)) := sorry

end students_move_bricks_l244_244954


namespace find_angle_between_slant_height_and_height_of_cone_l244_244391

-- Define a structure for 3D points
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Define the conditions given
def D := Point3D -- Denote the apex of the cone
def A := Point3D -- Denote one base vertex of the cone
def B := Point3D -- Denote second base vertex of the cone
def C := Point3D -- Denote third base vertex of the cone
def O := Point3D -- Foot of the right-perpendicular from D to the plane ABC

-- Define perpendicularity and the angle calculation as given in conditions
axiom AD_perp_BD : ∀ A D B : Point3D, ⟪A, D⟫ ⟂ ⟪B, D⟫
axiom AD_perp_CD : ∀ A D C : Point3D, ⟪A, D⟫ ⟂ ⟪C, D⟫
axiom BD_perp_CD : ∀ B D C : Point3D, ⟪B, D⟫ ⟂ ⟪C, D⟫
axiom DO_perp_ABC : ∀ D O A B C : Point3D, ⟪D, O⟫ ⟂ Plane[A, B, C]

-- Define the target angle result
def target_angle : ℝ := Real.arcsin(Real.sqrt 6 / 3)

-- Main theorem statement
theorem find_angle_between_slant_height_and_height_of_cone :
  (angle (Ray OA) (Ray DA) = target_angle) :=
  sorry

end find_angle_between_slant_height_and_height_of_cone_l244_244391


namespace ott_fraction_l244_244268

/-- 
Moe, Loki, Nick, and Pat each give $2 to Ott.
Moe gave Ott one-seventh of his money.
Loki gave Ott one-fifth of his money.
Nick gave Ott one-fourth of his money.
Pat gave Ott one-sixth of his money.
-/
def fraction_of_money_ott_now_has (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) : Prop :=
  A = 14 ∧ B = 10 ∧ C = 8 ∧ D = 12 ∧ (2 * (1 / 7 : ℚ)) = 2 ∧ (2 * (1 / 5 : ℚ)) = 2 ∧ (2 * (1 / 4 : ℚ)) = 2 ∧ (2 * (1 / 6 : ℚ)) = 2

theorem ott_fraction (A : ℕ) (B : ℕ) (C : ℕ) (D : ℕ) (h : fraction_of_money_ott_now_has A B C D) : 
  8 = (2 / 11 : ℚ) * (A + B + C + D) :=
by sorry

end ott_fraction_l244_244268


namespace solution_set_f_f_x_le_3_l244_244034

def f (x : ℝ) : ℝ :=
if x ≥ 0 then (-x^2) else (x^2 + 2*x)

theorem solution_set_f_f_x_le_3 :
  {x : ℝ | f (f x) ≤ 3} = Set.Iic (Real.sqrt 3) :=
by sorry

end solution_set_f_f_x_le_3_l244_244034


namespace intersection_of_sets_is_closed_interval_l244_244630

noncomputable def A := {x : ℝ | x ≤ 0 ∨ x ≥ 2}
noncomputable def B := {x : ℝ | x < 1}

theorem intersection_of_sets_is_closed_interval :
  A ∩ B = {x : ℝ | x ≤ 0} :=
sorry

end intersection_of_sets_is_closed_interval_l244_244630


namespace least_possible_faces_l244_244726

def number_of_faces : ℕ → ℕ → ℕ 
| a b := a + b

theorem least_possible_faces (a b : ℕ) (h1 : a ≥ 6) (h2 : b ≥ 6)
  (h3 : (nat.choose 2 4 + nat.choose 3 3 + nat.choose 4 2 + nat.choose 5 1) = 5)
  (h4 : (nat.choose 5 5 + nat.choose 6 4 + nat.choose 7 3 + nat.choose 8 2 + nat.choose 9 1
          + nat.choose 10 0) = 12)
  (h5 : ∃ k, k = 12 ∧ (nat.choose 6 8 + nat.choose 7 7 + nat.choose 8 6 + nat.choose 9 5
          + nat.choose 10 4 + nat.choose 11 3 + nat.choose 12 2 + nat.choose 13 1 + nat.choose 14 0) = k)
  (h6 : ∀ n, (n.to_nat * (15 * b)) = 4 * a ): number_of_faces a b = 27 := 
sorry

end least_possible_faces_l244_244726


namespace true_statements_count_l244_244773

variable {ℝ : Type*} [LinearOrderedField ℝ]

section
variable (f : ℝ → ℝ)

def condition1 : Prop := ∀ x : ℝ, f(x + 2) + f(2 - x) = 4 → (∃ x, f(x) = f(4 - x))

def condition2 : Prop := ∀ x : ℝ, f(x + 2) = f(2 - x) → ∀ x : ℝ, f(x) = f(4 - x)

def condition3 : Prop := ∀ x : ℝ, (f(x - 2) = f(4 - (x - 2))) ∧ (f(-x + 2) = f(4 - (-x + 2)))

theorem true_statements_count : condition1 f ∧ condition2 f ∧ condition3 f → (count_true_statements f = 3) :=
sorry
end

end true_statements_count_l244_244773


namespace parabola_condition_l244_244177

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244177


namespace product_of_local_and_absolute_value_l244_244449

def localValue (n : ℕ) (digit : ℕ) : ℕ :=
  match n with
  | 564823 =>
    match digit with
    | 4 => 4000
    | _ => 0 -- only defining for digit 4 as per problem
  | _ => 0 -- only case for 564823 is considered

def absoluteValue (x : ℤ) : ℤ := if x < 0 then -x else x

theorem product_of_local_and_absolute_value:
  localValue 564823 4 * absoluteValue 4 = 16000 :=
by
  sorry

end product_of_local_and_absolute_value_l244_244449


namespace tetrahedron_volume_0_l244_244881

def point := ℝ × ℝ × ℝ

def coplanar (p1 p2 p3 p4: point): Prop :=
  ∃ (a b c d: ℝ), ∀ (p: point), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 → (a * p.1 + b * p.2 + c * p.3 = d)

def x_plus_y_eq_z_plus_3_plane (p: point): Prop := 
  p.1 + p.2 = p.3 + 3

def volume_of_tetrahedron_is_zero (p1 p2 p3 p4: point): Prop :=
  ∀ {a b c d: ℝ}, (∀ (p: point), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 → (a * p.1 + b * p.2 + c * p.3 = d)) → 0 = 0

theorem tetrahedron_volume_0 : 
  volume_of_tetrahedron_is_zero (5, 8, 10) (10, 10, 17) (4, 45, 46) (2, 5, 4) :=
begin
  sorry
end

end tetrahedron_volume_0_l244_244881


namespace proof_problem_l244_244151

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244151


namespace vector_D_collinear_with_a_l244_244752

def is_collinear (a b : ℝ × ℝ × ℝ) : Prop :=
∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)

def vector_a : ℝ × ℝ × ℝ := (3, 0, -4)

def vector_D : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

theorem vector_D_collinear_with_a : is_collinear vector_a vector_D :=
sorry

end vector_D_collinear_with_a_l244_244752


namespace probability_heads_and_3_l244_244349

noncomputable def biased_coin_heads_prob : ℝ := 0.4
def die_sides : ℕ := 8

theorem probability_heads_and_3 : biased_coin_heads_prob * (1 / die_sides) = 0.05 := sorry

end probability_heads_and_3_l244_244349


namespace balls_in_boxes_l244_244538

theorem balls_in_boxes (b : ℕ) (k : ℕ) : (b = 6) → (k = 3) → (k^6 = 729) :=
begin
  intros hb hk,
  rw [hb, hk],
  norm_num,
end

end balls_in_boxes_l244_244538


namespace flux_through_torus_l244_244843

def vector_field (x y z : ℝ) : ℝ × ℝ × ℝ := (4 * x, -y, z)

def torus_volume (R1 R2 : ℝ) : ℝ := (Real.pi^2 / 4) * (R2 - R1)^2 * (R1 + R2)

theorem flux_through_torus (R1 R2 : ℝ) (R1_pos : 0 < R1) (R2_pos : R1 < R2) :
  let flux := Real.pi^2 * (R2 - R1)^2 * (R1 + R2) in
  (∫∫∫ (fun x y z => 4) in {v : ℝ × ℝ × ℝ | is_inside_torus v R1 R2 }) = flux :=
sorry

end flux_through_torus_l244_244843


namespace max_at_zero_l244_244690

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem max_at_zero : ∃ x, (∀ y, f y ≤ f x) ∧ x = 0 :=
by 
  sorry

end max_at_zero_l244_244690


namespace vertex_at_fixed_point_l244_244365

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 + 1

theorem vertex_at_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a 1 = 2 :=
by
  sorry

end vertex_at_fixed_point_l244_244365


namespace fibonacci_150_mod_9_l244_244294

def fibonacci : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 2) := fibonacci (n + 1) + fibonacci n

theorem fibonacci_150_mod_9 :
  fibonacci 150 % 9 = 8 :=
sorry

end fibonacci_150_mod_9_l244_244294


namespace custom_star_division_l244_244432

def custom_star (a b : ℝ) : ℝ :=
  if a > b then a else
  if a = b then 1 else b

theorem custom_star_division : 
  (custom_star 1.1 (7/3) - custom_star (1/3) 0.1) / custom_star (4/5) 0.8 = 2 :=
by
  sorry

end custom_star_division_l244_244432


namespace cos_alpha_gt_cos_beta_l244_244483

-- Define the angles in the context of the second quadrant
def in_second_quadrant (α β : ℝ) : Prop :=
  α ∈ set.Ioo (π / 2) π ∧ β ∈ set.Ioo (π / 2) π

-- Define the given conditions
def conditions (α β : ℝ) : Prop :=
  in_second_quadrant α β ∧ sin α > sin β

-- State the theorem to prove
theorem cos_alpha_gt_cos_beta {α β : ℝ} (h : conditions α β) : cos α > cos β :=
  sorry

end cos_alpha_gt_cos_beta_l244_244483


namespace volume_of_extended_parallelepiped_l244_244429

-- Define the dimensions of the rectangular parallelepiped
def dimensions : ℝ × ℝ × ℝ := (2, 3, 4)

-- Define the conditions
def is_positive_int (x : ℚ) : Prop := x.denom = 1

-- The volume calculation, as given by the problem
def volume (a b c : ℝ) : ℝ := a * b * c

def total_volume_with_extension (a b c : ℝ) : ℝ :=
  let main_volume := volume a b c
  let external_parallelepipeds := 2 * (a * b) * 1 + 2 * (a * c) * 1 + 2 * (b * c) * 1
  let eighth_spheres_volume := 8 * (1 / 8 * (4 / 3) * Real.pi)
  let quarter_cylinders_volume := 3 * (2 * Real.pi) + 3 * (b * Real.pi) + 3 * (c * Real.pi)
  main_volume + external_parallelepipeds + eighth_spheres_volume + quarter_cylinders_volume

-- The proof statement
theorem volume_of_extended_parallelepiped :
  let a := 2
  let b := 3
  let c := 4
  let volume_total := total_volume_with_extension a b c
  volume_total = (228 + 31 * Real.pi) / 3 ∧ is_positive_int 228 ∧
  is_positive_int 31 ∧ is_positive_int 3 ∧ Int.gcd 31 3 = 1 :=
by
  sorry

end volume_of_extended_parallelepiped_l244_244429


namespace identical_sets_l244_244822

def A : Set ℝ := {x : ℝ | ∃ y : ℝ, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {(x, y) : ℝ × ℝ | y = x^2 + 1}
def D : Set ℝ := {y : ℝ | 1 ≤ y}

theorem identical_sets : B = D :=
by
  sorry

end identical_sets_l244_244822


namespace num_valid_two_digit_primes_l244_244536

-- Define the set from which the digits are chosen
def digit_set := {3, 5, 7, 9}

-- Define a function to check if a number is a two-digit prime formed by different tens and units digits from digit_set
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ {37, 53, 59, 73, 79, 97} -- Set of prime numbers obtained in the solution

-- Define the main theorem
theorem num_valid_two_digit_primes : (set.filter is_valid_prime { n | ∃ t u, t ≠ u ∧ t ∈ digit_set ∧ u ∈ digit_set ∧ n = 10 * t + u }).card = 7 := 
by
  sorry

end num_valid_two_digit_primes_l244_244536


namespace range_of_x_l244_244467

def A (x : ℝ) : ℤ := ⌈x⌉ -- Define A(x) as the smallest integer not less than x

theorem range_of_x (x : ℝ) (h : A (2 * x + 1) = 3) : (1 / 2 < x ∧ x ≤ 1) :=
by
  have h1 : 2 < 2 * x + 1 ∧ 2 * x + 1 ≤ 3 := sorry
  sorry

end range_of_x_l244_244467


namespace polynomial_degree_l244_244808

-- Define the roots and necessary properties
def roots : List ℂ := [1 + sqrt 2, 2 + sqrt 3, 3 + sqrt 5] ++ 
                      List.range 1 501 |>.map (λ n => n + sqrt (n + 1)) ++
                      List.range 1 501 |>.map (λ n => n + sqrt (n + 2)) ++
                      [sqrt 8, -sqrt 8]

-- Assert the polynomial degree
theorem polynomial_degree (P : Polynomial ℚ) :
  (∀ x ∈ roots, P.eval x = 0) → P.degree.to_nat = 1960 :=
by sorry

end polynomial_degree_l244_244808


namespace max_peripheral_cities_l244_244771

-- Defining the conditions
def num_cities := 100
def max_transfers := 11
def unique_paths := true
def is_peripheral (A B : ℕ) (f : ℕ → ℕ → bool) : Prop := (f A B = false ∧ f B A = false ∧ 
                                                            A ≠ B ∧ ∀ k < max_transfers, f A B = false)

-- Mathematical proof problem statement
theorem max_peripheral_cities (f : ℕ → ℕ → bool) 
  (H1 : ∀ A B, A ≠ B → ∃ p, length p ≤ max_transfers ∧ unique_paths ∧ f A B = true) :
  ∃ x, num_cities - x = 89 ∧ (∀ A B, is_peripheral A B f → x < num_cities) :=
sorry

end max_peripheral_cities_l244_244771


namespace solve_abs_inequality_l244_244876

theorem solve_abs_inequality (x : ℝ) (h : abs ((8 - x) / 4) < 3) : -4 < x ∧ x < 20 := 
  sorry

end solve_abs_inequality_l244_244876


namespace find_xy_l244_244543

theorem find_xy (x y : ℤ) 
  (h1 : 8^x / 4^(x + y) = 16) 
  (h2 : 27^(x + y) / 9^(4 * y) = 729) : 
  x * y = 48 := 
sorry

end find_xy_l244_244543


namespace integral_area_eq_one_fourth_l244_244839

-- Define the function f(x) = x / (x^2 + 1)^2
def f (x : ℝ) : ℝ := x / (x^2 + 1)^2

-- State the theorem about the definite integral
theorem integral_area_eq_one_fourth : ∫ x in 0..1, f x = 1 / 4 := by
  sorry

end integral_area_eq_one_fourth_l244_244839


namespace mike_total_spending_is_correct_l244_244642

-- Definitions for the costs of the items
def cost_marbles : ℝ := 9.05
def cost_football : ℝ := 4.95
def cost_baseball : ℝ := 6.52
def cost_toy_car : ℝ := 3.75
def cost_puzzle : ℝ := 8.99
def cost_stickers : ℝ := 1.25

-- Definitions for the discounts
def discount_puzzle : ℝ := 0.15
def discount_toy_car : ℝ := 0.10

-- Definition for the coupon
def coupon_amount : ℝ := 5.00

-- Total spent by Mike on toys
def total_spent : ℝ :=
  cost_marbles + 
  cost_football + 
  cost_baseball + 
  (cost_toy_car - cost_toy_car * discount_toy_car) + 
  (cost_puzzle - cost_puzzle * discount_puzzle) + 
  cost_stickers - 
  coupon_amount

-- Proof statement
theorem mike_total_spending_is_correct : 
  total_spent = 27.7865 :=
by
  sorry

end mike_total_spending_is_correct_l244_244642


namespace Pythagorean_triple_l244_244394

theorem Pythagorean_triple (n : ℕ) (hn : n % 2 = 1) (hn_geq : n ≥ 3) :
  n^2 + ((n^2 - 1) / 2)^2 = ((n^2 + 1) / 2)^2 := by
  sorry

end Pythagorean_triple_l244_244394


namespace unit_vector_collinear_with_a_l244_244747

-- Given vector a
def a : ℝ × ℝ × ℝ := (3, 0, -4)

-- Define vector option D
def option_d : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

-- Define the condition for collinearity
def collinear (u v : ℝ × ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * (v.1), k * (v.2), k * (v.3))

-- Define the condition for unit vector
def is_unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  v.1^2 + v.2^2 + v.3^2 = 1

-- The main theorem statement
theorem unit_vector_collinear_with_a : 
  is_unit_vector option_d ∧ collinear option_d a :=
sorry

end unit_vector_collinear_with_a_l244_244747


namespace minimum_shirts_for_saving_money_l244_244393

-- Define the costs for Acme and Gamma
def acme_cost (x : ℕ) : ℕ := 60 + 10 * x
def gamma_cost (x : ℕ) : ℕ := 15 * x

-- Prove that the minimum number of shirts x for which a customer saves money by using Acme is 13
theorem minimum_shirts_for_saving_money : ∃ (x : ℕ), 60 + 10 * x < 15 * x ∧ x = 13 := by
  sorry

end minimum_shirts_for_saving_money_l244_244393


namespace largest_number_peripheral_cities_l244_244769

-- Define the conditions in Lean
def is_peripheral (city : ℕ) (neighbors : ℕ → set ℕ) (c1 c2 : ℕ) : Prop :=
  (c1 ≠ c2) ∧ (∀ p, p ∈ neighbors c1 -> p ≠ c2) ∧ (∀ n, n < 10 → ¬ connected_within_n_steps neighbors c1 c2 n)

def connected_within_n_steps (neighbors : ℕ → set ℕ) (c1 c2 : ℕ) (n : ℕ) : Prop :=
  ∃ (path : list ℕ), (length path) ≤ n ∧ path.head = c1 ∧ path.last = some c2 ∧ ∀ i ∈ path, i ∈ neighbors (path.nth_le i sorry)

def max_peripheral_cities (total_cities : ℕ) (neighbors : ℕ → set ℕ) : ℕ :=
  if total_cities ≠ 100 ∨ ∃ c1 c2, c1 ≠ c2 ∧ ¬ connected_within_n_steps neighbors c1 c2 11 then 0 else
    let peripheral_count := (total_cities - (11 : ℕ)) in
    peripheral_count

theorem largest_number_peripheral_cities (neighbors : ℕ → set ℕ) :
  max_peripheral_cities 100 neighbors = 89 := sorry

end largest_number_peripheral_cities_l244_244769


namespace parabola_condition_l244_244172

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244172


namespace diagonal_length_of_cuboid_l244_244792

theorem diagonal_length_of_cuboid
  (a b c : ℝ)
  (h1 : a * b = Real.sqrt 2)
  (h2 : b * c = Real.sqrt 3)
  (h3 : c * a = Real.sqrt 6) : 
  Real.sqrt (a^2 + b^2 + c^2) = Real.sqrt 6 := 
sorry

end diagonal_length_of_cuboid_l244_244792


namespace p_eq_two_circle_tangent_proof_l244_244216

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244216


namespace percentage_both_colors_l244_244351

theorem percentage_both_colors :
  ∀ (C : ℕ), C % 2 = 0 → 
  (0.60 * C + 0.45 * C - x = C) → 
  (x = 0.05 * C) :=
by 
  sorry

end percentage_both_colors_l244_244351


namespace student_selection_l244_244086

theorem student_selection : 
  let first_year := 4
  let second_year := 5
  let third_year := 4
  (first_year * second_year) + (first_year * third_year) + (second_year * third_year) = 56 := by
  let first_year := 4
  let second_year := 5
  let third_year := 4
  sorry

end student_selection_l244_244086


namespace total_distance_traveled_l244_244275

theorem total_distance_traveled:
  let speed1 := 30
  let time1 := 4
  let speed2 := 35
  let time2 := 5
  let speed3 := 25
  let time3 := 6
  let total_time := 20
  let time1_3 := time1 + time2 + time3
  let time4 := total_time - time1_3
  let speed4 := 40

  let distance1 := speed1 * time1
  let distance2 := speed2 * time2
  let distance3 := speed3 * time3
  let distance4 := speed4 * time4

  let total_distance := distance1 + distance2 + distance3 + distance4

  total_distance = 645 :=
  sorry

end total_distance_traveled_l244_244275


namespace problem_statement_l244_244915

-- Define the function f and its properties
def f : ℝ → ℝ := sorry

axiom odd_f : ∀ x, f(-x) = -f(x)
axiom periodic_f : ∀ x, f(x + 2) = -f(x)
axiom f_on_interval : ∀ x : ℝ, 0 < x ∧ x < 2 → f(x) = 2 * x^2

-- Lean statement to prove
theorem problem_statement : f 2011 = -2 :=
by sorry

end problem_statement_l244_244915


namespace distinct_x_intercepts_count_l244_244513

theorem distinct_x_intercepts_count : 
  ∃ (s : Finset ℂ), (y : ℂ → ℂ) (H : y = λ x, (x-5)*(x^2 - x - 6)), s.card = 3 ∧ (∀ x : ℂ, y x = 0 ↔ x ∈ s) :=
sorry

end distinct_x_intercepts_count_l244_244513


namespace n_fraction_of_sum_l244_244757

theorem n_fraction_of_sum (l : List ℝ) (h1 : l.length = 21) (n : ℝ) (h2 : n ∈ l)
  (h3 : ∃ m, l.erase n = m ∧ m.length = 20 ∧ n = 4 * (m.sum / 20)) :
  n = (l.sum) / 6 :=
by
  sorry

end n_fraction_of_sum_l244_244757


namespace alice_meets_john_time_l244_244118

-- Definitions according to conditions
def john_speed : ℝ := 4
def bob_speed : ℝ := 6
def alice_speed : ℝ := 3
def initial_distance_alice_john : ℝ := 2

-- Prove the required meeting time
theorem alice_meets_john_time : 2 / (john_speed + alice_speed) * 60 = 17 := 
by
  sorry

end alice_meets_john_time_l244_244118


namespace number_of_trivial_proper_subsets_of_A_l244_244807

def is_trivial_set (S : Finset ℕ) : Prop :=
  (∑ x in S, x^2) % 2 = 1

def A : Finset ℕ := Finset.range 2017 

theorem number_of_trivial_proper_subsets_of_A :
  ∃ n : ℕ, n = (2^2016 - 1) ∧
  (∃! S ⊆ A, S ≠ A ∧ S ≠ ∅ ∧ is_trivial_set S) : 
  ∃ (number_of_trivial_proper_subsets : ℕ), number_of_trivial_proper_subsets = 2^2016 - 1 :=
sorry

end number_of_trivial_proper_subsets_of_A_l244_244807


namespace true_statements_count_l244_244821

variable {α : Type*}
variables (M N : Set α)

theorem true_statements_count :
  let P1 := (M ∩ N ⊆ N)
  let P2 := (M ∩ N ⊆ M ∪ N)
  let P3 := (M ∪ N ⊆ N)
  let P4 := (M ⊆ N → M ∩ N = M)
  P1 ∧ P2 ∧ ¬P3 ∧ P4 :=
by
  intro P1 P2 P3 P4
  split
  -- Proof steps would go here
  repeat { sorry }

end true_statements_count_l244_244821


namespace mixed_concentration_correct_l244_244999

variable (a b : ℝ) -- masses of solution A and B

-- Assume the salinity of solutions
def salinity_A := 0.08
def salinity_B := 0.05
def mixed_salinity := 0.062

-- Given condition for the mixture
axiom mix_condition : salinity_A * a + salinity_B * b = mixed_salinity * (a + b)

-- Define new fractions
def quarter_A := (1/4) * salinity_A * a
def sixth_B := (1/6) * salinity_B * b

-- Compute the concentration when those fractions are mixed
def final_concentration := (quarter_A + sixth_B) / ((1/4) * a + (1/6) * b)

theorem mixed_concentration_correct : 
  final_concentration = 0.045 := sorry

end mixed_concentration_correct_l244_244999


namespace chocolates_difference_l244_244087

def robert_chocolates : Nat := 3 / 7 * 70
def nickel_chocolates : Nat := 1.2 * 40
def penny_chocolates : Nat := 3 / 8 * 80
def dime_chocolates : Nat := 1 / 2 * 90

theorem chocolates_difference :
  (robert_chocolates + nickel_chocolates) - (penny_chocolates + dime_chocolates) = 3 :=
by sorry

end chocolates_difference_l244_244087


namespace rooms_with_two_beds_l244_244411

variable (x y : ℕ)

theorem rooms_with_two_beds:
  x + y = 13 →
  2 * x + 3 * y = 31 →
  x = 8 :=
by
  intros h1 h2
  sorry

end rooms_with_two_beds_l244_244411


namespace diameter_increase_l244_244761

theorem diameter_increase (A A' D D' : ℝ)
  (hA_increase: A' = 4 * A)
  (hA: A = π * (D / 2)^2)
  (hA': A' = π * (D' / 2)^2) :
  D' = 2 * D :=
by 
  sorry

end diameter_increase_l244_244761


namespace parabola_p_and_circle_tangent_directrix_l244_244234

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244234


namespace sunflower_plants_l244_244951

theorem sunflower_plants
  (num_corn_plants : ℕ)
  (num_tomato_plants : ℕ)
  (max_plants_per_row : ℕ)
  (num_corn_rows : ℕ)
  (num_tomato_rows : ℕ)
  (num_sunflower_rows : ℕ) :
  num_corn_plants = 81 →
  num_tomato_plants = 63 →
  max_plants_per_row = 9 →
  num_corn_rows = num_corn_plants / max_plants_per_row →
  num_tomato_rows = num_tomato_plants / max_plants_per_row →
  num_sunflower_rows = num_corn_rows →
  num_sunflower_rows * max_plants_per_row = 81 :=
by
  intros h_corn h_tomato h_max h_corn_rows h_tomato_rows h_sunflower_rows
  rw [h_corn, h_tomato, h_max] at *
  rw [←h_sunflower_rows, ←h_corn_rows, ←h_tomato_rows]
  sorry

end sunflower_plants_l244_244951


namespace cartesian_eq_and_min_PA_PB_l244_244671

-- Definition of curve C1 in polar coordinates
def curve_C1 (ρ θ : ℝ) : Prop := ρ = 2 * cos θ

-- Definition of curve C2 in Cartesian coordinates
def curve_C2 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Minimum value of |PA||PB| for a given line passing through P intersecting C2
noncomputable def min_PA_PB : ℝ := 13 / 4

-- Theorem statement: Cartesian equation of C2 and minimum value of |PA||PB|
theorem cartesian_eq_and_min_PA_PB :
  (∀ x y : ℝ, curve_C1 x y → curve_C2 x y) ∧
  (∀ α : ℝ, ∃ t1 t2 : ℝ, 
    let line_through_P := (1 + t1*cos α, 2 + t2*sin α) in
    let PA_length := sqrt ((line_through_P.1 - P.1)^2 + (line_through_P.2 - P.2)^2) in
    let PB_length := sqrt ((line_through_P.1 - P.1)^2 + (line_through_P.2 - P.2)^2) in
    let prod_PA_PB := PA_length * PB_length in
    prod_PA_PB = 13 / (1 + 3 * (sin α)^2) ∧
    (∃ sin_max : ℝ, sin_max = 1 → prod_PA_PB = 13 / 4)) :=
sorry

end cartesian_eq_and_min_PA_PB_l244_244671


namespace probability_of_5_non_standard_parts_l244_244315

-- Definitions
def n : ℕ := 1000
def p : ℝ := 0.004
def k : ℕ := 5
def a : ℝ := n * p -- Expected number of defects
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Probability calculation using the Poisson approximation
def poisson_probability (k : ℕ) (a : ℝ) : ℝ := (a^k) * Real.exp (-a) / (factorial k)

-- The actual statement to prove
theorem probability_of_5_non_standard_parts : poisson_probability k a ≈ 0.1562 := by
  sorry

end probability_of_5_non_standard_parts_l244_244315


namespace latest_start_time_l244_244597

-- Define the times for each activity
def homework_time : ℕ := 30
def clean_room_time : ℕ := 30
def take_out_trash_time : ℕ := 5
def empty_dishwasher_time : ℕ := 10
def dinner_time : ℕ := 45

-- Define the total time required to finish everything in minutes
def total_time_needed : ℕ := homework_time + clean_room_time + take_out_trash_time + empty_dishwasher_time + dinner_time

-- Define the equivalent time in hours
def total_time_needed_hours : ℕ := total_time_needed / 60

-- Define movie start time and the time Justin gets home
def movie_start_time : ℕ := 20 -- (8 PM in 24-hour format)
def justin_home_time : ℕ := 17 -- (5 PM in 24-hour format)

-- Prove the latest time Justin can start his chores and homework
theorem latest_start_time : movie_start_time - total_time_needed_hours = 18 := by
  sorry

end latest_start_time_l244_244597


namespace median_after_adding_nine_l244_244369

theorem median_after_adding_nine
  (xs : List ℕ)
  (h_len : xs.length = 5)
  (h_sum : xs.sum = 29)
  (h_mode : ∃ k, List.mode xs = some k ∧ k = 4)
  (h_median : xs.sorted.nth 2 = some 5)
  : List.median (List.insert 9 xs) = some 5.5 :=
sorry

end median_after_adding_nine_l244_244369


namespace equation_three_no_real_roots_l244_244906

theorem equation_three_no_real_roots
  (a₁ a₂ a₃ : ℝ)
  (h₁ : a₁^2 - 4 ≥ 0)
  (h₂ : a₂^2 - 8 < 0)
  (h₃ : a₂^2 = a₁ * a₃) :
  a₃^2 - 16 < 0 :=
sorry

end equation_three_no_real_roots_l244_244906


namespace necessary_but_not_sufficient_l244_244363

theorem necessary_but_not_sufficient (x : ℝ) (h : x ≠ 1) : x^2 - 3 * x + 2 ≠ 0 :=
by
  intro h1
  -- Insert the proof here
  sorry

end necessary_but_not_sufficient_l244_244363


namespace max_value_ln_sub_x_on_0_e_l244_244309

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem max_value_ln_sub_x_on_0_e : ∃ (x : ℝ), 0 < x ∧ x ≤ Real.exp 1 ∧ (∀ y, 0 < y ∧ y ≤ Real.exp 1 → f(y) ≤ f(x)) ∧ f(x) = -1 :=
by
  sorry

end max_value_ln_sub_x_on_0_e_l244_244309


namespace projection_of_a_onto_b_l244_244050

-- Define the vectors a and b
variables (a b : ℝ × ℝ) 

-- Given conditions
def a_mag : ℝ := 2
def b_mag : ℝ := 1
def theta : ℝ := 2 * Real.pi / 3

-- Prove that the projection of a onto b equals -1
theorem projection_of_a_onto_b
  (ha : Real.sqrt (a.1 ^ 2 + a.2 ^ 2) = a_mag)
  (hb : Real.sqrt (b.1 ^ 2 + b.2 ^ 2) = b_mag)
  (angle : Real.angle a b = theta) :
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (b.1 ^ 2 + b.2 ^ 2)) = -1 :=
sorry

end projection_of_a_onto_b_l244_244050


namespace standard_equation_of_ellipse_l244_244478

-- Define the conditions
def isEccentricity (e : ℝ) := e = (Real.sqrt 3) / 3
def segmentLength (L : ℝ) := L = (4 * Real.sqrt 3) / 3

-- Define properties
def is_ellipse (a b c : ℝ) := a > b ∧ b > 0 ∧ (a^2 = b^2 + c^2) ∧ (c = (Real.sqrt 3) / 3 * a)

-- The problem statement
theorem standard_equation_of_ellipse
(a b c : ℝ) (E L : ℝ)
(hE : isEccentricity E)
(hL : segmentLength L)
(h : is_ellipse a b c)
: (a = Real.sqrt 3) ∧ (c = 1) ∧ (b = Real.sqrt 2) ∧ (segmentLength L)
  → ( ∀ x y : ℝ, ((x^2 / 3) + (y^2 / 2) = 1) ) := by
  sorry

end standard_equation_of_ellipse_l244_244478


namespace max_ratio_OB_OA_l244_244990

noncomputable def C1_polar_eqn (ρ θ : ℝ) : Prop := ρ * sin(θ + π / 4) = sqrt 2 / 2
noncomputable def C2_polar_eqn (ρ θ : ℝ) : Prop := ρ = 4 * cos θ
noncomputable def O_A_distance (ρ_A θ : ℝ) : ℝ := 1 / (cos θ + sin θ)
noncomputable def O_B_distance (ρ_B θ : ℝ) : ℝ := 4 * cos θ
noncomputable def ratio_OB_OA (α : ℝ) : ℝ := 2 + 2 * sqrt 2

theorem max_ratio_OB_OA {α : ℝ} (hα : 0 ≤ α ∧ α ≤ π / 2) :
    ∃ θ ρ_A ρ_B, C1_polar_eqn ρ_A θ ∧ C2_polar_eqn ρ_B θ ∧ 
    α = θ ∧ (O_B_distance ρ_B θ) / (O_A_distance ρ_A θ) = ratio_OB_OA α := 
sorry

end max_ratio_OB_OA_l244_244990


namespace length_of_lunch_break_is_48_minutes_l244_244650

noncomputable def paula_and_assistants_lunch_break : ℝ := sorry

theorem length_of_lunch_break_is_48_minutes
  (p h L : ℝ)
  (h_monday : (9 - L) * (p + h) = 0.6)
  (h_tuesday : (7 - L) * h = 0.3)
  (h_wednesday : (10 - L) * p = 0.1) :
  L = 0.8 :=
sorry

end length_of_lunch_break_is_48_minutes_l244_244650


namespace prime_count_l244_244523

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def from_digits (tens units : ℕ) : ℕ :=
  10 * tens + units

def is_valid_prime (tens units : ℕ) : Prop :=
  {3, 5, 7, 9}.contains tens ∧ 
  {3, 5, 7, 9}.contains units ∧ 
  tens ≠ units ∧ 
  is_prime (from_digits tens units)

theorem prime_count : 
  (finset.univ.filter (λ p, ∃ tens ∈ {3, 5, 7, 9}, ∃ units ∈ {3, 5, 7, 9}, tens ≠ units ∧ is_prime (from_digits tens units))).card = 6 :=
by
  sorry

end prime_count_l244_244523


namespace probability_same_group_l244_244719

theorem probability_same_group :
  let total_outcomes := 4 * 4 in
  let favorable_outcomes := 4 in
  let probability := favorable_outcomes / total_outcomes in
  probability = 1 / 4 :=
by
  sorry

end probability_same_group_l244_244719


namespace ratio_of_socks_l244_244113

theorem ratio_of_socks (y p : ℝ) (h1 : 5 * p + y * 2 * p = 5 * p + 4 * y * p / 3) :
  (5 : ℝ) / y = 11 / 2 :=
by
  sorry

end ratio_of_socks_l244_244113


namespace value_of_p_circle_tangent_to_directrix_l244_244138

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244138


namespace parallel_lines_perpendicular_to_same_plane_l244_244410

/-- Define Lines and Plane -/
variable {Point : Type*} [affine_space ℝ Point]
variable (l m : affine_subspace ℝ Point) [is_line l] [is_line m] -- Lines
variable (α : affine_subspace ℝ Point) [is_plane α] -- Plane

/-- Predicate for perpendicularity between lines and planes --/
def is_perpendicular (l : affine_subspace ℝ Point) (α : affine_subspace ℝ Point) : Prop :=
  ∀ (p ∈ l) (q ∈ α), ∀ (v : Point), v ∈ direction α → inner (v -ᵥ p) (v -ᵥ q) = 0
  
def are_parallel (l m : affine_subspace ℝ Point) : Prop :=
  ∀ (p ∈ l) (q ∈ m), inner (p -ᵥ q) = 0

theorem parallel_lines_perpendicular_to_same_plane (hl : is_perpendicular l α) (hm : is_perpendicular m α) : are_parallel l m :=
by sorry

end parallel_lines_perpendicular_to_same_plane_l244_244410


namespace cos_30_eq_sqrt3_div_2_l244_244338

theorem cos_30_eq_sqrt3_div_2 : Real.cos (Real.pi / 6) = (Real.sqrt 3) / 2 :=
by
  sorry

end cos_30_eq_sqrt3_div_2_l244_244338


namespace find_real_a_l244_244032

theorem find_real_a 
  (a : ℝ) 
  (z : ℂ := (1 - a*complex.I) / complex.I) 
  (hx : (-a, -1).fst + 2*(-a, -1).snd + 5 = 0) :
  a = 3 := 
by
  sorry

end find_real_a_l244_244032


namespace multiples_of_30_not_45_l244_244515

theorem multiples_of_30_not_45 : 
  let multiples_of_30 := {n | 100 ≤ n ∧ n < 1000 ∧ n % 30 = 0},
      multiples_of_45 := {n | 100 ≤ n ∧ n < 1000 ∧ n % 45 = 0} in
  (multiples_of_30.card - multiples_of_45.card = 20) :=
sorry

end multiples_of_30_not_45_l244_244515


namespace line_circle_disjoint_l244_244491

theorem line_circle_disjoint (a b : ℝ) (h : a^2 + b^2 < 1) :
  ∀ x y : ℝ, ¬ (x^2 + y^2 = 1 ∧ ax + by = 1) :=
by
  assume x y
  assume ⟨hx, hy⟩
  let d := 1 / real.sqrt (a^2 + b^2)
  have d_gt_one : d > 1, from sorry
  have d_eq : d = abs (a * 0 + b * 0 - 1) / real.sqrt (a^2 + b^2), from sorry
  have distance_to_origin : abs (a * 0 + b * 0 - 1) / real.sqrt (a^2 + b^2) > 1, from sorry
  exact absurd (le_of_eq hy) distance_to_origin
  sorry

end line_circle_disjoint_l244_244491


namespace problem_l244_244128

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244128


namespace always_positive_iff_k_gt_half_l244_244313

theorem always_positive_iff_k_gt_half (k : ℝ) :
  (∀ x : ℝ, k * x^2 + x + k > 0) ↔ k > 0.5 :=
sorry

end always_positive_iff_k_gt_half_l244_244313


namespace not_factorable_l244_244659

-- Define the quartic polynomial P(x)
def P (x : ℤ) : ℤ := x^4 + 2 * x^2 + 2 * x + 2

-- Define the quadratic polynomials with integer coefficients
def Q₁ (a b x : ℤ) : ℤ := x^2 + a * x + b
def Q₂ (c d x : ℤ) : ℤ := x^2 + c * x + d

-- Define the condition for factorization, and the theorem to be proven
theorem not_factorable :
  ¬ ∃ (a b c d : ℤ), ∀ x : ℤ, P x = (Q₁ a b x) * (Q₂ c d x) := by
  sorry

end not_factorable_l244_244659


namespace no_return_to_original_set_l244_244910

structure Point :=
(x : ℝ)
(y : ℝ)

def distance_sq (p1 p2 : Point) : ℝ := (p1.x - p2.x)^2 + (p1.y - p2.y)^2

def perpendicular (p1 p2 p3 : Point) : Prop :=
(p3.y - p1.y) * (p2.y - p1.y) = -(p3.x - p1.x) * (p2.x - p1.x)

def opposite_sides (p1 p2 p3 p4 : Point) : Prop :=
(p3.y - p1.y) * (p3.x - p2.x) < 0 ∧ (p4.y - p1.y) * (p4.x - p2.x) < 0

theorem no_return_to_original_set 
(points : list Point)
(orig_points : list Point)
(A B C D : Point)
(h1 : ∀ (A B : Point), ∃ (C D : Point),
  distance_sq A C = distance_sq B D ∧ perpendicular A B C ∧ perpendicular A B D ∧ opposite_sides A B C D)
: (∀ (op : list Point → list Point),
   let new_points := op points in magnitude_S new_points > magnitude_S points) →
  ¬(∃ (op_seq : list (list Point → list Point)),
     foldl (λ p op, op p) points op_seq = orig_points) := 
sorry

end no_return_to_original_set_l244_244910


namespace p_eq_two_circle_tangent_proof_l244_244223

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244223


namespace sin_alpha_eq_sin_beta_iff_l244_244460

theorem sin_alpha_eq_sin_beta_iff (α β : ℝ) : 
  (∃ k : ℤ, α = k * real.pi + (-1)^k * β) ↔ real.sin α = real.sin β :=
by
  intro alpha beta k
  sorry

end sin_alpha_eq_sin_beta_iff_l244_244460


namespace check_incorrect_statements_l244_244403

-- Definitions related to vectors
structure Vector (ℝ : Type) :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def are_parallel (v1 v2 : Vector ℝ) : Prop :=
  ∃ λ : ℝ, (λ ≠ 0 ∨ v1 = ⟨0, 0, 0⟩) ∧ v1 = λ • v2 

def are_equal (v1 v2 : Vector ℝ) : Prop :=
  v1 = v2

def are_collinear (v1 v2 : Vector ℝ) : Prop :=
  v1 = 0 ∨ v2 = 0 ∨ are_parallel v1 v2

-- Conditions based on vector properties
axiom parallel_vectors_def : ∀ v1 v2, are_parallel v1 v2 → (v1 = v2 ∨ v1 = -v2)
axiom equal_vectors_def : ∀ v1 v2, are_equal v1 v2 ↔ v1 = v2
axiom zero_vector_condition : ∀ v, are_parallel v ⟨0, 0, 0⟩ → ¬ are_collinear v ⟨0, 0, 0⟩

-- Statements to evaluate
def statement1 : Prop := ∀ v1 v2, are_parallel v1 v2 → are_equal v1 v2
def statement2 : Prop := ∀ v1 v2, ¬ are_equal v1 v2 → ¬ are_parallel v1 v2
def statement3 : Prop := ∀ v1 v2, are_collinear v1 v2 → are_equal v1 v2
def statement4 : Prop := ∀ v1 v2, are_equal v1 v2 → are_collinear v1 v2
def statement5 : Prop := ∀ v1 v2, (v1.x^2 + v1.y^2 + v1.z^2) = (v2.x^2 + v2.y^2 + v2.z^2) → are_equal v1 v2
def statement6 : Prop := ∀ v1 v2 v3, are_parallel v1 v3 → are_parallel v2 v3 → are_collinear v1 v2

-- The proof goal is to show which statements are incorrect
theorem check_incorrect_statements : ¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement5 ∧ ¬statement6 := 
sorry

end check_incorrect_statements_l244_244403


namespace length_of_goods_train_l244_244375

theorem length_of_goods_train (speed_kmh : ℕ) (platform_length : ℕ) (crossing_time : ℕ) (h_speed : speed_kmh = 72) (h_platform : platform_length = 250) (h_time : crossing_time = 22) :
  let speed_mps := (speed_kmh * 1000) / 3600 in
  let distance_covered := speed_mps * crossing_time in
  let length_of_train := distance_covered - platform_length in
  length_of_train = 190 :=
by
  sorry

end length_of_goods_train_l244_244375


namespace divide_square_into_trapezoids_l244_244847

theorem divide_square_into_trapezoids (A : ℝ) (hA : A = 4) :
  ∃ (trap1 trap2 trap3 trap4 : ℝ), trap1 = 1 ∧ trap2 = 2 ∧ trap3 = 3 ∧ trap4 = 4 ∧ 
    (∀ h : ℝ, h ∈ {trap1, trap2, trap3, trap4} → h ≤ A) :=
by
  sorry

end divide_square_into_trapezoids_l244_244847


namespace value_of_phi_l244_244868

noncomputable def f (x φ : ℝ) : ℝ := sin (2 * x + φ) + sqrt 3 * cos (2 * x + φ)

theorem value_of_phi (φ : ℝ) : 
  (∀ x : ℝ, f x φ = -f (-x) φ) ∧ 
  (∀ x ∈ Icc (0 : ℝ) (π / 4), deriv (f x φ) < 0) → 
  φ = 2 * π / 3 :=
  sorry

end value_of_phi_l244_244868


namespace trace_ellipse_or_line_segment_l244_244253

theorem trace_ellipse_or_line_segment 
  (a b : ℂ) :
  (∀ φ : ℝ, 0 ≤ φ ∧ φ ≤ 2 * Real.pi → ∃ ξ η : ℝ, (a * Complex.exp (Complex.I * φ) + b * Complex.exp (- Complex.I * φ)) = (ξ + Complex.I * η)) ∧
  ((a.norm_sq ≠ b.norm_sq → ∃ α β γ δ : ℝ, ∀ z : ℂ, (α * ((z.re : ℂ) + Complex.I * (z.im : ℂ)) + β = (z.re + Complex.I * z.im)) ∧ (γ * ((z.re : ℂ) + Complex.I * (z.im : ℂ)) + δ = (z.re + Complex.I * z.im))) ∨
  (a.norm_sq = b.norm_sq → ∃ m : ℝ, ∀ z : ℂ, ∃ ξ η : ℝ, (z = m * (ξ + Complex.I * η)))) :=
sorry

end trace_ellipse_or_line_segment_l244_244253


namespace polynomial_expansion_sum_l244_244547

theorem polynomial_expansion_sum :
  ∀ (a : ℕ → ℤ), (∀ x : ℝ, (1 - 5 * x)^2023 = ∑ i in Finset.range 2024, a i * x^i) →
    (∑ i in Finset.range 2024, a i / 5^i) = -1 :=
by
  intros a ha
  sorry

end polynomial_expansion_sum_l244_244547


namespace max_value_of_A_l244_244390

-- Define the telephone number problem
def valid_telephone_number (A B C D E F G H I J : ℕ) : Prop :=
  A > B ∧ B > C ∧
  D > E ∧ E > F ∧
  (D = E + 2) ∧ (E = F + 2) ∧
  G > H ∧ H > I ∧ I > J ∧
  (G = H + 1) ∧ (H = I + 1) ∧ (I = J + 1) ∧
  A + B + C = 10

-- The main theorem to prove
theorem max_value_of_A : ∃ (A B C D E F G H I J : ℕ), valid_telephone_number A B C D E F G H I J ∧ A = 7 :=
begin
  sorry
end

end max_value_of_A_l244_244390


namespace poisson_generating_function_geometric_generating_function_bernoulli_generating_function_l244_244361

/-- The generating function of a Poisson random variable -/
theorem poisson_generating_function (λ : ℝ) (s : ℝ) (h : 0 ≤ s ∧ s ≤ 1) :
  (∀ n : ℕ, p_n = (Real.exp (-λ) * λ^n / Nat.factorial n)) →
  ∑' n, p_n * s^n = Real.exp (-λ * (1 - s)) := sorry

/-- The generating function of a Geometric random variable -/
theorem geometric_generating_function (p : ℝ) (h : 0 < p ∧ p < 1) (s : ℝ) (hs : 0 ≤ s ∧ s ≤ 1) :
  (∀ n : ℕ, p_n = p * ((1 - p)^n)) →
  ∑' n, p_n * s^n = p / (1 - s * (1 - p)) := sorry

/-- The generating function and distribution of sum of i.i.d. Bernoulli random variables -/
theorem bernoulli_generating_function (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1) (n : ℕ) (s : ℝ) (hs : 0 ≤ s ∧ s ≤ 1) :
  (∀ k : ℕ, P (X = k) = Nat.choose n k * (p^k) * ((1 - p)^(n - k))) →
  ∑' k, P (X = k) * s^k = (p * s + (1 - p))^n ∧ 
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n → P (X = k) = Nat.choose n k * (p^k) * ((1 - p)^(n - k)) := sorry

end poisson_generating_function_geometric_generating_function_bernoulli_generating_function_l244_244361


namespace workers_together_time_l244_244360

theorem workers_together_time (A_time B_time : ℝ) (hA : A_time = 8) (hB : B_time = 12) : 
  1 / ((1 / A_time) + (1 / B_time)) = 4.8 :=
by
  rw [hA, hB]
  calc
    1 / ((1 / 8) + (1 / 12)) = 1 / ((3 / 24) + (2 / 24)) : by sorry
    ...                    = 1 / (5 / 24)             : by sorry
    ...                    = 24 / 5                   : by sorry
    ...                    = 4.8                      : by sorry

end workers_together_time_l244_244360


namespace circle_symmetry_line_l244_244290

theorem circle_symmetry_line :
  ∃ l: ℝ → ℝ → Prop, 
    (∀ x y, l x y → x - y + 2 = 0) ∧ 
    (∀ x y, l x y ↔ (x + 2)^2 + (y - 2)^2 = 4) :=
sorry

end circle_symmetry_line_l244_244290


namespace coefficient_x3_expansion_l244_244604

theorem coefficient_x3_expansion :
  let f := ∑ n in finset.range 48, (1 + (x : ℝ)) ^ (n + 3)
  polynomial.nat_degree (polynomial.coeff f 3) = nat.choose 51 4 := 
begin
  sorry
end

end coefficient_x3_expansion_l244_244604


namespace division_and_multiplication_l244_244344

theorem division_and_multiplication (x : ℝ) (h : x = 9) : (x / 6 * 12) = 18 := by
  sorry

end division_and_multiplication_l244_244344


namespace eggs_per_meal_l244_244263

noncomputable def initial_eggs_from_store : ℕ := 12
noncomputable def additional_eggs_from_neighbor : ℕ := 12
noncomputable def eggs_used_for_cooking : ℕ := 2 + 4
noncomputable def remaining_eggs_after_cooking : ℕ := initial_eggs_from_store + additional_eggs_from_neighbor - eggs_used_for_cooking
noncomputable def eggs_given_to_aunt : ℕ := remaining_eggs_after_cooking / 2
noncomputable def remaining_eggs_after_giving_to_aunt : ℕ := remaining_eggs_after_cooking - eggs_given_to_aunt
noncomputable def planned_meals : ℕ := 3

theorem eggs_per_meal : remaining_eggs_after_giving_to_aunt / planned_meals = 3 := 
by 
  sorry

end eggs_per_meal_l244_244263


namespace find_uv_l244_244855

theorem find_uv :
  ∃ (u v : ℝ), 
    (3 + 4 * u = -5 * v) ∧ (-2 - 7 * u = 1 + 4 * v) ∧ 
    (u = -93 / 14) ∧ (v = 33 / 7) :=
by {
  use [-93 / 14, 33 / 7],
  split,
  { -- Show 3 + 4 * u = -5 * v
    calc 3 + 4 * (-93 / 14) = 3 - 372 / 14 : by ring 
                        ... = (42 / 14) - (372 / 14) : by norm_num
                        ... = -330 / 14 : by norm_num
                        ... = -5 * (33 / 7) : by norm_num },
  split,
  { -- Show -2 - 7 * u = 1 + 4 * v
    calc -2 - 7 * (-93 / 14) = -2 + 651 / 14 : by ring 
                           ... = (-28 / 14) + 651 / 14 : by norm_num
                           ... = 623 / 14 : by norm_num
                           ... = 1 + 4 * (33 / 7) : by norm_num },
  split;
  { refl }
}

end find_uv_l244_244855


namespace find_n_l244_244619

theorem find_n (n : ℕ) (x : ℕ) (h₁ : x = (1 + 2) * (1 + 2^2) * (1 + 2^4) * (1 + 2^8) * ... * (1 + 2^n))
                (h₂ : x + 1 = 2^128) : n = 64 :=
sorry

end find_n_l244_244619


namespace F_value_l244_244959
-- Import necessary Lean libraries

-- Definition of the function f(n)
def f (n : ℕ) (r : ℕ) : ℤ := (-1) ^ r

-- Definition of the function F(n)
def F (n : ℕ) : ℤ :=
  ∑ d in (divisors n), f d (∑ r in (divisors n), r) -- Here r would need exact mapping from prime factorization; theoretical set up Dummy

-- The statement of the theorem to be proven
theorem F_value (n : ℕ) (is_perfect_square : ∀ p k, n = p^(2*k)) : F(n) = 1 ∨ F(n) = 0 :=
  sorry

-- Additional auxiliary statements and definitions might be required depending on the context of prime decomposition and sum calculations

end F_value_l244_244959


namespace cyclic_quad_diagonals_circle_l244_244860

variables {A B C D K L M N : Type} [has_circumscribed_circle A B C D] 
          {m_ac : midpoint A C M} {m_bd : midpoint B D N}
          (circle_adm : circumscribed_circle A D M) (circle_bcm : circumscribed_circle B C M)
          (h : points_intersecting_circle A D M at M L) (h2 : points_intersecting_circle B C M at M L)

theorem cyclic_quad_diagonals_circle :
  ∃ (circ : circumscribed_circle K L M N), true :=
sorry -- Proof goes here

end cyclic_quad_diagonals_circle_l244_244860


namespace rook_average_score_l244_244412

theorem rook_average_score (p q : ℕ) (hℚ : p.gcd q = 1 ∧ p = 291 ∧ q = 80) : 
  (let score := (2 * 120 + 3 * 216 + 4 * 222 + 5 * 130 + 6 * 31 + 7 * 1) / 720 in
   score = (p : ℚ) / (q : ℚ)) ∧ (p + q = 371) := sorry

end rook_average_score_l244_244412


namespace fourth_figure_dots_l244_244844

theorem fourth_figure_dots :
  let dots (n : Nat) := match n with
    | 0 => 1
    | k + 1 => dots k + 5 * k
  in dots 3 = 31 :=
by
  sorry

end fourth_figure_dots_l244_244844


namespace magic_king_total_episodes_l244_244706

theorem magic_king_total_episodes :
  (∑ i in finset.range 5, 20) + (∑ j in finset.range 5, 25) = 225 :=
by sorry

end magic_king_total_episodes_l244_244706


namespace tangent_iff_parallelogram_l244_244774

-- Defining the problem conditions
variables (O A B C D M N P : Type)
variables [circumcircle : ∀ O (A B C : Type), Prop]
variables (diameter : ∃ (D : Type), ∀ (A B C : Type), is_diameter D)
variables (intersect : ∀ (P : Type), (PO : Type) → (AB AC BC : Type), ∀ (M N P : Type), intersects_line P M N)

-- Defining the problem statement
theorem tangent_iff_parallelogram :
  (PD_is_tangent : ∀ (P D : Type), is_tangent P D) ↔ 
  (AMDN_parallelogram : ∀ (A M D N : Type), is_parallelogram A M D N) :=
sorry

end tangent_iff_parallelogram_l244_244774


namespace conjugate_point_l244_244551

variables (z : ℂ) (conjugate_z : ℂ) (point : ℝ × ℝ)

def z := (1/2 : ℝ) + (1 : ℝ)*complex.I

def conjugate_z := complex.conj z

def point := (conjugate_z.re, conjugate_z.im)

theorem conjugate_point :
  point = (1/2 : ℝ, -1) :=
sorry

end conjugate_point_l244_244551


namespace reciprocal_of_sum_eq_l244_244450

theorem reciprocal_of_sum_eq (a b c d : ℚ) (ha : a = 1/3) (hb : b = 1/4) (hcd : c = 12) : 
  let sum_fractions := a + b in
  let common_denominator := c in
  let simplified_sum := (4 * ha + 3 * hb) / common_denominator in
  let reciprocal := simplified_sum⁻¹ in
  reciprocal = 12 / 7 :=
by
  sorry

end reciprocal_of_sum_eq_l244_244450


namespace smallest_x_for_g_g_l244_244073

def g (x : ℝ) := real.sqrt (x - 5)

theorem smallest_x_for_g_g (x : ℝ) : (∀ x, (g (g x)).is_defined → x ≥ 30) :=
by
  intros x hx
  sorry

end smallest_x_for_g_g_l244_244073


namespace parabola_p_and_circle_tangent_directrix_l244_244226

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244226


namespace least_value_QGK_l244_244345

theorem least_value_QGK :
  ∃ (G K Q : ℕ), (10 * G + G) * G = 100 * Q + 10 * G + K ∧ G ≠ K ∧ (10 * G + G) ≥ 10 ∧ (10 * G + G) < 100 ∧  ∃ x, x = 44 ∧ 100 * G + 10 * 4 + 4 = (100 * Q + 10 * G + K) ∧ 100 * 0 + 10 * 4 + 4 = 044  :=
by
  sorry

end least_value_QGK_l244_244345


namespace factorial_expression_simplifies_l244_244415

theorem factorial_expression_simplifies :
    8! - 7 * 7! - 2 * 7! = -5040 := 
by
  sorry

end factorial_expression_simplifies_l244_244415


namespace minimum_value_problem_l244_244623

theorem minimum_value_problem (a b c : ℝ) (hb : a > 0 ∧ b > 0 ∧ c > 0)
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) : 
  ∃ x, (x = 47) ∧ (a / b + b / c + c / a) * (b / a + c / b + a / c) ≥ x :=
by
  sorry

end minimum_value_problem_l244_244623


namespace arithmetic_sequence_general_formula_sum_of_first_n_terms_l244_244574

-- Lean statement for part (1)
theorem arithmetic_sequence_general_formula (d : ℕ) (a : ℕ → ℕ)
  (h_seq : ∀ n m, a (n + m) = a n + m * d)
  (h_a1 : a 1 = 2)
  (h_a3_geometric_mean : a 3 * a 3 = a 1 * a 9) : 
  d = 2 ∧ ∀ n, a n = 2 * n :=
by sorry

-- Lean statement for part (2)
theorem sum_of_first_n_terms (a : ℕ → ℕ)
  (h_general_formula : ∀ n, a n = 2 * n) :
  ∀ n, (∑ k in finset.range n, 1 / (a k * a (k+1))) = n / (4 * (n + 1)) :=
by sorry

end arithmetic_sequence_general_formula_sum_of_first_n_terms_l244_244574


namespace value_of_p_circle_tangent_to_directrix_l244_244143

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244143


namespace simplest_quadratic_radical_is_B_l244_244400

-- Define the given options as Lean expressions
def option_A := Real.sqrt (2 / 3)
def option_B := 2 * Real.sqrt 2
def option_C := Real.sqrt 24
def option_D := Real.sqrt 81

-- Define the problem statement in Lean: prove that option B is the simplest quadratic radical
theorem simplest_quadratic_radical_is_B :
  (option_B = 2 * Real.sqrt 2) ∧
  (∀ x, x = option_A → (∃ z, option_B = z) ∧ (∃ k, option_B = k)) ∧
  (∀ x, x = option_C → (∃ z, option_B = z) ∧ (∃ k, option_B = k)) ∧
  (∀ x, x = option_D → (∃ z, option_B = z) ∧ (∃ k, option_B = k)) :=
  by sorry

end simplest_quadratic_radical_is_B_l244_244400


namespace diving_classes_on_weekdays_l244_244672

theorem diving_classes_on_weekdays 
  (x : ℕ) 
  (weekend_classes_per_day : ℕ := 4)
  (people_per_class : ℕ := 5)
  (total_people_3_weeks : ℕ := 270)
  (weekend_days : ℕ := 2)
  (total_weeks : ℕ := 3)
  (weekend_total_classes : ℕ := weekend_classes_per_day * weekend_days * total_weeks) 
  (total_people_weekends : ℕ := weekend_total_classes * people_per_class) 
  (total_people_weekdays : ℕ := total_people_3_weeks - total_people_weekends)
  (weekday_classes_needed : ℕ := total_people_weekdays / people_per_class)
  (weekly_weekday_classes : ℕ := weekday_classes_needed / total_weeks)
  (h : weekly_weekday_classes = x)
  : x = 10 := sorry

end diving_classes_on_weekdays_l244_244672


namespace parabola_condition_l244_244179

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244179


namespace n_prime_or_power_of_2_l244_244626

theorem n_prime_or_power_of_2 
  (n : ℕ) (hgt : n > 6)
  (a : ℕ → ℕ) (rel_prime : ∀ i, i > 0 → i < n → Nat.gcd (a i) n = 1) 
  (diff_eq : ∀ i j, i < j → j < n → a (i+1) - a i = a (i + 2) - a (i + 1) ∧ a (i + 1) - a i > 0) :
  (Nat.prime n ∨ ∃ k : ℕ, k > 0 ∧ n = 2^k) := 
sorry

end n_prime_or_power_of_2_l244_244626


namespace inequalities_quadrants_l244_244703

theorem inequalities_quadrants :
  (∀ x y : ℝ, y > 2 * x → y > 4 - x → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) := sorry

end inequalities_quadrants_l244_244703


namespace even_function_increasing_on_neg_l244_244493

noncomputable def is_even_function {α : Type*} [linear_ordered_field α] (f : α → α) :=
  ∀ x : α, f x = f (-x)

noncomputable def is_decreasing_on_pos {α : Type*} [linear_ordered_field α] (f : α → α) :=
  ∀ a b : α, 0 < a → a < b → f a > f b

theorem even_function_increasing_on_neg {α : Type*} [linear_ordered_field α] (f : α → α)
  (h_even : is_even_function f) (h_decreasing : is_decreasing_on_pos f) :
  ∀ x1 x2 : α, x1 < x2 → x2 < 0 → f x1 < f x2 :=
by
  intros x1 x2 h1 h2
  have h_pos1 : 0 < -x2 := by linarith
  have h_pos2 : -x2 < -x1 := by linarith
  have h_dec : f (-x2) > f (-x1) := h_decreasing (-x2) (-x1) h_pos1 h_pos2
  have h_ev1 : f (-x1) = f x1 := h_even x1
  have h_ev2 : f (-x2) = f x2 := h_even x2
  rw [h_ev1, h_ev2] at h_dec
  exact h_dec

'sorry

end even_function_increasing_on_neg_l244_244493


namespace inequality_condition_l244_244846

theorem inequality_condition (x y : ℝ) : 
  (2 * y - 3 * x > real.sqrt (9 * x^2)) ↔ 
  (x ≥ 0 ∧ y > 3 * x) ∨ (x < 0 ∧ y > 0) :=
by
  sorry

end inequality_condition_l244_244846


namespace parabola_condition_l244_244175

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244175


namespace prove_p_equals_2_l244_244198

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244198


namespace leo_needs_change_probability_l244_244318

theorem leo_needs_change_probability :
  let toys := 9
  let favorite_toy_cost := 4.00
  let leo_coins := 10
  let fifty_cent_value := 0.50
  let total_amount := (leo_coins * fifty_cent_value)
  let possible_toy_prices := List.range' 1 (toys + 1) |>.map (λ x => x * fifty_cent_value)
  let factorial (n : Nat) : Nat := if n = 0 then 1 else n * factorial (n - 1)
  let total_orders := factorial toys
  let favorable_orders := factorial (toys - 1) + 2 * factorial (toys - 2)
  let no_change_probability := (favorable_orders : ℚ) / (total_orders : ℚ)
  let needs_change_probability := 1 - no_change_probability
  needs_change_probability = 31 / 36 :=
by
  sorry

end leo_needs_change_probability_l244_244318


namespace homothety_incircle_tangent_circumcircle_l244_244998

-- Define the triangle vertices' coordinates and properties
variables {a b R r : Real}

def C : (Real × Real) := (0, 0)
def A : (Real × Real) := (2*a, 0)
def B : (Real × Real) := (0, 2*b)
def O : (Real × Real) := (a, b)
def I : (Real × Real) := (r, r)
def r_def (a b R : Real) : Real := a + b - R

-- Homothety transformation of I
def homothety_I (r : Real) : (Real × Real) := (2*r, 2*r)

def distance (p1 p2 : (Real × Real)) : Real := 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Main theorem statement
theorem homothety_incircle_tangent_circumcircle
  (h_right : ∃ a b : Real, a^2 + b^2 = R^2 ∧ r = a + b - R) :
  distance O (homothety_I r) + 2 * r = R :=
  sorry

end homothety_incircle_tangent_circumcircle_l244_244998


namespace paint_cost_is_200_l244_244298

-- Define the basic conditions and parameters
def side_length : ℕ := 5
def faces_of_cube : ℕ := 6
def area_per_face (side : ℕ) : ℕ := side * side
def total_surface_area (side : ℕ) (faces : ℕ) : ℕ := faces * area_per_face side
def coverage_per_kg : ℕ := 15
def cost_per_kg : ℕ := 20

-- Calculate total cost
def total_cost (side : ℕ) (faces : ℕ) (coverage : ℕ) (cost : ℕ) : ℕ :=
  let total_area := total_surface_area side faces
  let kgs_required := total_area / coverage
  kgs_required * cost

theorem paint_cost_is_200 :
  total_cost side_length faces_of_cube coverage_per_kg cost_per_kg = 200 :=
by
  sorry

end paint_cost_is_200_l244_244298


namespace triangle_angles_cos_sin_identity_l244_244605

theorem triangle_angles_cos_sin_identity :
  ∃ (p q r s : ℕ), 
  (p + q).gcd s = 1 ∧ 
  ¬ r.divisible_by_any_square_prime ∧ 
  p + q + r + s = 1239 ∧
  ∀ (A B C : ℝ), 
    (angles_of_triangle A B C) ∧ 
    (obtuse_angle C) →
      (cos A)^2 + (cos B)^2 + 2 * (sin A) * (sin B) * (cos C) = 11 / 7 →
      (cos B)^2 + (cos C)^2 + 2 * (sin B) * (sin C) * (cos A) = 19 / 12 →
      (cos C)^2 + (cos A)^2 + 2 * (sin C) * (sin A) * (cos B) = p - q * sqrt r / s := 
by 
  sorry

-- Definitions for certain properties
def angles_of_triangle (A B C : ℝ) : Prop :=
  sorry -- A condition that ensures A, B, C are angles of a triangle 

def obtuse_angle (C : ℝ) : Prop :=
  sorry -- A condition that ensures C is an obtuse angle 

def divisible_by_any_square_prime (r : ℕ) : Prop :=
  sorry -- A condition that checks if r is not divisible by any square of a prime 

end triangle_angles_cos_sin_identity_l244_244605


namespace cards_selection_count_l244_244060

noncomputable def numberOfWaysToChooseCards : Nat :=
  (Nat.choose 4 3) * 3 * (Nat.choose 13 2) * (13 ^ 2)

theorem cards_selection_count :
  numberOfWaysToChooseCards = 158184 := by
  sorry

end cards_selection_count_l244_244060


namespace sufficient_condition_for_q_implies_a_leq_zero_l244_244016

theorem sufficient_condition_for_q_implies_a_leq_zero (a x : ℝ) 
  (hp : log 2 (1 - x) < 0) 
  (hq : x > a)
  (suff : ∀ x, log 2 (1 - x) < 0 → x > a) 
  (not_necess : ¬ ∀ x, x > a → log 2 (1 - x) < 0) : a ≤ 0 := 
sorry

end sufficient_condition_for_q_implies_a_leq_zero_l244_244016


namespace prime_count_l244_244526

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def from_digits (tens units : ℕ) : ℕ :=
  10 * tens + units

def is_valid_prime (tens units : ℕ) : Prop :=
  {3, 5, 7, 9}.contains tens ∧ 
  {3, 5, 7, 9}.contains units ∧ 
  tens ≠ units ∧ 
  is_prime (from_digits tens units)

theorem prime_count : 
  (finset.univ.filter (λ p, ∃ tens ∈ {3, 5, 7, 9}, ∃ units ∈ {3, 5, 7, 9}, tens ≠ units ∧ is_prime (from_digits tens units))).card = 6 :=
by
  sorry

end prime_count_l244_244526


namespace watermelons_remaining_l244_244397

theorem watermelons_remaining :
  let initial_watermelons := 10 * 12
  let yesterdays_sale := 0.40 * initial_watermelons
  let remaining_after_yesterday := initial_watermelons - yesterdays_sale
  let todays_sale := (1 / 4) * remaining_after_yesterday
  let remaining_after_today := remaining_after_yesterday - todays_sale
  let tomorrows_sales := 1.5 * todays_sale
  let remaining_after_tomorrow := remaining_after_today - tomorrows_sales
  remaining_after_tomorrow = 27 :=
by
  sorry

end watermelons_remaining_l244_244397


namespace max_travel_distance_l244_244000

theorem max_travel_distance (front_tire_lifespan : ℕ) (rear_tire_lifespan : ℕ) 
  (h₁ : front_tire_lifespan = 24000) (h₂ : rear_tire_lifespan = 36000) : 
  ∃ (D : ℕ), D = 28800 :=
begin
  sorry
end

end max_travel_distance_l244_244000


namespace cost_of_building_fence_eq_3944_l244_244342

def area_square : ℕ := 289
def price_per_foot : ℕ := 58

theorem cost_of_building_fence_eq_3944 : 
  let side_length := (area_square : ℝ) ^ (1/2)
  let perimeter := 4 * side_length
  let cost := perimeter * (price_per_foot : ℝ)
  cost = 3944 :=
by
  sorry

end cost_of_building_fence_eq_3944_l244_244342


namespace ellipse_tangent_sum_l244_244425

theorem ellipse_tangent_sum :
  let ellipse := ∀ (x y : ℝ), 2 * x^2 + 3 * x * y + 5 * y^2 - 15 * x - 24 * y + 56 = 0
  ∃ c d : ℝ, (∀ (x y : ℝ), ellipse x y → (y / x = c ∨ y / x = d)) ∧ c + d = 3 / 34 := 
sorry

end ellipse_tangent_sum_l244_244425


namespace div_expression_l244_244420

variable {α : Type*} [Field α]

theorem div_expression (a b c : α) : 4 * a^2 * b^2 * c / (-2 * a * b^2) = -2 * a * c := by
  sorry

end div_expression_l244_244420


namespace remainder_correct_l244_244271

def dividend : ℝ := 13787
def divisor : ℝ := 154.75280898876406
def quotient : ℝ := 89
def remainder : ℝ := dividend - (divisor * quotient)

theorem remainder_correct: remainder = 14 := by
  -- Proof goes here
  sorry

end remainder_correct_l244_244271


namespace sin_double_angle_l244_244891

theorem sin_double_angle (α : ℝ) (h : cos (α - π / 4) = sqrt 2 / 4) : sin (2 * α) = -3 / 4 :=
sorry

end sin_double_angle_l244_244891


namespace radius_of_circumscribed_circle_of_right_triangle_l244_244678

theorem radius_of_circumscribed_circle_of_right_triangle 
  (a b c : ℝ)
  (h_area : (1 / 2) * a * b = 10)
  (h_inradius : (a + b - c) / 2 = 1)
  (h_hypotenuse : c = Real.sqrt (a^2 + b^2)) :
  c / 2 = 4.5 := 
sorry

end radius_of_circumscribed_circle_of_right_triangle_l244_244678


namespace knight_reach_black_squares_l244_244270

def knight_moves (n : ℕ) (start : ℤ × ℤ) : set (ℤ × ℤ) :=
  sorry -- Define the set of all reachable squares after 2n moves

def is_black_square (pos : ℤ × ℤ) : Prop :=
  (pos.1 + pos.2) % 2 = 0

theorem knight_reach_black_squares (n : ℕ) (start : ℤ × ℤ) (h_start : is_black_square start) :
  ∀ pos ∈ knight_moves n start, is_black_square pos :=
sorry

end knight_reach_black_squares_l244_244270


namespace combination_indices_solution_l244_244952

theorem combination_indices_solution (x : ℤ) : 
  ((nat.choose 34 (2 * x) = nat.choose 34 (4 * x - 8)) → (x = 4 ∨ x = 7)) :=
by
  intro h
  sorry

end combination_indices_solution_l244_244952


namespace largest_K_inequality_l244_244853

noncomputable def largest_K : ℝ := 18

theorem largest_K_inequality (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
(h_cond : a * b + b * c + c * a = a * b * c) :
( (a^a * (b^2 + c^2)) / ((a^a - 1)^2) + (b^b * (c^2 + a^2)) / ((b^b - 1)^2) + (c^c * (a^2 + b^2)) / ((c^c - 1)^2) )
≥ largest_K * ((a + b + c) / (a * b * c - 1)) ^ 2 :=
sorry

end largest_K_inequality_l244_244853


namespace coefficient_x_neg_two_in_binomial_expansion_l244_244892

theorem coefficient_x_neg_two_in_binomial_expansion :
  let f (x : ℝ) := x + 9 / x
  let n := 6
  ∀ (x : ℝ), (1 ≤ x ∧ x ≤ 4) → f x ≥ n →
  ( @binomial.coeff _ _ _ (x - 1/x)^n (-2) = 15 ) := by
  sorry

end coefficient_x_neg_two_in_binomial_expansion_l244_244892


namespace num_valid_two_digit_primes_l244_244534

-- Define the set from which the digits are chosen
def digit_set := {3, 5, 7, 9}

-- Define a function to check if a number is a two-digit prime formed by different tens and units digits from digit_set
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ {37, 53, 59, 73, 79, 97} -- Set of prime numbers obtained in the solution

-- Define the main theorem
theorem num_valid_two_digit_primes : (set.filter is_valid_prime { n | ∃ t u, t ≠ u ∧ t ∈ digit_set ∧ u ∈ digit_set ∧ n = 10 * t + u }).card = 7 := 
by
  sorry

end num_valid_two_digit_primes_l244_244534


namespace stuffed_animal_total_l244_244634

/-- McKenna has 34 stuffed animals. -/
def mckenna_stuffed_animals : ℕ := 34

/-- Kenley has twice as many stuffed animals as McKenna. -/
def kenley_stuffed_animals : ℕ := 2 * mckenna_stuffed_animals

/-- Tenly has 5 more stuffed animals than Kenley. -/
def tenly_stuffed_animals : ℕ := kenley_stuffed_animals + 5

/-- The total number of stuffed animals the three girls have. -/
def total_stuffed_animals : ℕ := mckenna_stuffed_animals + kenley_stuffed_animals + tenly_stuffed_animals

/-- Prove that the total number of stuffed animals is 175. -/
theorem stuffed_animal_total : total_stuffed_animals = 175 := by
  sorry

end stuffed_animal_total_l244_244634


namespace find_a_and_b_l244_244413

theorem find_a_and_b : 
  ∃ (a b : ℝ), (0 < a ∧ 0 < b) ∧ 
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ π → y = a * (1 / sin (b * x))) ∧ 
  (∀ x : ℝ, x = π / 4 → (a * (1 / sin (b * (π / 4))) = 4)) ∧ 
  (a = 4) ∧ (b = 2) :=
by
  sorry

end find_a_and_b_l244_244413


namespace parabola_condition_l244_244178

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244178


namespace max_reflections_l244_244424

theorem max_reflections (A B D : Point)
  (C : Line)
  (h1 : BeamStartingFromPoint A)
  (h2 : ReflectingBetweenLines AD and CD)
  (h3 : PerpendicularHitAtPoint B)
  (h4 : ReflectsBackToPointA A)
  (h5 : AngleOfIncidenceEqualsReflexion)
  (angleCDA : Angle CDA = 8) :
  ∃ n : ℕ, n ≤ 10 := sorry

end max_reflections_l244_244424


namespace felipe_total_time_l244_244866

-- Given definitions
def combined_time_without_breaks := 126
def combined_time_with_breaks := 150
def felipe_break := 6
def emilio_break := 2 * felipe_break
def carlos_break := emilio_break / 2

theorem felipe_total_time (F E C : ℕ) 
(h1 : F = E / 2) 
(h2 : C = F + E)
(h3 : (F + E + C) = combined_time_without_breaks)
(h4 : (F + felipe_break) + (E + emilio_break) + (C + carlos_break) = combined_time_with_breaks) : 
F + felipe_break = 27 := 
sorry

end felipe_total_time_l244_244866


namespace volume_of_double_size_cube_l244_244791

theorem volume_of_double_size_cube (V : ℝ) (h : V = 343) : 
  let a := (343 : ℝ)^(1 / 3) in
  let new_side := 2 * a in
  let new_volume := new_side^3 in
  new_volume = 2744 := by
  sorry

end volume_of_double_size_cube_l244_244791


namespace gum_needed_l244_244601

variable (number_of_cousins : ℕ) (gum_per_cousin : ℕ) (total_gum : ℕ)

-- Conditions
axiom h1 : number_of_cousins = 4
axiom h2 : gum_per_cousin = 5

-- Problem: Prove that the total gum needed is 20 pieces.
theorem gum_needed : total_gum = number_of_cousins * gum_per_cousin → total_gum = 20 :=
by
  intro h
  rw [h1, h2] at h
  simp at h
  exact h

end gum_needed_l244_244601


namespace james_planted_60_percent_l244_244112

theorem james_planted_60_percent :
  let total_trees := 2
  let plants_per_tree := 20
  let seeds_per_plant := 1
  let total_seeds := total_trees * plants_per_tree * seeds_per_plant
  let planted_trees := 24
  (planted_trees / total_seeds) * 100 = 60 := 
by
  sorry

end james_planted_60_percent_l244_244112


namespace sin_square_sum_l244_244842

theorem sin_square_sum : 
  (∑ n in Finset.range 91, (Real.sin (↑n * 2 * Real.pi / 180))^2) = 46 :=
by
sorry

end sin_square_sum_l244_244842


namespace pair_comparison_l244_244824

theorem pair_comparison :
  (∀ (a b : ℤ), (a, b) = (-2^4, (-2)^4) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (5^3, 3^5) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = (-(-3), -|-3|) → a ≠ b) ∧
  (∀ (a b : ℤ), (a, b) = ((-1)^2, (-1)^2008) → a = b) :=
by
  sorry

end pair_comparison_l244_244824


namespace circle_integer_points_count_l244_244083

theorem circle_integer_points_count :
  (∃ (x y : ℤ), x^2 + y^2 = 25) ↔ (∃! (p : ℤ × ℤ), let ⟨x, y⟩ := p in x^2 + y^2 = 25 ∧ ∀ (x y : ℤ), x^2 + y^2 = 25 → (x, y) ∈ set.univ) :=
sorry

end circle_integer_points_count_l244_244083


namespace duty_roster_arrangements_l244_244779

theorem duty_roster_arrangements : 
  let people := {p1, p2, p3, p4, p5 : Type} in
  let days := [1, 2, 3, 4, 5] in
  -- Each person can be scheduled for multiple days or not at all,
  -- but the same person cannot be scheduled on consecutive days.
  -- The total number of ways to arrange the duty roster is:
  5 * 4 * 4 * 4 * 4 = 1280 :=
by sorry

end duty_roster_arrangements_l244_244779


namespace collinear_unit_vector_l244_244749

def vector3 := ℝ × ℝ × ℝ

def is_unit_vector (v : vector3) : Prop :=
  let (x, y, z) := v in x^2 + y^2 + z^2 = 1

def are_collinear (v₁ v₂ : vector3) : Prop :=
  ∃ k : ℝ, v₂ = (k * v₁.1, k * v₁.2, k * v₁.3)

def vec_a : vector3 := (3, 0, -4)

def magnitude (v : vector3) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem collinear_unit_vector :
  magnitude vec_a = 5 →
  is_unit_vector (-3/5, 0, 4/5) →
  are_collinear vec_a (-3/5, 0, 4/5) :=
 by
  sorry

end collinear_unit_vector_l244_244749


namespace parabola_p_and_circle_tangent_directrix_l244_244229

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244229


namespace parabola_p_and_circle_tangent_directrix_l244_244235

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244235


namespace balls_into_boxes_l244_244540

theorem balls_into_boxes : (3^6 = 729) :=
by
  calc
    3^6 = 729 : sorry

end balls_into_boxes_l244_244540


namespace largest_non_disjoint_3_subsets_l244_244627

theorem largest_non_disjoint_3_subsets (X : Finset α) (hX : X.card = n) :
  ∃ (S : Finset (Finset α)), (∀ s1 s2 ∈ S, s1 ≠ s2 → (s1 ∩ s2).nonempty) ∧ S.card = (n-1)*(n-2)/2 := 
by sorry

end largest_non_disjoint_3_subsets_l244_244627


namespace parabola_focus_line_l244_244165

theorem parabola_focus_line (p : ℝ) (hp : p > 0) :
  (let focus := (p / 2, 0) in
   ∃ M N : (ℝ × ℝ), 
     let line := λ x, (-√3 * (x - 1)) in
     line (p / 2) = 0
     ∧ M.2 = line M.1
     ∧ N.2 = line N.1
     ∧ (M.2 ^ 2 = 2 * p * M.1)
     ∧ (N.2 ^ 2 = 2 * p * N.1)) → p = 2 :=
by
  intro h
  sorry

end parabola_focus_line_l244_244165


namespace correct_statement_combination_l244_244404

-- The conditions of the problem

-- Condition 1: Check for exactly three numbers divisible by 3 among {123, 365, 293, 18}
def divisible_by_3 (n : ℕ) : Bool := (n % 3 = 0)

-- Condition 2: If the radius of a circle is increased by 20%, the area increases by 44%
def area_increase_20_percent (r : ℝ) : Bool := 
  ((3.14 * (1.2 * r) ^ 2) - (3.14 * r ^ 2)) / (3.14 * r ^ 2) = 0.44

-- Condition 3: 45 has more divisors than 36
def num_divisors (n : ℕ) : ℕ := finset.card (finset.filter (λ d, n % d = 0) (finset.range (n+1)))
def condition_3 : Bool := num_divisors 45 > num_divisors 36

-- Condition 4: Arithmetic mean of {a, -2a, 4a} is a
def geom_prog_ratio_minus2 (a : ℝ) : Bool := ((a + (-2 * a) + 4 * a) / 3 = a)

-- Condition 5: For AP with 10th term < 5 and 12th term > 7, the common difference > 1
def arithmetic_prog_diff_greater_1 (a10 a12 : ℝ) : Bool := 
  (∃ d : ℝ, (a10 < 5) ∧ (a12 > 7) ∧ (d > 1))

-- Condition 6: 6.4 * 10^11 is a square of a natural number
def is_square_of_nat (n : ℝ) : Bool := ∃ m : ℕ, n = (m * m)

-- The statement combining all conditions and the correct answer
theorem correct_statement_combination : 
  let cond1 := (divisible_by_3 123) && !(divisible_by_3 365) && !(divisible_by_3 293) && (divisible_by_3 18)
  let cond2 := area_increase_20_percent 1  -- arbitrary r = 1 for illustration
  let cond3 := !condition_3
  let cond4 := geom_prog_ratio_minus2 1  -- arbitrary a = 1 for illustration
  let cond5 := arithmetic_prog_diff_greater_1 (-4) 5 
  let cond6 := is_square_of_nat (6.4 * 10^11)
in (cond1 = false) && (cond2 = true) && (cond3 = false) && (cond4 = true) && (cond5 = true) && (cond6 = true)
:= sorry

end correct_statement_combination_l244_244404


namespace theater_capacity_filled_l244_244813

variable (numSeats : ℕ)
variable (numPerformances : ℕ)
variable (totalRevenue : ℕ)
variable (ticketPrice : ℕ)

def percentageFilled (numSeats numPerformances totalRevenue ticketPrice : ℕ) : ℕ :=
  let totalTicketsSold := totalRevenue / ticketPrice
  let totalSeatsAvailable := numSeats * numPerformances
  (totalTicketsSold * 100) / totalSeatsAvailable

theorem theater_capacity_filled :
  numSeats = 400 → numPerformances = 3 → totalRevenue = 28800 → ticketPrice = 30 →
  percentageFilled numSeats numPerformances totalRevenue ticketPrice = 80 :=
by
  intros h1 h2 h3 h4
  unfold percentageFilled
  rw [h1, h2, h3, h4]
  dsimp
  norm_num
  sorry

end theater_capacity_filled_l244_244813


namespace average_age_union_l244_244573

theorem average_age_union (a b c d : ℕ)
  (avgA avgB avgC avgD avgAB avgAC avgBC avgABC : ℕ)
  (H1 : avgA = 30)
  (H2 : avgB = 25)
  (H3 : avgC = 35)
  (H4 : avgD = 40)
  (H5 : avgAB = 28)
  (H6 : avgAC = 32)
  (H7 : avgBC = 30)
  (H8 : avgABC = 31)
  : (30 * a + 25 * b + 35 * c + 40 * d) / (a + b + c + d) = 37.5 := 
by 
  sorry

end average_age_union_l244_244573


namespace find_special_n_l244_244445

open Nat

theorem find_special_n (m : ℕ) (hm : m ≥ 3) :
  ∃ (n : ℕ), 
    (n = m^2 - 2) ∧ (∃ (k : ℕ), 1 ≤ k ∧ k < n ∧ 2 * (Nat.choose n k) = (Nat.choose n (k - 1) + Nat.choose n (k + 1))) :=
by
  sorry

end find_special_n_l244_244445


namespace kieran_jump_time_l244_244121

def time_per_jump (seconds : ℕ) (jumps : ℕ) : ℕ := seconds / jumps

def total_time_for_jumps (time_per_jump : ℕ) (jumps : ℕ) : ℕ := time_per_jump * jumps

theorem kieran_jump_time :
  let seconds := 6 in
  let jumps := 4 in
  let total_jumps := 30 in
  let t : ℕ := time_per_jump seconds jumps in
  total_time_for_jumps t total_jumps = 45 :=
by
  sorry

end kieran_jump_time_l244_244121


namespace cylindrical_tank_capacity_correct_l244_244371

noncomputable def cylindrical_tank_capacity : ℝ :=
  let π := Real.pi
  let diameter := 14
  let radius := diameter / 2
  let depth := 12.00482999321725
  π * (radius^2) * depth

theorem cylindrical_tank_capacity_correct :
  (cylindrical_tank_capacity ≈ 1848.221) :=
by
  sorry

end cylindrical_tank_capacity_correct_l244_244371


namespace jacoby_sold_cookies_l244_244589

def jacoby_trip_conditions (total_needed hourly_wage hours_worked lottery_win sister_gift extra_needed cookie_price : ℕ) :=
  total_needed = 5000 ∧
  hourly_wage = 20 ∧
  hours_worked = 10 ∧
  lottery_win = 500 ∧
  sister_gift = 500 ∧
  extra_needed = 3214 ∧
  cookie_price = 4

theorem jacoby_sold_cookies :
  ∃ (cookies_sold : ℕ), jacoby_trip_conditions 5000 20 10 500 500 3214 4 →
  cookies_sold = 21 :=
  begin
    sorry
  end

end jacoby_sold_cookies_l244_244589


namespace poly_division_exceeds_million_l244_244811

theorem poly_division_exceeds_million :
  ∃ n : ℕ, (n > 10^6) ∧ ∀ line : set (ℝ × ℝ), (count_intersections line <= 40) :=
sorry

end poly_division_exceeds_million_l244_244811


namespace determinant_matrix_eq_four_l244_244554

theorem determinant_matrix_eq_four (x : ℝ) :
  let M := matrix.std_basis_matrix 3 3 (λ i j, if i = 0 then (if j = 0 then 3 * x else 2)
                                            else if i = 1 then (if j = 0 then x else 2 * x) 
                                            else 0)
  matrix.det M = 4 ↔ (x = -2/3 ∨ x = 1) :=
by sorry

end determinant_matrix_eq_four_l244_244554


namespace simplify_expr1_simplify_expr2_l244_244421

-- Problem 1
theorem simplify_expr1 : - real.sqrt 20 + (1 / real.sqrt 3)^(-2 : ℝ) + (real.pi - 3.14)^0 - abs (2 - real.sqrt 5) = -3 * real.sqrt 5 + 6 :=
by sorry

-- Problem 2
theorem simplify_expr2 : real.sqrt 2 * (real.sqrt 32 - (1 / real.sqrt 2)) - (real.sqrt 27 + real.sqrt 12) / real.sqrt 3 = 2 :=
by sorry

end simplify_expr1_simplify_expr2_l244_244421


namespace area_of_largest_circle_inside_square_l244_244643

theorem area_of_largest_circle_inside_square
  (A_square : ℝ)
  (pi_approx : ℝ) :
  (A_square = 400) →
  (pi_approx = 3.1) →
  ∃ A_circle : ℝ, A_circle = pi_approx * (real.sqrt A_square / 2) ^ 2 ∧ A_circle = 310 :=
by
  intros h1 h2
  sorry

end area_of_largest_circle_inside_square_l244_244643


namespace range_of_a_l244_244965

theorem range_of_a
    (a : ℝ)
    (h : ∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = 4 → x ^ 2 + y ^ 2 = 1) :
    a ∈ (Set.Ioo (-(3 * Real.sqrt 2 / 2)) (-(Real.sqrt 2 / 2)) ∪ Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by
  sorry

end range_of_a_l244_244965


namespace prime_square_mod_six_l244_244079

theorem prime_square_mod_six (p : ℕ) (hp : Nat.Prime p) (h : p > 5) : p^2 % 6 = 1 :=
by
  sorry

end prime_square_mod_six_l244_244079


namespace vivian_mail_may_l244_244727

theorem vivian_mail_may
  (doubling_pattern : ∀ n : ℕ, VivianMail n.succ = 2 * VivianMail n)
  (april_mail : VivianMail 0 = 5)
  (june_mail : VivianMail 2 = 20)
  (july_mail : VivianMail 3 = 40)
  (august_mail : VivianMail 4 = 80) :
  VivianMail 1 = 10 :=
by
  sorry

end vivian_mail_may_l244_244727


namespace only_A_is_quadratic_l244_244743

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def f_A (x : ℝ) : ℝ := (1/3) * x^2
def f_B (x : ℝ) : ℝ := 3 / (x^2)
def f_C (x : ℝ) : ℝ := -3 * x + 2
def f_D (x : ℝ) : ℝ := x / 3

theorem only_A_is_quadratic :
  is_quadratic f_A ∧ ¬ is_quadratic f_B ∧ ¬ is_quadratic f_C ∧ ¬ is_quadratic f_D :=
by
  sorry

end only_A_is_quadratic_l244_244743


namespace magic_king_total_episodes_l244_244704

theorem magic_king_total_episodes :
  (∑ i in finset.range 5, 20) + (∑ j in finset.range 5, 25) = 225 :=
by sorry

end magic_king_total_episodes_l244_244704


namespace translated_coordinates_of_B_l244_244987

-- Define the initial coordinates of points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-2, 0)

-- Define the translated coordinates of point A
def A' : ℝ × ℝ := (4, 0)

-- Define the expected coordinates of point B' after the translation
def B' : ℝ × ℝ := (1, -1)

-- Proof statement
theorem translated_coordinates_of_B (A A' B : ℝ × ℝ) (B' : ℝ × ℝ) :
  A = (1, 1) ∧ A' = (4, 0) ∧ B = (-2, 0) → B' = (1, -1) :=
by
  intros h
  sorry

end translated_coordinates_of_B_l244_244987


namespace min_value_of_expression_l244_244477

theorem min_value_of_expression (d : ℝ) (a_n S_n : ℕ → ℝ) :
  (∀ n : ℕ, a_n = 2 * n - 1) →
  (∀ n : ℕ, S_n = n^2) →
  d ≠ 0 →
  a_3^2 = a_1 * a_{13} →
  a_1 = 1 →
  (∀ n : ℕ, 2 * S_n + 16 / (a_n + 3) ≥ 4) :=
by
  intro h1 h2 hd h3 h4
  have h5 : ∀ n, 2 * S_n + 16 / (a_n + 3) = 4 :=
  sorry
  exact h5

end min_value_of_expression_l244_244477


namespace check_equations_l244_244072

-- Define the points
variables (A B C D : ℝ × ℝ)

-- Define the distance function
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Define the three given equations
def eq1 : Prop := dist A B + dist C D = dist A C + dist B D
def eq2 : Prop := dist A B + dist B C = dist A C + dist B D
def eq3 : Prop := dist A B - dist C D = dist A C + dist B D

theorem check_equations : ∃ n, n ∈ {0, 1, 2, 3} ∧
  ((∀ P1 P2 P3 P4 : ℝ × ℝ, dist P1 P2 + dist P3 P4 ≠ dist P1 P3 + dist P2 P4 ) → n = 0) ∧
  ((∀ P1 P2 P3 P4 : ℝ × ℝ, dist P1 P2 + dist P2 P3 ≠ dist P1 P3 + dist P2 P4 ) → n = 1) ∧
  ((∀ P1 P2 P3 P4 : ℝ × ℝ, dist P1 P2 - dist P3 P4 ≠ dist P1 P3 + dist P2 P4 ) → n = 2) ∧
  -- More conditions or specifics can be added here to finalize
  sorry -- The final completion of which conditions match or conflict.

end check_equations_l244_244072


namespace find_center_of_rotation_l244_244427

theorem find_center_of_rotation :
  ∃ c : ℂ, (f : ℂ → ℂ) = λ z, (1 + complex.I * real.sqrt 2) * z + (3 * real.sqrt 2 - 12 * complex.I) / 3 ∧
    f c = c ∧ c = -3 * real.sqrt 2 - 15 * complex.I :=
by
  sorry

end find_center_of_rotation_l244_244427


namespace tan_values_l244_244110

theorem tan_values (a : ℝ) (tan_a : ℤ) (tan_3a : ℤ) (h1 : tan(tan_a : ℝ) = a) (h2 : tan(tan_3a : ℝ) = 3 * a) :
  a = -1 ∨ a = 0 ∨ a = 1 := 
sorry

end tan_values_l244_244110


namespace convex_polyhedron_has_triangular_face_l244_244962

def convex_polyhedron : Type := sorry -- placeholder for the type of convex polyhedra
def face (P : convex_polyhedron) : Type := sorry -- placeholder for the type of faces of a polyhedron
def vertex (P : convex_polyhedron) : Type := sorry -- placeholder for the type of vertices of a polyhedron
def edge (P : convex_polyhedron) : Type := sorry -- placeholder for the type of edges of a polyhedron

-- The number of edges meeting at a specific vertex
def vertex_degree (P : convex_polyhedron) (v : vertex P) : ℕ := sorry

-- Number of edges or vertices on a specific face
def face_sides (P : convex_polyhedron) (f : face P) : ℕ := sorry

-- A polyhedron is convex
def is_convex (P : convex_polyhedron) : Prop := sorry

-- A face is a triangle if it has 3 sides
def is_triangle (P : convex_polyhedron) (f : face P) := face_sides P f = 3

-- The problem statement in Lean 4
theorem convex_polyhedron_has_triangular_face
  (P : convex_polyhedron)
  (h1 : is_convex P)
  (h2 : ∀ v : vertex P, vertex_degree P v ≥ 4) :
  ∃ f : face P, is_triangle P f :=
sorry

end convex_polyhedron_has_triangular_face_l244_244962


namespace gcd_of_128_144_480_450_l244_244328

theorem gcd_of_128_144_480_450 : Nat.gcd (Nat.gcd 128 144) (Nat.gcd 480 450) = 6 := 
by
  sorry

end gcd_of_128_144_480_450_l244_244328


namespace value_of_p_circle_tangent_to_directrix_l244_244145

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244145


namespace total_number_of_legs_in_house_l244_244836

def humans := 5
def dogs := 2
def cats := 3
def parrots := 4
def goldfish := 5
def legs_per_human := 2
def legs_per_dog := 4
def legs_per_cat := 4
def legs_per_parrot := 2
def legs_per_goldfish := 0

theorem total_number_of_legs_in_house : 
  humans * legs_per_human + dogs * legs_per_dog + cats * legs_per_cat + parrots * legs_per_parrot + goldfish * legs_per_goldfish = 38 :=
by 
  unfold humans dogs cats parrots goldfish legs_per_human legs_per_dog legs_per_cat legs_per_parrot legs_per_goldfish
  norm_num
  sorry

end total_number_of_legs_in_house_l244_244836


namespace gcd_polynomial_example_l244_244914

theorem gcd_polynomial_example (b : ℤ) (h : ∃ k : ℤ, b = 2 * 1177 * k) :
  Int.gcd (3 * b^2 + 34 * b + 76) (b + 14) = 2 :=
by
  sorry

end gcd_polynomial_example_l244_244914


namespace area_ABC_l244_244974

noncomputable def point := (ℝ × ℝ)

variables (A B C D E F : point)

-- Given conditions
variables (H1 : ∀ x y : point, D = (x + y) / 2)  -- D is the midpoint της BC
variables (H2 : ∃ k : ℝ, k > 0 ∧ E = k • A + (1 - k) • C ∧ 2 * (1 - k) = 3 * k)  -- AE:EC = 2:3
variables (H3 : ∃ l : ℝ, l > 0 ∧ F = l • A + (1 - l) • D ∧ 2 * (1 - l) = 1 * l)  -- AF:FD = 2:1
variables (H4 : ∃ area_def : ℝ, area_def > 0 ∧ area_def = 20)  -- Area of DEF is 20

-- Goal
theorem area_ABC : ∃ area_ABC : ℝ, area_ABC = 300 :=
sorry

end area_ABC_l244_244974


namespace price_of_books_sold_at_lower_price_l244_244392

-- Define the conditions
variable (n m p q t : ℕ) (earnings price_high price_low : ℝ)

-- The given conditions
def total_books : ℕ := 10
def books_high_price : ℕ := 2 * total_books / 5 -- 2/5 of total books
def books_low_price : ℕ := total_books - books_high_price
def high_price : ℝ := 2.50
def total_earnings : ℝ := 22

-- The proposition to prove
theorem price_of_books_sold_at_lower_price
  (h_books_high_price : books_high_price = 4)
  (h_books_low_price : books_low_price = 6)
  (h_total_earnings : total_earnings = 22)
  (h_high_price : high_price = 2.50) :
  (price_low = 2) := 
-- Proof goes here 
sorry

end price_of_books_sold_at_lower_price_l244_244392


namespace line_passes_through_fixed_point_l244_244269

theorem line_passes_through_fixed_point :
  ∀ k : ℝ, ∃ (x y : ℝ), (2 * k - 1) * x - (k + 3) * y - (k - 11) = 0 ∧ x = 2 ∧ y = 3 :=
by
  intro k
  use 2, 3
  constructor
  · calc
      (2 * k - 1) * 2 - (k + 3) * 3 - (k - 11)
          = (4 * k - 2) - (3 * k + 9) - (k - 11) : by ring
      ... = 4 * k - 2 - 3 * k - 9 - k + 11 : by ring
      ... = 4 * k - 3 * k - k - 11 : by ring
      ... = 0 : by ring
  constructor
  · rfl
  · rfl

end line_passes_through_fixed_point_l244_244269


namespace volume_of_rock_correct_l244_244381

-- Define the initial conditions
def tank_length := 30
def tank_width := 20
def water_depth := 8
def water_level_rise := 4

-- Define the volume function for the rise in water level
def calculate_volume_of_rise (length: ℕ) (width: ℕ) (rise: ℕ) : ℕ :=
  length * width * rise

-- Define the target volume of the rock
def volume_of_rock := 2400

-- The theorem statement that the volume of the rock is 2400 cm³
theorem volume_of_rock_correct :
  calculate_volume_of_rise tank_length tank_width water_level_rise = volume_of_rock :=
by 
  sorry

end volume_of_rock_correct_l244_244381


namespace normal_trip_distance_l244_244115

variable (S D : ℝ)

-- Conditions
axiom h1 : D = 3 * S
axiom h2 : D + 50 = 5 * S

theorem normal_trip_distance : D = 75 :=
by
  sorry

end normal_trip_distance_l244_244115


namespace inverse_direct_variation_l244_244631

theorem inverse_direct_variation (k : ℝ)
  (h1 : ∀ x y z : ℝ, x = k * (z / y^2))
  (h2 : ∀ y : ℝ, z = 2 * y - 1)
  (h3 : h1 1 3 5)
  (h4 : h2 3)
  : ∀ y x z : ℝ, y = 4 ∧ z = 7 → x = 63 / 80 :=
  by
    sorry

end inverse_direct_variation_l244_244631


namespace parabola_condition_l244_244170

noncomputable section

-- Define the parabola with parameter p
def parabola (p : ℝ) (h : p > 0) : set (ℝ × ℝ) :=
  {pt | pt.2 ^ 2 = 2 * p * pt.1}

-- Define the line equation
def line (x y : ℝ) : Prop :=
  y = -sqrt 3 * (x - 1)

-- Define the focus of the parabola
def focus (p : ℝ) : ℝ × ℝ :=
  (p / 2, 0)

-- Directrix of the parabola
def directrix (p : ℝ) : ℝ :=
  -p / 2

-- Check if the circle with MN as its diameter is tangent to the directrix
def isTangent (p : ℝ) (M N : ℝ × ℝ)
  (hM : M ∈ parabola p sorry)
  (hN : N ∈ parabola p sorry)
  : Prop :=
  let mid := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  let rad := (M.1 - N.1) / 2
  abs (mid.1 - directrix p) = rad

theorem parabola_condition (p : ℝ) (M N : ℝ × ℝ)
  (h : p > 0)
  (line_through_focus : line (p / 2) 0)
  (hM : M ∈ parabola p h)
  (hN : N ∈ parabola p h) :
  (p = 2) ∧ (isTangent p M N hM hN) :=
sorry

end parabola_condition_l244_244170


namespace p_eq_two_circle_tangent_proof_l244_244218

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244218


namespace parabola_p_and_circle_tangent_directrix_l244_244230

theorem parabola_p_and_circle_tangent_directrix :
  ∀ (p : ℝ) (M N : ℝ × ℝ), 
  (p > 0) →
  ((M, N) = Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                        (λ (x y : ℝ), y = -√3 * (x - 1)))) →
  ∃ (M N : ℝ × ℝ), 
  (Classical.some (exists_intersection (λ (x : ℝ), x ≥ 0 ∧ y^2 = 2 * p * x) 
                                   (λ (x y : ℝ), y = -√3 * (x - 1)))) = (M, N) → 
  p = 2 ∧ 
  ((distance_to_directrix ((M.1 + N.1) / 2, 0) (-p / 2) (circle_radius (M, N))) = 0) :=
begin
  sorry
end

end parabola_p_and_circle_tangent_directrix_l244_244230


namespace multiples_of_7_not_14_l244_244948

theorem multiples_of_7_not_14 (n : ℕ) : 
  ∃ count : ℕ, count = 25 ∧ 
    (∀ x, x < 350 → x % 7 = 0 → x % 14 ≠ 0 ↔ ∃ k, x = 7 * k ∧ k % 2 = 1) := 
by
  have count := (finset.range' 7 350).countp (λ x, x % 7 = 0 ∧ x % 14 ≠ 0)
  have h_count : count = 25 := sorry 
  exact ⟨25, h_count, sorry⟩

end multiples_of_7_not_14_l244_244948


namespace exist_tangent_circle_l244_244007

/-- Given a circle k and two points A and B outside the circle, 
there exists a circle ell such that it passes through points A 
and B and touches the circle k. -/
theorem exist_tangent_circle (k : Circle) (A B : Point)
  (hA : ¬ A ∈ k) (hB : ¬ B ∈ k)
  : ∃ ℓ : Circle, A ∈ ℓ ∧ B ∈ ℓ ∧ tangent ℓ k :=
  sorry

end exist_tangent_circle_l244_244007


namespace cricket_score_percentage_running_l244_244370

theorem cricket_score_percentage_running
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (singles : ℕ)
  (doubles : ℕ)
  (threes : ℕ)
  (runs_per_boundary : ℕ := 4)
  (runs_per_six : ℕ := 6)
  (runs_per_single : ℕ := 1)
  (runs_per_double : ℕ := 2)
  (runs_per_three : ℕ := 3)
  (expected_percentage : ℚ := 31.64)
  (H_total_runs : total_runs = 256)
  (H_boundaries : boundaries = 18)
  (H_sixes : sixes = 5)
  (H_singles : singles = 40)
  (H_doubles : doubles = 16)
  (H_threes : threes = 3) :
  let 
    runs_from_boundaries := boundaries * runs_per_boundary,
    runs_from_sixes := sixes * runs_per_six,
    total_runs_from_boundaries_and_sixes := runs_from_boundaries + runs_from_sixes,
    runs_from_running := (singles * runs_per_single) + (doubles * runs_per_double) + (threes * runs_per_three)
  in
  (runs_from_running : ℚ) / (total_runs : ℚ) * 100 = expected_percentage := 
sorry

end cricket_score_percentage_running_l244_244370


namespace profit_function_simplified_maximize_profit_l244_244368

-- Define the given conditions
def cost_per_product : ℝ := 3
def management_fee_per_product : ℝ := 3
def annual_sales_volume (x : ℝ) : ℝ := (12 - x) ^ 2

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - (cost_per_product + management_fee_per_product)) * annual_sales_volume x

-- Define the bounds for x
def x_bounds (x : ℝ) : Prop := 9 ≤ x ∧ x ≤ 11

-- Prove the profit function in simplified form
theorem profit_function_simplified (x : ℝ) (h : x_bounds x) :
    profit x = x ^ 3 - 30 * x ^ 2 + 288 * x - 864 :=
by
  sorry

-- Prove the maximum profit and the corresponding x value
theorem maximize_profit (x : ℝ) (h : x_bounds x) :
    (∀ y, (∃ x', x_bounds x' ∧ y = profit x') → y ≤ 27) ∧ profit 9 = 27 :=
by
  sorry

end profit_function_simplified_maximize_profit_l244_244368


namespace complex_z_conjugate_correct_l244_244496

noncomputable def complex_z_solution : Prop :=
  ∃ (z : ℂ), (i - 1) * z = 4 - 2 * i ∧ (conj z = -3 - i)

theorem complex_z_conjugate_correct : complex_z_solution :=
  sorry

end complex_z_conjugate_correct_l244_244496


namespace find_a4_l244_244903

-- Define the arithmetic sequence and the sum of the first N terms
def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

-- Sum of the first N terms in an arithmetic sequence
def sum_arithmetic_seq (a d N : ℕ) : ℕ := N * (2 * a + (N - 1) * d) / 2

-- Define the conditions
def condition1 (a d : ℕ) : Prop := a + (a + 2 * d) + (a + 4 * d) = 15
def condition2 (a d : ℕ) : Prop := sum_arithmetic_seq a d 4 = 16

-- Lean 4 statement to prove the value of a_4
theorem find_a4 (a d : ℕ) (h1 : condition1 a d) (h2 : condition2 a d) : arithmetic_seq a d 4 = 7 :=
sorry

end find_a4_l244_244903


namespace find_arithmetic_sequence_l244_244723

theorem find_arithmetic_sequence (a d : ℝ) (h1 : (a - d) + a + (a + d) = 12) (h2 : (a - d) * a * (a + d) = 48) :
  (a = 4 ∧ d = 2) ∨ (a = 4 ∧ d = -2) :=
sorry

end find_arithmetic_sequence_l244_244723


namespace num_candidates_math_dept_l244_244788

theorem num_candidates_math_dept (m : ℕ) : 
  ((7.choose 2 = 21) ∧ (m * 21 = 84)) → m = 4 :=
by
  intros h,
  cases h with h1 h2,
  have h3 : 21 = 21 := rfl, -- This line simply confirms 21 is 21; not really necessary, but clarifies that 21 is fixed.
  rw ←h1 at h3,
  have h4 : m * 21 = 84 := h2,
  sorry -- Skipping the actual proof steps; provided it's math equivalent process.

end num_candidates_math_dept_l244_244788


namespace hyperbola_trajectory_center_l244_244858

theorem hyperbola_trajectory_center :
  ∀ m : ℝ, ∃ (x y : ℝ), x^2 - y^2 - 6 * m * x - 4 * m * y + 5 * m^2 - 1 = 0 ∧ 2 * x + 3 * y = 0 :=
by
  sorry

end hyperbola_trajectory_center_l244_244858


namespace primes_product_less_than_20_l244_244333

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the product of a list of natural numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

theorem primes_product_less_than_20 :
  product primes_less_than_20 = 9699690 :=
by
  sorry

end primes_product_less_than_20_l244_244333


namespace ratio_of_height_to_radius_min_surface_area_l244_244011

theorem ratio_of_height_to_radius_min_surface_area 
  (r h : ℝ)
  (V : ℝ := 500)
  (volume_cond : π * r^2 * h = V)
  (surface_area : ℝ := 2 * π * r^2 + 2 * π * r * h) : 
  h / r = 2 :=
by
  sorry

end ratio_of_height_to_radius_min_surface_area_l244_244011


namespace largest_n_for_triangle_property_l244_244812

theorem largest_n_for_triangle_property :
  ∀ (S : Set ℕ), (∀ (a b c : ℕ), a ∈ S → b ∈ S → c ∈ S →
  a + b > c ∧ a + c > b ∧ b + c > a → 
  (6 ∈ S) → (7 ∈ S) →
  (∀ n ∈ (Set.range 33), S n) →
  (∃ (T : Finset ℕ), T.card = 5 → (∀ (a b c : ℕ), a, b, c ∈ T → a ≠ b → b ≠ c → a ≠ c → a + b > c ∧ a + c > b ∧ b + c > a) ∧ 33 ∉ S) →
  has_upper_bound (range (33 : Set ℕ)) :=
  sorry

end largest_n_for_triangle_property_l244_244812


namespace find_constant_value_l244_244078

variable (log10 : ℝ → ℝ)
-- Assume that log10 behaves as the base 10 logarithm
axiom log10_def : ∀ x > 0, log10 x = Real.log x / Real.log 10

variable (k : ℝ)

def condition1 (x : ℝ) : Prop :=
  log10 5 + log10 (5 * x + 1) = log10 (x + 5) + k

def condition2 : ℝ := 3

theorem find_constant_value (h1 : condition1 (condition2)) : k = 1 :=
by
  sorry

end find_constant_value_l244_244078


namespace increasing_a_eq_1_minimum_value_a_eq_e_l244_244931

noncomputable def f (x a : ℝ) : ℝ :=
  x - (a / x) - (a + 1) * Real.log x

theorem increasing_a_eq_1 (a : ℝ) : 
  (∀ x > 0, ∀ y > 0, x < y → f x 1 ≤ f y 1) ↔ (a = 1) :=
sorry

theorem minimum_value_a_eq_e (a : ℝ) : 
  (∃ x ∈ (Set.Icc 1 Real.exp 1), f x e = -2) ↔ (a = e) :=
sorry

end increasing_a_eq_1_minimum_value_a_eq_e_l244_244931


namespace sum_of_squares_and_sqrt_inequality_l244_244470

theorem sum_of_squares_and_sqrt_inequality
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (h : x + y + z = 1) :
  x^2 + y^2 + z^2 + 2 * real.sqrt (3 * x * y * z) ≤ 1 :=
begin
  sorry
end

end sum_of_squares_and_sqrt_inequality_l244_244470


namespace max_vertices_no_diagonal_correct_l244_244331

open Nat

def max_vertices_no_diagonal (n : ℕ) : ℕ :=
  n / 2

theorem max_vertices_no_diagonal_correct (n : ℕ) :
  ∀ (n_gon : list (ℝ × ℝ)),
    (length n_gon = n ∧ convex_hull n_gon ≠ n_gon) →
    (∃ (m : ℕ), m = floor (n / 2)) :=
by
  intros n_gon h
  rcases h with ⟨hn, hconv⟩
  use n / 2
  apply congr_arg
  sorry

end max_vertices_no_diagonal_correct_l244_244331


namespace sum_of_external_angles_of_octagon_staircase_slope_reduction_l244_244651

-- Problem A
theorem sum_of_external_angles_of_octagon : ∑ x in (fin 8).map (λ i, external_angles_octagon i), x = 360 :=
by sorry

-- Problem B
theorem staircase_slope_reduction :
  abs ((2.7 / real.sin (35 * real.pi / 180)) - (2.7 / real.sin (46 * real.pi / 180)) - 0.95) < 0.01 :=
by sorry

end sum_of_external_angles_of_octagon_staircase_slope_reduction_l244_244651


namespace cat_mouse_positions_l244_244100

/-- Movement patterns for the cat and the mouse. The cat moves clockwise through
    four squares and the mouse moves counterclockwise through eight exterior sections.
    Prove that after 359 moves, the cat will be in the bottom right corner and 
    the mouse will be in the left middle section. -/
theorem cat_mouse_positions(
  cat_moves_clockwise : ∀ (n : ℕ), n % 4 = 1 → "cat in top left" | n % 4 = 2 → "cat in top right" 
    | n % 4 = 3 → "cat in bottom right" | n % 4 = 0 → "cat in bottom left",
  mouse_moves_counterclockwise : ∀ (n : ℕ), n % 8 = 1 → "mouse in top middle" 
    | n % 8 = 2 → "mouse in top right" | n % 8 = 3 → "mouse in right middle" 
    | n % 8 = 4 → "mouse in bottom right" | n % 8 = 5 → "mouse in bottom middle" 
    | n % 8 = 6 → "mouse in bottom left" | n % 8 = 7 → "mouse in left middle" 
    | n % 8 = 0 → "mouse in top left"): 
  (cat_moves_clockwise 359 = "cat in bottom right") ∧ (mouse_moves_counterclockwise 359 = "mouse in left middle") :=
by
  sorry

end cat_mouse_positions_l244_244100


namespace magic_king_episodes_proof_l244_244712

-- Let's state the condition in terms of the number of seasons and episodes:
def total_episodes (seasons: ℕ) (episodes_first_half: ℕ) (episodes_second_half: ℕ) : ℕ :=
  (seasons / 2) * episodes_first_half + (seasons / 2) * episodes_second_half

-- Define the conditions for the "Magic King" show
def magic_king_total_episodes : ℕ :=
  total_episodes 10 20 25

-- The statement of the problem - to prove that the total episodes is 225
theorem magic_king_episodes_proof : magic_king_total_episodes = 225 :=
by
  sorry

end magic_king_episodes_proof_l244_244712


namespace toy_difference_l244_244830

variable (A M T : ℕ)

theorem toy_difference :
  M = 6 →
  A = 4 * M →
  A + M + T = 56 →
  |A - T| = 2 :=
by
  intros hM hA hSum
  sorry

end toy_difference_l244_244830


namespace union_A_B_eq_l244_244480

def A := { x : ℝ | real.log (x - 1) < 0 }
def B := { y : ℝ | ∃ x : ℝ, x ∈ A ∧ y = 2^x - 1 }

theorem union_A_B_eq : A ∪ B = {y : ℝ | 1 < y ∧ y < 3} :=
by sorry

end union_A_B_eq_l244_244480


namespace josh_marbles_l244_244596

/-- Josh had 16 marbles in his collection. He decided to triple his collection before losing 25% of them. 
    Prove that he now has 36 marbles in his collection. -/
theorem josh_marbles (initial_marbles : ℕ) (triple_factor : ℕ) (lost_percentage : ℝ) :
  initial_marbles = 16 → triple_factor = 3 → lost_percentage = 0.25 → 
  (initial_marbles * triple_factor - (initial_marbles * triple_factor * lost_percentage).to_nat) = 36 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  have step1 : 16 * 3 = 48 := by norm_num
  have step2 : (48 * 0.25 : ℝ) = 12 := by norm_num
  have step3 : (48 - 12).to_nat = 36 := by norm_num
  sorry

end josh_marbles_l244_244596


namespace range_of_a_l244_244548

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≤ 0) → 4 ≤ a := by
  assume H
  have H_max : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 ≤ 4 := by
    intro x hx
    calc
      x^2 ≤ 2^2 : by nlinarith [hx.left, hx.right]
      ... = 4   : by norm_num
  specialize H 2
  have h2 : 1 ≤ 2 ∧ 2 ≤ 2 := by norm_num
  specialize H h2
  have : 4 - a ≤ 0 := by
    exact H
  linarith

end range_of_a_l244_244548


namespace probability_at_least_6_heads_in_9_flips_l244_244799

theorem probability_at_least_6_heads_in_9_flips : 
  let total_outcomes := 2 ^ 9 in
  let successful_outcomes := Nat.choose 9 6 + Nat.choose 9 7 + Nat.choose 9 8 + Nat.choose 9 9 in
  successful_outcomes.toRational / total_outcomes.toRational = (130 : ℚ) / 512 :=
by
  sorry

end probability_at_least_6_heads_in_9_flips_l244_244799


namespace prove_p_equals_2_l244_244195

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244195


namespace parabola_conditions_l244_244243

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244243


namespace petya_can_divide_l244_244781

theorem petya_can_divide (n : ℕ) (c₁ c₂ : ℕ × ℕ) (h₁ : c₁.1 < 2 * n ∧ c₁.2 < 2 * n)
  (h₂ : c₂.1 < 2 * n ∧ c₂.2 < 2 * n) : ∃ f : (ℕ × ℕ) → bool, 
  (∀ (cell : ℕ × ℕ), cell.1 < 2 * n ∧ cell.2 < 2 * n → f cell = f c₁ ∨ f cell = f c₂) ∧ f c₁ ≠ f c₂ :=
begin
  sorry
end

end petya_can_divide_l244_244781


namespace math_problem_l244_244417

-- Define factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- State the theorem
theorem math_problem : fact 8 - 7 * fact 7 - 2 * fact 7 = -5040 := by
  sorry

end math_problem_l244_244417


namespace correct_commutative_property_usage_l244_244348

-- Definitions for the transformations
def transformA := 3 + (-2) = 2 + 3
def transformB := 4 + (-6) + 3 = (-6) + 4 + 3
def transformC := (5 + (-2)) + 4 = (5 + (-4)) + 2
def transformD := (1 / 6) + (-1) + (5 / 6) = ((1 / 6) + (5 / 6)) + 1

-- The theorem stating that transformB uses the commutative property correctly
theorem correct_commutative_property_usage : transformB :=
by
  sorry

end correct_commutative_property_usage_l244_244348


namespace relationship_of_y_values_l244_244091

theorem relationship_of_y_values (m n y1 y2 y3 : ℝ) (h1 : m < 0) (h2 : n > 0) 
  (hA : y1 = m * (-2) + n) (hB : y2 = m * (-3) + n) (hC : y3 = m * 1 + n) :
  y3 < y1 ∧ y1 < y2 := 
by 
  sorry

end relationship_of_y_values_l244_244091


namespace value_of_p_circle_tangent_to_directrix_l244_244144

-- Define the parabola and its properties
def parabola (p : ℝ) : { x : ℝ × ℝ // p > 0 ∧ x.2^2 = 2 * p * x.1 } :=
sorry

-- Define the line equation and its intersection with the parabola
def line_through_focus_intersects_parabola (p : ℝ) : { M N : ℝ × ℝ // 
  (y : (p > 0) ∧ (y = -sqrt(3) * (x - 1))) ∧ y passes through focus of the parabola (p/2, 0) 
  ∧ y intersects parabola C at M and N 
} :=
sorry

-- Define the correct value of p
theorem value_of_p : ∀ (p : ℝ), parabola p → (y = -sqrt(3) * (x - 1)) → 
  (focus : (p > 0) ∧ y passes through (p/2, 0)) → 
  p = 2 :=
by
  intros p h_parabola h_line_through_focus h_focus
  have h1 := (y passes through (p/2, 0))
  have h2 := solve for p to get 0 = -sqrt(3) * (p/2 - 1)
  have H := p = 2
  show p = 2, from H

-- Define if the circle with MN as diameter is tangent to the directrix
theorem circle_tangent_to_directrix : ∀ (p : ℝ), parabola p → 
  line_through_focus_intersects_parabola p → 
  (circle : radius = (|MN|/2)) ∧ (directrix = x = -1) ∧ 
  (distance = midpoint to directrix = radius) → 
  circle is tangent to directrix x = -1 :=
by
  intros p h_parabola h_line_through_focus h_directrix
  have h1 := midpoint of M and N
  have h2 := radius equals distance 1 + (5/3)
  have H := circle is tangent to directrix
  show circle is tangent to directrix, from H
sorry

end value_of_p_circle_tangent_to_directrix_l244_244144


namespace total_surface_area_of_cut_prism_l244_244782

theorem total_surface_area_of_cut_prism :
  let height_A := 1 / 4
      height_B := 1 / 5
      height_C := 1 / 6
      height_D := 1 - (height_A + height_B + height_C),
      base_area := 1 * 2,
      top_and_bottom_area := 4 * 1,
      side_area := 2 * (1 * 1),
      front_and_back_area := 2 * (2 * 1),
      total_surface_area := top_and_bottom_area + side_area + front_and_back_area
  in total_surface_area = 16 :=
by 
  let height_A := 1 / 4
  let height_B := 1 / 5
  let height_C := 1 / 6
  let height_D := 1 - (height_A + height_B + height_C)
  let base_area := 1 * 2
  let top_and_bottom_area := 4 * 1
  let side_area := 2 * (1 * 1)
  let front_and_back_area := 2 * (2 * 1)
  let total_surface_area := top_and_bottom_area + side_area + front_and_back_area
  have h : total_surface_area = 16 := sorry
  exact h

end total_surface_area_of_cut_prism_l244_244782


namespace length_of_FD_l244_244097

theorem length_of_FD : 
  let E := (8 : ℝ) / 3 in 
  let x := (32 : ℝ) / 9 in
  ∀ (F : ℝ), 
    8 - x = F ∧ 
    (8 - F) ^ 2 = x^2 + E^2 → 
    F = x := 
by
  intros E x F h,
  cases h,
  sorry

end length_of_FD_l244_244097


namespace comparison_inequality_l244_244894

noncomputable def x : ℝ := 1.2 ^ 0.9
noncomputable def y : ℝ := 1.1 ^ 0.8
noncomputable def z : ℝ := Real.logb 1.2 0.9

theorem comparison_inequality : x > y ∧ y > z := by
  sorry

end comparison_inequality_l244_244894


namespace labeling_ways_24_l244_244930

noncomputable def number_of_labelings : ℕ :=
  let grid_points := {(x : ℕ, y : ℕ) | x < 4 ∧ y < 4}
  let distances := {d | ∃ (p1 p2 : (ℕ × ℕ)), p1 ∈ grid_points ∧ p2 ∈ grid_points ∧ d = dist p1 p2}
  Nat.card {seq | seq ⊆ grid_points ∧ seq.card = 10 ∧ strictly_increasing (λ (i : ℕ), dist (seq.take i) (seq.drop i))}

theorem labeling_ways_24 : number_of_labelings = 24 := by
  sorry

end labeling_ways_24_l244_244930


namespace find_number_l244_244346

theorem find_number (x : ℝ) (h : 7 * x + 21.28 = 50.68) : x = 4.2 :=
sorry

end find_number_l244_244346


namespace max_value_range_l244_244933

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := 
if x ≤ a then -(x + 1) * Real.exp x else b * x - 1

theorem max_value_range (a b : ℝ) (M : ℝ) 
  (h1 : ∃ x, f x a b = M) 
  (h2 : ∀ x, f x a b ≤ M) : 
  0 < M ∧ M ≤ 1 / Real.exp 2 :=
begin
  sorry
end

end max_value_range_l244_244933


namespace magic_king_total_episodes_l244_244709

theorem magic_king_total_episodes
  (total_seasons : ℕ)
  (first_half_seasons : ℕ)
  (second_half_seasons : ℕ)
  (episodes_first_half : ℕ)
  (episodes_second_half : ℕ)
  (h1 : total_seasons = 10)
  (h2 : first_half_seasons = total_seasons / 2)
  (h3 : second_half_seasons = total_seasons / 2)
  (h4 : episodes_first_half = 20)
  (h5 : episodes_second_half = 25)
  : (first_half_seasons * episodes_first_half + second_half_seasons * episodes_second_half) = 225 :=
by
  sorry

end magic_king_total_episodes_l244_244709


namespace first_player_wins_game_l244_244993

theorem first_player_wins_game :
  ∀ (strip_length start_position : ℕ)
    (move : ℕ → ℕ)
    (can_move : ℕ → ℕ → Prop),
    strip_length = 2005 →
    start_position = 1003 →
    (∀ k, move k = 2^(k-1)) →
    (∀ pos k, can_move pos k ↔ pos + move k <= strip_length ∨ pos ≥ move k) →
    (∀ (start_position : ℕ) (moves : ℕ → ℕ) (player_move : ℕ → ℕ)
      (valid_move : ℕ → ℕ → Prop)
      (turns : ℕ → Prop),
      start_position = 1003 →
      ∀ n, turns n →
        (player_move n + moves n ≤ strip_length ∨ player_move n ≥ moves n) →
        game_winner = "First").

end first_player_wins_game_l244_244993


namespace volume_of_tank_l244_244815

theorem volume_of_tank
  (inlet_rate : ℕ := 3) -- cubic inches per minute
  (outlet_rate1 : ℕ := 12) -- cubic inches per minute
  (outlet_rate2 : ℕ := 6) -- cubic inches per minute
  (emptying_time : ℕ := 3456) -- minutes
  (inches_per_foot : ℕ := 12) :
  (inlet_rate, outlet_rate1, outlet_rate2, emptying_time, inches_per_foot) →
  let net_rate := (outlet_rate1 + outlet_rate2 - inlet_rate) 
  let volume_in_inches := net_rate * emptying_time 
  let inches_per_cubic_foot := inches_per_foot * inches_per_foot * inches_per_foot
  let volume_in_feet := volume_in_inches / inches_per_cubic_foot
  volume_in_feet = 30 :=
by 
  -- defining the parameters
  intros
  -- defining intermediate variables
  let net_rate := (outlet_rate1 + outlet_rate2 - inlet_rate)
  let volume_in_inches := net_rate * emptying_time
  let inches_per_cubic_foot := inches_per_foot * inches_per_foot * inches_per_foot
  let volume_in_feet := volume_in_inches / inches_per_cubic_foot
  -- asserting the volume
  exact sorry

end volume_of_tank_l244_244815


namespace trig_identity_solution_l244_244756

-- Conditions for the function's domain
def valid_domain (x : ℝ) : Prop := 
  cos (2 * x) ≠ 0 ∧ sin (2 * x) ≠ 0 ∧ ∃ (n : ℤ), 2 * x ≠ (Real.pi / 2) + n * Real.pi

-- The main theorem to be proved
theorem trig_identity_solution (x : ℝ) (k : ℤ) : 
  valid_domain x → 
  (1 / (Real.tan(2 * x) ^ 2 + (cos (2 * x))⁻²) + 
   1 / (Real.cot(2 * x) ^ 2 + (sin (2 * x))⁻²) = 2 / 3) → 
  x = (Real.pi / 8) * (2 * k + 1) :=
sorry

end trig_identity_solution_l244_244756


namespace problem_ellipse_problem_constant_l244_244904

noncomputable def ellipse_equation : Type := 
  {a b : ℝ // a > b ∧ b > 0 ∧ (b^2 + a^2 = 3*a^2)}

theorem problem_ellipse (a b : ℝ) (h₀ : a > b ∧ b > 0 ∧ (b^2 + a^2 = 3 * a^2)) :
  (∃ (T : ℝ × ℝ), (T.1 = 2) ∧ (T.2 = 1) ∧ (T.1^2 / 6 + T.2^2 / 3 = 1)) :=
sorry

theorem problem_constant (a b : ℝ) (T : ℝ × ℝ) (h₀ : a > b ∧ b > 0 ∧ (b^2 + a^2 = 3 * a^2)) (h₁ : (T.1 = 2) ∧ (T.2 = 1) ∧ (T.1^2 / 6 + T.2^2 / 3 = 1)):
  ∃ (λ : ℝ), λ = 4 / 5 ∧ (|PT|^2 = λ * |PA| * |PB|) :=
sorry

end problem_ellipse_problem_constant_l244_244904


namespace Rockets_won_38_games_l244_244454

-- Definitions for each team and their respective wins
variables (Sharks Dolphins Rockets Wolves Comets : ℕ)
variables (wins : Finset ℕ)
variables (shArks_won_more_than_Dolphins : Sharks > Dolphins)
variables (rockets_won_more_than_Wolves : Rockets > Wolves)
variables (rockets_won_fewer_than_Comets : Rockets < Comets)
variables (Wolves_won_more_than_25_games : Wolves > 25)
variables (possible_wins : wins = {28, 33, 38, 43})

-- Statement that the Rockets won 38 games given the conditions
theorem Rockets_won_38_games
  (shArks_won_more_than_Dolphins : Sharks > Dolphins)
  (rockets_won_more_than_Wolves : Rockets > Wolves)
  (rockets_won_fewer_than_Comets : Rockets < Comets)
  (Wolves_won_more_than_25_games : Wolves > 25)
  (possible_wins : wins = {28, 33, 38, 43}) :
  Rockets = 38 :=
sorry

end Rockets_won_38_games_l244_244454


namespace problem_l244_244500

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 3)

theorem problem (x : ℝ) :
  (∀ ε > 0, ∃ N : ℝ, ∀ m n : ℝ, m > N → n > N → Real.dist (f (m + ε)) (f m) < ε ∧
   Real.dist (f (n + ε)) (f n) < ε) ∧
  (Real.continuity_point f (-Real.pi / 6)) :=
begin
  split,
  { sorry }, -- Here's where the proof for periodicity would go
  { sorry }  -- Here's where the proof for the center of symmetry would go
end

#check @Real.sin_periodic
#check @Real.is_symm

end problem_l244_244500


namespace det_B_proof_l244_244628

noncomputable def det_B_condition : Prop :=
  ∃ (b c : ℝ), det ![![b, 3], ![-1, c]] = 5 ∧ 
                (col!(![b, 3], ![-1, c]) + 3 • (inv (col!(![b, 3], ![-1, c]))) = 0)

theorem det_B_proof : det_B_condition :=
sorry

end det_B_proof_l244_244628


namespace tangent_line_g_l244_244937

variable {α : Type*} [NormedField α] [CompleteSpace α]

noncomputable def f (x : α) : α
noncomputable def g (x : α) : α := x^2 + f(x)

theorem tangent_line_g 
  (tangent_f : ∀ x, x = (2:α) → f(x) = 2 * x - 1)
  (f_diff : ∀ (x : α), HasDerivAt f (2:α) 2) 
  : ∀ x, x = (2:α) → (let y := g(2) in g'(2) = 6) → (∀ y, y = (2:α) → has_tangent_at_point : has_tangent_at_point : has_tangent_at_point at (2, g(2)) = 6x - y - 5 = 0) := 
sorry

end tangent_line_g_l244_244937


namespace subset_size_l244_244882

noncomputable section

open Set

variables {A : Type} [LinearOrder A] {n : ℕ} (S : Fin (2^n) → Set A)

def property (S : Fin (2^n) → Set A) :=
  ∀ a b : Fin (2^n), (a < b) → ∀ x y z : A, (x < y ∧ y < z) → 
  (y ∈ S a ∧ z ∈ S a ∧ x ∈ S b ∧ z ∈ S b) → False

theorem subset_size (n : ℕ) (h : n ≥ 2) (S : Fin (2^n) → Set (Fin (2^(n+1))))
  (h_property : property S) :
  ∃ i : Fin (2^n), S i ⊆ (Fin (4 * n)) :=
sorry

end subset_size_l244_244882


namespace find_value_of_a_l244_244025

theorem find_value_of_a :
  (tan (600 * real.pi / 180) = real.sqrt 3) →
  (tan (600 * real.pi / 180) = a / -4) →
  a = -4 * real.sqrt 3 :=
sorry

end find_value_of_a_l244_244025


namespace vector_D_collinear_with_a_l244_244751

def is_collinear (a b : ℝ × ℝ × ℝ) : Prop :=
∃ k : ℝ, b = (k * a.1, k * a.2, k * a.3)

def vector_a : ℝ × ℝ × ℝ := (3, 0, -4)

def vector_D : ℝ × ℝ × ℝ := (-3/5, 0, 4/5)

theorem vector_D_collinear_with_a : is_collinear vector_a vector_D :=
sorry

end vector_D_collinear_with_a_l244_244751


namespace find_g_of_3_l244_244303

noncomputable def g (x : ℝ) : ℝ := sorry  -- Placeholder for the function g

theorem find_g_of_3 (h : ∀ x : ℝ, x ≠ 0 → 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 3 = 26 / 7 :=
by sorry

end find_g_of_3_l244_244303


namespace length_PS_l244_244090

-- Define the points P, Q, R, S in a 2D plane
variables {P Q R S T : Type*}
variables [has_norm P Q R S T]
variables (PQ QR RS : ℝ)

-- Define conditions
def PQ_eq_6 : PQ = 6 := sorry
def QR_eq_10 : QR = 10 := sorry
def RS_eq_25 : RS = 25 := sorry
def right_angle_Q : ∠ Q = 90 := sorry
def right_angle_R : ∠ R = 90 := sorry

-- Define the problem
theorem length_PS :
  PQ = 6 → QR = 10 → RS = 25 → ∠ Q = 90 → ∠ R = 90 → ∥ (P - S) ∥ = Real.sqrt 461 :=
by
sor
y

end length_PS_l244_244090


namespace student_ticket_count_l244_244455

theorem student_ticket_count (S N : ℕ) (h1 : S + N = 821) (h2 : 2 * S + 3 * N = 1933) : S = 530 :=
sorry

end student_ticket_count_l244_244455


namespace a_999_is_999_l244_244254

noncomputable def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else
    let x := (sequence (n - 1)) in
    ⌊(n^3 : ℚ) / x⌋.toNat

theorem a_999_is_999 : sequence 999 = 999 := 
  sorry

end a_999_is_999_l244_244254


namespace prove_p_equals_2_l244_244200

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244200


namespace contrapositive_proposition_l244_244700

theorem contrapositive_proposition
  (a b c d : ℝ) 
  (h : a + c ≠ b + d) : a ≠ b ∨ c ≠ d :=
sorry

end contrapositive_proposition_l244_244700


namespace behavior_of_f_in_interval_l244_244464

-- Define the function f(x)
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 3 * m * x + 3

-- Define the property of even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- The theorem statement
theorem behavior_of_f_in_interval (m : ℝ) (hf_even : is_even_function (f m)) :
  m = 0 → (∀ x : ℝ, -4 < x ∧ x < 0 → f 0 x < f 0 (-x)) ∧ (∀ x : ℝ, 0 < x ∧ x < 2 → f 0 (-x) > f 0 x) :=
by 
  sorry

end behavior_of_f_in_interval_l244_244464


namespace CP_PE_solution_l244_244973

variables {A B C D E P : Type} [nonempty A] [nonempty B] [nonempty C] [nonempty D] [nonempty E] [nonempty P]

-- Define the conditions as Lean definitions
def CD_DB_ratio (CD DB : ℝ) : Prop := CD / DB = 2
def AE_EB_ratio (AE EB : ℝ) : Prop := AE / EB = 1
def CP_PE_ratio (CP PE : ℝ) : ℝ := CP / PE

-- Rewrite the proof problem statement
theorem CP_PE_solution (CD DB AE EB CP PE : ℝ) (h1 : CD_DB_ratio CD DB) (h2 : AE_EB_ratio AE EB) : CP_PE_ratio CP PE = 6 :=
by
  sorry

end CP_PE_solution_l244_244973


namespace complementary_supplementary_angle_l244_244028

theorem complementary_supplementary_angle (x : ℝ) :
  (90 - x) * 3 = 180 - x → x = 45 :=
by 
  intro h
  sorry

end complementary_supplementary_angle_l244_244028


namespace problem_l244_244130

-- Definition and conditions of the problem
def origin := (0, 0 : ℝ)
def parabola (p : ℝ) : set (ℝ × ℝ) := { p | p.snd ^ 2 = 2 * p.fst * p }
def line := { p : ℝ × ℝ | p.snd = -sqrt 3 * (p.fst - 1) }
def focus (p : ℝ) := (p / 2, 0)
def directrix (p : ℝ) : ℝ := -p / 2

-- Problem statement with correct answers
theorem problem (p : ℝ) (M N : ℝ × ℝ)
  (hp : p > 0)
  (hline_focus : focus p ∈ line)
  (hM : M ∈ line ∩ parabola p)
  (hN : N ∈ line ∩ parabola p) :
  (p = 2) ∧ (let mid := ((M.fst + N.fst) / 2, (M.snd + N.snd) / 2)
             in abs (mid.fst - directrix p) = (dist M N) / 2) :=
by sorry

end problem_l244_244130


namespace alice_probability_l244_244820

noncomputable def probability_picking_exactly_three_green_marbles : ℚ :=
  let binom : ℚ := 35 -- binomial coefficient (7 choose 3)
  let prob_green : ℚ := 8 / 15 -- probability of picking a green marble
  let prob_purple : ℚ := 7 / 15 -- probability of picking a purple marble
  binom * (prob_green ^ 3) * (prob_purple ^ 4)

theorem alice_probability :
  probability_picking_exactly_three_green_marbles = 34454336 / 136687500 := by
  sorry

end alice_probability_l244_244820


namespace unique_solution_exists_l244_244354

theorem unique_solution_exists :
  ∃ (a b c d e : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  a + b = 1/7 * (c + d + e) ∧
  a + c = 1/5 * (b + d + e) ∧
  (a, b, c, d, e) = (1, 2, 3, 9, 9) :=
by {
  sorry
}

end unique_solution_exists_l244_244354


namespace find_f_neg1_l244_244617

-- Definitions based on conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

variable {b : ℝ} (f : ℝ → ℝ)

axiom odd_f : odd_function f
axiom f_form : ∀ x, 0 ≤ x → f x = 2^(x + 1) + 2 * x + b
axiom b_value : b = -2

theorem find_f_neg1 : f (-1) = -4 :=
sorry

end find_f_neg1_l244_244617


namespace p_eq_two_circle_tangent_proof_l244_244217

def origin := (0, 0)

def parabola (p : ℝ) := {xy : ℝ×ℝ // xy.2^2 = 2 * p * xy.1}

def focus (p : ℝ) : (ℝ × ℝ) := (p / 2, 0)

def line_through_focus (p : ℝ) : Prop := (focus p).2 = -sqrt 3 * ((focus p).1 - 1)

def directrix (p : ℝ) : {x : ℝ // x = - p / 2}

def intersects (p : ℝ) :=
  {P : ℝ×ℝ // ∃ M N : ℝ×ℝ, M ∈ parabola p ∧ N ∈ parabola p ∧
    M.2 = -√3 * (M.1 - 1) ∧ N.2 = -√3 * (N.1 - 1)}

theorem p_eq_two : ∃ (p : ℝ), line_through_focus p → p = 2 := sorry

def circle_tangent := ∀ (p : ℝ),
  ∀ (MN_mid : ℝ × ℝ),
    MN_mid.1 = (5/3 : ℝ) →
    MN_mid.2 = 0 →
    (4 / sqrt 3) = distance (MN_mid, (directrix p))

theorem circle_tangent_proof : circle_tangent := sorry

end p_eq_two_circle_tangent_proof_l244_244217


namespace probability_two_common_subjects_l244_244101

-- Definitions of the conditions
def subjects_1 := ℕ  -- 2 subjects: Physics and History
def subjects_2 := ℕ  -- 4 subjects: Ideological and Political Education, Geography, Chemistry, and Biology

def total_scenarios := (2 * choose 4 2) ^ 2  -- (C_2^1 * C_4^2)^2
def case_1_scenarios := 2 * 4 * (3 * 2)  -- C_2^1 * C_4^1 * A_3^2
def case_2_scenarios := 6 * (2 * 1)  -- C_4^2 * A_2^2
def favorable_scenarios := case_1_scenarios + case_2_scenarios

theorem probability_two_common_subjects : 
  (favorable_scenarios / total_scenarios : ℚ) = 5 / 12 := 
by
  sorry  -- proof to be filled in later

end probability_two_common_subjects_l244_244101


namespace max_peripheral_cities_l244_244770

-- Defining the conditions
def num_cities := 100
def max_transfers := 11
def unique_paths := true
def is_peripheral (A B : ℕ) (f : ℕ → ℕ → bool) : Prop := (f A B = false ∧ f B A = false ∧ 
                                                            A ≠ B ∧ ∀ k < max_transfers, f A B = false)

-- Mathematical proof problem statement
theorem max_peripheral_cities (f : ℕ → ℕ → bool) 
  (H1 : ∀ A B, A ≠ B → ∃ p, length p ≤ max_transfers ∧ unique_paths ∧ f A B = true) :
  ∃ x, num_cities - x = 89 ∧ (∀ A B, is_peripheral A B f → x < num_cities) :=
sorry

end max_peripheral_cities_l244_244770


namespace point_A_coordinates_parametric_equations_max_dot_product_l244_244982

def polar_ray_θ_eq_pi_over_6 (θ : ℝ) : Prop :=
  θ = Real.pi / 6

def circle_ρ_eq_2 (ρ : ℝ) : Prop :=
  ρ = 2

def ellipse_ρ2_eq_3_over_1_plus_2sin2θ (ρ θ : ℝ) : Prop :=
  ρ^2 = 3 / (1 + 2 * Real.sin θ^2)

noncomputable def polar_to_cartesian (ρ θ : ℝ) : (ℝ × ℝ) :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

noncomputable def point_A_cartesian_coords : ℝ × ℝ :=
  polar_to_cartesian 2 (Real.pi / 6)

theorem point_A_coordinates :
  point_A_cartesian_coords = (Real.sqrt 3, 1) := 
sorry

def ellipse_parametric (θ : ℝ) : ℝ × ℝ :=
  (Real.sqrt 3 * Real.cos θ, Real.sin θ)

theorem parametric_equations :
  ∀ {θ : ℝ}, (ellipse_parametric θ).fst^2 / 3 + (ellipse_parametric θ).snd^2 = 1 :=
sorry

noncomputable def coord_E : ℝ × ℝ :=
  (0, -1)

noncomputable def vector_AE : ℝ × ℝ :=
  ((Real.sqrt 3) - (Real.sqrt 3), 1 - (-1))

noncomputable def vector_AF (θ : ℝ) : ℝ × ℝ :=
  ((Real.sqrt 3 * Real.cos θ) - (Real.sqrt 3), (Real.sin θ) - 1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem max_dot_product :
  ∃ θ : ℝ, ∀ (E F: ℝ × ℝ), dot_product vector_AE (vector_AF θ) = 5 + Real.sqrt 13 :=
sorry

end point_A_coordinates_parametric_equations_max_dot_product_l244_244982


namespace incorrect_statements_in_triangle_l244_244488

theorem incorrect_statements_in_triangle :
  ∀ (A B C : ℝ) (a b c S : ℝ),
    (sin (2 * A) + sin (A - B + C) = sin (C - A - B) + 1 / 2) →
    (1 ≤ S ∧ S ≤ 2) →
    (a = 2 * R * sin A ∧ b = 2 * R * sin B ∧ c = 2 * R * sin C) →
    (abc = ab * c ∧ R^2 = 4 * S) →
    (¬(bc * (b + c) > 8) ∧ ab * (a + b) ≤ 16 * sqrt 2 ∧ (6 ≤ abc ∧ abc ≤ 12) ∧ ¬(12 ≤ abc ∧ abc ≤ 24)) :=
by sorry

end incorrect_statements_in_triangle_l244_244488


namespace collinear_unit_vector_l244_244748

def vector3 := ℝ × ℝ × ℝ

def is_unit_vector (v : vector3) : Prop :=
  let (x, y, z) := v in x^2 + y^2 + z^2 = 1

def are_collinear (v₁ v₂ : vector3) : Prop :=
  ∃ k : ℝ, v₂ = (k * v₁.1, k * v₁.2, k * v₁.3)

def vec_a : vector3 := (3, 0, -4)

def magnitude (v : vector3) : ℝ :=
  Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem collinear_unit_vector :
  magnitude vec_a = 5 →
  is_unit_vector (-3/5, 0, 4/5) →
  are_collinear vec_a (-3/5, 0, 4/5) :=
 by
  sorry

end collinear_unit_vector_l244_244748


namespace a_3_equals_35_l244_244031

noncomputable def S (n : ℕ) : ℕ := 5 * n ^ 2 + 10 * n
noncomputable def a (n : ℕ) : ℕ := S n - S (n - 1)

theorem a_3_equals_35 : a 3 = 35 := by
  sorry

end a_3_equals_35_l244_244031


namespace angle_AD_plane_BB1C1C_is_60_degrees_l244_244997

-- Define the structure of the triangular prism
structure TriangularPrism (V : Type*) :=
  (A B C A1 B1 C1 : V)
  (equal_edge_length : ∀ e₁ e₂ ∈ [{A, A1}, {B, B1}, {C, C1}, {A, B}, {B, C}, {C, A}], length e₁ = length e₂)
  (perpendicular_lateral_edges : ∀ (e₁ e₂ : Pair V), e₁ ∈ [{A, A1}, {B, B1}, {C, C1}] → e₂ ∈ [{A, B}, {B, C}, {C, A}] → is_perpendicular e₁ e₂)
  (D : V)
  (midpoint_D : is_midpoint D [{B, B1, C1, C}])

open Real

-- Required to formalize our angle in Lean
def angle_AD_plane_BB1C1C {V : Type*} [inner_product_space ℝ V] (prism : TriangularPrism V) : ℝ :=
  let AD := line prism.A prism.D
  let plane_BB1C1C := plane prisma.B prisma.B1 prisma.C1 prisma.C
  angle_between AD plane_BB1C1C

-- Proof statement
theorem angle_AD_plane_BB1C1C_is_60_degrees {V : Type*} [inner_product_space ℝ V]
  (prism : TriangularPrism V)
  (assumption1 : ∀ e₁ e₂ ∈ [{prism.A, prism.A1}, {prism.B, prism.B1}, {prism.C, prism.C1}, {prism.A, prism.B}, {prism.B, prism.C}, {prism.C, prism.A}], length e₁ = length e₂)
  (assumption2 : ∀ (e₁ e₂ : Pair V), e₁ ∈ [{prism.A, prism.A1}, {prism.B, prism.B1}, {prism.C, prism.C1}] → e₂ ∈ [{prism.A, prism.B}, {prism.B, prism.C}, {prism.C, prism.A}] → is_perpendicular e₁ e₂)
  (assumption3 : is_midpoint prism.D [{prism.B, prism.B1, prism.C1, prism.C}])
  : angle_AD_plane_BB1C1C prism = 60 :=
by
  sorry

end angle_AD_plane_BB1C1C_is_60_degrees_l244_244997


namespace quad_polynomial_inequality_l244_244621

theorem quad_polynomial_inequality (a b c x y : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
  let P (t : ℝ) := a * t ^ 2 + b * t + c in
  (P (x * y)) ^ 2 ≤ P (x ^ 2) * P (y ^ 2) :=
by
  sorry

end quad_polynomial_inequality_l244_244621


namespace integral_binom_equal_l244_244867

theorem integral_binom_equal (n k : ℕ) (hk : k ≤ n) :
  ∫ x in -1..1, (n.choose k) * (1 + x)^(n - k) * (1 - x)^k = (2:ℝ)^(n+1) / (n + 1) := by
  sorry

end integral_binom_equal_l244_244867


namespace fraction_is_percent_of_y_l244_244972

theorem fraction_is_percent_of_y (y : ℝ) (hy : y > 0) : 
  (2 * y / 5 + 3 * y / 10) / y = 0.7 :=
sorry

end fraction_is_percent_of_y_l244_244972


namespace inequality_solution_l244_244667

noncomputable def solve_inequality : set ℝ := {x : ℝ | (2 * x + 3) / (x + 5) > (5 * x + 7) / (3 * x + 14)}

theorem inequality_solution : 
  solve_inequality = {x : ℝ | (-103.86 < x ∧ x < -14 / 3) ∨ (-5 < x ∧ x < -0.14)} :=
by
  sorry

end inequality_solution_l244_244667


namespace solutions_proof_l244_244017

-- Define propositions p and q
variable (p q: Prop)
-- Define the conditions in Lean
def condition_p := ∀ (prism: Prop), (∃ (rhombus_base: Prop), prism = rhombus_base → ¬p)
def condition_q := ∀ (pyramid: Prop), (∃ (equilateral_triangle_base: Prop), pyramid = equilateral_triangle_base → ¬q)

-- Define the correct conclusions
def correct_conclusions := (¬p ∧ q=false) ∧ (p ∧ q) = false ∧ (p ∨ q) = false ∧ (¬p ∧ ¬q)

theorem solutions_proof (p q: Prop) (h1: condition_p) (h2: condition_q) :
  (¬p ∧ ¬q) := 
  sorry

end solutions_proof_l244_244017


namespace boy_current_age_l244_244741

theorem boy_current_age (x : ℕ) (h : 5 ≤ x) (age_statement : x = 2 * (x - 5)) : x = 10 :=
by
  sorry

end boy_current_age_l244_244741


namespace expected_value_of_winnings_l244_244372

noncomputable def winnings (n : ℕ) : ℕ := 2 * n - 1

theorem expected_value_of_winnings : 
  (1 / 6 : ℚ) * ((winnings 1) + (winnings 2) + (winnings 3) + (winnings 4) + (winnings 5) + (winnings 6)) = 6 :=
by
  sorry

end expected_value_of_winnings_l244_244372


namespace find_angle_x_l244_244992

theorem find_angle_x (angle_ABC angle_BAC angle_BCA angle_DCE angle_CED x : ℝ)
  (h1 : angle_ABC + angle_BAC + angle_BCA = 180)
  (h2 : angle_ABC = 70) 
  (h3 : angle_BAC = 50)
  (h4 : angle_DCE + angle_CED = 90)
  (h5 : angle_DCE = angle_BCA) :
  x = 30 :=
by
  sorry

end find_angle_x_l244_244992


namespace ellipse_equation_sum_l244_244826

theorem ellipse_equation_sum :
  ∃ (A B C D E F : ℤ),
    (∀ t : ℝ, 
      let x := (3 * (Real.sin t - 2)) / (3 - Real.cos t)
      let y := (4 * (Real.cos t - 4)) / (3 - Real.cos t)
      in A * x^2 + B * x * y + C * y^2 + D * x + E * y + F = 0) ∧ 
    Int.gcd (Int.gcd A B) (Int.gcd (Int.gcd C D) (Int.gcd E F)) = 1 ∧
    |A| + |B| + |C| + |D| + |E| + |F| = 496 :=
begin
  sorry
end

end ellipse_equation_sum_l244_244826


namespace remainder_of_sum_l244_244735

open Nat

theorem remainder_of_sum :
  (12345 + 12347 + 12349 + 12351 + 12353 + 12355 + 12357) % 16 = 9 :=
by 
  sorry

end remainder_of_sum_l244_244735


namespace remainder_geometric_series_sum_l244_244856

/-- Define the sum of the geometric series. --/
def geometric_series_sum (n : ℕ) : ℕ :=
  (13^(n+1) - 1) / 12

/-- The given geometric series. --/
def series_sum := geometric_series_sum 1004

/-- Define the modulo operation. --/
def mod_op (a b : ℕ) := a % b

/-- The main statement to prove. --/
theorem remainder_geometric_series_sum :
  mod_op series_sum 1000 = 1 :=
sorry

end remainder_geometric_series_sum_l244_244856


namespace determine_function_l244_244938

def f (x : ℝ) : ℝ := sorry

theorem determine_function (x : ℝ) (h : x ≠ 0) : (f(x) = 2 * f(1/x) + 3 * x) → (f(x) = -x - 2/x) :=
by 
sorry

end determine_function_l244_244938


namespace count_two_digit_primes_l244_244531

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def valid_digits : set ℕ := {3, 5, 7, 9}

def two_digit_primes := { n | ∃ (a b : ℕ), a ∈ valid_digits ∧ b ∈ valid_digits ∧ a ≠ b ∧ n = 10 * a + b ∧ is_prime n }

theorem count_two_digit_primes : (two_digit_primes : set ℕ).card = 7 := 
  sorry

end count_two_digit_primes_l244_244531


namespace prove_p_equals_2_l244_244201

-- Given conditions from the problem
variables {p : ℝ} {x y : ℝ}
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2 * p * x ∧ p > 0

def line (x y : ℝ) : Prop := y = -sqrt 3 * (x - 1)

-- Prove p = 2 given the provided condition about the line passing through the focus
theorem prove_p_equals_2 (h : ∃ (x_focus y_focus : ℝ), parabola p x_focus y_focus ∧ line x_focus y_focus) : p = 2 :=
by
  sorry

end prove_p_equals_2_l244_244201


namespace angles_equal_l244_244953

theorem angles_equal (A B C : ℝ) (h1 : A + B = 180) (h2 : B + C = 180) : A = C := sorry

end angles_equal_l244_244953


namespace median_perimeter_ratio_l244_244024

variables {A B C : Type*}
variables (AB BC AC AD BE CF : ℝ)
variable (l m : ℝ)

noncomputable def triangle_perimeter (AB BC AC : ℝ) : ℝ := AB + BC + AC
noncomputable def triangle_median_sum (AD BE CF : ℝ) : ℝ := AD + BE + CF

theorem median_perimeter_ratio (h1 : l = triangle_perimeter AB BC AC)
                                (h2 : m = triangle_median_sum AD BE CF) :
  m / l > 3 / 4 :=
by
  sorry

end median_perimeter_ratio_l244_244024


namespace num_integer_solutions_eq_4_l244_244056

theorem num_integer_solutions_eq_4 :
  {x : ℤ | (x^2 - 2 * x - 3)^(x + 3) = 1}.to_finset.card = 4 :=
sorry

end num_integer_solutions_eq_4_l244_244056


namespace parabola_conditions_l244_244245

-- Define the conditions of the problem
def origin : Point := (0, 0)

-- Define the parabola and line
def parabola (p : ℝ) := { y : ℝ // ∃ x : ℝ, y^2 = 2 * p * x }

def line := { y : ℝ // ∃ x : ℝ, y = -√3 * (x - 1) }

-- Define focus of the parabola
def focus (p : ℝ) : Point := (p / 2, 0)

-- Define directrix of the parabola
def directrix (p : ℝ) : set Point := { p : Point | p.1 = -p / 2 }

-- Check that the line passes through the focus
def passes_through_focus (p : ℝ) : Prop :=
  line.2 (focus p).2

-- Predicate for checking if the circle with MN as diameter is tangent to the directrix
def is_tangent_to_directrix (M N : Point) (l : set Point) : Prop :=
  let midpoint : Point := ((M.1 + N.1) / 2, (M.2 + N.2) / 2)
  in ∃ p ∈ l, distance midpoint p = distance M N / 2

-- The main theorem statement
theorem parabola_conditions (p : ℝ) (M N : Point) :
  (passes_through_focus p) → 
  (p = 2) ∧ 
  (is_tangent_to_directrix M N (directrix p)) :=
begin
  -- proof goes here
  sorry
end

end parabola_conditions_l244_244245


namespace find_a5_l244_244506

noncomputable theory

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = n * a n + 1

theorem find_a5 (a : ℕ → ℕ) (h : sequence a) : a 5 = 65 :=
by
  sorry

end find_a5_l244_244506


namespace customer_payment_probability_l244_244267

theorem customer_payment_probability :
  let total_customers := 100
  let age_40_50_non_mobile := 13
  let age_50_60_non_mobile := 27
  let total_40_60_non_mobile := age_40_50_non_mobile + age_50_60_non_mobile
  let probability := (total_40_60_non_mobile : ℚ) / total_customers
  probability = 2 / 5 := by
sorry

end customer_payment_probability_l244_244267


namespace proof_problem_l244_244154

-- Define the parabola and line intersecting conditions
def parabola_y_square_equals_2px (p : ℝ) : Prop :=
∀ x y : ℝ, y^2 = 2 * p * x

def line_passing_through_focus (p : ℝ) : Prop :=
let focus := (p / 2, 0) in
∀ x y : ℝ, y = -√3 * (x - 1) → (x, y) = focus

-- Define the properties to be proven
def p_equals_two (p : ℝ) : Prop := p = 2

def circle_with_diameter_MN_is_tangent_to_directrix (p : ℝ) : Prop :=
let directrix := -p / 2 in
∀ a b : ℝ, sqrt((a - b) ^ 2 + ((- √3 * (a - 1)) - (- √3 * (b - 1))) ^ 2) / 2 = abs(p / 2 + (a + b) / 2)

def triangle_OMN_not_isosceles (p : ℝ) : Prop :=
∀ a b : ℝ, 
let O := (0, 0)
    M := (a, -√3 * (a - 1))
    N := (b, -√3 * (b - 1)) in
sqrt(O.1^2 + O.2^2) ≠ sqrt(M.1^2 + M.2^2) ∧ sqrt(O.1^2 + O.2^2) ≠ sqrt(N.1^2 + N.2^2)

-- The main theorem to be proven
theorem proof_problem (p : ℝ) :
  parabola_y_square_equals_2px p →
  line_passing_through_focus p →
  p_equals_two p ∧
  circle_with_diameter_MN_is_tangent_to_directrix p ∧
  triangle_OMN_not_isosceles p :=
by sorry

end proof_problem_l244_244154


namespace equal_PA_PD_l244_244620

-- Definitions for the problem conditions
variable (A B C D E P : Type) [MetricSpace A] [MetricSpace B]
  [MetricSpace C] [MetricSpace D] [MetricSpace E]
  [MetricSpace P]
variable (ABCDE : list (MetricSpace A)) 
variable [Convex (List A)]

-- Conditions
noncomputable def pentagon_properties : Prop :=
  (ABCDE.length = 5) ∧
  (∀ (i j :ℕ), i ≠ j → (dist ABCDE.nth i ABCDE.nth j = dist ABCDE.nth j ABCDE.nth i)) ∧
  (is_right_angle (angle C B D)) ∧
  (is_right_angle (angle D E C)) ∧ 
  (P ∈ intersection (line_through A C) (line_through B D))

-- Proving the equality of PA and PD
theorem equal_PA_PD : pentagon_properties → dist P A = dist P D := 
by
  sorry -- Proof to be filled in later

end equal_PA_PD_l244_244620


namespace find_value_of_N_l244_244877

theorem find_value_of_N (N : ℝ) : 
  2 * ((3.6 * N * 2.50) / (0.12 * 0.09 * 0.5)) = 1600.0000000000002 → 
  N = 0.4800000000000001 :=
by
  sorry

end find_value_of_N_l244_244877


namespace parabola_focus_line_tangent_circle_l244_244212

-- Defining the problem conditions and required proof.
theorem parabola_focus_line_tangent_circle
  (O : Point)
  (focus : Point)
  (M N : Point)
  (line : ∀ x, Real)
  (parabola : ∀ x, Real)
  (directrix : Real)
  (p : Real)
  (hp_gt_0 : p > 0)
  (parabola_eq : ∀ x, parabola x = (√(2 * p * x)))
  (line_eq : ∀ x, line x = -√3 * (x - 1))
  (focus_eq : focus = (p/2, 0))
  (line_through_focus : ∀ y, line y = focus.2) 
  : p = 2 ∧ tangent ((M, N) : LineSegment) directrix := by
  sorry

end parabola_focus_line_tangent_circle_l244_244212


namespace loss_per_metre_l244_244384

def total_metres : ℕ := 500
def selling_price : ℕ := 18000
def cost_price_per_metre : ℕ := 41

theorem loss_per_metre :
  (cost_price_per_metre * total_metres - selling_price) / total_metres = 5 :=
by sorry

end loss_per_metre_l244_244384


namespace three_zeros_range_l244_244039

def piecewise_function (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then exp x - a * x else -x^2 - (a + 2) * x + 1

theorem three_zeros_range (a : ℝ) : 
  (∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    piecewise_function a x₁ = 0 ∧ 
    piecewise_function a x₂ = 0 ∧ 
    piecewise_function a x₃ = 0) ↔ a ∈ Set.Ioi (Real.exp 1) :=
sorry

end three_zeros_range_l244_244039


namespace curve_C1_cartesian_eq_and_intersection_distance_l244_244988

theorem curve_C1_cartesian_eq_and_intersection_distance :
  (∀ θ ρ, (ρ = cos θ - sin θ) → (∃ x y, x^2 + y^2 - x + y = 0)) ∧
  (∀ t, let x := 1/2 - (√2)/2 * t
        let y := (√2)/2 * t in 
    (x^2 + y^2 = x - y) → |√((1/2 - (√2)/2 * (t1)) - (1/2 - (√2)/2 * (t2))^2 + ((√2)/2 * (t1)) - ((√2)/2 * (t2))^2)| = (√6)/2) :=
  by sorry

end curve_C1_cartesian_eq_and_intersection_distance_l244_244988


namespace charlie_fewer_games_than_dana_l244_244276

theorem charlie_fewer_games_than_dana
  (P D C Ph : ℕ)
  (h1 : P = D + 5)
  (h2 : C < D)
  (h3 : Ph = C + 3)
  (h4 : Ph = 12)
  (h5 : P = Ph + 4) :
  D - C = 2 :=
by
  sorry

end charlie_fewer_games_than_dana_l244_244276


namespace max_distance_sum_l244_244258

open Real

-- Define the curve C
def C (x y : ℝ) : Prop := (sqrt (x^2 / 25) + sqrt (y^2 / 9) = 1)

-- Define the points F1 and F2
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)

-- Define the distance function |PF_1| + |PF_2|
def distance_sum (P F1 F2 : ℝ × ℝ) : ℝ :=
  let (px, py) := P
  let (x1, y1) := F1
  let (x2, y2) := F2
  Real.sqrt ((px - x1)^2 + (py - y1)^2) + Real.sqrt ((px - x2)^2 + (py - y2)^2)

-- Statement of the problem
theorem max_distance_sum (x y : ℝ) (h : C x y) : distance_sum (x, y) F1 F2 ≤ 10 :=
sorry

end max_distance_sum_l244_244258


namespace min_N_divisible_by_2010_squared_l244_244699

theorem min_N_divisible_by_2010_squared :
  ∃ (N : ℕ), (∀ k : ℕ, 1000 ≤ k ∧ k < k + N ∧ k + N ≤ 9999 →
    (∏ i in finset.range(N), k + i) % (2010 * 2010) = 0) ∧ N = 5 :=
by 
  -- sorry

end min_N_divisible_by_2010_squared_l244_244699


namespace exists_unique_root_of_continuous_monotonic_l244_244250

-- Given: f is continuous and monotonic on the interval [a, b]
-- and f(a)f(b) < 0
-- To Prove: There exists exactly one real root x in [a, b] such that f(x) = 0

theorem exists_unique_root_of_continuous_monotonic (a b : ℝ) 
  (f : ℝ → ℝ) (h_cont : ContinuousOn f (Set.Icc a b))
  (h_monotonic : ∀ x y ∈ Set.Icc a b, x ≤ y → f x ≤ f y)
  (h_sign_change : f a * f b < 0) :
  ∃! x ∈ Set.Icc a b, f x = 0 := 
sorry

end exists_unique_root_of_continuous_monotonic_l244_244250


namespace zero_8x5_table_possible_l244_244991

theorem zero_8x5_table_possible (table : Fin 8 → Fin 5 → ℕ) :
  ∃ moves : list (Fin 8 × Fin 5 → Fin 8 × Fin 5), ∀ (move_op : (Fin 8 × Fin 5 → Fin 8 × Fin 5) → (ℕ → ℕ) → Fin 8 → Fin 5 → ℕ),
    (∀ m ∈ moves, ∀ (i : Fin 8) (j : Fin 5), table i j = 0) :=
by
  sorry

end zero_8x5_table_possible_l244_244991


namespace range_of_m_l244_244908

-- Define the sets A and B
def setA := {x : ℝ | abs (x - 1) < 2}
def setB (m : ℝ) := {x : ℝ | x >= m}

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), (setA ∩ setB m = setA) → m <= -1 :=
by
  sorry

end range_of_m_l244_244908
