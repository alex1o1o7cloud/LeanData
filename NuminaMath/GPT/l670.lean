import Mathlib

namespace arithmetic_progression_value_of_x_l670_670500

theorem arithmetic_progression_value_of_x (x : ℝ):
  (x + 2) - (x - 2) = (3x + 4) - (x + 2) → x = 1 :=
by
  assume h : (x + 2) - (x - 2) = (3x + 4) - (x + 2)
  sorry

end arithmetic_progression_value_of_x_l670_670500


namespace exists_ai_aj_l670_670721

theorem exists_ai_aj (n k : ℕ) (h : k > (n + 1) / 2) :
  ∃ (a : Fin k → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ n) ∧ StrictMono a ∧ 
    ∃ i j, i < j ∧ a i + a 0 = a j := 
sorry

end exists_ai_aj_l670_670721


namespace number_of_starting_positions_l670_670071

/-- 
 Let 𝒞 be the hyperbola y² - 4x² = 4. Given a point P₀ on the x-axis, we construct a sequence of points {Pₙ} on the x-axis as follows: 
 let ℓₙ be the line with slope 1 passing through Pₙ, then Pₙ₊₁ is the orthogonal projection 
 of the point of intersection of ℓₙ and 𝒞 onto the x-axis. (If Pₙ = 0, then the sequence terminates.)
 Find the number of starting positions P₀ on the x-axis such that P₀ = P₂₀₂₃.
-/
theorem number_of_starting_positions:
  let 𝒞 := {p : ℝ × ℝ | p.2 ^ 2 - 4 * p.1 ^ 2 = 4}
  let P := ℕ → (ℝ × ℝ)
  let P_sequence (P : P) := ∀ n, 
    let ℓₙ := {q : ℝ × ℝ | q.2 = q.1 - P n.1}
    let intersection := (λ p : ℝ × ℝ, p ∈ 𝒞 ∧ p ∈ ℓₙ) (P n) 
    P (n+1) = (closest_point_on_x_axis intersection.1, 0)
  in ∃ P₀, (P_sequence P₀ ∧ P₀ = P 2023).card = 2 ^ 2023 - 2 := by
  sorry

end number_of_starting_positions_l670_670071


namespace arithmetic_sequence_ratio_l670_670316

open Nat

noncomputable def S (n : ℕ) : ℝ := n^2
noncomputable def T (n : ℕ) : ℝ := n * (2 * n + 3)

theorem arithmetic_sequence_ratio 
  (h : ∀ n : ℕ, (2 * n + 3) * S n = n * T n) : 
  (S 5 - S 4) / (T 6 - T 5) = 9 / 25 := by
  sorry

end arithmetic_sequence_ratio_l670_670316


namespace julia_min_correct_l670_670830

theorem julia_min_correct 
  (points_per_correct : ℤ)
  (points_per_incorrect : ℤ)
  (points_per_unanswered : ℤ)
  (problems_attempted : ℤ)
  (total_problems : ℤ)
  (min_required_points : ℤ) : 
  points_per_correct = 7 → 
  points_per_incorrect = -1 → 
  points_per_unanswered = 2 → 
  problems_attempted = 28 → 
  total_problems = 30 → 
  min_required_points = 150 →
  ∃ x : ℤ, x ≥ 22 ∧ 8 * x - 28 + 4 ≥ min_required_points := 
by 
  intros h1 h2 h3 h4 h5 h6
  use 22
  split
  { linarith }
  { sorry }

end julia_min_correct_l670_670830


namespace clock_angle_4_50_l670_670953

theorem clock_angle_4_50 : 
  let M := 4 * 30 + 25 in  -- Minute hand degree
  let H := 300 in         -- Hour hand degree
  let θ := abs (M - H)    -- Absolute difference
  (θ = 155) :=               -- Acute angle that gives the result
by {
  sorry
}

end clock_angle_4_50_l670_670953


namespace Ajay_total_time_l670_670639

def Ajay_walk_time (flat_speed_uphill_speed_factor downhill_speed_factor distance_flat distance_uphill distance_downhill : ℝ) : ℝ :=
  let speed_flat := 3
  let speed_uphill := speed_flat - (flat_speed_uphill_speed_factor * speed_flat)
  let speed_downhill := speed_flat + (downhill_speed_factor * speed_flat)
  let time_uphill := distance_uphill / speed_uphill
  let time_flat := distance_flat / speed_flat
  let time_downhill := distance_downhill / speed_downhill
  time_uphill + time_flat + time_downhill

theorem Ajay_total_time :
  Ajay_walk_time 0.2 0.1 25 15 20 = 20.64 := by
  sorry

end Ajay_total_time_l670_670639


namespace ratio_accepted_rejected_l670_670029

-- Definitions for the conditions given
def eggs_per_day : ℕ := 400
def ratio_accepted_to_rejected : ℕ × ℕ := (96, 4)
def additional_accepted_eggs : ℕ := 12

/-- The ratio of accepted eggs to rejected eggs on that particular day is 99:1. -/
theorem ratio_accepted_rejected (a r : ℕ) (h1 : ratio_accepted_to_rejected = (a, r)) 
  (h2 : (a + r) * (eggs_per_day / (a + r)) = eggs_per_day) 
  (h3 : additional_accepted_eggs = 12) :
  (a + additional_accepted_eggs) / r = 99 :=
  sorry

end ratio_accepted_rejected_l670_670029


namespace collinear_BNC_H_orthocenter_ABC_l670_670587

open EuclideanGeometry

variables {A B M C D E H N : Point}
variables {circ1 circ2 : Circle}

-- Conditions
def M_interior_AB := M ∈ Segment A B
def square_AMCD := square A M C D -- Definition for the square
def square_BEHM := square B E H M -- Definition for the square
def N_circumcircle_intersection := N ∈ circumcircle square_AMCD ∧ N ∈ circumcircle square_BEHM

-- Proof Problems
theorem collinear_BNC (hM_interior : M_interior_AB)
                      (h_square_AMCD : square_AMCD)
                      (h_square_BEHM : square_BEHM)
                      (h_N_intersection : N_circumcircle_intersection) :
        collinear B N C := 
sorry

theorem H_orthocenter_ABC (hM_interior : M_interior_AB)
                          (h_square_AMCD : square_AMCD)
                          (h_square_BEHM : square_BEHM)
                          (h_N_intersection : N_circumcircle_intersection) :
        is_orthocenter H A B C := 
sorry

end collinear_BNC_H_orthocenter_ABC_l670_670587


namespace intersection_area_l670_670220

open EuclideanGeometry

namespace CubeIntersection

def point := (ℝ × ℝ × ℝ)

def cube (edge_length : ℝ) : Prop := 
  edge_length = 30 

def PointCoordinates (A B C D P Q R : point) : Prop := 
  A = (0, 0, 0) ∧
  B = (30, 0, 0) ∧ 
  C = (30, 0, 30) ∧
  D = (30, 30, 30) ∧
  P = (10, 0, 0) ∧
  Q = (30, 0, 10) ∧
  R = (30, 20, 30)

def PlaneEquation (P Q R: point) (a b c d : ℝ) : Prop :=
  let ⟨Px, Py, Pz⟩ := P in
  let ⟨Qx, Qy, Qz⟩ := Q in
  let ⟨Rx, Ry, Rz⟩ := R in
  Px * a + Py * b + Pz * c = d ∧
  Qx * a + Qy * b + Qz * c = d ∧
  Rx * a + Ry * b + Rz * c = d

theorem intersection_area (A B C D P Q R : point) (a b c d : ℝ) 
  (h_cube : cube 30)
  (h_coords: PointCoordinates A B C D P Q R)
  (h_plane: PlaneEquation P Q R a b c d) :
  area_of_polygon_formed_by_plane_cube_intersection a b c d A B C D = 450 :=
sorry

end CubeIntersection

end intersection_area_l670_670220


namespace conditional_probability_l670_670615

noncomputable def fair_die_faces : ℕ := 6

-- Define the events A and B:
def event_A (x y : ℕ) : Prop := (x + y) % 2 = 0
def event_B (x y : ℕ) : Prop := (x ≠ y ∧ (x % 2 = 0 ∨ y % 2 = 0))

-- Probability space of two die rolls
def Ω := fin (fair_die_faces + 1) × fin (fair_die_faces + 1)

-- Indicator functions for the events
def indicator {α : Type} (P : α → Prop) [decidable_pred P] : α → ℕ
| x := if P x then 1 else 0

-- Define the probability measure
def probability (s : set Ω) : ℚ := (s.to_finset.card : ℚ) / ((Ω.to_finset.card : ℚ))

-- Calculate P(A)
def P_A : ℚ := probability {x | event_A x.1.val x.2.val}

-- Calculate P(A ∩ B)
def P_A_and_B : ℚ := probability {x | event_A x.1.val x.2.val ∧ event_B x.1.val x.2.val}

-- Calculate P(B|A)
def P_B_given_A : ℚ := P_A_and_B / P_A

-- The theorem we have to prove
theorem conditional_probability :
  P_B_given_A = 1 / 3 :=
sorry

end conditional_probability_l670_670615


namespace opposite_reciprocal_of_neg_five_l670_670149

theorem opposite_reciprocal_of_neg_five : 
  ∀ x : ℝ, x = -5 → - (1 / x) = 1 / 5 :=
by
  sorry

end opposite_reciprocal_of_neg_five_l670_670149


namespace conical_surfaces_of_revolution_l670_670664

theorem conical_surfaces_of_revolution {S : Point} 
  (A A' B B' C C' : Line) 
  (h1 : PassesThrough S A) 
  (h2 : PassesThrough S A')
  (h3 : PassesThrough S B)
  (h4 : PassesThrough S B')
  (h5 : PassesThrough S C)
  (h6 : PassesThrough S C')
  (h_not_coplanar : ¬ Coplanar {A, A', B, B', C, C'}) :
  ∃ (num_solutions : ℕ), num_solutions = 4 ∧ 
  ∀ (conical_surfaces : set (ConicalSurface {S A B C})),
  (∀ surface ∈ conical_surfaces, 
    (∀ line ∈ {A, A', B, B', C, C'}, GeneratrixOf surface line)) → 
    conical_surfaces = 4 := 
sorry

end conical_surfaces_of_revolution_l670_670664


namespace part1_part2_part3_l670_670049

-- Given conditions
def a : ℕ → ℤ
| 0 := 3
| (n + 1) := -a n - 2 * (n + 1) + 1

-- Proving (1) Actual values of a_2 and a_3
theorem part1 (h1 : a 1 = -6) (h2 : a 2 = 1) : 
  a 1 = -6 ∧ a 2 = 1 := sorry

-- Proving (2) That a_n + n is a geometric sequence and find a_n
theorem part2 (h1 : ∀ n, a (n + 1) + (n + 1) = (a n + n) * (-1)) (h2 : ∀ n, a n = 4 * (-1)^(n-1) - n) : 
  (∀ n, a (n + 1) + (n + 1) = (a n + n) * (-1)) ∧ (∀ n, a n = 4 * (-1)^(n-1) - n) := sorry

-- Proving (3) The sum of the first n terms S_n, where S_n = sum (a_k)
def S : ℕ → ℤ
| 0 := 0
| (n + 1) := S n + a (n + 1)

theorem part3 (n : ℕ) (sum_formula : S n = - (n^2 + n - 4) / 2 - 2 * (-1)^n) : 
  S n = - (n^2 + n - 4) / 2 - 2 * (-1)^n := sorry

end part1_part2_part3_l670_670049


namespace concentric_circle_area_ratio_l670_670303

def circle_area (r : ℝ) : ℝ := π * r^2

def ring_area (r_outer r_inner : ℝ) : ℝ := circle_area r_outer - circle_area r_inner

theorem concentric_circle_area_ratio :
  let radii := [1, 3, 5, 7, 9]
  let black_areas := [radii.head!].map circle_area ++ [ring_area radii[2] radii[1], ring_area radii[4] radii[3]]
  let white_areas := [ring_area radii[1] radii[0], ring_area radii[3] radii[2]]
  (black_areas.sum / white_areas.sum = (49 : ℚ) / 32) :=
by
  sorry

end concentric_circle_area_ratio_l670_670303


namespace solve_for_m_l670_670131

theorem solve_for_m :
  let x_values := [0, 1, 2, 3, 4]
  let y_values := [10, 15, m, 30, 35]
  let n := list.length x_values
  let x_mean := (list.sum x_values : ℝ) / n
  let y_mean := (list.sum y_values : ℝ) / n
  let regression_line (x : ℝ) := 6.5 * x + 9
  y_mean = regression_line x_mean -> m = 20 :=
by
  let x_values := [0, 1, 2, 3, 4]
  let y_values := [10, 15, m, 30, 35]
  let n := list.length x_values
  let x_mean := (list.sum x_values : ℝ) / n
  let y_mean := (list.sum y_values : ℝ) / n
  let regression_line (x : ℝ) := 6.5 * x + 9
  have hx_mean : x_mean = 2 := by
    simp [x_values, n]
  have : y_mean = regression_line x_mean := by
    sorry -- skip the actual proof
  
  have hy_mean : y_mean = (m + 90) / 5 := by
    simp [y_values, n]
  
  rw [hx_mean] at this
  have : (m + 90) / 5 = 22 := by
    simp [this, regression_line, x_mean]
  
  sorry

end solve_for_m_l670_670131


namespace abe_age_equation_l670_670520

theorem abe_age_equation (a : ℕ) (x : ℕ) (h1 : a = 19) (h2 : a + (a - x) = 31) : x = 7 :=
by
  sorry

end abe_age_equation_l670_670520


namespace algebraic_expression_value_l670_670114

noncomputable def algebraic_expression (x : ℝ) : ℝ := 
( (x / (x - 1)) - (1 / (x^2 - x)) ) / ( (x + 1)^2 / x )

def value_of_x : ℝ := 2 * real.sin (real.pi / 3) - real.tan (real.pi / 4)

theorem algebraic_expression_value :
  algebraic_expression value_of_x = real.sqrt 3 / 3 :=
by
  sorry

end algebraic_expression_value_l670_670114


namespace cannot_form_shape_B_l670_670198

-- Define the given pieces
def pieces : List (List (Nat × Nat)) :=
  [ [(1, 1)],
    [(1, 2)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 3)],
    [(1, 3)] ]

-- Define shape B requirement
def shapeB : List (Nat × Nat) := [(1, 6)]

theorem cannot_form_shape_B :
  ¬ (∃ (combinations : List (List (Nat × Nat))), combinations ⊆ pieces ∧ 
     (List.foldr (λ x acc => acc + x) 0 (combinations.map (List.foldr (λ y acc => acc + (y.1 * y.2)) 0)) = 6)) :=
sorry

end cannot_form_shape_B_l670_670198


namespace inclination_angle_range_l670_670878

def range_of_inclination_angle (α : ℝ) (θ : ℝ) : Prop :=
  let ℓ : ℝ := x + y * cos θ + 3
  θ ∈ ℝ →
  (∃ θ, θ ∈ Ω (ℓ x y cos θ) ↔ α ∈ [π/4, 3*π/4]) 
 
theorem inclination_angle_range :
  ∀ θ : ℝ, θ ∈ ℝ → range_of_inclination_angle α θ :=
begin
  sorry
end

end inclination_angle_range_l670_670878


namespace sin_D_in_right_triangle_l670_670829

theorem sin_D_in_right_triangle (D E F : ℝ)
  (h1 : ∠ E = 90) 
  (h2 : 4 * real.sin D = 5 * real.cos D) : 
  real.sin D = 5 * real.sqrt 41 / 41 :=
sorry

end sin_D_in_right_triangle_l670_670829


namespace initial_population_approx_l670_670151

noncomputable def initial_population := 
  let P := 500000.0 
  let t := 26.897352853986263 
  let T := 3.0 
  P / (2.0^(t / T))

theorem initial_population_approx : initial_population ≈ 1010.0 := 
by sorry

end initial_population_approx_l670_670151


namespace necessary_but_not_sufficient_condition_l670_670754

-- Definitions of function and conditions
def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x)
def ω_positive (ω : ℝ) : Prop := ω > 0

-- Definitions of the points
def passes_through (f : ℝ → ℝ) (point : ℝ × ℝ) : Prop :=
  f point.fst = point.snd

-- Lean statement for the proof problem
theorem necessary_but_not_sufficient_condition (ω : ℝ) (hω : ω_positive ω) :
  passes_through (f ω) (π / 2, 0) → ¬ (passes_through (f ω) (π / 4, 1)) :=
sorry

end necessary_but_not_sufficient_condition_l670_670754


namespace first_floor_cost_l670_670054

-- Definitions and assumptions
variables (F : ℝ)
variables (earnings_first_floor earnings_second_floor earnings_third_floor : ℝ)
variables (total_monthly_earnings : ℝ)

-- Conditions from the problem
def costs := F
def second_floor_costs := F + 20
def third_floor_costs := 2 * F
def first_floor_rooms := 3 * costs
def second_floor_rooms := 3 * second_floor_costs
def third_floor_rooms := 3 * third_floor_costs

-- Total monthly earnings
def total_earnings := first_floor_rooms + second_floor_rooms + third_floor_rooms

-- Equality condition
axiom total_earnings_is_correct : total_earnings = 165

-- Theorem to be proved
theorem first_floor_cost :
  (F = 8.75) :=
by
  have earnings_first_floor_eq := first_floor_rooms
  have earnings_second_floor_eq := second_floor_rooms
  have earnings_third_floor_eq := third_floor_rooms
  have total_earning_eq := total_earnings_is_correct
  sorry

end first_floor_cost_l670_670054


namespace cost_per_load_is_25_cents_l670_670713

def washes_per_bottle := 80
def price_per_bottle_on_sale := 20
def bottles := 2
def total_cost := bottles * price_per_bottle_on_sale -- 2 * 20 = 40
def total_loads := bottles * washes_per_bottle -- 2 * 80 = 160
def cost_per_load_in_dollars := total_cost / total_loads -- 40 / 160 = 0.25
def cost_per_load_in_cents := cost_per_load_in_dollars * 100

theorem cost_per_load_is_25_cents :
  cost_per_load_in_cents = 25 :=
by 
  sorry

end cost_per_load_is_25_cents_l670_670713


namespace no_prime_for_equation_l670_670046

theorem no_prime_for_equation (x k : ℕ) (p : ℕ) (h_prime : p.Prime) (h_eq : x^5 + 2 * x + 3 = p^k) : False := 
sorry

end no_prime_for_equation_l670_670046


namespace optimal_prevention_plan_l670_670394

noncomputable def expected_loss_without_measures : ℝ := 4 * 0.3
noncomputable def cost_without_measures : ℝ := expected_loss_without_measures

noncomputable def cost_A : ℝ := 45
noncomputable def prob_incident_with_A : ℝ := 1 - 0.9
noncomputable def expected_loss_with_A : ℝ := 4 * prob_incident_with_A
noncomputable def total_cost_with_A : ℝ := cost_A + expected_loss_with_A

noncomputable def cost_B : ℝ := 30
noncomputable def prob_incident_with_B : ℝ := 1 - 0.85
noncomputable def expected_loss_with_B : ℝ := 4 * prob_incident_with_B
noncomputable def total_cost_with_B : ℝ := cost_B + expected_loss_with_B

noncomputable def total_cost_with_A_and_B : ℝ := 
  let combined_cost := cost_A + cost_B
  let combined_prob := (1 - 0.9) * (1 - 0.85)
  let combined_expected_loss := 4 * combined_prob
  combined_cost + combined_expected_loss

theorem optimal_prevention_plan : min { cost_without_measures, total_cost_with_A, total_cost_with_B, total_cost_with_A_and_B } = total_cost_with_A_and_B :=
by
  sorry

end optimal_prevention_plan_l670_670394


namespace find_smallest_number_l670_670553

theorem find_smallest_number :
  ∃ x : ℕ, (x + 3) % 18 = 0 ∧ (x + 3) % 25 = 0 ∧ (x + 3) % 21 = 0 ∧ x = 3153 :=
begin
  sorry
end

end find_smallest_number_l670_670553


namespace sufficient_not_necessary_l670_670981

theorem sufficient_not_necessary (x : ℝ) :
  (x > 1 → x^2 - 2*x + 1 > 0) ∧ (¬(x^2 - 2*x + 1 > 0 → x > 1)) := by
  sorry

end sufficient_not_necessary_l670_670981


namespace line_through_intersection_parallel_to_y_axis_l670_670657

theorem line_through_intersection_parallel_to_y_axis:
  ∃ x, (∃ y, 3 * x + 2 * y - 5 = 0 ∧ x - 3 * y + 2 = 0) ∧
       (x = 1) :=
sorry

end line_through_intersection_parallel_to_y_axis_l670_670657


namespace probability_of_two_defective_in_four_tests_l670_670720

def num_components : ℕ := 6
def num_defective : ℕ := 2
def num_good : ℕ := 4
def tests_number : ℕ := 4

-- Probability calculation function (utility function)
noncomputable def binomial_probability (n k : ℕ) : ℚ :=
  nat.choose n k.realToRat * (k / n)^k.realToRat * ((n - k) / n)^((n - k).realToRat)

def probability_defective_found (components defective good tests : ℕ) : ℚ :=
  let p1 := 2/6 * 4/5 * 3/4 * 1/3 in
  let p2 := 4/6 * 2/5 * 3/4 * 1/3 in
  let p3 := 4/6 * 3/5 * 2/4 * 1/3 in
  let scenario_A := p1 + p2 + p3 in
  let scenario_B := 1/15 in
  scenario_A + scenario_B

theorem probability_of_two_defective_in_four_tests :
  probability_defective_found num_components num_defective num_good tests_number = 4/15 :=
by sorry

end probability_of_two_defective_in_four_tests_l670_670720


namespace distinct_monomials_count_l670_670836

theorem distinct_monomials_count :
  let expr := (x + y + z) ^ 2034 + (x - y - z) ^ 2034 in
  count_distinct_monomials expr = 1036324 :=
sorry

end distinct_monomials_count_l670_670836


namespace minimize_maximum_absolute_value_expression_l670_670688

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l670_670688


namespace find_a_l670_670763

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x + 1

theorem find_a
  (a : ℝ)
  (h_tangent : ∃ y : ℝ, (f a 2 = y) ∧ (f' a 1 = (f a 1 - y) / (1 - 2))) :
  a = 1 :=
by
  sorry

-- Helper function to define the derivative of f with respect to x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

end find_a_l670_670763


namespace max_product_of_two_groups_l670_670539

-- Define the set of digits available
def digits : Finset ℕ := {3, 4, 5, 6, 7, 8, 9}

-- Function to convert lists of digits to numbers
def digits_to_number (l : List ℕ) : ℕ :=
  List.foldl (λ acc d, 10 * acc + d) 0 l

-- The goal is to prove that product of 964 and 8753 is the largest possible
theorem max_product_of_two_groups :
  ∀ l1 l2 : List ℕ, l1 ∪ l2 = digits.to_list → l1 ≠ [] → l2 ≠ [] → 
  l1.length = 3 → l2.length = 4 →
  let n1 := digits_to_number l1
  let n2 := digits_to_number l2
  n1 * n2 ≤ 964 * 8753 :=
by
  -- To be filled with a proper proof
  sorry

end max_product_of_two_groups_l670_670539


namespace frog_jumps_count_l670_670085

-- Define the vertices of the hexagon and the adjacent relation
inductive Vertex
| A | B | C | D | E | F
deriving DecidableEq, Fintype

open Vertex

def adj (v1 v2 : Vertex) : Prop :=
  (v1 = A ∧ (v2 = B ∨ v2 = F)) ∨
  (v1 = B ∧ (v2 = A ∨ v2 = C)) ∨
  (v1 = C ∧ (v2 = B ∨ v2 = D)) ∨
  (v1 = D ∧ (v2 = C ∨ v2 = E)) ∨
  (v1 = E ∧ (v2 = D ∨ v2 = F)) ∨
  (v1 = F ∧ (v2 = E ∨ v2 = A))

-- Frog's jumps are represented as a sequence of vertices
def frog_jumps : List Vertex → Prop
| [] := false -- empty sequence is not valid
| [A] := true -- initial position
| A :: v1 :: [] := adj A v1 -- one jump
| v1 :: v2 :: [] := adj v1 v2 -- handle jumps
| v1 :: v2 :: xs := adj v1 v2 ∧ frog_jumps (v2 :: xs) -- recursion on the jumps

-- Definition of stopping condition: frog reaches D in at most 5 jumps or stops after 5 jumps
def stop_condition (seq : List Vertex) : Prop :=
  (seq.length ≤ 6 ∧ seq.last = some D) ∨ (seq.length = 6)

-- Count of possible sequences of jumps meeting the stop condition
def count_sequences : ℕ :=
  List.filter stop_condition (List.permutations [A,B,C,D,E,F]).length

theorem frog_jumps_count : count_sequences = 26 := 
  sorry

end frog_jumps_count_l670_670085


namespace sqrt_difference_l670_670555

theorem sqrt_difference:
  sqrt (49 + 81) - sqrt (36 - 9) = sqrt 130 - 3 * sqrt 3 :=
by
  sorry

end sqrt_difference_l670_670555


namespace monochromatic_isosceles_right_triangle_exists_l670_670597

theorem monochromatic_isosceles_right_triangle_exists (color : Point -> Color) :
  ∃ (A B C : Point), is_isosceles_right_triangle A B C ∧ (color A = color B ∧ color B = color C) :=
sorry

end monochromatic_isosceles_right_triangle_exists_l670_670597


namespace product_zero_count_l670_670708

theorem product_zero_count :
  ∃! (count : ℕ), count = 250 ∧
  (∀(n : ℕ), 1 ≤ n ∧ n ≤ 3000 → (∃ k, (1 + exp (4 * π * I * k / n))^n + 1 = 0) ↔ 
    (∃ m, n = 6 * m ∧ odd m)) :=
sorry

end product_zero_count_l670_670708


namespace actual_distance_is_correct_l670_670887

def scale := 6000000
def map_distance := 5 -- in cm

def actual_distance := map_distance * scale / 100000 -- conversion factor from cm to km

theorem actual_distance_is_correct :
  actual_distance = 300 :=
by
  simp [actual_distance, map_distance, scale]
  exact sorry

end actual_distance_is_correct_l670_670887


namespace number_of_primes_between_5000_and_8000_squared_l670_670003

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def p_in_range (p : ℕ) : Prop := 5000 < p^2 ∧ p^2 < 8000

-- Define the problem statement
theorem number_of_primes_between_5000_and_8000_squared : 
  {p : ℕ // is_prime p ∧ p_in_range p}.card = 5 :=
by
  sorry --Proof placeholder, not required as per the statement

end number_of_primes_between_5000_and_8000_squared_l670_670003


namespace mason_courses_not_finished_l670_670886

-- Each necessary condition is listed as a definition.
def coursesPerWall := 6
def bricksPerCourse := 10
def numOfWalls := 4
def totalBricksUsed := 220

-- Creating an entity to store the problem and prove it.
theorem mason_courses_not_finished : 
  (numOfWalls * coursesPerWall * bricksPerCourse - totalBricksUsed) / bricksPerCourse = 2 := 
by
  sorry

end mason_courses_not_finished_l670_670886


namespace num_teams_formed_l670_670945

-- Definitions of constants based on conditions in the problem.
def num_students : ℕ := 12
def team_size : ℕ := 6
def combinations_of_five := (Finset.powersetLen 5 (Finset.range num_students)).card
def combinations_per_team := (Finset.powersetLen 5 (Finset.range team_size)).card

-- The main theorem stating the result.
theorem num_teams_formed : combinations_of_five / combinations_per_team = 132 := by
  sorry

end num_teams_formed_l670_670945


namespace sqrt_subtraction_l670_670558

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l670_670558


namespace rationalize_denominator_l670_670106

theorem rationalize_denominator (A B C D E : ℤ) 
  (hB_lt_D : B < D) (h_fraction : (5 : ℝ) / (4*real.sqrt 7 + 3*real.sqrt 13) = (A*real.sqrt B + C*real.sqrt D) / E) 
  (h_simplest_form : true) -- assume we have the simplest terms, this would need further detail in a full proof
  : A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_l670_670106


namespace min_value_expression_l670_670081

open Real

def f (a b c : ℝ) : ℝ :=
  (a - 2) ^ 2 + (b / a - 1) ^ 2 + (c / b - 1) ^ 2 + (5 / c - 1) ^ 2 + (c - 4) ^ 2

theorem min_value_expression :
  ∃ a b c : ℝ, 2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧ f a b c = 10.1 :=
by
  use sqrt 5, sqrt 5, sqrt 5
  split
  all_goals norm_num
  sorry

end min_value_expression_l670_670081


namespace surface_area_of_circumscribed_sphere_l670_670157

theorem surface_area_of_circumscribed_sphere {a : ℝ} (h : a = 2) : 
  let d := a * sqrt 3 in
  let r := d / 2 in
  ∃ S, S = 4 * Real.pi * r^2 ∧ S = 12 * Real.pi :=
by {
  sorry
}

end surface_area_of_circumscribed_sphere_l670_670157


namespace factor_polynomial_l670_670683

theorem factor_polynomial (x y : ℝ) : 
  2*x^2 - x*y - 15*y^2 = (2*x - 5*y) * (x - 3*y) :=
sorry

end factor_polynomial_l670_670683


namespace gasoline_needed_l670_670111

variable (distance_trip : ℕ) (fuel_per_trip_distance : ℕ) (trip_distance : ℕ) (fuel_needed : ℕ)

theorem gasoline_needed (h1 : distance_trip = 140)
                       (h2 : fuel_per_trip_distance = 10)
                       (h3 : trip_distance = 70)
                       (h4 : fuel_needed = 20) :
  (fuel_per_trip_distance * (distance_trip / trip_distance)) = fuel_needed :=
by sorry

end gasoline_needed_l670_670111


namespace zero_unique_multiple_prime_l670_670565

-- Condition: let n be a number
def n : Int := sorry

-- Condition: let p be any prime number
def is_prime (p : Int) : Prop := sorry  -- Predicate definition for prime number

-- Proof problem statement
theorem zero_unique_multiple_prime (n : Int) :
  (∀ p : Int, is_prime p → (∃ k : Int, n * p = k * p)) ↔ (n = 0) := by
  sorry

end zero_unique_multiple_prime_l670_670565


namespace cannot_take_extreme_value_at_minus_one_increasing_function_range_p_l670_670765

noncomputable def f (p x : ℝ) := x^3 + 3 * p * x^2 + 3 * p * x + 1

theorem cannot_take_extreme_value_at_minus_one (p : ℝ) : 
  let f' (p x : ℝ) := 3 * x^2 + 6 * p * x + 3 * p in
  ¬ (exists a : ℝ, a = -1 ∧ f' p a = 0) :=
sorry

theorem increasing_function_range_p (p : ℝ) : 
  (∀ x ∈ Ioi (-1 : ℝ), 3 * x^2 + 6 * p * x + 3 * p ≥ 0) ↔ 
  0 ≤ p ∧ p ≤ 1 :=
sorry

end cannot_take_extreme_value_at_minus_one_increasing_function_range_p_l670_670765


namespace coach_A_spent_less_l670_670940

-- Definitions of costs and discounts for coaches purchases
def total_cost_before_discount_A : ℝ := 10 * 29 + 5 * 15
def total_cost_before_discount_B : ℝ := 14 * 2.50 + 1 * 18 + 4 * 25 + 1 * 72
def total_cost_before_discount_C : ℝ := 8 * 32 + 12 * 12

def discount_A : ℝ := 0.05 * total_cost_before_discount_A
def discount_B : ℝ := 0.10 * total_cost_before_discount_B
def discount_C : ℝ := 0.07 * total_cost_before_discount_C

def total_cost_after_discount_A : ℝ := total_cost_before_discount_A - discount_A
def total_cost_after_discount_B : ℝ := total_cost_before_discount_B - discount_B
def total_cost_after_discount_C : ℝ := total_cost_before_discount_C - discount_C

def combined_cost_B_C : ℝ := total_cost_after_discount_B + total_cost_after_discount_C
def difference_A_BC : ℝ := total_cost_after_discount_A - combined_cost_B_C

theorem coach_A_spent_less : difference_A_BC = -227.75 := by
  sorry

end coach_A_spent_less_l670_670940


namespace exists_lens_shape_curve_l670_670217

noncomputable def EquilateralTriangle := sorry

def isClosedCurve (curve : Set (ℝ × ℝ)) := sorry
def isNonSelfIntersecting (curve : Set (ℝ × ℝ)) := sorry
def isDifferentFromCircle (curve : Set (ℝ × ℝ)) := sorry
def canMoveTriangle (triangle : EquilateralTriangle) (curve : Set (ℝ × ℝ)) := sorry

theorem exists_lens_shape_curve :
  ∃ (curve : Set (ℝ × ℝ)),
    isClosedCurve curve ∧
    isNonSelfIntersecting curve ∧
    isDifferentFromCircle curve ∧
    (∃ (triangle : EquilateralTriangle), canMoveTriangle triangle curve) ∧
    (∀ (arc1 arc2 : Set (ℝ × ℝ)), isLensShape curve arc1 arc2) :=
sorry

end exists_lens_shape_curve_l670_670217


namespace train_speed_l670_670574

theorem train_speed (L_t L_b : ℕ) (T : ℝ) (h1 : L_t = 250) (h2 : L_b = 180) (h3 : T = 20) : 
  (L_t + L_b) / T = 21.5 := 
  by 
    rw [h1, h2, h3]
    norm_num
    sorry

end train_speed_l670_670574


namespace problem_statement_l670_670832

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Definition of arithmetic sequence
def arithmetic_sequence (n : ℕ) (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given conditions
variable (h1 : arithmetic_sequence a d)
variable (h2 : a 1 + 3 * a 8 + a 15 = 100)

-- Proof problem statement
theorem problem_statement : 2 * a 9 - a 10 = 20 := 
by
  sorry

end problem_statement_l670_670832


namespace recommended_cooking_time_is_5_minutes_l670_670259

-- Define the conditions
def time_cooked := 45 -- seconds
def time_remaining := 255 -- seconds

-- Define the total cooking time in seconds
def total_time_seconds := time_cooked + time_remaining

-- Define the conversion from seconds to minutes
def to_minutes (seconds : ℕ) : ℕ := seconds / 60

-- The main theorem to prove
theorem recommended_cooking_time_is_5_minutes :
  to_minutes total_time_seconds = 5 :=
by
  sorry

end recommended_cooking_time_is_5_minutes_l670_670259


namespace tangent_line_equation_l670_670766

theorem tangent_line_equation 
    (a b : ℝ) 
    (f : ℝ → ℝ) 
    (f' : ℝ → ℝ)
    (h₁ : f = λ x, x^3 + a*x^2 + b*x + 1)
    (h₂ : f' = λ x, 3*x^2 + 2*a*x + b)
    (h₃ : f' 1 = 2*a)
    (h₄ : f' 2 = -b) :
    6 * x + 2 * y - 1 = 0 := 
begin
  sorry
end

end tangent_line_equation_l670_670766


namespace cone_base_radius_l670_670172

open Real

theorem cone_base_radius (r_sector : ℝ) (θ_sector : ℝ) : 
    r_sector = 6 ∧ θ_sector = 120 → (∃ r : ℝ, 2 * π * r = θ_sector * π * r_sector / 180 ∧ r = 2) :=
by
  sorry

end cone_base_radius_l670_670172


namespace range_of_m_l670_670665

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, (x + m) * (2 - x) < 1) ↔ (-4 < m ∧ m < 0) :=
sorry

end range_of_m_l670_670665


namespace polygon_RS_plus_ST_l670_670495

/-- Given a polygon PQRSTU with area 78 and side lengths PQ = 10, QR = 11, and TU = 7, 
    prove that the sum of RS and ST equals 11. -/
theorem polygon_RS_plus_ST (P Q R S T U : Point)
  (h_area : area_polygon P Q R S T U = 78)
  (h_PQ : length_segment PQ = 10)
  (h_QR : length_segment QR = 11)
  (h_TU : length_segment TU = 7) :
  length_segment RS + length_segment ST = 11 :=
sorry

end polygon_RS_plus_ST_l670_670495


namespace distribute_balls_into_boxes_l670_670347

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l670_670347


namespace part1_min_area_part2_fixed_point_l670_670743

noncomputable def problem1 (m p k1 k2 : ℝ) (h_n : 0 = 0) (h_k: k1 * k2 = -1) : ℝ := p^2

theorem part1_min_area 
  (m p k1 k2 : ℝ)
  (h: 0 = 0)
  (h_k: k1 * k2 = -1) :
  let area := problem1 m p k1 k2 h h_k in
  area = p^2 :=
by sorry

def lineMN_fixed_point (m n p k1 k2 : ℝ) (λ : ℝ) (h_sum: k1 + k2 = λ) : ℝ × ℝ :=
(m - n / λ, p / λ)

theorem part2_fixed_point
  (m n p k1 k2 : ℝ)
  (λ : ℝ)
  (h_sum: k1 + k2 = λ) :
  let fixed_point := lineMN_fixed_point m n p k1 k2 λ h_sum in
  fixed_point = (m - n / λ, p / λ) :=
by sorry

end part1_min_area_part2_fixed_point_l670_670743


namespace minimize_maximum_absolute_value_expression_l670_670687

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l670_670687


namespace distance_between_parallel_lines_l670_670917

theorem distance_between_parallel_lines :
  let L1 := λ (x y : ℝ), 3 * x + 4 * y - 3 = 0
  let L2 := λ (x y : ℝ), 6 * x + 8 * y + 7 = 0
  ∀ (x y : ℝ), L1 x y → L2 x y → 
  Real.dist ((3 : ℝ) * (x : ℝ) + 4 * (y : ℝ) - 3) ((6 : ℝ) * (x : ℝ) + 8 * (y : ℝ) + 7) = 13 / 10 :=
begin
  sorry
end

end distance_between_parallel_lines_l670_670917


namespace normal_chord_min_area_proof_l670_670276

noncomputable def normal_chord_min_area (a : ℝ) : ℝ :=
  sorry

theorem normal_chord_min_area_proof :
  ∃ q : ℝ, normal_chord_min_area q = q :=
begin
  sorry
end

end normal_chord_min_area_proof_l670_670276


namespace books_about_history_and_science_l670_670808

theorem books_about_history_and_science (total books about_school books about_sports : ℕ) 
  (h_total : total books = 120)
  (h_school : books about_school = 25)
  (h_sports : books about_sports = 35) :
  books about_history_and_science = 60 :=
by
  sorry

end books_about_history_and_science_l670_670808


namespace arithmetic_mean_x_A_is_2003_l670_670980

def M : Set ℕ := { x | 1 ≤ x ∧ x ≤ 2002 }

def x_A (A : Set ℕ) [Nonempty A] : ℕ := (A.toFinset.max' (finite_mem_finset A).to_finset_nonempty) +
                                     (A.toFinset.min' (finite_mem_finset A).to_finset_nonempty)

noncomputable def arithmetic_mean_x_A : ℕ :=
  let all_x_A := { x_A A | A ∈ powerset M ∧ A ≠ ∅ }
  in (Finset.sum (all_x_A.toFinset)) / all_x_A.toFinset.card

theorem arithmetic_mean_x_A_is_2003 : arithmetic_mean_x_A = 2003 := by
  sorry

end arithmetic_mean_x_A_is_2003_l670_670980


namespace find_length_AC_l670_670839

-- Define the conditions and statement 
theorem find_length_AC (α : ℝ) (AB AC : ℝ) (A₆₀ : α = real.pi / 3)
  (AB₂ : AB = 2) (area_eq : (1 / 2) * AB * AC * real.sin α = √3 / 2) :
  AC = 1 :=
sorry

end find_length_AC_l670_670839


namespace median_of_set_l670_670118

theorem median_of_set (a : ℤ) (b : ℝ) (h1 : a ≠ 0) (h2 : 0 < b ∧ b < 1) (h3 : a * b^3 = Real.log b / Real.log 10) :
  (List.median [0, 1, a, b, 1/b]) = b := by
sorry

end median_of_set_l670_670118


namespace sum_of_squares_not_perfect_square_l670_670893

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n > 4) :
  ¬ (∃ k : ℕ, 10 * n^2 + 10 * n + 85 = k^2) :=
sorry

end sum_of_squares_not_perfect_square_l670_670893


namespace sum_modulo_7_is_zero_l670_670954

theorem sum_modulo_7_is_zero :
  (∑ i in Finset.range 148, i) % 7 = 0 := sorry

end sum_modulo_7_is_zero_l670_670954


namespace simplify_expression_l670_670113

theorem simplify_expression (w : ℝ) : (5 - 2 * w) - (4 + 5 * w) = 1 - 7 * w := by 
  sorry

end simplify_expression_l670_670113


namespace nat_as_sum_of_distinct_fib_terms_l670_670101

noncomputable def fib_seq (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 5
  else fib_seq (n - 1) + fib_seq (n - 2)

theorem nat_as_sum_of_distinct_fib_terms (n : ℕ) : 
  ∃ S : finset ℕ, (S.sum fib_seq = n) ∧ (∀ a b ∈ S, a ≠ b) :=
by
  sorry

end nat_as_sum_of_distinct_fib_terms_l670_670101


namespace original_price_of_car_l670_670177

-- Define the original price of the car based on the condition of the problem
def original_price (spent : ℝ) (percentage : ℝ) : ℝ := spent / percentage

-- Given conditions
def venny_spent : ℝ := 15000
def percentage_of_original : ℝ := 0.40

-- Statement to be proved
theorem original_price_of_car : original_price venny_spent percentage_of_original = 37500 := by
  sorry

end original_price_of_car_l670_670177


namespace largest_is_A_l670_670278

def A : ℝ := 3010 / 3009 + 3010 / 3011
def B : ℝ := 3010 / 3011 + 3012 / 3011
def C : ℝ := 3011 / 3010 + 3011 / 3012

theorem largest_is_A : A > B ∧ A > C :=
by
  sorry

end largest_is_A_l670_670278


namespace problem_domains_equal_l670_670251

/-- Proof problem:
    Prove that the domain of the function y = (x - 1)^(-1/2) is equal to the domain of the function y = ln(x - 1).
--/
theorem problem_domains_equal :
  {x : ℝ | x > 1} = {x : ℝ | x > 1} :=
by
  sorry

end problem_domains_equal_l670_670251


namespace scientific_notation_of_400000_l670_670092

theorem scientific_notation_of_400000 :
  (400000: ℝ) = 4 * 10^5 :=
by 
  sorry

end scientific_notation_of_400000_l670_670092


namespace part_a_part_b_l670_670986

-- Part (a)
theorem part_a (n : ℕ) (h : n ≥ 4): 
  ∃ (a : ℕ), let S := { a + i | i < n } in (a + n - 1) ∣ Nat.lcm (S.erase (a + n - 1)) :=
sorry

-- Part (b)
theorem part_b (h : n = 4): 
  ∃!(a : ℕ), let S := { a + i | i < 4 } in (a+3) ∣ Nat.lcm (S.erase (a + 3)) :=
sorry

end part_a_part_b_l670_670986


namespace daisy_area_proof_l670_670944
-- importing the Mathlib library

-- defining a noncomputable function to state the problem
noncomputable def daisy_area_problem : Prop :=
  let side_length := 12 in
  let square_area := side_length ^ 2 in
  let daisy_area := 48 in
  daisy_area = (square_area / 2)

-- asserting the problem statement
theorem daisy_area_proof : daisy_area_problem :=
by
  -- the proof is not required, so we use sorry.
  sorry

end daisy_area_proof_l670_670944


namespace vector_orthogonality_if_and_only_if_squared_sum_is_correct_l670_670073

variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V) (non_zero_a : a ≠ 0) (non_zero_b : b ≠ 0)

theorem vector_orthogonality_if_and_only_if_squared_sum_is_correct :
  (∥a + b∥ ^ 2 = ∥a∥ ^ 2 + ∥b∥ ^2) ↔ (inner a b = 0) := 
sorry

end vector_orthogonality_if_and_only_if_squared_sum_is_correct_l670_670073


namespace area_of_triangle_with_medians_l670_670649

theorem area_of_triangle_with_medians
  (s_a s_b s_c : ℝ) :
  (∃ t : ℝ, t = (1 / 3 : ℝ) * ((s_a + s_b + s_c) * (s_b + s_c - s_a) * (s_a + s_c - s_b) * (s_a + s_b - s_c)).sqrt) :=
sorry

end area_of_triangle_with_medians_l670_670649


namespace find_C_coordinates_l670_670892

/-- Points A, B, D with given coordinates -/
def A := (10, 7)
def B := (1, -5)
def D := (0, 1)

/-- Isosceles triangle ABC with AB = AC and altitude from A to BC intersects at D -/
theorem find_C_coordinates
  (h_isosceles: dist A B = dist A (C : ℝ × ℝ))
  (h_altitude: ∃ D, D = (0, 1) ∧ (2 * D.1 = B.1 + C.1) ∧ (2 * D.2 = B.2 + C.2)) :
  C = (-1, 7) :=
sorry  -- Proof omitted

end find_C_coordinates_l670_670892


namespace berry_difference_l670_670281

/-- Define the initial total number of berries on the bush -/
def total_berries : ℕ := 900

/-- Sergey collects 1 out of every 2 berries he picks -/
def sergey_collection_ratio : ℕ := 2

/-- Dima collects 2 out of every 3 berries he picks -/
def dima_collection_ratio : ℕ := 3

/-- Sergey picks berries twice as fast as Dima -/
def sergey_speed_multiplier : ℕ := 2

/-- Prove that the difference between berries collected in Sergey's and Dima's baskets is 100 -/
theorem berry_difference : 
  let total_picked_sergey := (sergey_speed_multiplier * total_berries) / (sergey_speed_multiplier + 1),
      total_picked_dima := total_berries / (sergey_speed_multiplier + 1),
      sergey_basket := total_picked_sergey / sergey_collection_ratio,
      dima_basket := (2 * total_picked_dima) / dima_collection_ratio
  in sergey_basket - dima_basket = 100 :=
by
  sorry -- proof to be completed

end berry_difference_l670_670281


namespace infinite_circles_intersect_at_most_two_l670_670407

theorem infinite_circles_intersect_at_most_two (r : ℝ) (x_i : ℕ → ℝ) :
  (∀ n : ℕ, ∃ c : ℝ, (x_i n, (x_i n)^2) = c) →
  (∀ m c : ℝ, ∃ (x : ℝ), (x, (x_i n)^2) intersects line (y = mx + c) where n ∈ ℕ) →
  ∃ arrangement : (ℕ → ℝ) × (ℝ → ℝ) → Prop,
    ∀ L : ℝ → ℝ, ∀ x y : ℝ,
      (L y = mx + c) ∧ (circle_centered_at (x_i n, (x_i n)^2) of radius r) → 
      intersects arrangement at most twice :=
by
  sorry

end infinite_circles_intersect_at_most_two_l670_670407


namespace exists_coprime_point_exceeds_2020_l670_670080

def Z_plus := {n : ℕ // n > 0}

def coprime_points : Set (ℕ × ℕ) := 
  {p | p.1 > 0 ∧ p.2 > 0 ∧ Nat.gcd p.1 p.2 = 1}

def euclidean_distance (p q : ℕ × ℕ) : ℝ :=
  Real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

theorem exists_coprime_point_exceeds_2020 : 
  ∃ (a b : ℕ), (a, b) ∈ coprime_points ∧ 
    ∀ (x y : ℕ), (x, y) ∈ coprime_points → 
      euclidean_distance (a, b) (x, y) > 2020 :=
sorry

end exists_coprime_point_exceeds_2020_l670_670080


namespace sailboat_speed_at_max_power_l670_670591

variable (C S ρ v_0 v N: ℝ)

-- Conditions
axiom sail_area : S = 5  -- S in square meters
axiom wind_speed : v_0 = 6  -- v_0 in m/s
axiom force_formula : C * S * ρ * (v_0 - v) ^ 2 / 2 = N / v

-- Proof statement
theorem sailboat_speed_at_max_power
  (h₁ : sail_area)
  (h₂ : wind_speed)
  (h₃ : force_formula) :
  v = v_0 / 3 :=
sorry

end sailboat_speed_at_max_power_l670_670591


namespace leo_assignment_solution_l670_670420

def leo_assignment_problem : Prop :=
  ∃ (first_part second_part third_part total_time : ℕ), 
  first_part = 25 ∧
  total_time = 120 ∧
  third_part = 45 ∧
  (second_part = total_time - (first_part + third_part)) ∧
  (second_part / first_part = 2)

theorem leo_assignment_solution : leo_assignment_problem :=
by {
  use [25, 50, 45, 120],
  simp,
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { exact rfl },
  split,
  { exact rfl },
  { rw [nat.div_eq_of_eq_mul_right, nat.mul_comm], exact rfl, exact dec_trivial, },
}

end leo_assignment_solution_l670_670420


namespace range_of_a_l670_670756

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x - a / 2 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (4 / 3 ≤ a ∧ a < 2) :=
by 
  intro h
  sorry

end range_of_a_l670_670756


namespace trigonometric_inequality_l670_670105

theorem trigonometric_inequality (x : ℝ) : 0 ≤ 5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ∧ 
                                            5 + 8 * Real.cos x + 4 * Real.cos (2 * x) + Real.cos (3 * x) ≤ 18 :=
by
  sorry

end trigonometric_inequality_l670_670105


namespace proof_problem_l670_670402

variables (PQ MN OR MP NR : ℝ) (s d : ℝ) (O : ℝ)

-- Given conditions:
def unit_circle_conditions : Prop :=
  (PQ ∥ OR) ∧ (MN ∥ OR) ∧ 
  (MP = s) ∧ (PQ = s) ∧ (NR = s) ∧ (MN = d)

-- Statements to prove:
def statements (d s : ℝ) : Prop :=
  (d - s = 1) ∧ 
  (d * s = 1) ∧ 
  (d^2 - s^2 = sqrt 5)

theorem proof_problem (h : unit_circle_conditions PQ MN OR MP NR s d O) :
  statements d s :=
by {
  sorry
}

end proof_problem_l670_670402


namespace constant_term_expansion_l670_670951

theorem constant_term_expansion :
  let poly1 := (λ x : ℤ, x^6 + x^2 + 3)
  let poly2 := (λ x : ℤ, x^4 + x^3 + 20)
  ∀ x : ℤ, (poly1 x * poly2 x).coeff 0 = 60 := by
  sorry

end constant_term_expansion_l670_670951


namespace price_restoration_percentage_l670_670515

noncomputable def original_price := 100
def reduced_price (P : ℝ) := 0.8 * P
def restored_price (P : ℝ) (x : ℝ) := P = x * reduced_price P

theorem price_restoration_percentage (P : ℝ) (x : ℝ) (h : restored_price P x) : x = 1.25 :=
by
  sorry

end price_restoration_percentage_l670_670515


namespace find_x_in_terms_of_y_l670_670365

theorem find_x_in_terms_of_y 
(h₁ : x ≠ 0) 
(h₂ : x ≠ 3) 
(h₃ : y ≠ 0) 
(h₄ : y ≠ 5) 
(h_eq : 3 / x + 2 / y = 1 / 3) : 
x = 9 * y / (y - 6) :=
by
  sorry

end find_x_in_terms_of_y_l670_670365


namespace min_distance_between_squares_l670_670572

-- Define the conditions and the problem in Lean
theorem min_distance_between_squares :
  ∀ (ABCD EFGH : ℝ) (s : ℝ) (area_overlap : ℝ) 
  (AB_parallel_EF : Prop) (side_length_one : s = 1) 
  (overlap_area: area_overlap = 1/16),
  s = 1 → -- side length of squares ABCD and EFGH is 1
  AB_parallel_EF → -- AB is parallel to EF
  overlap_area = 1/16 → -- the area of overlapping region is 1/16
  -- then the minimum distance between the centers of the two squares is √(14)/4
  ∃ (dist : ℝ), dist = (Real.sqrt 14) / 4 := 
begin
  intros ABCD EFGH s area_overlap AB_parallel_EF side_length_one overlap_area _ _ _,
  use (Real.sqrt 14) / 4,
  sorry, -- Proof goes here
end

end min_distance_between_squares_l670_670572


namespace degree_of_divisor_l670_670627

theorem degree_of_divisor (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15) 
  (hq : q.degree = 9) 
  (hr : r.degree = 4) 
  (hr_poly : r = (Polynomial.C 5) * (Polynomial.X^4) + (Polynomial.C 6) * (Polynomial.X^3) - (Polynomial.C 2) * (Polynomial.X) + (Polynomial.C 7)) 
  (hdiv : f = d * q + r) : 
  d.degree = 6 := 
sorry

end degree_of_divisor_l670_670627


namespace ratio_of_speeds_l670_670286

-- Conditions
def total_distance_Eddy : ℕ := 200 + 240 + 300
def total_distance_Freddy : ℕ := 180 + 420
def total_time_Eddy : ℕ := 5
def total_time_Freddy : ℕ := 6

-- Average speeds
def avg_speed_Eddy (d t : ℕ) : ℚ := d / t
def avg_speed_Freddy (d t : ℕ) : ℚ := d / t

-- Ratio of average speeds
def ratio_speeds (s1 s2 : ℚ) : ℚ := s1 / s2

theorem ratio_of_speeds : 
  ratio_speeds (avg_speed_Eddy total_distance_Eddy total_time_Eddy) 
               (avg_speed_Freddy total_distance_Freddy total_time_Freddy) 
  = 37 / 25 := by
  -- Proof omitted
  sorry

end ratio_of_speeds_l670_670286


namespace equation_of_circle_trajectory_of_midpoint_l670_670723

-- Define the function for the circle's equation.
def circle_equation (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - 2 * a * x = 0

-- First part: Prove that if (4, 2) lies on the circle passing through the origin, 
-- then the equation of the circle is x^2 + y^2 - 5x = 0.
theorem equation_of_circle (a : ℝ) : 
  (circle_equation a 4 2) → (a = 2.5) → (circle_equation 2.5 = circle_equation 5) :=
by 
sorry

-- Define the function for the midpoint's trajectory equation.
def trajectory_equation (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, x^2 + y^2 - a * x = 0

-- Second part: Prove that the trajectory of the midpoint of chord OA
-- on the circle is x^2 + y^2 - 2.5x = 0, given a circle x^2 + y^2 - 5x = 0.
theorem trajectory_of_midpoint (a : ℝ) (x y : ℝ) :
  (circle_equation 2.5 (2 * x) (2 * y)) → (trajectory_equation 2.5 x y) :=
by 
sorry

end equation_of_circle_trajectory_of_midpoint_l670_670723


namespace isosceles_trapezoid_AB_length_l670_670039

theorem isosceles_trapezoid_AB_length (BC AD : ℝ) (r : ℝ) (a : ℝ) (h_isosceles : BC = a) (h_ratio : AD = 3 * a) (h_area : 4 * a * r = Real.sqrt 3 / 2) (h_radius : r = a * Real.sqrt 3 / 2) :
  2 * a = 1 :=
by
 sorry

end isosceles_trapezoid_AB_length_l670_670039


namespace find_m_l670_670773

def is_constant_coefficient (a : ℤ) : Prop := a = 1

theorem find_m (m : ℤ) (f : ℤ → ℤ) (h : f = λ x, (m - 2) * x^(m^2 - 2*m)) : 
  (is_constant_coefficient (m - 2)) → m = 3 := by
  sorry

end find_m_l670_670773


namespace ratio_of_cosines_l670_670827

theorem ratio_of_cosines (ABC : Type) [triangle ABC] 
  (acute : ∀ A B C : ABC, angle A < 90 ∧ angle B < 90 ∧ angle C < 90) 
  (non_isosceles: ¬(angle A = angle B ∨ angle B = angle C ∨ angle C = angle A)) :
  (∀ O : Point, ∀ AH BB1 : Line, ∀ H : Point,
     is_circumcenter O ∧ is_altitude AH ∧ is_altitude BB1 ∧
     intersects_at H AH BB1 ∧
     (∃ CX CY : Segment, between CX AH H ∧ between CY BB1 H) →
     ∃ B C : Point, ∀ A : Point,
       ∃ (cos_angle_B cos_angle_A : ℝ),
         ratio_segments CX CY = cos_angle B / cos_angle A) := sorry

end ratio_of_cosines_l670_670827


namespace area_of_rhombus_l670_670548

-- Defining the conditions
def diagonal1 : ℝ := 20
def diagonal2 : ℝ := 30

-- Proving the area of the rhombus
theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = diagonal1) (h2 : d2 = diagonal2) : 
  (d1 * d2 / 2) = 300 := by
  sorry

end area_of_rhombus_l670_670548


namespace even_function_expression_l670_670746

noncomputable def f : ℝ → ℝ
| x => if x < 0 then x - x^4 else if x > 0 then -x - x^4 else 0

theorem even_function_expression (x : ℝ) (h : f (-x) = f x) (h_neg : ∀ x < 0, f x = x - x^4) : 
  (0 < x) → f x = -x - x^4 :=
by
  intros hx
  have h1 : f x = f (-x) := h
  have h2 : f (-x) = -x - x^4 := by 
    apply h_neg
    exact neg_lt_zero.mpr hx
  rw h2 at h1
  exact h1

end even_function_expression_l670_670746


namespace angles_of_terminal_side_on_line_y_equals_x_l670_670153

noncomputable def set_of_angles_on_y_equals_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * 180 + 45

theorem angles_of_terminal_side_on_line_y_equals_x (α : ℝ) :
  (∃ k : ℤ, α = k * 360 + 45) ∨ (∃ k : ℤ, α = k * 360 + 225) ↔ set_of_angles_on_y_equals_x α :=
by
  sorry

end angles_of_terminal_side_on_line_y_equals_x_l670_670153


namespace propositions_correctness_l670_670328

-- Define each proposition
def prop1 : Prop := ∃ α, sin α + cos α = 3 / 2
def prop2 : Prop := ∀ x, sin (5 / 2 * Real.pi - 2 * x) = sin (2 * x)
def prop3 : Prop := x = Real.pi / 8 → sin (2 * x + 5 * Real.pi / 4) = sin (2 * (x + Real.pi / 8) + 3 * Real.pi / 8)
def prop4 : Prop := ∀ x ∈ Ioo 0 (Real.pi / 2), e ^ (sin 2 * x) is an increasing function
def prop5 : Prop := ∀ α β : ℝ, 0 < α ∧ α < Real.pi / 2 ∧ 0 < β ∧ β < Real.pi / 2 ∧ α > β → tan α > tan β
def prop6 : Prop := ∀ x, 3 * sin (2 * x + Real.pi / 3) = 3 * sin (2 * (x + Real.pi / 6))

-- Statement showing which propositions are correct
theorem propositions_correctness : ¬ prop1 ∧ prop2 ∧ prop3 ∧ ¬ prop4 ∧ ¬ prop5 ∧ ¬ prop6 :=
by
  sorry

end propositions_correctness_l670_670328


namespace simplify_expression_l670_670681

theorem simplify_expression (x y z : ℝ) :
  3 * (x - (2 * y - 3 * z)) - 2 * ((3 * x - 2 * y) - 4 * z) = -3 * x - 2 * y + 17 * z :=
by
  sorry

end simplify_expression_l670_670681


namespace greatest_A_l670_670441

-- Define the function f(x) = x^2 - r2 * x + r3
def f (r2 r3 : ℝ) (x : ℝ) : ℝ := x^2 - r2 * x + r3

-- Define the sequence g_n recursively
def g (r2 r3 : ℝ) : ℕ → ℝ
| 0       := 0
| (n + 1) := f r2 r3 (g n)

-- Define the conditions on the sequence g_n
def cond1 (r2 r3 : ℝ) : Prop :=
∀ i : ℕ, 0 ≤ i ∧ i ≤ 2011 → g r2 r3 (2 * i) < g r2 r3 (2 * i + 1) ∧ 
                                   g r2 r3 (2 * i + 1) > g r2 r3 (2 * i + 2)

def cond2 (r2 r3 : ℝ) : Prop :=
∃ j : ℕ, ∀ i : ℕ, i > j → g r2 r3 (i + 1) > g r2 r3 i

def cond3 (r2 r3 : ℝ) : Prop :=
∀ M : ℝ, ∃ n : ℕ, g r2 r3 n > M

-- Prove that the greatest A satisfying A ≤ |r2| is 2
theorem greatest_A (r2 r3 : ℝ) :
  cond1 r2 r3 →
  cond2 r2 r3 →
  cond3 r2 r3 →
  ∃ A, A = 2 ∧ A ≤ |r2| :=
by
  sorry

end greatest_A_l670_670441


namespace team_C_games_count_l670_670911

theorem team_C_games_count 
  (C_games_won : ℕ) (D_games_won : ℕ) (C_games : ℕ)
  (hC1 : 4 * C_games_won = 3 * C_games)
  (hD1 : 10 * D_games_won = 7 * C_games)
  (hD2 : D_games_won = C_games_won + 5)
  (hD3 : C_games - C_games_won = D_games - D_games_won + 5) :
  C_games = 100 := 
begin
  -- Proof can be given here
  sorry
end

end team_C_games_count_l670_670911


namespace remainder_problem_l670_670819

theorem remainder_problem (n m q1 q2 : ℤ) (h1 : n = 11 * q1 + 1) (h2 : m = 17 * q2 + 3) :
  ∃ r : ℤ, (r = (5 * n + 3 * m) % 11) ∧ (r = (7 * q2 + 3) % 11) :=
by
  sorry

end remainder_problem_l670_670819


namespace perfect_squares_in_range_100_400_l670_670785

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l670_670785


namespace find_x_l670_670392

def angle_sum_condition (x : ℝ) := 6 * x + 3 * x + x + x + 4 * x = 360

theorem find_x (x : ℝ) (h : angle_sum_condition x) : x = 24 := 
by {
  sorry
}

end find_x_l670_670392


namespace magnitude_a_sub_b_l670_670338

variables {ℝ : Type*} [LinearOrderedField ℝ]

def vec_a (x : ℝ) : ℝ × ℝ := (4^x, 2^x)
def vec_b (x : ℝ) : ℝ × ℝ := (1, (2^x - 2) / 2^x)

noncomputable def magnitude (a : ℝ × ℝ) : ℝ := real.sqrt ((a.1)^2 + (a.2)^2)

theorem magnitude_a_sub_b (x : ℝ) (h : vec_a x.1 * vec_b x.1 + vec_a x.2 * vec_b x.2 = 0) : magnitude (vec_a x - vec_b x) = 2 :=
by
  sorry

end magnitude_a_sub_b_l670_670338


namespace perfect_squares_between_100_and_400_l670_670799

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l670_670799


namespace cesaro_mean_convergence_l670_670069

open_locale classical
noncomputable theory

-- Definitions of the sequences
def u_seq (u : ℕ → ℝ) (n : ℕ) : ℝ := u n
def c_seq (u : ℕ → ℝ) (n : ℕ) : ℝ := (1 / n) * (∑ k in finset.range n, u (k + 1))

-- Hypothesis: u_seq converges to a limit ell
def u_seq_converges (u : ℕ → ℝ) (ell : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (u n - ell) ≤ ε

-- Objective: c_seq converges to the same limit ell
def c_seq_converges (u : ℕ → ℝ) (ell : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, abs (c_seq u_seq n - ell) ≤ ε

-- Main theorem statement
theorem cesaro_mean_convergence {u : ℕ → ℝ} {ell : ℝ} (h : u_seq_converges u ell) : c_seq_converges u ell :=
sorry

end cesaro_mean_convergence_l670_670069


namespace solution_to_marketing_firm_problem_l670_670625

def marketing_firm_problem (total_households neither_A_B both_A_B : ℕ) : Prop :=
let only_B := 3 * both_A_B in
let total_using_soap := total_households - neither_A_B in
let only_A := total_using_soap - only_B - both_A_B in
only_A = 60

theorem solution_to_marketing_firm_problem (total_households neither_A_B both_A_B : ℕ) 
  (h1 : total_households = 260)
  (h2 : neither_A_B = 80)
  (h3 : both_A_B = 30) : 
  marketing_firm_problem total_households neither_A_B both_A_B := 
by {
  unfold marketing_firm_problem,
  rw [h1, h2, h3],
  norm_num
}

end solution_to_marketing_firm_problem_l670_670625


namespace extreme_point_of_f_l670_670445

def f' (x : ℝ) : ℝ := x^3 - 3 * x + 2

theorem extreme_point_of_f : 
  ∃ x : ℝ, f' x = 0 ∧ x = -2 ∧ (∀ ε > 0, ∀ y : ℝ, y ∈ Ioo (x - ε) x ∨ y ∈ Ioo x (x + ε) → f' y < 0) :=
sorry

end extreme_point_of_f_l670_670445


namespace circumradius_twice_l670_670060

-- Definitions and conditions
variables {A B C M N : Type*} [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty M] [Nonempty N]
variables (BE : A → B → Prop) (CF : A → C → Prop) (ABC : A → B → C → Prop) (I : A)
variables (EF : A → B → C → Prop) (circumcircleABC : A → B → C → Type*) (circumcircleMIN : A → M → N → Type*)

-- Definitions of triangle, bisectors, and incentre
def angle_bisector (angle : A → B → Prop) (bisector : A → B → Prop) : Prop :=
∀ x y, angle x y → bisector x y

def incentre (point : A) (triangle : A → B → C → Prop) : Prop :=
∀ x y z, triangle x y z → point = I

-- Definition of extension meeting circumcircle
def extension_meets_circumcircle (EF : A → B → C → Prop) (circumcircle : A → B → C → Type*) (points : A × M × N) : Prop :=
∃ m n, EF m n ∧ circumcircle m n

-- Definition of circumradius
def circumradius (triangle : A → M → N → Prop) (r : ℝ) : Prop :=
∀ x y z, triangle x y z → ∀ r', r = 2 * r'

-- Theorem statement
theorem circumradius_twice
  (h₁ : angle_bisector (λ B C, ∠B = ∠C) BE)
  (h₂ : angle_bisector (λ B C, ∠B = ∠C) CF)
  (h₃ : incentre I ABC)
  (h₄ : extension_meets_circumcircle EF circumcircleABC (M, N))
  (h₅ : circumradius ABC 1)
  : circumradius (λ M I N, ∠M = ∠I ∧ ∠I = ∠N) 2 :=
sorry

end circumradius_twice_l670_670060


namespace birdseed_mix_percentage_l670_670616

theorem birdseed_mix_percentage (x : ℝ) :
  (0.40 * x + 0.65 * (100 - x) = 50) → x = 60 :=
by
  sorry

end birdseed_mix_percentage_l670_670616


namespace abs_two_minus_sqrt_five_l670_670913

noncomputable def sqrt_5 : ℝ := Real.sqrt 5

theorem abs_two_minus_sqrt_five : |2 - sqrt_5| = sqrt_5 - 2 := by
  sorry

end abs_two_minus_sqrt_five_l670_670913


namespace tangents_parallel_to_x_axis_l670_670093

-- Define the function
def f (x : ℝ) : ℝ := x * (x - 4)^3

-- Statement of the theorem
theorem tangents_parallel_to_x_axis :
  ∃ p₁ p₂ : ℝ × ℝ, p₁ = (4, 0) ∧ p₂ = (1, -27) ∧
  (∀ x y, f' x = 0 → y = f x → (x, y) = p₁ ∨ (x, y) = p₂) :=
by
  -- Placeholder for the proof
  sorry

end tangents_parallel_to_x_axis_l670_670093


namespace starWarsEarned405_l670_670127

-- Definitions and Hypotheses
variables (cost_LionKing : ℕ) (earnings_LionKing : ℕ) (cost_StarWars : ℕ) (earnings_StarWars : ℕ)
variables (profit_LionKing half_profit_StarWars profit_StarWars : ℕ)

-- Conditions
def lionKingCost : cost_LionKing = 10 := rfl
def lionKingEarnings : earnings_LionKing = 200 := rfl
def starWarsCost : cost_StarWars = 25 := rfl
def lionKingProfit : profit_LionKing = earnings_LionKing - cost_LionKing := by rw [lionKingEarnings, lionKingCost]
def halfProfitCondition : profit_LionKing = profit_StarWars / 2 := by rw lionKingProfit
def starWarsEarnings : earnings_StarWars = cost_StarWars + profit_StarWars := by rw [starWarsCost]

-- Main theorem
theorem starWarsEarned405 : earnings_StarWars = 405 :=
by {
  rw [starWarsEarnings, lionKingProfit, halfProfitCondition, lionKingEarnings, starWarsCost],
  simp, 
  refine sorry,  -- Proof steps skipped
}

end starWarsEarned405_l670_670127


namespace sqrt_subtraction_l670_670563

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l670_670563


namespace find_general_term_l670_670327

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_general_term (a : ℕ → ℝ) (T : ℕ → ℝ) (h_arith_seq : arithmetic_sequence a) 
  (h_T_def : ∀ n : ℕ, T n = ∑ i in range n, (n - i) * a (i + 1))
  (h_T2 : T 2 = 7) (h_T3 : T 3 = 16) : 
  ∀ n : ℕ, a n = n + 1 :=
sorry

end find_general_term_l670_670327


namespace polygon_area_l670_670399

theorem polygon_area (n sides_perpendicular sides_congruent : ℕ) (h1 : n = 32) (h2 : sides_perpendicular = true) 
(h3 : sides_congruent = true) (perimeter : ℕ) (h4 : perimeter = 64) : 
  let s := perimeter / n in 
  let area := 36 * s * s in 
  area = 144 :=
by
  sorry

end polygon_area_l670_670399


namespace coefficient_of_1_div_x2_in_binomial_expansion_l670_670295

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

theorem coefficient_of_1_div_x2_in_binomial_expansion :
  let x := (1 : ℝ)
  let expr := sqrt x + 2 / x
  let expansion := expr ^ 5
  let r := 3 in
  binomial_coefficient 5 3 * 2^3 = 80
:= by
  sorry

end coefficient_of_1_div_x2_in_binomial_expansion_l670_670295


namespace Chloe_second_round_points_l670_670655

-- Conditions
def firstRoundPoints : ℕ := 40
def lastRoundPointsLost : ℕ := 4
def totalPoints : ℕ := 86
def secondRoundPoints : ℕ := 50

-- Statement to prove: Chloe scored 50 points in the second round
theorem Chloe_second_round_points :
  firstRoundPoints + secondRoundPoints - lastRoundPointsLost = totalPoints :=
by {
  -- Proof (not required, skipping with sorry)
  sorry
}

end Chloe_second_round_points_l670_670655


namespace smallest_integer_for_perfect_square_l670_670235

theorem smallest_integer_for_perfect_square :
  let y := 2^5 * 3^5 * 4^5 * 5^5 * 6^4 * 7^3 * 8^3 * 9^2
  ∃ z : ℕ, z = 70 ∧ (∃ k : ℕ, y * z = k^2) :=
by
  sorry

end smallest_integer_for_perfect_square_l670_670235


namespace maximum_area_of_triangle_l670_670740

theorem maximum_area_of_triangle :
  ∃ (b c : ℝ), (a = 2) ∧ (A = 60 * Real.pi / 180) ∧
  (∀ S : ℝ, S = (1/2) * b * c * Real.sin A → S ≤ Real.sqrt 3) :=
by sorry

end maximum_area_of_triangle_l670_670740


namespace sum_first_100_terms_l670_670833

noncomputable def a_n (n : ℕ) : ℕ := 1 + (n - 1)
noncomputable def seq_term (n : ℕ) : ℚ := (1 : ℚ) / (a_n n * a_n (n + 1))

theorem sum_first_100_terms : 
  let sum_n := (finset.range 100).sum (λ n, seq_term n.succ)
  sum_n = 100 / 101 :=
  by
    let sum_partial := (finset.range 100).sum (λ n, (1 : ℚ) / (n.succ * (n.succ + 1)))
    have h_simplify : sum_partial = 100 / 101 := sorry
    exact h_simplify

end sum_first_100_terms_l670_670833


namespace functional_equation_true_l670_670120

noncomputable def f : ℝ → ℝ := sorry

axiom f_defined (x : ℝ) : f x > 0
axiom f_property (a b : ℝ) : f a * f b = f (a + b)

theorem functional_equation_true :
  (f 0 = 1) ∧ 
  (∀ a, f (-a) = 1 / f a) ∧ 
  (∀ a, f a = (f (4 * a)) ^ (1 / 4)) ∧ 
  (∀ a, f (a^2) = (f a)^2) :=
by {
  sorry
}

end functional_equation_true_l670_670120


namespace sin_cos_power_equality_l670_670861

theorem sin_cos_power_equality (θ : ℝ) (h : cos (2 * θ) = 1 / 4) : (sin θ) ^ 6 + (cos θ) ^ 6 = 19 / 64 :=
by
  sorry

end sin_cos_power_equality_l670_670861


namespace age_of_son_l670_670970

theorem age_of_son (S F : ℕ) (h1 : F = S + 28) (h2 : F + 2 = 2 * (S + 2)) : S = 26 := 
by
  -- skip the proof
  sorry

end age_of_son_l670_670970


namespace shaded_rectangle_area_l670_670834

-- Define the square PQRS and its properties
def is_square (s : ℝ) := ∃ (PQ QR RS SP : ℝ), PQ = s ∧ QR = s ∧ RS = s ∧ SP = s

-- Define the conditions for the side lengths and segments
def side_length := 11
def top_left_height := 6
def top_right_height := 2
def width_bottom_right := 11 - 10
def width_top_right := 8

-- Calculate necessary dimensions
def shaded_rectangle_height := top_left_height - top_right_height
def shaded_rectangle_width := width_top_right - width_bottom_right

-- Proof statement
theorem shaded_rectangle_area (s : ℝ) (h1 : is_square s)
  (h2 : s = side_length)
  (h3 : shaded_rectangle_height = 4)
  (h4 : shaded_rectangle_width = 7) :
  4 * 7 = 28 := by
  sorry

end shaded_rectangle_area_l670_670834


namespace ann_boxes_less_than_n_l670_670883

-- Define the total number of boxes n
def n : ℕ := 12

-- Define the number of boxes Mark sold
def mark_sold : ℕ := n - 11

-- Define a condition on the number of boxes Ann sold
def ann_sold (A : ℕ) : Prop := 1 ≤ A ∧ A < n - mark_sold

-- The statement to prove
theorem ann_boxes_less_than_n : ∃ A : ℕ, ann_sold A ∧ n - A = 2 :=
by
  sorry

end ann_boxes_less_than_n_l670_670883


namespace find_difference_l670_670990

-- Define the given number
def givenNumber : ℝ := 640

-- Define a function to calculate percentage
def percentage (p : ℝ) (n : ℝ) : ℝ := (p / 100) * n

-- Define the percentages and numbers
def percent20 : ℝ := 20
def percent50 : ℝ := 50
def num650 : ℝ := 650

-- Define the expected amount of difference
def expectedDifference : ℝ := 190

-- The main theorem
theorem find_difference : 
  let diff := (percentage percent50 givenNumber) - (percentage percent20 num650) in
  diff = expectedDifference :=
by
  sorry

end find_difference_l670_670990


namespace horses_lcm_l670_670162

theorem horses_lcm :
  let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
  let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
  let time_T := lcm_six
  lcm_six = 420 ∧ (Nat.digits 10 time_T).sum = 6 := by
    let horse_times := [2, 3, 4, 5, 6, 7, 8, 9]
    let lcm_six := Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 7))))
    let time_T := lcm_six
    have h1 : lcm_six = 420 := sorry
    have h2 : (Nat.digits 10 time_T).sum = 6 := sorry
    exact ⟨h1, h2⟩

end horses_lcm_l670_670162


namespace infinitely_many_f_eq_n_l670_670731

open Nat

-- Definitions of functions and conditions
def f : ℕ+ → ℕ+
def g : ℕ+ → ℕ+

axiom surjective_g : ∀ (m : ℕ+), ∃ (n : ℕ+), g(n) = m
axiom equation_f_g : ∀ (n : ℕ+), 2 * (f(n))^2 = n^2 + (g(n))^2
axiom bound_f_n : ∀ (n : ℕ+), |f(n).val - n.val| ≤ 2005 * ⌊(n.val.to_real)^(1 / 2)⌋

-- Statement to prove: f(n) = n for infinitely many n
theorem infinitely_many_f_eq_n : ∃ᶠ (n : ℕ+) in at_top, f(n) = n := sorry

end infinitely_many_f_eq_n_l670_670731


namespace profit_without_discounts_is_46_19_percent_l670_670237

noncomputable def costPrice : ℝ := 100
noncomputable def productionFee : ℝ := 0.10 * costPrice
noncomputable def totalCostPrice : ℝ := costPrice + productionFee
noncomputable def sellingPriceWithDiscounts : ℝ := totalCostPrice * 1.25
noncomputable def sellingPrice : ℝ := sellingPriceWithDiscounts / (0.90 * 0.95)

def profitWithoutDiscounts : ℝ := sellingPrice - totalCostPrice
def profitPercentageWithoutDiscounts : ℝ := (profitWithoutDiscounts / totalCostPrice) * 100

theorem profit_without_discounts_is_46_19_percent :
  profitPercentageWithoutDiscounts ≈ 46.19 :=
by
  sorry

end profit_without_discounts_is_46_19_percent_l670_670237


namespace exists_subsets_union_eq_X_l670_670061

theorem exists_subsets_union_eq_X
  (n : ℕ) (hn : n > 6)
  (X : Finset α) (hX : X.card = n)
  (A : Finset (Finset α))
  (hA : ∀ (a : Finset α), a ∈ A → a.card = 5)
  (m : ℕ) (hm' : m = A.card)
  (hm : m > (n * (n - 1) * (n - 2) * (n - 3) * (4 * n - 15)) / 600) :
  ∃ (B : Finset (Finset α)), B.card = 6 ∧ B ⊆ A ∧ (B.bUnion id) = X :=
begin
  sorry
end

end exists_subsets_union_eq_X_l670_670061


namespace perfect_squares_100_to_400_l670_670806

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l670_670806


namespace actual_distance_is_correct_l670_670888

def scale := 6000000
def map_distance := 5 -- in cm

def actual_distance := map_distance * scale / 100000 -- conversion factor from cm to km

theorem actual_distance_is_correct :
  actual_distance = 300 :=
by
  simp [actual_distance, map_distance, scale]
  exact sorry

end actual_distance_is_correct_l670_670888


namespace angle_PBA_eq_angle_PCA_l670_670058

-- Definitions of the points and properties in the acute triangle ABC
variables (A B C D E P : Point)
variables [Triangle ABC]
variables (acuteABC : Triangle.isAcute ABC)
variables (ltAB_AC : AB.toSegment.length < AC.toSegment.length)
variables (pointsDEonBC : D ∈ BC ∧ E ∈ BC ∧ B ≠ C)
variables (BD_CE : BD.toSegment.length = CE.toSegment.length)
variables (D_between_BE : B ≠ E ∧ B ≠ D ∧ D ≠ E ∧ D ∉ Line.singleton B ∧ B ∈ Line BC ∧ D ∈ Line BC ∧ E ∈ Line BC)
variables (P_inside_ABC : InsideTriangle P ABC)
variables (PD_parallel_AE : P ∈ Line.parallelThrough D (Line AE))
variables (angle_PAB_EAC : Angle.of P A B = Angle.of E A C)

-- Goal: Prove that ∠PBA = ∠PCA
theorem angle_PBA_eq_angle_PCA : Angle.of P B A = Angle.of P C A :=
begin
  sorry
end

end angle_PBA_eq_angle_PCA_l670_670058


namespace symmetric_circle_l670_670042

-- Define given circle equation
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * x - 8 * y + 12 = 0

-- Define the line of symmetry
def line_equation (x y : ℝ) : Prop :=
  x + 2 * y - 5 = 0

-- Define the symmetric circle equation we need to prove
def symm_circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 = 8

-- Lean 4 theorem statement
theorem symmetric_circle (x y : ℝ) :
  (∃ a b : ℝ, circle_equation 2 4 ∧ line_equation a b ∧ (a, b) = (0, 0)) →
  symm_circle_equation x y :=
by sorry

end symmetric_circle_l670_670042


namespace rationalize_denominator_sum_l670_670109

theorem rationalize_denominator_sum :
  let A := -4
  let B := 7
  let C := 3
  let D := 13
  let E := 1
  A + B + C + D + E = 20 := by
    sorry

end rationalize_denominator_sum_l670_670109


namespace point_on_line_l670_670931

theorem point_on_line : ∀ (t : ℤ), 
  (∃ m : ℤ, (6 - 2) * m = 20 - 8 ∧ (10 - 6) * m = 32 - 20) →
  (∃ b : ℤ, 8 - 2 * m = b) →
  t = m * 35 + b → t = 107 :=
by
  sorry

end point_on_line_l670_670931


namespace hyperbola_asymptote_l670_670771

theorem hyperbola_asymptote (a : ℝ) (h : a > 0)
  (has_asymptote : ∀ x : ℝ, abs (9 / a * x) = abs (3 * x))
  : a = 3 :=
sorry

end hyperbola_asymptote_l670_670771


namespace shirt_cost_l670_670576

theorem shirt_cost (J S : ℕ) 
  (h₁ : 3 * J + 2 * S = 69) 
  (h₂ : 2 * J + 3 * S = 61) :
  S = 9 :=
by 
  sorry

end shirt_cost_l670_670576


namespace none_of_equalities_are_correct_l670_670660

theorem none_of_equalities_are_correct :
  ¬ (3 * 10^6 + 5 * 10^2 = 8 * 10^8) ∧
  ¬ (2^3 + 2^(-3) = 2^0) ∧
  ¬ (5 * 8 + 7 = 75) ∧
  ¬ (5 + 5 / 5 = 2) :=
by
  sorry

end none_of_equalities_are_correct_l670_670660


namespace apple_juice_production_l670_670493

noncomputable def apple_usage 
  (total_apples : ℝ) 
  (mixed_percentage : ℝ) 
  (juice_percentage : ℝ) 
  (sold_fresh_percentage : ℝ) : ℝ := 
  let mixed_apples := total_apples * mixed_percentage / 100
  let remainder_apples := total_apples - mixed_apples
  let juice_apples := remainder_apples * juice_percentage / 100
  juice_apples

theorem apple_juice_production :
  apple_usage 6 20 60 40 = 2.9 := 
by
  sorry

end apple_juice_production_l670_670493


namespace sale_in_fifth_month_l670_670619

theorem sale_in_fifth_month
  (s1 s2 s3 s4 s6 : ℕ)
  (avg : ℕ)
  (h1 : s1 = 5435)
  (h2 : s2 = 5927)
  (h3 : s3 = 5855)
  (h4 : s4 = 6230)
  (h6 : s6 = 3991)
  (hav : avg = 5500) :
  ∃ s5 : ℕ, s1 + s2 + s3 + s4 + s5 + s6 = avg * 6 ∧ s5 = 5562 := 
by
  sorry

end sale_in_fifth_month_l670_670619


namespace proof_problem_l670_670864

def f (x : ℤ) : ℤ := 3 * x + 5
def g (x : ℤ) : ℤ := 4 * x - 3

theorem proof_problem : 
  (f (g (f (g 3)))) / (g (f (g (f 3)))) = (380 / 653) := 
  by 
    sorry

end proof_problem_l670_670864


namespace inequality_proof_l670_670894

theorem inequality_proof (α β : ℝ) (h1 : cos α ^ 2 + sin α ^ 2 = 1) (h2 : cos β ^ 2 + sin β ^ 2 = 1) :
  1 / (cos α ^ 2) + 1 / (sin α ^ 2 * sin β ^ 2 * cos β ^ 2) ≥ 9 :=
by sorry

end inequality_proof_l670_670894


namespace original_price_of_car_l670_670178

-- Define the original price of the car based on the condition of the problem
def original_price (spent : ℝ) (percentage : ℝ) : ℝ := spent / percentage

-- Given conditions
def venny_spent : ℝ := 15000
def percentage_of_original : ℝ := 0.40

-- Statement to be proved
theorem original_price_of_car : original_price venny_spent percentage_of_original = 37500 := by
  sorry

end original_price_of_car_l670_670178


namespace possible_N_l670_670424

/-- 
  Let N be an integer with N ≥ 3, and let a₀, a₁, ..., a_(N-1) be pairwise distinct reals such that 
  aᵢ ≥ a_(2i mod N) for all i. Prove that N must be a power of 2.
-/
theorem possible_N (N : ℕ) (hN : N ≥ 3) (a : Fin N → ℝ) (h_distinct: Function.Injective a) 
  (h_condition : ∀ i : Fin N, a i ≥ a (⟨(2 * i) % N, sorry⟩)) 
  : ∃ k : ℕ, N = 2^k := 
sorry

end possible_N_l670_670424


namespace simplify_expression_l670_670906

theorem simplify_expression (i : ℂ) (h : i^2 = -1) : 3 * (2 - i) + i * (3 + 2 * i) = 4 :=
by
  sorry

end simplify_expression_l670_670906


namespace remaining_distance_is_one_l670_670051

def total_distance_to_grandma : ℕ := 78
def initial_distance_traveled : ℕ := 35
def bakery_detour : ℕ := 7
def pie_distance : ℕ := 18
def gift_detour : ℕ := 3
def next_travel_distance : ℕ := 12
def scenic_detour : ℕ := 2

def total_distance_traveled : ℕ :=
  initial_distance_traveled + bakery_detour + pie_distance + gift_detour + next_travel_distance + scenic_detour

theorem remaining_distance_is_one :
  total_distance_to_grandma - total_distance_traveled = 1 := by
  sorry

end remaining_distance_is_one_l670_670051


namespace calculate_x_l670_670260

theorem calculate_x :
  let x := 225 + 2 * 15 * 5 + 25 in
  225 = 15^2 → 25 = 5^2 → x = 400 :=
by
  intros h1 h2
  rw [h1, h2]
  have : (15 + 5) = 20 := by norm_num
  calc
    225 + 2 * 15 * 5 + 25
        = 15^2 + 2 * 15 * 5 + 5^2  : by rw [h1, h2]
    ... = (15 + 5)^2                : by ring
    ... = 20^2                      : by rw this
    ... = 400                       : by norm_num

end calculate_x_l670_670260


namespace problem1_problem2_problem3_l670_670775

-- Problem 1
theorem problem1 (x : ℝ) (h1 : x ∈ set.Icc (-1 : ℝ) (1 : ℝ)) : 
  set.Icc (1 / 4 : ℝ) (9 / 4 : ℝ) = range (λ x, x^2 - 2 * (real.sin (real.pi / 6)) * x + 1 / 4) :=
sorry

-- Problem 2
theorem problem2 (θ : ℝ) (h2 : (θ ∈ set.Icc (real.pi / 6 + 2*real.pi*k) (5*real.pi/6 + 2*real.pi*k)) ∨
                         (θ ∈ set.Icc (7*real.pi/6 + 2*real.pi*k) (11*real.pi/6 + 2*real.pi*k))):
  monotone_on (λ x, x^2 - 2 * real.sin θ * x + 1 / 4) (set.Icc (-1/2 : ℝ) (1/2 : ℝ)) :=
sorry

-- Problem 3
theorem problem3 (t : ℝ) (x1 x2 : ℝ) (h3 : x1 ∈ set.Icc (2 : ℝ) (3 : ℝ)) (h4 : x2 ∈ set.Icc (2 : ℝ) (3 : ℝ)) :
  (abs ((x1^2 - 2 * real.sin θ * x1 + 1 / 4) - (x2^2 - 2 * real.sin θ * x2 + 1 / 4)) ≤ 2 * real.sin θ * t^2 + 8 * t + 5) ↔ 
  t ∈ set.Icc ((2 - real.sqrt 3) / 2) ((2 + real.sqrt 3) / 2) :=
sorry

end problem1_problem2_problem3_l670_670775


namespace sqrt_subtraction_l670_670561

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l670_670561


namespace tangent_line_equation_range_of_a_l670_670764

noncomputable def f (x : ℝ) : ℝ := exp x - 2 * x - 1
noncomputable def g (a x : ℝ) : ℝ := a * f x + (1 - a) * exp x

theorem tangent_line_equation : 
  ∀ x : ℝ, y = f x → (0 : ℝ) (f 0) → y = -x :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, g a x = 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂) →
  a ∈ set.Ioi (sqrt (exp 1) / 2) :=
by
  sorry

end tangent_line_equation_range_of_a_l670_670764


namespace evaluate_heartsuit_l670_670272

-- Define the given operation
def heartsuit (x y : ℝ) : ℝ := abs (x - y)

-- State the proof problem in Lean
theorem evaluate_heartsuit (a b : ℝ) (h_a : a = 3) (h_b : b = -1) :
  heartsuit (heartsuit a b) (heartsuit (2 * a) (2 * b)) = 4 :=
by
  -- acknowledging that it's correct without providing the solution steps
  sorry

end evaluate_heartsuit_l670_670272


namespace basic_computer_price_l670_670976

variables (C P : ℕ)

theorem basic_computer_price (h1 : C + P = 2500)
                            (h2 : C + 500 + P = 6 * P) : C = 2000 :=
by
  sorry

end basic_computer_price_l670_670976


namespace number_of_square_tiles_l670_670994

/-- A box contains a collection of triangular tiles, square tiles, and pentagonal tiles. 
    There are a total of 30 tiles in the box and a total of 100 edges. 
    We need to show that the number of square tiles is 10. --/
theorem number_of_square_tiles (a b c : ℕ) (h1 : a + b + c = 30) (h2 : 3 * a + 4 * b + 5 * c = 100) : b = 10 := by
  sorry

end number_of_square_tiles_l670_670994


namespace ratio_of_tins_to_bags_l670_670214

def ounces_per_pound : ℕ := 16
def bag_of_chips_weight_oz : ℕ := 20
def tin_of_cookies_weight_oz : ℕ := 9
def number_of_bags_of_chips : ℕ := 6
def total_weight_lbs : ℕ := 21

theorem ratio_of_tins_to_bags :
  let total_weight_oz := total_weight_lbs * ounces_per_pound,
      weight_of_chips := number_of_bags_of_chips * bag_of_chips_weight_oz,
      weight_of_cookies := total_weight_oz - weight_of_chips,
      number_of_tins_of_cookies := weight_of_cookies / tin_of_cookies_weight_oz in
  number_of_tins_of_cookies / number_of_bags_of_chips = 4 := by
  sorry

end ratio_of_tins_to_bags_l670_670214


namespace imaginary_part_of_i_mul_1_minus_i_l670_670508

theorem imaginary_part_of_i_mul_1_minus_i : complex.im (i * (1 - i)) = 1 :=
by
  sorry

end imaginary_part_of_i_mul_1_minus_i_l670_670508


namespace linear_iff_m_eq_neg1_l670_670366

variables {m : ℝ} (x : ℝ)

def linear_form := (m - 1) * x ^ (abs m) + 2

theorem linear_iff_m_eq_neg1 : (∃ k b, linear_form x = k * x + b ∧ k ≠ 0) ↔ m = -1 := by sorry

end linear_iff_m_eq_neg1_l670_670366


namespace bar_graph_proportion_correct_l670_670279

def white : ℚ := 1/2
def black : ℚ := 1/4
def gray : ℚ := 1/8
def light_gray : ℚ := 1/16

theorem bar_graph_proportion_correct :
  (white = 1 / 2) ∧
  (black = white / 2) ∧
  (gray = black / 2) ∧
  (light_gray = gray / 2) →
  (white = 1 / 2) ∧
  (black = 1 / 4) ∧
  (gray = 1 / 8) ∧
  (light_gray = 1 / 16) :=
by
  intros
  sorry

end bar_graph_proportion_correct_l670_670279


namespace geometric_sequence_a6_l670_670395

variable {a : ℕ → ℝ} (h_geo : ∀ n, a (n+1) / a n = a (n+2) / a (n+1))

theorem geometric_sequence_a6 (h5 : a 5 = 2) (h7 : a 7 = 8) : a 6 = 4 ∨ a 6 = -4 :=
by
  sorry

end geometric_sequence_a6_l670_670395


namespace area_of_rectangle_l670_670234

-- Definitions given in the conditions
def width : ℚ := 81 / 4
def height : ℚ := 148 / 9
def expected_area : ℚ := 333

-- Problem statement
theorem area_of_rectangle : width * height = expected_area := by
  let w := width
  let h := height
  let area := w * h
  have numerator : ℚ := 81 * 148
  have denominator : ℚ := 4 * 9
  have simplification : numerator / denominator = expected_area := by
    calc
      numerator / denominator
           = 11988 / 36 : by rw [rat.div_num_denom]
         ... = 333 : by norm_cast
  exact simplification

end area_of_rectangle_l670_670234


namespace estimates_total_fish_l670_670989

theorem estimates_total_fish (m n k t : ℕ) (h1 : m = 120) (h2 : n = 100) (h3 : k = 10) (h4 : k * t = m * n) : t = 1200 :=
by
  have h : k * t = 120 * 100 := by simp [h1, h2, h3, h4]
  calc
    t = (120 * 100) / 10 := by rw [←h4, h3]; exact eq_div_of_mul_eq the h4.symm sorry
       ... = 1200       := by norm_num

end estimates_total_fish_l670_670989


namespace cubic_sum_difference_l670_670269

theorem cubic_sum_difference :
  let n := 50
  let positive_cubic_sum := (n * (n + 1) / 2) ^ 2
  let negative_cubic_sum := -positive_cubic_sum
  positive_cubic_sum - negative_cubic_sum = 3251250 :=
by
  let n := 50
  let positive_cubic_sum := (n * (n + 1) / 2) ^ 2
  let negative_cubic_sum := -positive_cubic_sum
  have h : positive_cubic_sum - negative_cubic_sum = 3251250 := sorry
  exact h

end cubic_sum_difference_l670_670269


namespace sector_radian_measure_l670_670749

theorem sector_radian_measure {r l : ℝ} 
  (h1 : 2 * r + l = 12) 
  (h2 : (1/2) * l * r = 8) : 
  (l / r = 1) ∨ (l / r = 4) :=
sorry

end sector_radian_measure_l670_670749


namespace number_of_valid_subsets_B_l670_670086

namespace Proof

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3}
def targetString : Set ℕ := {1, 3, 6}
def binaryRepresentation (s : Set ℕ) : Set ℕ := s

theorem number_of_valid_subsets_B : 
  (∃ B : Set ℕ, B ⊆ U ∧ binaryRepresentation (A ∪ B) = targetString) → 
  ({B : Set ℕ // B ⊆ U ∧ binaryRepresentation (A ∪ B) = targetString}.toFinset.card = 4) :=
sorry

end Proof

end number_of_valid_subsets_B_l670_670086


namespace greatest_prime_factor_of_4_pow_6_plus_8_pow_5_l670_670550

def greatest_prime_factor (n : ℕ) : ℕ :=
  if h : ∃ p, nat.prime p ∧ p ∣ n then
    nat.find h
  else
    1

theorem greatest_prime_factor_of_4_pow_6_plus_8_pow_5 : 
  greatest_prime_factor (4^6 + 8^5) = 3 :=
by 
  -- Sorry to skip actual proof steps
  sorry

end greatest_prime_factor_of_4_pow_6_plus_8_pow_5_l670_670550


namespace cristina_running_pace_l670_670455

theorem cristina_running_pace
  (nicky_pace : ℝ) (nicky_headstart : ℝ) (time_nicky_run : ℝ) 
  (distance_nicky_run : ℝ) (time_cristina_catch : ℝ) :
  (nicky_pace = 3) →
  (nicky_headstart = 12) →
  (time_nicky_run = 30) →
  (distance_nicky_run = nicky_pace * time_nicky_run) →
  (time_cristina_catch = time_nicky_run - nicky_headstart) →
  (cristina_pace : ℝ) →
  (cristina_pace = distance_nicky_run / time_cristina_catch) →
  cristina_pace = 5 :=
by
  sorry

end cristina_running_pace_l670_670455


namespace Maddie_bought_two_white_packs_l670_670089

theorem Maddie_bought_two_white_packs 
  (W : ℕ)
  (total_cost : ℕ)
  (cost_per_shirt : ℕ)
  (white_pack_size : ℕ)
  (blue_pack_size : ℕ)
  (blue_packs : ℕ)
  (cost_per_white_pack : ℕ)
  (cost_per_blue_pack : ℕ) :
  total_cost = 66 ∧ cost_per_shirt = 3 ∧ white_pack_size = 5 ∧ blue_pack_size = 3 ∧ blue_packs = 4 ∧ cost_per_white_pack = white_pack_size * cost_per_shirt ∧ cost_per_blue_pack = blue_pack_size * cost_per_shirt ∧ 3 * (white_pack_size * W + blue_pack_size * blue_packs) = total_cost → W = 2 :=
by
  sorry

end Maddie_bought_two_white_packs_l670_670089


namespace min_value_expression_l670_670425

theorem min_value_expression (α β : ℝ) (hα : α ≠ 0) (hβ : abs β = 1) : 
  ∃ m, m = min ((| (β + α) / (1 + α * β) |)) ∧ m = 1 :=
by
  sorry

end min_value_expression_l670_670425


namespace residue_calculation_l670_670704

theorem residue_calculation :
  (195 * 13 - 25 * 8 + 5) % 17 = 3 :=
by
  have h1 : 195 % 17 = 6 := by norm_num,
  have h2 : (195 * 13) % 17 = 11 := by norm_num,
  have h3 : 25 % 17 = 8 := by norm_num,
  have h4 : (25 * 8) % 17 = 13 := by norm_num,
  have h5 : 5 % 17 = 5 := by norm_num,
  sorry

end residue_calculation_l670_670704


namespace bicycle_frame_stability_l670_670993

-- Definitions 
def triangle (A B C : Type) := (A → B → C → Prop)
def stable (t : triangle) := ∀ A B C, t A B C →  ∃ a b c, a = b ∧ a = c ∧ b = c
def bicycle_frame (A B C : Type) := triangle A B C

-- Theorem stating that the bicycle frame's triangular shape is chosen for stability
theorem bicycle_frame_stability (A B C : Type) (t : triangle A B C) (hf : bicycle_frame A B C) :
  stable t :=
sorry

end bicycle_frame_stability_l670_670993


namespace num_possible_n_l670_670267

theorem num_possible_n (n : ℕ) : (∃ a b c : ℕ, 9 * a + 99 * b + 999 * c = 5000 ∧ n = a + 2 * b + 3 * c) ↔ n ∈ {x | x = a + 2 * b + 3 * c ∧ 0 ≤ 9 * (b + 12 * c) ∧ 9 * (b + 12 * c) ≤ 555} :=
sorry

end num_possible_n_l670_670267


namespace perfect_squares_in_range_100_400_l670_670787

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l670_670787


namespace extreme_value_range_of_a_l670_670336

noncomputable def f (x : ℝ) (a : ℝ) := 2 * x * Real.log x - a

theorem extreme_value (a : ℝ) :
  ∃ c : ℝ, c = f (Real.exp (-1)) a :=
by
  use - (2 / Real.exp 1) - a
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → e^x * f x a - x^2 - 1 ≥ 0) → a ≤ - (2 / Real.exp 1) :=
by
  intro h
  -- Define g(x) and analyze it
  let g := λ x, (x^2 + 1) / Real.exp x - 2 * x * Real.log x
  have : ∀ x ≥ 1, g x ≥ -a,
  { intro x hx,
    specialize h x hx,
    -- rewriting using the definition of f
    dsimp [f] at h,
    exact h },
  -- concluding the expected upper bound on a
  have g1 := g 1,
  specialize this 1 le_rfl,
  linarith [this]
  sorry

end extreme_value_range_of_a_l670_670336


namespace division_of_fractions_l670_670949

theorem division_of_fractions : (5 / 6) / (1 + 3 / 9) = 5 / 8 := by
  sorry

end division_of_fractions_l670_670949


namespace find_sum_A_B_C_l670_670921

theorem find_sum_A_B_C (A B C : ℤ)
  (h1 : ∀ x > 4, (x^2 : ℝ) / (A * x^2 + B * x + C) > 0.4)
  (h2 : A * (-2)^2 + B * (-2) + C = 0)
  (h3 : A * (3)^2 + B * (3) + C = 0)
  (h4 : 0.4 < 1 / (A : ℝ) ∧ 1 / (A : ℝ) < 1) :
  A + B + C = -12 :=
by
  sorry

end find_sum_A_B_C_l670_670921


namespace exists_right_triangle_in_trihedral_angle_l670_670016

theorem exists_right_triangle_in_trihedral_angle (P A B C : Point)
  (h1 : is_trihedral_angle P A B C)
  (h2 : acute_dihedral_angle P A B) :
  ∃ A_1 B_1 C_1 : Point,
    intersects_plane A_1 B_1 C_1 (trihedral_angle P A B C) ∧
    is_right_triangle A_1 B_1 C_1 :=
sorry

end exists_right_triangle_in_trihedral_angle_l670_670016


namespace range_of_a_l670_670755

-- Definitions according to the problem's conditions
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + 1
def f_derivative (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x
def x0 (a : ℝ) : ℝ := -a / 3

-- The formal statement
theorem range_of_a (a : ℝ) : (∃ x0, x0 > 0 ∧ f a x0 = 0 ∧ ∃ x1, f a x1 = 0 ∧ ∃ x2, f a x2 = 0 ∧ x1 ≠ x2 ∧ x1 ≠ x0 ∧ x2 ≠ x0) → a < -3 * real.sqrt3 2 / 2 := sorry

end range_of_a_l670_670755


namespace large_box_storage_l670_670239

theorem large_box_storage
  (small_box_vol : ℝ)
  (small_box_paperclips : ℝ)
  (large_box_vol : ℝ)
  (large_box_staples : ℝ)
  (large_box_div_vol : ℝ)
  (half_vol : large_box_div_vol = large_box_vol / 2)
  (paperclips_per_cm3 : small_box_paperclips = 2.5 * small_box_vol)
  (staples_per_cm3 : large_box_staples = 1.25 * large_box_vol)
  (half_paperclips : 90 = 2.5 * half_vol)
  (half_staples : 45 = 1.25 * half_vol)
: 
  (large_box_vol = 72) ∧
  (small_box_vol = 24) ∧
  (small_box_paperclips = 60) ∧
  (large_box_staples = 90) →
  (large_box_div_vol / 2 = 36) ∧
  (90 = 2.5 * 36) ∧
  (45 = 1.25 * 36) := 
by sorry

end large_box_storage_l670_670239


namespace exactly_three_distinct_solutions_l670_670438

theorem exactly_three_distinct_solutions (a : ℝ) : 
  (∃ (sols : set ℝ), sols = {x | abs (abs (x - a) - a) = 2} ∧ sols.card = 3) → a = 2 :=
by 
  sorry

end exactly_three_distinct_solutions_l670_670438


namespace pass_each_other_distance_l670_670677

open Real

def elliot_head_start : ℝ := 2 / 15
def elliot_up_speed : ℝ := 12 -- km/hr
def elliot_down_speed : ℝ := 18 -- km/hr
def emily_up_speed : ℝ := 14 -- km/hr
def emily_down_speed : ℝ := 20 -- km/hr
def hill_distance : ℝ := 6 -- km

-- Defining the statement to prove
theorem pass_each_other_distance :
  let t_meet : ℝ := 17 / 32 in
  let elliot_position := if t_meet ≤ 1 / 2 then elliot_up_speed * t_meet 
                         else 6 - elliot_down_speed * (t_meet - 1 / 2)
  let emily_position := emily_up_speed * (t_meet - elliot_head_start) in
  elliot_position = emily_position →
  6 - elliot_position = 169 / 48 :=
sorry

end pass_each_other_distance_l670_670677


namespace good_subsets_count_l670_670078

def is_good_subset (A : Finset ℕ) (S : Finset ℕ) : Prop :=
  ∃ a1 a2 a3, a1 ∈ S ∧ a2 ∈ S ∧ a3 ∈ S ∧ a1 < a2 ∧ a2 < a3 ∧ a3 ≥ a2 + 3 ∧ a2 ≥ a1 + 3

noncomputable def number_of_good_subsets (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).powerset.filter (λ S, S.card = 3 ∧ is_good_subset (Finset.range (n + 1)) S).card

theorem good_subsets_count (n : ℕ) (h : n ≥ 6) : number_of_good_subsets n = Nat.choose (n + 2) 3 :=
by sorry

end good_subsets_count_l670_670078


namespace no_minimum_of_f_over_M_l670_670596

/-- Define the domain M for the function y = log(3 - 4x + x^2) -/
def domain_M (x : ℝ) : Prop := (x > 3 ∨ x < 1)

/-- Define the function f(x) = 2x + 2 - 3 * 4^x -/
noncomputable def f (x : ℝ) : ℝ := 2 * x + 2 - 3 * 4^x

/-- The theorem statement:
    Prove that f(x) does not have a minimum value for x in the domain M -/
theorem no_minimum_of_f_over_M : ¬ ∃ x ∈ {x | domain_M x}, ∀ y ∈ {x | domain_M x}, f x ≤ f y := sorry

end no_minimum_of_f_over_M_l670_670596


namespace domain_of_f_l670_670133

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 3)) / Real.sqrt (1 - 2^x)

theorem domain_of_f :
  {x : ℝ | -3 < x ∧ x < 0} = {x : ℝ | ∃ (y : ℝ), f y = x} := 
begin
  sorry
end

end domain_of_f_l670_670133


namespace larger_number_ratio_l670_670947

theorem larger_number_ratio (x : ℕ) (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 8 * x) 
(h3 : (a - 24) * 9 = (b - 24) * 4) : b = 192 :=
sorry

end larger_number_ratio_l670_670947


namespace lenny_pens_left_l670_670419

noncomputable def pensLeft (boxes : ℕ) (pensPerBox : ℕ) (pctFriends : ℝ) (fracClassmates : ℝ) (fracCoworkers : ℝ) : ℕ :=
  let totalPens := boxes * pensPerBox
  let pensToFriends := (pctFriends * totalPens).toInt
  let pensAfterFriends := totalPens - pensToFriends
  let pensToClassmates := (pensAfterFriends * fracClassmates).toInt
  let pensAfterClassmates := pensAfterFriends - pensToClassmates
  let pensToCoworkers := (pensAfterClassmates * fracCoworkers).toInt
  let pensRemaining := pensAfterClassmates - pensToCoworkers
  pensRemaining

theorem lenny_pens_left :
  pensLeft 50 12 0.35 (1 / 3) (1 / 7) = 223 :=
by
  sorry

end lenny_pens_left_l670_670419


namespace number_of_students_in_each_group_l670_670166

theorem number_of_students_in_each_group
  (num_flags : ℕ)
  (num_groups_initial : ℕ)
  (num_groups_later : ℕ)
  (flag_increase_per_student : ℕ)
  (num_flags = 240)
  (num_groups_initial = 3)
  (num_groups_later = 2)
  (flag_increase_per_student = 4) :
  let num_students_per_group := 10 in
  num_students_per_group = 10 := by
  sorry

end number_of_students_in_each_group_l670_670166


namespace perfect_squares_100_to_400_l670_670804

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l670_670804


namespace no_triangle_formed_l670_670779

def line1 (x y : ℝ) := 2 * x - 3 * y + 1 = 0
def line2 (x y : ℝ) := 4 * x + 3 * y + 5 = 0
def line3 (m : ℝ) (x y : ℝ) := m * x - y - 1 = 0

theorem no_triangle_formed (m : ℝ) :
  (∀ x y, line1 x y → line3 m x y) ∨
  (∀ x y, line2 x y → line3 m x y) ∨
  (∃ x y, line1 x y ∧ line2 x y ∧ line3 m x y) ↔
  (m = -4/3 ∨ m = 2/3 ∨ m = 4/3) :=
sorry -- Proof to be provided

end no_triangle_formed_l670_670779


namespace cylinder_volume_ratio_l670_670375

variable (h r : ℝ)

theorem cylinder_volume_ratio (h r : ℝ) :
  let V_original := π * r^2 * h
  let h_new := 2 * h
  let r_new := 4 * r
  let V_new := π * (r_new)^2 * h_new
  V_new = 32 * V_original :=
by
  sorry

end cylinder_volume_ratio_l670_670375


namespace berry_difference_l670_670280

/-- Define the initial total number of berries on the bush -/
def total_berries : ℕ := 900

/-- Sergey collects 1 out of every 2 berries he picks -/
def sergey_collection_ratio : ℕ := 2

/-- Dima collects 2 out of every 3 berries he picks -/
def dima_collection_ratio : ℕ := 3

/-- Sergey picks berries twice as fast as Dima -/
def sergey_speed_multiplier : ℕ := 2

/-- Prove that the difference between berries collected in Sergey's and Dima's baskets is 100 -/
theorem berry_difference : 
  let total_picked_sergey := (sergey_speed_multiplier * total_berries) / (sergey_speed_multiplier + 1),
      total_picked_dima := total_berries / (sergey_speed_multiplier + 1),
      sergey_basket := total_picked_sergey / sergey_collection_ratio,
      dima_basket := (2 * total_picked_dima) / dima_collection_ratio
  in sergey_basket - dima_basket = 100 :=
by
  sorry -- proof to be completed

end berry_difference_l670_670280


namespace perfect_squares_between_100_and_400_l670_670789

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l670_670789


namespace number_of_discounted_tickets_l670_670090

def total_tickets : ℕ := 10
def full_price_ticket_cost : ℝ := 2.0
def discounted_ticket_cost : ℝ := 1.6
def total_spent : ℝ := 18.40

theorem number_of_discounted_tickets (F D : ℕ) : 
    F + D = total_tickets → 
    full_price_ticket_cost * ↑F + discounted_ticket_cost * ↑D = total_spent → 
    D = 4 :=
by
  intros h1 h2
  sorry

end number_of_discounted_tickets_l670_670090


namespace max_value_ineq_l670_670429

theorem max_value_ineq (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 2) :
  (ab / (a + b)) + (ac / (a + c)) + (bc / (b + c)) ≤ 1 :=
sorry

end max_value_ineq_l670_670429


namespace known_child_is_boy_l670_670225

def child_gender (is_boy : Bool) : Prop := if is_boy then "boy" else "girl"

def probability_of_both_boys (p : ℝ) : Prop := p = 0.5

theorem known_child_is_boy
  (has_two_children : True)
  (one_child_gender_certain : ∃ (g : Bool), child_gender g = "boy" ∨ child_gender g = "girl")
  (both_boys_probability : probability_of_both_boys 0.5) :
  ∃ (g : Bool), child_gender g = "boy" :=
sorry

end known_child_is_boy_l670_670225


namespace positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l670_670877

theorem positive_integers_ab_divides_asq_bsq_implies_a_eq_b
  (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : a * b ∣ a^2 + b^2) : a = b := by
  sorry

end positive_integers_ab_divides_asq_bsq_implies_a_eq_b_l670_670877


namespace result_m_plus_n_l670_670363

-- Definitions from the conditions
def like_terms (e1 e2 : ℕ × ℕ) : Prop :=
  e1.2 = e2.2

-- The main statement to prove
theorem result_m_plus_n (m n : ℕ)
  (h1 : like_terms (m, m + 1) (n - 1, 3)) :
  m + n = 5 :=
begin
  sorry
end

end result_m_plus_n_l670_670363


namespace problem_l670_670325

-- Define the necessary properties and the problem statement.
variable {f : ℝ → ℝ}

-- Define the conditions
variable (h_even : ∀ x, f x = f (-x))
variable (h_inc_neg : ∀ x y, -1 < x → x < 0 → -1 < y → y < 0 → x < y → f x < f y)
variable (h_acute_angles : ∀ (A B C : ℝ), 0 < A → A < π/2 → 0 < B → B < π/2 → 0 < C → C < π/2 → A + B + C = π)

-- State the problem to prove
theorem problem (A B C : ℝ) (h : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) :
  f (sin C) < f (cos B) :=
sorry

end problem_l670_670325


namespace surface_area_of_CXYZ_is_correct_l670_670241

noncomputable def prismHeight : ℝ := 16
noncomputable def sideLength : ℝ := 12
noncomputable def midpoint (x y : ℝ) : ℝ := (x + y) / 2

-- Defining points
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (sideLength, 0)
def C : ℝ × ℝ := (sideLength / 2, -(sideLength * sqrt 3 / 2))
def D : ℝ × ℝ := (sideLength / 2, -(sideLength * sqrt 3 / 2) - prismHeight)

-- Midpoints X, Y, Z
def X : ℝ × ℝ := midpoint A.1 C.1, midpoint A.2 C.2
def Y : ℝ × ℝ := midpoint B.1 C.1, midpoint B.2 C.2
def Z : ℝ × ℝ := midpoint C.1 D.1, midpoint C.2 D.2

-- Function to calculate the surface area of the solid CXYZ
def surfaceAreaCXYZ : ℝ := 
  48 + 9 * sqrt 3 + 3 * sqrt 91 

theorem surface_area_of_CXYZ_is_correct : 
  CXYZ_surface_area = 48 + 9 * sqrt 3 + 3 * sqrt 91 := 
sorry -- proof goes here

end surface_area_of_CXYZ_is_correct_l670_670241


namespace maj_prob_l670_670609

open_locale big_operators
noncomputable def probability_all_majors_chosen : ℚ :=
  let total_outcomes := 3^4 in
  let favorable_outcomes := nat.choose 4 2 * nat.factorial 3 in
  favorable_outcomes / total_outcomes

theorem maj_prob : probability_all_majors_chosen = 4 / 9 := by
  sorry

end maj_prob_l670_670609


namespace altitude_any_positive_real_l670_670248

theorem altitude_any_positive_real (h : ℝ) (b : ℝ) (m : ℝ) : 
  (b = 24) →
  (m = b / 2) →
  ((1 / 2) * b * h = m * h) →
  (h > 0) :=
by
  assume h b m
  assume hb24 hm_eq h_eq
  sorry

end altitude_any_positive_real_l670_670248


namespace p_t_expansion_range_of_m_l670_670330

noncomputable def f (x : ℝ) : ℝ := 2^(x+1)

def g (x : ℝ) : ℝ := (f x + f (-x)) / 2
def h (x : ℝ) : ℝ := (f x - f (-x)) / 2

def t (x : ℝ) : ℝ := h x

def p (t : ℝ) (m : ℝ) : ℝ := t^2 + 2 * m * t + m^2 - m + 1

theorem p_t_expansion (x : ℝ) (m : ℝ) : 
  p (t x) m = (t x)^2 + 2 * m * (t x) + m^2 - m + 1 := 
by sorry

theorem range_of_m (x : ℝ) (h_x_in_1_2 : x ∈ Icc (1 : ℝ) (2 : ℝ)) 
  {m : ℝ} (h_p : p (t x) m >= m^2 - m - 1) : 
  m ≥ -17 / 12 :=
by sorry

end p_t_expansion_range_of_m_l670_670330


namespace part_I_part_II_l670_670332

noncomputable def f (x a : ℝ) := (1/2) * x^2 - x + a * log x

theorem part_I (a : ℝ) : monotonic_on (f x a) (set.Ioi 0) -> a ≥ 1/4 :=
  sorry

theorem part_II (a α x₁ x₂ : ℝ) (hα : 0 < α ∧ α < 2/9) (hx : x₁ < x₂) (hx_domain : 0 < x₁ ∧ x₁ < 1/3) 
    (extreme_pts : f_deriv x₁ a = 0 ∧ f_deriv x₂ a = 0) :
    f(x₁, a) / x₂ > -5/12 - 1/3 * log 3 :=
  sorry

end part_I_part_II_l670_670332


namespace circle_radius_interval_l670_670607

noncomputable def ellipse := {x : ℝ × ℝ | (x.1^2 / 9) + (x.2^2 / 4) = 1}

noncomputable def foci : ℝ × ℝ := (sqrt 5, 0)

def passes_through_foci (r : ℝ) : Prop :=
  ∃ (c : ℝ × ℝ), (c.1^2 + c.2^2 = r^2) ∧ (foci ∈ circle r c)

def exactly_four_points (r : ℝ) : Prop :=
  ∃ (c : ℝ × ℝ), (∀ (p : ℝ × ℝ), p ∈ ellipse → p ∈ circle r c → ∃! (p),

theorem circle_radius_interval : 
  ∃ (a b : ℝ), ∀ (r : ℝ), passes_through_foci r ∧ exactly_four_points r → (sqrt 5 ≤ r ∧ r < b) ∧ a + b = 10 := 
  sorry

end circle_radius_interval_l670_670607


namespace problem_statement_l670_670876

open Real

variable (O H : Point) (x y z R : ℝ)
variable (circumcenter orthocenter : Point → ℝ)

def OH_squared (O H : Point) : ℝ := ℝ.norm (O - H) ^ 2

theorem problem_statement 
  (h1 : R = 5) 
  (h2 : x^2 + y^2 + z^2 = 75) :
  OH_squared O H = 150 := 
sorry

end problem_statement_l670_670876


namespace number_of_observations_l670_670163

theorem number_of_observations (n : ℕ) (h1 : 200 - 6 = 194) (h2 : 200 * n - n * 6 = n * 194) :
  n > 0 :=
by
  sorry

end number_of_observations_l670_670163


namespace remainder_3x_minus_6_divides_P_l670_670955

def P(x : ℝ) : ℝ := 5 * x^8 - 3 * x^7 + 2 * x^6 - 8 * x^4 + 3 * x^3 - 5
def D(x : ℝ) : ℝ := 3 * x - 6

theorem remainder_3x_minus_6_divides_P :
  P 2 = 915 :=
by
  sorry

end remainder_3x_minus_6_divides_P_l670_670955


namespace angle_condition_proof_l670_670405

-- Defining the given triangle and angles
variables (A B C E : Type) [triangleABC : Triangle A B C]

-- Defining the given conditions
def angleBAC : ℝ := 30
def three_times_AE_eq_EC (AE EC : ℝ) := 3 * AE = EC
def angleEBC : ℝ := 45

-- Defining what needs to be proved
def angleACB : ℝ := 105

-- Statement of the theorem
theorem angle_condition_proof (triangleABC : Triangle A B C)
                              (E_on_AC : E ∈ (segment A C))
                              (h1 : angleBAC = 30)
                              (h2 : three_times_AE_eq_EC (dist A E) (dist E C))
                              (h3 : ∠angleEBC = 45)
                              : ∠angleACB = 105 := 
sorry

end angle_condition_proof_l670_670405


namespace identify_odd_and_increasing_function_l670_670567

theorem identify_odd_and_increasing_function :
  ∀ (f : ℝ → ℝ),
    (f = (λ x, sin x) ∨ f = (λ x, x * abs x) ∨ f = (λ x, x ^ (1/2)) ∨ f = (λ x, x - 1/x)) →
    odd_function f ∧ increasing_function f.domain →
    f = (λ x, x * abs x) :=
sorry

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def increasing_function (S : Set ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ S, x < y → f x < f y

def FunctionOne : ℝ → ℝ := λ x, sin x
def FunctionTwo : ℝ → ℝ := λ x, x * abs x
def FunctionThree : ℝ → ℝ := λ x, x ^ (1/2)
def FunctionFour : ℝ → ℝ := λ x, x - (1 / x)

end identify_odd_and_increasing_function_l670_670567


namespace max_min_period_l670_670312

def f (a b x : ℝ) : ℝ := a - b * Real.cos x
def g (a b x : ℝ) : ℝ := -4 * a * Real.sin (b * x)

theorem max_min_period (a b : ℝ) (h_max : ∃ x, f a b x = 5 / 2) 
  (h_min : ∃ x, f a b x = -1 / 2) (hb : b ≠ 0) :
  (∃ x, g a b x = 4) ∧ (∃ p > 0, ∀ x, g a b (x + p) = g a b x ∧ p = 4 * π / 3) :=
by
  sorry

end max_min_period_l670_670312


namespace tangent_circle_proof_l670_670257

open EuclideanGeometry
open Circle
open Segment

theorem tangent_circle_proof
  (PA PB : Tangent)
  (Γ : Circle)
  (A B : Point)
  (h₁ : PA ∈ tangent_points A Γ)
  (h₂ : PB ∈ tangent_points B Γ)
  (C : Point)
  (h₃ : C ∈ minor_arc Γ A B)
  (tangentF : Tangent)
  (tangentE : Tangent)
  (F : Point) (h₄ : tangentF ∈ tangent_points C Γ ∧ F ∈ intersect_tangent PA tangentF)
  (E : Point) (h₅ : tangentE ∈ tangent_points C Γ ∧ E ∈ intersect_tangent PB tangentE)
  (Γ' : Circle)
  (h₆ : Γ' = circumcircle (Triangle P F E))
  (M N : Point)
  (h₇ : M ∈ intersect_circle Γ' Γ ∧ N ∈ intersect_circle Γ' Γ)
  (K : Point) (h₈ : K ∈ intersect_lines (line AC) (line MN))
  (D : Point) (h₉ : D ∈ projections B (line AC))
  :
  AC / CK = 2 * AD / DK :=
begin
  sorry
end

end tangent_circle_proof_l670_670257


namespace largest_n_satisfying_conditions_l670_670699

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, n = 181 ∧
    (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧
    ∃ k : ℤ, 2 * n + 79 = k^2 :=
by
  sorry

end largest_n_satisfying_conditions_l670_670699


namespace perimeter_of_large_square_is_588_l670_670528

/-
conditions:
1. sum of the four sides of the original square is 56 cm
2. the original square is divided into 4 equal smaller squares
3. a new large square is made from 441 smaller squares
goal:
prove the perimeter of the new large square is 588 cm
-/

namespace SquareProblem

noncomputable def perimeter_of_large_square (sum_sides_original_square : ℕ) (num_smaller_squares : ℕ) : ℕ :=
  let side_length_original_square := sum_sides_original_square / 4
  let side_length_smaller_square := side_length_original_square / 2
  let new_large_square_side := Int.ofNat (Int.sqrt num_smaller_squares)
  let side_length_large_square := new_large_square_side * side_length_smaller_square
  4 * side_length_large_square
  
theorem perimeter_of_large_square_is_588 
  (h1 : sum_sides_original_square = 56) 
  (h2 : num_smaller_squares = 441) : 
  perimeter_of_large_square 56 441 = 588 :=
by 
  sorry

end SquareProblem

end perimeter_of_large_square_is_588_l670_670528


namespace number_of_paths_AMC10_l670_670382

theorem number_of_paths_AMC10 : 
  (∃ (paths : nat), paths = 4 * 3 * 2 * 2 ∧ paths = 48) :=
by
  sorry

end number_of_paths_AMC10_l670_670382


namespace measure_of_angle_ADC_l670_670444

variables (A B C : ℝ) -- Defining the angles of triangle ABC
-- Conditions
axiom sum_of_angles_in_triangle : A + B + C = 180
axiom angle_DAC : ∀D, ∠DAC = 90 - A / 2
axiom angle_DBC : ∀D, ∠DBC = 90 - B / 2

-- Lean statement
theorem measure_of_angle_ADC (A B C : ℝ) (D : Point) : 
  A + B + C = 180 → 
  (∀ D, ∠DAC = 90 - A / 2) → 
  (∀ D, ∠DBC = 90 - B / 2) → 
  ∠ADC = (180 - C) / 2 :=
by
  sorry

end measure_of_angle_ADC_l670_670444


namespace carpet_shaded_area_is_correct_l670_670243

def total_shaded_area (carpet_side_length : ℝ) (large_square_side : ℝ) (small_square_side : ℝ) : ℝ :=
  let large_shaded_area := large_square_side * large_square_side
  let small_shaded_area := small_square_side * small_square_side
  large_shaded_area + 12 * small_shaded_area

theorem carpet_shaded_area_is_correct :
  ∀ (S T : ℝ), 
  12 / S = 4 →
  S / T = 4 →
  total_shaded_area 12 S T = 15.75 :=
by
  intros S T h1 h2
  sorry

end carpet_shaded_area_is_correct_l670_670243


namespace sum_between_9p5_and_10_l670_670519

noncomputable def sumMixedNumbers : ℚ :=
  (29 / 9) + (11 / 4) + (81 / 20)

theorem sum_between_9p5_and_10 :
  9.5 < sumMixedNumbers ∧ sumMixedNumbers < 10 :=
by
  sorry

end sum_between_9p5_and_10_l670_670519


namespace minimize_maximum_absolute_value_expression_l670_670689

theorem minimize_maximum_absolute_value_expression : 
  (∀ y : ℝ, ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2) →
  ∃ y : ℝ, (y = 2) ∧ (min_value = 0) :=
sorry -- Proof goes here

end minimize_maximum_absolute_value_expression_l670_670689


namespace advantageous_logs_l670_670398

theorem advantageous_logs :
  ∀ (l1 l2 : ℕ), l1 = 6 → l2 = 8 → 
  (∀ (p1 c1 p2 c2 : ℕ), p1 = l1 ∧ c1 = l1 - 1 ∧ p2 = l2 ∧ c2 = l2 - 1 →
  let total_pieces_l1 := (35 / c1) * p1,
      total_pieces_l2 := (35 / c2) * p2 in
  total_pieces_l1 > total_pieces_l2) :=
by
  intros l1 l2 hl1 hl2;
  intros p1 c1 p2 c2 hpdef;
  cases hpdef with hp1 hc1;
  cases hc1 with hp2 hc2;
  rw [hl1, hl2] at hp1 hc1 hp2 hc2;
  let total_pieces_l1 := (35 / (l1 - 1)) * l1;
  let total_pieces_l2 := (35 / (l2 - 1)) * l2;
  sorry

end advantageous_logs_l670_670398


namespace greatest_integer_le_x_squared_div_50_l670_670463

-- Define the conditions as given in the problem
def trapezoid (b h : ℝ) (x : ℝ) : Prop :=
  let baseDifference := 50
  let longerBase := b + baseDifference
  let midline := (b + longerBase) / 2
  let heightRatioFactor := 2
  let xSquared := 6875
  let regionAreaRatio := 2 / 1 -- represented as 2
  (let areaRatio := (b + midline) / (b + baseDifference / 2)
   areaRatio = regionAreaRatio) ∧
  (x = Real.sqrt xSquared) ∧
  (b = 50)

-- Define the theorem that captures the question
theorem greatest_integer_le_x_squared_div_50 (b h x : ℝ) (h_trapezoid : trapezoid b h x) :
  ⌊ (x^2) / 50 ⌋ = 137 :=
by sorry

end greatest_integer_le_x_squared_div_50_l670_670463


namespace describe_T_l670_670068

def T : Set (ℝ × ℝ) := 
  { p | ∃ x y : ℝ, p = (x, y) ∧ (
      (5 = x + 3 ∧ y - 6 ≤ 5) ∨
      (5 = y - 6 ∧ x + 3 ≤ 5) ∨
      (x + 3 = y - 6 ∧ x + 3 ≤ 5 ∧ y - 6 ≤ 5)
  )}

theorem describe_T : T = { p | ∃ x y : ℝ, p = (2, y) ∧ y ≤ 11 ∨
                                      p = (x, 11) ∧ x ≤ 2 ∨
                                      p = (x, x + 9) ∧ x ≤ 2 ∧ x + 9 ≤ 11 } :=
by
  sorry

end describe_T_l670_670068


namespace find_angle_C_c_squared_geq_4sqrt3S_l670_670732

variable (A B C a b c S : ℝ)
variable (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
variable (h₄ : tan (A / 2) * tan (B / 2) + sqrt 3 * (tan (A / 2) + tan (B / 2)) = 1)
variable (h₅ : S = 1 / 2 * a * b * sin C)
variable (h₆ : C = π - (A + B))

theorem find_angle_C (h₇ : A + B = π / 3) : C = 2 * π / 3 :=
by
  rw [h₆, h₇]
  linarith

theorem c_squared_geq_4sqrt3S 
  (h₇ : C = 2 * π / 3)
  (h₈ : c^2 = a^2 + b^2 - 2 * a * b * cos C) 
  (h₉ : S = (sqrt 3 / 4) * a * b) : 
    c^2 ≥ 4 * sqrt 3 * S :=
by
  rw [h₇, cos_two_pi_div_three] at h₈
  have : c^2 = a^2 + b^2 + a * b := by rw [cos_two_pi_div_three]; ring
  rw [h₉] at this
  linarith

end find_angle_C_c_squared_geq_4sqrt3S_l670_670732


namespace determine_k_l670_670669

theorem determine_k (k : ℝ) : (1 - 3 * k * (-2/3) = 7 * 3) → k = 10 :=
by
  intro h
  sorry

end determine_k_l670_670669


namespace solve_for_x_l670_670706

theorem solve_for_x : 
  ∃ x : ℚ, (real.sqrt (4 * x + 9) / real.sqrt (8 * x + 9) = real.sqrt 3 / 2) ∧ x = 9 / 8 :=
by
  sorry

end solve_for_x_l670_670706


namespace min_max_value_is_zero_l670_670686

def max_at_x (x : ℝ) (y : ℝ) : ℝ := |x^2 - 2 * x * y|

theorem min_max_value_is_zero :
  ∃ y ∈ set.univ, min (set.univ) (λ y, real.sup (set.Icc 0 2) (λ x, max_at_x x y)) = 0 :=
sorry

end min_max_value_is_zero_l670_670686


namespace number_of_ways_to_distribute_balls_l670_670355

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l670_670355


namespace necessary_but_not_sufficient_condition_l670_670730

theorem necessary_but_not_sufficient_condition {x m : ℝ} 
  (p : ∃ x, abs (x - 1) + abs (x - 3) < m) 
  (q : ∀ x, (0 < 7 - 3 * m) ∧ (7 - 3 * m < 1)) :
  (p → q) ∧ ¬ (q → p) :=
by 
  sorry

end necessary_but_not_sufficient_condition_l670_670730


namespace uphill_speed_approx_30_l670_670995

noncomputable def uphill_speed : ℝ :=
let distance_uphill : ℝ := 100
let distance_downhill : ℝ := 50
let speed_downhill : ℝ := 80
let avg_speed : ℝ := 37.89 in
150 / ((100 / uphill_speed) + (50 / 80))

theorem uphill_speed_approx_30 : uphill_speed ≈ 30 := sorry

end uphill_speed_approx_30_l670_670995


namespace valid_parametrizations_l670_670891

-- Define the line as a function
def line (x : ℝ) : ℝ := -2 * x + 7

-- Define vectors and their properties
structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def on_line (v : Vector2D) : Prop :=
  v.y = line v.x

def direction_vector (v1 v2 : Vector2D) : Vector2D :=
  ⟨v2.x - v1.x, v2.y - v1.y⟩

def is_multiple (v1 v2 : Vector2D) : Prop :=
  ∃ k : ℝ, v2.x = k * v1.x ∧ v2.y = k * v1.y

-- Define the given parameterizations
def param_A (t : ℝ) : Vector2D := ⟨0 + t * 5, 7 + t * 10⟩
def param_B (t : ℝ) : Vector2D := ⟨2 + t * 1, 3 + t * -2⟩
def param_C (t : ℝ) : Vector2D := ⟨7 + t * 4, 0 + t * -8⟩
def param_D (t : ℝ) : Vector2D := ⟨-1 + t * 2, 9 + t * 4⟩
def param_E (t : ℝ) : Vector2D := ⟨3 + t * 2, 1 + t * 0⟩

-- Define the theorem
theorem valid_parametrizations :
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨0, 7⟩ (param_A t)) ∧ on_line (param_A t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨2, 3⟩ (param_B t)) ∧ on_line (param_B t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨7, 0⟩ (param_C t)) ∧ on_line (param_C t)) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨-1, 9⟩ (param_D t)) ∧ on_line (param_D t) → False) ∧
  (∀ t, is_multiple ⟨1, -2⟩ (direction_vector ⟨3, 1⟩ (param_E t)) ∧ on_line (param_E t) → False) :=
by
  sorry

end valid_parametrizations_l670_670891


namespace find_matrix_N_l670_670700

theorem find_matrix_N (a b c d : ℝ) :
  ∃ N : Matrix (Fin 2) (Fin 2) ℝ, 
    (N * (Matrix.vec_cons (Matrix.vec_cons a b) (Matrix.vec_cons c d)) = 
    Matrix.vec_cons (Matrix.vec_cons (2 * a) (2 * b)) (Matrix.vec_cons (3 * c) (3 * d))) ∧
    (N = Matrix.vec_cons (Matrix.vec_cons 2 0) (Matrix.vec_cons 0 3)) := sorry

end find_matrix_N_l670_670700


namespace third_vertex_coordinates_of_obtuse_triangle_l670_670541

theorem third_vertex_coordinates_of_obtuse_triangle : 
  ∀ (x : ℝ), (x < 0) → ∃ (A B C : ℝ × ℝ), 
  A = (7, 3) ∧ B = (0, 0) ∧ C = (x, 0) ∧ 
  (1 / 2) * (real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)) * (real.abs x) = 24 :=
by
  assume x x_neg
  let A := (7, 3)
  let B := (0, 0)
  let C := (x, 0)
  exists A B C
  split
  repeat {split}
  repeat {sorry}

end third_vertex_coordinates_of_obtuse_triangle_l670_670541


namespace determine_selling_price_per_bike_maximize_profit_l670_670453

-- Define all necessary conditions
def last_year_sales_revenue : ℤ := 50000
def decrease_percent : ℤ := 20
def decrease_amount : ℤ := 400
def purchase_price_A : ℤ := 1100
def purchase_price_B : ℤ := 1400
def selling_price_A_this_year : ℤ := 1600
def selling_price_B : ℤ := 2000

-- Define constraints
noncomputable def total_bikes : ℤ := 60
-- Selling price per bike of type A this year
theorem determine_selling_price_per_bike (x : ℤ) : 
  (50000 * x = 40000 * (x + 400)) → 
  x = selling_price_A_this_year :=
by { intro h, sorry }

-- Define profitability and constraints
def profit (a b : ℤ) : ℤ :=
  (selling_price_A_this_year - purchase_price_A) * a + (selling_price_B - purchase_price_B) * b

-- Determine the number of type A and type B bikes to maximize profit
theorem maximize_profit (a b : ℤ) :
  a + b = total_bikes ∧ b ≤ 2 * a ∧ 0 ≤ a ∧ 0 ≤ b → 
  profit a b ≤ profit 20 40 :=
by { intro h, sorry }

end determine_selling_price_per_bike_maximize_profit_l670_670453


namespace software_package_cost_l670_670631

theorem software_package_cost 
  (devices : ℕ) 
  (cost_first : ℕ) 
  (devices_covered_first : ℕ) 
  (devices_covered_second : ℕ) 
  (savings : ℕ)
  (total_cost_first : ℕ := (devices / devices_covered_first) * cost_first)
  (total_cost_second : ℕ := total_cost_first - savings)
  (num_packages_second : ℕ := devices / devices_covered_second)
  (cost_second : ℕ := total_cost_second / num_packages_second) :
  devices = 50 ∧ cost_first = 40 ∧ devices_covered_first = 5 ∧ devices_covered_second = 10 ∧ savings = 100 →
  cost_second = 60 := 
by
  sorry

end software_package_cost_l670_670631


namespace XY2_equals_888_l670_670072

-- Defining the problem
variables (A B C T X Y : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space T]
variables [metric_space X] [metric_space Y]
variables {angle_AB_TC : angle A B}
variables {is_projection_T_on_AB : ∀ T X, orthogonal_projection A B T X}
variables {is_projection_T_on_AC : ∀ T Y, orthogonal_projection A C T Y}
variables {BT : ℝ} {CT : ℝ} {BC : ℝ} {TX2_TY2_XY2_sum : ℝ}
variables (B.equals : BT = 18) (C.equals : CT = 18) (BC.equals : BC = 24)

-- Assumptions and conditions
variables (TX2_TY2_XY2_sum.equals : TX ^ 2 + TY ^ 2 + XY ^ 2 = 1368)

-- Goal: Prove that the square of XY is 888
theorem XY2_equals_888 : XY ^ 2 = 888 :=
by
  sorry

end XY2_equals_888_l670_670072


namespace result_m_plus_n_l670_670362

-- Definitions from the conditions
def like_terms (e1 e2 : ℕ × ℕ) : Prop :=
  e1.2 = e2.2

-- The main statement to prove
theorem result_m_plus_n (m n : ℕ)
  (h1 : like_terms (m, m + 1) (n - 1, 3)) :
  m + n = 5 :=
begin
  sorry
end

end result_m_plus_n_l670_670362


namespace g_g_is_odd_l670_670985

def f (x : ℝ) : ℝ := x^3

def g (x : ℝ) : ℝ := f (f x)

theorem g_g_is_odd : ∀ x : ℝ, g (g (-x)) = -g (g x) :=
by 
-- proof will go here
sorry

end g_g_is_odd_l670_670985


namespace triangle_KLM_angles_l670_670094

theorem triangle_KLM_angles 
  (ABC : Type) [Nonempty ABC] [MetricSpace ABC] 
  (A B C A' C' B' M K L : ABC)
  (h_eq_triangles : is_equilateral_triangle A B C' ∧ is_equilateral_triangle A B' C)
  (h_M_divides_BC : divides_in_ratio M B C 3 1)
  (h_K_mid_AC' : midpoint K A C')
  (h_L_mid_B'C : midpoint L B' C) :
  ∃ θ₁ θ₂ θ₃, angles K L M = (θ₁, θ₂, θ₃) ∧ {θ₁, θ₂, θ₃} = {30°, 60°, 90°} :=
sorry

end triangle_KLM_angles_l670_670094


namespace probability_one_tail_in_three_flips_l670_670370

theorem probability_one_tail_in_three_flips :
  let p := 0.5 in
  let n := 3 in
  let k := 1 in
  binomial_probability (n choose k) (p^k) ((1 - p)^(n - k)) = 0.375 :=
by
  sorry

noncomputable def binomial_probability (n_choose_k : ℕ) (p_to_k : ℝ) (one_minus_p_to_n_minus_k : ℝ) : ℝ :=
  n_choose_k * p_to_k * one_minus_p_to_n_minus_k

end probability_one_tail_in_three_flips_l670_670370


namespace stapler_machines_l670_670614

theorem stapler_machines (x : ℝ) :
  (∃ (x : ℝ), x > 0) ∧
  ((∀ r1 r2 : ℝ, (r1 = 800 / 6) → (r2 = 800 / x) → (r1 + r2 = 800 / 3)) ↔
    (1 / 6 + 1 / x = 1 / 3)) :=
by sorry

end stapler_machines_l670_670614


namespace smallest_two_digit_palindrome_l670_670956

def is_palindrome {α : Type} [DecidableEq α] (xs : List α) : Prop :=
  xs = xs.reverse

-- A number is a two-digit palindrome in base 5 if it has the form ab5 where a and b are digits 0-4
def two_digit_palindrome_base5 (n : ℕ) : Prop :=
  ∃ a b : ℕ, a < 5 ∧ b < 5 ∧ a ≠ 0 ∧ n = a * 5 + b ∧ is_palindrome [a, b]

-- A number is a three-digit palindrome in base 2 if it has the form abc2 where a = c and b can vary (0-1)
def three_digit_palindrome_base2 (n : ℕ) : Prop :=
  ∃ a b c : ℕ, a < 2 ∧ b < 2 ∧ c < 2 ∧ a = c ∧ n = a * 4 + b * 2 + c ∧ is_palindrome [a, b, c]

theorem smallest_two_digit_palindrome :
  ∃ n, two_digit_palindrome_base5 n ∧ three_digit_palindrome_base2 n ∧
       (∀ m, two_digit_palindrome_base5 m ∧ three_digit_palindrome_base2 m → n ≤ m) :=
sorry

end smallest_two_digit_palindrome_l670_670956


namespace fraction_subtraction_l670_670691

theorem fraction_subtraction :
  (8 / 23) - (5 / 46) = 11 / 46 := by
  sorry

end fraction_subtraction_l670_670691


namespace num_perfect_squares_l670_670793

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l670_670793


namespace correct_option_is_D_l670_670672

def is_triangle_constructible (f : ℝ → ℝ) : Prop :=
  ∀ a b c : ℝ, (f a + f b > f c) ∧ (f a + f c > f b) ∧ (f b + f c > f a)

def problem_statement : Prop :=
  (is_triangle_constructible (λ x, 1)) ∧
  ¬(∀ f : ℝ → ℝ, is_triangle_constructible f → monotone f) ∧
  ¬(is_triangle_constructible (λ x, 1 / (x^2 + 1))) ∧
  (∀ f : ℝ → ℝ, (∀ x, f x ≥ Real.sqrt Real.e ∧ f x ≤ Real.e) → is_triangle_constructible f)

theorem correct_option_is_D : problem_statement :=
begin
  -- The theorem proof needs to be filled here.
  sorry
end

end correct_option_is_D_l670_670672


namespace probability_at_least_one_visits_guangzhou_l670_670912

-- Define the probabilities of visiting for persons A, B, and C
def p_A : ℚ := 2 / 3
def p_B : ℚ := 1 / 4
def p_C : ℚ := 3 / 5

-- Calculate the probability that no one visits
def p_not_A : ℚ := 1 - p_A
def p_not_B : ℚ := 1 - p_B
def p_not_C : ℚ := 1 - p_C

-- Calculate the probability that at least one person visits
def p_none_visit : ℚ := p_not_A * p_not_B * p_not_C
def p_at_least_one_visit : ℚ := 1 - p_none_visit

-- The statement we need to prove
theorem probability_at_least_one_visits_guangzhou : p_at_least_one_visit = 9 / 10 :=
by 
  sorry

end probability_at_least_one_visits_guangzhou_l670_670912


namespace ninth_term_arithmetic_sequence_l670_670136

theorem ninth_term_arithmetic_sequence 
  (a1 a17 d a9 : ℚ) 
  (h1 : a1 = 2 / 3) 
  (h17 : a17 = 3 / 2) 
  (h_formula : a17 = a1 + 16 * d) 
  (h9_formula : a9 = a1 + 8 * d) :
  a9 = 13 / 12 := by
  sorry

end ninth_term_arithmetic_sequence_l670_670136


namespace original_number_l670_670418

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end original_number_l670_670418


namespace unique_positive_integers_exists_l670_670968

theorem unique_positive_integers_exists (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) : 
  ∃! m n : ℕ, m^2 = n * (n + p) ∧ m = (p^2 - 1) / 2 ∧ n = (p - 1)^2 / 4 := by
  sorry

end unique_positive_integers_exists_l670_670968


namespace vector_parallel_condition_l670_670343

theorem vector_parallel_condition (k : ℝ) :
  let a := (2 * k + 2, 4)
  let b := (k + 1, 8)
  (2 * k + 2) * 8 - (k + 1) * 4 = 0 -> k = -1 :=
by
  let a := (2 * k + 2, 4)
  let b := (k + 1, 8)
  assume h : (2 * k + 2) * 8 - (k + 1) * 4 = 0
  sorry

end vector_parallel_condition_l670_670343


namespace final_silver_tokens_l670_670263

def tokens_state (x y : ℕ) := 100 - 3 * x + y ≥ 3 ∧ 100 + x - 4 * y ≥ 4

def silver_tokens_sum (x y : ℕ) : ℕ := 2 * x + 2 * y

theorem final_silver_tokens
: 
  let x := 30, let y := 20 in 
  tokens_state x y → 
  silver_tokens_sum x y = 100 := 
by {
  intros,
  sorry
}

end final_silver_tokens_l670_670263


namespace arrange_spies_l670_670242

-- Define the board as a 6x6 grid
def Board := fin 6 × fin 6

-- A spy occupies a single cell on the board
structure Spy where
  pos : Board

-- Line of sight function, checks if one spy can see another
def canSee (s1 s2 : Spy) : Prop :=
let dx := s2.pos.1 - s1.pos.1
let dy := s2.pos.2 - s1.pos.2
(dx = 0 ∧ (dy = 1 ∨ dy = 2)) ∨ (dy = 0 ∧ (dx = 1 ∨ dx = 2)) ∨ ((dx = 0 ∧ (dy = -1 ∨ dy = -2)) ∨ (dy = 0 ∧ (dx = -1 ∨ dx = -2)))

-- Prove the placement is valid
theorem arrange_spies : ∃ (spies : list Spy), spies.length = 18 ∧ ∀ s1 s2 ∈ spies, s1 ≠ s2 → ¬ canSee s1 s2 :=
by
  -- Assume a possible placement of spies that satisfies the conditions
  let spies : list Spy := [(0, 0), (0, 2), (0, 4), (1, 1), (1, 3), (1, 5), (2, 0), (2, 2), (2, 4),
                           (3, 1), (3, 3), (3, 5), (4, 0), (4, 2), (4, 4), (5, 1), (5, 3), (5, 5)]
  -- Check length condition
  have h_len : spies.length = 18 := by sorry
  -- Check non-intersecting line-of-sight condition
  have h_vis : ∀ s1 s2 ∈ spies, s1 ≠ s2 → ¬ canSee s1 s2 := by sorry
  -- Proof
  exact ⟨spies, h_len, h_vis⟩

end arrange_spies_l670_670242


namespace volume_of_red_tetrahedron_in_colored_cube_l670_670610

noncomputable def red_tetrahedron_volume (side_length : ℝ) : ℝ :=
  let cube_volume := side_length ^ 3
  let clear_tetrahedron_volume := (1/3) * (1/2 * side_length * side_length) * side_length
  let red_tetrahedron_volume := (cube_volume - 4 * clear_tetrahedron_volume)
  red_tetrahedron_volume

theorem volume_of_red_tetrahedron_in_colored_cube 
: red_tetrahedron_volume 8 = 512 / 3 := by
  sorry

end volume_of_red_tetrahedron_in_colored_cube_l670_670610


namespace product_of_y_coordinates_l670_670096

theorem product_of_y_coordinates :
  (∀ y : ℝ, (∀ (P : ℝ × ℝ), P.1 = -3 ∧ dist (P.1, P.2) (7, -3) = 15 → P.2 = y) →
    (∃ a b : ℝ, a = -3 + √125 ∧ b = -3 - √125 ∧ a * b = -116)) :=
begin
  intros y P h,
  rcases h with ⟨hx, hd⟩,
  cases eq_or_ne P.1 (-3) with h₀ h₀,
  { -- case when P.1 = -3
    have h₁ : P = (-3, y), from eq.symm h₀,

    -- compute y
    cases eq_or_ne (dist (7, -3) P) 15 with d₀ d₀,
    { -- case when dist = 15
      have h₂ : dist (7, -3) P = 15, from dist_comm (7, -3) P ▸ d₀,
      simp [dist, P] at h₂,

      -- Now solve based on the given constraint
      have h₃ : 100 + (-3 - y) ^ 2 = 225, by linarith,
      have h₄ : (-3 - y)^2 = 125, by {apply eq_sub_of_add_eq h₃, ring},

      -- Solve for y
      set y₁ := -3 + real.sqrt 125 with h₅,
      set y₂ := -3 - real.sqrt 125 with h₆,

      -- Compute the product of y₁ and y₂
      have hproduct : y₁ * y₂ = -116,
      { calc
          y₁ * y₂ = (-3 + real.sqrt 125) * (-3 - real.sqrt 125) : by rw [h₅, h₆]
          ... = (-3)^2 - (real.sqrt 125)^2 : by {apply z_mod.sqrt_square}
          ... = 9 - 125 : by {norm_num}
          ... = -116 : by ring
      },
      exact ⟨-3 + real.sqrt 125, -3 - real.sqrt 125, rfl, rfl, hproduct⟩
   },
    {sorry}
  },
  { sorry }
end

end product_of_y_coordinates_l670_670096


namespace calculate_cereal_cost_l670_670538

theorem calculate_cereal_cost
  (boxes_per_week : ℕ)
  (weeks_per_year : ℕ)
  (total_spend : ℕ)
  (total_boxes : ℕ := boxes_per_week * weeks_per_year)
  (cost_per_box : ℕ := total_spend / total_boxes) :
  boxes_per_week = 2 → weeks_per_year = 52 → total_spend = 312 → cost_per_box = 3 :=
by
  intros h1 h2 h3
  simp [h1, h2, total_boxes, h3, cost_per_box]
  sorry

end calculate_cereal_cost_l670_670538


namespace variance_of_Y_l670_670226

noncomputable def spectral_density (τ : ℝ) : ℝ := 6 * Real.exp (-2 * |τ|)

theorem variance_of_Y (ω : ℝ) :
  (∃ (s_x : ℝ → ℝ), ∀ ω, (s_x ω = ∫ τ in Real, spectral_density τ * Real.exp (- Complex.I * ω * τ) ∂τ) ∧
                      s_x ω = 12 / (Real.pi * (ω^2 + 4))) :=
by
  use λ ω, 12 / (Real.pi * (ω^2 + 4))
  intros ω
  split
  sorry
  refl

end variance_of_Y_l670_670226


namespace LHBC_concyclic_l670_670432

theorem LHBC_concyclic
  (O H D E F G L T : Point)
  (ABC: Triangle)
  (hO: O = circumcenter ABC)
  (hH: H = orthocenter ABC)
  (hD: exists P: Point, is_perpendicular_bisector P D AC ∧ is_perpendicular_bisector P D BC)
  (hE: exists Q: Point, is_perpendicular_bisector Q E AC ∧ is_perpendicular_bisector Q E BC)
  (hCirc_DEH: circumcircle (triangle DEH))
  (hF: is_meeting (circumcircle (triangle DEH)) AC F)
  (hG: is_meeting (circumcircle (triangle DEH)) BC G)
  (hL: is_meeting (circumcircle (triangle DEH)) OH L)
  (hT: is_meeting (CH ∣ FG) T)
  (concyclic_ABCT: concyclic {A, B, C, T}) :
  concyclic {L, H, B, C} :=
sorry

end LHBC_concyclic_l670_670432


namespace find_a_l670_670761

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main theorem statement
theorem find_a (a : ℝ) :
  let f1 := f a 1 in
  let f'1 := f' a 1 in
  let tangent_line_passing := (λ x, f1 + f'1 * (x - 1)) in
  tangent_line_passing 2 = 7 → a = 1 :=
by
  intros h
  sorry

end find_a_l670_670761


namespace tangent_product_eq_quarter_square_base_l670_670997

theorem tangent_product_eq_quarter_square_base
  (A B C P Q M : Point) -- Define points A, B, C, P, Q, M.
  (hABC_isosceles : is_isosceles_triangle A B C) -- Triangle ABC is isosceles.
  (hM_midpoint : midpoint M B C) -- M is the midpoint of BC.
  (h_circle_center : circle_center M) -- The circle's center is M.
  (h_circle_touches : circle_touches_legs A B C P Q M) -- The circle touches AB at P and AC at Q.
  (h_tangent_intersects : tangent_intersects AB AC P Q) -- A tangent to the circle intersects AB at P and AC at Q):
  (BP CQ BC : ℝ) -- Lengths of the respective segments.
  (h_side_lengths : BP = B.distance P ∧ CQ = C.distance Q ∧ BC = B.distance C) -- Define lengths in terms of the distances between points.
  : BP * CQ = (BC^2) / 4 := 
sorry

end tangent_product_eq_quarter_square_base_l670_670997


namespace area_of_rectangular_garden_l670_670509

-- Definition of conditions
def width : ℕ := 14
def length : ℕ := 3 * width

-- Statement for proof of the area of the rectangular garden
theorem area_of_rectangular_garden :
  length * width = 588 := 
by
  sorry

end area_of_rectangular_garden_l670_670509


namespace parabola_and_length_ef_l670_670326

theorem parabola_and_length_ef :
  ∃ a b : ℝ, (∀ x : ℝ, (x + 1) * (x - 3) = 0 → a * x^2 + b * x + 3 = 0) ∧ 
            (∀ x : ℝ, -a * x^2 + b * x + 3 = 7 / 4 → 
              ∃ x1 x2 : ℝ, x1 = -1 / 2 ∧ x2 = 5 / 2 ∧ abs (x2 - x1) = 3) := 
sorry

end parabola_and_length_ef_l670_670326


namespace ratio_of_C_to_A_l670_670249

open_locale classical
noncomputable theory

theorem ratio_of_C_to_A (x : ℝ) (m : ℝ) (C_investment : ℝ) (A_investment : ℝ) 
  (annual_gain : ℝ) (A_share : ℝ)
  (investment_time_A : ℝ) (investment_time_B : ℝ) (investment_time_C : ℝ)
  (investment_B : ℝ) 
  (h1 : A_investment = x)
  (h2 : investment_B = 2 * x)
  (h3 : C_investment = m * x)
  (h4 : investment_time_A = 12)
  (h5 : investment_time_B = 6)
  (h6 : investment_time_C = 4)
  (h7 : annual_gain = 15000)
  (h8 : A_share = 5000) : 
  3 * x = m * x :=
by 
  have h9 : A_share = annual_gain / 3, from sorry,
  have h10 : A_investment * investment_time_A = (A_investment * investment_time_A 
                                                + investment_B * investment_time_B
                                                + C_investment * investment_time_C) / 3, from sorry,
  have h11 : 12 * A_investment = 
             (12 * A_investment + 6 * investment_B + 4 * C_investment) / 3, from sorry,
  have h12 : 12 * x = 
             (12 * x + 12 * x + 4 * m * x) / 3, from sorry,
  have h13 : 12 * x = 4 * x + (4 / 3) * m * x, from sorry,
  have h14 : 12 * x = 8 * x + (4 / 3) * m * x, from sorry,
  have h15 : 4 * x = (4 / 3) * m * x, from sorry,
  have h16 : 3 * x = m * x, from sorry,
  exact h16ertig

end ratio_of_C_to_A_l670_670249


namespace triangle_is_isosceles_l670_670144

-- Define the essential structures and properties
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Triangle :=
  (A B C : Point)

structure Midpoint (p1 p2 : Point) :=
  (M : Point)
  (proof : M.x = (p1.x + p2.x) / 2 ∧ M.y = (p1.y + p2.y) / 2)

structure Median (t : Triangle) (m : Midpoint t.A t.B) :=
  (P : Point)
  (proof : P.x = (t.C.x + m.M.x) / 2 ∧ P.y = (t.C.y + m.M.y) / 2)

def are_perpendicular (v1 v2 : Point) : Prop :=
  (v1.x * v2.x + v1.y * v2.y) = 0

noncomputable def isosceles (t : Triangle) : Prop :=
  dist t.A t.B = dist t.A t.C

theorem triangle_is_isosceles
  (ABC : Triangle)
  (B1 : Midpoint ABC.A ABC.C)
  (C1 : Midpoint ABC.A ABC.B)
  (AM : Point)
  (M : Median ABC B1)
  (H₁ : are_perpendicular AM (B1.M, C1.M))
  : isosceles ABC := sorry


end triangle_is_isosceles_l670_670144


namespace smallest_n_for_last_four_digits_sum_l670_670277

theorem smallest_n_for_last_four_digits_sum :
  ∃ (n : ℕ) (a : Fin n → ℕ), (∀ i, 1 ≤ a i ∧ a i ≤ 15) ∧ 
  (let sum := (∑ i, (Nat.factorial (a i)) % 10000) in sum = 2001) ∧ 
  (∀ (m : ℕ) (b : Fin m → ℕ), (∀ j, 1 ≤ b j ∧ b j ≤ 15) ∧ 
  let bs := (∑ j, (Nat.factorial (b j)) % 10000) in bs = 2001 -> n ≤ m) :=
sorry

end smallest_n_for_last_four_digits_sum_l670_670277


namespace probability_of_rolling_perfect_square_l670_670121

theorem probability_of_rolling_perfect_square :
  let numbers := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
  let perfect_squares := {1, 4, 9}
  let total_perfect_squares := 3
  let total_outcomes := 10
  (total_perfect_squares : ℝ) / (total_outcomes : ℝ) = 3 / 10 := 
by
  sorry

end probability_of_rolling_perfect_square_l670_670121


namespace funnel_height_l670_670229

noncomputable def cone_height (V : ℝ) (r : ℝ) : ℝ :=
  (3 * V) / (Math.pi * r^2)

theorem funnel_height {V r : ℝ} (h : ℝ) (hV : V = 150) (hr : r = 4) :
  abs (h - 9) < 1 -> cone_height V r = h :=
by
  rw [hV, hr, cone_height]
  sorry

end funnel_height_l670_670229


namespace ride_cost_l670_670919

theorem ride_cost (joe_age_over_18 : Prop)
                   (joe_brother_age : Nat)
                   (joe_entrance_fee : ℝ)
                   (brother_entrance_fee : ℝ)
                   (total_spending : ℝ)
                   (rides_per_person : Nat)
                   (total_persons : Nat)
                   (total_entrance_fee : ℝ)
                   (amount_spent_on_rides : ℝ)
                   (total_rides : Nat) :
  joe_entrance_fee = 6 →
  brother_entrance_fee = 5 →
  total_spending = 20.5 →
  rides_per_person = 3 →
  total_persons = 3 →
  total_entrance_fee = 16 →
  amount_spent_on_rides = (total_spending - total_entrance_fee) →
  total_rides = (rides_per_person * total_persons) →
  (amount_spent_on_rides / total_rides) = 0.50 :=
by
  sorry

end ride_cost_l670_670919


namespace proof_l670_670391

noncomputable def line_standard_form (t : ℝ) : Prop :=
  let (x, y) := (t + 3, 3 - t)
  x + y = 6

noncomputable def circle_standard_form (θ : ℝ) : Prop :=
  let (x, y) := (2 * Real.cos θ, 2 * Real.sin θ + 2)
  x^2 + (y - 2)^2 = 4

noncomputable def distance_center_to_line (x1 y1 : ℝ) : ℝ :=
  let (a, b, c) := (1, 1, -6)
  let num := abs (a * x1 + b * y1 + c)
  let denom := Real.sqrt (a^2 + b^2)
  num / denom

theorem proof : 
  (∀ t, line_standard_form t) ∧ 
  (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → circle_standard_form θ) ∧ 
  distance_center_to_line 0 2 = 2 * Real.sqrt 2 :=
by
  sorry

end proof_l670_670391


namespace rent_percentage_l670_670171

theorem rent_percentage (E : ℝ) :
  let two_years_ago_rent := 0.25 * E
  let last_year_earnings := 1.35 * E
  let last_year_rent := 0.40 * last_year_earnings
  let this_year_earnings := 1.9575 * E
  let this_year_rent := 0.50 * this_year_earnings
  (this_year_rent / two_years_ago_rent) * 100 = 391.5 := by
  -- Definitions
  let two_years_ago_rent := 0.25 * E
  let last_year_earnings := 1.35 * E
  let last_year_rent := 0.40 * last_year_earnings
  let this_year_earnings := 1.9575 * E
  let this_year_rent := 0.50 * this_year_earnings
  
  -- Calculation and Proof
  have h1 : this_year_rent = 0.97875 * E := rfl
  have h2 : two_years_ago_rent = 0.25 * E := rfl
  calc
    (this_year_rent / two_years_ago_rent) * 100
    = (0.97875 * E / 0.25 * E) * 100 : by rw [h1, h2]
    = (0.97875 / 0.25) * 100 : by rw [mul_div_mul_left _ _ _ (ne_of_lt (by norm_num : 0 < E))]
    = 3.915 * 100 : by norm_num
    = 391.5 : by norm_num

end rent_percentage_l670_670171


namespace sum_of_primes_between_10_and_50_l670_670475

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def smallest_prime (l : List ℕ) : ℕ :=
  l.filter is_prime |>.minimum'

def largest_prime (l : List ℕ) : ℕ :=
  l.filter is_prime |>.maximum'

-- The list of numbers between 10 and 50
def nums : List ℕ := List.range' 10 (50 - 10 + 1)

theorem sum_of_primes_between_10_and_50 : (smallest_prime nums) + (largest_prime nums) = 58 :=
by
  sorry

end sum_of_primes_between_10_and_50_l670_670475


namespace sailboat_speed_at_max_power_l670_670590

variable (C S ρ v_0 v N: ℝ)

-- Conditions
axiom sail_area : S = 5  -- S in square meters
axiom wind_speed : v_0 = 6  -- v_0 in m/s
axiom force_formula : C * S * ρ * (v_0 - v) ^ 2 / 2 = N / v

-- Proof statement
theorem sailboat_speed_at_max_power
  (h₁ : sail_area)
  (h₂ : wind_speed)
  (h₃ : force_formula) :
  v = v_0 / 3 :=
sorry

end sailboat_speed_at_max_power_l670_670590


namespace dmitry_black_socks_l670_670284

theorem dmitry_black_socks (b : ℕ) : 
  let blue_socks := 14
  let initial_black_socks := 24
  let white_socks := 10
  let initial_total_socks := blue_socks + initial_black_socks + white_socks
  let new_total_socks := initial_total_socks + b
  let new_black_socks := initial_black_socks + b
  (new_black_socks : ℚ) = (3 / 5) * new_total_socks → b = 12 :=
by
  intros
  let initial_total_socks := 14 + 24 + 10
  let new_total_socks := initial_total_socks + b
  let new_black_socks := 24 + b
  suffices : (new_black_socks : ℚ) = (3 / 5) * new_total_socks → b = 12
  sorry

end dmitry_black_socks_l670_670284


namespace angle_equality_l670_670852

-- Definitions of the circles and points as given in the problem
variable {O A B C D P : Point}
variable {𝜔 γ: Circle}

-- Conditions: construction of the circles and the fact that they intersect and pass through required points
variable h1 : CenterOfCircle O 𝜔
variable h2 : OnCircle A 𝜔
variable h3 : OnCircle B 𝜔
variable h4 : OnCircle O γ
variable h5 : OnCircle A γ
variable h6 : OnCircle B γ
variable h7 : OnCircle P γ
variable h8 : LineThroughDiameter C D 𝜔
variable h9 : IntersectLineCircleProper (LineThroughPoints C D) γ P O

-- Proof goal
theorem angle_equality (h1 h2 h3 h4 h5 h6 h7 h8 h9) : ∠APC = ∠BPD := by
  sorry

end angle_equality_l670_670852


namespace split_faces_into_polyhedra_l670_670517

theorem split_faces_into_polyhedra (P : Type) [convex_polyhedron P] (faces : set (face P)) :
  ∃ (faces1 faces2 : set (face P)), 
    (faces = faces1 ∪ faces2) ∧ 
    (faces1 ∩ faces2 = ∅) ∧
    (∃ (P1 P2 : Type) [convex_polyhedron P1] [convex_polyhedron P2], 
      faces1 (face P1) ∧ faces2 (face P2)) :=
sorry

end split_faces_into_polyhedra_l670_670517


namespace original_number_is_857142_l670_670238

noncomputable def six_digit_number (N : ℕ) : Prop :=
  N % 10 = 2 ∧ (2 * 10^5 + N / 10) = 3 * N

theorem original_number_is_857142 : ∃ N : ℕ, six_digit_number N ∧ N = 857142 :=
by
  use 857142
  unfold six_digit_number
  split
  · norm_num
  · sorry

end original_number_is_857142_l670_670238


namespace sum_of_values_l670_670442

def f (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_of_values (n : ℕ) (hn : n = 2023) :
  (finset.sum (finset.range ((n - 1) / 2)) (λ k, f ((k + 1 : ℝ) / n) - f ((n - k - 1 : ℝ) / n))) = 0 :=
by
  sorry

end sum_of_values_l670_670442


namespace surface_area_circumscribed_sphere_l670_670489

-- Define the problem
theorem surface_area_circumscribed_sphere (a b c : ℝ)
    (h1 : a^2 + b^2 = 3)
    (h2 : b^2 + c^2 = 5)
    (h3 : c^2 + a^2 = 4) : 
    4 * Real.pi * (a^2 + b^2 + c^2) / 4 = 6 * Real.pi :=
by
  -- The proof is omitted
  sorry

end surface_area_circumscribed_sphere_l670_670489


namespace fraction_of_difference_l670_670621

theorem fraction_of_difference (A_s A_l : ℝ) (h_total : A_s + A_l = 500) (h_smaller : A_s = 225) :
  (A_l - A_s) / ((A_s + A_l) / 2) = 1 / 5 :=
by
  -- Proof goes here
  sorry

end fraction_of_difference_l670_670621


namespace equal_angles_of_isosceles_triangle_l670_670097

variable {α : Type}
variable [inner_product_space ℝ α]
variable (A B C D E F : α)
variable (BAC FDE : ℝ)

noncomputable
def isosceles_triangle (A B C : α) : Prop :=
  dist A B = dist B C

noncomputable
def points_on_sides (A B C D E F : α) : Prop :=
  ∃ (t₁ t₂ t₃ : ℝ), (0 ≤ t₁ ∧ t₁ ≤ 1) ∧ (0 ≤ t₂ ∧ t₂ ≤ 1) ∧ (0 ≤ t₃ ∧ t₃ ≤ 1) ∧ 
  D = t₁ • C + (1 - t₁) • A ∧ 
  E = t₂ • B + (1 - t₂) • A ∧ 
  F = t₃ • C + (1 - t₃) • B

noncomputable
def DE_DF_equal (D E F : α) : Prop := dist D E = dist D F

noncomputable
def AE_FC_AC (A E F C : α) : Prop := dist A E + dist F C = dist A C

theorem equal_angles_of_isosceles_triangle 
  (isosceles_triangle A B C) (points_on_sides A B C D E F) 
  (DE_DF_equal D E F) (AE_FC_AC A E F C) : 
  BAC = FDE :=
sorry

end equal_angles_of_isosceles_triangle_l670_670097


namespace solve_g_eq_3_l670_670421

def g (x : ℝ) : ℝ :=
if x < -1 then 4 * x + 8
else if x < 5 then 5 * x - 10
else x * x - 5

theorem solve_g_eq_3 : 
  {x : ℝ | g x = 3} = {-5 / 4, 13 / 5, 2 * Real.sqrt 2} :=
by
  sorry

end solve_g_eq_3_l670_670421


namespace sum_of_terms_l670_670006

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

theorem sum_of_terms (h : ∀ n, S n = n^2) : a 5 + a 6 + a 7 = 33 :=
by
  sorry

end sum_of_terms_l670_670006


namespace integral_cos_cubed_sin_eq_l670_670693

noncomputable def integral_cos_cubed_sin (x : ℝ) : ℝ :=
  ∫ (cos x)^3 * sin x dx

theorem integral_cos_cubed_sin_eq :
  ∀ (x : ℝ), (integral_cos_cubed_sin x) = -((cos x)^4) / 4 + C :=
by
  sorry -- Proof is not required as per the instruction

end integral_cos_cubed_sin_eq_l670_670693


namespace four_distinct_real_roots_l670_670711

theorem four_distinct_real_roots (m : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - 4 * |x| + 5 - m) ∧ ∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x4 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0 ∧ f x4 = 0) ↔ (1 < m ∧ m < 5) :=
by
  sorry

end four_distinct_real_roots_l670_670711


namespace correlation_relationships_l670_670505

def has_correlation (relationship : ℕ) : Prop :=
  relationship = 1 ∨ relationship = 3 ∨ relationship = 4 ∨ relationship = 5

axiom steelmaking : has_correlation 1
axiom point_on_curve : ¬ has_correlation 2
axiom citrus_yield : has_correlation 3
axiom tree_diameter_height : has_correlation 4
axiom age_wealth : has_correlation 5

theorem correlation_relationships : 
  ∀ (r : ℕ), r = 1 ∨ r = 3 ∨ r = 4 ∨ r = 5 → has_correlation r :=
by
  intros r hr
  cases hr
  case or.inl { exact steelmaking }
  case or.inr hr1 {
    cases hr1
    case or.inl { exact citrus_yield }
    case or.inr hr2 {
      cases hr2
      case or.inl { exact tree_diameter_height }
      case or.inr { exact age_wealth }
    }
  }

end correlation_relationships_l670_670505


namespace evaluate_expression_l670_670083

theorem evaluate_expression : 
  let x := -2023 in
  (abs (abs x + x) + abs x + x) = -2023 := 
by
  sorry

end evaluate_expression_l670_670083


namespace simplify_expression_l670_670969

variable (x : ℝ)

theorem simplify_expression : (5 * x + 2 * (4 + x)) = (7 * x + 8) := 
by
  sorry

end simplify_expression_l670_670969


namespace eq_has_infinite_solutions_l670_670306

theorem eq_has_infinite_solutions (b : ℝ) (x : ℝ) :
  5 * (3 * x - b) = 3 * (5 * x + 15) → b = -9 := by
sorry

end eq_has_infinite_solutions_l670_670306


namespace polygon_area_equals_9_l670_670294

-- Define the vertices
def vertex1 := (0, 0)
def vertex2 := (4, 3)
def vertex3 := (6, 0)
def vertex4 := (4, 6)

-- Define the theorem to prove the area of the polygon is 9 square units
theorem polygon_area_equals_9 :
  let vertices := [vertex1, vertex2, vertex3, vertex4],
  -- The Shoelace Theorem calculation goes here which we are abstracting
  -- by stating that the calculation of the area should be 9.
  (polygon_area vertices) = 9 :=
by {
  sorry -- Proof would go here
}

end polygon_area_equals_9_l670_670294


namespace find_k_m_max_f_on_interval_l670_670767

noncomputable def f (k : ℝ) (x : ℝ) := log x + x^2 - k * x

theorem find_k_m : 
    ∃ (k m : ℝ), (k = 4 ∧ m = 2) ∧ 
                let tang_line := (λ x y m => x + y + m = 0) in
                tang_line 1 (f 4 1) 2 := 
by {
    sorry 
}

theorem max_f_on_interval : 
    let k := 4 in
    let max_f := (e : ℝ) => ((exp 1 - 1) ^ 2 : ℝ) in
    (∀ x ∈ set.Icc (1 : ℝ) (exp 1), (f k x) ≤ max_f (exp 1)) :=
by {
    sorry 
}

end find_k_m_max_f_on_interval_l670_670767


namespace problem1_main_problem2_main_l670_670838

noncomputable def problem1 (a b : ℝ) (h1 : a > b) (h2 : b > 0) (P : ℝ × ℝ) (hP : P = (1, 3 / 2))
  (hfoci : ∀ f1 f2 : ℝ × ℝ, dist P f1 + dist P f2 = 4) : Prop :=
let ellipse_eq : ℝ × ℝ → Prop := λ (xy : ℝ × ℝ),
  let (x, y) := xy in
  (x^2) / (a^2) + (y^2) / (b^2) = 1 in
ellipse_eq (1, 3 / 2)

theorem problem1_main : ∃ a b : ℝ, problem1 a b :=
exists.intro 2 (exists.intro (real.sqrt 3) sorry)

def problem2 (M N : ℝ × ℝ) : Prop :=
(M.1 = -1 ∧ M.2 = 3 / 2 ∧ N.1 = -2 ∧ N.2 = 0) ∨
(M.1 = 1 ∧ M.2 = 9 / 2 ∧ N.1 = 2 ∧ N.2 = 6)

theorem problem2_main : ∃ M N : ℝ × ℝ, problem2 M N :=
have h1 : problem2 (-1, 3 / 2) (-2, 0), sorry,
have h2 : problem2 (1, 9 / 2) (2, 6), sorry,
exists.intro (-1, 3 / 2) (exists.intro (-2, 0) (or.inl h1))

end problem1_main_problem2_main_l670_670838


namespace find_xyz_l670_670871

theorem find_xyz (x y z : ℝ)
  (h1 : x > 4)
  (h2 : y > 4)
  (h3 : z > 4)
  (h4 : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 42) :
  (x, y, z) = (11, 9, 7) :=
by {
  sorry
}

end find_xyz_l670_670871


namespace number_of_diagonals_number_of_triangles_l670_670724

theorem number_of_diagonals (n : ℕ) (h : n ≥ 3) :
  let diagonals := n * (n - 3) / 2 in
  -- This is the formula for the number of diagonals in a convex n-gon
  diagonals = (finset.card (finset.filter (λ (p : finset (fin n)), p.card = n - 2) (finset.powerset_univ {i | i < n})) / 2) :=
sorry

theorem number_of_triangles (n : ℕ) (h : n ≥ 6) :
  let triangles := (finset.card (finset.filter (λ (p : finset (fin n)), p.card = 3) (finset.powerset_univ {i | i < n}))) +
                   4 * (finset.card (finset.filter (λ (p : finset (fin n)), p.card = 4) (finset.powerset_univ {i | i < n}))) +
                   5 * (finset.card (finset.filter (λ (p : finset (fin n)), p.card = 5) (finset.powerset_univ {i | i < n}))) +
                   (finset.card (finset.filter (λ (p : finset (fin n)), p.card = 6) (finset.powerset_univ {i | i < n}))) in
  -- This is the formula for the number of triangles formed by the sides and diagonals of a convex n-gon.
  triangles = (nat.choose n 3) + 4 * (nat.choose n 4) + 5 * (nat.choose n 5) + (nat.choose n 6) :=
sorry

end number_of_diagonals_number_of_triangles_l670_670724


namespace a5_a6_val_l670_670396

variable (a : ℕ → ℝ)
variable (r : ℝ)

axiom geom_seq (n : ℕ) : a (n + 1) = a n * r
axiom pos_seq (n : ℕ) : a n > 0

axiom a1_a2 : a 1 + a 2 = 1
axiom a3_a4 : a 3 + a 4 = 9

theorem a5_a6_val :
  a 5 + a 6 = 81 :=
by
  sorry

end a5_a6_val_l670_670396


namespace chord_ratio_l670_670540

theorem chord_ratio (A B C D P : Type) (AP BP CP DP : ℝ)
  (h1 : AP = 4) (h2 : CP = 9)
  (h3 : AP * BP = CP * DP) : BP / DP = 9 / 4 := 
by 
  sorry

end chord_ratio_l670_670540


namespace sqrt_subtraction_l670_670559

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l670_670559


namespace orthogonal_matrix_property_l670_670662

open Matrix

variables {a b c d : ℝ}
def A : Matrix (Fin 2) (Fin 2) ℝ := ![![a, b], ![c, d]]
def I : Matrix (Fin 2) (Fin 2) ℝ := 1

theorem orthogonal_matrix_property (hA : Aᵀ = A⁻¹) (k : ℝ) (hk : k = 1) :
  ((a + 1)^2 + b^2 = 1) ∧ (c^2 + (d + 1)^2 = 1) → (a^2 + b^2 + c^2 + d^2) = 2 :=
by {
  sorry
}

end orthogonal_matrix_property_l670_670662


namespace perfect_squares_100_to_400_l670_670805

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l670_670805


namespace tom_age_ratio_l670_670943

-- Define the variables and conditions
variables (T N : ℕ)

-- Condition 1: Tom's current age is twice the sum of his children's ages
def children_sum_current : ℤ := T / 2

-- Condition 2: Tom's age N years ago was three times the sum of their ages then
def children_sum_past : ℤ := (T / 2) - 2 * N

-- Main theorem statement proving the ratio T/N = 10 assuming given conditions
theorem tom_age_ratio (h1 : T = 2 * (T / 2)) 
                      (h2 : T - N = 3 * ((T / 2) - 2 * N)) : 
                      T / N = 10 :=
sorry

end tom_age_ratio_l670_670943


namespace length_of_YZ_l670_670059

noncomputable def square_area_one (ABCD : set (ℝ × ℝ)) : Prop :=
∃ (s : ℝ), s^2 = 1 ∧ (by { let side_points := (A, B, C, D) ∈ ABCD,
                            sorry -- Proof of square using this side length "s".
                          })

noncomputable def opposite_sides (A X : ℝ × ℝ) (CD : line (ℝ × ℝ)) : Prop :=
A ∈ half_space_left CD ∧ X ∈ half_space_right CD

noncomputable def lines_intersect_CD (AX BX : line (ℝ × ℝ)) (CD : line (ℝ × ℝ)) (YZ : set (ℝ × ℝ)) : Prop :=
∃ Y Z, Y ∈ CD ∧ Z ∈ CD ∧ Y ∈ AX ∧ Z ∈ BX ∧ YZ = {Y, Z}

noncomputable def triangle_area (XYZ : set (ℝ × ℝ)) : ℝ := 
let base := dist (point_1 XYZ) (point_2 XYZ),
    height := dist (point_3 XYZ) (line_span (point_1 XYZ, point_2 XYZ)) 
in 1/2 * base * height

theorem length_of_YZ (ABCD : set (ℝ × ℝ)) (X : ℝ × ℝ) (AX BX : line (ℝ × ℝ)) (YZ : set (ℝ × ℝ)) : 
  square_area_one ABCD →
  opposite_sides A X CD →
  lines_intersect_CD AX BX CD YZ →
  triangle_area XYZ = 2 / 3 →
  dist (point_1 YZ) (point_2 YZ) = 2/3 := 
sorry

end length_of_YZ_l670_670059


namespace approximate_number_of_fish_in_pond_l670_670380

theorem approximate_number_of_fish_in_pond :
  ∃ N : ℕ, N = 800 ∧
  (40 : ℕ) / N = (2 : ℕ) / (40 : ℕ) := 
sorry

end approximate_number_of_fish_in_pond_l670_670380


namespace abs_two_minus_sqrt_five_l670_670914

noncomputable def sqrt_5 : ℝ := Real.sqrt 5

theorem abs_two_minus_sqrt_five : |2 - sqrt_5| = sqrt_5 - 2 := by
  sorry

end abs_two_minus_sqrt_five_l670_670914


namespace combined_value_l670_670813

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end combined_value_l670_670813


namespace find_x_l670_670087

noncomputable def k (x y z : ℝ) : ℝ :=
  x * (y^2) / (z^3)

theorem find_x (y z : ℝ) (k : ℝ) : y = 3 → z = 2 → k = 9/8 → x = k * (z^3) / (y^2) → 
                                 y = 9 → z = 4 → x = 8/9 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  sorry

end find_x_l670_670087


namespace length_AD_l670_670264

noncomputable def radius (C : Circle) (hCirc : C.circumference = 18 * Real.pi) : Real :=
  classical.some (exists_of_circumference C hCirc)

theorem length_AD (C : Circle) (hCirc : C.circumference = 18 * Real.pi)
  (A B D : Point) (hAB : diameter A B) (hD_on_C : point_on_circle D C)
  (angle_ADB : ∠A D B = 45) :
  length_segment A D = 9 * Real.sqrt 2 :=
by
  sorry

end length_AD_l670_670264


namespace solve_for_y_l670_670158

theorem solve_for_y (y : ℝ) : 5 * y - 100 = 125 ↔ y = 45 := by
  sorry

end solve_for_y_l670_670158


namespace sqrt_subtraction_l670_670560

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l670_670560


namespace integral_sin2_cos4_eq_l670_670289

noncomputable def integral_sin2_cos4 (C : ℝ) : ℝ :=
  ∫ x in 0..1, (sin x)^2 * (cos x)^4

theorem integral_sin2_cos4_eq (C : ℝ) :
  integral_sin2_cos4 C = 
  (1 / 16) * (1 - (1 / 12) * (sin (6 * 1)) + (1 / 4) * (sin (2 * 1)) - (1 / 4) * (sin (4 * 1))) + C :=
  sorry

end integral_sin2_cos4_eq_l670_670289


namespace property_holds_for_1_and_4_l670_670146

theorem property_holds_for_1_and_4 (n : ℕ) : 
  (∀ q : ℕ, n % q^2 < q^(q^2) / 2) ↔ (n = 1 ∨ n = 4) :=
by sorry

end property_holds_for_1_and_4_l670_670146


namespace joan_gemstones_l670_670410

def number_of_gemstones (M : ℕ) (M_y : ℕ) (G_y : ℕ) : ℕ :=
  G_y

theorem joan_gemstones
  (M : ℕ)
  (h1 : M = 48)
  (M_y : ℕ)
  (h2 : M_y = M - 6)
  (G_y : ℕ)
  (h3 : G_y = M_y / 2) : 
  number_of_gemstones M M_y G_y = 21 := 
by 
  rw [number_of_gemstones, h1, h2, h3]
  rfl


end joan_gemstones_l670_670410


namespace max_value_y_l670_670722

-- We declare the variables
variable (x : ℝ)

-- We define the function y = sin(x) / (2 - cos(x))
def y (x : ℝ) : ℝ := sin x / (2 - cos x)

-- We need to prove that the maximum value of y for any real x is sqrt(3) / 3
theorem max_value_y : ∀ x : ℝ, y x ≤ sqrt 3 / 3 :=
sorry

end max_value_y_l670_670722


namespace angle_in_third_quadrant_l670_670598

theorem angle_in_third_quadrant (θ : ℤ) (hθ : θ = -510) : 
  (210 % 360 > 180 ∧ 210 % 360 < 270) := 
by
  have h : 210 % 360 = 210 := by norm_num
  sorry

end angle_in_third_quadrant_l670_670598


namespace car_r_speed_l670_670581

variable (v : ℝ)

theorem car_r_speed (h1 : (300 / v - 2 = 300 / (v + 10))) : v = 30 := 
sorry

end car_r_speed_l670_670581


namespace digit_B_divisible_by_9_l670_670504

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_l670_670504


namespace boat_speed_of_stream_l670_670205

theorem boat_speed_of_stream :
  ∀ (x : ℝ), 
    (∀ s_b : ℝ, s_b = 18) → 
    (∀ d1 d2 : ℝ, d1 = 48 → d2 = 32 → d1 / (18 + x) = d2 / (18 - x)) → 
    x = 3.6 :=
by 
  intros x h_speed h_distance
  sorry

end boat_speed_of_stream_l670_670205


namespace rhombus_longest_diagonal_l670_670629

noncomputable def length_of_longest_diagonal (area : ℝ) (ratio : ℝ) : ℝ :=
  let x := real.sqrt (area * 2 / (ratio * 3 * 5)) in
  5 * x

theorem rhombus_longest_diagonal (area : ℝ) (ratio : ℝ) (h_area : area = 150) (h_ratio : ratio = 5/3) :
  length_of_longest_diagonal area ratio = 10 * real.sqrt 5 :=
by
  rw [h_area, h_ratio]
  unfold length_of_longest_diagonal
  sorry

end rhombus_longest_diagonal_l670_670629


namespace number_of_ways_to_distribute_balls_l670_670353

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l670_670353


namespace range_of_k_l670_670138

noncomputable def g (x : ℝ) := (1 - Real.log x) / (x^2)

theorem range_of_k
  (f_increasing : ∀ x > 0, ∀ y > x, (Real.log x) / x - k * x ≤ (Real.log y) / y - k * y)
  : k ≤ -1 / (2 * Real.exp 3) :=
begin
  sorry
end

end range_of_k_l670_670138


namespace calculate_salary_l670_670227

-- Define the constants and variables
def food_percentage : ℝ := 0.35
def rent_percentage : ℝ := 0.25
def clothes_percentage : ℝ := 0.20
def transportation_percentage : ℝ := 0.10
def recreational_percentage : ℝ := 0.15
def emergency_fund : ℝ := 3000
def total_percentage : ℝ := food_percentage + rent_percentage + clothes_percentage + transportation_percentage + recreational_percentage

-- Define the salary
def salary (S : ℝ) : Prop :=
  (total_percentage - 1) * S = emergency_fund

-- The theorem stating the salary is 60000
theorem calculate_salary : ∃ S : ℝ, salary S ∧ S = 60000 :=
by
  use 60000
  unfold salary total_percentage
  sorry

end calculate_salary_l670_670227


namespace prove_m_add_n_l670_670360

-- Definitions from conditions
variables (m n : ℕ)

def condition1 : Prop := m + 1 = 3
def condition2 : Prop := m = n - 1

-- Statement to prove
theorem prove_m_add_n (h1 : condition1 m) (h2 : condition2 m n) : m + n = 5 := 
sorry

end prove_m_add_n_l670_670360


namespace total_books_on_shelves_l670_670938

theorem total_books_on_shelves 
  (num_bookshelves : ℕ) 
  (num_floors : ℕ) 
  (books_left : ℕ) 
  (books_removed : ℕ) :
  num_bookshelves = 28 →
  num_floors = 6 →
  books_left = 20 →
  books_removed = 2 →
  (num_bookshelves * num_floors * (books_left + books_removed)) = 3696 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num

end total_books_on_shelves_l670_670938


namespace find_abc_l670_670212

theorem find_abc (a b c : ℝ) 
  (h1 : 2 * b = a + c)  -- a, b, c form an arithmetic sequence
  (h2 : a + b + c = 12) -- The sum of a, b, and c is 12
  (h3 : (b + 2)^2 = (a + 2) * (c + 5)) -- a+2, b+2, and c+5 form a geometric sequence
: (a = 1 ∧ b = 4 ∧ c = 7) ∨ (a = 10 ∧ b = 4 ∧ c = -2) :=
sorry

end find_abc_l670_670212


namespace andrea_fewer_apples_l670_670966

theorem andrea_fewer_apples {total_apples given_to_zenny kept_by_yanna given_to_andrea : ℕ} 
  (h1 : total_apples = 60) 
  (h2 : given_to_zenny = 18) 
  (h3 : kept_by_yanna = 36) 
  (h4 : given_to_andrea = total_apples - kept_by_yanna - given_to_zenny) : 
  (given_to_andrea + 12 = given_to_zenny) := 
sorry

end andrea_fewer_apples_l670_670966


namespace households_neither_brand_A_nor_B_l670_670228

noncomputable def households_used_neither (total_surveyed : ℕ) 
  (brand_A_only : ℕ) 
  (both_brands : ℕ) 
  (brand_B_ratio : ℕ) : ℕ :=
total_surveyed - (brand_A_only + (brand_B_ratio * both_brands) + both_brands)

theorem households_neither_brand_A_nor_B (total_surveyed : ℕ) 
  (brand_A_only : ℕ) 
  (both_brands : ℕ) 
  (brand_B_ratio : ℕ)
  (h_total_surveyed : total_surveyed = 260)
  (h_brand_A_only : brand_A_only = 60)
  (h_both_brands  : both_brands = 30)
  (h_brand_B_ratio : brand_B_ratio = 3) : 
  households_used_neither total_surveyed brand_A_only both_brands brand_B_ratio = 80 :=
by {
  simp [households_used_neither, h_total_surveyed, h_brand_A_only, h_both_brands, h_brand_B_ratio],
  sorry
}

end households_neither_brand_A_nor_B_l670_670228


namespace piecewiseFunc_max_val_l670_670512

def piecewiseFunc (x : ℝ) : ℝ :=
  if x < 1 then x + 3 else -x + 6

theorem piecewiseFunc_max_val : ∃ x : ℝ, piecewiseFunc x = 5 ∧ (∀ y, piecewiseFunc y ≤ 5) :=
by 
  sorry

end piecewiseFunc_max_val_l670_670512


namespace platform_length_l670_670600

theorem platform_length
    (train_length : ℕ)
    (time_to_cross_tree : ℕ)
    (speed : ℕ)
    (time_to_pass_platform : ℕ)
    (platform_length : ℕ) :
    train_length = 1200 →
    time_to_cross_tree = 120 →
    speed = train_length / time_to_cross_tree →
    time_to_pass_platform = 150 →
    speed * time_to_pass_platform = train_length + platform_length →
    platform_length = 300 :=
by
  intros h_train_length h_time_to_cross_tree h_speed h_time_to_pass_platform h_pass_platform_eq
  sorry

end platform_length_l670_670600


namespace laundry_loads_l670_670714

theorem laundry_loads (usual_price : ℝ) (sale_price : ℝ) (cost_per_load : ℝ) (total_loads_2_bottles : ℝ) :
  usual_price = 25 ∧ sale_price = 20 ∧ cost_per_load = 0.25 ∧ total_loads_2_bottles = (2 * sale_price) / cost_per_load →
  (total_loads_2_bottles / 2) = 80 :=
by
  sorry

end laundry_loads_l670_670714


namespace solution_function_identity_l670_670694

def satisfies_identity (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, nat.iterate f (x + 1) y + nat.iterate f (y + 1) x = 2 * f (x + y)

theorem solution_function_identity (f : ℕ → ℕ) :
  satisfies_identity f → ∀ n : ℕ, f (f n) = f (n + 1) :=
by
  intro h
  sorry

end solution_function_identity_l670_670694


namespace valid_lecture_orders_l670_670626

theorem valid_lecture_orders :
  let total_orders := 7!
  let dependencies := 4
  total_orders / dependencies = 1260 :=
by
  sorry

end valid_lecture_orders_l670_670626


namespace avg_days_studied_is_correct_l670_670454

def num_students_studied_days (studies : List (ℕ × ℕ)) : ℕ :=
  studies.foldl (λ s t => s + t.1) 0

def total_days_studied (studies : List (ℕ × ℕ)) : ℕ :=
  studies.foldl (λ s t => s + t.1 * t.2) 0

def mean_days_studied (studies : List (ℕ × ℕ)) : ℝ :=
  total_days_studied studies / num_students_studied_days studies

theorem avg_days_studied_is_correct (studies : List (ℕ × ℕ)) (h : studies = [(2, 3), (4, 5), (9, 8), (5, 10), (3, 15), (2, 20)]) :
  mean_days_studied studies = 9.32 :=
by
  sorry

end avg_days_studied_is_correct_l670_670454


namespace percent_of_sum_l670_670011

theorem percent_of_sum (z y x w v : ℝ)
  (h1 : 0.45 * z = 0.72 * y)
  (h2 : y = 0.75 * x)
  (h3 : w = 0.60 * z^2)
  (h4 : z = 0.30 * w^(1/3))
  (h5 : v = 0.80 * x^(1/2)) :
  (z / (x + v)) * 100 ≈ 14.86 :=
by sorry

end percent_of_sum_l670_670011


namespace sum_of_possible_x_l670_670116

theorem sum_of_possible_x (x : ℂ) :
  (3:ℂ)^(x^2 + 4 * x + 4) = (27:ℂ)^(x + 1) →
  (x = (-1 + complex.I * real.sqrt 3) / 2 ∨ x = (-1 - complex.I * real.sqrt 3) / 2) →
  (complex.re ( (-1 + complex.I * real.sqrt 3) / 2 ) + complex.re ( (-1 - complex.I * real.sqrt 3) / 2 ) = -1) :=
by sorry

end sum_of_possible_x_l670_670116


namespace find_b_find_a_l670_670770

-- Definitions for the conditions
def f (x : ℝ) (b : ℝ) : ℝ :=
  (2 / 3) * ((2^x) / (2^x - 1) + b)

def g (x : ℝ) (a : ℝ) : ℝ :=
  Real.log 2 ((a - 1) * x^2 - 2 * x + 1)

-- Statement to prove b given condition of f being odd
theorem find_b (b : ℝ) :
  (∀ x : ℝ, f (-x) b = -f x b) → b = -1 / 2 :=
sorry

-- Statement to prove range for the real number a given the condition
theorem find_a (a : ℝ) :
  (∃ x₁ ∈ Icc 1 2, ∀ x₂ ∈ Icc 1 2, f x₁ (-1/2) ≥ g x₂ a) →
  2 < a ∧ a ≤ 9 / 4 :=
sorry

end find_b_find_a_l670_670770


namespace smallest_four_digit_divisible_by_55_l670_670186

theorem smallest_four_digit_divisible_by_55 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 55 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 55 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_55_l670_670186


namespace all_terms_are_integers_l670_670236

def seq : ℕ → ℚ
| 0 := 1
| 1 := 1
| 2 := 1
| 3 := 1
| (n+4) := (seq (n+1) * seq n) / (2 * seq (n+4)) + 1 / 2

theorem all_terms_are_integers : ∀ n, (seq n).denom = 1 := 
sorry

end all_terms_are_integers_l670_670236


namespace valid_selling_price_l670_670033

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end valid_selling_price_l670_670033


namespace sum_primes_gt_10_lt_20_l670_670957

def is_prime (n : ℕ) : Prop :=
  1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_primes_in_range (a b : ℕ) : ℕ :=
  ∑ n in finset.filter (λ x, is_prime x) (finset.Ico a b), n

theorem sum_primes_gt_10_lt_20 : sum_primes_in_range 10 20 = 60 :=
  sorry

end sum_primes_gt_10_lt_20_l670_670957


namespace quadratic_solution_l670_670484

theorem quadratic_solution : ∀ x : ℝ, (x^2 + 6*x + 2 = 0) ↔ (x = -3 + sqrt 7 ∨ x = -3 - sqrt 7) :=
by
  intro x
  -- the solution follows from proving both implications
  sorry

end quadratic_solution_l670_670484


namespace eighth_term_matchstick_count_l670_670782

def matchstick_sequence (n : ℕ) : ℕ := (n + 1) * 3

theorem eighth_term_matchstick_count : matchstick_sequence 8 = 27 :=
by
  -- the proof will go here
  sorry

end eighth_term_matchstick_count_l670_670782


namespace discount_ratio_l670_670992

def bill_amount := 110
def original_discount := 10
def longer_time_discount := 18.33

theorem discount_ratio :
  (longer_time_discount / original_discount) = 1.833 :=
by
  sorry

end discount_ratio_l670_670992


namespace train_speed_is_100_kmph_l670_670636

noncomputable def speed_of_train (length_of_train : ℝ) (time_to_cross_pole : ℝ) : ℝ :=
  (length_of_train / time_to_cross_pole) * 3.6

theorem train_speed_is_100_kmph :
  speed_of_train 100 3.6 = 100 :=
by
  sorry

end train_speed_is_100_kmph_l670_670636


namespace ABCD_concyclic_l670_670390

-- Define the conditions and construct the appropriate geometrical setups.
variables {A B C D E F M N : Point}
variables {α β : Angle}

-- Define the points and angles
variables (triangle_ABC : acute_angled_triangle A B C)
variables (E_on_BC : point_on_line E B C)
variables (F_on_BC : point_on_line F B C)
variables (angle_EAB_eq_ACB : angle E A B = angle A C B)
variables (angle_CAF_eq_ABC : angle C A F = angle A B C)
variables (E_mid_AM : midpoint E A M)
variables (F_mid_AN : midpoint F A N)
variables (D_intersection : intersect_lines_at_point (line_through B M) (line_through C N) D)

-- Prove that points A, B, D, and C are concyclic
theorem ABCD_concyclic : concyclic_points A B C D :=
by
  -- Here, you will fill in the proof details
  sorry

end ABCD_concyclic_l670_670390


namespace combined_value_of_a_and_b_l670_670810

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end combined_value_of_a_and_b_l670_670810


namespace mod_equiv_example_l670_670698

theorem mod_equiv_example : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 11) ∧ (n ≡ -5033 [MOD 12]) ∧ (n = 7) :=
by
  use 7
  split
  · exact ⟨by norm_num, by norm_num⟩
  · split
    · show 7 ≡ -5033 [MOD 12]
      sorry
    · rfl

end mod_equiv_example_l670_670698


namespace angle_B_is_pi_div_3_or_2pi_div_3_l670_670026

theorem angle_B_is_pi_div_3_or_2pi_div_3
  (A B : ℝ)
  (h1 : sin A + cos A = sqrt 2)
  (h2 : sqrt 3 * cos A = -sqrt 2 * cos (π/2 + B)) :
  B = π/3 ∨ B = 2*π/3 :=
sorry

end angle_B_is_pi_div_3_or_2pi_div_3_l670_670026


namespace lara_swimming_l670_670055

theorem lara_swimming (minutes_per_day1 : ℕ := 80) 
                      (days1 : ℕ := 6) 
                      (minutes_per_day2 : ℕ := 105) 
                      (days2 : ℕ := 2) 
                      (target_average : ℕ := 100) 
                      (total_days : ℕ := 9) : 
                      let day_ninth := 210 in
                      day_ninth = target_average * total_days - (minutes_per_day1 * days1 + minutes_per_day2 * days2) :=
by
  sorry

end lara_swimming_l670_670055


namespace correct_parentheses_make_equation_true_l670_670844

def equation_with_parentheses (a b c d : ℝ) : ℝ :=
  ((a + b) / c + d) / c

theorem correct_parentheses_make_equation_true : equation_with_parentheses 0.5 0.5 0.5 0.5 = 5 :=
  by
    sorry

end correct_parentheses_make_equation_true_l670_670844


namespace fifty_day_year_m_minus_1_l670_670842

def day_of_week : Type := 
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

def day_of_week_add_days (start_day: day_of_week) (days: ℕ) : day_of_week :=
  match start_day with
  | day_of_week.Monday    => day_of_week.nth (days % 7)
  | day_of_week.Tuesday   => day_of_week.nth ((days + 1) % 7)
  | day_of_week.Wednesday => day_of_week.nth ((days + 2) % 7)
  | day_of_week.Thursday  => day_of_week.nth ((days + 3) % 7)
  | day_of_week.Friday    => day_of_week.nth ((days + 4) % 7)
  | day_of_week.Saturday  => day_of_week.nth ((days + 5) % 7)
  | day_of_week.Sunday    => day_of_week.nth ((days + 6) % 7)

theorem fifty_day_year_m_minus_1 
  (M : ℕ) 
  (h1 : day_of_week_add_days day_of_week.Friday 249 = day_of_week.Friday)
  (h2 : day_of_week_add_days day_of_week.Friday 149 = day_of_week.Friday) : 
  day_of_week_add_days day_of_week.Friday 314 = day_of_week.Wednesday :=
by
  sorry

end fifty_day_year_m_minus_1_l670_670842


namespace recurrence_solution_proof_l670_670487

noncomputable def recurrence_relation (a : ℕ → ℚ) : Prop :=
  (∀ n ≥ 2, a n = 5 * a (n - 1) - 6 * a (n - 2) + n + 2) ∧
  a 0 = 27 / 4 ∧
  a 1 = 49 / 4

noncomputable def solution (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = 3 * 2^n + 3^n + n / 2 + 11 / 4

theorem recurrence_solution_proof : ∃ a : ℕ → ℚ, recurrence_relation a ∧ solution a :=
by { sorry }

end recurrence_solution_proof_l670_670487


namespace sqrt_difference_l670_670556

theorem sqrt_difference:
  sqrt (49 + 81) - sqrt (36 - 9) = sqrt 130 - 3 * sqrt 3 :=
by
  sorry

end sqrt_difference_l670_670556


namespace perfect_squares_between_100_and_400_l670_670798

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l670_670798


namespace translation_by_1_right_and_1_up_l670_670167

open Real

noncomputable def f (x : ℝ) : ℝ := log 3 x

noncomputable def h (x : ℝ) : ℝ := f (x - 1) + 1

theorem translation_by_1_right_and_1_up (x : ℝ) : h x = log 3 (3 * x - 3) :=
by
  sorry

end translation_by_1_right_and_1_up_l670_670167


namespace log_rule_subtraction_l670_670195

theorem log_rule_subtraction : log 2 6 - log 2 3 = 1 := 
by {
  sorry
}

end log_rule_subtraction_l670_670195


namespace compare_neg_fractions_l670_670265

theorem compare_neg_fractions : - (3 / 5 : ℚ) < - (1 / 5 : ℚ) :=
by
  sorry

end compare_neg_fractions_l670_670265


namespace incorrect_statements_AB_l670_670197

-- Definitions and conditions from the problem
def domain_condition_1 (f : ℝ → ℝ) : set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def range_condition_2 (f : ℝ → ℝ) (a b : ℝ) : set ℝ := {y | a ≤ y ∧ y ≤ b}
def odd_function_condition_3 (f : ℝ → ℝ) : Prop := 
  ∀ x, f(2*x+1) - (2^x - 1)/(2^x + 1) = -(f(2*(-x)+1) - (2^(-x) - 1)/(2^(-x) + 1))
def monotonically_increasing_condition_4 (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f(x) ≤ f(y)

-- Statement to verify the final answer
theorem incorrect_statements_AB
  (f : ℝ → ℝ)
  (a b : ℝ)
  (h1 : ∀ x, x ∈ domain_condition_1 f)
  (h2 : ∀ x, x ∈ range_condition_2 f a b)
  (h3 : odd_function_condition_3 f)
  (h4 : monotonically_increasing_condition_4 f) :
  (¬(∀ x : ℝ, x ∈ {x | 2 ≤ x ∧ x ≤ 3} → x ∈ domain_condition_1 (λ x, f (x - 1)))) ∧
  (¬(∀ y : ℝ, y ∈ range_condition_2 (λ x, f (Real.sin x)) a b)) := sorry

end incorrect_statements_AB_l670_670197


namespace perfect_squares_between_100_and_400_l670_670800

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l670_670800


namespace num_correct_statements_l670_670642

-- Define each statement as a proposition
def statement1 := ∀ (r : ℝ), ∀ (chord : ℝ), (radius_perpendicular_to_chord r chord = false)
def statement2 := ∀ (p : Parallelogram), (circumscribed_parallelogram_is_rhombus p = true)
def statement3 := ∀ (p : Parallelogram), (inscribed_parallelogram_is_rectangle p = true)
def statement4 := ∀ (q : Quadrilateral), (opposite_angles_cyclic_quad_supplementary q = true)
def statement5 := ∀ (arc1 arc2 : Arc), (arcs_of_equal_length_are_equal arc1 arc2 = false)
def statement6 := ∀ (th1 th2 : Angle), (arcs_equal_central_angles_equal th1 th2 = false)

-- Define a function to count the number of correct statements
def count_correct_statements : Nat :=
  [statement1, statement2, statement3, statement4, statement5, statement6].count (λ stmt, stmt = true)

-- The theorem to prove
theorem num_correct_statements : count_correct_statements = 3 :=
by
  sorry

end num_correct_statements_l670_670642


namespace sufficient_condition_for_unique_zero_l670_670271

theorem sufficient_condition_for_unique_zero (a : ℝ) :
  (a ≤ 0 ∨ a > 1) → 
  (∀ f,
    (∀ x : ℝ, 
      (0 < x → f x = Real.log 2 x) ∧ 
      (x ≤ 0 → f x = -Real.exp x + a)) →
    (∃! x : ℝ, f x = 0)) :=
by
  sorry

end sufficient_condition_for_unique_zero_l670_670271


namespace area_of_triangle_with_median_sides_l670_670057

theorem area_of_triangle_with_median_sides (A B C : Type) [metric_space A] 
[metric_space B] [metric_space C] (area_ABC : ℝ) (h : area_ABC = 92) :
  ∃ (D E F : Type) [metric_space D] [metric_space E] [metric_space F], 
  let area_DEF := (3 / 4) * area_ABC in area_DEF = 69 :=
by
  sorry

end area_of_triangle_with_median_sides_l670_670057


namespace combined_value_l670_670812

theorem combined_value (a b : ℝ) (h1 : 0.005 * a = 95 / 100) (h2 : b = 3 * a - 50) : a + b = 710 := by
  sorry

end combined_value_l670_670812


namespace bus_driver_hours_l670_670604

theorem bus_driver_hours (H : ℕ) : 
  let R := 20
  let OT := R + (3 / 4) * R
  let Hr := if H > 40 then 40 else H
  let Ho := if H > 40 then H - 40 else 0
  let total_comp := if H <= 40 then H * R else (Hr * R) + (Ho * OT)
  (total_comp = 1000) → (H = 46) :=
begin
  /- proof steps -/
  sorry
end

end bus_driver_hours_l670_670604


namespace ordered_pairs_count_l670_670000

theorem ordered_pairs_count :
  {p : ℕ × ℕ // 4 ≤ p.1 ∧ p.1 ≤ p.2 ∧ p.2 ≤ 2019 ∧ p.1 + p.2 = (p.1 - p.2)^2}.to_finset.card = 60 := 
by
  sorry

end ordered_pairs_count_l670_670000


namespace min_val_exponential_l670_670641

theorem min_val_exponential (x : ℝ) : ∃ y, (y = exp x + 4 * exp (-x)) ∧ (∀ z, z = exp x + 4 * exp (-x) → y ≤ z) ∧ y = 4 :=
by
  sorry

end min_val_exponential_l670_670641


namespace perfect_squares_in_range_100_400_l670_670784

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l670_670784


namespace projectile_reaches_24_meters_l670_670497

theorem projectile_reaches_24_meters (h : ℝ) (t : ℝ) (v₀ : ℝ) :
  (h = -4.9 * t^2 + 19.6 * t) ∧ (h = 24) → t = 4 :=
by
  intros
  sorry

end projectile_reaches_24_meters_l670_670497


namespace price_per_ticket_is_six_l670_670851

-- Definition of the conditions
def total_tickets (friends_tickets extra_tickets : ℕ) : ℕ :=
  friends_tickets + extra_tickets

def total_cost (tickets price_per_ticket : ℕ) : ℕ :=
  tickets * price_per_ticket

-- Given conditions
def friends_tickets : ℕ := 8
def extra_tickets : ℕ := 2
def total_spent : ℕ := 60

-- Formulate the problem to prove the price per ticket
theorem price_per_ticket_is_six :
  ∃ (price_per_ticket : ℕ), price_per_ticket = 6 ∧ 
  total_cost (total_tickets friends_tickets extra_tickets) price_per_ticket = total_spent :=
by
  -- The proof is not required; we assume its correctness here.
  sorry

end price_per_ticket_is_six_l670_670851


namespace part1_A_inter_B_and_union_A_B_part2_range_of_a_l670_670447

open Set

-- Define sets A and B under given conditions
def set_A : Set ℝ := { x | 2 * x^2 - 7 * x + 3 ≤ 0 }
def set_B (a : ℝ) : Set ℝ := { x | x^2 + a < 0 }

theorem part1_A_inter_B_and_union_A_B :
  set_B (-4) ∩ set_A = {x | (1/2 : ℝ) ≤ x ∧ x < 2} ∧
  set_B (-4) ∪ set_A = {x | -2 < x ∧ x ≤ 3} := sorry

theorem part2_range_of_a :
  {a : ℝ | (∅ ∩ set_A = (∅ : Set ℝ)) ∧ (∀ x, x ∈ (set_B a) → x ∈ (set_Aᶜ))} = {a | a ≥ -1/4} := sorry

end part1_A_inter_B_and_union_A_B_part2_range_of_a_l670_670447


namespace tan_alpha_value_l670_670717

theorem tan_alpha_value (α : ℝ) (h1 : cos (π + α) = - (sqrt 10 / 5)) (h2 : α ∈ set.Ioo (-π / 2) 0) : 
  tan α = - (sqrt 6 / 2) :=
by
  sorry

end tan_alpha_value_l670_670717


namespace length_of_bridge_l670_670634

def train_length : ℝ := 110
def train_speed_km_per_hr : ℝ := 72
def crossing_time_sec : ℝ := 12.598992080633549

def train_speed_m_per_sec : ℝ := (train_speed_km_per_hr * 1000) / 3600

def total_distance_covered : ℝ := train_speed_m_per_sec * crossing_time_sec

def bridge_length : ℝ := total_distance_covered - train_length

theorem length_of_bridge :
  bridge_length = 141.97984161267098 :=
begin
  sorry
end

end length_of_bridge_l670_670634


namespace Monotonicity_of_f_sum_of_zeros_gt_one_l670_670329

def f (x : ℝ) : ℝ := Real.ln x + 1 / (2 * x)

def g (x : ℝ) (m : ℝ) : ℝ := f x - m

theorem Monotonicity_of_f :
  (∀ x : ℝ, 0 < x → x < 1 / 2 → (f' x : ℝ) < 0) ∧
  (∀ x : ℝ, x > 1 / 2 → (f' x : ℝ) > 0) :=
sorry

theorem sum_of_zeros_gt_one (m : ℝ) (x₁ x₂ : ℝ) (h₁ : x₁ < x₂)
  (h₂ : g x₁ m = 0) (h₃ : g x₂ m = 0) : x₁ + x₂ > 1 :=
sorry

end Monotonicity_of_f_sum_of_zeros_gt_one_l670_670329


namespace max_prime_p_l670_670299

theorem max_prime_p (a b : ℕ) (p : ℕ) (hp : p.prime)
  (h : p = b / 2 * real.sqrt((a - b) / (a + b)))
  (ha : a ≠ 0)
  (hb : b ≠ 0) : 
  p = 5 :=
sorry

end max_prime_p_l670_670299


namespace num_ordered_triples_of_divisors_l670_670702

/-
Auxiliary function to count nonnegative integer solutions less than or equal to n
-/
def count_solutions_le (n k : ℕ) : ℕ :=
  (Finset.range (n + 2)).cardPowersetLen k / (nat.factorial k * nat.factorial (n + 1 - k))

theorem num_ordered_triples_of_divisors (p q r : ℕ) 
  (hp : p = 3) (hq : q = 2) (hr : r = 1) :
  ∑ (a b c : ℕ) in finset.range' (p+1) \times finset.range' (q+1) \times finset.range' (r+1),
  (count_solutions_le p 3) * (count_solutions_le q 3) * (count_solutions_le r 3) = 800 := 
sorry

end num_ordered_triples_of_divisors_l670_670702


namespace part1_part2_l670_670331

def f (x : ℝ) := |x + 2|

theorem part1 (x : ℝ) : 2 * f x < 4 - |x - 1| ↔ -7/3 < x ∧ x < -1 :=
by sorry

theorem part2 (a : ℝ) (m n : ℝ) (hmn : m + n = 1) (hm : 0 < m) (hn : 0 < n) :
  (∀ x, |x - a| - f x ≤ 1/m + 1/n) ↔ -6 ≤ a ∧ a ≤ 2 :=
by sorry

end part1_part2_l670_670331


namespace distinct_patterns_in_4x4_grid_l670_670344

theorem distinct_patterns_in_4x4_grid :
  let grid := (list.list (Fin 4)) × (list.list (Fin 4))
  ∃ (patterns : Finset grid), 
    (∀ x ∈ patterns, (∃ p, (p permutations preserving symmetry x))) →
    patterns.card = 14 :=
sorry

end distinct_patterns_in_4x4_grid_l670_670344


namespace number_of_parallel_or_perpendicular_pairs_is_one_l670_670661

theorem number_of_parallel_or_perpendicular_pairs_is_one :
  let m1 := 2
  let m2 := 3 / 2
  let m3 := 4
  let m4 := -2
  let m5 := 2
  (count (λ (pair : ℕ × ℕ), pair ∈ [(1, 2), (1, 3), (1, 4), (1,5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
                           ∧ (m1 = m2 ∧ pair = (1, 2) ∨ m1 = m3 ∧ pair = (1, 3) ∨ m1 = m4 ∧ pair = (1, 4) ∨ m1 = m5 ∧ pair = (1, 5) ∨ m2 = m3 ∧ pair = (2, 3) ∨ m2 = m4 ∧ pair = (2, 4) ∨ m2 = m5 ∧ pair = (2, 5) ∨ m3 = m4 ∧ pair = (3, 4) ∨ m3 = m5 ∧ pair = (3, 5) ∨ m4 = m5 ∧ pair = (4, 5))
                          ∨  m1 * m2 = -1 ∧ pair = (1, 2) ∨ m1 * m3 = -1 ∧ pair = (1, 3) ∨ m1 * m4 = -1 ∧ pair = (1, 4) ∨ m1 * m5 = -1 ∧ pair = (1, 5) ∨ m2 * m3 = -1 ∧ pair = (2, 3) ∨ m2 * m4 = -1 ∧ pair = (2, 4) ∨ m2 * m5 = -1 ∧ pair = (2, 5) ∨ m3 * m4 = -1 ∧ pair = (3, 4) ∨ m3 * m5 = -1 ∧ pair = (3, 5) ∨ m4 * m5 = -1 ∧ pair = (4, 5))) = 1 :=
by sorry

end number_of_parallel_or_perpendicular_pairs_is_one_l670_670661


namespace rhombus_area_l670_670386

theorem rhombus_area (A B C D E F G H: ℝ) (side: ℝ)
  (H_square: side = 4)
  (H_midpoints: E = 2 ∧ F = 2)
  (H_coords: (A, B, C, D) = ((0,0), (4,0), (4,4), (0,4))) :
  ∃ area: ℝ, area = 4  :=
by {
  sorry,
}

end rhombus_area_l670_670386


namespace functions_not_equal_l670_670671

def f (t : ℝ) : ℝ := 130 * t - 5 * t^2
def g (x : ℝ) : ℝ := 130 * x - 5 * x^2

theorem functions_not_equal : (∀ t : ℝ, 0 ≤ t → f t = g t) → false :=
begin
  assume h,
  have ex1 : ∃ x : ℝ, x < 0 := by { use -1, linarith },
  cases ex1 with x hx,
  have : ¬ (0 ≤ x) := by { linarith },
  have eqng := h x this,
  contradiction,
end

end functions_not_equal_l670_670671


namespace soccer_team_lineup_count_l670_670240

theorem soccer_team_lineup_count :
  let n := 18 in
  let goalkeeper_choices := 18 in
  let center_backs_choices := Nat.choose 17 2 * 2! in
  let left_back_choices := 15 in
  let right_back_choices := 14 in
  let midfielders_choices := Nat.choose 13 3 * 3! in
  goalkeeper_choices * center_backs_choices * left_back_choices * right_back_choices * midfielders_choices = 95_414_400 :=
by
  let n := 18
  let goalkeeper_choices := 18
  let center_backs_choices := Nat.choose 17 2 * 2!
  let left_back_choices := 15
  let right_back_choices := 14
  let midfielders_choices := Nat.choose 13 3 * 3!
  sorry

end soccer_team_lineup_count_l670_670240


namespace new_volume_of_cylinder_l670_670021

theorem new_volume_of_cylinder 
  (r h : ℝ) 
  (original_volume : ℝ) 
  (h1 : original_volume = π * r^2 * h) 
  (h2 : original_volume = 10) : 
  let new_radius := 3 * r
  let new_height := 4 * h
  let new_volume := 36 * π * r^2 * h in
  new_volume = 360 :=
by 
  sorry

end new_volume_of_cylinder_l670_670021


namespace num_perfect_squares_l670_670797

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l670_670797


namespace negation_of_exists_sin_gt_one_l670_670145

theorem negation_of_exists_sin_gt_one :
  (¬ ∃ x : ℝ, sin x > 1) ↔ (∀ x : ℝ, sin x ≤ 1) :=
by
  sorry

end negation_of_exists_sin_gt_one_l670_670145


namespace exists_infinite_consecutive_composite_numbers_l670_670104

theorem exists_infinite_consecutive_composite_numbers (n : ℕ) (h : 1 ≤ n) :
  ∃ (k : ℕ → ℕ), (∀ m, k m = (n+1)! + (m+2) → 2 ≤ m ∧ m ≤ n+1 ∧ ¬Prime (k m)) :=
by
  sorry

end exists_infinite_consecutive_composite_numbers_l670_670104


namespace smallest_positive_integer_y_l670_670554

theorem smallest_positive_integer_y
  (y : ℕ)
  (h_pos : 0 < y)
  (h_ineq : y^3 > 80) :
  y = 5 :=
sorry

end smallest_positive_integer_y_l670_670554


namespace fraction_sum_l670_670690

theorem fraction_sum : (1 / 3 : ℚ) + (2 / 7) + (3 / 8) = 167 / 168 := by
  sorry

end fraction_sum_l670_670690


namespace calculate_actual_distance_l670_670890

-- Definitions corresponding to the conditions
def map_scale : ℕ := 6000000
def map_distance_cm : ℕ := 5

-- The theorem statement corresponding to the proof problem
theorem calculate_actual_distance :
  (map_distance_cm * map_scale / 100000) = 300 := 
by
  sorry

end calculate_actual_distance_l670_670890


namespace max_radius_squared_of_sphere_within_cones_l670_670946

theorem max_radius_squared_of_sphere_within_cones :
  let r := (4 * (1 - 4 / Real.sqrt 116)) in
  r^2 = 8704 / 29 :=
by 
  sorry

end max_radius_squared_of_sphere_within_cones_l670_670946


namespace product1_trailing_zeros_product2_trailing_zeros_l670_670356

-- Define the products and their results
def product1 : ℕ := 125 * 8
def product2 : ℕ := 1350 * 2

-- Define a function to count the number of trailing zeros in a number
-- This is not the proof but a definition to help in the proof later
def trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  (n.to_digits 10).reverse.take_while (λ d, d = 0).length

-- Statement for the number of trailing zeros in product1
theorem product1_trailing_zeros : trailing_zeros product1 = 3 :=
by sorry

-- Statement for the number of trailing zeros in product2
theorem product2_trailing_zeros : trailing_zeros product2 = 2 :=
by sorry

end product1_trailing_zeros_product2_trailing_zeros_l670_670356


namespace f_is_periodic_f_expression_2_4_f_sum_2008_l670_670075

-- Definitions of conditions
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f(x)

def periodic (f : ℝ → ℝ) (P : ℝ) : Prop := ∀ x : ℝ, f (x + P) = f(x)

def f_condition1 (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (x + 2) = -f(x)

def f_initial (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f(x) = 2 * x + x^2

-- Statement for part 1
theorem f_is_periodic (f : ℝ → ℝ) (h_odd : is_odd f) (h_cond1 : f_condition1 f) : periodic f 4 := 
sorry

-- Statement for part 2
theorem f_expression_2_4 (f : ℝ → ℝ) (h_odd : is_odd f) (h_cond1 : f_condition1 f) (h_init : f_initial f) :
  ∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → f(x) = -x^2 + 6 * x - 8 :=
sorry

-- Statement for part 3
theorem f_sum_2008 (f : ℝ → ℝ) (h_odd : is_odd f) (h_cond1 : f_condition1 f) (h_periodic : periodic f 4) : 
  ∑ k in finset.range 2009, f k = 2008 := 
sorry

end f_is_periodic_f_expression_2_4_f_sum_2008_l670_670075


namespace number_of_ways_to_distribute_balls_l670_670354

theorem number_of_ways_to_distribute_balls (n m : ℕ) (h_n : n = 6) (h_m : m = 2) : (m ^ n = 64) :=
by
  rw [h_n, h_m]
  norm_num

end number_of_ways_to_distribute_balls_l670_670354


namespace winning_candidate_vote_percentage_l670_670828

theorem winning_candidate_vote_percentage 
(votes1 votes2 votes3 votes4 votes5 invalid_votes : ℕ)
(total_registered_voters : ℕ)
(minimum_turnout : ℕ)
(h1 : votes1 = 4136)
(h2 : votes2 = 7636)
(h3 : votes3 = 11628)
(h4 : votes4 = 8735)
(h5 : votes5 = 9917)
(h_invalid : invalid_votes = 458)
(h_total_registered : total_registered_voters = 45000)
(h_min_turnout : minimum_turnout = 60)
(valid_turnout : (votes1 + votes2 + votes3 + votes4 + votes5) ≥ (minimum_turnout * total_registered_voters) / 100) :
  let total_valid_votes := votes1 + votes2 + votes3 + votes4 + votes5 in 
  let total_votes_cast := total_valid_votes + invalid_votes in 
  let winning_vote := votes3 in
  (winning_vote * 100) / total_valid_votes = 28.33 := 
by 
  sorry

end winning_candidate_vote_percentage_l670_670828


namespace obtuse_scalene_triangle_l670_670518

theorem obtuse_scalene_triangle {k : ℕ} (h1 : 13 < k + 17) (h2 : 17 < 13 + k)
  (h3 : 13 < k + 17) (h4 : k ≠ 13) (h5 : k ≠ 17) 
  (h6 : 17^2 > 13^2 + k^2 ∨ k^2 > 13^2 + 17^2) 
  (h7 : (k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10 ∨ k = 22 ∨ 
        k = 23 ∨ k = 24 ∨ k = 25 ∨ k = 26 ∨ k = 27 ∨ k = 28 ∨ k = 29)) :
  ∃ n, n = 14 := 
by
  sorry

end obtuse_scalene_triangle_l670_670518


namespace find_abc_value_l670_670815

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom h1 : a + 1 / b = 5
axiom h2 : b + 1 / c = 2
axiom h3 : c + 1 / a = 9 / 4

theorem find_abc_value : a * b * c = (7 + Real.sqrt 21) / 8 :=
by
  sorry

end find_abc_value_l670_670815


namespace area_of_triangle_AGE_l670_670460

open Set
open Real

-- Define the problem conditions
def square_side_length : ℝ := 5
def point_B (s : ℝ) : EuclideanSpace ℝ 2 := ⟨0, 0⟩
def point_C (s : ℝ) : EuclideanSpace ℝ 2 := ⟨s, 0⟩
def point_D (s : ℝ) : EuclideanSpace ℝ 2 := ⟨s, s⟩
def point_A (s : ℝ) : EuclideanSpace ℝ 2 := ⟨0, s⟩
def point_E : EuclideanSpace ℝ 2 := ⟨2, 0⟩

-- intersection on circumscribed circle of triangle ABE and diagonal BD at G
def point_G : Type := { G : EuclideanSpace ℝ 2 // ∃ (x : ℝ), G = (x, 5 - x) }

-- Calculate the area of triangle AGE
noncomputable def area_triangle_AGE (A E G: EuclideanSpace ℝ 2) : ℝ :=
  0.5 * abs ((A.1 * (E.2 - G.2)) + (E.1 * (G.2 - A.2)) + (G.1 * (A.2 - E.2)))


-- The statement to prove
theorem area_of_triangle_AGE :
  ∀ (s : ℝ), s = 5 →
  ∃ G : point_G,
  area_triangle_AGE (point_A s) point_E (G : EuclideanSpace ℝ 2) = 54.5 :=
by
  intro s hs
  rw hs
  -- Proceed to prove, omitted as it's not required.
  sorry

end area_of_triangle_AGE_l670_670460


namespace ball_radius_l670_670200

theorem ball_radius 
  (r_cylinder : ℝ) (h_rise : ℝ) (v_approx : ℝ)
  (r_cylinder_value : r_cylinder = 12)
  (h_rise_value : h_rise = 6.75)
  (v_approx_value : v_approx = 3053.628) :
  ∃ (r_ball : ℝ), (4 / 3) * Real.pi * r_ball^3 = v_approx ∧ r_ball = 9 := 
by 
  use 9
  sorry

end ball_radius_l670_670200


namespace manuscript_typing_total_cost_l670_670933

theorem manuscript_typing_total_cost {n_pages n_revised_once n_revised_twice rest : ℕ} (cost_first : ℕ) (cost_revision : ℕ)
    (h1 : n_pages = 100) (h2 : cost_first = 10) (h3 : cost_revision = 5)
    (h4 : n_revised_once = 30) (h5 : n_revised_twice = 20) (h6 : rest = n_pages - n_revised_once - n_revised_twice):
    let total_cost := (n_pages * cost_first) + (n_revised_once * cost_revision) + (n_revised_twice * cost_revision * 2)
    in total_cost = 1350 :=
by
  sorry

end manuscript_typing_total_cost_l670_670933


namespace a_value_in_triangle_l670_670823

noncomputable def a_in_triangle
  (b c : ℝ) (B : ℝ) (h_b : b = 3) (h_c : c = 3) (h_B : B = π / 6) : ℝ :=
  let a := 3 * Real.sqrt 3 in
  a

theorem a_value_in_triangle (b c : ℝ) (B : ℝ)
  (h_b : b = 3) (h_c : c = 3) (h_B : B = π / 6) :
  a_in_triangle b c B h_b h_c h_B = 3 * Real.sqrt 3 :=
by {
  sorry
}

end a_value_in_triangle_l670_670823


namespace suff_and_nec_eq_triangle_l670_670209

noncomputable def triangle (A B C: ℝ) (a b c : ℝ) : Prop :=
(B + C = 2 * A) ∧ (b + c = 2 * a)

theorem suff_and_nec_eq_triangle (A B C a b c : ℝ) (h : triangle A B C a b c) :
  A = 60 ∧ B = 60 ∧ C = 60 ∧ a = b ∧ b = c :=
sorry

end suff_and_nec_eq_triangle_l670_670209


namespace find_x_l670_670012

theorem find_x (y x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y - 1) / (y^2 + 2 * y - 2)) : 
  x = y^2 + 2 * y - 1 := 
sorry

end find_x_l670_670012


namespace skiers_cannot_have_exactly_four_overtakes_l670_670456

-- Defining the problem statement
theorem skiers_cannot_have_exactly_four_overtakes (n : ℕ) (h : n = 9) :
  ¬ (∀ (s : fin n), ∃ (o : ℕ), o = 4 ∧ (participates_in_exactly_four_overtakes s o)) :=
by 
  sorry

-- Let's include participatory function definition. It can be summarized 
-- as a function that determines if a skier participates in exactly four overtake
def participates_in_exactly_four_overtakes : fin 9 → ℕ → Prop :=
  sorry

end skiers_cannot_have_exactly_four_overtakes_l670_670456


namespace average_marks_is_25_l670_670975

variable (M P C : ℕ)

def average_math_chemistry (M C : ℕ) : ℕ :=
  (M + C) / 2

theorem average_marks_is_25 (M P C : ℕ) 
  (h₁ : M + P = 30)
  (h₂ : C = P + 20) : 
  average_math_chemistry M C = 25 :=
by
  sorry

end average_marks_is_25_l670_670975


namespace f_1000_3_pow_2021_l670_670873

def f : ℕ → ℕ → ℕ
| a, b := if a > b then b else 
          if f (2 * a) b < a then f (2 * a) b else 
          (f (2 * a) b - a)

theorem f_1000_3_pow_2021 : f 1000 (3 ^ 2021) = 203 :=
by sorry

end f_1000_3_pow_2021_l670_670873


namespace third_term_binomial_expansion_l670_670835

theorem third_term_binomial_expansion :
  let a := 2 * x
  let b := - 1 / x^3
  let n := 5
  (n.choose 2) * a^(n-2) * b^2 = 80 / x^3 := by
  sorry

end third_term_binomial_expansion_l670_670835


namespace find_AX_l670_670692

-- Define the given conditions
variables {AC BC BX AX : ℝ}
def AC_value : AC = 27 := rfl
def BC_value : BC = 36 := rfl
def BX_value : BX = 30 := rfl

-- Define the Angle Bisector Theorem as an assumption
axiom angle_bisector_theorem (AC BC BX AX : ℝ) : AC / BC = AX / BX

-- State the theorem we want to prove
theorem find_AX : AC = 27 → BC = 36 → BX = 30 → AX = 22.5 :=
by {
  intros hAC hBC hBX,
  have h1 : AC / BC = AX / BX := angle_bisector_theorem AC BC BX AX,
  rw [hAC, hBC, hBX] at h1,
  -- Since Lean cannot handle straightforward numeric evaluation, we use sorry to indicate where computation should go
  sorry,
}

end find_AX_l670_670692


namespace number_of_sets_l670_670927

def a : Prop := sorry -- placeholder for "a" being an element
def b : Prop := sorry -- placeholder for "b" being an element
def c : Prop := sorry -- placeholder for "c" being an element

theorem number_of_sets (P : set Prop) :
  (∃ P, {a} ⊆ P ∧ P ⊆ {a, b, c}) → 
  (∃ S : finset (set Prop), S.card = 3 ∧ ∀ x ∈ S, {a} ⊆ x ∧ x ⊆ {a, b, c}) :=
sorry

end number_of_sets_l670_670927


namespace perfect_squares_100_to_400_l670_670803

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l670_670803


namespace probability_two_success_out_of_three_l670_670409

noncomputable def successful_shot (n : ℕ) : Prop := n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4

noncomputable def count_successful_shots (triplet : List ℕ) : ℕ :=
triplet.countp successful_shot

noncomputable def exactly_two_success (triplet : List ℕ) : Prop :=
count_successful_shots triplet = 2

noncomputable def groups : List (List ℕ) :=
[[9, 0, 7], [9, 6, 6], [1, 9, 1], [9, 2, 5], [2, 7, 1], 
 [9, 3, 2], [8, 1, 2], [4, 5, 8], [5, 6, 9], [6, 8, 3],
 [4, 3, 1], [2, 5, 7], [3, 9, 3], [0, 2, 7], [5, 5, 6],
 [4, 8, 8], [7, 3, 0], [1, 1, 3], [5, 3, 7], [9, 8, 9]]

theorem probability_two_success_out_of_three : 
  let count := groups.countp exactly_two_success in
  count / 20 = 0.25 :=
sorry

end probability_two_success_out_of_three_l670_670409


namespace min_value_correct_l670_670309

noncomputable def min_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) : ℝ :=
  3 + 2 * Real.sqrt 2

theorem min_value_correct (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  ∃ x y, x > 0 ∧ y > 0 ∧ x + y = 1 ∧ (∀ x y, x > 0 ∧ y > 0 ∧ x + y = 1 → 
  (frac_value : ℝ := 2 / x + 1 / y) ≥ (3 + 2 * Real.sqrt 2)) ∧ 
  (frac_value := 2 / x + 1 / y = min_value x y hx hy hxy) := sorry

end min_value_correct_l670_670309


namespace Tim_math_score_l670_670533

noncomputable def first_n_even_numbers_sum (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, 2 * (k + 1))

theorem Tim_math_score : first_n_even_numbers_sum 7 = 56 :=
by
  sorry

end Tim_math_score_l670_670533


namespace constant_term_expansion_l670_670181

theorem constant_term_expansion (x : ℝ) : 
  (5 * x + 1 / (5 * x)) ^ 8 = (70 : ℝ) + 
  ∑ k in finset.range 8 \ {4}, (finset.choose 8 k) * (5 * x) ^ k * (1 / (5 * x)) ^ (8 - k) :=
by
  sorry

end constant_term_expansion_l670_670181


namespace arithmetic_sequence_formula_geometric_sequence_formula_sum_of_products_formula_l670_670738

section sequences_problem

-- Definitions and given conditions
def a (n : ℕ) : ℕ := 3 * n
def b (n : ℕ) : ℕ := 2^(n - 1)

def S (n : ℕ) : ℕ := n * a(n) / 2  -- Sum of the first n terms (arithmetic series sum)

-- Provided conditions
axiom a1 : a 1 = 3
axiom b1 : b 1 = 1
axiom cond_1 : a 2 * b 2 = 12
axiom cond_2 : S 3 + b 2 = 20

-- Statements to prove
theorem arithmetic_sequence_formula : ∀ n : ℕ, a n = 3 * n :=
by sorry

theorem geometric_sequence_formula : ∀ n : ℕ, b n = 2^(n - 1) :=
by sorry

theorem sum_of_products_formula : ∀ n : ℕ, 
  ∑ i in Finset.range n, a i * b i = 3 * (2^(n-1) * (2 + n) - 1) :=
by sorry

end sequences_problem

end arithmetic_sequence_formula_geometric_sequence_formula_sum_of_products_formula_l670_670738


namespace min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l670_670575

-- Define the first problem
theorem min_cuts_for_eleven_day_stay : 
  (∀ (chain_len num_days : ℕ), chain_len = 11 ∧ num_days = 11 
  → (∃ (cuts : ℕ), cuts = 2)) := 
sorry

-- Define the second problem
theorem max_days_with_n_cuts : 
  (∀ (n chain_len days : ℕ), chain_len = (n + 1) * 2 ^ n - 1 
  → days = (n + 1) * 2 ^ n - 1) := 
sorry

end min_cuts_for_eleven_day_stay_max_days_with_n_cuts_l670_670575


namespace probability_zero_point_l670_670397

theorem probability_zero_point :
  let f (x a b : ℝ) := x^2 + 2 * a * x - b^2 + π
  let Ω := set.prod (set.Icc (-π) (π)) (set.Icc (-π) (π))
  let S := measure_theory.volume Ω
  let favorable_events := {p : ℝ × ℝ | p.1 ^ 2 + p.2 ^ 2 ≥ π}
  let s := measure_theory.volume favorable_events
  s / S = 3 / 4 :=
by sorry

end probability_zero_point_l670_670397


namespace find_value_l670_670728

variable {a : ℕ → ℝ}

-- Define the sequence condition
def seq_condition (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (Finset.range n).sum (λ k, a k ^ (1/2 : ℝ)) = 2^(n + 1)

-- Define the sum of the sequence
def seq_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- The main statement
theorem find_value (a : ℕ → ℝ) (n : ℕ) (h : seq_condition a n) :
  3 / 16 * seq_sum a n = 4^(n - 1) + 2 :=
  sorry

end find_value_l670_670728


namespace sum_of_first_20_terms_l670_670727

def seq : ℕ → ℕ
| 0     := 1
| 1     := 2
| (n+2) := (1 + cos (n * π / 2) ^ 2) * seq n + sin (n * π / 2) ^ 2

noncomputable def sum_first_20_terms : ℕ := 
  (List.range' 1 20).map seq |>.sum

theorem sum_of_first_20_terms : sum_first_20_terms = 2101 :=
by
  -- Proof goes here
  sorry

end sum_of_first_20_terms_l670_670727


namespace susan_fraction_apples_given_out_l670_670712

theorem susan_fraction_apples_given_out (frank_apples : ℕ) (frank_sold_fraction : ℚ) 
  (total_remaining_apples : ℕ) (susan_multiple : ℕ) 
  (H1 : frank_apples = 36) 
  (H2 : susan_multiple = 3) 
  (H3 : frank_sold_fraction = 1 / 3) 
  (H4 : total_remaining_apples = 78) :
  let susan_apples := susan_multiple * frank_apples
  let frank_sold_apples := frank_sold_fraction * frank_apples
  let frank_remaining_apples := frank_apples - frank_sold_apples
  let total_before_susan_gave_out := susan_apples + frank_remaining_apples
  let susan_gave_out := total_before_susan_gave_out - total_remaining_apples
  let susan_gave_fraction := susan_gave_out / susan_apples
  susan_gave_fraction = 1 / 2 :=
by
  sorry

end susan_fraction_apples_given_out_l670_670712


namespace flower_combinations_l670_670222

theorem flower_combinations : 
  (∃ n, (∃ r c : ℕ, 4 * r + 5 * c = 60 ∧ n = 1 + (c / 4) ∧ 0 ≤ c ∧ c ≤ 12)) ∧ 
  (∀ r c : ℕ, 4 * r + 5 * c = 60 → c % 4 = 0) →
  set.count {n | ∃ r c : ℕ, 4 * r + 5 * c = 60 ∧ n = 1 + (c / 4) ∧ 0 ≤ c ∧ c ≤ 12} = 4 := 
by sorry

end flower_combinations_l670_670222


namespace count_odd_3_digit_numbers_l670_670173

theorem count_odd_3_digit_numbers : 
  let digits := {0, 1, 2, 3}
  in ( ∃ n1 n2 n3 : ℕ, n1 ∈ digits ∧ n2 ∈ digits ∧ n3 ∈ digits ∧ 
        n1 ≠ n2 ∧ n1 ≠ n3 ∧ n2 ≠ n3 ∧ 
        (n3 = 1 ∨ n3 = 3) ∧ -- Last digit must be odd
        n1 ≠ 0) -- First digit can't be 0
      = 8 :=
by sorry

end count_odd_3_digit_numbers_l670_670173


namespace find_f79_l670_670507

noncomputable def f : ℝ → ℝ :=
  sorry

axiom condition1 : ∀ x y : ℝ, f (x * y) = x * f y
axiom condition2 : f 1 = 25

theorem find_f79 : f 79 = 1975 :=
by
  sorry

end find_f79_l670_670507


namespace problem_statement_l670_670008

theorem problem_statement (f : ℝ → ℝ) (a b : ℝ) (h₀ : ∀ x, f x = 4 * x + 3) (h₁ : a > 0) (h₂ : b > 0) :
  (∀ x, |f x + 5| < a ↔ |x + 3| < b) ↔ b ≤ a / 4 :=
sorry

end problem_statement_l670_670008


namespace roots_sum_product_l670_670010

theorem roots_sum_product (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (h : ∀ x : ℝ, x^2 - p*x - 2*q = 0) :
  (p + q = p) ∧ (p * q = -2*q) :=
by
  sorry

end roots_sum_product_l670_670010


namespace no_zero_in_range_l670_670866

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then ⌈1 / (x + 3)⌉
else if x < -3 then ⌊1 / (x + 3)⌋
else 0 -- This value is arbitrary as g(x) is not defined at x = -3

theorem no_zero_in_range : ¬ ∃ x : ℝ, g x = 0 :=
begin
  unfold g,
  intros h,
  cases h with x hx,
  rw if_neg (by linarith) at hx, -- Excludes the case x = -3
  cases (lt_or_gt_of_ne (ne_of_gt (by linarith.min)))
  case inl => rw if_neg (by linarith) at hx -- Case when x < -3
  case inr => rw if_pos (by linarith) at hx -- Case when x > -3
  -- Both cases imply contradiction as shown in the problem solution
  contradiction,
end

end no_zero_in_range_l670_670866


namespace mrs_jackson_boxes_l670_670091

theorem mrs_jackson_boxes (decorations_per_box used_decorations given_decorations : ℤ) 
(h1 : decorations_per_box = 15)
(h2 : used_decorations = 35)
(h3 : given_decorations = 25) :
  (used_decorations + given_decorations) / decorations_per_box = 4 := 
by sorry

end mrs_jackson_boxes_l670_670091


namespace find_value_of_expression_l670_670339

noncomputable def a : ℕ → ℝ
| 1     := 1
| (n+1) := (1/3)^(n+1) - a n

def T (n : ℕ) : ℝ :=
finset.sum (finset.range n) (λ k, a (k+1) * 3^(k+1))

theorem find_value_of_expression (n : ℕ) :
  4 * T n - a n * 3^(n+1) = n + 2 :=
sorry

end find_value_of_expression_l670_670339


namespace geometric_sequence_formula_arithmetic_sequence_sum_l670_670983

theorem geometric_sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) (r : ℝ) (hr_pos : r > 0) (h3 : a 3 = a 2 + 4) :
  ∀ n, a n = 2 ^ n :=
by
  sorry

theorem arithmetic_sequence_sum (b : ℕ → ℝ) (S : ℕ → ℝ) (h2 : b 1 = 1) (d : ℝ) (hd : d = 2) 
  (h4 : ∀ n, b (n + 1) = b n + d) (a : ℕ → ℝ) (h5 : ∀ n, a n = 2 ^ n) : 
  ∀ n, S n = (∑ i in finset.range n, a (i + 1) + b (i + 1)) = 2^(n + 1) + n^2 - 2 :=
by
  sorry

end geometric_sequence_formula_arithmetic_sequence_sum_l670_670983


namespace greatest_integer_le_x_squared_div_50_l670_670464

-- Define the conditions as given in the problem
def trapezoid (b h : ℝ) (x : ℝ) : Prop :=
  let baseDifference := 50
  let longerBase := b + baseDifference
  let midline := (b + longerBase) / 2
  let heightRatioFactor := 2
  let xSquared := 6875
  let regionAreaRatio := 2 / 1 -- represented as 2
  (let areaRatio := (b + midline) / (b + baseDifference / 2)
   areaRatio = regionAreaRatio) ∧
  (x = Real.sqrt xSquared) ∧
  (b = 50)

-- Define the theorem that captures the question
theorem greatest_integer_le_x_squared_div_50 (b h x : ℝ) (h_trapezoid : trapezoid b h x) :
  ⌊ (x^2) / 50 ⌋ = 137 :=
by sorry

end greatest_integer_le_x_squared_div_50_l670_670464


namespace money_spent_on_ferris_wheel_l670_670592

-- Conditions
def initial_tickets : ℕ := 6
def remaining_tickets : ℕ := 3
def ticket_cost : ℕ := 9

-- Prove that the money spent during the ferris wheel ride is 27 dollars
theorem money_spent_on_ferris_wheel : (initial_tickets - remaining_tickets) * ticket_cost = 27 := by
  sorry

end money_spent_on_ferris_wheel_l670_670592


namespace intersection_points_l670_670346

def equation1 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9
def equation2 (x y : ℝ) : Prop := x^2 + (y - 5)^2 = 25

theorem intersection_points :
  ∃ (x1 y1 x2 y2 : ℝ),
    equation1 x1 y1 ∧ equation2 x1 y1 ∧
    equation1 x2 y2 ∧ equation2 x2 y2 ∧
    (x1, y1) ≠ (x2, y2) ∧
    ∀ (x y : ℝ), equation1 x y ∧ equation2 x y → (x, y) = (x1, y1) ∨ (x, y) = (x2, y2) := sorry

end intersection_points_l670_670346


namespace distance_proof_l670_670570

-- Declare noncomputable environment because we may involve real numbers and operations.
noncomputable theory

-- Define the conditions as constants.
def xm_speed : ℝ := 12
def xm_travel_time : ℝ := 2.5
def father_speed : ℝ := 36
def father_delay : ℝ := 0.5

-- The distance between Xiao Ming's home and his grandmother's house.
def distance_between_homes : ℝ := xm_speed * xm_travel_time

-- Declare the theorem to find the distance between Xiao Ming's home and his grandmother's house.
theorem distance_proof : distance_between_homes = 45 := by
  sorry

end distance_proof_l670_670570


namespace estimate_fish_in_pond_l670_670458

theorem estimate_fish_in_pond
  (n m k : ℕ)
  (h_pr: k = 200)
  (h_cr: k = 8)
  (h_m: n = 200):
  n / (m / k) = 5000 := sorry

end estimate_fish_in_pond_l670_670458


namespace evaluate_expression_l670_670679

theorem evaluate_expression : 3 ^ 123 + 9 ^ 5 / 9 ^ 3 = 3 ^ 123 + 81 :=
by
  -- we add sorry as the proof is not required
  sorry

end evaluate_expression_l670_670679


namespace num_of_triples_l670_670703

theorem num_of_triples : 
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ (x * y + y * z = 100) ∧ (x * z = 56)) = 3 :=
sorry

end num_of_triples_l670_670703


namespace starWars_earnings_correct_l670_670126

-- Define the given conditions
def lionKing_cost : ℕ := 10
def lionKing_earnings : ℕ := 200
def starWars_cost : ℕ := 25
def lionKing_profit : ℕ := lionKing_earnings - lionKing_cost
def starWars_profit : ℕ := lionKing_profit * 2
def starWars_earnings : ℕ := starWars_profit + starWars_cost

-- The theorem which states that the Star Wars earnings are indeed 405 million
theorem starWars_earnings_correct : starWars_earnings = 405 := by
  -- proof goes here
  sorry

end starWars_earnings_correct_l670_670126


namespace total_number_of_fish_l670_670160

theorem total_number_of_fish (fishbowls : ℕ) (fish_per_bowl : ℕ) 
  (h1: fishbowls = 261) (h2: fish_per_bowl = 23) : 
  fishbowls * fish_per_bowl = 6003 := 
by
  sorry

end total_number_of_fish_l670_670160


namespace integral_sin4_cos4_eq_3pi_over_8_l670_670978

theorem integral_sin4_cos4_eq_3pi_over_8 : 
  (∫ x in 0..Real.pi, 2^4 * (Real.sin x)^4 * (Real.cos x)^4) = (3 * Real.pi) / 8 := 
by
  sorry

end integral_sin4_cos4_eq_3pi_over_8_l670_670978


namespace greatest_perimeter_l670_670942

theorem greatest_perimeter (A B C : Point ℝ)
                          (h_eq_triangle : equilateral_triangle A B C (12 : ℝ))
                          (pieces : list (triangle ℝ))
                          (h_six_pieces : pieces.length = 6)
                          (h_equal_area : ∀ t ∈ pieces, area t = area (equilateral_triangle A B C (12 : ℝ)) / 6) :
  ∃ t ∈ pieces, perimeter t = 22.62 :=
by
  sorry

end greatest_perimeter_l670_670942


namespace like_terms_constants_l670_670965

theorem like_terms_constants :
  ∀ (a b : ℚ), a = 1/2 → b = -1/3 → (a = 1/2 ∧ b = -1/3) → a + b = 1/2 + -1/3 :=
by
  intros a b ha hb h
  sorry

end like_terms_constants_l670_670965


namespace least_number_44597_l670_670952

noncomputable def lcm (m n : ℕ) : ℕ := (m * n) / (Nat.gcd m n)

noncomputable def leastNumberToAdd (a b c n : ℕ) : ℕ :=
  let l := lcm a (lcm b c)
  l - (n % l)

theorem least_number_44597 :
  leastNumberToAdd 29 37 43 1056 = 44597 :=
  by
  -- The remainder proof is omitted
  sorry

end least_number_44597_l670_670952


namespace expression_evaluation_l670_670190

theorem expression_evaluation :
  1 - (2 - (3 - 4 - (5 - 6))) = -1 :=
sorry

end expression_evaluation_l670_670190


namespace dot_product_dot_product_combination_l670_670719

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Conditions
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 3 := sorry
def angle_a_b : real.angle a b = real.pi / 3 := sorry

-- Problem
theorem dot_product :
  a.dot b = -3 := sorry

theorem dot_product_combination :
  (2 • a - b).dot (a + 3 • b) = -34 := sorry

noncomputable def vector_norm : ∥3 • a + b∥ = 3 * real.sqrt 3 := sorry

end dot_product_dot_product_combination_l670_670719


namespace count_prime_squares_between_bounds_l670_670001

-- Conditions
def square_bounds (p : ℕ) : Prop :=
  5000 < p^2 ∧ p^2 < 8000

-- Counting the primes within the given bounds
def primes_in_range : list ℕ :=
  [71, 73, 79, 83, 89]

theorem count_prime_squares_between_bounds :
  (primes_in_range.filter (λ p, p.prime)).length = 5 :=
by
  -- Theorem statement without proof
  sorry

end count_prime_squares_between_bounds_l670_670001


namespace cricketer_stats_l670_670219

theorem cricketer_stats :
  let total_runs := 225
  let total_balls := 120
  let boundaries := 4 * 15
  let sixes := 6 * 8
  let twos := 2 * 3
  let singles := 1 * 10
  let perc_boundaries := (boundaries / total_runs.toFloat) * 100
  let perc_sixes := (sixes / total_runs.toFloat) * 100
  let perc_twos := (twos / total_runs.toFloat) * 100
  let perc_singles := (singles / total_runs.toFloat) * 100
  let strike_rate := (total_runs.toFloat / total_balls.toFloat) * 100
  perc_boundaries = 26.67 ∧
  perc_sixes = 21.33 ∧
  perc_twos = 2.67 ∧
  perc_singles = 4.44 ∧
  strike_rate = 187.5 :=
by
  sorry

end cricketer_stats_l670_670219


namespace find_B_value_l670_670502

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end find_B_value_l670_670502


namespace curve_equation_l670_670297

-- Define the parametric conditions for curve C
def curve_param (θ : ℝ) : ℝ × ℝ := (cos θ - 1, sin θ + 1)

-- The proof problem statement
theorem curve_equation : ∀ x y θ,
  (x, y) = curve_param θ ↔ (x + 1) ^ 2 + (y - 1) ^ 2 = 1 :=
by sorry

end curve_equation_l670_670297


namespace lunch_break_duration_l670_670468

/-- Paula and her two helpers start at 7:00 AM and paint 60% of a house together,
    finishing at 5:00 PM. The next day, only the helpers paint and manage to
    paint 30% of another house, finishing at 3:00 PM. On the third day, Paula
    paints alone and paints the remaining 40% of the house, finishing at 4:00 PM.
    Prove that the length of their lunch break each day is 1 hour (60 minutes). -/
theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (0 < L) ∧ 
    (L < 10) ∧
    (∃ (p h : ℝ), 
       (10 - L) * (p + h) = 0.6 ∧
       (8 - L) * h = 0.3 ∧
       (9 - L) * p = 0.4) ∧  
    L = 1 :=
by
  sorry

end lunch_break_duration_l670_670468


namespace infinite_series_not_alg_solvable_l670_670644

/-- Representing the problem conditions as Lean definitions -/
def S1 : ℕ := 1 + 2 + 3 + … + 90
def S2 : ℕ := 1 + 2 + 3 + 4
def S3 : Nat := sorry  -- this represents the infinite series
def S4 : ℕ := 1^2 + 2^2 + 3^2 + … + 100^2

/-- The main statement which formalizes the problem that 
    an infinite series sum represented by S3 cannot be solved by a finite algorithm -/
theorem infinite_series_not_alg_solvable : 
  ¬ ∃ (f : ℕ → ℕ), 
    (∀ n, f n = S3) := sorry

end infinite_series_not_alg_solvable_l670_670644


namespace speed_of_bus_correct_l670_670098

noncomputable def speed_of_bus 
  (speed_car speed_motorcycle : ℝ) 
  (t_buns : ℝ) 
  (h1 : speed_car = 60) 
  (h2 : speed_motorcycle = 30)
  (p1 : ∃ t1 t2, t1 = 3 * t_buns ∧ t2 = 3 * t_buns ∧ speed_motorcycle * t1 = speed_car * (t1 + t2))
  (p2: ∃ t1 t2, t1 = 3 * t_buns ∧ t2 = 3 * t_buns ∧  speed_car * t1 = speed_motorcycle * (t1 + t2)) : ℝ :=
  40

@[simp]
theorem speed_of_bus_correct : 
  ∀ (speed_car speed_motorcycle : ℝ) 
  (t_buns : ℝ) 
  (h1 : speed_car = 60) 
  (h2 : speed_motorcycle = 30)
  (p1 : ∃ t1 t2, t1 = 3 * t_buns ∧ t2 = 3 * t_buns ∧ speed_motorcycle * t1 = speed_car * (t1 + t2))
  (p2: ∃ t1 t2, t1 = 3 * t_buns ∧ t2 = 3 * t_buns ∧ speed_car * t1 = speed_motorcycle * (t1 + t2)), 
  speed_of_bus speed_car speed_motorcycle t_buns h1 h2 p1 p2 = 40 :=
by
  intros
  simp
  sorry

end speed_of_bus_correct_l670_670098


namespace six_digit_phone_number_count_l670_670315

def six_digit_to_seven_digit_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) : ℕ :=
  let num_positions := 7
  let num_digits := 10
  num_positions * num_digits

theorem six_digit_phone_number_count (six_digit : ℕ) (h : 100000 ≤ six_digit ∧ six_digit < 1000000) :
  six_digit_to_seven_digit_count six_digit h = 70 := by
  -- Proof goes here
  sorry

end six_digit_phone_number_count_l670_670315


namespace abs_sub_sqrt5_l670_670916

theorem abs_sub_sqrt5 :
  |2 - real.sqrt 5| = real.sqrt 5 - 2 :=
by sorry

end abs_sub_sqrt5_l670_670916


namespace find_original_number_l670_670415

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end find_original_number_l670_670415


namespace find_p_l670_670430

noncomputable def polynomial : ℝ[X] := 9 * X^3 - 6 * X^2 - 45 * X + 54

theorem find_p (p : ℝ) (h : (X - C p)^2 ∣ polynomial) : 
  p = (2 - real.sqrt 139) / 9 := 
sorry

end find_p_l670_670430


namespace cost_of_plastering_is_334_point_8_l670_670972

def tank_length : ℝ := 25
def tank_width : ℝ := 12
def tank_depth : ℝ := 6
def cost_per_sq_meter : ℝ := 0.45

def bottom_area : ℝ := tank_length * tank_width
def long_wall_area : ℝ := 2 * (tank_length * tank_depth)
def short_wall_area : ℝ := 2 * (tank_width * tank_depth)
def total_surface_area : ℝ := bottom_area + long_wall_area + short_wall_area
def total_cost : ℝ := total_surface_area * cost_per_sq_meter

theorem cost_of_plastering_is_334_point_8 :
  total_cost = 334.8 :=
by
  sorry

end cost_of_plastering_is_334_point_8_l670_670972


namespace calculate_xy_yz_zx_l670_670879

variable (x y z : ℝ)

theorem calculate_xy_yz_zx (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
    (h1 : x^2 + x * y + y^2 = 75)
    (h2 : y^2 + y * z + z^2 = 49)
    (h3 : z^2 + z * x + x^2 = 124) : 
    x * y + y * z + z * x = 70 :=
sorry

end calculate_xy_yz_zx_l670_670879


namespace cleaning_time_ratio_l670_670881

/-- 
Given that Lilly and Fiona together take a total of 480 minutes to clean a room and Fiona
was cleaning for 360 minutes, prove that the ratio of the time Lilly spent cleaning 
to the total time spent cleaning the room is 1:4.
-/
theorem cleaning_time_ratio (total_time minutes Fiona_time : ℕ) 
  (h1 : total_time = 480)
  (h2 : Fiona_time = 360) : 
  (total_time - Fiona_time) / total_time = 1 / 4 :=
by
  sorry

end cleaning_time_ratio_l670_670881


namespace odd_prime_factor_exists_l670_670979

-- Given Conditions
variables {a : ℕ} {k : ℕ} (p : ℕ → ℕ) (hp : ∀ i, 1 ≤ i → i ≤ k → Nat.Prime (p i) ∧ p i % 2 = 1)
variable (hka : 2 ≤ k)
variable (ha : Nat.gcd a (Nat.prod (fun i => p i) (Finset.range k)) = 1)

-- Problem Statement
theorem odd_prime_factor_exists (a : ℕ) (k : ℕ) (p : ℕ → ℕ) (hp : ∀ i, 1 ≤ i → i ≤ k → Nat.Prime (p i) ∧ p i % 2 = 1) (hka : 2 ≤ k) (ha : Nat.gcd a (Nat.prod (fun i => p i) (Finset.range k)) = 1) :
  ∃ q : ℕ, Nat.Prime q ∧ q ≠ p 1 ∧ q ≠ p 2 ∧ ... ∧ q ≠ p k ∧ q ∣ (a ^ (Nat.prod (λ i, p i - 1) (Finset.range k)) - 1) := 
sorry

end odd_prime_factor_exists_l670_670979


namespace expected_number_of_successful_trials_l670_670194
open ProbabilityTheory

noncomputable def probability_of_success_per_trial : ℝ := 1 - (2/3) * (2/3) * (2/3)
noncomputable def expected_successful_trials (num_trials : ℕ) : ℝ := num_trials * probability_of_success_per_trial

theorem expected_number_of_successful_trials :
  expected_successful_trials 54 = 38 := by
  -- Defining the probability of success per trial
  have prob_success : probability_of_success_per_trial = 19 / 27 := sorry,
  -- Calculating the expected number of successful trials
  show expected_successful_trials 54 = 38,
  rw [expected_successful_trials, prob_success],
  norm_num,
  -- Expected result should match the calculation
  sorry

end expected_number_of_successful_trials_l670_670194


namespace solution_l670_670254

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def favorable_outcomes (event : ℕ → Prop) : ℕ :=
  (finset.range 6).sum (λ x, (finset.range 6).count (λ y, event (x + 1 + (y + 1))))

def highest_probability_event : Prop :=
  let primes := favorable_outcomes is_prime in
  let multiples_of_4 := favorable_outcomes (λ n, n % 4 = 0) in
  let perfect_squares := favorable_outcomes (λ n, n = 4 ∨ n = 9) in
  let total_7 := favorable_outcomes (λ n, n = 7) in
  let factors_of_12 := favorable_outcomes (λ n, n ∣ 12) in
  primes > multiples_of_4 ∧ primes > perfect_squares ∧ primes > total_7 ∧ primes > factors_of_12

theorem solution : highest_probability_event :=
by
  sorry

end solution_l670_670254


namespace trapezoid_area_l670_670132

-- Define the isosceles trapezoid with given properties
noncomputable def isosceles_trapezoid_area (d1 d2 : ℝ) (M : ℝ) 
  (perpendicular_diagonals : d1 * d2 / 2 = M) 
  (midsegment_value : M = 5) : Prop :=
  ∃ (h : ℝ), 1/2 * (2 * M) * (h) = 25

-- Define the theorem that we want to prove
theorem trapezoid_area : ∀ (d1 d2 : ℝ), isosceles_trapezoid_area d1 d2 5 (d1 * d2 / 2 = 5) (5 = 5) :=
sorry

end trapezoid_area_l670_670132


namespace f_eight_eq_two_l670_670140

-- Defining the predicates and functions based on the given problem's conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m > 1 ∧ m < n → ¬ (m ∣ n))

def unique_prime_sum (n : ℕ) (s : List ℕ) : Prop := 
  (∀ x ∈ s, is_prime x) ∧ s.sum = n ∧ s = s.sort (· ≤ ·)

-- The function f, defined to count unique prime sums for a number
def f (n : ℕ) : ℕ :=
  (Finset.univ.filter (λ s : List ℕ, unique_prime_sum n s)).card

-- The theorem statement
theorem f_eight_eq_two : f 8 = 2 := 
  sorry

end f_eight_eq_two_l670_670140


namespace ellipse_equation_1_hyperbola_equation_2_l670_670594

-- Define the problem and constants
def ellipse_passing_through_0_sqrt_3_with_eccentricity_half := 
  ∃ a b c : ℝ, a^2 = 4 ∧ b^2 = 3 ∧ c = 1 ∧ (0, real.sqrt 3) on ellipse ∧ e = 1/2 → 
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ∨ y^2 / 3 + x^2 / (4 / 3) = 1)

def hyperbola_same_asymptotes_and_passing_through_2_sqrt_5 :=
  ∃ λ : ℝ, (2, real.sqrt 5) on hyperbola ∧ ∀ x y : ℝ, y^2 / 4 - x^2 / 16 = 1

-- Results for the problems
theorem ellipse_equation_1 :
  ∃ a b c : ℝ, a^2 = 4 ∧ b^2 = 3 ∧ c = 1 ∧ (0, real.sqrt 3) on ellipse ∧ e = 1/2 → 
  (∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 ∨ y^2 / 3 + x^2 / (4 / 3) = 1) :=
sorry

theorem hyperbola_equation_2 :
  ∃ λ : ℝ, (2, real.sqrt 5) on hyperbola ∧ ∀ x y : ℝ, y^2 / 4 - x^2 / 16 = 1 :=
sorry

end ellipse_equation_1_hyperbola_equation_2_l670_670594


namespace invest_time_p_l670_670152

-- Conditions
variables (x t : ℝ) -- Real numbers for the common multiple and unknown period.
constant invested_p : ℝ := 7 * x -- Investment by p
constant invested_q : ℝ := 5 * x -- Investment by q
constant time_q : ℝ := 20 -- Time invested by q

-- Given: Ratio of profits is 7:10
constant profits_ratio : (invested_p * t) / (invested_q * time_q) = 7 / 10

-- Proving: Time invested by p is 14
theorem invest_time_p : t = 14 :=
by
  sorry -- proof is omitted

end invest_time_p_l670_670152


namespace parabola_properties_l670_670496

theorem parabola_properties :
  ∀ x : ℝ, (x - 3)^2 + 5 = (x-3)^2 + 5 ∧ 
  (x - 3)^2 + 5 > 0 ∧ 
  (∃ h : ℝ, h = 3 ∧ ∀ x1 x2 : ℝ, (x1 - h)^2 <= (x2 - h)^2) ∧ 
  (∃ h k : ℝ, h = 3 ∧ k = 5) := 
by 
  sorry

end parabola_properties_l670_670496


namespace measure_of_angle_DSO_l670_670403

theorem measure_of_angle_DSO 
  (DOG : Triangle)
  (angle_DGO_eq_DOG : DOG.angle DGO = DOG.angle DOG)
  (angle_DOG : DOG.angle DOG = 48)
  (bisect_OS : DOG.bisects OS DGO) :
  DOG.angle DSO = 108 := by
  sorry

end measure_of_angle_DSO_l670_670403


namespace total_players_count_l670_670599

def kabadi_players : ℕ := 10
def kho_kho_only_players : ℕ := 35
def both_games_players : ℕ := 5

theorem total_players_count : kabadi_players + kho_kho_only_players - both_games_players = 40 :=
by
  sorry

end total_players_count_l670_670599


namespace inequality_negatives_l670_670739

theorem inequality_negatives (a b : ℝ) (h1 : a < b) (h2 : b < 0) : (b / a) < 1 :=
by
  sorry

end inequality_negatives_l670_670739


namespace square_equiv_l670_670652

theorem square_equiv (x : ℝ) : 
  (7 - (x^3 - 49)^(1/3))^2 = 
  49 - 14 * (x^3 - 49)^(1/3) + ((x^3 - 49)^(1/3))^2 := 
by 
  sorry

end square_equiv_l670_670652


namespace crt_solution_l670_670695

/-- Congruences from the conditions -/
def congruences : Prop :=
  ∃ x : ℤ, 
    (x % 2 = 1) ∧
    (x % 3 = 2) ∧
    (x % 5 = 3) ∧
    (x % 7 = 4)

/-- The target result from the Chinese Remainder Theorem -/
def target_result : Prop :=
  ∃ x : ℤ, 
    (x % 210 = 53)

/-- The proof problem stating that the given conditions imply the target result -/
theorem crt_solution : congruences → target_result :=
by
  sorry

end crt_solution_l670_670695


namespace fermat_prime_divisibility_l670_670872

def F (k : ℕ) : ℕ := 2 ^ 2 ^ k + 1

theorem fermat_prime_divisibility {m n : ℕ} (hmn : m > n) : F n ∣ (F m - 2) :=
sorry

end fermat_prime_divisibility_l670_670872


namespace inequality_proof_l670_670991

open Nat

theorem inequality_proof (m n : ℕ) (a : Fin n → ℝ) (h_mn : m ∈ Set.Iic n) (h_n : n ≥ 2) 
(h_ai_pos : ∀ i, 0 < a i) (h_sum_ai : ∑ i, a i = 1) : 
  (∑ i, (a i)^(2 - ↑m) + ∑ j in (Finset.univ \ {i}), a j) / (1 - a i) ≥ n + (n^m - n) / (n - 1) :=
by
  sorry

end inequality_proof_l670_670991


namespace unique_scalar_matrix_l670_670292

variables (N : Matrix (Fin 3) (Fin 3) ℝ)

theorem unique_scalar_matrix (h : ∀ u : Fin 3 → ℝ, N.mul_vec u = 3 • u) :
  N = 3 • (1 : Matrix (Fin 3) (Fin 3) ℝ) :=
by
  sorry

end unique_scalar_matrix_l670_670292


namespace range_of_F_l670_670725

open Set

def f_K (K : Set ℝ) (x : ℝ) : ℝ :=
  if x ∈ K then 1 else 0

def F (M N : Set ℝ) (x : ℝ) : ℝ :=
  (f_K M x + f_K N x + 1) / (f_K (M ∪ N) x + 1)

theorem range_of_F (M N : Set ℝ) (hM : M ⊂ univ ∧ M ≠ ∅) (hN : N ⊂ univ ∧ N ≠ ∅) 
  (h_disjoint : M ∩ N = ∅) :
  range (F M N) = {1} := 
  sorry

end range_of_F_l670_670725


namespace sqrt_product_simplifies_l670_670482

theorem sqrt_product_simplifies :
  real.sqrt 12 * real.sqrt 75 = 30 :=
by
  -- proof omitted
  sorry

end sqrt_product_simplifies_l670_670482


namespace element_4_in_B_l670_670778

def U : set ℕ := {x | x ≤ 7}
def A : set ℕ
def B : set ℕ := U \ A
def B_c : set ℕ := U \ B

theorem element_4_in_B (h₁ : U = {x | x ≤ 7})
  (h₂ : A ∪ B = U) 
  (h₃ : A ∩ B_c = {2, 3, 5, 7}) : 4 ∈ B :=
sorry

end element_4_in_B_l670_670778


namespace infinitely_many_positive_integers_l670_670291

theorem infinitely_many_positive_integers (k : ℕ) (m := 13 * k + 1) (h : m ≠ 8191) :
  8191 = 2 ^ 13 - 1 → ∃ (m : ℕ), ∀ k : ℕ, (13 * k + 1) ≠ 8191 ∧ ∃ (t : ℕ), (2 ^ (13 * k) - 1) = 8191 * m * t := by
  intros
  sorry

end infinitely_many_positive_integers_l670_670291


namespace profit_calculation_l670_670031

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end profit_calculation_l670_670031


namespace binomial_expectation_l670_670164

noncomputable def E (X : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ :=
∑ k in Finset.range (n+1), k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

theorem binomial_expectation (n : ℕ) (p : ℝ) (h : p = 0.3) (h2 : n = 8) :
  E (λ k, k * (n.choose k : ℝ) * p^k * (1 - p)^(n - k)) n p = 2.4 := by
  rw [h, h2]
  sorry

end binomial_expectation_l670_670164


namespace both_true_of_neg_and_false_l670_670376

variable (P Q : Prop)

theorem both_true_of_neg_and_false (h : ¬ (P ∧ Q) = False) : P ∧ Q :=
by
  -- Proof goes here
  sorry

end both_true_of_neg_and_false_l670_670376


namespace eq1_solution_eq2_no_solution_l670_670486

theorem eq1_solution (x : ℝ) (h : x ≠ 0 ∧ x ≠ 2) :
  (2/x + 1/(x*(x-2)) = 5/(2*x)) ↔ x = 4 :=
by sorry

theorem eq2_no_solution (x : ℝ) (h : x ≠ 2) :
  (5*x - 4)/ (x - 2) = (4*x + 10) / (3*x - 6) - 1 ↔ false :=
by sorry

end eq1_solution_eq2_no_solution_l670_670486


namespace distance_from_point_to_plane_l670_670696

def M₁ : (ℝ × ℝ × ℝ) := (1, 2, -3)
def M₂ : (ℝ × ℝ × ℝ) := (1, 0, 1)
def M₃ : (ℝ × ℝ × ℝ) := (-2, -1, 6)
def M₀ : (ℝ × ℝ × ℝ) := (3, -2, -9)

theorem distance_from_point_to_plane : 
  let distance (p₁ : ℝ × ℝ × ℝ) (p₂ p₃ p₄ : ℝ × ℝ × ℝ) := 
    let ⟨x, y, z⟩ := p₁ in
    let ⟨a1, b1, c1⟩ := p₂ in
    let ⟨a2, b2, c2⟩ := p₃ in
    let ⟨a3, b3, c3⟩ := p₄ in
    let A := (b2 - b1) * (c3 - c1) - (c2 - c1) * (b3 - b1) in
    let B := (c2 - c1) * (a3 - a1) - (a2 - a1) * (c3 - c1) in
    let C := (a2 - a1) * (b3 - b1) - (b2 - b1) * (a3 - a1) in
    let D := -(A * a1 + B * b1 + C * c1) in
    let dist := |A * x + B * y + C * z + D| / Real.sqrt (A^2 + B^2 + C^2) in
    dist
  in distance M₀ M₁ M₂ M₃ = 2 * Real.sqrt 6 := 
sorry

end distance_from_point_to_plane_l670_670696


namespace container_ratio_l670_670640

theorem container_ratio (V1 V2 V3 : ℝ)
  (h1 : (3 / 4) * V1 = (5 / 8) * V2)
  (h2 : (5 / 8) * V2 = (1 / 2) * V3) :
  V1 / V3 = 1 / 2 :=
by
  sorry

end container_ratio_l670_670640


namespace vector_in_plane_l670_670427

noncomputable def matrix_vector := ℝ

structure Vector3 :=
(x : matrix_vector)
(y : matrix_vector)
(z : matrix_vector)

def dot_product (v1 v2 : Vector3) : matrix_vector :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

def proj (w v : Vector3) : Vector3 :=
  let scalar := dot_product v w / dot_product w w in
  ⟨scalar * w.x, scalar * w.y, scalar * w.z⟩

theorem vector_in_plane (v w : Vector3) (h : proj w v = w) :
  3 * v.x - 3 * v.y + v.z = 19 :=
sorry

end vector_in_plane_l670_670427


namespace area_of_APRQ_l670_670898

theorem area_of_APRQ
  (A B C D P Q R : ℝ × ℝ)
  (h_rect : (A = (0, 0)) ∧ (B = (a, 0)) ∧ (C = (a, b)) ∧ (D = (0, b)))
  (h_area : a * b = 100)
  (h_points : P = (0, b / 4) ∧ Q = (a / 4, b) ∧ R = (a, 3 * b / 4)) :
  let APRQ_area := 6.25
  in ∃ area : ℝ, area = APRQ_area := sorry

end area_of_APRQ_l670_670898


namespace square_perimeter_of_N_l670_670909

theorem square_perimeter_of_N (area_M : ℝ) (area_N : ℝ) (side_N : ℝ) (perimeter_N : ℝ)
  (h1 : area_M = 100)
  (h2 : area_N = 4 * area_M)
  (h3 : area_N = side_N * side_N)
  (h4 : perimeter_N = 4 * side_N) :
  perimeter_N = 80 := 
sorry

end square_perimeter_of_N_l670_670909


namespace percentage_increase_in_items_sold_l670_670218

-- Definitions
variables (P N M : ℝ)
-- Given conditions:
-- The new price of an item
def new_price := P * 0.90
-- The relationship between incomes
def income_increase := (P * 0.90) * M = P * N * 1.125

-- The problem statement
theorem percentage_increase_in_items_sold (h : income_increase P N M) :
  M = N * 1.25 :=
sorry

end percentage_increase_in_items_sold_l670_670218


namespace age_ordering_l670_670408

theorem age_ordering (S T A : ℕ) (h1 : S = 2 * T) (h2 : A = T) :
  (S > T ∧ T > A) ∧ (S > A) :=
by {
  sorry,
}

end age_ordering_l670_670408


namespace profit_percent_is_20_l670_670577

variable (C S : ℝ)

-- Definition from condition: The cost price of 60 articles is equal to the selling price of 50 articles
def condition : Prop := 60 * C = 50 * S

-- Definition of profit percent to be proven as 20%
def profit_percent_correct : Prop := ((S - C) / C) * 100 = 20

theorem profit_percent_is_20 (h : condition C S) : profit_percent_correct C S :=
sorry

end profit_percent_is_20_l670_670577


namespace length_of_TU_l670_670211

-- Define the conditions from step a
variable (QR SU PQ : ℝ)
variable (h₀ : QR = 30)
variable (h₁ : SU = 18)
variable (h₂ : PQ = 15)

-- State the equivalence of the triangles
variable (h₃ : ∃ (TU : ℝ), ∀ (P Q R T S U : ℝ), TriangleSim P Q R T S U)

-- Prove that TU equals 9.0 cm
theorem length_of_TU (h₀ : QR = 30) (h₁ : SU = 18) (h₂ : PQ = 15) (h₃ : ∃ (TU : ℝ), ∀ (P Q R T S U : ℝ), TriangleSim P Q R T S U) : 
  ∃ TU, TU = 9 :=
by
  sorry


end length_of_TU_l670_670211


namespace median_unchanged_after_removal_l670_670385

variable {α : Type} [LinearOrder α] 

/-- Prove that the median is the same before and after removing the highest and lowest scores. -/
theorem median_unchanged_after_removal (scores : List α) (h : 2 < scores.length) : 
  let sorted_scores := scores.qsort (· ≤ ·)
      remaining_scores := (sorted_scores.drop 1).init
  in sorted_scores.getMedian = remaining_scores.getMedian := by sorry

end median_unchanged_after_removal_l670_670385


namespace berries_difference_l670_670282

theorem berries_difference (total_berries : ℕ) (dima_rate : ℕ) (sergey_rate : ℕ)
  (sergey_berries_picked : ℕ) (dima_berries_picked : ℕ)
  (dima_basket : ℕ) (sergey_basket : ℕ) :
  total_berries = 900 →
  sergey_rate = 2 * dima_rate →
  sergey_berries_picked = 2 * (total_berries / 3) →
  dima_berries_picked = total_berries / 3 →
  sergey_basket = sergey_berries_picked / 2 →
  dima_basket = (2 * dima_berries_picked) / 3 →
  sergey_basket > dima_basket ∧ sergey_basket - dima_basket = 100 :=
by
  intro h_total h_rate h_sergey_picked h_dima_picked h_sergey_basket h_dima_basket
  sorry

end berries_difference_l670_670282


namespace mouse_can_pass_l670_670231

-- Definitions based on conditions
-- R is the initial radius of the sphere.
variable (R : ℝ)
-- The gap required for a mouse to pass through, typically around 16 centimeters.
def gap_required := 0.16  -- in meters

-- Main statement to be proven
theorem mouse_can_pass (h : ℝ) (condition : h = 1 / (2 * Real.pi)) : h > gap_required := 
sorry

end mouse_can_pass_l670_670231


namespace total_time_to_complete_job_l670_670474

-- Conditions: Define the working days for each person
def sam_days := 4
def lisa_days := 6
def tom_days := 2
def jessica_days := 3

-- Definition: Compute the rates of working
def sam_rate := 1 / sam_days
def lisa_rate := 1 / lisa_days
def tom_rate := 1 / tom_days
def jessica_rate := 1 / jessica_days

-- Compute the total rate when working together
def total_rate := sam_rate + lisa_rate + tom_rate + jessica_rate

-- The final proof statement
theorem total_time_to_complete_job : 1 / total_rate = 0.8 :=
by sorry

end total_time_to_complete_job_l670_670474


namespace count_less_than_0_4_l670_670345

theorem count_less_than_0_4 :
  let numbers := [{0.8}, {(1 : ℝ) / 2}, {0.9}, {(1 : ℝ) / 3}]
  let threshold := (0.4 : ℝ)
  ∃ (ls : List ℝ), ↑ls.count (λ x, x < threshold) = 1 := 
by
  let numbers := [0.8, (1 : ℝ) / 2, 0.9, (1 : ℝ) / 3]
  let threshold := (0.4 : ℝ)
  use numbers.filter (< threshold)
  have : [((1 : ℝ) / 3)].length = 1 := by simp [List.filter, List.length]
  exact this

end count_less_than_0_4_l670_670345


namespace incorrect_inequality_exists_l670_670568

theorem incorrect_inequality_exists :
  ∃ (x y : ℝ), x < y ∧ x^2 ≥ y^2 :=
by {
  sorry
}

end incorrect_inequality_exists_l670_670568


namespace least_possible_value_existance_l670_670449

-- Defining the existence of x, y, z, w with the given congruence conditions
theorem least_possible_value_existance :
  ∃ (x y z w : ℕ),
    (x % 9 = 2) ∧ (x % 7 = 4) ∧
    (y % 11 = 3) ∧ (y % 13 = 12) ∧
    (z % 17 = 8) ∧ (z % 19 = 6) ∧
    (w % 23 = 5) ∧ (w % 29 = 10) ∧
    ((y + z) - (x + w) = -326) :=
  sorry

end least_possible_value_existance_l670_670449


namespace average_of_primes_less_than_twenty_l670_670549

def primes_less_than_twenty : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]
def sum_primes : ℕ := 77
def count_primes : ℕ := 8
def average_primes : ℚ := 77 / 8

theorem average_of_primes_less_than_twenty : (primes_less_than_twenty.sum / count_primes : ℚ) = 9.625 := by
  sorry

end average_of_primes_less_than_twenty_l670_670549


namespace balloons_difference_l670_670210

theorem balloons_difference : 
  let James_balloons := 232
      Amy_balloons := 101 in
  James_balloons - Amy_balloons = 131 := by
  let James_balloons := 232
  let Amy_balloons := 101
  show James_balloons - Amy_balloons = 131
  sorry

end balloons_difference_l670_670210


namespace s_cmp_2_cp_squared_l670_670065

theorem s_cmp_2_cp_squared (a : ℝ) :
  ∀ (P : ℝ), P ∈ set.Icc 0 (a * real.sqrt 5) →
  let s := 2 * P^2 - 2 * a * real.sqrt 5 * P + 5 * a^2,
      CP_squared := a^2,
      two_CP_squared := 2 * CP_squared
  in (∃ P, s < two_CP_squared) ∧
     (P = 0 ∨ P = a * real.sqrt 5 ∨ P = (a * real.sqrt 5 / 2)) →
     (∃ P, s > two_CP_squared) ∧
     ∀ P, (s = two_CP_squared ↔ P = a * real.sqrt 5 / 2) :=
sorry

end s_cmp_2_cp_squared_l670_670065


namespace no_congruent_partition_of_closed_disk_l670_670470

noncomputable def closed_disk (r : ℝ) : set (ℝ × ℝ) :=
  {p | (p.1 ^ 2 + p.2 ^ 2) ≤ r ^ 2}

theorem no_congruent_partition_of_closed_disk (r : ℝ) (H1 H2 : set (ℝ × ℝ)) (O1 O2 : ℝ × ℝ) (hO1 : O1 = (0, 0)) (hO2 : O2 ≠ (0, 0))
  (hH1H2 : (closed_disk r) = H1 ∪ H2 ∧ H1 ∩ H2 = ∅ ∧ H1 ≃₂ H2 ∧ O1 ∈ H1 ∧ O2 ∈ H2) : false :=
sorry

end no_congruent_partition_of_closed_disk_l670_670470


namespace count_triples_eq_l670_670422

variables {G : Type*} [group G] [fintype G]
variables {A B C : finset G}
variables (N : finset G → finset G → finset G → ℕ)
  -- Define N as the number of (x, y, z) in U x V x W for which xyz = e
  (hN : ∀ U V W : finset G, N U V W = (U.product (V.product W)).card { p | p.1 * p.2.1 * p.2.2 = 1 })
  -- Assume G is partitioned into A, B and C (i.e., pairwise disjoint and G = A ∪ B ∪ C)
  [decidable_pred (∈ A ∪ B ∪ C)]
  (hAB_disjoint : disjoint A B)
  (hAC_disjoint : disjoint A C)
  (hBC_disjoint : disjoint B C)
  (hG_partition : (A ∪ B ∪ C) = @finset.univ G _)

theorem count_triples_eq (h_group_finite : fintype.card G < ∞) :
  N A B C = N C B A :=
sorry

end count_triples_eq_l670_670422


namespace volume_of_triangular_pyramid_l670_670302

theorem volume_of_triangular_pyramid
  (a b : ℝ) (β : ℝ)
  (h1 : a = (36 * b^2 * (Real.cos β)^2)^0.5 / sqrt (1 + 9 * (Real.cos β)^2))
  : 
  let volume := (36 * b^2 * (Real.cos β)^2 / (1 + 9 * (Real.cos β)^2))^(3/2) * Real.tan β / 24 in
  volume = (a^3 * Real.tan β / 24) :=
sorry

end volume_of_triangular_pyramid_l670_670302


namespace mary_needs_to_add_6_25_more_cups_l670_670884

def total_flour_needed : ℚ := 8.5
def flour_already_added : ℚ := 2.25
def flour_to_add : ℚ := total_flour_needed - flour_already_added

theorem mary_needs_to_add_6_25_more_cups :
  flour_to_add = 6.25 :=
sorry

end mary_needs_to_add_6_25_more_cups_l670_670884


namespace range_m_l670_670748

theorem range_m (m : ℝ) :
  (∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ abs (x - m) < 1) →
  -1 / 2 ≤ m ∧ m ≤ 4 / 3 :=
by
  intro h
  sorry

end range_m_l670_670748


namespace digit_B_divisible_by_9_l670_670503

theorem digit_B_divisible_by_9 (B : ℕ) (h1 : B ≤ 9) (h2 : (4 + B + B + 1 + 3) % 9 = 0) : B = 5 := 
by {
  /- Proof omitted -/
  sorry
}

end digit_B_divisible_by_9_l670_670503


namespace problem_4031_l670_670333

def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

theorem problem_4031 :
  (f 1) + (∑ i in finset.range 2006, f (i + 2) + f ((i + 2)⁻¹)) = 4031 / 2 :=
by
  sorry

end problem_4031_l670_670333


namespace opposite_of_neg_2023_l670_670929

theorem opposite_of_neg_2023 : -( -2023 ) = 2023 := by
  sorry

end opposite_of_neg_2023_l670_670929


namespace triangle_degenerate_l670_670050

-- Defining the side lengths of the original triangle.
def PQ := 15
def PR := 8
def QR := 13

-- Defining the modified side lengths.
def PQ' := 3 * PQ
def PR' := 2 * PR
def QR' := QR

-- Declaring the theorem that the modified triangle is degenerate (collinear points).
theorem triangle_degenerate : PQ' + PR' = QR' ∨ PQ' + QR' = PR' ∨ PR' + QR' = PQ' :=
by
  -- Using the given conditions and the defined side lengths:
  -- new side lengths are PQ' = 45, PR' = 16, QR' = 13.
  -- Checking the conditions for degeneracy:
  -- PR' + QR' = 16 + 13 = 29 which is less than PQ' = 45.
  -- Hence, the triangle degenerates into a line.
  exact Or.inr (Or.inr (by simp [PQ', PR', QR']))

end triangle_degenerate_l670_670050


namespace loop_n3_time_complexity_l670_670480

-- Definitions based on identified conditions
def costPolyMult (r n : ℕ) : ℕ := r^2 * (log n)^2
def costAdd (r n : ℕ) : ℕ := r^2 * log n
def totalCost (r n : ℕ) : ℕ := log n * costPolyMult r n
-- Big O notation as a definition
def bigO (f g : ℕ → ℕ) := ∃ c, ∀ x, f x ≤ c * g x

-- Proof problem statement
theorem loop_n3_time_complexity (r n : ℕ) : 
  bigO (λ n, totalCost r n) (λ n, r^2 * (log n)^3) :=
sorry

end loop_n3_time_complexity_l670_670480


namespace set_powerset_cardinality_l670_670926

-- Given sets M and N with M ∪ N = {a, b}, prove that there are 9 distinct pairs.
theorem set_powerset_cardinality : 
  (∃ (M N : Set ℕ), (M ∪ N = {1, 2})) ∧ 
  (Finite { (M, N) : Set ℕ × Set ℕ | M ∪ N = {1, 2}}) ∧
  (Fintype.card { (M, N) : Set ℕ × Set ℕ | M ∪ N = {1, 2}} = 9) := 
  sorry -- proof needed

end set_powerset_cardinality_l670_670926


namespace remainder_a25_div_26_l670_670074

def concatenate_numbers (n : ℕ) : ℕ :=
  -- Placeholder function for concatenating numbers from 1 to n
  sorry

theorem remainder_a25_div_26 :
  let a_25 := concatenate_numbers 25
  a_25 % 26 = 13 :=
by sorry

end remainder_a25_div_26_l670_670074


namespace initial_amount_of_money_l670_670534

-- Define the costs and purchased quantities
def cost_tshirt : ℕ := 8
def cost_keychain_set : ℕ := 2
def cost_bag : ℕ := 10
def tshirts_bought : ℕ := 2
def bags_bought : ℕ := 2
def keychains_bought : ℕ := 21

-- Define derived quantities
def sets_of_keychains_bought : ℕ := keychains_bought / 3

-- Define the total costs
def total_cost_tshirts : ℕ := tshirts_bought * cost_tshirt
def total_cost_bags : ℕ := bags_bought * cost_bag
def total_cost_keychains : ℕ := sets_of_keychains_bought * cost_keychain_set

-- Define the initial amount of money
def total_initial_amount : ℕ := total_cost_tshirts + total_cost_bags + total_cost_keychains

-- The theorem proving the initial amount Timothy had
theorem initial_amount_of_money : total_initial_amount = 50 := by
  -- The proof is not required, so we use sorry to skip it
  sorry

end initial_amount_of_money_l670_670534


namespace john_gym_visits_per_week_l670_670850

theorem john_gym_visits_per_week 
  (hours_lifting_per_day : ℝ)
  (fraction_cardio_warmup : ℝ)
  (total_hours_per_week : ℝ) :
  hours_lifting_per_day = 1 → 
  fraction_cardio_warmup = 1/3 →
  total_hours_per_week = 4 →
  let daily_gym_time := hours_lifting_per_day + fraction_cardio_warmup * hours_lifting_per_day,
      weekly_gym_visits := total_hours_per_week / daily_gym_time
  in weekly_gym_visits ≈ 3 :=
by
  intros h_lifting h_fraction h_total
  let daily_gym_time := hours_lifting_per_day + fraction_cardio_warmup * hours_lifting_per_day
  let weekly_gym_visits := total_hours_per_week / daily_gym_time
  have h_daily : daily_gym_time = 1 + 1/3 := by 
    rw [h_lifting, h_fraction, mul_one]
  have h_visits : weekly_gym_visits = 4 / (1 + 1/3) := by 
    rw [h_total, h_daily]
  have : (4 / (1 + 1/3)) ≈ 3 := by
    norm_num
  exact this

end john_gym_visits_per_week_l670_670850


namespace cover_with_L_shapes_l670_670270

def L_shaped (m n : ℕ) : Prop :=
  m > 1 ∧ n > 1 ∧ ∃ k, m * n = 8 * k -- Conditions and tiling pattern coverage.

-- Problem statement as a theorem
theorem cover_with_L_shapes (m n : ℕ) (h1 : m > 1) (h2 : n > 1) : (∃ k, m * n = 8 * k) ↔ L_shaped m n :=
-- Placeholder for the proof
sorry

end cover_with_L_shapes_l670_670270


namespace joan_gemstone_samples_l670_670412

theorem joan_gemstone_samples
  (minerals_yesterday : ℕ)
  (gemstones : ℕ)
  (h1 : minerals_yesterday + 6 = 48)
  (h2 : gemstones = minerals_yesterday / 2) :
  gemstones = 21 :=
by
  sorry

end joan_gemstone_samples_l670_670412


namespace problem_statement_l670_670431

noncomputable def floor_T (u v w x : ℝ) : ℤ :=
  ⌊u + v + w + x⌋

theorem problem_statement (u v w x : ℝ) (T : ℝ) (h₁: u^2 + v^2 = 3005) (h₂: w^2 + x^2 = 3005) (h₃: u * w = 1729) (h₄: v * x = 1729) :
  floor_T u v w x = 155 :=
by
  sorry

end problem_statement_l670_670431


namespace wheel_speed_is_12_l670_670077

-- Define the conditions

def circumference : ℝ := 15 / 5280  -- Circumference in miles, as 5280 feet = 1 mile
def time_decrease : ℝ := 1 / 14400  -- Time decrease in hours, as 1/4 second = 1/14400 hours
def speed_increase : ℝ := 6  -- Speed increase in miles per hour

-- Define the main theorem statement
theorem wheel_speed_is_12 (r t : ℝ) (hr_positive : 0 < r) (ht_positive : 0 < t) 
  (initial_eq : r * t = circumference)
  (final_eq : (r + speed_increase) * (t - time_decrease) = circumference) :
  r = 12 :=
by
  sorry

end wheel_speed_is_12_l670_670077


namespace arithmetic_mean_of_periodic_digits_l670_670617

theorem arithmetic_mean_of_periodic_digits (q p : ℕ) (k : ℕ) (digits : ℕ → ℕ) : 
  (∀ i, digits i < 10) → odd_prime p → p ≠ 5 → 
  (∃ n, 10^k ≡ 1 [MOD p]) → (∀ i, digits i < 10) → 
  (if k % 2 = 0 
   then (1 / k) * (∑ i in finset.range k, digits i) = 4.5 
   else (1 / k) * (∑ i in finset.range k, digits i) ≠ 4.5) :=
begin
  intros h_digits_prime h_odd_prime h_prime_neq_five h_period h_digit_bound,
  sorry
end

-- Helper Definition for odd prime
def odd_prime (n : ℕ) : Prop :=
  prime n ∧ n % 2 = 1

end arithmetic_mean_of_periodic_digits_l670_670617


namespace cat_finishes_food_on_sunday_l670_670900

-- Define the constants and parameters
def daily_morning_consumption : ℚ := 2 / 5
def daily_evening_consumption : ℚ := 1 / 5
def total_food : ℕ := 8
def days_in_week : ℕ := 7

-- Define the total daily consumption
def total_daily_consumption : ℚ := daily_morning_consumption + daily_evening_consumption

-- Define the sum of consumptions over each day until the day when all food is consumed
def food_remaining_after_days (days : ℕ) : ℚ := total_food - days * total_daily_consumption

-- Proposition that the food is finished on Sunday
theorem cat_finishes_food_on_sunday :
  ∃ days : ℕ, (food_remaining_after_days days ≤ 0) ∧ days ≡ 7 [MOD days_in_week] :=
sorry

end cat_finishes_food_on_sunday_l670_670900


namespace count_CONES_paths_l670_670659

def diagram : List (List Char) :=
  [[' ', ' ', 'C', ' ', ' ', ' '],
   [' ', 'C', 'O', 'C', ' ', ' '],
   ['C', 'O', 'N', 'O', 'C', ' '],
   [' ', 'N', 'E', 'N', ' ', ' '],
   [' ', ' ', 'S', ' ', ' ', ' ']]

def is_adjacent (pos1 pos2 : (Nat × Nat)) : Bool :=
  (pos1.1 = pos2.1 ∨ pos1.1 + 1 = pos2.1 ∨ pos1.1 = pos2.1 + 1) ∧
  (pos1.2 = pos2.2 ∨ pos1.2 + 1 = pos2.2 ∨ pos1.2 = pos2.2 + 1)

def valid_paths (diagram : List (List Char)) : Nat :=
  -- Implementation of counting paths that spell "CONES" skipped
  sorry

theorem count_CONES_paths (d : List (List Char)) 
  (h : d = [[' ', ' ', 'C', ' ', ' ', ' '],
            [' ', 'C', 'O', 'C', ' ', ' '],
            ['C', 'O', 'N', 'O', 'C', ' '],
            [' ', 'N', 'E', 'N', ' ', ' '],
            [' ', ' ', 'S', ' ', ' ', ' ']]): valid_paths d = 6 := 
by
  sorry

end count_CONES_paths_l670_670659


namespace function_satisfies_conditions_l670_670618

-- Conditions
variable (f : ℝ → ℝ)
variable (h1 : ∀ x, f(-x) + f(x) = 0) -- Condition 2
variable (h2 : ∀ x1 x2, x1 ≠ x2 → (f(x1) - f(x2)) / (x1 - x2) > 0) -- Condition 3

-- Proof Problem Statement
theorem function_satisfies_conditions : (f = (λ x, x)) := 
sorry

end function_satisfies_conditions_l670_670618


namespace min_value_of_f_five_l670_670820

open Real

def f (x a : ℝ) : ℝ := abs (x + 1) + 2 * abs (x - a)

theorem min_value_of_f_five (a : ℝ) :
  (∃ x, f x a = 5) → (a = -6 ∨ a = 4) :=
sorry

end min_value_of_f_five_l670_670820


namespace helmet_price_for_given_profit_helmet_price_for_max_profit_l670_670223

section helmet_sales

-- Define the conditions
variable (original_price : ℝ := 80) (initial_sales : ℝ := 200) (cost_price : ℝ := 50) 
variable (price_reduction_unit : ℝ := 1) (additional_sales_per_reduction : ℝ := 10)
variable (minimum_price_reduction : ℝ := 10)

-- Profits
def profit (x : ℝ) : ℝ :=
  (original_price - x - cost_price) * (initial_sales + additional_sales_per_reduction * x)

-- Prove the selling price when profit is 5250 yuan
theorem helmet_price_for_given_profit (GDP : profit 15 = 5250) : (original_price - 15) = 65 :=
by
  sorry

-- Prove the price for maximum profit
theorem helmet_price_for_max_profit : 
  ∃ x, x = 10 ∧ (original_price - x = 70) ∧ (profit x = 6000) :=
by 
  sorry

end helmet_sales

end helmet_price_for_given_profit_helmet_price_for_max_profit_l670_670223


namespace sandy_money_l670_670902

theorem sandy_money (X : ℝ) (h1 : 0.70 * X = 224) : X = 320 := 
by {
  sorry
}

end sandy_money_l670_670902


namespace number_of_divisors_of_12m2_l670_670076

theorem number_of_divisors_of_12m2
  (m : ℕ) (h1 : m % 2 = 0) (h2 : (nat.factors m).prod = 7) :
  nat.divisors_count (12 * m ^ 2) = 30 :=
sorry

end number_of_divisors_of_12m2_l670_670076


namespace negation_correct_l670_670513

-- Define the original statement as a predicate
def original_statement (x : ℝ) : Prop := x > 1 → x^2 ≤ x

-- Define the negation of the original statement as a predicate
def negated_statement : Prop := ∃ x : ℝ, x > 1 ∧ x^2 > x

-- Define the theorem that the negation of the original statement implies the negated statement
theorem negation_correct :
  ¬ (∀ x : ℝ, original_statement x) ↔ negated_statement := by
  sorry

end negation_correct_l670_670513


namespace simson_lines_intersect_properties_l670_670446

noncomputable def circumcircle (A B C : Point) : Circle := sorry  -- Placeholder for circumcircle definition
noncomputable def altitude (A B C : Point) : Line := sorry  -- Placeholder for altitude definition
noncomputable def simsonLine (P : Point) (A B C : Point) : Line := sorry  -- Placeholder for Simson line definition
noncomputable def triangleArea (A B C : Point) : ℝ := sorry  -- Placeholder for area calculation definition

variables {A B C A1 B1 C1 A2 B2 C2 A3 B3 C3 : Point}
variables (hA2 : (circumcircle A B C).intersect (altitude A B C) = A2)
variables (hB2 : (circumcircle A B C).intersect (altitude B A C) = B2)
variables (hC2 : (circumcircle A B C).intersect (altitude C A B) = C2)
variables (hA1 : A1 ∈ altitude A B C)
variables (hB1 : B1 ∈ altitude B A C)
variables (hC1 : C1 ∈ altitude C A B)
variables (hA3B3C3 : ∃ A3 B3 C3,
  A3 ∈ simsonLine A2 A B C ∧
  B3 ∈ simsonLine B2 A B C ∧
  C3 ∈ simsonLine C2 A B C)

theorem simson_lines_intersect_properties :
  (∃ A3 B3 C3, similar (A3 B3 C3) (A1 B1 C1)) ∧
  triangleArea A3 B3 C3 = 4 * triangleArea A1 B1 C1 :=
by
  sorry

end simson_lines_intersect_properties_l670_670446


namespace periodic_extension_l670_670741

noncomputable def f : ℝ → ℝ := λ x, if 0 ≤ x ∧ x < 2 then x^3 - x else sorry

theorem periodic_extension (x : ℝ) (hx : -2 ≤ x ∧ x < 0) :
  f(x) = x^3 + 6*x^2 + 11*x + 6 :=
by
  -- f is periodic with period 2, given by conditions
  have periodic : ∀ y : ℝ, f (y + 2) = f y := sorry,
  -- f(x) = (x + 2)^3 - (x + 2) when x in [-2, 0)
  have h : f(x + 2) = (x + 2)^3 - (x + 2), from sorry,
  rw [periodic, h],
  -- Therefore, the expression for f(x) when x ∈ [-2, 0) is x^3 + 6x^2 + 11x + 6
  sorry

end periodic_extension_l670_670741


namespace num_solutions_eq_8_l670_670865

noncomputable def f (x : ℝ) : ℝ :=
if ∃ (n : ℕ+), x = (n - 1) / n then x^2 else x

theorem num_solutions_eq_8 : (set.count { x : ℝ | f x = Real.log x }) = 8 := 
sorry

end num_solutions_eq_8_l670_670865


namespace max_ab_value_l670_670862

theorem max_ab_value {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : 6 * a + 8 * b = 72) : ab = 27 :=
by {
  sorry
}

end max_ab_value_l670_670862


namespace tan_x_eq_2_trigonometric_expression_value_l670_670737

noncomputable def sin (x : ℝ) : ℝ := sorry
noncomputable def cos (x : ℝ) : ℝ := sorry
noncomputable def tan (x : ℝ) : ℝ := sin x / cos x

theorem tan_x_eq_2
    (x : ℝ)
    (h : (sin x + cos x) / (sin x - cos x) = 3) :
    tan x = 2 := sorry

theorem trigonometric_expression_value
    (x : ℝ)
    (h1 : (sin x + cos x) / (sin x - cos x) = 3)
    (h2 : π < x ∧ x < 3 * π / 2) :
    sqrt((1 + sin x) / (1 - sin x)) - sqrt((1 - sin x) / (1 + sin x)) = -4 :=
begin
    sorry
end

end tan_x_eq_2_trigonometric_expression_value_l670_670737


namespace volume_of_red_tetrahedron_l670_670612

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length^3

def volume_of_tetrahedron (base_area : ℕ) (height : ℕ) : ℚ :=
  (1/3 : ℚ) * base_area * height

def smaller_tetrahedron_volume (side_length : ℕ) : ℚ :=
  volume_of_tetrahedron ((1/2 : ℚ) * side_length^2) side_length

theorem volume_of_red_tetrahedron :
  let cube_side_length := 8 in
  let cube_volume := volume_of_cube cube_side_length in
  let smaller_tetrahedrons_volume := 4 * smaller_tetrahedron_volume cube_side_length in
  cube_volume - smaller_tetrahedrons_volume = 512 / 3 :=
by
  sorry

end volume_of_red_tetrahedron_l670_670612


namespace knights_A_and_B_l670_670321

-- Definitions of knight and liar
def is_knight (A: Type) [Decidable A] := A
def is_liar (A: Type) [Decidable A] := ¬ A

-- Statements of A and B being knights
variable (A B : Prop)

-- A's statement condition
def A_statement : Prop := is_liar A ∨ is_knight B

-- Prove that A and B are both knights given A's statement
theorem knights_A_and_B (A B : Prop) (h: A_statement A B) : is_knight A ∧ is_knight B :=
by {
  sorry
}

end knights_A_and_B_l670_670321


namespace prove_triangle_sides_l670_670776

noncomputable def radius : ℝ := 4
def AD : ℝ := 6
def EC : ℝ := 8
def BD_BE : ℝ := 7

theorem prove_triangle_sides (r AD EC BD_BE : ℝ) (h_r : r = 4) (h_AD : AD = 6) (h_EC : EC = 8) (h_BD_BE : BD_BE = 7) :
  (6 + 7 = 13) ∧ (7 + 8 = 15) :=
by
  have h_AB : 6 + 7 = 13,
    exact (by norm_num : (6 : ℝ) + 7 = 13),
  have h_BC : 7 + 8 = 15,
    exact (by norm_num : (7 : ℝ) + 8 = 15),
  exact ⟨h_AB, h_BC⟩ 

end prove_triangle_sides_l670_670776


namespace fuel_for_empty_plane_per_mile_l670_670262

theorem fuel_for_empty_plane_per_mile :
  let F := 106000 / 400 - (35 * 3 + 70 * 2)
  F = 20 := 
by
  sorry

end fuel_for_empty_plane_per_mile_l670_670262


namespace berries_difference_l670_670283

theorem berries_difference (total_berries : ℕ) (dima_rate : ℕ) (sergey_rate : ℕ)
  (sergey_berries_picked : ℕ) (dima_berries_picked : ℕ)
  (dima_basket : ℕ) (sergey_basket : ℕ) :
  total_berries = 900 →
  sergey_rate = 2 * dima_rate →
  sergey_berries_picked = 2 * (total_berries / 3) →
  dima_berries_picked = total_berries / 3 →
  sergey_basket = sergey_berries_picked / 2 →
  dima_basket = (2 * dima_berries_picked) / 3 →
  sergey_basket > dima_basket ∧ sergey_basket - dima_basket = 100 :=
by
  intro h_total h_rate h_sergey_picked h_dima_picked h_sergey_basket h_dima_basket
  sorry

end berries_difference_l670_670283


namespace domain_of_f_x_l670_670744

theorem domain_of_f_x {f : ℝ → ℝ} 
  (h : ∀ x, x ∈ set.Icc (-2 : ℝ) 3 → (x + 1) ∈ set.Icc (-2 : ℝ) 3) :
  ∀ y, y ∈ set.Icc (-3 : ℝ) 2 → (y ∈ set.Icc (-3 : ℝ) 2) :=
by
  -- Provide the necessary structure but leave the proof as sorry.
  sorry

end domain_of_f_x_l670_670744


namespace polynomial_ascending_l670_670646

theorem polynomial_ascending (x : ℝ) :
  (x^2 - 2 - 5*x^4 + 3*x^3) = (-2 + x^2 + 3*x^3 - 5*x^4) :=
by sorry

end polynomial_ascending_l670_670646


namespace train_speed_correct_l670_670246

noncomputable def train_speed (length_meters : ℕ) (time_seconds : ℕ) : ℝ :=
  (length_meters : ℝ) / 1000 / (time_seconds / 3600)

theorem train_speed_correct :
  train_speed 2500 50 = 180 := 
by
  -- We leave the proof as sorry, the statement is sufficient
  sorry

end train_speed_correct_l670_670246


namespace quadratic_intersects_x_axis_if_and_only_if_k_le_four_l670_670768

-- Define the quadratic function
def quadratic_function (k x : ℝ) : ℝ :=
  (k - 3) * x^2 + 2 * x + 1

-- Theorem stating the relationship between the function intersecting the x-axis and k ≤ 4
theorem quadratic_intersects_x_axis_if_and_only_if_k_le_four
  (k : ℝ) :
  (∃ x : ℝ, quadratic_function k x = 0) ↔ k ≤ 4 :=
sorry

end quadratic_intersects_x_axis_if_and_only_if_k_le_four_l670_670768


namespace rational_solutions_product_l670_670526

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem rational_solutions_product :
  ∀ c : ℕ, (c > 0) → (is_perfect_square (49 - 12 * c)) → (∃ a b : ℕ, a = 4 ∧ b = 2 ∧ a * b = 8) :=
by sorry

end rational_solutions_product_l670_670526


namespace inequality_proof_l670_670084

theorem inequality_proof
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (6841 * x - 1) / 9973 + (9973 * y - 1) / 6841 = z) :
  x / 9973 + y / 6841 > 1 :=
sorry

end inequality_proof_l670_670084


namespace arithmetic_sum_l670_670044

open_locale classical

noncomputable def arithmetic_sequence (a : ℕ → ℤ) := ∀ n m, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sum (a : ℕ → ℤ) (h_seq : arithmetic_sequence a) (h1 : a 3 = 4) (h2 : a 101 = 36) :
  a 9 + a 52 + a 95 = 60 :=
sorry

end arithmetic_sum_l670_670044


namespace min_area_tangent_ellipse_l670_670633

theorem min_area_tangent_ellipse (a : ℝ) (a_pos : 0 < a) :
  ∃ A B : ℝ × ℝ, (A.1 = a ∨ A.2 = 4 * a) ∧ (B.1 = a ∨ B.2 = 4 * a) ∧ 
  (¬ (A = (0,0)) ∧ ¬ (B = (0,0))) ∧
  (let S := 1 / 2 * abs (A.1 * B.2 - B.1 * A.2) in S = 1/2 * a^2) :=
begin
  sorry
end

end min_area_tangent_ellipse_l670_670633


namespace product_of_distances_to_foci_l670_670308

open Real

theorem product_of_distances_to_foci 
  (a : ℝ) (h : a = 1) (C : ℝ → ℝ → Prop) (x y c : ℝ)
  (P : ℝ × ℝ) (hC : ∀ x y, C x y ↔ x^2 - y^2 = 1)
  (F1 F2 : ℝ × ℝ)
  (hF1 : F1 = (-c, 0)) (hF2 : F2 = (c, 0)) (hP : C (fst P) (snd P))
  (h_angle : ∠ F1 P F2 = 60) :
  |dist P F1 * dist P F2| = 4 :=
sorry

end product_of_distances_to_foci_l670_670308


namespace tangential_quadrilateral_l670_670584

theorem tangential_quadrilateral {A B C D : Point} 
  {r1 r2 r3 r4 : Real} {AB BC CD DA : Real} :
  circle_tangential_to_sides A D A B B C r1 → 
  circle_tangential_to_sides A B B C C D r2 →
  circle_tangential_to_sides B C C D D A r3 →
  circle_tangential_to_sides D A A B C D r4 →
  (AB / r1) + (CD / r3) = (BC / r2) + (DA / r4) :=
sorry

end tangential_quadrilateral_l670_670584


namespace perfect_squares_between_100_and_400_l670_670801

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l670_670801


namespace arc_length_BC_l670_670606

-- Define the conditions of the problem
def circumference : ℝ := 72
def angle_BAC_degrees : ℝ := 45

-- Define the formula to convert from degrees to proportion of the circle
def angle_BAC_proportion (angle_BAC_degrees : ℝ) : ℝ :=
  angle_BAC_degrees / 360

-- Define the length of the arc, given the proportion of the circle and the circumference
def arc_length (proportion : ℝ) (circumference : ℝ) : ℝ :=
  proportion * circumference

theorem arc_length_BC : arc_length (angle_BAC_proportion angle_BAC_degrees) circumference = 9 :=
by
  sorry

end arc_length_BC_l670_670606


namespace solve_x_eq_40_l670_670115

theorem solve_x_eq_40 : ∀ (x : ℝ), x + 2 * x = 400 - (3 * x + 4 * x) → x = 40 :=
by
  intro x
  intro h
  sorry

end solve_x_eq_40_l670_670115


namespace dustin_pages_per_hour_l670_670675

-- Defining constants and conditions
@[irreducible] def Sam_pages_per_hour : ℕ := 24
@[irreducible] def forty_minutes_to_hours: ℚ := 2 / 3
@[irreducible] def Dustin_extra_pages_in_40min: ℕ := 34

-- Defining the problem statement
theorem dustin_pages_per_hour :
  ∃ D : ℚ, D = 75 :=
by
  -- Use the conditions given in the problem to prove this theorem
  let Sam_pages_in_40min : ℚ := Sam_pages_per_hour * forty_minutes_to_hours
  let Dustin_pages_in_40min : ℚ := Sam_pages_in_40min + Dustin_extra_pages_in_40min
  have h : (forty_minutes_to_hours * (75 : ℚ)) = Dustin_pages_in_40min,
  { sorry },  -- Proof to be completed
  use 75,
  exact eq.symm h

end dustin_pages_per_hour_l670_670675


namespace rhombus_inscribed_circle_radius_l670_670826

theorem rhombus_inscribed_circle_radius (d1 d2 : ℝ) (h1 : d1 = 8) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) in
  let area := (d1 * d2) / 2 in
  let radius := area / (4 * a) in
  radius = 30 / Real.sqrt 241 :=
by
  { rw [h1, h2], -- Substitute d1 = 8, d2 = 30
    sorry }

end rhombus_inscribed_circle_radius_l670_670826


namespace neg_exists_lt_1000_l670_670774

open Nat

theorem neg_exists_lt_1000 : (¬ ∃ n : ℕ, 2^n < 1000) = ∀ n : ℕ, 2^n ≥ 1000 := by
  sorry

end neg_exists_lt_1000_l670_670774


namespace lines_perpendicular_to_same_plane_are_parallel_l670_670009

-- Define basic geometric entities
variables (m n : Line) (α β : Plane)

-- Define conditions
def non_coincident_lines : Prop := m ≠ n
def non_coincident_planes : Prop := α ≠ β
def m_perpendicular_to_alpha : Prop := m ⊥ α
def n_perpendicular_to_alpha : Prop := n ⊥ α

-- Problem statement
theorem lines_perpendicular_to_same_plane_are_parallel :
  non_coincident_lines m n →
  non_coincident_planes α β →
  m_perpendicular_to_alpha m α →
  n_perpendicular_to_alpha n α →
  Parallel m n :=
by
  -- Proof is not needed as per instruction
  sorry

end lines_perpendicular_to_same_plane_are_parallel_l670_670009


namespace sugar_percentage_l670_670465

theorem sugar_percentage (S : ℝ) (P : ℝ) : 
  (3 / 4 * S * 0.10 + (1 / 4) * S * P / 100 = S * 0.20) → 
  P = 50 := 
by 
  intro h
  sorry

end sugar_percentage_l670_670465


namespace problem_statement_1_problem_statement_2_problem_statement_3_problem_statement_3_not_necessary_problem_statement_4_l670_670987

theorem problem_statement_1 (P Q : Prop) : (P → Q) ↔ (Qᶜ → Pᶜ) := sorry

theorem problem_statement_2 (A B C : ℝ) (h1 : ∠B = 60) : (2 * ∠B = ∠A + ∠C) ↔ (∠A, ∠B, ∠C form an arithmetic sequence) := sorry

theorem problem_statement_3 (a b : ℝ) : (ab = 1) → (ax + y - 1 = 0 ∧ x + by - 1 = 0 → parallel) := sorry

theorem problem_statement_3_not_necessary (a b : ℝ) : (ab ≠ 1) ∧ (parallel ↔ ab - 1 = 0) := sorry

theorem problem_statement_4 (a b m : ℝ) : (am^2 < bm^2) ↔ (a < b) := sorry

end problem_statement_1_problem_statement_2_problem_statement_3_problem_statement_3_not_necessary_problem_statement_4_l670_670987


namespace cars_overtake_distance_l670_670168

def speed_red_car : ℝ := 30
def speed_black_car : ℝ := 50
def time_to_overtake : ℝ := 1
def distance_between_cars : ℝ := 20

theorem cars_overtake_distance :
  (speed_black_car - speed_red_car) * time_to_overtake = distance_between_cars :=
by sorry

end cars_overtake_distance_l670_670168


namespace original_card_deck_count_l670_670843

theorem original_card_deck_count (r b : ℕ) :
  (r : ℚ) / (r + b : ℚ) = 2 / 5 →
  (r + 3 : ℚ) / (r + b + 3 : ℚ) = 1 / 2 →
  r + b = 15 :=
by
  sorry

end original_card_deck_count_l670_670843


namespace sum_of_first_five_multiples_of_15_l670_670189

theorem sum_of_first_five_multiples_of_15 : (15 + 30 + 45 + 60 + 75) = 225 :=
by sorry

end sum_of_first_five_multiples_of_15_l670_670189


namespace defective_box_identification_l670_670923

noncomputable def find_defective_box (w : ℕ) : ℕ :=
  let expected_weight := 55 * 100 in
  w - expected_weight

theorem defective_box_identification (standard_weight defective_weight : ℕ) (num_boxes num_defective_parts num_standard_parts per_box: ℕ)
  (h1 : num_boxes = 10)
  (h2 : defective_weight = 101)
  (h3 : standard_weight = 100)
  (h4 : num_defective_parts = 10)
  (h5 : num_standard_parts = 90)
  (h6 : per_box = 9)
  (w : ℕ)
  (total_weight : w = 5500 + find_defective_box w) :
  find_defective_box w = ∑ i in finset.range num_boxes, (i + 1) :=
sorry

end defective_box_identification_l670_670923


namespace symmetry_of_log_graphs_l670_670673

theorem symmetry_of_log_graphs :
  (∀ x : ℝ, y = log 3 x → y = log (1/3) (9 * x)) = 
  (∀ x : ℝ, y = log 3 x → y = - log 3 x - 2) → 
  (∀ x : ℝ, - log 3 (9 * x) = log 3 x + 2) → 
  (∀ x : ℝ, symm == y = -1) :=
by
  sorry

end symmetry_of_log_graphs_l670_670673


namespace choir_members_max_l670_670605

theorem choir_members_max (x r m : ℕ) 
  (h1 : r * x + 3 = m)
  (h2 : (r - 3) * (x + 2) = m) 
  (h3 : m < 150) : 
  m = 759 :=
sorry

end choir_members_max_l670_670605


namespace exactly_three_distinct_solutions_l670_670437

theorem exactly_three_distinct_solutions (a : ℝ) : 
  (∃ (sols : set ℝ), sols = {x | abs (abs (x - a) - a) = 2} ∧ sols.card = 3) → a = 2 :=
by 
  sorry

end exactly_three_distinct_solutions_l670_670437


namespace hyperbola_asymptotes_l670_670498

theorem hyperbola_asymptotes (x y : ℝ) : x^2 - 4 * y^2 = -1 → (x = 2 * y) ∨ (x = -2 * y) := 
by
  intro h
  sorry

end hyperbola_asymptotes_l670_670498


namespace part1_part2_l670_670025

section TriangleProofs

variables (A B C a b c : ℝ) (A_pos : A > 0) (B_pos : B > 0) (C_pos : C > 0) (a_pos : a > 0) (b_pos : b > 0) (c_pos : c > 0)

-- Given conditions
def condition1 : Prop := sqrt 3 * a * sin B - b * cos A = b
def condition2 : Prop := b + c = 4

-- Questions restated as propositions
def prop1 : Prop := A = π / 3
def prop2 : Prop := let area := 1/2 * b * c * sin A in area = sqrt 3

theorem part1 (h : condition1) : prop1 := sorry

theorem part2 (h1 : condition1) (h2 : condition2) (a_min : a = 2) : prop2 := sorry

end TriangleProofs

end part1_part2_l670_670025


namespace count_prime_squares_between_bounds_l670_670002

-- Conditions
def square_bounds (p : ℕ) : Prop :=
  5000 < p^2 ∧ p^2 < 8000

-- Counting the primes within the given bounds
def primes_in_range : list ℕ :=
  [71, 73, 79, 83, 89]

theorem count_prime_squares_between_bounds :
  (primes_in_range.filter (λ p, p.prime)).length = 5 :=
by
  -- Theorem statement without proof
  sorry

end count_prime_squares_between_bounds_l670_670002


namespace slope_of_line_l670_670934

theorem slope_of_line {x y : ℝ} : 
  (∃ (x y : ℝ), 0 = 3 * x + 4 * y + 12) → ∀ (m : ℝ), m = -3/4 :=
by
  sorry

end slope_of_line_l670_670934


namespace point_in_plane_region_l670_670964

-- Defining the condition that the inequality represents a region on the plane
def plane_region (x y : ℝ) : Prop := x + 2 * y - 1 > 0

-- Stating that the point (0, 1) lies within the plane region represented by the inequality
theorem point_in_plane_region : plane_region 0 1 :=
by {
    sorry
}

end point_in_plane_region_l670_670964


namespace sum_of_zeros_of_transformed_parabola_l670_670141

noncomputable def transformed_parabola (x : ℝ) : ℝ :=
  -(x - 7)^2 + 7

theorem sum_of_zeros_of_transformed_parabola :
  let sum_zeros : ℝ := (7 + Real.sqrt 7) + (7 - Real.sqrt 7)
  in sum_zeros = 14 := by
sorry

end sum_of_zeros_of_transformed_parabola_l670_670141


namespace total_paintable_area_l670_670885

-- Define the dimensions of a bedroom
def bedroom_length : ℕ := 10
def bedroom_width : ℕ := 12
def bedroom_height : ℕ := 9

-- Define the non-paintable area per bedroom
def non_paintable_area_per_bedroom : ℕ := 74

-- Number of bedrooms
def number_of_bedrooms : ℕ := 4

-- The total paintable area that we need to prove
theorem total_paintable_area : 
  4 * (2 * (bedroom_length * bedroom_height) + 2 * (bedroom_width * bedroom_height) - non_paintable_area_per_bedroom) = 1288 := 
by
  sorry

end total_paintable_area_l670_670885


namespace total_distance_proof_l670_670624

-- Declare the variables and constants
variables (t_1 t_2 t_3 : ℝ) (d_1 d_2 d_3 total_distance : ℝ)

-- Define the conditions
def travel_conditions : Prop :=
  t_1 + t_2 + t_3 = 60 ∧
  d_1 = 20 * t_1 ∧
  d_2 = 15 * t_2 ∧
  d_3 = 27 * t_3 

-- Define the total distance formula
def calculate_total_distance : ℝ := d_1 + d_2 + d_3

-- The statement to prove that the total distance is as calculated
theorem total_distance_proof (h : travel_conditions t_1 t_2 t_3 d_1 d_2 d_3) :
  calculate_total_distance t_1 t_2 t_3 d_1 d_2 d_3 = 20 * t_1 + 15 * t_2 + 27 * t_3 := by
  sorry

end total_distance_proof_l670_670624


namespace zero_not_in_range_of_g_l670_670869

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈1 / (x + 3)⌉
  else ⌊1 / (x + 3)⌋

theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l670_670869


namespace minimum_disks_needed_l670_670880

-- Define the problem setup as constants
def total_files : ℕ := 30
def disk_capacity_MB : ℚ := 1.44
def file_sizes : list ℚ := [0.9, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 
                            0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.3, 0.3, 0.3, 0.3, 0.3]

-- Predicate to determine if the file arrangement is within disk capacity
def fits_on_disk (disk_cap : ℚ) (files : list ℚ) : Prop :=
  ∀ l ∈ files, l ≤ disk_cap

-- Defining the proof problem to demonstrate minimum disk usage
theorem minimum_disks_needed (files : list ℚ) (disk_cap : ℚ) : ℕ :=
  if h : fits_on_disk disk_cap files then
    sorry -- Proof required to show the exact arrangement to fit all files
  else
    sorry -- Fallback in case cannot fit (hypothetical, if miscalculation occurs)

-- Example Usage
example : ∃ n, fits_on_disk disk_capacity_MB file_sizes ∧ n = 15 :=
begin
  use minimum_disks_needed file_sizes disk_capacity_MB,
  split,
  { sorry, -- Proof of fits_on_disk will be detailed here
  },
  { sorry, -- Proof that the minimum number of disks is indeed 15
  }
end

end minimum_disks_needed_l670_670880


namespace johny_journey_distance_l670_670414

def south_distance : ℕ := 40
def east_distance : ℕ := south_distance + 20
def north_distance : ℕ := 2 * east_distance
def total_distance : ℕ := south_distance + east_distance + north_distance

theorem johny_journey_distance :
  total_distance = 220 := by
  sorry

end johny_journey_distance_l670_670414


namespace rebecca_total_money_l670_670896

noncomputable def total_money (haircuts perms dye_jobs tips cost_per_haircut cost_per_perm cost_per_dyejob hair_dye_cost_per_box : ℕ) : ℕ :=
  let earnings_from_haircuts := haircuts * cost_per_haircut
  let earnings_from_perms := perms * cost_per_perm
  let earnings_from_dyejobs := dye_jobs * cost_per_dyejob
  let total_earnings := earnings_from_haircuts + earnings_from_perms + earnings_from_dyejobs + tips
  let total_cost_of_hair_dye := dye_jobs * hair_dye_cost_per_box
  total_earnings - total_cost_of_hair_dye

theorem rebecca_total_money :
  total_money 4 1 2 50 30 40 60 10 = 310 :=
by
  simp [total_money]
  rfl

end rebecca_total_money_l670_670896


namespace geometric_sequence_a4_l670_670837

theorem geometric_sequence_a4 (a : ℕ → ℝ) (r : ℝ) (h3 : a 3 = 2) (h5 : a 5 = 16) :
  a 4 = 4 * real.sqrt 2 ∨ a 4 = -4 * real.sqrt 2 :=
by
  sorry

end geometric_sequence_a4_l670_670837


namespace complex_abs_eq_number_of_real_values_c_l670_670305

theorem complex_abs_eq (c : ℝ) : 
  (∣complex.mk (2 / 3) (-c)∣ = 5 / 6) ↔ (c = 1 / 2 ∨ c = -1 / 2) :=
by sorry

theorem number_of_real_values_c :
  {c : ℝ | ∣complex.mk (2 / 3) (-c)∣ = 5 / 6}.to_finset.card = 2 :=
by sorry

end complex_abs_eq_number_of_real_values_c_l670_670305


namespace difference_of_cubes_l670_670369

theorem difference_of_cubes (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (h3 : m^2 - n^2 = 43) : m^3 - n^3 = 1387 :=
by
  sorry

end difference_of_cubes_l670_670369


namespace locus_point_P_constant_value_FS_FT_l670_670311

-- Definitions of the given conditions
variables (a x y : ℝ)

-- Condition 1: Ellipse equation definition
def ellipse (a x y : ℝ) := (x^2) / (1 + a^2) + y^2 = 1

-- Condition 2: Point M on x-axis
def point_M' (m : ℝ) := (m, 0)

-- Condition 3: Point N on y-axis
def point_N' (n : ℝ) := (0, n)

-- Condition 4: Perpendicular vectors
def perpendicular_vectors (m n a : ℝ) := m * a = -n^2

-- Condition 5: Point P condition
def point_P_condition (m n x y : ℝ) := (m, 0) = (2 * 0, 2 * n) + (x, y)

-- Definition of focus F's coordinate
def focus_F (a : ℝ) := (a, 0)

-- Theorem 1: Equation of the locus of point P is y^2 = 4ax
theorem locus_point_P (a x y : ℝ) (m n : ℝ)
  (h₁ : ellipse a x y) 
  (h₂ : point_M' m) 
  (h₃ : point_N' n) 
  (h₄ : perpendicular_vectors m n a) 
  (h₅ : point_P_condition m n x y) 
  : y^2 = 4 * a * x :=
sorry

-- Theorem 2: Value of FS · FT is a constant 0
theorem constant_value_FS_FT (a : ℝ) (F S T : ℝ × ℝ)
  (h₁ : focus_F a = F) 
  (h₂ : F = S ∨ F = T ↔ ∃ P, ellipse a (fst P) (snd P))
  : F.1 * S.1 + F.2 * T.2 = 0 :=
sorry

end locus_point_P_constant_value_FS_FT_l670_670311


namespace second_graders_borrowed_books_l670_670142

theorem second_graders_borrowed_books
  (initial_books : ℕ)
  (remaining_books : ℕ)
  (initial_books_eq : initial_books = 75)
  (remaining_books_eq : remaining_books = 57)
  : initial_books - remaining_books = 18 :=
by
  rw [initial_books_eq, remaining_books_eq]
  sorry

end second_graders_borrowed_books_l670_670142


namespace pyramid_volume_correct_l670_670630

noncomputable def base_area : ℝ := 648 * (3 / 7)
noncomputable def side_length : ℝ := real.sqrt base_area
noncomputable def triangle_area : ℝ := (1 / 3) * base_area
noncomputable def height : ℝ := (2 * triangle_area) / side_length
noncomputable def pyramid_volume : ℝ := (1 / 3) * base_area * height

theorem pyramid_volume_correct : 
  pyramid_volume = (4232 * real.sqrt 6) / 9 := 
by
  sorry

end pyramid_volume_correct_l670_670630


namespace infinite_non_prime_seq_l670_670904

-- Let's state the theorem in Lean
theorem infinite_non_prime_seq (k : ℕ) : 
  ∃ᶠ n in at_top, ∀ i : ℕ, (1 ≤ i ∧ i ≤ k) → ¬ Nat.Prime (n + i) := 
sorry

end infinite_non_prime_seq_l670_670904


namespace area_of_quadrilateral_AMCD_l670_670208

theorem area_of_quadrilateral_AMCD (AB AD AM : ℝ)
  (h_AB : AB = 4)
  (h_AD : AD = 6)
  (h_AM : AM = 4 * Real.sqrt 3) :
  ∃ (A B C D M : ℝ × ℝ),
  parallelogram A B C D ∧
  bisects_angle A B D M ∧
  find_area_quadrilateral A M C D = 36 :=
by sorry

end area_of_quadrilateral_AMCD_l670_670208


namespace value_range_f_l670_670523

noncomputable def f (x : ℝ) : ℝ := sin x - sqrt 3 * cos x

theorem value_range_f:
  ∀ x ∈ set.Icc (0 : ℝ) (real.pi / 2), 
  ∃ y ∈ set.Icc (-sqrt 3) 1, f x = y := 
sorry

end value_range_f_l670_670523


namespace area_triangle_cpq_correct_l670_670423

noncomputable def area_triangle_cpq : ℚ :=
  let A := (0, 0: ℚ)
  let B := (10, 0: ℚ)
  let D := (0, 5: ℚ)
  let C := (10, 5: ℚ)
  let P := (5 / 2, 5: ℚ)
  let Q := (10, 5 / 4: ℚ)
  (1 / 2) * abs (10 * (5 - 5 / 4) + 5 / 2 * (5 / 4 - 5) + 10 * (5 - 5))

theorem area_triangle_cpq_correct :
  ∃ (A B C D P Q : (ℚ × ℚ)), 
    let A := (0, 0: ℚ)
    let B := (10, 0: ℚ)
    let D := (0, 5: ℚ)
    let C := (10, 5: ℚ)
    let P := (5 / 2, 5: ℚ)
    let Q := (10, 5 / 4: ℚ)
    rectangle A B C D ∧
    B.1 - A.1 = 10 ∧ D.2 - A.2 = 5 ∧
    point_on_segment P C D ∧ point_on_segment Q B C ∧
    slope B D = slope P Q ∧ right_angle (A, P, Q) ∧
    area_triangle_cpq = 75 / 8 :=
by
  sorry

end area_triangle_cpq_correct_l670_670423


namespace solve_problem_l670_670668

noncomputable def problem_statement : Prop :=
  let num := (Finset.sum (Finset.range 1000) (λ k, (1001 - k) / (k + 1)))
  let denom := (Finset.sum (Finset.range (1001 - 1)) (λ k, 1 / (k + 2)))
  num / denom = 1001

theorem solve_problem : problem_statement := by
  sorry

end solve_problem_l670_670668


namespace tiling_count_mod_1000_is_861_l670_670645

theorem tiling_count_mod_1000_is_861 : 
  let N := 24861 in 
  N % 1000 = 861 :=
  sorry

end tiling_count_mod_1000_is_861_l670_670645


namespace monotonicity_f_inequality_x1_x2_l670_670769

noncomputable def e : ℝ := Real.exp 1

def f (x : ℝ) : ℝ := (2 * e - x) * Real.log x

-- We will need the domain constraints for our variables
variables {x x1 x2 : ℝ} (h_x_pos : 0 < x)
variables (h_x_in_interval : x1 ∈ Set.Ioo 0 1) (h_x2_in_interval : x2 ∈ Set.Ioo 0 1)
variables (h_condition : x2 * Real.log x1 - x1 * Real.log x2 = 2 * e * x1 * x2 * (Real.log x1 - Real.log x2))

-- The proof statements themselves
theorem monotonicity_f :
  (∀ x : ℝ, x ∈ Set.Ioo 0 e → sorry) ∧ -- f(x) is strictly increasing in (0, e)
  (∀ x : ℝ, x ∈ Set.Ioo e (real.exp 1) → sorry) := -- f(x) is strictly decreasing in (e, +∞)
sorry

theorem inequality_x1_x2 :
  2 * e < (1 / x1) + (1 / x2) ∧ (1 / x1) + (1 / x2) < 2 * e + 1 :=
sorry

end monotonicity_f_inequality_x1_x2_l670_670769


namespace arrangement_count_l670_670716

theorem arrangement_count (A B C D E : Type) :
  let total_arrangements := 5 * 4 * 3 in
  let head_A_arrangements := (4 - 1) * 3 * 2 in
  total_arrangements - head_A_arrangements = 36 := 
by
  sorry

end arrangement_count_l670_670716


namespace starWarsEarned405_l670_670128

-- Definitions and Hypotheses
variables (cost_LionKing : ℕ) (earnings_LionKing : ℕ) (cost_StarWars : ℕ) (earnings_StarWars : ℕ)
variables (profit_LionKing half_profit_StarWars profit_StarWars : ℕ)

-- Conditions
def lionKingCost : cost_LionKing = 10 := rfl
def lionKingEarnings : earnings_LionKing = 200 := rfl
def starWarsCost : cost_StarWars = 25 := rfl
def lionKingProfit : profit_LionKing = earnings_LionKing - cost_LionKing := by rw [lionKingEarnings, lionKingCost]
def halfProfitCondition : profit_LionKing = profit_StarWars / 2 := by rw lionKingProfit
def starWarsEarnings : earnings_StarWars = cost_StarWars + profit_StarWars := by rw [starWarsCost]

-- Main theorem
theorem starWarsEarned405 : earnings_StarWars = 405 :=
by {
  rw [starWarsEarnings, lionKingProfit, halfProfitCondition, lionKingEarnings, starWarsCost],
  simp, 
  refine sorry,  -- Proof steps skipped
}

end starWarsEarned405_l670_670128


namespace max_modulus_of_complex_l670_670185

theorem max_modulus_of_complex (Z : ℂ) (h : ∥Z + (1 / Z)∥ = 1) : 
  ∥Z∥ ≤ (1 + Real.sqrt 5) / 2 := 
sorry

end max_modulus_of_complex_l670_670185


namespace other_discount_percentage_l670_670252

noncomputable def list_price : ℝ := 70
noncomputable def final_price : ℝ := 56.16
noncomputable def first_discount_percentage : ℝ := 10

theorem other_discount_percentage :
  ∃ D : ℝ, 63 - (D / 100) * 63 = 56.16 ∧ D ≈ 10.857 :=
begin
  sorry
end

end other_discount_percentage_l670_670252


namespace largest_invalid_number_of_friends_l670_670174

theorem largest_invalid_number_of_friends
  (N : ℕ)
  (h1 : 1 ≤ N ∧ N ≤ 150)
  (h2 : 96.8 * N / 100 ≤ N ∧ N ≤ 97.6 * N / 100) :
  N ≠ 125 :=
sorry

end largest_invalid_number_of_friends_l670_670174


namespace least_sum_of_exponents_l670_670368

theorem least_sum_of_exponents : 
  ∃ (exponents : List ℕ), List.sum exponents = 24 ∧ (∃ (powers : List ℕ), 
  List.pairwise (≠) powers ∧ List.sum (powers.map (λ x => 2^x)) = 640 ∧ 
  exponents = powers) :=
sorry

end least_sum_of_exponents_l670_670368


namespace serving_cost_is_50_cents_l670_670781

def cost_per_serving_in_cents (original_price : ℝ) (bulk_weight : ℝ) (coupon_value : ℝ) (serving_size : ℝ) : ℝ :=
  (original_price - coupon_value) / bulk_weight * serving_size * 100

theorem serving_cost_is_50_cents : cost_per_serving_in_cents 25 40 5 1 = 50 := by
  sorry

end serving_cost_is_50_cents_l670_670781


namespace sum_of_unions_eq_l670_670571

theorem sum_of_unions_eq:
  let S := Finset.range 2005 in
  let F := { A : Finset S → Finset S → Finset S → Finset S → Prop | A.Fst \subset S ∧ A.Snd \subset S ∧ A.Thd \subset S ∧ A.Fourth \subset S } in
  ∑ (A₁ A₂ A₃ A₄ ∈ F), (A₁ ∪ A₂ ∪ A₃ ∪ A₄).card = 2 ^ 8016 * 2005 * 15 :=
by
  sorry

end sum_of_unions_eq_l670_670571


namespace find_r_and_s_l670_670066

theorem find_r_and_s (r s : ℝ) :
  (∀ m : ℝ, ¬(∃ x : ℝ, x^2 + 10 * x = m * (x - 10) + 5) ↔ r < m ∧ m < s) →
  r + s = 60 :=
sorry

end find_r_and_s_l670_670066


namespace obtuse_angle_eq_120_l670_670179

-- Definitions for the points and the setup
universe u
noncomputable theory

variables (O A_1 A_2 A_3 A_4 A_5 A_6 X : Type)

-- Given conditions
def is_center (O : Type) : Prop := true
def on_circumference (O Ai : Type) : Prop := true
def angle (O Ai Aj : Type) : ℝ := 75 -- This is given angle between points A1 & A2

-- Define obtuse angle intersection to be proved
def obtuse_angle_at_intersection : ℝ := 120

-- Main theorem statement
theorem obtuse_angle_eq_120
  (hO : is_center O)
  (hA1 : on_circumference O A_1)
  (hA2 : on_circumference O A_2)
  (hA3 : on_circumference O A_3)
  (hA4 : on_circumference O A_4)
  (hA5 : on_circumference O A_5)
  (hA6 : on_circumference O A_6)
  (h_angle_A1O_A2 : angle O A_1 A_2 = 75)
  (h_angle_A1O_A6 : angle O A_1 A_6 = 15) -- derived in steps but assumed given for theorem
  (h_intersection : X = intersection_of_lines (line_through A_1 A_2) (line_through A_5 A_6)) :
  intersection_angle (side_of_triangle A_1 A_2) (side_of_triangle A_5 A_6) = obtuse_angle_at_intersection :=
sorry

end obtuse_angle_eq_120_l670_670179


namespace shirt_cost_l670_670203

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 66) : S = 12 :=
by
  sorry

end shirt_cost_l670_670203


namespace average_marks_of_class_l670_670973

theorem average_marks_of_class :
  (∀ (students total_students: ℕ) (marks95 marks0: ℕ) (avg_remaining: ℕ),
    total_students = 25 →
    students = 3 →
    marks95 = 95 →
    students = 5 →
    marks0 = 0 →
    (total_students - students - students) = 17 →
    avg_remaining = 45 →
    ((students * marks95 + students * marks0 + (total_students - students - students) * avg_remaining) / total_students) = 42)
:= sorry

end average_marks_of_class_l670_670973


namespace price_reduction_is_not_10_yuan_l670_670216

theorem price_reduction_is_not_10_yuan (current_price original_price : ℝ)
  (CurrentPrice : current_price = 45)
  (Reduction : current_price = 0.9 * original_price)
  (TenPercentReduction : 0.1 * original_price = 10) :
  (original_price - current_price) ≠ 10 := by
  sorry

end price_reduction_is_not_10_yuan_l670_670216


namespace find_a_l670_670762

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x + 1

theorem find_a
  (a : ℝ)
  (h_tangent : ∃ y : ℝ, (f a 2 = y) ∧ (f' a 1 = (f a 1 - y) / (1 - 2))) :
  a = 1 :=
by
  sorry

-- Helper function to define the derivative of f with respect to x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1

end find_a_l670_670762


namespace cyclic_quadrilateral_BCXY_l670_670389

noncomputable theory

variables {A B C F P D E X Y : Type}
variables [Geometry A B C F P D E X Y]

-- Given conditions 
variables (h1 : ∃ (ABC : Triangle), isAcuteAngleTriangle ABC)
variables (h2 : ∃ (F : Point), isFootOfAltitude F A)
variables (h3 : ∃ (P : Point), isOnSegment P (AF))
variables (h4 : linesThroughPParallel P A C B)
variables (h5 : ∃ (D E : Point), linesMeet BC D E)
variables (h6 : X ≠ A ∧ Y ≠ A)
variables (h7 : ∃ (circABD circACE : Circle), onCircumcicle X ABD ∧ onCircumcicle Y ACE)
variables (h8 : lineSegment D A = lineSegment D X)
variables (h9 : lineSegment E A = lineSegment E Y)

-- Main theorem to prove
theorem cyclic_quadrilateral_BCXY
  (ABC : Triangle) (F P D E X Y : Point)
  (h1 : isAcuteAngleTriangle ABC)
  (h2 : isFootOfAltitude F A)
  (h3 : isOnSegment P (AF))
  (h4 : linesThroughPParallel P A C B)
  (h5 : linesMeet BC D E)
  (h6 : X ≠ A ∧ Y ≠ A)
  (h7 : onCircumcicle X ABD ∧ onCircumcicle Y ACE)
  (h8 : lineSegment D A = lineSegment D X)
  (h9 : lineSegment E A = lineSegment E Y) :
  cyclicQuadrilateral B C X Y :=
by sorry

end cyclic_quadrilateral_BCXY_l670_670389


namespace tan_x_parallel_min_positive_period_fx_decreasing_intervals_l670_670448

-- Definitions of vectors a and b
def vec_a (x : ℝ) : ℝ × ℝ := (-Real.sin x, 1)
def vec_b (x : ℝ) : ℝ × ℝ := (Real.sin x - Real.cos x, 1)

-- Definition of dot product as the given function
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

-- (Ⅰ) If vectors are parallel, prove tan x = 1/2
theorem tan_x_parallel (x : ℝ) (h : ∃ k : ℝ, vec_a x = k • vec_b x) : Real.tan x = 1 / 2 := by
  sorry

-- (Ⅱ) Prove the minimum positive period of the function
theorem min_positive_period : ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = Real.pi := by
  sorry

-- (Ⅲ) Prove the intervals where the function is decreasing
theorem fx_decreasing_intervals (k : ℤ) : 
  ∀ x : ℝ, f' x < 0 → k * Real.pi + Real.pi / 8 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 8 := by
  sorry

end tan_x_parallel_min_positive_period_fx_decreasing_intervals_l670_670448


namespace efficient_paths_count_l670_670207

theorem efficient_paths_count (n : ℕ) (h : n = 2016) : 
  (choose (2 * n - 2) (n - 1)) = binomial (2 * n - 2) (n - 2) := by
  sorry

end efficient_paths_count_l670_670207


namespace students_sampled_from_second_grade_l670_670620

def arithmetic_sequence (a d : ℕ) : Prop :=
  3 * a - d = 1200

def stratified_sampling (total students second_grade : ℕ) : ℕ :=
  (second_grade * students) / total

theorem students_sampled_from_second_grade 
  (total students : ℕ)
  (h1 : total = 1200)
  (h2 : students = 48)
  (a d : ℕ)
  (h3 : arithmetic_sequence a d)
: stratified_sampling total students a = 16 :=
by
  rw [h1, h2]
  sorry

end students_sampled_from_second_grade_l670_670620


namespace ball_travel_distance_l670_670601

-- Define the initial drop and the rebounding fraction
def initial_drop : ℝ := 80
def rebound_fraction : ℝ := 1 / 3

-- Sum of distances for the sequence of drops and rebounds up to the fifth hit
noncomputable def total_distance : ℝ :=
  initial_drop + -- first descent
  (initial_drop * rebound_fraction) + -- first ascent
  (initial_drop * rebound_fraction) + -- second descent
  (initial_drop * rebound_fraction * rebound_fraction) + -- second ascent
  (initial_drop * rebound_fraction * rebound_fraction) + -- third descent
  (initial_drop * rebound_fraction * rebound_fraction * rebound_fraction) + -- third ascent
  (initial_drop * rebound_fraction * rebound_fraction * rebound_fraction) + -- fourth descent
  (initial_drop * rebound_fraction * rebound_fraction * rebound_fraction * rebound_fraction) + -- fourth ascent
  (initial_drop * rebound_fraction * rebound_fraction * rebound_fraction * rebound_fraction) -- fifth descent

theorem ball_travel_distance : total_distance = 158.02 := sorry

end ball_travel_distance_l670_670601


namespace find_petra_age_l670_670521

namespace MathProof
  -- Definitions of the given conditions
  variables (P M : ℕ)
  axiom sum_of_ages : P + M = 47
  axiom mother_age_relation : M = 2 * P + 14
  axiom mother_actual_age : M = 36

  -- The proof goal which we need to fill later
  theorem find_petra_age : P = 11 :=
  by
    -- Using the axioms we have
    sorry -- Proof steps, which you don't need to fill according to the instructions
end MathProof

end find_petra_age_l670_670521


namespace triangle_angle_eca_l670_670401

namespace Geometry

open Real

def isosceles_triangle (A B C : Type) [metric_space A] [add_group A] [has_adjoint B] :
  Prop :=
  dist A B = dist A C ∧ dist A B ≠ 0 ∧ dist B C ≠ 0 ∧
  angle B A C = angle C A B

variables {A B C D E : Type} [metric_space A] [add_group A]
[has_add B] [metric_space B] [has_adjoint C] [has_adjoint B]
{P Q R: A}

theorem triangle_angle_eca :
  isosceles_triangle P Q R -- Given triangle PQR is isosceles with PQ = PR
  ∧ angle Q P R = 40        -- PQR, ∠QPR = 40°
  ∧ D ∈ line_segment P R    -- Point D is on line segment PR
  ∧ angle_bisector Q R D    -- QR is the angle bisector of ∠QPR
  ∧ (dist_bisector Q R D E) -- QR extended to E, such that DE = AD
  → angle E P R = 40        -- Prove that angle EPR (equivalent to ∠ECA in original problem) is 40°
:= 
sorry

end Geometry

end triangle_angle_eca_l670_670401


namespace hyperbola_equation_l670_670024

theorem hyperbola_equation (h₁ : ∀ (x y : ℝ), x^2 + y^2 / 2 = 1 → (∃ a, y = a * sqrt 2)) 
                           (h₂ : ∀ e₁ e₂ : ℝ, e₁ * e₂ = 1) :
  ∃ a b : ℝ, a = sqrt 2 ∧ b = sqrt 2 ∧ (∀ x y, y^2 / 2 - x^2 / 2 = (1 : ℝ)) :=
by
  sorry

end hyperbola_equation_l670_670024


namespace trail_length_l670_670472

noncomputable def length_of_trail (T : ℝ) (u_d_ratio: ℝ) (S_u S_d : ℝ) (total_time : ℝ) : ℝ :=
(1 / (u_d_ratio / S_u + (1 - u_d_ratio) / S_d)) * total_time

theorem trail_length :
  let T := 130 / 60 in
  let s_u := 2 in
  let s_d := 3 in
  let ratio := 0.6 in
  length_of_trail T ratio s_u s_d (130 / 60) = 2.308 :=
by
  sorry

end trail_length_l670_670472


namespace prove_s90_zero_l670_670707

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a 0) + (n * (n - 1) * (a 1 - a 0)) / 2)

theorem prove_s90_zero (a : ℕ → ℕ) (h_arith : is_arithmetic_sequence a) (h : sum_of_first_n_terms a 30 = sum_of_first_n_terms a 60) :
  sum_of_first_n_terms a 90 = 0 :=
sorry

end prove_s90_zero_l670_670707


namespace product_gt_one_l670_670922

theorem product_gt_one 
  (m : ℚ) (b : ℚ)
  (hm : m = 3 / 4)
  (hb : b = 5 / 2) :
  m * b > 1 := 
by
  sorry

end product_gt_one_l670_670922


namespace perfect_squares_in_range_100_400_l670_670786

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l670_670786


namespace line_MN_fixed_point_l670_670322

noncomputable def center := (0 : ℝ, 0 : ℝ)
noncomputable def right_focus := (2 * Real.sqrt 5, 0 : ℝ)
noncomputable def eccentricity := Real.sqrt 5

-- Define the hyperbola parameters
noncomputable def a := 2
noncomputable def b := 4

-- Define vertices
noncomputable def A1 := (-2, 0 : ℝ)
noncomputable def A2 := (2, 0 : ℝ)

-- Define fixed line
def line_x_fixed := { p : ℝ × ℝ | p.1 = -1 }

-- Define point P
variable (P : ℝ × ℝ)
variable hP : P ∈ line_x_fixed

-- Define intersection points
variable (M N : ℝ × ℝ)

-- The relationship between P, A1, A2, M, and N cannot be encoded precisely
-- since this depends on the solution strategy. Omitting detailed coordinate
-- calculations for abstract fixed-point theorem.

theorem line_MN_fixed_point :
    ∀ (P : ℝ × ℝ) (hP : P.1 = -1) -- Point P moves on x = -1
    (M N : ℝ × ℝ) -- M and N are intersections with the hyperbola
    (hM : (4 * M.1^2 - M.2^2 = 16)) -- M is on the hyperbola
    (hN : (4 * N.1^2 - N.2^2 = 16)), -- N is on the hyperbola
    ∃ (fixed_point : ℝ × ℝ), -- There exists a fixed point
    let line_MN := λ x, ((N.2 - M.2) / (N.1 - M.1)) * (x - M.1) + M.2 in
    fixed_point = (-4, 0) ∧
    (∃ x, line_MN x = fixed_point.2) := 
begin
  sorry,
end

end line_MN_fixed_point_l670_670322


namespace separator_is_comma_l670_670566

-- Define the format of an input statement with multiple variables
def input_statement_format (x y : String) : String :=
  "input " ++ x ++ ", " ++ y

-- The main theorem
theorem separator_is_comma (x y : String) : 
  (input_statement_format x y).separator = ',' :=
sorry -- Proof placeholder

end separator_is_comma_l670_670566


namespace variance_transformed_variable_l670_670818

noncomputable def X (ω : Ω) : ℕ := sorry  -- Assume we have a random variable X

axiom X_is_binomial_10_0_6 : X ~ binomial 10 0.6

theorem variance_transformed_variable : variance (λ ω, 3 * X ω + 9) = 21.6 :=
by
  -- Definitions for the transformed variable and binomial variance
  have variance_X : variance X = 10 * 0.6 * (1 - 0.6) := sorry
  have linear_transformation_variance : ∀ (a b : ℝ), variance (λ ω, a * X ω + b) = a^2 * variance X := sorry
  -- Conclude the proof using the given facts
  sorry

end variance_transformed_variable_l670_670818


namespace solve_equation_l670_670485

theorem solve_equation :
  ∀ x : ℝ, x ≠ 1 → x ≠ 2 → (x + 1) / (x - 1) = 1 / (x - 2) + 1 → x = 3 := by
  sorry

end solve_equation_l670_670485


namespace find_x_plus_y_l670_670752

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2010) (h2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005) (h3 : 0 ≤ y ∧ y ≤ Real.pi) : 
  x + y = 2009 + Real.pi / 2 := 
sorry

end find_x_plus_y_l670_670752


namespace no_solution_for_f_eq_z_l670_670221

noncomputable def f (z : ℂ) : ℂ := 2 * complex.I * conj z

theorem no_solution_for_f_eq_z (z : ℂ) (h1 : complex.abs z = 10) :
  f z = z → False :=
by
  sorry

end no_solution_for_f_eq_z_l670_670221


namespace volume_of_red_tetrahedron_in_colored_cube_l670_670611

noncomputable def red_tetrahedron_volume (side_length : ℝ) : ℝ :=
  let cube_volume := side_length ^ 3
  let clear_tetrahedron_volume := (1/3) * (1/2 * side_length * side_length) * side_length
  let red_tetrahedron_volume := (cube_volume - 4 * clear_tetrahedron_volume)
  red_tetrahedron_volume

theorem volume_of_red_tetrahedron_in_colored_cube 
: red_tetrahedron_volume 8 = 512 / 3 := by
  sorry

end volume_of_red_tetrahedron_in_colored_cube_l670_670611


namespace hexagon_overlapping_area_l670_670628

theorem hexagon_overlapping_area:
  ∀ (s α : ℝ),
  s = 1 →
  0 < α ∧ α < π / 3 →
  real.cos α = real.sqrt 3 / 2 →
  let area := (3 * real.sqrt 3) / 2
  in 
  (s ^ 2 * area / 3) = real.sqrt 3 / 2 :=
by 
  intros s α h_s h_α h_cos g 
  let area := (3 * real.sqrt 3) / 2
  sorry

end hexagon_overlapping_area_l670_670628


namespace percentage_increase_l670_670020

-- Define the initial and final prices as constants
def P_inicial : ℝ := 5.00
def P_final : ℝ := 5.55

-- Define the percentage increase proof
theorem percentage_increase : ((P_final - P_inicial) / P_inicial) * 100 = 11 := 
by
  sorry

end percentage_increase_l670_670020


namespace quadrilateral_shape_and_common_points_l670_670079

-- Definitions
def isOnEllipse (x y : ℝ) : Prop := (x^2 / 8) + (y^2 / 2) = 1
def ellipseC1 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1
def line (k m x y : ℝ) : Prop := y = k * x + m
def bisectsChord (x0 y0 k m x1 y1 x2 y2 : ℝ) : Prop :=
  (x0 = (x1 + x2)/2) ∧ (y0 = (y1 + y2)/2)
def areaTriangle (pX pY mX mY nX nY : ℝ) : Prop :=
  0.5 * abs ((pX * (mY - nY) + mX * (nY - pY) + nX * (pY - mY)) / 2) = 1

-- The theorem with given conditions

theorem quadrilateral_shape_and_common_points
  (x0 y0 k m xM yM xN yN : ℝ)
  (hEllipse : isOnEllipse x0 y0)
  (hIntersects : ellipseC1 xM yM ∧ ellipseC1 xN yN ∧ line k m xM yM ∧ line k m xN yN)
  (hBisect : bisectsChord x0 y0 k m xM yM xN yN)
  (hIneq : m * y0 > -1)
  (hArea : areaTriangle x0 y0 xM yM xN yN) :
  (∃ x1 y1 x2 y2 : ℝ, ellipseC1 x1 y1 ∧ ellipseC1 x2 y2 ∧ x1 ≠ x2 ∧ line k m x1 y1 ∧ line k m x2 y2) → 
  -- Proving the quadrilateral is a parallelogram
  (∀ xM yM xN yN : ℝ, ellipseC1 xM yM ∧ ellipseC1 xN yN → 
    P(x0, y0), M, N, and O form a parallelogram with \( PM \equiv OM \) and  \( ON \equiv NP \))
  -- Proving the number of common points between line PM and the ellipse C1 is 1
  ∀ M P : Prop, ¬(PM ∧ (ellipseC1 ∧ line k m P)).
  sorry

end quadrilateral_shape_and_common_points_l670_670079


namespace normal_phd_time_l670_670536

-- Define conditions
def years_to_finish_BS : ℕ := 3
def combined_program_time : ℕ := 6
def fraction_of_normal_time : ℚ := 3/4

-- Define the problem statement
theorem normal_phd_time (P : ℕ) : (P + years_to_finish_BS) / fraction_of_normal_time.to_nat = combined_program_time → P = 5 :=
sorry

end normal_phd_time_l670_670536


namespace ball_arrangement_count_l670_670040

theorem ball_arrangement_count :
  ∃ n : Nat, n = 216 
  ∧ valid_ball_config_count n := sorry

end ball_arrangement_count_l670_670040


namespace obtuse_angle_condition_l670_670342

noncomputable def vector_a : ℝ × ℝ := (1, -2)
noncomputable def vector_b (λ : ℝ) : ℝ × ℝ := (1, λ)

theorem obtuse_angle_condition (λ : ℝ) : 
  let a := vector_a in
  let b := vector_b λ in
  a.1 * b.1 + a.2 * b.2 < 0 → λ > 1/2 :=
by
  intro h
  sorry

end obtuse_angle_condition_l670_670342


namespace music_marks_l670_670122

variable (M : ℕ) -- Variable to represent marks in music

/-- Conditions -/
def science_marks : ℕ := 70
def social_studies_marks : ℕ := 85
def total_marks : ℕ := 275
def physics_marks : ℕ := M / 2

theorem music_marks :
  science_marks + M + social_studies_marks + physics_marks M = total_marks → M = 80 :=
by
  sorry

end music_marks_l670_670122


namespace additional_hours_to_travel_l670_670247

theorem additional_hours_to_travel (distance1 time1 rate distance2 : ℝ)
  (H1 : distance1 = 360)
  (H2 : time1 = 3)
  (H3 : rate = distance1 / time1)
  (H4 : distance2 = 240)
  :
  distance2 / rate = 2 := 
sorry

end additional_hours_to_travel_l670_670247


namespace remainder_abc_mod_7_l670_670816

variable (a b c : ℕ)
hypothesis (h1 : a < 7)
hypothesis (h2 : b < 7)
hypothesis (h3 : c < 7)
hypothesis (h4 : (a + 3 * b + 2 * c) % 7 = 3)
hypothesis (h5 : (2 * a + b + 3 * c) % 7 = 2)
hypothesis (h6 : (3 * a + 2 * b + c) % 7 = 1)

theorem remainder_abc_mod_7 : (a * b * c) % 7 = 4 :=
sorry

end remainder_abc_mod_7_l670_670816


namespace point_coordinates_l670_670388

noncomputable def coordinates_of_point (x y : ℝ) : Prop :=
  P(x, y) ∧ x < 0 ∧ y < 0 ∧ abs y = 8 ∧ abs x = 5

theorem point_coordinates :
  ∃ (x y : ℝ), coordinates_of_point x y ∧ (x, y) = (-5, -8) :=
by
  sorry

end point_coordinates_l670_670388


namespace cardinality_of_intersection_l670_670017

theorem cardinality_of_intersection :
  let A := {x : ℝ | |2 * x - 3| ≤ 1}
  let B := {x : ℕ | (-2 : ℝ) < x ∧ (x : ℝ) < 5}
  (A ∩ B).card = 2 :=
by
  -- Define the sets A and B
  let A := {x : ℝ | |2 * x - 3| ≤ 1}
  let B := {x : ℕ | (-2 : ℝ) < x ∧ (x : ℝ) < 5}
  -- Show that A ∩ B has 2 elements
  have h1 : A ∩ B = {1, 2} := sorry
  have h2 : (A ∩ B).card = 2 := sorry
  exact h2

end cardinality_of_intersection_l670_670017


namespace eval_double_sum_l670_670680

theorem eval_double_sum : 
  (∑ i in Finset.range 100, ∑ j in Finset.range 50, (i + 1) + (j + 1)) = 380000 :=
by
  sorry

end eval_double_sum_l670_670680


namespace correlation_is_strong_linear_l670_670729

theorem correlation_is_strong_linear (r : ℝ) (hr : r = -0.990) : |r| ≈ 1 :=
by
  sorry

end correlation_is_strong_linear_l670_670729


namespace bisection_method_third_interval_l670_670960

noncomputable def bisection_method_interval (f : ℝ → ℝ) (a b : ℝ) (n : ℕ) : (ℝ × ℝ) :=
  sorry  -- Definition of the interval using bisection method, but this is not necessary.

theorem bisection_method_third_interval (f : ℝ → ℝ) :
  (bisection_method_interval f (-2) 4 3) = (-1/2, 1) :=
sorry

end bisection_method_third_interval_l670_670960


namespace solve_circle_problem_l670_670656

noncomputable def circle_problem : Prop :=
  ∃ (r1 r2 : ℝ) (m : ℝ),
    (9, 6) ∈ (circle_intersection_points r1 r2 ∧
    r1 * r2 = 68 ∧
    tangent_to_x_axis_and_line r1 r2 m ∧
    (∃ (a b c : ℕ), m = (↑a * ↑√b) / ↑c ∧ a + b + c = 6))

axiom circle_intersection_points : ℝ → ℝ → set (ℝ × ℝ)
axiom tangent_to_x_axis_and_line : ℝ → ℝ → ℝ → Prop

theorem solve_circle_problem : circle_problem := sorry

end solve_circle_problem_l670_670656


namespace ratio_of_discretionary_income_l670_670676

theorem ratio_of_discretionary_income
  (net_monthly_salary : ℝ) 
  (vacation_fund_pct : ℝ) 
  (savings_pct : ℝ) 
  (socializing_pct : ℝ) 
  (gifts_amt : ℝ)
  (D : ℝ) 
  (ratio : ℝ)
  (salary : net_monthly_salary = 3700)
  (vacation_fund : vacation_fund_pct = 0.30)
  (savings : savings_pct = 0.20)
  (socializing : socializing_pct = 0.35)
  (gifts : gifts_amt = 111)
  (discretionary_income : D = gifts_amt / 0.15)
  (net_salary_ratio : ratio = D / net_monthly_salary) :
  ratio = 1 / 5 := sorry

end ratio_of_discretionary_income_l670_670676


namespace perfect_squares_in_range_100_400_l670_670783

theorem perfect_squares_in_range_100_400 : ∃ n : ℕ, (∀ m, 100 ≤ m^2 → m^2 ≤ 400 → m^2 = (m - 10 + 1)^2) ∧ n = 9 := 
by
  sorry

end perfect_squares_in_range_100_400_l670_670783


namespace gcd_common_prime_l670_670709

def gcd (a b : ℕ) : ℕ := Nat.gcd a b

theorem gcd_common_prime (m n : ℕ) (hm : m > 0) (hn : n > 0)
  (H : ∀ (x y : ℕ), (x ∣ m) → (y ∣ n) → gcd (x + y) (m * n) > 1) : gcd m n > 1 :=
sorry

end gcd_common_prime_l670_670709


namespace joggers_meet_again_at_correct_time_l670_670648

-- Define the joggers and their lap times
def bob_lap_time := 3
def carol_lap_time := 5
def ted_lap_time := 8

-- Calculate the Least Common Multiple (LCM) of their lap times
def lcm_joggers := Nat.lcm (Nat.lcm bob_lap_time carol_lap_time) ted_lap_time

-- Start time is 9:00 AM
def start_time := 9 * 60  -- in minutes

-- The time (in minutes) we get back together is start_time plus the LCM
def earliest_meeting_time := start_time + lcm_joggers

-- Convert the meeting time to hours and minutes
def hours := earliest_meeting_time / 60
def minutes := earliest_meeting_time % 60

-- Define an expected result
def expected_meeting_hour := 11
def expected_meeting_minute := 0

-- Prove that all joggers will meet again at the correct time
theorem joggers_meet_again_at_correct_time :
  hours = expected_meeting_hour ∧ minutes = expected_meeting_minute :=
by
  -- Here you would provide the proof, but we'll use sorry for brevity
  sorry

end joggers_meet_again_at_correct_time_l670_670648


namespace find_c_solve_inequality_l670_670758

def f (x : ℝ) (c : ℝ) : ℝ :=
if 0 < x ∧ x < c then c * x + 1 else
if c ≤ x ∧ x < 1 then 2^(-x / c^2) + 1 else 0

theorem find_c : ∃ c : ℝ, 0 < c ∧ c < 1 ∧ f (c^2) c = 9/8 := by
  sorry

theorem solve_inequality : 
  ∃ A B : Set ℝ, A = { x : ℝ | sqrt 2 / 4 < x ∧ x < 1/2 } ∧ B = { x : ℝ | 1/2 ≤ x ∧ x < 5/8 }
  ∧ ∀ x : ℝ, f x (1/2) > sqrt 2 / 8 + 1 ↔ x ∈ A ∪ B := by
  sorry

end find_c_solve_inequality_l670_670758


namespace calculate_NY_l670_670110

-- Definitions for the problem setup
structure Point :=
(x : ℝ)
(y : ℝ)

def Length (p1 p2 : Point) : ℝ :=
  real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Given conditions
variable (L M N X Y Z : Point)

axiom LZ_length : Length L Z = 4
axiom ZM_NX_length : Length Z M = 3 ∧ Length N X = 3
axiom right_triangle_XYZ : ∃ θ, θ = 90 ∧ Length X Y = real.sqrt (Length Z X^2 - Length Z Y^2)

-- The hypothesis of hypotenuse being 7
axiom hypotenuse_XZ_length : Length X Z = 7

-- The proof goal
theorem calculate_NY : Length N Y = 3 := by
  -- Proof goes here
  sorry

end calculate_NY_l670_670110


namespace perfect_squares_between_100_and_400_l670_670791

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l670_670791


namespace compare_abc_l670_670718

noncomputable def a : ℝ := real.exp (real.log 4 / 3)
noncomputable def b : ℝ := real.log 3 / real.log (1 / 4)
noncomputable def c : ℝ := real.log (1 / 4) / real.log (1 / 3)

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l670_670718


namespace henry_income_percent_increase_l670_670202

theorem henry_income_percent_increase :
  let original_income : ℝ := 120
  let new_income : ℝ := 180
  let increase := new_income - original_income
  let percent_increase := (increase / original_income) * 100
  percent_increase = 50 :=
by
  sorry

end henry_income_percent_increase_l670_670202


namespace problem_abc_l670_670099

theorem problem_abc :
  let n := 2024
  let probability := (∑ (a b c : ℕ) in finset.range(n + 1),
  if (a > 0 ∧ b > 0 ∧ c > 0 ∧ (a * b * c + a * b + a) % 4 = 0) then 1 else 0 : ℚ) / n ^ 3 
  
  probability = 25 / 64 := 
sorry

end problem_abc_l670_670099


namespace div_by_66_l670_670100

theorem div_by_66 :
  (43 ^ 23 + 23 ^ 43) % 66 = 0 := 
sorry

end div_by_66_l670_670100


namespace fraction_passengers_from_asia_l670_670367

theorem fraction_passengers_from_asia (P : ℕ)
  (hP : P = 108)
  (frac_NA : ℚ) (frac_EU : ℚ) (frac_AF : ℚ)
  (Other_continents : ℕ)
  (h_frac_NA : frac_NA = 1/12)
  (h_frac_EU : frac_EU = 1/4)
  (h_frac_AF : frac_AF = 1/9)
  (h_Other_continents : Other_continents = 42) :
  (P * (1 - (frac_NA + frac_EU + frac_AF)) - Other_continents) / P = 1/6 :=
by
  sorry

end fraction_passengers_from_asia_l670_670367


namespace log_identity_l670_670593

theorem log_identity : (1/2) * log 6 12 - log 6 (sqrt 2) = 1/2 :=
by
  sorry

end log_identity_l670_670593


namespace sequence_correct_l670_670400

noncomputable def sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else
  if n = 1 then 1 else
  (1 / 3) * ((4 / 3) ^ (n - 2))

theorem sequence_correct (n : ℕ) : 
  sequence n = 
  if n = 1 then 1 else (1 / 3) * ((4 / 3) ^ (n - 2)) := 
sorry

end sequence_correct_l670_670400


namespace incorrect_proposition_D_main_theorem_l670_670196

open Classical

theorem incorrect_proposition_D :
  (¬(p ∧ q) ↔ ¬p ∨ ¬q) → ¬ (¬(p ∧ q) → ¬p ∧ ¬q) :=
 by {
  intro h,
  exact not_and_distrib.mp h
}

variable (x : ℝ)

def proposition_A : Prop :=
  (x^2 - 3 * x + 2 = 0 → x = 1) ↔ (x ≠ 1 → x^2 - 3 * x + 2 ≠ 0)

def proposition_B : Prop :=
  x = 1 → x^2 - 3 * x + 2 = 0

def proposition_C : Prop :=
  (∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0)

def proposition_D (p q : Prop) : Prop :=
  ((¬(p ∧ q)) → (¬p ∧ ¬q)) = false

theorem main_theorem (p q: Prop) :
  proposition_D p q :=
begin
  exact incorrect_proposition_D sorry
end

end incorrect_proposition_D_main_theorem_l670_670196


namespace systematic_sampling_interval_l670_670041
-- Import the required library

-- Define the population size and sample size
def N : ℕ := 1200
def n : ℕ := 30

-- Define the interval calculation and the proof goal
theorem systematic_sampling_interval :
  let k := N / n in
  k = 40 :=
by
  have k_def : k = 1200 / 30 := rfl
  exact k_def

end systematic_sampling_interval_l670_670041


namespace find_expression_domain_f_exists_odd_l670_670586

noncomputable def f (a x : ℝ) : ℝ := log ((x + 2 * a + 1) / (x - 3 * a + 1))

-- Problem 1: Find the expression for f(x)
theorem find_expression (a x : ℝ) (h : a ≠ 0) :
    (f a x) = log ((x + (2 * a + 1)) / (x + (-3 * a + 1))) :=
sorry

-- Problem 2: Find the domain of f(x)
theorem domain_f (a : ℝ) (h : a ≠ 0) :
    (a > 0 → set_of (λ x, (-∞ < x ∧ x < -2 * a - 1) ∨ (3 * a - 1 < x ∧ x < ∞))) ∧
    (a < 0 → set_of (λ x, (-∞ < x ∧ x < 3 * a - 1) ∨ (-2 * a - 1 < x ∧ x < ∞))) :=
sorry

-- Problem 3: Determine if there exists an a such that f(x) is odd or even (in this case, odd)
theorem exists_odd (a : ℝ) (h : a ≠ 0) :
    (∃ a : ℝ, a = 0.2 ∧ ∀ x : ℝ, f a (-x) = -f a x) :=
sorry

end find_expression_domain_f_exists_odd_l670_670586


namespace dot_product_correct_l670_670650

def vector1 : ℝ × ℝ × ℝ := (2, -3, 0)
def vector2 : ℝ × ℝ × ℝ := (-4, 1, 5)

theorem dot_product_correct : 
  let dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 
  in dot_product vector1 vector2 = -11 :=
by {
  let dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3,
  show dot_product vector1 vector2 = -11,
  sorry
}

end dot_product_correct_l670_670650


namespace volume_of_red_tetrahedron_l670_670613

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length^3

def volume_of_tetrahedron (base_area : ℕ) (height : ℕ) : ℚ :=
  (1/3 : ℚ) * base_area * height

def smaller_tetrahedron_volume (side_length : ℕ) : ℚ :=
  volume_of_tetrahedron ((1/2 : ℚ) * side_length^2) side_length

theorem volume_of_red_tetrahedron :
  let cube_side_length := 8 in
  let cube_volume := volume_of_cube cube_side_length in
  let smaller_tetrahedrons_volume := 4 * smaller_tetrahedron_volume cube_side_length in
  cube_volume - smaller_tetrahedrons_volume = 512 / 3 :=
by
  sorry

end volume_of_red_tetrahedron_l670_670613


namespace zero_not_in_range_of_g_l670_670868

noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then ⌈1 / (x + 3)⌉
  else ⌊1 / (x + 3)⌋

theorem zero_not_in_range_of_g : ∀ x : ℝ, x ≠ -3 → g x ≠ 0 := by
  sorry

end zero_not_in_range_of_g_l670_670868


namespace wario_missed_field_goals_wide_right_l670_670543

theorem wario_missed_field_goals_wide_right :
  ∀ (attempts missed_fraction wide_right_fraction : ℕ), 
  attempts = 60 →
  missed_fraction = 1 / 4 →
  wide_right_fraction = 20 / 100 →
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  wide_right = 3 :=
by
  intros attempts missed_fraction wide_right_fraction h1 h2 h3
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  sorry

end wario_missed_field_goals_wide_right_l670_670543


namespace find_a_l670_670435

def has_three_distinct_solutions (f : Real → Real) [∀ x, Decidable (f x = 0)] : Prop :=
  ∃ (x1 x2 x3 : Real), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

theorem find_a (a : Real) :
  (has_three_distinct_solutions (λ x, abs (abs (x - a) - a) - 2)) ↔ a = 2 :=
by
  sorry

end find_a_l670_670435


namespace one_has_property_P_two_has_property_P_three_does_not_have_property_P_count_non_property_P_from_1_to_2014_l670_670874

def has_property_P (n : ℤ) : Prop :=
  ∃ x y z : ℤ, n = x^3 + y^3 + z^3 - 3 * x * y * z

theorem one_has_property_P : has_property_P 1 :=
sorry

theorem two_has_property_P : has_property_P 2 :=
sorry

theorem three_does_not_have_property_P : ¬ has_property_P 3 :=
sorry

theorem count_non_property_P_from_1_to_2014 : 
  ∃ count : ℕ, count=448 ∧ count = (∑ n in (Finset.range 2014).filter (λ n, ¬ has_property_P (n + 1)), 1) :=
sorry

end one_has_property_P_two_has_property_P_three_does_not_have_property_P_count_non_property_P_from_1_to_2014_l670_670874


namespace z_value_binomial_coeff_l670_670595

theorem z_value (b : ℝ) (h_pos : 0 < b) (h_pure_imaginary : ∀ (z : ℂ), (z - 2)^2 = (1 - b^2, - 2 * b) → z = 3 + Complex.i) : z = 3 + Complex.i :=
by
sorry

theorem binomial_coeff (n : ℕ) (h_sum_eq_16 : 2 ^ n = 16) : 
  (nat.choose 4 2 * 3^2 = 54) :=
by
sorry

end z_value_binomial_coeff_l670_670595


namespace area_problem_l670_670317

def side_of_square_bdef (ab bc : ℝ) (h_ab : ab = 2) (h_bc : bc = 2) : ℝ :=
let ac := ab * Real.sqrt 2 in
4 + 2 * Real.sqrt 2

def area_of_square_bdef (s : ℝ) : ℝ :=
s ^ 2

def area_of_surrounding_rectangle (s : ℝ) : ℝ :=
2 * s ^ 2

def area_of_regular_octagon (s : ℝ) : ℝ :=
let a := 2 * Real.sqrt 2 in
2 * (1 + Real.sqrt 2) * a ^ 2

def total_area_composed (s area_square area_rectangle area_octagon : ℝ) : ℝ :=
area_square + area_rectangle - area_octagon

theorem area_problem (ab bc : ℝ) (h_ab : ab = 2) (h_bc : bc = 2) :
  let s := side_of_square_bdef ab bc h_ab h_bc in
  let area_square := area_of_square_bdef s in
  let area_rectangle := area_of_surrounding_rectangle s in
  let area_octagon := area_of_regular_octagon s in
  total_area_composed s area_square area_rectangle area_octagon = 56 + 24 * Real.sqrt 2 :=
by { sorry }

end area_problem_l670_670317


namespace ninetieth_number_value_l670_670658

theorem ninetieth_number_value :
  (∃ (row_number : ℕ) (total_elements : ℕ),
    (row_number % 2 = 1 → total_elements = row_number + 3) ∧
    (row_number % 2 = 0 → total_elements = row_number * 2) ∧
    ∑ i in range(row_number - 1), (if i % 2 = 1 then i + 3 else i * 2) < 90 ∧
    ∑ i in range(row_number) (if i % 2 = 1 then i + 3 else i * 2) ≥ 90 ∧
    total_elements * (row_number div 2 + 1) = 20) :=
sorry

end ninetieth_number_value_l670_670658


namespace range_of_m_l670_670777

/-- Prove the range of values for m satisfying given set conditions -/
theorem range_of_m (m : ℝ) (A B : set ℝ) 
  (hA : ∀ x, x ∈ A ↔ -1 ≤ x ∧ x ≤ 6) 
  (hB : ∀ x, x ∈ B ↔ m - 1 ≤ x ∧ x ≤ 2 * m + 1) 
  (hAB : ∀ x, x ∈ B → x ∈ A) :
  (m ∈ set.Ioo (⊥ : ℝ) (-2) ∨ m ∈ set.Icc (0 : ℝ) (5 / 2)) := 
sorry

end range_of_m_l670_670777


namespace jake_dial_correct_probability_l670_670846

theorem jake_dial_correct_probability :
  let first_three_digits := {407, 410, 413}
  let last_five_digits := {0, 2, 4, 5, 8}
  let total_phone_numbers := 3 * (last_five_digits.cardperm)
  in (1 / total_phone_numbers) = (1 / 360) := by
  sorry

end jake_dial_correct_probability_l670_670846


namespace rationalize_denominator_sum_l670_670108

theorem rationalize_denominator_sum :
  let A := -4
  let B := 7
  let C := 3
  let D := 13
  let E := 1
  A + B + C + D + E = 20 := by
    sorry

end rationalize_denominator_sum_l670_670108


namespace trains_opposite_directions_not_equal_l670_670170

variables {α : Type} [AddCommGroup α] [Module ℝ α]

def are_opposite_directions (a b : α) : Prop :=
  ∃ (k : ℝ), k < 0 ∧ a = k • b

theorem trains_opposite_directions_not_equal
  {a b : α}
  (h_displacement_same : |a| = |b|)
  (h_opposite_directions : are_opposite_directions a b) :
  a ≠ b :=
by
  sorry

end trains_opposite_directions_not_equal_l670_670170


namespace random_points_probability_l670_670853

noncomputable def probability_distance_condition_met (r : ℝ) (a b c : ℕ) [NeZero c] : Prop :=
  let p := Real.arcsin (2 / 3)
  1 - p / Real.pi = (a - b * Real.pi) / c

theorem random_points_probability :
  ∃ (a b c : ℕ), gcd a b c = 1 ∧ a + b + c = 21 ∧ probability_distance_condition_met 1 a b c :=
begin
  sorry
end

end random_points_probability_l670_670853


namespace part_a_part_b_part_c_l670_670939

namespace EulerianBridgeProblem

-- Graph definition with islands and bridges
noncomputable def Graph := Type

-- Vertex definition as islands
variable (Island : Type)

-- Context of Eulerian path or walk
structure EulerianPath (G : Graph) :=
(vertices : Finset Island)
(edges : Finset (Island × Island))
(adj : ∀ {u v : Island}, (u, v) ∈ edges ∨ (v, u) ∈ edges)
(degree : Island → Nat)
(path : List Island)
(is_eulerian : ∀ e ∈ edges, ∃! i, (path.nth i = some (e.1) ∧ path.nth (i+1) = some (e.2))
 ∨ (path.nth i = some (e.2) ∧ path.nth (i+1) = some (e.1)))

variable {G : Graph}
variable {T : Island}

-- Part (a): How many bridges lead from Troekratny if the tourist did not start and did not finish at this island?
theorem part_a (h : EulerianPath G) (start_ne_T : h.path.head ≠ some T) (end_ne_T : h.path.last ≠ some T) :
  h.degree T = 6 := sorry

-- Part (b): How many bridges lead from Troekratny if the tourist started but did not finish at this island?
theorem part_b (h : EulerianPath G) (start_T : h.path.head = some T) (end_ne_T : h.path.last ≠ some T) :
  h.degree T = 5 := sorry

-- Part (c): How many bridges lead from Troekratny if the tourist started and finished at this island?
theorem part_c (h : EulerianPath G) (start_T : h.path.head = some T) (end_T : h.path.last = some T) :
  h.degree T = 4 := sorry

end EulerianBridgeProblem

end part_a_part_b_part_c_l670_670939


namespace perfect_squares_between_100_and_400_l670_670802

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def count_perfect_squares_between (a b : ℕ) : ℕ :=
  (finset.Ico a b).filter is_perfect_square .card

theorem perfect_squares_between_100_and_400 : count_perfect_squares_between 101 400 = 9 :=
by
  -- The space for the proof is intentionally left as a placeholder
  sorry

end perfect_squares_between_100_and_400_l670_670802


namespace perpendicular_lines_intersect_at_single_point_l670_670103

/-- Prove that the perpendicular lines drawn through the midpoints of the sides of 
    an inscribed quadrilateral intersect at a single point. -/
theorem perpendicular_lines_intersect_at_single_point
  {A B C D P Q R S : Point}
  (h_inscribed : is_inscribed A B C D)
  (h_midpoints : midpoint P A B ∧ midpoint Q B C ∧ midpoint R C D ∧ midpoint S D A) :
  ∃! O', ∀ Q : Line, is_opposite_side(P, A, B, D) ∧ is_opposite_side(Q, B, C, A) ∧
                    is_opposite_side(R, C, D, B) ∧ is_opposite_side(S, D, A, C) → 
                    intersects_at_single_point(Q, O') :=
begin
  sorry
end

end perpendicular_lines_intersect_at_single_point_l670_670103


namespace tokyo_flood_damage_l670_670043

def damage_in_usd (damage_in_yen : ℕ) (exchange_rate_yen_per_usd : ℕ) : ℝ :=
  damage_in_yen / exchange_rate_yen_per_usd 

theorem tokyo_flood_damage :
  damage_in_usd 2000000000 110 = 18181818 := by
  sorry

end tokyo_flood_damage_l670_670043


namespace maximize_power_speed_l670_670588

-- Define the necessary parameters and constants
variables (C S ρ v0 v : ℝ)
constants (hS : S = 5) (hv0 : v0 = 6)

-- Define the force equation
def force (v : ℝ) : ℝ := (C * S * ρ * (v0 - v)^2) / 2

-- Define the power equation
def power (v : ℝ) : ℝ := force v * v

-- Statement to prove that the speed v that maximizes the power is v0 / 3
theorem maximize_power_speed :
  (v = v0 / 3) → (v0 = 6) → (S = 5) → (v_0 = 6) := by
sory

end maximize_power_speed_l670_670588


namespace exists_negative_root_iff_p_geq_neg_three_eighths_l670_670290

theorem exists_negative_root_iff_p_geq_neg_three_eighths (p : ℝ) :
    (∃ x : ℝ, x < 0 ∧ x^4 - 4*p*x^3 + x^2 - 4*p*x + 1 = 0) ↔ p ∈ set.Ici (-3/8) :=
begin
    sorry
end

end exists_negative_root_iff_p_geq_neg_three_eighths_l670_670290


namespace range_of_a_l670_670733

section
variables (a : ℝ)
def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a = 1 ∨ a ≤ -2 :=
sorry
end

end range_of_a_l670_670733


namespace perpendicular_line_l670_670374

theorem perpendicular_line (A : ℝ × ℝ) (l1 : ℝ → ℝ) (l2 : ℝ → ℝ) :
  A = (-1, 3) → 
  (∀ x, l1 x = (1/2) * x - (3/2)) → 
  (∀ x, l2 x = -2 * x + 1) → 
  l2 (-1) = 3 :=
by
s -- sorry-- Add sorry to mark the proof as missing.

end perpendicular_line_l670_670374


namespace line_equation_l670_670622

theorem line_equation :
  ∀ (x y : ℝ),
    (fin 2 -> ℝ) (λ i, if i = 0 then 3 else -4) 
    ⬝ (λ i, if i = 0 then x + 2 else y - 8) = 0 →
    y = (3 / 4) * x + 9.5 :=
by
  intros x y h
  have : 3 * (x + 2) - 4 * (y - 8) = 0, from h
  sorry

end line_equation_l670_670622


namespace valid_selling_price_l670_670034

-- Define the initial conditions
def cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def sales_increase_per_dollar_decrease : ℝ := 4
def max_profit : ℝ := 13600
def min_selling_price : ℝ := 150

-- Define x as the price reduction per item
variable (x : ℝ)

-- Define the function relationship of the daily sales volume y with respect to x
def sales_volume (x : ℝ) := 100 + 4 * x

-- Define the selling price based on the price reduction
def selling_price (x : ℝ) := 200 - x

-- Calculate the profit based on the selling price and sales volume
def profit (x : ℝ) := (selling_price x - cost_price) * (sales_volume x)

-- Lean theorem statement to prove the given conditions lead to the valid selling price
theorem valid_selling_price (x : ℝ) 
  (h1 : profit x = 13600)
  (h2 : selling_price x ≥ 150) : 
  selling_price x = 185 :=
sorry

end valid_selling_price_l670_670034


namespace swim_time_against_current_l670_670232

theorem swim_time_against_current :
  let swim_speed := 4 -- km/h
  let water_speed := 2 -- km/h
  let distance := 6 * 4 -- km
  let effective_speed_against := swim_speed - water_speed
  time := distance / effective_speed_against
  in time = 12 := 
by 
  -- proof goes here, but we are skipping with sorry
  sorry

end swim_time_against_current_l670_670232


namespace remainder_2023_div_73_l670_670552

theorem remainder_2023_div_73 : 2023 % 73 = 52 := 
by
  -- Proof goes here
  sorry

end remainder_2023_div_73_l670_670552


namespace part1_part2_l670_670387

noncomputable def circle_param_eq := (α : ℝ) → (x : ℝ) × (y : ℝ)
| α => (5 * Real.cos α, -6 + 5 * Real.sin α)

noncomputable def polar_eq_circle_c := (ρ θ : ℝ) : Prop :=
ρ^2 + 12 * ρ * Real.sin θ + 11 = 0

def tan_defined_alpha_0 := (t : ℝ) : Prop := t = Real.sqrt 5 / 2

theorem part1 (α : ℝ) : ∃ ρ θ, circle_param_eq α = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ polar_eq_circle_c ρ θ :=
begin
  -- proof of polar equation conversion
  sorry
end

theorem part2 (α_0 : ℝ) (hα_0 : tan_defined_alpha_0 α_0) : ∃ A B : ℝ, 
  ∃ ρ1 ρ2, polar_eq_circle_c ρ1 α_0 ∧ polar_eq_circle_c ρ2 α_0 ∧ 
  (A, B) = (ρ1, ρ2) ∧ (|A - B| = 6) :=
begin
  -- proof of intersection length calculation
  sorry
end

end part1_part2_l670_670387


namespace students_playing_both_l670_670578

theorem students_playing_both : 
  let total_students := 450
  let football_players := 325
  let cricket_players := 175
  let neither_players := 50
  let total_playing_either := total_students - neither_players
  let both_players := football_players + cricket_players - total_playing_either
  in both_players = 100 :=
by
  sorry

end students_playing_both_l670_670578


namespace rectangle_area_l670_670982

theorem rectangle_area (ABCD : ℝ) (E F : ℝ) (h1 : BE < CF)
  (h2 : ∠AB'C' = ∠B'EA)
  (h3 : AB' = 7)
  (h4 : BE = 25) :
  ABCD = 686 + 175 * Real.sqrt 7 :=
by
  sorry

end rectangle_area_l670_670982


namespace shift_function_right_down_l670_670499

theorem shift_function_right_down (x : ℝ) :
    (∀ y : ℝ, y = 1 / x → y = 1 / (x - 2) - 1) := 
begin
  intros y h,
  rw h,
  sorry
end

end shift_function_right_down_l670_670499


namespace min_product_of_distances_l670_670623

noncomputable def parabola := { p : ℝ × ℝ | p.2^2 = 4 * p.1 }
def focus : ℝ × ℝ := (1, 0)
def origin : ℝ × ℝ := (0, 0)

theorem min_product_of_distances (A B : ℝ × ℝ) (hA : A ∈ parabola) (hB : B ∈ parabola)
  (hline : ∃ k : ℝ, ∀ y ∃ x, y = k * (x - 1) ∧ (x, y) = A ∨ (x, y) = B) :
  |dist A focus| * |dist B focus| = 4 :=
sorry

end min_product_of_distances_l670_670623


namespace find_a_l670_670436

def has_three_distinct_solutions (f : Real → Real) [∀ x, Decidable (f x = 0)] : Prop :=
  ∃ (x1 x2 x3 : Real), x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0

theorem find_a (a : Real) :
  (has_three_distinct_solutions (λ x, abs (abs (x - a) - a) - 2)) ↔ a = 2 :=
by
  sorry

end find_a_l670_670436


namespace find_n_between_50_and_150_l670_670298

theorem find_n_between_50_and_150 :
  ∃ (n : ℤ), 50 ≤ n ∧ n ≤ 150 ∧
  n % 7 = 0 ∧ 
  n % 9 = 3 ∧ 
  n % 6 = 3 ∧ 
  n % 4 = 1 ∧
  n = 105 :=
by
  sorry

end find_n_between_50_and_150_l670_670298


namespace probability_of_fx_leq_zero_is_3_over_10_l670_670753

noncomputable def fx (x : ℝ) : ℝ := -x + 2

def in_interval (x : ℝ) (a b : ℝ) : Prop := a ≤ x ∧ x ≤ b

def probability_fx_leq_zero : ℚ :=
  let interval_start := -5
  let interval_end := 5
  let fx_leq_zero_start := 2
  let fx_leq_zero_end := 5
  (fx_leq_zero_end - fx_leq_zero_start) / (interval_end - interval_start)

theorem probability_of_fx_leq_zero_is_3_over_10 :
  probability_fx_leq_zero = 3 / 10 :=
sorry

end probability_of_fx_leq_zero_is_3_over_10_l670_670753


namespace bisector_of_exterior_angle_sq_l670_670471

variables {A B C P Q : Type}
variables (a b m n u v : ℕ)
variables (ℓ ℓ* : ℝ)

-- Given conditions
axiom external_bisector_triangle (h1 : ℓ = sqrt (a * b - m * n))
    (h2 : v * n = m * u) : ℓ* = sqrt((u + n) ^ 2 - ℓ^2) :=
begin
  sorry
end

-- Main statement to prove
theorem bisector_of_exterior_angle_sq (a b m n u v : ℕ)
    (ℓ ℓ* : ℝ)
    (h1: ℓ^2 = a * b - m * n)
    (h2: v * n = m * u): ℓ*^2 = u * v - a * b := 
by 
begin
  sorry
end

end bisector_of_exterior_angle_sq_l670_670471


namespace jennifer_spent_fraction_on_museum_ticket_l670_670052

theorem jennifer_spent_fraction_on_museum_ticket 
  (initial_money : ℝ)
  (sandwich_fraction : ℝ)
  (book_fraction : ℝ)
  (left_over_money : ℝ) 
  (initial_cond : initial_money = 120) 
  (sandwich_cond : sandwich_fraction = 1/5) 
  (book_cond : book_fraction = 1/2) 
  (left_over_cond : left_over_money = 16) :
  let sandwich_money := initial_money * sandwich_fraction in
  let book_money := initial_money * book_fraction in
  let total_spent := initial_money - left_over_money in
  let combined_spent_money := sandwich_money + book_money in
  let museum_ticket_money := total_spent - combined_spent_money in
  (museum_ticket_money / initial_money) = 1/6 :=
by
  sorry

end jennifer_spent_fraction_on_museum_ticket_l670_670052


namespace sqrt_product_simplifies_l670_670481

theorem sqrt_product_simplifies :
  real.sqrt 12 * real.sqrt 75 = 30 :=
by
  -- proof omitted
  sorry

end sqrt_product_simplifies_l670_670481


namespace best_statistical_chart_for_temperature_changes_l670_670535

noncomputable def best_chart (data: List ℝ) : string :=
  "Line Chart"

theorem best_statistical_chart_for_temperature_changes (temperatures: List ℝ) :
  best_chart temperatures = "Line Chart" :=
by
  sorry

end best_statistical_chart_for_temperature_changes_l670_670535


namespace collinear_A2_B2_C2_perpendicular_to_OH_l670_670064

noncomputable theory

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables (ABC : triangle A B C) (A1 B1 C1 A2 B2 C2 : A)
variables (O : A) (H : A)

def orthogonal_projection (P : A) (s : set A) : A := sorry -- Definition of orthogonal projection
def intersection (l1 l2 : set A) : A := sorry -- Definition of line intersection

axiom orthogonal_projection_exists (P : A) (s : set A) : ∃ P', orthogonal_projection P s = P'

-- Given conditions
axiom cond1 : ¬(right_angled ABC) ∧ ¬(isosceles ABC)
axiom cond2 : A1 = orthogonal_projection A (side BC ABC)
axiom cond3 : B1 = orthogonal_projection B (side CA ABC)
axiom cond4 : C1 = orthogonal_projection C (side AB ABC)
axiom cond5 : A2 = intersection (line BC) (line B1C1)
axiom cond6 : B2 = intersection (line AC) (line A1C1)
axiom cond7 : C2 = intersection (line AB) (line A1B1)

-- Prove collinearity of A2, B2, and C2
theorem collinear_A2_B2_C2 :
  collinear A2 B2 C2 :=
sorry

-- Prove perpendicularity to (OH)
theorem perpendicular_to_OH :
  is_perpendicular (line_through A2 B2 C2) (line_through O H) :=
sorry

end collinear_A2_B2_C2_perpendicular_to_OH_l670_670064


namespace range_of_S_l670_670439

theorem range_of_S (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  1 < (a / (a + b + d) + b / (a + b + c) + c / (b + c + d) + d / (a + c + d)) < 2 :=
by
  sorry

end range_of_S_l670_670439


namespace joan_gemstone_samples_l670_670413

theorem joan_gemstone_samples
  (minerals_yesterday : ℕ)
  (gemstones : ℕ)
  (h1 : minerals_yesterday + 6 = 48)
  (h2 : gemstones = minerals_yesterday / 2) :
  gemstones = 21 :=
by
  sorry

end joan_gemstone_samples_l670_670413


namespace parallel_lines_condition_l670_670772

def l1_slope (m : ℝ) : ℝ := -(m + 3) / 5
def l2_slope (m : ℝ) : ℝ := -2 / (m + 6)

theorem parallel_lines_condition (m : ℝ) :
  l1_slope m = l2_slope m ↔ m = -8 := by
  sorry

end parallel_lines_condition_l670_670772


namespace complex_magnitude_product_l670_670287

theorem complex_magnitude_product :
  |Complex.mk 7 (-24) * Complex.mk (-5) 10| = 125 * Real.sqrt 5 :=
by
  sorry

end complex_magnitude_product_l670_670287


namespace find_pairs_l670_670710

theorem find_pairs (m n : ℕ) (h : (m + n) * |m - n| = 2021) :
  (m = 1011 ∧ n = 1010) ∨ (m = 45 ∧ n = 2) :=
by
  sorry

end find_pairs_l670_670710


namespace males_band_not_orchestra_l670_670123

variables {F_band F_orch F_both : ℕ} {M_band M_orch M_students : ℕ}
variables {total_students : ℕ}

def females_in_band := F_band = 120
def females_in_orchestra := F_orch = 90
def females_in_both := F_both = 70
def males_in_band := M_band = 90
def males_in_orchestra := M_orch = 120
def total_students_or := total_students = 250

theorem males_band_not_orchestra :
  females_in_band → females_in_orchestra → females_in_both → males_in_band → males_in_orchestra →
  total_students_or → 
  let total_females := F_band + F_orch - F_both in
  let total_males := total_students - total_females in
  let males_both := M_band + M_orch - total_males in
  let males_band_not_in_orchestra := M_band - males_both in
  males_band_not_in_orchestra = 0 :=
sorry

end males_band_not_orchestra_l670_670123


namespace probability_exceeds_200_and_within_600_chi_squared_relationship_l670_670608

-- Definition of API ranges and days
def api_days : List (ℕ × ℕ) := [
  (50, 4), (100, 13), (150, 18), (200, 30), (250, 9), (300, 11), (∞, 15)
]

-- Definition for the economic loss S
def economic_loss (ω : ℝ) : ℝ :=
if ω <= 100 then 0 else if ω <= 300 then 4 * ω - 400 else 2000

-- Definition for days with API between 150 and 250
def days_150_to_250 := 30 + 9

-- Theorems to be proved for Part 1
theorem probability_exceeds_200_and_within_600 : 
  (days_150_to_250 / 100 : ℝ) = 0.39 :=
by
  -- We assume the division of days_150_to_250 by 100 is equal to 0.39
  sorry

-- Contingency table values
def a := 22
def b := 8
def c := 63
def d := 7
def n := 100

-- Definition for the Chi-square statistic K^2
def K_squared (a b c d n : ℝ) : ℝ :=
  n * ((a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Computed K^2 value for the question
def K2_value := K_squared a b c d n

-- Theorems to be proved for Part 2
theorem chi_squared_relationship :
  K2_value > 3.841 :=
by
  -- We assume the calculated K2_value is greater than the critical value for 95% confidence
  sorry

end probability_exceeds_200_and_within_600_chi_squared_relationship_l670_670608


namespace balls_in_boxes_l670_670352

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l670_670352


namespace exactly_one_correct_l670_670643

-- Definitions based on given conditions
def statement1 := ∀ (P L : Type), ∃! M : L → P, parallel L M

def statement2 := ∀ {a b c : Type}, (parallel a b ∧ parallel a c) → parallel b c

def statement3 := ∀ (seg1 seg2 : Type), ¬intersect seg1 seg2 → parallel seg1 seg2

def statement4 := ∀ (line1 line2 : Type), ¬intersect line1 line2 → parallel line1 line2

-- Prove that exactly one statement is correct
theorem exactly_one_correct : (statement1 → false) ∧
                              (statement2 → true) ∧
                              (statement3 → false) ∧
                              (statement4 → false) ↔ (statement2 = true ∧ statement1 = false ∧ statement3 = false ∧ statement4 = false) := 
by 
  sorry

end exactly_one_correct_l670_670643


namespace quadrilateral_is_parallelogram_l670_670510

theorem quadrilateral_is_parallelogram (a b c d : ℝ) (h : (a - c) ^ 2 + (b - d) ^ 2 = 0) : 
  -- The theorem states that if lengths a, b, c, d of a quadrilateral satisfy the given equation,
  -- then the quadrilateral must be a parallelogram.
  a = c ∧ b = d :=
by {
  sorry
}

end quadrilateral_is_parallelogram_l670_670510


namespace max_full_marks_probability_l670_670124

-- Define the total number of mock exams
def total_mock_exams : ℕ := 20
-- Define the number of full marks scored in mock exams
def full_marks_in_mocks : ℕ := 8

-- Define the probability of event A (scoring full marks in the first test)
def P_A : ℚ := full_marks_in_mocks / total_mock_exams

-- Define the probability of not scoring full marks in the first test
def P_neg_A : ℚ := 1 - P_A

-- Define the probability of event B (scoring full marks in the second test)
def P_B : ℚ := 1 / 2

-- Define the maximum probability of scoring full marks in either the first or the second test
def max_probability : ℚ := P_A + P_neg_A * P_B

-- The main theorem conjecture
theorem max_full_marks_probability :
  max_probability = 7 / 10 :=
by
  -- Inserting placeholder to skip the proof for now
  sorry

end max_full_marks_probability_l670_670124


namespace smallest_three_digit_number_l670_670357

theorem smallest_three_digit_number (digits : Finset ℕ) (h_digits : digits = {0, 3, 5, 6}) : 
  ∃ n, n = 305 ∧ ∀ m, (m ∈ digits) → (m ≠ 0) → (m < 305) → false :=
by
  sorry

end smallest_three_digit_number_l670_670357


namespace intersection_P_Q_l670_670443

open Set

def P := { -3, 0, 2, 4 }
def Q := { x : ℝ | -1 < x ∧ x < 3 }

theorem intersection_P_Q :
  P ∩ Q = {0, 2} :=
sorry

end intersection_P_Q_l670_670443


namespace shinyoung_initial_candies_l670_670478

theorem shinyoung_initial_candies : 
  ∀ (C : ℕ), 
    (C / 2) - ((C / 6) + 5) = 5 → 
    C = 30 := by
  intros C h
  sorry

end shinyoung_initial_candies_l670_670478


namespace simplify_trig_expression_l670_670905

theorem simplify_trig_expression (x : ℝ) : 
  (2 + 2 * sin x - 2 * cos x) / (2 + 2 * sin x + 2 * cos x) = tan (x / 2) :=
by
  sorry

end simplify_trig_expression_l670_670905


namespace building_height_is_74_l670_670184

theorem building_height_is_74
  (building_shadow : ℚ)
  (flagpole_height : ℚ)
  (flagpole_shadow : ℚ)
  (ratio_valid : building_shadow / flagpole_shadow = 21 / 8)
  (flagpole_height_value : flagpole_height = 28)
  (building_shadow_value : building_shadow = 84)
  (flagpole_shadow_value : flagpole_shadow = 32) :
  ∃ (h : ℚ), h = 74 := by
  sorry

end building_height_is_74_l670_670184


namespace reflection_lies_on_circumcircle_of_BDE_l670_670433

variables {A B C I D E : Type*}
variables [triangle ABC : is_triangle A B C]
variables [incenter ABC I : is_incenter A B C I]

-- Given AB = AC and AB ≠ BC
variables [equality1 : AB = AC]
variable [inequality1 : AB ≠ BC]

-- BI intersects AC at D
axiom B_inter_AC_at_D : intersects B I C D

-- Line through D perpendicular to AC meets AI at E
axiom perpendicular_through_D_meets_AI_at_E : perpendicular_line_through D C intersects A I E

-- Define the reflection of I over AC
noncomputable def reflectionI_AC : Type* := reflection_over_line I C

-- Main theorem
theorem reflection_lies_on_circumcircle_of_BDE :
  ∃ O : Type*, circumcircle B D E O ∧ reflectionI_AC ∈ O := 
by sorry

end reflection_lies_on_circumcircle_of_BDE_l670_670433


namespace largest_quotient_in_set_l670_670551

def set_of_numbers := {-36, -6, -4, 3, 7, 9}

theorem largest_quotient_in_set : ∃ (a b : ℤ), a ∈ set_of_numbers ∧ b ∈ set_of_numbers ∧ b ≠ 0 ∧ a / b = 9 :=
by
  sorry

end largest_quotient_in_set_l670_670551


namespace fish_population_approximation_l670_670036

noncomputable def approx_number_of_fish_in_lake
    (total_tagged : ℕ) (caught_second_time : ℕ) (tagged_second_time : ℕ) : ℕ :=
  let N := (total_tagged * caught_second_time) / tagged_second_time
  in N * caught_second_time

theorem fish_population_approximation :
  approx_number_of_fish_in_lake 500 300 45 = 8100 :=
by
  sorry

end fish_population_approximation_l670_670036


namespace find_bounds_l670_670857

open Set

variable {U : Type} [TopologicalSpace U]

def A := {x : ℝ | 3 ≤ x ∧ x ≤ 4}
def C_UA := {x : ℝ | x > 4 ∨ x < 3}

theorem find_bounds (T : Type) [TopologicalSpace T] : 3 = 3 ∧ 4 = 4 := 
 by sorry

end find_bounds_l670_670857


namespace milk_needed_l670_670882

theorem milk_needed (r_milk_flour : ℕ → ℕ → Prop)
  (milk_used : ℕ) (flour_used : ℕ) (total_flour : ℕ) :
  r_milk_flour milk_used flour_used →
  flour_used = 250 →
  milk_used = 75 →
  total_flour = 1250 →
  ∃ total_milk, total_milk = (total_flour / flour_used) * milk_used ∧ total_milk = 375 :=
by
  intros h_r hf hf' hf''
  use (total_flour / flour_used) * milk_used
  split
  { exact rfl }
  { calc (total_flour / flour_used) * milk_used
        = (1250 / 250) * 75 : by rw [hf, hf']
    ... = 5 * 75 : by norm_num
    ... = 375 : by norm_num }

end milk_needed_l670_670882


namespace last_two_digits_of_sum_l670_670651

theorem last_two_digits_of_sum :
  (List.sum (List.map (λ n, Nat.factorial n) [5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]) % 100) = 60 :=
by 
  sorry

end last_two_digits_of_sum_l670_670651


namespace cistern_filling_time_l670_670199

-- Definitions translated from conditions
def pipeA_fill_time := 10  -- Pipe A fills the cistern in 10 hours
def pipeB_empty_time := 15 -- Pipe B empties the full cistern in 15 hours

-- The proof statement
theorem cistern_filling_time (C : ℕ) (hA : C / pipeA_fill_time) (hB : C / pipeB_empty_time) : true :=
  let pipeA_rate := C / pipeA_fill_time
  let pipeB_rate := C / pipeB_empty_time
  let net_rate := (pipeA_rate - pipeB_rate)
  have netTime : (C / net_rate) = 30 := by sorry
  trivial

end cistern_filling_time_l670_670199


namespace circumradius_correct_inradius_correct_l670_670135

-- Define the lengths of the edges
def A := 2
def B := 4
def C := 6

-- Calculate the circumradius of the pyramid
def circumradius : ℝ := (real.sqrt (2 ^ 2 + 4 ^ 2 + 6 ^ 2)) / 2

-- Calculate the inradius of the pyramid
def inradius (volume : ℝ) (area_sum : ℝ) : ℝ := (3 * volume) / area_sum

-- Volume of the pyramid
def pyramid_volume : ℝ := (1/6 : ℝ) * A * B * C

-- Sum of the areas of the faces of the pyramid
def area_sum : ℝ :=
  let face1 := (1/2 : ℝ) * A * B
  let face2 := (1/2 : ℝ) * A * C
  let face3 := (1/2 : ℝ) * B * C
  let face4 := real.sqrt (40) + real.sqrt (20) + real.sqrt (52)
  face1 + face2 + face3 + face4

-- The correct answers
theorem circumradius_correct : circumradius = real.sqrt 14 :=
by
  sorry

theorem inradius_correct : inradius pyramid_volume area_sum = (2 / 3) :=
by
  sorry

end circumradius_correct_inradius_correct_l670_670135


namespace distribute_balls_into_boxes_l670_670348

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l670_670348


namespace limit_r_as_m_approaches_zero_l670_670854

noncomputable def L (m : ℝ) : ℝ := -real.sqrt (m + 6)

theorem limit_r_as_m_approaches_zero :
  ∀ m : ℝ, -6 < m → m < 6 →
  tendsto (λ m : ℝ, (L (-m) - L m) / m) (𝓝 0) (𝓝 (1 / real.sqrt 6)) :=
by
  sorry

end limit_r_as_m_approaches_zero_l670_670854


namespace factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l670_670288

theorem factorize_polynomial_1 (x y : ℝ) : 
  12 * x ^ 3 * y - 3 * x * y ^ 2 = 3 * x * y * (4 * x ^ 2 - y) := 
by sorry

theorem factorize_polynomial_2 (x : ℝ) : 
  x - 9 * x ^ 3 = x * (1 + 3 * x) * (1 - 3 * x) :=
by sorry

theorem factorize_polynomial_3 (a b : ℝ) : 
  3 * a ^ 2 - 12 * a * b * (a - b) = 3 * (a - 2 * b) ^ 2 := 
by sorry

end factorize_polynomial_1_factorize_polynomial_2_factorize_polynomial_3_l670_670288


namespace perimeter_of_triangle_l670_670319

-- Assume definitions for the foci and intersection points
variable (F1 F2 A B : ℝ × ℝ)

-- Given condition: Ellipse equation parameters
def ellipse := ∀ (x y : ℝ), (x / 4)^2 + (y / 3)^2 = 1

-- Condition 1: Points F1 and F2 are the foci of the ellipse
def is_foci (F1 F2 : ℝ × ℝ) := (Ellipsis)

-- Condition 2: Points A and B intersect the ellipse and the line passes through F1
def intersects_ellipse_and_line (F1 A B : ℝ × ℝ) := 
(A ∈ ellipse) ∧ (B ∈ ellipse) ∧ (Line_intersects F1 A B)

-- The question: Proving the perimeter is 16
theorem perimeter_of_triangle (F1 F2 A B : ℝ × ℝ) 
(ellipse : ∀ (x y : ℝ), (x / 4)^2 + (y / 3)^2 = 1)
(hf : is_foci F1 F2) (hi : intersects_ellipse_and_line F1 A B) :
  triangle_perimeter F1 F2 A B = 16 :=
sorry

end perimeter_of_triangle_l670_670319


namespace find_PS_in_triangle_l670_670404

theorem find_PS_in_triangle
  (P Q R S : Type)
  [MetricSpace P]
  [MetricSpace Q]
  [MetricSpace R]
  [MetricSpace S]
  (PQ PR QR : ℝ)
  (QS_SR_ratio : ℝ)
  (foot_perpendicular : ∃ S, is_foot_perpendicular P Q R S):
  PQ = 13 ∧ PR = 15 ∧ QS_SR_ratio = 3/4 →
  ∃ PS : ℝ, PS = Real.sqrt 97 :=
begin
  sorry,
end

end find_PS_in_triangle_l670_670404


namespace smallest_integer_y_l670_670188

theorem smallest_integer_y : ∃ (y : ℤ), (7 + 3 * y < 25) ∧ (∀ z : ℤ, (7 + 3 * z < 25) → y ≤ z) ∧ y = 5 :=
by
  sorry

end smallest_integer_y_l670_670188


namespace parabola_line_intersection_constant_line_equation_area_l670_670314

noncomputable def parabola_intersection (y1 y2 : ℝ) (k : ℝ) := (y1 * y2 = -18)

theorem parabola_line_intersection_constant
  (y1 y2 : ℝ) :
  (parabola_intersection y1 y2 (k)) :=
sorry
  
theorem line_equation_area
  (x1 x2 y1 y2 : ℝ)
  (h1 : x2 = 1)
  (h2 : y1 + y2 = 2)
  (h3 : y1 * y2 = -18)
  (h4 : |x1 * y1| = 4)
  (area_AOB : (∃ k: ℝ, 2*k*x1 + 3*k*y1 - 9 = 0 ∨ 2*k*x2 - 3*k*y2 - 9 = 0)) :
  ( area_AOB → (2*x1 + 3*y1 - 9) = 0 ∨ (2*x2 - 3*y2 - 9) = 0) 
  :=
sorry

end parabola_line_intersection_constant_line_equation_area_l670_670314


namespace cube_rotation_volume_l670_670899

theorem cube_rotation_volume (e : ℝ) (h : e = 1) : 
  volume_of_solid_rotated_around_body_diagonal e = (Real.sqrt 3 / 3) * Real.pi :=
by 
  sorry

end cube_rotation_volume_l670_670899


namespace jacket_price_restoration_l670_670204

variable (P : ℝ) (r1 r2 : ℝ)
variable (h1 : r1 = 0.20) (h2 : r2 = 0.25) (hP_positive : P > 0)

theorem jacket_price_restoration :
  let sale_price_1 := P * (1 - r1) in
  let sale_price_2 := sale_price_1 * (1 - r2) in
  let needed_increase := (P - sale_price_2) / sale_price_2 in
  (needed_increase * 100).approx_eq 66.67 :=
by
  sorry

end jacket_price_restoration_l670_670204


namespace solution_set_of_inequality_l670_670155

theorem solution_set_of_inequality (x : ℝ) : x^2 < -2 * x + 15 ↔ -5 < x ∧ x < 3 := 
sorry

end solution_set_of_inequality_l670_670155


namespace find_A_range_of_b2_and_c2_l670_670027

-- Define the problem conditions
variables {A B C : ℝ} {a b c : ℝ}
axiom triangle_sides_and_angles :
  (a + b) * (Real.sin A - Real.sin B) = c * (Real.sin C - Real.sin B)

theorem find_A (h : triangle_sides_and_angles) : A = Real.pi / 3 :=
  sorry

theorem range_of_b2_and_c2 (h : triangle_sides_and_angles) (ha : a = 4) :
  16 < b^2 + c^2 ∧ b^2 + c^2 ≤ 32 :=
  sorry

end find_A_range_of_b2_and_c2_l670_670027


namespace fixed_point_l670_670139

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(2*x - 1)

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_ne_one : a ≠ 1) : f a (1/2) = 1 :=
by
  sorry

end fixed_point_l670_670139


namespace least_triangle_perimeter_l670_670511

theorem least_triangle_perimeter (a b : ℕ) (h₁ : a = 24) (h₂ : b = 30) :
  ∃ x : ℕ, x > 6 ∧ 24 + 30 + x = 61 :=
by
  have x : ℕ := 7
  use x
  split
  { show x > 6
    exact Nat.lt_succ_self 6 }
  { show 24 + 30 + x = 61
    sorry }

end least_triangle_perimeter_l670_670511


namespace john_twice_as_old_l670_670848

def sam_age := 9
def john_age := 27

theorem john_twice_as_old (x : ℕ) :
  (sam_age = 9) ∧ (john_age = 3 * sam_age) ∧ (27 + x = 2 * (9 + x)) → x = 9 :=
by
  intros h
  cases h with sam_eq hs
  cases hs with john_eq eq
  sorry

end john_twice_as_old_l670_670848


namespace negate_proposition_l670_670925

def p (x : ℝ) : Prop := x^2 + x - 6 > 0
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

def neg_p (x : ℝ) : Prop := x^2 + x - 6 ≤ 0
def neg_q (x : ℝ) : Prop := -3 ≤ x ∧ x ≤ 2

theorem negate_proposition (x : ℝ) :
  (¬ (p x → q x)) ↔ (neg_p x → neg_q x) :=
by unfold p q neg_p neg_q; apply sorry

end negate_proposition_l670_670925


namespace geometric_concepts_cases_l670_670697

theorem geometric_concepts_cases :
  (∃ x y, x = "rectangle" ∧ y = "rhombus") ∧ 
  (∃ x y z, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "acute_triangle") ∧ 
  (∃ x y z u, x = "parallelogram" ∧ y = "rectangle" ∧ z = "square" ∧ u = "acute_angled_rhombus") ∧ 
  (∃ x y z u t, x = "polygon" ∧ y = "triangle" ∧ z = "isosceles_triangle" ∧ u = "equilateral_triangle" ∧ t = "right_triangle") ∧ 
  (∃ x y z u, x = "right_triangle" ∧ y = "isosceles_triangle" ∧ z = "obtuse_triangle" ∧ u = "scalene_triangle") :=
by {
  sorry
}

end geometric_concepts_cases_l670_670697


namespace infinite_geometric_series_sum_l670_670682

theorem infinite_geometric_series_sum :
  let a := (5 : ℚ) / 3
  let r := -(3 : ℚ) / 4
  ∑' n : ℕ, a * r ^ n = 20 / 21 := by
  sorry

end infinite_geometric_series_sum_l670_670682


namespace journey_proportion_and_time_ratio_l670_670585

-- Define the constants for the problem
variable (v_p : ℝ) -- Speed of walkers
variable (v_b : ℝ := 7 * v_p) -- Speed of bus
variable (d : ℝ) -- Distance from A to B
variable (n : ℝ := 1 / 4) -- Proportion of group the bus can carry

-- The time ratio if using direct transportation compared to the optimized approach
constant time_ratio : ℝ := 49 / 25

-- Prove the proportion of the journey traveled by bus and on foot, and the time ratio
theorem journey_proportion_and_time_ratio
  (h1 : ∀ g1 g2 g3 g4 : ℝ, g1 = n * d ∧ g2 = n * d ∧ g3 = n * d ∧ g4 = n * d) 
  (h2 : v_b = 7 * v_p) 
  (h3 : ∀ l : ℝ, l = 4 * n * d) -- total travel distance by bus for each group 
  :
  (3 / 7, 4 / 7, 49 / 25) = (3 / 7, 4 / 7, time_ratio) :=
by
  -- Skipping the proof
  sorry

end journey_proportion_and_time_ratio_l670_670585


namespace number_with_smallest_prime_factor_in_C_l670_670477

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def smallest_prime_factor (n : ℕ) : ℕ :=
  if is_prime n then n
  else if n % 2 = 0 then 2
  else if n % 3 = 0 then 3
  else if n % 5 = 0 then 5
  else if n % 7 = 0 then 7
  else if n % 11 = 0 then 11
  else if n % 13 = 0 then 13
  else n -- This simplifies the small range we are dealing with

def min_prime_factor_in_set (s : finset ℕ) : ℕ :=
  finset.min' s ((((finset.val s).map smallest_prime_factor).to_finset))

def smallest_prime_factor_proof (C: finset ℕ) (x: ℕ) : Prop :=
  min_prime_factor_in_set C = smallest_prime_factor x ∧ x ∈ C

def set_C : finset ℕ := {47, 49, 51, 53, 55}

theorem number_with_smallest_prime_factor_in_C :
  smallest_prime_factor_proof set_C 51 :=
by
  sorry

end number_with_smallest_prime_factor_in_C_l670_670477


namespace magnitude_a_minus_2b_l670_670751

noncomputable theory

variables (a b : ℝ^3)

-- Defining the conditions
def unit_vector (v : ℝ^3) := ∥v∥ = 1
def angle_pi_over_3 (a b : ℝ^3) := real_inner a b = real.cos (real.pi / 3)

-- The main theorem to be proven
theorem magnitude_a_minus_2b (huva : unit_vector a) (huhb : unit_vector b) (haab : angle_pi_over_3 a b) :
  ∥a - (2 : ℝ) • b∥ = real.sqrt 3 := 
sorry

end magnitude_a_minus_2b_l670_670751


namespace single_digit_pairs_l670_670674

theorem single_digit_pairs:
  ∃ x y: ℕ, x ≠ 1 ∧ x ≠ 9 ∧ y ≠ 1 ∧ y ≠ 9 ∧ x < 10 ∧ y < 10 ∧ 
  (x * y < 100 ∧ ((x * y) % 10 + (x * y) / 10 == x ∨ (x * y) % 10 + (x * y) / 10 == y))
  → (x, y) ∈ [(3, 4), (3, 7), (6, 4), (6, 7)] :=
by
  sorry

end single_digit_pairs_l670_670674


namespace circle_standard_equation_l670_670022

-- Define the problem conditions and prove the standard equation of the circle
theorem circle_standard_equation
  (r : ℝ) 
  (h : ℝ) 
  (k : ℝ)
  (hyp1 : r = 1)
  (hyp2 : (h, k) = (0, 1)) :
  ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 :=
begin
  -- completing the theorem statement with the given data
  intros x y,
  rw [hyp1, hyp2],
  exact (x^2 + (y - 1)^2 = 1)
end

end circle_standard_equation_l670_670022


namespace find_m_l670_670358

theorem find_m (m : ℝ) (a : ℕ → ℝ) :
  (∀ x : ℝ, (1 + m * x)^6 = ∑ i in Finset.range 7, a i * x ^ i)
  → (a 1 - a 2 + a 3 - a 4 + a 5 - a 6 = -63)
  → ((1 - m)^6 = 64) :=
by
  intros H1 H2
  have h : (a 0 = 1) := by sorry
  have h1 : (a 0 - a 1 + a 2 - a 3 + a 4 - a 5 + a 6 = (1 - m)^6) := by sorry
  have h2 : (a 0 - a 1 + a 2 - a 3 + a 4 - a 5 + a 6 = 64) := by sorry
  rw ←h2 at h1
  have h3 : (1 - m)^6 = 64 := by sorry
  exact h3

end find_m_l670_670358


namespace brick_length_l670_670603

theorem brick_length 
  (width : ℝ) (height : ℝ) (num_bricks : ℕ)
  (wall_length : ℝ) (wall_width : ℝ) (wall_height : ℝ)
  (brick_vol : ℝ) :
  width = 10 →
  height = 7.5 →
  num_bricks = 27000 →
  wall_length = 27 →
  wall_width = 2 →
  wall_height = 0.75 →
  brick_vol = width * height * (20:ℝ) →
  wall_length * wall_width * wall_height * 1000000 = num_bricks * brick_vol :=
by
  intros
  sorry

end brick_length_l670_670603


namespace part_I_part_II_l670_670757

noncomputable def f (x : ℝ) : ℝ :=
  |x - (1/2)| + |x + (1/2)|

def solutionSetM : Set ℝ :=
  { x : ℝ | -1 < x ∧ x < 1 }

theorem part_I :
  { x : ℝ | f x < 2 } = solutionSetM := 
sorry

theorem part_II (a b : ℝ) (ha : a ∈ solutionSetM) (hb : b ∈ solutionSetM) :
  |a + b| < |1 + a * b| :=
sorry

end part_I_part_II_l670_670757


namespace ratio_of_square_areas_l670_670117

theorem ratio_of_square_areas (y : ℝ) (hy : y > 0) : 
  (y^2 / (3 * y)^2) = 1 / 9 :=
sorry

end ratio_of_square_areas_l670_670117


namespace number_subtracted_from_10000_l670_670013

theorem number_subtracted_from_10000 (x : ℕ) (h : 10000 - x = 9001) : x = 999 := by
  sorry

end number_subtracted_from_10000_l670_670013


namespace abs_ratio_sum_of_products_l670_670005

noncomputable def abs_sum_of_products {n s : ℕ} (a : Fin n → ℂ) : ℂ := 
  ∑ t in Finset.univ.filter (λ t, t < Finset.card (Finset.powersetLen s Finset.univ)), (∏ i in t.1.to_finset, a i)

theorem abs_ratio_sum_of_products {n s : ℕ} (a : Fin n → ℂ) (r : ℝ) (hr : r ≠ 0)
  (habs : ∀ i, |a i| = r) (hn: s ≤ n) : 
  ∀ T_s' T_{n-s}', abs (T_s' / T_{n-s}') = r^(2 * s - n) :=
by
  -- Introduce the constraints and define the sums
  sorry

end abs_ratio_sum_of_products_l670_670005


namespace remainder_of_count_divided_by_500_l670_670855

def has_more_ones_than_zeros (n : ℕ) : Prop :=
  let bits := n.bits
  (bits.count Nat.one) > (bits.count Nat.zero)

noncomputable def count_satisfying_numbers : ℕ :=
  (List.range' 1 1500).count has_more_ones_than_zeros

theorem remainder_of_count_divided_by_500 :
  (count_satisfying_numbers % 500) = 152 := by
  sorry

end remainder_of_count_divided_by_500_l670_670855


namespace solution_set_of_quadratic_inequality_l670_670156

theorem solution_set_of_quadratic_inequality (x : ℝ) : x^2 < x + 6 ↔ -2 < x ∧ x < 3 := 
by
  sorry

end solution_set_of_quadratic_inequality_l670_670156


namespace lines_through_origin_l670_670062

theorem lines_through_origin (n : ℕ) (h : 0 < n) :
    ∃ S : Finset (ℤ × ℤ), 
    (∀ xy : ℤ × ℤ, xy ∈ S ↔ (0 ≤ xy.1 ∧ xy.1 ≤ n ∧ 0 ≤ xy.2 ∧ xy.2 ≤ n ∧ Int.gcd xy.1 xy.2 = 1)) ∧
    S.card ≥ n^2 / 4 := 
sorry

end lines_through_origin_l670_670062


namespace oblique_projection_area_oblique_projection_area_transformed_l670_670014

-- Define the oblique projection method conditions
def base_unchanged (b: ℝ) : Prop := b = b

def height_halved (h: ℝ) (h': ℝ) : Prop := h' = h / 2

-- Main proof statement
theorem oblique_projection_area (A: ℝ) (b: ℝ) (h: ℝ) (b_reduced: base_unchanged b) (h_reduced: height_halved h h') :
  A' = (h' * b) / (h * b) * A := by
  sorry

-- Prove the area becomes 𝑓(√2)/4 times the original.
theorem oblique_projection_area_transformed (A_original: ℝ) (b: ℝ) (h: ℝ) (b_reduced: base_unchanged b) (h_reduced: height_halved h (h / 2)) :
  let A' := (h / 2 * b) / (h * b) * A_original in  A' = A_original * (sqrt 2 / 4) := by
  sorry

end oblique_projection_area_oblique_projection_area_transformed_l670_670014


namespace max_points_is_18_l670_670459

-- Define points on a coordinate plane
variables {Point : Type} [HasCoords Point] (n : ℕ)

-- Define colors
inductive Color
| Red | Green | Yellow

-- Assume no three points are collinear
def noThreeCollinear (points : set Point) : Prop :=
  ∀ (p1 p2 p3 : Point), p1 ∈ points → p2 ∈ points → p3 ∈ points → 
  ¬ areCollinear p1 p2 p3

-- Define a set of points with their colors
structure ColoredPoints :=
  (points : set Point)
  (color : Point → Color)

-- Define the condition functions
def redTriangleHasGreen {ColoredPoints : ColoredPoints} : Prop :=
  ∀ (p1 p2 p3 : Point), 
  ColoredPoints.color p1 = Color.Red → ColoredPoints.color p2 = Color.Red → ColoredPoints.color p3 = Color.Red → 
  (∃ p : Point, p ∈ insideTriangle p1 p2 p3 ∧ ColoredPoints.color p = Color.Green)

def greenTriangleHasYellow {ColoredPoints : ColoredPoints} : Prop :=
  ∀ (p1 p2 p3 : Point), 
  ColoredPoints.color p1 = Color.Green → ColoredPoints.color p2 = Color.Green → ColoredPoints.color p3 = Color.Green → 
  (∃ p : Point, p ∈ insideTriangle p1 p2 p3 ∧ ColoredPoints.color p = Color.Yellow)

def yellowTriangleHasRed {ColoredPoints : ColoredPoints} : Prop :=
  ∀ (p1 p2 p3 : Point), 
  ColoredPoints.color p1 = Color.Yellow → ColoredPoints.color p2 = Color.Yellow → ColoredPoints.color p3 = Color.Yellow → 
  (∃ p : Point, p ∈ insideTriangle p1 p2 p3 ∧ ColoredPoints.color p = Color.Red)

def maxPoints (ColoredPoints : ColoredPoints) : Prop :=
  redTriangleHasGreen ColoredPoints ∧
  greenTriangleHasYellow ColoredPoints ∧
  yellowTriangleHasRed ColoredPoints

-- Now, state the main theorem
theorem max_points_is_18 (ColoredPoints : ColoredPoints) 
  (h_noThreeCollinear : noThreeCollinear ColoredPoints.points)
  (h_maxPoints : maxPoints ColoredPoints) :
  n ≤ 18 := 
sorry

end max_points_is_18_l670_670459


namespace train_passes_man_in_15_seconds_l670_670635

theorem train_passes_man_in_15_seconds
  (length_of_train : ℝ)
  (speed_of_train : ℝ)
  (speed_of_man : ℝ)
  (direction_opposite : Bool)
  (h1 : length_of_train = 275)
  (h2 : speed_of_train = 60)
  (h3 : speed_of_man = 6)
  (h4 : direction_opposite = true) : 
  ∃ t : ℝ, t = 15 :=
by
  sorry

end train_passes_man_in_15_seconds_l670_670635


namespace max_rainbow_vertices_bound_l670_670180
noncomputable theory

def is_rainbow {n : ℕ} (colors: Finset (Fin (n - 1))) (v : Fin n) (adj_list: Fin n → Finset (Fin (n - 1))) : Prop :=
  (adj_list v) = colors

def max_rainbow_vertices (n : ℕ) (colors: Finset (Fin (n - 1))) (adj_list: Fin n → Finset (Fin (n - 1))) : ℕ :=
  ∑ v in Finset.univ.filter (is_rainbow colors adj_list), 1

theorem max_rainbow_vertices_bound (n : ℕ) (colors: Finset (Fin (n - 1))) (adj_list: Fin n → Finset (Fin (n - 1))) :
  (max_rainbow_vertices n colors adj_list ≤ if even n then n else n - 1) :=
  sorry

end max_rainbow_vertices_bound_l670_670180


namespace dinner_cost_l670_670255

variable (total_cost : ℝ)
variable (tax_rate : ℝ)
variable (tip_rate : ℝ)
variable (pre_tax_cost : ℝ)
variable (tip : ℝ)
variable (tax : ℝ)
variable (final_cost : ℝ)

axiom h1 : total_cost = 27.50
axiom h2 : tax_rate = 0.10
axiom h3 : tip_rate = 0.15
axiom h4 : tax = tax_rate * pre_tax_cost
axiom h5 : tip = tip_rate * pre_tax_cost
axiom h6 : final_cost = pre_tax_cost + tax + tip

theorem dinner_cost : pre_tax_cost = 22 := by sorry

end dinner_cost_l670_670255


namespace calc_segments_length_l670_670824

noncomputable theory

def segment_length_proof (r : ℝ) (ch : ℝ) (a n b : ℝ) : Prop := 
  r = 6 ∧ ch = 10 ∧ a = Real.sqrt 11 ∧ 
  let an := 6 - a in
  let nb := 6 + a in
  an = 6 - Real.sqrt 11 ∧ nb = 6 + Real.sqrt 11

theorem calc_segments_length : 
  ∃ r ch a n b, 
    segment_length_proof r ch a n b :=
by {
  use 6,
  use 10,
  use Real.sqrt 11,
  use 6 - Real.sqrt 11,
  use 6 + Real.sqrt 11,
  dsimp [segment_length_proof],
  repeat { split },
  exact trivial,
  exact trivial,
  exact trivial,
  exact trivial,
}

end calc_segments_length_l670_670824


namespace trapezoid_problem_solution_l670_670461

noncomputable def greatestIntegerNotExceeding (x : ℝ) : ℕ :=
  ⌊x⌋.toNat

theorem trapezoid_problem_solution :
  ∀ (b h : ℝ), b = 37.5 → (h > 0) → 
  let x := 62.5 in
  greatestIntegerNotExceeding (x^2 / 50) = 78 := 
by
  intros b h hb h_pos x
  sorry

end trapezoid_problem_solution_l670_670461


namespace train_length_approx_l670_670637

noncomputable def speed_in_kmh := 108 -- speed in km/hr
noncomputable def time_in_seconds := 4.666293363197611 -- time in seconds

noncomputable def speed_in_mps := speed_in_kmh * 1000 / 3600 -- converting km/hr to m/s

noncomputable def length_of_train := speed_in_mps * time_in_seconds -- calculating the length in meters

theorem train_length_approx (h1 : speed_in_kmh = 108) (h2 : time_in_seconds = 4.666293363197611) :
  length_of_train ≈ 140 := by
  sorry

end train_length_approx_l670_670637


namespace angle_is_ninety_degrees_l670_670070

open Real

def a : ℝ^3 := ![2, -3, -4]
def b : ℝ^3 := ![Real.sqrt 3, 5, -2]
def c : ℝ^3 := ![8, -5, 12]

def dot_product (u v : ℝ^3) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def new_vector : ℝ^3 :=
  (dot_product a c) • b - (dot_product a b) • c

theorem angle_is_ninety_degrees :
  dot_product a new_vector = 0 := sorry

end angle_is_ninety_degrees_l670_670070


namespace estimate_within_1000_to_2000_correct_l670_670999

def total_employees := 2000
def sample_size := 200
def within_1000_m := 10
def within_2000_m := 30

-- Define a function to estimate the number of employees in (1000, 2000] meters.
def estimate_within_1000_to_2000 (sample_size total_employees within_1000_m within_2000_m : ℕ) : ℕ :=
  let within_range_sample := within_2000_m - within_1000_m
  in (within_range_sample * total_employees) / sample_size

-- The theorem to prove the estimation.
theorem estimate_within_1000_to_2000_correct :
  estimate_within_1000_to_2000 sample_size total_employees within_1000_m within_2000_m = 200 :=
by
  sorry

end estimate_within_1000_to_2000_correct_l670_670999


namespace smallest_possible_value_last_number_sequence_l670_670632

theorem smallest_possible_value_last_number_sequence :
  ∃ (a : ℕ → ℕ), 
    a 1 = 1 ∧
    a 2 = 1 ∧
    (∀ k : ℕ, 3 ≤ k → a k % a (k - 1) = 0 ∧ a k % (a (k - 1) + a (k - 2)) = 0) ∧
    a 2020 = nat.factorial 2019 :=
sorry

end smallest_possible_value_last_number_sequence_l670_670632


namespace non_intersecting_quadrilaterals_l670_670266

-- The statement of the problem in Lean

theorem non_intersecting_quadrilaterals (n : ℕ) (points : Fin 4n → ℝ × ℝ)
  (h_collinear : ∀ (a b c : Fin 4n), ¬ collinear ({points a, points b, points c} : Set (ℝ × ℝ))) :
  ∃ (quadrilaterals : Fin n → Finset (Fin 4n)), 
    (∀ k, quadrilaterals k = {4*k + 1, 4*k + 2, 4*k + 3, 4*k + 4}) ∧ 
    (∀ i j, i ≠ j → Disjoint (quadrilaterals i) (quadrilaterals j)) := 
sorry

end non_intersecting_quadrilaterals_l670_670266


namespace minimum_weight_of_crates_l670_670245

theorem minimum_weight_of_crates (max_crates : ℕ) (weight_per_crate : ℕ) (h1 : max_crates = 5) (h2 : weight_per_crate ≥ 120) : max_crates * weight_per_crate ≥ 600 :=
by
  rw h1
  have h3 : 5 * 120 = 600 := rfl
  calc
    5 * weight_per_crate ≥ 5 * 120 : by linarith
                ... = 600 : h3
  sorry

end minimum_weight_of_crates_l670_670245


namespace sequences_and_sums_l670_670750

-- Definitions from conditions
def a7 : ℕ := 7
def a7_val : ℤ := 16
def S6 : ℕ := 6
def S6_val : ℤ := 33
def b1 : ℚ := 1 / 2
def b2_x : ℚ := 2
def b3_x : ℚ := 1
def line_x_y_condition (x y : ℚ) := x - 8 * y = 0

-- Definitions of required general term sequences
def a_n (n : ℕ) : ℤ := 3 * n - 5
def b_n (n : ℕ) : ℚ := (1 / 2) ^ n

-- Sum T_n of the first n terms of sequence a_n + b_n
def T_n (n : ℕ) : ℚ :=
  (∑ i in finset.range (n + 1), (3 * i - 5)) + (∑ i in finset.range (n + 1), (1 / 2) ^ i)

-- The desired theorem to be proved
theorem sequences_and_sums (n : ℕ) :
  a_n a7 = a7_val ∧
  (∑ i in finset.range (S6 + 1), a_n i) = S6_val ∧
  b_n 1 = b1 ∧
  line_x_y_condition b2_x (b_n 2) ∧
  line_x_y_condition b3_x (b_n 3) ∧
  T_n n = (3 * n ^ 2 - 7 * n) / 2 + 1 - (1 / 2) ^ n :=
by
  sorry

end sequences_and_sums_l670_670750


namespace ratio_a_b_is_zero_l670_670137

-- Setting up the conditions
variables (a y b : ℝ)
variable (d : ℝ)
-- Condition for arithmetic sequence
axiom h1 : a + d = y
axiom h2 : y + d = b
axiom h3 : b + d = 3 * y

-- The Lean statement to prove
theorem ratio_a_b_is_zero (h1 : a + d = y) (h2 : y + d = b) (h3 : b + d = 3 * y) : a / b = 0 :=
sorry

end ratio_a_b_is_zero_l670_670137


namespace no_square_root_l670_670963

theorem no_square_root (a : ℤ) (p : ℤ) : 
  (∀ (x y : ℤ), ¬(x > 0 ∧ y > 0 ∧ a = x^2 + 2*y^2 ∧ p = 2*x*y)) → 
  (a, p) = (54, 12) :=
  by
    intro H
    apply H
    intros x y
    intro con
    cases con with pos1 con
    cases con with pos2 con
    cases con with eq1 eq2
    sorry

end no_square_root_l670_670963


namespace length_PQ_l670_670666

def triangle_PQR (P Q R : Type) [EuclideanGeometry P Q R] : Prop :=
  right_angle ∠ P Q R ∧
  PR = 15 ∧
  angle P Q R = 45

theorem length_PQ (P Q R : Type) [EuclideanGeometry P Q R] (h : triangle_PQR P Q R) : PQ = 15 :=
by 
  apply sorry

end length_PQ_l670_670666


namespace initial_dimes_proof_l670_670901

variable (initial_dimes : ℕ)
variable (given_dimes : ℕ := 7)
variable (final_dimes : ℕ := 16)

theorem initial_dimes_proof : initial_dimes + given_dimes = final_dimes → initial_dimes = 9 :=
by
  intro h
  rw [add_comm] at h
  exact nat.sub_eq_of_eq_add h sorry

end initial_dimes_proof_l670_670901


namespace tapA_turned_off_time_l670_670583

noncomputable def tapA_rate := 1 / 45
noncomputable def tapB_rate := 1 / 40
noncomputable def tapB_fill_time := 23

theorem tapA_turned_off_time :
  ∃ t : ℕ, t * (tapA_rate + tapB_rate) + tapB_fill_time * tapB_rate = 1 ∧ t = 9 :=
by
  sorry

end tapA_turned_off_time_l670_670583


namespace perpendicular_line_angle_l670_670831

-- Given conditions as Lean definitions
def is_perpendicular (θ : ℝ) : Prop :=
  let line1 := (λ t, (1 + t * Real.cos θ, t * Real.sin θ)) in
  ∀ t : ℝ, t ≠ 0 → line1 t.1 / t = - (line1 t.2 / t)

def valid_range (θ : ℝ) : Prop := 0 ≤ θ ∧ θ < Real.pi

-- The statement to be proved in Lean
theorem perpendicular_line_angle (θ : ℝ) (h1 : is_perpendicular θ) (h2 : valid_range θ) :
  θ = (3 * Real.pi / 4) :=
sorry

end perpendicular_line_angle_l670_670831


namespace gcd_polynomials_l670_670320

theorem gcd_polynomials (b : ℤ) (h: ∃ k : ℤ, b = 2 * k * 953) :
  Int.gcd (3 * b^2 + 17 * b + 23) (b + 19) = 34 :=
sorry

end gcd_polynomials_l670_670320


namespace log_lim_infty_pos_log_lim_zero_pos_log_lim_infty_neg_log_lim_zero_neg_l670_670903

-- Original conditions provided in the problem
variables {a x : ℝ}

theorem log_lim_infty_pos (h : a > 1) : filter.tendsto (λ x, real.log x / real.log a) filter.at_top filter.at_top :=
sorry

theorem log_lim_zero_pos (h : a > 1) : filter.tendsto (λ x, real.log x / real.log a) (filter.comap (λ x, x⁻¹) filter.at_top) filter.at_bot :=
sorry

theorem log_lim_infty_neg (h : a < 1) : filter.tendsto (λ x, real.log x / real.log a) filter.at_top filter.at_bot :=
sorry

theorem log_lim_zero_neg (h : a < 1) : filter.tendsto (λ x, real.log x / real.log a) (filter.comap (λ x, x⁻¹) filter.at_top) filter.at_top :=
sorry

end log_lim_infty_pos_log_lim_zero_pos_log_lim_infty_neg_log_lim_zero_neg_l670_670903


namespace sum_of_sequence_l670_670337

theorem sum_of_sequence (n : ℕ) (n_pos : 0 < n) :
  let f := λ x : ℕ, x^2 + x in
  (∑ k in Finset.range n, 1 / (f (k + 1))) = n / (n + 1) := by
    sorry

end sum_of_sequence_l670_670337


namespace new_trailers_added_l670_670165

theorem new_trailers_added :
  let initial_trailers := 25
  let initial_average_age := 15
  let years_passed := 3
  let current_average_age := 12
  let total_initial_age := initial_trailers * (initial_average_age + years_passed)
  ∀ n : Nat, 
    ((25 * 18) + (n * 3) = (25 + n) * 12) →
    n = 17 := 
by
  intros
  sorry

end new_trailers_added_l670_670165


namespace quadratic_inequality_a_value_l670_670705

theorem quadratic_inequality_a_value (a t : ℝ)
  (h_a1 : ∀ x : ℝ, t * x ^ 2 - 6 * x + t ^ 2 = 0 → (x = a ∨ x = 1))
  (h_t : t < 0) :
  a = -3 :=
by
  sorry

end quadratic_inequality_a_value_l670_670705


namespace find_divisor_l670_670530

theorem find_divisor (d : ℕ) : (55 / d) + 10 = 21 → d = 5 :=
by 
  sorry

end find_divisor_l670_670530


namespace sufficient_not_necessary_condition_l670_670870

theorem sufficient_not_necessary_condition (x y : ℝ) : 
  (x - y) * x^4 < 0 → x < y ∧ ¬(x < y → (x - y) * x^4 < 0) := 
sorry

end sufficient_not_necessary_condition_l670_670870


namespace total_boys_fraction_of_girls_l670_670148

theorem total_boys_fraction_of_girls
  (n : ℕ)
  (b1 g1 b2 g2 : ℕ)
  (h_equal_students : b1 + g1 = b2 + g2)
  (h_ratio_class1 : b1 / g1 = 2 / 3)
  (h_ratio_class2: b2 / g2 = 4 / 5) :
  ((b1 + b2) / (g1 + g2) = 19 / 26) :=
by sorry

end total_boys_fraction_of_girls_l670_670148


namespace common_point_r_polar_line_AB_l670_670047

-- Define the transformation to Cartesian coordinates for C1.
def C1 (r β : ℝ) : ℝ × ℝ :=
  (3 + r * Real.cos β, 3 + r * Real.sin β)

-- Define the Cartesian equation equivalent for C2.
def C2 : ℝ × ℝ → Prop := 
  λ (p : ℝ × ℝ), (p.1 - 1) ^ 2 + (p.2 - 1) ^ 2 = 2

-- Define the polar coordinate equation for the polar line.
def polar_line_equation (ρ θ : ℝ) : ℝ :=
  2 * ρ * Real.cos θ + 2 * ρ * Real.sin θ

theorem common_point_r (r : ℝ) (h1 : r > 0) (h2 : ∀ β : ℝ, ∃ θ : ℝ, ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + π / 4) ∧ 
    (3 + r * Real.cos β - 1) ^ 2 + (3 + r * Real.sin β - 1) ^ 2 = 2) : 
  r = Real.sqrt 2 ∨ r = 3 * Real.sqrt 2 :=
sorry

theorem polar_line_AB (r : ℝ) (h3 : ∃ A B : ℝ × ℝ, (2 * A.1 + 2 * A.2 = 3 ∨ 2 * A.1 + 2 * A.2 = 5) ∧ 
    (3 + r * Real.cos 0 - 3 + r * Real.sin 0 - 3) > 0 ∧ 
    (A ≠ B) ∧ 
    (Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2) = Real.sqrt 30 / 2)) :
  (∀ θ : ℝ, polar_line_equation r θ - 3 = 0 ∨ polar_line_equation r θ - 5 = 0) :=
sorry

end common_point_r_polar_line_AB_l670_670047


namespace ratio_of_areas_l670_670384

theorem ratio_of_areas (ABC equilateral : Triangle) (DE FG HI parallel_to_base : Line)
  (AD_eq_DG_eq_GI : length AD = length DG ∧ length DG = length GI) :
  area HIJC / area ABC = 5 / 9 :=
by 
  sorry

end ratio_of_areas_l670_670384


namespace quadratic_solution_l670_670747

theorem quadratic_solution (m n x : ℝ)
  (h1 : (x - m)^2 + n = 0) 
  (h2 : ∃ (a b : ℝ), a ≠ b ∧ (x = a ∨ x = b) ∧ (a - m)^2 + n = 0 ∧ (b - m)^2 + n = 0
    ∧ (a = -1 ∨ a = 3) ∧ (b = -1 ∨ b = 3)) :
  x = -3 ∨ x = 1 :=
by {
  sorry
}

end quadratic_solution_l670_670747


namespace starWars_earnings_correct_l670_670125

-- Define the given conditions
def lionKing_cost : ℕ := 10
def lionKing_earnings : ℕ := 200
def starWars_cost : ℕ := 25
def lionKing_profit : ℕ := lionKing_earnings - lionKing_cost
def starWars_profit : ℕ := lionKing_profit * 2
def starWars_earnings : ℕ := starWars_profit + starWars_cost

-- The theorem which states that the Star Wars earnings are indeed 405 million
theorem starWars_earnings_correct : starWars_earnings = 405 := by
  -- proof goes here
  sorry

end starWars_earnings_correct_l670_670125


namespace abs_sub_sqrt5_l670_670915

theorem abs_sub_sqrt5 :
  |2 - real.sqrt 5| = real.sqrt 5 - 2 :=
by sorry

end abs_sub_sqrt5_l670_670915


namespace no_solutions_for_k41_solutions_for_k39_l670_670035

-- Defining the conditions
variable (n : ℕ) (k : ℕ)
variable (C : ℚ) -- represents number of connections in the original group
variable (F : ℕ → ℚ) -- represents the number of connections in a subset
noncomputable theory

-- Given conditions
def condition1 : Prop := n > 20
def condition2 : Prop := ∃ (p : ℕ) (q : ℕ), p < q ∧ p ≤ n ∧ q ≤ n -- At least one pair knows each other
def condition3 : Prop := ∀ (p : ℕ) (q : ℕ), p = q → F(p) = F(q) -- Knowing is symmetric
def condition4 : Prop := ∀ (x : ℕ), x ∈ (finset.range (nat.choose n 20)) → F(x) ≤ C * (n-k)/n

-- Problem statement for k = 41
theorem no_solutions_for_k41 : 
  ∀ (n : ℕ), (k = 41) → (condition1 n) → (condition2 n) → (condition3 n) → (condition4 n) → false := 
  by sorry

-- Problem statement for k = 39
theorem solutions_for_k39 : 
  ∀ (n : ℕ), (k = 39) → (condition1 n) → (condition2 n) → (condition3 n) → (condition4 n) → n ≥ 381 := 
  by sorry

end no_solutions_for_k41_solutions_for_k39_l670_670035


namespace chord_with_integer_lengths_l670_670469

theorem chord_with_integer_lengths (O P : Point) (r : ℝ) (d : ℝ)
  (h1 : distance O P = 8) (h2 : r = 17) : 
  ∃ n : ℕ, n = 5 ∧ (∀ L, L ∈ integer_chord_lengths O P r d → L = 30 ∨ L = 31 ∨ L = 32 ∨ L = 33 ∨ L = 34) :=
sorry

end chord_with_integer_lengths_l670_670469


namespace sin_6_cos_6_theta_proof_l670_670859

noncomputable def sin_6_cos_6_theta (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : ℝ :=
  Real.sin θ ^ 6 + Real.cos θ ^ 6

theorem sin_6_cos_6_theta_proof (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : 
  sin_6_cos_6_theta θ h = 19 / 64 :=
by
  sorry

end sin_6_cos_6_theta_proof_l670_670859


namespace plane_intersects_cube_l670_670959

theorem plane_intersects_cube (P : set (set ℝ)) (C : set (set ℝ)) (isosceles_triangle trapezoid heptagon pentagon : set (set ℝ)) : 
  (P ∩ C = isosceles_triangle ∨ P ∩ C = trapezoid ∨ P ∩ C = heptagon ∨ P ∩ C = pentagon) → P ∩ C ≠ heptagon :=
begin
  sorry
end

end plane_intersects_cube_l670_670959


namespace wario_missed_field_goals_wide_right_l670_670544

theorem wario_missed_field_goals_wide_right :
  ∀ (attempts missed_fraction wide_right_fraction : ℕ), 
  attempts = 60 →
  missed_fraction = 1 / 4 →
  wide_right_fraction = 20 / 100 →
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  wide_right = 3 :=
by
  intros attempts missed_fraction wide_right_fraction h1 h2 h3
  let missed := attempts * missed_fraction
  let wide_right := missed * wide_right_fraction
  sorry

end wario_missed_field_goals_wide_right_l670_670544


namespace num_wide_right_misses_l670_670546

-- Define the conditions
variables (totalFieldGoals : ℕ) (missFraction : ℚ) (wideRightFraction : ℚ)

-- Given conditions
def totalFieldGoals := 60
def missFraction := 1 / 4
def wideRightFraction := 20 / 100

-- Calculate the actual number of field goals
def missedFieldGoals := totalFieldGoals * missFraction
def wideRightMisses := missedFieldGoals * wideRightFraction

-- Theorem to be proved
theorem num_wide_right_misses : wideRightMisses = 3 := sorry

end num_wide_right_misses_l670_670546


namespace original_price_of_car_l670_670175

theorem original_price_of_car (spent price_percent original_price : ℝ) (h1 : spent = 15000) (h2 : price_percent = 0.40) (h3 : spent = price_percent * original_price) : original_price = 37500 :=
by
  sorry

end original_price_of_car_l670_670175


namespace balanced_path_divides_square_l670_670230

def balanced_path (n : ℕ) (path : Fin (2 * n + 1) → ℕ × ℕ) : Prop :=
  (path 0 = (0, 0)) ∧ 
  (path (Fin.last (2 * n)) = (n, n)) ∧
  (∀ i : Fin (2 * n), (path i.succ = (path i).1 + 1, (path i).2) ∨ (path i.succ = (path i).1, (path i).2 + 1)) ∧
  (∑ i in Finset.univ, (path i).1 = ∑ i in Finset.univ, (path i).2)

def area (p : Fin (2 * n + 1) → ℕ × ℕ) : ℤ :=
  ∑ i in Finset.range (2 * n), ((p i.succ).1 + (p i.succ).2 - (p i).1 - (p i).2) / 2

theorem balanced_path_divides_square (n : ℕ) (path : Fin (2 * n + 1) → ℕ × ℕ)
  (h_balanced : balanced_path n path) :
  area path = (n * n) / 2 :=
sorry

end balanced_path_divides_square_l670_670230


namespace profit_calculation_l670_670032

-- Define the initial conditions
def initial_cost_price : ℝ := 100
def initial_selling_price : ℝ := 200
def initial_sales_volume : ℝ := 100
def price_decrease_effect : ℝ := 4
def daily_profit_target : ℝ := 13600
def minimum_selling_price : ℝ := 150

-- Define the function relationship of daily sales volume with respect to x
def sales_volume (x : ℝ) : ℝ := initial_sales_volume + price_decrease_effect * x

-- Define the selling price
def selling_price (x : ℝ) : ℝ := initial_selling_price - x

-- Define the profit function
def profit (x : ℝ) : ℝ := (selling_price x - initial_cost_price) * sales_volume x

theorem profit_calculation (x : ℝ) (hx : selling_price x ≥ minimum_selling_price) :
  profit x = daily_profit_target ↔ selling_price x = 185 := by
  sorry

end profit_calculation_l670_670032


namespace find_EQ_length_l670_670537

theorem find_EQ_length (a b c d : ℕ) (parallel : Prop) (circle_tangent : Prop) :
  a = 105 ∧ b = 45 ∧ c = 21 ∧ d = 80 ∧ parallel ∧ circle_tangent → (∃ x : ℚ, x = 336 / 5) :=
by
  sorry

end find_EQ_length_l670_670537


namespace cylindrical_to_cartesian_coordinates_l670_670323

theorem cylindrical_to_cartesian_coordinates :
  ∃ (x y z : ℝ), (∃ (r θ : ℝ), r = 2 ∧ θ = (5/6)*Real.pi ∧ z = -1 ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ) ∧
  x = -Real.sqrt 3 ∧ y = 1 ∧ z = -1 :=
by {
  -- By the provided problem, we need to prove the existence of the Cartesian coordinates given the cylindrical ones
  use [-Real.sqrt 3, 1, -1],
  split,
  use [2, (5/6)*Real.pi],
  simp,
  split,
  exact Real.two_cos_pi_mul_div_six,
  split,
  exact Real.two_sin_pi_mul_div_six,
  refl,
  split,
  ring,
  ring,
  refl,
}

end cylindrical_to_cartesian_coordinates_l670_670323


namespace original_number_l670_670417

theorem original_number (n : ℕ) (h : (2 * (n + 2) - 2) / 2 = 7) : n = 6 := by
  sorry

end original_number_l670_670417


namespace AB_not_together_correct_l670_670161

-- Definitions based on conditions
def total_people : ℕ := 5

-- The result from the complementary counting principle
def total_arrangements : ℕ := 120
def AB_together_arrangements : ℕ := 48

-- The arrangement count of A and B not next to each other
def AB_not_together_arrangements : ℕ := total_arrangements - AB_together_arrangements

theorem AB_not_together_correct : 
  AB_not_together_arrangements = 72 :=
sorry

end AB_not_together_correct_l670_670161


namespace probability_of_gift_exchange_l670_670307

-- Definitions based on conditions
def boys : ℕ := 4
def girls : ℕ := 4
def total_people : ℕ := boys + girls

def total_ways_to_write_names : ℕ := (total_people^total_people)

-- Assume the conditions
def each_receives_one_gift (s : Set (order total_people)) : Prop := 
  ∀ (p : order total_people), ∃! (q : order total_people), s ⟨p, q⟩

def no_two_people_exchange (s : Set (order total_people)) : Prop := 
  ∀ (a b : order total_people), s ⟨a, b⟩ → ¬ (s ⟨b, a⟩)

def valid_configurations (s : Set (order total_people)) : Prop :=
  each_receives_one_gift s ∧ no_two_people_exchange s

def total_valid_configurations : ℕ := sorry -- To be filled in with the correct count

def probability : ℚ := total_valid_configurations / total_ways_to_write_names.to_rat

theorem probability_of_gift_exchange : probability = 9 / 2048 := 
  begin
    sorry -- Proof goes here
  end

end probability_of_gift_exchange_l670_670307


namespace sum_odd_minus_even_l670_670958

theorem sum_odd_minus_even :
  (∑ e in ((Finset.Icc 1 10).product (Finset.Icc 1 10)).filter (λ ij, ij.1 < ij.2 ∧ (ij.1 + ij.2) % 2 = 1), ij.1 + ij.2) -
  (∑ e in ((Finset.Icc 1 10).product (Finset.Icc 1 10)).filter (λ ij, ij.1 < ij.2 ∧ (ij.1 + ij.2) % 2 = 0), ij.1 + ij.2) = 55 :=
by
  sorry

end sum_odd_minus_even_l670_670958


namespace quadratic_has_one_real_root_l670_670274

theorem quadratic_has_one_real_root (k : ℝ) : 
  (∃ (x : ℝ), -2 * x^2 + 8 * x + k = 0 ∧ ∀ y, -2 * y^2 + 8 * y + k = 0 → y = x) ↔ k = -8 := 
by
  sorry

end quadratic_has_one_real_root_l670_670274


namespace solve_w_from_system_of_equations_l670_670908

open Real

variables (w x y z : ℝ)

theorem solve_w_from_system_of_equations
  (h1 : 2 * w + x + y + z = 1)
  (h2 : w + 2 * x + y + z = 2)
  (h3 : w + x + 2 * y + z = 2)
  (h4 : w + x + y + 2 * z = 1) :
  w = -1 / 5 :=
by
  sorry

end solve_w_from_system_of_equations_l670_670908


namespace cricket_game_initial_overs_l670_670825

def initial_overs (run_rate_initial run_rate_remaining : ℝ) (remaining_overs target_runs : ℝ) : ℝ :=
  let y := run_rate_initial * x
  let z := run_rate_remaining * remaining_overs
  x

theorem cricket_game_initial_overs :
  ∀ (x : ℝ), (4.2 * x) + (8.42 * 30) = 282 → x = 7 := by
  intro x
  intro h
  sorry

end cricket_game_initial_overs_l670_670825


namespace find_original_number_l670_670416

theorem find_original_number
  (n : ℤ)
  (h : (2 * (n + 2) - 2) / 2 = 7) :
  n = 6 := 
sorry

end find_original_number_l670_670416


namespace problem1_problem2_l670_670654

theorem problem1 (x y : ℝ) (hx : x = real.sqrt 27) (hy : y = real.sqrt 3) :
  real.sqrt 27 + real.sqrt 3 = 4 * real.sqrt 3 :=
by {
  rw [hx, real.sqrt_mul, real.sqrt_mul],
  linarith,
  all_goals {real.norm_num}
}

theorem problem2 (a b : ℝ) (ha : a = real.sqrt 2) (hb : b = 1) :
  (real.sqrt 2 + 1) * (real.sqrt 2 - 1) = 1 :=
by {
  rw [ha, hb, mul_sub],
  norm_num,
  rw [pow_two, pow_two],
  linarith
}

end problem1_problem2_l670_670654


namespace intersection_complement_A_B_l670_670341

-- Definitions of the universal set U, and sets A and B.
def U := set ℝ
def A := {x : ℝ | x ≤ -2}
def B := {x : ℝ | x < 1}

-- Definition of the complement of A with respect to U.
-- Here U is implicitly the universal set of ℝ.
def complement_A := {x : ℝ | x > -2}

-- The statement we want to prove
theorem intersection_complement_A_B : 
  (complement_A ∩ B) = {x : ℝ | -2 < x ∧ x < 1} :=
by 
  sorry

end intersection_complement_A_B_l670_670341


namespace transform_equation_l670_670250

theorem transform_equation (x : ℝ) :
  x^2 + 4 * x + 1 = 0 → (x + 2)^2 = 3 :=
by
  intro h
  sorry

end transform_equation_l670_670250


namespace area_of_triangle_ABC_l670_670950

structure Point := (x y : ℝ)

def A := Point.mk 2 3
def B := Point.mk 9 3
def C := Point.mk 4 12

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * ((B.x - A.x) * (C.y - A.y))

theorem area_of_triangle_ABC :
  area_of_triangle A B C = 31.5 :=
by
  -- Proof is omitted
  sorry

end area_of_triangle_ABC_l670_670950


namespace prove_m_add_n_l670_670361

-- Definitions from conditions
variables (m n : ℕ)

def condition1 : Prop := m + 1 = 3
def condition2 : Prop := m = n - 1

-- Statement to prove
theorem prove_m_add_n (h1 : condition1 m) (h2 : condition2 m n) : m + n = 5 := 
sorry

end prove_m_add_n_l670_670361


namespace circle_line_tangent_l670_670514

theorem circle_line_tangent (θ : ℝ) :
  let center := (0 : ℝ, 0 : ℝ)
  let radius := 1
  let line : ℝ → ℝ → ℝ := λ x y, x * Real.cos θ + y * Real.sin θ - 1
  let distance := (Real.abs 1) / (Real.sqrt (Real.cos θ ^ 2 + Real.sin θ ^ 2))
  distance = 1 := by
    sorry

end circle_line_tangent_l670_670514


namespace largest_increase_in_1993_l670_670150

def profitMargins : List ℕ := [10, 20, 30, 60, 70, 75, 80, 82, 86, 70]

def yearOfLargestIncrease (margins : List ℕ) : ℕ :=
  let differences := List.zipWith (λ x y => y - x) margins (List.tail margins)
  let maxDiffIndex := List.maxArg differences
  maxDiffIndex + 1990 + 1

theorem largest_increase_in_1993 : yearOfLargestIncrease profitMargins = 1993 :=
sorry

end largest_increase_in_1993_l670_670150


namespace pascals_triangle_ratio_l670_670379

-- Define Pascal's triangle
def pascal (n k : ℕ) : ℕ :=
  if k = 0 ∨ k = n then 1 else pascal (n - 1) (k - 1) + pascal (n - 1) k

-- Function to count 1s in the first n rows of Pascal's triangle
def count_ones (n : ℕ) : ℕ :=
  2 * n - 1

-- Function to count non-1s in the first n rows of Pascal's triangle
def count_non_ones (n : ℕ) : ℕ :=
  (n - 2) * (n - 1) / 2

-- Ratio of non-1s to 1s in Pascal's triangle
def ratio_non_ones_to_ones (n : ℕ) : ℚ :=
  (count_non_ones n : ℚ) / (count_ones n : ℚ)

-- The proof statement
theorem pascals_triangle_ratio (n : ℕ) (h : n ≥ 2) :
  ratio_non_ones_to_ones n = (n^2 - 3*n + 2) / (4*n - 2) :=
by
  sorry

end pascals_triangle_ratio_l670_670379


namespace cube_edge_length_l670_670134

-- Define the edge length 'a'
variable (a : ℝ)

-- Given conditions: 6a^2 = 24
theorem cube_edge_length (h : 6 * a^2 = 24) : a = 2 :=
by {
  -- The actual proof would go here, but we use sorry to skip it as per instructions.
  sorry
}

end cube_edge_length_l670_670134


namespace num_perfect_squares_l670_670796

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l670_670796


namespace exists_consecutive_natural_numbers_satisfy_equation_l670_670261

theorem exists_consecutive_natural_numbers_satisfy_equation :
  ∃ (n a b c d: ℕ), a = n ∧ b = n+2 ∧ c = n-1 ∧ d = n+1 ∧ n>0 ∧ a * b - c * d = 11 :=
by
  sorry

end exists_consecutive_natural_numbers_satisfy_equation_l670_670261


namespace exists_three_diff_numbers_equal_sum_l670_670067

def digit_sum (n : ℕ) : ℕ :=
  let digits := (n.digits 10).map (λ x => x + 1 - 1) in
  digits.foldr (λ d acc => d + acc) 0

theorem exists_three_diff_numbers_equal_sum : ∃ (m n p : ℕ), m ≠ n ∧ n ≠ p ∧ p ≠ m ∧ (m + digit_sum m = n + digit_sum n ∧ n + digit_sum n = p + digit_sum p) :=
by
  -- Example values from the solution
  let m := 9999999999999
  let n := 10000000000098
  let p := 10000000000107

  -- Digit sums
  let Sm := digit_sum m
  let Sn := digit_sum n
  let Sp := digit_sum p

  -- Check sums
  have h0 : m ≠ n := by sorry
  have h1 : n ≠ p := by sorry
  have h2 : p ≠ m := by sorry
  have h3 : m + Sm = n + Sn := by sorry
  have h4 : n + Sn = p + Sp := by sorry
  
  exact ⟨m, n, p, h0, h1, h2, ⟨h3, h4⟩⟩

end exists_three_diff_numbers_equal_sum_l670_670067


namespace mark_charged_more_hours_l670_670579

variable {p k m : ℕ}

theorem mark_charged_more_hours (h1 : p + k + m = 216)
                                (h2 : p = 2 * k)
                                (h3 : p = m / 3) :
                                m - k = 120 :=
sorry

end mark_charged_more_hours_l670_670579


namespace least_positive_difference_l670_670112

def geom_seq (a₀ r : ℕ) := {n : ℕ | ∃ k, n = a₀ * r ^ k}
def arith_seq (b₀ d : ℕ) := {m : ℕ | ∃ k, m = b₀ + k * d}

theorem least_positive_difference 
  (S₁ : Set ℕ) (S₂ : Set ℕ)
  (h1 : S₁ = geom_seq 3 2)
  (h2 : ∀ n ∈ S₁, n ≤ 400)
  (h3 : S₂ = arith_seq 30 30)
  (h4 : ∀ m ∈ S₂, m ≤ 400) :
  ∃ d : ℕ, (∀ a ∈ S₁, ∀ b ∈ S₂, a ≠ b → d = abs (a - b))
  ∧ d = 6 := sorry

end least_positive_difference_l670_670112


namespace cot_ratios_triangle_l670_670406

theorem cot_ratios_triangle
  (a b c : ℝ)
  (h : 9 * a ^ 2 + 9 * b ^ 2 - 19 * c ^ 2 = 0) :
  ∃ cotA cotB cotC : ℝ, 
    cotC / (cotA + cotB) = 5 / 9 :=
begin
  sorry
end

end cot_ratios_triangle_l670_670406


namespace function_extreme_value_range_l670_670335

theorem function_extreme_value_range (a : ℝ) : 
  (∀ f : ℝ → ℝ, f = (λ x, a * x^3 - 2 * x^2 + 4 * x - 7) → 
  (∃ x_max x_min : ℝ, is_max_on f {x | true} x_max ∧ is_min_on f {x | true} x_min)) ↔ 
  (a < 1 / 3 ∧ a ≠ 0) :=
sorry

end function_extreme_value_range_l670_670335


namespace parabola_hyperbola_tangent_l670_670670

noncomputable def parabola : ℝ → ℝ := λ x => x^2 + 5

noncomputable def hyperbola (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y => y^2 - m * x^2 = 1

theorem parabola_hyperbola_tangent (m : ℝ) : 
  (m = 10 + 4*Real.sqrt 6 ∨ m = 10 - 4*Real.sqrt 6) →
  ∃ x y, parabola x = y ∧ hyperbola m x y ∧ 
    ∃ c b a, a * y^2 + b * y + c = 0 ∧ a = 1 ∧ c = 5 * m - 1 ∧ b = -m ∧ b^2 - 4*a*c = 0 :=
by
  sorry

end parabola_hyperbola_tangent_l670_670670


namespace basketball_club_lineup_ways_l670_670215

theorem basketball_club_lineup_ways :
  ∃ (n : ℕ), n = 15 * 14 * 13 * 12 * 11 * 10 ∧ n = 3276000 :=
by
  have lineup_ways : 15 * 14 * 13 * 12 * 11 * 10 = 3276000 := by rfl
  use 15 * 14 * 13 * 12 * 11 * 10
  exact ⟨rfl, lineup_ways⟩

end basketball_club_lineup_ways_l670_670215


namespace marco_strawberries_l670_670451

def total_strawberries : ℕ := 23
def dads_strawberries : ℕ := 9

theorem marco_strawberries (T : ℕ) (W_dad : ℕ) (h1 : T = 23) (h2 : W_dad = 9) : (T - W_dad = 14) :=
by
  rw [h1, h2]
  exact rfl

end marco_strawberries_l670_670451


namespace find_x0_l670_670822

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem find_x0 :
  ∃ x0 : ℝ, (f x0 + (Real.log x0 + 1) = 1) ∧ x0 = 1 :=
by
  use 1
  unfold f
  apply And.intro
  {
    -- Proof of f 1 + (Real.log 1 + 1) = 1
    calc
      f 1 + (Real.log 1 + 1)
          = 1 * Real.log 1 + (Real.log 1 + 1) : by rfl
      ... = 0 + (0 + 1) : by rw Real.log_one
      ... = 1 : by ring
  }
  {
    -- Proof that x0 = 1
    refl
  }


end find_x0_l670_670822


namespace highest_power_of_14_dividing_40_factorial_l670_670183

theorem highest_power_of_14_dividing_40_factorial : 
  ∃ (m : ℕ), (14 ^ m ∣ nat.factorial 40) ∧ ∀ n, (14 ^ n ∣ nat.factorial 40) → n ≤ 5 := 
sorry

end highest_power_of_14_dividing_40_factorial_l670_670183


namespace arithmetic_sequence_max_value_l670_670522

theorem arithmetic_sequence_max_value (a : ℕ → ℝ) (d : ℝ) 
  (h_pos : ∀ n, 0 < a n) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_eq : a 2 + 2 * a 5 = 6) : 
  ∃ n, (a 3) * (a 5) = 4 := 
by 
  sorry

end arithmetic_sequence_max_value_l670_670522


namespace sum_of_b_j_equals_a0_b0_l670_670742

theorem sum_of_b_j_equals_a0_b0 (a_0 b_0 : ℕ) (a : ℕ → ℕ) (b : ℕ → ℕ) (p : ℕ)
  (h_a₀ : a 0 = a_0)
  (h_b₀ : b 0 = b_0)
  (ha_rec : ∀ k, a (k + 1) = a k / 2)
  (hb_rec : ∀ k, b (k + 1) = 2 * b k)
  (h_ap : a p = 1) :
  ∑ j in Nat.range (p + 1), if a j % 2 = 1 then b j else 0 = a_0 * b_0 :=
sorry

end sum_of_b_j_equals_a0_b0_l670_670742


namespace triangle_inequality_root_l670_670023

theorem triangle_inequality_root {a b c : ℝ} (n : ℕ) (hn : n ≥ 2) 
  (habc1 : a + b > c) (habc2 : b + c > a) (habc3 : a + c > b) :
  (Real.root n a + Real.root n b > Real.root n c) ∧
  (Real.root n b + Real.root n c > Real.root n a) ∧
  (Real.root n a + Real.root n c > Real.root n b) := 
  sorry

end triangle_inequality_root_l670_670023


namespace sin_C_value_l670_670840

theorem sin_C_value (A B C : ℝ) (a b c : ℝ) 
  (h_a : a = 1) 
  (h_b : b = 1/2) 
  (h_cos_A : Real.cos A = (Real.sqrt 3) / 2) 
  (h_angles : A + B + C = Real.pi) 
  (h_sides : Real.sin A / a = Real.sin B / b) :
  Real.sin C = (Real.sqrt 15 + Real.sqrt 3) / 8 :=
by 
  sorry

end sin_C_value_l670_670840


namespace incorrect_statement_is_C_l670_670569

theorem incorrect_statement_is_C : 
  (∀ x : ℝ, x = 0.09 → (∃ y : ℝ, y^2 = x ∧ (y = 0.3 ∨ y = -0.3))) ∧
  (√(1/9) = 1/3) ∧
  (∀ x : ℝ, x = 0 → (∃ y : ℝ, y^3 = x ∧ y = 0))  →
  ¬ (∃ y : ℝ, (y^3 = 1) ∧ (y = 1 ∨ y = -1)) :=
begin
  -- the proof will go here
  sorry
end

end incorrect_statement_is_C_l670_670569


namespace exists_three_cycle_l670_670244

variable {α : Type}

def tournament (P : α → α → Prop) : Prop :=
  (∃ (participants : List α), participants.length ≥ 3) ∧
  (∀ x y, x ≠ y → P x y ∨ P y x) ∧
  (∀ x, ∃ y, P x y)

theorem exists_three_cycle {α : Type} (P : α → α → Prop) :
  tournament P → ∃ A B C, P A B ∧ P B C ∧ P C A :=
by
  sorry

end exists_three_cycle_l670_670244


namespace slips_with_2_is_9_l670_670490

-- Definitions for the conditions
def total_slips : ℕ := 12
def expected_value : ℝ := 3.25
def number_of_slips_with_2 (x : ℕ) : ℝ := (x / total_slips) * 2
def number_of_slips_with_7 (x : ℕ) : ℝ := ((total_slips - x) / total_slips) * 7
def expected_value_formula (x : ℕ) : ℝ := number_of_slips_with_2 x + number_of_slips_with_7 x

-- Theorem to prove the number of slips with 2 is 9
theorem slips_with_2_is_9 (x : ℕ) (h : expected_value_formula x = expected_value) : x = 9 :=
by 
  sorry

end slips_with_2_is_9_l670_670490


namespace range_of_a_l670_670324

noncomputable def f (a x : ℝ) : ℝ :=
if x ≤ 1 then Real.exp x - a * x^2 else 2 * a + Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ ≤ x₂ → f a x₁ ≤ f a x₂) ↔ (a ∈ set.Icc (Real.exp 1 / 3) (Real.exp 1 / 2)) :=
by {
  sorry
}

end range_of_a_l670_670324


namespace derivative_product_at_1_l670_670393

variable {𝕜 : Type*} [NormedField 𝕜] [NormedSpace ℝ 𝕜]
variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]

variable (f g : 𝕜 → E)
variable (hf : Differentiable 𝕜 f)
variable (hg : Differentiable 𝕜 g)
variable (hf_val : f 1 = -1)
variable (hf'_val : deriv f 1 = 2)
variable (hg_val : g 1 = -2)
variable (hg'_val : deriv g 1 = 1)

theorem derivative_product_at_1 :
  deriv (fun x => f x * g x) 1 = -5 :=
by
  sorry

end derivative_product_at_1_l670_670393


namespace range_of_a_for_inequality_l670_670759

def f (a x : ℝ) := a * Real.log x + (1 - a) / 2 * x^2 - x

theorem range_of_a_for_inequality {a : ℝ} (h : 0 < a ∧ a < 1) : 
  (∃ (x : ℝ), 1 ≤ x ∧ f a x < a / (a - 1)) ↔ 0 < a ∧ a < Real.sqrt 2 - 1 := by
  sorry

end range_of_a_for_inequality_l670_670759


namespace area_CA_l670_670841

noncomputable theory

variables {A B C O A' B' : Type}
variables [nonempty A] [nonempty B] [nonempty C] [nonempty O] [nonempty A'] [nonempty B']
variables (S_Triangle : A → B → C → ℝ)
variables (AA' BB' : O → Prop)
variables (OnSegmentBC : A' → B → C → Prop)
variables (OnSegmentAC : B' → A → C → Prop)

def triangle_area (x y z : Type) [nonempty x] [nonempty y] [nonempty z] := ℝ

axiom condition1 : AA' A O
axiom condition2 : BB' B O
axiom condition3 : OnSegmentBC A' B C
axiom condition4 : OnSegmentAC B' A C
axiom area_AOB' : triangle_area A O B' = 3
axiom area_OAB : triangle_area O A B = 2
axiom area_OA'B : triangle_area O A' B = 1

theorem area_CA'B' : triangle_area C A' B' = 15.5 :=
sorry

end area_CA_l670_670841


namespace range_of_a_l670_670334

noncomputable def f (x a : ℝ) : ℝ := (Real.sin x)^2 + a * Real.cos x + a

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x a ≤ 1) → a ≤ 0 :=
by
  sorry

end range_of_a_l670_670334


namespace largest_angle_in_hexagon_l670_670494

theorem largest_angle_in_hexagon :
  ∀ (x : ℝ), (2 * x + 3 * x + 3 * x + 4 * x + 4 * x + 5 * x = 720) →
  5 * x = 1200 / 7 :=
by
  intros x h
  sorry

end largest_angle_in_hexagon_l670_670494


namespace find_a_l670_670760

noncomputable def f (a x : ℝ) : ℝ := a * x^3 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * a * x^2 + 1

-- Main theorem statement
theorem find_a (a : ℝ) :
  let f1 := f a 1 in
  let f'1 := f' a 1 in
  let tangent_line_passing := (λ x, f1 + f'1 * (x - 1)) in
  tangent_line_passing 2 = 7 → a = 1 :=
by
  intros h
  sorry

end find_a_l670_670760


namespace finite_uncross_operations_l670_670547

-- Definitions for the segments and operations
structure Point :=
(x : ℝ)
(y : ℝ)

def distance (A B : Point) : ℝ :=
real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

def segments_intersect (A B C D : Point) : Prop :=
-- Precise mathematical definition of intersection goes here
sorry

-- Main theorem to prove
theorem finite_uncross_operations (P : set Point) (S : set (Point × Point))
  (h_no_three_collinear : ∀ A B C : Point, A ∈ P → B ∈ P → C ∈ P → (segments_collinear A B C → (A = B ∨ B = C ∨ C = A)))
  (h_intersecting_segments : ∀ (A B C D : Point), (A, B) ∈ S → (C, D) ∈ S → segments_intersect A B C D) : 
  ∃ N : ℕ, ∀ k : ℕ, k > N → ¬ (∃ (A B C D : Point), (A, B), (C, D) ∈ S ∧ segments_intersect A B C D ∧ distance A C + distance B D < distance A B + distance C D) :=
sorry

end finite_uncross_operations_l670_670547


namespace isosceles_triangle_perimeter_l670_670930

def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 2) (h3 : a > 0) (h4 : b > 0) (h5 : a ≠ b) :
  let c := a,
      d := b + a + a in
  is_triangle b a a → d = 12 :=
by
  -- Place the proof here
  sorry

end isosceles_triangle_perimeter_l670_670930


namespace model_tower_height_l670_670130

-- Conditions
def new_tower_volume := 50000 -- in liters
def model_sphere_volume := 0.05 -- in liters
def new_tower_height := 60 -- in meters

-- Question and answer translation
theorem model_tower_height :
  let volume_ratio := new_tower_volume / model_sphere_volume in
  let scale_factor := Real.cbrt volume_ratio in
  new_tower_height / scale_factor = 0.6 :=
by
  sorry

end model_tower_height_l670_670130


namespace Granger_payment_correct_l670_670780

noncomputable def Granger_total_payment : ℝ :=
  let spam_per_can := 3.0
  let peanut_butter_per_jar := 5.0
  let bread_per_loaf := 2.0
  let spam_quantity := 12
  let peanut_butter_quantity := 3
  let bread_quantity := 4
  let spam_dis := 0.1
  let peanut_butter_tax := 0.05
  let spam_cost := spam_quantity * spam_per_can
  let peanut_butter_cost := peanut_butter_quantity * peanut_butter_per_jar
  let bread_cost := bread_quantity * bread_per_loaf
  let spam_discount := spam_dis * spam_cost
  let peanut_butter_tax_amount := peanut_butter_tax * peanut_butter_cost
  let spam_final_cost := spam_cost - spam_discount
  let peanut_butter_final_cost := peanut_butter_cost + peanut_butter_tax_amount
  let total := spam_final_cost + peanut_butter_final_cost + bread_cost
  total

theorem Granger_payment_correct :
  Granger_total_payment = 56.15 :=
by
  sorry

end Granger_payment_correct_l670_670780


namespace integral_abs_x_minus_two_l670_670653

theorem integral_abs_x_minus_two : ∫ x in (0:ℝ)..4, |x - 2| = 4 := 
by
  sorry

end integral_abs_x_minus_two_l670_670653


namespace formation_count_correct_l670_670971

-- Definition of the conditions given in the problem
def num_musicians : ℕ := 240
def valid_divisors (n : ℕ) : Prop := n ≥ 8 ∧ n ≤ 30
def formation_count (musicians : ℕ) : ℕ :=
  (finset.filter valid_divisors (finset.divisors musicians)).card

-- The theorem to prove the number of different rectangular formations
theorem formation_count_correct : formation_count num_musicians = 8 :=
by
  sorry

end formation_count_correct_l670_670971


namespace crows_trees_system_l670_670895

-- Let's define the given conditions
variables (x y : ℕ)

-- The system of equations translation
theorem crows_trees_system (h1 : 3 * y + 5 = x) (h2 : 5 * (y - 1) = x) :
    ∃ x y, 3 * y + 5 = x ∧ 5 * (y - 1) = x :=
by {
  use [x, y],
  exact ⟨h1, h2⟩,
}

end crows_trees_system_l670_670895


namespace flower_nectar_water_content_l670_670193

noncomputable def percentage_of_water_in_flower_nectar (P_h : ℝ) (W_nectar : ℝ) (W_honey : ℝ) : ℝ :=
  (P_h * W_honey) / W_nectar * 100

theorem flower_nectar_water_content : 
  ∀ (W_nectar W_honey : ℝ) (P_h : ℝ), 
    W_nectar = 1.4 → 
    W_honey = 1 → 
    P_h = 0.30 →
    percentage_of_water_in_flower_nectar P_h W_nectar W_honey ≈ 21.43 :=
by
  intros
  rw [percentage_of_water_in_flower_nectar]
  norm_num
  sorry

end flower_nectar_water_content_l670_670193


namespace touch_to_all_on_state_l670_670028

namespace ArrayTransform

def state := bool
def ButtonArray (m n : ℕ) := Array (Array state)

def toggle (arr : ButtonArray 40 50) (i j : ℕ) : ButtonArray 40 50 := sorry

theorem touch_to_all_on_state : ∃ (touches_needed : ℕ), touches_needed = 2000 ∧
  ∀ (initial_state : ButtonArray 40 50),
  (∀ i j, initial_state[i][j] = false) →
  let final_state := foldl (λ arr p => toggle arr p.1 p.2) initial_state [(i, j) | i < 40, j < 50] in
  (∀ i j, final_state[i][j] = true) :=
sorry
end ArrayTransform

end touch_to_all_on_state_l670_670028


namespace pyarelal_loss_l670_670201

theorem pyarelal_loss (A P : ℝ) (totalLoss : ℝ) (hA : A = (1/9) * P) (hTotalLoss : totalLoss = 1200) : 
  let ratio := 1 + 9 in
  let pyarelalLoss := (9 / ratio) * totalLoss in
  pyarelalLoss = 1080 := 
by
  sorry

end pyarelal_loss_l670_670201


namespace solve_base_6_addition_l670_670483

variables (X Y k : ℕ)

theorem solve_base_6_addition (h1 : Y + 3 = X) (h2 : ∃ k, X + 5 = 2 + 6 * k) : X + Y = 3 :=
sorry

end solve_base_6_addition_l670_670483


namespace experiment_matches_frequencies_l670_670191

def frequencies : List (ℕ × ℝ) :=
[(100, 0.60), (200, 0.30), (300, 0.50), (400, 0.36), (500, 0.42), 
 (600, 0.38), (700, 0.41), (800, 0.39), (900, 0.40), (1000, 0.40)]

def experiment_C_prob : ℝ := 2 / 5

theorem experiment_matches_frequencies : 
  ∃ (n: ℕ) (freq: ℝ), 
  (n, freq) ∈ frequencies → freq = experiment_C_prob :=
sorry

end experiment_matches_frequencies_l670_670191


namespace inequality_proof_l670_670434

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hab : a + b < 2) : 
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a * b)) ∧ 
  (1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a * b) ↔ 0 < a ∧ a = b ∧ a < 1) := 
by 
  sorry

end inequality_proof_l670_670434


namespace number_of_quadruples_l670_670063

theorem number_of_quadruples (p n : ℕ) (hp : p.prime) (hn : 0 < n) :
    let S := Finset.range (p^n)
    (Finset.card {quadruples ∈ S.product S.product S | 
        ∃ (a1 a2 a3 a4 : ℕ), a1 ∈ S ∧ a2 ∈ S ∧ a3 ∈ S ∧ a4 ∈ S ∧ 
        a1 * a2 + a3 * a4 ≡ -1 [MOD p^n]} : ℕ) = p^(3*n) :=
sorry

end number_of_quadruples_l670_670063


namespace min_max_value_is_zero_l670_670685

def max_at_x (x : ℝ) (y : ℝ) : ℝ := |x^2 - 2 * x * y|

theorem min_max_value_is_zero :
  ∃ y ∈ set.univ, min (set.univ) (λ y, real.sup (set.Icc 0 2) (λ x, max_at_x x y)) = 0 :=
sorry

end min_max_value_is_zero_l670_670685


namespace ship_length_in_emilys_steps_l670_670678

variable (L E S : ℝ)

-- Conditions from the problem:
variable (cond1 : 240 * E = L + 240 * S)
variable (cond2 : 60 * E = L - 60 * S)

-- Theorem to prove:
theorem ship_length_in_emilys_steps (cond1 : 240 * E = L + 240 * S) (cond2 : 60 * E = L - 60 * S) : 
  L = 96 * E := 
sorry

end ship_length_in_emilys_steps_l670_670678


namespace rodney_lift_l670_670897

theorem rodney_lift :
  ∃ (Ry : ℕ), 
  (∃ (Re R Ro : ℕ), 
  Re + Ry + R + Ro = 450 ∧
  Ry = 2 * R ∧
  R = Ro + 5 ∧
  Re = 3 * Ro - 20 ∧
  20 ≤ Ry ∧ Ry ≤ 200 ∧
  20 ≤ R ∧ R ≤ 200 ∧
  20 ≤ Ro ∧ Ro ≤ 200 ∧
  20 ≤ Re ∧ Re ≤ 200) ∧
  Ry = 140 :=
by
  sorry

end rodney_lift_l670_670897


namespace balls_in_boxes_l670_670351

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l670_670351


namespace unique_7_tuple_count_l670_670701

theorem unique_7_tuple_count :
  ∃! (x : ℕ → ℝ) (zero_le_x : (∀ i, 0 ≤ i → i ≤ 6 → true)),
  (2 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2 + (x 4 - x 5)^2 + (x 5 - x 6)^2 + x 6^2 = 1 / 8 :=
by
  sorry

end unique_7_tuple_count_l670_670701


namespace sum_of_M_l670_670516

theorem sum_of_M (x y z w M : ℕ) (hxw : w = x + y + z) (hM : M = x * y * z * w) (hM_cond : M = 12 * (x + y + z + w)) :
  ∃ sum_M, sum_M = 2208 :=
by 
  sorry

end sum_of_M_l670_670516


namespace probability_A_3_red_2_yellow_expectation_after_exchange_3_l670_670525

variable (r y: ℕ) -- r for red balls, y for yellow balls
variable (a b a_r: ℕ) -- a and b for boxes, a_r for red balls in A
variable (total: ℕ)

-- Conditions
def conditions (r y a b: ℕ) : Prop :=
  r = 5 ∧ y = 5 ∧ a = 5 ∧ b = 5

-- Part (1)
def probability_A_contains_3_red_2_yellow (r y a b a_r: ℕ) [conditions r y a b] : ℚ :=
  have total_ways := Nat.choose (r + y) a
  have red_ways := Nat.choose r 3
  have yellow_ways := Nat.choose y 2
  (red_ways * yellow_ways) / total_ways

theorem probability_A_3_red_2_yellow :
  probability_A_contains_3_red_2_yellow 5 5 5 5 3 = 25 / 63 := sorry

-- Part (2)
def expectation_after_exchange (r y a b a_r: ℕ) [conditions r y a b] : ℚ :=
  let p0 := (Nat.choose a_r 3 * Nat.choose b a  3) / (Nat.choose a 3 * Nat.choose b 3)
  let p1 := (Nat.choose a_r 2 * Nat.choose (a - a_r) 1 * Nat.choose b a 3 + Nat.choose a_r 3 * Nat.choose (a - a_r) 2 * Nat.choose b a 2) / (Nat.choose a 3 * Nat.choose b 3)
  let p2 := (Nat.choose a_r 1 * Nat.choose (a - a_r) 2 * Nat.choose b a 2 + Nat.choose a_r 2 * Nat.choose (a - a_r) 1 * Nat.choose b a 1 * Nat.choose b a 2 + Nat.choose a_r 3 * Nat.choose (a - a_r) 1) / (Nat.choose a 3 * Nat.choose b 3)
  let p3 := (Nat.choose a_r 1 * Nat.choose (a - a_r) 2 * Nat.choose (b - 1) 2 + Nat.choose a_r 2 * Nat.choose (a - a_r) 1 * Nat.choose (b - 2) 2) / (Nat.choose a 3 * Nat.choose b 3)
  let p4 := (Nat.choose a_r 1 * Nat.choose (a - a_r) 2) / (Nat.choose a 3 * Nat.choose b 3)
  0 * p0 + 1 * p1 + 2 * p2 + 3 * p3 + 4 * p4

theorem expectation_after_exchange_3 :
  expectation_after_exchange 5 5 5 5 3 = 12 / 5 := sorry

end probability_A_3_red_2_yellow_expectation_after_exchange_3_l670_670525


namespace sum_even_2_to_20_sum_odd_1_to_19_b_minus_a_l670_670371

def sum_even (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

def sum_odd (n : ℕ) : ℕ :=
  (n / 2) * (1 + n)

theorem sum_even_2_to_20 : sum_even 20 = 110 := by
  -- This is the sum of the even numbers from 2 to 20
  sorry

theorem sum_odd_1_to_19 : sum_odd 19 = 100 := by
  -- This is the sum of the odd numbers from 1 to 19
  sorry

theorem b_minus_a : 
  let a := sum_even 20
  let b := sum_odd 19
  b - a = -10 := by
    have ha : a = sum_even 20 := rfl
    have hb : b = sum_odd 19 := rfl
    rw [ha, hb]
    have e1 : sum_even 20 = 110 := by apply sum_even_2_to_20
    have e2 : sum_odd 19 = 100 := by apply sum_odd_1_to_19
    rw [e1, e2]
    norm_num

end sum_even_2_to_20_sum_odd_1_to_19_b_minus_a_l670_670371


namespace unpronounceable_7_letter_words_count_l670_670268

def alphabet := {A, B}

def is_unpronounceable (word : list alphabet) : Prop :=
  ∃ (w1 w2 w3 : alphabet) (w1_eq_w2 : w1 = w2) (w2_eq_w3 : w2 = w3), 
    list.chain (≠) word w1 ∧ list.chain (≠) word w2 ∧ list.chain (≠) word w3

def number_of_unpronounceable_7_letter_words : ℕ :=
  (list.replicate 7 alphabet).filter is_unpronounceable |>.length

theorem unpronounceable_7_letter_words_count :
  number_of_unpronounceable_7_letter_words = 86 := 
sorry

end unpronounceable_7_letter_words_count_l670_670268


namespace yunjeong_locker_problem_l670_670143

theorem yunjeong_locker_problem
  (l r f b : ℕ)
  (h_l : l = 7)
  (h_r : r = 13)
  (h_f : f = 8)
  (h_b : b = 14)
  (same_rows : ∀ pos1 pos2 : ℕ, pos1 = pos2) :
  (l - 1) + (r - 1) + (f - 1) + (b - 1) = 399 := sorry

end yunjeong_locker_problem_l670_670143


namespace total_goats_l670_670467

theorem total_goats (w_goats : ℕ) (h_washington : w_goats = 140) : 
  let p_goats := w_goats + 40 in
  p_goats + w_goats = 320 :=
by
  let p_goats := w_goats + 40
  have h_p_goats : p_goats = 180 := by sorry
  calc
    p_goats + w_goats 
    = 180 + 140 : by rw [h_p_goats, h_washington]
    _ = 320 : by rw [Nat.add_comm (140 : ℕ) 180]

end total_goats_l670_670467


namespace num_perfect_squares_l670_670795

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l670_670795


namespace rectangle_length_l670_670924

theorem rectangle_length (b : ℝ) (h1 : ∃ b, (∀ (length breadth : ℝ), length = 2 * breadth ∧ 
  (length - 5) * (breadth + 5) = 2 * breadth ^ 2 + 75)) : 
  ∃ l, l = 40 :=
begin
  sorry
end

end rectangle_length_l670_670924


namespace range_of_a_l670_670019

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - 2 * x - a

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 a = 0 ∧ f x2 a = 0) ↔ a > 2 - 2 * Real.log 2 :=
by
  sorry

end range_of_a_l670_670019


namespace spider_butterfly_result_l670_670977

namespace WebGame

def K : ℕ := sorry
def R : ℕ := sorry

theorem spider_butterfly_result :
  (K = 4 ∧ R = 7 → "Draw") ∧
  (K = 3 ∧ R = 7 → "Draw") ∧
  (K = 4 ∧ R = 10 → "Butterfly wins") ∧
  (K ≥ 2 ∧ R ≥ 3 → 
    ((K ≥ Nat.ceil (R / 2)) → "Draw") ∧
    ((K < Nat.ceil (R / 2)) → "Butterfly wins")) :=
sorry

end WebGame

end spider_butterfly_result_l670_670977


namespace determine_correct_path_l670_670532

variable (A B C : Type)
variable (truthful : A → Prop)
variable (whimsical : A → Prop)
variable (answers : A → Prop)
variable (path_correct : A → Prop)

-- Conditions
axiom two_truthful_one_whimsical (x y z : A) : (truthful x ∧ truthful y ∧ whimsical z) ∨ 
                                                (truthful x ∧ truthful z ∧ whimsical y) ∨ 
                                                (truthful y ∧ truthful z ∧ whimsical x)

axiom traveler_aware : ∀ x y : A, truthful x → ¬ truthful y
axiom siblings : A → B → C → Prop
axiom ask_sibling : A → B → C → Prop

-- Conditions formalized
axiom ask_about_truthfulness (x y : A) : answers x → (truthful y ↔ ¬truthful y)

theorem determine_correct_path (x y z : A) :
  (truthful x ∧ ¬truthful y ∧ path_correct x) ∨
  (¬truthful x ∧ truthful y ∧ path_correct y) ∨
  (¬truthful x ∧ ¬truthful y ∧ truthful z ∧ path_correct z) :=
sorry

end determine_correct_path_l670_670532


namespace any_triangle_divided_into_three_obtuse_triangles_l670_670102

noncomputable def incenter (A B C : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] : Type* := sorry

theorem any_triangle_divided_into_three_obtuse_triangles (A B C : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] : Prop :=
  ∃ (O : incenter A B C),
  (∠ A O B > π / 2) ∧ (∠ B O C > π / 2) ∧ (∠ C O A > π / 2)

end any_triangle_divided_into_three_obtuse_triangles_l670_670102


namespace sqrt_difference_l670_670557

theorem sqrt_difference:
  sqrt (49 + 81) - sqrt (36 - 9) = sqrt 130 - 3 * sqrt 3 :=
by
  sorry

end sqrt_difference_l670_670557


namespace clock_adjustment_l670_670998

theorem clock_adjustment :
  let start := ⟨15, 3, 2023, 13, 0⟩ in -- March 15, 2023, 1 P.M.
  let end := ⟨22, 3, 2023, 9, 0⟩ in -- March 22, 2023, 9 A.M.
  let days := (6 : ℕ) in -- From March 15 to March 21
  let hours := (20 : ℕ) in -- From March 21, 1 P.M. to March 22, 9 A.M.
  let per_day_loss := (3 : ℕ) in -- 3 minutes per day loss
  let total_loss := (days + 1) * per_day_loss in -- Total minutes lost (incl. partial days)
  ∃ correction: ℕ, correction = 21 :=
by
  sorry

end clock_adjustment_l670_670998


namespace value_of_f_x1_l670_670506

def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem value_of_f_x1 
  (x0 x1 : ℝ) 
  (h_root : f x0 = 0) 
  (h_pos : 0 < x1) 
  (h_ordering : x1 < x0) : 
  f x1 < 0 :=
sorry

end value_of_f_x1_l670_670506


namespace partition_diff_l670_670377

theorem partition_diff {A : Type} (S : Finset ℕ) (S_card : S.card = 67)
  (P : Finset (Finset ℕ)) (P_card : P.card = 4) :
  ∃ (U : Finset ℕ) (hU : U ∈ P), ∃ (a b c : ℕ) (ha : a ∈ U) (hb : b ∈ U) (hc : c ∈ U),
  a = b - c ∧ (1 ≤ a ∧ a ≤ 67) :=
by sorry

end partition_diff_l670_670377


namespace combined_chocolate_bars_l670_670602

theorem combined_chocolate_bars :
  let total_chocolate_bars := 60
  let number_of_people := 5
  let per_person_share := total_chocolate_bars / number_of_people
  let tom_to_sue := per_person_share / 2
  let mike_from_rita := 2
  let mike_initial := per_person_share
  let rita_initial := per_person_share
  let anita_initial := per_person_share in
  (mike_initial + mike_from_rita) + (rita_initial - mike_from_rita) + anita_initial = 36 :=
by
  sorry

end combined_chocolate_bars_l670_670602


namespace find_B_value_l670_670501

def divisible_by_9 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 9 * k

theorem find_B_value (B : ℕ) :
  divisible_by_9 (4 * 10^4 + B * 10^3 + B * 10^2 + 1 * 10 + 3) →
  0 ≤ B ∧ B ≤ 9 →
  B = 5 :=
sorry

end find_B_value_l670_670501


namespace exists_unique_real_root_l670_670300

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem exists_unique_real_root : ∃! x ∈ Ioo 1 2, f x = 0 :=
sorry

end exists_unique_real_root_l670_670300


namespace compare_abc_l670_670007

noncomputable def a : ℝ := 3 ^ Real.sqrt 2
noncomputable def b : ℝ := 3 ^ 0.3
noncomputable def c : ℝ := 0.9 ^ 3.1

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l670_670007


namespace find_y_intercept_l670_670154

theorem find_y_intercept (m : ℝ) (x₁ : ℝ) (y₁ : ℝ) (h_slope : m = 3) (h_x_intercept : x₁ = 4) (h_y_intercept : y₁ = 0) :
  (0, -12) = (0, y₁ - m*x₁) :=
by
  -- Slope-intercept form of the line given the specified slope and x-intercept.
  have eqn : ∀ x, y₁ + m * (x - x₁) = m * x + (y₁ - m * x₁),
  -- Substituting x = 0
  have y_int := y₁ - m * x₁,
  sorry

end find_y_intercept_l670_670154


namespace how_many_pages_l670_670845

theorem how_many_pages (cost_per_pages : ℤ) (cents_per_dollar : ℤ) (pages_per_cents : ℤ → ℤ → ℤ) : 
  pages_per_cents 4 7 * (30 * 100) / 1 ≥ 1714 :=
by
  assume h1 : cost_per_pages = 7
  assume h2 : cents_per_dollar = 100
  assume h3 : pages_per_cents 4 7 = 4 / 7
  have h4 : total_cents = 3000
    by sorry
  have h5 : total_pages = pages_per_cents 4 7 * total_cents
    by sorry
  show h5 ≥ 1714
    by sorry

end how_many_pages_l670_670845


namespace product_of_diagonals_l670_670875

variable {A B C D X : Type}
variable (circle : Set (EuclideanGeometry ℝ)) -- Assuming we are working in the Euclidean plane over real numbers
variable [Nonempty circle]
variables [Circle A] [Circle B] [Circle C] [Circle D]

-- Definition: A cyclic quadrilateral
def is_cyclic_quadrilateral (A B C D : circle) : Prop :=
  ∃ (γ : Set (Metric.Sphere ℝ )), γ = circle.annotation

-- Intersection of diagonals AC and BD
def lines_intersect (A C B D : circle) (X : Type) : Prop :=
  ∃ X, is_line (A, C) ∧ is_line (B, D)

-- Prove the result based on these definitions
theorem product_of_diagonals (h_cyclic: is_cyclic_quadrilateral A B C D) 
                             (h_intersect: lines_intersect A C B D X) :
  AX * CX = BX * DX :=
sorry

end product_of_diagonals_l670_670875


namespace arrange_in_ascending_order_l670_670863

noncomputable def a : ℝ := Real.log 7 / Real.log 3
noncomputable def b : ℝ := 2 ^ 3.3
noncomputable def c : ℝ := 0.8 ^ 3.3

theorem arrange_in_ascending_order : c < a ∧ a < b :=
by
  sorry

end arrange_in_ascending_order_l670_670863


namespace distance_between_lines_l670_670918

theorem distance_between_lines : 
  let L1 := (1:ℝ) * x + 2 * y - 1 = 0 in 
  let L2 := (2:ℝ) * x + 4 * y + 3 = 0 in
  ∃ d : ℝ, d = (|(-2) - 3|) / (real.sqrt (2^2 + 4^2)) ∧ d = (real.sqrt 5) / 2 :=
by {
  sorry
}

end distance_between_lines_l670_670918


namespace John_can_lift_now_l670_670849

def originalWeight : ℕ := 135
def trainingIncrease : ℕ := 265
def bracerIncreaseFactor : ℕ := 6

def newWeight : ℕ := originalWeight + trainingIncrease
def bracerIncrease : ℕ := newWeight * bracerIncreaseFactor
def totalWeight : ℕ := newWeight + bracerIncrease

theorem John_can_lift_now :
  totalWeight = 2800 :=
by
  -- proof steps go here
  sorry

end John_can_lift_now_l670_670849


namespace perfect_squares_between_100_and_400_l670_670788

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l670_670788


namespace tom_total_amount_l670_670056

-- Definitions of the initial conditions
def initial_amount : ℕ := 74
def amount_earned : ℕ := 86

-- Main statement to prove
theorem tom_total_amount : initial_amount + amount_earned = 160 := 
by
  -- sorry added to skip the proof
  sorry

end tom_total_amount_l670_670056


namespace correct_prop_is_4_l670_670920

-- Define the propositions as conditions
def prop1 : Prop := ∀ (L1 L2 : Line), ¬(∃ P : Point, P ∈ L1 ∧ P ∈ L2) → parallel L1 L2
def prop2 : Prop := ∀ (α β : Angle), (corresponding α β) → α = β
def prop3 : Prop := ∀ (a b : ℝ), (a^2 = b^2) → a = b
def prop4 : Prop := ∀ (L1 L2 : Line), ∀ P : Point, (P ∈ (L1 ∩ L2)) → vertical_angles P L1 L2

-- The main theorem: Proposition 4 is true
theorem correct_prop_is_4 : prop4 :=
sorry

end correct_prop_is_4_l670_670920


namespace trapezoid_problem_solution_l670_670462

noncomputable def greatestIntegerNotExceeding (x : ℝ) : ℕ :=
  ⌊x⌋.toNat

theorem trapezoid_problem_solution :
  ∀ (b h : ℝ), b = 37.5 → (h > 0) → 
  let x := 62.5 in
  greatestIntegerNotExceeding (x^2 / 50) = 78 := 
by
  intros b h hb h_pos x
  sorry

end trapezoid_problem_solution_l670_670462


namespace dorothy_interest_earned_l670_670492

theorem dorothy_interest_earned :
  let P := 2000
  let r := 0.02
  let n := 3
  let final_balance := P * (1 + r)^n 
  let rounded_balance := Int.round final_balance
  let interest_earned := rounded_balance - P
  interest_earned = 122 := by
  sorry

end dorothy_interest_earned_l670_670492


namespace sum_of_first_10_terms_arithmetic_sequence_l670_670048

theorem sum_of_first_10_terms_arithmetic_sequence :
  let a : ℕ → ℕ := λ n, 2 + 2 * (n - 1),
      S : ℕ → ℕ := λ n, n * (2 * 2 + (n - 1) * 2) / 2
  in S 10 = 110 :=
by
  sorry

end sum_of_first_10_terms_arithmetic_sequence_l670_670048


namespace xiaochun_age_l670_670531

theorem xiaochun_age
  (x y : ℕ)
  (h1 : x = y - 18)
  (h2 : 2 * (x + 3) = y + 3) :
  x = 15 :=
sorry

end xiaochun_age_l670_670531


namespace angle_between_tangents_theorem_l670_670129

noncomputable def angle_between_tangents : ℝ :=
  let O := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}
  let A := (3, Real.sqrt 7)
  in if (∃! t1 t2 : ℝ → ℝ, (∀ x, x ∈ O -> t1 x = 0) ∧ t1 A.1 A.2 = 0 ∧
                              (∀ x, x ∈ O -> t2 x = 0) ∧ t2 A.1 A.2 = 0)
     then Real.pi / 3 else 0

theorem angle_between_tangents_theorem :
  ∃! t1 t2 : ℝ → ℝ, (∀ x, x ∈ O -> t1 x = 0) ∧ t1 A.1 A.2 = 0 ∧
                     (∀ x, x ∈ O -> t2 x = 0) ∧ t2 A.1 A.2 = 0 →
  angle_between_tangents = Real.pi / 3 := sorry

end angle_between_tangents_theorem_l670_670129


namespace Amy_Bob_together_28_times_l670_670258

def players : Nat := 12

def set1 (p : Finset (Fin 12)) : Prop :=
  p.card = 6

def unique_matchups (games : Finset (Finset (Fin 12))) : Prop :=
  ∀ p₁ p₂ : Finset (Fin 12), set1 p₁ → set1 p₂ → p₁ ≠ p₂ → p₁ ∩ p₂ = ∅

def Chris_and_Dave_everywhere (games : Finset (Finset (Fin 12))) : Prop :=
  ∀ g ∈ games, {5, 6} ⊆ g

theorem Amy_Bob_together_28_times :
  ∀ (games : Finset (Finset (Fin 12))),
  unique_matchups games →
  Chris_and_Dave_everywhere games →
  (∃ count,
    count = games.filter (λ g, {0, 1} ⊆ g).card ∧
    count = 28) :=
by
  sorry

end Amy_Bob_together_28_times_l670_670258


namespace Mike_total_expenditure_l670_670304

theorem Mike_total_expenditure :
  let speakers := 118.54
  let tires := 106.33
  let window_tints := 85.27
  let maintenance := 199.75
  let steering_cover := 15.63
  speakers + tires + window_tints + maintenance + steering_cover = 525.52 :=
begin
  sorry,
end

end Mike_total_expenditure_l670_670304


namespace tank_filling_time_l670_670948

noncomputable def fill_time (R1 R2 R3 : ℚ) : ℚ :=
  1 / (R1 + R2 + R3)

theorem tank_filling_time :
  let R1 := 1 / 18
  let R2 := 1 / 30
  let R3 := -1 / 45
  fill_time R1 R2 R3 = 15 :=
by
  intros
  unfold fill_time
  sorry

end tank_filling_time_l670_670948


namespace probability_of_even_sum_and_same_number_l670_670169

noncomputable def probability_even_sum_same_number : ℚ :=
  let die_faces := {1, 2, 3, 4, 5, 6}
  let outcomes := { (d1, d2) | d1 ∈ die_faces ∧ d2 ∈ die_faces }
  let favorable_outcomes := { (d1, d2) | d1 = d2 ∧ (d1 + d2) % 2 = 0 }
  (favorable_outcomes.to_finset.card : ℚ) / (outcomes.to_finset.card : ℚ)

theorem probability_of_even_sum_and_same_number :
  probability_even_sum_same_number = 1 / 12 :=
by sorry

end probability_of_even_sum_and_same_number_l670_670169


namespace angle_between_b_and_c_l670_670734

variables {𝕜 : Type*} [IsROrC 𝕜]
open ComplexConjugate

noncomputable def magnitude (v : 𝕜 × 𝕜) : ℝ :=
Real.sqrt((v.1) * (v.1) + (v.2) * (v.2))

noncomputable def dot_product (u v : 𝕜 × 𝕜) : ℝ :=
(u.1 * v.1 + u.2 * v.2).re

variables (a b c : 𝕜 × 𝕜)
variables (theta : ℝ)

axiom a_cond : 2 • a - b = (-1, Real.sqrt 3)
axiom c_def : c = (1, Real.sqrt 3)
axiom dot_ac : dot_product a c = 3
axiom magn_b : magnitude b = 4

theorem angle_between_b_and_c : 
  dot_product b c = magnitude b * magnitude c * Real.cos theta → theta = Real.pi / 3 :=
by 
  sorry

end angle_between_b_and_c_l670_670734


namespace no_integer_roots_l670_670428

theorem no_integer_roots (P : ℤ[X]) (a b c : ℤ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a)
  (hPa : |P.eval a| = 1) (hPb : |P.eval b| = 1) (hPc : |P.eval c| = 1) : 
  ∀ r : ℤ, P.eval r ≠ 0 :=
by
  intro r
  sorry

end no_integer_roots_l670_670428


namespace middle_number_in_8th_row_l670_670457

-- Define a function that describes the number on the far right of the nth row.
def far_right_number (n : ℕ) : ℕ := n^2

-- Define a function that calculates the number of elements in the nth row.
def row_length (n : ℕ) : ℕ := 2 * n - 1

-- Define the middle number in the nth row.
def middle_number (n : ℕ) : ℕ := 
  let mid_index := (row_length n + 1) / 2
  far_right_number (n - 1) + mid_index

-- Statement to prove the middle number in the 8th row is 57
theorem middle_number_in_8th_row : middle_number 8 = 57 :=
by
  -- Placeholder for proof
  sorry

end middle_number_in_8th_row_l670_670457


namespace determine_k_l670_670273

noncomputable def linearly_dependent (u v : ℝ → ℝ) : Prop :=
  ∃ (c₁ c₂ : ℝ), (c₁ ≠ 0 ∨ c₂ ≠ 0) ∧ (c₁ • u + c₂ • v = 0)

theorem determine_k (k : ℝ) : 
  linearly_dependent (λ i, if i = 1 then 1 else if i = 2 then 4 else -1)
                     (λ i, if i = 1 then 2 else if i = 2 then k else 3) ↔ k = 8 := 
sorry

end determine_k_l670_670273


namespace ratio_of_areas_l670_670488

theorem ratio_of_areas (s : ℝ) :
  let A := (0 : ℝ, 0 : ℝ)
      B := (4*s, 0)
      E := (3*s, 0)
      F := (4*s, s)
      side_EFGH := (cmath.sqrt(2) * s)
      area_ABCD := (4*s) * (4*s)
      area_EFGH := side_EFGH * side_EFGH in
  area_EFGH / area_ABCD = 1 / 8 :=
by {
  sorry
}

end ratio_of_areas_l670_670488


namespace domain_of_f_l670_670182

def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 6))

theorem domain_of_f (x : ℝ) : x ≠ 4.5 ↔ x ∈ Set.Iio 4.5 ∪ Set.Ioi 4.5 := by
  sorry

end domain_of_f_l670_670182


namespace sqrt_subtraction_l670_670562

theorem sqrt_subtraction :
  (Real.sqrt (49 + 81)) - (Real.sqrt (36 - 9)) = (Real.sqrt 130) - (3 * Real.sqrt 3) :=
sorry

end sqrt_subtraction_l670_670562


namespace probability_of_hyperbola_chosen_from_set_E_l670_670527

noncomputable def probability_of_hyperbola_not_greater_eccentricity (a : ℝ) (c : ℝ := 1) (E₁_eccentricity : ℝ := 2) : ℝ :=
  if (1 < 1 / a ∧ 1 / a ≤ E₁_eccentricity) ∧ (0 < a ∧ a < c) then 1 / 2 else 0

theorem probability_of_hyperbola_chosen_from_set_E :
  let C := { (x, y) | (2 / 3) * x ^ 2 + 2 * y ^ 2 = 1 }
  ∃ P : ℝ × ℝ, P = (1, -3/2) →
  let E₁ := { (x, y) | x ∈ ℝ ∧ y ∈ ℝ } -- Assuming a dummy definition for E₁, replace with the actual equation if needed
  ∃ c : ℝ, c = 1 →
  let b := ℝ
  ∃ a : ℝ, (1 / a^2 - 9 / (4 * (1 - a^2))) = 1 ∧ a = 0.5 →
  let E₁_eccentricity := c / a
  E₁_eccentricity = 2 →
  let probability := probability_of_hyperbola_not_greater_eccentricity a c E₁_eccentricity
  probability = 1 / 2 :=
by {
  -- providing Lean with the necessary assumptions, later can be replaced with actual proof logic
  sorry
}

end probability_of_hyperbola_chosen_from_set_E_l670_670527


namespace necklace_cost_l670_670088

-- Define the conditions
constant total_sales : ℕ := 80
constant num_necklaces : ℕ := 4
constant num_rings : ℕ := 8
constant cost_per_ring : ℕ := 4

-- The statement to be proven
theorem necklace_cost :
  let total_ring_sales := num_rings * cost_per_ring
  let sales_from_necklaces := total_sales - total_ring_sales
  let N := sales_from_necklaces / num_necklaces
  N = 12 :=
by
  sorry

end necklace_cost_l670_670088


namespace find_M_l670_670809

theorem find_M :
  ∃ (M : ℕ), 1001 + 1003 + 1005 + 1007 + 1009 = 5100 - M ∧ M = 75 :=
by
  sorry

end find_M_l670_670809


namespace coloring_ways_l670_670053

theorem coloring_ways (grid : matrix (fin 3) (fin 3) (fin 3)) (colors : fin 3) :
  (∀ i j, grid i j ≠ grid (i+1) j ∧ grid i j ≠ grid i (j+1)) →
  (∃ n, n = 3) :=
sorry

end coloring_ways_l670_670053


namespace product_M1_M2_l670_670426

theorem product_M1_M2 :
  (∃ M1 M2 : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 3 →
    (45 * x - 36) / (x^2 - 4 * x + 3) = M1 / (x - 1) + M2 / (x - 3)) ∧
    M1 * M2 = -222.75) :=
sorry

end product_M1_M2_l670_670426


namespace both_games_players_l670_670988

theorem both_games_players (kabadi_players kho_kho_only total_players both_games : ℕ)
  (h_kabadi : kabadi_players = 10)
  (h_kho_kho_only : kho_kho_only = 15)
  (h_total : total_players = 25)
  (h_equation : kabadi_players + kho_kho_only + both_games = total_players) :
  both_games = 0 :=
by
  -- question == answer given conditions
  sorry

end both_games_players_l670_670988


namespace sin_6_cos_6_theta_proof_l670_670858

noncomputable def sin_6_cos_6_theta (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : ℝ :=
  Real.sin θ ^ 6 + Real.cos θ ^ 6

theorem sin_6_cos_6_theta_proof (θ : ℝ) (h : Real.cos (2 * θ) = 1 / 4) : 
  sin_6_cos_6_theta θ h = 19 / 64 :=
by
  sorry

end sin_6_cos_6_theta_proof_l670_670858


namespace fourth_person_height_l670_670582

variable (H : ℝ)
variable (height1 height2 height3 height4 : ℝ)

theorem fourth_person_height
  (h1 : height1 = H)
  (h2 : height2 = H + 2)
  (h3 : height3 = H + 4)
  (h4 : height4 = H + 10)
  (avg_height : (height1 + height2 + height3 + height4) / 4 = 78) :
  height4 = 84 :=
by
  sorry

end fourth_person_height_l670_670582


namespace proj_onto_line_is_constant_l670_670192

theorem proj_onto_line_is_constant :
  ∀ (a c d : ℝ), 
  (c + 3 * d = 0) → 
  ∃ (p : ℝ × ℝ), 
  p = (1 / (10 * d)) * ⟨-3 * d, d⟩ ∧ 
  (p = ((a * c + (3 * a + 1) * d) / (c^2 + d^2)) • ⟨c, d⟩) :=
by
  intro a c d h_c_d
  use ⟨-3 / 10, 1 / 10⟩
  sorry

end proj_onto_line_is_constant_l670_670192


namespace number_of_primes_between_5000_and_8000_squared_l670_670004

-- Define the conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p
def p_in_range (p : ℕ) : Prop := 5000 < p^2 ∧ p^2 < 8000

-- Define the problem statement
theorem number_of_primes_between_5000_and_8000_squared : 
  {p : ℕ // is_prime p ∧ p_in_range p}.card = 5 :=
by
  sorry --Proof placeholder, not required as per the statement

end number_of_primes_between_5000_and_8000_squared_l670_670004


namespace number_of_correct_statements_l670_670147

-- Definitions based on conditions
def statement1 : Prop := "Deductive reasoning is reasoning from general to specific."
def statement2 : Prop := "The conclusion obtained from deductive reasoning is definitely correct."
def statement3 : Prop := "The general pattern of deductive reasoning is in the form of a 'syllogism'."
def statement4 : Prop := "The correctness of the conclusion obtained from deductive reasoning depends on the major premise, minor premise, and the form of reasoning."

-- Proving the number of correct statements is 3 given the conditions
theorem number_of_correct_statements : (statement1 ∧ ¬statement2 ∧ statement3 ∧ statement4) → (3 = 3) :=
by sorry

end number_of_correct_statements_l670_670147


namespace joan_gemstones_l670_670411

def number_of_gemstones (M : ℕ) (M_y : ℕ) (G_y : ℕ) : ℕ :=
  G_y

theorem joan_gemstones
  (M : ℕ)
  (h1 : M = 48)
  (M_y : ℕ)
  (h2 : M_y = M - 6)
  (G_y : ℕ)
  (h3 : G_y = M_y / 2) : 
  number_of_gemstones M M_y G_y = 21 := 
by 
  rw [number_of_gemstones, h1, h2, h3]
  rfl


end joan_gemstones_l670_670411


namespace cost_of_fencing_each_side_l670_670015

theorem cost_of_fencing_each_side (total_cost : ℕ) (num_sides : ℕ) (h1 : total_cost = 288) (h2 : num_sides = 4) : (total_cost / num_sides) = 72 := by
  sorry

end cost_of_fencing_each_side_l670_670015


namespace central_angle_of_sector_l670_670936

-- Given
def base_area (r : ℝ) : ℝ := π * r^2
def lateral_area (r : ℝ) (l : ℝ) : ℝ := π * r * l
def total_surface_area (r : ℝ) (l : ℝ) : ℝ := base_area r + lateral_area r l

-- Condition: The surface area of the cone is three times its base area
def condition (r : ℝ) (l : ℝ) : Prop := total_surface_area r l = 3 * base_area r

-- The central angle of the sector formed by unrolling the cone's lateral surface
def central_angle (arc_length : ℝ) (radius : ℝ) : ℝ := arc_length / radius

-- Correct answer: The central angle is 180 degrees (π radians)
theorem central_angle_of_sector (r l : ℝ) (h : condition r l) : central_angle (2 * π * r) l = π := by
  -- Prove that given conditions, the central angle is π radians
  sorry

end central_angle_of_sector_l670_670936


namespace find_f_5_l670_670119

noncomputable def f : ℝ → ℝ := sorry -- linear function to be defined based on conditions
noncomputable def f_inv : ℝ → ℝ := sorry -- inverse function corresponding to f

-- Given conditions
axiom f_linear : ∃ (a b : ℝ), ∀ x : ℝ, f(x) = a * x + b
axiom f_eq : ∀ x : ℝ, f(x) = 3 * f_inv(x) - 2
axiom f_at_3 : f(3) = 5

-- Question to answer (translation to Lean statement form):
theorem find_f_5 : f(5) = (14 * Real.sqrt 3) / 3 + 1 := sorry

end find_f_5_l670_670119


namespace grade_more_problems_l670_670638

theorem grade_more_problems (worksheets_total problems_per_worksheet worksheets_graded: ℕ)
  (h1 : worksheets_total = 9)
  (h2 : problems_per_worksheet = 4)
  (h3 : worksheets_graded = 5):
  (worksheets_total - worksheets_graded) * problems_per_worksheet = 16 :=
by
  sorry

end grade_more_problems_l670_670638


namespace range_of_a_l670_670821

def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + a^2 - 4

theorem range_of_a (a : ℝ) (h₀ : a > 0)
  (h₁ : ∀ x ∈ set.Icc (a-2) a^2, f x a ∈ set.Icc (-4) 0) :
  1 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l670_670821


namespace equal_clique_assignment_l670_670381

-- Definitions of the conditions
def mutual_friendship (contestants : Type) (friend : contestants → contestants → Prop) : Prop :=
  ∀ a b, friend a b ↔ friend b a

def is_clique (contestants : Type) (friend : contestants → contestants → Prop) (group : set contestants) : Prop :=
  (∀ a b ∈ group, friend a b) ∧ (group = ∅ ∨ group = {x} ∨ group.nonempty)

-- The math proof problem
theorem equal_clique_assignment
  {contestants : Type} (friend : contestants → contestants → Prop)
  (participants : set contestants) :
  mutual_friendship contestants friend →
  (∀ group ⊆ participants, is_clique contestants friend group → |group| ≤ 2 * ((|group| + 1) / 2)) →
  (∃ max_clique, is_clique contestants friend max_clique ∧ ∀ clique, is_clique contestants friend clique → |max_clique| ≥ |clique|) →
  2 ∣ (∃ max_clique, is_clique contestants friend max_clique ∧ ∀ clique, is_clique contestants friend clique → |max_clique| ≥ |clique|).fst →
  ∃ (roomA roomB : set contestants), 
  (∀ (a b ∈ roomA), friend a b) ∧ (∀ (c d ∈ roomB), friend c d) ∧ 
  (let C := {group : set contestants | is_clique contestants friend group} in
    ∀ max_clique ∈ C, (|max_clique| = |max_clique|)) := sorry

end equal_clique_assignment_l670_670381


namespace simplify_expr_l670_670564

noncomputable def expr : ℝ := sqrt (25 * sqrt (16 * sqrt 9))

theorem simplify_expr : expr = 10 * (real.sqrt 3) ^ (1 / 4) :=
by
  sorry

end simplify_expr_l670_670564


namespace find_constant_l670_670378

theorem find_constant (t : ℝ) (constant : ℝ) :
  (x = constant - 3 * t) → (y = 2 * t - 3) → (t = 0.8) → (x = y) → constant = 1 :=
by
  intros h1 h2 h3 h4
  sorry

end find_constant_l670_670378


namespace eval_x_squared_minus_y_squared_l670_670735

theorem eval_x_squared_minus_y_squared (x y : ℝ) (h1 : 3 * x + 2 * y = 30) (h2 : 4 * x + 2 * y = 34) : x^2 - y^2 = -65 :=
by
  sorry

end eval_x_squared_minus_y_squared_l670_670735


namespace smallest_number_diminished_by_10_l670_670206

theorem smallest_number_diminished_by_10 (x : ℕ) (h : ∀ n, x - 10 = 24 * n) : x = 34 := 
  sorry

end smallest_number_diminished_by_10_l670_670206


namespace counterexample_exists_l670_670663

-- Define prime predicate
def is_prime (n : ℕ) : Prop :=
∀ m, m ∣ n → m = 1 ∨ m = n

def counterexample_to_statement (n : ℕ) : Prop :=
  is_prime n ∧ ¬ is_prime (n + 2)

theorem counterexample_exists : ∃ n ∈ [3, 5, 11, 17, 23], is_prime n ∧ ¬ is_prime (n + 2) :=
by
  sorry

end counterexample_exists_l670_670663


namespace original_price_of_car_l670_670176

theorem original_price_of_car (spent price_percent original_price : ℝ) (h1 : spent = 15000) (h2 : price_percent = 0.40) (h3 : spent = price_percent * original_price) : original_price = 37500 :=
by
  sorry

end original_price_of_car_l670_670176


namespace f_a1_a3_a5_positive_l670_670745

theorem f_a1_a3_a5_positive (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf_odd : ∀ x, f (-x) = - f x)
  (hf_mono : ∀ x y, x < y → f x < f y)
  (ha_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (ha3_pos : 0 < a 3) :
  0 < f (a 1) + f (a 3) + f (a 5) :=
sorry

end f_a1_a3_a5_positive_l670_670745


namespace circles_exceeding_n_squared_l670_670037

noncomputable def num_circles (n : ℕ) : ℕ :=
  if n >= 8 then 
    5 * n + 4 * (n - 1)
  else 
    n * n

theorem circles_exceeding_n_squared (n : ℕ) (hn : n ≥ 8) : num_circles n > n^2 := 
by {
  sorry
}

end circles_exceeding_n_squared_l670_670037


namespace no_factorization_of_f_l670_670440

theorem no_factorization_of_f (n : ℤ) (h : 1 < n) :
  ¬ ∃ (g h : ℕ[X]), g.degree ≥ 1 ∧ h.degree ≥ 1 ∧ 
    (∀ (a : ℤ), a ∈ g.coeff) ∧ 
    (∀ (b : ℤ), b ∈ h.coeff) ∧ 
    (polynomial.eval₂ g.has_int_coeff : ℕ[X] → ℤ[X]) (λ x : ℤ, (C x : ℕ[X] → ℤ[X])) g * 
    (polynomial.eval₂ h.has_int_coeff : ℕ[X] → ℤ[X]) (λ x : ℤ, (C x : ℕ[X] → ℤ[X])) h = 
    polynomial.eval₂ (Polynomial.of_finsupp $ add_monoid_algebra.of_finsupp x) (λ x : ℤ, (C x : ℕ → ℤ)) (Polynomial.of_finsupp $ add_monoid.algebra.to_finsupp <$>) :=
by
  sorry

end no_factorization_of_f_l670_670440


namespace find_f1_l670_670082

theorem find_f1 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) + (-x) ^ 2 = -(f x + x ^ 2))
  (h2 : ∀ x, f (-x) + 2 ^ (-x) = f x + 2 ^ x) :
  f 1 = -7 / 4 := by
sorry

end find_f1_l670_670082


namespace solution_set_of_inequality_l670_670935

theorem solution_set_of_inequality :
  {x : ℝ | 9 * x^2 + 6 * x + 1 ≤ 0} = {-1 / 3} :=
by {
  sorry -- Proof goes here
}

end solution_set_of_inequality_l670_670935


namespace perfect_squares_100_to_400_l670_670807

theorem perfect_squares_100_to_400 :
  {n : ℕ | 100 ≤ n^2 ∧ n^2 ≤ 400}.card = 11 :=
by {
  sorry
}

end perfect_squares_100_to_400_l670_670807


namespace number_of_people_got_on_train_l670_670529

theorem number_of_people_got_on_train (initial_people : ℕ) (people_got_off : ℕ) (final_people : ℕ) (x : ℕ) 
  (h_initial : initial_people = 78) 
  (h_got_off : people_got_off = 27) 
  (h_final : final_people = 63) 
  (h_eq : final_people = initial_people - people_got_off + x) : x = 12 :=
by 
  sorry

end number_of_people_got_on_train_l670_670529


namespace parallelogram_area_correct_l670_670293

def parallelogram_area (base height : ℝ) : ℝ := base * height

theorem parallelogram_area_correct (base height : ℝ) (h_base : base = 30) (h_height : height = 12) : parallelogram_area base height = 360 :=
by
  rw [h_base, h_height]
  simp [parallelogram_area]
  sorry

end parallelogram_area_correct_l670_670293


namespace find_possible_ks_l670_670159

noncomputable theory
open_locale classical

-- Define the number of people
def num_people : ℕ := 100

-- Define natural number k such that k < 100
def valid_k (k : ℕ) : Prop :=
  k < num_people

-- Define the set of possible values for k
def possible_ks : set ℕ :=
  {1, 3, 4, 9, 19, 24, 49, 99}

-- Main theorem statement
theorem find_possible_ks (k : ℕ) (h_k : valid_k k) :
  (∃ k, valid_k k ∧
  ∀ (i : ℕ) (h_i : i < num_people), 
  (i mod (k + 1) = 0 -> 
  ∀ (j : ℕ) (h_j : j ≤ k), ((i + j) mod num_people) mod (k + 1) ≠ 1) ∧
  (i mod (k + 1) ≠ 0 -> 
  (∃ (j : ℕ) (h_j : j ≤ k), ((i + j) mod num_people) mod (k + 1) = 1))) 
  ↔ k ∈ possible_ks :=
sorry

end find_possible_ks_l670_670159


namespace math_problem_l670_670491

noncomputable def parametric_equation_line (x y t : ℝ) : Prop :=
  x = 1 + (1/2) * t ∧ y = -5 + (Real.sqrt 3 / 2) * t

noncomputable def polar_equation_circle (ρ θ : ℝ) : Prop :=
  ρ = 8 * Real.sin θ

noncomputable def line_disjoint_circle (sqrt3 x y d : ℝ) : Prop :=
  sqrt3 = Real.sqrt 3 ∧ x = 0 ∧ y = 4 ∧ d = (9 + sqrt3) / 2 ∧ d > 4

theorem math_problem 
  (t θ x y ρ sqrt3 d : ℝ) :
  parametric_equation_line x y t ∧
  polar_equation_circle ρ θ ∧
  line_disjoint_circle sqrt3 x y d :=
by
  sorry

end math_problem_l670_670491


namespace water_percentage_fresh_grapes_l670_670715

variable (Wf Wd : ℝ)
variable (Pd : ℝ := 0.20)

def percentage_water_in_fresh_grapes (P : ℝ) : Prop :=
  Wf = 10 ∧ Wd = 1.25 ∧
  Pd = 0.20 ∧
  (Wf - (Wd - Pd * Wd)) / Wf * 100 = P

theorem water_percentage_fresh_grapes :
  percentage_water_in_fresh_grapes 90 :=
by
  sorry

end water_percentage_fresh_grapes_l670_670715


namespace num_wide_right_misses_l670_670545

-- Define the conditions
variables (totalFieldGoals : ℕ) (missFraction : ℚ) (wideRightFraction : ℚ)

-- Given conditions
def totalFieldGoals := 60
def missFraction := 1 / 4
def wideRightFraction := 20 / 100

-- Calculate the actual number of field goals
def missedFieldGoals := totalFieldGoals * missFraction
def wideRightMisses := missedFieldGoals * wideRightFraction

-- Theorem to be proved
theorem num_wide_right_misses : wideRightMisses = 3 := sorry

end num_wide_right_misses_l670_670545


namespace focus_of_hyperbola_l670_670296

def coordinates_of_focus (a b : ℝ) (ha : a^2 = 3) (hb : b^2 = 1) : Prop :=
  let c := sqrt (a^2 + b^2)
  (c, 0) = (2, 0)

theorem focus_of_hyperbola : coordinates_of_focus (sqrt 3) 1 (by norm_num) (by norm_num) :=
sorry

end focus_of_hyperbola_l670_670296


namespace max_links_opened_l670_670224

variable (N : ℕ)
hypothesis (h : N > 3)

theorem max_links_opened (h : N > 3) : (max_links_to_open N = Nat.floor (3 * N / 4)) :=
by
  sorry

end max_links_opened_l670_670224


namespace calculate_actual_distance_l670_670889

-- Definitions corresponding to the conditions
def map_scale : ℕ := 6000000
def map_distance_cm : ℕ := 5

-- The theorem statement corresponding to the proof problem
theorem calculate_actual_distance :
  (map_distance_cm * map_scale / 100000) = 300 := 
by
  sorry

end calculate_actual_distance_l670_670889


namespace milk_left_l670_670450

theorem milk_left (initial_milk : ℝ) (given_away : ℝ) (h_initial : initial_milk = 5) (h_given : given_away = 18 / 4) :
  ∃ remaining_milk : ℝ, remaining_milk = initial_milk - given_away ∧ remaining_milk = 1 / 2 :=
by
  use 1 / 2
  sorry

end milk_left_l670_670450


namespace combined_value_of_a_and_b_l670_670811

theorem combined_value_of_a_and_b :
  (∃ a b : ℝ,
    0.005 * a = 95 / 100 ∧
    b = 3 * a - 50 ∧
    a + b = 710) :=
sorry

end combined_value_of_a_and_b_l670_670811


namespace donor_multiple_l670_670996

def cost_per_box (food_cost : ℕ) (supplies_cost : ℕ) : ℕ := food_cost + supplies_cost

def total_initial_cost (num_boxes : ℕ) (cost_per_box : ℕ) : ℕ := num_boxes * cost_per_box

def additional_boxes (total_boxes : ℕ) (initial_boxes : ℕ) : ℕ := total_boxes - initial_boxes

def donor_contribution (additional_boxes : ℕ) (cost_per_box : ℕ) : ℕ := additional_boxes * cost_per_box

def multiple (donor_contribution : ℕ) (initial_cost : ℕ) : ℕ := donor_contribution / initial_cost

theorem donor_multiple 
    (initial_boxes : ℕ) (box_cost : ℕ) (total_boxes : ℕ) (donor_multi : ℕ)
    (h1 : initial_boxes = 400) 
    (h2 : box_cost = 245) 
    (h3 : total_boxes = 2000)
    : donor_multi = 4 :=
by
    let initial_cost := total_initial_cost initial_boxes box_cost
    let additional_boxes := additional_boxes total_boxes initial_boxes
    let contribution := donor_contribution additional_boxes box_cost
    have h4 : contribution = 392000 := sorry
    have h5 : initial_cost = 98000 := sorry
    have h6 : donor_multi = contribution / initial_cost := sorry
    -- Therefore, the multiple should be 4
    exact sorry

end donor_multiple_l670_670996


namespace max_roses_l670_670473

theorem max_roses (individual_cost dozen_cost two_dozen_cost budget : ℝ) 
  (h1 : individual_cost = 7.30) 
  (h2 : dozen_cost = 36) 
  (h3 : two_dozen_cost = 50) 
  (h4 : budget = 680) : 
  ∃ n, n = 316 :=
by
  sorry

end max_roses_l670_670473


namespace sin_cos_power_equality_l670_670860

theorem sin_cos_power_equality (θ : ℝ) (h : cos (2 * θ) = 1 / 4) : (sin θ) ^ 6 + (cos θ) ^ 6 = 19 / 64 :=
by
  sorry

end sin_cos_power_equality_l670_670860


namespace lines_perpendicular_l670_670301

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := y = -3 * x - 7
def line2 (x y : ℝ) (c : ℝ) : Prop := 15 * y + c * x = 30

-- Define the slope of the first line
def slope1 : ℝ := -3

-- Define the slope of the second line in terms of c
def slope2 (c : ℝ) : ℝ := -c / 15

-- The property that two lines are perpendicular
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- The proof problem
theorem lines_perpendicular (c : ℝ) :
  line1 x y ∧ line2 x y c →
  perpendicular_slopes slope1 (slope2 c) →
  c = -5 :=
by
  sorry

end lines_perpendicular_l670_670301


namespace probability_exact_two_integer_points_in_square_l670_670856

def square_diagonal := (1/8, 5/8, -1/8, -5/8)

def random_point (x y : ℝ) : Prop := 
  0 ≤ x ∧ x ≤ 1000 ∧ 0 ≤ y ∧ y ≤ 1000

noncomputable def side_length : ℝ :=
  let (x1, y1, x2, y2) := square_diagonal
  in (real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)) / real.sqrt 2

def T (x y : ℝ) : Prop :=
  let s := side_length
  in ∃ a b : ℤ, x - s/2 < a ∧ a < x + s/2 ∧ y - s/2 < b ∧ b < y + s/2

theorem probability_exact_two_integer_points_in_square :
  ∀ (x y : ℝ), random_point x y → 
  (∃ count : ℕ, count = (finset.card (finset.filter (T x y) int.points) ∧ count = 2)) →
  measure_theory.volume {v | random_point v.1 v.2 ∧ 
    ∃ count : ℕ, count = (finset.card (finset.filter (T v.1 v.2) int.points) ∧ count = 2)} = (4/25) :=
sorry

end probability_exact_two_integer_points_in_square_l670_670856


namespace prob_defective_l670_670647

/-- Assume there are two boxes of components. 
    The first box contains 10 pieces, including 2 defective ones; 
    the second box contains 20 pieces, including 3 defective ones. --/
def box1_total : ℕ := 10
def box1_defective : ℕ := 2
def box2_total : ℕ := 20
def box2_defective : ℕ := 3

/-- Randomly select one box from the two boxes, 
    and then randomly pick 1 component from that box. --/
def prob_select_box : ℚ := 1 / 2

/-- Probability of selecting a defective component given that box 1 was selected. --/
def prob_defective_given_box1 : ℚ := box1_defective / box1_total

/-- Probability of selecting a defective component given that box 2 was selected. --/
def prob_defective_given_box2 : ℚ := box2_defective / box2_total

/-- The probability of selecting a defective component is 7/40. --/
theorem prob_defective :
  prob_select_box * prob_defective_given_box1 + prob_select_box * prob_defective_given_box2 = 7 / 40 :=
sorry

end prob_defective_l670_670647


namespace kl_divides_bm_in_equal_ratio_l670_670095

theorem kl_divides_bm_in_equal_ratio
  (ABC : Type) [triangle ABC]
  (A B C K L M : ABC)
  (AK : rat) (KB : rat) (BL : rat) (LC : rat) (CM : rat) (MA : rat) :
  AK / KB = 2 / 3 →
  BL / LC = 1 / 2 →
  CM / MA = 3 / 1 →
  let O := intersect (segment BM) (segment KL) in
  ratio_of_segments O B BM = 1 / 1 :=
by
  sorry

end kl_divides_bm_in_equal_ratio_l670_670095


namespace missing_striped_tiles_l670_670941

theorem missing_striped_tiles : ∀ (initial_checkerboard : nat → nat → bool) (fallen_tiles : nat → nat → bool), 
  (∀ i j, initial_checkerboard i j = false ∨ initial_checkerboard i j = true) ∧
  (∀ i j, fallen_tiles i j = true → initial_checkerboard i j = true) ∧
  (∃ remainder_checkerboard : nat → nat → bool, 
    (∀ i j, remainder_checkerboard i j = initial_checkerboard i j ∧ fallen_tiles i j = false)) →
  (∃ n, n = 15) :=
begin
  sorry
end

end missing_striped_tiles_l670_670941


namespace perfect_squares_between_100_and_400_l670_670792

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l670_670792


namespace count_ordered_pairs_satisfying_conditions_l670_670275

open Real

def satisfies_conditions (a b : ℤ) : Prop :=
  a ^ 2 + b ^ 2 < 25 ∧
  a ^ 2 + b ^ 2 < 10 * a ∧
  a ^ 2 + b ^ 2 < 10 * b

theorem count_ordered_pairs_satisfying_conditions :
  (finset.univ.filter (λ (ab : ℤ × ℤ), satisfies_conditions ab.1 ab.2)).card = 13 :=
by 
  sorry

end count_ordered_pairs_satisfying_conditions_l670_670275


namespace rationalize_denominator_l670_670107

theorem rationalize_denominator (A B C D E : ℤ) 
  (hB_lt_D : B < D) (h_fraction : (5 : ℝ) / (4*real.sqrt 7 + 3*real.sqrt 13) = (A*real.sqrt B + C*real.sqrt D) / E) 
  (h_simplest_form : true) -- assume we have the simplest terms, this would need further detail in a full proof
  : A + B + C + D + E = 22 :=
sorry

end rationalize_denominator_l670_670107


namespace water_percentage_calc_l670_670573

noncomputable def percentage_of_water
    (initial_volume : ℕ)
    (profit_percentage : ℕ)
    (water_volume : ℕ)
    (total_volume : ℕ) : ℕ :=
  (water_volume * 100) / total_volume

theorem water_percentage_calc (V : ℕ)
    (P : ℕ)
    (new_volume : ℕ)
    (water_volume : ℕ)
    (percentage_water : ℕ) : ℕ :=
begin
  -- Initial volumes and percentages
  assume V = 1,
  assume P = 25,

  -- Calculate new volume
  have new_volume := V + (P * V) / 100,

  -- Calculate added water volume
  have water_volume := new_volume - V,

  -- Calculate percentage of water in the mixture
  have percentage_water := (water_volume * 100) / new_volume,

  -- Final result
  exact 20
end

end water_percentage_calc_l670_670573


namespace square_covering_equal_rectangles_and_perimeter_l670_670524

constant list_of_squares : list ℕ :=
  [172, 1, 5, 7, 11, 20, 27, 34, 41, 42, 43, 44, 61, 85, 95, 108, 113, 118, 123, 136, 168, 183, 194, 205, 209, 231]

def total_area_squares (l : list ℕ) : ℕ :=
  list.sum (l.map (λ x : ℕ, x * x))

theorem square_covering_equal_rectangles_and_perimeter :
  total_area_squares list_of_squares = 608 * 608 →
  ∃ (r1 r2 : list ℕ), r1 ≠ r2 ∧ r1 ≠ [] ∧ r2 ≠ [] ∧ 
  list.sum r1 = list.sum r2 ∧ 
  (∃ (last10 : list ℕ), last10 = [113, 118, 123, 136, 168, 183, 194, 205, 209, 231] ∧ 
   (∀ x ∈ last10, x ∈ r1 ++ r2) ∧ 
   list.sum (r1 ++ r2) = 608 * 4) :=
by
  sorry

end square_covering_equal_rectangles_and_perimeter_l670_670524


namespace arrangement_count_l670_670256

theorem arrangement_count :
  let students := [A, B, C, D]
  let schools := [S1, S2, S3]
  let valid_arrangements := {arr | 
    ∀ s ∈ schools, ∃ t ∈ students, arr t = s ∧ -- each school must have at least 1 student
                 ∀ t1 t2 ∈ students, t1 ≠ t2 → arr t1 ≠ arr t2, -- each student can only go to one school
                 arr A ≠ arr B -- A and B are not in the same school 
    }
  valid_arrangements.card = 30 :=
  sorry

end arrangement_count_l670_670256


namespace school_distance_l670_670847

open Real

-- Definitions based on the conditions
def time_in_rush_hour := (20 / 60 : ℝ)  -- time in hours
def time_no_traffic := (15 / 60 : ℝ)  -- time in hours
def speed_increase := (15 : ℝ)  -- speed increment in miles per hour

-- Main theorem to prove the distance to school
theorem school_distance (v : ℝ) (d : ℝ) :
  d = v * time_in_rush_hour ∧ d = (v + speed_increase) * time_no_traffic → d = 15 :=
by
  sorry

end school_distance_l670_670847


namespace maximize_power_speed_l670_670589

-- Define the necessary parameters and constants
variables (C S ρ v0 v : ℝ)
constants (hS : S = 5) (hv0 : v0 = 6)

-- Define the force equation
def force (v : ℝ) : ℝ := (C * S * ρ * (v0 - v)^2) / 2

-- Define the power equation
def power (v : ℝ) : ℝ := force v * v

-- Statement to prove that the speed v that maximizes the power is v0 / 3
theorem maximize_power_speed :
  (v = v0 / 3) → (v0 = 6) → (S = 5) → (v_0 = 6) := by
sory

end maximize_power_speed_l670_670589


namespace num_perfect_squares_l670_670794

theorem num_perfect_squares (a b : ℤ) (h₁ : a = 100) (h₂ : b = 400) : 
  ∃ n : ℕ, (100 < n^2) ∧ (n^2 < 400) ∧ (n = 9) :=
by
  sorry

end num_perfect_squares_l670_670794


namespace mark_father_money_l670_670452

theorem mark_father_money 
  (books_bought : ℕ)
  (cost_per_book : ℕ)
  (money_left : ℕ)
  (total_cost : ℕ := books_bought * cost_per_book)
  (money_given : ℕ := total_cost + money_left) :
  books_bought = 10 → cost_per_book = 5 → money_left = 35 → money_given = 85 := by
  intros h1 h2 h3
  simp [h1, h2, h3]
  sorry

end mark_father_money_l670_670452


namespace triangle_largest_angle_l670_670383

theorem triangle_largest_angle (x : ℝ) (AB : ℝ) (AC : ℝ) (BC : ℝ) (h1 : AB = x + 5) 
                               (h2 : AC = 2 * x + 3) (h3 : BC = x + 10)
                               (h_angle_A_largest : BC > AB ∧ BC > AC)
                               (triangle_inequality_1 : AB + AC > BC)
                               (triangle_inequality_2 : AB + BC > AC)
                               (triangle_inequality_3 : AC + BC > AB) :
  1 < x ∧ x < 7 ∧ 6 = 6 := 
by {
  sorry
}

end triangle_largest_angle_l670_670383


namespace focus_of_parabola_l670_670018

theorem focus_of_parabola (a : ℝ) : 
  (∃ c : ℝ, y^2 = a * x ∧ x = -1 ∧ c = 4 → (a = 4)) → (coordinates_of_focus (1, 0)) :=
by
  sorry

end focus_of_parabola_l670_670018


namespace solve_equation_l670_670907

-- Definition for the greatest integer function
def floor (x : ℝ) : ℤ := Int.floor x

-- Hypothesis: there exists a real number x that fulfills the given equation.
theorem solve_equation :
  ∃ x : ℝ, x^3 - (↑(floor x) : ℝ) = 3 ∧ x = (4: ℝ)^(1/3 : ℝ) := 
sorry

end solve_equation_l670_670907


namespace selection_of_students_l670_670476

theorem selection_of_students (A B C : student) (students : finset student) (h_total : |students| = 9) (h_ABC : {A, B, C} ⊆ students) : 
  (∑ s in students.powerset, if s.card = 4 ∧ ({A, B, C} ∩ s).card ≥ 2 then 1 else 0) = 51 := 
sorry

end selection_of_students_l670_670476


namespace min_max_value_is_zero_l670_670684

def max_at_x (x : ℝ) (y : ℝ) : ℝ := |x^2 - 2 * x * y|

theorem min_max_value_is_zero :
  ∃ y ∈ set.univ, min (set.univ) (λ y, real.sup (set.Icc 0 2) (λ x, max_at_x x y)) = 0 :=
sorry

end min_max_value_is_zero_l670_670684


namespace balls_in_boxes_l670_670350

theorem balls_in_boxes (n m : Nat) (h : n = 6) (k : m = 2) : (m ^ n) = 64 := by
  sorry

end balls_in_boxes_l670_670350


namespace tile_in_center_l670_670479

-- Define the coloring pattern of the grid
inductive Color
| A | B | C

-- Predicates for grid, tile placement, and colors
def Grid := Fin 5 × Fin 5

def is_1x3_tile (t : Grid × Grid × Grid) : Prop :=
  -- Ensure each tuple t represents three cells that form a $1 \times 3$ tile
  sorry

def is_tiling (g : Grid → Option Color) : Prop :=
  -- Ensure the entire grid is correctly tiled with the given tiles and within the coloring pattern
  sorry

def center : Grid := (Fin.mk 2 (by decide), Fin.mk 2 (by decide))

-- The theorem statement
theorem tile_in_center (g : Grid → Option Color) : is_tiling g → 
  (∃! tile : Grid, g tile = some Color.B) :=
sorry

end tile_in_center_l670_670479


namespace number_of_pints_of_paint_l670_670814

-- Statement of the problem
theorem number_of_pints_of_paint (A B : ℝ) (N : ℕ) 
  (large_cube_paint : ℝ) (hA : A = 4) (hB : B = 2) (hN : N = 125) 
  (large_cube_paint_condition : large_cube_paint = 1) : 
  (N * (B / A) ^ 2 * large_cube_paint = 31.25) :=
by {
  -- Given the conditions
  sorry
}

end number_of_pints_of_paint_l670_670814


namespace max_length_third_side_l670_670910

noncomputable def length_of_third_side (P Q R : ℝ) (a b : ℝ) (h1 : cos (4 * P) + cos (4 * Q) + cos (4 * R) = 1) (h2 : a = 7) (h3 : b = 24)
  : ℝ :=
  25

theorem max_length_third_side (P Q R : ℝ) (a b : ℝ) (h1 : cos (4 * P) + cos (4 * Q) + cos (4 * R) = 1) (h2 : a = 7) (h3 : b = 24)
  : length_of_third_side P Q R a b h1 h2 h3 = 25 :=
  sorry

end max_length_third_side_l670_670910


namespace parabola_equation_slope_product_minimum_l670_670726

noncomputable def parabola_condition (p : ℝ) (h : p > 0) : Prop :=
  ∃ t : ℝ, (3 + (p / 2) = 4 ∧ (3,t).dist (0,p / 2) = 4)

theorem parabola_equation (p : ℝ) (hp : p > 0) (h : parabola_condition p hp) : 
  ∃ x y : ℝ, y^2 = 4 * x := 
sorry

noncomputable def slope_product_condition : Prop :=
  ∃ m k1 k2 : ℝ, (k1 * k2 = (-1) / (m^2 + 4) ∧ ∀ m', -1 / (m'^2 + 4) ≥ -1 / (m^2 + 4)) 

theorem slope_product_minimum :
  slope_product_condition → 
  ∃ k1 k2 : ℝ, k1 * k2 = -1 / 4 :=
sorry

end parabola_equation_slope_product_minimum_l670_670726


namespace find_CD_length_l670_670045

open TriangleTheory

theorem find_CD_length
  (ABC : Triangle)
  (AB AC BC : ℝ)
  (D : Point)
  (AD : Segment)
  (hAB : AB = 8)
  (hAC : AC = 10)
  (hBC : BC = 17) -- Considering the required correction 
  (hAD_bisector : AD.is_angle_bisector ∠ABC)
  (hBD : 5) :
  (DC : ℝ) := by
  sorry

example : find_CD_length ABC AB AC BC D AD hAB hAC hBC hAD_bisector hBD = 6.25 := by
  sorry

end find_CD_length_l670_670045


namespace exprB_cannot_be_simplified_by_binomial_l670_670962

-- Definitions for the given binomial expressions
def exprA (m n : ℝ) := (-m - n) * (-m + n)
def exprB (m n : ℝ) := (-m - n) * (m + n)
def exprC (m n p : ℝ) := (mn + p) * (mn - p)
def exprD (m n : ℝ) := (0.3 * m - n) * (-n - 0.3 * m)

-- Proof statement that exprB cannot be simplified using the square of a binomial formula
theorem exprB_cannot_be_simplified_by_binomial (m n : ℝ) :
  ¬ ∃ (a b : ℝ), exprB m n = a^2 - b^2 := 
sorry

end exprB_cannot_be_simplified_by_binomial_l670_670962


namespace probability_divisible_by_9_l670_670372

theorem probability_divisible_by_9 (f : Fin 5 → ℕ) (h : ∑ i, f i = 36) (h1 : ∀ i, 1 ≤ f 0) : 
  ∃ n, (f 0 :: f 1 :: f 2 :: f 3 :: f 4 = [9, 9, 9, 9, 0] ∨ f 0 :: f 1 :: f 2 :: f 3 :: f 4 = [8, 8, 8, 8, 4]) → 
  ∀ n ∈ list.permutations (f 0 :: f 1 :: f 2 :: f 3 :: f 4), 9 ∣ ∑ i in n, f i :=
begin
  sorry
end

end probability_divisible_by_9_l670_670372


namespace ex1_l670_670817

-- Define the condition
def ex1_cond (x : ℝ) : Prop :=
  7 ^ (4 * x) = 2401

-- Define the goal
def ex1_goal (x : ℝ) : Prop :=
  7 ^ (4 * x - 1) = 343

-- State the theorem using the condition to imply the goal
theorem ex1 (x : ℝ) (h : ex1_cond x) : ex1_goal x :=
by
  sorry

end ex1_l670_670817


namespace distribute_balls_into_boxes_l670_670349

theorem distribute_balls_into_boxes :
  let balls := 6
  let boxes := 2
  (boxes ^ balls) = 64 :=
by 
  sorry

end distribute_balls_into_boxes_l670_670349


namespace min_possible_value_box_l670_670359

theorem min_possible_value_box :
  ∃ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15 ∧ a^2 + b^2 = 61) ∧
  ∀ (a b : ℤ), (a * b = 30 ∧ abs a ≤ 15 ∧ abs b ≤ 15) → (a^2 + b^2 ≥ 61) :=
by {
  sorry
}

end min_possible_value_box_l670_670359


namespace irene_age_calculation_l670_670285

theorem irene_age_calculation (eddie_age : ℝ) (becky_factor : ℝ) (jason_factor : ℝ) (irene_factor : ℝ) 
  (h1 : eddie_age = 92)
  (h2 : becky_factor = 1 / 4)
  (h3 : jason_factor = 1.5)
  (h4 : irene_factor = 3 / 2) : 
  let becky_age := eddie_age * becky_factor,
      jason_age := becky_age * jason_factor,
      total_age := becky_age + jason_age,
      irene_age := irene_factor * total_age
  in irene_age = 86.25 :=
by sorry

end irene_age_calculation_l670_670285


namespace tagged_fish_in_second_catch_l670_670030

theorem tagged_fish_in_second_catch :
  ∀ (T : ℕ),
    (40 > 0) →
    (800 > 0) →
    (T / 40 = 40 / 800) →
    T = 2 := 
by
  intros T h1 h2 h3
  sorry

end tagged_fish_in_second_catch_l670_670030


namespace problem_intersection_l670_670340

open Set

-- Define sets P and Q based on given conditions
def P : Set ℝ := {x | x^2 - 9 < 0}
def Q : Set ℝ := {x | x^2 - 1 > 0}

-- The problem statement to prove
theorem problem_intersection : P ∩ Q = (-3, -1) ∪ (1, 3) := 
by sorry

end problem_intersection_l670_670340


namespace ratio_expression_l670_670364

theorem ratio_expression (A B C : ℚ) (h : A / B = 3 / 1 ∧ B / C = 1 / 6) : 
  (4 * A + 3 * B) / (5 * C - 2 * A) = 5 / 8 :=
by sorry

end ratio_expression_l670_670364


namespace find_distance_MF_l670_670318

-- Define the parabola and point conditions
def parabola (x y : ℝ) := y^2 = 8 * x

-- Define the focus of the parabola
def F : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the distance squared between two points
def dist_squared (A B : ℝ × ℝ) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2

-- Prove the required statement
theorem find_distance_MF (x y : ℝ) (hM : parabola x y) (h_dist: dist_squared (x, y) O = 3 * (x + 2)) :
  dist_squared (x, y) F = 9 := by
  sorry

end find_distance_MF_l670_670318


namespace inequality_c_l670_670667

noncomputable def smallest_c := 1 / 2

theorem inequality_c (c : ℝ) : (0 < c) ∧ (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → 
    (real.cbrt (x * y) + c * |x^(2/3) - y^(2/3)| ≥ (x^(2/3) + y^(2/3)) / 2)) → c ≥ smallest_c :=
by sorry

end inequality_c_l670_670667


namespace smallest_integer_proof_l670_670187

def smallest_integer_condition (n : ℤ) : Prop := n^2 - 15 * n + 56 ≤ 0

theorem smallest_integer_proof :
  ∃ n : ℤ, smallest_integer_condition n ∧ ∀ m : ℤ, smallest_integer_condition m → n ≤ m :=
sorry

end smallest_integer_proof_l670_670187


namespace certain_number_less_32_l670_670213

theorem certain_number_less_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 :=
by
  sorry

end certain_number_less_32_l670_670213


namespace ramu_profit_percent_l670_670580

theorem ramu_profit_percent (purchase_price repair_costs selling_price : ℝ)
  (h_purchase : purchase_price = 42000) 
  (h_repairs : repair_costs = 13000) 
  (h_selling : selling_price = 66900) :
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100 ≈ 21.64 :=
by {
  rw [h_purchase, h_repairs, h_selling],
  sorry,
}

end ramu_profit_percent_l670_670580


namespace perfect_squares_between_100_and_400_l670_670790

theorem perfect_squares_between_100_and_400 :
  let n := 11
  let m := 19
  list.count (list.map (λ x, x * x) (list.range (m - n + 1) + (fun c => c + n))) = 9 := by
    sorry  -- Proof omitted

end perfect_squares_between_100_and_400_l670_670790


namespace courses_form_a_set_l670_670961

def group_of_people_who_like_airplanes (S : Type) : Prop :=
∃ p : S, p ∈ S ∧ likes_airplanes p

def sufficiently_small_negative_numbers (R : ℝ) : Prop :=
∃ x : ℝ, x ∈ set_of (λ n, n < 0) ∧ small_enough x

def students_with_poor_eyesight (S : Type) (class : S) : Prop :=
∃ s : S, s ∈ class ∧ poor_eyesight s

def courses_of_class_on_particular_day (C : Type) (school : C) : Prop :=
∃ c : C, c ∈ school ∧ has_classes_on_day c.ParticularDay

theorem courses_form_a_set (C : Type) (school : C) : 
  ¬ group_of_people_who_like_airplanes C ∧
  ¬ sufficiently_small_negative_numbers ℝ ∧
  ¬ students_with_poor_eyesight C school ∧
  courses_of_class_on_particular_day C school :=
begin
  sorry
end

end courses_form_a_set_l670_670961


namespace number_of_subsets_of_three_element_set_l670_670928

theorem number_of_subsets_of_three_element_set :
  ∃ (S : Finset ℕ), S.card = 3 ∧ S.powerset.card = 8 :=
sorry

end number_of_subsets_of_three_element_set_l670_670928


namespace find_a_of_purely_imaginary_l670_670736

-- We need to define some basic complex number operations and properties
open Complex

theorem find_a_of_purely_imaginary (a : ℝ) (h : Im ((2 * I - 1) / (1 + a * I)) = (2 * I - 1) / (1 + a * I)) : a = 1 / 2 :=
by
  -- Detail the provided conditions and construct the required proof
  have h1 : (2 : ℂ) = ↑(2 : ℝ) := by norm_cast
  have h2 : (1 : ℂ) = ↑(1 : ℝ) := by norm_cast
  have ha : (a : ℂ) = ↑a := by norm_cast

  let z := (2 * I - 1) / (1 + a * (I : ℂ))
  let numer := 2 * I - 1
  let denom := 1 + a * I

  calc
    Re (z) = Re (numer / denom) : rfl
    ... = (Re numer * Re denom + Im numer * Im denom) / (Re denom^2 + Im denom^2) : sorry
    ... = ((0 - 1 * 1) + (2 * 1) * a) / (1 + a^2) : sorry
    ... = 0 + a : sorry

  have ha_eq_zero : a = (1 - 2 * a) / (1 + a^2) := by sorry
  have ha_part : a * (1 + a^2) = 1 - 2 * a : sorry
  have ha_solve : a = 1 / 2 := by sorry

  exact ha_solve

end find_a_of_purely_imaginary_l670_670736


namespace no_zero_in_range_l670_670867

noncomputable def g (x : ℝ) : ℤ :=
if x > -3 then ⌈1 / (x + 3)⌉
else if x < -3 then ⌊1 / (x + 3)⌋
else 0 -- This value is arbitrary as g(x) is not defined at x = -3

theorem no_zero_in_range : ¬ ∃ x : ℝ, g x = 0 :=
begin
  unfold g,
  intros h,
  cases h with x hx,
  rw if_neg (by linarith) at hx, -- Excludes the case x = -3
  cases (lt_or_gt_of_ne (ne_of_gt (by linarith.min)))
  case inl => rw if_neg (by linarith) at hx -- Case when x < -3
  case inr => rw if_pos (by linarith) at hx -- Case when x > -3
  -- Both cases imply contradiction as shown in the problem solution
  contradiction,
end

end no_zero_in_range_l670_670867


namespace find_range_of_m_l670_670984

-- Define properties of ellipses and hyperbolas
def isEllipseY (m : ℝ) : Prop := (8 - m > 2 * m - 1 ∧ 2 * m - 1 > 0)
def isHyperbola (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- The range of 'm' such that (p ∨ q) is true and (p ∧ q) is false
def p_or_q_true_p_and_q_false (m : ℝ) : Prop := 
  (isEllipseY m ∨ isHyperbola m) ∧ ¬ (isEllipseY m ∧ isHyperbola m)

-- The range of the real number 'm'
def range_of_m (m : ℝ) : Prop := 
  (-1 < m ∧ m ≤ 1/2) ∨ (2 ≤ m ∧ m < 3)

-- Prove that the above conditions imply the correct range for m
theorem find_range_of_m (m : ℝ) : p_or_q_true_p_and_q_false m → range_of_m m :=
by
  sorry

end find_range_of_m_l670_670984


namespace four_digit_numbers_count_four_digit_numbers_divisible_by_3_count_eightyfifth_number_in_sequence_l670_670967

-- Define the set of digits
def digits := {0, 1, 2, 3, 4, 5}

-- Define the condition for forming a four-digit number without repetition
def is_four_digit_number (n : Nat) : Prop :=
  let digits_list := List.ofNat n;
  n >= 1000 ∧ n < 10000 ∧ -- four-digit
  digits_list.all (λ d => d ∈ digits) ∧ -- all digits from the set
  digits_list.nodup -- no repetition

-- Prove that the number of different four-digit numbers is 300
theorem four_digit_numbers_count : {n : Nat // is_four_digit_number n}.card = 300 :=
by sorry

-- Define the condition for a number to be divisible by 3
def is_divisible_by_3 (n : Nat) : Prop :=
  (List.ofNat n).sum % 3 = 0

-- Prove that the number of four-digit numbers divisible by 3 is 96
theorem four_digit_numbers_divisible_by_3_count : {n : Nat // is_four_digit_number n ∧ is_divisible_by_3 n}.card = 96 :=
by sorry

-- Prove that the 85th number in ascending order is 2301
theorem eightyfifth_number_in_sequence : 
  List.sort (λ a b => List.lex (<) (List.ofNat a) (List.ofNat b)) (List.filter is_four_digit_number (range 1000 10000)).nth 84 = 2301 :=
by sorry

end four_digit_numbers_count_four_digit_numbers_divisible_by_3_count_eightyfifth_number_in_sequence_l670_670967


namespace equation_of_line_equation_of_circle_equation_of_tangent_l670_670313

-- Given points A(2, 1) and B(6, 3), prove the equation of line l.
theorem equation_of_line : ∃ l : ℝ → ℝ, ∀ P : ℝ × ℝ, (P = (2, 1) ∨ P = (6, 3)) → P.2 = l P.1 :=
begin
  use (λ x, (1/2) * x),
  intros P hP,
  cases hP; 
  simp [hP],
end

-- Given the center of circle C lies on line l (y = 1/2 x) and it is tangent to the x-axis at (2, 0),
-- prove the equation of circle C.
theorem equation_of_circle : ∃ C : ℝ × ℝ → ℝ, ∀ P : ℝ × ℝ, (P.2 = 1/2 * P.1) ∧ (C (2, 0) = 0) → 
  C P = (P.1 - 2)^2 + P.2^2 - 2 * P.2 :=
begin
  use (λ P, (P.1 - 2)^2 + P.2^2 - 2 * P.2),
  intros P hP,
  cases hP with hp1 hp2,
  simp [hp2],
  sorry
end

-- Given points S and T where line ST is tangent to circle C at point B(6, 3),
-- prove the equation of line ST.
theorem equation_of_tangent : ∃ st : ℝ → ℝ, ∀ P : ℝ × ℝ, (P = (2, 0) ∨ P = (6, 3)) → P.2 = st P.1 :=
begin
  use (λ x, -2 * x + 11/2),
  intros P hP,
  cases hP; 
  simp [hP],
  sorry
end

end equation_of_line_equation_of_circle_equation_of_tangent_l670_670313


namespace common_difference_arithmetic_sequence_l670_670038

noncomputable def first_term : ℕ := 5
noncomputable def last_term : ℕ := 50
noncomputable def sum_terms : ℕ := 275

theorem common_difference_arithmetic_sequence :
  ∃ d n, (last_term = first_term + (n - 1) * d) ∧ (sum_terms = n * (first_term + last_term) / 2) ∧ d = 5 :=
  sorry

end common_difference_arithmetic_sequence_l670_670038


namespace routes_from_P_to_Q_l670_670542

-- Define the points and connections
inductive Point
| P | Q | R | S | T | U

open Point

-- Define the routes in terms of a relation
def route : Point → Point → Prop
| P, R => true
| P, S => true
| P, Q => true
| R, T => true
| R, U => true
| S, T => true
| S, U => true
| S, Q => true
| T, Q => true
| U, Q => true
| _, _ => false

-- Define a function to count the routes
def count_routes : Point → Point → ℕ
| P, Q => (if route P R then count_routes R Q else 0) + 
          (if route P S then count_routes S Q else 0) + 
          (if route P Q then 1 else 0)
| R, Q => (if route R T then count_routes T Q else 0) + 
          (if route R U then count_routes U Q else 0)
| S, Q => (if route S T then count_routes T Q else 0) + 
          (if route S U then count_routes U Q else 0) + 
          (if route S Q then 1 else 0)
| T, Q => 1
| U, Q => 1
| _, _ => 0

-- Main theorem statement
theorem routes_from_P_to_Q : count_routes P Q = 6 := by
  sorry

end routes_from_P_to_Q_l670_670542


namespace existential_proposition_l670_670932

theorem existential_proposition :
  (∃ x y : ℝ, x + y > 1) ∧ (∀ P : Prop, (∃ x y : ℝ, x + y > 1 → P) → P) :=
sorry

end existential_proposition_l670_670932


namespace sum_of_powers_is_odd_l670_670373

theorem sum_of_powers_is_odd : 
  (even (2 ^ 1990)) ∧ (odd (3 ^ 1990)) ∧ (odd (7 ^ 1990)) ∧ (odd (9 ^ 1990)) → 
  odd (2 ^ 1990 + 3 ^ 1990 + 7 ^ 1990 + 9 ^ 1990): 
by
  sorry

end sum_of_powers_is_odd_l670_670373


namespace cube_with_pyramid_sum_l670_670466

theorem cube_with_pyramid_sum : 
  let cube_faces := 6 in
  let cube_edges := 12 in
  let cube_vertices := 8 in
  let pyramid_faces := 4 in
  let pyramid_base_face := 1 in
  let pyramid_edges := 4 in
  let pyramid_vertices := 1 in
  let total_faces := cube_faces - pyramid_base_face + pyramid_faces in
  let total_edges := cube_edges + pyramid_edges in
  let total_vertices := cube_vertices + pyramid_vertices in
  total_faces + total_edges + total_vertices = 34 :=
by
  sorry

end cube_with_pyramid_sum_l670_670466


namespace sum_even_integers_302_to_400_l670_670974

theorem sum_even_integers_302_to_400 : 
  let even_numbers := {n | 302 ≤ n ∧ n ≤ 400 ∧ n % 2 = 0} in
  ∑ k in even_numbers, k = 17550 :=
by
  sorry

end sum_even_integers_302_to_400_l670_670974


namespace min_value_four_x_plus_one_over_x_l670_670310

theorem min_value_four_x_plus_one_over_x (x : ℝ) (hx : x > 0) : 4*x + 1/x ≥ 4 := by
  sorry

end min_value_four_x_plus_one_over_x_l670_670310


namespace loss_percent_l670_670253

theorem loss_percent (cost_price selling_price loss_percent : ℝ) 
  (h_cost_price : cost_price = 600)
  (h_selling_price : selling_price = 550)
  (h_loss_percent : loss_percent = 8.33) : 
  (loss_percent = ((cost_price - selling_price) / cost_price) * 100) := 
by
  rw [h_cost_price, h_selling_price]
  sorry

end loss_percent_l670_670253


namespace chess_games_total_l670_670937

theorem chess_games_total (num_players : ℕ) (plays_each_other_twice : ∀ p1 p2, p1 ≠ p2 → (#games p1 p2) = 2)
  (each_game_two_players : ∀ g, (∃ p1 p2, g ∈ {p1, p2})) :
  num_players = 20 → total_number_of_games = 380 :=
by
  sorry

end chess_games_total_l670_670937


namespace proposition_false_at_4_l670_670233

open Nat

def prop (n : ℕ) : Prop := sorry -- the actual proposition is not specified, so we use sorry

theorem proposition_false_at_4 :
  (∀ k : ℕ, k > 0 → (prop k → prop (k + 1))) →
  ¬ prop 5 →
  ¬ prop 4 :=
by
  intros h_induction h_proposition_false_at_5
  sorry

end proposition_false_at_4_l670_670233
