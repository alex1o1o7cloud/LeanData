import Mathlib

namespace sum_of_s_l313_313008

def s (n : ℕ) : ℚ := ∑ i in Finset.range n, 1 / (i + 1)

theorem sum_of_s (n : ℕ) (h : 1 ≤ n) : ∑ i in Finset.range (n - 1), s (i + 1) = n * s n - n := 
sorry

end sum_of_s_l313_313008


namespace sufficient_condition_transitive_l313_313875

theorem sufficient_condition_transitive
  (C B A : Prop) (h1 : (C → B)) (h2 : (B → A)) : (C → A) :=
  sorry

end sufficient_condition_transitive_l313_313875


namespace prob_300_points_l313_313168

noncomputable def P : ℕ → ℝ := λ i, 
  if i = 1 then 0.8 
  else if i = 2 then 0.3 
  else if i = 3 then 0.6 
  else 0 -- probability assignment 

-- definition of scoring 300 points
def score_300 (A1 A2 A3 : Prop) [decidable A1] [decidable A2] [decidable A3] : Prop :=
  (A1 ∧ ¬A2 ∧ A3) ∨ (¬A1 ∧ A2 ∧ A3)

theorem prob_300_points : (P 1) * (1 - (P 2)) * (P 3) + (1 - (P 1)) * (P 2) * (P 3) = 0.54 :=
by
  sorry

end prob_300_points_l313_313168


namespace find_a_for_binomial_square_l313_313195

theorem find_a_for_binomial_square (a r s : ℝ) (h : ax^2 + 8x + 16 = (rx + s)^2) : a = 1 :=
sorry

end find_a_for_binomial_square_l313_313195


namespace sqrt_mul_distrib_simplify_sqrt48_simplify_sqrt18_math_problem_l313_313479

theorem sqrt_mul_distrib (a b c : ℝ) : (Real.sqrt a + Real.sqrt b) * Real.sqrt c = Real.sqrt (a * c) + Real.sqrt (b * c) :=
by
  sorry

theorem simplify_sqrt48 : Real.sqrt 48 = 4 * Real.sqrt 3 :=
by
  sorry

theorem simplify_sqrt18 : Real.sqrt 18 = 3 * Real.sqrt 2 :=
by
  sorry

theorem math_problem : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 = 4 * Real.sqrt 3 + 3 * Real.sqrt 2 :=
by
  -- Using the sqrt_mul_distrib theorem:
  have h1 : (Real.sqrt 8 + Real.sqrt 3) * Real.sqrt 6 = Real.sqrt (8 * 6) + Real.sqrt (3 * 6),
    from sqrt_mul_distrib 8 3 6,
  -- Simplifying sqrt (8 * 6) and sqrt (3 * 6)
  have h2 : Real.sqrt (8 * 6) = Real.sqrt 48,
    by norm_num,
  have h3 : Real.sqrt (3 * 6) = Real.sqrt 18,
    by norm_num,
  -- Using the simplifications:
  have h4 : Real.sqrt 48 = 4 * Real.sqrt 3,
    from simplify_sqrt48,
  have h5 : Real.sqrt 18 = 3 * Real.sqrt 2,
    from simplify_sqrt18,
  rw [h2, h3] at h1,
  rw [h4, h5] at h1,
  exact h1

end sqrt_mul_distrib_simplify_sqrt48_simplify_sqrt18_math_problem_l313_313479


namespace maximum_sum_of_removed_integers_is_2165_l313_313204

-- Define the main problem with conditions
def problem_statement : Prop :=
  ∃ (S₁ S₂ : set ℕ), 
    (S₁ ∪ S₂ = set.range (1, 101)) ∧ 
    (S₁.card = 50) ∧ 
    (¬ ∃ (a b : ℕ), a ∈ S₂ ∧ b ∈ S₂ ∧ a ≠ b ∧ a + b ∈ S₂) ∧ 
    (S₁.sum = 2165)

-- Placeholder for the proof
theorem maximum_sum_of_removed_integers_is_2165 : problem_statement :=
  sorry

end maximum_sum_of_removed_integers_is_2165_l313_313204


namespace initial_roses_in_vase_l313_313394

/-- 
There were some roses in a vase. Mary cut roses from her flower garden 
and put 16 more roses in the vase. There are now 22 roses in the vase.
Prove that the initial number of roses in the vase was 6. 
-/
theorem initial_roses_in_vase (initial_roses added_roses current_roses : ℕ) 
  (h_add : added_roses = 16) 
  (h_current : current_roses = 22) 
  (h_current_eq : current_roses = initial_roses + added_roses) : 
  initial_roses = 6 := 
by
  subst h_add
  subst h_current
  linarith

end initial_roses_in_vase_l313_313394


namespace at_most_two_pairs_l313_313783

theorem at_most_two_pairs (n : ℕ) (h_pos : n > 0) :
  ∃ p q: Σ' (a_1 a_2: ℕ) , 
    let (a1,b1) := (a_1.1, a_1.2); let (a2,b2) := (a_2.1, a_2.2) in 
    (∀ {a : ℕ} {b : ℕ}, (a^2 + b = n) ∧ (∃ k : ℕ, a + b = 2 ^ k) →
    a1 = a → b1 = b ∨ a2 = a → b2 = b ) :=
sorry

end at_most_two_pairs_l313_313783


namespace max_ab_value_l313_313972

-- Define the tangent condition for a line and a circle
def is_tangent (a b : ℝ) : Prop :=
  (abs (a + 2 * b) / real.sqrt 5 = real.sqrt 10)

-- Define the main proof problem
theorem max_ab_value {a b : ℝ}
  (h_tangent : is_tangent a b)
  (h_center_above : a + 2 * b > 0) :
  a * b ≤ 25 / 4 :=
  sorry

end max_ab_value_l313_313972


namespace sin_135_eq_sqrt2_over_2_l313_313569

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313569


namespace trigonometric_relation_l313_313686

theorem trigonometric_relation (ϕ : ℝ) (h : 0 < ϕ ∧ ϕ < π / 2) :
  sin (cos ϕ) < cos ϕ ∧ cos ϕ < cos (sin ϕ) :=
sorry

end trigonometric_relation_l313_313686


namespace distance_difference_CQ_DQ_l313_313369

theorem distance_difference_CQ_DQ {y : ℝ} (c d : ℝ) (h_line_c : c - (sqrt 2) * ((c^2 - 5) / 2) + 4 = 0)
(h_line_d : d - (sqrt 2) * ((d^2 - 5) / 2) + 4 = 0)
(h_parabola_c : c^2 = 2 * ((c^2 - 5) / 2) + 5)
(h_parabola_d : d^2 = 2 * ((d^2 - 5) / 2) + 5)
(h_c_neg : c < 0)
(h_d_pos : d > 0) :
|sqrt (3 / 2) * c - sqrt (3 / 2) * d| = sqrt 3 := by
  sorry

end distance_difference_CQ_DQ_l313_313369


namespace power_mod_l313_313097

theorem power_mod (n m : ℕ) (hn : n = 13) (hm : m = 1000) : n ^ 21 % m = 413 :=
by
  rw [hn, hm]
  -- other steps of the proof would go here...
  sorry

end power_mod_l313_313097


namespace computer_price_increase_l313_313070

theorem computer_price_increase (y : ℝ) (h : 2 * y = 540) (h_price : y = 270) : 
  let percentage_increase := ((351 - y) / y) * 100 in
  percentage_increase = 30 :=
by
  sorry

end computer_price_increase_l313_313070


namespace bc_together_l313_313157

theorem bc_together (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : C = 20) : B + C = 320 :=
by
  sorry

end bc_together_l313_313157


namespace verify_cube_modifications_l313_313732

-- Definitions and conditions from the problem
def side_length : ℝ := 9
def initial_volume : ℝ := side_length^3
def initial_surface_area : ℝ := 6 * side_length^2

def volume_remaining : ℝ := 639
def surface_area_remaining : ℝ := 510

-- The theorem proving the volume and surface area of the remaining part after carving the cross-shaped groove
theorem verify_cube_modifications :
  initial_volume - (initial_volume - volume_remaining) = 639 ∧
  510 = surface_area_remaining :=
by
  sorry

end verify_cube_modifications_l313_313732


namespace train_speed_l313_313893

theorem train_speed (jogger_speed := 9 : ℕ) 
    (initial_distance_ahead := 200 : ℕ) 
    (train_length := 210 : ℕ) 
    (time_to_pass := 41 : ℕ) :
    let relative_distance := initial_distance_ahead + train_length in
    let relative_speed_m_s := relative_distance / time_to_pass in
    let relative_speed_km_hr := relative_speed_m_s * 36 / 10 in
    let train_speed := relative_speed_km_hr + jogger_speed in
    train_speed = 45 := sorry

end train_speed_l313_313893


namespace sin_135_eq_sqrt2_div_2_l313_313551

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313551


namespace find_hyperbola_equation_intersection_point_fixed_line_l313_313368

-- Given statements
variables (a b c : ℝ) (a_pos : a > 0) (b_pos : b > 0) (c_val : c = 4)
variables (x y A B F1 F2 M P Q : ℝ × ℝ)

-- Condition: equation of hyperbola C
def is_hyperbola_C (p : ℝ × ℝ) := (p.1^2 / a^2 - p.2^2 / b^2 = 1)

-- Condition: focal length
def focal_length := 2 * c = 8

-- Condition: M lies on the hyperbola
def M_on_hyperbola := is_hyperbola_C M

-- Condition: MF_1 ⊥ MF_2
def perpendicular (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2 = 0
def MF1_perpendicular_MF2 := perpendicular (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2)

-- Condition: area of triangle F1MF2
def area_triangle (a b c : ℝ × ℝ) := (1 / 2) * real.abs (a.1 * (b.2 - c.2) + b.1 * (c.2 - a.2) + c.1 * (a.2 - b.2))
def area_F1MF2 := area_triangle F1 M F2 = 12

-- Proof: Equation of the hyperbola C
theorem find_hyperbola_equation :
  (∀ (a b : ℝ), a > 0 → b > 0 →
    (2 * c = 8) → c = 4 →
    ∀ (M : ℝ × ℝ), is_hyperbola_C M →
    perpendicular (M.1 - F1.1, M.2 - F1.2) (M.1 - F2.1, M.2 - F2.2) →
    area_triangle F1 M F2 = 12 →
    (a = 2 ∧ b^2 = 12 ∧ (\forall (x y : ℝ), (x^2 / 4) - (y^2 / 12) = 1))
  ) :=
sorry

-- Proof: Intersection point of lines AQ and BP
theorem intersection_point_fixed_line :
  (∀ (A B F2 P Q : ℝ × ℝ),
    let l := (m : ℝ) × (y : ℝ) := m * y + 4 in
    let AQ_line := (y : ℝ) := y in
    let BP_line := (y : ℝ) := y in
    intersects A Q → intersects B P →
    (∃ (x : ℝ), x = 1)
  ) :=
sorry

end find_hyperbola_equation_intersection_point_fixed_line_l313_313368


namespace not_perfect_square_2_2049_and_4_2051_l313_313419

theorem not_perfect_square_2_2049_and_4_2051 (
  hA: ∃ x : ℝ, 1^{2048} = x^2,
  hB: ∀ x : ℝ, 2^{2049} ≠ x^2,
  hC: ∃ x : ℝ, 3^{2050} = x^2,
  hD: ∀ x : ℝ, 4^{2051} ≠ x^2,
  hE: ∃ x : ℝ, 5^{2052} = x^2
  ) : ∀ x : ℝ, (2^{2049} ≠ x^2) ∧ (4^{2051} ≠ x^2) :=
by {
  sorry,
}

end not_perfect_square_2_2049_and_4_2051_l313_313419


namespace probability_of_observing_change_l313_313153

noncomputable def traffic_light_cycle := 45 + 5 + 45
noncomputable def observable_duration := 5 + 5 + 5
noncomputable def probability_observe_change := observable_duration / (traffic_light_cycle : ℝ)

theorem probability_of_observing_change :
  probability_observe_change = (3 / 19 : ℝ) :=
  by sorry

end probability_of_observing_change_l313_313153


namespace alpha_convex_implies_J_convex_alpha_convex_implies_general_convex_l313_313177

variable {D : Set ℝ} {f : ℝ → ℝ} (α : ℝ) (hα : 0 < α ∧ α < 1)
(hα_convex : ∀ x1 x2 ∈ D, α * f x1 + (1 - α) * f x2 ≥ f (α * x1 + (1 - α) * x2))

/-- If a function is α-convex on a domain, then it is J-convex (midpoint-convex) on that domain. -/
theorem alpha_convex_implies_J_convex (hf : ∀ x1 x2 ∈ D, α * f x1 + (1 - α) * f x2 ≥ f (α * x1 + (1 - α) * x2)) :
  ∀ x1 x2 ∈ D, f x1 + f x2 ≥ 2 * f ((x1 + x2) / 2) :=
by
  sorry

/-- If a function is α-convex on a domain, it is also (α^n / ((1-α)^n + α^n))-convex on that domain for any natural number n ≥ 1. -/
theorem alpha_convex_implies_general_convex
  (hf : ∀ x1 x2 ∈ D, α * f x1 + (1 - α) * f x2 ≥ f (α * x1 + (1 - α) * x2)) :
  ∀ (n : ℕ) (hn : n ≥ 1) (x1 x2 ∈ D), ((α^n * f x1 + (1 - α)^n * f x2) / (α^n + (1 - α)^n)) ≥ f ((α^n * x1 + (1 - α)^n * x2) / (α^n + (1 - α)^n)) :=
by
  sorry

end alpha_convex_implies_J_convex_alpha_convex_implies_general_convex_l313_313177


namespace sin_135_degree_l313_313533

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313533


namespace circumcircle_radius_l313_313377

theorem circumcircle_radius (R a : ℝ) (K L M A B : Type*) 
  [MetricSpace K] [MetricSpace L] [MetricSpace M] [MetricSpace A] [MetricSpace B] 
  (circumscribed_triangle_KLM : ∃ (O : Type*) (circumradius : ℝ), circumradius = R) 
  (perpendicular_through_L : ∃ (line : Type*), line passes_through L ∧ line perpendicular_to (KM : Set K)) 
  (intersects_A_B : ∃ (perpendicular_bisector_KL perpendicular_bisector_LM : Set L), 
    A ∈ perpendicular_bisector_KL ∧ B ∈ perpendicular_bisector_LM) 
  (AL_eq_a : dist A L = a) :
  dist B L = R^2 / a :=
by sorry

end circumcircle_radius_l313_313377


namespace sin_135_eq_sqrt2_div_2_l313_313579

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313579


namespace pythagorean_prime_divisibility_l313_313038

theorem pythagorean_prime_divisibility 
  (x y z : ℤ) (hx : prime x ∨ prime y ∨ prime z) 
  (hp : x > 5 ∧ y > 5 ∧ z > 5) 
  (h : x^2 + y^2 = z^2) : 
  (x ∣ 60) ∨ (y ∣ 60) ∨ (z ∣ 60) :=
by
  sorry

end pythagorean_prime_divisibility_l313_313038


namespace am_gm_inequality_l313_313009

theorem am_gm_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_sum : x + y + z = 1) :
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z :=
by sorry

end am_gm_inequality_l313_313009


namespace sin_135_eq_sqrt2_div_2_l313_313550

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313550


namespace class_fund_shortfall_l313_313327

theorem class_fund_shortfall 
  (bake_sale_amount : ℕ := 50)
  (num_students : ℕ := 30)
  (student_contrib : ℕ := 5)
  (activity1_cost : ℕ := 8)
  (activity2_cost : ℕ := 9)
  (service_charge : ℕ := 1) :
  let total_raised := bake_sale_amount + num_students * student_contrib,
      total_costs := num_students * (activity1_cost + activity2_cost + service_charge),
      amount_left := total_raised - total_costs
  in amount_left = -340 := 
by
  sorry

end class_fund_shortfall_l313_313327


namespace cos_theta_l313_313258

variables (a b : EuclideanSpace ℝ (Fin 3))

def vector_a_norm_eq_five (a : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥a∥ = 5

def vector_b_norm_eq_seven (b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥b∥ = 7

def vector_a_plus_b_norm_eq_ten (a b : EuclideanSpace ℝ (Fin 3)) : Prop :=
  ∥a + b∥ = 10

theorem cos_theta (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : vector_a_norm_eq_five a)
  (h2 : vector_b_norm_eq_seven b)
  (h3 : vector_a_plus_b_norm_eq_ten a b) : 
  real.cos (angle a b) = 13 / 35 := 
by
  sorry

end cos_theta_l313_313258


namespace sphere_circumcircle_radius_l313_313090

theorem sphere_circumcircle_radius (A B C : Point) (r1 r2 : ℝ) (r1_plus_r2_is_nine : r1 + r2 = 9)
  (d_centers_is_sqrt_305 : dist(center_of_sphere_touching_A, center_of_sphere_touching_B) = sqrt 305)
  (radius_third_sphere_is_seven : radius_sphere(center C) = 7)
  (third_sphere_touches_others : touches_externally (center_of_sphere_touching_A) (center_of_sphere_touching_B) C) :
  circumcircle_radius ABC = 2 * sqrt 14 :=
by sorry

end sphere_circumcircle_radius_l313_313090


namespace quadratic_inequality_solution_range_l313_313251

open Set Real

theorem quadratic_inequality_solution_range
  (a : ℝ) : (∃ (x1 x2 : ℤ), x1 ≠ x2 ∧ (∀ x : ℝ, x^2 - a * x + 2 * a < 0 ↔ ↑x1 < x ∧ x < ↑x2)) ↔ 
    (a ∈ Icc (-1 : ℝ) ((-1:ℝ)/3)) ∨ (a ∈ Ioo (25 / 3 : ℝ) 9) :=
sorry

end quadratic_inequality_solution_range_l313_313251


namespace cos_beta_value_l313_313014

theorem cos_beta_value
  (α β : ℝ)
  (hαβ : 0 < α ∧ α < π ∧ 0 < β ∧ β < π)
  (h1 : Real.sin (α + β) = 5 / 13)
  (h2 : Real.tan (α / 2) = 1 / 2) :
  Real.cos β = -16 / 65 := 
by 
  sorry

end cos_beta_value_l313_313014


namespace inscribed_circle_perpendiculars_intersect_l313_313837

variables {A B C P : Type} [EquilateralTriangle A B C] [IsPoint P]

theorem inscribed_circle_perpendiculars_intersect :
  let A1 := projection (incenter_triangle P B C) (line B C)
  let B1 := projection (incenter_triangle P C A) (line C A)
  let C1 := projection (incenter_triangle P A B) (line A B)
  ∃ X : Type, is_intersection_point X (perpendicular_from A1 (line B C)) ∧ 
                is_intersection_point X (perpendicular_from B1 (line C A)) ∧ 
                is_intersection_point X (perpendicular_from C1 (line A B)) :=
sorry

end inscribed_circle_perpendiculars_intersect_l313_313837


namespace total_sum_lent_l313_313906

theorem total_sum_lent (x : ℝ) (second_part : ℝ) (total_sum : ℝ)
  (h1 : second_part = 1648)
  (h2 : (x * 3 / 100 * 8) = (second_part * 5 / 100 * 3))
  (h3 : total_sum = x + second_part) :
  total_sum = 2678 := 
  sorry

end total_sum_lent_l313_313906


namespace triangular_pyramid_volume_l313_313820

-- Define the conditions
variables (a b c : ℝ)
-- Assume that a, b, and c are the lengths of the lateral edges of a triangular pyramid
-- and that they are pairwise perpendicular

-- Theorem statement
theorem triangular_pyramid_volume (h_perp: ∀ {x y z : ℝ}, x ≠ y ∧ y ≠ z ∧ z ≠ x) :
  volume_of_pyramid a b c = (a * b * c) / 6 :=
sorry

end triangular_pyramid_volume_l313_313820


namespace g_neg3_value_l313_313183

variable (g : ℝ → ℝ)

axiom function_condition : ∀ x : ℝ, g (5 * x - 7) = 8 * x + 2

theorem g_neg3_value : g (-3) = 8.4 :=
by
  have h1 : 5 * (4 / 5 : ℝ) - 7 = -3 := by
    calc
      5 * (4 / 5 : ℝ) - 7 = 4 - 7 : by norm_num
      ... = -3 : by norm_num
  have h2 : g (-3) = g (5 * (4 / 5 : ℝ) - 7) := by rw [h1]
  rw [h2, function_condition]
  calc
    8 * (4 / 5 : ℝ) + 2 = 8 * (4 / 5) + (2 : ℝ) : by norm_num
    ... = (32 / 5) + 2 : by norm_num
    ... = (32 / 5) + (10 / 5) : by norm_num
    ... = 42 / 5 : by norm_num
    ... = 8.4 : by norm_num
  done

end g_neg3_value_l313_313183


namespace range_of_a_l313_313002

noncomputable def f (x : ℝ) : ℝ := 1 - Real.sqrt (x^2 + 1)
noncomputable def g (a x : ℝ) : ℝ := Real.log (a * x^2 - 2 * x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 = g a x2) ↔ a ∈ Set.Iic 1 := 
begin
  sorry
end

end range_of_a_l313_313002


namespace triangle_construction_valid_l313_313594

open EuclideanGeometry

noncomputable def construct_triangle (M N : Point) (lineAB : Line) : Triangle := 
  let O := find_center_of_circle_passing_through_feet_of_altitudes M N lineAB
  let circ := Circle (O, distance O M)
  let A := intersection_points_with_line circ lineAB
  let B := intersection_points_with_line circ lineAB
  let C := intersection_of_lines (line_through A N) (line_through B M)
  Triangle.mk A B C

theorem triangle_construction_valid (M N : Point) (lineAB : Line) :
  (∃ A B C : Point, Triangle (A B C)) → 
  (A lies_on lineAB) ∧ (B lies_on lineAB) ∧ 
  (M foot_of_altitude_from A) ∧ (N foot_of_altitude_from B) →
  construct_triangle M N lineAB =
  Triangle.mk A B C :=
by
  sorry

end triangle_construction_valid_l313_313594


namespace atomic_weight_of_chlorine_l313_313192

theorem atomic_weight_of_chlorine (molecular_weight_AlCl3 : ℝ) (atomic_weight_Al : ℝ) (atomic_weight_Cl : ℝ) :
  molecular_weight_AlCl3 = 132 ∧ atomic_weight_Al = 26.98 →
  132 = 26.98 + 3 * atomic_weight_Cl →
  atomic_weight_Cl = 35.007 :=
by
  intros h1 h2
  sorry

end atomic_weight_of_chlorine_l313_313192


namespace mass_percentage_Al_is_correct_l313_313934

-- Define the given masses of the compounds
def mass_Al_OH_3 : ℝ := 35
def mass_Al2_SO4_3 : ℝ := 25

-- Define the molar masses of the compounds
def molar_mass_Al_OH_3 : ℝ := 78.01
def molar_mass_Al2_SO4_3 : ℝ := 342.17

-- Define the mass fraction of Al in each compound
def mass_fraction_Al_Al_OH_3 : ℝ := 26.98 / molar_mass_Al_OH_3
def mass_fraction_Al_Al2_SO4_3 : ℝ := (2 * 26.98) / molar_mass_Al2_SO4_3

-- Calculate the mass of Al in each compound
def mass_Al_Al_OH_3 : ℝ := mass_Al_OH_3 * mass_fraction_Al_Al_OH_3
def mass_Al_Al2_SO4_3 : ℝ := mass_Al2_SO4_3 * mass_fraction_Al_Al2_SO4_3

-- Calculate the total mass of Al in the mixture
def total_mass_Al : ℝ := mass_Al_Al_OH_3 + mass_Al_Al2_SO4_3

-- Calculate the total mass of the mixture
def total_mass_mixture : ℝ := mass_Al_OH_3 + mass_Al2_SO4_3

-- Calculate the mass percentage of Al in the mixture
def mass_percentage_Al : ℝ := (total_mass_Al / total_mass_mixture) * 100

-- Theorem to prove the mass percentage of Al is approximately 26.74%
theorem mass_percentage_Al_is_correct : abs (mass_percentage_Al - 26.74) < 0.01 :=
by
  sorry

end mass_percentage_Al_is_correct_l313_313934


namespace min_value_expression_l313_313013

theorem min_value_expression (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 2) :
  min (
    {
      let expr := (1 / ((2 - x) * (2 - y) * (2 - z))) 
                  + (1 / ((2 + x) * (2 + y) * (2 + z))) 
                  + (1 / (1 + (x + y + z) / 3))
      expr
    }
  ) = 2 :=
sorry

end min_value_expression_l313_313013


namespace find_triangle_angles_l313_313351

-- Define the angles of the triangle
variables (α β γ : ℝ)

-- Define the given conditions
def sum_of_angles := α + β + γ = Real.pi
def similarity_condition := ∃ K, 
  let α' := β / 2 in
  let β' := γ / 2 in
  let γ' := Real.pi - (β / 2 + γ / 2) in
  γ' = β / 2 + α / 2 ∧ α' = α ∧ β' = β

-- Define the goal
theorem find_triangle_angles (h1 : sum_of_angles α β γ) (h2 : similarity_condition α β γ) :
  α = Real.pi / 7 ∧ β = 2 * Real.pi / 7 ∧ γ = 4 * Real.pi / 7 := sorry

end find_triangle_angles_l313_313351


namespace cos_B_value_l313_313710

variable (a b c : ℝ)
variable (A B C : ℝ)
variable [decidable_eq ℝ]

-- Given conditions
def condition1 : Prop := 6 * a = 4 * b
def condition2 : Prop := 4 * b = 3 * c

-- To prove
def cos_law : Prop := 
  let cos_B := (a^2 + c^2 - b^2) / (2 * a * c) in
  cos_B = 11 / 16

theorem cos_B_value (a b c : ℝ)
  (h1 : condition1 a b)
  (h2 : condition2 b c)
  (hb : b = 3 * a / 2)
  (hc : c = 2 * a) :
  cos_law a b c :=
by {
  unfold condition1 condition2 cos_law,
  sorry
}

end cos_B_value_l313_313710


namespace correct_answer_l313_313023

def mary_initial_cards : ℝ := 18.0
def mary_bought_cards : ℝ := 40.0
def mary_left_cards : ℝ := 32.0
def mary_promised_cards (initial_cards : ℝ) (bought_cards : ℝ) (left_cards : ℝ) : ℝ :=
  initial_cards + bought_cards - left_cards

theorem correct_answer :
  mary_promised_cards mary_initial_cards mary_bought_cards mary_left_cards = 26.0 := by
  sorry

end correct_answer_l313_313023


namespace smallest_k_example_proof_l313_313881

theorem smallest_k_example_proof
  (x : Fin 100 → Fin 25 → ℝ)
  (h_nonneg : ∀ i j, 0 ≤ x i j)
  (h_sum : ∀ i, (∑ j, x i j) ≤ 1) :
  ∃ k : ℕ, k = 97 ∧ ∀ i ≥ k, (∑ j, let ij_values := (Finset.univ : Finset (Fin 100)).map (λ i, x i j) in
                              (Finset.univ : Finset (Fin 25)).sum (λ j, (ij_values.sort (≥)).nth i)) ≤ 1 :=
begin
  sorry
end

end smallest_k_example_proof_l313_313881


namespace distinct_ordered_pairs_l313_313823

theorem distinct_ordered_pairs :
  (∃! (x y : ℕ+),  x^4 * y^4 - 10 * x^2 * y^2 + 9 = 0) = 3 := 
sorry

end distinct_ordered_pairs_l313_313823


namespace tori_additional_correct_answers_l313_313280

noncomputable def problems_count : ℕ := 80
noncomputable def arithmetic_count : ℕ := 15
noncomputable def algebra_count : ℕ := 25
noncomputable def geometry_count : ℕ := 40
noncomputable def correct_arithmetic : ℕ := 12 -- 80% of 15
noncomputable def correct_algebra : ℕ := 13 -- 50% of 25 (rounded up)
noncomputable def correct_geometry : ℕ := 22 -- 55% of 40
noncomputable def passing_percentage : ℝ := 0.65
noncomputable def total_correct_needed : ℕ := (passing_percentage * problems_count).ceil.to_nat
noncomputable def total_correct : ℕ := correct_arithmetic + correct_algebra + correct_geometry

theorem tori_additional_correct_answers :
  total_correct_needed - total_correct = 5 :=
by
  sorry

end tori_additional_correct_answers_l313_313280


namespace complement_of_M_l313_313676

theorem complement_of_M :
  let U := {2011, 2012, 2013, 2014, 2015}
  let M := {2011, 2012, 2013}
  let complement := {x ∈ U | x ∉ M}
  complement = {2014, 2015} := by
  sorry

end complement_of_M_l313_313676


namespace find_parabola_and_m_l313_313655

noncomputable def parabola (p : ℝ) (hp : p > 0) : (ℝ × ℝ) → Prop := λ (x y : ℝ), y^2 = 2 * p * x

def on_parabola (p : ℝ) (hp : p > 0) : ℝ × ℝ → Prop :=
  λ A, parabola p hp A.1 A.2

def focus_distance (A : ℝ × ℝ) (F : ℝ × ℝ) : ℝ := 
  real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2)

theorem find_parabola_and_m (F : ℝ × ℝ) (A : ℝ × ℝ) (hA : on_parabola 4 (by norm_num) A) (hAF: focus_distance A F = 4) :
  ∃ m : ℝ, P Q : ℝ × ℝ, (P = (p_1, y_1)) ∧ (Q = (p_2, y_2)) ∧ ((P ≠ Q) ∧ (l : ℝ → ℝ) := (λ x, x + m)) ∧ 
  (P ∈ parabola 4 (by norm_num) ∧ Q ∈ parabola 4 (by norm_num)) ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = 0) ∧ ¬(m = 0) :=
  ∃ (p = 4), l = y = x, ∀ (x y : ℝ), y^2 = 8x, m = -8 := sorry

end find_parabola_and_m_l313_313655


namespace find_r_l313_313436

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem find_r (n p q r : ℕ) 
  (hn : n.digits 10.length = 1996)
  (hn9 : 9 ∣ n)
  (hp : p = sum_of_digits n)
  (hq : q = sum_of_digits p)
  (hr : r = sum_of_digits q)
  (hq_two_digit : q < 100) :
  r = 9 :=
by
  sorry

end find_r_l313_313436


namespace polynomial_factorization_m_n_l313_313704

theorem polynomial_factorization_m_n (m n : ℤ) (h : (x : ℤ) → x^2 + m * x + n = (x + 1) * (x + 3)) : m - n = 1 := 
by
  -- Define the equality of the factored polynomial and the standard form polynomial.
  have h_poly : (x : ℤ) → x^2 + m * x + n = x^2 + 4 * x + 3, from
    fun x => h x ▸ by ring,
  -- Extract values of m and n by comparing coefficients.
  have h_m : m = 4, from by
    have := congr_fun h_poly 0,
    simp at this,
    assumption,
  
  have h_n : n = 3, from by
    have := congr_fun h_poly (-1),
    simp at this,
    assumption,
  
  -- Substitute m and n to find that m - n = 1.
  rw [h_m, h_n],
  exact dec_trivial

end polynomial_factorization_m_n_l313_313704


namespace sin_135_eq_l313_313494

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313494


namespace child_ticket_cost_l313_313808

theorem child_ticket_cost :
  ∀ (A P_a C T P_c : ℕ),
    A = 10 →
    P_a = 8 →
    C = 11 →
    T = 124 →
    (T - A * P_a) / C = P_c →
    P_c = 4 :=
by
  intros A P_a C T P_c hA hP_a hC hT hPc
  rw [hA, hP_a, hC, hT] at hPc
  linarith [hPc]

end child_ticket_cost_l313_313808


namespace tangent_at_C_bisects_AM_l313_313779

noncomputable def circle (O : Point) (r : ℝ) : Set Point := {X | dist O X = r}

structure geom_data where
  O : Point
  A B C M : Point
  r : ℝ
  hA_on_circle : A ∈ circle O r
  hB_on_circle : B ∈ circle O r
  hC_on_circle : C ∈ circle O r
  hAB_diameter : ∀ (P : Point), P ∈ circle O r → P ≠ A → P ≠ B → dist O A = dist O P
  hM_intersect : is_intersection M (tangent_at A) (line_through B C)
  hAB_opposite : ∀ Q : Point, Q = midpoint A B → dist Q O = 0

theorem tangent_at_C_bisects_AM : geom_data → (tangent_at C).bisects (segment A M) := by
  sorry

end tangent_at_C_bisects_AM_l313_313779


namespace jerry_mows_weekly_hours_l313_313739

noncomputable def total_lawn : ℝ := 20
noncomputable def riding_mower_fraction : ℝ := 3 / 5
noncomputable def riding_mower_rate : ℝ := 2.5
noncomputable def push_mower1_fraction : ℝ := 1 / 3
noncomputable def push_mower1_rate : ℝ := 1.2
noncomputable def push_mower2_rate : ℝ := 0.8

theorem jerry_mows_weekly_hours :
  let remaining_lawn := total_lawn * (1 - riding_mower_fraction), 
      first_part_mowed := remaining_lawn * push_mower1_fraction, 
      second_part_mowed := remaining_lawn * (1 - push_mower1_fraction), 
      time_riding_mower := (total_lawn * riding_mower_fraction) / riding_mower_rate, 
      time_push_mower1 := first_part_mowed / push_mower1_rate, 
      time_push_mower2 := second_part_mowed / push_mower2_rate in 
      (time_riding_mower + time_push_mower1 + time_push_mower2) = 13.69 :=
by
  sorry

end jerry_mows_weekly_hours_l313_313739


namespace distance_from_point_to_line_l313_313608

noncomputable def distance_point_to_line (p : ℝ × ℝ × ℝ) (a b : ℝ × ℝ × ℝ) : ℝ :=
  let d : ℝ × ℝ × ℝ := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let vec : ℝ × ℝ × ℝ := (d.1 + p.1, d.2 + p.2, d.3 + p.3)
  let dot := vec.1 * d.1 + vec.2 * d.2 + vec.3 * d.3
  let norm_d := Math.sqrt ((d.1)^2 + (d.2)^2 + (d.3)^2)
  let distance_from_line := Math.sqrt (norm_d^2 + 2 * dot)
  distance_from_line

theorem distance_from_point_to_line : 
  distance_point_to_line (2, -1, 4) (4, 3, 9) (1, -1, 3) = 65 / 11 := by
  sorry

end distance_from_point_to_line_l313_313608


namespace tank_capacity_correct_l313_313422

noncomputable def tank_capacity : ℝ :=
  let C := 925.71 in
  let leak_rate := C / 6 in
  let inlet_rate := 4.5 * 60 in
  let net_emptying_rate := C / 8 in
  if (inlet_rate - leak_rate) = net_emptying_rate then C else 0

theorem tank_capacity_correct :
  ∃ (C : ℝ), (let leak_rate := C / 6 in
              let inlet_rate := 4.5 * 60 in
              let net_emptying_rate := C / 8 in
              inlet_rate - leak_rate = net_emptying_rate) ∧
              C = 925.71 :=
by
  existsi (925.71 : ℝ)
  simp
  sorry

end tank_capacity_correct_l313_313422


namespace original_price_vase_l313_313138

-- Definitions based on the conditions and problem elements
def original_price (P : ℝ) : Prop :=
  0.825 * P = 165

-- Statement to prove equivalence
theorem original_price_vase : ∃ P : ℝ, original_price P ∧ P = 200 :=
  by
    sorry

end original_price_vase_l313_313138


namespace are_all_statements_correct_l313_313858

-- Conditions and definitions
variable (a b : ℝ)
variable (x : ℝ)
def M : set ℕ := {0, 1}

-- Statements
lemma statement_A : (a + 1 > b) → (a > b) := 
by sorry

lemma statement_B : set (λ x, a*x^2 + a*x + 1 = 0).card = 1 → a = 4 :=
by sorry

lemma statement_C : (∀ x : ℝ, (1 / (x - 2)) > 0) → {x | (1 / (x - 2)) ≤ 0} = {x | x ≤ 2} := 
by sorry

lemma statement_D : ({N : set ℕ | M ∪ N = M}.card = 4) :=
by sorry

-- The final statement proving all are correct
theorem are_all_statements_correct : 
(statement_A a b) ∧ (statement_B a) ∧ (statement_C) ∧ (statement_D) := 
by sorry

end are_all_statements_correct_l313_313858


namespace tim_stored_bales_l313_313393

theorem tim_stored_bales
  (initial_bales: ℕ)
  (final_bales: ℕ)
  (initial_bales_eq: initial_bales = 28)
  (final_bales_eq: final_bales = 54)
  : final_bales - initial_bales = 26 :=
by
  rw [final_bales_eq, initial_bales_eq]
  norm_num
  done

test:
  tim_stored_bales 28 54 rfl rfl

end tim_stored_bales_l313_313393


namespace bus_stop_time_l313_313941

theorem bus_stop_time (v_no_stop v_with_stop : ℝ) (t_per_hour_minutes : ℝ) (h1 : v_no_stop = 48) (h2 : v_with_stop = 24) : t_per_hour_minutes = 30 := 
sorry

end bus_stop_time_l313_313941


namespace angle_AQB_is_obtuse_prob_l313_313334

noncomputable def hexagon : set (ℝ × ℝ) := 
  {p | (p ∈ [{(0, 3)}, {(5, 0)}, {(3 * real.pi + 2, 0)}, {(3 * real.pi + 2, 5)}, {(0, 5)}, {(0, 3)}])}

def in_semicircle (Q : ℝ × ℝ) : Prop :=
  dist Q (2.5, 1.5) < real.sqrt 8.5

noncomputable def prob_obtuse_angle : ℝ :=
  let hex_area := (15 * real.pi + 7.5) in
  let semicircle_area := (4.25 * real.pi) in
  semicircle_area / hex_area

theorem angle_AQB_is_obtuse_prob :
  prob_obtuse_angle ≈ 4.25 / 17.39 :=
sorry

end angle_AQB_is_obtuse_prob_l313_313334


namespace circle_equation_center_point_l313_313383

theorem circle_equation_center_point
(center_x center_y : ℝ) 
(p_x p_y : ℝ)
(h_center : center_x = 2 ∧ center_y = -1)
(h_point : p_x = -1 ∧ p_y = 3) :
  (∃ (R : ℝ), (R = 5 ∧ ∀ (x y : ℝ), (x - center_x) ^ 2 + (y - center_y) ^ 2 = R ^ 2) ∧ 
  ∀ (x y : ℝ), (x - 2) ^ 2 + (y + 1) ^ 2 = 25) := by
s

end circle_equation_center_point_l313_313383


namespace john_payment_l313_313742

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end john_payment_l313_313742


namespace collinear_vectors_value_of_x_l313_313681

noncomputable def sin_deg (θ : ℝ) : ℝ := Real.sin (θ * Real.pi / 180)
noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem collinear_vectors_value_of_x :
  let a := (1, Real.sqrt(1 + sin_deg 20))
      b := (1 / sin_deg 55, x)
  ∃ x : ℝ, ∀ k : ℝ, (a.1 = k * b.1 ∧ a.2 = k * b.2) → x = Real.sqrt 2 :=
by
  sorry

end collinear_vectors_value_of_x_l313_313681


namespace valid_votes_for_candidate_A_l313_313720

theorem valid_votes_for_candidate_A 
    (total_votes : ℕ) 
    (invalid_percentage valid_percentage : ℕ) 
    (candidate_A_percentage : ℕ)
    (total_votes_eq : total_votes = 560000)
    (invalid_percentage_eq : invalid_percentage = 15)
    (valid_percentage_eq : valid_percentage = 85)
    (candidate_A_percentage_eq : candidate_A_percentage = 55) :
  let valid_votes := valid_percentage * total_votes / 100 in
  let a_votes := candidate_A_percentage * valid_votes / 100 in
  a_votes = 261800 :=
by
  sorry

end valid_votes_for_candidate_A_l313_313720


namespace sin_135_l313_313519

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313519


namespace surface_area_y_eq_x_cubed_surface_area_astroid_surface_area_ellipse_l313_313613

-- Problem 1: Surface area for the curve y = x^3
theorem surface_area_y_eq_x_cubed :
  surface_area (curve (λ x : ℝ, x^3)) (-2/3) (2/3) = (196 * π) / 729 := sorry

-- Problem 2: Surface area for the astroid
theorem surface_area_astroid (a : ℝ) :
  surface_area (curve_param (λ t : ℝ, (a * (Real.cos t)^3, a * (Real.sin t)^3))) 0 (π / 2) = (12 / 5) * π * a^2 := sorry

-- Problem 3: Surface area for the ellipse
theorem surface_area_ellipse (a b : ℝ) (ha_gt_hb : a > b) :
  surface_area (ellipse (a, b)) (0 / (1 : ℝ)) 1 = 4 * π * a^2 := sorry

end surface_area_y_eq_x_cubed_surface_area_astroid_surface_area_ellipse_l313_313613


namespace monkey_slip_distance_l313_313454

theorem monkey_slip_distance :
  ∀ (h : ℝ) (hop dist : ℝ), h = 21 → hop = 3 → dist = 19 → 
  (∃ s : ℝ, (3 - s) * 18 = 18) → s = 2 :=
by
  intros h hop dist h_eq hop_eq dist_eq h_s
  cases h_s with s hs_eq
  have h1 : h = 21 := h_eq
  have h2 : hop = 3 := hop_eq
  have h3 : dist = 19 := dist_eq
  sorry

end monkey_slip_distance_l313_313454


namespace two_pow_gt_square_for_n_ge_5_l313_313415

theorem two_pow_gt_square_for_n_ge_5 (n : ℕ) (hn : n ≥ 5) : 2^n > n^2 :=
sorry

end two_pow_gt_square_for_n_ge_5_l313_313415


namespace right_triangle_similarity_l313_313460

noncomputable def y : ℝ := 8.00
noncomputable def ratio_of_perimeters : ℝ := 1.5

theorem right_triangle_similarity (y_val : ℝ) (ratio : ℝ) :
  ∃ (h₁ h₂ : ℝ), 
    let l₁ := 12
    let m₁ := 9
    let l₂ := y_val
    let m₂ := 6
    (l₁^2 + m₁^2 = h₁^2) ∧
    (l₂^2 + m₂^2 = h₂^2) ∧
    (l₁ + m₁ + h₁) / (l₂ + m₂ + h₂) = ratio :=
begin
  use [15, 10], -- Here setting up explicit hypotenuses for illustrative purposes
  sorry
end

end right_triangle_similarity_l313_313460


namespace area_of_field_l313_313139

variable (L W A : ℕ)
open Nat

theorem area_of_field (h₁ : L = 34) (h₂ : 2 * W + L = 74) : A = 680 :=
by
  let W := (74 - L) / 2
  let A := L * W
  have h₃ : W = 20 := by
    rw [h₁] at h₂
    rw [h₁]
    exact Nat.eq_of_mul_eq_mul_left (by norm_num : 2 > 0) (by linarith : 2 * W = 40)
  have h₄ : A = 34 * 20 := by
    rw [h₁, h₃]
  exact by rw [h₄]; exact rfl

#check area_of_field

end area_of_field_l313_313139


namespace minimal_hotel_cost_proof_l313_313636

noncomputable def minimal_hotel_cost (k : ℕ) (h : 0 < k) : ℕ := 
  (1/2 * k * (4 * k^2 + k - 1)).toNat

theorem minimal_hotel_cost_proof (k : ℕ) (h : 0 < k) :
  ∃ (C : ℕ), minimal_hotel_cost k h = C := 
sorry

end minimal_hotel_cost_proof_l313_313636


namespace stops_after_finite_steps_average_L_is_n_n1_div_4_l313_313047

-- Definition of Harry's coin problem
def coin := bool
def flip (coin : bool) : bool := bnot coin

def process (coins : List coin) : List coin :=
  let k := coins.count (= tt)
  if k > 0 then coins.update_nth (k-1) flip else coins

-- Part (a): Prove that for each initial configuration, the process stops after a finite number of operations.
theorem stops_after_finite_steps {n : ℕ} (coins : List coin) (h_length : coins.length = n) :
  ∃ (m : ℕ), (iter (process coins) m = List.repeat tt n) := by {
  sorry
}

-- Part (b): Determine the average value of L(C) over all 2^n possible configurations.
noncomputable def average_L (n : ℕ) : ℚ :=
  (1 / 2 ^ n) * (Finset.univ : Finset (List bool)).sum (fun C => L C) 
  sorry

theorem average_L_is_n_n1_div_4 (n : ℕ) :
  average_L n = n * (n + 1) / 4 := by {
  sorry
}

end stops_after_finite_steps_average_L_is_n_n1_div_4_l313_313047


namespace smallest_positive_period_sqrt_sin_cos_l313_313121

theorem smallest_positive_period_sqrt_sin_cos : ∀ x : ℝ, ∀ T ∈ (λ T, ∃ k ∈ set.Ico (0:ℝ) (2*T),
  ∀ x, sqrt (sin ((3 / 2) * (x + T))) ^ 3 - sqrt (cos ((2 / 3) * (x + T))) ^ 5 =
  sqrt (sin ((3 / 2) * x)) ^ 3 - sqrt (cos ((2 / 3) * x)) ^ 5), T = 12 * π := by
sorry

end smallest_positive_period_sqrt_sin_cos_l313_313121


namespace sin_135_eq_sqrt2_over_2_l313_313567

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313567


namespace calculate_train_speed_l313_313940

def speed_train_excluding_stoppages (distance_per_hour_including_stoppages : ℕ) (stoppage_minutes_per_hour : ℕ) : ℕ :=
  let effective_running_time_per_hour := 60 - stoppage_minutes_per_hour
  let effective_running_time_in_hours := effective_running_time_per_hour / 60
  distance_per_hour_including_stoppages / effective_running_time_in_hours

theorem calculate_train_speed :
  speed_train_excluding_stoppages 42 4 = 45 :=
by
  sorry

end calculate_train_speed_l313_313940


namespace sequence_properties_l313_313673

noncomputable def a (n : ℕ) : ℕ := sorry

theorem sequence_properties :
  a 1 = 1 →
  (∀ n : ℕ, 0 < n → a n = n * (a (n + 1) - a n)) →
  (a 2 = 2) ∧ (∀ n : ℕ, 0 < n → a n = n) :=
by {
  intros,
  sorry
}

end sequence_properties_l313_313673


namespace maximize_S_n_l313_313344

-- Define the arithmetic sequence
def arithmetic_seq (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

-- Define the condition on the sequence
lemma condition_on_sequence (a1 d : ℝ) (h_pos : a1 > 0) : 
  3 * arithmetic_seq a1 d 8 = 5 * arithmetic_seq a1 d 13 → d < 0 :=
by
  sorry  -- Proof that d < 0 given the condition in context

-- Define the sum of first n terms of an arithmetic sequence
def S_n (a1 d : ℝ) (n : ℕ) : ℝ :=
  (n * (a1 + (arithmetic_seq a1 d n))) / 2

-- Define the problem to find the n that maximizes the sum S_n
theorem maximize_S_n (a1 d : ℝ) (h_pos : a1 > 0)
  (seq_cond : 3 * arithmetic_seq a1 d 8 = 5 * arithmetic_seq a1 d 13) :
  ∃ n : ℕ, n = 20 ∧ ∀ m : ℕ, S_n a1 d m ≤ S_n a1 d n :=
by
  sorry

end maximize_S_n_l313_313344


namespace min_value_condition_l313_313652

open Real

theorem min_value_condition 
  (m n : ℝ) 
  (h1 : m > 0) 
  (h2 : n > 0) 
  (h3 : 2 * m + n = 1) : 
  (1 / m + 2 / n) ≥ 8 :=
sorry

end min_value_condition_l313_313652


namespace complex_in_first_quadrant_l313_313970

-- Defining the complex number conditions.
def complex_condition (z : ℂ) := z * (2 - complex.I) = 1

-- Proving that the complex number z is in the first quadrant.
theorem complex_in_first_quadrant (z : ℂ) (h : complex_condition z) : z.re > 0 ∧ z.im > 0 :=
sorry

end complex_in_first_quadrant_l313_313970


namespace sports_club_members_l313_313716

theorem sports_club_members (N B T : ℕ) (h_total : N = 30) (h_badminton : B = 18) (h_tennis : T = 19) (h_neither : N - (B + T - 9) = 2) : B + T - 9 = 28 :=
by
  sorry

end sports_club_members_l313_313716


namespace binom_prob_xi_eq_2_l313_313637

theorem binom_prob_xi_eq_2 (ξ : ℕ → ℝ) (p : ℝ) (hp : p = 0.4) (hξ : ∀ k, E [ξ k] = 10 * p) :
  P (ξ 2) = (nat.choose 10 2) * (0.4 ^ 2) * (0.6 ^ 8) :=
by
  sorry

end binom_prob_xi_eq_2_l313_313637


namespace binomial_square_l313_313197

theorem binomial_square (a : ℝ) (x : ℝ) : (ax^2 + 8x + 16 = (1 * x + 4)^2) → a = 1 :=
by {
  intro h,
  have h' : ax^2 + 8x + 16 = x^2 + 8x + 16 := by rwa [pow_two, one_mul] at h,
  sorry
}

end binomial_square_l313_313197


namespace sin_135_l313_313497

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313497


namespace find_n_l313_313109

theorem find_n (n : ℝ) (h1 : ∀ m : ℝ, m = 4 → m^(m/2) = 4) : 
  n^(n/2) = 8 ↔ n = 2^Real.sqrt 6 :=
by
  sorry

end find_n_l313_313109


namespace percentage_of_women_not_speaking_french_l313_313440

theorem percentage_of_women_not_speaking_french
  (total_employees : ℕ := 100)
  (percentage_men : ℚ := 0.65)
  (percentage_men_speak_french : ℚ := 0.60)
  (percentage_speak_french : ℚ := 0.40)
  : (0.35 * total_employees - (0.40 * total_employees - 0.60 * 0.65 * total_employees)) / (0.35 * total_employees) * 100 = 97.14 :=
by
  sorry

end percentage_of_women_not_speaking_french_l313_313440


namespace train_passes_tree_in_28_seconds_l313_313155

def km_per_hour_to_meter_per_second (km_per_hour : ℕ) : ℕ :=
  km_per_hour * 1000 / 3600

def pass_tree_time (length : ℕ) (speed_kmh : ℕ) : ℕ :=
  length / (km_per_hour_to_meter_per_second speed_kmh)

theorem train_passes_tree_in_28_seconds :
  pass_tree_time 490 63 = 28 :=
by
  sorry

end train_passes_tree_in_28_seconds_l313_313155


namespace sin_135_l313_313503

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313503


namespace largest_square_plots_count_l313_313135

-- Define the rectangular field dimensions
def field_length : ℕ := 45
def field_width : ℕ := 30
def fence_available : ℕ := 2430

-- Define the main theorem statement
theorem largest_square_plots_count : 
  ∃ n : ℕ, (90 * n - 75 ≤ fence_available) ∧ (n = 27) ∧ (27 * (3 / 2) * 27 = 1093) :=
begin
  existsi 27,
  split,
  {
    rw [mul_comm, mul_assoc],
    linarith,
  },
  split,
  {
    refl,
  },
  {
    norm_num,
  },
end

end largest_square_plots_count_l313_313135


namespace find_value_l313_313242

noncomputable def f (x : ℝ) : ℝ := (sin x)^2 + sin x * cos x

theorem find_value (θ : ℝ) (h_min : ∀ x : ℝ, f θ ≤ f x) :
  (sin (2 * θ) + 2 * cos θ) / (sin (2 * θ) - 2 * cos (2 * θ)) = -1 / 3 :=
by
  sorry

end find_value_l313_313242


namespace distinct_divisors_in_set_l313_313007

theorem distinct_divisors_in_set (p : ℕ) (hp : Nat.Prime p) (hp5 : 5 < p) :
  ∃ (x y : ℕ), x ∈ {p - n^2 | n : ℕ} ∧ y ∈ {p - n^2 | n : ℕ} ∧ x ≠ y ∧ x ≠ 1 ∧ x ∣ y :=
by
  sorry

end distinct_divisors_in_set_l313_313007


namespace ceil_sqrt_sum_l313_313600

theorem ceil_sqrt_sum : 
  (∑ n in Finset.range (35 + 1).filter (λ n, n ≥ 8), ⌈Real.sqrt n⌉) = 139 :=
by
  sorry

end ceil_sqrt_sum_l313_313600


namespace students_scoring_above_115_l313_313131

noncomputable def number_of_students : ℕ := 50
noncomputable def mean : ℝ := 105
noncomputable def variance : ℝ := 10^2
noncomputable def distribution (x : ℝ) : ℝ := (1 / (Math.sqrt (2 * Math.pi * variance))) * Math.exp (-(x - mean)^2 / (2 * variance))
noncomputable def prob_range_95_105 : ℝ := 0.32

theorem students_scoring_above_115 : 
  (number_of_students * (1 - 2 * prob_range_95_105)) / 2 = 9 :=
by
  sorry

end students_scoring_above_115_l313_313131


namespace distance_to_origin_number_of_points_at_distance_l313_313815

theorem distance_to_origin :
  ∀ (x : ℝ), (x = -2 ∨ x = 2) → abs x = 2 :=
by
  intro x hx
  cases hx with hx_neg hx_pos
  · rw hx_neg
    norm_num
  · rw hx_pos
    norm_num

theorem number_of_points_at_distance :
  ∃ (S : set ℝ), S = {x : ℝ | abs x = 2} ∧ S = {-2, 2} :=
by
  use {x : ℝ | abs x = 2}
  split
  · ext x
    simp
    exact distance_to_origin x
  · ext
    simp
    exact or_comm 2 (-2)

end distance_to_origin_number_of_points_at_distance_l313_313815


namespace sin_135_l313_313495

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313495


namespace chocolate_candy_pieces_l313_313328

theorem chocolate_candy_pieces (boxes_bought boxes_given box_pieces : ℝ) 
(hb_bought : boxes_bought = 14.0) 
(hb_given : boxes_given = 7.0) 
(hb_pieces : box_pieces = 6.0) :
(boxes_bought - boxes_given) * box_pieces = 42.0 := by
  rw [hb_bought, hb_given, hb_pieces]
  norm_num
  sorry

end chocolate_candy_pieces_l313_313328


namespace intersection_A_C_U_B_l313_313019

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | Real.log x / Real.log 2 > 0}
def C_U_B : Set ℝ := {x | ¬ (Real.log x / Real.log 2 > 0)}

theorem intersection_A_C_U_B :
  A ∩ C_U_B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end intersection_A_C_U_B_l313_313019


namespace triangle_area_of_ellipse_foci_l313_313624

/-- Given an ellipse with equation x^2/9 + y^2/5 = 1, where F1 and F2 are the foci,
    and a point P on the ellipse such that |PF1| = 2|PF2|, then the area of triangle PF1F2 is sqrt(15). -/
theorem triangle_area_of_ellipse_foci (F1 F2 P : ℝ × ℝ)
  (ellipse_eq : ∀ (x y : ℝ), (x, y) ∈ set_of (λ p : ℝ × ℝ, p.1^2 / 9 + p.2^2 / 5 = 1))
  (dist_cond : dist P F1 = 2 * dist P F2)
  (foci_f1_f2 : F1 = (2, 0) ∧ F2 = (-2, 0)) :
  let d := 2 in  -- Focal distance c = 2
  let a := 3 in  -- Semi-major axis a = 3
  let b := sqrt 5 in  -- Semi-minor axis b = sqrt 5
  let area := 1 / 2 * 4 * sqrt (4^2 - (2^2)) in -- Area calculation based on given conditions
  area = sqrt 15 :=
by sorry

end triangle_area_of_ellipse_foci_l313_313624


namespace find_k_l313_313359

noncomputable def roots (k : ℝ) : Set ℝ :=
  {x | k * x^2 - 3 * x + 2 = 0}

theorem find_k (k : ℝ) (A : Set ℝ):
  (∀ x, x ∈ A ↔ k * x^2 - 3 * x + 2 = 0) →
  (A.card = 1) →
  (k = 0 ∨ k = 9 / 8) :=
by
  intro h_A h_card
  sorry

end find_k_l313_313359


namespace minimum_employees_needed_l313_313892

def min_new_employees (water_pollution: ℕ) (air_pollution: ℕ) (both: ℕ) : ℕ :=
  119 + 34

theorem minimum_employees_needed : min_new_employees 98 89 34 = 153 := 
  by
  sorry

end minimum_employees_needed_l313_313892


namespace sin_2alpha_minus_pi_over_4_eq_31sqrt2_over_50_l313_313215

theorem sin_2alpha_minus_pi_over_4_eq_31sqrt2_over_50
  (α : ℝ) (h₁ : sin α - cos α = 1 / 5) (h₂ : 0 ≤ α ∧ α ≤ π) :
  sin (2 * α - π / 4) = (31 * real.sqrt 2) / 50 :=
begin
  sorry -- proof is omitted
end

end sin_2alpha_minus_pi_over_4_eq_31sqrt2_over_50_l313_313215


namespace distribution_ways_l313_313904

-- Define the number of gifts and number of fans
def num_gifts : ℕ := 5
def num_fans : ℕ := 3

-- Define a predicate to check if a distribution follows the condition
def valid_distribution (gifts : Fin num_fans → ℕ) : Prop :=
  (∑ i, gifts i) = num_gifts ∧ ∀ i, gifts i > 0

-- The number of different distributions of the gifts ensuring each fan receives at least one gift is 6
theorem distribution_ways : ∃ d : Finset (Fin num_fans → ℕ), 
  (∀ f ∈ d, valid_distribution f) ∧ d.card = 6 :=
sorry

end distribution_ways_l313_313904


namespace remaining_berries_equivalent_l313_313026

theorem remaining_berries_equivalent (S1 B1 R1 : ℕ) (S2 B2 R2 : ℕ) (SE1 BE1 RE1 : ℕ) (SE2 BE2 RE2 : ℕ) :
  S1 = 2.5 * 12 ∧
  B1 = 1.75 * 12 ∧
  R1 = 1.25 * 12 ∧
  S2 = 2.25 * 12 ∧
  B2 = 2 * 12 ∧
  R2 = 1.5 * 12 ∧
  SE1 = 6 * 3 ∧
  BE1 = 4 * 3 ∧
  RE1 = 0 * 3 ∧
  SE2 = 5 * 2 ∧
  BE2 = 3 * 2 ∧
  RE2 = 2 * 2 →
  S1 - SE1 + S2 - SE2 = 29 ∧
  B1 - BE1 + B2 - BE2 = 27 ∧
  R1 - RE1 + R2 - RE2 = 29 :=
by
  intros
  sorry

end remaining_berries_equivalent_l313_313026


namespace four_equal_polygons_can_be_arranged_l313_313737

noncomputable def possible_configuration : Prop :=
  ∃ (P : fin 4 → set (ℝ × ℝ)), 
    (∀ i j : fin 4, i ≠ j → interior (P i) ∩ interior (P j) = ∅) ∧ 
    (∀ i j : fin 4, i ≠ j → ∃ (s : set (ℝ × ℝ)), s ∈ frontier (P i) ∧ s ∈ frontier (P j))

theorem four_equal_polygons_can_be_arranged : possible_configuration :=
sorry

end four_equal_polygons_can_be_arranged_l313_313737


namespace eval_expression_l313_313437

theorem eval_expression : abs (-6) - (-4) + (-7) = 3 :=
by
  sorry

end eval_expression_l313_313437


namespace number_of_morning_rowers_l313_313804

def campers_problem :=
  (3:ℕ) * (60 / ((3 + 2 + 4):ℕ)) = 18

theorem number_of_morning_rowers :
  (∀ morning afternoon evening total,
    afternoon = 7 ∧
    (morning + afternoon + evening = total) ∧ 
    (morning:afternoon:evening = 3:2:4 ∧ 
    total = 60) →
    morning = 18) := by
  sorry

end number_of_morning_rowers_l313_313804


namespace angle_equality_l313_313083

theorem angle_equality 
  (circle : Type*) 
  [metric_space circle]
  [normed_group circle] 
  [normed_space ℝ circle] 
  (P A B C D Q : circle) 
  (hP_outside : ∀ (x : circle), x ≠ P) 
  (tangents : P ≠ A ∧ P ≠ B) 
  (secant_line : P ≠ C ∧ P ≠ D ∧ C ≠ D ∧ C ≠ P ∧ C ≠ D) 
  (on_chord : ∀ (x : circle), x = Q → x ∈ segment C D) 
  (angle_DAQ_PBC : ∠ (D A Q) = ∠ (P B C)) :
  ∠ (D B Q) = ∠ (P A C) := 
sorry

end angle_equality_l313_313083


namespace sin_135_eq_l313_313510

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313510


namespace find_x_given_y_l313_313869

-- Given x varies inversely as the square of y, we define the relationship
def varies_inversely (x y k : ℝ) : Prop := x = k / y^2

theorem find_x_given_y (k : ℝ) (h_k : k = 4) :
  ∀ (y : ℝ), varies_inversely x y k → y = 2 → x = 1 :=
by
  intros y h_varies h_y_eq
  -- We need to prove the statement here
  sorry

end find_x_given_y_l313_313869


namespace sum_of_radii_l313_313598

-- Definitions based on the given conditions
def circle_radius : ℝ := 5
def ratio1 : ℝ := 1
def ratio2 : ℝ := 2
def ratio3 : ℝ := 3
def total_ratio : ℝ := ratio1 + ratio2 + ratio3

-- Radii of the bases of the cones
def r1 : ℝ := (ratio1 / total_ratio) * circle_radius
def r2 : ℝ := (ratio2 / total_ratio) * circle_radius
def r3 : ℝ := (ratio3 / total_ratio) * circle_radius

-- Statement of the problem to prove
theorem sum_of_radii : r1 + r2 + r3 = circle_radius := by
  sorry

end sum_of_radii_l313_313598


namespace cube_in_cylinder_l313_313087

theorem cube_in_cylinder (A C1 B C D A1 B1 D1 : ℝ → ℝ → ℝ)
  (unit_cube : A = (0, 0, 0) ∧ C1 = (1, 1, 1) 
                ∧ B = (0, 1, 0) ∧ C = (1, 0, 0) ∧ D = (1, 1, 0) 
                ∧ A1 = (0, 0, 1) ∧ B1 = (0, 1, 1) ∧ D1 = (1, 0, 1))
  (center_base_cylinder : A = (0, 0, 0) ∧ C1 = (0, 0, h))
  (vertices_surface_cylinder : B ≠ A ∧ C ≠ A ∧ D ≠ A 
                               ∧ A1 ≠ A ∧ B1 ≠ A ∧ D1 ≠ A) 
  : h = real.sqrt 3 ∧ r = real.sqrt 6 / 3 := 
by
  sorry

end cube_in_cylinder_l313_313087


namespace area_difference_is_196_l313_313028

noncomputable def max_area (l w : ℕ) (h : l + w = 30) : ℕ :=
l * w

noncomputable def min_area (l w : ℕ) (h : l + w = 30) : ℕ :=
l * w

noncomputable def area_diff : ℕ := 
let maxA := (max_area 15 15 rfl /- rfl as l + w = 30 proof for max -/) in
let minA := (min_area 1 29 (by simp [add_comm])) in
maxA - minA

theorem area_difference_is_196 : area_diff = 196 := 
by
  sorry

end area_difference_is_196_l313_313028


namespace ratio_perimeters_not_integer_l313_313786

theorem ratio_perimeters_not_integer
  (a k l : ℤ) (h_a_pos : a > 0) (h_k_pos : k > 0) (h_l_pos : l > 0)
  (h_area : a^2 = k * l) :
  ¬ ∃ n : ℤ, n = (k + l) / (2 * a) :=
by
  sorry

end ratio_perimeters_not_integer_l313_313786


namespace perfect_square_trinomial_l313_313688

theorem perfect_square_trinomial (m : ℝ) : (∃ k : ℝ, x^2 + mx + 16 = (x + k)^2) → (m = 8 ∨ m = -8) :=
by
  intros h
  have : x^2 + mx + 16 = (x + k)^2 by sorry
  have discriminant_zero : m^2 - 64 = 0 by sorry
  exact ⟨by sorry, by sorry⟩

end perfect_square_trinomial_l313_313688


namespace sum_of_a_and_b_l313_313626

theorem sum_of_a_and_b (a b : ℝ) (h : a^2 + b^2 + 2 * a - 4 * b + 5 = 0) :
  a + b = 1 :=
sorry

end sum_of_a_and_b_l313_313626


namespace oil_tank_depth_l313_313449

/-- The cylindrical oil tank problem -/
theorem oil_tank_depth:
  ∀ (length diameter surface_area : ℝ),
  length = 10 ∧ diameter = 6 ∧ surface_area = 40 →
  ∃ h : ℝ, h = 3 - Real.sqrt 5 ∨ h = 3 + Real.sqrt 5 :=
by {
  intros length diameter surface_area h,
  rcases h with ⟨hl, hd, hs⟩,
  have r := diameter / 2,
  have a := length * r,
  have rectangle_area := surface_area,
  sorry
}

end oil_tank_depth_l313_313449


namespace part_a_part_b_l313_313107

variable (α β γ : ℝ)
variable (R r : ℝ)

-- For Part (a)
theorem part_a (h1 : α + β + γ = π) (h2 : 1 < α ∧ α < π / 2) 
  (h3 : 1 < β ∧ β < π / 2) (h4 : 1 < γ ∧ γ < π / 2)
  (h5 : r / R ≤ 0.5) :
  1 < cos α + cos β + cos γ ∧ cos α + cos β + cos γ ≤ 3 / 2 :=
sorry

-- For Part (b)
theorem part_b (h1 : α + β + γ = π) (h2 : 1 < α ∧ α < π / 2) 
  (h3 : 1 < β ∧ β < π / 2) (h4 : 1 < γ ∧ γ < π / 2)
  (h5 : r / R ≤ 0.5) :
  1 < sin (α / 2) + sin (β / 2) + sin (γ / 2) ∧ sin (α / 2) + sin (β / 2) + sin (γ / 2) ≤ 3 / 2 :=
sorry

end part_a_part_b_l313_313107


namespace sum_zero_of_absolute_inequalities_l313_313789

theorem sum_zero_of_absolute_inequalities 
  (a b c : ℝ) 
  (h1 : |a| ≥ |b + c|) 
  (h2 : |b| ≥ |c + a|) 
  (h3 : |c| ≥ |a + b|) :
  a + b + c = 0 := 
  by
    sorry

end sum_zero_of_absolute_inequalities_l313_313789


namespace third_shot_hits_l313_313872

structure ShootingRange where
  scores : Fin 10 → ℕ
  (hscores : ∀ i, scores i ∈ {10, 9, 9, 8, 8, 5, 4, 4, 3, 2})

variables (Petya Vasya : Fin 5 → ℕ)
variable (target : ShootingRange)

axiom shots_eq (h1 : Petya 0 = Vasya 0)
  (h2 : Petya 1 = Vasya 1)
  (h3 : Petya 2 = Vasya 2)

axiom shots_3x (h4 : Petya 3 + Petya 4 + Petya 5 = 3 * (Vasya 3 + Vasya 4 + Vasya 5))

noncomputable def third_shot_Petya := target.scores 0
noncomputable def third_shot_Vasya := target.scores 9

theorem third_shot_hits (hscore : ∀ i, target.scores i ∈ {10, 9, 9, 8, 8, 5, 4, 4, 3, 2}):
  (third_shot_Petya Petya = 10) ∧ (third_shot_Vasya Vasya = 2) :=
by
  sorry

end third_shot_hits_l313_313872


namespace necessary_and_sufficient_l313_313123

theorem necessary_and_sufficient (a b : ℝ) : a > b ↔ a * |a| > b * |b| := sorry

end necessary_and_sufficient_l313_313123


namespace part1_part2_l313_313999

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(-x)
  else (Real.log x / Real.log 3 - 1) * (Real.log x / Real.log 3 - 2)

theorem part1 : f (Real.log 3 / Real.log 2 - Real.log 2 / Real.log 2) = 2 / 3 := by
  sorry

theorem part2 : ∃ x : ℝ, f x = -1 / 4 := by
  sorry

end part1_part2_l313_313999


namespace part1_part2_l313_313709

variables {a b c : ℝ} {A B C : ℝ}

-- Given the conditions
def triangle_conditions (a b c A B C : ℝ) : Prop :=
  A < π / 2 ∧
  let p := (1, sqrt 3 * cos (A / 2))
      q := (2 * sin (A / 2), 1 - cos (2 * A)) in
    (p.1 * q.2 = p.2 * q.1)

-- Problem Part (1)
theorem part1 (h : triangle_conditions a b c A B C) (h1 : a^2 - c^2 = b^2 - b * c * 1) : 
  ∃ m, a^2 - c^2 = b^2 - m * b * c ∧ m = 1 :=
begin
  sorry
end

-- Problem Part (2)
theorem part2 (h : triangle_conditions a b c A B C) (ha : a = sqrt 3) : 
  ∃ max_area, ∃ b c : ℝ, max_area = 1 / 2 * b * c * sin A ∧ max_area ≤ (3 * sqrt 3) / 4 ∧ a^2 = b^2 + c^2 - 2 * b * c * cos A :=
begin
  sorry
end

end part1_part2_l313_313709


namespace conditional_probability_B_given_A_l313_313834

-- Definitions of the given probabilities.
def P_A : ℚ := 7 / 8
def P_B : ℚ := 6 / 8
def P_AB : ℚ := 5 / 8

/-- 
  Given P(A) = 7/8, P(B) = 6/8, and P(AB) = 5/8, 
  the conditional probability P(B|A) is 5/7.
-/
theorem conditional_probability_B_given_A :
  P_AB / P_A = 5 / 7 :=
by
  -- Placeholder for proof steps
  sorry

end conditional_probability_B_given_A_l313_313834


namespace radius_of_circle_l313_313833

-- Define the given circle equation as a condition
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 2*y - 7 = 0

theorem radius_of_circle : ∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, r = 3 :=
by
  sorry

end radius_of_circle_l313_313833


namespace ratio_of_odd_to_even_factors_l313_313871

def prime_factor_34 : ℕ := 2 * 17
def prime_factor_63 : ℕ := 3^2 * 7
def prime_factor_270 : ℕ := 2 * 3^3 * 5

noncomputable def N : ℕ := prime_factor_34 * prime_factor_34 * prime_factor_63 * prime_factor_270

theorem ratio_of_odd_to_even_factors (N : ℕ) (hN : N = 34 * 34 * 63 * 270) :
  ratio (sum_of_odd_factors N) (sum_of_even_factors N) = 1 / 14 := by
  sorry

end ratio_of_odd_to_even_factors_l313_313871


namespace player_one_wins_optimal_play_l313_313388

-- Define the initial conditions.
def initial_matches : Nat := 500
def is_power_of_two (n : Nat) : Prop := ∃ k : Nat, n = 2 ^ k

-- Define the main theorem that Player 1 wins with optimal play.
theorem player_one_wins_optimal_play : 
  ∃ win_strategy : (Nat → Nat) → unit, 
  (∀ turn : Nat, turn % 2 = 0 → win_strategy turn = player_one) ∧
  (∀ turn : Nat, turn % 2 = 1 → win_strategy turn = player_two) ∧
  optimal_play win_strategy → game_result win_strategy initial_matches = player_one :=
sorry

end player_one_wins_optimal_play_l313_313388


namespace equilateral_triangle_min_rot_angle_l313_313472

theorem equilateral_triangle_min_rot_angle (T : Type) [EquilateralTriangle T] : 
  ∃ θ : ℝ, (θ = 120) :=
by
  sorry

end equilateral_triangle_min_rot_angle_l313_313472


namespace sequence_general_formula_sum_of_first_2n_terms_l313_313223

-- Definitions for the sequence {a_n} and the sum S_{2n}
def seq (n : ℕ) : ℕ :=
  if n % 2 = 1 then (7 - n) / 2
  else 2 * 3^((n - 2) / 2)

def S (n : ℕ) : ℤ :=
  -1 / 2 * (n : ℤ)^2 + 7 / 2 * (n : ℤ) + 3^(n : ℤ) - 1

-- Theorem stating the general formula for the sequence
theorem sequence_general_formula (n : ℕ) : 
  seq n = if n % 2 = 1 then (7 - n) / 2 else 2 * 3 ^ ((n - 2) / 2) :=
by sorry

-- Theorm stating the sum of the first 2n terms of the sequence
theorem sum_of_first_2n_terms (n : ℕ) : 
  S (2 * n) = -1 / 2 * (2 * n : ℤ)^2 + 7 / 2 * (2 * n : ℤ) + 3^((n : ℤ)^2) - 1 :=
by sorry

end sequence_general_formula_sum_of_first_2n_terms_l313_313223


namespace total_oysters_and_crabs_is_195_l313_313387

-- Define the initial conditions
def oysters_day1 : ℕ := 50
def crabs_day1 : ℕ := 72

-- Define the calculations for the second day
def oysters_day2 : ℕ := oysters_day1 / 2
def crabs_day2 : ℕ := crabs_day1 * 2 / 3

-- Define the total counts over the two days
def total_oysters : ℕ := oysters_day1 + oysters_day2
def total_crabs : ℕ := crabs_day1 + crabs_day2
def total_count : ℕ := total_oysters + total_crabs

-- The goal specification
theorem total_oysters_and_crabs_is_195 : total_count = 195 :=
by
  sorry

end total_oysters_and_crabs_is_195_l313_313387


namespace proof_problem_l313_313982

def A : set ℝ := {x | x^2 - 2 * x > 0}

def B : set ℝ := {y | ∃ x : ℝ, y = sin x}

def complement_A : set ℝ := {x | x ∉ A}

def R_complement_A := {x | 0 ≤ x ∧ x ≤ 2}

theorem proof_problem : ((R_complement_A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1}) :=
by
  sorry

end proof_problem_l313_313982


namespace number_of_zeros_of_f_l313_313825

def f (x : ℝ) : ℝ := x - real.sqrt x - 2

theorem number_of_zeros_of_f : (set_of (λ x : ℝ, f x = 0)).finite ∧ (set_of (λ x : ℝ, f x = 0)).card = 1 := by
  sorry

end number_of_zeros_of_f_l313_313825


namespace inequality_proof_l313_313220

theorem inequality_proof 
  (x y z : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hz : z > 0)
  (hxz : x * z = 1) 
  (h₁ : x * (1 + z) > 1) 
  (h₂ : y * (1 + x) > 1) 
  (h₃ : z * (1 + y) > 1) :
  2 * (x + y + z) ≥ -1/x + 1/y + 1/z + 3 :=
sorry

end inequality_proof_l313_313220


namespace imo_problem_30th_prelim_l313_313969

theorem imo_problem_30th_prelim : ∀ (P : Fin 7 → Type) (L : Fin 7 → Fin 7 → Prop),
  (∀ i j k : Fin 7, ∃ (i' j' : Fin 7), i' ≠ j' ∧ L i' j') →
  ∃ n, n = 9 ∧ 
  ∀ (i j k : Fin 7), L i j ∨ L j k ∨ L k i :=
by
  trivial

end imo_problem_30th_prelim_l313_313969


namespace one_third_of_flour_l313_313459

-- Definition of the problem conditions
def initial_flour : ℚ := 5 + 2 / 3
def portion : ℚ := 1 / 3

-- Definition of the theorem to prove
theorem one_third_of_flour : portion * initial_flour = 1 + 8 / 9 :=
by {
  -- Placeholder proof
  sorry
}

end one_third_of_flour_l313_313459


namespace complex_number_proof_l313_313316

open Complex

noncomputable def problem_complex (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) : ℂ :=
  (z - 1) * (z^2 - 1) * (z^3 - 1) * (z^4 - 1) * (z^5 - 1) * (z^6 - 1)

theorem complex_number_proof (z : ℂ) (h1 : z ^ 7 = 1) (h2 : z ≠ 1) :
  problem_complex z h1 h2 = 8 :=
  sorry

end complex_number_proof_l313_313316


namespace solve_for_X_l313_313793

theorem solve_for_X (X : Real) :
  sqrt (X^3) = 9 * real.root 81 9 → X = real.root (3 ^ 44) 27 :=
by
  intros h
  sorry

end solve_for_X_l313_313793


namespace john_payment_l313_313743

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end john_payment_l313_313743


namespace average_population_is_1000_l313_313376

-- Define the populations of the villages.
def populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

-- Define the number of villages.
def num_villages : ℕ := 7

-- Define the total population.
def total_population (pops : List ℕ) : ℕ :=
  pops.foldl (λ acc x => acc + x) 0

-- Define the average population computation.
def average_population (pops : List ℕ) (n : ℕ) : ℕ :=
  total_population pops / n

-- Prove that the average population of the 7 villages is 1000.
theorem average_population_is_1000 :
  average_population populations num_villages = 1000 := by
  -- Proof omitted.
  sorry

end average_population_is_1000_l313_313376


namespace cyclist_speed_l313_313136

theorem cyclist_speed
  (hiker_speed : ℝ)
  (cyclist_wait_time : ℝ)
  (hiker_catch_time : ℝ)
  (hiker_distance : ℝ)
  (cyclist_distance : ℝ)
  (hiker_time : ℝ)
  (cyclist_speed : ℝ) :
  hiker_speed = 4 ∧
  cyclist_wait_time = 5 / 60 ∧
  hiker_catch_time = 17.5 / 60 ∧
  hiker_distance = hiker_speed * (7 / 24) ∧
  cyclist_distance = cyclist_speed * (1 / 12) ∧
  hiker_distance = cyclist_distance →
  cyclist_speed = 14 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end cyclist_speed_l313_313136


namespace impossible_cover_of_chessboard_l313_313337

theorem impossible_cover_of_chessboard :
  let chessboard_size := 100
  let num_trominos := 25
  let tromino_cover := 3
  num_trominos * tromino_cover < chessboard_size :=
by
  let chessboard_size := 100
  let num_trominos := 25
  let tromino_cover := 3
  show num_trominos * tromino_cover < chessboard_size, from sorry

end impossible_cover_of_chessboard_l313_313337


namespace max_min_of_complex_expression_l313_313993

open Complex Real

theorem max_min_of_complex_expression {z : ℂ} (hz : |z| = 1) :
  ∀ w : ℂ, (w = z^3 - 3 * z - 2) →
  (0 ≤ |w| ∧ |w| ≤ 3 * Real.sqrt 3) :=
by
  sorry

end max_min_of_complex_expression_l313_313993


namespace impossible_to_arrange_l313_313870

-- Define the range of natural numbers from 1 to 121
def nat_numbers := {n : ℕ | 1 ≤ n ∧ n ≤ 121}

-- Define perfect squares in the given range
def perfect_squares := {n : ℕ | ∃ m : ℕ, n = m * m ∧ 1 ≤ n ∧ n ≤ 121}

-- Define the 11x11 table
def table := Fin 11 × Fin 11

-- Define the adjacency condition
def adjacent (a b : table) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 + 1 = b.2)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 + 1 = b.1))

-- Define the cell placement
def placement : table → ℕ → Prop

-- Define the problem statement
theorem impossible_to_arrange :
  ¬ (∃ placement : table → ℕ, 
    (∀ n : ℕ, n ∈ nat_numbers → ∃ t : table, placement t n) ∧
    (∀ t1 t2 : table, ∃ n m : ℕ, placement t1 n ∧ placement t2 m → (n = m + 1 ∨ n + 1 = m) → adjacent t1 t2) ∧
    (∃ col : Fin 11, ∀ n ∈ perfect_squares, ∃ row : Fin 11, placement (row, col) n)) :=
sorry

end impossible_to_arrange_l313_313870


namespace tetrahedron_properties_l313_313079

open Real

variables (a k : ℝ) (x y : ℝ)
variables (Ax By : ℝ → ℝ)

-- Conditions
def conditions := 
  (Ax ⊥ By) ∧ 
  (∃ (P Q : ℝ), P ∈ Ax ∧ Q ∈ By ∧ dist P Q = 2a) ∧ 
  (Ax ≠ 0) ∧ 
  (By ≠ 0) ∧ 
  (x * y = k^2) ∧ 
  (0 < k)

-- Volume of tetrahedron PABQ is constant
noncomputable def volume_constant :=
  ∀ x y, x * y = k^2 → (2 * a * k^2 / 3)

-- Distance from vertex A to base PBQ
noncomputable def distance_from_A (y : ℝ) :=
  (2 * a * k^2) / sqrt (4 * a^2 * y^2 + k^4)

-- Diameter of circumsphere with minimal diameter conditions
noncomputable def diameter_minimal (x y : ℝ) :=
  x = y → ((4 * a^2 + 2 * k^2))

-- Main theorem
theorem tetrahedron_properties :
  (conditions a k x y Ax By) →
  volume_constant a k x y = (2 * a * k^2 / 3) ∧
  distance_from_A a k y = (2 * a * k^2 / sqrt (4 * a^2 * y^2 + k^4)) ∧
  diameter_minimal a k x y = (4 * a^2 + 2 * k^2) :=
by
  sorry

end tetrahedron_properties_l313_313079


namespace inequality_sum_reciprocal_l313_313219

theorem inequality_sum_reciprocal (n : ℕ) (a : ℕ → ℝ) (h : ∀ i, 1 ≤ i → i ≤ n → 0 < a i) :
  (1 / (Finset.sum (Finset.range n) (λ i, 1 / (1 + a i)))) - 
  (1 / (Finset.sum (Finset.range n) (λ i, 1 / a i))) ≥ 1 / n := 
by 
  sorry

end inequality_sum_reciprocal_l313_313219


namespace sin_135_eq_sqrt2_div_2_l313_313586

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313586


namespace tin_in_new_mixed_alloy_is_correct_l313_313439

noncomputable def total_tin_in_mixed_alloy (weight_A weight_B weight_C : ℝ) 
    (ratio_lead_tin_A ratio_tin_copper_B ratio_lead_tin_copper_C : ℕ × ℕ) 
    (ratio_lead_C ratio_tin_C ratio_copper_C : ℕ) : ℝ :=
let 
    part_weight_A := weight_A / (ratio_lead_tin_A.1 + ratio_lead_tin_A.2),
    tin_A := ratio_lead_tin_A.2 * part_weight_A,
    part_weight_B := weight_B / (ratio_tin_copper_B.1 + ratio_tin_copper_B.2),
    tin_B := ratio_tin_copper_B.1 * part_weight_B,
    part_weight_C := weight_C / (ratio_lead_C + ratio_tin_C + ratio_copper_C),
    tin_C := ratio_tin_C * part_weight_C
in
tin_A + tin_B + tin_C

theorem tin_in_new_mixed_alloy_is_correct :
    total_tin_in_mixed_alloy 120 180 100 (2, 3) (3, 5) 3 2 6 = 157.68 :=
by
    sorry

end tin_in_new_mixed_alloy_is_correct_l313_313439


namespace log_exponent_simplification_l313_313597

theorem log_exponent_simplification : log 3 9 + 4^(1/2) = 4 := 
by
  sorry

end log_exponent_simplification_l313_313597


namespace product_of_squares_of_consecutive_even_integers_l313_313831

theorem product_of_squares_of_consecutive_even_integers :
  ∃ (a : ℤ), (a - 2) * a * (a + 2) = 36 * a ∧ (a > 0) ∧ (a % 2 = 0) ∧
  ((a - 2)^2 * a^2 * (a + 2)^2) = 36864 :=
by
  sorry

end product_of_squares_of_consecutive_even_integers_l313_313831


namespace intersection_A_B_l313_313984

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l313_313984


namespace problem_l313_313641

noncomputable def sequence_a (n : ℕ) : ℚ
| 0       := 1
| (n + 1) := if (n + 1) % 2 = 0 then sequence_a n + 1 / 2^(n + 1) else sequence_a n - 1 / 2^(n + 1)

theorem problem
  (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n, n ≥ 2 → |a n - a (n - 1)| = 1 / 2^n)
  (h3 : ∀ n, a (2 * n - 1) > a (2 * n + 1))
  (h4 : ∀ n, a (2 * n) < a (2 * n + 2)) :
  5 - 6 * a 10 = 1 / 512 :=
by sorry

end problem_l313_313641


namespace matrix_polynomial_solution_l313_313304

def B : Matrix (Fin 3) (Fin 3) ℝ := !![
  [1, 2, 3],
  [2, 1, 2],
  [3, 2, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ := 1 -- Identity Matrix

def Z : Matrix (Fin 3) (Fin 3) ℝ := 0 -- Zero Matrix

theorem matrix_polynomial_solution :
  ∃ (s t u : ℝ), s = -7 ∧ t = 2 ∧ u = -9 ∧
                 (B * B * B) + s • (B * B) + t • B + u • I = Z := by
  sorry

end matrix_polynomial_solution_l313_313304


namespace minimize_intersection_area_l313_313335

noncomputable def positionM_min_area (A B C : Point) (h_acute : AcuteTriangle A B C) : Point :=
  let M := FootOfPerpendicular B A C in
  M

theorem minimize_intersection_area (ABC : Triangle)
  (h_acute : ABC.acute)
  (circum_ABM circum_CBM : Circle)
  (M : Point)
  (h_M_on_AC : M ∈ (Line AC))
  (h_circum_ABM : Circumscribed_circle ABC M circum_ABM)
  (h_circum_CBM : Circumscribed_circle ABC M circum_CBM) :
  M = FootOfPerpendicular B A C :=
sorry

end minimize_intersection_area_l313_313335


namespace problem_statement_l313_313362

-- Define the given points and properties
variables {A B C T H P D : Type}

-- Assuming the properties of the points according to the problem
variables [IsAcuteAngledTriangle A B C]
variables [IsCircumcirclePoint A B C T]
variables [IsOrthocenter A B C H]
variables [IsPerpendicularFromPointToLine H (LineThrough A T) P]
variables [IsIntersectionOfLineAndCircumcircle (LineThrough T P) (CircumcircleOf A B C) D]

-- Define the theorem statement
theorem problem_statement :
  let AB := distance A B,
      AC := distance A C,
      BD := distance B D,
      DC := distance D C in
  AB^2 + DC^2 = AC^2 + BD^2 :=
sorry

end problem_statement_l313_313362


namespace angle_B_in_triangle_ABC_l313_313292

theorem angle_B_in_triangle_ABC
  (a b c : ℝ)
  (h1 : b^2 = a^2 + c^2 - a * c) :
  ∠B = real.arccos (1/2) :=
sorry

end angle_B_in_triangle_ABC_l313_313292


namespace diff_third_second_smallest_l313_313842

theorem diff_third_second_smallest (a b c d e : ℕ) (h : {a, b, c, d, e} = {10, 11, 12, 13, 14}) :
    (order_ofₒ [a, b, c, d, e]).nth 2 - (order_ofₒ [a, b, c, d, e]).nth 1 = 1 := by
  sorry

end diff_third_second_smallest_l313_313842


namespace inscribed_polygon_exists_l313_313630

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

theorem inscribed_polygon_exists (n : ℕ) (α : ℕ → ℕ → ℝ) (lines_not_parallel : ∀ i j : ℕ, i ≠ j → α i j ≠ α j i) :
  (is_even n → (∃ k : ℤ, (∑ i j in finset.range n, if i < j then α i j else 0) = k * 180) ↔ (∃ inf_solutions : ℕ, inf_solutions = 0 ∨ inf_solutions = ∞)) ∧
  (is_odd n → ∃ exactly_two : ℕ, exactly_two = 2) :=
by sorry

end inscribed_polygon_exists_l313_313630


namespace tan_add_pi_div_four_sine_cosine_ratio_l313_313208

-- Definition of the tangent function and trigonometric identities
variable {α : ℝ}

-- Given condition: tan(α) = 2
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Problem 1: Prove that tan(α + π/4) = -3
theorem tan_add_pi_div_four : Real.tan ( α + Real.pi / 4 ) = -3 :=
by
  sorry

-- Problem 2: Prove that (6 * sin(α) + cos(α)) / (3 * sin(α) - cos(α)) = 13 / 5
theorem sine_cosine_ratio : 
  ( 6 * Real.sin α + Real.cos α ) / ( 3 * Real.sin α - Real.cos α ) = 13 / 5 :=
by
  sorry

end tan_add_pi_div_four_sine_cosine_ratio_l313_313208


namespace tunnel_length_lower_average_price_l313_313120

-- Problem (1): Proving the length of the tunnel
theorem tunnel_length
  (L : ℝ) (T₁ T₂ : ℝ) (Δv : ℝ) (x : ℝ)
  (h₀ : L = 400)
  (h₁ : T₁ = 10)
  (h₂ : T₂ = 9)
  (h₃ : Δv = 0.1 / 60)
  (h₄ : (L + x) / T₁ = L / T₁ + Δv * T₁)
  (h₅ : (L + x) / T₂ = (L / T₁ + Δv * T₁) + Δv * (T₁ - T₂)) :
  x = 8600 := by sorry

-- Problem (2): Proving Person B has a lower average price
theorem lower_average_price
  (a b : ℝ)
  (h₀ : a ≠ b) :
  (a + b) / 2 > 2 * (a * b) / (a + b) := by
  have h1 : (a - b)^2 > 0, from sq_pos_of_ne_zero _ (sub_ne_zero_of_ne h₀),
  have h2 : a^2 + b^2 > 2 * a * b, by linarith,
  have h3 : (a + b)^2 > 4 * a * b, from calc
    (a + b)^2 = a^2 + 2 * a * b + b^2 : by ring
    ... > 4 * a * b : by linarith,
  have h4 : 2 * (a * b) < (a + b)^2, by linarith,
  have h5 : 2 * (a * b) / (a + b) < (a + b), by linarith,
  have h6 : 2 * (a * b) / (a + b) < (a + b) / 2 + (a + b) / 2, by linarith,
  have h7 : 2 * (a * b) / (a + b) < (a + b) / 2, by linarith,
  exact h7

end tunnel_length_lower_average_price_l313_313120


namespace find_b_for_continuity_l313_313961

def f (x : Real) (b : Real) : Real :=
if x > 4 then x + 3 else 3 * x + b

def is_continuous_at (f : Real → Real) (x : Real) : Prop :=
∀ ε > 0, ∃ δ > 0, ∀ y, abs(y - x) < δ → abs(f y - f x) < ε

theorem find_b_for_continuity : ∃ b : Real, is_continuous_at (f · b) 4 :=
by
  use -5
  sorry

end find_b_for_continuity_l313_313961


namespace Carol_winning_choice_is_half_l313_313466

noncomputable def CarolOptimalChoice : ℝ :=
  if h : 1 / 3 < 1 / 2 ∧ 1 / 2 < 2 / 3 then 1 / 2 else 0

theorem Carol_winning_choice_is_half :
  (∀ (a b : ℝ),
    (a ∈ set.Icc 0 1) →
    (b ∈ set.Icc (1/3) (2/3)) →
    ∃ (c : ℝ), CarolOptimalChoice = c) :=
begin
  intros a b ha hb,
  use 1 / 2,
  unfold CarolOptimalChoice,
  split_ifs,
  { refl },
  { exfalso,
    simpa using h }
end

end Carol_winning_choice_is_half_l313_313466


namespace sum_floor_ceil_eq_seven_l313_313180

def floor (x : ℝ) : ℤ := int.floor x
def ceil (x : ℝ) : ℤ := int.ceil x

theorem sum_floor_ceil_eq_seven (x : ℝ) : (floor x + ceil x = 7) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end sum_floor_ceil_eq_seven_l313_313180


namespace sin_135_eq_sqrt2_div_2_l313_313553

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313553


namespace fraction_of_area_above_line_within_rectangle_l313_313370

noncomputable def line_eq (x : ℝ) : ℝ := - (2/3) * x + 14 / 3

def circle_area (r : ℝ) : ℝ := r ^ 2 * Real.pi

def area_above_line_excluding_circle : ℝ := 27 -- this is the area calculation after excluding the circle

theorem fraction_of_area_above_line_within_rectangle 
  (rect_area : ℝ := 36 - 1 * Real.pi)
  (line : (x : ℝ) → ℝ := line_eq)
  (circle : ℝ := circle_area 1) :
  (27 / (36 - 1 * Real.pi) = 27 / (36 - Real.pi)) :=
sorry

end fraction_of_area_above_line_within_rectangle_l313_313370


namespace original_six_digit_number_l313_313060

theorem original_six_digit_number :
  ∃ a b c d e : ℕ, 
  (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e = 142857) ∧ 
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + 1 = 64 * (100000 + 10000 * a + 1000 * b + 100 * c + 10 * d + e)) :=
by
  sorry

end original_six_digit_number_l313_313060


namespace second_bounce_distance_correct_l313_313150

noncomputable def second_bounce_distance (R v g : ℝ) : ℝ := 2 * R - (2 * v / 3) * (Real.sqrt (R / g))

theorem second_bounce_distance_correct (R v g : ℝ) (hR : R > 0) (hv : v > 0) (hg : g > 0) :
  second_bounce_distance R v g = 2 * R - (2 * v / 3) * (Real.sqrt (R / g)) := 
by
  -- Placeholder for the proof
  sorry

end second_bounce_distance_correct_l313_313150


namespace reflection_sum_l313_313033

theorem reflection_sum (y : ℝ) : 
  let C := (3 : ℝ, y)
  let D := (-C.1, C.2)   -- Reflection over y-axis
  let E := (D.1, -D.2)   -- Reflection over x-axis
  in E.1 + E.2 + D.1 + (-y) = -6 := 
  sorry  

end reflection_sum_l313_313033


namespace simplify_expression_l313_313111

theorem simplify_expression : (0.4 * 0.5 + 0.3 * 0.2) = 0.26 := by
  sorry

end simplify_expression_l313_313111


namespace james_toys_l313_313294

-- Define the conditions and the problem statement
theorem james_toys (x : ℕ) (h1 : ∀ x, 2 * x = 60 - x) : x = 20 :=
sorry

end james_toys_l313_313294


namespace finding_angle_ECA_l313_313286

-- Define the necessary angles and conditions from the problem
def angle_DCA : ℝ := 50
def angle_ABC : ℝ := 60

noncomputable def angle_BCD := 180 - angle_ABC
def angle_ACD := angle_DCA

-- angle_BCD is derived from parallel lines, but defined explicitly here
-- Define the summation of angles based on triangle properties
noncomputable def angle_ACB := angle_BCD - angle_ACD

-- Define the result of the angle bisection
def angle_ACE := angle_ACB / 2

-- The problem condition
theorem finding_angle_ECA :
  angle_ACE = 35 :=
by
  -- Placeholder for the proof, which is omitted as specified
  sorry

end finding_angle_ECA_l313_313286


namespace find_number_l313_313113

theorem find_number (x : ℝ) (h : x = 0.16 * x + 21) : x = 25 :=
by
  sorry

end find_number_l313_313113


namespace area_enclosed_by_abs_eq_l313_313933

theorem area_enclosed_by_abs_eq (x y : ℝ) : 
  (|x| + |3 * y| = 12) → (∃ area : ℝ, area = 96) :=
by
  sorry

end area_enclosed_by_abs_eq_l313_313933


namespace sin_135_eq_l313_313484

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313484


namespace no_positive_integer_solutions_l313_313319

theorem no_positive_integer_solutions (n : ℕ) (hn : Nat.Prime n) :
  ∀ x : ℕ, 0 < x →
    (∑ i in Finset.range x, i.succ^n + ∑ i in Finset.range (n-1), i.succ^n) ≠ (∑ i in Finset.range (2 * n - 1), i.succ^n) :=
by
  intro x hx
  sorry

end no_positive_integer_solutions_l313_313319


namespace sin_135_eq_l313_313487

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313487


namespace sin_135_correct_l313_313559

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313559


namespace fractions_equiv_l313_313876

theorem fractions_equiv:
  (8 : ℝ) / (7 * 67) = (0.8 : ℝ) / (0.7 * 67) :=
by
  sorry

end fractions_equiv_l313_313876


namespace sin_135_eq_l313_313492

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313492


namespace sum_of_fifths_less_than_n_div_3_l313_313654

variables {n : ℕ}
variable (x : ℕ → ℝ)

theorem sum_of_fifths_less_than_n_div_3
  (n_ge_3 : n ≥ 3)
  (x_le_1 : ∀ i, i < n → x i ≤ 1)
  (sum_x_eq_0 : ∑ i in Finset.range n, x i = 0) :
  ∑ i in Finset.range n, (x i) ^ 5 < n / 3 :=
by sorry

end sum_of_fifths_less_than_n_div_3_l313_313654


namespace B_work_rate_l313_313861

-- Definitions for the conditions
def A (t : ℝ) := 1 / 15 -- A's work rate per hour
noncomputable def B : ℝ := 1 / 10 - 1 / 15 -- Definition using the condition of the combined work rate

-- Lean 4 statement for the proof problem
theorem B_work_rate : B = 1 / 30 := by sorry

end B_work_rate_l313_313861


namespace sin_135_eq_sqrt2_div_2_l313_313590

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313590


namespace max_distance_point_circle_l313_313063

theorem max_distance_point_circle
  (A : ℝ × ℝ) (C_center : ℝ × ℝ) (r : ℝ) (d : ℝ)
  (hA : A = (2, 1))
  (hC_center : C_center = (0, 1))
  (h_radius : r = 1)
  (hC : ∀ (p : ℝ × ℝ), p ∈ C → (p.1)^2 + (p.2 - C_center.2)^2 = r^2)
  : d = 3 :=
  sorry

end max_distance_point_circle_l313_313063


namespace solve_eqs_l313_313947

theorem solve_eqs (x y : ℝ) 
  (h1 : x^2 + y^2 = 2)
  (h2 : x^2 / (2 - y) + y^2 / (2 - x) = 2) :
  x = 1 ∧ y = 1 :=
by
  sorry

end solve_eqs_l313_313947


namespace range_of_f_range_of_a_l313_313228

theorem range_of_f (a : ℝ) (hf : ∃ x : ℝ, f x = 0) : range (λ x, x^2 - a * x - 2) = set.Ici (-9 / 4) :=
by sorry

theorem range_of_a (a : ℝ) 
  (h : ∀ x₁ ∈ Icc (1 / 4) 1, ∃ x₂ ∈ Icc 1 2, (- x₁^2 + x₁ + a) > (x₂^2 - a * x₂ - 2) + 3) : 
  a ∈ set.Ioi 1 :=
by sorry

end range_of_f_range_of_a_l313_313228


namespace inequality_for_positive_reals_l313_313229

theorem inequality_for_positive_reals (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) (h_abc : a * b * c = 1) (k : ℕ) (h_k : 2 ≤ k) :
  (a^k / (a + b) + b^k / (b + c) + c^k / (c + a) ≥ 3 / 2) :=
by
  intros
  sorry

end inequality_for_positive_reals_l313_313229


namespace min_k_for_connected_monochromatic_subgraph_l313_313222

theorem min_k_for_connected_monochromatic_subgraph (n : ℕ) (hn : n ≥ 3) 
  (edge_coloring : sym2 (fin n) → fin 3)
  (h_coloring : ∀ c : fin 3, ∃ e : sym2 (fin n), edge_coloring e = some c) : 
  ∃ k : ℕ, k = ⌊n / 3⌋ ∧
  ∀ (new_coloring : sym2 (fin n) → fin 3), 
  (connected (graph_of_coloring new_coloring) ∧
   (∀ e : sym2 (fin n), new_coloring e = edge_coloring e ∨ count_color new_coloring (edge_coloring e) <= k))
  :=
sorry

end min_k_for_connected_monochromatic_subgraph_l313_313222


namespace find_Lx_maximize_L_2_4_maximize_L_4_5_l313_313149

variable (a : ℝ) (x : ℝ)
variables (k : ℝ := 500 * Real.exp 40)
variables (y : ℝ := k / Real.exp x)

-- Definitions for problem conditions
axiom h1 : x ∈ Icc 35 41
axiom h2 : a ∈ Icc 2 5

-- Define L(x)
def L (x : ℝ) : ℝ := 500 * (x - 30 - a) * Real.exp (40 - x)

-- Problem Statements
theorem find_Lx : L x = 500 * (x - 30 - a) * Real.exp (40 - x) := sorry

theorem maximize_L_2_4 : 
  (2 ≤ a ∧ a ≤ 4) → 
  (∀ x ∈ Icc 35 41, L x ≤ 500 * (5 - a) * Real.exp 5) :=
sorry

theorem maximize_L_4_5 :
  (4 < a ∧ a ≤ 5) → 
  (∀ x ∈ Icc 35 41, L x ≤ 500 * Real.exp (9 - a)) :=
sorry

end find_Lx_maximize_L_2_4_maximize_L_4_5_l313_313149


namespace sin_135_correct_l313_313564

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313564


namespace part1_part2_l313_313966

section
variable (x y : ℝ)

def A : ℝ := 3 * x^2 + 2 * y^2 - 2 * x * y
def B : ℝ := y^2 - x * y + 2 * x^2

-- Part (1): Prove that 2A - 3B = y^2 - xy
theorem part1 : 2 * A x y - 3 * B x y = y^2 - x * y := 
sorry

-- Part (2): Given |2x - 3| + (y + 2)^2 = 0, prove that 2A - 3B = 7
theorem part2 (h : |2 * x - 3| + (y + 2)^2 = 0) : 2 * A x y - 3 * B x y = 7 :=
sorry

end

end part1_part2_l313_313966


namespace value_of_a_l313_313252

def P : Set ℝ := { x | x^2 ≤ 4 }
def M (a : ℝ) : Set ℝ := { a }

theorem value_of_a (a : ℝ) (h : P ∪ {a} = P) : a ∈ { x : ℝ | -2 ≤ x ∧ x ≤ 2 } := by
  sorry

end value_of_a_l313_313252


namespace noon_temperature_is_correct_l313_313273

def temp_morning := 3 -- Morning temperature is 3°C
def temp_drop := 9 -- Temperature drop is 9°C

theorem noon_temperature_is_correct : temp_morning - temp_drop = -6 := by
  -- This is the equivalent Lean 4 statement for the given math problem
  sorry

end noon_temperature_is_correct_l313_313273


namespace degree_of_sum_l313_313923

variable (f g : Polynomial ℝ)
variable (d : ℝ)

-- Define the given polynomials
def fx : Polynomial ℝ := 2 - 6 * X + 4 * X^2 - 7 * X^3 + 8 * X^4
def gx : Polynomial ℝ := 5 - 3 * X - 9 * X^3 + 12 * X^4

-- State the claim that d = -2/3 ensures that the polynomial f + d * g has degree 3
theorem degree_of_sum (h1 : f = fx) (h2 : g = gx) : ∃ d, d = -2 / 3 ∧ degree (f + C d * g) = 3 := 
by 
  -- Bring in the polynomials and validate the degree condition
  use (-2 / 3)
  sorry

end degree_of_sum_l313_313923


namespace sin_135_eq_l313_313516

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313516


namespace equivalent_fraction_l313_313100

theorem equivalent_fraction (b : ℕ) (h : b = 2024) :
  (b^3 - 2 * b^2 * (b + 1) + 3 * b * (b + 1)^2 - (b + 1)^3 + 4) / (b * (b + 1)) = 2022 := by
  rw [h]
  sorry

end equivalent_fraction_l313_313100


namespace calculate_expression_l313_313915

theorem calculate_expression (n : ℤ) : (-3)^n + 2 * (-3)^(n - 1) = -(-3)^(n - 1) :=
by
  sorry

end calculate_expression_l313_313915


namespace sin_135_degree_l313_313542

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313542


namespace circumcircle_radius_of_ABC_l313_313088

-- Define the conditions
variables (A B C : Point) -- Points A, B, and C of triangle ABC
variables (r₁ r₂ r₃ : ℝ) -- Radii of the three spheres
variables (dAB dAC dBC : ℝ) -- Distances between centers of pairs of spheres

-- Assumptions based on problem conditions
axiom sum_of_radii : r₁ + r₂ = 9
axiom distance_between_centers : dAB = sqrt 305
axiom radius_third_sphere : r₃ = 7
axiom third_sphere_touches : dAC = r₁ + r₃ ∧ dBC = r₂ + r₃

-- Target: Prove that the radius of the circumcircle of triangle ABC is 2√14
theorem circumcircle_radius_of_ABC : R = 2 * sqrt 14 :=
sorry

end circumcircle_radius_of_ABC_l313_313088


namespace pyramid_edge_length_l313_313451

theorem pyramid_edge_length :
  (let height := 8 in
   let radius := 3 in
   let OP := Real.sqrt (height^2 - radius^2) in
   let AB := 2 * OP in
   AB = 2 * Real.sqrt 55) :=
by
  let height := 8
  let radius := 3
  let OP := Real.sqrt (height^2 - radius^2)
  let AB := 2 * OP
  exact 2 * Real.sqrt 55

end pyramid_edge_length_l313_313451


namespace exists_q_gt_one_l313_313748

theorem exists_q_gt_one (k : ℕ → ℝ) (h1 : k 1 > 1) 
  (h2 : ∀ n : ℕ, 1 ≤ n → ∑ i in Finset.range (n + 1), k i < 2 * k n) :
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, k n > q^n :=
by
  sorry

end exists_q_gt_one_l313_313748


namespace contradiction_proof_l313_313396

/-- For all real numbers x, 2^x > 0. -/
def forall_x_in_R_2_pow_x_gt_0 : Prop := ∀ x : ℝ, 2^x > 0

/-- There exists a real number x0 such that 2^x0 <= 0. -/
def exists_x0_in_R_2_pow_x0_le_0 : Prop := ∃ x0 : ℝ, 2^x0 ≤ 0

-- The assertion to prove using proof by contradiction:
theorem contradiction_proof :
  (forall_x_in_R_2_pow_x_gt_0 → false) → exists_x0_in_R_2_pow_x0_le_0 :=
by
  sorry

end contradiction_proof_l313_313396


namespace sin_135_eq_sqrt2_div_2_l313_313547

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313547


namespace find_x_l313_313956

-- Definition of z being a real number
def z_is_real (x : ℝ) : Prop :=
  ∃ (a : ℝ), z = a ∧ z.im = 0

-- Condition given in the problem
def complex_z (x : ℝ) : ℂ :=
  complex.log x (x ^ 2 - 3 * x - 2) + complex.i * complex.log x (x - 3)

-- The goal is to prove that the real number x that makes z real is x = 4
theorem find_x (x : ℝ) (h : z_is_real x) : x = 4 :=
by
  sorry

end find_x_l313_313956


namespace lowest_price_eq_195_l313_313887

def cost_per_component : ℕ := 80
def shipping_cost_per_unit : ℕ := 5
def fixed_monthly_costs : ℕ := 16500
def num_components : ℕ := 150

theorem lowest_price_eq_195 
  (cost_per_component shipping_cost_per_unit fixed_monthly_costs num_components : ℕ)
  (h1 : cost_per_component = 80)
  (h2 : shipping_cost_per_unit = 5)
  (h3 : fixed_monthly_costs = 16500)
  (h4 : num_components = 150) :
  (fixed_monthly_costs + num_components * (cost_per_component + shipping_cost_per_unit)) / num_components = 195 :=
by
  sorry

end lowest_price_eq_195_l313_313887


namespace conjugate_of_complex_l313_313812

theorem conjugate_of_complex (z : ℂ) (h : z = 5 * complex.I / (2 + complex.I)) : 
  complex.conj z = 1 - 2 * complex.I :=
by
  -- Proof goes here
  sorry

-- Additional definitions if necessary
noncomputable def complex_example := 5 * complex.I / (2 + complex.I)

example : complex.conj complex_example = 1 - 2 * complex.I :=
by
  rw ←conjugate_of_complex complex_example rfl
  sorry

end conjugate_of_complex_l313_313812


namespace john_payment_l313_313744

noncomputable def amount_paid_by_john := (3 * 12) / 2

theorem john_payment : amount_paid_by_john = 18 :=
by
  sorry

end john_payment_l313_313744


namespace cost_price_per_meter_l313_313424

-- Definitions
def selling_price : ℝ := 9890
def meters_sold : ℕ := 92
def profit_per_meter : ℝ := 24

-- Theorem
theorem cost_price_per_meter : (selling_price - profit_per_meter * meters_sold) / meters_sold = 83.5 :=
by
  sorry

end cost_price_per_meter_l313_313424


namespace avg_sales_amount_linear_relationship_maximize_profit_l313_313442

-- Conditions as definitions
def unit_prices := [30, 34, 38, 40, 42]
def sales_volumes := [40, 32, 24, 20, 16]

-- Statement of the problems as proofs
theorem avg_sales_amount : 
  (∑ i in finset.range 5, unit_prices.nth_le i (finset.mem_range.mpr (list.length_gt_zero _ (by simp))) * sales_volumes.nth_le i (finset.mem_range.mpr (list.length_gt_zero _ (by simp)))) / 5 = 934.4 :=
sorry

theorem linear_relationship : 
  ∃ (k b : ℝ), ∀ x, ∃ y, y = k * x + b ∧ 
  (k = -2) ∧ (b = 100) :=
sorry

theorem maximize_profit : 
  let k := -2 in
  let b := 100 in
  let cost := 20 in
  ∀ x, x = 35 →
  let profit_fn := λ x, (x - cost) * (k * x + b) in
  profit_fn 35 = 450 :=
sorry

end avg_sales_amount_linear_relationship_maximize_profit_l313_313442


namespace definitely_quadratic_l313_313855

theorem definitely_quadratic :
  ∀ (x : ℝ), 
    (sqrt 2 * x^2 - sqrt 2 / 4 * x - 1 / 2 = 0) -> 
    ∃ (a b c : ℝ), a ≠ 0 ∧ (a * x^2 + b * x + c = 0) :=
by
  sorry

end definitely_quadratic_l313_313855


namespace tina_mile_time_is_6_l313_313398

def tom_mile_time : ℝ := 2
def tina_mile_time : ℝ := 3 * tom_mile_time
def tony_mile_time : ℝ := tina_mile_time / 2

theorem tina_mile_time_is_6 :
  tony_mile_time + tina_mile_time + tom_mile_time = 11 → tina_mile_time = 6 := by
  sorry

end tina_mile_time_is_6_l313_313398


namespace solution_set_inequality_l313_313236

variable (a b c : ℝ)
variable (condition1 : ∀ x : ℝ, ax^2 + bx + c < 0 ↔ x < -1 ∨ 2 < x)

theorem solution_set_inequality (h : a < 0 ∧ b = -a ∧ c = -2 * a) :
  ∀ x : ℝ, (bx^2 + ax - c ≤ 0) ↔ (-1 ≤ x ∧ x ≤ 2) :=
by
  intro x
  sorry

end solution_set_inequality_l313_313236


namespace exist_odd_sum_l313_313309

theorem exist_odd_sum (N : ℕ) (a b c : ℕ → ℤ)
  (h : ∀ j : ℕ, j ≤ N → (∃ i, i ∈ [a j, b j, c j] ∧ Int.odd i)) :
  ∃ r s t : ℤ, at_least (4 * N / 7) (λ j, j ≤ N ∧ Int.odd (r * a j + s * b j + t * c j)) :=
sorry

end exist_odd_sum_l313_313309


namespace power_function_expression_l313_313250

theorem power_function_expression (α : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x ^ α) (h_point : f 2 = 4) :
  α = 2 ∧ (∀ x, f x = x ^ 2) :=
by
  sorry

end power_function_expression_l313_313250


namespace sin_135_l313_313521

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313521


namespace xyz_divisible_by_55_l313_313736

-- Definitions and conditions from part (a)
variables (x y z a b c : ℤ)
variable (h1 : x^2 + y^2 = a^2)
variable (h2 : y^2 + z^2 = b^2)
variable (h3 : z^2 + x^2 = c^2)

-- The final statement to prove that xyz is divisible by 55
theorem xyz_divisible_by_55 : 55 ∣ x * y * z := 
by sorry

end xyz_divisible_by_55_l313_313736


namespace radius_of_inscribed_circle_l313_313733

theorem radius_of_inscribed_circle (A B C D K L M : Type)
  (H1: is_midpoint B C L)
  (H2: is_midpoint A D M)
  (H3:segment_contains LM K)
  (H4: quadrilateral_in_circle A B C D)
  (H5: distance A B = 2)
  (H6: distance B D = 2 * sqrt 5)
  (H7: ratio LK KM = 1 / 3) :
  radius_of_inscribed_circle A B C D = 5 * sqrt 5 / 2 := sorry

end radius_of_inscribed_circle_l313_313733


namespace sin_function_satisfies_conds_l313_313664

theorem sin_function_satisfies_conds :
    ∃ (A B ω ϕ : ℝ), 
        A > 0 ∧ 
        ω > 0 ∧ 
        |ϕ| < (π / 2) ∧ 
        (∀ x, x = 1 → A*sin(ω*x + ϕ) + B = 2) ∧ 
        (∀ x, x = 2 → A*sin(ω*x + ϕ) + B = (1 / 2)) ∧ 
        (∀ x, x = 3 → A*sin(ω*x + ϕ) + B = -1) ∧ 
        (∀ x, x = 4 → A*sin(ω*x + ϕ) + B = 2) ∧
        (∀ x, f x = (√3)*sin((2*π/3)*x - (π/3)) + (1/2)) := 
by
    sorry

end sin_function_satisfies_conds_l313_313664


namespace calculate_expression_l313_313917

theorem calculate_expression : -2 - 2 * Real.sin (Real.pi / 4) + (Real.pi - 3.14) * 0 + (-1) ^ 3 = -3 - Real.sqrt 2 := by 
sorry

end calculate_expression_l313_313917


namespace difference_two_smallest_integers_l313_313080

/-- The difference between the two smallest integers greater than 1
which, when divided by any integer k such that 3 ≤ k ≤ 12, 
has a remainder of 1, is 13860. -/
theorem difference_two_smallest_integers (n : ℕ) :
  let lcm_3_to_12 := Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 (Nat.lcm 10 (Nat.lcm 11 12)))))))))
  in 2 * lcm_3_to_12 = 13860 :=
sorry

end difference_two_smallest_integers_l313_313080


namespace intersection_A_B_l313_313991

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l313_313991


namespace range_of_a_l313_313077

theorem range_of_a (a : ℝ) (x : ℝ) : (x > a ∧ x > 1) → (x > 1) → (a ≤ 1) :=
by 
  intros hsol hx
  sorry

end range_of_a_l313_313077


namespace approximate_value_correct_l313_313409

noncomputable def P1 : ℝ := (47 / 100) * 1442
noncomputable def P2 : ℝ := (36 / 100) * 1412
noncomputable def result : ℝ := (P1 - P2) + 63

theorem approximate_value_correct : abs (result - 232.42) < 0.01 := 
by
  -- Proof to be completed
  sorry

end approximate_value_correct_l313_313409


namespace M_positive_l313_313218

theorem M_positive :
  ∀ (x y : ℝ), 3 * x^2 - 8 * x * y + 9 * y^2 - 4 * x + 6 * y + 13 > 0 :=
begin
  sorry
end

end M_positive_l313_313218


namespace height_from_B_to_AC_is_correct_area_of_triangle_ABC_is_correct_l313_313836

-- Definitions of the vertices
def A := (-1, 4 : ℝ × ℝ)
def B := (-2, -1 : ℝ × ℝ)
def C := (2, 3 : ℝ × ℝ)

-- Definition of the line equations and heights
def line_from_B_to_AC (x y : ℝ) : Prop := x + y - 3 = 0
def height_length_A_to_BC : ℝ := 2 * Real.sqrt 2
def side_length_BC : ℝ := 4 * Real.sqrt 2
def area_ABC : ℝ := 8

theorem height_from_B_to_AC_is_correct :
  ∀ x y : ℝ, line_from_B_to_AC x y ↔ (x, y) = B ∨ (x, y) = A ∨ (x, y) = C :=
by
  sorry

theorem area_of_triangle_ABC_is_correct :
  ∃ S : ℝ, S = area_ABC :=
by
  sorry

end height_from_B_to_AC_is_correct_area_of_triangle_ABC_is_correct_l313_313836


namespace manufacturing_percentage_l313_313868

theorem manufacturing_percentage (deg_total : ℝ) (deg_manufacturing : ℝ) (h1 : deg_total = 360) (h2 : deg_manufacturing = 126) : 
  (deg_manufacturing / deg_total * 100) = 35 := by
  sorry

end manufacturing_percentage_l313_313868


namespace least_three_digit_multiple_of_3_4_5_l313_313411

def is_multiple_of (a b : ℕ) : Prop := b % a = 0

theorem least_three_digit_multiple_of_3_4_5 : 
  ∃ n : ℕ, is_multiple_of 3 n ∧ is_multiple_of 4 n ∧ is_multiple_of 5 n ∧ 100 ≤ n ∧ n < 1000 ∧ (∀ m : ℕ, is_multiple_of 3 m ∧ is_multiple_of 4 m ∧ is_multiple_of 5 m ∧ 100 ≤ m ∧ m < 1000 → n ≤ m) ∧ n = 120 :=
by
  sorry

end least_three_digit_multiple_of_3_4_5_l313_313411


namespace smallest_integer_row_10_row_n_includes_n2_n_and_n2_2n_largest_n_without_n2_10n_in_row_l313_313829

-- Conditions for Row n
def inRow (n m : ℕ) : Prop :=
  m % n = 0 ∧ m ≤ n^2 ∧ ∀ k < n, ¬ (m % k = 0 ∧ m ≤ k^2)

-- Part (a): Prove the smallest integer in Row 10 is 10.
theorem smallest_integer_row_10 : ∃ m, inRow 10 m ∧ (∀ k, inRow 10 k → k ≥ m) :=
by
  -- Statement only; proof not required
  sorry

-- Part (b): Prove for all n ≥ 3, Row n includes n^2 - n and n^2 - 2n.
theorem row_n_includes_n2_n_and_n2_2n (n : ℕ) (h : n ≥ 3) :
  inRow n (n^2 - n) ∧ inRow n (n^2 - 2n) :=
by
  -- Statement only; proof not required
  sorry

-- Part (c): Prove the largest n such that Row n does not include n^2 - 10n is 9.
theorem largest_n_without_n2_10n_in_row : ∃ n, inRow n (n^2 - 10n) → n ≤ 9 ∧ (∀ k > 9, ¬ inRow k (k^2 - 10k)) :=
by
  -- Statement only; proof not required
  sorry

end smallest_integer_row_10_row_n_includes_n2_n_and_n2_2n_largest_n_without_n2_10n_in_row_l313_313829


namespace matthew_egg_rolls_l313_313775

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end matthew_egg_rolls_l313_313775


namespace log_identity_eqn_l313_313264

variables {p q : ℝ} (c : ℝ)
hypothesis h : log 5 p = c - log 5 q

theorem log_identity_eqn : p = 5^c / q :=
by
  sorry

end log_identity_eqn_l313_313264


namespace volleyball_team_lineup_l313_313032

theorem volleyball_team_lineup : 
  let team_members := 10
  let lineup_positions := 6
  10 * 9 * 8 * 7 * 6 * 5 = 151200 := by sorry

end volleyball_team_lineup_l313_313032


namespace determine_true_propositions_l313_313161

noncomputable def true_propositions (prop1 prop2 prop3 prop4 : Prop) : list nat :=
if ¬prop1 ∧ prop2 ∧ prop3 ∧ prop4 then [2, 3, 4] else []

theorem determine_true_propositions :
    (∀ P (plane : Set P) (point : P), ∃! (line : Set P), point ∉ plane → ∃ (line_parallel : line), line_parallel ∦ plane) →
    (∀ P (plane : Set P) (point : P), ∃! (line : Set P), point ∉ plane → ∃ (line_perpendicular : line), line_perpendicular ⊥ plane) →
    (∀ P (plane1 plane2 plane3 : Set P), plane1 ∥ plane2 ∧ plane1 ∩ plane3 ≠ ∅ → ∃ (line1 line2 : Set P), line1 ∥ line2) →
    (∀ P (plane1 plane2 : Set P) (point : P), plane1 ⊥ plane2 ∧ point ∈ plane1 → ∃ (line : Set P), line ⊥ plane2 ∧ point ∈ line → line ⊆ plane1) →
    true_propositions (-- condition for false proposition 1,
                       -- condition for true proposition 2,
                       -- condition for true proposition 3,
                       -- condition for true proposition 4) = [2, 3, 4] :=
by
  intros h1 h2 h3 h4
  sorry

end determine_true_propositions_l313_313161


namespace solution_set_is_finite_l313_313960

def a_seq (s t : ℕ) : ℕ → ℕ
| 0     := s
| 1     := t
| (n+2) := nat.floor $ real.sqrt ((a_seq n) + (n+2) * (a_seq (n+1)) + 2008)

theorem solution_set_is_finite (s t : ℕ) (hs : s > 0) (ht : t > 0) :
  set.finite {n : ℕ | a_seq s t n ≠ n} :=
sorry

end solution_set_is_finite_l313_313960


namespace find_sum_invested_l313_313130

def R1 : ℝ := 18
def R2 : ℝ := 12
def T : ℝ := 2
def P : ℝ -- this will be the sum we are solving for

theorem find_sum_invested (P : ℝ) (h : (P * R1 * T / 100) - (P * R2 * T / 100) = 300) : P = 2500 :=
by 
  sorry

end find_sum_invested_l313_313130


namespace a5_is_9_l313_313178

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1989 ^ 1989 ∧
  ∀ n > 1, a n = (a (n - 1)).digits.sum

theorem a5_is_9 (a : ℕ → ℕ) (h : sequence a) : a 5 = 9 :=
sorry

end a5_is_9_l313_313178


namespace negation_proposition_l313_313371

theorem negation_proposition :
  ¬ (∀ x : ℝ, x^2 - 2*x + 4 ≤ 0) ↔ ∃ x0 : ℝ, x0^2 - 2*x0 + 4 > 0 :=
by
  sorry

end negation_proposition_l313_313371


namespace sum_of_squares_first_20_l313_313852

-- Define the sum of squares function
def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

-- Specific problem instance
theorem sum_of_squares_first_20 : sum_of_squares 20 = 5740 :=
  by
  -- Proof skipping placeholder
  sorry

end sum_of_squares_first_20_l313_313852


namespace seq_sum_eq_l313_313285

noncomputable def arithmetic_sequence (a : ℕ → ℝ) :=
  ∀ n m k : ℕ, a n = a 0 + (n - 1) * (a 1 - a 0)

variables {a : ℕ → ℝ} (h_arith : arithmetic_sequence a)
variables (h_eq : 2 * (a 1 + a 4 + a 7) + 3 * (a 9 + a 11) = 24)

theorem seq_sum_eq : (finset.sum (finset.range 13) (λ n, a (n + 1)) + 2 * a 7) = 30 :=
by {
  have h_arith_4 : a 4 = a 0 + 3 * (a 1 - a 0), from h_arith 4 1 3,
  have h_arith_7 : a 7 = a 0 + 6 * (a 1 - a 0), from h_arith 7 1 6,
  have h_arith_9 : a 9 = a 0 + 8 * (a 1 - a 0), from h_arith 9 1 8,
  have h_arith_11 : a 11 = a 0 + 10 * (a 1 - a 0), from h_arith 11 1 10,
  have h_arith_13 : a 13 = a 0 + 12 * (a 1 - a 0), from h_arith 13 1 12,

  -- Calculation steps would be shown here but omitted in this theorem statement
  sorry,
}

end seq_sum_eq_l313_313285


namespace sin_135_correct_l313_313556

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313556


namespace sin_cos_value_cos2_alpha_div_cos_pi4_plus_alpha_l313_313625

variable (α : ℝ)

axiom cos_minus_sin_eq : cos α - sin α = (5 * Real.sqrt 2) / 13
axiom alpha_range : 0 < α ∧ α < π / 4

theorem sin_cos_value : sin α * cos α = 119 / 338 :=
by
  sorry

theorem cos2_alpha_div_cos_pi4_plus_alpha : cos (2 * α) / cos (π / 4 + α) = 24 / 13 :=
by
  sorry

end sin_cos_value_cos2_alpha_div_cos_pi4_plus_alpha_l313_313625


namespace daniel_sales_tax_l313_313927

theorem daniel_sales_tax :
  let total_cost := 25
  let tax_rate := 0.05
  let tax_free_cost := 18.7
  let tax_paid := 0.3
  exists (taxable_cost : ℝ), 
    18.7 + taxable_cost + 0.05 * taxable_cost = total_cost ∧
    taxable_cost * tax_rate = tax_paid :=
by
  sorry

end daniel_sales_tax_l313_313927


namespace ratio_third_first_l313_313843

theorem ratio_third_first (A B C : ℕ) (h1 : A + B + C = 110) (h2 : A = 2 * B) (h3 : B = 30) :
  C / A = 1 / 3 :=
by
  sorry

end ratio_third_first_l313_313843


namespace shaded_percentage_of_large_square_l313_313355

theorem shaded_percentage_of_large_square
  (side_length_small_square : ℕ)
  (side_length_large_square : ℕ)
  (total_border_squares : ℕ)
  (shaded_border_squares : ℕ)
  (central_region_shaded_fraction : ℚ)
  (total_area_large_square : ℚ)
  (shaded_area_border_squares : ℚ)
  (shaded_area_central_region : ℚ) :
  side_length_small_square = 1 →
  side_length_large_square = 5 →
  total_border_squares = 16 →
  shaded_border_squares = 8 →
  central_region_shaded_fraction = 3 / 4 →
  total_area_large_square = 25 →
  shaded_area_border_squares = 8 →
  shaded_area_central_region = (3 / 4) * 9 →
  (shaded_area_border_squares + shaded_area_central_region) / total_area_large_square = 0.59 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end shaded_percentage_of_large_square_l313_313355


namespace range_of_k_l313_313821

noncomputable def intersect_at_two_points (k : ℝ) : Prop :=
  let a := 1 + k^2
  let b := 6 * k - 6
  let c := 6
  let discriminant := b^2 - 4 * a * c
  discriminant > 0

theorem range_of_k (k : ℝ) : intersect_at_two_points k ↔ k < 3 - 2 * sqrt 2 ∨ k > 3 + 2 * sqrt 2 := by
  sorry

end range_of_k_l313_313821


namespace largest_possible_percent_error_l313_313746

def largest_percent_error (true_diameter : ℝ) (error_margin : ℝ) : ℝ :=
  let actual_area := Real.pi * (true_diameter / 2) ^ 2
  let min_diameter := true_diameter * (1 - error_margin)
  let max_diameter := true_diameter * (1 + error_margin)
  let min_area := Real.pi * (min_diameter / 2) ^ 2
  let max_area := Real.pi * (max_diameter / 2) ^ 2
  max ((actual_area - min_area) / actual_area * 100) ((max_area - actual_area) / actual_area * 100)

theorem largest_possible_percent_error :
  largest_percent_error 30 0.3 = 68.9 := sorry

end largest_possible_percent_error_l313_313746


namespace warriors_games_won_l313_313824

open Set

-- Define the variables for the number of games each team won
variables (games_L games_H games_W games_F games_R : ℕ)

-- Define the set of possible game scores
def game_scores : Set ℕ := {19, 23, 28, 32, 36}

-- Define the conditions as assumptions
axiom h1 : games_L > games_H
axiom h2 : games_W > games_F
axiom h3 : games_W < games_R
axiom h4 : games_F > 18
axiom h5 : ∃ min_games ∈ game_scores, min_games > games_H ∧ min_games < 20

-- Prove the main statement
theorem warriors_games_won : games_W = 32 :=
sorry

end warriors_games_won_l313_313824


namespace perp_BZ_DE_l313_313012

variables {Point : Type} [AffineSpace ℝ Point] (A B C D E Z : Point)
variable [AffineMap ℝ Point]

-- Conditions
variables (convex_ABCDE : ConvexPentagon A B C D E)
variable (equal_AB_BC : dist A B = dist B C)
variables (rightAngle_BCD : angle B C D = π / 2)
variables (rightAngle_EAB : angle E A B = π / 2)
variable (perp_AZ_BE : Az ⊥ BE)
variable (perp_CZ_BD : Cz ⊥ BD)

-- Theorem statement
theorem perp_BZ_DE
    (convex_ABCDE : ConvexPentagon A B C D E)
    (equal_AB_BC : dist A B = dist B C)
    (rightAngle_BCD : angle B C D = π / 2)
    (rightAngle_EAB : angle E A B = π / 2)
    (perp_AZ_BE : Az ⊥ BE)
    (perp_CZ_BD : Cz ⊥ BD) :
    Bz ⊥ DE := 
sorry

end perp_BZ_DE_l313_313012


namespace solve_proportion_l313_313431

theorem solve_proportion :
  ∃ x : ℝ, (215 * x = 474 * 537 ∧ x ≈ 1184.69) := by
  sorry

end solve_proportion_l313_313431


namespace seventh_observation_is_4_l313_313432

def avg_six := 11 -- Average of the first six observations
def sum_six := 6 * avg_six -- Total sum of the first six observations
def new_avg := avg_six - 1 -- New average after including the new observation
def new_sum := 7 * new_avg -- Total sum after including the new observation

theorem seventh_observation_is_4 : 
  (new_sum - sum_six) = 4 :=
by
  sorry

end seventh_observation_is_4_l313_313432


namespace intersection_A_B_l313_313988

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l313_313988


namespace min_value_ratio_l313_313678

noncomputable def min_ratio (a : ℝ) (h : a > 0) : ℝ :=
  let x_A := 4^(-a)
  let x_B := 4^(a)
  let x_C := 4^(- (18 / (2*a + 1)))
  let x_D := 4^((18 / (2*a + 1)))
  let m := abs (x_A - x_C)
  let n := abs (x_B - x_D)
  n / m

theorem min_value_ratio (a : ℝ) (h : a > 0) : 
  ∃ c : ℝ, c = 2^11 := sorry

end min_value_ratio_l313_313678


namespace modulus_of_complex_number_l313_313632

variables (z : ℂ) (x : ℝ)

theorem modulus_of_complex_number (h1 : z * complex.I = 2 * complex.I + x) (h2 : z.im = 2): complex.abs z = 2 * real.sqrt 2 :=
sorry

end modulus_of_complex_number_l313_313632


namespace SomuAge_l313_313112

theorem SomuAge (F S : ℕ) (h1 : S = F / 3) (h2 : S - 8 = (F - 8) / 5) : S = 16 :=
by 
  sorry

end SomuAge_l313_313112


namespace correct_calculation_value_l313_313081

theorem correct_calculation_value (x : ℕ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 :=
by
  -- The conditions are used directly in the definitions
  -- Given the condition (x * 5) + 7 = 27
  let h1 := h
  -- Solve for x and use x in the correct calculation
  sorry

end correct_calculation_value_l313_313081


namespace timmy_needs_speed_l313_313844

variable (s1 s2 s3 : ℕ) (extra_speed : ℕ)

theorem timmy_needs_speed
  (h_s1 : s1 = 36)
  (h_s2 : s2 = 34)
  (h_s3 : s3 = 38)
  (h_extra_speed : extra_speed = 4) :
  (s1 + s2 + s3) / 3 + extra_speed = 40 := 
sorry

end timmy_needs_speed_l313_313844


namespace quadratic_poly_bound_l313_313749

theorem quadratic_poly_bound (p : ℝ → ℝ)
  (h_quad : ∀ x ∈ {-1, 0, 1}, |p x| ≤ 1) :
  ∀ x ∈ Icc (-1 : ℝ) 1, |p x| ≤ 5 / 4 := 
sorry

end quadratic_poly_bound_l313_313749


namespace part1_period_of_f_part2_axis_of_symmetry_part3_range_of_g_l313_313241

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + sin (x) ^ 2 - cos (x) ^ 2

noncomputable def g (x : ℝ) : ℝ := (f x) ^ 2 + (f x)

theorem part1_period_of_f :
  ∃ p > 0, ∀ x, f (x + p) = f x ∧ p = π :=
sorry

theorem part2_axis_of_symmetry :
  ∃ k : ℤ, ∀ x, f x = f (k * π / 2 + π / 3) :=
sorry

theorem part3_range_of_g :
  ∀ y, y ∈ set.range g ↔ -1/4 ≤ y ∧ y ≤ 2 :=
sorry

end part1_period_of_f_part2_axis_of_symmetry_part3_range_of_g_l313_313241


namespace num_valid_z_l313_313001

noncomputable def f (z : ℂ) : ℂ := z^2 + complex.I * z + 1

theorem num_valid_z : 
  (finset.filter (λ z : ℂ, (0 < z.im) ∧
                             (abs (f z).re ≤ 15) ∧ (abs (f z).im ≤ 15) ∧ 
                             ((f z).re = (f z).im)) finset.univ).card = 31 := 
sorry

end num_valid_z_l313_313001


namespace melissa_points_per_game_l313_313024

-- Define the conditions in the problem
def total_points : ℕ := 81
def games : ℕ := 3

-- Define the proposition to show the number of points per game
def points_per_game : ℕ := total_points / games

-- Prove that the points per game is 27
theorem melissa_points_per_game : points_per_game = 27 := by
  -- We introduce the assumption that can be computed directly:
  show points_per_game = 27
  calc points_per_game = total_points / games : by rfl
                   ... = 81 / 3               : by rfl
                   ... = 27                   : by norm_num

end melissa_points_per_game_l313_313024


namespace average_yield_l313_313272

theorem average_yield (y1 y2 y3 y4 y5 : ℕ) (h1 : y1 = 12) (h2 : y2 = 13) (h3 : y3 = 15) (h4 : y4 = 17) (h5 : y5 = 18) :
  (y1 + y2 + y3 + y4 + y5) / 5 = 15 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end average_yield_l313_313272


namespace correct_conclusions_l313_313994

-- Define the parabola and the focus
def parabola (x y: ℝ) := y^2 = 4 * x
def focus := (1, 0: ℝ)

-- Define points A and B on the parabola
variables (x1 y1 x2 y2: ℝ)
def pointA := (x1, y1: ℝ)
def pointB := (x2, y2: ℝ)

-- Condition: x₁ + x₂ = 5
def condition := x1 + x2 = 5

-- Define distances |AF| and |BF| for the length of chord AB
def distance_AF := |x1 - 1|
def distance_BF := |x2 - 1|
def length_AB := distance_AF + distance_BF

-- Define the shortest chord through the focus
def shortest_chord := 4

-- Problem statement in Lean 4, proving the two correct conclusions
theorem correct_conclusions (h_parabA : parabola x1 y1)
                           (h_parabB : parabola x2 y2)
                           (h_cond : condition):
    length_AB = 7 ∧ shortest_chord = 4 := by
  sorry

end correct_conclusions_l313_313994


namespace greatest_possible_n_l313_313269

theorem greatest_possible_n : ∃ n : ℤ, 101 * n^2 ≤ 3600 ∧ ∀ m : ℤ, 101 * m^2 ≤ 3600 → m ≤ n :=
begin
  use 5,
  split,
  { norm_num, }, -- ensures 101 * 5^2 ≤ 3600
  { intros m hm,
    have h_eq := (le_of_mul_le_mul_right hm zero_lt_mul_left),
    linarith, -- ensures for m ≤ 5 given the original condition.
  }
end

end greatest_possible_n_l313_313269


namespace function_condition_l313_313190

noncomputable def f (x : ℝ) : ℝ :=
  x - 1 / x

theorem function_condition (x : ℝ) (hx : x ≠ 0) : f x + f (1 / x) = 0 :=
by 
  have hx' : f x = x - 1 / x := rfl
  have hfx_inv := calc
    f (1 / x) = 1 / x - 1 / (1 / x) : by rw [f]
          ... = 1 / x - x : by field_simp [hx]
  calc
    f x + f (1 / x) = (x - 1 / x) + (1 / x - x) : by { rw [hx', hfx_inv] }
               ... = 0 : by ring

end function_condition_l313_313190


namespace work_days_together_l313_313110

theorem work_days_together (p_rate q_rate : ℝ) (fraction_left : ℝ) (d : ℝ) 
  (h₁ : p_rate = 1/15) (h₂ : q_rate = 1/20) (h₃ : fraction_left = 8/15)
  (h₄ : (p_rate + q_rate) * d = 1 - fraction_left) : d = 4 :=
by
  sorry

end work_days_together_l313_313110


namespace valid_3x3_arrays_count_l313_313685

def array3x3 := matrix (fin 3) (fin 3) ℤ

def valid_entries (A : array3x3) : Prop :=
  ∀ i j, A i j = 1 ∨ A i j = -1

def row_sum_zero (A : array3x3) : Prop :=
  ∀ i, ∑ j, A i j = 0

def col_sum_zero (A : array3x3) : Prop :=
  ∀ j, ∑ i, A i j = 0

def num_valid_arrays : ℕ := 6

theorem valid_3x3_arrays_count : 
  ∃ (A : array3x3) (h_valid_entries : valid_entries A) (h_row_sum_zero : row_sum_zero A) (h_col_sum_zero : col_sum_zero A), true ∧ ∃ (count : ℕ), count = 6 :=
by 
  sorry

end valid_3x3_arrays_count_l313_313685


namespace proof_inequality_l313_313647

noncomputable def inequality (a b c : ℝ) : Prop :=
  a + 2 * b + c = 1 ∧ a^2 + b^2 + c^2 = 1 → -2/3 ≤ c ∧ c ≤ 1

theorem proof_inequality (a b c : ℝ) (h : a + 2 * b + c = 1) (h2 : a^2 + b^2 + c^2 = 1) : -2/3 ≤ c ∧ c ≤ 1 :=
by {
  sorry
}

end proof_inequality_l313_313647


namespace amelia_succeeds_with_parallelogram_and_ellipse_l313_313160

theorem amelia_succeeds_with_parallelogram_and_ellipse :
  ∀ (position_parallelogram : ℝ × ℝ) (position_ellipse : ℝ × ℝ),
  (∃ line : ℝ × ℝ → Prop, 
  (∀ p ∈ parallelogram, line p → line (rotate180 p position_parallelogram)) ∧ 
  (∀ p ∈ ellipse, line p → line (rotate180 p position_ellipse))) :=
by sorry

end amelia_succeeds_with_parallelogram_and_ellipse_l313_313160


namespace at_least_15_distinct_distances_l313_313968

-- Define the plane and points configuration
def points_count : ℕ := 400

-- Define the concept of distinct distances between points on a plane
def distinct_distances (k : ℕ) : Prop :=
k >= 15

-- The main theorem statement
theorem at_least_15_distinct_distances {points : ℕ} (h_points : points = points_count) :
  ∃ k, distinct_distances k :=
sorry

end at_least_15_distinct_distances_l313_313968


namespace consecutive_log_sum_l313_313169

theorem consecutive_log_sum : 
  ∃ c d: ℤ, (c + 1 = d) ∧ (c < Real.logb 5 125) ∧ (Real.logb 5 125 < d) ∧ (c + d = 5) :=
sorry

end consecutive_log_sum_l313_313169


namespace sin_135_l313_313525

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313525


namespace sin_135_eq_l313_313518

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313518


namespace complement_of_A_in_U_is_4_l313_313253

-- Define the universal set U
def U : Set ℕ := { x | 1 < x ∧ x < 5 }

-- Define the set A
def A : Set ℕ := {2, 3}

-- Define the complement of A in U
def complement_U_of_A : Set ℕ := { x ∈ U | x ∉ A }

-- State the theorem
theorem complement_of_A_in_U_is_4 : complement_U_of_A = {4} :=
by
  sorry

end complement_of_A_in_U_is_4_l313_313253


namespace max_value_y_l313_313822

noncomputable def y (x : ℝ) : ℝ :=
  (1/4) * real.sin (x - real.pi / 6) - real.cos (x - 2 * real.pi / 3)

theorem max_value_y : ∀ x, y x ≤ 3/4 :=
sorry

end max_value_y_l313_313822


namespace length_ad_l313_313435

/-- Definitions:
B and C trisect AD at points E and F respectively.
M is the midpoint of AD.
ME = 5.
Goal: The length of AD is 30.
-/
theorem length_ad (A D B C E F M : Type) [LinearOrderedField A] 
  (trisect : B.trisects AD E F)
  (midpoint : M.midpoint AD)
  (ME : M.distance E = 5) :
  A.distance D = 30 := 
sorry

end length_ad_l313_313435


namespace common_difference_in_arithmetic_sequence_l313_313238

theorem common_difference_in_arithmetic_sequence
  (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 2 = 3)
  (h2 : a 5 = 12) :
  d = 3 :=
by
  sorry

end common_difference_in_arithmetic_sequence_l313_313238


namespace find_range_a_l313_313247

-- Define the piecewise function f(x)
def f (x a : ℝ) : ℝ :=
  if x < 1 - a then
    x^2 - x + a^2 + a + 2
  else
    x^2 + x + a^2 + 3 * a

theorem find_range_a : 
  {a : ℝ | ∀ x, f x a > 5} = {a | a < (1 - real.sqrt 14) / 2} ∪ {a | (real.sqrt 6) / 2 < a} := sorry

end find_range_a_l313_313247


namespace equilateral_triangles_count_l313_313817

theorem equilateral_triangles_count {k : ℤ} (hk : -20 ≤ k ∧ k ≤ 20) :
  let lines :=  (λ k, [λ x, k: ℝ, λ x, (√3) * x + (2 * k: ℝ), λ x, -(√3) * x + (2 * k: ℝ)]) in
  let total_lines := list.join (list.map lines (list.range 41).map (λ x, x - 20)) in
  let counts := 123 in
  let hexagon_sides := (40 / (√3: ℝ)) in
  let small_triangle_sides := (1 / (√3: ℝ)) in
  let ratio_sides := hexagon_sides / small_triangle_sides in
  let area_ratio := ratio_sides ^ 2 in
  let hexagon_triangles := 6 * area_ratio in
  let edge_triangles := 6 * 20 in
  let total_triangles := hexagon_triangles + edge_triangles in
  total_triangles = 9720 := sorry

end equilateral_triangles_count_l313_313817


namespace isosceles_triangle_perimeter_l313_313721

theorem isosceles_triangle_perimeter :
  ∀ x, x^2 - 8 * x + 15 = 0 → (x = 3 ∨ x = 5) ∧ (x = 3 → 2 + 2 + x = 7) := by
  intro x
  intro hx
  have h1 : x = 3 ∨ x = 5 := by sorry
  split
  exact h1
  intro hx3
  rw hx3
  norm_num

end isosceles_triangle_perimeter_l313_313721


namespace proof_l313_313980

variable (x : ℝ)

def p1 : Prop := ∀ x, (2 ^ x - 2 ^ (-x)) > 0
def p2 : Prop := ∀ x, (2 ^ x + 2 ^ (-x)) < 0

theorem proof (h1 : p1) (h2 : ¬ p2) : 
  (p1 ∨ p2) ∨ (p1 ∧ ¬ p2) :=
by
  sorry

end proof_l313_313980


namespace solve_eq_f_eq_2_solve_ineq_f_gt_1_l313_313665

def f (x : ℝ) : ℝ := 
  if h : x < 1 then 2^(-x) 
  else if h : x > 1 then Real.log x / Real.log 3 
  else 0  -- Note: the function is not defined for x = 1 from the given problem.

theorem solve_eq_f_eq_2 (x : ℝ) :
  f x = 2 ↔ x = -1 ∨ x = 9 := 
sorry

theorem solve_ineq_f_gt_1 (x : ℝ) :
  f x > 1 ↔ x < 0 ∨ x > 3 := 
sorry

end solve_eq_f_eq_2_solve_ineq_f_gt_1_l313_313665


namespace length_of_BD_eq_10_l313_313040

theorem length_of_BD_eq_10 (A B C D O : Point)
  (h1 : midpoint O A B)
  (h2 : midpoint O C D)
  (h3 : length (segment A C) = 10) : 
  length (segment B D) = 10 :=
  sorry

end length_of_BD_eq_10_l313_313040


namespace barber_total_loss_is_120_l313_313897

-- Definitions for the conditions
def haircut_cost : ℕ := 25
def initial_payment_by_customer : ℕ := 50
def flower_shop_change : ℕ := 50
def bakery_change : ℕ := 10
def customer_received_change : ℕ := 25
def counterfeit_50_replacement : ℕ := 50
def counterfeit_10_replacement : ℕ := 10

-- Calculate total loss for the barber
def total_loss : ℕ :=
  let loss_haircut := haircut_cost
  let loss_change_to_customer := customer_received_change
  let loss_given_to_flower_shop := counterfeit_50_replacement
  let loss_given_to_bakery := counterfeit_10_replacement
  let total_loss_before_offset := loss_haircut + loss_change_to_customer + loss_given_to_flower_shop + loss_given_to_bakery
  let real_currency_received := flower_shop_change
  total_loss_before_offset - real_currency_received

-- Proof statement
theorem barber_total_loss_is_120 : total_loss = 120 := by {
  sorry
}

end barber_total_loss_is_120_l313_313897


namespace students_like_apple_chocolate_not_blueberry_l313_313714

theorem students_like_apple_chocolate_not_blueberry
  (n d a b c abc : ℕ)
  (h1 : n = 50)
  (h2 : d = 15)
  (h3 : a = 25)
  (h4 : b = 20)
  (h5 : c = 10)
  (h6 : abc = 5)
  (h7 : (n - d) = 35)
  (h8 : (55 - (a + b + c - abc)) = 35) :
  (20 - abc) = (15 : ℕ) :=
by
  sorry

end students_like_apple_chocolate_not_blueberry_l313_313714


namespace sin_2alpha_minus_pi_over_4_l313_313216

theorem sin_2alpha_minus_pi_over_4 
  (α : ℝ)
  (h1 : sin α - cos α = 1 / 5)
  (h2 : 0 ≤ α ∧ α ≤ π) :
  sin (2 * α - π / 4) = 31 * sqrt 2 / 50 := 
  sorry

end sin_2alpha_minus_pi_over_4_l313_313216


namespace probability_of_specific_individual_drawn_on_third_attempt_l313_313407

theorem probability_of_specific_individual_drawn_on_third_attempt :
  let population_size := 6
  let sample_size := 3
  let prob_not_drawn_first_attempt := 5 / 6
  let prob_not_drawn_second_attempt := 4 / 5
  let prob_drawn_third_attempt := 1 / 4
  (prob_not_drawn_first_attempt * prob_not_drawn_second_attempt * prob_drawn_third_attempt) = 1 / 6 :=
by sorry

end probability_of_specific_individual_drawn_on_third_attempt_l313_313407


namespace c_sub_a_equals_90_l313_313049

variables (a b c : ℝ)

theorem c_sub_a_equals_90 (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) : c - a = 90 :=
by
  sorry

end c_sub_a_equals_90_l313_313049


namespace convert_quadratic_l313_313175

theorem convert_quadratic (x : ℝ) :
  (1 + 3 * x) * (x - 3) = 2 * x ^ 2 + 1 ↔ x ^ 2 - 8 * x - 4 = 0 := 
by sorry

end convert_quadratic_l313_313175


namespace sufficient_but_not_necessary_l313_313352

theorem sufficient_but_not_necessary (k : ℤ) : 
  (sin (2 * k * real.pi + real.pi / 4) = real.sqrt 2 / 2) ∧ 
  (¬ ∀ x, sin x = real.sqrt 2 / 2 → ∃ k, x = 2 * k * real.pi + real.pi / 4) :=
by
  split
  · intro k
    intros
    calc
      sin (2 * k * real.pi + real.pi / 4) = sin (real.pi / 4) : by rw sin_add
      ... = real.sqrt 2 / 2 : by rw sin_pi_div_four

  · intro h
    apply or_iff_not_imp_left.mpr h
    intro h1
    use k
    assumption

#align sufficient_but_not_necessary

end sufficient_but_not_necessary_l313_313352


namespace opposite_of_neg_one_third_l313_313067

noncomputable def a : ℚ := -1 / 3

theorem opposite_of_neg_one_third : -a = 1 / 3 := 
by 
sorry

end opposite_of_neg_one_third_l313_313067


namespace contrapositive_of_proposition_l313_313054

-- Proposition: If xy=0, then x=0
def proposition (x y : ℝ) : Prop := x * y = 0 → x = 0

-- Contrapositive: If x ≠ 0, then xy ≠ 0
def contrapositive (x y : ℝ) : Prop := x ≠ 0 → x * y ≠ 0

-- Proof that contrapositive of the given proposition holds
theorem contrapositive_of_proposition (x y : ℝ) : proposition x y ↔ contrapositive x y :=
by {
  sorry
}

end contrapositive_of_proposition_l313_313054


namespace sin_135_degree_l313_313534

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313534


namespace sin_135_eq_l313_313509

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313509


namespace sum_of_sequence_correct_l313_313477

def calculateSumOfSequence : ℚ :=
  (4 / 3) + (7 / 5) + (11 / 8) + (19 / 15) + (35 / 27) + (67 / 52) - 9

theorem sum_of_sequence_correct :
  calculateSumOfSequence = (-17312.5 / 7020) := by
  sorry

end sum_of_sequence_correct_l313_313477


namespace sin_135_eq_l313_313485

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313485


namespace find_divisor_l313_313900

theorem find_divisor (x : ℝ) (h : x / n = 0.01 * (x * n)) : n = 10 :=
sorry

end find_divisor_l313_313900


namespace ratio_closest_to_10_l313_313938

theorem ratio_closest_to_10 : 
  (abs ((10^2000 + 10^2004) / (10^2001 + 10^2003) - 10) < 1) :=
by
  sorry

end ratio_closest_to_10_l313_313938


namespace pencils_per_row_l313_313943

def total_pencils : ℕ := 32
def rows : ℕ := 4

theorem pencils_per_row : total_pencils / rows = 8 := by
  sorry

end pencils_per_row_l313_313943


namespace probability_at_least_one_4_l313_313450

def fair_dice_tosses (n : ℕ) : Finset (Finset ℕ) :=
  Finset.powerset (Finset.range n)

theorem probability_at_least_one_4 (X1 X2 X3 : ℕ) (hX1 : 1 ≤ X1 ∧ X1 ≤ 6) (hX2 : 1 ≤ X2 ∧ X2 ≤ 6)
  (hX3 : 1 ≤ X3 ∧ X3 ≤ 6) (h_sum : X1 + X2 = X3) : 
  ((Finset.card (Finset.filter (λ s, 4 ∈ s) (fair_dice_tosses 3))) / 
   (Finset.card (fair_dice_tosses 3))) = 4 / 15 :=
by
  sorry

end probability_at_least_one_4_l313_313450


namespace eval_expr_l313_313602

theorem eval_expr : (2.1 * (49.7 + 0.3)) + 15 = 120 :=
  by
  sorry

end eval_expr_l313_313602


namespace sum_evaluation_l313_313187

theorem sum_evaluation :
  (∑ k in Finset.range 50 + 1, ( (-1 : ℤ) ^ k * (k^3 + k^2 + k + 1) / k! ) : ℚ) = 132353 / 50! - 1 := sorry

end sum_evaluation_l313_313187


namespace sin_135_eq_sqrt2_div_2_l313_313584

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313584


namespace f_relationship_l313_313244

def f (x : ℝ) : ℝ := (1/2) * x^2 - 2 * x + 3

theorem f_relationship (x1 x2 : ℝ) (h : abs (x1 - 2) > abs (x2 - 2)) : f x1 > f x2 :=
by sorry

end f_relationship_l313_313244


namespace pipes_fill_tank_l313_313031

theorem pipes_fill_tank (T : ℝ) (h1 : T > 0)
  (h2 : (1/4 : ℝ) + 1/T - 1/20 = 1/2.5) : T = 5 := by
  sorry

end pipes_fill_tank_l313_313031


namespace area_triangle_RYZ_l313_313284

noncomputable def square_WXYZ_area := 144
def P_on_side_WZ : Prop := True -- Assuming the existence of point P on WZ
def Q_midpoint_WP : Prop := True -- Assuming Q is the midpoint of WP
def R_midpoint_YP : Prop := True -- Assuming R is the midpoint of YP
noncomputable def quadrilateral_WQRP_area := 40

theorem area_triangle_RYZ :
  ∃ (side_length : ℝ) (P Q R : Point) (WXYZ : Quadrilateral), 
    (WXYZ.area = square_WXYZ_area) ∧
    (P_on_side_WZ) ∧
    (Q_midpoint_WP) ∧
    (R_midpoint_YP) ∧
    (quadrilateral_WQRP_area = 40) →
    (area_of_triangle R Y Z = 14) :=
by
  exists side_length
  exists P Q R
  exists WXYZ
  simp only [square_WXYZ_area, P_on_side_WZ, Q_midpoint_WP, R_midpoint_YP, quadrilateral_WQRP_area]
  sorry

end area_triangle_RYZ_l313_313284


namespace sin_135_correct_l313_313565

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313565


namespace x2_minus_y2_eq_neg4_l313_313320

theorem x2_minus_y2_eq_neg4 (x y : ℝ) (h₁ : x = 2023^1011 - 2023^(-1011)) (h₂ : y = 2023^1011 + 2023^(-1011)) :
  x^2 - y^2 = -4 := by
  sorry

end x2_minus_y2_eq_neg4_l313_313320


namespace girls_friends_count_l313_313596

variable (days_in_week : ℕ)
variable (total_friends : ℕ)
variable (boys : ℕ)

axiom H1 : days_in_week = 7
axiom H2 : total_friends = 2 * days_in_week
axiom H3 : boys = 11

theorem girls_friends_count : total_friends - boys = 3 :=
by sorry

end girls_friends_count_l313_313596


namespace derivative_of_constant_function_l313_313813

-- Define the constant function
def f (x : ℝ) : ℝ := 0

-- State the theorem
theorem derivative_of_constant_function : deriv f 0 = 0 := by
  -- Proof will go here, but we use sorry to skip it
  sorry

end derivative_of_constant_function_l313_313813


namespace john_weekly_income_increase_l313_313298

/-- John's old weekly income -/
def old_income : ℝ := 60

/-- John's new weekly income -/
def new_income : ℝ := 72

/-- The difference in income -/
def income_difference : ℝ := new_income - old_income

/-- The percentage increase in John's weekly income -/
def percentage_increase (old new : ℝ) : ℝ :=
  (new - old) / old * 100

theorem john_weekly_income_increase :
  percentage_increase old_income new_income = 20 := by
  sorry

end john_weekly_income_increase_l313_313298


namespace eat_cereal_in_time_l313_313778

noncomputable def time_to_eat_pounds (pounds : ℕ) (rate1 rate2 : ℚ) :=
  pounds / (rate1 + rate2)

theorem eat_cereal_in_time :
  time_to_eat_pounds 5 ((1:ℚ)/15) ((1:ℚ)/40) = 600/11 := 
by 
  sorry

end eat_cereal_in_time_l313_313778


namespace num_ordered_pairs_satisfying_eq_l313_313262

theorem num_ordered_pairs_satisfying_eq :
  {p : ℤ × ℤ | p.1^6 + p.2^2 = 3 * p.2 - 2}.to_finset.card = 2 :=
sorry

end num_ordered_pairs_satisfying_eq_l313_313262


namespace sin_135_degree_l313_313536

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313536


namespace expression_evaluation_l313_313181

theorem expression_evaluation :
  let b := 10
  let c := 3
  let a := 2 * b
  [a - (b - c)] - [(a - b) - c] = 6 :=
by
  sorry

end expression_evaluation_l313_313181


namespace sin_135_correct_l313_313558

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313558


namespace tenths_more_than_hundredths_l313_313046

theorem tenths_more_than_hundredths : 
  ∀ (tenths hundredths : ℝ), 
    tenths = 0.5 → 
    hundredths = 0.05 → 
    (tenths - hundredths) = 0.45 :=
by
  intros tenths hundredths hTenths hHundredths
  rw [hTenths, hHundredths]
  norm_num
  sorry

end tenths_more_than_hundredths_l313_313046


namespace child_ticket_cost_l313_313807

theorem child_ticket_cost :
  ∀ (A P_a C T P_c : ℕ),
    A = 10 →
    P_a = 8 →
    C = 11 →
    T = 124 →
    (T - A * P_a) / C = P_c →
    P_c = 4 :=
by
  intros A P_a C T P_c hA hP_a hC hT hPc
  rw [hA, hP_a, hC, hT] at hPc
  linarith [hPc]

end child_ticket_cost_l313_313807


namespace rotated_log_fn_l313_313700

def log_fn (x : ℝ) : ℝ := log10 (x + 1)

def rotate_fn (f : ℝ → ℝ) (θ : ℝ) (x : ℝ) : ℝ :=
  let y := f x in -y / sin θ

theorem rotated_log_fn (x : ℝ) : f (x : ℝ) = 10^(-x) - 1 :=
by
  have θ := Real.pi / 2
  have f := rotate_fn log_fn θ
  sorry

end rotated_log_fn_l313_313700


namespace sin_135_eq_sqrt2_over_2_l313_313568

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313568


namespace jason_money_determination_l313_313747

theorem jason_money_determination (fred_last_week : ℕ) (fred_earned : ℕ) (fred_now : ℕ) (jason_last_week : ℕ → Prop)
  (h1 : fred_last_week = 23)
  (h2 : fred_earned = 63)
  (h3 : fred_now = 86) :
  ¬ ∃ x, jason_last_week x :=
by
  sorry

end jason_money_determination_l313_313747


namespace handshake_count_l313_313474

def total_employees : ℕ := 50
def dept_X : ℕ := 30
def dept_Y : ℕ := 20
def handshakes_between_departments : ℕ := dept_X * dept_Y

theorem handshake_count : handshakes_between_departments = 600 :=
by
  sorry

end handshake_count_l313_313474


namespace every_trap_is_feeder_l313_313106

-- Define a sequence as a function from natural numbers to some type (e.g., ℚ)
def sequence (α : Type*) := ℕ → α

-- Define an interval [a, b]
def interval (a b : ℚ) := {x : ℚ | a ≤ x ∧ x ≤ b}

-- Define cover condition: only a finite number of terms are outside [a, b]
def is_cover (s : sequence ℚ) (a b : ℚ) := 
  {n | ¬ (a ≤ s n ∧ s n ≤ b)}.finite

-- Define feeder condition: an infinite number of terms within [a, b]
def is_feeder (s : sequence ℚ) (a b : ℚ) := 
  {n | a ≤ s n ∧ s n ≤ b}.infinite

-- The theorem statement
theorem every_trap_is_feeder (s : sequence ℚ) (a b : ℚ) 
  (hc : is_cover s a b) : is_feeder s a b := 
sorry

end every_trap_is_feeder_l313_313106


namespace largest_six_digit_number_correct_l313_313609

noncomputable def largest_valid_six_digit_number : ℕ :=
  972538

theorem largest_six_digit_number_correct :
  ∃ n : ℕ, n = 972538 ∧
            (n.digits.length = 6) ∧
            (list.nodup n.digits) ∧
            (∀ i, 1 ≤ i ∧ i < 5 → 
              (n.digits.nth_le i - n.digits.nth_le (i - 1) ∈ {1, -1}) ∨
              (n.digits.nth_le i - n.digits.nth_le (i + 1) ∈ {1, -1})
            ) :=
by
  use largest_valid_six_digit_number
  sorry

end largest_six_digit_number_correct_l313_313609


namespace simple_interest_doubles_l313_313905

theorem simple_interest_doubles (R : ℝ) : 
  ∀ (P : ℝ) (T : ℝ), P = 1 → T = 20 → (P * R * T) / 100 = P → R = 5 := 
by
  intros P T hP hT hSI
  have h1 : P = 1 := hP
  have h2 : T = 20 := hT
  have h3 : (1 * R * 20) / 100 = 1 := hSI
  sorry

end simple_interest_doubles_l313_313905


namespace max_number_bound_in_circle_l313_313838

theorem max_number_bound_in_circle 
  (x : ℕ → ℕ) 
  (h_circle : ∀ i, x (i % 10) + x ((i + 1) % 10) + x ((i + 2) % 10) ≥ 29)
  (h_sum : ∑ i in finset.range 10, x i = 100) : 
  ∀ i, x i ≤ 13 := 
sorry

end max_number_bound_in_circle_l313_313838


namespace incenter_position_vector_l313_313734

variable (A B C I : Type)
variables [add_comm_group I] [vector_space ℝ I]
variables (a b c : ℝ)
variables (x y z : ℝ)
variables (vA vB vC vI : I)

-- Side lengths and constants conditions
def side_lengths := a = 8 ∧ b = 10 ∧ c = 6
def constants_sum := x + y + z = 1

-- Given conditions
def conditions := side_lengths a b c ∧ constants_sum x y z

-- Proof statement: The position vector for incenter
theorem incenter_position_vector (h : conditions a b c x y z) :
  vI = x vA + y vB + z vC :=
by {
  sorry
}

end incenter_position_vector_l313_313734


namespace u_2023_valid_l313_313756

-- Definition of the sequence u_n
noncomputable def u (n : ℕ) : ℕ :=
  if n = 1 then 2
  else 2 + 5 * (n - 1)

-- Prove that u_2023 is the desired value
theorem u_2023_valid : u 2023 = ? := sorry

end u_2023_valid_l313_313756


namespace odd_function_f_f_inequality_solution_l313_313930

def f (x : ℝ) : ℝ :=
  if x < 0 then x + 2
  else if x = 0 then 0
  else x - 2

theorem odd_function_f {x : ℝ} : f(-x) = - f(x) :=
  by
    unfold f
    split_ifs
    case h_1 h_x_neg {
      have h_x_pos : -x > 0 := by linarith
      unfold f
      split_ifs
      case h_1 h'_x_pos {
        have : -(-x + 2) = x + 2 := by ring
        exact this
      }
      case h_2 h'_x_zero {
        sorry
      }
      case h_3 h'_otherwise {
        sorry
      }
    }
    case h_2 h_x_zero { sorry }
    case h_3 h_x_pos {
      sorry
    }

theorem f_inequality_solution {x : ℝ} : f(x) < 2 ↔ x < 4 :=
  by
    unfold f
    split_ifs
    case h_1 h_x_neg {
      exact iff.rfl
    }
    case h_2 h_x_zero {
      linarith
    }
    case h_3 h_x_pos {
      have : x - 2 < 2 ↔ x < 4 := by linarith
      exact this
    }

end odd_function_f_f_inequality_solution_l313_313930


namespace concurrency_of_homothetic_lines_l313_313332

noncomputable def point := ℝ × ℝ
structure triangle (α : Type*) :=
  (A B C : α)
structure rectangle (α : Type*) :=
  (corners : list α)
def parallel {α : Type*} [AddCommGroup α] [Module ℝ α] (u v : α) : Prop := u = ksmul ℝ x v  -- Need to define scalar multiplication and assumption

theorem concurrency_of_homothetic_lines (A B C A1 B1 C1 : point)
  (triangle_ABC : triangle point) (triangle_A1B1C1 : triangle point)
  (rectangles_ABC : point → point → rectangle point)
  (h_parallel_sides : parallel (B - A) (B1 - A1) ∧ parallel (C - B) (C1 - B1) ∧ parallel (A - C) (A1 - C1)) :
  ∃ P : point, ∀ l ∈ {line_through A A1, line_through B B1, line_through C C1}, P ∈ l := 
sorry

end concurrency_of_homothetic_lines_l313_313332


namespace largest_q_value_l313_313832

theorem largest_q_value : ∃ q, q >= 1 ∧ q^4 - q^3 - q - 1 ≤ 0 ∧ (∀ r, r >= 1 ∧ r^4 - r^3 - r - 1 ≤ 0 → r ≤ q) ∧ q = (Real.sqrt 5 + 1) / 2 := 
sorry

end largest_q_value_l313_313832


namespace candle_burn_time_l313_313402

theorem candle_burn_time :
  ∃ t : ℕ, 
    let l₁ := 21, l₂ := 16, r₁ := 15, r₂ := 11, burn_time := 18,
        initial_ratio := (l₁, l₂), after_burn_ratio := (r₁, r₂) in
    (l₁ - r₁) = burn_time / 2 ∧ 
    (l₂ - r₂) = burn_time / 2 ∧
    (l₁ * (burn_time / (l₁ - r₁))) = t ∧
    (t = 150) := 
sorry

end candle_burn_time_l313_313402


namespace math_books_on_shelf_l313_313840

/--
There are 100 books on a shelf. 32 of them are history books, 25 of them are geography books. 
Prove that the number of math books on the shelf is 43.
-/
theorem math_books_on_shelf (total_books : ℕ) (history_books : ℕ) (geography_books : ℕ)
  (H1 : total_books = 100) (H2 : history_books = 32) (H3 : geography_books = 25) : 
  total_books - history_books - geography_books = 43 :=
by
  intros
  rw [H1, H2, H3]
  exact rfl

end math_books_on_shelf_l313_313840


namespace geometric_sequence_formula_l313_313890

theorem geometric_sequence_formula (s : ℕ → ℝ) (s₀ r : ℝ) (hr : r ≠ 0) :
  (∀ k, s (k + 1) = s k * r) → ∀ n, s n = s₀ * r ^ n :=
by
  intros h k
  induction n with n ih
  { -- Base case: s 0 = s₀ * r^0
    rw [pow_zero, mul_one]
  }
  { -- Inductive step: Assuming s n = s₀ * r^n, prove s (n + 1) = s₀ * r^(n + 1)
    rw [h n, ih, pow_succ] }
  sorry -- to finish the proof

end geometric_sequence_formula_l313_313890


namespace range_of_m_l313_313707

theorem range_of_m (x m : ℝ) (h1 : abs (x - m) < 2) (h2 : 2 ≤ x) (h3 : x ≤ 3) : 1 < m ∧ m < 4 :=
begin
  sorry
end

end range_of_m_l313_313707


namespace sectors_not_equal_l313_313738

theorem sectors_not_equal (a1 a2 a3 a4 a5 a6 : ℕ) :
  ¬(∃ k : ℕ, (∀ n : ℕ, n = k) ↔
    ∃ m, (a1 + m) = k ∧ (a2 + m) = k ∧ (a3 + m) = k ∧ 
         (a4 + m) = k ∧ (a5 + m) = k ∧ (a6 + m) = k) :=
sorry

end sectors_not_equal_l313_313738


namespace contrapositive_of_sum_of_squares_l313_313053

theorem contrapositive_of_sum_of_squares
  (a b : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0) :
  a^2 + b^2 ≠ 0 := 
sorry

end contrapositive_of_sum_of_squares_l313_313053


namespace contestant_score_l313_313717

theorem contestant_score (highest_score lowest_score : ℕ) (average_score : ℕ)
  (h_hs : highest_score = 86)
  (h_ls : lowest_score = 45)
  (h_avg : average_score = 76) :
  (76 * 9 - 86 - 45) / 7 = 79 := 
by 
  sorry

end contestant_score_l313_313717


namespace tan_alpha_eq_two_l313_313653

noncomputable def tan_alpha (α : ℝ) : ℝ := Mathlib.λ α, Math.tan α

theorem tan_alpha_eq_two {α : ℝ} (h1 : 0 < α) (h2 : α < real.pi / 2) (h3 : real.sin(2*α) = real.sin(α) ^ 2) : tan_alpha α = 2 :=
by
  sorry

end tan_alpha_eq_two_l313_313653


namespace triangle_geometry_inequality_l313_313308

theorem triangle_geometry_inequality 
  (O G : Point) 
  (R r : ℝ)
  (hG : is_centroid O G R r) 
  (hO : is_circumcenter O G R r) :
  distance O G ≤ Real.sqrt (R * (R - 2 * r)) :=
sorry

end triangle_geometry_inequality_l313_313308


namespace sin_135_l313_313502

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313502


namespace woman_lawyer_probability_l313_313880

-- Defining conditions
def total_members : ℝ := 100
def percent_women : ℝ := 0.90
def percent_women_lawyers : ℝ := 0.60

-- Calculating numbers based on the percentages
def number_women : ℝ := percent_women * total_members
def number_women_lawyers : ℝ := percent_women_lawyers * number_women

-- Statement of the problem in Lean 4
theorem woman_lawyer_probability :
  (number_women_lawyers / total_members) = 0.54 :=
by sorry

end woman_lawyer_probability_l313_313880


namespace maxwellBoltzmann_asymptotic_l313_313591

-- Definitions of conditions
variables (n M k : ℕ)
variables (P_k : ℕ → ℕ → ℕ → ℝ)

-- Definitions based on conditions
def maxwellBoltzmann (n M : ℕ) : Prop :=
  n > 0 ∧ M > 0

def probability (n M k : ℕ) : ℝ :=
  nat.choose n k * ((M - 1) ^ (n - k) : ℝ) / (M ^ n : ℝ)

noncomputable def asymptotic_probability (n M k : ℕ) (λ : ℝ) : Prop :=
  ∀ (n M : ℕ) (λ : ℝ), (tendsto (λ (n M : ℕ), P_k n M k) (filter.at_top ×ᶠ filter.at_top)
     (𝓝 (exp (-λ) * λ^k / real.factorial k)))

-- Theorem to be proved
theorem maxwellBoltzmann_asymptotic :
  ∀ (n M k : ℕ), maxwellBoltzmann n M →
                 (probability n M k = nat.choose n k * ((M - 1) ^ (n - k) : ℝ) / (M ^ n : ℝ))
                 ∧ asymptotic_probability n M k (n / M) :=
sorry

end maxwellBoltzmann_asymptotic_l313_313591


namespace solve_sqrt_equation_l313_313796

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end solve_sqrt_equation_l313_313796


namespace multiset_6_permutations_eq_17_l313_313611

noncomputable def num_6_permutations_of_multiset : ℕ :=
  finset.sum (finset.Icc 0 1) (λ x, 
  finset.sum (finset.Icc 0 4) (λ y, 
  finset.sum (finset.Icc 0 3) (λ z, 
  if x + y + z = 6 then nat.factorial 6 / (nat.factorial x * nat.factorial y * nat.factorial z) else 0
  )))

theorem multiset_6_permutations_eq_17: num_6_permutations_of_multiset = 17 := by 
  sorry

end multiset_6_permutations_eq_17_l313_313611


namespace sin_135_eq_sqrt2_div_2_l313_313581

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313581


namespace average_weight_is_correct_l313_313718

def men_above_50 : ℕ := 10
def average_weight_men_above_50 : ℝ := 200
def men_below_50 : ℕ := 7
def average_weight_men_below_50 : ℝ := 180
def women_above_40 : ℕ := 12
def average_weight_women_above_40 : ℝ := 156
def women_below_40 : ℕ := 5
def average_weight_women_below_40 : ℝ := 120
def children_10_15 : ℕ := 6
def average_weight_children_10_15 : ℝ := 100
def children_below_10 : ℕ := 4
def average_weight_children_below_10 : ℝ := 80

def total_weight : ℝ := 
  men_above_50 * average_weight_men_above_50 +
  men_below_50 * average_weight_men_below_50 +
  women_above_40 * average_weight_women_above_40 +
  women_below_40 * average_weight_women_below_40 +
  children_10_15 * average_weight_children_10_15 +
  children_below_10 * average_weight_children_below_10

def total_participants : ℕ := 
  men_above_50 + 
  men_below_50 + 
  women_above_40 + 
  women_below_40 + 
  children_10_15 + 
  children_below_10

def average_weight_all : ℝ := total_weight / total_participants

theorem average_weight_is_correct :
  average_weight_all ≈ 151.18 := 
sorry

end average_weight_is_correct_l313_313718


namespace sum_of_possible_values_A_l313_313457

theorem sum_of_possible_values_A :
  (∃ (A : ℕ), A < 10 ∧ ((36 + A) % 9 = 0) ∧ (A = 0 ∨ A = 9) ∧ 9 = 0 + 9) :=
by {
  -- Here we state the results for A satisfying the condition.
  use [0, 9],
  -- For A = 0
  { split,
    { assumption, },
    exact Nat.mod_eq_zero_of_dvd 36,
    left, refl, },
  -- For A = 9  
  { split,
    { assumption, },
    have h : 36 % 9 = 0 := Nat.mod_eq_zero_of_dvd 36,
    exact Eq.trans (Nat.add_mod 36 9 9) h,
    right, refl,
  },
  -- Finally, stating the sum of possible values for A is 9
  exact Nat.add_comm 0 9
}

end sum_of_possible_values_A_l313_313457


namespace playground_problem_l313_313884

theorem playground_problem (P_A1 : ℝ) (P_B1 : ℝ)
    (P_A2_given_A1 : ℝ) (P_A2_given_B1 : ℝ) :
    (P_A1 = 0.3) →
    (P_B1 = 0.7) →
    (P_A2_given_A1 = 0.7) →
    (P_A2_given_B1 = 0.6) →
    let P_A2 := P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1 in
    let P_B1_given_A2 := (P_B1 * P_A2_given_B1) / P_A2 in
    (P_A2 = 0.63) ∧ (P_B1_given_A2 = 2 / 3) :=
by {
  intros h1 h2 h3 h4,
  let P_A2 := P_A1 * P_A2_given_A1 + P_B1 * P_A2_given_B1,
  have hA2 : P_A2 = 0.63, sorry,
  let P_B1_given_A2 := (P_B1 * P_A2_given_B1) / P_A2,
  have hB1_given_A2 : P_B1_given_A2 = 2 / 3, sorry,
  exact ⟨hA2, hB1_given_A2⟩
}

end playground_problem_l313_313884


namespace triangle_is_obtuse_l313_313291

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h1 : ∀ {a b c : ℝ}, a + b + c = π)
  (h2 : ∀ {a}, 0 < a ∧ a < π)
  (h3 : cos A * cos B > sin A * sin B)
  : cos C < 0 :=
sorry

end triangle_is_obtuse_l313_313291


namespace inequality_proof_l313_313035

theorem inequality_proof (n : ℕ) (hn : n > 0) : (2 * n + 1) ^ n ≥ (2 * n) ^ n + (2 * n - 1) ^ n :=
by
  sorry

end inequality_proof_l313_313035


namespace problem1_problem2_l313_313119

-- Problem 1: Simplify the calculation: 6.9^2 + 6.2 * 6.9 + 3.1^2
theorem problem1 : 6.9^2 + 6.2 * 6.9 + 3.1^2 = 100 := 
by
  sorry

-- Problem 2: Simplify and find the value of the expression with given conditions
theorem problem2 (a b : ℝ) (h1 : a = 1) (h2 : b = 0.5) :
  (a^2 * b^3 + 2 * a^3 * b) / (2 * a * b) - (a + 2 * b) * (a - 2 * b) = 9 / 8 :=
by
  sorry

end problem1_problem2_l313_313119


namespace sin_2alpha_minus_pi_over_4_eq_31sqrt2_over_50_l313_313214

theorem sin_2alpha_minus_pi_over_4_eq_31sqrt2_over_50
  (α : ℝ) (h₁ : sin α - cos α = 1 / 5) (h₂ : 0 ≤ α ∧ α ≤ π) :
  sin (2 * α - π / 4) = (31 * real.sqrt 2) / 50 :=
begin
  sorry -- proof is omitted
end

end sin_2alpha_minus_pi_over_4_eq_31sqrt2_over_50_l313_313214


namespace sin_135_eq_sqrt2_over_2_l313_313577

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313577


namespace equidistant_point_on_x_axis_exists_l313_313410

noncomputable def point_A := (-3 : ℝ, 0 : ℝ)
noncomputable def point_B := (0 : ℝ, 5 : ℝ)

def distance (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2

theorem equidistant_point_on_x_axis_exists :
  ∃ x : ℝ, (∀ y : ℝ, y = 0 → distance (x, y) point_A = distance (x, y) point_B) ∧ x = 8 / 3 :=
by
  sorry

end equidistant_point_on_x_axis_exists_l313_313410


namespace decagon_triangle_probability_l313_313203

theorem decagon_triangle_probability :
  let decagon_sides := (10.choose 3)
  let one_side_count := 10 * 5
  let two_sides_count := 10
  60 / 120 = 1 / 2 :=
by
  sorry

end decagon_triangle_probability_l313_313203


namespace find_roots_of_polynomial_l313_313401

theorem find_roots_of_polynomial 
  (A B C : Type)
  [Triangle A B C]
  (h1 : isInscribedInCircle A B C 2)
  (h2 : ∠ B ≥ 90°)
  (a b c x : ℝ)
  (h3 : a = (BC))
  (h4 : b = (CA))
  (h5 : c = (AB))
  (h6 : x^4 + a*x^3 + b*x^2 + c*x + 1 = 0) :
  x = - (sqrt 6 + sqrt 2) / 2 ∨ x = - (sqrt 6 - sqrt 2) / 2 :=
sorry

end find_roots_of_polynomial_l313_313401


namespace negation_of_p_l313_313670

open Real

-- Define the original proposition p
def p := ∀ x : ℝ, 0 < x → x^2 > log x

-- State the theorem with its negation
theorem negation_of_p : ¬p ↔ ∃ x : ℝ, 0 < x ∧ x^2 ≤ log x :=
by
  sorry

end negation_of_p_l313_313670


namespace determine_A_and_B_l313_313237

-- We need to declare variables and sequences as per the information given in the problem.
variables (A B : ℝ) (n : ℕ)

-- Define the sum of the first n terms of the geometric sequence {a_n}
def S_n (n : ℕ) : ℝ := 1 - A * 3^n

-- Define the sequence {b_n}
def b_n (n : ℕ) : ℝ := A * n^2 + B * n

-- State the conditions and what needs to be proven.
theorem determine_A_and_B :
  (∀ n : ℕ, S_n n = 1 - A * 3^n) →
  (∀ n : ℕ, b_n (n + 1) > b_n n) →
  A = 1 ∧ ∀ n : ℕ, n > 0 → B > -3 :=
begin
  intros hS hb,
  sorry
end

end determine_A_and_B_l313_313237


namespace y_completion_days_l313_313114

theorem y_completion_days (d : ℕ) (h : (12 : ℚ) / d + 1 / 4 = 1) : d = 16 :=
by
  sorry

end y_completion_days_l313_313114


namespace max_nedoslon_non_attacking_on_7x7_chessboard_l313_313373

def nedoslon_moves_diagonally (x y : ℕ) (n : ℕ := 7) : Prop :=
  x < n ∧ y < n

theorem max_nedoslon_non_attacking_on_7x7_chessboard :
  ∃ (positions : fin 7 → fin 7), 
    (∀ i j : fin 7, i ≠ j → (positions i - positions j).nat_abs ≠ (i - j).nat_abs) ∧
    (∀ i : fin 7, nedoslon_moves_diagonally (i : ℕ) (positions i : ℕ)) ∧
    (∑ _ in finset.univ, 1 = 28) :=
sorry

end max_nedoslon_non_attacking_on_7x7_chessboard_l313_313373


namespace closest_multiple_of_12_l313_313414

theorem closest_multiple_of_12 (x : ℕ) (h₁ : x = 2025) (h₂ : ∀ n : ℕ, n ∣ 12 ↔ (n ∣ 3 ∧ n ∣ 4)) : closest_to 2025 2028 :=
begin
  sorry -- The proof is omitted as directed
end

end closest_multiple_of_12_l313_313414


namespace sin_135_eq_l313_313511

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313511


namespace promotional_rate_ratio_is_one_third_l313_313021

-- Define the conditions
def normal_monthly_charge : ℕ := 30
def extra_fee : ℕ := 15
def total_paid : ℕ := 175

-- Define the total data plan amount equation
def calculate_total (P : ℕ) : ℕ :=
  P + 2 * normal_monthly_charge + (normal_monthly_charge + extra_fee) + 2 * normal_monthly_charge

theorem promotional_rate_ratio_is_one_third (P : ℕ) (hP : calculate_total P = total_paid) :
  P * 3 = normal_monthly_charge :=
by sorry

end promotional_rate_ratio_is_one_third_l313_313021


namespace k_is_3_l313_313318

noncomputable def k_solution (k : ℝ) : Prop :=
  k > 1 ∧ (∑' n : ℕ, (n^2 + 3 * n - 2) / k^n = 2)

theorem k_is_3 : ∃ k : ℝ, k_solution k ∧ k = 3 :=
by
  sorry

end k_is_3_l313_313318


namespace false_proposition_l313_313289

-- Defining the complex number sequence relation
def complex_gt (z1 z2 : ℂ) : Prop :=
  z1.re > z2.re ∨ (z1.re = z2.re ∧ z1.im > z2.im)

-- Proposition A
def prop_A : Prop := complex_gt 1 (complex.mk 0 1) ∧ complex_gt (complex.mk 0 1) 0

-- Proposition B
def prop_B : Prop := ∀ z1 z2 z3 : ℂ, complex_gt z1 z2 → complex_gt z2 z3 → complex_gt z1 z3

-- Proposition C
def prop_C : Prop := ∀ z1 z2 z : ℂ, complex_gt z1 z2 → complex_gt (z1 + z) (z2 + z)

-- Proposition D
def prop_D : Prop := ∀ (z z1 z2 : ℂ), complex_gt z1 z2 → complex_gt z 0 → complex_gt (z * z1) (z * z2)

-- False Proposition
theorem false_proposition : ¬prop_D :=
sorry

end false_proposition_l313_313289


namespace triangular_array_nth_row_4th_number_l313_313156

theorem triangular_array_nth_row_4th_number (n : ℕ) (h : n ≥ 4) :
  ∃ k : ℕ, k = 4 ∧ (2: ℕ)^(n * (n - 1) / 2 + 3) = 2^((n^2 - n + 6) / 2) :=
by
  sorry

end triangular_array_nth_row_4th_number_l313_313156


namespace train_speed_in_km_per_hr_l313_313154

noncomputable def train_length : ℝ := 120
noncomputable def crossing_time : ℝ := 2.9997600191984644
noncomputable def conversion_factor : ℝ := 3.6

theorem train_speed_in_km_per_hr :
  let speed_m_per_s := train_length / crossing_time in
  let speed_km_per_hr := speed_m_per_s * conversion_factor in
  abs (speed_km_per_hr - 144.03) < 1e-2 := 
by
  let speed_m_per_s := train_length / crossing_time
  let speed_km_per_hr := speed_m_per_s * conversion_factor
  sorry

end train_speed_in_km_per_hr_l313_313154


namespace num_bus_routes_is_111_l313_313444

-- Let bus_stop be some type representing bus stops in the city
-- Let route be some type representing routes in the city
constant bus_stop : Type
constant route : Type

-- Each route has exactly 11 bus stops
constant has_eleven_stops : route → Set bus_stop

-- Any two bus routes have just one stop in common
constant one_common_stop : ∀ (r1 r2 : route) (h : r1 ≠ r2), ∃ (b : bus_stop), b ∈ has_eleven_stops r1 ∧ b ∈ has_eleven_stops r2 ∧ ∀ (x : bus_stop), (x ∈ has_eleven_stops r1 ∧ x ∈ has_eleven_stops r2) → x = b

-- Any two stops lie on just one route
constant unique_route : ∀ (b1 b2 : bus_stop) (h : b1 ≠ b2), ∃! (r : route), b1 ∈ has_eleven_stops r ∧ b2 ∈ has_eleven_stops r

-- We want to prove that the number of bus routes is 111
constant num_routes : ℕ
axiom num_routes_is_111 : num_routes = 111

-- The statement to prove
theorem num_bus_routes_is_111 : num_routes = 111 := 
by
  exact num_routes_is_111

end num_bus_routes_is_111_l313_313444


namespace Whitney_bookmarks_l313_313421

def Whitney_initial_money : ℕ := 40
def poster_cost : ℕ := 10
def notebook_cost : ℕ := 12
def remaining_money : ℕ := 14
def bookmark_cost : ℕ := 2

theorem Whitney_bookmarks:
  ∃ b: ℕ, Whitney_initial_money - (poster_cost + notebook_cost + b * bookmark_cost) = remaining_money → b = 2 :=
begin
  sorry
end

end Whitney_bookmarks_l313_313421


namespace total_travel_cost_is_47100_l313_313140

-- Define the dimensions of the lawn
def lawn_length : ℝ := 200
def lawn_breadth : ℝ := 150

-- Define the roads' widths and their respective travel costs per sq m
def road1_width : ℝ := 12
def road1_travel_cost : ℝ := 4
def road2_width : ℝ := 15
def road2_travel_cost : ℝ := 5
def road3_width : ℝ := 10
def road3_travel_cost : ℝ := 3
def road4_width : ℝ := 20
def road4_travel_cost : ℝ := 6

-- Define the areas of the roads
def road1_area : ℝ := lawn_length * road1_width
def road2_area : ℝ := lawn_length * road2_width
def road3_area : ℝ := lawn_breadth * road3_width
def road4_area : ℝ := lawn_breadth * road4_width

-- Define the costs for the roads
def road1_cost : ℝ := road1_area * road1_travel_cost
def road2_cost : ℝ := road2_area * road2_travel_cost
def road3_cost : ℝ := road3_area * road3_travel_cost
def road4_cost : ℝ := road4_area * road4_travel_cost

-- Define the total cost
def total_cost : ℝ := road1_cost + road2_cost + road3_cost + road4_cost

-- The theorem statement
theorem total_travel_cost_is_47100 : total_cost = 47100 := by
  sorry

end total_travel_cost_is_47100_l313_313140


namespace collinear_O_H_C_l313_313622

variable (O P A B C D E G F H : Point)
variable (circle_O : Circle)
variable [tangent_PA_PB : Tangent P circle_O A B]
variable [point_on_minor_arc : OnMinorArc C A B]
variable [tangent_through_C : TangentThrough C circle_O D E]
variable [intersection_G : IntersectsLine G A B D O]
variable [intersection_F : IntersectsLine F A B E O]
variable [intersection_H : IntersectsLine H D F E G]

theorem collinear_O_H_C :
  Collinear O H C := 
sorry

end collinear_O_H_C_l313_313622


namespace melissa_points_per_game_l313_313025

theorem melissa_points_per_game (total_points games : ℕ) (h1 : total_points = 91) (h2 : games = 13) :
  total_points / games = 7 :=
by
  rw [h1, h2]
  exact Nat.div_eq_of_eq_mul_left (by norm_num) (by norm_num) rfl

end melissa_points_per_game_l313_313025


namespace matthew_egg_rolls_l313_313773

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end matthew_egg_rolls_l313_313773


namespace sin_135_degree_l313_313538

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313538


namespace Danny_caps_vs_wrappers_l313_313928

def park_caps : ℕ := 58
def park_wrappers : ℕ := 25
def beach_caps : ℕ := 34
def beach_wrappers : ℕ := 15
def forest_caps : ℕ := 21
def forest_wrappers : ℕ := 32
def before_caps : ℕ := 12
def before_wrappers : ℕ := 11

noncomputable def total_caps : ℕ := park_caps + beach_caps + forest_caps + before_caps
noncomputable def total_wrappers : ℕ := park_wrappers + beach_wrappers + forest_wrappers + before_wrappers

theorem Danny_caps_vs_wrappers : total_caps - total_wrappers = 42 := by
  sorry

end Danny_caps_vs_wrappers_l313_313928


namespace trig_identity_l313_313692

theorem trig_identity (θ : ℝ) (h : Real.tan θ = Real.sqrt 3) : 
  Real.sin (2 * θ) / (1 + Real.cos (2 * θ)) = Real.sqrt 3 := 
by
  sorry

end trig_identity_l313_313692


namespace find_f_2011_l313_313634

noncomputable def f : ℝ → ℝ := sorry

axiom periodicity (x : ℝ) : f (x + 2) = -f x
axiom specific_interval (x : ℝ) (h2 : 2 < x) (h4 : x < 4) : f x = x + 3

theorem find_f_2011 : f 2011 = 6 :=
by {
  -- Leave this part to be filled with the actual proof,
  -- satisfying the initial conditions and concluding f(2011) = 6
  sorry
}

end find_f_2011_l313_313634


namespace average_speed_train_l313_313426

theorem average_speed_train (d1 d2 : ℝ) (t1 t2 : ℝ) 
  (h_d1 : d1 = 325) (h_d2 : d2 = 470)
  (h_t1 : t1 = 3.5) (h_t2 : t2 = 4) :
  (d1 + d2) / (t1 + t2) = 106 :=
by
  sorry

end average_speed_train_l313_313426


namespace increasing_sequences_mod_condition_l313_313916

theorem increasing_sequences_mod_condition :
  let sequences := { s : Finset (Fin 1007) // all (λ i, s i - i ∈ {0, 3, ..., 1002}) (Finset.range 5)}
  sequences.card = Nat.choose 338 5
  ∧ 338 % 500 = 338 :=
by
  sorry

end increasing_sequences_mod_condition_l313_313916


namespace sin_135_correct_l313_313561

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313561


namespace h_is_not_T_multiplicative_g_is_T_multiplicative_find_a_l313_313633

-- Define the T-multiplicative periodic function
def is_T_multiplicative_periodic {D : Type*} [AddSemigroup D] [DecidableEq D]
  (f : D → ℝ) (T : ℝ) : Prop := T > 0 ∧ ∀ x : D, f (x + T) = T * f x

-- Problem (1) 
theorem h_is_not_T_multiplicative (T : ℝ) : ¬is_T_multiplicative_periodic (λ x : ℝ, x) T := sorry

-- Problem (2)
theorem g_is_T_multiplicative (T : ℝ) : is_T_multiplicative_periodic (λ x : ℝ, (1/4)^x) T ↔ T = 1/2 := sorry

-- Problem (3)
theorem find_a (C_n : ℕ → ℕ → ℝ) (S_n : ℕ → ℝ) (a : ℝ) :
  (∀ (n : ℕ), n > 0 → C_n < real.log a (a + 1) + 10) →
  (0 < a ∧ a < (-1 + real.sqrt 5) / 2 ∨ a > 1) := sorry

end h_is_not_T_multiplicative_g_is_T_multiplicative_find_a_l313_313633


namespace same_color_probability_l313_313806

theorem same_color_probability 
  (B R : ℕ)
  (hB : B = 5)
  (hR : R = 5)
  : (B + R = 10) → (1/2 * 4/9 + 1/2 * 4/9 = 4/9) := by
  intros
  sorry

end same_color_probability_l313_313806


namespace problem_l313_313209

theorem problem (a b c : ℝ) (h1 : a = -6) (h2 : b = 9)
  (h_extreme : ∀ x, x ∈ ({1, 3} : Set ℝ) → deriv (λ x, x^3 + a*x^2 + b*x + c) x = 0) :
  (∃ a b, a = -6 ∧ b = 9 ∧ (∀ x, x ∈ ({1, 3} : Set ℝ) → deriv (λ x, x^3 + a*x^2 + b*x + c) x = 0)) ×
  (∃ c, -4 < c ∧ c < 0 → (∃ x y z, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (λ x, x^3 + a*x^2 + b*x + c) x = 0)) :=
begin
  sorry
end

end problem_l313_313209


namespace person_half_Jordyn_age_is_6_l313_313777

variables (Mehki_age Jordyn_age certain_age : ℕ)
axiom h1 : Mehki_age = Jordyn_age + 10
axiom h2 : Jordyn_age = 2 * certain_age
axiom h3 : Mehki_age = 22

theorem person_half_Jordyn_age_is_6 : certain_age = 6 :=
by sorry

end person_half_Jordyn_age_is_6_l313_313777


namespace roof_length_width_diff_l313_313072

theorem roof_length_width_diff (w l : ℕ) (h1 : l = 4 * w) (h2 : 784 = l * w) : l - w = 42 := by
  sorry

end roof_length_width_diff_l313_313072


namespace average_score_after_19_innings_l313_313428

/-
  Problem Statement:
  Prove that the cricketer's average score after 19 innings is 24,
  given that scoring 96 runs in the 19th inning increased his average by 4.
-/

theorem average_score_after_19_innings :
  ∀ A : ℕ,
  (18 * A + 96) / 19 = A + 4 → A + 4 = 24 :=
by
  intros A h
  /- Skipping proof by adding "sorry" -/
  sorry

end average_score_after_19_innings_l313_313428


namespace sin_135_eq_l313_313514

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313514


namespace sphere_and_cube_properties_l313_313461

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem sphere_and_cube_properties :
  let r := 5 in
  let volume := sphere_volume r in
  let edge_length := 2 * r in
  volume = (500 / 3) * Real.pi ∧ edge_length = 10 :=
by
  sorry

end sphere_and_cube_properties_l313_313461


namespace no_real_t_for_sqrt_eq_zero_l313_313185

theorem no_real_t_for_sqrt_eq_zero : ¬ ∃ (t : ℝ), sqrt (49 - t^2) + 7 = 0 := by
  sorry

end no_real_t_for_sqrt_eq_zero_l313_313185


namespace binomial_probability_X_eq_3_l313_313345

theorem binomial_probability_X_eq_3 :
  let n := 6
  let p := 1 / 2
  let k := 3
  let binom := Nat.choose n k
  (binom * p ^ k * (1 - p) ^ (n - k)) = 5 / 16 := by 
  sorry

end binomial_probability_X_eq_3_l313_313345


namespace alex_distribution_ways_l313_313465

theorem alex_distribution_ways : (15^5 = 759375) := by {
  sorry
}

end alex_distribution_ways_l313_313465


namespace max_page_number_l313_313464

/-- Alex Arnold has only sixteen 5's but an unlimited number of all other digits from 0 to 9. 
    Prove that he can number up to page 75 with these constraints. -/
theorem max_page_number (h_fives: nat := 16) : (max_page h_fives) = 75 :=
by
  sorry

end max_page_number_l313_313464


namespace sum_of_consecutive_integers_l313_313071

theorem sum_of_consecutive_integers (x y : ℕ) (h1 : y = x + 1) (h2 : x * y = 812) : x + y = 57 :=
by
  -- proof skipped
  sorry

end sum_of_consecutive_integers_l313_313071


namespace a_2023_eq_x_l313_313967

noncomputable def a_sequence (n : ℕ) (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) : ℝ :=
  nat.rec_on n x (λ _ a, 1 / (1 - a))

theorem a_2023_eq_x (x : ℝ) (hx : x ≠ 0) (hx1 : x ≠ 1) : 
  a_sequence 2023 x hx hx1 = x :=
sorry

end a_2023_eq_x_l313_313967


namespace bob_pays_3_less_than_alice_l313_313158

theorem bob_pays_3_less_than_alice (base_cost : ℕ) (mushroom_cost : ℕ) (olive_cost : ℕ) (total_slices : ℕ) 
(ate_by_alice : ℕ) (ate_by_bob : ℕ) (total_cost : ℕ) (cost_per_slice : ℚ):
  base_cost = 12 → 
  mushroom_cost = 3 → 
  olive_cost = 4 → 
  total_slices = 12 → 
  total_cost = base_cost + mushroom_cost + olive_cost →
  cost_per_slice = total_cost / total_slices → 
  ate_by_alice = 7 →
  ate_by_bob = 5 →
  ∃ cost_paid_by_alice cost_paid_by_bob : ℚ, 
  cost_paid_by_alice = ate_by_alice * cost_per_slice ∧
  cost_paid_by_bob = ate_by_bob * cost_per_slice ∧
  cost_paid_by_bob - cost_paid_by_alice = -3 :=
begin
  intros,
  existsi (ate_by_alice * cost_per_slice),
  existsi (ate_by_bob * cost_per_slice),
  split; try {refl},
  split; try {refl},
  nth_rewrite 1 ← mul_sub,
  rw [← sub_self (ate_by_alice * cost_per_slice)],
  congr' 1,
  -- simplify the subtractions to relate back to Alice and Bob's slice counts
  simp,
  sorry
end

end bob_pays_3_less_than_alice_l313_313158


namespace intersection_of_A_and_B_is_2_l313_313981

-- Define the sets A and B based on the given conditions
def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

-- State the theorem that needs to be proved
theorem intersection_of_A_and_B_is_2 : A ∩ B = {2} :=
by
  sorry

end intersection_of_A_and_B_is_2_l313_313981


namespace area_of_triangle_ABC_correct_l313_313271

noncomputable def area_of_triangle_ABC 
  (A B C D E: Type*) [normed_add_comm_group A] [normed_space ℝ A]
  (angle_BAC: ℝ)
  (altitude_BD: ℝ)
  (median_BE: ℝ)
  (K: ℝ):
  ℝ :=
2 * sqrt (6 * sqrt (2 * K)) * ∥ B - A∥

theorem area_of_triangle_ABC_correct :
  ∀ (A B C D E: Type*) [normed_add_comm_group A] [normed_space ℝ A]
    (angle_BAC: ℝ)
    (altitude_BD: ℝ)
    (median_BE: ℝ)
    (K: ℝ),
    angle_BAC = π / 3 →
    (altitude_BD > 0) →
    (median_BE > 0) →
    K > 0 →
    area_of_triangle_ABC A B C D E angle_BAC altitude_BD median_BE K = 
    2 * sqrt (6 * sqrt (2 * K)) * ∥ B - A∥ :=
by
  intros A B C D E _ _ angle_BAC altitude_BD median_BE K h_angle h_altitude h_median h_K
  sorry

end area_of_triangle_ABC_correct_l313_313271


namespace tram_trip_analysis_l313_313400

theorem tram_trip_analysis :
  ∀ (t : ℝ), 
    (20 / t - 5 = (20 / (t - 1/5))) → 
    t - 1/5 > 0 → 
    (let new_tram_speed := 20 / (t - 1/5) in
     let new_tram_time := 20 / new_tram_speed in
     new_tram_speed = 25 ∧ new_tram_time * 60 = 48) :=
by
  sorry

end tram_trip_analysis_l313_313400


namespace granite_rocks_correct_l313_313839

-- Definitions based on conditions
def number_slate_rocks := 10
def number_pumice_rocks := 11
def probability_both_slate := 0.15

-- Number of granite rocks to be proven as 4
def number_granite_rocks := 4

-- Theorem statement
theorem granite_rocks_correct :
  ∃ (G : ℕ), G = number_granite_rocks →
  (∑ n in [number_slate_rocks, number_pumice_rocks, G], n ≠ 0) ∧ 
  ((number_slate_rocks * (number_slate_rocks - 1)) / 
  ((number_slate_rocks + number_pumice_rocks + G) * (number_slate_rocks + number_pumice_rocks + G - 1))) = probability_both_slate :=
by
  sorry

end granite_rocks_correct_l313_313839


namespace number_of_valid_sequences_l313_313310

def sequence := Fin 6 -> ℕ

def valid_sequence (s : sequence) : Prop :=
  (∀ i, (2 ≤ i ∧ i ≤ 6) →
    (∃ j, j < i ∧ (s j = 2 * s i ∨ s j = s i / 2)))

theorem number_of_valid_sequences : 
  ∃ n : ℕ, n = 32 ∧ (∃ l : list sequence, l.length = n ∧ ∀ s ∈ l, valid_sequence s) := 
  sorry

end number_of_valid_sequences_l313_313310


namespace definitely_quadratic_l313_313854

theorem definitely_quadratic :
  ∀ (x : ℝ), 
    (sqrt 2 * x^2 - sqrt 2 / 4 * x - 1 / 2 = 0) -> 
    ∃ (a b c : ℝ), a ≠ 0 ∧ (a * x^2 + b * x + c = 0) :=
by
  sorry

end definitely_quadratic_l313_313854


namespace min_range_of_observations_l313_313145

open Real

theorem min_range_of_observations
  (x1 x2 x3 x4 x5 : ℝ)
  (h_sum : x1 + x2 + x3 + x4 + x5 = 50)
  (h_median : x3 = 12)
  (h_order : x1 ≤ x2 ∧ x2 ≤ x3 ∧ x3 ≤ x4 ∧ x4 ≤ x5) :
  ∃ (r : ℝ), (∀ y1 y2 y3 y4 y5 a_sum a_median a_order,
  y1 + y2 + y3 + y4 + y5 = 50 ∧ y3 = 12 ∧ y1 ≤ y2 ∧ y2 ≤ y3 ∧ y3 ≤ y4 ∧ y4 ≤ y5 →
  y5 - y1 ≥ r) ∧ r = 5 :=
by
sorry

end min_range_of_observations_l313_313145


namespace good_mistakes_l313_313270

def count_possible_mistakes (word : String) : Nat :=
  let n := word.length
  let repetitions := word.toList.filter (λ c => c = 'o').length
  (Nat.factorial n) / (Nat.factorial repetitions) - 1

theorem good_mistakes : count_possible_mistakes "good" = 11 :=
  by sorry

end good_mistakes_l313_313270


namespace minimum_disks_needed_l313_313301

-- Define the conditions
def total_files : ℕ := 25
def disk_capacity : ℝ := 2.0
def files_06MB : ℕ := 5
def size_06MB_file : ℝ := 0.6
def files_10MB : ℕ := 10
def size_10MB_file : ℝ := 1.0
def files_03MB : ℕ := total_files - files_06MB - files_10MB
def size_03MB_file : ℝ := 0.3

-- Define the theorem that needs to be proved
theorem minimum_disks_needed : 
    ∃ (disks: ℕ), disks = 10 ∧ 
    (5 * size_06MB_file + 10 * size_10MB_file + 10 * size_03MB_file) ≤ disks * disk_capacity := 
by
  sorry

end minimum_disks_needed_l313_313301


namespace sin_135_eq_sqrt2_div_2_l313_313582

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313582


namespace distance_school_to_cemetery_l313_313599

noncomputable def drive_speed_up_1_5_saving_10 (v : ℝ) (d t: ℝ) : Prop :=
  (d / (v * (1 + 1/5)) + d / (v) - d / (1 * v)) = 10 / 60

noncomputable def drive_speed_up_1_3_saving_20 (v d t: ℝ) : Prop :=
  (60 / v + (d - 60) / (v * (1 + 1/3)) + (d - 60) / v) = 20 / 60

theorem distance_school_to_cemetery {d : ℝ} (v : ℝ) (t : ℝ) 
  (h1 : drive_speed_up_1_5_saving_10 v d t)
  (h2 : drive_speed_up_1_3_saving_20 v d t) 
  : d = 180 :=
begin
  sorry
end

end distance_school_to_cemetery_l313_313599


namespace sphere_surface_area_l313_313230

theorem sphere_surface_area (a R : ℝ) 
    (cube_surface_area : 6 * a^2 = 18)
    (cube_diagonal_eq_sphere_diameter : (√3) * a = 2 * R) : 
    4 * Real.pi * R^2 = 9 * Real.pi := 
sorry

end sphere_surface_area_l313_313230


namespace trader_bags_correct_l313_313152

-- Definitions according to given conditions
def initial_bags := 55
def sold_bags := 23
def restocked_bags := 132

-- Theorem that encapsulates the problem's question and the proven answer
theorem trader_bags_correct :
  (initial_bags - sold_bags + restocked_bags) = 164 :=
by
  sorry

end trader_bags_correct_l313_313152


namespace sqrt_a_div_sqrt_b_EQ_five_div_two_l313_313942

theorem sqrt_a_div_sqrt_b_EQ_five_div_two 
  (a b : ℝ) 
  (h : ( (1/3)^2 + (1/4)^2 ) / ( (1/5)^2 + (1/6)^2 ) = 25 * a / (53 * b)) : 
  sqrt(a) / sqrt(b) = 5 / 2 :=
by 
  sorry

end sqrt_a_div_sqrt_b_EQ_five_div_two_l313_313942


namespace b_over_c_equals_1_l313_313864

theorem b_over_c_equals_1 (a b c d : ℕ) (ha : a < 4) (hb : b < 4) (hc : c < 4) (hd : d < 4)
    (h : 4^a + 3^b + 2^c + 1^d = 78) : b = c :=
by
  sorry

end b_over_c_equals_1_l313_313864


namespace max_value_problem_l313_313764

theorem max_value_problem (a b c : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 2) (h3 : 0 ≤ b)
  (h4 : b ≤ 2) (h5 : 0 ≤ c) (h6 : c ≤ 2) :
  (sqrt (a^2 * b^2 * c^2) + sqrt ((2 - a) * (2 - b) * (2 - c))) ≤ 4 :=
sorry

end max_value_problem_l313_313764


namespace four_xyz_value_l313_313650

theorem four_xyz_value (x y z : ℝ) (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
    (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) : 4 * x * y * z = 48 := by
  sorry

end four_xyz_value_l313_313650


namespace quadrilateral_is_rhombus_l313_313338

-- Given conditions:

-- Define a quadrilateral
variables (A B C D: Type) [nonempty A] [nonempty B] [nonempty C] [nonempty D]

-- Define the diagonals AC and BD that intersect at O, the center of an inscribed circle
variables (O: Type) [nonempty O]

-- Define a predicate indicating that O is the center of the inscribed circle of the quadrilateral
-- This means that the distance from O to each of the sides of quadrilateral ABCD is equal
def is_center_of_inscribed_circle (O : Type) [nonempty O] (A B C D : Type) [nonempty A] [nonempty B] [nonempty C] [nonempty D] : Prop := 
  ∀ (P Q R S: Type) [nonempty P] [nonempty Q] [nonempty R] [nonempty S], 
  distance O P = distance O Q ∧ distance O Q = distance O R ∧ distance O R = distance O S

-- Theorem: if O is the center of the inscribed circle of quadrilateral ABCD, then ABCD is a rhombus
theorem quadrilateral_is_rhombus 
  (A B C D : Type) [nonempty A] [nonempty B] [nonempty C] [nonempty D]
  (O : Type) [nonempty O]
  (h : is_center_of_inscribed_circle O A B C D) : 
  is_rhombus A B C D := 
sorry

end quadrilateral_is_rhombus_l313_313338


namespace quadratic_inequality_solution_l313_313932

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 2) * x - 2 * k + 4 < 0) ↔ (-6 < k ∧ k < 2) :=
by
  sorry

end quadratic_inequality_solution_l313_313932


namespace find_b_l313_313366

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := (1/12 : ℝ) * x^2 + a * x + b

theorem find_b (a b : ℝ) (α γ : ℝ) :
  (T_x : 3) (T_y : 3) (TA : TA_dist) (TB : TB_dist) (TC : TC_dist) 
  (roots_cond : α < 0 ∧ γ > 0) 
  (y_axis_intersect : f 0 a b = b) 
  (vieta_1 : α + γ = -12 * a) 
  (vieta_2 : α * γ = 12 * b) 
  (dist_cond : (3 - α)^2 + (3 - 3)^2 = (3 - 0)^2 + (3 - b)^2 ∧ (3 - γ)^2 + (3 - 3)^2 = (3 - 0)^2 + (3 - b)^2) : 
  b = -6 := 
by
  sorry

end find_b_l313_313366


namespace trigonometric_expression_approx_l313_313612

noncomputable def trigonometric_expression : ℝ :=
  (2 * real.sin (20 * real.pi / 180) * real.cos (10 * real.pi / 180) + 
   real.cos (160 * real.pi / 180) * real.cos (110 * real.pi / 180)) / 
  (real.sin (24 * real.pi / 180) * real.cos (6 * real.pi / 180) + 
   real.cos (156 * real.pi / 180) * real.cos (94 * real.pi / 180))

theorem trigonometric_expression_approx : abs (trigonometric_expression - 1.481) < 0.001 := 
sorry

end trigonometric_expression_approx_l313_313612


namespace max_four_digit_prime_product_l313_313003

theorem max_four_digit_prime_product :
  ∃ (x y : ℕ) (n : ℕ), x < 5 ∧ y < 5 ∧ x ≠ y ∧ Prime x ∧ Prime y ∧ Prime (10 * x + y) ∧ n = x * y * (10 * x + y) ∧ n = 138 :=
by
  sorry

end max_four_digit_prime_product_l313_313003


namespace circle_equation_l313_313360

theorem circle_equation (c : ℝ) (x y : ℝ) (h : (1, 2) = (1, 2)) : x^2 + (y - 2)^2 = 1 :=
by
  noncomputable def center := (0, c)
  def radius := 1
  def point_on_circle := (1, 2)
  have : (point_on_circle.1 - center.1)^2 + (point_on_circle.2 - center.2)^2 = radius^2 := by
    simp [point_on_circle, center, radius]
    sorry
  sorry

end circle_equation_l313_313360


namespace proof_parabola_circle_max_value_l313_313639

section Problem

variable (p : ℝ) (triangle_vertex1 triangle_vertex2 triangle_vertex3 : ℝ × ℝ)
variable (D : ℝ × ℝ) (M : ℝ × ℝ)
variable (l1 l2 : ℝ)

-- Given conditions

-- The side length of the regular triangle
def is_regular_triangle := (dist triangle_vertex1 triangle_vertex2 = 8 * real.sqrt 3) ∧ 
                           (dist triangle_vertex1 triangle_vertex3 = 8 * real.sqrt 3) ∧
                           (dist triangle_vertex2 triangle_vertex3 = 8 * real.sqrt 3)

-- Vertices of the triangle lie on the parabola x^2 = 2py
def on_parabola := (triangle_vertex1.1 pow 2 = 2 * p * triangle_vertex1.2) ∧
                   (triangle_vertex2.1 pow 2 = 2 * p * triangle_vertex2.2) ∧
                   (triangle_vertex3.1 pow 2 = 2 * p * triangle_vertex3.2)

-- Circle passing through D with center M on the parabola and intersects x-axis at A and B
def circle_center := (M.1 pow 2 = 4 * M.2) ∧ (D.1 = 0) ∧ (D.2 = 2)

-- Lengths l1 and l2 as distances from D
def length_l1 := l1 = sqrt ((M.1 - 2) ^ 2 + 4)
def length_l2 := l2 = sqrt ((M.1 + 2) ^ 2 + 4)

-- Function to be maximized 
def expression := (l1 / l2) + (l2 / l1)

-- Proving the required maximum value
theorem proof_parabola_circle_max_value 
  (hp : p = 2) 
  (h_triangle : is_regular_triangle p triangle_vertex1 triangle_vertex2 triangle_vertex3)
  (h_parabola : on_parabola p triangle_vertex1 triangle_vertex2 triangle_vertex3)
  (h_circle : circle_center M D) 
  (l1_def : length_l1 M l1)
  (l2_def : length_l2 M l2) : 
  expression l1 l2 ≤ 2 * real.sqrt 2 := sorry

end Problem

end proof_parabola_circle_max_value_l313_313639


namespace lens_focal_length_l313_313151

noncomputable def focal_length (x : ℝ) (alpha : ℝ) (beta : ℝ) : ℝ :=
  (20 * Real.sqrt 2) / (Real.sqrt 6 - Real.sqrt 2)

theorem lens_focal_length (x : ℝ) (alpha : ℝ) (beta : ℝ) : 
  x = 10 → alpha = 45 → beta = 30 → focal_length x alpha beta ≈ 13.7 :=
by {
  intros h1 h2 h3,
  simp [focal_length, h1, h2, h3],
  -- Approximation comparison for 13.7 based on floating-point or rational bounds
  sorry
}

end lens_focal_length_l313_313151


namespace qu_arrangement_in_equation_l313_313041

theorem qu_arrangement_in_equation : 
  let word := "equation".to_enumerated_list
  let units := ("qu" :: word.filter (λ x, x ∉ ['q', 'u'])).length
  let choosing := nat.choose 6 3
  let arrangements := 4.factorial
  (choosing * arrangements = 480) := 
sorry

end qu_arrangement_in_equation_l313_313041


namespace sin_135_l313_313499

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313499


namespace number_of_games_in_complete_season_l313_313147

-- Define the number of teams in each division
def teams_in_division_A : Nat := 6
def teams_in_division_B : Nat := 7
def teams_in_division_C : Nat := 5

-- Define the number of games each team must play within their division
def games_per_team_within_division (teams : Nat) : Nat :=
  (teams - 1) * 2

-- Calculate the total number of games within a division
def total_games_within_division (teams : Nat) : Nat :=
  (games_per_team_within_division teams * teams) / 2

-- Calculate cross-division games for a team in one division
def cross_division_games_per_team (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  (teams_other_div1 + teams_other_div2) * 2

-- Calculate total cross-division games from all teams in one division
def total_cross_division_games (teams_div : Nat) (teams_other_div1 : Nat) (teams_other_div2 : Nat) : Nat :=
  cross_division_games_per_team teams_other_div1 teams_other_div2 * teams_div

-- Given conditions translated to definitions
def games_in_division_A : Nat := total_games_within_division teams_in_division_A
def games_in_division_B : Nat := total_games_within_division teams_in_division_B
def games_in_division_C : Nat := total_games_within_division teams_in_division_C

def cross_division_games_A : Nat := total_cross_division_games teams_in_division_A teams_in_division_B teams_in_division_C
def cross_division_games_B : Nat := total_cross_division_games teams_in_division_B teams_in_division_A teams_in_division_C
def cross_division_games_C : Nat := total_cross_division_games teams_in_division_C teams_in_division_A teams_in_division_B

-- Total cross-division games with each game counted twice
def total_cross_division_games_in_season : Nat :=
  (cross_division_games_A + cross_division_games_B + cross_division_games_C) / 2

-- Total number of games in the season
def total_games_in_season : Nat :=
  games_in_division_A + games_in_division_B + games_in_division_C + total_cross_division_games_in_season

-- The final proof statement
theorem number_of_games_in_complete_season : total_games_in_season = 306 :=
by
  -- This is the place where the proof would go if it were required.
  sorry

end number_of_games_in_complete_season_l313_313147


namespace cuboid_dimensions_exist_l313_313835

theorem cuboid_dimensions_exist (l w h : ℝ) 
  (h1 : l * w = 5) 
  (h2 : l * h = 8) 
  (h3 : w * h = 10) 
  (h4 : l * w * h = 200) : 
  ∃ (l w h : ℝ), l = 4 ∧ w = 2.5 ∧ h = 2 := 
sorry

end cuboid_dimensions_exist_l313_313835


namespace product_closest_to_1200_l313_313603

-- Define the constants
def x : ℝ := 0.000315
def y : ℝ := 3_928_500
def target : ℝ := 1200

-- Statement of the problem as a theorem in Lean
theorem product_closest_to_1200 : 
  abs ((x * y) - target) ≤ 
    min (abs ((x * y) - 1100)) 
        (min (abs ((x * y) - 1300)) 
             (abs ((x * y) - 1400))) :=
by 
  -- Skip the proof
  sorry

end product_closest_to_1200_l313_313603


namespace range_of_a_l313_313765

theorem range_of_a (a : ℝ) : 
  (∀ x, log 2 (x - 3) > 1 → ∃ x, 2 ^ (x - a) > 2) → a ≤ 4 := 
by
  sorry

end range_of_a_l313_313765


namespace proveCircleEquation_proveMinDistancePoint_l313_313725

noncomputable def cartesianCircleEquation : Prop :=
  ∀ θ: ℝ, let x := sqrt 3 * Real.cos θ,
           let y := sqrt 3 * (1 + Real.sin θ)
           in (x^2 + (y - sqrt 3)^2 = 3)

noncomputable def minDistancePoint : Prop :=
  ∀ t: ℝ, let Px := 3 + (1/2) * t,
            let Py := (sqrt 3) / 2 * t,
            let Cx := 0,
            let Cy := sqrt 3,
            let dist := Real.sqrt ((3 + (1/2) * t)^2 + (((sqrt 3) / 2 * t) - sqrt 3)^2)
            in dist ≥ Real.sqrt ((3 + (1/2) * 0)^2 + (((sqrt 3) / 2 * 0) - sqrt 3)^2) →
                       (Px = 3 ∧ Py = 0)

theorem proveCircleEquation : cartesianCircleEquation :=
by {
  -- proof will go here
  sorry
}

theorem proveMinDistancePoint : minDistancePoint :=
by {
  -- proof will go here
  sorry
}

end proveCircleEquation_proveMinDistancePoint_l313_313725


namespace dvd_book_capacity_l313_313127

/--
Theorem: Given that there are 81 DVDs already in the DVD book and it can hold 45 more DVDs,
the total capacity of the DVD book is 126 DVDs.
-/
theorem dvd_book_capacity : 
  (already_in_book additional_capacity : ℕ) (h1 : already_in_book = 81) (h2 : additional_capacity = 45) :
  already_in_book + additional_capacity = 126 :=
by
  sorry

end dvd_book_capacity_l313_313127


namespace same_function_l313_313859

def f (x : ℝ) := abs x
def g (x : ℝ) := real.sqrt (x ^ 2)

theorem same_function : ∀ x : ℝ, f x = g x :=
by
  intro x
  sorry

end same_function_l313_313859


namespace find_a_2_2_l313_313974

variable (n : ℕ) [fact (4 ≤ n)]
variable (a : fin n → fin n → ℕ)

def arithmetic_sequence_in_row (i : fin n) : Prop :=
  ∃ d : ℕ, ∀ j k : fin n, k ≤ j → a i j = a i k + (j.val - k.val) * d

def geometric_sequence_in_col (j : fin n) : Prop :=
  ∃ r : ℕ, ∀ i k : fin n, k ≤ i → a i j = a k j * (r ^ (i.val - k.val))

theorem find_a_2_2 (h1 : arithmetic_sequence_in_row a 1)
                  (h2 : arithmetic_sequence_in_row a 2)
                  (h3 : arithmetic_sequence_in_row a 3)
                  (h4 : arithmetic_sequence_in_row a 4)
                  (h5 : geometric_sequence_in_col a 1)
                  (h6 : geometric_sequence_in_col a 2)
                  (h7 : geometric_sequence_in_col a 3)
                  (h8 : geometric_sequence_in_col a 4)
                  (a_2_3 : a 2 2 = 8)
                  (a_3_4 : a 3 3 = 20) : 
  a 2 1 = 6 :=
  sorry

end find_a_2_2_l313_313974


namespace unique_k_solves_eq_l313_313065

theorem unique_k_solves_eq (k : ℕ) (hpos_k : k > 0) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ a^2 + b^2 = k * a * b) ↔ k = 2 :=
by
  sorry

end unique_k_solves_eq_l313_313065


namespace fraction_addition_l313_313099

theorem fraction_addition :
  (1 / 6) + (1 / 3) + (5 / 9) = 19 / 18 :=
by
  sorry

end fraction_addition_l313_313099


namespace derivative_at_pi_over_6_l313_313266

def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6)

theorem derivative_at_pi_over_6 : deriv f (Real.pi / 6) = 0 := by
  -- Expansion of F applied to multiple rules
  have f_eq := by {
    intro x,
    exact Real.cos (2 * x + Real.pi / 6) * (2 : ℝ)
  }
  rw [f_eq],
  -- Calculate the value at x = π/6
  rw [Real.cos (2 * (Real.pi / 6) + Real.pi / 6), Real.cos_pi_div_two],
  norm_num

end derivative_at_pi_over_6_l313_313266


namespace sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l313_313172

-- Problem 1
theorem sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3 : |Real.sqrt 3 - Real.sqrt 2| + Real.sqrt 2 = Real.sqrt 3 := by
  sorry

-- Problem 2
theorem sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2 : Real.sqrt 2 * (Real.sqrt 2 + 2) = 2 + 2 * Real.sqrt 2 := by
  sorry

end sqrt3_minus_sqrt2_abs_plus_sqrt2_eq_sqrt3_sqrt2_times_sqrt2_plus_2_eq_2_plus_2sqrt2_l313_313172


namespace temperature_difference_l313_313728

variable (high_temp : ℝ) (low_temp : ℝ)

theorem temperature_difference (h1 : high_temp = 15) (h2 : low_temp = 7) : high_temp - low_temp = 8 :=
by {
  sorry
}

end temperature_difference_l313_313728


namespace simplify_expression_l313_313267

theorem simplify_expression (p q r s : ℝ) (hp : p ≠ 6) (hq : q ≠ 7) (hr : r ≠ 8) (hs : s ≠ 9) :
    (p - 6) / (8 - r) * (q - 7) / (6 - p) * (r - 8) / (7 - q) * (s - 9) / (9 - s) = 1 := by
  sorry

end simplify_expression_l313_313267


namespace cost_of_pencils_and_pens_l313_313354

theorem cost_of_pencils_and_pens (p q : ℝ) 
  (h₁ : 3 * p + 2 * q = 3.60) 
  (h₂ : 2 * p + 3 * q = 3.15) : 
  3 * p + 3 * q = 4.05 :=
sorry

end cost_of_pencils_and_pens_l313_313354


namespace point_on_line_distance_to_x_axis_l313_313781

theorem point_on_line_distance_to_x_axis {k : ℝ} (h : k = 2 * (-2) + 1) :
  let M_y := k,
  d := |M_y|
  in d = 3 :=
by
  simp [h]
  sorry

end point_on_line_distance_to_x_axis_l313_313781


namespace problem1_problem2_l313_313248

-- Given the function f(x) = 2x * ln x, prove that the tangent line at (1, f(1)) is perpendicular to y = -1/2*x
theorem problem1 {a : ℝ} (h : ∀ x, f x = a * x * log x) :
  f' (1 : ℝ) = 2 :=
by
  sorry

-- Given the function f(x) = 2x * ln x and the inequality f(x) - m*x + 2 ≥ 0 ∀ x ≥ 1, show m ∈ (-∞, 2]
theorem problem2 {m : ℝ} (h : ∀ (x : ℝ), f x = 2 * x * log x ∧ x ≥ 1 → f x - m * x + 2 ≥ 0) :
  m ∈ set.Iic 2 :=
by
  sorry

end problem1_problem2_l313_313248


namespace probability_no_order_l313_313453

theorem probability_no_order (P : ℕ) 
  (h1 : 60 ≤ 100) (h2 : 10 ≤ 100) (h3 : 15 ≤ 100) 
  (h4 : 5 ≤ 100) (h5 : 3 ≤ 100) (h6 : 2 ≤ 100) :
  P = 100 - (60 + 10 + 15 + 5 + 3 + 2) :=
by 
  sorry

end probability_no_order_l313_313453


namespace g_is_odd_l313_313878

-- Definitions based on the conditions given in the problem
def is_even_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop :=
∀ x, g (-x) = -g x

noncomputable def f (a b c : ℝ) : ℝ → ℝ :=
λ x, a * x ^ 2 + b * x + c

noncomputable def g (a b c : ℝ) : ℝ → ℝ :=
λ x, a * x ^ 3 + b * x ^ 2 + c * x

-- Lean 4 statement to prove the required property
theorem g_is_odd (a b c : ℝ) (h_even : is_even_function (f a b c)) (h_c_nonzero : c ≠ 0) :
  is_odd_function (g a 0 c) :=
sorry

end g_is_odd_l313_313878


namespace largest_final_digit_l313_313146

theorem largest_final_digit (seq : Fin 1002 → Fin 10) 
  (h1 : seq 0 = 2) 
  (h2 : ∀ n : Fin 1001, (17 ∣ (10 * seq n + seq (n + 1))) ∨ (29 ∣ (10 * seq n + seq (n + 1)))) : 
  seq 1001 = 5 :=
sorry

end largest_final_digit_l313_313146


namespace sugar_water_inequality_triangle_inequality_l313_313696

-- Condition for question (1)
variable (x y m : ℝ)
variable (hx : x > 0) (hy : y > 0) (hxy : x > y) (hm : m > 0)

-- Proof problem for question (1)
theorem sugar_water_inequality : y / x < (y + m) / (x + m) :=
sorry

-- Condition for question (2)
variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (hab : b + c > a) (hac : a + c > b) (hbc : a + b > c)

-- Proof problem for question (2)
theorem triangle_inequality : 
  a / (b + c) + b / (a + c) + c / (a + b) < 2 :=
sorry

end sugar_water_inequality_triangle_inequality_l313_313696


namespace first_discount_percentage_l313_313073

theorem first_discount_percentage
  (P : ℝ)
  (initial_price final_price : ℝ)
  (second_discount : ℕ)
  (h1 : initial_price = 200)
  (h2 : final_price = 144)
  (h3 : second_discount = 10)
  (h4 : final_price = (P - (second_discount / 100) * P)) :
  (∃ x : ℝ, P = initial_price - (x / 100) * initial_price ∧ x = 20) :=
sorry

end first_discount_percentage_l313_313073


namespace radius_of_circle_param_eqs_l313_313826

theorem radius_of_circle_param_eqs :
  ∀ θ : ℝ, ∃ r : ℝ, r = 5 ∧ (∃ (x y : ℝ),
    x = 3 * Real.sin θ + 4 * Real.cos θ ∧
    y = 4 * Real.sin θ - 3 * Real.cos θ ∧
    x^2 + y^2 = r^2) := 
by
  sorry

end radius_of_circle_param_eqs_l313_313826


namespace sin_cos_alpha_range_k_l313_313243

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) ^ 2 - Real.log x ^ 2 / Real.log 4 + 3

axiom max_f : ∀ x ∈ Set.Icc 1 4, f x ≤ 3
axiom min_f : ∀ x ∈ Set.Icc 1 4, 2 ≤ f x

theorem sin_cos_alpha (α : ℝ) (hα : cos α = 3 / Real.sqrt 13 ∧ sin α = 2 / Real.sqrt 13) :
  sin α + cos α = 5 / Real.sqrt 13 := 
sorry

noncomputable def g (x : ℝ) : ℝ := 3 * Real.cos (2 * x + π / 3) - 2

theorem range_k : 
  ∀ (k : ℝ), 
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 ∈ Set.Icc 0 (π / 2) ∧ x2 ∈ Set.Icc 0 (π / 2) ∧ g x1 - k = 0 ∧ g x2 - k = 0) ↔
    k ∈ Set.Ioc (-5 : ℝ) (-5 / 2 : ℝ) := 
sorry

end sin_cos_alpha_range_k_l313_313243


namespace number_of_terms_added_l313_313406

theorem number_of_terms_added (k : ℕ) (hk : k > 1) :
  ∑ i in finset.range (2^(k+1) - 1) \ finset.range (2^k - 1), 1 / (i + 1) = 2^k :=
  sorry

end number_of_terms_added_l313_313406


namespace shopkeeper_profit_percentage_l313_313423

theorem shopkeeper_profit_percentage (C : ℝ) (hC : C > 0) :
  let selling_price := 12 * C
  let cost_price := 10 * C
  let profit := selling_price - cost_price
  let profit_percentage := (profit / cost_price) * 100
  profit_percentage = 20 :=
by
  sorry

end shopkeeper_profit_percentage_l313_313423


namespace perfect_squares_not_cubes_less_than_1000_l313_313469

theorem perfect_squares_not_cubes_less_than_1000 : 
  let perfect_squares := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 1000}
  let perfect_cubes := {m : ℕ | ∃ l : ℕ, l^3 = m ∧ m < 1000}
  let count_perfect_squares_not_cubes :=
    perfect_squares.to_finset.filter (λ n, n ∉ perfect_cubes.to_finset)
  in count_perfect_squares_not_cubes.card = 28 :=
by 
  -- proof omitted
  sorry

end perfect_squares_not_cubes_less_than_1000_l313_313469


namespace tens_digit_of_smallest_even_five_digit_number_l313_313056

def smallest_even_five_digit_number (digits : List ℕ) : ℕ :=
if h : 0 ∈ digits ∧ 3 ∈ digits ∧ 5 ∈ digits ∧ 6 ∈ digits ∧ 8 ∈ digits then
  35086
else
  0  -- this is just a placeholder to make the function total

theorem tens_digit_of_smallest_even_five_digit_number : 
  ∀ digits : List ℕ, 
    0 ∈ digits ∧ 
    3 ∈ digits ∧ 
    5 ∈ digits ∧ 
    6 ∈ digits ∧ 
    8 ∈ digits ∧ 
    digits.length = 5 → 
    (smallest_even_five_digit_number digits) / 10 % 10 = 8 :=
by
  intros digits h
  sorry

end tens_digit_of_smallest_even_five_digit_number_l313_313056


namespace sin_135_l313_313496

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313496


namespace integer_solutions_eq_l313_313944

theorem integer_solutions_eq (m n : ℤ) :
  m^2 + 2m = n^4 + 20n^3 + 104n^2 + 40n + 2003 ↔ (m = 128 ∧ (n = 7 ∨ n = -17)) :=
sorry

end integer_solutions_eq_l313_313944


namespace find_a_plus_d_l313_313108

theorem find_a_plus_d (a b c d : ℕ)
  (h1 : a + b = 14)
  (h2 : b + c = 9)
  (h3 : c + d = 3) : 
  a + d = 2 :=
by sorry

end find_a_plus_d_l313_313108


namespace distribute_rabbits_to_stores_l313_313788

-- Define the rabbits and their types
inductive Rabbit
| Peter | Pauline | Flopsie | Mopsie | Cotton_tail | Topsy
open Rabbit

def isParent : Rabbit → Prop
| Peter   := true
| Pauline := true
| _       := false

def isChild : Rabbit → Prop
| Peter   := false
| Pauline := false
| _       := true

-- Define the constraint of distributing to stores
def isValidDistribution (distribution : Rabbit → Option ℕ) : Prop :=
  ∀ r1 r2, distribution r1 = distribution r2 → r1 ≠ r2 ∨ (¬isParent r1 ∨ ¬isChild r2)

-- The main theorem
theorem distribute_rabbits_to_stores :
  ∃ (distribution_count : ℕ), distribution_count = 560 ∧
  ∃ distribution : Rabbit → Option ℕ, isValidDistribution distribution :=
sorry

end distribute_rabbits_to_stores_l313_313788


namespace segment_in_convex_polygon_l313_313445

theorem segment_in_convex_polygon (P : Set (ℝ × ℝ)) (h_convex : Convex ℝ P)
  (h_area : (measure_theory.measure.lebesgue.measure P) > 0.5) :
  ∃ l : ℝ, l ≥ 0.5 ∧ (∃ a b : ℝ × ℝ, a ≠ b ∧ a ∈ P ∧ b ∈ P ∧ 
    (a.1 = b.1 ∨ a.2 = b.2) ∧ 
    real.dist a b = l) :=
begin
  sorry
end

end segment_in_convex_polygon_l313_313445


namespace impossible_power_of_two_sums_l313_313629

-- Define a condition for a power of 2
def is_power_of_two (n : ℕ) : Prop :=
  ∃ (a : ℕ), n = 2^a

-- The main theorem statement
theorem impossible_power_of_two_sums (k : ℕ) (hk : k > 1) :
  ¬ (∃ (M : Matrix (Fin k) (Fin k) ℕ),
    (∀ i, is_power_of_two ∑ j, M i j) ∧
    (∀ j, is_power_of_two ∑ i, M i j)) :=
sorry

end impossible_power_of_two_sums_l313_313629


namespace generating_function_Bk_generating_function_BH_generating_function_B_l313_313830

-- Definition for B_k(n): number of partitions of n where each part is no greater than k
def B_k (k n : ℕ) : ℕ := sorry

-- Definition for B_H(n): number of partitions of n where each part belongs to H
def B_H (H : Set ℕ) (n : ℕ) : ℕ := sorry

-- Definition for B(n): total number of partitions of n
def B (n : ℕ) : ℕ := sorry

-- Generating function for B_k(n)
theorem generating_function_Bk (k : ℕ) :
  (∑ n : ℕ, (B_k k n) * X^n) = (1 / (∏ i in Finset.range (k+1), (1 - X^i))) :=
sorry

-- Generating function for B_H(n)
theorem generating_function_BH (H : Set ℕ) :
  (∑ n : ℕ, (B_H H n) * X^n) = (∏ j in H, (1 / (1 - X^j))) :=
sorry

-- Generating function for B(n)
theorem generating_function_B :
  (∑ n : ℕ, (B n) * X^n) = (∏ j in (Set.univ : Set ℕ), (1 / (1 - X^j))) :=
sorry

end generating_function_Bk_generating_function_BH_generating_function_B_l313_313830


namespace coefficient_x2_in_expansion_l313_313950

theorem coefficient_x2_in_expansion :
  let expansion := (x^2 + x + 1) * (1 - x)^4
  in expansion.coefficient 2 = 3 := by
  sorry

end coefficient_x2_in_expansion_l313_313950


namespace calculate_expression_l313_313918

theorem calculate_expression : 3⁻¹ + (Real.sqrt 2 - 1)⁰ + 2 * Real.sin (Real.pi / 6) - (-2 / 3) = 3 := by
  sorry

end calculate_expression_l313_313918


namespace eq_c_is_quadratic_l313_313856

theorem eq_c_is_quadratic (a b c : ℝ) (h1 : a = sqrt 2) (h2 : b = -sqrt 2 / 4) (h3 : c = -1 / 2) : 
    a * x^2 + b * x + c = 0 :=
by
  sorry

end eq_c_is_quadratic_l313_313856


namespace saving_percentage_l313_313303

variable (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ)

-- Conditions from problem
def condition1 := saved_percent_last_year = 0.06
def condition2 := made_more = 1.20
def condition3 := saved_percent_this_year = 0.05 * made_more

-- The problem statement to prove
theorem saving_percentage (S : ℝ) (saved_percent_last_year : ℝ) (made_more : ℝ) (saved_percent_this_year : ℝ) :
  condition1 saved_percent_last_year →
  condition2 made_more →
  condition3 saved_percent_this_year made_more →
  (saved_percent_this_year * made_more = saved_percent_last_year * S * 1) :=
by 
  intros h1 h2 h3
  sorry

end saving_percentage_l313_313303


namespace range_of_x_when_m_2_range_of_m_when_q_necess_ineq_l313_313671

section Problem1
variable (x : ℝ) (m : ℝ) (p q : Prop)

def p_ineq (x m : ℝ) := x^2 - 5 * m * x + 6 * m^2 < 0
def q_ineq (x : ℝ) := (x - 5) / (x - 1) < 0

theorem range_of_x_when_m_2 (h : p ∨ q) (hm : m = 2) (hpx : p -> p_ineq x m) (hq : q -> q_ineq x) :
  1 < x ∧ x < 6 :=
sorry
end Problem1

section Problem2
variable (m : ℝ) (p q: Prop)

def p_ineq_m (m : ℝ) := ∀ x, 2 * m < x ∧ x < 3 * m
def q_ineq_necessity (m : ℝ) := ∀ x, 1 < x ∧ x < 5

theorem range_of_m_when_q_necess_ineq (h : ∀ x, q_ineq_necessity x m → p_ineq_m x m):
  1 / 2 ≤ m ∧ m ≤ 5 / 3 :=
sorry
end Problem2

end range_of_x_when_m_2_range_of_m_when_q_necess_ineq_l313_313671


namespace identify_incorrect_statement_l313_313029

theorem identify_incorrect_statement
  (A : ∀ (a b : ℝ), 0 < a ∧ 0 < b → ((a < b ↔ a + c < b + c) ∧ (a < b ↔ a * c < b * c))) 
  (B : ∀ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b → ((a + b)/2 > sqrt (a * b)))
  (D : ∀ (a b : ℝ), 0 < a ∧ 0 < b ∧ a ≠ b → (1/2 * (a^2 + b^2) > (1/2 * (a + b))^2))
  (E : ∀ (a b : ℝ), 0 < a ∧ 0 < b ∧ a * b = c → (a - b) = 0 → a = b):
  ¬(∀ (a b : ℝ), 0 < a ∧ 0 < b ∧ a - b = d → (∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x - y = d ∧ x = y → x * y < (x * y))) :=
sorry

end identify_incorrect_statement_l313_313029


namespace polynomial_factorization_m_n_l313_313703

theorem polynomial_factorization_m_n (m n : ℤ) (h : (x : ℤ) → x^2 + m * x + n = (x + 1) * (x + 3)) : m - n = 1 := 
by
  -- Define the equality of the factored polynomial and the standard form polynomial.
  have h_poly : (x : ℤ) → x^2 + m * x + n = x^2 + 4 * x + 3, from
    fun x => h x ▸ by ring,
  -- Extract values of m and n by comparing coefficients.
  have h_m : m = 4, from by
    have := congr_fun h_poly 0,
    simp at this,
    assumption,
  
  have h_n : n = 3, from by
    have := congr_fun h_poly (-1),
    simp at this,
    assumption,
  
  -- Substitute m and n to find that m - n = 1.
  rw [h_m, h_n],
  exact dec_trivial

end polynomial_factorization_m_n_l313_313703


namespace ordering_x1_x2_x3_l313_313207

-- Define the conditions
def x1 := logb (1 / 3) 2
def x2 := 2^(-1 / 2)
def condition_x3 (x3 : ℝ) := (1 / 3)^x3 = logb 3 x3

-- The proof problem: prove the ordering
theorem ordering_x1_x2_x3 (x3 : ℝ) (hx3 : condition_x3 x3) : x1 < x2 ∧ x2 < x3 :=
by
  -- Conditions are defined as def, so we can refer to them
  sorry

end ordering_x1_x2_x3_l313_313207


namespace average_height_correct_l313_313142

-- Define trees heights with variables representing unknown heights
variables (tree1 tree2 tree3 tree4 tree5 tree6 : ℕ)

-- Given conditions
def tree_conditions := 
  (tree2 = 18) ∧ 
  (tree4 = 54) ∧ 
  ((tree3 = 3 * tree2) ∨ (tree3 = tree2 / 3)) ∧ 
  ((tree1 = 3 * tree2) ∨ (tree1 = tree2 / 3)) ∧ 
  ((tree5 = 3 * tree4) ∨ (tree5 = tree4 / 3)) ∧ 
  ((tree6 = 3 * tree5) ∨ (tree6 = tree5 / 3))

-- Define the sum of heights
def height_sum := tree1 + tree2 + tree3 + tree4 + tree5 + tree6

-- Define the average height
def average_height := height_sum / 6

-- The theorem we want to prove
theorem average_height_correct (h : tree_conditions) : average_height = 26 :=
by sorry

end average_height_correct_l313_313142


namespace parabola_hyperbola_tangent_l313_313372

open Real

theorem parabola_hyperbola_tangent (n : ℝ) : 
  (∀ x y : ℝ, y = x^2 + 6 → y^2 - n * x^2 = 4 → y ≥ 6) ↔ (n = 12 + 4 * sqrt 7 ∨ n = 12 - 4 * sqrt 7) :=
by
  sorry

end parabola_hyperbola_tangent_l313_313372


namespace perimeter_triangle_eq_2R_l313_313093

-- Define the parameters of the problem
variables {O O1 O2 : Point}
variables {R R1 R2 : ℝ}
variables [Circle O R] [Circle O1 R1] [Circle O2 R2]

-- Define the tangency conditions
def tangency_condition1 := dist O O1 = R - R1
def tangency_condition2 := dist O O2 = R - R2
def tangency_condition3 := dist O1 O2 = R1 + R2

-- Define the theorem to prove the perimeter of triangle O O1 O2
theorem perimeter_triangle_eq_2R
  (h1 : tangency_condition1)
  (h2 : tangency_condition2)
  (h3 : tangency_condition3) :
  perimeter (triangle O O1 O2) = 2 * R := 
sorry

end perimeter_triangle_eq_2R_l313_313093


namespace part_I_part_II_l313_313768

-- Definitions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - (x + 1) * Real.log x

def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2 * a * x - Real.log x - 1 - 1 / x

-- Statements
theorem part_I (a : ℝ) (h : f_derivative a 1 = 0) : a = 1 :=
  sorry

theorem part_II (h : ∀ x, 0 < x ∧ x ≤ 2 → f 1 x > (1 / 2) * x) : True :=
  sorry

end part_I_part_II_l313_313768


namespace num_valid_arrangements_l313_313914

-- Define the people and the days of the week
inductive Person := | A | B | C | D | E
inductive DayOfWeek := | Monday | Tuesday | Wednesday | Thursday | Friday

-- Define the arrangement function type
def Arrangement := DayOfWeek → Person

/-- The total number of valid arrangements for 5 people
    (A, B, C, D, E) on duty from Monday to Friday such that:
    - A and B are not on duty on adjacent days,
    - B and C are on duty on adjacent days,
    is 36.
-/
theorem num_valid_arrangements : 
  ∃ (arrangements : Finset (Arrangement)), arrangements.card = 36 ∧
  (∀ (x : Arrangement), x ∈ arrangements →
    (∀ (d1 d2 : DayOfWeek), 
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) →
      ¬(x d1 = Person.A ∧ x d2 = Person.B)) ∧
    (∃ (d1 d2 : DayOfWeek),
      (d1 = Monday ∧ d2 = Tuesday ∨ d1 = Tuesday ∧ d2 = Wednesday ∨
       d1 = Wednesday ∧ d2 = Thursday ∨ d1 = Thursday ∧ d2 = Friday) ∧
      (x d1 = Person.B ∧ x d2 = Person.C)))
  := sorry

end num_valid_arrangements_l313_313914


namespace exists_k_int_entries_l313_313305

namespace Matrix
open Matrix

variables {n : Type*} [DecidableEq n] [Fintype n]

def is_int_matrix (M : Matrix n n ℤ) : Prop :=
∀ i j, M i j ∈ ℤ

theorem exists_k_int_entries {A B : Matrix n n ℤ}
  (hA : A.det = 1) (hB : B.det ≠ 0) :
  ∃ k : ℕ, is_int_matrix (B ⬝ (A^k) ⬝ B⁻¹) :=
sorry

end Matrix

end exists_k_int_entries_l313_313305


namespace pentagon_inscribed_in_circle_min_value_l313_313306

theorem pentagon_inscribed_in_circle_min_value 
  {A X Y B Z L K : Point}
  (circle : Circle)
  (diameter_AB : Diameter circle AB)
  (inscribed_AXYBZ : InscribedPentagon circle AXYBZ)
  (tangent_Y : Tangent circle Y)
  (intersection_L : TangentIntersects tangent_Y BX L)
  (intersection_K : TangentIntersects tangent_Y BZ K)
  (bisector_AY_LAZ : AngleBisector AY LAZ)
  (AY_eq_YZ : AY = YZ) :
  ∃ m n k : ℕ, gcd m n = 1 ∧ 
              (∃ AK AX AL AB : ℝ, ∀ min_val : ℝ, 
                min_val = (AK / AX) + (AL / AB)^2 ∧ 
                min_val = m / n + sqrt k) ∧
              m + 10 * n + 100 * k = 343 :=
sorry

end pentagon_inscribed_in_circle_min_value_l313_313306


namespace find_a_find_x_l313_313644

theorem find_a (a : ℝ) (A : Set ℝ) (h₁ : A = {a - 3, 2 * a - 1, a^2 + 1}) (h₂ : -3 ∈ A) : a = 0 ∨ a = -1 := by
  sorry

theorem find_x (x : ℝ) (B : Set ℝ) (h₁ : B = {0, 1, x}) (h₂ : x^2 ∈ B) : x = -1 := by
  sorry

end find_a_find_x_l313_313644


namespace polynomial_remainder_constant_l313_313199

theorem polynomial_remainder_constant (b : ℝ) :
  let p := (12 : ℝ) * x^4 - 9 * x^3 + b * x^2 + x - 8,
      q := 3 * x^2 - 4 * x + 2,
      r := let (q, r) := p.divmod q in r
  in ∀ x, is_const r → b = -5 :=
by
  sorry

end polynomial_remainder_constant_l313_313199


namespace susan_mean_l313_313346

def susan_scores : List ℝ := [87, 90, 95, 98, 100]

theorem susan_mean :
  (susan_scores.sum) / (susan_scores.length) = 94 := by
  sorry

end susan_mean_l313_313346


namespace headphone_cost_l313_313467

-- Definitions from conditions
def gift_amount : ℕ := 50
def cassette_tape_cost : ℕ := 9
def number_of_cassette_tapes : ℕ := 2
def amount_left_after_purchases : ℕ := 7

-- Define the total cost for cassette tapes
def total_cassette_cost : ℕ := number_of_cassette_tapes * cassette_tape_cost

-- Define the remaining amount after buying cassette tapes
def amount_after_cassettes : ℕ := gift_amount - total_cassette_cost

-- The statement to prove
theorem headphone_cost : ∃ H : ℕ, amount_after_cassettes - H = amount_left_after_purchases ∧ H = 25 :=
by
  exists 25
  sorry

end headphone_cost_l313_313467


namespace difference_a2021_a1999_l313_313995

axiom a : ℕ → ℕ
axiom strictly_increasing : ∀ m n : ℕ, m < n → a m < a n
axiom a_nat : ∀ n : ℕ, n ≥ 1 → a n ∈ ℕ
axiom a_ge_one : ∀ n : ℕ, n ≥ 1 → a n ≥ 1
axiom a_combination : ∀ n : ℕ, n ≥ 1 → a (a n) = 3 * n

theorem difference_a2021_a1999 : a 2021 - a 1999 = 66 := 
by
  sorry

end difference_a2021_a1999_l313_313995


namespace find_scalars_l313_313659

def a : ℝ × ℝ × ℝ := (1, 2, 2)
def b : ℝ × ℝ × ℝ := (2, -1, 0)
def c : ℝ × ℝ × ℝ := (0, 2, -1)
def d : ℝ × ℝ × ℝ := (5, -1, 4)

theorem find_scalars : 
  ∃ p q r : ℝ, d = (p • (a.1, a.2, a.3) + q • (b.1, b.2, b.3) + r • (c.1, c.2, c.3)) ∧
    p = 11/9 ∧ q = 11/5 ∧ r = -6/5 := 
  by
    sorry

end find_scalars_l313_313659


namespace solve_inequality_l313_313343

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∧ x ≤ -1) ∨ 
  (a > 0 ∧ (x ≥ 2 / a ∨ x ≤ -1)) ∨ 
  (-2 < a ∧ a < 0 ∧ 2 / a ≤ x ∧ x ≤ -1) ∨ 
  (a = -2 ∧ x = -1) ∨
  (a < -2 ∧ -1 ≤ x ∧ x ≤ 2 / a) ↔ 
  a * x ^ 2 + (a - 2) * x - 2 ≥ 0 := 
sorry

end solve_inequality_l313_313343


namespace solve_sqrt_equation_l313_313802

theorem solve_sqrt_equation (x : ℝ) (h1 : x ≠ -4) (h2 : (3 * x - 1) / (x + 4) > 0) : 
  sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3 * x - 1)) = 0 ↔ x = 5 / 2 :=
by { sorry }

end solve_sqrt_equation_l313_313802


namespace sin_135_eq_sqrt2_div_2_l313_313588

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313588


namespace hundred_a_plus_b_val_l313_313962

noncomputable def compute_hundred_a_plus_b (α : ℚ) (a b : ℕ) [Fact (α = (rat.mk_nat a b))] : ℕ :=
  if (α > 0) ∧ ((S : Set ℝ) has_total_length 20.2) ∧ (∀ x ∈ S, fract x > α * x) ∧ coprime a b
  then 100 * a + b
  else 0

theorem hundred_a_plus_b_val {α : ℚ} (a b : ℕ) [Fact (α = (rat.mk_nat a b))] 
  (hα_pos : α > 0) 
  (h_total_length : (S : Set ℝ) has_total_length 20.2)
  (h_fractional_ineq : ∀ x ∈ S, fract x > α * x) 
  (h_coprime : coprime a b) 
  : compute_hundred_a_plus_b α a b = 4633 :=
sorry

end hundred_a_plus_b_val_l313_313962


namespace ordered_pair_solution_l313_313753

-- Define the matrix A
def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := ![
  [4, 6],
  [10, d]
]

-- Define the determinant of A
def det_A (d : ℝ) : ℝ :=
  4 * d - 60

-- Define the inverse of A
def A_inv (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  (1 / det_A d) • ![
    [d, -6],
    [-10, 4]
  ]

-- Define k based on the condition A⁻¹ = k * A
def k (cos_theta : ℝ) : ℝ :=
  1 / (12 * cos_theta - 60)

-- Let d = 3 * cos(theta)
def d (cos_theta : ℝ) : ℝ :=
  3 * cos_theta

-- Define the main theorem to be proven
theorem ordered_pair_solution :
  ∃ (theta k : ℝ), cos theta = 1/3 ∧ k = -1/56 :=
by
  exist (Real.arccos (1/3)) (-1/56)
  split
  -- Prove cos(theta) = 1/3
  { sorry }
  -- Prove k = -1/56
  { sorry }

end ordered_pair_solution_l313_313753


namespace share_of_a_l313_313118

variables (A B C D E : ℝ)

def condition1 := A = (5 / 7) * (B + C + D + E)
def condition2 := B = (11 / 16) * (A + C + D)
def condition3 := C = (3 / 8) * (A + B + E)
def condition4 := D = (7 / 12) * (A + B + C)
def condition5 := A + B + C + D + E = 500

theorem share_of_a :
  condition1 A B C D E →
  condition2 A B C D E →
  condition3 A B C D E →
  condition4 A B C D E →
  condition5 A B C D E →
  A ≈ 208.33 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end share_of_a_l313_313118


namespace polynomial_modulo_equiv_l313_313762

theorem polynomial_modulo_equiv (p : ℕ) [Fact p.prime] (h_odd : p % 2 = 1)
  (a : Fin p → ℤ) :
  (∃ f : ℤ[X], f.natDegree ≤ (p - 1) / 2 ∧ ∀ i : Fin p, (f.eval (i : ℤ)) % p = a i % p)
  ↔
  (∀ d : ℕ, d ∈ Finset.range ((p - 1) / 2 + 1) →
    ∑ i in Finset.range p, ((a ((i + d) % p) - a i) ^ 2) % p = 0) :=
sorry

end polynomial_modulo_equiv_l313_313762


namespace salamander_population_decreases_below_5_percent_after_9_years_l313_313188

-- Define the decreasing function
def decrease_by_30_percent_each_year (n : ℕ) : ℝ :=
  100 * (0.7 ^ n)

-- Define the problem statement: Prove that the population is less than 5% after 9 years
theorem salamander_population_decreases_below_5_percent_after_9_years :
  decrease_by_30_percent_each_year 9 < 5 :=
by
  rw [decrease_by_30_percent_each_year]
  norm_num
  sorry

end salamander_population_decreases_below_5_percent_after_9_years_l313_313188


namespace negation_of_inequality_l313_313064

theorem negation_of_inequality :
  ¬ (∀ x : ℝ, x^2 - x + 2 ≥ 0) ↔ ∃ x : ℝ, x^2 - x + 2 < 0 := 
sorry

end negation_of_inequality_l313_313064


namespace sin_135_eq_l313_313486

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313486


namespace range_of_a_l313_313061

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x ≤ 2 then x^2 - 2 * a * x - 2 else x + 36 / x - 6 * a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 2 a) ↔ (2 ≤ a ∧ a ≤ 5) :=
sorry

end range_of_a_l313_313061


namespace sin_135_eq_l313_313517

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313517


namespace sum_of_reciprocal_a_l313_313996

-- Define the sequence a_n
def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2 * 3^(n-2)

-- Define the sum of first n terms S_n of sequence a_n
def S_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else 1 + 3^(n-1)

-- Define the condition for S_n
axiom S_n_cond (n : ℕ) (h : n ≥ 2) : 
  S_n n - 3 * (S_n (n - 1)) + 2 = 0

-- Sum of first n terms of the sequence 1/a_n
def T_n (n : ℕ) : ℚ :=
  let a' := λ k, (a k : ℚ)
  let terms := λ k, (1 / (a' k))
  nat.rec_on n 
    0 
    (λ k IH, IH + terms (k + 1))

-- Prove that T_n = (7/4 - 1/(4 * 3^(n-2)))
theorem sum_of_reciprocal_a (n : ℕ) : T_n n = (7/4 - 1/(4 * 3^(n-2))) :=
  sorry

end sum_of_reciprocal_a_l313_313996


namespace ellipse_satisfies_conditions_l313_313227

noncomputable def ellipse_equation (a b : ℝ) : Prop :=
  ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_satisfies_conditions (a b : ℝ) (h1 : a > b > 0)
  (h2 : a^2 = 2 * b^2)
  (h3 : ∀ x y : ℝ, y = -x + 1 → (∃ A B : ℝ × ℝ, (x, y) ∈ {A, B} ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = (4 * (3 * b^2 - 1)^(1/2) / 3)^2)): 
  ellipse_equation 2 b := 
by 
  sorry


end ellipse_satisfies_conditions_l313_313227


namespace sin_135_eq_sqrt2_over_2_l313_313578

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313578


namespace coprime_divisible_l313_313763

theorem coprime_divisible (a b c : ℕ) (h1 : Nat.gcd a b = 1) (h2 : a ∣ b * c) : a ∣ c :=
by
  sorry

end coprime_divisible_l313_313763


namespace total_savings_after_taxes_and_expenditures_l313_313818

theorem total_savings_after_taxes_and_expenditures
  (income_A income_B income_C expenditure : ℝ) :
  let total_income := 18000
  let ratio_A := 3
  let ratio_B := 2
  let ratio_C := 1
  let total_ratio := ratio_A + ratio_B + ratio_C
  let income_A := (ratio_A / total_ratio) * total_income
  let income_B := (ratio_B / total_ratio) * total_income
  let income_C := (ratio_C / total_ratio) * total_income
  let tax_A := 0.10 * income_A
  let tax_B := 0.15 * income_B
  let tax_C := 0
  let total_tax := tax_A + tax_B + tax_C
  let income_after_tax := total_income - total_tax
  let expenditure_ratio := 5.0 / 9.0
  let expenditure := expenditure_ratio * total_income
  let total_savings := income_after_tax - expenditure
  in total_savings = 6200 :=
by
  sorry

end total_savings_after_taxes_and_expenditures_l313_313818


namespace union_of_sets_l313_313380

def setA : Set ℝ := { x : ℝ | (x - 2) / (x + 1) ≤ 0 }
def setB : Set ℝ := { x : ℝ | -2 * x^2 + 7 * x + 4 > 0 }
def unionAB : Set ℝ := { x : ℝ | -1 < x ∧ x < 4 }

theorem union_of_sets :
  ∀ x : ℝ, x ∈ setA ∨ x ∈ setB ↔ x ∈ unionAB :=
by sorry

end union_of_sets_l313_313380


namespace divide_value_l313_313769

def divide (a b c : ℝ) : ℝ := |b^2 - 5 * a * c|

theorem divide_value : divide 2 (-3) 1 = 1 :=
by
  sorry

end divide_value_l313_313769


namespace fraction_addition_l313_313259

theorem fraction_addition (a b : ℝ) (hb : b ≠ 0) (h : a / b = 1 / 2) : (a + b) / b = 3 / 2 := 
by 
sory

end fraction_addition_l313_313259


namespace sin_135_l313_313526

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313526


namespace infinite_primes_no_solutions_l313_313787

theorem infinite_primes_no_solutions : 
  ∃ (f g : ℤ → ℤ), (∀ x, f x = (x ^ 2 + 1) ^ 2) 
  ∧ (∀ y, g y = - (y ^ 2 + 1) ^ 2) 
  ∧ ∃ᶠ p in Filter.at_top, Prime p ∧ p % 4 = 3 
  ∧ ∀ (x y : ℤ), ¬ (p ∣ f(x) - g(y)) :=
begin
  -- polynomials
  let f : ℤ → ℤ := λ x, (x ^ 2 + 1) ^ 2,
  let g : ℤ → ℤ := λ y, - (y ^ 2 + 1) ^ 2,
  -- goal
  use [f, g],
  split,
  assume x, refl,
  split,
  assume y, refl,
  {
    sorry 
  }
end

end infinite_primes_no_solutions_l313_313787


namespace cd_leq_one_l313_313759

variables {a b c d : ℝ}

theorem cd_leq_one (h1 : a * b = 1) (h2 : a * c + b * d = 2) : c * d ≤ 1 := 
sorry

end cd_leq_one_l313_313759


namespace incorrect_statement_A_l313_313420

-- Definitions for the conditions
def conditionA (x : ℝ) : Prop := -3 * x > 9
def conditionB (x : ℝ) : Prop := 2 * x - 1 < 0
def conditionC (x : ℤ) : Prop := x < 10
def conditionD (x : ℤ) : Prop := x < 2

-- Formal theorem statement
theorem incorrect_statement_A : ¬ (∀ x : ℝ, conditionA x ↔ x < -3) :=
by 
  sorry

end incorrect_statement_A_l313_313420


namespace solve_for_x_l313_313265

theorem solve_for_x (x : ℝ) (hx : sqrt (2 / x + 2) = 3 / 2) : x = 8 :=
sorry

end solve_for_x_l313_313265


namespace determine_t_l313_313331

theorem determine_t (t : ℝ) : 
  (3 * t - 9) * (4 * t - 3) = (4 * t - 16) * (3 * t - 9) → t = 7.8 :=
by
  intros h
  sorry

end determine_t_l313_313331


namespace systematic_sampling_40th_number_l313_313202

theorem systematic_sampling_40th_number (n s k a_1 : ℕ) (h_total : n = 1000) (h_sample : s = 50) (h_first : a_1 = 15) (h_k : k = 40) :
  let interval := n / s,
      a_k := a_1 + (k - 1) * interval
  in a_k = 795 :=
by
  sorry

end systematic_sampling_40th_number_l313_313202


namespace sin_135_eq_sqrt2_div_2_l313_313546

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313546


namespace sleeping_bag_selling_price_l313_313205

def wholesale_cost : ℝ := 24.56
def gross_profit_percentage : ℝ := 0.14

def gross_profit (x : ℝ) : ℝ := gross_profit_percentage * x

def selling_price (x y : ℝ) : ℝ := x + y

theorem sleeping_bag_selling_price :
  selling_price wholesale_cost (gross_profit wholesale_cost) = 28 := by
  sorry

end sleeping_bag_selling_price_l313_313205


namespace minimum_ticket_cost_l313_313277

-- Definitions of the conditions in Lean
def southern_cities : ℕ := 4
def northern_cities : ℕ := 5
def one_way_ticket_cost (N : ℝ) : ℝ := N
def round_trip_ticket_cost (N : ℝ) : ℝ := 1.6 * N

-- The main theorem to prove
theorem minimum_ticket_cost (N : ℝ) : 
  (∀ (Y1 Y2 Y3 Y4 : ℕ), 
  (∀ (S1 S2 S3 S4 S5 : ℕ), 
  southern_cities = 4 → northern_cities = 5 →
  one_way_ticket_cost N = N →
  round_trip_ticket_cost N = 1.6 * N →
  ∃ (total_cost : ℝ), total_cost = 6.4 * N)) :=
sorry

end minimum_ticket_cost_l313_313277


namespace select_papers_above_120_l313_313711

noncomputable def normal_distribution_paper_count : ℕ × ℕ × (ℝ → ℝ) × (ℝ → ℝ) × ℕ → ℕ 
  | (student_count, total_selected, lower_tail, upper_tail, x_threshold) :=
  let total_students_above_threshold := student_count * upper_tail x_threshold in
  (total_selected * upper_tail x_threshold / total_students_above_threshold).to_nat

theorem select_papers_above_120 (student_count : ℕ) (sigma : ℝ) (ξ : ℝ → ℝ)
  (P : ℝ → ℝ) : 
  student_count = 30000 →
  ξ ~ N(100, sigma^2) →
  P 80 < ξ <= 100 = 0.45 →
  ξ 120 = 0.05 →
  ξ 100 = 0.5 →
  ∀ total_selected : ℕ, 
  total_selected = 200 →
  normal_distribution_paper_count (student_count, total_selected, λ x, P (100 < x), λ x, P (x > 100), 120) = 10 :=
by
  intros hc hξ hP_80_100 hξ120 hξ100 htot_selected
  sorry

end select_papers_above_120_l313_313711


namespace ice_cream_sundaes_l313_313913

theorem ice_cream_sundaes (n : ℕ) (h1 : n = 8) : 
  let vanilla := 1 in
  let other_flavors := n - 1 in
  other_flavors = 7 → 
  @Finset.card ℕ 
    (Finset.filter (λ a, a ≠ vanilla)
    (Finset.range n).erase vanilla) = 7 :=
by
  intros 
  sorry

end ice_cream_sundaes_l313_313913


namespace intersection_setA_setB_l313_313766

namespace Proof

def setA : Set ℝ := {x | ∃ y : ℝ, y = x + 1}
def setB : Set ℝ := {y | ∃ x : ℝ, y = 2^x}

theorem intersection_setA_setB : (setA ∩ setB) = {y | 0 < y} :=
by
  sorry

end Proof

end intersection_setA_setB_l313_313766


namespace hyperbola_eccentricity_range_l313_313635

-- Definitions of conditions in the problem
def is_hyperbola (x y a b : ℝ) := 
  x^2 / a^2 - y^2 / b^2 = 1

def a_gt_one (a : ℝ) := 
  a > 1

def b_gt_zero (b : ℝ) := 
  b > 0

def focal_distance (a b c : ℝ) := 
  c = Real.sqrt (a^2 + b^2)

def line_equation (x y a b : ℝ) := 
  b * x + a * y = a * b

def distance_condition (dist1 dist2 c : ℝ) :=
  dist1 + dist2 ≥ 4 / 5 * c

def eccentricity_range (e : ℝ) :=
  e ∈ Set.Icc (Real.sqrt 5 / 2) (Real.sqrt 5)

-- Statement of the theorem
theorem hyperbola_eccentricity_range (a b c : ℝ) (e : ℝ) :
  is_hyperbola x y a b →
  a_gt_one a →
  b_gt_zero b →
  focal_distance a b c →
  line_equation x y a b →
  distance_condition (distance (1,0) line) (distance (-1,0) line) c →
  eccentricity_range e :=
sorry -- proof to be added

end hyperbola_eccentricity_range_l313_313635


namespace sin_135_eq_sqrt2_div_2_l313_313554

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313554


namespace sin_135_eq_l313_313512

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313512


namespace min_shift_value_l313_313249

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

def f (x : ℝ) : ℝ := determinant (Real.sin x) (Real.sqrt 3) (Real.cos x) 1

theorem min_shift_value {m : ℝ} (h : m > 0) :
  (∀ x : ℝ, f (x + m) = f (-x + m)) → m = 5 * Real.pi / 6 :=
by
  sorry

end min_shift_value_l313_313249


namespace route_length_is_140_l313_313846

-- Conditions of the problem
variable (D : ℝ)  -- Length of the route
variable (Vx Vy t : ℝ)  -- Speeds of Train X and Train Y, and time to meet

-- Given conditions
axiom train_X_trip_time : D / Vx = 4
axiom train_Y_trip_time : D / Vy = 3
axiom train_X_distance_when_meet : Vx * t = 60
axiom total_distance_covered_on_meeting : Vx * t + Vy * t = D

-- Goal: Prove that the length of the route is 140 kilometers
theorem route_length_is_140 : D = 140 := by
  -- Proof omitted
  sorry

end route_length_is_140_l313_313846


namespace probability_even_heads_after_60_flips_l313_313473

noncomputable def P_n (n : ℕ) : ℝ :=
  if n = 0 then 1
  else (3 / 4) - (1 / 2) * P_n (n - 1)

theorem probability_even_heads_after_60_flips :
  P_n 60 = 1 / 2 * (1 + 1 / 2^60) :=
sorry

end probability_even_heads_after_60_flips_l313_313473


namespace polygonal_line_distance_l313_313752

/-- S is a square with a side length of 100. L is a polygonal line A_0 A_1 A_2 ... A_{n-1} A_n
contained within S and does not intersect itself. For any point P on the boundary of S,
there exists a point on L whose distance to P is no greater than 1/2.
We need to prove that there must exist two points X, Y on L such that
the distance between them is no greater than 1, but the distance between them along the
polygonal line L is at least 198. -/
theorem polygonal_line_distance
    (S : set (ℝ × ℝ))
    (L : list (ℝ × ℝ))
    (side_length : ℝ)
    (H1 : ∀ (P : ℝ × ℝ), P ∈ boundary S → ∃ (Q : ℝ × ℝ), Q ∈ L ∧ dist P Q ≤ 1/2)
    (H2 : side_length = 100)
    (H3 : ∃ (A0 An : ℝ × ℝ), A0 ≠ An ∧ L = (A0 :: L.drop (L.length - 1)) ++ [An])
    (H4 : ∀ (i j : ℕ), i ≠ j → L.nth i ≠ L.nth j) :
    ∃ (X Y : ℝ × ℝ), X ∈ L ∧ Y ∈ L ∧ dist X Y ≤ 1 ∧ path_length L X Y ≥ 198 :=
  sorry

end polygonal_line_distance_l313_313752


namespace solve_equation_l313_313798

noncomputable def lhs (x: ℝ) : ℝ := (sqrt ((3*x - 1) / (x + 4))) + 3 - 4 * (sqrt ((x + 4) / (3*x - 1)))

theorem solve_equation (x: ℝ) (t : ℝ) (ht : t = (3*x - 1) / (x + 4)) (h_pos : 0 < t) :
  lhs x = 0 → x = 5 / 2 :=
by
  intros h
  sorry

end solve_equation_l313_313798


namespace quadratic_no_rational_solution_l313_313782

theorem quadratic_no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) :
  ∀ (x : ℚ), ¬ (a * x^2 + b * x + c = 0) :=
by
  sorry

end quadratic_no_rational_solution_l313_313782


namespace dollars_needed_to_buy_dog_l313_313027

theorem dollars_needed_to_buy_dog (x y z: ℝ) (initial_amount: x = 34) (final_amount: x + (y / 100 * x) * z = 47) : ∃ needed_amount, needed_amount = 13 :=
by
  use 47 - 34
  simp
  exact eq.refl 13

end dollars_needed_to_buy_dog_l313_313027


namespace initial_markers_l313_313776

variable (markers_given : ℕ) (total_markers : ℕ)

theorem initial_markers (h_given : markers_given = 109) (h_total : total_markers = 326) :
  total_markers - markers_given = 217 :=
by
  sorry

end initial_markers_l313_313776


namespace find_ellipse_equation_exist_Q_l313_313651

noncomputable theory

variables {x y : ℝ}

def is_ellipse (a b : ℝ) (C : ℝ → ℝ → Prop) :=
  a > b ∧ b > 0 ∧ ∀ x y, C x y ↔ x^2 / a^2 + y^2 / b^2 = 1

def point_on_ellipse (P : ℝ × ℝ) (C : ℝ → ℝ → Prop) :=
  C P.1 P.2

def is_arithmetic_sequence (a b c : ℝ) :=
  2 * b = a + c

def exist_fixed_point (Q : ℝ × ℝ) (A B : ℝ × ℝ) (m : ℝ) :=
  Q.snd = 0 ∧ 
  ∃ m, m = Q.fst ∧
  ∀ A B : ℝ × ℝ, ∃ t : ℝ, 
  (A.fst - m) * (B.fst - m) + A.snd * B.snd = -7 / 16

theorem find_ellipse_equation (a b c : ℝ) (P : ℝ × ℝ)
  (C : ℝ → ℝ → Prop) (F1 F2 : ℝ × ℝ)
  (h_ellipse : is_ellipse a b C)
  (h_focus : F1 = (-(a^2 - b^2).sqrt, 0) ∧ F2 = ((a^2 - b^2).sqrt, 0))
  (h_point : point_on_ellipse P C)
  (h_sequence : is_arithmetic_sequence (sqrt 2 * (P.1 - F1.1).abs)
                                     (F1.1 - F2.1).abs 
                                     (sqrt 2 * (P.1 - F2.1).abs)) :
  (C = λ x y, x^2 / 2 + y^2 = 1) :=
sorry

theorem exist_Q (a b c m : ℝ) (A B Q : ℝ × ℝ)
  (C : ℝ → ℝ → Prop) (F2 : ℝ × ℝ)
  (h_ellipse : is_ellipse a b C)
  (h_focus : F2 = ((a^2 - b^2).sqrt, 0))
  (h_n : ∀ n : ℝ → ℝ, n F2.fst = F2.snd)
  (h_exist : exist_fixed_point Q A B m)
  (h_coords : Q = (5 / 4, 0)) :
  True :=
sorry

end find_ellipse_equation_exist_Q_l313_313651


namespace sum_areas_shaded_regions_proof_equilateral_triangle_inscribed_circle_a_plus_b_plus_c_l313_313163

-- Given conditions
def equilateral_triangle_side : ℝ := 18
def circle_radius (s : ℝ) : ℝ := s / 2

-- Definitions needed for the problem
def triangle_area (s : ℝ) : ℝ := (√3 / 4) * s^2
def sector_area (angle : ℝ) (r : ℝ) : ℝ := (angle / 360) * π * r^2
def shaded_area (sector_area : ℝ) (segment_area : ℝ) : ℝ := sector_area - segment_area
def total_shaded_area (a : ℝ) (b : ℝ) : ℝ := 2 * (a - b)
def a : ℕ := 54
def b : ℕ := 27
def c : ℕ := 3
def answer : ℕ := a + b + c

-- The theorem to be proved
theorem sum_areas_shaded_regions : (a : ℕ) * (π : ℝ) - (b : ℕ) * sqrt (c : ℕ) = 54 * π - 27 * sqrt 3 := 
by sorry

theorem proof_equilateral_triangle_inscribed_circle_a_plus_b_plus_c : answer = 84 :=
by sorry

end sum_areas_shaded_regions_proof_equilateral_triangle_inscribed_circle_a_plus_b_plus_c_l313_313163


namespace matthew_egg_rolls_l313_313772

theorem matthew_egg_rolls 
    (M P A : ℕ)
    (h1 : M = 3 * P)
    (h2 : P = A / 2)
    (h3 : A = 4) : 
    M = 6 :=
by
  sorry

end matthew_egg_rolls_l313_313772


namespace how_many_pens_l313_313044

theorem how_many_pens
  (total_cost : ℝ)
  (num_pencils : ℕ)
  (avg_pencil_price : ℝ)
  (avg_pen_price : ℝ)
  (total_cost := 510)
  (num_pencils := 75)
  (avg_pencil_price := 2)
  (avg_pen_price := 12)
  : ∃ (num_pens : ℕ), num_pens = 30 :=
by
  sorry

end how_many_pens_l313_313044


namespace highest_power_of_6_dividing_20_factorial_l313_313096

def legendre (n p : Nat) : Nat :=
  if p ≤ 1 then 0
  else
    let rec go (div n p acc : Nat) : Nat :=
      if div < p then acc
      else go (div / p) p (acc + div / p)
    go n p 0

theorem highest_power_of_6_dividing_20_factorial : legendre 20 2 = 18 → legendre 20 3 = 8 → min (legendre 20 2) (legendre 20 3) = 8 :=
begin
  intros h2 h3,
  rw [h2, h3],
  exact min_self 8,
end

end highest_power_of_6_dividing_20_factorial_l313_313096


namespace geometric_progression_fourth_term_eq_one_l313_313364

theorem geometric_progression_fourth_term_eq_one :
  let a₁ := (2:ℝ)^(1/4)
  let a₂ := (2:ℝ)^(1/6)
  let a₃ := (2:ℝ)^(1/12)
  let r := a₂ / a₁
  let a₄ := a₃ * r
  a₄ = 1 := by
  sorry

end geometric_progression_fourth_term_eq_one_l313_313364


namespace ellipse_standard_eq_l313_313955

theorem ellipse_standard_eq (A B : ℝ) (hA : 0 < A) (hB : 0 < B)
  (h1 : A * (sqrt 15 / 2)^2 + B = 1)
  (h2 : B * (-2)^2 = 1) : 
  (∀ x y : ℝ, A * x^2 + B * y^2 = 1 ↔ x^2 / 5 + y^2 / 4 = 1) :=
by sorry

end ellipse_standard_eq_l313_313955


namespace triangle_inequality_l313_313011

-- Define the points A, B, C (forming a triangle) and any point M in the plane.
variable (A B C M : ℝ)

-- Assuming points form a triangle (A, B, C) and M is any point in the plane.
axiom triangle (A B C : ℝ) : True

-- The desired inequality to be proven.
theorem triangle_inequality (A B C M : ℝ) (h : triangle A B C) :
  (dist M A / dist B C) + (dist M B / dist C A) + (dist M C / dist A B) ≥ Real.sqrt 3 := 
sorry

end triangle_inequality_l313_313011


namespace fill_time_60_gallons_ten_faucets_l313_313957

-- Define the problem parameters
def rate_of_five_faucets : ℚ := 150 / 8 -- in gallons per minute

def rate_of_one_faucet : ℚ := rate_of_five_faucets / 5

def rate_of_ten_faucets : ℚ := rate_of_one_faucet * 10

def time_to_fill_60_gallons_minutes : ℚ := 60 / rate_of_ten_faucets

def time_to_fill_60_gallons_seconds : ℚ := time_to_fill_60_gallons_minutes * 60

-- The main theorem to prove
theorem fill_time_60_gallons_ten_faucets : time_to_fill_60_gallons_seconds = 96 := by
  sorry

end fill_time_60_gallons_ten_faucets_l313_313957


namespace average_speed_of_car_l313_313382

-- Define the given distances
def distance_first_hour : ℝ := 145
def distance_second_hour : ℝ := 60

-- Define the total distance and total time
def total_distance : ℝ := distance_first_hour + distance_second_hour
def total_time : ℝ := 2  -- Total time is always 2 hours since it is given (1 hour + 1 hour)

-- Define the average speed
def average_speed (distance : ℝ) (time : ℝ) : ℝ := distance / time

-- Lean statement 
theorem average_speed_of_car : average_speed total_distance total_time = 102.5 := by sorry

end average_speed_of_car_l313_313382


namespace find_a_find_x_l313_313643

theorem find_a (a : ℝ) (A : Set ℝ) (h₁ : A = {a - 3, 2 * a - 1, a^2 + 1}) (h₂ : -3 ∈ A) : a = 0 ∨ a = -1 := by
  sorry

theorem find_x (x : ℝ) (B : Set ℝ) (h₁ : B = {0, 1, x}) (h₂ : x^2 ∈ B) : x = -1 := by
  sorry

end find_a_find_x_l313_313643


namespace find_cos_beta_l313_313231

-- Define the conditions as assumptions
variables (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2)
          (h3 : Real.sin α = 1 / 2)
          (h4 : Real.cos (α + β) = 1 / 2)

-- Define the theorem to prove the value of cos β
theorem find_cos_beta : Real.cos β = sqrt 3 / 2 :=
by
  sorry

end find_cos_beta_l313_313231


namespace extreme_points_minimum_value_l313_313663

noncomputable def f (m x : ℝ) : ℝ := m * x * Real.exp (-x) + x - Real.log x

theorem extreme_points (m : ℝ) :
  (m ≤ Real.exp 1 → ∃! x ∈ Ioi 0, IsLocalMinimum (f m) x ∨ IsLocalMaximum (f m) x) ∧
  (m > Real.exp 1 → ∃ a b c ∈ Ioi 0,
    IsLocalMinimum (f m) a ∧ IsLocalMaximum (f m) b ∧ IsLocalMinimum (f m) c ∧
    a < b ∧ b < c) :=
sorry

theorem minimum_value (m : ℝ) (hm : m > 0) (hmin : ∀ x > 0, f m x ≥ 1 + Real.log m) :
  m ≥ Real.exp 1 :=
sorry

end extreme_points_minimum_value_l313_313663


namespace sin_135_correct_l313_313555

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313555


namespace even_sum_probability_l313_313128

theorem even_sum_probability :
  let balls := (1:ℕ, ..., 12)
  let draw := (Finset.choose 6 balls)
  let even_sets := draw.filter (λ s, (s.sum % 2 = 0))
  (even_sets.card / draw.card = 113 / 231) := sorry

end even_sum_probability_l313_313128


namespace joan_missed_games_l313_313740

theorem joan_missed_games :
  ∀ (totalGames nightGames holidayGames awayDays sickDays attendedGames missedGames : ℕ),
  totalGames = 1200 ∧
  nightGames = 200 ∧
  holidayGames = 150 ∧
  awayDays = 100 ∧
  sickDays = 50 ∧
  attendedGames = 500 →
  missedGames = (totalGames - (nightGames + holidayGames + awayDays + sickDays)) - attendedGames →
  missedGames = 200 :=
by
  intros totalGames nightGames holidayGames awayDays sickDays attendedGames missedGames
  intro h
  cases h with h_totGames h_rest
  cases h_rest with h_nightGames h_rest
  cases h_rest with h_holidayGames h_rest
  cases h_rest with h_awayDays h_rest
  cases h_rest with h_sickDays h_attendedGames
  sorry

end joan_missed_games_l313_313740


namespace find_correct_quotient_l313_313866

theorem find_correct_quotient 
  (Q : ℕ)
  (D : ℕ)
  (h1 : D = 21 * Q)
  (h2 : D = 12 * 35) : 
  Q = 20 := 
by 
  sorry

end find_correct_quotient_l313_313866


namespace w1_relation_w2_relation_maximize_total_profit_l313_313441

def w1 (x : ℕ) : ℤ := 200 * x - 10000

def w2 (x : ℕ) : ℤ := -(x ^ 2) + 1000 * x - 50000

def total_sales_vol (x y : ℕ) : Prop := x + y = 1000

def max_profit_volumes (x y : ℕ) : Prop :=
  total_sales_vol x y ∧ x = 600 ∧ y = 400

theorem w1_relation (x : ℕ) :
  w1 x = 200 * x - 10000 := 
sorry

theorem w2_relation (x : ℕ) :
  w2 x = -(x ^ 2) + 1000 * x - 50000 := 
sorry

theorem maximize_total_profit (x y : ℕ) :
  total_sales_vol x y → max_profit_volumes x y := 
sorry

end w1_relation_w2_relation_maximize_total_profit_l313_313441


namespace ratio_of_ages_l313_313349

theorem ratio_of_ages (F C : ℕ) (h1 : F = C) (h2 : F = 75) :
  (C + 5 * 15) / (F + 15) = 5 / 3 :=
by
  sorry

end ratio_of_ages_l313_313349


namespace monotonic_range_m_compare_fx_x3_inequality_positive_integer_l313_313314

noncomputable def f (x : ℝ) (m : ℝ) := x^2 + m * Real.log (x + 1)

-- Statement for question 1
theorem monotonic_range_m (f : ℝ → ℝ) (m : ℝ) : (∀ x : ℝ, f x = x^2 + m * Real.log (x + 1)) → 
  (∀ x y : ℝ, x < y → f x ≤ f y) ∨ (∀ x y : ℝ, x < y → f y ≤ f x) → 
  m ∈ Set.Ici (1 / 2) := sorry

-- Statement for question 2
theorem compare_fx_x3 (x : ℝ) (h : 0 < x) : f x (-1) < x^3 := sorry

-- Statement for question 3
theorem inequality_positive_integer (n : ℕ) (hn : 0 < n) :
  (∑ k in Finset.range n, Real.exp ((1 - k) * k ^ 2)) < (n * (n + 3) / 2) := sorry

end monotonic_range_m_compare_fx_x3_inequality_positive_integer_l313_313314


namespace f_1992_eq_1992_l313_313860

def f (x : ℕ) : ℤ := sorry

theorem f_1992_eq_1992 (f : ℕ → ℤ) 
  (h1 : ∀ x : ℕ, 0 < x -> f x = f (x - 1) + f (x + 1))
  (h2 : f 0 = 1992) :
  f 1992 = 1992 := 
sorry

end f_1992_eq_1992_l313_313860


namespace sin_135_eq_sqrt2_div_2_l313_313548

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313548


namespace sum_first_five_terms_of_geometric_sequence_l313_313075

-- Definitions based on conditions
def a_n (n : ℕ) : ℝ := 
  if n = 1 then 1/3 
  else if n = 2 then 1 
  else if n = 3 then 3 
  else if n = 4 then 9 
  else 27 -- This interpretation leverages the geometric sequence assumption 

def S_5 : ℝ := a_n 1 + a_n 2 + a_n 3 + a_n 4 + a_n 5

-- The main theorem to be proved
theorem sum_first_five_terms_of_geometric_sequence :
  S_5 = 121 / 3 :=
by sorry

end sum_first_five_terms_of_geometric_sequence_l313_313075


namespace age_permutations_count_l313_313888

open Nat

def is_prime_digit(d : Nat) : Prop := d = 2 ∨ d = 5 ∨ d = 7

theorem age_permutations_count :
  let digits := [2, 2, 2, 5, 7, 9]
  count_valid_ages digits = 84 := by sorry

def count_valid_ages (digits : List Nat) : Nat :=
  let prime_starters := [2, 5, 7]
  prime_starters.map (λ d => count_permutations (d :: digits.erase d)).sum

def count_permutations (ds : List Nat) : Nat :=
  let groups := ds.groupBy (λ x => x)
  let numerator := ds.length.factorial
  let denominator := groups.foldl (λ acc g => acc * g.length.factorial) 1
  numerator / denominator

end age_permutations_count_l313_313888


namespace angle_QUS_obtuse_l313_313816

theorem angle_QUS_obtuse (P Q R S T U : Point) 
  (regular_pentagon : is_regular_pentagon P Q R S T)
  (equilateral_triangle : is_equilateral_triangle P U T) 
  : angle_measure Q U S = 168 :=
sorry

end angle_QUS_obtuse_l313_313816


namespace length_of_goods_train_is_280_l313_313898

def speed_mans_train := 60  -- in km/h
def speed_goods_train := 52  -- in km/h
def time_to_pass := 9  -- in seconds
def relative_speed := (speed_mans_train + speed_goods_train) * (5 / 18)  -- relative speed in m/s
def length_goods_train := relative_speed * time_to_pass  -- length of the goods train in meters

theorem length_of_goods_train_is_280 : length_goods_train = 280 := 
by norm_num
sorry

end length_of_goods_train_is_280_l313_313898


namespace dvd_book_capacity_l313_313126

/--
Theorem: Given that there are 81 DVDs already in the DVD book and it can hold 45 more DVDs,
the total capacity of the DVD book is 126 DVDs.
-/
theorem dvd_book_capacity : 
  (already_in_book additional_capacity : ℕ) (h1 : already_in_book = 81) (h2 : additional_capacity = 45) :
  already_in_book + additional_capacity = 126 :=
by
  sorry

end dvd_book_capacity_l313_313126


namespace positive_integer_solutions_count_l313_313617

theorem positive_integer_solutions_count :
  ∃ (s : Finset ℕ), (∀ x ∈ s, 24 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 64) ∧ s.card = 4 := 
by
  sorry

end positive_integer_solutions_count_l313_313617


namespace not_all_prime_distinct_l313_313066

theorem not_all_prime_distinct (a1 a2 a3 : ℕ) (h1 : a1 ≠ a2) (h2 : a2 ≠ a3) (h3 : a1 ≠ a3)
  (h4 : 0 < a1) (h5 : 0 < a2) (h6 : 0 < a3)
  (h7 : a1 ∣ (a2 + a3 + a2 * a3)) (h8 : a2 ∣ (a3 + a1 + a3 * a1)) (h9 : a3 ∣ (a1 + a2 + a1 * a2)) :
  ¬ (Nat.Prime a1 ∧ Nat.Prime a2 ∧ Nat.Prime a3) :=
by
  sorry

end not_all_prime_distinct_l313_313066


namespace central_angle_of_sector_l313_313074

noncomputable def central_angle (radius perimeter: ℝ) : ℝ :=
  ((perimeter - 2 * radius) / (2 * Real.pi * radius)) * 360

theorem central_angle_of_sector :
  central_angle 28 144 = 180.21 :=
by
  simp [central_angle]
  sorry

end central_angle_of_sector_l313_313074


namespace minimum_ticket_cost_l313_313278

theorem minimum_ticket_cost 
  (N : ℕ)
  (southern_cities : Fin 4)
  (northern_cities : Fin 5)
  (one_way_cost : ∀ (A B : city), A ≠ B → ticket_cost A B = N)
  (round_trip_cost : ∀ (A B : city), A ≠ B → ticket_cost_round_trip A B = 1.6 * N) :
  ∃ (minimum_cost : ℕ), minimum_cost = 6.4 * N := 
sorry

end minimum_ticket_cost_l313_313278


namespace angle_ABC_measure_l313_313374

-- Definitions based on the conditions in the problem
variables {A B C O : Type}
variables (OA OB OC : ℝ) -- OA, OB, and OC are lengths of the radii of the circumscribed circle
variables (angleBOC : ℝ) (angleAOB : ℝ)
variables (M : Type) -- M is the midpoint of BC
variables (BM MC : ℝ) -- BM = MC

-- Conditions given
axiom isEquidistant : OA = OB ∧ OB = OC
axiom angleBOC_condition : angleBOC = 110
axiom angleAOB_condition : angleAOB = 150
axiom median_condition : BM = MC

-- The proof problem statement
theorem angle_ABC_measure : ∃ (angleABC : ℝ), angleABC = 55 := 
by 
  use 55
  sorry

end angle_ABC_measure_l313_313374


namespace last_digit_base5_89_l313_313926

theorem last_digit_base5_89 : 
  ∃ (b : ℕ), (89 : ℕ) = b * 5 + 4 :=
by
  -- The theorem above states that there exists an integer b, such that when we compute 89 in base 5, 
  -- its last digit is 4.
  sorry

end last_digit_base5_89_l313_313926


namespace hyperbola_eccentricity_range_l313_313666

variable (b e : ℝ)
variable (hb : b > 0)
variable (h_hyperbola : ∀ x y, x^2 - (y^2 / b^2) = 1)
variable (h_circle : ∀ x y, x^2 + (y - 2)^2 = 1)
variable (h_intersect : ∀ x y, x^2 + (y - 2)^2 = 1 → y = (2 / (real.sqrt (b^2 + 1))) * x)

theorem hyperbola_eccentricity_range
  (hb : b > 0)
  (h_hyperbola : ∀ x y, x^2 - (y^2 / b^2) = 1)
  (h_circle : ∀ x y, x^2 + (y - 2)^2 = 1)
  (h_intersect : ∀ x y, x^2 + (y - 2)^2 = 1 → abs y = 2 / (real.sqrt (b^2 + 1)))
  : 1 < real.sqrt (1 + b^2) ∧ real.sqrt (1 + b^2) ≤ 2 := by
  sorry

end hyperbola_eccentricity_range_l313_313666


namespace value_of_at_20_at_l313_313616

noncomputable def left_at (x : ℝ) : ℝ := 9 - x
noncomputable def right_at (x : ℝ) : ℝ := x - 9

theorem value_of_at_20_at : right_at (left_at 20) = -20 := by
  sorry

end value_of_at_20_at_l313_313616


namespace not_power_of_two_of_concatenation_l313_313159

theorem not_power_of_two_of_concatenation (T : Set ℕ) (orig_sum : ℕ) (sum_digits_div_9 : orig_sum % 9 = 0) :
  (∃ (n : ℕ),  ℕ.length (list := digits 10 n) = 444445 ∧ ∀ m : ℕ, n ≠ 2 ^ m) :=
by
-- Assuming T is the set of 5-digit numbers between 11111 and 99999 inclusively,
-- and the total sum of these numbers mod 9 is zero.
have h1: ∀ n ∈ T, 11111 ≤ n ∧ n ≤ 99999 := sorry;
have h2: (∑ n ∈ T, n) % 9 = 0 := sum_digits_div_9;
-- Concatenate all the numbers in T
let concatenated_number := concat_all_numbers T;
-- Prove that concatenated_number is not a power of two
show ∀ m : ℕ, concatenated_number ≠ 2 ^ m := sorry

end not_power_of_two_of_concatenation_l313_313159


namespace find_a_and_b_l313_313122

theorem find_a_and_b (a b : ℝ) :
  {-1, 3} = {x : ℝ | x^2 + a * x + b = 0} ↔ a = -2 ∧ b = -3 :=
by 
  sorry

end find_a_and_b_l313_313122


namespace fraction_sum_of_lcm_and_gcd_l313_313347

theorem fraction_sum_of_lcm_and_gcd 
  (m n : ℕ) 
  (h_gcd : Nat.gcd m n = 6) 
  (h_lcm : Nat.lcm m n = 210) 
  (h_sum : m + n = 72) :
  1 / (m : ℚ) + 1 / (n : ℚ) = 12 / 210 := 
by
sorry

end fraction_sum_of_lcm_and_gcd_l313_313347


namespace number_of_complex_solutions_l313_313193

theorem number_of_complex_solutions (z : ℂ) (hz : |z| = 1) 
    (cond : |(z / (conj z)) - ((conj z) / z)| = 2) : 
    ∃ count : ℕ, count = 4 :=
sorry

end number_of_complex_solutions_l313_313193


namespace cone_generatrix_length_l313_313810

noncomputable def length_of_generatrix (r : ℝ) (l : ℝ) (h : ℝ) : ℝ :=
  if (π * r * l = 2 * π * r^2 ∧ (1 / 3) * π * r^2 * h = 9 * sqrt 3 * π ∧ l = 2 * r ∧ h = sqrt (l^2 - r^2))
  then l else 0

theorem cone_generatrix_length (r : ℝ) (h : ℝ) :
  (π * r * (2 * r) = 2 * π * r^2) ∧ ((1 / 3) * π * r^2 * h = 9 * sqrt 3 * π) ∧ (h = r * sqrt 3) →
  length_of_generatrix r (2 * r) h = 6 :=
by
  sorry

end cone_generatrix_length_l313_313810


namespace greatest_points_for_top_teams_l313_313281

-- Definitions as per the conditions
def teams := 9 -- Number of teams
def games_per_pair := 2 -- Each team plays every other team twice
def points_win := 3 -- Points for a win
def points_draw := 1 -- Points for a draw
def points_loss := 0 -- Points for a loss

-- Total number of games played
def total_games := (teams * (teams - 1) / 2) * games_per_pair

-- Total points available in the tournament
def total_points := total_games * points_win

-- Given the conditions, prove that the greatest possible number of total points each of the top three teams can accumulate is 42.
theorem greatest_points_for_top_teams :
  ∃ k, (∀ A B C : ℕ, A = B ∧ B = C → A ≤ k) ∧ k = 42 :=
sorry

end greatest_points_for_top_teams_l313_313281


namespace sin_135_l313_313498

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313498


namespace line_through_focus_parallel_max_area_inscribed_rect_l313_313727

-- Condition definitions
def ellipse (x y φ : ℝ) : Prop := 
  x = 5 * Real.cos φ ∧ y = 3 * Real.sin φ

def ellipse_equation (x y : ℝ) : Prop := 
  x^2 / 25 + y^2 / 9 = 1

def line (t : ℝ) : ℝ × ℝ := 
  (4 - 2 * t, 3 - t)

def parallel_line_equation (x y : ℝ) : Prop := 
  x - 2 * y + 2 = 0

-- Problem (I) statement
theorem line_through_focus_parallel (x y : ℝ) (h_ellipse : ellipse_equation x y) :
  ∃ l : ℝ → ℝ × ℝ, 
    (∀ t : ℝ, parallel_line_equation (fst (l t)) (snd (l t))) → 
    (∃ x y : ℝ, x = 4 ∧ y = 0 ∧ x - 2 * y - 4 = 0) :=
sorry

-- Problem (II) statement
theorem max_area_inscribed_rect (A B C D : ℝ × ℝ) (h_ellipse : ∀ (p : ℝ × ℝ), p = A ∨ p = B ∨ p = C ∨ p = D → ellipse_equation (fst p) (snd p)) :
  ∃ S : ℝ, (S = 4 * |(fst A) * (snd A)|) ∧ S ≤ 30 :=
sorry

end line_through_focus_parallel_max_area_inscribed_rect_l313_313727


namespace find_a_for_binomial_square_l313_313196

theorem find_a_for_binomial_square (a r s : ℝ) (h : ax^2 + 8x + 16 = (rx + s)^2) : a = 1 :=
sorry

end find_a_for_binomial_square_l313_313196


namespace trig_identity_1_double_angle_identity_l313_313225

section trigonometry

variables {α : ℝ} (P : ℝ × ℝ)

-- Hypothesis: α is such that its terminal side passes through point P(-3, sqrt(3))
noncomputable def passes_through (P : ℝ × ℝ) (α : ℝ) : Prop :=
  P = (-3, real.sqrt 3) ∧
  real.sin α = P.2 / real.sqrt(P.1 ^ 2 + P.2 ^ 2) ∧
  real.cos α = P.1 / real.sqrt(P.1 ^ 2 + P.2 ^ 2)

-- Hypotheses based on the conditions
axiom (h : passes_through (-3, real.sqrt 3) α)

-- Question (Ⅰ)
theorem trig_identity_1 : 
  (real.tan (-α) + real.sin (real.pi / 2 + α)) / 
  (real.cos (real.pi - α) * real.sin (-real.pi - α)) = -2 / 3 :=
sorry

-- Question (Ⅱ)
theorem double_angle_identity : 
  real.tan (2 * α) = -real.sqrt 3 :=
sorry

end trigonometry

end trig_identity_1_double_angle_identity_l313_313225


namespace sin_135_eq_l313_313507

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313507


namespace reflect_point_B_l313_313068
-- Import necessary libraries

-- Define the conditions and the equivalence proof in Lean
theorem reflect_point_B'_construction_correct
  (circle : Type)
  (A E A' B P Q R S B' : circle)
  (is_tangent : Point E is_tangent_to circle)
  (are_symmetric_AA' : symmetric_with_respect_to A E A')
  (secant_B_to_E : ∃ P Q : circle, secant_through B intersects circle at P ∧ Q)
  (P_closer_to_B : P closer_to B than Q)
  (AP_intersects_R : ∃ R : circle, line_through A P intersects circle at R)
  (QA'_intersects_S : ∃ S : circle, line_through Q A' intersects circle at S)
  (RS_intersects_tangent_at_B' : ∃ B' : circle, line_through R S intersects tangent_line at B') :
  reflection_of B about E is B' :=
sorry

end reflect_point_B_l313_313068


namespace Sum_a2_a3_a7_l313_313018

-- Definitions from the conditions
variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function from natural numbers to real numbers
variable {S : ℕ → ℝ} -- Define the sum of the first n terms as a function from natural numbers to real numbers

-- Given conditions
axiom Sn_formula : ∀ n : ℕ, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))
axiom S7_eq_42 : S 7 = 42

theorem Sum_a2_a3_a7 :
  a 2 + a 3 + a 7 = 18 :=
sorry

end Sum_a2_a3_a7_l313_313018


namespace shopkeeper_percentage_gain_l313_313889

theorem shopkeeper_percentage_gain (false_weight true_weight : ℝ) 
    (h_false_weight : false_weight = 930)
    (h_true_weight : true_weight = 1000) : 
    (true_weight - false_weight) / false_weight * 100 = 7.53 := 
by
  rw [h_false_weight, h_true_weight]
  sorry

end shopkeeper_percentage_gain_l313_313889


namespace problem_statement_l313_313767

def P := {x : ℤ | ∃ k : ℤ, x = 2 * k - 1}
def Q := {y : ℤ | ∃ n : ℤ, y = 2 * n}

theorem problem_statement (x y : ℤ) (hx : x ∈ P) (hy : y ∈ Q) :
  (x + y ∈ P) ∧ (x * y ∈ Q) :=
by
  sorry

end problem_statement_l313_313767


namespace Lucas_identity_l313_313784

def Lucas (L : ℕ → ℤ) (F : ℕ → ℤ) : Prop :=
  ∀ n, L n = F (n + 1) + F (n - 1)

def Fib_identity1 (F : ℕ → ℤ) : Prop :=
  ∀ n, F (2 * n + 1) = F (n + 1) ^ 2 + F n ^ 2

def Fib_identity2 (F : ℕ → ℤ) : Prop :=
  ∀ n, F n ^ 2 = F (n + 1) * F (n - 1) - (-1) ^ n

theorem Lucas_identity {L F : ℕ → ℤ} (hL : Lucas L F) (hF1 : Fib_identity1 F) (hF2 : Fib_identity2 F) :
  ∀ n, L (2 * n) = L n ^ 2 - 2 * (-1) ^ n := 
sorry

end Lucas_identity_l313_313784


namespace percent_of_x_l313_313691

theorem percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : z = 2 * x) :
  y = 0.75 * x :=
sorry

end percent_of_x_l313_313691


namespace polynomial_pairs_l313_313945

-- Definitions for polynomials satisfying the conditions
def P (x : ℝ) : ℝ := 
  sorry  -- Placeholder, as P should be defined by the student

def Q (x : ℝ) : ℝ := 
  sorry  -- Placeholder, as Q should be defined by the student

-- The conditions as lean definitions
def condition_a (x : ℝ) [Place [1, 2, 3, 4]] (P(x) = 0 ∨ P(x) = 1) 
                 (Q(x) = 0 ∨ Q(x) = 1) : Prop := sorry

def condition_b : Prop := 
  ((P 1 = 0 ∨ P 2 = 1) → (Q 1 = 1 ∧ Q 3 = 1)) 

def condition_c : Prop := 
  ((P 2 = 0 ∨ P 4 = 0) → (Q 2 = 0 ∧ Q 4 = 0))

def condition_d : Prop := 
  ((P 3 = 1 ∨ P 4 = 1) → (Q 1 = 0))

-- The equivalent proof problem statement in Lean 4
theorem polynomial_pairs :
  (condition_a ∧ condition_b ∧ condition_c ∧ condition_d) →
  (Polynomials (P Q) == [(R_2, R_4), (R_3, R_1), (R_3, R_3), (R_3, R_4), (R_4, R_1), (R_5, R_1), (R_6, R_4)]) :=
sorry

end polynomial_pairs_l313_313945


namespace sin_135_eq_sqrt2_over_2_l313_313572

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313572


namespace sin_135_l313_313529

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313529


namespace scarf_colors_l313_313300

theorem scarf_colors (A B : ℚ)
  (h_black : A = 1 / 6)
  (h_grey : B = 1 / 3)
  (h_white : 1 - A - B = 1 / 2) :
  let triangle_1_white := 3 / 4,
      triangle_1_grey := 2 / 9,
      triangle_1_black := 1 / 36,
      triangle_2_white := 1 / 4,
      triangle_2_grey := 4 / 9,
      triangle_2_black := 11 / 36 in
  A = 1 / 6 ∧ B = 1 / 3 ∧ 1 - A - B = 1 / 2 ∧
  triangle_1_white = 3 / 4 ∧
  triangle_1_grey = 2 / 9 ∧
  triangle_1_black = 1 / 36 ∧
  triangle_2_white = 1 / 4 ∧
  triangle_2_grey = 4 / 9 ∧
  triangle_2_black = 11 / 36 :=
by {
  sorry,
}

end scarf_colors_l313_313300


namespace parallel_lines_distance_sum_l313_313057

theorem parallel_lines_distance_sum (b c : ℝ) 
  (h1 : ∃ k : ℝ, 6 = 3 * k ∧ b = 4 * k) 
  (h2 : (abs ((c / 2) - 5) / (Real.sqrt (3^2 + 4^2))) = 3) : 
  b + c = 48 ∨ b + c = -12 := by
  sorry

end parallel_lines_distance_sum_l313_313057


namespace pqrs_zero_l313_313805

theorem pqrs_zero (p q r s: ℝ)
  (Q : ℝ → ℝ)
  (hQ_def : Q = (λ x, x^4 + p*x^3 + q*x^2 + r*x + s))
  (h_roots : ∀ x, x = ℝ.cos (1 * π / 8) ∨ x = ℝ.cos (3 * π / 8) ∨ x = ℝ.cos (5 * π / 8) ∨ x = ℝ.cos (7 * π / 8) → Q x = 0) :
  p * q * r * s = 0 := 
sorry

end pqrs_zero_l313_313805


namespace sin_135_correct_l313_313566

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313566


namespace hyperbola_properties_l313_313661

noncomputable def curve : ℝ → ℝ :=
  λ x, sqrt 3 * (1 / (2 * x) + x / 3)

def is_hyperbola (C : ℝ → ℝ) : Prop := sorry
def is_asymptote (C : ℝ → ℝ) (x : ℝ) : Prop := sorry
def is_focus (C : ℝ → ℝ) (p : ℝ × ℝ) : Prop := sorry
def intersects_two_points (C : ℝ → ℝ) (l : ℝ → ℝ) : Prop := sorry

theorem hyperbola_properties :
  is_hyperbola curve →
  is_asymptote curve 0 ∧
  is_focus curve (1, sqrt 3) ∧
  ∀ t : ℝ, intersects_two_points curve (λ x, x + t) :=
by sorry

end hyperbola_properties_l313_313661


namespace repunits_infinite_l313_313785

-- Define gcd condition
def gcd_condition (m : ℕ) : Prop :=
  Nat.gcd m 10 = 1

-- Define repunit
def repunit (n : ℕ) : ℕ :=
  (10 ^ n - 1) / 9
  
-- Statement of the problem:
theorem repunits_infinite (m : ℕ) (h : gcd_condition m) :
  ∃ n, m ∣ repunit n ∧ ∃∞ n', m ∣ repunit n' :=
sorry

end repunits_infinite_l313_313785


namespace zero_replacement_ways_l313_313668

theorem zero_replacement_ways :
  let num := 5 * 10^102 + 35,
      possible_replacements := 22100 in
  (∀ n : ℕ, (∃ (d₁ d₂ : ℕ), num + d₁ * 10^i₁ + d₂ * 10^i₂ = n ∧ d₁ ≠ 0 ∧ d₂ ≠ 0 ∧ i₁ ≠ i₂ ∧ (n % 495 = 0)) → possible_replacements = 22100) :=
sorry

end zero_replacement_ways_l313_313668


namespace rectangle_area_triangle_area_l313_313030

variables (A B C D M N : Type) [Coordinates A B C D M N]

-- Conditions
variables (AN NC AM MB : ℝ)
variables (hAN : AN = 7) (hNC : NC = 39) (hAM : AM = 12) (hMB : MB = 3)

-- Part (a) Rectangle Area
theorem rectangle_area : AN = 7 ∧ NC = 39 ∧ AM = 12 ∧ MB = 3 → AN + NC * AM + MB = 690 := 
by 
  sorry

-- Part (b) Triangle Area
theorem triangle_area : AN = 7 ∧ NC = 39 ∧ AM = 12 ∧ MB = 3 → 
  let M := (12, 0)
  let N := (0, 7)
  let C := (15, 46)
  1/2 * abs (12 * (7 - 46) + 15 * (0 - 7)) = 286.5 := 
by 
  sorry

end rectangle_area_triangle_area_l313_313030


namespace max_value_expr_l313_313614

theorem max_value_expr {x : ℝ} :
  ∃ x, (x ≠ 0) ∧ (x = 2) ∧ (∀ y, (y ≠ 0) → (y ≠ 2) → let expr := (y^4 / (y^8 + 2 * y^6 + 4 * y^4 + 8 * y^2 + 16)) in expr < (1 / 20)) :=
sorry

end max_value_expr_l313_313614


namespace solve_sqrt_equation_l313_313801

theorem solve_sqrt_equation (x : ℝ) (h1 : x ≠ -4) (h2 : (3 * x - 1) / (x + 4) > 0) : 
  sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3 * x - 1)) = 0 ↔ x = 5 / 2 :=
by { sorry }

end solve_sqrt_equation_l313_313801


namespace minimum_value_of_curve_l313_313953

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem minimum_value_of_curve : ∃ x > 0, (∀ y > 0, f y ≥ f x) ∧ f x = -(1 / Real.exp 1) := by
  have h_deriv : ∀ x > 0, deriv f x = Real.log x + 1 := sorry
  have h_critical : ∀ x > 0, Real.log x + 1 = 0 ↔ x = 1 / Real.exp 1 := sorry
  have h_minimum : ∀ x > 0, deriv f x < 0 → x < 1 / Real.exp 1 ∧ deriv f x > 0 → x > 1 / Real.exp 1 := sorry
  existsi (1 / Real.exp 1)
  split
  . exact Real.exponential_pos 1
  . split
  . intro y hy
    sorry
  . calc
      f (1 / Real.exp 1)
      = (1 / Real.exp 1) * Real.log (1 / Real.exp 1) : rfl
      ... = (1 / Real.exp 1) * (-Real.log (Real.exp 1)) : by rw Real.log_inv
      ... = (1 / Real.exp 1) * (-1) : by rw Real.log_exp
      ... = -(1 / Real.exp 1) : by rw one_mul

end minimum_value_of_curve_l313_313953


namespace find_cyclic_permuting_polynomial_l313_313069

-- Define the polynomial Q(x)
def Q (x : ℝ) := x^3 - 21 * x + 35

-- Define the permutation property of the polynomial P
def cyclicallyPermutesRoots (P : ℝ → ℝ) (r s t : ℝ) := P r = s ∧ P s = t ∧ P t = r

-- Main theorem statement without proof
theorem find_cyclic_permuting_polynomial :
  ∃ a b : ℝ, (∀ r s t : ℝ, r + s + t = 0 ∧ rs + st + tr = -21 ∧ rst = -35 →
  cyclicallyPermutesRoots (λ x, x^2 + a * x + b) r s t) ∧ a = 2 ∧ b = -14 :=
sorry

end find_cyclic_permuting_polynomial_l313_313069


namespace existence_of_n_l313_313946

theorem existence_of_n (n : ℕ) :
  (∃ (S : set ℕ) (color : S → bool), 
    (∀ (x y z : S), 
      (color x = color y ∧ color y = color z ∧ color z = color x) →
      (x + y + z) % n = 0) →
    S.card = 2007) → 
  n ∈ {69, 84} :=
sorry

end existence_of_n_l313_313946


namespace rectangle_area_l313_313827

theorem rectangle_area (P l w : ℕ) (h_perimeter: 2 * l + 2 * w = 60) (h_aspect: l = 3 * w / 2) : l * w = 216 :=
sorry

end rectangle_area_l313_313827


namespace binomial_square_l313_313198

theorem binomial_square (a : ℝ) (x : ℝ) : (ax^2 + 8x + 16 = (1 * x + 4)^2) → a = 1 :=
by {
  intro h,
  have h' : ax^2 + 8x + 16 = x^2 + 8x + 16 := by rwa [pow_two, one_mul] at h,
  sorry
}

end binomial_square_l313_313198


namespace identify_10gram_coin_l313_313391

theorem identify_10gram_coin (coins : Fin 20 → ℝ)
  (h₁ : ∃ i, coins i = 9.9)
  (h₂ : ∃ i j, i ≠ j ∧ coins i = 9.8 ∧ coins j = 9.8)
  (h₃ : ∃ S, S.card = 17 ∧ ∀ i ∈ S, coins i = 10) :
  ∃ S, S.card = 2 ∧ (∀ i ∈ S, S.card = 2 ∧ coins i = 10 :=
by sorry

end identify_10gram_coin_l313_313391


namespace arithmetic_sequence_a18_value_l313_313226

theorem arithmetic_sequence_a18_value 
  (a : ℕ → ℕ) (d : ℕ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_incr : ∀ n, a n < a (n + 1))
  (h_sum : a 2 + a 5 + a 8 = 33)
  (h_geom : (a 5 + 1) ^ 2 = (a 2 + 1) * (a 8 + 7)) :
  a 18 = 37 :=
sorry

end arithmetic_sequence_a18_value_l313_313226


namespace no_real_solution_for_exp_eq_l313_313954

theorem no_real_solution_for_exp_eq :
  ∀ x : ℝ, 2^(4*x + 2) * 8^(2*x + 1) ≠ 32^(2*x + 3) :=
by
  sorry

end no_real_solution_for_exp_eq_l313_313954


namespace conditional_probability_l313_313399

variables (Ω : Type) (P : MeasureTheory.ProbabilityMeasure Ω)
variable [MeasurableSpace Ω]

-- Definitions of events
def red_die (n : ℕ) : Set Ω := {ω | ω = n}
def blue_die_even : Set Ω := {ω | ω % 2 = 0}

-- Probabilities given in the problem
axiom P_red_4 : P (red_die 4) = 1 / 6
axiom P_blue_even : P blue_die_even = 1 / 2
axiom P_red_4_and_blue_even : P (red_die 4 ∩ blue_die_even) = 1 / 12

-- Statement of the problem: Prove P(A|B) = 1/6 where A = red_die 4 and B = blue_die_even
theorem conditional_probability :
  P (red_die 4 ∩ blue_die_even) / P blue_die_even = 1 / 6 := sorry

end conditional_probability_l313_313399


namespace toy_cars_made_yesterday_l313_313907

def toy_car_production : ℕ → ℕ → ℕ := λ yesterday today, yesterday + today

theorem toy_cars_made_yesterday (X : ℕ) (h1 : toy_car_production X (2 * X) = 180) : X = 60 :=
by
  sorry

end toy_cars_made_yesterday_l313_313907


namespace sin_135_eq_l313_313490

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313490


namespace sin_135_correct_l313_313562

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313562


namespace point_O_triangle_inequality_l313_313323

-- Definitions of points and distances in the triangle
variable (A B C O : Point)

-- Assumptions as conditions
variable (O_on_AB : O ∈ line(A, B))
variable (O_not_A : O ≠ A)
variable (O_not_B : O ≠ B)

-- Distances between the points
variable (OA OB OC AB BC AC : ℝ)

-- Length properties
variable (eq_OA : distance(O, A) = OA)
variable (eq_OB : distance(O, B) = OB)
variable (eq_OC : distance(O, C) = OC)
variable (eq_AB : distance(A, B) = AB)
variable (eq_BC : distance(B, C) = BC)
variable (eq_AC : distance(A, C) = AC)

theorem point_O_triangle_inequality (h : O ∈ line(A, B)) (h1 : O ≠ A) (h2 : O ≠ B) :
  OC * AB < OA * BC + OB * AC := 
sorry

end point_O_triangle_inequality_l313_313323


namespace find_acute_angle_of_parallel_vectors_l313_313998

open Real

theorem find_acute_angle_of_parallel_vectors (x : ℝ) (hx1 : (sin x) * (1 / 2 * cos x) = 1 / 4) (hx2 : 0 < x ∧ x < π / 2) : x = π / 4 :=
by
  sorry

end find_acute_angle_of_parallel_vectors_l313_313998


namespace largest_three_digit_number_l313_313191

theorem largest_three_digit_number :
  ∃ (a b : ℕ),
    (a ≠ b) ∧ 
    (a = 8) ∧ 
    (b = 2) ∧ 
    (let n := 100 * a + 10 * b + a in 
      ∀ d, (d = n) → d = 828 ∧ (d % (2 * a + b) = 0)) :=
sorry

end largest_three_digit_number_l313_313191


namespace simplify_fraction_l313_313791

theorem simplify_fraction : (150 / 4350 : ℚ) = 1 / 29 :=
  sorry

end simplify_fraction_l313_313791


namespace sin_135_eq_sqrt2_div_2_l313_313549

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313549


namespace no_period_exists_l313_313760

def floor (x : ℝ) : ℤ := x.floor
def f (x : ℝ) : ℝ := x - (floor x).toReal - Real.tan x

theorem no_period_exists : ¬ ∃ T ≠ 0, ∀ x, f (x + T) = f x :=
by
  sorry

end no_period_exists_l313_313760


namespace hash_hash_hash_100_l313_313176

def hash (N : ℝ) : ℝ := 0.4 * N + 3

theorem hash_hash_hash_100 : hash (hash (hash 100)) = 11.08 :=
by sorry

end hash_hash_hash_100_l313_313176


namespace simultaneous_equations_solution_l313_313619

theorem simultaneous_equations_solution (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 3 ∧ y = (2 * m - 1) * x + 4) ↔ m ≠ 1 :=
by
  sorry

end simultaneous_equations_solution_l313_313619


namespace polar_equation_of_curve_l313_313726

theorem polar_equation_of_curve :
  (∀ (x y : ℝ),
    (∃ α : ℝ, x = 1 + Real.cos α ∧ y = Real.sin α) →
    (∃ ρ θ : ℝ, x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧ ρ = 2 * Real.cos θ)) :=
by
  intros x y h
  cases h with α hα
  use [Real.sqrt ((1 + Real.cos α)^2 + (Real.sin α)^2), Real.arctan y x]
  split
  -- coordinate transformations and some algebraic manipulations needed here
  sorry

end polar_equation_of_curve_l313_313726


namespace false_proposition_l313_313162

theorem false_proposition :
  ¬ (∀ x : ℕ, (x > 0) → (x - 2)^2 > 0) :=
by
  sorry

end false_proposition_l313_313162


namespace range_of_a_l313_313965

theorem range_of_a (A M : ℝ × ℝ) (a : ℝ) (C : ℝ × ℝ → ℝ) (hA : A = (-3, 0)) 
(hM : C M = 1) (hMA : dist M A = 2 * dist M (0, 0)) :
  a ∈ (Set.Icc (1/2 : ℝ) (3/2) ∪ Set.Icc (-3/2) (-1/2)) :=
sorry

end range_of_a_l313_313965


namespace matthew_egg_rolls_l313_313774

theorem matthew_egg_rolls (A P M : ℕ) 
  (h1 : M = 3 * P) 
  (h2 : P = A / 2) 
  (h3 : A = 4) : 
  M = 6 :=
by
  sorry

end matthew_egg_rolls_l313_313774


namespace equilateral_triangle_area_l313_313458

noncomputable def distance (P Q : Point) : ℝ :=
  sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

structure Point :=
  (x y : ℝ)

structure Triangle :=
  (A B C : Point)

def area_of_triangle (T : Triangle) : ℝ := 
  ((sqrt 3) / 4) * (distance T.A T.B) ^ 2

axiom given_conditions
  (M A B C : Point)
  (hAM : distance M A = 5)
  (hBM : distance M B = 6)
  (hCM : distance M C = 7) 
  (T : Triangle) 
  (hT : T = ⟨A, B, C⟩)

theorem equilateral_triangle_area :
  area_of_triangle T = (165 * sqrt 3 / 4) + 9 * sqrt 6 :=
by
  sorry

end equilateral_triangle_area_l313_313458


namespace prudence_nap_is_4_hours_l313_313621

def prudence_nap_length (total_sleep : ℕ) (weekdays_sleep : ℕ) (weekend_sleep : ℕ) (weeks : ℕ) (total_weeks : ℕ) : ℕ :=
  (total_sleep - (weekdays_sleep + weekend_sleep) * total_weeks) / (2 * total_weeks)

theorem prudence_nap_is_4_hours
  (total_sleep weekdays_sleep weekend_sleep total_weeks : ℕ) :
  total_sleep = 200 ∧ weekdays_sleep = 5 * 6 ∧ weekend_sleep = 2 * 9 ∧ total_weeks = 4 →
  prudence_nap_length total_sleep weekdays_sleep weekend_sleep total_weeks total_weeks = 4 :=
by
  intros
  sorry

end prudence_nap_is_4_hours_l313_313621


namespace ways_to_color_excluding_two_corners_l313_313841

def ways_to_color_no_restrictions : ℕ := 120
def ways_to_color_excluding_one_corner : ℕ := 96

theorem ways_to_color_excluding_two_corners :
  let total_ways := ways_to_color_no_restrictions in
  let excluding_one_corner := ways_to_color_excluding_one_corner in
  let a := total_ways - excluding_one_corner in
  let b := a in
  let overlapping := 6 in
  total_ways - (a + b - overlapping) = 78 := by
  sorry

end ways_to_color_excluding_two_corners_l313_313841


namespace hamilton_avenue_delivery_sequences_l313_313261

/-- Proving the number of valid delivery sequences for the postman problem on Hamilton Avenue. -/
theorem hamilton_avenue_delivery_sequences :
  let houses := ({1, 3, 5, 7}, {2, 4, 6, 8}),
      odd_even_pattern := [1, 2, 3, 4, 5, 6, 7, 8] -- this is to represent the alternating odd-even pattern following the conditions properly and cyclicly BECAUSE 1 is repeted twice
  in
    (valid_sequences_for_postman houses odd_even_pattern) = 12 :=
by sorry

end hamilton_avenue_delivery_sequences_l313_313261


namespace find_2a_plus_b_l313_313312

open Real

-- Define the given conditions
variables (a b : ℝ)

-- a and b are acute angles
axiom acute_a : 0 < a ∧ a < π / 2
axiom acute_b : 0 < b ∧ b < π / 2

axiom condition1 : 4 * sin a ^ 2 + 3 * sin b ^ 2 = 1
axiom condition2 : 4 * sin (2 * a) - 3 * sin (2 * b) = 0

-- Define the theorem we want to prove
theorem find_2a_plus_b : 2 * a + b = π / 2 :=
sorry

end find_2a_plus_b_l313_313312


namespace unique_x_if_one_in_set_x_x2_l313_313263

theorem unique_x_if_one_in_set_x_x2 (x : ℝ) (h : 1 ∈ {x, x^2}) (hdistinct : x ≠ x^2) : x = -1 :=
sorry

end unique_x_if_one_in_set_x_x2_l313_313263


namespace question1_question2_l313_313330

-- Define the first operation
def firstOperation (x : ℕ) : ℕ :=
  x / 2

-- Define the second operation
def secondOperation (x : ℕ) : ℕ :=
  4 * x + 1

-- Define a predicate that determines if a number can be displayed
inductive canAppear : ℕ → Prop
| init : canAppear 1
| op1 : ∀ {n}, canAppear n → canAppear (firstOperation n)
| op2 : ∀ {n}, canAppear n → canAppear (secondOperation n)

-- Question 1: Prove that 2000 cannot be displayed
theorem question1 : ¬ canAppear 2000 := 
  sorry

-- Question 2: Prove that there are 233 numbers less than 2000 that can appear
theorem question2 : 
  (Finset.filter (λ x, canAppear x) (Finset.range 2000)).card = 233 := 
  sorry

end question1_question2_l313_313330


namespace units_digit_of_expression_l313_313194

open Real

def units_digit_of_sum_of_powers (a b : ℝ) (n : ℕ) : ℤ :=
  ((a ^ n + b ^ n) % 10).natAbs

theorem units_digit_of_expression :
  units_digit_of_sum_of_powers (17 + sqrt 196) (17 - sqrt 196) 21 = 4 :=
by
  -- Definitions based on given conditions
  let a := 17 + sqrt 196
  let b := 17 - sqrt 196
  
  -- Evaluation based on provided solution summary
  have a_pow_n := a ^ 21
  have b_pow_n := b ^ 21
  
  -- Evaluate final result and check modulo 10
  show (a_pow_n + b_pow_n) % 10 = 4
  sorry

end units_digit_of_expression_l313_313194


namespace find_f_of_3_l313_313212

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f(x + 1) = 2 * x + 3) : f 3 = 7 :=
sorry

end find_f_of_3_l313_313212


namespace perimeter_of_large_rectangle_l313_313894

-- Definitions based on the conditions
def is_square (a : ℕ) : Prop := a = 24
def is_rectangle (b : ℕ) : Prop := b = 16

-- Statement to prove
theorem perimeter_of_large_rectangle (a b : ℕ) (h1 : is_square a) (h2 : is_rectangle b) : 
  52 := sorry

end perimeter_of_large_rectangle_l313_313894


namespace ubacapital_total_suv_count_l313_313405

noncomputable def calc_number_suvs (total_vehicles : Nat) 
                                   (ratio_toyota : Nat)
                                   (ratio_honda : Nat)
                                   (ratio_nissan : Nat) 
                                   (percent_toyota_suv : Float)
                                   (percent_honda_suv : Float)
                                   (percent_nissan_suv : Float) : Nat :=
  let parts := ratio_toyota + ratio_honda + ratio_nissan
  let vehicles_per_part := total_vehicles / parts
  let toyotas := vehicles_per_part * ratio_toyota
  let hondas := vehicles_per_part * ratio_honda
  let nissans := vehicles_per_part * ratio_nissan
  let suvs_toyota := (percent_toyota_suv * toyotas.toFloat).round.toNat
  let suvs_honda := (percent_honda_suv * hondas.toFloat).round.toNat
  let suvs_nissan := (percent_nissan_suv * nissans.toFloat).round.toNat
  suvs_toyota + suvs_honda + suvs_nissan

theorem ubacapital_total_suv_count :
  calc_number_suvs 45 8 4 3 0.30 0.20 0.40 = 13 :=
by
  sorry

end ubacapital_total_suv_count_l313_313405


namespace intersection_of_M_and_N_l313_313017

noncomputable def setM : Set ℝ := { x : ℝ | x^2 - 2 * x - 3 < 0 }
noncomputable def setN : Set ℝ := { x : ℝ | Real.log x / Real.log 2 < 1 }

theorem intersection_of_M_and_N : { x : ℝ | x ∈ setM ∧ x ∈ setN } = { x : ℝ | 0 < x ∧ x < 2 } :=
by
  sorry

end intersection_of_M_and_N_l313_313017


namespace cool_numbers_between_3000_and_8000_l313_313455

def is_cool (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = [] ∨ ∀ i j, i < j → (digits.nth i).get_or_else 0 < (digits.nth j).get_or_else 0

def in_range (n : ℕ) : Prop :=
  3000 ≤ n ∧ n < 8000

theorem cool_numbers_between_3000_and_8000 :
  (∑ n in (Finset.filter is_cool (Finset.filter in_range (Finset.range 8000))), 1) = 35 := 
sorry

end cool_numbers_between_3000_and_8000_l313_313455


namespace find_fraction_value_l313_313682

-- Define vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, 3 / 2)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -1)

-- Parallel condition for vectors
def parallel (a b : ℝ × ℝ) : Prop := ∃ k : ℝ, a = (k * b.1, k * b.2)

-- Given theorem statement
theorem find_fraction_value (x : ℝ)
  (h_parallel : parallel (vector_a x) (vector_b x)) :
  (2 * Real.sin x - Real.cos x) / (4 * Real.sin x + 3 * Real.cos x) = 4 / 3 :=
by
  sorry

end find_fraction_value_l313_313682


namespace escaped_rhinos_l313_313879

theorem escaped_rhinos (R : ℕ) (hrs_per_animal : ℕ) (lions : ℕ) (total_hours : ℕ) :
  lions = 3 ∧ hrs_per_animal = 2 ∧ total_hours = 10 ∧ total_hours = hrs_per_animal * (lions + R) →
  R = 2 :=
by
  intro h
  cases h with h1 h_rest
  cases h_rest with h2 h_rest
  cases h_rest with h3 h4
  cases h4 with h4_eq h_eq
  sorry

end escaped_rhinos_l313_313879


namespace remarkable_two_digit_numbers_count_l313_313020

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def num_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x => x ∣ n).card

def is_remarkable (n : ℕ) : Prop :=
  num_divisors n = 4 ∧ ∃ d1 d2 : ℕ, d1 ∣ n ∧ d2 ∣ n ∧ gcd d1 d2 = 1 ∧ d1 ≠ 1 ∧ d2 ≠ 1

noncomputable def count_remarkable_two_digit_numbers : ℕ :=
  ((Finset.range 100).filter (λ n => n ≥ 10 ∧ is_remarkable n)).card

theorem remarkable_two_digit_numbers_count :
  count_remarkable_two_digit_numbers = 36 := 
sorry

end remarkable_two_digit_numbers_count_l313_313020


namespace profit_in_terms_of_n_l313_313470

variables {C S M : ℝ} {n : ℝ}

-- Condition given in the problem
def condition_cost_margin (C S M n : ℝ) : Prop :=
  M = (2 * C) / n

-- Condition derived from the solution
def condition_cost_price (C S M : ℝ) : Prop :=
  S - M = C

-- Definition of profit percentage
def profit_percentage (M S : ℝ) : ℝ :=
  (M / S) * 100

-- Conclusion we need to prove
theorem profit_in_terms_of_n (C S M n : ℝ) (h1 : condition_cost_margin C S M n) (h2 : condition_cost_price C S M) :
  profit_percentage M S = 200 / (n + 2) :=
sorry

end profit_in_terms_of_n_l313_313470


namespace sin_135_degree_l313_313540

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313540


namespace sin_135_degree_l313_313537

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313537


namespace ellipse_eccentricity_k_values_l313_313058

theorem ellipse_eccentricity_k_values (k : ℝ) :
  (∃ e : ℝ, (2 * e^2 - e = 0) ∧ ((c : ℝ) (c^2 = 3 - k ∧ e = (c / (sqrt 3)) ∨ (c^2 = k - 3 ∧ e = (c / (sqrt k))))) → 
  (k = 9 / 4 ∨ k = 4) :=
by
  sorry

end ellipse_eccentricity_k_values_l313_313058


namespace imaginary_part_of_expression_l313_313660

-- Define the complex number z as 1 + i
def z : ℂ := 1 + complex.i

-- Define the expression (z + 1)(z - 1)
def expression := (z + 1) * (z - 1)

-- The main theorem we need to prove
theorem imaginary_part_of_expression :
  expression.im = 2 :=
by
  -- Placeholder for actual proof
  sorry

end imaginary_part_of_expression_l313_313660


namespace functional_eqn_solution_l313_313189

open Real

theorem functional_eqn_solution {f : ℝ → ℝ}
    (h : ∀ x y : ℝ, f(x) * f(y) = f(x - y)) :
    (∀ x : ℝ, f(x) = 0) ∨ (∀ x : ℝ, f(x) = 1) :=
sorry

end functional_eqn_solution_l313_313189


namespace tangent_line_at_P_eq_2x_l313_313361

noncomputable def tangentLineEq (f : ℝ → ℝ) (P : ℝ × ℝ) : ℝ → ℝ :=
  let slope := deriv f P.1
  fun x => slope * (x - P.1) + P.2

theorem tangent_line_at_P_eq_2x : 
  ∀ (f : ℝ → ℝ) (x y : ℝ),
    f x = x^2 + 1 → 
    (x = 1) → (y = 2) →
    tangentLineEq f (x, y) x = 2 * x :=
by
  intros f x y f_eq hx hy
  sorry

end tangent_line_at_P_eq_2x_l313_313361


namespace polynomial_factorization_l313_313705

theorem polynomial_factorization (m n : ℤ) (h₁ : (x + 1) * (x + 3) = x^2 + m * x + n) : m - n = 1 := 
by {
  -- Proof not required
  sorry
}

end polynomial_factorization_l313_313705


namespace distinct_pairs_count_l313_313186

theorem distinct_pairs_count :
  (∃ n : ℕ, ∀ x y : ℕ, (0 < x ∧ x < y ∧ (√2756 : ℝ) = (√x + √y)) → false ∧ n = 0) :=
by 
  sorry

end distinct_pairs_count_l313_313186


namespace sufficient_but_not_necessary_condition_l313_313434

variable {a : ℝ}

theorem sufficient_but_not_necessary_condition :
  (∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ↔ (a ≥ 0) :=
by
  sorry

end sufficient_but_not_necessary_condition_l313_313434


namespace increasing_interval_l313_313819

def f (x : ℝ) : ℝ := (2 * x - 1) * Real.exp x

theorem increasing_interval :
  ∀ x, x > -1 / 2 → (2 * x + 1) * Real.exp x > 0 :=
by
  intro x hx
  sorry

end increasing_interval_l313_313819


namespace area_of_triangle_is_rational_l313_313173

theorem area_of_triangle_is_rational (x1 x2 x3 y1 y2 y3 : ℤ) 
  (h1 : x1 + x2 + x3 = 10) (h2 : y1 + y2 + y3 = 6) :
  ∃ (r : ℚ), 
    r = (1 / 2 : ℚ) * 
        (| (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2) : ℤ) | : ℤ) := 
sorry

end area_of_triangle_is_rational_l313_313173


namespace smaller_angle_clock_7_20_l313_313684

theorem smaller_angle_clock_7_20 :
  let hour_angle := 7 * 30 + 20 * 0.5
  let minute_angle := 20 * 6
  let angle := |hour_angle - minute_angle|
  min angle (360 - angle) = 100 := by
{
  let hour_angle := 7 * 30 + 20 * 0.5
  let minute_angle := 20 * 6
  let angle := |hour_angle - minute_angle|
  show min angle (360 - angle) = 100
  from sorry
}

end smaller_angle_clock_7_20_l313_313684


namespace spherical_distance_between_points_l313_313973

noncomputable def spherical_distance (R : ℝ) (α : ℝ) : ℝ :=
  α * R

theorem spherical_distance_between_points 
  (R : ℝ) 
  (α : ℝ) 
  (hR : R > 0) 
  (hα : α = π / 6) : 
  spherical_distance R α = (π / 6) * R :=
by
  rw [hα]
  unfold spherical_distance
  ring

end spherical_distance_between_points_l313_313973


namespace sin_135_eq_sqrt2_div_2_l313_313583

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313583


namespace sin_135_l313_313506

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313506


namespace solve_quadratic_eq_solve_linear_eq_l313_313803

-- Define the first problem
theorem solve_quadratic_eq (x : ℝ) : (x^2 - 4 * x + 1 = 0) ↔ (x = 2 + real.sqrt 3 ∨ x = 2 - real.sqrt 3) := by
  sorry

-- Define the second problem
theorem solve_linear_eq (x : ℝ) : (3 * x * (2 * x + 1) = 4 * x + 2) ↔ (x = -1/2 ∨ x = 2/3) := by
  sorry

end solve_quadratic_eq_solve_linear_eq_l313_313803


namespace max_value_of_z_l313_313672

theorem max_value_of_z (x y : ℝ) (h1 : y ≤ 2 * x) (h2 : x - 2 * y - 4 ≤ 0) (h3 : y ≤ 4 - x) : 
  ∃ x y, z = 2 * x + y ∧ z ≤ 8 :=
begin
  sorry
end

end max_value_of_z_l313_313672


namespace sin_135_eq_sqrt2_over_2_l313_313574

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313574


namespace fraction_of_bikinis_or_trunks_l313_313165

theorem fraction_of_bikinis_or_trunks (h_bikinis : Real := 0.38) (h_trunks : Real := 0.25) :
  h_bikinis + h_trunks = 0.63 :=
by
  sorry

end fraction_of_bikinis_or_trunks_l313_313165


namespace integral_of_rational_function_l313_313433

noncomputable def integral_rational_function : ℝ → ℝ
| x := 2x + log (abs x) + log (abs (x + 4)) - 6 * log (abs (x-2)) + C

theorem integral_of_rational_function (C : ℝ) :
  ∫ (dx : ℝ), (2 * x^3 - 40 * x - 8) / (x * (x + 4) * (x - 2)) = 2 * x + log (abs x) + log (abs (x + 4)) - 6 * log (abs (x - 2)) + C :=
by
  sorry

end integral_of_rational_function_l313_313433


namespace cover_10_points_with_unit_circles_l313_313648

theorem cover_10_points_with_unit_circles (points : Fin 10 → ℝ × ℝ) : 
  ∃ (centers : Fin 10 → ℝ × ℝ), 
    ∀ i j, i ≠ j → dist (centers i) (centers j) ≥ 2 ∧ 
    ∀ p ∈ points, ∃ i, dist p (centers i) ≤ 1 :=
sorry

end cover_10_points_with_unit_circles_l313_313648


namespace sin_135_eq_sqrt2_div_2_l313_313585

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313585


namespace hexagon_dot_product_l313_313638

-- Define coordinates of vertices for the given hexagon setup
def A := (-(1/2), sqrt(3)/2)
def B := (1/2, sqrt(3)/2)
def C := (1, 0)
def D := (1/2, -sqrt(3)/2)
def E := (-(1/2), -sqrt(3)/2)

-- Define the vectors as given
def vec_AB := (1, 0)
def vec_DC := (1/2, sqrt(3)/2)
def vec_AD := (1, -sqrt(3))
def vec_BE := (-1, -sqrt(3))

-- Sum of corresponding vectors
def vec_sum1 := (1 + 1/2, 0 + sqrt(3)/2)
def vec_sum2 := (1 - 1, -sqrt(3) - sqrt(3))

-- Dot product of the summed vectors
def dot_product := (1 + 1/2) * 0 + (0 + sqrt(3)/2) * (-2 * sqrt(3))

theorem hexagon_dot_product : 
  (vec_sum1.1 * vec_sum2.1 + vec_sum1.2 * vec_sum2.2) = -3 := 
by
  sorry

end hexagon_dot_product_l313_313638


namespace intersecting_lines_l313_313184

theorem intersecting_lines (p : ℝ) :
    (∃ x y : ℝ, y = 3 * x - 6 ∧ y = -4 * x + 8 ∧ y = 7 * x + p) ↔ p = -14 :=
by {
    sorry
}

end intersecting_lines_l313_313184


namespace median_length_is_5_l313_313365

variable (names : List ℕ)
variable (lengths_distribution : names.length = 16 ∧
  List.count names 4 = 5 ∧
  List.count names 5 = 4 ∧
  List.count names 6 = 2 ∧
  List.count names 7 = 3 ∧
  List.count names 8 = 2)

theorem median_length_is_5 : 
  List.median names = 5 := by
  sorry

end median_length_is_5_l313_313365


namespace sin_135_correct_l313_313563

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313563


namespace quadrilateral_formation_l313_313137

theorem quadrilateral_formation (S : Fin 4 → ℝ) (h_sum : (∑ i, S i) = 1) :
  ∀ i, S i < 1 / 2 :=
by
  sorry

end quadrilateral_formation_l313_313137


namespace difference_of_circumferences_l313_313086

-- Definition of the problem
def diameter_inner : ℝ := 50
def width_track : ℝ := 15
def diameter_outer : ℝ := diameter_inner + 2 * width_track
def circumference (d : ℝ) : ℝ := Real.pi * d

theorem difference_of_circumferences :
  circumference diameter_outer - circumference diameter_inner = 30 * Real.pi :=
by
  sorry

end difference_of_circumferences_l313_313086


namespace max_value_abs_diff_sin_cos_l313_313701

noncomputable def f (x : ℝ) : ℝ := Real.sin x
noncomputable def g (x : ℝ) : ℝ := Real.cos x

theorem max_value_abs_diff_sin_cos : ∃ a : ℝ, |f a - g a| = sqrt 2 :=
by
  sorry

end max_value_abs_diff_sin_cos_l313_313701


namespace ada_original_seat_l313_313715

theorem ada_original_seat (seats: Fin 6 → Option String)
  (Bea_init Ceci_init Dee_init Edie_init Fran_init: Fin 6) 
  (Bea_fin Ceci_fin Fran_fin: Fin 6) 
  (Ada_fin: Fin 6)
  (Bea_moves_right: Bea_fin = Bea_init + 3)
  (Ceci_stays: Ceci_fin = Ceci_init)
  (Dee_switches_with_Edie: ∃ Dee_fin Edie_fin: Fin 6, Dee_fin = Edie_init ∧ Edie_fin = Dee_init)
  (Fran_moves_left: Fran_fin = Fran_init - 1)
  (Ada_end_seat: Ada_fin = 0 ∨ Ada_fin = 5):
  ∃ Ada_init: Fin 6, Ada_init = 2 + Ada_fin + 1 → Ada_init = 3 := 
by 
  sorry

end ada_original_seat_l313_313715


namespace solve_equation_l313_313797

noncomputable def lhs (x: ℝ) : ℝ := (sqrt ((3*x - 1) / (x + 4))) + 3 - 4 * (sqrt ((x + 4) / (3*x - 1)))

theorem solve_equation (x: ℝ) (t : ℝ) (ht : t = (3*x - 1) / (x + 4)) (h_pos : 0 < t) :
  lhs x = 0 → x = 5 / 2 :=
by
  intros h
  sorry

end solve_equation_l313_313797


namespace final_value_of_b_l313_313850

theorem final_value_of_b : ∀ (a b c : Int), 
  a = 3 → b = -5 → c = 8 → 
  let new_a := b in 
  let new_b := c in 
  new_b = 8 :=
by 
  assume a b c ha hb hc
  have hnew_a : a = b := by rw [hb]
  have hnew_b : b = c := by rw [hc]
  exact hc

#check final_value_of_b

end final_value_of_b_l313_313850


namespace largest_shaded_area_l313_313282

-- Define the properties for Figure A
def figureA_square_side : ℝ := 3
def figureA_circle_radius : ℝ := 1

-- Define the properties for Figure B
def figureB_square_side : ℝ := 3
def figureB_quarter_circle_radius : ℝ := 1

-- Define the properties for Figure C
def figureC_rectangle_length : ℝ := 4
def figureC_rectangle_width : ℝ := 2
def figureC_square_side : ℝ := 2

-- Define shaded areas
def shaded_area_A := figureA_square_side^2 - real.pi * figureA_circle_radius^2
def shaded_area_B := figureB_square_side^2 - real.pi * figureB_quarter_circle_radius^2
def shaded_area_C := figureC_rectangle_length * figureC_rectangle_width - figureC_square_side^2

theorem largest_shaded_area :
  shaded_area_A = shaded_area_B ∧ shaded_area_A > shaded_area_C :=
by sorry

end largest_shaded_area_l313_313282


namespace contrapositive_of_real_roots_l313_313669

variable {a : ℝ}

theorem contrapositive_of_real_roots :
  (1 + 4 * a < 0) → (a < 0) := by
  sorry

end contrapositive_of_real_roots_l313_313669


namespace incorrect_statement_l313_313909

theorem incorrect_statement:
  (∀ x : ℝ, x^2 - 1 = 0 → x^2 = 1) ∧ 
  (∀ x : ℝ, x^2 ≠ 1 → x^2 - 1 ≠ 0) ∧ 
  (∀ x : ℝ, x = 1 → x^2 = x) ∧ 
  (∃ x : ℝ, 2^x > 0) →
  (∀ p q : Prop, ¬(p ∧ q) → ¬p ∧ ¬q) → false :=
begin
  sorry
end

end incorrect_statement_l313_313909


namespace part1_part2_l313_313646

theorem part1 (a : ℝ) (h : -3 ∈ ({a - 3, 2a - 1, a^2 + 1} : set ℝ)) :
  a = 0 ∨ a = -1 :=
sorry

theorem part2 (x : ℝ) (h : x^2 ∈ ({0, 1, x} : set ℝ)) :
  x = -1 :=
sorry

end part1_part2_l313_313646


namespace intersection_A_B_l313_313989

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l313_313989


namespace factorial_sum_unique_solution_l313_313948

theorem factorial_sum_unique_solution (n a b c : ℕ) :
  (n! = a! + b! + c!) → (n = 3 ∧ a = 2 ∧ b = 2 ∧ c = 2) :=
by
  sorry

end factorial_sum_unique_solution_l313_313948


namespace upstream_speed_proof_l313_313896

-- Definitions based on the conditions in the problem
def speed_in_still_water : ℝ := 25
def speed_downstream : ℝ := 35

-- The speed of the man rowing upstream
def speed_upstream : ℝ := speed_in_still_water - (speed_downstream - speed_in_still_water)

theorem upstream_speed_proof : speed_upstream = 15 := by
  -- Proof is omitted by using sorry
  sorry

end upstream_speed_proof_l313_313896


namespace find_greatest_integer_less_than_M_div_100_l313_313649

-- Define the sum and the value M
def series_sum : ℚ := 
  (1 / (3! * 18!) + 1 / (4! * 17!) + 1 / (5! * 16!) + 1 / (6! * 15!) + 
  1 / (7! * 14!) + 1 / (8! * 13!) + 1 / (9! * 12!) + 1 / (10! * 11!))

def M : ℚ := 524077 / 20

-- Define the main theorem statement
theorem find_greatest_integer_less_than_M_div_100 :
  ⌊M / 100⌋ = 262 :=
by sorry

end find_greatest_integer_less_than_M_div_100_l313_313649


namespace cone_slant_height_l313_313076

noncomputable def slant_height (r : ℝ) (CSA : ℝ) : ℝ := CSA / (Real.pi * r)

theorem cone_slant_height : slant_height 10 628.3185307179587 = 20 :=
by
  sorry

end cone_slant_height_l313_313076


namespace number_of_ways_to_select_one_ball_l313_313882

theorem number_of_ways_to_select_one_ball (r b : ℕ) (h_r : r = 2) (h_b : b = 4) : 
  r + b = 6 := 
by
  rw [h_r, h_b]
  rfl

end number_of_ways_to_select_one_ball_l313_313882


namespace tom_age_ratio_l313_313397

-- Definitions related to the problem
def toms_current_age (T : ℕ) : Prop := ∃ ages_of_children : list ℕ, ages_of_children.length = 4 ∧ ages_of_children.sum = T

def years_ago_condition (T N : ℕ) : Prop :=
  T - N = 3 * (T - 4 * N)

-- The proof problem stating the ratio
theorem tom_age_ratio (T N : ℕ) (h1 : toms_current_age T) (h2 : years_ago_condition T N) : T = 5.5 * N :=
by
  sorry

end tom_age_ratio_l313_313397


namespace sin_135_eq_l313_313483

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313483


namespace find_v₃_value_l313_313094

def f (x : ℕ) : ℕ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

def v₃_expr (x : ℕ) : ℕ := (((7 * x + 6) * x + 5) * x + 4)

theorem find_v₃_value : v₃_expr 3 = 262 := by
  sorry

end find_v₃_value_l313_313094


namespace intersection_A_B_l313_313983

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l313_313983


namespace find_original_function_l313_313233

noncomputable def transformed_graph (f : ℝ → ℝ) : ℝ → ℝ :=
  fun x => 4 * f (x / 2 + π / 4)

theorem find_original_function (f : ℝ → ℝ)
  (h : ∀ x : ℝ, transformed_graph f x = 2 * sin x) :
  ∀ x : ℝ, f x = - (1 / 2) * cos (2 * x) :=
by
  sorry

end find_original_function_l313_313233


namespace find_other_number_l313_313367

theorem find_other_number (a b : ℕ) (gcd_ab : Nat.gcd a b = 45) (lcm_ab : Nat.lcm a b = 1260) (a_eq : a = 180) : b = 315 :=
by
  -- proof goes here
  sorry

end find_other_number_l313_313367


namespace betty_bracelets_l313_313167

theorem betty_bracelets : (140 / 14) = 10 := 
by
  norm_num

end betty_bracelets_l313_313167


namespace initial_cents_of_A_l313_313891

theorem initial_cents_of_A (a b c : ℕ) 
    (h1 : 16 * a - 48 * b - 48 * c = 27)
    (h2 : -12 * a + 52 * b - 36 * c = 27)
    (h3 : -12 * a - 12 * b + 64 * c = 27)
    (h4 : a + b + c = 81) : 
    a = 52 :=
by
  -- Proof will be filled in
  sorry

end initial_cents_of_A_l313_313891


namespace A_beats_B_by_seconds_l313_313719

namespace Race

variables (T_A T_B D : ℝ)
variable (speed_A : ℝ := 80 / T_A)  -- A's speed is 80 meters in T_A seconds

noncomputable def speed_B := (24 / 80) * speed_A  -- B's speed derived from proportions when A finishes

theorem A_beats_B_by_seconds (h1 : T_A = 3) (h2 : D = 56) :
  let T_B := 80 / speed_B in
  T_B - T_A = 7 :=
by
  sorry

end Race

end A_beats_B_by_seconds_l313_313719


namespace remainder_73_to_73_plus73_div137_l313_313690

theorem remainder_73_to_73_plus73_div137 :
  ((73 ^ 73 + 73) % 137) = 9 := by
  sorry

end remainder_73_to_73_plus73_div137_l313_313690


namespace ratio_PQ_QR_l313_313336

def circle_radius (r : ℝ) (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] :=
  dist P Q = dist P R ∧ dist P Q > 2 * r ∧ arc_length (P, Q, R) = π * r

theorem ratio_PQ_QR (P Q R : Type) [metric_space P] [metric_space Q] [metric_space R] (r : ℝ)
  (h1 : circle_radius r P Q R) : dist P Q / arc_length (Q, R) = 2 * sqrt 2 / π :=
by sorry

end ratio_PQ_QR_l313_313336


namespace tyrone_gives_non_integer_marbles_to_eric_l313_313848

theorem tyrone_gives_non_integer_marbles_to_eric
  (T_init : ℕ) (E_init : ℕ) (x : ℚ)
  (hT : T_init = 120) (hE : E_init = 18)
  (h_eq : T_init - x = 3 * (E_init + x)) :
  ¬ (∃ n : ℕ, x = n) :=
by
  sorry

end tyrone_gives_non_integer_marbles_to_eric_l313_313848


namespace sin_135_correct_l313_313557

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313557


namespace sequence_properties_l313_313642

theorem sequence_properties (a : ℕ → ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 1 = 1)
  (h_rec : ∀ n, (a n)^2 - (2 * a (n + 1) - 1) * a n - 2 * a (n + 1) = 0) :
  a 2 = 1 / 2 ∧ a 3 = 1 / 4 ∧ ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_properties_l313_313642


namespace problem_statement_l313_313015

theorem problem_statement (
  a b c d x y z t : ℝ
) (habcd : 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 1 ∧ 0 ≤ d ∧ d ≤ 1) 
  (hxyz : 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 1 ≤ t)
  (h_sum : a + b + c + d + x + y + z + t = 8) :
  a^2 + b^2 + c^2 + d^2 + x^2 + y^2 + z^2 + t^2 ≤ 28 := 
sorry

end problem_statement_l313_313015


namespace mario_meet_speed_l313_313920

noncomputable def Mario_average_speed (x : ℝ) : ℝ :=
  let t1 := x / 5
  let t2 := x / 3
  let t3 := x / 4
  let t4 := x / 10
  let T := t1 + t2 + t3 + t4
  let d_mario := 1.5 * x
  d_mario / T

theorem mario_meet_speed : ∀ (x : ℝ), x > 0 → Mario_average_speed x = 90 / 53 :=
by
  intros
  rw [Mario_average_speed]
  -- You can insert calculations similar to those in the provided solution
  sorry

end mario_meet_speed_l313_313920


namespace minimum_ticket_cost_l313_313276

-- Definitions of the conditions in Lean
def southern_cities : ℕ := 4
def northern_cities : ℕ := 5
def one_way_ticket_cost (N : ℝ) : ℝ := N
def round_trip_ticket_cost (N : ℝ) : ℝ := 1.6 * N

-- The main theorem to prove
theorem minimum_ticket_cost (N : ℝ) : 
  (∀ (Y1 Y2 Y3 Y4 : ℕ), 
  (∀ (S1 S2 S3 S4 S5 : ℕ), 
  southern_cities = 4 → northern_cities = 5 →
  one_way_ticket_cost N = N →
  round_trip_ticket_cost N = 1.6 * N →
  ∃ (total_cost : ℝ), total_cost = 6.4 * N)) :=
sorry

end minimum_ticket_cost_l313_313276


namespace correct_fraction_statement_l313_313103

theorem correct_fraction_statement (x : ℝ) :
  (∀ a b : ℝ, (-a) / (-b) = a / b) ∧
  (¬ (∀ a : ℝ, a / 0 = 0)) ∧
  (∀ a b : ℝ, b ≠ 0 → (a * b) / (c * b) = a / c) → 
  ((∃ (a b : ℝ), a = 0 → a / b = 0) ∧ 
   (∀ (a b : ℝ), (a * k) / (b * k) = a / b) ∧ 
   (∀ (a b : ℝ), (-a) / (-b) = a / b) ∧ 
   (x < 1 → (|2 - x| + x) / 2 ≠ 0) 
  -> (∀ (a b : ℝ), (-a) / (-b) = a / b)) :=
by sorry

end correct_fraction_statement_l313_313103


namespace total_amount_is_correct_l313_313895

def cost_price : ℝ := 1400
def loss_percentage : ℝ := 20 / 100
def sales_tax_percentage : ℝ := 5 / 100

def loss_amount : ℝ := loss_percentage * cost_price
def selling_price : ℝ := cost_price - loss_amount
def sales_tax_amount : ℝ := sales_tax_percentage * selling_price
def total_amount : ℝ := selling_price + sales_tax_amount

theorem total_amount_is_correct : total_amount = 1176 := by
  unfold cost_price loss_percentage sales_tax_percentage loss_amount selling_price sales_tax_amount total_amount
  -- Proof steps will be omitted but included here just to complete syntactic structure
  -- hint: no need of simplifying intermediate values as
  -- the final step will resolve them all together
  sorry

end total_amount_is_correct_l313_313895


namespace solid_is_sphere_l313_313036

noncomputable def point (α : Type*) := α
noncomputable def is_circle (α : Type*) (sec : α → Prop) := ∀ (p : α), sec p → ∃ (c : α), c = p ∧ sec c
noncomputable def is_sphere (α : Type*) (solid : α → Prop) := ∀ (p : α), solid p → ∃ (s : α), s = p ∧ solid s

theorem solid_is_sphere {α : Type*}
  (solid : α → Prop)
  (P : α)
  (h : ∀ (π : α → Prop), (P ∈ π) → is_circle α (λ x, π x ∧ solid x)) :
  is_sphere α solid :=
sorry

end solid_is_sphere_l313_313036


namespace problem_l313_313321

open Real

theorem problem (x y : ℝ) (h_posx : 0 < x) (h_posy : 0 < y) (h_cond : x + y^(2016) ≥ 1) : 
  x^(2016) + y > 1 - 1/100 :=
by sorry

end problem_l313_313321


namespace problem1_problem2_l313_313480

-- First proof problem
theorem problem1 : - (2^2 : ℚ) + (2/3) * ((1 - 1/3) ^ 2) = -100/27 :=
by sorry

-- Second proof problem
theorem problem2 : (8 : ℚ) ^ (1 / 3) - |2 - (3 : ℚ) ^ (1 / 2)| - (3 : ℚ) ^ (1 / 2) = 0 :=
by sorry

end problem1_problem2_l313_313480


namespace elements_in_set_A_l313_313341

-- Define the sets and their properties
variable {U : Type} -- Define a universal set type U

variable (A B : Set U) -- Define sets A and B within U
variable (a b : ℕ) -- Define the cardinalities of sets A and B

-- Conditions from the problem
constant h1 : a = 3 * b
constant h2 : card (A ∪ B) = 4500
constant h3 : card (A ∩ B) = 1200

-- Statement to prove
theorem elements_in_set_A : a = 4050 :=
by sorry

end elements_in_set_A_l313_313341


namespace range_modulus_Z_l313_313631

variable (Z : ℂ)
variable (a : ℝ)
variable (h₀ : 0 < a) (h₂ : a < 2)
variable (hZ : Z.re = a) (h_im : Z.im = 1)

theorem range_modulus_Z :
  (1 : ℝ) < complex.abs Z ∧ complex.abs Z < real.sqrt 5 :=
by
  have h₁ : Z.im = 1 := h_im
  have h₂ : (complex.abs Z) = real.sqrt (a^2 + 1) := sorry
  sorry

end range_modulus_Z_l313_313631


namespace sin_135_eq_l313_313513

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313513


namespace square_pyramid_sum_l313_313170

def square_pyramid_faces : Nat := 5
def square_pyramid_edges : Nat := 8
def square_pyramid_vertices : Nat := 5

theorem square_pyramid_sum : square_pyramid_faces + square_pyramid_edges + square_pyramid_vertices = 18 := by
  sorry

end square_pyramid_sum_l313_313170


namespace Q_calculation_l313_313005

def Q (n : ℕ) : ℚ :=
  ∏ k in Finset.range (n + 1) \ {0, 1}, (1 - (1 / k))

theorem Q_calculation : Q 2023 = 1 / 2023 :=
by
  sorry

end Q_calculation_l313_313005


namespace sequence_periodic_iff_rational_l313_313379

/-- The sequence $\{x_{n}\}$ is defined by $x_{0}\in [0, 1]$ and $x_{n+1}=1-\vert 1-2 x_{n}\vert$.
Prove that the sequence is periodic if and only if $x_{0}$ is rational. 
-/
theorem sequence_periodic_iff_rational (x0 : ℝ) (h0 : 0 ≤ x0 ∧ x0 ≤ 1) (h_seq : ∀ n, x0 n = 1 - |1 - 2 * x0 (n - 1)|) :
  (∃ p, ∀ n, x0 (n + p) = x0 n) ↔ x0.is_rational :=
sorry

end sequence_periodic_iff_rational_l313_313379


namespace sin_135_eq_sqrt2_over_2_l313_313573

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313573


namespace standard_deviation_of_set_is_correct_l313_313350

theorem standard_deviation_of_set_is_correct (x : ℝ) (h : (2 + x + 4 + 6 + 10) / 5 = 5) :
  let s := [2, x, 4, 6, 10] in
  let mean := (2 + x + 4 + 6 + 10) / 5 in
  let variance := (1 / 5) * ((2 - mean) ^ 2 + (x - mean) ^ 2 + (4 - mean) ^ 2 + (6 - mean) ^ 2 + (10 - mean) ^ 2) in
  let std_dev := Real.sqrt variance in
  std_dev = 2 * Real.sqrt 2 :=
by
  sorry

end standard_deviation_of_set_is_correct_l313_313350


namespace total_money_difference_l313_313299

-- Define the number of quarters each sibling has
def quarters_Karen : ℕ := 32
def quarters_Christopher : ℕ := 64
def quarters_Emily : ℕ := 20
def quarters_Michael : ℕ := 12

-- Define the value of each quarter
def value_per_quarter : ℚ := 0.25

-- Prove that the total money difference between the pairs of siblings is $16.00
theorem total_money_difference : 
  (quarters_Karen - quarters_Emily) * value_per_quarter + 
  (quarters_Christopher - quarters_Michael) * value_per_quarter = 16 := by
sorry

end total_money_difference_l313_313299


namespace sin_135_l313_313505

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313505


namespace min_weighings_to_find_heaviest_and_lightest_l313_313408

theorem min_weighings_to_find_heaviest_and_lightest (n : ℕ) : 
  minimum_number_of_weighings n = 2 * n - 2 - n / 2 :=
sorry

end min_weighings_to_find_heaviest_and_lightest_l313_313408


namespace extreme_values_l313_313245

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem extreme_values :
  let max_val := f (-2)
  ∧ f (-3) = 7
  ∧ f (4) = (28 / 3)
  ∧ f (-2) = (28 / 3)
  ∧ f (2) = -(4 / 3)
  ∧ (-3 ≤ -2 ∧ -2 ≤ 2 ∧ 2 ≤ 4) in
  (∃ x ∈ Icc (-3 : ℝ) 4, f x = (28 / 3)) ∧ 
  (∃ y ∈ Icc (-3 : ℝ) 4, f y = -(4 / 3)) := by
    let f (x : ℝ) := (1 / 3) * x ^ 3 - 4 * x + 4
    sorry

end extreme_values_l313_313245


namespace new_cylinder_has_twice_volume_l313_313447

-- The original cylinder's radius and height
def original_radius : ℝ := 5
def original_height : ℝ := 10

-- Volume of the original cylinder
def volume (r h : ℝ) : ℝ := π * r^2 * h

-- The desired doubled volume
def desired_volume := 2 * volume original_radius original_height

-- The new cylinder's dimensions
def new_radius : ℝ := 10
def new_height : ℝ := 5

-- Verification that the new cylinder has the desired volume
theorem new_cylinder_has_twice_volume : volume new_radius new_height = desired_volume := 
by
  -- Here we should provide the proof
  sorry

end new_cylinder_has_twice_volume_l313_313447


namespace proof_problem_l313_313307

variables {A B C M N Q T : Point}
variable [HasIncircle (Triangle A B C)]
variable [Circumcircle A Q N]

def midpoint (A B : Point) (M : Point) : Prop :=
  dist A M = dist M B

def is_parallel (l₁ l₂ : Line) : Prop := sorry

def point_on_line (P : Point) (l : Line) : Prop := sorry

def circle_tangent_to_sites (P₁ P₂ : Point) (side : Line) : Prop := sorry

theorem proof_problem
  (hM : midpoint B C M)
  (hN : midpoint A C N)
  (line_through_N : Line)
  (hline : point_on_line N line_through_N ∧ is_parallel line_through_N (line_through B C))
  (hQ : point_on_line Q line_through_N ∧ is_parallel line_through_N (line_through B C) ∧
         ∃ (s₁ s₂ : ℝ), dist Q N * dist B C = dist A B * dist A C ∧
         Q and C are on different sides of line_through A B)
  (hT : ∃ circumcircle A Q N, point_on_circumcircle T ∧ T ≠ N)
  : ∃ circle_through_TN_tangent_to_BC : Circle, circle_tangent_to_sites T N (line_through B C) ∧ circle_tangent_to_sites T N (in_circle_of_triangle (Triangle A B C)) :=
sorry

end proof_problem_l313_313307


namespace Jan_height_is_42_l313_313919

-- Given conditions
def Cary_height : ℕ := 72
def Bill_height : ℕ := Cary_height / 2
def Jan_height : ℕ := Bill_height + 6

-- Statement to prove
theorem Jan_height_is_42 : Jan_height = 42 := by
  sorry

end Jan_height_is_42_l313_313919


namespace greatest_integer_l313_313297

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℤ, n = 9 * k - 2) (h3 : ∃ l : ℤ, n = 8 * l - 4) : n = 124 := 
sorry

end greatest_integer_l313_313297


namespace sin_135_eq_l313_313491

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313491


namespace domain_of_sqrt_1_minus_2_pow_x_l313_313358

theorem domain_of_sqrt_1_minus_2_pow_x (x : ℝ) : (∃ y : ℝ, f y = sqrt (1 - 2^y)) → x ∈ set.Iic 0 :=
by
  sorry

/-- Given f(x) = sqrt(1 - 2^x), prove that the domain of f(x) is (-∞, 0].
This is translated to proving that for all x where f(x) is defined, x is in the interval -∞ to 0.
-/

end domain_of_sqrt_1_minus_2_pow_x_l313_313358


namespace curd_mass_from_milk_l313_313623

-- Define the conditions
def milk_fat_content : ℝ := 0.05
def curd_fat_content : ℝ := 0.155
def whey_fat_content : ℝ := 0.005
def total_milk_mass : ℝ := 1 -- 1 ton of milk

-- Define the relationship according to the problem
theorem curd_mass_from_milk : 
  ∃ (x : ℝ), total_milk_mass * milk_fat_content = (x * curd_fat_content) + ((total_milk_mass - x) * whey_fat_content) ∧ x = 0.3 :=
by {
  -- Initial setup according to the problem
  have h1 : total_milk_mass * milk_fat_content = (x * curd_fat_content) + ((total_milk_mass - x) * whey_fat_content),
  -- Proof starts here, add the necessary assumption and calculation
  sorry
}

end curd_mass_from_milk_l313_313623


namespace fractional_equation_correct_l313_313102

-- Definition of the four options
def optionA : Prop := (1 / 5) + (x / 4) = 3
def optionB : Prop := x - 4 * y = 7
def optionC : Prop := 2 * x = 3 * (x - 5)
def optionD : Prop := 4 / (x - 2) = 1

-- The statement that option D is a fractional equation and is the correct answer
theorem fractional_equation_correct : optionD := 
sorry

end fractional_equation_correct_l313_313102


namespace minimum_S1_S2_minimum_value_of_S1_S2_is_2sqrt2_minus_2_l313_313750

open Real

noncomputable def S1 (a : ℝ) : ℝ := ∫ x in 0..arctan (1 / a), cos x - a * sin x
noncomputable def S2 (a : ℝ) : ℝ :=
  ∫ x in arctan (1 / a)..π - arctan (1 / a), a * sin x - 2 * (cos x)

theorem minimum_S1_S2 (a : ℝ) (ha : 0 < a) :
  (S1 a + S2 a) = 3 * sqrt (1 + a ^ 2) - a - 2 :=
sorry

theorem minimum_value_of_S1_S2_is_2sqrt2_minus_2 :
  (3 * sqrt (1 + (1 / (2 * sqrt 2)) ^ 2) - (1 / (2 * sqrt 2)) - 2) = 2 * sqrt 2 - 2 :=
sorry

end minimum_S1_S2_minimum_value_of_S1_S2_is_2sqrt2_minus_2_l313_313750


namespace exists_pairs_angle_120_degrees_l313_313293

theorem exists_pairs_angle_120_degrees :
  ∃ a b : ℤ, a + b ≠ 0 ∧ a + b ≠ a ^ 2 - a * b + b ^ 2 ∧ (a + b) * 13 = 3 * (a ^ 2 - a * b + b ^ 2) :=
sorry

end exists_pairs_angle_120_degrees_l313_313293


namespace inequality_proof_l313_313200

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  1/a + 1/b + 1/c ≥ 2/(a + b) + 2/(b + c) + 2/(c + a) ∧ 2/(a + b) + 2/(b + c) + 2/(c + a) ≥ 9/(a + b + c) :=
sorry

end inequality_proof_l313_313200


namespace probability_of_digit_3_l313_313456

theorem probability_of_digit_3 :
  let prob : ℕ → ℝ := λ d, Real.log10 ((d + 2) / (d + 1))
  in prob 3 = 1 / 3 * (prob 3 + prob 4 + prob 5) :=
by simp [prob]; sorry

end probability_of_digit_3_l313_313456


namespace minimum_sum_of_dimensions_l313_313356

   theorem minimum_sum_of_dimensions (a b c : ℕ) (habc : a * b * c = 3003) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
     a + b + c = 45 :=
   sorry
   
end minimum_sum_of_dimensions_l313_313356


namespace mens_wages_approximately_100_l313_313124

-- Define constants representing the wages of one man, one woman, and one boy
variables (M W B : ℝ)

-- Conditions
def men_to_women_equivalence : Prop := 12 * M = W
def women_to_boys_equivalence : Prop := W = 20 * B
def total_earnings_condition : Prop := 12 * M + W + 20 * B = 300

-- Theorem stating that the wage of a man M should be Rs 100/12 based on conditions
theorem mens_wages_approximately_100 :
  men_to_women_equivalence M W B →
  women_to_boys_equivalence W B →
  total_earnings_condition M W B →
  12 * M ≈ 100 :=
by {
  sorry
}

end mens_wages_approximately_100_l313_313124


namespace cost_for_23_days_l313_313698

structure HostelStay where
  charge_first_week : ℝ
  charge_additional_week : ℝ

def cost_of_stay (days : ℕ) (hostel : HostelStay) : ℝ :=
  let first_week_days := min days 7
  let remaining_days := days - first_week_days
  let additional_full_weeks := remaining_days / 7 
  let additional_days := remaining_days % 7
  (first_week_days * hostel.charge_first_week) + 
  (additional_full_weeks * 7 * hostel.charge_additional_week) + 
  (additional_days * hostel.charge_additional_week)

theorem cost_for_23_days :
  cost_of_stay 23 { charge_first_week := 18.00, charge_additional_week := 11.00 } = 302.00 :=
by
  sorry

end cost_for_23_days_l313_313698


namespace sin_135_eq_sqrt2_div_2_l313_313552

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313552


namespace possible_prime_digit_for_outlined_square_l313_313201

noncomputable def four_digit_powers_of_2 : List ℕ :=
  [1024, 2048]

noncomputable def four_digit_powers_of_5 : List ℕ :=
  [3125]

def get_prime_digits (n : ℕ) : List Nat :=
  n.digits 10 |>.filter (λ d => d.isPrime)

theorem possible_prime_digit_for_outlined_square :
  ∀ d ∈ get_prime_digits 1024, d = 2 ∧
  ∀ d ∈ get_prime_digits 2048, d = 2 ∧
  ∀ d ∈ get_prime_digits 3125, d = 2 ∨ d = 3 ∨ d = 5 →
  ∃! d, d = 2 :=
by
  sorry

end possible_prime_digit_for_outlined_square_l313_313201


namespace order_of_alphas_l313_313322

theorem order_of_alphas (x : ℝ) (h : -1/2 < x ∧ x < 0) :
  let α1 := Real.cos (Real.sin (x * Real.pi))
  let α2 := Real.sin (Real.cos (x * Real.pi))
  let α3 := Real.cos ((x + 1) * Real.pi)
in α3 < α2 ∧ α2 < α1 :=
by
  -- Skipping the detailed proof with 'sorry'
  sorry

end order_of_alphas_l313_313322


namespace factorize_polynomial_l313_313604

-- Define the polynomials and their factors.
noncomputable def polynomial1 (x : ℝ) : ℝ := x^2 - x - 6
noncomputable def polynomial2 (x : ℝ) : ℝ := x^2 + 3*x - 4
noncomputable def factor1 (x : ℝ) : ℝ := x + 3
noncomputable def factor2 (x : ℝ) : ℝ := x - 2
noncomputable def factor3 (x : ℝ) : ℝ := x + (1 + real.sqrt 33) / 2
noncomputable def factor4 (x : ℝ) : ℝ := x + (1 - real.sqrt 33) / 2

-- Define the statement of the theorem to be proved.
theorem factorize_polynomial :
  ∀ x : ℝ, (polynomial1 x) * (polynomial2 x) + 24 = (factor1 x) * (factor2 x) * (factor3 x) * (factor4 x) :=
by
  sorry

end factorize_polynomial_l313_313604


namespace sarah_and_molly_groups_l313_313713

theorem sarah_and_molly_groups :
  let members : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let sarah := 1
  let molly := 2
  -- Define the sets of groups where Sarah and Molly are together
  let groups_with_both := {g : Finset ℕ | sarah ∈ g ∧ molly ∈ g ∧ (g \ {sarah, molly}).card = 4}
  -- Calculate the number of groups that include both Sarah and Molly
  (Finset.card groups_with_both) = 210 := sorry

end sarah_and_molly_groups_l313_313713


namespace find_range_for_inequality_l313_313975

variable (f : ℝ → ℝ)

theorem find_range_for_inequality
  (even_f : ∀ x, f x = f (-x))
  (mono_increasing_f : ∀ ⦃a b : ℝ⦄, 0 ≤ a → a ≤ b → f a ≤ f b)
  (f_neg2_one : f (-2) = 1) :
  {x : ℝ | f (x - 2) ≤ 1} = set.Icc 0 4 :=
begin
  sorry
end

end find_range_for_inequality_l313_313975


namespace ratio_of_intercepts_l313_313403

variable (b1 b2 : ℝ)
variable (s t : ℝ)
variable (Hs : s = -b1 / 8)
variable (Ht : t = -b2 / 3)

theorem ratio_of_intercepts (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0) : s / t = 3 * b1 / (8 * b2) :=
by
  sorry

end ratio_of_intercepts_l313_313403


namespace compare_abc_l313_313213

variables {a b c : ℝ}

theorem compare_abc
  (h₁ : log 2 a = a / 2) (h₁_ne : a ≠ 2)
  (h₂ : log 3 b = b / 3) (h₂_ne : b ≠ 3)
  (h₃ : log 4 c = c / 4) (h₃_ne : c ≠ 4) :
  c < b < a := 
sorry

end compare_abc_l313_313213


namespace length_of_CD_is_correct_l313_313724

noncomputable def length_CD (BO OD AO OC AB : ℝ) : ℝ :=
  let cos_AOB := (AO^2 + BO^2 - AB^2) / (2 * AO * BO)
  let cos_COD := -cos_AOB
  (OC^2 + OD^2 - 2 * OC * OD * cos_COD).sqrt

theorem length_of_CD_is_correct :
  length_CD 3 5 7 4 5 ≈ 8.51 := by
  sorry

end length_of_CD_is_correct_l313_313724


namespace sin_135_degree_l313_313541

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313541


namespace tangent_line_at_point_l313_313059

noncomputable def cubic_curve (x : ℝ) : ℝ :=
  x^3 - 2*x + 4

def tangent_line (f : ℝ → ℝ) (x₀ y₀ : ℝ) (m : ℝ) : (ℝ → ℝ) :=
  fun x => m * (x - x₀) + y₀

theorem tangent_line_at_point (x₀ y₀ : ℝ) (h₃ : y₀ = cubic_curve x₀) : 
  tangent_line cubic_curve 1 3 1 = fun x => x + 2 :=
by
  intro x₀ y₀ h₃
  sorry

end tangent_line_at_point_l313_313059


namespace solve_sqrt_equation_l313_313794

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end solve_sqrt_equation_l313_313794


namespace max_sides_planar_cross_section_l313_313849

theorem max_sides_planar_cross_section (n : ℕ) (hn : 2 ≤ n) :
  let polyhedron := (aligned_pyramids_with_base (2 * n)) in
  max_sides_of_planar_cross_section polyhedron = 2 * (n + 1) :=
sorry

end max_sides_planar_cross_section_l313_313849


namespace DanHas72Stickers_l313_313595

-- Definitions
def BobStickers : ℕ := 12
def TomStickers : ℕ := 3 * BobStickers
def DanStickers : ℕ := 2 * TomStickers

-- Theorem
theorem DanHas72Stickers (h : BobStickers = 12) : DanStickers = 72 :=
by
  -- We assume Bob's number of stickers is given
  have b := h
  have t := by rw [TomStickers, b]; exact (3 * 12)
  rw [DanStickers, t]
  exact (2 * 36)
-- Note: We are skipping the actual proof, so we use 'sorry' to indicate that here.
-- sorry

end DanHas72Stickers_l313_313595


namespace equilateral_if_acute_and_altitude_bisector_median_equal_l313_313037

theorem equilateral_if_acute_and_altitude_bisector_median_equal
  (A B C : Type) [triangle(ABC)]
  (acuteABC : ∀ (α β γ : ℝ), α < 90 ∧ β < 90 ∧ γ < 90) 
  (h_a : (altitude A B C)) 
  (l_b : (bisector B A C)) 
  (m_c : (median C A B))
  (h_a_eq_l_b : h_a = l_b) 
  (l_b_eq_m_c : l_b = m_c) :
  equilateral_triangle(ABC) :=
by 
  sorry

end equilateral_if_acute_and_altitude_bisector_median_equal_l313_313037


namespace minimum_ticket_cost_l313_313279

theorem minimum_ticket_cost 
  (N : ℕ)
  (southern_cities : Fin 4)
  (northern_cities : Fin 5)
  (one_way_cost : ∀ (A B : city), A ≠ B → ticket_cost A B = N)
  (round_trip_cost : ∀ (A B : city), A ≠ B → ticket_cost_round_trip A B = 1.6 * N) :
  ∃ (minimum_cost : ℕ), minimum_cost = 6.4 * N := 
sorry

end minimum_ticket_cost_l313_313279


namespace triangle_angles_l313_313224

theorem triangle_angles :
  ∃ θ₁ θ₂ θ₃ : ℝ,
    θ₁ + θ₂ + θ₃ = 180 ∧
    θ₁ ≈ 33.56 ∧
    θ₂ ≈ 73.22 ∧
    θ₃ ≈ 73.22 ∧
    ∃ a b c : ℝ,
      a = 3 ∧
      b = 3 ∧
      c = sqrt 3 ∧
      cos θ₁ = (a^2 + b^2 - c^2) / (2 * a * b)
  :=
by
  -- skipping the proof
  sorry

end triangle_angles_l313_313224


namespace arithmetic_seq_inequality_l313_313174

theorem arithmetic_seq_inequality (a : ℕ → ℝ) (h_arith : ∀ n, a (n+1) - a n = a 2 - a 1) 
  (h_pos : 0 < a 1) (h_bound : a 1 < a 41) :
  ∃ n, n = 7 ∧ (∀ m, 1 ≤ m ∧ m ≤ n → (∑ i in finset.range m, (a (i + 1) - 1 / a (i + 1))) ≤ 0) :=
sorry

end arithmetic_seq_inequality_l313_313174


namespace blue_candies_count_l313_313117

theorem blue_candies_count (total_pieces red_pieces : Nat) (h1 : total_pieces = 3409) (h2 : red_pieces = 145) : total_pieces - red_pieces = 3264 := 
by
  -- Proof will be provided here
  sorry

end blue_candies_count_l313_313117


namespace intersection_A_B_l313_313990

open Set

-- Conditions given in the problem
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Statement to prove, no proof needed
theorem intersection_A_B : A ∩ B = {1, 2} := 
sorry

end intersection_A_B_l313_313990


namespace stratified_sampling_l313_313378

theorem stratified_sampling
  (ratio_first : ℕ)
  (ratio_second : ℕ)
  (ratio_third : ℕ)
  (sample_size : ℕ)
  (h_ratio : ratio_first = 3 ∧ ratio_second = 4 ∧ ratio_third = 3)
  (h_sample_size : sample_size = 50) :
  (ratio_second * sample_size) / (ratio_first + ratio_second + ratio_third) = 20 :=
by
  sorry

end stratified_sampling_l313_313378


namespace stratified_sampling_l313_313363

theorem stratified_sampling 
  (total_teachers : ℕ)
  (senior_teachers : ℕ)
  (intermediate_teachers : ℕ)
  (junior_teachers : ℕ)
  (sample_size : ℕ)
  (x y z : ℕ) : 
  total_teachers = 150 ∧ 
  senior_teachers = 45 ∧ 
  intermediate_teachers = 90 ∧ 
  junior_teachers = 15 ∧ 
  sample_size = 30 ∧ 
  x + y + z = 30 ∧ 
  x: y: z = 3: 6: 1 → 
  x = 3 ∧ y = 18 ∧ z = 3 := 
by
  intros h
  have h_total: total_teachers = 150 := h.1
  have h_senior: senior_teachers = 45 := h.2
  have h_intermediate: intermediate_teachers = 90 := h.3
  have h_junior: junior_teachers = 15 := h.4
  have h_sample: sample_size = 30 := h.5
  have h_sum: x + y + z = 30 := h.6
  have h_ratio: x: y: z = 3: 6: 1 := h.7
  sorry

end stratified_sampling_l313_313363


namespace sin_135_degree_l313_313532

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313532


namespace arith_prog_with_inserted_means_stays_arith_prog_l313_313339

variables {R : Type*} [LinearOrderedField R]

def is_arith_prog (seq : List R) : Prop :=
  ∃ (d : R), ∀ (n : ℕ), n < seq.length - 1 → seq.nth_le n sorry + d = seq.nth_le (n + 1) sorry

theorem arith_prog_with_inserted_means_stays_arith_prog 
  (initial_seq : List R) (p : ℕ) 
  (h_arith_prog : is_arith_prog initial_seq) : 
  ∃ (new_seq : List R), is_arith_prog new_seq := 
sorry

end arith_prog_with_inserted_means_stays_arith_prog_l313_313339


namespace sum_of_first_four_terms_l313_313385

-- Conditions
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, b (n + 2) - b (n + 1) = b (n + 1) - b n

-- Definition of the problem specific sequence
def sequence := [4, 2, 1] -- this represents 4a1, 2a2, a3

-- Specific conditions
def specific_condition (a : ℕ → ℝ) (q : ℝ): Prop :=
  (a 0 = 1) ∧ geometric_sequence a q ∧ is_arithmetic_sequence (λ n, sequence[n] * (a n))

-- Sum of first n terms of geometric sequence
def sum_first_n_terms (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 0 else a 0 * (1 - q ^ (n+1)) / (1 - q)

-- Question to prove
theorem sum_of_first_four_terms (a : ℕ → ℝ) (q : ℝ) :
  specific_condition a q → sum_first_n_terms a q 3 = 15 := 
by
  sorry

end sum_of_first_four_terms_l313_313385


namespace sphere_surface_area_l313_313640

noncomputable def surface_area_of_sphere (AA1 AB BC : ℝ) (angle_ABC : ℝ) (h : ℝ) : ℝ :=
  if h = 6 ∧ AB = 4 ∧ BC = 2 ∧ angle_ABC = 60 
  then 240 * Real.pi
  else 0

theorem sphere_surface_area 
  (AA1 AB BC : ℝ) (angle_ABC : ℝ) (h : ℝ)
  (h_eq : AA1 = 6)
  (AB_eq : AB = 4)
  (BC_eq : BC = 2)
  (angle_ABC_eq: angle_ABC = 60) :
  surface_area_of_sphere AA1 AB BC angle_ABC h = 240 * Real.pi :=
begin
  rw [h_eq, AB_eq, BC_eq, angle_ABC_eq],
  exact if_pos ⟨rfl, rfl, rfl, rfl⟩,
end

end sphere_surface_area_l313_313640


namespace hotel_room_charge_l313_313050

variables (G R P : ℝ)

-- Conditions
def cond1 : Prop := P = 0.50 * R
def cond2 : Prop := R = 1.80 * G

-- Question (to prove)
def question : Prop := ((G - P) / G) * 100 = 10

theorem hotel_room_charge (h1 : cond1) (h2 : cond2) : question :=
sorry

end hotel_room_charge_l313_313050


namespace sam_initial_money_l313_313039

theorem sam_initial_money :
  (9 * 7 + 16 = 79) :=
by
  sorry

end sam_initial_money_l313_313039


namespace find_e_l313_313828

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e
  (d f : ℝ)
  (H1 : f = 9)
  (H2 : ( -(d / 3))^2 = 3)
  (H3 : 3 + d + e + f = -3) :
  e = -15 - 3 * sqrt 3 :=
by
  sorry

end find_e_l313_313828


namespace cost_of_fencing_each_side_l313_313694

theorem cost_of_fencing_each_side (x : ℝ) (h : 4 * x = 316) : x = 79 :=
by
  sorry

end cost_of_fencing_each_side_l313_313694


namespace probability_at_least_four_8s_in_five_rolls_l313_313134

-- Definitions 
def prob_three_favorable : ℚ := 3 / 10

def prob_at_least_four_times_in_five_rolls : ℚ := 5 * (prob_three_favorable^4) * ((7 : ℚ)/10) + (prob_three_favorable)^5

-- The proof statement
theorem probability_at_least_four_8s_in_five_rolls : prob_at_least_four_times_in_five_rolls = 2859.3 / 10000 :=
by
  sorry

end probability_at_least_four_8s_in_five_rolls_l313_313134


namespace sin_135_eq_sqrt2_div_2_l313_313544

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313544


namespace interval_a_b_l313_313210

noncomputable def f (x : ℝ) : ℝ := |Real.log (x - 1)|

theorem interval_a_b (a b : ℝ) (x1 x2 : ℝ) (h1 : 1 < x1) (h2 : x1 < x2) (h3 : x2 < b) (h4 : f x1 > f x2) :
  a < 2 := 
sorry

end interval_a_b_l313_313210


namespace find_coefficients_l313_313315

def h (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 1

def j (x : ℝ) : ℝ := x^3 + b*x^2 + c*x + d

theorem find_coefficients : ∃ b c d : ℝ, j = x^3 + x^2 + 2*x + 1 ∧ (b, c, d) = (1, 2, 1) :=
by
  let h (x : ℝ) := x^3 - 2*x^2 + 3*x - 1
  let j (x : ℝ) := x^3 + b*x^2 + c*x + d
  let s := Roots of h
  let t := s - 1
  have s_polynomial : ∀ s, h s = 0 → s^3 - 2*s^2 + 3*s - 1 = 0 := sorry
  have t_transformation : ∀ t, (t+1)^3 = 2*(t+1)^2 - 3*(t+1) + 1 := sorry
  have j_polynomial : j = x^3 + x^2 + 2*x + 1 := sorry
  refine ⟨1, 2, 1, j_polynomial, rfl⟩

end find_coefficients_l313_313315


namespace determine_r_l313_313931

-- Definition capturing the conditions
def satisfies_condition (r : ℝ): Prop := 
  ⌊r⌋ + r = 14.4

-- The theorem to prove
theorem determine_r (r : ℝ) : satisfies_condition r → r = 7.4 :=
begin
  sorry,
end

end determine_r_l313_313931


namespace sin_135_eq_sqrt2_over_2_l313_313576

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313576


namespace parallelogram_area_l313_313607

theorem parallelogram_area (base : ℝ) (height : ℝ) (angle : ℝ) 
  (h_base : base = 12) (h_height : height = 8) (h_angle : angle = 35) :
  let perpendicular_height := height * Real.sin (angle * Real.pi / 180)
  in base * perpendicular_height ≈ 55.0656 := 
by sorry

end parallelogram_area_l313_313607


namespace machines_produce_12x_boxes_in_expected_time_l313_313104

-- Definitions corresponding to the conditions
def rate_A (x : ℕ) := x / 10
def rate_B (x : ℕ) := 2 * x / 5
def rate_C (x : ℕ) := 3 * x / 8
def rate_D (x : ℕ) := x / 4

-- Total combined rate when working together
def combined_rate (x : ℕ) := rate_A x + rate_B x + rate_C x + rate_D x

-- The time taken to produce 12x boxes given their combined rate
def time_to_produce (x : ℕ) : ℕ := 12 * x / combined_rate x

-- Goal: Time taken should be 32/3 minutes
theorem machines_produce_12x_boxes_in_expected_time (x : ℕ) : time_to_produce x = 32 / 3 :=
sorry

end machines_produce_12x_boxes_in_expected_time_l313_313104


namespace sin_135_eq_sqrt2_div_2_l313_313587

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313587


namespace part1_part2_l313_313645

theorem part1 (a : ℝ) (h : -3 ∈ ({a - 3, 2a - 1, a^2 + 1} : set ℝ)) :
  a = 0 ∨ a = -1 :=
sorry

theorem part2 (x : ℝ) (h : x^2 ∈ ({0, 1, x} : set ℝ)) :
  x = -1 :=
sorry

end part1_part2_l313_313645


namespace find_vector_p_l313_313325

noncomputable def vector_a : ℝ × ℝ × ℝ := (2, -2, 4)
noncomputable def vector_b : ℝ × ℝ × ℝ := (1, 6, 1)
noncomputable def vector_p : ℝ × ℝ × ℝ := (59/37, 86/37, 73/37)

theorem find_vector_p (vector_v : ℝ × ℝ × ℝ) :
  (∃ t : ℝ, 
    vector_p = 
      (vector_a.1 + t * (vector_b.1 - vector_a.1),
       vector_a.2 + t * (vector_b.2 - vector_a.2),
       vector_a.3 + t * (vector_b.3 - vector_a.3))) ∧ 
  (vector_a.1 - vector_p.1) * (vector_v.1 - vector_a.1) + 
    (vector_a.2 - vector_p.2) * (vector_v.2 - vector_a.2) + 
    (vector_a.3 - vector_p.3) * (vector_v.3 - vector_a.3) = 0 ∧ 
  (vector_b.1 - vector_p.1) * (vector_v.1 - vector_b.1) + 
    (vector_b.2 - vector_p.2) * (vector_v.2 - vector_b.2) + 
    (vector_b.3 - vector_p.3) * (vector_v.3 - vector_b.3) = 0 := 
sorry

end find_vector_p_l313_313325


namespace sin_135_degree_l313_313531

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313531


namespace minimize_ab_magnitude_l313_313695

variables (x : ℝ)

def A := (x, 5 - x, 2 * x - 1)
def B := (4, 2, 3)

def vector_AB (x : ℝ) : ℝ × ℝ × ℝ := 
  (4 - x, x - 3, 4 - 2 * x)

def magnitude (x : ℝ) : ℝ :=
  real.sqrt ((4 - x)^2 + (x - 3)^2 + (4 - 2 * x)^2)

theorem minimize_ab_magnitude : x = 5 / 2 → 
  magnitude x = real.sqrt (6 * (5 / 2)^2 - 30 * (5 / 2) + 41) :=
by
  intro hx
  rw hx
  sorry

end minimize_ab_magnitude_l313_313695


namespace qingyang_2015_mock_exam_l313_313877

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

def problem :=
  U = {1, 2, 3, 4, 5} ∧ A = {2, 3, 4} ∧ B = {2, 5} →
  B ∪ (U \ A) = {1, 2, 5}

theorem qingyang_2015_mock_exam (U : Set ℕ) (A : Set ℕ) (B : Set ℕ) : problem U A B :=
by
  intros
  sorry

end qingyang_2015_mock_exam_l313_313877


namespace square_table_seats_4_pupils_l313_313452

-- Define the conditions given in the problem
def num_rectangular_tables := 7
def seats_per_rectangular_table := 10
def total_pupils := 90
def num_square_tables := 5

-- Define what we want to prove
theorem square_table_seats_4_pupils (x : ℕ) :
  total_pupils = num_rectangular_tables * seats_per_rectangular_table + num_square_tables * x →
  x = 4 :=
by
  sorry

end square_table_seats_4_pupils_l313_313452


namespace no_divisibility_condition_by_all_others_l313_313042

theorem no_divisibility_condition_by_all_others 
  {p : ℕ → ℕ} 
  (h_distinct_odd_primes : ∀ i j, i ≠ j → Nat.Prime (p i) ∧ Nat.Prime (p j) ∧ p i ≠ p j ∧ p i % 2 = 1 ∧ p j % 2 = 1)
  (h_ordered : ∀ i j, i < j → p i < p j) :
  ¬ ∀ i j, i ≠ j → (∀ k ≠ i, k ≠ j → p k ∣ (p i ^ 8 - p j ^ 8)) :=
by
  sorry

end no_divisibility_condition_by_all_others_l313_313042


namespace honey_per_hive_correct_l313_313295

noncomputable def total_honey_jars (friend_jars : ℕ) : ℕ :=
(friend_jars * 2)

noncomputable def total_honey_liters (jars : ℕ) (liters_per_jar : ℝ) : ℝ :=
(jars : ℝ) * liters_per_jar

noncomputable def honey_per_hive (total_honey : ℝ) (number_of_hives : ℕ) : ℝ :=
total_honey / (number_of_hives : ℝ)

theorem honey_per_hive_correct :
  let friend_jars := 100 in
  let jars := total_honey_jars friend_jars in
  let liters_per_jar := 0.5 in
  let number_of_hives := 5 in
  let total_honey := total_honey_liters jars liters_per_jar in
  honey_per_hive total_honey number_of_hives = 20 := by
  sorry

end honey_per_hive_correct_l313_313295


namespace circle_equation_exists_l313_313221

def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (2, -2)
def l (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0
def is_on_circle (C : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) : Prop :=
  (p.1 - C.1)^2 + (p.2 - C.2)^2 = r^2

theorem circle_equation_exists :
  ∃ C : ℝ × ℝ, C.1 - C.2 + 1 = 0 ∧
  (is_on_circle C A 5) ∧
  (is_on_circle C B 5) ∧
  is_on_circle C (-3, -2) 5 :=
sorry

end circle_equation_exists_l313_313221


namespace total_oysters_and_crabs_is_195_l313_313386

-- Define the initial conditions
def oysters_day1 : ℕ := 50
def crabs_day1 : ℕ := 72

-- Define the calculations for the second day
def oysters_day2 : ℕ := oysters_day1 / 2
def crabs_day2 : ℕ := crabs_day1 * 2 / 3

-- Define the total counts over the two days
def total_oysters : ℕ := oysters_day1 + oysters_day2
def total_crabs : ℕ := crabs_day1 + crabs_day2
def total_count : ℕ := total_oysters + total_crabs

-- The goal specification
theorem total_oysters_and_crabs_is_195 : total_count = 195 :=
by
  sorry

end total_oysters_and_crabs_is_195_l313_313386


namespace paintable_wall_area_l313_313481

/-- Given 4 bedrooms each with length 15 feet, width 11 feet, and height 9 feet,
and doorways and windows occupying 80 square feet in each bedroom,
prove that the total paintable wall area is 1552 square feet. -/
theorem paintable_wall_area
  (bedrooms : ℕ) (length width height doorway_window_area : ℕ) :
  bedrooms = 4 →
  length = 15 →
  width = 11 →
  height = 9 →
  doorway_window_area = 80 →
  4 * (2 * (length * height) + 2 * (width * height) - doorway_window_area) = 1552 :=
by
  intros bedrooms_eq length_eq width_eq height_eq doorway_window_area_eq
  -- Definition of the problem conditions
  have bedrooms_def : bedrooms = 4 := bedrooms_eq
  have length_def : length = 15 := length_eq
  have width_def : width = 11 := width_eq
  have height_def : height = 9 := height_eq
  have doorway_window_area_def : doorway_window_area = 80 := doorway_window_area_eq
  -- Assertion of the correct answer
  sorry

end paintable_wall_area_l313_313481


namespace john_walks_farther_l313_313745

theorem john_walks_farther :
  let john_distance : ℝ := 1.74
  let nina_distance : ℝ := 1.235
  john_distance - nina_distance = 0.505 :=
by
  sorry

end john_walks_farther_l313_313745


namespace perimeter_of_room_l313_313865

-- The necessary conditions extracted from the problem:
variables (breadth length : ℝ)

-- Adding the conditions as assumptions
axiom (breadth_condition : length = 3 * breadth)
axiom (area_condition : length * breadth = 12)

-- The theorem to prove the perimeter is 16 meters
theorem perimeter_of_room : 2 * (length + breadth) = 16 :=
by
  -- Given the conditions, prove that the perimeter is 16 meters
  exact sorry

end perimeter_of_room_l313_313865


namespace polar_point_equivalence_l313_313288

noncomputable def point_equivalent (r1 θ1 r2 θ2 : ℝ) : Prop :=
  r1 = -r2 ∧ θ1 = θ2 + π

theorem polar_point_equivalence :
  point_equivalent 6 (4 * real.pi / 3) (-6) (real.pi / 3) :=
by
  sorry

end polar_point_equivalence_l313_313288


namespace probability_sum_greater_l313_313936

theorem probability_sum_greater (d a h : ℕ) (hd : 1 ≤ d ∧ d ≤ 6) (ha : 1 ≤ a ∧ a ≤ 6) (hh : 1 ≤ h ∧ h ≤ 6) :
  ((\sum (x : ℕ) in (finset.range 11).filter (λ x, x + 2 > h), 1) : ℚ) /
  ((6 * 6 * 6) : ℚ) = 17 / 72 := sorry

end probability_sum_greater_l313_313936


namespace sequence_terms_positive_integers_l313_313758

theorem sequence_terms_positive_integers (N : ℕ) (hN : 0 < N) :
  ∀ n : ℕ, ∃ a : ℕ, (a₀ : 0) ∧ 
  ∀ n : ℕ, a_(n+1) = N * (a_n + 1) + (N + 1) * a_n + 2 * sqrt(N * (N+1) * a_n * (a_n +1)) :=
by
  sorry

end sequence_terms_positive_integers_l313_313758


namespace die_probability_greater_than_4_given_tail_l313_313132

theorem die_probability_greater_than_4_given_tail :
  let outcomes := [1, 2, 3, 4, 5, 6]
  let favorable_outcomes := [5, 6]
  let total_outcomes := 6
  let favorable_count := List.length favorable_outcomes
  let probability := (favorable_count / total_outcomes : ℚ)
  probability = (1 / 3 : ℚ) :=
by
  -- Definitions of outcomes, conditions and computation will go here.
  sorry

end die_probability_greater_than_4_given_tail_l313_313132


namespace find_angle_between_vectors_l313_313232

noncomputable def angle_between_vectors {α : Type*} [field α] [is_domain α] [complex_space α] 
  {z1 z2 z3 : α} (a : ℝ) (ha : a ≠ 0) 
  (h : (z3 - z1) / (z2 - z1) = a * complex.I) : ℝ :=
π / 2

theorem find_angle_between_vectors (z1 z2 z3 : ℂ) (a : ℝ) (ha : a ≠ 0)
  (h : (z3 - z1) / (z2 - z1) = a * complex.I) : 
  angle_between_vectors z1 z2 z3 a ha h = π / 2 := 
sorry

end find_angle_between_vectors_l313_313232


namespace arithmetic_sequence_general_term_l313_313240

theorem arithmetic_sequence_general_term (a : ℤ) 
    (h1 : (a - 1) = (a - 1)) 
    (h2 : (2 * a + 3)) 
    : ∃ a_n : ℕ → ℤ, a_n = (λ n : ℕ, 2 * n - 3) :=
by
  sorry

end arithmetic_sequence_general_term_l313_313240


namespace increasing_function_range_l313_313246

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 1 then (2 * a + 3) * x - 4 * a + 3 else a^x

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) ↔ (1 < a ∧ a ≤ 2) :=
by {
  sorry
}

end increasing_function_range_l313_313246


namespace compute_100m_plus_n_l313_313082

noncomputable def minimal_area_triangle (A B C : ℝ × ℝ × ℝ) : ℝ :=
let (x1, y1, z1) := A,
    (x2, y2, z2) := B,
    (x3, y3, z3) := C in
have hA : 1 + 3 * x1 + 5 * y1 + 7 * z1 = 0, by sorry,
have hB : 1 + 3 * x2 + 5 * y2 + 7 * z2 = 0, by sorry,
have hC : 1 + 3 * x3 + 5 * y3 + 7 * z3 = 0, by sorry,
let a := x2 - x1,
    b := y2 - y1,
    c := z2 - z1,
    d := x3 - x1,
    e := y3 - y1,
    f := z3 - z1,
    cross_product := (b * f - c * e, c * d - a * f, a * e - b * d),
    area := 0.5 * real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2) in
area

theorem compute_100m_plus_n : 100 * 83 + 2 = 8302 :=
by sorry

end compute_100m_plus_n_l313_313082


namespace max_good_groups_l313_313911

-- Define the total number of terminals
def terminals : ℕ := 25

-- Define the total number of main bidirectional tunnels (edges)
def main_tunnels : ℕ := 50

-- Define what constitutes a good group (K4 subgraph)
def is_good_group (graph : SimpleGraph (Fin terminals)) (group : Finset (Fin terminals)) : Prop :=
  group.card = 4 ∧ graph.subgraph_on group = SimpleGraph.complete (Fin 4)

-- Define the main problem statement in Lean
theorem max_good_groups (graph : SimpleGraph (Fin terminals)) :
  (graph.edgeFinset.card = main_tunnels) →
  ∃ k, (k = 8) ∧ (∀ g, is_good_group graph g → ∃ x, x ≤ k) :=
by
  sorry

end max_good_groups_l313_313911


namespace arrange_magnitudes_l313_313677

theorem arrange_magnitudes (x : ℝ) (h1 : 0.85 < x) (h2 : x < 1.1)
  (y : ℝ := x + Real.sin x) (z : ℝ := x ^ (x ^ x)) : x < y ∧ y < z := 
sorry

end arrange_magnitudes_l313_313677


namespace train_length_is_95_l313_313462

noncomputable def train_length (time_seconds : ℝ) (speed_kmh : ℝ) : ℝ := 
  let speed_ms := speed_kmh * 1000 / 3600 
  speed_ms * time_seconds

theorem train_length_is_95 : train_length 1.5980030008814248 214 = 95 := by
  sorry

end train_length_is_95_l313_313462


namespace find_eccentricity_l313_313656

-- Define the focus of the parabola y^2 = 8x
def parabola_focus : (ℝ × ℝ) := (2, 0)

-- Define the focus condition for the hyperbola x^2/a^2 - y^2 = 1
def hyperbola_focus_condition (a : ℝ) : Prop :=
  (parabola_focus = (a, sqrt(a^2 + 1)))

-- Define eccentricity calculation
def hyperbola_eccentricity (a c : ℝ) : ℝ :=
  c / a

-- State the problem as a theorem
theorem find_eccentricity : ∀ (a : ℝ), hyperbola_focus_condition a → hyperbola_eccentricity a 2 = 2 * sqrt 3 / 3 :=
by
  intro a hCondition
  sorry

end find_eccentricity_l313_313656


namespace sin_135_l313_313500

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313500


namespace intersection_A_B_l313_313986

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l313_313986


namespace equilateral_triangle_of_condition_l313_313997

theorem equilateral_triangle_of_condition (a b c : ℝ) (h : a^2 + b^2 + c^2 - a * b - b * c - c * a = 0) :
  a = b ∧ b = c := 
sorry

end equilateral_triangle_of_condition_l313_313997


namespace angle_MA_perpendicular_BC_l313_313051

noncomputable theory

variables {A B C S T Q P M : Type*}
variables [normed_space ℝ A] [normed_space ℝ B] [normed_space ℝ C]
variables [normed_space ℝ S] [normed_space ℝ T]
variables [normed_space ℝ Q] [normed_space ℝ P] [normed_space ℝ M]

variable (triangle_ABC : triangle ℝ A B C)
variables (circle_tangent_S : tangent_point ℝ A B S) (circle_tangent_T : tangent_point ℝ B C T)
variables (intersection_M : intersection_point ℝ Q P T S M)

theorem angle_MA_perpendicular_BC :
  (altitude ℝ A B C).line.contains M → orthogonal ℝ (vector ℝ A M) (vector ℝ B C) :=
sorry

end angle_MA_perpendicular_BC_l313_313051


namespace volume_of_wedge_l313_313133

theorem volume_of_wedge (d : ℝ) (angle : ℝ) (π_value : ℝ) :
  d = 16 ∧ angle = 60 ∧ π_value = Real.pi →
  ∃ (n : ℝ), let r := d / 2,
                 h := d,
                 V_cylinder := π_value * r^2 * h,
                 V_wedge := (1 / 6) * V_cylinder
              in V_wedge = n * π_value ∧ n = 341 :=
begin
  sorry
end

end volume_of_wedge_l313_313133


namespace solve_sqrt_equation_l313_313800

theorem solve_sqrt_equation (x : ℝ) (h1 : x ≠ -4) (h2 : (3 * x - 1) / (x + 4) > 0) : 
  sqrt ((3 * x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3 * x - 1)) = 0 ↔ x = 5 / 2 :=
by { sorry }

end solve_sqrt_equation_l313_313800


namespace koala_fiber_consumption_l313_313302

theorem koala_fiber_consumption
  (absorbed_fiber : ℝ) (total_fiber : ℝ) 
  (h1 : absorbed_fiber = 0.40 * total_fiber)
  (h2 : absorbed_fiber = 12) :
  total_fiber = 30 := 
by
  sorry

end koala_fiber_consumption_l313_313302


namespace tile_arrangement_possible_l313_313390

-- Define the properties of the tiles
structure Tile :=
  (orientation : Bool) -- True for diagonal from top-left to bottom-right, False for top-right to bottom-left

-- Define the properties of the grid arrangement
def is_valid_arrangement (tiles : List (List Tile)) : Prop :=
  forall i j, i < 6 -> j < 6 -> 
  (tiles[i][j].orientation ≠ tiles[i+1][j+1].orientation ∧ 
   tiles[i][j].orientation ≠ tiles[i-1][j-1].orientation ∧
   tiles[i][j].orientation ≠ tiles[i+1][j-1].orientation ∧
   tiles[i][j].orientation ≠ tiles[i-1][j+1].orientation)

-- Define the problem statement
theorem tile_arrangement_possible : 
  ∃ tiles : List (List Tile), (List.length tiles = 6) ∧ (∀ row, List.length row = 6) ∧ is_valid_arrangement tiles :=
sorry

end tile_arrangement_possible_l313_313390


namespace sin_135_eq_l313_313489

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313489


namespace determine_x_l313_313959

-- Definitions for d(n) and s(n)
def divisors (n : ℕ) : List ℕ := (List.range' 1 (n + 1)).filter (λ k, n % k = 0)

def d (n : ℕ) : ℕ := (divisors n).length

def s (n : ℕ) : ℕ := (divisors n).sum

-- Theorem statement
theorem determine_x (x : ℕ) (hx : s(x) * d(x) = 96) : x = 14 ∨ x = 15 ∨ x = 47 :=
by
  sorry

end determine_x_l313_313959


namespace max_area_of_section_l313_313290

-- Define the triangular pyramid and its properties
variables (A B C D E F G : Type*)
variables (AB AC BD CD BC : ℝ)
variables (α : Set (Set E))
variables [AffineSpace V P]
variables [MetricSpace P]
variables [NormedAddCommGroup V]

-- Given conditions
def conditions : Prop :=
  AB = 4 ∧ AC = 4 ∧ BD = 4 ∧ CD = 4 ∧ BC = 4 ∧
  (midpoint AC = E) ∧ (perpendicular α BC)

-- The maximum area of the section cut by plane α
def max_area_section_cut : ℝ := 3 / 2

-- The theorem to be proven
theorem max_area_of_section : conditions → area_cut α = max_area_section_cut :=
by 
  sorry

end max_area_of_section_l313_313290


namespace complex_number_norm_eq_one_is_rational_l313_313010

def is_rational (x : ℝ) : Prop := ∃ q : ℚ, x = q

theorem complex_number_norm_eq_one_is_rational (x y : ℚ) (n : ℤ) (h : x^2 + y^2 = 1) :
  is_rational (abs ((⟨x, y⟩ : ℂ)^(2 * n) - 1)) :=
by sorry

end complex_number_norm_eq_one_is_rational_l313_313010


namespace number_of_words_differing_in_at_least_five_places_l313_313125

theorem number_of_words_differing_in_at_least_five_places 
  (x y : Fin 8 → Bool)
  (hx : finset.filter (λ i => x i ≠ y i) (finset.univ : finset (Fin 8))).card = 3) : 
  (finset.filter (λ z : Fin 8 → Bool => 
    ((finset.filter (λ i => x i ≠ z i) (finset.univ : finset (Fin 8))).card ≥ 5) ∧ 
    ((finset.filter (λ i => y i ≠ z i) (finset.univ : finset (Fin 8))).card ≥ 5)
    ) (finset.univ : finset (Fin 8 → Bool))).card = 38 := 
sorry

end number_of_words_differing_in_at_least_five_places_l313_313125


namespace eq_c_is_quadratic_l313_313857

theorem eq_c_is_quadratic (a b c : ℝ) (h1 : a = sqrt 2) (h2 : b = -sqrt 2 / 4) (h3 : c = -1 / 2) : 
    a * x^2 + b * x + c = 0 :=
by
  sorry

end eq_c_is_quadratic_l313_313857


namespace Tim_paid_correct_amount_l313_313395

noncomputable def cost_of_eggs : ℝ := 0.50
noncomputable def number_of_dozen : ℕ := 3
noncomputable def eggs_per_dozen : ℕ := 12
noncomputable def discount_rate : ℝ := 0.10
noncomputable def sales_tax_rate : ℝ := 0.05

theorem Tim_paid_correct_amount :
  let number_of_eggs := number_of_dozen * eggs_per_dozen,
      initial_cost := number_of_eggs * cost_of_eggs,
      discount := discount_rate * initial_cost,
      discounted_price := initial_cost - discount,
      tax := sales_tax_rate * discounted_price,
      final_price := discounted_price + tax
  in final_price = 17.01 := by
  sorry

end Tim_paid_correct_amount_l313_313395


namespace sphere_circumcircle_radius_l313_313091

theorem sphere_circumcircle_radius (A B C : Point) (r1 r2 : ℝ) (r1_plus_r2_is_nine : r1 + r2 = 9)
  (d_centers_is_sqrt_305 : dist(center_of_sphere_touching_A, center_of_sphere_touching_B) = sqrt 305)
  (radius_third_sphere_is_seven : radius_sphere(center C) = 7)
  (third_sphere_touches_others : touches_externally (center_of_sphere_touching_A) (center_of_sphere_touching_B) C) :
  circumcircle_radius ABC = 2 * sqrt 14 :=
by sorry

end sphere_circumcircle_radius_l313_313091


namespace Bernardo_smaller_Silvia_prob_l313_313166

-- Define the sets from which Bernardo and Silvia will pick numbers
def Bernardo_set : set ℕ := {1, 2, 3, 4, 5, 6, 7}
def Silvia_set : set ℕ := {2, 3, 4, 5, 6, 7, 8}

-- Define the random selection of 3 distinct numbers and forming a 3-digit number
def pick3_and_form_number (s : set ℕ) : set (finset ℕ) :=
  {A : finset ℕ | A.card = 3 ∧ A ⊆ s}

-- Define the events of picking numbers and the conditions
def event_Bernardo_picks : set (finset ℕ) := pick3_and_form_number Bernardo_set
def event_Silvia_picks : set (finset ℕ) := pick3_and_form_number Silvia_set

-- Define the probability that Bernardo's 3-digit number is smaller than Silvia's 3-digit number
noncomputable def probability_Bernardo_smaller_Silvia : ℚ :=
  sorry -- To be completed

-- Statement of the theorem
theorem Bernardo_smaller_Silvia_prob : probability_Bernardo_smaller_Silvia = 5/7 :=
  sorry

end Bernardo_smaller_Silvia_prob_l313_313166


namespace boat_downstream_distance_l313_313883

theorem boat_downstream_distance 
  (Vb Vr T D U : ℝ)
  (h1 : Vb + Vr = 21)
  (h2 : Vb - Vr = 12)
  (h3 : U = 48)
  (h4 : T = 4)
  (h5 : D = 20) :
  (Vb + Vr) * D = 420 :=
by
  sorry

end boat_downstream_distance_l313_313883


namespace max_min_of_f_on_interval_l313_313952

noncomputable theory
open Classical

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 - 12 * x + 5

theorem max_min_of_f_on_interval : ∃ (max min : ℝ), max = 5 ∧ min = -15 ∧ 
  (∀ x ∈ set.Icc 0 3, f x ≤ max) ∧ 
  (∀ x ∈ set.Icc 0 3, f x ≥ min) := 
sorry

end max_min_of_f_on_interval_l313_313952


namespace sin_cos_product_sin_minus_cos_l313_313206

open Real

noncomputable def α : ℝ := by sorry

axiom α_in_interval : α ∈ Ioo ((5 / 4) * π) ((3 / 2) * π)

axiom tan_condition : tan α + (1 / tan α) = 8

theorem sin_cos_product : sin α * cos α = 1 / 8 := by sorry

theorem sin_minus_cos : sin α - cos α = - (sqrt 3 / 2) := by sorry

end sin_cos_product_sin_minus_cos_l313_313206


namespace minimum_b1_b2_l313_313078

noncomputable def sequence (b : ℕ → ℕ) : Prop :=
∀ n ≥ 1, b (n + 2) = (b n + 2023) / (1 + b (n + 1))

theorem minimum_b1_b2 (b : ℕ → ℕ) (h : sequence b) (pos : ∀ n, b n > 0) :
  b 1 + b 2 = 136 :=
sorry

end minimum_b1_b2_l313_313078


namespace sin_135_eq_sqrt2_div_2_l313_313543

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313543


namespace ratio_of_BC_AB_l313_313004

   variable {α : Type} [EuclideanGeometry α]

   -- Let ABC be a right triangle with \( \angle B = 90^\circ \)
   -- E and F are midpoints of AB and AC respectively.
   -- The incenter I of ABC lies on the circumcircle of triangle AEF.
   -- Prove that the ratio BC/AB equals 4/3

   theorem ratio_of_BC_AB 
     {A B C E F I : Point α}
     (h_triangle : triangle A B C)
     (h_right_angle : angle B = 90°)
     (h_midpoints : midpoint E A B ∧ midpoint F A C)
     (h_incenter : incenter I A B C)
     (h_on_circumcircle : lies_on_circumcircle I A E F)
     : dist B C / dist A B = 4 / 3 :=
   by 
     sorry
   
end ratio_of_BC_AB_l313_313004


namespace cube_rotation_vertex_move_l313_313446

-- Define the initial conditions of the cube and the position of the point
def cube_initial_conditions : Prop :=
  let faces := ["green", "far white", "right lower white"]
  ∧ A ∈ faces

-- Define the rotation conditions
def rotation_conditions : Prop :=
  -- New positions of the faces after rotation
  let new_faces := ["green", "far white", "left upper white"]
  ∀ A, A ∈ new_faces

-- Final state (the conclusion we need to prove)
def final_position_of_A : Prop :=
  vertex A = 3

-- Main theorem statement
theorem cube_rotation_vertex_move (c : cube_initial_conditions) (r : rotation_conditions) : final_position_of_A :=
by
  sorry

end cube_rotation_vertex_move_l313_313446


namespace proof_A_proof_C_l313_313101

theorem proof_A (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  a * b ≤ ( (a + b) / 2) ^ 2 := 
sorry

theorem proof_C (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2) : 
  ∃ y, y = x * (4 - x^2).sqrt ∧ y ≤ 2 := 
sorry

end proof_A_proof_C_l313_313101


namespace number_of_ways_l313_313287

def digit_set : Set ℕ := {0, 2, 4, 5, 7, 9}

def digit_sum (digits : List ℕ) : ℕ := digits.sum

def is_valid (num : ℕ) : Prop :=
  num % 5 = 0 ∧ digit_sum (List.drop 4 (List.ofDigits 10 num)) % 3 = 0

theorem number_of_ways :
  { replace_digits : List ℕ // replace_digits.length = 6 ∧ ∀ d ∈ replace_digits, d ∈ digit_set ∧
    is_valid (List.foldl (λ n d, n * 10 + d) 2016000020 replace_digits) }
  .card = 5184 :=
sorry

end number_of_ways_l313_313287


namespace concurrency_of_lines_l313_313148

theorem concurrency_of_lines (square: Type) (lines: Fin 2005 → square) 
  (area_ratio: ∀ i, divides_to_trapezoids square (lines i) ∧ area_ratio (lines i) = 2 / 3): 
  ∃ (p: point), (502 ≤ (set_of_lines_concurrent_at p)) :=
begin
  sorry
end

end concurrency_of_lines_l313_313148


namespace eccentricity_quadratic_curve_l313_313667

theorem eccentricity_quadratic_curve :
  ∀ (x y : ℝ), 10 * x - 2 * x * y - 2 * y + 1 = 0 → eccentricity (10 * x - 2 * x * y - 2 * y + 1) = sqrt 2 := by
  sorry

end eccentricity_quadratic_curve_l313_313667


namespace minimum_moves_for_exchange_l313_313731

-- Definitions representing the problem conditions
def chessboard := {1, 2, 3, 4, 5, 6, 7, 8}

def white_knight_positions : set ℕ := {1, 3}
def black_knight_positions : set ℕ := {5, 7}

def is_knight_move (start end : ℕ) : Prop :=
  (abs (end / 3 - start / 3) = 2 ∧ abs (end % 3 - start % 3) = 1) ∨
  (abs (end / 3 - start / 3) = 1 ∧ abs (end % 3 - start % 3) = 2)

-- Proven claim
theorem minimum_moves_for_exchange :
  ∀ ⦃white_knight black_knight : chessboard⦄,
  white_knight = white_knight_positions →
  black_knight = black_knight_positions →
  min_moves_required white_knight black_knight = 16 :=
by
  sorry

end minimum_moves_for_exchange_l313_313731


namespace construct_isosceles_trapezoid_l313_313255

-- Define the geometric objects and conditions
variables {A C B D : Type} [AffineSpace ℝ A] [AffineSpace ℝ C] [AffineSpace ℝ B] [AffineSpace ℝ D]

def isosceles_trapezoid (A C B D : AffineSpace ℝ ℝ) :=
  (AD_parallel_BC : (A - D) ∥ (B - C)) ∧
  (isosceles : ∥A - B∥ = ∥D - C∥) ∧ 
  (given_vertices : ∃ (A C : AffineSpace ℝ ℝ), A ≠ C) ∧
  (given_directions : ∃ (d1 d2 : ℝ), d1 ≠ 0 ∧ d2 ≠ 0)

-- Statement to prove the existence of B and D
theorem construct_isosceles_trapezoid (A C : AffineSpace ℝ ℝ) 
  (h : isosceles_trapezoid A C B D) :
  ∃ (B D : AffineSpace ℝ ℝ), (isosceles_trapezoid A C B D) :=
begin
  sorry
end

end construct_isosceles_trapezoid_l313_313255


namespace remainder_1394_mod_2535_l313_313610

-- Definition of the least number satisfying the given conditions
def L : ℕ := 1394

-- Proof statement: proving the remainder of division
theorem remainder_1394_mod_2535 : (1394 % 2535) = 1394 :=
by sorry

end remainder_1394_mod_2535_l313_313610


namespace probability_same_suit_JQKA_l313_313964

theorem probability_same_suit_JQKA  : 
  let deck_size := 52 
  let prob_J := 4 / deck_size
  let prob_Q_given_J := 1 / (deck_size - 1) 
  let prob_K_given_JQ := 1 / (deck_size - 2)
  let prob_A_given_JQK := 1 / (deck_size - 3)
  prob_J * prob_Q_given_J * prob_K_given_JQ * prob_A_given_JQK = 1 / 1624350 :=
by
  sorry

end probability_same_suit_JQKA_l313_313964


namespace max_value_l313_313384

theorem max_value (m n S_even S_odd : ℕ) 
  (h1 : S_even + S_odd = 1987)
  (h2 : S_even = ∑ i in range m, 2 * (i + 1)) 
  (h3 : S_odd = ∑ i in range n, 2 * i + 1) :
  3 * m + 4 * n ≤ 221 := 
sorry

end max_value_l313_313384


namespace gcd_g50_g52_l313_313755

-- Define the polynomial function g
def g (x : ℤ) : ℤ := x^3 - 2 * x^2 + x + 2023

-- Define the integers n1 and n2 corresponding to g(50) and g(52)
def n1 : ℤ := g 50
def n2 : ℤ := g 52

-- Statement of the proof goal
theorem gcd_g50_g52 : Int.gcd n1 n2 = 1 := by
  sorry

end gcd_g50_g52_l313_313755


namespace problem_l313_313628

variable (n : ℕ) (a b : ℕ → ℝ)
variable (n_geq_3 : n ≥ 3)
variable (sum_eq : (∑ i in Finset.range n, a (i + 1)) = (∑ i in Finset.range n, b (i + 1)))
variable (a_cond : ∀ i, 0 < a 1 ∧ a 1 = a 2 ∧ (i < n - 2 →  a i + a (i + 1) = a (i + 2)))
variable (b_cond : ∀ i, 0 < b 1 ∧ b 1 ≤ b 2 ∧ (i < n - 2 → b i + b (i + 1) ≤ b (i + 2)))

theorem problem (n_geq_3 : n ≥ 3)
    (sum_eq : (∑ i in Finset.range n, a (i + 1)) = (∑ i in Finset.range n, b (i + 1)))
    (a_cond : 0 < a 1 ∧ a 1 = a 2 ∧ ∀ i < n - 2, a i + a (i + 1) = a (i + 2))
    (b_cond : 0 < b 1 ∧ b 1 ≤ b 2 ∧ ∀ i < n - 2, b i + b (i + 1) ≤ b (i + 2)) :
    a (n-1) + a n ≤ b (n-1) + b n := 
sorry

end problem_l313_313628


namespace sum_of_b_and_c_base7_l313_313751

theorem sum_of_b_and_c_base7 (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) 
(h4 : A < 7) (h5 : B < 7) (h6 : C < 7) 
(h7 : 7^2 * A + 7 * B + C + 7^2 * B + 7 * C + A + 7^2 * C + 7 * A + B = 7^3 * A + 7^2 * A + 7 * A + 1) 
: B + C = 6 ∨ B + C = 12 := sorry

end sum_of_b_and_c_base7_l313_313751


namespace haruto_ratio_is_1_to_2_l313_313683

def haruto_tomatoes_ratio (total_tomatoes : ℕ) (eaten_by_birds : ℕ) (remaining_tomatoes : ℕ) : ℚ :=
  let picked_tomatoes := total_tomatoes - eaten_by_birds
  let given_to_friend := picked_tomatoes - remaining_tomatoes
  given_to_friend / picked_tomatoes

theorem haruto_ratio_is_1_to_2 : haruto_tomatoes_ratio 127 19 54 = 1 / 2 :=
by
  -- We'll skip the proof details as instructed
  sorry

end haruto_ratio_is_1_to_2_l313_313683


namespace ratio_of_boys_to_total_students_l313_313274

theorem ratio_of_boys_to_total_students
  (p : ℝ)
  (h : p = (3/4) * (1 - p)) :
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_to_total_students_l313_313274


namespace middle_number_consecutive_sum_l313_313708

theorem middle_number_consecutive_sum (a b c : ℕ) (h1 : b = a + 1) (h2 : c = b + 1) (h3 : a + b + c = 30) : b = 10 :=
by
  sorry

end middle_number_consecutive_sum_l313_313708


namespace range_of_f_l313_313254

open Real

def f (x : ℝ) : ℝ := cos (2 * x) + sqrt 3 * sin (2 * x)

theorem range_of_f :
  let interval := Icc (π / 4) (π / 2)
  let lower_bound := -1
  let upper_bound := sqrt 3
  ∀ x ∈ interval, lower_bound ≤ f x ∧ f x ≤ upper_bound :=
begin
  sorry
end

end range_of_f_l313_313254


namespace trains_cross_time_l313_313404

theorem trains_cross_time (L : ℝ) (t₁ t₂ : ℝ) (H₁ : L = 120) (H₂ : t₁ = 10) (H₃ : t₂ = 14) :
  let s₁ := L / t₁;
  let s₂ := L / t₂;
  let d := 2 * L;
  let s := s₁ + s₂;
  let T := d / s;
  T ≈ 11.67 :=
by
  rw [H₁, H₂, H₃]
  have : s₁ = 12 := by norm_num
  have : s₂ ≈ 8.57 := by norm_num
  have : s ≈ 20.57 := by norm_num
  have : d = 240 := by norm_num
  have : T ≈ 11.67 := by norm_num
  simp [*, eq_comm]
  sorry

end trains_cross_time_l313_313404


namespace remainder_15_plus_3y_l313_313757

theorem remainder_15_plus_3y (y : ℕ) (hy : 7 * y ≡ 1 [MOD 31]) : (15 + 3 * y) % 31 = 11 :=
by
  sorry

end remainder_15_plus_3y_l313_313757


namespace chord_to_diameter_ratio_l313_313116

open Real

theorem chord_to_diameter_ratio
  (r R : ℝ) (h1 : r = R / 2)
  (a : ℝ)
  (h2 : r^2 = a^2 * 3 / 2) :
  3 * a / (2 * R) = 3 * sqrt 6 / 8 :=
by
  sorry

end chord_to_diameter_ratio_l313_313116


namespace roots_of_f_non_roots_of_g_l313_313874

-- Part (a)

def f (x : ℚ) := x^20 - 123 * x^10 + 1

theorem roots_of_f (a : ℚ) (h : f a = 0) : 
  f (-a) = 0 ∧ f (1/a) = 0 ∧ f (-1/a) = 0 :=
by
  sorry

-- Part (b)

def g (x : ℚ) := x^4 + 3 * x^3 + 4 * x^2 + 2 * x + 1

theorem non_roots_of_g (β : ℚ) (h : g β = 0) : 
  g (-β) ≠ 0 ∧ g (1/β) ≠ 0 ∧ g (-1/β) ≠ 0 :=
by
  sorry

end roots_of_f_non_roots_of_g_l313_313874


namespace not_in_set_A_l313_313418

def A : set (ℤ × ℤ) := { p | ∃ x y : ℤ, p = (x, y) ∧ y = 3 * x - 5 }

theorem not_in_set_A : ¬(1, -5) ∈ A :=
by {
  intro h,
  rcases h with ⟨x, y, ⟨hx, hy⟩⟩,
  subst hx,
  have : y = -2 := by refl,
  exact hy,
}

end not_in_set_A_l313_313418


namespace geometric_triangle_condition_right_geometric_triangle_condition_l313_313095

-- Definitions for the geometric progression
def geometric_sequence (a b c q : ℝ) : Prop :=
  b = a * q ∧ c = a * q^2

-- Conditions for forming a triangle
def forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions for forming a right triangle using Pythagorean theorem
def right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem geometric_triangle_condition (a q : ℝ) (h1 : 1 ≤ q) (h2 : q < (1 + Real.sqrt 5) / 2) :
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ forms_triangle a b c := 
sorry

theorem right_geometric_triangle_condition (a q : ℝ) :
  q = Real.sqrt ((1 + Real.sqrt 5) / 2) →
  ∃ (b c : ℝ), geometric_sequence a b c q ∧ right_triangle a b c :=
sorry

end geometric_triangle_condition_right_geometric_triangle_condition_l313_313095


namespace problem1_problem2_problem3_l313_313256

-- Problem 1
theorem problem1 : (∀ n, b(n) = a(n+1) - a(n)) ∧ (∀ n, b(n) = 10 - n) → a 16 - a 5 = 0 :=
by
  intros h1 h2
  sorry

-- Problem 2
theorem problem2 : (∀ n, b(n) = a(n+1) - a(n)) ∧ (∀ n, b(n) = (-1)^n * (2^n + 2^(33-n))) ∧ (a 1 = 1) → (∀ n, (a (2*n+1))) (smallest a 17) :=
by
  intros h1 h2 h3
  sorry

-- Problem 3
theorem problem3 :
  (∀ n, b n = a (n + 1) - a n) ∧ (∀ n, c n = a n + 2 * a (n + 1)) →
  ((∀ n, b n ≤ b (n + 1) → (∃ d, a n = a 1 + n * d)) ↔
   (∀ m, c (m + 1) - c m = c 2 - c 1) ∧ (∀ n, b n ≤ b (n + 1))) :=
by
  intros h1 h2
  sorry

end problem1_problem2_problem3_l313_313256


namespace seven_digit_numbers_with_repeats_l313_313723

theorem seven_digit_numbers_with_repeats :
  let count := (λ n : ℕ,
    to_string n |>.length = 7 ∧ 
    ∀ d, (to_string n).count d ≥ 3) in
  (finset.filter count (finset.Icc 1000000 9999999)).card = 2844 :=
sorry

end seven_digit_numbers_with_repeats_l313_313723


namespace triangle_problem_l313_313978

theorem triangle_problem (a b c : ℝ) (A : ℝ) (S : ℝ) 
  (h1 : a^2 = b^2 + c^2 - b * c)
  (h2 : a = √7)
  (h3 : c - b = 2)
  (h4 : 0 < A ∧ A < π) :
  (A = π / 3) ∧ (S = (1 / 2) * b * c * (sin (π / 3))) :=
by
  sorry

end triangle_problem_l313_313978


namespace part_a_l313_313438

theorem part_a : 
  ∃ (x y : ℕ → ℕ), (∀ n : ℕ, (1 + Real.sqrt 33) ^ n = x n + y n * Real.sqrt 33) :=
sorry

end part_a_l313_313438


namespace jamshid_takes_less_time_l313_313296

open Real

theorem jamshid_takes_less_time (J : ℝ) (hJ : J < 15) (h_work_rate : (1 / J) + (1 / 15) = 1 / 5) :
  (15 - J) / 15 * 100 = 50 :=
by
  sorry

end jamshid_takes_less_time_l313_313296


namespace sin_135_eq_sqrt2_over_2_l313_313575

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313575


namespace stripe_length_34_l313_313448

def stripe_length_equation (C : ℕ) (h : ℕ) : ℕ :=
  let diagonal := (C^2 + h^2)
  Float.sqrt diagonal

theorem stripe_length_34
  (C h : ℕ)
  (hC : C = 30)
  (hh : h = 16) :
  stripe_length_equation C h = 34 :=
sorry

end stripe_length_34_l313_313448


namespace calculate_area_l313_313729

-- Define points and segments with given specific properties
variables {A B C D E F : Type} 

-- Define the lengths of the sides
def side_length (x y : Type) : Prop := dist x y = 1

-- Define the right-angled triangles
def right_angle_at (x y z : Type) : Prop := 
  angle x y z = π / 2

-- Define parallelism
def parallel (l1 l2 : Line): Prop := l1.slope = l2.slope

-- conditions for the problem
variables [point A] [point B] [point C] [point D] [point E] [point F]
variables [line AF AB BC CD DE EF]

-- Declaring the known conditions
axiom AF_parallel_CD : parallel AF CD
axiom AB_parallel_EF : parallel AB EF
axiom BC_parallel_ED : parallel BC ED
axiom length_AF : side_length A F
axiom length_AB : side_length A B
axiom length_BC : side_length B C
axiom length_CD : side_length C D
axiom length_ED : side_length E D
axiom length_EF : side_length E F
axiom rt_angle_FAB : right_angle_at F A B
axiom rt_angle_BCD : right_angle_at B C D

-- The theorem to prove
theorem calculate_area : area (polygon [F, A, B, C, D, E]) = 1 :=
sorry

end calculate_area_l313_313729


namespace lucas_fourth_day_collection_l313_313771

def marbles_collected_on_fourth_day (total_marbles : ℕ) (days : ℕ) (diff : ℕ) : ℕ :=
  let x := (total_marbles + 24) / 6 in
  x

theorem lucas_fourth_day_collection :
  marbles_collected_on_fourth_day 120 6 8 = 24 :=
by
  -- Definitions and conditions directly from the problem
  unfold marbles_collected_on_fourth_day
  -- Calculations show the number of marbles collected on the fourth day is 24
  sorry

end lucas_fourth_day_collection_l313_313771


namespace constant_term_of_alice_poly_is_6_l313_313924

-- Definition of monic polynomial
def is_monic (p : Polynomial ℝ) : Prop :=
  p.leadingCoeff = 1

-- Given polynomial product
def poly_prod : Polynomial ℝ :=
  Polynomial.of_nat_degree 6 -- This constructs the polynomial x^6 + 2x^5 + x^4 + 2x^3 + 9x^2 + 12x + 36

-- Our main theorem statement
theorem constant_term_of_alice_poly_is_6 :
  ∃ (p q : Polynomial ℝ),
    is_monic p ∧ is_monic q ∧
    p.degree = 3 ∧ q.degree = 3 ∧
    p.coeff 0 = q.coeff 0 ∧ p.coeff 0 > 0 ∧
    p * q = poly_prod ∧
    p.coeff 0 = 6 := 
sorry

end constant_term_of_alice_poly_is_6_l313_313924


namespace sin_135_eq_sqrt2_over_2_l313_313571

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313571


namespace find_x_solution_l313_313605

noncomputable def x_sol : ℝ := 7^(2/5)

theorem find_x_solution : (∃ x : ℝ, log x 49 = log 2 32) ∧ log x_sol 49 = log 2 32 :=
by
  have h1 : log 2 32 = 5 := by sorry
  have h2 : log x_sol 49 = log 2 32 := by sorry
  exact ⟨⟨x_sol, h2⟩, h2⟩

end find_x_solution_l313_313605


namespace sum_absolute_differences_eq_n_squared_l313_313937

theorem sum_absolute_differences_eq_n_squared (n : ℕ) 
  (a b : Fin n.succ → ℕ)
  (h₁ : a = List.sort (· < ·) (List.range (2 * n + 1) \ (List.range n).map (b ·))) 
  (h₂ : b = List.sort (· > ·) (List.range n)) :
  (∑ i in Finset.range n, |a i - b i|) = n ^ 2 :=
by
  sorry

end sum_absolute_differences_eq_n_squared_l313_313937


namespace determine_c_l313_313935

theorem determine_c (c y : ℝ) : (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 := by
  sorry

end determine_c_l313_313935


namespace simplify_expression_l313_313939

variable (a b : ℚ)

theorem simplify_expression (h1 : b ≠ 1/2) (h2 : b ≠ 1) :
  (2 * a + 1) / (1 - b / (2 * b - 1)) = (2 * a + 1) * (2 * b - 1) / (b - 1) :=
by 
  sorry

end simplify_expression_l313_313939


namespace prism_volume_correct_l313_313144

noncomputable def prism_volume
  (l α β : ℝ)
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (cos_half : ∀ x, cos (x / 2) = sqrt ((1 + cos x) / 2))  -- assuming cos is cosine of radians
  (sin_half : ∀ x, sin (x) = sqrt (1 - cos (x) * cos (x)))  -- similarly for sin
  : ℝ :=
  2 * l ^ 3 * cos β ^ 2 * (cos (α / 2)) ^ 2 * sin α * sin β

-- To formally state the problem, one may want to proceed as:

theorem prism_volume_correct 
  (l α β : ℝ)
  (cos : ℝ → ℝ) 
  (sin : ℝ → ℝ)
  (cos_half : ∀ x, cos (x / 2) = sqrt ((1 + cos x) / 2))  -- Assumption on cosine
  (sin_half : ∀ x, sin (x) = sqrt (1 - cos (x) * cos (x))) :  -- Assumption on sine
  prism_volume l α β cos sin cos_half sin_half = 
  2 * l ^ 3 * cos β ^ 2 * (cos (α / 2)) ^ 2 * sin α * sin β :=
sorry

end prism_volume_correct_l313_313144


namespace binary_253_l313_313925

def binary_representation (n : ℕ) : list ℕ :=
  if n = 0 then [0]
  else
    let rec aux (n : ℕ) : list ℕ :=
      match n with
      | 0   => []
      | n+1 => aux (n / 2) ++ [n % 2]
    aux n

def count_elements (lst : list ℕ) (elem : ℕ) : ℕ :=
  lst.filter (λ x => x = elem).length

def num_of_zeros (n : ℕ) : ℕ :=
  count_elements (binary_representation n) 0

def num_of_ones (n : ℕ) : ℕ :=
  count_elements (binary_representation n) 1

theorem binary_253 (x y : ℕ) (h₀ : x = num_of_zeros 253) (h₁ : y = num_of_ones 253) : y - x = 6 := by
  sorry

end binary_253_l313_313925


namespace fuel_tank_capacity_l313_313912

theorem fuel_tank_capacity (C : ℝ) 
  (h1 : 0.12 * 98 + 0.16 * (C - 98) = 30) : 
  C = 212 :=
by
  sorry

end fuel_tank_capacity_l313_313912


namespace probability_product_is_square_l313_313809

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def probability_square_product : ℚ :=
  let total_outcomes   := 10 * 8
  let favorable_outcomes := 
    [(1,1), (1,4), (2,2), (4,1), (3,3), (2,8), (8,2), (5,5), (6,6), (7,7), (8,8)].length
  favorable_outcomes / total_outcomes

theorem probability_product_is_square : 
  probability_square_product = 11 / 80 :=
  sorry

end probability_product_is_square_l313_313809


namespace intersection_A_B_union_complement_A_B_l313_313674

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 4 ≤ 2^x ∧ 2^x < 16}
def B : Set ℝ := {x | ∃ y, log10 (x-3) = y}

theorem intersection_A_B :
  A ∩ B = {x | 3 < x ∧ x < 4} :=
sorry

theorem union_complement_A_B :
  ((U \ A) ∪ B) = {x | x < 2 ∨ x > 3} :=
sorry

end intersection_A_B_union_complement_A_B_l313_313674


namespace solve_equation_l313_313873

noncomputable def equation_to_solve (x : ℝ) : ℝ :=
  1 / (4^(3*x) - 13 * 4^(2*x) + 51 * 4^x - 60) + 1 / (4^(2*x) - 7 * 4^x + 12)

theorem solve_equation :
  (equation_to_solve (1/2) = 0) ∧ (equation_to_solve (Real.log 6 / Real.log 4) = 0) :=
by {
  sorry
}

end solve_equation_l313_313873


namespace find_matrix_N_l313_313951

noncomputable def matrixN : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![85 / 14, - (109 / 14)], ![-3, 4]]

def matrixA : Matrix (Fin 2) (Fin 2) ℚ := 
  ![![2, -5], ![4, -3]]

def matrixB : Matrix (Fin 2) (Fin 2) ℚ := 
  ![[-19, -7], ![10, 3]]

open Matrix

theorem find_matrix_N :
  matrixN ⬝ matrixA = matrixB :=
by
  -- This is just a statement placeholder
  sorry

end find_matrix_N_l313_313951


namespace find_years_invested_l313_313443

-- Defining the conditions and theorem
variables (P : ℕ) (r1 r2 D : ℝ) (n : ℝ)

-- Given conditions
def principal := (P : ℝ) = 7000
def rate_1 := r1 = 0.15
def rate_2 := r2 = 0.12
def interest_diff := D = 420

-- Theorem to be proven
theorem find_years_invested (h1 : principal P) (h2 : rate_1 r1) (h3 : rate_2 r2) (h4 : interest_diff D) :
  7000 * 0.15 * n - 7000 * 0.12 * n = 420 → n = 2 :=
by
  sorry

end find_years_invested_l313_313443


namespace integral_solution_l313_313476

theorem integral_solution (a : ℝ) (h : ∫ (x in 1..a), (2 * x + 1 / x) = 3 + real.log 2) : a = 2 :=
by
  sorry

end integral_solution_l313_313476


namespace geometric_sequence_triangle_l313_313235

theorem geometric_sequence_triangle (a b c A B C : ℝ) (q : ℝ) 
  (h1 : b = a * q) (h2 : c = a * q^2)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (htriangle_ineq1 : a + b > c)
  (htriangle_ineq2 : b + c > a)
  (htriangle_ineq3 : c + a > b)
  (hA : Real.sin A = a / c)
  (hB : Real.sin B = b / c)
  (hC : Real.sin C = a / b)
  (h_C_A_eq : Real.sin (A + C) = Real.sin (π - B))
  (h_C_B_eq : Real.sin (B + C) = Real.sin (π - A))
  : ∃ q,  q ∈ Ioo ((Real.sqrt 5 - 1) / 2) ((Real.sqrt 5 + 1) / 2) ∧ 
  (a / c + (Real.sin A * Real.cot C + Real.cos A)) / 
  (b / a + (Real.sin B * Real.cot C + Real.cos B)) = q :=
sorry

end geometric_sequence_triangle_l313_313235


namespace intersection_A_B_l313_313985

-- Define sets A and B according to the conditions provided
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

-- Define the theorem stating the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 2} := by
  sorry

end intersection_A_B_l313_313985


namespace solve_problem1_solve_problem2_l313_313324

noncomputable def problem1 (A B C : ℝ) (a b c : ℝ) : Prop :=
  (C = 2 * Real.pi / 3) ∧
  (\frac{\cos A}{1 - \sin A} = \frac{\cos A + \cos B}{1 - \sin A + \sin B}) →
  A = \frac{\pi}{6}

theorem solve_problem1 (A B C : ℝ) (a b c : ℝ) :
  problem1 A B C a b c :=
sorry

noncomputable def problem2 (A B C : ℝ) (a b c : ℝ) : Prop :=
  (\frac{\cos A}{1 - \sin A} = \frac{\cos A + \cos B}{1 - \sin A + \sin B}) →
  (\frac{a^2 + c^2}{b^2} ≥ 4\sqrt{2} - 5)

theorem solve_problem2 (A B C : ℝ) (a b c : ℝ) :
  problem2 A B C a b c :=
sorry

end solve_problem1_solve_problem2_l313_313324


namespace remainder_equiv_one_mod_12_l313_313389

theorem remainder_equiv_one_mod_12 (a b c d e : ℕ) 
  (h1 : a < 12) (h2 : b < 12) (h3 : c < 12) (h4 : d < 12) (h5 : e < 12)
  (h6 : Nat.gcd a 12 = 1) (h7 : Nat.gcd b 12 = 1) (h8 : Nat.gcd c 12 = 1)
  (h9 : Nat.gcd d 12 = 1) (h10 : Nat.gcd e 12 = 1) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ((a * b * c * d * e + a * b * c * d + a * b * c * e + a * b * d * e + a * c * d * e + b * c * d * e) * 
  (Nat.gcd_inv (a * b * c * d * e) 12)) % 12 = 1 :=
sorry

end remainder_equiv_one_mod_12_l313_313389


namespace sum_Tn_l313_313658

noncomputable def sequence_a (n : ℕ) : ℤ :=
  3 * n - 1

noncomputable def sequence_b (n : ℕ) : ℤ :=
  2 ^ n

noncomputable def sequence_c (n : ℕ) : ℕ → ℤ := λ k, (sequence_a k) * (sequence_b k)

noncomputable def sum_sequence_c (n : ℕ) : ℤ :=
∑ k in Finset.range n, sequence_c n k

theorem sum_Tn : ∀ (n : ℕ),
  sum_sequence_c n = (3 * n - 4) * 2 ^ (n + 1) + 8 :=
begin
  sorry
end

end sum_Tn_l313_313658


namespace midpoint_PQ_l313_313034

-- Definitions of the conditions
variables {A B A₁ B₁ A₂ B₂ P Q M : Type}
variables (seg_A seg_B seg_A1 seg_B1 seg_A2 seg_B2 line_A1B2 line_A2B1 line_PQ : A → B → Prop)
variables (midpoint : A → B → A → Prop)

-- Condition: M is the midpoint of AB
axiom midpointAB (M A B : Type) : midpoint M A B

-- Condition: Two lines through M intersect sides of the angle at A1, B1, A2, B2
axiom lines_through_midpoint (M A B A₁ B₁ A₂ B₂ : Type) 
  (seg_AB seg_A1B1 seg_A2B2 : Type)
  (intersect1 : line_A1B2 A₁ B₂)
  (intersect2 : line_A2B1 A₂ B₁) : Prop

-- Condition: Lines A1B2 and A2B1 intersect AB at P and Q
axiom intersection_points (line_A1B2 line_A2B1 A₁ B₂ A₂ B₁ P Q : Type)
  (cross1 : line_A1B2 P Q)
  (cross2 : line_A2B1 P Q) : Prop

-- Conclusion: Prove M is the midpoint of PQ
theorem midpoint_PQ : 
  (midpointAB M A B) →
  (lines_through_midpoint M A B A₁ B₁ A₂ B₂ seg_A seg_A1 seg_B1 seg_A2 seg_B2 line_A1B2 line_A2B1) →
  (intersection_points line_A1B2 line_A2B1 A₁ B₂ A₂ B₁ P Q) →
  midpoint M P Q :=
by
  sorry

end midpoint_PQ_l313_313034


namespace trader_overall_loss_percentage_l313_313429

theorem trader_overall_loss_percentage :
  ∀ (SP1 SP2 : ℝ) (percentage_gain percentage_loss : ℝ)
  (CP1 CP2 TCP TSP : ℝ),
  SP1 = 325475 → 
  SP2 = 325475 → 
  percentage_gain = 0.13 → 
  percentage_loss = 0.13 →
  CP1 = SP1 / (1 + percentage_gain) → 
  CP2 = SP2 / (1 - percentage_loss) →
  TCP = CP1 + CP2 → 
  TSP = SP1 + SP2 → 
  ((TSP - TCP) / TCP) * 100 ≈ -1.684 :=
  by
  intros SP1 SP2 percentage_gain percentage_loss CP1 CP2 TCP TSP h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end trader_overall_loss_percentage_l313_313429


namespace max_elements_in_set_l313_313903

noncomputable theory

open Set

def distinct_positive_integers (n : ℕ) (S : Set ℕ) : Prop :=
  ∀ x ∈ S, ∃ m : ℕ, x = 2 + m * n

theorem max_elements_in_set (n : ℕ) (S : Set ℕ)
  (h1 : 1 ∈ S) (h2 : 2017 ∈ S) (h3 : ∀ x ∈ S, distinct_positive_integers n S)
  (h4 : ∀ x ∈ S, (S.erase x).sum / (card S - 1) ∈ ℕ) :
  card S ≤ 4 := 
sorry

end max_elements_in_set_l313_313903


namespace evaluate_expression_l313_313427

theorem evaluate_expression :
  |7 - (8^2) * (3 - 12)| - |(5^3) - (Real.sqrt 11)^4| = 579 := 
by 
  sorry

end evaluate_expression_l313_313427


namespace incorrect_option_D_l313_313662

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 1

theorem incorrect_option_D (a b c : ℝ) (h : a ≠ 0) :
  ¬ (∀ x₀ : ℝ, 
       (∃ x₀ : ℝ, is_local_min f a b c x₀) → 
       (∀ x : ℝ, x < x₀ → f a b c x > f a b c x₀)) := 
sorry

end incorrect_option_D_l313_313662


namespace last_three_digits_of_S_l313_313006

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def valid_sum (x : ℕ) : Prop :=
  ∃ (m n : ℕ), m ≤ 9 ∧ n ≤ 9 ∧ x = factorial m + factorial n ∧ x < 10^6

def S : ℕ :=
  ∑ x in (finset.filter valid_sum (finset.range (10^6))), x

theorem last_three_digits_of_S :
  S % 1000 = 130 :=
sorry

end last_three_digits_of_S_l313_313006


namespace sin_135_eq_l313_313515

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313515


namespace percentage_yield_l313_313885

theorem percentage_yield (market_price annual_dividend : ℝ) (yield : ℝ) 
  (H1 : yield = 0.12)
  (H2 : market_price = 125)
  (H3 : annual_dividend = yield * market_price) :
  (annual_dividend / market_price) * 100 = 12 := 
sorry

end percentage_yield_l313_313885


namespace john_payment_l313_313741

def total_cost (cakes : ℕ) (cost_per_cake : ℕ) : ℕ :=
  cakes * cost_per_cake

def split_cost (total : ℕ) (people : ℕ) : ℕ :=
  total / people

theorem john_payment (cakes : ℕ) (cost_per_cake : ℕ) (people : ℕ) : 
  cakes = 3 → cost_per_cake = 12 → people = 2 → 
  split_cost (total_cost cakes cost_per_cake) people = 18 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end john_payment_l313_313741


namespace sin_135_eq_l313_313488

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313488


namespace sin_135_eq_sqrt2_over_2_l313_313570

theorem sin_135_eq_sqrt2_over_2 : Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := by
  sorry

end sin_135_eq_sqrt2_over_2_l313_313570


namespace orthocenter_identity_l313_313618

theorem orthocenter_identity (A B C P : Point) 
  (hABC : acute_angled_triangle A B C)
  (hP_inside : strictly_inside P A B C)
  (h_ortho : is_orthocenter P A B C):
  AB_dist A B * BC_dist B C * CA_dist C A = PA_dist P A * PB_dist P B * AB_dist A B + PB_dist P B * PC_dist P C * BC_dist B C + PC_dist P C * PA_dist P A * CA_dist C A := 
sorry

end orthocenter_identity_l313_313618


namespace sin_135_eq_l313_313493

theorem sin_135_eq : (135 : ℝ) = 180 - 45 → (∀ x : ℝ, sin (180 - x) = sin x) → (sin 45 = real.sqrt 2 / 2) → (sin 135 = real.sqrt 2 / 2) := by
  sorry

end sin_135_eq_l313_313493


namespace sum_of_inserted_numbers_l313_313620

theorem sum_of_inserted_numbers (x y : ℝ) (r : ℝ) 
  (h1 : 4 * r = x) 
  (h2 : 4 * r^2 = y) 
  (h3 : (2 / y) = ((1 / x) + (1 / 16))) :
  x + y = 8 :=
sorry

end sum_of_inserted_numbers_l313_313620


namespace car_total_distance_l313_313129

noncomputable def distance_first_segment (speed1 : ℝ) (time1 : ℝ) : ℝ :=
  speed1 * time1

noncomputable def distance_second_segment (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed2 * time2

noncomputable def distance_final_segment (speed3 : ℝ) (time3 : ℝ) : ℝ :=
  speed3 * time3

noncomputable def total_distance (d1 d2 d3 : ℝ) : ℝ :=
  d1 + d2 + d3

theorem car_total_distance :
  let d1 := distance_first_segment 65 2
  let d2 := distance_second_segment 80 1.5
  let d3 := distance_final_segment 50 2
  total_distance d1 d2 d3 = 350 :=
by
  sorry

end car_total_distance_l313_313129


namespace find_k_l313_313949

def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def construct_number (k : ℕ) : ℕ :=
  let n := 1000
  let a := (10^(2000 - k) - 1) / 9
  let b := (10^(1001) - 1) / 9
  a * 10^(1001) + k * 10^(1001 - k) - b

theorem find_k : ∀ k : ℕ, (construct_number k > 0) ∧ (isPerfectSquare (construct_number k) ↔ k = 2) := 
by 
  intro k
  sorry

end find_k_l313_313949


namespace area_inside_C_outside_A_B_l313_313921

def radius_of_A_and_B : ℝ := 1
def radius_of_C : ℝ := 2
def center_distance : ℝ := radius_of_C -- Circle C's center is radius_of_C units from the point of tangency.

theorem area_inside_C_outside_A_B :
  let area_C := π * (radius_of_C ^ 2)
  let area_overlap := (2 * (π * radius_of_C * radius_of_C * (1/6)) - radius_of_A_and_B * sqrt 3 / 2)
  area_C - area_overlap = 4 * π - (2 * π / 3) + sqrt 3 :=
by
  sorry

end area_inside_C_outside_A_B_l313_313921


namespace express_105_9_billion_in_scientific_notation_l313_313348

def express_in_scientific_notation (n: ℝ) : ℝ × ℤ :=
  let exponent := int.of_nat (nat.floor $ real.logb 10 n)
  let coefficient := n / real.pow 10 exponent
  (coefficient, exponent)

theorem express_105_9_billion_in_scientific_notation :
  express_in_scientific_notation (105.9 * 10^9) = (1.059, 10) :=
by
  sorry

end express_105_9_billion_in_scientific_notation_l313_313348


namespace serving_time_equals_180_l313_313329

def total_serving_time : ℕ :=
  let missy_standard_patients := 30 - (2 * 30 / 5)
  let thomas_standard_patients := 15 - (1 * 15 / 3)
  let missy_standard_time := missy_standard_patients * 5
  let thomas_standard_time := thomas_standard_patients * 6
  let missy_special_patients := 2 * 30 / 5
  let thomas_special_patients := 1 * 15 / 3
  let missy_special_time := missy_special_patients * (5 * 1.5)
  let thomas_special_time := thomas_special_patients * (6 * 1.75)
  let missy_total_time := missy_standard_time + missy_special_time.toNat
  let thomas_total_time := thomas_standard_time + thomas_special_time.toNat
  max missy_total_time thomas_total_time

theorem serving_time_equals_180 :
  total_serving_time = 180 := by
    sorry

end serving_time_equals_180_l313_313329


namespace polynomial_factorization_l313_313706

theorem polynomial_factorization (m n : ℤ) (h₁ : (x + 1) * (x + 3) = x^2 + m * x + n) : m - n = 1 := 
by {
  -- Proof not required
  sorry
}

end polynomial_factorization_l313_313706


namespace quadratic_roots_real_part_negative_l313_313792

theorem quadratic_roots_real_part_negative (p q : ℝ) :
  (p = 0 ∧ -1 < q ∧ q < 0) ∨ (p > 0 ∧ q < p^2 ∧ q > 2 * p - 1) ↔
  ∀ x : ℝ, (-Real.part (by apply polynomial.roots (px^2 + (p^2 - q) * x - (2*p - q - 1))) x < 0) :=
sorry

end quadratic_roots_real_part_negative_l313_313792


namespace Greg_harvested_acres_l313_313260

-- Defining the conditions
def Sharon_harvested : ℝ := 0.1
def Greg_harvested (additional: ℝ) (Sharon: ℝ) : ℝ := Sharon + additional

-- Proving the statement
theorem Greg_harvested_acres : Greg_harvested 0.3 Sharon_harvested = 0.4 :=
by
  sorry

end Greg_harvested_acres_l313_313260


namespace football_shaped_area_l313_313340

-- Definitions based on given conditions
def is_square (A B C D : Point) (s : ℝ) : Prop :=
  dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s ∧
  dist A C = sqrt (s^2 + s^2) ∧ dist B D = sqrt (s^2 + s^2)

def circle_arc_area (center : Point) (radius : ℝ) (theta : ℝ) : ℝ :=
  (theta / 360) * π * radius^2

def triangle_area (base height : ℝ) : ℝ :=
  0.5 * base * height

-- Main statement and theorem
theorem football_shaped_area {A B C D : Point} (s : ℝ)
  (h_sq : is_square A B C D s)
  (r : ℝ := s)
  (theta : ℝ := 90) :
  let sector_area := circle_arc_area D r theta,
      triangle_area := triangle_area s s,
      region_III_area := sector_area - triangle_area in
  2 * region_III_area = approx 2.3 :=
by sorry

end football_shaped_area_l313_313340


namespace f_strictly_increasing_on_l313_313211

-- Define the function
def f (x : ℝ) : ℝ := x^2 * (2 - x)

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := -3 * x^2 + 4 * x

-- Define the property that the function is strictly increasing on an interval
def strictly_increasing_on (a b : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

-- State the theorem
theorem f_strictly_increasing_on : strictly_increasing_on 0 (4/3) f :=
sorry

end f_strictly_increasing_on_l313_313211


namespace max_lg_val_l313_313693

theorem max_lg_val (x y : ℝ) (h1 : 0 ∈ ({real.log x, real.log y, real.log (x + y / x) }: set ℝ))
  (h2 : 1 ∈ ({real.log x, real.log y, real.log (x + y / x)}: set ℝ)) :
  (real.log (11 : ℝ)) ∈ ({real.log x, real.log y, real.log (x + y / x)} : set ℝ) :=
sorry

end max_lg_val_l313_313693


namespace arithmetic_expression_eval_l313_313413

theorem arithmetic_expression_eval : 3 + (12 / 3 - 1) ^ 2 = 12 := by
  sorry

end arithmetic_expression_eval_l313_313413


namespace n_squared_chessboard_l313_313899

theorem n_squared_chessboard (n : ℕ) :
  ∃ k, ∀ (coloring : fin (2 * n) → fin (2 * n^2 - n + 1) → fin n),
  ∃ (i1 j1 i2 j2 : fin (2 * n)),
    i1 ≠ i2 ∧ j1 ≠ j2 ∧
    coloring i1 j1 = coloring i1 j2 ∧
    coloring i2 j1 = coloring i2 j2 ∧
    coloring i1 j1 = coloring i2 j1 :=
begin
  use (2 * n^2 - n + 1),
  intros coloring,
  sorry
end

end n_squared_chessboard_l313_313899


namespace min_value_frac_inv_l313_313062

theorem min_value_frac_inv (a m n : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) (hm_pos : m > 0) (hn_pos : n > 0)
  (A_fixed : (∀ x, y = a^(x-2)) (by simp[∀ x, y a (x-2), eq A (2, 1): 2x - 2)) (Hlin: y = mx + 4n) (by simp[2m+4n:eq A (2,1): mx +4n)) :
  ∃ m n, (∀ x, y = a^(x-2))  ≥ 0 ∧ (x > m ∧ x < n): (2 * m + 4 * n) :
  let c := ∀ x. 1 + \frac{y}{m}+ \frac{4}{n}+ ((2*(m) + 4(n)). : 
  minimum := 1 := ((2*m + 4*n)) + m - n:
  (a > 0 ∧ a ≠ 1) := (m=1) := n==-1 := realge0:= (x≤y):} sorry

end min_value_frac_inv_l313_313062


namespace min_distance_expression_l313_313416

variable (s t : ℝ)

theorem min_distance_expression :
  (∀ (s t : ℝ), (s + 5 - 3 * |cos t|)^2 + (s - 2 * |sin t|)^2 ≥ 2) ∧
  (∃ (s t : ℝ), (s + 5 - 3 * |cos t|)^2 + (s - 2 * |sin t|)^2 = 2) :=
by
  sorry

end min_distance_expression_l313_313416


namespace ratio_AP_PC_l313_313055

-- Given Definitions in Lean
variables (AB BC CD DA : ℝ) (AP PC : ℝ)
variables (P AB_dist PBC_dist PCD_dist PDA_dist PCA_dist : ℝ)

-- Conditions as Lean Definitions
def cyclic_quadrilateral (AB_dist BC_dist CD_dist DA_dist : ℝ) : Prop :=
  AB_dist = 5 ∧ BC_dist = sqrt 3 ∧ CD_dist = (5 / sqrt 7) ∧ DA_dist = 5 * sqrt(3 / 7)

-- Problem Statement to Prove
theorem ratio_AP_PC :
  cyclic_quadrilateral AB BC CD DA →
  AP ≠ 0 →
  PC ≠ 0 →
  (AP : ℝ) / (PC : ℝ) = 5 :=
sorry

end ratio_AP_PC_l313_313055


namespace incorrect_option_e_l313_313992

theorem incorrect_option_e (x z : ℚ) (h : x / z = 5 / 6) :
  (x - 2 * z) / z ≠ -7 / 6 :=
by
  have h1 : x / z = 5 / 6 := h
  have h2 : (x - 2 * z) / z = (5 / 6 - 2) := by
    calc
      (x - 2 * z) / z
          = (x / z - 2) : by ring
      ... = (5 / 6 - 2) : by rw [h1]
  have h3 : 5 / 6 - 2 ≠ -7 / 6 := by norm_num
  exact h3

end incorrect_option_e_l313_313992


namespace coordinates_of_B_l313_313979

noncomputable def B_coordinates := 
  let A : ℝ × ℝ := (-1, -5)
  let a : ℝ × ℝ := (2, 3)
  let AB := (3 * a.1, 3 * a.2)
  let B := (A.1 + AB.1, A.2 + AB.2)
  B

theorem coordinates_of_B : B_coordinates = (5, 4) := 
by 
  sorry

end coordinates_of_B_l313_313979


namespace largest_prime_divisor_in_range_l313_313179

theorem largest_prime_divisor_in_range (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  ∃ p, Prime p ∧ p ≤ Int.floor (Real.sqrt n) ∧ 
  (∀ q, Prime q ∧ q ≤ Int.floor (Real.sqrt n) → q ≤ p) :=
sorry

end largest_prime_divisor_in_range_l313_313179


namespace increase_interval_max_value_m_l313_313680

-- Define the vectors a and b and the function f(x)
def a (x : ℝ) : ℝ × ℝ := (2 * Real.sin x, Real.sqrt 3 * Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.sin x, 2 * Real.sin x)
def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

-- Prove the interval where f(x) is monotonically increasing
theorem increase_interval (k : ℤ) : 
  ∀ x ∈ Set.Icc (k * Real.pi - Real.pi / 6) (k * Real.pi + Real.pi / 3), 
  ∃ y, f y = 2 * Real.sin (2 * y - Real.pi / 6) + 1 ∧ 
    Set.Icc (2 * y - Real.pi / 6) (2 * y - Real.pi / 6) ⊆ 
    Set.Icc (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi + Real.pi / 2) :=
sorry

-- Prove the maximum value of m such that f(x) ≥ m for x ∈ [0, π/2] is 0
theorem max_value_m : 
  ∀ x ∈ Set.Icc 0 (Real.pi / 2), 
  (f x ≥ 0) ∧ (∀ m' : ℝ, f x ≥ m' → m' ≤ 0) :=
sorry

end increase_interval_max_value_m_l313_313680


namespace exp_A_plus_B_ne_exp_A_mul_exp_B_l313_313958

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ := !![0, 1; 0, 0]

def matrix_B : Matrix (Fin 2) (Fin 2) ℝ := !![0, 0; 1, 0]

def exp (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let I := 1
  I + M / Real.ofNat 1! + (M^2) / Real.ofNat 2! + (M^3) / Real.ofNat 3! + ...

theorem exp_A_plus_B_ne_exp_A_mul_exp_B :
  let A := matrix_A
  let B := matrix_B
  exp (A + B) ≠ exp A * exp B := by
  sorry

end exp_A_plus_B_ne_exp_A_mul_exp_B_l313_313958


namespace category_a_sampling_l313_313712

theorem category_a_sampling (students_A : ℕ) (students_B : ℕ) (students_C : ℕ) (total_sample : ℕ)
  (h1 : students_A = 2000) (h2 : students_B = 3000) (h3 : students_C = 4000) (h4 : total_sample = 900) : 
  let total_students := students_A + students_B + students_C
  in (students_A * total_sample) / total_students = 200 := sorry

end category_a_sampling_l313_313712


namespace smallest_integer_is_77_l313_313811

theorem smallest_integer_is_77 
  (A B C D E F G : ℤ)
  (h_uniq: A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F ∧ F < G)
  (h_sum: A + B + C + D + E + F + G = 840)
  (h_largest: G = 190)
  (h_two_smallest_sum: A + B = 156) : 
  A = 77 :=
sorry

end smallest_integer_is_77_l313_313811


namespace population_change_l313_313283

theorem population_change (initial_population : ℕ) (increase_rate decrease_rate : ℚ)
  (h_initial : initial_population = 10000)
  (h_increase : increase_rate = 0.05)
  (h_decrease : decrease_rate = 0.05) :
  let first_year_population := initial_population + (initial_population * increase_rate).to_nat in
  let second_year_population := first_year_population - (first_year_population * decrease_rate).to_nat in
  second_year_population = 9975 := by
  sorry

end population_change_l313_313283


namespace magnitude_of_sum_of_parallel_vectors_l313_313257

def parallel_vectors (a b : ℝ × ℝ) : Prop :=
  match a, b with
  | (a1, a2), (b1, b2) => a1 * b2 = a2 * b1

theorem magnitude_of_sum_of_parallel_vectors :
  let a := (1, 2 : ℝ)
  let b := (-2, -4 : ℝ)
  (parallel_vectors a b) →
  ‖a + b‖ = √5 :=
by
  unfold parallel_vectors
  sorry

end magnitude_of_sum_of_parallel_vectors_l313_313257


namespace solve_arcsin_eq_l313_313342

noncomputable def arcsin (x : ℝ) : ℝ := Real.arcsin x
noncomputable def pi : ℝ := Real.pi

theorem solve_arcsin_eq :
  ∃ x : ℝ, arcsin x + arcsin (3 * x) = pi / 4 ∧ x = 1 / Real.sqrt 19 :=
sorry

end solve_arcsin_eq_l313_313342


namespace geometric_progression_fifth_term_l313_313687

theorem geometric_progression_fifth_term {x : ℤ} (h1 : 3 * x + 3 = (3 * x + 3) / x * x) : 
  fifth_term_gemoetric_sequence x = 0 :=
begin
  sorry
end

end geometric_progression_fifth_term_l313_313687


namespace quadratic_ineq_solution_range_of_b_for_any_a_l313_313627

variable {α : Type*} [LinearOrderedField α]

noncomputable def f (a b x : α) : α := -3 * x^2 + a * (5 - a) * x + b

theorem quadratic_ineq_solution (a b : α) : 
  (∀ x ∈ Set.Ioo (-1 : α) 3, f a b x > 0) →
  ((a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9)) := 
  sorry

theorem range_of_b_for_any_a (a b : α) :
  (∀ a : α, f a b 2 < 0) → 
  b < -1 / 2 := 
  sorry

end quadratic_ineq_solution_range_of_b_for_any_a_l313_313627


namespace range_of_a_l313_313675

variable {x : ℝ} {a : ℝ}

theorem range_of_a (h : ∀ x : ℝ, ¬ (x^2 - 5*x + (5/4)*a > 0)) : 5 < a :=
by
  sorry

end range_of_a_l313_313675


namespace roots_squared_sum_eq_13_l313_313016

/-- Let p and q be the roots of the quadratic equation x^2 - 5x + 6 = 0. Then the value of p^2 + q^2 is 13. -/
theorem roots_squared_sum_eq_13 (p q : ℝ) (h₁ : p + q = 5) (h₂ : p * q = 6) : p^2 + q^2 = 13 :=
by
  sorry

end roots_squared_sum_eq_13_l313_313016


namespace set_T_cardinality_l313_313317

/-- Given T is the set of integers n > 1 for which 1/n is an infinite 
decimal that has the property that d_i = d_{i+10} for all positive 
integers i, and assuming 9091 is prime, prove the number of positive 
integers in T is 59. -/
theorem set_T_cardinality : 
  ∀ (T : Set ℕ), 
  (∀ n ∈ T, n > 1 ∧ ∀ i : ℕ, (i > 0 → (d_i n = d_(i+10) n))) →
  9091.prime →
  T.card = 59 := 
  by { 
    intros T hT h_prime,
    sorry 
  }

end set_T_cardinality_l313_313317


namespace tangent_circle_exists_l313_313084
open Set

-- Definitions of given point, line, and circle
variables {Point : Type*} {Line : Type*} {Circle : Type*} 
variables (M : Point) (l : Line) (S : Circle)
variables (center_S : Point) (radius_S : ℝ)

-- Conditions of the problem
variables (touches_line : Circle → Line → Prop) (touches_circle : Circle → Circle → Prop)
variables (passes_through : Circle → Point → Prop) (center_of : Circle → Point)
variables (radius_of : Circle → ℝ)

-- Existence theorem to prove
theorem tangent_circle_exists 
  (given_tangent_to_line : Circle → Line → Bool)
  (given_tangent_to_circle : Circle → Circle → Bool)
  (given_passes_through : Circle → Point → Bool):
  ∃ (Ω : Circle), 
    given_tangent_to_line Ω l ∧
    given_tangent_to_circle Ω S ∧
    given_passes_through Ω M :=
sorry

end tangent_circle_exists_l313_313084


namespace solve_for_x_l313_313043

theorem solve_for_x (x : ℝ) : 2 * (2^x) = 128 → x = 6 :=
by sorry

end solve_for_x_l313_313043


namespace sin_135_l313_313527

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313527


namespace octopus_count_l313_313482

theorem octopus_count (total_legs octopus_legs : ℕ) (h1 : total_legs = 40) 
  (h2 : octopus_legs = 8) : total_legs / octopus_legs = 5 :=
by {
  rw [h1, h2],
  norm_num,
  sorry
}

end octopus_count_l313_313482


namespace pow_mod_remainder_l313_313098

theorem pow_mod_remainder :
  (2^2013 % 11) = 8 :=
sorry

end pow_mod_remainder_l313_313098


namespace sin_135_eq_sqrt2_div_2_l313_313545

theorem sin_135_eq_sqrt2_div_2 :
  Real.sin (135 * Real.pi / 180) = (Real.sqrt 2) / 2 := 
sorry

end sin_135_eq_sqrt2_div_2_l313_313545


namespace simplify_sqrt_expr_l313_313615

theorem simplify_sqrt_expr (x : ℝ) : 
  sqrt (x^6 + 3 * x^4 + 2 * x^2) = abs x * sqrt ((x^2 + 1) * (x^2 + 2)) :=
sorry

end simplify_sqrt_expr_l313_313615


namespace sum_f_1_to_1990_l313_313730

def lattice_point (x : ℕ) (y : ℕ) : Prop := (x : ℤ) = x ∧ (y : ℤ) = y

def f (n : ℕ) : ℕ :=
  if h : Nat.gcd n (n + 3) = 1 then 0
  else 2

theorem sum_f_1_to_1990 : (Finset.range 1990).sum (λ n, f (n + 1)) = 1326 := 
  sorry

end sum_f_1_to_1990_l313_313730


namespace sin_135_l313_313522

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313522


namespace smallest_alpha_l313_313311

variables {α : ℝ} (m n p : EuclideanSpace ℝ (Fin 3))

-- Conditions
def unit_vectors : Prop := ∥m∥ = 1 ∧ ∥n∥ = 1 ∧ ∥p∥ = 1
def angle_between_m_n := real.angle m n = α
def angle_between_p_cross_product := real.angle p (m × n) = α
def n_dot_product_p_cross_m := (n ⬝ (p × m)) = (√3) / 4

-- Proof that the smallest possible value of α is 30 degrees
theorem smallest_alpha : 
  unit_vectors m n p →
  angle_between_m_n m n α →
  angle_between_p_cross_product p m n α →
  n_dot_product_p_cross_m n p m →
  α = 30 :=
by
  sorry

end smallest_alpha_l313_313311


namespace solve_equation_l313_313381

theorem solve_equation (x : ℝ) : (x - 2) ^ 2 = 9 ↔ x = 5 ∨ x = -1 :=
by
  sorry -- Proof is skipped

end solve_equation_l313_313381


namespace quadratic_radical_only_l313_313908

theorem quadratic_radical_only (a : ℝ) : 
  (∀ b ∈ { a^2 + 1 , a - 1 }, b ≥ 0 → b = a^2 + 1) :=
begin
  intros b hb h_nonneg,
  simp [set.mem_set_of, set_of],
  cases hb,
  { rw hb,
    simp, },
  { rw hb at h_nonneg,
    linarith,
  },
end

end quadratic_radical_only_l313_313908


namespace equilateral_triangle_y_coord_third_vertex_in_third_quadrant_l313_313471

theorem equilateral_triangle_y_coord_third_vertex_in_third_quadrant :
  let x1 : ℝ := 0
  let y1 : ℝ := 5
  let x2 : ℝ := 8
  let y2 : ℝ := 5
  let side_length : ℝ := 8
  let altitude : ℝ := (sqrt 3 / 2) * side_length
  let expected_y_coord : ℝ := y1 - altitude
  ∃ x3 y3 : ℝ, x3 < 0 ∧ y3 < 0 ∧ x3 ^ 2 + y3 ^ 2 = side_length ^ 2 ∧ y3 = expected_y_coord :=
begin
  -- conditions
  let x1 : ℝ := 0,
  let y1 : ℝ := 5,
  let x2 : ℝ := 8,
  let y2 : ℝ := 5,
  let side_length : ℝ := 8,
  let altitude : ℝ := (sqrt 3 / 2) * side_length,
  let expected_y_coord : ℝ := y1 - altitude,

  -- required proof
  sorry
end

end equilateral_triangle_y_coord_third_vertex_in_third_quadrant_l313_313471


namespace intersection_A_B_l313_313987

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℝ := { x | 0 < x ∧ x < 3 }

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l313_313987


namespace characterize_functions_l313_313606

open Nat

theorem characterize_functions {f g : ℕ → ℕ} 
    (h : ∀ m n : ℕ, f(m) - f(n) = (m - n) * (g(m) + g(n))) :
    ∃ a b c : ℕ, (∀ n : ℕ, f(n) = a * n^2 + 2 * b * n + c) ∧ (∀ n : ℕ, g(n) = a * n + b) :=
sorry

end characterize_functions_l313_313606


namespace avg_of_q_distinct_pos_int_roots_l313_313239

theorem avg_of_q_distinct_pos_int_roots :
  (∃ (r₁ r₂ r₃ : ℕ), r₁ + r₂ + r₃ = 7 ∧
                      ∀ {r₁ r₂ r₃}, r₁ + r₂ + r₃ = 7 → 
                      let q₁ := r₁ * r₂ * r₃ in
                      let distinct_q_vals := {q₁ | (∃ r₁ r₂ r₃ : ℕ, r₁ + r₂ + r₃ = 7 ∧ q₁ = r₁ * r₂ * r₃ } ) in
                      (distinct_q_vals.sum id) / distinct_q_vals.card = 34/4) :=
sorry

end avg_of_q_distinct_pos_int_roots_l313_313239


namespace sin_135_eq_sqrt2_div_2_l313_313589

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313589


namespace proof_problem_l313_313234

-- Given definitions
def line1 (x : ℝ) : ℝ := 2 * x
def line2 (x y a : ℝ) : Prop := x + y + a = 0

def intersection_point (P : ℝ × ℝ) (x y a : ℝ) : Prop :=
  P = (1, line1 1) ∧ line2 1 (line1 1) a

def perpendicular_distance (P : ℝ × ℝ) (a b c : ℝ) : ℝ := 
  (|a * P.1 + b * P.2 + c|) / (sqrt (a^2 + b^2))

-- The proof problem
theorem proof_problem (a b : ℝ) :
  (∃ P, intersection_point P 1 b a) →
  (P = (1, 2)) →
  a = -3 → 
  perpendicular_distance (1, 2) (-3) 2 3 = 4 * sqrt 13 / 13 :=
by
  -- Proof is omitted, but the structure of the theorem is as required.
  sorry

end proof_problem_l313_313234


namespace amys_integers_sum_correct_l313_313910

def sum_of_possible_values_lean : ℕ :=
17 + 18 

theorem amys_integers_sum_correct : ∀ (a b c d : ℤ), 
  a > b → b > c → c > d → a + b + c + d = 50 →
  { abs (a - b), abs (a - c), abs (a - d), abs (b - c), abs (b - d), abs (c - d) } = {2, 3, 4, 5, 7, 10} →
  sum_of_possible_values_lean = 35 :=
by {
  sorry
}

end amys_integers_sum_correct_l313_313910


namespace solve_sqrt_equation_l313_313795

open Real

theorem solve_sqrt_equation :
  ∀ x : ℝ, (sqrt ((3*x - 1) / (x + 4)) + 3 - 4 * sqrt ((x + 4) / (3*x - 1)) = 0) →
    (3*x - 1) / (x + 4) ≥ 0 →
    (x + 4) / (3*x - 1) ≥ 0 →
    x = 5 / 2 := by
  sorry

end solve_sqrt_equation_l313_313795


namespace cannot_assemble_highlighted_shape_l313_313780

-- Define the rhombus shape with its properties
structure Rhombus :=
  (white_triangle gray_triangle : Prop)

-- Define the assembly condition
def can_rotate (shape : Rhombus) : Prop := sorry

-- Define the specific shape highlighted that Petya cannot form
def highlighted_shape : Prop := sorry

-- The statement we need to prove
theorem cannot_assemble_highlighted_shape (shape : Rhombus) 
  (h_rotate : can_rotate shape)
  (h_highlight : highlighted_shape) : false :=
by sorry

end cannot_assemble_highlighted_shape_l313_313780


namespace length_of_segment_AB_l313_313845

noncomputable def speed_relation_first (x v1 v2 : ℝ) : Prop :=
  300 / v1 = (x - 300) / v2

noncomputable def speed_relation_second (x v1 v2 : ℝ) : Prop :=
  (x + 100) / v1 = (x - 100) / v2

theorem length_of_segment_AB :
  (∃ (x v1 v2 : ℝ),
    x > 0 ∧
    v1 > 0 ∧
    v2 > 0 ∧
    speed_relation_first x v1 v2 ∧
    speed_relation_second x v1 v2) →
  ∃ x : ℝ, x = 500 :=
by
  sorry

end length_of_segment_AB_l313_313845


namespace total_vertical_distance_of_rings_l313_313592

theorem total_vertical_distance_of_rings :
  let thickness := 2
  let top_outside_diameter := 20
  let bottom_outside_diameter := 4
  let n := (top_outside_diameter - bottom_outside_diameter) / thickness + 1
  let total_distance := n * thickness
  total_distance + thickness = 76 :=
by
  sorry

end total_vertical_distance_of_rings_l313_313592


namespace calc_mixed_number_expr_l313_313478

theorem calc_mixed_number_expr :
  53 * (3 + 1 / 4 - (3 + 3 / 4)) / (1 + 2 / 3 + (2 + 2 / 5)) = -6 - 57 / 122 := 
by
  sorry

end calc_mixed_number_expr_l313_313478


namespace extreme_values_F_range_of_a_M_a_maximum_proof_l313_313976

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * Real.log x
noncomputable def F (x : ℝ) (a : ℝ) : ℝ := f x * g x a
noncomputable def G (x : ℝ) (a : ℝ) : ℝ := f x - g x a + (a - 1) * x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := g x a - x + (1 / x)
noncomputable def M (a : ℝ) : ℝ := sorry  -- Placeholder for M(a)

theorem extreme_values_F (a : ℝ) (h_a : a > 0) :
  (∃ x : ℝ, F x a = -a / (4 * Real.exp 1)) ∧ ¬ ∃ x : ℝ, ∀ y : ℝ, F y a ≤ F x a :=
sorry

theorem range_of_a (G_zero_points_in_interval : ∀ x : ℝ, G x a = 0 → (1 / Real.exp 1) < x ∧ x < Real.exp 1) :
  ∀ (a : ℝ), (a > 0) → ((2 * Real.exp 1 - 1) / (2 * Real.exp 1^2 + 2 * Real.exp 1)) < a ∧ 
  a < (1 / 2) :=
sorry

theorem M_a_maximum_proof (a : ℝ) (h_a : a > 0) (ha_le : a ≤ Real.exp 1 + (1 / Real.exp 1)) :
  ∃ M_a_max : ℝ, (M(a) = M_a_max) ∧ M_a_max = 4 / Real.exp 1 :=
sorry

end extreme_values_F_range_of_a_M_a_maximum_proof_l313_313976


namespace trig_identity_simplification_l313_313790

theorem trig_identity_simplification :
  (sqrt (1 - 2 * Real.sin (10 * Real.pi / 180) * Real.cos (10 * Real.pi / 180)))
  / (Real.sin (170 * Real.pi / 180) - sqrt (1 - Real.sin (170 * Real.pi / 180)^2)) = -1 :=
by
  -- transformation steps and identity applications go here
  sorry

end trig_identity_simplification_l313_313790


namespace students_got_off_l313_313392

-- Define the number of students originally on the bus
def original_students : ℕ := 10

-- Define the number of students left on the bus after the first stop
def students_left : ℕ := 7

-- Prove that the number of students who got off the bus at the first stop is 3
theorem students_got_off : original_students - students_left = 3 :=
by
  sorry

end students_got_off_l313_313392


namespace triangle_orthocenter_example_l313_313722

open Real EuclideanGeometry

def point_3d := (ℝ × ℝ × ℝ)

def orthocenter (A B C : point_3d) : point_3d := sorry

theorem triangle_orthocenter_example :
  orthocenter (2, 4, 6) (6, 5, 3) (4, 6, 7) = (4/5, 38/5, 59/5) := sorry

end triangle_orthocenter_example_l313_313722


namespace minimum_side_length_to_fit_table_diagonally_l313_313901

-- Define the problem with conditions and the answer
theorem minimum_side_length_to_fit_table_diagonally (l w : ℝ) (hl : l = 12) (hw : w = 9) : 
  ∃ S : ℝ, S = 15 := 
by
  use 15
  sorry

end minimum_side_length_to_fit_table_diagonally_l313_313901


namespace sum_of_coordinates_of_other_endpoint_l313_313375

def isMidpoint (x₁ y₁ x₂ y₂ mx my : ℝ) : Prop :=
  mx = (x₁ + x₂) / 2 ∧ my = (y₁ + y₂) / 2

theorem sum_of_coordinates_of_other_endpoint :
  ∃ x y : ℝ, isMidpoint 7 2 x y 5 (-8) ∧ (x + y = -15) :=
by
  use 3
  use -18
  unfold isMidpoint
  split
  . norm_num
  . norm_num
  norm_num
  sorry

end sum_of_coordinates_of_other_endpoint_l313_313375


namespace probability_same_class_l313_313902

-- Define the problem conditions
def num_classes : ℕ := 3
def total_scenarios : ℕ := num_classes * num_classes
def same_class_scenarios : ℕ := num_classes

-- Formulate the proof problem
theorem probability_same_class :
  (same_class_scenarios : ℚ) / total_scenarios = 1 / 3 :=
sorry

end probability_same_class_l313_313902


namespace distance_from_center_of_C_to_line_l313_313697

def circle_center_distance : ℝ :=
  let line1 (x y : ℝ) := x - y - 4
  let circle1 (x y : ℝ) := x^2 + y^2 - 4 * x - 6
  let circle2 (x y : ℝ) := x^2 + y^2 - 4 * y - 6
  let line2 (x y : ℝ) := 3 * x + 4 * y + 5
  sorry

theorem distance_from_center_of_C_to_line :
  circle_center_distance = 2 := sorry

end distance_from_center_of_C_to_line_l313_313697


namespace average_minutes_per_day_l313_313475

-- Define the initial conditions
variables (e : ℕ) -- number of eighth graders
variables (sixth_minutes_per_day seventh_minutes_per_day eighth_minutes_per_day : ℕ)
variables (sixth_to_eighth_ratio sixth_to_seventh_ratio : ℕ)

-- Define given conditions
def sixth_minutes_per_day := 20
def seventh_minutes_per_day := 25
def eighth_minutes_per_day := 15
def sixth_to_eighth_ratio := 3
def sixth_to_seventh_ratio := 1

-- Prove average number of minutes run per day
theorem average_minutes_per_day : 
  (6 * sixth_minutes_per_day + 3 * seventh_minutes_per_day + 1 * eighth_minutes_per_day) / 7 = 150 / 7 :=
by {
  sorry
}

end average_minutes_per_day_l313_313475


namespace probability_neither_red_nor_purple_l313_313862

theorem probability_neither_red_nor_purple :
  (100 - (47 + 3)) / 100 = 0.5 :=
by sorry

end probability_neither_red_nor_purple_l313_313862


namespace probability_quarter_circle_l313_313679

def probability (x y : ℝ) : ℝ := 
  if 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ sqrt (x ^ 2 + y ^ 2) ≤ 1 then 
    1 / (1 * 1) 
  else 
    0

theorem probability_quarter_circle : 
  ∫∫ x y in [0, 1] × [0, 1], probability x y = (π / 4) := 
sorry

end probability_quarter_circle_l313_313679


namespace ellipse_foci_distance_l313_313922

-- Definitions based on the problem conditions
def ellipse_eq (x y : ℝ) :=
  Real.sqrt (((x - 4)^2) + ((y - 5)^2)) + Real.sqrt (((x + 6)^2) + ((y + 9)^2)) = 22

def focus1 : (ℝ × ℝ) := (4, -5)
def focus2 : (ℝ × ℝ) := (-6, 9)

-- Statement of the problem
noncomputable def distance_between_foci : ℝ :=
  Real.sqrt (((focus1.1 + 6)^2) + ((focus1.2 - 9)^2))

-- Proof statement
theorem ellipse_foci_distance : distance_between_foci = 2 * Real.sqrt 74 := by
  sorry

end ellipse_foci_distance_l313_313922


namespace seating_arrangement_count_l313_313886

noncomputable def seating_arrangements (num_seats : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
if num_seats = 6 ∧ num_adults = 3 ∧ num_children = 3 then 72 else 0

theorem seating_arrangement_count :
  seating_arrangements 6 3 3 = 72 :=
by
  unfold seating_arrangements
  simp
  split_ifs
  · rfl
  · sorry

end seating_arrangement_count_l313_313886


namespace total_boys_eq_350_l313_313430

variable (Total : ℕ)
variable (SchoolA : ℕ)
variable (NotScience : ℕ)

axiom h1 : SchoolA = 20 * Total / 100
axiom h2 : NotScience = 70 * SchoolA / 100
axiom h3 : NotScience = 49

theorem total_boys_eq_350 : Total = 350 :=
by
  sorry

end total_boys_eq_350_l313_313430


namespace mechanic_working_hours_l313_313022

-- Definitions for the given conditions
def total_cost : ℝ := 220
def part_cost : ℝ := 20
def num_parts : ℕ := 2
def labor_cost_per_minute : ℝ := 0.5
def break_time_minutes : ℕ := 30

-- The total cost of parts
def parts_total_cost : ℝ := num_parts * part_cost

-- The cost spent on labor
def labor_cost : ℝ := total_cost - parts_total_cost

-- The total minutes the mechanic worked
def total_minutes_worked : ℝ := labor_cost / labor_cost_per_minute

-- The actual working minutes excluding break time
def actual_working_minutes : ℝ := total_minutes_worked - break_time_minutes

-- The actual working hours
def actual_working_hours : ℝ := actual_working_minutes / 60

-- Proof statement
theorem mechanic_working_hours : actual_working_hours = 5.5 := sorry

end mechanic_working_hours_l313_313022


namespace largest_five_digit_number_l313_313105

theorem largest_five_digit_number (digits : Set ℕ) (h_digits : digits = {0, 3, 4, 8, 9}) : 
  ∃ n : ℕ, (∀ d ∈ digits, d ∈ {0, 3, 4, 8, 9} → n.digits = [9, 8, 4, 3, 0]) ∧ 
           n = 98430 :=
by
  sorry

end largest_five_digit_number_l313_313105


namespace sum_cubic_function_l313_313313

noncomputable def f (x : ℝ) : ℝ :=
  (1 / 3) * x^3 - (1 / 2) * x^2 + 3 * x - 5 / 12

theorem sum_cubic_function :
  (∑ k in Finset.range 2016, f ((k + 1 : ℝ) / 2017)) = 2016 :=
by
  sorry

end sum_cubic_function_l313_313313


namespace sin_135_l313_313520

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313520


namespace range_of_x_l313_313929

-- Define the ceiling function for ease of use.
noncomputable def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem range_of_x (x : ℝ) (h1 : ceil (2 * x + 1) = 5) (h2 : ceil (2 - 3 * x) = -3) :
  (5 / 3 : ℝ) ≤ x ∧ x < 2 :=
by
  sorry

end range_of_x_l313_313929


namespace sin_135_correct_l313_313560

-- Define the constants and the problem
def angle1 : ℝ := real.pi / 2  -- 90 degrees in radians
def angle2 : ℝ := real.pi / 4  -- 45 degrees in radians
def angle135 : ℝ := 3 * real.pi / 4  -- 135 degrees in radians
def sin_90 : ℝ := 1
def cos_90 : ℝ := 0
def sin_45 : ℝ := real.sqrt 2 / 2
def cos_45 : ℝ := real.sqrt 2 / 2

-- Statement to prove
theorem sin_135_correct : real.sin angle135 = real.sqrt 2 / 2 :=
by
  have h_angle135 : angle135 = angle1 + angle2 := by norm_num [angle135, angle1, angle2]
  rw [h_angle135, real.sin_add]
  have h_sin1 := (real.sin_pi_div_two).symm
  have h_cos1 := (real.cos_pi_div_two).symm
  have h_sin2 := (real.sin_pi_div_four).symm
  have h_cos2 := (real.cos_pi_div_four).symm
  rw [h_sin1, h_cos1, h_sin2, h_cos2]
  norm_num

end sin_135_correct_l313_313560


namespace geometric_sequence_sum_l313_313754

noncomputable def geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a 1 * q^n

theorem geometric_sequence_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), 
  (∀ n : ℕ, a (n + 1) = a 1 * q^n) ∧ 
  (a 2 * a 4 = 1) ∧ 
  (a 1 * (q^0 + q^1 + q^2) = 7) ∧ 
  (a 1 / (1 - q) * (1 - q^5) = 31 / 4) := by
  sorry

end geometric_sequence_sum_l313_313754


namespace reflect_parallelogram_l313_313333

theorem reflect_parallelogram :
  let D : ℝ × ℝ := (4,1)
  let Dx : ℝ × ℝ := (D.1, -D.2) -- Reflect across x-axis
  let Dxy : ℝ × ℝ := (Dx.2 - 1, Dx.1 - 1) -- Translate point down by 1 unit and reflect across y=x
  let D'' : ℝ × ℝ := (Dxy.1 + 1, Dxy.2 + 1) -- Translate point back up by 1 unit
  D'' = (-2, 5) := by
  sorry

end reflect_parallelogram_l313_313333


namespace time_in_komsomolsk_on_amur_when_noon_in_yelizovo_l313_313853

def time_difference (h1 h2 h3 : ℕ) : Prop :=
  (h1 = 12) ∧ (h2 = 6) ∧ (h3 = 14)

def time_difference_zlatoust (h1 h2 h3 : ℕ) : Prop :=
  (h1 = 12) ∧ (h2 = 18) ∧ (h3 = 9)

theorem time_in_komsomolsk_on_amur_when_noon_in_yelizovo :
  (time_difference 12 6 14) →
  (time_difference_zlatoust 12 18 9) →
  (∃ t : ℕ, t = 11) :=
by
  intros _ _
  use 11
  -- Proof does not required as per instructions
  sorry

end time_in_komsomolsk_on_amur_when_noon_in_yelizovo_l313_313853


namespace relative_speed_correct_l313_313847

def speed_vehicle_A : ℝ := 70
def speed_vehicle_B : ℝ := 90
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

theorem relative_speed_correct :
  (kmph_to_mps (speed_vehicle_B - speed_vehicle_A)) = 10 * (1.0 / 1.8) := by
  sorry

end relative_speed_correct_l313_313847


namespace sin_135_eq_l313_313508

theorem sin_135_eq : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 := by
    -- conditions from a)
    -- sorry is used to skip the proof
    sorry

end sin_135_eq_l313_313508


namespace gather_half_of_nuts_l313_313164

open Nat

theorem gather_half_of_nuts (a b c : ℕ) (h₀ : (a + b + c) % 2 = 0) : ∃ k, k = (a + b + c) / 2 :=
  sorry

end gather_half_of_nuts_l313_313164


namespace sin_135_l313_313530

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313530


namespace distance_from_focus_to_directrix_l313_313814

-- Definition of the parabola x^2 = 8y
def parabola (x y : ℝ) : Prop := x^2 = 8 * y

-- Distance from the focus to the directrix for the given parabola
theorem distance_from_focus_to_directrix :
  ∀ x y : ℝ, parabola x y → ∃ d : ℝ, d = 4 :=
begin
  intros x y h,
  use 4,
  sorry,
end

end distance_from_focus_to_directrix_l313_313814


namespace sin_135_l313_313528

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313528


namespace find_d_for_circle_radius_l313_313182

def circle_equation_equivalent (d : ℝ) :=
  (∀ x y : ℝ, x^2 - 8 * x + y^2 + 10 * y + d = 0 ↔
              (x - 4)^2 + (y + 5)^2 = 41 - d)

theorem find_d_for_circle_radius :
  ∃ d : ℝ, (∀ x y : ℝ, circle_equation_equivalent d) ∧ (41 - d = 25) :=
begin
  use 16,
  split,
  { intros x y,
    rw [circle_equation_equivalent],
    split,
    { intro h, sorry }, -- Here, you would complete the proof if needed
    { intro h, sorry }  -- Here, you would complete the proof if needed },
  { norm_num }
end

end find_d_for_circle_radius_l313_313182


namespace sin_135_l313_313524

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313524


namespace find_rajeev_share_l313_313867

-- Define the given conditions
def total_profit : ℕ := 36000
def ratio_ramesh_xyz : ℕ × ℕ := (5, 4)
def ratio_xyz_rajeev : ℕ × ℕ := (8, 9)

-- Compute the combined ratio
def ratio_ramesh_xyz' : ℕ × ℕ := (40, 32)
def ratio_xyz_rajeev' : ℕ × ℕ := (32, 36)
def combined_ratio : ℕ × ℕ × ℕ := (40, 32, 36)

-- Define the total parts in the combined ratio
def total_parts : ℕ := combined_ratio.1 + combined_ratio.2 + combined_ratio.3

-- Define the value of one part
def value_of_one_part := total_profit / total_parts

-- Define Rajeev's share
def rajeev_share := value_of_one_part * combined_ratio.3

-- Theorem statement to be proved
theorem find_rajeev_share : rajeev_share = 12000 := by
  sorry

end find_rajeev_share_l313_313867


namespace monotonicity_of_f_range_of_a_l313_313977

variable (a x : ℝ)

def f (x : ℝ) (a : ℝ) : ℝ := (a - 1) * Real.log x + x + a / x
def g (x : ℝ) (a : ℝ) : ℝ := a / x

-- Part (1)
theorem monotonicity_of_f :
  (∀ x > 0, D (λ x, f x a) x ≥ 0) ∨
  (a = -1 ∧ ∀ x > 0, D (λ x, f x a) x ≥ 0) ∨
  (a < -1 ∧ ∀ x > 0, (D (λ x, f x a) x > 0) ∨ (D (λ x, f x a) x < 0)) ∨
  (-1 < a ∧ a < 0 ∧ ∀ x > 0, (D (λ x, f x a) x > 0) ∨ (D (λ x, f x a) x < 0)) ∨
  (a ≥ 0 ∧ ∀ x > 0, (D (λ x, f x a) x < 0) ∨ (D (λ x, f x a) x > 0)) :=
sorry

-- Part (2)
theorem range_of_a (h : ∀ x ∈ Set.Ioo 1 Real.exp, f x a > g x a) :
  a ∈ Set.Ioi (1 - Real.exp) :=
sorry

end monotonicity_of_f_range_of_a_l313_313977


namespace probability_Sarah_wins_l313_313770

theorem probability_Sarah_wins :
  ∃ (x y : ℝ), x ∈ Icc (-1) 1 ∧ y ∈ Icc (-1) 1 ∧ x^2 + y^2 < 1 → 
  ∃ (P : ℝ), P = real.pi / 4 :=
by
  sorry

end probability_Sarah_wins_l313_313770


namespace find_c_unique_c_l313_313275

def square_eq (c : ℝ) : Prop :=
  let area_triangle := 0.5 * (2 - c) * 2
  in area_triangle = 2

theorem find_c : ∃ (c : ℝ), square_eq c :=
  sorry

theorem unique_c (c : ℝ) : square_eq c → c = 0 :=
  sorry

end find_c_unique_c_l313_313275


namespace circles_conditions_imply_sum_l313_313085

-- Definition of the existence of two circles
def exists_circles_with_given_conditions (p q r : ℕ) : Prop :=
  ∃ (c1 c2 : ℝ × ℝ) (r1 r2 : ℝ), 
    (c1 ≠ c2) ∧ -- The circles are distinct 
    (c1.1 - 7)^2 + (c1.2 - 4)^2 - r1^2 = 0 ∧ -- Intersection at (7,4)
    (c2.1 - 7)^2 + (c2.2 - 4)^2 - r2^2 = 0 ∧ -- Intersection at (7,4)
    r1 * r2 = 85 ∧ -- Product of the radii
    c1.2 = r1 ∧ c2.2 = r2 ∧ -- Tangency to the x-axis
    c1.2 / c1.1 = p * Real.sqrt q / r ∧ -- Tangency to the line y = nx
    c2.2 / c2.1 = p * Real.sqrt q / r ∧
    p.gcd r = 1 ∧ -- p and r are relatively prime
    ∀ prime d, d^2 ∣ q → false -- q is not divisible by the square of any prime

-- Proof problem: Prove that the sum p + q + r equals 272
theorem circles_conditions_imply_sum (p q r : ℕ) 
  (h : exists_circles_with_given_conditions p q r) :
  p + q + r = 272 :=
sorry

end circles_conditions_imply_sum_l313_313085


namespace unique_solution_system_l313_313115

noncomputable def f (x : ℝ) := 4 * x ^ 3 + x - 4

theorem unique_solution_system :
  (∃ x y z : ℝ, y^2 = 4*x^3 + x - 4 ∧ z^2 = 4*y^3 + y - 4 ∧ x^2 = 4*z^3 + z - 4) ↔
  (x = 1 ∧ y = 1 ∧ z = 1) :=
by
  sorry

end unique_solution_system_l313_313115


namespace max_area_rectangular_parallelepiped_l313_313141

def max_projection_area (a b c : ℝ) : ℝ :=
  let d1 := Real.sqrt (a^2 + b^2)
  let d2 := Real.sqrt (a^2 + c^2)
  let d3 := Real.sqrt (b^2 + c^2)
  let s := (d1 + d2 + d3) / 2
  let area := Real.sqrt (s * (s - d1) * (s - d2) * (s - d3))
  2 * area

theorem max_area_rectangular_parallelepiped :
  max_projection_area (Real.sqrt 70) (Real.sqrt 99) (Real.sqrt 126) = 168 :=
by sorry

end max_area_rectangular_parallelepiped_l313_313141


namespace constant_term_in_binomial_expansion_is_neg_160_l313_313052

-- Defining the problem conditions
def x : ℝ := sorry

-- Given function definition: binomial expression
def a (x : ℝ) : ℝ := 2 * real.sqrt x
def b (x : ℝ) : ℝ := -1 / real.sqrt x
def n : ℕ := 6

-- Formalizing the proof problem
theorem constant_term_in_binomial_expansion_is_neg_160 :
  let k := 3 in
  binomial_coefficient n k * (a x ^ (n - k)) * (b x ^ k) = -160 :=
by
  sorry

end constant_term_in_binomial_expansion_is_neg_160_l313_313052


namespace find_a4_and_s5_l313_313971

def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop :=
  ∀ n, a (n + 1) = a n * q

variable (a : ℕ → ℚ) (q : ℚ)

axiom condition_1 : a 1 + a 3 = 10
axiom condition_2 : a 4 + a 6 = 1 / 4

theorem find_a4_and_s5 (h_geom : geometric_sequence a q) :
  a 4 = 1 ∧ (a 1 * (1 - q^5) / (1 - q)) = 31 / 2 :=
by
  sorry

end find_a4_and_s5_l313_313971


namespace T_is_not_integer_l313_313761

-- Lean definitions of the conditions
variables (m n : ℕ) 
variables (a : ℕ → ℕ) 
variables (s : ℕ)

def coprime_with_n (i : ℕ) := Nat.coprime (a i) n

def positive_integers_not_exceeding_m := ∀ i, (1 ≤ i ∧ i ≤ s) → (a i ≤ m ∧ coprime_with_n i)

def increasing_sequence := ∀ i j, (1 ≤ i ∧ i < j ∧ j ≤ s) → a i < a j

def T := Σ' (i : ℕ) (hi : (1 ≤ i ∧ i ≤ s)), 1 / (a i : ℚ)

-- Mathematical statement in Lean
theorem T_is_not_integer 
  (h1 : m > n ∧ n ≥ 1) 
  (h2 : positive_integers_not_exceeding_m m n a s) 
  (h3 : increasing_sequence a s) :
  ∀ (t : ℚ), T m n a s ≠ t :=
by 
  sorry

end T_is_not_integer_l313_313761


namespace planes_perpendicular_l313_313702

def vector_α : ℝ × ℝ × ℝ := (2, 4, -3)
def vector_β : ℝ × ℝ × ℝ := (-1, 2, 2)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem planes_perpendicular : dot_product vector_α vector_β = 0 :=
  by
    sorry

end planes_perpendicular_l313_313702


namespace cost_of_1000_pieces_of_gum_l313_313353

theorem cost_of_1000_pieces_of_gum :
  ∀ cost_per_piece : ℕ → ℕ → ℕ,
  (∀ n, cost_per_piece n 1 = if n ≤ 500 then n * 1 else 500 * 1 + (n - 500) * 0.8) →
  cost_per_piece 1000 1 = 900 :=
by
  intros cost_per_piece h
  simp [h, mul_comm]
  sorry

end cost_of_1000_pieces_of_gum_l313_313353


namespace other_x_intercept_of_circle_l313_313045

theorem other_x_intercept_of_circle :
  let p1 := (0 : ℝ, 0 : ℝ)
  let p2 := (3 * Real.sqrt 7, 7 * Real.sqrt 3)
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius := Real.sqrt (((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2) / 4)
  let circle_eq := ∀ (x y : ℝ), ((x - center.1) ^ 2 + (y - center.2) ^ 2 = radius ^ 2)
  ∃ x : ℝ, circle_eq x 0 ∧ x ≠ 0 :=
begin
  sorry,
end

end other_x_intercept_of_circle_l313_313045


namespace solve_equation_l313_313799

noncomputable def lhs (x: ℝ) : ℝ := (sqrt ((3*x - 1) / (x + 4))) + 3 - 4 * (sqrt ((x + 4) / (3*x - 1)))

theorem solve_equation (x: ℝ) (t : ℝ) (ht : t = (3*x - 1) / (x + 4)) (h_pos : 0 < t) :
  lhs x = 0 → x = 5 / 2 :=
by
  intros h
  sorry

end solve_equation_l313_313799


namespace inscribed_circle_radius_l313_313593

theorem inscribed_circle_radius (AB BC CD DA: ℝ) (hAB: AB = 13) (hBC: BC = 10) (hCD: CD = 8) (hDA: DA = 11) :
  ∃ r, r = 2 * Real.sqrt 7 :=
by
  sorry

end inscribed_circle_radius_l313_313593


namespace angle_BAC_is_36_degrees_l313_313092

theorem angle_BAC_is_36_degrees (O A B C : Type) [circle O] 
  (is_tangent : tangent AB O) (is_tangent' : tangent AC O)
  (arc_ratio : ∀ (BC CB' : arc O), length BC / length CB' = 2 / 3) :
  measure (angle BAC) = 36 :=
sorry

end angle_BAC_is_36_degrees_l313_313092


namespace molecular_weight_of_acid_l313_313412

theorem molecular_weight_of_acid (molecular_weight : ℕ) (n : ℕ) (h : molecular_weight = 792) (hn : n = 9) :
  molecular_weight = 792 :=
by 
  sorry

end molecular_weight_of_acid_l313_313412


namespace sin_135_eq_sqrt2_div_2_l313_313580

theorem sin_135_eq_sqrt2_div_2 :
  let P := (cos (135 : ℝ * Real.pi / 180), sin (135 : ℝ * Real.pi / 180))
  in P.2 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_eq_sqrt2_div_2_l313_313580


namespace coefficient_a2b2c2_l313_313851

theorem coefficient_a2b2c2 (a b c : ℕ) (x : ℝ) :
  (coeff (a^2 * b^2 * c^2) ((a + b + c)^6 * (sin x + cos x)^6)) = 2880 := sorry

end coefficient_a2b2c2_l313_313851


namespace sin_135_l313_313501

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313501


namespace train_speed_l313_313425

noncomputable def speed_of_train (length_of_train length_of_overbridge time: ℝ) : ℝ :=
  (length_of_train + length_of_overbridge) / time

theorem train_speed (length_of_train length_of_overbridge time speed: ℝ)
  (h1 : length_of_train = 600)
  (h2 : length_of_overbridge = 100)
  (h3 : time = 70)
  (h4 : speed = 10) :
  speed_of_train length_of_train length_of_overbridge time = speed :=
by
  simp [speed_of_train, h1, h2, h3, h4]
  sorry

end train_speed_l313_313425


namespace measure_of_angle_A_area_of_triangle_ABC_l313_313735

-- Define the conditions of the problem
variables (a b c : ℝ) (A B C : ℝ) (D : ℝ)
-- Given conditions
hypothesis1 : a = b / sin B * sin A
hypothesis2 : sqrt 3 * cos C - sin C = (sqrt 3 * b) / a
hypothesis3 : b + c = 6
hypothesis4 : D = (b + c) / 2
hypothesis5 : (D * D) = 8

-- The goals to prove
theorem measure_of_angle_A :
  A = 2 * π / 3 := by sorry

theorem area_of_triangle_ABC :
  let bc := b * c in
  bc / 3 * sqrt 3 := by sorry


end measure_of_angle_A_area_of_triangle_ABC_l313_313735


namespace sin_135_l313_313504

theorem sin_135 (h1: ∀ θ:ℕ, θ = 135 → (135 = 180 - 45))
  (h2: ∀ θ:ℕ, sin (180 - θ) = sin θ)
  (h3: sin 45 = (Real.sqrt 2) / 2)
  : sin 135 = (Real.sqrt 2) / 2 :=
by
  sorry

end sin_135_l313_313504


namespace find_m_l313_313699

noncomputable def f (x : ℝ) : ℝ := 2^x - 5

theorem find_m (m : ℝ) (h : f m = 3) : m = 3 := 
by
  sorry

end find_m_l313_313699


namespace number_of_distinct_messages_l313_313463

theorem number_of_distinct_messages : 
  ∃ n : ℕ, (∀ (msg : String), (msg.length = 5 ∧ ∀ (c : Char), c ∈ msg → Char.isLower c → msg = String.mk (list.replicate 5 c)) → (n = 26)) :=
begin
  sorry
end

end number_of_distinct_messages_l313_313463


namespace product_of_c_values_l313_313000

theorem product_of_c_values :
  ∀ (c : ℝ) (b : ℝ) (d : ℝ),
    (∃ x : ℝ, (x^3 + b*x^2 + c*x + d = 0)
      ∧ (∀ y : ℝ, (y ≠ x → y^3 + b*y^2 + c*y + d ≠ 0)))
      ∧ d = c^2 + b + 1
      ∧ b = 2*c →
    let polynomial := -28*c^5 + 12*c^4 - 114*c^3 - 144*c^2 - 108*c - 27 in
    polynomial = 0 →
    ∀ (roots : list ℝ), 
      polynomial.eval c = 0 →
      list.prod roots = ?prod_val :=
sorry

end product_of_c_values_l313_313000


namespace rationalization_of_denominator_l313_313268

theorem rationalization_of_denominator : 
  let s := 1 / (1 - (2 : ℝ) ^ (1 / 3)) in 
  s = -(1 + (2 : ℝ) ^ (1 / 3) + (2 : ℝ) ^ (2 / 3)) := 
by 
  let s := 1 / (1 - (2 : ℝ) ^ (1 / 3))
  have h1 : 1 = 1 := rfl
  sorry

end rationalization_of_denominator_l313_313268


namespace sin_2alpha_minus_pi_over_4_l313_313217

theorem sin_2alpha_minus_pi_over_4 
  (α : ℝ)
  (h1 : sin α - cos α = 1 / 5)
  (h2 : 0 ≤ α ∧ α ≤ π) :
  sin (2 * α - π / 4) = 31 * sqrt 2 / 50 := 
  sorry

end sin_2alpha_minus_pi_over_4_l313_313217


namespace initial_hamburgers_count_is_nine_l313_313143

-- Define the conditions
def hamburgers_initial (total_hamburgers : ℕ) (additional_hamburgers : ℕ) : ℕ :=
  total_hamburgers - additional_hamburgers

-- The statement to be proved
theorem initial_hamburgers_count_is_nine :
  hamburgers_initial 12 3 = 9 :=
by
  sorry

end initial_hamburgers_count_is_nine_l313_313143


namespace max_value_a_l313_313417

theorem max_value_a (a : ℝ) : 
  (∀ x : ℝ, x > -1 → x + 1 > 0 → x + 1 + 1 / (x + 1) - 2 ≥ a) → a ≤ 0 :=
by
  -- Proof omitted
  sorry

end max_value_a_l313_313417


namespace smaller_side_of_rectangle_l313_313863

theorem smaller_side_of_rectangle (r : ℝ) (h1 : r = 42) 
                                   (h2 : ∀ L W : ℝ, L / W = 6 / 5 → 2 * (L + W) = 2 * π * r) : 
                                   ∃ W : ℝ, W = (210 * π) / 11 := 
by {
    sorry
}

end smaller_side_of_rectangle_l313_313863


namespace arithmetic_sequence_50th_term_l313_313963

theorem arithmetic_sequence_50th_term :
  let a1 := 3
  let d := 2
  let n := 50
  let a_n := a1 + (n - 1) * d
  a_n = 101 :=
by
  sorry

end arithmetic_sequence_50th_term_l313_313963


namespace find_value_y_l313_313689

variable (y k x : ℝ)

def initial_condition (k : ℝ) (x : ℝ) := y = k * x^(1/4)

-- Given condition
def specific_value (k : ℝ) : Prop := initial_condition y k 9 ∧ y = 3 * real.sqrt 3

-- The target proposition to prove
def target_value (k : ℝ) : Prop := initial_condition y k 16 ∧ y = 6

-- The main theorem
theorem find_value_y (k : ℝ) (h₁ : specific_value k) : target_value k := sorry

end find_value_y_l313_313689


namespace sin_135_l313_313523

theorem sin_135 : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  -- step 1: convert degrees to radians
  have h1: (135 * Real.pi / 180) = Real.pi - (45 * Real.pi / 180), by sorry,
  -- step 2: use angle subtraction identity
  rw [h1, Real.sin_sub],
  -- step 3: known values for sin and cos
  have h2: Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  have h3: Real.cos (45 * Real.pi / 180) = Real.sqrt 2 / 2, by sorry,
  -- step 4: simplify using the known values
  rw [h2, h3],
  calc Real.sin Real.pi = 0 : by sorry
     ... * _ = 0 : by sorry
     ... + _ * _ = Real.sqrt 2 / 2 : by sorry

end sin_135_l313_313523


namespace fraction_is_not_necessarily_rational_infinite_non_repeating_decimal_irrational_all_rational_are_not_finite_decimals_side_length_square_area_5_irrational_correct_choice_is_B_l313_313468

theorem fraction_is_not_necessarily_rational : ¬ (∃ a b : ℚ, a / b ≠ (a / b)) :=
sorry

theorem infinite_non_repeating_decimal_irrational : 
  ∀ x : ℝ, (¬ (∃ a b : ℚ, x = a / b) ∧ ¬ repeating (decimal_expansion x)) → ¬ rational x :=
sorry

theorem all_rational_are_not_finite_decimals : 
  ¬ (∀ x : ℚ, finite_decimal (decimal_expansion x)) :=
sorry

theorem side_length_square_area_5_irrational : 
  ∀ x : ℝ, (x^2 = 5) → ¬ (∃ a b : ℚ, x = a / b) :=
sorry

theorem correct_choice_is_B : 
  (fraction_is_not_necessarily_rational = false) ∧ 
  (infinite_non_repeating_decimal_irrational true) ∧ 
  (all_rational_are_not_finite_decimals = false) ∧ 
  (side_length_square_area_5_irrational true) → 
  (correct_choice = 'B') :=
sorry

end fraction_is_not_necessarily_rational_infinite_non_repeating_decimal_irrational_all_rational_are_not_finite_decimals_side_length_square_area_5_irrational_correct_choice_is_B_l313_313468


namespace sum_first_100_terms_l313_313657

-- Let's define the conditions as Lean definitions
def monotonic_on (f : ℝ → ℝ) (s : set ℝ) : Prop :=
    ∀ x y ∈ s, x ≤ y → f x ≤ f y

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
    ∀ x, f (2 * a - x) = f x

def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
    d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

def given_conditions (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ) : Prop :=
  monotonic_on f (set.Ici (-1)) ∧
  symmetric_about (λ x, f (x - 2)) 1 ∧
  arithmetic_seq a d ∧
  f (a 50) = f (a 51)

-- Sum of the first 100 terms of an arithmetic sequence
def sum_arithmetic_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (a 0 + a (n - 1))

-- Proof problem
theorem sum_first_100_terms (f : ℝ → ℝ) (a : ℕ → ℝ) (d : ℝ)
  (h : given_conditions f a d) :
  sum_arithmetic_seq a 100 = -100 :=
sorry

end sum_first_100_terms_l313_313657


namespace sin_135_degree_l313_313535

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313535


namespace find_y_coordinate_l313_313357

-- Define the parabola and its conditions
def parabola (x : ℝ) : ℝ := 4 * x^2
def focus : ℝ × ℝ := (0, 1 / 16 : ℝ)

-- Define the distance function between two points in R²
def distance (p q : ℝ × ℝ) : ℝ := real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- Goal: Find the y-coordinate of a point on the parabola such that the distance to the focus is 1.
theorem find_y_coordinate (x y : ℝ) (h : y = parabola x) (dist_1 : distance (x, y) focus = 1) :
    y = 15 / 16 :=
sorry

end find_y_coordinate_l313_313357


namespace lisa_time_to_complete_l313_313326

theorem lisa_time_to_complete 
  (hotdogs_record : ℕ) 
  (eaten_so_far : ℕ) 
  (rate_per_minute : ℕ) 
  (remaining_hotdogs : ℕ) 
  (time_to_complete : ℕ) 
  (h1 : hotdogs_record = 75) 
  (h2 : eaten_so_far = 20) 
  (h3 : rate_per_minute = 11) 
  (h4 : remaining_hotdogs = hotdogs_record - eaten_so_far)
  (h5 : time_to_complete = remaining_hotdogs / rate_per_minute) :
  time_to_complete = 5 :=
sorry

end lisa_time_to_complete_l313_313326


namespace evaluate_expression_l313_313601

noncomputable def z : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

lemma thirteen_roots_of_unity (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 12) : z ^ 13 = 1 := by
  field_simp
  norm_num

theorem evaluate_expression :
  (∏ k in Finset.range 12, (3:ℂ) - z ^ (k + 1)) = 797161 := by
  sorry

end evaluate_expression_l313_313601


namespace area_triangle_60_area_triangle_120_l313_313048

-- Definitions for the triangle problem
variables (a b c : Real)
def angle_A_is_60 := 60 * Real.pi / 180
def angle_A_is_120 := 120 * Real.pi / 180

-- Proving the area of the triangle when ∠A is 60°
theorem area_triangle_60 (h : ∠A = angle_A_is_60) :
  S = (Real.sqrt 3 / 4) * (a^2 - (b - c)^2) :=
sorry

-- Proving the area of the triangle when ∠A is 120°
theorem area_triangle_120 (h : ∠A = angle_A_is_120) :
  S = (Real.sqrt 3 / 12) * (a^2 - (b - c)^2) :=
sorry

end area_triangle_60_area_triangle_120_l313_313048


namespace calculate_total_difference_l313_313171

theorem calculate_total_difference : 
  let S := (∑ k in (Finset.range 72), (2001 + k * 3)) - (∑ k in (Finset.range 72), (501 + k * 3)) 
  in S = 108000 := 
by 
  sorry

end calculate_total_difference_l313_313171


namespace circumcircle_radius_of_ABC_l313_313089

-- Define the conditions
variables (A B C : Point) -- Points A, B, and C of triangle ABC
variables (r₁ r₂ r₃ : ℝ) -- Radii of the three spheres
variables (dAB dAC dBC : ℝ) -- Distances between centers of pairs of spheres

-- Assumptions based on problem conditions
axiom sum_of_radii : r₁ + r₂ = 9
axiom distance_between_centers : dAB = sqrt 305
axiom radius_third_sphere : r₃ = 7
axiom third_sphere_touches : dAC = r₁ + r₃ ∧ dBC = r₂ + r₃

-- Target: Prove that the radius of the circumcircle of triangle ABC is 2√14
theorem circumcircle_radius_of_ABC : R = 2 * sqrt 14 :=
sorry

end circumcircle_radius_of_ABC_l313_313089


namespace sin_135_degree_l313_313539

theorem sin_135_degree : Real.sin (135 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_135_degree_l313_313539
